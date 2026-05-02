# Eval Testset v2 设计文档

**日期：** 2026-05-02
**项目：** GitHub RAG — 评估测试集 v2 及评估脚本适配
**目标：** 适配父子块架构，修复 Recall 退化为命中率的问题，通过 embedding 相似度补充多仓库 relevant 标注

---

## 1. 背景

RAG 后端已升级为父子块两级索引（chunker + indexer 重写，Pinecone 存子块，检索结果 Max Pooling 到父块）。原测试集存在两个问题：

1. **数据源失效**：`generate_testset.py` 读取 `chunk_metadata.json`，该文件已不存在，现由 `parent_chunks.json` 替代
2. **Recall 无意义**：每条 query 只有 1 个 `relevant_repo_id`，Recall@5 退化为二元命中率（0 或 1），与 MRR 信息高度重叠

---

## 2. 架构变化对评估的影响

| 项目 | 旧架构 | 新架构 |
|------|--------|--------|
| 索引单元 | 单层 chunk | 子块（Pinecone）+ 父块（本地） |
| 检索返回 | chunk dict list | NodeWithScore（父块粒度） |
| 结果 ID | `parent_id`（如 `psf/requests__0`） | 父块 ID，如 `psf/requests__0`、`psf/requests__1` |
| 同仓库多父块 | 可能出现 | **可能出现**（Max Pooling 只在子块→父块层面去重，同一仓库的不同 section 可各自进入 Top-5） |

**关键结论：** 评估粒度应为**仓库**，不是父块。由于同一仓库的不同 section 可能同时出现在 Top-5 中，`evaluate.py` 必须在计算指标前先按 `full_name` 对 `retrieved_repos` 去重（保留顺序）：

```python
retrieved_repos = list(dict.fromkeys(n.metadata["full_name"] for n in nodes))
```

---

## 3. 测试集生成：`generate_testset.py` 改写

### 3.1 数据源

从 `backend/parent_chunks.json` 中提取每个仓库的唯一代表记录：取 `parent_id` 以 `__0` 结尾的那条（即 intro section），构建 `{full_name: chunk}` 字典，一个仓库一条。

### 3.2 仓库代表文本

用于 embedding 相似度计算（补充 relevant）和 query 生成 prompt：

```
代表文本 = description（如果非空）
         + "\n" + chunk["content"][:200]（过滤后，见下）
```

**intro 过滤规则**（满足任一则跳过 intro，只用 description）：
- 去除 HTML 标签、`&nbsp;` 后长度 < 30 字符
- 字母字符占比 < 20%（判定为乱码/纯链接/纯符号）

### 3.3 分层抽样

策略与 v1 完全一致，目标 400 条：

| 层 | 配额 |
|----|------|
| Top 5 语言各一层 | 30 条/层 = 150 条 |
| 语言 Other 层 | 30 条 |
| Stars low/mid/high 各一层 | 20 条/层 = 60 条 |
| 随机层 | 160 条 |

去重优先级：Language > Stars > Random。

### 3.4 补充多仓库 relevant（核心新增）

对每条抽样仓库，通过 embedding 相似度从全量 corpus 中补充 2–4 个额外相关仓库，使 `relevant_repo_ids` 包含多个仓库，让 Recall 有真正的统计意义。

**步骤：**

1. **嵌入全量仓库代表文本**：对 corpus 中所有仓库（约 1 万条）的代表文本调用 `text-embedding-3-small` 生成向量，批量处理（batch_size=100），结果缓存到 `eval/repo_embeddings.json`（`{full_name: [float, ...]}`）以避免重复计算
2. **计算 cosine 相似度**：对 400 个 source repo，各自与全量 corpus 做点积（向量已 L2 归一化），取 Top 20 候选
3. **过滤 + 选取**：从 Top 20 中排除 source repo 自身，取相似度 > 0.75 的仓库，最多取 4 个；若不足 2 个则不补充（保留原始 1 个 relevant，避免低质量标注污染测试集）
4. **写入 `relevant_repo_ids`**：`[source_repo, extra_1, extra_2, ...]`，source repo 始终在首位

**成本估算：**
- 全量仓库嵌入（约 1.6 万个）：按 batch 100 约 160 次请求，代表文本平均约 50 tokens，总计约 800k tokens
- 400 个 query 生成：约 400 次 LLM 调用，与 v1 一致

### 3.5 输出格式：`testset.json`

```json
[
  {
    "query": "a lightweight Python HTTP client library",
    "relevant_repo_ids": ["psf/requests", "encode/httpx", "aio-libs/aiohttp"],
    "query_type": "semantic",
    "meta": {
      "language": "Python",
      "stars_tier": "high",
      "stratum": "lang_Python",
      "source_repo": "psf/requests"
    }
  }
]
```

**注意：** `relevant_repo_ids` 存仓库名（`owner/repo`），不存父块 ID。

---

## 4. 评估脚本：`evaluate.py` 适配

### 4.1 命中判断逻辑

旧逻辑（精确 parent_id 匹配）：
```python
retrieved_ids = [r["parent_id"] for r in results]
hits = r in relevant_set
```

新逻辑（仓库名匹配，含去重）：
```python
# nodes 是 NodeWithScore list，保留原始 nodes 供 LLM judge 使用
# 先按 full_name 去重（同一仓库多个 section 可能同时出现），保留顺序
retrieved_repos = list(dict.fromkeys(n.metadata["full_name"] for n in nodes))
hits = repo in relevant_set   # relevant_set = {"psf/requests", "encode/httpx", ...}

# LLM judge 需要完整 metadata，通过 full_name 从 nodes 中查找
node_by_repo = {n.metadata["full_name"]: n for n in reversed(nodes)}  # 保留最高 score 的那条
```

### 4.2 retriever 调用适配

旧调用（已不存在）：
```python
from retriever import hybrid_search, _load_artifacts
results, debug = hybrid_search(query, pinecone_index, return_debug=True)
```

新调用：
```python
from retriever import load_retriever
retriever = load_retriever(pinecone_index)
nodes = retriever.retrieve(query)
retrieved_repos = [n.metadata["full_name"] for n in nodes]
```

### 4.3 贡献分析适配

旧的 `debug_info` 通过 `hybrid_search(return_debug=True)` 返回，新架构中需在 `CustomRetriever` 上添加等价能力。

**`retriever.py` 改动：** 在 `CustomRetriever` 上增加 `last_debug` 属性，`_retrieve()` 执行后写入：

```python
def _retrieve(self, query_bundle):
    ...
    self.last_debug = {
        "vector_candidate_repos": list(vector_ranked.keys()),   # parent_id 列表
        "bm25_candidate_repos":   list(bm25_ranked.keys()),
    }
    return results
```

**`evaluate.py` 贡献标签逻辑：**

```python
nodes = retriever.retrieve(query)
debug = retriever.last_debug
vector_set = {pid.rsplit("__", 1)[0] for pid in debug["vector_candidate_repos"]}
bm25_set   = {pid.rsplit("__", 1)[0] for pid in debug["bm25_candidate_repos"]}

for repo in retrieved_repos:
    in_v = repo in vector_set
    in_b = repo in bm25_set
    label = "both" if in_v and in_b else ("vector_only" if in_v else "bm25_only")
```

注意：候选集使用全量（`_vector_search` 返回最多 20 个 parent_id，`_bm25_search` 同），而非仅 top-5，确保贡献分析不遗漏通过 RRF 融合升上来的结果。

### 4.4 Recall 指标现在的语义

- 分母：`len(relevant_repo_ids)`（现在 > 1，通常 2–5）
- 分子：Top-5 结果中属于 `relevant_repo_ids` 的仓库数
- 真正度量"相关仓库中有多少比例被召回"

---

## 5. 文件变动清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `backend/retriever.py` | 小改 | `CustomRetriever` 增加 `last_debug` 属性，暴露两路候选 parent_id |
| `eval/generate_testset.py` | 改写 | 适配 parent_chunks.json，加 embedding 相似度补充 relevant |
| `eval/evaluate.py` | 改写 | 命中逻辑改为仓库名匹配，retriever 调用改为 CustomRetriever，贡献分析读 last_debug |
| `eval/testset.json` | 重新生成 | 运行新 generate_testset.py 后提交 |
| `eval/repo_embeddings.npy` + `eval/repo_embeddings_index.json` | 新增（不提交） | 全量仓库向量缓存（npy）+ full_name 顺序索引（json），加入 .gitignore |

---

## 6. 注意事项

- `repo_embeddings.json` 存储约 1.6 万个仓库的 1536 维向量，JSON 格式约 490MB，建议改用 `.npy` 二进制格式（约 98MB）；无论哪种格式均加入 `.gitignore`，不提交到 repo
- 相似度阈值 0.75 可在运行后根据实际分布调整；若 400 条中大多数仍只有 1 个 relevant，可适当降低至 0.70
- `evaluate.py` 中贡献分析（BM25 only / Pinecone only / both）本次移除；若后续需要，在 `CustomRetriever` 中添加 `return_debug` 参数
- 旧 `testset.json` 在重新生成前建议备份（`testset_v1.json`）
