# RAG 评估体系升级设计文档

**日期：** 2026-05-02
**项目：** GitHub Project Search Engine — 评估体系升级
**基于：** [2026-04-29-rag-chunking-embedding-upgrade-design.md](2026-04-29-rag-chunking-embedding-upgrade-design.md)

---

## 1. 升级动机

| 问题 | 说明 |
|------|------|
| 评估粒度过粗 | 现有 testset 以 repo 为单位标注 relevant，而检索系统返回的是 parent chunk，粒度不匹配 |
| Recall 语义不准 | 现有 Recall@5 等同于 Hit Rate（relevant 集合大小为 1），无法代表真正的召回率，不宜写在简历上 |
| Ground truth 质量差 | 多 repo relevant 通过 embedding 相似度自动生成，实际大部分 query 仍只有 1 个 relevant repo |
| 接口过时 | `evaluate.py` 调用旧版 `hybrid_search` / `_load_artifacts`，环境变量仍为 `GLM_*`，与当前架构不匹配 |

---

## 2. 升级目标

1. 将评估粒度从 repo 级别改为 **parent chunk 级别**
2. 用 LLM 逐 chunk 打分生成高质量 ground truth，使 Recall@5 具备真实语义
3. 将 testset 生成、ground truth 标注、评估执行三个职责分离到独立脚本
4. 评估脚本直接复用 `CustomRetriever`，与生产系统行为一致
5. 统一环境变量命名为 `LLM_*`

---

## 3. 整体架构

```
eval/
├── generate_testset.py       # Step 1: 生成 queries，无 ground truth
├── annotate_groundtruth.py   # Step 2: LLM 标注 relevant chunk IDs
├── evaluate.py               # Step 3: 跑指标，输出报告
├── testset_queries.json      # Step 1 输出（中间产物，不 commit）
├── testset.json              # Step 2 输出（commit）
└── report.json               # Step 3 输出（commit）
```

**职责分离的价值：** query 生成需要消耗 LLM 费用，标注逻辑如需调整（换 prompt、换模型）可单独重跑 `annotate_groundtruth.py`，无需重新生成 queries。

---

## 4. generate_testset.py 设计

### 4.1 变更说明

| 项目 | 变更前 | 变更后 |
|------|--------|--------|
| 输出文件 | `testset.json`（含 relevant_repo_ids） | `testset_queries.json`（无 ground truth） |
| `_find_similar` | 用 embedding 相似度找跨仓库 relevant | **删除** |
| embedding 缓存 | `repo_embeddings.npy` / `repo_embeddings_index.json` | **不再需要** |
| 分层采样逻辑 | 保留 | **保留不变** |
| `_generate_query` | 保留 | **保留不变** |

**main() 中需要删除的逻辑：**
- `_find_similar` 的调用及其结果变量（`extras`、`relevant_repo_ids`）
- `_embed_all_repos` 的调用及相关 embedding cache 逻辑
- `testset` 条目中的 `relevant_repo_ids` 字段
- 输出路径改为 `testset_queries.json`（而非 `testset.json`）

### 4.2 输出格式

```json
[
  {
    "query": "lightweight Python HTTP client library",
    "query_type": "semantic",
    "meta": {
      "source_repo": "psf/requests",
      "source_chunk_id": "psf/requests__0",
      "language": "Python",
      "stars_tier": "high",
      "stratum": "lang_Python"
    }
  }
]
```

### 4.3 保留逻辑

- 分层采样策略：Top-5 语言层（各 30 条）+ Other 语言层（30 条）+ Stars 层（各 20 条）+ 随机层（160 条）= 400 条
- `_generate_query`：query 类型在 semantic / keyword 间交替

---

## 5. annotate_groundtruth.py 设计（新脚本）

### 5.1 输入输出

- **输入：** `eval/testset_queries.json`，`backend/parent_chunks.json`
- **输出：** `eval/testset.json`
- **环境变量：** `LLM_API_KEY`，`LLM_API_URL`，`LLM_MODEL_ID`

**`parent_chunks.json` 条目结构（每个 chunk 对象包含以下字段）：**

```json
{
  "parent_id": "psf/requests__0",
  "full_name": "psf/requests",
  "section_index": 0,
  "section_title": "__intro__",
  "content": "Requests is a simple, yet elegant, HTTP library...",
  "description": "A simple, yet elegant HTTP library.",
  "language": "Python",
  "stars": 52000
}
```

过滤同仓库 chunk 时使用 `chunk["full_name"] == source_repo`；`section_title` 和 `content` 均保证存在（无 README 仓库的 section_title 为 `"__description__"`）。

### 5.2 标注流程

```
对每个 query：
  1. 从 parent_chunks.json 取出 source_repo 下所有父 chunk
     过滤方式：chunk["full_name"] == source_repo
     （等价于：parent_id 以 source_repo + "__" 开头）
  2. 逐 chunk 调用 LLM，传入完整 chunk content（不截断）
  3. LLM 返回 0 或 1（是否与 query relevant）
     解析规则：response.strip().startswith("1") → 1，startswith("0") → 0，其他视为调用失败
  4. 收集所有标注为 1 的 chunk IDs → relevant_chunk_ids
  5. 后处理：若 len(relevant_chunk_ids) < 2，丢弃该条目
```

**不设 relevant 数量上限。** 若 LLM 标注了 6 个 relevant chunks，全部保留。Recall@5 的分母即为实际标注数量，上限自然受 Top-5 限制。

### 5.3 LLM 标注 Prompt

```
User query: {query}
Repository: {full_name}
Section title: {section_title}
Content:
{content}

Is this repository section relevant to the user query?
Reply with only 0 (not relevant) or 1 (relevant), no explanation.
```

### 5.4 输出格式

```json
[
  {
    "query": "lightweight Python HTTP client library",
    "query_type": "semantic",
    "relevant_chunk_ids": ["psf/requests__0", "psf/requests__2"],
    "meta": {
      "source_repo": "psf/requests",
      "source_chunk_id": "psf/requests__0",
      "language": "Python",
      "stars_tier": "high",
      "stratum": "lang_Python"
    }
  }
]
```

### 5.5 容错与幂等

- LLM 调用异常或返回非 0/1 内容时：跳过该 chunk（不标注为 1），打印 warning，继续处理下一个 chunk
- 支持断点续跑：启动时读取已有的 `testset.json`，提取其中所有 `meta.source_chunk_id`，跳过 `testset_queries.json` 中 `meta.source_chunk_id` 已在其中的条目
- **被丢弃的条目（relevant < 2）不写入 `testset.json`，因此不会出现在已处理集合中，下次续跑时会重新处理。** 这是预期行为：重新处理的成本仅为该 repo 的 chunk 数量（通常 5~15 次 LLM 调用），可接受。如需避免重复调用，可自行维护一个单独的 `processed_chunk_ids.json` 跳过列表，但默认实现无需此机制。

---

## 6. evaluate.py 设计

### 6.1 变更说明

| 项目 | 变更前 | 变更后 |
|------|--------|--------|
| 检索调用 | `hybrid_search` / `_load_artifacts`（旧接口） | 直接使用 `CustomRetriever` |
| relevant 粒度 | repo ID（`facebook/react`） | parent chunk ID（`facebook/react__1`） |
| `retrieved_ids` | `r["parent_id"]` from hybrid_search 结果 | `n.node.metadata["parent_id"]` from `NodeWithScore` |
| 环境变量 | `GLM_*` | `LLM_*` |
| `_contribution_label` 内部 key | `debug["pinecone_candidate_ids"]` | `debug["vector_candidate_ids"]` |
| `_contribution_label` 返回值 | `"pinecone_only"` | `"vector_only"` |
| `total_contrib` dict key | `"pinecone_only"` | `"vector_only"` |
| 报告字段 | `"pinecone_only_pct"` | `"vector_only_pct"` |

**环境变量：** `evaluate.py` 在运行时需要以下所有环境变量：
- `PINECONE_API_KEY`，`PINECONE_INDEX_NAME`（Pinecone 连接）
- `LLM_API_KEY`，`LLM_API_URL`（`load_retriever` 内部初始化 embedding 模型所需）
- `LLM_MODEL_ID`（LLM judge 调用所需）

### 6.2 检索调用方式

```python
# 脚本启动时初始化（load_retriever 内部需要 LLM_API_KEY / LLM_API_URL）
retriever = load_retriever(pinecone_index)

# 每条 query（评估时不设过滤条件）
retriever.language = ""
retriever.min_stars = 0
retriever.topics = []
nodes = retriever.retrieve(query)
retrieved_ids = [n.node.metadata["parent_id"] for n in nodes]
debug = retriever.last_debug  # {"vector_candidate_ids": [...], "bm25_candidate_ids": [...]}
```

### 6.3 指标定义

所有指标计算逻辑不变，只是 relevant 集合从 repo ID 改为 parent chunk ID：

| 指标 | 定义 |
|------|------|
| Precision@5 | hits / len(retrieved)，hits = retrieved ∩ relevant |
| Recall@5 | hits / len(relevant)，分母为 LLM 标注数量（无上限） |
| MRR | 第一个 relevant 结果的倒数排名 |
| Soft Precision@5 | (ground_truth_hits + judge_hits) / judge_valid |

**Soft Precision@5 各项定义：**
- `ground_truth_hits`：retrieved 中属于 `relevant_chunk_ids` 的数量
- `judge_hits`：retrieved 中不属于 `relevant_chunk_ids`、但 LLM judge 返回 1 的数量
- `judge_valid`：retrieved 中属于 `relevant_chunk_ids`、或 LLM judge 返回有效值（0 或 1）的数量（即排除 judge 调用失败的条目）

### 6.4 报告结构

报告结构与现有 `report.json` 保持一致，新增 `relevant_count_distribution` 字段用于了解 ground truth 数量分布：

```json
{
  "date": "2026-05-02",
  "total_queries": 400,
  "metrics": {
    "precision_at_5": 0.0,
    "recall_at_5": 0.0,
    "mrr": 0.0,
    "soft_precision_at_5": 0.0
  },
  "relevant_count_distribution": {
    "2": 120,
    "3": 180,
    "4": 60,
    "5+": 40
  },
  "contribution": {
    "bm25_only_pct": 0.0,
    "vector_only_pct": 0.0,
    "both_pct": 0.0,
    "unknown_pct": 0.0
  },
  "by_query_type": {
    "semantic": {"precision_at_5": 0.0, "mrr": 0.0},
    "keyword":  {"precision_at_5": 0.0, "mrr": 0.0}
  },
  "by_language": {},
  "by_stars_tier": {},
  "low_score_samples": []
}
```

---

## 7. 已知局限性

| 局限 | 说明 |
|------|------|
| 跨仓库 relevant 未覆盖 | Ground truth 只标注同仓库内的父 chunk，跨仓库相关性不纳入评估 |
| Recall@5 上限 | 当 relevant > 5 时，Recall@5 无法达到 1.0，这反映任务本身的特性而非系统缺陷 |
| Ground truth 固定 | 系统为确定性检索，对同一 testset 每次评估结果相同；评估的真正价值在于对比不同系统配置的 A/B 测试 |
| LLM 标注一致性 | 逐 chunk 打分存在 LLM 判断不一致的风险，可通过固定 model + temperature=0.0 缓解 |

---

## 8. 文件变更总结

```
eval/
├── generate_testset.py       # 修改：移除 _find_similar、embedding 缓存；输出改为 testset_queries.json
├── annotate_groundtruth.py   # 新增：LLM 逐 chunk 标注，输出 testset.json
├── evaluate.py               # 修改：接口对齐 CustomRetriever，粒度改为 parent chunk
├── testset_queries.json      # 新增（不 commit）
├── testset.json              # 格式变更（commit）
└── report.json               # 格式小幅变更（commit）
```

**不涉及 backend/ 任何文件的修改。**
