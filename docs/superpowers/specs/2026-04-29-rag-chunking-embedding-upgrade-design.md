# RAG Chunking & Embedding 升级设计文档

**日期：** 2026-04-29
**项目：** GitHub Project Search Engine — Chunking、Embedding、检索层升级
**基于：** [2026-04-15-github-rag-upgrade-design.md](2026-04-15-github-rag-upgrade-design.md)

---

## 1. 升级动机

| 问题 | 说明 |
|------|------|
| chunk 体积过大 | 原方案仅按 Markdown heading 切块，README 长段落导致单 chunk 过长，语义精度不足 |
| 嵌入模型偏弱 | `all-MiniLM-L6-v2`（384 维）语义质量有限，不支持多语言跨语言检索 |
| 元数据冗余低效 | 父块元数据冗余存入每个子块 Pinecone metadata，结构混乱 |
| 无工业框架 | 全手写 pipeline，缺乏可向面试官展示的框架使用经验 |

---

## 2. 升级目标

1. 两层切块：heading 切块 + 递归二次切块，控制子 chunk 在合理 token 范围内
2. 换用 `text-embedding-3-small`（1536 维，多语言，支持中文提问检索英文内容）
3. 清晰的存储划分：子 chunk 向量化存 Pinecone，父 chunk 完整文本存本地
4. 引入 LlamaIndex 框架（方案 B：indexing + retrieval + query engine 全面接入，BM25+RRF 自定义实现）
5. 统一环境变量命名：`GLM_*` → `LLM_*`

---

## 3. 整体架构变更

### 技术选型对比

| 层级 | 升级前 | 升级后 |
|------|--------|--------|
| 切块策略 | heading 切块（单层） | heading 切块 + 递归二次切块（两层） |
| 嵌入模型 | `all-MiniLM-L6-v2`（384 维） | `text-embedding-3-small`（1536 维） |
| 向量数据库维度 | 384 | 1536（需删除重建 index） |
| 父块存储 | Pinecone metadata 冗余存储 | 本地 `parent_chunks.json` |
| 子块存储 | Pinecone（含全部元数据冗余） | Pinecone（向量 + 过滤字段 + parent_id） |
| BM25 检索对象 | 子 chunk | 父 chunk |
| 向量检索归约 | 最终输出时去重 | 归约阶段 Max Pooling（提前去重） |
| RAG 框架 | 无 | LlamaIndex（方案 B） |

---

## 4. 数据层设计

### 4.1 两层切块策略

**第一层：Heading 切块（父 chunk）**

- 按 `#`～`######` heading 行分割，每个 heading + 其下方正文 = 一个父 chunk
- 最小长度阈值：**20 token**（用 `tiktoken` `cl100k_base` 计算），低于阈值的块合并到前一块
- README 首段（第一个 heading 之前的内容）单独作为第 0 个父 chunk
- 父 chunk 不向量化，只存本地

**第二层：递归二次切块（子 chunk）**

- 使用 LlamaIndex `SentenceSplitter` 对每个父 chunk 内容进行递归分割
- 分隔符优先级：`\n\n` → `\n` → ` `
- 目标大小：**512 token**，overlap：**50 token**
- 子 chunk 向量化后存入 Pinecone，携带 `parent_id` 关联父 chunk

**无 README 的仓库：**

- 用仓库 description 字段创建一个父 chunk（同时也是唯一子 chunk）

### 4.2 存储划分

| 存储 | 内容 | 用途 |
|------|------|------|
| **Pinecone**（子 chunk） | 1536 维向量 + `parent_id`、`section_title`、`chunk_index` + 所有元数据过滤字段 | 向量检索 + pre-filter |
| **`parent_chunks.json`**（本地） | `parent_id` → `{content, full_name, description, clone_url, language, stars, forks, watchers, issues, topics, create_time, update_time, push_time}` | BM25 检索 + LLM prompt 构建 |
| **`bm25_index.pkl`**（本地） | 基于父 chunk `content` 构建的 BM25Okapi 索引 | BM25 关键词检索 |
| **`filter_options.json`**（本地） | languages / topics / stars_range | `/api/filters/options` 端点 |

**Pinecone metadata 字段（子 chunk 级别）：**

```
parent_id, section_title, chunk_index,
language, stars, forks, watchers, issues, topics,
create_time, update_time, push_time
```

注意：`description`、`clone_url` 等展示字段不存 Pinecone（不用于过滤），由 `parent_chunks.json` 提供。子 chunk 的文本内容（`content`）也不存 Pinecone metadata（避免超 40KB 限制），由 `parent_chunks.json` 通过 `parent_id` 提供。

### 4.3 Pinecone Index 重建

embedding 维度从 384 → 1536，必须删除旧 index 并重建：

```
1. Pinecone 控制台删除旧 index
2. indexer.py 自动创建新 index（dimension=1536, metric=cosine）
3. 重跑 indexer.py
```

---

## 5. 检索层设计

### 5.1 检索流程

```
query
  ├── 向量检索路（LlamaIndex VectorIndexRetriever）
  │     ↓ Pinecone pre-filter（language / stars / topics / forks 等）
  │     ↓ Top-60 子 chunk（带 parent_id）
  │     ↓ 遍历结果，按 parent_id 去重，凑满 20 个不重复父 chunk
  │     ↓ 每个父 chunk 取其最高排名子 chunk 的 rank（Max Pooling）
  │     → 父 chunk 排名列表 A（最多 20 个）
  │
  ├── BM25 检索路（rank_bm25，直接检索父 chunk）
  │     ↓ post-filter（language / stars / topics 等）
  │     → 父 chunk 排名列表 B（最多 20 个）
  │
  └── RRF 融合（k=60）
        ↓ 两路父 chunk 排名合并
        → Top-5 父 chunk
        → 从 parent_chunks.json 读取完整文本 + 元数据
        → 返回 NodeWithScore 列表（LlamaIndex 标准格式）
```

**空结果降级：** 若两路均为空，返回提示"未找到符合过滤条件的项目，请放宽筛选条件"。

### 5.2 Max Pooling 归约逻辑

```python
# 遍历 Pinecone Top-60 结果
seen_parents = {}  # parent_id -> best_rank
for rank, match in enumerate(pc_results):
    pid = match.metadata["parent_id"]
    if pid not in seen_parents:
        seen_parents[pid] = rank  # 记录最高排名（最小 rank 值）
    if len(seen_parents) >= 20:
        break
# seen_parents 即为向量路的父 chunk 排名列表 A
```

### 5.3 RRF 公式

```python
# score(d) = Σ 1 / (k + rank(d))，k=60
# 合并列表 A 和列表 B，取融合后 Top-5
```

---

## 6. LlamaIndex 接入方式（方案 B）

### 6.1 组件划分

| 层级 | LlamaIndex 组件 | 说明 |
|------|----------------|------|
| 切块 | `SentenceSplitter` | 递归二次切块，token 计数 |
| 向量化 & 索引 | `OpenAIEmbedding` + `PineconeVectorStore` + `VectorStoreIndex` | 替换手写 batch encode + upsert |
| 检索 | `CustomRetriever(BaseRetriever)` | 自定义实现 BM25+向量+RRF，对外暴露标准接口 |
| Query Engine | `RetrieverQueryEngine` | 串联 CustomRetriever + LLM |
| LLM | `OpenAILike` | 指向 AiHubMix base_url，兼容 GLM/GPT |

### 6.2 CustomRetriever

继承 `BaseRetriever`，实现 `_retrieve(query_bundle) -> List[NodeWithScore]`：

- 内部完整实现 BM25 + 向量双路检索 + Max Pooling + RRF
- 返回标准 `NodeWithScore`，每个 node 的 `text` 为父 chunk 完整内容
- 可直接插入 `RetrieverQueryEngine`，无需修改上层代码

### 6.3 Prompt 构建

- 保留现有 `llm.py` 的 System Prompt 内容不变
- 通过 LlamaIndex `PromptTemplate` 注入检索结果和对话历史
- 对话历史截取最近 6 条，每个父 chunk 内容不截断（原方案 800 字符限制废除）

---

## 7. 文件结构变更

```
backend/
├── chunker.py          # 新增 recursive_split()（SentenceSplitter 封装）；token 计数改用 tiktoken
├── indexer.py          # 换用 OpenAIEmbedding + PineconeVectorStore；生成 parent_chunks.json
├── retriever.py        # 重写为 CustomRetriever(BaseRetriever)
├── llm.py              # 小改：环境变量名 GLM_* → LLM_*（已完成）
├── main.py             # 小改：调用方式适配 RetrieverQueryEngine
├── models.py           # 不变
├── bm25_index.pkl      # 重建（基于父 chunk）
├── parent_chunks.json  # 新增（替代 chunk_metadata.json）
├── chunk_metadata.json # 废弃
├── filter_options.json # 不变
└── requirements.txt    # 新增 llama-index、tiktoken、openai；移除 sentence-transformers
```

---

## 8. 环境变量变更

| 旧变量名 | 新变量名 | 说明 |
|---------|---------|------|
| `GLM_API_KEY` | `LLM_API_KEY` | LLM 服务 API Key |
| `GLM_API_URL` | `LLM_API_URL` | LLM 服务 base URL |
| `GLM_MODEL_ID` | `LLM_MODEL_ID` | 模型 ID（如 gpt-4o-mini） |
| `PINECONE_API_KEY` | 不变 | — |
| `PINECONE_INDEX_NAME` | 不变 | — |
| `ALLOWED_ORIGINS` | 不变 | — |

新增：

| 变量名 | 说明 |
|--------|------|
| `OPENAI_API_KEY` | text-embedding-3-small 调用 Key（AiHubMix，同 LLM_API_KEY 值） |
| `OPENAI_BASE_URL` | embedding 服务 base URL（AiHubMix，同 LLM_API_URL 值） |

---

## 9. 依赖变更

**新增：**
```
llama-index
llama-index-vector-stores-pinecone
llama-index-embeddings-openai
llama-index-llms-openai
tiktoken
```

**移除：**
```
sentence-transformers
```

---

## 10. 简历亮点更新

| 技术 | 体现能力 |
|------|---------|
| LlamaIndex 框架 | 工业级 RAG 框架使用经验 |
| Two-level chunking | 对切块策略的深度理解（heading + 递归二次切块） |
| Smaller-to-Bigger Retrieval | Advanced RAG 架构模式（子 chunk 检索，父 chunk 返回） |
| Max Pooling 归约 | 自定义多子块聚合逻辑，解决结果坍缩问题 |
| text-embedding-3-small | OpenAI 多语言嵌入模型，支持跨语言检索 |
| BM25 + RRF 混合检索 | 保留，自定义实现而非框架默认配置 |
