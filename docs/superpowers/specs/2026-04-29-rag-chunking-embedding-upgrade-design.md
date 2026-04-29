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

1. 两层切块：heading 切块 + 二次切块，控制子 chunk 在合理 token 范围内
2. 换用 `text-embedding-3-small`（1536 维，多语言，支持中文提问检索英文内容）
3. 清晰的存储划分：子 chunk 向量化存 Pinecone，父 chunk 完整文本存本地
4. 引入 LlamaIndex 框架（方案 B：`SentenceSplitter` 切块 + `OpenAIEmbedding` 向量化 + `CustomRetriever` 自定义检索 + `RetrieverQueryEngine` 离线测试）
5. 统一环境变量命名：`GLM_*` → `LLM_*`

---

## 3. 整体架构变更

### 技术选型对比

| 层级 | 升级前 | 升级后 |
|------|--------|--------|
| 切块策略 | heading 切块（单层） | heading 切块 + 二次切块（两层） |
| 嵌入模型 | `all-MiniLM-L6-v2`（384 维） | `text-embedding-3-small`（1536 维） |
| 向量数据库维度 | 384 | 1536（需删除重建 index） |
| 父块存储 | Pinecone metadata 冗余存储 | 本地 `parent_chunks.json` |
| 子块存储 | Pinecone（含全部元数据冗余） | Pinecone（向量 + 过滤字段 + `parent_id` + `full_name`） |
| BM25 检索对象 | 子 chunk | 父 chunk |
| 向量检索归约 | 最终输出时去重 | 归约阶段 Max Pooling（提前去重） |
| RAG 框架 | 无 | LlamaIndex（SentenceSplitter + OpenAIEmbedding + CustomRetriever） |

### LlamaIndex 使用范围说明

本项目使用 LlamaIndex 的以下组件，**Pinecone 写入（upsert）和向量检索均通过 Pinecone SDK 直接调用**，不使用 `PineconeVectorStore` / `VectorStoreIndex` 管理写入（原因：需要自定义 metadata schema 和 vector ID 格式）：

| 组件 | 来源包 | 用途 |
|------|--------|------|
| `SentenceSplitter` | `llama-index-core` | 二次切块（父 chunk → 子 chunk） |
| `OpenAIEmbedding` | `llama-index-embeddings-openai` | 批量向量化子 chunk |
| `BaseRetriever` | `llama-index-core` | `CustomRetriever` 的基类 |
| `QueryBundle` | `llama-index-core` | `_retrieve()` 方法入参类型 |
| `NodeWithScore`, `TextNode` | `llama-index-core` | `_retrieve()` 返回值类型 |
| `RetrieverQueryEngine` | `llama-index-core` | 离线评估 / 测试用，不在请求路径中 |

---

## 4. 数据层设计

### 4.1 parent_id 定义

**parent_id = `{full_name}__{section_index}`**（全局唯一）

- `full_name`：仓库完整名称，如 `facebook/react`
- `section_index`：该仓库内父 chunk 的序号，从 0 开始
- 示例：`facebook/react__0`、`facebook/react__1`

一个仓库有多个父 chunk（每个 heading section 一个），每个父 chunk 下有一到多个子 chunk。

Pinecone 子 chunk 的 vector ID 格式：`{parent_id}__child{child_index}`
示例：`facebook/react__1__child2`

### 4.2 两层切块策略

**第一层：Heading 切块（父 chunk）**

- 按 `#`～`######` heading 行分割，每个 heading + 其下方正文 = 一个父 chunk
- README 首段（第一个 heading 之前的内容）单独作为 `section_index=0`、`section_title="__intro__"` 的父 chunk
- 最小长度阈值：**20 token**（`tiktoken` `cl100k_base`）。低于阈值的块合并到前一块。此处替换原有的 50 字符阈值
- 父 chunk 不向量化，只存入本地 `parent_chunks.json`

**第二层：二次切块（子 chunk）**

- 使用 LlamaIndex `SentenceSplitter` 对每个父 chunk 的 `content` 进行切块
- 参数：`chunk_size=512`（token），`chunk_overlap=50`（token），tokenizer 设为 `tiktoken` `cl100k_base`
- `SentenceSplitter` 在句子边界切分，不足一句时回退到字符级切分
- 枚举 `SentenceSplitter` 输出，手动赋值 `child_index`（从 0 开始）
- 子 chunk 向量化后存入 Pinecone，携带 `parent_id`、`full_name` 及过滤字段

**无 README 的仓库：**

- 用 `description` 字段创建一个父 chunk（`section_index=0`，`section_title="__description__"`）
- 该父 chunk 同时产生唯一一个子 chunk（`child_index=0`），`content` 与父 chunk 相同

### 4.3 存储划分

| 存储 | 内容 | 用途 |
|------|------|------|
| **Pinecone**（子 chunk） | 1536 维向量 + metadata（见下） | 向量检索 + pre-filter |
| **`parent_chunks.json`**（本地） | `parent_id` → parent chunk 对象（见下） | BM25 检索 + LLM prompt 构建 |
| **`bm25_index.pkl`**（本地） | 基于父 chunk `content` 构建的 BM25Okapi 索引 | BM25 关键词检索 |
| **`bm25_parent_ids.json`**（本地） | 与 BM25 corpus 一一对应的有序 parent_id 列表 | BM25 rank index → parent_id 映射 |
| **`filter_options.json`**（本地） | languages / topics / stars_range | `/api/filters/options` 端点 |

**Pinecone 子 chunk metadata 字段：**

```json
{
  "parent_id": "facebook/react__1",
  "full_name": "facebook/react",
  "section_title": "## Installation",
  "child_index": 2,
  "language": "JavaScript",
  "stars": 232612,
  "forks": 47696,
  "watchers": 232612,
  "issues": 952,
  "topics": ["declarative", "frontend", "javascript"],
  "create_time": "2013-05-24T16:15:54Z",
  "update_time": "2025-02-26T05:53:48Z",
  "push_time": "2025-02-26T00:09:24Z"
}
```

`description`、`clone_url`、子 chunk 文本内容均不存 Pinecone metadata。

**`parent_chunks.json` 条目示例：**

```json
{
  "facebook/react__1": {
    "parent_id": "facebook/react__1",
    "full_name": "facebook/react",
    "section_title": "## Installation",
    "section_index": 1,
    "content": "React has been designed for gradual adoption...",
    "description": "The library for web and native user interfaces.",
    "clone_url": "https://github.com/facebook/react.git",
    "language": "JavaScript",
    "stars": 232612,
    "forks": 47696,
    "watchers": 232612,
    "issues": 952,
    "topics": ["declarative", "frontend", "javascript"],
    "create_time": "2013-05-24T16:15:54Z",
    "update_time": "2025-02-26T05:53:48Z",
    "push_time": "2025-02-26T00:09:24Z"
  }
}
```

**`bm25_parent_ids.json` 格式：**

```json
["facebook/react__0", "facebook/react__1", "torvalds/linux__0", ...]
```

顺序由 `indexer.py` 中 `df.iterrows()` 的遍历顺序决定（即 CSV merge 后的行顺序）。`bm25_index.pkl` 和 `bm25_parent_ids.json` 必须在同一次 `indexer.py` 运行中生成，严禁分开重建。启动时校验：`len(bm25_parent_ids) == bm25.corpus_size`，不一致则抛出异常。`parent_chunks.json` 内部为字典，key 顺序无意义；顺序只对 `bm25_parent_ids.json` 有要求。

### 4.4 Pinecone Index 重建

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
  ├── 向量检索路（CustomRetriever 内部，直接调用 Pinecone SDK）
  │     ↓ pinecone_index.query(vector, top_k=60, filter={...}, include_metadata=True)
  │     ↓ pre-filter 字段：language / stars / topics（OR 逻辑）
  │        forks / watchers / issues 已存入 metadata，当前不作为过滤维度，未来可扩展
  │     ↓ 遍历 Top-60 结果，按 parent_id 去重，凑满 20 个不重复父 chunk（Max Pooling）
  │     → 父 chunk 排名字典 A（parent_id → best_rank）
  │
  ├── BM25 检索路（CustomRetriever 内部，rank_bm25）
  │     ↓ 在 bm25_parent_ids 有序列表上打分，取 Top-20
  │     ↓ post-filter：按 parent_chunks[pid] 中的元数据过滤
  │        过滤字段：language（精确匹配）/ min_stars（≥）/ topics（OR）
  │     → 父 chunk 排名字典 B（parent_id → rank）
  │
  └── RRF 融合（k=60）
        ↓ 两路父 chunk 排名字典合并
        → Top-5 parent_id
        → 从 parent_chunks 读取完整 content + 元数据
        → 构建 List[NodeWithScore] 返回
```

**空结果降级：** 若两路均为空，`_retrieve()` 返回空列表，`main.py` 处理为"未找到符合过滤条件的项目，请放宽筛选条件"。

### 5.2 Max Pooling 归约逻辑

```python
# 遍历 Pinecone Top-60 子 chunk 结果
seen_parents = {}  # parent_id -> best_rank（rank 越小越好）
for rank, match in enumerate(pc_results.matches):
    pid = match.metadata["parent_id"]
    if pid not in seen_parents:
        seen_parents[pid] = rank  # 同一父 chunk 只记录最高排名
    if len(seen_parents) >= 20:
        break
# seen_parents 即为向量路的父 chunk 排名字典 A
```

### 5.3 BM25 post-filter 逻辑

```python
bm25_scores = bm25.get_scores(tokenized_query)
top20_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:20]

bm25_ranked = {}  # parent_id -> rank
rank = 0
for idx in top20_indices:
    pid = bm25_parent_ids[idx]
    meta = parent_chunks[pid]
    if _apply_filter(meta, language, min_stars, topics):
        bm25_ranked[pid] = rank  # rank 按过滤后顺序重新编号（0,1,2,...）
        rank += 1
# bm25_ranked 即为 BM25 路的父 chunk 排名字典 B
# 注意：rank 值基于过滤后的顺序，而非原始 BM25 得分顺序，这是 post-filter 的预期行为
```

### 5.4 RRF 公式

```python
# score(d) = Σ 1 / (k + rank(d))，k=60
# 合并字典 A（向量路）和字典 B（BM25 路），取融合后 Top-5 parent_id
```

---

## 6. LlamaIndex 接入细节

### 6.1 OpenAIEmbedding 初始化

LlamaIndex 不自动读取 `OPENAI_BASE_URL`，必须显式传入：

```python
from llama_index.embeddings.openai import OpenAIEmbedding
embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.environ["LLM_API_KEY"],
    api_base=os.environ["LLM_API_URL"],
)
```

嵌入和 LLM 复用同一组 `LLM_API_KEY` / `LLM_API_URL`，无需额外 `OPENAI_*` 变量。

### 6.2 indexer.py 执行流程

```
1. 读取三个 CSV，合并数据（现有逻辑保留）
2. 对每个仓库（按 df.iterrows() 顺序）：
   a. 清洗 README（现有 clean_readme() 保留）
   b. 第一层：heading 切块 → 生成父 chunk 列表
      - parent_id = f"{full_name}__{section_index}"
      - 短块（< 20 token，tiktoken cl100k_base）合并到前一块
   c. 第二层：对每个父 chunk 用 SentenceSplitter 切块
      - enumerate 输出，手动赋 child_index（从 0 开始）
      - vector ID = f"{parent_id}__child{child_index}"
3. 批量 OpenAIEmbedding.get_text_embedding_batch() 向量化所有子 chunk
4. 直接调用 Pinecone SDK upsert，写入向量 + metadata（见 §4.3）
5. 将所有父 chunk 写入 parent_chunks.json（key=parent_id，按 df.iterrows() 顺序）
6. 提取有序 parent_id 列表（与步骤 2 遍历顺序一致），构建 BM25，同时写入：
   - bm25_index.pkl
   - bm25_parent_ids.json
7. 提取 filter_options（languages / topics / stars_range），写入 filter_options.json
```

**注意：** 步骤 4 和步骤 6 必须在同一次运行中完成，严禁单独重建任一文件。

### 6.3 CustomRetriever 接口

```python
class CustomRetriever(BaseRetriever):
    def __init__(
        self,
        pinecone_index,           # Pinecone Index 对象
        parent_chunks: dict,      # 从 parent_chunks.json 加载
        bm25: BM25Okapi,          # 从 bm25_index.pkl 加载
        bm25_parent_ids: list,    # 从 bm25_parent_ids.json 加载
        embed_model: OpenAIEmbedding,
    ): ...

    # 过滤参数在每次检索前通过属性设置（见下方线程安全说明）
    # retriever.language = "Python"
    # retriever.min_stars = 1000
    # retriever.topics = ["web"]

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # 实现 §5.1～§5.4 的完整检索逻辑
        # 每个返回的 NodeWithScore：
        #   node.text = parent_chunk["content"]
        #   node.metadata = parent_chunk（全部字段）
        ...
```

### 6.4 main.py 调用方式

`main.py` 保留现有 SSE 流式输出架构，不使用 `RetrieverQueryEngine.query()`（不支持 SSE streaming）。

```python
# lifespan 启动时初始化 CustomRetriever
retriever = CustomRetriever(
    pinecone_index=pc.Index(...),
    parent_chunks=json.load(open("parent_chunks.json")),
    bm25=pickle.load(open("bm25_index.pkl", "rb")),
    bm25_parent_ids=json.load(open("bm25_parent_ids.json")),
    embed_model=OpenAIEmbedding(...),
)
# 启动时校验 BM25 corpus 一致性
assert len(retriever.bm25_parent_ids) == retriever.bm25.corpus_size

# 每次请求
retriever.language = language
retriever.min_stars = min_stars
retriever.topics = topics
nodes = retriever.retrieve(query)                       # List[NodeWithScore]
docs = [n.metadata for n in nodes]                     # 取父 chunk 元数据 + content
return StreamingResponse(stream_answer(docs, messages_dicts), ...)
```

`RetrieverQueryEngine` 仅用于离线评估脚本，不在请求路径中。

**线程安全说明：** 过滤参数通过属性注入（mutable attribute）而非方法参数传递。Railway 部署使用单 worker，不存在并发请求互相覆盖属性的风险。若未来扩展为多 worker，需改为在 `_retrieve()` 中通过 `QueryBundle.custom_embedding_strs` 或工厂方法传入过滤参数。

### 6.5 Prompt 构建（llm.py 变更）

- 保留 System Prompt 内容不变
- `build_prompt_messages()` 中移除 `[:800]` 截断（父 chunk 内容不截断）
- 对话历史截取最近 6 条，逻辑不变

---

## 7. 文件结构变更

```
backend/
├── chunker.py           # 重写：heading 切块（20 token 阈值）+ SentenceSplitter 二次切块
├── indexer.py           # 重写：按 §6.2 流程，OpenAIEmbedding + 直接 Pinecone upsert
├── retriever.py         # 重写：CustomRetriever(BaseRetriever)，实现 §5.1～§5.4
├── llm.py               # 小改：移除 build_prompt_messages 中的 [:800] 截断（env var rename 已完成）
├── main.py              # 小改：lifespan 初始化 CustomRetriever，请求路径调用 retriever.retrieve()
├── models.py            # 不变
├── bm25_index.pkl       # 重建（基于父 chunk content）
├── bm25_parent_ids.json # 新增（BM25 corpus 有序 parent_id 列表）
├── parent_chunks.json   # 新增（替代 chunk_metadata.json）
├── chunk_metadata.json  # 废弃
├── filter_options.json  # 不变
└── requirements.txt     # 见 §9
```

---

## 8. 环境变量变更

| 旧变量名 | 新变量名 | 说明 |
|---------|---------|------|
| `GLM_API_KEY` | `LLM_API_KEY` | LLM + Embedding 服务 API Key（AiHubMix） |
| `GLM_API_URL` | `LLM_API_URL` | LLM + Embedding 服务 base URL（AiHubMix） |
| `GLM_MODEL_ID` | `LLM_MODEL_ID` | LLM 模型 ID（如 gpt-4o-mini） |
| `PINECONE_API_KEY` | 不变 | — |
| `PINECONE_INDEX_NAME` | 不变 | — |
| `ALLOWED_ORIGINS` | 不变 | — |

嵌入模型通过 `LLM_API_KEY` / `LLM_API_URL` 显式初始化，不新增 `OPENAI_*` 变量。

---

## 9. 依赖变更

**新增：**
```
llama-index-core
llama-index-embeddings-openai
tiktoken
```

**保留（已在 requirements.txt）：**
```
openai>=1.51.0    # llm.py 直接调用，以及 OpenAIEmbedding 底层依赖
rank-bm25==0.2.2
pinecone-client==3.2.2
```

**移除：**
```
sentence-transformers
```

**说明：** `llama-index-llms-openai` / `llama-index-llms-openai-like` 均不引入，LLM 调用由 `llm.py` 直接使用 `openai` SDK 完成。`RetrieverQueryEngine` 离线测试时可选安装 `llama-index-core`（已包含）。

---

## 10. 简历亮点更新

| 技术 | 体现能力 |
|------|---------|
| LlamaIndex 框架 | 工业级 RAG 框架使用经验（SentenceSplitter + OpenAIEmbedding + BaseRetriever） |
| Two-level chunking | 对切块策略的深度理解（heading 语义切块 + SentenceSplitter 二次切块） |
| Smaller-to-Bigger Retrieval | Advanced RAG 架构模式（子 chunk 检索，父 chunk 返回） |
| Max Pooling 归约 | 自定义多子块聚合逻辑，解决结果坍缩问题 |
| text-embedding-3-small | OpenAI 多语言嵌入模型，支持跨语言检索 |
| BM25 + RRF 混合检索 | 自定义实现，非框架默认配置，面试可完整讲解 |
