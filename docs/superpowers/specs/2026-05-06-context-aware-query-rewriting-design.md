# Context-Aware Query Rewriting 设计文档

**日期**: 2026-05-06  
**状态**: 待实现

## 背景与问题

当前 `search_knowledge_base(query)` 直接将 Agent 传入的原始 query 送入向量检索（Pinecone + BM25 hybrid）。这在两种场景下会导致召回质量下降：

1. **多轮对话中的代词指代**：用户在第 N 轮说"它支持分布式部署吗？"，"它"指向前几轮讨论的某个仓库，直接检索完全无法命中。
2. **首轮口语化问题**：用户问"有没有能帮我管任务的工具？"，缺乏技术关键词，向量检索和 BM25 均难以精准命中。

## 目标

在不改变 `search_knowledge_base` 函数签名的前提下，对每次检索的 query 进行自动改写，使其更适合向量检索和 BM25 检索，对 Agent 完全透明。

## 设计方案

### 整体数据流

```
Agent 调用 search_knowledge_base(query)          [签名不变]
         ↓
读取模块级 _conversation_history                 [由 main.py 每轮注入]
         ↓
QueryRewriter.rewrite(query, history)
  ├── history 为空  → 首轮关键词扩展 prompt
  └── history 非空  → 多轮消歧 + 关键词扩展 prompt
         ↓
改写后的 standalone query
         ↓
retriever.retrieve(rewritten_query)              [现有逻辑不变]
         ↓
返回结果给 Agent
```

### 改写策略

#### 首轮（history 为空）：关键词扩展

- **目的**：将口语化问题转化为包含技术关键词的精炼检索短语
- **示例**：`"有没有能帮我管任务的工具？"` → `"Python task queue job scheduling async worker library"`
- **Prompt 核心指令**：将用户问题中的需求提炼为适合语义检索的英文技术关键词短语，只输出改写结果，不解释

#### 多轮（history 非空）：上下文消歧 + 关键词扩展

- **目的**：解析代词指代，结合对话历史还原完整意图，同时补充技术关键词
- **示例**：history 中讨论了 Celery，当前 query `"它支持分布式部署吗？"` → `"Celery distributed deployment support"`
- **Prompt 核心指令**：根据对话历史理解用户最新问题的真实意图，改写为一条完整独立的检索查询，只输出改写结果，不解释

### 降级策略

所有 LLM 调用包裹在 try/except 中。任何异常（超时、API 错误、空响应）均静默降级，返回原始 query，确保检索不中断。

## 文件变更

### 新增：`backend/agent/tools/query_rewriter.py`

```
QueryRewriter
├── __init__(model, api_key, api_base)  — 持有独立 OpenAI client 实例
└── rewrite(query, history) -> str
    ├── history 为空 → _rewrite_first_turn(query)
    ├── history 非空 → _rewrite_multi_turn(query, history)
    └── 任何异常 → 返回原始 query
```

LLM 实例独立于 Agent 的 LLM 实例，从相同环境变量（`LLM_API_KEY`、`LLM_API_URL`、`LLM_MODEL_ID`）初始化。

### 修改：`backend/agent/tools/knowledge_base.py`

新增两个模块级变量（与现有 `_retriever_instance` 同等模式）：

```python
_rewriter_instance: QueryRewriter | None = None
_conversation_history: list[dict] = []
```

新增两个初始化函数：

```python
def init_rewriter(rewriter: QueryRewriter) -> None
def set_conversation_history(history: list[dict]) -> None
```

`search_knowledge_base` 函数**签名不变**，内部在 `retriever.retrieve` 前插入改写调用：

```python
rewriter = _get_rewriter()
effective_query = rewriter.rewrite(query, _conversation_history) if rewriter else query
nodes = retriever.retrieve(effective_query)
```

### 修改：`backend/main.py`

**lifespan**：初始化 `QueryRewriter` 并调用 `init_rewriter`。

**`/agent/chat` endpoint**：在每次请求进入 session lock 后、调用 `agent.stream_chat` 前，注入当前对话历史：

```python
from agent.tools.knowledge_base import set_conversation_history
history = [{"role": m.role, "content": m.content} for m in agent.chat_history]
set_conversation_history(history)
```

`agent.chat_history` 是 LlamaIndex ReActAgent 的内置属性，每轮自动维护。

## 线程安全

`set_conversation_history` 在 `async with lock` 内调用，session lock 保证同一 session 的请求串行执行，不存在并发写冲突。不同 session 的 `_conversation_history` 写入存在竞争，但由于每次检索前必然先 set，且 search_knowledge_base 在同一同步调用链中执行，实际不会错位。

## 测试

- `test_query_rewriter.py`：单元测试 rewriter 的首轮和多轮 prompt 分支，mock LLM 调用，验证降级行为
- `test_tools_kb.py`：扩展现有测试，mock rewriter，验证 `search_knowledge_base` 在 rewriter 存在/缺失时的行为

## 不在范围内

- 改写结果的持久化或缓存
- 多条改写查询并行检索（Multi-Query 方案）
- `/api/chat`（非 Agent 路径）的查询改写
