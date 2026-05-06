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
从 ContextVar 读取当前请求的 conversation_history  [由 main.py 每轮注入]
         ↓
QueryRewriter.rewrite(query, history)
  ├── history 为空  → 首轮关键词扩展 prompt
  └── history 非空  → 多轮消歧 + 关键词扩展 prompt
         ↓
改写后的 standalone query（英文技术关键词）
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
- **Prompt 核心指令**：根据对话历史理解用户最新问题的真实意图，改写为一条完整独立的英文检索查询，只输出改写结果，不解释
- **History 内容过滤**：只传入 `role` 为 `"user"` 或 `"assistant"` 的消息，过滤 ReAct 中间推理 scratchpad；最多取最近 10 条消息，避免 prompt 过长

**输出语言要求**：两种模式的改写结果均强制为英文，与 Pinecone 索引语言保持一致。

### 降级策略

所有 LLM 调用包裹在 try/except 中。任何异常（超时、API 错误、空响应）均静默降级，返回原始 query，确保检索不中断。`QueryRewriter` 构造时设置显式 HTTP timeout（默认 5 秒），避免卡在无响应的 LLM 请求上。

### 启动失败处理

`QueryRewriter` 在 `lifespan` 中以 fail-soft 方式初始化：若初始化失败（如环境变量缺失），记录 warning 日志，跳过 `init_rewriter`，保持 `_rewriter_instance = None`。运行时 `search_knowledge_base` 检测到 rewriter 为 None 时直接使用原始 query，功能降级但不影响启动。

## 文件变更

### 新增：`backend/agent/tools/query_rewriter.py`

```
QueryRewriter
├── __init__(model, api_key, api_base, timeout=5)  — 持有独立 OpenAI client 实例
└── rewrite(query, history) -> str
    ├── history 为空 → _rewrite_first_turn(query)
    ├── history 非空 → _rewrite_multi_turn(query, history[-10:])  [最近10条]
    └── 任何异常 → 返回原始 query（静默降级）
```

LLM 实例独立于 Agent 的 LLM 实例，从相同环境变量（`LLM_API_KEY`、`LLM_API_URL`、`LLM_MODEL_ID`）初始化。

### 修改：`backend/agent/tools/knowledge_base.py`

使用 `contextvars.ContextVar` 替代模块级 list，实现跨 session 的请求级隔离：

```python
import contextvars
from agent.tools.query_rewriter import QueryRewriter

_rewriter_instance: QueryRewriter | None = None
_conversation_history_var: contextvars.ContextVar[list[dict]] = contextvars.ContextVar(
    "_conversation_history", default=[]
)
```

新增初始化与注入函数：

```python
def init_rewriter(rewriter: QueryRewriter) -> None

def set_conversation_history(history: list[dict]) -> None:
    _conversation_history_var.set(history)
```

`search_knowledge_base` 函数**签名不变**，内部在 `retriever.retrieve` 前插入改写调用：

```python
rewriter = _get_rewriter()
history = _conversation_history_var.get()
effective_query = rewriter.rewrite(query, history) if rewriter else query
nodes = retriever.retrieve(effective_query)
```

### 修改：`backend/main.py`

**lifespan**：以 fail-soft 方式初始化 `QueryRewriter` 并调用 `init_rewriter`：

```python
try:
    from agent.tools.query_rewriter import QueryRewriter
    from agent.tools.knowledge_base import init_rewriter
    rewriter = QueryRewriter(
        model=os.getenv("LLM_MODEL_ID", "gpt-4o-mini"),
        api_key=os.environ["LLM_API_KEY"],
        api_base=os.environ["LLM_API_URL"],
    )
    init_rewriter(rewriter)
except Exception as e:
    logging.warning(f"[startup] QueryRewriter 初始化失败，查询改写已禁用: {e}")
```

**`/agent/chat` endpoint**：在 session lock 内、`agent.stream_chat` 前注入对话历史（只取 user/assistant 消息，过滤 ReAct scratchpad）：

```python
from agent.tools.knowledge_base import set_conversation_history

history = [
    {"role": m.role, "content": m.content}
    for m in agent.chat_history
    if m.role in ("user", "assistant")
]
set_conversation_history(history)
```

`agent.chat_history` 是 LlamaIndex ReActAgent 的内置属性，每轮自动维护。

## 线程安全

`contextvars.ContextVar` 通过 Python asyncio 的 context propagation 机制传播：`loop.run_in_executor` 提交任务时，executor thread 自动继承提交时的 context 副本。因此每个请求的 executor thread 持有自己的 `_conversation_history_var` 值，不同 session 的并发请求天然隔离，无需额外锁。

## 测试

- **`test_query_rewriter.py`**（新增）：
  - 首轮 prompt 分支：验证 history 为空时调用 `_rewrite_first_turn`
  - 多轮 prompt 分支：验证 history 非空时调用 `_rewrite_multi_turn`，且 history 截断为最近 10 条
  - LLM 异常降级：mock LLM 抛出异常，验证返回原始 query
  - 空响应降级：mock LLM 返回空字符串，验证返回原始 query
  - 并发隔离：两个线程分别设置不同 history，验证互不干扰（验证 ContextVar 行为）

- **`test_tools_kb.py`**（扩展）：
  - rewriter 存在时：mock rewriter.rewrite 返回改写 query，验证 retriever 收到改写后的 query
  - rewriter 为 None 时：验证 retriever 收到原始 query（降级路径）
  - 同一 turn 内多次调用：验证两次 `search_knowledge_base` 调用均使用相同 history

## 不在范围内

- 改写结果的持久化或缓存
- 多条改写查询并行检索（Multi-Query 方案）
- `/api/chat`（非 Agent 路径）的查询改写
