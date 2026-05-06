# Context-Aware Query Rewriting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变 `search_knowledge_base` 函数签名的前提下，为知识库检索添加对话感知的查询改写，解决多轮对话中代词指代和首轮口语化问题。

**Architecture:** 新增独立的 `QueryRewriter` 类（持有自己的 LLM client），通过 `contextvars.ContextVar` 将每轮对话历史以请求级隔离的方式注入 `knowledge_base.py`，在 `retriever.retrieve` 前自动改写 query，对 Agent 完全透明。关键实现细节：`run_in_executor` 不自动传播 ContextVar，必须在 `main.py` 中用 `ctx.run()` 显式捕获并传播 context。

**Tech Stack:** Python 3.10+, openai SDK (OpenAI-compatible), contextvars, FastAPI lifespan, LlamaIndex ReActAgent

---

## File Map

| 操作 | 路径 | 职责 |
|------|------|------|
| 新增 | `backend/agent/tools/query_rewriter.py` | QueryRewriter 类，持有独立 LLM client，实现两种改写策略 |
| 修改 | `backend/agent/tools/knowledge_base.py` | 添加 ContextVar、rewriter 注入函数，在检索前调用改写 |
| 修改 | `backend/main.py` | lifespan 中 fail-soft 初始化 rewriter；`/agent/chat` 中注入对话历史并用 `ctx.run()` 传播 context |
| 新增 | `backend/tests/agent/test_query_rewriter.py` | QueryRewriter 单元测试 |
| 修改 | `backend/tests/agent/test_tools_kb.py` | 修复现有测试（实现返回字符串，不是 dict），添加 rewriter 集成测试 |

---

## Task 1: 实现 QueryRewriter 类

**Files:**
- Create: `backend/agent/tools/query_rewriter.py`
- Test: `backend/tests/agent/test_query_rewriter.py`

- [ ] **Step 1: 写首轮改写的失败测试**

新建 `backend/tests/agent/test_query_rewriter.py`：

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from unittest.mock import MagicMock, patch


def _make_rewriter():
    from agent.tools.query_rewriter import QueryRewriter
    return QueryRewriter(
        model="gpt-4o-mini",
        api_key="test-key",
        api_base="https://api.example.com/v1",
    )


def _mock_completion(text: str):
    """返回一个模拟 OpenAI chat completion 响应。"""
    choice = MagicMock()
    choice.message.content = text
    response = MagicMock()
    response.choices = [choice]
    return response


def test_first_turn_calls_llm_and_returns_rewritten_query():
    rewriter = _make_rewriter()
    with patch.object(rewriter._client.chat.completions, "create",
                      return_value=_mock_completion("Python task queue async worker")) as mock_create:
        result = rewriter.rewrite("有没有能管任务的工具", history=[])

    assert result == "Python task queue async worker"
    mock_create.assert_called_once()
    call_messages = mock_create.call_args.kwargs["messages"]
    assert not any("history" in str(m).lower() for m in call_messages if m["role"] == "system")
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
cd backend && python -m pytest tests/agent/test_query_rewriter.py::test_first_turn_calls_llm_and_returns_rewritten_query -v
```

期望：`ModuleNotFoundError: No module named 'agent.tools.query_rewriter'`

- [ ] **Step 3: 实现 QueryRewriter**

新建 `backend/agent/tools/query_rewriter.py`：

```python
from __future__ import annotations
from openai import OpenAI

_FIRST_TURN_SYSTEM = (
    "You are a search query optimizer for a GitHub repository search engine. "
    "The search index contains English text. "
    "Given the user's question (which may be in any language), "
    "extract the core technical intent and rewrite it as a concise English keyword phrase "
    "suitable for semantic and BM25 retrieval. "
    "Output ONLY the rewritten query. No explanation. No punctuation at the end."
)

_MULTI_TURN_SYSTEM = (
    "You are a search query optimizer for a GitHub repository search engine. "
    "The search index contains English text. "
    "Given a conversation history and the user's latest question, "
    "resolve any pronouns or references (e.g. '它', 'this', 'that project') "
    "using the conversation context, then rewrite the question as a concise, "
    "self-contained English keyword phrase suitable for semantic and BM25 retrieval. "
    "Output ONLY the rewritten query. No explanation. No punctuation at the end."
)


class QueryRewriter:
    def __init__(self, model: str, api_key: str, api_base: str, timeout: float = 5.0):
        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=api_base, timeout=timeout)

    def rewrite(self, query: str, history: list[dict]) -> str:
        try:
            if not history:
                return self._rewrite_first_turn(query)
            return self._rewrite_multi_turn(query, history)
        except Exception:
            return query

    def _rewrite_first_turn(self, query: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _FIRST_TURN_SYSTEM},
                {"role": "user", "content": query},
            ],
            max_tokens=64,
            temperature=0,
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten if rewritten else query

    def _rewrite_multi_turn(self, query: str, history: list[dict]) -> str:
        recent = history[-30:]
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in recent
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _MULTI_TURN_SYSTEM},
                {"role": "user", "content": f"Conversation history:\n{history_text}\n\nLatest question: {query}"},
            ],
            max_tokens=64,
            temperature=0,
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten if rewritten else query
```

- [ ] **Step 4: 运行首轮测试，确认通过**

```bash
cd backend && python -m pytest tests/agent/test_query_rewriter.py::test_first_turn_calls_llm_and_returns_rewritten_query -v
```

期望：`PASSED`

- [ ] **Step 5: 补充其余 QueryRewriter 测试（不含 ContextVar 测试，那在 Task 2 做）**

在 `backend/tests/agent/test_query_rewriter.py` 末尾追加：

```python
def test_multi_turn_includes_history_in_prompt():
    rewriter = _make_rewriter()
    history = [
        {"role": "user", "content": "推荐一个 Python 异步任务队列"},
        {"role": "assistant", "content": "推荐 Celery，它是一个分布式任务队列。"},
    ]
    with patch.object(rewriter._client.chat.completions, "create",
                      return_value=_mock_completion("Celery distributed deployment")) as mock_create:
        result = rewriter.rewrite("它支持分布式部署吗", history=history)

    assert result == "Celery distributed deployment"
    call_messages = mock_create.call_args.kwargs["messages"]
    user_msg = next(m for m in call_messages if m["role"] == "user")
    assert "Celery" in user_msg["content"]
    assert "它支持分布式部署吗" in user_msg["content"]


def test_multi_turn_truncates_history_to_last_30():
    rewriter = _make_rewriter()
    history = [{"role": "user", "content": f"msg {i}"} for i in range(50)]
    with patch.object(rewriter._client.chat.completions, "create",
                      return_value=_mock_completion("rewritten")) as mock_create:
        rewriter.rewrite("latest question", history=history)

    call_messages = mock_create.call_args.kwargs["messages"]
    user_msg = next(m for m in call_messages if m["role"] == "user")
    # 保留 history[-30:]，即 msg 20..49，msg 19 不应出现
    assert "msg 19" not in user_msg["content"]
    assert "msg 20" in user_msg["content"]


def test_llm_exception_returns_original_query():
    rewriter = _make_rewriter()
    with patch.object(rewriter._client.chat.completions, "create",
                      side_effect=Exception("API error")):
        result = rewriter.rewrite("original query", history=[])
    assert result == "original query"


def test_empty_llm_response_returns_original_query():
    rewriter = _make_rewriter()
    with patch.object(rewriter._client.chat.completions, "create",
                      return_value=_mock_completion("")):
        result = rewriter.rewrite("original query", history=[])
    assert result == "original query"
```

- [ ] **Step 6: 运行所有 QueryRewriter 测试，确认全部通过**

```bash
cd backend && python -m pytest tests/agent/test_query_rewriter.py -v
```

期望：所有5个测试 `PASSED`

- [ ] **Step 7: Commit**

```bash
git add backend/agent/tools/query_rewriter.py backend/tests/agent/test_query_rewriter.py
git commit -m "feat(agent): add QueryRewriter with first-turn and multi-turn rewrite strategies"
```

---

## Task 2: 更新 knowledge_base.py 接入 QueryRewriter

**Files:**
- Modify: `backend/agent/tools/knowledge_base.py`
- Modify: `backend/tests/agent/test_tools_kb.py`

- [ ] **Step 1: 修复现有 test_tools_kb.py 并添加新测试（含 ContextVar 隔离测试）**

现有测试断言 `search_knowledge_base` 返回 dict list，但实际实现返回字符串。将 `backend/tests/agent/test_tools_kb.py` 全部替换为：

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from unittest.mock import MagicMock, patch


def _make_node(repo_name, text, score):
    node = MagicMock()
    node.node.metadata = {"full_name": repo_name, "language": "Python", "stars": 500, "topics": ["web"]}
    node.node.get_content.return_value = text
    node.score = score
    return node


def _mock_retriever(nodes):
    r = MagicMock()
    r.retrieve.return_value = nodes
    return r


# ── 基础行为测试 ───────────────────────────────────────────────────────────────

def test_returns_formatted_string_with_repo_names():
    from agent.tools.knowledge_base import search_knowledge_base
    mock_retriever = _mock_retriever([
        _make_node("a/repo", "some text", 0.9),
        _make_node("b/repo", "other text", 0.7),
    ])
    with patch("agent.tools.knowledge_base._get_retriever", return_value=mock_retriever), \
         patch("agent.tools.knowledge_base._get_rewriter", return_value=None):
        result = search_knowledge_base("async task queue")
    assert "a/repo" in result
    assert "b/repo" in result


def test_returns_at_most_5_repos():
    from agent.tools.knowledge_base import search_knowledge_base
    mock_retriever = _mock_retriever([
        _make_node(f"repo/{i}", "text", 0.9 - i * 0.1) for i in range(8)
    ])
    with patch("agent.tools.knowledge_base._get_retriever", return_value=mock_retriever), \
         patch("agent.tools.knowledge_base._get_rewriter", return_value=None):
        result = search_knowledge_base("query")
    # 最多5条结果，用分隔符数量验证：5条结果有4个 "\n\n"
    assert result.count("\n\n") <= 4


def test_empty_results_returns_not_found_message():
    from agent.tools.knowledge_base import search_knowledge_base
    mock_retriever = _mock_retriever([])
    with patch("agent.tools.knowledge_base._get_retriever", return_value=mock_retriever), \
         patch("agent.tools.knowledge_base._get_rewriter", return_value=None):
        result = search_knowledge_base("query")
    assert "未找到" in result


# ── QueryRewriter 集成测试 ─────────────────────────────────────────────────────

def test_rewriter_rewrites_query_before_retrieval():
    """rewriter 存在时，retriever 收到改写后的 query。"""
    from agent.tools.knowledge_base import search_knowledge_base
    mock_retriever = _mock_retriever([_make_node("a/repo", "text", 0.9)])
    mock_rewriter = MagicMock()
    mock_rewriter.rewrite.return_value = "rewritten query"

    with patch("agent.tools.knowledge_base._get_retriever", return_value=mock_retriever), \
         patch("agent.tools.knowledge_base._get_rewriter", return_value=mock_rewriter):
        search_knowledge_base("original query")

    mock_retriever.retrieve.assert_called_once_with("rewritten query")


def test_no_rewriter_uses_original_query():
    """rewriter 为 None 时，retriever 收到原始 query。"""
    from agent.tools.knowledge_base import search_knowledge_base
    mock_retriever = _mock_retriever([_make_node("a/repo", "text", 0.9)])

    with patch("agent.tools.knowledge_base._get_retriever", return_value=mock_retriever), \
         patch("agent.tools.knowledge_base._get_rewriter", return_value=None):
        search_knowledge_base("original query")

    mock_retriever.retrieve.assert_called_once_with("original query")


def test_rewriter_receives_current_history():
    """rewriter.rewrite 被调用时收到当前注入的 history。"""
    from agent.tools.knowledge_base import search_knowledge_base, set_conversation_history
    mock_retriever = _mock_retriever([])
    mock_rewriter = MagicMock()
    mock_rewriter.rewrite.return_value = "query"
    history = [{"role": "user", "content": "previous message"}]

    with patch("agent.tools.knowledge_base._get_retriever", return_value=mock_retriever), \
         patch("agent.tools.knowledge_base._get_rewriter", return_value=mock_rewriter):
        set_conversation_history(history)
        search_knowledge_base("new query")

    mock_rewriter.rewrite.assert_called_once_with("new query", history)


def test_two_calls_in_same_turn_use_same_history():
    """同一 turn 内两次调用 search_knowledge_base，rewriter 均收到相同 history。"""
    from agent.tools.knowledge_base import search_knowledge_base, set_conversation_history
    mock_retriever = _mock_retriever([])
    mock_rewriter = MagicMock()
    mock_rewriter.rewrite.return_value = "query"
    history = [{"role": "user", "content": "context"}]

    with patch("agent.tools.knowledge_base._get_retriever", return_value=mock_retriever), \
         patch("agent.tools.knowledge_base._get_rewriter", return_value=mock_rewriter):
        set_conversation_history(history)
        search_knowledge_base("first call")
        search_knowledge_base("second call")

    assert mock_rewriter.rewrite.call_count == 2
    for call_args in mock_rewriter.rewrite.call_args_list:
        assert call_args.args[1] == history


def test_contextvar_isolation_in_executor_context():
    """验证 ctx.run() 模式下不同 context 的 ContextVar 互不干扰（模拟 run_in_executor 的显式 ctx.run 用法）。"""
    import asyncio
    import contextvars
    from agent.tools.knowledge_base import set_conversation_history, _conversation_history_var

    results = {}

    async def run_two_sessions():
        # Session A 设置 history
        ctx_a = contextvars.copy_context()
        def set_a():
            set_conversation_history([{"role": "user", "content": "hello from A"}])
            results["a"] = _conversation_history_var.get()
        ctx_a.run(set_a)

        # Session B 设置不同 history
        ctx_b = contextvars.copy_context()
        def set_b():
            set_conversation_history([{"role": "user", "content": "hello from B"}])
            results["b"] = _conversation_history_var.get()
        ctx_b.run(set_b)

    asyncio.run(run_two_sessions())
    assert results["a"][0]["content"] == "hello from A"
    assert results["b"][0]["content"] == "hello from B"
    # 主 context 不受影响
    assert _conversation_history_var.get() == []
```

- [ ] **Step 2: 运行测试，确认所有测试因 `_get_rewriter` 不存在而失败**

```bash
cd backend && python -m pytest tests/agent/test_tools_kb.py -v
```

期望：**所有测试 `FAILED`**（`patch` 目标 `agent.tools.knowledge_base._get_rewriter` 不存在，触发 `AttributeError`）

- [ ] **Step 3: 更新 knowledge_base.py**

将 `backend/agent/tools/knowledge_base.py` 全部替换为：

```python
from __future__ import annotations
import contextvars

_retriever_instance = None
_rewriter_instance = None
_conversation_history_var: contextvars.ContextVar[list[dict]] = contextvars.ContextVar(
    "_conversation_history", default=[]
)


def _get_retriever():
    return _retriever_instance


def _get_rewriter():
    return _rewriter_instance


def init_retriever(retriever) -> None:
    global _retriever_instance
    _retriever_instance = retriever


def init_rewriter(rewriter) -> None:
    global _rewriter_instance
    _rewriter_instance = rewriter


def set_conversation_history(history: list[dict]) -> None:
    _conversation_history_var.set(history)


def search_knowledge_base(query: str) -> str:
    """搜索知识库，返回最多5条相关仓库，按相关度降序排列。
    返回格式为纯文本，每条结果包含 repo_name（owner/repo 格式，可直接传给 github_repo_info）和摘要。"""
    retriever = _get_retriever()
    rewriter = _get_rewriter()
    history = _conversation_history_var.get()
    effective_query = rewriter.rewrite(query, history) if rewriter else query

    nodes = retriever.retrieve(effective_query)

    items = []
    for node in nodes:
        meta = node.node.metadata
        items.append({
            "repo_name": meta.get("full_name", ""),
            "score": float(node.score or 0.0),
            "summary": node.node.get_content()[:300],
        })

    items.sort(key=lambda r: r["score"], reverse=True)
    items = items[:5]

    if not items:
        return "知识库中未找到相关仓库。"

    lines = []
    for i, item in enumerate(items, 1):
        lines.append(
            f"[{i}] repo_name: {item['repo_name']}\n"
            f"    摘要: {item['summary']}"
        )
    return "\n\n".join(lines)
```

- [ ] **Step 4: 运行所有 knowledge_base 测试，确认全部通过**

```bash
cd backend && python -m pytest tests/agent/test_tools_kb.py -v
```

期望：所有8个测试 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add backend/agent/tools/knowledge_base.py backend/tests/agent/test_tools_kb.py
git commit -m "feat(agent): integrate QueryRewriter into search_knowledge_base via ContextVar"
```

---

## Task 3: 更新 main.py 注入 QueryRewriter 和对话历史

**Files:**
- Modify: `backend/main.py`

**关键说明**：`loop.run_in_executor` 在 Python 中不自动传播 ContextVar 到 executor 线程。必须在提交前用 `contextvars.copy_context()` 捕获当前 context，再用 `ctx.run()` 在 executor 中执行，才能让 `_conversation_history_var` 的值在 `agent.stream_chat` 执行链中可见。

- [ ] **Step 1: 在 lifespan 中 fail-soft 初始化 QueryRewriter**

在 `backend/main.py` 的 `lifespan` 函数中，`init_kb_retriever(_retriever)` 这行**之后**（`yield` **之前**）添加：

```python
    try:
        from agent.tools.query_rewriter import QueryRewriter
        from agent.tools.knowledge_base import init_rewriter
        _rewriter = QueryRewriter(
            model=os.getenv("LLM_MODEL_ID", "gpt-4o-mini"),
            api_key=os.environ["LLM_API_KEY"],
            api_base=os.environ["LLM_API_URL"],
        )
        init_rewriter(_rewriter)
        print("[startup] QueryRewriter initialized.", flush=True)
    except Exception as e:
        logging.warning(f"[startup] QueryRewriter 初始化失败，查询改写已禁用: {e}")
```

- [ ] **Step 2: 在 `/agent/chat` 中注入对话历史并用 ctx.run() 传播 context**

在 `backend/main.py` 顶部 import 区域添加：

```python
import contextvars
```

将 `agent_chat` → `event_stream` → `async with lock` 块内的 executor 调用部分替换。

**原来的代码**（在 `_session_manager.touch(session_id)` 之后）：

```python
            try:
                loop = asyncio.get_running_loop()
                fut = loop.run_in_executor(None, lambda: agent.stream_chat(request.message))
```

**替换为**：

```python
            from agent.tools.knowledge_base import set_conversation_history
            _history = [
                {"role": str(m.role.value if hasattr(m.role, "value") else m.role),
                 "content": m.content}
                for m in agent.chat_history
                if str(m.role.value if hasattr(m.role, "value") else m.role) in ("user", "assistant")
            ]
            set_conversation_history(_history)

            try:
                loop = asyncio.get_running_loop()
                ctx = contextvars.copy_context()
                fut = loop.run_in_executor(None, lambda: ctx.run(agent.stream_chat, request.message))
```

> **注意**：`m.role` 是 LlamaIndex `MessageRole` 枚举，`.value` 取其字符串值（`"user"` / `"assistant"`）。以计划中的 `.value` 处理为准，直接用 `m.role in ("user", "assistant")` 会因类型不匹配而过滤失败。

- [ ] **Step 3: 本地启动服务验证初始化日志**

```bash
cd backend && uvicorn main:app --reload
```

期望日志中出现：
```
[startup] QueryRewriter initialized.
[startup] Ready.
```

若 `.env` 中没有 `LLM_API_KEY`，期望出现 warning 而非崩溃：
```
WARNING: [startup] QueryRewriter 初始化失败，查询改写已禁用: ...
```

- [ ] **Step 4: Commit**

```bash
git add backend/main.py
git commit -m "feat(main): initialize QueryRewriter in lifespan and inject chat history with ctx.run into /agent/chat"
```

---

## Task 4: 全套测试 & 收尾

- [ ] **Step 1: 运行全部后端测试，确认无回归**

```bash
cd backend && python -m pytest tests/ -v
```

期望：所有测试 `PASSED`，无 `FAILED` 或 `ERROR`

- [ ] **Step 2: 手动端到端验证（需要服务运行中）**

发送两轮对话到 `/agent/chat`：

**第1轮**（首轮改写验证）：
```json
{"session_id": "test-001", "message": "有没有好用的 Python 异步任务队列？"}
```
观察 server 日志：`search_knowledge_base` 被调用，retriever 收到英文关键词 query（而非原始中文）。

**第2轮**（多轮消歧验证）：
```json
{"session_id": "test-001", "message": "它支持分布式部署吗？"}
```
观察 server 日志：`search_knowledge_base` 的 query 应包含第1轮推荐的仓库名，而非直接传入"它"。

- [ ] **Step 3: 检查 git status 并收尾**

```bash
git status
```

若有遗漏未提交的文件，明确 add 后提交：

```bash
git add <specific-file>
git commit -m "chore: finalize context-aware query rewriting implementation"
```
