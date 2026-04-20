# GitHub RAG Agent Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the existing single-pass RAG pipeline into a mid-tier ReAct agent that calls GitHub API tools, reasons over results, and streams its thinking process to the frontend via SSE.

**Architecture:** FastAPI backend drives a ReAct loop (max 6 iterations): RAG always runs first, then the LLM uses OpenAI function calling to invoke GitHub API tools (issues, commits, PRs, file content) until it calls `final_answer`. Frontend renders a collapsible `ThinkingPanel` showing live Thought/Action/Observation events, followed by the streamed final answer.

**Tech Stack:** Python 3.11, FastAPI, OpenAI SDK (GLM via aihubmix.com), httpx, React 18, Vite, Tailwind CSS

**Spec:** `docs/superpowers/specs/2026-04-20-github-rag-agent-design.md`

---

## File Map

### Backend
| File | Change | Responsibility |
|------|--------|---------------|
| `backend/tools.py` | Create | Tool schemas, `build_tool_registry()` factory, GitHub API functions |
| `backend/agent.py` | Create | ReAct loop: drives LLM → tool dispatch → SSE event emission |
| `backend/llm.py` | Modify | Add `call_with_tools()` and `stream_final_answer()` |
| `backend/main.py` | Modify | `/api/chat` route calls `agent.run()` instead of direct RAG→LLM |
| `backend/requirements.txt` | Modify | Add `httpx==0.27.0` |
| `backend/test_function_calling.py` | Delete | Temporary test file, no longer needed after Task 1 |

### Frontend
| File | Change | Responsibility |
|------|--------|---------------|
| `frontend/src/api/client.js` | Modify | `onChunk` passes full parsed event object (not just `.text`) |
| `frontend/src/hooks/useChat.js` | Modify | Add `thinkingSteps` state, dispatch by event `type` |
| `frontend/src/components/ThinkingPanel.jsx` | Create | Renders live Thought/Action/Observation feed, collapses on answer |
| `frontend/src/App.jsx` | Modify | Add `ThinkingPanel` between `FilterPanel` and `ChatWindow` |

---

## Task 1: tools.py — Tool Schemas and GitHub API Functions

**Files:**
- Create: `backend/tools.py`

- [ ] **Step 1: Create tools.py with tool schemas**

`backend/tools.py`:
```python
import os, json
import httpx
from typing import Callable
from retriever import hybrid_search

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Search the local GitHub repository index using hybrid BM25 + vector retrieval. Always the first tool to call. Can be called again with a more specific query if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query in natural language"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_repo_issues",
            "description": "Get recent issues for a GitHub repository. Use to assess bug count, known problems, and community activity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Repository in owner/repo format, e.g. scrapy/scrapy"},
                    "state": {"type": "string", "enum": ["open", "closed", "all"], "description": "Issue state filter"},
                    "limit": {"type": "integer", "description": "Number of issues to return (max 10)", "default": 10},
                },
                "required": ["repo"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_repo_commits",
            "description": "Get recent commits for a GitHub repository. Use to assess whether the project is actively maintained.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Repository in owner/repo format"},
                    "limit": {"type": "integer", "description": "Number of commits to return (max 10)", "default": 10},
                },
                "required": ["repo"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_repo_pulls",
            "description": "Get recent pull requests for a GitHub repository. Use to assess community ecosystem and development roadmap.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Repository in owner/repo format"},
                    "state": {"type": "string", "enum": ["open", "closed", "all"], "description": "PR state filter"},
                    "limit": {"type": "integer", "description": "Number of PRs to return (max 10)", "default": 10},
                },
                "required": ["repo"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_file_content",
            "description": "Get the content of a specific file in a GitHub repository. Use to inspect README, usage examples, or API design.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Repository in owner/repo format"},
                    "path": {"type": "string", "description": "File path within the repo, e.g. README.md"},
                },
                "required": ["repo", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Call this when you have gathered enough information to answer the user's question. Provide the complete final answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "The complete answer to present to the user"},
                },
                "required": ["answer"],
            },
        },
    },
]

def _gh_headers() -> dict:
    token = os.environ.get("GITHUB_TOKEN", "")
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

def _gh_get(url: str, params: dict = None) -> dict | list | str:
    try:
        r = httpx.get(url, headers=_gh_headers(), params=params, timeout=10)
        if r.status_code == 404:
            return {"error": "仓库不存在或无权访问"}
        if r.status_code == 403:
            return {"error": "GitHub API 速率限制，请稍后重试"}
        r.raise_for_status()
        return r.json()
    except httpx.TimeoutException:
        return {"error": "GitHub API 请求超时"}
    except Exception as e:
        return {"error": f"GitHub API 错误: {str(e)[:100]}"}

def _fmt_issues(data: list) -> str:
    if not data:
        return "没有找到 issues。"
    lines = []
    for item in data[:10]:
        labels = ", ".join(l["name"] for l in item.get("labels", []))
        lines.append(f"#{item['number']} [{item['state']}] {item['title']} (comments: {item.get('comments', 0)}, labels: {labels or '无'}, created: {item['created_at'][:10]})")
    return "\n".join(lines)

def _fmt_commits(data: list) -> str:
    if not data:
        return "没有找到 commits。"
    lines = []
    for item in data[:10]:
        msg = item["commit"]["message"].split("\n")[0][:80]
        author = item["commit"]["author"]["name"]
        date = item["commit"]["author"]["date"][:10]
        lines.append(f"{date} [{author}] {msg}")
    return "\n".join(lines)

def _fmt_pulls(data: list) -> str:
    if not data:
        return "没有找到 PR。"
    lines = []
    for item in data[:10]:
        merged = item.get("merged_at", "")[:10] if item.get("merged_at") else "-"
        lines.append(f"#{item['number']} [{item['state']}] {item['title']} (created: {item['created_at'][:10]}, merged: {merged})")
    return "\n".join(lines)

def build_tool_registry(pinecone_index, filters: dict) -> dict[str, Callable]:
    """
    Factory: builds tool registry closed over pinecone_index and filters.
    filters holds the original request filters so rag_search re-calls use the same constraints.
    """
    def tool_rag_search(query: str) -> str:
        docs = hybrid_search(
            query=query,
            pinecone_index=pinecone_index,
            language=filters.get("language", ""),
            min_stars=filters.get("min_stars", 0),
            topics=filters.get("topics", []),
            top_k=5,
        )
        if not docs:
            return "RAG 检索未找到匹配结果，请换一个查询词。"
        parts = []
        for i, doc in enumerate(docs):
            parts.append(
                f"[Repo {i+1}] {doc['full_name']} | {doc.get('language','')} | ★{doc.get('stars',0)}\n"
                f"Description: {doc.get('description','')}\n"
                f"README snippet: {doc.get('content','')[:400]}"
            )
        return "\n\n".join(parts)

    def tool_get_repo_issues(repo: str, state: str = "open", limit: int = 10) -> str:
        data = _gh_get(f"https://api.github.com/repos/{repo}/issues", {"state": state, "per_page": min(limit, 10)})
        if isinstance(data, dict) and "error" in data:
            return data["error"]
        return _fmt_issues(data)

    def tool_get_repo_commits(repo: str, limit: int = 10) -> str:
        data = _gh_get(f"https://api.github.com/repos/{repo}/commits", {"per_page": min(limit, 10)})
        if isinstance(data, dict) and "error" in data:
            return data["error"]
        return _fmt_commits(data)

    def tool_get_repo_pulls(repo: str, state: str = "open", limit: int = 10) -> str:
        data = _gh_get(f"https://api.github.com/repos/{repo}/pulls", {"state": state, "per_page": min(limit, 10)})
        if isinstance(data, dict) and "error" in data:
            return data["error"]
        return _fmt_pulls(data)

    def tool_get_file_content(repo: str, path: str) -> str:
        data = _gh_get(f"https://api.github.com/repos/{repo}/contents/{path}")
        if isinstance(data, dict) and "error" in data:
            return data["error"]
        if isinstance(data, dict) and data.get("encoding") == "base64":
            import base64
            content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
            return content[:4000]
        return "无法读取文件内容。"

    return {
        "rag_search": tool_rag_search,
        "get_repo_issues": tool_get_repo_issues,
        "get_repo_commits": tool_get_repo_commits,
        "get_repo_pulls": tool_get_repo_pulls,
        "get_file_content": tool_get_file_content,
        "final_answer": lambda answer: answer,
    }
```

- [ ] **Step 2: Add httpx to requirements.txt**

Edit `backend/requirements.txt` — add after the last line:
```
httpx==0.27.0
```

Install it:
```bash
cd backend
pip install httpx==0.27.0
```

- [ ] **Step 3: Smoke-test tools in REPL**

```python
import os
from dotenv import load_dotenv
load_dotenv()
from tools import build_tool_registry, _gh_get

# Test GitHub API functions directly (no Pinecone needed)
print(_gh_get("https://api.github.com/repos/scrapy/scrapy/issues", {"per_page": 2}))
```

Expected: list of issue dicts, no "error" key.

- [ ] **Step 4: Commit**

```bash
git add backend/tools.py backend/requirements.txt
git commit -m "feat: tool schemas and GitHub API functions for ReAct agent"
```

---

## Task 2: llm.py — Add call_with_tools and stream_final_answer

**Files:**
- Modify: `backend/llm.py`

- [ ] **Step 1: Add `call_with_tools()` to llm.py**

Add after the existing `_get_client()` function (before `SYSTEM_PROMPT`):

```python
def call_with_tools(messages: list[dict], tools: list[dict]) -> dict:
    """
    Non-streaming call with tool schemas. Returns the raw assistant message dict.
    Raises on network/auth errors; caller handles gracefully.
    """
    c = _get_client()
    response = c.chat.completions.create(
        model=os.environ["GLM_MODEL_ID"],
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    msg = response.choices[0].message
    return {
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in (msg.tool_calls or [])
        ],
    }
```

- [ ] **Step 2: Add `stream_final_answer()` to llm.py**

Add after `call_with_tools()`:

```python
def stream_final_answer(messages: list[dict]) -> Generator[str, None, None]:
    """Stream the final answer given a pre-assembled message list (including all tool history)."""
    c = _get_client()
    try:
        response = c.chat.completions.create(
            model=os.environ["GLM_MODEL_ID"],
            messages=messages,
            stream=True,
            max_tokens=4096,
        )
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                yield f"data: {json.dumps({'type': 'answer', 'text': delta.content}, ensure_ascii=False)}\n\n"
        response.close()
    except RateLimitError:
        yield 'data: {"type": "answer", "text": "请求过于频繁，请稍等几秒后再试。"}\n\n'
    except Exception as e:
        yield f'data: {{"type": "answer", "text": "后端错误：{type(e).__name__}: {str(e)[:100]}"}}\n\n'
    yield "data: [DONE]\n\n"
```

- [ ] **Step 3: Quick test in REPL**

```python
import os
from dotenv import load_dotenv
load_dotenv()
from llm import call_with_tools, stream_final_answer
from tools import TOOL_SCHEMAS

msgs = [{"role": "user", "content": "查一下 scrapy/scrapy 最近有什么 issue？"}]
result = call_with_tools(msgs, TOOL_SCHEMAS)
print(result)
# Expected: dict with tool_calls containing get_repo_issues or rag_search

for chunk in stream_final_answer([{"role": "user", "content": "推荐一个爬虫框架"}]):
    print(chunk, end="")
# Expected: SSE lines with type=answer, ending with [DONE]
```

- [ ] **Step 4: Commit**

```bash
git add backend/llm.py
git commit -m "feat: add call_with_tools and stream_final_answer to llm.py"
```

---

## Task 3: agent.py — ReAct Loop

**Files:**
- Create: `backend/agent.py`

- [ ] **Step 1: Create agent.py**

`backend/agent.py`:
```python
import json
from typing import Generator
from llm import call_with_tools, stream_final_answer, SYSTEM_PROMPT
from tools import TOOL_SCHEMAS, build_tool_registry

MAX_ITERATIONS = 6

def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

def run(
    messages: list[dict],
    filters: dict,
    pinecone_index,
) -> Generator[str, None, None]:
    """
    Main agent entry point. Yields SSE-formatted strings.
    messages: full conversation history (role/content dicts)
    filters: original request filters {language, min_stars, topics}
    pinecone_index: Pinecone index object from main.py lifespan
    """
    registry = build_tool_registry(pinecone_index, filters)

    # --- Step 0: RAG (always runs first, not counted in iterations) ---
    query = messages[-1]["content"]
    rag_result = registry["rag_search"](query)
    yield _sse({"type": "action", "tool": "rag_search", "input": {"query": query}})
    yield _sse({"type": "observation", "text": rag_result[:600]})

    # Build working message list for the ReAct loop
    # System prompt + conversation history + initial RAG context
    history = messages[-6:] if len(messages) > 6 else messages
    prior = history[:-1]
    latest_user = history[-1]["content"]

    working_msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    working_msgs.extend(prior)
    working_msgs.append({
        "role": "user",
        "content": (
            f"以下是 RAG 检索到的相关 GitHub 仓库信息：\n\n{rag_result}\n\n"
            f"用户问题：{latest_user}\n\n"
            f"你可以调用工具获取更多信息，在信息足够时调用 final_answer。"
        ),
    })

    # --- ReAct loop ---
    for iteration in range(MAX_ITERATIONS):
        remaining = MAX_ITERATIONS - iteration
        # Inject remaining iterations hint on last 2 rounds to encourage synthesis
        if remaining <= 2 and iteration > 0:
            working_msgs.append({
                "role": "user",
                "content": f"[系统提示：你还剩 {remaining} 次工具调用机会，请尽快综合已有信息给出答案。]"
            })

        try:
            assistant_msg = call_with_tools(working_msgs, TOOL_SCHEMAS)
        except Exception as e:
            yield _sse({"type": "observation", "text": f"LLM 调用失败: {str(e)[:100]}，强制结束。"})
            break

        # Emit thought if present
        thought_text = assistant_msg.get("content", "").strip()
        if thought_text:
            yield _sse({"type": "thought", "text": thought_text})

        tool_calls = assistant_msg.get("tool_calls", [])

        # No tool calls — treat as implicit final_answer
        if not tool_calls:
            if thought_text:
                yield from stream_final_answer(working_msgs + [assistant_msg])
            else:
                yield from stream_final_answer(working_msgs)
            return

        # Append raw assistant message to maintain OpenAI protocol
        working_msgs.append(assistant_msg)

        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            try:
                tool_args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                tool_args = {}

            yield _sse({"type": "action", "tool": tool_name, "input": tool_args})

            # final_answer is the normal exit path
            if tool_name == "final_answer":
                answer_text = tool_args.get("answer", "")
                # Stream the answer token-by-token (re-use stream_final_answer on the answer text directly)
                yield _sse({"type": "answer", "text": answer_text})
                yield "data: [DONE]\n\n"
                return

            # Execute tool
            func = registry.get(tool_name)
            if func is None:
                observation = f"未知工具: {tool_name}"
            else:
                try:
                    observation = func(**tool_args)
                except Exception as e:
                    observation = f"工具执行错误: {str(e)[:100]}"

            yield _sse({"type": "observation", "text": str(observation)[:800]})

            # Append tool result in OpenAI format
            working_msgs.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": str(observation),
            })

    # Loop exhausted — force final answer with accumulated context
    yield _sse({"type": "thought", "text": "已达到最大工具调用次数，根据已有信息给出答案。"})
    yield from stream_final_answer(working_msgs)
```

- [ ] **Step 2: Smoke-test agent in REPL**

```python
import os
from dotenv import load_dotenv
load_dotenv()
from pinecone import Pinecone
from agent import run

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

filters = {"language": "", "min_stars": 0, "topics": []}
messages = [{"role": "user", "content": "推荐一个活跃维护的 Python 爬虫框架"}]

for event in run(messages, filters, index):
    print(event, end="")
```

Expected: sequence of `thought`, `action`, `observation` SSE events, ending with `answer` events and `[DONE]`. The agent should call at least one GitHub API tool before final_answer.

- [ ] **Step 3: Commit**

```bash
git add backend/agent.py
git commit -m "feat: ReAct agent loop with tool dispatch and SSE event emission"
```

---

## Task 4: main.py — Wire Agent into /api/chat

**Files:**
- Modify: `backend/main.py`

- [ ] **Step 1: Replace chat route to use agent.run()**

In `backend/main.py`:
1. Add `from agent import run as agent_run` to the top-level imports section (alongside other `from ... import` lines)
2. Remove the now-unused `from llm import stream_answer` import
3. Replace the entire `chat` function with:

```python
@app.post("/api/chat")
async def chat(request: ChatRequest):
    import time
    t0 = time.perf_counter()

    filters = request.filters
    filters_dict = {
        "language": filters.language if filters else "",
        "min_stars": filters.min_stars if filters else 0,
        "topics": filters.topics if filters else [],
    }

    messages_dicts = [{"role": m.role, "content": m.content} for m in request.messages]
    index = get_pinecone_index()

    def timed_stream():
        first = True
        t_start = time.perf_counter()
        for chunk in agent_run(messages_dicts, filters_dict, index):
            if first:
                print(f"[timing] first event: {time.perf_counter() - t_start:.2f}s", flush=True)
                first = False
            yield chunk
        print(f"[timing] total: {time.perf_counter() - t0:.2f}s", flush=True)

    return StreamingResponse(
        timed_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```


- [ ] **Step 2: Start backend and test with curl**

```bash
cd backend
uvicorn main:app --reload --port 8000
```

```bash
curl -N -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "推荐一个活跃维护的Python爬虫框架，我关心是否有活跃的社区"}]}'
```

Expected: stream of SSE events with `type` fields (`action`, `observation`, `thought`, `answer`), ending with `[DONE]`.

- [ ] **Step 3: Commit**

```bash
git add backend/main.py
git commit -m "feat: wire agent.run() into /api/chat route"
```

---

## Task 5: client.js — Pass Full Event Object to onChunk

**Files:**
- Modify: `frontend/src/api/client.js`

- [ ] **Step 1: Update onChunk to pass full parsed object**

In `frontend/src/api/client.js`, change line 47:

Old:
```js
if (parsed.text) onChunk(parsed.text);
```

New:
```js
if (parsed.text || parsed.type) onChunk(parsed);
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/api/client.js
git commit -m "feat: pass full SSE event object to onChunk for type-based dispatch"
```

---

## Task 6: useChat.js — Add thinkingSteps State and Type Dispatch

**Files:**
- Modify: `frontend/src/hooks/useChat.js`

- [ ] **Step 1: Rewrite useChat.js**

`frontend/src/hooks/useChat.js`:
```js
import { useState, useCallback } from "react";
import { sendChat } from "../api/client";

export function useChat() {
  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);
  const [thinkingSteps, setThinkingSteps] = useState([]);
  const [filters, setFilters] = useState({
    language: "",
    min_stars: 0,
    topics: [],
  });

  const sendMessage = useCallback(
    (userText) => {
      if (isStreaming || !userText.trim()) return;
      setError(null);
      setThinkingSteps([]); // reset per turn

      const userMsg = { role: "user", content: userText };
      const assistantMsg = { role: "assistant", content: "" };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setIsStreaming(true);

      sendChat({
        messages: [...messages, userMsg],
        filters,
        onChunk: (event) => {
          const type = event.type;

          if (type === "thought") {
            setThinkingSteps((prev) => [...prev, { type: "thought", text: event.text }]);
          } else if (type === "action") {
            setThinkingSteps((prev) => [...prev, { type: "action", tool: event.tool, input: event.input }]);
          } else if (type === "observation") {
            setThinkingSteps((prev) => [...prev, { type: "observation", text: event.text }]);
          } else {
            // type === "answer" or legacy (no type field)
            const text = event.text || "";
            setMessages((prev) => {
              const updated = [...prev];
              updated[updated.length - 1] = {
                ...updated[updated.length - 1],
                content: updated[updated.length - 1].content + text,
              };
              return updated;
            });
          }
        },
        onDone: () => setIsStreaming(false),
        onError: (err) => {
          setIsStreaming(false);
          setError(err.message || "Something went wrong. Please try again.");
        },
      });
    },
    [messages, filters, isStreaming]
  );

  return { messages, isStreaming, error, thinkingSteps, filters, setFilters, sendMessage };
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/hooks/useChat.js
git commit -m "feat: add thinkingSteps state and type-based SSE dispatch in useChat"
```

---

## Task 7: ThinkingPanel.jsx — Live Thinking Feed Component

**Files:**
- Create: `frontend/src/components/ThinkingPanel.jsx`

- [ ] **Step 1: Create ThinkingPanel.jsx**

`frontend/src/components/ThinkingPanel.jsx`:
```jsx
import { useEffect, useRef, useState } from "react";

export function ThinkingPanel({ thinkingSteps, isStreaming, hasAnswer }) {
  const bottomRef = useRef(null);
  const [collapsed, setCollapsed] = useState(false);

  // Auto-collapse when answer starts arriving
  useEffect(() => {
    if (hasAnswer) setCollapsed(true);
  }, [hasAnswer]);

  // Reset collapse state on new turn
  useEffect(() => {
    if (isStreaming && thinkingSteps.length === 0) setCollapsed(false);
  }, [isStreaming, thinkingSteps.length]);

  useEffect(() => {
    if (!collapsed) bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [thinkingSteps, collapsed]);

  if (thinkingSteps.length === 0 && !isStreaming) return null;

  return (
    <div className="border-b bg-gray-50 text-xs">
      <button
        className="w-full flex items-center gap-2 px-4 py-2 text-gray-500 hover:text-gray-700"
        onClick={() => setCollapsed((c) => !c)}
      >
        <span className="text-base">{collapsed ? "▸" : "▾"}</span>
        <span>{collapsed ? "查看思考过程" : "思考过程"}</span>
        {isStreaming && !hasAnswer && (
          <span className="ml-auto animate-pulse text-blue-500">● 思考中...</span>
        )}
      </button>

      {!collapsed && (
        <div className="px-4 pb-3 space-y-1.5 max-h-48 overflow-y-auto">
          {thinkingSteps.map((step, i) => {
            if (step.type === "thought") {
              return (
                <p key={i} className="text-gray-600 italic">
                  💭 {step.text}
                </p>
              );
            }
            if (step.type === "action") {
              const inputStr = step.input ? Object.values(step.input).join(", ") : "";
              return (
                <p key={i} className="text-blue-600 font-medium">
                  🔧 {step.tool}({inputStr})
                </p>
              );
            }
            if (step.type === "observation") {
              return (
                <p key={i} className="text-gray-500 pl-5 border-l-2 border-gray-300">
                  {step.text}
                </p>
              );
            }
            return null;
          })}
          <div ref={bottomRef} />
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/ThinkingPanel.jsx
git commit -m "feat: ThinkingPanel component with live thought/action/observation feed"
```

---

## Task 8: App.jsx — Wire ThinkingPanel into Layout

**Files:**
- Modify: `frontend/src/App.jsx`

- [ ] **Step 1: Update App.jsx**

`frontend/src/App.jsx`:
```jsx
import { useEffect, useState } from "react";
import { ChatWindow } from "./components/ChatWindow";
import { InputBar } from "./components/InputBar";
import { FilterPanel } from "./components/FilterPanel";
import { ThinkingPanel } from "./components/ThinkingPanel";
import { useChat } from "./hooks/useChat";
import { getFilterOptions } from "./api/client";

export default function App() {
  const { messages, isStreaming, error, thinkingSteps, filters, setFilters, sendMessage } = useChat();
  const [filterOptions, setFilterOptions] = useState(null);

  useEffect(() => {
    getFilterOptions()
      .then(setFilterOptions)
      .catch(() => {});
  }, []);

  const hasAnswer = messages.length > 0 && messages[messages.length - 1].role === "assistant"
    && messages[messages.length - 1].content.length > 0;

  return (
    <div className="flex flex-col h-screen max-w-3xl mx-auto border-x">
      <header className="p-4 border-b">
        <h1 className="text-lg font-semibold">GitHub Project Search</h1>
        <p className="text-xs text-gray-500">
          Powered by ReAct Agent — Pinecone + BM25 + GitHub API + GLM
        </p>
      </header>
      <FilterPanel
        filters={filters}
        setFilters={setFilters}
        filterOptions={filterOptions}
      />
      <ThinkingPanel
        thinkingSteps={thinkingSteps}
        isStreaming={isStreaming}
        hasAnswer={hasAnswer}
      />
      <ChatWindow messages={messages} error={error} />
      <InputBar onSend={sendMessage} isStreaming={isStreaming} />
    </div>
  );
}
```

- [ ] **Step 2: Start both servers and do full end-to-end test**

Terminal 1:
```bash
cd backend && uvicorn main:app --reload --port 8000
```

Terminal 2:
```bash
cd frontend && npm run dev
```

Open http://localhost:5173 and test:

1. Ask "推荐一个活跃维护的 Python 爬虫框架" — verify ThinkingPanel appears with tool calls and observations, then collapses when answer starts streaming
2. Click "查看思考过程" to expand collapsed panel
3. Ask a follow-up question — verify `thinkingSteps` resets (old thinking steps disappear)
4. Apply a language filter, ask again — verify filters are respected in RAG results
5. Ask something that should trigger GitHub API tools ("scrapy 最近有哪些新特性") — verify `get_repo_commits` or `get_repo_pulls` appears in the thinking panel

- [ ] **Step 3: Commit**

```bash
git add frontend/src/App.jsx
git commit -m "feat: wire ThinkingPanel into App layout — full ReAct agent UI complete"
```

---

## Task 9: Cleanup

**Files:**
- Delete: `backend/test_function_calling.py`

- [ ] **Step 1: Remove test script**

```bash
# Only run git rm if the file is tracked; if it was never committed, just delete it
git ls-files --error-unmatch backend/test_function_calling.py 2>/dev/null \
  && git rm backend/test_function_calling.py \
  || rm -f backend/test_function_calling.py
git add -A
git commit -m "chore: remove temporary function calling test script"
```

---

## Summary

| Task | Files | Output |
|------|-------|--------|
| 1 | `tools.py`, `requirements.txt` | Tool schemas + GitHub API functions |
| 2 | `llm.py` | `call_with_tools()` + `stream_final_answer()` |
| 3 | `agent.py` | Full ReAct loop with SSE emission |
| 4 | `main.py` | Chat route wired to agent |
| 5 | `client.js` | Full event object passed to `onChunk` |
| 6 | `useChat.js` | `thinkingSteps` state + type dispatch |
| 7 | `ThinkingPanel.jsx` | Live thinking feed component |
| 8 | `App.jsx` | Full integration + end-to-end test |
| 9 | cleanup | Remove temp test file |
