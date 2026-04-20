# GitHub RAG Agent Upgrade — Design Spec

**Date:** 2026-04-20  
**Status:** Draft  
**Goal:** Upgrade the existing GitHub RAG system from a single-pass RAG pipeline into a mid-tier ReAct agent that can call GitHub API tools, reason over the results, and stream its thinking process to the frontend.

---

## Background

The current system is a linear pipeline:

```
User input → hybrid_search (BM25 + Pinecone) → build_prompt → stream_answer → SSE
```

This works well for simple "recommend me a framework" queries, but cannot:
- Assess whether a project is actively maintained
- Retrieve real issue counts, recent commits, or PR activity
- Adapt its investigation based on intermediate findings

The upgrade introduces a ReAct (Reason + Act) loop between retrieval and answer generation, keeping the existing RAG as the mandatory first step.

---

## Architecture

### Upgraded Flow

```
User input
  │
  ▼
hybrid_search (RAG — always runs first, not counted in ReAct iterations)
  │
  ▼
ReAct Loop (max 6 iterations)
  ├─ LLM Thought  →  tool choice (function calling)
  ├─ Tool execution (backend)
  ├─ Observation injected into context
  └─ repeat, or LLM calls final_answer → exit loop
  │
  ▼
Stream final answer
  │
  ▼
SSE (three event types: thought / action+observation / answer)
```

### Key Principles

- **RAG is always step zero.** It runs before the ReAct loop and its results are injected as the initial observation. The LLM may call `rag_search` again within the loop with a different query if needed.
- **Tools are pure functions.** Each tool is an independent Python function with no side effects, making them easy to test and extend.
- **ReAct loop is max 6 iterations.** `final_answer` is the normal exit path; the iteration cap is a safety net.
- **Existing files are minimally changed.** `chunker.py`, `indexer.py`, `retriever.py`, `models.py` are untouched.

---

## Tool Set

### Tool Definitions

| Tool | Inputs | Returns | Typical trigger |
|------|--------|---------|----------------|
| `rag_search` | `query: str`, `filters: dict` | Top-5 RAG results (existing hybrid search) | First step always; optionally re-called with a refined query |
| `get_repo_issues` | `repo: str` (owner/repo), `state: str` (open/closed), `limit: int` | List of recent issues: title, labels, created_at, comment_count | User cares about bugs, stability, known problems |
| `get_repo_commits` | `repo: str`, `limit: int` | Recent commits: message, author, date | User wants to know if project is actively maintained |
| `get_repo_pulls` | `repo: str`, `state: str`, `limit: int` | Recent PRs: title, state, created_at, merged_at | User cares about community activity and roadmap |
| `get_file_content` | `repo: str`, `path: str` | File content truncated to 4000 characters | User wants to see API design, usage examples, or core implementation |
| `final_answer` | `answer: str` | None (loop termination signal) | LLM has sufficient information to answer |

### GitHub API Notes

- All GitHub API calls use `GET` endpoints from the public REST API (`https://api.github.com`).
- A `GITHUB_TOKEN` env var is used for authentication to avoid rate limits (60 req/hr unauthenticated vs 5000/hr authenticated).
- Each tool caps its response size before injecting into context (issues: 10 items, commits: 10 items, PRs: 10 items, file: 4000 chars) to prevent context bloat. The file cap is set to 4000 chars to capture installation and usage sections of typical READMEs (common READMEs are 5,000–20,000 chars; 2000 chars is too short).
- GitHub API errors (404, rate limit) are caught and returned as an Observation string, not raised as exceptions — the LLM can reason about failures.

### Tool Extensibility

Adding a new tool requires:
1. Write the Python function in `tools.py`
2. Add its JSON schema to `TOOL_SCHEMAS` list in `tools.py`
3. Register its name in `TOOL_REGISTRY` dict in `tools.py`

No changes to `agent.py`, `llm.py`, or `main.py` are needed.

---

## Backend File Changes

### New Files

**`backend/tools.py`**
- `TOOL_SCHEMAS: list[dict]` — OpenAI-compatible function calling schemas for all 6 tools
- `TOOL_REGISTRY: dict[str, Callable]` — maps tool name → function; built via `build_tool_registry(pinecone_index)` factory function called at startup, which closes over the Pinecone index so `rag_search` can access it without global state
- One function per tool; each returns a plain string (the Observation)
- GitHub API calls use `httpx` (explicitly pinned in requirements.txt; note: `httpx` is also present as a transitive dependency of `openai`, but must be explicitly declared to avoid fragility)

**`backend/agent.py`**
- `run(messages, filters) -> Generator[str, None, None]` — main entry point, yields SSE-formatted strings. The `filters` value from the original request is held in scope and forwarded whenever the LLM invokes `rag_search` — the LLM does not control filter values.
- Internal ReAct loop:
  1. Call `hybrid_search` → emit initial `observation` event
  2. Loop up to `MAX_ITERATIONS = 6`:
     - Call `llm.call_with_tools()` → get Thought + tool choice (or `final_answer`)
     - Emit `thought` event (streamed)
     - If tool call: execute via `TOOL_REGISTRY`, emit `action` + `observation` events
     - If `final_answer`: emit `answer` events (streamed), break
  3. If loop exhausted without `final_answer`: force final answer with current context

### Modified Files

**`backend/llm.py`**
- Add `call_with_tools(messages, tools) -> dict` — non-streaming call that returns the full LLM response including tool call decisions; used inside the ReAct loop. **Pre-condition: verify that the configured `GLM_MODEL_ID` via `aihubmix.com` supports OpenAI-compatible function calling before starting implementation — this is a go/no-go gate.**
- Add `stream_final_answer(messages: list[dict]) -> Generator[str, None, None]` — streaming call for the final answer step; takes the pre-assembled message list (including all tool call/observation history) and streams SSE-formatted strings. This is the sole final-answer path used by `agent.py`.
- Existing `stream_answer()` is kept for backwards compatibility but is no longer called from the main chat path once the agent is active.

**`backend/main.py`**
- `/api/chat` route: replace direct `hybrid_search → stream_answer` with `agent.run()`
- No other changes

**`backend/requirements.txt`**
- Add `httpx==0.27.0` (or `requests==2.32.0`) for GitHub API calls

---

## SSE Event Protocol

All events use the existing `data: {...}\n\n` format. The `type` field is added to distinguish event categories. The frontend falls back gracefully: if `type` is missing, treat as legacy `answer` text.

```
# Thought — LLM reasoning, emitted as a single event after call_with_tools() returns (not token-by-token)
data: {"type": "thought", "text": "scrapy/scrapy 有很高的 star 数，但我需要确认它是否仍在积极维护..."}

# Action — tool invocation, single event
data: {"type": "action", "tool": "get_repo_commits", "input": {"repo": "scrapy/scrapy", "limit": 10}}

# Observation — tool result summary, single event
data: {"type": "observation", "text": "最近10次 commit：最新于2024-11-03，提交者 kmike，消息：'Fix asyncio compatibility...'"}

# Answer — final answer, streamed token by token
data: {"type": "answer", "text": "根据调查，以下是推荐的爬虫框架：\n\n**1. scrapy/scrapy**..."}

# Done
data: [DONE]
```

---

## Frontend Changes

### New UI: Thinking Panel

A collapsible `ThinkingPanel` component sits between `FilterPanel` and `ChatWindow`. It is only visible when `isStreaming === true` and collapses (with a summary line) once the answer starts.

**ThinkingPanel behavior:**
- Shows a live-updating feed of Thought text (streamed, rendered as plain text)
- Each Action is shown as a pill: `🔧 get_repo_commits: scrapy/scrapy`
- Each Observation is shown as an indented summary line below its Action
- Auto-scrolls to the latest entry
- Collapses to "查看思考过程 ▸" once `answer` events begin

### Modified Files

**`frontend/src/hooks/useChat.js`**
- Add `thinkingSteps: []` state — array of `{type, text/tool/input}` objects; **reset to `[]` at the start of each `sendMessage` call** so turn N+1 does not show thinking steps from turn N
- `onChunk` now receives the **full parsed event object** (with `type` field), not just the `.text` string — requires the `client.js` change below
- Dispatches by `type`:
  - `thought` → append as a new thinking step (single complete event, not token-by-token)
  - `action` → append new action step
  - `observation` → append observation to last action step
  - `answer` → append to `messages` (existing behavior)
  - missing `type` → treat as `answer` (backwards compat)

**`frontend/src/api/client.js`**
- Update `sendChat`'s `onChunk` callback to pass the **full parsed event object** instead of just `parsed.text`: `if (parsed.text || parsed.type) onChunk(parsed)`
- This enables `useChat.js` to dispatch by `type` field

**`frontend/src/components/ThinkingPanel.jsx`** (new)
- Renders `thinkingSteps` array
- Collapsed state after answer starts

**`frontend/src/App.jsx`**
- Add `ThinkingPanel` between `FilterPanel` and `ChatWindow`
- Pass `thinkingSteps` and `isStreaming` props

---

## Context Management

To prevent the ReAct loop from bloating the LLM context window:

- **Thought text** is not added as an extra context block. The raw assistant message object (containing `content` and `tool_calls` fields as returned by the API) is appended to the message history as required by the OpenAI function calling protocol; the thought text within `content` travels along with it but is not separately re-injected.
- **Tool calls and observations** follow the standard OpenAI function calling message structure: append assistant message (with tool_calls), then append tool result message with `role: "tool"`. This is required for the API to accept subsequent calls.
- **RAG results** are injected once as the first user-side context block, not repeated
- **Conversation history** is still capped at last 6 messages (existing behavior)
- **Per-tool output cap:** issues/commits/PRs truncated to 10 items, file content to 4000 chars

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| GitHub API 404 | Observation: "仓库不存在或无权访问" — LLM continues with other tools |
| GitHub API rate limit | Observation: "GitHub API 速率限制，跳过此工具" — LLM continues |
| LLM returns malformed tool call | Log warning, treat as `final_answer` with current context |
| ReAct loop hits max iterations | Force `final_answer` call with accumulated context |
| `GITHUB_TOKEN` not set | Tools still work (unauthenticated, 60 req/hr limit); log warning at startup |

---

## Environment Variables

Added to `backend/.env`:
```
GITHUB_TOKEN=your_github_personal_access_token  # optional but recommended
```

---

## Out of Scope

- No changes to the indexer, chunker, or Pinecone index
- No user-facing tool configuration (which tools are enabled is hardcoded in `agent.py`)
- No parallel tool execution (tools run sequentially within each ReAct iteration)
- No persistence of agent reasoning across sessions
- Deployment (Railway/Vercel) configuration is unchanged
