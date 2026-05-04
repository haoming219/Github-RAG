# ReAct Agent 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 RAG 检索系统之上叠加一个 ReAct Agent 层，使其能够推荐仓库、深度调研、并生成结构化报告。

**Architecture:** 使用 LlamaIndex 原生 `ReActAgent`，将现有 `retriever.py`/`llm.py` 完全保持不变，在 `backend/agent/` 目录下新建 Agent 层。Agent 通过工具集调用知识库与 GitHub API，由 FastAPI 新增 `/agent/chat` 端点暴露 SSE 流式接口，Session 管理存储在进程内存中。

**Tech Stack:** Python 3.10+、LlamaIndex `ReActAgent`、FastAPI SSE、`httpx`（异步 HTTP）、`google-search-results`（SerpAPI SDK）、pytest

---

## 文件结构

### 新建文件

| 文件路径 | 职责 |
|---|---|
| `backend/agent/__init__.py` | 包标记（空文件） |
| `backend/agent/agent.py` | ReActAgent 工厂函数、工具注册、对话循环 |
| `backend/agent/prompts.py` | Agent system prompt 常量 |
| `backend/agent/session.py` | Session 存储、30min 过期、并发队列 |
| `backend/agent/tools/knowledge_base.py` | 封装 `CustomRetriever`，暴露 `search_knowledge_base` 工具 |
| `backend/agent/tools/github.py` | `github_repo_info`、`github_search_code`、`github_get_file` |
| `backend/agent/tools/web_search.py` | `web_search`（SerpAPI 封装） |
| `backend/agent/tools/report.py` | `generate_report`（Markdown 生成与保存） |
| `backend/agent/types.py` | `RepoResult`、`RepoProfile`、`CodeResult` TypedDict |
| `backend/tests/agent/test_tools_kb.py` | `search_knowledge_base` 单元测试 |
| `backend/tests/agent/test_tools_github.py` | GitHub 工具单元测试 |
| `backend/tests/agent/test_tools_web_search.py` | `web_search` 单元测试 |
| `backend/tests/agent/test_tools_report.py` | `generate_report` 单元测试 |
| `backend/tests/agent/test_session.py` | Session 过期与队列测试 |
| `backend/tests/agent/__init__.py` | 包标记（空文件） |

### 修改文件

| 文件路径 | 改动内容 |
|---|---|
| `backend/main.py` | 新增 `/agent/chat` 端点（SSE）、`/agent/reports/{filename}` 端点；lifespan 初始化 Agent |
| `backend/requirements.txt` | 添加 `httpx`、`google-search-results`、`llama-index-agent-openai` |

### 不改动文件

`backend/retriever.py`、`backend/llm.py`、`backend/models.py`、`backend/chunker.py`、`backend/indexer.py`

---

## Task 1：添加依赖

**Files:**
- Modify: `backend/requirements.txt`

- [x] **Step 1: 更新 requirements.txt**

```
# Agent 新增依赖
httpx>=0.27.0              # 异步 HTTP 客户端，用于 GitHub API
google-search-results>=2.4.2  # SerpAPI 官方 Python SDK，web_search 工具使用
llama-index-agent-openai   # LlamaIndex ReActAgent 实现
llama-index-llms-openai    # LlamaIndex OpenAI LLM 包装（agent.py 直接 import）
```

将以上四行追加到 `backend/requirements.txt` 末尾。

- [x] **Step 2: 安装依赖**

```bash
cd backend
pip install httpx google-search-results llama-index-agent-openai llama-index-llms-openai
```

Expected: 无报错，可 `import httpx; from serpapi import GoogleSearch; from llama_index.agent.openai import ReActAgent; from llama_index.llms.openai import OpenAI`

- [x] **Step 3: Commit**

```bash
git add backend/requirements.txt
git commit -m "chore: add httpx, google-search-results, llama-index-agent-openai for ReAct agent"
```

---

## Task 2：类型定义

**Files:**
- Create: `backend/agent/types.py`
- Create: `backend/agent/__init__.py`

- [x] **Step 1: 写失败测试**

创建 `backend/tests/agent/__init__.py`（空文件），然后创建 `backend/tests/agent/test_types.py`：

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from agent.types import RepoResult, RepoProfile, CodeResult

def test_repo_result_keys():
    r: RepoResult = {"repo_name": "a/b", "chunk_text": "x", "score": 0.9, "source": "vector"}
    assert r["score"] == 0.9

def test_repo_profile_keys():
    p: RepoProfile = {
        "full_name": "a/b", "description": "desc", "stars": 100, "forks": 10,
        "language": "Python", "license": "MIT", "readme_summary": "...",
        "last_commit": "2026-01-01", "commits_last_30d": 5,
        "top_contributors": ["alice"], "open_issues_count": 3,
    }
    assert p["stars"] == 100

def test_code_result_keys():
    c: CodeResult = {"file_path": "src/main.py", "snippet": "def f(): ...", "url": "https://github.com/a/b/blob/main/src/main.py"}
    assert "snippet" in c
```

- [x] **Step 2: 运行，确认失败**

```bash
cd backend
pytest tests/agent/test_types.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent'`

- [x] **Step 3: 创建包文件与类型定义**

创建 `backend/agent/__init__.py`（空文件）。

创建 `backend/agent/types.py`：

```python
from typing import TypedDict


class RepoResult(TypedDict):
    repo_name: str
    chunk_text: str
    score: float
    source: str


class RepoProfile(TypedDict):
    full_name: str
    description: str
    stars: int
    forks: int
    language: str
    license: str
    readme_summary: str
    last_commit: str
    commits_last_30d: int
    top_contributors: list[str]
    open_issues_count: int


class CodeResult(TypedDict):
    file_path: str
    snippet: str
    url: str
```

- [x] **Step 4: 运行，确认通过**

```bash
cd backend
pytest tests/agent/test_types.py -v
```

Expected: 3 passed

- [x] **Step 5: Commit**

```bash
git add backend/agent/__init__.py backend/agent/types.py backend/tests/agent/__init__.py backend/tests/agent/test_types.py
git commit -m "feat(agent): add TypedDict definitions for RepoResult, RepoProfile, CodeResult"
```

---

## Task 3：知识库工具 `search_knowledge_base`

**Files:**
- Create: `backend/agent/tools/__init__.py`
- Create: `backend/agent/tools/knowledge_base.py`
- Test: `backend/tests/agent/test_tools_kb.py`

- [x] **Step 1: 写失败测试**

创建 `backend/agent/tools/__init__.py`（空文件）。

创建 `backend/tests/agent/test_tools_kb.py`：

```python
import sys, pathlib, types
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from unittest.mock import MagicMock, patch
from agent.tools.knowledge_base import search_knowledge_base


def _make_node(repo_name, text, score):
    node = MagicMock()
    node.node.metadata = {
        "full_name": repo_name,
        "language": "Python",
        "stars": 500,
        "topics": ["web"],
    }
    node.node.get_content.return_value = text
    node.score = score
    return node


def test_returns_list_of_repo_results():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        _make_node("a/repo", "some text", 0.9),
        _make_node("b/repo", "other text", 0.7),
    ]
    with patch("agent.tools.knowledge_base._get_retriever", return_value=mock_retriever):
        results = search_knowledge_base("async task queue")
    assert len(results) == 2
    assert results[0]["repo_name"] == "a/repo"
    assert results[0]["score"] == 0.9
    assert results[0]["source"] == "hybrid"


def test_returns_at_most_5():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        _make_node(f"repo/{i}", "text", 0.9 - i * 0.1) for i in range(8)
    ]
    with patch("agent.tools.knowledge_base._get_retriever", return_value=mock_retriever):
        results = search_knowledge_base("query")
    assert len(results) <= 5


def test_sorted_by_score_descending():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        _make_node("low/repo", "text", 0.3),
        _make_node("high/repo", "text", 0.9),
    ]
    with patch("agent.tools.knowledge_base._get_retriever", return_value=mock_retriever):
        results = search_knowledge_base("query")
    assert results[0]["score"] >= results[1]["score"]
```

- [x] **Step 2: 运行，确认失败**

```bash
cd backend
pytest tests/agent/test_tools_kb.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent.tools.knowledge_base'`

- [x] **Step 3: 实现 `knowledge_base.py`**

创建 `backend/agent/tools/knowledge_base.py`：

```python
from __future__ import annotations
from typing import TYPE_CHECKING

from agent.types import RepoResult

_retriever_instance = None


def _get_retriever():
    return _retriever_instance


def init_retriever(retriever) -> None:
    global _retriever_instance
    _retriever_instance = retriever


def search_knowledge_base(query: str) -> list[RepoResult]:
    """搜索知识库，返回最多5条相关仓库结果，按相关度降序排列。"""
    retriever = _get_retriever()
    nodes = retriever.retrieve(query)
    results: list[RepoResult] = []
    for node in nodes:
        meta = node.node.metadata
        results.append(RepoResult(
            repo_name=meta.get("full_name", ""),
            chunk_text=node.node.get_content(),
            score=float(node.score or 0.0),
            source="hybrid",
        ))
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:5]
```

- [x] **Step 4: 运行，确认通过**

```bash
cd backend
pytest tests/agent/test_tools_kb.py -v
```

Expected: 3 passed

- [x] **Step 5: Commit**

```bash
git add backend/agent/tools/__init__.py backend/agent/tools/knowledge_base.py backend/tests/agent/test_tools_kb.py
git commit -m "feat(agent): implement search_knowledge_base tool wrapping existing retriever"
```

---

## Task 4：GitHub 工具（`github_repo_info`、`github_search_code`、`github_get_file`）

**Files:**
- Create: `backend/agent/tools/github.py`
- Test: `backend/tests/agent/test_tools_github.py`

- [x] **Step 1: 写失败测试**

创建 `backend/tests/agent/test_tools_github.py`：

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import pytest
from unittest.mock import patch, MagicMock
from agent.tools.github import (
    github_repo_info,
    github_search_code,
    github_get_file,
    _parse_repo_name,
    TRUNCATION_LIMIT,
)


def test_parse_repo_name_owner_slash():
    assert _parse_repo_name("encode/httpx") == "encode/httpx"


def test_parse_repo_name_full_url():
    assert _parse_repo_name("https://github.com/encode/httpx") == "encode/httpx"


def test_parse_repo_name_full_url_trailing_slash():
    assert _parse_repo_name("https://github.com/encode/httpx/") == "encode/httpx"


def _mock_httpx_get(url, **kwargs):
    resp = MagicMock()
    resp.status_code = 200
    if "/repos/encode/httpx" == url.split("api.github.com")[1].rstrip("/"):
        resp.json.return_value = {
            "full_name": "encode/httpx", "description": "A next-gen HTTP client",
            "stargazers_count": 10000, "forks_count": 500, "language": "Python",
            "license": {"spdx_id": "BSD-3-Clause"},
            "pushed_at": "2026-04-01T00:00:00Z", "open_issues_count": 42,
        }
    elif "/readme" in url:
        import base64
        resp.json.return_value = {"content": base64.b64encode(b"# httpx\nAsync HTTP client").decode()}
    elif "/commits" in url:
        resp.json.return_value = [{}] * 15
    elif "/contributors" in url:
        resp.json.return_value = [{"login": "tomchristie"}, {"login": "alex"}]
    else:
        resp.json.return_value = {}
    return resp


def test_github_repo_info_returns_profile():
    with patch("agent.tools.github.httpx.get", side_effect=_mock_httpx_get):
        with patch("agent.tools.github._summarize_readme", return_value="HTTP client library"):
            profile = github_repo_info("encode/httpx")
    assert profile["full_name"] == "encode/httpx"
    assert profile["stars"] == 10000
    assert profile["license"] == "BSD-3-Clause"
    assert profile["top_contributors"] == ["tomchristie", "alex"]


def test_github_repo_info_accepts_url():
    with patch("agent.tools.github.httpx.get", side_effect=_mock_httpx_get):
        with patch("agent.tools.github._summarize_readme", return_value="HTTP client"):
            profile = github_repo_info("https://github.com/encode/httpx")
    assert profile["full_name"] == "encode/httpx"


def test_github_repo_info_rate_limited():
    resp = MagicMock()
    resp.status_code = 429
    with patch("agent.tools.github.httpx.get", return_value=resp):
        result = github_repo_info("encode/httpx")
    assert result["error"] == "rate_limited"


def test_github_search_code_returns_list():
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "items": [
            {
                "path": "src/client.py",
                "html_url": "https://github.com/encode/httpx/blob/main/src/client.py",
                "text_matches": [{"fragment": "class AsyncClient:"}],
            }
        ]
    }
    with patch("agent.tools.github.httpx.get", return_value=resp):
        results = github_search_code("encode/httpx", "AsyncClient")
    assert len(results) == 1
    assert results[0]["file_path"] == "src/client.py"


def test_github_get_file_truncates_long_content():
    import base64
    long_content = "x" * 10000
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"content": base64.b64encode(long_content.encode()).decode()}
    with patch("agent.tools.github.httpx.get", return_value=resp):
        result = github_get_file("encode/httpx", "big_file.py")
    assert "[...内容已截断" in result
    assert len(result) < 10000


def test_github_get_file_short_content_not_truncated():
    import base64
    content = "def hello(): pass"
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"content": base64.b64encode(content.encode()).decode()}
    with patch("agent.tools.github.httpx.get", return_value=resp):
        result = github_get_file("encode/httpx", "small.py")
    assert result == content
```

- [x] **Step 2: 运行，确认失败**

```bash
cd backend
pytest tests/agent/test_tools_github.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent.tools.github'`

- [x] **Step 3: 实现 `github.py`**

创建 `backend/agent/tools/github.py`：

```python
from __future__ import annotations
import os, base64, re
from datetime import datetime, timezone, timedelta

import httpx

from agent.types import RepoProfile, CodeResult

GITHUB_API = "https://api.github.com"
TRUNCATION_LIMIT = 8000
_HALF = TRUNCATION_LIMIT // 2


def _headers() -> dict:
    token = os.getenv("GITHUB_TOKEN", "")
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _parse_repo_name(repo_url_or_name: str) -> str:
    if repo_url_or_name.startswith("http"):
        path = repo_url_or_name.rstrip("/").split("github.com/")[-1]
        parts = path.split("/")
        return f"{parts[0]}/{parts[1]}"
    return repo_url_or_name


def _summarize_readme(text: str) -> str:
    # 延迟导入，避免循环依赖
    from llm import _get_client
    import os
    client = _get_client()
    resp = client.chat.completions.create(
        model=os.getenv("LLM_MODEL_ID", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": "你是技术文档分析专家。用中文提炼以下 README，500字以内，突出项目用途、核心特性、适用场景。"},
            {"role": "user", "content": text[:4000]},
        ],
        max_tokens=700,
    )
    return resp.choices[0].message.content.strip()


def github_repo_info(repo_url_or_name: str) -> RepoProfile | dict:
    """获取 GitHub 仓库的完整信息，包括基础数据、README摘要、贡献者等。接受 'owner/repo' 或完整 GitHub URL。"""
    repo = _parse_repo_name(repo_url_or_name)
    headers = _headers()

    resp = httpx.get(f"{GITHUB_API}/repos/{repo}", headers=headers, timeout=10)
    if resp.status_code in (429, 403):
        return {"error": "rate_limited", "message": "GitHub API 速率限制，请稍后重试"}
    data = resp.json()

    # README
    readme_text = ""
    r_resp = httpx.get(f"{GITHUB_API}/repos/{repo}/readme", headers=headers, timeout=10)
    if r_resp.status_code == 200:
        encoded = r_resp.json().get("content", "")
        readme_text = base64.b64decode(encoded.replace("\n", "")).decode("utf-8", errors="replace")

    readme_summary = _summarize_readme(readme_text) if readme_text else ""

    # 最近30天 commit 数
    since = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    c_resp = httpx.get(
        f"{GITHUB_API}/repos/{repo}/commits",
        headers=headers,
        params={"since": since, "per_page": 100},
        timeout=10,
    )
    commits_last_30d = len(c_resp.json()) if c_resp.status_code == 200 else 0

    # 贡献者（最多5人）
    contrib_resp = httpx.get(
        f"{GITHUB_API}/repos/{repo}/contributors",
        headers=headers,
        params={"per_page": 5},
        timeout=10,
    )
    top_contributors = [c["login"] for c in contrib_resp.json()] if contrib_resp.status_code == 200 else []

    license_id = ""
    if data.get("license"):
        license_id = data["license"].get("spdx_id", "")

    return RepoProfile(
        full_name=data.get("full_name", repo),
        description=data.get("description") or "",
        stars=data.get("stargazers_count", 0),
        forks=data.get("forks_count", 0),
        language=data.get("language") or "",
        license=license_id,
        readme_summary=readme_summary,
        last_commit=data.get("pushed_at", ""),
        commits_last_30d=commits_last_30d,
        top_contributors=top_contributors[:5],
        open_issues_count=data.get("open_issues_count", 0),
    )


def github_search_code(repo: str, query: str) -> list[CodeResult]:
    """在指定仓库内搜索代码，返回最多5条匹配结果。repo 格式为 'owner/repo'。"""
    headers = {**_headers(), "Accept": "application/vnd.github.text-match+json"}
    resp = httpx.get(
        f"{GITHUB_API}/search/code",
        headers=headers,
        params={"q": f"{query} repo:{repo}"},
        timeout=10,
    )
    if resp.status_code in (429, 403):
        return [{"error": "rate_limited", "message": "GitHub API 速率限制，请稍后重试"}]
    items = resp.json().get("items", [])[:5]
    results: list[CodeResult] = []
    for item in items:
        fragments = item.get("text_matches", [])
        snippet = fragments[0]["fragment"] if fragments else ""
        results.append(CodeResult(
            file_path=item.get("path", ""),
            snippet=snippet,
            url=item.get("html_url", ""),
        ))
    return results


def github_get_file(repo: str, path: str) -> str:
    """获取仓库中指定文件的原始内容。超过8000字符时截断中间部分，保留首尾各4000字符。"""
    headers = _headers()
    resp = httpx.get(f"{GITHUB_API}/repos/{repo}/contents/{path}", headers=headers, timeout=10)
    if resp.status_code in (429, 403):
        return "GitHub API 速率限制，请稍后重试"
    if resp.status_code != 200:
        return f"无法获取文件：HTTP {resp.status_code}"

    encoded = resp.json().get("content", "")
    content = base64.b64decode(encoded.replace("\n", "")).decode("utf-8", errors="replace")

    if len(content) <= TRUNCATION_LIMIT:
        return content

    total = len(content)
    return (
        content[:_HALF]
        + f"\n[...内容已截断，共 {total} 字符...]\n"
        + content[-_HALF:]
    )
```

- [x] **Step 4: 运行，确认通过**

```bash
cd backend
pytest tests/agent/test_tools_github.py -v
```

Expected: 8 passed

- [x] **Step 5: Commit**

```bash
git add backend/agent/tools/github.py backend/tests/agent/test_tools_github.py
git commit -m "feat(agent): implement github_repo_info, github_search_code, github_get_file tools"
```

---

## Task 5：`web_search` 工具（SerpAPI 封装）

**Files:**
- Create: `backend/agent/tools/web_search.py`
- Test: `backend/tests/agent/test_tools_web_search.py`

- [x] **Step 1: 写失败测试**

创建 `backend/tests/agent/test_tools_web_search.py`：

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from unittest.mock import patch, MagicMock
from agent.tools.web_search import web_search


def _mock_search_results():
    return {
        "organic_results": [
            {
                "title": "httpx: A next-gen HTTP client",
                "snippet": "httpx is a fully featured HTTP client for Python 3.",
                "link": "https://www.python-httpx.org/",
            },
            {
                "title": "encode/httpx GitHub",
                "snippet": "A next generation HTTP client for Python.",
                "link": "https://github.com/encode/httpx",
            },
        ]
    }


def test_returns_formatted_string():
    with patch("agent.tools.web_search.GoogleSearch") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.get_dict.return_value = _mock_search_results()
        mock_cls.return_value = mock_instance
        result = web_search("httpx Python HTTP client")
    assert "httpx: A next-gen HTTP client" in result
    assert "https://www.python-httpx.org/" in result


def test_returns_at_most_5_results():
    many_results = {
        "organic_results": [
            {"title": f"Result {i}", "snippet": "desc", "link": f"https://example.com/{i}"}
            for i in range(10)
        ]
    }
    with patch("agent.tools.web_search.GoogleSearch") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.get_dict.return_value = many_results
        mock_cls.return_value = mock_instance
        result = web_search("query")
    # 最多5条，每条包含链接，用换行数粗略验证
    assert result.count("https://") <= 5


def test_handles_empty_results():
    with patch("agent.tools.web_search.GoogleSearch") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.get_dict.return_value = {"organic_results": []}
        mock_cls.return_value = mock_instance
        result = web_search("very obscure query")
    assert "未找到" in result or result.strip() == ""


def test_handles_missing_api_key(monkeypatch):
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    result = web_search("test query")
    assert "[web_search 错误" in result


def test_handles_api_exception():
    with patch("agent.tools.web_search.GoogleSearch") as mock_cls:
        mock_cls.side_effect = Exception("network error")
        result = web_search("query")
    assert "[web_search 错误" in result
```

- [x] **Step 2: 运行，确认失败**

```bash
cd backend
pytest tests/agent/test_tools_web_search.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent.tools.web_search'`

- [x] **Step 3: 实现 `web_search.py`**

创建 `backend/agent/tools/web_search.py`：

```python
from __future__ import annotations
import os

from serpapi import GoogleSearch

SEARCH_LIMIT = 5


def web_search(query: str) -> str:
    """在互联网上搜索相关信息，返回最多5条结果（标题、摘要、链接）。适合搜索教程博客、类似仓库推荐、技术说明等。"""
    api_key = os.getenv("SERPAPI_API_KEY", "")
    if not api_key:
        return "[web_search 错误：未配置 SERPAPI_API_KEY]"
    try:
        search = GoogleSearch({"q": query, "api_key": api_key, "num": SEARCH_LIMIT})
        data = search.get_dict()
    except Exception as e:
        return f"[web_search 错误：{e}]"

    results = data.get("organic_results", [])[:SEARCH_LIMIT]
    if not results:
        return "未找到相关结果。"

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "（无标题）")
        snippet = r.get("snippet", "")
        link = r.get("link", "")
        lines.append(f"{i}. **{title}**\n   {snippet}\n   {link}")

    return "\n\n".join(lines)
```

- [x] **Step 4: 运行，确认通过**

```bash
cd backend
pytest tests/agent/test_tools_web_search.py -v
```

Expected: 5 passed

- [x] **Step 5: Commit**

```bash
git add backend/agent/tools/web_search.py backend/tests/agent/test_tools_web_search.py
git commit -m "feat(agent): implement web_search tool via SerpAPI, replacing fetch_url"
```

---

## Task 6：`generate_report` 工具

**Files:**
- Create: `backend/agent/tools/report.py`
- Test: `backend/tests/agent/test_tools_report.py`

- [x] **Step 1: 写失败测试**

创建 `backend/tests/agent/test_tools_report.py`：

```python
import sys, pathlib, shutil
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import pytest
from unittest.mock import patch
from agent.tools.report import generate_report, _safe_filename


def test_safe_filename_replaces_slash():
    assert _safe_filename("encode/httpx") == "encode_httpx"


def test_safe_filename_no_path_traversal():
    name = _safe_filename("../evil/repo")
    assert ".." not in name
    assert "/" not in name


def test_generate_report_creates_file(tmp_path):
    with patch("agent.tools.report.REPORTS_DIR", tmp_path):
        result = generate_report("encode/httpx", {"stars": 10000, "description": "HTTP client"})
    assert "path" in result
    assert "content" in result
    # result["path"] 是相对路径如 "reports/encode_httpx_2026-05-03.md"
    # 文件实际写入到 tmp_path，用 filename 部分拼接验证
    actual_file = tmp_path / pathlib.Path(result["path"]).name
    assert actual_file.exists()
    assert "encode_httpx" in result["path"]


def test_generate_report_no_collision(tmp_path):
    with patch("agent.tools.report.REPORTS_DIR", tmp_path):
        r1 = generate_report("encode/httpx", {"description": "v1"})
        r2 = generate_report("encode/httpx", {"description": "v2"})
    assert r1["path"] != r2["path"]


def test_generate_report_content_includes_repo_name(tmp_path):
    with patch("agent.tools.report.REPORTS_DIR", tmp_path):
        result = generate_report("encode/httpx", {"description": "async http", "stars": 5000})
    assert "encode/httpx" in result["content"]
```

- [x] **Step 2: 运行，确认失败**

```bash
cd backend
pytest tests/agent/test_tools_report.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent.tools.report'`

- [x] **Step 3: 实现 `report.py`**

创建 `backend/agent/tools/report.py`：

```python
from __future__ import annotations
import re
from datetime import date
from pathlib import Path

REPORTS_DIR = Path(__file__).parent.parent.parent.parent / "reports"


def _safe_filename(repo: str) -> str:
    safe = repo.replace("/", "_")
    safe = re.sub(r"[^\w\-]", "", safe)
    return safe


def generate_report(repo: str, content: dict) -> dict:
    """根据收集到的数据生成仓库 Markdown 报告，保存到 reports/ 目录，并返回文件路径与内容。"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    base_name = f"{_safe_filename(repo)}_{today}"
    file_path = REPORTS_DIR / f"{base_name}.md"

    # 文件名冲突时追加后缀
    counter = 1
    while file_path.exists():
        file_path = REPORTS_DIR / f"{base_name}_{counter}.md"
        counter += 1

    md = _render_markdown(repo, content)
    file_path.write_text(md, encoding="utf-8")
    # 返回相对路径（相对于项目根目录），与设计文档规范一致
    relative_path = f"reports/{file_path.name}"
    return {"path": relative_path, "content": md}


def _render_markdown(repo: str, content: dict) -> str:
    profile = content.get("profile", content)
    code_results = content.get("code_results", [])
    kb_results = content.get("kb_results", [])

    stars = profile.get("stars", "N/A")
    forks = profile.get("forks", "N/A")
    language = profile.get("language", "N/A")
    license_ = profile.get("license", "")
    description = profile.get("description", "")
    readme_summary = profile.get("readme_summary", "")
    last_commit = profile.get("last_commit", "")
    commits_30d = profile.get("commits_last_30d", "N/A")
    contributors = ", ".join(profile.get("top_contributors", []))
    open_issues = profile.get("open_issues_count", "N/A")

    code_section = ""
    if code_results:
        code_section = "## 3. 关键代码片段\n\n"
        for r in code_results[:3]:
            code_section += f"**{r.get('file_path', '')}**\n```\n{r.get('snippet', '')}\n```\n[查看源码]({r.get('url', '')})\n\n"

    kb_section = ""
    if kb_results:
        kb_section = "## 5. 知识库相关仓库\n\n"
        for r in kb_results[:3]:
            kb_section += f"- **{r.get('repo_name', '')}** (score: {r.get('score', 0):.2f})\n"
        kb_section += "\n"

    return f"""# 仓库完整画像：{repo}

## 1. 基础信息
- **Stars:** {stars} | **Forks:** {forks} | **主要语言:** {language} | **License:** {license_ or '未知'}
- **最后 Commit:** {last_commit}
- **过去30天 Commit 数:** {commits_30d}

## 2. 项目简介
{description}

{readme_summary}

{code_section}## 4. 社区健康度
- **主要贡献者:** {contributors or '未知'}
- **Open Issues:** {open_issues}

{kb_section}## 6. 总结与推荐
> 以上信息由 ReAct Agent 自动收集整理。
"""
```

- [x] **Step 4: 运行，确认通过**

```bash
cd backend
pytest tests/agent/test_tools_report.py -v
```

Expected: 5 passed

- [x] **Step 5: Commit**

```bash
git add backend/agent/tools/report.py backend/tests/agent/test_tools_report.py
git commit -m "feat(agent): implement generate_report tool saving markdown to reports/"
```

---

## Task 7：System Prompt

**Files:**
- Create: `backend/agent/prompts.py`

- [x] **Step 1: 创建 `prompts.py`**

创建 `backend/agent/prompts.py`：

```python
REACT_AGENT_SYSTEM_PROMPT = """\
你是一个专业的 GitHub 仓库调研 Agent。你能够根据用户需求，从知识库中推荐仓库，并使用 GitHub API 深度调研仓库详情。

## 工具使用优先级
1. 优先调用 search_knowledge_base 检索知识库
2. 知识库结果不足时，调用 GitHub 工具（github_repo_info、github_search_code、github_get_file）补充
3. 需要搜索互联网信息时，调用 web_search（如搜索相关博客、类似仓库推荐、技术说明）

## 行为规则
- 当用户提供 GitHub 仓库链接时，自动触发完整分析：
  1. github_repo_info → 2. github_get_file(README) → 3. github_search_code → 4. search_knowledge_base → 5. generate_report
  整个过程无需用户额外确认，直接生成报告并返回。
- 对于模糊推荐请求（场景 A），先推荐，必要时调用 web_search 补充互联网信息，若用户要求生成报告，再调用 generate_report 前向用户确认。
- 每次工具调用后判断信息是否充足，不足则继续调用。
- 单次对话工具调用总次数上限为 8 次；达到上限时，用已有信息尽力回答，并告知用户：
  "工具调用次数已达上限，以下是基于现有信息的回答"。
- 始终用中文回答用户。
"""
```

- [x] **Step 2: Commit**

```bash
git add backend/agent/prompts.py
git commit -m "feat(agent): add ReAct agent system prompt"
```

---

## Task 8：Session 管理

**Files:**
- Create: `backend/agent/session.py`
- Test: `backend/tests/agent/test_session.py`

- [x] **Step 1: 写失败测试**

创建 `backend/tests/agent/test_session.py`：

```python
import sys, pathlib, asyncio, time
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import pytest
from unittest.mock import MagicMock, patch
from agent.session import SessionManager

SESSION_TTL = 30 * 60  # 30 分钟


@pytest.fixture
def manager():
    return SessionManager(ttl_seconds=2)  # 测试用短 TTL


def test_get_creates_session(manager):
    agent = manager.get_or_create("sid-1", factory=lambda: MagicMock())
    assert agent is not None


def test_get_returns_same_instance(manager):
    factory = MagicMock(side_effect=lambda: MagicMock())
    a1 = manager.get_or_create("sid-2", factory=factory)
    a2 = manager.get_or_create("sid-2", factory=factory)
    assert a1 is a2
    assert factory.call_count == 1


def test_expired_session_recreated(manager):
    factory = MagicMock(side_effect=lambda: MagicMock())
    a1 = manager.get_or_create("sid-3", factory=factory)
    # 手动将时间戳设置为过期
    manager._sessions["sid-3"]["last_active"] = time.time() - 10
    a2 = manager.get_or_create("sid-3", factory=factory)
    assert a1 is not a2
    assert factory.call_count == 2


def test_touch_updates_last_active(manager):
    manager.get_or_create("sid-4", factory=lambda: MagicMock())
    old_ts = manager._sessions["sid-4"]["last_active"]
    time.sleep(0.01)
    manager.touch("sid-4")
    assert manager._sessions["sid-4"]["last_active"] > old_ts
```

- [x] **Step 2: 运行，确认失败**

```bash
cd backend
pytest tests/agent/test_session.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent.session'`

- [x] **Step 3: 实现 `session.py`**

创建 `backend/agent/session.py`：

```python
from __future__ import annotations
import asyncio
import time
from typing import Callable, Any


class SessionManager:
    """进程内 Session 存储，支持 TTL 过期和并发请求队列化。"""

    def __init__(self, ttl_seconds: int = 30 * 60):
        self._ttl = ttl_seconds
        self._sessions: dict[str, dict] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def get_or_create(self, session_id: str, factory: Callable[[], Any]) -> Any:
        entry = self._sessions.get(session_id)
        if entry and (time.time() - entry["last_active"]) < self._ttl:
            return entry["agent"]
        agent = factory()
        self._sessions[session_id] = {"agent": agent, "last_active": time.time()}
        return agent

    def touch(self, session_id: str) -> None:
        if session_id in self._sessions:
            self._sessions[session_id]["last_active"] = time.time()

    async def get_lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]
```

- [x] **Step 4: 运行，确认通过**

```bash
cd backend
pytest tests/agent/test_session.py -v
```

Expected: 4 passed

- [x] **Step 5: Commit**

```bash
git add backend/agent/session.py backend/tests/agent/test_session.py
git commit -m "feat(agent): implement SessionManager with TTL expiry and per-session locks"
```

---

## Task 9：ReAct Agent 主入口

**Files:**
- Create: `backend/agent/agent.py`

- [ ] **Step 1: 创建 `agent.py`**

创建 `backend/agent/agent.py`：

```python
from __future__ import annotations
import os

from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent   # ReActAgent 在 core，不在 agent.openai
from llama_index.llms.openai import OpenAI

from agent.prompts import REACT_AGENT_SYSTEM_PROMPT
from agent.tools.knowledge_base import search_knowledge_base
from agent.tools.github import github_repo_info, github_search_code, github_get_file
from agent.tools.web_search import web_search
from agent.tools.report import generate_report

MAX_ITERATIONS = 8


def _make_tools() -> list[FunctionTool]:
    return [
        FunctionTool.from_defaults(fn=search_knowledge_base),
        FunctionTool.from_defaults(fn=github_repo_info),
        FunctionTool.from_defaults(fn=github_search_code),
        FunctionTool.from_defaults(fn=github_get_file),
        FunctionTool.from_defaults(fn=web_search),
        FunctionTool.from_defaults(fn=generate_report),
    ]


def create_agent() -> ReActAgent:
    llm = OpenAI(
        model=os.getenv("LLM_MODEL_ID", "gpt-4o-mini"),
        api_key=os.getenv("LLM_API_KEY", ""),
        api_base=os.getenv("LLM_API_URL", ""),
    )
    return ReActAgent.from_tools(
        _make_tools(),
        llm=llm,
        system_prompt=REACT_AGENT_SYSTEM_PROMPT,
        max_iterations=MAX_ITERATIONS,
        verbose=True,
    )
```

- [ ] **Step 2: Commit**

```bash
git add backend/agent/agent.py
git commit -m "feat(agent): wire up ReActAgent with all tools and system prompt"
```

---

## Task 10：FastAPI 端点（`/agent/chat` SSE + `/agent/reports/{filename}`）

**Files:**
- Modify: `backend/main.py`

- [ ] **Step 1: 读取现有 `main.py` 确认当前结构**

运行：

```bash
cd backend
grep -n "def \|async def \|@app\." main.py
```

预期看到 `lifespan`、`/api/chat`、`/api/filters/options`、`/health` 等端点。

- [ ] **Step 2: 在 `main.py` 中添加 Agent 初始化与端点**

在 `main.py` 中进行以下修改：

**a) 在顶部 import 区添加：**

```python
import json
import asyncio
from pathlib import Path

from llama_index.core.callbacks import CallbackManager, CBEventType
from llama_index.core.callbacks.base import BaseCallbackHandler
from llama_index.core.agent import AgentChatResponse

from agent.agent import create_agent
from agent.session import SessionManager
from agent.tools.knowledge_base import init_retriever as init_kb_retriever

_session_manager = SessionManager(ttl_seconds=30 * 60)
```

**b) 在 `lifespan` 的 startup 阶段（`yield` 之前），先运行 `grep -n "load_retriever\|_retriever" backend/main.py` 确认变量名，然后追加（实际变量名以 main.py 为准，下面以 `_retriever` 为例）：**

```python
init_kb_retriever(_retriever)  # 注意：变量名以 main.py 中 load_retriever() 赋值为准

# 未配置 GITHUB_TOKEN 时打印警告
import os, logging
if not os.getenv("GITHUB_TOKEN"):
    logging.warning("GITHUB_TOKEN not set — GitHub API rate limit: 60 req/hour (anonymous)")
```

**c) 新增 Pydantic 请求模型（在现有 models import 之后）：**

```python
from pydantic import BaseModel

class AgentChatRequest(BaseModel):
    session_id: str
    message: str
```

**d) 新增两个端点（在 `/health` 端点之后）。注意：`asyncio`、`json`、`CallbackManager`、`CBEventType`、`BaseCallbackHandler`、`AgentChatResponse` 已在 Step 2a 的 import 中添加，此处不要重复导入。**

实现方式：继承 LlamaIndex 的 `AgentChatResponse` streaming 回调，在工具调用发生时向 SSE 队列写入 `agent_step` 消息，最终回答以 `token` 逐字符流式输出。

```python
# 以下 class 和 endpoint 直接追加到 main.py，import 已在 Step 2a 中完成


class _SseStepHandler(BaseCallbackHandler):
    """将 LlamaIndex 工具调用事件写入异步队列，供 SSE 流读取。"""

    TOOL_MESSAGES = {
        "search_knowledge_base": "正在查询知识库...",
        "github_repo_info": "正在从 GitHub 获取仓库信息：{input}",
        "github_search_code": "正在搜索代码：{input}",
        "github_get_file": "正在读取文件：{input}",
        "web_search": "正在搜索互联网：{input}",
        "generate_report": "正在生成完整报告...",
    }

    def __init__(self, queue: asyncio.Queue):
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self._queue = queue
        self._loop = asyncio.get_event_loop()

    def on_event_start(self, event_type, payload=None, **kwargs):
        if event_type == CBEventType.FUNCTION_CALL:
            tool_name = (payload or {}).get("tool", {}).get("name", "")
            tool_input = str((payload or {}).get("tool_input", ""))[:80]
            tpl = self.TOOL_MESSAGES.get(tool_name, f"正在调用工具：{tool_name}")
            msg = tpl.format(input=tool_input)
            self._loop.call_soon_threadsafe(
                self._queue.put_nowait,
                json.dumps({"type": "agent_step", "content": msg}, ensure_ascii=False),
            )

    def on_event_end(self, event_type, payload=None, **kwargs):
        pass

    def start_trace(self, trace_id=None): pass
    def end_trace(self, trace_id=None, trace_map=None): pass


@app.post("/agent/chat")
async def agent_chat(request: AgentChatRequest):
    session_id = request.session_id
    lock = await _session_manager.get_lock(session_id)

    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()
        handler = _SseStepHandler(queue)

        async with lock:
            agent = _session_manager.get_or_create(session_id, factory=create_agent)
            # 注入回调（每次请求重新绑定）
            agent.callback_manager = CallbackManager([handler])
            _session_manager.touch(session_id)

            try:
                loop = asyncio.get_event_loop()

                # 在后台线程运行同步 agent.chat()，期间步骤事件写入队列
                fut = loop.run_in_executor(None, lambda: agent.chat(request.message))

                # 边等待 fut 完成，边把队列中的 agent_step 事件立即 yield 给客户端
                while not fut.done():
                    try:
                        msg = queue.get_nowait()
                        yield f"data: {msg}\n\n"
                    except asyncio.QueueEmpty:
                        await asyncio.sleep(0.05)

                # 清空剩余事件
                while not queue.empty():
                    msg = queue.get_nowait()
                    yield f"data: {msg}\n\n"

                response: AgentChatResponse = await fut
                # 流式输出最终回答（逐 token）
                for char in str(response):
                    yield f"data: {json.dumps({'type': 'token', 'content': char}, ensure_ascii=False)}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

    from fastapi.responses import StreamingResponse
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/agent/reports/{filename}")
async def get_report(filename: str):
    from fastapi.responses import PlainTextResponse
    reports_dir = Path(__file__).parent.parent / "reports"
    # 安全校验：禁止路径穿越
    safe_name = Path(filename).name
    file_path = reports_dir / safe_name
    if not file_path.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="报告不存在")
    return PlainTextResponse(file_path.read_text(encoding="utf-8"))
```

- [ ] **Step 3: 手动启动服务器，验证端点注册**

```bash
cd backend
uvicorn main:app --reload --port 8000
```

在另一终端：

```bash
curl -s http://localhost:8000/health
```

Expected: `{"status":"ok"}`

```bash
curl -s http://localhost:8000/docs | grep -c "agent/chat"
```

Expected: 输出 `1`（端点已注册）

停止服务器（Ctrl+C）。

- [ ] **Step 4: Commit**

```bash
git add backend/main.py
git commit -m "feat(agent): add /agent/chat SSE endpoint and /agent/reports/{filename} to FastAPI"
```

---

## Task 11：全量测试 & 最终验证

- [ ] **Step 1: 运行所有测试，确认全部通过**

```bash
cd backend
pytest tests/ -v
```

Expected: 全部 PASSED，无 FAILED/ERROR

- [ ] **Step 2: 验证目录结构符合设计文档**

```bash
find backend/agent -type f | sort
```

Expected 输出包含：

```
backend/agent/__init__.py
backend/agent/agent.py
backend/agent/prompts.py
backend/agent/session.py
backend/agent/tools/__init__.py
backend/agent/tools/web_search.py
backend/agent/tools/github.py
backend/agent/tools/knowledge_base.py
backend/agent/tools/report.py
backend/agent/types.py
```

- [ ] **Step 3: 端对端冒烟测试（需要实际环境变量）**

确认 `.env` 已配置 `PINECONE_API_KEY`、`LLM_API_KEY`、`LLM_API_URL`，然后：

```bash
cd backend
uvicorn main:app --port 8000 &
curl -s -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test-001", "message": "有没有好用的Python异步HTTP客户端？"}' \
  --no-buffer | head -20
```

Expected: 看到 SSE 格式的 `data: {"type": "token", ...}` 输出流。

- [ ] **Step 4: Commit（如有未提交改动）**

```bash
git add -A
git status  # 确认无遗漏
git commit -m "test(agent): final integration verification"
```

---

## 依赖关系总结

```
Task 1 (依赖) → Task 2 (类型) → Task 3 (KB工具) ┐
                                → Task 4 (GH工具) ├→ Task 9 (Agent) → Task 10 (FastAPI) → Task 11
                                → Task 5 (fetch)  ┤
                                → Task 6 (report) ┘
                                → Task 7 (prompt) ┤
                                → Task 8 (session)┘
```

Tasks 2–8 在 Task 1 完成后可并行实现（互相独立）。Task 9 需要 Tasks 3–7 完成。Task 10 需要 Tasks 8–9 完成。
