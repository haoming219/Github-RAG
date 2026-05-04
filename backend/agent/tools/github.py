from __future__ import annotations
import os, base64
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
    from llm import _get_client
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

    readme_text = ""
    r_resp = httpx.get(f"{GITHUB_API}/repos/{repo}/readme", headers=headers, timeout=10)
    if r_resp.status_code == 200:
        encoded = r_resp.json().get("content", "")
        readme_text = base64.b64decode(encoded.replace("\n", "")).decode("utf-8", errors="replace")

    readme_summary = _summarize_readme(readme_text) if readme_text else ""

    since = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    c_resp = httpx.get(
        f"{GITHUB_API}/repos/{repo}/commits",
        headers=headers,
        params={"since": since, "per_page": 100},
        timeout=10,
    )
    commits_last_30d = len(c_resp.json()) if c_resp.status_code == 200 else 0

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
