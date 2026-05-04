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
