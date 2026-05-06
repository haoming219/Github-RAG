from __future__ import annotations
import contextvars

_retriever_instance = None
_rewriter_instance = None
_MISSING = object()
_conversation_history_var: contextvars.ContextVar = contextvars.ContextVar(
    "_conversation_history", default=_MISSING
)


def _get_history() -> list[dict]:
    val = _conversation_history_var.get()
    return [] if val is _MISSING else val


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
    if retriever is None:
        return "知识库未初始化，请联系管理员。"
    rewriter = _get_rewriter()
    history = _get_history()
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
