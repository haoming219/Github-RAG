from __future__ import annotations

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
