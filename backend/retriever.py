import json, pickle, pathlib, os
from typing import Optional
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from llama_index.embeddings.openai import OpenAIEmbedding
from rank_bm25 import BM25Okapi

_BASE = pathlib.Path(__file__).parent


def _apply_filter(meta: dict, language: str, min_stars: int, topics: list) -> bool:
    if language and meta.get("language", "") != language:
        return False
    if min_stars and int(meta.get("stars", 0)) < min_stars:
        return False
    if topics:
        repo_topics = meta.get("topics", [])
        if not any(t in repo_topics for t in topics):
            return False
    return True


def _rrf(ranked_dicts: list[dict], k: int = 60, top_k: int = 100) -> list:
    """
    Reciprocal Rank Fusion over a list of {id: rank} dicts.
    Returns a list of IDs sorted by fused score descending.
    """
    scores: dict = {}
    for ranked in ranked_dicts:
        for pid, rank in ranked.items():
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]


def _max_pool_vector_results(matches: list, limit: int = 20) -> dict:
    """
    Aggregate Pinecone child-chunk matches to parent level.
    For each parent_id, keep only the best (lowest) rank.
    Returns {parent_id: best_rank} for up to `limit` distinct parents.
    """
    seen: dict = {}
    for rank, match in enumerate(matches):
        pid = match.metadata["parent_id"]
        if pid not in seen:
            seen[pid] = rank
        if len(seen) >= limit:
            break
    return seen


class CustomRetriever(BaseRetriever):
    def __init__(
        self,
        pinecone_index,
        parent_chunks: dict,
        bm25: BM25Okapi,
        bm25_parent_ids: list,
        embed_model: OpenAIEmbedding,
    ):
        super().__init__()
        self._pinecone_index = pinecone_index
        self._parent_chunks = parent_chunks
        self._bm25 = bm25
        self._bm25_parent_ids = bm25_parent_ids
        self._embed_model = embed_model
        # Filter attributes — set before each retrieve() call
        self.language: str = ""
        self.min_stars: int = 0
        self.topics: list = []

    def _build_pc_filter(self) -> dict:
        f = {}
        if self.language:
            f["language"] = {"$eq": self.language}
        if self.min_stars:
            f["stars"] = {"$gte": self.min_stars}
        if self.topics:
            f["topics"] = {"$in": self.topics}
        return f

    def _vector_search(self, query: str) -> dict:
        vector = self._embed_model.get_text_embedding(query)
        pc_filter = self._build_pc_filter()
        kwargs = {"vector": vector, "top_k": 60, "include_metadata": True}
        if pc_filter:
            kwargs["filter"] = pc_filter
        results = self._pinecone_index.query(**kwargs)
        return _max_pool_vector_results(results.matches, limit=20)

    def _bm25_search(self, query: str) -> dict:
        tokenized = query.lower().split()
        scores = self._bm25.get_scores(tokenized)
        top20 = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:20]
        ranked = {}
        rank = 0
        for idx in top20:
            pid = self._bm25_parent_ids[idx]
            meta = self._parent_chunks.get(pid, {})
            if _apply_filter(meta, self.language, self.min_stars, self.topics):
                ranked[pid] = rank
                rank += 1
        return ranked

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query = query_bundle.query_str
        vector_ranked = self._vector_search(query)
        bm25_ranked = self._bm25_search(query)

        ranked_lists = [d for d in [vector_ranked, bm25_ranked] if d]
        if not ranked_lists:
            return []

        fused = _rrf(ranked_lists, top_k=5)

        results = []
        for pid in fused:
            chunk = self._parent_chunks.get(pid)
            if not chunk:
                continue
            node = TextNode(text=chunk["content"], metadata=chunk)
            results.append(NodeWithScore(node=node, score=1.0))
            if len(results) >= 5:
                break
        return results


def load_retriever(pinecone_index) -> "CustomRetriever":
    """Load all local artifacts and return a ready CustomRetriever."""
    with open(_BASE / "parent_chunks.json", encoding="utf-8") as f:
        parent_chunks = json.load(f)
    with open(_BASE / "bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open(_BASE / "bm25_parent_ids.json", encoding="utf-8") as f:
        bm25_parent_ids = json.load(f)

    assert len(bm25_parent_ids) == bm25.corpus_size, (
        f"BM25 corpus size mismatch: {len(bm25_parent_ids)} ids vs {bm25.corpus_size} docs. "
        "Regenerate bm25_index.pkl and bm25_parent_ids.json together."
    )

    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.environ["LLM_API_KEY"],
        api_base=os.environ["LLM_API_URL"],
    )
    return CustomRetriever(
        pinecone_index=pinecone_index,
        parent_chunks=parent_chunks,
        bm25=bm25,
        bm25_parent_ids=bm25_parent_ids,
        embed_model=embed_model,
    )
