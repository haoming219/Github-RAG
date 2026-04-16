import json, pickle, pathlib
import os
from sentence_transformers import SentenceTransformer

_BASE = pathlib.Path(__file__).parent  # always relative to this file, not cwd

_model = None
_bm25 = None
_chunk_metadata = None

def _load_artifacts():
    global _model, _bm25, _chunk_metadata
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    if _bm25 is None:
        with open(_BASE / "bm25_index.pkl", "rb") as f:
            _bm25 = pickle.load(f)
    if _chunk_metadata is None:
        with open(_BASE / "chunk_metadata.json", encoding="utf-8") as f:
            _chunk_metadata = json.load(f)

def _apply_filter(meta: dict, language: str, min_stars: int, topics: list[str]) -> bool:
    if language and meta.get("language", "") != language:
        return False
    if min_stars and int(meta.get("stars", 0)) < min_stars:
        return False
    if topics:
        repo_topics = meta.get("topics", [])  # stored as real list in chunk_metadata.json
        if not any(t in repo_topics for t in topics):
            return False
    return True

def _rrf(ranked_lists: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion over multiple ranked lists of chunk indices."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)

def hybrid_search(
    query: str,
    pinecone_index,
    language: str = "",
    min_stars: int = 0,
    topics: list[str] = None,
    top_k: int = 5
) -> list[dict]:
    if topics is None:
        topics = []
    _load_artifacts()

    # Build Pinecone metadata filter
    pc_filter = {}
    if language:
        pc_filter["language"] = {"$eq": language}
    if min_stars:
        pc_filter["stars"] = {"$gte": min_stars}
    # topics: stored as a real list in Pinecone (see indexer.py), so $in works directly
    if topics:
        pc_filter["topics"] = {"$in": topics}

    # --- Pinecone vector search ---
    query_emb = _model.encode(query).tolist()
    pc_kwargs = {"vector": query_emb, "top_k": 20, "include_metadata": True}
    if pc_filter:
        pc_kwargs["filter"] = pc_filter
    pc_results = pinecone_index.query(**pc_kwargs)

    # Build Pinecone ranked list: map vector id → chunk_metadata index
    id_to_meta_idx = {
        f"{m['parent_id']}__chunk{m['chunk_index']}": i
        for i, m in enumerate(_chunk_metadata)
    }
    pinecone_ranked = []
    for match in pc_results["matches"]:
        idx = id_to_meta_idx.get(match["id"])
        if idx is not None:
            pinecone_ranked.append(idx)

    # --- BM25 search ---
    tokenized_query = query.lower().split()
    bm25_scores = _bm25.get_scores(tokenized_query)
    bm25_ranked_all = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:20]

    # Post-filter BM25 results
    bm25_ranked = [
        i for i in bm25_ranked_all
        if _apply_filter(_chunk_metadata[i], language, min_stars, topics)
    ]

    # --- RRF fusion ---
    ranked_lists = [pinecone_ranked]
    if bm25_ranked:
        ranked_lists.append(bm25_ranked)
    # If both empty, return empty with a signal
    if not pinecone_ranked and not bm25_ranked:
        return []

    fused = _rrf(ranked_lists)[:top_k]

    results = []
    seen_parents = set()
    for idx in fused:
        meta = _chunk_metadata[idx]
        parent_id = meta["parent_id"]
        # Deduplicate: include only one chunk per repo in the prompt context
        if parent_id in seen_parents:
            continue
        seen_parents.add(parent_id)
        results.append(meta)
        if len(results) >= top_k:
            break

    return results
