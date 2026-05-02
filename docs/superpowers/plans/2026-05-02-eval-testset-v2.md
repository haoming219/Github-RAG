# Eval Testset v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the RAG evaluation pipeline to the parent-child chunk architecture: add `last_debug` to `CustomRetriever`, rewrite `generate_testset.py` with embedding-based multi-repo relevant labeling, and rewrite `evaluate.py` with repo-level hit logic and real Recall.

**Architecture:** `CustomRetriever._retrieve()` writes `last_debug` after each call, exposing both vector and BM25 candidate parent_ids for contribution analysis. `generate_testset.py` embeds all ~16k repo representative texts (description + filtered intro), caches to `.npy`, then finds top similar repos per sampled entry to populate multi-repo `relevant_repo_ids`. `evaluate.py` replaces the old `hybrid_search` call with `load_retriever`, deduplicates results by `full_name`, and reads `last_debug` for contribution labels.

**Tech Stack:** Python 3.x, LlamaIndex (`CustomRetriever`), numpy (vector cache + cosine similarity), OpenAI SDK (`text-embedding-3-small`), rank-bm25, Pinecone SDK, existing `backend/parent_chunks.json` + `bm25_index.pkl`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `backend/retriever.py` | Small edit | Add `last_debug` attribute to `CustomRetriever._retrieve()` |
| `backend/tests/test_retriever.py` | Edit | Add test for `last_debug` population |
| `eval/generate_testset.py` | Rewrite | Load parent_chunks.json, embed all repos, find similar repos, generate queries |
| `eval/evaluate.py` | Rewrite | Use CustomRetriever, repo-level dedup, last_debug contribution analysis |
| `eval/repo_embeddings.npy` | Create (not committed) | Float32 matrix of all repo embeddings, shape (N, 1536) |
| `eval/repo_embeddings_index.json` | Create (not committed) | Ordered list of full_names matching npy rows |
| `eval/testset.json` | Regenerate | New testset with multi-repo relevant_repo_ids |
| `.gitignore` | Edit | Add eval/repo_embeddings.npy and eval/repo_embeddings_index.json |

---

## Task 1: Add `last_debug` to `CustomRetriever`

**Files:**
- Modify: `backend/retriever.py:104-124`
- Modify: `backend/tests/test_retriever.py`

- [ ] **Step 1: Write failing test**

Add to `backend/tests/test_retriever.py`:

```python
def test_last_debug_populated():
    """_retrieve() must write last_debug with both candidate lists after every call."""
    import types, json, pickle, pathlib, os
    # Build a minimal CustomRetriever with no-op internals
    from retriever import CustomRetriever, _rrf

    class _FakeEmbed:
        def get_text_embedding(self, text):
            return [0.0] * 1536

    class _FakePinecone:
        def query(self, **kwargs):
            ns = types.SimpleNamespace()
            ns.matches = []
            return ns

    # One-entry BM25 so corpus_size=1
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([["hello"]])
    parent_chunks = {"repo/a__0": {"content": "hello world", "full_name": "repo/a",
                                    "language": "", "stars": 0, "topics": []}}
    bm25_parent_ids = ["repo/a__0"]

    retriever = CustomRetriever(
        pinecone_index=_FakePinecone(),
        parent_chunks=parent_chunks,
        bm25=bm25,
        bm25_parent_ids=bm25_parent_ids,
        embed_model=_FakeEmbed(),
    )

    from llama_index.core.schema import QueryBundle
    retriever._retrieve(QueryBundle(query_str="hello"))

    assert hasattr(retriever, "last_debug")
    assert "vector_candidate_ids" in retriever.last_debug
    assert "bm25_candidate_ids" in retriever.last_debug
    assert isinstance(retriever.last_debug["vector_candidate_ids"], list)
    assert isinstance(retriever.last_debug["bm25_candidate_ids"], list)
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd backend
python -m pytest tests/test_retriever.py::test_last_debug_populated -v
```

Expected: `FAILED` — `AttributeError: 'CustomRetriever' object has no attribute 'last_debug'`

- [ ] **Step 3: Add `last_debug` to `_retrieve()`**

In `backend/retriever.py`, replace the `_retrieve` method:

```python
def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
    query = query_bundle.query_str
    vector_ranked = self._vector_search(query)
    bm25_ranked = self._bm25_search(query)

    self.last_debug = {
        "vector_candidate_ids": list(vector_ranked.keys()),
        "bm25_candidate_ids":   list(bm25_ranked.keys()),
    }

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
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
cd backend
python -m pytest tests/test_retriever.py -v
```

Expected: all PASSED

- [ ] **Step 5: Commit**

```bash
git add backend/retriever.py backend/tests/test_retriever.py
git commit -m "feat(retriever): expose last_debug with vector/BM25 candidate ids after each retrieve"
```

---

## Task 2: Rewrite `generate_testset.py`

**Files:**
- Rewrite: `eval/generate_testset.py`
- Edit: `.gitignore`

This script runs once and produces `eval/testset.json`. It has two phases:
1. Embed all repos → cache to `eval/repo_embeddings.npy` + `eval/repo_embeddings_index.json`
2. Stratified sample 400 repos → for each, find similar repos + generate query via LLM

- [ ] **Step 1: Update `.gitignore`**

Add these two lines to the root `.gitignore` (or `eval/.gitignore` if one exists):

```
eval/repo_embeddings.npy
eval/repo_embeddings_index.json
```

- [ ] **Step 2: Rewrite `eval/generate_testset.py`**

```python
"""
One-time test set generator (v2). Run from project root:
    python eval/generate_testset.py

Reads:  backend/parent_chunks.json
Writes: eval/testset.json          (commit this)
        eval/repo_embeddings.npy   (do NOT commit — ~98MB)
        eval/repo_embeddings_index.json  (do NOT commit)
Env:    LLM_API_KEY, LLM_API_URL, LLM_MODEL_ID  (for embedding + query generation)
"""
import json, os, re, random, sys, pathlib
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).parent.parent / "backend" / ".env")

BACKEND  = pathlib.Path(__file__).parent.parent / "backend"
EVAL_DIR = pathlib.Path(__file__).parent
OUT_PATH        = EVAL_DIR / "testset.json"
EMB_NPY_PATH    = EVAL_DIR / "repo_embeddings.npy"
EMB_INDEX_PATH  = EVAL_DIR / "repo_embeddings_index.json"

# Quota config — same as v1
LANGUAGE_LAYERS    = 5
LANGUAGE_OTHER     = True
PER_LANGUAGE_LAYER = 30
PER_STARS_TIER     = 20
RANDOM_LAYER       = 160
TARGET_TOTAL       = 400

STARS_TIERS = {
    "low":  (0,     999),
    "mid":  (1000,  9999),
    "high": (10000, float("inf")),
}

SIMILARITY_THRESHOLD = 0.75
MAX_EXTRA_RELEVANT   = 4
MIN_EXTRA_RELEVANT   = 2   # if fewer pass threshold, don't add extras

EMBED_MODEL  = "text-embedding-3-small"
EMBED_DIM    = 1536
EMBED_BATCH  = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client() -> OpenAI:
    api_url  = os.environ["LLM_API_URL"]
    base_url = api_url.rsplit("/chat/completions", 1)[0] if "/chat/completions" in api_url else api_url
    return OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=base_url)


def _stars_tier(stars: int) -> str:
    for tier, (lo, hi) in STARS_TIERS.items():
        if lo <= stars <= hi:
            return tier
    return "low"


def _clean_intro(text: str) -> str:
    """Strip HTML tags and &nbsp; from intro content."""
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&nbsp;", " ")
    return text.strip()


def _intro_is_useful(cleaned: str) -> bool:
    """Return True if the cleaned intro adds meaningful content."""
    if len(cleaned) < 30:
        return False
    alpha_count = sum(1 for c in cleaned if c.isalpha())
    if len(cleaned) == 0 or alpha_count / len(cleaned) < 0.20:
        return False
    return True


def _repr_text(chunk: dict) -> str:
    """Build the representative text for a repo: description + filtered intro."""
    description = (chunk.get("description") or "").strip()
    intro_raw   = (chunk.get("content") or "")[:200]
    intro_clean = _clean_intro(intro_raw)
    parts = [p for p in [description, intro_clean if _intro_is_useful(intro_clean) else ""] if p]
    return " \n".join(parts) or chunk.get("full_name", "")


# ---------------------------------------------------------------------------
# Step 1: Load parent_chunks.json → one entry per repo (the __0 chunk)
# ---------------------------------------------------------------------------

def _load_repo_index() -> dict:
    """Return {full_name: chunk_dict} using only the __0 parent chunk per repo."""
    with open(BACKEND / "parent_chunks.json", encoding="utf-8") as f:
        all_chunks = json.load(f)
    repo_index = {}
    for pid, chunk in all_chunks.items():
        if pid.endswith("__0"):
            full_name = chunk.get("full_name") or pid.rsplit("__", 1)[0]
            repo_index[full_name] = chunk
    return repo_index


# ---------------------------------------------------------------------------
# Step 2: Embed all repos (with caching)
# ---------------------------------------------------------------------------

def _embed_all_repos(client: OpenAI, repo_index: dict) -> tuple[np.ndarray, list[str]]:
    """
    Returns (matrix, names) where matrix[i] is the L2-normalised embedding
    for names[i]. Uses cached .npy if present and covers all repos.
    """
    names = list(repo_index.keys())

    # Load cache if it covers the same set of repos
    if EMB_NPY_PATH.exists() and EMB_INDEX_PATH.exists():
        cached_names = json.loads(EMB_INDEX_PATH.read_text(encoding="utf-8"))
        if cached_names == names:
            print("Loading embeddings from cache...", flush=True)
            matrix = np.load(str(EMB_NPY_PATH))
            return matrix, names
        print("Cache stale (repo list changed), re-embedding...", flush=True)

    texts  = [_repr_text(repo_index[n]) for n in names]
    vecs   = []

    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        resp  = client.embeddings.create(model=EMBED_MODEL, input=batch)
        batch_vecs = [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
        vecs.extend(batch_vecs)
        print(f"  Embedded {min(i + EMBED_BATCH, len(texts))}/{len(texts)}", flush=True)

    matrix = np.array(vecs, dtype=np.float32)
    # L2 normalise rows for fast cosine via dot product
    norms  = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    np.save(str(EMB_NPY_PATH), matrix)
    EMB_INDEX_PATH.write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
    print(f"Embeddings cached to {EMB_NPY_PATH}", flush=True)
    return matrix, names


# ---------------------------------------------------------------------------
# Step 3: Stratified sampling — identical logic to v1
# ---------------------------------------------------------------------------

def _stratified_sample(repo_index: dict) -> list[tuple[str, dict, str]]:
    repos = list(repo_index.items())

    lang_counts: dict[str, int] = {}
    for _, meta in repos:
        lang = meta.get("language") or "Other"
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    top_langs    = sorted(lang_counts, key=lambda l: lang_counts[l], reverse=True)[:LANGUAGE_LAYERS]
    top_lang_set = set(top_langs)

    lang_buckets:  dict[str, list] = {lang: [] for lang in top_langs}
    lang_other_bucket: list        = []
    stars_buckets: dict[str, list] = {tier: [] for tier in STARS_TIERS}

    for pid, meta in repos:
        lang  = meta.get("language") or "Other"
        stars = int(meta.get("stars", 0))
        if lang in top_lang_set:
            lang_buckets[lang].append((pid, meta))
        else:
            lang_other_bucket.append((pid, meta))
        stars_buckets[_stars_tier(stars)].append((pid, meta))

    used_pids: set[str]                 = set()
    selected:  list[tuple[str, dict, str]] = []

    def _pick(bucket, quota, label):
        random.shuffle(bucket)
        count = 0
        for pid, meta in bucket:
            if pid in used_pids:
                continue
            used_pids.add(pid)
            selected.append((pid, meta, label))
            count += 1
            if count >= quota:
                break

    for lang in top_langs:
        _pick(lang_buckets[lang], PER_LANGUAGE_LAYER, f"lang_{lang}")
    if LANGUAGE_OTHER:
        _pick(lang_other_bucket, PER_LANGUAGE_LAYER, "lang_other")
    for tier in STARS_TIERS:
        _pick(stars_buckets[tier], PER_STARS_TIER, f"stars_{tier}")
    remaining = [(pid, meta) for pid, meta in repos if pid not in used_pids]
    _pick(remaining, RANDOM_LAYER, "random")

    print(f"Sampled {len(selected)} repos (target {TARGET_TOTAL})", flush=True)
    return selected


# ---------------------------------------------------------------------------
# Step 4: Find similar repos for each sampled repo
# ---------------------------------------------------------------------------

def _find_similar(
    source_name: str,
    matrix: np.ndarray,
    names: list[str],
) -> list[str]:
    """
    Return up to MAX_EXTRA_RELEVANT repo names with cosine similarity > SIMILARITY_THRESHOLD,
    excluding the source itself. Returns [] if fewer than MIN_EXTRA_RELEVANT qualify.
    """
    if source_name not in names:
        return []
    idx  = names.index(source_name)
    sims = matrix @ matrix[idx]          # dot product = cosine (rows are normalised)
    sims[idx] = -1.0                      # exclude self

    top_indices = np.argsort(sims)[::-1][:20]
    extras = [names[i] for i in top_indices if sims[i] > SIMILARITY_THRESHOLD]
    extras = extras[:MAX_EXTRA_RELEVANT]

    if len(extras) < MIN_EXTRA_RELEVANT:
        return []
    return extras


# ---------------------------------------------------------------------------
# Step 5: Generate query via LLM
# ---------------------------------------------------------------------------

QUERY_TYPE_CYCLE = ["semantic", "keyword"]

PROMPT_TEMPLATE = """\
You are a GitHub user. Based on the following open-source repository information, generate a natural language search query in English.
Requirements:
1. Query type: {query_type} (semantic: e.g. "a Python library for doing X" / keyword: e.g. "project that supports Z feature")
2. Do NOT include the repository name or organization name
3. The query should be something a real user would type, under 20 words
4. Output only the query text, no explanation

Repository info:
Description: {description}
README snippet: {readme_snippet}"""


def _generate_query(client: OpenAI, chunk: dict, query_type: str) -> str:
    description    = (chunk.get("description") or "(no description)").strip()
    readme_snippet = _clean_intro((chunk.get("content") or "")[:300])
    prompt = PROMPT_TEMPLATE.format(
        query_type=query_type,
        description=description,
        readme_snippet=readme_snippet or "(no readme)",
    )
    resp = client.chat.completions.create(
        model=os.environ["LLM_MODEL_ID"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(42)

    print("Loading parent_chunks.json...", flush=True)
    repo_index = _load_repo_index()
    print(f"Total unique repos: {len(repo_index)}", flush=True)

    client = _make_client()

    print("Embedding all repos (or loading cache)...", flush=True)
    matrix, names = _embed_all_repos(client, repo_index)

    sampled = _stratified_sample(repo_index)

    testset = []
    for i, (full_name, chunk, stratum) in enumerate(sampled):
        query_type = QUERY_TYPE_CYCLE[i % 2]

        extras = _find_similar(full_name, matrix, names)
        relevant_repo_ids = [full_name] + extras

        print(f"[{i+1}/{len(sampled)}] {full_name} ({query_type}) | extras={len(extras)}", flush=True)
        try:
            query = _generate_query(client, chunk, query_type)
        except Exception as e:
            print(f"  WARNING: query generation failed for {full_name}: {e}", flush=True)
            continue

        stars = int(chunk.get("stars", 0))
        testset.append({
            "query":             query,
            "relevant_repo_ids": relevant_repo_ids,
            "query_type":        query_type,
            "meta": {
                "language":    chunk.get("language") or "Other",
                "stars_tier":  _stars_tier(stars),
                "stratum":     stratum,
                "source_repo": full_name,
            },
        })

    OUT_PATH.write_text(json.dumps(testset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone. {len(testset)} queries written to {OUT_PATH}", flush=True)

    # Print distribution summary
    extras_counts = [len(t["relevant_repo_ids"]) - 1 for t in testset]
    has_extras = sum(1 for c in extras_counts if c > 0)
    print(f"Queries with ≥1 extra relevant: {has_extras}/{len(testset)} "
          f"(avg extras per query: {sum(extras_counts)/max(len(extras_counts),1):.2f})", flush=True)
    print("Next step: git add eval/testset.json && git commit -m 'data: regenerate eval testset v2'")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify syntax**

```bash
python -c "import ast; ast.parse(open('eval/generate_testset.py').read()); print('syntax OK')"
```

Expected: `syntax OK`

- [ ] **Step 4: Commit**

```bash
git add eval/generate_testset.py .gitignore
git commit -m "feat(eval): rewrite generate_testset.py for parent-chunk arch with embedding-based multi-repo relevant"
```

---

## Task 3: Rewrite `evaluate.py`

**Files:**
- Rewrite: `eval/evaluate.py`

- [ ] **Step 1: Rewrite `eval/evaluate.py`**

```python
"""
Repeatable evaluation script (v2). Run from project root:
    python eval/evaluate.py

Reads:  eval/testset.json, backend/parent_chunks.json, backend/bm25_index.pkl
Writes: eval/report.json
Env:    PINECONE_API_KEY, PINECONE_INDEX_NAME, LLM_API_KEY, LLM_API_URL, LLM_MODEL_ID
"""
import json, os, sys, pathlib
from datetime import date
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone

ROOT = pathlib.Path(__file__).parent.parent
load_dotenv(ROOT / "backend" / ".env")
sys.path.insert(0, str(ROOT / "backend"))

from retriever import load_retriever

TESTSET_PATH = ROOT / "eval" / "testset.json"
REPORT_PATH  = ROOT / "eval" / "report.json"

LOW_SCORE_SAMPLE_COUNT = 50
SAMPLE_SIZE = None  # set to int to run a subset


# ---------------------------------------------------------------------------
# Pinecone init
# ---------------------------------------------------------------------------

def _init_pinecone():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    return pc.Index(os.environ["PINECONE_INDEX_NAME"])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _precision(retrieved: list[str], relevant: set[str]) -> float:
    if not retrieved:
        return 0.0
    return sum(1 for r in retrieved if r in relevant) / len(retrieved)


def _recall(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    return sum(1 for r in retrieved if r in relevant) / len(relevant)


def _mrr(retrieved: list[str], relevant: set[str]) -> float:
    for rank, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

def _make_glm_client() -> OpenAI:
    api_url  = os.environ["LLM_API_URL"]
    base_url = api_url.rsplit("/chat/completions", 1)[0] if "/chat/completions" in api_url else api_url
    return OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=base_url)


JUDGE_PROMPT = """\
User query: {query}
Repository: {repo_full_name}
Description: {description}
README snippet: {readme_snippet}

Is this repository relevant to the user query?
Reply with only 0 (not relevant) or 1 (relevant), no explanation."""


def _judge(client: OpenAI, query: str, node_metadata: dict) -> int | None:
    prompt = JUDGE_PROMPT.format(
        query=query,
        repo_full_name=node_metadata.get("full_name", ""),
        description=node_metadata.get("description", ""),
        readme_snippet=(node_metadata.get("content") or "")[:300],
    )
    try:
        resp = client.chat.completions.create(
            model=os.environ["LLM_MODEL_ID"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("1"):
            return 1
        if text.startswith("0"):
            return 0
        return None
    except Exception as e:
        print(f"    judge error: {e}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main():
    print("Loading testset...", flush=True)
    with open(TESTSET_PATH, encoding="utf-8") as f:
        testset = json.load(f)
    if SAMPLE_SIZE is not None:
        testset = testset[:SAMPLE_SIZE]
    print(f"Testset: {len(testset)} queries", flush=True)

    print("Loading retriever...", flush=True)
    pinecone_index = _init_pinecone()
    retriever      = load_retriever(pinecone_index)
    glm_client     = _make_glm_client()

    per_query    = []
    total_contrib = {"bm25_only": 0, "vector_only": 0, "both": 0, "unknown": 0}

    for i, item in enumerate(testset):
        query      = item["query"]
        relevant   = set(item["relevant_repo_ids"])
        query_type = item.get("query_type", "unknown")
        meta_info  = item.get("meta", {})

        print(f"[{i+1}/{len(testset)}] {query[:60]}", flush=True)

        nodes = retriever.retrieve(query)
        debug = retriever.last_debug

        # Dedup by full_name (same repo may appear as multiple sections), preserve rank order
        retrieved_repos = list(dict.fromkeys(
            n.metadata["full_name"] for n in nodes
        ))
        # Keep one node per repo for LLM judge (highest score = first occurrence after dedup)
        node_by_repo = {}
        for n in nodes:
            fn = n.metadata["full_name"]
            if fn not in node_by_repo:
                node_by_repo[fn] = n

        # Contribution labels (repo-level, using full candidate sets)
        vector_set = {pid.rsplit("__", 1)[0] for pid in debug["vector_candidate_ids"]}
        bm25_set   = {pid.rsplit("__", 1)[0] for pid in debug["bm25_candidate_ids"]}

        prec = _precision(retrieved_repos, relevant)
        rec  = _recall(retrieved_repos, relevant)
        mrr  = _mrr(retrieved_repos, relevant)

        retrieved_detail = []
        for repo in retrieved_repos:
            in_v = repo in vector_set
            in_b = repo in bm25_set
            label = "both" if in_v and in_b else ("vector_only" if in_v else ("bm25_only" if in_b else "unknown"))
            total_contrib[label] = total_contrib.get(label, 0) + 1

            if repo in relevant:
                judge_score = None
                is_hit = True
            else:
                node_meta   = node_by_repo[repo].metadata
                judge_score = _judge(glm_client, query, node_meta)
                is_hit      = False

            retrieved_detail.append({
                "repo":         repo,
                "ground_truth": is_hit,
                "llm_judge":    judge_score,
                "contribution": label,
            })

        judge_hits  = sum(1 for d in retrieved_detail if d["ground_truth"] or d["llm_judge"] == 1)
        judge_valid = sum(1 for d in retrieved_detail if d["ground_truth"] or d["llm_judge"] is not None)
        soft_prec   = judge_hits / judge_valid if judge_valid > 0 else 0.0

        per_query.append({
            "query":             query,
            "query_type":        query_type,
            "meta":              meta_info,
            "relevant_repo_ids": list(relevant),
            "retrieved":         retrieved_detail,
            "precision":         prec,
            "recall":            rec,
            "mrr":               mrr,
            "soft_precision":    soft_prec,
        })

    # ---------------------------------------------------------------------------
    # Aggregate
    # ---------------------------------------------------------------------------

    def _mean(key):
        vals = [q[key] for q in per_query]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def _mean_by(key, field, value):
        vals = [q[key] for q in per_query if q.get(field) == value]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    total_slots = sum(len(q["retrieved"]) for q in per_query)

    def _contrib_pct(label):
        return round(total_contrib.get(label, 0) / total_slots, 4) if total_slots > 0 else 0.0

    langs = {q["meta"].get("language") for q in per_query if q["meta"].get("language")}
    by_language = {}
    for lang in langs:
        subset = [q for q in per_query if q["meta"].get("language") == lang]
        by_language[lang] = {
            "precision_at_5": round(sum(q["precision"] for q in subset) / len(subset), 4),
            "recall_at_5":    round(sum(q["recall"]    for q in subset) / len(subset), 4),
            "mrr":            round(sum(q["mrr"]        for q in subset) / len(subset), 4),
            "count":          len(subset),
        }

    by_stars = {}
    for tier in ("low", "mid", "high"):
        subset = [q for q in per_query if q["meta"].get("stars_tier") == tier]
        if subset:
            by_stars[tier] = {
                "precision_at_5": round(sum(q["precision"] for q in subset) / len(subset), 4),
                "recall_at_5":    round(sum(q["recall"]    for q in subset) / len(subset), 4),
                "mrr":            round(sum(q["mrr"]        for q in subset) / len(subset), 4),
                "count":          len(subset),
            }

    low_samples = sorted(per_query, key=lambda q: q["soft_precision"])[:LOW_SCORE_SAMPLE_COUNT]

    report = {
        "date":          str(date.today()),
        "total_queries": len(per_query),
        "metrics": {
            "precision_at_5":      _mean("precision"),
            "recall_at_5":         _mean("recall"),
            "mrr":                 _mean("mrr"),
            "soft_precision_at_5": _mean("soft_precision"),
        },
        "contribution": {
            "bm25_only_pct":    _contrib_pct("bm25_only"),
            "vector_only_pct":  _contrib_pct("vector_only"),
            "both_pct":         _contrib_pct("both"),
            "unknown_pct":      _contrib_pct("unknown"),
        },
        "by_query_type": {
            qt: {
                "precision_at_5": _mean_by("precision", "query_type", qt),
                "recall_at_5":    _mean_by("recall",    "query_type", qt),
                "mrr":            _mean_by("mrr",       "query_type", qt),
            }
            for qt in ("semantic", "keyword")
        },
        "by_language":   by_language,
        "by_stars_tier": by_stars,
        "low_score_samples": low_samples,
    }

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    m  = report["metrics"]
    c  = report["contribution"]
    bt = report["by_query_type"]
    print(f"""
=== RAG Evaluation Report ===
Date: {report['date']}
Total queries: {report['total_queries']}

--- Core Metrics ---
Precision@5 (hard):  {m['precision_at_5']}
Recall@5:            {m['recall_at_5']}
MRR:                 {m['mrr']}

--- Soft Metrics (LLM judge) ---
Soft Precision@5:    {m['soft_precision_at_5']}

--- Retrieval Source Contribution ---
BM25 only:           {c['bm25_only_pct']:.1%}
Vector only:         {c['vector_only_pct']:.1%}
Both:                {c['both_pct']:.1%}
Unknown:             {c['unknown_pct']:.1%}

--- Query Type Breakdown ---
Semantic queries:    Precision={bt['semantic']['precision_at_5']}  Recall={bt['semantic']['recall_at_5']}  MRR={bt['semantic']['mrr']}
Keyword  queries:    Precision={bt['keyword']['precision_at_5']}  Recall={bt['keyword']['recall_at_5']}  MRR={bt['keyword']['mrr']}

Full report saved to: eval/report.json
Low-score samples ({LOW_SCORE_SAMPLE_COUNT}): see report.json → low_score_samples
""")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('eval/evaluate.py').read()); print('syntax OK')"
```

Expected: `syntax OK`

- [ ] **Step 3: Commit**

```bash
git add eval/evaluate.py
git commit -m "feat(eval): rewrite evaluate.py for parent-chunk arch with repo-level dedup and real Recall"
```

---

## Task 4: Run `generate_testset.py` and commit new testset

This task requires live API credentials (`LLM_API_KEY`, `LLM_API_URL`, `LLM_MODEL_ID` in `backend/.env`). Pinecone credentials are NOT needed here — `generate_testset.py` only calls the embedding/LLM API.

- [ ] **Step 1: Back up old testset**

```bash
cp eval/testset.json eval/testset_v1.json
```

- [ ] **Step 2: Run generate_testset.py**

```bash
python eval/generate_testset.py
```

Watch for:
- Embedding phase: `Embedded 100/15921 ... Embedded 15921/15921`
- Sampling phase: `Sampled 400 repos`
- Query generation: `[1/400] owner/repo (semantic) | extras=N`
- Final summary: extras distribution

If the final line shows `Queries with ≥1 extra relevant: X/400` with X < 100, the threshold 0.75 is too strict — lower it to 0.70 in `SIMILARITY_THRESHOLD` and re-run.

- [ ] **Step 3: Inspect a sample of the output**

```bash
python -c "
import json
with open('eval/testset.json') as f:
    ts = json.load(f)
print(f'Total: {len(ts)}')
multi = [t for t in ts if len(t['relevant_repo_ids']) > 1]
print(f'With multiple relevant: {len(multi)}')
print('Sample entry:')
import random; random.seed(0)
print(json.dumps(random.choice(multi), indent=2, ensure_ascii=False))
"
```

Expected: total 400, multiple relevant count > 0, sample shows 2+ relevant_repo_ids.

- [ ] **Step 4: Commit testset (delete the v1 backup)**

```bash
git add eval/testset.json
git rm --cached eval/testset_v1.json 2>/dev/null; rm -f eval/testset_v1.json
git commit -m "data: regenerate eval testset v2 with multi-repo relevant and parent-chunk arch"
```

---

## Task 5: Smoke-test `evaluate.py` on 5 queries

This verifies the evaluation pipeline end-to-end before running the full 400-query eval. Requires live API credentials.

- [ ] **Step 1: Run on 5 queries**

Temporarily set `SAMPLE_SIZE = 5` at the top of `eval/evaluate.py`, run, then reset to `None`:

```bash
python -c "
import subprocess, sys
# patch SAMPLE_SIZE=5 for smoke test
content = open('eval/evaluate.py').read()
patched = content.replace('SAMPLE_SIZE = None', 'SAMPLE_SIZE = 5')
open('eval/evaluate.py', 'w').write(patched)
"
python eval/evaluate.py
python -c "
content = open('eval/evaluate.py').read()
restored = content.replace('SAMPLE_SIZE = 5', 'SAMPLE_SIZE = None')
open('eval/evaluate.py', 'w').write(restored)
"
```

Expected: runs without error, prints metrics summary, `eval/report.json` created.

- [ ] **Step 2: Verify report.json structure**

```bash
python -c "
import json
r = json.load(open('eval/report.json'))
print('Keys:', list(r.keys()))
print('Metrics:', r['metrics'])
print('Contribution:', r['contribution'])
assert 'recall_at_5' in r['metrics'], 'recall missing'
assert 'vector_only_pct' in r['contribution'], 'contribution missing'
print('Structure OK')
"
```

Expected: `Structure OK`, `recall_at_5` present, `vector_only_pct` present (replacing old `pinecone_only_pct`).

- [ ] **Step 3: Commit smoke-test cleanup (if any file changed)**

```bash
git status
# if only report.json is dirty (it's gitignored), nothing to commit
# if evaluate.py was accidentally left patched, restore and commit
```
