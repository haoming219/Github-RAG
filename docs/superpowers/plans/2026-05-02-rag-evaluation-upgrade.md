# RAG Evaluation Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the RAG evaluation pipeline from repo-level ground truth to parent-chunk-level LLM-annotated ground truth, split into three independent scripts: `generate_testset.py`, `annotate_groundtruth.py`, and `evaluate.py`.

**Architecture:** Three-script pipeline where `generate_testset.py` produces queries-only (`testset_queries.json`), `annotate_groundtruth.py` annotates same-repo parent chunks via LLM per-chunk scoring to produce `testset.json`, and `evaluate.py` drives the real `CustomRetriever` against that testset to produce `report.json`. Each script is independently re-runnable; only `testset.json` and `report.json` are committed.

**Tech Stack:** Python 3.11, openai SDK, pinecone-client, LlamaIndex (`CustomRetriever`, `load_retriever`), python-dotenv, rank-bm25.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `eval/generate_testset.py` | Modify | Strip embedding/similarity logic; output `testset_queries.json` with `source_chunk_id` |
| `eval/annotate_groundtruth.py` | Create | LLM per-chunk 0/1 scoring against same-repo chunks; idempotent resume |
| `eval/evaluate.py` | Modify | Replace old `hybrid_search` interface with `CustomRetriever`; chunk-level relevant set; rename `pinecone_only` → `vector_only` |
| `eval/testset_queries.json` | Generated | Intermediate, not committed |
| `eval/testset.json` | Generated | Committed after annotation |
| `eval/report.json` | Generated | Committed after evaluation |

---

## Task 1: Rewrite `generate_testset.py`

**Files:**
- Modify: `eval/generate_testset.py`

Remove all embedding/similarity code. Keep stratified sampling and query generation. Output to `testset_queries.json` with `source_chunk_id` in `meta`.

- [ ] **Step 1: Write the complete new `generate_testset.py`**

Replace the entire file with:

```python
"""
Test set query generator (v3). Run from project root:
    python eval/generate_testset.py

Reads:  backend/parent_chunks.json
Writes: eval/testset_queries.json  (intermediate — do NOT commit)
Env:    LLM_API_KEY, LLM_API_URL, LLM_MODEL_ID
"""
import json, os, re, random, pathlib
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).parent.parent / "backend" / ".env")

BACKEND  = pathlib.Path(__file__).parent.parent / "backend"
EVAL_DIR = pathlib.Path(__file__).parent
OUT_PATH = EVAL_DIR / "testset_queries.json"

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


def _make_client() -> OpenAI:
    api_url = os.environ["LLM_API_URL"]
    base_url = api_url.rsplit("/chat/completions", 1)[0] if "/chat/completions" in api_url else api_url
    return OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=base_url)


def _stars_tier(stars: int) -> str:
    for tier, (lo, hi) in STARS_TIERS.items():
        if lo <= stars <= hi:
            return tier
    return "low"


def _clean_intro(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    return text.replace("&nbsp;", " ").strip()


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


def _stratified_sample(repo_index: dict) -> list[tuple[str, dict, str]]:
    repos = list(repo_index.items())

    lang_counts: dict[str, int] = {}
    for _, meta in repos:
        lang = meta.get("language") or "Other"
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    top_langs    = sorted(lang_counts, key=lambda l: lang_counts[l], reverse=True)[:LANGUAGE_LAYERS]
    top_lang_set = set(top_langs)

    lang_buckets:       dict[str, list] = {lang: [] for lang in top_langs}
    lang_other_bucket:  list            = []
    stars_buckets:      dict[str, list] = {tier: [] for tier in STARS_TIERS}

    for pid, meta in repos:
        lang  = meta.get("language") or "Other"
        stars = int(meta.get("stars", 0))
        if lang in top_lang_set:
            lang_buckets[lang].append((pid, meta))
        else:
            lang_other_bucket.append((pid, meta))
        stars_buckets[_stars_tier(stars)].append((pid, meta))

    used_pids: set[str]                    = set()
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


def main():
    random.seed(42)

    print("Loading parent_chunks.json...", flush=True)
    repo_index = _load_repo_index()
    print(f"Total unique repos: {len(repo_index)}", flush=True)

    client  = _make_client()
    sampled = _stratified_sample(repo_index)

    queries = []
    for i, (full_name, chunk, stratum) in enumerate(sampled):
        query_type      = QUERY_TYPE_CYCLE[i % 2]
        source_chunk_id = f"{full_name}__0"
        stars           = int(chunk.get("stars", 0))

        print(f"[{i+1}/{len(sampled)}] {full_name} ({query_type})", flush=True)
        try:
            query = _generate_query(client, chunk, query_type)
        except Exception as e:
            print(f"  WARNING: query generation failed for {full_name}: {e}", flush=True)
            continue

        queries.append({
            "query":      query,
            "query_type": query_type,
            "meta": {
                "source_repo":     full_name,
                "source_chunk_id": source_chunk_id,
                "language":        chunk.get("language") or "Other",
                "stars_tier":      _stars_tier(stars),
                "stratum":         stratum,
            },
        })

    OUT_PATH.write_text(json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone. {len(queries)} queries written to {OUT_PATH}", flush=True)
    print("Next step: python eval/annotate_groundtruth.py")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the file has no syntax errors**

```
python -c "import ast; ast.parse(open('eval/generate_testset.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add eval/generate_testset.py
git commit -m "refactor(eval): rewrite generate_testset.py — drop similarity logic, output testset_queries.json"
```

---

## Task 2: Create `annotate_groundtruth.py`

**Files:**
- Create: `eval/annotate_groundtruth.py`

New script. Reads `testset_queries.json` + `parent_chunks.json`, scores each same-repo chunk with LLM, writes `testset.json`. Idempotent: skips already-annotated queries on resume.

- [ ] **Step 1: Write `eval/annotate_groundtruth.py`**

```python
"""
Ground truth annotator. Run from project root:
    python eval/annotate_groundtruth.py

Reads:  eval/testset_queries.json, backend/parent_chunks.json
Writes: eval/testset.json          (commit this)
Env:    LLM_API_KEY, LLM_API_URL, LLM_MODEL_ID
"""
import json, os, pathlib
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).parent.parent / "backend" / ".env")

BACKEND        = pathlib.Path(__file__).parent.parent / "backend"
EVAL_DIR       = pathlib.Path(__file__).parent
QUERIES_PATH   = EVAL_DIR / "testset_queries.json"
TESTSET_PATH   = EVAL_DIR / "testset.json"

MIN_RELEVANT = 2

JUDGE_PROMPT = """\
User query: {query}
Repository: {full_name}
Section title: {section_title}
Content:
{content}

Is this repository section relevant to the user query?
Reply with only 0 (not relevant) or 1 (relevant), no explanation."""


def _make_client() -> OpenAI:
    api_url  = os.environ["LLM_API_URL"]
    base_url = api_url.rsplit("/chat/completions", 1)[0] if "/chat/completions" in api_url else api_url
    return OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=base_url)


def _score_chunk(client: OpenAI, query: str, chunk: dict) -> int | None:
    """Return 1 (relevant), 0 (not relevant), or None (parse/call failure)."""
    prompt = JUDGE_PROMPT.format(
        query=query,
        full_name=chunk["full_name"],
        section_title=chunk["section_title"],
        content=chunk["content"],
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
        print(f"    WARNING: unexpected LLM response '{text}' — treating as failure", flush=True)
        return None
    except Exception as e:
        print(f"    WARNING: LLM call failed: {e}", flush=True)
        return None


def main():
    print("Loading queries...", flush=True)
    with open(QUERIES_PATH, encoding="utf-8") as f:
        queries = json.load(f)
    print(f"Total queries: {len(queries)}", flush=True)

    print("Loading parent_chunks.json...", flush=True)
    with open(BACKEND / "parent_chunks.json", encoding="utf-8") as f:
        all_chunks = json.load(f)

    # Build lookup: full_name -> list of chunk dicts (all sections for that repo)
    repo_chunks: dict[str, list[dict]] = {}
    for chunk in all_chunks.values():
        fn = chunk["full_name"]
        repo_chunks.setdefault(fn, []).append(chunk)

    # Resume support: load already-annotated source_chunk_ids
    done_ids: set[str] = set()
    existing: list[dict] = []
    if TESTSET_PATH.exists():
        with open(TESTSET_PATH, encoding="utf-8") as f:
            existing = json.load(f)
        done_ids = {item["meta"]["source_chunk_id"] for item in existing}
        print(f"Resuming: {len(done_ids)} queries already annotated", flush=True)

    client   = _make_client()
    results  = list(existing)
    skipped  = 0
    discarded = 0

    for i, item in enumerate(queries):
        source_chunk_id = item["meta"]["source_chunk_id"]
        source_repo     = item["meta"]["source_repo"]
        query           = item["query"]

        if source_chunk_id in done_ids:
            skipped += 1
            continue

        candidates = repo_chunks.get(source_repo, [])
        if not candidates:
            print(f"[{i+1}/{len(queries)}] WARNING: no chunks found for {source_repo}", flush=True)
            discarded += 1
            continue

        print(f"[{i+1}/{len(queries)}] {source_repo} ({len(candidates)} chunks)", flush=True)

        relevant_chunk_ids = []
        for chunk in candidates:
            score = _score_chunk(client, query, chunk)
            if score == 1:
                relevant_chunk_ids.append(chunk["parent_id"])

        if len(relevant_chunk_ids) < MIN_RELEVANT:
            print(f"  → discarded (only {len(relevant_chunk_ids)} relevant chunk(s))", flush=True)
            discarded += 1
            continue

        results.append({
            "query":              query,
            "query_type":         item["query_type"],
            "relevant_chunk_ids": relevant_chunk_ids,
            "meta":               item["meta"],
        })

        # Write incrementally after each accepted entry
        TESTSET_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nDone. {len(results)} entries in testset.json "
          f"(skipped={skipped}, discarded={discarded})", flush=True)
    print("Next step: git add eval/testset.json && git commit -m 'data: annotate ground truth v3'")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

```
python -c "import ast; ast.parse(open('eval/annotate_groundtruth.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add eval/annotate_groundtruth.py
git commit -m "feat(eval): add annotate_groundtruth.py — LLM per-chunk ground truth annotation"
```

---

## Task 3: Rewrite `evaluate.py`

**Files:**
- Modify: `eval/evaluate.py`

Replace old `hybrid_search`/`_load_artifacts` interface with `CustomRetriever`. Change `relevant` from repo IDs to chunk IDs. Rename all `pinecone_only` → `vector_only`. Add `relevant_count_distribution` to report. Update the LLM judge prompt from repo-level context (`description` + `readme_snippet`) to chunk-level context (`section_title` + `content`), consistent with how `annotate_groundtruth.py` scores chunks.

- [ ] **Step 1: Write the complete new `evaluate.py`**

Replace the entire file with:

```python
"""
Repeatable evaluation script. Run from project root:
    python eval/evaluate.py

Reads:  eval/testset.json
Writes: eval/report.json
Env:    PINECONE_API_KEY, PINECONE_INDEX_NAME,
        LLM_API_KEY, LLM_API_URL, LLM_MODEL_ID
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
SAMPLE_SIZE = None  # set to an int to run on a subset


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

def _init_pinecone():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    return pc.Index(os.environ["PINECONE_INDEX_NAME"])


def _make_llm_client() -> OpenAI:
    api_url  = os.environ["LLM_API_URL"]
    base_url = api_url.rsplit("/chat/completions", 1)[0] if "/chat/completions" in api_url else api_url
    return OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=base_url)


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
# LLM judge (for non-ground-truth retrieved chunks)
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
User query: {query}
Repository: {repo_full_name}
Section title: {section_title}
Content:
{content}

Is this repository section relevant to the user query?
Reply with only 0 (not relevant) or 1 (relevant), no explanation."""


def _judge(client: OpenAI, query: str, meta: dict) -> int | None:
    prompt = JUDGE_PROMPT.format(
        query=query,
        repo_full_name=meta.get("full_name", ""),
        section_title=meta.get("section_title", ""),
        content=meta.get("content", ""),
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
# Contribution analysis
# ---------------------------------------------------------------------------

def _contribution_label(chunk_id: str, debug: dict) -> str:
    in_vector = chunk_id in set(debug["vector_candidate_ids"])
    in_bm25   = chunk_id in set(debug["bm25_candidate_ids"])
    if in_vector and in_bm25:
        return "both"
    if in_vector:
        return "vector_only"
    if in_bm25:
        return "bm25_only"
    return "unknown"


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

    print("Initialising retriever...", flush=True)
    pinecone_index = _init_pinecone()
    retriever      = load_retriever(pinecone_index)
    llm_client     = _make_llm_client()

    per_query     = []
    total_contrib = {"bm25_only": 0, "vector_only": 0, "both": 0, "unknown": 0}

    for i, item in enumerate(testset):
        query      = item["query"]
        relevant   = set(item["relevant_chunk_ids"])
        query_type = item.get("query_type", "unknown")
        meta_info  = item.get("meta", {})

        print(f"[{i+1}/{len(testset)}] {query[:60]}...", flush=True)

        # Retrieve with no filters (evaluation uses full index)
        retriever.language  = ""
        retriever.min_stars = 0
        retriever.topics    = []
        nodes = retriever.retrieve(query)

        retrieved_ids = [n.node.metadata["parent_id"] for n in nodes]
        # last_debug is only set after a successful retrieve(); guard against missing attribute
        debug = getattr(retriever, "last_debug", {"vector_candidate_ids": [], "bm25_candidate_ids": []})

        prec = _precision(retrieved_ids, relevant)
        rec  = _recall(retrieved_ids, relevant)
        mrr  = _mrr(retrieved_ids, relevant)

        retrieved_detail = []
        for n in nodes:
            chunk_id = n.node.metadata["parent_id"]
            meta     = n.node.metadata
            label    = _contribution_label(chunk_id, debug)
            total_contrib[label] = total_contrib.get(label, 0) + 1

            if chunk_id in relevant:
                judge_score = None  # relevance already known
                is_hit      = True
            else:
                judge_score = _judge(llm_client, query, meta)
                is_hit      = False

            retrieved_detail.append({
                "parent_id":    chunk_id,
                "description":  meta.get("description", ""),
                "ground_truth": is_hit,
                "llm_judge":    judge_score,
                "contribution": label,
            })

        # Soft precision
        judge_hits  = sum(1 for d in retrieved_detail if d["ground_truth"] or d["llm_judge"] == 1)
        judge_valid = sum(1 for d in retrieved_detail if d["ground_truth"] or d["llm_judge"] is not None)
        soft_prec   = judge_hits / judge_valid if judge_valid > 0 else 0.0

        per_query.append({
            "query":               query,
            "query_type":          query_type,
            "meta":                meta_info,
            "relevant_chunk_ids":  list(relevant),
            "retrieved":           retrieved_detail,
            "precision":           prec,
            "recall":              rec,
            "mrr":                 mrr,
            "soft_precision":      soft_prec,
        })

    # ---------------------------------------------------------------------------
    # Aggregation
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
    for lang in sorted(langs):
        subset = [q for q in per_query if q["meta"].get("language") == lang]
        by_language[lang] = {
            "precision_at_5": round(sum(q["precision"] for q in subset) / len(subset), 4),
            "mrr":            round(sum(q["mrr"]       for q in subset) / len(subset), 4),
            "count":          len(subset),
        }

    by_stars = {}
    for tier in ("low", "mid", "high"):
        subset = [q for q in per_query if q["meta"].get("stars_tier") == tier]
        if subset:
            by_stars[tier] = {
                "precision_at_5": round(sum(q["precision"] for q in subset) / len(subset), 4),
                "mrr":            round(sum(q["mrr"]       for q in subset) / len(subset), 4),
                "count":          len(subset),
            }

    # Relevant count distribution
    rel_dist: dict[str, int] = {}
    for q in per_query:
        n = len(q["relevant_chunk_ids"])
        key = str(n) if n < 5 else "5+"
        rel_dist[key] = rel_dist.get(key, 0) + 1

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
        "relevant_count_distribution": rel_dist,
        "contribution": {
            "bm25_only_pct":    _contrib_pct("bm25_only"),
            "vector_only_pct":  _contrib_pct("vector_only"),
            "both_pct":         _contrib_pct("both"),
            "unknown_pct":      _contrib_pct("unknown"),
        },
        "by_query_type": {
            qt: {
                "precision_at_5": _mean_by("precision", "query_type", qt),
                "mrr":            _mean_by("mrr", "query_type", qt),
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
BM25 only:     {c['bm25_only_pct']:.1%}
Vector only:   {c['vector_only_pct']:.1%}
Both:          {c['both_pct']:.1%}
Unknown:       {c['unknown_pct']:.1%}

--- Query Type Breakdown ---
Semantic:  Precision={bt['semantic']['precision_at_5']}  MRR={bt['semantic']['mrr']}
Keyword:   Precision={bt['keyword']['precision_at_5']}  MRR={bt['keyword']['mrr']}

Full report saved to: eval/report.json
Low-score samples ({LOW_SCORE_SAMPLE_COUNT}): see report.json → low_score_samples
""")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

```
python -c "import ast; ast.parse(open('eval/evaluate.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add eval/evaluate.py
git commit -m "refactor(eval): rewrite evaluate.py — CustomRetriever interface, chunk-level relevant, vector_only rename"
```

---

## Task 4: Update `.gitignore` and verify file exclusions

> **Complete this task before running `generate_testset.py` for the first time**, so that `testset_queries.json` is never accidentally staged.

**Files:**
- Modify: `.gitignore` (or create if absent)

`testset_queries.json` must not be committed. Check whether it is already excluded.

- [ ] **Step 1: Check if `testset_queries.json` is already gitignored**

```bash
git check-ignore -v eval/testset_queries.json
```

If no output (not ignored), proceed to Step 2. If already ignored, skip to Step 3.

- [ ] **Step 2: Add to `.gitignore` if not already excluded**

Add the line `eval/testset_queries.json` to `.gitignore`.

- [ ] **Step 3: Commit if changed**

```bash
git add .gitignore
git commit -m "chore: exclude eval/testset_queries.json from git"
```

---

## Task 5: Smoke-test the full pipeline end-to-end

This task verifies the three scripts are wired together correctly before running at full scale. It does **not** require actual Pinecone or LLM credentials — it only checks that the scripts parse and import cleanly, and that the output formats are correct on a tiny sample.

- [ ] **Step 1: Verify all three scripts parse without import errors**

```bash
cd d:/PythonProjects/TestProject
python -c "
import ast, pathlib
for p in ['eval/generate_testset.py', 'eval/annotate_groundtruth.py', 'eval/evaluate.py']:
    ast.parse(pathlib.Path(p).read_text(encoding='utf-8'))
    print(f'OK: {p}')
"
```

Expected: three `OK` lines, no exceptions.

- [ ] **Step 2: Verify `testset_queries.json` output schema (if you have run `generate_testset.py` previously)**

If `eval/testset_queries.json` exists, check the first entry:

```bash
python -c "
import json
data = json.load(open('eval/testset_queries.json', encoding='utf-8'))
item = data[0]
assert 'query' in item
assert 'query_type' in item
assert 'meta' in item
assert 'source_chunk_id' in item['meta']
assert 'source_repo' in item['meta']
print('Schema OK:', item['meta']['source_chunk_id'])
"
```

Expected: prints `Schema OK: <some_repo>__0`

- [ ] **Step 3: Verify `testset.json` output schema (if you have run `annotate_groundtruth.py` previously)**

If `eval/testset.json` exists after annotation, check the first entry:

```bash
python -c "
import json
data = json.load(open('eval/testset.json', encoding='utf-8'))
item = data[0]
assert 'query' in item
assert 'relevant_chunk_ids' in item
assert isinstance(item['relevant_chunk_ids'], list)
assert len(item['relevant_chunk_ids']) >= 2
assert 'source_chunk_id' in item['meta']
print('Schema OK:', item['relevant_chunk_ids'])
"
```

Expected: prints `Schema OK: [...]` with at least 2 chunk IDs.

- [ ] **Step 4: Commit smoke test results (none — no new files produced here)**

No commit needed for this task.

---

## Execution Order

When running the full pipeline for real:

```
1. python eval/generate_testset.py
   → produces eval/testset_queries.json

2. python eval/annotate_groundtruth.py
   → produces eval/testset.json  (resumable — safe to interrupt and restart)
   → git add eval/testset.json && git commit -m "data: annotate ground truth v3"

3. python eval/evaluate.py
   → produces eval/report.json
   → git add eval/report.json && git commit -m "data: evaluation report v3"
```
