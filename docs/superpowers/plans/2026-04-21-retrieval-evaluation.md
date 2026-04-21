# Retrieval Evaluation System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repeatable RAG retrieval evaluation tool that measures Precision@5, Recall@5, MRR, soft precision (LLM judge), and BM25 vs Pinecone contribution from a 400-query stratified test set.

**Architecture:** Two-phase: (1) one-time test set generation via LLM-assisted stratified sampling from `chunk_metadata.json`; (2) repeatable evaluation script that calls `hybrid_search()` with a new `return_debug=True` flag to expose intermediate ranked lists for contribution analysis. All artifacts live in `eval/`, reusing existing backend env vars and dependencies.

**Tech Stack:** Python, `openai` SDK (GLM-compatible), `pinecone`, `sentence-transformers`, `rank_bm25` — all already in `backend/requirements.txt`.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `backend/retriever.py` | Modify | Add `return_debug` parameter to `hybrid_search()` |
| `eval/__init__.py` | Create | Empty — makes `eval` a package so scripts can import from `backend/` |
| `eval/generate_testset.py` | Create | Stratified sampling + GLM query generation → `eval/testset.json` |
| `eval/evaluate.py` | Create | Load testset, run hybrid_search, compute metrics, LLM judge, output report |
| `eval/testset.json` | Generated | Fixed ground truth — commit to repo after generation |
| `eval/report.json` | Generated | Per-run output — do NOT commit (add to `.gitignore`) |
| `.gitignore` | Modify | Add `eval/report.json` |

---

## Task 1: Add `return_debug` to `hybrid_search()`

**Files:**
- Modify: `backend/retriever.py:41-115`

This is a pure additive change — new optional parameter with default `False`, no behavior change for existing callers.

- [ ] **Step 1.1: Replace the `fused` line and add `return_debug` parameter**

Open `backend/retriever.py`. Make these two targeted edits:

**Edit 1** — function signature (line 41-48): add `return_debug: bool = False` parameter and update return type hint:

```python
def hybrid_search(
    query: str,
    pinecone_index,
    language: str = "",
    min_stars: int = 0,
    topics: list[str] = None,
    top_k: int = 5,
    return_debug: bool = False
) -> "list[dict] | tuple[list[dict], dict]":
```

**Edit 2** — replace **only line 100** (`fused = _rrf(ranked_lists)[:top_k]`) with the following two lines. Keep lines 97–99 (the early-return guard `if not pinecone_ranked and not bm25_ranked: return []`) **exactly as-is** — do NOT remove them:

```python
    fused_all = _rrf(ranked_lists)
    fused = fused_all[:top_k]
```

Then append the debug block **after** the existing `results = []` deduplication loop (after line 113 `break`), just before the final `return results`:

```python
    if return_debug:
        debug_info = {
            "pinecone_candidate_ids": [_chunk_metadata[i]["parent_id"] for i in pinecone_ranked],
            "bm25_candidate_ids": [_chunk_metadata[i]["parent_id"] for i in bm25_ranked],
            "fused_ranked_ids": [_chunk_metadata[i]["parent_id"] for i in fused_all[:top_k * 3]],
        }
        return results, debug_info
    return results
```

The final `hybrid_search()` structure should look like this (early-return guard preserved):

```python
    # --- RRF fusion ---
    ranked_lists = [pinecone_ranked]
    if bm25_ranked:
        ranked_lists.append(bm25_ranked)
    # If both empty, return empty
    if not pinecone_ranked and not bm25_ranked:
        return []

    fused_all = _rrf(ranked_lists)
    fused = fused_all[:top_k]

    results = []
    seen_parents = set()
    for idx in fused:
        meta = _chunk_metadata[idx]
        parent_id = meta["parent_id"]
        if parent_id in seen_parents:
            continue
        seen_parents.add(parent_id)
        results.append(meta)
        if len(results) >= top_k:
            break

    if return_debug:
        debug_info = {
            "pinecone_candidate_ids": [_chunk_metadata[i]["parent_id"] for i in pinecone_ranked],
            "bm25_candidate_ids": [_chunk_metadata[i]["parent_id"] for i in bm25_ranked],
            "fused_ranked_ids": [_chunk_metadata[i]["parent_id"] for i in fused_all[:top_k * 3]],
        }
        return results, debug_info
    return results
```

- [ ] **Step 1.2: Verify existing callers are unaffected**

Run a quick grep to confirm no existing caller passes positional args beyond `top_k`:

```bash
grep -n "hybrid_search(" backend/main.py
```

Expected: one call site at line ~75, using keyword args only. `return_debug` defaults to `False` so no change needed there.

- [ ] **Step 1.3: Smoke-test the change manually**

```bash
cd d:/PythonProjects/TestProject/backend
python -c "
import os; os.environ.setdefault('PINECONE_API_KEY','x'); os.environ.setdefault('PINECONE_INDEX_NAME','x')
from retriever import hybrid_search
import inspect
sig = inspect.signature(hybrid_search)
assert 'return_debug' in sig.parameters, 'parameter missing'
import ast, pathlib
src = pathlib.Path('retriever.py').read_text(encoding='utf-8')
assert 'if not pinecone_ranked and not bm25_ranked' in src, 'early-return guard missing!'
print('OK: return_debug parameter present, early-return guard preserved')
"
```

Expected output: `OK: return_debug parameter present, early-return guard preserved`

- [ ] **Step 1.4: Commit**

```bash
git add backend/retriever.py
git commit -m "feat(eval): add return_debug param to hybrid_search for contribution analysis"
```

---

## Task 2: Scaffold `eval/` directory

**Files:**
- Create: `eval/__init__.py`
- Modify: `.gitignore`

- [ ] **Step 2.1: Create `eval/__init__.py`**

Create an empty file:

```bash
mkdir eval
touch eval/__init__.py
```

- [ ] **Step 2.2: Add `eval/report.json` to `.gitignore`**

Append to the project root `.gitignore` (create if it doesn't exist):

```
eval/report.json
```

- [ ] **Step 2.3: Commit**

```bash
git add eval/__init__.py .gitignore
git commit -m "chore(eval): scaffold eval/ directory"
```

---

## Task 3: Build `generate_testset.py`

**Files:**
- Create: `eval/generate_testset.py`

This script runs once, writes `eval/testset.json`, and exits. It reads from `backend/chunk_metadata.json` and calls the GLM API for each sampled repo.

- [ ] **Step 3.1: Create `eval/generate_testset.py`**

```python
"""
One-time test set generator. Run from project root:
    python eval/generate_testset.py

Reads:  backend/chunk_metadata.json
Writes: eval/testset.json  (commit this file to repo)
Env:    GLM_API_KEY, GLM_API_URL, GLM_MODEL_ID
"""
import json, os, random, sys, pathlib
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).parent.parent / "backend" / ".env")

BACKEND = pathlib.Path(__file__).parent.parent / "backend"
OUT_PATH = pathlib.Path(__file__).parent / "testset.json"

# Quota configuration
LANGUAGE_LAYERS = 5          # top-N languages get their own layer
LANGUAGE_OTHER_LAYER = True  # one extra "other" layer
PER_LANGUAGE_LAYER = 30      # repos per language layer  (6 layers × 30 = 180)
PER_STARS_TIER = 20          # repos per stars tier      (3 tiers  × 20 = 60)
RANDOM_LAYER = 160           # purely random repos
TARGET_TOTAL = 400

STARS_TIERS = {
    "low":  (0,     999),
    "mid":  (1000,  9999),
    "high": (10000, float("inf")),
}


def _load_chunk_metadata() -> list[dict]:
    with open(BACKEND / "chunk_metadata.json", encoding="utf-8") as f:
        return json.load(f)


def _build_parent_index(chunks: list[dict]) -> dict:
    """Build {parent_id: {chunk_index: chunk_entry}} for O(1) lookup."""
    idx: dict = {}
    for entry in chunks:
        pid = entry["parent_id"]
        ci = entry.get("chunk_index", 0)
        idx.setdefault(pid, {})[ci] = entry
    return idx


def _unique_parents(chunks: list[dict]) -> dict:
    """Return {parent_id: first_chunk_with_parent_fields} — one entry per repo."""
    seen = {}
    for entry in chunks:
        pid = entry["parent_id"]
        if pid not in seen:
            seen[pid] = entry
    return seen


def _stars_tier(stars: int) -> str:
    for tier, (lo, hi) in STARS_TIERS.items():
        if lo <= stars <= hi:
            return tier
    return "low"


def _stratified_sample(all_parents: dict) -> list[tuple[str, dict, str]]:
    """
    Returns list of (parent_id, parent_meta, stratum_label).
    Stratum labels: "lang_{language}", "lang_other", "stars_{tier}", "random"
    Priority for dedup: language > stars > random
    """
    repos = list(all_parents.items())  # [(parent_id, meta), ...]

    # --- Count languages, pick top-N ---
    lang_counts: dict[str, int] = {}
    for _, meta in repos:
        lang = meta.get("language") or "Other"
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    top_langs = sorted(lang_counts, key=lambda l: lang_counts[l], reverse=True)[:LANGUAGE_LAYERS]
    top_lang_set = set(top_langs)

    # Buckets
    lang_buckets: dict[str, list] = {lang: [] for lang in top_langs}
    lang_other_bucket: list = []
    stars_buckets: dict[str, list] = {tier: [] for tier in STARS_TIERS}

    for pid, meta in repos:
        lang = meta.get("language") or "Other"
        stars = int(meta.get("stars", 0))
        if lang in top_lang_set:
            lang_buckets[lang].append((pid, meta))
        else:
            lang_other_bucket.append((pid, meta))
        stars_buckets[_stars_tier(stars)].append((pid, meta))

    used_pids: set[str] = set()
    selected: list[tuple[str, dict, str]] = []

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

    # Language layers (highest priority)
    for lang in top_langs:
        _pick(lang_buckets[lang], PER_LANGUAGE_LAYER, f"lang_{lang}")
    if LANGUAGE_OTHER_LAYER:
        _pick(lang_other_bucket, PER_LANGUAGE_LAYER, "lang_other")

    # Stars layers
    for tier in STARS_TIERS:
        _pick(stars_buckets[tier], PER_STARS_TIER, f"stars_{tier}")

    # Random layer
    all_remaining = [(pid, meta) for pid, meta in repos if pid not in used_pids]
    _pick(all_remaining, RANDOM_LAYER, "random")

    print(f"Sampled {len(selected)} repos (target {TARGET_TOTAL})", flush=True)
    return selected


def _get_chunk0_content(parent_id: str, parent_idx: dict) -> str:
    chunks_for_repo = parent_idx.get(parent_id, {})
    chunk0 = chunks_for_repo.get(0)
    if chunk0:
        return chunk0.get("content", "")[:500]
    return ""


def _make_glm_client() -> OpenAI:
    api_url = os.environ["GLM_API_URL"]
    # GLM_API_URL may be the full endpoint (e.g. https://aihubmix.com/v1/chat/completions)
    # or already a base URL — handle both to avoid double-appending /chat/completions
    base_url = api_url.rsplit("/chat/completions", 1)[0] if "/chat/completions" in api_url else api_url
    return OpenAI(api_key=os.environ["GLM_API_KEY"], base_url=base_url)


QUERY_TYPE_CYCLE = ["semantic", "keyword"]

PROMPT_TEMPLATE = """\
你是一个 GitHub 用户。根据以下开源仓库的信息，生成一条自然语言搜索查询。
要求：
1. 查询类型：{query_type}（语义型：用"我想找一个做X的Y语言库" / 关键词型：用"有没有支持Z feature的项目"）
2. 不能出现仓库名称或组织名称
3. 查询应是用户会真实输入的问题，50字以内
4. 只输出查询文本，不要任何解释

仓库信息：
Description: {description}
README 摘要: {readme_snippet}"""


def _generate_query(client: OpenAI, description: str, readme_snippet: str, query_type: str) -> str:
    prompt = PROMPT_TEMPLATE.format(
        query_type=query_type,
        description=description or "(no description)",
        readme_snippet=readme_snippet or "(no readme)",
    )
    resp = client.chat.completions.create(
        model=os.environ["GLM_MODEL_ID"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


def main():
    random.seed(42)
    print("Loading chunk_metadata.json...", flush=True)
    chunks = _load_chunk_metadata()
    parent_idx = _build_parent_index(chunks)
    all_parents = _unique_parents(chunks)
    print(f"Total unique repos: {len(all_parents)}", flush=True)

    sampled = _stratified_sample(all_parents)
    client = _make_glm_client()

    testset = []
    for i, (pid, meta, stratum) in enumerate(sampled):
        query_type = QUERY_TYPE_CYCLE[i % 2]
        readme_snippet = _get_chunk0_content(pid, parent_idx)

        print(f"[{i+1}/{len(sampled)}] Generating query for {pid} ({query_type})...", flush=True)
        try:
            query = _generate_query(
                client,
                description=meta.get("description", ""),
                readme_snippet=readme_snippet,
                query_type=query_type,
            )
        except Exception as e:
            print(f"  WARNING: GLM call failed for {pid}: {e}", flush=True)
            continue

        stars = int(meta.get("stars", 0))
        testset.append({
            "query": query,
            "relevant_repo_ids": [pid],
            "query_type": query_type,
            "meta": {
                "language": meta.get("language") or "Other",
                "stars_tier": _stars_tier(stars),
                "stratum": stratum,
                "source_repo": pid,
            },
        })

    OUT_PATH.write_text(json.dumps(testset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone. {len(testset)} queries written to {OUT_PATH}", flush=True)
    print("Next step: git add eval/testset.json && git commit -m 'data: add eval testset'")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.2: Dry-run sampling logic (without GLM calls)**

Verify sampling works on the real data by patching `_generate_query` to skip API calls:

```bash
cd d:/PythonProjects/TestProject
python -c "
import sys; sys.path.insert(0, 'backend')
import json, random, pathlib
random.seed(42)

exec(open('eval/generate_testset.py').read().split('def main')[0])  # load helpers only

chunks = _load_chunk_metadata()
parent_idx = _build_parent_index(chunks)
all_parents = _unique_parents(chunks)
sampled = _stratified_sample(all_parents)

strata = {}
for pid, meta, stratum in sampled:
    strata[stratum] = strata.get(stratum, 0) + 1
for k, v in sorted(strata.items()):
    print(f'  {k}: {v}')
print(f'Total: {len(sampled)}')
"
```

Expected: each `lang_*` stratum shows ~30, each `stars_*` shows ~20, `random` shows ~160, total ~400.

- [ ] **Step 3.3: Commit `generate_testset.py`**

```bash
git add eval/generate_testset.py
git commit -m "feat(eval): add generate_testset.py — stratified LLM query generator"
```

---

## Task 4: Build `evaluate.py`

**Files:**
- Create: `eval/evaluate.py`

- [ ] **Step 4.1: Create `eval/evaluate.py`**

```python
"""
Repeatable evaluation script. Run from project root:
    python eval/evaluate.py

Reads:  eval/testset.json, backend/chunk_metadata.json, backend/bm25_index.pkl
Writes: eval/report.json
Env:    PINECONE_API_KEY, PINECONE_INDEX_NAME, GLM_API_KEY, GLM_API_URL, GLM_MODEL_ID
"""
import json, os, sys, pathlib
from datetime import date
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone

ROOT = pathlib.Path(__file__).parent.parent
load_dotenv(ROOT / "backend" / ".env")
sys.path.insert(0, str(ROOT / "backend"))

from retriever import hybrid_search, _load_artifacts

TESTSET_PATH = ROOT / "eval" / "testset.json"
REPORT_PATH  = ROOT / "eval" / "report.json"
BACKEND      = ROOT / "backend"

LOW_SCORE_SAMPLE_COUNT = 50


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
    hits = sum(1 for r in retrieved if r in relevant)
    return hits / len(retrieved)


def _recall(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved if r in relevant)
    return hits / len(relevant)


def _mrr(retrieved: list[str], relevant: set[str]) -> float:
    for rank, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

def _make_glm_client() -> OpenAI:
    api_url = os.environ["GLM_API_URL"]
    # GLM_API_URL may be the full endpoint (e.g. https://aihubmix.com/v1/chat/completions)
    # or already a base URL — handle both to avoid double-appending /chat/completions
    base_url = api_url.rsplit("/chat/completions", 1)[0] if "/chat/completions" in api_url else api_url
    return OpenAI(api_key=os.environ["GLM_API_KEY"], base_url=base_url)


JUDGE_PROMPT = """\
用户查询：{query}
仓库名称：{repo_full_name}
仓库描述：{description}
README 摘要：{readme_snippet}

该仓库是否与用户查询相关？
请只回答 0（不相关）或 1（相关），不要解释。"""


def _judge(client: OpenAI, query: str, meta: dict) -> int | None:
    readme_snippet = meta.get("content", "")[:300]
    prompt = JUDGE_PROMPT.format(
        query=query,
        repo_full_name=meta.get("full_name") or meta.get("parent_id", ""),
        description=meta.get("description", ""),
        readme_snippet=readme_snippet,
    )
    try:
        resp = client.chat.completions.create(
            model=os.environ["GLM_MODEL_ID"],
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

def _contribution_label(repo_id: str, debug: dict) -> str:
    in_pc  = repo_id in set(debug["pinecone_candidate_ids"])
    in_bm25 = repo_id in set(debug["bm25_candidate_ids"])
    if in_pc and in_bm25:
        return "both"
    if in_pc:
        return "pinecone_only"
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
    print(f"Testset: {len(testset)} queries", flush=True)

    print("Loading backend artifacts...", flush=True)
    _load_artifacts()
    pinecone_index = _init_pinecone()
    glm_client = _make_glm_client()

    # Per-query results
    per_query = []

    # Aggregate counters
    total_contrib = {"bm25_only": 0, "pinecone_only": 0, "both": 0, "unknown": 0}

    for i, item in enumerate(testset):
        query       = item["query"]
        relevant    = set(item["relevant_repo_ids"])
        query_type  = item.get("query_type", "unknown")
        meta_info   = item.get("meta", {})

        print(f"[{i+1}/{len(testset)}] {query[:60]}...", flush=True)

        results, debug = hybrid_search(
            query=query,
            pinecone_index=pinecone_index,
            return_debug=True,
        )

        retrieved_ids = [r["parent_id"] for r in results]

        prec  = _precision(retrieved_ids, relevant)
        rec   = _recall(retrieved_ids, relevant)
        mrr   = _mrr(retrieved_ids, relevant)

        # Contribution labels for each retrieved result
        retrieved_detail = []
        for r in results:
            rid = r["parent_id"]
            label = _contribution_label(rid, debug)
            total_contrib[label] = total_contrib.get(label, 0) + 1

            # LLM judge only for non-ground-truth results
            if rid in relevant:
                judge_score = None  # already known relevant
                is_hit = True
            else:
                judge_score = _judge(glm_client, query, r)
                is_hit = False

            retrieved_detail.append({
                "parent_id":    rid,
                "description":  r.get("description", ""),
                "ground_truth": is_hit,
                "llm_judge":    judge_score,
                "contribution": label,
            })

        # Soft precision: ground truth hits + judge=1 hits / len(retrieved)
        judge_hits = sum(
            1 for d in retrieved_detail
            if d["ground_truth"] or d["llm_judge"] == 1
        )
        judge_valid = sum(
            1 for d in retrieved_detail
            if d["ground_truth"] or d["llm_judge"] is not None
        )
        soft_prec = judge_hits / judge_valid if judge_valid > 0 else 0.0

        per_query.append({
            "query":            query,
            "query_type":       query_type,
            "meta":             meta_info,
            "relevant_repo_ids": list(relevant),
            "retrieved":        retrieved_detail,
            "precision":        prec,
            "recall":           rec,
            "mrr":              mrr,
            "soft_precision":   soft_prec,
        })

    # ---------------------------------------------------------------------------
    # Aggregate metrics
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

    # By language (from meta.language)
    langs = {q["meta"].get("language") for q in per_query if q["meta"].get("language")}
    by_language = {}
    for lang in langs:
        subset = [q for q in per_query if q["meta"].get("language") == lang]
        by_language[lang] = {
            "precision_at_5": round(sum(q["precision"] for q in subset) / len(subset), 4),
            "mrr":            round(sum(q["mrr"] for q in subset) / len(subset), 4),
            "count":          len(subset),
        }

    # By stars tier
    by_stars = {}
    for tier in ("low", "mid", "high"):
        subset = [q for q in per_query if q["meta"].get("stars_tier") == tier]
        if subset:
            by_stars[tier] = {
                "precision_at_5": round(sum(q["precision"] for q in subset) / len(subset), 4),
                "mrr":            round(sum(q["mrr"] for q in subset) / len(subset), 4),
                "count":          len(subset),
            }

    # Low-score samples for human review
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
            "bm25_only_pct":     _contrib_pct("bm25_only"),
            "pinecone_only_pct": _contrib_pct("pinecone_only"),
            "both_pct":          _contrib_pct("both"),
            "unknown_pct":       _contrib_pct("unknown"),
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

    # ---------------------------------------------------------------------------
    # Terminal summary
    # ---------------------------------------------------------------------------
    m = report["metrics"]
    c = report["contribution"]
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
Pinecone only:       {c['pinecone_only_pct']:.1%}
Both:                {c['both_pct']:.1%}
Unknown:             {c['unknown_pct']:.1%}

--- Query Type Breakdown ---
Semantic queries:    Precision={bt['semantic']['precision_at_5']}  MRR={bt['semantic']['mrr']}
Keyword  queries:    Precision={bt['keyword']['precision_at_5']}  MRR={bt['keyword']['mrr']}

Full report saved to: eval/report.json
Low-score samples ({LOW_SCORE_SAMPLE_COUNT}): see report.json → low_score_samples
""")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.2: Verify imports work without running GLM/Pinecone**

```bash
cd d:/PythonProjects/TestProject
python -c "
import sys; sys.path.insert(0, 'backend')
import ast, pathlib
src = pathlib.Path('eval/evaluate.py').read_text(encoding='utf-8')
ast.parse(src)
print('OK: evaluate.py parses without syntax errors')
"
```

Expected: `OK: evaluate.py parses without syntax errors`

- [ ] **Step 4.3: Commit `evaluate.py`**

```bash
git add eval/evaluate.py
git commit -m "feat(eval): add evaluate.py — metrics, LLM judge, contribution analysis"
```

---

## Task 5: Generate testset and run evaluation

- [ ] **Step 5.1: Run `generate_testset.py`**

Ensure `.env` in `backend/` has `GLM_API_KEY`, `GLM_API_URL`, `GLM_MODEL_ID` set, then:

```bash
cd d:/PythonProjects/TestProject
python eval/generate_testset.py
```

Expected output ends with:
```
Done. 400 queries written to eval/testset.json
Next step: git add eval/testset.json && git commit -m 'data: add eval testset'
```

If fewer than 400 succeed (GLM errors), re-run — the script overwrites output each time and uses `random.seed(42)` for reproducibility.

- [ ] **Step 5.2: Spot-check 5 entries in `testset.json`**

```bash
python -c "
import json
data = json.load(open('eval/testset.json', encoding='utf-8'))
print(f'Total: {len(data)}')
for item in data[:5]:
    print(f'  [{item[\"query_type\"]}] {item[\"query\"]}  → {item[\"relevant_repo_ids\"][0]}')
"
```

Expected: 5 lines with reasonable Chinese queries, each mapped to a `parent_id` like `"tiangolo/fastapi"`.

- [ ] **Step 5.3: Commit testset**

```bash
git add eval/testset.json
git commit -m "data: add 400-query stratified eval testset"
```

- [ ] **Step 5.4: Run `evaluate.py`**

Ensure `PINECONE_API_KEY` and `PINECONE_INDEX_NAME` are also in `.env`, then:

```bash
cd d:/PythonProjects/TestProject
python eval/evaluate.py
```

This will take ~15–30 minutes (400 queries × Pinecone latency + ~1600 LLM judge calls). Terminal summary prints at the end.

- [ ] **Step 5.5: Verify `report.json` structure**

```bash
python -c "
import json
r = json.load(open('eval/report.json', encoding='utf-8'))
assert 'metrics' in r
assert 'contribution' in r
assert 'low_score_samples' in r
assert len(r['low_score_samples']) <= 50
print('Precision@5:', r['metrics']['precision_at_5'])
print('MRR:        ', r['metrics']['mrr'])
print('BM25 only:  ', r['contribution']['bm25_only_pct'])
print('Pinecone only:', r['contribution']['pinecone_only_pct'])
print('OK: report.json structure valid')
"
```

Expected: numeric values printed, ends with `OK: report.json structure valid`

- [ ] **Step 5.6: Human spot-check of low-score samples**

Open `eval/report.json`, scroll to `low_score_samples`. For 10–20 entries, manually verify: does the retrieved result seem relevant to the query even though LLM judge said 0? If judge errors are common (>30%), re-run with a lower-temperature judge prompt.

- [ ] **Step 5.7: Final commit**

```bash
git add .gitignore
git commit -m "chore: ensure eval/report.json is gitignored"
```

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `python eval/generate_testset.py` | Regenerate testset (only if corpus changes) |
| `python eval/evaluate.py` | Run evaluation, outputs `eval/report.json` |
| `cat eval/report.json \| python -m json.tool \| head -40` | Quick peek at report |
