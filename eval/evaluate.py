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
