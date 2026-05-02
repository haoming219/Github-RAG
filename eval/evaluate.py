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
