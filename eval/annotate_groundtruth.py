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
        full_name=chunk.get("full_name", ""),
        section_title=chunk.get("section_title", ""),
        content=chunk.get("content", ""),
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
        fn = chunk.get("full_name") or ""
        if fn:
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
            pid = chunk.get("parent_id")
            if score == 1 and pid:
                relevant_chunk_ids.append(pid)

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
