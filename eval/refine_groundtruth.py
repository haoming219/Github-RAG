"""
Ground truth refiner.  Run from project root:
    python eval/refine_groundtruth.py

Reads:  eval/testset.json, backend/parent_chunks.json
Writes: eval/testset.json  (in-place, adds refined_chunk_ids field)
Env:    LLM_API_KEY, LLM_API_URL, LLM_MODEL_ID

For entries with more than MAX_RELEVANT relevant chunks, asks the LLM to
pick the 2-4 most important ones.  Entries at or below MAX_RELEVANT are
kept as-is.  Already-refined entries are skipped on resume.
"""
import json, os, pathlib
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).parent.parent / "backend" / ".env")

BACKEND       = pathlib.Path(__file__).parent.parent / "backend"
EVAL_DIR      = pathlib.Path(__file__).parent
TESTSET_PATH  = EVAL_DIR / "testset.json"

MAX_RELEVANT  = 4   # keep entries with <= this many as-is
TARGET_MIN    = 2
TARGET_MAX    = 4

REFINE_PROMPT = """\
You are building a retrieval evaluation dataset. For the user query below, \
multiple repository sections have already been judged relevant. Your task is \
to select only the MOST ESSENTIAL {target_min}-{target_max} sections — \
the ones a user would be most satisfied to land on first.

User query: {query}

Candidate sections (each shown as INDEX | TITLE):
{candidates}

Reply with ONLY the indices of the best {target_min}-{target_max} sections, \
comma-separated (e.g. 0,2,5). No explanation."""


def _make_client() -> OpenAI:
    api_url  = os.environ["LLM_API_URL"]
    base_url = api_url.rsplit("/chat/completions", 1)[0] if "/chat/completions" in api_url else api_url
    return OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=base_url)


def _refine(client: OpenAI, query: str, chunk_ids: list[str],
            chunk_lookup: dict[str, dict]) -> list[str] | None:
    """Ask LLM to pick the best 2-4 from chunk_ids. Returns refined list or None on failure."""
    candidates_info = []
    for idx, cid in enumerate(chunk_ids):
        chunk = chunk_lookup.get(cid, {})
        title = chunk.get("section_title", "(no title)").strip()
        # trim title to keep prompt compact
        if len(title) > 120:
            title = title[:120] + "…"
        candidates_info.append(f"{idx} | {title}")

    candidates_str = "\n".join(candidates_info)
    prompt = REFINE_PROMPT.format(
        query=query,
        target_min=TARGET_MIN,
        target_max=TARGET_MAX,
        candidates=candidates_str,
    )

    try:
        resp = client.chat.completions.create(
            model=os.environ["LLM_MODEL_ID"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=32,
            temperature=0.0,
        )
        text = resp.choices[0].message.content.strip()
        indices = [int(x.strip()) for x in text.split(",") if x.strip().isdigit()]
        # validate range and cardinality
        valid = [i for i in indices if 0 <= i < len(chunk_ids)]
        if TARGET_MIN <= len(valid) <= TARGET_MAX:
            return [chunk_ids[i] for i in valid]
        print(f"    WARNING: LLM returned out-of-range/wrong-count indices '{text}' — skipping refine", flush=True)
        return None
    except Exception as e:
        print(f"    WARNING: LLM call failed: {e}", flush=True)
        return None


def main():
    print("Loading testset.json...", flush=True)
    with open(TESTSET_PATH, encoding="utf-8") as f:
        testset: list[dict] = json.load(f)
    print(f"Total entries: {len(testset)}", flush=True)

    print("Loading parent_chunks.json...", flush=True)
    with open(BACKEND / "parent_chunks.json", encoding="utf-8") as f:
        all_chunks: dict[str, dict] = json.load(f)

    client = _make_client()
    refined_count = 0
    skipped_small = 0
    skipped_done  = 0
    failed        = 0

    for i, item in enumerate(testset):
        source_chunk_id = item.get("meta", {}).get("source_chunk_id", f"entry_{i}")
        chunk_ids = item["relevant_chunk_ids"]

        # Already refined in a previous run
        if "refined_chunk_ids" in item:
            skipped_done += 1
            continue

        if len(chunk_ids) <= MAX_RELEVANT:
            skipped_small += 1
            continue

        query = item["query"]
        print(f"[{i+1}/{len(testset)}] {source_chunk_id}  ({len(chunk_ids)} → refining)", flush=True)

        result = _refine(client, query, chunk_ids, all_chunks)
        if result is None:
            failed += 1
            continue

        item["refined_chunk_ids"] = result
        print(f"  → kept {len(result)}: {result}", flush=True)
        refined_count += 1

        # Incremental write after each entry
        TESTSET_PATH.write_text(
            json.dumps(testset, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(
        f"\nDone. refined={refined_count}, skipped_small={skipped_small}, "
        f"already_done={skipped_done}, failed={failed}",
        flush=True,
    )
    print(
        "Entries with refined_chunk_ids: "
        + str(sum(1 for x in testset if "refined_chunk_ids" in x))
    )


if __name__ == "__main__":
    main()
