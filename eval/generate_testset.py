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
