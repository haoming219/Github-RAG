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
