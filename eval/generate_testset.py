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
    name_to_idx: dict[str, int],
) -> list[str]:
    """
    Return up to MAX_EXTRA_RELEVANT repo names with cosine similarity > SIMILARITY_THRESHOLD,
    excluding the source itself. Returns [] if fewer than MIN_EXTRA_RELEVANT qualify.
    """
    idx = name_to_idx.get(source_name)
    if idx is None:
        return []
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

    # Build index for O(1) lookup in _find_similar
    name_to_idx = {name: i for i, name in enumerate(names)}

    sampled = _stratified_sample(repo_index)

    testset = []
    for i, (full_name, chunk, stratum) in enumerate(sampled):
        query_type = QUERY_TYPE_CYCLE[i % 2]

        extras = _find_similar(full_name, matrix, names, name_to_idx)
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
    print(f"Queries with >=1 extra relevant: {has_extras}/{len(testset)} "
          f"(avg extras per query: {sum(extras_counts)/max(len(extras_counts),1):.2f})", flush=True)
    print("Next step: git add eval/testset.json && git commit -m 'data: regenerate eval testset v2'")


if __name__ == "__main__":
    main()
