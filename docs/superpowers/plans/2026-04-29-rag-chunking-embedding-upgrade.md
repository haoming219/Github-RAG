# RAG Chunking & Embedding Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the GitHub RAG backend with two-level chunking (heading + SentenceSplitter), text-embedding-3-small embeddings, and a LlamaIndex-based CustomRetriever with BM25+RRF hybrid search.

**Architecture:** Parent chunks are heading-level README sections stored locally in `parent_chunks.json`. Child chunks are sentence-split sub-sections vectorized into Pinecone (dim=1536) with full repo metadata. At query time, CustomRetriever runs vector search (Max Pooling to parent level) + BM25 search on parent chunks, fuses both with RRF, and returns Top-5 parent chunks as NodeWithScore objects to the existing SSE streaming path.

**Tech Stack:** Python 3.x, LlamaIndex (`llama-index-core`, `llama-index-embeddings-openai`), Pinecone SDK, rank-bm25, tiktoken, FastAPI, OpenAI SDK (llm.py unchanged except truncation removal)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `backend/chunker.py` | Rewrite | `clean_readme()` (keep) + `split_by_headings()` (token threshold) + `split_into_children()` (SentenceSplitter) |
| `backend/indexer.py` | Rewrite | Load CSVs → two-level chunk → embed via OpenAIEmbedding → upsert Pinecone → write local files |
| `backend/retriever.py` | Rewrite | `CustomRetriever(BaseRetriever)` with vector + BM25 + RRF |
| `backend/llm.py` | Small edit | Remove `[:800]` truncation in `build_prompt_messages()` |
| `backend/main.py` | Small edit | Lifespan loads `CustomRetriever`; request path calls `retriever.retrieve()` |
| `backend/requirements.txt` | Edit | Add `llama-index-core`, `llama-index-embeddings-openai`, `tiktoken`; remove `sentence-transformers` |
| `backend/tests/test_chunker.py` | Create | Unit tests for heading splitter and child splitter |
| `backend/tests/test_retriever.py` | Create | Unit tests for Max Pooling, BM25 post-filter, RRF |
| `backend/eval_retriever.py` | Create | Offline RetrieverQueryEngine eval helper (not in request path) |

---

## Task 1: Update requirements.txt

**Files:**
- Modify: `backend/requirements.txt`

- [x] **Step 1: Edit requirements.txt**

Replace `sentence-transformers==3.0.1` with the new deps:

```
fastapi==0.111.0
uvicorn[standard]==0.29.0
python-dotenv==1.0.1
llama-index-core
llama-index-embeddings-openai
tiktoken
rank-bm25==0.2.2
pinecone-client==3.2.2
openai>=1.51.0
pandas==2.2.2
pydantic>=2.9.0
```

- [x] **Step 2: Install and verify**

```bash
cd backend
pip install -r requirements.txt
python -c "from llama_index.core.node_parser import SentenceSplitter; from llama_index.embeddings.openai import OpenAIEmbedding; from llama_index.core.retrievers import BaseRetriever; import tiktoken; print('OK')"
```

Expected: `OK`

- [x] **Step 3: Commit**

```bash
git add backend/requirements.txt
git commit -m "chore: replace sentence-transformers with llama-index-core + tiktoken"
```

---

## Task 2: Rewrite chunker.py

**Files:**
- Rewrite: `backend/chunker.py`
- Create: `backend/tests/test_chunker.py`

- [ ] **Step 1: Create tests directory and write failing tests**

```bash
mkdir -p backend/tests
touch backend/tests/__init__.py
```

Create `backend/tests/test_chunker.py`:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from chunker import clean_readme, split_by_headings, split_into_children

# --- clean_readme ---

def test_clean_readme_removes_badges():
    text = "[![Build](https://img.shields.io/badge/build-passing)](https://example.com)\nHello"
    assert "[![" not in clean_readme(text)
    assert "Hello" in clean_readme(text)

def test_clean_readme_removes_html():
    text = "<div>Some content</div>\nHello"
    assert "<div>" not in clean_readme(text)
    assert "Hello" in clean_readme(text)

def test_clean_readme_removes_image_links():
    text = "![logo](https://example.com/logo.png)\nHello"
    assert "![" not in clean_readme(text)
    assert "Hello" in clean_readme(text)

# --- split_by_headings ---

def test_split_by_headings_basic():
    text = "Intro paragraph here with enough content to pass the threshold.\n\n## Installation\nRun pip install.\n\n## Usage\nImport and use it in your project."
    chunks = split_by_headings(text, "owner/repo")
    assert len(chunks) >= 2
    # intro chunk
    assert chunks[0]["section_title"] == "__intro__"
    assert chunks[0]["section_index"] == 0
    assert chunks[0]["parent_id"] == "owner/repo__0"
    # heading chunk
    assert any(c["section_title"] == "## Installation" for c in chunks)

def test_split_by_headings_short_chunk_merged():
    # A chunk under 20 tokens should be merged into previous
    text = "This is a long enough intro paragraph to stand on its own.\n\n## Section A\nThis section has good content here.\n\n## Tiny\nHi"
    chunks = split_by_headings(text, "owner/repo")
    # "Hi" alone is < 20 tokens, should be merged into previous chunk
    contents = [c["content"] for c in chunks]
    assert not any(c == "Hi" for c in contents)

def test_split_by_headings_no_readme_fallback():
    chunks = split_by_headings("", "owner/repo", description="A cool library")
    assert len(chunks) == 1
    assert chunks[0]["section_title"] == "__description__"
    assert chunks[0]["content"] == "A cool library"
    assert chunks[0]["parent_id"] == "owner/repo__0"

def test_split_by_headings_parent_id_format():
    text = "Intro.\n\n## A\nContent A with enough text to pass threshold.\n\n## B\nContent B with enough text to pass threshold."
    chunks = split_by_headings(text, "facebook/react")
    for i, c in enumerate(chunks):
        assert c["parent_id"] == f"facebook/react__{i}"
        assert c["section_index"] == i

# --- split_into_children ---

def test_split_into_children_short_content_single_child():
    # Content under 512 tokens should produce exactly one child chunk
    parent = {
        "parent_id": "owner/repo__0",
        "content": "Short content that is well under five hundred twelve tokens."
    }
    children = split_into_children(parent)
    assert len(children) == 1
    assert children[0]["parent_id"] == "owner/repo__0"
    assert children[0]["child_index"] == 0
    assert children[0]["vector_id"] == "owner/repo__0__child0"
    assert "content" in children[0]

def test_split_into_children_long_content_multiple_children():
    # ~600 tokens of content should produce 2 children
    long_text = "This is a sentence. " * 100  # ~300 words ≈ 400 tokens
    parent = {"parent_id": "owner/repo__1", "content": long_text * 2}
    children = split_into_children(parent)
    assert len(children) >= 2
    for i, c in enumerate(children):
        assert c["child_index"] == i
        assert c["vector_id"] == f"owner/repo__1__child{i}"

def test_split_into_children_child_index_sequential():
    long_text = "Word sentence here. " * 150
    parent = {"parent_id": "repo/name__2", "content": long_text}
    children = split_into_children(parent)
    indices = [c["child_index"] for c in children]
    assert indices == list(range(len(children)))
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd backend
python -m pytest tests/test_chunker.py -v 2>&1 | head -40
```

Expected: multiple FAILED / ImportError on `split_into_children`

- [ ] **Step 3: Rewrite chunker.py**

```python
import re
import tiktoken
from llama_index.core.node_parser import SentenceSplitter

_tokenizer = tiktoken.get_encoding("cl100k_base")
_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50, tokenizer=_tokenizer.encode)

MIN_TOKENS = 20


def _count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))


def clean_readme(text: str) -> str:
    text = re.sub(r'\[!\[.*?\]\(.*?\)\]\(.*?\)', '', text)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def split_by_headings(text: str, full_name: str, description: str = "") -> list[dict]:
    """
    First-level split: divide cleaned README text at Markdown heading boundaries.
    Returns parent chunk dicts with parent_id, section_index, section_title, content.
    If text is empty, falls back to a single description chunk.
    """
    if not text.strip():
        return [{
            "parent_id": f"{full_name}__0",
            "full_name": full_name,
            "section_index": 0,
            "section_title": "__description__",
            "content": description or full_name,
        }]

    lines = text.split('\n')
    raw_chunks = []
    current_title = "__intro__"
    current_lines = []

    for line in lines:
        if re.match(r'^#{1,6}\s', line):
            content = '\n'.join(current_lines).strip()
            if content:
                raw_chunks.append({"section_title": current_title, "content": content})
            current_title = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    content = '\n'.join(current_lines).strip()
    if content:
        raw_chunks.append({"section_title": current_title, "content": content})

    # Merge chunks below MIN_TOKENS threshold into previous chunk
    merged = []
    for chunk in raw_chunks:
        if merged and _count_tokens(chunk["content"]) < MIN_TOKENS:
            merged[-1]["content"] += "\n" + chunk["content"]
        else:
            merged.append(chunk)

    # Assign parent_id and section_index
    result = []
    for i, chunk in enumerate(merged):
        result.append({
            "parent_id": f"{full_name}__{i}",
            "full_name": full_name,
            "section_index": i,
            "section_title": chunk["section_title"],
            "content": chunk["content"],
        })
    return result


def split_into_children(parent: dict) -> list[dict]:
    """
    Second-level split: divide a parent chunk's content using SentenceSplitter.
    Returns child chunk dicts with parent_id, child_index, vector_id, content.
    """
    from llama_index.core.schema import Document
    doc = Document(text=parent["content"])
    nodes = _splitter.get_nodes_from_documents([doc])
    children = []
    for i, node in enumerate(nodes):
        children.append({
            "parent_id": parent["parent_id"],
            "child_index": i,
            "vector_id": f"{parent['parent_id']}__child{i}",
            "content": node.get_content(),
        })
    return children
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd backend
python -m pytest tests/test_chunker.py -v
```

Expected: all PASSED

- [ ] **Step 5: Commit**

```bash
git add backend/chunker.py backend/tests/__init__.py backend/tests/test_chunker.py
git commit -m "feat(chunker): two-level chunking with SentenceSplitter and tiktoken token threshold"
```

---

## Task 3: Rewrite retriever.py

**Files:**
- Rewrite: `backend/retriever.py`
- Create: `backend/tests/test_retriever.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_retriever.py`:

```python
import sys, pathlib, pickle, json, types
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from retriever import _apply_filter, _rrf, _max_pool_vector_results

# --- _apply_filter ---

def test_apply_filter_no_constraints():
    meta = {"language": "Python", "stars": 5000, "topics": ["web"]}
    assert _apply_filter(meta, language="", min_stars=0, topics=[]) is True

def test_apply_filter_language_match():
    meta = {"language": "Python", "stars": 100, "topics": []}
    assert _apply_filter(meta, language="Python", min_stars=0, topics=[]) is True

def test_apply_filter_language_mismatch():
    meta = {"language": "Go", "stars": 100, "topics": []}
    assert _apply_filter(meta, language="Python", min_stars=0, topics=[]) is False

def test_apply_filter_min_stars():
    meta = {"language": "", "stars": 500, "topics": []}
    assert _apply_filter(meta, language="", min_stars=1000, topics=[]) is False
    assert _apply_filter(meta, language="", min_stars=500, topics=[]) is True

def test_apply_filter_topics_or_logic():
    meta = {"language": "", "stars": 0, "topics": ["web", "api"]}
    assert _apply_filter(meta, language="", min_stars=0, topics=["api", "ml"]) is True
    assert _apply_filter(meta, language="", min_stars=0, topics=["ml"]) is False

# --- _rrf ---

def test_rrf_single_list():
    # single list: rank order preserved
    result = _rrf([{"a": 0, "b": 1, "c": 2}], k=60)
    assert result[0] == "a"
    assert result[1] == "b"

def test_rrf_two_lists_boost_overlap():
    # "b" appears in both lists → higher score than "a" (only in list 1)
    list_a = {"a": 0, "b": 1}
    list_b = {"b": 0, "c": 1}
    result = _rrf([list_a, list_b], k=60)
    assert result[0] == "b"

def test_rrf_empty_list_ignored():
    result = _rrf([{"a": 0, "b": 1}, {}], k=60)
    assert "a" in result

def test_rrf_returns_top_k():
    ranked = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5}
    result = _rrf([ranked], k=60, top_k=3)
    assert len(result) == 3

# --- _max_pool_vector_results ---

def _make_match(parent_id):
    m = types.SimpleNamespace()
    m.metadata = {"parent_id": parent_id}
    return m

def test_max_pool_deduplicates():
    matches = [
        _make_match("repo/a__0"),
        _make_match("repo/a__0"),  # duplicate — same parent
        _make_match("repo/b__0"),
    ]
    result = _max_pool_vector_results(matches, limit=20)
    assert len(result) == 2
    assert "repo/a__0" in result
    assert result["repo/a__0"] == 0  # best rank is 0 (first occurrence)

def test_max_pool_respects_limit():
    matches = [_make_match(f"repo/x__{i}") for i in range(30)]
    result = _max_pool_vector_results(matches, limit=10)
    assert len(result) == 10

def test_max_pool_records_first_seen_rank():
    # First occurrence at rank 0 should win over second at rank 2
    matches = [
        _make_match("repo/a__0"),  # rank 0
        _make_match("repo/b__0"),  # rank 1
        _make_match("repo/a__0"),  # rank 2 — duplicate, should be ignored
    ]
    result = _max_pool_vector_results(matches, limit=20)
    assert result["repo/a__0"] == 0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd backend
python -m pytest tests/test_retriever.py -v 2>&1 | head -40
```

Expected: FAILED / ImportError

- [ ] **Step 3: Rewrite retriever.py**

```python
import json, pickle, pathlib, os
from typing import Optional
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from llama_index.embeddings.openai import OpenAIEmbedding
from rank_bm25 import BM25Okapi

_BASE = pathlib.Path(__file__).parent


def _apply_filter(meta: dict, language: str, min_stars: int, topics: list) -> bool:
    if language and meta.get("language", "") != language:
        return False
    if min_stars and int(meta.get("stars", 0)) < min_stars:
        return False
    if topics:
        repo_topics = meta.get("topics", [])
        if not any(t in repo_topics for t in topics):
            return False
    return True


def _rrf(ranked_dicts: list[dict], k: int = 60, top_k: int = 100) -> list:
    """
    Reciprocal Rank Fusion over a list of {id: rank} dicts.
    Returns a list of IDs sorted by fused score descending.
    """
    scores: dict = {}
    for ranked in ranked_dicts:
        for pid, rank in ranked.items():
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]


def _max_pool_vector_results(matches: list, limit: int = 20) -> dict:
    """
    Aggregate Pinecone child-chunk matches to parent level.
    For each parent_id, keep only the best (lowest) rank.
    Returns {parent_id: best_rank} for up to `limit` distinct parents.
    """
    seen: dict = {}
    for rank, match in enumerate(matches):
        pid = match.metadata["parent_id"]
        if pid not in seen:
            seen[pid] = rank
        if len(seen) >= limit:
            break
    return seen


class CustomRetriever(BaseRetriever):
    def __init__(
        self,
        pinecone_index,
        parent_chunks: dict,
        bm25: BM25Okapi,
        bm25_parent_ids: list,
        embed_model: OpenAIEmbedding,
    ):
        super().__init__()
        self._pinecone_index = pinecone_index
        self._parent_chunks = parent_chunks
        self._bm25 = bm25
        self._bm25_parent_ids = bm25_parent_ids
        self._embed_model = embed_model
        # Filter attributes — set before each retrieve() call
        self.language: str = ""
        self.min_stars: int = 0
        self.topics: list = []

    def _build_pc_filter(self) -> dict:
        f = {}
        if self.language:
            f["language"] = {"$eq": self.language}
        if self.min_stars:
            f["stars"] = {"$gte": self.min_stars}
        if self.topics:
            f["topics"] = {"$in": self.topics}
        return f

    def _vector_search(self, query: str) -> dict:
        vector = self._embed_model.get_text_embedding(query)
        pc_filter = self._build_pc_filter()
        kwargs = {"vector": vector, "top_k": 60, "include_metadata": True}
        if pc_filter:
            kwargs["filter"] = pc_filter
        results = self._pinecone_index.query(**kwargs)
        return _max_pool_vector_results(results.matches, limit=20)

    def _bm25_search(self, query: str) -> dict:
        tokenized = query.lower().split()
        scores = self._bm25.get_scores(tokenized)
        top20 = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:20]
        ranked = {}
        rank = 0
        for idx in top20:
            pid = self._bm25_parent_ids[idx]
            meta = self._parent_chunks.get(pid, {})
            if _apply_filter(meta, self.language, self.min_stars, self.topics):
                ranked[pid] = rank
                rank += 1
        return ranked

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query = query_bundle.query_str
        vector_ranked = self._vector_search(query)
        bm25_ranked = self._bm25_search(query)

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


def load_retriever(pinecone_index) -> "CustomRetriever":
    """Load all local artifacts and return a ready CustomRetriever."""
    with open(_BASE / "parent_chunks.json", encoding="utf-8") as f:
        parent_chunks = json.load(f)
    with open(_BASE / "bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open(_BASE / "bm25_parent_ids.json", encoding="utf-8") as f:
        bm25_parent_ids = json.load(f)

    assert len(bm25_parent_ids) == bm25.corpus_size, (
        f"BM25 corpus size mismatch: {len(bm25_parent_ids)} ids vs {bm25.corpus_size} docs. "
        "Regenerate bm25_index.pkl and bm25_parent_ids.json together."
    )

    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.environ["LLM_API_KEY"],
        api_base=os.environ["LLM_API_URL"],
    )
    return CustomRetriever(
        pinecone_index=pinecone_index,
        parent_chunks=parent_chunks,
        bm25=bm25,
        bm25_parent_ids=bm25_parent_ids,
        embed_model=embed_model,
    )
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd backend
python -m pytest tests/test_retriever.py -v
```

Expected: all PASSED

- [ ] **Step 5: Commit**

```bash
git add backend/retriever.py backend/tests/test_retriever.py
git commit -m "feat(retriever): CustomRetriever with BM25+vector+RRF hybrid search via LlamaIndex BaseRetriever"
```

---

## Task 4: Rewrite indexer.py

**Files:**
- Rewrite: `backend/indexer.py`

Note: indexer.py is a one-shot script, not called at server startup. Tests require live API keys so we skip unit tests and rely on a dry-run smoke check instead.

- [ ] **Step 1: Rewrite indexer.py**

```python
import os, json, pickle, ast, pathlib, time
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from llama_index.embeddings.openai import OpenAIEmbedding
from rank_bm25 import BM25Okapi
from chunker import clean_readme, split_by_headings, split_into_children

load_dotenv()

_BASE = pathlib.Path(__file__).parent
DATA_DIR = _BASE.parent

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
EMBED_DIM = 1536


def load_data() -> pd.DataFrame:
    df1 = pd.read_csv(DATA_DIR / "github_basic.csv")
    df2 = pd.read_csv(DATA_DIR / "github_readmes.csv")
    df3 = pd.read_csv(DATA_DIR / "github_readmes2.csv")

    df2 = df2[["Full Name", "Readme Content"]].drop_duplicates(subset=["Full Name"])
    df3 = df3[["Full Name", "Readme Content1"]].rename(
        columns={"Readme Content1": "Readme Content"}
    ).drop_duplicates(subset=["Full Name"])

    df_readme = pd.concat([df2, df3]).drop_duplicates(subset=["Full Name"], keep="first")
    df = df1.merge(df_readme, on="Full Name", how="left").fillna("")
    return df.drop(columns=["ID"], errors="ignore")


def build_repo_meta(row) -> dict:
    topics_raw = str(row.get("Topics", ""))
    try:
        topics = ast.literal_eval(topics_raw) if topics_raw.startswith("[") else []
    except Exception:
        topics = []
    return {
        "full_name": str(row["Full Name"]),
        "clone_url": str(row.get("Clone URL", "")),
        "description": str(row.get("Description", "")),
        "topics": topics,
        "language": str(row.get("Language", "")),
        "stars": int(row.get("Stars", 0) or 0),
        "forks": int(row.get("Forks", 0) or 0),
        "watchers": int(row.get("Watchers", 0) or 0),
        "issues": int(row.get("Issues", 0) or 0),
        "create_time": str(row.get("Create_Time", "")),
        "update_time": str(row.get("Update_Time", "")),
        "push_time": str(row.get("Push_Time", "")),
    }


def main():
    print("Loading data...")
    df = load_data()

    print("Building parent and child chunks...")
    all_parent_chunks: dict = {}   # parent_id -> full parent chunk dict
    all_child_chunks: list = []    # flat list of child chunk dicts (for embedding)
    ordered_parent_ids: list = []  # BM25 corpus order (df.iterrows() order)

    for _, row in df.iterrows():
        meta = build_repo_meta(row)
        full_name = meta["full_name"]
        readme = str(row.get("Readme Content", "")).strip()

        if readme:
            cleaned = clean_readme(readme)
            parents = split_by_headings(cleaned, full_name, description=meta["description"])
        else:
            parents = split_by_headings("", full_name, description=meta["description"])

        for parent in parents:
            # Enrich parent chunk with repo metadata
            parent.update({k: v for k, v in meta.items() if k != "full_name"})
            all_parent_chunks[parent["parent_id"]] = parent
            ordered_parent_ids.append(parent["parent_id"])

            children = split_into_children(parent)
            all_child_chunks.extend(children)

    print(f"Parent chunks: {len(all_parent_chunks)}, Child chunks: {len(all_child_chunks)}")

    # --- Pinecone setup ---
    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    index = pc.Index(INDEX_NAME)

    # --- Embed and upsert child chunks ---
    print("Embedding and uploading child chunks to Pinecone...")
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.environ["LLM_API_KEY"],
        api_base=os.environ["LLM_API_URL"],
    )

    batch_size = 100
    for i in range(0, len(all_child_chunks), batch_size):
        batch = all_child_chunks[i:i + batch_size]
        texts = [c["content"] for c in batch]
        embeddings = embed_model.get_text_embedding_batch(texts, show_progress=False)

        vectors = []
        for child, emb in zip(batch, embeddings):
            parent = all_parent_chunks[child["parent_id"]]
            vectors.append({
                "id": child["vector_id"],
                "values": emb,
                "metadata": {
                    "parent_id": child["parent_id"],
                    "full_name": parent["full_name"],
                    "section_title": parent["section_title"],
                    "child_index": child["child_index"],
                    "language": parent["language"],
                    "stars": parent["stars"],
                    "forks": parent["forks"],
                    "watchers": parent["watchers"],
                    "issues": parent["issues"],
                    "topics": parent["topics"],
                    "create_time": parent["create_time"],
                    "update_time": parent["update_time"],
                    "push_time": parent["push_time"],
                },
            })
        index.upsert(vectors=vectors)
        print(f"  Uploaded {min(i + batch_size, len(all_child_chunks))}/{len(all_child_chunks)}")
        time.sleep(0.5)  # avoid rate limit

    # --- Write parent_chunks.json ---
    print("Writing parent_chunks.json...")
    with open(_BASE / "parent_chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_parent_chunks, f, ensure_ascii=False)

    # --- Build BM25 on parent chunks ---
    print("Building BM25 index on parent chunks...")
    corpus_texts = [all_parent_chunks[pid]["content"] for pid in ordered_parent_ids]
    tokenized = [t.lower().split() for t in corpus_texts]
    bm25 = BM25Okapi(tokenized)

    with open(_BASE / "bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(_BASE / "bm25_parent_ids.json", "w", encoding="utf-8") as f:
        json.dump(ordered_parent_ids, f, ensure_ascii=False)

    # Sanity check
    assert len(ordered_parent_ids) == bm25.corpus_size, "BM25 corpus size mismatch!"

    # --- filter_options ---
    print("Building filter_options.json...")
    languages = sorted(set(
        all_parent_chunks[pid]["language"]
        for pid in all_parent_chunks
        if all_parent_chunks[pid].get("language")
    ))
    all_topics: set = set()
    all_stars: list = []
    for pid, chunk in all_parent_chunks.items():
        all_topics.update(chunk.get("topics", []))
        all_stars.append(chunk.get("stars", 0))

    filter_options = {
        "languages": languages,
        "topics": sorted(all_topics),
        "stars_range": {"min": min(all_stars), "max": max(all_stars)},
    }
    with open(_BASE / "filter_options.json", "w", encoding="utf-8") as f:
        json.dump(filter_options, f, ensure_ascii=False)

    print("Done. Commit parent_chunks.json, bm25_index.pkl, bm25_parent_ids.json, filter_options.json.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test dry run (no API calls)**

```bash
cd backend
python -c "
import ast, pathlib, pandas as pd
from chunker import clean_readme, split_by_headings, split_into_children

# Simulate one repo
readme = '## Installation\nRun pip install.\n\n## Usage\nImport and use the library in your code.'
cleaned = clean_readme(readme)
parents = split_by_headings(cleaned, 'test/repo', description='A test repo')
print(f'Parents: {len(parents)}')
for p in parents:
    children = split_into_children(p)
    print(f'  {p[\"parent_id\"]} -> {len(children)} children')
print('Smoke test OK')
"
```

Expected output:
```
Parents: 2
  test/repo__0 -> 1 children
  test/repo__1 -> 1 children
Smoke test OK
```

- [ ] **Step 3: Commit**

```bash
git add backend/indexer.py
git commit -m "feat(indexer): rewrite with two-level chunking, OpenAIEmbedding, parent_chunks.json output"
```

---

## Task 5: Update llm.py

**Files:**
- Modify: `backend/llm.py:86`

- [ ] **Step 1: Remove the [:800] truncation**

In `backend/llm.py`, find `build_prompt_messages()` and change:

```python
snippet = doc.get("content", "")[:800]
```

to:

```python
snippet = doc.get("content", "")
```

And update the variable name for clarity:

```python
        context_parts.append(
            f"[Repo {i+1}] {repo_name} | {language} | ★{stars}\n"
            f"Description: {description}\n"
            f"Details: {snippet}"
        )
```

remains unchanged (just the slice is removed).

- [ ] **Step 2: Verify syntax**

```bash
cd backend
python -c "import llm; print('llm.py OK')"
```

Expected: `llm.py OK`

- [ ] **Step 3: Commit**

```bash
git add backend/llm.py
git commit -m "feat(llm): remove 800-char truncation on parent chunk content"
```

---

## Task 6: Update main.py

**Files:**
- Modify: `backend/main.py`

- [ ] **Step 1: Rewrite main.py**

Replace the existing content with:

```python
import os, json, pathlib
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pinecone import Pinecone

from models import ChatRequest, FilterOptions, StarsRange
from retriever import load_retriever
from llm import stream_answer

_BASE = pathlib.Path(__file__).parent

_retriever = None
_filter_options: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _retriever, _filter_options
    print("[startup] Connecting to Pinecone...", flush=True)
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    pinecone_index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

    print("[startup] Loading retriever artifacts (parent_chunks, BM25, embed model)...", flush=True)
    _retriever = load_retriever(pinecone_index)

    print("[startup] Loading filter_options.json...", flush=True)
    with open(_BASE / "filter_options.json", encoding="utf-8") as f:
        _filter_options = json.load(f)

    print("[startup] Ready.", flush=True)
    yield


app = FastAPI(title="GitHub RAG API", lifespan=lifespan)

origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/filters/options", response_model=FilterOptions)
def filters_options():
    sr = _filter_options.get("stars_range", {"min": 0, "max": 0})
    return FilterOptions(
        languages=_filter_options.get("languages", []),
        topics=_filter_options.get("topics", []),
        stars_range=StarsRange(min=sr["min"], max=sr["max"]),
    )


@app.post("/api/chat")
async def chat(request: ChatRequest):
    import time
    t0 = time.perf_counter()

    filters = request.filters
    language = filters.language if filters else ""
    min_stars = filters.min_stars if filters else 0
    topics = filters.topics if filters else []

    _retriever.language = language or ""
    _retriever.min_stars = min_stars or 0
    _retriever.topics = topics or []

    query = request.messages[-1].content
    t1 = time.perf_counter()
    nodes = _retriever.retrieve(query)
    t2 = time.perf_counter()
    print(f"[timing] retrieve: {t2 - t1:.2f}s | results: {len(nodes)}", flush=True)

    docs = [n.metadata for n in nodes]
    messages_dicts = [{"role": m.role, "content": m.content} for m in request.messages]

    def timed_stream():
        import time
        first_chunk = True
        t_start = time.perf_counter()
        for chunk in stream_answer(docs, messages_dicts):
            if first_chunk:
                print(f"[timing] time to first LLM chunk: {time.perf_counter() - t_start:.2f}s", flush=True)
                first_chunk = False
            yield chunk
        print(f"[timing] total stream: {time.perf_counter() - t_start:.2f}s", flush=True)
        print(f"[timing] total request: {time.perf_counter() - t0:.2f}s", flush=True)

    return StreamingResponse(
        timed_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 2: Verify syntax**

```bash
cd backend
python -c "import ast; ast.parse(open('main.py').read()); print('main.py syntax OK')"
```

Expected: `main.py syntax OK`

- [ ] **Step 3: Commit**

```bash
git add backend/main.py
git commit -m "feat(main): replace hybrid_search with CustomRetriever.retrieve() in request path"
```

---

## Task 7: Run full test suite

**Files:** none

- [ ] **Step 1: Run all backend tests**

```bash
cd backend
python -m pytest tests/ -v
```

Expected: all tests PASSED (test_chunker.py + test_retriever.py)

- [ ] **Step 2: Verify server starts without errors (no live API keys needed for import check)**

```bash
cd backend
python -c "
import os
os.environ.setdefault('PINECONE_API_KEY', 'dummy')
os.environ.setdefault('PINECONE_INDEX_NAME', 'dummy')
os.environ.setdefault('LLM_API_KEY', 'dummy')
os.environ.setdefault('LLM_API_URL', 'https://aihubmix.com/v1')
os.environ.setdefault('LLM_MODEL_ID', 'gpt-4o-mini')
import main
print('main.py imports OK')
"
```

Expected: `main.py imports OK`

- [ ] **Step 3: Final commit**

```bash
git status  # confirm only expected files remain modified
git add backend/tests/test_chunker.py backend/tests/test_retriever.py backend/tests/__init__.py
git commit -m "chore: verify all tests pass after RAG upgrade"
```

---

## Task 8: Add RetrieverQueryEngine offline eval helper

**Files:**
- Create: `backend/eval_retriever.py`

This is a standalone offline script (not part of the request path) that wires `CustomRetriever` into a `RetrieverQueryEngine` for interactive testing and evaluation, satisfying the spec requirement in §3.

- [ ] **Step 1: Create eval_retriever.py**

```python
"""
Offline evaluation helper: wraps CustomRetriever in a RetrieverQueryEngine
for interactive query testing without starting the FastAPI server.

Usage:
    cd backend
    python eval_retriever.py "Python web framework"
"""
import os, sys, json, pickle, pathlib
from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from retriever import load_retriever

_BASE = pathlib.Path(__file__).parent


def main():
    query = " ".join(sys.argv[1:]) or "Python web framework"

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    pinecone_index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

    retriever = load_retriever(pinecone_index)

    # Wire into RetrieverQueryEngine for offline evaluation
    # (not used in production SSE path — see main.py)
    engine = RetrieverQueryEngine.from_args(retriever=retriever)

    print(f"\nQuery: {query}\n{'='*60}")
    nodes = retriever.retrieve(query)
    print(f"Retrieved {len(nodes)} parent chunks:\n")
    for i, n in enumerate(nodes, 1):
        meta = n.metadata
        print(f"[{i}] {meta.get('full_name')} | {meta.get('language')} | ★{meta.get('stars')}")
        print(f"    {meta.get('section_title')}")
        print(f"    {n.node.text[:200]}...\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

```bash
cd backend
python -c "import ast; ast.parse(open('eval_retriever.py').read()); print('eval_retriever.py syntax OK')"
```

Expected: `eval_retriever.py syntax OK`

- [ ] **Step 3: Commit**

```bash
git add backend/eval_retriever.py
git commit -m "feat(eval): add RetrieverQueryEngine offline eval helper"
```

---

## Task 9: Rebuild Pinecone index (manual, requires live credentials)

This task runs outside the normal test cycle and requires live API keys. Run only after Tasks 1–8 are complete.

- [x] **Step 1: Delete old Pinecone index**

Go to [Pinecone console](https://app.pinecone.io), delete the existing `github-rag` index (dimension 384).

- [x] **Step 2: Run indexer.py**

```bash
cd backend
python indexer.py
```

Expected final lines:
```
Done. Commit parent_chunks.json, bm25_index.pkl, bm25_parent_ids.json, filter_options.json.
```

- [x] **Step 3: Commit generated artifacts**

```bash
git add backend/parent_chunks.json backend/bm25_index.pkl backend/bm25_parent_ids.json backend/filter_options.json
git commit -m "data: rebuild index with text-embedding-3-small and two-level chunking"
```

- [ ] **Step 4: Smoke-test the running server**

```bash
cd backend
uvicorn main:app --reload
```

In another terminal:
```bash
curl -s http://localhost:8000/health
# Expected: {"status":"ok"}

curl -s http://localhost:8000/api/filters/options | python -m json.tool | head -10
# Expected: languages, topics, stars_range fields present

curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Python web framework"}],"filters":{}}' \
  --no-buffer | head -5
# Expected: SSE stream with data: {"text": "..."} lines
```
