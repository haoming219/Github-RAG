import os, json, pickle, ast, pathlib
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from rank_bm25 import BM25Okapi
from chunker import clean_readme, split_by_headings

load_dotenv()

_BASE = pathlib.Path(__file__).parent  # always relative to this file, not cwd
DATA_DIR = _BASE.parent  # CSVs are in the project root (one level up from backend/)
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]

def load_data():
    df1 = pd.read_csv(DATA_DIR / "github_basic.csv")
    df2 = pd.read_csv(DATA_DIR / "github_readmes.csv")
    df3 = pd.read_csv(DATA_DIR / "github_readmes2.csv")

    # Normalize column names
    df2 = df2[["Full Name", "Readme Content"]].drop_duplicates(subset=["Full Name"])
    df3 = df3[["Full Name", "Readme Content1"]].rename(
        columns={"Readme Content1": "Readme Content"}
    ).drop_duplicates(subset=["Full Name"])

    # Merge readmes: prefer df2, fall back to df3
    df_readme = pd.concat([df2, df3]).drop_duplicates(subset=["Full Name"], keep="first")
    df = df1.merge(df_readme, on="Full Name", how="left").fillna("")
    df = df.drop(columns=["ID"], errors="ignore")
    return df

def build_parent(row) -> dict:
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

    print("Building chunks...")
    all_chunks = []
    parents = {}
    for _, row in df.iterrows():
        parent = build_parent(row)
        full_name = parent["full_name"]
        parents[full_name] = parent
        readme = str(row.get("Readme Content", "")).strip()
        if readme:
            cleaned = clean_readme(readme)
            chunks = split_by_headings(cleaned, full_name)
        else:
            # No README: create a single chunk from description
            chunks = [{
                "parent_id": full_name,
                "section_title": "__description__",
                "content": parent["description"] or full_name,
                "chunk_index": 0
            }]
        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")

    # --- Pinecone upload ---
    print("Uploading to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pc.Index(INDEX_NAME)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        texts = [c["content"] for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False)
        vectors = []
        for j, (chunk, emb) in enumerate(zip(batch, embeddings)):
            parent = parents[chunk["parent_id"]]
            meta = {
                "parent_id": chunk["parent_id"],
                "section_title": chunk["section_title"],
                "chunk_index": chunk["chunk_index"],
                **parent,  # topics stored as a real list — Pinecone supports string arrays
            }
            vectors.append({
                "id": f"{chunk['parent_id']}__chunk{chunk['chunk_index']}",
                "values": emb.tolist(),
                "metadata": meta
            })
        index.upsert(vectors=vectors)
        print(f"Uploaded {min(i+batch_size, len(all_chunks))}/{len(all_chunks)}")

    # --- BM25 index ---
    print("Building BM25 index...")
    tokenized = [c["content"].lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized)
    with open(_BASE / "bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)

    # chunk_metadata: list index → parent metadata + chunk info
    chunk_metadata = []
    for chunk in all_chunks:
        parent = parents[chunk["parent_id"]]
        chunk_metadata.append({
            "parent_id": chunk["parent_id"],
            "section_title": chunk["section_title"],
            "chunk_index": chunk["chunk_index"],
            "content": chunk["content"],
            **parent
        })
    with open(_BASE / "chunk_metadata.json", "w", encoding="utf-8") as f:
        json.dump(chunk_metadata, f, ensure_ascii=False)

    # --- filter_options ---
    print("Building filter options...")
    languages = sorted(set(p["language"] for p in parents.values() if p["language"]))
    all_topics = set()
    for p in parents.values():
        all_topics.update(p["topics"])
    stars_values = [p["stars"] for p in parents.values()]
    filter_options = {
        "languages": languages,
        "topics": sorted(all_topics),
        "stars_range": {"min": min(stars_values), "max": max(stars_values)}
    }
    with open(_BASE / "filter_options.json", "w", encoding="utf-8") as f:
        json.dump(filter_options, f, ensure_ascii=False)

    print("Done. Commit bm25_index.pkl, chunk_metadata.json, filter_options.json.")

if __name__ == "__main__":
    main()
