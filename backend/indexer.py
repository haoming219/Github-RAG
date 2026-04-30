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
    ordered_parent_ids: list = []  # BM25 corpus order

    for _, row in df.iterrows():
        meta = build_repo_meta(row)
        full_name = meta["full_name"]
        readme = str(row.get("Readme Content", "")).strip()
        if not readme:
            continue  # skip repos without README — only index repos with real content

        cleaned = clean_readme(readme)
        parents = split_by_headings(cleaned, full_name, description=meta["description"])

        for parent in parents:
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
        time.sleep(0.5)

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
    for chunk in all_parent_chunks.values():
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
