"""
Offline evaluation helper: wraps CustomRetriever in a RetrieverQueryEngine
for interactive query testing without starting the FastAPI server.

Usage:
    cd backend
    python eval_retriever.py "Python web framework"
"""
import os, sys, pathlib
from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone
from llama_index.core.query_engine import RetrieverQueryEngine
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
