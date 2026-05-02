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
