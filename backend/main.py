import os, json, pathlib
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pinecone import Pinecone

from models import ChatRequest, FilterOptions, StarsRange
from retriever import hybrid_search, _load_artifacts
from llm import stream_answer

_BASE = pathlib.Path(__file__).parent

_pinecone_index = None
_filter_options: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """预加载所有耗时资源，消除第一次请求的冷启动延迟。"""
    global _pinecone_index, _filter_options
    print("[startup] 加载 sentence-transformers 模型、BM25 索引、chunk metadata...", flush=True)
    _load_artifacts()
    print("[startup] 连接 Pinecone...", flush=True)
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    _pinecone_index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
    print("[startup] 加载 filter_options.json...", flush=True)
    with open(_BASE / "filter_options.json", encoding="utf-8") as f:
        _filter_options = json.load(f)
    print("[startup] 全部预加载完成，服务就绪。", flush=True)
    yield

app = FastAPI(title="GitHub RAG API", lifespan=lifespan)

# CORS
origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_pinecone_index():
    return _pinecone_index

def get_filter_options() -> dict:
    return _filter_options

@app.get("/api/filters/options", response_model=FilterOptions)
def filters_options():
    opts = get_filter_options()
    sr = opts.get("stars_range", {"min": 0, "max": 0})
    return FilterOptions(
        languages=opts.get("languages", []),
        topics=opts.get("topics", []),
        stars_range=StarsRange(min=sr["min"], max=sr["max"])
    )

@app.post("/api/chat")
async def chat(request: ChatRequest):
    import time
    t0 = time.perf_counter()

    filters = request.filters
    language = filters.language if filters else ""
    min_stars = filters.min_stars if filters else 0
    topics = filters.topics if filters else []

    index = get_pinecone_index()

    t1 = time.perf_counter()
    docs = hybrid_search(
        query=request.messages[-1].content,
        pinecone_index=index,
        language=language or "",
        min_stars=min_stars or 0,
        topics=topics or [],
        top_k=5
    )
    t2 = time.perf_counter()
    print(f"[timing] hybrid_search: {t2 - t1:.2f}s | docs returned: {len(docs)}", flush=True)

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
        print(f"[timing] total stream duration: {time.perf_counter() - t_start:.2f}s", flush=True)
        print(f"[timing] total request time: {time.perf_counter() - t0:.2f}s", flush=True)

    return StreamingResponse(
        timed_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

@app.get("/health")
def health():
    return {"status": "ok"}
