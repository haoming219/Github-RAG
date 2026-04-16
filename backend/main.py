import os, json, pathlib
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pinecone import Pinecone

from models import ChatRequest, FilterOptions, StarsRange
from retriever import hybrid_search
from llm import stream_answer

_BASE = pathlib.Path(__file__).parent

app = FastAPI(title="GitHub RAG API")

# CORS
origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Pinecone index at startup
_pinecone_index = None
def get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        _pinecone_index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
    return _pinecone_index

# Load filter options at startup
_filter_options: dict = {}
def get_filter_options() -> dict:
    global _filter_options
    if not _filter_options:
        with open(_BASE / "filter_options.json", encoding="utf-8") as f:
            _filter_options = json.load(f)
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
def chat(request: ChatRequest):
    filters = request.filters
    language = filters.language if filters else ""
    min_stars = filters.min_stars if filters else 0
    topics = filters.topics if filters else []

    index = get_pinecone_index()
    docs = hybrid_search(
        query=request.messages[-1].content,
        pinecone_index=index,
        language=language or "",
        min_stars=min_stars or 0,
        topics=topics or [],
        top_k=5
    )

    messages_dicts = [{"role": m.role, "content": m.content} for m in request.messages]
    return StreamingResponse(
        stream_answer(docs, messages_dicts),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

@app.get("/health")
def health():
    return {"status": "ok"}
