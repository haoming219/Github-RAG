import os, json, pathlib, asyncio, logging, contextvars
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pinecone import Pinecone
from pydantic import BaseModel

from llama_index.core.callbacks import CallbackManager, CBEventType
from llama_index.core.callbacks.base import BaseCallbackHandler
from llama_index.core.callbacks.schema import EventPayload
from llama_index.core.agent import AgentChatResponse
from llama_index.core.chat_engine.types import StreamingAgentChatResponse

from models import FilterOptions, StarsRange
from retriever import load_retriever
from agent.agent import create_agent
from agent.session import SessionManager
from agent.tools.knowledge_base import init_retriever as init_kb_retriever
from agent.tools.knowledge_base import set_conversation_history

_session_manager = SessionManager(ttl_seconds=30 * 60)

_BASE = pathlib.Path(__file__).parent

_retriever = None
_filter_options: dict = {}


class AgentChatRequest(BaseModel):
    session_id: str
    message: str


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

    init_kb_retriever(_retriever)
    _llm_api_key = os.getenv("LLM_API_KEY")
    _llm_api_base = os.getenv("LLM_API_URL")
    if not _llm_api_key or not _llm_api_base:
        logging.warning("[startup] LLM_API_KEY 或 LLM_API_URL 未配置，QueryRewriter 已禁用")
    else:
        try:
            from agent.tools.query_rewriter import QueryRewriter
            from agent.tools.knowledge_base import init_rewriter
            _rewriter = QueryRewriter(
                model=os.getenv("QUERY_MODEL_ID", "gpt-4o-mini"),
                api_key=_llm_api_key,
                api_base=_llm_api_base,
            )
            init_rewriter(_rewriter)
            print("[startup] QueryRewriter initialized.", flush=True)
        except Exception as e:
            logging.warning(f"[startup] QueryRewriter 初始化失败，查询改写已禁用: {e}")
    if not os.getenv("GITHUB_TOKEN"):
        logging.warning("GITHUB_TOKEN not set — GitHub API rate limit: 60 req/hour (anonymous)")

    print("[startup] Ready.", flush=True)
    yield


app = FastAPI(title="GitHub RAG API", lifespan=lifespan)

origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.get("/api/filters/options", response_model=FilterOptions)
def filters_options():
    sr = _filter_options.get("stars_range", {"min": 0, "max": 0})
    return FilterOptions(
        languages=_filter_options.get("languages", []),
        topics=_filter_options.get("topics", []),
        stars_range=StarsRange(min=sr["min"], max=sr["max"]),
    )


# @app.post("/api/chat")
# async def chat(request: ChatRequest):
#     import time
#     t0 = time.perf_counter()
#
#     filters = request.filters
#     language = filters.language if filters else ""
#     min_stars = filters.min_stars if filters else 0
#     topics = filters.topics if filters else []
#
#     _retriever.language = language or ""
#     _retriever.min_stars = min_stars or 0
#     _retriever.topics = topics or []
#
#     query = request.messages[-1].content
#     t1 = time.perf_counter()
#     nodes = _retriever.retrieve(query)
#     t2 = time.perf_counter()
#     print(f"[timing] retrieve: {t2 - t1:.2f}s | results: {len(nodes)}", flush=True)
#
#     docs = [n.metadata for n in nodes]
#     messages_dicts = [{"role": m.role, "content": m.content} for m in request.messages]
#
#     def timed_stream():
#         import time
#         first_chunk = True
#         t_start = time.perf_counter()
#         try:
#             for chunk in stream_answer(docs, messages_dicts):
#                 if first_chunk:
#                     print(f"[timing] time to first LLM chunk: {time.perf_counter() - t_start:.2f}s", flush=True)
#                     first_chunk = False
#                 yield chunk
#         except Exception as e:
#             logging.exception("stream_answer error")
#             yield f'data: {{"text": "流式响应错误：{type(e).__name__}: {str(e)[:100]}"}}\n\n'
#             yield "data: [DONE]\n\n"
#         print(f"[timing] total stream: {time.perf_counter() - t_start:.2f}s", flush=True)
#         print(f"[timing] total request: {time.perf_counter() - t0:.2f}s", flush=True)
#
#     return StreamingResponse(
#         timed_stream(),
#         media_type="text/event-stream",
#         headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
#     )


@app.get("/health")
def health():
    return {"status": "ok"}


class _SseStepHandler(BaseCallbackHandler):
    """将 LlamaIndex 工具调用事件写入异步队列，供 SSE 流读取。"""

    TOOL_MESSAGES = {
        "search_knowledge_base": "正在查询知识库...",
        "github_repo_info": "正在从 GitHub 获取仓库信息：{input}",
        "github_search_code": "正在搜索代码：{input}",
        "github_get_file": "正在读取文件：{input}",
        "web_search": "正在搜索互联网：{input}",
    }

    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self._queue = queue
        self._loop = loop

    def on_event_start(self, event_type, payload=None, **kwargs):
        if event_type == CBEventType.FUNCTION_CALL:
            p = payload or {}
            tool_meta = p.get(EventPayload.TOOL)
            tool_name = tool_meta.get_name() if tool_meta else ""
            tool_input = str(p.get(EventPayload.FUNCTION_CALL, ""))[:80]
            tpl = self.TOOL_MESSAGES.get(tool_name, f"正在调用工具：{tool_name}")
            msg = tpl.format(input=tool_input)
            print(f"[agent_step] {msg}", flush=True)
            self._loop.call_soon_threadsafe(
                self._queue.put_nowait,
                json.dumps({"type": "agent_step", "content": msg}, ensure_ascii=False),
            )

    def on_event_end(self, event_type, payload=None, **kwargs):
        if event_type == CBEventType.FUNCTION_CALL:
            output = str((payload or {}).get(EventPayload.FUNCTION_OUTPUT, ""))[:300]
            print(f"[agent_tool_output] {output}", flush=True)

    def start_trace(self, trace_id=None): pass
    def end_trace(self, trace_id=None, trace_map=None): pass


@app.post("/agent/chat")
async def agent_chat(request: AgentChatRequest):
    session_id = request.session_id
    lock = await _session_manager.get_lock(session_id)

    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()
        handler = _SseStepHandler(queue, asyncio.get_running_loop())

        async with lock:
            agent = _session_manager.get_or_create(session_id, factory=create_agent)
            cb = CallbackManager([handler])
            agent.agent_worker.callback_manager = cb
            _session_manager.touch(session_id)

            _history = [
                {"role": str(m.role.value if hasattr(m.role, "value") else m.role),
                 "content": m.content}
                for m in agent.chat_history
                if str(m.role.value if hasattr(m.role, "value") else m.role) in ("user", "assistant")
            ]
            set_conversation_history(_history)

            try:
                loop = asyncio.get_running_loop()
                ctx = contextvars.copy_context()
                fut = loop.run_in_executor(None, lambda: ctx.run(agent.stream_chat, request.message))

                # Drain agent_step events while the agent is running (tool calls phase)
                while not fut.done():
                    try:
                        msg = queue.get_nowait()
                        yield f"data: {msg}\n\n"
                    except asyncio.QueueEmpty:
                        await asyncio.sleep(0.05)

                while not queue.empty():
                    msg = queue.get_nowait()
                    yield f"data: {msg}\n\n"

                response: StreamingAgentChatResponse = await fut

                # Stream the final LLM response token by token
                for token in response.response_gen:
                    yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/agent/reports/{filename}")
async def get_report(filename: str):
    from fastapi import HTTPException
    from fastapi.responses import PlainTextResponse
    reports_dir = Path(__file__).parent.parent / "reports"
    safe_name = Path(filename).name
    file_path = reports_dir / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="报告不存在")
    return PlainTextResponse(file_path.read_text(encoding="utf-8"))
