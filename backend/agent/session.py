from __future__ import annotations
import asyncio
import time
from typing import Callable, Any


class SessionManager:
    """进程内 Session 存储，支持 TTL 过期和并发请求队列化。"""

    def __init__(self, ttl_seconds: int = 30 * 60):
        self._ttl = ttl_seconds
        self._sessions: dict[str, dict] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def get_or_create(self, session_id: str, factory: Callable[[], Any]) -> Any:
        entry = self._sessions.get(session_id)
        if entry and (time.time() - entry["last_active"]) < self._ttl:
            return entry["agent"]
        agent = factory()
        self._sessions[session_id] = {"agent": agent, "last_active": time.time()}
        return agent

    def touch(self, session_id: str) -> None:
        if session_id in self._sessions:
            self._sessions[session_id]["last_active"] = time.time()

    async def get_lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]
