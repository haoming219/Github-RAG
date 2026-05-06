from __future__ import annotations
from openai import OpenAI

_FIRST_TURN_SYSTEM = (
    "You are a search query optimizer for a GitHub repository search engine. "
    "The search index contains English text. "
    "Given the user's question (which may be in any language), "
    "extract the core technical intent and rewrite it as a concise English keyword phrase "
    "suitable for semantic and BM25 retrieval. "
    "Output ONLY the rewritten query. No explanation. No punctuation at the end."
)

_MULTI_TURN_SYSTEM = (
    "You are a search query optimizer for a GitHub repository search engine. "
    "The search index contains English text. "
    "Given a conversation history and the user's latest question, "
    "resolve any pronouns or references (e.g. '它', 'this', 'that project') "
    "using the conversation context, then rewrite the question as a concise, "
    "self-contained English keyword phrase suitable for semantic and BM25 retrieval. "
    "Output ONLY the rewritten query. No explanation. No punctuation at the end."
)


class QueryRewriter:
    def __init__(self, model: str, api_key: str, api_base: str, timeout: float = 5.0):
        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=api_base, timeout=timeout)

    def rewrite(self, query: str, history: list[dict]) -> str:
        try:
            if not history:
                return self._rewrite_first_turn(query)
            return self._rewrite_multi_turn(query, history)
        except Exception:
            return query

    def _rewrite_first_turn(self, query: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _FIRST_TURN_SYSTEM},
                {"role": "user", "content": query},
            ],
            max_tokens=64,
            temperature=0,
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten if rewritten else query

    def _rewrite_multi_turn(self, query: str, history: list[dict]) -> str:
        recent = history[-30:]
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in recent
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _MULTI_TURN_SYSTEM},
                {"role": "user", "content": f"Conversation history:\n{history_text}\n\nLatest question: {query}"},
            ],
            max_tokens=64,
            temperature=0,
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten if rewritten else query
