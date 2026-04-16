import os, json
from openai import OpenAI
from typing import Generator

client = None

def _get_client():
    global client
    if client is None:
        # GLM_API_URL is the full endpoint (e.g. https://aihubmix.com/v1/chat/completions)
        # OpenAI SDK base_url must be the root without the path suffix
        api_url = os.environ["GLM_API_URL"]  # https://aihubmix.com/v1/chat/completions
        base_url = api_url.rsplit("/chat/completions", 1)[0]  # https://aihubmix.com/v1
        client = OpenAI(
            api_key=os.environ["GLM_API_KEY"],
            base_url=base_url,
        )
    return client

SYSTEM_PROMPT = """You are a helpful assistant that recommends open-source GitHub projects.
When given context about relevant repositories, provide clear and natural recommendations.
Summarize each project in plain language. Include project names and why they are relevant to the user's question.
Do not mention how you found the projects or any internal retrieval process."""

def build_prompt_messages(retrieved_docs: list[dict], messages: list[dict]) -> list[dict]:
    """
    Build the message list for GLM API:
    - System prompt
    - Up to last 6 messages from conversation history (excluding the latest user message)
    - Injected context message with retrieved docs
    - Latest user message
    """
    # Truncate history to last 6 messages
    history = messages[-6:] if len(messages) > 6 else messages
    latest_user = history[-1]["content"]
    prior_history = history[:-1]

    # Build context string from retrieved docs
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        snippet = doc.get("content", "")[:800]
        repo_name = doc.get("full_name", "")
        language = doc.get("language", "")
        stars = doc.get("stars", 0)
        description = doc.get("description", "")
        context_parts.append(
            f"[Repo {i+1}] {repo_name} | {language} | \u2605{stars}\n"
            f"Description: {description}\n"
            f"Details: {snippet}"
        )
    context_str = "\n\n".join(context_parts)

    prompt_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    prompt_messages.extend(prior_history)
    prompt_messages.append({
        "role": "user",
        "content": f"Here are some relevant GitHub repositories:\n\n{context_str}\n\nQuestion: {latest_user}"
    })
    return prompt_messages

def stream_answer(retrieved_docs: list[dict], messages: list[dict]) -> Generator[str, None, None]:
    """Yield SSE-formatted strings for each chunk from GLM API."""
    if not retrieved_docs:
        yield 'data: {"text": "\u672a\u627e\u5230\u7b26\u5408\u8fc7\u6ee4\u6761\u4ef6\u7684\u9879\u76ee\uff0c\u8bf7\u653e\u5bbd\u7b5b\u9009\u6761\u4ef6\u3002"}\n\n'
        yield "data: [DONE]\n\n"
        return

    prompt_messages = build_prompt_messages(retrieved_docs, messages)
    c = _get_client()
    response = c.chat.completions.create(
        model=os.environ["GLM_MODEL_ID"],
        messages=prompt_messages,
        stream=True,
        max_tokens=1024,
    )
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            yield f"data: {json.dumps({'text': delta.content}, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"
