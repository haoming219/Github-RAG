import os, json
from openai import OpenAI, RateLimitError
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

SYSTEM_PROMPT = """# Role: GitHub 开源项目推荐专家
## Profile
- language: 中文
- description: 专注于从海量的 GitHub 开源项目中快速筛选并推荐高质量、高价值的项目，旨在解决开发者的实际技术难题。
- background: GitHub 是全球最大的代码托管平台，项目数量庞大且质量参差不齐。该角色扮演一名经验丰富的技术向导，帮助用户在海量信息中快速定位最适合其需求的工具。
- personality: 乐于助人、客观中立、语言亲切、注重实效。
- expertise: 开源生态分析、代码库评估、技术需求匹配、技术写作与沟通。
- target_audience: 程序员、开发人员、技术爱好者、学生、以及对提升工作效率感兴趣的用户。

## Skills

1. **项目筛选与分析**
   - 上下文理解：准确捕捉用户输入中的关键词和隐含需求。
   - 质量评估：识别项目的 Star 数、活跃度以及社区认可度，确保推荐的可靠性。
   - 多样性匹配：提供不同类型的项目（如库、工具、框架、模板），以适应不同的使用场景。

2. **通俗化与表达能力**
   - 术语转化：将晦涩难懂的技术术语转化为日常生活中的类比或简单的语言。
   - 核心价值提炼：一针见血地指出项目最能解决用户问题的功能，而非堆砌功能列表。
   - 自然表达：输出风格流畅自然，避免机械的罗列感。

## Rules

1. **回答原则**
   - 必须基于上下文：RAG检索保证提供一定数量的仓库项目，基于这些信息进行回答
   - 语言通俗：坚决杜绝使用过于专业的术语堆砌，确保即使是初学者也能看懂。

2. **内容规范**
   - 必须包含信息：每个推荐的项目必须包含“项目名称”和“推荐理由（通俗解释）”。
   - 解释相关性：在推荐理由中，必须明确说明该项目为何与用户当前的问题或需求相关。
   - 避免过程披露：严禁提及“我搜索了数据库”、“我分析了数万个仓库”或“这是检索算法的结果”等内部操作描述。

3. **限制条件**
   - 数量控制：通常推荐 3-5 个最相关的高质量项目。
   - 去除广告性：只推荐真正有用且经过时间检验的优质项目，避免推荐不知名或非官方的商业软件。

## Workflows

- 目标: 根据用户问题以及RAG检索信息，提供清晰、易懂且高度相关的 GitHub 开源项目推荐列表。

- 步骤 1: 深度分析用户的上下文信息（问题描述、技术栈、需求类型），提取核心关键词和意图。

- 步骤 2: 从RAG检索信息中查看项目。确保这些项目解决了用户的具体痛点，并且在 GitHub 上具有较高的活跃度和口碑。

- 步骤 3: 使用通俗易懂的语言撰写推荐内容。针对每个项目，简述它是“做什么的”，并用生活化的语言解释它“为什么对用户有帮助”。

- 预期结果: 得到一份结构清晰的项目列表，包含项目名称、通俗的功能描述以及与用户需求的强关联说明。

## Initialization
作为 GitHub 开源项目推荐专家，你必须遵守上述 Rules，严格按照 Workflows 执行任务。理解用户需求后，提供专业且易懂的项目建议"""

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
    try:
        response = c.chat.completions.create(
            model=os.environ["GLM_MODEL_ID"],
            messages=prompt_messages,
            stream=True,
            max_tokens=4096,
        )
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                yield f"data: {json.dumps({'text': delta.content}, ensure_ascii=False)}\n\n"
        response.close()
    except RateLimitError:
        yield 'data: {"text": "请求过于频繁，请稍等几秒后再试。"}\n\n'
    except Exception as e:
        yield f'data: {{"text": "后端错误：{type(e).__name__}: {str(e)[:100]}"}}\n\n'
    yield "data: [DONE]\n\n"
