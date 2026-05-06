from __future__ import annotations
import os

from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

from agent.prompts import REACT_AGENT_SYSTEM_PROMPT
from agent.tools.knowledge_base import search_knowledge_base
from agent.tools.github import github_repo_info, github_search_code, github_get_file
from agent.tools.web_search import web_search
from agent.tools.report import generate_report

# Each tool call produces ~3 reasoning steps (Thought + Action + Observation).
# Set this to 3× the desired max tool calls per turn.
MAX_TOOL_CALLS_PER_TURN = 6
MAX_ITERATIONS = MAX_TOOL_CALLS_PER_TURN * 3


def _make_tools() -> list[FunctionTool]:
    return [
        FunctionTool.from_defaults(fn=search_knowledge_base),
        FunctionTool.from_defaults(fn=github_repo_info),
        FunctionTool.from_defaults(fn=github_search_code),
        FunctionTool.from_defaults(fn=github_get_file),
        FunctionTool.from_defaults(fn=web_search),
        FunctionTool.from_defaults(fn=generate_report),
    ]


def create_agent() -> ReActAgent:
    llm = OpenAI(
        model=os.getenv("LLM_MODEL_ID", "gpt-4o-mini"),
        api_key=os.getenv("LLM_API_KEY", ""),
        api_base=os.getenv("LLM_API_URL", ""),
    )
    return ReActAgent.from_tools(
        _make_tools(),
        llm=llm,
        system_prompt=REACT_AGENT_SYSTEM_PROMPT,
        max_iterations=MAX_ITERATIONS,
        verbose=True,
    )
