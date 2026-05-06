import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from unittest.mock import MagicMock, patch


def _make_rewriter():
    from agent.tools.query_rewriter import QueryRewriter
    return QueryRewriter(
        model="gpt-4o-mini",
        api_key="test-key",
        api_base="https://api.example.com/v1",
    )


def _mock_completion(text: str):
    """返回一个模拟 OpenAI chat completion 响应。"""
    choice = MagicMock()
    choice.message.content = text
    response = MagicMock()
    response.choices = [choice]
    return response


def test_first_turn_calls_llm_and_returns_rewritten_query():
    rewriter = _make_rewriter()
    with patch.object(rewriter._client.chat.completions, "create",
                      return_value=_mock_completion("Python task queue async worker")) as mock_create:
        result = rewriter.rewrite("有没有能管任务的工具", history=[])

    assert result == "Python task queue async worker"
    mock_create.assert_called_once()
    call_messages = mock_create.call_args.kwargs["messages"]
    assert not any("history" in str(m).lower() for m in call_messages if m["role"] == "system")


def test_multi_turn_includes_history_in_prompt():
    rewriter = _make_rewriter()
    history = [
        {"role": "user", "content": "推荐一个 Python 异步任务队列"},
        {"role": "assistant", "content": "推荐 Celery，它是一个分布式任务队列。"},
    ]
    with patch.object(rewriter._client.chat.completions, "create",
                      return_value=_mock_completion("Celery distributed deployment")) as mock_create:
        result = rewriter.rewrite("它支持分布式部署吗", history=history)

    assert result == "Celery distributed deployment"
    call_messages = mock_create.call_args.kwargs["messages"]
    user_msg = next(m for m in call_messages if m["role"] == "user")
    assert "Celery" in user_msg["content"]
    assert "它支持分布式部署吗" in user_msg["content"]


def test_multi_turn_truncates_history_to_last_30():
    rewriter = _make_rewriter()
    history = [{"role": "user", "content": f"msg {i}"} for i in range(50)]
    with patch.object(rewriter._client.chat.completions, "create",
                      return_value=_mock_completion("rewritten")) as mock_create:
        rewriter.rewrite("latest question", history=history)

    call_messages = mock_create.call_args.kwargs["messages"]
    user_msg = next(m for m in call_messages if m["role"] == "user")
    # 保留 history[-30:]，即 msg 20..49，msg 19 不应出现
    assert "msg 19" not in user_msg["content"]
    assert "msg 20" in user_msg["content"]


def test_llm_exception_returns_original_query():
    rewriter = _make_rewriter()
    with patch.object(rewriter._client.chat.completions, "create",
                      side_effect=Exception("API error")):
        result = rewriter.rewrite("original query", history=[])
    assert result == "original query"


def test_empty_llm_response_returns_original_query():
    rewriter = _make_rewriter()
    with patch.object(rewriter._client.chat.completions, "create",
                      return_value=_mock_completion("")):
        result = rewriter.rewrite("original query", history=[])
    assert result == "original query"
