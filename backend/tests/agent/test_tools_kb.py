import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from unittest.mock import MagicMock, patch


def _make_node(repo_name, text, score):
    node = MagicMock()
    node.node.metadata = {"full_name": repo_name, "language": "Python", "stars": 500, "topics": ["web"]}
    node.node.get_content.return_value = text
    node.score = score
    return node


def _mock_retriever(nodes):
    r = MagicMock()
    r.retrieve.return_value = nodes
    return r


# ── 基础行为测试 ───────────────────────────────────────────────────────────────

def test_returns_formatted_string_with_repo_names():
    from agent.tools.knowledge_base import search_knowledge_base
    mock_retriever = _mock_retriever([
        _make_node("a/repo", "some text", 0.9),
        _make_node("b/repo", "other text", 0.7),
    ])
    with patch("agent.tools.knowledge_base._get_retriever", return_value=mock_retriever), \
         patch("agent.tools.knowledge_base._get_rewriter", return_value=None):
        result = search_knowledge_base("async task queue")
    assert "a/repo" in result
    assert "b/repo" in result


def test_returns_at_most_5_repos():
    from agent.tools.knowledge_base import search_knowledge_base
    mock_retriever = _mock_retriever([
        _make_node(f"repo/{i}", "text", 0.9 - i * 0.1) for i in range(8)
    ])
    with patch("agent.tools.knowledge_base._get_retriever", return_value=mock_retriever), \
         patch("agent.tools.knowledge_base._get_rewriter", return_value=None):
        result = search_knowledge_base("query")
    # 最多5条结果，用分隔符数量验证：5条结果有4个 "\n\n"
    assert result.count("\n\n") <= 4


def test_empty_results_returns_not_found_message():
    from agent.tools.knowledge_base import search_knowledge_base
    mock_retriever = _mock_retriever([])
    with patch("agent.tools.knowledge_base._get_retriever", return_value=mock_retriever), \
         patch("agent.tools.knowledge_base._get_rewriter", return_value=None):
        result = search_knowledge_base("query")
    assert "未找到" in result


# ── QueryRewriter 集成测试 ─────────────────────────────────────────────────────

def test_rewriter_rewrites_query_before_retrieval():
    """rewriter 存在时，retriever 收到改写后的 query。"""
    from agent.tools.knowledge_base import search_knowledge_base
    mock_retriever = _mock_retriever([_make_node("a/repo", "text", 0.9)])
    mock_rewriter = MagicMock()
    mock_rewriter.rewrite.return_value = "rewritten query"

    with patch("agent.tools.knowledge_base._get_retriever", return_value=mock_retriever), \
         patch("agent.tools.knowledge_base._get_rewriter", return_value=mock_rewriter):
        search_knowledge_base("original query")

    mock_retriever.retrieve.assert_called_once_with("rewritten query")


def test_no_rewriter_uses_original_query():
    """rewriter 为 None 时，retriever 收到原始 query。"""
    from agent.tools.knowledge_base import search_knowledge_base
    mock_retriever = _mock_retriever([_make_node("a/repo", "text", 0.9)])

    with patch("agent.tools.knowledge_base._get_retriever", return_value=mock_retriever), \
         patch("agent.tools.knowledge_base._get_rewriter", return_value=None):
        search_knowledge_base("original query")

    mock_retriever.retrieve.assert_called_once_with("original query")


def test_rewriter_receives_current_history():
    """rewriter.rewrite 被调用时收到当前注入的 history。"""
    from agent.tools.knowledge_base import search_knowledge_base, set_conversation_history
    mock_retriever = _mock_retriever([])
    mock_rewriter = MagicMock()
    mock_rewriter.rewrite.return_value = "query"
    history = [{"role": "user", "content": "previous message"}]

    with patch("agent.tools.knowledge_base._get_retriever", return_value=mock_retriever), \
         patch("agent.tools.knowledge_base._get_rewriter", return_value=mock_rewriter):
        set_conversation_history(history)
        search_knowledge_base("new query")

    mock_rewriter.rewrite.assert_called_once_with("new query", history)


def test_two_calls_in_same_turn_use_same_history():
    """同一 turn 内两次调用 search_knowledge_base，rewriter 均收到相同 history。"""
    from agent.tools.knowledge_base import search_knowledge_base, set_conversation_history
    mock_retriever = _mock_retriever([])
    mock_rewriter = MagicMock()
    mock_rewriter.rewrite.return_value = "query"
    history = [{"role": "user", "content": "context"}]

    with patch("agent.tools.knowledge_base._get_retriever", return_value=mock_retriever), \
         patch("agent.tools.knowledge_base._get_rewriter", return_value=mock_rewriter):
        set_conversation_history(history)
        search_knowledge_base("first call")
        search_knowledge_base("second call")

    assert mock_rewriter.rewrite.call_count == 2
    for call_args in mock_rewriter.rewrite.call_args_list:
        assert call_args.args[1] == history


def test_contextvar_isolation_in_executor_context():
    """验证 ctx.run() 模式下不同 context 的 ContextVar 互不干扰。"""
    import asyncio
    import contextvars
    from agent.tools.knowledge_base import set_conversation_history, _conversation_history_var, _get_history
    set_conversation_history([])  # 重置主 context

    results = {}

    async def run_two_sessions():
        # 在任何 ctx.run 之前先同时 copy 两个 context
        ctx_a = contextvars.copy_context()
        ctx_b = contextvars.copy_context()

        def set_a():
            set_conversation_history([{"role": "user", "content": "hello from A"}])
            results["a"] = _get_history()
        ctx_a.run(set_a)

        def read_and_set_b():
            # ctx_b 是在 ctx_a.run() 之前 copy 的，所以不应该看到 ctx_a 的写入
            results["b_before_set"] = _get_history()
            set_conversation_history([{"role": "user", "content": "hello from B"}])
            results["b"] = _get_history()
        ctx_b.run(read_and_set_b)

    asyncio.run(run_two_sessions())
    assert results["a"][0]["content"] == "hello from A"
    assert results["b"][0]["content"] == "hello from B"
    # 关键：ctx_b 在自己 set 之前，不应该看到 ctx_a 的值
    assert results["b_before_set"] == []
    # 主 context 不受影响
    assert _get_history() == []
