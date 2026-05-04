import sys, pathlib, types
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from unittest.mock import MagicMock, patch
from agent.tools.knowledge_base import search_knowledge_base


def _make_node(repo_name, text, score):
    node = MagicMock()
    node.node.metadata = {
        "full_name": repo_name,
        "language": "Python",
        "stars": 500,
        "topics": ["web"],
    }
    node.node.get_content.return_value = text
    node.score = score
    return node


def test_returns_list_of_repo_results():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        _make_node("a/repo", "some text", 0.9),
        _make_node("b/repo", "other text", 0.7),
    ]
    with patch("agent.tools.knowledge_base._get_retriever", return_value=mock_retriever):
        results = search_knowledge_base("async task queue")
    assert len(results) == 2
    assert results[0]["repo_name"] == "a/repo"
    assert results[0]["score"] == 0.9
    assert results[0]["source"] == "hybrid"


def test_returns_at_most_5():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        _make_node(f"repo/{i}", "text", 0.9 - i * 0.1) for i in range(8)
    ]
    with patch("agent.tools.knowledge_base._get_retriever", return_value=mock_retriever):
        results = search_knowledge_base("query")
    assert len(results) <= 5


def test_sorted_by_score_descending():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        _make_node("low/repo", "text", 0.3),
        _make_node("high/repo", "text", 0.9),
    ]
    with patch("agent.tools.knowledge_base._get_retriever", return_value=mock_retriever):
        results = search_knowledge_base("query")
    assert results[0]["score"] >= results[1]["score"]
