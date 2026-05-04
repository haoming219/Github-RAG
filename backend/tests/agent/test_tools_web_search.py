import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from unittest.mock import patch, MagicMock
from agent.tools.web_search import web_search


def _mock_search_results():
    return {
        "organic_results": [
            {
                "title": "httpx: A next-gen HTTP client",
                "snippet": "httpx is a fully featured HTTP client for Python 3.",
                "link": "https://www.python-httpx.org/",
            },
            {
                "title": "encode/httpx GitHub",
                "snippet": "A next generation HTTP client for Python.",
                "link": "https://github.com/encode/httpx",
            },
        ]
    }


def test_returns_formatted_string(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "fake-key")
    with patch("agent.tools.web_search.GoogleSearch") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.get_dict.return_value = _mock_search_results()
        mock_cls.return_value = mock_instance
        result = web_search("httpx Python HTTP client")
    assert "httpx: A next-gen HTTP client" in result
    assert "https://www.python-httpx.org/" in result


def test_returns_at_most_5_results(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "fake-key")
    many_results = {
        "organic_results": [
            {"title": f"Result {i}", "snippet": "desc", "link": f"https://example.com/{i}"}
            for i in range(10)
        ]
    }
    with patch("agent.tools.web_search.GoogleSearch") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.get_dict.return_value = many_results
        mock_cls.return_value = mock_instance
        result = web_search("query")
    assert result.count("https://") <= 5


def test_handles_empty_results(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "fake-key")
    with patch("agent.tools.web_search.GoogleSearch") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.get_dict.return_value = {"organic_results": []}
        mock_cls.return_value = mock_instance
        result = web_search("very obscure query")
    assert "未找到" in result or result.strip() == ""


def test_handles_missing_api_key(monkeypatch):
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    result = web_search("test query")
    assert "[web_search 错误" in result


def test_handles_api_exception():
    with patch("agent.tools.web_search.GoogleSearch") as mock_cls:
        mock_cls.side_effect = Exception("network error")
        result = web_search("query")
    assert "[web_search 错误" in result
