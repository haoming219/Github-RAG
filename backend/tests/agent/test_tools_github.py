import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import pytest
from unittest.mock import patch, MagicMock
from agent.tools.github import (
    github_repo_info,
    github_search_code,
    github_get_file,
    _parse_repo_name,
    TRUNCATION_LIMIT,
)


def test_parse_repo_name_owner_slash():
    assert _parse_repo_name("encode/httpx") == "encode/httpx"


def test_parse_repo_name_full_url():
    assert _parse_repo_name("https://github.com/encode/httpx") == "encode/httpx"


def test_parse_repo_name_full_url_trailing_slash():
    assert _parse_repo_name("https://github.com/encode/httpx/") == "encode/httpx"


def _mock_httpx_get(url, **kwargs):
    resp = MagicMock()
    resp.status_code = 200
    if "/repos/encode/httpx" == url.split("api.github.com")[1].rstrip("/"):
        resp.json.return_value = {
            "full_name": "encode/httpx", "description": "A next-gen HTTP client",
            "stargazers_count": 10000, "forks_count": 500, "language": "Python",
            "license": {"spdx_id": "BSD-3-Clause"},
            "pushed_at": "2026-04-01T00:00:00Z", "open_issues_count": 42,
        }
    elif "/readme" in url:
        import base64
        resp.json.return_value = {"content": base64.b64encode(b"# httpx\nAsync HTTP client").decode()}
    elif "/commits" in url:
        resp.json.return_value = [{}] * 15
    elif "/contributors" in url:
        resp.json.return_value = [{"login": "tomchristie"}, {"login": "alex"}]
    else:
        resp.json.return_value = {}
    return resp


def test_github_repo_info_returns_profile():
    with patch("agent.tools.github.httpx.get", side_effect=_mock_httpx_get):
        with patch("agent.tools.github._summarize_readme", return_value="HTTP client library"):
            profile = github_repo_info("encode/httpx")
    assert profile["full_name"] == "encode/httpx"
    assert profile["stars"] == 10000
    assert profile["license"] == "BSD-3-Clause"
    assert profile["top_contributors"] == ["tomchristie", "alex"]


def test_github_repo_info_accepts_url():
    with patch("agent.tools.github.httpx.get", side_effect=_mock_httpx_get):
        with patch("agent.tools.github._summarize_readme", return_value="HTTP client"):
            profile = github_repo_info("https://github.com/encode/httpx")
    assert profile["full_name"] == "encode/httpx"


def test_github_repo_info_rate_limited():
    resp = MagicMock()
    resp.status_code = 429
    with patch("agent.tools.github.httpx.get", return_value=resp):
        result = github_repo_info("encode/httpx")
    assert result["error"] == "rate_limited"


def test_github_search_code_returns_list():
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "items": [
            {
                "path": "src/client.py",
                "html_url": "https://github.com/encode/httpx/blob/main/src/client.py",
                "text_matches": [{"fragment": "class AsyncClient:"}],
            }
        ]
    }
    with patch("agent.tools.github.httpx.get", return_value=resp):
        results = github_search_code("encode/httpx", "AsyncClient")
    assert len(results) == 1
    assert results[0]["file_path"] == "src/client.py"


def test_github_get_file_truncates_long_content():
    import base64
    long_content = "x" * 10000
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"content": base64.b64encode(long_content.encode()).decode()}
    with patch("agent.tools.github.httpx.get", return_value=resp):
        result = github_get_file("encode/httpx", "big_file.py")
    assert "[...内容已截断" in result
    assert len(result) < 10000


def test_github_get_file_short_content_not_truncated():
    import base64
    content = "def hello(): pass"
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"content": base64.b64encode(content.encode()).decode()}
    with patch("agent.tools.github.httpx.get", return_value=resp):
        result = github_get_file("encode/httpx", "small.py")
    assert result == content
