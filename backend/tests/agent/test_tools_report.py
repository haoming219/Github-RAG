import sys, pathlib, shutil
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import pytest
from unittest.mock import patch
from agent.tools.report import generate_report, _safe_filename


def test_safe_filename_replaces_slash():
    assert _safe_filename("encode/httpx") == "encode_httpx"


def test_safe_filename_no_path_traversal():
    name = _safe_filename("../evil/repo")
    assert ".." not in name
    assert "/" not in name


def test_generate_report_creates_file(tmp_path):
    with patch("agent.tools.report.REPORTS_DIR", tmp_path):
        result = generate_report("encode/httpx", {"stars": 10000, "description": "HTTP client"})
    assert "path" in result
    assert "content" in result
    actual_file = tmp_path / pathlib.Path(result["path"]).name
    assert actual_file.exists()
    assert "encode_httpx" in result["path"]


def test_generate_report_no_collision(tmp_path):
    with patch("agent.tools.report.REPORTS_DIR", tmp_path):
        r1 = generate_report("encode/httpx", {"description": "v1"})
        r2 = generate_report("encode/httpx", {"description": "v2"})
    assert r1["path"] != r2["path"]


def test_generate_report_content_includes_repo_name(tmp_path):
    with patch("agent.tools.report.REPORTS_DIR", tmp_path):
        result = generate_report("encode/httpx", {"description": "async http", "stars": 5000})
    assert "encode/httpx" in result["content"]
