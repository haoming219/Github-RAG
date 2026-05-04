import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from agent.types import RepoResult, RepoProfile, CodeResult

def test_repo_result_keys():
    r: RepoResult = {"repo_name": "a/b", "chunk_text": "x", "score": 0.9, "source": "vector"}
    assert r["score"] == 0.9

def test_repo_profile_keys():
    p: RepoProfile = {
        "full_name": "a/b", "description": "desc", "stars": 100, "forks": 10,
        "language": "Python", "license": "MIT", "readme_summary": "...",
        "last_commit": "2026-01-01", "commits_last_30d": 5,
        "top_contributors": ["alice"], "open_issues_count": 3,
    }
    assert p["stars"] == 100

def test_code_result_keys():
    c: CodeResult = {"file_path": "src/main.py", "snippet": "def f(): ...", "url": "https://github.com/a/b/blob/main/src/main.py"}
    assert "snippet" in c
