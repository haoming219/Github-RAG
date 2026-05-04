from typing import TypedDict


class RepoResult(TypedDict):
    repo_name: str
    chunk_text: str
    score: float
    source: str


class RepoProfile(TypedDict):
    full_name: str
    description: str
    stars: int
    forks: int
    language: str
    license: str
    readme_summary: str
    last_commit: str
    commits_last_30d: int
    top_contributors: list[str]
    open_issues_count: int


class CodeResult(TypedDict):
    file_path: str
    snippet: str
    url: str
