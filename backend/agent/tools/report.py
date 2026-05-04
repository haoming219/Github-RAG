from __future__ import annotations
import re
from datetime import date
from pathlib import Path

REPORTS_DIR = Path(__file__).parent.parent.parent.parent / "reports"


def _safe_filename(repo: str) -> str:
    safe = repo.replace("/", "_")
    safe = re.sub(r"[^\w\-]", "", safe)
    return safe


def generate_report(repo: str, content: dict) -> dict:
    """根据收集到的数据生成仓库 Markdown 报告，保存到 reports/ 目录，并返回文件路径与内容。"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    base_name = f"{_safe_filename(repo)}_{today}"
    file_path = REPORTS_DIR / f"{base_name}.md"

    counter = 1
    while file_path.exists():
        file_path = REPORTS_DIR / f"{base_name}_{counter}.md"
        counter += 1

    md = _render_markdown(repo, content)
    file_path.write_text(md, encoding="utf-8")
    relative_path = f"reports/{file_path.name}"
    return {"path": relative_path, "content": md}


def _render_markdown(repo: str, content: dict) -> str:
    profile = content.get("profile", content)
    code_results = content.get("code_results", [])
    kb_results = content.get("kb_results", [])

    stars = profile.get("stars", "N/A")
    forks = profile.get("forks", "N/A")
    language = profile.get("language", "N/A")
    license_ = profile.get("license", "")
    description = profile.get("description", "")
    readme_summary = profile.get("readme_summary", "")
    last_commit = profile.get("last_commit", "")
    commits_30d = profile.get("commits_last_30d", "N/A")
    contributors = ", ".join(profile.get("top_contributors", []))
    open_issues = profile.get("open_issues_count", "N/A")

    code_section = ""
    if code_results:
        code_section = "## 3. 关键代码片段\n\n"
        for r in code_results[:3]:
            code_section += f"**{r.get('file_path', '')}**\n```\n{r.get('snippet', '')}\n```\n[查看源码]({r.get('url', '')})\n\n"

    kb_section = ""
    if kb_results:
        kb_section = "## 5. 知识库相关仓库\n\n"
        for r in kb_results[:3]:
            kb_section += f"- **{r.get('repo_name', '')}** (score: {r.get('score', 0):.2f})\n"
        kb_section += "\n"

    return f"""# 仓库完整画像：{repo}

## 1. 基础信息
- **Stars:** {stars} | **Forks:** {forks} | **主要语言:** {language} | **License:** {license_ or '未知'}
- **最后 Commit:** {last_commit}
- **过去30天 Commit 数:** {commits_30d}

## 2. 项目简介
{description}

{readme_summary}

{code_section}## 4. 社区健康度
- **主要贡献者:** {contributors or '未知'}
- **Open Issues:** {open_issues}

{kb_section}## 6. 总结与推荐
> 以上信息由 ReAct Agent 自动收集整理。
"""
