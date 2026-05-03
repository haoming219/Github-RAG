# ReAct Agent 设计文档

**日期：** 2026-05-03  
**状态：** 已批准

---

## 1. 背景与目标

当前系统是一个针对 GitHub 仓库的 RAG 检索系统，能从向量知识库中检索相关仓库和代码片段。本次升级目标是将其扩展为一个**仓库推荐与深度调研 Agent**，使其能够：

- 根据用户模糊的概念问题，从知识库中推荐相关仓库
- 当知识库信息不足时，自动调用外部工具（GitHub API、网页抓取等）补充信息
- 当用户直接提供 GitHub 链接时，自动触发完整分析流程
- 最终能为用户生成一份结构化的完整仓库画像报告

---

## 2. 架构方案：ReAct Agent

采用 **ReAct（Reason → Act → Observe）** 架构，使用 LlamaIndex 原生 `ReActAgent`。LLM 在每一步自主决定：是否需要调用工具、调用哪个工具、何时信息足够可以回答用户。

**核心原则：**
- 知识库优先：先查知识库，不足时再用外部工具补充
- 现有 `retriever.py` / `llm.py` 完全不改动，Agent 是叠加在上面的新层
- 工具调用上限：单次对话轮次最多 8 个工具调用，防止无限循环

---

## 3. 目录结构

```
backend/
├── agent/
│   ├── agent.py          # ReAct Agent 主入口，工具注册与对话循环
│   ├── tools/
│   │   ├── knowledge_base.py   # 封装现有 retriever.py
│   │   ├── github.py           # GitHub API 工具集
│   │   ├── fetch_url.py        # 网页内容抓取
│   │   └── report.py           # 报告生成与保存
│   └── prompts.py        # Agent system prompt
├── main.py               # FastAPI，新增 /agent/chat 端点
├── retriever.py          # 不变
└── llm.py                # 不变
```

---

## 4. 工具集

### 数据结构定义

```python
class RepoResult(TypedDict):
    repo_name: str        # "owner/repo" 格式
    chunk_text: str       # 相关代码或文档片段
    score: float          # 相关度分数（0.0 - 1.0）
    source: str           # "vector" | "bm25" | "hybrid"

class RepoProfile(TypedDict):
    full_name: str        # "owner/repo"
    description: str
    stars: int
    forks: int
    language: str
    license: str          # SPDX 标识，如 "MIT"，无则为 ""
    readme_summary: str   # LLM 提炼的 README 摘要，200字以内
    last_commit: str      # ISO 8601 日期字符串
    commits_last_30d: int
    top_contributors: list[str]   # GitHub 用户名列表，最多5个
    open_issues_count: int

class CodeResult(TypedDict):
    file_path: str        # 仓库内相对路径
    snippet: str          # 匹配的代码片段（上下文各3行）
    url: str              # GitHub 文件 URL
```

### 4.1 `search_knowledge_base(query: str) → list[RepoResult]`
- 调用现有混合检索（向量 + BM25）
- 返回最多 5 条结果，按 score 降序排列

### 4.2 `github_repo_info(repo_url_or_name: str) → RepoProfile`
- 接受 `"owner/repo"` 或完整 GitHub URL（自动解析）
- 调用 GitHub REST API `GET /repos/{owner}/{repo}`
- README 内容通过 `GET /repos/{owner}/{repo}/readme` 获取后由 LLM 摘要
- 认证：通过环境变量 `GITHUB_TOKEN` 提供 Personal Access Token（Bearer token）；未设置则以匿名方式请求（速率限制 60次/小时）
- 错误处理：HTTP 429/403 返回 `{"error": "rate_limited", "message": "GitHub API 速率限制，请稍后重试"}` 并终止当前工具调用链

### 4.3 `github_search_code(repo: str, query: str) → list[CodeResult]`
- `repo` 格式为 `"owner/repo"`
- 调用 GitHub Code Search API `GET /search/code?q={query}+repo:{repo}`
- 返回最多 5 条结果
- 同上，认证与错误处理策略与 4.2 一致

### 4.4 `github_get_file(repo: str, path: str) → str`
- 拉取指定文件原始内容（`GET /repos/{owner}/{repo}/contents/{path}`）
- 长度限制：返回内容超过 **8000 字符**时，截取前 4000 字符 + 末尾 4000 字符，中间插入 `\n[...内容已截断，共 {total} 字符...]\n`
- 不额外调用 LLM 做摘要，截断标记足以让 Agent 判断是否需要进一步查询特定部分

### 4.5 `fetch_url(url: str) → str`
- 抓取任意 URL 页面内容并提取纯文本（去除 HTML 标签）
- **SSRF 防护：** 拒绝以下请求，返回错误字符串 `"[fetch_url 错误：不允许访问该地址]"`：
  - 私有 IP 范围（`10.x`, `172.16-31.x`, `192.168.x`, `127.x`, `169.254.x`）
  - 非 HTTP/HTTPS 协议
  - 仅允许公网可路由 IP 或公共域名
- 长度限制：返回内容超过 **8000 字符**时，截取前 8000 字符并追加 `[内容已截断]`

### 4.6 `generate_report(repo: str, content: dict) → str`
- `repo`：`"owner/repo"` 格式，用于生成文件名
- `content`：包含各工具收集到的数据，Agent 负责在调用前填充
- 文件名格式：`{owner}_{repo}_{YYYY-MM-DD}.md`（`/` 替换为 `_`，禁止路径穿越字符）
- 保存路径：项目根目录下的 `reports/` 目录（固定，不可配置）
- 文件名冲突时追加 `_1`, `_2` 后缀
- 返回：`{"path": "reports/encode_httpx_2026-05-03.md", "content": "..."}`

---

## 5. 典型对话流程

### 场景 A：模糊概念推荐

```
用户："有没有好用的 Python 异步任务队列？"
  → search_knowledge_base("Python 异步任务队列")
  → 返回候选仓库列表，向用户推荐

用户："告诉我更多关于 Dramatiq 的信息"
  → github_repo_info("Bogdanp/dramatiq")
  → github_search_code("Bogdanp/dramatiq", "middleware")
  → 综合回答

用户："帮我生成一份完整报告"
  → generate_report(...)
  → 返回报告路径
```

### 场景 B：直接提供 GitHub 链接（自动触发完整分析）

```
用户："帮我分析这个仓库：https://github.com/encode/httpx"
  → github_repo_info("encode/httpx")
  → github_get_file("encode/httpx", "README.md")
  → github_search_code("encode/httpx", "核心模块")
  → search_knowledge_base("httpx")
  → generate_report(...)
  → 返回完整报告，无需用户额外确认
```

---

## 6. System Prompt 核心指令

1. 优先调用 `search_knowledge_base`，知识库结果不足时再调用 GitHub 工具补充
2. 当用户提供 GitHub 仓库链接时，自动触发完整分析流程（依次：仓库信息 → README → 代码搜索 → 知识库 → 生成报告），**整个过程无需用户额外确认**
3. 对于场景 A（推荐 + 问答流程），生成报告前需确认用户是否需要
4. 每次工具调用后判断信息是否足够，不够则继续调用
5. 单次对话轮次工具调用上限为 8 次；达到上限时，用已有信息尽力回答并告知用户

---

## 7. 报告结构

```markdown
# 仓库完整画像：{repo_name}

## 1. 基础信息
- Stars / Forks / License / 主要语言
- 最近活跃度（最后 commit 时间、过去30天 commit 数）

## 2. 项目简介
- README 摘要（LLM 提炼，200字以内）

## 3. 核心架构
- 目录结构概览
- 关键模块说明

## 4. 代码质量评估
- 测试覆盖情况（有无 tests/ 目录、CI 配置）
- 文档完整性

## 5. 社区健康度
- Contributors 数量与分布
- Open Issues 数量与趋势
- 与同类项目对比（来自知识库）

## 6. 总结与推荐
- 适用场景
- 优缺点
- 综合评分（1-10）
```

---

## 8. API 接口

### 新增端点

```
POST /agent/chat
Body:     { "session_id": "xxx", "message": "用户输入" }
Response: SSE 流式输出（text/event-stream）
```

```
GET /agent/reports/{filename}
Response: 报告 Markdown 文件内容（text/plain）
```

### SSE 消息类型

```json
// Agent 状态提示
{"type": "agent_step", "content": "正在查询知识库..."}
{"type": "agent_step", "content": "正在从 GitHub 获取仓库信息：encode/httpx"}
{"type": "agent_step", "content": "正在读取文件：README.md"}
{"type": "agent_step", "content": "正在生成完整报告..."}

// 最终回答文字流
{"type": "token", "content": "httpx 是一个..."}

// 错误事件
{"type": "error", "content": "GitHub API 速率限制，请稍后重试"}
{"type": "error", "content": "工具调用次数已达上限，以下是基于现有信息的回答"}
```

前端行为：
- 收到 `agent_step`：在对话框上方显示动态提示条
- 收到 `token`：提示条消失，开始流式显示答案
- 收到 `error`：提示条显示为错误状态，展示错误信息

---

## 9. Session 管理

- `session_id` 由前端生成（UUID），每个 session 对应一个 Agent 实例存储在后端内存中
- Agent 实例生命周期：首次请求时创建，**30 分钟无活动后**自动销毁
- 同一 `session_id` 并发请求：第二个请求等待第一个完成后再处理（队列化，不并发）
- Session 存储在进程内存中，不持久化（服务重启后所有 session 清空）

---

## 10. GitHub 认证配置

- 环境变量：`GITHUB_TOKEN`（Personal Access Token，需 `public_repo` 读权限）
- 已设置 token：速率限制 5000次/小时
- 未设置 token：匿名请求，速率限制 60次/小时，启动时打印警告日志
- Token 读取位置：`backend/config.py` 或 `.env` 文件（通过 `python-dotenv` 加载）

---

## 11. 不在范围内

- 持久化对话历史到数据库
- 多用户并发的 session 池管理
- Agent 行为的可视化调试界面
- 自动定时分析（无自动化，全部由用户主动触发）
- GitHub Enterprise 支持
- OAuth 流程（仅支持 PAT）
