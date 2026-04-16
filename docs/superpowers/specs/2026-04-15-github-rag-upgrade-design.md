# GitHub RAG 升级设计文档

**日期：** 2026-04-15  
**项目：** GitHub Project Search Engine — 生产级 RAG 系统升级  
**目标：** 将课程项目升级为可公开访问的全栈 RAG 应用，用于求职简历展示

---

## 1. 项目背景

### 原项目技术栈
- **检索**：`sentence-transformers` (all-MiniLM-L6-v2) + 余弦相似度，向量存于内存
- **生成**：`Mistral-7B-Instruct-v0.2`，本地加载（13GB+）
- **前端**：Gradio
- **数据**：三个静态 CSV 文件，手动合并构建 corpus
- **部署**：Google Colab（无公开持久 URL）

### 升级动机
- Colab 无法提供持久公开 URL，简历无法附链接
- 本地加载大模型不适合云部署
- 检索质量有限，缺少工业级向量数据库
- 技术栈含金量不足，无法充分体现全栈工程能力

---

## 2. 升级目标

升级后项目需满足：
1. 有公开可访问的 URL，可直接写在简历上
2. 技术栈覆盖 RAG 全链路（检索、生成、部署）
3. 前后端分离，体现全栈工程能力
4. 所有服务使用免费 tier，零成本运行

---

## 3. 整体架构

### 技术选型

| 层级 | 原来 | 升级后 |
|------|------|--------|
| 向量存储 | 内存（每次重新 encode） | Pinecone（持久化向量数据库） |
| 检索方式 | 纯向量相似度 | BM25 + Pinecone 混合检索（RRF 融合） |
| 文档分块 | 整段文本拼接 | Markdown heading 语义切分（父子块设计） |
| LLM | Mistral-7B 本地加载 | GLM 智谱 API（兼容 OpenAI SDK，流式输出） |
| 前端 | Gradio | React（多轮对话 + 过滤器 + 流式渲染） |
| 后端 | 无 | FastAPI（Python） |
| 部署 | Google Colab | Vercel（前端）+ Railway（后端） |

### 数据流

```
用户输入查询
     ↓
[React 前端] — 维护 messages[] 对话历史
     ↓ POST /api/chat（含 messages[] + filters）
[FastAPI 后端]
     ↓
  ┌──────────────────────────────────────┐
  │             检索层                    │
  │  1. BM25 关键词检索子块               │
  │  2. Pinecone 向量语义检索子块         │
  │  3. RRF（Reciprocal Rank Fusion）融合 │
  │  4. Top-5 子块 → 附带父块元数据       │
  └──────────────────────────────────────┘
     ↓ 构建 Prompt（含对话历史 + 检索结果）
[GLM API] — stream=True 流式输出
     ↓ SSE stream
[React 前端] — 逐字渲染，光标闪烁效果
```

---

## 4. 数据层设计

### 原始数据文件
- `github_basic.csv`：仓库元数据（Full Name, Clone URL, Description, Topics, Language, Stars, Forks, Watchers, Issues, Create_Time, Update_Time, Push_Time）
- `github_readmes.csv`：第一批 README 内容（ID, Full Name, Readme Content）
- `github_readmes2.csv`：第二批 README 内容（Full Name, Readme Content1）

### 父子块设计

数据天然形成父子结构，与 Parent-Child Chunking 完全吻合：

**父块（每个仓库一个）：**
```
{
  "full_name": "facebook/react",
  "clone_url": "https://github.com/facebook/react.git",
  "description": "The library for web and native user interfaces.",
  "topics": ["declarative", "frontend", "javascript", ...],
  "language": "JavaScript",
  "stars": 232612,
  "forks": 47696,
  "watchers": 232612,
  "issues": 952,
  "create_time": "2013-05-24T16:15:54Z",
  "update_time": "2025-02-26T05:53:48Z",
  "push_time": "2025-02-26T00:09:24Z"
}
```

**子块（每个 README heading 段落一个）：**
```
{
  "parent_id": "facebook/react",       # 用于回查父块
  "section_title": "## Installation",  # Markdown heading 文本
  "content": "React has been designed for gradual adoption...",  # 正文
  "chunk_index": 2                      # 第几块，用于排序
}
```

### Markdown 语义切分规则

- 按 `#`、`##`、`###` 等 heading 行作为分割边界
- 每个 heading + 其下方正文 = 一个子块
- README 首段（heading 之前的内容）单独作为第 0 块
- 最小块长度阈值：内容少于 50 字符的块合并到相邻块

### README 清洗规则

README 内容包含大量噪音，切分前需清洗：
- 去除 Markdown badge 语法：`[![...](...)(...)`
- 去除 HTML 标签：`<img>`, `<br>`, `<div>` 等
- 去除纯图片链接：`![...](...)` 
- 保留纯文本、代码块、普通链接文本

### Pinecone 存储方案

- **向量**：`all-MiniLM-L6-v2` encode 子块 `content` 字段
- **metadata**：`{parent_id, section_title, chunk_index}` + 父块全部字段（冗余存储，避免检索后二次查询）
- **index 配置**：dimension=384（MiniLM 输出维度），metric=cosine

---

## 5. 检索层设计

### BM25 检索
- 库：`rank_bm25`（版本锁定在 `requirements.txt`，与 Python 版本一致）
- 索引对象：所有子块的 `content` 字段
- 索引序列化：`bm25_index.pkl`，连同 `chunk_metadata.json`（chunk 索引到父块元数据的映射）一起提交到 repo
- 估算文件大小：视 corpus 大小约 5–20MB，在 Git 和 Railway 部署范围内
- 查询时：对最新用户问题做 BM25 检索，返回 Top-20 候选
- **过滤方式（Post-filter）**：BM25 先返回 Top-20，然后根据 `chunk_metadata.json` 中的元数据对结果做后过滤，丢弃不满足 language/stars/topics 条件的子块
- **空结果降级**：若 post-filter 后 BM25 候选为空（如某语言关键词无命中），BM25 贡献项从 RRF 中省略，仅使用 Pinecone 结果排名；若两路均为空，返回提示"未找到符合过滤条件的项目，请放宽筛选条件"

### Pinecone 向量检索
- 查询时：encode 最新用户问题 → Pinecone query，返回 Top-20 候选
- 支持 metadata filter：按 `language`、`stars`（范围）、`topics` 过滤（OR 逻辑：topics 列表中任一匹配即可）
- BM25 使用同样的 post-filter 逻辑，确保两路结果过滤条件一致

### RRF 融合（Reciprocal Rank Fusion）
```python
# RRF 公式：score(d) = Σ 1 / (k + rank(d))，k=60
# 合并 BM25 排名和 Pinecone 排名，取融合后 Top-5
```

### 父块回查
- Top-5 子块通过 `parent_id` 从 Pinecone metadata 中直接读取父块信息
- 无需额外数据库查询（元数据已冗余存储在 Pinecone）

---

## 6. 后端 API 设计（FastAPI）

### 项目结构

```
backend/
├── main.py              # FastAPI 入口，注册路由，配置 CORS
├── indexer.py           # 一次性数据索引脚本（本地运行）
├── retriever.py         # 混合检索逻辑（BM25 + Pinecone + RRF）
├── chunker.py           # Markdown 语义切分 + 清洗
├── llm.py               # GLM API 调用 + 流式输出封装
├── models.py            # Pydantic 请求/响应模型定义
├── bm25_index.pkl       # 序列化 BM25 索引（indexer 生成后提交）
├── chunk_metadata.json  # BM25 rank index → 父块元数据映射（indexer 生成后提交）
├── filter_options.json  # 过滤器选项（languages/topics/stars_range，indexer 生成后提交）
├── requirements.txt
└── .env                 # PINECONE_API_KEY, GLM_API_KEY（必须加入 .gitignore，不可提交）, PINECONE_INDEX_NAME
```

### API 端点

**POST /api/chat**
```json
// Request
{
  "messages": [
    {"role": "user", "content": "推荐一个Python爬虫框架"},
    {"role": "assistant", "content": "这里有几个项目..."},
    {"role": "user", "content": "第一个有没有异步支持？"}
  ],
  "filters": {
    "language": "Python",
    "min_stars": 1000,
    "topics": ["crawler"]
  }
}
// Response: SSE stream，逐 chunk 返回文本
```

**GET /api/filters/options**
```json
// Response
{
  "languages": ["Python", "JavaScript", "Go", ...],
  "topics": ["web", "machine-learning", "crawler", ...],
  "stars_range": {"min": 0, "max": 232612}
}
```
数据来源：`indexer.py` 生成 `filter_options.json` 并提交到 repo，`main.py` 启动时从磁盘读取，该端点直接返回其内容。

### 流式输出实现
- FastAPI 使用 `StreamingResponse`，media_type 为 `text/event-stream`
- GLM API 调用开启 `stream=True`，逐 chunk 转发
- 前端使用 `fetch` + `ReadableStream` 接收 SSE

### SSE Wire Format（前后端协议）
```
# 每个内容 chunk：
data: {"text": "chunk content here"}\n\n

# 流结束信号：
data: [DONE]\n\n
```
前端收到 `[DONE]` 后设置 `isStreaming=false`，解除输入框禁用。网络错误或非 200 响应时同样设置 `isStreaming=false` 并显示内联错误消息。

### Prompt 构建
- 系统 prompt：角色定义（GitHub 项目推荐助手），要求自然语言输出，不暴露内部检索过程
- 检索结果：Top-5 子块内容 + 对应父块元数据，每块限制 800 字符
- 对话历史：传入最近 **6 条** messages（3轮对话），防止长会话超出 GLM context window
- 最新问题：messages[] 最后一条 user 消息


---

## 7. 前端设计（React）

### 项目结构

```
frontend/
├── src/
│   ├── components/
│   │   ├── ChatWindow.jsx      # 对话消息列表，自动滚动到底部
│   │   ├── MessageBubble.jsx   # 单条消息，支持 react-markdown 渲染
│   │   ├── InputBar.jsx        # 输入框 + 发送按钮
│   │   └── FilterPanel.jsx     # Language 下拉 / Stars 滑块 / Topics 多选
│   ├── hooks/
│   │   └── useChat.js          # 管理 messages[] + 流式接收 + filters 状态
│   ├── api/
│   │   └── client.js           # 封装 fetch 调用后端 /api/chat
│   ├── App.jsx
│   └── main.jsx
├── package.json
└── vercel.json                 # 后端 API URL 环境变量配置
```

### 状态管理

```javascript
// useChat.js 核心状态
const [messages, setMessages] = useState([])      // 完整对话历史
const [isStreaming, setIsStreaming] = useState(false)
const [filters, setFilters] = useState({
  language: "",
  min_stars: 0,
  topics: []
})
```

### 多轮对话实现
- `messages[]` 数组维护完整对话历史，存于前端内存（不持久化，刷新清空）
- 每次发送：将完整 `messages[]` + `filters` POST 到后端（后端截取最近 6 条传给 LLM）
- 收到流式响应：逐 chunk append 到最后一条 assistant message

### 流式渲染
- 使用 `fetch` + `response.body.getReader()` 读取 SSE
- 解析 SSE 帧：收到 `data: [DONE]` 时结束，否则解析 `{"text": "..."}` 追加到消息
- 流结束或发生网络错误时：设置 `isStreaming=false`，解除输入框禁用，显示内联错误提示

### UI 功能
- 对话气泡：用户消息右对齐，AI 回答左对齐
- Markdown 渲染：AI 回答使用 `react-markdown` 渲染
- 流式打字效果：AI 回答逐字追加，末尾光标闪烁
- FilterPanel：Language 下拉、Stars 最小值输入、Topics 多选标签
- InputBar：Enter 发送，Shift+Enter 换行，流式输出中禁用发送按钮
- 样式：Tailwind CSS 基础类，优先跑通功能，后续再打磨 UI

### 主要依赖
```
react-markdown    # 渲染 AI 输出中的 Markdown 格式
tailwindcss       # 基础样式
```

---

## 8. 部署方案

### 索引脚本（一次性，本地运行）
```
本地运行 python backend/indexer.py
  1. 读取三个 CSV，合并数据
  2. 合并两个 README 文件（注意列名差异：Readme Content vs Readme Content1，需统一归一化）
  3. 清洗 README（去除 badge、HTML、图片语法）
  4. 按 Markdown heading 切分子块
  5. all-MiniLM-L6-v2 encode 所有子块
  6. 上传向量 + metadata 到 Pinecone
  7. 构建 BM25 索引，序列化为 bm25_index.pkl
  8. 生成 chunk_metadata.json（BM25 rank index → 父块元数据映射）
  9. 提取 filter options（languages, topics, stars_range），保存为 filter_options.json
  10. 将 bm25_index.pkl / chunk_metadata.json / filter_options.json 提交到 repo
```

### 后端 → Railway
```
仓库：backend/ 目录
部署：Railway 连接 GitHub，自动部署
环境变量：
  PINECONE_API_KEY
  PINECONE_INDEX_NAME=github-rag
  GLM_API_KEY
  ALLOWED_ORIGINS=https://your-app.vercel.app,http://localhost:5173
```
**注意**：Railway 免费 Trial 仅提供一次性 $5 额度，会过期。长期保持 portfolio 在线需使用 Hobby 计划（$5/月）。备选方案：Render.com 免费 tier（$0，但闲置 15 分钟后会休眠，冷启动约 30 秒）。

### 前端 → Vercel
```
仓库：frontend/ 目录
部署：Vercel 连接 GitHub，自动部署
环境变量：
  VITE_API_URL=https://your-backend.railway.app
```

### 免费 Tier 说明
| 服务 | 免费限制 | 是否够用 |
|------|---------|---------|
| Pinecone | 1个index，100万向量 | 够用 |
| Railway | 每月 $5 额度 | 轻量 FastAPI 够用 |
| Vercel | 无限静态部署 | 够用 |
| GLM API | 有免费额度 | 够用（演示用） |

---

## 9. 简历亮点总结

| 技术 | 体现能力 |
|------|---------|
| Pinecone 向量数据库 | 工业级向量存储与检索 |
| BM25 + RRF 混合检索 | 对 RAG 检索原理的深度理解 |
| Markdown heading 语义切分 | 对数据特性的理解，超越通用方案 |
| 父子块设计 | Advanced RAG 架构模式 |
| GLM API + SSE 流式输出 | LLM API 集成与流式工程 |
| FastAPI 后端 | Python 后端工程能力 |
| React 前端 | 前端工程能力 |
| Vercel + Railway 云部署 | 全栈云部署经验 |

---

## 10. 未来升级方向（Future Work）

- **Reranker**：引入 `cross-encoder/ms-marco-MiniLM-L-6-v2` 对检索结果二次排序，进一步提升检索质量
- **UI 打磨**：升级视觉设计，添加动画效果
- **用户反馈**：添加对 AI 回答的点赞/踩功能，收集数据
