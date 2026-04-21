# RAG 检索质量评估系统设计文档

**日期：** 2026-04-21  
**项目：** GitHub Project Search Engine — 检索质量评估模块  
**目标：** 构建可重复运行的评估工具，量化 Precision@5、Recall@5、MRR 四项指标及 BM25/Pinecone 双路贡献，支持简历展示和系统持续优化

---

## 1. 背景与动机

当前系统采用 BM25 + Pinecone 混合检索 + RRF 融合，但从未对检索效果进行定量评估，存在以下风险：

- 无法回答"检索效果到底怎么样"这一核心问题
- 无法判断 BM25 和 Pinecone 哪路贡献更大，或是否有一路几乎无效
- 调整参数（如 `top_k`、RRF 的 `k` 值）后无法量化效果变化
- 简历展示时缺乏可支撑的数据

评估系统需满足：
1. **可重复运行**：每次修改检索参数后可快速重跑对比
2. **自动化**：无需人工标注即可生成 ground truth（LLM 辅助生成）
3. **人工可验证**：输出低分样本供人工抽查，确保 LLM judge 可信度
4. **覆盖多样性**：测试集分层抽样，覆盖不同语言、星数、topics

---

## 2. 整体架构

```
eval/
├── generate_testset.py   # 一次性运行，生成并保存测试集
├── evaluate.py           # 可重复运行，输出评估报告
├── testset.json          # 生成后提交到 repo，作为固定 ground truth
└── report.json           # 每次评估输出（不提交到 repo）
```

**运行流程：**

```
[第一步，只跑一次]
python eval/generate_testset.py
→ 读取 chunk_metadata.json 进行分层抽样
→ 调用 GLM API 为每个 repo 生成 query
→ 输出 testset.json（提交到 repo）

[此后可重复运行]
python eval/evaluate.py
→ 读取 testset.json
→ 对每条 query 调用 hybrid_search()（含中间结果）
→ 计算 Precision@5 / Recall@5 / MRR
→ 调用 GLM judge 扩充 soft precision
→ 输出 report.json + 终端摘要
```

---

## 3. 测试集生成：`generate_testset.py`

### 3.1 分层抽样策略

从 `chunk_metadata.json` 中读取所有父块元数据，按以下维度分层，目标生成 **400 条** query（精确配额，不依赖随机去重）。400 条在 95% 置信水平下误差约 ±4.8%，足以区分调参前后的效果差异。

**Language 层（共 180 条）：**
按语言统计仓库数量，取 Top 5 语言各为一层，其余归"其他"层，共 6 层，每层 **30 条**。

**Stars 层（共 60 条）：**
实际 corpus 中 stars 分布极度不均（low <1k 约占 70%，mid 1k~10k 约占 29%，high >10k 约占 1%），因此采用**强制均等配额**（非按比例抽样），每档各 **20 条**：

| 档位 | 定义 | 配额 | 备注 |
|------|------|------|------|
| low | stars < 1000 | 20 条 | 代表长尾项目 |
| mid | 1000 ≤ stars < 10000 | 20 条 | 代表主流项目 |
| high | stars ≥ 10000 | 20 条 | 最具辨识度，面试最常被问到 |

**随机层（共 160 条）：**
从全量 corpus 中随机抽样，不按任何维度过滤，确保测试集同时具备统计代表性（不只是多样性覆盖）。

**去重规则：** 三层独立抽样，若同一 repo 出现在多层中，优先级为 Language 层 > Stars 层 > 随机层，低优先级层重新抽取补足配额。最终测试集恰好 **400 条**，无重复 repo。

### 3.2 Query 生成 Prompt

对每个抽样的 repo，取以下内容作为上下文：
- `description`（父块字段）
- 第 0 号子块的 `content`（README 首段，最多 500 字符）

**chunk_metadata.json 查找说明：** `chunk_metadata.json` 是一个 JSON 数组，位置索引 `i` 对应 BM25 语料库第 `i` 条。读取后需先构建 `{parent_id: {chunk_index: entry}}` 二级 dict，再按 `parent_id` + `chunk_index=0` 查找首块，避免对 ~267k 条数据逐行线性扫描。

调用 GLM 生成 **1 条** query，要求交替生成以下两种类型（按 repo 顺序轮换）：
- **语义型**："我想找一个做 X 的 Y 语言库"（需求导向，不出现 repo 名）
- **关键词型**："有没有支持 Z feature 的项目"（技术特性导向，不出现 repo 名）

生成 Prompt 模板：
```
你是一个 GitHub 用户。根据以下开源仓库的信息，生成一条自然语言搜索查询。
要求：
1. 查询类型：{query_type}（语义型 / 关键词型）
2. 不能出现仓库名称或组织名称
3. 查询应是用户会真实输入的问题，50字以内
4. 只输出查询文本，不要解释

仓库信息：
Description: {description}
README 摘要: {readme_snippet}
```

### 3.3 输出格式：`testset.json`

```json
[
  {
    "query": "我想找一个轻量的 Python HTTP 客户端库",
    "relevant_repo_ids": ["psf/requests"],
    "query_type": "semantic",
    "meta": {
      "language": "Python",
      "stars_tier": "high",
      "source_repo": "psf/requests"
    }
  },
  {
    "query": "有没有支持异步的 Python web 框架",
    "relevant_repo_ids": ["tiangolo/fastapi"],
    "query_type": "keyword",
    "meta": {
      "language": "Python",
      "stars_tier": "high",
      "source_repo": "tiangolo/fastapi"
    }
  }
]
```

**注意：** `relevant_repo_ids` 初始只包含生成该 query 的 source repo（1 条）。人工抽查时可手动追加实际相关的其他 repo，但非必须。

### 3.4 成本估算

- 约 400 次 GLM API 调用（每次 ~200 input tokens + ~50 output tokens）
- 预计总 token 消耗 ~100k tokens，GLM 免费额度内可完成

---

## 4. 评估引擎：`evaluate.py`

### 4.1 retriever.py 改动

当前 `hybrid_search()` 只返回最终融合结果，评估需要暴露中间结果用于贡献分析。在 `hybrid_search()` 返回值中增加可选的 `debug` 模式：

```python
def hybrid_search(
    query: str,
    pinecone_index,
    language: str = "",
    min_stars: int = 0,
    topics: list[str] = None,
    top_k: int = 5,
    return_debug: bool = False   # 新增参数
) -> list[dict] | tuple[list[dict], dict]:
    ...
    # 在 fused = _rrf(ranked_lists)[:top_k] 这行之前，先保存完整 RRF 输出
    fused_all = _rrf(ranked_lists)          # 完整排名，未截断
    fused = fused_all[:top_k]               # 截断为 top_k，替换原有赋值

    if return_debug:
        debug_info = {
            # 使用全量候选集（最多 20 条），而非仅 top-5，确保贡献分析正确
            "pinecone_candidate_ids": [_chunk_metadata[i]["parent_id"] for i in pinecone_ranked],
            "bm25_candidate_ids": [_chunk_metadata[i]["parent_id"] for i in bm25_ranked],
            # fused_all 是 RRF 完整排名，results 是去重后的最终输出
            "fused_ranked_ids": [_chunk_metadata[i]["parent_id"] for i in fused_all[:top_k * 3]],
        }
        return results, debug_info
    return results
```

**关键说明：**
- `pinecone_candidate_ids` / `bm25_candidate_ids` 使用各路全量候选（最多 20 条），而非只取 top-5，因为 RRF 融合后排名靠前的结果可能来自各自候选中第 8~15 位，仅看 top-5 会漏掉这部分贡献
- 去重逻辑（`seen_parents`）在 `results` 层面，`fused` 不受影响，故 `fused` 长度恒等于 `top_k`
- 在 `evaluate.py` 中初始化 Pinecone 连接：参考 `backend/main.py` 中的 `init_pinecone()` 调用方式复用相同初始化逻辑

### 4.2 评估指标计算

对 testset.json 中的每条 query，执行：

```python
results, debug = hybrid_search(query, pinecone_index, return_debug=True)
retrieved_ids = [r["parent_id"] for r in results]   # Top-5 结果的 parent_id
relevant_ids = set(item["relevant_repo_ids"])         # ground truth
```

**指标定义：**

| 指标 | 公式 | 说明 |
|------|------|------|
| Precision@5 | `|retrieved ∩ relevant| / len(retrieved_ids)` | Top 结果中相关结果占比；分母用 `len(retrieved_ids)` 而非固定 5，因为去重后实际返回数可能 < 5 |
| Recall@5 | `|retrieved ∩ relevant| / |relevant|` | relevant 中被检索结果覆盖的比例；当前 ground truth 每条只有 1 个 relevant，等价于命中率（0 或 1） |
| MRR | `1 / rank_of_first_relevant` | 第一个相关结果排在第几位（无命中则 0） |

**关于 Recall@5 的说明：** 当前每条测试样本只有 1 个 relevant_repo_id，Recall@5 实际等同于命中率（hit rate），与 MRR 信息重叠度较高。Recall@5 主要作为人工补充 relevant_repo_ids 后的扩展指标，当前阶段重点关注 Precision@5 和 MRR。

**整体指标** = 所有 query 的各指标均值。

### 4.3 BM25 / Pinecone 贡献分析

对每条 query，利用 `debug_info` 中的中间结果判断最终结果来源。**关键：使用各路全量候选集（最多 20 条）做集合判断**，而非只看各路 top-5，因为 RRF 融合后的结果可能来自各路候选中第 6~20 位：

```
pinecone_set = set(debug_info["pinecone_candidate_ids"])
bm25_set     = set(debug_info["bm25_candidate_ids"])

对于最终 results 中的每个 repo_id：
- 若 repo_id 在 pinecone_set 且在 bm25_set → "both"
- 若只在 pinecone_set                       → "pinecone_only"
- 若只在 bm25_set                           → "bm25_only"
```

汇总所有 query，统计三类来源在 Top-5 位置中的占比：

```
BM25 独占贡献:     XX%
Pinecone 独占贡献: XX%
双路共同命中:      XX%
```

### 4.4 LLM-as-judge（Soft Precision）

对于 Top-5 中 **未被 ground truth 覆盖** 的结果，调用 GLM 判断是否实际相关：

Judge Prompt 模板：
```
用户查询：{query}
仓库名称：{repo_full_name}
仓库描述：{description}
README 摘要：{readme_snippet，最多 300 字符}

该仓库是否与用户查询相关？
请只回答 0（不相关）或 1（相关），不要解释。
```

**Soft Precision@5** = `(ground truth 命中数 + LLM judge 为 1 的数量) / len(retrieved_ids)`

**成本控制：** 只对 ground truth 未命中的结果调用 judge（平均每条 query 约 4 次调用），总计 ~1600 次调用。每次调用约 150 input tokens + 5 output tokens，总计约 248k tokens。注意：中文字符计 token 方式因模型而异，实际消耗可能略超此估算，运行前需确认 GLM 免费额度是否充足。

**错误处理：** judge 调用失败（网络超时、API 错误）时，该位置记录 `llm_judge: null`，排除在 soft precision 计算之外（不计入分子也不计入分母）。

### 4.5 低分样本输出（人工抽查）

将 LLM judge 打分最低（soft precision 最低）的 **50 条** query 及其 Top-5 结果写入 `report.json` 的 `low_score_samples` 字段，供人工逐条核查（400 条测试集取 50 条约为 12.5%，抽查比例与 150 条时保持一致）。

### 4.6 终端摘要输出

```
=== RAG Evaluation Report ===
Date: 2026-04-21
Total queries: 400

--- Core Metrics ---
Precision@5 (hard):  0.XX
Recall@5:            0.XX
MRR:                 0.XX

--- Soft Metrics (LLM judge) ---
Soft Precision@5:    0.XX

--- Retrieval Source Contribution ---
BM25 only:           XX%
Pinecone only:       XX%
Both:                XX%

--- Query Type Breakdown ---
Semantic queries:    Precision=0.XX, MRR=0.XX
Keyword queries:     Precision=0.XX, MRR=0.XX

Full report saved to: eval/report.json
Low-score samples (30): see report.json → low_score_samples
```

### 4.7 `report.json` 完整结构

```json
{
  "date": "2026-04-21",
  "total_queries": 400,
  "metrics": {
    "precision_at_5": 0.62,
    "recall_at_5": 0.71,
    "mrr": 0.58,
    "soft_precision_at_5": 0.74
  },
  "contribution": {
    "bm25_only_pct": 0.34,
    "pinecone_only_pct": 0.41,
    "both_pct": 0.25
  },
  "by_query_type": {
    "semantic": {"precision_at_5": 0.60, "mrr": 0.55},
    "keyword": {"precision_at_5": 0.64, "mrr": 0.61}
  },
  "by_language": {
    "Python": {"precision_at_5": 0.65, "count": 20},
    "JavaScript": {"precision_at_5": 0.58, "count": 20}
  },
  "by_stars_tier": {
    "high": {"precision_at_5": 0.70, "count": 20},
    "mid": {"precision_at_5": 0.61, "count": 20},
    "low": {"precision_at_5": 0.55, "count": 20}
  },
  "low_score_samples": [
    {
      "query": "...",
      "relevant_repo_ids": ["..."],
      "retrieved": [
        {"parent_id": "...", "description": "...", "llm_judge": 0}
      ],
      "soft_precision": 0.2
    }
  ]
}
```

---

## 5. 依赖与环境

评估脚本复用 `backend/` 目录中已有的所有 artifacts：

| 依赖项 | 说明 |
|--------|------|
| `backend/chunk_metadata.json` | 分层抽样数据源 |
| `backend/bm25_index.pkl` | BM25 索引 |
| `backend/retriever.py` | 混合检索逻辑（需小改加 `return_debug` 参数） |
| `PINECONE_API_KEY` / `PINECONE_INDEX_NAME` | 已有环境变量 |
| `GLM_API_KEY` / `GLM_API_URL` / `GLM_MODEL_ID` | 已有环境变量，同时用于生成 query 和 judge |

无需新增 Python 依赖，所有依赖已在 `backend/requirements.txt` 中。

---

## 6. 注意事项

- `testset.json` 提交到 repo，作为固定 baseline；每次重跑 `evaluate.py` 对比的是同一测试集
- `report.json` 加入 `.gitignore`，不提交；需要对比历史结果时手动保存重命名
- `generate_testset.py` 只跑一次；如果 corpus 更新（重新 indexer），需重新生成测试集
- `return_debug=True` 参数对正常 API 请求无影响，不改变原有返回值结构
- Stars 档位划分：`<1000` 为 low，`1000~10000` 为 mid，`>10000` 为 high

---

## 7. 面试说明要点

运行评估后，可以这样回答面试官的问题：

- **"检索效果怎么样？"** → "我构建了 400 条分层测试集（覆盖 6 个语言层、3 个星数档、160 条随机样本），Precision@5 约 X%，MRR 约 X%"
- **"BM25 和向量检索哪个更有用？"** → "BM25 贡献约 X%，Pinecone 贡献约 X%，约 X% 结果由两路共同命中，说明混合检索有实际价值"
- **"怎么保证评估结果可信？"** → "用 LLM 自动生成测试集后，人工抽查了 30 条低分样本验证 judge 可信度"
