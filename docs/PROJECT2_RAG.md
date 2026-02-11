# 工程级 RAG 项目执行方案（不是 Demo）

## 1. 项目目标

把“问一句答一句”的玩具系统升级为可上线的 RAG：

- **稳定输入**：支持 PDF / Markdown 解析与增量更新
- **稳定检索**：支持多策略召回与重排序
- **稳定输出**：降低 hallucination，输出可追溯引用
- **稳定迭代**：用评估集驱动优化而不是凭感觉调参

---

## 2. 技术栈（与你给出的建议一致）

- 编排：LangChain / LlamaIndex（二选一，建议先 LangChain）
- 向量库：FAISS / Chroma（本地实验足够）
- Embedding：OpenAI text-embedding-3-large vs BGE-large-zh
- Rerank：bge-reranker-base / large
- 前端：Streamlit

---

## 3. 系统架构

1. **Ingestion Layer**
   - PDF 解析（页码、标题、表格）
   - Markdown 解析（标题层级、代码块隔离）
   - 文档标准化：清洗、去重、打 metadata

2. **Chunking Layer**
   - Fixed-size（基线）
   - Recursive（结构感知）
   - Semantic（语义边界）

3. **Index & Retrieval Layer**
   - Dense retrieval（向量召回）
   - Sparse/BM25（可选）
   - Hybrid fusion（可选）

4. **Reranking Layer**
   - Top-k 结果送入 Cross-Encoder rerank
   - 输出 top-n 证据片段

5. **Generation Layer**
   - 强约束 Prompt：仅依据 evidence 回答
   - 引用输出：文档名 + chunk id + 页码

6. **Evaluation Layer**
   - Retrieval: Recall@k, MRR, nDCG
   - Generation: groundedness / hallucination rate / answer correctness

---

## 4. 为什么会 hallucinate（面试高频）

1. **召回不全**：真正证据没进上下文。
2. **Chunk 粒度不合适**：太碎导致语义断裂，太大导致噪声。
3. **Embedding 不匹配**：中文场景下英文偏置模型效果差。
4. **重排序缺失**：Top-k 虽相关，但不是“回答该问题”的最佳证据。
5. **Prompt 约束弱**：模型在证据不足时会“补全”。

---

## 5. 对比实验设计（你项目的亮点）

### 5.1 Chunk 策略对比

- fixed(500/100 overlap)
- recursive(按标题与段落)
- semantic(按句向量断点)

指标：Recall@5、答案正确率、平均延迟。

### 5.2 Embedding 对比

- OpenAI vs BGE
- 数据集分中文/英文问题

指标：召回、成本、延迟、跨域鲁棒性。

### 5.3 Index 对比

- FAISS vs Chroma

指标：索引构建时长、查询延迟、内存占用、易运维性。

### 5.4 Rerank 收益

- 有无 reranker 的 A/B

指标：Top-1 命中率、最终回答 groundedness。

---

## 6. 可交付里程碑（4 周版）

### Week 1
- 文档 ingestion + 标准化 + metadata schema
- baseline: fixed chunk + FAISS + 单 embedding

### Week 2
- chunk 策略对比框架
- embedding 对比框架

### Week 3
- 接入 bge-reranker
- 构建评估集（100~300 条真实问题）

### Week 4
- Streamlit 可视化（问题、证据、引用、评分）
- 输出实验报告（参数、指标、结论、下一步）

---

## 7. 简历表述（优化版）

> Built a production-style RAG system with PDF/Markdown ingestion, chunking and embedding A/B evaluations, FAISS/Chroma indexing, and cross-encoder reranking; improved grounded answer rate and reduced hallucination by **XX%** on a custom evaluation set.

可替换的量化项：
- hallucination rate 从 22% 降到 11%
- Recall@5 提升 18%
- P95 latency 降低 27%

---

## 8. 下一步（如果继续做“工程级”）

- 接入权限过滤（部门/角色）
- 增量索引与文档版本管理
- 观测：trace + token 成本监控
- 线上反馈闭环（bad case 自动入评测集）
