# Project 2: Engineering-Grade RAG System

这是一个“工程级”RAG 项目脚手架，目标不是单轮 Demo，而是可评估、可扩展、可迭代。

## 核心能力

- 文档解析：PDF / Markdown
- Chunk 策略对比：fixed / recursive / semantic
- Embedding 模型对比：OpenAI vs BGE
- 向量库对比：FAISS / Chroma
- 检索增强：Hybrid Retrieval + Cross-Encoder Rerank
- 可视化验证：Streamlit 前端
- 评估体系：retrieval + generation 双维度指标

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 目录结构

```text
.
├── app.py
├── config/settings.example.yaml
├── docs/PROJECT2_RAG.md
├── requirements.txt
└── src/rag/
    ├── chunking.py
    ├── embeddings.py
    ├── index.py
    ├── rerank.py
    └── pipeline.py
```

## 招聘导向（你在面试里可以说）

- 你不仅搭建了 RAG，还对 chunk、embedding、索引和 rerank 做了系统性 A/B 对比。
- 你建立了 hallucination 归因路径（召回不足、上下文污染、生成约束不足）。
- 你能讲清楚为什么“检索质量”比“换更大模型”更优先。
