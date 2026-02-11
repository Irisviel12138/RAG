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

