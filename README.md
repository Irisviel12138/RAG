# Project 2: Engineering-Grade RAG System

这是一个“工程级”RAG 项目脚手架，目标不是单轮 Demo，而是可评估、可扩展、可迭代。

## 核心能力

- 文档解析：PDF / DOCX / PPTX / Markdown
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



## 接入真实 LLM

默认是 `extractive` 回答模式。若要启用 OpenAI：

```bash
export OPENAI_API_KEY=your_key
streamlit run app.py
```

然后在侧边栏将“回答模式”切换为 `openai`。


## 真实文档导入

在 Streamlit 侧边栏可直接上传 `PDF / DOCX / PPTX`，系统会自动抽取文本并写入索引。



## 文档排版噪声说明

PDF/PPT 由布局恢复文本时，可能出现断行、符号缺失等问题。当前解析器已做基础清洗（Unicode 归一化、异常字符清理、空白规整），对数学公式类文档建议：

- 优先使用源 DOCX/Markdown；
- 或采用 OCR/版面分析增强方案（如 Mathpix、Nougat、MinerU）。
