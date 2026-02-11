import os

import streamlit as st

from src.rag.llm import LLMConfig
from src.rag.parsers import parse_document
from src.rag.pipeline import PipelineConfig, RAGPipeline

st.set_page_config(page_title="Engineering RAG", layout="wide")
st.title("工程级 RAG 实验台（test）")

with st.sidebar:
    st.header("数据导入")
    llm_provider = st.selectbox("回答模式", ["extractive", "openai", "ollama"], index=0)
    default_model = "gpt-4o-mini" if llm_provider == "openai" else "qwen2.5:7b"
    llm_model = st.text_input("LLM 模型", default_model)
    ollama_base_url = st.text_input("Ollama 地址", "http://localhost:11434")

    if llm_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        st.warning("未检测到 OPENAI_API_KEY，将自动降级为 extractive 模式。")
    if llm_provider == "ollama":
        st.info("无需云 API，需本地先启动 Ollama 服务。")

    if (
        "pipeline" not in st.session_state
        or st.session_state.get("provider") != llm_provider
        or st.session_state.get("model") != llm_model
        or st.session_state.get("ollama_base_url") != ollama_base_url
    ):
        cfg = PipelineConfig(
            llm=LLMConfig(
                provider=llm_provider,
                model=llm_model,
                temperature=0.0,
                ollama_base_url=ollama_base_url,
            )
        )
        st.session_state.pipeline = RAGPipeline(cfg)
        st.session_state.provider = llm_provider
        st.session_state.model = llm_model
        st.session_state.ollama_base_url = ollama_base_url

    st.subheader("方式 A：手动文本")
    doc_id = st.text_input("文档 ID", "doc-1")
    content = st.text_area("文档内容（示例）", "RAG 的核心是检索质量与证据约束。")
    if st.button("写入索引（手动文本）"):
        st.session_state.pipeline.ingest(doc_id, content)
        st.success(f"已写入索引：{doc_id}")

    st.subheader("方式 B：上传真实文档")
    files = st.file_uploader(
        "上传 PDF / DOCX / PPTX",
        type=["pdf", "docx", "pptx"],
        accept_multiple_files=True,
    )
    if st.button("写入索引（上传文件）"):
        if not files:
            st.warning("请先上传至少一个文件。")
        else:
            for f in files:
                try:
                    parsed = parse_document(f.name, f.getvalue())
                    if not parsed.content.strip():
                        st.warning(f"文件无可提取文本：{f.name}")
                        continue
                    st.session_state.pipeline.ingest(parsed.doc_id, parsed.content)
                    st.success(f"已写入：{f.name} -> doc_id={parsed.doc_id}")
                except Exception as exc:
                    st.error(f"解析失败 {f.name}: {exc}")

st.header("问答")
query = st.text_input("问题", "为什么 RAG 会 hallucinate?")
if st.button("检索并回答"):
    result = st.session_state.pipeline.answer(query)
    st.subheader("答案")
    st.write(result["answer"])

    st.subheader("证据片段")
    for i, ev in enumerate(result["evidence"], start=1):
        st.markdown(f"**[{i}]** {ev}")
