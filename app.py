import os

import streamlit as st

from src.rag.llm import LLMConfig
from src.rag.pipeline import PipelineConfig, RAGPipeline

st.set_page_config(page_title="Engineering RAG", layout="wide")
st.title("工程级 RAG 实验台（test）")

with st.sidebar:
    st.header("数据导入")
    llm_provider = st.selectbox("回答模式", ["extractive", "openai"], index=0)
    llm_model = st.text_input("LLM 模型", "gpt-4o-mini")

    if llm_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        st.warning("未检测到 OPENAI_API_KEY，将自动降级为 extractive 模式。")

    if "pipeline" not in st.session_state or st.session_state.get("provider") != llm_provider:
        cfg = PipelineConfig(llm=LLMConfig(provider=llm_provider, model=llm_model, temperature=0.0))
        st.session_state.pipeline = RAGPipeline(cfg)
        st.session_state.provider = llm_provider

import streamlit as st

from src.rag.pipeline import PipelineConfig, RAGPipeline

st.set_page_config(page_title="Engineering RAG", layout="wide")
st.title("工程级 RAG 实验台（Project 2）")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline(PipelineConfig())

with st.sidebar:
    st.header("数据导入")
    doc_id = st.text_input("文档 ID", "doc-1")
    content = st.text_area("文档内容（示例）", "RAG 的核心是检索质量与证据约束。")
    if st.button("写入索引"):
        st.session_state.pipeline.ingest(doc_id, content)
        st.success("已写入索引")

st.header("问答")
query = st.text_input("问题", "为什么 RAG 会 hallucinate？")
if st.button("检索并回答"):
    result = st.session_state.pipeline.answer(query)
    st.subheader("答案")
    st.write(result["answer"])

    st.subheader("证据片段")
    for i, ev in enumerate(result["evidence"], start=1):
        st.markdown(f"**[{i}]** {ev}")
