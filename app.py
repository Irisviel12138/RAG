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
