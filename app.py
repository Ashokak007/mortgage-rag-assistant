"""Streamlit chat UI.  Run: streamlit run app.py"""
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag.generator import GeminiGenerator, answer
from rag.retriever import SentenceTransformerEmbedder, VectorIndex

load_dotenv()
st.set_page_config(page_title="Mortgage Document Assistant", page_icon="🏠")
st.title("🏠 Mortgage Document Assistant")
st.caption("Answers come only from the PDFs in data/, with page citations.")


@st.cache_resource
def load_components():
    return VectorIndex.load(Path("index"), SentenceTransformerEmbedder()), GeminiGenerator()


try:
    index, generator = load_components()
except Exception as e:  # show setup problems in the UI instead of a stack trace
    st.error(str(e))
    st.stop()

k = st.sidebar.slider("Passages to retrieve (k)", 1, 8, 4)
if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    st.chat_message(role).markdown(content)

if question := st.chat_input("Ask about loan estimates, closing costs, PMI..."):
    st.chat_message("user").markdown(question)
    with st.spinner("Searching documents..."):
        result = answer(question, index, generator, k=k)
    with st.chat_message("assistant"):
        st.markdown(result["answer"])
        for i, s in enumerate(result["sources"], 1):
            with st.expander(f"[{i}] {s['source']} — page {s['page']} (similarity {s['score']})"):
                st.write(s["text"])
    st.session_state.history += [("user", question), ("assistant", result["answer"])]
