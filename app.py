"""Streamlit web interface."""

import streamlit as st
from pathlib import Path
import sys
import os

# Set these BEFORE importing any libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

sys.path.insert(0, "src")

from docvision import LegalGPT

st.set_page_config(page_title="LegalGPT", page_icon="⚖️", layout="wide")

# Session state
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("⚖️ LegalGPT")
st.markdown("*AI-powered legal document assistant*")

# Sidebar
with st.sidebar:
    st.header("📄 Documents")

    files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    use_hybrid = st.checkbox("Use Hybrid Search", value=False)

    if files and st.button("Process"):
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        for f in files:
            (upload_dir / f.name).write_bytes(f.read())

        with st.spinner("Processing..."):
            st.session_state.pipeline = LegalGPT(use_hybrid=use_hybrid)
            st.session_state.pipeline.ingest_documents(str(upload_dir))

        st.success(f"✓ Processed {len(files)} documents!")

    if st.session_state.pipeline and st.session_state.pipeline.is_ready:
        st.success("✓ Ready to answer questions")
    else:
        st.info("👆 Upload documents to start")

    st.header("⚙️ Settings")
    top_k = st.slider("Chunks to retrieve", 3, 10, 5)

# Main chat
if not (st.session_state.pipeline and st.session_state.pipeline.is_ready):
    st.info("👈 Upload and process documents to begin")
else:
    # Display history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander("📚 Sources"):
                    for src in msg["sources"]:
                        st.write(f"- {src['document']}, Page {src['page']}")

    # Input
    if question := st.chat_input("Ask about your documents..."):
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.pipeline.query(question, top_k)

                st.markdown(result["answer"])

                with st.expander("📚 Sources"):
                    for src in result["sources"]:
                        st.write(f"- {src['document']}, Page {src['page']}")

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    }
                )
