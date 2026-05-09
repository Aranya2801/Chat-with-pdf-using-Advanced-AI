"""
Sidebar Component
=================
Handles:
  - PDF upload & processing
  - Model settings
  - Session controls
  - Quick stats
"""

import streamlit as st
import hashlib
import time
from pathlib import Path

from src.utils.pdf_processor import PDFProcessor
from src.vectorstore.vector_store_manager import VectorStoreManager
from src.agents.orchestrator import AgentOrchestrator


def render_sidebar():
    st.markdown("### ⚙️ Configuration")

    # ── Model Settings ─────────────────────────────────────────────────────────
    with st.expander("🤖 Model Settings", expanded=False):
        st.session_state.settings["model"] = st.selectbox(
            "LLM Model",
            ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
            index=0,
            help="GPT-4o recommended for best accuracy",
        )
        st.session_state.settings["temperature"] = st.slider(
            "Temperature", 0.0, 1.0,
            value=st.session_state.settings.get("temperature", 0.1),
            step=0.05,
            help="Lower = more factual, Higher = more creative",
        )
        st.session_state.settings["top_k"] = st.slider(
            "Retrieval Top-K", 3, 20,
            value=st.session_state.settings.get("top_k", 8),
            help="Number of document chunks to retrieve per query",
        )
        st.session_state.settings["memory_window"] = st.slider(
            "Memory Window (turns)", 2, 20,
            value=st.session_state.settings.get("memory_window", 10),
        )

    # ── Retrieval Settings ─────────────────────────────────────────────────────
    with st.expander("🔍 Retrieval Settings", expanded=False):
        st.session_state.settings["retrieval_mode"] = st.selectbox(
            "Retrieval Mode",
            ["hybrid", "dense", "bm25"],
            help="Hybrid = BM25 + Dense (recommended)",
        )
        st.session_state.settings["rerank"] = st.toggle(
            "Cross-Encoder Reranking",
            value=st.session_state.settings.get("rerank", True),
            help="Improves accuracy at slight latency cost",
        )
        st.session_state.settings["citation_mode"] = st.toggle(
            "Auto Citations",
            value=st.session_state.settings.get("citation_mode", True),
        )
        st.session_state.settings["use_agents"] = st.toggle(
            "Multi-Agent Mode",
            value=st.session_state.settings.get("use_agents", True),
            help="Route queries to specialized AI agents",
        )

    st.divider()

    # ── PDF Upload ─────────────────────────────────────────────────────────────
    st.markdown("### 📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        new_files = []
        for f in uploaded_files:
            file_hash = hashlib.md5(f.read()).hexdigest()
            f.seek(0)
            if file_hash not in st.session_state.uploaded_hashes:
                new_files.append((f, file_hash))

        if new_files:
            if st.button("🚀 Process Documents", use_container_width=True, type="primary"):
                _process_pdfs(new_files)
        else:
            st.success("✅ All files already processed")

    # ── Active Documents ───────────────────────────────────────────────────────
    if st.session_state.pdf_docs:
        st.markdown("### 📚 Active Documents")
        for doc_name in st.session_state.pdf_docs:
            st.markdown(f"&nbsp;&nbsp;📄 `{doc_name}`")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear All", use_container_width=True):
                _clear_session()
        with col2:
            if st.button("💾 Save Index", use_container_width=True):
                if st.session_state.vector_store:
                    st.session_state.vector_store.save()
                    st.success("Saved!")

    # ── Quick Stats ────────────────────────────────────────────────────────────
    if st.session_state.vector_store:
        st.divider()
        st.markdown("### 📊 Index Stats")
        stats = st.session_state.vector_store.get_stats()
        col1, col2 = st.columns(2)
        col1.metric("Chunks", stats.get("indexed", 0))
        col2.metric("Backend", stats.get("backend", "–").upper())

    # ── Session Info ───────────────────────────────────────────────────────────
    st.divider()
    st.caption(f"🔑 Session: `{st.session_state.session_id}`")
    st.caption(f"💬 Messages: {len(st.session_state.messages)}")
    queries = st.session_state.analytics.get("queries", 0)
    st.caption(f"🔍 Queries this session: {queries}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _process_pdfs(new_files: list):
    """Process uploaded PDF files."""
    processor = PDFProcessor(st.session_state.settings)

    all_docs = []
    progress = st.progress(0, text="Initializing…")

    for i, (file, file_hash) in enumerate(new_files):
        progress.progress(
            (i + 1) / len(new_files),
            text=f"Processing {file.name}…",
        )
        try:
            file_bytes = file.read()
            chunks = processor.process(file_bytes, file.name)
            all_docs.extend(chunks)
            st.session_state.uploaded_hashes.add(file_hash)
            if file.name not in st.session_state.pdf_docs:
                st.session_state.pdf_docs.append(file.name)
            st.session_state.analytics["docs_processed"] += 1
        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")

    if all_docs:
        progress.progress(0.9, text="Building vector index…")
        vs_manager = VectorStoreManager(st.session_state.settings)

        if st.session_state.vector_store:
            st.session_state.vector_store.add_documents(all_docs)
        else:
            vs_manager.create_from_documents(all_docs)
            st.session_state.vector_store = vs_manager

        # Initialize orchestrator
        st.session_state.orchestrator = AgentOrchestrator(
            vector_store=st.session_state.vector_store,
            settings=st.session_state.settings,
        )

        progress.progress(1.0, text="Done!")
        time.sleep(0.5)
        progress.empty()
        st.success(f"✅ Indexed {len(all_docs)} chunks from {len(new_files)} file(s)!")
        st.rerun()


def _clear_session():
    keys_to_clear = ["messages", "pdf_docs", "vector_store", "orchestrator",
                     "uploaded_hashes", "notes_cache"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()
