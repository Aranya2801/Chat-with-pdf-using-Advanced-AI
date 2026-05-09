"""
╔══════════════════════════════════════════════════════════════╗
║     Chat-with-PDF using Advanced AI                         ║
║     Multi-Agent RAG System with Hybrid Search               ║
║     Author: Aranya2801 | MIT-Grade Architecture             ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import os
import time
import json
import hashlib
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from src.agents.orchestrator import AgentOrchestrator
from src.components.sidebar import render_sidebar
from src.components.chat_interface import render_chat
from src.components.pdf_viewer import render_pdf_viewer
from src.components.analytics_dashboard import render_analytics
from src.utils.session_manager import SessionManager
from src.utils.logger import setup_logger

load_dotenv()
logger = setup_logger(__name__)

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChatPDF · Advanced AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/Aranya2801/Chat-with-pdf-using-Advanced-AI",
        "Report a bug": "https://github.com/Aranya2801/Chat-with-pdf-using-Advanced-AI/issues",
        "About": "### ChatPDF Advanced AI\nMulti-agent RAG with hybrid semantic search",
    },
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
def load_css():
    css_path = Path("assets/style.css")
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ── Session Initialization ─────────────────────────────────────────────────────
def init_session():
    session = SessionManager()
    defaults = {
        "messages": [],
        "pdf_docs": [],
        "vector_store": None,
        "orchestrator": None,
        "session_id": session.generate_id(),
        "active_tab": "chat",
        "analytics": {
            "queries": 0,
            "docs_processed": 0,
            "avg_response_time": 0,
            "satisfaction_scores": [],
        },
        "settings": {
            "model": "gpt-4o",
            "embedding_model": "text-embedding-3-large",
            "retrieval_mode": "hybrid",
            "top_k": 8,
            "rerank": True,
            "citation_mode": True,
            "language": "auto",
            "temperature": 0.1,
            "memory_window": 10,
            "use_agents": True,
        },
        "uploaded_hashes": set(),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()

# ── Main Layout ────────────────────────────────────────────────────────────────
def main():
    # Header
    col_logo, col_title, col_status = st.columns([1, 6, 2])
    with col_logo:
        st.markdown("## 🧠")
    with col_title:
        st.markdown(
            """
            <h1 style='margin:0; font-size:1.8rem; font-weight:800; 
                       background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Chat with PDF · Advanced AI
            </h1>
            <p style='margin:0; color:#6b7280; font-size:0.85rem;'>
            Multi-Agent RAG · Hybrid Search · Real-time Citations · Auto-Summary
            </p>
            """,
            unsafe_allow_html=True,
        )
    with col_status:
        if st.session_state.vector_store:
            st.success(f"✅ {len(st.session_state.pdf_docs)} PDF(s) Active")
        else:
            st.info("📤 Upload PDFs to begin")

    st.divider()

    # Sidebar
    with st.sidebar:
        render_sidebar()

    # Main Tabs
    tab_chat, tab_viewer, tab_analytics, tab_notes = st.tabs(
        ["💬 AI Chat", "📄 PDF Viewer", "📊 Analytics", "📝 Smart Notes"]
    )

    with tab_chat:
        render_chat()

    with tab_viewer:
        render_pdf_viewer()

    with tab_analytics:
        render_analytics()

    with tab_notes:
        render_smart_notes()


def render_smart_notes():
    """Auto-generated smart notes from PDFs."""
    st.subheader("📝 Smart Notes — Auto-Generated")
    if not st.session_state.vector_store:
        st.info("Upload and process PDFs to generate smart notes.")
        return

    col1, col2 = st.columns([3, 1])
    with col2:
        note_type = st.selectbox(
            "Note Style",
            ["Executive Summary", "Key Concepts", "Q&A Format", "Bullet Points", "Mind Map"],
        )
        if st.button("🔄 Regenerate Notes", use_container_width=True):
            st.session_state["notes_cache"] = None

    with col1:
        if "notes_cache" not in st.session_state or not st.session_state.notes_cache:
            with st.spinner("Generating intelligent notes…"):
                if st.session_state.orchestrator:
                    notes = st.session_state.orchestrator.generate_notes(note_type)
                    st.session_state.notes_cache = notes

        if st.session_state.get("notes_cache"):
            st.markdown(st.session_state.notes_cache)
            st.download_button(
                "⬇️ Download Notes",
                st.session_state.notes_cache,
                file_name=f"smart_notes_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
            )


if __name__ == "__main__":
    main()
