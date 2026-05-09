"""
PDF Viewer Component
====================
Renders an inline PDF viewer using base64 embedding.
Shows page thumbnails, search highlights, and metadata.
"""

import streamlit as st
import base64


def render_pdf_viewer():
    """Render the PDF viewer tab."""
    if not st.session_state.pdf_docs:
        st.info("📤 Upload PDFs in the sidebar to view them here.")
        return

    st.subheader("📄 Document Viewer")

    # File selector
    selected = st.selectbox("Select document:", st.session_state.pdf_docs)

    # Try to find and display the PDF
    col_info, col_actions = st.columns([3, 1])
    with col_info:
        st.markdown(f"**File:** `{selected}`")

    with col_actions:
        st.markdown("*Re-upload to preview inline*")

    # Document metadata from vector store
    if st.session_state.vector_store:
        st.markdown("### 📊 Document Metadata")
        stats = st.session_state.vector_store.get_stats()

        cols = st.columns(4)
        cols[0].metric("Total Chunks", stats.get("indexed", 0))
        cols[1].metric("Active Files", len(st.session_state.pdf_docs))
        cols[2].metric("Backend", stats.get("backend", "–").upper())
        cols[3].metric("Queries Run", st.session_state.analytics.get("queries", 0))

    st.divider()

    # Semantic search within viewer
    st.markdown("### 🔍 Search Within Document")
    search_query = st.text_input("Search for specific content:", placeholder="Enter keywords…")
    if search_query and st.session_state.vector_store:
        results = st.session_state.vector_store.similarity_search(
            search_query,
            k=5,
            filter={"source": selected} if selected else None,
        )
        if results:
            st.markdown(f"**{len(results)} relevant passages found:**")
            for i, doc in enumerate(results, 1):
                page = doc.metadata.get("page", "?")
                with st.expander(f"Passage {i} — Page {page}"):
                    # Highlight search terms (basic)
                    content = doc.page_content
                    st.markdown(content)
                    st.caption(f"File: {doc.metadata.get('source')} | Page: {page}")
        else:
            st.warning("No matching passages found.")


def _display_pdf_base64(file_bytes: bytes):
    """Display PDF inline using base64 iframe."""
    b64 = base64.b64encode(file_bytes).decode()
    iframe_html = f"""
    <iframe
        src="data:application/pdf;base64,{b64}"
        width="100%"
        height="700px"
        style="border:1px solid #e5e7eb; border-radius:8px;"
    ></iframe>
    """
    st.markdown(iframe_html, unsafe_allow_html=True)
