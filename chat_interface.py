"""
Chat Interface Component
========================
Renders the main conversational UI with:
  - Streaming token display
  - Source citation cards
  - Agent metadata badges
  - Feedback buttons (thumbs up/down)
  - Export chat history
  - Suggested follow-up questions
  - Quiz mode toggle
"""

import streamlit as st
import time
from datetime import datetime


# ── Suggested Starter Prompts ──────────────────────────────────────────────────
STARTER_PROMPTS = [
    "📋 Summarize the main findings of this document",
    "🔍 What are the key concepts explained here?",
    "📊 Are there any statistics or numerical data?",
    "❓ Generate 5 quiz questions from this document",
    "🗂️ What is the document structure and main sections?",
    "🧠 Explain the most complex idea in simple terms",
]


def render_chat():
    """Render the full chat interface."""
    if not st.session_state.vector_store:
        _render_welcome()
        return

    _render_chat_history()
    _render_input()


# ── Welcome Screen ─────────────────────────────────────────────────────────────

def _render_welcome():
    st.markdown(
        """
        <div style='text-align:center; padding: 3rem 1rem;'>
          <div style='font-size: 4rem; margin-bottom: 1rem;'>🧠</div>
          <h2 style='font-weight: 800; color: #1f2937;'>Chat with Your Documents</h2>
          <p style='color: #6b7280; max-width: 500px; margin: 0 auto;'>
            Upload one or more PDFs using the sidebar, then ask anything.
            Our multi-agent AI will retrieve, reason, and cite answers in real time.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### ✨ What you can do:")
    cols = st.columns(3)
    features = [
        ("💬", "Natural Language Q&A", "Ask questions in plain English"),
        ("🔍", "Hybrid Search", "BM25 + Semantic + Reranking"),
        ("📚", "Auto Citations", "Every answer cites exact pages"),
        ("🤖", "Multi-Agent", "Specialized agents per intent"),
        ("📝", "Smart Notes", "Auto-generated summaries"),
        ("🧩", "Quiz Generator", "Test your knowledge"),
    ]
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div style='background:#f9fafb; border-radius:12px; padding:1rem; 
                            margin-bottom:0.5rem; border:1px solid #e5e7eb;'>
                  <div style='font-size:1.5rem;'>{icon}</div>
                  <div style='font-weight:700; font-size:0.9rem;'>{title}</div>
                  <div style='color:#6b7280; font-size:0.8rem;'>{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── Chat History ───────────────────────────────────────────────────────────────

def _render_chat_history():
    """Render conversation messages."""
    for i, msg in enumerate(st.session_state.messages):
        role = msg["role"]
        with st.chat_message(role, avatar="🧑" if role == "user" else "🤖"):
            st.markdown(msg["content"])

            # Show metadata for assistant messages
            if role == "assistant" and msg.get("meta"):
                _render_meta_badges(msg["meta"])

            # Show sources
            if role == "assistant" and msg.get("sources"):
                with st.expander(f"📚 {len(msg['sources'])} Source(s)", expanded=False):
                    _render_sources(msg["sources"])

            # Feedback buttons
            if role == "assistant":
                col1, col2, col3 = st.columns([1, 1, 8])
                with col1:
                    if st.button("👍", key=f"up_{i}", help="Good answer"):
                        st.session_state.analytics["satisfaction_scores"].append(1)
                with col2:
                    if st.button("👎", key=f"down_{i}", help="Bad answer"):
                        st.session_state.analytics["satisfaction_scores"].append(0)

    # Suggested follow-ups (shown after first exchange)
    if st.session_state.messages:
        _render_followups()


def _render_meta_badges(meta: dict):
    """Show agent/model metadata as compact badges."""
    agent = meta.get("agent", "?")
    intent = meta.get("intent", "?")
    latency = meta.get("latency_ms", 0)
    sources_count = meta.get("sources_count", 0)

    badge_html = f"""
    <div style='margin-top:0.5rem; display:flex; gap:0.4rem; flex-wrap:wrap;'>
      <span style='background:#dbeafe; color:#1e40af; padding:2px 8px; border-radius:99px; font-size:0.7rem; font-weight:600;'>
        🤖 {agent.title()} Agent
      </span>
      <span style='background:#d1fae5; color:#065f46; padding:2px 8px; border-radius:99px; font-size:0.7rem; font-weight:600;'>
        🎯 {intent}
      </span>
      <span style='background:#fef3c7; color:#92400e; padding:2px 8px; border-radius:99px; font-size:0.7rem; font-weight:600;'>
        ⚡ {latency}ms
      </span>
      <span style='background:#f3e8ff; color:#6b21a8; padding:2px 8px; border-radius:99px; font-size:0.7rem; font-weight:600;'>
        📚 {sources_count} sources
      </span>
    </div>
    """
    st.markdown(badge_html, unsafe_allow_html=True)


def _render_sources(sources: list[dict]):
    """Render source cards in a grid."""
    for src in sources[:6]:
        st.markdown(
            f"""
            <div style='background:#f8fafc; border-left:3px solid #6366f1; 
                        padding:0.6rem 0.8rem; margin-bottom:0.4rem; border-radius:4px;'>
              <div style='font-size:0.8rem; font-weight:700; color:#374151;'>
                📄 {src.get('file', 'Document')} — Page {src.get('page', '?')}
              </div>
              <div style='font-size:0.75rem; color:#6b7280; margin-top:0.2rem;'>
                {src.get('chunk', '')[:180]}…
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_followups():
    """Show clickable suggested follow-up questions."""
    st.markdown(
        "<div style='color:#9ca3af; font-size:0.75rem; margin-top:1rem;'>💡 Suggested follow-ups:</div>",
        unsafe_allow_html=True,
    )
    follow_ups = [
        "Can you elaborate on that?",
        "What are the implications?",
        "Give me a concrete example",
        "Summarize in bullet points",
    ]
    cols = st.columns(len(follow_ups))
    for i, (col, prompt) in enumerate(zip(cols, follow_ups)):
        with col:
            if st.button(prompt, key=f"followup_{i}", use_container_width=True):
                _handle_query(prompt)


# ── Input Area ─────────────────────────────────────────────────────────────────

def _render_input():
    """Render chat input with action buttons."""
    # Starter prompts (only before first message)
    if not st.session_state.messages:
        st.markdown("#### 🚀 Try asking:")
        cols = st.columns(2)
        for i, prompt in enumerate(STARTER_PROMPTS[:4]):
            with cols[i % 2]:
                if st.button(prompt, use_container_width=True, key=f"start_{i}"):
                    _handle_query(prompt.split(" ", 1)[1])  # strip emoji

    # Main input
    query = st.chat_input(
        "Ask anything about your documents…",
        key="chat_input",
    )
    if query:
        _handle_query(query)

    # Action row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🧩 Generate Quiz", use_container_width=True):
            _handle_query("Generate a 5-question quiz from this document")
    with col2:
        if st.button("📋 Summarize All", use_container_width=True):
            _handle_query("Provide a comprehensive executive summary of all documents")
    with col3:
        if st.button("📊 Extract Tables", use_container_width=True):
            _handle_query("Extract and display all tables and numerical data")
    with col4:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Export button
    if st.session_state.messages:
        chat_text = _export_chat()
        st.download_button(
            "⬇️ Export Chat",
            chat_text,
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True,
        )


# ── Query Handler ──────────────────────────────────────────────────────────────

def _handle_query(query: str):
    """Process a query and stream the response."""
    if not st.session_state.orchestrator:
        st.error("No documents loaded. Please upload PDFs first.")
        return

    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.analytics["queries"] += 1

    # Stream assistant response
    with st.chat_message("assistant", avatar="🤖"):
        response_placeholder = st.empty()
        full_response = ""
        meta_data = {}
        sources_data = []

        try:
            t0 = time.perf_counter()
            stream = st.session_state.orchestrator.chat(query, stream=True)

            for chunk in stream:
                chunk_type = chunk.get("type")
                if chunk_type == "token":
                    full_response += chunk["content"]
                    response_placeholder.markdown(full_response + "▌")
                elif chunk_type == "meta":
                    meta_data = chunk["content"]
                elif chunk_type == "sources":
                    sources_data = chunk["content"]

            response_placeholder.markdown(full_response)

            # Show badges and sources inline
            if meta_data:
                _render_meta_badges(meta_data)
            if sources_data:
                with st.expander(f"📚 {len(sources_data)} Source(s)", expanded=False):
                    _render_sources(sources_data)

        except Exception as e:
            full_response = f"❌ Error: {str(e)}\n\nPlease check your API key and try again."
            response_placeholder.error(full_response)

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "meta": meta_data,
        "sources": sources_data,
        "timestamp": datetime.utcnow().isoformat(),
    })
    st.rerun()


def _export_chat() -> str:
    """Export conversation as markdown."""
    lines = [f"# Chat Export — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]
    for msg in st.session_state.messages:
        role = "**You**" if msg["role"] == "user" else "**AI Assistant**"
        lines.append(f"\n{role}:\n{msg['content']}\n")
        if msg.get("sources"):
            lines.append("*Sources:*")
            for src in msg["sources"]:
                lines.append(f"- {src.get('file')} p.{src.get('page')}")
        lines.append("\n---")
    return "\n".join(lines)
