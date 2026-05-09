"""
Analytics Dashboard Component
==============================
Real-time usage analytics:
  - Query volume over time
  - Agent usage distribution
  - Response time metrics
  - Satisfaction scores
  - Document coverage heatmap
"""

import streamlit as st
import json
from datetime import datetime


def render_analytics():
    """Render the analytics dashboard tab."""
    st.subheader("📊 Session Analytics")

    analytics = st.session_state.analytics
    messages = st.session_state.messages

    # ── Top KPI Metrics ────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)

    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    latencies = [m.get("meta", {}).get("latency_ms", 0) for m in assistant_msgs if m.get("meta")]
    avg_latency = int(sum(latencies) / len(latencies)) if latencies else 0

    scores = analytics.get("satisfaction_scores", [])
    satisfaction = f"{int(sum(scores)/len(scores)*100)}%" if scores else "N/A"

    col1.metric("💬 Total Queries", analytics.get("queries", 0))
    col2.metric("📄 Docs Processed", analytics.get("docs_processed", 0))
    col3.metric("⚡ Avg Latency", f"{avg_latency}ms")
    col4.metric("😊 Satisfaction", satisfaction)
    col5.metric("🧠 Messages", len(messages))

    st.divider()

    # ── Agent Distribution ─────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 🤖 Agent Usage Distribution")
        agent_counts: dict[str, int] = {}
        for msg in assistant_msgs:
            agent = msg.get("meta", {}).get("agent", "unknown")
            agent_counts[agent] = agent_counts.get(agent, 0) + 1

        if agent_counts:
            total = sum(agent_counts.values())
            for agent, count in sorted(agent_counts.items(), key=lambda x: -x[1]):
                pct = count / total
                bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
                st.markdown(
                    f"`{agent:<12}` {bar} **{count}** ({pct:.0%})"
                )
        else:
            st.info("No queries yet.")

    with col_right:
        st.markdown("#### 🎯 Intent Classification")
        intent_counts: dict[str, int] = {}
        for msg in assistant_msgs:
            intent = msg.get("meta", {}).get("intent", "unknown")
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

        if intent_counts:
            total = sum(intent_counts.values())
            INTENT_COLORS = {
                "factual": "🔵",
                "reasoning": "🟣",
                "summary": "🟢",
                "table": "🟡",
                "comparison": "🟠",
                "definition": "🔴",
                "procedure": "⚫",
            }
            for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
                icon = INTENT_COLORS.get(intent, "⚪")
                pct = count / total
                st.markdown(f"{icon} `{intent:<12}` — **{count}** queries ({pct:.0%})")
        else:
            st.info("No queries yet.")

    st.divider()

    # ── Response Time Timeline ─────────────────────────────────────────────────
    st.markdown("#### ⚡ Response Latency Timeline")
    if latencies:
        latency_data = []
        for i, lat in enumerate(latencies):
            latency_data.append({"Query #": i + 1, "Latency (ms)": lat})

        # Simple ASCII chart
        max_lat = max(latencies)
        for i, lat in enumerate(latencies, 1):
            bar_len = int((lat / max_lat) * 40) if max_lat > 0 else 0
            bar = "█" * bar_len
            color = "🟢" if lat < 2000 else ("🟡" if lat < 5000 else "🔴")
            st.markdown(f"Q{i:02d} {color} {bar} {lat}ms")
    else:
        st.info("Latency data will appear after your first query.")

    st.divider()

    # ── Satisfaction Timeline ──────────────────────────────────────────────────
    st.markdown("#### 😊 Feedback Timeline")
    if scores:
        score_display = " ".join("👍" if s == 1 else "👎" for s in scores)
        positive = sum(scores)
        st.markdown(f"{score_display}")
        st.markdown(f"**{positive}/{len(scores)} positive** feedback")
    else:
        st.info("Use 👍/👎 buttons on responses to track satisfaction.")

    st.divider()

    # ── Export Analytics ───────────────────────────────────────────────────────
    st.markdown("#### 📥 Export Analytics Report")
    report = {
        "session_id": st.session_state.session_id,
        "generated_at": datetime.utcnow().isoformat(),
        "summary": {
            "total_queries": analytics.get("queries", 0),
            "docs_processed": analytics.get("docs_processed", 0),
            "avg_latency_ms": avg_latency,
            "satisfaction_rate": satisfaction,
        },
        "agent_distribution": agent_counts,
        "intent_distribution": intent_counts,
        "latencies_ms": latencies,
        "satisfaction_scores": scores,
    }
    st.download_button(
        "⬇️ Download JSON Report",
        data=json.dumps(report, indent=2),
        file_name=f"analytics_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json",
        use_container_width=True,
    )
