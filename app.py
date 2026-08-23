"""
Streamlit UI for the multi-agent research pipeline defined in pipeline.py / agents.py.

Run with:
    streamlit run app.py
"""

import time
from datetime import datetime

import streamlit as st

from agents import (
    build_reader_agent,
    build_search_agent,
    writer_chain,
    critic_chain,
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLING
# ============================================================
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #1a1f2e 0%, #0e1117 60%);
    }
    section[data-testid="stSidebar"] {
        background-color: #131722;
        border-right: 1px solid #262c3a;
    }
    .agent-card {
        background: linear-gradient(135deg, #1c2333 0%, #171c29 100%);
        border: 1px solid #2a3145;
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 14px;
    }
    .agent-title {
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        color: #8ea2ff;
        margin-bottom: 6px;
    }
    .pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-left: 8px;
    }
    .pill-done { background: #16351f; color: #4ade80; border: 1px solid #21532c; }
    .pill-run  { background: #362a13; color: #fbbf24; border: 1px solid #5c451c; }
    .pill-wait { background: #24283a; color: #7a819c; border: 1px solid #333952; }
    .hero {
        padding: 28px 30px;
        border-radius: 18px;
        background: linear-gradient(120deg, #1e2a4a 0%, #171c2e 100%);
        border: 1px solid #2c3557;
        margin-bottom: 24px;
    }
    .hero h1 { margin-bottom: 4px; }
    .hero p { color: #9aa4c0; margin-top: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE
# ============================================================
if "state" not in st.session_state:
    st.session_state.state = {}
if "history" not in st.session_state:
    st.session_state.history = []
if "running" not in st.session_state:
    st.session_state.running = False

STEP_LABELS = ["Search", "Read / Scrape", "Write", "Critique"]

if "step_status" not in st.session_state:
    st.session_state.step_status = {label: "wait" for label in STEP_LABELS}


def pill(status: str) -> str:
    mapping = {
        "done": ("Done", "pill-done"),
        "run": ("Running", "pill-run"),
        "wait": ("Pending", "pill-wait"),
    }
    text, cls = mapping[status]
    return f'<span class="pill {cls}">{text}</span>'


def truncate(text: str, max_chars: int) -> str:
    """Trim text to a safe character budget to avoid TPM/token limit errors."""
    if text is None:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n...[truncated to stay within model token limits]"


def extract_text(result):
    """Best-effort extraction of plain text from an agent/chain result."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        if "messages" in result:
            try:
                return result["messages"][-1].content
            except Exception:
                pass
        for key in ("content", "output", "text"):
            if key in result:
                return str(result[key])
        return str(result)
    if hasattr(result, "content"):
        return result.content
    return str(result)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🧠 Research System")
    st.caption("Multi-agent pipeline: Search → Read → Write → Critique")

    st.divider()
    st.markdown("#### Past Topics")
    if st.session_state.history:
        for h in reversed(st.session_state.history[-10:]):
            st.markdown(f"- {h}")
    else:
        st.caption("No topics researched yet.")

    st.divider()
    st.markdown("#### Model Limits")
    st.caption("Lower this if you hit a 413 'Request too large' / TPM rate-limit error from your model provider.")
    max_chars = st.slider(
        "Max characters sent per LLM call",
        min_value=1500,
        max_value=12000,
        value=5000,
        step=500,
        disabled=st.session_state.running,
    )

    st.divider()
    if st.button("🗑️ Clear session", use_container_width=True):
        st.session_state.state = {}
        st.session_state.step_status = {label: "wait" for label in STEP_LABELS}
        st.rerun()

# ============================================================
# HERO / INPUT
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>🧠 Multi-Agent Research System</h1>
        <p>Enter a topic and watch the Search, Reader, Writer, and Critic agents work in real time.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([5, 1])
with col1:
    topic = st.text_input(
        "Research topic",
        placeholder="e.g. Latest advances in solid-state batteries",
        label_visibility="collapsed",
        disabled=st.session_state.running,
    )
with col2:
    run_clicked = st.button(
        "🚀 Run",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.running,
    )

# ============================================================
# STATUS STRIP
# ============================================================
status_cols = st.columns(4)
status_placeholders = {}
for c, label in zip(status_cols, STEP_LABELS):
    with c:
        ph = st.empty()
        status_placeholders[label] = ph
        ph.markdown(
            f'<div class="agent-card"><div class="agent-title">{label}{pill(st.session_state.step_status[label])}</div></div>',
            unsafe_allow_html=True,
        )


def set_step(label, status):
    st.session_state.step_status[label] = status
    status_placeholders[label].markdown(
        f'<div class="agent-card"><div class="agent-title">{label}{pill(status)}</div></div>',
        unsafe_allow_html=True,
    )


# ============================================================
# OUTPUT TABS
# ============================================================
tab_search, tab_read, tab_report, tab_critic = st.tabs(
    ["🔎 Search Results", "📄 Scraped Content", "📝 Report", "🧐 Critic Feedback"]
)

search_ph = tab_search.empty()
read_ph = tab_read.empty()
report_ph = tab_report.empty()
critic_ph = tab_critic.empty()


def render_existing():
    s = st.session_state.state
    if s.get("search_results"):
        search_ph.markdown(s["search_results"])
    if s.get("scraped_content"):
        read_ph.markdown(s["scraped_content"])
    if s.get("report"):
        report_ph.markdown(s["report"])
        tab_report.download_button(
            "⬇️ Download report (.md)",
            s["report"],
            file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        )
    if s.get("feedback"):
        critic_ph.markdown(s["feedback"])


render_existing()

# ============================================================
# PIPELINE EXECUTION
# ============================================================
if run_clicked and topic.strip():
    st.session_state.running = True
    st.session_state.state = {}
    st.session_state.step_status = {label: "wait" for label in STEP_LABELS}
    state = st.session_state.state

    try:
        # ---------------- STEP 1: SEARCH ----------------
        set_step("Search", "run")
        with st.spinner("Search agent is looking for recent, reliable sources..."):
            search_agent = build_search_agent()
            search_result = search_agent.invoke(
                {
                    "messages": [
                        (
                            "user",
                            f"Find recent, reliable and detailed information about: {topic}",
                        )
                    ]
                }
            )
            state["search_results"] = extract_text(search_result)
        set_step("Search", "done")
        search_ph.markdown(state["search_results"])

        # ---------------- STEP 2: READ / SCRAPE ----------------
        set_step("Read / Scrape", "run")
        with st.spinner("Reader agent is scraping the top resource..."):
            reader_agent = build_reader_agent()
            reader_result = reader_agent.invoke(
                {
                    "messages": [
                        (
                            "user",
                            f"Based on the following search results about '{topic}', "
                            f"pick the most relevant URL and scrape it for deeper content.\n\n"
                            f"Search Results:\n{state['search_results'][:800]}",
                        )
                    ]
                }
            )
            state["scraped_content"] = truncate(extract_text(reader_result), max_chars)
        set_step("Read / Scrape", "done")
        read_ph.markdown(state["scraped_content"])

        # ---------------- STEP 3: WRITE ----------------
        set_step("Write", "run")
        with st.spinner("Writer agent is drafting the research report..."):
            # Split the char budget between the two sources so neither dominates
            half_budget = max_chars // 2
            research_combined = (
                f"SEARCH RESULTS:\n{truncate(state['search_results'], half_budget)}\n\n"
                f"DETAILED SCRAPED CONTENT:\n{truncate(state['scraped_content'], half_budget)}"
            )
            report_result = writer_chain.invoke(
                {"topic": topic, "research": research_combined}
            )
            state["report"] = extract_text(report_result)
        set_step("Write", "done")
        report_ph.markdown(state["report"])
        tab_report.download_button(
            "⬇️ Download report (.md)",
            state["report"],
            file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        )

        # ---------------- STEP 4: CRITIQUE ----------------
        set_step("Critique", "run")
        with st.spinner("Critic agent is reviewing the report..."):
            feedback_result = critic_chain.invoke(
                {"report": truncate(state["report"], max_chars)}
            )
            state["feedback"] = extract_text(feedback_result)
        set_step("Critique", "done")
        critic_ph.markdown(state["feedback"])

        st.session_state.history.append(topic)
        st.success("Pipeline complete! Explore the tabs above for full results.")

    except Exception as e:
        msg = str(e)
        if "413" in msg or "rate_limit_exceeded" in msg or "tokens per minute" in msg:
            st.error(
                "Pipeline failed: the request was too large for your model's "
                "token-per-minute limit. Lower 'Max characters sent per LLM call' "
                "in the sidebar and try again."
            )
        else:
            st.error(f"Pipeline failed: {e}")

    finally:
        st.session_state.running = False

elif run_clicked and not topic.strip():
    st.warning("Please enter a topic before running the pipeline.")