"""
app.py — SciAssistant Main Entry Point

FIXES:
  - on_status callback previously tried to update Streamlit UI directly from
    background thread — this is not allowed. Now it only writes to session_state
    (a dict, which is thread-safe for simple writes) and the UI reads it on rerun.
  - Agent initialization is guarded with a try/except to show a useful error
    instead of a blank page when API key is missing
  - Sidebar model badge updates correctly after fallback via session_state
  - Demo Mode implemented correctly using st.cache_data
  - Session clear also resets agent.last_result to prevent stale reports
"""

import streamlit as st
import os
import threading
import plotly.express as px
import pandas as pd
from dotenv import load_dotenv

from core.agent import ResearchAgent
from core.pdf_chat import PDFChatSession
from core.rag_pipeline import get_chroma_stats
from ui.tab_research import render_research_tab
from ui.tab_document import render_document_tab

# ── Page Config (must be first Streamlit call) ──
st.set_page_config(
    page_title="SciAssistant — AI Research",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()


# ══════════════════════════════════════════════
# CSS Loader
# ══════════════════════════════════════════════
def _load_css():
    with open("ui/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# Session State Initialization
# ══════════════════════════════════════════════
def _init_session():
    """
    Initialize all session state keys. Called once per browser session.
    Agent is a heavyweight object — cached in session_state, not st.cache_resource,
    because it holds threading primitives that must be per-session.
    """
    if "agent" not in st.session_state:
        # Status callback — ONLY writes to session_state (thread-safe for dict writes)
        # Never calls any Streamlit rendering function from this callback!
        def on_status(step: str, detail: str):
            st.session_state["_status_step"] = step
            st.session_state["_status_detail"] = detail

        # Fallback callback — writes new model name to session_state
        def on_fallback(old: str, new: str, err: str):
            st.session_state["active_model"] = new
            # Store for display — will be shown on next Streamlit rerun
            st.session_state["_last_fallback"] = (old, new, err)

        try:
            st.session_state.agent = ResearchAgent(
                on_status=on_status,
                on_fallback=on_fallback,
            )
            st.session_state.active_model = st.session_state.agent.router.current_model
        except Exception as e:
            st.session_state["_init_error"] = str(e)

    if "pdf_session" not in st.session_state:
        agent = st.session_state.get("agent")
        if agent:
            st.session_state.pdf_session = PDFChatSession(
                embedding_model=agent.embedding_model,
                router=agent.router,
            )
        else:
            # Agent failed to init — create pdf_session without shared embedding model
            try:
                st.session_state.pdf_session = PDFChatSession()
            except Exception:
                pass

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "pdf_messages" not in st.session_state:
        st.session_state.pdf_messages = []

    if "active_model" not in st.session_state:
        st.session_state.active_model = "–"




# ══════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════
def _render_sidebar():
    with st.sidebar:
        # Logo
        st.markdown(
            """<div class="sidebar-logo">
                <div class="sidebar-logo-icon">🔬</div>
                <div>
                    <div class="sidebar-logo-text">SciAssistant</div>
                    <div class="sidebar-logo-sub">AI Research Assistant v2.1</div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        # Model status
        active_model = st.session_state.get("active_model", "–")
        st.markdown(
            f"""<div class="model-status">
                <div class="model-pulse"></div>
                <div class="model-info">
                    <div class="model-name">{active_model}</div>
                    <div class="model-label">Active Model</div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        # Show fallback alert if model switched
        if "_last_fallback" in st.session_state:
            old, new, _ = st.session_state.pop("_last_fallback")
            st.toast(f"🔄 Model switched: {old} → {new}", icon="⚠️")

        st.divider()

        # Settings
        st.markdown(
            '<div style="font-family:var(--font-display);font-size:0.7rem;'
            'font-weight:600;color:var(--text-muted);text-transform:uppercase;'
            'letter-spacing:0.1em;margin-bottom:0.6rem;">⚙️ Settings</div>',
            unsafe_allow_html=True,
        )
        
        st.divider()


        
        # Query History
        history = st.session_state.agent.query_history if hasattr(st.session_state.agent, "query_history") else []
        if history:
            with st.expander("🕒 Query History", expanded=False):
                for i, item in enumerate(reversed(history)):
                    st.markdown(f"**{item['timestamp']}**")
                    if st.button(f"🔍 {item['query'][:40]}..." if len(item['query'])>40 else f"🔍 {item['query']}", key=f"hist_btn_{i}", use_container_width=True, help=item['query']):
                        st.session_state.restore_history_data = {
                            "audited_report": item.get("audited_report", ""),
                            "translated_query": item.get("translated_query", item["query"]),
                            "research_notes": item.get("research_notes", ""),
                            "elapsed_total": 0.0,
                            "now_str": item["timestamp"],
                            "model": item["model"],
                            "prompt": item["query"],
                            "depth": item["depth"]
                        }
                        st.rerun()
                    st.markdown(f"<div style='font-size:0.7em; color:var(--text-muted); margin-bottom: 10px;'>{item['model']} · {item['depth']}</div>", unsafe_allow_html=True)
                    
            st.divider()

        # Architecture info
        with st.expander("🏗️ Architecture", expanded=False):
            st.markdown(
                """**Pipeline:** LangGraph · 3 Nodes

**Node 1 — Researcher**
- Gemini (auto-fallback)
- Query translation (AR/EN)
- Local RAG + ArXiv search

**Node 2 — Writer**
- SciAssistant (Ollama local)
- LaTeX math formatting
- Real-time streaming

**Node 3 — Validator**
- Gemini citation audit
- Hallucination correction
- Reference verification

**Models (fallback order)**
1. gemini-2.5-flash ⭐⭐⭐⭐⭐
2. gemini-3.1-flash-lite ⭐⭐⭐⭐
3. gemini-2.5-flash-lite ⭐⭐⭐
4. gemini-2.0-flash ⭐⭐
5. gemini-1.5-flash ⭐
"""
            )

        st.divider()

        # Pipeline info
        with st.expander("🛠️ Execution Pipeline", expanded=False):
            st.markdown(
                """
<div style="padding-top: 5px; padding-bottom: 5px; margin-left: 5px;">
    <div style="margin-bottom: 5px; text-align: center;">
        <div style="font-weight: 600; font-size: 0.85em; background-color: rgba(100, 100, 255, 0.1); padding: 8px; border-radius: 8px; border: 1px solid rgba(100, 100, 255, 0.2);">🌍 Translation Layer<br><span style="font-size: 0.8em; color: var(--text-muted); font-weight: 400;">Gemini 2.5 Flash: Translates AR → EN</span></div>
    </div>
    <div style="text-align: center; color: var(--text-muted); margin-bottom: 5px; font-size: 1.2em;">↓</div>
    <div style="margin-bottom: 5px; text-align: center;">
        <div style="font-weight: 600; font-size: 0.85em; background-color: rgba(100, 255, 100, 0.1); padding: 8px; border-radius: 8px; border: 1px solid rgba(100, 255, 100, 0.2);">🧠 Intelligent Router<br><span style="font-size: 0.8em; color: var(--text-muted); font-weight: 400;">Queries: DuckDuckGo, Wikipedia, ArXiv, RAG</span></div>
    </div>
    <div style="text-align: center; color: var(--text-muted); margin-bottom: 5px; font-size: 1.2em;">↓</div>
    <div style="margin-bottom: 5px; text-align: center;">
        <div style="font-weight: 600; font-size: 0.85em; background-color: rgba(255, 150, 50, 0.1); padding: 8px; border-radius: 8px; border: 1px solid rgba(255, 150, 50, 0.2);">🦴 Synthesis<br><span style="font-size: 0.8em; color: var(--text-muted); font-weight: 400;">Gemini 2.5 Flash: Distills data into Fact Skeleton</span></div>
    </div>
    <div style="text-align: center; color: var(--text-muted); margin-bottom: 5px; font-size: 1.2em;">↓</div>
    <div style="margin-bottom: 5px; text-align: center;">
        <div style="font-weight: 600; font-size: 0.85em; background-color: rgba(255, 50, 100, 0.1); padding: 8px; border-radius: 8px; border: 1px solid rgba(255, 50, 100, 0.2);">✍️ Final Generation<br><span style="font-size: 0.8em; color: var(--text-muted); font-weight: 400;">SciAssistant (Local) or Gemini Flash Lite</span></div>
    </div>
</div>
""", unsafe_allow_html=True
            )

        st.divider()

        # Session controls
        if st.button("🗑️ Clear All Sessions", use_container_width=True, type="secondary"):
            keys_to_clear = [
                "messages", "pdf_messages", "doc_summary",
                "last_pdf_hash", "last_pdf_name", "last_pdf_data",
                "_status_step", "_status_detail", "last_report_data"
            ]
            for key in keys_to_clear:
                st.session_state.pop(key, None)

            if "pdf_session" in st.session_state:
                st.session_state.pdf_session.reset()

            if "agent" in st.session_state:
                st.session_state.agent._last_result = None

            st.success("✅ Sessions cleared.")
            st.rerun()

        # API key warning
        if not os.getenv("GOOGLE_API_KEY"):
            st.warning(
                "⚠️ **GOOGLE_API_KEY** not set in `.env`.\n"
                "Add it to enable Gemini research nodes.",
                icon="🔑",
            )


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════
def main():
    _load_css()
    _init_session()

    # Handle init errors
    if "_init_error" in st.session_state:
        st.error(
            f"❌ **Agent initialization failed:**\n\n"
            f"```\n{st.session_state['_init_error']}\n```\n\n"
            f"Check your `.env` file and make sure `GOOGLE_API_KEY` is set."
        )
        st.stop()

    _render_sidebar()

    # ── Hero Header ──
    st.markdown(
        """<div class="hero-container">
            <div>
                <div class="hero-badge">🔬 AI Research Suite</div>
                <h1 class="hero-title">SciAssistant</h1>
                <p class="hero-subtitle">
                    Autonomous academic research · Hallucination-free reports · Arabic &amp; English
                </p>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Tabs ──
    tab1, tab2 = st.tabs([
        "🌐  Global Research Room",
        "📄  Interactive Document Lab",
    ])

    with tab1:
        render_research_tab()

    with tab2:
        render_document_tab()


if __name__ == "__main__":
    main()
