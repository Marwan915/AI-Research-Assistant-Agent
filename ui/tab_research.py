"""
ui/tab_research.py — Global Research Room

FIXES:
  - Removed busy-wait polling loop that read session_state from background thread
    (session_state is NOT thread-safe in Streamlit)
  - Uses agent.events (threading.Event) for proper stage synchronization
  - st.write_stream() is called synchronously AFTER writer_started event fires
  - Validator wait uses agent.events.validator_done instead of checking last_result
  - Status UI uses placeholder.empty() correctly to prevent stacking
  - Added proper error handling for pipeline failures
"""

import streamlit as st
import time
from datetime import datetime
import streamlit.components.v1 as components

def inject_google_translate():
    components.html(
        """
        <script>
        const doc = window.parent.document;
        if (!doc.getElementById('google_translate_element')) {
            const div = doc.createElement('div');
            div.id = 'google_translate_element';
            div.style.position = 'fixed';
            div.style.bottom = '20px';
            div.style.right = '20px';
            div.style.zIndex = '999999';
            doc.body.appendChild(div);

            const script1 = doc.createElement('script');
            script1.type = 'text/javascript';
            script1.innerHTML = `
                function googleTranslateElementInit() {
                    new google.translate.TranslateElement(
                        {pageLanguage: 'en', includedLanguages: 'ar,en', layout: google.translate.TranslateElement.InlineLayout.SIMPLE},
                        'google_translate_element'
                    );
                }
            `;
            doc.body.appendChild(script1);

            const script2 = doc.createElement('script');
            script2.type = 'text/javascript';
            script2.src = 'https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit';
            doc.body.appendChild(script2);
        }
        </script>
        """,
        height=0,
        width=0,
    )



def _render_pipeline_steps(
    placeholders: dict,
    researcher_state: str = "pending",
    writer_state: str = "pending",
    validator_state: str = "pending",
    researcher_detail: str = "",
    writer_detail: str = "",
    validator_detail: str = "",
):
    """Render the three pipeline steps into their respective placeholders."""
    steps = {
        "researcher": {
            "icon": "🔍",
            "label": "Phase 1 — Knowledge Retrieval",
            "state": researcher_state,
            "detail": researcher_detail,
        },
        "writer": {
            "icon": "✍️",
            "label": "Phase 2 — SciAssistant Drafting",
            "state": writer_state,
            "detail": writer_detail,
        },
        "validator": {
            "icon": "⚖️",
            "label": "Phase 3 — Citation Audit",
            "state": validator_state,
            "detail": validator_detail,
        },
    }

    icons = {"pending": "○", "active": "◉", "done": "✓", "error": "✗"}

    for key, cfg in steps.items():
        ph = placeholders[key]
        state = cfg["state"]
        icon_char = icons.get(state, "○")
        detail_html = f'<div class="step-detail">{cfg["detail"]}</div>' if cfg["detail"] else ""

        ph.markdown(
            f"""<div class="pipeline-step {state}">
                <div class="step-icon {state}">{icon_char}</div>
                <div>
                    <div>{cfg["icon"]} {cfg["label"]}</div>
                    {detail_html}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )


def _render_report_outputs(result: dict, index: int = 0):
    if not result:
        return
        
    agent = st.session_state.agent
    final_report = result.get("audited_report", "")
    translated_query = result.get("translated_query", "Unknown query")
    elapsed_total = result.get("elapsed_total", 0.0)
    now_str = result.get("now_str", "")
    model_name = result.get("model", "Unknown model")
    prompt = result.get("prompt", "")
    depth = result.get("depth", "Quick Summary")

    # Report display
    st.markdown(
        f"""<div class="report-container" style="margin-top: 0.5rem; margin-bottom: 1rem;">
            <div class="report-header">
                <span class="report-title-text">📜 Verified Academic Report</span>
                <span class="report-meta">{now_str} · {model_name} · {depth}</span>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(f'<div dir="auto">{final_report}</div>', unsafe_allow_html=True)

    # Research Notes expander (Transparency Matrix)
    with st.expander("🔬 View Research Notes (Fact Skeleton)", expanded=False):
        st.markdown("*Raw facts gathered before writing — your transparency audit trail.*")
        st.markdown(result.get("research_notes", "No notes available."))

    # Caption + Download
    col1, col2, col3 = st.columns([2.5, 1, 1])
    with col1:
        st.caption(
            f"⏱️ Completed in **{elapsed_total:.1f}s** · "
            f"Model: **{model_name}** · "
            f"Query: *{translated_query[:60]}…*"
        )
    
    html_content = f"""<!DOCTYPE html>
<html dir="auto">
<head>
<meta charset="utf-8">
<title>Research Report</title>
<style>
    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; max-width: 800px; margin: 40px auto; padding: 20px; color: #333; }}
    h1, h2, h3 {{ color: #1e293b; }}
    pre {{ background: #f1f5f9; padding: 15px; border-radius: 8px; overflow-x: auto; }}
    code {{ background: #f1f5f9; padding: 2px 5px; border-radius: 4px; }}
    .meta {{ color: #64748b; font-size: 0.9em; margin-bottom: 30px; border-bottom: 1px solid #e2e8f0; padding-bottom: 10px; }}
</style>
<!-- Include MathJax for LaTeX equations rendering in HTML -->
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>
<body>
<div class="meta">Generated by SciAssistant on {now_str} | Model: {model_name}</div>
{final_report.replace('\n', '<br>')}
<script>
    // Simple basic formatting for markdown headers and bold
    document.body.innerHTML = document.body.innerHTML
        .replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>')
        .replace(/### (.*?)<br>/g, '<h3>$1</h3>')
        .replace(/## (.*?)<br>/g, '<h2>$1</h2>')
        .replace(/# (.*?)<br>/g, '<h1>$1</h1>');
</script>
</body>
</html>"""

    with col2:
        st.download_button(
            label="📥 HTML",
            data=html_content.encode('utf-8'),
            file_name=f"report_{int(time.time())}.html",
            mime="text/html",
            use_container_width=True,
            key=f"dl_html_{index}_{hash(now_str + prompt)}"
        )
        
    with col3:
        pdf_html_content = html_content.replace('</script>\n</body>', '</script>\n<script>window.onload = function() { window.print(); }</script>\n</body>')
        st.download_button(
            label="📥 PDF (Print)",
            data=pdf_html_content.encode('utf-8'),
            file_name=f"report_print_{int(time.time())}.html",
            mime="text/html",
            use_container_width=True,
            help="Download as HTML that auto-opens the print dialog to save as PDF.",
            key=f"dl_pdf_{index}_{hash(now_str + prompt)}"
        )


def render_research_tab():
    agent = st.session_state.agent
    
    inject_google_translate()

    # ── Handle History Restoration FIRST ──────
    if "restore_history_data" in st.session_state:
        restored = st.session_state.restore_history_data
        if not st.session_state.get("viewing_history"):
            st.session_state._stashed_messages = st.session_state.messages.copy()
            st.session_state.viewing_history = True
        st.session_state.messages = [
            {"role": "user", "content": restored.get("prompt")},
            {"role": "assistant", "is_report": True, "report_data": restored}
        ]
        del st.session_state["restore_history_data"]

    # If viewing history, show a back button
    if st.session_state.get("viewing_history"):
        if st.button("⬅️ Back to Session", key="back_to_session"):
            st.session_state.messages = st.session_state.get("_stashed_messages", [])
            st.session_state.viewing_history = False
            st.rerun()

    # ── Chat History ──────────────────────────
    for i, message in enumerate(st.session_state.messages):
        role = message["role"]
        if message.get("is_report"):
            _render_report_outputs(message.get("report_data"), index=i)
        else:
            with st.chat_message(role):
                st.markdown(
                    f'<div class="chat-bubble {role}" dir="auto">{message["content"]}</div>',
                    unsafe_allow_html=True,
                )

    # ── Settings ──────────────────────────────
    st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
    current_depth = st.radio(
        "📝 Drafting Mode",
        ["Quick Summary", "Full Academic Paper Draft"],
        horizontal=True,
        key="output_depth",
        label_visibility="collapsed"
    )

    # ── Input ─────────────────────────────────
    prompt = st.chat_input(
        "Ask a technical research question… (Arabic or English)"
    )

    if not prompt:
        return

    if "last_report_data" in st.session_state:
        del st.session_state["last_report_data"]

    # Handle exit from history view on new prompt
    if st.session_state.get("viewing_history"):
        st.session_state.messages = st.session_state.get("_stashed_messages", [])
        st.session_state.viewing_history = False

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(
            f'<div class="chat-bubble user" dir="auto">{prompt}</div>',
            unsafe_allow_html=True,
        )

    # ── Pipeline Execution ────────────────────
    with st.chat_message("assistant"):

        # -- Pipeline status card --
        pipeline_container = st.empty()
        with pipeline_container.container():
            st.markdown('<div class="pipeline-card">', unsafe_allow_html=True)
            ph_researcher = st.empty()
            ph_writer = st.empty()
            ph_validator = st.empty()
            st.markdown("</div>", unsafe_allow_html=True)

        placeholders = {
            "researcher": ph_researcher,
            "writer": ph_writer,
            "validator": ph_validator,
        }

        # Initial render
        _render_pipeline_steps(
            placeholders,
            researcher_state="active",
            researcher_detail="Initializing pipeline…",
        )

        start_time = time.time()

        # -- Check Cache for Double Execution Syndrome --
        cached_research_notes = ""
        cached_translated_query = ""
        for i in range(len(st.session_state.messages) - 2, -1, -1):
            msg = st.session_state.messages[i]
            if msg["role"] == "user" and msg["content"] == prompt:
                if i + 1 < len(st.session_state.messages):
                    next_msg = st.session_state.messages[i+1]
                    if next_msg.get("is_report"):
                        cached_research_notes = next_msg["report_data"].get("research_notes", "")
                        cached_translated_query = next_msg["report_data"].get("translated_query", "")
                break

        # -- Launch pipeline in background thread --
        current_depth = st.session_state.get("output_depth", "Quick Summary")
        agent.run_async(
            prompt, 
            output_depth=current_depth, 
            thread_id=f"session_{id(st.session_state)}",
            cached_research_notes=cached_research_notes,
            translated_query=cached_translated_query
        )

        # ── Phase 1: Wait for Researcher ──────
        # Update detail via status callback stored in session
        # The callback stores messages in session_state — safe because
        # Streamlit re-reads session_state on each script run
        researcher_done = agent.events.researcher_done.wait(timeout=180)

        if agent.events.pipeline_failed.is_set():
            err = agent.events.error
            st.error(f"❌ Pipeline failed during research phase: {err}")
            return

        elapsed_research = time.time() - start_time

        _render_pipeline_steps(
            placeholders,
            researcher_state="done",
            researcher_detail=f"Completed in {elapsed_research:.1f}s",
            writer_state="active",
            writer_detail="SciAssistant is writing…",
        )

        # ── Phase 2: Stream Writer Output ─────
        # Wait for writer node to actually start (queue is ready)
        agent.events.writer_started.wait(timeout=30)

        # Placeholder for the streaming text
        stream_placeholder = st.empty()

        # Stream chunks — this blocks until None sentinel is received
        full_draft = ""
        try:
            # Use st.write_stream with a generator
            full_draft = stream_placeholder.write_stream(agent.stream_chunks(timeout=300))
        except Exception as e:
            st.warning(f"⚠️ Streaming display error: {e}. Waiting for completion…")
            agent.events.writer_done.wait(timeout=300)

        elapsed_write = time.time() - start_time - elapsed_research

        _render_pipeline_steps(
            placeholders,
            researcher_state="done",
            researcher_detail=f"Completed in {elapsed_research:.1f}s",
            writer_state="done",
            writer_detail=f"Draft written in {elapsed_write:.1f}s",
            validator_state="active",
            validator_detail="Auditing citations…",
        )

        # ── Phase 3: Wait for Pipeline ────────
        pipeline_done = agent.events.pipeline_done.wait(timeout=120)

        if agent.events.pipeline_failed.is_set() or not pipeline_done:
            err = agent.events.error or "Pipeline timed out."
            st.error(f"❌ Pipeline failed: {err}")
            return

        elapsed_total = time.time() - start_time

        _render_pipeline_steps(
            placeholders,
            researcher_state="done",
            researcher_detail=f"Completed in {elapsed_research:.1f}s",
            writer_state="done",
            writer_detail=f"Draft written in {elapsed_write:.1f}s",
            validator_state="done",
            validator_detail=f"Audit done • Total: {elapsed_total:.1f}s",
        )

        # Clear the streaming placeholder and the pipeline UI to save space
        stream_placeholder.empty()
        pipeline_container.empty()

        # ── Final Report ──────────────────────
        result = agent.last_result
        if not result or "audited_report" not in result:
            st.error("❌ Pipeline completed but no report was returned.")
            return

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        report_data = {
            "audited_report": result["audited_report"],
            "translated_query": result.get("translated_query", prompt),
            "research_notes": result.get("research_notes", "No notes available."),
            "elapsed_total": elapsed_total,
            "now_str": now_str,
            "model": agent.router.current_model,
            "prompt": prompt,
            "depth": current_depth
        }
        st.session_state.last_report_data = report_data

        # Add final report to chat history directly as a rich report card
        st.session_state.messages.append(
            {"role": "assistant", "is_report": True, "report_data": report_data}
        )
        
        st.rerun()
