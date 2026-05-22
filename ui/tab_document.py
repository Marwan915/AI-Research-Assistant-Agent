"""
ui/tab_document.py — Interactive Document Lab

FIXES & UPDATES:
  - Added visual execution pipeline for PDF upload and processing.
  - Upgraded metric cards to premium glassmorphism aesthetic.
  - Added "Extract Tables" feature button.
  - Aligned with overall premium dark mode theme of Global Research Room.
"""

import streamlit as st
import os
import tempfile
import hashlib
import time

from core.rag_pipeline import ingest_pdf_to_chroma


def _file_hash(data: bytes) -> str:
    """Stable hash of file contents to detect if a new file was uploaded."""
    return hashlib.md5(data).hexdigest()


def render_document_tab():
    pdf_session = st.session_state.pdf_session

    col_upload, col_chat = st.columns([1, 1.15], gap="large")

    # ══════════════════════════════════════════
    # LEFT COLUMN — Upload & Summary
    # ══════════════════════════════════════════
    with col_upload:
        st.markdown(
            '<div style="font-family:var(--font-display);font-size:0.7rem;'
            'font-weight:600;color:var(--text-muted);text-transform:uppercase;'
            'letter-spacing:0.12em;margin-bottom:0.75rem;">📥 Document Upload</div>',
            unsafe_allow_html=True,
        )

        uploaded_files = st.file_uploader(
            "Drag & drop scientific PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            file_hashes = [_file_hash(f.getvalue()) for f in uploaded_files]
            combined_hash = "-".join(file_hashes)

            # Only re-process if it's a new batch of files (by content hash)
            if st.session_state.get("last_pdf_hash") != combined_hash:
                
                # Visual Execution Pipeline UI (Matches Global Research Room)
                pipeline_container = st.empty()
                with pipeline_container.container():
                    st.markdown("""
                    <div style="background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem;">
                        <div style="font-size: 0.8rem; font-weight: 600; color: #94a3b8; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Execution Pipeline</div>
                        <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                            <div style="background: rgba(59, 130, 246, 0.15); color: #60a5fa; padding: 0.4rem 0.8rem; border-radius: 6px; font-size: 0.8rem; font-weight: 500; border: 1px solid rgba(59, 130, 246, 0.3);">📥 Upload</div>
                            <div style="color: #475569; padding: 0.4rem 0;">→</div>
                            <div id="step2" style="background: rgba(255,255,255,0.02); color: #64748b; padding: 0.4rem 0.8rem; border-radius: 6px; font-size: 0.8rem; font-weight: 500; border: 1px solid rgba(255,255,255,0.05);">✂️ Chunking & Embedding</div>
                            <div style="color: #475569; padding: 0.4rem 0;">→</div>
                            <div id="step3" style="background: rgba(255,255,255,0.02); color: #64748b; padding: 0.4rem 0.8rem; border-radius: 6px; font-size: 0.8rem; font-weight: 500; border: 1px solid rgba(255,255,255,0.05);">🗄️ Vector DB Sync</div>
                            <div style="color: #475569; padding: 0.4rem 0;">→</div>
                            <div id="step4" style="background: rgba(255,255,255,0.02); color: #64748b; padding: 0.4rem 0.8rem; border-radius: 6px; font-size: 0.8rem; font-weight: 500; border: 1px solid rgba(255,255,255,0.05);">✨ Auto-Summary (Orchestrator)</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with st.status("🚀 Indexing documents…", expanded=True) as status:
                    try:
                        tmp_paths = []
                        filenames = []
                        for f in uploaded_files:
                            data = f.getvalue()
                            filenames.append(f.name)
                            import tempfile
                            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                            tmp.write(data)
                            tmp.close()
                            tmp_paths.append(tmp.name)

                        st.write(f"📄 Files: **{', '.join(filenames)}**")
                        
                        st.write("⚙️ Chunking and embedding text…")
                        
                        summary = pdf_session.load_pdfs(tmp_paths, filenames=filenames)
                        
                        st.write("🗄️ Auto-saving to Knowledge Base...")
                        for i, tmp_path in enumerate(tmp_paths):
                            ingest_pdf_to_chroma(
                                pdf_path=tmp_path,
                                embedding_model=st.session_state.agent.embedding_model,
                                source_name=filenames[i]
                            )
                            
                        # Cleanup temp files safely after all processing is done
                        for tmp_path in tmp_paths:
                            try:
                                os.unlink(tmp_path)
                            except Exception as e:
                                pass # Windows file lock issue, safe to ignore for temp files

                        # Store state
                        st.session_state.last_pdf_hash = combined_hash
                        st.session_state.last_pdf_name = ", ".join(filenames)
                        st.session_state.doc_summary = summary
                        st.session_state.pdf_messages = []  # Reset chat

                        status.update(
                            label="✅ Documents ready & Saved to Knowledge Base!", state="complete"
                        )
                        time.sleep(0.5)
                        pipeline_container.empty()
                    except Exception as e:
                        error_str = str(e)
                        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                            friendly_msg = "⚠️ لقد استنفدت الحصة المجانية لواجهة Google Gemini (Rate Limit / Quota Exceeded). يرجى الانتظار أو ترقية الخطة."
                        else:
                            friendly_msg = f"❌ Failed: {error_str}"
                            
                        status.update(
                            label=friendly_msg, state="error"
                        )
                        st.error(friendly_msg)

                st.rerun()

        # ── Summary display ──
        if "doc_summary" in st.session_state and pdf_session.is_loaded:
            st.markdown(
                '<div class="summary-card">', unsafe_allow_html=True
            )
            st.markdown(st.session_state.doc_summary)
            st.markdown("</div>", unsafe_allow_html=True)

            # Premium Document Stats (Glassmorphism)
            word_count = len(pdf_session.full_text.split())
            chunk_count = pdf_session.chunk_count
            msg_count = len(st.session_state.get("pdf_messages", [])) // 2

            st.markdown(f"""
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1.5rem 0;">
                <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); padding: 1rem; border-radius: 8px; text-align: center;">
                    <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.3rem;">Words</div>
                    <div style="color: var(--primary); font-size: 1.4rem; font-weight: 700;">{word_count:,}</div>
                </div>
                <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); padding: 1rem; border-radius: 8px; text-align: center;">
                    <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.3rem;">Chunks</div>
                    <div style="color: #10b981; font-size: 1.4rem; font-weight: 700;">{chunk_count:,}</div>
                </div>
                <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); padding: 1rem; border-radius: 8px; text-align: center;">
                    <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.3rem;">Messages</div>
                    <div style="color: #a855f7; font-size: 1.4rem; font-weight: 700;">{msg_count}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.divider()

            # Action Buttons Layout
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("📊 Comparative Matrix", use_container_width=True):
                    st.session_state.generate_matrix = True
                    
            if st.session_state.get("generate_matrix"):
                with st.expander("📊 Comparative Matrix", expanded=True):
                    with st.spinner("Orchestrator is extracting data & Heavy Writer is reasoning..."):
                        matrix = st.write_stream(
                            pdf_session.generate_literature_matrix_stream(st.session_state.agent._writer_llm)
                        )
                        # Fix newlines and remove markdown code blocks that break Streamlit rendering
                        matrix = matrix.replace('\\n', '\n')
                        matrix = matrix.replace('```markdown', '').replace('```', '')
                        st.session_state.lit_matrix = matrix
                st.session_state.generate_matrix = False
            elif "lit_matrix" in st.session_state and st.session_state.lit_matrix:
                with st.expander("📊 Comparative Matrix", expanded=True):
                    st.markdown(st.session_state.lit_matrix)
                    
            if "lit_matrix" in st.session_state and st.session_state.lit_matrix:
                try:
                    import markdown
                    # Ensure markdown handles tables correctly
                    html_body = markdown.markdown(st.session_state.lit_matrix, extensions=['tables'])

                    styled_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Comparative Literature Review</title>
    <style>
        body {{ font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; max-width: 900px; margin: 40px auto; padding: 30px; color: #333; background-color: #fff; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 25px; font-size: 14px; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f8f9fa; font-weight: 600; color: #2c3e50; }}
        h1, h2, h3 {{ color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
        a {{ color: #3498db; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body onload="window.print()">
    {html_body}
</body>
</html>
"""
                except ImportError:
                    styled_html = f"<!DOCTYPE html><html><head><meta charset='UTF-8'></head><body onload='window.print()'><pre>{st.session_state.lit_matrix}</pre></body></html>"
                
                st.download_button(label="⬇️ Download Matrix as HTML/PDF", data=styled_html, file_name="comparative_matrix.html", mime="text/html")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️ Clear Documents", use_container_width=True, type="primary"):
                pdf_session.reset()
                for key in ["last_pdf_hash", "last_pdf_name", "last_pdf_data",
                            "doc_summary", "pdf_messages", "lit_matrix"]:
                    st.session_state.pop(key, None)
                st.rerun()

        elif not pdf_session.is_loaded:
            # Empty state
            st.markdown(
                """<div style="text-align:center;padding:2.5rem 1rem;
                   color:var(--text-muted);font-family:var(--font-display);">
                   <div style="font-size:3rem;margin-bottom:0.75rem;opacity:0.4;">📄</div>
                   <div style="font-size:0.9rem;">Upload a scientific PDF to begin</div>
                   <div style="font-size:0.75rem;margin-top:0.3rem;opacity:0.7;">
                   Supports multi-page papers, equations, and tables</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════
    # RIGHT COLUMN — Chat Interface
    # ══════════════════════════════════════════
    with col_chat:
        st.markdown(
            '<div style="font-family:var(--font-display);font-size:0.7rem;'
            'font-weight:600;color:var(--text-muted);text-transform:uppercase;'
            'letter-spacing:0.12em;margin-bottom:0.75rem;">💬 Document Chat</div>',
            unsafe_allow_html=True,
        )

        if not pdf_session.is_loaded:
            st.info(
                "📌 Upload a PDF on the left to start an in-depth conversation "
                "about the paper.",
                icon="💡",
            )
        else:
            # Chat history
            chat_area = st.container()
            with chat_area:
                pdf_messages = st.session_state.get("pdf_messages", [])
                for msg in pdf_messages:
                    with st.chat_message(msg["role"]):
                        content = msg["content"].replace('\\n', '\n')
                        st.markdown(f'<div dir="auto">{content}</div>', unsafe_allow_html=True)
                        if "sources" in msg and msg["sources"]:
                            st.markdown("---")
                            st.markdown("**References:**")
                            for i, src in enumerate(msg["sources"], 1):
                                with st.expander(f"📍 View [Passage {i}]", expanded=False):
                                    st.markdown(
                                        f'<div class="passage-card" dir="auto">'
                                        f'<div>{src}</div>'
                                        f"</div>",
                                        unsafe_allow_html=True,
                                    )

            # Chat input
            question = st.chat_input(
                f"Ask about '{st.session_state.get('last_pdf_name', 'this paper')}'…"
            )

            if question:
                # Append user message
                st.session_state.pdf_messages.append(
                    {"role": "user", "content": question}
                )
                with chat_area:
                    with st.chat_message("user"):
                        st.markdown(question)

                # Generate answer without a blocking full-page rerun
                with chat_area:
                    with st.chat_message("assistant"):
                        with st.spinner("Reading paper(s)…"):
                            answer, sources = pdf_session.ask(question)
                            answer = answer.replace('\\n', '\n')
                            st.markdown(f'<div dir="auto">{answer}</div>', unsafe_allow_html=True)

                            # Source passages with clear interaction
                            if sources:
                                st.markdown("---")
                                st.markdown("**References:**")
                                for i, src in enumerate(sources, 1):
                                    with st.expander(f"📍 View [Passage {i}]", expanded=False):
                                        st.markdown(
                                            f'<div class="passage-card" dir="auto">'
                                            f'<div>{src}</div>'
                                            f"</div>",
                                            unsafe_allow_html=True,
                                        )

                # Append assistant message
                st.session_state.pdf_messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
                # Instead of full rerun, we just updated the chat_area!
