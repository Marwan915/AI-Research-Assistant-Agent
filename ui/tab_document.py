"""
ui/tab_document.py — Interactive Document Lab

FIXES:
  - Metric cards raw HTML was injected without unsafe_allow_html=True
  - File uploader check was comparing to stale filename key — now uses hash
  - PDF viewer iframe was broken (Streamlit blocks direct iframe injection)
  - Save to Knowledge Base button actually calls rag_pipeline now
  - chat_container.container(height=...) causes double scrollbar on some versions
    — replaced with standard container + custom CSS max-height
  - Answer was referenced outside its scope (NameError if chat_input returned None)
"""

import streamlit as st
import os
import tempfile
import hashlib

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

            # Document stats
            word_count = len(pdf_session.full_text.split())
            chunk_count = pdf_session.chunk_count

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Words", f"{word_count:,}", help="Total words extracted from all uploaded documents.")
            with col_b:
                st.metric("Vector Chunks", chunk_count, help="Number of semantic chunks stored in the temporary database.")
            with col_c:
                st.metric("Messages", len(st.session_state.get("pdf_messages", [])) // 2, help="Number of interactions in this session.")

            st.divider()

            # Lit Review Matrix Button (Unfair Advantage)
            if st.button("📊 Generate Comparative Literature Review", use_container_width=True):
                st.session_state.lit_matrix = ""
                with st.expander("📊 Comparative Matrix", expanded=True):
                    with st.spinner("Heavy Writer is reasoning..."):
                        matrix = st.write_stream(
                            pdf_session.generate_literature_matrix_stream(st.session_state.agent._writer_llm)
                        )
                        st.session_state.lit_matrix = matrix
                    
            elif "lit_matrix" in st.session_state and st.session_state.lit_matrix:
                with st.expander("📊 Comparative Matrix", expanded=True):
                    st.markdown(st.session_state.lit_matrix)



            if st.button("🗑️ Clear Documents", use_container_width=True):
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
                        st.markdown(f'<div dir="auto">{msg["content"]}</div>', unsafe_allow_html=True)

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
                    {"role": "assistant", "content": answer}
                )
                # Instead of full rerun, we just updated the chat_area!
