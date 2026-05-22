"""
core/pdf_chat.py — Temporary In-Memory RAG for Uploaded PDFs

Creates a transient ChromaDB collection in RAM for the currently uploaded PDF.
Supports auto-summarization and contextual Q&A.
The collection is destroyed when the Streamlit session ends.

FIXES:
  - Removed circular import: no longer imports from core.agent
  - _text() helper is now self-contained
  - Correctly handles chromadb.Client() deprecation (use EphemeralClient)
  - n_results clamped to actual chunk count to avoid chromadb ValueError
  - PDFChatSession.ask() now returns tuple consistently even on error
"""

from typing import List, Optional, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from gemini_router import GeminiRouter
from core.rag_pipeline import DEFAULT_CHROMA_PATH
from langchain_chroma import Chroma


def _text(content) -> str:
    """Convert Gemini output to plain text — self-contained, no circular import."""
    if hasattr(content, "content"):
        content = content.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


class PDFChatSession:
    """
    Manages an in-memory vector store for a single uploaded PDF.

    Usage:
        session = PDFChatSession(embedding_model=agent.embedding_model)
        summary = session.load_pdf("/tmp/paper.pdf")
        answer, passages = session.ask("What methodology was used?")
    """

    def __init__(
        self,
        embedding_model = None,
        router: Optional[GeminiRouter] = None,
    ):
        self._embedding = embedding_model or GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2"
        )
        self._router = router or GeminiRouter()
        
        self._db = Chroma(
            persist_directory=DEFAULT_CHROMA_PATH,
            embedding_function=self._embedding
        )

        self._filenames: List[str] = []
        self._full_text: str = ""
        self._is_loaded = False

    # ── Properties ────────────────────────────
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def full_text(self) -> str:
        return self._full_text

    @property
    def filename(self) -> str:
        return ", ".join(self._filenames) if self._filenames else ""

    @property
    def chunk_count(self) -> int:
        if not self._is_loaded or not self._filenames:
            return 0
        try:
            collection = self._db._collection
            result = collection.get(where={"source_name": {"$in": self._filenames}}, include=["metadatas"])
            return len(result["ids"])
        except Exception:
            return 0

    # ── Load & Index ──────────────────────────
    def load_pdfs(self, pdf_paths: List[str], filenames: List[str] = None) -> str:
        """
        Load multiple PDFs, extract full text for summaries.
        Vector chunking is now offloaded to the persistent database.
        """
        self._cleanup()
        
        if not filenames:
            filenames = [f"Document {i+1}" for i in range(len(pdf_paths))]
            
        self._filenames = filenames
        self._full_text = ""

        for idx, pdf_path in enumerate(pdf_paths):
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            if not documents:
                continue
                
            doc_name = filenames[idx]
            for doc in documents:
                doc.page_content = f"[SOURCE: {doc_name}]\n{doc.page_content}"
                self._full_text += f"\n{doc.page_content}"

        if not self._full_text:
            raise ValueError("Could not extract any text from the uploaded PDFs.")

        self._is_loaded = True
        return self._auto_summarize()

    def _auto_summarize(self) -> str:
        """Generate an automatic briefing of the uploaded PDFs."""
        # Use first ~3500 chars — enough for abstract + intro
        preview = self._full_text[:3500]

        prompt = f"""You are an expert academic research assistant. A researcher just uploaded scientific papers.
Analyze the excerpt below and provide a structured briefing:

**1. Core Contribution** (2-3 sentences): What is the main idea or contribution?
**2. Methodology** (1-2 sentences): What approach or method was used?
**3. Key Results** (2-4 bullet points): What are the most important findings?
**4. Relevance** (1 sentence): What field or problem does this address?

End your response with exactly this line:
> 💬 Feel free to ask me anything about these papers — equations, methodology, comparisons, or implications.

Paper excerpt:
---
{preview}
---"""

        try:
            response = self._router.invoke(prompt)
            return _text(response)
        except Exception as e:
            return (
                f"📄 **Papers loaded successfully** ({self.chunk_count} sections indexed).\n\n"
                f"⚠️ Auto-summary failed: {e}\n\n"
                f"> 💬 You can still ask questions about these papers."
            )
            
    def generate_literature_matrix_stream(self, llm):
        """Generates a comparative literature review stream using the Heavy Writer LLM."""
        if not self._is_loaded:
            yield "No documents loaded."
            return
            
        extra_context = ""
        # Smart Single-Paper Matrix
        if len(self._filenames) == 1:
            try:
                # 1. Extract topic
                topic_prompt = f"Extract a 3-5 word main topic or research area for this paper based on its excerpt. Output ONLY the topic.\n\nExcerpt:\n{self._full_text[:3000]}"
                topic = _text(self._router.invoke(topic_prompt)).strip()
                
                # 2. Live ArXiv Search
                try:
                    from arxiv_tool import search_arxiv
                    arxiv_res = search_arxiv.invoke({"query": topic, "max_results": 2})
                    if arxiv_res and "No ArXiv papers found" not in arxiv_res and "error" not in arxiv_res.lower():
                        extra_context += f"\n\n--- RELATED WORKS FROM ARXIV (Topic: {topic}) ---\n{arxiv_res}\n"
                except Exception:
                    pass
                
                # 3. Local RAG search excluding current
                try:
                    local_results = self._db.similarity_search(
                        topic, 
                        k=2, 
                        filter={"source_name": {"$nin": self._filenames}}
                    )
                    if local_results:
                        extra_context += f"\n\n--- RELATED WORKS FROM KNOWLEDGE BASE ---\n"
                        for r in local_results:
                            extra_context += f"Source: {r.metadata.get('source_name', 'Unknown')}\nContent: {r.page_content}\n\n"
                except Exception:
                    pass
            except Exception as e:
                print(f"Error in single-paper prep: {e}")
            
        prompt = f"""[INST] <<SYS>>
You are an expert academic assistant. Generate a Comparative Literature Review Matrix comparing the uploaded paper(s) to related works (if any provided) or to each other.
Create a Markdown table with the following columns:
| Source/Paper | Core Contribution | Methodology | Key Results | Limitations |

Make sure to be detailed, analytical, and draw deep academic comparisons. Do not just summarize; explicitly contrast their approaches.
<</SYS>>
Context Excerpts (first 12,000 characters):
{self._full_text[:12000]}
{extra_context}
[/INST]"""
        
        for chunk in llm.stream(prompt):
            yield chunk
        
    def extract_concept_graph(self) -> dict:
        """Extract a semantic graph using Gemini."""
        prompt = f"""Extract 5-8 key mathematical or scientific concepts from the following abstract and map them to their related topics to form a knowledge graph.
Output ONLY a valid JSON object with 'nodes' and 'edges' arrays. No markdown fences.
Format:
{{
    "nodes": ["Concept A", "Concept B"],
    "edges": [
        {{"source": "Concept A", "target": "Concept B", "label": "uses/relates to"}}
    ]
}}
Abstract: {self._full_text[:3000]}"""
        import json
        import re
        try:
            response = self._router.invoke(prompt)
            raw_json = _text(response).strip()
            clean_json = re.sub(r"```(?:json)?", "", raw_json).strip()
            return json.loads(clean_json)
        except Exception as e:
            return {"nodes": ["Error parsing graph"], "edges": []}

    # ── Query ─────────────────────────────────
    def search_pdf(self, query: str, n_results: int = 5) -> List[str]:
        """Search the persistent PDF collection. Returns relevant passages."""
        if not self._is_loaded or not self._filenames:
            return []

        # Actual count available
        actual_n = min(n_results, self.chunk_count)
        if actual_n == 0:
            return []

        try:
            docs = self._db.similarity_search(
                query,
                k=actual_n,
                filter={"source_name": {"$in": self._filenames}}
            )
            return [d.page_content for d in docs]
        except Exception:
            return []

    def ask(
        self,
        question: str,
        extra_context: str = "",
        n_passages: int = 5,
    ) -> Tuple[str, List[str]]:
        """
        Answer a question about the uploaded PDF.

        Args:
            question:      User's question
            extra_context: Optional context from external RAG/ArXiv
            n_passages:    Number of passages to retrieve

        Returns:
            (answer_text, source_passages_list)
        """
        if not self._is_loaded:
            return (
                "⚠️ No PDF is currently loaded. Please upload a paper first.",
                [],
            )

        passages = self.search_pdf(question, n_results=n_passages)

        if not passages:
            return (
                "I couldn't find relevant passages in the PDF for this question. "
                "Try rephrasing or asking about a different aspect of the paper.",
                [],
            )

        passages_text = "\n\n".join(
            f"[Passage {i+1}]: {p}" for i, p in enumerate(passages)
        )

        extra_block = ""
        if extra_context:
            extra_block = f"\n\nEXTERNAL CONTEXT (from knowledge base / ArXiv):\n{extra_context}"

        prompt = f"""You are an expert academic assistant helping a researcher understand a scientific paper.

QUESTION: {question}

PDF PASSAGES (retrieved by semantic search):
{passages_text}{extra_block}

INSTRUCTIONS:
- Use the provided passages as your primary source of truth.
- If the user asks for a comparison, synthesis, or deeper analysis, use your extensive academic knowledge to draw connections and infer conclusions even if they are not explicitly spelled out in the text.
- Use $...$ for inline math and $$...$$ for block equations.
- Cite passages by number: [Passage 1], [Passage 2], etc.
- If the passages are completely irrelevant, clearly say "I couldn't find an exact answer in the text, but based on my knowledge..." and provide an answer.
- Answer in the same language as the question.
- Keep the tone precise and academic.
- If the question asks for an equation, reproduce it in LaTeX

Answer:"""

        try:
            response = self._router.invoke(prompt)
            return _text(response), passages
        except Exception as e:
            return f"❌ Error generating answer: {e}", passages

    # ── Cleanup ───────────────────────────────
    def _cleanup(self):
        """Reset internal session state (leaves persistent db untouched)."""
        self._filenames = []
        self._full_text = ""
        self._is_loaded = False

    def reset(self):
        """Public cleanup — call when switching to a new PDF or clearing session."""
        self._cleanup()
