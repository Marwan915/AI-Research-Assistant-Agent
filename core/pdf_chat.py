"""
core/pdf_chat.py — Temporary In-Memory RAG for Uploaded PDFs

Creates a transient ChromaDB collection in RAM for the currently uploaded PDF.
Supports auto-summarization and contextual Q&A.
The collection is destroyed when the Streamlit session ends.

FIXES & UPDATES:
  - Orchestrator-Worker pattern for Literature Matrix and Auto-Summary
  - ArXiv + Tavily dual-search for comprehensive literature review
  - Strict LaTeX rules applied to prevent Streamlit rendering crashes
  - In-Text Citation Mapping (Page numbers tracked)
  - Data & Table Extraction feature added
"""

import json
import re
from typing import List, Optional, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from gemini_router import GeminiRouter
from core.rag_pipeline import DEFAULT_CHROMA_PATH
from langchain_chroma import Chroma
from arxiv_tool import search_arxiv
from tavily import TavilyClient
import os


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

def _extract_json(text: str) -> str:
    """Helper to extract JSON from a markdown code block if present."""
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


class PDFChatSession:
    """
    Manages an in-memory vector store for a single uploaded PDF.
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
        
        tavily_key = os.environ.get("TAVILY_API_KEY")
        self._tavily_client = TavilyClient(api_key=tavily_key) if tavily_key else None

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
                # 4. NEW FEATURE: In-Text Citation Mapping (Page numbers)
                page_num = doc.metadata.get('page', 0)
                if isinstance(page_num, int):
                    page_num += 1 # PyPDFLoader is 0-indexed
                
                chunk_header = f"[SOURCE: {doc_name} | PAGE: {page_num}]"
                doc.page_content = f"{chunk_header}\n{doc.page_content}"
                self._full_text += f"\n{doc.page_content}"

        if not self._full_text:
            raise ValueError("Could not extract any text from the uploaded PDFs.")

        self._is_loaded = True
        return self._auto_summarize()

    def _auto_summarize(self) -> str:
        """Generate an automatic briefing of the uploaded PDFs using Orchestrator-Worker pattern."""
        preview = self._full_text[:6000]

        # STEP 1: Orchestrator (Extract structured JSON)
        orchestrator_prompt = f"""You are a scientific data extractor. Read the following paper excerpt and extract the core information into a strict JSON format.
Output ONLY a JSON object with the following keys: "Core_Contribution", "Methodology", "Key_Results", "Relevance", "Extracted_Facts" (an array of specific claims with their page numbers if found).

Excerpt:
---
{preview}
---"""
        try:
            raw_response = self._router.invoke(orchestrator_prompt)
            json_str = _extract_json(_text(raw_response))
            # Just to validate it's parseable
            json.loads(json_str)
        except Exception as e:
            return (
                f"📄 **Papers loaded successfully** ({self.chunk_count} sections indexed).\n\n"
                f"⚠️ Auto-summary extraction failed: {e}\n\n"
                f"> 💬 You can still ask questions about these papers."
            )
            
        # STEP 2: Worker (Write the final summary)
        writer_prompt = f"""You are an expert academic research assistant. 
Use the following structured JSON data extracted from the user's uploaded documents to write a structured briefing.

JSON DATA:
{json_str}

Format your output exactly as follows:
**1. Core Contribution** (2-3 sentences): ...
**2. Methodology** (1-2 sentences): ...
**3. Key Results** (2-4 bullet points): ...
**4. Relevance** (1 sentence): ...

CRITICAL: Do NOT use block LaTeX (`$$`). Do NOT use mathematical symbols like +, =, or > on a separate line. Always embed math conversationally within the paragraph text using plain English where possible, or use simple inline `$` ONLY if absolutely necessary and mathematically sound.

End your response with exactly this line:
> 💬 Feel free to ask me anything about these papers — equations, methodology, comparisons, or implications.
"""
        try:
            final_response = self._router.invoke(writer_prompt)
            return _text(final_response)
        except Exception as e:
            return f"⚠️ Auto-summary writing failed: {e}"
            
    def generate_literature_matrix_stream(self, llm):
        """Generates a comparative literature review stream using the Orchestrator-Worker pattern and ArXiv+Tavily."""
        if not self._is_loaded:
            yield "No documents loaded."
            return
            
        extra_context = ""
        topic = ""
        
        # 1. Extract topic for searching
        try:
            topic_prompt = f"Extract a 3-5 word main topic or research area for this paper based on its excerpt. Output ONLY the topic.\n\nExcerpt:\n{self._full_text[:3000]}"
            topic = _text(self._router.invoke(topic_prompt)).strip()
        except Exception:
            topic = "scientific research"

        # 2. Dual Search: ArXiv (First/Academic) + Tavily (Second/Broad)
        try:
            arxiv_res = search_arxiv.invoke({"query": topic, "max_results": 2})
            if arxiv_res and "No ArXiv papers found" not in arxiv_res and "error" not in arxiv_res.lower():
                extra_context += f"\n\n--- RELATED WORKS FROM ARXIV (Topic: {topic}) ---\n{arxiv_res}\n"
        except Exception:
            pass

        try:
            if self._tavily_client:
                tavily_res = self._tavily_client.search(f"{topic} latest research breakthroughs", max_results=2)
                tavily_results = tavily_res.get('results', [])
                if tavily_results:
                    extra_context += f"\n\n--- RELATED WORKS FROM TAVILY (Topic: {topic}) ---\n"
                    for r in tavily_results:
                        extra_context += f"Title: {r['title']}\nURL: {r['url']}\nContent: {r['content'][:400]}\n\n"
        except Exception:
            pass
            
        # STEP 1: Orchestrator (Extract JSON matrix)
        yield "🧠 Analyzing paper and searching literature...\n\n"
        orchestrator_prompt = f"""You are a scientific data orchestrator. Read the context and the external literature context, then extract a comparative analysis into a strict JSON format.
Output ONLY a JSON array of objects. Each object should represent a paper (either the uploaded paper or an external one) with keys: "Source", "Core_Contribution", "Methodology", "Key_Results", "Limitations", "URLs".

Context Excerpt (Uploaded Paper):
{self._full_text[:8000]}

External Context (ArXiv + Tavily):
{extra_context}"""
        
        try:
            raw_response = self._router.invoke(orchestrator_prompt)
            json_str = _extract_json(_text(raw_response))
            json.loads(json_str) # Validate
        except Exception as e:
            yield f"❌ Orchestrator failed to extract matrix data: {e}"
            return

        # STEP 2: Heavy Writer LLM (Streams final output)
        writer_prompt = f"""[INST] <<SYS>>
You are an expert academic assistant. Generate a formal Comparative Literature Review Matrix comparing the uploaded paper(s) to related works.
You MUST base your output STRICTLY on the following structured JSON data.

JSON DATA:
{json_str}

Create a Markdown table with the following columns:
| Source/Paper | Core Contribution | Methodology | Key Results | Limitations |

After the table, write 1-2 paragraphs of deep academic comparison explicitly contrasting their approaches.
If URLs are provided in the JSON, include them as references below the table.

CRITICAL: You MUST conclude your response with a ### References section. You must list the URLs from Tavily and ArXiv that the Orchestrator provided to you. If you generated a comparative analysis, you must cite the external papers/links used.
When listing References at the end, you MUST format every URL as a clickable Markdown link using this exact syntax: - [Author/Title](URL). Never output raw URLs without the markdown brackets.

CRITICAL INSTRUCTION: DO NOT use any conversational filler or introductory phrases (e.g., do not say 'Here is the matrix' or 'Sure'). Output ONLY the raw Markdown table and the subsequent analysis.

CRITICAL: Do NOT use block LaTeX (`$$`). Do NOT use mathematical symbols like +, =, or > on a separate line. Always embed math conversationally within the paragraph text using plain English where possible, or use simple inline `$` ONLY if absolutely necessary and mathematically sound.
<</SYS>>
[/INST]"""
        
        try:
            for chunk in llm.stream(writer_prompt):
                yield chunk
        except Exception as e:
            yield f"\n❌ Writer failed: {e}"




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
            extra_block = f"\n\nEXTERNAL CONTEXT:\n{extra_context}"

        prompt = f"""You are an expert academic assistant helping a researcher understand a scientific paper.

QUESTION: {question}

PDF PASSAGES (retrieved by semantic search, includes source and page number):
{passages_text}{extra_block}

INSTRUCTIONS:
- Use the provided passages as your primary source of truth.
- IN-TEXT CITATION: Whenever you make a claim based on a passage, you MUST append the Page Number or Chunk ID to the end of the sentence (e.g., "According to the methodology... [PAGE: 4]").
- If the user asks for a comparison, synthesis, or deeper analysis, use your extensive academic knowledge to draw connections and infer conclusions even if they are not explicitly spelled out in the text.
- Cite passages by number: [Passage 1], [Passage 2], etc.
- If the passages are completely irrelevant, clearly say "I couldn't find an exact answer in the text, but based on my knowledge..." and provide an answer.
- Answer in the same language as the question.
- Keep the tone precise and academic.

CRITICAL: Do NOT use block LaTeX (`$$`). Do NOT use mathematical symbols like +, =, or > on a separate line. Always embed math conversationally within the paragraph text using plain English where possible, or use simple inline `$` ONLY if absolutely necessary and mathematically sound.

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

