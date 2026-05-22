"""
core/rag_pipeline.py — RAG Pipeline Wrapper

FIXES:
  - Unified CHROMA_PATH to "chroma_db_v3" (matches agent.py — was split between v2/v3)
  - get_embedding_model() now returns GoogleGenerativeAIEmbeddings (matches agent.py)
  - ingest_pdf_to_chroma accepts both embedding types via duck typing
  - Added get_all_sources() helper for sidebar analytics
  - Added delete_source() for knowledge base management
"""

import os
from typing import List, Optional, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Monkey patch to fix batch embedding flattening bug in langchain-google-genai v4
def _patched_embed_documents(self, texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    try:
        # Wrap each text in the expected Content structure to prevent aggregation in gemini-embedding-2
        contents = [{"parts": [{"text": t}]} for t in texts]
        batch_size = 100
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_contents = contents[i : i + batch_size]
            config = self._build_config(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=self.output_dimensionality
            )
            result = self.client.models.embed_content(
                model=self.model,
                contents=batch_contents,
                config=config
            )
            embeddings.extend([list(e.values) for e in result.embeddings])
        return embeddings
    except Exception as e:
        err_msg = str(e).lower()
        if "429" in err_msg or "exhausted" in err_msg or "quota" in err_msg:
            raise RuntimeError(f"Rate Limit / Quota Exceeded during embedding. Please try again later or upgrade your plan. ({e})")
        # Fallback to sequential for other transient errors
        return [self.embed_query(t) for t in texts]
GoogleGenerativeAIEmbeddings.embed_documents = _patched_embed_documents

from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════
# Constants — single source of truth for paths
# ══════════════════════════════════════════════
DEFAULT_CHROMA_PATH = "chroma_db_v3"        # ← FIXED: was "chroma_db_v2" (mismatch with agent.py)
DEFAULT_KNOWLEDGE_BASE = "knowledge_base"
EMBEDDING_MODEL_NAME = "models/gemini-embedding-2"
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 400
BATCH_SIZE = 500


def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """
    Return the shared Google embedding model instance.
    FIXED: was HuggingFaceEmbeddings — now matches agent.py (GoogleGenerativeAIEmbeddings).
    Using two different embedding models for write vs read produces garbage retrieval results.
    """
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)


def load_pdf(filepath: str) -> List:
    """Safely load a single PDF. Returns list of Document objects or [] on failure."""
    try:
        loader = PyPDFLoader(filepath)
        return loader.load()
    except Exception as e:
        print(f"⚠️ Warning: Failed to load {filepath}: {e}")
        return []


def chunk_documents(
    documents: List,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List:
    """Split documents into semantic chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(documents)


def ingest_pdf_to_chroma(
    pdf_path: str,
    chroma_path: str = DEFAULT_CHROMA_PATH,
    embedding_model: Optional[Any] = None,
    source_name: Optional[str] = None,
) -> int:
    """
    Process a single PDF and add it to the persistent ChromaDB.

    Args:
        pdf_path:        Path to the PDF file (must still exist at call time)
        chroma_path:     Path to ChromaDB directory
        embedding_model: Pre-loaded model (avoids reloading if already in memory)
        source_name:     Optional label for metadata (defaults to filename)

    Returns:
        Number of chunks added (0 if failed)

    NOTE: Do NOT delete the pdf_path before calling this function.
    tab_document.py previously called os.unlink() before this returned — that is fixed there.
    """
    emb = embedding_model or get_embedding_model()
    documents = load_pdf(pdf_path)
    if not documents:
        return 0

    fname = source_name or os.path.basename(pdf_path)
    for doc in documents:
        doc.metadata["source_name"] = fname

    chunks = chunk_documents(documents)
    if not chunks:
        return 0

    db = Chroma(persist_directory=chroma_path, embedding_function=emb)

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        db.add_documents(batch)

    return len(chunks)


def get_chroma_stats(chroma_path: str = DEFAULT_CHROMA_PATH) -> dict:
    """Return basic statistics about the persistent ChromaDB."""
    try:
        emb = get_embedding_model()
        db = Chroma(persist_directory=chroma_path, embedding_function=emb)
        count = db._collection.count()
        return {
            "total_chunks": count,
            "path": os.path.abspath(chroma_path),
            "exists": True,
            "status": "healthy" if count > 0 else "empty",
        }
    except Exception as e:
        return {
            "total_chunks": 0,
            "path": chroma_path,
            "exists": False,
            "status": "error",
            "error": str(e),
        }


def get_all_sources(chroma_path: str = DEFAULT_CHROMA_PATH) -> dict:
    """
    Return a dict of {source_name: chunk_count} for all ingested documents.
    Used by sidebar analytics to show what's in the knowledge base.
    """
    try:
        emb = get_embedding_model()
        db = Chroma(persist_directory=chroma_path, embedding_function=emb)
        result = db.get(include=["metadatas"])
        metadatas = result.get("metadatas", [])

        sources: dict = {}
        for m in metadatas:
            src = m.get("source_name") or m.get("source", "Unknown")
            sources[src] = sources.get(src, 0) + 1

        return sources
    except Exception:
        return {}


def delete_source_from_chroma(
    source_name: str,
    chroma_path: str = DEFAULT_CHROMA_PATH,
) -> int:
    """
    Delete all chunks belonging to a specific source document.
    Returns the number of chunks deleted.
    """
    try:
        emb = get_embedding_model()
        db = Chroma(persist_directory=chroma_path, embedding_function=emb)
        result = db.get(include=["metadatas"])

        ids_to_delete = [
            doc_id
            for doc_id, meta in zip(result["ids"], result["metadatas"])
            if meta.get("source_name") == source_name
            or meta.get("source") == source_name
        ]

        if ids_to_delete:
            db.delete(ids=ids_to_delete)

        return len(ids_to_delete)
    except Exception as e:
        print(f"⚠️ Could not delete source '{source_name}': {e}")
        return 0