"""
build_rag_v3.py — Incremental RAG Database Builder for chroma_db_v3

Scans the 'knowledge_base' folder and imports PDFs into ChromaDB.
Checks existing documents in the database to support resume/incremental loads
(avoids re-embedding previously indexed files to save quota).
"""

import os
import time
import sys
import io
from dotenv import load_dotenv

# Force UTF-8 encoding for Windows terminal stdout/stderr to support emojis & Arabic
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Ensure we can load packages from current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.rag_pipeline import (
    get_all_sources,
    ingest_pdf_to_chroma,
    get_chroma_stats,
    DEFAULT_CHROMA_PATH,
    DEFAULT_KNOWLEDGE_BASE
)

load_dotenv()

def main():
    print("=" * 70)
    print("🚀 [START] Incremental RAG Database Build (chroma_db_v3)")
    print(f"📂 Knowledge Base Directory: {DEFAULT_KNOWLEDGE_BASE}")
    print(f"🗄️ Database Path: {DEFAULT_CHROMA_PATH}")
    print("=" * 70)

    # 1. Check existing sources in database to support incremental ingestion
    print("🔍 Inspecting existing database sources...")
    try:
        existing_sources = get_all_sources()
        print(f"✓ Found {len(existing_sources)} unique documents already in database.")
    except Exception as e:
        print(f"⚠️ Error reading existing sources: {e}. Starting fresh.")
        existing_sources = {}

    # 2. Scan knowledge base directory
    if not os.path.exists(DEFAULT_KNOWLEDGE_BASE):
        print(f"❌ Error: Knowledge base directory '{DEFAULT_KNOWLEDGE_BASE}' not found!")
        sys.exit(1)

    all_files = [f for f in os.listdir(DEFAULT_KNOWLEDGE_BASE) if f.lower().endswith(".pdf")]
    total_files = len(all_files)
    print(f"📋 Found {total_files} PDF files in '{DEFAULT_KNOWLEDGE_BASE}'.")

    # 3. Filter files that are not already indexed
    files_to_ingest = []
    for f in all_files:
        if f in existing_sources:
            print(f"⏭️ Skipping (Already indexed): {f[:50]}...")
        else:
            files_to_ingest.append(f)

    to_ingest_count = len(files_to_ingest)
    print("-" * 70)
    print(f"📥 Files to index: {to_ingest_count} / {total_files}")
    print("-" * 70)

    if to_ingest_count == 0:
        print("🎉 All files are already indexed! Database is fully up to date.")
        stats = get_chroma_stats()
        print(f"📊 Total Database Chunks: {stats.get('total_chunks', 0)}")
        sys.exit(0)

    # 4. Perform Ingestion incrementally
    success_count = 0
    failed_count = 0
    start_time = time.time()

    for idx, f in enumerate(files_to_ingest, 1):
        filepath = os.path.join(DEFAULT_KNOWLEDGE_BASE, f)
        print(f"\n⚡ Ingesting [{idx}/{to_ingest_count}]: {f[:55]}...")
        
        try:
            # Let's count chunks
            t0 = time.time()
            chunks_added = ingest_pdf_to_chroma(filepath, source_name=f)
            t1 = time.time()
            
            if chunks_added > 0:
                print(f"✅ Success! Added {chunks_added} chunks in {t1-t0:.1f}s")
                success_count += 1
            else:
                print(f"⚠️ Skipped: File loaded but produced 0 chunks (corrupted or empty?)")
                failed_count += 1
                
        except Exception as e:
            print(f"❌ Ingestion Failed: {type(e).__name__}: {e}")
            failed_count += 1

        # Small pause between files to prevent overwhelming API rate limits
        time.sleep(0.5)

    # 5. Summary
    end_time = time.time()
    elapsed = end_time - start_time
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)

    stats = get_chroma_stats()
    print("\n" + "=" * 70)
    print("🎯 Ingestion Complete Summary")
    print(f"✅ Successfully indexed: {success_count} files")
    print(f"❌ Failed / Skipped: {failed_count} files")
    print(f"📊 Current Total Chunks in Database: {stats.get('total_chunks', 0)}")
    print(f"⏱️ Elapsed Time: {mins}m {secs}s")
    print("=" * 70)

if __name__ == "__main__":
    main()
