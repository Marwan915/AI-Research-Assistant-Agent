import os
import sys
import io
from dotenv import load_dotenv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.rag_pipeline import get_chroma_stats, get_all_sources

load_dotenv()

def main():
    stats = get_chroma_stats()
    print("=" * 60)
    print("📊 Current ChromaDB v3 Stats:")
    print(f"   - Status: {stats.get('status')}")
    print(f"   - Total Chunks: {stats.get('total_chunks')}")
    print(f"   - Path: {stats.get('path')}")
    print("=" * 60)
    
    sources = get_all_sources()
    print(f"📂 Total Indexed Documents: {len(sources)}")
    if sources:
        print("\nTop 5 Ingested Documents:")
        sorted_srcs = sorted(sources.items(), key=lambda x: x[1], reverse=True)
        for doc, chunks in sorted_srcs[:5]:
            print(f"   - {doc[:50]}: {chunks} chunks")
    print("=" * 60)

if __name__ == "__main__":
    main()
