"""
main_app.py — Unified CLI Client for SciAssistant AI Research Agent (LangGraph 3-Node)

Wraps the unified 'ResearchAgent' class, ensuring the terminal application
uses the exact same state machine, embedding functions, local Chroma DB collections (v3),
and Sequential Fallback Router as the Streamlit Web Application.
"""

import os
import sys
import io
import time
from datetime import datetime
from dotenv import load_dotenv

# Force UTF-8 encoding for Windows terminal stdout/stderr to support emojis & Arabic
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

load_dotenv()

# Ensure we can load packages from current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.agent import ResearchAgent

def print_banner():
    print("=" * 80)
    print("🚀 [SYSTEM] Booting SciAssistant Research Agent CLI (LangGraph 3-Node)...")
    print("=" * 80)

def main():
    print_banner()

    # 1. Status Callbacks
    def on_status(step: str, detail: str):
        icons = {
            "init": "⚙️",
            "researcher": "🧠",
            "tools": "🛠️",
            "writer": "📝",
            "validator": "⚖️",
        }
        icon = icons.get(step, "🔄")
        print(f"   [{icon} {step.upper()}] {detail}", flush=True)

    def on_fallback(old: str, new: str, err: str):
        print(f"\n⚠️  [FALLBACK] Router switching models: {old} ➡️ {new}", flush=True)
        print(f"   Reason: {err}\n", flush=True)

    # 2. Agent Initialization
    try:
        agent = ResearchAgent(
            on_status=on_status,
            on_fallback=on_fallback,
            chroma_path="chroma_db_v3"
        )
    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
        print("Please check that your GOOGLE_API_KEY is set correctly in the .env file.")
        sys.exit(1)

    print("\n🎓" * 10 + " SciAssistant CLI READY " + "🎓" * 10)
    print("Type your research question below. Type 'exit' or 'quit' to quit.\n")

    thread_id = "marwan_cli_session"

    # 3. Interactive Loop
    while True:
        try:
            user_input = input("[User]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break

        print("-" * 80)
        print(f"🔎 Starting research for: '{user_input}'")
        print("-" * 80)

        t0 = time.time()
        try:
            # Run agent synchronously using the unified StateGraph
            result = agent.run(user_input, thread_id=thread_id)
            elapsed = time.time() - t0

            if not result or "audited_report" not in result:
                print("\n❌ Error: Pipeline executed but failed to return a valid audited report.")
                continue

            # Print audited report
            print("\n" + "⭐" * 20 + " FINAL AUDITED ACADEMIC REPORT " + "⭐" * 20)
            print(result["audited_report"])
            print("⭐" * 71)

            # Save report in multiple formats (Markdown, LaTeX, DOCX)
            print("\n💾 [Logging System] Saving reports...")
            
            # Save MD
            md_path = agent.save_report(user_input, result["audited_report"], format="md")
            print(f"   ✓ Markdown: {md_path}")
            
            # Save LaTeX
            tex_path = agent.save_report(user_input, result["audited_report"], format="tex")
            print(f"   ✓ LaTeX: {tex_path}")

            # Save DOCX
            try:
                docx_path = agent.save_report(user_input, result["audited_report"], format="docx")
                print(f"   ✓ Word (DOCX): {docx_path}")
            except Exception as e:
                print(f"   ⚠️ Word export failed: {e}")

            print(f"\n✨ Ingestion, writing and verification finished successfully in {elapsed:.1f}s.")
            
        except Exception as e:
            print(f"\n❌ Pipeline failed with exception: {type(e).__name__}: {e}")

        print("-" * 80)

if __name__ == "__main__":
    main()