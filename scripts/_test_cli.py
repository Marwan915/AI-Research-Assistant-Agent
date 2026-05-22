import os
import sys
import io
from dotenv import load_dotenv

# Force UTF-8 encoding for Windows terminal stdout/stderr to support emojis & Arabic
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

load_dotenv()

from core.agent import ResearchAgent
from core.pdf_chat import PDFChatSession

def test_pdf():
    print("--- TESTING PDF UPLOAD & EMBEDDING ---")
    pdf_path = "t-deed report.pdf"
    if not os.path.exists(pdf_path):
        print(f"File {pdf_path} not found!")
        return

    session = PDFChatSession()
    try:
        print("Loading PDFs...")
        summary = session.load_pdfs([pdf_path], filenames=["t-deed report.pdf"])
        print("Success! Summary:")
        print(summary.encode('ascii', 'ignore').decode('ascii'))
    except Exception as e:
        print(f"PDF Load Failed: {type(e).__name__}: {e}")

def test_agent():
    print("\n--- TESTING AGENT PIPELINE ---")
    agent = ResearchAgent()
    try:
        print("Running pipeline for: 'Explain vehicular edge computing'")
        result = agent.run("Explain vehicular edge computing")
        
        if result and "final_report" in result:
            print("Success! Report excerpt:")
            print(result["final_report"][:500])
        else:
            print(f"Result is None or missing report! Error event: {agent.events.error}")
            
    except Exception as e:
        print(f"Agent Pipeline Failed: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_pdf()
    test_agent()
