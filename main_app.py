import os
import time
import sys
import io
import json
import re
from datetime import datetime
from typing import Annotated, List, TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# إعداد الترميز لبيئة ويندوز
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# استيراد المكتبات
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaLLM

# مكتبات بناء الرسم البياني وإدارة الحالة
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from arxiv_tool import search_arxiv

# ==========================================================
# 1. State Management with Pydantic
# ==========================================================
class AgentState(BaseModel):
    input_query: str = Field(..., description="The user's original question")
    translated_query: str = Field(default="", description="The query translated to English (if needed)")
    research_notes: str = Field(default="", description="The extracted facts from Gemini Pro")
    final_report: str = Field(default="", description="The final draft from SciAssistant")
    audited_report: str = Field(default="", description="The verified report after Gemini Pro audit (Zero Hallucination)")

# ==========================================================
# 2. Global Models Initialization
# ==========================================================
load_dotenv()
print("=" * 80)
print("🚀 [SYSTEM] Booting AI Research Assistance Agent (LangGraph 3-Node Edition)...")
print("=" * 80)

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
db = Chroma(persist_directory="chroma_db_v2", embedding_function=embedding_model)
base_retriever = db.as_retriever(search_kwargs={"k": 15})
cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=cross_encoder, top_n=5)
advanced_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

# ==========================================================
# دالة مساعدة: تحويل مخرج Gemini إلى نص سواء كان string أو list
# ==========================================================
def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join(
            item.get('text', '') if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)

# حارس البوابة السريع - للمهام الخفيفة (ترجمة، استخراج كلمات مفتاحية)
router_llm = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-3.1-flash-lite", temperature=0.0)
# العقل المدبر - للاستنتاج العلمي المعقد (gemini-3.1-flash-lite متاح مجاناً)
researcher_llm = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-3.1-flash-lite", temperature=0.0)
# الكاتب المحلي
writer_llm = OllamaLLM(model="SciAssistant") 

# ==========================================================
# 3. Define Internal Tools
# ==========================================================
@tool
def search_local_papers(query: str) -> str:
    """Search the user's local database of academic PDFs."""
    print(f"\n   [Tool Log] 🛠️ Reading local DB for: '{query}'")
    try:
        docs = advanced_retriever.invoke(query)
        if not docs:
            return "No local data found."
        return "\n\n".join([f"[Local Source]: {d.page_content}" for d in docs])
    except Exception as e:
        return f"Error: {e}"

# ==========================================================
# 4. Define Graph Nodes (عقد المعمارية الثلاثية)
# ==========================================================
def researcher_node(state: AgentState):
    """
    Node 1: The Brain gathers facts. Uses Flash-Lite for routing, Pro for synthesis.
    """
    print("\n🧠 [Node 1] Researcher is gathering facts...")
    
    # --- خطوة الترجمة (Translation Gateway) ---
    translation_prompt = f"""
    Analyze the following query. If it is NOT in English, translate it to English accurately. 
    If it is already in English, return the original query.
    Output ONLY the English query, nothing else. No conversational text.
    Query: '{state.input_query}'
    """
    translated_query = _text(router_llm.invoke(translation_prompt).content).strip().replace('"', '')
    print(f"   [System Log] Processing query as: '{translated_query}'")
    # -------------------------------------------
    
    keyword_prompt = f"Extract exactly ONE short search query (max 4 words) from this request for a local database: '{translated_query}'. Reply ONLY with the keywords."
    local_keywords = _text(router_llm.invoke(keyword_prompt).content).strip().replace('"', '')
    
    arxiv_keyword_prompt = f"Extract exactly ONE short search query (max 4 words) from this request for ArXiv: '{translated_query}'. Reply ONLY with the keywords."
    arxiv_keywords = _text(router_llm.invoke(arxiv_keyword_prompt).content).strip().replace('"', '')

    local_data = search_local_papers.invoke(local_keywords)
    print(f"   [Tool Log] 🌐 Contacting ArXiv for: '{arxiv_keywords}'")
    web_data = search_arxiv.invoke({"query": arxiv_keywords, "max_results": 2})
    
    synthesis_prompt = f"""
    You are an elite researcher. Synthesize a highly technical, bulleted "Fact Skeleton" using ONLY the data below.
    Include exact mathematical formulas. Cite sources explicitly. DO NOT write an essay.
    
    [LOCAL DATA]:
    {local_data}
    
    [WEB DATA]:
    {web_data}
    """
    response = researcher_llm.invoke(synthesis_prompt)
    
    # نحتفظ بالنسخة المترجمة في الـ State
    return {"translated_query": translated_query, "research_notes": _text(response.content)}


def writer_node(state: AgentState):
    """
    Node 2: The Hands (SciAssistant) writes the academic report.
    """
    print("\n📝 [Node 2] Writer (SciAssistant) is drafting the report... (This may take a minute)")
    
    slm_prompt = f"""[INST] <<SYS>>
You are an expert academic writer. Your task is to transform the "Research Notes" into a formal academic report.
RULES:
1. You MUST use the exact headings: "# 1. Introduction", "# 2. Mathematical Foundation", "# 3. Recent Advancements", "# 4. Conclusion".
2. You MUST copy all mathematical formulas exactly as they appear in the notes.
3. NEVER talk to the user. NEVER apologize. NEVER say "In conclusion, this report...". Just write the academic text.
<</SYS>>

Research Notes:
{state.research_notes}
[/INST]
# 1. Introduction
"""
    
    print("\n" + "✍️" * 15 + " DRAFTING PROGRESS " + "✍️" * 15)
    final_text = "# 1. Introduction\n" # Because we forced the start
    for chunk in writer_llm.stream(slm_prompt):
        print(chunk, end="", flush=True)
        final_text += chunk
        
    print("\n" + "✍️" * 45)
    
    return {"final_report": final_text}

def validator_node(state: AgentState):
    """
    Node 3: The Auditor (Gemini Pro) - Scientific Validation & Citation Correction
    """
    print("\n⚖️ [Node 3] Auditor (Gemini Pro) is correcting references...")
    
    # الموجه يطلب مخرجات JSON فقط لتقليل استهلاك التوكنز لأقصى حد!
    audit_prompt = f"""
    You are a strict academic editor. I will provide RAW RESEARCH NOTES and a DRAFT REPORT.
    The junior writer may have hallucinated inline citations (e.g., "Smith et al.") and a fake References section.
    
    YOUR TASK:
    1. Identify any fake inline citations in the DRAFT.
    2. Match them to the correct source from the RAW NOTES (e.g., "[Local Source 1]").
    3. Generate a correct, clean "References" section based ONLY on the RAW NOTES.
    
    OUTPUT FORMAT:
    You MUST output ONLY a valid JSON object. No markdown, no conversational text.
    {{
        "replacements": {{
            "hallucinated text 1": "Correct Source Tag",
            "hallucinated text 2": "Correct Source Tag"
        }},
        "true_references": "\\n**References:**\\n1. [Local Source]...\\n2. [Web Data]..."
    }}

    RAW RESEARCH NOTES:
    {state.research_notes}

    DRAFT REPORT:
    {state.final_report}
    """
    
    # جمع المخرج وتحويله إلى نص مضمون
    response = researcher_llm.invoke(audit_prompt)
    
    try:
        # 1. تنظيف وتحليل الـ JSON
        json_str = _text(response.content).replace('```json', '').replace('```', '').strip()
        corrections = json.loads(json_str)
        
        cleaned_report = state.final_report
        
        # 2. عملية الاستبدال مجاناً باستخدام بايثون!
        for fake, real in corrections.get("replacements", {}).items():
            print(f"   [System Log] 🔄 Replacing fake citation: '{fake}' -> '{real}'")
            cleaned_report = cleaned_report.replace(fake, real)
            
        # 3. قص قسم المراجع الوهمي الذي كتبه الموديل المحلي باستخدام التعابير النمطية (Regex)
        cleaned_report = re.split(r'\n(?:\*\*|#\s*)?(?:References|Bibliography|Works Cited).*\n', cleaned_report, flags=re.IGNORECASE)[0]
        
        # 4. إضافة المراجع الحقيقية التي كتبها Gemini Pro في نهاية التقرير
        cleaned_report += "\n\n" + corrections.get("true_references", "")
        
        return {"audited_report": cleaned_report}
        
    except Exception as e:
        print(f"   [System Log] ⚠️ Cost-saving validation failed, using Regex fallback. Error: {e}")
        # خطة بديلة (Fallback) في حال فشل الـ JSON: قص المراجع الوهمية وإلصاق الحقائق
        fallback_report = re.split(r'\n(?:\*\*|#\s*)?(?:References|Bibliography).*\n', state.final_report, flags=re.IGNORECASE)[0]
        fallback_report += "\n\n**Sources Used:**\n" + state.research_notes
        return {"audited_report": fallback_report}

# ==========================================================
# 5. Build the 3-Node Graph
# ==========================================================
workflow = StateGraph(AgentState)

workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("validator", validator_node)

workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "validator") # يرسل المسودة للمدقق
workflow.add_edge("validator", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# ==========================================================
# 6. User Interface
# ==========================================================
def run_assistant():
    config = {"configurable": {"thread_id": "marwan_final_agent_003"}}
    
    print("\n" + "🎓" * 10 + " AI RESEARCH ASSISTANT READY " + "🎓" * 10)
    
    while True:
        user_input = input("\n[User]: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            break
            
        print("-" * 50)
        result = app.invoke(
            {"input_query": user_input},
            config=config
        )
        
        # طباعة التقرير النهائي المُدقق والمضمون 100%
        print("\n" + "⭐" * 20 + " FINAL AUDITED ACADEMIC REPORT " + "⭐" * 20)
        print(result["audited_report"])
        print("⭐" * 65)

        # ----------------------------------------------------------
        # Logging System: حفظ التقرير النهائي
        # ----------------------------------------------------------
        os.makedirs("outputs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join("outputs", f"report_{timestamp}.md")
        
        try:
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(f"# Final Audited Academic Report\n\n")
                f.write(f"**Original Query:** {user_input}\n\n")
                f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("---\n\n")
                f.write(result["audited_report"])
            print(f"\n✅ [Logging System] Report successfully saved to: {output_filename}")
        except Exception as e:
            print(f"\n❌ [Logging System] Failed to save report: {e}")

if __name__ == "__main__":
    run_assistant()