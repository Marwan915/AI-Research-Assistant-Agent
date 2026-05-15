import os
import time
import sys
import io
from dotenv import load_dotenv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ==========================================================
# 1. Imports (استيراد المكتبات الأساسية)
# ==========================================================
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# استيراد نموذجك المحلي
from langchain_ollama import OllamaLLM 

from arxiv_tool import search_arxiv

# ==========================================================
# 2. Environment Setup
# ==========================================================
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

print("=" * 80)
print("🚀 [START] Initializing The Ultimate Architecture (Brain & Hands)...")
print("=" * 80)

# ==========================================================
# 3. Global Models Initialization (تحميل النماذج)
# ==========================================================
print("⚙️ Loading Local Knowledge Base (ChromaDB V2)...")
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
db = Chroma(persist_directory="chroma_db_v2", embedding_function=embedding_model)
base_retriever = db.as_retriever(search_kwargs={"k": 15})

print("⚙️ Loading Cross-Encoder Reranker...")
cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=cross_encoder, top_n=5)

advanced_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=base_retriever
)

print("⚙️ Connecting to Local Writer Model (SciAssistant)...")
# تحميل الموديل الذي تعبنا في تدريبه ككاتب أكاديمي
local_writer = OllamaLLM(model="SciAssistant") 

# ==========================================================
# 4. Local RAG Tool Definition
# ==========================================================
@tool
def search_local_papers(query: str) -> str:
    """
    Search the LOCAL database of downloaded scientific papers.
    Use this tool to extract highly technical concepts or theories from the user's private database.
    """
    print(f"\n   [System Log] 🛠️ Accessing Local Database for: '{query}'")
    try:
        retrieved_docs = advanced_retriever.invoke(query)
        if not retrieved_docs:
            return "No relevant information found in the local database."
        return "\n\n".join([f"[Local Source {i+1}]: {doc.page_content}" for i, doc in enumerate(retrieved_docs)])
    except Exception as e:
        return f"Error reading local database: {e}"

# ==========================================================
# 5. Agent Engine Setup (العقل المدبر - DeepSeek)
# ==========================================================
print("⚙️ Booting up DeepSeek Orchestrator (The Brain)...")
llm = ChatDeepSeek(
    api_key=DEEPSEEK_API_KEY,
    model="deepseek-chat",
    temperature=0.0
)

tools = [search_local_papers, search_arxiv]

# موجه DeepSeek: نطلب منه استخراج الحقائق على شكل "نقاط" لتكون مسودة قوية
system_instruction = """
You are 'The Brain', an elite academic orchestrator.
Your ONLY job is to research and extract raw, highly accurate facts and mathematical formulas.
Use 'search_local_papers' for foundational theory and 'search_arxiv' for recent advancements.

CRITICAL INSTRUCTIONS:
- DO NOT write a full essay or report.
- Synthesize the information you find into a STRICT, highly technical, bulleted "Fact Skeleton".
- Include exact mathematical formulas if found.
- Explicitly cite the source next to each fact (e.g., [Local Source 1] or [ArXiv: Title]).
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_instruction),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ==========================================================
# 6. The Writer Setup (اليد الكاتبة - SciAssistant)
# ==========================================================
# هذا الموجه يستغل الـ Fine-tuning ليجعل الموديل المحلي يكتب بأسلوب أكاديمي بناءً على مسودة DeepSeek
writer_prompt_template = """
You are 'The Writer', an expert academic author. 
Your task is to take the following raw "Fact Skeleton" provided by the research agent and turn it into a comprehensive, beautifully written, and highly technical academic report.

RULES:
1. Maintain all technical depth, mathematical formulas, and citations provided in the skeleton.
2. Do not invent any new facts. Rely ONLY on the provided skeleton.
3. Use a sophisticated, academic tone with clear headings, introductions, and conclusions.

Raw Fact Skeleton:
{skeleton}

Please write the final academic report:
"""
writer_prompt = ChatPromptTemplate.from_template(writer_prompt_template)

# ==========================================================
# 7. Execution & Testing (الاختبار النهائي للمسار المدمج)
# ==========================================================
test_query = (
    "Search my local database to explain the mathematical foundation of 'Backpropagation'. "
    "Then, search ArXiv for 2 recent papers discussing 'Backpropagation optimization or alternatives'. "
    "Synthesize the findings."
)

print(f"\n🗣️ User Query: {test_query}")
print("-" * 60)
print("🧠 PHASE 1: The Brain (DeepSeek) is researching and building the skeleton... (Please wait)")

start_time = time.time()

try:
    # المرحلة الأولى: استخراج المسودة (DeepSeek)
    research_response = agent_executor.invoke({"input": test_query})
    fact_skeleton = research_response["output"]
    
    print("\n" + "=" * 40)
    print("📝 PHASE 2: The Hands (SciAssistant) is writing the final academic report...")
    print("=" * 40)
    
    # المرحلة الثانية: كتابة التقرير النهائي (SciAssistant)
    # نمرر المسودة التي كتبها DeepSeek إلى نموذجك المحلي
    final_report_chain = writer_prompt | local_writer
    final_academic_report = final_report_chain.invoke({"skeleton": fact_skeleton})

    print("\n" + "🌟" * 30)
    print("✅ THE FINAL ACADEMIC PRODUCT:\n")
    print(final_academic_report)
    print("🌟" * 30)
    
except Exception as e:
    print(f"\n❌ Error during execution: {e}")

end_time = time.time()
print("\n" + "=" * 60)
print(f"⏱️ Total Execution Time: {(end_time - start_time) / 60:.2f} minutes")