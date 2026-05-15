import os
import time
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ==========================================================
# 1. إعدادات المسارات للنسخة المتقدمة
# ==========================================================
DATA_PATH = "knowledge_base"
# مجلد جديد لنحتفظ بالنسخة القديمة كنسخة احتياطية
CHROMA_PATH = "chroma_db_v2" 

print("=" * 60)
print("🚀 [START] Initializing Advanced RAG Database Build (V2)...")
print(f"📂 Source Directory: {DATA_PATH}")
print("=" * 60)

start_time = time.time()

# ==========================================================
# 2. القراءة الآمنة للملفات (Robust Document Ingestion)
# ==========================================================
print("📚 Scanning and reading scientific papers (Filtering corrupted files)...")

documents = []
failed_files = 0
success_files = 0

# المرور على الملفات وقراءتها بشكل فردي لتجنب انهيار النظام بسبب الملفات التالفة
for filename in os.listdir(DATA_PATH):
    if filename.endswith(".pdf"):
        filepath = os.path.join(DATA_PATH, filename)
        try:
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            documents.extend(docs)
            success_files += 1
            print(f"✅ Loaded: {filename[:40]}...")
        except Exception as e:
            # تخطي الملفات التالفة
            print(f"⚠️ Warning: Skipped corrupted file - {filename}")
            failed_files += 1

if not documents:
    print("❌ FATAL ERROR: No valid text extracted from the directory. Exiting.")
    exit()

print("-" * 60)
print(f"✅ Successfully loaded {len(documents)} pages from {success_files} valid PDF files.")
print("-" * 60)

# ==========================================================
# 3. التقطيع إلى فقرات (Semantic Chunking)
# ==========================================================
print("✂️ Chunking documents into processable semantic blocks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,   
    chunk_overlap=200, 
    length_function=len
)
chunks = text_splitter.split_documents(documents)
print(f"✅ Generated {len(chunks)} text chunks ready for vectorization.")

# ==========================================================
# 4. بناء المتجهات باستخدام النموذج الأكاديمي (BGE-Base)
# ==========================================================
print("\n🧠 Initializing BAAI/bge-base-en-v1.5 Embedding Model...")
# نموذج BGE أثقل قليلاً من MiniLM ولكنه يمتلك قدرات هائلة في فهم السياق المعقد
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

print(f"💾 Initializing empty ChromaDB vector store (V2)...")
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)

BATCH_SIZE = 500 
print(f"🔄 Ingesting {len(chunks)} chunks in batches of {BATCH_SIZE}...")

# ضخ البيانات على دفعات للحفاظ على استقرار الذاكرة
for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Vectorizing & Ingesting"):
    batch = chunks[i : i + BATCH_SIZE]
    db.add_documents(batch)

# ==========================================================
# 5. التقرير الختامي
# ==========================================================
end_time = time.time()
exec_mins = int((end_time - start_time) // 60)
exec_secs = int((end_time - start_time) % 60)

print("\n" + "=" * 60)
print(f"🎯 [SUCCESS] Advanced Knowledge Base Fully Operational!")
print(f"📁 Vector database persisted at: ./{CHROMA_PATH}")
print(f"⏱️ Total Build Time: {exec_mins}m {exec_secs}s")
print("=" * 60)