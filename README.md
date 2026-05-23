# 🎓 AI Research Assistant Agent (Terminal Edition) 

> **💡 فرع التطوير الحالي / Current Active Branch:** `local-web-ui` (Premium Streamlit Interface)

---

### 🚀 تشغيل الواجهة الرسومية الجديدة (Launch Web UI)
للحصول على أفضل تجربة بحث وتفاعل مع المستندات، قم بتشغيل الواجهة الرسومية الجديدة:
```bash
streamlit run app.py
```
*ملاحظة: نسخة سطر الأوامر (Terminal) لا تزال متاحة عبر `python main_app.py`.*

---

## 🇸🇦 اللغة العربية (Arabic Version)

مرحباً بك في مشروع **مساعد البحث الأكاديمي الذكي**. هذا المستودع مخصص لبناء عميل ذكي مستقل (Autonomous Agent) قادر على البحث في المصادر المحلية الموثقة ودمجها مع محركات البحث المباشرة لإنتاج تقارير علمية رصينة، دقيقة، ومُدققة آلياً ضد الهلوسة الأكاديمية.

---

### 🗺️ خارطة الطريق للمشروع (Roadmap)
تم تقسيم تطوير هذا المشروع إلى 3 مراحل رئيسية عبر فروع منفصلة في Git:
1.  **فرع `Terminal-Edition`:** نسخة سطر الأوامر (Terminal Interface) المتكاملة.
2.  **فرع `local-web-ui` (الحالي):** واجهة مستخدم محلية فاخرة (Local Web GUI/Dashboard) باستخدام Streamlit.
3.  **الفرع الثالث (لاحقاً):** النسخة السحابية الكاملة المتاحة للجميع (Hosted Online SaaS Production).

---

### 📥 التحميل السريع واختصار الوقت (Fast-Track Download)
إذا أردت البدء فوراً وتجربة البرنامج دون الحاجة لانتظار سحب الأوراق العلمية وبناء قاعدة المتجهات، قمنا بتوفير **مجلد مضغوط جاهز للتحميل** يحتوي على:
*   📂 **`chroma_db_v2`:** قاعدة المتجهات المبنية بالكامل لـ 120+ بحث علمي.
*   📂 **`knowledge_base`:** الأوراق البحثية الكاملة بصيغة PDF.
*   🤖 **`My_AI_Agent`:** ملفات النموذج المحلي المصغر والـ Fine-Tuned GGUF لتشغيلها مباشرة على Ollama.

🔗 **[حمل المجلد الجاهز من Google Drive من هنا](https://drive.google.com/file/d/1y8kXvZtC7J85kDe5gAQQ6W4QP8stvotv/view?usp=sharing)**

> **💡 الاستخدام:** قم بفك ضغط الملف ووضع مجلدي `chroma_db_v2` و `knowledge_base` مباشرة في المجلد الرئيسي للمشروع.

---

### 🧠 كيف يعمل النظام؟ (System Architecture)
يعتمد المحرك على معمارية **LangGraph** بـ 3 عقد متصلة تضمن جودة التقرير:

1.  **العقدة الأولى (Researcher):** 
    يستعمل `gemini-3.1-flash-lite` لترجمة استفسارك وسحب كلمات بحثية دقيقة، ثم ينبش قاعدة البيانات المحلية وقاعدة بيانات ArXiv العالمية لجمع أهم الفقرات والحقائق ويصيغ "الهيكل المعرفي" (Fact Skeleton).
2.  **العقدة الثانية (Writer):** 
    عقدة محلية بالكامل تعتمد على نموذج `SciAssistant` المستضاف عبر Ollama لصياغة التقرير الأكاديمي الطويل وتنسيق المعادلات الرياضية بـ LaTeX.
3.  **العقدة الثالثة (Validator / Auditor):** 
    العقدة الصارمة؛ تأخذ المسودة وتقوم بمطابقة كل اقتباس بالحقائق الأصلية، وتكشف المصادر الوهمية وتصححها آلياً لتضمن تقريراً معتمداً وصادقاً 100%.

---

### 📁 تشريح ملفات المشروع (Project Structure)

*   **`main_app.py`**: المحرك الأساسي والتنفيذي للمشروع والربط بين العقد الثلاث وتوليد التقارير وحفظها.
*   **`arxiv_tool.py`**: أداة البحث المباشر في مستودع الأبحاث العالمي مع حماية مدمجة ضد الحظر (Rate Limit Shield).
*   **`automated_ingestion.py`**: سكريبت لسحب مئات الأوراق العلمية من ArXiv أوتوماتيكياً في تخصصات علوم الحاسب لتغذية مكتبتك.
*   **`build_rag_v2.py`**: العقل المدبر لبناء قاعدة ChromaDB، يقوم بتقطيع النصوص دلالياً وتحويلها لمتجهات باستخدام نموذج `BAAI/bge-base-en-v1.5` الأكاديمي.
*   **`archive_old_experiments/llm_as_judge_eval.py`**: نظام تقييم فائق الكفاءة؛ يقوم بقياس جودة 10 إجابات للنموذج المحلي دفعة واحدة عبر Gemini API لضمان ثبات الجودة الأكاديمية.
*   **`model_training/`**: مجلد يحتوي على كود تدريب النموذج المصغر (Fine-Tuning) ببيانات علمية باستخدام Unsloth و QLoRA.
*   **`.gitignore` / `.gitkeep`**: إعدادات احترافية لحماية المشروع وتسهيل العمل الجماعي دون رفع ملفات ثقيلة.

---

### 🚀 خطوات التشغيل والتجربة (How to Run)

#### خطوة 0: تحميل الكود وفتحه في الـ IDE
قم بفتح الترمينال على جهازك (أو داخل VSCode / PyCharm) ونفذ ما يلي لنسخ المشروع والانتقال للفرع الصحيح:
```bash
# 1. نسخ المشروع من قيت هب
git clone https://github.com/Marwan915/AI-Research-Assistant-Agent.git

# 2. الدخول لمجلد المشروع
cd AI-Research-Assistant-Agent

# 3. الانتقال إلى فرع الواجهة الرسومية (مهم جداً!)
git checkout local-web-ui
```
بعد ذلك، افتح المجلد باستخدام محررك المفضل (مثل `code .` لـ VSCode).

#### خطوة 1: تجهيز البيئة وتثبيت المكتبات
داخل الترمينال الخاص بالـ IDE، أنشئ البيئة الافتراضية وثبّت المكتبات:
```bash
# إنشاء بيئة افتراضية
python -m venv venv

# تفعيل البيئة (نظام ويندوز)
.\venv\Scripts\activate

# تثبيت المكتبات المطلوبة
pip install -r requirements.txt
```

#### خطوة 2: إعداد مفاتيح الـ API
قم بنسخ ملف `.env.example` وتغيير اسمه إلى `.env` ثم ضع مفتاح Gemini API الحقيقي الخاص بك:
```env
GOOGLE_API_KEY=your_real_api_key_here
```

#### خطوة 3: تشغيل النموذج المحلي (Ollama)
بعد تحميل ملفات الموديل المصغر من رابط قوقل درايف ووضعها في مجلد المشروع، قم بفتح Terminal جديد وشغل الأوامر الآتية لبناء النموذج في Ollama تحت اسم `SciAssistant`:
```bash
# بناء النموذج المحلي
ollama create SciAssistant -f Modelfile
```

#### خطوة 4: إطلاق محرك البحث الذكي!
الآن قم بتشغيل البرنامج الرئيسي واستمتع بالتجربة:
```bash
streamlit run app.py
```
سيقوم النظام بحفظ التقارير النهائية تلقائياً في مجلد **`outputs/`** كملفات Markdown غنية.

---
---

## 🇺🇸 English Version

Welcome to the **AI Research Assistant Agent** project. This repository hosts an autonomous intelligent agent designed to perform hybrid searches across local verified sources and live academic databases, synthesizing extensive, hallucination-free scientific reports.

---

### 🗺️ Project Roadmap
Development is split across three core phases, separated by Git branches:
1.  **`Terminal-Edition` Branch:** Fully functional Terminal/CLI edition.
2.  **`local-web-ui` Branch (Current):** Local premium Web GUI/Dashboard powered by Streamlit.
3.  **Third Phase (Later):** Hosted SaaS Production deployment for public usage.

---

### 📥 Fast-Track Execution (Pre-built Dataset)
To skip the lengthy process of downloading academic papers and constructing the vector embedding database, we provide a **pre-packaged ZIP archive** containing:
*   📂 **`chroma_db_v2`:** Fully compiled vector database with 120+ pre-vectorized papers.
*   📂 **`knowledge_base`:** Corresponding PDF documents.
*   🤖 **`My_AI_Agent`:** Pre-configured fine-tuned local GGUF model files for instant Ollama hosting.

🔗 **[Download Pre-built Assets from Google Drive Here](https://drive.google.com/file/d/1y8kXvZtC7J85kDe5gAQQ6W4QP8stvotv/view?usp=sharing)**

> **💡 Usage:** Simply extract the downloaded ZIP file and place the `chroma_db_v2` and `knowledge_base` directories directly into the root of the project directory.

---

### 🧠 System Architecture
The engine coordinates a **3-Node LangGraph** pipeline to enforce scientific rigor:

1.  **Node 1 (Researcher):** 
    Powered by `gemini-3.1-flash-lite`. Translates user input, extracts precise search keywords, inspects the local RAG database alongside ArXiv API endpoints, and synthesizes a raw fact-verified skeleton.
2.  **Node 2 (Writer):** 
    Completely local node utilizing `SciAssistant` hosted via Ollama. It drafts the extensive academic report, formatting advanced mathematical derivations in clean LaTeX syntax.
3.  **Node 3 (Validator / Auditor):** 
    The compliance gatekeeper. It audits the draft, matching every hallucinated citation to a real source from Node 1 via JSON mapping, automatically correcting discrepancies using Python and Regex logic to yield 100% verifiable final reports.

---

### 📁 Project Structure Breakdown

*   **`main_app.py`**: The core execution controller managing LangGraph node execution, configuration, and file operations.
*   **`arxiv_tool.py`**: Custom Search tool querying the ArXiv registry with custom built-in rate-limit defenses.
*   **`automated_ingestion.py`**: Automatic batch downloader scraping target CS papers from ArXiv to populate the knowledge base.
*   **`build_rag_v2.py`**: The indexing engine chunking texts and vectorizing them via `BAAI/bge-base-en-v1.5` into a high-performance ChromaDB persistent store.
*   **`archive_old_experiments/llm_as_judge_eval.py`**: Automated evaluation benchmark utilizing batch calls to Gemini for cost-efficient performance grading.
*   **`model_training/`**: Directory containing the SLM Fine-Tuning Jupyter notebook leveraging Unsloth, QLoRA, and dynamic dataset fetching.
*   **`.gitignore` / `.gitkeep`**: Professional workspace configuration files restricting giant binaries from being uploaded to source control.

---

### 🚀 How to Run and Test

#### Step 0: Clone the Repository and Open IDE
Launch your local Terminal (or VSCode/PyCharm terminal) and run the following commands to clone the repo and switch to the correct branch:
```bash
# 1. Clone the repo
git clone https://github.com/Marwan915/AI-Research-Assistant-Agent.git

# 2. Change directory
cd AI-Research-Assistant-Agent

# 3. Checkout to the Web UI edition branch (Crucial!)
git checkout local-web-ui
```
Afterwards, open the folder inside your preferred IDE (e.g., `code .` for VSCode).

#### Step 1: Environment Setup
Create a virtual environment and install project dependencies:
```bash
# Create Virtual Environment
python -m venv venv

# Activate Virtual Environment (Windows)
.\venv\Scripts\activate

# Install necessary libraries
pip install -r requirements.txt
```

#### Step 2: Configure Environment Variables
Copy the `.env.example` file, rename the new file to `.env`, and input your official Gemini API key:
```env
GOOGLE_API_KEY=your_real_api_key_here
```

#### Step 3: Instantiate Local LLM (Ollama)
After downloading the model asset from the Google Drive link provided above, open a Terminal window and run the creation command to register `SciAssistant` into your local Ollama instance:
```bash
# Build model configuration
ollama create SciAssistant -f Modelfile
```

#### Step 4: Run the Application
Everything is perfectly configured. Run the main application loop:
```bash
streamlit run app.py
```
Generated markdown reports will be automatically archived with accurate timestamps in the **`outputs/`** directory for structured viewing.
