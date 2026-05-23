# 🎓 AI Research Assistant Agent (Web UI Edition) 

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

### ✅ خريطة استيفاء متطلبات المشروع (Project Rubric Mapping)

| الشرط / المهمة (Requirement) | الحالة (Status) | التفاصيل وأين تم تحقيقها في الكود (How it was achieved) |
| :--- | :---: | :--- |
| **المرحلة الأولى (Phase 1)** | | |
| **1. التعديل على نموذج (SLM) باستخدام Unsloth/QLoRA** | ✔️ محقق | تم إنجاز التدريب والتعديل (Fine-Tuning) لنموذج `SciAssistant` مسبقاً عبر منصة Kaggle، والكود متوفر في مجلد `model_training/`. |
| **2. اختيار مجموعة بيانات (Corpus) تناسب الفكرة** | ✔️ محقق | تم اختيار الأبحاث العلمية الأكاديمية (CS Papers) وبنينا عليها قاعدة بيانات المتجهات `chroma_db_v3` و `knowledge_base`. |
| **3. إجراء تقييم للنموذج (LLM as a judge)** | ✔️ محقق | تم بناء ملف `archive_old_experiments/llm_as_judge_eval.py` ليقوم بتقييم جودة إجابات النموذج المحلي آلياً (استخدمنا Gemini كـ Judge). |
| **4. تحويل النموذج المصغر إلى وكيل عبر LangChain** | ✔️ محقق | النموذج المحلي `SciAssistant` يتم تشغيله عبر `OllamaLLM` ويعمل كعقدة (الكاتب/Writer) أساسية ضمن بيئة LangChain. |
| **5. إدارة الحالة (State Management) بـ Pydantic** | ✔️ محقق | تم استخدام Pydantic لتعريف حالة الوكيل (في ملف `core/agent.py`) عبر الكلاس `class AgentState(BaseModel)`. |
| **6. مستودع Github مع وجود Branches (فروع)** | ✔️ محقق | المشروع مرفوع على مستودع قيت هب وفيه فروع متعددة تعكس العمل الجماعي مثل (`main`, `Terminal-Edition`, `local-web-ui`). |
| | | |
| **المرحلة الثانية (Phase 2)** | | |
| **7. بناء Multi-agentic workflow باستخدام LangGraph** | ✔️ محقق | قلب النظام في `core/agent.py` يعتمد على `StateGraph` ويحتوي على 3 وكلاء (عقد): الباحث (Researcher)، الكاتب (Writer)، والمدقق (Validator). |
| **8. دعم الذاكرة المتعددة (Multi-turn memory)** | ✔️ محقق | تم تفعيل الذاكرة باستخدام `MemorySaver` الخاص بـ LangGraph، مما يسمح بحفظ سياق المحادثة (Thread ID) لكل جلسة. |
| **9. إدارة الحالة عبر Pydantic داخل مسار العمل** | ✔️ محقق | الوكلاء يمررون الـ `AgentState` المبني بـ Pydantic بينهم لضمان صحة البيانات (المدخلات، المخرجات، التقرير النهائي، الخ). |
| **10. دمج نظام RAG داخل مسار العمل** | ✔️ محقق | العقدة الأولى (الوكيل الباحث) تستخدم تقنية RAG للبحث الدلالي داخل قاعدة البيانات المحلية `chroma_db_v3` لاستخراج الحقائق والنصوص. |
| **11. معالجة الأخطاء والانهيارات (Error Handling)** | ✔️ محقق | نظامنا يحتوي على `GeminiRouter` الذي يقوم بعمل (Automatic Fallback) في حال فشل نموذج، ويحتوي `arxiv_tool.py` على حماية من الحظر (Rate Limit Shield 429). |
| **12. تزويد الوكلاء بالأدوات (Tools) المناسبة** | ✔️ محقق | الوكلاء مجهزون بأدوات قوية جداً: أداة `search_arxiv` للبحث المباشر، وأدوات لقراءة الـ PDFs المحلية والتفاعل معها. |

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

### ✅ Project Rubric Mapping

| Requirement | Status | How it was achieved in the code |
| :--- | :---: | :--- |
| **Phase 1** | | |
| **1. Fine-tune an SLM using Unsloth/QLoRA** | ✔️ Achieved | The Fine-Tuning for the `SciAssistant` model was completed via Kaggle, and the code is available in the `model_training/` directory. |
| **2. Pick a relevant text Corpus** | ✔️ Achieved | We selected Academic CS Papers and built our vector database `chroma_db_v3` and `knowledge_base` upon them. |
| **3. Do LLM as a judge evaluation** | ✔️ Achieved | We built `archive_old_experiments/llm_as_judge_eval.py` to automatically evaluate the local model's quality (using Gemini as the Judge). |
| **4. Turn the SLM into an agent using LangChain** | ✔️ Achieved | The local `SciAssistant` model runs via `OllamaLLM` and acts as the core (Writer) node within the LangChain environment. |
| **5. Use Pydantic for state management** | ✔️ Achieved | Pydantic was used to define the agent's state (in `core/agent.py`) via `class AgentState(BaseModel)`. |
| **6. GitHub repo with branches for team members** | ✔️ Achieved | The project is hosted on GitHub with multiple branches reflecting teamwork (e.g., `main`, `Terminal-Edition`, `local-web-ui`). |
| | | |
| **Phase 2** | | |
| **7. Build Multi-agentic workflow using LangGraph** | ✔️ Achieved | The system core in `core/agent.py` relies on a `StateGraph` containing 3 nodes: Researcher, Writer, and Validator. |
| **8. Multi-turn memory support** | ✔️ Achieved | Memory is enabled using LangGraph's `MemorySaver`, allowing conversation context (Thread ID) to be saved per session. |
| **9. State management via Pydantic in the workflow** | ✔️ Achieved | Agents pass the Pydantic-based `AgentState` between them to ensure data integrity (inputs, outputs, final reports, etc.). |
| **10. RAG integration in the workflow** | ✔️ Achieved | The first node (Researcher) utilizes RAG for semantic search within the local `chroma_db_v3` database to extract facts. |
| **11. Error Handling** | ✔️ Achieved | Our system features `GeminiRouter` for Automatic Fallback if a model fails, and `arxiv_tool.py` includes a Rate Limit Shield (HTTP 429). |
| **12. Equip agents with appropriate Tools** | ✔️ Achieved | Agents are equipped with powerful tools: `search_arxiv` for live searching, and tools for reading and interacting with local PDFs. |

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
