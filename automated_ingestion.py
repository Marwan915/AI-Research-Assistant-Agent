import os
import time
import arxiv

# ==========================================================
# 1. إعدادات النظام والتصنيفات (Configuration)
# ==========================================================
# المجلد الذي سيتلقى آلاف الأوراق العلمية
DOWNLOAD_DIR = "knowledge_base"
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

# تصنيفات علوم الحاسب المعتمدة في ArXiv التي سيبحث فيها النظام
search_categories = ["cat:cs.AI", "cat:cs.LG", "cat:cs.DB", "cat:cs.SE", "cat:cs.DS"]

# كم عدد الأبحاث التي تريد سحبها من "كل" تصنيف؟
MAX_RESULTS_PER_CATEGORY = 50

print(f"⚙️ Starting Automated Ingestion Pipeline...")
print(f"📂 Target Directory: {DOWNLOAD_DIR}")
print(f"🎯 Total expected papers: {MAX_RESULTS_PER_CATEGORY * len(search_categories)}\n")
print("=" * 60)

# ==========================================================
# 2. محرك البحث والتحميل (Fetch Engine)
# ==========================================================
total_downloaded = 0

# بدء حساب الوقت من هنا قبل الدخول في حلقة البحث
start_time = time.time()

for category in search_categories:
    print(f"\n🔍 Querying ArXiv for category: {category}...")
    
    # إعداد عميل البحث (Client) لـ ArXiv
    client = arxiv.Client()
    search = arxiv.Search(
        query=category,
        max_results=MAX_RESULTS_PER_CATEGORY,
        sort_by=arxiv.SortCriterion.SubmittedDate # جلب أحدث الأبحاث دائماً
    )
    
    # تنفيذ البحث والمرور على النتائج
    results = client.results(search)
    
    for paper in results:
        # تنظيف اسم الملف من الرموز الممنوعة في الويندوز لتجنب أخطاء الحفظ
        safe_title = "".join([c for c in paper.title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
        filename = f"{safe_title}.pdf"
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        
        # التحقق: إذا كان الملف موجوداً مسبقاً، تخطاه (لتوفير الوقت وعدم التكرار)
        if os.path.exists(filepath):
            print(f"⏭️ Skipping (Already exists): {filename[:50]}...")
            continue
            
        try:
            print(f"⬇️ Downloading: {filename[:50]}...")
            # تحميل الورقة العلمية وحفظها في المجلد
            paper.download_pdf(dirpath=DOWNLOAD_DIR, filename=filename)
            total_downloaded += 1
            
            # ⚠️ مهم جداً: تأخير زمني لمدة 3 ثوانٍ بين كل تحميل
            # سيرفرات ArXiv ستقوم بحظر الـ IP الخاص بك (Rate Limit) إذا قمت بتحميل الملفات بدون توقف.
            time.sleep(3) 
            
        except Exception as e:
            print(f"❌ Error downloading {filename[:30]}: {e}")

# ==========================================================
# 3. حساب الوقت والتقرير النهائي (Summary)
# ==========================================================
# إيقاف المؤقت وحساب المدة المستغرقة
end_time = time.time()
execution_time_seconds = end_time - start_time

# تحويل الثواني إلى دقائق وثوانٍ لتكون القراءة أوضح
execution_minutes = int(execution_time_seconds // 60)
execution_seconds = int(execution_time_seconds % 60)

print("\n" + "=" * 60)
print(f"✅ Ingestion Complete!")
print(f"📥 New papers downloaded successfully: {total_downloaded}")
print(f"📂 Check the '{DOWNLOAD_DIR}' folder.")
print(f"⏱️ Total Execution Time: {execution_minutes} minutes and {execution_seconds} seconds.")
print("=" * 60)