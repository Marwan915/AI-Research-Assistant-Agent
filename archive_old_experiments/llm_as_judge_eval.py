import os
import re
import sys
import io
import time
import json
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI

# إعداد الترميز لبيئة ويندوز
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# إعداد النماذج
load_dotenv()

# 1. نموذج الفاين تيون (Fine-tuned) فقط - محلي مجاني
fine_tuned_model = OllamaLLM(model="SciAssistant")

# 2. القاضي (The Judge) - طلب واحد فقط لكل الأسئلة = لا نستنفد الـ quota
# gemini-3.1-flash-lite: مؤكد يشتغل مجاناً
judge_llm = ChatGoogleGenerativeAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-3.1-flash-lite",
    temperature=0.0
)

# قائمة بـ 10 أسئلة تقنية متنوعة
questions = [
    "Explain the derivation of the loss function in Neural Networks.",
    "What are the differences between TCP and UDP protocols?",
    "Describe the concept of 'Quantum Entanglement' in simple terms.",
    "How does the Paxos algorithm achieve distributed consensus?",
    "Explain the mechanism of Action Potentials in neurons.",
    "What is the time complexity of the QuickSort algorithm in the best and worst cases, and why?",
    "Describe the architecture and advantages of Transformer models in NLP.",
    "How does CRISPR-Cas9 work for gene editing?",
    "Explain the principles of public-key cryptography (RSA).",
    "What is the theory of General Relativity and how does it explain gravity?"
]


def get_sciassistant_response(question, idx):
    """يحصل على إجابة SciAssistant المحلية ويقيس الوقت."""
    print(f"   ⏳ [{idx}/10] Getting response from SciAssistant: {question[:50]}...")
    start = time.time()
    response = fine_tuned_model.invoke(question)
    elapsed = time.time() - start
    print(f"   ✅ Done in {elapsed:.1f}s")
    return response, elapsed


def extract_text(content):
    """
    يستخرج النص من مخرج Gemini سواء كان string أو list.
    gemini-3.1-flash-lite يرجع list of dicts.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get('text', ''))
            elif isinstance(item, str):
                parts.append(item)
        return ''.join(parts)
    return str(content)


def batch_judge(qa_pairs):
    """
    يرسل كل الأسئلة والأجوبة في طلب API واحد لـ Gemini.
    يطلب منه إعادة JSON مع scores وsummary.
    """
    # بناء النص الكامل للتقييم
    qa_block = ""
    for i, (q, a) in enumerate(qa_pairs, 1):
        qa_block += f"""
--- Question {i} ---
Q: {q}
A: {a}
"""

    batch_prompt = f"""You are an expert academic judge evaluating a fine-tuned AI model called SciAssistant.
SciAssistant is a small fine-tuned model designed for academic scientific Q&A.

Below are 10 questions and SciAssistant's answers. Score EACH answer on:
- Technical Accuracy (5 pts): Is the science/logic correct? No hallucinations?
- Academic Tone (3 pts): Professional? No AI artifacts like '$\\boxed{{}}$' or 'The final answer is'?
- Structure (2 pts): Clear headings and organization?
Total = 10 pts per answer.

{qa_block}

YOU MUST respond with ONLY a valid JSON object. No markdown, no explanation outside JSON.
Format:
{{
  "evaluations": [
    {{
      "question_number": 1,
      "score": 7.5,
      "brief_critique": "One sentence summary of strengths and weaknesses."
    }},
    ...
  ],
  "overall_summary": "3-5 sentence final summary of SciAssistant's overall academic performance, strengths, weaknesses, and suitability for academic use."
}}
"""
    print("\n⚖️ Sending ALL 10 Q&A pairs to Gemini in ONE batch call...")
    verdict = judge_llm.invoke(batch_prompt)
    # استخراج النص من مخرج Gemini (سواء string أو list)
    return extract_text(verdict.content)


if __name__ == "__main__":
    total_start_time = time.time()

    print("\n" + "="*55)
    print("🚀 PHASE 1: Collecting SciAssistant answers (local, fast)")
    print("="*55)

    # المرحلة 1: جمع كل الإجابات محلياً (مجاني وسريع)
    qa_pairs = []
    response_times = []
    for i, q in enumerate(questions, 1):
        resp, t = get_sciassistant_response(q, i)
        qa_pairs.append((q, resp))
        response_times.append(t)

    print("\n" + "="*55)
    print("⚖️ PHASE 2: Single Gemini batch evaluation (1 API call)")
    print("="*55)

    # المرحلة 2: تقييم دفعي واحد من Gemini
    scores = []
    evaluations = []
    overall_summary = ""

    try:
        raw_response = batch_judge(qa_pairs)
        # تنظيف الـ JSON من أي markdown
        clean_json = re.sub(r'```(?:json)?|```', '', raw_response).strip()
        data = json.loads(clean_json)

        evaluations = data.get("evaluations", [])
        overall_summary = data.get("overall_summary", "N/A")

        for ev in evaluations:
            if ev.get("score") is not None:
                scores.append(float(ev["score"]))

        print("✅ Gemini batch evaluation complete!")

    except Exception as e:
        print(f"⚠️ Gemini batch evaluation failed: {e}")
        overall_summary = "Could not generate summary due to API error."

    # طباعة النتائج التفصيلية
    print("\n" + "="*55)
    print("📋 DETAILED RESULTS PER QUESTION")
    print("="*55)

    for i, (q, resp) in enumerate(qa_pairs, 1):
        print(f"\n\n🟢 Question {i}/10")
        print(f"❓ {q}")
        print("-"*45)
        print(f"📝 [SciAssistant Response]:\n{resp}")
        print("-"*45)

        # إيجاد التقييم المقابل
        ev = next((e for e in evaluations if e.get("question_number") == i), None)
        if ev:
            score_val = ev.get('score', 'N/A')
            critique = ev.get('brief_critique', 'N/A')
            print(f"📊 [Gemini Critique]: {critique}")
            print(f"⭐ Score: {score_val}/10")
        else:
            print(f"📊 [Gemini Critique]: N/A")
            print(f"⭐ Score: N/A")

        print(f"⏱️ SciAssistant response time: {response_times[i-1]:.2f}s")

    # الملخص النهائي
    total_time = time.time() - total_start_time
    avg_score = sum(scores) / len(scores) if scores else 0.0

    print("\n\n" + "="*55)
    print("🏁 EVALUATION COMPLETED")
    print("="*55)
    print(f"✅ Total Questions Evaluated: {len(questions)}")
    print(f"📊 Scores Collected: {len(scores)}/{len(questions)}")
    print(f"⭐ Average Score: {avg_score:.1f}/10")
    print(f"🕒 Total Time Taken: {total_time:.2f} seconds")
    print(f"🌐 Total Gemini API Calls Used: 1")

    print("\n" + "="*55)
    print("📝 FINAL SUMMARY FROM GEMINI:")
    print("="*55)
    print(f"\n{overall_summary}")