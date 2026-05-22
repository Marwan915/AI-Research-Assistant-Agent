import os
import sys

from core.agent import ResearchAgent

agent = ResearchAgent()

questions = [
    ("What is the difference between Array and Linked List in terms of memory allocation?", "Quick Summary"),
    ("Explain the concept of RESTful APIs and name four main HTTP methods.", "Full Academic Paper Draft"),
    ("ما هو الفرق بين هندسة البرمجيات Frontend وهندسة Backend؟", "Quick Summary"),
    ("كيف يعمل بروتوكول HTTPS على تأمين البيانات المتبادلة عبر الإنترنت؟", "Full Academic Paper Draft")
]

with open("test_results.txt", "w", encoding="utf-8") as f:
    for q, depth in questions:
        print(f"Testing: {q} ({depth})")
        res = agent.run(q, thread_id="test_script")
        # override depth explicitly since run() forces Quick Summary
        agent._run_internal(q, depth, thread_id="test_script")
        res = agent.last_result
        if res and "audited_report" in res:
            f.write(f"\n{'='*50}\nQ: {q}\nMODE: {depth}\n{'-'*50}\n")
            f.write(res["audited_report"])
            f.write(f"\n{'-'*50}\nFACT SKELETON:\n{res.get('research_notes','')}\n")
        else:
            f.write(f"\n{'='*50}\nQ: {q}\nFAILED TO GENERATE.\n")

print("Done. Check test_results.txt")
