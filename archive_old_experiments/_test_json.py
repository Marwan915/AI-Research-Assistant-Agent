import os, re, json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

judge_llm = ChatGoogleGenerativeAI(
    api_key=os.getenv('GOOGLE_API_KEY'),
    model='gemini-3.1-flash-lite',
    temperature=0.0
)

msg = 'Reply with ONLY valid JSON, no markdown: {"test": true, "score": 7}'
test = judge_llm.invoke(msg)
raw = test.content
print("RAW:", repr(raw[:500]))
clean = re.sub(r'```(?:json)?|```', '', raw).strip()
print("CLEAN:", repr(clean[:200]))
try:
    data = json.loads(clean)
    print("PARSED OK:", data)
except Exception as e:
    print("PARSE ERROR:", e)
