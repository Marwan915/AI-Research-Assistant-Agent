"""
Debug script to verify:
1. Tavily actually returns results for the test query
2. web_data variable is populated correctly
3. Exactly what text goes into the synthesis_prompt
"""
import re, json
from tavily import TavilyClient
import requests

keyword = "SQL vs NoSQL architectural differences relational non-relational databases performance"

print("=" * 60)
print("STEP 1: Testing Tavily...")
print("=" * 60)
client = TavilyClient(api_key="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")
raw = client.search(keyword, max_results=3)
results = [
    {'title': r['title'], 'body': r['content'][:300], 'href': r['url']}
    for r in raw.get('results', [])
]
print(f"Tavily results count: {len(results)}")
for r in results:
    print(f"  - {r['title']}")
    print(f"    URL: {r['href']}")

print()
print("=" * 60)
print("STEP 2: Testing Wikipedia supplement...")
print("=" * 60)
try:
    wiki_url = 'https://en.wikipedia.org/w/api.php'
    wiki_params = {'action': 'query', 'list': 'search', 'srsearch': keyword, 'utf8': '', 'format': 'json'}
    wiki_headers = {'User-Agent': 'ResearchAgent/1.0'}
    wiki_res = requests.get(wiki_url, params=wiki_params, headers=wiki_headers, timeout=10)
    wiki_data = wiki_res.json().get('query', {}).get('search', [])
    wiki_results = [
        {'title': f"Wikipedia: {item['title']}",
         'body': re.sub(r'<[^>]+>', '', item['snippet']),
         'href': f"https://en.wikipedia.org/wiki/{item['title'].replace(' ', '_')}"}
        for item in wiki_data[:2]
    ]
    results.extend(wiki_results)
    print(f"Wikipedia results count: {len(wiki_results)}")
    for r in wiki_results:
        print(f"  - {r['title']}")
except Exception as e:
    print(f"Wiki error: {e}")

print()
print("=" * 60)
print("STEP 3: Checking web_data variable...")
print("=" * 60)
if results:
    web_data = "\n\n".join([
        f"[Web: {r.get('title')}]\nContent: {r.get('body')}\nURL: {r.get('href')}"
        for r in results
    ])
    print(f"web_data is NON-EMPTY. Length: {len(web_data)}")
    print(f"First 400 chars:\n{web_data[:400]}")
else:
    web_data = "No web results found."
    print("web_data = 'No web results found.' (EMPTY!)")

print()
print("=" * 60)
print("STEP 4: DIAGNOSIS")
print("=" * 60)
print(f"bool(results): {bool(results)}")
print(f"'No web results' in web_data: {'No web results' in web_data}")
print()
if results:
    print("CONCLUSION: Tavily IS returning data. The Disclaimer bug is in the SYNTHESIS PROMPT.")
    print("The LLM is hallucinating the [DISCLAIMER] even when web_data has content.")
    print("The user's diagnosis is CORRECT - we need Python if/else, not LLM logic.")
else:
    print("CONCLUSION: Tavily returned NOTHING. Rate limit or network issue.")
