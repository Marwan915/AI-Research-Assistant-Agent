import arxiv
import time
from langchain_core.tools import tool

@tool
def search_arxiv(query: str, max_results: int = 2) -> str:
    """
    Search the ArXiv scientific database for papers.
    Use this tool to find academic papers, their authors, abstracts, and PDF links.
    Always provide a specific search query.
    """
    try:
        # تأخير زمني متعمد لثانيتين قبل إرسال أي طلب لتجنب الحظر (HTTP 429)
        time.sleep(2)
        
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        for paper in client.results(search):
            paper_info = (
                f"Title: {paper.title}\n"
                f"Authors: {', '.join([author.name for author in paper.authors])}\n"
                f"Published: {paper.published.date()}\n"
                f"PDF Link: {paper.pdf_url}\n"
                f"Abstract: {paper.summary}\n"
                f"{'-'*40}"
            )
            results.append(paper_info)
            
            # تأخير إضافي بسيط بين قراءة كل نتيجة
            time.sleep(1)
            
        if not results:
            return f"No ArXiv papers found for the query: '{query}'."
            
        return "\n".join(results)
        
    except Exception as e:
        # إذا حصل خطأ (مثل 429)، نطلب من الوكيل التوقف عن المحاولة لهذه الكلمة
        return f"ArXiv API Error (Rate Limit or Network): {str(e)}. Please stop trying to search ArXiv for now and rely on local data."

if __name__ == "__main__":
    print("Testing ArXiv Tool with Rate Limit Protection...")
    print(search_arxiv.invoke({"query": "LLM Optimization", "max_results": 1}))