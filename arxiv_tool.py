"""
arxiv_tool.py — ArXiv Search with Rate-Limit Protection

Built-in delays prevent HTTP 429 errors from ArXiv servers.
Returns structured paper info for the researcher node.
"""

import arxiv
import time
from langchain_core.tools import tool


@tool
def search_arxiv(query: str, max_results: int = 2) -> str:
    """
    Search the ArXiv scientific database for papers.
    Use this tool to find recent academic papers, their authors, abstracts, and PDF links.
    Always provide a specific, focused search query.
    """
    try:
        # Mandatory delay before any request — prevents rate limiting
        time.sleep(2)

        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        results = []
        for paper in client.results(search):
            authors = ", ".join(a.name for a in paper.authors[:4])
            if len(paper.authors) > 4:
                authors += " et al."

            paper_info = (
                f"Title: {paper.title}\n"
                f"Authors: {authors}\n"
                f"Published: {paper.published.date()}\n"
                f"ArXiv ID: {paper.entry_id}\n"
                f"PDF: {paper.pdf_url}\n"
                f"Abstract: {paper.summary[:500]}{'…' if len(paper.summary) > 500 else ''}\n"
                f"{'─' * 40}"
            )
            results.append(paper_info)
            time.sleep(1)  # Small delay between results

        if not results:
            return f"No ArXiv papers found for: '{query}'."

        return f"[ArXiv Search: '{query}']\n\n" + "\n".join(results)

    except Exception as e:
        err = str(e)
        if "429" in err or "too many" in err.lower():
            return (
                f"ArXiv rate limit reached for '{query}'. "
                f"Stop ArXiv searches and rely on local database only."
            )
        return f"ArXiv API error for '{query}': {err}"


if __name__ == "__main__":
    print("Testing ArXiv Tool…")
    result = search_arxiv.invoke({"query": "transformer attention mechanism", "max_results": 1})
    print(result)