"""
Wraps the Tavily search API so agent nodes can pull live web results.
Tavily is used because it's built specifically for LLM/agent workflows.
"""

import os
from tavily import TavilyClient


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Runs a web search and returns a list of {url, content} dicts.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Add it to your .env file. "
        )

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",  # pulls fuller content per result
    )

    results = []
    for r in response.get("results", []):
        results.append({
            "url": r.get("url", ""),
            "content": r.get("content", ""),
        })
    return results