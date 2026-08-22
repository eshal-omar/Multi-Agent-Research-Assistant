"""
Wraps the Tavily search API so agent nodes can pull live web results.
Tavily is used because it's built specifically for LLM/agent workflows.
"""

import os
from tavily import TavilyClient


def search_web(query: str, max_results: int = 4, max_chars_per_result: int = 800) -> list[dict]:
    """
    Runs a web search and returns a list of {url, content} dicts.
    Content is truncated per result to keep total prompt size well under
    Groq's free-tier tokens-per-minute limit.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Add it to your .env file. "
            "Get a free key at https://tavily.com"
        )

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",
    )

    results = []
    for r in response.get("results", []):
        content = r.get("content", "")
        if len(content) > max_chars_per_result:
            content = content[:max_chars_per_result] + "..."
        results.append({
            "url": r.get("url", ""),
            "content": content,
        })
    return results