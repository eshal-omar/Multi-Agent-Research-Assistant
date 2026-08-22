"""
Defines the multi-agent research pipeline using LangGraph.

Pipeline (4 agents / nodes):

  1. search    -> pulls web results via Tavily for the topic (or a
                  follow-up query if the critic found a gap)
  2. summarize -> condenses raw search results into a clean summary
  3. critique  -> fact-checks / gap-checks the summary; decides whether
                  to loop back to "search" for more info or move on
  4. write     -> produces the final structured report

The critique -> search loop can run up to MAX_ITERATIONS times before
the graph forces a move to the final report, so it can't loop forever.
"""

import os
from typing import TypedDict, List

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

from backend.tools import search_web

MAX_ITERATIONS = 3


# ---------------------------------------------------------------------------
# Shared state that flows through every node in the graph
# ---------------------------------------------------------------------------
class ResearchState(TypedDict):
    topic: str
    search_results: List[dict]
    summary: str
    critique: str
    needs_more_research: bool
    iteration: int
    report: str


# ---------------------------------------------------------------------------
# LLM used by every agent (swap model name / provider here if needed)
# ---------------------------------------------------------------------------
def get_llm():
    return ChatGroq(
        model=os.environ.get("MODEL_NAME", "openai/gpt-oss-120b"),
        temperature=0.3,
    )


# ---------------------------------------------------------------------------
# Node 1: Search agent
# ---------------------------------------------------------------------------
def search_node(state: ResearchState) -> dict:
    # First pass searches the raw topic. Later passes search whatever
    # follow-up query the critic agent generated.
    query = state["topic"] if state["iteration"] == 0 else state["critique"]

    new_results = search_web(query, max_results=4)
    combined_results = state.get("search_results", []) + new_results

    return {
        "search_results": combined_results,
        "iteration": state["iteration"] + 1,
    }


# ---------------------------------------------------------------------------
# Node 2: Summarizer agent
# ---------------------------------------------------------------------------
def summarize_node(state: ResearchState) -> dict:
    llm = get_llm()

    sources_text = "\n\n".join(
        f"Source: {r['url']}\nContent: {r['content']}"
        for r in state["search_results"]
    )
    max_sources_chars = 6000
    if len(sources_text) > max_sources_chars:
        sources_text = sources_text[:max_sources_chars] + "\n\n[truncated for length]"
    prompt = f"""You are a research summarizer agent.

Topic: "{state['topic']}"

Using ONLY the search results below, write a clear, well-organized summary
covering the key facts, figures, and different perspectives. Reference
source URLs inline in parentheses where relevant. Do not invent facts that
aren't in the search results.

Search results:
{sources_text}
"""
    response = llm.invoke(prompt)
    return {"summary": response.content}


# ---------------------------------------------------------------------------
# Node 3: Critic / fact-checker agent
# ---------------------------------------------------------------------------
def critique_node(state: ResearchState) -> dict:
    llm = get_llm()

    prompt = f"""You are a critical fact-checking agent reviewing a research
summary before it gets published.

Topic: "{state['topic']}"

Summary to review:
{state['summary']}

Check for: missing perspectives, unsupported claims, outdated info, or
important sub-questions that weren't covered.

If the summary is solid and well-supported, respond with EXACTLY the
single word: COMPLETE

Otherwise, respond with ONE specific, well-formed follow-up search query
(not a sentence explaining what's missing, just the query itself) that
would fill the single biggest gap.
"""
    response = llm.invoke(prompt)
    critique_text = response.content.strip()

    needs_more = (
        critique_text != "COMPLETE"
        and state["iteration"] < MAX_ITERATIONS
    )

    return {
        "critique": critique_text,
        "needs_more_research": needs_more,
    }


# ---------------------------------------------------------------------------
# Node 4: Writer / synthesizer agent
# ---------------------------------------------------------------------------
def write_node(state: ResearchState) -> dict:
    llm = get_llm()

    all_urls = sorted({r["url"] for r in state["search_results"]})
    sources_list = "\n".join(f"- {u}" for u in all_urls)

    critique_note = (
        state["critique"] if state["critique"] != "COMPLETE" else "None — summary was comprehensive."
    )

    prompt = f"""You are the writer agent. Produce a final, polished research
report on "{state['topic']}".

Use this structure:
# {state['topic']}
## Overview
## Key Findings (use bullet points)
## Open Questions / Caveats
## Sources

Base the report on this summary and the fact-checker's final note.
Do not fabricate sources — only use the ones listed below.

Summary:
{state['summary']}

Fact-checker's final note:
{critique_note}

Sources to list at the end:
{sources_list}
"""
    response = llm.invoke(prompt)
    return {"report": response.content}


# ---------------------------------------------------------------------------
# Routing logic: after critique, either loop back to search or move to write
# ---------------------------------------------------------------------------
def route_after_critique(state: ResearchState) -> str:
    return "search" if state["needs_more_research"] else "write"


# ---------------------------------------------------------------------------
# Build & compile the graph
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("search", search_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("critique", critique_node)
    graph.add_node("write", write_node)

    graph.set_entry_point("search")
    graph.add_edge("search", "summarize")
    graph.add_edge("summarize", "critique")
    graph.add_conditional_edges(
        "critique",
        route_after_critique,
        {"search": "search", "write": "write"},
    )
    graph.add_edge("write", END)

    return graph.compile()


def initial_state(topic: str) -> ResearchState:
    return {
        "topic": topic,
        "search_results": [],
        "summary": "",
        "critique": "",
        "needs_more_research": True,
        "iteration": 0,
        "report": "",
    }