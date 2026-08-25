import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from backend.graph import build_graph, initial_state

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon=" ",
    layout="wide",
)

st.title("Multi-Agent Research Assistant")
st.caption("Search agent → Summarizer agent → Critic agent → Writer agent (built with LangGraph)")

# --- Sanity check for API keys before doing anything else ------------------
missing_keys = []
if not os.environ.get("GROQ_API_KEY"):
    missing_keys.append("GROQ_API_KEY")
if not os.environ.get("TAVILY_API_KEY"):
    missing_keys.append("TAVILY_API_KEY")

if missing_keys:
    st.error(
        f"Missing environment variable(s): {', '.join(missing_keys)}. "
        "Copy `.env.example` to `.env`, fill in your keys, and restart the app."
    )
    st.stop()

# --- Cache the compiled graph across reruns ---------------------------------
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

# --- Sidebar: how it works ---------------------------------------------------
with st.sidebar:
    st.header("How it works")
    st.markdown(
        """
        1. **Search agent** — pulls live web results (Tavily)
        2. **Summarizer agent** — condenses results into a draft
        3. **Critic agent** — checks for gaps; may send it back
           to search for a follow-up round (max 3 rounds)
        4. **Writer agent** — produces the final structured report
        """
    )
    st.divider()
    # st.caption("Model: " + os.environ.get("MODEL_NAME", "claude-sonnet-4-6"))

# --- Main input ---------------------------------------------------------
topic = st.text_input(
    "Enter a research topic or question:",
    placeholder="e.g. What are the environmental tradeoffs of lithium vs sodium-ion batteries?",
)
run_button = st.button("Run Research", type="primary", disabled=not topic)

if run_button and topic:
    progress_box = st.status("Agents are working...", expanded=True)
    running_state = dict(initial_state(topic))
    final_state = None

    try:
        with progress_box:
            for step_output in st.session_state.graph.stream(running_state):
                node_name, update = list(step_output.items())[0]
                running_state.update(update)

                if node_name == "search":
                    st.write(f"**Search agent** — round {running_state['iteration']} "
                             f"({len(running_state['search_results'])} sources so far)")
                elif node_name == "summarize":
                    st.write("**Summarizer agent** drafted a summary")
                elif node_name == "critique":
                    if running_state["needs_more_research"]:
                        st.write(f"**Critic agent** found a gap → follow-up search: "
                                 f"*{running_state['critique']}*")
                    else:
                        st.write("**Critic agent** approved the summary")
                elif node_name == "write":
                    st.write("**Writer agent** finalized the report")

                final_state = running_state

        progress_box.update(label="Research complete", state="complete")

    except Exception as e:
        progress_box.update(label="Something went wrong", state="error")
        st.exception(e)
        st.stop()

    st.divider()
    st.subheader("Final Report")
    st.markdown(final_state["report"])

    with st.expander(f"Sources used ({len(final_state['search_results'])} total)"):
        seen = set()
        for r in final_state["search_results"]:
            if r["url"] not in seen:
                st.markdown(f"- [{r['url']}]({r['url']})")
                seen.add(r["url"])

    st.download_button(
        "Download report as Markdown",
        data=final_state["report"],
        file_name="research_report.md",
        mime="text/markdown",
    )