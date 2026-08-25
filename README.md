# Multi-Agent Research Assistant
An AI powered research tool that automates the tedious part of research: searching, reading, cross-checking and writing the findings. Give it a topic and 4 collaborating AI agents will produce a report with citations.


## How it works
- Search agent: Pulls live web results for the topic via Tavily 
- Summarizer agent: Condenses raw search results into a clean, cited draft 
- Critic agent: Fact-checks the draft and decides: is this complete, or does it need another research pass? 
- Writer agent: Produces the final structured report (Overview, Key Findings, Caveats, Sources) 

In the Critic -> Search loop: if the critic finds a gap, it generates a targeted follow-up query and sends the whole pipeline back to search for another round (capped at 3 rounds to avoid infinite loops).

## Features 
- Live multi-agent progress shown in real time in the UI
- Self-correcting research loop: the critic agent can trigger follow-up searches
- Source-cited reports with clickable links
- Export reports as markdown

## Tech stack 
- Orchestration: LangGraph
- Frontend: Streamlit
- Programming Language: Python

## Setup
### Prerequisites
- Python 3.10+
- A free Groq API key(https://console.groq.com) 
- A free Tavily API key(https://tavily.com) 

### Setup

```bash
git clone https://github.com/eshal-omar/Multi-Agent-Research-Assistant.git
cd research-assistant

python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# open .env and add your GROQ_API_KEY and TAVILY_API_KEY
```

### Run

```bash
streamlit run app.py
```

Open `http://localhost:8501`, enter a research topic, and watch the agents work.


## Note
Be wary of staying in limits to avoid being charged when using the API keys.

