# GenAI & Agentic AI Learning

Hands-on learning path for building LLM agents: from a plain Python loop to LangGraph and Google ADK.

## Roadmap

| Step | Topic                                | Status |
| ---- | ------------------------------------ | ------ |
| 00   | Setup (Python venv, Gemini API)      | ✅     |
| 01   | Plain-Python agent (no framework)    | ⏳     |
| 02   | LangGraph basics                     | ⏳     |
| 03   | LangGraph memory + human-in-the-loop | ⏳     |
| 04   | Google ADK basics                    | ⏳     |
| 05   | LangGraph vs ADK comparison          | ⏳     |

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your Gemini API key
python 00_setup/test_key.py
```

## Tech

Python 3.13 · Gemini API · LangGraph · Google ADK

## Learnings

See [NOTES.md](NOTES.md) and each folder's README.
