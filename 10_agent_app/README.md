# 10: Agent app (Streamlit + FastAPI + LangGraph + MCP)

Streamlit → FastAPI → LangGraph agent (Ollama qwen3:4b) → MCP server (07_mcp/server.py)

The backend discovers tools from the MCP server at startup (langchain-mcp-adapters),
and streams each step (tool call → result → answer) to the UI.

## Run

    uvicorn backend:app --app-dir 10_agent_app --reload --port 8001
    streamlit run 10_agent_app/frontend.py --server.port 8502
