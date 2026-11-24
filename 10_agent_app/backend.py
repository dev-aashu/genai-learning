"""
Step 10: FastAPI backend that runs a LangGraph agent (local Ollama model)
whose tools come from our MCP server (07_mcp/server.py).

Streamlit → FastAPI → LangGraph agent → MCP server tools
"""
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
MCP_SERVER = ROOT / "07_mcp" / "server.py"

SYSTEM = SystemMessage(
    "You are a helpful assistant. Answer general knowledge questions directly. "
    "Use a tool ONLY when needed: calculator for math, get_current_time for the time, "
    "list_data_files / read_local_file for files in the data folder."
)

# One MCP client for the whole app. It starts 07_mcp/server.py over stdio when tools are used.
mcp_client = MultiServerMCPClient({
    "local_tools": {
        "command": sys.executable,
        "args": [str(MCP_SERVER)],
        "transport": "stdio",
    }
})
TOOLS = []  # filled at startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    """On startup: ask the MCP server which tools it has (tool discovery)."""
    global TOOLS
    TOOLS = await mcp_client.get_tools()
    print(f"✅ Loaded MCP tools: {[t.name for t in TOOLS]}")
    yield


app = FastAPI(title="Agent + MCP Backend", lifespan=lifespan)


def build_agent(model_name: str):
    """Same graph as Step 08, but tools come from MCP and we use the prebuilt ToolNode."""
    model = ChatOllama(model=model_name, temperature=0, reasoning=False).bind_tools(TOOLS)

    async def llm_call(state: MessagesState):
        response = await model.ainvoke([SYSTEM] + state["messages"])
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("llm_call", llm_call)
    builder.add_node("tools", ToolNode(TOOLS))           # prebuilt version of our tool_node
    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges("llm_call", tools_condition)  # prebuilt should_continue
    builder.add_edge("tools", "llm_call")
    return builder.compile()


def to_text(content) -> str:
    """Model/tool content can be a string or a list of parts; return plain text without thinking."""
    if isinstance(content, list):
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content).split("</think>")[-1].strip()


class ChatRequest(BaseModel):
    messages: list[dict]
    model: str = "qwen3:4b"


@app.get("/health")
def health():
    return {"status": "ok", "tools": [t.name for t in TOOLS]}


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    agent = build_agent(req.model)

    async def generate():
        try:
            # stream_mode="updates" gives us each graph step as it happens
            async for update in agent.astream(
                {"messages": req.messages}, {"recursion_limit": 10}, stream_mode="updates"
            ):
                for node, data in update.items():
                    for msg in data.get("messages", []):
                        if node == "llm_call" and msg.tool_calls:
                            for call in msg.tool_calls:
                                yield f"🔧 **{call['name']}** `{json.dumps(call['args'])}`\n\n"
                        elif node == "llm_call":
                            yield to_text(msg.content)
                        elif node == "tools":
                            yield f"✅ `{msg.name}` → {to_text(msg.content)[:300]}\n\n"
        except Exception as e:
            yield f"\n\n❌ {type(e).__name__}: {e}"

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")