"""
Step 08: Same LangGraph agent as Step 02, but with a local open-source model (Ollama)
instead of Gemini. Only the model setup changes; the graph stays exactly the same.
"""
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph

MODEL = "qwen3:4b"   # try also: "llama3.2:3b"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# ---------- 1. Tools (same as before) ----------
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression, e.g. '23*47' or 'sqrt(16)'."""
    allowed = {"__builtins__": {}, "sqrt": math.sqrt, "pi": math.pi}
    try:
        return str(eval(expression, allowed))
    except Exception as e:
        return f"Error: {e}"


@tool
def get_current_time() -> str:
    """Get the current local date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def read_local_file(filename: str) -> str:
    """Read a text file from the project's data/ folder. Pass only the file name, e.g. 'notes.txt'."""
    path = (DATA_DIR / filename).resolve()
    if not path.is_relative_to(DATA_DIR) or not path.is_file():
        return f"Error: '{filename}' not found in data/."
    return path.read_text(encoding="utf-8", errors="replace")[:5000]


tools = [calculator, get_current_time, read_local_file]
tools_by_name = {t.name: t for t in tools}


# ---------- 2. Model: the ONLY real change from Step 02 ----------
model = ChatOllama(model="qwen3:4b", temperature=0, reasoning=False).bind_tools(tools)

SYSTEM = SystemMessage(
    "You are a helpful assistant. Answer general knowledge questions directly. "
    "Use a tool ONLY when needed: calculator for math, get_current_time for the time, "
    "read_local_file for files in data/."
)


# ---------- 3. Nodes (same as Step 02) ----------
def llm_call(state: MessagesState):
    response = model.invoke([SYSTEM] + state["messages"])
    return {"messages": [response]}


def tool_node(state: MessagesState):
    results = []
    for call in state["messages"][-1].tool_calls:
        print(f"  🔧 Tool call: {call['name']}({call['args']})")
        output = tools_by_name[call["name"]].invoke(call["args"])
        print(f"  ✅ Result: {output}")
        results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
    return {"messages": results}


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    return "tool_node" if state["messages"][-1].tool_calls else END


# ---------- 4. Graph (same as Step 02) ----------
builder = StateGraph(MessagesState)
builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", tool_node)
builder.add_edge(START, "llm_call")
builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
builder.add_edge("tool_node", "llm_call")
agent = builder.compile()


def clean(text) -> str:
    """Remove any thinking text and return only the final answer."""
    if not isinstance(text, str):
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    return text.split("</think>")[-1].strip()


if __name__ == "__main__":
    print(f"LangGraph + Ollama agent ready ({MODEL}). Type 'exit' to quit.\n")
    while True:
        q = input("You: ")
        if q.strip().lower() in ("exit", "quit"):
            break
        start = time.time()
        try:
            result = agent.invoke({"messages": [HumanMessage(q)]}, {"recursion_limit": 10})
            print("Agent:", clean(result["messages"][-1].content))
        except Exception as e:
            print(f"Agent: ❌ {type(e).__name__}: {e}")
        print(f"  ⏱️ {time.time() - start:.1f}s\n")