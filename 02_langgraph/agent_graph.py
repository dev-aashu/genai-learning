"""
Step 02: The same calculator + time agent, rebuilt as a LangGraph graph.
Graph: START -> llm_call -> (tool call?) -> tool_node -> llm_call ... -> END
"""
import math
import os
from datetime import datetime
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, MessagesState, StateGraph

load_dotenv()


# ---------- 1. Tools (same as Step 01, now with @tool) ----------
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


tools = [calculator, get_current_time]
tools_by_name = {t.name: t for t in tools}


# ---------- 2. Model (with timeout, retries and fallback models) ----------
def make_model(name: str):
    return ChatGoogleGenerativeAI(
        model=name,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        timeout=90,
        max_retries=2,
    ).bind_tools(tools)


model = make_model("gemini-3.5-flash-lite").with_fallbacks(
    [make_model("gemini-3.7-flash"), make_model("gemini-3.8-flash")]
)


# ---------- 3. Nodes ----------
def llm_call(state: MessagesState):
    """Node: ask the model what to do next."""
    system = SystemMessage("You are a helpful assistant. Use tools for math and for the current time.")
    response = model.invoke([system] + state["messages"])
    return {"messages": [response]}


def tool_node(state: MessagesState):
    """Node: run every tool the model asked for."""
    results = []
    for call in state["messages"][-1].tool_calls:
        print(f"  🔧 Tool call: {call['name']}({call['args']})")
        output = tools_by_name[call["name"]].invoke(call["args"])
        print(f"  ✅ Result: {output}")
        results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
    return {"messages": results}


# ---------- 4. Conditional edge ----------
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """If the model asked for a tool, go to tool_node; otherwise stop."""
    if state["messages"][-1].tool_calls:
        return "tool_node"
    return END


# ---------- 5. Build and compile the graph ----------
builder = StateGraph(MessagesState)
builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", tool_node)
builder.add_edge(START, "llm_call")
builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
builder.add_edge("tool_node", "llm_call")
agent = builder.compile()


# ---------- 6. Helpers ----------
def text_of(message) -> str:
    """Gemini may return content as a list of parts; join the text parts."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


def save_graph_diagram():
    """Save the graph as a Mermaid diagram that GitHub can render."""
    fence = "`" * 3
    path = os.path.join(os.path.dirname(__file__), "graph.md")
    with open(path, "w") as f:
        f.write(f"# Agent graph\n\n{fence}mermaid\n{agent.get_graph().draw_mermaid()}\n{fence}\n")
    print(f"Graph diagram saved to {path}\n")


# ---------- 7. Simple chat in terminal ----------
if __name__ == "__main__":
    save_graph_diagram()
    print("LangGraph agent ready. Type 'exit' to quit.\n")
    while True:
        q = input("You: ")
        if q.strip().lower() in ("exit", "quit"):
            break
        try:
            result = agent.invoke(
                {"messages": [HumanMessage(q)]},
                {"recursion_limit": 10},  # like max_steps in Step 01
            )
            print("Agent:", text_of(result["messages"][-1]), "\n")
        except Exception as e:
            print(f"Agent: ❌ {type(e).__name__}: {e}\n")