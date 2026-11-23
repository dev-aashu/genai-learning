"""
Step 03: LangGraph agent with
  1. Memory  -> a checkpointer saves the conversation per thread_id
  2. Human-in-the-loop -> interrupt() asks for approval before any tool runs
"""
import math
import os
import uuid
from datetime import datetime
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt

load_dotenv()


# ---------- 1. Tools (same as Step 02) ----------
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


# ---------- 2. Model (same as Step 02) ----------
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
    system = SystemMessage("You are a helpful assistant. Use tools for math and for the current time.")
    response = model.invoke([system] + state["messages"])
    return {"messages": [response]}


def tool_node(state: MessagesState):
    """Ask the human for approval, then run (or skip) the requested tools."""
    tool_calls = state["messages"][-1].tool_calls

    # NEW: pause the graph here and wait for the human's answer
    decision = interrupt({
        "question": "Approve these tool calls?",
        "tool_calls": [{"name": c["name"], "args": c["args"]} for c in tool_calls],
    })

    results = []
    for call in tool_calls:
        if decision == "yes":
            output = tools_by_name[call["name"]].invoke(call["args"])
            print(f"  ✅ {call['name']} -> {output}")
        else:
            output = "The user denied permission to run this tool."
            print(f"  🚫 {call['name']} skipped")
        results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
    return {"messages": results}


# ---------- 4. Conditional edge (same as Step 02) ----------
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    if state["messages"][-1].tool_calls:
        return "tool_node"
    return END


# ---------- 5. Build graph WITH a checkpointer ----------
builder = StateGraph(MessagesState)
builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", tool_node)
builder.add_edge(START, "llm_call")
builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
builder.add_edge("tool_node", "llm_call")

checkpointer = InMemorySaver()                      # NEW: memory
agent = builder.compile(checkpointer=checkpointer)  # NEW: attach memory


# ---------- 6. Helpers ----------
def text_of(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


def run_turn(user_input: str, config: dict) -> str:
    """Run one user turn, handling any approval pauses."""
    result = agent.invoke({"messages": [HumanMessage(user_input)]}, config)

    # While the graph is paused, ask the human and resume
    while result.get("__interrupt__"):
        request = result["__interrupt__"][0].value
        print("  ⏸️  Agent wants to run:")
        for c in request["tool_calls"]:
            print(f"      🔧 {c['name']}({c['args']})")
        answer = input("  Approve? (yes/no): ").strip().lower()
        result = agent.invoke(Command(resume=answer), config)

    return text_of(result["messages"][-1])


# ---------- 7. Chat ----------
if __name__ == "__main__":
    thread_id = str(uuid.uuid4())[:8]
    print("Agent with memory + approval ready.")
    print("Commands: 'new' = start a fresh conversation, 'exit' = quit\n")
    print(f"🧵 Thread: {thread_id}\n")

    while True:
        q = input("You: ")
        cmd = q.strip().lower()
        if cmd in ("exit", "quit"):
            break
        if cmd == "new":
            thread_id = str(uuid.uuid4())[:8]
            print(f"\n🧵 New thread: {thread_id} (memory reset)\n")
            continue

        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 10}
        try:
            print("Agent:", run_turn(q, config), "\n")
        except Exception as e:
            print(f"Agent: ❌ {type(e).__name__}: {e}\n")