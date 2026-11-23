"""
Step 06: The same agent loop as Step 01, but with a small open-source model
running locally on my Mac via Ollama. No API key, no internet needed.
"""
import math
import time
from datetime import datetime
from pathlib import Path

from ollama import chat

MODEL = "qwen3:4b" # try also: "qwen3:1.7b"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# ---------- Tools (plain functions: Ollama reads type hints + docstrings) ----------
def calculator(expression: str) -> str:
    """Evaluate a math expression.

    Args:
        expression: A math expression like '23*47' or 'sqrt(16)'.
    """
    allowed = {"__builtins__": {}, "sqrt": math.sqrt, "pi": math.pi}
    try:
        return str(eval(expression, allowed))
    except Exception as e:
        return f"Error: {e}"


def get_current_time() -> str:
    """Get the current local date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_local_file(filename: str) -> str:
    """Read a text file from the project's data/ folder.

    Args:
        filename: Only the file name, e.g. 'notes.txt'.
    """
    path = (DATA_DIR / filename).resolve()
    if not path.is_relative_to(DATA_DIR) or not path.is_file():
        return f"Error: '{filename}' not found in data/."
    return path.read_text(encoding="utf-8", errors="replace")[:5000]


TOOLS = {f.__name__: f for f in [calculator, get_current_time, read_local_file]}


# ---------- The agent loop (same idea as Step 01) ----------
def run_agent(user_message: str, max_steps: int = 5) -> str:
    messages = [
        {"role": "system", "content": (
            "You are a helpful assistant. Answer general knowledge questions directly "
            "from your own knowledge. Use a tool ONLY when it is needed: calculator for "
            "math, get_current_time for the current time, read_local_file for files in data/. "
            "/no_think"
        )},
        {"role": "user", "content": user_message},
    ]
    for step in range(max_steps):
        response = chat(model=MODEL, messages=messages, tools=list(TOOLS.values()), think=False)
        messages.append(response.message)

        if not response.message.tool_calls:          # no tool -> final answer
            content = response.message.content or ""
            return content.split("</think>")[-1].strip()   # keep only the text after thinking

        for call in response.message.tool_calls:     # run each requested tool
            name, args = call.function.name, call.function.arguments
            print(f"  🔧 Tool call: {name}({args})")
            func = TOOLS.get(name)
            output = func(**args) if func else f"Error: unknown tool '{name}'"
            print(f"  ✅ Result: {output}")
            messages.append({"role": "tool", "tool_name": name, "content": str(output)})

    return "Stopped: too many steps."


if __name__ == "__main__":
    print(f"Local agent ready ({MODEL}). Type 'exit' to quit.\n")
    while True:
        q = input("You: ")
        if q.strip().lower() in ("exit", "quit"):
            break
        start = time.time()
        try:
            print("Agent:", run_agent(q))
        except Exception as e:
            print(f"Agent: ❌ {type(e).__name__}: {e}")
        print(f"  ⏱️ {time.time() - start:.1f}s\n")