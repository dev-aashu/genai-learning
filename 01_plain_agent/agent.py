"""
Plain-Python agent (no framework).
Two tools: calculator + get_current_time.
The loop: send message -> model asks for tool -> run tool -> send result back -> repeat until final answer.
"""
import math
import time
from datetime import datetime

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()
client = genai.Client(http_options=types.HttpOptions(timeout=90_000))  # 90 seconds
MODELS = ["gemini-3.5-flash-lite", "gemini-3.7-flash", "gemini-3.8-flash"]


# ---------- 1. Tools (plain Python functions) ----------
def calculator(expression: str) -> str:
    """Evaluate a math expression like '23*47' or 'sqrt(16)'."""
    allowed = {"__builtins__": {}, "sqrt": math.sqrt, "pi": math.pi}
    try:
        return str(eval(expression, allowed))
    except Exception as e:
        return f"Error: {e}"


def get_current_time() -> str:
    """Return the current local date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOLS = {"calculator": calculator, "get_current_time": get_current_time}


# ---------- 2. Tell the model what tools exist ----------
tool_schema = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="calculator",
        description="Evaluate a math expression, e.g. '23*47' or 'sqrt(16)'.",
        parameters=types.Schema(
            type="OBJECT",
            properties={"expression": types.Schema(type="STRING")},
            required=["expression"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_current_time",
        description="Get the current local date and time.",
    ),
])
config = types.GenerateContentConfig(tools=[tool_schema])


# ---------- 3. Call the model (with retry for busy server) ----------
def call_model(messages, retries: int = 2):
    """Try each model in MODELS; retry if busy (503) or rate-limited (429)."""
    for model in MODELS:
        for attempt in range(1, retries + 1):
            try:
                return client.models.generate_content(
                    model=model, contents=messages, config=config
                )
            except errors.APIError as e:
                if e.code not in (429, 503, 504):
                    raise
                print(f"  ⏳ {model} busy ({e.code}), attempt {attempt}/{retries}")
                time.sleep(3 * attempt)
        print(f"  ↪️ Switching from {model} to next model...")
    raise RuntimeError("All models are busy. Try again later.")


# ---------- 4. The agent loop ----------
def run_agent(user_message: str, max_steps: int = 5) -> str:
    messages = [types.Content(role="user", parts=[types.Part(text=user_message)])]

    for step in range(max_steps):
        response = call_model(messages)
        messages.append(response.candidates[0].content)  # keep model's turn in history

        calls = response.function_calls
        if not calls:                      # no tool needed -> final answer
            return response.text

        results = []
        for call in calls:                 # model wants tool(s)
            args = dict(call.args or {})
            print(f"  🔧 Tool call: {call.name}({args})")
            output = TOOLS[call.name](**args)
            print(f"  ✅ Result: {output}")
            results.append(types.Part.from_function_response(
                name=call.name, response={"result": output}
            ))
        messages.append(types.Content(role="user", parts=results))

    return "Stopped: too many steps."


# ---------- 5. Simple chat in terminal ----------
if __name__ == "__main__":
    print("Plain agent ready. Type 'exit' to quit.\n")
    while True:
        q = input("You: ")
        if q.strip().lower() in ("exit", "quit"):
            break
        try:
            print("Agent:", run_agent(q), "\n")
        except Exception as e:
            print(f"Agent: ❌ {type(e).__name__}: {e}\n")