"""
Step 04: The same calculator + time agent, built with Google ADK.
Tools are plain Python functions; ADK reads the name, docstring and type hints.
"""
import math
from datetime import datetime

from google.adk.agents import Agent


def calculator(expression: str) -> dict:
    """Evaluate a math expression, e.g. '23*47' or 'sqrt(16)'.

    Args:
        expression: The math expression to evaluate.

    Returns:
        A dict with 'status' and either 'result' or 'error_message'.
    """
    allowed = {"__builtins__": {}, "sqrt": math.sqrt, "pi": math.pi}
    try:
        return {"status": "success", "result": str(eval(expression, allowed))}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


def get_current_time() -> dict:
    """Get the current local date and time.

    Returns:
        A dict with 'status' and 'time'.
    """
    return {"status": "success", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


root_agent = Agent(
    name="calc_agent",
    model="gemini-3.5-flash-lite",
    description="An agent that does math and tells the current time.",
    instruction=(
        "You are a helpful assistant. Use the calculator tool for any math "
        "and the get_current_time tool for the current date or time."
    ),
    tools=[calculator, get_current_time],
)