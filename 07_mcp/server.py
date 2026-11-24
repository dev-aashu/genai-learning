"""
Step 07: MCP server exposing our 3 tools.
Any MCP client (our agent, Claude Desktop, Cursor...) can discover and call these tools.
"""
import math
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("genai-learning-tools")
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@mcp.tool()
def calculator(expression: str) -> str:
    """Evaluate a math expression, e.g. '23*47' or 'sqrt(16)'."""
    allowed = {"__builtins__": {}, "sqrt": math.sqrt, "pi": math.pi}
    try:
        return str(eval(expression, allowed))
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def get_current_time() -> str:
    """Get the current local date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@mcp.tool()
def read_local_file(filename: str) -> str:
    """Read a text file from the project's data/ folder. Pass only the file name, e.g. 'notes.txt'."""
    path = (DATA_DIR / filename).resolve()
    if not path.is_relative_to(DATA_DIR) or not path.is_file():
        available = ", ".join(p.name for p in DATA_DIR.glob("*") if p.is_file())
        return f"Error: '{filename}' not found. Available files: {available}"
    return path.read_text(encoding="utf-8", errors="replace")[:5000]

@mcp.tool()
def list_data_files() -> str:
    """List all files available in the project's data/ folder."""
    files = [p.name for p in DATA_DIR.glob("*") if p.is_file()]
    return ", ".join(files) if files else "No files found."

if __name__ == "__main__":
    mcp.run()  # stdio transport: talks over stdin/stdout