"""
Step 07: Agent as an MCP client.
Same Ollama loop as Step 06, but tools come from the MCP server (server.py),
not from this file. The agent discovers tools at startup and calls them over MCP.
"""
import asyncio
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import chat

MODEL = "qwen3:4b"
SERVER = Path(__file__).resolve().parent / "server.py"
SYSTEM = (
    "You are a helpful assistant. Answer general knowledge questions directly. "
    "Use a tool ONLY when needed: calculator for math, get_current_time for the time, "
    "read_local_file for files in data/. /no_think"
)


def to_ollama_tools(mcp_tools):
    """Convert MCP tool definitions into the format Ollama expects."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema,
            },
        }
        for t in mcp_tools
    ]


async def run_agent(session: ClientSession, tools: list, user_message: str, max_steps: int = 5) -> str:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message}]

    for step in range(max_steps):
        response = chat(model=MODEL, messages=messages, tools=tools, think=False)
        messages.append(response.message)

        if not response.message.tool_calls:
            content = response.message.content or ""
            return content.split("</think>")[-1].strip()

        for call in response.message.tool_calls:
            name, args = call.function.name, call.function.arguments
            print(f"  🔧 MCP tool call: {name}({args})")
            result = await session.call_tool(name, args)          # runs on the MCP server
            output = "".join(c.text for c in result.content if hasattr(c, "text"))
            print(f"  ✅ Result: {output}")
            messages.append({"role": "tool", "tool_name": name, "content": output})

    return "Stopped: too many steps."


async def main():
    # Start server.py as a subprocess and talk to it over stdio
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools = (await session.list_tools()).tools        # tool discovery
            print(f"Connected to MCP server. Tools: {[t.name for t in mcp_tools]}")
            tools = to_ollama_tools(mcp_tools)
            print(f"MCP agent ready ({MODEL}). Type 'exit' to quit.\n")

            while True:
                q = input("You: ")
                if q.strip().lower() in ("exit", "quit"):
                    break
                start = time.time()
                try:
                    print("Agent:", await run_agent(session, tools, q))
                except Exception as e:
                    print(f"Agent: ❌ {type(e).__name__}: {e}")
                print(f"  ⏱️ {time.time() - start:.1f}s\n")


if __name__ == "__main__":
    asyncio.run(main())