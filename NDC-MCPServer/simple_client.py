"""
Simple MCP client that spawns `example_mcp_server.py`, lists its tools,
and calls a couple of them.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import anyio

import mcp.types as types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


PROJECT_ROOT = Path(__file__).resolve().parent
SERVER_PATH = PROJECT_ROOT / "example_mcp_server.py"


async def run_client() -> None:
    if not SERVER_PATH.exists():
        raise FileNotFoundError(f"Unable to locate MCP server script at {SERVER_PATH}")

    server = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
        cwd=str(PROJECT_ROOT),
    )

    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            print("Initializing MCP session...")
            await session.initialize()
            print("Session ready.")

            tools = await session.list_tools()
            tool_names = [tool.name for tool in tools.tools]
            print(f"Server exposes tools: {tool_names}")

            if "greet" in tool_names:
                greet_result = await session.call_tool("greet", {"name": "Cursor"})
                print("greet('Cursor') result:")
                _print_tool_result(greet_result)
            else:
                print("Tool 'greet' not found on server.")

            if "add_numbers" in tool_names:
                add_result = await session.call_tool("add_numbers", {"a": 50, "b": 90})
                print("add_numbers(50, 90) result:")
                _print_tool_result(add_result)
            else:
                print("Tool 'add_numbers' not found on server.")


def _print_tool_result(result: types.CallToolResult) -> None:
    if result.isError:
        print("  Tool returned an error:", result.content)
        return

    if result.content:
        for item in result.content:
            if isinstance(item, types.TextContent):
                print(" ", item.text)
            else:
                print(" ", item)

    if result.structuredContent is not None:
        print("  Structured content:", result.structuredContent)


def main() -> None:
    try:
        anyio.run(run_client)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

