"""Keyless MCP search + page reading for the ORIGINAL LangChain ODR graph.

Run: python -m local.free_search_mcp
Connect to: http://127.0.0.1:8001/mcp
"""

from mcp.server.fastmcp import FastMCP
from local.free_web import read_public_webpage, search_web

mcp = FastMCP("Mr Dre Free Web Tools", host="127.0.0.1", port=8001)


@mcp.tool()
def web_search(query: str, num_results: int = 5) -> list[dict]:
    """Search the public web and return page titles, URLs, and snippets.

    Use read_webpage to read the content of promising sources before citing.
    """
    return search_web(query, num_results=num_results)


@mcp.tool()
def read_webpage(url: str) -> dict:
    """Retrieve the actual text of a public web page, for evidence and citations."""
    return read_public_webpage(url)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
