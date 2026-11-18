"""Keyless MCP search and webpage reading for the original ODR graph."""

from mrdre_baseline.services.web import read_public_webpage, search_web

# MCP SDK 2.x moved HTTP binding options from FastMCP() to MCPServer.run().
try:
    from mcp.server.mcpserver import MCPServer
except ImportError:  # MCP SDK 1.x
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("Mr Dre Free Web Tools", host="127.0.0.1", port=8001)
    _mcp_v2 = False
else:
    mcp = MCPServer("Mr Dre Free Web Tools")
    _mcp_v2 = True


@mcp.tool()
def web_search(query: str, num_results: int = 5) -> list[dict]:
    """Search the public web for page titles, URLs and snippets."""
    return search_web(query, num_results=num_results)


@mcp.tool()
def read_webpage(url: str) -> dict:
    """Retrieve the readable content of a public webpage."""
    return read_public_webpage(url)


def main() -> None:
    if _mcp_v2:
        mcp.run(transport="streamable-http", host="127.0.0.1", port=8001)
    else:
        mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
