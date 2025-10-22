"""Tool-related models for LiveLLM Proxy Client."""

from typing import Literal
from pydantic import BaseModel, Field


class ToolKind:
    """Type of tool."""
    WEB_SEARCH: str = "web_search"
    MCP_STREAMABLE_SERVER: str = "mcp_streamable_server"


class WebSearchInput(BaseModel):
    """Web search tool configuration."""
    search_context_size: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="The context size for the search"
    )


class MCPStreamableServerInput(BaseModel):
    """MCP server tool configuration."""
    url: str = Field(description="The URL of the MCP server")
    prefix: str = Field(description="The prefix of the MCP server")

