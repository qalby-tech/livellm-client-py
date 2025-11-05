# models for tools
from pydantic import BaseModel, Field, field_validator
from typing import Literal
from enum import Enum

class ToolKind(str, Enum):
    WEB_SEARCH = "web_search"
    MCP_STREAMABLE_SERVER = "mcp_streamable_server"

class Tool(BaseModel):
    kind: ToolKind
    input: BaseModel


class ToolInput(BaseModel):
    kind: ToolKind
    kwargs: dict = Field(default_factory=dict, description="Additional keyword arguments for the MCP server")

class WebSearchInput(ToolInput):
    kind: ToolKind = Field(ToolKind.WEB_SEARCH, description="Web search kind of tool")
    search_context_size: Literal['low', 'medium', 'high'] = Field('medium', description="The context size for the search")

    @field_validator('kind')
    def validate_kind(cls, v):
        if v != ToolKind.WEB_SEARCH:
            raise ValueError(f"Invalid kind: {v}")
        return v

class MCPStreamableServerInput(ToolInput):
    kind: ToolKind = Field(ToolKind.MCP_STREAMABLE_SERVER, description="Mcp kind of tool")
    url: str = Field(..., description="The URL of the MCP server")
    prefix: str = Field(..., description="The prefix of the MCP server")
    timeout: int = Field(15, description="The timeout in seconds for the MCP server")

    @field_validator('kind')
    def validate_kind(cls, v):
        if v != ToolKind.MCP_STREAMABLE_SERVER:
            raise ValueError(f"Invalid kind: {v}")
        return v