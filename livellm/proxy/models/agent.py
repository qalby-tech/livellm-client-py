"""Agent-related models for LiveLLM Proxy Client."""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from .message import TextMessage, BinaryMessage
from .tool import WebSearchInput, MCPStreamableServerInput


class AgentRequest(BaseModel):
    """Request to run an agent."""
    model: str = Field(description="The model to use")
    messages: List[Union[TextMessage, BinaryMessage]] = Field(description="Conversation messages")
    tools: List[Union[WebSearchInput, MCPStreamableServerInput]] = Field(description="Tools available to the agent")
    gen_config: Optional[Dict[str, Any]] = Field(
        None,
        description="The configuration for the generation"
    )


class AgentResponseUsage(BaseModel):
    """Token usage information."""
    input_tokens: int = Field(description="The number of input tokens used")
    output_tokens: int = Field(description="The number of output tokens used")


class AgentResponse(BaseModel):
    """Response from an agent run."""
    output: str = Field(description="The output of the response")
    usage: AgentResponseUsage = Field(description="The usage of the response")

