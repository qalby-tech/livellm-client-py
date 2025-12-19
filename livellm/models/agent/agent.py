# models for full run: AgentRequest, AgentResponse

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Union
from .chat import TextMessage, BinaryMessage, ToolCallMessage, ToolReturnMessage
from .tools import WebSearchInput, MCPStreamableServerInput
from ..common import BaseRequest


class AgentRequest(BaseRequest):
    model: str = Field(..., description="The model to use")
    messages: List[Union[TextMessage, BinaryMessage, ToolCallMessage, ToolReturnMessage]] = Field(..., description="The messages to use")
    tools: List[Union[WebSearchInput, MCPStreamableServerInput]] = Field(default_factory=list, description="The tools to use")
    gen_config: Optional[dict] = Field(default=None, description="The configuration for the generation")
    include_history: bool = Field(default=False, description="Whether to include full conversation history in the response")

class AgentResponseUsage(BaseModel):
    input_tokens: int = Field(..., description="The number of input tokens used")
    output_tokens: int = Field(..., description="The number of output tokens used")

class AgentResponse(BaseModel):
    output: str = Field(..., description="The output of the response")
    usage: AgentResponseUsage = Field(..., description="The usage of the response")
    history: Optional[List[Union[TextMessage, BinaryMessage, ToolCallMessage, ToolReturnMessage]]] = Field(default=None, description="Full conversation history including tool calls and returns (only included when include_history=true)")