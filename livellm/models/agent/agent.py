# models for full run: AgentRequest, AgentResponse

from pydantic import BaseModel, Field
from typing import Optional, List, Union, Any, Dict
from enum import Enum
from .chat import TextMessage, BinaryMessage, ToolCallMessage, ToolReturnMessage
from .tools import WebSearchInput, MCPStreamableServerInput
from .output_schema import OutputSchema, PropertyDef
from ..common import BaseRequest


class ContextOverflowStrategy(str, Enum):
    """Strategy for handling context overflow when text exceeds context_limit."""
    TRUNCATE = "truncate"  # Take beginning, middle, and end portions
    RECYCLE = "recycle"    # Iteratively process chunks, merging results


class AgentRequest(BaseRequest):
    model: str = Field(..., description="The model to use")
    messages: List[Union[TextMessage, BinaryMessage, ToolCallMessage, ToolReturnMessage]] = Field(..., description="The messages to use")
    tools: List[Union[WebSearchInput, MCPStreamableServerInput]] = Field(default_factory=list, description="The tools to use")
    gen_config: Optional[dict] = Field(default=None, description="The configuration for the generation")
    include_history: bool = Field(default=False, description="Whether to include full conversation history in the response")
    output_schema: Optional[Union[OutputSchema, Dict[str, Any]]] = Field(default=None, description="JSON schema for structured output. Can be an OutputSchema, a dict representing a JSON schema, or will be converted from a Pydantic BaseModel.")
    context_limit: int = Field(default=0, description="Maximum context size in tokens. If <= 0, context overflow handling is disabled.")
    context_overflow_strategy: ContextOverflowStrategy = Field(default=ContextOverflowStrategy.TRUNCATE, description="Strategy for handling context overflow: 'truncate' or 'recycle'")

class AgentResponseUsage(BaseModel):
    input_tokens: int = Field(..., description="The number of input tokens used")
    output_tokens: int = Field(..., description="The number of output tokens used")

class AgentResponse(BaseModel):
    output: str = Field(..., description="The output of the response (JSON string when using output_schema)")
    usage: AgentResponseUsage = Field(..., description="The usage of the response")
    history: Optional[List[Union[TextMessage, BinaryMessage, ToolCallMessage, ToolReturnMessage]]] = Field(default=None, description="Full conversation history including tool calls and returns (only included when include_history=true)")
