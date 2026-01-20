# models for full run: AgentRequest, AgentResponse

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Union, Any, Dict, Type
from .chat import TextMessage, BinaryMessage, ToolCallMessage, ToolReturnMessage
from .tools import WebSearchInput, MCPStreamableServerInput
from .output_schema import OutputSchema, PropertyDef
from ..common import BaseRequest
import json


class AgentRequest(BaseRequest):
    model: str = Field(..., description="The model to use")
    messages: List[Union[TextMessage, BinaryMessage, ToolCallMessage, ToolReturnMessage]] = Field(..., description="The messages to use")
    tools: List[Union[WebSearchInput, MCPStreamableServerInput]] = Field(default_factory=list, description="The tools to use")
    gen_config: Optional[dict] = Field(default=None, description="The configuration for the generation")
    include_history: bool = Field(default=False, description="Whether to include full conversation history in the response")
    output_schema: Optional[Union[OutputSchema, Dict[str, Any]]] = Field(default=None, description="JSON schema for structured output. Can be an OutputSchema, a dict representing a JSON schema, or will be converted from a Pydantic BaseModel.")

class AgentResponseUsage(BaseModel):
    input_tokens: int = Field(..., description="The number of input tokens used")
    output_tokens: int = Field(..., description="The number of output tokens used")

class AgentResponse(BaseModel):
    output: str = Field(..., description="The output of the response (JSON string when using output_schema)")
    usage: AgentResponseUsage = Field(..., description="The usage of the response")
    history: Optional[List[Union[TextMessage, BinaryMessage, ToolCallMessage, ToolReturnMessage]]] = Field(default=None, description="Full conversation history including tool calls and returns (only included when include_history=true)")
    structured_output: Optional[Dict[str, Any]] = Field(default=None, description="Parsed structured output when output_schema is provided in the request")
    
    @model_validator(mode="after")
    def parse_structured_output(self) -> "AgentResponse":
        """Parse the output as JSON if it appears to be structured output."""
        if self.structured_output is None and self.output:
            # Try to parse output as JSON for structured output
            try:
                parsed = json.loads(self.output)
                if isinstance(parsed, dict):
                    self.structured_output = parsed
            except (json.JSONDecodeError, TypeError):
                # Not JSON or not a dict, leave structured_output as None
                pass
        return self