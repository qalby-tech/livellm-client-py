"""Error-related models for LiveLLM Proxy Client."""

from typing import List, Union
from pydantic import BaseModel, Field


class ValidationError(BaseModel):
    """Validation error details."""
    loc: List[Union[str, int]] = Field(description="Location of the error")
    msg: str = Field(description="Error message")
    type: str = Field(description="Error type")


class HTTPValidationError(BaseModel):
    """HTTP validation error response."""
    detail: List[ValidationError]

