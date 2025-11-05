# models for chat messages
from pydantic import BaseModel, Field, model_validator, field_serializer
from enum import Enum
from typing import Optional, Union

class MessageRole(str, Enum):
    USER = "user"
    MODEL = "model"
    SYSTEM = "system"


class Message(BaseModel):
    role: Union[MessageRole, str] = Field(..., description="The role of the message")
    
class TextMessage(Message):
    content: str = Field(..., description="The content of the message")

class BinaryMessage(Message):
    """always from user"""
    content: str = Field(..., description="The base64 encoded content of the message")
    mime_type: str = Field(..., description="The MIME type of the content, only user can supply such")
    caption: Optional[str] = Field(None, description="Caption for the binary message")

    @model_validator(mode="after")
    def validate_content(self) -> "BinaryMessage":
        if self.role == MessageRole.MODEL:
            raise ValueError("MIME type are meant for user messages only")
        return self

