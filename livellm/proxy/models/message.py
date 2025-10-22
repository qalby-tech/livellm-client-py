"""Message-related models for LiveLLM Proxy Client."""

from typing import Optional, Union
import base64
from pydantic import BaseModel, Field


class MessageRole:
    """Role of a message."""
    USER: str = "user"
    MODEL: str = "model"
    SYSTEM: str = "system"


class TextMessage(BaseModel):
    """Text message in a conversation."""
    role: str = Field(description="The role of the message")
    content: str = Field(description="The content of the message")


class BinaryMessage(BaseModel):
    """Binary message (e.g., image, audio) - always from user."""
    role: str = Field(default=MessageRole.USER, description="The role of the message")
    content: str = Field(description="The content of the message as a base64 encoded string")
    mime_type: str = Field(description="The MIME type of the content, only user can supply such")
    caption: Optional[str] = Field(None, description="Caption for the binary message")
    
    @classmethod
    def from_bytes(
        cls,
        content: bytes,
        mime_type: str,
        role: str = MessageRole.USER,
        caption: Optional[str] = None
    ) -> "BinaryMessage":
        """
        Create a BinaryMessage from raw bytes by encoding to base64.
        
        Args:
            content: Raw binary content
            mime_type: MIME type of the content
            role: Message role (default: user)
            caption: Optional caption
            
        Returns:
            BinaryMessage with base64 encoded content
        """
        base64_content = base64.b64encode(content).decode('utf-8')
        return cls(
            role=role,
            content=base64_content,
            mime_type=mime_type,
            caption=caption
        )

