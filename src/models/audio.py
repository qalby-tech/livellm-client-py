"""Audio-related models for LiveLLM Proxy Client."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from typing import Tuple, TypeAlias


class SpeakRequest(BaseModel):
    """Request to convert text to speech."""
    model: str = Field(description="The model to use")
    text: str = Field(description="The text to speak")
    voice: str = Field(description="The voice to use")
    output_format: str = Field(description="The output format of the audio")
    gen_config: Optional[Dict[str, Any]] = Field(
        None,
        description="The configuration for the generation"
    )


FileType: TypeAlias = Tuple[str, bytes, str] # (filename, file_content, content_type)

class TranscribeResponse(BaseModel):
    """Response from audio transcription."""
    text: str = Field(description="The text of the transcription")
    language: Optional[str] = Field(None, description="The language of the transcription")

