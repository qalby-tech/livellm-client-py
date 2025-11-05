from pydantic import BaseModel, Field, field_validator
from typing import Optional, TypeAlias, Tuple, AsyncIterator
from enum import Enum
from ..common import BaseRequest

SpeakStreamResponse: TypeAlias = Tuple[AsyncIterator[bytes], str, int]


class SpeakMimeType(str, Enum):
    PCM = "audio/pcm"
    WAV = "audio/wav"
    MP3 = "audio/mpeg"
    ULAW = "audio/ulaw"
    ALAW = "audio/alaw"

class SpeakRequest(BaseRequest):
    model: str = Field(..., description="The model to use")
    text: str = Field(..., description="The text to speak")
    voice: str = Field(..., description="The voice to use")
    mime_type: SpeakMimeType = Field(..., description="The MIME type of the output audio")
    sample_rate: int = Field(..., description="The target sample rate of the output audio")
    chunk_size: int = Field(default=20, description="Chunk size in milliseconds for streaming (default: 20ms)")
    gen_config: Optional[dict] = Field(default=None, description="The configuration for the generation")
