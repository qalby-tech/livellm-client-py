from pydantic import BaseModel, Field, field_validator
from typing import Optional, TypeAlias, Tuple, AsyncIterator
from enum import Enum
from ..common import BaseRequest
import base64

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

class EncodedSpeakResponse(BaseModel):
    audio: bytes | str = Field(..., description="The audio data as a base64 encoded string")
    content_type: SpeakMimeType = Field(..., description="The content type of the audio")
    sample_rate: int = Field(..., description="The sample rate of the audio")

    @field_validator("audio", mode="after")
    @classmethod
    def validate_audio(cls, v: bytes | str) -> bytes:
        """
        Ensure that `audio` is always returned as raw bytes.

        - If the server returns a base64-encoded *string*, decode it.
        - If the server already returned raw bytes, pass them through.
        """
        if isinstance(v, str):
            # Server sent base64-encoded string → decode to raw bytes
            return base64.b64decode(v)
        # Already bytes → assume it's raw audio
        return v