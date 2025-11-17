from pydantic import BaseModel, Field, field_validator
from livellm.models.audio.speak import SpeakMimeType
import base64

class TranscriptionInitWsRequest(BaseModel):
    provider_uid: str = Field(..., description="The provider uid")
    model: str = Field(..., description="The model")
    language: str = Field(default="auto", description="The language")
    input_sample_rate: int = Field(default=24000, description="The input sample rate")
    input_audio_format: SpeakMimeType = Field(default=SpeakMimeType.PCM, description="The input audio format (pcm, ulaw, alaw)")
    gen_config: dict = Field(default={}, description="The generation configuration")


class TranscriptionAudioChunkWsRequest(BaseModel):
    audio: str = Field(..., description="The audio (base64 encoded)")

    @field_validator('audio', mode='before')
    @classmethod
    def validate_audio(cls, v: str | bytes) -> str:
        """
        encode audio to base64 string if needed
        """
        if isinstance(v, bytes):
            return base64.b64encode(v).decode("utf-8")
        elif isinstance(v, str):
            return v # already base64 encoded
        else:
            raise ValueError(f"Invalid audio type: {type(v)}")

class TranscriptionWsResponse(BaseModel):
    transcription: str = Field(..., description="The transcription")
    is_end: bool = Field(..., description="Whether the response is the end of the transcription")
