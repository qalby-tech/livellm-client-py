from pydantic import BaseModel, Field, model_validator
from typing import List
from .common import BaseRequest
from .audio.speak import SpeakRequest
from .audio.transcribe import TranscribeRequest
from .agent.agent import AgentRequest
from enum import Enum

class FallbackStrategy(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    
class FallbackRequest(BaseModel):
    requests: List[BaseRequest] = Field(..., description="List of requests to try as fallbacks")
    strategy: FallbackStrategy = Field(FallbackStrategy.SEQUENTIAL, description="The strategy to use for fallback")
    timeout_per_request: int = Field(default=360, description="The timeout to use for each request")
    
class AgentFallbackRequest(FallbackRequest):
    requests: List[AgentRequest] = Field(..., description="List of agent requests to try as fallbacks")

class AudioFallbackRequest(FallbackRequest):
    requests: List[SpeakRequest] = Field(..., description="List of audio requests to try as fallbacks")

class TranscribeFallbackRequest(FallbackRequest):
    requests: List[TranscribeRequest] = Field(..., description="List of transcribe requests to try as fallbacks")