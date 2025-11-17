from pydantic import BaseModel, Field
from enum import Enum
from typing import Union, Optional

class WsAction(str, Enum):
    AGENT_RUN = "agent_run"
    AGENT_RUN_STREAM = "agent_run_stream"
    AUDIO_SPEAK = "audio_speak"
    AUDIO_SPEAK_STREAM = "audio_speak_stream"
    AUDIO_TRANSCRIBE = "audio_transcribe"
    TRANSCRIPTION_SESSION = "transcription_session"


class WsStatus(str, Enum):
    STREAMING = "streaming"
    SUCCESS = "success"
    ERROR = "error"

class WsRequest(BaseModel):
    action: WsAction = Field(..., description="The action to perform")
    payload: Union[dict, BaseModel] = Field(..., description="The payload for the action")


class WsResponse(BaseModel):
    status: WsStatus = Field(..., description="The status of the response")
    action: WsAction = Field(..., description="The action that was performed")
    data: Union[dict, BaseModel] = Field(..., description="The data for the response")
    error: Optional[str] = Field(default=None, description="The error message if the response is an error")