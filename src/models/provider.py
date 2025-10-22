"""Provider-related models for LiveLLM Proxy Client."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class Creds(BaseModel):
    """Provider credentials."""
    api_key: str = Field(description="The API key to use for the provider")
    provider: str = Field(description="The provider to use")
    base_url: Optional[str] = Field(None, description="The base URL to use for the provider")


class ModelCapability(Enum):
    """Model capabilities enumeration."""
    AUDIO_AGENT = "audio_agent"
    IMAGE_AGENT = "image_agent"
    VIDEO_AGENT = "video_agent"
    SPEAK = "speak"
    TRANSCRIBE = "transcribe"


class Model(BaseModel):
    """Model configuration."""
    name: str = Field(description="The name of the model")
    capabilities: List[ModelCapability] = Field(description="The capabilities of the model")


class ProviderConfig(BaseModel):
    """Provider configuration."""
    creds: Creds = Field(description="The credentials for the provider")
    models: List[Model] = Field(description="The models for the provider")

