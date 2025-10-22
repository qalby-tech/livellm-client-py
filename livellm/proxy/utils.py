"""Utility functions for LiveLLM Proxy Client."""

from typing import Optional, List
from .models import ProviderConfig, Creds, Model, ModelCapability


def create_openai_provider_config(api_key: str, base_url: Optional[str] = None) -> ProviderConfig:
    """
    Create OpenAI provider configuration.
    
    Args:
        api_key: OpenAI API key
        base_url: Optional custom base URL
        
    Returns:
        ProviderConfig for OpenAI
    """
    return ProviderConfig(
        creds=Creds(api_key=api_key, provider="openai", base_url=base_url),
        models=[
            Model(name="gpt-5-mini", capabilities=[]),
            Model(name="gpt-5-nano", capabilities=[]),
            Model(name="gpt-5", capabilities=[]),
            Model(name="gpt-4o", capabilities=[ModelCapability.IMAGE_AGENT]),
            Model(name="gpt-4o-mini", capabilities=[ModelCapability.IMAGE_AGENT]),
            Model(name="tts-1", capabilities=[ModelCapability.SPEAK]),
            Model(name="tts-1-hd", capabilities=[ModelCapability.SPEAK]),
            Model(name="whisper-1", capabilities=[ModelCapability.TRANSCRIBE]),
        ]
    )


def create_google_provider_config(api_key: str, base_url: Optional[str] = None) -> ProviderConfig:
    """
    Create Google provider configuration.
    
    Args:
        api_key: Google API key
        base_url: Optional custom base URL
        
    Returns:
        ProviderConfig for Google
    """
    gemini_caps = [ModelCapability.IMAGE_AGENT, ModelCapability.VIDEO_AGENT, ModelCapability.AUDIO_AGENT]
    return ProviderConfig(
        creds=Creds(api_key=api_key, provider="google", base_url=base_url),
        models=[
            Model(name="gemini-2.5-flash-lite", capabilities=gemini_caps),
            Model(name="gemini-2.5-flash", capabilities=gemini_caps),
            Model(name="gemini-2.5-pro", capabilities=gemini_caps),
        ],
    )


def create_elevenlabs_provider_config(api_key: str, base_url: Optional[str] = None) -> ProviderConfig:
    """
    Create ElevenLabs provider configuration.
    
    Args:
        api_key: ElevenLabs API key
        base_url: Optional custom base URL
        
    Returns:
        ProviderConfig for ElevenLabs
    """
    return ProviderConfig(
        creds=Creds(api_key=api_key, provider="elevenlabs", base_url=base_url),
        models=[
            Model(name="elevenlabs_multilingual_v2", capabilities=[ModelCapability.SPEAK]),
            Model(name="eleven_flash_v2_5", capabilities=[ModelCapability.SPEAK]),
            Model(name="eleven_flash_v2", capabilities=[ModelCapability.SPEAK]),
            Model(name="eleven_v3", capabilities=[ModelCapability.SPEAK]),
            Model(name="scribe_v1", capabilities=[ModelCapability.TRANSCRIBE]),
        ],
    )


def create_anthropic_provider_config(api_key: str, base_url: Optional[str] = None) -> ProviderConfig:
    """
    Create Anthropic provider configuration.
    
    Args:
        api_key: Anthropic API key
        base_url: Optional custom base URL
        
    Returns:
        ProviderConfig for Anthropic
    """
    return ProviderConfig(
        creds=Creds(api_key=api_key, provider="anthropic", base_url=base_url),
        models=[
            Model(name="claude-sonnet-3.5", capabilities=[]),
            Model(name="claude-sonnet-4.0", capabilities=[]),
            Model(name="claude-sonnet-4.5", capabilities=[]),
            Model(name="claude-haiku-4.5", capabilities=[]),
        ]
    )

