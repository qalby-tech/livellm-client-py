"""Raw HTTP client for the LiveLLM Proxy API using httpx."""

from typing import Any, AsyncIterator, Dict, Optional, Union
import json
import httpx

from .models import (
    AgentRequest,
    AgentResponse,
    SpeakRequest,
    TranscribeResponse,
    FileType,
)

class LivellmProxyClient:
    """
    Raw HTTP client for the LiveLLM Proxy API.
    
    This client provides direct access to all API endpoints using httpx.
    
    Args:
        base_url: Base URL of the LiveLLM Proxy API (e.g., "https://api.example.com")
        timeout: Request timeout in seconds (default: 30.0)
        http2: Enable HTTP/2 support (default: True)
    """
    
    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout
        )
            
    def build_headers(
        self,
        api_key: str,
        provider: str,
        base_url: Optional[str] = None,
    ) -> Dict[str, str]:
        """Build common headers for API requests."""
        headers = {
            "X-Api-Key": api_key,
            "X-Provider": provider,
        }
        if base_url:
            headers["X-Base-Url"] = base_url
        return headers
    
    # ========================================================================
    # Agent Endpoints
    # ========================================================================
    
    async def agent_run(
        self,
        request: AgentRequest,
        api_key: str,
        provider: str,
        base_url: Optional[str] = None,
    ) -> AgentResponse:
        """
        Run an agent with the specified configuration.
        
        Args:
            request: Agent request configuration
            api_key: API key for the provider
            provider: Provider to use (openai, google, anthropic, groq)
            base_url: Optional custom base URL for the provider
            
        Returns:
            AgentResponse with output and usage information
        """
        
        response = await self.client.post(
            "/agent/run",
            json=request.model_dump(exclude_none=True),
            headers=self.build_headers(api_key, provider, base_url),
        )
        if response.status_code != 200:
            error_response = response.json()
            raise ValueError(f"Error running agent: {error_response}")
        
        return AgentResponse.model_validate(response.json())
    
    async def agent_run_stream(
        self,
        request: AgentRequest,
        api_key: str,
        provider: str,
        base_url: Optional[str] = None,
    ) -> AsyncIterator[AgentResponse]:
        """
        Stream agent responses as newline-delimited JSON (NDJSON).
        
        Args:
            request: Agent request configuration
            api_key: API key for the provider
            provider: Provider to use (openai, google, anthropic, groq)
            base_url: Optional custom base URL for the provider
            
        Yields:
            AgentResponse objects for each streamed chunk
        """        
        async with self.client.stream(
            "POST",
            "/agent/run_stream",
            json=request.model_dump(exclude_none=True),
            headers=self.build_headers(api_key, provider, base_url),
        ) as response:
            if response.status_code != 200:
                error_response = await response.aread()
                error_response = error_response.decode("utf-8")
                raise ValueError(f"Error streaming agent: {error_response}")
            
            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                yield AgentResponse.model_validate(data)
    
    # ========================================================================
    # Audio Endpoints
    # ========================================================================
    
    async def audio_speak(
        self,
        request: SpeakRequest,
        api_key: str,
        provider: str,
        base_url: Optional[str] = None,
    ) -> bytes:
        """
        Convert text to speech using the specified audio provider.
        
        Args:
            request: Speak request configuration
            api_key: API key for the provider
            provider: Provider to use (openai, elevenlabs)
            base_url: Optional custom base URL for the provider
            
        Returns:
            Raw audio bytes
        """
        
        response = await self.client.post(
            "/audio/speak",
            json=request.model_dump(exclude_none=True),
            headers=self.build_headers(api_key, provider, base_url),
        )
        response.raise_for_status()
        
        return response.content
    
    async def audio_speak_stream(
        self,
        request: SpeakRequest,
        api_key: str,
        provider: str,
        base_url: Optional[str] = None,
    ) -> AsyncIterator[bytes]:
        """
        Stream text to speech audio using the specified audio provider.
        
        Args:
            request: Speak request configuration
            api_key: API key for the provider
            provider: Provider to use (openai, elevenlabs)
            base_url: Optional custom base URL for the provider
            
        Yields:
            Audio bytes chunks
        """
        async with self.client.stream(
            "POST",
            "/audio/speak_stream",
            json=request.model_dump(exclude_none=True),
            headers=self.build_headers(api_key, provider, base_url),
        ) as response:
            if response.status_code != 200:
                error_response = response.decode("utf-8")
                raise ValueError(f"Error streaming audio: {error_response}")
            
            async for chunk in response.aiter_bytes():
                if chunk:
                    yield chunk
    
    async def audio_transcribe(
        self,
        model: str,
        file: FileType,
        api_key: str,
        provider: str,
        language: Optional[str] = None,
        gen_config: Optional[Dict[str, Any]] = None,
        base_url: Optional[str] = None,
    ) -> TranscribeResponse:
        """
        Transcribe audio to text using the specified audio provider.
        
        Args:
            model: Model to use for transcription
            file: Audio file as bytes or tuple of (filename, bytes)
            api_key: API key for the provider
            provider: Provider to use (openai, elevenlabs)
            language: Optional language hint
            gen_config: Optional generation configuration (as JSON string)
            base_url: Optional custom base URL for the provider
            
        Returns:
            TranscribeResponse with transcribed text and language
        """
        headers = self.build_headers(api_key, provider, base_url)
        
        # Prepare form data
        data = {"model": model}
        if language:
            data["language"] = language
        if gen_config:
            data["gen_config"] = json.dumps(gen_config)
        
        # Prepare file
        files = {"file": file}
        
        response = await self.client.post(
            "/audio/transcribe",
            data=data,
            files=files,
            headers=headers,
        )
        response.raise_for_status()
        
        return TranscribeResponse.model_validate(response.json())
    
    # ========================================================================
    # Health Check
    # ========================================================================
    
    async def ping(self) -> Dict[str, Any]:
        """
        Health check endpoint.
        
        Returns:
            Response from the ping endpoint
        """
        response = await self.client.get("/ping")
        response.raise_for_status()
        return response.json()

