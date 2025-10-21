"""Raw HTTP client for the LiveLLM Proxy API using httpx."""

from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Union
import json
import httpx
from pydantic import BaseModel, Field


# ============================================================================
# Request/Response Models
# ============================================================================

class MessageRole(str):
    """Role of a message."""
    USER = "user"
    MODEL = "model"
    SYSTEM = "system"


class TextMessage(BaseModel):
    """Text message in a conversation."""
    role: MessageRole = Field(description="The role of the message")
    content: str = Field(description="The content of the message")


class BinaryMessage(BaseModel):
    """Binary message (e.g., image, audio) - always from user."""
    role: MessageRole = Field(default=MessageRole.USER, description="The role of the message")
    content: bytes = Field(description="The content of the message", format="binary")
    mime_type: str = Field(description="The MIME type of the content, only user can supply such")
    caption: Optional[str] = Field(None, description="Caption for the binary message")


class ToolKind(str):
    """Type of tool."""
    WEB_SEARCH = "web_search"
    MCP_STREAMABLE_SERVER = "mcp_streamable_server"


class WebSearchInput(BaseModel):
    """Web search tool configuration."""
    search_context_size: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="The context size for the search"
    )


class MCPStreamableServerInput(BaseModel):
    """MCP server tool configuration."""
    url: str = Field(description="The URL of the MCP server")
    prefix: str = Field(description="The prefix of the MCP server")


class AgentRequest(BaseModel):
    """Request to run an agent."""
    model: str = Field(description="The model to use")
    messages: List[Union[TextMessage, BinaryMessage]] = Field(description="Conversation messages")
    tools: List[Union[WebSearchInput, MCPStreamableServerInput]] = Field(description="Tools available to the agent")
    gen_config: Optional[Dict[str, Any]] = Field(
        None,
        description="The configuration for the generation"
    )


class AgentResponseUsage(BaseModel):
    """Token usage information."""
    input_tokens: int = Field(description="The number of input tokens used")
    output_tokens: int = Field(description="The number of output tokens used")


class AgentResponse(BaseModel):
    """Response from an agent run."""
    output: str = Field(description="The output of the response")
    usage: AgentResponseUsage = Field(description="The usage of the response")


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


class TranscribeResponse(BaseModel):
    """Response from audio transcription."""
    text: str = Field(description="The text of the transcription")
    language: Optional[str] = Field(None, description="The language of the transcription")


class ValidationError(BaseModel):
    """Validation error details."""
    loc: List[Union[str, int]] = Field(description="Location of the error")
    msg: str = Field(description="Error message")
    type: str = Field(description="Error type")


class HTTPValidationError(BaseModel):
    """HTTP validation error response."""
    detail: List[ValidationError]


# ============================================================================
# Raw HTTP Client
# ============================================================================

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
        timeout: float = 30.0,
        http2: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            http2=http2,
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
        response.raise_for_status()
        
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
                error_response = response.decode("utf-8")
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
        file: Union[bytes, tuple[str, bytes]],
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
        if isinstance(file, tuple):
            files = {"file": file}
        else:
            files = {"file": ("audio", file)}
        
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

