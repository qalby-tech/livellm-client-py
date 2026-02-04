"""LiveLLM Client - Python client for the LiveLLM Proxy and Realtime APIs."""
import asyncio
import httpx
import json
import warnings
from typing import List, Optional, AsyncIterator, Union, overload, Dict, Any, Type
from .models.common import Settings, SuccessResponse
from .models.agent.agent import AgentRequest, AgentResponse
from .models.agent.output_schema import OutputSchema
from .models.audio.speak import SpeakRequest, EncodedSpeakResponse
from .models.audio.transcribe import TranscribeRequest, TranscribeResponse, File
from .models.fallback import AgentFallbackRequest, AudioFallbackRequest, TranscribeFallbackRequest
import websockets
from .models.ws import WsRequest, WsResponse, WsStatus, WsAction
from .transcripton import TranscriptionWsClient
from uuid import uuid4
import logging
from abc import ABC, abstractmethod
from importlib.metadata import version, PackageNotFoundError
from pydantic import BaseModel


logger = logging.getLogger(__name__)

try:
    __version__ = version("livellm")
except PackageNotFoundError:
    __version__ = "unknown"

DEFAULT_USER_AGENT = f"livellm-python/{__version__}"

class BaseLivellmClient(ABC):

    # Default timeout (set by subclasses)
    timeout: Optional[float] = None

    @overload
    async def agent_run(
        self,
        request: Union[AgentRequest, AgentFallbackRequest],
        *,
        timeout: Optional[float] = None,
    ) -> AgentResponse:
        ...
    
    @overload
    async def agent_run(
        self,
        *,
        provider_uid: str,
        model: str,
        messages: list,
        tools: Optional[list] = None,
        include_history: bool = False,
        output_schema: Optional[Union[OutputSchema, Dict[str, Any], Type[BaseModel]]] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> AgentResponse:
        ...
    
    
    @abstractmethod
    async def handle_agent_run(
        self, 
        request: Union[AgentRequest, AgentFallbackRequest],
        timeout: Optional[float] = None
    ) -> AgentResponse:
        ...
    
    async def agent_run(
        self,
        request: Optional[Union[AgentRequest, AgentFallbackRequest]] = None,
        *,
        provider_uid: Optional[str] = None,
        model: Optional[str] = None,
        messages: Optional[list] = None,
        tools: Optional[list] = None,
        include_history: bool = False,
        output_schema: Optional[Union[OutputSchema, Dict[str, Any], Type[BaseModel]]] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> AgentResponse:
        """
        Run an agent request.
        
        Can be called in two ways:
        
        1. With a request object:
           await client.agent_run(AgentRequest(...))
           await client.agent_run(AgentFallbackRequest(...))
           
        2. With individual parameters (keyword arguments):
           await client.agent_run(
               provider_uid="...",
               model="gpt-4",
               messages=[TextMessage(...)],
               tools=[],
               include_history=False,
               output_schema=MyPydanticModel  # or OutputSchema(...) or dict
           )
        
        Args:
            request: An AgentRequest or AgentFallbackRequest object
            provider_uid: The provider UID string
            model: The model to use
            messages: List of messages
            tools: Optional list of tools
            gen_config: Optional generation configuration
            include_history: Whether to include full conversation history in the response
            output_schema: Optional schema for structured output. Can be:
                - An OutputSchema instance
                - A dict representing a JSON schema
                - A Pydantic BaseModel class (will be converted to OutputSchema)
            timeout: Optional timeout in seconds (overrides default client timeout)
            
        Returns:
            AgentResponse with the agent's output. If output_schema was provided,
            the output will be a JSON string matching the schema.
        """
        # Check if first argument is a request object
        if request is not None:
            if not isinstance(request, (AgentRequest, AgentFallbackRequest)):
                raise TypeError(
                    f"First positional argument must be AgentRequest or AgentFallbackRequest, got {type(request)}"
                )
            return await self.handle_agent_run(request, timeout=timeout)
    
        # Otherwise, use keyword arguments
        if provider_uid is None or model is None or messages is None:
            raise ValueError(
                "provider_uid, model, and messages are required. "
                "Alternatively, pass an AgentRequest object as the first positional argument."
            )
        
        # Convert output_schema if it's a Pydantic BaseModel class
        resolved_schema = self._resolve_output_schema(output_schema)
        
        agent_request = AgentRequest(
            provider_uid=provider_uid,
            model=model,
            messages=messages,
            tools=tools or [],
            gen_config=kwargs or None,
            include_history=include_history,
            output_schema=resolved_schema
        )
        return await self.handle_agent_run(agent_request, timeout=timeout)
    
    def _resolve_output_schema(
        self, 
        output_schema: Optional[Union[OutputSchema, Dict[str, Any], Type[BaseModel]]]
    ) -> Optional[Union[OutputSchema, Dict[str, Any]]]:
        """
        Resolve the output_schema parameter to an OutputSchema or dict.
        
        If a Pydantic BaseModel class is provided, convert it to OutputSchema.
        """
        if output_schema is None:
            return None
        
        # Check if it's a class (not an instance) that's a subclass of BaseModel
        if isinstance(output_schema, type) and issubclass(output_schema, BaseModel):
            return OutputSchema.from_pydantic(output_schema)
        
        # Already an OutputSchema or dict, return as-is
        return output_schema
    
    @overload
    def agent_run_stream(
        self,
        request: Union[AgentRequest, AgentFallbackRequest],
        *,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[AgentResponse]:
        ...
    
    @overload
    def agent_run_stream(
        self,
        *,
        provider_uid: str,
        model: str,
        messages: list,
        tools: Optional[list] = None,
        include_history: bool = False,
        output_schema: Optional[Union[OutputSchema, Dict[str, Any], Type[BaseModel]]] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> AsyncIterator[AgentResponse]:
        ...
    
    
    @abstractmethod
    async def handle_agent_run_stream(
        self, 
        request: Union[AgentRequest, AgentFallbackRequest],
        timeout: Optional[float] = None
    ) -> AsyncIterator[AgentResponse]:
        ...
    
    async def agent_run_stream(
        self,
        request: Optional[Union[AgentRequest, AgentFallbackRequest]] = None,
        *,
        provider_uid: Optional[str] = None,
        model: Optional[str] = None,
        messages: Optional[list] = None,
        tools: Optional[list] = None,
        include_history: bool = False,
        output_schema: Optional[Union[OutputSchema, Dict[str, Any], Type[BaseModel]]] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> AsyncIterator[AgentResponse]:
        """
        Run an agent request with streaming response.
        
        Can be called in two ways:
        
        1. With a request object:
           async for chunk in client.agent_run_stream(AgentRequest(...)):
               ...
           async for chunk in client.agent_run_stream(AgentFallbackRequest(...)):
               ...
           
        2. With individual parameters (keyword arguments):
           async for chunk in client.agent_run_stream(
               provider_uid="...",
               model="gpt-4",
               messages=[TextMessage(...)],
               tools=[],
               include_history=False,
               output_schema=MyPydanticModel  # or OutputSchema(...) or dict
           ):
               ...
        
        Args:
            request: An AgentRequest or AgentFallbackRequest object
            provider_uid: The provider UID string
            model: The model to use
            messages: List of messages
            tools: Optional list of tools
            gen_config: Optional generation configuration
            include_history: Whether to include full conversation history in the response
            output_schema: Optional schema for structured output. Can be:
                - An OutputSchema instance
                - A dict representing a JSON schema
                - A Pydantic BaseModel class (will be converted to OutputSchema)
            timeout: Optional timeout in seconds (overrides default client timeout)
            
        Returns:
            AsyncIterator of AgentResponse chunks. If output_schema was provided,
            the output will be a JSON string matching the schema.
        """
        # Check if first argument is a request object
        if request is not None:
            if not isinstance(request, (AgentRequest, AgentFallbackRequest)):
                raise TypeError(
                    f"First positional argument must be AgentRequest or AgentFallbackRequest, got {type(request)}"
                )
            stream = self.handle_agent_run_stream(request, timeout=timeout)
        else:
            # Otherwise, use keyword arguments
            if provider_uid is None or model is None or messages is None:
                raise ValueError(
                    "provider_uid, model, and messages are required. "
                    "Alternatively, pass an AgentRequest object as the first positional argument."
                )
            
            # Convert output_schema if it's a Pydantic BaseModel class
            resolved_schema = self._resolve_output_schema(output_schema)
            
            agent_request = AgentRequest(
                provider_uid=provider_uid,
                model=model,
                messages=messages,
                tools=tools or [],
                gen_config=kwargs or None,
                include_history=include_history,
                output_schema=resolved_schema
            )
            stream = self.handle_agent_run_stream(agent_request, timeout=timeout)
        
        async for chunk in stream:
            yield chunk
    
    @overload
    async def speak(
        self,
        request: Union[SpeakRequest, AudioFallbackRequest],
        *,
        timeout: Optional[float] = None,
    ) -> bytes:
        ...
    
    @overload
    async def speak(
        self,
        *,
        provider_uid: str,
        model: str,
        text: str,
        voice: str,
        mime_type: str,
        sample_rate: int,
        chunk_size: int = 20,
        timeout: Optional[float] = None,
        **kwargs
    ) -> bytes:
        ...
    
    
    @abstractmethod
    async def handle_speak(
        self, 
        request: Union[SpeakRequest, AudioFallbackRequest],
        timeout: Optional[float] = None
    ) -> bytes:
        ...
    
    async def speak(
        self,
        request: Optional[Union[SpeakRequest, AudioFallbackRequest]] = None,
        *,
        provider_uid: Optional[str] = None,
        model: Optional[str] = None,
        text: Optional[str] = None,
        voice: Optional[str] = None,
        mime_type: Optional[str] = None,
        sample_rate: Optional[int] = None,
        chunk_size: int = 20,
        timeout: Optional[float] = None,
        **kwargs
    ) -> bytes:
        """
        Generate speech from text.
        
        Can be called in two ways:
        
        1. With a request object:
           await client.speak(SpeakRequest(...))
           await client.speak(AudioFallbackRequest(...))
           
        2. With individual parameters (keyword arguments):
           await client.speak(
               provider_uid="...",
               model="tts-1",
               text="Hello, world!",
               voice="alloy",
               mime_type="audio/pcm",
               sample_rate=24000
           )
        
        Args:
            request: A SpeakRequest or AudioFallbackRequest object
            provider_uid: The provider UID string
            model: The model to use for TTS
            text: The text to convert to speech
            voice: The voice to use
            mime_type: The MIME type of the output audio
            sample_rate: The sample rate of the output audio
            chunk_size: Chunk size in milliseconds (default: 20ms)
            timeout: Optional timeout in seconds (overrides default client timeout)
            gen_config: Optional generation configuration
            
        Returns:
            Audio data as bytes
        """
        # Check if first argument is a request object
        if request is not None:
            if not isinstance(request, (SpeakRequest, AudioFallbackRequest)):
                raise TypeError(
                    f"First positional argument must be SpeakRequest or AudioFallbackRequest, got {type(request)}"
                )
            return await self.handle_speak(request, timeout=timeout)
        
        # Otherwise, use keyword arguments
        if provider_uid is None or model is None or text is None or voice is None or mime_type is None or sample_rate is None:
            raise ValueError(
                "provider_uid, model, text, voice, mime_type, and sample_rate are required. "
                "Alternatively, pass a SpeakRequest object as the first positional argument."
            )
        
        speak_request = SpeakRequest(
            provider_uid=provider_uid,
            model=model,
            text=text,
            voice=voice,
            mime_type=mime_type,
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            gen_config=kwargs or None
        )
        return await self.handle_speak(speak_request, timeout=timeout)
    
    @overload
    def speak_stream(
        self,
        request: Union[SpeakRequest, AudioFallbackRequest],
        *,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[bytes]:
        ...
    
    @overload
    def speak_stream(
        self,
        *,
        provider_uid: str,
        model: str,
        text: str,
        voice: str,
        mime_type: str,
        sample_rate: int,
        chunk_size: int = 20,
        timeout: Optional[float] = None,
        **kwargs
    ) -> AsyncIterator[bytes]:
        ...
    
    
    @abstractmethod
    async def handle_speak_stream(
        self, 
        request: Union[SpeakRequest, AudioFallbackRequest],
        timeout: Optional[float] = None
    ) -> AsyncIterator[bytes]:
        ...
    
    async def speak_stream(
        self,
        request: Optional[Union[SpeakRequest, AudioFallbackRequest]] = None,
        *,
        provider_uid: Optional[str] = None,
        model: Optional[str] = None,
        text: Optional[str] = None,
        voice: Optional[str] = None,
        mime_type: Optional[str] = None,
        sample_rate: Optional[int] = None,
        chunk_size: int = 20,
        timeout: Optional[float] = None,
        **kwargs
    ) -> AsyncIterator[bytes]:
        """
        Generate speech from text with streaming response.
        
        Can be called in two ways:
        
        1. With a request object:
           async for chunk in client.speak_stream(SpeakRequest(...)):
               ...
           async for chunk in client.speak_stream(AudioFallbackRequest(...)):
               ...
           
        2. With individual parameters (keyword arguments):
           async for chunk in client.speak_stream(
               provider_uid="...",
               model="tts-1",
               text="Hello, world!",
               voice="alloy",
               mime_type="audio/pcm",
               sample_rate=24000
           ):
               ...
        
        Args:
            request: A SpeakRequest or AudioFallbackRequest object
            provider_uid: The provider UID string
            model: The model to use for TTS
            text: The text to convert to speech
            voice: The voice to use
            mime_type: The MIME type of the output audio
            sample_rate: The sample rate of the output audio
            chunk_size: Chunk size in milliseconds (default: 20ms)
            timeout: Optional timeout in seconds (overrides default client timeout)
            gen_config: Optional generation configuration
            
        Returns:
            AsyncIterator of audio data chunks as bytes
        """
        # Check if first argument is a request object
        if request is not None:
            if not isinstance(request, (SpeakRequest, AudioFallbackRequest)):
                raise TypeError(
                    f"First positional argument must be SpeakRequest or AudioFallbackRequest, got {type(request)}"
                )
            speak_stream = self.handle_speak_stream(request, timeout=timeout)
        else:
            # Otherwise, use keyword arguments
            if provider_uid is None or model is None or text is None or voice is None or mime_type is None or sample_rate is None:
                raise ValueError(
                    "provider_uid, model, text, voice, mime_type, and sample_rate are required. "
                    "Alternatively, pass a SpeakRequest object as the first positional argument."
                )
            
            speak_request = SpeakRequest(
                provider_uid=provider_uid,
                model=model,
                text=text,
                voice=voice,
                mime_type=mime_type,
                sample_rate=sample_rate,
                chunk_size=chunk_size,
                gen_config=kwargs or None
            )
            speak_stream = self.handle_speak_stream(speak_request, timeout=timeout)
        async for chunk in speak_stream:
            yield chunk
    
    @overload
    async def transcribe(
        self,
        request: Union[TranscribeRequest, TranscribeFallbackRequest],
        *,
        timeout: Optional[float] = None,
    ) -> TranscribeResponse:
        ...
    
    @overload
    async def transcribe(
        self,
        *,
        provider_uid: str,
        file: File,
        model: str,
        language: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> TranscribeResponse:
        ...
        
    
    @abstractmethod
    async def handle_transcribe(
        self, 
        request: Union[TranscribeRequest, TranscribeFallbackRequest],
        timeout: Optional[float] = None
    ) -> TranscribeResponse:
        ...
    
    async def transcribe(
        self,
        request: Optional[Union[TranscribeRequest, TranscribeFallbackRequest]] = None,
        *,
        provider_uid: Optional[str] = None,
        file: Optional[File] = None,
        model: Optional[str] = None,
        language: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> TranscribeResponse:
        """
        Transcribe audio to text.
        
        Can be called in two ways:
        
        1. With a request object:
           await client.transcribe(TranscribeRequest(...))
           
        2. With individual parameters (keyword arguments):
           await client.transcribe(
               provider_uid="...",
               file=("filename", audio_bytes, "audio/wav"),
               model="whisper-1"
           )
        
        Args:
            request: A TranscribeRequest or TranscribeFallbackRequest object
            provider_uid: The provider UID string
            file: The audio file as a tuple (filename, content, content_type)
            model: The model to use for transcription
            language: Optional language code
            timeout: Optional timeout in seconds (overrides default client timeout)
            gen_config: Optional generation configuration
            
        Returns:
            TranscribeResponse with transcription text and detected language
        """
        # Check if first argument is a request object
        if request is not None:
            if not isinstance(request, (TranscribeRequest, TranscribeFallbackRequest)):
                raise TypeError(
                    f"First positional argument must be TranscribeRequest or TranscribeFallbackRequest, got {type(request)}"
                )
            # JSON-based request
            return await self.handle_transcribe(request, timeout=timeout)
            
        # Otherwise, use keyword arguments with multipart form-data request
        if provider_uid is None or file is None or model is None:
            raise ValueError(
                "provider_uid, file, and model are required. "
                "Alternatively, pass a TranscribeRequest object as the first positional argument."
            )
        
        transcribe_request = TranscribeRequest(
            provider_uid=provider_uid,
            file=file,
            model=model,
            language=language,
            gen_config=kwargs or None
        )
        return await self.handle_transcribe(transcribe_request, timeout=timeout)


class LivellmWsClient(BaseLivellmClient):
    """WebSocket-based LiveLLM client for real-time bidirectional communication."""

    def __init__(
        self, 
        base_url: str,
        user_agent: Optional[str] = None,
        timeout: Optional[float] = None,
        max_size: Optional[int] = None,
        max_buffer_size: Optional[int] = None
    ):
        # Convert HTTP(S) URL to WS(S) URL
        base_url = base_url.rstrip("/")
        if base_url.startswith("https://"):
            ws_url = base_url.replace("https://", "wss://")
        elif base_url.startswith("http://"):
            ws_url = base_url.replace("http://", "ws://")
        else:
            ws_url = base_url

        # Root WebSocket base URL (without path) and main /ws endpoint
        self._ws_root_base_url = ws_url
        self.base_url = f"{ws_url}/livellm/ws"
        self.timeout = timeout
        self.user_agent = user_agent or DEFAULT_USER_AGENT
        self.websocket = None
        self.sessions: Dict[str, asyncio.Queue] = {}
        self.max_buffer_size = max_buffer_size or 0 # None means unlimited buffer size
        # Lazily-created clients
        self._transcription = None
        self.max_size = max_size or 1024 * 1024 * 10 # 10MB is default max size

        self.__listen_for_responses_task = None
        
    async def connect(self):
        """Establish WebSocket connection."""
        if self.websocket is not None:
            return self.websocket
        
        self.websocket = await websockets.connect(
            self.base_url, 
            open_timeout=self.timeout,
            close_timeout=self.timeout,
            max_size=self.max_size,
            additional_headers={"User-Agent": self.user_agent}
        )
        self.__listen_for_responses_task = asyncio.create_task(self.listen_for_responses())
                
        return self.websocket
    
    async def listen_for_responses(self):
        while True:
            response_data = await self.websocket.recv()
            response = WsResponse(**json.loads(response_data))            
            try:
                self.sessions[response.session_id].put_nowait(response)
            except asyncio.QueueFull:
                self.sessions[response.session_id].get_nowait()
                logger.warning(f"Session {response.session_id} buffer is full, dropping oldest message")
        
    async def get_or_update_session(self, session_id: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = asyncio.Queue(maxsize=self.max_buffer_size)
        return self.sessions[session_id]
    
    async def disconnect(self):
        """Close WebSocket connection."""
        if self.websocket is not None:
            await self.websocket.close()
            self.websocket = None
        if self.__listen_for_responses_task is not None:
            self.__listen_for_responses_task.cancel()
            self.__listen_for_responses_task = None
        self.sessions.clear()
    
    def _get_effective_timeout(self, timeout: Optional[float]) -> Optional[float]:
        """Get effective timeout: per-request timeout overrides default."""
        return timeout if timeout is not None else self.timeout
    
    async def get_response(self, action: WsAction, payload: dict, timeout: Optional[float] = None) -> dict:
        """Send a request and wait for response."""
        if self.websocket is None:
            await self.connect()
        
        session_id = uuid4().hex
        request = WsRequest(session_id=session_id, action=action, payload=payload)
        q = await self.get_or_update_session(session_id)
        await self.websocket.send(json.dumps(request.model_dump()))
        
        effective_timeout = self._get_effective_timeout(timeout)
        
        try:
            if effective_timeout:
                response: WsResponse = await asyncio.wait_for(q.get(), timeout=effective_timeout)
            else:
                response: WsResponse = await q.get()
        except asyncio.TimeoutError:
            self.sessions.pop(session_id, None)
            raise TimeoutError(f"Request timed out after {effective_timeout} seconds")
        
        self.sessions.pop(session_id)
        if response.status == WsStatus.ERROR:
            raise Exception(f"WebSocket failed: {response.error}")
        elif response.status == WsStatus.SUCCESS:
            return response.data
        else:
            raise Exception(f"WebSocket failed with unknown status: {response}")
    
    async def get_response_stream(self, action: WsAction, payload: dict, timeout: Optional[float] = None) -> AsyncIterator[dict]:
        """Send a request and stream responses."""
        if self.websocket is None:
            await self.connect()
        
        session_id = uuid4().hex
        request = WsRequest(session_id=session_id, action=action, payload=payload)
        q = await self.get_or_update_session(session_id)
        await self.websocket.send(json.dumps(request.model_dump()))
        
        effective_timeout = self._get_effective_timeout(timeout)
        
        while True:
            try:
                if effective_timeout:
                    response: WsResponse = await asyncio.wait_for(q.get(), timeout=effective_timeout)
                else:
                    response: WsResponse = await q.get()
            except asyncio.TimeoutError:
                self.sessions.pop(session_id, None)
                raise TimeoutError(f"Request timed out after {effective_timeout} seconds")
            
            if response.status == WsStatus.STREAMING:
                yield response.data
            elif response.status == WsStatus.SUCCESS:
                self.sessions.pop(session_id)
                break
            elif response.status == WsStatus.ERROR:
                self.sessions.pop(session_id)
                raise Exception(f"WebSocket failed: {response.error}")
            else:
                self.sessions.pop(session_id)
                raise Exception(f"WebSocket failed with unknown status: {response}")
    
    # Implement abstract methods from BaseLivellmClient
    
    async def handle_agent_run(
        self, 
        request: Union[AgentRequest, AgentFallbackRequest],
        timeout: Optional[float] = None
    ) -> AgentResponse:
        """Handle agent run via WebSocket."""
        response = await self.get_response(
            WsAction.AGENT_RUN,
            request.model_dump(),
            timeout=timeout
        )
        return AgentResponse(**response)
    
    async def handle_agent_run_stream(
        self, 
        request: Union[AgentRequest, AgentFallbackRequest],
        timeout: Optional[float] = None
    ) -> AsyncIterator[AgentResponse]:
        """Handle streaming agent run via WebSocket."""
        async for response in self.get_response_stream(WsAction.AGENT_RUN_STREAM, request.model_dump(), timeout=timeout):
            yield AgentResponse(**response)
    
    async def handle_speak(
        self, 
        request: Union[SpeakRequest, AudioFallbackRequest],
        timeout: Optional[float] = None
    ) -> bytes:
        """Handle speak request via WebSocket."""
        response = await self.get_response(
            WsAction.AUDIO_SPEAK,
            request.model_dump(),
            timeout=timeout
        )
        return EncodedSpeakResponse(**response).audio
    
    async def handle_speak_stream(
        self, 
        request: Union[SpeakRequest, AudioFallbackRequest],
        timeout: Optional[float] = None
    ) -> AsyncIterator[bytes]:
        """Handle streaming speak request via WebSocket."""
        async for response in self.get_response_stream(WsAction.AUDIO_SPEAK_STREAM, request.model_dump(), timeout=timeout):
            yield EncodedSpeakResponse(**response).audio
    
    async def handle_transcribe(
        self, 
        request: Union[TranscribeRequest, TranscribeFallbackRequest],
        timeout: Optional[float] = None
    ) -> TranscribeResponse:
        """Handle transcribe request via WebSocket."""
        response = await self.get_response(
            WsAction.AUDIO_TRANSCRIBE,
            request.model_dump(),
            timeout=timeout
        )
        return TranscribeResponse(**response)
    
    # Context manager support
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    @property
    def transcription(self) -> TranscriptionWsClient:
        """
        Lazily-initialized WebSocket transcription client that shares the same
        server base URL and timeout as this realtime client.
        """
        if self._transcription is None:
            # Use the ws root (e.g. ws://host:port) and let TranscriptionWsClient
            # append its own /livellm/ws/transcription path.
            self._transcription = TranscriptionWsClient(
                base_url=self._ws_root_base_url,
                timeout=self.timeout,
            )
        return self._transcription

class LivellmClient(BaseLivellmClient):
    """HTTP-based LiveLLM client for request-response communication."""

    def __init__(
        self, 
        base_url: str,
        user_agent: Optional[str] = None,
        timeout: Optional[float] = None,
        configs: Optional[List[Settings]] = None
        ):
        # Root server URL (http/https, without trailing slash)
        self._root_base_url = base_url.rstrip("/")
        # HTTP API base URL for this client
        self.base_url = f"{self._root_base_url}/livellm"
        self.timeout = timeout
        self.user_agent = user_agent or DEFAULT_USER_AGENT
        # Create client without timeout - we'll pass timeout per-request
        self.client = httpx.AsyncClient(base_url=self.base_url)
        self.settings = []
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": self.user_agent,
        }
        # Lazily-created realtime (WebSocket) client
        self._realtime = None
        if configs:
            self.update_configs_post_init(configs)

    def _get_effective_timeout(self, timeout: Optional[float]) -> Optional[float]:
        """Get effective timeout: per-request timeout overrides default."""
        return timeout if timeout is not None else self.timeout

    @property
    def realtime(self) -> LivellmWsClient:
        """
        Lazily-initialized WebSocket client for realtime operations (agent, audio, etc.)
        that shares the same server base URL and timeout as this HTTP client.

        Example:
            client = LivellmClient(base_url=\"http://localhost:8000\")
            async with client.realtime as session:
                response = await session.agent_run(...)
        """
        if self._realtime is None:
            # Pass the same root base URL; LivellmWsClient will handle ws/wss conversion.
            self._realtime = LivellmWsClient(self._root_base_url, user_agent=self.user_agent, timeout=self.timeout)
        return self._realtime
    
    def update_configs_post_init(self, configs: List[Settings]) -> SuccessResponse:
        """
        Update the configs after the client is initialized.
        Args:
            configs: The configs to update.
        """
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            for config in configs:
                response = client.post(f"{self.base_url}/providers/config", json=config.model_dump())
                response.raise_for_status()
                self.settings.append(config)
            return SuccessResponse(success=True, message="Configs updated successfully")
        

    async def delete(self, endpoint: str, timeout: Optional[float] = None) -> dict:
        """
        Delete a resource from the given endpoint and return the response.
        Args:
            endpoint: The endpoint to delete from.
            timeout: Optional timeout override.
        Returns:
            The response from the endpoint.
        """
        effective_timeout = self._get_effective_timeout(timeout)
        response = await self.client.delete(endpoint, headers=self.headers, timeout=effective_timeout)
        response.raise_for_status()
        return response.json()
    
    async def post_multipart(
        self,
        files: dict,
        data: dict,
        endpoint: str,
        timeout: Optional[float] = None
    ) -> dict:
        """
        Post a multipart request to the given endpoint and return the response.
        Args:
            files: The files to send in the request.
            data: The data to send in the request.
            endpoint: The endpoint to post to.
            timeout: Optional timeout override.
        Returns:
            The response from the endpoint.
        """
        effective_timeout = self._get_effective_timeout(timeout)
        # Don't pass Content-Type header for multipart - httpx will set it automatically
        response = await self.client.post(endpoint, files=files, data=data, timeout=effective_timeout)
        response.raise_for_status()
        return response.json()
    
    
    async def get(
        self,
        endpoint: str,
        timeout: Optional[float] = None
    ) -> dict:
        """
        Get a request from the given endpoint and return the response.
        Args:
            endpoint: The endpoint to get from.
            timeout: Optional timeout override.
        Returns:
            The response from the endpoint.
        """
        effective_timeout = self._get_effective_timeout(timeout)
        response = await self.client.get(endpoint, headers=self.headers, timeout=effective_timeout)
        response.raise_for_status()
        return response.json()
    
    async def post(
        self, 
        json_data: dict, 
        endpoint: str, 
        expect_stream: bool = False, 
        expect_json: bool = True,
        timeout: Optional[float] = None
    ) -> Union[dict, bytes, AsyncIterator[Union[dict, bytes]]]:
        """
        Post a request to the given endpoint and return the response.
        If expect_stream is True, return an AsyncIterator of the response.
        If expect_json is True, return the response as a JSON object.
        Otherwise, return the response as bytes.
        Args:
            json_data: The JSON data to send in the request.
            endpoint: The endpoint to post to.
            expect_stream: Whether to expect a stream response.
            expect_json: Whether to expect a JSON response.
            timeout: Optional timeout override.
        Returns:
            The response from the endpoint.
        Raises:
            Exception: If the response is not 200 or 201.
        """
        effective_timeout = self._get_effective_timeout(timeout)
        response = await self.client.post(endpoint, json=json_data, headers=self.headers, timeout=effective_timeout)
        if response.status_code not in [200, 201]:
            error_response = await response.aread()
            error_response = error_response.decode("utf-8")
            raise Exception(f"Failed to post to {endpoint}: {error_response}")
        if expect_stream:
            async def json_stream_response() -> AsyncIterator[dict]:
                async for chunk in response.aiter_lines():
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    yield json.loads(chunk)
            async def bytes_stream_response() -> AsyncIterator[bytes]:
                async for chunk in response.aiter_bytes():
                    yield chunk
            stream_response = json_stream_response if expect_json else bytes_stream_response
            return stream_response()
        else:
            if expect_json:
                return response.json()
            else:
                return response.content
    
    async def ping(self, timeout: Optional[float] = None) -> SuccessResponse:
        result = await self.get("ping", timeout=timeout)
        return SuccessResponse(**result)
    
    async def update_config(self, config: Settings, timeout: Optional[float] = None) -> SuccessResponse:
        result = await self.post(config.model_dump(), "providers/config", expect_json=True, timeout=timeout)
        self.settings.append(config)
        return SuccessResponse(**result)
    
    async def update_configs(self, configs: List[Settings], timeout: Optional[float] = None) -> SuccessResponse:
        for config in configs:
            await self.update_config(config, timeout=timeout)
        return SuccessResponse(success=True, message="Configs updated successfully")
    
    async def get_configs(self, timeout: Optional[float] = None) -> List[Settings]:
        result = await self.get("providers/configs", timeout=timeout)
        return [Settings(**config) for config in result]
    
    async def delete_config(self, config_uid: str, timeout: Optional[float] = None) -> SuccessResponse:
        result = await self.delete(f"providers/config/{config_uid}", timeout=timeout)
        return SuccessResponse(**result)
    
    async def cleanup(self):
        """
        Delete all the created settings resources and close the client.
        Should be called when you're done using the client.
        """
        for config in self.settings:
            config: Settings = config
            await self.delete_config(config.uid)
        await self.client.aclose()
        # Also close any realtime WebSocket client if it was created
        if self._realtime is not None:
            await self._realtime.disconnect()

    # Implement abstract methods from BaseLivellmClient
    
    async def handle_agent_run(
        self, 
        request: Union[AgentRequest, AgentFallbackRequest],
        timeout: Optional[float] = None
    ) -> AgentResponse:
        """Handle agent run via HTTP."""
        result = await self.post(request.model_dump(), "agent/run", expect_json=True, timeout=timeout)
        return AgentResponse(**result)
    
    async def handle_agent_run_stream(
        self, 
        request: Union[AgentRequest, AgentFallbackRequest],
        timeout: Optional[float] = None
    ) -> AsyncIterator[AgentResponse]:
        """Handle streaming agent run via HTTP."""
        stream = await self.post(request.model_dump(), "agent/run_stream", expect_stream=True, expect_json=True, timeout=timeout)
        async for chunk in stream:
            yield AgentResponse(**chunk)
    
    async def handle_speak(
        self, 
        request: Union[SpeakRequest, AudioFallbackRequest],
        timeout: Optional[float] = None
    ) -> bytes:
        """Handle speak request via HTTP."""
        return await self.post(request.model_dump(), "audio/speak", expect_json=False, timeout=timeout)
    
    async def handle_speak_stream(
        self, 
        request: Union[SpeakRequest, AudioFallbackRequest],
        timeout: Optional[float] = None
    ) -> AsyncIterator[bytes]:
        """Handle streaming speak request via HTTP."""
        speak_stream = await self.post(request.model_dump(), "audio/speak_stream", expect_stream=True, expect_json=False, timeout=timeout)
        async for chunk in speak_stream:
            yield chunk
    
    async def handle_transcribe(
        self, 
        request: Union[TranscribeRequest, TranscribeFallbackRequest],
        timeout: Optional[float] = None
    ) -> TranscribeResponse:
        """Handle transcribe request via HTTP."""
        result = await self.post(request.model_dump(), "audio/transcribe_json", expect_json=True, timeout=timeout)
        return TranscribeResponse(**result)
