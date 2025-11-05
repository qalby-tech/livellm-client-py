"""LiveLLM Client - Python client for the LiveLLM Proxy and Realtime APIs."""
import asyncio
import httpx
import json
import warnings
from typing import List, Optional, AsyncIterator, Union, overload
from .models.common import Settings, SuccessResponse
from .models.agent.agent import AgentRequest, AgentResponse
from .models.audio.speak import SpeakRequest
from .models.audio.transcribe import TranscribeRequest, TranscribeResponse, File
from .models.fallback import AgentFallbackRequest, AudioFallbackRequest, TranscribeFallbackRequest

class LivellmClient:

    def __init__(
        self, 
        base_url: str, 
        timeout: Optional[float] = None,
        configs: Optional[List[Settings]] = None
        ):
        base_url = base_url.rstrip("/")
        self.base_url = f"{base_url}/livellm"
        self.timeout = timeout
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) \
            if self.timeout else httpx.AsyncClient(base_url=self.base_url)
        self.settings = []
        self.headers = {
            "Content-Type": "application/json",
        }
        if configs:
            self.update_configs_post_init(configs)
    

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
        

    async def delete(self, endpoint: str) -> dict:
        """
        Delete a resource from the given endpoint and return the response.
        Args:
            endpoint: The endpoint to delete from.
        Returns:
            The response from the endpoint.
        """
        response = await self.client.delete(endpoint, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    async def post_multipart(
        self,
        files: dict,
        data: dict,
        endpoint: str
    ) -> dict:
        """
        Post a multipart request to the given endpoint and return the response.
        Args:
            files: The files to send in the request.
            data: The data to send in the request.
            endpoint: The endpoint to post to.
        Returns:
            The response from the endpoint.
        """
        # Don't pass Content-Type header for multipart - httpx will set it automatically
        response = await self.client.post(endpoint, files=files, data=data)
        response.raise_for_status()
        return response.json()
    
    
    async def get(
        self,
        endpoint: str
    ) -> dict:
        """
        Get a request from the given endpoint and return the response.
        Args:
            endpoint: The endpoint to get from.
        Returns:
            The response from the endpoint.
        """
        response = await self.client.get(endpoint, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    async def post(
        self, 
        json_data: dict, 
        endpoint: str, 
        expect_stream: bool = False, 
        expect_json: bool = True
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
        Returns:
            The response from the endpoint.
        Raises:
            Exception: If the response is not 200 or 201.
        """
        response = await self.client.post(endpoint, json=json_data, headers=self.headers)
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

    
    async def ping(self) -> SuccessResponse:
        result = await self.get("ping")
        return SuccessResponse(**result)
    
    async def update_config(self, config: Settings) -> SuccessResponse:
        result = await self.post(config.model_dump(), "providers/config", expect_json=True)
        self.settings.append(config)
        return SuccessResponse(**result)
    
    async def update_configs(self, configs: List[Settings]) -> SuccessResponse:
        for config in configs:
            await self.update_config(config)
        return SuccessResponse(success=True, message="Configs updated successfully")
    
    async def get_configs(self) -> List[Settings]:
        result = await self.get("providers/configs")
        return [Settings(**config) for config in result]
    
    async def delete_config(self, config_uid: str) -> SuccessResponse:
        result = await self.delete(f"providers/config/{config_uid}")
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
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    def __del__(self):
        """
        Destructor to clean up resources when the client is garbage collected.
        This will close the HTTP client and attempt to delete configs if cleanup wasn't called.
        Note: It's recommended to use the async context manager or call cleanup() explicitly.
        """
        # Warn user if cleanup wasn't called
        if self.settings:
            warnings.warn(
                "LivellmClient is being garbage collected without explicit cleanup. "
                "Provider configs may not be deleted from the server. "
                "Consider using 'async with' or calling 'await client.cleanup()' explicitly.",
                ResourceWarning,
                stacklevel=2
            )
        
        # Close the httpx client synchronously
        # httpx.AsyncClient stores a sync Transport that needs cleanup
        try:
            with httpx.Client(base_url=self.base_url) as client:
                for config in self.settings:
                    config: Settings = config
                    client.delete("providers/config/{config.uid}", headers=self.headers)
        except Exception:
            # Silently fail - we're in a destructor
            pass

    @overload
    async def agent_run(
        self,
        request: Union[AgentRequest, AgentFallbackRequest],
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
        **kwargs
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
               tools=[]
           )
        
        Args:
            request: An AgentRequest or AgentFallbackRequest object
            provider_uid: The provider UID string
            model: The model to use
            messages: List of messages
            tools: Optional list of tools
            gen_config: Optional generation configuration
            
        Returns:
            AgentResponse with the agent's output
        """
        # Check if first argument is a request object
        if request is not None:
            if not isinstance(request, (AgentRequest, AgentFallbackRequest)):
                raise TypeError(
                    f"First positional argument must be AgentRequest or AgentFallbackRequest, got {type(request)}"
                )
            result = await self.post(request.model_dump(), "agent/run", expect_json=True)
            return AgentResponse(**result)
        
        # Otherwise, use keyword arguments
        if provider_uid is None or model is None or messages is None:
            raise ValueError(
                "provider_uid, model, and messages are required. "
                "Alternatively, pass an AgentRequest object as the first positional argument."
            )
        
        agent_request = AgentRequest(
            provider_uid=provider_uid,
            model=model,
            messages=messages,
            tools=tools or [],
            gen_config=kwargs or None
        )
        result = await self.post(agent_request.model_dump(), "agent/run", expect_json=True)
        return AgentResponse(**result)
    
    @overload
    def agent_run_stream(
        self,
        request: Union[AgentRequest, AgentFallbackRequest],
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
        **kwargs
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
               tools=[]
           ):
               ...
        
        Args:
            request: An AgentRequest or AgentFallbackRequest object
            provider_uid: The provider UID string
            model: The model to use
            messages: List of messages
            tools: Optional list of tools
            gen_config: Optional generation configuration
            
        Returns:
            AsyncIterator of AgentResponse chunks
        """
        # Check if first argument is a request object
        if request is not None:
            if not isinstance(request, (AgentRequest, AgentFallbackRequest)):
                raise TypeError(
                    f"First positional argument must be AgentRequest or AgentFallbackRequest, got {type(request)}"
                )
            stream = await self.post(request.model_dump(), "agent/run_stream", expect_stream=True, expect_json=True)
            async for chunk in stream:
                yield AgentResponse(**chunk)
        else:
            # Otherwise, use keyword arguments
            if provider_uid is None or model is None or messages is None:
                raise ValueError(
                    "provider_uid, model, and messages are required. "
                    "Alternatively, pass an AgentRequest object as the first positional argument."
                )
            
            agent_request = AgentRequest(
                provider_uid=provider_uid,
                model=model,
                messages=messages,
                tools=tools or [],
                gen_config=kwargs or None
            )
            stream = await self.post(agent_request.model_dump(), "agent/run_stream", expect_stream=True, expect_json=True)
            async for chunk in stream:
                yield AgentResponse(**chunk)
    
    @overload
    async def speak(
        self,
        request: Union[SpeakRequest, AudioFallbackRequest],
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
        **kwargs
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
            return await self.post(request.model_dump(), "audio/speak", expect_json=False)
        
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
        return await self.post(speak_request.model_dump(), "audio/speak", expect_json=False)
    
    @overload
    def speak_stream(
        self,
        request: Union[SpeakRequest, AudioFallbackRequest],
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
        **kwargs
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
            speak_stream = await self.post(request.model_dump(), "audio/speak_stream", expect_stream=True, expect_json=False)
            async for chunk in speak_stream:
                yield chunk
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
            speak_stream = await self.post(speak_request.model_dump(), "audio/speak_stream", expect_stream=True, expect_json=False)
            async for chunk in speak_stream:
                yield chunk
    
    @overload
    async def transcribe(
        self,
        request: Union[TranscribeRequest, TranscribeFallbackRequest],
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
        **kwargs
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
            result = await self.post(request.model_dump(), "audio/transcribe_json", expect_json=True)
            return TranscribeResponse(**result)
        
        # Otherwise, use keyword arguments with multipart form-data request
        if provider_uid is None or file is None or model is None:
            raise ValueError(
                "provider_uid, file, and model are required. "
                "Alternatively, pass a TranscribeRequest object as the first positional argument."
            )
        
        files = {
            "file": file
        }
        data = {
            "provider_uid": provider_uid,
            "model": model,
            "language": language,
            "gen_config": json.dumps(kwargs) if kwargs else None
        }
        result = await self.post_multipart(files, data, "audio/transcribe")
        return TranscribeResponse(**result)
         

        

