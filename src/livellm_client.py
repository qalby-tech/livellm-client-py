"""High-level client for LiveLLM Proxy with fallback and transformation support."""

from typing import Optional, List, Union, Tuple, AsyncIterator, Dict, Any, Callable, Coroutine
import logging

from .raw_client import LivellmProxyClient
from .models import (
    TextMessage,
    BinaryMessage,
    MessageRole,
    WebSearchInput,
    MCPStreamableServerInput,
    AgentRequest,
    AgentResponse,
    SpeakRequest,
    TranscribeResponse,
    Creds,
    ModelCapability,
    Model,
    ProviderConfig,
    FileType,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LivellmProxy:

    def __init__(self, base_url: str, primary_creds: Creds, providers: List[ProviderConfig]):
        """
        a list of creds is used to handle fallback
        fallback works like this:
        on each request, the client will heuristically check if model can actually run given the input
        if it can't, the client will try the next cred in the list
        in agent case, if some of the messages are not supported by th model,
        the client will transform this exact message and then try the original.
        """
        self.primary_creds = primary_creds
        self.providers = providers
        self.client = LivellmProxyClient(base_url=base_url)
    

    async def __get_fallback_creds(self, model: str, primary_creds: Creds) -> List[Creds]:
        fallback_creds = []
        for provider in self.providers:
            for provided_model in provider.models:
                if provided_model.name == model:
                    fallback_creds.append(provider.creds)
        return [primary_creds] + fallback_creds
    
    async def __run_with_fallback(self, executable: Callable[Creds, Any], model: str, primary_creds: Creds, stream: bool = False) -> Any:
        creds = await self.__get_fallback_creds(model, primary_creds)
        for cred in creds:
            try:
                if stream:
                    return executable(cred)
                else:
                    return await executable(cred)
            except Exception as e:
                logger.error(f"Error running executable with creds {cred}: {e}")
                continue
        raise ValueError(f"No model with name {model} found")
    
    async def __agent_run_with_fallback(self, request: AgentRequest, primary_creds: Creds) -> AgentResponse:
        def exec_agent(cred: Creds) -> Coroutine[Any, Any, AgentResponse]:
            return self.client.agent_run(request, cred.api_key, cred.provider, cred.base_url)
        return await self.__run_with_fallback(exec_agent, request.model, primary_creds)
    
    async def __agent_run_stream_with_fallback(self, request: AgentRequest, primary_creds: Creds) -> AsyncIterator[AgentResponse]:
        def exec_agent_stream(cred: Creds) -> AsyncIterator[AgentResponse]:
            return self.client.agent_run_stream(request, cred.api_key, cred.provider, cred.base_url)
        response: AsyncIterator[AgentResponse] = await self.__run_with_fallback(exec_agent_stream, request.model, primary_creds, stream=True)
        async for chunk in response:
            yield chunk

    async def __audio_speak_with_fallback(self, request: SpeakRequest, primary_creds: Creds) -> bytes:
        def exec_speak(cred: Creds) -> Coroutine[Any, Any, bytes]:
            return self.client.audio_speak(request, cred.api_key, cred.provider, cred.base_url)
        return await self.__run_with_fallback(exec_speak, request.model, primary_creds)
    
    async def __audio_speak_stream_with_fallback(self, request: SpeakRequest, primary_creds: Creds) -> AsyncIterator[bytes]:
        def exec_speak_stream(cred: Creds) -> AsyncIterator[bytes]:
            return self.client.audio_speak_stream(request, cred.api_key, cred.provider, cred.base_url)
        response: AsyncIterator[bytes] = await self.__run_with_fallback(exec_speak_stream, request.model, primary_creds, stream=True)
        async for chunk in response:
            yield chunk
    
    async def __audio_transcribe_with_fallback(
        self, 
        model: str,
        file: FileType,
        primary_creds: Creds,
        language: Optional[str] = None,
        gen_config: Optional[Dict[str, Any]] = None
    ) -> TranscribeResponse:
        def exec_transcribe(cred: Creds) -> Coroutine[Any, Any, TranscribeResponse]:
            return self.client.audio_transcribe(
                model=model,
                file=file,
                api_key=cred.api_key,
                provider=cred.provider,
                language=language,
                gen_config=gen_config,
                base_url=cred.base_url
            )
        return await self.__run_with_fallback(exec_transcribe, model, primary_creds)

    
    async def __run_agent_as_binary_transformer(
        self, 
        binary_message: BinaryMessage, 
        model: str, 
        system_prompt: str,
        creds: Creds
    ) -> str:
        """
        Run an agent as a binary transformer.
        """
        messages = [
            TextMessage(role=MessageRole.SYSTEM, content=system_prompt),
            binary_message,
        ]
        agent_request = AgentRequest(
            model=model,
            messages=messages,
            tools=[],
            gen_config={
                "temperature": 0.0
            },
        )
        response = await self.__agent_run_with_fallback(agent_request, creds)
        return response.output
    
    async def __audio_to_text(self, binary_message, model: str, creds: Creds) -> str:
        system = """
        You will act as ASR.
        You will be given an audio file and you will need to transcribe it to text.
        Transcribe the audio in a language that is most likely to be the language of the audio.
        Return ONLY the text of the transcription. Nothing more
        Return result like this:
        <audio_transcription>
        [text of the transcription]
        </audio_transcription>
        """
        return await self.__run_agent_as_binary_transformer(binary_message, model, system, creds)

    async def __image_to_text(self, binary_message, model: str, creds: Creds) -> str:
        system = """
        You will act as OCR.
        You will be given an image file and you will need to fully describe the image in detail.
        Return ONLY description of the image. Nothing more
        The description should be in a language that is most likely to be the language of the image.
        Return result like this:
        <image_description>
        [description of the image]
        </image_description>
        """
        return await self.__run_agent_as_binary_transformer(binary_message, model, system, creds)
    
    async def __video_to_text(self, binary_message, model: str, creds: Creds) -> str:
        system = """
        You will act as VSR.
        You will be given a video file and you will need to fully describe the video in detail.
        Return ONLY description of the video. Nothing more
        The description should be in a language that is most likely to be the language of the video.
        Return result like this:
        <video_description>
        [description of the video]
        </video_description>
        """
        return await self.__run_agent_as_binary_transformer(binary_message, model, system, creds)
    


    async def __find_model_with_capability(self, capability: ModelCapability) -> Tuple[Model, Creds]:
        """
        Find a model with the specified capability.
        """
        for provider in self.providers:
            for provided_model in provider.models:
                if capability in provided_model.capabilities:
                    return provided_model, provider.creds
        raise ValueError(f"No model with capability {capability} found")

    async def __binary_to_text(self, binary_message: BinaryMessage, primary_model: Model) -> str:
        """
        Transform a binary message to a text message using model using fallback providers.
        """
        # use mime_type to determine the type of the binary message
        if "image" in binary_message.mime_type:
            if ModelCapability.IMAGE_AGENT in primary_model.capabilities:
                model, creds = primary_model, self.primary_creds
            else:
                model, creds = await self.__find_model_with_capability(ModelCapability.IMAGE_AGENT)
            return await self.__image_to_text(binary_message, model.name, creds)
        elif "video" in binary_message.mime_type:
            if ModelCapability.VIDEO_AGENT in primary_model.capabilities:
                model, creds = primary_model, self.primary_creds
            else:
                model, creds = await self.__find_model_with_capability(ModelCapability.VIDEO_AGENT)
            return await self.__video_to_text(binary_message, model.name, creds)
        elif "audio" in binary_message.mime_type:
            if ModelCapability.AUDIO_AGENT in primary_model.capabilities:
                model, creds = primary_model, self.primary_creds
            else:
                model, creds = await self.__find_model_with_capability(ModelCapability.AUDIO_AGENT)
            return await self.__audio_to_text(binary_message, model.name, creds)
        else:
            raise ValueError(f"Unsupported mime type: {binary_message.mime_type}")
        
    async def __binaries_to_text(self, messages: List[Union[TextMessage, BinaryMessage]], primary_model: Model) -> List[TextMessage]:
        """
        Transform a message to a text message using model using fallback providers.
        """
        __messages = []
        for message in messages:
            if isinstance(message, BinaryMessage):
                text_content = await self.__binary_to_text(message, primary_model)
                __messages.append(TextMessage(role=message.role, content=text_content))
            else:
                __messages.append(message)
        return __messages

    
    async def __required_capabilities(self, messages: List[Union[TextMessage, BinaryMessage]]) -> List[ModelCapability]:
        """
        Find the required capabilities for the messages.
        """
        required_capabilities = []
        for message in messages:
            if isinstance(message, BinaryMessage):
                if "image" in message.mime_type:
                    required_capabilities.append(ModelCapability.IMAGE_AGENT)
                elif "video" in message.mime_type:
                    required_capabilities.append(ModelCapability.VIDEO_AGENT)
                elif "audio" in message.mime_type:
                    required_capabilities.append(ModelCapability.AUDIO_AGENT)
                else:
                    raise ValueError(f"Unsupported mime type: {message.mime_type}")
        return required_capabilities
    
    async def __preprocess_messages(
        self,
        messages: List[Union[TextMessage, BinaryMessage]],
        model: str,
        force_binary_transformation: bool = False,
        model_capabilities: Optional[List[ModelCapability]] = None
    ) -> List[Union[TextMessage, BinaryMessage]]:
        """
        Preprocess messages to ensure they are compatible with the model.
        
        Args:
            messages: List of messages to preprocess
            model: Model name
            force_binary_transformation: If True, always transform binary messages to text
            model_capabilities: List of model capabilities (if not set, counts as no capabilities)
            
        Returns:
            Preprocessed messages (either original or transformed)
        """
        model_obj = Model(name=model, capabilities=model_capabilities or [])
        
        if force_binary_transformation:
            return await self.__binaries_to_text(messages, model_obj)
        
        required_capabilities = await self.__required_capabilities(messages)
        if not all(capability in model_obj.capabilities for capability in required_capabilities):
            return await self.__binaries_to_text(messages, model_obj)
        
        return messages
    
    
    async def agent_run(
        self,
        model: str,
        messages: List[Union[TextMessage, BinaryMessage]],
        tools: List[Union[WebSearchInput, MCPStreamableServerInput]],
        force_binary_transformation: bool = False,
        model_capabilities: Optional[List[ModelCapability]] = None,
        **gen_config
    ) -> Tuple[AgentResponse, List[Union[TextMessage, BinaryMessage]]]:
        """
        Run an agent with the specified configuration.
        if model does not support some messages, the client will transform them to be compatible with 
        the model e.g. to text
        and then try the original message again.
        if you want to safe transformed messages
        if force_binary_transformation is True, the client will always transform all bindary messages to text messages.
        if model capabilities are not set, it will count as no capabilities
        """
        messages = await self.__preprocess_messages(
            messages, model, force_binary_transformation, model_capabilities
        )
        
        agent_request = AgentRequest(
            model=model,
            messages=messages,
            tools=tools,
            gen_config=gen_config,
        )

        response = await self.__agent_run_with_fallback(
            agent_request, self.primary_creds
        )
        return response, messages
    
    async def agent_run_stream(
        self,
        model: str,
        messages: List[Union[TextMessage, BinaryMessage]],
        tools: List[Union[WebSearchInput, MCPStreamableServerInput]],
        force_binary_transformation: bool = False,
        model_capabilities: Optional[List[ModelCapability]] = None,
        **gen_config
    ) -> Tuple[AsyncIterator[AgentResponse], List[Union[TextMessage, BinaryMessage]]]:
        """
        Stream agent responses with the specified configuration.
        if model does not support some messages, the client will transform them to be compatible with 
        the model e.g. to text
        and then try the original message again.
        if you want to safe transformed messages
        if force_binary_transformation is True, the client will always transform all bindary messages to text messages.
        if model capabilities are not set, it will count as no capabilities
        """
        messages = await self.__preprocess_messages(
            messages, model, force_binary_transformation, model_capabilities
        )
        
        agent_request = AgentRequest(
            model=model,
            messages=messages,
            tools=tools,
            gen_config=gen_config,
        )

        response = self.__agent_run_stream_with_fallback(
            agent_request, self.primary_creds
        )
        return response, messages
    
    async def audio_speak(
        self,
        model: str,
        text: str,
        voice: str,
        output_format: str,
        **gen_config
    ) -> bytes:
        """
        Convert text to speech using the specified audio provider with automatic fallback.
        
        Args:
            model: Model to use for TTS
            text: Text to convert to speech
            voice: Voice to use
            output_format: Output audio format
            **gen_config: Additional generation configuration
            
        Returns:
            Audio bytes
        """
        request = SpeakRequest(
            model=model,
            text=text,
            voice=voice,
            output_format=output_format,
            gen_config=gen_config if gen_config else None,
        )
        return await self.__audio_speak_with_fallback(request, self.primary_creds)
    
    async def audio_speak_stream(
        self,
        model: str,
        text: str,
        voice: str,
        output_format: str,
        **gen_config
    ) -> AsyncIterator[bytes]:
        """
        Stream text to speech audio using the specified audio provider with automatic fallback.
        
        Args:
            model: Model to use for TTS
            text: Text to convert to speech
            voice: Voice to use
            output_format: Output audio format
            **gen_config: Additional generation configuration
            
        Yields:
            Audio bytes chunks
        """
        request = SpeakRequest(
            model=model,
            text=text,
            voice=voice,
            output_format=output_format,
            gen_config=gen_config if gen_config else None,
        )
        async for chunk in self.__audio_speak_stream_with_fallback(request, self.primary_creds):
            yield chunk
    
    async def audio_transcribe(
        self,
        model: str,
        file: FileType,
        language: Optional[str] = None,
        **gen_config
    ) -> TranscribeResponse:
        """
        Transcribe audio to text using the specified audio provider with automatic fallback.
        
        Args:
            model: Model to use for transcription
            file: Audio file as bytes or tuple of (filename, bytes)
            language: Optional language hint
            **gen_config: Additional generation configuration
            
        Returns:
            TranscribeResponse with transcribed text and language
        """
        return await self.__audio_transcribe_with_fallback(
            model=model,
            file=file,
            primary_creds=self.primary_creds,
            language=language,
            gen_config=gen_config if gen_config else None
        )
