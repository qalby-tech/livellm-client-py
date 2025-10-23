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
from functools import lru_cache

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LivellmProxy:

    def __init__(self, base_url: str, providers: List[ProviderConfig], timeout: float = 30.0):
        """
        Provider-based model resolution with fallback support.
        Fallback works like this:
        on each request, the client will find all providers that support the given model
        and try them in sequence until success.
        in agent case, if some of the messages are not supported by the model,
        the client will transform this exact message and then try the original.
        """
        self.providers = providers
        self.client = LivellmProxyClient(base_url=base_url, timeout=timeout)
    
    @lru_cache(maxsize=128)
    def __get_providers_for_model(self, model: str) -> List[Creds]:
        """
        Find all providers that support the given model.
        Returns a list of credentials for providers that have the model.
        """
        provider_creds = []
        for provider in self.providers:
            for provided_model in provider.models:
                if provided_model.name == model:
                    provider_creds.append(provider.creds)
                    break  # Found the model in this provider, move to next provider
        
        if not provider_creds:
            raise ValueError(f"No providers found that support model: {model}")
        
        return provider_creds
    
    
    @lru_cache(maxsize=128)
    def __get_model_capabilities(self, model: str) -> List[ModelCapability]:
        """
        Get model capabilities from the first provider that supports the model.
        """
        for provider in self.providers:
            for provided_model in provider.models:
                if provided_model.name == model:
                    return provided_model.capabilities
        
        raise ValueError(f"No providers found that support model: {model}")
    
    async def __run_with_fallback(self, executable: Callable[Creds, Any], model: str, stream: bool = False) -> Any:
        creds = self.__get_providers_for_model(model)
        errors = []
        for cred in creds:
            try:
                if stream:
                    return executable(cred)
                else:
                    return await executable(cred)
            except Exception as e:
                logger.error(f"Error running executable with creds {cred}: {e}")
                errors.append((model, str(e)))
                continue
        errors_str = "\n".join([f"\nModel: {model},\nError: {error}" for model, error in errors])
        raise ValueError(f"After all fallback attempts, the executable failed: Errors: {errors_str}")
    
    async def __agent_run_with_fallback(self, request: AgentRequest) -> AgentResponse:
        def exec_agent(cred: Creds) -> Coroutine[Any, Any, AgentResponse]:
            return self.client.agent_run(request, cred.api_key, cred.provider, cred.base_url)
        return await self.__run_with_fallback(exec_agent, request.model)
    
    async def __agent_run_stream_with_fallback(self, request: AgentRequest) -> AsyncIterator[AgentResponse]:
        def exec_agent_stream(cred: Creds) -> AsyncIterator[AgentResponse]:
            return self.client.agent_run_stream(request, cred.api_key, cred.provider, cred.base_url)
        response: AsyncIterator[AgentResponse] = await self.__run_with_fallback(exec_agent_stream, request.model, stream=True)
        async for chunk in response:
            yield chunk

    async def __audio_speak_with_fallback(self, request: SpeakRequest) -> bytes:
        def exec_speak(cred: Creds) -> Coroutine[Any, Any, bytes]:
            return self.client.audio_speak(request, cred.api_key, cred.provider, cred.base_url)
        return await self.__run_with_fallback(exec_speak, request.model)
    
    async def __audio_speak_stream_with_fallback(self, request: SpeakRequest) -> AsyncIterator[bytes]:
        def exec_speak_stream(cred: Creds) -> AsyncIterator[bytes]:
            return self.client.audio_speak_stream(request, cred.api_key, cred.provider, cred.base_url)
        response: AsyncIterator[bytes] = await self.__run_with_fallback(exec_speak_stream, request.model, stream=True)
        async for chunk in response:
            yield chunk
    
    async def __audio_transcribe_with_fallback(
        self, 
        model: str,
        file: FileType,
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
        return await self.__run_with_fallback(exec_transcribe, model)

    
    async def __run_agent_as_binary_transformer(
        self, 
        binary_message: BinaryMessage, 
        model: str, 
        system_prompt: str
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
        response = await self.__agent_run_with_fallback(agent_request)
        return response.output
    
    async def __run_binary_transformer_with_fallback(self, binary_message: BinaryMessage, models: List[Model], system_prompt: str) -> str:
        errors = []
        for model in models:
            try:
                return await self.__run_agent_as_binary_transformer(binary_message, model.name, system_prompt)
            except Exception as e:
                logger.error(f"Error running binary transformer with model {model.name}: {e}")
                errors.append((model.name, str(e)))
                continue
        errors_str = "\n".join([f"\nModel: {model},\nError: {error}" for model, error in errors])
        raise ValueError(f"After all fallback attempts, the binary transformer failed: Errors: {errors_str}")
    
    async def __audio_to_text(self, binary_message, models: List[Model]) -> str:
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
        return await self.__run_binary_transformer_with_fallback(binary_message, models, system)

    async def __image_to_text(self, binary_message, models: List[Model]) -> str:
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
        return await self.__run_binary_transformer_with_fallback(binary_message, models, system)
    
    async def __video_to_text(self, binary_message, models: List[Model]) -> str:
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
        return await self.__run_binary_transformer_with_fallback(binary_message, models, system)
    


    async def __find_models_with_capability(self, capability: ModelCapability) -> List[Model]:
        """
        Find all models with the specified capability.
        """
        models = []
        for provider in self.providers:
            for provided_model in provider.models:
                if capability in provided_model.capabilities:
                    models.append(provided_model)
        if not models:
            raise ValueError(f"No model with capability {capability} found")
        return models

    async def __binary_to_text(self, binary_message: BinaryMessage, primary_model: Model) -> str:
        """
        Transform a binary message to a text message using model using fallback providers.
        """
        # use mime_type to determine the type of the binary message
        if "image" in binary_message.mime_type:
            models = []
            if ModelCapability.IMAGE_AGENT in primary_model.capabilities:
                models.append(primary_model)
            models += await self.__find_models_with_capability(ModelCapability.IMAGE_AGENT)
            return await self.__image_to_text(binary_message, models)
        elif "video" in binary_message.mime_type:
            models = []
            if ModelCapability.VIDEO_AGENT in primary_model.capabilities:
                models.append(primary_model)
            models += await self.__find_models_with_capability(ModelCapability.VIDEO_AGENT)
            return await self.__video_to_text(binary_message, models)
        elif "audio" in binary_message.mime_type:
            models = []
            if ModelCapability.AUDIO_AGENT in primary_model.capabilities:
                models.append(primary_model)
            models += await self.__find_models_with_capability(ModelCapability.AUDIO_AGENT)
            return await self.__audio_to_text(binary_message, models)
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

    
    def __required_capabilities(self, messages: List[Union[TextMessage, BinaryMessage]]) -> List[ModelCapability]:
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
        force_binary_transformation: bool = False
    ) -> List[Union[TextMessage, BinaryMessage]]:
        """
        Preprocess messages to ensure they are compatible with the model.
        
        Args:
            messages: List of messages to preprocess
            model: Model name
            force_binary_transformation: If True, always transform binary messages to text
            
        Returns:
            Preprocessed messages (either original or transformed)
        """
        # Get model capabilities from the first provider that supports this model
        model_capabilities = self.__get_model_capabilities(model)
        model_obj = Model(name=model, capabilities=model_capabilities)
        
        if force_binary_transformation:
            return await self.__binaries_to_text(messages, model_obj)
        
        required_capabilities = self.__required_capabilities(messages)
        if not all(capability in model_obj.capabilities for capability in required_capabilities):
            return await self.__binaries_to_text(messages, model_obj)
        
        return messages
    
    
    async def agent_run(
        self,
        model: str,
        messages: List[Union[TextMessage, BinaryMessage]],
        tools: List[Union[WebSearchInput, MCPStreamableServerInput]],
        force_binary_transformation: bool = False,
        **gen_config
    ) -> Tuple[AgentResponse, List[Union[TextMessage, BinaryMessage]]]:
        """
        Run an agent with the specified configuration.
        if model does not support some messages, the client will transform them to be compatible with 
        the model e.g. to text
        and then try the original message again.
        if you want to safe transformed messages
        if force_binary_transformation is True, the client will always transform all bindary messages to text messages.
        Model capabilities are automatically discovered from the first provider that supports the model.
        """
        messages = await self.__preprocess_messages(
            messages, model, force_binary_transformation
        )
        
        agent_request = AgentRequest(
            model=model,
            messages=messages,
            tools=tools,
            gen_config=gen_config,
        )

        response = await self.__agent_run_with_fallback(agent_request)
        return response, messages
    
    async def agent_run_stream(
        self,
        model: str,
        messages: List[Union[TextMessage, BinaryMessage]],
        tools: List[Union[WebSearchInput, MCPStreamableServerInput]],
        force_binary_transformation: bool = False,
        **gen_config
    ) -> Tuple[AsyncIterator[AgentResponse], List[Union[TextMessage, BinaryMessage]]]:
        """
        Stream agent responses with the specified configuration.
        if model does not support some messages, the client will transform them to be compatible with 
        the model e.g. to text
        and then try the original message again.
        if you want to safe transformed messages
        if force_binary_transformation is True, the client will always transform all bindary messages to text messages.
        Model capabilities are automatically discovered from the first provider that supports the model.
        """
        messages = await self.__preprocess_messages(
            messages, model, force_binary_transformation
        )
        
        agent_request = AgentRequest(
            model=model,
            messages=messages,
            tools=tools,
            gen_config=gen_config,
        )

        response = self.__agent_run_stream_with_fallback(agent_request)
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
        return await self.__audio_speak_with_fallback(request)
    
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
        async for chunk in self.__audio_speak_stream_with_fallback(request):
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
            language=language,
            gen_config=gen_config if gen_config else None
        )
