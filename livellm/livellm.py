"""LiveLLM Client - Python client for the LiveLLM Proxy and Realtime APIs."""
import httpx
import json
from typing import List, Optional, AsyncIterator, Union
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
            async def stream_response() -> AsyncIterator[Union[dict, bytes]]:
                async for chunk in response.aiter_lines():
                    if expect_json:
                        chunk = chunk.strip()
                        if not chunk:
                            continue
                        yield json.loads(chunk)
                    else:
                        yield chunk
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
            await self.delete_config(config.uid)
        await self.client.aclose()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    

    async def agent_run(
        self,
        request: Union[AgentRequest, AgentFallbackRequest]
    ) -> AgentResponse:
        result = await self.post(request.model_dump(), "agent/run", expect_json=True)
        return AgentResponse(**result)
    
    async def agent_run_stream(
        self,
        request: Union[AgentRequest, AgentFallbackRequest]
    ) -> AsyncIterator[AgentResponse]:
        stream = await self.post(request.model_dump(), "agent/run_stream", expect_stream=True, expect_json=True)
        async for chunk in stream:
            yield AgentResponse(**chunk)
    
    async def speak(
        self,
        request: Union[SpeakRequest, AudioFallbackRequest]
    ) -> bytes:
        return await self.post(request.model_dump(), "audio/speak", expect_json=False)
    
    async def speak_stream(
        self,
        request: Union[SpeakRequest, AudioFallbackRequest]
    ) -> AsyncIterator[bytes]:
        return await self.post(request.model_dump(), "audio/speak_stream", expect_stream=True, expect_json=False)
    

    async def transcribe(
        self,
        provider_uid: str,
        file: File,
        model: str,
        language: Optional[str] = None,
        gen_config: Optional[dict] = None
    ) -> TranscribeResponse:
        files = {
            "file": file
        }
        data = {
            "provider_uid": provider_uid,
            "model": model,
            "language": language,
            "gen_config": json.dumps(gen_config) if gen_config else None
        }
        result = await self.post_multipart(files, data, "audio/transcribe")
        return TranscribeResponse(**result)
    
    async def transcribe_json(
        self,
        request: Union[TranscribeRequest, TranscribeFallbackRequest]
    ) -> TranscribeResponse:
        result = await self.post(request.model_dump(), "audio/transcribe_json", expect_json=True)
        return TranscribeResponse(**result)
         

        

