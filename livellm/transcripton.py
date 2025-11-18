from livellm.models.transcription import (
    TranscriptionInitWsRequest, 
    TranscriptionAudioChunkWsRequest, 
    TranscriptionWsResponse)
from livellm.models.ws import WsResponse, WsStatus
from typing import Optional, AsyncIterator
import websockets
import asyncio
import json


class TranscriptionWsClient:
    def __init__(self, base_url: str, timeout: Optional[float] = None, max_size: Optional[int] = None):
        self.base_url = base_url.rstrip("/")
        self.url = f"{base_url}/livellm/ws/transcription"
        self.timeout = timeout
        self.websocket = None
        self.max_size = max_size or 1024 * 1024 * 10 # 10MB is default max size
    
    async def connect(self):
        """
        Connect to the transcription websocket server.
        """
        self.websocket = await websockets.connect(
            self.url,
            open_timeout=self.timeout,
            close_timeout=self.timeout,
            max_size=self.max_size
        )
    
    async def disconnect(self):
        """
        Disconnect from the transcription websocket server.
        """
        if self.websocket is not None:
            await self.websocket.close()
            self.websocket = None
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
    
    async def start_session(
        self, 
        request: TranscriptionInitWsRequest, 
        source: AsyncIterator[TranscriptionAudioChunkWsRequest]
    ) -> AsyncIterator[TranscriptionWsResponse]:
        """
        Start a transcription session.
        
        Args:
            request: The initialization request for the transcription session.
            source: An async iterator that yields audio chunks to transcribe.
            
        Returns:
            An async iterator of transcription session responses.
            
        Example:
            ```python
            async def audio_source():
                with open("audio.pcm", "rb") as f:
                    while chunk := f.read(4096):
                        yield TranscriptionAudioChunkWsRequest(audio=chunk)
            
            async with TranscriptionWsClient(url) as client:
                async for response in client.start_session(init_request, audio_source()):
                    print(response.transcription)
                    if response.is_end:
                        break
            ```
        """
        # Send initialization request
        await self.websocket.send(request.model_dump_json())
        
        # Wait for initialization response
        response_data = await self.websocket.recv()
        response = WsResponse(**json.loads(response_data))
        if response.status == WsStatus.ERROR:
            raise Exception(f"Failed to start transcription session: {response.error}")

        # Start sending audio chunks in background
        async def send_chunks():
            try:
                async for chunk in source:
                    await self.websocket.send(chunk.model_dump_json())
            except Exception as e:
                # If there's an error sending chunks, close the websocket
                print(f"Error sending chunks: {e}")
                await self.websocket.close()
                raise e
        
        send_task = asyncio.create_task(send_chunks())
        
        # Receive transcription responses
        try:
            while not send_task.done():
                response_data = await self.websocket.recv()
                transcription_response = TranscriptionWsResponse(**json.loads(response_data))
                yield transcription_response
                
                # Stop if we received the final transcription
                if transcription_response.is_end:
                    break
        except websockets.ConnectionClosed:
            pass
        finally:
            # Cancel the send task if still running
            if not send_task.done():
                send_task.cancel()
                try:
                    await send_task
                except asyncio.CancelledError:
                    pass