from livellm.models.transcription import (
    TranscriptionInitWsRequest,
    TranscriptionInitWsResponse,
    TranscriptionAudioChunkWsRequest, 
    TranscriptionWsResponse)
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
    ) -> AsyncIterator[list[TranscriptionWsResponse]]:
        """
        Start a transcription session.
        
        Args:
            request: The initialization request for the transcription session.
            source: An async iterator that yields audio chunks to transcribe.
            
        Returns:
            An async iterator that yields lists of transcription responses.
            Each list contains all responses that accumulated since the last yield,
            ordered from oldest to newest (last element is the most recent).
            This prevents slow processing from stalling the entire loop.
            
        Example:
            ```python
            async def audio_source():
                with open("audio.pcm", "rb") as f:
                    while chunk := f.read(4096):
                        yield TranscriptionAudioChunkWsRequest(audio=chunk)
            
            async with TranscriptionWsClient(url) as client:
                async for responses in client.start_session(init_request, audio_source()):
                    # responses is a list, newest transcription is last
                    latest = responses[-1]
                    print(f"Latest: {latest.transcription}")
                    
                    # Process all transcriptions if needed
                    for resp in responses:
                        print(resp.transcription)
            ```
        """
        # Send initialization request as JSON
        await self.websocket.send(request.model_dump_json())
        
        # Wait for initialization response
        response_data = await self.websocket.recv()
        init_response = TranscriptionInitWsResponse(**json.loads(response_data))
        if not init_response.success:
            raise Exception(f"Failed to start transcription session: {init_response.error}")

        # Queue to collect incoming transcription responses
        response_queue: asyncio.Queue[TranscriptionWsResponse | None] = asyncio.Queue()
        receiver_done = False

        # Start sending audio chunks in background
        async def send_chunks():
            try:
                async for chunk in source:
                    await self.websocket.send(chunk.model_dump_json())
            except websockets.ConnectionClosed:
                # Connection closed, stop sending
                pass
            except Exception as e:
                # If there's an error sending chunks, close the websocket
                print(f"Error sending chunks: {e}")
                await self.websocket.close()
                raise e
        
        # Receive transcription responses in background
        async def receive_responses():
            nonlocal receiver_done
            try:
                while True:
                    try:
                        response_data = await self.websocket.recv()
                        transcription_response = TranscriptionWsResponse(**json.loads(response_data))
                        await response_queue.put(transcription_response)
                    except websockets.ConnectionClosed:
                        break
            finally:
                receiver_done = True
                await response_queue.put(None)  # Signal end of stream
        
        send_task = asyncio.create_task(send_chunks())
        receive_task = asyncio.create_task(receive_responses())
        
        try:
            while True:                
                # Wait for at least one response
                first_response = await response_queue.get()
                if first_response is None:
                    # End of stream
                    break
                
                # Collect all additional responses that have accumulated (non-blocking)
                responses = [first_response]
                while True:
                    try:
                        additional = response_queue.get_nowait()
                        if additional is None:
                            # End of stream, yield what we have and exit
                            yield responses
                            return
                        responses.append(additional)
                    except asyncio.QueueEmpty:
                        break
                
                yield responses
        finally:
            # Cancel tasks if still running
            for task in [send_task, receive_task]:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass