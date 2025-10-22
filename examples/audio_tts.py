"""
Audio Text-to-Speech (TTS) Examples

This example demonstrates how to use the LivellmProxy client for text-to-speech generation,
both in regular and streaming modes.
"""

import asyncio
from pathlib import Path
from livellm import (
    LivellmProxy,
    Creds,
    create_openai_provider_config,
)


async def test_regular_tts():
    """Test regular text-to-speech generation."""
    print("=== Regular TTS ===")
    
    # Configuration - Update these with your actual credentials
    BASE_URL = "http://localhost:8000"  # Your proxy server URL
    API_KEY = "your-api-key-here"  # Your API key
    OPENAI_BASE_URL = "https://api.openai.com/v1"  # OpenAI API URL
    
    # Initialize the proxy client
    proxy = LivellmProxy(
        base_url=BASE_URL,
        primary_creds=Creds(
            api_key=API_KEY,
            provider="openai",
            base_url=OPENAI_BASE_URL
        ),
        providers=[
            create_openai_provider_config(API_KEY, base_url=OPENAI_BASE_URL),
        ]
    )
    
    # Generate speech
    audio_bytes = await proxy.audio_speak(
        model="tts-1",
        text="Hello, this is a test of text to speech.",
        voice="alloy",
        output_format="mp3",
    )
    
    print(f"Audio bytes received: {len(audio_bytes)} bytes")
    
    # Save audio file
    output_path = Path(__file__).parent / "output_speech.mp3"
    with open(output_path, "wb") as f:
        f.write(audio_bytes)
    print(f"Audio saved to: {output_path}")


async def test_streaming_tts():
    """Test streaming text-to-speech generation."""
    print("\n=== Streaming TTS ===")
    
    # Configuration - Update these with your actual credentials
    BASE_URL = "http://localhost:8000"  # Your proxy server URL
    API_KEY = "your-api-key-here"  # Your API key
    OPENAI_BASE_URL = "https://api.openai.com/v1"  # OpenAI API URL
    
    # Initialize the proxy client
    proxy = LivellmProxy(
        base_url=BASE_URL,
        primary_creds=Creds(
            api_key=API_KEY,
            provider="openai",
            base_url=OPENAI_BASE_URL
        ),
        providers=[
            create_openai_provider_config(API_KEY, base_url=OPENAI_BASE_URL),
        ]
    )
    
    # Generate streaming speech
    output_path = Path(__file__).parent / "output_speech_stream.mp3"
    total_bytes = 0
    
    with open(output_path, "wb") as f:
        async for chunk in proxy.audio_speak_stream(
            model="tts-1",
            text="This is a streaming test of text to speech. It should come in chunks.",
            voice="alloy",
            output_format="mp3",
        ):
            f.write(chunk)
            total_bytes += len(chunk)
            print(f"Received chunk: {len(chunk)} bytes", flush=True)
    
    print(f"Total audio bytes received: {total_bytes} bytes")
    print(f"Audio saved to: {output_path}")


async def main():
    """Run both TTS examples."""
    await test_regular_tts()
    await test_streaming_tts()


if __name__ == "__main__":
    asyncio.run(main())