"""
Audio Transcription Example

This example demonstrates how to use the LivellmProxy client for audio transcription.
"""

import asyncio
from pathlib import Path
from livellm import (
    LivellmProxy,
    Creds,
    create_openai_provider_config,
)


def get_test_audio_path() -> Path:
    """Get path to test audio file."""
    # Look for test audio in the examples/assets directory
    examples_dir = Path(__file__).parent
    return examples_dir / "assets/test_audio.mp3"


async def main():
    """Run audio transcription example."""
    # Configuration - Update these with your actual credentials
    BASE_URL = "http://localhost:8000"  # Your proxy server URL
    API_KEY = "your-api-key-here"  # Your API key
    OPENAI_BASE_URL = "https://api.openai.com/v1"  # OpenAI API URL
    
    # Check if test audio file exists
    audio_path = get_test_audio_path()
    if not audio_path.exists():
        print(f"WARNING: Test audio file not found at {audio_path}")
        print("Please place a test audio file at examples/assets/test_audio.mp3")
        return
    
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
    
    # Read audio file
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    
    # Transcribe audio
    response = await proxy.audio_transcribe(
        model="whisper-1",
        file=("test_audio.mp3", audio_bytes, "audio/mp3"),
    )
    
    print(f"Transcription: {response.text}")
    print(f"Language: {response.language}")
    
    # Transcribe with language hint
    response2 = await proxy.audio_transcribe(
        model="whisper-1",
        file=("test_audio.mp3", audio_bytes, "audio/mp3"),
        language="en",
    )
    
    print(f"Transcription (with language hint): {response2.text}")


if __name__ == "__main__":
    asyncio.run(main())