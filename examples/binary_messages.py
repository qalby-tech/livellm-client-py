"""
Binary Messages Examples

This example demonstrates how to use the LivellmProxy client with binary messages
(audio and images) and how they are automatically transformed when the model
doesn't support them.
"""

import asyncio
from pathlib import Path
from livellm import (
    LivellmProxy,
    Creds,
    TextMessage,
    BinaryMessage,
    create_google_provider_config,
)


def get_test_audio_path() -> Path:
    """Get path to test audio file."""
    # Look for test audio in the examples/assets directory
    examples_dir = Path(__file__).parent
    return examples_dir / "assets/test_audio.mp3"


async def test_audio_binary_unsupported():
    """Test sending audio as binary message to agent that doesn't support it."""
    print("=== Audio Binary Message (Unsupported Model) ===")
    
    # Configuration - Update these with your actual credentials
    BASE_URL = "http://localhost:8000"  # Your proxy server URL
    API_KEY = "your-api-key-here"  # Your API key
    GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com"  # Google API URL
    
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
            base_url="https://api.openai.com/v1"
        ),
        providers=[
            create_google_provider_config(API_KEY, base_url=GOOGLE_BASE_URL),
        ]
    )
    
    # Read audio file
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    
    # Create binary message with audio
    binary_message = BinaryMessage.from_bytes(
        content=audio_bytes,
        mime_type="audio/mp3",
        caption="What is in this audio?"
    )
    
    # Send to model without audio capabilities
    # The client should automatically transform audio to text
    response, messages = await proxy.agent_run(
        model="gpt-4o",  # This model doesn't support audio input
        messages=[
            binary_message,
            TextMessage(role="user", content="Based on the audio, what did you hear?"),
        ],
        tools=[],
        model_capabilities=[],  # No audio capability
    )
    
    print(f"Response: {response.output}")
    print(f"Transformed messages count: {len(messages)}")
    
    # Check that binary message was transformed to text
    for i, msg in enumerate(messages):
        print(f"Message {i}: {type(msg).__name__}")
        if isinstance(msg, TextMessage):
            print(f"  Content preview: {msg.content[:100]}...")


async def test_audio_binary_force_transform():
    """Test forcing binary message transformation even if model supports it."""
    print("\n=== Audio Binary Message (Force Transform) ===")
    
    # Configuration - Update these with your actual credentials
    BASE_URL = "http://localhost:8000"  # Your proxy server URL
    API_KEY = "your-api-key-here"  # Your API key
    GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com"  # Google API URL
    
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
            base_url="https://api.openai.com/v1"
        ),
        providers=[
            create_google_provider_config(API_KEY, base_url=GOOGLE_BASE_URL),
        ]
    )
    
    # Read audio file
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    
    # Create binary message with audio
    binary_message = BinaryMessage.from_bytes(
        content=audio_bytes,
        mime_type="audio/mp3",
    )
    
    # Force transformation even if model might support it
    response, messages = await proxy.agent_run(
        model="gpt-4o",
        messages=[
            binary_message,
            TextMessage(role="user", content="What did you hear?"),
        ],
        tools=[],
        force_binary_transformation=True,  # Force transformation
    )
    
    print(f"Response: {response.output}")
    print(f"Transformed messages count: {len(messages)}")
    
    # Check that binary message was transformed to text
    for i, msg in enumerate(messages):
        print(f"Message {i}: {type(msg).__name__}")


async def test_image_binary_unsupported():
    """Test sending image as binary message to agent that doesn't support it."""
    print("\n=== Image Binary Message (Unsupported Model) ===")
    
    # Configuration - Update these with your actual credentials
    BASE_URL = "http://localhost:8000"  # Your proxy server URL
    API_KEY = "your-api-key-here"  # Your API key
    GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com"  # Google API URL
    
    # Create a simple test image (1x1 PNG)
    # This is a minimal valid PNG file
    png_bytes = bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,  # 1x1 pixels
        0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
        0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41,  # IDAT chunk
        0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
        0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00,
        0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,  # IEND chunk
        0x42, 0x60, 0x82
    ])
    
    # Initialize the proxy client
    proxy = LivellmProxy(
        base_url=BASE_URL,
        primary_creds=Creds(
            api_key=API_KEY,
            provider="openai",
            base_url="https://api.openai.com/v1"
        ),
        providers=[
            create_google_provider_config(API_KEY, base_url=GOOGLE_BASE_URL),
        ]
    )
    
    # Create binary message with image
    binary_message = BinaryMessage.from_bytes(
        content=png_bytes,
        mime_type="image/png",
        caption="A test image"
    )
    
    # Send to model without image capabilities
    response, messages = await proxy.agent_run(
        model="gpt-4o",
        messages=[
            binary_message,
            TextMessage(role="user", content="Describe what you see."),
        ],
        tools=[],
        model_capabilities=[],  # No image capability
    )
    
    print(f"Response: {response.output}")
    print(f"Transformed messages count: {len(messages)}")
    
    # Check that binary message was transformed
    for i, msg in enumerate(messages):
        print(f"Message {i}: {type(msg).__name__}")


async def main():
    """Run all binary message examples."""
    await test_audio_binary_unsupported()
    await test_audio_binary_force_transform()
    await test_image_binary_unsupported()


if __name__ == "__main__":
    asyncio.run(main())