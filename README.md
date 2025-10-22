# LiveLLM Proxy Client

A Python client for the LiveLLM Proxy API with production features including automatic binary message transformation, multi-provider support, and comprehensive audio/text processing capabilities.

## Installation

### Using uv (Recommended)

```bash
# Install from Git repository
uv add git+https://github.com/qalby-tech/livellm-client-py.git

# Or install from local directory
uv add .
```

### Using pip

```bash
pip install git+https://github.com/qalby-tech/livellm-client-py.git
```

## Quick Start

```python
import asyncio
from livellm import LivellmProxy, Creds, TextMessage
from livellm import create_openai_provider_config

async def main():
    # Initialize the proxy client
    proxy = LivellmProxy(
        base_url="http://localhost:8000",
    primary_creds=Creds(
        api_key="your-api-key",
        provider="openai",
        base_url="https://api.openai.com/v1"
    ),
    providers=[
        create_openai_provider_config("your-api-key", base_url="https://api.openai.com/v1"),
        ]
    )
    
    # Run a basic conversation
    response, messages = await proxy.agent_run(
        model="gpt-4o",
        messages=[
            TextMessage(role="user", content="Hello, how are you?"),
        ],
        tools=[],
    )
    
    print(f"Response: {response.output}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Features

- **Multi-Provider Support**: Configure multiple AI providers (OpenAI, Google, etc.)
- **Automatic Binary Transformation**: Audio and image messages are automatically transcribed/described when models don't support them
- **Streaming Support**: Real-time streaming for both text and audio generation
- **Audio Processing**: Text-to-speech (TTS) and speech-to-text (STT) capabilities
- **Production Ready**: Built with httpx and pydantic for reliability and type safety

## Examples

The `examples/` directory contains comprehensive examples for different use cases:

### Basic Usage
- [`basic_agent.py`](examples/basic_agent.py) - Simple text conversation
- [`streaming_agent.py`](examples/streaming_agent.py) - Streaming text responses

### Audio Processing
- [`audio_tts.py`](examples/audio_tts.py) - Text-to-speech generation (regular and streaming)
- [`audio_transcribe.py`](examples/audio_transcribe.py) - Audio transcription

### Advanced Features
- [`binary_messages.py`](examples/binary_messages.py) - Working with audio and image messages
- [`raw_client.py`](examples/raw_client.py) - Direct API access

## Configuration

### Provider Setup

```python
from livellm import create_openai_provider_config, create_google_provider_config

# OpenAI provider
openai_provider = create_openai_provider_config(
    api_key="your-openai-key",
    base_url="https://api.openai.com/v1"
)

# Google provider
google_provider = create_google_provider_config(
    api_key="your-google-key",
    base_url="https://generativelanguage.googleapis.com"
)
```

### Client Initialization

```python
from livellm import LivellmProxy, Creds

proxy = LivellmProxy(
    base_url="http://localhost:8000",  # Your proxy server URL
    primary_creds=Creds(
        api_key="your-primary-key",
        provider="openai",
        base_url="https://api.openai.com/v1"
    ),
    providers=[
        openai_provider,
        google_provider,
        # Add more providers as needed
    ]
)
```

## API Reference

### Core Methods

#### `agent_run(model, messages, tools, **kwargs)`
Run a conversation with the specified model.

**Parameters:**
- `model` (str): Model identifier (e.g., "gpt-4o", "claude-3")
- `messages` (List[Message]): List of conversation messages
- `tools` (List[Tool]): List of available tools
- `model_capabilities` (List[ModelCapability], optional): Model capabilities
- `force_binary_transformation` (bool, optional): Force transformation of binary messages

**Returns:**
- `response`: Agent response with output and usage information
- `messages`: Processed message list

#### `agent_run_stream(model, messages, tools, **kwargs)`
Run a streaming conversation.

**Returns:**
- `stream`: Async generator of response chunks
- `messages`: Processed message list

#### `audio_speak(model, text, voice, output_format, **kwargs)`
Generate speech from text.

**Parameters:**
- `model` (str): TTS model (e.g., "tts-1")
- `text` (str): Text to convert to speech
- `voice` (str): Voice identifier (e.g., "alloy", "nova")
- `output_format` (str): Output format ("mp3", "wav", etc.)

**Returns:**
- `bytes`: Audio data

#### `audio_speak_stream(model, text, voice, output_format, **kwargs)`
Generate streaming speech from text.

**Returns:**
- `AsyncGenerator[bytes]`: Streaming audio chunks

#### `audio_transcribe(model, file, language=None, **kwargs)`
Transcribe audio to text.

**Parameters:**
- `model` (str): STT model (e.g., "whisper-1")
- `file` (tuple): File tuple (filename, bytes, mime_type)
- `language` (str, optional): Language hint

**Returns:**
- `TranscriptionResponse`: Transcription result with text and language

### Message Types

#### `TextMessage`
```python
TextMessage(role="user", content="Hello, world!")
```

#### `BinaryMessage`
```python
# From bytes
BinaryMessage.from_bytes(
    content=audio_bytes,
    mime_type="audio/wav",
    caption="Audio description"
)

# From file
BinaryMessage.from_file(
    file_path="path/to/file.wav",
    caption="Audio description"
)
```

## Binary Message Transformation

The client automatically handles binary messages (audio, images) when the target model doesn't support them:

1. **Audio messages** are automatically transcribed to text using the configured STT model
2. **Image messages** are automatically described using vision models
3. **Force transformation** can be enabled with `force_binary_transformation=True`

```python
# Audio will be automatically transcribed if model doesn't support audio
response, messages = await proxy.agent_run(
    model="gpt-4o",  # Doesn't support audio
    messages=[
        BinaryMessage.from_bytes(audio_bytes, "audio/wav", "What's in this audio?"),
        TextMessage(role="user", content="Based on the audio, what did you hear?")
    ],
    tools=[]
)
```

## Error Handling

The client includes comprehensive error handling with detailed error messages:

```python
from livellm import ValidationError, HTTPValidationError

try:
    response, messages = await proxy.agent_run(...)
except (ValidationError, HTTPValidationError) as e:
    print(f"Error: {e}")
```

## Requirements

- Python 3.12+
- httpx >= 0.27.0
- pydantic >= 2.0.0

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on the GitHub repository.
