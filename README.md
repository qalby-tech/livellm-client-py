# LiveLLM Python Client

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python client library for the LiveLLM Server - a unified proxy for AI agent, audio, and transcription services.

## Features

- 🚀 **Async-first design** - Built on httpx for high-performance async operations
- 🔒 **Type-safe** - Full type hints and Pydantic validation
- 🎯 **Multi-provider support** - OpenAI, Google, Anthropic, Groq, ElevenLabs
- 🔄 **Streaming support** - Real-time streaming for agent and audio responses
- 🛠️ **Agent tools** - Web search and MCP server integration
- 🎙️ **Audio services** - Text-to-speech and transcription
- ⚡ **Fallback strategies** - Sequential and parallel fallback handling
- 📦 **Smart resource management** - Automatic cleanup via GC, context managers, or manual control
- 🧹 **Memory safe** - No resource leaks with multiple cleanup strategies

## Installation

```bash
pip install livellm
```

Or with development dependencies:

```bash
pip install livellm[testing]
```

## Quick Start

```python
import asyncio
from livellm import LivellmClient
from livellm.models import Settings, ProviderKind, AgentRequest, TextMessage, MessageRole
from pydantic import SecretStr

async def main():
    # Initialize the client with context manager for automatic cleanup
    async with LivellmClient(base_url="http://localhost:8000") as client:
        # Configure a provider
        config = Settings(
            uid="my-openai-config",
            provider=ProviderKind.OPENAI,
            api_key=SecretStr("your-api-key")
        )
        await client.update_config(config)
        
        # Run an agent query
        request = AgentRequest(
            provider_uid="my-openai-config",
            model="gpt-4",
            messages=[
                TextMessage(role=MessageRole.USER, content="Hello, how are you?")
            ],
            tools=[]
        )
        
        response = await client.agent_run(request)
        print(response.output)

asyncio.run(main())
```

## Configuration

### Client Initialization

```python
from livellm import LivellmClient

# Basic initialization
client = LivellmClient(base_url="http://localhost:8000")

# With timeout
client = LivellmClient(
    base_url="http://localhost:8000",
    timeout=30.0
)

# With pre-configured providers (sync operation)
from livellm.models import Settings, ProviderKind
from pydantic import SecretStr

configs = [
    Settings(
        uid="openai-config",
        provider=ProviderKind.OPENAI,
        api_key=SecretStr("sk-..."),
        base_url="https://api.openai.com/v1"  # Optional custom base URL
    ),
    Settings(
        uid="anthropic-config",
        provider=ProviderKind.ANTHROPIC,
        api_key=SecretStr("sk-ant-..."),
        blacklist_models=["claude-instant-1"]  # Optional model blacklist
    )
]

client = LivellmClient(
    base_url="http://localhost:8000",
    configs=configs
)
```

### Provider Configuration

Supported providers:
- `OPENAI` - OpenAI GPT models
- `GOOGLE` - Google Gemini models
- `ANTHROPIC` - Anthropic Claude models
- `GROQ` - Groq models
- `ELEVENLABS` - ElevenLabs text-to-speech

```python
# Add a provider configuration
config = Settings(
    uid="unique-provider-id",
    provider=ProviderKind.OPENAI,
    api_key=SecretStr("your-api-key"),
    base_url="https://custom-endpoint.com",  # Optional
    blacklist_models=["deprecated-model"]     # Optional
)
await client.update_config(config)

# Get all configurations
configs = await client.get_configs()

# Delete a configuration
await client.delete_config("unique-provider-id")
```

## Usage Examples

### Agent Services

#### Basic Agent Run

```python
from livellm.models import AgentRequest, TextMessage, MessageRole

request = AgentRequest(
    provider_uid="my-openai-config",
    model="gpt-4",
    messages=[
        TextMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        TextMessage(role=MessageRole.USER, content="Explain quantum computing")
    ],
    tools=[],
    gen_config={"temperature": 0.7, "max_tokens": 500}
)

response = await client.agent_run(request)
print(f"Output: {response.output}")
print(f"Tokens used - Input: {response.usage.input_tokens}, Output: {response.usage.output_tokens}")
```

**Note:** You can use either `MessageRole` enum or string values for the `role` parameter:

```python
# Using enum (recommended for type safety)
TextMessage(role=MessageRole.USER, content="Hello")

# Using string (more convenient)
TextMessage(role="user", content="Hello")

# Both work identically and serialize correctly
```

#### Streaming Agent Response

```python
request = AgentRequest(
    provider_uid="my-openai-config",
    model="gpt-4",
    messages=[
        TextMessage(role=MessageRole.USER, content="Tell me a story")
    ],
    tools=[]
)

stream = await client.agent_run_stream(request)
async for chunk in stream:
    print(chunk.output, end="", flush=True)
```

#### Agent with Binary Messages

```python
import base64

# Read and encode image
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

from livellm.models import BinaryMessage

request = AgentRequest(
    provider_uid="my-openai-config",
    model="gpt-4-vision",
    messages=[
        BinaryMessage(
            role=MessageRole.USER,
            content=image_data,
            mime_type="image/jpeg",
            caption="What's in this image?"
        )
    ],
    tools=[]
)

response = await client.agent_run(request)
```

#### Agent with Web Search Tool

```python
from livellm.models import WebSearchInput, ToolKind

request = AgentRequest(
    provider_uid="my-openai-config",
    model="gpt-4",
    messages=[
        TextMessage(role=MessageRole.USER, content="What's the latest news about AI?")
    ],
    tools=[
        WebSearchInput(
            kind=ToolKind.WEB_SEARCH,
            search_context_size="high"  # Options: "low", "medium", "high"
        )
    ]
)

response = await client.agent_run(request)
```

#### Agent with MCP Server Tool

```python
from livellm.models import MCPStreamableServerInput, ToolKind

request = AgentRequest(
    provider_uid="my-openai-config",
    model="gpt-4",
    messages=[
        TextMessage(role=MessageRole.USER, content="Execute tool")
    ],
    tools=[
        MCPStreamableServerInput(
            kind=ToolKind.MCP_STREAMABLE_SERVER,
            url="http://mcp-server:8080",
            prefix="mcp_",
            timeout=15,
            kwargs={"custom_param": "value"}
        )
    ]
)

response = await client.agent_run(request)
```

### Audio Services

#### Text-to-Speech

```python
from livellm.models import SpeakRequest, SpeakMimeType

request = SpeakRequest(
    provider_uid="elevenlabs-config",
    model="eleven_turbo_v2",
    text="Hello, this is a test of text to speech.",
    voice="rachel",
    mime_type=SpeakMimeType.MP3,
    sample_rate=44100,
    gen_config={"stability": 0.5, "similarity_boost": 0.75}
)

# Get audio as bytes
audio_bytes = await client.speak(request)
with open("output.mp3", "wb") as f:
    f.write(audio_bytes)
```

#### Streaming Text-to-Speech

```python
request = SpeakRequest(
    provider_uid="elevenlabs-config",
    model="eleven_turbo_v2",
    text="This is a longer text that will be streamed.",
    voice="rachel",
    mime_type=SpeakMimeType.MP3,
    sample_rate=44100,
    chunk_size=20  # Chunk size in milliseconds
)

# Stream audio chunks
stream = await client.speak_stream(request)
with open("output.mp3", "wb") as f:
    async for chunk in stream:
        f.write(chunk)
```

#### Audio Transcription (Multipart)

```python
# Using multipart upload
with open("audio.mp3", "rb") as f:
    file_tuple = ("audio.mp3", f.read(), "audio/mpeg")

response = await client.transcribe(
    provider_uid="openai-config",
    file=file_tuple,
    model="whisper-1",
    language="en",
    gen_config={"temperature": 0.2}
)

print(f"Transcription: {response.text}")
print(f"Detected language: {response.language}")
```

#### Audio Transcription (JSON)

```python
import base64
from livellm.models import TranscribeRequest

with open("audio.mp3", "rb") as f:
    audio_data = base64.b64encode(f.read()).decode("utf-8")

request = TranscribeRequest(
    provider_uid="openai-config",
    model="whisper-1",
    file=("audio.mp3", audio_data, "audio/mpeg"),
    language="en"
)

response = await client.transcribe_json(request)
```

### Fallback Strategies

#### Sequential Fallback (Try each provider in order)

```python
from livellm.models import AgentFallbackRequest, FallbackStrategy

fallback_request = AgentFallbackRequest(
    requests=[
        AgentRequest(
            provider_uid="primary-provider",
            model="gpt-4",
            messages=[TextMessage(role=MessageRole.USER, content="Hello")],
            tools=[]
        ),
        AgentRequest(
            provider_uid="backup-provider",
            model="claude-3",
            messages=[TextMessage(role=MessageRole.USER, content="Hello")],
            tools=[]
        )
    ],
    strategy=FallbackStrategy.SEQUENTIAL,
    timeout_per_request=30
)

response = await client.agent_run(fallback_request)
```

#### Parallel Fallback (Try all providers simultaneously)

```python
fallback_request = AgentFallbackRequest(
    requests=[
        AgentRequest(provider_uid="provider-1", model="gpt-4", messages=messages, tools=[]),
        AgentRequest(provider_uid="provider-2", model="claude-3", messages=messages, tools=[]),
        AgentRequest(provider_uid="provider-3", model="gemini-pro", messages=messages, tools=[])
    ],
    strategy=FallbackStrategy.PARALLEL,
    timeout_per_request=10
)

response = await client.agent_run(fallback_request)
```

#### Audio Fallback

```python
from livellm.models import AudioFallbackRequest

fallback_request = AudioFallbackRequest(
    requests=[
        SpeakRequest(provider_uid="elevenlabs", model="model-1", text=text, voice="voice1", 
                     mime_type=SpeakMimeType.MP3, sample_rate=44100),
        SpeakRequest(provider_uid="openai", model="tts-1", text=text, voice="alloy",
                     mime_type=SpeakMimeType.MP3, sample_rate=44100)
    ],
    strategy=FallbackStrategy.SEQUENTIAL
)

audio = await client.speak(fallback_request)
```

## Resource Management

The client provides multiple ways to manage resources and cleanup:

### 1. Automatic Cleanup (Garbage Collection)

The client automatically cleans up when garbage collected:

```python
async def main():
    client = LivellmClient(base_url="http://localhost:8000")
    
    # Use client...
    response = await client.ping()
    
    # No explicit cleanup needed - handled automatically when object is destroyed
    # Note: Provider configs are deleted synchronously from the server

asyncio.run(main())
```

**Note**: While automatic cleanup works, it shows a `ResourceWarning` if configs exist to encourage explicit cleanup for immediate resource release.

### 2. Context Manager (Recommended)

Use async context managers for guaranteed cleanup:

```python
async with LivellmClient(base_url="http://localhost:8000") as client:
    config = Settings(uid="temp-config", provider=ProviderKind.OPENAI, 
                      api_key=SecretStr("key"))
    await client.update_config(config)
    
    # Use client...
    response = await client.ping()
    
# Automatically cleans up configs and closes HTTP client
```

### 3. Manual Cleanup

Explicitly call cleanup in a try/finally block:

```python
client = LivellmClient(base_url="http://localhost:8000")
try:
    # Use client...
    response = await client.ping()
finally:
    await client.cleanup()
```

### Cleanup Behavior

The `cleanup()` method:
- Deletes all provider configs created by the client
- Closes the HTTP client connection
- Is idempotent (safe to call multiple times)

The `__del__()` destructor (automatic cleanup):
- Triggers when the object is garbage collected
- Synchronously deletes provider configs from the server
- Closes the HTTP client connection
- Shows a `ResourceWarning` if configs exist (to encourage explicit cleanup)

## API Reference

### Client Methods

#### Health Check
- `ping() -> SuccessResponse` - Check server health

#### Configuration Management
- `update_config(config: Settings) -> SuccessResponse` - Add/update a provider config
- `update_configs(configs: List[Settings]) -> SuccessResponse` - Add/update multiple configs
- `get_configs() -> List[Settings]` - Get all provider configurations
- `delete_config(config_uid: str) -> SuccessResponse` - Delete a provider config

#### Agent Services
- `agent_run(request: AgentRequest | AgentFallbackRequest) -> AgentResponse` - Run agent query
- `agent_run_stream(request: AgentRequest | AgentFallbackRequest) -> AsyncIterator[AgentResponse]` - Stream agent response

#### Audio Services
- `speak(request: SpeakRequest | AudioFallbackRequest) -> bytes` - Text-to-speech
- `speak_stream(request: SpeakRequest | AudioFallbackRequest) -> AsyncIterator[bytes]` - Streaming TTS
- `transcribe(provider_uid, file, model, language?, gen_config?) -> TranscribeResponse` - Multipart transcription
- `transcribe_json(request: TranscribeRequest | TranscribeFallbackRequest) -> TranscribeResponse` - JSON transcription

#### Cleanup
- `cleanup() -> None` - Clean up resources and close client (async)
- `__aenter__() / __aexit__()` - Async context manager support
- `__del__()` - Automatic cleanup when garbage collected (sync)

### Models

#### Common Models
- `Settings` - Provider configuration
- `ProviderKind` - Enum of supported providers
- `SuccessResponse` - Generic success response
- `BaseRequest` - Base class for all requests

#### Agent Models
- `AgentRequest` - Agent query request
- `AgentResponse` - Agent query response
- `AgentResponseUsage` - Token usage information
- `TextMessage` - Text-based message
- `BinaryMessage` - Binary message (images, audio, etc.)
- `MessageRole` - Enum: USER, MODEL, SYSTEM

#### Tool Models
- `ToolKind` - Enum: WEB_SEARCH, MCP_STREAMABLE_SERVER
- `WebSearchInput` - Web search tool configuration
- `MCPStreamableServerInput` - MCP server tool configuration

#### Audio Models
- `SpeakRequest` - Text-to-speech request
- `SpeakMimeType` - Enum: PCM, WAV, MP3, ULAW, ALAW
- `TranscribeRequest` - Transcription request
- `TranscribeResponse` - Transcription response

#### Fallback Models
- `FallbackStrategy` - Enum: SEQUENTIAL, PARALLEL
- `AgentFallbackRequest` - Agent fallback configuration
- `AudioFallbackRequest` - Audio fallback configuration
- `TranscribeFallbackRequest` - Transcription fallback configuration

## Error Handling

The client raises exceptions for HTTP errors:

```python
try:
    response = await client.agent_run(request)
except Exception as e:
    print(f"Error: {e}")
```

For more granular error handling:

```python
import httpx

try:
    response = await client.ping()
except httpx.HTTPStatusError as e:
    print(f"HTTP error: {e.response.status_code}")
except httpx.RequestError as e:
    print(f"Request error: {e}")
```

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[testing]"

# Run tests
pytest tests/
```

### Type Checking

The library is fully typed. Run type checking with:

```bash
pip install mypy
mypy livellm
```

## Requirements

- Python 3.10+
- httpx >= 0.27.0
- pydantic >= 2.0.0

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Links

- [GitHub Repository](https://github.com/qalby-tech/livellm-client-py)
- [Issue Tracker](https://github.com/qalby-tech/livellm-client-py/issues)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and changes.
