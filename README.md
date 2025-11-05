# LiveLLM Python Client

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python client library for the LiveLLM Server - a unified proxy for AI agent, audio, and transcription services.

## Features

- 🚀 **Async-first** - Built on httpx for high-performance operations
- 🔒 **Type-safe** - Full type hints and Pydantic validation
- 🎯 **Multi-provider** - OpenAI, Google, Anthropic, Groq, ElevenLabs
- 🔄 **Streaming** - Real-time streaming for agent and audio
- 🛠️ **Flexible API** - Use request objects or keyword arguments
- 🎙️ **Audio services** - Text-to-speech and transcription
- ⚡ **Fallback strategies** - Sequential and parallel handling
- 🧹 **Auto cleanup** - Context managers and garbage collection

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
from livellm.models import Settings, ProviderKind, TextMessage, MessageRole

async def main():
    # Initialize with automatic provider setup
    async with LivellmClient(
        base_url="http://localhost:8000",
        configs=[
            Settings(
                uid="openai",
                provider=ProviderKind.OPENAI,
                api_key="your-api-key"
            )
        ]
    ) as client:
        # Simple keyword arguments style (gen_config as kwargs)
        response = await client.agent_run(
            provider_uid="openai",
            model="gpt-4",
            messages=[TextMessage(role="user", content="Hello!")],
            temperature=0.7
        )
        print(response.output)

asyncio.run(main())
```

## Configuration

### Client Initialization

```python
from livellm import LivellmClient
from livellm.models import Settings, ProviderKind

# Basic
client = LivellmClient(base_url="http://localhost:8000")

# With timeout and pre-configured providers
client = LivellmClient(
    base_url="http://localhost:8000",
    timeout=30.0,
    configs=[
        Settings(
            uid="openai",
            provider=ProviderKind.OPENAI,
            api_key="sk-...",
            base_url="https://api.openai.com/v1"  # Optional
        ),
        Settings(
            uid="anthropic",
            provider=ProviderKind.ANTHROPIC,
            api_key="sk-ant-...",
            blacklist_models=["claude-instant-1"]  # Optional
        )
    ]
)
```

### Supported Providers

`OPENAI` • `GOOGLE` • `ANTHROPIC` • `GROQ` • `ELEVENLABS`

```python
# Add provider dynamically
await client.update_config(Settings(
    uid="my-provider",
    provider=ProviderKind.OPENAI,
    api_key="your-api-key"
))

# List and delete
configs = await client.get_configs()
await client.delete_config("my-provider")
```

## Usage Examples

### Agent Services

#### Two Ways to Call Methods

All methods support **two calling styles**:

**Style 1: Keyword arguments** (kwargs become `gen_config`)
```python
response = await client.agent_run(
    provider_uid="openai",
    model="gpt-4",
    messages=[TextMessage(role="user", content="Hello!")],
    temperature=0.7,
    max_tokens=500
)
```

**Style 2: Request objects**
```python
from livellm.models import AgentRequest

response = await client.agent_run(
    AgentRequest(
        provider_uid="openai",
        model="gpt-4",
        messages=[TextMessage(role="user", content="Hello!")],
        gen_config={"temperature": 0.7, "max_tokens": 500}
    )
)
```

#### Basic Agent Run

```python
from livellm.models import TextMessage

# Using kwargs (recommended for simplicity)
response = await client.agent_run(
    provider_uid="openai",
    model="gpt-4",
    messages=[
        TextMessage(role="system", content="You are helpful."),
        TextMessage(role="user", content="Explain quantum computing")
    ],
    temperature=0.7,
    max_tokens=500
)
print(f"Output: {response.output}")
print(f"Tokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out")
```

#### Streaming Agent Response

```python
# Streaming also supports both styles
stream = client.agent_run_stream(
    provider_uid="openai",
    model="gpt-4",
    messages=[TextMessage(role="user", content="Tell me a story")],
    temperature=0.8
)

async for chunk in stream:
    print(chunk.output, end="", flush=True)
```

#### Agent with Vision (Binary Messages)

```python
import base64
from livellm.models import BinaryMessage

with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

response = await client.agent_run(
    provider_uid="openai",
    model="gpt-4-vision",
    messages=[
        BinaryMessage(
            role="user",
            content=image_data,
            mime_type="image/jpeg",
            caption="What's in this image?"
        )
    ]
)
```

#### Agent with Tools

```python
from livellm.models import WebSearchInput, MCPStreamableServerInput, ToolKind

# Web search tool
response = await client.agent_run(
    provider_uid="openai",
    model="gpt-4",
    messages=[TextMessage(role="user", content="Latest AI news?")],
    tools=[WebSearchInput(
        kind=ToolKind.WEB_SEARCH,
        search_context_size="high"  # low, medium, or high
    )]
)

# MCP server tool
response = await client.agent_run(
    provider_uid="openai",
    model="gpt-4",
    messages=[TextMessage(role="user", content="Run custom tool")],
    tools=[MCPStreamableServerInput(
        kind=ToolKind.MCP_STREAMABLE_SERVER,
        url="http://mcp-server:8080",
        prefix="mcp_",
        timeout=15
    )]
)
```

### Audio Services

#### Text-to-Speech

```python
from livellm.models import SpeakMimeType

# Non-streaming
audio = await client.speak(
    provider_uid="openai",
    model="tts-1",
    text="Hello, world!",
    voice="alloy",
    mime_type=SpeakMimeType.MP3,
    sample_rate=24000,
    speed=1.0  # kwargs become gen_config
)
with open("output.mp3", "wb") as f:
    f.write(audio)

# Streaming
audio = bytes()
async for chunk in client.speak_stream(
    provider_uid="openai",
    model="tts-1",
    text="Hello, world!",
    voice="alloy",
    mime_type=SpeakMimeType.PCM,
    sample_rate=24000
):
    audio += chunk

# Save PCM as WAV
import wave
with wave.open("output.wav", "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(24000)
    wf.writeframes(audio)
```

#### Transcription

```python
# Method 1: Multipart upload (kwargs style)
with open("audio.wav", "rb") as f:
    audio_bytes = f.read()

transcription = await client.transcribe(
    provider_uid="openai",
    file=("audio.wav", audio_bytes, "audio/wav"),
    model="whisper-1",
    language="en",  # Optional
    temperature=0.0  # kwargs become gen_config
)
print(f"Text: {transcription.text}")
print(f"Language: {transcription.language}")

# Method 2: JSON request object (base64-encoded)
import base64
from livellm.models import TranscribeRequest

audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
transcription = await client.transcribe(
    TranscribeRequest(
        provider_uid="openai",
        file=("audio.wav", audio_b64, "audio/wav"),
        model="whisper-1"
    )
)
```

### Fallback Strategies

Handle failures automatically with sequential or parallel fallback:

```python
from livellm.models import AgentRequest, AgentFallbackRequest, FallbackStrategy, TextMessage

messages = [TextMessage(role="user", content="Hello!")]

# Sequential: try each in order until one succeeds
response = await client.agent_run(
    AgentFallbackRequest(
        strategy=FallbackStrategy.SEQUENTIAL,
        requests=[
            AgentRequest(provider_uid="primary", model="gpt-4", messages=messages, tools=[]),
            AgentRequest(provider_uid="backup", model="claude-3", messages=messages, tools=[])
        ],
        timeout_per_request=30
    )
)

# Parallel: try all simultaneously, use first success
response = await client.agent_run(
    AgentFallbackRequest(
        strategy=FallbackStrategy.PARALLEL,
        requests=[
            AgentRequest(provider_uid="p1", model="gpt-4", messages=messages, tools=[]),
            AgentRequest(provider_uid="p2", model="claude-3", messages=messages, tools=[]),
            AgentRequest(provider_uid="p3", model="gemini-pro", messages=messages, tools=[])
        ],
        timeout_per_request=10
    )
)

# Also works for audio
from livellm.models import AudioFallbackRequest, SpeakRequest

audio = await client.speak(
    AudioFallbackRequest(
        strategy=FallbackStrategy.SEQUENTIAL,
        requests=[
            SpeakRequest(provider_uid="elevenlabs", model="turbo", text="Hi", 
                        voice="rachel", mime_type=SpeakMimeType.MP3, sample_rate=44100),
            SpeakRequest(provider_uid="openai", model="tts-1", text="Hi",
                        voice="alloy", mime_type=SpeakMimeType.MP3, sample_rate=44100)
        ]
    )
)
```

## Resource Management

**Recommended**: Use context managers for automatic cleanup.

```python
# ✅ Best: Context manager (auto cleanup)
async with LivellmClient(base_url="http://localhost:8000") as client:
    response = await client.ping()
# Configs deleted, connection closed automatically

# ✅ Good: Manual cleanup
client = LivellmClient(base_url="http://localhost:8000")
try:
    response = await client.ping()
finally:
    await client.cleanup()

# ⚠️ OK: Garbage collection (shows warning if configs exist)
client = LivellmClient(base_url="http://localhost:8000")
response = await client.ping()
# Cleaned up when object is destroyed
```

## API Reference

### Client Methods

**Configuration**
- `ping()` - Health check
- `update_config(config)` / `update_configs(configs)` - Add/update providers
- `get_configs()` - List all configurations
- `delete_config(uid)` - Remove provider

**Agent**
- `agent_run(request | **kwargs)` - Run agent (blocking)
- `agent_run_stream(request | **kwargs)` - Run agent (streaming)

**Audio**
- `speak(request | **kwargs)` - Text-to-speech (blocking)
- `speak_stream(request | **kwargs)` - Text-to-speech (streaming)
- `transcribe(request | **kwargs)` - Speech-to-text

**Cleanup**
- `cleanup()` - Release resources
- `async with client:` - Auto cleanup (recommended)

### Key Models

**Core**
- `Settings(uid, provider, api_key, base_url?, blacklist_models?)` - Provider config
- `ProviderKind` - `OPENAI` | `GOOGLE` | `ANTHROPIC` | `GROQ` | `ELEVENLABS`

**Messages**
- `TextMessage(role, content)` - Text message
- `BinaryMessage(role, content, mime_type, caption?)` - Image/audio message
- `MessageRole` - `USER` | `MODEL` | `SYSTEM` (or use strings: `"user"`, `"model"`, `"system"`)

**Requests**
- `AgentRequest(provider_uid, model, messages, tools?, gen_config?)`
- `SpeakRequest(provider_uid, model, text, voice, mime_type, sample_rate, gen_config?)`
- `TranscribeRequest(provider_uid, file, model, language?, gen_config?)`

**Tools**
- `WebSearchInput(kind=ToolKind.WEB_SEARCH, search_context_size)`
- `MCPStreamableServerInput(kind=ToolKind.MCP_STREAMABLE_SERVER, url, prefix?, timeout?)`

**Fallback**
- `AgentFallbackRequest(strategy, requests, timeout_per_request?)`
- `AudioFallbackRequest(strategy, requests, timeout_per_request?)`
- `FallbackStrategy` - `SEQUENTIAL` | `PARALLEL`

**Responses**
- `AgentResponse(output, usage{input_tokens, output_tokens}, ...)`
- `TranscribeResponse(text, language)`

## Error Handling

```python
import httpx

try:
    response = await client.agent_run(
        provider_uid="openai",
        model="gpt-4",
        messages=[TextMessage(role="user", content="Hi")]
    )
except httpx.HTTPStatusError as e:
    print(f"HTTP {e.response.status_code}: {e.response.text}")
except httpx.RequestError as e:
    print(f"Request failed: {e}")
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[testing]"

# Run tests
pytest tests/

# Type checking
mypy livellm
```

## Requirements

- Python 3.10+
- httpx >= 0.27.0
- pydantic >= 2.0.0

## Links

- [GitHub Repository](https://github.com/qalby-tech/livellm-client-py)
- [Issue Tracker](https://github.com/qalby-tech/livellm-client-py/issues)

## License

MIT License - see LICENSE file for details.
