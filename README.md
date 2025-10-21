# LiveLLM Proxy Client - Python

Python client for the LiveLLM Proxy API with support for agent interactions, text-to-speech, and speech-to-text across multiple AI providers.

## Installation

```bash
pip install -e .
```

## Features

- 🤖 **Agent API**: Run conversational agents with tool support
- 🎤 **Text-to-Speech**: Convert text to audio using OpenAI or ElevenLabs
- 📝 **Speech-to-Text**: Transcribe audio files to text
- 🔄 **Streaming Support**: Stream responses for real-time interactions
- ⚡ **Async/Sync**: Both synchronous and asynchronous interfaces
- 🔧 **Type-Safe**: Full type hints with Pydantic models
- 🌐 **Multi-Provider**: Support for OpenAI, Google, Anthropic, Groq, and more
- 🔁 **Automatic Fallback**: Seamlessly fallback to alternative providers on failure

## Quick Start

### Basic Agent Interaction

```python
from livellm_client import RawClient, AgentRequest, TextMessage, MessageRole

# Initialize the client
client = RawClient(base_url="https://your-proxy-url.com")

# Create a simple agent request
request = AgentRequest(
    model="gpt-4",
    messages=[
        TextMessage(role=MessageRole.USER, content="Hello, how are you?")
    ],
    tools=[]
)

# Run the agent
response = client.agent_run(
    request=request,
    api_key="your-openai-api-key",
    provider="openai"
)

print(f"Response: {response.output}")
print(f"Tokens used: {response.usage.input_tokens} in, {response.usage.output_tokens} out")
```

### Streaming Agent Response

```python
from livellm_client import RawClient, AgentRequest, TextMessage, MessageRole

client = RawClient(base_url="https://your-proxy-url.com")

request = AgentRequest(
    model="gpt-4",
    messages=[
        TextMessage(role=MessageRole.USER, content="Tell me a story")
    ],
    tools=[]
)

# Stream the response
for chunk in client.agent_run_stream(
    request=request,
    api_key="your-openai-api-key",
    provider="openai"
):
    print(chunk.output, end="", flush=True)
```

### Agent with Web Search Tool

```python
from livellm_client import (
    RawClient, AgentRequest, TextMessage, 
    MessageRole, WebSearchInput
)

client = RawClient(base_url="https://your-proxy-url.com")

request = AgentRequest(
    model="gpt-4",
    messages=[
        TextMessage(role=MessageRole.USER, content="What's the latest news about AI?")
    ],
    tools=[
        WebSearchInput(search_context_size="medium")
    ]
)

response = client.agent_run(
    request=request,
    api_key="your-openai-api-key",
    provider="openai"
)

print(response.output)
```

### Text-to-Speech

```python
from livellm_client import RawClient, SpeakRequest

client = RawClient(base_url="https://your-proxy-url.com")

request = SpeakRequest(
    model="tts-1",
    text="Hello, this is a test of the text to speech system.",
    voice="alloy",
    output_format="mp3"
)

# Get audio bytes
audio_bytes = client.audio_speak(
    request=request,
    api_key="your-openai-api-key",
    provider="openai"
)

# Save to file
with open("output.mp3", "wb") as f:
    f.write(audio_bytes)
```

### Text-to-Speech Streaming

```python
from livellm_client import RawClient, SpeakRequest

client = RawClient(base_url="https://your-proxy-url.com")

request = SpeakRequest(
    model="tts-1",
    text="This will be streamed as audio chunks.",
    voice="alloy",
    output_format="mp3"
)

# Stream audio chunks
with open("output.mp3", "wb") as f:
    for chunk in client.audio_speak_stream(
        request=request,
        api_key="your-openai-api-key",
        provider="openai"
    ):
        f.write(chunk)
```

### Speech-to-Text (Transcription)

```python
from livellm_client import RawClient

client = RawClient(base_url="https://your-proxy-url.com")

# Read audio file
with open("audio.mp3", "rb") as f:
    audio_bytes = f.read()

# Transcribe
response = client.audio_transcribe(
    model="whisper-1",
    file=("audio.mp3", audio_bytes),
    api_key="your-openai-api-key",
    provider="openai",
    language="en"  # optional
)

print(f"Transcription: {response.text}")
print(f"Language: {response.language}")
```

### Async Usage

```python
import asyncio
from livellm_client import RawClient, AgentRequest, TextMessage, MessageRole

async def main():
    client = RawClient(base_url="https://your-proxy-url.com")
    
    request = AgentRequest(
        model="gpt-4",
        messages=[
            TextMessage(role=MessageRole.USER, content="Hello!")
        ],
        tools=[]
    )
    
    # Async agent call
    response = await client.agent_run_async(
        request=request,
        api_key="your-openai-api-key",
        provider="openai"
    )
    
    print(response.output)
    
    # Async streaming
    async for chunk in client.agent_run_stream_async(
        request=request,
        api_key="your-openai-api-key",
        provider="openai"
    ):
        print(chunk.output, end="", flush=True)

asyncio.run(main())
```

### Context Manager

```python
from livellm_client import RawClient

# Use with context manager to ensure cleanup
with RawClient(base_url="https://your-proxy-url.com") as client:
    response = client.ping()
    print(response)
```

## API Reference

### RawClient

The main HTTP client for interacting with the LiveLLM Proxy API.

**Constructor:**
- `base_url` (str): Base URL of the API
- `timeout` (float): Request timeout in seconds (default: 30.0)
- `http2` (bool): Enable HTTP/2 (default: True)

**Agent Methods:**
- `agent_run()`: Run an agent and get the complete response
- `agent_run_stream()`: Stream agent responses as they're generated
- `agent_run_async()`: Async version of agent_run
- `agent_run_stream_async()`: Async version of agent_run_stream

**Audio Methods:**
- `audio_speak()`: Convert text to speech (returns complete audio)
- `audio_speak_stream()`: Stream text to speech audio
- `audio_transcribe()`: Transcribe audio to text
- `audio_speak_async()`: Async version of audio_speak
- `audio_speak_stream_async()`: Async version of audio_speak_stream
- `audio_transcribe_async()`: Async version of audio_transcribe

**Utility Methods:**
- `ping()`: Health check endpoint
- `ping_async()`: Async version of ping

### Models

All request/response models are Pydantic models with full validation:

- `AgentRequest`: Configuration for agent runs
- `AgentResponse`: Response from agent runs
- `TextMessage`: Text message in a conversation
- `BinaryMessage`: Binary message (image, audio, etc.)
- `SpeakRequest`: Text-to-speech request
- `TranscribeResponse`: Transcription response
- `WebSearchInput`: Web search tool configuration
- `MCPStreamableServerInput`: MCP server tool configuration

## Supported Providers

### Agent Providers
- OpenAI (`openai`)
- Google (`google`)
- Anthropic (`anthropic`)
- Groq (`groq`)

### Audio Providers
- OpenAI (`openai`)
- ElevenLabs (`elevenlabs`)

## Error Handling

```python
import httpx
from livellm_client import RawClient, HTTPValidationError

client = RawClient(base_url="https://your-proxy-url.com")

try:
    response = client.agent_run(...)
except httpx.HTTPStatusError as e:
    print(f"HTTP error: {e.response.status_code}")
    print(f"Details: {e.response.text}")
except httpx.RequestError as e:
    print(f"Request error: {e}")
```

## Advanced Configuration

### Custom Base URLs

You can specify a custom base URL for the provider:

```python
response = client.agent_run(
    request=request,
    api_key="your-api-key",
    provider="openai",
    base_url="https://custom-openai-endpoint.com"
)
```

### Generation Configuration

Pass custom generation parameters:

```python
request = AgentRequest(
    model="gpt-4",
    messages=[...],
    tools=[],
    gen_config={
        "temperature": 0.7,
        "max_tokens": 1000,
        "top_p": 0.9
    }
)
```

## Advanced Client with Automatic Fallback

The `LivellmProxy` class provides automatic fallback across multiple providers:

```python
from livellm_client import (
    LivellmProxy,
    Creds,
    TextMessage,
    MessageRole,
    create_openai_provider_config,
    create_anthropic_provider_config,
)

# Configure multiple providers for automatic fallback
providers = [
    create_openai_provider_config("openai-key"),
    create_anthropic_provider_config("anthropic-key"),  # Fallback
]

proxy = LivellmProxy(
    base_url="https://your-proxy-url.com",
    primary_creds=Creds(api_key="openai-key", provider="openai", base_url=None),
    providers=providers
)

# If primary provider fails, automatically tries fallback providers
response, _ = await proxy.agent_run(
    model="gpt-4o-mini",
    messages=[TextMessage(role=MessageRole.USER, content="Hello!")],
    tools=[]
)
```

The advanced client also handles:
- **Binary message transformation**: Automatically converts images/audio/video to text for models that don't support them
- **Model capability detection**: Intelligently routes requests based on model capabilities
- **Message preprocessing**: Ensures message compatibility with target models

## Testing

### Run All Tests

```bash
# Set environment variables
export PROXY_BASE_URL="http://localhost:8000"
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-google-key"  # Optional
export ANTHROPIC_API_KEY="your-anthropic-key"  # Optional
export ELEVENLABS_API_KEY="your-elevenlabs-key"  # Optional

# Run tests
python test_client.py
```

The test suite includes:
- Agent run (non-streaming)
- Agent run (streaming)
- Agent with web search tool
- Agent with binary messages (images)
- Text-to-speech
- Text-to-speech streaming
- Speech-to-text transcription
- Fallback behavior validation

### Example Usage

See `example_usage.py` for a simple demonstration of all features:

```bash
python example_usage.py
```

## Development

```bash
# Install in development mode
pip install -e .

# Run tests
python test_client.py

# Try example usage
python example_usage.py
```

## License

MIT License

