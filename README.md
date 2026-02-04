# LiveLLM Python Client

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python client library for the LiveLLM Server - a unified proxy for AI agent, audio, and transcription services.

## Features

- 🚀 **Async-first** - Built on httpx and websockets for high-performance operations
- 🔒 **Type-safe** - Full type hints and Pydantic validation
- 🎯 **Multi-provider** - OpenAI, Google, Anthropic, Groq, ElevenLabs
- 🔄 **Streaming** - Real-time streaming for agent and audio
- 🛠️ **Flexible API** - Use request objects or keyword arguments
- 📋 **Structured Output** - Get validated JSON responses with schema support (Pydantic, OutputSchema, or dict)
- 📏 **Context Overflow Management** - Automatic handling of large texts with truncate/recycle strategies
- ⏱️ **Per-Request Timeout** - Override default timeout for individual requests
- 🎙️ **Audio services** - Text-to-speech and transcription
- 🎤 **Real-Time Transcription** - WebSocket-based live audio transcription with bidirectional streaming
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

# With default timeout and pre-configured providers
client = LivellmClient(
    base_url="http://localhost:8000",
    timeout=30.0,  # Default timeout for all requests
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

### Per-Request Timeout Override

The timeout provided in `__init__` is the default, but you can override it for individual requests:

```python
# Client with 30s default timeout
client = LivellmClient(base_url="http://localhost:8000", timeout=30.0)

# Uses default 30s timeout
response = await client.agent_run(
    provider_uid="openai",
    model="gpt-4",
    messages=[TextMessage(role="user", content="Hello")]
)

# Override with 120s timeout for this specific request
response = await client.agent_run(
    provider_uid="openai",
    model="gpt-4",
    messages=[TextMessage(role="user", content="Write a long essay...")],
    timeout=120.0  # Override for this request only
)

# Works with streaming too
async for chunk in client.agent_run_stream(
    provider_uid="openai",
    model="gpt-4",
    messages=[TextMessage(role="user", content="Tell me a story")],
    timeout=300.0  # 5 minutes for streaming
):
    print(chunk.output, end="")

# Works with all methods: speak(), speak_stream(), transcribe(), etc.
audio = await client.speak(
    provider_uid="openai",
    model="tts-1",
    text="Hello world",
    voice="alloy",
    mime_type=SpeakMimeType.MP3,
    sample_rate=24000,
    timeout=60.0
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

#### Agent with Conversation History

You can request the full conversation history (including tool calls and returns) by setting `include_history=True`:

```python
from livellm.models import TextMessage, ToolCallMessage, ToolReturnMessage

# Request with history enabled
response = await client.agent_run(
    provider_uid="openai",
    model="gpt-4",
    messages=[TextMessage(role="user", content="Search for latest AI news")],
    tools=[WebSearchInput(kind=ToolKind.WEB_SEARCH)],
    include_history=True  # Enable history in response
)

print(f"Output: {response.output}")

# Access full conversation history including tool interactions
if response.history:
    for msg in response.history:
        if isinstance(msg, TextMessage):
            print(f"{msg.role}: {msg.content}")
        elif isinstance(msg, ToolCallMessage):
            print(f"Tool Call: {msg.tool_name}({msg.args})")
        elif isinstance(msg, ToolReturnMessage):
            print(f"Tool Return from {msg.tool_name}: {msg.content}")
```

**History Message Types:**
- `TextMessage` - Regular text messages (user, model, system)
- `BinaryMessage` - Images or other binary content
- `ToolCallMessage` - Tool invocations made by the agent
  - `tool_name` - Name of the tool called
  - `args` - Arguments passed to the tool
- `ToolReturnMessage` - Results returned from tool calls
  - `tool_name` - Name of the tool that was called
  - `content` - The return value from the tool

**Use cases:**
- Debugging tool interactions
- Maintaining conversation state across multiple requests
- Auditing and logging complete conversations
- Building conversational UIs with full context visibility

#### Agent with Structured Output

Get structured JSON responses from the agent by providing an output schema. The agent will return a JSON string matching your schema in the `output` field.

**Three ways to define a schema:**

**1. Using Pydantic BaseModel (Recommended)**
```python
import json
from pydantic import BaseModel
from livellm.models import TextMessage

class Person(BaseModel):
    name: str
    age: int
    occupation: str

response = await client.agent_run(
    provider_uid="openai",
    model="gpt-4",
    messages=[TextMessage(role="user", content="Extract info: John is a 28-year-old engineer")],
    output_schema=Person  # Pass the BaseModel class directly
)

# response.output is a JSON string: '{"name": "John", "age": 28, "occupation": "engineer"}'
print(type(response.output))  # <class 'str'>

# Parse the JSON string yourself if needed
data = json.loads(response.output)
print(f"Name: {data['name']}")
print(f"Age: {data['age']}")
print(f"Occupation: {data['occupation']}")

# Or validate with your Pydantic model
person = Person.model_validate_json(response.output)
print(f"Name: {person.name}")
```

**2. Using OutputSchema**
```python
from livellm.models import OutputSchema, PropertyDef, TextMessage

schema = OutputSchema(
    title="Person",
    description="A person's information",
    properties={
        "name": PropertyDef(type="string", description="The person's name"),
        "age": PropertyDef(type="integer", minimum=0, maximum=150, description="Age in years"),
        "email": PropertyDef(type="string", pattern="^[^@]+@[^@]+\\.[^@]+$", description="Email address"),
    },
    required=["name", "age", "email"]
)

response = await client.agent_run(
    provider_uid="openai",
    model="gpt-4",
    messages=[TextMessage(role="user", content="Tell me about a person")],
    output_schema=schema
)
```

**3. Using a dictionary (JSON Schema)**
```python
schema_dict = {
    "title": "Person",
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "The person's name"},
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "email": {"type": "string", "pattern": "^[^@]+@[^@]+\\.[^@]+$"}
    },
    "required": ["name", "age", "email"]
}

response = await client.agent_run(
    provider_uid="openai",
    model="gpt-4",
    messages=[TextMessage(role="user", content="Extract person info")],
    output_schema=schema_dict
)
```

**Complex nested schemas:**
```python
from pydantic import BaseModel
from typing import List, Optional

class Address(BaseModel):
    street: str
    city: str
    zip_code: str

class Person(BaseModel):
    name: str
    age: int
    addresses: List[Address]
    phone: Optional[str] = None

response = await client.agent_run(
    provider_uid="openai",
    model="gpt-4",
    messages=[TextMessage(role="user", content="Extract person with addresses")],
    output_schema=Person  # Nested models are automatically resolved
)
```

**With streaming:**
```python
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    key_points: List[str]
    word_count: int

stream = client.agent_run_stream(
    provider_uid="openai",
    model="gpt-4",
    messages=[TextMessage(role="user", content="Summarize this article")],
    output_schema=Summary
)

async for chunk in stream:
    print(chunk.output, end="", flush=True)

# After streaming completes, parse the full JSON output
full_output = "".join([chunk.output async for chunk in stream])
data = json.loads(full_output)
```

**Response fields:**
- `output` - The JSON string response matching your schema

**Use cases:**
- Data extraction and parsing
- API response formatting
- Structured data generation
- Type-safe responses
- Integration with type-checked code

#### Context Overflow Management

Handle large texts that exceed model context windows with automatic truncation or iterative processing:

```python
from livellm.models import TextMessage, ContextOverflowStrategy, OutputSchema, PropertyDef

# TRUNCATE strategy (default): Preserves beginning, middle, and end
# Works with both streaming and non-streaming
response = await client.agent_run(
    provider_uid="openai",
    model="gpt-4",
    messages=[
        TextMessage(role="system", content="Summarize the document."),
        TextMessage(role="user", content=very_long_document)
    ],
    context_limit=4000,  # Max tokens
    context_overflow_strategy=ContextOverflowStrategy.TRUNCATE
)

# RECYCLE strategy: Iteratively processes chunks and merges results
# Useful for extraction tasks - processes entire document
# Requires output_schema for JSON merging
output_schema = OutputSchema(
    title="ExtractedInfo",
    properties={
        "topics": PropertyDef(type="array", items={"type": "string"}),
        "key_figures": PropertyDef(type="array", items={"type": "string"})
    },
    required=["topics", "key_figures"]
)

response = await client.agent_run(
    provider_uid="openai",
    model="gpt-4",
    messages=[
        TextMessage(role="system", content="Extract all topics and key figures."),
        TextMessage(role="user", content=very_long_document)
    ],
    context_limit=3000,
    context_overflow_strategy=ContextOverflowStrategy.RECYCLE,
    output_schema=output_schema
)

# Parse the merged results
import json
result = json.loads(response.output)
print(f"Topics: {result['topics']}")
print(f"Key figures: {result['key_figures']}")
```

**Strategy comparison:**

| Strategy | How it works | Best for | Streaming |
|----------|--------------|----------|-----------|
| `TRUNCATE` | Takes beginning, middle, end portions | Summarization, Q&A | ✅ Yes |
| `RECYCLE` | Processes chunks iteratively, merges JSON | Full document extraction | ❌ No |

**Parameters:**
- `context_limit` (int, default: 0) - Maximum tokens. If ≤ 0, overflow handling is disabled
- `context_overflow_strategy` (ContextOverflowStrategy, default: TRUNCATE) - Strategy to use

**Notes:**
- System prompts are always preserved (never truncated)
- Token counting includes a 20% safety buffer
- RECYCLE requires `output_schema` for JSON merging

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

### Real-Time Transcription (WebSocket)

The realtime transcription API is available either **directly** via `TranscriptionWsClient` or **through** `LivellmClient.realtime.transcription`.

#### Using `TranscriptionWsClient` directly

```python
import asyncio
from livellm import TranscriptionWsClient
from livellm.models import (
    TranscriptionInitWsRequest,
    TranscriptionAudioChunkWsRequest,
    SpeakMimeType,
)

async def transcribe_live_direct():
    base_url = "ws://localhost:8000"  # WebSocket base URL

    async with TranscriptionWsClient(base_url, timeout=30) as client:
        # Define audio source (file, microphone, stream, etc.)
        async def audio_source():
            with open("audio.pcm", "rb") as f:
                while chunk := f.read(4096):
                    yield TranscriptionAudioChunkWsRequest(audio=chunk)
                    await asyncio.sleep(0.1)  # Simulate real-time

        # Initialize transcription session
        init_request = TranscriptionInitWsRequest(
            provider_uid="openai",
            model="gpt-4o-mini-transcribe",
            language="en",  # or "auto" for detection
            input_sample_rate=24000,
            input_audio_format=SpeakMimeType.PCM,
            gen_config={},
        )

        # Stream audio and receive transcriptions
        # Each iteration yields a list of responses (oldest to newest)
        async for responses in client.start_session(init_request, audio_source()):
            # Get the latest transcription (last element)
            latest = responses[-1]
            print(f"Latest transcription: {latest.transcription}")
            
            # Process all accumulated transcriptions if needed
            if len(responses) > 1:
                print(f"  (received {len(responses)} chunks)")
                for resp in responses:
                    print(f"    - {resp.transcription}")

asyncio.run(transcribe_live_direct())
```

#### Using `LivellmClient.realtime.transcription` (and running agents while listening)

```python
import asyncio
from livellm import LivellmClient
from livellm.models import (
    TextMessage,
    TranscriptionInitWsRequest,
    TranscriptionAudioChunkWsRequest,
    SpeakMimeType,
)

async def transcribe_and_chat():
    # Central HTTP client; .realtime and .transcription expose WebSocket APIs
    client = LivellmClient(base_url="http://localhost:8000", timeout=30)

    async with client.realtime as realtime:
        async with realtime.transcription as t_client:
            async def audio_source():
                with open("audio.pcm", "rb") as f:
                    while chunk := f.read(4096):
                        yield TranscriptionAudioChunkWsRequest(audio=chunk)
                        await asyncio.sleep(0.1)

            init_request = TranscriptionInitWsRequest(
                provider_uid="openai",
                model="gpt-4o-mini-transcribe",
                language="en",
                input_sample_rate=24000,
                input_audio_format=SpeakMimeType.PCM,
                gen_config={},
            )

            # Listen for transcriptions and, for each batch, run an agent request
            # Each iteration yields a list of responses - newest is last
            async for responses in t_client.start_session(init_request, audio_source()):
                # Use the latest transcription for the agent
                latest = responses[-1]
                print("User said:", latest.transcription)

                # You can call agent_run (or speak, etc.) while the transcription stream is active
                # Even if this is slow, transcriptions accumulate and won't stall the loop
                agent_response = await realtime.agent_run(
                    provider_uid="openai",
                    model="gpt-4",
                    messages=[
                        TextMessage(role="user", content=latest.transcription),
                    ],
                    temperature=0.7,
                )
                print("Agent:", agent_response.output)

asyncio.run(transcribe_and_chat())
```

**Supported Audio Formats:**
- **PCM**: 16-bit uncompressed (recommended)
- **μ-law**: 8-bit telephony format (North America/Japan)
- **A-law**: 8-bit telephony format (Europe/rest of world)

**Use Cases:**
- 🎙️ Voice assistants and chatbots
- 📝 Live captioning and subtitles
- 🎤 Meeting transcription
- 🗣️ Voice commands and control

**See also:** 
- [TRANSCRIPTION_CLIENT.md](TRANSCRIPTION_CLIENT.md) - Complete transcription guide
- [example_transcription.py](example_transcription.py) - Python examples
- [example_transcription_browser.html](example_transcription_browser.html) - Browser demo

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

All methods accept an optional `timeout` parameter to override the default client timeout.

**Configuration**
- `ping(timeout?)` - Health check
- `update_config(config, timeout?)` / `update_configs(configs, timeout?)` - Add/update providers
- `get_configs(timeout?)` - List all configurations
- `delete_config(uid, timeout?)` - Remove provider

**Agent**
- `agent_run(request | **kwargs, timeout?)` - Run agent (blocking)
- `agent_run_stream(request | **kwargs, timeout?)` - Run agent (streaming)

**Audio**
- `speak(request | **kwargs, timeout?)` - Text-to-speech (blocking)
- `speak_stream(request | **kwargs, timeout?)` - Text-to-speech (streaming)
- `transcribe(request | **kwargs, timeout?)` - Speech-to-text

**Real-Time Transcription (TranscriptionWsClient)**
- `connect()` - Establish WebSocket connection
- `disconnect()` - Close WebSocket connection
- `start_session(init_request, audio_source)` - Start bidirectional streaming transcription; yields `list[TranscriptionWsResponse]` (accumulated responses, newest last)
- `async with client:` - Auto connection management (recommended)

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
- `ToolCallMessage(role, tool_name, args)` - Tool invocation by agent
- `ToolReturnMessage(role, tool_name, content)` - Tool execution result
- `MessageRole` - `USER` | `MODEL` | `SYSTEM` | `TOOL_CALL` | `TOOL_RETURN` (or use strings)

**Requests**
- `AgentRequest(provider_uid, model, messages, tools?, gen_config?, include_history?, output_schema?, context_limit?, context_overflow_strategy?)` - Set `include_history=True` to get full conversation. Set `output_schema` for structured JSON output. Set `context_limit` and `context_overflow_strategy` for handling large texts.
- `SpeakRequest(provider_uid, model, text, voice, mime_type, sample_rate, gen_config?)`
- `TranscribeRequest(provider_uid, file, model, language?, gen_config?)`
- `TranscriptionInitWsRequest(provider_uid, model, language?, input_sample_rate?, input_audio_format?, gen_config?)`
- `TranscriptionAudioChunkWsRequest(audio)` - Audio chunk for streaming

**Context Overflow**
- `ContextOverflowStrategy` - `TRUNCATE` | `RECYCLE`

**Tools**
- `WebSearchInput(kind=ToolKind.WEB_SEARCH, search_context_size)`
- `MCPStreamableServerInput(kind=ToolKind.MCP_STREAMABLE_SERVER, url, prefix?, timeout?)`

**Structured Output**
- `OutputSchema(title, description?, properties, required?, additionalProperties?)` - JSON Schema for structured output
- `PropertyDef(type, description?, enum?, default?, minLength?, maxLength?, pattern?, minimum?, maximum?, items?, ...)` - Property definition with validation constraints
- `OutputSchema.from_pydantic(model)` - Convert a Pydantic BaseModel class to OutputSchema

**Fallback**
- `AgentFallbackRequest(strategy, requests, timeout_per_request?)`
- `AudioFallbackRequest(strategy, requests, timeout_per_request?)`
- `FallbackStrategy` - `SEQUENTIAL` | `PARALLEL`

**Responses**
- `AgentResponse(output, usage{input_tokens, output_tokens}, history?)` - `history` included when `include_history=True`. `output` is a JSON string when `output_schema` is provided.
- `TranscribeResponse(text, language)`
- `TranscriptionWsResponse(transcription, received_at)` - Real-time transcription result; yielded as `list[TranscriptionWsResponse]` with newest last

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
- websockets >= 15.0.1

## Documentation

- [README.md](README.md) - Main documentation (you are here)
- [TRANSCRIPTION_CLIENT.md](TRANSCRIPTION_CLIENT.md) - Complete real-time transcription guide
- [CLIENT_EXAMPLES.md](CLIENT_EXAMPLES.md) - Usage examples for all features
- [example_transcription.py](example_transcription.py) - Python transcription examples
- [example_transcription_browser.html](example_transcription_browser.html) - Browser demo

## Links

- [GitHub Repository](https://github.com/qalby-tech/livellm-client-py)
- [Issue Tracker](https://github.com/qalby-tech/livellm-client-py/issues)

## License

MIT License - see LICENSE file for details.
