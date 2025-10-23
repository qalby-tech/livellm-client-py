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
- [`custom_provider_config.py`](examples/custom_provider_config.py) - Custom provider configurations
- [`mcp_tools_example.py`](examples/mcp_tools_example.py) - Using MCP tools with agents

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
    providers=[
        openai_provider,
        google_provider,
        # Add more providers as needed
    ]
)
```

### Custom Provider Configurations

The client supports custom provider configurations that allow you to define your own models, capabilities, and endpoints. This is particularly useful for:

#### Why Use Custom Configurations?

1. **Cost Optimization**: Only include models you actually use, reducing initialization overhead
2. **Performance**: Faster startup times with fewer models to process
3. **Self-Hosted Models**: Use custom base URLs for your own model deployments
4. **Capability Matching**: Define exactly which models support which features
5. **Fallback Reliability**: Automatic failover between providers when models are unavailable

#### How It Works

The client uses a fallback system that:
- Tries the primary provider first
- Automatically falls back to alternative providers if the model fails or doesn't support the input
- Transforms binary messages (audio/images) when models don't support them natively
- Heuristically checks model capabilities before making requests

#### Creating Custom Configurations

```python
from livellm import ProviderConfig, Creds, Model, ModelCapability

def create_custom_provider_config(api_key: str, base_url: str):
    """Create a custom provider configuration."""
    return ProviderConfig(
        creds=Creds(
            api_key=api_key,
            provider="custom",  # Your provider name
            base_url=base_url
        ),
        models=[
            # Define only the models you need
            Model(
                name="my-custom-model",
                capabilities=[ModelCapability.IMAGE_AGENT]  # Supports images
            ),
            Model(
                name="my-text-model",
                capabilities=[]  # Text-only model
            ),
        ]
    )

# Use in your client
proxy = LivellmProxy(
    base_url="http://localhost:8000",
    providers=[
        create_custom_provider_config("custom-key", "http://localhost:11434"),
        # Other providers...
    ]
)
```

#### Model Capabilities

Define what each model can handle:

```python
from livellm import ModelCapability

# Available capabilities:
ModelCapability.AUDIO_AGENT    # Can process audio messages
ModelCapability.IMAGE_AGENT    # Can process image messages  
ModelCapability.VIDEO_AGENT    # Can process video messages
ModelCapability.SPEAK          # Can generate speech (TTS)
ModelCapability.TRANSCRIBE     # Can transcribe audio (STT)
```

#### Self-Hosted Models

Use custom configurations for self-hosted models:

```python
def create_self_hosted_config(api_key: str, base_url: str):
    """Configuration for self-hosted models."""
    return ProviderConfig(
        creds=Creds(
            api_key=api_key,
            provider="self-hosted",
            base_url=base_url  # Your model server URL
        ),
        models=[
            Model(
                name="llama-3.1-8b",
                capabilities=[]  # Text-only
            ),
            Model(
                name="llava-1.6-7b", 
                capabilities=[ModelCapability.IMAGE_AGENT]
            ),
        ]
    )
```

#### Fallback Behavior Example

```python
# The client will automatically try different providers
response, messages = await proxy.agent_run(
    model="gpt-4o-mini",  # Try this model first
    messages=[
        TextMessage(role="user", content="Hello!")
    ],
    tools=[],
)
# If gpt-4o-mini fails, it will try other providers with the same model
# If the model doesn't exist in other providers, it will try similar models
```

For a complete example, see [`custom_provider_config.py`](examples/custom_provider_config.py).

### MCP Tools Integration

The client supports MCP (Model Context Protocol) tools, allowing agents to interact with external services and resources through streamable servers.

#### What are MCP Tools?

MCP tools enable agents to:
- **File System Operations**: Read, write, and manage files
- **Database Queries**: Execute database operations and queries
- **API Integrations**: Connect to external APIs and services
- **Custom Business Logic**: Implement domain-specific functionality
- **Real-time Data Access**: Access live data sources and streams

#### Using MCP Tools

```python
from livellm import MCPStreamableServerInput, WebSearchInput

# Create MCP tool configuration
mcp_tool = MCPStreamableServerInput(
    url="http://localhost:3000",  # Your MCP server URL
    prefix="filesystem"  # Tool namespace/prefix
)

# Use with agent
response, messages = await proxy.agent_run(
    model="gpt-4o",
    messages=[
        TextMessage(role="user", content="List files in the current directory")
    ],
    tools=[mcp_tool],
)
```

#### Multiple Tools Example

```python
# Combine MCP tools with other tools
mcp_tool = MCPStreamableServerInput(
    url="http://localhost:3000",
    prefix="database"
)

web_search_tool = WebSearchInput(
    search_context_size="high"
)

response, messages = await proxy.agent_run(
    model="gpt-4o",
    messages=[
        TextMessage(role="user", content="Search for AI news and save to database")
    ],
    tools=[mcp_tool, web_search_tool],
)
```

#### Streaming with MCP Tools

```python
# MCP tools work with streaming responses
stream, messages = await proxy.agent_run_stream(
    model="gpt-4o",
    messages=[
        TextMessage(role="user", content="Process data using the API tool")
    ],
    tools=[MCPStreamableServerInput(url="http://localhost:3000", prefix="api")],
)

async for chunk in stream:
    print(chunk.output, end="", flush=True)
```

#### Common MCP Tool Configurations

```python
# File system operations
filesystem_tool = MCPStreamableServerInput(
    url="http://localhost:3001",
    prefix="filesystem"
)

# Database operations
database_tool = MCPStreamableServerInput(
    url="http://localhost:3002",
    prefix="database"
)

```

#### Benefits of MCP Tools

1. **Extensibility**: Add custom functionality to your agents
2. **Real-time Data**: Connect to live data sources and APIs
3. **Modularity**: Separate tool logic from agent logic
4. **Streaming Support**: Handle long-running operations efficiently
5. **Security**: Controlled access to resources with audit trails

For a complete example with multiple use cases, see [`mcp_tools_example.py`](examples/mcp_tools_example.py).

## API Reference

### Core Methods

#### `agent_run(model, messages, tools, **kwargs)`
Run a conversation with the specified model.

**Parameters:**
- `model` (str): Model identifier (e.g., "gpt-4o", "claude-3")
- `messages` (List[Message]): List of conversation messages
- `tools` (List[Union[WebSearchInput, MCPStreamableServerInput]]): List of available tools
- `force_binary_transformation` (bool, optional): Force transformation of binary messages

**Returns:**
- `response`: Agent response with output and usage information
- `messages`: Processed message list

#### `agent_run_stream(model, messages, tools, **kwargs)`
Run a streaming conversation.

**Parameters:**
- `model` (str): Model identifier (e.g., "gpt-4o", "claude-3")
- `messages` (List[Message]): List of conversation messages
- `tools` (List[Union[WebSearchInput, MCPStreamableServerInput]]): List of available tools
- `force_binary_transformation` (bool, optional): Force transformation of binary messages

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
        BinaryMessage.from_bytes(audio_bytes, "audio/wav", "What's in this audio?")
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
