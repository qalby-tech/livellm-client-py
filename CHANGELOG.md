# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2025-11-05

### Added
- **Flexible API**: All methods now support two calling styles
  - **Keyword arguments style**: Pass parameters directly as kwargs (automatically converted to `gen_config`)
  - **Request object style**: Traditional approach using request objects
- Method overloading with proper type hints for both calling styles
- Automatic `gen_config` handling from kwargs for cleaner API usage

### Changed
- `agent_run()` now accepts both `AgentRequest` object or individual kwargs (`provider_uid`, `model`, `messages`, `tools`, `**kwargs`)
- `agent_run_stream()` now accepts both `AgentRequest` object or individual kwargs
- `speak()` now accepts both `SpeakRequest` object or individual kwargs
- `speak_stream()` now accepts both `SpeakRequest` object or individual kwargs
- `transcribe()` now accepts both `TranscribeRequest` object or individual kwargs
- All kwargs (like `temperature`, `max_tokens`, `speed`, etc.) are automatically converted to `gen_config`

### Improved
- Better developer experience with more intuitive API calls
- Enhanced type safety with `@overload` decorators
- More concise code examples in documentation
- Updated README with clear examples of both calling styles

### Example
```python
# Old style (still works)
response = await client.agent_run(
    AgentRequest(
        provider_uid="openai",
        model="gpt-4",
        messages=[...],
        gen_config={"temperature": 0.7}
    )
)

# New style (simpler)
response = await client.agent_run(
    provider_uid="openai",
    model="gpt-4",
    messages=[...],
    temperature=0.7
)
```

## [1.1.1] - 2025-11-04

### Added
- Initial release of LiveLLM Python Client
- Async client with full support for all LiveLLM endpoints
- Support for agent services (text and streaming)
- Support for audio services (text-to-speech and transcription)
- Fallback strategies for high availability (sequential and parallel)
- Context manager support for automatic resource cleanup
- Health check endpoint

### Models
- Agent models: `AgentRequest`, `AgentResponse`, `AgentResponseUsage`, `AgentFallbackRequest`
- Audio models: `SpeakRequest`, `SpeakMimeType`, `TranscribeRequest`, `TranscribeResponse`, `AudioFallbackRequest`, `TranscribeFallbackRequest`
- Message models: `TextMessage`, `BinaryMessage`, `MessageRole`
- Tool models: `WebSearchInput`, `MCPStreamableServerInput`, `ToolKind`
- Provider models: `Settings`, `ProviderKind`
- Common models: `BaseRequest`, `SuccessResponse`, `FallbackStrategy`

### Features
- Comprehensive error handling with detailed error messages
- Full type safety with Pydantic validation
- HTTP/2 support via httpx
- Configurable timeouts
- Multipart file upload support for transcription
- Base64 encoded file support for JSON transcription
- Streaming support for agent responses and text-to-speech
- Multiple provider configuration management

[1.3.0]: https://github.com/qalby-tech/livellm-client-py/releases/tag/v1.3.0
[1.2.0]: https://github.com/qalby-tech/livellm-client-py/releases/tag/v1.2.0

