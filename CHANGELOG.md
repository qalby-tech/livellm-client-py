# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-11-04

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

[1.1.0]: https://github.com/qalby-tech/livellm-client-py/releases/tag/v1.1.0

