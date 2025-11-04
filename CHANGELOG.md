# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-11-04

### Added
- Initial release of LiveLLM Python Client
- Async client with full support for all LiveLLM endpoints

### Models
- Agent models: AgentRequest, AgentResponse, AgentFallbackRequest
- Audio models: SpeakRequest, SpeakResponse, TranscribeRequest, TranscribeResponse
- Message models: TextMessage, BinaryMessage
- Tool models: WebSearchInput, MCPStreamableServerInput
- Provider models: Settings, ProviderKind
- Common models: BaseRequest, SuccessResponse, FallbackStrategy

### Features
- Comprehensive error handling
- Full type safety with Pydantic validation

[1.1.0]: https://github.com/qalby-tech/livellm-client-py/releases/tag/v1.1.0

