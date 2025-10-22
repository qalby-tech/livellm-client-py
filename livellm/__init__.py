"""LiveLLM Client - Python client for the LiveLLM Proxy and Realtime APIs."""

from .proxy import (
    # Main clients
    LivellmProxy,
    LivellmProxyClient,
    
    # Message models
    MessageRole,
    TextMessage,
    BinaryMessage,
    
    # Tool models
    ToolKind,
    WebSearchInput,
    MCPStreamableServerInput,
    
    # Agent models
    AgentRequest,
    AgentResponseUsage,
    AgentResponse,
    
    # Audio models
    SpeakRequest,
    TranscribeResponse,
    
    # Error models
    ValidationError,
    HTTPValidationError,
    
    # Provider models
    Creds,
    ModelCapability,
    Model,
    ProviderConfig,
    
    # Utility functions
    create_openai_provider_config,
    create_google_provider_config,
    create_elevenlabs_provider_config,
    create_anthropic_provider_config,
)

__version__ = "0.1.0"

__all__ = [
    # Main clients
    "LivellmProxy",
    "LivellmProxyClient",
    
    # Message models
    "MessageRole",
    "TextMessage",
    "BinaryMessage",
    
    # Tool models
    "ToolKind",
    "WebSearchInput",
    "MCPStreamableServerInput",
    
    # Agent models
    "AgentRequest",
    "AgentResponseUsage",
    "AgentResponse",
    
    # Audio models
    "SpeakRequest",
    "TranscribeResponse",
    
    # Error models
    "ValidationError",
    "HTTPValidationError",
    
    # Provider models
    "Creds",
    "ModelCapability",
    "Model",
    "ProviderConfig",
    
    # Utility functions
    "create_openai_provider_config",
    "create_google_provider_config",
    "create_elevenlabs_provider_config",
    "create_anthropic_provider_config",
    
    # Version
    "__version__",
]
