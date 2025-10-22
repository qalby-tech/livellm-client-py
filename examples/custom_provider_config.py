"""
Example: Custom Provider Configuration

This example demonstrates how to create custom provider configurations
for different AI providers and use cases. Custom configurations allow you to:

1. Define your own models with specific capabilities
2. Use custom base URLs for self-hosted models
3. Configure fallback providers for reliability
4. Optimize for specific use cases (cost, performance, features)
"""

import asyncio
from livellm import LivellmProxy, Creds, TextMessage, ProviderConfig, Model, ModelCapability


def create_custom_openai_config(api_key: str, base_url: str = None):
    """
    Create a custom OpenAI provider configuration.
    
    This example shows how to:
    - Define only the models you actually use
    - Set custom capabilities for specific models
    - Use a custom base URL for self-hosted OpenAI-compatible APIs
    """
    return ProviderConfig(
        creds=Creds(
            api_key=api_key,
            provider="openai",
            base_url=base_url or "https://api.openai.com/v1"
        ),
        models=[
            # Only include models you actually use to reduce overhead
            Model(
                name="gpt-4o-mini",
                capabilities=[ModelCapability.IMAGE_AGENT]  # Supports images
            ),
            Model(
                name="gpt-4o",
                capabilities=[ModelCapability.IMAGE_AGENT]  # Supports images
            ),
            # TTS models
            Model(
                name="tts-1",
                capabilities=[ModelCapability.SPEAK]
            ),
            # STT models
            Model(
                name="whisper-1",
                capabilities=[ModelCapability.TRANSCRIBE]
            ),
        ]
    )


def create_custom_anthropic_config(api_key: str, base_url: str = None):
    """
    Create a custom Anthropic provider configuration.
    
    This example shows how to:
    - Configure Claude models with specific capabilities
    - Use custom base URLs for self-hosted instances
    """
    return ProviderConfig(
        creds=Creds(
            api_key=api_key,
            provider="anthropic",
            base_url=base_url or "https://api.anthropic.com"
        ),
        models=[
            # Fast, cost-effective model for simple tasks
            Model(
                name="claude-haiku-4.5",
                capabilities=[]  # Text-only model
            ),
            # More capable model for complex reasoning
            Model(
                name="claude-sonnet-3.5",
                capabilities=[]  # Text-only model
            ),
            # Most capable model for complex tasks
            Model(
                name="claude-sonnet-4.0",
                capabilities=[]  # Text-only model
            ),
        ]
    )


def create_custom_google_config(api_key: str, base_url: str = None):
    """
    Create a custom Google provider configuration.
    
    This example shows how to:
    - Configure Gemini models with multimedia capabilities
    - Use different models for different use cases
    """
    # Define capabilities for different model tiers
    basic_caps = [ModelCapability.IMAGE_AGENT]  # Images only
    advanced_caps = [
        ModelCapability.IMAGE_AGENT,
        ModelCapability.VIDEO_AGENT,
        ModelCapability.AUDIO_AGENT
    ]
    
    return ProviderConfig(
        creds=Creds(
            api_key=api_key,
            provider="google",
            base_url=base_url or "https://generativelanguage.googleapis.com"
        ),
        models=[
            # Fast, lightweight model for simple tasks
            Model(
                name="gemini-2.5-flash-lite",
                capabilities=basic_caps
            ),
            # Balanced model for most use cases
            Model(
                name="gemini-2.5-flash",
                capabilities=advanced_caps
            ),
            # Most capable model for complex tasks
            Model(
                name="gemini-2.5-pro",
                capabilities=advanced_caps
            ),
        ]
    )


def create_self_hosted_config(api_key: str, base_url: str):
    """
    Create a configuration for a self-hosted model.
    
    This example shows how to:
    - Use custom base URLs for self-hosted models
    - Define custom model names
    - Set appropriate capabilities
    """
    return ProviderConfig(
        creds=Creds(
            api_key=api_key,
            provider="custom",  # Custom provider name
            base_url=base_url
        ),
        models=[
            # Self-hosted Llama model
            Model(
                name="llama-3.1-8b",
                capabilities=[]  # Text-only
            ),
            # Self-hosted multimodal model
            Model(
                name="llava-1.6-7b",
                capabilities=[ModelCapability.IMAGE_AGENT]
            ),
        ]
    )


async def demonstrate_custom_configs():
    """
    Demonstrate how to use custom provider configurations.
    """
    # Create custom provider configurations
    openai_config = create_custom_openai_config("your-openai-key")
    anthropic_config = create_custom_anthropic_config("your-anthropic-key")
    google_config = create_custom_google_config("your-google-key")
    
    # Optional: Add self-hosted models
    # self_hosted_config = create_self_hosted_config("your-key", "http://localhost:11434")
    
    # Initialize the proxy with custom configurations
    proxy = LivellmProxy(
        base_url="http://localhost:8000",  # Your proxy server URL
        primary_creds=Creds(
            api_key="your-primary-key",
            provider="openai",
            base_url="https://api.openai.com/v1"
        ),
        providers=[
            openai_config,
            anthropic_config,
            google_config,
            # self_hosted_config,  # Uncomment if using self-hosted models
        ]
    )
    
    # Test with different models
    print("Testing custom provider configurations...")
    
    # Test with OpenAI model
    try:
        response, messages = await proxy.agent_run(
            model="gpt-4o-mini",
            messages=[
                TextMessage(role="user", content="Hello! What can you do?")
            ],
            tools=[],
        )
        print(f"OpenAI Response: {response.output}")
    except Exception as e:
        print(f"OpenAI Error: {e}")
    
    # Test with Anthropic model
    try:
        response, messages = await proxy.agent_run(
            model="claude-haiku-4.5",
            messages=[
                TextMessage(role="user", content="Hello! What can you do?")
            ],
            tools=[],
        )
        print(f"Anthropic Response: {response.output}")
    except Exception as e:
        print(f"Anthropic Error: {e}")
    
    # Test with Google model
    try:
        response, messages = await proxy.agent_run(
            model="gemini-2.5-flash",
            messages=[
                TextMessage(role="user", content="Hello! What can you do?")
            ],
            tools=[],
        )
        print(f"Google Response: {response.output}")
    except Exception as e:
        print(f"Google Error: {e}")


async def demonstrate_fallback_behavior():
    """
    Demonstrate how fallback works with custom configurations.
    
    The client will automatically try different providers if:
    1. The primary provider fails
    2. The model doesn't support the input (e.g., audio/image messages)
    3. The model is not available
    """
    # Create configurations with different model availability
    openai_config = create_custom_openai_config("your-openai-key")
    anthropic_config = create_custom_anthropic_config("your-anthropic-key")
    
    proxy = LivellmProxy(
        base_url="http://localhost:8000",
        primary_creds=Creds(
            api_key="your-primary-key",
            provider="openai",
            base_url="https://api.openai.com/v1"
        ),
        providers=[openai_config, anthropic_config]
    )
    
    print("\nDemonstrating fallback behavior...")
    
    # This will try gpt-4o-mini first, then fallback to claude-haiku-4.5 if needed
    try:
        response, messages = await proxy.agent_run(
            model="gpt-4o-mini",  # This model is in both configs
            messages=[
                TextMessage(role="user", content="What's the weather like?")
            ],
            tools=[],
        )
        print(f"Fallback Response: {response.output}")
    except Exception as e:
        print(f"Fallback Error: {e}")


if __name__ == "__main__":
    print("Custom Provider Configuration Example")
    print("=" * 50)
    
    # Run the examples
    asyncio.run(demonstrate_custom_configs())
    asyncio.run(demonstrate_fallback_behavior())
    
    print("\n" + "=" * 50)
    print("Example completed!")
    print("\nKey benefits of custom provider configurations:")
    print("1. Cost optimization - only include models you use")
    print("2. Performance - faster initialization with fewer models")
    print("3. Flexibility - custom base URLs for self-hosted models")
    print("4. Reliability - automatic fallback between providers")
    print("5. Capability matching - models with right features for your use case")
