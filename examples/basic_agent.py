"""
Basic Agent Run Example

This example demonstrates how to use the LivellmProxy client for basic text conversations.
"""

import asyncio
from livellm import (
    LivellmProxy,
    Creds,
    TextMessage,
    create_google_provider_config,
    ProviderConfig,
    Model,
    ModelCapability,
)


async def main():
    """Run a basic agent conversation."""
    # Configuration - Update these with your actual credentials
    BASE_URL = "http://localhost:8000"  # Your proxy server URL
    API_KEY = "your-api-key-here"  # Your API key
    OPENAI_BASE_URL = "https://api.openai.com/v1"  # OpenAI API URL
    GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com"  # Google API URL
    
    # Create OpenAI provider config
    openai_config = ProviderConfig(
        creds=Creds(
            api_key=API_KEY,
            provider="openai",
            base_url=OPENAI_BASE_URL
        ),
        models=[
            Model(
                name="gpt-4o",
                capabilities=[ModelCapability.TEXT_AGENT, ModelCapability.IMAGE_AGENT]
            ),
        ]
    )
    
    # Initialize the proxy client
    proxy = LivellmProxy(
        base_url=BASE_URL,
        providers=[
            openai_config,
            create_google_provider_config(API_KEY, base_url=GOOGLE_BASE_URL),
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
    print(f"Usage: {response.usage}")
    print(f"Messages count: {len(messages)}")


if __name__ == "__main__":
    asyncio.run(main())
