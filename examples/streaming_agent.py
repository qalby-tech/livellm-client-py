"""
Streaming Agent Example

This example demonstrates how to use the LivellmProxy client for streaming responses.
"""

import asyncio
from livellm import (
    LivellmProxy,
    Creds,
    TextMessage,
    create_openai_provider_config,
)


async def main():
    """Run a streaming agent conversation."""
    # Configuration - Update these with your actual credentials
    BASE_URL = "http://localhost:8000"  # Your proxy server URL
    API_KEY = "your-api-key-here"  # Your API key
    OPENAI_BASE_URL = "https://api.openai.com/v1"  # OpenAI API URL
    
    # Initialize the proxy client
    proxy = LivellmProxy(
        base_url=BASE_URL,
        providers=[
            create_openai_provider_config(API_KEY, base_url=OPENAI_BASE_URL),
        ]
    )
    
    # Run a streaming conversation
    stream, messages = await proxy.agent_run_stream(
        model="gpt-4o",
        messages=[
            TextMessage(role="user", content="Count from 1 to 5 slowly."),
        ],
        tools=[],
    )
    
    # Process streaming chunks
    chunks = []
    async for chunk in stream:
        print(f"Chunk: {chunk.output}", end="", flush=True)
        chunks.append(chunk)
    
    print()
    print(f"Total chunks received: {len(chunks)}")
    print(f"Messages count: {len(messages)}")


if __name__ == "__main__":
    asyncio.run(main())
