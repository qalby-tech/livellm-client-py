"""
Raw Client Example

This example demonstrates how to use the raw LivellmProxyClient for direct API access.
"""

import asyncio
from livellm import LivellmProxyClient


async def main():
    """Test raw client health check endpoint."""
    # Configuration - Update these with your actual credentials
    BASE_URL = "http://localhost:8000"  # Your proxy server URL
    
    # Initialize the raw client
    client = LivellmProxyClient(base_url=BASE_URL)
    
    # Test health check
    response = await client.ping()
    
    print(f"Ping response: {response}")


if __name__ == "__main__":
    asyncio.run(main())
