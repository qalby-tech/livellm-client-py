"""
Example: Using MCP (Model Context Protocol) Tools with Agents

This example demonstrates how to use MCP tools with the LivellmProxy client.
MCP tools allow agents to interact with external services and resources through
streamable servers.

MCP tools are useful for:
- File system operations
- Database queries
- API integrations
- Custom tool implementations
- Real-time data access
"""

import asyncio
from livellm import (
    LivellmProxy,
    Creds,
    TextMessage,
    MCPStreamableServerInput,
    WebSearchInput,
    create_openai_provider_config,
)


async def demonstrate_mcp_tools():
    """
    Demonstrate how to use MCP tools with agents.
    """
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
    
    print("=== MCP Tools Example ===")
    
    # Example 1: Using MCP Streamable Server Tool
    print("\n1. Using MCP Streamable Server Tool")
    
    # Create MCP tool configuration
    mcp_tool = MCPStreamableServerInput(
        url="http://localhost:3000",  # Your MCP server URL
        prefix="filesystem"  # Tool prefix/namespace
    )
    
    try:
        response, messages = await proxy.agent_run(
            model="gpt-4o",
            messages=[
                TextMessage(
                    role="user", 
                    content="List the files in the current directory using the filesystem tool."
                ),
            ],
            tools=[mcp_tool],
        )
        
        print(f"Response: {response.output}")
        print(f"Usage: {response.usage}")
        
    except Exception as e:
        print(f"MCP Tool Error: {e}")
        print("Note: Make sure your MCP server is running at http://localhost:3000")
    
    # Example 2: Using Multiple Tools (MCP + Web Search)
    print("\n2. Using Multiple Tools (MCP + Web Search)")
    
    # Create multiple tools
    mcp_tool = MCPStreamableServerInput(
        url="http://localhost:3000",
        prefix="database"
    )
    
    web_search_tool = WebSearchInput(
        search_context_size="high"  # Get more context from web search
    )
    
    try:
        response, messages = await proxy.agent_run(
            model="gpt-4o",
            messages=[
                TextMessage(
                    role="user", 
                    content="Search for the latest AI news and then save the results to the database using the available tools."
                ),
            ],
            tools=[mcp_tool, web_search_tool],
        )
        
        print(f"Response: {response.output}")
        
    except Exception as e:
        print(f"Multiple Tools Error: {e}")
    
    # Example 3: Streaming with MCP Tools
    print("\n3. Streaming with MCP Tools")
    
    mcp_tool = MCPStreamableServerInput(
        url="http://localhost:3000",
        prefix="api"
    )
    
    try:
        stream, messages = await proxy.agent_run_stream(
            model="gpt-4o",
            messages=[
                TextMessage(
                    role="user", 
                    content="Use the API tool to fetch data and explain what you found."
                ),
            ],
            tools=[mcp_tool],
        )
        
        print("Streaming response:")
        async for chunk in stream:
            print(chunk.output, end="", flush=True)
        print()  # New line after streaming
        
    except Exception as e:
        print(f"Streaming MCP Error: {e}")


async def demonstrate_custom_mcp_configurations():
    """
    Demonstrate different MCP tool configurations for various use cases.
    """
    print("\n=== Custom MCP Configurations ===")
    
    # Example configurations for different MCP servers
    mcp_configs = [
        # File system operations
        MCPStreamableServerInput(
            url="http://localhost:3001",
            prefix="filesystem"
        ),
        
        # Database operations
        MCPStreamableServerInput(
            url="http://localhost:3002", 
            prefix="database"
        ),
        
        # API integrations
        MCPStreamableServerInput(
            url="http://localhost:3003",
            prefix="external_api"
        ),
        
        # Custom business logic
        MCPStreamableServerInput(
            url="http://localhost:3004",
            prefix="business_tools"
        ),
    ]
    
    print("Available MCP tool configurations:")
    for i, config in enumerate(mcp_configs, 1):
        print(f"{i}. {config.prefix} server at {config.url}")
    
    # Example of using multiple MCP tools together
    print("\nExample: Using multiple MCP tools in one request")
    
    # This would be used in a real scenario
    example_tools = [
        MCPStreamableServerInput(url="http://localhost:3001", prefix="filesystem"),
        MCPStreamableServerInput(url="http://localhost:3002", prefix="database"),
        WebSearchInput(search_context_size="medium"),
    ]
    
    print("Tools configured:")
    for tool in example_tools:
        if hasattr(tool, 'prefix'):
            print(f"- MCP Tool: {tool.prefix} at {tool.url}")
        else:
            print(f"- Web Search Tool: {tool.search_context_size} context")


def explain_mcp_benefits():
    """
    Explain the benefits and use cases of MCP tools.
    """
    print("\n=== Why Use MCP Tools? ===")
    print("""
MCP (Model Context Protocol) tools provide several key benefits:

1. **Extensibility**: Add custom functionality to your agents
   - File system operations
   - Database queries
   - API integrations
   - Custom business logic

2. **Real-time Data Access**: Connect to live data sources
   - Current market data
   - Live system metrics
   - Real-time user information

3. **Modularity**: Separate tool logic from agent logic
   - Tools can be developed independently
   - Easy to add/remove functionality
   - Reusable across different agents

4. **Streaming Support**: Handle long-running operations
   - Large file processing
   - Complex database queries
   - Real-time data streams

5. **Security**: Controlled access to resources
   - Sandboxed tool execution
   - Permission-based access
   - Audit trails

Common Use Cases:
- File management and processing
- Database operations and queries
- External API integrations
- Custom business workflows
- Real-time monitoring and alerts
- Data analysis and reporting
""")


if __name__ == "__main__":
    print("MCP Tools Example")
    print("=" * 50)
    
    # Run the examples
    asyncio.run(demonstrate_mcp_tools())
    asyncio.run(demonstrate_custom_mcp_configurations())
    
    # Explain benefits
    explain_mcp_benefits()
    
    print("\n" + "=" * 50)
    print("Example completed!")
    print("\nTo use MCP tools:")
    print("1. Set up an MCP server (e.g., using the MCP SDK)")
    print("2. Configure the server URL and prefix")
    print("3. Add the MCPStreamableServerInput to your tools list")
    print("4. The agent will automatically use the tools when needed")
