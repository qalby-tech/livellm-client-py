"""
Smoke tests for LiveLLM Client.

These tests verify basic functionality of the client library.
They use a mock HTTP server to simulate the LiveLLM server responses.
"""

import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import json
import base64

from livellm import LivellmClient, LivellmWsClient, TranscriptionWsClient
from livellm.models import (
    Settings,
    ProviderKind,
    SuccessResponse,
    AgentRequest,
    AgentResponse,
    AgentResponseUsage,
    TextMessage,
    BinaryMessage,
    ToolCallMessage,
    ToolReturnMessage,
    MessageRole,
    SpeakRequest,
    SpeakMimeType,
    TranscribeRequest,
    TranscribeResponse,
    WebSearchInput,
    MCPStreamableServerInput,
    ToolKind,
    AgentFallbackRequest,
    AudioFallbackRequest,
    TranscribeFallbackRequest,
    FallbackStrategy,
)
from pydantic import SecretStr


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx.AsyncClient."""
    client = AsyncMock(spec=httpx.AsyncClient)
    return client


@pytest.fixture
async def client(mock_httpx_client):
    """Create a LivellmClient with mocked httpx client."""
    with patch("livellm.livellm.httpx.AsyncClient", return_value=mock_httpx_client):
        client = LivellmClient(base_url="http://localhost:8000")
        client.client = mock_httpx_client
        yield client


class TestClientInitialization:
    """Test client initialization."""

    def test_client_init_basic(self):
        """Test basic client initialization."""
        client = LivellmClient(base_url="http://localhost:8000")
        assert client.base_url == "http://localhost:8000/livellm"
        assert client.timeout is None

    def test_client_init_with_timeout(self):
        """Test client initialization with timeout."""
        client = LivellmClient(base_url="http://localhost:8000", timeout=30.0)
        assert client.timeout == 30.0

    def test_client_init_strips_trailing_slash(self):
        """Test that trailing slash is removed from base_url."""
        client = LivellmClient(base_url="http://localhost:8000/")
        assert client.base_url == "http://localhost:8000/livellm"

    @patch("livellm.livellm.httpx.Client")
    def test_client_init_with_configs(self, mock_sync_client):
        """Test client initialization with provider configs."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_sync_client.return_value.__enter__.return_value.post.return_value = mock_response

        configs = [
            Settings(
                uid="test-config",
                provider=ProviderKind.OPENAI,
                api_key=SecretStr("test-key"),
            )
        ]

        client = LivellmClient(base_url="http://localhost:8000", configs=configs)
        assert len(client.settings) == 1
        assert client.settings[0].uid == "test-config"


class TestHealthCheck:
    """Test health check endpoints."""

    @pytest.mark.asyncio
    async def test_ping(self, client, mock_httpx_client):
        """Test ping endpoint."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"success": True, "message": "pong"}
        mock_httpx_client.get.return_value = mock_response

        result = await client.ping()

        assert isinstance(result, SuccessResponse)
        assert result.success is True
        mock_httpx_client.get.assert_called_once_with("ping", headers=client.headers, timeout=None)


class TestConfigurationManagement:
    """Test provider configuration management."""

    @pytest.mark.asyncio
    async def test_update_config(self, client, mock_httpx_client):
        """Test updating a single provider configuration."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "message": "Config updated"}
        mock_httpx_client.post.return_value = mock_response

        config = Settings(
            uid="test-openai",
            provider=ProviderKind.OPENAI,
            api_key=SecretStr("sk-test"),
        )

        result = await client.update_config(config)

        assert isinstance(result, SuccessResponse)
        assert result.success is True
        assert len(client.settings) == 1
        mock_httpx_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_configs(self, client, mock_httpx_client):
        """Test updating multiple provider configurations."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "message": "Config updated"}
        mock_httpx_client.post.return_value = mock_response

        configs = [
            Settings(uid="config-1", provider=ProviderKind.OPENAI, api_key=SecretStr("key1")),
            Settings(uid="config-2", provider=ProviderKind.ANTHROPIC, api_key=SecretStr("key2")),
        ]

        result = await client.update_configs(configs)

        assert isinstance(result, SuccessResponse)
        assert len(client.settings) == 2
        assert mock_httpx_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_get_configs(self, client, mock_httpx_client):
        """Test retrieving all configurations."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = [
            {
                "uid": "test-config",
                "provider": "openai",
                "api_key": "sk-test",
                "base_url": None,
                "blacklist_models": None,
            }
        ]
        mock_httpx_client.get.return_value = mock_response

        result = await client.get_configs()

        assert isinstance(result, list)
        mock_httpx_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_config(self, client, mock_httpx_client):
        """Test deleting a configuration."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"success": True, "message": "Config deleted"}
        mock_httpx_client.delete.return_value = mock_response

        result = await client.delete_config("test-config")

        assert isinstance(result, SuccessResponse)
        assert result.success is True
        mock_httpx_client.delete.assert_called_once()


class TestAgentServices:
    """Test agent-related functionality."""

    @pytest.mark.asyncio
    async def test_agent_run(self, client, mock_httpx_client):
        """Test basic agent run."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output": "Hello! How can I help you?",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        mock_httpx_client.post.return_value = mock_response

        request = AgentRequest(
            provider_uid="test-provider",
            model="gpt-4",
            messages=[TextMessage(role=MessageRole.USER, content="Hello")],
            tools=[],
        )

        result = await client.agent_run(request)

        assert isinstance(result, AgentResponse)
        assert result.output == "Hello! How can I help you?"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 20
        mock_httpx_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_run_with_binary_message(self, client, mock_httpx_client):
        """Test agent run with binary message."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output": "This is an image of a cat.",
            "usage": {"input_tokens": 100, "output_tokens": 10},
        }
        mock_httpx_client.post.return_value = mock_response

        image_data = base64.b64encode(b"fake-image-data").decode("utf-8")

        request = AgentRequest(
            provider_uid="test-provider",
            model="gpt-4-vision",
            messages=[
                BinaryMessage(
                    role=MessageRole.USER,
                    content=image_data,
                    mime_type="image/jpeg",
                    caption="What's in this image?",
                )
            ],
            tools=[],
        )

        result = await client.agent_run(request)

        assert isinstance(result, AgentResponse)
        assert "image" in result.output.lower() or "cat" in result.output.lower()

    @pytest.mark.asyncio
    async def test_agent_run_with_web_search(self, client, mock_httpx_client):
        """Test agent run with web search tool."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output": "According to recent news...",
            "usage": {"input_tokens": 50, "output_tokens": 100},
        }
        mock_httpx_client.post.return_value = mock_response

        request = AgentRequest(
            provider_uid="test-provider",
            model="gpt-4",
            messages=[TextMessage(role=MessageRole.USER, content="Latest AI news?")],
            tools=[WebSearchInput(kind=ToolKind.WEB_SEARCH, search_context_size="high")],
        )

        result = await client.agent_run(request)

        assert isinstance(result, AgentResponse)

    @pytest.mark.asyncio
    async def test_agent_run_with_mcp_tool(self, client, mock_httpx_client):
        """Test agent run with MCP server tool."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output": "Tool executed successfully",
            "usage": {"input_tokens": 30, "output_tokens": 15},
        }
        mock_httpx_client.post.return_value = mock_response

        request = AgentRequest(
            provider_uid="test-provider",
            model="gpt-4",
            messages=[TextMessage(role=MessageRole.USER, content="Execute tool")],
            tools=[
                MCPStreamableServerInput(
                    kind=ToolKind.MCP_STREAMABLE_SERVER,
                    url="http://mcp-server:8080",
                    prefix="mcp_",
                    timeout=15,
                )
            ],
        )

        result = await client.agent_run(request)

        assert isinstance(result, AgentResponse)

    @pytest.mark.asyncio
    async def test_agent_run_with_history(self, client, mock_httpx_client):
        """Test agent run with conversation history."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output": "Based on my search, here's what I found...",
            "usage": {"input_tokens": 50, "output_tokens": 100},
            "history": [
                {"role": "user", "content": "Search for AI news"},
                {"role": "tool_call", "tool_name": "web_search", "args": {"query": "AI news"}},
                {"role": "tool_return", "tool_name": "web_search", "content": "AI news results..."},
                {"role": "model", "content": "Based on my search, here's what I found..."},
            ],
        }
        mock_httpx_client.post.return_value = mock_response

        request = AgentRequest(
            provider_uid="test-provider",
            model="gpt-4",
            messages=[TextMessage(role=MessageRole.USER, content="Search for AI news")],
            tools=[WebSearchInput(kind=ToolKind.WEB_SEARCH, search_context_size="high")],
            include_history=True,  # Enable history
        )

        result = await client.agent_run(request)

        assert isinstance(result, AgentResponse)
        assert result.output == "Based on my search, here's what I found..."
        assert result.history is not None
        assert len(result.history) == 4
        
        # Verify message types in history
        assert isinstance(result.history[0], TextMessage)
        assert result.history[0].role == "user"
        assert isinstance(result.history[1], ToolCallMessage)
        assert result.history[1].tool_name == "web_search"
        assert isinstance(result.history[2], ToolReturnMessage)
        assert result.history[2].tool_name == "web_search"
        assert isinstance(result.history[3], TextMessage)
        assert result.history[3].role == "model"

    @pytest.mark.asyncio
    async def test_agent_run_stream(self, client, mock_httpx_client):
        """Test streaming agent run."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        async def mock_aiter_lines():
            yield json.dumps({"output": "Hello ", "usage": {"input_tokens": 5, "output_tokens": 2}})
            yield json.dumps({"output": "World!", "usage": {"input_tokens": 5, "output_tokens": 2}})

        mock_response.aiter_lines = mock_aiter_lines
        mock_httpx_client.post.return_value = mock_response

        request = AgentRequest(
            provider_uid="test-provider",
            model="gpt-4",
            messages=[TextMessage(role=MessageRole.USER, content="Hello")],
            tools=[],
        )

        stream = client.agent_run_stream(request)
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].output == "Hello "
        assert chunks[1].output == "World!"


class TestAudioServices:
    """Test audio-related functionality."""

    @pytest.mark.asyncio
    async def test_speak(self, client, mock_httpx_client):
        """Test text-to-speech."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake-audio-data"
        mock_httpx_client.post.return_value = mock_response

        request = SpeakRequest(
            provider_uid="elevenlabs-config",
            model="eleven_turbo_v2",
            text="Hello world",
            voice="rachel",
            mime_type=SpeakMimeType.MP3,
            sample_rate=44100,
        )

        result = await client.speak(request)

        assert isinstance(result, bytes)
        assert result == b"fake-audio-data"

    @pytest.mark.asyncio
    async def test_speak_stream(self, client, mock_httpx_client):
        """Test streaming text-to-speech."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        async def mock_aiter_bytes():
            yield b"chunk1"
            yield b"chunk2"
            yield b"chunk3"

        mock_response.aiter_bytes = mock_aiter_bytes
        mock_httpx_client.post.return_value = mock_response

        request = SpeakRequest(
            provider_uid="elevenlabs-config",
            model="eleven_turbo_v2",
            text="Hello world",
            voice="rachel",
            mime_type=SpeakMimeType.MP3,
            sample_rate=44100,
        )

        stream = client.speak_stream(request)
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0] == b"chunk1"

    @pytest.mark.asyncio
    async def test_transcribe_json(self, client, mock_httpx_client):
        """Test audio transcription with JSON."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "text": "This is transcribed text.",
            "language": "en",
        }
        mock_httpx_client.post.return_value = mock_response

        audio_data = base64.b64encode(b"fake-audio-data").decode("utf-8")

        request = TranscribeRequest(
            provider_uid="openai-config",
            model="whisper-1",
            file=("audio.mp3", audio_data, "audio/mpeg"),
        )

        result = await client.transcribe(request)

        assert isinstance(result, TranscribeResponse)
        assert result.text == "This is transcribed text."


class TestFallbackStrategies:
    """Test fallback functionality."""

    @pytest.mark.asyncio
    async def test_agent_fallback_sequential(self, client, mock_httpx_client):
        """Test sequential fallback for agent."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output": "Response from fallback",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        mock_httpx_client.post.return_value = mock_response

        messages = [TextMessage(role=MessageRole.USER, content="Hello")]

        fallback_request = AgentFallbackRequest(
            requests=[
                AgentRequest(provider_uid="provider-1", model="gpt-4", messages=messages, tools=[]),
                AgentRequest(provider_uid="provider-2", model="claude-3", messages=messages, tools=[]),
            ],
            strategy=FallbackStrategy.SEQUENTIAL,
            timeout_per_request=30,
        )

        result = await client.agent_run(fallback_request)

        assert isinstance(result, AgentResponse)

    @pytest.mark.asyncio
    async def test_audio_fallback_sequential(self, client, mock_httpx_client):
        """Test sequential fallback for audio."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"audio-from-fallback"
        mock_httpx_client.post.return_value = mock_response

        fallback_request = AudioFallbackRequest(
            requests=[
                SpeakRequest(
                    provider_uid="provider-1",
                    model="model-1",
                    text="Hello",
                    voice="voice1",
                    mime_type=SpeakMimeType.MP3,
                    sample_rate=44100,
                ),
                SpeakRequest(
                    provider_uid="provider-2",
                    model="model-2",
                    text="Hello",
                    voice="voice2",
                    mime_type=SpeakMimeType.MP3,
                    sample_rate=44100,
                ),
            ],
            strategy=FallbackStrategy.SEQUENTIAL,
        )

        result = await client.speak(fallback_request)

        assert isinstance(result, bytes)



class TestRealtimeClients:
    """Test realtime WebSocket client helpers."""

    def test_realtime_property_returns_ws_client(self):
        """LivellmClient.realtime should lazily create a LivellmWsClient."""
        client = LivellmClient(base_url="http://localhost:8000")
        ws_client = client.realtime

        assert isinstance(ws_client, LivellmWsClient)

    def test_transcription_property_returns_transcription_client(self):
        """LivellmWsClient.transcription should lazily create a TranscriptionWsClient."""
        client = LivellmClient(base_url="http://localhost:8000")
        ws_client = client.realtime
        transcription_client = ws_client.transcription

        assert isinstance(transcription_client, TranscriptionWsClient)



class TestErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_http_error_handling(self, client, mock_httpx_client):
        """Test that HTTP errors are raised properly."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.aread = AsyncMock(return_value=b"Internal Server Error")
        mock_httpx_client.post.return_value = mock_response

        request = AgentRequest(
            provider_uid="test-provider",
            model="gpt-4",
            messages=[TextMessage(role=MessageRole.USER, content="Hello")],
            tools=[],
        )

        with pytest.raises(Exception) as exc_info:
            await client.agent_run(request)

        assert "Failed to post" in str(exc_info.value)


class TestModels:
    """Test model validation."""

    def test_settings_serialization(self):
        """Test Settings model serialization."""
        config = Settings(
            uid="test-uid",
            provider=ProviderKind.OPENAI,
            api_key=SecretStr("secret-key"),
            base_url="https://api.openai.com",
            blacklist_models=["deprecated-model"],
        )

        serialized = config.model_dump()

        assert serialized["uid"] == "test-uid"
        assert serialized["provider"] == "openai"
        assert serialized["api_key"] == "secret-key"
        assert serialized["base_url"] == "https://api.openai.com"
        assert "deprecated-model" in serialized["blacklist_models"]

    def test_text_message_creation(self):
        """Test TextMessage creation."""
        message = TextMessage(role=MessageRole.USER, content="Hello")

        assert message.role == MessageRole.USER
        assert message.content == "Hello"

    def test_text_message_with_string_role(self):
        """Test TextMessage creation with string role."""
        # Test with string role
        message = TextMessage(role="user", content="Hello")
        
        assert message.role == MessageRole.USER
        assert message.content == "Hello"
        
        # Test serialization to ensure it's JSON serializable
        serialized = message.model_dump()
        assert serialized["role"] == "user"
        assert serialized["content"] == "Hello"
        
        # Test with different roles
        message_model = TextMessage(role="model", content="Hi there")
        assert message_model.role == MessageRole.MODEL
        
        message_system = TextMessage(role="system", content="System message")
        assert message_system.role == MessageRole.SYSTEM

    def test_binary_message_validation(self):
        """Test BinaryMessage validation."""
        # Should work with USER role
        message = BinaryMessage(
            role=MessageRole.USER,
            content="base64-encoded-data",
            mime_type="image/jpeg",
            caption="Test image",
        )
        assert message.role == MessageRole.USER

        # Should fail with MODEL role
        with pytest.raises(ValueError):
            BinaryMessage(
                role=MessageRole.MODEL,
                content="base64-encoded-data",
                mime_type="image/jpeg",
            )

    def test_transcribe_request_base64_decoding(self):
        """Test TranscribeRequest base64 decoding."""
        audio_data = base64.b64encode(b"test-audio").decode("utf-8")

        request = TranscribeRequest(
            provider_uid="test-provider",
            model="whisper-1",
            file=("test.mp3", audio_data, "audio/mpeg"),
        )

        # The validator should have decoded the base64 content
        filename, content, content_type = request.file
        assert filename == "test.mp3"
        assert content_type == "audio/mpeg"

    def test_web_search_input_validation(self):
        """Test WebSearchInput validation."""
        tool = WebSearchInput(kind=ToolKind.WEB_SEARCH, search_context_size="high")

        assert tool.kind == ToolKind.WEB_SEARCH
        assert tool.search_context_size == "high"

        # Should reject wrong kind
        with pytest.raises(ValueError):
            WebSearchInput(kind=ToolKind.MCP_STREAMABLE_SERVER, search_context_size="high")

    def test_mcp_server_input_validation(self):
        """Test MCPStreamableServerInput validation."""
        tool = MCPStreamableServerInput(
            kind=ToolKind.MCP_STREAMABLE_SERVER,
            url="http://localhost:8080",
            prefix="mcp_",
            timeout=30,
        )

        assert tool.kind == ToolKind.MCP_STREAMABLE_SERVER
        assert tool.url == "http://localhost:8080"
        assert tool.prefix == "mcp_"
        assert tool.timeout == 30

        # Should reject wrong kind
        with pytest.raises(ValueError):
            MCPStreamableServerInput(
                kind=ToolKind.WEB_SEARCH,
                url="http://localhost:8080",
                prefix="mcp_",
            )

