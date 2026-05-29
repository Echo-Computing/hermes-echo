"""Unit tests for OllamaClient"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from hermes_cli.tools.ollama_client import OllamaClient


class TestOllamaClient:
    """OllamaClientのテスト"""

    @pytest.fixture
    def ollama_client(self, test_config):
        """OllamaClientインスタンス"""
        return OllamaClient(
            api_url=test_config.ollama.api_url,
            model=test_config.ollama.model,
            temperature=test_config.ollama.temperature,
            max_tokens=test_config.ollama.max_tokens,
            timeout=test_config.ollama.timeout,
            retry=test_config.ollama.retry,
        )

    @pytest.mark.asyncio
    async def test_init(self, ollama_client, test_config):
        """初期化テスト"""
        assert ollama_client.api_url == test_config.ollama.api_url
        assert ollama_client.model == test_config.ollama.model
        assert ollama_client.temperature == test_config.ollama.temperature
        assert ollama_client.max_tokens == test_config.ollama.max_tokens

    @pytest.mark.asyncio
    async def test_chat_success(self, ollama_client, mock_ollama_response):
        """正常なチャットテスト"""
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json = MagicMock(return_value=mock_ollama_response)
            mock_post.return_value = mock_response

            result = await ollama_client.chat("Test prompt")

            assert result is not None
            assert isinstance(result, str)
            assert result == mock_ollama_response["message"]["content"]
            mock_post.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_with_retry(self, ollama_client):
        """リトライ機能テスト"""
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response_fail = AsyncMock()
            mock_response_fail.status_code = 500
            mock_response_fail.raise_for_status = AsyncMock(
                side_effect=httpx.HTTPStatusError("Error", request=MagicMock(), response=MagicMock())
            )

            mock_response_success = AsyncMock()
            mock_response_success.status_code = 200
            mock_response_success.json = MagicMock(
                return_value={
                    "message": {"content": "Success after retry"},
                    "done": True,
                }
            )

            mock_post.side_effect = [
                mock_response_fail,
                mock_response_fail,
                mock_response_success,
            ]

            result = await ollama_client.chat("Test prompt")

            assert result is not None
            assert result == "Success after retry"
            assert mock_post.call_count == 3

    @pytest.mark.asyncio
    async def test_chat_all_retries_failed(self, ollama_client):
        """全リトライ失敗テスト"""
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status_code = 500
            mock_response.raise_for_status = AsyncMock(
                side_effect=httpx.HTTPStatusError("Error", request=MagicMock(), response=MagicMock())
            )
            mock_post.return_value = mock_response

            with pytest.raises(Exception):
                await ollama_client.chat("Test prompt")

            assert mock_post.call_count == ollama_client.retry
