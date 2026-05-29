"""Unit tests for SearxNGClient"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import json

from hermes_cli.tools.container_use_client import SearxNGClient


class TestSearxNGClient:
    """SearxNGClientのテスト"""

    @pytest.fixture
    def searxng_client(self, test_config):
        """SearxNGClientインスタンス"""
        # Mock redis.from_url to avoid connecting to real Redis
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()
        mock_redis.close = AsyncMock()

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            client = SearxNGClient(
                searxng_url=test_config.search.searxng_base_url,
                redis_url=test_config.search.redis_url,
                cache_ttl=test_config.search.cache_ttl,
            )
        return client

    @pytest.mark.asyncio
    async def test_init(self, searxng_client, test_config):
        """初期化テスト"""
        assert searxng_client.searxng_url == test_config.search.searxng_base_url
        assert searxng_client.cache_ttl == test_config.search.cache_ttl

    @pytest.mark.asyncio
    async def test_search_success(self, searxng_client, mock_search_response):
        """正常な検索テスト"""
        with patch("httpx.AsyncClient.get") as mock_get:
            # モックレスポンス設定
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json = MagicMock(return_value=mock_search_response)
            mock_get.return_value = mock_response

            # テスト実行
            result = await searxng_client.search("test query")

            # 検証
            assert result is not None
            assert len(result.results) == 2
            assert result.results[0].title == "Test Result 1"
            assert result.results[1].title == "Test Result 2"
            mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_cache(self, searxng_client, mock_search_response):
        """キャッシュヒット時の検索テスト"""
        # Set up redis mock to return cached data
        cached_json = json.dumps({
            "query": "test query",
            "results": [
                {"title": "Test Result 1", "url": "https://example.com/1",
                 "snippet": "Test content 1", "engine": "google", "score": None},
                {"title": "Test Result 2", "url": "https://example.com/2",
                 "snippet": "Test content 2", "engine": "bing", "score": None},
            ],
            "total_results": 2,
            "search_time": 0.0,
            "cached": False,
        })
        searxng_client.redis_client.get.return_value = cached_json

        with patch("httpx.AsyncClient.get") as mock_get:
            result = await searxng_client.search("test query")

            assert result is not None
            assert len(result.results) == 2
            # キャッシュヒットしたのでHTTPリクエストは呼ばれない
            mock_get.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_failure(self, searxng_client):
        """検索失敗テスト"""
        searxng_client.redis_client.get.return_value = None

        with patch("httpx.AsyncClient.get") as mock_get:
            mock_get.side_effect = Exception("Connection error")

            with pytest.raises(Exception):
                await searxng_client.search("test query")

            mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_empty_results(self, searxng_client):
        """空の検索結果テスト"""
        searxng_client.redis_client.get.return_value = None

        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json = MagicMock(return_value={"query": "test", "results": []})
            mock_get.return_value = mock_response

            result = await searxng_client.search("test query")

            assert len(result.results) == 0
            mock_get.assert_called_once()
