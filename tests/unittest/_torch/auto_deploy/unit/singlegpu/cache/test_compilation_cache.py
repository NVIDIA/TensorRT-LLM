"""Tests for the simplified export cache module."""

import tempfile
from pathlib import Path

from tensorrt_llm._torch.auto_deploy.cache import CacheKey, ExportCacheConfig, GraphSerializer


class TestCacheKey:
    """Tests for CacheKey."""

    def test_from_model_config(self):
        """Test creating a cache key from model config."""
        model = "meta-llama/Llama-2-7b"
        config = {"strict": False, "patch_list": None}

        key = CacheKey.from_model_config(model, config)

        assert key.model_id == model
        assert len(key.config_hash) == 16

    def test_to_cache_path(self):
        """Test generating cache path."""
        key = CacheKey(model_id="test/model", config_hash="abc123")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            path = key.to_cache_path(cache_dir)

            assert "model" in str(path)
            assert path.parent == cache_dir

    def test_cache_key_deterministic(self):
        """Test that same config produces same key."""
        config = {"strict": False, "model_kwargs": {"dtype": "bfloat16"}}

        key1 = CacheKey.from_model_config("model", config)
        key2 = CacheKey.from_model_config("model", config)

        assert key1.config_hash == key2.config_hash

    def test_different_config_different_key(self):
        """Test that different configs produce different keys."""
        config1 = {"strict": False}
        config2 = {"strict": True}

        key1 = CacheKey.from_model_config("model", config1)
        key2 = CacheKey.from_model_config("model", config2)

        assert key1.config_hash != key2.config_hash


class TestExportCacheConfig:
    """Tests for ExportCacheConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ExportCacheConfig()

        assert config.enabled is True
        assert config.cache_dir is not None

    def test_custom_cache_dir(self):
        """Test custom cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExportCacheConfig(cache_dir=tmpdir)
            assert str(config.cache_dir) == tmpdir

    def test_disabled_cache(self):
        """Test disabled cache."""
        config = ExportCacheConfig(enabled=False)
        assert config.enabled is False


class TestGraphSerializer:
    """Tests for GraphSerializer."""

    def test_is_valid_cache_empty_dir(self):
        """Test validation with empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assert GraphSerializer.is_valid_cache(Path(tmpdir)) is False

    def test_is_valid_cache_nonexistent(self):
        """Test validation with non-existent directory."""
        assert GraphSerializer.is_valid_cache(Path("/nonexistent")) is False
