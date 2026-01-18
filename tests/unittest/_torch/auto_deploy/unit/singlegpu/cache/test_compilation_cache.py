"""Unit tests for the AutoDeploy compilation cache module."""

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
import torch.nn as nn
from torch.fx import GraphModule, symbolic_trace

from tensorrt_llm._torch.auto_deploy.cache.cache_key import (
    CacheKey,
    _extract_graph_affecting_config,
)
from tensorrt_llm._torch.auto_deploy.cache.cache_manager import (
    CompilationCacheConfig,
    CompilationCacheManager,
)
from tensorrt_llm._torch.auto_deploy.cache.graph_serializer import GraphSerializer

# ================================================================================================
# Test Fixtures
# ================================================================================================


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache tests."""
    temp_dir = tempfile.mkdtemp(prefix="autodeploy_cache_test_")
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_transforms_config() -> Dict[str, Any]:
    """Sample transforms configuration for testing."""
    return {
        "build_model": {
            "stage": "factory",
            "enabled": True,
            "device": "meta",
        },
        "export_to_gm": {
            "stage": "export",
            "enabled": True,
            "strict": False,
            "clone_state_dict": False,
        },
        "match_attention": {
            "stage": "pattern_matcher",
            "enabled": True,
            "skip_on_error": False,  # Should be filtered out
        },
        "disabled_transform": {
            "stage": "pattern_matcher",
            "enabled": False,
        },
    }


@pytest.fixture
def simple_graph_module() -> GraphModule:
    """Create a simple GraphModule for testing serialization."""

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)
            self.register_buffer("scale", torch.ones(5))

        def forward(self, x):
            return self.linear(x) * self.scale

    model = SimpleModel()
    gm = symbolic_trace(model)
    gm.meta["_autodeploy"] = {"transform_history": {"test": "data"}}
    return gm


# ================================================================================================
# CacheKey Tests
# ================================================================================================


class TestCacheKey:
    """Tests for CacheKey class."""

    def test_cache_key_creation(self, sample_transforms_config):
        """Test basic cache key creation."""
        key = CacheKey.from_config(
            model="meta-llama/Llama-3.1-8B",
            transforms_config=sample_transforms_config,
            world_size=4,
            local_rank=0,
        )

        assert key.model_id == "meta-llama/Llama-3.1-8B"
        assert key.world_size == 4
        assert key.local_rank == 0
        assert len(key.transforms_config_hash) == 12
        assert "disabled_transform" not in key.enabled_transforms
        assert "build_model" in key.enabled_transforms

    def test_cache_key_deterministic(self, sample_transforms_config):
        """Test that cache keys are deterministic."""
        key1 = CacheKey.from_config(
            model="test-model",
            transforms_config=sample_transforms_config,
            world_size=1,
            local_rank=0,
        )
        key2 = CacheKey.from_config(
            model="test-model",
            transforms_config=sample_transforms_config,
            world_size=1,
            local_rank=0,
        )

        assert key1.transforms_config_hash == key2.transforms_config_hash

    def test_cache_key_different_configs(self, sample_transforms_config):
        """Test that different configs produce different keys."""
        key1 = CacheKey.from_config(
            model="test-model",
            transforms_config=sample_transforms_config,
            world_size=1,
            local_rank=0,
        )

        # Modify config
        modified_config = sample_transforms_config.copy()
        modified_config["export_to_gm"]["strict"] = True

        key2 = CacheKey.from_config(
            model="test-model",
            transforms_config=modified_config,
            world_size=1,
            local_rank=0,
        )

        assert key1.transforms_config_hash != key2.transforms_config_hash

    def test_cache_key_different_world_size(self, sample_transforms_config):
        """Test that different world sizes produce different cache paths."""
        key1 = CacheKey.from_config(
            model="test-model",
            transforms_config=sample_transforms_config,
            world_size=1,
            local_rank=0,
        )
        key2 = CacheKey.from_config(
            model="test-model",
            transforms_config=sample_transforms_config,
            world_size=4,
            local_rank=0,
        )

        cache_dir = Path("/tmp/cache")
        assert key1.to_cache_path(cache_dir) != key2.to_cache_path(cache_dir)

    def test_cache_key_to_cache_path(self, temp_cache_dir, sample_transforms_config):
        """Test cache path generation."""
        key = CacheKey.from_config(
            model="meta-llama/Llama-3.1-8B-Instruct",
            transforms_config=sample_transforms_config,
            world_size=1,
            local_rank=0,
        )

        cache_path = key.to_cache_path(temp_cache_dir)

        assert cache_path.parent == temp_cache_dir
        assert "Llama-3" in cache_path.name or "Llama_3" in cache_path.name

    def test_cache_key_to_dict_roundtrip(self, sample_transforms_config):
        """Test serialization roundtrip."""
        key = CacheKey.from_config(
            model="test-model",
            transforms_config=sample_transforms_config,
            world_size=2,
            local_rank=1,
        )

        key_dict = key.to_dict()
        restored_key = CacheKey.from_dict(key_dict)

        assert restored_key.model_id == key.model_id
        assert restored_key.transforms_config_hash == key.transforms_config_hash
        assert restored_key.world_size == key.world_size
        assert restored_key.local_rank == key.local_rank


class TestExtractGraphAffectingConfig:
    """Tests for config filtering function."""

    def test_filters_non_graph_affecting_keys(self):
        """Test that non-graph-affecting keys are filtered."""
        config = {
            "stage": "pattern_matcher",
            "enabled": True,
            "strict": True,
            "skip_on_error": False,  # Should be filtered
            "run_graph_cleanup": True,  # Should be filtered
        }

        filtered = _extract_graph_affecting_config(config)

        assert "stage" in filtered
        assert "enabled" in filtered
        assert "strict" in filtered
        assert "skip_on_error" not in filtered
        assert "run_graph_cleanup" not in filtered


# ================================================================================================
# GraphSerializer Tests
# ================================================================================================


class TestGraphSerializer:
    """Tests for GraphSerializer class."""

    def test_save_and_load(self, temp_cache_dir, simple_graph_module):
        """Test basic save and load functionality."""
        cache_path = temp_cache_dir / "test_cache"
        metadata = {"test_key": "test_value", "cache_version": "1.0"}

        # Save
        GraphSerializer.save(simple_graph_module, cache_path, metadata)

        # Verify files exist
        assert (cache_path / GraphSerializer.GRAPH_FILE).exists()
        assert (cache_path / GraphSerializer.METADATA_FILE).exists()
        assert (cache_path / GraphSerializer.CODE_FILE).exists()

        # Load
        loaded_gm, loaded_metadata = GraphSerializer.load(cache_path)

        assert loaded_metadata["test_key"] == "test_value"
        assert isinstance(loaded_gm, GraphModule)

    def test_save_preserves_param_shapes(self, temp_cache_dir, simple_graph_module):
        """Test that parameter shapes are preserved."""
        cache_path = temp_cache_dir / "test_cache"

        # Get original shapes
        orig_shapes = {name: param.shape for name, param in simple_graph_module.named_parameters()}

        # Save and load
        GraphSerializer.save(simple_graph_module, cache_path, {})
        loaded_gm, _ = GraphSerializer.load(cache_path, device="meta")

        # Verify shapes
        for name, param in loaded_gm.named_parameters():
            assert param.shape == orig_shapes[name], f"Shape mismatch for {name}"

    def test_save_preserves_buffer_shapes(self, temp_cache_dir, simple_graph_module):
        """Test that buffer shapes are preserved."""
        cache_path = temp_cache_dir / "test_cache"

        # Get original buffer shapes
        orig_shapes = {name: buf.shape for name, buf in simple_graph_module.named_buffers()}

        # Save and load
        GraphSerializer.save(simple_graph_module, cache_path, {})
        loaded_gm, _ = GraphSerializer.load(cache_path, device="meta")

        # Verify shapes
        for name, buf in loaded_gm.named_buffers():
            assert buf.shape == orig_shapes[name], f"Shape mismatch for buffer {name}"

    def test_is_valid_cache(self, temp_cache_dir, simple_graph_module):
        """Test cache validation."""
        cache_path = temp_cache_dir / "test_cache"

        # Invalid before save
        assert not GraphSerializer.is_valid_cache(cache_path)

        # Valid after save
        GraphSerializer.save(simple_graph_module, cache_path, {})
        assert GraphSerializer.is_valid_cache(cache_path)

        # Invalid if graph file is deleted
        (cache_path / GraphSerializer.GRAPH_FILE).unlink()
        assert not GraphSerializer.is_valid_cache(cache_path)

    def test_load_to_meta_device(self, temp_cache_dir, simple_graph_module):
        """Test loading to meta device."""
        cache_path = temp_cache_dir / "test_cache"

        GraphSerializer.save(simple_graph_module, cache_path, {})
        loaded_gm, _ = GraphSerializer.load(cache_path, device="meta")

        for param in loaded_gm.parameters():
            assert param.device.type == "meta"


# ================================================================================================
# CompilationCacheManager Tests
# ================================================================================================


class TestCompilationCacheConfig:
    """Tests for CompilationCacheConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = CompilationCacheConfig()

        assert config.enabled is True
        assert config.save_cache is True
        assert config.load_cache is True
        assert config.max_cache_size_gb == 50.0

    def test_custom_cache_dir(self, temp_cache_dir):
        """Test custom cache directory."""
        config = CompilationCacheConfig(cache_dir=str(temp_cache_dir))

        assert config.cache_dir == temp_cache_dir

    def test_env_var_expansion(self, temp_cache_dir, monkeypatch):
        """Test environment variable expansion in cache_dir."""
        monkeypatch.setenv("TEST_CACHE_DIR", str(temp_cache_dir))

        config = CompilationCacheConfig(cache_dir="$TEST_CACHE_DIR/compilation")

        assert str(temp_cache_dir) in str(config.cache_dir)


class TestCompilationCacheManager:
    """Tests for CompilationCacheManager class."""

    def test_manager_initialization(self, temp_cache_dir):
        """Test manager initialization."""
        config = CompilationCacheConfig(cache_dir=str(temp_cache_dir))
        manager = CompilationCacheManager(config)

        assert manager.config == config
        assert temp_cache_dir.exists()

    def test_set_cache_key(self, temp_cache_dir, sample_transforms_config):
        """Test setting cache key."""
        config = CompilationCacheConfig(cache_dir=str(temp_cache_dir))
        manager = CompilationCacheManager(config)

        key = CacheKey.from_config(
            model="test-model",
            transforms_config=sample_transforms_config,
            world_size=1,
            local_rank=0,
        )
        manager.set_cache_key(key)

        assert manager.get_cache_path() is not None

    def test_has_valid_cache_empty(self, temp_cache_dir, sample_transforms_config):
        """Test has_valid_cache with no cache."""
        config = CompilationCacheConfig(cache_dir=str(temp_cache_dir))
        manager = CompilationCacheManager(config)

        key = CacheKey.from_config(
            model="test-model",
            transforms_config=sample_transforms_config,
            world_size=1,
            local_rank=0,
        )
        manager.set_cache_key(key)

        assert not manager.has_valid_cache()

    def test_save_and_load_graph(
        self, temp_cache_dir, sample_transforms_config, simple_graph_module
    ):
        """Test saving and loading a graph through the manager."""
        config = CompilationCacheConfig(cache_dir=str(temp_cache_dir))
        manager = CompilationCacheManager(config)

        key = CacheKey.from_config(
            model="test-model",
            transforms_config=sample_transforms_config,
            world_size=1,
            local_rank=0,
        )
        manager.set_cache_key(key)

        # Save
        manager.save_graph_to_cache(
            gm=simple_graph_module,
            transform_history={"test": "history"},
            cached_stage="pattern_matcher",
        )

        # Verify cache exists
        assert manager.has_valid_cache()

        # Load
        result = manager.load_cached_graph()
        assert result is not None

        loaded_gm, metadata = result
        assert isinstance(loaded_gm, GraphModule)
        assert metadata["cached_stage"] == "pattern_matcher"

    def test_invalidate_cache(self, temp_cache_dir, sample_transforms_config, simple_graph_module):
        """Test cache invalidation."""
        config = CompilationCacheConfig(cache_dir=str(temp_cache_dir))
        manager = CompilationCacheManager(config)

        key = CacheKey.from_config(
            model="test-model",
            transforms_config=sample_transforms_config,
            world_size=1,
            local_rank=0,
        )
        manager.set_cache_key(key)

        # Save then invalidate
        manager.save_graph_to_cache(
            gm=simple_graph_module,
            transform_history={},
            cached_stage="export",
        )
        assert manager.has_valid_cache()

        manager.invalidate_cache()
        assert not manager.has_valid_cache()

    def test_list_caches(self, temp_cache_dir, sample_transforms_config, simple_graph_module):
        """Test listing cached compilations."""
        config = CompilationCacheConfig(cache_dir=str(temp_cache_dir))
        manager = CompilationCacheManager(config)

        # Create multiple caches
        for i in range(3):
            key = CacheKey.from_config(
                model=f"test-model-{i}",
                transforms_config=sample_transforms_config,
                world_size=1,
                local_rank=0,
            )
            manager.set_cache_key(key)
            manager.save_graph_to_cache(
                gm=simple_graph_module,
                transform_history={},
                cached_stage="export",
            )

        # List caches
        caches = manager.list_caches()
        assert len(caches) == 3

    def test_disabled_cache_no_save(
        self, temp_cache_dir, sample_transforms_config, simple_graph_module
    ):
        """Test that disabled cache doesn't save."""
        config = CompilationCacheConfig(
            cache_dir=str(temp_cache_dir),
            save_cache=False,
        )
        manager = CompilationCacheManager(config)

        key = CacheKey.from_config(
            model="test-model",
            transforms_config=sample_transforms_config,
            world_size=1,
            local_rank=0,
        )
        manager.set_cache_key(key)

        manager.save_graph_to_cache(
            gm=simple_graph_module,
            transform_history={},
            cached_stage="export",
        )

        # Should not have saved
        assert not manager.has_valid_cache()

    def test_disabled_cache_no_load(
        self, temp_cache_dir, sample_transforms_config, simple_graph_module
    ):
        """Test that disabled load doesn't load."""
        # First save with enabled cache
        config1 = CompilationCacheConfig(cache_dir=str(temp_cache_dir))
        manager1 = CompilationCacheManager(config1)

        key = CacheKey.from_config(
            model="test-model",
            transforms_config=sample_transforms_config,
            world_size=1,
            local_rank=0,
        )
        manager1.set_cache_key(key)
        manager1.save_graph_to_cache(
            gm=simple_graph_module,
            transform_history={},
            cached_stage="export",
        )

        # Now try to load with load disabled
        config2 = CompilationCacheConfig(
            cache_dir=str(temp_cache_dir),
            load_cache=False,
        )
        manager2 = CompilationCacheManager(config2)
        manager2.set_cache_key(key)

        # has_valid_cache should return False when load is disabled
        assert not manager2.has_valid_cache()


# ================================================================================================
# Integration Tests
# ================================================================================================


class TestCacheIntegration:
    """Integration tests for the cache system."""

    def test_cache_key_consistency_across_sessions(self, temp_cache_dir, sample_transforms_config):
        """Test that cache keys are consistent across different sessions."""
        # Simulate two separate sessions
        key1 = CacheKey.from_config(
            model="meta-llama/Llama-3.1-8B",
            transforms_config=sample_transforms_config,
            world_size=4,
            local_rank=2,
        )

        key2 = CacheKey.from_config(
            model="meta-llama/Llama-3.1-8B",
            transforms_config=sample_transforms_config,
            world_size=4,
            local_rank=2,
        )

        assert key1.to_cache_path(temp_cache_dir) == key2.to_cache_path(temp_cache_dir)

    def test_full_save_load_cycle(
        self, temp_cache_dir, sample_transforms_config, simple_graph_module
    ):
        """Test a full save/load cycle simulating actual usage."""
        # Session 1: Save
        config1 = CompilationCacheConfig(cache_dir=str(temp_cache_dir))
        manager1 = CompilationCacheManager(config1)

        key = CacheKey.from_config(
            model="test-model",
            transforms_config=sample_transforms_config,
            world_size=1,
            local_rank=0,
        )
        manager1.set_cache_key(key)

        transform_history = {
            "build_model": {"skipped": False, "num_matches": 1},
            "export_to_gm": {"skipped": False, "num_matches": 1},
        }

        manager1.save_graph_to_cache(
            gm=simple_graph_module,
            transform_history=transform_history,
            cached_stage="pattern_matcher",
        )

        # Session 2: Load
        config2 = CompilationCacheConfig(cache_dir=str(temp_cache_dir))
        manager2 = CompilationCacheManager(config2)
        manager2.set_cache_key(key)

        assert manager2.has_valid_cache()

        result = manager2.load_cached_graph()
        assert result is not None

        loaded_gm, metadata = result
        assert metadata["cached_stage"] == "pattern_matcher"
        assert "transform_history" in metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
