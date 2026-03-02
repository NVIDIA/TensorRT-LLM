# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the AutoDeploy graph transformation cache."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from tensorrt_llm._torch.auto_deploy.transform.graph_cache import (
    _GRAPH_FILENAME,
    _METADATA_FILENAME,
    GraphCache,
)


@pytest.fixture
def cache_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def cache(cache_dir):
    return GraphCache(cache_dir=str(cache_dir), max_entries=5, max_size_gb=1.0)


def _make_simple_module():
    """Create a small module on meta device for testing serialization."""
    with torch.device("meta"):
        mod = nn.Linear(16, 32)
    return mod


def _sample_config():
    return {
        "build_model": {"stage": "factory", "run_per_gm": False},
        "export_to_gm": {"stage": "export", "clone_state_dict": False},
        "match_rope_pattern": {"stage": "pattern_matcher", "run_shape_prop": True},
    }


# ---------------------------------------------------------------------------
# Environment variable gating
# ---------------------------------------------------------------------------


def test_is_enabled_default():
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("AD_ENABLE_CACHING", None)
        assert GraphCache.is_enabled() is False


def test_is_enabled_set_to_1():
    with patch.dict(os.environ, {"AD_ENABLE_CACHING": "1"}):
        assert GraphCache.is_enabled() is True


def test_is_enabled_set_to_0():
    with patch.dict(os.environ, {"AD_ENABLE_CACHING": "0"}):
        assert GraphCache.is_enabled() is False


# ---------------------------------------------------------------------------
# Cache key determinism
# ---------------------------------------------------------------------------


def test_cache_key_deterministic(cache):
    config = _sample_config()
    key1 = cache.compute_cache_key(
        pre_weight_config=config,
        model_id="meta-llama/Llama-2-7b",
        model_kwargs={},
        world_size=1,
        rank=0,
        max_seq_len=512,
        max_batch_size=8,
    )
    key2 = cache.compute_cache_key(
        pre_weight_config=config,
        model_id="meta-llama/Llama-2-7b",
        model_kwargs={},
        world_size=1,
        rank=0,
        max_seq_len=512,
        max_batch_size=8,
    )
    assert key1 == key2


def test_cache_key_differs_on_model(cache):
    config = _sample_config()
    kwargs = dict(
        pre_weight_config=config,
        model_kwargs={},
        world_size=1,
        rank=0,
        max_seq_len=512,
        max_batch_size=8,
    )
    key_a = cache.compute_cache_key(model_id="model-a", **kwargs)
    key_b = cache.compute_cache_key(model_id="model-b", **kwargs)
    assert key_a != key_b


def test_cache_key_differs_on_rank(cache):
    config = _sample_config()
    kwargs = dict(
        pre_weight_config=config,
        model_id="model",
        model_kwargs={},
        world_size=2,
        max_seq_len=512,
        max_batch_size=8,
    )
    key_r0 = cache.compute_cache_key(rank=0, **kwargs)
    key_r1 = cache.compute_cache_key(rank=1, **kwargs)
    assert key_r0 != key_r1


def test_cache_key_differs_on_seq_len(cache):
    config = _sample_config()
    kwargs = dict(
        pre_weight_config=config,
        model_id="model",
        model_kwargs={},
        world_size=1,
        rank=0,
        max_batch_size=8,
    )
    key_512 = cache.compute_cache_key(max_seq_len=512, **kwargs)
    key_1024 = cache.compute_cache_key(max_seq_len=1024, **kwargs)
    assert key_512 != key_1024


def test_cache_key_differs_on_config(cache):
    config_a = _sample_config()
    config_b = {**_sample_config(), "extra_transform": {"stage": "sharding"}}
    kwargs = dict(
        model_id="model",
        model_kwargs={},
        world_size=1,
        rank=0,
        max_seq_len=512,
        max_batch_size=8,
    )
    key_a = cache.compute_cache_key(pre_weight_config=config_a, **kwargs)
    key_b = cache.compute_cache_key(pre_weight_config=config_b, **kwargs)
    assert key_a != key_b


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


def test_save_and_load_roundtrip(cache, cache_dir):
    mod = _make_simple_module()
    key = "abc123"

    cache.save(key, mod)

    entry = cache_dir / key
    assert (entry / _GRAPH_FILENAME).exists()
    assert (entry / _METADATA_FILENAME).exists()

    loaded = cache.load(key)
    assert loaded is not None
    assert isinstance(loaded, nn.Module)


def test_load_returns_none_on_miss(cache):
    assert cache.load("nonexistent_key") is None


def test_metadata_timestamps_updated_on_load(cache, cache_dir):
    mod = _make_simple_module()
    key = "ts_test"
    cache.save(key, mod)

    meta_path = cache_dir / key / _METADATA_FILENAME
    with open(meta_path) as f:
        meta_before = json.load(f)

    import time

    time.sleep(0.05)
    cache.load(key)

    with open(meta_path) as f:
        meta_after = json.load(f)

    assert meta_after["last_accessed"] >= meta_before["last_accessed"]


# ---------------------------------------------------------------------------
# Corruption handling
# ---------------------------------------------------------------------------


def test_load_removes_corrupt_entry(cache, cache_dir):
    key = "corrupt"
    entry = cache_dir / key
    entry.mkdir()
    (entry / _GRAPH_FILENAME).write_text("this is not a valid torch file")
    (entry / _METADATA_FILENAME).write_text('{"last_accessed": 0}')

    result = cache.load(key)
    assert result is None
    assert not entry.exists()


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------


def test_eviction_by_max_entries(cache_dir):
    cache = GraphCache(cache_dir=str(cache_dir), max_entries=3, max_size_gb=100)
    mod = _make_simple_module()

    for i in range(5):
        cache.save(f"entry_{i}", mod)

    entries = list(cache_dir.iterdir())
    assert len(entries) <= 3


def test_eviction_removes_oldest(cache_dir):
    cache = GraphCache(cache_dir=str(cache_dir), max_entries=2, max_size_gb=100)
    mod = _make_simple_module()

    cache.save("old", mod)

    import time

    time.sleep(0.05)
    cache.save("mid", mod)
    time.sleep(0.05)

    # Touch "old" so it becomes recent
    cache.load("old")
    time.sleep(0.05)

    # This save should evict "mid" (least recently accessed), not "old"
    cache.save("new", mod)

    remaining = {p.name for p in cache_dir.iterdir() if p.is_dir()}
    assert "old" in remaining
    assert "new" in remaining


# ---------------------------------------------------------------------------
# Disk error handling
# ---------------------------------------------------------------------------


def test_save_handles_permission_error(cache):
    mod = _make_simple_module()
    bad_cache = GraphCache(cache_dir="/proc/nonexistent_path", max_entries=5, max_size_gb=1)
    # Should not raise -- logs a warning and continues
    bad_cache.save("key", mod)


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------


def test_env_overrides_cache_dir():
    with patch.dict(os.environ, {"AD_CACHE_DIR": "/tmp/my_custom_cache"}):
        cache = GraphCache()
        assert cache._cache_dir == Path("/tmp/my_custom_cache")


def test_env_overrides_max_entries():
    with patch.dict(os.environ, {"AD_CACHE_MAX_ENTRIES": "42"}):
        cache = GraphCache()
        assert cache._max_entries == 42


def test_env_overrides_max_size():
    with patch.dict(os.environ, {"AD_CACHE_MAX_SIZE_GB": "25"}):
        cache = GraphCache()
        assert cache._max_size_bytes == 25 * (1024**3)
