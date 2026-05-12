# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from tensorrt_llm._torch.pyexecutor import resource_manager


def _base_kwargs() -> dict:
    return {
        "num_kv_heads_per_layer": [2, 2],
        "size_per_head": 128,
        "tokens_per_block": 64,
        "blocks_per_window": {1024: (8, 0)},
        "max_num_sequences": 1,
        "max_beam_width": 1,
        "max_attention_window_vec": [1024],
        "temp_attention_window_inputs": None,
        "dtype": object(),
        "sink_token_length": 0,
        "stream": 123,
        "max_sequence_length": 1024,
        "chunk_size": 1024,
        "enable_block_reuse": False,
    }


def test_kv_cache_manager_cpp_drops_removed_temp_window_kwarg(monkeypatch):
    class CurrentBinding:
        def __init__(self, **kwargs):
            if "temp_attention_window_inputs" in kwargs:
                raise TypeError("unexpected keyword argument: temp_attention_window_inputs")
            if "chunk_size" not in kwargs:
                raise TypeError("missing required keyword argument: chunk_size")
            self.kwargs = kwargs

    monkeypatch.setattr(resource_manager, "KVCacheManagerCpp", CurrentBinding)

    manager = resource_manager._create_kv_cache_manager_cpp(_base_kwargs())

    assert "temp_attention_window_inputs" not in manager.kwargs
    assert manager.kwargs["chunk_size"] == 1024


def test_kv_cache_manager_cpp_can_still_fall_back_before_chunk_size(monkeypatch):
    class LegacyBinding:
        def __init__(self, **kwargs):
            if "chunk_size" in kwargs:
                raise TypeError("unexpected keyword argument: chunk_size")
            if "temp_attention_window_inputs" not in kwargs:
                raise TypeError("missing required keyword argument: temp_attention_window_inputs")
            self.kwargs = kwargs

    monkeypatch.setattr(resource_manager, "KVCacheManagerCpp", LegacyBinding)

    manager = resource_manager._create_kv_cache_manager_cpp(_base_kwargs())

    assert "chunk_size" not in manager.kwargs
    assert manager.kwargs["temp_attention_window_inputs"] is None


def test_kv_cache_manager_cpp_preserves_original_error(monkeypatch):
    class BrokenBinding:
        def __init__(self, **kwargs):
            raise TypeError("bad dtype")

    monkeypatch.setattr(resource_manager, "KVCacheManagerCpp", BrokenBinding)

    with pytest.raises(TypeError, match="bad dtype"):
        resource_manager._create_kv_cache_manager_cpp(_base_kwargs())
