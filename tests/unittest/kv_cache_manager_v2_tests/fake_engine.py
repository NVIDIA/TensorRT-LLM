# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
from collections.abc import Sequence
from functools import cached_property
from importlib.util import find_spec
from typing import TYPE_CHECKING, NamedTuple

if not TYPE_CHECKING and find_spec("kv_cache_manager_v2") is not None:
    from kv_cache_manager_v2 import (
        AttentionLayerConfig,
        BeamIndex,
        CudaStream,
        DataRole,
        KVCacheManagerConfig,
        LayerId,
        TokenIdExt,
        _KVCache,
    )
    from kv_cache_manager_v2._common import BAD_PAGE_INDEX, NDEBUG, MemAddress
    from kv_cache_manager_v2._utils import (
        div_up,
        exact_div,
        get_uniform_attribute,
        overlap,
        temporary_sys_path,
        typed_range,
        value_or,
    )
else:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        AttentionLayerConfig,
        BeamIndex,
        CudaStream,
        DataRole,
        KVCacheManagerConfig,
        LayerId,
        TokenIdExt,
        _KVCache,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._common import BAD_PAGE_INDEX, NDEBUG, MemAddress
    from tensorrt_llm.runtime.kv_cache_manager_v2._utils import (
        div_up,
        exact_div,
        get_uniform_attribute,
        overlap,
        temporary_sys_path,
        typed_range,
        value_or,
    )

import os

with temporary_sys_path(os.path.dirname(os.path.abspath(__file__))):
    from kernels import check_values, fill_values


class Step(NamedTuple):
    kv_cache: _KVCache
    input: list[TokenIdExt]  # when empty, just check history
    history: list[TokenIdExt]


class Role:
    """Constants for data roles in KV cache management."""

    KEY = DataRole("key")
    VALUE = DataRole("value")
    KEY_BLOCK_QUANT = DataRole("key_block_quant")
    VALUE_BLOCK_QUANT = DataRole("value_block_quant")


roles = (Role.KEY, Role.VALUE, Role.KEY_BLOCK_QUANT, Role.VALUE_BLOCK_QUANT)


class FakeEngine:
    cfg: KVCacheManagerConfig

    def __init__(self, config: KVCacheManagerConfig) -> None:
        super().__init__()
        self.cfg = config

    @property
    def tokens_per_block(self) -> int:
        return self.cfg.tokens_per_block

    @cached_property
    def layers(self) -> dict[LayerId, AttentionLayerConfig]:
        return {
            layer.layer_id: layer
            for layer in sorted(self.cfg.layers, key=lambda layer: layer.layer_id)
        }

    def execute(self, batch: Sequence[Step], stream: CudaStream) -> None:
        assert batch
        manager = get_uniform_attribute(batch, lambda step: step.kv_cache.manager)
        for kv_cache, input, history in batch:
            for layer_id, layer_cfg in self.layers.items():
                for buf_id, buf in enumerate(layer_cfg.buffers):
                    role = buf.role
                    assert NDEBUG or buf.size == manager.get_page_stride(layer_id, role)
                    for beam in typed_range(kv_cache.beam_width):
                        # check history
                        self._check_pages(kv_cache, layer_id, buf_id, beam, history, stream)
                        # write new token
                        if input:
                            self._write_new_tokens(
                                kv_cache, len(history), layer_id, buf_id, beam, input, stream
                            )

    def _check_pages(
        self,
        kv_cache: _KVCache,
        layer_id: LayerId,
        buf_id: int,
        beam: BeamIndex,
        history: Sequence[TokenIdExt],
        stream: CudaStream,
    ):
        manager = kv_cache.manager
        tokens_per_block = self.tokens_per_block
        layer_cfg = self.layers[layer_id]
        buf = layer_cfg.buffers[buf_id]
        role = buf.role
        token_bytes = exact_div(buf.size, tokens_per_block)
        pool = manager.get_mem_pool_base_address(layer_id, role)
        stride = manager.get_page_stride(layer_id, role)
        lc_id = manager._storage._layer_to_life_cycle_ids[layer_id]
        pages = kv_cache.get_page_indices(lc_id, beam)
        capacity = kv_cache.capacity
        history_len = len(history)
        assert len(history) == history_len
        window = (
            (0, capacity)
            if layer_cfg.window_size is None
            else (max(0, history_len + 1 - layer_cfg.window_size), capacity)
        )
        sink = value_or(layer_cfg.num_sink_tokens, 0)
        # check history
        for ordinal, page in enumerate(pages):
            if page == BAD_PAGE_INDEX:
                continue
            page_range = (tokens_per_block * ordinal, tokens_per_block * (ordinal + 1))
            need_page = overlap(page_range, (0, sink)) or overlap(page_range, window)
            if need_page:
                assert page != BAD_PAGE_INDEX
            else:
                assert kv_cache.history_length != history_len or page == BAD_PAGE_INDEX
            addr = MemAddress(pool + stride * page)
            tokens = history[tokens_per_block * ordinal : tokens_per_block * (ordinal + 1)]
            check_values(addr, token_bytes, layer_id, buf_id, beam, tokens, stream)

    def _write_new_tokens(
        self,
        kv_cache: _KVCache,
        history_len: int,
        layer_id: LayerId,
        buf_id: int,
        beam: BeamIndex,
        input: Sequence[TokenIdExt],
        stream: CudaStream,
    ):
        manager = kv_cache.manager
        tokens_per_block = self.tokens_per_block
        layer_cfg = self.layers[layer_id]
        buf = layer_cfg.buffers[buf_id]
        role = buf.role
        token_bytes = exact_div(buf.size, self.tokens_per_block)
        pool = manager.get_mem_pool_base_address(layer_id, role)
        stride = manager.get_page_stride(layer_id, role)
        lc_id = manager._storage._layer_to_life_cycle_ids[layer_id]
        pages = kv_cache.get_page_indices(lc_id, beam)[
            : div_up(history_len + len(input), tokens_per_block)
        ]
        capacity = kv_cache.capacity
        input_range = (history_len, history_len + len(input))
        assert input_range[1] <= capacity
        ordinal_beg = input_range[0] // tokens_per_block
        pages = itertools.islice(pages, ordinal_beg, None)
        ordinal = None
        for i, page in enumerate(pages):
            ordinal = ordinal_beg + i
            assert page != BAD_PAGE_INDEX
            page_range = (tokens_per_block * ordinal, tokens_per_block * (ordinal + 1))
            batch_range = tuple(i for i in overlap(input_range, page_range))
            assert batch_range
            tokens = input[(batch_range[0] - history_len) : (batch_range[1] - history_len)]
            addr = MemAddress(
                pool + stride * page + token_bytes * (batch_range[0] % tokens_per_block)
            )
            # print('layer_id={}, buf_id={}, beam={}, i={}, addr={}, tokens={}'.format(
            #     layer_id, buf_id, beam, i, addr, tokens))
            fill_values(addr, token_bytes, layer_id, buf_id, beam, tokens, stream)
        assert ordinal is None or ordinal + 1 == div_up(input_range[1], tokens_per_block)
