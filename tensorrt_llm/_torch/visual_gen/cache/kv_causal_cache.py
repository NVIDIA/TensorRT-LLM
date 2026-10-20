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
"""Paged K/V cache for a causal video rollout.

One rollout is one logical sequence of tokens

    [ fixed region | committed history | staged tokens ]

held in the pool of ``KVCacheManagerV2``. A token's logical position is its absolute position in the
rollout, and logical page ``p`` holds positions ``[p * tokens_per_page, (p + 1) * tokens_per_page)``
for the whole rollout, so the page table is indexed by absolute page and never rotates or moves a
token.

The manager's own sliding-window life cycle, with the fixed region as its sink tokens, decides what
stays resident: the fixed region's pages and the pages that still hold any token of the window. It
frees only pages that are entirely unused, so up to two pages keep some unused tokens: the one
where the fixed region ends and the one where the window begins. Their entries in the page table
are valid; which of their tokens a block may read is the attention kernel's responsibility, not the
cache's. A freed page's entry is 0 and is never read.

Each forward stages its new tokens' K/V at ``staging_offset`` and attends, per causal block of
``causal_block_size`` new tokens, to the fixed region, the ``window_tokens`` before the block's
first token, and everything from there to the end of the block. The kernel reads the cache through
``page_table`` and ``kv_ranges``.

Rotary positions are the caller's responsibility and are absolute over the rollout.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheManagerV2,
    _gpu_alloc_granularity,
)
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.math_utils import ceil_div

_DTYPES = {
    torch.bfloat16: tensorrt_llm.bindings.DataType.BF16,
    torch.float16: tensorrt_llm.bindings.DataType.HALF,
}

# The V2 sequence id of the one rollout. V2 registers every sequence under an
# integer id that normally comes from an LlmRequest; a rollout has none, so it
# gets a fixed one, as V2's own guard-page and capture-dummy sequences do.
_ROLLOUT_REQUEST_ID = 0


class CausalKVCacheManager(KVCacheManagerV2):
    """``KVCacheManagerV2`` driven as one long-lived sequence, with a windowed life cycle.

    The constructor allocates the page pool; ``open`` starts a rollout in it and ``close`` ends it,
    after which the cache may be opened again. No ``LlmRequest`` is ever created.

    Args:
        num_layers: attention layers that persist K/V (the generator tower).
        num_kv_heads: K/V heads this rank holds, i.e. the head count the attention layer it serves
            was built with. Head sharding (tensor parallel, Ulysses) is settled by the model before
            K/V reach the cache; the cache is a per-rank pool and knows nothing about the topology.
        head_dim, dtype: K/V geometry.
        tokens_per_page: page size, a power of two.
        pin_tokens: the first tokens committed are the fixed region: always visible, never evicted.
        window_tokens: history each causal block sees before its first token, in tokens
            (``(window_frames - 1) * tokens_per_frame``).
        max_staged_tokens: most tokens a forward may write before a commit
            (``chunk_frames * tokens_per_frame``).
        max_seq_len: most tokens one rollout may hold: the committed tokens and the room to stage
            after them. It sizes the page table, not the pool.

    Device tensors, allocated once and rewritten in place by ``open`` and ``commit``
    (so a CUDA graph captured around a forward keeps reading current values):

    * ``page_table``: ``[max_pages]`` int32, logical page to pool page, in the units ``kv_buffer``
        is indexed by. Entries of freed pages and of pages past the staged tokens are 0.
    * ``kv_ranges(num_blocks, block_size)``: the key range of each causal block.
    """

    @staticmethod
    def pool_pages_for(
        *, tokens_per_page: int, pin_tokens: int, window_tokens: int, max_staged_tokens: int
    ) -> int:
        """Pages one rollout of this geometry holds at its peak: the fixed region's pages,
        and the pages the window and the staged tokens touch, one more than their length
        when it is not page aligned."""
        _check_geometry(tokens_per_page, pin_tokens, window_tokens, max_staged_tokens)
        return ceil_div(pin_tokens, tokens_per_page) + (
            ceil_div(window_tokens + max_staged_tokens, tokens_per_page) + 1
        )

    @staticmethod
    def pool_bytes_for(
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        tokens_per_page: int,
        pin_tokens: int,
        window_tokens: int,
        max_staged_tokens: int,
    ) -> int:
        """Device bytes the pool of this geometry takes from the GPU: K and V of every
        token of every page in every layer, rounded up to the pool allocator's physical
        chunk size, since the pool is mapped in whole chunks."""
        pages = CausalKVCacheManager.pool_pages_for(
            tokens_per_page=tokens_per_page,
            pin_tokens=pin_tokens,
            window_tokens=window_tokens,
            max_staged_tokens=max_staged_tokens,
        )
        raw = (
            pages
            * tokens_per_page
            * num_layers
            * 2
            * num_kv_heads
            * head_dim
            * torch.tensor([], dtype=dtype).element_size()
        )
        granularity = _gpu_alloc_granularity(raw)
        return ceil_div(raw, granularity) * granularity

    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        tokens_per_page: int,
        pin_tokens: int,
        window_tokens: int,
        max_staged_tokens: int,
        max_seq_len: int,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        if dtype not in _DTYPES:
            raise ValueError(f"CausalKVCacheManager supports bf16/fp16 K/V, got {dtype}")
        _check_geometry(tokens_per_page, pin_tokens, window_tokens, max_staged_tokens)
        if max_seq_len < pin_tokens + max_staged_tokens:
            raise ValueError(
                f"max_seq_len {max_seq_len} is below pin_tokens + max_staged_tokens "
                f"({pin_tokens} + {max_staged_tokens})"
            )
        self._num_layers = num_layers
        self._kv_dtype = dtype
        self._window_tokens = window_tokens
        self._pin_tokens = pin_tokens
        self._max_staged_tokens = max_staged_tokens

        kv_cache_config = KvCacheConfig(
            # By bytes: ``max_tokens`` would also cap ``max_seq_len`` at the pool, which
            # a window exists to outgrow.
            max_gpu_total_bytes=self.pool_bytes_for(
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=dtype,
                tokens_per_page=tokens_per_page,
                pin_tokens=pin_tokens,
                window_tokens=window_tokens,
                max_staged_tokens=max_staged_tokens,
            ),
            # The manager keeps the window's first token live, so a window of
            # ``window_tokens`` before a block's first token is ``window_tokens + 1`` here.
            max_attention_window=[window_tokens + 1],
            enable_block_reuse=False,
            host_cache_size=0,  # nothing is ever suspended, so no host tier
            max_util_for_resume=1.0,  # and no resume headroom
        )
        super().__init__(
            kv_cache_config,
            CacheType.SELF,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_page,
            max_seq_len=max_seq_len,
            max_batch_size=1,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            dtype=_DTYPES[dtype],
            max_num_tokens=max_staged_tokens,
            execution_stream=stream,
            num_sink_tokens=pin_tokens,
        )
        if self.max_seq_len < max_seq_len:
            raise RuntimeError(
                f"KVCacheManagerV2 clamped max_seq_len to {self.max_seq_len} < {max_seq_len}"
            )
        if self.num_pools != 1:
            raise RuntimeError(f"expected one K/V pool, got {self.num_pools}")
        scales = {self.get_layer_page_index_scale(i) for i in range(num_layers)}
        if len(scales) != 1:
            raise RuntimeError(f"layers disagree on page index scale: {sorted(scales)}")
        (scale,) = scales
        if scale % self.kv_factor:
            raise RuntimeError(
                f"page index scale {scale} not a multiple of kv_factor {self.kv_factor}"
            )
        # ``get_buffers(layer)`` is a view whose index unit is one K/V page pair, so base
        # page ``p`` sits at view index ``p * page_view_scale``.
        self.page_view_scale = scale // self.kv_factor

        self._kv_cache = None
        self._buffers: List[torch.Tensor] = []
        self._shut_down = False
        self._committed_tokens = 0
        self._fixed_tokens = 0
        self._max_pages = self.max_blocks_per_seq
        device = torch.device("cuda", torch.cuda.current_device())
        self._table = torch.zeros(self._max_pages, dtype=torch.int32, device=device)
        # Key ranges per (num_blocks, block_size), made on first use.
        self._ranges: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        # Pool page and slot of each token the next forward may stage, in staging order.
        staged_capacity = pin_tokens + max_staged_tokens
        self._staged_pages = torch.zeros(staged_capacity, dtype=torch.int64, device=device)
        self._staged_slots = torch.zeros(staged_capacity, dtype=torch.int64, device=device)
        self._host_pages = np.full(self._max_pages, -1, dtype=np.int64)

    # ------------------------------------------------------------------ lifecycle

    def open(self) -> None:
        """Start a rollout: the sequence is empty and ``pin_tokens + max_staged_tokens``
        tokens may be staged."""
        if self._kv_cache is not None:
            raise RuntimeError("cache already open; call close() first")
        kv_cache = self._create_kv_cache(_ROLLOUT_REQUEST_ID, None, None, is_dummy=True)
        if kv_cache is None:
            raise RuntimeError("KVCacheManagerV2 has no free sequence slot")
        try:
            if not kv_cache.resume(self._stream.cuda_stream):
                raise RuntimeError("KVCacheManagerV2 could not resume the rollout sequence")
            # We never commit tokens to the manager's radix tree; the pages are ours.
            kv_cache.stop_committing()
            self._committed_tokens = 0
            self._fixed_tokens = 0
            self._kv_cache = kv_cache
            self._resize()
        except Exception:
            self._kv_cache = None
            self._remove_kv_cache(_ROLLOUT_REQUEST_ID)
            raise
        # The views alias the pool, which V2 may map lazily: take them once pages exist.
        self._buffers = [
            self.get_buffers(layer, kv_layout="HND") for layer in range(self._num_layers)
        ]

    def close(self) -> None:
        if self._kv_cache is None:
            return
        self._kv_cache = None
        self._buffers = []
        self._remove_kv_cache(_ROLLOUT_REQUEST_ID)

    def shutdown(self) -> None:
        self.close()
        super().shutdown()
        self._shut_down = True

    @property
    def is_shut_down(self) -> bool:
        """Whether the pool is gone for good; a closed cache may still be reopened."""
        return self._shut_down

    @property
    def is_open(self) -> bool:
        return self._kv_cache is not None

    def _require_open(self) -> None:
        if self._kv_cache is None:
            raise RuntimeError("cache is not open; call open() first")

    # ------------------------------------------------------------------ geometry

    @property
    def tokens_per_page(self) -> int:
        """Page size of the K/V pool. The parent class calls it ``tokens_per_block``;
        in this cache and its backends "block" is only ever a causal block."""
        return self.tokens_per_block

    @property
    def window_tokens(self) -> int:
        return self._window_tokens

    @property
    def pin_tokens(self) -> int:
        return self._pin_tokens

    @property
    def max_staged_tokens(self) -> int:
        return self._max_staged_tokens

    @property
    def fixed_tokens(self) -> int:
        """Tokens every block sees for the whole rollout; logical ``[0, fixed_tokens)``.
        The first ``pin_tokens`` committed tokens."""
        return self._fixed_tokens

    @property
    def staging_offset(self) -> int:
        """Logical position of the first staged token: the tokens committed so far."""
        return self._committed_tokens

    @property
    def staging_capacity(self) -> int:
        """Most tokens the next forward may stage. The fixed region may be staged in one
        go on top of ``max_staged_tokens``."""
        return self._max_staged_tokens + self._pin_tokens - self._fixed_tokens

    @property
    def page_table(self) -> torch.Tensor:
        """``[max_pages]`` int32 device table: logical page to pool page, in the units
        ``kv_buffer`` is indexed by."""
        return self._table

    def kv_ranges(self, num_blocks: int, block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(start, length)``, int32 device tensors of ``num_blocks`` entries: causal block
        ``i`` of ``block_size`` new tokens attends to the logical key positions
        ``[start[i], start[i] + length[i])`` -- the ``window_tokens`` before the block through
        its end, and to the first ``fixed_tokens`` positions.

        Persistent per ``(num_blocks, block_size)`` and refreshed in place by ``commit``;
        the first call for a pair must not be inside a CUDA graph capture.
        """
        self._require_open()
        key = (num_blocks, block_size)
        ranges = self._ranges.get(key)
        if ranges is None:
            device = self._table.device
            ranges = (
                torch.zeros(num_blocks, dtype=torch.int32, device=device),
                torch.zeros(num_blocks, dtype=torch.int32, device=device),
            )
            self._ranges[key] = ranges
            self._fill_ranges(key, ranges)
        return ranges

    def _fill_ranges(self, key: Tuple[int, int], ranges: Tuple[torch.Tensor, torch.Tensor]) -> None:
        num_blocks, block_size = key
        first = self._committed_tokens + block_size * np.arange(num_blocks, dtype=np.int64)
        start = np.maximum(first - self._window_tokens, 0)
        ranges[0].copy_(torch.from_numpy(start.astype(np.int32)))
        ranges[1].copy_(torch.from_numpy((first + block_size - start).astype(np.int32)))

    def host_page_table(self) -> List[int]:
        """Logical page to base page, from the host copy, -1 for a page that is freed or
        not yet allocated. For tests and diagnostics; kernels read ``page_table``."""
        self._require_open()
        return [int(p) for p in self._host_pages]

    @staticmethod
    def causal_blocks(num_tokens: int, causal_block_size: Optional[int]) -> Tuple[int, int]:
        """The block size and block count a forward of ``num_tokens`` real tokens uses:
        one block of all of them when ``causal_block_size`` is None; otherwise the size
        must be positive and divide ``num_tokens``. Shared by the attention backends so
        they agree on the contract."""
        size = num_tokens if causal_block_size is None else int(causal_block_size)
        if size <= 0 or num_tokens % size:
            raise ValueError(
                f"{num_tokens} tokens do not split into causal blocks of {causal_block_size}"
            )
        return size, num_tokens // size

    # ------------------------------------------------------------------ the window

    def commit(self, num_tokens: int) -> None:
        """The ``num_tokens`` staged tokens, written at ``staging_offset``, are final. Those
        that fall within the first ``pin_tokens`` become fixed; the window moves past the
        rest and the pages it has left entirely are freed. The count is the caller's: a
        forward may stage fewer tokens than ``max_staged_tokens``."""
        self._require_open()
        if not 0 < num_tokens <= self.staging_capacity:
            raise ValueError(
                f"commit of {num_tokens} tokens; the next forward stages 1..{self.staging_capacity}"
            )
        committed = self._committed_tokens + num_tokens
        fixed = min(self._pin_tokens, committed)
        capacity = committed + self._max_staged_tokens + self._pin_tokens - fixed
        if capacity > self.max_seq_len:
            raise ValueError(
                f"committing {num_tokens} tokens leaves {committed} committed and room to stage "
                f"up to {capacity}, past max_seq_len {self.max_seq_len}"
            )
        self._committed_tokens, self._fixed_tokens = committed, fixed
        self._resize()

    def _resize(self) -> None:
        """Give the sequence the committed tokens as history and the staging room after
        them, then re-read the table."""
        history = self._committed_tokens
        capacity = history + self.staging_capacity
        # Release the pages the window has left before backing the new staging room: one
        # call would allocate first, and the pool holds only the steady state.
        if not (self._kv_cache.resize(None, history) and self._kv_cache.resize(capacity)):
            raise RuntimeError(
                f"KVCacheManagerV2 could not back {capacity} tokens with history {history}; "
                "the pool is smaller than the geometry needs"
            )
        pages = np.asarray(self._kv_cache.get_base_page_indices(0), dtype=np.int64)
        self._host_pages = pages
        table = np.where(pages >= 0, pages * self.page_view_scale, 0).astype(np.int32)
        self._table.copy_(torch.from_numpy(table))
        self._refresh_staged(table)
        for key, ranges in self._ranges.items():
            self._fill_ranges(key, ranges)

    def _refresh_staged(self, table: np.ndarray) -> None:
        tpb = self.tokens_per_page
        count = self.staging_capacity
        pos = self._committed_tokens + np.arange(count, dtype=np.int64)
        self._staged_pages[:count].copy_(torch.from_numpy(table[pos // tpb].astype(np.int64)))
        self._staged_slots[:count].copy_(torch.from_numpy(pos % tpb))

    # ------------------------------------------------------------------ pool access

    def kv_buffer(self, layer_idx: int) -> torch.Tensor:
        """The layer's pool view ``[view_pages, 2, num_kv_heads, tokens_per_page, head_dim]``
        (plane 0 is K, plane 1 is V). Pool page ``p`` of ``page_table`` is ``buf[p]``."""
        self._require_open()
        return self._buffers[layer_idx]

    def write_staged(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Stage ``T`` tokens of K/V, the forward's new tokens, at logical positions
        ``[staging_offset, staging_offset + T)``. ``k``/``v`` are ``[T, num_kv_heads, head_dim]``;
        strides are free, so slices of a fused QKV projection go in without a copy. Rewrites the
        same positions on every call until ``commit``."""
        buf = self.kv_buffer(layer_idx)
        num_tokens, heads, head_dim = k.shape
        if v.shape != k.shape:
            raise ValueError(f"k/v shape mismatch: {tuple(k.shape)} vs {tuple(v.shape)}")
        if num_tokens > self.staging_capacity or (heads, head_dim) != (buf.shape[2], buf.shape[4]):
            raise ValueError(
                f"staged write of [{num_tokens}, {heads}, {head_dim}] does not fit "
                f"[{self.staging_capacity}, {buf.shape[2]}, {buf.shape[4]}]"
            )
        if k.dtype != buf.dtype or v.dtype != buf.dtype:
            raise TypeError(f"K/V dtype {k.dtype} does not match the cache's {buf.dtype}")
        pages = self._staged_pages[:num_tokens]
        slots = self._staged_slots[:num_tokens]
        # The advanced indices on dims 0, 1 and 3 broadcast to [T] and come first, so the
        # indexed view is [T, num_kv_heads, head_dim].
        buf[pages, 0, :, slots] = k
        buf[pages, 1, :, slots] = v


def _check_geometry(
    tokens_per_page: int, pin_tokens: int, window_tokens: int, max_staged_tokens: int
) -> None:
    if tokens_per_page <= 0 or tokens_per_page & (tokens_per_page - 1):
        raise ValueError(f"tokens_per_page must be a power of two, got {tokens_per_page}")
    if window_tokens < 0 or pin_tokens < 0 or max_staged_tokens <= 0:
        raise ValueError(
            "window_tokens and pin_tokens must be non-negative and max_staged_tokens positive"
        )
