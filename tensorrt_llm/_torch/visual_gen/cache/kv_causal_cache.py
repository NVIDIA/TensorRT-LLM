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

One rollout is one logical sequence laid out as

    [ fixed region | rolling history | staged tokens ]

held in a single pool of ``KVCacheManagerV2``. The manager provides the pool
and the pages; this class owns the page table and the sliding window. Nothing
here uses the manager's own sliding-window eviction, block reuse, or request
scheduling, and no ``LlmRequest`` is ever created.

The fixed region is the first ``pin_tokens`` tokens of the sequence, declared
by ``open``: every frame sees them for the whole rollout and they are never
evicted. Tokens become fixed when they are committed, so nothing can be pinned
after it was evicted. What those tokens are is the model's business. The
rolling history slides: ``commit`` turns the staged tokens (written by forwards, not
yet committed) into history and drops
whole pages from the front once the history exceeds the window, by rotating
the table; no history moves. When the fixed region ends mid-page the first
history tokens share its last page, and after a rotation the fixed region's
tail is copied into the page that now starts the history.

Attention reads exactly what the model's window says. A paged kernel reads
whole pages in table order and cuts only at the end, so every causal block of a
forward gets its own table row: all pages that are legitimate in full, then a
private region holding the few legitimate slots of partially-legitimate pages
followed by the block's own tokens. The kernel is told the cached length up to
the block's own tokens and reads nothing older than the window and nothing
of later blocks. See ``_CausalBlockLayout``.

Rotary positions are the caller's business and are absolute over the rollout;
storage positions here never grow past the resident capacity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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


@dataclass
class _CausalBlockLayout:
    """Everything a forward that cuts the staged tokens into causal blocks of one size
    reads.

    Block ``i`` holds staged tokens ``[i*block_size, (i+1)*block_size)`` and may see the
    fixed region, the ``window_tokens`` before it, the earlier blocks and itself. Its
    table row lists every page that is legitimate in full, then ``region_pages``
    private pages holding, in order, the legitimate slots of the partially
    legitimate pages (at most three: where the fixed region ends, where the window
    starts, where the block starts) and the block's own tokens. ``cached[i]`` counts
    the keys before the block's own tokens. Slots holding fixed or history tokens
    are copied at ``commit``; slots holding this forward's staged tokens (earlier
    blocks' tokens on a partial page, and the block's own) are written by every
    ``write_staged``. All device tensors are persistent views into
    the cache's packed buffers, rewritten in place by ``commit``, never
    on the forward path; ``host`` holds numpy views of the same layout on the host
    side of that upload.
    """

    num_blocks: int
    block_size: int
    region_pages: int
    regions: np.ndarray  # [num_blocks, region_pages] int64 layer-0 view indices, host
    host: Dict[str, np.ndarray]  # host twins of rows, seq_len_*, own_slots, extra_*
    rows: torch.Tensor  # [num_blocks, row_len] int32 layer-0 view indices, 0-padded
    # The rows in the TRTLLM attention op's block-offset encoding, K plane then V
    # plane: ``row * kv_factor`` and ``+ kv_offset``. The op reads this tensor directly.
    block_offsets: torch.Tensor  # [1, num_blocks, 2, row_len] int32
    seq_len_q: torch.Tensor  # [num_blocks] int32, all block_size
    seq_len_kv: torch.Tensor  # [num_blocks] int32, cached + block_size
    cached: List[int]  # host copy
    # Slot ids (``view_page * tokens_per_page + slot``) of the private slots holding
    # each staged token as one of its block's own tokens.
    own_slots: torch.Tensor  # [num_blocks*block_size] int64
    # Earlier blocks' tokens on a block's partial pages (at most two: where its window
    # starts and where it starts): the staged token and the private slot it goes to.
    # A fixed 2*(tpb-1) entries per block, padded with a
    # harmless repeat of the block's own first token, so the first ``n`` blocks'
    # entries are a prefix and a captured forward replays them.
    extra_src: torch.Tensor  # [num_blocks*2*(tpb-1)] int64 staged token index
    extra_dst: torch.Tensor  # [num_blocks*2*(tpb-1)] int64 slot id


@dataclass
class _GeometryState:
    """Device state a geometry's forwards read, allocated the first time that
    geometry is opened and kept at the same addresses for the life of the cache, so
    a CUDA graph captured over it stays valid when the geometry is opened again.
    Every per-layout tensor is a view into one of two packed buffers with a host
    twin, rewritten in place by ``commit``."""

    layouts: Dict[int, _CausalBlockLayout]  # by causal block size
    packed_i32: torch.Tensor
    packed_i64: torch.Tensor
    host_i32: np.ndarray
    host_i64: np.ndarray
    # Tail of the int64 buffer: the pieces copied at commit, as layer-0 pool rows.
    piece_src: torch.Tensor
    piece_dst: torch.Tensor
    host_piece_src: np.ndarray
    host_piece_dst: np.ndarray
    # [max_staged_tokens] int64 slot ids (view_page * tokens_per_page + slot) of the
    # staged tokens' logical positions, the shared pages later blocks and later
    # commits read; and scratch to recompute them without allocating.
    staged_slots: torch.Tensor
    logical: torch.Tensor
    logical_page: torch.Tensor
    view_page: torch.Tensor  # int32: gathers from the int32 table


class CausalKVCacheManager(KVCacheManagerV2):
    """``KVCacheManagerV2`` driven as one long-lived sequence with a caller-owned table.

    The constructor allocates the page pool; ``open`` lays a rollout's geometry out
    inside it: the pinned prefix, the sliding window, the staged tokens (written by
    forwards, not yet committed) and the causal blocks' private pages.

    Args:
        num_layers: attention layers that persist K/V (the generator tower).
        num_kv_heads: K/V heads this rank holds, i.e. the head count the attention
            backend it serves was built with. Head sharding (tensor parallel,
            Ulysses) is settled by the model before K/V reach the cache; the
            cache is a per-rank pool and knows nothing about the topology.
        head_dim, dtype: K/V geometry.
        tokens_per_page: page size, a power of two (both paged kernels require
            it). Each backend may add its own constraint and raises if unmet.
        pool_tokens: size of the page pool, a multiple of ``tokens_per_page``;
            ``pool_tokens_for`` gives what a geometry needs.
    """

    @staticmethod
    def _region_pages(tokens_per_page: int, block_size: int) -> int:
        """Private pages one causal block needs: up to three partial pages' worth of
        visible slots plus its own tokens."""
        return ceil_div(3 * (tokens_per_page - 1) + block_size, tokens_per_page)

    @staticmethod
    def _geometry_pages(
        tokens_per_page: int,
        pin_tokens: int,
        window_tokens: int,
        max_staged_tokens: int,
        causal_block_sizes: Sequence[int],
    ) -> Tuple[int, int, Tuple[int, ...]]:
        """Validate a geometry and return its page counts: ``(resident, private,
        sizes)``. Resident tokens peak at pin + window + (tpb - 1) stale + staged; the
        extra page covers the stale tokens and an unaligned fixed-region end."""
        tpb = tokens_per_page
        if tpb <= 0 or tpb & (tpb - 1):
            raise ValueError(f"tokens_per_page must be a power of two, got {tpb}")
        if min(window_tokens, max_staged_tokens) <= 0 or pin_tokens < 0:
            raise ValueError(
                "window_tokens and max_staged_tokens must be positive and pin_tokens non-negative"
            )
        sizes = tuple(dict.fromkeys(int(b) for b in causal_block_sizes))
        if not sizes or any(not 0 < b <= max_staged_tokens for b in sizes):
            raise ValueError(
                f"causal_block_sizes {tuple(causal_block_sizes)} must be positive and at most "
                f"max_staged_tokens {max_staged_tokens}"
            )
        resident = ceil_div(pin_tokens + window_tokens + max_staged_tokens, tpb) + 1
        private = sum(
            (max_staged_tokens // b) * CausalKVCacheManager._region_pages(tpb, b) for b in sizes
        )
        return resident, private, sizes

    @staticmethod
    def pool_tokens_for(
        *,
        tokens_per_page: int,
        pin_tokens: int,
        window_tokens: int,
        max_staged_tokens: int,
        causal_block_sizes: Sequence[int],
    ) -> int:
        """Pool tokens a geometry needs: resident pages (pinned prefix, window, staged
        tokens, one spare) plus the private pages of every block size. A pool of at
        least this many tokens can ``open`` the geometry."""
        resident, private, _ = CausalKVCacheManager._geometry_pages(
            tokens_per_page, pin_tokens, window_tokens, max_staged_tokens, causal_block_sizes
        )
        return (resident + private) * tokens_per_page

    @staticmethod
    def bytes_for_pool_tokens(
        pool_tokens: int, *, num_layers: int, num_kv_heads: int, head_dim: int, dtype: torch.dtype
    ) -> int:
        """Device bytes a pool of ``pool_tokens`` takes from the GPU: K and V of every
        token in every layer, rounded up to the pool allocator's physical chunk size,
        since the pool is mapped in whole chunks."""
        raw = (
            pool_tokens
            * num_layers
            * 2
            * num_kv_heads
            * head_dim
            * torch.tensor([], dtype=dtype).element_size()
        )
        return ceil_div(raw, _gpu_alloc_granularity(raw)) * _gpu_alloc_granularity(raw)

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
        causal_block_sizes: Sequence[int],
    ) -> int:
        """Device bytes a pool sized for this geometry takes from the GPU."""
        return CausalKVCacheManager.bytes_for_pool_tokens(
            CausalKVCacheManager.pool_tokens_for(
                tokens_per_page=tokens_per_page,
                pin_tokens=pin_tokens,
                window_tokens=window_tokens,
                max_staged_tokens=max_staged_tokens,
                causal_block_sizes=causal_block_sizes,
            ),
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )

    @property
    def pool_bytes(self) -> int:
        """Device bytes of this cache's page pool."""
        return self.bytes_for_pool_tokens(
            self._pool_tokens,
            num_layers=self._num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=self._kv_dtype,
        )

    @property
    def pool_tokens(self) -> int:
        """Tokens the page pool holds, every geometry included."""
        return self._pool_tokens

    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        tokens_per_page: int,
        pool_tokens: int,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        if dtype not in _DTYPES:
            raise ValueError(f"CausalKVCacheManager supports bf16/fp16 K/V, got {dtype}")
        if tokens_per_page <= 0 or tokens_per_page & (tokens_per_page - 1):
            # cuDNN's paged SDPA and trtllm-gen's KV block array both require it.
            raise ValueError(f"tokens_per_page must be a power of two, got {tokens_per_page}")
        tpb = tokens_per_page
        if pool_tokens <= 0 or pool_tokens % tpb:
            raise ValueError(
                f"pool_tokens must be a positive multiple of tokens_per_page {tpb}, got {pool_tokens}"
            )
        self._num_layers = num_layers
        self._kv_dtype = dtype
        self._pool_tokens = pool_tokens
        self._pool_pages = pool_tokens // tpb
        # Staged tokens a block's private region holds besides its own: at most two
        # partial pages of earlier blocks, where its window starts and where it starts.
        self._start_entries = 2 * (tpb - 1)

        kv_cache_config = KvCacheConfig(
            max_tokens=pool_tokens,
            max_attention_window=[pool_tokens],
            enable_block_reuse=False,
            host_cache_size=0,  # nothing is ever suspended, so no host tier
            max_util_for_resume=1.0,  # and no resume headroom: the pool is exactly the tokens
        )
        super().__init__(
            kv_cache_config,
            CacheType.SELF,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tpb,
            max_seq_len=pool_tokens,
            max_batch_size=1,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            dtype=_DTYPES[dtype],
            max_num_tokens=pool_tokens,
            execution_stream=stream,
        )
        if self.max_seq_len < pool_tokens:
            raise RuntimeError(
                f"KVCacheManagerV2 clamped max_seq_len to {self.max_seq_len} < {pool_tokens}; "
                "the pool could not back every page"
            )
        if self.num_pools != 1:
            raise RuntimeError(f"expected one K/V pool, got {self.num_pools}")
        # A physical slot holds every layer's page; ``get_buffers(layer)`` is a view
        # whose index unit is one K/V page pair, so base page ``p`` of a layer sits
        # at view index ``p * page_view_scale`` (1 for a single layer).
        scales = {self.get_layer_page_index_scale(i) for i in range(num_layers)}
        if len(scales) != 1:
            raise RuntimeError(f"layers disagree on page index scale: {sorted(scales)}")
        (scale,) = scales
        if scale % self.kv_factor:
            raise RuntimeError(
                f"page index scale {scale} not a multiple of kv_factor {self.kv_factor}"
            )
        self.page_view_scale = scale // self.kv_factor

        self._kv_cache = None
        self._buffers: List[torch.Tensor] = []
        self._shut_down = False
        # The open geometry; set by open().
        self.window_tokens = 0
        self.max_staged_tokens = 0
        self.causal_block_sizes: Tuple[int, ...] = ()
        self.num_pages = 0  # resident pages: pinned prefix, window, staged tokens, spare
        self.capacity = 0  # resident tokens, num_pages * tokens_per_page
        self._pin_tokens = 0
        self._fixed_tokens = 0
        self._history_tokens = 0
        # Device-side state read by the kernels. Every tensor here lives for the
        # life of the cache and is rewritten in place by open() and commit(), never
        # on the forward path, so a CUDA graph captured around a forward keeps
        # reading correct values after the cache moves.
        #
        # The table: layer-0 view indices in logical order, the fixed region's full
        # pages first (never rotated), then the ring: the page the fixed region's
        # tail shares with the first history tokens (if any), history, free pages.
        # Sized for the whole pool; a geometry uses the first num_pages entries.
        self._table: Optional[torch.Tensor] = None  # [pool_pages] int32
        self._fixed_pages = 0
        # Host copy of the resident table, rotated in lockstep with the device one, so
        # that nothing is ever read back; and the private pages' view indices.
        self._host_table: Optional[np.ndarray] = None  # [num_pages] int64
        self._host_private: Optional[np.ndarray] = None  # [pool_pages] int64
        self._piece_offsets: Optional[torch.Tensor] = None  # [layers * 2H] pool rows
        # Per-geometry state, and the open geometry's fields bound as attributes.
        self._states: Dict[Tuple[int, int, Tuple[int, ...]], _GeometryState] = {}
        self._layouts: Dict[int, _CausalBlockLayout] = {}
        self._packed_i32: Optional[torch.Tensor] = None
        self._packed_i64: Optional[torch.Tensor] = None
        self._host_i32: Optional[np.ndarray] = None
        self._host_i64: Optional[np.ndarray] = None
        self._piece_src: Optional[torch.Tensor] = None
        self._piece_dst: Optional[torch.Tensor] = None
        self._host_piece_src: Optional[np.ndarray] = None
        self._host_piece_dst: Optional[np.ndarray] = None
        self._mirrors = (0, 0)  # live commit-time pieces whose sources are resident tokens
        self._staged_slots: Optional[torch.Tensor] = None
        self._logical: Optional[torch.Tensor] = None
        self._logical_page: Optional[torch.Tensor] = None
        self._view_page: Optional[torch.Tensor] = None
        self._block_offsets_size: Optional[int] = None
        self._kv_heads_local = 0
        self._rows_per_page = 0  # pool rows of head_dim per view page: 2 * H * tpb
        self._kv_offset = 0  # the V plane's offset in the op's block-offset encoding

    # ------------------------------------------------------------------ lifecycle

    def open(
        self,
        *,
        pin_tokens: int = 0,
        window_tokens: int,
        max_staged_tokens: int,
        causal_block_sizes: Sequence[int],
    ) -> None:
        """Start a rollout with this geometry; the sequence starts empty.

        Args:
            pin_tokens: the first tokens committed become the fixed region: always
                visible, never evicted. Commit them like any other tokens; a commit
                may exceed ``max_staged_tokens`` as long as the excess is pinned.
            window_tokens: history each block may see before itself, in tokens
                (``(window_frames - 1) * tokens_per_frame``).
            max_staged_tokens: most tokens a forward may write before a commit
                (``chunk_frames * tokens_per_frame``); a forward may write fewer.
            causal_block_sizes: every causal block size a forward may use, at most
                ``max_staged_tokens``. A forward of ``T`` tokens cut into blocks of
                size ``b`` needs ``b`` declared and ``T`` a multiple of ``b`` within
                ``max_staged_tokens``: all staged tokens as one block is a denoising
                step, one frame per block is the clean pass, a shorter final chunk as
                one block is its own size. Each size gets its own private pages,
                ``max_staged_tokens // b`` blocks' worth.

        The geometry must fit the pool: ``pool_tokens_for`` of it at most
        ``pool_tokens``. Device state is allocated per geometry the first time it is
        opened and kept, so graphs captured over a geometry replay after it is
        opened again.
        """
        if self._kv_cache is not None:
            raise RuntimeError("cache already open; call close() first")
        tpb = self.tokens_per_page
        num_pages, num_private, sizes = self._geometry_pages(
            tpb, pin_tokens, window_tokens, max_staged_tokens, causal_block_sizes
        )
        needed = (num_pages + num_private) * tpb
        if needed > self._pool_tokens:
            raise ValueError(
                f"geometry needs {needed} pool tokens (pin {pin_tokens}, window {window_tokens}, "
                f"staged {max_staged_tokens}, blocks {sizes}); the pool holds {self._pool_tokens}"
            )

        kv_cache = self._create_kv_cache(_ROLLOUT_REQUEST_ID, None, None, is_dummy=True)
        if kv_cache is None:
            raise RuntimeError("KVCacheManagerV2 has no free sequence slot")
        try:
            if not kv_cache.resume(self._stream.cuda_stream):
                raise RuntimeError("KVCacheManagerV2 could not resume the rollout sequence")
            # We never commit tokens; the pages are ours to address directly.
            kv_cache.stop_committing()
            if not kv_cache.resize(self._pool_tokens):
                raise RuntimeError(
                    f"KVCacheManagerV2 could not back {self._pool_pages} pages for the rollout"
                )
        except Exception:
            self._release(kv_cache)
            raise

        total_pages = num_pages + num_private
        # V2 owns and may rewrite that buffer: copy it.
        pages = np.array(kv_cache.get_base_page_indices(0), dtype=np.int64)[:total_pages]
        if pages.size != total_pages or pages.min() < 0:
            self._release(kv_cache)
            raise RuntimeError(f"expected {total_pages} backed pages, got {pages}")
        # Logical order is ours to choose. Ascending keeps consecutive page ids
        # consecutive after rotation whenever V2 handed out a contiguous range, which
        # lets write_range copy a prompt in few pieces; nothing depends on it.
        pages.sort()

        # Each layer's pool view, looked up once: the lookup goes through the
        # manager's C++ binding, which torch.compile cannot trace.
        buffers = [self.get_buffers(layer, kv_layout="HND") for layer in range(self._num_layers)]
        buf = buffers[0]
        # Layer ``l``'s page view is layer 0's shifted by ``l`` pages; commit-time
        # copies address every layer through layer 0's view with that offset.
        page_bytes = buf[0].numel() * buf.element_size()
        for layer, layer_buf in enumerate(buffers):
            if layer_buf is None or layer_buf.data_ptr() != buf.data_ptr() + layer * page_bytes:
                self._release(kv_cache)
                raise RuntimeError("K/V pool layout changed: layers are not page-interleaved")
        self._buffers = buffers
        self._kv_cache = kv_cache
        try:
            self._lay_out(
                kv_cache,
                pages,
                buf,
                pin_tokens,
                window_tokens,
                max_staged_tokens,
                sizes,
                num_pages,
                num_private,
            )
        except Exception:
            # Nothing may look open after a failed open: release the sequence and
            # forget the half-built state, so the caller can retry.
            self._kv_cache = None
            self._buffers = []
            self._release(kv_cache)
            raise

    def _lay_out(
        self,
        kv_cache,
        pages: np.ndarray,
        buf: torch.Tensor,
        pin_tokens: int,
        window_tokens: int,
        max_staged_tokens: int,
        sizes: Tuple[int, ...],
        num_pages: int,
        num_private: int,
    ) -> None:
        """The geometry-dependent half of ``open``: bind the pages, allocate or reuse
        the geometry's device state and write the kernel-facing tensors."""
        tpb = self.tokens_per_page
        self.window_tokens = window_tokens
        self.max_staged_tokens = max_staged_tokens
        self.causal_block_sizes = sizes
        self.num_pages = num_pages
        self.capacity = num_pages * tpb
        self._pin_tokens = pin_tokens
        self._fixed_tokens = 0
        self._history_tokens = 0
        self._fixed_pages = 0
        device = buf.device
        self._kv_heads_local = buf.shape[2]
        self._rows_per_page = 2 * self._kv_heads_local * tpb
        self._kv_offset = int(self.kv_offset[0])
        scaled = pages * self.page_view_scale
        if self._table is None:
            self._table = torch.zeros(self._pool_pages, dtype=torch.int32, device=device)
            self._host_private = np.zeros(self._pool_pages, dtype=np.int64)
            # A piece's layer-0 slot row plus these offsets gives that slot's row in
            # every layer, K and V, every head: + layer*rpp + (kv*H + head)*tpb.
            layer_rows = np.arange(self._num_layers, dtype=np.int64)[:, None] * self._rows_per_page
            kv_head = np.arange(2 * self._kv_heads_local, dtype=np.int64)[None, :] * tpb
            self._piece_offsets = torch.from_numpy((layer_rows + kv_head).reshape(-1)).to(device)
        # Reopened: the device tensors a captured forward points at stay where they
        # are and only their contents change (layout regions view _host_private).
        self._table[:num_pages].copy_(torch.from_numpy(scaled[:num_pages]))
        self._host_private[:num_private] = scaled[num_pages:]
        self._host_table = scaled[:num_pages].copy()
        key = (window_tokens, max_staged_tokens, sizes)
        state = self._states.get(key)
        if state is None:
            state = self._allocate_state(device, max_staged_tokens, sizes, num_private)
            self._states[key] = state
        self._bind_state(state)
        self._block_offsets_size = None
        self._refresh_device_state()

    def close(self) -> None:
        if self._kv_cache is None:
            return
        kv_cache, self._kv_cache = self._kv_cache, None
        # Device state outlives the sequence so that forwards captured in a CUDA
        # graph before close() stay valid after the next open().
        self._block_offsets_size = None
        self._release(kv_cache)

    def _release(self, kv_cache) -> None:
        self.kv_cache_map.pop(_ROLLOUT_REQUEST_ID, None)
        kv_cache.discard_pending_stats()
        kv_cache.close()
        self.impl.clear_stats_excluded(_ROLLOUT_REQUEST_ID)
        self.index_mapper.remove_sequence(_ROLLOUT_REQUEST_ID)

    def shutdown(self) -> None:
        self.close()
        super().shutdown()
        self._shut_down = True

    @property
    def is_shut_down(self) -> bool:
        """Whether the pool is gone for good; a closed cache may still be reopened."""
        return self._shut_down

    # ------------------------------------------------------------------ geometry

    @property
    def tokens_per_page(self) -> int:
        """Page size of the K/V pool. The parent class calls it ``tokens_per_block``;
        in this cache and its backends "block" is only ever a causal block."""
        return self.tokens_per_block

    @property
    def pin_tokens(self) -> int:
        """Tokens the open rollout pins: the first this many committed become fixed."""
        return self._pin_tokens

    @property
    def geometry(self) -> Tuple[int, int, Tuple[int, ...]]:
        """``(window_tokens, max_staged_tokens, causal_block_sizes)`` of the open rollout;
        the key of its device state, for callers that cache metadata derived from it."""
        return (self.window_tokens, self.max_staged_tokens, self.causal_block_sizes)

    @property
    def fixed_tokens(self) -> int:
        """Tokens every block sees for the whole rollout; logical ``[0, fixed_tokens)``."""
        return self._fixed_tokens

    @property
    def history_tokens(self) -> int:
        """Rolling tokens resident before the staged tokens, stale ones included."""
        return self._history_tokens

    @property
    def staging_offset(self) -> int:
        """Logical position where the staged tokens' K/V are written: the fixed region
        plus the resident history. A position in the resident sequence, not a count of
        tokens generated: it falls by a page whenever eviction drops one. Rotary
        positions are absolute over the rollout and are the caller's business."""
        return self._fixed_tokens + self._history_tokens

    @property
    def table(self) -> torch.Tensor:
        """``[num_pages]`` int32 device table of layer-0 view indices in logical order
        (a view of the pool-sized table a captured graph holds)."""
        self._require_open()
        return self._table[: self.num_pages]

    def request_ids(self, num_causal_blocks: int) -> List[int]:
        """The request ids attention metadata must carry: one per causal block, all the
        one rollout sequence."""
        return [_ROLLOUT_REQUEST_ID] * num_causal_blocks

    def block_table(self) -> List[int]:
        """Every base page in logical order, from the host copy of the table. For
        ``write_range`` and for tests; kernels read ``page_table`` directly."""
        self._require_open()
        return (self._host_table // self.page_view_scale).tolist()

    # ------------------------------------------------------------------ the window

    def commit(self, num_tokens: int) -> None:
        """The ``num_tokens`` staged tokens, written at ``staging_offset``, are final. Those
        that fall within the first ``pin_tokens`` become fixed; the rest become
        history, and the window slides. The count is the caller's: a forward may stage
        fewer tokens than ``max_staged_tokens``."""
        self._require_open()
        pinned = max(
            0, min(self._pin_tokens, self.staging_offset + num_tokens) - self._fixed_tokens
        )
        if num_tokens <= 0 or num_tokens - pinned > self.max_staged_tokens:
            raise ValueError(
                f"commit of {num_tokens} tokens, {pinned} of them pinned; the rest may be at "
                f"most max_staged_tokens {self.max_staged_tokens}"
            )
        if pinned:
            self._fixed_tokens += pinned
            self._fixed_pages = self._fixed_tokens // self.tokens_per_page
        self._history_tokens += num_tokens - pinned
        refill = None
        excess = self._history_tokens - self.window_tokens
        if excess > 0:
            drop_pages = excess // self.tokens_per_page
            if drop_pages:
                # Recycle: the oldest pages become the newest free pages. Rotated in
                # place, so the table's address, which captured graphs hold, never changes.
                ring = self._table[self._fixed_pages : self.num_pages]
                torch.ops.trtllm.rotate_rows_(ring, -drop_pages)  # dropped pages go to the tail
                host_ring = self._host_table[self._fixed_pages :]
                old_head = int(host_ring[0])
                host_ring[:] = np.roll(host_ring, -drop_pages)
                refill = (old_head, int(host_ring[0]))
                self._history_tokens -= drop_pages * self.tokens_per_page
        self._refresh_device_state(refill)

    # ------------------------------------------------------------------ causal blocks

    def _allocate_state(
        self,
        device: torch.device,
        max_staged_tokens: int,
        sizes: Tuple[int, ...],
        num_private: int,
    ) -> _GeometryState:
        """Allocate a geometry's layouts as views into two packed buffers (int32 and
        int64), each with a host twin, plus the pieces area and the staged-slot
        scratch; once per geometry."""
        tpb = self.tokens_per_page
        specs = []  # (size, n, region_pages, i32 fields, i64 fields)
        for size in sizes:
            n, region_pages = max_staged_tokens // size, self._region_pages(tpb, size)
            i32 = {
                # A row holds a block's whole pages then its region, never more than
                # the pool has; sized for the pool so every geometry's rows agree.
                "rows": (n, self._pool_pages),
                "block_offsets": (1, n, 2, self._pool_pages),
                "seq_len_q": (n,),
                "seq_len_kv": (n,),
            }
            i64 = {
                "own_slots": (n * size,),
                "extra_src": (n * self._start_entries,),
                "extra_dst": (n * self._start_entries,),
            }
            specs.append((size, n, region_pages, i32, i64))
        # Fixed and history slots copied per block: the shared page's refill, then
        # per block at most three partial pages.
        piece_capacity = tpb + sum(n * 3 * (tpb - 1) for _, n, _, _, _ in specs)
        numel_i32 = sum(int(np.prod(shape)) for *_, i32, _ in specs for shape in i32.values())
        numel_i64 = sum(int(np.prod(shape)) for *_, i64 in specs for shape in i64.values())
        numel_i64 += 2 * piece_capacity
        packed_i32 = torch.zeros(numel_i32, dtype=torch.int32, device=device)
        packed_i64 = torch.zeros(numel_i64, dtype=torch.int64, device=device)
        host_i32 = np.zeros(numel_i32, dtype=np.int32)
        host_i64 = np.zeros(numel_i64, dtype=np.int64)
        offsets = {"i32": 0, "i64": 0}

        def take(kind: str, shape) -> Tuple[torch.Tensor, np.ndarray]:
            dev, host = (packed_i32, host_i32) if kind == "i32" else (packed_i64, host_i64)
            numel, first = int(np.prod(shape)), offsets[kind]
            offsets[kind] += numel
            return dev[first : first + numel].view(shape), host[first : first + numel].reshape(
                shape
            )

        layouts: Dict[int, _CausalBlockLayout] = {}
        first_private = 0
        for size, n, region_pages, i32, i64 in specs:
            fields, host = {}, {}
            for kind, group in (("i32", i32), ("i64", i64)):
                for name, shape in group.items():
                    fields[name], host[name] = take(kind, shape)
            host["seq_len_q"][:] = size
            regions = self._host_private[first_private : first_private + n * region_pages]
            first_private += n * region_pages
            layouts[size] = _CausalBlockLayout(
                num_blocks=n,
                block_size=size,
                region_pages=region_pages,
                regions=regions.reshape(n, region_pages),
                host=host,
                cached=[0] * n,
                **fields,
            )
        assert first_private == num_private
        piece_src, host_piece_src = take("i64", (piece_capacity,))
        piece_dst, host_piece_dst = take("i64", (piece_capacity,))
        logical, logical_page, staged_slots = (
            torch.empty(max_staged_tokens, dtype=torch.int64, device=device) for _ in range(3)
        )
        view_page = torch.empty(max_staged_tokens, dtype=torch.int32, device=device)
        return _GeometryState(
            layouts=layouts,
            packed_i32=packed_i32,
            packed_i64=packed_i64,
            host_i32=host_i32,
            host_i64=host_i64,
            piece_src=piece_src,
            piece_dst=piece_dst,
            host_piece_src=host_piece_src,
            host_piece_dst=host_piece_dst,
            staged_slots=staged_slots,
            logical=logical,
            logical_page=logical_page,
            view_page=view_page,
        )

    def _bind_state(self, state: _GeometryState) -> None:
        """Make ``state`` the open geometry's: its tensors become the attributes the
        rest of the cache reads."""
        self._layouts = state.layouts
        self._packed_i32, self._packed_i64 = state.packed_i32, state.packed_i64
        self._host_i32, self._host_i64 = state.host_i32, state.host_i64
        self._piece_src, self._piece_dst = state.piece_src, state.piece_dst
        self._host_piece_src, self._host_piece_dst = state.host_piece_src, state.host_piece_dst
        self._staged_slots = state.staged_slots
        self._logical, self._logical_page, self._view_page = (
            state.logical,
            state.logical_page,
            state.view_page,
        )

    def _layout(self, causal_block_size: int) -> _CausalBlockLayout:
        self._require_open()
        layout = self._layouts.get(causal_block_size)
        if layout is None:
            raise ValueError(
                f"causal block size {causal_block_size} was not declared; the open geometry "
                f"serves {self.causal_block_sizes}"
            )
        return layout

    @property
    def max_causal_blocks(self) -> int:
        """Most causal blocks any declared size cuts the staged tokens into."""
        return max(self.max_staged_tokens // b for b in self.causal_block_sizes)

    def causal_block_lengths(self, causal_block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-block int32 device tensors ``(seq_len_q, seq_len_kv)`` for every block of
        that size when all ``max_staged_tokens`` are staged; a forward of fewer tokens
        uses the leading ones.

        ``seq_len_kv[i]`` is the number of keys in block ``i``'s table row: the fixed
        region, the window before the block, the earlier blocks and the block itself.
        Persistent, refreshed by ``commit``, so a captured forward keeps
        reading the right lengths.
        """
        layout = self._layout(causal_block_size)
        return layout.seq_len_q, layout.seq_len_kv

    def cached_tokens(self, causal_block_size: int) -> List[int]:
        """Keys in each block's row before the block's own tokens (host ints)."""
        return list(self._layout(causal_block_size).cached)

    def page_table(self, causal_block_size: int) -> torch.Tensor:
        """``[num_blocks, row_len]`` int32 device table of layer-0 view indices.

        Row ``i`` is block ``i``'s key sequence; entries past its length are 0 and
        never read. Persistent, refreshed in place by ``commit``.
        """
        return self._layout(causal_block_size).rows

    @staticmethod
    def causal_blocks(num_tokens: int, causal_block_size: Optional[int]) -> Tuple[int, int]:
        """The block size and block count a forward of ``num_tokens`` real tokens uses:
        one block of all of them when ``causal_block_size`` is None; otherwise the size
        must be positive and divide ``num_tokens``. Shared by the cache and the
        attention backends so they agree on the contract."""
        size = num_tokens if causal_block_size is None else int(causal_block_size)
        if size <= 0 or num_tokens % size:
            raise ValueError(
                f"{num_tokens} tokens do not split into causal blocks of {causal_block_size}"
            )
        return size, num_tokens // size

    def block_offsets(self, causal_block_size: int) -> torch.Tensor:
        """``[1, num_blocks, 2, row_len]`` int32 device tensor: ``page_table`` in the
        TRTLLM attention op's block-offset encoding (K plane, then V plane), the
        layout its ``kv_cache_block_offsets`` argument has. Persistent, refreshed in
        place by ``commit``, so a kernel bound to it reads the current rows without
        any metadata refresh."""
        return self._layout(causal_block_size).block_offsets

    def set_causal_block_size(self, causal_block_size: int) -> None:
        """Declare the block size the next ``copy_batch_block_offsets`` describes."""
        self._layout(causal_block_size)
        self._block_offsets_size = causal_block_size

    # ------------------------------------------------------------------ device state

    def _refresh_device_state(self, refill: Optional[Tuple[int, int]] = None) -> None:
        """Rewrite every kernel-facing device tensor for the current table, fixed region
        and ``past``. Runs on the host in ``open`` and ``commit``, never on the
        forward path."""
        # Staged token t lands at slot id view_page * tpb + slot, with view_page the
        # table entry of its logical page and slot its offset in that page.
        tpb = self.tokens_per_page
        torch.arange(
            self.staging_offset, self.staging_offset + self.max_staged_tokens, out=self._logical
        )
        torch.floor_divide(self._logical, tpb, out=self._logical_page)
        torch.index_select(self._table, 0, self._logical_page, out=self._view_page)
        torch.remainder(self._logical, tpb, out=self._staged_slots)
        self._staged_slots.add_(self._view_page, alpha=tpb)
        # Everything else is host arithmetic on the host table, uploaded at the end
        # with one plain copy per packed buffer. The first copy blocks the host until
        # the GPU has finished the forwards already queued, so the GPU then idles for
        # the host's remaining commit work and the next launch: a few microseconds
        # per chunk. Pinned buffers with asynchronous copies would hide that, at the
        # cost of double-buffering them across commits.
        # A rotation moves the page the fixed region's tail shares with the history;
        # its first slots are copied into the page that now starts the history first,
        # since the private regions' copies below read them from there.
        num_refill = self._queue_refill(refill)
        num_pieces = num_refill
        for layout in self._layouts.values():
            num_pieces = self._refresh_layout(layout, num_pieces)
        self._packed_i32.copy_(torch.from_numpy(self._host_i32))
        self._packed_i64.copy_(torch.from_numpy(self._host_i64))
        if num_refill:
            self._copy_pieces(0, num_refill)
        if num_pieces > num_refill:
            self._copy_pieces(num_refill, num_pieces)
        self._mirrors = (num_refill, num_pieces)

    def _queue_refill(self, refill: Optional[Tuple[int, int]]) -> int:
        """Queue the fixed region's tail slots from the shared page's old copy to its
        new one (view pages ``refill``) at the start of the pieces area; returns the count."""
        tail = self._fixed_tokens % self.tokens_per_page
        if refill is None or tail == 0:
            return 0
        old_page, new_page = refill
        slot = np.arange(tail)
        self._host_piece_src[:tail] = old_page * self._rows_per_page + slot
        self._host_piece_dst[:tail] = new_page * self._rows_per_page + slot
        return tail

    def _refresh_layout(self, blk: _CausalBlockLayout, num_pieces: int) -> int:
        """Rebuild one block size's host twins for the current state. Appends the fixed
        and history slots its private regions need to the pieces area from index
        ``num_pieces`` on; returns the new count.

        Block ``i``'s row is every page it may see in full, then its private region:
        first the visible slots of the pages it sees only partly, in logical order,
        then its own tokens.
        """
        past, size = self.staging_offset, blk.block_size
        blk.host["rows"].fill(0)
        for i in range(blk.num_blocks):
            spans = self._visible_spans(past + i * size)
            runs = self._whole_page_runs(spans)
            partial = self._partial_page_positions(spans, runs)
            region = blk.regions[i]
            self._fill_row(blk, i, runs, region)
            blk.cached[i] = sum(b - a for a, b in runs) * self.tokens_per_page + partial.size
            own = self._fill_own_slots(blk, i, region, first=partial.size)
            # Fixed and history tokens are copied now, for every layer; the staged
            # tokens of earlier blocks on partial pages are written by write_staged.
            k = np.arange(partial.size)
            static = partial < past
            num_pieces = self._queue_pieces(partial[static], region, k[static], num_pieces)
            self._fill_staged_pieces(blk, i, partial[~static] - past, region, k[~static], own[0])
        blk.host["seq_len_kv"][:] = np.asarray(blk.cached, dtype=np.int32) + size
        return num_pieces

    def _visible_spans(self, start: int) -> List[Tuple[int, int]]:
        """Logical positions a block starting at ``start`` sees before its own tokens:
        the fixed region and the window before it; one span when they touch."""
        fixed = self._fixed_tokens
        win_start = max(fixed, start - self.window_tokens)
        spans = [(0, start)] if win_start <= fixed else [(0, fixed), (win_start, start)]
        return [(lo, hi) for lo, hi in spans if hi > lo]

    def _whole_page_runs(self, spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Logical page ranges ``[a, b)`` the spans cover in full."""
        tpb = self.tokens_per_page
        return [(ceil_div(lo, tpb), max(ceil_div(lo, tpb), hi // tpb)) for lo, hi in spans]

    def _partial_page_positions(
        self, spans: List[Tuple[int, int]], runs: List[Tuple[int, int]]
    ) -> np.ndarray:
        """Visible positions on pages the spans cover only partly, ascending. At most
        three such pages: where the fixed region ends, where the window starts and
        where the block starts."""
        tpb = self.tokens_per_page
        edges = sorted({p for lo, hi in spans for p in (lo // tpb, (hi - 1) // tpb)})
        pages = [p for p in edges if not any(a <= p < b for a, b in runs)]
        pos = (np.asarray(pages, dtype=np.int64)[:, None] * tpb + np.arange(tpb)).ravel()
        visible = np.zeros(pos.shape, dtype=bool)
        for lo, hi in spans:
            visible |= (pos >= lo) & (pos < hi)
        return pos[visible]

    def _region_slot(self, region: np.ndarray, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """View page and slot of the ``k``-th slots of a private region."""
        return region[k // self.tokens_per_page], k % self.tokens_per_page

    def _fill_row(
        self, blk: _CausalBlockLayout, i: int, runs: List[Tuple[int, int]], region: np.ndarray
    ) -> None:
        """Block ``i``'s table row: its whole pages in logical order, then its region."""
        whole = np.concatenate([self._host_table[a:b] for a, b in runs] or [np.zeros(0, np.int64)])
        row = blk.host["rows"][i]
        row[: whole.size] = whole
        row[whole.size : whole.size + region.size] = region
        offsets = blk.host["block_offsets"][0, i]
        offsets[0] = row * self.kv_factor
        offsets[1] = offsets[0] + self._kv_offset

    def _fill_own_slots(
        self, blk: _CausalBlockLayout, i: int, region: np.ndarray, first: int
    ) -> np.ndarray:
        """Slot ids of block ``i``'s own tokens: its region from slot ``first`` on."""
        size = blk.block_size
        page, slot = self._region_slot(region, first + np.arange(size))
        own = blk.host["own_slots"][i * size : (i + 1) * size]
        own[:] = page * self.tokens_per_page + slot
        return own

    def _queue_pieces(
        self, positions: np.ndarray, region: np.ndarray, k: np.ndarray, num_pieces: int
    ) -> int:
        """Append copies of resident ``positions`` to region slots ``k`` (layer-0 pool
        rows) to the pieces area; returns the new count."""
        tpb, rpp = self.tokens_per_page, self._rows_per_page
        page, slot = self._region_slot(region, k)
        stop = num_pieces + positions.size
        self._host_piece_src[num_pieces:stop] = (
            self._host_table[positions // tpb] * rpp + positions % tpb
        )
        self._host_piece_dst[num_pieces:stop] = page * rpp + slot
        return stop

    def _fill_staged_pieces(
        self,
        blk: _CausalBlockLayout,
        i: int,
        tokens: np.ndarray,
        region: np.ndarray,
        k: np.ndarray,
        own_first: int,
    ) -> None:
        """Block ``i``'s entries for the staged ``tokens`` on its partial pages, written
        to region slots ``k`` by write_staged. Padded to a fixed count with a harmless
        repeat of the block's own first token, also when that token is written by the
        attention kernel itself (``own_tokens=False``): same value, same slot."""
        per_block = self._start_entries
        if tokens.size > per_block:
            raise RuntimeError("more staged tokens on partial pages than two pages hold")
        page, slot = self._region_slot(region, k)
        entries = slice(i * per_block, (i + 1) * per_block)
        src, dst = blk.host["extra_src"][entries], blk.host["extra_dst"][entries]
        src[:] = i * blk.block_size
        dst[:] = own_first
        src[: tokens.size] = tokens
        dst[: tokens.size] = page * self.tokens_per_page + slot

    def _copy_pieces(self, first: int, stop: int) -> None:
        """Copy pieces ``[first, stop)`` (layer-0 slot rows, uploaded) for every layer,
        K and V, every head: one gather and one scatter over the whole pool."""
        src = (self._piece_src[first:stop, None] + self._piece_offsets).view(-1)
        dst = (self._piece_dst[first:stop, None] + self._piece_offsets).view(-1)
        pool = self.kv_buffer(0)
        pool_rows = pool.view(-1, pool.shape[-1])
        pool_rows.index_copy_(0, dst, pool_rows.index_select(0, src))

    # ------------------------------------------------------------------ direct pool access

    def kv_buffer(self, layer_idx: int) -> torch.Tensor:
        """The layer's pool view ``[view_pages, 2, num_kv_heads, tokens_per_page, head_dim]``.

        Base page ``p`` of this layer is ``buf[p * page_view_scale]``.
        """
        self._require_open()
        return self._buffers[layer_idx]

    def write_range(self, layer_idx: int, start: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Write ``k``/``v`` ``[T, num_kv_heads, head_dim]`` at logical ``[start, start + T)``.

        Eager, host-addressed: a partial page at either end is one copy each, whole
        pages go one group of physically consecutive pages at a time. Committed tokens
        that block rows read through a private copy are updated in those copies too.
        For the fixed region and for tests; the per-forward write is ``write_staged``.
        """
        if v.shape != k.shape:
            raise ValueError(f"k/v shape mismatch: {tuple(k.shape)} vs {tuple(v.shape)}")
        buf = self.kv_buffer(layer_idx)
        if k.dtype != buf.dtype:
            raise TypeError(f"K/V dtype {k.dtype} does not match the cache's {buf.dtype}")
        self._for_each_page_run(
            start,
            k.shape[0],
            lambda page, slot, t0, t1: self._write_page_slots(buf, page, slot, k[t0:t1], v[t0:t1]),
            lambda page, count, t0, t1: self._write_whole_pages(
                buf, page, count, k[t0:t1], v[t0:t1]
            ),
        )
        pieces = self._mirror_pieces(start, k.shape[0])
        if pieces.size:
            src, dst = self._piece_rows(pieces, layer_idx, (0, 1))
            pool_rows = self.kv_buffer(0).view(-1, buf.shape[-1])
            pool_rows.index_copy_(0, dst, pool_rows.index_select(0, src))

    def zero_values(self, layer_idx: int, start: int, num_tokens: int) -> None:
        """Zero the V vectors (all heads) of logical ``[start, start + num_tokens)`` in
        one layer, leaving K as written, in the home slots and in every private copy
        block rows read. Eager, host-addressed, like ``write_range``."""
        buf = self.kv_buffer(layer_idx)
        vs = self.page_view_scale

        def slots(page, slot, t0, t1):
            buf[page * vs, 1, :, slot : slot + t1 - t0].zero_()

        def whole(page, count, t0, t1):
            buf[page * vs : (page + count) * vs : vs, 1].zero_()

        self._for_each_page_run(start, num_tokens, slots, whole)
        pieces = self._mirror_pieces(start, num_tokens)
        if pieces.size:
            _, dst = self._piece_rows(pieces, layer_idx, (1,))
            self.kv_buffer(0).view(-1, buf.shape[-1]).index_fill_(0, dst, 0)

    def _mirror_pieces(self, start: int, num_tokens: int) -> np.ndarray:
        """Indices of the live commit-time pieces whose source token lies in logical
        ``[start, start + num_tokens)``: the private copies an edit of those tokens
        must reach. The refill pieces are excluded; their sources are a page the
        table no longer points at."""
        lo, hi = self._mirrors
        if lo == hi or num_tokens == 0:
            return np.empty(0, dtype=np.int64)
        tpb, rpp = self.tokens_per_page, self._rows_per_page
        pos = np.arange(start, start + num_tokens)
        home = self._host_table[pos // tpb] * rpp + pos % tpb
        return lo + np.flatnonzero(np.isin(self._host_piece_src[lo:hi], home))

    def _piece_rows(
        self, pieces: np.ndarray, layer_idx: int, planes: Sequence[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Device index tensors ``(src, dst)`` of the pool rows ``pieces`` occupy in
        ``layer_idx`` for the given planes (0 = K, 1 = V), every head."""
        tpb, rpp, heads = self.tokens_per_page, self._rows_per_page, self._kv_heads_local
        offsets = np.array(
            [layer_idx * rpp + (plane * heads + h) * tpb for plane in planes for h in range(heads)],
            dtype=np.int64,
        )
        src = (self._host_piece_src[pieces][:, None] + offsets).reshape(-1)
        dst = (self._host_piece_dst[pieces][:, None] + offsets).reshape(-1)
        device = self._table.device
        return torch.from_numpy(src).to(device), torch.from_numpy(dst).to(device)

    def _for_each_page_run(self, start: int, n: int, on_slots, on_whole_pages) -> None:
        """Walk logical ``[start, start + n)`` as page runs: ``on_slots(page, slot, t0,
        t1)`` for a partial page at either end, ``on_whole_pages(first_page, count,
        t0, t1)`` for physically consecutive whole pages; ``t0:t1`` index the tokens."""
        self._require_open()
        if n == 0:
            return
        if not 0 <= start <= start + n <= self.capacity:
            raise ValueError(f"[{start}, {start + n}) outside the cache's {self.capacity} tokens")
        tpb = self.tokens_per_page
        table = self.block_table()
        end = start + n
        first_whole, end_whole = ceil_div(start, tpb), end // tpb  # whole pages [first, end)
        if first_whole > end_whole:  # the whole range lies inside one page
            on_slots(table[start // tpb], start % tpb, 0, n)
            return
        if start % tpb:
            head = first_whole * tpb - start
            on_slots(table[start // tpb], start % tpb, 0, head)
        page_idx = first_whole
        while page_idx < end_whole:
            run_end = page_idx + 1
            while run_end < end_whole and table[run_end] == table[run_end - 1] + 1:
                run_end += 1
            on_whole_pages(
                table[page_idx], run_end - page_idx, page_idx * tpb - start, run_end * tpb - start
            )
            page_idx = run_end
        if end % tpb:
            on_slots(table[end_whole], 0, end_whole * tpb - start, n)

    def _write_page_slots(
        self, buf: torch.Tensor, page: int, slot: int, k: torch.Tensor, v: torch.Tensor
    ) -> None:
        """A run of tokens inside one page, at ``slot`` onward; one copy per tensor."""
        page *= self.page_view_scale
        buf[page, 0, :, slot : slot + k.shape[0]].copy_(k.transpose(0, 1))
        buf[page, 1, :, slot : slot + v.shape[0]].copy_(v.transpose(0, 1))

    def _write_whole_pages(
        self, buf: torch.Tensor, first_page: int, count: int, k: torch.Tensor, v: torch.Tensor
    ) -> None:
        """``count`` whole, physically consecutive pages starting at ``first_page``; one copy per tensor."""
        tpb, vs = self.tokens_per_page, self.page_view_scale
        dst = slice(first_page * vs, (first_page + count) * vs, vs)
        buf[dst, 0].copy_(k.view(count, tpb, -1, k.shape[-1]).transpose(1, 2))
        buf[dst, 1].copy_(v.view(count, tpb, -1, v.shape[-1]).transpose(1, 2))

    def write_staged(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        causal_block_size: Optional[int] = None,
        *,
        own_tokens: bool = True,
    ) -> None:
        """Stage ``T`` tokens of K/V, the forward's new tokens, cut into causal blocks
        of ``causal_block_size`` (default: one block of ``T``).

        ``k``/``v`` are ``[T, num_kv_heads, head_dim]`` with a contiguous ``head_dim``;
        token and head strides are free, so slices of a fused QKV projection go in
        without a copy. They go to the logical positions from ``staging_offset`` (where
        later blocks and, after commit, later forwards read them) and to the blocks'
        private regions: the earlier blocks' tokens on each block's partial pages,
        and, with ``own_tokens``, each block's own tokens. One scatter kernel reads
        K/V once for the logical and own-token slots, a second (several blocks only)
        handles the partial pages; both are driven by slot ids rebuilt on
        ``commit``, so the write replays correctly inside a CUDA graph.
        A kernel that writes the new tokens into the region itself (trtllm-gen)
        passes ``own_tokens=False``.
        """
        self._require_open()
        num_tokens, heads, head_dim = k.shape
        if v.shape != k.shape:
            raise ValueError(f"k/v shape mismatch: {tuple(k.shape)} vs {tuple(v.shape)}")
        buf = self.kv_buffer(layer_idx)
        if num_tokens > self.max_staged_tokens or (heads, head_dim) != (buf.shape[2], buf.shape[4]):
            raise ValueError(
                f"staged write of [{num_tokens}, {heads}, {head_dim}] does not fit "
                f"[{self.max_staged_tokens}, {buf.shape[2]}, {buf.shape[4]}]"
            )
        if k.dtype != buf.dtype or v.dtype != buf.dtype:
            raise TypeError(f"K/V dtype {k.dtype} does not match the cache's {buf.dtype}")
        size, num_blocks = self.causal_blocks(num_tokens, causal_block_size)
        layout = self._layout(size)
        if num_blocks > layout.num_blocks:
            raise ValueError(
                f"{num_tokens} tokens are more than the {layout.num_blocks} causal blocks of "
                f"{size} this geometry stages"
            )
        torch.ops.trtllm.scatter_kv_slots_(
            buf,
            k,
            v,
            self._staged_slots[:num_tokens],
            layout.own_slots[:num_tokens] if own_tokens else None,
        )
        if num_blocks > 1:
            extra = num_blocks * self._start_entries
            torch.ops.trtllm.scatter_kv_slots_(
                buf, k, v, layout.extra_dst[:extra], src=layout.extra_src[:extra]
            )

    # ------------------------------------------------------------------ manager hook

    def copy_batch_block_offsets(
        self,
        dst_tensor: torch.Tensor,
        request_ids: List[int],
        beam_width: int,
        num_contexts: int,
        num_seqs: int,
        max_blocks: Optional[int] = None,
    ) -> None:
        """Hand the attention metadata *our* rows instead of the manager's.

        ``dst_tensor`` is ``[num_pools, max_num_sequences, 2, max_blocks_per_seq]``
        of int32: K offsets then V offsets per sequence, each ``page * index_scale``
        (+ ``kv_offset`` for V) -- the same encoding the manager's device copy uses.
        Every request is a causal block of the size declared with
        ``set_causal_block_size``; row ``i`` is that block's key sequence.
        """
        ids = list(request_ids)
        if (
            not 0 < len(ids) <= self.max_causal_blocks
            or any(r != _ROLLOUT_REQUEST_ID for r in ids)
            or num_seqs != len(ids)
            or beam_width != 1
        ):
            raise ValueError(
                f"CausalKVCacheManager serves one sequence (id {_ROLLOUT_REQUEST_ID}) in up to "
                f"{self.max_causal_blocks} causal blocks; got request_ids={ids}, "
                f"num_seqs={num_seqs}, beam_width={beam_width}"
            )
        size = self._block_offsets_size
        if size is None or num_seqs > self._layouts[size].num_blocks:
            raise ValueError(
                f"block offsets for {num_seqs} sequences need set_causal_block_size "
                f"first with a size that cuts the staged tokens into at least that many blocks; "
                f"declared: {size}"
            )
        offsets = self._layouts[size].block_offsets[0, :num_seqs]
        n = offsets.shape[-1]
        dst_tensor[0, :num_seqs, :, :n] = offsets
        if n < dst_tensor.shape[-1]:
            dst_tensor[0, :num_seqs, :, n:].zero_()

    def get_batch_cache_indices(
        self,
        request_ids: List[int],
        layer_idx: Optional[int] = None,
        num_blocks_per_seq: Optional[Sequence[int]] = None,
    ) -> List[List[int]]:
        """The rotated table in V2's per-layer view units, one row per request.

        Overrides the inherited accessor, which would report the pages in
        allocation order and disagree with what the attention backends read.
        Every request is the one sequence.
        """
        ids = list(request_ids)
        if any(r != _ROLLOUT_REQUEST_ID for r in ids):
            raise ValueError(
                f"CausalKVCacheManager serves request {_ROLLOUT_REQUEST_ID} only; got {ids}"
            )
        row = self._host_table.tolist()
        widths = [len(row)] * len(ids) if num_blocks_per_seq is None else list(num_blocks_per_seq)
        return [row[:n] for n in widths]

    def get_batch_cache_indices_flat(
        self, request_ids: List[int], num_blocks: List[int], layer_idx: Optional[int] = None
    ) -> torch.Tensor:
        rows = self.get_batch_cache_indices(request_ids, layer_idx, num_blocks)
        return torch.tensor([p for row in rows for p in row], dtype=torch.int32)

    @property
    def is_open(self) -> bool:
        """Whether a rollout is open; device state of a closed cache is not live."""
        return self._kv_cache is not None

    def _require_open(self) -> None:
        if self._kv_cache is None:
            raise RuntimeError("cache is not open; call open() first")
