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
"""Paged K/V cache for a block-causal video rollout.

One rollout is one logical sequence laid out as

    [ fixed region | rolling history | staged tokens ]

held in a single pool of ``KVCacheManagerV2``. The manager allocates the pages;
the page table, the sliding window and the fixed region are this class's own.

Taken from V2:

- the pool and its pages: one sequence (``_create_kv_cache(is_dummy=True)``)
  resized to cover the whole pool;
- the per-layer views of the pool (``get_buffers``, the page index scale,
  ``kv_offset``);
- the attributes the TRTLLM attention op reads through its ``kv_cache_manager``
  (pool pointers and pool mapping, ``layer_offsets``, ``tokens_per_block``,
  ``max_seq_len``), with ``copy_batch_block_offsets`` overridden to hand it each
  causal block's page list.

Switched off: page reuse (``enable_block_reuse=False``, ``stop_committing()``),
request scheduling (no ``LlmRequest`` is created), the host tier
(``host_cache_size=0``) and V2's own sliding window (its window is the whole pool).

Why the window is not V2's:

- V2's window is per query token over absolute page ordinals, and the kernel
  masks it; here all queries of a causal block share one window start and attend
  fully within the causal block.
- The PyTorch path has no attention sinks (V2 builds its layers with
  ``num_sink_tokens=None``; ``sink_token_length`` is ignored), hence the fixed
  region.
- V2's page lists grow with absolute position up to ``max_seq_len``; the rotated
  page table stays at ``num_pages`` pages.

The fixed region is the first ``pin_tokens`` tokens of the sequence, declared
by ``open``: every causal block sees them for the whole rollout and they are never
evicted. Tokens become fixed when they are committed, so nothing can be pinned
after it was evicted. What those tokens are is the model's business. The
rolling history slides: ``commit`` turns the staged tokens (written by forwards, not
yet committed) into history and drops
whole pages from the front once the history exceeds the window, by rotating
the table; no history moves. When the fixed region ends mid-page the first
history tokens share its last page, and after a rotation the fixed region's
tail is copied into the page that now starts the history.

Attention reads exactly what the model's window says. A paged kernel reads
whole pages in list order and cuts only at the end, so every causal block of a
forward gets its own page list: all pages the causal block attends to in full,
then a few private pages holding copies of the tokens it attends to on its partial
pages (committed_private and current_private), followed by its own tokens. The
kernel is told the cached length up to the causal block's own tokens and reads
nothing older than the window and nothing of later causal blocks. See
``_CausalBlockLayout``.

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
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.math_utils import ceil_div

from .rollout_kv_cache_interface import ForwardPlan, RolloutKVCacheInterface, RolloutSpec

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

    Causal block ``i`` holds staged tokens ``[i*causal_block_size,
    (i+1)*causal_block_size)`` and attends to the fixed region, the ``window_tokens``
    before it, the earlier causal blocks and itself. Its page list holds every page it
    attends to in full, then its ``num_private_pages`` private pages. These hold, in
    order, copies of the tokens it attends to on its partial pages (at most three:
    where the fixed region ends, where the window starts, where the causal block
    starts), then its own tokens. ``cached_tokens`` counts the keys before each causal
    block's own tokens.

    The copies on partial pages come in two kinds. committed_private: committed tokens
    (fixed or history), copied at ``commit``. current_private: earlier causal blocks'
    tokens of the same forward, written by every ``write_staged``, as are each causal
    block's own tokens. All device tensors are persistent views into the cache's packed
    buffers, rewritten in place on the device by ``commit``, never on the forward path.
    """

    num_causal_blocks: int
    causal_block_size: int
    num_private_pages: int
    # [num_causal_blocks, num_private_pages] int32 layer-0 view indices
    private_pages: torch.Tensor
    host_private_pages: np.ndarray  # the same on the host, for host-addressed edits
    # [num_causal_blocks, max_pages] int32 layer-0 view indices, 0-padded: each causal
    # block's page list.
    causal_block_pages: torch.Tensor
    # The page lists in the TRTLLM attention op's offset encoding, K plane then V
    # plane: ``page * kv_factor`` and ``+ kv_offset``. The op reads this tensor directly.
    causal_block_offsets: torch.Tensor  # [1, num_causal_blocks, 2, max_pages] int32
    seq_len_q: torch.Tensor  # [num_causal_blocks] int32, all causal_block_size
    seq_len_kv: torch.Tensor  # [num_causal_blocks] int32, cached + causal_block_size
    # Slot ids (``view_page * tokens_per_page + slot``) of the private slots holding
    # each staged token as one of its causal block's own tokens.
    own_slots: torch.Tensor  # [num_causal_blocks*causal_block_size] int64
    # current_private: earlier causal blocks' tokens of this forward on a causal block's
    # partial pages (at most two: where its window starts and where it starts), as the
    # staged token and the private slot it goes to. A fixed 2*(tpb-1) entries per causal
    # block, padded with a harmless repeat of its own first token, so the first ``n``
    # causal blocks' entries are a prefix and a captured forward replays them.
    current_private_src: torch.Tensor  # [num_causal_blocks*2*(tpb-1)] int64 staged token index
    current_private_dst: torch.Tensor  # [num_causal_blocks*2*(tpb-1)] int64 slot id
    # committed_private: committed tokens on a causal block's partial pages, copied into
    # its private pages at commit. 3*(tpb-1) entries per causal block, (source pool row,
    # destination pool row), -1 past the used ones.
    committed_private_src: torch.Tensor  # [num_causal_blocks*3*(tpb-1)] int64
    committed_private_dst: torch.Tensor  # [num_causal_blocks*3*(tpb-1)] int64


@dataclass
class _SpecState:
    """Device state a spec's forwards read, allocated the first time that spec is
    opened and kept at the same addresses for the life of the cache, so a CUDA graph
    captured over it stays valid when the spec is opened again.
    Every per-layout tensor is a view into one of two packed buffers, rewritten in
    place on the device by ``commit``."""

    layouts: Dict[int, _CausalBlockLayout]  # by causal block size
    packed_i32: torch.Tensor
    packed_i64: torch.Tensor
    # Every copy made at commit, as layer-0 pool rows: the fixed tail's refill (one
    # page of entries) first, then each layout's committed_private; one launch copies
    # them all.
    commit_copy_src: torch.Tensor
    commit_copy_dst: torch.Tensor
    refill_src: torch.Tensor
    refill_dst: torch.Tensor
    # [max_staged_tokens] maps each staged token's logical position to its home slot id
    # (view_page * tokens_per_page + slot) on the shared pages, where later causal blocks
    # and later commits read it.
    staged_slots: torch.Tensor


class BlockCausalKVCacheManager(KVCacheManagerV2, RolloutKVCacheInterface):
    """``KVCacheManagerV2`` driven as one long-lived sequence with a caller-owned page
    table.

    The constructor allocates the page pool; ``open`` lays a rollout's ``RolloutSpec``
    out inside it: the pinned prefix, the sliding window, the staged tokens (written by
    forwards, not yet committed) and the causal blocks' private pages.

    One instance is one pool holding one open rollout at a time, batch size 1.
    Several streams (for example the two branches of classifier-free guidance) take
    one instance each.

    A pipeline drives the rollout through ``RolloutKVCacheInterface`` (``open``,
    ``plan``, ``commit``, ``close``, ``shutdown``) and puts the prompt in with
    ``write_range(layer_idx, 0, k, v)`` on every layer followed by ``commit`` of its
    length. The other members are the paged layout's own and serve its attention
    backends, tests and benchmark: ``write_staged``, ``write_range`` past the fixed
    region, ``zero_values``, ``kv_buffer``, ``table``/``page_table``,
    ``causal_block_pages``/``causal_block_lengths``/``causal_block_offsets``,
    ``cached_tokens``, ``staging_offset``, ``history_tokens``, ``capacity`` and the
    pool sizing helpers.

    Terms, used the same way in this cache, its attention backends and its kernels.
    "Block" alone is never used: in the LLM code it means a page.

    - page: the pool's unit of storage, ``tokens_per_page`` tokens. ``KVCacheManagerV2``
      and the TRTLLM attention op call it a block (``tokens_per_block``,
      ``kv_cache_block_offsets``).
    - page table: the cache's map from logical pages to pool pages, ``page_table()``.
    - rollout: one generated video, from ``open`` to ``close``: the fixed region and every
      chunk generated after it, one chunk at a time. One instance holds one open rollout.
    - chunk: the frames one forward processes. Their tokens are the staged tokens.
    - causal block: a run of staged tokens with full attention among themselves,
      causal to earlier causal blocks and to the cache. One request in the TRTLLM
      attention metadata. A denoising step is one causal block of the whole chunk; the
      clean pass is one causal block per frame.
    - causal block pages: the pages one causal block reads, in key order: those it
      attends to in full, then its private pages. ``causal_block_pages(size)``.
    - private pages: pages owned by one causal block. They hold copies of the tokens
      it attends to on its partial pages, then its own tokens.
    - partial page: a page the causal block attends to only in part; at most three,
      where the fixed region ends, where the window starts and where the causal block
      starts. The tokens it attends to on one such page form a segment.

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
            ``pool_tokens_for`` gives what a ``RolloutSpec`` needs.
    """

    @staticmethod
    def _private_pages_per_causal_block(tokens_per_page: int, causal_block_size: int) -> int:
        """Private pages one causal block needs: up to three partial pages' worth of
        visible slots plus its own tokens."""
        return ceil_div(3 * (tokens_per_page - 1) + causal_block_size, tokens_per_page)

    @staticmethod
    def _spec_pages(tokens_per_page: int, spec: RolloutSpec) -> Tuple[int, int]:
        """A spec's page counts: ``(resident, private)``. Resident tokens peak at pin +
        window + (tpb - 1) stale + staged; the extra page covers the stale tokens and an
        unaligned fixed-region end."""
        tpb = tokens_per_page
        if tpb <= 0 or tpb & (tpb - 1):
            raise ValueError(f"tokens_per_page must be a power of two, got {tpb}")
        resident = (
            ceil_div(spec.max_pin_tokens + spec.window_tokens + spec.max_staged_tokens, tpb) + 1
        )
        private = sum(
            n * BlockCausalKVCacheManager._private_pages_per_causal_block(tpb, size)
            for size, n in spec.causal_blocks.items()
        )
        return resident, private

    @staticmethod
    def pool_tokens_for(spec: RolloutSpec, *, tokens_per_page: int) -> int:
        """Pool tokens a spec needs: resident pages (pinned prefix, window, staged
        tokens, one spare) plus the private pages of its causal blocks. A pool of at least
        this many tokens can ``open`` the spec."""
        resident, private = BlockCausalKVCacheManager._spec_pages(tokens_per_page, spec)
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
        spec: RolloutSpec,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        tokens_per_page: int,
    ) -> int:
        """Device bytes a pool sized for this spec takes from the GPU."""
        return BlockCausalKVCacheManager.bytes_for_pool_tokens(
            BlockCausalKVCacheManager.pool_tokens_for(spec, tokens_per_page=tokens_per_page),
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
        """Tokens the page pool holds, every spec included."""
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
            raise ValueError(f"BlockCausalKVCacheManager supports bf16/fp16 K/V, got {dtype}")
        if tokens_per_page <= 0 or tokens_per_page & (tokens_per_page - 1):
            # The commit-time layout kernel computes pages and slots with shifts and masks.
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
        # current_private entries per causal block: earlier causal blocks' tokens of the
        # same forward on at most two partial pages, where its window starts and where it
        # starts.
        self._current_private = 2 * (tpb - 1)

        kv_cache_config = KvCacheConfig(
            max_tokens=pool_tokens,
            max_attention_window=[pool_tokens],
            tokens_per_block=tpb,
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
        # Metadata attention backends build over this cache, keyed by backend and
        # causal block size. Kept here, not by the backend, so the cache's lifetime is the
        # caller's; the entries hold the cache, so shutdown() clears them.
        self.attention_metadata: Dict[Tuple, object] = {}
        # The open spec and its fields; set by open().
        self._spec: Optional[RolloutSpec] = None
        self.window_tokens = 0
        self.max_staged_tokens = 0
        self.causal_block_sizes: Tuple[int, ...] = ()
        self.num_pages = 0  # resident pages: pinned prefix, window, staged tokens, spare
        self.capacity = 0  # resident tokens, num_pages * tokens_per_page
        self._pin_tokens = 0
        self._fixed_tokens = 0
        self._history_tokens = 0
        # Most tokens a plan or a staged write covered since the last commit. Plans are
        # made on the host for every forward, so this holds under CUDA graph replay too.
        self._staged_tokens = 0
        # Device-side state read by the kernels. Every tensor here lives for the
        # life of the cache and is rewritten in place by open() and commit(), never
        # on the forward path, so a CUDA graph captured around a forward keeps
        # reading correct values after the cache moves.
        #
        # The page table: layer-0 view indices in logical order, the fixed region's full
        # pages first (never rotated), then the ring: the page the fixed region's
        # tail shares with the first history tokens (if any), history, free pages.
        # Sized for the whole pool; a spec uses the first num_pages entries.
        self._table: Optional[torch.Tensor] = None  # [pool_pages] int32
        self._fixed_pages = 0
        # Host copy of the resident page table, rotated in lockstep with the device one, so
        # that nothing is ever read back; and the private pages' view indices.
        self._host_table: Optional[np.ndarray] = None  # [num_pages] int64
        self._host_private: Optional[np.ndarray] = None  # [pool_pages] int64
        # Per-spec state, and the open spec's fields bound as attributes.
        self._states: Dict[RolloutSpec, _SpecState] = {}
        self._layouts: Dict[int, _CausalBlockLayout] = {}
        self._commit_copy_src: Optional[torch.Tensor] = None
        self._commit_copy_dst: Optional[torch.Tensor] = None
        self._refill_src: Optional[torch.Tensor] = None
        self._refill_dst: Optional[torch.Tensor] = None
        self._staged_slots: Optional[torch.Tensor] = None
        self._offsets_causal_block_size: Optional[int] = None
        self._kv_heads_local = 0
        self._rows_per_page = 0  # pool rows of head_dim per view page: 2 * H * tpb
        self._kv_offset = 0  # the V plane's offset in the TRTLLM op's offset encoding

    def _reserve_guard_page(self) -> None:
        """No guard page: the attention backends over this cache never park page-list
        entries on one, and ``open`` claims every page of the pool."""
        if self._guard_page_fill:
            logger.warning(
                "BlockCausalKVCacheManager: TRTLLM_KV_GUARD_PAGE is ignored; this cache "
                "does not use a guard page"
            )
        self._guard_page_fill = ""

    # ------------------------------------------------------------------ lifecycle

    def open(self, spec: RolloutSpec, *, pin_tokens: int = 0) -> "BlockCausalKVCacheManager":
        """Start a rollout with this spec; the sequence starts empty. Returns the cache,
        so ``with cache.open(spec, pin_tokens=n):`` closes the rollout on exit.

        Args:
            spec: the rollout's geometry. It must fit the pool: ``pool_tokens_for`` of
                it at most ``pool_tokens``.
            pin_tokens: the first tokens committed become the fixed region: always
                visible, never evicted. At most ``spec.max_pin_tokens``. Commit them like
                any other tokens; a commit may exceed ``max_staged_tokens`` as long as
                the excess is pinned.

        Device state is allocated per spec the first time it is opened and kept, so
        graphs captured over a spec replay after it is opened again, whatever
        ``pin_tokens`` the reopen takes.
        """
        if self._kv_cache is not None:
            raise RuntimeError("cache already open; call close() first")
        if not 0 <= pin_tokens <= spec.max_pin_tokens:
            raise ValueError(
                f"pin_tokens {pin_tokens} must be within [0, spec.max_pin_tokens "
                f"{spec.max_pin_tokens}]"
            )
        tpb = self.tokens_per_page
        num_pages, num_private = self._spec_pages(tpb, spec)
        needed = (num_pages + num_private) * tpb
        if needed > self._pool_tokens:
            raise ValueError(
                f"{spec} needs {needed} pool tokens; the pool holds {self._pool_tokens}"
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
            self._remove_kv_cache(_ROLLOUT_REQUEST_ID)
            raise

        total_pages = num_pages + num_private
        # V2 owns and may rewrite that buffer: copy it.
        pages = np.array(kv_cache.get_base_page_indices(0), dtype=np.int64)[:total_pages]
        if pages.size != total_pages or pages.min() < 0:
            self._remove_kv_cache(_ROLLOUT_REQUEST_ID)
            raise RuntimeError(f"expected {total_pages} backed pages, got {pages}")
        # Logical order is ours to choose. Ascending keeps consecutive page ids
        # consecutive after rotation whenever V2 handed out a contiguous range, which
        # lets write_range copy a prompt in few copies; nothing depends on it.
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
                self._remove_kv_cache(_ROLLOUT_REQUEST_ID)
                raise RuntimeError("K/V pool layout changed: layers are not page-interleaved")
        self._buffers = buffers
        self._kv_cache = kv_cache
        try:
            self._open_with_spec(kv_cache, pages, buf, spec, pin_tokens, num_pages, num_private)
        except Exception:
            # Nothing may look open after a failed open: release the sequence and
            # forget the half-built state, so the caller can retry.
            self._kv_cache = None
            self._buffers = []
            self._remove_kv_cache(_ROLLOUT_REQUEST_ID)
            raise
        return self

    def __enter__(self) -> "BlockCausalKVCacheManager":
        self._require_open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _open_with_spec(
        self,
        kv_cache,
        pages: np.ndarray,
        buf: torch.Tensor,
        spec: RolloutSpec,
        pin_tokens: int,
        num_pages: int,
        num_private: int,
    ) -> None:
        """The spec-dependent half of ``open``: bind the pages, allocate or reuse the
        spec's device state and write the kernel-facing tensors."""
        tpb = self.tokens_per_page
        self._spec = spec
        self.window_tokens = spec.window_tokens
        self.max_staged_tokens = spec.max_staged_tokens
        self.causal_block_sizes = spec.causal_block_sizes
        self.num_pages = num_pages
        self.capacity = num_pages * tpb
        self._pin_tokens = pin_tokens
        self._fixed_tokens = 0
        self._history_tokens = 0
        self._staged_tokens = 0
        self._fixed_pages = 0
        device = buf.device
        self._kv_heads_local = buf.shape[2]
        self._rows_per_page = 2 * self._kv_heads_local * tpb
        self._kv_offset = int(self.kv_offset[0])
        scaled = pages * self.page_view_scale
        if self._table is None:
            self._table = torch.zeros(self._pool_pages, dtype=torch.int32, device=device)
            self._host_private = np.zeros(self._pool_pages, dtype=np.int64)
        # Reopened: the device tensors a captured forward points at stay where they
        # are and only their contents change.
        self._table[:num_pages].copy_(torch.from_numpy(scaled[:num_pages]))
        self._host_private[:num_private] = scaled[num_pages:]
        self._host_table = scaled[:num_pages].copy()
        state = self._states.get(spec)
        if state is None:
            state = self._allocate_state(device, spec, num_private)
            self._states[spec] = state
        self._bind_state(state)
        for layout in self._layouts.values():
            layout.private_pages.copy_(torch.from_numpy(layout.host_private_pages.astype(np.int32)))
        self._offsets_causal_block_size = None
        self._refresh_device_state()

    def close(self) -> None:
        if self._kv_cache is None:
            return
        self._kv_cache = None
        # Device state outlives the sequence so that forwards captured in a CUDA
        # graph before close() stay valid after the next open().
        self._offsets_causal_block_size = None
        self._remove_kv_cache(_ROLLOUT_REQUEST_ID)

    def shutdown(self) -> None:
        """Release the pool. Required, like ``free`` after ``malloc``: the cache's GPU
        memory and the attention metadata built over it are released here. The metadata
        refers back to the cache, so a cache dropped without ``shutdown()`` is left to
        Python's cyclic garbage collector. Closes the open rollout first; the cache cannot
        be reopened afterwards."""
        self.close()
        self.attention_metadata.clear()
        super().shutdown()
        self._shut_down = True

    @property
    def is_shut_down(self) -> bool:
        """Whether the pool is gone for good; a closed cache may still be reopened."""
        return self._shut_down

    # ------------------------------------------------------------------ the open rollout

    @property
    def tokens_per_page(self) -> int:
        """Page size of the K/V pool; the parent class calls it ``tokens_per_block``."""
        return self.tokens_per_block

    @property
    def pin_tokens(self) -> int:
        """Tokens the open rollout pins: the first this many committed become fixed."""
        return self._pin_tokens

    @property
    def fixed_tokens(self) -> int:
        """Tokens every causal block sees for the whole rollout; logical ``[0, fixed_tokens)``."""
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
        """``[num_pages]`` int32 device page table of layer-0 view indices in logical
        order (a view of the pool-sized table a captured graph holds)."""
        self._require_open()
        return self._table[: self.num_pages]

    def request_ids(self, num_causal_blocks: int) -> List[int]:
        """The request ids attention metadata must carry: one per causal block, all the
        one rollout sequence."""
        return [_ROLLOUT_REQUEST_ID] * num_causal_blocks

    def page_table(self) -> List[int]:
        """The page table: every resident base page in logical order, from the host
        copy. For ``write_range`` and for tests; kernels read ``causal_block_pages``."""
        self._require_open()
        return (self._host_table // self.page_view_scale).tolist()

    # ------------------------------------------------------------------ the window

    def commit(self, num_tokens: Optional[int] = None) -> None:
        """The ``num_tokens`` tokens written at ``staging_offset`` are final. Those that
        fall within the first ``pin_tokens`` become fixed; the rest become history, and
        the window slides.

        After forwards (planned with ``plan`` or staged with ``write_staged``), the count
        is the most tokens they covered (the default) and any other count raises. Tokens
        written with ``write_range`` alone, such as a prompt, are committed by passing
        their count.
        """
        self._require_open()
        staged = self._staged_tokens
        if num_tokens is None:
            if not staged:
                raise ValueError("nothing was staged since the last commit; pass num_tokens")
            num_tokens = staged
        elif staged and num_tokens != staged:
            raise ValueError(
                f"commit of {num_tokens} tokens, but the forwards since the last commit "
                f"staged {staged}"
            )
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
        self._staged_tokens = 0
        drop_pages = 0
        excess = self._history_tokens - self.window_tokens
        if excess > 0:
            drop_pages = excess // self.tokens_per_page
            if drop_pages:
                # Recycle: the oldest pages become the newest free pages. Rotated in
                # place, so the table's address, which captured graphs hold, never changes.
                ring = self._table[self._fixed_pages : self.num_pages]
                torch.ops.trtllm.rotate_rows_(ring, -drop_pages)  # dropped pages go to the tail
                host_ring = self._host_table[self._fixed_pages :]
                host_ring[:] = np.roll(host_ring, -drop_pages)
                self._history_tokens -= drop_pages * self.tokens_per_page
        self._refresh_device_state(drop_pages)

    # ------------------------------------------------------------------ causal blocks

    def _allocate_state(
        self, device: torch.device, spec: RolloutSpec, num_private: int
    ) -> _SpecState:
        """Allocate a spec's layouts as views into two packed buffers (int32 and int64),
        plus the commit copy lists and the staged slot ids; once per spec."""
        tpb = self.tokens_per_page
        entries = []  # (size, n, num_private_pages, i32 fields, i64 fields)
        for size, n in spec.causal_blocks.items():
            num_private_pages = self._private_pages_per_causal_block(tpb, size)
            i32 = {
                # A page list holds a causal block's full pages then its private pages,
                # never more than the pool has; sized for the pool so every spec's lists
                # agree.
                "causal_block_pages": (n, self._pool_pages),
                "causal_block_offsets": (1, n, 2, self._pool_pages),
                "seq_len_q": (n,),
                "seq_len_kv": (n,),
            }
            i64 = {
                "own_slots": (n * size,),
                "current_private_src": (n * self._current_private,),
                "current_private_dst": (n * self._current_private,),
            }
            entries.append((size, n, num_private_pages, i32, i64))
        # Copies made at commit: the fixed tail's refill (one page of entries), then each
        # causal block's committed_private (at most three partial pages); contiguous, so one
        # launch copies all.
        commit_copy_capacity = tpb + sum(n * 3 * (tpb - 1) for _, n, _, _, _ in entries)
        numel_i32 = sum(int(np.prod(shape)) for *_, i32, _ in entries for shape in i32.values())
        numel_i64 = sum(int(np.prod(shape)) for *_, i64 in entries for shape in i64.values())
        numel_i64 += 2 * commit_copy_capacity
        packed_i32 = torch.zeros(numel_i32, dtype=torch.int32, device=device)
        packed_i64 = torch.zeros(numel_i64, dtype=torch.int64, device=device)
        offsets = {"i32": 0, "i64": 0}

        def take(kind: str, shape) -> torch.Tensor:
            dev = packed_i32 if kind == "i32" else packed_i64
            numel, first = int(np.prod(shape)), offsets[kind]
            offsets[kind] += numel
            return dev[first : first + numel].view(shape)

        commit_copy_src = take("i64", (commit_copy_capacity,))
        commit_copy_dst = take("i64", (commit_copy_capacity,))
        layouts: Dict[int, _CausalBlockLayout] = {}
        first_private, first_committed = 0, tpb
        for size, n, num_private_pages, i32, i64 in entries:
            fields = {name: take("i32", shape) for name, shape in i32.items()}
            fields.update({name: take("i64", shape) for name, shape in i64.items()})
            fields["seq_len_q"].fill_(size)
            host_private_pages = self._host_private[
                first_private : first_private + n * num_private_pages
            ]
            first_private += n * num_private_pages
            num_committed = n * 3 * (tpb - 1)
            layouts[size] = _CausalBlockLayout(
                num_causal_blocks=n,
                causal_block_size=size,
                num_private_pages=num_private_pages,
                private_pages=torch.zeros(n, num_private_pages, dtype=torch.int32, device=device),
                host_private_pages=host_private_pages.reshape(n, num_private_pages),
                committed_private_src=commit_copy_src[
                    first_committed : first_committed + num_committed
                ],
                committed_private_dst=commit_copy_dst[
                    first_committed : first_committed + num_committed
                ],
                **fields,
            )
            first_committed += num_committed
        assert first_private == num_private and first_committed == commit_copy_capacity
        return _SpecState(
            layouts=layouts,
            packed_i32=packed_i32,
            packed_i64=packed_i64,
            commit_copy_src=commit_copy_src,
            commit_copy_dst=commit_copy_dst,
            refill_src=commit_copy_src[:tpb],
            refill_dst=commit_copy_dst[:tpb],
            staged_slots=torch.empty(spec.max_staged_tokens, dtype=torch.int64, device=device),
        )

    def _bind_state(self, state: _SpecState) -> None:
        """Make ``state`` the open spec's: its tensors become the attributes the
        rest of the cache reads."""
        self._layouts = state.layouts
        self._commit_copy_src, self._commit_copy_dst = state.commit_copy_src, state.commit_copy_dst
        self._refill_src, self._refill_dst = state.refill_src, state.refill_dst
        self._staged_slots = state.staged_slots

    def _layout(self, causal_block_size: int) -> _CausalBlockLayout:
        self._require_open()
        layout = self._layouts.get(causal_block_size)
        if layout is None:
            raise ValueError(
                f"causal block size {causal_block_size} was not declared; the open spec "
                f"declares {self.causal_block_sizes}"
            )
        return layout

    @property
    def max_causal_blocks(self) -> int:
        """Most causal blocks one forward of the open spec uses, over every size."""
        return max(self.spec.causal_blocks.values())

    @property
    def max_keys_per_causal_block(self) -> int:
        """Upper bound on the keys any causal block of the open spec reads, whatever
        ``pin_tokens`` an open takes: the fixed region, the window and itself."""
        spec = self.spec
        return spec.max_pin_tokens + spec.window_tokens + max(spec.causal_block_sizes)

    def plan(self, num_tokens: int, causal_block_size: Optional[int] = None) -> ForwardPlan:
        """A forward of ``num_tokens`` real new tokens cut into causal blocks of
        ``causal_block_size`` (default: one causal block of all of them), for the attention
        layers to take as ``kv=``. The size must be declared in the open spec, divide
        ``num_tokens``, and cut it into at most the declared count of causal blocks."""
        self._require_open()
        size, num_causal_blocks = self._causal_blocks(num_tokens, causal_block_size)
        layout = self._layout(size)
        if num_tokens > self.max_staged_tokens or num_causal_blocks > layout.num_causal_blocks:
            raise ValueError(
                f"{num_tokens} tokens in causal blocks of {size} exceed the open spec: at most "
                f"{layout.num_causal_blocks} causal blocks and {self.max_staged_tokens} tokens"
            )
        self._staged_tokens = max(self._staged_tokens, num_tokens)
        return ForwardPlan(self, self.spec, num_tokens, size, num_causal_blocks)

    def causal_block_lengths(self, causal_block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-causal-block int32 device tensors ``(seq_len_q, seq_len_kv)`` for as many
        causal blocks of that size as the open spec declares; a forward of fewer causal
        blocks uses the leading ones.

        ``seq_len_kv[i]`` is the number of keys in causal block ``i``'s page list: the
        fixed region, the window before it, the earlier causal blocks and itself.
        Persistent, refreshed by ``commit``, so a captured forward keeps
        reading the right lengths.
        """
        layout = self._layout(causal_block_size)
        return layout.seq_len_q, layout.seq_len_kv

    def cached_tokens(self, causal_block_size: int) -> List[int]:
        """Keys in each causal block's page list before its own tokens (host ints, from
        the host model of the layout)."""
        layout = self._layout(causal_block_size)
        return [self._host_causal_block(layout, i)[0] for i in range(layout.num_causal_blocks)]

    def causal_block_pages(self, causal_block_size: int) -> torch.Tensor:
        """``[num_causal_blocks, max_pages]`` int32 device tensor of layer-0 view indices.

        Row ``i`` is causal block ``i``'s page list, in key order; entries past its
        length are 0 and never read. Persistent, refreshed in place by ``commit``.
        """
        return self._layout(causal_block_size).causal_block_pages

    @staticmethod
    def _causal_blocks(num_tokens: int, causal_block_size: Optional[int]) -> Tuple[int, int]:
        """The causal block size and count a forward of ``num_tokens`` real tokens uses:
        one causal block of all of them when ``causal_block_size`` is None; otherwise
        the size must be positive and divide ``num_tokens``."""
        size = num_tokens if causal_block_size is None else int(causal_block_size)
        if size <= 0 or num_tokens % size:
            raise ValueError(
                f"{num_tokens} tokens do not split into causal blocks of {causal_block_size}"
            )
        return size, num_tokens // size

    def causal_block_offsets(self, causal_block_size: int) -> torch.Tensor:
        """``[1, num_causal_blocks, 2, max_pages]`` int32 device tensor:
        ``causal_block_pages`` in the TRTLLM attention op's offset encoding (K plane,
        then V plane), the layout its ``kv_cache_block_offsets`` argument has.
        Persistent, refreshed in place by ``commit``, so a kernel bound to it reads the
        current page lists without any metadata refresh."""
        return self._layout(causal_block_size).causal_block_offsets

    def set_causal_block_size(self, causal_block_size: int) -> None:
        """Declare the causal block size the next ``copy_batch_block_offsets`` describes."""
        self._layout(causal_block_size)
        self._offsets_causal_block_size = causal_block_size

    # ------------------------------------------------------------------ device state

    def _refresh_device_state(self, drop_pages: int = 0) -> None:
        """Rewrite the causal block page lists and private-page contents for the next
        forward.

        For each declared causal block size, one kernel is launched. The first launch
        also computes the shared resident staging slots and pinned-tail copy indices.

        Then launch one kernel to copy committed partial-page tokens into private
        pages and, if needed, restore the pinned tail in resident storage.

        ``drop_pages`` is the number of pages ``commit()`` rotated to the ring's end.
        If the pinned prefix ends mid-page, its tail remains in a moved page;
        copy it from there into the page now holding that logical position.

        All output tensors are updated in place, preserving CUDA graph references.
        """
        shared = True
        for layout in self._layouts.values():
            torch.ops.trtllm.block_causal_layout_(
                self._table,
                layout.private_pages,
                layout.causal_block_pages,
                layout.causal_block_offsets,
                layout.seq_len_kv,
                layout.own_slots,
                layout.current_private_src,
                layout.current_private_dst,
                layout.committed_private_src,
                layout.committed_private_dst,
                self._staged_slots if shared else None,
                self._refill_src if shared else None,
                self._refill_dst if shared else None,
                layout.causal_block_size,
                self.tokens_per_page,
                self.staging_offset,
                self._fixed_tokens,
                self.window_tokens,
                self.num_pages,
                drop_pages,
                self.kv_factor,
                self._kv_offset,
                self._rows_per_page,
            )
            shared = False
        torch.ops.trtllm.copy_kv_slots_(
            self._buffers[0],
            self._commit_copy_src,
            self._commit_copy_dst,
            self._num_layers,
            self._kv_heads_local,
            self.tokens_per_page,
        )

    # The host model of the layouts: the arithmetic the layout kernel performs, for
    # host-addressed edits of committed tokens and for ``cached_tokens``.

    def _host_causal_block(self, layout: _CausalBlockLayout, i: int) -> Tuple[int, np.ndarray]:
        """Causal block ``i``: ``(keys before its own tokens, visible positions on its
        partial pages, ascending)``; the ``k``-th such position occupies private slot
        ``k``."""
        spans = self._visible_spans(self.staging_offset + i * layout.causal_block_size)
        runs = self._whole_page_runs(spans)
        partial = self._partial_page_positions(spans, runs)
        return sum(b - a for a, b in runs) * self.tokens_per_page + partial.size, partial

    def _visible_spans(self, start: int) -> List[Tuple[int, int]]:
        """Logical positions a causal block starting at ``start`` sees before its own tokens:
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
        where the causal block starts."""
        tpb = self.tokens_per_page
        edges = sorted({p for lo, hi in spans for p in (lo // tpb, (hi - 1) // tpb)})
        pages = [p for p in edges if not any(a <= p < b for a, b in runs)]
        pos = (np.asarray(pages, dtype=np.int64)[:, None] * tpb + np.arange(tpb)).ravel()
        visible = np.zeros(pos.shape, dtype=bool)
        for lo, hi in spans:
            visible |= (pos >= lo) & (pos < hi)
        return pos[visible]

    def _private_slot(
        self, private_pages: np.ndarray, k: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """View page and slot of a causal block's ``k``-th private slots."""
        return private_pages[k // self.tokens_per_page], k % self.tokens_per_page

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
        that causal blocks read through a private copy are updated in those copies too.
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
        home, copy = self._committed_private_rows(start, k.shape[0])
        if home.size:
            src, dst = self._layer_pool_rows(home, copy, layer_idx, (0, 1))
            pool_rows = self.kv_buffer(0).view(-1, buf.shape[-1])
            pool_rows.index_copy_(0, dst, pool_rows.index_select(0, src))

    def zero_values(self, layer_idx: int, start: int, num_tokens: int) -> None:
        """Zero the V vectors (all heads) of logical ``[start, start + num_tokens)`` in
        one layer, leaving K as written, in the home slots and in every private copy
        causal blocks read. Eager, host-addressed, like ``write_range``."""
        buf = self.kv_buffer(layer_idx)
        vs = self.page_view_scale

        def slots(page, slot, t0, t1):
            buf[page * vs, 1, :, slot : slot + t1 - t0].zero_()

        def whole(page, count, t0, t1):
            buf[page * vs : (page + count) * vs : vs, 1].zero_()

        self._for_each_page_run(start, num_tokens, slots, whole)
        home, copy = self._committed_private_rows(start, num_tokens)
        if home.size:
            _, dst = self._layer_pool_rows(home, copy, layer_idx, (1,))
            self.kv_buffer(0).view(-1, buf.shape[-1]).index_fill_(0, dst, 0)

    def _committed_private_rows(self, start: int, num_tokens: int) -> Tuple[np.ndarray, np.ndarray]:
        """Layer-0 pool rows ``(home, copy)`` of the committed_private copies of logical
        ``[start, start + num_tokens)``: what an edit of those tokens must reach besides
        their home slots. The last commit made the same copies on the device; the refill
        is not among them, its source page is gone from the table."""
        tpb, rpp = self.tokens_per_page, self._rows_per_page
        stop = min(start + num_tokens, self.staging_offset)
        homes, copies = [], []
        for layout in self._layouts.values():
            for i in range(layout.num_causal_blocks):
                _, partial = self._host_causal_block(layout, i)
                k = np.flatnonzero((partial >= start) & (partial < stop))
                if k.size == 0:
                    continue
                pos = partial[k]
                page, slot = self._private_slot(layout.host_private_pages[i], k)
                homes.append(self._host_table[pos // tpb] * rpp + pos % tpb)
                copies.append(page * rpp + slot)
        if not homes:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        return np.concatenate(homes), np.concatenate(copies)

    def _layer_pool_rows(
        self, home: np.ndarray, copy: np.ndarray, layer_idx: int, planes: Sequence[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Device index tensors ``(src, dst)`` of the pool rows the layer-0 rows ``home``
        and ``copy`` occupy in ``layer_idx`` for the given planes (0 = K, 1 = V), every head."""
        tpb, rpp, heads = self.tokens_per_page, self._rows_per_page, self._kv_heads_local
        offsets = np.array(
            [layer_idx * rpp + (plane * heads + h) * tpb for plane in planes for h in range(heads)],
            dtype=np.int64,
        )
        src = (home[:, None] + offsets).reshape(-1)
        dst = (copy[:, None] + offsets).reshape(-1)
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
        table = self.page_table()
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
        of ``causal_block_size`` (default: one causal block of ``T``).

        ``k``/``v`` are ``[T, num_kv_heads, head_dim]`` with a contiguous ``head_dim``;
        token and head strides are free, so slices of a fused QKV projection go in
        without a copy. They go to the logical positions from ``staging_offset`` (where
        later causal blocks and, after commit, later forwards read them) and to the causal
        blocks' private pages: current_private (earlier causal blocks' tokens on each
        causal block's partial pages) and, with ``own_tokens``, each causal block's own
        tokens. One scatter kernel reads K/V once for the logical and own-token slots, a
        second (several causal blocks only) handles the partial pages; both are driven by slot ids rebuilt on
        ``commit``, so the write replays correctly inside a CUDA graph.
        A kernel that writes the new tokens into the private pages itself (trtllm-gen)
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
        size, num_causal_blocks = self._causal_blocks(num_tokens, causal_block_size)
        layout = self._layout(size)
        if num_causal_blocks > layout.num_causal_blocks:
            raise ValueError(
                f"{num_tokens} tokens are more than the {layout.num_causal_blocks} causal blocks of "
                f"{size} the open spec declares"
            )
        self._staged_tokens = max(self._staged_tokens, num_tokens)
        torch.ops.trtllm.scatter_kv_slots_(
            buf,
            k,
            v,
            self._staged_slots[:num_tokens],
            layout.own_slots[:num_tokens] if own_tokens else None,
        )
        if num_causal_blocks > 1:
            num_current = num_causal_blocks * self._current_private
            torch.ops.trtllm.scatter_kv_slots_(
                buf,
                k,
                v,
                layout.current_private_dst[:num_current],
                src=layout.current_private_src[:num_current],
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
        """Hand the attention metadata *our* page lists instead of the manager's.

        ``dst_tensor`` is ``[num_pools, max_num_sequences, 2, max_blocks_per_seq]``
        of int32: K offsets then V offsets per sequence, each ``page * index_scale``
        (+ ``kv_offset`` for V) -- the same encoding the manager's device copy uses.
        Every request is a causal block of the size declared with
        ``set_causal_block_size``; sequence ``i`` gets causal block ``i``'s page list.
        """
        ids = list(request_ids)
        if (
            not 0 < len(ids) <= self.max_causal_blocks
            or any(r != _ROLLOUT_REQUEST_ID for r in ids)
            or num_seqs != len(ids)
            or beam_width != 1
        ):
            raise ValueError(
                f"BlockCausalKVCacheManager serves one sequence (id {_ROLLOUT_REQUEST_ID}) in up to "
                f"{self.max_causal_blocks} causal blocks; got request_ids={ids}, "
                f"num_seqs={num_seqs}, beam_width={beam_width}"
            )
        size = self._offsets_causal_block_size
        if size is None or num_seqs > self._layouts[size].num_causal_blocks:
            raise ValueError(
                f"page offsets for {num_seqs} sequences need set_causal_block_size "
                f"first with a size that cuts the staged tokens into at least that many causal blocks; "
                f"declared: {size}"
            )
        offsets = self._layouts[size].causal_block_offsets[0, :num_seqs]
        n = offsets.shape[-1]
        dst_tensor[0, :num_seqs, :, :n] = offsets
        if n < dst_tensor.shape[-1]:
            dst_tensor[0, :num_seqs, :, n:].zero_()

    def get_batch_cache_indices(
        self,
        request_ids: List[int],
        layer_idx: Optional[int] = None,
        num_blocks_per_seq: Optional[Sequence[int]] = None,
        *,
        raw_indices: bool = False,
    ) -> List[List[int]]:
        """The rotated page table in V2's per-layer view units, one list per request;
        with ``raw_indices``, as base pages (``page_table()``).

        Overrides the inherited accessor, which would report the pages in
        allocation order and disagree with what the attention backends read.
        Every request is the one sequence.
        """
        ids = list(request_ids)
        if any(r != _ROLLOUT_REQUEST_ID for r in ids):
            raise ValueError(
                f"BlockCausalKVCacheManager serves request {_ROLLOUT_REQUEST_ID} only; got {ids}"
            )
        row = self.page_table() if raw_indices else self._host_table.tolist()
        widths = [len(row)] * len(ids) if num_blocks_per_seq is None else list(num_blocks_per_seq)
        return [row[:n] for n in widths]

    def get_batch_cache_indices_flat(
        self, request_ids: List[int], num_blocks: List[int], layer_idx: Optional[int] = None
    ) -> torch.Tensor:
        rows = self.get_batch_cache_indices(request_ids, layer_idx, num_blocks)
        return torch.tensor([p for row in rows for p in row], dtype=torch.int32)

    @property
    def spec(self) -> Optional[RolloutSpec]:
        """The open rollout's spec; ``None`` before the first ``open``."""
        return self._spec

    @property
    def is_open(self) -> bool:
        """Whether a rollout is open; device state of a closed cache is not live."""
        return self._kv_cache is not None

    def _require_open(self) -> None:
        if self._kv_cache is None:
            raise RuntimeError("cache is not open; call open() first")
