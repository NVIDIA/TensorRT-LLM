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

One rollout is one logical sequence of tokens

    [ fixed region | history | staged tokens ]

held in a pool of pages. A token's logical position is its position in the rollout, and logical
page ``p`` holds positions ``[p * tokens_per_page, (p + 1) * tokens_per_page)`` for the whole
rollout: no token ever moves. The fixed region's pages are pool pages ``[0, fixed_pages)``; every
later logical page ``p`` is pool page ``fixed_pages + p % ring_pages``, the ring being as many pages
as the window and the staging room can touch. A logical page is resident while it holds a token of
the fixed region, of the window before the next forward, or of the room the next forward stages
into; its pool page is reused, whole, by the logical page ``ring_pages`` later.

The staged tokens are cut into causal blocks. Causal block ``i`` attends to the fixed region, to
the ``window_tokens`` before its first token, and to the staged tokens up to its end. A paged
kernel reads whole pages in page-list order and cuts only at a sequence's length, while these
spans start and end mid-page. So each causal block gets its own page list:

    [ whole pages of the fixed region | whole pages of its window | private pages ]

The private pages hold copies of the tokens the causal block sees on its partial pages (where the
fixed region ends, where its window starts and where it ends; at most ``3 * (tokens_per_page - 1)``
tokens), packed from the first slot, so the list's only partial page is its last. Every query of a
causal block sees all its keys, so their order in the list does not matter.

``commit`` works out every causal block's page list, length and private-slot sources on the host,
from the host page map, and uploads them in one copy into device tensors that stay at the same
address. Each forward then writes its K/V with one kernel per layer, which also fills the private
pages: from the new K/V for staged tokens, from the pool for committed ones. A CUDA graph captured
around a forward therefore replays correctly after a commit.

Rotary positions are the caller's business and are absolute over the rollout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.math_utils import ceil_div

from .rollout_kv_cache_interface import ForwardPlan, RolloutKVCacheInterface, RolloutSpec


@triton.jit
def _pool_offset(
    slot_id, h, HEADS: tl.constexpr, TOKENS_PER_PAGE: tl.constexpr, HEAD_DIM: tl.constexpr
):
    """Element offset of K head ``h`` of pool slot ``slot_id`` in a layer's pool
    ``[pages, 2, HEADS, TOKENS_PER_PAGE, HEAD_DIM]``."""
    slot_id = slot_id.to(tl.int64)
    page = slot_id // TOKENS_PER_PAGE
    slot = slot_id % TOKENS_PER_PAGE
    return ((page * 2 * HEADS + h) * TOKENS_PER_PAGE + slot) * HEAD_DIM


@triton.jit
def _stage_kernel(
    pool_ptr,
    k_ptr,
    v_ptr,
    home_dst_ptr,
    private_src_ptr,
    private_dst_ptr,
    num_home,
    num_entries,
    k_token_stride,
    k_head_stride,
    v_token_stride,
    v_head_stride,
    HEADS: tl.constexpr,
    TOKENS_PER_PAGE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Program ``(j, h)`` makes copies ``[j * BLOCK_E, (j + 1) * BLOCK_E)`` of head ``h``, K and V.

    Copy ``e < num_home`` writes staged token ``e`` to its home slot ``home_dst[e]``. Copy
    ``num_home + p`` fills private slot ``private_dst[p]`` (skipped when negative) from
    ``private_src[p]``: a staged token when non-negative, otherwise pool slot ``-src - 1``.
    Slots are ``page * TOKENS_PER_PAGE + slot`` in the layer's pool.
    """
    e = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
    h = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    is_home = e < num_home
    is_private = (e >= num_home) & (e < num_entries)
    p = e - num_home
    src = tl.where(is_home, e, tl.load(private_src_ptr + p, mask=is_private, other=0))
    dst = tl.where(
        is_home,
        tl.load(home_dst_ptr + e, mask=is_home, other=-1),
        tl.load(private_dst_ptr + p, mask=is_private, other=-1),
    )
    valid = (is_home | is_private) & (dst >= 0)
    from_tokens = src >= 0
    d_mask = (d < HEAD_DIM)[None, :]
    token_mask = (valid & from_tokens)[:, None] & d_mask
    pool_mask = (valid & ~from_tokens)[:, None] & d_mask
    plane = HEADS * TOKENS_PER_PAGE * HEAD_DIM
    dst_offset = _pool_offset(tl.maximum(dst, 0), h, HEADS, TOKENS_PER_PAGE, HEAD_DIM)
    dst_offset = dst_offset[:, None] + d[None, :]
    src_offset = _pool_offset(tl.maximum(-src - 1, 0), h, HEADS, TOKENS_PER_PAGE, HEAD_DIM)
    src_offset = src_offset[:, None] + d[None, :]
    token = tl.maximum(src, 0).to(tl.int64)[:, None]
    k_offset = token * k_token_stride + h * k_head_stride + d[None, :]
    v_offset = token * v_token_stride + h * v_head_stride + d[None, :]
    keys = tl.where(
        token_mask,
        tl.load(k_ptr + k_offset, mask=token_mask, other=0.0),
        tl.load(pool_ptr + src_offset, mask=pool_mask, other=0.0),
    )
    tl.store(pool_ptr + dst_offset, keys, mask=token_mask | pool_mask)
    values = tl.where(
        token_mask,
        tl.load(v_ptr + v_offset, mask=token_mask, other=0.0),
        tl.load(pool_ptr + src_offset + plane, mask=pool_mask, other=0.0),
    )
    tl.store(pool_ptr + dst_offset + plane, values, mask=token_mask | pool_mask)


class PagedKVTables(NamedTuple):
    """What a paged attention kernel reads for one forward's causal blocks, as int32 device
    tensors: sequence ``b`` attends to the first ``seq_lens_kv[b]`` tokens of the pool pages
    ``page_table[b]`` lists, in order.

    ``kv_block_offsets`` lists the same pages in the TRTLLM attention op's format, which gives K
    and V separate page ids: viewing a layer's pool ``[num_pages, 2, H, page_size, D]`` as
    ``[2 * num_pages, H, page_size, D]``, page ``p``'s K is at ``2p`` and its V at ``2p + 1``.
    """

    page_table: torch.Tensor  # [num_causal_blocks, max_pages]
    seq_lens_q: torch.Tensor  # [num_causal_blocks], all causal_block_size
    seq_lens_kv: torch.Tensor  # [num_causal_blocks]
    cu_seqlens_kv: torch.Tensor  # [num_causal_blocks + 1]
    cu_seqlens_q: torch.Tensor  # [num_causal_blocks + 1], entry i is i * causal_block_size
    kv_block_offsets: torch.Tensor  # [num_causal_blocks, 2, max_pages]: K, then V page ids


@dataclass
class _Layout:
    """Device views of one causal block size's page lists and private-slot sources."""

    num_causal_blocks: int
    page_table: torch.Tensor  # [n, max_pages]
    seq_lens_q: torch.Tensor  # [n]
    seq_lens_kv: torch.Tensor  # [n]
    cu_seqlens_kv: torch.Tensor  # [n + 1]
    cu_seqlens_q: torch.Tensor  # [n + 1]
    kv_block_offsets: torch.Tensor  # [n, 2, max_pages]
    # A fixed ``private_slots`` entries per causal block, so the first ``m`` causal blocks'
    # entries are a prefix; see ``_stage_kernel``.
    private_src: torch.Tensor  # [n * private_slots]
    private_dst: torch.Tensor  # [n * private_slots]
    first_private_page: int


@dataclass
class _HostMirror:
    """A pinned host copy of a spec's device state, and the upload that last read it."""

    buffer: torch.Tensor  # int32, pinned
    views: Dict[str, np.ndarray]
    uploaded: Optional[torch.cuda.Event] = None


@dataclass
class _SpecState:
    """A spec's device state, allocated the first time it is opened and kept, so a CUDA graph
    captured over the spec stays valid when it is opened again. Every tensor is a view of
    ``device``, which ``commit`` overwrites in one copy from one of two host mirrors, taking
    turns so that a commit never waits for the previous one's upload."""

    device: torch.Tensor  # int32
    mirrors: List[_HostMirror]
    home_dst: torch.Tensor  # [max_staged_tokens]
    layouts: Dict[int, _Layout]
    next_mirror: int = 0


class PagedRolloutKVCache(RolloutKVCacheInterface):
    """A pool of K/V pages holding one rollout at a time, batch size 1.

    The constructor allocates the pool; ``open`` lays a ``RolloutSpec`` out in it, ``close`` ends
    the rollout and ``shutdown`` releases the pool. Several streams (for example the two branches
    of classifier-free guidance) take one instance each.

    A pipeline drives it through ``RolloutKVCacheInterface``, and puts a prompt in with
    ``write_range(layer_idx, 0, k, v)`` on every layer followed by ``commit`` of its length. The
    attention backends stage each forward with ``write_staged`` and read ``kv_buffer`` through
    ``tables``.

    Args:
        num_layers: attention layers that keep K/V.
        num_kv_heads: K/V heads this rank holds: the head count of the attention backend it
            serves. Head sharding (tensor parallel, Ulysses) is settled before K/V reach the
            cache.
        head_dim, dtype: K/V geometry; bf16 or fp16.
        tokens_per_page: page size, a power of two. Each backend may constrain it further
            (``support_kv_cache_page_size``).
        pool_tokens: size of the pool, a multiple of ``tokens_per_page``; ``pool_tokens_for``
            gives what a spec needs.
        device: where the pool lives; the current CUDA device by default.
    """

    @staticmethod
    def private_pages_per_causal_block(tokens_per_page: int) -> int:
        """Private pages of one causal block: up to three partial pages' visible tokens."""
        return ceil_div(3 * (tokens_per_page - 1), tokens_per_page)

    @staticmethod
    def _spec_pages(tokens_per_page: int, spec: RolloutSpec) -> Tuple[int, int]:
        """A spec's ``(resident, private)`` pages. Resident: the fixed region's pages, and the
        pages the window and the staged tokens touch, one more than their length when it is not
        page aligned."""
        tpb = tokens_per_page
        if tpb <= 0 or tpb & (tpb - 1):
            raise ValueError(f"tokens_per_page must be a power of two, got {tpb}")
        resident = (
            ceil_div(spec.max_pin_tokens, tpb)
            + ceil_div(spec.window_tokens + spec.max_staged_tokens, tpb)
            + 1
        )
        private = sum(spec.causal_blocks.values()) * (
            PagedRolloutKVCache.private_pages_per_causal_block(tpb)
        )
        return resident, private

    @staticmethod
    def pool_tokens_for(spec: RolloutSpec, *, tokens_per_page: int) -> int:
        """Pool tokens a spec needs; a pool at least this large can ``open`` it."""
        resident, private = PagedRolloutKVCache._spec_pages(tokens_per_page, spec)
        return (resident + private) * tokens_per_page

    @staticmethod
    def bytes_for_pool_tokens(
        pool_tokens: int, *, num_layers: int, num_kv_heads: int, head_dim: int, dtype: torch.dtype
    ) -> int:
        """Device bytes of a pool of ``pool_tokens``: K and V of every token in every layer."""
        element_size = torch.tensor([], dtype=dtype).element_size()
        return pool_tokens * num_layers * 2 * num_kv_heads * head_dim * element_size

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
        """Device bytes of a pool sized for this spec."""
        return PagedRolloutKVCache.bytes_for_pool_tokens(
            PagedRolloutKVCache.pool_tokens_for(spec, tokens_per_page=tokens_per_page),
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )

    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        tokens_per_page: int,
        pool_tokens: int,
        device: Optional[torch.device] = None,
    ) -> None:
        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"PagedRolloutKVCache supports bf16/fp16 K/V, got {dtype}")
        tpb = tokens_per_page
        if tpb <= 0 or tpb & (tpb - 1):
            raise ValueError(f"tokens_per_page must be a power of two, got {tpb}")
        if pool_tokens <= 0 or pool_tokens % tpb:
            raise ValueError(
                f"pool_tokens must be a positive multiple of tokens_per_page {tpb}, "
                f"got {pool_tokens}"
            )
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.tokens_per_page = tpb
        self.pool_tokens = pool_tokens
        self.pool_pages = pool_tokens // tpb
        self.private_slots = 3 * (tpb - 1)
        device = torch.device("cuda", torch.cuda.current_device()) if device is None else device
        # Zeros: a kernel never reads a slot nobody wrote, but finite values make a mistake
        # visible as a wrong result rather than a NaN.
        self._pool: Optional[torch.Tensor] = torch.zeros(
            num_layers,
            self.pool_pages,
            2,
            num_kv_heads,
            tpb,
            head_dim,
            dtype=dtype,
            device=device,
        )
        self._device = device
        self._spec: Optional[RolloutSpec] = None
        self._is_open = False
        self._states: Dict[RolloutSpec, _SpecState] = {}
        self._state: Optional[_SpecState] = None
        self.window_tokens = 0
        self.max_staged_tokens = 0
        self._pin_tokens = 0
        self._fixed_tokens = 0
        self._committed_tokens = 0
        # Most tokens a plan covered since the last commit; plans are made on the host for
        # every forward, so this holds under CUDA graph replay too.
        self._staged_tokens = 0
        # Pool pages of the fixed region, and of the ring the later logical pages take turns in.
        self._fixed_pages = 0
        self._ring_pages = 0

    # ------------------------------------------------------------------ lifecycle

    @property
    def spec(self) -> Optional[RolloutSpec]:
        return self._spec

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def is_shut_down(self) -> bool:
        """Whether the pool is gone for good; a closed cache may still be reopened."""
        return self._pool is None

    def open(self, spec: RolloutSpec, *, pin_tokens: int = 0) -> "PagedRolloutKVCache":
        """Start a rollout of ``spec``; the sequence starts empty and its first ``pin_tokens``
        committed tokens become the fixed region, seen by every causal block and never evicted.
        Returns the cache, so ``with cache.open(spec):`` closes the rollout on exit."""
        if self._pool is None:
            raise RuntimeError("cache is shut down")
        if self._is_open:
            raise RuntimeError("cache already open; call close() first")
        if not 0 <= pin_tokens <= spec.max_pin_tokens:
            raise ValueError(
                f"pin_tokens {pin_tokens} must be within [0, spec.max_pin_tokens "
                f"{spec.max_pin_tokens}]"
            )
        resident, private = self._spec_pages(self.tokens_per_page, spec)
        if resident + private > self.pool_pages:
            raise ValueError(
                f"{spec} needs {(resident + private) * self.tokens_per_page} pool tokens; the "
                f"pool holds {self.pool_tokens}"
            )
        state = self._states.get(spec)
        if state is None:
            state = self._allocate_state(spec)
            self._states[spec] = state
        self._spec, self._state = spec, state
        self.window_tokens = spec.window_tokens
        self.max_staged_tokens = spec.max_staged_tokens
        self._pin_tokens = pin_tokens
        self._fixed_tokens = 0
        self._committed_tokens = 0
        self._staged_tokens = 0
        self._fixed_pages = ceil_div(pin_tokens, self.tokens_per_page)
        self._ring_pages = (
            ceil_div(spec.window_tokens + spec.max_staged_tokens, self.tokens_per_page) + 1
        )
        self._is_open = True
        self._refresh()
        return self

    def __enter__(self) -> "PagedRolloutKVCache":
        self._require_open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        # Device state outlives the rollout, so that forwards captured in a CUDA graph stay
        # valid after the next open() of the same spec.
        self._is_open = False

    def shutdown(self) -> None:
        """Release the pool; the cache cannot be opened afterwards."""
        self.close()
        self._pool = None
        self._states.clear()
        self._state = None

    def _require_open(self) -> None:
        if not self._is_open:
            raise RuntimeError("cache is not open; call open() first")

    # ------------------------------------------------------------------ the open rollout

    @property
    def pin_tokens(self) -> int:
        """Tokens the open rollout pins: the first this many committed become fixed."""
        return self._pin_tokens

    @property
    def fixed_tokens(self) -> int:
        """Tokens every causal block sees for the whole rollout; logical ``[0, fixed_tokens)``."""
        return self._fixed_tokens

    @property
    def staging_offset(self) -> int:
        """Logical position of the next forward's first token: the tokens committed so far."""
        return self._committed_tokens

    @property
    def history_tokens(self) -> int:
        """Committed tokens resident past the fixed region: the window, and the stale tokens
        before it on its first page."""
        first_window_page = max(self._committed_tokens - self.window_tokens, 0)
        first_window_page //= self.tokens_per_page
        return self._committed_tokens - max(
            self._fixed_tokens, first_window_page * self.tokens_per_page
        )

    @property
    def resident_tokens(self) -> int:
        """Most tokens the open spec keeps resident, private pages aside."""
        return self._spec_pages(self.tokens_per_page, self._spec)[0] * self.tokens_per_page

    @property
    def staging_capacity(self) -> int:
        """Tokens resident from ``staging_offset`` on: what the next commit may cover. The
        rest of the fixed region may be committed on top of ``max_staged_tokens``."""
        return self.max_staged_tokens + self._pin_tokens - self._fixed_tokens

    def plan(self, num_tokens: int, causal_block_size: Optional[int] = None) -> ForwardPlan:
        """A forward of ``num_tokens`` real new tokens cut into causal blocks of
        ``causal_block_size`` (default: one causal block of all of them). The size must be
        declared in the open spec, divide ``num_tokens``, and cut it into at most the declared
        count of causal blocks."""
        self._require_open()
        size = num_tokens if causal_block_size is None else int(causal_block_size)
        if size <= 0 or num_tokens % size:
            raise ValueError(
                f"{num_tokens} tokens do not split into causal blocks of {causal_block_size}"
            )
        layout = self._state.layouts.get(size)
        if layout is None:
            raise ValueError(
                f"causal block size {size} was not declared; the open spec declares "
                f"{self._spec.causal_block_sizes}"
            )
        num_causal_blocks = num_tokens // size
        if num_tokens > self.max_staged_tokens or num_causal_blocks > layout.num_causal_blocks:
            raise ValueError(
                f"{num_tokens} tokens in causal blocks of {size} exceed the open spec: at most "
                f"{layout.num_causal_blocks} causal blocks and {self.max_staged_tokens} tokens"
            )
        self._staged_tokens = max(self._staged_tokens, num_tokens)
        return ForwardPlan(self, self._spec, num_tokens, size, num_causal_blocks)

    def commit(self, num_tokens: Optional[int] = None) -> None:
        """The ``num_tokens`` tokens written at ``staging_offset`` are final. Those within the
        first ``pin_tokens`` become fixed, the window moves past the rest, and the pages it left
        go back to the pool.

        After planned forwards the count is the most tokens they covered (the default) and any
        other count raises. Tokens written with ``write_range`` alone, such as a prompt, are
        committed by passing their count.
        """
        self._require_open()
        staged = self._staged_tokens
        if num_tokens is None:
            if not staged:
                raise ValueError("nothing was planned since the last commit; pass num_tokens")
            num_tokens = staged
        elif staged and num_tokens != staged:
            raise ValueError(
                f"commit of {num_tokens} tokens, but the forwards since the last commit "
                f"staged {staged}"
            )
        if not 0 < num_tokens <= self.staging_capacity:
            pinned = self._pin_tokens - self._fixed_tokens
            raise ValueError(
                f"commit of {num_tokens} tokens; at most max_staged_tokens "
                f"{self.max_staged_tokens} plus the {pinned} still to pin"
            )
        self._committed_tokens += num_tokens
        self._fixed_tokens = min(self._pin_tokens, self._committed_tokens)
        self._staged_tokens = 0
        self._refresh()

    # ------------------------------------------------------------------ device state

    def _allocate_state(self, spec: RolloutSpec) -> _SpecState:
        """Lay a spec's device tensors out in one int32 buffer, with a pinned host mirror."""
        tpb = self.tokens_per_page
        max_pages = (
            ceil_div(spec.max_pin_tokens, tpb)
            + ceil_div(spec.window_tokens + max(spec.causal_block_sizes), tpb)
            + self.private_pages_per_causal_block(tpb)
        )
        shapes: Dict[str, Tuple[int, ...]] = {"home_dst": (spec.max_staged_tokens,)}
        for size, n in spec.causal_blocks.items():
            shapes[f"{size}.page_table"] = (n, max_pages)
            shapes[f"{size}.seq_lens_q"] = (n,)
            shapes[f"{size}.seq_lens_kv"] = (n,)
            shapes[f"{size}.cu_seqlens_kv"] = (n + 1,)
            shapes[f"{size}.cu_seqlens_q"] = (n + 1,)
            shapes[f"{size}.kv_block_offsets"] = (n, 2, max_pages)
            shapes[f"{size}.private_src"] = (n * self.private_slots,)
            shapes[f"{size}.private_dst"] = (n * self.private_slots,)
        # Every view starts on a 128-byte boundary: kernels take them as aligned tensors.
        align = 32
        numel = sum(ceil_div(int(np.prod(s)), align) * align for s in shapes.values())
        device = torch.zeros(numel, dtype=torch.int32, device=self._device)
        mirrors = [
            _HostMirror(torch.zeros(numel, dtype=torch.int32, pin_memory=prefer_pinned()), {})
            for _ in range(2)
        ]
        views: Dict[str, torch.Tensor] = {}
        first = 0
        for name, shape in shapes.items():
            count = int(np.prod(shape))
            views[name] = device[first : first + count].view(shape)
            for mirror in mirrors:
                mirror.views[name] = mirror.buffer.numpy()[first : first + count].reshape(shape)
            first += ceil_div(count, align) * align
        layouts = {}
        first_private_page = self.pool_pages - self._spec_pages(tpb, spec)[1]
        for size, n in spec.causal_blocks.items():
            for mirror in mirrors:
                mirror.views[f"{size}.seq_lens_q"][:] = size
                mirror.views[f"{size}.cu_seqlens_q"][:] = np.arange(n + 1) * size
            layouts[size] = _Layout(
                num_causal_blocks=n,
                page_table=views[f"{size}.page_table"],
                seq_lens_q=views[f"{size}.seq_lens_q"],
                seq_lens_kv=views[f"{size}.seq_lens_kv"],
                cu_seqlens_kv=views[f"{size}.cu_seqlens_kv"],
                cu_seqlens_q=views[f"{size}.cu_seqlens_q"],
                kv_block_offsets=views[f"{size}.kv_block_offsets"],
                private_src=views[f"{size}.private_src"],
                private_dst=views[f"{size}.private_dst"],
                first_private_page=first_private_page,
            )
            first_private_page += n * self.private_pages_per_causal_block(tpb)
        return _SpecState(
            device=device, mirrors=mirrors, home_dst=views["home_dst"], layouts=layouts
        )

    def _refresh(self) -> None:
        """Rewrite the device state for the next forwards: one upload of what the host works
        out from the new staging offset."""
        state = self._state
        mirror = state.mirrors[state.next_mirror]
        state.next_mirror = 1 - state.next_mirror
        if mirror.uploaded is not None:
            # The upload two commits ago may still read this mirror.
            mirror.uploaded.synchronize()
        views = mirror.views
        staged = self._committed_tokens + np.arange(self.max_staged_tokens)
        views["home_dst"][:] = self._slot_ids(staged)
        for size, layout in state.layouts.items():
            self._write_layout(views, size, layout)
        state.device.copy_(mirror.buffer, non_blocking=True)
        if mirror.uploaded is None:
            mirror.uploaded = torch.cuda.Event()
        mirror.uploaded.record()

    def _write_layout(self, views: Dict[str, np.ndarray], size: int, layout: _Layout) -> None:
        """Every causal block of ``size``'s page list, lengths and private-slot sources."""
        tpb = self.tokens_per_page
        staging_offset = self._committed_tokens
        page_table = views[f"{size}.page_table"]
        seq_lens_kv = views[f"{size}.seq_lens_kv"]
        cu_seqlens_kv = views[f"{size}.cu_seqlens_kv"]
        private_src = views[f"{size}.private_src"].reshape(-1, self.private_slots)
        private_dst = views[f"{size}.private_dst"].reshape(-1, self.private_slots)
        page_table.fill(0)
        private_dst.fill(-1)
        ppb = self.private_pages_per_causal_block(tpb)
        cu_seqlens_kv[0] = 0
        for i in range(layout.num_causal_blocks):
            whole, partial = self.causal_block_keys(staging_offset + i * size, size)
            private_page = layout.first_private_page + i * ppb
            used = len(whole) + ceil_div(partial.size, tpb)
            page_table[i, : len(whole)] = self._pool_pages(whole)
            page_table[i, len(whole) : used] = private_page + np.arange(used - len(whole))
            seq_lens_kv[i] = len(whole) * tpb + partial.size
            cu_seqlens_kv[i + 1] = cu_seqlens_kv[i] + seq_lens_kv[i]
            kv_block_offsets = views[f"{size}.kv_block_offsets"]
            kv_block_offsets[i, 0] = 2 * page_table[i]
            kv_block_offsets[i, 1] = 2 * page_table[i] + 1
            k = np.arange(partial.size)
            private_dst[i, : partial.size] = (private_page + k // tpb) * tpb + k % tpb
            # A committed token is copied from its home slot, a staged one from the new K/V.
            committed = partial < staging_offset
            sources = partial - staging_offset
            if committed.any():
                sources[committed] = -self._slot_ids(partial[committed]) - 1
            private_src[i, : partial.size] = sources

    def causal_block_keys(self, first: int, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """What a causal block of ``size`` tokens from logical ``first`` attends to: the
        logical pages it sees in full, in page-list order, and the positions it sees on its
        partial pages, which its private pages hold in this order."""
        tpb = self.tokens_per_page
        fixed, end = self._fixed_tokens, first + size
        start = max(first - self.window_tokens, 0)
        spans = [(0, end)] if start <= fixed else [(0, fixed), (start, end)]
        whole: List[np.ndarray] = []
        partial: List[np.ndarray] = []
        for lo, hi in spans:
            if hi <= lo:
                continue
            first_whole, end_whole = ceil_div(lo, tpb), hi // tpb
            if first_whole > end_whole:  # inside one page
                partial.append(np.arange(lo, hi))
                continue
            whole.append(np.arange(first_whole, end_whole))
            partial.append(np.arange(lo, first_whole * tpb))
            partial.append(np.arange(end_whole * tpb, hi))
        empty = np.zeros(0, dtype=np.int64)
        whole_pages = np.concatenate(whole) if whole else empty
        positions = np.concatenate(partial) if partial else empty
        return whole_pages.astype(np.int64), positions.astype(np.int64)

    def _slot_ids(self, positions: np.ndarray) -> np.ndarray:
        """Pool slot ids (``page * tokens_per_page + slot``) of resident logical positions."""
        pages, slots = self.locate(positions)
        return pages * self.tokens_per_page + slots

    def _pool_pages(self, logical: np.ndarray) -> np.ndarray:
        """Pool pages of logical pages, resident or not."""
        fixed = self._fixed_pages
        return np.where(logical < fixed, logical, fixed + logical % self._ring_pages)

    def locate(self, positions) -> Tuple[np.ndarray, np.ndarray]:
        """Pool pages and slots of resident logical positions."""
        self._require_open()
        positions = np.asarray(positions, dtype=np.int64)
        tpb = self.tokens_per_page
        logical = positions // tpb
        first_window_page = max(self._committed_tokens - self.window_tokens, 0) // tpb
        end_page = ceil_div(self._committed_tokens + self.staging_capacity, tpb)
        resident = (logical >= 0) & (
            (logical < self._fixed_pages) | ((logical >= first_window_page) & (logical < end_page))
        )
        if not resident.all():
            raise ValueError(f"logical page {int(logical[~resident].flat[0])} is not resident")
        return self._pool_pages(logical), positions % tpb

    # ------------------------------------------------------------------ pool access

    def kv_buffer(self, layer_idx: int) -> torch.Tensor:
        """The layer's pool ``[pool_pages, 2, num_kv_heads, tokens_per_page, head_dim]``; plane
        0 is K, plane 1 is V."""
        self._require_open()
        return self._pool[layer_idx]

    def tables(self, plan: ForwardPlan) -> PagedKVTables:
        """The page lists and lengths of the plan's causal blocks. Persistent device tensors,
        rewritten in place by ``commit``."""
        plan.validate()
        layout = self._state.layouts[plan.causal_block_size]
        n = plan.num_causal_blocks
        return PagedKVTables(
            layout.page_table[:n],
            layout.seq_lens_q[:n],
            layout.seq_lens_kv[:n],
            layout.cu_seqlens_kv[: n + 1],
            layout.cu_seqlens_q[: n + 1],
            layout.kv_block_offsets[:n],
        )

    def write_staged(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, plan: ForwardPlan
    ) -> None:
        """Stage the plan's new tokens in one layer and fill that layer's private pages.

        ``k``/``v`` are ``[plan.num_tokens, num_kv_heads, head_dim]`` with a contiguous
        ``head_dim``; token and head strides are free, so slices of a fused QKV projection go
        in without a copy. They go to logical ``[staging_offset, staging_offset +
        num_tokens)``, rewritten by every forward until ``commit``.
        """
        plan.validate()
        buf = self.kv_buffer(layer_idx)
        if v.shape != k.shape or tuple(k.shape) != (plan.num_tokens, *buf.shape[2:3], buf.shape[4]):
            raise ValueError(
                f"staged k {tuple(k.shape)} / v {tuple(v.shape)}; the plan stages "
                f"[{plan.num_tokens}, {buf.shape[2]}, {buf.shape[4]}]"
            )
        if k.dtype != buf.dtype or v.dtype != buf.dtype:
            raise TypeError(f"K/V dtype {k.dtype} does not match the cache's {buf.dtype}")
        if k.stride(2) != 1:
            k = k.contiguous()
        if v.stride(2) != 1:
            v = v.contiguous()
        layout = self._state.layouts[plan.causal_block_size]
        num_private = plan.num_causal_blocks * self.private_slots
        num_entries = plan.num_tokens + num_private
        block_e = 32
        _stage_kernel[(triton.cdiv(num_entries, block_e), self.num_kv_heads)](
            buf,
            k,
            v,
            self._state.home_dst,
            layout.private_src,
            layout.private_dst,
            plan.num_tokens,
            num_entries,
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            HEADS=self.num_kv_heads,
            TOKENS_PER_PAGE=self.tokens_per_page,
            HEAD_DIM=self.head_dim,
            BLOCK_D=triton.next_power_of_2(self.head_dim),
            BLOCK_E=block_e,
        )

    def write_range(self, layer_idx: int, start: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Write ``k``/``v`` ``[T, num_kv_heads, head_dim]`` at resident logical ``[start, start +
        T)`` of one layer. Eager and host-addressed: for a prompt before its commit, and for
        tests. Private copies of committed tokens are refreshed by every forward, so an edit
        reaches them too."""
        buf = self.kv_buffer(layer_idx)
        if v.shape != k.shape or k.dim() != 3:
            raise ValueError(
                f"k/v must be [T, heads, head_dim]: {tuple(k.shape)}, {tuple(v.shape)}"
            )
        if k.dtype != buf.dtype or v.dtype != buf.dtype:
            raise TypeError(f"K/V dtype {k.dtype} does not match the cache's {buf.dtype}")
        num_tokens = k.shape[0]
        if num_tokens == 0:
            return
        if start < 0:
            raise ValueError(f"start {start} is negative")
        pages, slots = self.locate(np.arange(start, start + num_tokens))
        pages = torch.from_numpy(pages).to(buf.device)
        slots = torch.from_numpy(slots).to(buf.device)
        # Indexing [pages, heads, tpb, D] by (page, :, slot) gives [T, heads, D].
        buf[:, 0][pages, :, slots] = k
        buf[:, 1][pages, :, slots] = v
