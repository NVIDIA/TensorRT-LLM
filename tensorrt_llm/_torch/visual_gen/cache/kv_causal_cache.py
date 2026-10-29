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

held in a pool of pages. A token's logical position is its absolute position in the rollout, and
logical page ``p`` holds positions ``[p * tokens_per_page, (p + 1) * tokens_per_page)`` for the
whole rollout, so the page table is indexed by absolute page and never rotates or moves a token.

A logical page stays resident while it holds a token of the fixed region, of the window before the
next forward, or of the staging room; every other page goes back to the pool, whole. Up to two
resident pages keep some unused tokens: the one where the fixed region ends and the one where the
window begins. Their entries in the page table are valid; which of their tokens a block may read is
the attention kernel's responsibility, not the cache's. A freed page's entry is 0 and is never read.

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
import triton
import triton.language as tl

from tensorrt_llm.math_utils import ceil_div


@triton.jit
def _write_staged_kernel(
    pool_ptr,
    k_ptr,
    v_ptr,
    pages_ptr,
    slots_ptr,
    num_tokens,
    k_token_stride,
    k_head_stride,
    v_token_stride,
    v_head_stride,
    HEADS: tl.constexpr,
    TOKENS_PER_PAGE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Program ``(j, h)`` writes head ``h`` of staged tokens ``[j * BLOCK_T, (j + 1) * BLOCK_T)``
    to their pool pages and slots, K to plane 0 and V to plane 1."""
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    h = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    in_tokens = t < num_tokens
    mask = in_tokens[:, None] & (d < HEAD_DIM)[None, :]
    page = tl.load(pages_ptr + t, mask=in_tokens, other=0)
    slot = tl.load(slots_ptr + t, mask=in_tokens, other=0)
    # Pool element offset of (page, plane 0, h, slot, 0); plane 1 is HEADS pages of heads later.
    dst = (((page * 2 * HEADS + h) * TOKENS_PER_PAGE + slot) * HEAD_DIM)[:, None] + d[None, :]
    plane = HEADS * TOKENS_PER_PAGE * HEAD_DIM
    t64 = t.to(tl.int64)[:, None]
    keys = tl.load(k_ptr + t64 * k_token_stride + h * k_head_stride + d[None, :], mask=mask)
    tl.store(pool_ptr + dst, keys, mask=mask)
    values = tl.load(v_ptr + t64 * v_token_stride + h * v_head_stride + d[None, :], mask=mask)
    tl.store(pool_ptr + dst + plane, values, mask=mask)


class CausalKVCacheManager:
    """A rollout's K/V in a pool of pages, with a windowed life cycle.

    The constructor allocates the pool; ``open`` starts a rollout in it and ``close`` ends it,
    after which the cache may be opened again.

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
        extra_pages: pool pages past the geometry's, numbered from ``num_pool_pages``, that the
            cache never hands out: scratch for a backend that must keep its own pages in the same
            pool (see ``CuDNNCausalKVPages``).
        device: where the pool lives; the current CUDA device by default.

    Device tensors, allocated once and rewritten in place by ``open`` and ``commit``
    (so a CUDA graph captured around a forward keeps reading current values):

    * ``page_table``: ``[max_pages]`` int32, logical page to the pool page ``kv_buffer`` is indexed
        by. Entries of freed pages and of pages past the staged tokens are 0.
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
        """Device bytes the pool of this geometry takes: K and V of every token of every page
        in every layer."""
        pages = CausalKVCacheManager.pool_pages_for(
            tokens_per_page=tokens_per_page,
            pin_tokens=pin_tokens,
            window_tokens=window_tokens,
            max_staged_tokens=max_staged_tokens,
        )
        element_size = torch.tensor([], dtype=dtype).element_size()
        return pages * tokens_per_page * num_layers * 2 * num_kv_heads * head_dim * element_size

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
        extra_pages: int = 0,
        device: Optional[torch.device] = None,
    ) -> None:
        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"CausalKVCacheManager supports bf16/fp16 K/V, got {dtype}")
        _check_geometry(tokens_per_page, pin_tokens, window_tokens, max_staged_tokens)
        if max_seq_len < pin_tokens + max_staged_tokens:
            raise ValueError(
                f"max_seq_len {max_seq_len} is below pin_tokens + max_staged_tokens "
                f"({pin_tokens} + {max_staged_tokens})"
            )
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.tokens_per_page = tokens_per_page
        self.pin_tokens = pin_tokens
        self.window_tokens = window_tokens
        self.max_staged_tokens = max_staged_tokens
        self.max_seq_len = max_seq_len
        if extra_pages < 0:
            raise ValueError(f"extra_pages must be non-negative, got {extra_pages}")
        self.extra_pages = extra_pages
        self.num_pool_pages = self.pool_pages_for(
            tokens_per_page=tokens_per_page,
            pin_tokens=pin_tokens,
            window_tokens=window_tokens,
            max_staged_tokens=max_staged_tokens,
        )
        device = torch.device("cuda", torch.cuda.current_device()) if device is None else device
        # Zeros, so that a kernel reading past the end of a range, which it then masks, reads
        # finite values.
        self._pool: Optional[torch.Tensor] = torch.zeros(
            num_layers,
            self.num_pool_pages + extra_pages,
            2,
            num_kv_heads,
            tokens_per_page,
            head_dim,
            dtype=dtype,
            device=device,
        )
        self._is_open = False
        self._committed_tokens = 0
        self._fixed_tokens = 0
        self._max_pages = ceil_div(max_seq_len, tokens_per_page)
        # Logical page to pool page, -1 when not resident, and the pool pages not in use.
        self._host_pages = np.full(self._max_pages, -1, dtype=np.int64)
        self._free_pages: List[int] = []
        self._table = torch.zeros(self._max_pages, dtype=torch.int32, device=device)
        # Key ranges per (num_blocks, block_size), made on first use.
        self._ranges: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        # Pool page and slot of each token the next forward may stage, in staging order.
        staged_capacity = pin_tokens + max_staged_tokens
        self._staged_pages = torch.zeros(staged_capacity, dtype=torch.int64, device=device)
        self._staged_slots = torch.zeros(staged_capacity, dtype=torch.int64, device=device)

    # ------------------------------------------------------------------ lifecycle

    def open(self) -> None:
        """Start a rollout: the sequence is empty and ``pin_tokens + max_staged_tokens``
        tokens may be staged."""
        if self._pool is None:
            raise RuntimeError("cache is shut down")
        if self._is_open:
            raise RuntimeError("cache already open; call close() first")
        self._host_pages.fill(-1)
        self._free_pages = list(range(self.num_pool_pages - 1, -1, -1))
        self._committed_tokens = 0
        self._fixed_tokens = 0
        self._is_open = True
        self._refresh()

    def close(self) -> None:
        self._is_open = False

    def shutdown(self) -> None:
        """Release the pool for good."""
        self.close()
        self._pool = None

    @property
    def is_open(self) -> bool:
        return self._is_open

    def _require_open(self) -> None:
        if not self._is_open:
            raise RuntimeError("cache is not open; call open() first")

    # ------------------------------------------------------------------ geometry

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
        return self.max_staged_tokens + self.pin_tokens - self._fixed_tokens

    @property
    def page_table(self) -> torch.Tensor:
        """``[max_pages]`` int32 device table: logical page to pool page."""
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
        start = np.maximum(first - self.window_tokens, 0)
        ranges[0].copy_(torch.from_numpy(start.astype(np.int32)))
        ranges[1].copy_(torch.from_numpy((first + block_size - start).astype(np.int32)))

    def host_page_table(self) -> List[int]:
        """Logical page to pool page, from the host copy, -1 for a page that is freed or
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
        rest and the pages it has left entirely go back to the pool. The count is the
        caller's: a forward may stage fewer tokens than ``max_staged_tokens``."""
        self._require_open()
        if not 0 < num_tokens <= self.staging_capacity:
            raise ValueError(
                f"commit of {num_tokens} tokens; the next forward stages 1..{self.staging_capacity}"
            )
        committed = self._committed_tokens + num_tokens
        fixed = min(self.pin_tokens, committed)
        capacity = committed + self.max_staged_tokens + self.pin_tokens - fixed
        if capacity > self.max_seq_len:
            raise ValueError(
                f"committing {num_tokens} tokens leaves {committed} committed and room to stage "
                f"up to {capacity}, past max_seq_len {self.max_seq_len}"
            )
        self._committed_tokens, self._fixed_tokens = committed, fixed
        self._refresh()

    def _refresh(self) -> None:
        """Keep the pages the fixed region, the window and the staging room touch, back the
        new ones, and rewrite the device state."""
        tpb = self.tokens_per_page
        end_page = ceil_div(self._committed_tokens + self.staging_capacity, tpb)
        sink_pages = min(ceil_div(self.pin_tokens, tpb), end_page)
        first_window_page = max(self._committed_tokens - self.window_tokens, 0) // tpb
        # Free first: the pool holds only the steady state, not the old and new pages at once.
        for page in np.flatnonzero(self._host_pages >= 0):
            if sink_pages <= page < first_window_page or page >= end_page:
                self._free_pages.append(int(self._host_pages[page]))
                self._host_pages[page] = -1
        for page in (*range(sink_pages), *range(first_window_page, end_page)):
            if self._host_pages[page] < 0:
                if not self._free_pages:
                    raise RuntimeError("the pool is smaller than the geometry needs")
                self._host_pages[page] = self._free_pages.pop()
        table = np.where(self._host_pages >= 0, self._host_pages, 0).astype(np.int32)
        self._table.copy_(torch.from_numpy(table))
        count = self.staging_capacity
        pos = self._committed_tokens + np.arange(count, dtype=np.int64)
        self._staged_pages[:count].copy_(torch.from_numpy(table[pos // tpb].astype(np.int64)))
        self._staged_slots[:count].copy_(torch.from_numpy(pos % tpb))
        for key, ranges in self._ranges.items():
            self._fill_ranges(key, ranges)

    # ------------------------------------------------------------------ pool access

    def kv_buffer(self, layer_idx: int) -> torch.Tensor:
        """The layer's pool ``[num_pool_pages + extra_pages, 2, num_kv_heads, tokens_per_page,
        head_dim]`` (plane 0 is K, plane 1 is V). Pool page ``p`` of ``page_table`` is ``buf[p]``."""
        self._require_open()
        return self._pool[layer_idx]

    def write_staged(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Stage ``T`` tokens of K/V, the forward's new tokens, at logical positions
        ``[staging_offset, staging_offset + T)``. ``k``/``v`` are ``[T, num_kv_heads, head_dim]``
        with a contiguous ``head_dim``; token and head strides are free, so slices of a fused QKV
        projection go in without a copy. Rewrites the same positions on every call until
        ``commit``."""
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
        if num_tokens == 0:
            return
        if k.stride(2) != 1:
            k = k.contiguous()
        if v.stride(2) != 1:
            v = v.contiguous()
        block_t = 32
        _write_staged_kernel[(triton.cdiv(num_tokens, block_t), heads)](
            buf,
            k,
            v,
            self._staged_pages,
            self._staged_slots,
            num_tokens,
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            HEADS=heads,
            TOKENS_PER_PAGE=self.tokens_per_page,
            HEAD_DIM=head_dim,
            BLOCK_D=triton.next_power_of_2(head_dim),
            BLOCK_T=block_t,
        )


def _check_geometry(
    tokens_per_page: int, pin_tokens: int, window_tokens: int, max_staged_tokens: int
) -> None:
    if tokens_per_page <= 0 or tokens_per_page & (tokens_per_page - 1):
        raise ValueError(f"tokens_per_page must be a power of two, got {tokens_per_page}")
    if window_tokens < 0 or pin_tokens < 0 or max_staged_tokens <= 0:
        raise ValueError(
            "window_tokens and pin_tokens must be non-negative and max_staged_tokens positive"
        )
