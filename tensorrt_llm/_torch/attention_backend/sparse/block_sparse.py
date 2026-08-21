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

"""Common routing utilities for block-sparse attention kernels."""

from dataclasses import dataclass, field
from functools import cache
from math import ceil
from typing import ClassVar, Literal

import torch

from .params import SparseParams

_BITS_PER_WORD = 32
_SIGNED_INT32_MAX = torch.iinfo(torch.int32).max


def _is_current_stream_capturing(device: torch.device) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    with torch.cuda.device(device):
        return torch.cuda.is_current_stream_capturing()


@dataclass(frozen=True, slots=True)
class BlockSparseRoutes:
    """Canonical BSR routes consumed by a block-sparse attention runtime.

    ``block_indptr`` has shape ``[B, Hkv, Qblocks + 1]`` and
    ``block_indices`` is flat. The kernel runtime owns validation of those raw
    tensors and of their values.
    """

    block_indptr: torch.Tensor
    block_indices: torch.Tensor
    max_blocks_per_row: int

    def __post_init__(self) -> None:
        if isinstance(self.max_blocks_per_row, bool) or not isinstance(
            self.max_blocks_per_row, int
        ):
            raise TypeError("max_blocks_per_row must be a Python integer")
        if self.max_blocks_per_row < 0:
            raise ValueError("max_blocks_per_row must be non-negative")


@dataclass(frozen=True, slots=True)
class BlockSparseParams(SparseParams):
    """Static block geometry selecting generic block-sparse attention."""

    q_block_size: int
    kv_block_size: int
    uses_framework_prediction: ClassVar[bool] = False
    allows_fallback_fmha: ClassVar[bool] = False
    algorithm: Literal["block_sparse"] = field(init=False, default="block_sparse")

    def __post_init__(self) -> None:
        for name, value in (
            ("q_block_size", self.q_block_size),
            ("kv_block_size", self.kv_block_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a Python integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class BlockSparseForwardInputs:
    """Live routing inputs for one generic block-sparse forward call."""

    routes: BlockSparseRoutes
    kv_valid_bits: torch.Tensor | None = None


class BlockSparseRouteBuilder:
    """Lower uniform selected-block tables into graph-stable canonical BSR routes.

    The cache follows the lifetime of its owning attention adapter. Dynamic
    block IDs remain live tensors; only shape-derived immutable indptr tensors
    are retained.
    """

    def __init__(self) -> None:
        self._uniform_indptr_cache: dict[tuple[torch.device, int, int, int, int], torch.Tensor] = {}

    def _get_uniform_indptr(
        self,
        *,
        device: torch.device,
        batch_size: int,
        num_kv_heads: int,
        num_q_blocks: int,
        blocks_per_row: int,
    ) -> torch.Tensor:
        key = (device, batch_size, num_kv_heads, num_q_blocks, blocks_per_row)
        block_indptr = self._uniform_indptr_cache.get(key)
        if block_indptr is not None:
            return block_indptr
        if _is_current_stream_capturing(device):
            raise RuntimeError(
                "block-sparse route cache miss during CUDA Graph capture; "
                "run an eager warmup with the same selected-block shape first"
            )

        total_entries = batch_size * num_kv_heads * num_q_blocks * blocks_per_row
        if total_entries > _SIGNED_INT32_MAX:
            raise OverflowError("block-sparse route offsets must fit in signed int32")
        row_offsets = torch.arange(
            num_q_blocks + 1,
            dtype=torch.int32,
            device=device,
        ).reshape(1, 1, -1)
        head_offsets = torch.arange(
            batch_size * num_kv_heads,
            dtype=torch.int32,
            device=device,
        ).reshape(batch_size, num_kv_heads, 1)
        block_indptr = (
            head_offsets * (num_q_blocks * blocks_per_row) + row_offsets * blocks_per_row
        ).contiguous()
        self._uniform_indptr_cache[key] = block_indptr
        return block_indptr

    def from_uniform_selected_blocks(
        self,
        selected_blocks: torch.Tensor,
    ) -> BlockSparseRoutes:
        """Lower ``[B, Hkv, Qblocks, K]`` selected IDs into canonical BSR.

        Selection order may follow scores. PrimTS consumes ascending block IDs,
        so each row is sorted while the shape-derived indptr is reused.
        """

        if not isinstance(selected_blocks, torch.Tensor):
            raise TypeError("selected_blocks must be a torch.Tensor")
        if selected_blocks.ndim != 4:
            raise ValueError("selected_blocks must have shape [B, Hkv, Qblocks, K]")
        if selected_blocks.dtype != torch.int32:
            raise TypeError("selected_blocks must have dtype torch.int32")
        batch_size, num_kv_heads, num_q_blocks, blocks_per_row = map(int, selected_blocks.shape)
        if min(batch_size, num_kv_heads, num_q_blocks) <= 0:
            raise ValueError("selected_blocks B, Hkv, and Qblocks dimensions must be positive")

        block_indptr = self._get_uniform_indptr(
            device=selected_blocks.device,
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
            num_q_blocks=num_q_blocks,
            blocks_per_row=blocks_per_row,
        )
        block_indices = torch.sort(selected_blocks, dim=-1).values.reshape(-1).contiguous()
        return BlockSparseRoutes(
            block_indptr=block_indptr,
            block_indices=block_indices,
            max_blocks_per_row=blocks_per_row,
        )


@cache
def _get_bit_weights(device: torch.device) -> torch.Tensor:
    bit_positions = torch.arange(_BITS_PER_WORD, dtype=torch.int64, device=device)
    return torch.bitwise_left_shift(torch.ones_like(bit_positions), bit_positions)


def pack_kv_token_mask(
    kv_token_mask: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Pack a bool ``[Skv]`` or ``[B, Skv]`` mask into UInt32 words."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be a Python integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not isinstance(kv_token_mask, torch.Tensor):
        raise TypeError("kv_token_mask must be a torch.Tensor")
    if kv_token_mask.dtype != torch.bool:
        raise TypeError("kv_token_mask must have dtype torch.bool")
    if kv_token_mask.ndim == 1:
        seq_len_kv = int(kv_token_mask.shape[0])
        batched_mask = kv_token_mask.unsqueeze(0).expand(batch_size, -1)
    elif kv_token_mask.ndim == 2 and int(kv_token_mask.shape[0]) == batch_size:
        seq_len_kv = int(kv_token_mask.shape[1])
        batched_mask = kv_token_mask
    else:
        raise ValueError(
            f"kv_token_mask must have shape [Skv] or [{batch_size}, Skv], "
            f"got {tuple(kv_token_mask.shape)}"
        )
    if seq_len_kv <= 0:
        raise ValueError("kv_token_mask sequence length must be positive")

    padded_length = ceil(seq_len_kv / _BITS_PER_WORD) * _BITS_PER_WORD
    if padded_length != seq_len_kv:
        padding = torch.zeros(
            (batch_size, padded_length - seq_len_kv),
            dtype=torch.bool,
            device=kv_token_mask.device,
        )
        batched_mask = torch.cat((batched_mask, padding), dim=1)
    words = (
        batched_mask.reshape(batch_size, -1, _BITS_PER_WORD).to(torch.int64)
        * _get_bit_weights(kv_token_mask.device)
    ).sum(dim=-1)
    return words.to(torch.uint32).contiguous()


__all__ = [
    "BlockSparseForwardInputs",
    "BlockSparseParams",
    "BlockSparseRouteBuilder",
    "BlockSparseRoutes",
    "pack_kv_token_mask",
]
