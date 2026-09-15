# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Lowered parameters and typed forward arguments of the glm_kpool sparse backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from ..params import SparseBackendForwardArgs, SparseParams

#: Invalid-slot marker in every index tensor this backend consumes. It is the
#: FlashMLA sparse kernel's own invalid contract ("-1 or >= s_kv"), so padded
#: or unselected slots pass through unchanged; clamping them into range would
#: attend a real row.
INDEX_SENTINEL = -1


@dataclass(frozen=True)
class GlmKpoolSparseParams(SparseParams):
    """Lowered runtime parameters for the GLM k-pool sparse-MLA backend."""

    algorithm: Literal["glm_kpool"] = field(init=False, default="glm_kpool")
    #: Latent (compressed KV) width; also the absorbed query head width and
    #: the kernel's d_qk == d_v. 512 on this checkpoint.
    kv_lora_rank: int = 512
    #: Pre-absorption query/key head width; sets the softmax scale. Fully
    #: NoPE: there is no rope component on top of it.
    qk_nope_head_dim: int = 256
    #: Low-rank query bottleneck width; carried into ``MLAParams`` so the
    #: backend's MLA identity states the real checkpoint geometry.
    q_lora_rank: int = 1536
    #: Per-head value width after the absorbed V projection.
    v_head_dim: int = 256
    #: Number of key positions the expanded selection may cover.
    index_topk: int = 2048
    #: Members per compressed pool.
    index_kpool: int = 4
    #: Whether the incomplete trailing pool is always appended.
    index_always_select_tail: bool = True
    #: Indexer key width; the cached per-token row is ``[k | gate | pool key]``.
    index_head_dim: int = 128

    def __post_init__(self) -> None:
        if self.index_kpool <= 0 or self.index_kpool & (self.index_kpool - 1):
            raise ValueError("glm_kpool requires a positive power-of-two index_kpool")
        if self.index_topk <= 0 or self.index_topk % self.index_kpool:
            raise ValueError(
                "glm_kpool requires index_topk to be a positive multiple of index_kpool"
            )
        if not self.index_always_select_tail:
            raise ValueError("glm_kpool requires index_always_select_tail=True")

    @property
    def indices_block_size(self) -> int:
        return 1

    @property
    def packed_state_dim(self) -> int:
        """Width of the per-token ``[k | gate]`` pair the model layer writes."""
        return 2 * self.index_head_dim

    @property
    def cache_row_dim(self) -> int:
        """Width of one indexer cache row: ``[k | gate | pool key]``.

        The pool key of pool ``j`` lives in the trailing ``index_head_dim``
        columns of the row at position ``j * index_kpool`` (its first member);
        it is maintained incrementally by :meth:`GlmKpoolSparseAttention.
        update_pool_keys` so decode never rebuilds pools from scratch.
        """
        return 3 * self.index_head_dim

    @property
    def select_k(self) -> int:
        """Number of pools a query selects."""
        return self.index_topk // self.index_kpool

    @property
    def kernel_output_width(self) -> int:
        """``output_width`` padded to the FlashMLA top-k tile (64)."""
        return -(-self.output_width // 64) * 64

    @property
    def output_width(self) -> int:
        """Fixed logical width of the expanded index rows the model emits."""
        return self.index_topk + (self.index_kpool - 1 if self.index_always_select_tail else 0)


@dataclass(kw_only=True, slots=True)
class GlmKpoolBackendForwardArgs(SparseBackendForwardArgs):
    """``SparseBackendForwardArgs`` plus the backend's own row-id selection.

    ``topk_rows`` carries a selection already translated to latent-cache row
    ids (int32 ``[T, kernel_output_width]``, ``-1`` invalid) by
    :meth:`GlmKpoolSparseAttention.expand_selection`; when present it is
    consumed directly and ``topk_indices`` (request-local positions) is not
    needed.
    """

    topk_rows: torch.Tensor | None = None
