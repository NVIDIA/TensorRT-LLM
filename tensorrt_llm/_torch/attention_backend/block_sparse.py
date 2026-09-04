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

"""Algorithm-neutral inputs for block-sparse FMHA implementations."""

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True, slots=True)
class BlockSparseForwardInputs:
    """Block geometry and live routing payload for one attention call.

    Exactly one routing representation is present. Canonical BSR uses
    ``block_indptr`` and ``block_indices``; packed bitmask routing uses
    ``exact_block_bits``. Paired K/V summaries enable proxy routes without
    encoding an algorithm name in this shared carrier.
    """

    q_block_size: int
    kv_block_size: int
    max_blocks_per_row: int | None = None
    block_indptr: torch.Tensor | None = None
    block_indices: torch.Tensor | None = None
    exact_block_bits: torch.Tensor | None = None
    k_summary: torch.Tensor | None = None
    v_summary: torch.Tensor | None = None
    kv_valid_bits: torch.Tensor | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("q_block_size", self.q_block_size),
            ("kv_block_size", self.kv_block_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a Python integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        if self.max_blocks_per_row is not None:
            if isinstance(self.max_blocks_per_row, bool) or not isinstance(
                self.max_blocks_per_row, int
            ):
                raise TypeError("max_blocks_per_row must be a Python integer")
            if self.max_blocks_per_row < 0:
                raise ValueError("max_blocks_per_row must be non-negative")

        has_indptr = self.block_indptr is not None
        has_indices = self.block_indices is not None
        if has_indptr != has_indices:
            raise ValueError("block_indptr and block_indices must be provided together")

        has_bsr = has_indptr
        has_bitmask = self.exact_block_bits is not None
        if has_bsr == has_bitmask:
            raise ValueError("exactly one route representation must be provided")
        if has_bsr and self.max_blocks_per_row is None:
            raise ValueError("BSR routes require max_blocks_per_row")

        has_k_summary = self.k_summary is not None
        has_v_summary = self.v_summary is not None
        if has_k_summary != has_v_summary:
            raise ValueError("k_summary and v_summary must be provided together")

    @property
    def sparse_format(self) -> Literal["bsr", "bitmask"]:
        """Routing representation selected by the live payload."""

        return "bitmask" if self.exact_block_bits is not None else "bsr"

    @property
    def use_proxy_routes(self) -> bool:
        """Whether unselected blocks are represented by K/V summaries."""

        return self.k_summary is not None


__all__ = ["BlockSparseForwardInputs"]
