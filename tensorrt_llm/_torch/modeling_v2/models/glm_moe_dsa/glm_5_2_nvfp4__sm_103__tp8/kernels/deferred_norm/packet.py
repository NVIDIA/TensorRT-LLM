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
"""The hidden-state packet a deferred_norm deferred hop hands to the next layer.

``DeferredNormHidden`` stands in for the normalized hidden state ``h =
bf16((x * r) * gamma)`` that the post_moe_hop would return. It is deliberately not
a tensor: a consumer that does not know the packet fails at its first tensor
operation instead of silently computing with the un-normalized activation.
deferred_norm-aware consumers take ``a_frag`` (``a = bf16(x * gamma)`` in the MMA
operand layout below) and the square-sum partials, and apply ``r`` after
their projection; every other consumer calls :meth:`materialize`, which
returns post_moe_hop's normalized output bit for bit.

``a_frag`` layout (``FRAG_ROWS = 8`` token rows, ``K`` columns): for every
16-column block ``s`` and lane ``l = 4 g + t`` of an ``mma.m16n8k16`` B
fragment (token ``g``, column pair ``t``), four BF16 values ``a[g, 16 s + 2 t
+ (0, 1)]`` and ``a[g, 16 s + 8 + 2 t + (0, 1)]``, i.e. flat index ``((s *
32 + l) * 2 + half) * 2 + pair``. A warp then reads one 256-byte run per K
step. Rows ``>= R`` are not written.
"""

from dataclasses import dataclass
from typing import Callable

import torch

FRAG_ROWS = 8


def frag_elems(k: int) -> int:
    return (k // 16) * 32 * 4


def frag_from_rows(a: torch.Tensor) -> torch.Tensor:
    """``a_frag`` of a ``[R <= 8, K]`` tensor (rows ``>= R`` zero)."""
    rows, k = a.shape
    pad = a.new_zeros(FRAG_ROWS, k)
    pad[:rows] = a
    return pad.view(FRAG_ROWS, k // 16, 2, 4, 2).permute(1, 0, 3, 2, 4).reshape(-1).contiguous()


def rows_from_frag(a_frag: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    """Inverse of :func:`frag_from_rows` for the first ``rows`` rows."""
    t = a_frag.view(k // 16, FRAG_ROWS, 4, 2, 2).permute(1, 0, 3, 2, 4).reshape(FRAG_ROWS, k)
    return t[:rows].contiguous()


@dataclass(frozen=True)
class DeferredNormHidden:
    """Deferred RMSNorm output of a post-MoE hop.

    Attributes:
        a_frag: bf16 ``[frag_elems(H)]``, ``bf16(x * gamma)`` in the MMA operand
            layout (module docstring).
        sumsq: fp32 ``[R, P]``, per-warp sums of bf16-rounded squares of ``x``
            in the order the RMSNorm sums them (``P`` = 24 for H = 6144).
        residual: bf16 ``[R, H]``, ``x`` (the hop's residual output); also the
            tensor view used for shapes and output allocation.
        gamma: bf16 ``[H]``, the next layer's input RMSNorm weight.
        eps: RMSNorm epsilon.
        materializer: launches the kernel behind :meth:`materialize`.
    """

    a_frag: torch.Tensor
    sumsq: torch.Tensor
    residual: torch.Tensor
    gamma: torch.Tensor
    eps: float
    materializer: Callable[["DeferredNormHidden"], torch.Tensor]

    @property
    def shape(self) -> torch.Size:
        return self.residual.shape

    @property
    def dtype(self) -> torch.dtype:
        return self.residual.dtype

    @property
    def device(self) -> torch.device:
        return self.residual.device

    @property
    def num_rows(self) -> int:
        return self.residual.shape[0]

    def materialize(self) -> torch.Tensor:
        """Return the normalized hidden state ``bf16((x * r) * gamma)``."""
        return self.materializer(self)


def is_deferred(hidden_states) -> bool:
    return isinstance(hidden_states, DeferredNormHidden)
