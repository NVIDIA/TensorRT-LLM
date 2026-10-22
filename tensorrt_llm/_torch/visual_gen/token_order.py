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
"""Token orderings for block-sparse attention over a 3D latent grid.

Block-sparse attention kernels (Sol-Attn, skip-softmax) decide per block of
consecutive tokens. In the native T-H-W raster order a block is a run along
width inside one frame row, so blocks have weak 3D locality. A Morton
(Z-order) permutation makes consecutive tokens cover compact 3D neighborhoods
instead. Dense attention is permutation-equivariant, so the reorder only
changes the result of block-wise decisions, never the dense math.

A model applies the permutation once to its token sequence and RoPE tables
before the block stack and the inverse after it; everything in between (norms,
MLP, adaLN, attention) is per-token or permutation-equivariant.
"""

from typing import Dict, Optional, Tuple

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.visual_gen.sparse_attention import TokenOrder

_MORTON_CACHE: Dict[Tuple[int, int, int], Tuple[torch.Tensor, torch.Tensor]] = {}


def _part1by2(value: torch.Tensor) -> torch.Tensor:
    """Spread the low 21 bits of ``value`` so that two zero bits follow each bit."""
    value = value & 0x1FFFFF
    value = (value | (value << 32)) & 0x1F00000000FFFF
    value = (value | (value << 16)) & 0x1F0000FF0000FF
    value = (value | (value << 8)) & 0x100F00F00F00F00F
    value = (value | (value << 4)) & 0x10C30C30C30C30C3
    value = (value | (value << 2)) & 0x1249249249249249
    return value


def morton3d_permutation(
    grid: Tuple[int, int, int], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(perm, inverse)`` for a ``(frames, height, width)`` token grid.

    ``x.index_select(seq_dim, perm)`` puts tokens in Morton order;
    ``index_select(seq_dim, inverse)`` restores the raster order. Results are
    cached per grid on CPU and moved to ``device`` on each call.
    """
    key = tuple(int(x) for x in grid)
    hit = _MORTON_CACHE.get(key)
    if hit is None:
        frames, height, width = key
        linear = torch.arange(frames * height * width, dtype=torch.long)
        frame_area = height * width
        t = linear // frame_area
        rem = linear - t * frame_area
        h = rem // width
        w = rem - h * width
        code = _part1by2(w) | (_part1by2(h) << 1) | (_part1by2(t) << 2)
        perm = torch.argsort(code)
        hit = (perm, torch.argsort(perm))
        _MORTON_CACHE[key] = hit
    return hit[0].to(device), hit[1].to(device)


def resolve_token_order(sparse_attention_config) -> TokenOrder:
    """Read the ``token_order`` a sparse-attention config asks for (``native`` if none).

    The value is already validated by ``SolAttentionConfig``.
    """
    return getattr(sparse_attention_config, "token_order", None) or "native"


def require_native_token_order(sparse_attention_config, model_name: str) -> None:
    """Raise when a config asks for a token order that ``model_name`` does not wire.

    Only models that apply :class:`TokenReorder` around their block stack honor
    ``token_order``; everywhere else a non-native order would silently run as
    native, so the pipeline rejects it at construction.
    """
    order = resolve_token_order(sparse_attention_config)
    if order != "native":
        raise ValueError(
            f"sparse_attention_config.token_order={order!r} is only supported by the "
            f"Wan pipelines; use token_order='native' for {model_name}."
        )


class TokenReorder:
    """Per-model helper that applies a token ordering around a block stack.

    ``permute`` / ``restore`` are identities when the order is ``native``. The
    permutation is a fixed index tensor per grid, so it is safe under CUDA graph
    capture and torch.compile. Index tensors are kept for every grid and device
    this instance has prepared: a captured graph replays the pointers it saw at
    capture time, so switching grids must not release them.
    """

    def __init__(self, order: TokenOrder):
        self.order = order
        self._perm: Optional[torch.Tensor] = None
        self._inverse: Optional[torch.Tensor] = None
        self._grid: Optional[Tuple[int, int, int]] = None
        self._indices: Dict[
            Tuple[Tuple[int, int, int], torch.device], Tuple[torch.Tensor, torch.Tensor]
        ] = {}

    @property
    def active(self) -> bool:
        return self.order != "native"

    def prepare(self, grid: Tuple[int, int, int], device: torch.device) -> None:
        """Resolve the permutation for ``grid``; a no-op when the grid is unchanged."""
        if not self.active:
            return
        key = tuple(int(x) for x in grid)
        if self._grid == key and self._perm is not None and self._perm.device == device:
            return
        hit = self._indices.get((key, device))
        if hit is None:
            hit = morton3d_permutation(key, device)
            self._indices[(key, device)] = hit
            logger.info(
                f"[token-order] {self.order} ordering over grid {key} ({hit[0].numel()} tokens)"
            )
        self._perm, self._inverse = hit
        self._grid = key

    def permute(self, tensor: Optional[torch.Tensor], dim: int = 1) -> Optional[torch.Tensor]:
        if tensor is None or not self.active:
            return tensor
        return tensor.index_select(dim, self._perm)

    def restore(self, tensor: Optional[torch.Tensor], dim: int = 1) -> Optional[torch.Tensor]:
        if tensor is None or not self.active:
            return tensor
        return tensor.index_select(dim, self._inverse)
