# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-place RoPE of the DSpark drafter's q / k heads from a precomputed fp32 cos / sin table (flashinfer)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

try:
    # Registered only where flashinfer imports; the drafter declines its generic block without it.
    from tensorrt_llm._torch.custom_ops import (
        flashinfer_apply_rope_with_cos_sin_cache_inplace as _rope_kernel,
    )
except ImportError:  # pragma: no cover - flashinfer is part of the sm_100 image
    _rope_kernel = None


def available() -> bool:
    """Whether the flashinfer kernel is registered in this build."""
    return _rope_kernel is not None


class K3DraftRope(OpWrapper):
    """Rotate every ``head_size`` head of ``query`` [N, H * head_size] and ``key`` [N, KV * head_size] in place, row
    ``i`` at position ``positions[i]``, from ``cos_sin_cache`` [max_pos, head_size] fp32 (``[cos | sin]`` halves).
    NeoX layout when ``is_neox``, else GPT-J. Returns None."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"{heads}/{kv} heads of 64, N={rows}, {table}: " + why,
            spec=dict(heads=heads, kv=kv, rows=rows, table=table),
        )
        for heads, kv, why in (
            (4, 1, "RadixArk drafter, one TP16 rank (64 / 16 heads)"),
            (6, 1, "the drafter entries' layout"),
            (0, 5, "k only: the context K of all 5 layers, one dummy q head"),
        )
        for rows, table in ((7, "yarn"), (56, "plain"), (1000, "yarn"))
    )

    note = """
    The kernel and arguments the built-in DFlash drafter runs for q / k RoPE and for
    the context K (flashinfer apply_rope_with_cos_sin_cache_inplace). q and k
    are bf16 views with a dense head dimension; positions int32 [N]. The table
    is whatever RoPE the drafter declares (plain or YaRN, built from its
    RopeParams); the kernel only gathers rows of it. With a q of 0 heads the
    caller passes a one-head scratch q, as the built-in drafter does. The reference rounds the
    rotated fp32 values once to bf16; the kernel computes in fp32 too.
    """

    def __call__(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        head_size: int,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
    ) -> None:
        _rope_kernel(positions, query, key, head_size, cos_sin_cache, is_neox)

    def reference(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        head_size: int,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
    ) -> None:
        half = head_size // 2
        table = cos_sin_cache.float()[positions.long()]
        cos, sin = table[:, None, :half], table[:, None, half:]
        for x in (query, key):
            y = x.float().reshape(x.shape[0], -1, head_size)
            if is_neox:
                y1, y2 = y[..., :half], y[..., half:]
                out = torch.cat([y1 * cos - y2 * sin, y2 * cos + y1 * sin], -1)
            else:
                y1, y2 = y[..., 0::2], y[..., 1::2]
                out = torch.stack([y1 * cos - y2 * sin, y2 * cos + y1 * sin], -1).flatten(-2)
            x.copy_(out.reshape(x.shape).to(x.dtype))

    def is_valid(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        head_size: int,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
    ) -> None:
        assert cos_sin_cache.dtype == torch.float32, "the cos / sin table must be fp32"
        assert cos_sin_cache.dim() == 2 and cos_sin_cache.shape[1] == head_size, (
            f"the table must be [max_pos, head_size={head_size}]; got {tuple(cos_sin_cache.shape)}"
        )
        assert positions.dtype == torch.int32 and positions.numel() == key.shape[0], (
            "positions must be int32, one per row"
        )
        assert query.shape[0] == key.shape[0], "query and key must have the same rows"
        for x in (query, key):
            assert x.shape[-1] % head_size == 0 and x.stride(-1) == 1, (
                "each row must be whole heads with a dense last dimension"
            )


k3_draft_rope = K3DraftRope()
