# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RMSNorm + RoPE of the MLA drafter's context latents of every layer in one CuTe DSL launch,
the RoPE table looked up in the kernel."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._dspark_mla_rope import dspark_mla_ctx_norm_rope
from .k3_dspark_rmsnorm_rope import rmsnorm_rope_reference


class K3DsparkMlaCtxNormRope(OpWrapper):
    """``[L, N, kv_lora_rank + rope_dim]`` BF16, new: for each layer ``l`` the RMSNorm of
    ``latent[:, l, :kv_lora_rank]`` times ``norm_weight[l]``, and the latent's RoPE columns
    rotated by ``freqs[positions]`` (interleaved pairs, not normalized).

    ``latent`` is BF16 ``[N, L, head_dim]`` with contiguous ``(L, head_dim)`` rows and any token
    stride; ``norm_weight`` BF16 ``[L, kv_lora_rank]``; ``positions`` int64 ``[N]``; ``freqs``
    FP32 ``[P, rope_dim // 2, 2]`` (cos, sin), contiguous.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS = (
        Cell(
            why="helix16 ULL: the accepted tokens of 4 requests x 8 (DSpark 7), 5 layers",
            spec=dict(rows=32, layers=5),
        ),
        Cell(
            why="helix8 c240: 20 requests x 6 (DSpark 5), 5 layers",
            spec=dict(rows=120, layers=5),
        ),
        Cell(
            why="helix8 / helix16: one accepted token",
            spec=dict(rows=1, layers=5),
        ),
    )

    note = """
    Bitwise ``k3_dspark_rmsnorm_rope(latent[:, l], norm_weight[l], freqs[positions], 1, rope_dim,
    eps, True, True, False, kv_lora_rank)`` per layer, in one launch (``dspark_tail_v2``). A
    process-wide compile cache holds one kernel per (L, kv_lora_rank, rope_dim, eps).
    """

    def __call__(
        self,
        latent: torch.Tensor,
        norm_weight: torch.Tensor,
        positions: torch.Tensor,
        freqs: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        return dspark_mla_ctx_norm_rope(latent, norm_weight, positions, freqs, eps)

    def reference(
        self,
        latent: torch.Tensor,
        norm_weight: torch.Tensor,
        positions: torch.Tensor,
        freqs: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        num_layers = latent.shape[1]
        kv_lora_rank = norm_weight.shape[1]
        rope_dim = latent.shape[-1] - kv_lora_rank
        table = freqs[positions]
        return torch.stack(
            [
                rmsnorm_rope_reference(
                    latent[:, layer],
                    norm_weight[layer],
                    table,
                    1,
                    rope_dim,
                    eps,
                    True,
                    True,
                    False,
                    kv_lora_rank,
                )
                for layer in range(num_layers)
            ]
        )

    def is_valid(
        self,
        latent: torch.Tensor,
        norm_weight: torch.Tensor,
        positions: torch.Tensor,
        freqs: torch.Tensor,
        eps: float,
    ) -> None:
        # The kernel indexes the table by position without a bound.
        if positions.numel():
            assert int(positions.max()) < freqs.shape[0], "a position past the RoPE table"
            assert int(positions.min()) >= 0, "negative position"


k3_dspark_mla_ctx_norm_rope = K3DsparkMlaCtxNormRope()
