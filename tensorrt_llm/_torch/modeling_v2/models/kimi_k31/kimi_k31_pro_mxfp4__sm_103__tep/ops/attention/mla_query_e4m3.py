# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA: the absorbed query as per-tensor E4M3 ``[latent 512 | rope 64]`` rows (Triton)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.attention.mla_query_e4m3 import quantize_query_e4m3

_FP8_MAX = 448.0


def _spec(tokens, **overrides) -> dict:
    spec = dict(tokens=tokens, heads=96, fused=True)
    spec.update(overrides)
    return spec


class K31MlaQueryE4m3(OpWrapper):
    """``out[n, h] = e4m3(clamp([q_latent[n, h] | q_pe[n, h]] * scale, -448, 448))`` (FP32 product, round to
    nearest even). ``q_latent`` BF16 ``[n, H, 512]`` and ``q_pe`` BF16 ``[n, H, 64]`` with unit last strides
    (views of one fused ``[n, H, 576]`` query are fine), ``scale`` an FP32 ``[1]`` device tensor, ``out``
    contiguous E4M3 ``[n, H, 576]``. Returns None.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(why="one decode token, 96 heads", spec=_spec(1)),
        Cell(why="DEP16 verify step: 98 rows", spec=_spec(98)),
        Cell(why="DEP16 verify step: 224 rows", spec=_spec(224)),
        Cell(why="a context chunk of 2048 rows", spec=_spec(2048)),
        Cell(why="separate latent and rope tensors", spec=_spec(33, fused=False)),
        Cell(
            why="saturation: values beyond 448 after scaling",
            spec=_spec(16, data="wide"),
        ),
    )
    note = """
    ARCHS lists what was driven: sm_103 (GB300).

    The TEP perf target's token-sharded context prefill (ShardQ) quantizes the absorbed query (q_nope @ W_UK |
    q_pe) with the layer's KV quantization scale (kv_scale_orig_quant) before the PrimTS sparse MLA, as the MLA
    generation preprocessing does for the FP8 latent cache.
    Bitwise equal to the reference.
    """

    def __call__(
        self,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor,
        scale: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        quantize_query_e4m3(q_latent, q_pe, scale, out)

    def reference(
        self,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor,
        scale: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        lat = (q_latent.float() * scale.float()).clamp(-_FP8_MAX, _FP8_MAX)
        rope = (q_pe.float() * scale.float()).clamp(-_FP8_MAX, _FP8_MAX)
        out.copy_(torch.cat([lat, rope], dim=-1).to(torch.float8_e4m3fn))

    def is_valid(
        self,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor,
        scale: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        assert q_latent.shape[:2] == q_pe.shape[:2] == out.shape[:2], "rows and heads must match"
        assert q_latent.shape[-1] == 512 and q_pe.shape[-1] == 64 and out.shape[-1] == 576


k31_mla_query_e4m3 = K31MlaQueryE4m3()
