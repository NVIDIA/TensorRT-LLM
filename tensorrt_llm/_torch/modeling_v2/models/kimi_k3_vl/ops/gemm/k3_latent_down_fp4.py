# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MoE latent_down projection with the routed experts' NVFP4 input quantization fused into
its epilogue."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)

from ...kernels.k3r_gemm import k3_latent_fp8_gemm as _latent
from . import _fp8_reference as ref


class K3LatentDownFp4(OpWrapper):
    """NVFP4 ``(codes uint8 [M, N / 2], scales uint8 [M, N / 16])`` of BF16 ``x @ dequant(w)^T``.

    ``w``: the UE8M0-resmoothed E4M3 ``[N, K]`` weight, ``w_scale`` its DeepGEMM-packed scale,
    ``global_sf``: the consumer's FP32 ``[1]`` NVFP4 global scale (``fp4_quantize``'s). Scales in
    the linear (row-major) layout.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    #: The MoE latent_down (N = 3584, K = 7168) of every deployment: the decode buckets (on SM107
    #: the low-M route, 1 x 32) and a tp1 context chunk (the persistent GEMM, 1 x 128).
    CELLS: tuple[Cell, ...] = (
        Cell(why="one row: decode without MTP", spec={"m": 1}),
        Cell(why="one request with seven draft tokens", spec={"m": 8}),
        Cell(why="eight requests: the largest helix graph", spec={"m": 64}),
        Cell(why="a tp1 context chunk: the persistent MXFP8 GEMM on every arch", spec={"m": 1024}),
    )

    note = """
    Two kernels chained with PDL: the activation quantized to E4M3 with one UE8M0 exponent per
    1 x 128, then a persistent MXFP8 GEMM (no split-K) whose epilogue rounds to BF16 and quantizes
    with ``fp4_quantize``'s arithmetic. Up to ``k3_latent_fp8_gemm.lowm_route_max_m(N, K)`` rows
    (a per-architecture table; on SM107 the decode token counts) the split-K GEMM of
    ``k3_fp8w_lowm_gemm`` runs instead, with 1 x 32 activation exponents and the same epilogue. The
    outputs are new contiguous tensors; the current stream is used.
    """

    def __call__(
        self, x: torch.Tensor, w: torch.Tensor, w_scale: torch.Tensor, global_sf: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _latent.k3_latent_down_fp4(x, w, w_scale, global_sf)

    def reference(
        self, x: torch.Tensor, w: torch.Tensor, w_scale: torch.Tensor, global_sf: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``fp4_quantize`` (catalog reference, linear scales) of the BF16 projection of the
        quantized activation (1 x 32 exponents where the op runs the low-M route, else 1 x 128)."""
        m = x.shape[0]
        n, k = w.shape
        group = 32 if m <= _latent.lowm_route_max_m(n, k) else 128
        y = ref.lowm_gemm(x, w, w_scale, group)
        codes, scales = fp4_quantize.reference(y, global_sf, 16, False, False)
        return codes, scales.view(m, n // 16)


k3_latent_down_fp4 = K3LatentDownFp4()
