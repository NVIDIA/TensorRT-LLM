# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense-MLP gate_up NVFP4 GEMM with SwiGLU + NVFP4 quantize in its epilogue
(``native_ops.nvfp4_gate_up_swiglu_nvfp4``, built from the target's ``csrc/``).

An SM120 CUTLASS block-scaled GEMM whose epilogue writes down_proj's NVFP4 input
(packed E2M1 and swizzled E4M3 scales) instead of the ``[M, 2I]`` BF16 gate_up
output. It needs the gate_up weight rows permuted at load
(``_nvfp4_recipes.permute_gate_up_for_swiglu_epilogue``) so that each thread's
accumulator pair is one output's (gate, up) and a lane quad owns two whole
16-element scale blocks. ``tile`` picks the CTA tile: 0 = 256x128x128, 1 =
128x128x256, 2 = tile 1 with the MMAs of tile rows 64-127 skipped (only for
``M <= 64``, where those rows are padding).
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)

from ... import native_ops
from ..quantization._nvfp4_recipes import gate_up_swiglu_row_permutation
from ..quantization._nvfp4_reference import SF_VEC, dequantize, swizzled_sf_numel, unswizzle_sf


def _cell(why: str, m: int, tile: int) -> Cell:
    return Cell(why=why, spec=dict(n=34816, k=5120, m=m, tile=tile))


class Nvfp4GateUpSwigluNvfp4(OpWrapper):
    """``fp4_quantize(silu(gate) * up, out_scale, 16)`` with ``[gate | up] =
    bf16(alpha * (act @ weight.T))`` un-permuted, as ``(fp4 [M, N/4] uint8, swizzled
    sf uint8)``; the SF buffer covers ``ceil128(M)`` rows, padding rows zero.

    ``act_fp4`` ``[M, K/2]`` / ``act_sf`` swizzled: gate_up's NVFP4 input;
    ``weight`` ``[N, K/2]`` / ``weight_scale`` swizzled: gate_up's weight with rows in
    `gate_up_swiglu_row_permutation` order; ``alpha`` fp32 ``[1]``; ``out_scale`` fp32
    ``[1]``: down_proj's activation global scale.
    """

    ARCHS = frozenset({Arch.SM_120})

    # Qwen3.8-27B tp1 MLP (hidden 5120, intermediate 17408) from 41 rows (dispatch table):
    # tile 2 at 41-64, tile 1 at 65-447, tile 0 from 448; up to max_num_tokens 8192.
    CELLS: tuple[Cell, ...] = (
        _cell("M=41: first epilogue row, lower-half tile", 41, 2),
        _cell("M=64: last lower-half-tile row", 64, 2),
        _cell("M=65: first 128x128x256 row", 65, 1),
        _cell("M=129: second 128-row tile, 127 padding rows", 129, 1),
        _cell("M=447: last 128x128x256 row", 447, 1),
        _cell("M=448: first 256x128x128 row", 448, 0),
        _cell("M=2079: mixed prefill step", 2079, 0),
        _cell("M=8192: max_num_tokens", 8192, 0),
    )

    note = """
    Metadata consumed: none. Routing (modeling.py): rows >= 41 and not while the
    AutoTuner tunes (tuning runs the unfused gate_up so its row buckets are tuned).

    Preconditions: weight rows permuted at load (the GEMM's column order is that
    permutation; an unpermuted weight is silently a different SwiGLU pairing); N and K
    multiples of 128; the activation from `fp4_quantize(x, gate_up_act_global, 16)` or
    the post-attention norm's NVFP4 epilogue; `tile=2` only for M <= 64 (the op refuses
    larger M loudly).

    Numerics: bitwise the unfused path on the same permuted weight -- the CUTLASS FP4
    GEMM's `bf16_rn(alpha * acc)`, then `silu_mul_nvfp4_permuted`
    (`h = bf16_rn(up * (sigmoid(g) * g))` with Triton's sigmoid, `div.full(1,
    ex2.approx(-g * log2e) + 1)`, and the `fp4_quantize` tail: `rcp.approx.ftz`,
    `mul.rn`, `cvt.rn.satfinite.e4m3x2`, `cvt.rn.satfinite.e2m1x2`). All three tiles
    produce the same bytes. Against `reference` (fp32 GEMM, torch's sigmoid) only the
    BF16 intermediate's last bit can differ, and with it a block's scale or a code.

    PDL: launched with the PDL attribute (TRTLLM_ENABLE_PDL).
    """

    def __call__(
        self,
        act_fp4: torch.Tensor,
        weight: torch.Tensor,
        act_sf: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        out_scale: torch.Tensor,
        tile: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fp4, sf = native_ops.nvfp4_gate_up_swiglu_nvfp4(
            act_fp4, weight, act_sf, weight_scale, alpha, out_scale, tile
        )
        return fp4, sf

    def reference(
        self,
        act_fp4: torch.Tensor,
        weight: torch.Tensor,
        act_sf: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        out_scale: torch.Tensor,
        tile: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """fp32 GEMM of the dequantized operands (TF32 off) times alpha, rounded to BF16;
        columns put back in ``[gate | up]`` order; ``bf16(silu(gate) * up)`` in fp32 with
        torch's sigmoid; `fp4_quantize`'s native-torch NVFP4 with swizzled scales.
        `tile` changes nothing and is ignored."""
        m, half = act_fp4.shape
        n = weight.shape[0]
        k = half * 2
        d = n // 2
        a = dequantize(act_fp4, unswizzle_sf(act_sf, m, k // SF_VEC))
        w = dequantize(weight, unswizzle_sf(weight_scale, n, k // SF_VEC))
        with true_fp32_matmul():
            h_perm = (alpha.float().reshape(()) * (a @ w.t())).to(torch.bfloat16)
        del a, w
        inv = torch.argsort(gate_up_swiglu_row_permutation(n).to(h_perm.device))
        h = h_perm[:, inv].float()
        gate, up = h[:, :d], h[:, d:]
        act = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)
        return fp4_quantize.reference(act, out_scale, SF_VEC, False, True)

    def is_valid(
        self,
        act_fp4: torch.Tensor,
        weight: torch.Tensor,
        act_sf: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        out_scale: torch.Tensor,
        tile: int,
    ) -> None:
        # The op loads alpha[0] and out_scale[0] and reads both scale buffers over their
        # padded rectangles without checking their lengths.
        assert alpha.numel() == 1 and out_scale.numel() == 1, (
            "alpha and out_scale must hold exactly one element"
        )
        k = act_fp4.shape[1] * 2
        assert act_sf.numel() >= swizzled_sf_numel(act_fp4.shape[0], k // SF_VEC), (
            "act_sf is shorter than the padded swizzled rectangle of the activation"
        )
        assert weight_scale.numel() >= swizzled_sf_numel(weight.shape[0], k // SF_VEC), (
            "weight_scale is shorter than the padded swizzled rectangle of the weight"
        )


nvfp4_gate_up_swiglu_nvfp4 = Nvfp4GateUpSwigluNvfp4()
