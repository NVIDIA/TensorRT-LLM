# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tuned NVFP4 W4A4 GEMM for SM120: ``alpha * (act @ weight.T)`` in BF16 or FP32.

Every block-scaled NVFP4 projection of this target goes through here: the dense
MLP (gate_up below the SwiGLU-epilogue rows, down_proj), the W4A4 copies of the FP8
projections above their decode caps (in_proj_qkvz, out_proj, qkv_proj, o_proj) and
the lm_head at 56 rows and more (FP32 logits). The AutoTuner picks one of four
SM120 GEMMs per (N, K, row bucket) -- TRT-LLM's CUTLASS FP4 GEMM, cuBLASLt's FP4
GEMM, the CUDA-core GEMM at up to 8 rows, and FlashInfer's SM120 CUTLASS GEMM alone
on under-filled grids at 64-255 rows -- profiling with a cold L2, with extra row
buckets for mixed prefill+decode steps picked near-tie-stably
(``_nvfp4_gemm_runner`` has the candidates and the tuning configuration).

``tune_rows`` is the row count the tactic is chosen for. It is the call's own row
count except in the final decoder layer of a long prefill step, which runs only the
rows the logits read and must take the tactic -- the kernel and its summation order
-- the unsliced call would have taken, so the rows it keeps are bitwise those of the
unsliced run.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

from ..quantization._nvfp4_reference import SF_VEC, dequantize, swizzled_sf_numel, unswizzle_sf
from ._nvfp4_gemm_runner import run

#: Output columns the reference dequantizes at a time (the lm_head is 248320 wide).
_REF_CHUNK_N = 16384


def _cell(
    why: str,
    n: int,
    k: int,
    m: int,
    out: torch.dtype = torch.bfloat16,
    tune_rows: int | None = None,
) -> Cell:
    return Cell(
        why=why, spec=dict(n=n, k=k, m=m, out=out, tune_rows=m if tune_rows is None else tune_rows)
    )


class Nvfp4GemmSm120(OpWrapper):
    """``(alpha * (act @ weight.T)).to(output_dtype)`` over block-scaled NVFP4 operands.

    ``act_fp4`` ``[M, K/2]`` and ``weight`` ``[N, K/2]`` packed E2M1 (uint8), ``act_sf``
    and ``weight_scale`` their 128x4-swizzled E4M3 scale buffers, ``alpha`` fp32 ``[1]``.
    Returns a fresh ``[M, N]`` tensor of ``output_dtype`` (BF16, or FP32 for the
    lm_head).
    """

    ARCHS = frozenset({Arch.SM_120})

    # Qwen3.8-27B tp1, max_num_tokens 8192 (dispatch table): gate_up (34816, 5120) at 1-40
    # rows (and every row count while the AutoTuner tunes), down (5120, 17408) at every row
    # count, the FP8 projections' W4A4 copies above their decode caps -- in_proj_qkvz
    # (16384, 5120) and the (5120, 6144) out_proj / o_proj from 33 rows, qkv_proj
    # (14336, 5120) from 65 -- and the lm_head (248320, 5120) from 56 rows in FP32.
    CELLS: tuple[Cell, ...] = (
        _cell("gate_up M=1: cuda_core among the candidates", 34816, 5120, 1),
        _cell("gate_up M=8: last cuda_core row", 34816, 5120, 8),
        _cell("gate_up M=40: last row below the SwiGLU epilogue", 34816, 5120, 40),
        _cell("down M=4: decode", 5120, 17408, 4),
        _cell("down M=64: FlashInfer alone (40 tiles on 188 SMs)", 5120, 17408, 64),
        _cell("down M=255: FlashInfer's last row (bucket 128)", 5120, 17408, 255),
        _cell("down M=2079: mixed bucket 2080, cuBLASLt algorithm ids", 5120, 17408, 2079),
        _cell("down M=8191: mixed bucket 8191, max_num_tokens - 1", 5120, 17408, 8191),
        _cell("qkvz W4A4 M=33: first row above the Marlin cap", 16384, 5120, 33),
        _cell("qkvz W4A4 M=2048: one prefill chunk", 16384, 5120, 2048),
        _cell("qkv W4A4 M=65: first row above the qkv Marlin cap", 14336, 5120, 65),
        _cell("o_proj/out_proj W4A4 M=33", 5120, 6144, 33),
        _cell("o_proj/out_proj W4A4 M=128: FlashInfer alone", 5120, 6144, 128),
        _cell(
            "o_proj final-layer slice: 5 read rows tuned as the 2080-row step",
            5120,
            6144,
            5,
            tune_rows=2080,
        ),
        _cell("lm_head M=56: first W4A4 row, FP32 logits", 248320, 5120, 56, out=torch.float32),
        _cell(
            "lm_head M=256: top of the tuned lm_head buckets", 248320, 5120, 256, out=torch.float32
        ),
    )

    note = """
    Metadata consumed: `tune_rows` (the step's unsliced row count in the final layer's
    row slice; otherwise `act_fp4.shape[0]`). The tactic is the AutoTuner's: the first
    call of a (N, K, dtype) while the AutoTuner tunes profiles every row bucket up to
    the call's rows; outside tuning a call takes its bucket's cached tactic, and on a
    cache miss the CUTLASS default config (as the built-in nvfp4_gemm does).

    Preconditions: both operands and both scale buffers contiguous; scale buffers of at
    least `pad_up(rows, 128) * pad_up(K/16, 4)` bytes; `alpha` one fp32 element. The
    activation comes from `fp4_quantize(x, act_global, 16)` (swizzled) or from a
    producer that writes the same bytes (the gated RMSNorm, the attention output gate,
    the post-attention norm).

    Numerics: every candidate multiplies the E2M1 x E4M3-scaled operands exactly and
    accumulates in fp32 (cuBLASLt split-K reduces in fp32), times alpha, one rounding to
    `output_dtype`; the candidates differ in summation order only. `fi_cutlass` is offered
    only for BF16/FP16 output.

    Tuning: cold L2; mixed buckets {p+32 for p >= 2048, p-1 for p >= 4096} with a 3%
    near-tie band and the power-of-two parent preferred; cuBLASLt tactics above 1024
    rows are algorithm identities. FlashInfer's module is JIT-built once per process;
    if it cannot be built the other candidates serve its rows (a warning is logged).

    Hard limits
      M          1..8192 (max_num_tokens)            driven 1..8191
      N, K       the projections above               all driven
      fi_cutlass 64 <= M <= 255, 2*ceil(N/128)*ceil(M/128) <= #SMs, N >= 1024,
                 N % 128 == 0, K % 256 == 0           driven at 64, 128, 255
    """

    def __call__(
        self,
        act_fp4: torch.Tensor,
        weight: torch.Tensor,
        act_sf: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        output_dtype: torch.dtype,
        tune_rows: int,
    ) -> torch.Tensor:
        return run(act_fp4, weight, act_sf, weight_scale, alpha, output_dtype, tune_rows)

    def reference(
        self,
        act_fp4: torch.Tensor,
        weight: torch.Tensor,
        act_sf: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        output_dtype: torch.dtype,
        tune_rows: int,
    ) -> torch.Tensor:
        """Dequantize both operands from their packed and swizzled form, one fp32
        product (TF32 off), alpha, one rounding. `tune_rows` selects a tactic, which
        changes summation order only; it is ignored."""
        m, half = act_fp4.shape
        n = weight.shape[0]
        k = half * 2
        a = dequantize(act_fp4, unswizzle_sf(act_sf, m, k // SF_VEC))
        w_sf = unswizzle_sf(weight_scale, n, k // SF_VEC)
        scale = alpha.float().reshape(())
        out = torch.empty((m, n), dtype=output_dtype, device=act_fp4.device)
        with true_fp32_matmul():
            for n0 in range(0, n, _REF_CHUNK_N):
                n1 = min(n, n0 + _REF_CHUNK_N)
                w = dequantize(weight[n0:n1], w_sf[n0:n1])
                out[:, n0:n1] = (scale * (a @ w.t())).to(output_dtype)
        return out

    def is_valid(
        self,
        act_fp4: torch.Tensor,
        weight: torch.Tensor,
        act_sf: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        output_dtype: torch.dtype,
        tune_rows: int,
    ) -> None:
        # cuBLASLt ignores strides and none of the candidates rejects a longer alpha or a
        # short scale buffer (the kernels index the padded rectangle): each returns a
        # plausible wrong product.
        assert act_fp4.is_contiguous() and weight.is_contiguous(), (
            "act_fp4 and weight must be contiguous; cuBLASLt ignores strides"
        )
        assert act_sf.is_contiguous() and weight_scale.is_contiguous(), (
            "act_sf and weight_scale must be contiguous; cuBLASLt ignores strides"
        )
        assert alpha.numel() == 1, "alpha must hold exactly one element"
        if act_fp4.dim() == 2 and weight.dim() == 2:
            k = act_fp4.shape[1] * 2
            assert act_sf.numel() >= swizzled_sf_numel(act_fp4.shape[0], k // SF_VEC), (
                "act_sf is shorter than the padded swizzled rectangle of the activation"
            )
            assert weight_scale.numel() >= swizzled_sf_numel(weight.shape[0], k // SF_VEC), (
                "weight_scale is shorter than the padded swizzled rectangle of the weight"
            )
        # A tactic tuned for fewer rows than the call has is not the unsliced call's.
        assert tune_rows >= act_fp4.shape[0], (
            f"tune_rows {tune_rows} is below the call's {act_fp4.shape[0]} rows"
        )


nvfp4_gemm_sm120 = Nvfp4GemmSm120()
