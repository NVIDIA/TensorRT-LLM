# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One MXFP8 projection on the engine-tuned path: dynamic MXFP8 quantize of a bf16 activation + block-scaled GEMM."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)
from tensorrt_llm._torch.custom_ops.torch_custom_ops import mxfp8_quantize_gemm_autotuned

_BLOCK = 32


def _unswizzle(scale: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """[rows, cols] UE8M0 bytes from the 128x4-swizzled layout (rows padded to 128, columns to 4)."""
    pad_r = (rows + 127) // 128 * 128
    pad_c = (cols + 3) // 4 * 4
    s = scale.view(torch.uint8).reshape(pad_r // 128, pad_c // 4, 32, 4, 4)
    return s.permute(0, 3, 2, 1, 4).reshape(pad_r, pad_c)[:rows, :cols]


def _dequant(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    rows, k = x.shape
    s = _unswizzle(scale, rows, k // _BLOCK).to(torch.int32)
    factor = torch.pow(2.0, (s - 127).to(torch.float32))
    return (
        x.view(torch.float8_e4m3fn).float().view(rows, k // _BLOCK, _BLOCK) * factor.unsqueeze(-1)
    ).view(rows, k)


def mxfp8_reference(act, act_scale, weight, weight_scale) -> torch.Tensor:
    """fp32 product of two dequantized MXFP8 operands (swizzled UE8M0 scales)."""
    with true_fp32_matmul():
        return _dequant(act, act_scale) @ _dequant(weight, weight_scale).t()


class M3Mxfp8Linear(OpWrapper):
    """``x @ weight^T`` for bf16 ``x`` [M, K] and an MXFP8 weight [N, K] (e4m3 + 128x4-swizzled UE8M0 scales), bf16
    out, on the backend the engine selected:

    * ``tuned_quantize_gemm`` (decode CUDA graphs once the engine tuned per-bucket backends): the AutoTuner's quantize
      and GEMM per token bucket (``tune`` profiles them in the engine's tuning pass);
    * otherwise the native ``mxfp8_quantize`` (swizzled scales) and then FlashInfer's MXFP8 GEMM (``flashinfer``) or
      the CUTLASS block-scaled GEMM, through the autotuner while the engine tunes it (``native_tuning``), scaled by
      ``alpha`` (fp32 [1], 1.0 for these checkpoints).

    Every MXFP8 projection of the target (the attention QKV and o_proj, the dense MLPs and the shared expert) runs it
    outside the decode-step kernels.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="sparse-layer QKV, a prefill chunk, native GEMM",
            spec=dict(m=2048, n=1536, k=6144, path="native"),
        ),
        Cell(
            why="o_proj, a prefill chunk, native GEMM",
            spec=dict(m=2048, n=6144, k=1024, path="native"),
        ),
        Cell(
            why="dense down projection, eager decode, native GEMM",
            spec=dict(m=4, n=6144, k=1536, path="native"),
        ),
        Cell(
            why="shared gate_up, FlashInfer GEMM",
            spec=dict(m=64, n=768, k=6144, path="flashinfer"),
        ),
        Cell(why="o_proj, tuned quantize + GEMM", spec=dict(m=64, n=6144, k=1024, path="tuned")),
    )

    note = """
    Exactly the MXFP8 product of the MXFP8 round trip of x (fp32 accumulation, one bf16 rounding): `reference`
    quantizes x with the same block rule (UE8M0 scale = the block's power-of-two ceiling over e4m3's 448) and
    multiplies dequantized operands in fp32; the test's band covers the accumulation order.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        output_dtype: torch.dtype,
        tuned_quantize_gemm: bool,
        tune: bool,
        flashinfer: bool,
        native_tuning: bool,
    ) -> torch.Tensor:
        if tuned_quantize_gemm:
            return mxfp8_quantize_gemm_autotuned(x, weight, weight_scale, output_dtype, tune=tune)
        act, act_scale = torch.ops.trtllm.mxfp8_quantize(x, True)
        if flashinfer:
            return torch.ops.trtllm.flashinfer_mm_mxfp8(
                act, act_scale, weight, weight_scale, output_dtype
            )
        gemm = (
            torch.ops.trtllm.mxfp8_mxfp8_gemm_autotuned
            if native_tuning
            else torch.ops.trtllm.mxfp8_mxfp8_gemm
        )
        return gemm(act, act_scale, weight, weight_scale, alpha, output_dtype)

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        output_dtype: torch.dtype,
        tuned_quantize_gemm: bool,
        tune: bool,
        flashinfer: bool,
        native_tuning: bool,
    ) -> torch.Tensor:
        """The MXFP8 round trip of ``x`` times the dequantized weight, times ``alpha``; the backend flags are
        ignored."""
        m, k = x.shape
        blocks = x.float().view(m, k // _BLOCK, _BLOCK)
        amax = blocks.abs().amax(-1, keepdim=True)
        exp = torch.ceil(torch.log2(torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))))
        scale = torch.pow(2.0, exp)
        q = (blocks / scale).to(torch.float8_e4m3fn).float() * scale
        with true_fp32_matmul():
            out = q.view(m, k) @ _dequant(weight, weight_scale).t()
        return (out * alpha.float()).to(output_dtype)


m3_mxfp8_linear = M3Mxfp8Linear()
