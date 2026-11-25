# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The context query projection as stock's fused query path runs it (`DeepseekV41Attention.
_project_prefill_query_and_swa`): the q_norm and the MXFP8 quantization of its output in one launch (`rms_norm_mxfp8`),
then the Q-B GEMM with the 64-channel GPT-J RoPE of every 512-channel head in its epilogue (`query_gemm_rope`, CuTe
DSL). The norm is not folded into the GEMM: only the RoPE is (with both folded, a column's pair partner misses the row's
1/rms scale).
"""

from __future__ import annotations

from typing import Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .csa2_mxfp8_linear import _dequant
from .csa2_rope import csa2_rope_tail

_HEAD = 512
_ROPE = 64


class Csa2QueryGemmRope(OpWrapper):
    """`q [M, N] bf16 = rope_tails(rms_norm(x) @ W.T)`: `x` bf16 `[M, K]` (the q_a latent, unit column stride; K a
    multiple of 128, at most 8192), `norm` `(gamma bf16 [K], eps)`, `weight` E4M3 `[N, K]` (N a multiple of 512) with
    its CUTLASS-swizzled UE8M0 scales `weight_sw`, `alpha` fp32 `[1]` ones, `pos` int32 `[M]`, `cos_sin` the engine's
    fp32 `[P, 64]` table (32 cos then 32 sin per position). Also returns the normalized latent's MXFP8 `(data, scales)`,
    which another projection of the same latent (the index query) can read. M of at least one 128-row tile (stock runs
    it from 1024 rows)."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="dep2 context step: 64-head q_b (N 32768, K 1280), 1024 rows of varying RMS",
            spec=dict(m=1024, n=32768, k=1280, rms_spread=True),
        ),
        Cell(
            why="dep2 context step of 4100 rows (a partial last tile)",
            spec=dict(m=4100, n=32768, k=1280, rms_spread=True),
        ),
        Cell(
            why="300 rows, 8 heads (N 4096): fewer heads than one 256-column tile pair",
            spec=dict(m=300, n=4096, k=1280, rms_spread=True),
        ),
        Cell(
            why="dep8 decode, 160 rows (40 requests x 4) of the 64-head q_b",
            spec=dict(m=160, n=32768, k=1280, rms_spread=True),
        ),
        Cell(
            why="dep8 decode, 24 rows: the smallest step past the low-M rows",
            spec=dict(m=24, n=32768, k=1280, rms_spread=True),
        ),
    )
    note = """
    The latent is normalized and quantized to MXFP8 (a 3-bit mantissa) before the GEMM, as stock's query path does: gate
    3% relative RMS against the fp32 product of the unquantized normalized rows; against `csa2_mxfp8_linear` over the
    same MXFP8 latent (GEMM, then a separate RoPE) within bf16 rounding of the output. Rows have RMS spread over two
    decades, so a norm or RoPE that read another row's scale would show.
    """

    def __call__(
        self, x, norm, weight, weight_sw, alpha, pos, cos_sin
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        from tensorrt_llm._torch.attention.backends.sparse.csa2.projection import rms_norm_mxfp8
        from tensorrt_llm._torch.attention.backends.sparse.csa2.query_projection import (
            query_gemm_rope,
        )

        _, data, scales = rms_norm_mxfp8(x, norm[0], norm[1], normalized=False)
        q, _ = query_gemm_rope(data, scales, weight, weight_sw, alpha, pos, cos_sin, quantize=False)
        return q, (data, scales)

    def reference(
        self, x, norm, weight, weight_sw, alpha, pos, cos_sin, weight_scale=None
    ) -> torch.Tensor:
        """fp32 `rms_norm(x) * gamma @ dequant(W).T`, RoPE tails rotated (`csa2_rope_tail.reference`); `weight_scale`
        the row-major 1x32 UE8M0 scales of `weight` (the test passes them)."""
        a = x.float()
        a = a * torch.rsqrt(a.square().mean(-1, keepdim=True) + norm[1]) * norm[0].float()
        y = a @ _dequant(weight, weight_scale).t()
        m, n = y.shape
        table = torch.stack([cos_sin[:, :32], cos_sin[:, 32:]], dim=-1)
        return csa2_rope_tail.reference(
            y.view(m, n // _HEAD, _HEAD), table, pos, _ROPE, False
        ).reshape(m, n)

    def is_valid(self, x, norm, weight, weight_sw, alpha, pos, cos_sin, weight_scale=None) -> None:
        m, k = x.shape
        assert x.dtype == torch.bfloat16 and x.stride(1) == 1 and k % 128 == 0 and k <= 8192
        assert (
            norm[0].shape == (k,) and norm[0].dtype == torch.bfloat16 and norm[0].device == x.device
        )
        assert alpha.dtype == torch.float32 and alpha.numel() == 1, (
            "alpha is the GEMM's one fp32 global scale"
        )
        assert (
            weight.dtype == torch.float8_e4m3fn
            and weight.shape[1] == k
            and weight.shape[0] % _HEAD == 0
        )
        assert weight_sw.dtype == torch.uint8 and weight_sw.numel() == weight.shape[0] * k // 32
        assert pos.dtype == torch.int32 and pos.shape == (m,) and pos.is_contiguous()
        assert (
            cos_sin.dtype == torch.float32 and cos_sin.is_contiguous() and cos_sin.shape[-1] == 64
        )


csa2_query_gemm_rope = Csa2QueryGemmRope()


class Csa2QueryGemmFp8(OpWrapper):
    """`csa2_query_gemm_rope` emitting only its unit-scale E4M3 Q (stock's `fp8_query_gemm`: the fused context path's
    query when the context kernel reads only the E4M3 Q, `_unit_scale_context_query`): `q [M, N] E4M3 =
    e4m3(rope_tails(rms_norm(x) @ W.T))` at scale one, never stored in bf16, plus the normalized latent's MXFP8
    `(data, scales)`. Same operands as `csa2_query_gemm_rope`. A tile's first call compiles it, so a graph capture
    needs an eager call of the same rows first (TRT-LLM's graph warm-up runs).

    Also stock's decode FP8 query (`_project_query_latent` with `fp8_query`, from 32 rows): `x_mxfp8` is the normalized
    latent already in MXFP8 (`csa2_rms_norm_mxfp8`'s output, shared with the index q_b), used instead of normalizing
    `x` again. `amax` (int32 `[1]`, zero on entry): the GEMM also raises it to the FP32 bits of `amax(|q|)`, the word
    the `gen_fmha` staging caps its scale with (stock's `_FP8Query.word`)."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="dep2 context step: 64-head q_b (N 32768, K 1280), 1024 rows of varying RMS",
            spec=dict(m=1024, n=32768, k=1280, rms_spread=True),
        ),
        Cell(
            why="dep2 context step of 4100 rows (a partial last tile, the wide 2-CTA tiles)",
            spec=dict(m=4100, n=32768, k=1280, rms_spread=True),
        ),
        Cell(
            why="dep8 decode step of 1280 rows (320 requests x 4)",
            spec=dict(m=1280, n=32768, k=1280, rms_spread=True),
        ),
        Cell(
            why="dep32 decode step of 150 rows (one 256-row tile)",
            spec=dict(m=150, n=32768, k=1280, rms_spread=True),
        ),
        Cell(
            why="the smallest decode step on the FP8 query (32 rows)",
            spec=dict(m=32, n=32768, k=1280, rms_spread=True),
        ),
    )
    note = """
    The query is E4M3 at unit scale (as the context kernel quantizes a bf16 one on load with its unit staging scale):
    gate 7% relative RMS of the dequantized query against the fp32 product, and the E4M3 rounding of
    `csa2_query_gemm_rope`'s bf16 query within one E4M3 step.
    """

    def __call__(
        self, x, norm, weight, weight_sw, alpha, pos, cos_sin, x_mxfp8=None, amax=None
    ) -> tuple:
        from tensorrt_llm._torch.attention.backends.sparse.csa2.projection import rms_norm_mxfp8
        from tensorrt_llm._torch.attention.backends.sparse.csa2.query_projection import (
            fp8_query_gemm,
        )

        if x_mxfp8 is None:
            _, data, scales = rms_norm_mxfp8(x, norm[0], norm[1], normalized=False)
        else:
            data, scales = x_mxfp8
        q = torch.empty(x.shape[0], weight.shape[0], dtype=torch.float8_e4m3fn, device=x.device)
        if not fp8_query_gemm(data, scales, weight, weight_sw, alpha, pos, cos_sin, q, amax):
            raise RuntimeError(
                "the FP8 query GEMM must be compiled by an eager step before a graph capture"
            )
        return q, (data, scales)

    def reference(
        self,
        x,
        norm,
        weight,
        weight_sw,
        alpha,
        pos,
        cos_sin,
        weight_scale=None,
        x_mxfp8=None,
        amax=None,
    ) -> torch.Tensor:
        """fp32, as `csa2_query_gemm_rope.reference` (the test compares the dequantized E4M3 query); `x_mxfp8` is the
        MXFP8 form of the same normalized `x`, so the reference reads `x`; `amax` is ignored (the test checks the word
        against the returned query)."""
        return csa2_query_gemm_rope.reference(
            x, norm, weight, weight_sw, alpha, pos, cos_sin, weight_scale
        )

    def is_valid(
        self,
        x,
        norm,
        weight,
        weight_sw,
        alpha,
        pos,
        cos_sin,
        weight_scale=None,
        x_mxfp8=None,
        amax=None,
    ) -> None:
        csa2_query_gemm_rope.is_valid(x, norm, weight, weight_sw, alpha, pos, cos_sin, weight_scale)
        if x_mxfp8 is not None:
            data, scales = x_mxfp8
            assert (
                data.dtype == torch.float8_e4m3fn and data.shape == x.shape and data.is_contiguous()
            )
            assert scales.is_contiguous(), (
                "x_mxfp8 scales are read as one contiguous swizzled buffer"
            )
            assert scales.dtype == torch.uint8 and scales.numel() == -(-x.shape[0] // 128) * 128 * (
                x.shape[1] // 32
            ), "x_mxfp8 scales are the CUTLASS-swizzled 128x4 UE8M0 tiles of rms_norm_mxfp8"
        if amax is not None:
            assert amax.dtype == torch.int32 and amax.numel() >= 1 and amax.device == x.device, (
                "amax: an int32 device word"
            )


csa2_query_gemm_fp8 = Csa2QueryGemmFp8()
