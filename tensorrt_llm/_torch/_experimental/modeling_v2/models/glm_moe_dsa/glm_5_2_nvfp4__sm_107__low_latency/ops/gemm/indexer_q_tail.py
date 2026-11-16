# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA indexer wq_b from the raw q_a as one split-K GEMM whose epilogue finishes the indexer Q:
RoPE, the per-head E4M3 scale, the E4M3 Q and the scaled head weights."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

from ._low_m_splitk import operands_overlap
from ._low_m_splitk_fused import IndexerQTail, SplitKEpilogue, idx_qtail_plan, run_splitk_idx_qtail

#: GLM-5.2's DSA indexer (replicated on every rank): 32 heads of 128, RoPE on the first 64
#: columns as GPT-J pairs, UE8M0 scales, weight scale 128 ** -0.5 * 32 ** -0.5; q_lora_rank 2048
#: and the q_a RMSNorm eps 1e-5.
_GLM = dict(
    n_heads=32,
    head_dim=128,
    rope_dim=64,
    interleave=True,
    ue8m0=True,
    weight_scale=128**-0.5 * 32**-0.5,
    k=2048,
    eps=1e-5,
)

#: The E4M3 ratio of the indexer prologue: max(amax, 1e-12) * fp32(1 / 448).
_AMAX_FLOOR = 1e-12
_INV_E4M3_MAX = 1.0 / 448.0


class GlmIndexerQTail(OpWrapper):
    """The DSA indexer's FP8 Q, Q scales and scaled head weights from the raw q_a; returns
    `(q_fp8, q_scale, weights_out)`.

    The GEMM is the indexer wq_b with the q_a RMSNorm folded into its weight (`weight` is
    `W_wq_b * gamma_q_a`, per column, built in FP64 and rounded to BF16 once at load): `y = bf16(
    rstd_t * (q_a @ weight.T))`, `rstd_t = rsqrt(mean(q_a[t] ** 2) + eps)`, rounded to BF16 as a
    BF16 store would. Then, per (token t, head h) of `y`, the indexer prologue's tail in its
    order: RoPE on the head's first `rope_dim` columns (GPT-J pairs with `interleave`, NeoX
    halves otherwise; cos/sin from the row of `positions[t]`) as `bf16(fma(x, cos, rn(+-partner *
    sin)))`; `scale = max(amax, 1e-12) * fp32(1 / 448)`, rounded up to a power of two with
    `ue8m0`; `q_fp8 = e4m3(q * (1 / scale))`; `q_scale = scale`; `weights_out = (weights_in *
    scale) * weight_scale`.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="full-indexer layer, verify step and MTP draft step 0: 8 tokens, 32 heads of 128; "
            "half-head tiles in (2, 2) clusters (128 CTAs)",
            spec=dict(_GLM, m=8),
        ),
        Cell(
            why="1 token, decode without MTP: one row in an 8-row tile (the same tactic)",
            spec=dict(_GLM, m=1),
        ),
    )

    note = """
    Kernel: `SplitKIndexerQTailGemmKernel`, the row-RMS fused split-K GEMM at the shape's
    `splitk_default_tactic` (64-row tiles, split-K 2 for 32 x 128 outputs from 2048) whose
    64-row tiles hold half a head; the two halves of a head run in one cluster whose owner CTAs
    exchange their row amax through DSMEM. The owner's epilogue warps compute the row RMS after
    the grid dependency wait, then read the positions, the cos/sin rows and weights_in while the
    mainloop runs. PDL follows `TRTLLM_ENABLE_PDL` (default on): before its wait the kernel reads
    only the weight.

    The values are those of the fused indexer prologue run on wq_b's BF16 output: the same
    operations in the same order. The interleaved RoPE and UE8M0 scales are GLM-5.2's; the NeoX
    form and plain FP32 scales are implemented but not cells.

    Loud, so not guarded (ValueError before the launch): M outside [1, 32], K not a multiple of
    128, N not n_heads * head_dim, a head_dim that is neither one nor two 64-row tiles, operands
    of the wrong dtype, shape or stride, a rope_dim the epilogue cannot split. The first call of
    a (tactic, Q tail) compiles the kernel; make it before CUDA-graph capture.
    """

    def __call__(
        self,
        q_a: torch.Tensor,
        weight: torch.Tensor,
        positions: torch.Tensor,
        cos_sin: torch.Tensor,
        weights_in: torch.Tensor,
        q_fp8: torch.Tensor,
        q_scale: torch.Tensor,
        weights_out: torch.Tensor,
        eps: float,
        n_heads: int,
        head_dim: int,
        rope_dim: int,
        interleave: bool,
        ue8m0: bool,
        weight_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Write `q_fp8`, `q_scale` and `weights_out` and return them.

        Args:
            q_a: [M, q_lora_rank] BF16, the raw q_a (before its RMSNorm); unit inner stride, a
                row pitch of a multiple of 8 elements (a row view of the fused-A output works).
            weight: [n_heads * head_dim, q_lora_rank] BF16, contiguous: wq_b, norm-folded.
            positions: [>= M] int32 RoPE positions, contiguous.
            cos_sin: [table positions, >= rope_dim] FP32 with unit inner stride: per position
                the rope_dim / 2 cosines, then the rope_dim / 2 sines.
            weights_in: [M, n_heads] BF16 head weights (weights_proj output) with unit inner
                stride (a column view of the fused-A output works).
            q_fp8: [M, n_heads, head_dim] float8_e4m3fn, contiguous (output).
            q_scale: [M, n_heads, 1] FP32, contiguous (output).
            weights_out: [M, n_heads] FP32, contiguous (output).
            eps: The q_a RMSNorm epsilon.
            n_heads, head_dim, rope_dim, interleave, ue8m0, weight_scale: The indexer's Q tail.
        """
        m, k = q_a.shape
        qtail = IndexerQTail(
            n_heads=n_heads,
            head_dim=head_dim,
            rope_dim=rope_dim,
            interleave=interleave,
            ue8m0=ue8m0,
            weight_scale=float(weight_scale),
        )
        tactic = idx_qtail_plan(qtail, m, weight.shape[0], k, q_a.device.index)
        run_splitk_idx_qtail(
            q_a,
            weight.t(),
            q_fp8.view(torch.uint8).view(m, -1),
            get_env_enable_pdl(),
            tactic,
            SplitKEpilogue(row_rms=True, rms_k=k, rms_eps=float(eps)),
            qtail,
            positions=positions,
            cos_sin=cos_sin,
            weights_in=weights_in,
            q_scale_out=q_scale.view(m, n_heads),
            weights_out=weights_out,
        )
        return q_fp8, q_scale, weights_out

    def reference(
        self,
        q_a: torch.Tensor,
        weight: torch.Tensor,
        positions: torch.Tensor,
        cos_sin: torch.Tensor,
        weights_in: torch.Tensor,
        q_fp8: torch.Tensor,
        q_scale: torch.Tensor,
        weights_out: torch.Tensor,
        eps: float,
        n_heads: int,
        head_dim: int,
        rope_dim: int,
        interleave: bool,
        ue8m0: bool,
        weight_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(q_fp8, q_scale, weights_out)` as new tensors. The GEMM in FP32 (TF32 off); every
        rounding the prologue applies, applied here; each RoPE fma in FP64, rounded to FP32
        once. The output arguments are not read."""
        m = q_a.shape[0]
        x = q_a.float()
        rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        with true_fp32_matmul():
            y = ((x @ weight.float().t()) * rstd).to(torch.bfloat16).float()
        y = y.view(m, n_heads, head_dim)
        half = rope_dim // 2
        rows = cos_sin.reshape(cos_sin.shape[0], -1)[positions[:m].long(), :rope_dim]
        cos = rows[:, :half].unsqueeze(1)
        sin = rows[:, half:rope_dim].unsqueeze(1)
        pe = y[..., :rope_dim]
        if interleave:
            first, second = pe[..., 0::2], pe[..., 1::2]
        else:
            first, second = pe[..., :half], pe[..., half:]

        def fma(a, b, c):
            return (a.double() * b.double() + c.double()).float()

        rot_first = fma(first, cos, -second * sin).to(torch.bfloat16).float()
        rot_second = fma(second, cos, first * sin).to(torch.bfloat16).float()
        if interleave:
            pe = torch.stack((rot_first, rot_second), dim=-1).flatten(-2)
        else:
            pe = torch.cat((rot_first, rot_second), dim=-1)
        q = torch.cat((pe, y[..., rope_dim:]), dim=-1)
        ratio = q.abs().amax(dim=-1, keepdim=True).clamp_min(_AMAX_FLOOR) * torch.tensor(
            _INV_E4M3_MAX, dtype=torch.float32
        )
        scale = _ue8m0_round_up(ratio) if ue8m0 else ratio
        q_ref = (q * (1.0 / scale)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        weights_ref = (weights_in.float() * scale.squeeze(-1)) * torch.tensor(
            weight_scale, dtype=torch.float32
        )
        return q_ref, scale, weights_ref

    def is_valid(
        self,
        q_a: torch.Tensor,
        weight: torch.Tensor,
        positions: torch.Tensor,
        cos_sin: torch.Tensor,
        weights_in: torch.Tensor,
        q_fp8: torch.Tensor,
        q_scale: torch.Tensor,
        weights_out: torch.Tensor,
        eps: float,
        n_heads: int,
        head_dim: int,
        rope_dim: int,
        interleave: bool,
        ue8m0: bool,
        weight_scale: float,
    ) -> None:
        # A position outside the cos/sin table reads past it: garbage, not an error.
        used = positions[: q_a.shape[0]].long()
        assert bool((used >= 0).all()) and int(used.max()) < cos_sin.shape[0], (
            "every position must index a row of the cos/sin table"
        )
        # The epilogue stores while other clusters still read q_a, the weight and weights_in.
        outputs = {"q_fp8": q_fp8, "q_scale": q_scale, "weights_out": weights_out}
        operands = {"q_a": q_a, "weight": weight, "weights_in": weights_in}
        for output_name, output in outputs.items():
            for name, operand in operands.items():
                assert not operands_overlap(output, operand), (
                    f"{output_name} must not share memory with {name}"
                )


def _ue8m0_round_up(ratio: torch.Tensor) -> torch.Tensor:
    """The positive FP32 `ratio` with its mantissa cleared and, if it was nonzero, its exponent
    raised by one."""
    bits = ratio.view(torch.int32)
    exponent = bits & 0x7F800000
    exponent = exponent + ((bits & 0x007FFFFF) != 0).to(torch.int32) * 0x00800000
    return exponent.view(torch.float32)


indexer_q_tail = GlmIndexerQTail()
