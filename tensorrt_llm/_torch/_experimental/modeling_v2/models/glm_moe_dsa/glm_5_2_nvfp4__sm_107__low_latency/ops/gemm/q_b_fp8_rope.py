# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MLA q_b of a decode step as one split-K GEMM: k_b absorbed, the q_a RMSNorm folded, and the
RoPE'd E4M3 attention Q written by the epilogue."""

from __future__ import annotations

from typing import Literal

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

from ._low_m_splitk import operands_overlap
from ._low_m_splitk_fused import SplitKEpilogue, q_b_plan, run_splitk_fused

#: GLM-5.2 MLA at tp16: 4 heads per rank, kv_lora_rank 512 + qk_rope_head_dim 64 per head,
#: q_lora_rank 2048, RMSNorm eps 1e-5.
_GLM = dict(heads=4, head_dim=576, rope_dim=64, k=2048, eps=1e-5)

#: E4M3's largest finite magnitude.
_E4M3_MAX = 448.0


class GlmQBFp8Rope(OpWrapper):
    """The generation attention Q of a GLM MLA layer from the raw q_a, as E4M3 bits; returns `out`.

    `out[t, h * head_dim + c] = e4m3(q_scale * rope_t(rstd_t * (q_a @ weight.T))[t, h, c])`, where
    `rstd_t = rsqrt(mean(q_a[t] ** 2) + eps)` and `rope_t` rotates the last `rope_dim` columns of
    every `head_dim`-wide head as GPT-J pairs (2p, 2p + 1) by the (cos, sin) pair p of the token's
    position, `kv_lens[s] - seq_len + i` for token i of sequence s (`seq_len` tokens each). The
    conversion rounds to nearest even and saturates at +-448. Everything is FP32 from the GEMM's
    accumulator to the conversion; nothing is rounded to BF16 on the way.

    `weight` is q_b with k_b absorbed and the q_a norm weight folded in, built once at load (see
    the target's weights): for head h, rows [h * head_dim, h * head_dim + kv_lora_rank) are
    `k_b_trans[h] @ W_q_b[h, :qk_nope]` (`k_b_trans` the MLA's [heads, kv_lora_rank, qk_nope]
    k_b) and the next `rope_dim` rows are `W_q_b[h, qk_nope:]`, every column k times
    `gamma_q_a[k]`, computed in FP64 and rounded to BF16 once. The output is the
    `[tokens, heads, kv_lora_rank + rope_dim]` FP8 Q the sparse-MLA generation kernel reads.

    `plan` picks the launch, never the result's definition: `regular` (the shape's
    `splitk_default_tactic`), `beside_topk` (128-row tiles, no split-K, three stages, for the
    copy queued beside the DSA indexer top-k), `early_gather` (split-K 4 where the regular tactic
    is split-K 8, beside a flash decode that gathers before its wait). The plans differ in the
    FP32 summation order only.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="full-indexer layer, verify step and MTP draft step 0: 8 tokens of one request, "
            "beside the top-k (128-row tiles, no split-K, 18 CTAs)",
            spec=dict(_GLM, m=8, seq_len=8, plan="beside_topk"),
        ),
        Cell(
            why="shared-indexer layer, verify step: 8 tokens, early-gather split-K 4 (72 CTAs)",
            spec=dict(_GLM, m=8, seq_len=8, plan="early_gather"),
        ),
        Cell(
            why="MTP draft steps 1-6: 1 token, regular split-K 8 (144 CTAs)",
            spec=dict(_GLM, m=1, seq_len=1, plan="regular"),
        ),
        Cell(
            why="two requests of 8 tokens: per-sequence positions, 16-row tiles (regular plan)",
            spec=dict(_GLM, m=16, seq_len=8, plan="regular"),
        ),
        *(
            Cell(
                why=f"TP4 absorbed q_b: {m} rows, {plan}",
                spec=dict(_GLM, heads=16, m=m, seq_len=6 if m >= 6 else 1, plan=plan),
            )
            for m in (1, 3, 6, 12, 18)
            for plan in ("regular", "beside_topk", "early_gather")
        ),
    )

    note = """
    Kernel: the split-K GEMM's mainloop and DSMEM reduction with the owner CTA's epilogue staged
    in shared memory. The owner's epilogue warps compute the row RMS from global memory while the
    mainloop runs (after the grid dependency wait) and read kv_lens, the cos/sin rows and
    q_scale after the MMA. PDL follows `TRTLLM_ENABLE_PDL` (default on): before its wait the
    kernel reads only the weight.

    Loud, so not guarded (ValueError before the launch): M outside [1, 32], K not a multiple of
    128, N not a multiple of head_dim, M not a multiple of seq_len, too few kv_lens, an `out`
    that is not contiguous uint8, a weight that is not a contiguous [N, K], a q_a without unit
    inner stride and a 16-byte row pitch, a geometry the epilogue rejects. The first call of a
    (tactic, epilogue) compiles the kernel; make it before CUDA-graph capture.
    """

    def __call__(
        self,
        q_a: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        kv_lens: torch.Tensor,
        cos_sin: torch.Tensor,
        q_scale: torch.Tensor,
        eps: float,
        head_dim: int,
        rope_dim: int,
        seq_len: int,
        plan: Literal["regular", "beside_topk", "early_gather"],
    ) -> torch.Tensor:
        """Write `out` and return it.

        Args:
            q_a: [M, q_lora_rank] BF16, the raw q_a (before its RMSNorm); unit inner stride, a
                row pitch of a multiple of 8 elements (a row view of the fused-A output works).
            weight: [heads * head_dim, q_lora_rank] BF16, contiguous: absorbed and norm-folded.
            out: [M, heads * head_dim] uint8, contiguous: E4M3 bits.
            kv_lens: [sequences] int32 KV lengths including this step's tokens.
            cos_sin: [positions, rope_dim, 2] FP32 contiguous (cos, sin) pairs; position p's
                pair i rotates columns 2i and 2i + 1 of every head's RoPE slice (the first
                rope_dim / 2 pairs of a row are read).
            q_scale: [1] FP32, the factor applied before the E4M3 conversion.
            eps: The q_a RMSNorm epsilon.
            head_dim: kv_lora_rank + rope_dim.
            rope_dim: The RoPE width at the end of every head.
            seq_len: Tokens per sequence; M is sequences * seq_len.
            plan: The launch plan.
        """
        m, k = q_a.shape
        tactic = q_b_plan(plan, m, weight.shape[0], k, q_a.device.index)
        epilogue = SplitKEpilogue(
            row_rms=True,
            rms_k=k,
            rms_eps=float(eps),
            fp8_rope=True,
            head_dim=head_dim,
            rope_dim=rope_dim,
            seq_len=seq_len,
        )
        return run_splitk_fused(
            q_a,
            weight.t(),
            out,
            get_env_enable_pdl(),
            tactic,
            epilogue,
            kv_lens=kv_lens,
            cos_sin=cos_sin,
            q_scale=q_scale,
        )

    def reference(
        self,
        q_a: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        kv_lens: torch.Tensor,
        cos_sin: torch.Tensor,
        q_scale: torch.Tensor,
        eps: float,
        head_dim: int,
        rope_dim: int,
        seq_len: int,
        plan: Literal["regular", "beside_topk", "early_gather"],
    ) -> torch.Tensor:
        """The E4M3 bits the op writes, as uint8 [M, N]: FP32 throughout (TF32 off), one
        conversion at the end. `out` and `plan` are not read."""
        m = q_a.shape[0]
        x = q_a.float()
        rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        with true_fp32_matmul():
            q = (x @ weight.float().t()) * rstd
        q = q.view(m, -1, head_dim)
        nope = head_dim - rope_dim
        token = torch.arange(m, device=q.device)
        seq = token // seq_len
        position = kv_lens.long()[seq] - seq_len + (token - seq * seq_len)
        pairs = cos_sin.view(-1, rope_dim, 2)[position, : rope_dim // 2]
        cos = pairs[..., 0].unsqueeze(1)
        sin = pairs[..., 1].unsqueeze(1)
        pe = q[..., nope:].reshape(m, q.shape[1], rope_dim // 2, 2)
        even, odd = pe[..., 0], pe[..., 1]
        rotated = torch.stack((cos * even - sin * odd, cos * odd + sin * even), dim=-1)
        q = torch.cat((q[..., :nope], rotated.flatten(-2)), dim=-1) * q_scale.float()[0]
        q = q.clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
        return q.view(torch.uint8).reshape(m, -1)

    def is_valid(
        self,
        q_a: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        kv_lens: torch.Tensor,
        cos_sin: torch.Tensor,
        q_scale: torch.Tensor,
        eps: float,
        head_dim: int,
        rope_dim: int,
        seq_len: int,
        plan: Literal["regular", "beside_topk", "early_gather"],
    ) -> None:
        # The kernel indexes the flat table as (position * rope_dim + pair) * 2: a table of
        # another row width is read at the wrong rows without complaint.
        assert cos_sin.dim() == 3 and tuple(cos_sin.shape[1:]) == (rope_dim, 2), (
            f"cos_sin must be [positions, rope_dim={rope_dim}, 2], got {tuple(cos_sin.shape)}"
        )
        # A position outside the table reads past it (or before it): garbage, not an error.
        sequences = q_a.shape[0] // seq_len
        lens = kv_lens[:sequences].long()
        assert bool((lens >= seq_len).all()), "kv_lens must cover the step's own tokens"
        assert int(lens.max()) <= cos_sin.shape[0], (
            "the cos/sin table must hold every position up to max(kv_lens) - 1"
        )
        # The epilogue stores while other clusters still read q_a and the weight.
        for operand, name in ((q_a, "q_a"), (weight, "weight")):
            assert not operands_overlap(out, operand), f"out must not share memory with {name}"


q_b_fp8_rope = GlmQBFp8Rope()
