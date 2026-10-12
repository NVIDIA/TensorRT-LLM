# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""General matmul with optional fused bias via the trtllm cuBLASLt gemm op."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import ULP, Arch, Cell, OpWrapper, assert_within_ulp, true_fp32_matmul

# The DeepSeek-V4.1 target's calls: (role, K, N, dtype, out_dtype, bias, weight std, rows, what).
# Every call passes a contiguous [M, K] activation and the `.t()` of a contiguous [N, K] weight.
# `rows` are the ends of the every-M sweep the shared test runs at that call, plus the edges of
# the patch embed's refused band. Weight std None is unit-normal (the core calls); the vision
# calls take N(0, 1/K) weights so their outputs are O(1) and a unit-normal bias is not lost.
_DSV41_CALLS = (
    ("compressor-r2", 5120, 1024, torch.float32, None, False, None, (1, 16384),
     "fp32 [M, 5120] -> [M, 1024], [kv || score] of a ratio-2 KV source"),
    ("router-fp32", 5120, 384, torch.float32, None, False, None, (1, 8192),
     "fp32 [M, 5120] -> [M, 384], the router gate on fp32 inputs"),
    ("dspark-gate", 5120, 128, torch.float32, None, False, None, (1, 8192),
     "fp32 [M, 5120] -> [M, 128], DSpark drafter MoE gate, M = 5G"),
    ("markov-head", 256, 129280, torch.float32, None, False, None, (1, 8192),
     "fp32 [M, 256] -> [M, 129280], DSpark Markov head, M = G"),
    ("compressor-r1", 5120, 512, torch.bfloat16, None, False, None, (1, 16384),
     "bf16 [M, 5120] -> [M, 512], compressor of a ratio-1 KV source"),
    ("index-k", 512, 128, torch.bfloat16, None, False, None, (1, 16384),
     "bf16 [M, 512] -> [M, 128], index-K of the compressed latent"),
    ("weights-proj", 5120, 32, torch.bfloat16, None, False, None, (1, 32768),
     "bf16 [M, 5120] -> [M, 32], indexer weights_proj"),
    ("router", 5120, 384, torch.bfloat16, torch.float32, False, None, (1, 8192),
     "bf16 [M, 5120] -> fp32 [M, 384], the router gate's logits"),
    ("patch-embed", 588, 1024, torch.bfloat16, None, True, 588**-0.5, (1, 2304, 2433, 17037),
     "bf16 [P, 588] -> [P, 1024] + bias; P 2305..2432 is refused, pad to >= 2433"),
    ("vit-wqkv", 1024, 3072, torch.bfloat16, None, True, 1024**-0.5, (1, 17037),
     "bf16 [P, 1024] -> [P, 3072] + bias, vision block wqkv"),
    ("vit-wo", 1024, 1024, torch.bfloat16, None, True, 1024**-0.5, (1, 17037),
     "bf16 [P, 1024] -> [P, 1024] + bias, vision block wo"),
    ("vit-w1", 1024, 5632, torch.bfloat16, None, False, 1024**-0.5, (1, 17037),
     "bf16 [P, 1024] -> [P, 5632], vision block w1 (gate | up)"),
    ("vit-w2", 2816, 1024, torch.bfloat16, None, False, 2816**-0.5, (1, 17037),
     "bf16 [P, 2816] -> [P, 1024], vision block w2"),
    ("aligner-w1", 9216, 5120, torch.bfloat16, None, True, 9216**-0.5, (1, 2048),
     "bf16 [C, 9216] -> [C, 5120] + bias, aligner w1"),
    ("aligner-w2", 5120, 5120, torch.bfloat16, None, True, 5120**-0.5, (1, 2048),
     "bf16 [C, 5120] -> [C, 5120] + bias, aligner w2"),
)  # fmt: skip

_DSV41_CELLS = tuple(
    Cell(
        why=f"deepseek-v4.1 {role} M={m}: {what}; every M {rows[0]}..{rows[-1]} swept",
        spec=dict(m=m, k=k, n=n, dtype=dt, bias=bias, out_dtype=out_dt, w_std=w_std),
    )
    for role, k, n, dt, out_dt, bias, w_std, rows, what in _DSV41_CALLS
    for m in rows
)

# The M that `is_valid` refuses at bf16 [M, 588] x [588, 1024] -> bf16 (DeepSeek-V4.1 patch embed).
_PATCH_EMBED_SPLIT_K_ROWS = range(2305, 2433)


class CublasMm(OpWrapper):
    """`mat_a @ mat_b (+ bias)`, fp32-accumulated, one cublas_mm call.

    `mat_b` is read column-major, which is what a stored weight's `.t()` already
    is -- the targets pass `weight.t()` and no transpose kernel runs.
    """

    ARCHS = frozenset({Arch.SM_103})

    # Both targets use this for every dense projection, so the cells are the
    # three argument shapes that differ in kind rather than in size: fused bias
    # (gpt-oss carries one on qkv, o and the router), no bias (deepseek's
    # projections), and an out_dtype that is not the input's (deepseek's router
    # asks for fp32 logits). DeepSeek-V4.1's calls follow (`_DSV41_CALLS`): one
    # cell per end of each call's swept row range, plus the edges of the patch
    # embed's refused band.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="gpt-oss qkv with fused bias: hidden 2880 -> (64 + 8 + 8) * 64",
            spec=dict(m=16, k=2880, n=5120, dtype=torch.bfloat16, bias=True, out_dtype=None),
        ),
        Cell(
            why="deepseek q_a, no bias: hidden 7168 -> q_lora 1536",
            spec=dict(m=16, k=7168, n=1536, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="deepseek router: bf16 in, fp32 logits out, 256 experts",
            spec=dict(
                m=16, k=7168, n=256, dtype=torch.bfloat16, bias=False, out_dtype=torch.float32
            ),
        ),
        Cell(
            why="decode-shaped: one token, the shape every projection sees per step",
            spec=dict(m=1, k=7168, n=1536, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
    ) + _DSV41_CELLS

    note = """
    `output_buffer_kind` and `group` are the op's multi-stream and collective
    arguments. Both shipped targets leave them at their defaults, so neither is
    a cell axis and neither is certified here.

    DeepSeek-V4.1: every call passes bias=None and out_dtype=None except the
    router (bf16 in, fp32 logits) and the vision calls (their checkpoint bias,
    bf16). No metadata. cuBLAS picks its kernel from M with no algorithm
    pinned, and on sm_103 the kernel changes every 16 to 64 rows, so the shared
    test sweeps every M from 1 to each call's largest: 16384 at both
    compressors and index-K, 32768 at weights_proj, 8192 at the router (the
    target splits the routed MoE's rows into chunks of at most 8192) and the
    two DSpark calls, 17037 at the vision tower and 2048 at the aligner. The
    fp32 calls are IEEE fp32, not TF32. No core call takes a split-K kernel
    with bf16 partial sums at any swept M, and at every M at least 0.99 of a
    bf16 call's elements are bit-identical to the rounded-once reference.
    Hazard: bf16 K = 588, N = 1024 (the patch embed) at M 2305..2432, with or
    without bias, takes such a kernel and leaves the bf16 gate on 1.4% or more
    of its elements; `is_valid` refuses it, and the first P rows of a call
    padded with zero rows to 2433 are in the gate. CUDA-graph replays are
    bitwise equal to eager at the row counts the target captures (every batch
    G in 1..32, 40..128 step 8, 192, 256, times 1, 3 and 6, and also 4 and 24
    at the router; 5G at the DSpark gate, G at the Markov head) and at the
    vision tower's P 1517, 4680, 5325, 8772, 17037 and aligner C 182, 546,
    601, 986, 1902. Rows of one call are not bitwise those of a call with
    more rows except at index-K. Not certified for DeepSeek-V4.1: M above the
    swept ranges, capture at other row counts or of several calls sharing a
    graph pool, fp8 or fp16 inputs, and the patch embed band itself.
    """

    def raw_call(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        output_buffer_kind: int = 0,
        group: list[int] | None = None,
    ) -> torch.Tensor:
        return torch.ops.trtllm.cublas_mm(mat_a, mat_b, bias, out_dtype, output_buffer_kind, group)

    def reference(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        output_buffer_kind: int = 0,
        group: list[int] | None = None,
    ) -> torch.Tensor:
        """fp32-accumulated, TF32 off -- see `true_fp32_matmul`.

        fp32 operands are the exception: their reference is the float64
        product, left in float64. The op is IEEE fp32 there, so an fp32
        reference would be the op's own arithmetic in another order, and the
        float64 dtype is what tells `compare` to use the fp32 gate.
        `output_buffer_kind` and `group` are ignored.
        """
        if mat_a.dtype == torch.float32:
            ref = mat_a.double() @ mat_b.double()
            return ref + bias.double() if bias is not None else ref
        with true_fp32_matmul():
            ref = mat_a.float() @ mat_b.float()
        if bias is not None:
            ref = ref + bias.float()
        return ref.to(out_dtype if out_dtype is not None else mat_a.dtype)

    def is_valid(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        output_buffer_kind: int = 0,
        group: list[int] | None = None,
    ) -> None:
        # The kernel reads mat_a as dense row-major and mat_b as dense
        # column-major; other layouts produce silently wrong results.
        assert mat_a.is_contiguous(), "mat_a must be dense row-major [M, K]"
        assert mat_b.stride(0) == 1 and mat_b.stride(1) == mat_b.shape[0], (
            "mat_b must be dense column-major [K, N] (e.g. weight.t())"
        )
        if bias is not None:
            out_dt = out_dtype if out_dtype is not None else mat_a.dtype
            # A bias in the wrong dtype or shape is accepted by the op and
            # produces silently wrong results; with fp32 inputs the bias is
            # accepted but silently ignored.
            assert mat_a.dtype != torch.float32, "bias is silently ignored for fp32 inputs"
            assert bias.dtype == out_dt, "bias dtype must equal the output dtype"
            assert bias.shape == (mat_b.shape[1],) and bias.is_contiguous(), (
                "bias must be a contiguous [N] tensor"
            )
        # Measured on sm_103 at DeepSeek-V4.1's vision patch embed: bf16
        # [M, 588] x [588, 1024] -> bf16 at M 2305..2432, with or without bias,
        # is the one place cuBLASLt's heuristic picks a split-K kernel whose
        # partial sums are stored in bf16. Every partial is rounded to bf16
        # before the sum: at most 0.685 of the elements are bit-identical to
        # the rounded-once product (0.9996 or more at every other M) and 1.4% or
        # more leave the bf16 gate. The first M rows of a call padded with zero
        # rows to 2433 are in the gate.
        assert not (
            mat_a.dtype == torch.bfloat16
            and out_dtype in (None, torch.bfloat16)
            and mat_a.dim() == 2
            and tuple(mat_b.shape) == (588, 1024)
            and mat_a.shape[0] in _PATCH_EMBED_SPLIT_K_ROWS
        ), "bf16 K=588 N=1024 at M 2305..2432 takes a bf16-partial split-K kernel; pad M to >= 2433"

    def compare(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        """One GEMM's worth of accumulation-order difference.

        Kernel and reference consume bit-identical operands and differ only in
        the order cuBLAS and torch sum over K. Over random operands a few
        outputs land near zero, where that reordering shows up as a relative
        difference of 10 on an absolute difference of 1.5e-4 -- which is what
        torch's default band rejects, and what no bf16 result could represent
        anyway: one ulp at these rows' scale is about 6e-1.

        So: 4 ulp element-wise against the row scale, 2 ulp of relative RMS.
        Measured worst case across this entry's cells is 1.5e-4 absolute, or
        roughly 3e-4 ulp -- four orders of margin, and a band that a genuinely
        wrong product would miss by orders rather than squeak past.
        DeepSeek-V4.1's bf16 cells, its bf16 -> fp32 router included, take this
        gate unchanged; their every-M sweeps in the shared test hold the
        tighter checks measured there.

        fp32 operands are scored differently, recognised by the float64
        reference (see `reference`): the largest |out - float64 product| must
        be at most 512 fp32 ulp (2^-14) of the output's RMS. Measured on sm_103
        with unit-variance operands, the op sat at most 1.56e-3 from float64 at
        K = 5120 over every M to 16384, and 1.05e-4 at K = 256. The output RMS
        is sqrt(K) there, so the gate reads 4.4e-3 and 9.8e-4: the absolute
        per-K gates those sweeps held (5e-3 and 1e-3), put in the one unit this
        method can see and rounded down. A TF32 product of the same operands
        lands 0.09 to 0.11 (K = 5120) and 0.022 to 0.027 (K = 256) away, more
        than 20x outside. The bf16 scoring cannot see TF32 at all: 0.1 on a row
        scale near 250 is a tenth of a bf16 ulp.
        """
        if ref.dtype == torch.float64:
            assert out.dtype == torch.float32, (out.dtype, ref.dtype)
            assert out.shape == ref.shape, (out.shape, ref.shape)
            err = (out.double() - ref).abs().max().item()
            rms = ref.pow(2).mean().sqrt().item()
            gate = 512 * ULP[torch.float32] * rms
            assert err <= gate, f"max |out - float64| {err:.3e} exceeds {gate:.3e} (2^-14 of RMS)"
            return
        # Scored in bf16 ulp even when the output is fp32: every cell's
        # operands are bf16, so the fp32 out_dtype the router asks for is a
        # wider container for a bf16-limited product, not a more precise one.
        # Against fp32's own ulp the same 2.4e-6 difference reads as 5x over.
        assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0, ulp=ULP[torch.bfloat16])


cublas_mm = CublasMm()
