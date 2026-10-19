# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""General matmul with optional fused bias via the trtllm cuBLASLt gemm op."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper, true_fp32_matmul


class CublasMm(OpWrapper):
    """`mat_a @ mat_b (+ bias)`, fp32-accumulated, one cublas_mm call.

    `mat_b` is read column-major, which is what a stored weight's `.t()` already
    is -- the targets pass `weight.t()` and no transpose kernel runs.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103, Arch.SM_107})

    # The targets use this for every bf16 dense projection, so the first cells
    # are the three argument shapes that differ in kind rather than in size: fused bias
    # (gpt-oss carries one on qkv, o and the router), no bias (deepseek's
    # projections), and an out_dtype that is not the input's (deepseek's router
    # asks for fp32 logits).
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
        # nemotron_h's Ultra at tep4: every dense projection of a rank, at the three
        # row counts that pick different cuBLAS kernels -- one decode token, the
        # 32-row CUDA-graph ceiling, and an 8192-token prefill step.
        *(
            Cell(
                why=f"nemotron_h {what}, T={m}{tail}",
                spec=dict(m=m, k=k, n=n, dtype=torch.bfloat16, bias=False, out_dtype=None),
            )
            for what, k, n, first in (
                ("qkv", 8192, 2304, ": hidden 8192 -> (16 + 1 + 1) x 128 = 2304"),
                ("o_proj / fc2_latent", 2048, 8192, ": 2048 -> 8192, a row-parallel partial"),
                ("fc1_latent", 8192, 2048, ": hidden 8192 -> latent 2048"),
            )
            for m, tail in (
                (1, first),
                (32, ", the decode CUDA-graph ceiling"),
                (8192, ", a max_num_tokens prefill step"),
            )
        ),
        *(
            Cell(
                why=f"nemotron_h router, T={m}{tail}",
                spec=dict(
                    m=m, k=8192, n=512, dtype=torch.bfloat16, bias=False, out_dtype=torch.float32
                ),
            )
            for m, tail in (
                (1, ": bf16-exact weight, fp32 logits for 512 experts"),
                (32, ", the decode CUDA-graph ceiling"),
                (8192, ", a max_num_tokens prefill step"),
            )
        ),
        # nemotron_h's Ultra BF16 at tep8: every dense projection a rank runs that is
        # not already a cell above -- the mamba in_proj (z | x | B | C | dt rows of one
        # rank), the 8-q-head qkv and o_proj, and the two halves of the shared expert --
        # at the same three row counts. fc1_latent, fc2_latent / the mamba out_proj and
        # the router are the cells above: their shapes do not depend on the rank count.
        *(
            Cell(
                why=f"nemotron_h BF16 tep8 {what}, T={m}{tail}",
                spec=dict(m=m, k=k, n=n, dtype=torch.bfloat16, bias=False, out_dtype=None),
            )
            for what, k, n, first in (
                ("mamba in_proj", 8192, 4384, ": hidden 8192 -> 2 x 2048 + 2 x 128 + 32 = 4384"),
                ("qkv", 8192, 1280, ": hidden 8192 -> (8 + 1 + 1) x 128 = 1280"),
                ("o_proj", 1024, 8192, ": 8 x 128 = 1024 -> hidden 8192, a row-parallel partial"),
                ("shared up", 8192, 1280, ": hidden 8192 -> 10240 / 8 = 1280"),
                ("shared down", 1280, 8192, ": 1280 -> hidden 8192, a row-parallel partial"),
            )
            for m, tail in (
                (1, first),
                (32, ", the decode CUDA-graph ceiling"),
                (8192, ", a max_num_tokens prefill step"),
            )
        ),
        # nemotron_h's Ultra at dep4: attention is replicated, so a rank runs the
        # unsharded qkv and o_proj over its own tokens.
        *(
            Cell(
                why=f"nemotron_h dep4 {what}, T={m}{tail}",
                spec=dict(m=m, k=8192, n=n, dtype=torch.bfloat16, bias=False, out_dtype=None),
            )
            for what, n, first in (
                ("qkv", 8704, ": hidden 8192 -> (64 + 2 + 2) x 128 = 8704"),
                ("o_proj", 8192, ": 64 x 128 = 8192 -> hidden 8192, unsharded"),
            )
            for m, tail in (
                (1, first),
                (32, ", the decode CUDA-graph ceiling"),
                (8192, ", a max_num_tokens prefill step"),
            )
        ),
    )

    note = """
    `output_buffer_kind` and `group` are the op's multi-stream and collective
    arguments. The shipped targets leave them at their defaults, so neither is
    a cell axis and neither is certified here.

    `out_dtype=torch.float32` over bf16 operands is a genuinely fp32 result,
    not a bf16 one in a wider container: a product of two bf16 values has at
    most 16 significant bits and is exact in fp32, so the only error left is
    fp32 accumulation. Measured on sm_107 against the exact product at K 8192:
    5-12 fp32 ulp RMS up to 32 rows, 76 at 8192 rows (the large-row tensor-core
    kernels accumulate with more error), against about 14,000 for the same
    logits rounded to bf16. That is what a router reading its logits against a
    bias near 32 relies on. The router weight nemotron_h stores in fp32 is
    bf16-exact, so it is passed as bf16 at no cost. For the same reason TF32
    would not truncate these operands either: bf16 values are exact in TF32's
    10-bit mantissa. TF32 only bites fp32 operands that are not already
    bf16-exact.
    """

    def __call__(
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
        """fp32-accumulated, TF32 off -- see `true_fp32_matmul`."""
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


cublas_mm = CublasMm()
