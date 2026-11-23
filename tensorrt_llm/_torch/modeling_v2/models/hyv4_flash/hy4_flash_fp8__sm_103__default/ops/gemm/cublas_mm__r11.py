# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""General matmul with optional fused bias via the trtllm cuBLASLt gemm op, at this checkpoint's projections."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul


class CublasMm(OpWrapper):
    """`mat_a @ mat_b (+ bias)`, fp32-accumulated; `mat_b` is read column-major, as a stored weight's `.t()` is."""

    ARCHS = frozenset({Arch.SM_103})

    # The first cell is the shared entry's own; the rest are this checkpoint's call sites.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode-shaped: one token, the shape every projection sees per step",
            spec=dict(m=1, k=7168, n=1536, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="hy4 decode q_a: hidden 3072 -> q_lora 1024",
            spec=dict(m=1, k=3072, n=1024, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="hy4 64-row q_a: hidden 3072 -> q_lora 1024",
            spec=dict(m=64, k=3072, n=1024, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="hy4 prefill q_a over a 5120-token chunk: 3072 -> 1024",
            spec=dict(m=5120, k=3072, n=1024, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="hy4 decode kv_a with mqa rope: 3072 -> 512 + 64",
            spec=dict(m=1, k=3072, n=576, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="hy4 prefill kv_a over a 5120-token chunk: 3072 -> 576",
            spec=dict(m=5120, k=3072, n=576, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="hy4 decode kv_b on a bf16 layer: 512 -> 64 heads x 224",
            spec=dict(m=1, k=512, n=14336, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="hy4 prefill kv_b over a 5120-token chunk: 512 -> 14336",
            spec=dict(m=5120, k=512, n=14336, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="hy4 ctx-engine sparse-layer kv_b at attn_tp 2: 512 -> 16 x 448",
            spec=dict(m=1, k=512, n=7168, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="hy4 ctx-engine kv_b, a 5120-token chunk at attn_tp 2: 512 -> 7168",
            spec=dict(m=5120, k=512, n=7168, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="hy4 decode router gate: hidden 3072 -> 512 experts",
            spec=dict(m=1, k=3072, n=512, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="hy4 prefill router gate over a 5120-token chunk",
            spec=dict(m=5120, k=3072, n=512, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="hy4 layer-0 MLP gate_up: hidden 3072 -> 2 x 6912",
            spec=dict(m=1, k=3072, n=13824, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="hy4 layer-0 MLP gate_up over a 5120-token chunk",
            spec=dict(m=5120, k=3072, n=13824, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="hy4 layer-0 MLP down: 13824 -> hidden 3072",
            spec=dict(m=1, k=13824, n=3072, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="hy4 layer-0 MLP down over a 5120-token chunk",
            spec=dict(m=5120, k=13824, n=3072, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="hy4 latent/shared-expert down: 1536 -> hidden 3072",
            spec=dict(m=1, k=1536, n=3072, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="hy4 latent down over a 5120-token chunk: 1536 -> 3072",
            spec=dict(m=5120, k=1536, n=3072, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="hy4 lm_head at tp4: hidden 3072 -> 30208 vocab columns",
            spec=dict(m=1, k=3072, n=30208, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="hy4 lm_head, a 64-row block: 3072 -> 30208",
            spec=dict(m=64, k=3072, n=30208, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        Cell(
            why="hy4 lm_head over a 5120-token chunk: 3072 -> 30208",
            spec=dict(m=5120, k=3072, n=30208, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
    )

    note = """
    `output_buffer_kind` and `group` are uncertified (callers leave the defaults). The lm_head cells certify the
    width only: this target's float32 `LMHead` goes through `torch.mm`, not this op.
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
        """fp32-accumulated, TF32 off."""
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
        # Other layouts and a wrong bias are accepted by the op and give silently wrong results.
        assert mat_a.is_contiguous(), "mat_a must be dense row-major [M, K]"
        assert mat_b.stride(0) == 1 and mat_b.stride(1) == mat_b.shape[0], (
            "mat_b must be dense column-major [K, N] (e.g. weight.t())"
        )
        if bias is not None:
            out_dt = out_dtype if out_dtype is not None else mat_a.dtype
            assert mat_a.dtype != torch.float32, "bias is silently ignored for fp32 inputs"
            assert bias.dtype == out_dt, "bias dtype must equal the output dtype"
            assert bias.shape == (mat_b.shape[1],) and bias.is_contiguous(), (
                "bias must be a contiguous [N] tensor"
            )


cublas_mm = CublasMm()
