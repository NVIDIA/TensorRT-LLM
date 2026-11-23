# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""General matmul with optional fused bias via the trtllm cuBLASLt gemm op.

This target's copy of `catalog/gemm/cublas_mm`, certified at the projections
this checkpoint's engine calls: q_a 3072->1024, kv_a 3072->576, kv_b 512->14336
and 512->7168, the router gate 3072->512, the layer-0 dense MLP 3072->13824 and
13824->3072 and the shared-expert / latent down projection 1536->3072, at one
decode row, a 64-row block and a 5120-token prefill chunk.

`kv_b` is the one projection whose width follows attention TP: the
disaggregated deployment's context engine runs at attn_tp 2, where a sparse
layer's `heads_local` is 16 (`dims.py`), so the weight it passes is
512 -> 16 x 448 = 7168. The 14336 cells cover the whole checkpoint tensor at
attn_tp 1 and the windowed layer's 32 local heads at attn_tp 2.
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul


class CublasMm(OpWrapper):
    """`mat_a @ mat_b (+ bias)`, fp32-accumulated, one cublas_mm call.

    `mat_b` is read column-major, which is what a stored weight's `.t()` already
    is -- the targets pass `weight.t()` and no transpose kernel runs.

    `torch.ops.trtllm.cublas_mm` is a compiled-in C++ kernel
    (`cpp/tensorrt_llm/thop/cublasScaledMM.cpp`, `TORCH_LIBRARY_FRAGMENT(trtllm)`),
    so the path that runs is the wheel's binary, not a JIT compilation.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The first cell is the shared entry's own. The rest are this checkpoint's
    # call sites, re-derived from its geometry: rows span a decode step, a
    # 64-row block (the two shapes the checkpoint runs at that width) and a
    # 5120-token prefill chunk, the largest row count its engine passes.
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
    `output_buffer_kind` and `group` are the op's multi-stream and collective
    arguments. The covered targets leave them at their defaults, so neither is
    a cell axis and neither is certified here.

    Certified on sm_103, this target's only arch, with the cells above. The
    engine's largest prefill chunk is 5120 rows; larger row counts and the
    indexer's own small projections (n = 128, 2048, 16) are uncertified.
    `torch.ops.trtllm.cublas_mm` is a compiled-in C++ kernel, not a JIT path.

    The two 7168-wide cells are the disaggregated deployment's context engine:
    at attn_tp 2 a sparse layer holds 16 of its 32 heads (`dims.py`), so the
    `kv_b` weight it passes is 512 -> 7168. The 14336 cells cover the same
    engine's windowed layers (32 local heads) and attn_tp 1's sparse width.

    The lm_head cells are the engine's rank-local vocabulary at tp4, a shape
    this op is not called with: the shell's `LMHead` is built with this
    target's `lm_head_output_dtype` of float32, so `LMHead.apply_linear` takes
    `_matmul_with_output_dtype` (`torch.mm`), and `Hyv4FlashCore` leaves
    `use_custom_cublas_mm` unset (`_torch/models/modeling_utils.py` reads it
    when it builds the head). They certify the width, not a call site.
    Worst measured difference across the cells: 0.46 of the test's 4-ulp
    element band, on the latent down over a 5120-token chunk, and 2.8e-4
    relative RMS (3.6% of the 2-ulp band) on the layer-0 MLP down over the
    same. Each cell prints its own two numbers, so the certifying run's log
    carries them. A product over another weight lands at 146x the element
    band.
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
