# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM107 narrow BF16 projections: fixed direct-kernel tactics, the cublas_mm entry otherwise."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import CublasMm

# (M, N, K) -> DirectTactic(block_size, outputs_per_block, rows_per_block). M is the decode
# rows of a captured batch: 3 per sequence for the draft's kv_a, 4 per sequence for the
# target's kv_a and indexer projections. Every CTA re-reads all of A, so the direct kernel's
# cost grows with M x N while cuBLAS's stays nearly flat; 16x512, 16x576 and larger M stay on
# cuBLAS.
_LOW_M_DIRECT_TACTICS: dict[tuple[int, int, int], tuple[int, int, int]] = {
    (3, 576, 6144): (128, 1, 3),
    (4, 160, 6144): (128, 2, 4),
    (4, 512, 6144): (128, 1, 4),
    (4, 576, 6144): (128, 1, 4),
    (6, 576, 6144): (128, 2, 6),
    (8, 160, 6144): (128, 2, 8),
    (8, 512, 6144): (128, 2, 8),
    (8, 576, 6144): (128, 2, 8),
    (12, 576, 6144): (64, 2, 4),
    (16, 160, 6144): (128, 2, 8),
}


def _low_m_direct_tactic(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype | None,
    output_buffer_kind: int,
    group: list[int] | None,
) -> tuple[int, int, int] | None:
    if (
        bias is not None
        or output_buffer_kind != 0
        or group is not None
        or mat_a.dtype != torch.bfloat16
        or mat_b.dtype != torch.bfloat16
        or out_dtype not in (None, torch.bfloat16)
        or mat_a.ndim != 2
        or mat_b.ndim != 2
        or not mat_a.is_cuda
        or mat_a.device != mat_b.device
        or not mat_a.is_contiguous()
        or mat_b.stride() != (1, mat_b.shape[0])
        or mat_a.shape[1] != mat_b.shape[0]
        or mat_a.data_ptr() % 32 != 0
        or mat_b.data_ptr() % 32 != 0
    ):
        return None
    tactic = _LOW_M_DIRECT_TACTICS.get((mat_a.shape[0], mat_b.shape[1], mat_a.shape[1]))
    if tactic is None or torch.cuda.get_device_capability(mat_a.device) != (10, 7):
        return None
    return tactic


class LowMCublasMm(CublasMm):
    """SM107 narrow BF16 projections, with the cuBLAS contract for other calls.

    Fixed direct-kernel tactics accumulate in FP32 and round once to BF16.
    FP32 outputs, fused bias, collective buffers, operands below 32-byte alignment, other layouts/shapes, and other
    architectures use the original cuBLAS op. The direct kernel uses ordinary
    stream ordering, like the cuBLAS path it replaces.
    """

    ARCHS = frozenset({Arch.SM_107})
    note = """
    Exact finite BF16 direct tactics at the cells below. Other calls use
    CublasMm, including FP32 router logits and collective output buffers.
    There is no mutable tactic cache: dtype, layout, output role, architecture,
    and exact matrix dimensions gate the fixed table on every call.
    """
    CELLS = tuple(
        Cell(
            why=f"SM107 low-M direct projection {m}x{k}x{n}",
            spec=dict(m=m, k=k, n=n, dtype=torch.bfloat16, bias=False, out_dtype=None),
        )
        for m, n, k in _LOW_M_DIRECT_TACTICS
    )

    def __call__(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        output_buffer_kind: int = 0,
        group: list[int] | None = None,
    ) -> torch.Tensor:
        tactic = _low_m_direct_tactic(mat_a, mat_b, bias, out_dtype, output_buffer_kind, group)
        if tactic is None:
            return torch.ops.trtllm.cublas_mm(
                mat_a, mat_b, bias, out_dtype, output_buffer_kind, group
            )
        from flashinfer.gemm.kernels.dense_bf16_gemm_direct import DirectTactic, run_direct_dense

        out = mat_a.new_empty((mat_a.shape[0], mat_b.shape[1]))
        return run_direct_dense(mat_a, mat_b, out, False, DirectTactic(*tactic))


low_m_cublas_mm = LowMCublasMm()
