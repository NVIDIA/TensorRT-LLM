# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small-M BF16 projections on FlashInfer's SM10x direct or split-K CuTe DSL GEMM."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

# (N, K) -> ((largest M, kernel, tactic), ...) in ascending M; the first row whose M bound
# covers the call wins. Measured per M on B200 against cuBLAS (cold L2, graph replay): every
# row beat cuBLAS by 0.7 to 3.0 us. Shapes and M values outside the table take cuBLAS.
_CHOICES: Dict[Tuple[int, int], Tuple[Tuple[int, str, tuple], ...]] = {
    # KDA [f_a | g_a | b] (128 + 128 + 16 local heads), the gate path of every KDA layer.
    (272, 4096): (
        (1, "direct", (256, 2, 1)),
        (2, "direct", (256, 2, 2)),
        (4, "direct", (256, 1, 4)),
        (8, "direct", (64, 1, 8)),
        (12, "direct", (128, 2, 4)),
        (32, "splitk", (64, 8, 4, 11)),
    ),
    # KDA o_proj, row-parallel: 16 local heads x 128 -> hidden.
    (4096, 2048): (
        (2, "direct", (64, 2, 2)),
        (4, "splitk", (128, 8, 2, 6)),
        (8, "splitk", (64, 8, 1, 10)),
        (16, "splitk", (64, 16, 1, 9)),
        (32, "splitk", (64, 32, 1, 9)),
    ),
    # Indexer [k | compress gate | head weights] (128 + 128 + 32), replicated.
    (288, 4096): (
        (1, "direct", (256, 2, 1)),
        (2, "direct", (256, 2, 2)),
        (4, "direct", (128, 1, 4)),
        (8, "direct", (64, 1, 8)),
        (12, "direct", (128, 2, 4)),
        (32, "splitk", (64, 8, 4, 11)),
    ),
    # MLA [q_a | kv_a] (1536 + 512), replicated, on its dequantized BF16 copy.
    (2048, 4096): (
        (1, "direct", (256, 2, 1)),
        (2, "direct", (128, 2, 2)),
        (8, "splitk", (128, 8, 4, 6)),
        (16, "splitk", (64, 16, 2, 10)),
        (32, "splitk", (64, 32, 2, 9)),
    ),
    # MLA o_proj, row-parallel: 16 local heads x 256 -> hidden, dequantized BF16 copy.
    (4096, 4096): (
        (1, "direct", (256, 2, 1)),
        (8, "splitk", (64, 8, 2, 11)),
        (16, "splitk", (64, 16, 2, 11)),
        (32, "splitk", (64, 32, 2, 9)),
    ),
    # LM head, vocabulary-parallel: 154880 / 4 rows. The MTP draft heads run 1 row, verify 4 per request.
    (38720, 4096): (
        (1, "direct", (128, 1, 1)),
        (2, "direct", (128, 2, 2)),
        (8, "splitk", (128, 8, 2, 6)),
        (16, "splitk", (64, 16, 1, 11)),
    ),
    # Indexer wq_b (32 heads x 128) and MLA q_b (16 local heads x 256): both q_lora -> 4096.
    (4096, 1536): (
        (8, "splitk", (64, 8, 1, 12)),
        (16, "splitk", (64, 16, 1, 11)),
        (32, "splitk", (64, 32, 1, 9)),
    ),
}


def low_m_choice(m: int, n: int, k: int) -> Optional[Tuple[str, tuple]]:
    """``(kernel, tactic)`` for an ``[m, k] x [n, k]^T`` product, or None for cuBLAS.

    A direct tactic ``(block_size, outputs_per_block, rows_per_block)`` serves only an M
    that its row tile divides; otherwise the next row of the table is tried.
    """
    for bound, kernel, tactic in _CHOICES.get((n, k), ()):
        if m > bound:
            continue
        if kernel == "direct" and (tactic[2] > m or m % tactic[2] or n % tactic[1]):
            continue
        return kernel, tactic
    return None


class LowMBf16Gemm(OpWrapper):
    """``x @ weight.T`` for BF16 ``x [M, K]`` and ``weight [N, K]``, FP32 accumulation rounded once.

    Takes FlashInfer's direct (SIMT, register-prefetched weight) or split-K tensor-core
    kernel at the tactic ``low_m_choice`` names, and ``torch.nn.functional.linear``
    (cuBLAS) for any shape or M the table does not cover, so it is safe on every call.
    """

    ARCHS = frozenset({Arch.SM_100})

    CELLS: tuple[Cell, ...] = (
        Cell(why="KDA [f_a | g_a | b], MTP-3 verify of one request", spec=dict(m=4, n=272, k=4096)),
        Cell(why="KDA [f_a | g_a | b], verify of eight requests", spec=dict(m=32, n=272, k=4096)),
        Cell(why="KDA o_proj, one request", spec=dict(m=4, n=4096, k=2048)),
        Cell(why="KDA o_proj, eight requests", spec=dict(m=32, n=4096, k=2048)),
        Cell(why="indexer [k | gate | weights], MTP draft row", spec=dict(m=1, n=288, k=4096)),
        Cell(why="indexer [k | gate | weights], one request", spec=dict(m=4, n=288, k=4096)),
        Cell(why="indexer wq_b, one request", spec=dict(m=4, n=4096, k=1536)),
        Cell(why="indexer wq_b, eight requests", spec=dict(m=32, n=4096, k=1536)),
        Cell(why="MLA [q_a | kv_a], one request", spec=dict(m=4, n=2048, k=4096)),
        Cell(why="MLA o_proj, one request", spec=dict(m=4, n=4096, k=4096)),
        Cell(why="MLA o_proj, MTP draft row", spec=dict(m=1, n=4096, k=4096)),
        Cell(why="LM head shard, MTP draft row", spec=dict(m=1, n=38720, k=4096)),
        Cell(why="LM head shard, verify of one request", spec=dict(m=4, n=38720, k=4096)),
        Cell(
            why="LM head shard beyond the table (24 rows): cuBLAS", spec=dict(m=24, n=38720, k=4096)
        ),
        Cell(why="a shape outside the table: cuBLAS", spec=dict(m=4, n=6144, k=4096)),
    )

    note = """
    The first call of a (kernel, tactic, M) compiles the CuTe DSL kernel, which takes seconds;
    the engine's warmup makes those calls before CUDA-graph capture. The output is allocated per
    call, which inside a capture is the graph pool's stable address. PDL follows
    ``TRTLLM_ENABLE_PDL``.
    """

    def __call__(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # The kernels take DLPack views, which refuse tensors that require grad (parameters).
        weight = weight.detach()
        m, k = x.shape
        n = weight.shape[0]
        choice = low_m_choice(m, n, k) if x.is_contiguous() and weight.is_contiguous() else None
        if choice is None:
            return torch.nn.functional.linear(x, weight)
        kernel, tactic = choice
        out = torch.empty((m, n), dtype=x.dtype, device=x.device)
        if kernel == "direct":
            from flashinfer.gemm.kernels.dense_bf16_gemm_direct import (
                DirectTactic,
                run_direct_dense,
            )

            return run_direct_dense(x, weight.t(), out, get_env_enable_pdl(), DirectTactic(*tactic))
        from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import (
            SplitKTactic,
            run_splitk_dense,
        )

        return run_splitk_dense(
            x, weight.t(), None, out, get_env_enable_pdl(), SplitKTactic(*tactic)
        )

    def reference(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        with true_fp32_matmul():
            return (x.float() @ weight.float().t()).to(x.dtype)

    def is_valid(self, x: torch.Tensor, weight: torch.Tensor) -> None:
        # Non-contiguous operands fall back to cuBLAS rather than reach the kernels, which
        # assume row-major x and the stored [N, K] weight.
        assert x.dim() == 2 and weight.dim() == 2 and x.shape[1] == weight.shape[1]
        assert x.dtype == weight.dtype == torch.bfloat16


low_m_bf16_gemm = LowMBf16Gemm()
