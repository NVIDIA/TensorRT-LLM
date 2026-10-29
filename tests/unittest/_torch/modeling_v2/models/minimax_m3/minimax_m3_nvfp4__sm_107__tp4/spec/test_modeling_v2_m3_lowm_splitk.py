# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the target's copy of FlashInfer's split-K low-M GEMM (ops/m3_lowm_splitk,
ops/m3_draft_lowm): bit for bit FlashInfer's kernel at every certified split-K cell of the drafter's
linears, eager and in a CUDA graph, with and without PDL."""

import pytest
import torch
from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import SplitKTactic as FiTactic
from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import run_splitk_dense as fi_run

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4.ops.gemm.m3_draft_lowm import (  # noqa: E501
    ROWS,
    tactic_for,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4.ops.m3_lowm_splitk import (  # noqa: E501
    SplitKTactic,
    run_splitk_dense,
)

assert torch.cuda.is_available(), "m3_lowm_splitk requires a CUDA device"

# Every certified (N, K, rows) of the TP4 drafter table that a split-K tactic serves: (N, K, rows, tactic).
CELLS = [
    (nk[0], nk[1], m, tactic_for(nk[0], nk[1], m)[1])
    for nk, rows in sorted(ROWS.items())
    for m in rows
    if tactic_for(nk[0], nk[1], m)[0] == "s"
]


def test_the_table_has_the_certified_split_k_cells() -> None:
    assert sorted((n, k, m) for n, k, m, _ in CELLS) == [
        (2304, 12288, 4),
        (6144, 2048, 4),
        (6144, 4608, 1),
        (6144, 4608, 4),
        (9216, 6144, 4),
    ]


@pytest.mark.parametrize("use_pdl", [False, True], ids=["plain", "pdl"])
@pytest.mark.parametrize("cell", CELLS, ids=[f"{n}x{k}@{m}" for n, k, m, _ in CELLS])
def test_bitwise_flashinfer(cell, use_pdl) -> None:
    n, k, m, tactic = cell
    g = torch.Generator(device="cuda").manual_seed(n + 7 * k + m)
    for trial in range(3):
        x = (torch.randn(m, k, generator=g, device="cuda") * (1.0 + trial)).to(torch.bfloat16)
        w = (torch.randn(n, k, generator=g, device="cuda") / k**0.5).to(torch.bfloat16)
        ref = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
        got = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
        fi_run(x, w.t(), None, ref, use_pdl, FiTactic(*tactic))
        run_splitk_dense(x, w.t(), None, got, use_pdl, SplitKTactic(*tactic))
        torch.cuda.synchronize()
        assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), f"trial {trial}"
    # In a CUDA graph, as the drafter runs it.
    cap = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_splitk_dense(x, w.t(), None, cap, use_pdl, SplitKTactic(*tactic))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(cap.view(torch.int16), ref.view(torch.int16))
