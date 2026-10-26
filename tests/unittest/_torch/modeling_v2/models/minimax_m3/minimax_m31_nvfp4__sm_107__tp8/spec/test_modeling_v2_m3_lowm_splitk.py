# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the target's copy of FlashInfer's split-K low-M GEMM (kernels/m3_lowm_splitk,
modeling.DRAFT_LOWM_OWN_SPLITK): bit for bit FlashInfer's kernel at every certified split-K cell of the drafter's
linears, eager and in a CUDA graph, with and without PDL."""

import pytest
import torch
from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import SplitKTactic as FiTactic
from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import run_splitk_dense as fi_run

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.kernels.m3_lowm_splitk import (  # noqa: E501
    SplitKTactic,
    run_splitk_dense,
    run_splitk_dense_swiglu,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.ops.gemm.m3_draft_lowm import (  # noqa: E501
    TACTICS,
)

assert torch.cuda.is_available(), "m3_lowm_splitk requires a CUDA device"

# Every split-K cell of the TP8 drafter table: (N, K, max rows, tactic) for kind "s".
CELLS = [
    (nk[0], nk[1], rows, tactic)
    for nk, entries in sorted(TACTICS.items())
    for rows, kind, tactic in entries
    if kind == "s"
]


def test_the_table_has_the_four_certified_split_k_cells() -> None:
    assert sorted((n, k, m) for n, k, m, _ in CELLS) == [
        (6144, 1024, 1),
        (6144, 1024, 4),
        (6144, 2304, 1),
        (6144, 2304, 4),
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


# The drafter MLP's down projection: the SwiGLU folded into the GEMM's B load (c2).
DOWN_CELLS = [c for c in CELLS if (c[0], c[1]) == (6144, 2304)]


@pytest.mark.parametrize("use_pdl", [False, True], ids=["plain", "pdl"])
@pytest.mark.parametrize("cell", DOWN_CELLS, ids=[f"{n}x{k}@{m}" for n, k, m, _ in DOWN_CELLS])
def test_swiglu_prologue_is_bitwise_silu_then_split_k(cell, use_pdl) -> None:
    n, k, m, tactic = cell
    g = torch.Generator(device="cuda").manual_seed(31 * n + m)
    w = (torch.randn(n, k, generator=g, device="cuda") / k**0.5).to(torch.bfloat16)
    scratch = torch.full((8, k), 7, dtype=torch.bfloat16, device="cuda")
    for trial in range(3):
        gate_up = (torch.randn(m, 2 * k, generator=g, device="cuda") * (2.0 + trial)).to(
            torch.bfloat16
        )
        gate_up[:, :6] = torch.tensor([60.0, -60.0, 0.0, -0.0, 88.0, -88.0], dtype=torch.bfloat16)
        act = torch.ops.trtllm.silu_and_mul(gate_up)
        ref = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
        fi_run(act, w.t(), None, ref, use_pdl, FiTactic(*tactic))
        got = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
        run_splitk_dense_swiglu(gate_up, w.t(), got, use_pdl, SplitKTactic(*tactic), scratch)
        torch.cuda.synchronize()
        assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), f"trial {trial}"
        assert torch.equal(scratch[:m].view(torch.int16), act.view(torch.int16)), (
            "the B scratch holds the activation"
        )
    cap = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_splitk_dense_swiglu(gate_up, w.t(), cap, use_pdl, SplitKTactic(*tactic), scratch)
    scratch.fill_(3)  # a replay rewrites what it reads
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(cap.view(torch.int16), ref.view(torch.int16))
