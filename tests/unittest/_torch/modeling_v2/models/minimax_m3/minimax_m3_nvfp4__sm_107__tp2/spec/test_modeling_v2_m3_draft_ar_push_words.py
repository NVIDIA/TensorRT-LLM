# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test of the drafter boundary push (modeling.DRAFT_AR_PUSH / DRAFT_AR_PUSH_DIRECT) on one rank: the push variants
of the split-K copy (ops/m3_lowm_splitk) and the direct copy (ops/m3_lowm_direct) write FlashInfer's output
bit for bit, and store exactly the boundary all-reduce's canonical exchange words of that output: word (t, column pair)
= bf16 lo | bf16 hi << 16 with -0.0 halves as +0.0, at parity * maxt * world * H / 2 + (t * world + rank) * H / 2 +
column / 2, the parity being the (t, column / (H / 8)) flag's low bit; every other word stays untouched. Eager and in a
CUDA graph, both parities, at each certified push cell of the TP2 table (the 1-row direct and 4-row split-K o_proj /
down)."""

import pytest
import torch
from flashinfer.gemm.kernels.dense_bf16_gemm_direct import DirectTactic as FiDirect
from flashinfer.gemm.kernels.dense_bf16_gemm_direct import run_direct_dense as fi_direct
from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import SplitKTactic as FiSplitK
from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import run_splitk_dense as fi_splitk

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops import (  # noqa: E501
    m3_lowm_direct,
    m3_lowm_splitk,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops.gemm.m3_draft_lowm import (  # noqa: E501
    ROWS,
    tactic_for,
)

assert torch.cuda.is_available(), "the push variants require a CUDA device"

H, MAXT, EMPTY = 6144, 8, -2147483648
# The boundary linears (output width H): o_proj and down, each certified row count on split-K or an even direct width,
# up to the 8-token bucket the push arms on (ops/m3_draft_lowm.arm_push); the batch-4/8 rows past it run unpushed.
CELLS = [
    (nk[1], m, *tactic_for(nk[0], nk[1], m))
    for nk, rows in sorted(ROWS.items())
    if nk[0] == H and nk[1] != 18432  # o_proj / down, not the fc
    for m in rows
    if m <= MAXT
    and (tactic_for(nk[0], nk[1], m)[0] == "s" or tactic_for(nk[0], nk[1], m)[1][1] % 2 == 0)
]


def _expected_words(y: torch.Tensor, flags: torch.Tensor, world: int, rank: int) -> torch.Tensor:
    m = y.shape[0]
    half = MAXT * world * (H // 2)
    ws = torch.full((2 * half,), EMPTY, dtype=torch.int32, device="cuda")
    h16 = y.contiguous().view(torch.int16)
    h16 = torch.where(h16 == -32768, torch.zeros_like(h16), h16)
    words = h16.view(torch.int32).view(m, H // 2)
    for t in range(m):
        for c in range(8):
            par = int(flags[t * 8 + c].item()) & 1
            lo = par * half + (t * world + rank) * (H // 2) + c * (H // 16)
            ws[lo : lo + H // 16] = words[t, c * (H // 16) : (c + 1) * (H // 16)]
    return ws


@pytest.mark.parametrize("parity", [0, 1])
@pytest.mark.parametrize("cell", CELLS, ids=[f"{k}@{m}{kind}" for k, m, kind, _ in CELLS])
def test_push_words_and_output(cell, parity) -> None:
    k, m, kind, tactic = cell
    world, rank = 2, 1
    g = torch.Generator(device="cuda").manual_seed(k + m + parity)
    x = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
    x[0, :8] = (
        0.0  # zero / signed-zero outputs exercise the -0 canonicalization through the weights below
    )
    w = (torch.randn(H, k, generator=g, device="cuda") / k**0.5).to(torch.bfloat16)
    w[:4] = 0.0
    w[4:8] = -0.0
    flags = (
        torch.randint(0, 7, (MAXT * 8,), generator=g, device="cuda", dtype=torch.int32) * 2 + parity
    )
    ref = torch.empty(m, H, dtype=torch.bfloat16, device="cuda")
    if kind == "d":
        fi_direct(x, w.t(), ref, True, FiDirect(*tactic))
    else:
        fi_splitk(x, w.t(), None, ref, True, FiSplitK(*tactic))

    def run(out, ws):
        ar = (ws.data_ptr(), flags.data_ptr(), rank, world, MAXT)
        if kind == "d":
            m3_lowm_direct.run_direct_dense_push(
                x, w.t(), out, True, m3_lowm_direct.DirectTactic(*tactic), ar
            )
        else:
            m3_lowm_splitk.run_splitk_dense_push(
                x, w.t(), out, True, m3_lowm_splitk.SplitKTactic(*tactic), ar
            )

    want = _expected_words(ref, flags, world, rank)
    for graph in (False, True):
        out = torch.empty(m, H, dtype=torch.bfloat16, device="cuda")
        ws = torch.full_like(want, EMPTY)
        if graph:
            run(out, ws)  # compile outside capture
            torch.cuda.synchronize()
            ws.fill_(EMPTY)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                run(out, ws)
            ws.fill_(EMPTY)
            gr.replay()
        else:
            run(out, ws)
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), f"output (graph={graph})"
        assert torch.equal(ws, want), f"exchange words (graph={graph})"
