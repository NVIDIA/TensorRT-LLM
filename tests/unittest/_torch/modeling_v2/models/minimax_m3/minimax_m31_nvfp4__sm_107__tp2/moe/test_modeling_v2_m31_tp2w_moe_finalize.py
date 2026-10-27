# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2W_MOE_FINALIZE (a port of the M3 tp2 test): kernels/m3_moe_finalize finalizes the routed experts (+ the shared
expert) from the TRTLLM-Gen runner's unfinalized parts, bitwise to the runner's finalizeKernelVecLoad order (acc = 0;
acc += w_k * fc2[idx_k] for k in order, skipping -1; one BF16 rounding) and to ``shared.add_(routed)``, at the rows the
target routes there (49..16K), with dropped (-1) slots, padded FC2 rows, eager and CUDA-graph replay."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.kernels.m3_moe_finalize import (  # noqa: E501
    m3_moe_finalize,
)

H, TOP_K = 6144, 4


def _parts(rows: int, seed: int, pad: int = 128):
    g = torch.Generator(device="cuda").manual_seed(seed)
    n = rows * TOP_K
    fc2 = torch.randn((n + pad, H), device="cuda", generator=g).to(torch.bfloat16)
    idx = torch.randperm(n, device="cuda", generator=g).to(torch.int32).reshape(rows, TOP_K)
    drop = torch.rand((rows, TOP_K), device="cuda", generator=g) < 0.02
    idx = torch.where(drop, torch.full_like(idx, -1), idx)
    w = torch.rand((rows, TOP_K), device="cuda", generator=g).to(torch.bfloat16)
    shared = torch.randn((rows, H), device="cuda", generator=g).to(torch.bfloat16)
    return fc2, w, idx.contiguous(), shared


def _reference(fc2, w, idx):
    acc = torch.zeros((w.shape[0], H), dtype=torch.float32, device=fc2.device)
    for k in range(TOP_K):
        valid = (idx[:, k] >= 0).unsqueeze(1)
        x = fc2[idx[:, k].clamp(min=0).long()].float()
        acc = torch.where(valid, acc + w[:, k : k + 1].float() * x, acc)
    return acc.to(torch.bfloat16)


@pytest.mark.parametrize("rows", [49, 64, 65, 192, 1000, 4096, 16384])
def test_bitwise_to_the_runner_order(rows: int) -> None:
    fc2, w, idx, shared = _parts(rows, rows)
    ref = _reference(fc2, w, idx)
    out = m3_moe_finalize(fc2, w, idx)
    assert out is not None and torch.equal(out.view(torch.int16), ref.view(torch.int16))
    fused = m3_moe_finalize(fc2, w, idx, shared)
    assert torch.equal(fused.view(torch.int16), shared.clone().add_(ref).view(torch.int16))


def test_graph_replay_matches_eager() -> None:
    fc2, w, idx, shared = _parts(512, 7)
    eager = m3_moe_finalize(fc2, w, idx, shared)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = m3_moe_finalize(fc2, w, idx, shared)
    for seed in (8, 9):
        f2, w2, i2, s2 = _parts(512, seed)
        fc2.copy_(f2)
        w.copy_(w2)
        idx.copy_(i2)
        shared.copy_(s2)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, m3_moe_finalize(fc2, w, idx, shared))
    assert eager.shape == out.shape


def test_declines_other_layouts() -> None:
    fc2, w, idx, shared = _parts(64, 3)
    assert m3_moe_finalize(fc2.float(), w, idx) is None
    assert m3_moe_finalize(fc2, w, idx.long()) is None
    assert m3_moe_finalize(fc2, w, idx, shared[:, :128].contiguous()) is None
