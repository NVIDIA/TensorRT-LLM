# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2W_MOE_FINALIZE (a port of the M3 tp2 test): kernels/m3_moe_finalize finalizes the routed experts (+ the shared
expert) from the TRTLLM-Gen runner's unfinalized parts, bitwise to the runner's finalizeKernelVecLoad order (acc = 0;
acc += w_k * fc2[idx_k] for k in order, skipping -1; one BF16 rounding) and to ``shared.add_(routed)``, at rows across
the range the target routes there (49..16K), with dropped (-1) slots, padded FC2 rows, eager and CUDA-graph replay.
The native family op (``minimax_m31_moe_finalize``) that replaces it is bit for bit the Triton kernel on every
work-split table entry."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2.kernels.m3_moe_finalize import (  # noqa: E501
        m3_moe_finalize,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.native import family_ops

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("the sm_107 TP2 target runs on SM107 only", allow_module_level=True)

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


@pytest.mark.parametrize("rows", [49, 65, 4096])
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


@pytest.mark.parametrize("rows", [1, 49, 4096])
@pytest.mark.parametrize("fp32_weights", [False, True])
@pytest.mark.parametrize("with_shared", [False, True])
def test_native_finalize_is_the_triton_finalize_on_every_cfg(
    rows, fp32_weights, with_shared
) -> None:
    fc2, w, idx, shared = _parts(rows, rows * 31 + 4)
    if fp32_weights:
        w = (w.float() * 2.0**-126).contiguous()  # denormal products
    sh = shared if with_shared else None
    ref = m3_moe_finalize(fc2, w, idx, sh)
    assert ref is not None
    for cfg in (-1, 0, 1, 2, 3, 4, 5, 6, 7):
        got = family_ops().minimax_m31_moe_finalize(fc2, w, idx, sh, cfg)
        assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), f"cfg {cfg}"
