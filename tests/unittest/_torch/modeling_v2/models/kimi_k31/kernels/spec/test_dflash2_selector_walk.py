# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 selector kernels piece by piece, bitwise against the torch functions they replace:
the scoring operands, the SplitMix64 uniforms, and the walk on given edge scores."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.spec.dflash2_selector_walk import (
    selector_operands,
    selector_walk,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_sampling import (
    dflash2_selector_walk,
    dflash2_uniforms,
)
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    get_sm_version() not in (103, 107),
    reason="the K3.1 DFlash2 selector kernels are certified on sm_103 and sm_107",
)

VOCAB = 163840
RANK = 256


@pytest.mark.parametrize("n", [1, 3, 64])
@pytest.mark.parametrize("slots", [1, 4, 7])
def test_operands_are_the_torch_expressions(n, slots) -> None:
    g = torch.Generator(device="cuda").manual_seed(n * 10 + slots)
    pred_cb = torch.randn(VOCAB, RANK, generator=g, device="cuda").to(torch.bfloat16)
    succ_cb = torch.randn(VOCAB, RANK, generator=g, device="cuda").to(torch.bfloat16)
    cand = torch.randint(0, VOCAB, (n, slots, 32), generator=g, device="cuda")
    gate = torch.randn(n, slots, RANK, generator=g, device="cuda").to(torch.bfloat16)
    anchor = torch.randint(0, VOCAB, (n,), generator=g, device="cuda")
    pred, succ = selector_operands(pred_cb, succ_cb, cand, gate, anchor)
    pred_ids = torch.cat((anchor[:, None, None].expand(-1, 1, 32), cand[:, :-1]), dim=1)
    assert torch.equal(succ, succ_cb[cand].float())
    assert torch.equal(pred, pred_cb[pred_ids].float() * gate.float()[:, :, None])


def _walk_inputs(n, slots, seed, kind="randn"):
    g = torch.Generator(device="cuda").manual_seed(seed)
    cand = torch.randint(0, VOCAB, (n, slots, 32), generator=g, device="cuda")
    scores = 3.0 * torch.randn(n, slots, 32, 32, generator=g, device="cuda")
    if kind == "ties":
        scores = torch.randint(-2, 3, (n, slots, 32, 32), generator=g, device="cuda").float()
    elif kind == "peaked":
        scores[..., 0] += 30.0
    seed_t = torch.randint(0, 2**62, (n,), generator=g, device="cuda")
    offset = torch.randint(0, 2**62, (n,), generator=g, device="cuda")
    temps = 2.0 * torch.rand(n, generator=g, device="cuda")
    temps[::4] = 0.0
    temps[1::5] = 1e-7
    greedy = torch.rand(n, generator=g, device="cuda") < 0.3
    return cand, scores, seed_t, offset, temps, greedy


@pytest.mark.parametrize("n", [1, 2, 33, 64])
@pytest.mark.parametrize("slots", [1, 4, 7])
@pytest.mark.parametrize("kind", ["randn", "ties", "peaked"])
def test_walk_and_uniforms_are_bitwise(n, slots, kind) -> None:
    cand, scores, seed_t, offset, temps, greedy = _walk_inputs(n, slots, n * 100 + slots, kind)
    uniforms = dflash2_uniforms(seed_t, offset, slots)
    tokens, q, path = dflash2_selector_walk(cand, scores, uniforms, temps, greedy)
    tok32, tok64, q_k, path_k, u_k = selector_walk(
        scores, cand, seed_t, offset, temps, greedy_mask=greedy, write_uniforms=True
    )
    assert torch.equal(u_k, uniforms)
    assert torch.equal(path_k, path)
    assert torch.equal(tok64, tokens)
    assert torch.equal(tok32, tokens.to(torch.int32))
    assert torch.equal(q_k, q)


def test_uniforms_at_extreme_seeds() -> None:
    seeds = torch.tensor([0, 1, 2**63 - 1, -(2**63), -1, 12345], dtype=torch.int64, device="cuda")
    offsets = torch.tensor([0, -1, 2**62, 7, -(2**63), 2**63 - 1], dtype=torch.int64, device="cuda")
    n, slots = seeds.numel(), 7
    cand = torch.zeros(n, slots, 32, dtype=torch.int64, device="cuda")
    scores = torch.zeros(n, slots, 32, 32, device="cuda")
    temps = torch.ones(n, device="cuda")
    *_, u_k = selector_walk(scores, cand, seeds, offsets, temps, write_uniforms=True)
    assert torch.equal(u_k, dflash2_uniforms(seeds, offsets, slots))


def test_walk_is_capturable() -> None:
    n, slots = 16, 7
    cand, scores, seed_t, offset, temps, greedy = _walk_inputs(n, slots, 5)
    selector_walk(scores, cand, seed_t, offset, temps, greedy_mask=greedy)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        _, tok64, q, path, _ = selector_walk(
            scores, cand, seed_t, offset, temps, greedy_mask=greedy
        )
    torch.cuda.current_stream().wait_stream(stream)
    fresh = _walk_inputs(n, slots, 6)
    for static, new in zip((cand, scores, seed_t, offset, temps, greedy), fresh):
        static.copy_(new)
    graph.replay()
    tokens, q_ref, path_ref = dflash2_selector_walk(
        cand, scores, dflash2_uniforms(seed_t, offset, slots), temps, greedy
    )
    torch.cuda.synchronize()
    assert torch.equal(tok64, tokens) and torch.equal(q, q_ref) and torch.equal(path, path_ref)
