# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TAIL_BK3 / TAIL16_G3 on the tp2 target: the position-0 Markov embedding, the block sampler's per-row parameters
(minimax_m31_expand_block_sampling) and the block draft-prob store (minimax_m31_scatter_rows). CPU: the stock
fallback. GPU (SM107, ops built): bitwise the in-tree torch sequences, and the store under graph replay."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2 import (  # noqa: E501
        dspark_markov_worker as w1,
    )
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2.ops.spec import (  # noqa: E501
        m31_markov_epilogue as epi,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.native import family_ops
from tensorrt_llm._torch.models.modeling_speculative import markov_prev_embeddings
from tensorrt_llm._torch.speculative.interface import _RNG_SLOT_SPAN, SpecWorkerBase


class _Base:
    def _block_sampling_params(self, spec_metadata, num_contexts, batch_size, K):
        return "stock"


class _Worker(w1.S4WorkerMixin, _Base):
    pass


def _meta(n, device="cpu", top_ks_dtype=torch.int32):
    return SimpleNamespace(
        request_temperatures=torch.rand(n, device=device),
        request_top_ks=torch.zeros(n, dtype=top_ks_dtype, device=device),
        request_top_ps=torch.rand(n, device=device),
        request_min_ps=torch.zeros(n, device=device),
        request_seeds=torch.randint(0, 2**40, (n,), device=device),
        request_offsets=torch.randint(0, 2**40, (n,), device=device),
        rng_draft_slot=lambda step: 4 + step,
    )


def test_unsupported_dtype_takes_the_stock_params(monkeypatch):
    monkeypatch.setattr(w1, "markov_epilogue_op", lambda: object())
    meta = _meta(8, top_ks_dtype=torch.int64)
    assert _Worker()._block_sampling_params(meta, 0, 8, 3) == "stock"


_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the TAIL_BK3 / TAIL16_G3 ops run on SM107",
)


def _ops_built():
    assert epi.markov_epilogue_op() is not None, "this build lacks the Markov epilogue ops"
    for op in ("minimax_m31_expand_block_sampling", "minimax_m31_scatter_rows"):
        assert getattr(family_ops(), op, None) is not None, f"this build lacks mv2_minimax_m3::{op}"


def _same(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype == torch.bfloat16:
        return torch.equal(a.view(torch.int16), b.view(torch.int16))
    if a.dtype == torch.float32:
        return torch.equal(a.view(torch.int32), b.view(torch.int32))
    return torch.equal(a, b)


@_GPU
@pytest.mark.parametrize("n", [1, 28, 112])
def test_markov_embedding_bitwise(n):
    _ops_built()
    v, rank = 200064, 256
    w = (torch.randn(v, rank, device="cuda") * 0.05).to(torch.bfloat16)
    w[7, 3], w[11, :] = float("nan"), -0.0
    prev = torch.randint(0, v, (n,), device="cuda")
    edge = torch.tensor([-1, v, 0, v - 1, 7, 11], device="cuda")
    prev[: min(n, 6)] = edge[: min(n, 6)]
    assert _same(markov_prev_embeddings(prev, w), epi._prev_embeddings(prev, w))


@_GPU
@pytest.mark.parametrize("n, k", [(1, 3), (112, 7)])
def test_sampling_params_bitwise(n, k):
    _ops_built()
    meta = _meta(n + 2, device="cuda")
    meta.request_temperatures[2] = float("nan")
    meta.request_offsets[-1] = 2**63 - 40
    got = _Worker()._block_sampling_params(meta, 2, n + 2, k)
    slot = meta.rng_draft_slot(0)
    sl = slice(2, n + 2)
    offsets = (meta.request_offsets[sl] + slot * _RNG_SLOT_SPAN).repeat_interleave(
        k
    ) + torch.arange(k, dtype=torch.int64, device="cuda").repeat(n) * _RNG_SLOT_SPAN
    ref = (
        meta.request_temperatures[sl].repeat_interleave(k),
        meta.request_top_ks[sl].repeat_interleave(k),
        meta.request_top_ps[sl].repeat_interleave(k),
        meta.request_min_ps[sl].repeat_interleave(k),
        meta.request_seeds[sl].repeat_interleave(k),
        offsets,
    )
    assert got != "stock" and all(_same(a, b) for a, b in zip(ref, got))


def test_unsupported_dtype_takes_the_stock_store(monkeypatch):
    monkeypatch.setattr(w1, "markov_epilogue_op", lambda: object())
    dst = torch.zeros(8, 3, 16, dtype=torch.float64)
    src = torch.randn(2, 3, 16, dtype=torch.float64)
    slots = torch.tensor([1, 2])
    ref = dst.clone()
    SpecWorkerBase._store_block_draft_probs(ref, slots, src)
    w1.S4WorkerMixin._store_block_draft_probs(dst, slots, src)
    assert torch.equal(ref, dst)


@_GPU
@pytest.mark.parametrize("k, pad", [(3, 8), (7, 0)])
def test_store_bitwise_the_stock_store(k, pad):
    _ops_built()
    v = 200064
    dst0 = torch.randn(64, 7, v + pad, device="cuda")
    probs = torch.rand(4, k, v, device="cuda")
    probs[0, 0, 5] = float("nan")
    probs[1, -1, :100] = float("-inf")
    slots = torch.tensor([3, 17, 40, 9], dtype=torch.int64, device="cuda")
    ref, got = dst0.clone(), dst0.clone()
    SpecWorkerBase._store_block_draft_probs(ref, slots, probs)
    w1.S4WorkerMixin._store_block_draft_probs(got, slots, probs)
    assert torch.equal(ref.view(torch.int32), got.view(torch.int32))


@_GPU
def test_graph_changed_inputs_and_duplicate_padding():
    _ops_built()
    n, k, v = 56, 3, 16385
    dst = torch.zeros(n + 4, k + 1, v + 1, device="cuda")
    src = torch.randn(n, k, v, device="cuda")
    slots = torch.arange(n, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        w1.S4WorkerMixin._store_block_draft_probs(dst, slots, src)
    torch.cuda.current_stream().wait_stream(stream)
    for seed in (17, 83, 137):
        torch.manual_seed(seed)
        src.normal_()
        slots.copy_(torch.randperm(n + 4, device="cuda")[:n])
        # Graph padding can share a dummy slot; identical source bits are required.
        slots[-1].copy_(slots[-2])
        src[-1].copy_(src[-2])
        dst.zero_()
        ref = dst.clone()
        SpecWorkerBase._store_block_draft_probs(ref, slots, src)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(ref.view(torch.int32), dst.view(torch.int32))
