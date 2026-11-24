# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which Top-K the Kimi K3.1 DSA indexer runs: the deterministic GVR on SM 10.3 for every target (the perf targets
on any architecture), the CUDA radix on other architectures; on SM 10.3 the general target's prefill Top-K then
keeps the smaller columns on tied scores and repeats bit for bit."""

import pytest
import torch
import torch.nn as nn

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention import dsa_indexer
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.indexer_topk_prefill_gvr import (  # noqa: E501
    indexer_topk_prefill_gvr,
)

__extra_import_path__ = ["../../../..", "."]
from _k31_dsa_inputs import on_sm103  # noqa: E402

TOP_K = 2048


@pytest.fixture
def fresh_gate():
    dsa_indexer.gvr_topk_device.cache_clear()
    yield
    dsa_indexer.gvr_topk_device.cache_clear()


def _indexer(perf: bool) -> dsa_indexer.K31Indexer:
    """An Indexer with only what forward_prepared and gvr_topk read (no projections or caches)."""
    indexer = dsa_indexer.K31Indexer.__new__(dsa_indexer.K31Indexer)
    nn.Module.__init__(indexer)
    indexer.index_topk = TOP_K
    indexer._k31_perf = perf
    return indexer


@pytest.mark.parametrize(
    "capability, expected", [((10, 3), True), ((10, 7), False), ((10, 0), False), ((12, 0), False)]
)
def test_gvr_topk_device_is_sm103_only(capability, expected, monkeypatch, fresh_gate):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    assert dsa_indexer.gvr_topk_device(0) is expected


@pytest.mark.parametrize(
    "perf, capability, expected",
    [
        (False, (10, 3), "gvr"),  # the general target on SM 10.3
        (False, (10, 7), "radix"),  # the general target elsewhere keeps the radix
        (True, (10, 3), "gvr"),  # the perf targets
        (True, (10, 7), "gvr"),
    ],
)
def test_prefill_topk_dispatch(perf, capability, expected, monkeypatch, fresh_gate):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    calls = []
    monkeypatch.setattr(
        dsa_indexer, "indexer_topk_prefill_gvr", lambda *a, **k: calls.append("gvr")
    )
    monkeypatch.setattr(dsa_indexer, "indexer_topk_prefill", lambda *a, **k: calls.append("radix"))
    indexer = _indexer(perf)
    rows, keys = 4, 3000
    logits = torch.zeros(rows, keys)
    indexer._call_mqa_logits = lambda *a, **k: logits
    out = torch.empty(rows, TOP_K, dtype=torch.int32)
    ends = torch.full((rows,), keys, dtype=torch.int32)
    indexer.forward_prepared(
        torch.zeros(rows, 1),
        torch.zeros(keys, 1),
        torch.zeros(keys, 1),
        torch.zeros(rows, 1),
        torch.zeros(rows, dtype=torch.int32),
        ends,
        out,
    )
    assert calls == [expected]
    assert indexer.gvr_topk(logits.device) is (expected == "gvr")


@on_sm103
@pytest.mark.parametrize("n", [2100, 3000, 6000])
def test_general_target_prefill_ties_repeat_bit_for_bit(n, fresh_gate):
    """The general target (no perf backend) on SM 10.3: a fresh n-token prompt with tied integer scores selects
    the stable-sort reference's set (ties to the smaller column), and repeated calls write the same output."""
    gen = torch.Generator(device="cuda").manual_seed(n)
    logits = torch.randint(0, 64, (n, n), generator=gen, device="cuda").float()
    starts = torch.zeros(n, dtype=torch.int32, device="cuda")
    ends = torch.arange(1, n + 1, dtype=torch.int32, device="cuda")
    indexer = _indexer(perf=False)
    indexer._call_mqa_logits = lambda *a, **k: logits
    out = torch.full((n, TOP_K), -5, dtype=torch.int32, device="cuda")
    q, k = torch.zeros(n, 1, device="cuda"), torch.zeros(n, 1, device="cuda")

    def call():
        out.fill_(-5)
        indexer.forward_prepared(q, k, k, q, starts, ends, out)
        torch.cuda.synchronize()
        return out.clone()

    first = call()
    ref = torch.empty_like(first)
    indexer_topk_prefill_gvr.reference(logits, starts, ends, ref, TOP_K)

    def ordered(t):
        return torch.where(t < 0, 2**31 - 1, t).sort(dim=1).values

    assert torch.equal(ordered(first), ordered(ref))
    for _ in range(20):
        assert torch.equal(call(), first)
