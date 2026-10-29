# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM107 tokens-only FULL sampling: support, RNG identity, distributions and graph reuse.

Run this file in separate processes with TRTLLM_ENABLE_PDL=0 and =1. Individual
sample IDs are compared only within this implementation, never across different
sampling algorithms. Dense logits retain the real154880-column stride throughout.
"""

import math

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.spec.verify_head_sample import (  # noqa: E501
    VOCAB_SIZE,
    VerifyHeadSampleState,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.spec.verify_head_sample import (  # noqa: E501
    glm_verify_head_sample as op,
)
from tensorrt_llm._torch.pyexecutor.sampler.ops import custom as shared
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401

pytestmark = pytest.mark.skipif(get_sm_version() != 107, reason="GLM Rubin sampling requires SM107")


def _inputs(rows, *, active=32, varied=True):
    logits = torch.full((rows, VOCAB_SIZE), -1000.0, device="cuda")
    values = (
        torch.linspace(0.0, -4.0, active, device="cuda")
        if varied
        else torch.zeros(active, device="cuda")
    )
    logits[:, :active] = values
    temps = torch.ones(rows, device="cuda")
    ks = torch.full((rows,), torch.iinfo(torch.int32).max, device="cuda", dtype=torch.int32)
    ps = torch.full((rows,), 0.95, device="cuda")
    mins = torch.zeros(rows, device="cuda")
    seeds = torch.full((rows,), 1337, device="cuda", dtype=torch.int64)
    offsets = torch.arange(rows, device="cuda", dtype=torch.int64) * 32 + (1 << 62)
    return [logits, temps, ks, ps, mins, seeds, offsets]


def _args(state, inputs):
    return (*inputs, state.row_counters, state.workspace)


def _assert_support(tokens, probabilities):
    assert tokens.dtype == torch.int32 and tokens.shape == (len(probabilities),)
    assert ((tokens >= 0) & (tokens < VOCAB_SIZE)).all()
    assert (probabilities.gather(1, tokens.long()[:, None]) > 0).all()


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    rows = cell.spec["rows"]
    state = VerifyHeadSampleState("cuda")
    inputs = _inputs(rows)
    # Mixed rows drive split top-p, min-p only, top-k and the composed filtered path.
    inputs[2][1::4] = 8
    inputs[4][2::4] = 0.15
    inputs[2][3::4] = 12
    inputs[4][3::4] = 0.05
    inputs[3][2::4] = 1
    certifying(op)
    tokens = op(*_args(state, inputs))
    reference = op.reference(*_args(state, inputs))
    _assert_support(tokens, reference)
    assert not state.row_counters.any()


@pytest.mark.parametrize("rows", [6, 12, 18])
def test_rng_follows_rows_across_permutation_and_split(rows):
    state = VerifyHeadSampleState("cuda")
    inputs = _inputs(rows)
    inputs[2][1::3] = 7
    inputs[4][2::3] = 0.1
    whole = state.sample(*inputs)
    again = state.sample(*inputs)
    assert torch.equal(whole, again)
    order = torch.arange(rows - 1, -1, -1, device="cuda")
    permuted = state.sample(*(x[order].contiguous() for x in inputs))
    assert torch.equal(permuted, whole[order])
    # Singleton must use the per-row stream, including offsets in the unseeded domain.
    singles = torch.cat([state.sample(*(x[i : i + 1] for x in inputs)) for i in range(rows)])
    assert torch.equal(singles, whole)
    parts = torch.cat(
        [state.sample(*(x[a:b] for x in inputs)) for a, b in ((0, rows // 2), (rows // 2, rows))]
    )
    assert torch.equal(parts, whole)
    assert not state.row_counters.any()


def test_scalar_rng_separates_rows_and_per_row_rng_does_not():
    state = VerifyHeadSampleState("cuda")
    inputs = _inputs(24, varied=False)
    inputs[3].fill_(1)
    inputs[6].fill_(64)
    same_stream = state.sample(*inputs)
    assert torch.equal(same_stream, same_stream[:1].expand(24))
    scalar_inputs = [*inputs[:5], inputs[5][:1], inputs[6][:1]]
    scalar = state.sample(*scalar_inputs)
    assert torch.equal(scalar, state.sample(*scalar_inputs))
    assert scalar.unique().numel() > 1
    # Per-token offsets change subsequences rather than overlapping draws within one stream.
    inputs[6].copy_(torch.arange(24, device="cuda", dtype=torch.int64))
    assert state.sample(*inputs).unique().numel() > 1


@pytest.mark.parametrize(
    "k,p,min_p", [(0, 0.8, 0.0), (8, 0.8, 0.05), (0, 1.0, 0.15), (7, 1.0, 0.0)]
)
def test_distribution_on_real_verify_rows(k, p, min_p):
    state = VerifyHeadSampleState("cuda")
    inputs = _inputs(18, active=16)
    inputs[2].fill_(k)
    inputs[3].fill_(p)
    inputs[4].fill_(min_p)
    expected = op.reference(*_args(state, inputs))[0].cpu()
    samples = []
    for iteration in range(128):
        inputs[6].copy_(torch.arange(18, device="cuda", dtype=torch.int64) + iteration * 18)
        samples.append(state.sample(*inputs))
    tokens = torch.cat(samples)
    counts = torch.bincount(tokens.long(), minlength=VOCAB_SIZE).float().cpu()
    assert not counts[expected == 0].any()
    measured = counts / tokens.numel()
    # 2304 independent subsequences: TV0.08 is wider than the expected multinomial
    # fluctuation for <=16 bins, yet rejects missing filters or deterministic fallback.
    assert (measured - expected).abs().sum().item() * 0.5 < 0.08
    assert not state.row_counters.any()


@pytest.mark.parametrize("direction", [-1, 0, 1])
def test_top_p_boundary_keeps_complete_ties(direction):
    state = VerifyHeadSampleState("cuda")
    inputs = _inputs(24, active=4, varied=False)
    inputs[0][:, 2:4] = -math.log(2.0)
    boundary = torch.tensor(2.0 / 3.0, dtype=torch.float32, device="cuda")
    if direction:
        boundary = torch.nextafter(boundary, torch.tensor(float("inf") * direction, device="cuda"))
    inputs[3].fill_(boundary)
    # Above each half-weight token is exactly two unit weights. FULL admits the
    # whole half-weight tie at mass==top_p*total; one ULP below excludes both.
    samples = []
    for i in range(16):
        inputs[6].add_(24)
        samples.append(state.sample(*inputs))
    tokens = torch.cat(samples)
    if direction < 0:
        assert (tokens < 2).all()
    else:
        assert (tokens < 4).all() and (tokens == 2).any() and (tokens == 3).any()
    # The current-main materialized FULL support is an independent implementation.
    current = shared.fused_compute_probs_from_logits(*inputs[:5])
    assert (current[:, :2] > 0).all()
    assert bool((current[:, 2:4] > 0).all()) == (direction >= 0)
    assert not state.row_counters.any()


@pytest.mark.parametrize("min_p", [0.0, 1.0, 2.0])
def test_top_k_and_min_p_retain_weight_ties(min_p):
    state = VerifyHeadSampleState("cuda")
    inputs = _inputs(24, active=8, varied=False)
    inputs[0][:, 4:8] = -1
    inputs[2].fill_(1)
    inputs[3].fill_(0.0)
    inputs[4].fill_(min_p)
    out = state.sample(*inputs)
    assert (out < 4).all() and out.unique().numel() > 1
    assert not state.row_counters.any()


def test_graph_replay_changes_filter_roles_without_recapture():
    state = VerifyHeadSampleState("cuda")
    inputs = _inputs(18)
    # Warm-up on a side stream as required by CUDA graph capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            state.sample(*inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = state.sample(*inputs)
    for min_p in (0.0, 0.05, 1.0, 0.0):
        inputs[4].fill_(min_p)
        inputs[2].fill_(0)
        inputs[2][::3] = 8 if min_p != 1 else 1
        inputs[3].fill_(0.95)
        inputs[3][1::3] = 1
        inputs[1].fill_(0.7 if min_p else 1.0)
        inputs[6].add_(32)
        # Sampling params are completed step inputs and can change between replays.
        graph.replay()
        actual = output.clone()
        expected = state.sample(*inputs)
        assert torch.equal(actual, expected)
        _assert_support(actual, op.reference(*_args(state, inputs)))
        assert not state.row_counters.any()
    assert state.workspace.numel() == 1440384


def test_delayed_logits_producer_and_persistent_counter_reuse():
    state = VerifyHeadSampleState("cuda")
    inputs = _inputs(18)
    original = inputs[0].clone()
    baseline = state.sample(*inputs)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        inputs[0].fill_(-1000)
        torch.cuda._sleep(1000000)
        inputs[0].copy_(original)
        result = state.sample(*inputs)
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(result, baseline)
    # Reusing one max-capacity arena with different row counts must not preserve stale
    # counters or depend on per-call scratch offsets from the previous shape.
    for rows in (1, 24, 6, 18, 12, 1):
        batch = _inputs(rows)
        actual = state.sample(*batch)
        assert torch.equal(actual, state.sample(*batch))
        assert not state.row_counters.any()


def test_input_contract_and_mutation_schema():
    state = VerifyHeadSampleState("cuda")
    inputs = _inputs(6)
    assert state.supported(*inputs)
    bad = list(inputs)
    bad[2] = bad[2].long()
    assert not state.supported(*bad)
    bad = list(inputs)
    bad[5] = bad[5][:2]
    assert not state.supported(*bad)
    bad = list(inputs)
    bad[0] = bad[0][:, ::2]
    assert not state.supported(*bad)
    with pytest.raises(ValueError):
        state.sample(*bad)
    schema = torch.ops.trtllm.glm_verify_head_sample_tokens.default._schema
    writes = {a.name for a in schema.arguments if a.alias_info and a.alias_info.is_write}
    assert writes == {"row_counters", "workspace"}
    with pytest.raises(RuntimeError, match="workspace"):
        op(*inputs, state.row_counters, state.workspace[:16])


@pytest.mark.parametrize("k,min_p", [(0, 0.0), (1, 0.0), (8, 0.05), (VOCAB_SIZE, 1.0)])
def test_masked_logits_have_zero_weight(k, min_p, certifying):  # noqa: F811
    state = VerifyHeadSampleState("cuda")
    inputs = _inputs(18, active=4)
    inputs[0][:, 4:] = -float("inf")
    # Masked holes and a singleton finite row exercise empty partial segments.
    inputs[0][::2, 1] = -float("inf")
    inputs[0][-1].fill_(-float("inf"))
    inputs[0][-1, VOCAB_SIZE - 1] = 2.0
    inputs[2].fill_(k)
    inputs[4].fill_(min_p)
    certifying(op)
    tokens = op(*_args(state, inputs))
    _assert_support(tokens, op.reference(*_args(state, inputs)))
    assert tokens[-1] == VOCAB_SIZE - 1
    assert torch.isfinite(inputs[0].gather(1, tokens.long()[:, None])).all()
    assert not state.row_counters.any()
