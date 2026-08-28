# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tensorrt_llm._torch.attention_backend.sparse.dsa.gvr_prior_tracker import (
    GvrPriorTracker,
)

NUM_LAYERS = 2
CAPACITY = 6
TOP_K = 4

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _make_prior(device):
    prior = torch.zeros(NUM_LAYERS, CAPACITY, TOP_K, dtype=torch.int32, device=device)
    return prior


def _stamp(prior, row, value):
    prior[:, row, :] = value


def _row(prior, row):
    return prior[0, row].tolist()


ARANGE = list(range(TOP_K))


@pytest.mark.parametrize("device", DEVICES)
def test_unknown_requests_are_seeded(device):
    prior = _make_prior(device)
    tracker = GvrPriorTracker()
    tracker.realign(prior, gen_ids=[10, 11], ctx_ids=[])
    assert _row(prior, 0) == ARANGE
    assert _row(prior, 1) == ARANGE


@pytest.mark.parametrize("device", DEVICES)
def test_steady_state_is_a_no_op(device):
    prior = _make_prior(device)
    tracker = GvrPriorTracker()
    tracker.realign(prior, gen_ids=[10, 11], ctx_ids=[20])
    _stamp(prior, 0, 100)
    _stamp(prior, 1, 101)
    _stamp(prior, 2, 200)  # prefill update wrote ctx request 20's row
    snapshot = prior.clone()
    tracker.realign(prior, gen_ids=[10, 11], ctx_ids=[20])
    assert torch.equal(prior, snapshot)


@pytest.mark.parametrize("device", DEVICES)
def test_completion_shifts_rows_left(device):
    prior = _make_prior(device)
    tracker = GvrPriorTracker()
    tracker.realign(prior, gen_ids=[10, 11, 12], ctx_ids=[])
    _stamp(prior, 0, 100)
    _stamp(prior, 1, 101)
    _stamp(prior, 2, 102)
    # Request 10 completed: 11 and 12 shift to rows 0 and 1.
    tracker.realign(prior, gen_ids=[11, 12], ctx_ids=[])
    assert _row(prior, 0) == [101] * TOP_K
    assert _row(prior, 1) == [102] * TOP_K


@pytest.mark.parametrize("device", DEVICES)
def test_ctx_to_gen_conversion_keeps_prefill_prior(device):
    prior = _make_prior(device)
    tracker = GvrPriorTracker()
    tracker.realign(prior, gen_ids=[10], ctx_ids=[20])
    _stamp(prior, 0, 100)
    _stamp(prior, 1, 200)  # prefill update for request 20 at row num_gen + 0
    tracker.realign(prior, gen_ids=[10, 20], ctx_ids=[])
    assert _row(prior, 0) == [100] * TOP_K
    assert _row(prior, 1) == [200] * TOP_K


@pytest.mark.parametrize("device", DEVICES)
def test_completion_and_conversion_combined(device):
    prior = _make_prior(device)
    tracker = GvrPriorTracker()
    tracker.realign(prior, gen_ids=[10, 11], ctx_ids=[20])
    _stamp(prior, 0, 100)
    _stamp(prior, 1, 101)
    _stamp(prior, 2, 200)
    # Request 10 completed while 20 converts to generation.
    tracker.realign(prior, gen_ids=[11, 20], ctx_ids=[])
    assert _row(prior, 0) == [101] * TOP_K
    assert _row(prior, 1) == [200] * TOP_K


@pytest.mark.parametrize("device", DEVICES)
def test_first_step_decode_request_is_seeded(device):
    """Disagg / full-prefix-reuse: first step on this engine is decode."""
    prior = _make_prior(device)
    tracker = GvrPriorTracker()
    tracker.realign(prior, gen_ids=[10, 11], ctx_ids=[])
    _stamp(prior, 0, 100)
    _stamp(prior, 1, 101)
    tracker.realign(prior, gen_ids=[10, 99], ctx_ids=[])
    assert _row(prior, 0) == [100] * TOP_K
    assert _row(prior, 1) == ARANGE


@pytest.mark.parametrize("device", DEVICES)
def test_position_swap_is_alias_safe(device):
    prior = _make_prior(device)
    tracker = GvrPriorTracker()
    tracker.realign(prior, gen_ids=[10, 11], ctx_ids=[])
    _stamp(prior, 0, 100)
    _stamp(prior, 1, 101)
    tracker.realign(prior, gen_ids=[11, 10], ctx_ids=[])
    assert _row(prior, 0) == [101] * TOP_K
    assert _row(prior, 1) == [100] * TOP_K


@pytest.mark.parametrize("device", DEVICES)
def test_reset_forgets_ownership(device):
    prior = _make_prior(device)
    tracker = GvrPriorTracker()
    tracker.realign(prior, gen_ids=[10], ctx_ids=[])
    _stamp(prior, 0, 100)
    tracker.reset()
    tracker.realign(prior, gen_ids=[10], ctx_ids=[])
    assert _row(prior, 0) == ARANGE
