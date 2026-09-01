# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replaying the greedy decode tail from a captured CUDA graph."""

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.sampler.greedy_sample_kernels import (
    ARGMAX_SPLITS,
    greedy_argmax_scatter,
)
from tensorrt_llm._torch.pyexecutor.sampler.greedy_tail_graph import (
    RING_SIZE,
    WARMUP_STEPS,
    GreedyTailGraph,
)

skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for the triton kernels and graph capture",
)

VOCAB = 512
SLOTS = 4


def _buffers(num_rows=1):
    logits = torch.randn(num_rows, VOCAB, device="cuda")
    new_tokens = torch.zeros(1, SLOTS, 1, dtype=torch.int32, device="cuda")
    dest = torch.arange(num_rows, dtype=torch.int64, device="cuda")
    return logits, new_tokens, dest


def _warm(tail, logits, new_tokens, dest):
    """Run out the eager steps a fresh set of addresses has to survive."""
    for _ in range(WARMUP_STEPS):
        assert tail.run(logits, new_tokens, dest, 1) is None


def _replay(tail, logits, new_tokens, dest):
    result = tail.run(logits, new_tokens, dest, 1)
    assert result is not None
    host, slot = result
    torch.cuda.synchronize()
    return host, slot


@skip_no_cuda
def test_out_buffers_match_the_allocating_form():
    logits, new_tokens, dest = _buffers()
    expected = greedy_argmax_scatter(logits, new_tokens.clone(), dest, 1)

    out = torch.empty(1, dtype=torch.int32, device="cuda")
    partials = torch.empty((1, ARGMAX_SPLITS), dtype=torch.int64, device="cuda")
    actual = greedy_argmax_scatter(logits, new_tokens, dest, 1, out=out, partials=partials)

    assert actual is out
    torch.testing.assert_close(actual, expected)


@skip_no_cuda
@pytest.mark.parametrize("num_rows", [1, 3])
def test_replay_tracks_new_logits(num_rows):
    logits, new_tokens, dest = _buffers(num_rows)
    tail = GreedyTailGraph()
    _warm(tail, logits, new_tokens, dest)

    for _ in range(2 * RING_SIZE):
        # Refresh in place: the buffer addresses stay put, as they do while a
        # batch is stable, so every step here is captured or replayed.
        logits.copy_(torch.randn_like(logits))
        expected = torch.argmax(logits, dim=-1).to(torch.int32)

        host, slot = _replay(tail, logits, new_tokens, dest)

        torch.testing.assert_close(host, expected.cpu())
        torch.testing.assert_close(new_tokens[0, :num_rows, 0], expected)
        tail.release(slot)


@skip_no_cuda
def test_ties_resolve_to_the_lowest_index_like_argmax():
    logits, new_tokens, dest = _buffers()
    tail = GreedyTailGraph()
    _warm(tail, logits, new_tokens, dest)
    _replay(tail, logits, new_tokens, dest)

    logits.fill_(-1.0)
    logits[0, 17] = 5.0
    logits[0, 400] = 5.0
    host, _ = _replay(tail, logits, new_tokens, dest)

    assert host.item() == 17


@skip_no_cuda
def test_ring_runs_dry_until_slots_come_back():
    logits, new_tokens, dest = _buffers()
    tail = GreedyTailGraph()
    _warm(tail, logits, new_tokens, dest)

    slots = [_replay(tail, logits, new_tokens, dest)[1] for _ in range(RING_SIZE)]
    assert len(set(slots)) == RING_SIZE

    # Every read-back buffer is still owned by a caller, so the tail declines
    # rather than overwriting one.
    assert tail.run(logits, new_tokens, dest, 1) is None

    tail.release(slots[0])
    assert tail.run(logits, new_tokens, dest, 1) is not None


@skip_no_cuda
def test_inputs_that_move_every_step_are_never_captured():
    tail = GreedyTailGraph()
    new_tokens = torch.zeros(1, SLOTS, 1, dtype=torch.int32, device="cuda")
    dest = torch.zeros(1, dtype=torch.int64, device="cuda")

    for _ in range(4 * WARMUP_STEPS):
        # A fresh allocation every step: capturing it would cost more than the
        # replay saves, and the graph would be stale before it was used.
        logits = torch.randn(1, VOCAB, device="cuda")
        assert tail.run(logits, new_tokens, dest, 1) is None


@skip_no_cuda
def test_a_settled_batch_is_recaptured_after_a_move():
    tail = GreedyTailGraph()
    new_tokens = torch.zeros(1, SLOTS, 1, dtype=torch.int32, device="cuda")
    dest = torch.zeros(1, dtype=torch.int64, device="cuda")
    logits = torch.randn(1, VOCAB, device="cuda")

    _warm(tail, logits, new_tokens, dest)
    for _ in range(3):
        _, slot = _replay(tail, logits, new_tokens, dest)
        tail.release(slot)

    # A batch change after a long run is ordinary; the tail must settle onto
    # the new addresses rather than give up.
    moved = torch.randn(1, VOCAB, device="cuda")
    _warm(tail, moved, new_tokens, dest)
    host, _ = _replay(tail, moved, new_tokens, dest)

    torch.testing.assert_close(host, torch.argmax(moved, dim=-1).to(torch.int32).cpu())


@skip_no_cuda
def test_a_failed_capture_leaves_the_tail_eager():
    logits, new_tokens, dest = _buffers()
    tail = GreedyTailGraph()
    _warm(tail, logits, new_tokens, dest)

    def _boom(*args, **kwargs):
        raise RuntimeError("CUDA error: operation failed during capture")

    tail._capture = _boom
    assert tail.run(logits, new_tokens, dest, 1) is None
    # Not retried: a capture that failed once is not attempted again.
    tail._capture = None
    assert tail.run(logits, new_tokens, dest, 1) is None
