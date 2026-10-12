# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the Philox offset window counter.

The one-model spec path gives each request a Philox seed/offset. The offset
must advance once per sampling pass, otherwise a seeded request redraws the
same numbers.

It deliberately does NOT come from ``request.py_decoding_iter``: under the
overlap scheduler ``_forward_step`` (which populates this state) runs before
the previous batch's ``_update_requests``, which is what increments that
field, so a request in adjacent batches would be seen at the same iteration
twice. ``_rng_window_counter`` counts the windows actually handed out instead.

These tests call ``_take_rng_window_offsets``, the part of
``_populate_request_rng_state`` that owns the counter, without allocating the
CUDA buffers the latter also fills.
"""

import types
from typing import Optional

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.sampler.ops.custom import UNSEEDED_OFFSET_BASE
from tensorrt_llm._torch.speculative.interface import (
    _RNG_SLOT_SPAN,
    DEFAULT_SAMPLING_SEED,
    SpecMetadata,
    SpecWorkerBase,
)

MAX_DRAFT_LEN = 3
# One slot per target row, per draft step and for the acceptance kernel.
WINDOW = ((MAX_DRAFT_LEN + 1) + MAX_DRAFT_LEN + 1) * _RNG_SLOT_SPAN
BASE = UNSEEDED_OFFSET_BASE


def _meta(max_num_requests: int = 8) -> SpecMetadata:
    return SpecMetadata(
        max_num_requests=max_num_requests,
        max_draft_len=MAX_DRAFT_LEN,
        max_total_draft_tokens=MAX_DRAFT_LEN,
    )


def _request(
    slot: Optional[int],
    decoding_iter: int = 0,
    seed: Optional[int] = None,
    request_id: int = 0,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        py_seq_slot=slot,
        py_decoding_iter=decoding_iter,
        seed=seed,
        py_request_id=request_id,
    )


def _offsets(meta: SpecMetadata, requests: list[types.SimpleNamespace]) -> list[int]:
    """The offset base each request gets for one sampling pass."""
    return meta._take_rng_window_offsets(
        requests, [request.seed is not None for request in requests]
    )


# --- seeded requests: one counter per slot -----------------------------------


def test_seeded_offsets_advance_across_passes() -> None:
    meta = _meta()
    reqs = [_request(0, seed=7), _request(1, seed=11)]
    assert _offsets(meta, reqs) == [0, 0]
    assert _offsets(meta, reqs) == [WINDOW, WINDOW]
    assert _offsets(meta, reqs) == [2 * WINDOW, 2 * WINDOW]


def test_seeded_stale_decoding_iter_does_not_repeat_a_window() -> None:
    # The overlap-scheduler case: py_decoding_iter is unchanged between two
    # adjacent batches because _update_requests has not run yet. Keyed off that
    # field both passes would share an offset window; the counter must not.
    meta = _meta()
    reqs = [_request(0, decoding_iter=5, seed=7), _request(1, decoding_iter=5, seed=11)]
    first = _offsets(meta, reqs)
    second = _offsets(meta, reqs)  # same py_decoding_iter, next pass
    assert first == [0, 0]
    assert second == [WINDOW, WINDOW]
    assert first != second


def test_seeded_counter_is_keyed_by_slot_not_batch_position() -> None:
    # Batch composition shifts between iterations; a seeded slot's stream must
    # follow the slot, not where it happens to sit in the batch.
    meta = _meta()
    _offsets(meta, [_request(0, seed=7), _request(1, seed=11), _request(2, seed=13)])
    # Slot 1 finishes; slot 2 moves to batch position 0.
    assert _offsets(meta, [_request(2, seed=13), _request(0, seed=7)]) == [WINDOW, WINDOW]


def test_seeded_slot_reuse_resets_the_stream() -> None:
    # A seeded request must be reproducible regardless of what previously ran
    # on its slot: when a recycled slot shows up under a new request id, the
    # counter restarts at 0 instead of continuing the finished request's
    # stream.
    meta = _meta()
    first = _request(0, seed=7, request_id=1)
    assert _offsets(meta, [first]) == [0]
    assert _offsets(meta, [first]) == [WINDOW]
    # First request finishes; a new seeded request lands on the same slot.
    second = _request(0, seed=7, request_id=2)
    assert _offsets(meta, [second]) == [0]
    assert _offsets(meta, [second]) == [WINDOW]


def test_seeded_slot_reuse_reset_leaves_other_counters_alone() -> None:
    meta = _meta()
    survivor = _request(1, seed=11, request_id=1)
    assert _offsets(meta, [_request(0, seed=7, request_id=2), survivor]) == [0, 0]
    # Slot 0 is recycled to a new request; the survivor on slot 1 and the
    # shared unseeded counter must keep advancing where they left off.
    _offsets(meta, [_request(3)])
    assert _offsets(meta, [_request(0, seed=7, request_id=3), survivor]) == [0, WINDOW]
    assert _offsets(meta, [_request(3)]) == [BASE + WINDOW]


# --- unseeded requests: one shared counter ------------------------------------


def test_unseeded_serial_requests_on_fresh_slots_get_distinct_windows() -> None:
    # SlotManager hands each serial request a slot no earlier request used, so
    # a per-slot counter would give every one of them offset 0 with the shared
    # default seed and (in batch row 0) identical tokens.
    meta = _meta()
    seen: list[int] = []
    for slot in range(meta.max_num_requests):
        seen.extend(_offsets(meta, [_request(slot)]))
    assert seen == [BASE + i * WINDOW for i in range(meta.max_num_requests)]


def test_unseeded_requests_in_one_batch_get_distinct_windows() -> None:
    meta = _meta()
    assert _offsets(meta, [_request(0), _request(1), _request(2)]) == [
        BASE,
        BASE + WINDOW,
        BASE + 2 * WINDOW,
    ]
    # Next pass continues from where the shared counter left off.
    assert _offsets(meta, [_request(0), _request(1), _request(2)]) == [
        BASE + 3 * WINDOW,
        BASE + 4 * WINDOW,
        BASE + 5 * WINDOW,
    ]


def test_unseeded_windows_differ_from_a_default_seeded_request() -> None:
    # Unseeded requests draw with DEFAULT_SAMPLING_SEED, which a user may also
    # pass as their seed; the two must still not share a (seed, offset).
    meta = _meta()
    seeded = _request(0, seed=DEFAULT_SAMPLING_SEED, request_id=1)
    for _ in range(4):
        seeded_offset, unseeded_offset = _offsets(meta, [seeded, _request(1, request_id=2)])
        assert seeded_offset < BASE <= unseeded_offset


def test_unseeded_windows_never_repeat_across_slot_reuse() -> None:
    meta = _meta()
    seen: list[int] = []
    # Slot 0 is reused by a later request, then slot 1 is reused, etc.
    for slot in [0, 1, 0, 2, 1, 0, None, 3]:
        seen.extend(_offsets(meta, [_request(slot)]))
    assert len(set(seen)) == len(seen)


def test_dummy_requests_do_not_perturb_seeded_slots() -> None:
    # CUDA-graph padding requests have py_seq_slot=None and no seed. They draw
    # from the shared unseeded counter and must leave seeded slots' streams
    # untouched.
    meta = _meta()
    assert _offsets(meta, [_request(0, seed=7)]) == [0]
    _offsets(meta, [_request(None), _request(None)])
    assert _offsets(meta, [_request(0, seed=7)]) == [WINDOW]


# --- shared across graph copies ----------------------------------------------


def test_graph_copy_shares_the_counters() -> None:
    # create_cuda_graph_metadata shallow-copies, and the copies are reseated as
    # the live spec_metadata on replay. A per-copy counter would restart the
    # stream on every graph replay.
    meta = _meta()
    graph_meta = meta.create_cuda_graph_metadata(4)
    assert graph_meta is not meta

    assert _offsets(meta, [_request(0, seed=7)]) == [0]
    # The replayed graph copy continues the same stream, not a fresh one.
    assert _offsets(graph_meta, [_request(0, seed=7)]) == [WINDOW]
    assert _offsets(meta, [_request(0, seed=7)]) == [2 * WINDOW]

    # Same for the shared unseeded counter: a context step runs eagerly and
    # the generation steps replay a graph, and neither may hand out a window
    # the other already did.
    assert _offsets(meta, [_request(1)]) == [BASE]
    assert _offsets(graph_meta, [_request(1)]) == [BASE + WINDOW]
    assert _offsets(meta, [_request(1)]) == [BASE + 2 * WINDOW]


# --- the slots of one window --------------------------------------------------


@pytest.mark.parametrize("is_tree", [False, True], ids=["linear", "tree"])
def test_window_slots_do_not_overlap(is_tree: bool) -> None:
    """Every row a request samples in one step draws from its own stretch of the
    window, and no stretch reaches into the next step's window."""
    meta = SpecMetadata(
        max_num_requests=1,
        max_draft_len=MAX_DRAFT_LEN,
        max_total_draft_tokens=10 if is_tree else MAX_DRAFT_LEN,
        is_spec_dec_tree=is_tree,
    )
    assert _offsets(meta, [_request(0, seed=7)]) == [0]
    window = _offsets(meta, [_request(0, seed=7)])[0]

    target_rows = (10 if is_tree else MAX_DRAFT_LEN) + 1
    slots = list(range(target_rows))
    slots += [meta.rng_draft_slot(step) for step in range(MAX_DRAFT_LEN)]
    slots.append(meta.rng_accept_slot)

    used: set[int] = set()
    for slot in slots:
        stretch = set(range(slot * _RNG_SLOT_SPAN, (slot + 1) * _RNG_SLOT_SPAN))
        assert used.isdisjoint(stretch), f"slot {slot} overlaps another row's draws"
        used |= stretch
    assert max(used) < window


def test_block_rows_take_consecutive_slots() -> None:
    """A block sampler's K rows per request start at ``slot`` and take one slot each."""
    meta = types.SimpleNamespace(
        request_seeds=torch.tensor([5, 7, 9]),
        request_offsets=torch.tensor([0, 1000, BASE]),
    )
    seeds, offsets = SpecWorkerBase._rng_state_per_request(None, meta, 1, 3, repeat=3, slot=4)
    assert seeds.tolist() == [7, 7, 7, 9, 9, 9]
    assert offsets.tolist() == [
        base + (4 + row) * _RNG_SLOT_SPAN for base in (1000, BASE) for row in range(3)
    ]
