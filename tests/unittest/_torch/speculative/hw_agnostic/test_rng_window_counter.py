# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the per-slot Philox offset window counter.

The one-model spec path gives each request a Philox seed/offset. The offset
must advance once per sampling pass, otherwise a seeded request redraws the
same numbers.

It deliberately does NOT come from ``request.py_decoding_iter``: under the
overlap scheduler ``_forward_step`` (which populates this state) runs before
the previous batch's ``_update_requests``, which is what increments that
field, so a request in adjacent batches would be seen at the same iteration
twice. ``_rng_window_counter`` counts the windows actually handed out instead.

These tests drive the counter the way ``_populate_request_rng_state`` does,
without allocating the CUDA buffers that function also fills.
"""

import types
from typing import Optional

from tensorrt_llm._torch.speculative.interface import SpecMetadata

MAX_DRAFT_LEN = 3
WINDOW = MAX_DRAFT_LEN + 1


def _meta(max_num_requests: int = 8) -> SpecMetadata:
    return SpecMetadata(
        max_num_requests=max_num_requests,
        max_draft_len=MAX_DRAFT_LEN,
        max_total_draft_tokens=MAX_DRAFT_LEN,
    )


def _request(slot: Optional[int], decoding_iter: int = 0) -> types.SimpleNamespace:
    return types.SimpleNamespace(py_seq_slot=slot, py_decoding_iter=decoding_iter)


def _offsets(meta: SpecMetadata, requests: list[types.SimpleNamespace]) -> list[int]:
    """The offset base each request gets, as _populate_request_rng_state computes it."""
    out: list[int] = []
    for request in requests:
        slot = request.py_seq_slot
        step = meta._rng_window_counter.get(slot, 0)
        meta._rng_window_counter[slot] = step + 1
        out.append(step * WINDOW)
    return out


def test_offsets_advance_across_passes() -> None:
    meta = _meta()
    reqs = [_request(0), _request(1)]
    assert _offsets(meta, reqs) == [0, 0]
    assert _offsets(meta, reqs) == [WINDOW, WINDOW]
    assert _offsets(meta, reqs) == [2 * WINDOW, 2 * WINDOW]


def test_stale_decoding_iter_does_not_repeat_a_window() -> None:
    # The overlap-scheduler case: py_decoding_iter is unchanged between two
    # adjacent batches because _update_requests has not run yet. Keyed off that
    # field both passes would share an offset window; the counter must not.
    meta = _meta()
    reqs = [_request(0, decoding_iter=5), _request(1, decoding_iter=5)]
    first = _offsets(meta, reqs)
    second = _offsets(meta, reqs)  # same py_decoding_iter, next pass
    assert first == [0, 0]
    assert second == [WINDOW, WINDOW]
    assert first != second


def test_counter_is_keyed_by_slot_not_batch_position() -> None:
    # Batch composition shifts between iterations; a slot's stream must follow
    # the slot, not where it happens to sit in the batch.
    meta = _meta()
    _offsets(meta, [_request(0), _request(1), _request(2)])
    # Slot 1 finishes; slot 2 moves to batch position 0.
    assert _offsets(meta, [_request(2), _request(0)]) == [WINDOW, WINDOW]


def test_dummy_requests_do_not_perturb_real_slots() -> None:
    # CUDA-graph padding requests have py_seq_slot=None. They share one
    # counter and must leave real slots' streams untouched.
    meta = _meta()
    assert _offsets(meta, [_request(0)]) == [0]
    _offsets(meta, [_request(None), _request(None)])
    assert _offsets(meta, [_request(0)]) == [WINDOW]


def test_graph_copy_shares_the_counter() -> None:
    # create_cuda_graph_metadata shallow-copies, and the copies are reseated as
    # the live spec_metadata on replay. A per-copy counter would restart the
    # stream on every graph replay.
    meta = _meta()
    graph_meta = meta.create_cuda_graph_metadata(4)
    assert graph_meta is not meta

    assert _offsets(meta, [_request(0)]) == [0]
    # The replayed graph copy continues the same stream, not a fresh one.
    assert _offsets(graph_meta, [_request(0)]) == [WINDOW]
    assert _offsets(meta, [_request(0)]) == [2 * WINDOW]
