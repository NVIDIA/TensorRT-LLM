# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Consumed overlap batches release indices for requests omitted from the next forward."""
from types import SimpleNamespace

from tensorrt_llm._torch.pyexecutor.py_executor import _release_consumed_batch_indices


def _req(batch_idx):
    return SimpleNamespace(py_batch_idx=batch_idx)


def _batch(ctx=(), gen=()):
    return SimpleNamespace(context_requests=list(ctx),
                           generation_requests=list(gen))


def test_skipped_generation_request_is_released():
    kept, skipped = _req(1), _req(2)
    _release_consumed_batch_indices(_batch(gen=[kept, skipped]),
                                    _batch(gen=[kept]))
    assert kept.py_batch_idx == 1
    assert skipped.py_batch_idx is None


def test_request_that_finished_context_then_got_skipped_is_released():
    # Last context chunk ran in the consumed batch (py_batch_idx set by
    # _prepare_tp_inputs); the scheduler skipped it this iteration.
    finished_ctx = _req(5)
    other_gen = _req(6)
    _release_consumed_batch_indices(_batch(ctx=[finished_ctx], gen=[other_gen]),
                                    _batch(gen=[other_gen]))
    assert finished_ctx.py_batch_idx is None
    assert other_gen.py_batch_idx == 6


def test_request_that_finished_context_and_runs_now_keeps_index():
    finished_ctx = _req(5)
    _release_consumed_batch_indices(_batch(ctx=[finished_ctx]),
                                    _batch(gen=[finished_ctx]))
    assert finished_ctx.py_batch_idx == 5


def test_context_request_scheduled_again_as_context_keeps_index():
    chunked = _req(3)
    _release_consumed_batch_indices(_batch(ctx=[chunked]), _batch(ctx=[chunked]))
    assert chunked.py_batch_idx == 3


def test_nothing_scheduled_releases_everything():
    a, b = _req(1), _req(2)
    _release_consumed_batch_indices(_batch(ctx=[a], gen=[b]), None)
    assert a.py_batch_idx is None and b.py_batch_idx is None
