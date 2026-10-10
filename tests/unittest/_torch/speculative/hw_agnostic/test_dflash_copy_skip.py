# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the DFlash worker's per-step slot buffers and copy_to_device_if_changed.

DFlashSpecMetadata.prepare writes batch_indices_cuda and the worker's _batch_to_slot, and the
worker's post-prefill rebuild writes _batch_to_slot again. Both buffers go through
copy_to_device_if_changed, which skips a copy whose values equal the last ones it copied. The
tests check that each buffer always holds the latest mapping, including a step whose mapping
returns to an earlier one after the rebuild wrote another, and that an unchanged step enqueues
no copy.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata, DFlashWorker

pytestmark = pytest.mark.cpu_only

DUMMY_SLOT = 0


def _worker(req_to_slot):
    worker = SimpleNamespace(
        _ctx_buf_inited=True,
        _req_to_slot=dict(req_to_slot),
        _req_ctx_pos={},
        _free_slots=[],
        _graph_dummy_id_floor=1 << 30,
        _dummy_slot=DUMMY_SLOT,
        _batch_to_slot=torch.zeros(8, dtype=torch.long),
        _write_ctx_len=lambda updates: None,
    )
    # Generation requests are already assigned in these tests; prepare assigns nothing.
    worker._assign_slot = lambda rid: worker._req_to_slot.get(rid)
    return worker


def _metadata(worker, request_ids, num_generations):
    return SimpleNamespace(
        request_ids=list(request_ids),
        num_generations=num_generations,
        batch_indices_cuda=torch.full((8,), -1, dtype=torch.int),
        _dflash_worker=worker,
    )


def _prepare(metadata, request_ids, num_generations):
    metadata.request_ids = list(request_ids)
    metadata.num_generations = num_generations
    DFlashSpecMetadata.prepare(metadata)


def _rebuild(worker, request_ids):
    DFlashWorker._rebuild_batch_to_slot(worker, list(request_ids))


def test_mapping_that_returns_after_a_rebuild_is_written_again():
    # Request 5 decodes on slot 1. Request 8 arrives as a context request, so prepare maps it to
    # the dummy slot; prefill then assigns it slot 4 and the rebuild writes [4, 1].
    worker = _worker({5: 1})
    metadata = _metadata(worker, [8, 5], num_generations=1)
    _prepare(metadata, [8, 5], num_generations=1)
    assert worker._batch_to_slot[:2].tolist() == [DUMMY_SLOT, 1]
    worker._req_to_slot[8] = 4
    _rebuild(worker, [8, 5])
    assert worker._batch_to_slot[:2].tolist() == [4, 1]
    # Request 8 is gone and context request 9 takes its row: prepare maps [dummy, 1] again, the
    # values it copied two writes ago. The buffer must hold them, not the rebuild's [4, 1].
    del worker._req_to_slot[8]
    _prepare(metadata, [9, 5], num_generations=1)
    assert worker._batch_to_slot[:2].tolist() == [DUMMY_SLOT, 1]


def test_rebuild_then_prepare_back_to_back_keep_the_latest_mapping():
    worker = _worker({5: 1, 6: 2})
    metadata = _metadata(worker, [5, 6], num_generations=2)
    for request_ids, slots in (([5, 6], [1, 2]), ([6, 5], [2, 1]), ([5, 6], [1, 2])):
        _rebuild(worker, request_ids)
        assert worker._batch_to_slot[:2].tolist() == slots
        _prepare(metadata, request_ids, num_generations=2)
        assert worker._batch_to_slot[:2].tolist() == slots


def test_batch_indices_hold_the_batch_rows():
    worker = _worker({5: 1, 6: 2, 7: 3})
    metadata = _metadata(worker, [5], num_generations=1)
    for request_ids in ([5], [5, 6, 7], [5, 6], [5, 6, 7]):
        _prepare(metadata, request_ids, num_generations=len(request_ids))
        n = len(request_ids)
        assert metadata.batch_indices_cuda[:n].tolist() == list(range(n))


def test_an_unchanged_step_enqueues_no_copy(monkeypatch):
    worker = _worker({5: 1, 6: 2})
    metadata = _metadata(worker, [5, 6], num_generations=2)
    _prepare(metadata, [5, 6], num_generations=2)
    copies = []
    original = torch.Tensor.copy_

    def counted(dst, src, *args, **kwargs):
        copies.append(tuple(dst.shape))
        return original(dst, src, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", counted)
    _prepare(metadata, [5, 6], num_generations=2)
    monkeypatch.undo()
    assert copies == []
    assert worker._batch_to_slot[:2].tolist() == [1, 2]
