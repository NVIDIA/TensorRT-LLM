# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for ``SpecMetadata._upload_batch_slot_ids``.

The slot-id upload of ``populate_sampling_params_for_one_model`` goes through an
open step-input stager only when that stager opts in (``stages_batch_slot_ids``),
and is issued directly otherwise or when the stager declines it. The method only reads ``batch_slot_ids`` and
``step_input_stager``, so it is called unbound on a ``types.SimpleNamespace``
with a host ``batch_slot_ids``.
"""

import types

import pytest
import torch

from tensorrt_llm._torch.speculative.interface import SpecMetadata

ROWS = 6
FILL = -7


class _Stager:
    """Records ``copy`` calls; ``accept`` decides whether it takes the write."""

    def __init__(self, accept: bool, stages_batch_slot_ids: bool = True):
        self.accept = accept
        self.stages_batch_slot_ids = stages_batch_slot_ids
        self.calls = []

    def copy(self, dst, src):
        self.calls.append((dst, src.clone(), src.is_pinned()))
        return self.accept


def _meta(stager=None):
    return types.SimpleNamespace(
        batch_slot_ids=torch.full((ROWS,), FILL, dtype=torch.long),
        step_input_stager=stager,
    )


def test_without_a_stager_the_slots_are_copied_here():
    meta = _meta()
    SpecMetadata._upload_batch_slot_ids(meta, [3, 1, 5])
    assert meta.batch_slot_ids.tolist() == [3, 1, 5] + [FILL] * (ROWS - 3)


def test_an_accepting_stager_takes_the_write():
    stager = _Stager(accept=True)
    meta = _meta(stager)
    SpecMetadata._upload_batch_slot_ids(meta, [4, 2])
    assert len(stager.calls) == 1
    dst, src, pinned = stager.calls[0]
    assert dst.data_ptr() == meta.batch_slot_ids.data_ptr()
    assert dst.shape == (2,) and dst.dtype == torch.long
    assert src.dtype == torch.long and src.tolist() == [4, 2]
    assert not pinned
    # The stager owns the write: nothing was copied here.
    assert meta.batch_slot_ids.tolist() == [FILL] * ROWS


@pytest.mark.parametrize("stager", [_Stager(accept=True, stages_batch_slot_ids=False), object()])
def test_a_stager_that_does_not_take_the_table_is_bypassed(stager):
    meta = _meta(stager)
    SpecMetadata._upload_batch_slot_ids(meta, [4, 2])
    assert meta.batch_slot_ids.tolist() == [4, 2] + [FILL] * (ROWS - 2)
    assert getattr(stager, "calls", []) == []


@pytest.mark.parametrize("slots", [[0], [5, 0, 2, 1]])
def test_a_declined_write_is_copied_here(slots):
    stager = _Stager(accept=False)
    meta = _meta(stager)
    SpecMetadata._upload_batch_slot_ids(meta, slots)
    assert len(stager.calls) == 1
    assert meta.batch_slot_ids.tolist() == slots + [FILL] * (ROWS - len(slots))
