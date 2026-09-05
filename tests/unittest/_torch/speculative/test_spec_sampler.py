# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Ragged one-model speculative sampling and penalty tests."""

import types
from typing import Optional

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.sampler import penalties as penalty_ops
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase
from tensorrt_llm._torch.speculative.spec_sampler_base import SpecSampler


class _StubSpecWorker(SpecWorkerBase):
    """Concrete ``SpecWorkerBase`` that stubs out the abstract API."""

    @property
    def max_draft_len(self) -> int:
        return 8

    def _forward_impl(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError


def _make_worker(value: Optional[float] = None) -> _StubSpecWorker:
    worker = _StubSpecWorker()
    if value is not None:
        worker.force_num_accepted_tokens = value
    return worker


def _penalty_mapping_meta(
    slot_ids: list[int], verify_lens: Optional[list[int]] = None
) -> types.SimpleNamespace:
    metadata = types.SimpleNamespace(
        batch_slot_ids=torch.tensor(slot_ids, dtype=torch.int64),
        is_ragged_verify=verify_lens is not None,
        verify_lens=None,
        qo_indptr=None,
        total_verify_tokens=None,
    )
    if verify_lens is not None:
        lens = torch.tensor(verify_lens, dtype=torch.int32)
        metadata.verify_lens = lens
        metadata.qo_indptr = torch.cat(
            [torch.zeros(1, dtype=torch.int32), torch.cumsum(lens, dim=0)]
        )
        metadata.total_verify_tokens = sum(verify_lens)
    return metadata


def test_occurrence_penalty_uniform_row_mapping_is_unchanged():
    """An all-full ragged layout must reproduce the existing uniform mapping."""
    draft_len = 4
    draft_tokens = torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24]])
    uniform = _penalty_mapping_meta([3, 7, 9])
    ragged = _penalty_mapping_meta([3, 7, 9], [draft_len + 1, draft_len + 1])
    num_rows = 1 + 2 * (draft_len + 1)

    uniform_mapping = penalty_ops.build_row_mapping(
        uniform,
        num_contexts=1,
        batch_size=3,
        draft_len=draft_len,
        draft_tokens=draft_tokens,
        device=torch.device("cpu"),
        num_logit_rows=num_rows,
    )
    ragged_mapping = penalty_ops.build_row_mapping(
        ragged,
        num_contexts=1,
        batch_size=3,
        draft_len=draft_len,
        draft_tokens=draft_tokens,
        device=torch.device("cpu"),
        num_logit_rows=num_rows,
    )

    assert uniform_mapping is not None and ragged_mapping is not None
    for uniform_tensor, ragged_tensor in zip(uniform_mapping, ragged_mapping):
        assert torch.equal(uniform_tensor, ragged_tensor)


def test_occurrence_penalty_ragged_row_mapping_uses_each_request_window():
    """Packed rows must stay with their request and see only its earlier drafts."""
    draft_tokens = torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24]])
    metadata = _penalty_mapping_meta([3, 7, 9], [2, 4])

    mapping = penalty_ops.build_row_mapping(
        metadata,
        num_contexts=1,
        batch_size=3,
        draft_len=4,
        draft_tokens=draft_tokens,
        device=torch.device("cpu"),
        num_logit_rows=7,
    )

    assert mapping is not None
    row_slots, intra_tokens, intra_valid = mapping
    assert row_slots.tolist() == [3, 7, 7, 9, 9, 9, 9]
    assert intra_tokens.tolist() == [
        [0, 0, 0, 0],
        [11, 12, 13, 14],
        [11, 12, 13, 14],
        [21, 22, 23, 24],
        [21, 22, 23, 24],
        [21, 22, 23, 24],
        [21, 22, 23, 24],
    ]
    assert intra_valid.tolist() == [
        [False, False, False, False],
        [False, False, False, False],
        [True, False, False, False],
        [False, False, False, False],
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
    ]


def test_occurrence_penalty_applies_to_ragged_packed_rows(monkeypatch):
    """The worker must run, rather than silently skip, the ragged penalty pass."""
    metadata = _penalty_mapping_meta([3, 7, 9], [2, 4])
    metadata.enable_penalty = True
    captured = {}

    def _apply(logits, spec_metadata, row_slots, intra_tokens, intra_valid):
        captured["row_slots"] = row_slots.clone()
        captured["intra_tokens"] = intra_tokens.clone()
        captured["intra_valid"] = intra_valid.clone()
        logits[:, 0].copy_(row_slots.to(logits.dtype))

    monkeypatch.setattr(penalty_ops, "apply_penalties", _apply)
    logits = torch.zeros((7, 3))
    draft_tokens = torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24]])

    penalized = _make_worker()._apply_occurrence_penalties(
        logits, draft_tokens, num_contexts=1, batch_size=3, spec_metadata=metadata
    )

    assert torch.equal(logits, torch.zeros_like(logits))
    assert penalized[:, 0].tolist() == [3.0, 7.0, 7.0, 9.0, 9.0, 9.0, 9.0]
    assert captured["row_slots"].numel() == logits.shape[0]
    assert captured["intra_valid"].sum().item() == 7


def test_verify_window_snapshot_wins_over_next_overlap_step():
    request = types.SimpleNamespace(py_request_id=7, py_verify_len=2)
    snapshot = SpecSampler._snapshot_verify_lens([request])
    request.py_verify_len = 5

    assert snapshot == {7: 2}
    assert SpecSampler._verified_len(request, 5, snapshot) == 2


def test_uniform_verify_window_keeps_runtime_draft_length():
    request = types.SimpleNamespace(py_request_id=7)

    assert SpecSampler._snapshot_verify_lens([request]) is None
    assert SpecSampler._verified_len(request, 5, None) == 5


def test_device_verify_window_wins_over_host_shape_split():
    request = types.SimpleNamespace(py_request_id=7, py_seq_slot=1, py_verify_len=4)

    assert SpecSampler._verified_len(request, 5, {7: 4}, [0, 3]) == 2


class _RequestsMustNotBeScanned:
    def __iter__(self):
        raise AssertionError("policy-window request scan reached the hot path")


def test_native_marker_ignores_next_overlap_window():
    request = types.SimpleNamespace(py_request_id=7, py_verify_len=2)
    snapshot = SpecSampler._snapshot_policy_windows_for_step(
        _RequestsMustNotBeScanned(),
        native_uniform=True,
        host_snapshot_required=False,
        device_verify_lens_available=False,
    )

    assert snapshot == {}
    assert SpecSampler._verified_len(request, 5, snapshot) == 5


def test_device_window_source_does_not_scan_requests():
    snapshot = SpecSampler._snapshot_policy_windows_for_step(
        _RequestsMustNotBeScanned(),
        native_uniform=False,
        host_snapshot_required=False,
        device_verify_lens_available=True,
    )

    assert snapshot is None


def test_host_marker_preserves_current_overlap_window():
    request = types.SimpleNamespace(py_request_id=7, py_verify_len=2)
    snapshot = SpecSampler._snapshot_policy_windows_for_step(
        [request],
        native_uniform=False,
        host_snapshot_required=True,
        device_verify_lens_available=False,
    )
    request.py_verify_len = 5

    assert snapshot == {7: 2}
    assert SpecSampler._verified_len(request, 5, snapshot) == 2


def test_dspark_forward_publishes_one_verify_window_source():
    from tensorrt_llm._torch.speculative.dspark import _publish_policy_window_output
    from tensorrt_llm._torch.speculative.dspark_schedule import (
        HOST_POLICY_WINDOWS_SNAPSHOT_OUTPUT,
        NATIVE_UNIFORM_VERIFY_OUTPUT,
    )

    native_outputs = {}
    _publish_policy_window_output(native_outputs, None, batch_size=3)
    assert native_outputs == {NATIVE_UNIFORM_VERIFY_OUTPUT: True}

    host_outputs = {}
    _publish_policy_window_output(host_outputs, torch.tensor([3, 5]), batch_size=3)
    assert host_outputs == {HOST_POLICY_WINDOWS_SNAPSHOT_OUTPUT: True}

    device_outputs = {}
    verify_lens = torch.tensor([3, 5, 2])
    _publish_policy_window_output(device_outputs, verify_lens, batch_size=3)
    assert torch.equal(device_outputs["verify_lens"], verify_lens)


@pytest.mark.parametrize(
    ("native_uniform", "host_snapshot_required", "device_verify_lens_available"),
    [
        (True, True, False),
        (True, False, True),
        (False, True, True),
    ],
)
def test_conflicting_verify_window_sources_fail_closed(
    native_uniform, host_snapshot_required, device_verify_lens_available
):
    with pytest.raises(RuntimeError, match="published"):
        SpecSampler._snapshot_policy_windows_for_step(
            [],
            native_uniform=native_uniform,
            host_snapshot_required=host_snapshot_required,
            device_verify_lens_available=device_verify_lens_available,
        )


def test_ragged_strict_acceptance_stops_at_each_request_window():
    worker = _make_worker(0.0)
    worker._sample_tokens_for_batch = lambda *args: torch.tensor(
        [1, 100, 2, 8, 4, 101], dtype=torch.int64
    )
    metadata = _penalty_mapping_meta([0, 1], [2, 4])
    draft_tokens = torch.tensor([[1, 9, 9], [2, 3, 4]], dtype=torch.int64)

    _, accepted = worker._sample_and_accept_draft_tokens_base(
        logits=torch.zeros((6, 8)),
        draft_tokens=draft_tokens,
        num_contexts=0,
        batch_size=2,
        spec_metadata=metadata,
    )

    assert accepted.tolist() == [2, 2]


def test_ragged_rejection_guard_uses_packed_logit_count():
    metadata = types.SimpleNamespace(
        draft_probs=torch.empty((2, 3, 8)),
        batch_slot_ids=torch.arange(2, dtype=torch.long),
        is_ragged_verify=True,
        verify_lens=torch.tensor([2, 4], dtype=torch.int32),
        qo_indptr=torch.tensor([0, 2, 6], dtype=torch.int32),
        total_verify_tokens=6,
    )
    draft_tokens = torch.zeros((2, 3), dtype=torch.int64)
    valid_logits = torch.zeros((6, 8))
    short_logits = torch.zeros((5, 8))

    assert SpecWorkerBase._rejection_buffers_valid(
        object(), draft_tokens, 3, 8, 0, 2, valid_logits, metadata
    )
    assert not SpecWorkerBase._rejection_buffers_valid(
        object(), draft_tokens, 3, 8, 0, 2, short_logits, metadata
    )
