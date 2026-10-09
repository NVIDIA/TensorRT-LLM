# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for DSpark ragged CUDA-graph keys and padding state."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import (
    CUDA_GRAPH_DUMMY_REQUEST_ID,
    ADPShapeAgreement,
    CUDAGraphRunner,
    KeyType,
    cuda_graph_dummy_request_id,
)


class _Batch:
    def __init__(self, generation_requests):
        self.generation_requests = list(generation_requests)
        self.context_requests = []
        self.encoder_requests = []
        self.can_run_cuda_graph = True

    @property
    def batch_size(self):
        return len(self.generation_requests)

    @property
    def num_context_requests(self):
        return 0

    @property
    def num_generation_requests(self):
        return len(self.generation_requests)


def _runner_stub():
    runner = CUDAGraphRunner.__new__(CUDAGraphRunner)
    runner.clear = lambda: None
    return runner


def _agreement(batch, *, iteration=9, peer_sizes=(3, 4), reuse=True):
    return ADPShapeAgreement(
        iteration=iteration,
        batch_identity=id(batch),
        local_batch_size=batch.batch_size,
        peer_batch_sizes=peer_sizes,
        all_can_graph=True,
        widest_batch_size=max(peer_sizes),
        graph_batch_size=4,
        draft_len=5,
        padding_ready=True,
        ragged_bucket=16,
        reuse_graph_shape=reuse,
    )


def test_padding_dummy_ids_are_disjoint():
    ids = {
        cuda_graph_dummy_request_id(draft_len, variant=variant, max_draft_len=5)
        for variant in (0, 1)
        for draft_len in range(6)
    }

    assert len(ids) == 12
    assert (
        cuda_graph_dummy_request_id(5, variant=0, max_draft_len=5)
        == CUDA_GRAPH_DUMMY_REQUEST_ID - 5
    )
    with pytest.raises(ValueError, match="invalid CUDA padding dummy identity"):
        cuda_graph_dummy_request_id(5, variant=2, max_draft_len=5)


def test_shape_agreement_is_bound_to_iteration_batch_and_phase():
    batch = _Batch([object()] * 3)
    agreement = _agreement(batch)

    assert agreement.matches(batch, 9, padded=False)
    assert not agreement.matches(batch, 8, padded=False)
    assert not agreement.matches(_Batch([object()] * 3), 9, padded=False)

    batch.generation_requests.append(object())
    assert agreement.matches(batch, 9, padded=True)
    assert not agreement.matches(batch, 9, padded=False)


def test_ragged_graph_key_requires_every_row_and_the_agreed_bucket():
    runner = _runner_stub()
    runner._dspark_trims_submitted_tokens = True
    runner.agreed_ragged_bucket = 22
    batch = _Batch(
        [
            SimpleNamespace(py_verify_len=3),
            SimpleNamespace(py_verify_len=5),
        ]
    )

    assert runner._ragged_verify_bucket(batch) == 22
    batch.generation_requests[1].py_verify_len = None
    assert runner._ragged_verify_bucket(batch) is None
    runner._dspark_trims_submitted_tokens = False
    assert runner._ragged_verify_bucket(batch) is None


def test_ragged_graph_key_controls_the_submitted_token_count():
    runner = _runner_stub()

    assert (
        runner._get_num_tokens_for_key(
            KeyType(
                batch_size=4,
                draft_len=5,
                is_first_draft=False,
                ragged_verify_bucket=14,
            )
        )
        == 14
    )
    with pytest.raises(ValueError, match="mixed context"):
        runner._get_num_tokens_for_key(
            KeyType(
                batch_size=4,
                draft_len=5,
                is_first_draft=False,
                num_contexts=1,
                ragged_verify_bucket=14,
            )
        )


@pytest.mark.parametrize(
    ("padded_size", "num_requests", "expected"),
    [(4, 3, True), (4, 4, True), (8, 3, False), (0, 3, False), (4, 0, False)],
)
def test_will_pad_to_matches_the_size_guards(padded_size, num_requests, expected):
    runner = _runner_stub()
    runner.supported_batch_sizes = [1, 2, 4]
    runner.config = SimpleNamespace(batch_size=4)

    assert runner.will_pad_to(padded_size, num_requests) is expected


def test_zero_real_padding_uses_distinct_low_and_high_dummies():
    runner = _runner_stub()
    runner.enabled = True
    runner.padding_enabled = True
    runner.max_supported_batch_size = 4
    runner.config = SimpleNamespace(
        enable_attention_dp=False,
        mapping=SimpleNamespace(tp_size=1),
        batch_size=4,
    )
    runner.adp_shape_agreement = None
    runner.ragged_pad_verify_len = 2
    runner.ragged_zero_real_high_rows = 2
    runner.spec_config = SimpleNamespace(enable_confidence_scheduling=True, max_draft_len=5)
    runner._can_run_cuda_graph_batch = lambda _batch: True
    runner._round_up_batch_size_with_draft_len = lambda *_args: 4

    scheduled_dummy = SimpleNamespace(is_attention_dp_dummy=True, py_verify_len=3)
    low = SimpleNamespace(py_verify_len=None)
    high = SimpleNamespace(py_verify_len=None)
    runner._get_or_create_padding_dummy = (
        lambda _resource, _draft, variant=0: low if variant == 0 else high
    )
    batch = _Batch([scheduled_dummy])

    added = runner._get_padded_batch(batch, SimpleNamespace(), runtime_draft_len=5)

    assert added == 3
    assert batch.generation_requests == [scheduled_dummy, high, low, low]
    assert [row.py_verify_len for row in batch.generation_requests] == [3, 3, 2, 2]
    assert sum(1 + row.py_verify_len for row in batch.generation_requests) == 14


def test_releasing_padding_dummies_invalidates_the_cached_agreement():
    runner = _runner_stub()
    low = object()
    high = object()
    runner.padding_dummy_requests = {5: low}
    runner.secondary_padding_dummy_requests = {5: high}
    runner.adp_shape_agreement = object()
    freed = []
    runner._padding_dummy_managers = lambda _resource_manager: [
        SimpleNamespace(free_resources=freed.append)
    ]

    assert runner.release_padding_dummy(SimpleNamespace(), 5)
    assert freed == [low, high]
    assert runner.adp_shape_agreement is None
