# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for DSpark policy agreement in the executor."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import ADPShapeAgreement
from tensorrt_llm._torch.pyexecutor.py_executor import (
    _DSPARK_ADP_WIRE_TRAILER_LEN,
    _DSPARK_EXACT_NATIVE_YIELD_INDEX,
    _DSPARK_EXACT_WIRE_PREFIX_LEN,
    PyExecutor,
    _classify_dspark_exact_generation_rows,
    _decode_dspark_exact_expected_yields,
    _dspark_exact_common_graph_batch_size,
    _dspark_exact_secondary_padding_ready,
    _encode_dspark_exact_expected_yields,
    _validate_dspark_adp_acceptance_gate,
    _validate_dspark_exact_bucket,
)


class _Batch:
    def __init__(self, generation_requests):
        self.generation_requests = generation_requests
        self.context_requests = []

    @property
    def batch_size(self):
        return len(self.generation_requests)

    @property
    def num_context_requests(self):
        return 0

    @property
    def num_generation_requests(self):
        return len(self.generation_requests)


def _request(*, adp_dummy=False, other_dummy=False):
    return SimpleNamespace(
        is_dummy=bool(adp_dummy or other_dummy),
        is_attention_dp_dummy=bool(adp_dummy),
    )


def _payload(*, rows, native, compact):
    payload = [0] * (_DSPARK_EXACT_WIRE_PREFIX_LEN + len(compact) + _DSPARK_ADP_WIRE_TRAILER_LEN)
    payload[1] = rows
    payload[_DSPARK_EXACT_NATIVE_YIELD_INDEX] = native
    payload[_DSPARK_EXACT_WIRE_PREFIX_LEN : _DSPARK_EXACT_WIRE_PREFIX_LEN + len(compact)] = compact
    return payload


def _agreement(batch, peer_sizes=(3, 4)):
    return ADPShapeAgreement(
        iteration=9,
        batch_identity=id(batch),
        local_batch_size=batch.batch_size,
        peer_batch_sizes=peer_sizes,
        all_can_graph=True,
        widest_batch_size=max(peer_sizes),
        graph_batch_size=4,
        draft_len=5,
        padding_ready=True,
        ragged_bucket=16,
        reuse_graph_shape=True,
    )


def test_exact_yield_encoding_is_conservative_and_pads_missing_cells():
    exact_local = SimpleNamespace(
        native_expected_yield=10.0,
        compact_expected_yields=(9.25, float("nan")),
    )

    native, compact = _encode_dspark_exact_expected_yields(
        exact_local=exact_local, num_cells=3, yield_scale=100
    )

    assert native == 1000
    assert compact == (925, -1, -1)


def test_exact_yield_encoding_rejects_an_invalid_scale():
    with pytest.raises(ValueError, match="scale must be positive"):
        _encode_dspark_exact_expected_yields(exact_local=None, num_cells=1, yield_scale=0)


def test_exact_yield_decode_aggregates_yield_and_worst_rank_loss():
    cells = ((128, 512), (128, 640))
    payloads = [
        _payload(
            rows=2,
            native=10_000_000,
            compact=(8_000_000, 9_000_000),
        ),
        _payload(
            rows=4,
            native=20_000_000,
            compact=(18_000_000, 14_000_000),
        ),
        _payload(rows=0, native=0, compact=(0, 0)),
    ]

    native, compact, max_loss = _decode_dspark_exact_expected_yields(
        payloads=payloads,
        exact_cells=cells,
        graph_batch_size=128,
        yield_scale=1_000_000,
    )

    assert native == 30.0
    assert compact == {512: 26.0, 640: 23.0}
    assert max_loss == {512: 1.0, 640: 1.5}


def test_exact_yield_decode_fails_closed_on_an_infeasible_peer():
    payloads = [
        _payload(rows=2, native=10_000_000, compact=(8_000_000,)),
        _payload(rows=4, native=20_000_000, compact=(-1,)),
    ]

    _, compact, max_loss = _decode_dspark_exact_expected_yields(
        payloads=payloads,
        exact_cells=((128, 512),),
        graph_batch_size=128,
        yield_scale=1_000_000,
    )

    assert compact == {512: 0.0}
    assert max_loss == {512: 0.0}


def test_exact_row_classifier_authenticates_only_the_idle_adp_dummy():
    real = [_request(), _request()]
    assert _classify_dspark_exact_generation_rows(real) == (real, False)
    assert _classify_dspark_exact_generation_rows([]) == (None, False)
    assert _classify_dspark_exact_generation_rows([_request(adp_dummy=True)]) == (
        [],
        True,
    )
    assert _classify_dspark_exact_generation_rows([real[0], _request(adp_dummy=True)]) == (
        None,
        False,
    )
    assert _classify_dspark_exact_generation_rows([_request(other_dummy=True)]) == (None, False)


def test_exact_common_graph_size_preserves_all_rank_termination():
    def round_up(rows):
        return 16 if rows <= 16 else 32

    assert _dspark_exact_common_graph_batch_size([[0, 16], [0, 0]], round_up) == 16

    def reject_round_up(_rows):
        raise AssertionError("all-zero peers must not perform a graph lookup")

    assert _dspark_exact_common_graph_batch_size([[0, 0], [0, 0]], reject_round_up) == 0


@pytest.mark.parametrize(
    ("budget", "secondary_ready", "expected"),
    [(48, False, True), (49, False, True), (50, False, False), (50, True, True)],
)
def test_secondary_dummy_is_required_only_for_a_multirow_remainder(
    budget, secondary_ready, expected
):
    payloads = [[0, 16, 0], [0, 0, int(secondary_ready)]]
    assert (
        _dspark_exact_secondary_padding_ready(
            payloads,
            graph_batch_size=16,
            verifier_budget=budget,
            secondary_ready_index=2,
        )
        is expected
    )


def test_exact_bucket_must_match_the_selected_graph_key():
    _validate_dspark_exact_bucket(exact_shape=(16, 48, 3), bucket=48)
    _validate_dspark_exact_bucket(exact_shape=None, bucket=None)
    with pytest.raises(RuntimeError, match="selected captured verifier bucket"):
        _validate_dspark_exact_bucket(exact_shape=(16, 48, 3), bucket=47)


def test_attention_dp_rejects_a_rank_local_acceptance_gate():
    with pytest.raises(ValueError, match="acceptance_rate_window_size"):
        _validate_dspark_adp_acceptance_gate(
            confidence_enabled=True,
            attention_dp_enabled=True,
            speculation_gate_enabled=True,
        )
    _validate_dspark_adp_acceptance_gate(
        confidence_enabled=True,
        attention_dp_enabled=False,
        speculation_gate_enabled=True,
    )


@pytest.mark.parametrize(
    ("peer_sizes", "expected"),
    [((3, 4), (True, True)), ((3, 0), (False, True))],
)
def test_can_queue_reuses_policy_agreement_without_another_collective(peer_sizes, expected):
    batch = _Batch([object()] * 3)

    def reject_allgather(_value):
        raise AssertionError("cached agreement issued another collective")

    executor = PyExecutor.__new__(PyExecutor)
    executor.enable_attention_dp = True
    executor.dist = SimpleNamespace(tp_size=8, tp_allgather=reject_allgather)
    executor.model_engine = SimpleNamespace(
        _dspark_confidence_enabled=True,
        cuda_graph_runner=SimpleNamespace(adp_shape_agreement=_agreement(batch, peer_sizes)),
    )
    executor.kv_connector_manager = None
    executor.kv_cache_transceiver = None
    executor.drafter = None
    executor.iter_counter = 9
    executor._dspark_dynamic_handled_signature = (9, id(batch), 3, 0, 3)
    executor._handle_dynamic_draft_len = lambda _batch: None

    assert executor._can_queue(batch) == expected


def test_rebalance_suspends_both_dummy_variants_and_releases_the_pair_once():
    low = SimpleNamespace(py_request_id=10)
    high = SimpleNamespace(py_request_id=11)
    suspended_ids = []
    resumed_ids = []

    class _Manager:
        def is_request_active(self, request_id):
            return request_id in {10, 11}

        def suspend_request(self, request):
            suspended_ids.append(request.py_request_id)

        def resume_request(self, request):
            resumed_ids.append(request.py_request_id)
            return request.py_request_id != 11

    released = []
    runner = SimpleNamespace(
        padding_dummy_requests={5: low},
        secondary_padding_dummy_requests={5: high},
        release_padding_dummy=lambda resource, draft_len: released.append((resource, draft_len)),
    )
    executor = PyExecutor.__new__(PyExecutor)
    executor.model_engine = SimpleNamespace(cuda_graph_runner=runner)
    executor.resource_manager = object()
    manager = _Manager()

    suspended = executor._suspend_padding_dummies_for_rebalance(manager)
    executor._resume_padding_dummies_after_rebalance(manager, suspended)

    assert suspended == [(5, 0, low), (5, 1, high)]
    assert suspended_ids == [10, 11]
    assert resumed_ids == [10, 11]
    assert released == [(executor.resource_manager, 5)]
