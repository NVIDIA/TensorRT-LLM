# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transfer-window admission: which scheduler-fitting gen-init requests may
start receiving this iteration, and the scheduler V2 allocation revert for
the ones that may not.

Admission is rank-local (rank 0 decides under PP, every rank for itself under
attention DP), so every case runs one ``CoordinatorHarness``.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from coordinator_harness import CoordinatorHarness, TransferRequest
from fake_dist import FakeDistGroup

from tensorrt_llm._torch.disaggregation.orchestration.admission import (
    DisaggTransferAdmissionController,
)
from tensorrt_llm._torch.disaggregation.orchestration.coordinator import (
    early_transfer_window_eligible,
)
from tensorrt_llm.bindings import LlmRequestState

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def _async_transfer_mode(monkeypatch) -> None:
    """Asynchronous generation transfers unless a test sets a mode knob itself."""
    monkeypatch.delenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", raising=False)
    monkeypatch.delenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", raising=False)


def _controller(max_tokens_in_buffer, tokens_per_block=32) -> DisaggTransferAdmissionController:
    return DisaggTransferAdmissionController(max_tokens_in_buffer, tokens_per_block)


def _harness(max_tokens_in_buffer=32, **kwargs) -> CoordinatorHarness:
    """Harness with a transfer window of one 32-token block by default."""
    return CoordinatorHarness(admission_controller=_controller(max_tokens_in_buffer), **kwargs)


def _candidate(rid: int, tokens: int = 32) -> TransferRequest:
    """A scheduler-fitting gen-init request of ``tokens`` prompt tokens."""
    return TransferRequest(
        rid,
        is_context_only_request=False,
        state=LlmRequestState.DISAGG_GENERATION_INIT,
        py_prompt_len=tokens,
    )


def _receiving(h: CoordinatorHarness, rid: int, tokens: int = 32) -> TransferRequest:
    """An active gen request whose receive is in flight, holding window budget."""
    req = TransferRequest(
        rid,
        is_context_only_request=False,
        state=LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS,
        is_disagg_generation_transmission_in_progress=True,
        py_prompt_len=tokens,
    )
    h.active.append(req)
    return req


# -- admit --------------------------------------------------------------------


def test_a_full_window_defers_the_candidate_and_reverts_its_v2_kv() -> None:
    """An active transfer holds the whole window: nothing is admitted, the
    caller learns it is blocked by active transfers, and the deferred
    candidate's V2 KV growth is handed to the executor to revert."""
    h = _harness(is_kv_manager_v2=True)
    _receiving(h, 1)
    candidate = _candidate(2)

    admitted, blocked = h.coordinator.admit([candidate])

    assert (admitted, blocked) == ([], True)
    assert h.effects.reverted == [[candidate]]


def test_v1_kv_manager_defers_without_reverting() -> None:
    """Scheduler V1 does not grow KV for gen-init requests while scheduling,
    so a deferred candidate has nothing to give back."""
    h = _harness(is_kv_manager_v2=False)
    _receiving(h, 1)

    admitted, blocked = h.coordinator.admit([_candidate(2)])

    assert (admitted, blocked) == ([], True)
    assert h.effects.reverted == []


def test_the_window_admits_the_head_and_reverts_only_the_deferred_tail() -> None:
    """Budget for one block: the first candidate is admitted, the second is
    deferred and reverted. Something was admitted, so the caller is not
    blocked by active transfers."""
    h = _harness(is_kv_manager_v2=True)
    first, second = _candidate(1), _candidate(2)

    admitted, blocked = h.coordinator.admit([first, second])

    assert (admitted, blocked) == ([first], False)
    assert h.effects.reverted == [[second]]


def test_async_python_v2_pp1_uses_scheduler_admission() -> None:
    """Only scheduler-admitted requests reach the receive entry point."""
    h = _harness(is_kv_manager_v2=True, consumes_transfer_buffer=False)
    candidates = [_candidate(2)]

    admitted, blocked = h.coordinator.admit(candidates)

    assert (admitted, blocked) == (candidates, False)
    assert h.effects.reverted == []

    _receiving(h, 1)
    h.coordinator._admission_controller.early_admission_blocked = True
    assert h.coordinator.admit([]) == ([], True)
    assert h.effects.reverted == []


@pytest.mark.parametrize("peer_count", [0, 1])
def test_byte_admission_reconciles_ranks_even_when_one_admits_nothing(peer_count) -> None:
    from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

    group = FakeDistGroup(world_size=2, tp_size=2)
    harnesses = []
    candidates = [[_candidate(1), _candidate(2)], [_candidate(1)][:peer_count]]
    for rank in range(2):
        transceiver = object.__new__(KvCacheTransceiverV2)
        transceiver._dist = group.rank(rank)
        transceiver._mapping = SimpleNamespace(tp_size=2, cp_size=1, enable_attention_dp=False)
        transceiver._kv_cache_manager = Mock()
        transceiver._transfer_worker = SimpleNamespace(recv_bounce_capacity_bytes=rank * 2048)
        controller = DisaggTransferAdmissionController(
            None, None, max_transfer_bytes=1024, python_transceiver=transceiver
        )
        controller.early_admission_blocked = rank == 1 and peer_count == 0
        h = CoordinatorHarness(
            admission_controller=controller, is_kv_manager_v2=True, dist=group.rank(rank)
        )
        h.coordinator._transceiver = transceiver
        harnesses.append(h)

    # A rank without Python bounce must join admission when a peer has it.
    assert group.run(
        lambda rank: harnesses[rank].coordinator._transceiver.get_receive_admission_capacity_bytes()
    ) == [0, 2048]
    results = group.run(lambda rank: harnesses[rank].coordinator.admit(candidates[rank]))

    for rank, (admitted, blocked) in enumerate(results):
        assert admitted == candidates[rank][:peer_count]
        assert blocked == (peer_count == 0)
        assert harnesses[rank].effects.reverted == []
        manager = harnesses[rank].coordinator._transceiver._kv_cache_manager
        assert manager.suspend_request.call_count == len(candidates[rank]) - peer_count


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(dict(is_kv_manager_v2=False), id="v1"),
        pytest.param(dict(is_kv_manager_v2=True, pp_size=2), id="pp2"),
    ],
)
def test_async_python_keeps_the_window_without_v2_and_pp1(kwargs) -> None:
    h = _harness(consumes_transfer_buffer=False, **kwargs)
    _receiving(h, 1)

    admitted, blocked = h.coordinator.admit([_candidate(2)])

    assert (admitted, blocked) == ([], True)


def test_sync_python_runtime_keeps_the_window(monkeypatch) -> None:
    """Synchronous receives block the executor, so the window still bounds how
    many are started per iteration even though the buffer is not consumed."""
    monkeypatch.setenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", "1")
    h = _harness(is_kv_manager_v2=True, consumes_transfer_buffer=False)
    first, second = _candidate(1), _candidate(2)

    admitted, blocked = h.coordinator.admit([first, second])

    assert (admitted, blocked) == ([first], False)
    assert h.effects.reverted == [[second]]


def test_gen_only_benchmark_bypasses_the_window(monkeypatch) -> None:
    """gen_only_no_context transfers nothing, so the budget does not apply."""
    monkeypatch.setenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", "1")
    h = _harness(is_kv_manager_v2=True)
    _receiving(h, 1)
    candidates = [_candidate(2), _candidate(3)]

    admitted, blocked = h.coordinator.admit(candidates)

    assert (admitted, blocked) == (candidates, False)
    assert h.effects.reverted == []


@pytest.mark.parametrize(
    "controller",
    [
        pytest.param(None, id="no_controller"),
        pytest.param(_controller(max_tokens_in_buffer=0), id="disabled_window"),
    ],
)
def test_without_an_active_window_everything_is_admitted(controller) -> None:
    h = CoordinatorHarness(admission_controller=controller, is_kv_manager_v2=True)
    _receiving(h, 1)
    candidates = [_candidate(2), _candidate(3)]

    admitted, blocked = h.coordinator.admit(candidates)

    assert (admitted, blocked) == (candidates, False)
    assert h.effects.reverted == []


def test_nothing_fitting_admits_nothing_without_touching_the_window() -> None:
    h = _harness(is_kv_manager_v2=True)
    _receiving(h, 1)

    assert h.coordinator.admit([]) == ([], False)
    assert h.effects.reverted == []


# -- revert_deferred_gen_init --------------------------------------------------


def test_pp_follower_reverts_local_candidates_missing_from_the_canonical_schedule() -> None:
    """A non-first PP rank schedules locally; the V2 KV it grew for candidates
    rank 0 did not admit is reverted, matched by request id."""
    h = CoordinatorHarness(is_kv_manager_v2=True)
    canonical = _candidate(1)
    local_canonical, local_only = _candidate(1), _candidate(2)

    h.coordinator.revert_deferred_gen_init([local_canonical, local_only], [canonical])

    assert h.effects.reverted == [[local_only]]


@pytest.mark.parametrize(
    "is_kv_manager_v2, candidates, admitted",
    [
        pytest.param(False, [_candidate(2)], [], id="v1_has_nothing_to_revert"),
        pytest.param(True, [], [_candidate(1)], id="no_local_candidates"),
        pytest.param(True, [_candidate(1)], [_candidate(1)], id="everything_admitted"),
    ],
)
def test_revert_is_a_no_op_without_deferred_v2_candidates(
    is_kv_manager_v2, candidates, admitted
) -> None:
    h = CoordinatorHarness(is_kv_manager_v2=is_kv_manager_v2)

    h.coordinator.revert_deferred_gen_init(candidates, admitted)

    assert h.effects.reverted == []


# -- early window predicate ---------------------------------------------------


def test_early_window_needs_the_python_runtime_with_v2_and_pp1() -> None:
    pp1, pp2 = SimpleNamespace(pp_size=1), SimpleNamespace(pp_size=2)
    async_python = SimpleNamespace(consumes_transfer_buffer=False)
    cpp = SimpleNamespace(consumes_transfer_buffer=True)

    assert early_transfer_window_eligible(async_python, pp1, True)
    assert not early_transfer_window_eligible(cpp, pp1, True)
    assert not early_transfer_window_eligible(async_python, pp1, False)
    assert not early_transfer_window_eligible(async_python, pp2, True)
    assert not early_transfer_window_eligible(None, pp1, True)


def test_early_window_does_not_assume_pp1_when_dist_lacks_pp_size() -> None:
    """Missing PP metadata must not enable local pre-allocation admission."""
    async_python = SimpleNamespace(consumes_transfer_buffer=False)

    with pytest.raises(AttributeError):
        early_transfer_window_eligible(async_python, SimpleNamespace(), True)


@pytest.mark.parametrize(
    "mode,eligible",
    [
        ("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", False),
        ("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", True),
    ],
)
def test_early_window_supports_sync_but_skips_no_context_benchmarks(
    monkeypatch, mode: str, eligible: bool
) -> None:
    monkeypatch.setenv(mode, "1")
    transceiver = SimpleNamespace(consumes_transfer_buffer=False)
    assert early_transfer_window_eligible(transceiver, SimpleNamespace(pp_size=1), True) is eligible
