# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transfer-window admission: which scheduler-fitting gen-init requests may
start receiving this iteration, and the scheduler V2 allocation revert for
the ones that may not.

Admission is rank-local (rank 0 decides under PP, every rank for itself under
attention DP), so every case runs one ``CoordinatorHarness``.
"""

from types import SimpleNamespace

import pytest
from coordinator_harness import CoordinatorHarness, TransferRequest

from tensorrt_llm._torch.disaggregation.orchestration.admission import (
    DisaggTransferAdmissionController,
)
from tensorrt_llm._torch.disaggregation.orchestration.coordinator import (
    transfer_window_bypass_eligible,
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


def test_async_python_v2_pp1_bypasses_the_window() -> None:
    """The asynchronous Python transceiver does not consume the C++ transfer
    buffer; with KV cache manager V2 and PP1 the scheduler's own KV admission
    bounds it, so the executor-level window is skipped."""
    h = _harness(is_kv_manager_v2=True, consumes_transfer_buffer=False)
    _receiving(h, 1)
    candidates = [_candidate(2), _candidate(3)]

    admitted, blocked = h.coordinator.admit(candidates)

    assert (admitted, blocked) == (candidates, False)
    assert h.effects.reverted == []


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


# -- window bypass predicate ---------------------------------------------------


def test_window_bypass_needs_the_async_python_runtime_with_v2_and_pp1() -> None:
    pp1, pp2 = SimpleNamespace(pp_size=1), SimpleNamespace(pp_size=2)
    async_python = SimpleNamespace(consumes_transfer_buffer=False)
    cpp = SimpleNamespace(consumes_transfer_buffer=True)

    assert transfer_window_bypass_eligible(async_python, pp1, True)
    assert not transfer_window_bypass_eligible(cpp, pp1, True)
    assert not transfer_window_bypass_eligible(async_python, pp1, False)
    assert not transfer_window_bypass_eligible(async_python, pp2, True)
    assert not transfer_window_bypass_eligible(None, pp1, True)


def test_window_bypass_does_not_assume_pp1_when_dist_lacks_pp_size() -> None:
    """A dist without ``pp_size`` must not silently count as PP1: bypassing
    the window on a misconfigured executor would drop a real budget."""
    async_python = SimpleNamespace(consumes_transfer_buffer=False)

    with pytest.raises(AttributeError):
        transfer_window_bypass_eligible(async_python, SimpleNamespace(), True)
