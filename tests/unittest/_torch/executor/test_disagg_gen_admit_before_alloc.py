# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRTLLM_DISAGG_GEN_ADMIT_BEFORE_ALLOC: the KV-transfer window applied before KV allocation.

Without the switch, KVCacheV2Scheduler allocates KV for every DISAGG_GENERATION_INIT request,
DisaggTransferAdmissionController.select() admits the FCFS prefix that fits the transfer budget, and
the coordinator reverts the rest. With the switch, a DisaggTransferAdmissionGate closes at the first
candidate that misses the budget and the scheduler skips the remaining generation-init requests
without allocating them; the switch also keeps the window on where the coordinator would bypass it.
The scheduler path is tested with the V2 scheduler's own factories in
kv_cache/test_kv_cache_v2_scheduler.py (TestDisaggAdmitBeforeAlloc).

No GPU required.
"""

import random
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.disaggregation.orchestration.admission import (
    ADMIT_BEFORE_ALLOC_ENV,
    KEEP_TRANSFER_WINDOW_ENV,
    DisaggTransferAdmissionController,
    DisaggTransferAdmissionGate,
)
from tensorrt_llm._torch.disaggregation.orchestration.coordinator import (
    transfer_window_bypass_eligible,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        ADMIT_BEFORE_ALLOC_ENV,
        KEEP_TRANSFER_WINDOW_ENV,
        "TRTLLM_DISAGG_BENCHMARK_GEN_ONLY",
        "TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP",
    ):
        monkeypatch.delenv(name, raising=False)


def ids(reqs):
    return [r.request_id for r in reqs]


# ---------------------------------------------------------------------------
# Gate vs select(): same admitted prefix, closes at the first deferred candidate
# ---------------------------------------------------------------------------
class TestGateMatchesSelect:
    @staticmethod
    def _req(i, prompt_len, in_progress=False):
        return SimpleNamespace(
            request_id=i,
            py_request_id=i,
            total_input_len_cp=prompt_len,
            py_prompt_len=prompt_len,
            prompt_len=prompt_len,
            is_disagg_generation_transmission_in_progress=in_progress,
        )

    def test_disabled_budget_has_no_gate(self):
        controller = DisaggTransferAdmissionController(
            max_tokens_in_buffer=None, tokens_per_block=64
        )
        assert controller.begin_gate([]) is None

    @pytest.mark.parametrize("seed", range(400))
    def test_randomized_equivalence(self, seed):
        rng = random.Random(seed)
        tpb = rng.choice([16, 32, 64])
        controller = DisaggTransferAdmissionController(
            max_tokens_in_buffer=rng.randint(1, 4000), tokens_per_block=tpb
        )
        active = [
            self._req(1000 + k, rng.randint(1, 2500), in_progress=rng.random() < 0.7)
            for k in range(rng.randint(0, 3))
        ]
        candidates = [self._req(k, rng.randint(0, 3000)) for k in range(rng.randint(0, 12))]

        reference = controller.select(active, candidates)

        gate = controller.begin_gate(active)
        observed, skipped = [], []
        for c in candidates:
            if gate.closed:
                skipped.append(c)
                continue
            observed.append(c)
            gate.observe_candidate(c)
        gated = controller.select(active, observed)

        assert ids(gated.admitted_requests) == ids(reference.admitted_requests)
        assert gated.limited_by_budget == reference.limited_by_budget
        assert gated.is_blocked_by_active_transfers() == reference.is_blocked_by_active_transfers()
        assert gated.active_transfer_blocks == reference.active_transfer_blocks
        assert gated.admitted_transfer_blocks == reference.admitted_transfer_blocks
        # Only the candidate that closed the gate is allocated and then deferred.
        assert gated.deferred_request_count == (1 if reference.deferred_request_count else 0)
        assert len(skipped) == max(reference.deferred_request_count - 1, 0)


# ---------------------------------------------------------------------------
# Transfer window: either switch keeps it where the coordinator would bypass it
# ---------------------------------------------------------------------------
class TestTransferWindow:
    @staticmethod
    def _bypass():
        transceiver = SimpleNamespace(consumes_transfer_buffer=False)
        dist = SimpleNamespace(pp_size=1)
        return transfer_window_bypass_eligible(transceiver, dist, True)

    def test_bypassed_without_the_switch(self):
        assert self._bypass()

    @pytest.mark.parametrize("switch", [ADMIT_BEFORE_ALLOC_ENV, KEEP_TRANSFER_WINDOW_ENV])
    def test_kept_with_either_switch(self, monkeypatch, switch):
        monkeypatch.setenv(switch, "1")
        assert not self._bypass()

    def test_window_alone_has_no_gate(self, monkeypatch):
        # The window without the gate is the allocate-then-revert admission.
        monkeypatch.setenv(KEEP_TRANSFER_WINDOW_ENV, "1")
        assert PyExecutor._make_disagg_transfer_admission_gate(_make_executor()) is None


# ---------------------------------------------------------------------------
# Executor wiring
# ---------------------------------------------------------------------------
def _make_executor(max_tokens_in_buffer=128, tokens_per_block=64):
    executor = object.__new__(PyExecutor)
    executor.kv_cache_transceiver = Mock()
    executor._disagg_transfer_admission_controller = DisaggTransferAdmissionController(
        max_tokens_in_buffer=max_tokens_in_buffer, tokens_per_block=tokens_per_block
    )
    executor.active_requests = []
    return executor


class TestExecutorWiring:
    def test_switch_off_by_default(self):
        assert PyExecutor._make_disagg_transfer_admission_gate(_make_executor()) is None

    def test_switch_on_returns_gate(self, monkeypatch):
        monkeypatch.setenv(ADMIT_BEFORE_ALLOC_ENV, "1")
        gate = PyExecutor._make_disagg_transfer_admission_gate(_make_executor())
        assert isinstance(gate, DisaggTransferAdmissionGate)
        assert not gate.closed

    def test_no_gate_without_transceiver(self, monkeypatch):
        monkeypatch.setenv(ADMIT_BEFORE_ALLOC_ENV, "1")
        executor = _make_executor()
        executor.kv_cache_transceiver = None
        assert PyExecutor._make_disagg_transfer_admission_gate(executor) is None

    def test_no_gate_when_budget_disabled(self, monkeypatch):
        monkeypatch.setenv(ADMIT_BEFORE_ALLOC_ENV, "1")
        executor = _make_executor(max_tokens_in_buffer=None)
        assert PyExecutor._make_disagg_transfer_admission_gate(executor) is None

    def test_no_gate_for_gen_only_no_context_benchmark(self, monkeypatch):
        # That mode skips the transfer window in DisaggTransferCoordinator.admit.
        monkeypatch.setenv(ADMIT_BEFORE_ALLOC_ENV, "1")
        monkeypatch.setenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", "1")
        assert PyExecutor._make_disagg_transfer_admission_gate(_make_executor()) is None

    @pytest.mark.parametrize("wrapped", [False, True], ids=["v2", "multimodal_wrapper"])
    @pytest.mark.parametrize("switch", [False, True])
    def test_schedule_installs_and_clears_gate(self, monkeypatch, switch, wrapped):
        if switch:
            monkeypatch.setenv(ADMIT_BEFORE_ALLOC_ENV, "1")
        executor = _make_executor()
        executor.kv_cache_manager = SimpleNamespace()
        executor.inflight_req_ids = set()
        executor.is_encoder_decoder = False
        executor.enable_attention_dp = False
        executor.enable_batch_waiting = False
        executor._maybe_record_hang_diagnostic_phase = lambda *args, **kwargs: None
        executor._cap_context_by_total_kv_len = lambda reqs: reqs

        seen = {}

        class _Scheduler:
            gate = None

            def set_disagg_transfer_admission_gate(self, gate):
                self.gate = gate

            def schedule_request(self, active_requests, inflight_request_ids):
                seen["gate"] = self.gate
                return SimpleNamespace(
                    encoder_requests=[],
                    context_requests=[],
                    generation_requests=[],
                    paused_requests=[],
                    scheduled_mm_encoder_items=[],
                    recompute_paused_requests=[],
                    fitting_disagg_gen_init_requests=[],
                    num_fitting_requests=0,
                )

        inner = _Scheduler()
        if wrapped:
            # MultimodalScheduler: the LLM scheduler as .scheduler, schedule_request forwarded.
            class _Wrapper:
                scheduler = inner

                def schedule_request(self, active_requests, inflight_request_ids):
                    return inner.schedule_request(active_requests, inflight_request_ids)

            executor.scheduler = _Wrapper()
        else:
            executor.scheduler = inner
        PyExecutor._schedule(executor)

        assert isinstance(seen["gate"], DisaggTransferAdmissionGate) == switch
        assert inner.gate is None
