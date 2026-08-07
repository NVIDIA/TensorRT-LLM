# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functional unit tests for the KVCacheManagerV2 rebalance hook in
PyExecutor (``_can_pause_for_rebalance``, ``_maybe_rebalance_kv_pools``,
``_consume_previous_batch_for_rebalance``).

These tests intentionally do not spin up a real PyExecutor: PyExecutor's
constructor pulls in the model engine, sampler, scheduler, KV cache
manager, distributed, etc.  Instead we follow the same pattern as
``test_py_executor.py`` and call the methods under test as unbound
attribute lookups on a ``MagicMock(spec=PyExecutor)`` with just the
fields each method reads.

The accuracy of pool rebalancing itself (i.e., that suspend/adjust/resume
preserves generated tokens) is covered by the integration accuracy test;
here we only verify the call chain and gate logic.
"""

from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm.runtime.kv_cache_manager_v2 import OutOfPagesError

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_executor(
    *,
    enable_kv_pool_rebalance: bool = True,
    pp_size: int = 1,
    tp_size: int = 1,
    enable_attention_dp: bool = False,
    kv_cache_transceiver=None,
    is_warmup: bool = False,
    is_shutdown: bool = False,
    max_beam_width: int = 1,
    drafter=None,
    need_adjustment: bool = True,
    active_requests=None,
    previous_batch=None,
    rebalance_check_interval: int = 1,
) -> MagicMock:
    """Construct a MagicMock shaped like PyExecutor with exactly the
    attributes the rebalance code path reads.

    ``rebalance_check_interval`` defaults to 1 so the iteration throttle is
    transparent for tests that are not about the throttle itself; the throttle
    gets its own coverage in ``TestRebalanceCheckThrottle``.
    """
    exe = MagicMock(spec=PyExecutor)

    # Gate inputs.
    exe.enable_kv_pool_rebalance = enable_kv_pool_rebalance
    exe.dist = MagicMock(pp_size=pp_size, tp_size=tp_size)
    exe.enable_attention_dp = enable_attention_dp
    exe.kv_cache_transceiver = kv_cache_transceiver
    exe.is_warmup = is_warmup
    exe.is_shutdown = is_shutdown
    exe.drafter = drafter

    # Iteration throttle state.
    exe._rebalance_check_interval = rebalance_check_interval
    exe._rebalance_check_counter = 0

    # Bind the real agreement helper so _maybe_rebalance_kv_pools exercises it
    # rather than getting a truthy MagicMock back.
    exe._agreed_need_adjustment = lambda: PyExecutor._agreed_need_adjustment(exe)

    # KV cache manager (resource-manager wrapper).
    exe.kv_cache_manager = MagicMock()
    exe.kv_cache_manager.max_beam_width = max_beam_width
    exe.kv_cache_manager.impl = MagicMock()
    exe.kv_cache_manager.impl.need_adjustment = need_adjustment

    # is_request_active returns True for every id we tracked, False for
    # everything else.  Tests set active_requests to a list of mocks with
    # py_request_id attributes.
    exe.active_requests = active_requests or []
    active_ids = {r.py_request_id for r in exe.active_requests}
    exe.kv_cache_manager.is_request_active.side_effect = lambda rid: rid in active_ids

    # Previous batch (overlap loop).
    exe.previous_batch = previous_batch

    return exe


def _make_request(req_id: int) -> MagicMock:
    req = MagicMock()
    req.py_request_id = req_id
    return req


# --------------------------------------------------------------------------- #
# Gate tests
# --------------------------------------------------------------------------- #


class TestCanPauseForRebalance:
    """Cover every short-circuit branch of ``_can_pause_for_rebalance``."""

    def test_default_setup_returns_true(self):
        exe = _make_executor()
        assert PyExecutor._can_pause_for_rebalance(exe) is True

    def test_flag_off_returns_false(self):
        exe = _make_executor(enable_kv_pool_rebalance=False)
        assert PyExecutor._can_pause_for_rebalance(exe) is False

    def test_pp_size_gt_one_returns_false(self):
        exe = _make_executor(pp_size=2)
        assert PyExecutor._can_pause_for_rebalance(exe) is False

    def test_transceiver_present_returns_false(self):
        exe = _make_executor(kv_cache_transceiver=MagicMock())
        assert PyExecutor._can_pause_for_rebalance(exe) is False

    def test_warmup_returns_false(self):
        exe = _make_executor(is_warmup=True)
        assert PyExecutor._can_pause_for_rebalance(exe) is False

    def test_shutdown_returns_false(self):
        exe = _make_executor(is_shutdown=True)
        assert PyExecutor._can_pause_for_rebalance(exe) is False

    def test_beam_width_gt_one_returns_false(self):
        exe = _make_executor(max_beam_width=2)
        assert PyExecutor._can_pause_for_rebalance(exe) is False

    def test_drafter_present_returns_false(self):
        exe = _make_executor(drafter=MagicMock())
        assert PyExecutor._can_pause_for_rebalance(exe) is False


# --------------------------------------------------------------------------- #
# _maybe_rebalance_kv_pools
# --------------------------------------------------------------------------- #


class TestMaybeRebalanceKvPools:
    """The hook body: synchronize -> drain -> suspend -> adjust -> resume."""

    def test_no_op_when_need_adjustment_false(self, monkeypatch):
        exe = _make_executor(need_adjustment=False, active_requests=[_make_request(1)])
        monkeypatch.setattr("torch.cuda.current_stream", MagicMock())

        PyExecutor._maybe_rebalance_kv_pools(exe)

        exe.kv_cache_manager.impl.adjust.assert_not_called()
        exe.kv_cache_manager.suspend_request.assert_not_called()
        exe.kv_cache_manager.resume_request.assert_not_called()

    def test_fires_full_cycle(self, monkeypatch):
        reqs = [_make_request(1), _make_request(2)]
        exe = _make_executor(active_requests=reqs)
        # Stub the consume helper (its own behavior is covered below).
        exe._consume_previous_batch_for_rebalance = MagicMock()
        monkeypatch.setattr("torch.cuda.current_stream", MagicMock())

        PyExecutor._maybe_rebalance_kv_pools(exe)

        exe._consume_previous_batch_for_rebalance.assert_called_once()
        exe.kv_cache_manager.impl.adjust.assert_called_once()
        assert exe.kv_cache_manager.suspend_request.call_count == 2
        assert exe.kv_cache_manager.resume_request.call_count == 2

    def test_skips_already_suspended_requests(self, monkeypatch):
        active = _make_request(1)
        suspended = _make_request(2)
        exe = _make_executor(active_requests=[active])
        exe.active_requests = [active, suspended]
        # Override side_effect: only req 1 is active on GPU.
        exe.kv_cache_manager.is_request_active.side_effect = lambda rid: rid == 1
        exe._consume_previous_batch_for_rebalance = MagicMock()
        monkeypatch.setattr("torch.cuda.current_stream", MagicMock())

        PyExecutor._maybe_rebalance_kv_pools(exe)

        # Only the active request was suspended and resumed.
        exe.kv_cache_manager.suspend_request.assert_called_once_with(active)
        exe.kv_cache_manager.resume_request.assert_called_once_with(active)

    def test_expected_adjust_failure_does_not_skip_resume(self, monkeypatch, caplog):
        """OutOfPagesError from adjust() is the one expected runtime failure.

        It must be swallowed so paused requests are still resumed.
        """
        reqs = [_make_request(1)]
        exe = _make_executor(active_requests=reqs)
        exe._consume_previous_batch_for_rebalance = MagicMock()
        exe.kv_cache_manager.impl.adjust.side_effect = OutOfPagesError("boom")
        monkeypatch.setattr("torch.cuda.current_stream", MagicMock())

        # Should not raise.
        PyExecutor._maybe_rebalance_kv_pools(exe)

        exe.kv_cache_manager.suspend_request.assert_called_once()
        exe.kv_cache_manager.resume_request.assert_called_once()

    def test_unexpected_adjust_failure_propagates(self, monkeypatch):
        """Any non-OutOfPagesError (programmer bug) must propagate.

        Such errors fail fast rather than being downgraded to a warning.
        """
        reqs = [_make_request(1)]
        exe = _make_executor(active_requests=reqs)
        exe._consume_previous_batch_for_rebalance = MagicMock()
        exe.kv_cache_manager.impl.adjust.side_effect = RuntimeError("boom")
        monkeypatch.setattr("torch.cuda.current_stream", MagicMock())

        with pytest.raises(RuntimeError, match="boom"):
            PyExecutor._maybe_rebalance_kv_pools(exe)


# --------------------------------------------------------------------------- #
# _consume_previous_batch_for_rebalance
# --------------------------------------------------------------------------- #


class TestConsumePreviousBatch:
    """Overlap-mode drain helper."""

    def test_no_op_when_previous_batch_none(self):
        exe = _make_executor(previous_batch=None)
        PyExecutor._consume_previous_batch_for_rebalance(exe)
        exe._update_requests.assert_not_called()
        exe._send_kv_async.assert_not_called()
        exe._flush_pending_transfer_responses.assert_not_called()
        exe._process_previous_batch.assert_not_called()

    def test_consumes_and_clears(self):
        prev = MagicMock()
        prev.sample_state = MagicMock()
        prev.scheduled_requests.all_requests.return_value = [_make_request(1)]
        exe = _make_executor(previous_batch=prev)
        # perf_manager needs compute_batch_gpu_times.
        exe.perf_manager = MagicMock()

        PyExecutor._consume_previous_batch_for_rebalance(exe)

        exe._update_requests.assert_called_once_with(prev.sample_state)
        exe._send_kv_async.assert_called_once()
        exe._flush_pending_transfer_responses.assert_called_once()
        exe._process_previous_batch.assert_called_once()
        exe.perf_manager.compute_batch_gpu_times.assert_called_once()
        assert exe.previous_batch is None


# --------------------------------------------------------------------------- #
# Iteration throttle
# --------------------------------------------------------------------------- #


class TestRebalanceCheckThrottle:
    """``_can_pause_for_rebalance`` only lets a check through every N iterations.

    The throttle must be driven purely by rank-uniform state: all ranks have to
    reach the agreement collective on the same iteration or they deadlock.
    """

    def test_returns_false_until_interval_reached(self):
        exe = _make_executor(rebalance_check_interval=4)
        results = [PyExecutor._can_pause_for_rebalance(exe) for _ in range(4)]
        assert results == [False, False, False, True]

    def test_counter_resets_after_firing(self):
        exe = _make_executor(rebalance_check_interval=3)
        results = [PyExecutor._can_pause_for_rebalance(exe) for _ in range(6)]
        assert results == [False, False, True, False, False, True]

    def test_interval_of_one_checks_every_iteration(self):
        exe = _make_executor(rebalance_check_interval=1)
        assert all(PyExecutor._can_pause_for_rebalance(exe) for _ in range(5))

    def test_counter_not_advanced_when_a_gate_rejects(self):
        # The counter sits behind every gate, so an executor that cannot
        # rebalance accumulates no throttle credit.  Every gate is rank-uniform,
        # which is what keeps the counter -- and therefore the iteration the
        # agreement collective runs on -- identical across ranks.
        exe = _make_executor(enable_kv_pool_rebalance=False, rebalance_check_interval=2)
        for _ in range(10):
            assert PyExecutor._can_pause_for_rebalance(exe) is False
        assert exe._rebalance_check_counter == 0

    def test_counter_not_advanced_during_warmup(self):
        exe = _make_executor(is_warmup=True, rebalance_check_interval=2)
        for _ in range(10):
            assert PyExecutor._can_pause_for_rebalance(exe) is False
        assert exe._rebalance_check_counter == 0

        # Once warmup clears, the throttle starts counting from scratch.
        exe.is_warmup = False
        assert [PyExecutor._can_pause_for_rebalance(exe) for _ in range(2)] == [False, True]


# --------------------------------------------------------------------------- #
# Cross-rank agreement on the rebalance trigger
# --------------------------------------------------------------------------- #


class TestAgreedNeedAdjustment:
    """Rank 0 of the TP group decides; the decision is broadcast.

    Rationale: every input to ``need_adjustment`` is deterministic given the
    request stream except the 120s cooldown, which reads a per-rank
    ``steady_clock``.  Two TP ranks straddling that boundary would rebalance on
    different iterations, and TP ranks schedule independently, so their batches
    could then diverge.
    """

    def test_single_rank_reads_locally_without_broadcast(self):
        exe = _make_executor(tp_size=1, need_adjustment=True)
        assert PyExecutor._agreed_need_adjustment(exe) is True
        exe.dist.tp_broadcast.assert_not_called()

    def test_tp_broadcasts_rank0_decision(self):
        exe = _make_executor(tp_size=4, need_adjustment=True)
        exe.dist.tp_broadcast.return_value = True

        assert PyExecutor._agreed_need_adjustment(exe) is True
        exe.dist.tp_broadcast.assert_called_once_with(True, root=0)

    def test_tp_rank_follows_broadcast_over_its_own_reading(self):
        # This is the case the whole mechanism exists for: the local read says
        # "rebalance" but rank 0 says no, so this rank must not rebalance.
        exe = _make_executor(tp_size=2, need_adjustment=True)
        exe.dist.tp_broadcast.return_value = False

        assert PyExecutor._agreed_need_adjustment(exe) is False

    def test_tp_rank_follows_broadcast_when_local_says_no(self):
        exe = _make_executor(tp_size=2, need_adjustment=False)
        exe.dist.tp_broadcast.return_value = True

        assert PyExecutor._agreed_need_adjustment(exe) is True

    def test_attention_dp_decides_independently(self):
        # ADP ranks own independent request streams and independent KV caches,
        # so forcing rank 0's decision on them would starve a rank that needs
        # to rebalance when rank 0 does not.
        exe = _make_executor(tp_size=4, enable_attention_dp=True, need_adjustment=True)
        assert PyExecutor._agreed_need_adjustment(exe) is True
        exe.dist.tp_broadcast.assert_not_called()

    def test_no_rebalance_when_agreement_says_no(self, monkeypatch):
        exe = _make_executor(tp_size=2, need_adjustment=True, active_requests=[_make_request(1)])
        exe.dist.tp_broadcast.return_value = False
        monkeypatch.setattr("torch.cuda.current_stream", MagicMock())

        PyExecutor._maybe_rebalance_kv_pools(exe)

        exe.kv_cache_manager.impl.adjust.assert_not_called()
        exe.kv_cache_manager.suspend_request.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
