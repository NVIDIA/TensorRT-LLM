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
    cp_size: int = 1,
    enable_attention_dp: bool = False,
    kv_cache_transceiver=None,
    is_warmup: bool = False,
    is_shutdown: bool = False,
    max_beam_width: int = 1,
    drafter=None,
    need_adjustment: bool = True,
    active_requests=None,
    previous_batch=None,
    padding_dummies=None,
    has_cuda_graph_runner: bool = True,
    rebalance_check_interval: int = 1,
) -> MagicMock:
    """Construct a MagicMock shaped like PyExecutor with exactly the
    attributes the rebalance code path reads.

    ``padding_dummies`` is the runner's ``{draft_len: dummy}`` map; its
    dummies count as active on GPU, like real pre-allocated ones.

    ``rebalance_check_interval`` defaults to 1 so the iteration throttle is
    transparent for tests that are not about the throttle itself; the throttle
    gets its own coverage in ``TestRebalanceCheckThrottle``.
    """
    exe = MagicMock(spec=PyExecutor)

    # Gate inputs.
    exe.enable_kv_pool_rebalance = enable_kv_pool_rebalance
    exe.dist = MagicMock(pp_size=pp_size, tp_size=tp_size, cp_size=cp_size)
    exe.enable_attention_dp = enable_attention_dp
    exe.kv_cache_transceiver = kv_cache_transceiver
    exe.is_warmup = is_warmup
    exe.is_shutdown = is_shutdown
    exe.drafter = drafter

    # Iteration throttle state.  The throttle keys on iter_counter, the
    # loop-wide iteration index, so tests drive that directly.
    exe._rebalance_check_interval = rebalance_check_interval
    exe.iter_counter = 0

    # Bind the real agreement helper so _maybe_rebalance_kv_pools exercises it
    # rather than getting a truthy MagicMock back.
    exe._agreed_need_adjustment = lambda: PyExecutor._agreed_need_adjustment(exe)

    # KV cache manager (resource-manager wrapper).
    exe.kv_cache_manager = MagicMock()
    exe.kv_cache_manager.max_beam_width = max_beam_width
    exe.kv_cache_manager.impl = MagicMock()
    exe.kv_cache_manager.impl.need_adjustment = need_adjustment

    # CUDA-graph runner holding the pre-allocated padding dummies.  A bare
    # MagicMock would be truthy and non-iterable in the suspend helper.
    padding_dummies = dict(padding_dummies or {})
    exe.model_engine = MagicMock()
    exe.resource_manager = MagicMock()
    if has_cuda_graph_runner:
        exe.model_engine.cuda_graph_runner = MagicMock()
        exe.model_engine.cuda_graph_runner.padding_dummy_requests = padding_dummies
        # Mirror the real helper: pop the dummy from the runner's map.
        exe.model_engine.cuda_graph_runner.release_padding_dummy.side_effect = (
            lambda _rm, draft_len: padding_dummies.pop(draft_len, None) is not None
        )
    else:
        exe.model_engine.cuda_graph_runner = None

    # is_request_active returns True for every id we tracked, False for
    # everything else.  Tests set active_requests to a list of mocks with
    # py_request_id attributes.
    exe.active_requests = active_requests or []
    active_ids = {r.py_request_id for r in exe.active_requests}
    active_ids |= {d.py_request_id for d in padding_dummies.values()}
    exe.kv_cache_manager.is_request_active.side_effect = lambda rid: rid in active_ids
    exe.kv_cache_manager.resume_request.return_value = True

    # Bind the padding-dummy helpers to their real implementations so that
    # _maybe_rebalance_kv_pools tests exercise the actual path rather than
    # MagicMock stand-ins supplied by spec=PyExecutor.
    exe._suspend_padding_dummies_for_rebalance = (
        lambda mgr: PyExecutor._suspend_padding_dummies_for_rebalance(exe, mgr)
    )
    exe._resume_padding_dummies_after_rebalance = (
        lambda mgr, suspended: PyExecutor._resume_padding_dummies_after_rebalance(
            exe, mgr, suspended
        )
    )

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
# CUDA-graph padding dummies
# --------------------------------------------------------------------------- #


class TestPaddingDummies:
    """``adjust()`` requires every living KV cache to be suspended, but the
    CUDA-graph padding dummies stay ACTIVE across iterations and never appear
    in ``active_requests``.  The hook must suspend them too -- and must not
    free them, since they are pre-allocated at warmup precisely so that a
    loaded KV cache cannot deny them later (PR #16072).
    """

    @staticmethod
    def _fire(exe, monkeypatch):
        exe._consume_previous_batch_for_rebalance = MagicMock()
        monkeypatch.setattr("torch.cuda.current_stream", MagicMock())
        PyExecutor._maybe_rebalance_kv_pools(exe)

    def test_padding_dummy_is_suspended_before_adjust(self, monkeypatch):
        dummy = _make_request(999)
        exe = _make_executor(active_requests=[_make_request(1)], padding_dummies={0: dummy})

        call_order = []
        exe.kv_cache_manager.suspend_request.side_effect = lambda req: call_order.append(
            ("suspend", req.py_request_id)
        )
        exe.kv_cache_manager.impl.adjust.side_effect = lambda: call_order.append(("adjust", None))

        self._fire(exe, monkeypatch)

        assert call_order.index(("suspend", 999)) < call_order.index(("adjust", None))

    def test_padding_dummy_is_resumed_and_never_freed(self, monkeypatch):
        """The dummy must survive the rebalance: freeing it would return to
        the lazy re-allocation that #16072 removed, which can fail against a
        loaded cache and drop padded batches to eager mode permanently.
        """
        dummy = _make_request(999)
        runner_map = {0: dummy}
        exe = _make_executor(padding_dummies=runner_map)

        self._fire(exe, monkeypatch)

        exe.kv_cache_manager.resume_request.assert_called_once_with(dummy)
        exe.kv_cache_manager.free_resources.assert_not_called()
        assert exe.model_engine.cuda_graph_runner.padding_dummy_requests == {0: dummy}

    def test_every_captured_draft_length_is_covered(self, monkeypatch):
        """#16072 retains one dummy per captured draft length, not just one."""
        dummies = {0: _make_request(999), 3: _make_request(996)}
        exe = _make_executor(padding_dummies=dummies)

        self._fire(exe, monkeypatch)

        suspended = {
            c.args[0].py_request_id for c in exe.kv_cache_manager.suspend_request.call_args_list
        }
        assert suspended == {999, 996}

    def test_unresumable_padding_dummy_is_released_and_dropped(self, monkeypatch):
        """Nothing reschedules a padding dummy, and the runner hands back a
        cached dummy without checking that its cache is live -- so one that
        cannot be resumed must not be left suspended in the runner's map.

        The release goes through the runner rather than the KV cache manager
        directly: a dummy is registered with up to four managers, and the
        runner owns that list.
        """
        dummy = _make_request(999)
        exe = _make_executor(padding_dummies={0: dummy})
        exe.kv_cache_manager.resume_request.return_value = False
        runner = exe.model_engine.cuda_graph_runner

        self._fire(exe, monkeypatch)

        runner.release_padding_dummy.assert_called_once_with(exe.resource_manager, 0)
        exe.kv_cache_manager.free_resources.assert_not_called()

    def test_already_suspended_padding_dummy_is_left_alone(self, monkeypatch):
        dummy = _make_request(999)
        exe = _make_executor(padding_dummies={0: dummy})
        exe.kv_cache_manager.is_request_active.side_effect = lambda rid: False

        self._fire(exe, monkeypatch)

        exe.kv_cache_manager.suspend_request.assert_not_called()
        exe.kv_cache_manager.resume_request.assert_not_called()

    def test_missing_cuda_graph_runner_is_tolerated(self, monkeypatch):
        """Engines without a CUDA-graph runner must still rebalance."""
        exe = _make_executor(active_requests=[_make_request(1)], has_cuda_graph_runner=False)

        self._fire(exe, monkeypatch)

        exe.kv_cache_manager.impl.adjust.assert_called_once()


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

    The throttle keys on ``iter_counter``, the loop-wide iteration index, rather
    than on a counter of its own.  That makes the set of iterations a rank fires
    the agreement collective on a pure function of the iteration number, so it
    cannot drift between ranks.
    """

    @staticmethod
    def _fired_on(exe: MagicMock, iterations: int) -> list[int]:
        """Run ``iterations`` executor iterations and report which ones opened
        the gate.

        Args:
            exe: The PyExecutor stand-in to drive.
            iterations: How many iterations to step through.

        Returns:
            The ``iter_counter`` values on which the gate returned ``True``.
        """
        fired = []
        for i in range(iterations):
            exe.iter_counter = i
            if PyExecutor._can_pause_for_rebalance(exe):
                fired.append(i)
        return fired

    def test_fires_once_per_interval(self) -> None:
        exe = _make_executor(rebalance_check_interval=4)
        assert self._fired_on(exe, 12) == [0, 4, 8]

    def test_interval_of_one_checks_every_iteration(self) -> None:
        exe = _make_executor(rebalance_check_interval=1)
        assert self._fired_on(exe, 5) == [0, 1, 2, 3, 4]

    def test_config_gate_suppresses_every_iteration(self) -> None:
        exe = _make_executor(enable_kv_pool_rebalance=False, rebalance_check_interval=2)
        assert self._fired_on(exe, 10) == []

    def test_gate_rejection_does_not_shift_the_schedule(self) -> None:
        """Regression guard for the cadence-drift deadlock.

        A throttle counting its own eligible iterations would only advance on
        iterations that cleared every gate, so a rank that bailed out early even
        once would fire on a different set of iterations than its peers from
        then on -- and the two would meet ``_agreed_need_adjustment``'s
        broadcast out of step.  Keying on ``iter_counter`` makes the schedule
        independent of that history: a rank that skips a check rejoins the
        common cadence instead of being permanently offset from it.
        """
        interval = 4
        iterations = 16
        # A peer rank that never bails out: the cadence everyone must share.
        reference = self._fired_on(_make_executor(rebalance_check_interval=interval), iterations)

        # This rank alone is briefly shut down, over a window that straddles one
        # of its firing iterations.
        window = (3, 4, 5)
        exe = _make_executor(rebalance_check_interval=interval)
        fired = []
        for i in range(iterations):
            exe.iter_counter = i
            exe.is_shutdown = i in window
            if PyExecutor._can_pause_for_rebalance(exe):
                fired.append(i)

        # The property that matters: this rank may *miss* checks, but it must
        # never fire on an iteration its peers do not.  Anything else means it
        # carried a lasting offset out of the window and would meet them at
        # _agreed_need_adjustment's broadcast out of step.
        assert set(fired) <= set(reference), (
            f"rank drifted off the shared cadence: fired on {fired}, peers fire "
            f"on {reference}; the extra iterations {sorted(set(fired) - set(reference))} "
            "would enter the agreement collective alone"
        )
        # And it really did skip the check inside the window (not vacuous).
        assert [i for i in reference if i not in fired] == [4]
        assert fired == [0, 8, 12]


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

    def test_cp_broadcasts_even_when_tp_size_is_one(self):
        # A pure-CP job has mapping.tp_size == 1, so keying the agreement on
        # tp_size alone would leave CP unsynchronized -- yet a request is split
        # across CP ranks, and CP runs on the same executor loops as TP (loop
        # choice keys only on pp_size), so CP ranks also schedule independently.
        exe = _make_executor(tp_size=1, cp_size=4, need_adjustment=True)
        exe.dist.cp_broadcast.return_value = False

        assert PyExecutor._agreed_need_adjustment(exe) is False
        exe.dist.cp_broadcast.assert_called_once_with(True, root=0)
        exe.dist.tp_broadcast.assert_not_called()

    def test_tp_and_cp_chain_propagates_global_rank0(self):
        # CP first, then TP: after the CP step a rank holds V(its tp_rank, cp0),
        # and the TP step replaces it with V(tp0, cp0) -- global rank 0's value.
        #
        # The local reading and the CP result are deliberately *different* here.
        # If they matched, the final assertion would pass whether or not the TP
        # step actually consumes the CP step's result, and the chaining -- the
        # whole point of this test -- would go unverified.
        exe = _make_executor(tp_size=2, cp_size=2, need_adjustment=True)
        exe.dist.cp_broadcast.return_value = False
        exe.dist.tp_broadcast.return_value = False

        assert PyExecutor._agreed_need_adjustment(exe) is False
        exe.dist.cp_broadcast.assert_called_once_with(True, root=0)
        # Called with the CP result (False), not the local reading (True).
        exe.dist.tp_broadcast.assert_called_once_with(False, root=0)

    def test_single_rank_touches_no_collective(self):
        exe = _make_executor(tp_size=1, cp_size=1, need_adjustment=True)
        assert PyExecutor._agreed_need_adjustment(exe) is True
        exe.dist.cp_broadcast.assert_not_called()
        exe.dist.tp_broadcast.assert_not_called()

    def test_attention_dp_still_broadcasts_over_cp(self):
        """ADP suppresses the TP hop only -- CP ranks must still agree.

        Under ADP the TP dimension is the DP dimension, so those ranks own
        independent request streams and decide independently.  But CP is
        orthogonal: inside one DP replica the CP ranks split the *same* request
        along the sequence dimension, so they have to admit it together.
        Skipping the CP hop here would reintroduce the divergence this whole
        mechanism exists to remove, once per replica.  This mirrors the
        scheduler's own propagation, which gates ``tp_broadcast`` on
        ``not enable_attention_dp`` but runs ``cp_broadcast`` unconditionally.
        """
        exe = _make_executor(tp_size=2, cp_size=2, enable_attention_dp=True, need_adjustment=True)
        exe.dist.cp_broadcast.return_value = False

        # The CP result wins over this rank's own reading...
        assert PyExecutor._agreed_need_adjustment(exe) is False
        exe.dist.cp_broadcast.assert_called_once_with(True, root=0)
        # ...but the TP hop stays suppressed, so replicas remain independent.
        exe.dist.tp_broadcast.assert_not_called()

    def test_attention_dp_without_cp_touches_no_collective(self):
        exe = _make_executor(tp_size=2, cp_size=1, enable_attention_dp=True, need_adjustment=True)
        assert PyExecutor._agreed_need_adjustment(exe) is True
        exe.dist.cp_broadcast.assert_not_called()
        exe.dist.tp_broadcast.assert_not_called()

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
