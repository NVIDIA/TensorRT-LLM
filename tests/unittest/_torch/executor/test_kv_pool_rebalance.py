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

from tensorrt_llm._torch.distributed.communicator import ReduceOp
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm.runtime.kv_cache_manager_v2._exceptions import OutOfPagesError

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
) -> MagicMock:
    """Construct a MagicMock shaped like PyExecutor with exactly the
    attributes the rebalance code path reads.
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


def _make_request(req_id: int, is_dummy: bool = False) -> MagicMock:
    req = MagicMock()
    req.py_request_id = req_id
    req.is_dummy = is_dummy
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

    def test_plain_tp_size_gt_one_returns_false(self):
        # Plain TP (no attention DP) still needs bit-identical pool layouts.
        exe = _make_executor(tp_size=2)
        assert PyExecutor._can_pause_for_rebalance(exe) is False

    def test_cp_size_gt_one_returns_false(self):
        exe = _make_executor(cp_size=2)
        assert PyExecutor._can_pause_for_rebalance(exe) is False

    def test_attention_dp_with_tp_returns_true(self):
        # Attention DP keeps per-rank pools, so it is allowed and the pause
        # is coordinated collectively in _maybe_rebalance_kv_pools.
        exe = _make_executor(enable_attention_dp=True, tp_size=2)
        assert PyExecutor._can_pause_for_rebalance(exe) is True

    def test_attention_dp_with_cp_returns_false(self):
        # CP is still gated off even under attention DP.
        exe = _make_executor(enable_attention_dp=True, tp_size=2, cp_size=2)
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

    def test_dummy_requests_are_not_suspended(self, monkeypatch):
        real = _make_request(1)
        dummy = _make_request(2, is_dummy=True)
        exe = _make_executor(active_requests=[real, dummy])
        # Both ids report active; only the real one should be touched.
        exe.kv_cache_manager.is_request_active.side_effect = lambda rid: True
        exe._consume_previous_batch_for_rebalance = MagicMock()
        monkeypatch.setattr("torch.cuda.current_stream", MagicMock())

        PyExecutor._maybe_rebalance_kv_pools(exe)

        exe.kv_cache_manager.suspend_request.assert_called_once_with(real)
        exe.kv_cache_manager.resume_request.assert_called_once_with(real)


class TestMaybeRebalanceKvPoolsAttentionDP:
    """Collective coordination of the pause across attention-DP TP ranks."""

    def test_vote_yes_fires_full_cycle_and_barrier(self, monkeypatch):
        reqs = [_make_request(1), _make_request(2)]
        exe = _make_executor(
            tp_size=2, enable_attention_dp=True, need_adjustment=True, active_requests=reqs
        )
        exe.dist.tp_allreduce.return_value = 1
        exe._consume_previous_batch_for_rebalance = MagicMock()
        monkeypatch.setattr("torch.cuda.current_stream", MagicMock())

        PyExecutor._maybe_rebalance_kv_pools(exe)

        # Voted with the local need via a MAX (logical-OR) reduction.
        exe.dist.tp_allreduce.assert_called_once_with(1, op=ReduceOp.MAX)
        exe._consume_previous_batch_for_rebalance.assert_called_once()
        exe.kv_cache_manager.impl.adjust.assert_called_once()
        assert exe.kv_cache_manager.suspend_request.call_count == 2
        assert exe.kv_cache_manager.resume_request.call_count == 2
        exe.dist.tp_barrier.assert_called_once()

    def test_vote_no_is_noop(self, monkeypatch):
        reqs = [_make_request(1)]
        exe = _make_executor(
            tp_size=2, enable_attention_dp=True, need_adjustment=False, active_requests=reqs
        )
        exe.dist.tp_allreduce.return_value = 0
        exe._consume_previous_batch_for_rebalance = MagicMock()
        sync = MagicMock()
        monkeypatch.setattr("torch.cuda.current_stream", sync)

        PyExecutor._maybe_rebalance_kv_pools(exe)

        # No rank wants to adjust -> bail before any real work, no barrier.
        exe.dist.tp_allreduce.assert_called_once_with(0, op=ReduceOp.MAX)
        exe._consume_previous_batch_for_rebalance.assert_not_called()
        exe.kv_cache_manager.impl.adjust.assert_not_called()
        exe.kv_cache_manager.suspend_request.assert_not_called()
        exe.dist.tp_barrier.assert_not_called()
        sync.assert_not_called()

    def test_pauses_for_peer_without_local_adjust(self, monkeypatch):
        """A rank with no local need still pauses if a peer voted yes, but
        does not touch its own pools."""
        reqs = [_make_request(1)]
        exe = _make_executor(
            tp_size=2, enable_attention_dp=True, need_adjustment=False, active_requests=reqs
        )
        # Local need is 0 but a peer wants in -> MAX vote returns 1.
        exe.dist.tp_allreduce.return_value = 1
        exe._consume_previous_batch_for_rebalance = MagicMock()
        monkeypatch.setattr("torch.cuda.current_stream", MagicMock())

        PyExecutor._maybe_rebalance_kv_pools(exe)

        exe.dist.tp_allreduce.assert_called_once_with(0, op=ReduceOp.MAX)
        # Lockstep pause happens...
        exe._consume_previous_batch_for_rebalance.assert_called_once()
        exe.kv_cache_manager.suspend_request.assert_called_once()
        exe.kv_cache_manager.resume_request.assert_called_once()
        exe.dist.tp_barrier.assert_called_once()
        # ...but this rank does not adjust its own pools.
        exe.kv_cache_manager.impl.adjust.assert_not_called()


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
