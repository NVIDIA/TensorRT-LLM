# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""KV recompute after an in-flight weight update under PD disaggregation.

``PyExecutor.recompute_active_requests`` terminates and pauses every active
request so its KV cache is rebuilt with the new weights. On a context engine
that also swept up context-only requests whose blocks were pinned for (or
about to enter) the KV transfer to the generation engine; the next
``_send_kv_async`` then looked the freed sequence up in the KV cache manager
and the executor loop died with ``IndexError: unordered_map::at``. These tests
pin the guards added for that:

* transfer-bound requests are skipped by the recompute;
* on a disaggregated engine every context-only and generation-only request
  is left alone (the recompute reduces to the reuse-tree reset);
* a failing KV hand-over marks the request as failed (for the existing
  rank-safe error paths) instead of killing the executor loop.
"""

from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.pyexecutor.py_executor import AsyncTransferManager, PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm._torch.disaggregation.orchestration.coordinator import (
    DisaggTransferCoordinator)


def _request(request_id: int, *, state=LlmRequestState.GENERATION_IN_PROGRESS,
             is_context_only: bool = False, is_generation_only: bool = False):
    request = MagicMock()
    request.py_request_id = request_id
    request.state = state
    request.is_context_only_request = is_context_only
    request.is_generation_only_request.return_value = is_generation_only
    request.is_context_finished = True
    request.is_finished_due_to_length = False
    request.is_finished_due_to_cancellation = False
    return request


class _RecomputeExecutor:
    """Minimal stand-in for PyExecutor exposing what recompute_active_requests uses."""

    _DISAGG_TRANSFER_BOUND_STATES = PyExecutor._DISAGG_TRANSFER_BOUND_STATES
    _is_disagg_transfer_bound = PyExecutor._is_disagg_transfer_bound

    def __init__(self, active_requests, *, transceiver, transfer_manager,
                 previous_batch=None):
        self.active_requests = list(active_requests)
        self.kv_cache_transceiver = transceiver
        self.async_transfer_manager = transfer_manager
        # recompute runs at a control-action boundary: no in-flight batch.
        self.dist = MagicMock(pp_size=1)
        self.previous_batch = previous_batch
        self.terminated = []
        self.paused = []
        self.reset_calls = 0

    def _consume_previous_batch_for_rebalance(self):
        # Must never be reached: it gathers under attention-DP.
        raise AssertionError("recompute must not consume the in-flight batch")

    def _terminate_requests(self, requests):
        self.terminated.extend(requests)

    def _pause_requests(self, requests):
        self.paused.extend(requests)

    def reset_prefix_cache(self):
        self.reset_calls += 1


def _transfer_manager():
    resource_manager = MagicMock()
    kv_cache_manager = MagicMock()
    kv_cache_manager.store_blocks_for_reuse.return_value = 7
    resource_manager.resource_managers = {
        ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager,
    }
    return AsyncTransferManager(resource_manager), kv_cache_manager


class TestRecomputeSkipsTransferBoundRequests:

    def test_aggregated_engine_recomputes_everything(self):
        """Without a transceiver (aggregated engine) nothing is skipped."""
        reqs = [_request(1), _request(2, state=LlmRequestState.CONTEXT_INIT)]
        ex = _RecomputeExecutor(reqs, transceiver=None, transfer_manager=None)

        PyExecutor.recompute_active_requests(ex)

        assert ex.terminated == reqs
        assert ex.paused == reqs
        assert ex.reset_calls == 1

    def test_in_flight_batch_is_a_precondition_violation(self):
        """The control action retires the overlap batch first; recompute must
        never consume it itself (that path gathers on some ADP ranks only)."""
        ex = _RecomputeExecutor([_request(3)], transceiver=None, transfer_manager=None,
                                previous_batch=MagicMock())

        with pytest.raises(RuntimeError, match="quiescent"):
            PyExecutor.recompute_active_requests(ex)
        assert ex.terminated == []

    def test_context_only_requests_are_left_alone(self):
        """On a context engine every context-only request is transfer-bound.

        This is deliberately broader than "in transfer": a context-only
        request that has not even started its prefill is skipped too, so the
        ctx engine's recompute reduces to the reuse-tree reset.
        """
        manager, _ = _transfer_manager()
        unstarted_ctx = _request(10, state=LlmRequestState.CONTEXT_INIT, is_context_only=True)
        in_transfer = _request(11, is_context_only=True)
        manager.start_transfer(in_transfer)  # -> DISAGG_CONTEXT_TRANS_IN_PROGRESS, pinned
        ex = _RecomputeExecutor([unstarted_ctx, in_transfer],
                                transceiver=MagicMock(), transfer_manager=manager)

        PyExecutor.recompute_active_requests(ex)

        assert ex.terminated == []
        assert ex.paused == []
        # The reuse index is still cleared: it is the only stale state a ctx engine owns.
        assert ex.reset_calls == 1
        assert 11 in manager.requests_in_transfer()

    def test_generation_engine_leaves_generation_only_requests_alone(self, capfd):
        """On a generation engine every generation-only request is left as is,
        whether its cache is still arriving or it is already decoding.

        Pausing a decoding generation-only request would re-prefill its whole
        prompt on the decode engine, where it starves behind the running
        decodes until the engine runs dry (observed: ~1700 s stalls, client
        timeouts, KV-transfer timeouts on the retries). The request finishes
        its current turn on the cache it holds; the reuse-tree reset makes the
        next turn prefill under the new weights.
        """
        manager, _ = _transfer_manager()
        decoding = _request(20, is_generation_only=True)
        waiting = _request(21, state=LlmRequestState.DISAGG_GENERATION_INIT,
                           is_generation_only=True)
        receiving = _request(22, state=LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS,
                             is_generation_only=True)
        landed = _request(23, state=LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE,
                          is_generation_only=True)
        ex = _RecomputeExecutor([decoding, waiting, receiving, landed],
                                transceiver=MagicMock(), transfer_manager=manager)

        PyExecutor.recompute_active_requests(ex)

        assert ex.terminated == []
        assert ex.paused == []
        # The disaggregated recompute reduces to the reuse-tree reset.
        assert ex.reset_calls == 1
        # Rank-local: the predicate and the recompute must not enter collectives.
        ex.dist.tp_allgather.assert_not_called()
        ex.dist.tp_gather.assert_not_called()
        # Operators verify the fix from this marker on every gen rank.
        assert ("TRTLLM_RECOMPUTE_ACTIVE_REQUESTS_CALLED active_requests=4 "
                "recompute=0 disagg_transfer_bound=4") in capfd.readouterr().out

    def test_mixed_engine_still_recomputes_full_requests(self, capfd):
        """A request that is neither context-only nor generation-only (an
        aggregated request sharing an engine that also has a transceiver) is
        recomputed even when disaggregated requests are present."""
        manager, _ = _transfer_manager()
        full = _request(24)  # CONTEXT_AND_GENERATION, decoding
        gen_only = _request(25, is_generation_only=True)
        ctx_only = _request(26, state=LlmRequestState.CONTEXT_INIT, is_context_only=True)
        # Neither flag set but sitting in a transfer state: the state-set
        # safety net must still keep it out of the recompute.
        in_transfer_state = _request(
            28, state=LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS)
        ex = _RecomputeExecutor([full, gen_only, ctx_only, in_transfer_state],
                                transceiver=MagicMock(), transfer_manager=manager)

        PyExecutor.recompute_active_requests(ex)

        assert ex.terminated == [full]
        assert ex.paused == [full]
        assert ex.reset_calls == 1
        ex.dist.tp_allgather.assert_not_called()
        assert ("active_requests=4 recompute=1 disagg_transfer_bound=3"
                in capfd.readouterr().out)

    def test_aggregated_engine_recomputes_generation_only_typed_requests_too(self):
        """Without a transceiver the type is irrelevant: everything is recomputed."""
        req = _request(27, is_generation_only=True)
        ex = _RecomputeExecutor([req], transceiver=None, transfer_manager=None)

        PyExecutor.recompute_active_requests(ex)

        assert ex.terminated == [req]
        assert ex.paused == [req]

    @pytest.mark.parametrize("with_transceiver", [True, False],
                             ids=["transceiver", "kv_connector_only"])
    def test_pinned_by_async_send_is_bound_regardless_of_state(self, with_transceiver):
        """Blocks pinned for an asynchronous send must never be freed by a
        recompute, whether the send belongs to the cache transceiver or to a
        KV connector (no transceiver at all), and whatever the request state."""
        manager, _ = _transfer_manager()
        req = _request(30, is_context_only=False)
        manager.start_transfer(req)
        req.state = LlmRequestState.GENERATION_IN_PROGRESS  # state alone would not protect it
        ex = _RecomputeExecutor([req, _request(31)],
                                transceiver=MagicMock() if with_transceiver else None,
                                transfer_manager=manager)

        assert PyExecutor._is_disagg_transfer_bound(ex, req) is True
        PyExecutor.recompute_active_requests(ex)
        assert [r.py_request_id for r in ex.terminated] == [31]
        assert 30 in manager.requests_in_transfer()


def _coordinator(manager, kv_cache_manager, transceiver):
    """DisaggTransferCoordinator over the fake transceiver / real transfer manager
    (main moved the context-send path out of PyExecutor._send_kv_async)."""
    registry = MagicMock()
    registry.canceled_request_ids.return_value = []
    transceiver.has_retired_send_session.return_value = False
    transceiver.has_inflight_transfer.return_value = False
    transceiver.pipeline_transfer_enabled = False
    transceiver._fp4_mla_bridge_enabled = False
    return DisaggTransferCoordinator(
        transceiver=transceiver,
        transfer_manager=manager,
        kv_cache_manager=kv_cache_manager,
        dist=MagicMock(),
        effects=MagicMock(),
        registry=registry,
        enable_attention_dp=False,
        force_terminate_ctx_for_partial_reuse=False,
    )


class TestSendKvAsyncFailsRequestNotLoop:

    def _build(self):
        manager, kv_cache_manager = _transfer_manager()
        transceiver = MagicMock()
        transceiver.kv_transfer_timeout_ms = None
        return _coordinator(manager, kv_cache_manager, transceiver), manager, kv_cache_manager, transceiver

    def test_missing_blocks_fail_only_that_request(self):
        coordinator, manager, kv_cache_manager, transceiver = self._build()
        bad = _request(40, is_context_only=True)
        good = _request(41, is_context_only=True)
        # Only the first send raises; the second must still go through.
        transceiver.respond_and_send_async.side_effect = [
            IndexError("unordered_map::at"), None
        ]

        coordinator.send_completed_context([bad, good])

        # The failed request is only marked; the rank-safe error paths
        # (handle_errors_synced) pick it up from its state.
        assert bad.state == LlmRequestState.DISAGG_TRANS_ERROR
        # start_transfer() was undone for the failed request: untracked and unpinned.
        assert 40 not in manager.requests_in_transfer()
        kv_cache_manager.unpin_blocks_by_id.assert_called_once_with(7)
        # The healthy request completed the hand-over normally.
        assert 41 in manager.requests_in_transfer()
        assert good.state == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

    def test_healthy_path_unchanged(self):
        coordinator, manager, kv_cache_manager, transceiver = self._build()
        req = _request(50, is_context_only=True)

        coordinator.send_completed_context([req])

        transceiver.respond_and_send_async.assert_called_once_with(req)
        assert 50 in manager.requests_in_transfer()
        kv_cache_manager.unpin_blocks_by_id.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__])
