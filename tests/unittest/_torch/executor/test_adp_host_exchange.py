# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Per-iteration attention-DP host exchanges: one int64 collective per
exchange, and the cuda-graph batch agreement gathered once per iteration."""

import types
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from tensorrt_llm._torch.distributed.communicator import Distributed
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


class _CountingDist:
    """Returns scripted rows for tp_allgather_int64 and counts the calls."""

    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def tp_allgather_int64(self, values):
        self.calls.append(list(values))
        return np.array(self.rows, dtype=np.int64)


def _make_batch(num_generation):
    batch = ScheduledRequests()
    batch.generation_requests = [SimpleNamespace(py_request_id=i) for i in range(num_generation)]
    return batch


def _make_runner(dist, tp_size=2, max_batch=64):
    runner = object.__new__(CUDAGraphRunner)
    runner.enabled = True
    runner.padding_enabled = True
    runner.max_supported_batch_size = max_batch
    runner.enable_encoder_decoder_mixed_cuda_graph = False
    runner.config = SimpleNamespace(
        enable_attention_dp=True,
        mapping=SimpleNamespace(tp_size=tp_size),
        dist=dist,
        batch_size=max_batch,
        use_mrope=False,
    )
    runner._round_up_batch_size_with_draft_len = lambda bs, draft_len: 4 if bs <= 4 else 8
    runner._get_or_create_padding_dummy = lambda rm, draft_len: SimpleNamespace(py_request_id=-1)
    return runner


def test_pad_batch_reuses_rows_offered_by_executor():
    # Rows as gathered in _can_queue: [batch_size, can_run] per rank.
    dist = _CountingDist([[3, 1], [4, 1]])
    runner = _make_runner(dist)
    batch = _make_batch(3)

    runner.offer_adp_batch_info(3, True, np.array([[3, 1], [4, 1]]))
    padding = CUDAGraphRunner._get_padded_batch(runner, batch, Mock(), 0)

    assert dist.calls == []
    assert padding == 1
    assert batch.batch_size == 4
    assert runner._adp_post_pad == (batch, 4, True, True)


def test_pad_batch_falls_back_when_offer_does_not_match():
    dist = _CountingDist([[2, 1], [2, 1]])
    runner = _make_runner(dist)
    batch = _make_batch(2)

    runner.offer_adp_batch_info(3, True, np.array([[3, 1], [4, 1]]))
    CUDAGraphRunner._get_padded_batch(runner, batch, Mock(), 0)

    # Stale offer discarded; a fresh exchange carries [can_run, batch_size].
    assert dist.calls == [[1, 2]]
    assert runner._adp_offer is None


def test_maybe_get_cuda_graph_reuses_pad_batch_verdict():
    dist = _CountingDist([[1, 4], [1, 4]])
    runner = _make_runner(dist)
    runner.get_graph_key = Mock(return_value="key")
    runner.graph_metadata = {}
    runner._capture_allowed = False
    runner._get_seq_len_mode = Mock(return_value=False)
    runner._is_mixed_encoder_decoder_batch = Mock(return_value=False)
    batch = _make_batch(4)

    runner._adp_post_pad = (batch, 4, True, True)
    result = CUDAGraphRunner.maybe_get_cuda_graph(
        runner, batch, enable_spec_decode=False, attn_metadata=object()
    )

    assert dist.calls == []
    assert runner._adp_post_pad is None
    assert result == (None, None, None)  # no captured graph in this stub


def test_maybe_get_cuda_graph_returns_none_when_a_peer_cannot_run():
    dist = _CountingDist([[1, 4], [0, 4]])
    runner = _make_runner(dist)
    runner._is_mixed_encoder_decoder_batch = Mock(return_value=False)
    batch = _make_batch(4)

    runner._adp_post_pad = (batch, 4, False, True)
    result = CUDAGraphRunner.maybe_get_cuda_graph(
        runner, batch, enable_spec_decode=False, attn_metadata=object()
    )

    assert result == (None, None, None)
    assert dist.calls == []


def test_maybe_get_cuda_graph_gathers_when_no_verdict():
    dist = _CountingDist([[1, 4], [1, 3]])
    runner = _make_runner(dist)
    runner._is_mixed_encoder_decoder_batch = Mock(return_value=False)
    batch = _make_batch(4)

    result = CUDAGraphRunner.maybe_get_cuda_graph(
        runner, batch, enable_spec_decode=False, attn_metadata=object()
    )

    assert dist.calls == [[1, 4]]
    assert result == (None, None, None)  # sizes differ across ranks


def test_can_queue_offers_gathered_rows_to_runner():
    executor = object.__new__(PyExecutor)
    executor.enable_attention_dp = True
    gathered = np.array([[2, 1], [0, 1]])
    executor.dist = SimpleNamespace(tp_allgather_int64=Mock(return_value=gathered))
    offer = Mock()
    executor.model_engine = SimpleNamespace(
        cuda_graph_runner=SimpleNamespace(offer_adp_batch_info=offer)
    )

    can_queue, this_rank = PyExecutor._can_queue(
        executor, types.SimpleNamespace(batch_size=2, can_run_cuda_graph=True)
    )

    executor.dist.tp_allgather_int64.assert_called_once_with([2, 1])
    offer.assert_called_once()
    assert offer.call_args.args[0] == 2 and offer.call_args.args[1] is True
    assert can_queue is False and this_rank is True


class _ObjectPathDist(Distributed):
    """Minimal backend without buffer collectives: exercises the defaults."""

    def __init__(self, mapping, peers):
        super().__init__(mapping)
        self.peers = peers

    def barrier(self):
        pass

    def tp_barrier(self):
        pass

    def broadcast(self, obj, root=0):
        return obj

    def allgather(self, obj, root=0):
        return [obj] * len(self.peers)

    def allreduce(self, obj, op=None):
        return obj

    def tp_allreduce(self, obj, op=None):
        return obj

    def tp_broadcast(self, obj, root=0, **kwargs):
        return obj

    def cp_broadcast(self, obj, root=0, **kwargs):
        return obj

    def tp_allgather(self, obj, *, small_payload=False):
        return [list(p) for p in self.peers]

    def cp_allgather(self, obj, *, small_payload=False):
        return [obj]


def test_default_int64_helpers_route_through_object_paths():
    mapping = Mapping(world_size=2, rank=0, tp_size=2)
    dist = _ObjectPathDist(mapping, peers=[[5, 6], [7, 8]])

    rows = dist.tp_allgather_int64([5, 6])
    assert rows.dtype == np.int64 and rows.tolist() == [[5, 6], [7, 8]]
    assert dist.tp_cp_allgather_int64([5, 6]).tolist() == [[5, 6], [7, 8]]
    assert dist.broadcast_int64([3]).tolist() == [3]
    assert dist.tp_cp_broadcast_int64([4, 5]).tolist() == [4, 5]
