# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""reset_prefix_cache with CUDA graph padding dummies on KVCacheManagerV2.

KVCacheManagerV2 refuses to clear reusable blocks while a KV cache is open, and a padding dummy
keeps one open between steps. An RL weight sync resets the prefix cache at its end, so the reset
releases the padding dummies first and re-creates them once the reuse state is cleared.
Conditional attention-DP execution needs its reserved draft-length-0 dummy back before the next
``_can_queue``.

The runner and executor are the real classes; only the KV cache manager is a stand-in, which
enforces the native open-cache check of ``clear_reusable_blocks()``.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner, DSparkCUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import RemoteTailPhase, ScheduledRequests

pytestmark = pytest.mark.cpu_only


class _KvManager:
    """KVCacheManagerV2 stand-in: tracks open KV caches and rejects a clear while any is open."""

    def __init__(self):
        self.open_ids = set()
        self.events = []
        self.refuse_new_dummies = False
        self.refuse_after_clear = False

    def add_dummy_requests(self, request_ids, **kwargs):
        if self.refuse_new_dummies:
            return None
        self.open_ids.update(request_ids)
        self.events.extend(("create", request_id) for request_id in request_ids)
        return [SimpleNamespace(py_request_id=request_id) for request_id in request_ids]

    def free_resources(self, request):
        self.open_ids.discard(request.py_request_id)
        self.events.append(("release", request.py_request_id))

    def reset_reuse_state(self, on_cleared=None):
        if self.open_ids:
            raise RuntimeError(
                f"clear_reusable_blocks() with {len(self.open_ids)} KV cache(s) still open"
            )
        self.events.append(("reset",))
        self.refuse_new_dummies = self.refuse_after_clear


def _runner(runner_cls=CUDAGraphRunner, spec_config=None):
    runner = runner_cls.__new__(runner_cls)
    runner.padding_dummy_requests = {}
    if runner_cls is DSparkCUDAGraphRunner:
        runner.secondary_padding_dummy_requests = {}
        runner.adp_shape_agreement = None
    runner.spec_config = spec_config
    runner.is_encoder_decoder = False
    runner.config = SimpleNamespace(use_mrope=False, max_beam_width=1)
    return runner


def _executor(runner, *, kv_manager_v2=True, remote_tail_adp=False):
    kv = _KvManager()
    executor = PyExecutor.__new__(PyExecutor)
    executor._is_kv_manager_v2 = kv_manager_v2
    executor._enable_remote_tail_adp = remote_tail_adp
    executor.enable_attention_dp = remote_tail_adp
    executor.enable_joint_kv_cache_reuse = False
    executor.kv_cache_manager = kv
    executor.resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: kv})
    executor.model_engine = SimpleNamespace(cuda_graph_runner=runner, route_capture=None)
    return executor, kv


def _hold(executor, runner, draft_lens, variant=0):
    """Allocate padding dummies the way warmup and the padded steps do."""
    for draft_len in draft_lens:
        dummy = runner._get_or_create_padding_dummy(
            executor.resource_manager, draft_len, variant=variant
        )
        assert dummy is not None


def _ids(cache):
    return {draft_len: dummy.py_request_id for draft_len, dummy in cache.items()}


def test_v2_reset_releases_the_padding_dummies_and_re_creates_them_after_the_clear():
    runner = _runner()
    executor, kv = _executor(runner)
    _hold(executor, runner, (0, 3))
    held = _ids(runner.padding_dummy_requests)
    kv.events.clear()

    executor.reset_prefix_cache()

    assert kv.events == [
        ("release", held[0]),
        ("release", held[3]),
        ("reset",),
        ("create", held[0]),
        ("create", held[3]),
    ]
    assert _ids(runner.padding_dummy_requests) == held
    assert kv.open_ids == set(held.values())


def test_v2_reset_re_creates_both_dummy_variants_low_rows_first():
    spec_config = SimpleNamespace(
        decoding_type="DSpark",
        max_draft_len=2,
        spec_dec_mode=SimpleNamespace(use_one_engine=lambda: False),
        get_runtime_tokens_per_gen_step=lambda draft_len: 1 + draft_len,
    )
    runner = _runner(DSparkCUDAGraphRunner, spec_config)
    executor, kv = _executor(runner)
    _hold(executor, runner, (0, 2))
    _hold(executor, runner, (2,), variant=1)
    low = _ids(runner.padding_dummy_requests)
    high = _ids(runner.secondary_padding_dummy_requests)
    kv.events.clear()

    executor.reset_prefix_cache()

    assert kv.events[-3:] == [("create", low[0]), ("create", low[2]), ("create", high[2])]
    assert ("reset",) in kv.events
    assert _ids(runner.padding_dummy_requests) == low
    assert _ids(runner.secondary_padding_dummy_requests) == high


def test_rejected_reset_still_restores_the_padding_dummies():
    """A real KV cache still open fails the clear; the engine keeps serving on its dummies."""
    runner = _runner()
    executor, kv = _executor(runner)
    _hold(executor, runner, (0,))
    held = _ids(runner.padding_dummy_requests)
    kv.open_ids.add(7)

    with pytest.raises(RuntimeError, match="still open"):
        executor.reset_prefix_cache()

    assert ("reset",) not in kv.events
    assert _ids(runner.padding_dummy_requests) == held
    assert kv.open_ids == {7, held[0]}


def test_dummy_that_cannot_be_re_created_is_left_to_lazy_re_creation():
    runner = _runner()
    executor, kv = _executor(runner)
    _hold(executor, runner, (0,))
    kv.refuse_after_clear = True

    executor.reset_prefix_cache()

    assert kv.events[-1] == ("reset",)
    assert not runner.padding_dummy_requests
    assert not kv.open_ids


def test_v1_reset_keeps_the_padding_dummies():
    """The V1 manager has no open-cache precondition; its dummies stay cached."""
    runner = _runner()
    executor, kv = _executor(runner, kv_manager_v2=False)
    _hold(executor, runner, (0,))
    held = _ids(runner.padding_dummy_requests)
    kv.reset_reuse_state = lambda on_cleared=None: kv.events.append(("reset",))
    kv.events.clear()

    executor.reset_prefix_cache()

    assert kv.events == [("reset",)]
    assert _ids(runner.padding_dummy_requests) == held


def test_engines_without_a_cuda_graph_runner_reset_as_before():
    executor, kv = _executor(None)

    executor.reset_prefix_cache()

    assert kv.events == [("reset",)]


def _idle_decode_batch():
    batch = ScheduledRequests()
    batch.attention_dp_phase = RemoteTailPhase.DECODE
    batch.is_attention_dp_phase_idle = True
    return batch


def test_remote_tail_reset_keeps_the_execution_row_for_the_next_queue_decision():
    runner = _runner()
    executor, kv = _executor(runner, remote_tail_adp=True)
    # Startup reserves one row per draft length (_initialize_remote_tail_adp).
    _hold(executor, runner, (0, 1))
    held = _ids(runner.padding_dummy_requests)

    executor.reset_prefix_cache()

    assert ("reset",) in kv.events
    assert _ids(runner.padding_dummy_requests) == held
    executor.dist = Mock(tp_size=2)
    executor.dist.tp_allgather_int64.return_value = torch.tensor([[1, 0], [2, 2]])
    assert executor._can_queue(_idle_decode_batch()) == (True, False)
    executor.dist.tp_allgather_int64.assert_called_once_with([1, 0])


def test_remote_tail_reset_that_loses_the_execution_row_stops_every_rank_at_the_next_queue():
    runner = _runner()
    executor, kv = _executor(runner, remote_tail_adp=True)
    _hold(executor, runner, (0,))
    kv.refuse_after_clear = True

    # No local raise: the next queue decision reports the loss on every rank together.
    executor.reset_prefix_cache()

    assert 0 not in runner.padding_dummy_requests
    executor.dist = Mock(tp_size=2)
    executor.dist.tp_allgather_int64.return_value = torch.tensor([[-1, 0], [2, 2]])
    with pytest.raises(RuntimeError, match="lost its reserved CUDA graph padding request"):
        executor._can_queue(_idle_decode_batch())
    executor.dist.tp_allgather_int64.assert_called_once_with([-1, 0])
