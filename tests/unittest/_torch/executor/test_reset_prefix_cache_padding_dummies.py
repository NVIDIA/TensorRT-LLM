# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""reset_prefix_cache with CUDA graph padding dummies on KVCacheManagerV2.

KVCacheManagerV2 refuses to clear reusable blocks while a KV cache is open, and a padding dummy
keeps one open between steps. An RL weight sync resets the prefix cache at its end, so the reset
releases the padding dummies first and re-creates them once the reuse state is cleared.
Conditional attention-DP execution needs every reserved execution row (one per draft length it
can run) back before the next ``_can_queue``, and reports a lost one on every rank there.

The runner and executor are the real classes; only the KV cache manager is a stand-in, which
enforces the native open-cache check of ``clear_reusable_blocks()``.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import py_executor as executor_module
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner, DSparkCUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import RemoteTailPhase, ScheduledRequests

pytestmark = pytest.mark.cpu_only


class _KvManager:
    """KVCacheManagerV2 stand-in: tracks open KV caches and rejects a clear while any is open.

    A labeled manager tags its events with the label, so two managers can share one event log.
    """

    def __init__(self, label=None, events=None):
        self.label = label
        self.open_ids = set()
        self.events = [] if events is None else events
        self.refuse_new_dummies = False
        self.refuse_after_clear = False
        # Request IDs refused from now on, and from the clear on.
        self.refused_ids = set()
        self.refuse_ids_after_clear = set()
        # Request IDs whose allocation raises.
        self.raising_ids = set()

    def _event(self, kind, *args):
        return (kind, *args) if self.label is None else (kind, self.label, *args)

    def add_dummy_requests(self, request_ids, draft_kv_cache_manager=None, **kwargs):
        if self.raising_ids.intersection(request_ids):
            raise RuntimeError("allocator failure")
        if self.refuse_new_dummies or self.refused_ids.intersection(request_ids):
            return None
        # Like KVCacheManagerV2, a one-model draft manager gets the dummy's draft cache too.
        for manager in (self, draft_kv_cache_manager):
            if manager is not None:
                manager.open_ids.update(request_ids)
                manager.events.extend(manager._event("create", rid) for rid in request_ids)
        return [SimpleNamespace(py_request_id=request_id) for request_id in request_ids]

    def free_resources(self, request):
        self.open_ids.discard(request.py_request_id)
        self.events.append(self._event("release", request.py_request_id))

    def reset_reuse_state(self, on_cleared=None):
        if self.open_ids:
            raise RuntimeError(
                f"clear_reusable_blocks() with {len(self.open_ids)} KV cache(s) still open"
            )
        self.events.append(self._event("reset"))
        self.refuse_new_dummies = self.refuse_after_clear
        self.refused_ids |= self.refuse_ids_after_clear


def _runner(runner_cls=CUDAGraphRunner, spec_config=None):
    runner = runner_cls.__new__(runner_cls)
    runner.padding_dummy_requests = {}
    if runner_cls is DSparkCUDAGraphRunner:
        runner.secondary_padding_dummy_requests = {}
        runner.adp_shape_agreement = None
    runner.spec_config = spec_config
    runner.is_encoder_decoder = False
    runner.config = SimpleNamespace(use_mrope=False, max_beam_width=1)
    # No graphs were captured, so there is nothing for __del__ to clear.
    runner.clear = lambda **_kwargs: None
    return runner


def _dspark_runner():
    spec_config = SimpleNamespace(
        decoding_type="DSpark",
        max_draft_len=2,
        spec_dec_mode=SimpleNamespace(use_one_engine=lambda: False),
        get_runtime_tokens_per_gen_step=lambda draft_len: 1 + draft_len,
    )
    return _runner(DSparkCUDAGraphRunner, spec_config)


def _executor(
    runner, *, kv_manager_v2=True, remote_tail_adp=False, reserved_draft_lens=None, draft_kv=False
):
    """``draft_kv`` adds a one-model draft KV cache manager with joint KV cache reuse."""
    kv = _KvManager(label="target" if draft_kv else None)
    executor = PyExecutor.__new__(PyExecutor)
    executor._is_kv_manager_v2 = kv_manager_v2
    executor._enable_remote_tail_adp = remote_tail_adp
    if reserved_draft_lens is not None:
        # What _initialize_remote_tail_adp records; the class default reserves draft length 0.
        executor._remote_tail_reserved_draft_lens = reserved_draft_lens
    executor.enable_attention_dp = remote_tail_adp
    executor.kv_cache_manager = kv
    managers = {ResourceManagerType.KV_CACHE_MANAGER: kv}
    executor.enable_joint_kv_cache_reuse = draft_kv
    if draft_kv:
        executor.draft_kv_cache_manager = _KvManager(label="draft", events=kv.events)
        managers[ResourceManagerType.DRAFT_KV_CACHE_MANAGER] = executor.draft_kv_cache_manager
    executor.resource_manager = ResourceManager(managers)
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
    runner = _dspark_runner()
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


def test_remote_tail_reset_keeps_the_execution_rows_for_the_next_queue_decision():
    runner = _runner()
    executor, kv = _executor(runner, remote_tail_adp=True, reserved_draft_lens=(0, 1))
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


@pytest.mark.parametrize(
    "reserved_draft_lens,stops", [((0, 1), True), ((0,), False)], ids=["reserved", "not-reserved"]
)
def test_remote_tail_reset_that_loses_the_draft_length_1_row(reserved_draft_lens, stops):
    """A phase-idle rank runs the row of its runtime draft length and never creates it on demand,
    so losing a reserved row other than row 0 must stop every rank at the next queue decision too.
    A lost row that is not reserved is left to lazy re-creation."""
    runner = _runner()
    executor, kv = _executor(runner, remote_tail_adp=True, reserved_draft_lens=reserved_draft_lens)
    _hold(executor, runner, (0, 1))
    held = _ids(runner.padding_dummy_requests)
    kv.refuse_ids_after_clear = {held[1]}

    with (
        patch.object(executor_module.logger, "error") as error,
        patch.object(executor_module.logger, "warning") as warning,
    ):
        executor.reset_prefix_cache()

    assert _ids(runner.padding_dummy_requests) == {0: held[0]}
    reported = error if stops else warning
    reported.assert_called_once()
    assert ("draft length 1" if stops else "draft_len=1") in reported.call_args.args[0]
    (warning if stops else error).assert_not_called()
    executor.dist = Mock(tp_size=2)
    if stops:
        executor.dist.tp_allgather_int64.return_value = torch.tensor([[-1, 0], [2, 2]])
        with pytest.raises(RuntimeError, match="lost its reserved CUDA graph padding request"):
            executor._can_queue(_idle_decode_batch())
        executor.dist.tp_allgather_int64.assert_called_once_with([-1, 0])
    else:
        executor.dist.tp_allgather_int64.return_value = torch.tensor([[1, 0], [2, 2]])
        assert executor._can_queue(_idle_decode_batch()) == (True, False)
        executor.dist.tp_allgather_int64.assert_called_once_with([1, 0])


def test_runner_lists_held_dummies_by_variant_then_draft_length():
    runner = _dspark_runner()
    executor, _ = _executor(runner)
    _hold(executor, runner, (2, 0))
    _hold(executor, runner, (2,), variant=1)
    low = runner.padding_dummy_requests
    high = runner.secondary_padding_dummy_requests

    assert runner.held_padding_dummies() == [(0, 0, low[0]), (2, 0, low[2]), (2, 1, high[2])]


def test_runner_releases_one_variant_without_its_sibling():
    runner = _dspark_runner()
    executor, kv = _executor(runner)
    _hold(executor, runner, (2,))
    _hold(executor, runner, (2,), variant=1)
    low = _ids(runner.padding_dummy_requests)
    runner.adp_shape_agreement = object()

    # A row that is not held is skipped.
    assert runner.release_padding_dummies(executor.resource_manager, [(2, 1), (0, 0)]) == [(2, 1)]

    assert _ids(runner.padding_dummy_requests) == low
    assert not runner.secondary_padding_dummy_requests
    assert kv.open_ids == {low[2]}
    # Releasing a row invalidates the agreement that certified it.
    assert runner.adp_shape_agreement is None


def test_runner_restore_re_creates_released_rows_and_reports_the_failed_ones():
    runner = _runner()
    executor, kv = _executor(runner)
    _hold(executor, runner, (0, 3))
    held = _ids(runner.padding_dummy_requests)

    released = runner.release_padding_dummies(executor.resource_manager)
    assert released == [(0, 0), (3, 0)]
    assert not runner.padding_dummy_requests and not kv.open_ids

    kv.refused_ids = {held[3]}
    assert runner.restore_padding_dummies(executor.resource_manager, released) == [(3, 0)]
    assert _ids(runner.padding_dummy_requests) == {0: held[0]}
    # A row that is still held is left as it is.
    assert runner.restore_padding_dummies(executor.resource_manager, [(0, 0)]) == []
    assert kv.events.count(("create", held[0])) == 2


@pytest.mark.parametrize("clear_rejected", [True, False], ids=["clear-rejected", "clear-ok"])
def test_a_raising_re_creation_neither_masks_the_reset_nor_skips_the_other_rows(clear_rejected):
    """The restore is cleanup. An allocator error while re-creating one dummy is logged; the other
    dummies are still re-created; a rejected reset still raises its own error and a successful one
    returns. The lost reserved row still stops every rank together at the next queue decision."""
    runner = _runner()
    executor, kv = _executor(runner, remote_tail_adp=True)
    _hold(executor, runner, (0, 3))
    held = _ids(runner.padding_dummy_requests)
    if clear_rejected:
        kv.open_ids.add(7)  # A real KV cache is still open, so the clear is rejected.
    kv.raising_ids = {held[0]}

    with patch.object(executor_module.logger, "error") as error:
        if clear_rejected:
            with pytest.raises(RuntimeError, match="still open"):
                executor.reset_prefix_cache()
        else:
            executor.reset_prefix_cache()

    assert (("reset",) in kv.events) != clear_rejected
    assert _ids(runner.padding_dummy_requests) == {3: held[3]}
    messages = [call.args[0] for call in error.call_args_list]
    assert any("allocator failure" in message for message in messages)
    assert any("execution row for draft length 0" in message for message in messages)
    executor.dist = Mock(tp_size=2)
    executor.dist.tp_allgather_int64.return_value = torch.tensor([[-1, 0], [2, 2]])
    with pytest.raises(RuntimeError, match="lost its reserved CUDA graph padding request"):
        executor._can_queue(_idle_decode_batch())
    executor.dist.tp_allgather_int64.assert_called_once_with([-1, 0])


@pytest.mark.parametrize(
    "draft_clear_rejected", [False, True], ids=["draft-clear-ok", "draft-clear-rejected"]
)
def test_joint_reuse_reset_releases_both_managers_first_and_restores_both(draft_clear_rejected):
    """With joint KV cache reuse the reset also clears the one-model draft manager, which holds a
    draft cache for every dummy. Both managers release the dummies before the first clear, and both
    get them back whether the draft clear goes through or is rejected."""
    spec_config = SimpleNamespace(
        spec_dec_mode=SimpleNamespace(use_one_engine=lambda: True),
        get_runtime_tokens_per_gen_step=lambda draft_len: 1 + draft_len,
    )
    runner = _runner(spec_config=spec_config)
    executor, kv = _executor(runner, draft_kv=True)
    draft = executor.draft_kv_cache_manager
    _hold(executor, runner, (0, 1))
    held = set(_ids(runner.padding_dummy_requests).values())
    both = {(label, request_id) for label in ("target", "draft") for request_id in held}
    assert draft.open_ids == held
    if draft_clear_rejected:
        draft.open_ids.add(7)  # A real draft cache is still open.
    kv.events.clear()

    if draft_clear_rejected:
        with pytest.raises(RuntimeError, match="still open"):
            executor.reset_prefix_cache()
    else:
        executor.reset_prefix_cache()

    first_clear = kv.events.index(("reset", "target"))
    assert {event[0] for event in kv.events[:first_clear]} == {"release"}
    assert {event[1:] for event in kv.events[:first_clear]} == both
    assert (("reset", "draft") in kv.events) != draft_clear_rejected
    created = {event[1:] for event in kv.events[first_clear:] if event[0] == "create"}
    assert created == both
    assert kv.open_ids == held
    assert draft.open_ids == (held | {7} if draft_clear_rejected else held)
