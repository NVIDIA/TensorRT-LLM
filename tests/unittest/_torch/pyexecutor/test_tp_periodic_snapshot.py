# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-rank agreement for expanding an already scheduled context chunk."""

import contextlib
import gc
import multiprocessing
import threading
import time
import weakref
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import tensorrt_llm._torch.pyexecutor.py_executor as executor_module
from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin
from tensorrt_llm._torch.pyexecutor.error_classification import ErrorBudget
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import MultimodalScheduler
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler
from tensorrt_llm._torch.pyexecutor.scheduler.tp_periodic_snapshot import (
    coordinate_periodic_snapshot,
    finish_periodic_snapshots_tp,
)
from tensorrt_llm.llmapi.llm_args import ContextChunkingPolicy


class _Exchange:
    def __init__(self):
        self.barrier = threading.Barrier(2, timeout=5)
        self.values = [None, None]

    def for_rank(self, rank):
        def gather(value):
            self.values[rank] = value
            self.barrier.wait()
            result = list(self.values)
            self.barrier.wait()
            return result

        return SimpleNamespace(tp_allgather=gather, tp_allgather_int64=gather)


class _Context:
    py_request_id = 9
    context_current_position = 0
    context_remaining_length = 2048
    prompt_len = 2048
    context_chunk_size = 256
    expect_snapshot_points = list(range(256, 2049, 256))
    py_ctx_pre_resize_cap = 0
    py_mamba_inforward_snapshot_point = None
    py_multimodal_data = None
    py_mm_encoder_state = None
    is_dummy_request = False
    is_attention_dp_dummy = False
    is_context_init_state = True

    @property
    def is_last_context_chunk(self):
        return self.context_current_position + self.context_chunk_size == self.prompt_len


class _CommitMetadataErrorContext(_Context):
    @property
    def context_chunk_size(self):
        return 256

    @context_chunk_size.setter
    def context_chunk_size(self, value):
        raise RuntimeError("injected commit_metadata_error")


class _Cache:
    def __init__(self, expansion_ok=True):
        self.capacity = 256
        self.expansion_ok = expansion_ok
        self.is_active = True
        self.history_length = 0
        self.num_committed_tokens = 0
        self.pending_ssm_snapshot_slot = None

    def try_resize_capacity_without_eviction(self, capacity):
        if capacity > self.capacity and not self.expansion_ok:
            return False
        self.capacity = capacity
        return True

    def resize(self, capacity):
        self.capacity = capacity
        return True


class _Manager:
    def __init__(
        self, *, capacity=4, reservation_ok=True, expansion_ok=True, reservation_error=False
    ):
        self.periodic_snapshot_capacity = capacity
        self.reservation_ok = reservation_ok
        self.reservation_error = reservation_error
        self.periodic_snapshot_alignment = 64
        self.kv_cache_config = SimpleNamespace(
            mamba_state_config=SimpleNamespace(periodic_snapshot_interval=256)
        )
        self.kv_cache_map = {9: _Cache(expansion_ok)}
        self._kv_alloc_extra_tokens = 0
        self.reservation = None
        self.periodic_snapshot_requires_consensus = True
        self._periodic_snapshot_model_support = True
        self._supports_periodic_snapshots = True
        self._periodic_snapshot_supports_tp = True
        self.enable_block_reuse = True
        self.is_draft = False

    def context_replay_tokens(self, request):
        return None

    def is_speculative(self):
        return False

    def reserve_periodic_snapshots(self, request, end, points, *, without_eviction=False):
        assert without_eviction
        if self.reservation_error:
            raise RuntimeError("injected reservation error")
        if not self.reservation_ok:
            return False
        self.reservation = (end, points)
        return True

    def release_periodic_snapshots(self, request):
        self.reservation = None

    def _fill_fresh_kv_pages(self, request_id):
        pass


def _run_pair(batches, managers, *, expect_errors=False):
    exchange = _Exchange()
    errors = []

    def run(rank):
        try:
            coordinate_periodic_snapshot(
                batches[rank],
                managers[rank],
                exchange.for_rank(rank),
                extra_budget=768,
                max_context_length=1024,
                chunk_unit_size=64,
            )
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=run, args=(rank,)) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not any(thread.is_alive() for thread in threads)
    if not expect_errors:
        assert not errors
    return errors


@pytest.mark.cpu_only
def test_two_ranks_extend_the_same_scheduled_context():
    requests = [_Context(), _Context()]
    batches = [ScheduledRequests(), ScheduledRequests()]
    managers = [_Manager(), _Manager()]
    for batch, request in zip(batches, requests):
        batch.append_context_request(request)
        batch.append_generation_request(SimpleNamespace(py_request_id=5, py_beam_width=1))

    _run_pair(batches, managers)

    assert [request.context_chunk_size for request in requests] == [1024, 1024]
    assert [manager.kv_cache_map[9].capacity for manager in managers] == [1024, 1024]
    assert [manager.reservation for manager in managers] == [
        (1024, (256, 512, 768)),
        (1024, (256, 512, 768)),
    ]


@pytest.mark.cpu_only
@pytest.mark.parametrize("failure", ["reservation", "capacity"])
def test_one_rank_refusing_resources_keeps_the_baseline(failure):
    requests = [_Context(), _Context()]
    batches = [ScheduledRequests(), ScheduledRequests()]
    managers = [
        _Manager(),
        _Manager(reservation_ok=failure != "reservation", expansion_ok=failure != "capacity"),
    ]
    for batch, request in zip(batches, requests):
        batch.append_context_request(request)

    _run_pair(batches, managers)

    assert [request.context_chunk_size for request in requests] == [256, 256]
    assert [manager.kv_cache_map[9].capacity for manager in managers] == [256, 256]
    assert [manager.reservation for manager in managers] == [None, None]


@pytest.mark.cpu_only
def test_one_rank_reservation_error_is_reported_to_both_after_cleanup():
    requests = [_Context(), _Context()]
    batches = [ScheduledRequests(), ScheduledRequests()]
    managers = [_Manager(), _Manager(reservation_error=True)]
    for batch, request in zip(batches, requests):
        batch.append_context_request(request)

    errors = _run_pair(batches, managers, expect_errors=True)

    assert len(errors) == 2
    assert all("injected reservation error" in str(error) for error in errors)
    assert [manager.kv_cache_map[9].capacity for manager in managers] == [256, 256]
    assert [manager.reservation for manager in managers] == [None, None]


@pytest.mark.cpu_only
def test_ranks_use_the_smaller_staging_capacity():
    requests = [_Context(), _Context()]
    batches = [ScheduledRequests(), ScheduledRequests()]
    managers = [_Manager(), _Manager(capacity=3)]
    for batch, request in zip(batches, requests):
        batch.append_context_request(request)

    _run_pair(batches, managers)

    assert [request.context_chunk_size for request in requests] == [768, 768]
    assert [manager.reservation for manager in managers] == [(768, (256, 512)), (768, (256, 512))]


@pytest.mark.cpu_only
def test_different_baseline_batches_fail_together_before_forward():
    request = _Context()
    batches = [ScheduledRequests(), ScheduledRequests()]
    batches[0].append_context_request(request)
    managers = [_Manager(), _Manager()]

    errors = _run_pair(batches, managers, expect_errors=True)

    assert len(errors) == 2
    assert all(isinstance(error, RuntimeError) and "baseline" in str(error) for error in errors)
    assert request.context_chunk_size == 256
    assert managers[0].kv_cache_map[9].capacity == 256
    assert all(manager.reservation is None for manager in managers)


@pytest.mark.cpu_only
def test_same_baseline_with_one_ineligible_rank_keeps_the_batch():
    requests = [_Context(), _Context()]
    batches = [ScheduledRequests(), ScheduledRequests()]
    managers = [_Manager(), _Manager()]
    managers[1].kv_cache_map[9].is_active = False
    for batch, request in zip(batches, requests):
        batch.append_context_request(request)

    _run_pair(batches, managers)

    assert [request.context_chunk_size for request in requests] == [256, 256]
    assert all(manager.reservation is None for manager in managers)


@pytest.mark.cpu_only
def test_different_pause_decisions_fail_together_before_forward():
    requests = [_Context(), _Context()]
    batches = [ScheduledRequests(), ScheduledRequests()]
    managers = [_Manager(), _Manager()]
    for batch, request in zip(batches, requests):
        batch.append_context_request(request)
    batches[1].recompute_paused_requests = [SimpleNamespace(py_request_id=7)]

    errors = _run_pair(batches, managers, expect_errors=True)

    assert len(errors) == 2
    assert all(isinstance(error, RuntimeError) and "baseline" in str(error) for error in errors)


@pytest.mark.cpu_only
def test_different_context_completion_fails_together_before_forward():
    requests = [_Context(), _Context()]
    requests[1].prompt_len = 256
    requests[1].context_remaining_length = 256
    batches = [ScheduledRequests(), ScheduledRequests()]
    managers = [_Manager(), _Manager()]
    for batch, request in zip(batches, requests):
        batch.append_context_request(request)

    errors = _run_pair(batches, managers, expect_errors=True)

    assert len(errors) == 2
    assert all(isinstance(error, RuntimeError) and "baseline" in str(error) for error in errors)


def _make_executor(dist, batch, request, manager, wrapped, mm_active, mm_model=False):
    dist.tp_size, dist.cp_size, dist.pp_size = 2, 1, 1
    ex = object.__new__(PyExecutor)
    ex.dist = dist
    ex.enable_attention_dp = False
    ex._is_kv_manager_v2 = True
    ex.is_shutdown = False
    ex.waiting_queue = []
    ex.active_requests = [request]
    if mm_active:
        ex.active_requests.append(
            SimpleNamespace(py_mm_encoder_state=object(), py_multimodal_data=None)
        )
    ex.kv_cache_manager = manager
    ex.kv_connector_manager = None
    ex.kv_cache_transceiver = None
    ex.is_encoder_decoder = False
    ex._mm_encoder_item_scheduling_enabled = wrapped
    ex._ctx_mla_kv_len_cap = None
    ex.ctx_decoder_batcher = None
    ex.drafter = None
    ex.model_engine = SimpleNamespace(
        is_spec_decode=False, model=MultimodalModelMixin() if mm_model else None
    )
    ex._packed_sync_enabled = False
    ex._disagg_coordinator = Mock()
    scheduler = object.__new__(KVCacheV2Scheduler)
    scheduler.chunking_enabled = True
    scheduler.chunking_policy = ContextChunkingPolicy.FORCE_CHUNK
    scheduler._ctx_starvation_iterations = None
    scheduler.periodic_snapshot_tp_enabled = False
    scheduler.periodic_snapshot_extra_budget = 768
    scheduler.max_context_length = 1024
    scheduler.chunk_unit_size = 64
    ex.scheduler = object.__new__(MultimodalScheduler) if wrapped else scheduler
    if wrapped:
        ex.scheduler.scheduler = scheduler
    ex._release_unused_connector_reservations = Mock()
    ex._poll_encoder_steps = Mock()
    ex._fetch_and_activate_new_requests = Mock(return_value=[])
    ex._handle_control_request = Mock()
    ex._progress_context_cancellations = Mock(side_effect=lambda reqs: reqs)
    ex._cancel_waiting_disagg_requests = lambda reqs: reqs
    ex._activate_completed_remote_tails = Mock()
    ex._sweep_transfer_held_failures = Mock()
    ex._init_iter_stats_if_sampled = Mock(return_value=None)
    ex._pad_attention_dp_dummy_request = Mock()
    ex._pad_empty_attention_dp_batch = Mock()
    ex._prefetch_for_context_requests = Mock()
    caps_at_schedule = []

    def schedule():
        caps_at_schedule.append(scheduler.periodic_snapshot_tp_enabled)
        return batch, [], 1

    ex._schedule = Mock(side_effect=schedule)
    return ex, scheduler, caps_at_schedule


def _gloo_rank(rank, rendezvous, ready, results):
    import torch.distributed as distributed

    ready.wait()
    distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )

    class GlooDist:
        def __init__(self):
            self.object_gathers = 0
            self.int64_gathers = 0

        def tp_allgather(self, value):
            self.object_gathers += 1
            gathered = [None, None]
            distributed.all_gather_object(gathered, value)
            return gathered

        def tp_allgather_int64(self, values):
            self.int64_gathers += 1
            tensor = torch.tensor(values, dtype=torch.int64)
            gathered = [torch.empty_like(tensor), torch.empty_like(tensor)]
            distributed.all_gather(gathered, tensor)
            return [item.tolist() for item in gathered]

    observations = []
    try:
        for scenario in (
            "success",
            "reservation_failure",
            "reservation_error",
            "rollback_error",
            "rollback_false",
            "release_error",
            "forward_commit_success",
            "forward_commit_error",
            "commit_resize_error",
            "commit_fill_error",
            "commit_metadata_error",
            "commit_batch_error",
            "ineligible",
            "different_baseline",
            "different_pause",
            "different_completion",
        ):
            request = (
                _CommitMetadataErrorContext()
                if rank == 1 and scenario == "commit_metadata_error"
                else _Context()
            )
            batch = ScheduledRequests()
            if scenario == "different_completion" and rank == 1:
                request.prompt_len = 256
                request.context_remaining_length = 256
            if scenario != "different_baseline" or rank == 0:
                batch.append_context_request(request)
            if scenario == "different_pause" and rank == 1:
                batch.recompute_paused_requests = [SimpleNamespace(py_request_id=7)]
            manager = _Manager(
                reservation_ok=not (
                    scenario
                    in ("reservation_failure", "rollback_error", "rollback_false", "release_error")
                    and rank == 1
                ),
                reservation_error=scenario == "reservation_error" and rank == 1,
            )
            if rank == 0 and scenario in ("rollback_error", "rollback_false"):
                cache = manager.kv_cache_map[9]
                resize = cache.try_resize_capacity_without_eviction

                def fail_rollback(capacity):
                    if capacity == 256:
                        if scenario == "rollback_error":
                            raise RuntimeError("injected rollback_error")
                        return False
                    return resize(capacity)

                cache.try_resize_capacity_without_eviction = fail_rollback
            if rank == 0 and scenario == "release_error":
                release = manager.release_periodic_snapshots

                def fail_release(request):
                    release(request)
                    raise RuntimeError("injected release_error")

                manager.release_periodic_snapshots = fail_release
            if rank == 1 and scenario in (
                "commit_resize_error",
                "commit_fill_error",
                "commit_batch_error",
            ):

                def fail_commit(*args, **kwargs):
                    raise RuntimeError(f"injected {scenario}")

                if scenario == "commit_resize_error":
                    manager.kv_cache_map[9].resize = fail_commit
                elif scenario == "commit_fill_error":
                    manager._fill_fresh_kv_pages = fail_commit
                else:
                    batch.reset_context_requests = fail_commit
            if scenario == "ineligible" and rank == 1:
                manager.kv_cache_map[9].is_active = False
            error = None
            try:
                coordinate_periodic_snapshot(
                    batch,
                    manager,
                    GlooDist(),
                    extra_budget=768,
                    max_context_length=1024,
                    chunk_unit_size=64,
                )
                if scenario in ("forward_commit_success", "forward_commit_error"):
                    assert batch.periodic_snapshot_tp

                    def finish(_batch):
                        manager.reservation = None
                        if scenario == "forward_commit_error" and rank == 1:
                            raise RuntimeError("injected forward_commit_error")

                    manager.finish_periodic_snapshots = finish
                    finish_periodic_snapshots_tp(batch, manager, GlooDist())
            except RuntimeError as exc:
                error = str(exc)
            observations.append(
                (
                    scenario,
                    request.context_chunk_size,
                    manager.kv_cache_map[9].capacity,
                    manager.reservation,
                    error,
                )
            )

        request = _Context()
        batch = ScheduledRequests()
        batch.append_context_request(request)
        ex, scheduler, caps = _make_executor(
            GlooDist(), batch, request, _Manager(), wrapped=True, mm_active=rank == 1
        )
        ex._prepare_and_schedule_batch()
        observations.append(("mm_active", request.context_chunk_size, caps[-1]))
        ex.active_requests = [request]
        ex._prepare_and_schedule_batch()
        observations.append(("text_only", request.context_chunk_size, caps[-1]))

        request = _Context()
        if rank == 1:
            request.py_multimodal_data = {"multimodal_embedding": object()}
        batch = ScheduledRequests()
        batch.append_context_request(request)
        ex, scheduler, caps = _make_executor(
            GlooDist(), batch, request, _Manager(), wrapped=False, mm_active=False, mm_model=True
        )
        ex._prepare_and_schedule_batch()
        observations.append(("precomputed_mm", request.context_chunk_size, caps[-1]))
        request.py_multimodal_data = None
        generation = SimpleNamespace(
            py_request_id=5,
            py_beam_width=1,
            num_draft_tokens=0,
            py_batch_idx=None,
            py_mm_encoder_state=None,
            is_context_init_state=False,
            py_multimodal_data={"mrope_config": {"mrope_position_deltas": 1}},
            is_attention_dp_dummy=False,
        )
        batch.append_generation_request(generation)
        ex.active_requests = [request, generation]
        ex._prepare_and_schedule_batch()
        observations.append(("mm_decode", request.context_chunk_size, caps[-1]))

        decode_only = ScheduledRequests()
        decode_only.append_generation_request(generation)
        dist = GlooDist()
        coordinate_periodic_snapshot(
            decode_only,
            _Manager(),
            dist,
            extra_budget=768,
            max_context_length=1024,
            chunk_unit_size=64,
        )
        observations.append(("decode_only", dist.object_gathers, dist.int64_gathers))
        results.put((rank, observations))
    finally:
        distributed.destroy_process_group()


@pytest.mark.cpu_only
def test_two_process_gloo_agreement_and_failure_paths(tmp_path):
    """Each case uses two real processes and matching collectives, bounded by the process-group timeout."""
    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue()
    ready = ctx.Barrier(2, timeout=40)
    rendezvous = str(tmp_path / "tp-periodic-rendezvous")
    workers = [
        ctx.Process(target=_gloo_rank, args=(rank, rendezvous, ready, results)) for rank in range(2)
    ]
    try:
        for worker in workers:
            worker.start()
        deadline = time.monotonic() + 60
        observations = dict(
            results.get(timeout=max(0.1, deadline - time.monotonic())) for _ in workers
        )
        for worker in workers:
            worker.join(timeout=5)
            assert worker.exitcode == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()
        results.close()

    assert set(observations) == {0, 1}
    for rank in (0, 1):
        (
            success,
            failed,
            error,
            rollback_error,
            rollback_false,
            release_error,
            forward_success,
            forward_error,
            resize_error,
            fill_error,
            metadata_error,
            batch_error,
            ineligible,
            mismatch,
            pause,
            completion,
            mm_active,
            text,
            precomputed,
            decode,
            decode_only,
        ) = observations[rank]
        assert success == ("success", 1024, 1024, (1024, (256, 512, 768)), None)
        assert failed == ("reservation_failure", 256, 256, None, None)
        assert error[:4] == ("reservation_error", 256, 256, None)
        assert "injected reservation error" in error[4]
        for label, observation in (
            ("rollback_error", rollback_error),
            ("rollback_false", rollback_false),
            ("release_error", release_error),
        ):
            assert observation[0] == label
            assert observation[3] is None, (rank, observation)
            assert observation[4] is not None and (
                label in observation[4]
                or (
                    label == "rollback_false"
                    and "restore the scheduled context capacity" in observation[4]
                )
            ), (rank, observation)
        assert forward_success == ("forward_commit_success", 1024, 1024, None, None)
        assert forward_error[:4] == ("forward_commit_error", 1024, 1024, None)
        assert "injected forward_commit_error" in forward_error[4]
        for label, observation in (
            ("commit_resize_error", resize_error),
            ("commit_fill_error", fill_error),
            ("commit_metadata_error", metadata_error),
            ("commit_batch_error", batch_error),
        ):
            assert observation[0] == label
            assert observation[3] is None, (rank, observation)
            assert observation[4] is not None and f"injected {label}" in observation[4], (
                rank,
                observation,
            )
        assert ineligible == ("ineligible", 256, 256, None, None)
        assert mismatch[:4] == ("different_baseline", 256, 256, None)
        assert "baseline" in mismatch[4]
        assert pause[:4] == ("different_pause", 256, 256, None)
        assert "baseline" in pause[4]
        assert completion[:4] == ("different_completion", 256, 256, None)
        assert "baseline" in completion[4]
        assert mm_active == ("mm_active", 256, False)
        assert text == ("text_only", 1024, True)
        assert precomputed == ("precomputed_mm", 256, False)
        assert decode == ("mm_decode", 1024, True)
        assert decode_only == ("decode_only", 1, 0)


def _native_collective_rank(rank, rendezvous, ready, results, backend):
    import torch.distributed as distributed

    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2 import (
        MambaHybridCacheManagerV2,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        AttentionLayerConfig,
        BufferConfig,
        GpuCacheTierConfig,
        KVCacheManager,
        KVCacheManagerConfig,
        SsmLayerConfig,
    )

    class TorchDist:
        def __init__(self):
            self.votes = []

        def tp_allgather(self, value):
            gathered = [None, None]
            distributed.all_gather_object(gathered, value)
            return gathered

        def tp_allgather_int64(self, values):
            self.votes.append(tuple(values))
            device = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
            tensor = torch.tensor(values, dtype=torch.int64, device=device)
            gathered = [torch.empty_like(tensor), torch.empty_like(tensor)]
            distributed.all_gather(gathered, tensor)
            return [item.cpu().tolist() for item in gathered]

    class NativeManager(_Manager):
        def __init__(self, cache):
            super().__init__()
            self.kv_cache_map = {9: cache}
            self.periodic_snapshot_plan = {}
            self.last_reservation_succeeded = None
            self.fail_commit_fill = False

        def reserve_periodic_snapshots(self, *args, **kwargs):
            result = MambaHybridCacheManagerV2.reserve_periodic_snapshots(self, *args, **kwargs)
            self.last_reservation_succeeded = result
            return result

        release_periodic_snapshots = MambaHybridCacheManagerV2.release_periodic_snapshots

        def _fill_fresh_kv_pages(self, request_id):
            if self.fail_commit_fill:
                raise RuntimeError("injected native commit fill error")

    ready.wait()
    torch.cuda.set_device(rank if backend == "nccl" else 0)
    distributed.init_process_group(
        backend,
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    stream = torch.cuda.Stream()
    config = KVCacheManagerConfig(
        tokens_per_block=64,
        cache_tiers=[GpuCacheTierConfig(quota=4 << 20)],
        layers=[
            AttentionLayerConfig(layer_id=0, buffers=[BufferConfig(role="key", size=8192)]),
            SsmLayerConfig(layer_id=1, buffers=[BufferConfig(role="state", size=16384)]),
        ],
        max_util_for_resume=1.0,
        commit_min_snapshot=True,
    )
    native = KVCacheManager(config)
    cache = native.create_kv_cache()
    filler = None
    observations = []
    try:
        assert cache.resume(stream.cuda_stream)
        assert cache.resize(256)
        manager = NativeManager(cache)
        group = native.get_life_cycle_pool_group_indices()[native.get_layer_group_id(1)]
        attention_group = native.get_life_cycle_pool_group_indices()[native.get_layer_group_id(0)]
        free_before = native.get_storage_statistics()[group].free

        for scenario in (
            "success",
            "reservation_failure",
            "capacity_failure",
            "finish_failure",
            "commit_failure",
        ):
            if scenario in ("reservation_failure", "capacity_failure") and rank == 1:
                filler = native.create_kv_cache()
                assert filler.resume(stream.cuda_stream)
                if scenario == "reservation_failure":
                    available = native.get_storage_statistics()[group].free
                    assert available > 3
                    assert (
                        len(filler.acquire_ssm_snapshot_slots_without_eviction(available - 2))
                        == available - 2
                    )
                else:
                    available = native.get_storage_statistics()[attention_group].free
                    assert available > 12
                    assert filler.resize((available - 2) * 64)
                    assert native.get_storage_statistics()[attention_group].free < 12
            available_before = native.get_storage_statistics()[group].free
            attention_before = native.get_storage_statistics()[attention_group].free
            history_before = cache.history_length
            manager.last_reservation_succeeded = None
            request = _Context()
            batch = ScheduledRequests()
            batch.append_context_request(request)
            dist = TorchDist()
            manager.fail_commit_fill = scenario == "commit_failure" and rank == 1
            error = None
            try:
                coordinate_periodic_snapshot(
                    batch,
                    manager,
                    dist,
                    extra_budget=768,
                    max_context_length=1024,
                    chunk_unit_size=64,
                )
                if scenario == "finish_failure":
                    assert batch.periodic_snapshot_tp

                    def finish(_batch):
                        manager.release_periodic_snapshots(request)
                        if rank == 1:
                            raise RuntimeError("injected native forward commit error")

                    manager.finish_periodic_snapshots = finish
                    finish_periodic_snapshots_tp(batch, manager, dist)
            except RuntimeError as exc:
                error = str(exc)
            plan = manager.periodic_snapshot_plan.get(request.py_request_id)
            observations.append(
                (
                    scenario,
                    request.context_chunk_size,
                    cache.capacity,
                    None if plan is None else (plan.end, plan.points, len(plan.slots)),
                    native.get_storage_statistics()[group].free,
                    manager.last_reservation_succeeded,
                    dist.votes,
                    error,
                )
            )
            if scenario != "success":
                assert cache.pending_ssm_snapshot_slot is None
                assert cache.history_length == history_before
                assert native.get_storage_statistics()[group].free == available_before
                if scenario not in ("commit_failure", "finish_failure"):
                    assert native.get_storage_statistics()[attention_group].free == attention_before
            if plan is not None:
                manager.release_periodic_snapshots(request)
                assert cache.try_resize_capacity_without_eviction(256)
            if scenario == "finish_failure":
                # The production executor terminates the rank group; this test
                # resets its isolated cache only to exercise the next case.
                assert cache.try_resize_capacity_without_eviction(256)
            if scenario == "success":
                assert native.get_storage_statistics()[group].free == free_before
            if filler is not None:
                filler.release_ssm_snapshot_slots()
                filler.close()
                filler = None
    finally:
        if filler is not None:
            filler.release_ssm_snapshot_slots()
            filler.close()
        cache.close()
        native.shutdown()
        stream.synchronize()
        distributed.destroy_process_group()
    results.put((rank, observations))


def _run_two_process_native_cache_reservation(tmp_path, backend):
    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue()
    ready = ctx.Barrier(2, timeout=60)
    rendezvous = str(tmp_path / f"tp-periodic-native-{backend}-rendezvous")
    workers = [
        ctx.Process(
            target=_native_collective_rank, args=(rank, rendezvous, ready, results, backend)
        )
        for rank in range(2)
    ]
    try:
        for worker in workers:
            worker.start()
        deadline = time.monotonic() + 150
        observations = dict(
            results.get(timeout=max(0.1, deadline - time.monotonic())) for _ in workers
        )
        for worker in workers:
            worker.join(timeout=5)
            assert worker.exitcode == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()
        results.close()

    assert set(observations) == {0, 1}
    for rank in (0, 1):
        success, failure, capacity_failure, finish_failure, commit_failure = observations[rank]
        assert success[:4] == ("success", 1024, 1024, (1024, (256, 512, 768), 3))
        assert failure[:4] == ("reservation_failure", 256, 256, None)
        assert success[5] is True
        assert failure[5] is (rank == 0)
        assert success[6] == [(1, 0), (0,)]
        assert failure[6] == ([(1, 0), (0,)] if rank == 0 else [(0, 0), (0,)])
        assert capacity_failure[:4] == ("capacity_failure", 256, 256, None)
        assert capacity_failure[5] is True
        assert capacity_failure[6] == ([(1, 0), (0,)] if rank == 0 else [(0, 0), (0,)])
        assert commit_failure[0] == "commit_failure"
        assert commit_failure[3] is None
        assert commit_failure[5] is True
        assert commit_failure[6] == [(1, 0), (int(rank == 1),)]
        assert "injected native commit fill error" in commit_failure[7]
        assert finish_failure[0] == "finish_failure"
        assert finish_failure[3] is None
        assert finish_failure[5] is True
        assert finish_failure[6] == [(1, 0), (0,), (int(rank == 1),)]
        assert "injected native forward commit error" in finish_failure[7]
        if rank == 1:
            assert failure[4] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_two_process_native_cache_reservation_with_gloo(tmp_path):
    """Two processes share one GPU; each owns real cache pools and a Gloo rank."""
    _run_two_process_native_cache_reservation(tmp_path, "gloo")


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_two_gpu_native_cache_reservation_with_nccl(tmp_path):
    """Two GPUs agree on native reservations and a one-rank resource failure."""
    _run_two_process_native_cache_reservation(tmp_path, "nccl")


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "peer_capacity, wrapped, mm_rank, embedding_rank, expected",
    [
        (4, False, None, None, 1024),
        (1, False, None, None, 256),
        (4, True, None, None, 1024),
        (4, True, 1, None, 256),
        (4, True, None, 1, 256),
        (4, False, None, 1, 256),
        (4, True, None, "decode", 1024),
        (4, (True, False), None, None, 1024),
        (4, (True, False), 0, None, 256),
    ],
)
def test_executor_coordinates_periodic_snapshots_with_scheduler(
    peer_capacity,
    wrapped,
    mm_rank,
    embedding_rank,
    expected,
):
    requests = [_Context(), _Context()]
    batches = [ScheduledRequests(), ScheduledRequests()]
    managers = [_Manager(), _Manager(capacity=peer_capacity)]
    exchange = _Exchange()
    executors = []
    schedulers = []
    caps_at_schedule = []
    for rank, (batch, request, manager) in enumerate(zip(batches, requests, managers)):
        if rank == embedding_rank:
            request.py_multimodal_data = {"multimodal_embedding": object()}
        batch.append_context_request(request)
        rank_wrapped = wrapped[rank] if isinstance(wrapped, tuple) else wrapped
        ex, scheduler, caps = _make_executor(
            exchange.for_rank(rank),
            batch,
            request,
            manager,
            rank_wrapped,
            rank == mm_rank,
            mm_model=embedding_rank is not None,
        )
        if embedding_rank == "decode":
            generation = SimpleNamespace(
                py_request_id=5,
                py_beam_width=1,
                num_draft_tokens=0,
                py_batch_idx=None,
                py_mm_encoder_state=None,
                is_context_init_state=False,
                py_multimodal_data={"mrope_config": {"mrope_position_deltas": 1}},
                is_attention_dp_dummy=False,
            )
            batch.append_generation_request(generation)
            ex.active_requests.append(generation)
        executors.append(ex)
        schedulers.append(scheduler)
        caps_at_schedule.append(caps)

    errors = []

    def run(ex):
        try:
            ex._prepare_and_schedule_batch()
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=run, args=(ex,)) for ex in executors]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not any(thread.is_alive() for thread in threads)
    assert not errors, errors
    assert [request.context_chunk_size for request in requests] == [expected, expected]
    expected_cap = peer_capacity > 1 and mm_rank is None and embedding_rank in (None, "decode")
    assert caps_at_schedule == [[expected_cap], [expected_cap]]
    assert [scheduler.periodic_snapshot_tp_enabled for scheduler in schedulers] == [
        expected_cap,
        expected_cap,
    ]
    assert [getattr(ex, "_periodic_snapshot_tp_used", False) for ex in executors] == [
        peer_capacity > 1,
        peer_capacity > 1,
    ]
    cache_refs = [weakref.ref(manager) for manager in managers]
    for ex in executors:
        ex.kv_cache_manager = None
    managers.clear()
    del manager
    gc.collect()
    assert all(ref() is None for ref in cache_refs)


@pytest.mark.cpu_only
def test_failed_tp_capability_vote_marks_executor_for_fail_stop():
    request = _Context()
    batch = ScheduledRequests()
    batch.append_context_request(request)
    dist = SimpleNamespace(tp_allgather_int64=Mock(side_effect=RuntimeError("vote failed")))
    executor, _, _ = _make_executor(dist, batch, request, _Manager(), False, False)

    with pytest.raises(RuntimeError, match="vote failed"):
        executor._prepare_and_schedule_batch()

    assert executor._periodic_snapshot_tp_used


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "message",
    [
        "CUDA error: device-side assert triggered",
        "CUDA error: unspecified launch failure",
        "CUDA error: misaligned address",
        "CUDA error: an illegal instruction was encountered",
    ],
)
def test_late_fatal_error_after_periodic_tp_escapes_error_handler(message):
    executor = object.__new__(PyExecutor)
    executor.enable_attention_dp = False
    executor.kv_connector_manager = None
    executor.dist = SimpleNamespace(world_size=2)
    executor._periodic_snapshot_tp_used = True
    executor._fatal_error = None
    executor._error_budget = ErrorBudget()

    with pytest.raises(RuntimeError, match=message):
        executor._handle_errors(message)


@pytest.mark.cpu_only
@pytest.mark.parametrize("message", ["CUDA out of memory", "network misaligned address"])
def test_periodic_tp_fatal_classifier_keeps_recoverable_errors(message):
    assert not executor_module._periodic_tp_unrecoverable_error(message)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "periodic_tp_used, message, prior_message, cuda_frame, suppress_context, raises",
    [
        (True, "CUDA error: device-side assert triggered", None, False, False, True),
        (True, "cleanup failed", "CUDA error: device-side assert triggered", False, False, True),
        (True, "cleanup failed", "CUDA error: device-side assert triggered", False, True, True),
        (True, "ordinary prefetch failure", None, False, False, False),
        (True, "network misaligned address", None, True, False, False),
        (True, "network misaligned address", "CUDA out of memory", False, False, False),
        (False, "CUDA error: device-side assert triggered", None, False, False, False),
    ],
)
def test_periodic_tp_propagates_fatal_prefetch_errors(
    monkeypatch, periodic_tp_used, message, prior_message, cuda_frame, suppress_context, raises
):
    executor = object.__new__(PyExecutor)
    executor._periodic_snapshot_tp_used = periodic_tp_used
    executor.model_engine = SimpleNamespace(model=object())
    executor.previous_batch = None
    executor.active_requests = [
        SimpleNamespace(state=executor_module.LlmRequestState.CONTEXT_INIT, py_request_id=9)
    ]
    executor.llm_args = SimpleNamespace(
        multimodal_config=SimpleNamespace(encoder_side_stream_max_ahead=1)
    )
    error = RuntimeError(message)
    if cuda_frame:
        try:
            exec(
                compile(
                    "raise RuntimeError('network misaligned address')", "/torch/cuda/foo.py", "exec"
                )
            )
        except RuntimeError as caught:
            error = caught
    if prior_message is not None:
        error.__context__ = RuntimeError(prior_message)
        error.__suppress_context__ = suppress_context
    log = Mock()
    monkeypatch.setattr(executor_module, "logger", log)
    monkeypatch.setattr(
        executor_module,
        "maybe_prefetch_mm_encoder_for_next_iter",
        Mock(side_effect=error),
    )
    batch = ScheduledRequests()
    if raises:
        with pytest.raises(RuntimeError) as caught:
            executor._maybe_prefetch_next_iter_mm_encoders(batch)
        assert caught.value is error
        if suppress_context:
            assert prior_message in str(log.error.call_args)
    else:
        executor._maybe_prefetch_next_iter_mm_encoders(batch)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "message", ["CUDA error: device-side assert triggered", "CUDA error: misaligned address"]
)
@pytest.mark.parametrize("chain", ["direct", "cause", "context"])
def test_fatal_input_copy_wait_after_periodic_tp_is_not_swallowed(message, chain):
    executor = object.__new__(PyExecutor)
    executor.enable_attention_dp = False
    executor.kv_connector_manager = None
    executor.dist = SimpleNamespace(world_size=2)
    executor._periodic_snapshot_tp_used = True
    executor._fatal_error = None
    executor._error_budget = ErrorBudget()
    executor.active_requests = []
    executor._pending_transfer_responses = []
    executor._pending_response_terminations = []
    executor._enqueue_responses = Mock()
    fatal = RuntimeError(message)
    error = fatal if chain == "direct" else RuntimeError("input copy cleanup failed")
    if chain == "cause":
        error.__cause__ = fatal
    elif chain == "context":
        error.__context__ = fatal
        error.__suppress_context__ = True
    executor._wait_for_model_engine_input_copy = Mock(side_effect=error)

    with pytest.raises(RuntimeError) as caught:
        executor._handle_errors("request validation failed", requests=[], charge_budget=False)

    assert caught.value is error


@pytest.mark.cpu_only
def test_request_scoped_error_text_does_not_abort_periodic_tp():
    executor = object.__new__(PyExecutor)
    executor.enable_attention_dp = False
    executor.kv_connector_manager = None
    executor.dist = SimpleNamespace(world_size=2)
    executor._periodic_snapshot_tp_used = True
    executor._fatal_error = None
    executor.is_shutdown = False
    executor._error_budget = ErrorBudget()
    executor.active_requests = []
    executor._wait_for_model_engine_input_copy = Mock()
    executor._pending_transfer_responses = []
    executor._pending_response_terminations = []
    executor._enqueue_responses = Mock()

    executor._handle_errors(
        "CUDA error: device-side assert triggered", requests=[], charge_budget=False
    )

    assert executor._fatal_error is None
    assert not executor.is_shutdown


@pytest.mark.cpu_only
@pytest.mark.parametrize("periodic_tp_used", [False, True])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_rank_crash_watchdog_delivery_policy(monkeypatch, periodic_tp_used, cleanup_fails):
    executor = object.__new__(PyExecutor)
    executor._periodic_snapshot_tp_used = periodic_tp_used
    executor._event_loop_error_delivered = threading.Event()
    executor._event_loop_error_delivered.set()
    executor._apply_dynamo_recompile_limit = Mock()
    executor.is_warmup = True
    executor.garbage_collection_gen0_threshold = 100
    executor.dist = SimpleNamespace(world_size=2)
    executor._executor_loop_cleanup = Mock(
        side_effect=RuntimeError("cleanup failure") if cleanup_fails else None
    )
    executor.event_loop = Mock(side_effect=RuntimeError("injected late failure"))
    watchdog = SimpleNamespace(cancel=Mock(), deadline=1.0)
    start = Mock(return_value=watchdog)
    kill = Mock()
    monkeypatch.setattr(
        executor_module, "host_profiler_context", lambda **_kwargs: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        executor_module, "customized_gc_thresholds", lambda *_args: contextlib.nullcontext()
    )
    monkeypatch.setattr(executor_module, "start_rank_crash_kill_watchdog", start)
    monkeypatch.setattr(executor_module, "hard_kill_on_rank_crash", kill)

    with pytest.raises(
        RuntimeError, match="cleanup failure" if cleanup_fails else "injected late failure"
    ):
        executor._event_loop_wrapper()

    expected = None if periodic_tp_used else executor._event_loop_error_delivered
    assert start.call_args.kwargs["error_delivered"] is expected
    assert kill.call_args.kwargs["error_delivered"] is expected
