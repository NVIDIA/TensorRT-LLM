# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for distributed warmup fail-stop and single-rank recovery."""

import contextlib
from collections.abc import Iterator
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import model_engine as model_engine_module
from tensorrt_llm._torch.pyexecutor import py_executor as py_executor_module
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine


class _StandInMambaCacheManager:
    """Concrete stand-in for the abstract Mamba cache-manager type check."""

    def __init__(self, available_tokens: int = 8) -> None:
        self._available_tokens = available_tokens

    def get_num_available_tokens(self, **_kwargs: object) -> int:
        return self._available_tokens


def _engine(
    *,
    world_size: int = 1,
    dwdp_size: int = 0,
    tp_size: int = 1,
    tp_peer_values: tuple[int, ...] = (),
    peer_values: tuple[int, ...] | None = None,
    with_dist: bool = True,
) -> PyTorchModelEngine:
    """Build an engine carrying only what the warmup policy reads.

    ``dist`` is ``Optional`` on the real engine -- stub engines such as
    ``DummyModelEngine`` construct without a communicator -- so ``with_dist``
    covers that state too.

    ``tp_allgather`` returns this rank's value followed by ``tp_peer_values``;
    ``allgather`` does the same over ``peer_values`` for the forward group.
    ``peer_values`` defaults to peers that mirror this rank, i.e. a world that
    agrees with whatever this rank reports.
    """
    engine = object.__new__(PyTorchModelEngine)

    def _allgather(value):
        peers = [value] * (world_size - 1) if peer_values is None else list(peer_values)
        return [value, *peers]

    engine.dist = (
        SimpleNamespace(
            world_size=world_size,
            rank=3,
            tp_allgather=lambda value: [value, *tp_peer_values],
            allgather=_allgather,
        )
        if with_dist
        else None
    )
    engine.mapping = SimpleNamespace(
        dwdp_enabled=dwdp_size > 1,
        has_cp_helix=lambda: False,
        tp_size=tp_size,
    )
    engine._reset_moe_alltoall_state = mock.Mock()
    return engine


@contextlib.contextmanager
def _released_batch(batch: object) -> Iterator[object]:
    yield batch


@contextlib.contextmanager
def _no_cuda_side_effects() -> Iterator[tuple[mock.Mock, mock.Mock]]:
    with (
        mock.patch.object(torch.cuda, "empty_cache") as empty_cache,
        mock.patch.object(torch.cuda, "synchronize") as synchronize,
    ):
        yield empty_cache, synchronize


@contextlib.contextmanager
def _guard_env(*, global_size: int) -> Iterator[tuple[mock.Mock, mock.Mock]]:
    with (
        mock.patch.object(py_executor_module, "start_rank_crash_kill_watchdog") as watchdog,
        mock.patch.object(py_executor_module, "propagate_hard_kill") as hard_kill,
        mock.patch.object(py_executor_module, "global_mpi_size", return_value=global_size),
    ):
        yield watchdog, hard_kill


def _run_guard(*, world_size: int, dwdp_size: int, error: BaseException) -> None:
    dist = SimpleNamespace(world_size=world_size)
    mapping = SimpleNamespace(dwdp_enabled=dwdp_size > 1)
    with py_executor_module._distributed_warmup_guard(dist, mapping):
        raise error


@pytest.mark.parametrize(
    "world_size,dwdp_size,global_size,expected_peer_count",
    [
        (1, 0, 1, None),
        (2, 0, 1, 2),
        (1, 4, 4, 4),
    ],
)
def test_guard_topology_policy(
    world_size: int,
    dwdp_size: int,
    global_size: int,
    expected_peer_count: int | None,
) -> None:
    error = ValueError("warmup failed")
    with _guard_env(global_size=global_size) as (watchdog, hard_kill):
        with pytest.raises(ValueError) as excinfo:
            _run_guard(world_size=world_size, dwdp_size=dwdp_size, error=error)

    assert excinfo.value is error
    # The watchdog's grace period is what lets the setup/RPC path report the
    # original exception; aborting the world here would replace it with an
    # exit code.
    hard_kill.assert_not_called()
    if expected_peer_count is None:
        watchdog.assert_not_called()
    else:
        watchdog.assert_called_once_with(expected_peer_count, error_delivered=None)


@pytest.mark.parametrize("signal", [KeyboardInterrupt, SystemExit])
def test_guard_leaves_teardown_signals_unarmed(signal: type) -> None:
    with _guard_env(global_size=4) as (watchdog, hard_kill):
        with pytest.raises(signal):
            _run_guard(world_size=4, dwdp_size=0, error=signal())

    watchdog.assert_not_called()
    hard_kill.assert_not_called()


@pytest.mark.parametrize(
    "world_size,dwdp_size,has_batch,expected",
    [
        (1, 0, False, False),
        # Every rank in the forward group lacks a batch. The allgather proves
        # nobody is left inside the collective, so skipping strands no one --
        # this is the shape a max_batch_size=1 config simply cannot build.
        (2, 0, False, False),
        # DWDP peers live outside `self.dist`, so unanimity cannot be observed
        # and the shape stays fatal.
        (1, 4, False, None),
        (2, 0, True, True),
    ],
)
def test_warmup_batch_policy(
    world_size: int, dwdp_size: int, has_batch: bool, expected: bool | None
) -> None:
    engine = _engine(world_size=world_size, dwdp_size=dwdp_size)
    batch = object() if has_batch else None
    if expected is None:
        with pytest.raises(RuntimeError, match="cannot skip the shape"):
            engine._should_run_warmup_batch(batch, 128, "general")
    else:
        assert engine._should_run_warmup_batch(batch, 128, "general") is expected


def test_symmetric_absence_skips_instead_of_failing_the_world() -> None:
    """The regression this fix exists for.

    `max_batch_size=1` cannot build the mixed context+generation attention
    shape, on any rank, at any KV cache size. Every rank reaches the same
    verdict from configuration alone, so the group agrees and skips. Failing
    here took down ten multi-GPU tests that had always skipped this shape.
    """
    engine = _engine(world_size=2, tp_size=2, peer_values=(0,))

    assert (
        engine._should_run_warmup_batch(None, 2, "attention, num_tokens=2, num_gen_requests=1")
        is False
    )


def test_group_disagreement_still_fails_with_the_ranks_named() -> None:
    """Asymmetric capacity is the deadlock the agreement exists to catch."""
    engine = _engine(world_size=3, peer_values=(1, 0))

    with pytest.raises(RuntimeError) as excinfo:
        engine._should_run_warmup_batch(None, 128, "general")

    message = str(excinfo.value)
    assert "rank(s) [0, 2]" in message
    assert "would deadlock" in message


def test_pipeline_only_world_agrees_without_a_tp_group() -> None:
    """TP=1 with pipeline peers is where the TP-only check gathered nothing.

    Three of the ten failures were disaggregated context servers at
    `tensor_parallel_size: 1, pipeline_parallel_size: 2`.
    """
    engine = _engine(world_size=2, tp_size=1, peer_values=(0,))

    assert engine._should_run_warmup_batch(None, 128, "general") is False


def test_agreement_group_is_absent_for_dwdp() -> None:
    engine = _engine(world_size=4, dwdp_size=4)

    assert engine._warmup_agreement_allgather() is None


@pytest.mark.parametrize(
    "world_size,peer_values,flag,expected",
    [
        (1, None, True, True),
        (2, (1,), True, True),
        # One pipeline stage owns the guided decoder and opts out; the group
        # follows it rather than letting the others enter alone.
        (2, (0,), True, False),
        (2, (1,), False, False),
    ],
)
def test_phase_entry_flag_is_agreed(
    world_size: int, peer_values: tuple[int, ...] | None, flag: bool, expected: bool
) -> None:
    engine = _engine(world_size=world_size, peer_values=peer_values)

    assert engine._agree_warmup_flag(flag) is expected


def test_shape_list_narrows_to_what_every_rank_proposed() -> None:
    """Attention-DP gives ranks different capacities and so different shapes."""
    engine = _engine(world_size=2)
    engine.dist.allgather = lambda value: [value, [[1, 0], [2, 0]]]

    assert engine._agree_warmup_shapes([(1, 0), (2, 0), (4096, 0)]) == [(1, 0), (2, 0)]


def test_shape_list_is_untouched_without_a_group() -> None:
    engine = _engine(world_size=1)
    configs = [(1, 0), (128, 0)]

    assert engine._agree_warmup_shapes(configs) == configs


def test_warmup_batch_policy_without_communicator() -> None:
    """An engine built without a communicator cannot strand anyone: skip, don't raise."""
    engine = _engine(with_dist=False)

    assert engine._should_run_warmup_batch(None, 128, "general") is False


def test_tp_agreement_needs_a_communicator_not_just_a_mapping() -> None:
    """A mapping can claim TP peers that no communicator exists to reach.

    The agreement check gates on `tp_size`, so it has to answer for that
    state rather than walking into a collective on `None`.
    """
    engine = _engine(tp_size=2, with_dist=False)

    assert engine._should_run_warmup_batch(object(), 128, "general") is True
    assert engine._should_run_warmup_batch(None, 128, "general") is False


def test_tp_disagreement_reports_the_ranks_that_lost_their_batch() -> None:
    """Asymmetric attention-DP capacity is the deadlock this check exists for.

    The rank that still has a batch must not walk into a forward its peer
    will never join, and the failure has to name the peer so the KV cache
    fraction can be raised.

    DWDP is what still reaches the TP-only check: its peers are outside
    ``self.dist``, so no group agreement is available and the narrower TP
    check remains the only thing standing between an asymmetric world and a
    deadlock.
    """
    engine = _engine(world_size=2, dwdp_size=4, tp_size=2, tp_peer_values=(0,))

    with pytest.raises(RuntimeError) as excinfo:
        engine._should_run_warmup_batch(object(), 128, "general")

    message = str(excinfo.value)
    assert "TP rank(s) [1]" in message
    assert "[128, 0]" in message


def test_tp_agreement_lets_a_symmetric_world_run() -> None:
    engine = _engine(world_size=2, tp_size=2, tp_peer_values=(1,))

    assert engine._should_run_warmup_batch(object(), 128, "general") is True


def _general_warmup_engine(*, world_size: int, dwdp_size: int) -> PyTorchModelEngine:
    engine = _engine(world_size=world_size, dwdp_size=dwdp_size)
    batch = object()
    engine._create_warmup_request = mock.Mock(return_value=batch)
    engine._release_batch_context = lambda *_a, **_kw: _released_batch(batch)
    return engine


@pytest.mark.parametrize(
    "world_size,dwdp_size,is_fatal",
    [(1, 0, False), (2, 0, True), (1, 4, True)],
)
def test_general_warmup_oom_policy(world_size: int, dwdp_size: int, is_fatal: bool) -> None:
    engine = _general_warmup_engine(world_size=world_size, dwdp_size=dwdp_size)
    error = torch.OutOfMemoryError("asymmetric OOM")
    engine.forward = mock.Mock(side_effect=error if is_fatal else [error, None])

    with _no_cuda_side_effects() as (empty_cache, _synchronize):
        if is_fatal:
            with pytest.raises(torch.OutOfMemoryError) as excinfo:
                engine._general_warmup_impl(object(), [(128, 0), (64, 0)])
            assert excinfo.value is error
        else:
            engine._general_warmup_impl(object(), [(128, 0), (64, 0)])

    if is_fatal:
        # The remaining shape is never attempted: this rank's peers are
        # already stuck in the failed forward's collectives.
        engine.forward.assert_called_once()
    else:
        assert engine.forward.call_count == 2
        # A retry after an OOM between dispatch() and combine() has to start
        # from a clean MoE all-to-all state.
        engine._reset_moe_alltoall_state.assert_called_once_with()
        empty_cache.assert_called_once_with()


_KV_ALLOC_ERROR = "Can't allocate new blocks for window size 8"


def _mamba_engine(*, world_size: int = 1, dwdp_size: int = 0) -> tuple[PyTorchModelEngine, object]:
    engine = _engine(world_size=world_size, dwdp_size=dwdp_size)
    engine.kv_cache_manager_key = "kv"
    engine.max_num_tokens = 8
    engine.batch_size = 4
    engine.max_seq_len = 8
    engine.original_max_draft_len = 0
    engine.is_draft_model = False
    engine.llm_args = SimpleNamespace(enable_autotuner=False)
    engine.no_cuda_graph = contextlib.nullcontext
    batch = object()
    engine._release_batch_context = lambda *_a, **_kw: _released_batch(batch)

    cache_manager = _StandInMambaCacheManager()
    resource_manager = SimpleNamespace(
        get_resource_manager=lambda key: cache_manager if key == "kv" else None
    )
    return engine, resource_manager


def _run_mamba_warmup(engine: PyTorchModelEngine, resource_manager: object) -> None:
    with (
        mock.patch.object(
            model_engine_module, "MambaHybridCacheManager", _StandInMambaCacheManager
        ),
        mock.patch.object(
            model_engine_module.Mamba2Metadata,
            "force_initial_states_for_warmup",
            side_effect=contextlib.nullcontext,
        ),
        mock.patch.object(model_engine_module, "clear_memory_buffers"),
    ):
        engine._run_mamba_hybrid_warmup(resource_manager)


@pytest.mark.parametrize(
    "error,recoverable",
    [
        (torch.OutOfMemoryError("OOM"), True),
        (RuntimeError(_KV_ALLOC_ERROR), True),
        (RuntimeError("unexpected"), False),
    ],
)
def test_mamba_preforward_error_policy_when_alone(error: Exception, recoverable: bool) -> None:
    engine, resource_manager = _mamba_engine()
    engine._create_warmup_request = mock.Mock(side_effect=error)
    engine.forward = mock.Mock()

    with _no_cuda_side_effects():
        if recoverable:
            _run_mamba_warmup(engine, resource_manager)
        else:
            with pytest.raises(RuntimeError, match="unexpected"):
                _run_mamba_warmup(engine, resource_manager)

    engine.forward.assert_not_called()
    # The failure predates dispatch(), so there is no half-finished MoE
    # all-to-all exchange to unwind.
    engine._reset_moe_alltoall_state.assert_not_called()


def test_mamba_midforward_runtime_error_recovers_when_alone() -> None:
    engine, resource_manager = _mamba_engine()
    engine._create_warmup_request = mock.Mock(return_value=object())
    engine.forward = mock.Mock(side_effect=RuntimeError("mid-forward failure"))

    with _no_cuda_side_effects():
        _run_mamba_warmup(engine, resource_manager)

    # Every shape is attempted, and each failed forward has to leave the MoE
    # all-to-all state clean for the shape that follows it.
    assert engine.forward.call_count >= 1
    assert engine._reset_moe_alltoall_state.call_count == engine.forward.call_count


@pytest.mark.parametrize("phase", ["pre-forward", "mid-forward"])
def test_mamba_error_is_fatal_when_distributed(phase: str) -> None:
    engine, resource_manager = _mamba_engine(world_size=2)
    error = RuntimeError(_KV_ALLOC_ERROR)
    if phase == "pre-forward":
        engine._create_warmup_request = mock.Mock(side_effect=error)
        engine.forward = mock.Mock()
    else:
        engine._create_warmup_request = mock.Mock(return_value=object())
        engine.forward = mock.Mock(side_effect=error)

    with _no_cuda_side_effects():
        with pytest.raises(RuntimeError) as excinfo:
            _run_mamba_warmup(engine, resource_manager)

    assert excinfo.value is error
    # The remaining shape is never attempted: recovering locally would leave
    # peers waiting in a forward this rank has abandoned.
    engine._create_warmup_request.assert_called_once()


def _encoder_engine(*, world_size: int) -> PyTorchModelEngine:
    engine = _engine(world_size=world_size)
    engine.no_encoder_cuda_graph = contextlib.nullcontext
    engine._create_encoder_warmup_inputs = mock.Mock(return_value={"input_ids": [0]})
    return engine


def test_encoder_oom_recovers_when_alone() -> None:
    engine = _encoder_engine(world_size=1)
    engine.encoder_forward = mock.Mock(side_effect=[torch.OutOfMemoryError("OOM"), None])

    with _no_cuda_side_effects() as (empty_cache, _synchronize):
        engine._general_warmup_encoder([(2, 16, 8), (1, 8, 8)])

    assert engine.encoder_forward.call_count == 2
    empty_cache.assert_called_once_with()


def test_encoder_oom_is_fatal_when_distributed() -> None:
    engine = _encoder_engine(world_size=2)
    error = torch.OutOfMemoryError("OOM")
    engine.encoder_forward = mock.Mock(side_effect=error)

    with _no_cuda_side_effects():
        with pytest.raises(torch.OutOfMemoryError) as excinfo:
            engine._general_warmup_encoder([(2, 16, 8), (1, 8, 8)])

    assert excinfo.value is error
    # The second shape is never attempted; peers are stuck in the first.
    engine.encoder_forward.assert_called_once()
