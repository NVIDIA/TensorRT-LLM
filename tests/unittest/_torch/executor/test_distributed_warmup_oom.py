# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warmup fail-stop policy.

Warmup runs before the executor loop, so a rank that skips or swallows a
failure its peers do not leaves them blocked in a collective forever. Two
mechanisms enforce that, and both are covered here:

  * ``_distributed_warmup_guard`` — the ``PyExecutor`` boundary that arms the
    rank-crash kill for any escaping exception.
  * ``_should_run_warmup_batch`` plus the per-phase handlers — the model
    engine's local-recovery-only-when-alone policy.
"""

import contextlib
from collections.abc import Iterator
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import model_engine as model_engine_module
from tensorrt_llm._torch.pyexecutor import py_executor as py_executor_module
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

# (world_size, dwdp_size) triples spanning the distributed predicate: alone,
# multi-rank via TP/PP/CP, and DWDP (whose peers are invisible to dist).
LOCAL_TOPOLOGY = (1, 0)
DISTRIBUTED_TOPOLOGIES = [(2, 0), (1, 4)]


class _StandInMambaCacheManager:
    """Stand-in for the ``isinstance`` gate in ``_run_mamba_hybrid_warmup``.

    ``MambaHybridCacheManager`` is an abstract mixin (the concrete classes are
    ``CppMambaHybridCacheManager`` / ``MambaHybridCacheManagerV2``), so it
    cannot be instantiated -- not even via ``object.__new__``. Patching the
    name the engine resolves keeps the gate satisfied without dragging in a
    real cache manager.
    """

    def __init__(self, available_tokens: int = 8) -> None:
        self._available_tokens = available_tokens

    def get_num_available_tokens(self, **_kwargs: object) -> int:
        return self._available_tokens


def _engine(*, world_size: int = 1, dwdp_size: int = 0) -> PyTorchModelEngine:
    engine = object.__new__(PyTorchModelEngine)
    engine.dist = SimpleNamespace(world_size=world_size, rank=3)
    engine.mapping = SimpleNamespace(
        dwdp_enabled=dwdp_size > 1,
        has_cp_helix=lambda: False,
        tp_size=1,
    )
    engine._reset_moe_alltoall_state = mock.Mock()
    engine._assert_all_tp_ranks_have_warmup_batch = mock.Mock()
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


# ---------------------------------------------------------------------------
# _distributed_warmup_guard: the PyExecutor boundary.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _guard_env(*, global_size: int) -> Iterator[mock.Mock]:
    with (
        mock.patch.object(py_executor_module, "start_rank_crash_kill_watchdog") as watchdog,
        mock.patch.object(py_executor_module, "global_mpi_size", return_value=global_size),
    ):
        yield watchdog


def _run_guard(*, world_size: int, dwdp_size: int, error: BaseException) -> None:
    dist = SimpleNamespace(world_size=world_size)
    mapping = SimpleNamespace(dwdp_enabled=dwdp_size > 1)
    with py_executor_module._distributed_warmup_guard(dist, mapping):
        raise error


@pytest.mark.parametrize("world_size,dwdp_size", DISTRIBUTED_TOPOLOGIES)
@pytest.mark.parametrize(
    "error",
    [
        torch.OutOfMemoryError("warmup OOM"),
        RuntimeError("rank-local CUDA error"),
        ValueError("programming error"),
    ],
)
def test_guard_arms_kill_and_reraises_original(
    world_size: int, dwdp_size: int, error: Exception
) -> None:
    """Any escaping exception arms the kill, and propagates unchanged."""
    with _guard_env(global_size=8) as watchdog:
        with pytest.raises(type(error)) as excinfo:
            _run_guard(world_size=world_size, dwdp_size=dwdp_size, error=error)

    # Identity, not just type: the client must see the original traceback.
    assert excinfo.value is error
    watchdog.assert_called_once_with(8, error_delivered=None)


@pytest.mark.parametrize("signal", [KeyboardInterrupt, SystemExit])
def test_guard_leaves_teardown_signals_unarmed(signal: type) -> None:
    """Ctrl-C / SystemExit must not become exit 137, as in the event loop."""
    with _guard_env(global_size=4) as watchdog:
        with pytest.raises(signal):
            _run_guard(world_size=4, dwdp_size=0, error=signal())

    watchdog.assert_not_called()


def test_guard_does_not_arm_kill_when_alone() -> None:
    """With no peers there is nobody to strand, so the error just propagates."""
    with _guard_env(global_size=1) as watchdog:
        with pytest.raises(torch.OutOfMemoryError):
            _run_guard(world_size=1, dwdp_size=0, error=torch.OutOfMemoryError("OOM"))

    watchdog.assert_not_called()


def test_guard_uses_global_mpi_size_for_dwdp() -> None:
    """DWDP peers live in COMM_WORLD; dist.world_size is 1 and would not fire."""
    with _guard_env(global_size=4) as watchdog:
        with pytest.raises(torch.OutOfMemoryError):
            _run_guard(world_size=1, dwdp_size=4, error=torch.OutOfMemoryError("OOM"))

    watchdog.assert_called_once_with(4, error_delivered=None)


# ---------------------------------------------------------------------------
# _should_run_warmup_batch: the centralized missing-batch policy.
# ---------------------------------------------------------------------------


def test_missing_batch_is_skipped_when_alone() -> None:
    engine = _engine(world_size=LOCAL_TOPOLOGY[0], dwdp_size=LOCAL_TOPOLOGY[1])
    assert engine._should_run_warmup_batch(None, 128, "general") is False
    engine._assert_all_tp_ranks_have_warmup_batch.assert_not_called()


@pytest.mark.parametrize("world_size,dwdp_size", DISTRIBUTED_TOPOLOGIES)
def test_missing_batch_is_fatal_when_distributed(world_size: int, dwdp_size: int) -> None:
    """A rank-local skip would strand peers already inside the forward."""
    engine = _engine(world_size=world_size, dwdp_size=dwdp_size)
    with pytest.raises(RuntimeError, match="cannot skip the shape"):
        engine._should_run_warmup_batch(None, 128, "general")


def test_present_batch_runs_after_tp_agreement_check() -> None:
    engine = _engine(world_size=2)
    assert engine._should_run_warmup_batch(object(), 128, "general") is True
    engine._assert_all_tp_ranks_have_warmup_batch.assert_called_once()


# ---------------------------------------------------------------------------
# General warmup: local recovery vs. distributed rethrow.
# ---------------------------------------------------------------------------


def _general_warmup_engine(*, world_size: int, dwdp_size: int) -> PyTorchModelEngine:
    engine = _engine(world_size=world_size, dwdp_size=dwdp_size)
    batch = object()
    engine._create_warmup_request = mock.Mock(return_value=batch)
    engine._release_batch_context = lambda *_a, **_kw: _released_batch(batch)
    return engine


def test_general_warmup_oom_recovers_when_alone() -> None:
    engine = _general_warmup_engine(world_size=1, dwdp_size=0)
    engine.forward = mock.Mock(side_effect=[torch.OutOfMemoryError("OOM"), None])

    with _no_cuda_side_effects() as (empty_cache, synchronize):
        engine._general_warmup_impl(object(), [(128, 0), (64, 0)])

    # Recovered, and the next (smaller) shape still ran.
    assert engine.forward.call_count == 2
    engine._reset_moe_alltoall_state.assert_called_once_with()
    empty_cache.assert_called_once_with()
    synchronize.assert_called_once_with()


@pytest.mark.parametrize("world_size,dwdp_size", DISTRIBUTED_TOPOLOGIES)
def test_general_warmup_oom_is_fatal_when_distributed(world_size: int, dwdp_size: int) -> None:
    engine = _general_warmup_engine(world_size=world_size, dwdp_size=dwdp_size)
    error = torch.OutOfMemoryError("asymmetric OOM")
    engine.forward = mock.Mock(side_effect=error)

    with _no_cuda_side_effects() as (empty_cache, _synchronize):
        with pytest.raises(torch.OutOfMemoryError) as excinfo:
            engine._general_warmup_impl(object(), [(128, 0), (64, 0)])

    assert excinfo.value is error
    # No local recovery, and no attempt at the remaining shape.
    engine.forward.assert_called_once()
    engine._reset_moe_alltoall_state.assert_not_called()
    empty_cache.assert_not_called()


# ---------------------------------------------------------------------------
# Mamba hybrid warmup: pre-forward vs. mid-forward, per exception type.
# ---------------------------------------------------------------------------

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
    """Only capacity failures are recoverable; other RuntimeErrors are defects."""
    engine, resource_manager = _mamba_engine()
    engine._create_warmup_request = mock.Mock(side_effect=error)
    engine.forward = mock.Mock()

    with _no_cuda_side_effects():
        if recoverable:
            _run_mamba_warmup(engine, resource_manager)
        else:
            with pytest.raises(RuntimeError, match="unexpected"):
                _run_mamba_warmup(engine, resource_manager)

    # Nothing reached forward either way, so no MoE state can be pending.
    engine.forward.assert_not_called()
    engine._reset_moe_alltoall_state.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        torch.OutOfMemoryError("OOM"),
        RuntimeError("mid-forward failure"),
    ],
)
def test_mamba_midforward_error_recovers_when_alone(error: Exception) -> None:
    engine, resource_manager = _mamba_engine()
    engine._create_warmup_request = mock.Mock(return_value=object())
    engine.forward = mock.Mock(side_effect=error)

    with _no_cuda_side_effects():
        _run_mamba_warmup(engine, resource_manager)

    # Both shapes attempted, and the A2A state reset after each failure.
    assert engine.forward.call_count == 2
    assert engine._reset_moe_alltoall_state.call_count == 2


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

    with _no_cuda_side_effects() as (empty_cache, _synchronize):
        with pytest.raises(RuntimeError) as excinfo:
            _run_mamba_warmup(engine, resource_manager)

    assert excinfo.value is error
    engine._reset_moe_alltoall_state.assert_not_called()
    empty_cache.assert_not_called()


# ---------------------------------------------------------------------------
# Encoder warmup.
# ---------------------------------------------------------------------------


def _encoder_engine(*, world_size: int) -> PyTorchModelEngine:
    engine = _engine(world_size=world_size)
    engine.no_encoder_cuda_graph = contextlib.nullcontext
    engine._create_encoder_warmup_inputs = mock.Mock(return_value={"input_ids": [0]})
    return engine


def test_encoder_oom_recovers_when_alone() -> None:
    engine = _encoder_engine(world_size=1)
    engine.encoder_forward = mock.Mock(side_effect=torch.OutOfMemoryError("OOM"))

    with _no_cuda_side_effects() as (empty_cache, _synchronize):
        engine._general_warmup_encoder([(2, 16, 8)])

    empty_cache.assert_called_once_with()


def test_encoder_oom_is_fatal_when_distributed() -> None:
    engine = _encoder_engine(world_size=2)
    error = torch.OutOfMemoryError("OOM")
    engine.encoder_forward = mock.Mock(side_effect=error)

    with _no_cuda_side_effects() as (empty_cache, _synchronize):
        with pytest.raises(torch.OutOfMemoryError) as excinfo:
            engine._general_warmup_encoder([(2, 16, 8)])

    assert excinfo.value is error
    empty_cache.assert_not_called()
