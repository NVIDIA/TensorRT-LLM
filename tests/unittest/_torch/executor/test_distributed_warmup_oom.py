# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
from collections.abc import Iterator
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import model_engine as model_engine_module
from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import MambaHybridCacheManager
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine


def _engine(*, world_size: int, dwdp_enabled: bool = False) -> PyTorchModelEngine:
    engine = object.__new__(PyTorchModelEngine)
    engine.dist = SimpleNamespace(world_size=world_size, rank=3)
    engine.mapping = SimpleNamespace(
        dwdp_enabled=dwdp_enabled,
        has_cp_helix=lambda: False,
        tp_size=1,
    )
    return engine


@contextlib.contextmanager
def _released_batch(batch: object) -> Iterator[object]:
    yield batch


@contextlib.contextmanager
def _fatal_path() -> Iterator[tuple[mock.Mock, mock.Mock]]:
    with (
        mock.patch.object(model_engine_module, "global_mpi_rank", return_value=7),
        mock.patch.object(model_engine_module.logger, "error") as log_error,
        mock.patch.object(model_engine_module.hang_detector, "propagate_hard_kill") as hard_kill,
    ):
        yield log_error, hard_kill


def _general_warmup_engine(*, world_size: int) -> PyTorchModelEngine:
    engine = _engine(world_size=world_size)
    batch = object()
    engine._create_warmup_request = mock.Mock(return_value=batch)
    engine._release_batch_context = lambda *_args, **_kwargs: _released_batch(batch)
    engine._assert_all_tp_ranks_have_warmup_batch = mock.Mock()
    engine._reset_moe_alltoall_state = mock.Mock()
    return engine


def test_general_distributed_oom_hard_kills_and_fails_closed() -> None:
    engine = _general_warmup_engine(world_size=2)
    error = torch.OutOfMemoryError("asymmetric OOM")
    engine.forward = mock.Mock(side_effect=error)

    with (
        _fatal_path() as (log_error, hard_kill),
        mock.patch.object(torch.cuda, "empty_cache") as empty_cache,
        mock.patch.object(torch.cuda, "synchronize") as synchronize,
        pytest.raises(torch.OutOfMemoryError, match="asymmetric OOM"),
    ):
        engine._general_warmup_impl(object(), [(128, 0), (64, 0)])

    hard_kill.assert_called_once_with()
    engine._reset_moe_alltoall_state.assert_not_called()
    empty_cache.assert_not_called()
    synchronize.assert_not_called()
    engine.forward.assert_called_once()
    message = log_error.call_args.args[0]
    assert "global_rank=7" in message
    assert "model_rank=3" in message
    assert "num_tokens=128" in message


def test_general_distributed_oom_hard_kills_when_logging_fails() -> None:
    engine = _general_warmup_engine(world_size=2)
    engine.forward = mock.Mock(side_effect=torch.OutOfMemoryError("original OOM"))

    with (
        mock.patch.object(model_engine_module, "global_mpi_rank", return_value=7),
        mock.patch.object(
            model_engine_module.logger,
            "error",
            side_effect=RuntimeError("logging failed"),
        ),
        mock.patch.object(model_engine_module.hang_detector, "propagate_hard_kill") as hard_kill,
        pytest.raises(torch.OutOfMemoryError, match="original OOM"),
    ):
        engine._general_warmup_impl(object(), [(128, 0)])

    hard_kill.assert_called_once_with()


def test_general_single_rank_oom_recovers_and_continues() -> None:
    engine = _general_warmup_engine(world_size=1)
    engine.forward = mock.Mock(side_effect=[torch.OutOfMemoryError("OOM"), None])

    with (
        mock.patch.object(model_engine_module.hang_detector, "propagate_hard_kill") as hard_kill,
        mock.patch.object(torch.cuda, "empty_cache") as empty_cache,
        mock.patch.object(torch.cuda, "synchronize") as synchronize,
    ):
        engine._general_warmup_impl(object(), [(128, 0), (64, 0)])

    hard_kill.assert_not_called()
    engine._reset_moe_alltoall_state.assert_called_once_with()
    empty_cache.assert_called_once_with()
    synchronize.assert_called_once_with()
    assert engine.forward.call_count == 2


@pytest.mark.parametrize(
    ("world_size", "dwdp_enabled"),
    [(2, False), (1, True)],
)
def test_distributed_forward_detection_covers_tp1_and_dwdp(
    world_size: int, dwdp_enabled: bool
) -> None:
    engine = _engine(world_size=world_size, dwdp_enabled=dwdp_enabled)
    assert engine.mapping.tp_size == 1
    assert engine._is_distributed_forward()


def _mamba_warmup_engine(*, world_size: int) -> tuple[PyTorchModelEngine, object]:
    engine = _engine(world_size=world_size)
    kv_cache_manager = object.__new__(MambaHybridCacheManager)
    kv_cache_manager.get_num_available_tokens = mock.Mock(return_value=8)
    engine.kv_cache_manager_key = "kv"
    engine.max_num_tokens = 8
    engine.batch_size = 4
    engine.max_seq_len = 8
    engine.original_max_draft_len = 0
    engine.llm_args = SimpleNamespace(enable_autotuner=False)
    engine.no_cuda_graph = contextlib.nullcontext
    batch = object()
    engine._release_batch_context = lambda *_args, **_kwargs: _released_batch(batch)
    engine._assert_all_tp_ranks_have_warmup_batch = mock.Mock()
    engine._reset_moe_alltoall_state = mock.Mock()
    engine.is_draft_model = False

    resource_manager = SimpleNamespace(
        get_resource_manager=lambda key: kv_cache_manager if key == "kv" else None
    )
    return engine, resource_manager


def _run_mamba_warmup(engine: PyTorchModelEngine, resource_manager: object) -> None:
    with (
        mock.patch.object(
            model_engine_module.Mamba2Metadata,
            "force_initial_states_for_warmup",
            side_effect=contextlib.nullcontext,
        ),
        mock.patch.object(model_engine_module, "clear_memory_buffers"),
    ):
        engine._run_mamba_hybrid_warmup(resource_manager)


def test_mamba_expected_preforward_kv_allocation_error_recovers() -> None:
    engine, resource_manager = _mamba_warmup_engine(world_size=1)
    engine._create_warmup_request = mock.Mock(
        side_effect=[RuntimeError("Can't allocate new blocks for window size 8"), object()]
    )
    engine.forward = mock.Mock()

    with (
        mock.patch.object(model_engine_module.hang_detector, "propagate_hard_kill") as hard_kill,
        mock.patch.object(torch.cuda, "empty_cache"),
        mock.patch.object(torch.cuda, "synchronize"),
    ):
        _run_mamba_warmup(engine, resource_manager)

    hard_kill.assert_not_called()
    engine.forward.assert_called_once()


def test_mamba_distributed_preforward_kv_allocation_error_is_fatal() -> None:
    engine, resource_manager = _mamba_warmup_engine(world_size=2)
    engine._create_warmup_request = mock.Mock(
        side_effect=RuntimeError("Can't allocate new blocks for window size 8")
    )
    engine.forward = mock.Mock()

    with (
        _fatal_path() as (log_error, hard_kill),
        mock.patch.object(torch.cuda, "empty_cache") as empty_cache,
        pytest.raises(RuntimeError, match="Can't allocate new blocks"),
    ):
        _run_mamba_warmup(engine, resource_manager)

    hard_kill.assert_called_once_with()
    engine.forward.assert_not_called()
    empty_cache.assert_not_called()


def test_mamba_distributed_unexpected_preforward_runtime_error_is_fatal() -> None:
    engine, resource_manager = _mamba_warmup_engine(world_size=2)
    engine._create_warmup_request = mock.Mock(side_effect=RuntimeError("unexpected"))
    engine.forward = mock.Mock()

    with (
        _fatal_path() as (_log_error, hard_kill),
        mock.patch.object(torch.cuda, "empty_cache") as empty_cache,
        pytest.raises(RuntimeError, match="unexpected"),
    ):
        _run_mamba_warmup(engine, resource_manager)

    hard_kill.assert_called_once_with()
    engine.forward.assert_not_called()
    empty_cache.assert_not_called()


def test_mamba_distributed_preforward_oom_is_fatal() -> None:
    engine, resource_manager = _mamba_warmup_engine(world_size=2)
    engine._create_warmup_request = mock.Mock(side_effect=torch.OutOfMemoryError("OOM"))
    engine.forward = mock.Mock()

    with (
        _fatal_path() as (log_error, hard_kill),
        mock.patch.object(torch.cuda, "empty_cache") as empty_cache,
        pytest.raises(torch.OutOfMemoryError, match="OOM"),
    ):
        _run_mamba_warmup(engine, resource_manager)

    hard_kill.assert_called_once_with()
    engine.forward.assert_not_called()
    engine._reset_moe_alltoall_state.assert_not_called()
    empty_cache.assert_not_called()


def test_mamba_single_rank_midforward_oom_recovers() -> None:
    engine, resource_manager = _mamba_warmup_engine(world_size=1)
    engine._create_warmup_request = mock.Mock(return_value=object())
    engine.forward = mock.Mock(side_effect=[torch.OutOfMemoryError("OOM"), None])

    with (
        mock.patch.object(model_engine_module.hang_detector, "propagate_hard_kill") as hard_kill,
        mock.patch.object(torch.cuda, "empty_cache") as empty_cache,
        mock.patch.object(torch.cuda, "synchronize") as synchronize,
    ):
        _run_mamba_warmup(engine, resource_manager)

    hard_kill.assert_not_called()
    engine._reset_moe_alltoall_state.assert_called_once_with()
    assert empty_cache.call_count == 2
    synchronize.assert_called_once_with()
    assert engine.forward.call_count == 2


@pytest.mark.parametrize("error", [torch.OutOfMemoryError("OOM"), RuntimeError("unexpected")])
def test_mamba_distributed_midforward_error_is_fatal(error: Exception) -> None:
    engine, resource_manager = _mamba_warmup_engine(world_size=2)
    engine._create_warmup_request = mock.Mock(return_value=object())
    engine.forward = mock.Mock(side_effect=error)

    with (
        _fatal_path() as (log_error, hard_kill),
        mock.patch.object(torch.cuda, "empty_cache") as empty_cache,
        pytest.raises(type(error), match=str(error)),
    ):
        _run_mamba_warmup(engine, resource_manager)

    hard_kill.assert_called_once_with()
    engine._reset_moe_alltoall_state.assert_not_called()
    empty_cache.assert_not_called()


def _encoder_warmup_engine(*, world_size: int) -> PyTorchModelEngine:
    engine = _engine(world_size=world_size)
    engine.no_encoder_cuda_graph = contextlib.nullcontext
    engine._create_encoder_warmup_inputs = mock.Mock(return_value={"input_ids": [0]})
    return engine


def test_encoder_distributed_oom_is_fatal() -> None:
    engine = _encoder_warmup_engine(world_size=2)
    engine.encoder_forward = mock.Mock(side_effect=torch.OutOfMemoryError("OOM"))

    with (
        _fatal_path() as (log_error, hard_kill),
        mock.patch.object(torch.cuda, "empty_cache") as empty_cache,
        mock.patch.object(torch.cuda, "synchronize") as synchronize,
        pytest.raises(torch.OutOfMemoryError, match="OOM"),
    ):
        engine._general_warmup_encoder([(2, 16, 8)])

    hard_kill.assert_called_once_with()
    empty_cache.assert_not_called()
    synchronize.assert_not_called()


def test_encoder_single_rank_oom_recovers() -> None:
    engine = _encoder_warmup_engine(world_size=1)
    engine.encoder_forward = mock.Mock(side_effect=torch.OutOfMemoryError("OOM"))

    with (
        mock.patch.object(model_engine_module.hang_detector, "propagate_hard_kill") as hard_kill,
        mock.patch.object(torch.cuda, "empty_cache") as empty_cache,
        mock.patch.object(torch.cuda, "synchronize") as synchronize,
    ):
        engine._general_warmup_encoder([(2, 16, 8)])

    hard_kill.assert_not_called()
    empty_cache.assert_called_once_with()
    synchronize.assert_not_called()
