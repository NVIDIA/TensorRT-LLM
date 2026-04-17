# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for multi-node support: external launch detection and worker device assignment.

All tests run without GPU — CUDA/dist operations are mocked out.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm._torch.visual_gen.config import VisualGenArgs
from tensorrt_llm._torch.visual_gen.executor import run_diffusion_worker
from tensorrt_llm.visual_gen.visual_gen import DiffusionRemoteClient, _detect_external_launch

# =============================================================================
# _detect_external_launch()
# =============================================================================


class TestDetectExternalLaunch:
    """_detect_external_launch() reads env vars — no GPU required."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        for var in [
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "MASTER_ADDR",
            "MASTER_PORT",
            "SLURM_PROCID",
            "SLURM_NTASKS",
            "SLURM_LOCALID",
        ]:
            monkeypatch.delenv(var, raising=False)

    def test_no_env_vars_returns_none(self):
        assert _detect_external_launch() is None

    def test_torchrun_single_rank_returns_none(self, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")
        assert _detect_external_launch() is None

    def test_torchrun_multi_rank(self, monkeypatch):
        monkeypatch.setenv("RANK", "2")
        monkeypatch.setenv("WORLD_SIZE", "8")
        monkeypatch.setenv("LOCAL_RANK", "2")
        monkeypatch.setenv("MASTER_ADDR", "node0")
        monkeypatch.setenv("MASTER_PORT", "29500")

        rank, local_rank, world_size, master_addr, master_port = _detect_external_launch()
        assert rank == 2
        assert local_rank == 2
        assert world_size == 8
        assert master_addr == "node0"
        assert master_port == 29500

    def test_torchrun_default_local_rank_and_port(self, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("MASTER_ADDR", "node0")

        rank, local_rank, world_size, master_addr, master_port = _detect_external_launch()
        assert local_rank == 0  # defaults to RANK when LOCAL_RANK absent
        assert master_port == 29500  # defaults to 29500 when MASTER_PORT absent

    def test_torchrun_missing_master_addr_raises(self, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "4")
        with pytest.raises(RuntimeError, match="MASTER_ADDR must be set"):
            _detect_external_launch()

    def test_slurm_multi_task(self, monkeypatch):
        monkeypatch.setenv("SLURM_PROCID", "1")
        monkeypatch.setenv("SLURM_NTASKS", "4")
        monkeypatch.setenv("SLURM_LOCALID", "1")
        monkeypatch.setenv("MASTER_ADDR", "node0")
        monkeypatch.setenv("MASTER_PORT", "29500")

        rank, local_rank, world_size, master_addr, master_port = _detect_external_launch()
        assert rank == 1
        assert local_rank == 1
        assert world_size == 4
        assert master_addr == "node0"

    def test_slurm_single_task_returns_none(self, monkeypatch):
        monkeypatch.setenv("SLURM_PROCID", "0")
        monkeypatch.setenv("SLURM_NTASKS", "1")
        assert _detect_external_launch() is None

    def test_slurm_missing_master_addr_raises(self, monkeypatch):
        monkeypatch.setenv("SLURM_PROCID", "0")
        monkeypatch.setenv("SLURM_NTASKS", "4")
        with pytest.raises(RuntimeError, match="MASTER_ADDR must be set"):
            _detect_external_launch()


# =============================================================================
# run_diffusion_worker — device assignment
# =============================================================================


class TestRunDiffusionWorkerDeviceAssignment:
    """Verify device_id assignment logic — no GPU required.

    Regression: adding LOCAL_RANK env-var lookup meant spawned child processes
    (which inherit the parent's LOCAL_RANK, e.g. LOCAL_RANK=0 from SLURM) would
    all compute device_id=0, causing NCCL "Duplicate GPU detected" errors.
    """

    @pytest.fixture(autouse=True)
    def _clean_rank_env(self, monkeypatch):
        for var in ["RANK", "WORLD_SIZE", "LOCAL_RANK"]:
            monkeypatch.delenv(var, raising=False)

    def _call_worker(self, rank, world_size, local_rank, mock_set_device):
        """Invoke run_diffusion_worker with all CUDA/dist calls mocked."""
        from tensorrt_llm._torch.visual_gen.executor import DiffusionExecutor

        mock_exec = MagicMock()
        mock_exec.pipeline = None

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=8),
            patch("torch.cuda.set_device", mock_set_device),
            patch("torch.distributed.init_process_group"),
            patch("torch.distributed.destroy_process_group"),
            patch.object(DiffusionExecutor, "__new__", return_value=mock_exec),
            patch.object(DiffusionExecutor, "__init__", return_value=None),
        ):
            run_diffusion_worker(
                rank=rank,
                world_size=world_size,
                master_addr="127.0.0.1",
                master_port=29500,
                request_queue_addr="tcp://127.0.0.1:29501",
                response_queue_addr="tcp://127.0.0.1:29502",
                diffusion_args=VisualGenArgs(checkpoint_path="/tmp/model"),
                local_rank=local_rank,
            )

    def test_explicit_local_rank_overrides_env(self, monkeypatch):
        """Regression: explicit local_rank=3 must be used even when LOCAL_RANK=0 in env.

        Without passing local_rank=rank in single-node spawn kwargs, workers inherit
        LOCAL_RANK=0 from the parent shell and all get device_id=0.
        """
        monkeypatch.setenv("LOCAL_RANK", "0")  # stale env from parent SLURM job
        mock_set_device = MagicMock()

        self._call_worker(rank=3, world_size=8, local_rank=3, mock_set_device=mock_set_device)

        mock_set_device.assert_called_once_with(3)

    def test_without_local_rank_falls_back_to_env(self, monkeypatch):
        """When local_rank is not passed, LOCAL_RANK env var is used (correct for torchrun)."""
        monkeypatch.setenv("LOCAL_RANK", "5")
        mock_set_device = MagicMock()

        self._call_worker(rank=5, world_size=8, local_rank=None, mock_set_device=mock_set_device)

        mock_set_device.assert_called_once_with(5)

    def test_without_local_rank_no_env_falls_back_to_rank(self, monkeypatch):
        """When local_rank and LOCAL_RANK are both absent, global rank is used."""
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        mock_set_device = MagicMock()

        self._call_worker(rank=6, world_size=8, local_rank=None, mock_set_device=mock_set_device)

        mock_set_device.assert_called_once_with(6)


# =============================================================================
# DiffusionRemoteClient — single-node spawn kwargs
# =============================================================================


class TestSingleNodeSpawnLocalRank:
    """Regression: DiffusionRemoteClient must pass local_rank=rank to each mp.Process.

    Without explicit local_rank in spawn kwargs, workers inherit LOCAL_RANK from the
    parent process environment (e.g., LOCAL_RANK=0 from a SLURM allocation shell).
    All workers then compute device_id=0 and NCCL raises "Duplicate GPU detected".
    """

    @pytest.fixture(autouse=True)
    def _clean_ext_launch_env(self, monkeypatch):
        for var in ["RANK", "WORLD_SIZE", "SLURM_PROCID", "SLURM_NTASKS"]:
            monkeypatch.delenv(var, raising=False)

    def test_spawn_kwargs_include_correct_local_rank(self, monkeypatch):
        """Each mp.Process must receive local_rank=rank, not inherit LOCAL_RANK from env."""
        monkeypatch.setenv("LOCAL_RANK", "0")  # stale env — would assign all workers device 0

        n_workers = 4  # dit_cfg_size=2 * dit_ulysses_size=2

        captured_kwargs = []
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = True

        mock_ctx = MagicMock()
        mock_ctx.Process.side_effect = lambda target, kwargs: (
            captured_kwargs.append(dict(kwargs)) or mock_proc
        )

        args = VisualGenArgs(
            checkpoint_path="/tmp/model",
            parallel={"dit_cfg_size": 2, "dit_ulysses_size": 2},
        )

        # Pre-set all threading.Event instances so event_loop_ready.wait() returns immediately.
        original_event = threading.Event

        def pre_set_event():
            e = original_event()
            e.set()
            return e

        with (
            patch("tensorrt_llm.visual_gen.visual_gen._detect_external_launch", return_value=None),
            patch("tensorrt_llm.visual_gen.visual_gen.mp.get_context", return_value=mock_ctx),
            patch("tensorrt_llm.visual_gen.visual_gen.threading.Thread") as mock_thread_cls,
            patch("tensorrt_llm.visual_gen.visual_gen.threading.Event", side_effect=pre_set_event),
            patch.object(DiffusionRemoteClient, "_wait_ready"),
        ):
            mock_thread_cls.return_value = MagicMock()  # thread.start() is a no-op
            DiffusionRemoteClient(args=args)

        assert len(captured_kwargs) == n_workers, (
            f"Expected {n_workers} spawned processes, got {len(captured_kwargs)}"
        )
        for i, kwargs in enumerate(captured_kwargs):
            assert "local_rank" in kwargs, (
                f"Worker {i}: 'local_rank' missing from spawn kwargs. "
                "Workers would inherit LOCAL_RANK from the parent process env."
            )
            assert kwargs["local_rank"] == i, (
                f"Worker {i}: expected local_rank={i}, got {kwargs['local_rank']}. "
                f"With LOCAL_RANK=0 in env, all workers would get device_id=0."
            )
