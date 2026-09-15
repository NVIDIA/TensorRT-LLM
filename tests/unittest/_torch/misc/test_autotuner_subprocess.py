# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import fcntl
import math
import multiprocessing
import os
import pathlib
import signal
import stat
import time
import types
from unittest import mock

import pytest
import torch

from tensorrt_llm._torch import autotuner


class SelectiveRunner(autotuner.TunableRunner):
    def get_valid_tactics(self, inputs, profile, **kwargs) -> list[int]:
        return [0, 1, 2]

    def should_profile_tactic_in_subprocess(
        self, custom_op, inputs, tactic, tuning_config, **kwargs
    ) -> bool:
        return tactic > 0

    def forward(self, inputs, tactic=-1, **kwargs) -> torch.Tensor:
        return inputs[0]


@pytest.mark.parametrize("enabled", [False, True])
def test_subprocess_profiling_requires_opt_in(monkeypatch, enabled) -> None:
    monkeypatch.delenv("TLLM_AUTOTUNER_MULTIPROCESS", raising=False)
    if enabled:
        monkeypatch.setenv("TLLM_AUTOTUNER_MULTIPROCESS", "1")
    tuner = autotuner.AutoTuner()
    assert tuner._get_subprocess_tactics(
        "test", SelectiveRunner(), [], [0, 1, 2], autotuner.TuningConfig()
    ) == ([1, 2] if enabled else [])


@pytest.mark.parametrize("kind", ["tp", "pp", "ep", "dwdp"])
def test_distributed_runtime_never_spawns(monkeypatch, kind) -> None:
    monkeypatch.setenv("TLLM_AUTOTUNER_MULTIPROCESS", "1")
    tuner = autotuner.AutoTuner()
    # In particular, pure PP has tp_size=1 and no TP Distributed object.
    tuner.mapping = types.SimpleNamespace(
        world_size=1 if kind == "dwdp" else 2,
        tp_size=2 if kind == "tp" else 1,
        dwdp_enabled=kind == "dwdp",
    )
    assert not tuner._is_subprocess_profiling_enabled(autotuner.TuningConfig())


@pytest.mark.parametrize("strategy", list(autotuner.DistributedTuningStrategy))
def test_subprocess_strategy_gate(monkeypatch, strategy) -> None:
    monkeypatch.setenv("TLLM_AUTOTUNER_MULTIPROCESS", "1")
    expected = strategy in (
        autotuner.DistributedTuningStrategy.INDEPENDENT,
        autotuner.DistributedTuningStrategy.PARALLEL,
    )
    assert (
        autotuner.AutoTuner()._is_subprocess_profiling_enabled(
            autotuner.TuningConfig(distributed_tuning_strategy=strategy)
        )
        is expected
    )


def test_runner_opt_in_and_minimum_tactics(monkeypatch) -> None:
    monkeypatch.setenv("TLLM_AUTOTUNER_MULTIPROCESS", "1")
    tuner = autotuner.AutoTuner()
    config = autotuner.TuningConfig()
    runner = SelectiveRunner()
    assert not autotuner.TunableRunner.should_profile_tactic_in_subprocess(
        runner, "test", [], 1, config
    )
    assert tuner._get_subprocess_tactics("test", runner, [], [0, 1], config) == []
    monkeypatch.setenv("TLLM_AUTOTUNER_MP_MIN_TACTICS", "1")
    assert tuner._get_subprocess_tactics("test", runner, [], [0, 1], config) == [1]


@pytest.mark.parametrize("value", [None, True, 7, 0.25, "scalar"])
def test_subprocess_scalar_inputs_survive(value) -> None:
    spec = autotuner._serialize_subprocess_tensor_spec(value)
    assert autotuner._make_subprocess_tensor(spec, None) == value


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.int32, torch.bool, torch.complex64]
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_subprocess_tensor_metadata_survives(dtype, device) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    original = torch.empty(3, 5, dtype=dtype, device=device).t()
    spec = autotuner._serialize_subprocess_tensor_spec(original)
    tensor = autotuner._make_subprocess_tensor(spec, None)
    assert tensor.shape == original.shape
    assert tensor.stride() == original.stride()
    assert tensor.dtype == original.dtype
    assert tensor.device == original.device
    assert tensor.data_ptr() != original.data_ptr()


@pytest.mark.parametrize("fallback", [False, True])
def test_process_start_failure_fallback(monkeypatch, fallback) -> None:
    monkeypatch.setenv("TLLM_AUTOTUNER_MP_FALLBACK_LOCAL_ON_PROCESS_FAILURE", str(int(fallback)))
    tuner = autotuner.AutoTuner()
    calls = []

    def fail_start(**kwargs) -> None:
        raise OSError("cannot start profiling process")

    def local_profile(**kwargs) -> float:
        calls.append(kwargs["tactic"])
        return float(kwargs["tactic"])

    monkeypatch.setattr(autotuner, "ProcessPoolExecutor", fail_start)
    monkeypatch.setattr(tuner, "_profile_single_kernel", local_profile)
    results = tuner._profile_tactics_in_subprocesses(
        "test", SelectiveRunner(), 0, [torch.empty(4)], [1, 2], autotuner.TuningConfig()
    )
    assert calls == ([1, 2] if fallback else [])
    assert [tactic for tactic, _, _ in results] == [1, 2]
    assert all((error is None) == fallback for _, _, error in results)
    assert all(math.isfinite(latency) == fallback for _, latency, _ in results)


def test_local_and_subprocess_results_share_selection_and_cache(monkeypatch) -> None:
    monkeypatch.setenv("TLLM_AUTOTUNER_MULTIPROCESS", "1")
    tuner = autotuner.AutoTuner()
    runner = SelectiveRunner()
    profile = autotuner.OptimizationProfile(shapes=[[autotuner.StaticDim(4)]])
    config = autotuner.TuningConfig()
    local_calls = []

    def local_profile(**kwargs) -> float:
        local_calls.append(kwargs["tactic"])
        return 10.0

    def subprocess_profile(
        custom_op, runner, runner_id, inputs, tactics, config, **kwargs
    ) -> list[tuple[int, float, str | None]]:
        assert tactics == [1, 2]
        return [(1, 1.0, None), (2, float("inf"), "invalid tactic")]

    monkeypatch.setattr(tuner, "_profile_single_kernel", local_profile)
    monkeypatch.setattr(tuner, "_profile_tactics_in_subprocesses", subprocess_profile)
    tuner.stats.tuned_op_profiled_configs["test"] = 0
    tuner.stats.failed_profiling_count["test"] = set()
    result = tuner._profile_runners("test", [runner], [torch.empty(4)], profile, config)
    assert result == (0, 1, 1.0, True)
    assert local_calls == [0]
    key = tuner.profiling_cache.get_cache_key(
        "test", runner, profile.get_opt_shapes(), config, apply_map_to_tuning_buckets=False
    )
    assert tuner.profiling_cache[key] == (0, 1, 1.0)
    assert key in tuner.stats.failed_profiling_count["test"]


def test_rubin_split_k_subprocess_eligibility(monkeypatch) -> None:
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as ops

    if not hasattr(ops, "CuteDSLBf16RubinGemmRunner"):
        pytest.skip("CuTe DSL is unavailable")
    runner = ops.CuteDSLBf16RubinGemmRunner()
    inputs = [torch.empty(16, 4096), torch.empty(256, 4096), torch.empty(16, 256)]
    config = autotuner.TuningConfig()
    tactic = ("base", False, (128, 128), (1, 1), 0, 2)
    monkeypatch.setattr(ops, "get_current_locality_domain", lambda: None)
    assert runner.should_profile_tactic_in_subprocess("test", inputs, tactic, config)
    assert not runner.should_profile_tactic_in_subprocess(
        "test", inputs, tactic[:-1] + (1,), config
    )
    assert not runner.should_profile_tactic_in_subprocess(
        "test", inputs[:2] + [torch.empty(16, 512)], tactic, config
    )
    monkeypatch.setattr(ops, "get_current_locality_domain", lambda: 0)
    assert not runner.should_profile_tactic_in_subprocess("test", inputs, tactic, config)


def test_rubin_split_k_real_subprocesses(monkeypatch) -> None:
    from tensorrt_llm import _utils
    from tensorrt_llm._torch import cute_dsl_utils
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as ops

    if not torch.cuda.is_available() or _utils.get_sm_version() != 107:
        pytest.skip("requires Rubin")
    if not cute_dsl_utils.IS_CUTLASS_DSL_RUBIN_AVAILABLE:
        pytest.skip("requires CuTe DSL with Rubin support")
    monkeypatch.setenv("TLLM_AUTOTUNER_MP_WORKERS", "2")
    # This test must exercise spawn; a local fallback is not a passing result.
    monkeypatch.setenv("TLLM_AUTOTUNER_MP_FALLBACK_LOCAL_ON_PROCESS_FAILURE", "0")
    runner = ops.CuteDSLBf16RubinGemmRunner()
    inputs = [
        torch.randn(16, 4096, dtype=torch.bfloat16, device="cuda"),
        torch.randn(256, 4096, dtype=torch.bfloat16, device="cuda"),
        torch.empty(16, 256, dtype=torch.bfloat16, device="cuda"),
    ]
    config = autotuner.TuningConfig(use_cuda_graph=True)
    tactics = [
        tactic
        for tactic in runner.get_valid_tactics(inputs, autotuner.OptimizationProfile())
        if runner.should_profile_tactic_in_subprocess("test", inputs, tactic, config)
    ][:2]
    assert len(tactics) == 2
    tuner = autotuner.AutoTuner(warmup=1, repeat=2)
    results = tuner._profile_tactics_in_subprocesses("test", runner, 0, inputs, tactics, config)
    assert [tactic for tactic, _, _ in results] == tactics
    assert all(
        error is None and math.isfinite(latency) and latency > 0 for _, latency, error in results
    ), results
    best_tactic, _, _ = min(results, key=lambda result: result[1])
    runner(inputs, tactic=best_tactic)
    torch.testing.assert_close(inputs[2], inputs[0] @ inputs[1].t(), atol=1.0, rtol=0.05)


def test_worker_kernel_failure_is_reported(monkeypatch) -> None:
    monkeypatch.setenv("TLLM_AUTOTUNER_MULTIPROCESS", "1")
    runner = SelectiveRunner()

    def fail_forward(*args, **kwargs) -> None:
        raise ValueError("invalid kernel configuration")

    monkeypatch.setattr(runner, "forward", fail_forward)
    result = autotuner._profile_tactic_in_subprocess(
        {
            "device_index": None,
            "runner": runner,
            "kwargs": {},
            "profile_lock_path": None,
            "input_specs": [autotuner._serialize_subprocess_tensor_spec(torch.empty(4))],
        },
        2,
    )
    assert result[0] == 2
    assert math.isinf(result[1])
    assert "invalid kernel configuration" in result[2]


@pytest.mark.parametrize("value", [object(), [], {}, (1, 2)])
def test_unsupported_subprocess_inputs(value) -> None:
    with pytest.raises(TypeError, match="tensors and scalar inputs"):
        autotuner._serialize_subprocess_tensor_spec(value)


@pytest.mark.parametrize("shape,stride", [((3, 5), (0, 1)), ((3, 5), (1, 0))])
def test_expanded_tensor_rejected_before_allocation(monkeypatch, shape, stride) -> None:
    original = torch.empty_strided(shape, stride)
    with pytest.raises(ValueError, match="expanded tensors"):
        autotuner._serialize_subprocess_tensor_spec(original)
    spec = {
        "is_tensor": True,
        "shape": shape,
        "stride": stride,
        "dtype": "float32",
        "device_type": "cpu",
        "device_index": None,
    }
    with mock.patch.object(torch, "empty_strided") as allocate:
        with pytest.raises(ValueError, match="expanded tensors"):
            autotuner._make_subprocess_tensor(spec, None)
        allocate.assert_not_called()


def test_singleton_zero_stride_is_supported() -> None:
    original = torch.empty_strided((1, 5), (0, 1))
    spec = autotuner._serialize_subprocess_tensor_spec(original)
    reconstructed = autotuner._make_subprocess_tensor(spec, None)
    assert reconstructed.shape == original.shape
    assert reconstructed.stride() == original.stride()


@pytest.mark.parametrize("input_kind", ["unsupported", "expanded"])
def test_serialization_failure_uses_local_fallback(monkeypatch, input_kind) -> None:
    original = object() if input_kind == "unsupported" else torch.empty(1, 5).expand(3, 5)
    tuner = autotuner.AutoTuner()
    monkeypatch.setenv("TLLM_AUTOTUNER_MP_FALLBACK_LOCAL_ON_PROCESS_FAILURE", "1")
    with (
        mock.patch.object(autotuner, "ProcessPoolExecutor") as pool,
        mock.patch.object(tuner, "_profile_single_kernel", return_value=2.0) as local,
    ):
        results = tuner._profile_tactics_in_subprocesses(
            "test", SelectiveRunner(), 0, [original], [1, 2], autotuner.TuningConfig()
        )
    pool.assert_not_called()
    assert local.call_count == 2
    assert local.call_args.kwargs["inputs"][0] is original
    assert results == [(1, 2.0, None), (2, 2.0, None)]


def test_eligibility_exception_uses_local_profiling(monkeypatch) -> None:
    monkeypatch.setenv("TLLM_AUTOTUNER_MULTIPROCESS", "1")
    runner = SelectiveRunner()
    tuner = autotuner.AutoTuner()
    profile = autotuner.OptimizationProfile(shapes=[[autotuner.StaticDim(4)]])
    tuner.stats.tuned_op_profiled_configs["test"] = 0
    tuner.stats.failed_profiling_count["test"] = set()
    with (
        mock.patch.object(
            runner, "should_profile_tactic_in_subprocess", side_effect=RuntimeError("hook failed")
        ),
        mock.patch.object(tuner, "_profile_single_kernel", return_value=1.0) as local,
        mock.patch.object(autotuner, "ProcessPoolExecutor") as pool,
    ):
        result = tuner._profile_runners(
            "test", [runner], [torch.empty(4)], profile, autotuner.TuningConfig()
        )
    pool.assert_not_called()
    assert [call.kwargs["tactic"] for call in local.call_args_list] == [0, 1, 2]
    assert result == (0, 0, 1.0, False)


@pytest.mark.parametrize("workers,expected", [(None, 1), ("0", 1), ("-1", 1), ("2", 2), ("9", 3)])
def test_subprocess_worker_count(monkeypatch, workers, expected) -> None:
    monkeypatch.delenv("TLLM_AUTOTUNER_MP_WORKERS", raising=False)
    if workers is not None:
        monkeypatch.setenv("TLLM_AUTOTUNER_MP_WORKERS", workers)
    assert autotuner.AutoTuner()._get_subprocess_worker_count(3) == expected


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf", "invalid"])
def test_invalid_timeout_falls_back_before_spawn(monkeypatch, timeout) -> None:
    monkeypatch.setenv("TLLM_AUTOTUNER_MP_TIMEOUT_SECONDS", timeout)
    monkeypatch.setenv("TLLM_AUTOTUNER_MP_FALLBACK_LOCAL_ON_PROCESS_FAILURE", "1")
    tuner = autotuner.AutoTuner()
    with (
        mock.patch.object(autotuner, "ProcessPoolExecutor") as pool,
        mock.patch.object(tuner, "_profile_single_kernel", return_value=1.0) as local,
    ):
        results = tuner._profile_tactics_in_subprocesses(
            "test", SelectiveRunner(), 0, [], [1, 2], autotuner.TuningConfig()
        )
    pool.assert_not_called()
    assert local.call_count == 2
    assert results == [(1, 1.0, None), (2, 1.0, None)]


def test_private_default_lock_directory(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("TLLM_AUTOTUNER_MP_PROFILE_LOCK_PATH", raising=False)
    monkeypatch.delenv("TLLM_AUTOTUNER_MP_PROFILE_LOCK", raising=False)
    monkeypatch.setattr(autotuner.tempfile, "gettempdir", lambda: str(tmp_path))
    tuner = autotuner.AutoTuner()
    path = pathlib.Path(tuner._get_subprocess_profile_lock_path(0))
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert path.parent.stat().st_uid == os.getuid()
    assert tuner._get_subprocess_profile_lock_path(0) == str(path)
    with autotuner._subprocess_profile_lock(str(path), exclusive=True):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    path.write_text("keep existing contents")
    with autotuner._subprocess_profile_lock(str(path), exclusive=False):
        assert path.read_text() == "keep existing contents"


@pytest.mark.parametrize("kind", ["symlink", "permissions", "owner"])
def test_unsafe_default_lock_directory(monkeypatch, tmp_path, kind) -> None:
    monkeypatch.delenv("TLLM_AUTOTUNER_MP_PROFILE_LOCK_PATH", raising=False)
    monkeypatch.delenv("TLLM_AUTOTUNER_MP_PROFILE_LOCK", raising=False)
    monkeypatch.setattr(autotuner.tempfile, "gettempdir", lambda: str(tmp_path))
    directory = tmp_path / f"tllm_autotuner_{os.getuid()}"
    if kind == "symlink":
        directory.symlink_to(tmp_path, target_is_directory=True)
    else:
        directory.mkdir(mode=0o700)
        if kind == "permissions":
            directory.chmod(0o755)
        else:
            original = directory.lstat()
            monkeypatch.setattr(
                autotuner.Path,
                "lstat",
                lambda self: types.SimpleNamespace(
                    st_mode=original.st_mode, st_uid=os.getuid() + 1
                ),
            )
    with pytest.raises(PermissionError, match="Unsafe subprocess profiling lock directory"):
        autotuner.AutoTuner()._get_subprocess_profile_lock_path(0)


def test_lock_symlink_cannot_truncate_target(tmp_path) -> None:
    target = tmp_path / "target"
    target.write_text("preserve me")
    path = tmp_path / "profile.lock"
    path.symlink_to(target)
    with pytest.raises(OSError):
        with autotuner._subprocess_profile_lock(str(path), exclusive=True):
            pytest.fail("symlink lock must be rejected")
    assert target.read_text() == "preserve me"


def test_lock_released_and_descriptor_closed_after_exception(tmp_path) -> None:
    path = str(tmp_path / "profile.lock")
    with mock.patch.object(autotuner.os, "close", wraps=os.close) as close:
        with pytest.raises(RuntimeError, match="test failure"):
            with autotuner._subprocess_profile_lock(path, exclusive=True):
                raise RuntimeError("test failure")
        close.assert_called_once()
    with open(path) as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock_file, fcntl.LOCK_UN)


def _process_management_probe(config: dict, tactic: int) -> tuple[int, float, None]:
    """Exercise real spawn, exit and lock waits without launching unsafe kernels."""
    ready = pathlib.Path(config["kwargs"]["marker"] + ".ready")
    if tactic != 0:
        deadline = time.monotonic() + 20
        while not ready.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("first tactic did not finish")
            time.sleep(0.01)
    if tactic == 2:
        marker = pathlib.Path(config["kwargs"]["marker"])
        while not marker.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("second tactic did not start")
            time.sleep(0.01)
    with autotuner._subprocess_profile_lock(config["profile_lock_path"], exclusive=True):
        if tactic == 0:
            ready.touch()
        if tactic == 1:
            pathlib.Path(config["kwargs"]["marker"]).write_text(str(os.getpid()))
            if config["kwargs"]["mode"] == "crash":
                os._exit(17)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            time.sleep(600)
        return tactic, 1.0, None


@pytest.mark.parametrize("mode,workers", [("crash", 1), ("hang", 1), ("hang", 2)])
@pytest.mark.timeout(90)
def test_worker_crash_and_hang_never_retry_in_parent(monkeypatch, tmp_path, mode, workers) -> None:
    monkeypatch.setenv("TLLM_AUTOTUNER_MP_WORKERS", str(workers))
    monkeypatch.setenv("TLLM_AUTOTUNER_MP_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("TLLM_AUTOTUNER_MP_FALLBACK_LOCAL_ON_PROCESS_FAILURE", "1")
    lock_path = str(tmp_path / "profile.lock")
    monkeypatch.setenv("TLLM_AUTOTUNER_MP_PROFILE_LOCK_PATH", lock_path)
    monkeypatch.setattr(autotuner, "_profile_tactic_in_subprocess", _process_management_probe)
    marker = tmp_path / "worker.pid"
    tuner = autotuner.AutoTuner()
    before = {p.pid for p in multiprocessing.active_children()}
    start = time.monotonic()
    with mock.patch.object(tuner, "_profile_tactics_locally_after_subprocess_failure") as local:
        results = tuner._profile_tactics_in_subprocesses(
            "test",
            SelectiveRunner(),
            0,
            [],
            [0, 1, 2],
            autotuner.TuningConfig(),
            mode=mode,
            marker=str(marker),
        )
    assert time.monotonic() - start < 40
    assert marker.exists(), "worker must reach the intentional crash/hang"
    local.assert_not_called()
    assert results[0] == (0, 1.0, None)
    assert all(math.isinf(latency) and error for _, latency, error in results[1:])
    assert {p.pid for p in multiprocessing.active_children()} <= before
    with open(lock_path) as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock_file, fcntl.LOCK_UN)


def test_all_subprocess_tactics_fail_with_warning(monkeypatch) -> None:
    tuner = autotuner.AutoTuner()
    future = mock.Mock()
    future.result.return_value = (1, float("inf"), "CUDA out of memory")
    executor = mock.Mock()
    executor.submit.return_value = future
    monkeypatch.setattr(autotuner, "ProcessPoolExecutor", lambda **kwargs: executor)
    monkeypatch.setattr(autotuner, "as_completed", lambda futures, timeout: iter(futures))
    with (
        mock.patch.object(autotuner.logger, "warning_once") as warning,
        mock.patch.object(tuner, "_profile_tactics_locally_after_subprocess_failure") as local,
    ):
        results = tuner._profile_tactics_in_subprocesses(
            "test", SelectiveRunner(), 0, [], [1], autotuner.TuningConfig()
        )
    assert results == [(1, float("inf"), "CUDA out of memory")]
    local.assert_not_called()
    assert any("All subprocess tactics failed" in call.args[0] for call in warning.call_args_list)


def test_worker_reconstruction_error_is_reported(monkeypatch) -> None:
    monkeypatch.setenv("TLLM_AUTOTUNER_MULTIPROCESS", "1")
    runner = SelectiveRunner()
    with (
        mock.patch.object(
            autotuner, "_make_subprocess_tensor", side_effect=ValueError("bad layout")
        ),
        mock.patch.object(runner, "forward") as forward,
    ):
        result = autotuner._profile_tactic_in_subprocess(
            {
                "device_index": None,
                "runner": runner,
                "kwargs": {},
                "profile_lock_path": None,
                "input_specs": [{}],
            },
            2,
        )
    forward.assert_not_called()
    assert result[0] == 2
    assert math.isinf(result[1])
    assert "bad layout" in result[2]
