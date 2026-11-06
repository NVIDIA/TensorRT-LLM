# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit GPU regression check for a cold MiniMax FMHA cache.

Run this file explicitly on a machine with the packaged FMHA wheel and four
supported GPUs. It uses the same benchmark case and default warmup as the
production perf test, while checking that it never creates a JIT cache.
Both checks use the perf harness's explicit MPI launcher.

The optional A/B check compares the installed package with a private copy that
omits only precompiled artifacts and bytecode. Both arms disable the perf
harness's extra traffic warmup and retain the original eight warmup requests.
Set LLM_MODELS_ROOT to the directory containing MiniMax-M3-NVFP4.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from .test_perf import run_perf_test

_CASE = (
    "perf/test_perf.py::test_perf["
    "minimax_m3_fp4-bench-pytorch-float4-maxbs:128-maxnt:4096-"
    "input_output_len:1024,1024-ep:4-gpus:4]"
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_minimax_uses_precompiled_fmha_with_cold_cache(
    tmp_path,
    monkeypatch,
    trt_performance_cache_fpath,
    trt_gpu_clock_lock,
    llm_session_data_writer,
    output_dir,
    llm_venv,
    llm_root,
):
    if torch.cuda.device_count() < 4:
        pytest.skip("MiniMax performance case requires four GPUs")

    cache = tmp_path / "cold-fmha-cache"
    monkeypatch.setenv("MINFER_FMHA_CACHE_DIR", str(cache))
    monkeypatch.setenv("TRTLLM_BENCH_USE_MPIRUN", "1")

    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
        require_msa_module,
    )

    require_msa_module()
    from fmha_sm100 import jit

    for kind in ("variants", "plan", "sparse_topk", "reduction"):
        assert jit._aot_module_path(kind) is not None, f"Missing packaged {kind} module"

    run_perf_test(
        _CASE,
        trt_performance_cache_fpath,
        trt_gpu_clock_lock,
        llm_session_data_writer,
        output_dir,
        llm_venv,
        llm_root,
    )

    assert not cache.exists(), f"Benchmark unexpectedly created FMHA JIT cache {cache}"


def _copy_jit_control(source: Path, destination: Path) -> Path:
    """Copy identical sources and headers without touching installed artifacts."""
    package = destination / "fmha_sm100"
    shutil.copytree(source, package, ignore=shutil.ignore_patterns("_aot", "__pycache__"))
    return package


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kernel_mode", ["packaged", "jit"])
def test_minimax_precompiled_fmha_ab(
    kernel_mode,
    tmp_path,
    monkeypatch,
    record_property,
    trt_performance_cache_fpath,
    trt_gpu_clock_lock,
    llm_session_data_writer,
    output_dir,
    llm_venv,
    llm_root,
) -> None:
    if torch.cuda.device_count() < 4:
        pytest.skip("MiniMax performance case requires four GPUs")

    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
        require_msa_module,
    )

    require_msa_module()
    from fmha_sm100 import jit

    artifact_bytes = 0
    for kind in ("variants", "plan", "sparse_topk", "reduction"):
        artifact = jit._aot_module_path(kind)
        assert artifact is not None, f"Missing packaged {kind} module"
        artifact_bytes += artifact.stat().st_size
    record_property("fmha_artifact_bytes", artifact_bytes)
    record_property("fmha_source_sha256", jit._aot_source_digest())
    record_property("kernel_mode", kernel_mode)

    cache = tmp_path / "cold-fmha-cache"
    monkeypatch.setenv("MINFER_FMHA_CACHE_DIR", str(cache))
    monkeypatch.setenv("TRTLLM_BENCH_USE_MPIRUN", "1")
    monkeypatch.setenv("TRTLLM_PERF_WARMUP_PASS", "0")
    if kernel_mode == "jit":
        package = _copy_jit_control(Path(jit.__file__).parent, tmp_path / "control")
        pythonpath = os.environ.get("PYTHONPATH")
        monkeypatch.setenv(
            "PYTHONPATH",
            str(package.parent) + (os.pathsep + pythonpath if pythonpath else ""),
        )
        selected = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import importlib.util; print(importlib.util.find_spec('fmha_sm100').origin)",
            ],
            text=True,
        ).strip()
        assert Path(selected).resolve().parent == package.resolve()
        assert not (package / "_aot").exists()

    arm_output_dir = Path(output_dir) / kernel_mode
    arm_output_dir.mkdir(parents=True, exist_ok=True)
    run_perf_test(
        _CASE,
        trt_performance_cache_fpath,
        trt_gpu_clock_lock,
        llm_session_data_writer,
        str(arm_output_dir),
        llm_venv,
        llm_root,
    )

    if kernel_mode == "jit":
        assert any(cache.rglob("*.so")), "Control did not compile an FMHA module"
    else:
        assert not cache.exists(), f"Benchmark unexpectedly created FMHA JIT cache {cache}"
