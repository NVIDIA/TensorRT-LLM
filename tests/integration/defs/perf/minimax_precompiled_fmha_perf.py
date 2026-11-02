# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit GPU regression check for a cold MiniMax FMHA cache.

Run this file explicitly on a machine with the packaged FMHA wheel and four
supported GPUs. It uses the same benchmark case and default warmup as the
production perf test, while checking that it never creates a JIT cache.
"""

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
