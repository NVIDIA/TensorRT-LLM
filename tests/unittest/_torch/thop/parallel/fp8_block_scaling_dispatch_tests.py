# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch.custom_ops.fp8_block_scaling_dispatch import (
    DispatchBackend,
    DispatchCache,
    DispatchEntry,
)
from tensorrt_llm._torch.custom_ops.fp8_block_scaling_dispatch_cache import write_dispatch_cache
from tensorrt_llm._torch.custom_ops.fp8_block_scaling_dispatch_runtime import (
    _load_cache,
    make_dispatch_key,
    make_runtime_identity,
)

_SM_VERSION = sum(
    component * multiplier
    for component, multiplier in zip(torch.cuda.get_device_capability(), (10, 1))
)

pytestmark = pytest.mark.skipif(
    _SM_VERSION != 90,
    reason=f"The test is for Hopper only. Current SM is {_SM_VERSION}.",
)


def _per_block_cast_to_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    padded_m = ((m + 127) // 128) * 128
    padded_n = ((n + 127) // 128) * 128
    x_padded = torch.zeros((padded_m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    blocks = x_padded.view(-1, 128, padded_n // 128, 128)
    amax = blocks.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    quantized = (blocks * (448.0 / amax)).to(torch.float8_e4m3fn)
    return (
        quantized.view_as(x_padded)[:m, :n].contiguous(),
        (amax / 448.0).view(blocks.size(0), blocks.size(2)),
    )


def _inputs(m: int = 1024, n: int = 2112, k: int = 2048) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    a_bf16 = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / k
    b_bf16 = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / k
    a, a_scale = torch.ops.trtllm.fp8_quantize_1x128(a_bf16)
    b, b_scale = _per_block_cast_to_fp8(b_bf16)
    return a, b, a_scale, b_scale


def _cache_path(
    path: Path,
    inputs: tuple[torch.Tensor, ...],
    backend: DispatchBackend,
) -> Path:
    a, b, a_scale, b_scale = inputs
    identity = make_runtime_identity(a)
    key = make_dispatch_key(a, b, a_scale, b_scale)
    write_dispatch_cache(
        path,
        DispatchCache(
            identity=identity,
            entries=(DispatchEntry(key=key, backend=backend),),
        ),
    )
    return path


def test_forced_backend_ops_ignore_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    inputs = _inputs()

    # When
    monkeypatch.setenv("TRTLLM_FP8_BLOCK_SCALING_GEMM_BACKEND", "deep_gemm")
    trt_output = torch.ops.trtllm.fp8_block_scaling_gemm_trtllm(*inputs)
    monkeypatch.setenv("TRTLLM_FP8_BLOCK_SCALING_GEMM_BACKEND", "trtllm")
    deep_gemm_output = torch.ops.trtllm.fp8_block_scaling_gemm_deep_gemm(*inputs)

    # Then
    torch.testing.assert_close(trt_output, deep_gemm_output, atol=1e-3, rtol=1e-3)


def test_validated_cache_routes_public_op_to_deep_gemm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given
    inputs = _inputs()
    cache_path = _cache_path(tmp_path / "dispatch.json", inputs, DispatchBackend.DEEP_GEMM)
    monkeypatch.setenv("TRTLLM_FP8_BLOCK_SCALING_GEMM_DISPATCH_CACHE", str(cache_path))
    _load_cache.cache_clear()

    # When
    output = torch.ops.trtllm.fp8_block_scaling_gemm(*inputs)
    expected = torch.ops.trtllm.fp8_block_scaling_gemm_deep_gemm(*inputs)

    # Then
    torch.testing.assert_close(output, expected, atol=0, rtol=0)


def test_small_m_guard_routes_to_trt_without_using_deep_gemm_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given
    inputs = _inputs(m=512)
    cache_path = _cache_path(tmp_path / "dispatch.json", inputs, DispatchBackend.DEEP_GEMM)
    monkeypatch.setenv("TRTLLM_FP8_BLOCK_SCALING_GEMM_DISPATCH_CACHE", str(cache_path))
    _load_cache.cache_clear()

    # When
    output = torch.ops.trtllm.fp8_block_scaling_gemm(*inputs)
    expected = torch.ops.trtllm.fp8_block_scaling_gemm_trtllm(*inputs)

    # Then
    torch.testing.assert_close(output, expected, atol=0, rtol=0)


def test_deep_gemm_availability_reports_hopper_build_support() -> None:
    # Given
    reference = torch.empty(0, device="cuda")

    # When
    available = torch.ops.trtllm.fp8_block_scaling_gemm_deep_gemm_available(reference)

    # Then
    assert available is True
