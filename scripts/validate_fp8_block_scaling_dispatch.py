#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate forced backends and runtime FP8 dispatch on a Hopper GPU."""

import os
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

from tensorrt_llm._torch.custom_ops.fp8_block_scaling_dispatch import (
    DispatchBackend,
    DispatchCache,
    DispatchEntry,
    DispatchReason,
)
from tensorrt_llm._torch.custom_ops.fp8_block_scaling_dispatch_cache import write_dispatch_cache
from tensorrt_llm._torch.custom_ops.fp8_block_scaling_dispatch_runtime import (
    _load_cache,
    get_dispatch_decision,
    make_dispatch_key,
    make_runtime_identity,
)

_CACHE_ENV = "TRTLLM_FP8_BLOCK_SCALING_GEMM_DISPATCH_CACHE"


def _inputs(m: int, n: int = 2112, k: int = 2048) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    a_bf16 = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / k
    b_bf16 = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / k
    a, a_scale = torch.ops.trtllm.fp8_quantize_1x128(a_bf16)

    padded_n = ((n + 127) // 128) * 128
    b_padded = torch.zeros((padded_n, k), device="cuda", dtype=torch.bfloat16)
    b_padded[:n] = b_bf16
    blocks = b_padded.view(-1, 128, k // 128, 128)
    amax = blocks.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    b = (blocks * (448.0 / amax)).to(torch.float8_e4m3fn)
    b = b.view_as(b_padded)[:n].contiguous()
    b_scale = (amax / 448.0).view(blocks.size(0), blocks.size(2))
    return a, b, a_scale, b_scale


def _write_cache(
    path: Path,
    inputs: tuple[torch.Tensor, ...],
    backend: DispatchBackend,
    *,
    stale: bool = False,
) -> None:
    a, b, a_scale, b_scale = inputs
    identity = make_runtime_identity(a)
    if stale:
        identity = replace(identity, trtllm_build_id="stale-build")
    entry = DispatchEntry(key=make_dispatch_key(a, b, a_scale, b_scale), backend=backend)
    write_dispatch_cache(path, DispatchCache(identity=identity, entries=(entry,)))


def _activate_cache(path: Path) -> None:
    os.environ[_CACHE_ENV] = str(path)
    _load_cache.cache_clear()


def main() -> None:
    print(f"device={torch.cuda.get_device_name()}")
    capability = torch.cuda.get_device_capability()
    print(f"capability={capability}")
    inputs = _inputs(m=1024)
    available = torch.ops.trtllm.fp8_block_scaling_gemm_deep_gemm_available(inputs[0])
    if capability != (9, 0):
        assert available is False
        _activate_cache(Path("/nonexistent/fp8_dispatch_cache.json"))
        decision = get_dispatch_decision(*inputs)
        output = torch.ops.trtllm.fp8_block_scaling_gemm(*inputs)
        expected = torch.ops.trtllm.fp8_block_scaling_gemm_trtllm(*inputs)
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        assert decision.backend is DispatchBackend.TRTLLM
        assert decision.reason is DispatchReason.UNSUPPORTED_ARCH
        print("unsupported_arch=trtllm:PASS")
        print("RESULT PASS")
        return

    assert available is True

    trt_output = torch.ops.trtllm.fp8_block_scaling_gemm_trtllm(*inputs)
    deep_gemm_output = torch.ops.trtllm.fp8_block_scaling_gemm_deep_gemm(*inputs)
    torch.testing.assert_close(trt_output, deep_gemm_output, atol=1e-3, rtol=1e-3)
    print("forced_backends=PASS")

    with tempfile.TemporaryDirectory() as temporary_directory:
        cache_path = Path(temporary_directory) / "dispatch.json"
        _write_cache(cache_path, inputs, DispatchBackend.DEEP_GEMM)
        cache_mtime_ns = cache_path.stat().st_mtime_ns
        _activate_cache(cache_path)
        decision = get_dispatch_decision(*inputs)
        output = torch.ops.trtllm.fp8_block_scaling_gemm(*inputs)
        torch.testing.assert_close(output, deep_gemm_output, atol=0, rtol=0)
        assert decision.backend is DispatchBackend.DEEP_GEMM
        assert cache_path.stat().st_mtime_ns == cache_mtime_ns
        print(f"cache_hit={decision.reason.value}:PASS")

        stale_path = Path(temporary_directory) / "stale.json"
        _write_cache(stale_path, inputs, DispatchBackend.DEEP_GEMM, stale=True)
        _activate_cache(stale_path)
        decision = get_dispatch_decision(*inputs)
        output = torch.ops.trtllm.fp8_block_scaling_gemm(*inputs)
        torch.testing.assert_close(output, trt_output, atol=0, rtol=0)
        print(f"stale_cache={decision.reason.value}:PASS")

        small_inputs = _inputs(m=512)
        small_path = Path(temporary_directory) / "small.json"
        _write_cache(small_path, small_inputs, DispatchBackend.DEEP_GEMM)
        _activate_cache(small_path)
        decision = get_dispatch_decision(*small_inputs)
        output = torch.ops.trtllm.fp8_block_scaling_gemm(*small_inputs)
        expected = torch.ops.trtllm.fp8_block_scaling_gemm_trtllm(*small_inputs)
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        print(f"small_m={decision.reason.value}:PASS")

        _activate_cache(cache_path)
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            captured_output = torch.ops.trtllm.fp8_block_scaling_gemm(*inputs)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured_output, trt_output, atol=0, rtol=0)
        print("cuda_graph_capture=trtllm:PASS")

    print("RESULT PASS")


if __name__ == "__main__":
    main()
