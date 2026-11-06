# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""paged_cache.fp8_e4m3_saturate (FP8_SATURATE): the E4M3 cache writes and the transient FP8 query casts saturate to
+-448 like the CUDA writers (__NV_SATFINITE); every finite in-range value keeps the plain cast's bits, values already
in E4M3 pass through, and the kill switch is exactly the plain cast. Where torch's cast already saturates (this image)
the conversion is the cast alone, one kernel; otherwise it clamps first, the same bytes."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels import paged_cache

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP8 target runs on SM107 only",
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fp8_e4m3_saturate_matches_satfinite(dtype, monkeypatch) -> None:
    g = torch.Generator(device="cuda").manual_seed(0)
    inside = (torch.randn(4096, generator=g, device="cuda") * 100).clamp(-448, 448).to(dtype)
    plain = inside.to(torch.float8_e4m3fn)
    assert torch.equal(
        paged_cache.fp8_e4m3_saturate(inside).view(torch.uint8), plain.view(torch.uint8)
    )
    edge = torch.tensor(
        [449.0, 463.0, 464.0, 1e4, float("inf"), -500.0, -float("inf")], dtype=dtype, device="cuda"
    )
    want = torch.tensor([448.0, 448.0, 448.0, 448.0, 448.0, -448.0, -448.0], device="cuda")
    assert torch.equal(paged_cache.fp8_e4m3_saturate(edge).float(), want)
    nan = paged_cache.fp8_e4m3_saturate(torch.tensor([float("nan")], dtype=dtype, device="cuda"))
    assert torch.isnan(nan.float()).all()
    already = edge.clamp(-448, 448).to(torch.float8_e4m3fn)
    assert paged_cache.fp8_e4m3_saturate(already) is already
    monkeypatch.setattr(paged_cache, "FP8_SATURATE", False)
    killed = paged_cache.fp8_e4m3_saturate(edge)
    assert torch.equal(killed.view(torch.uint8), edge.to(torch.float8_e4m3fn).view(torch.uint8))


def test_cache_write_saturates() -> None:
    """The cache-dtype conversion every write_kv_slots branch applies (compared as bytes: torch has no float8
    index_copy_ kernel), and an E4M3 source passing through untouched."""
    values = torch.tensor([[1.0, 500.0, -1e5, float("inf")]], dtype=torch.bfloat16, device="cuda")
    out = paged_cache._to_cache_dtype(values, torch.float8_e4m3fn)
    assert out.dtype == torch.float8_e4m3fn
    want = torch.tensor([[1.0, 448.0, -448.0, 448.0]], device="cuda").to(torch.float8_e4m3fn)
    assert torch.equal(out.view(torch.uint8), want.view(torch.uint8))
    assert paged_cache._to_cache_dtype(out, torch.float8_e4m3fn) is out
    bf = paged_cache._to_cache_dtype(values, torch.bfloat16)
    assert bf.dtype == torch.bfloat16 and torch.equal(bf, values)


_EDGES = [
    448.0,
    449.0,
    463.0,
    464.0,
    1e4,
    float("inf"),
    -448.0,
    -449.0,
    -464.0,
    -1e4,
    -float("inf"),
    float("nan"),
    0.0,
]


def _cuda_kernels(fn) -> int:
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if e.device_type.name == "CUDA")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("cast_saturates", [True, False], ids=["cast_only", "clamp_fallback"])
def test_fp8_e4m3_saturate_bytes_equal_clamp_then_cast(dtype, cast_saturates, monkeypatch) -> None:
    """Both paths give the bytes of clamp(+-448) followed by the cast, edge values included."""
    monkeypatch.setattr(paged_cache, "_CAST_SATURATES", cast_saturates)
    x = torch.tensor(_EDGES, dtype=dtype, device="cuda")
    want = x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    assert torch.equal(paged_cache.fp8_e4m3_saturate(x).view(torch.uint8), want.view(torch.uint8))


def test_fp8_e4m3_saturate_takes_the_cast_only_path_here(monkeypatch) -> None:
    """This image's torch cast saturates, on the CPU (the import-time probe) and on CUDA, so the conversion is one
    kernel; with the probe failing it falls back to clamp + cast, two kernels."""
    assert paged_cache._CAST_SATURATES
    x = torch.tensor(_EDGES, dtype=torch.bfloat16, device="cuda")
    assert torch.equal(
        x.to(torch.float8_e4m3fn).view(torch.uint8),
        x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8),
    )
    assert _cuda_kernels(lambda: paged_cache.fp8_e4m3_saturate(x)) == 1
    monkeypatch.setattr(paged_cache, "_CAST_SATURATES", False)
    assert _cuda_kernels(lambda: paged_cache.fp8_e4m3_saturate(x)) == 2
