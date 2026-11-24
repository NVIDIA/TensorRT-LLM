# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``mxfp8_quantize_r128c4``: the bytes of ``torch.ops.trtllm.mxfp8_quantize(x, True, 32)`` for every
BF16 block maximum, every tile shape, partial row / column tiles, and under CUDA graph replay."""

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.gemm.mxfp8_quantize_r128c4 import (  # noqa: E501
    mxfp8_quantize_r128c4,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="needs an SM107 GPU",
)


def _cpp(x: torch.Tensor):
    return torch.ops.trtllm.mxfp8_quantize(x, True, 32)


def _assert_same(got, want) -> None:
    assert got[0].shape == want[0].shape and got[1].shape == want[1].shape
    assert torch.equal(got[0].view(torch.uint8), want[0].view(torch.uint8)), "E4M3 values differ"
    assert torch.equal(got[1], want[1]), "scale bytes differ"


def _activation(m: int, k: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, generator=g, device="cuda")
    x[:, ::7] *= 30.0
    x = x.to(torch.bfloat16)
    x[0, :32] = 0  # an all-zero block: scale byte 0, zero values
    return x


def _every_block_max() -> torch.Tensor:
    """One 32-value block per positive finite BF16 (0x0000..0x7F7F, subnormals included) as its
    signed maximum magnitude, the other 31 values random below it: ``[255, 4096]``."""
    amax = (torch.arange(0, 0x7F80, dtype=torch.int32, device="cuda") << 16).view(torch.float32)
    n = amax.numel()
    g = torch.Generator(device="cuda").manual_seed(7)
    sign = torch.randint(0, 2, (n, 32), generator=g, device="cuda") * 2 - 1
    x = (amax[:, None] * torch.rand(n, 32, generator=g, device="cuda") * sign).to(torch.bfloat16)
    at = torch.randint(0, 32, (n,), generator=g, device="cuda")
    x[torch.arange(n, device="cuda"), at] = (amax * sign[:, 0]).to(torch.bfloat16)
    return x[: n // 128 * 128].reshape(n // 128, 128 * 32).contiguous()


@pytest.mark.parametrize(
    "config", [None, (128, 4, 8), (128, 4, 16), (128, 8, 16), (256, 16, 8), (64, 32, 32)]
)
def test_every_block_maximum(config) -> None:
    x = _every_block_max()
    _assert_same(mxfp8_quantize_r128c4(x, config), _cpp(x))


def test_nan_and_inf_blocks() -> None:
    """NaN and infinity where the C++ kernel's ``__hmax2`` / ``cuda_max`` order decides the scale."""
    x = _activation(256, 4096, 11)
    pick = torch.rand(
        x.shape, generator=torch.Generator(device="cuda").manual_seed(3), device="cuda"
    )
    x[pick < 0.004] = float("nan")
    x[(pick >= 0.004) & (pick < 0.006)] = float("inf")
    x[(pick >= 0.006) & (pick < 0.008)] = -float("inf")
    x[1, :64] = float("nan")  # whole NaN blocks
    x[2, 1:32:2] = float("nan")  # every odd element
    x[3, 0:32:2] = float("nan")  # every even element
    _assert_same(mxfp8_quantize_r128c4(x), _cpp(x))


@pytest.mark.parametrize("k", [32, 96, 160, 1536, 7168])
@pytest.mark.parametrize("m", [1, 127, 129, 300, 2954])
def test_matches_the_cpp_kernel(m: int, k: int) -> None:
    """Partial 128-row tiles (padding rows' scales 0) and K / 32 not a multiple of 4 (padding
    columns' scales 0)."""
    x = _activation(m, k, m * 131 + k)
    _assert_same(mxfp8_quantize_r128c4(x), _cpp(x))


@pytest.mark.parametrize("config", [(128, 4, 8), (128, 8, 16), (256, 16, 32), (64, 128, 8)])
def test_any_tile_matches(config) -> None:
    """Warp tiles up to the scale matrix's 128-row padding and wider than K stay inside it."""
    x = _activation(300, 1568, sum(config))
    _assert_same(mxfp8_quantize_r128c4(x, config), _cpp(x))


def test_a_replayed_graph_matches_eager() -> None:
    x = _activation(2954, 7168, 1)
    eager = mxfp8_quantize_r128c4(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = mxfp8_quantize_r128c4(x)
    graph.replay()
    torch.cuda.synchronize()
    _assert_same(out, eager)


def test_refuses_a_partial_block() -> None:
    with pytest.raises(ValueError, match="K % 32"):
        mxfp8_quantize_r128c4(torch.zeros(4, 48, dtype=torch.bfloat16, device="cuda"))
