# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp patch_embed_conv3d entry.

The reference is a true-fp32 matmul of the flattened patch against the
flattened filter: a different op, accumulation order and memory layout from
the convolution under test, so it checks the contraction pairing rather than
restating it. Beyond the cells: the one-patch-per-row guard and graph capture.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.vision.patch_embed_conv3d import (  # noqa: E501
    patch_embed_conv3d as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import ULP, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "patch_embed_conv3d requires a CUDA device"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Default rtol; atol raised to 4 ulps of the dtype at the result's RMS.

    Summing 1536 signed products leaves a few outputs per thousand near
    zero, where the relative gate measures nothing and torch's 1e-5 atol is
    three orders below one bf16 ulp. Measured at the checkpoint's geometry:
    max |conv - fp32 reference| was 4.31 / 4.38 / 7.81 ulps at the RMS for
    64 / 256 / 4096 bf16 patches (1-2 ulps of the element's own magnitude,
    which the untouched rtol covers); fp16 4.36 / 4.39 / 8.15. Wrong
    variants land 325-392x outside this gate (bias dropped: 10.9x, bounded
    by `max|bias|` ~ 0.02). For fp32 the 1e-5 floor binds, so nothing is
    loosened there.
    """
    assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    rms = ref.float().pow(2).mean().sqrt().item()
    rtol = {torch.bfloat16: 1.6e-2, torch.float16: 1e-3, torch.float32: 1.3e-6}[ref.dtype]
    atol = max(1e-5, 4.0 * rms * ULP[ref.dtype])
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


IN_CHANNELS = 3
KT, KH, KW = 2, 16, 16
EMBED_DIM = 1152
PATCH_NUMEL = IN_CHANNELS * KT * KH * KW  # 1536


def _operands(
    num_patches,
    dtype=torch.bfloat16,
    seed=0,
    in_channels=IN_CHANNELS,
    kernel=(KT, KH, KW),
    embed_dim=EMBED_DIM,
    bias=True,
):
    """Pixels in [-1, 1] (processor mean/std 0.5), weights at the tower's
    initializer_range 0.02."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    kt, kh, kw = kernel
    pixel_values = torch.empty(num_patches, in_channels * kt * kh * kw, device="cuda")
    pixel_values = pixel_values.uniform_(-1.0, 1.0, generator=g).to(dtype)
    weight = (
        0.02 * torch.randn(embed_dim, in_channels, kt, kh, kw, generator=g, device="cuda")
    ).to(dtype)
    bias_t = (0.02 * torch.randn(embed_dim, generator=g, device="cuda")).to(dtype) if bias else None
    return pixel_values, weight, bias_t


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    spec = cell.spec
    pixel_values, weight, bias = _operands(
        spec["num_patches"], dtype=spec["dtype"], seed=zlib.crc32(cell.why.encode())
    )
    before = pixel_values.clone()
    out = op(pixel_values, weight, bias)
    assert out.shape == (pixel_values.shape[0], weight.shape[0])
    assert out.dtype == pixel_values.dtype and out.is_contiguous()
    assert out.data_ptr() != pixel_values.data_ptr()
    assert torch.equal(pixel_values, before), "the pixels are read-only"
    _compare(out, op.reference(pixel_values, weight, bias))


def test_guard_refuses_rows_that_are_not_one_patch() -> None:
    """Two patches per row are split into twice the rows, without an error."""
    pixel_values, weight, bias = _operands(8, seed=8)
    doubled = pixel_values.reshape(4, 2 * PATCH_NUMEL)
    assert op(doubled, weight, bias).shape == (8, EMBED_DIM)
    with pytest.raises(AssertionError, match="each row must be exactly one patch"):
        with validating(op):
            op(doubled, weight, bias)


def test_cuda_graph_capture_and_replay() -> None:
    pixel_values, weight, bias = _operands(64, seed=12)

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            op(pixel_values, weight, bias)
    torch.cuda.current_stream().wait_stream(warmup)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(pixel_values, weight, bias)

    pixel_values.copy_(_operands(64, seed=13)[0])
    graph.replay()
    torch.cuda.synchronize()
    _compare(captured, op.reference(pixel_values, weight, bias))
