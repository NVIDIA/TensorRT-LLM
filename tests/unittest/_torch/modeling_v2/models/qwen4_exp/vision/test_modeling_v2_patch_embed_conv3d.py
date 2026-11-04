# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp default target's patch_embed_conv3d entry.

The reference is a true-fp32 matmul of the flattened patch against the
flattened filter. The op is the same contraction made by cuBLASLt, so each
cell also holds the reference to an fp32 `F.conv3d` of the 5-D patches: that
pins the flatten order the two share to `nn.Conv3d`'s rather than letting one
restate the other.
"""

import contextlib
import zlib

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.vision.patch_embed_conv3d import (  # noqa: E501
    patch_embed_conv3d as op,
)

__extra_import_path__ = ["../../.."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "patch_embed_conv3d requires a CUDA device"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """torch's default band for the dtype.

    One fp32 accumulation and one rounding on both sides, so the difference is
    accumulation order alone: nearly every output is bitwise equal and the rest
    are one-ulp flips of the element, inside the default bf16 band. Wrong
    variants (bias dropped, kh/kw transposed, channel-fastest rows, kt swapped)
    land thousands of times outside it.
    """
    assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    torch.testing.assert_close(out, ref)


@contextlib.contextmanager
def _cudnn_true_fp32():
    prev = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = prev


def _conv3d_fp32(pixel_values, weight, bias):
    """`nn.Conv3d` semantics, independently: an fp32 cuDNN convolution of the
    5-D patches with the stride equal to the kernel."""
    out_channels, in_channels, kt, kh, kw = weight.shape
    with _cudnn_true_fp32():
        out = F.conv3d(
            pixel_values.float().view(-1, in_channels, kt, kh, kw),
            weight.float(),
            None if bias is None else bias.float(),
            stride=(kt, kh, kw),
        )
    return out.view(-1, out_channels)


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

    # The reference is the convolution: fp32 against an fp32 F.conv3d.
    ref32 = op.reference(pixel_values.float(), weight.float(), bias.float())
    torch.testing.assert_close(ref32, _conv3d_fp32(pixel_values, weight, bias))

    _compare(out, op.reference(pixel_values, weight, bias))
