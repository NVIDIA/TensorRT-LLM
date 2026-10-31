# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp unified target's patch_embed_conv3d entry.

The reference is a true-fp32 matmul of the flattened patch against the
flattened filter. The op is the same contraction made by cuBLASLt, so each
cell also holds the reference to an fp32 `F.conv3d` of the 5-D patches: that
pins the flatten order the two share to `nn.Conv3d`'s rather than letting one
restate the other. Beyond the cells: each guard against a measured silent
answer, and graph capture.
"""

import contextlib
import zlib

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog.vision.patch_embed_conv3d import (  # noqa: E501
    patch_embed_conv3d as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "patch_embed_conv3d requires a CUDA device"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """torch's default band for the dtype.

    One fp32 accumulation and one rounding on both sides, so the difference is
    accumulation order alone. Measured at the checkpoint's geometry (64 / 4096 /
    32768 bf16 patches): 99.91-99.92% of outputs bitwise equal, the rest
    one-ulp flips of the element, the worst element at 0.49 of the default
    bf16 band. Wrong variants land 4800 (bias dropped) to 1.6e5 (kh/kw
    transposed, channel-fastest rows, kt swapped) times outside it.
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


def _is_close(out, ref) -> bool:
    try:
        torch.testing.assert_close(out, ref)
    except AssertionError:
        return False
    return True


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


def test_guard_refuses_rows_that_are_not_one_patch() -> None:
    """Two patches per row are split into twice the rows, without an error."""
    pixel_values, weight, bias = _operands(8, seed=8)
    doubled = pixel_values.reshape(4, 2 * PATCH_NUMEL)
    assert op(doubled, weight, bias).shape == (8, EMBED_DIM)
    with pytest.raises(AssertionError, match="each row must be exactly one patch"):
        with validating(op):
            op(doubled, weight, bias)


def test_guard_refuses_strided_pixel_rows() -> None:
    """A row-strided view (2-D, or 5-D with a batch stride) still views to
    `[N, 1536]` and is read as if dense."""
    pixel_values, weight, bias = _operands(256, seed=21)
    ref = op.reference(pixel_values, weight, bias)
    wide = torch.zeros(256, 2 * PATCH_NUMEL, dtype=pixel_values.dtype, device="cuda")
    wide[:, :PATCH_NUMEL] = pixel_values
    batched = torch.zeros(256, 2, IN_CHANNELS, KT, KH, KW, dtype=pixel_values.dtype, device="cuda")
    batched[:, 0] = pixel_values.view(256, IN_CHANNELS, KT, KH, KW)
    for strided in (wide[:, :PATCH_NUMEL], batched[:, 0]):
        assert not strided.is_contiguous()
        assert not _is_close(op(strided, weight, bias), ref), "the hazard this guard covers is gone"
        with pytest.raises(AssertionError, match="pixel_values must be contiguous"):
            with validating(op):
                op(strided, weight, bias)


def test_guard_refuses_a_strided_weight() -> None:
    """A weight sliced along its output channels views to `[1152, 1536]` and is
    read as if dense."""
    pixel_values, weight, bias = _operands(256, seed=22)
    ref = op.reference(pixel_values, weight, bias)
    stacked = torch.zeros(2 * EMBED_DIM, IN_CHANNELS, KT, KH, KW, dtype=weight.dtype, device="cuda")
    stacked[::2] = weight
    sliced = stacked[::2]
    assert not _is_close(op(pixel_values, sliced, bias), ref), (
        "the hazard this guard covers is gone"
    )
    with pytest.raises(AssertionError, match="weight must be contiguous"):
        with validating(op):
            op(pixel_values, sliced, bias)


def test_guard_refuses_a_bias_the_epilogue_misreads() -> None:
    """A bias in another dtype, or strided, is read as dense output-dtype
    elements: a plausible answer from the wrong bias. A short one is read past
    its end, so it is refused without calling the op."""
    pixel_values, weight, bias = _operands(256, seed=23)
    ref = op.reference(pixel_values, weight, bias)
    spread = torch.zeros(2 * EMBED_DIM, dtype=bias.dtype, device="cuda")
    spread[::2] = bias
    for bad, match in (
        (bias.float(), "bias dtype"),
        (bias.half(), "bias dtype"),
        (spread[::2], "contiguous"),
    ):
        assert not _is_close(op(pixel_values, weight, bad), ref), (
            "the hazard this guard covers is gone"
        )
        with pytest.raises(AssertionError, match=match):
            with validating(op):
                op(pixel_values, weight, bad)
    with pytest.raises(AssertionError, match="contiguous \\[1152\\]"):
        with validating(op):
            op(pixel_values, weight, bias[: EMBED_DIM // 2].contiguous())


def test_guard_refuses_fp32_with_a_bias() -> None:
    """fp32 input drops the bias: the answer is the bias-free product."""
    pixel_values, weight, bias = _operands(256, dtype=torch.float32, seed=24)
    out = op(pixel_values, weight, bias)
    torch.testing.assert_close(out, op.reference(pixel_values, weight, None))
    assert not _is_close(out, op.reference(pixel_values, weight, bias))
    with pytest.raises(AssertionError, match="dropped for fp32"):
        with validating(op):
            op(pixel_values, weight, bias)


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
