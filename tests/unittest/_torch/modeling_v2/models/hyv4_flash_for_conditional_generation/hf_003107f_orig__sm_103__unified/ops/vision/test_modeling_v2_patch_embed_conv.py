# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for this target's patch_embed_conv entry.

The reference is a true-fp32 matmul of the flattened window against the
flattened filter. The op is the same contraction made by cuBLASLt, so each
cell also holds the reference to an fp32 `F.conv2d` of the 4-D windows: that
pins the flatten order the two share to `nn.Conv2d`'s rather than letting one
restate the other. Beyond the cells: each guard against a measured silent
answer, and graph capture.
"""

import contextlib
import zlib

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.ops.vision.patch_embed_conv import (  # noqa: E501
    patch_embed_conv as op,
)

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "patch_embed_conv requires a CUDA device"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """torch's default band for the dtype.

    One fp32 accumulation and one rounding on both sides, so the difference is
    accumulation order alone: nearly every output is bitwise equal and the rest
    are one-ulp flips of the element, inside the default bf16 band. Wrong
    variants (bias dropped, kh/kw transposed, channel-fastest rows) land far
    outside it.
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


def _conv2d_fp32(pixel_values, weight, bias):
    """`nn.Conv2d` semantics, independently: an fp32 cuDNN convolution of the
    4-D windows with the stride equal to the kernel."""
    out_channels, in_channels, kh, kw = weight.shape
    with _cudnn_true_fp32():
        out = F.conv2d(
            pixel_values.float().view(-1, in_channels, kh, kw),
            weight.float(),
            None if bias is None else bias.float(),
            stride=(kh, kw),
        )
    return out.view(-1, out_channels)


IN_CHANNELS = 3
KH, KW = 16, 16
EMBED_DIM = 1152
PATCH_NUMEL = IN_CHANNELS * KH * KW  # 768


def _operands(
    num_rows,
    dtype=torch.bfloat16,
    seed=0,
    out_channels=EMBED_DIM,
    in_channels=IN_CHANNELS,
    kernel=(KH, KW),
    bias=True,
):
    """Pixels in [-1, 1] (processor mean/std 0.5), weights at the tower's
    initializer_range 0.02."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    kh, kw = kernel
    pixel_values = torch.empty(num_rows, in_channels * kh * kw, device="cuda")
    pixel_values = pixel_values.uniform_(-1.0, 1.0, generator=g).to(dtype)
    weight = (0.02 * torch.randn(out_channels, in_channels, kh, kw, generator=g, device="cuda")).to(
        dtype
    )
    bias_t = (
        (0.02 * torch.randn(out_channels, generator=g, device="cuda")).to(dtype) if bias else None
    )
    return pixel_values, weight, bias_t


def _is_close(out, ref) -> bool:
    try:
        torch.testing.assert_close(out, ref)
    except AssertionError:
        return False
    return True


def _measured(out, ref) -> str:
    """The numbers quoted in the entry's note, per cell."""
    diff = (out.float() - ref.float()).abs()
    equal = (out == ref).float().mean().item()
    rel_l2 = (diff.norm() / ref.float().norm()).item()
    # Torch's bf16 defaults: rtol 1.6e-2, atol 1e-5.
    band = (diff / (1e-5 + 1.6e-2 * ref.float().abs())).max().item()
    rms = ref.float().pow(2).mean().sqrt().item()
    ulp = 2.0**-8
    return (
        f"bitwise equal {equal * 100:.3f}%, rel L2 {rel_l2:.2e}, "
        f"worst {band:.3f} of torch's band, max {diff.max().item() / (ulp * rms):.2f} bf16 ulps of RMS"
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    spec = cell.spec
    pixel_values, weight, bias = _operands(
        spec["num_rows"],
        dtype=spec["dtype"],
        seed=zlib.crc32(cell.why.encode()),
        out_channels=spec["out_channels"],
        in_channels=spec["in_channels"],
        kernel=(spec["kernel"], spec["kernel"]),
    )
    before = pixel_values.clone()
    out = op(pixel_values, weight, bias)
    assert out.shape == (pixel_values.shape[0], weight.shape[0])
    assert out.dtype == pixel_values.dtype and out.is_contiguous()
    assert out.data_ptr() != pixel_values.data_ptr()
    assert torch.equal(pixel_values, before), "the pixels are read-only"

    # The reference is the convolution: fp32 against an fp32 F.conv2d.
    ref32 = op.reference(pixel_values.float(), weight.float(), bias.float())
    torch.testing.assert_close(ref32, _conv2d_fp32(pixel_values, weight, bias))

    ref = op.reference(pixel_values, weight, bias)
    print(f"{cell.why}: {_measured(out, ref)}")
    _compare(out, ref)


def test_four_d_windows_are_the_flat_rows() -> None:
    """`[N, in, kh, kw]` is the same contraction as `[N, in*kh*kw]`: the merger
    gathers its 2x2 blocks in that form, and the patch embedding's windows are
    already 4-D in the image processor's output."""
    for seed, (num_rows, in_channels, kernel, out_channels) in enumerate(
        ((256, IN_CHANNELS, (KH, KW), EMBED_DIM), (64, 1152, (2, 2), 4608)), start=31
    ):
        pixel_values, weight, bias = _operands(
            num_rows,
            seed=seed,
            out_channels=out_channels,
            in_channels=in_channels,
            kernel=kernel,
        )
        four_d = pixel_values.view(num_rows, in_channels, *kernel)
        assert four_d.is_contiguous()
        out_4d = op(four_d, weight, bias)
        torch.testing.assert_close(out_4d, op(pixel_values, weight, bias), rtol=0, atol=0)
        _compare(out_4d, op.reference(four_d, weight, bias))


def test_guard_refuses_rows_that_are_not_one_window() -> None:
    """Two windows per row are split into twice the rows, without an error."""
    pixel_values, weight, bias = _operands(8, seed=8)
    doubled = pixel_values.reshape(4, 2 * PATCH_NUMEL)
    assert op(doubled, weight, bias).shape == (8, EMBED_DIM)
    with pytest.raises(AssertionError, match="each row must be exactly one kernel window"):
        with validating(op):
            op(doubled, weight, bias)


def test_guard_refuses_strided_pixel_rows() -> None:
    """A row-strided view (2-D, or 4-D with a batch stride) still views to
    `[N, 768]` and is read as if dense."""
    pixel_values, weight, bias = _operands(256, seed=21)
    ref = op.reference(pixel_values, weight, bias)
    wide = torch.zeros(256, 2 * PATCH_NUMEL, dtype=pixel_values.dtype, device="cuda")
    wide[:, :PATCH_NUMEL] = pixel_values
    batched = torch.zeros(256, 2, IN_CHANNELS, KH, KW, dtype=pixel_values.dtype, device="cuda")
    batched[:, 0] = pixel_values.view(256, IN_CHANNELS, KH, KW)
    for strided in (wide[:, :PATCH_NUMEL], batched[:, 0]):
        assert not strided.is_contiguous()
        assert not _is_close(op(strided, weight, bias), ref), "the hazard this guard covers is gone"
        with pytest.raises(AssertionError, match="pixel_values must be contiguous"):
            with validating(op):
                op(strided, weight, bias)


def test_guard_refuses_a_strided_weight() -> None:
    """A weight sliced along its output channels views to `[1152, 768]` and is
    read as if dense."""
    pixel_values, weight, bias = _operands(256, seed=22)
    ref = op.reference(pixel_values, weight, bias)
    stacked = torch.zeros(2 * EMBED_DIM, IN_CHANNELS, KH, KW, dtype=weight.dtype, device="cuda")
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


@pytest.mark.parametrize("num_rows", [1024, 16384])
def test_cuda_graph_capture_and_replay(num_rows) -> None:
    pixel_values, weight, bias = _operands(num_rows, seed=12)

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            op(pixel_values, weight, bias)
    torch.cuda.current_stream().wait_stream(warmup)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(pixel_values, weight, bias)

    pixel_values.copy_(_operands(num_rows, seed=13)[0])
    graph.replay()
    torch.cuda.synchronize()
    _compare(captured, op.reference(pixel_values, weight, bias))
