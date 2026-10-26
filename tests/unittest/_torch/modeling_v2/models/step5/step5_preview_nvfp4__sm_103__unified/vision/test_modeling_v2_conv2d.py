# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the step5 conv2d entry (patch embedding and downsamplers).

The reference is im2col (`F.unfold`) and a true-fp32 matmul, rounded once: a
different op, accumulation order and memory layout from the convolution under
test, so it checks the window arithmetic (stride, padding, kernel orientation)
rather than restating it. Outputs are compared as the rows the tower consumes,
one `[out_channels]` vector per output position.

The bias of a convolution under torch is added to the conv result after that
result has been rounded to bf16, so the bias cells carry two roundings where the
reference carries one; `test_bias_is_added_after_rounding` pins that model.
"""

import zlib

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.step5.step5_preview_nvfp4__sm_103__unified.ops.vision.conv2d import (  # noqa: E501
    DOWNSAMPLER1,
    DOWNSAMPLER2,
    PATCH_EMBED,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.step5.step5_preview_nvfp4__sm_103__unified.ops.vision.conv2d import (  # noqa: E501
    conv2d as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import ULP, assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "conv2d requires a CUDA device"

#: The band, in bf16 units (2^-8) of each output row's largest magnitude.
#: Let ulp_e be the spacing at an element; within one binade of the row max M,
#: ulp_e <= 2^-7 * M = 2 units. The reference rounds the exact sum once (<= 0.5
#: ulp_e). The op rounds its fp32 sum (accumulation-order error ~2^-20 relative,
#: far below 0.5 ulp_e) and, with a bias, rounds again after the bf16 add: <= 1
#: ulp_e from the exact result. So |op - reference| <= 1.5 ulp_e <= 3 units.
#: The relative RMS of two independent roundings against one is 0.5 ulp_e, and
#: ulp_e / |x| averages about 1.45 units over a Gaussian output: about 0.73.
#: Measured on sm_103 over the cells: element 1.98 (no bias) and 2.00 (bias)
#: units; relative RMS 0.017 (no bias) and 0.738 (bias).
ELEMENT_ULP = 3.0
RMS_ULP = 1.0

#: Input processor normalization (CLIP mean and std): pixels in [0, 1] map to
#: about [-1.79, 2.15] per channel.
_MEAN = (0.48145466, 0.4578275, 0.40821073)
_STD = (0.26862954, 0.26130258, 0.27577711)


def _rows(t: torch.Tensor) -> torch.Tensor:
    """`[N, K, P, Q]` -> `[N*P*Q, K]`, the tower's row view."""
    return t.permute(0, 2, 3, 1).reshape(-1, t.shape[1])


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    assert_within_ulp(_rows(out), _rows(ref), element_ulp=ELEMENT_ULP, rms_ulp=RMS_ULP)


def _ulps(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """(max element error, relative RMS), in the band's units."""
    o, r = _rows(out).float(), _rows(ref).float()
    scale = r.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    element = ((o - r).abs() / scale).max().item() / ULP[torch.bfloat16]
    rms = ((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()).item() / ULP[torch.bfloat16]
    return element, rms


def _build(spec: dict, seed: int):
    """Operands of one cell.

    Pixels are uniform [0, 1] through the processor's normalization; a tower
    activation is N(0, 1) laid out `[N, H, W, C]` (the tower's rows). Weights
    are N(0, 1/fan_in), the bias N(0, 0.02^2).
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    tiles, side = spec["tiles"], spec["side"]
    in_ch, out_ch, kernel = spec["in_channels"], spec["out_channels"], spec["kernel"]
    if in_ch == 3:
        mean = torch.tensor(_MEAN, device="cuda")[:, None, None]
        std = torch.tensor(_STD, device="cuda")[:, None, None]
        x = (torch.rand(tiles, 3, side, side, generator=g, device="cuda") - mean) / std
    else:
        x = torch.randn(tiles, side, side, in_ch, generator=g, device="cuda").permute(0, 3, 1, 2)
    x = x.to(spec["dtype"])
    if spec["layout"] == "nhwc":
        x = x.contiguous(memory_format=torch.channels_last)
    else:
        assert spec["layout"] == "nchw", spec["layout"]
        x = x.contiguous()
    fan_in = in_ch * kernel * kernel
    weight = torch.randn(out_ch, in_ch, kernel, kernel, generator=g, device="cuda")
    weight = (weight / fan_in**0.5).to(spec["dtype"])
    bias = None
    if spec["bias"]:
        bias = (0.02 * torch.randn(out_ch, generator=g, device="cuda")).to(spec["dtype"])
    return x, weight, bias, spec["stride"], spec["padding"]


def _spec(conv: tuple, tiles: int, side: int, layout: str = "nhwc") -> dict:
    in_ch, out_ch, kernel, stride, padding, bias = conv
    return dict(
        tiles=tiles,
        side=side,
        in_channels=in_ch,
        out_channels=out_ch,
        kernel=kernel,
        stride=stride,
        padding=padding,
        bias=bias,
        layout=layout,
        dtype=torch.bfloat16,
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    spec = cell.spec
    x, weight, bias, stride, padding = _build(spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    before = x.clone()
    out = op(x, weight, bias, stride, padding)
    side_out = (spec["side"] + 2 * padding - spec["kernel"]) // stride + 1
    assert out.shape == (spec["tiles"], spec["out_channels"], side_out, side_out)
    assert out.dtype == x.dtype
    channels_last = spec["layout"] == "nhwc"
    assert out.is_contiguous(memory_format=torch.channels_last) == channels_last
    assert out.is_contiguous() != channels_last
    assert torch.equal(x, before), "the input is read-only"
    _compare(out, op.reference(x, weight, bias, stride, padding))


@pytest.mark.parametrize(
    "conv, tiles, side", [(DOWNSAMPLER1, 4, 52), (DOWNSAMPLER2, 4, 26)], ids=["ds1", "ds2"]
)
def test_bias_is_added_after_rounding(conv, tiles, side) -> None:
    """The op equals `bf16(bf16(conv) + bias)` up to its accumulation order.

    Elements differ from that two-rounding model only where the op's fp32 sum
    and the exact one round to different bf16 values, which the bias add then
    carries; from the once-rounded reference about a quarter of them differ.
    An element the model misses is one whose conv part already differs, so the
    op with a bias misses the model on no more elements than the op without one
    misses the reference. A carried miss is up to 1 ulp_e of the conv value plus
    a re-rounding of the sum: at most 2 ulp_e, 4 units.

    Measured on sm_103 at 4, 47 and 92 tiles: 7.2e-3 (downsampler 1) and 1.37e-2
    (downsampler 2) of the conv parts round apart; 5.2e-3 and 1.04e-2 of the
    outputs miss the model, by at most 2.84 units; 0.257-0.260 miss the
    once-rounded reference.
    """
    x, weight, bias, stride, padding = _build(_spec(conv, tiles, side), seed=21)
    out = op(x, weight, bias, stride, padding)
    unbiased = op.reference(x, weight, None, stride, padding)
    off_conv = (op(x, weight, None, stride, padding) != unbiased).float().mean().item()
    twice = unbiased + bias[:, None, None]
    element, _ = _ulps(out, twice)
    off_twice = (out != twice).float().mean().item()
    once = op.reference(x, weight, bias, stride, padding)
    off_once = (out != once).float().mean().item()
    assert element <= 4.0, element
    assert off_once > 0.1, off_once
    assert off_twice <= off_conv, (off_twice, off_conv)
    assert off_twice <= off_once / 10, (off_twice, off_once)


@pytest.mark.parametrize(
    "conv, side",
    [(PATCH_EMBED, 728), (DOWNSAMPLER1, 52), (DOWNSAMPLER2, 26)],
    ids=["patch", "ds1", "ds2"],
)
def test_wrong_variants_land_outside_the_band(conv, side) -> None:
    """Each plausible slip fails the band, and by how much.

    Measured on sm_103 over the cells' shapes and these (element / RMS units): a
    flipped kernel 454-625 / 339-355; a transposed kernel 396-594 / 295-342;
    input channels rolled by one 456-588 / 358-362; padding 0 with the result
    shifted 505-662 / 348-357; the bias dropped 6.5-9.4 / 5.1-5.4 (bounded by
    the bias's 0.02 scale against outputs of about unit RMS), the one variant
    within 100x of the band, so it is held to 2x.
    """
    x, weight, bias, stride, padding = _build(_spec(conv, 1, side, "nchw"), seed=33)
    ref = op.reference(x, weight, bias, stride, padding)
    wrong = {
        "flipped kernel": F.conv2d(x, weight.flip(-1, -2), bias, stride, padding),
        "transposed kernel": F.conv2d(x, weight.transpose(-1, -2), bias, stride, padding),
        "channels rolled": F.conv2d(x.roll(1, dims=1), weight, bias, stride, padding),
    }
    if bias is not None:
        wrong["bias dropped"] = F.conv2d(x, weight, None, stride, padding)
        unpadded = F.conv2d(x, weight, bias, stride, 0)
        wrong["padding 0, shifted"] = F.pad(unpadded, (0, 1, 0, 1))
    for name, value in wrong.items():
        element, rms = _ulps(value, ref)
        floor = 2.0 if name == "bias dropped" else 100.0
        assert element > floor * ELEMENT_ULP or rms > floor * RMS_ULP, (name, element, rms)
        with pytest.raises(AssertionError):
            _compare(value, ref)


def test_guard_refuses_a_partial_patch() -> None:
    """A 730 px tile patchifies to the same 52 x 52 grid as its top-left 728 px."""
    spec = _spec(PATCH_EMBED, 1, 730, "nchw")
    x, weight, bias, stride, padding = _build(spec, seed=44)
    out = op(x, weight, bias, stride, padding)
    assert out.shape == (1, 1536, 52, 52)
    cropped = x[..., :728, :728].contiguous()
    _compare(out, op.reference(cropped, weight, bias, stride, padding))
    with pytest.raises(AssertionError, match="needs whole patches"):
        with validating(op):
            op(x, weight, bias, stride, padding)
    with validating(op):
        op(cropped, weight, bias, stride, padding)
