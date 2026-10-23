# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp patch_embed_conv3d entry.

The reference is a true-fp32 matmul of the flattened patch against the
flattened filter: a different op, accumulation order and memory layout from
the convolution under test, so it checks the contraction pairing rather than
restating it.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
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


def _build(spec, seed):
    pixel_values, weight, bias = _operands(
        spec["num_patches"],
        dtype=spec["dtype"],
        seed=seed,
        in_channels=spec.get("in_channels", IN_CHANNELS),
        kernel=spec.get("kernel", (KT, KH, KW)),
        embed_dim=spec.get("embed_dim", EMBED_DIM),
        bias=spec.get("bias", True),
    )
    layout = spec.get("layout", "flat")
    n, row = pixel_values.shape
    if layout == "five_d":
        pixel_values = pixel_values.view(-1, *weight.shape[1:])
    elif layout == "column_slice":
        wide = torch.empty(n, 2 * row, device="cuda", dtype=pixel_values.dtype)
        wide[:, :row] = pixel_values
        pixel_values = wide[:, :row]
        assert not pixel_values.is_contiguous() and pixel_values.stride(-1) == 1
    elif layout == "transposed":
        pixel_values = pixel_values.t().contiguous().t()
        assert pixel_values.stride(-1) != 1
    return pixel_values, weight, bias


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    pixel_values, weight, bias = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    before = pixel_values.clone()
    previous = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        out = op(pixel_values, weight, bias)
    finally:
        torch.backends.cudnn.allow_tf32 = previous
    assert out.shape == (pixel_values.shape[0], weight.shape[0])
    assert out.dtype == pixel_values.dtype and out.is_contiguous()
    assert out.data_ptr() != pixel_values.data_ptr()
    assert torch.equal(pixel_values, before), "the pixels are read-only"
    if not pixel_values.is_contiguous() and pixel_values.stride(-1) == 1:
        assert torch.equal(out, op(pixel_values.contiguous(), weight, bias))
    _compare(out, op.reference(pixel_values, weight, bias))


def test_five_dimensional_input_is_the_same_call() -> None:
    pixel_values, weight, bias = _operands(64, seed=1)
    flat = op(pixel_values, weight, bias)
    explicit = op(pixel_values.view(-1, IN_CHANNELS, KT, KH, KW), weight, bias)
    assert torch.equal(flat, explicit)


def test_zero_bias_is_the_identity() -> None:
    pixel_values, weight, _ = _operands(128, seed=2, bias=False)
    zero = torch.zeros(EMBED_DIM, dtype=weight.dtype, device="cuda")
    assert torch.equal(op(pixel_values, weight, None), op(pixel_values, weight, zero))


def test_rows_are_independent() -> None:
    """Same shape, so cuDNN selects the same algorithm: asserted bitwise."""
    pixel_values, weight, bias = _operands(128, seed=5)
    order = torch.randperm(128, generator=torch.Generator().manual_seed(0)).to("cuda")
    out = op(pixel_values, weight, bias)
    assert torch.equal(op(pixel_values[order].contiguous(), weight, bias), out[order])


def test_wrong_variants_land_outside_the_gate() -> None:
    """Three structural mis-derivations land hundreds of times outside
    `_compare`'s atol; dropping the bias is bounded by `max|bias|` (~0.02 at the
    tower's initializer scale) and is asserted to *be* the bias."""
    pixel_values, weight, bias = _operands(64, seed=6)
    out = op(pixel_values, weight, bias).float()
    ref = op.reference(pixel_values, weight, bias)
    rms = ref.float().pow(2).mean().sqrt().item()
    atol = max(1e-5, 4.0 * rms * 2.0**-8)
    correct_dev = (out - ref.float()).abs().max().item()
    rows = pixel_values.float()

    def _matmul(w):
        with true_fp32_matmul():
            return rows @ w.reshape(w.shape[0], -1).float().t() + bias.float()

    structural = {
        "row read as (T, C, Ph, Pw)": _matmul(weight.permute(0, 2, 1, 3, 4)),
        "kernel flipped (true convolution)": _matmul(weight.flip(2, 3, 4)),
        "patch rows and columns transposed": _matmul(weight.transpose(-1, -2)),
    }
    print(f"\n  gate atol {atol:.5f}; correct deviation {correct_dev:.5f}")
    for name, wrong in structural.items():
        dev = (out - wrong).abs().max().item()
        print(f"  {name:38s} dev {dev:9.5f}  ({dev / atol:8.1f}x the gate)")
        assert dev > 100.0 * atol, (name, dev, atol)
        with pytest.raises(AssertionError):
            _compare(out.to(ref.dtype), wrong.to(ref.dtype))

    no_bias = (out - (ref.float() - bias.float())).abs().max().item()
    print(f"  {'bias dropped':38s} dev {no_bias:9.5f}  ({no_bias / atol:8.1f}x the gate)")
    assert no_bias > 5.0 * atol, (no_bias, atol)
    assert abs(no_bias - bias.abs().max().item()) <= correct_dev


def test_guard_refuses_an_image_volume() -> None:
    """The hazard, unguarded: a [1, C, D, H, W] volume with the element count of
    16 patches returns a [16, 1152] result of the right shape and dtype computed
    from the wrong pixels."""
    g = torch.Generator(device="cuda").manual_seed(7)
    volume = torch.empty(1, IN_CHANNELS, KT, 4 * KH, 4 * KW, device="cuda")
    volume = volume.uniform_(-1.0, 1.0, generator=g).to(torch.bfloat16)
    _, weight, bias = _operands(1, seed=7)

    reinterpreted = op(volume, weight, bias)
    meant = (
        torch.nn.functional.conv3d(volume, weight, bias, stride=(KT, KH, KW))
        .flatten(2)
        .squeeze(0)
        .t()
    )
    assert reinterpreted.shape == meant.shape == (16, EMBED_DIM)
    dev = (reinterpreted.float() - meant.float()).abs().max().item()
    print(f"\n  unguarded volume reinterpretation deviates by {dev:.5f}")
    assert dev > 0.1, dev

    with pytest.raises(AssertionError, match="each row must be exactly one patch"):
        with validating(op):
            op(volume, weight, bias)


def test_guard_refuses_rows_that_are_not_one_patch() -> None:
    """Two patches per row are split into twice the rows, without an error."""
    pixel_values, weight, bias = _operands(8, seed=8)
    doubled = pixel_values.reshape(4, 2 * PATCH_NUMEL)
    assert op(doubled, weight, bias).shape == (8, EMBED_DIM)
    for bad in (doubled, pixel_values[:, :-1].contiguous(), pixel_values.reshape(-1)):
        with pytest.raises(AssertionError, match="each row must be exactly one patch"):
            with validating(op):
                op(bad, weight, bias)


def test_dtype_and_device_mismatches_raise() -> None:
    """Loud from torch itself, so the entry does not guard them."""
    pixel_values, weight, bias = _operands(8, seed=10)
    with pytest.raises(RuntimeError, match="should be the same"):
        op(pixel_values, weight.float(), bias.float())
    with pytest.raises(RuntimeError, match="should be the same"):
        op(pixel_values, weight, bias.float())
    with pytest.raises(RuntimeError, match="should be the same"):
        op(pixel_values, weight.cpu(), bias.cpu())


def test_fp32_accumulates_in_tf32_unless_cudnn_is_told_otherwise() -> None:
    pixel_values, weight, bias = _operands(256, dtype=torch.float32, seed=11)
    ref = op.reference(pixel_values, weight, bias)
    previous = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = True
        tf32_dev = (op(pixel_values, weight, bias) - ref).abs().max().item()
        torch.backends.cudnn.allow_tf32 = False
        exact = op(pixel_values, weight, bias)
    finally:
        torch.backends.cudnn.allow_tf32 = previous
    exact_dev = (exact - ref).abs().max().item()
    print(f"\n  fp32 conv, cudnn TF32 on {tf32_dev:.3e}; off {exact_dev:.3e}")
    _compare(exact, ref)
    assert tf32_dev > 20.0 * exact_dev, (tf32_dev, exact_dev)


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
