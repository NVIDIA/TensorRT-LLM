# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the cuda_tile_rms_norm entry.

Weights are drawn uniformly over the range of the checkpoint tensors each
cell's call site normalizes with (all fp32 on disk), so a dropped `1 +` or a
weight read at the wrong precision is not masked by a near-zero weight:

- `input_ln0`: `model.layers.0.input_layernorm.weight`;
- `final_norm`: `model.norm.weight`;
- `q_norm` / `k_norm`: the 92 `self_attn.{q,k}_norm.weight` tensors;
- `idx_q_norm`: the 23 `self_attn.sparse_indexer_q_norm.weight` tensors,
  rounded to bf16 as the caller passes them.

Activations are standard normal with a per-row scale drawn log-uniformly over
2^-6..2^6, and the 4096-wide residual rows carry four outlier channels 64x
larger, so the reduction sees the dynamic range a residual stream has.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.cuda_tile_rms_norm import (  # noqa: E501
    cuda_tile_rms_norm as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "cuda_tile_rms_norm requires a CUDA device"
assert hasattr(torch.ops.trtllm, "cuda_tile_rms_norm"), (
    "trtllm::cuda_tile_rms_norm is not registered: cuda.tile or tileiras is missing"
)

DEVICE = torch.device("cuda")
EPS = 1e-5

#: (lo, hi) over each checkpoint weight family named in the module docstring.
WEIGHT_RANGES = {
    "input_ln0": (-0.449177, 0.19973),
    "final_norm": (-0.0253533, 0.856253),
    "q_norm": (-1.05068, 1.81067),
    "k_norm": (-1.02274, 1.73373),
    "idx_q_norm": (-0.952638, 0.209962),
}

#: The bands, in units of the spacing of the output dtype in the reference
#: value's own binade (see `_compare`).
BF16_ELEMENT_SPACINGS = 1.0
BF16_MISMATCH_FRACTION = 1e-3
FP32_ELEMENT_SPACINGS = 8.0
FP32_RMS_SPACINGS = 2.0

MANTISSA_BITS = {torch.bfloat16: 7, torch.float32: 23}


def _gen(seed: int) -> torch.Generator:
    return torch.Generator(device=DEVICE).manual_seed(seed)


def _build(spec, seed):
    gen = _gen(seed)
    rows, width, dtype = spec["rows"], spec["width"], spec["dtype"]
    x = torch.randn(rows, width, generator=gen, device=DEVICE)
    x = x * torch.exp2(torch.rand(rows, 1, generator=gen, device=DEVICE) * 12 - 6)
    if width == 4096:
        channels = torch.randperm(width, generator=gen, device=DEVICE)[:4]
        x[:, channels] *= 64.0
    lo, hi = WEIGHT_RANGES[spec["family"]]
    weight = torch.rand(width, generator=gen, device=DEVICE) * (hi - lo) + lo
    weight = weight.to(spec.get("weight_dtype", torch.float32))
    return x.to(dtype), weight


def _args(x, weight):
    return dict(x=x, weight=weight, eps=EPS, static_persistent=True, gather=True, use_gemma=True)


def _spacing(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The spacing of `dtype` in the binade of each value (2^-126 floor)."""
    _, exponent = torch.frexp(values.float().abs().clamp_min(2.0**-126))
    return torch.exp2((exponent - 1 - MANTISSA_BITS[dtype]).double())


def _compare(out: torch.Tensor, ref: torch.Tensor) -> dict:
    """Gate the op against the entry's reference, rounded once to the output dtype.

    The kernel's fp32 value carries the roundings of the products, the rsqrt
    and the fp32 sum of squares; measured on the fp32 cells, it sits within 4
    fp32 spacings of the once-rounded reference (relative RMS <= 0.6). So:

    - bf16 out: rounding that fp32 value to bf16 lands on one of the two bf16
      neighbours of the exact value, the correctly rounded one except where the
      exact value lies within a few fp32 spacings of a rounding midpoint. Gate:
      every element within one bf16 spacing (of the larger magnitude's binade)
      of the correctly rounded reference, and at most 1e-3 of elements off it
      (a smooth input puts ~2^-12 of values that close to a midpoint; a wrong
      formula moves half of them).
    - fp32 out: the reference is the fp64 value rounded once (<= 0.5 spacing),
      so the two differ by <= ~4.1 spacings; gate 8 spacings per element and a
      relative RMS of 2 spacings, both in fp32 spacings of the reference's own
      binade. torch's fp32 default (rtol 1.3e-6, ~11 spacings) is looser.
    """
    assert out.dtype == ref.dtype and out.shape == ref.shape
    o, r = out.double(), ref.double()
    diff = (o - r).abs()
    spacing = _spacing(torch.maximum(o.abs(), r.abs()), out.dtype)
    max_spacings = (diff / spacing).max().item()
    mismatch = (o != r).double().mean().item()
    rel_rms = ((diff.square().mean().sqrt()) / r.square().mean().sqrt().clamp_min(1e-30)).item()
    rms_spacings = rel_rms / 2.0 ** -MANTISSA_BITS[out.dtype]
    stats = dict(max_spacings=max_spacings, mismatch=mismatch, rms_spacings=rms_spacings)
    print(f"stats {out.dtype} {tuple(out.shape)}: {stats}")
    if out.dtype == torch.bfloat16:
        assert max_spacings <= BF16_ELEMENT_SPACINGS, stats
        assert mismatch <= BF16_MISMATCH_FRACTION, stats
    else:
        assert out.dtype == torch.float32
        assert max_spacings <= FP32_ELEMENT_SPACINGS, stats
        assert rms_spacings <= FP32_RMS_SPACINGS, stats
    return stats


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    x, weight = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    before = (x.clone(), weight.clone())
    out = op(**_args(x, weight))
    assert out.shape == x.shape and out.dtype == x.dtype
    assert out.is_contiguous() and out.data_ptr() != x.data_ptr()
    assert torch.equal(before[0], x) and torch.equal(before[1], weight), (
        "inputs must not be mutated"
    )
    assert torch.equal(out, op(**_args(x, weight))), "two launches must agree bitwise"
    _compare(out, op.reference(**_args(x, weight)))


def _cell(family, rows, dtype=torch.bfloat16):
    return next(
        c.spec
        for c in op.CELLS
        if c.spec["family"] == family and c.spec["rows"] == rows and c.spec["dtype"] == dtype
    )


def test_one_cutile_kernel_per_call():
    """The certified call is one launch of the cuTile persistent kernel, not some other norm."""
    from torch.profiler import ProfilerActivity, profile

    for spec in (_cell("input_ln0", 128, torch.float32), _cell("q_norm", 8192)):
        x, weight = _build(spec, seed=3)
        op(**_args(x, weight))  # compile outside the profiled region
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            op(**_args(x, weight))
            torch.cuda.synchronize()
        kernels = [e.name for e in prof.events() if e.device_type.name == "CUDA"]
        assert len(kernels) == 1, kernels
        assert kernels[0].startswith("rms_norm_kernel_static_persistent"), kernels


def test_cuda_graph_capture_and_replay():
    """A decode graph captures the norm: replay on refilled inputs equals eager bitwise."""
    for spec in (_cell("input_ln0", 128, torch.float32), _cell("q_norm", 8192)):
        x, weight = _build(spec, seed=7)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            op(**_args(x, weight))  # warm-up compiles the kernel outside capture
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = op(**_args(x, weight))
        x2, weight2 = _build(spec, seed=8)
        x.copy_(x2)
        weight.copy_(weight2)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, op(**_args(x, weight)))
        _compare(out, op.reference(**_args(x, weight)))


def test_band_rejects_the_wrong_scale_forms():
    """The band tells the zero-centred fp32 scale from its two plausible mistakes.

    Plain `w` instead of `1 + w`, and the fp32 q-norm weight rounded to bf16
    before the call (its `1 + w` then loses up to 2^-9 |w|, catastrophically
    near w = -1). Both run through the op itself and must fall outside.
    """
    spec = _cell("q_norm", 8192)
    x, weight = _build(spec, seed=11)
    ref = op.reference(**_args(x, weight))
    plain = torch.ops.trtllm.cuda_tile_rms_norm(x, weight, EPS, True, True, False)
    narrowed = torch.ops.trtllm.cuda_tile_rms_norm(x, weight.bfloat16(), EPS, True, True, True)
    for name, wrong in (("plain w", plain), ("bf16 weight", narrowed)):
        with pytest.raises(AssertionError) as excinfo:
            _compare(wrong, ref)
        print(f"{name}: outside the band -- {excinfo.value}")


def test_guard_refuses_a_weight_of_the_wrong_length():
    """The op accepts any weight length; the tiled kernel then answers wrongly.

    A 193-long weight whose first 192 entries are the intended ones: the
    non-persistent tiled kernel returns a full-shape result that is not the
    norm with those 192 weights, and no mode raises.
    """
    gen = _gen(23)
    x = torch.randn(64, 192, generator=gen, device=DEVICE).bfloat16()
    weight = (torch.rand(193, generator=gen, device=DEVICE) * 2 - 1).float()
    ref = op.reference(x, weight[:192], EPS, False, False, True)
    unguarded = torch.ops.trtllm.cuda_tile_rms_norm(x, weight, EPS, False, False, True)
    assert unguarded.shape == x.shape
    with pytest.raises(AssertionError) as excinfo:
        _compare(unguarded, ref)
    print(f"193-long weight, tiled kernel: outside the band -- {excinfo.value}")
    for static_persistent, gather in ((True, True), (False, True)):
        out = torch.ops.trtllm.cuda_tile_rms_norm(
            x, weight[:191].contiguous(), EPS, static_persistent, gather, True
        )
        assert out.shape == x.shape, "a short weight is accepted without error"

    for bad in (weight, weight[:191].contiguous(), weight[:192].reshape(1, 192)):
        with pytest.raises(AssertionError, match="as long as the normalized width"):
            with validating(op):
                op(x, bad, EPS, True, True, True)


def test_zero_rows_fail_loudly():
    """An empty batch is rejected at launch, not answered: the caller skips the call."""
    weight = torch.zeros(4096, device=DEVICE)
    with pytest.raises(RuntimeError, match="Failed to launch cuTile kernel"):
        op(**_args(torch.empty(0, 4096, device=DEVICE), weight))
