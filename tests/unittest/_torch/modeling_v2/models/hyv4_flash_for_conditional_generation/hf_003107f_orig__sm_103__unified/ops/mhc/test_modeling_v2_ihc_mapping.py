# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ihc_mapping entry this target owns.

The entry is a macro composition, so its `reference` is the same formula in
float64: the cells prove the fp32 composition is that formula to within its own
rounding, and the tests after them cover what a mirror cannot -- that the
projection is IEEE fp32 and not this device's TF32 default, that a plausible
wrong variant (the pre-GEMM norm left out) is far outside the band, that the
magnitude moves the post half only, that `out` is built from the pre half only,
that every wrong stream layout is loud (the entry carries no `is_valid`), and
that the decode path captures and replays in a CUDA graph.
"""

import zlib

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.ops.mhc.ihc_mapping import (  # noqa: E501
    ihc_mapping as op,
)

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    ULP,
    assert_within_ulp,
    certifying,
)

assert torch.cuda.is_available(), "ihc_mapping requires a CUDA device"

CELLS = op.CELLS

#: `out` is bf16, rounded once from fp32; the entry and the float64 reference
#: differ only by the fp32 path's own rounding, so an element moves by at most
#: one bf16 rounding step of its row's largest value. Measured worst across the
#: cells on sm_103: 1.652 / 0.01125 ulp (connection, 5120 rows). Narrower than
#: torch's bf16 default (rtol 1.6e-2 + atol 1e-5, ~4.1 ulp of these rows).
_OUT_ELEMENT_ULP = 2.5
_OUT_RMS_ULP = 0.05

#: The gates are fp32 and are the sensitive half: the projection accumulates
#: 12288 products in fp32 and the reference in float64, so the difference is
#: the accumulation's own error, read against the half's row scale. Measured
#: worst across the cells on sm_103: 149.38 / 5.619 ulp (connection, 5120 rows,
#: pre half, the model's widest hc_fn at rms 0.0493); the other halves sit at
#: 0.71..12.10 / 0.54..2.51 ulp. A row whose four pre gates are all small makes
#: the half's row scale small while the absolute error still tracks the logit,
#: which is why the pre half is the worst one. This band is wider than torch's
#: fp32 default (~11 + 42 ulp here), so
#: `test_the_band_rejects_a_wrong_formula` measures the separation from a
#: plausible wrong variant.
_GATE_ELEMENT_ULP = 320.0
_GATE_RMS_ULP = 16.0

# The checkpoint's own element statistics, read from its shards: hc_fn rms
# 0.0133 at layer-0 attn (0.0130 for hc_head_fn, 0.0493 the widest of the 85),
# hc_base at layer-0 attn rms 1.79 mean -1.60, the head's rms 1.11 mean -1.10,
# hc_scale at layer-0 attn [1.520, 0.015], hc_head_scale 0.158.
CKPT_BASE_CONNECTION = (-1.60, 0.80)
CKPT_BASE_HEAD = (-1.10, 0.15)
CKPT_SCALE_CONNECTION = (1.52012, 0.0148)
CKPT_SCALE_HEAD = 0.158408


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows, streams, hidden, groups = (
        spec["rows"],
        spec["streams"],
        spec["hidden"],
        spec["groups"],
    )
    gate_rows = groups * streams
    x = torch.randn(rows, streams, hidden, generator=g, device="cuda") * spec["stream_sigma"]
    x = x.to(spec["dtype"])
    fn = torch.randn(gate_rows, streams * hidden, generator=g, device="cuda") * spec["fn_rms"]
    if groups == 2:
        mean, std = CKPT_BASE_CONNECTION
        scale = CKPT_SCALE_CONNECTION
    else:
        mean, std = CKPT_BASE_HEAD
        scale = (CKPT_SCALE_HEAD,)
    base = torch.randn(gate_rows, generator=g, device="cuda") * std + mean
    return (
        x,
        fn.to(torch.float32),
        base.to(torch.float32),
        torch.tensor(scale, dtype=torch.float32, device="cuda"),
        spec["post_magnitude"],
        spec["eps"],
        spec["norm_eps"],
    )


def _measure(got: torch.Tensor, want: torch.Tensor) -> tuple[float, float]:
    """The same two quantities `assert_within_ulp` gates, for the message."""
    ulp = ULP[want.dtype]
    a = got.float().reshape(-1, got.shape[-1])
    b = want.float().reshape(-1, want.shape[-1])
    row_scale = b.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    element = ((a - b).abs() / (row_scale * ulp)).max().item()
    rms = ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-9)).item() / ulp
    return element, rms


def _cosine(got: torch.Tensor, want: torch.Tensor) -> float:
    # In float64: two fp32 tensors that agree to 1e-7 have a cosine of
    # 1 - 1e-14, but an fp32 dot product cannot resolve closer to 1 than its own
    # epsilon, and would report 1 - 1.2e-7 for them.
    a, b = got.double().reshape(-1), want.double().reshape(-1)
    return (a @ b / (a.norm() * b.norm())).item()


def _halves(gates: torch.Tensor, ref_gates: torch.Tensor, streams: int):
    """The pre and post halves apart: each against its own row scale, so a
    nearly-constant post half is not measured against the pre half's spread."""
    if gates.shape[-1] == streams:
        return [(gates, ref_gates)]
    return [
        (gates[..., :streams], ref_gates[..., :streams]),
        (gates[..., streams:], ref_gates[..., streams:]),
    ]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    args = _build(spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    before = args[0].clone()

    out, gates = op(*args)
    streams = spec["streams"]
    assert out.shape == (spec["rows"], spec["hidden"]), out.shape
    assert out.dtype == spec["dtype"]
    assert gates.shape == (spec["rows"], spec["groups"] * streams), gates.shape
    assert gates.dtype is torch.float32
    assert out.data_ptr() != args[0].data_ptr(), "the op must return a new tensor"
    assert torch.equal(args[0], before), "the streams are read-only"

    ref_out, ref_gates = op.reference(*args)
    cos = _cosine(out, ref_out)
    assert cos > 0.999999, f"cosine {cos:.9f}: out diverged from the reference"
    assert_within_ulp(out, ref_out, element_ulp=_OUT_ELEMENT_ULP, rms_ulp=_OUT_RMS_ULP)

    for got, want in _halves(gates, ref_gates, streams):
        cos = _cosine(got, want)
        assert cos > 0.9999999, f"cosine {cos:.9f}: gates diverged from the reference"
        assert_within_ulp(got, want, element_ulp=_GATE_ELEMENT_ULP, rms_ulp=_GATE_RMS_ULP)


def _linear_under(precision: str, flat: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """`F.linear` with this process's fp32 matmul policy set to `precision`."""
    allow = torch.backends.cuda.matmul.allow_tf32
    prev = torch.backends.cuda.matmul.fp32_precision
    try:
        torch.backends.cuda.matmul.allow_tf32 = precision == "tf32"
        torch.backends.cuda.matmul.fp32_precision = precision
        return F.linear(flat, weight)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow
        torch.backends.cuda.matmul.fp32_precision = prev


def test_the_fp32_wrapper_pins_the_projection() -> None:
    """What the wrapper buys at this op's shape, and that the switch is live.

    This device's fp32 matmul policy defaults to TF32. The entry runs its
    projection under `true_fp32_matmul()`, so the result is the same number
    whatever the ambient policy is. The control shows the policy really does
    change a product on this device (a wide enough `N` for cuBLAS to pick a
    TF32 GEMM); the op's own `N = 8` is then measured against the gate band.
    """
    spec = CELLS[2].spec  # 5120 rows: a GEMM, not the decode gemv
    args = _build(spec, seed=5)
    flat = args[0].flatten(-2).float()
    with true_fp32_matmul():
        ieee = F.linear(flat, args[1])
    element, rms = _measure(_linear_under("tf32", flat, args[1]), ieee)

    wide = torch.randn(64, flat.shape[-1], device="cuda") * 0.02
    with true_fp32_matmul():
        wide_ieee = F.linear(flat, wide)
    wide_element, _ = _measure(_linear_under("tf32", flat, wide), wide_ieee)
    assert wide_element > _GATE_ELEMENT_ULP, (
        f"the TF32 switch moved even a wide product by only {wide_element:.1f} ulp: "
        f"this process cannot be put into TF32 at all, so the wrapper is untested"
    )
    # Measured on sm_103: 9776.3 / 1741.1 ulp at the op's own shape, 4088.2 ulp
    # for the control. At one row the product is a gemv, where TF32 does not
    # apply (1.7 ulp); the cells that matter for the band are the 5120-row ones.
    assert element > 10.0 * _GATE_ELEMENT_ULP, (
        f"TF32 moved the op's own projection by only {element:.1f} / {rms:.1f} ulp "
        f"against the {_GATE_ELEMENT_ULP} / {_GATE_RMS_ULP} gate band: the wrapper "
        f"is not load-bearing at this shape"
    )


def _wrong_variant(streams, fn, base, scale, magnitude, eps, norm_eps):
    """The plausible mistake this op's own math invites: the pre-GEMM RMS norm
    left out, so the projection sees the streams' raw scale instead of a unit
    RMS one.

    Mirrors `__call__`'s signature so it takes the same argument list;
    `norm_eps` is what it fails to use.
    """
    n = streams.shape[-2]
    flat = streams.flatten(-2).float()
    with true_fp32_matmul():
        logits = F.linear(flat, fn)
    pre = torch.sigmoid(logits[..., :n] * scale[0] + base[:n]) + eps
    post = magnitude * torch.sigmoid(logits[..., n:] * scale[1] + base[n:]) + eps
    gates = torch.cat((pre, post), dim=-1)
    out = (gates[..., :n].unsqueeze(-1) * streams).sum(dim=-2)
    return out.to(streams.dtype), gates


def test_the_band_rejects_a_wrong_formula() -> None:
    """The gate band is wider than torch's fp32 default, so measure what it
    still rejects.

    The widest cell (5120 rows, the model's widest hc_fn) run with the
    pre-GEMM norm left out: the streams' 8x scale reaches the sigmoid, so the
    gates are a different distribution, not a rounding of the same one.
    """
    spec = CELLS[2].spec
    args = _build(spec, seed=9)
    ref_out, ref_gates = op.reference(*args)
    bad_out, bad_gates = _wrong_variant(*args)
    n = spec["streams"]
    element, rms = _measure(bad_gates[..., :n], ref_gates[..., :n])
    out_element, _ = _measure(bad_out, ref_out)
    assert element > 100.0 * _GATE_ELEMENT_ULP, (
        f"the wrong variant sits {element:.1f} ulp from the reference, inside 100x "
        f"the {_GATE_ELEMENT_ULP} ulp band: the band would not tell them apart"
    )
    assert rms > 100.0 * _GATE_RMS_ULP, f"the wrong variant's relative RMS is only {rms:.1f} ulp"
    assert out_element > 10.0 * _OUT_ELEMENT_ULP
    # The cosine is not the discriminator for this variant: the post half is a
    # near-constant 2.0 in both and dominates the vector, so it reads 0.9506.
    # The band's own two metrics are what reject it.


def test_the_post_half_is_the_only_thing_the_magnitude_moves() -> None:
    """The magnitude multiplies the post half's sigmoid, before the eps.

    A magnitude of 0 leaves `post_magnitude * sigmoid(...) + eps` at exactly
    eps, so this pins both where the factor sits and that the pre half and
    `out` never see it.
    """
    spec = CELLS[0].spec
    args = _build(spec, seed=6)
    out, gates = op(*args)
    out0, gates0 = op(args[0], args[1], args[2], args[3], 0.0, args[5], args[6])
    n = spec["streams"]
    assert torch.equal(out0, out), "out must not depend on the post magnitude"
    assert torch.equal(gates0[..., :n], gates[..., :n]), "the pre half must not see it"
    assert torch.all(gates0[..., n:] == spec["eps"]), (
        f"post gates at magnitude 0 must be exactly eps; got "
        f"{gates0[..., n:].min().item():.9g}..{gates0[..., n:].max().item():.9g}"
    )
    assert torch.all(gates[..., n:] > spec["eps"]), "the post half must be gated"


def test_out_is_built_from_the_pre_half_only() -> None:
    """Moving the post half's base leaves `out` bitwise unchanged."""
    spec = CELLS[0].spec
    args = _build(spec, seed=7)
    out, gates = op(*args)
    moved = args[2].clone()
    moved[spec["streams"] :] += 3.0
    out2, gates2 = op(args[0], args[1], moved, *args[3:])
    n = spec["streams"]
    assert torch.equal(out2, out), "out must be the pre-weighted sum alone"
    assert torch.equal(gates2[..., :n], gates[..., :n]), "the pre half must not see it"
    assert not torch.equal(gates2[..., n:], gates[..., n:]), "the post half must see it"


def test_every_wrong_stream_shape_is_loud() -> None:
    """The entry carries no `is_valid`, and this is why: it never answers
    plausibly to a wrong stream layout.

    The two layouts a caller can actually get wrong both raise from the
    arithmetic -- the flattened `[rows, N*H]` form (2-D, so `flatten(-2)`
    collapses it to one row) in the projection, and a stream axis that is not
    `N` (here 3 streams of 4096, the same total width) in the gate-row branch
    check. A guard restating either would only swap its error for a worse one.
    """
    spec = dict(CELLS[0].spec, rows=4)
    args = _build(spec, seed=8)

    flat = args[0].flatten(-2)
    assert flat.shape == (spec["rows"], spec["streams"] * spec["hidden"])
    with pytest.raises(RuntimeError, match="shapes cannot be multiplied"):
        op(flat, *args[1:])

    mis_split = args[0].reshape(spec["rows"], 3, 4096)
    with pytest.raises(ValueError, match="fn has 8 rows for 3 streams"):
        op(mis_split, *args[1:])

    # And the shape it does accept is the one the cells drive.
    out, _ = op(*args)
    assert out.shape == (spec["rows"], spec["hidden"])


def test_cuda_graph_capture_and_replay() -> None:
    """The decode path runs under a captured graph in the engine.

    The fp32 wrapper's switch is host-side inside the captured region, so this
    also shows the capture still succeeds with it there.
    """
    cell = CELLS[0]
    args = _build(cell.spec, seed=11)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            op(*args)
    torch.cuda.current_stream().wait_stream(warmup)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out, graph_gates = op(*args)

    args[0].copy_(_build(cell.spec, seed=12)[0])
    graph.replay()
    torch.cuda.synchronize()
    first = (graph_out.clone(), graph_gates.clone())
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(graph_out, first[0]), "replay must be deterministic"
    assert torch.equal(graph_gates, first[1])

    eager_out, eager_gates = op(*args)
    assert_within_ulp(graph_out, eager_out, element_ulp=_OUT_ELEMENT_ULP, rms_ulp=_OUT_RMS_ULP)
    for got, want in _halves(graph_gates, eager_gates, cell.spec["streams"]):
        assert_within_ulp(got, want, element_ulp=_GATE_ELEMENT_ULP, rms_ulp=_GATE_RMS_ULP)
