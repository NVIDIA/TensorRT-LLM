# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_silu_and_mul catalog entry.

The entry owns what is certified. This file turns a cell's spec into tensors
and drives the inputs the guard exists to refuse.
"""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.flashinfer_silu_and_mul import (
    flashinfer_silu_and_mul as op,
)

__extra_import_path__ = [".."]
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "flashinfer_silu_and_mul requires a CUDA device"


def _build(spec, seed):
    """A cell's input; `data="vit"` is the vision-tower data of `_vit_input`, else plain randn."""
    if spec.get("data") == "vit":
        rows, width = spec["shape"]
        assert width == 2 * _VIT_D and spec["dtype"] is torch.bfloat16, spec
        return _vit_input(rows, seed)
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(spec["shape"], generator=g, device="cuda").to(spec["dtype"])


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:40] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    x = _build(cell.spec, seed=abs(hash(cell.why)) % 2**31)
    with validating(op):
        out = op(x)
    assert out.shape == (*x.shape[:-1], x.shape[-1] // 2)
    assert out.data_ptr() != x.data_ptr(), "the op must return a new tensor"
    op.compare(out, op.reference(x))
    if cell.spec.get("data") == "vit":
        # The vision-tower gate on top: 1 ulp everywhere but the fast-math flushes.
        _assert_flushes_only_in_planted_rows(_assert_vit_gates(out, x, cell.why[:40]))


@pytest.mark.parametrize(
    "width,match",
    [
        (4095, "must be even"),
        # 24 bf16 elements is 48 bytes per row, which the op's own row check
        # accepts; the half is 24 bytes, so the second half starts mid-vector.
        (24, "16-byte vectors"),
        # The only width that reaches the third assert: for the half to be a
        # whole number of 16-byte vectors *and* smaller than one, it has to be
        # zero. Worth a case anyway -- it is the difference between a guard
        # that is unreachable and one that is merely narrow.
        (0, "at least one 16-byte vector"),
    ],
)
def test_guard_refuses_widths_the_kernel_faults_on(width, match) -> None:
    """A misaligned half kills the CUDA context rather than raising.

    That is why these are `is_valid` and not left to the op: the failure is
    unrecoverable and its message names neither the op nor the shape.
    """
    x = torch.randn(4, width, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match=match):
        with validating(op):
            op(x)


# ─── The DeepSeek-V4.1 vision tower's MLP call: every row count 1..17037 ──────
#
# The multimodal target's ViT block runs `flashinfer_silu_and_mul(cublas_mm(h2, w1_t))`:
# a contiguous bf16 [P, 5632] input whose first 2816 columns are the gate and the last
# 2816 the up projection (the checkpoint's w1 is [gate; up]), no clamp, one call per
# tower call. The tower groups a request's images into calls of at most 17037 patches
# (<= 9216 per image, <= 16 images), so P, the row count, is anything in 1..17037. It
# runs eagerly (the tower is never CUDA-graph captured). Rows only size the grid: one
# CTA per row, `min(d / 8, 1024)` threads, the row offset computed in int64.

_VIT_D = 2816
_VIT_N_MAX = 17037
# Tower-call row counts measured on the prototype (one image: 1517, 8772, 9189, the
# per-image maximum 9216, the mid-size images 2340 / 2352 and a padded 2433; several
# images: 3 x 1560, 3774 + 1551, 8787 + 8250 = 17037, and a 6-image request split into
# 15657 + 3120), then values either side of the 352-thread CTA's and 128-row edges.
_VIT_ROWS = (1517, 2340, 2352, 2433, 3120, 4680, 5325, 8772, 8787, 9189, 9216, 15657, 17037)
_VIT_EDGE_ROWS = (1, 2, 3, 127, 128, 129, 351, 352, 353, 8191, 8192, 8193, 17036)


def _vit_input(n: int, seed: int) -> torch.Tensor:
    """bf16 [n, 2 * 2816]: N(0, 3) gates and ups, with edge rows planted at fixed places.

    Row 0 all zeros; row 1 gates of magnitude 40..300 (both signs), far past any
    SwiGLU clamp; row 2 ups of magnitude 1e3..3e4; row 3 gates of magnitude 1e-30..1e-20;
    the last row gates of -80..-1000 (silu underflows to -0). They sit inside every
    prefix the sweep takes (rows 0..3) or at its top (the last row).
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(n, 2 * _VIT_D, generator=g, device="cuda") * 3.0
    sign = torch.randint(0, 2, (_VIT_D,), generator=g, device="cuda").float() * 2 - 1

    def mag(lo: float, hi: float) -> torch.Tensor:
        return lo + (hi - lo) * torch.rand(_VIT_D, generator=g, device="cuda")

    if n > 0:
        x[0] = 0.0
    if n > 1:
        x[1, :_VIT_D] = sign * mag(40.0, 300.0)
    if n > 2:
        x[2, _VIT_D:] = sign * mag(1e3, 3e4)
    if n > 3:
        x[3, :_VIT_D] = sign * torch.exp(mag(-69.0, -46.0))  # 1e-30 .. 1e-20
    if n > 4:
        x[n - 1, :_VIT_D] = -mag(80.0, 1000.0)
    return x.to(torch.bfloat16)


def _ordered_bits(t: torch.Tensor) -> torch.Tensor:
    """bf16 bit patterns mapped to integers that are monotonic in value (-0 and +0 coincide)."""
    b = t.view(torch.int16).int()
    return torch.where(b < 0, -(b & 0x7FFF), b)


def _ulp_distance(out: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Per-element distance in bf16 ulps (int32)."""
    return (_ordered_bits(out) - _ordered_bits(ref)).abs()


def _ulp_stats(out: torch.Tensor, ref: torch.Tensor):
    """(max distance in bf16 ulps, fraction of elements bit-identical)."""
    ulps = int(_ulp_distance(out, ref).max().item())
    exact = (out.view(torch.int16) == ref.view(torch.int16)).double().mean().item()
    return ulps, exact


# The kernel is built with fast math. Two consequences, pinned over every finite bf16
# gate in test_vit_width_every_finite_bf16_gate: (a) the silu division `x / (1 + e^-x)`
# is the fast reciprocal division, which returns 0 once the denominator passes 2^126,
# i.e. for gates in (-88.72, -87.33] (bf16: -88.5, -88, -87.5), where the exact result
# is still nonzero (|ref| <= 8.8e-37 * |up|); from -88.72 down, e^-x overflows and the
# fp32 reference is 0 as well. (b) fp32 subnormals are flushed: where silu(gate) or the
# product is below 2^-126 in magnitude, the kernel returns 0.
_FTZ = 2.0**-126
_FDIV_ZERO = (-88.72, -87.33)


def _flush_set(x: torch.Tensor) -> torch.Tensor:
    """Elements whose kernel result may be 0 where the fp32 reference is a tiny nonzero."""
    xf = x.float()
    d = x.shape[-1] // 2
    g, u = xf[..., :d], xf[..., d:]
    s = F.silu(g)
    return (s.abs() < _FTZ) | ((s * u).abs() < _FTZ) | ((g > _FDIV_ZERO[0]) & (g <= _FDIV_ZERO[1]))


def _assert_vit_gates(out: torch.Tensor, x: torch.Tensor, where: str) -> torch.Tensor:
    """The certification gates at the ViT call: default bf16 assert_close against the
    fp32 reference (silu and product in fp32, one rounding); every element within 1 bf16
    ulp of it except the fast-math flushes above, which must be exactly zero; and >= 0.9999
    of the other elements bit-identical (__expf is the only other difference). Returns the
    flushed elements, so a caller can say where they are allowed to be."""
    ref = op.reference(x)
    assert out.shape == ref.shape and out.dtype == torch.bfloat16
    op.compare(out, ref)
    dist = _ulp_distance(out, ref)
    flushed = _flush_set(x) & (out == 0) & (dist > 1)
    ok = (dist <= 1) | flushed
    n_bad = int((~ok).sum().item())
    exact = (out.view(torch.int16) == ref.view(torch.int16))[~flushed].double().mean().item()
    worst = int(dist[~flushed].max().item()) if dist.numel() else 0
    print(
        f"[vit {where}] rows={x.shape[0]}: max {worst} ulp outside {int(flushed.sum().item())} "
        f"fast-math flushes to 0, bit-exact {exact:.6f}"
    )
    assert n_bad == 0 and exact >= 0.9999, (where, n_bad, worst, exact)
    return flushed


def _assert_flushes_only_in_planted_rows(flushed: torch.Tensor) -> None:
    """On _vit_input data the flush set is only reachable through the planted rows of
    large negative gates: row 1 (-300..-40) and the last row (-1000..-80)."""
    n = flushed.shape[0]
    rows = set(flushed.any(1).nonzero().flatten().tolist())
    assert rows <= {1, n - 1}, sorted(rows)[:8]


def test_v41_vit_width_n_max_against_the_reference() -> None:
    """The 17037-row call (the tower's largest) against the fp32 reference, edge rows included.

    No clamp is applied: row 1's gates reach 300 and its outputs are 300-scale, where a
    SwiGLU limit of 10 (the text model's shared-expert form) would cap them.
    """
    x = _vit_input(_VIT_N_MAX, seed=2816)
    out = op(x)
    _assert_flushes_only_in_planted_rows(_assert_vit_gates(out, x, "n_max"))
    assert out.is_contiguous() and out.data_ptr() != x.data_ptr()
    assert torch.equal(out[0], torch.zeros_like(out[0]))  # silu(0) * 0
    big = out[1].float().abs().max().item()
    assert big > 100.0, f"row 1 (gates 40..300) peaked at {big}: something clamped it"
    assert bool(torch.isfinite(out.float()).all())


def test_v41_vit_width_wrong_references_are_rejected() -> None:
    """Controls for the gates above: swapped [up; gate] halves and a SwiGLU clamp at 10 both fail them."""
    x = _vit_input(4096, seed=2817)
    out = op(x)
    xf = x.float()
    gate, up = xf[:, :_VIT_D], xf[:, _VIT_D:]
    swapped = (F.silu(up) * gate).to(torch.bfloat16)
    clamped = (F.silu(gate.clamp(max=10.0)) * up.clamp(-10.0, 10.0)).to(torch.bfloat16)
    for name, wrong in (("halves swapped", swapped), ("clamp 10", clamped)):
        ulps, exact = _ulp_stats(out, wrong)
        print(f"[control {name}] max {ulps} ulp, bit-exact {exact:.6f}")
        assert ulps > 1 and (exact < 0.9999 or ulps > 1000), (name, ulps, exact)
    # The clamp shows only where a value is past the limit: rows 1 and 2 carry those.
    ulps_rows12, _ = _ulp_stats(out[1:3], clamped[1:3])
    assert ulps_rows12 > 1000, ulps_rows12


def test_v41_vit_width_every_row_count_is_a_prefix_of_n_max() -> None:
    """Every P = 1..17037: the op on the first P rows returns exactly the
    first P rows of the 17037-row result, which is held to the reference above.

    Each output is NaN-poisoned before it is freed, so a row the kernel skipped cannot
    inherit correct bits from the previous (one row shorter) call's recycled block; the
    fraction of calls whose output landed on the previous call's block is reported. Then
    the same kernel (the op body's flashinfer call, with a caller-owned `out`) writes into
    a NaN buffer one row longer than P, ascending: row P must still be NaN after the call.
    """
    from flashinfer.activation import silu_and_mul as _flashinfer_silu_and_mul

    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    x = _vit_input(_VIT_N_MAX, seed=2818)
    full = op(x)
    _assert_vit_gates(full, x, "sweep n_max")
    full_bits = full.view(torch.int16)
    bad = torch.zeros((), dtype=torch.int64, device="cuda")
    reused, prev_ptr = 0, None
    for r in range(1, _VIT_N_MAX + 1):
        out = op(x[:r])
        assert out.shape == (r, _VIT_D)
        reused += int(out.data_ptr() == prev_ptr)
        bad += (out.view(torch.int16) != full_bits[:r]).sum()
        out.fill_(float("nan"))
        prev_ptr = out.data_ptr()
        del out
    assert bad.item() == 0, f"{bad.item()} elements differ from the 17037-row result's rows"
    print(
        f"[vit sweep] P=1..{_VIT_N_MAX}: every prefix bitwise; {reused} of {_VIT_N_MAX - 1} "
        f"outputs landed on the previous call's NaN-poisoned block"
    )

    buf = torch.full((_VIT_N_MAX + 1, _VIT_D), float("nan"), dtype=torch.bfloat16, device="cuda")
    tail_written = torch.zeros((), dtype=torch.int64, device="cuda")
    pdl = get_env_enable_pdl()
    for r in range(1, _VIT_N_MAX + 1):
        _flashinfer_silu_and_mul(x[:r], out=buf[:r], enable_pdl=pdl)
        tail_written += (~torch.isnan(buf[r])).sum()
    assert tail_written.item() == 0, f"{tail_written.item()} elements written past the last row"
    assert torch.equal(buf[:_VIT_N_MAX].view(torch.int16), full_bits)


def test_v41_vit_width_row_sweep_harness_sees_a_one_row_shift() -> None:
    """Control for the sweep: the same bitwise check rejects a read one row off at every r tried."""
    x = _vit_input(_VIT_N_MAX, seed=2818)
    full = op(x)
    for r in (1, 2, 4, 351, 352, 353, 8192, _VIT_N_MAX - 1):
        shifted = op(x[1 : r + 1])
        n_bad = int((shifted.view(torch.int16) != full.view(torch.int16)[:r]).sum().item())
        print(f"[control row shift] r={r}: {n_bad} / {r * _VIT_D} elements differ")
        assert n_bad > r * _VIT_D // 2


def test_v41_vit_width_one_kernel_and_launch_shape_over_the_rows() -> None:
    """One kernel symbol at every probed P, block fixed at 352 threads, grid (P, 1, 1).

    Read from the profiler's own trace of each launch; the flashinfer module is JIT-built
    once per activation, never per shape.
    """
    import json
    import os
    import tempfile

    from torch.profiler import ProfilerActivity, profile

    rows = sorted(set(_VIT_ROWS) | set(_VIT_EDGE_ROWS) | set(range(1, 10)))
    x = _vit_input(_VIT_N_MAX, seed=2819)
    op(x[:1])  # build / load the module outside the trace
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for r in rows:
            op(x[:r])
        torch.cuda.synchronize()
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        prof.export_chrome_trace(path)
        with open(path) as f:
            trace = json.load(f)
    finally:
        os.remove(path)
    kernels = [e for e in trace["traceEvents"] if e.get("cat") == "kernel"]
    assert len(kernels) == len(rows), (
        len(kernels),
        len(rows),
        sorted({k["name"] for k in kernels}),
    )
    names = {k["name"] for k in kernels}
    assert len(names) == 1 and "act_and_mul_kernel" in next(iter(names)), names
    grids = [tuple(k["args"]["grid"]) for k in sorted(kernels, key=lambda k: k["ts"])]
    blocks = {tuple(k["args"]["block"]) for k in kernels}
    assert blocks == {(352, 1, 1)}, blocks
    assert grids == [(r, 1, 1) for r in rows], list(zip(rows, grids))[:8]
    print(
        f"[vit probe] {len(rows)} row counts 1..{max(rows)}: one kernel {next(iter(names))[:80]}, "
        f"block (352,1,1), grid (P,1,1)"
    )


@pytest.mark.parametrize("rows", _VIT_ROWS + _VIT_EDGE_ROWS)
def test_v41_vit_width_fresh_inputs_at_the_tower_row_counts(rows: int) -> None:
    """Each measured tower row count on its own fresh input, against the reference."""
    x = _vit_input(rows, seed=3000 + rows)
    _assert_flushes_only_in_planted_rows(_assert_vit_gates(op(x), x, f"fresh P={rows}"))


def test_v41_vit_width_every_finite_bf16_gate() -> None:
    """Every finite bf16 gate value against ups 1, 1.5, -3, 1e4 and 1e-4: within 1 ulp of
    the fp32 reference everywhere except the pinned fast-math flush set (_flush_set), where
    the result is 0 or within 1 ulp; the flush set's reference magnitudes stay below
    1e-36 * max(1, |up|), and it is exactly where the kernel disagrees with the reference."""
    bits = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)
    g = bits.view(torch.bfloat16)
    g = g[torch.isfinite(g.float())].cuda()
    n = g.numel()
    rows = (n + _VIT_D - 1) // _VIT_D
    gate = torch.zeros(rows * _VIT_D, dtype=torch.bfloat16, device="cuda")
    gate[:n] = g
    for up in (1.0, 1.5, -3.0, 1e4, 1e-4):
        x = torch.empty(rows, 2 * _VIT_D, dtype=torch.bfloat16, device="cuda")
        x[:, :_VIT_D] = gate.view(rows, _VIT_D)
        x[:, _VIT_D:] = up
        out = op(x)
        ref = op.reference(x)
        dist = _ulp_distance(out, ref).flatten()[:n]
        fl = _flush_set(x).flatten()[:n]
        o, r = out.flatten()[:n], ref.flatten()[:n]
        wrong = dist > 1
        zero_in_fl = int((fl & (o == 0) & (r != 0)).sum().item())
        print(
            f"[vit gates] up={up:g}: {n} finite gates; {int(wrong.sum().item())} > 1 ulp, all in the flush set "
            f"({int(fl.sum().item())} elements, {zero_in_fl} returned 0 over a nonzero reference); "
            f"max {int(dist[~fl].max().item())} ulp outside it"
        )
        assert not bool((wrong & ~fl).any()), g[(wrong & ~fl)][:8].tolist()
        assert bool(((o[fl] == 0) | (dist[fl] <= 1)).all())
        assert r[fl].float().abs().max().item() <= 1e-36 * max(1.0, abs(up))
        # The division flush: gates -88.5 / -88 / -87.5 return 0 where the reference is a
        # nonzero normal number (at |up| >= 1; at 1e-4 the reference itself underflows).
        for v in (-88.5, -88.0, -87.5) if abs(up) >= 1 else ():
            k = int((g.float() == v).nonzero()[0])
            assert o[k].item() == 0 and r[k].item() != 0, (up, v, o[k].item(), r[k].item())
