# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_rmsnorm catalog entry.

The entry owns what is certified: `CELLS` is the list, `reference` is what the
kernel is gated against, `is_valid` is the guard. This file owns only what a
cell cannot carry -- turning a spec into real tensors, and the band -- plus
the inputs the guard exists to refuse and the layouts the op refuses itself.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (  # noqa: E501
    flashinfer_rmsnorm as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "flashinfer_rmsnorm requires a CUDA device"

# Row r of every input is scaled by _ROW_SCALES[r % 4], so its mean square is
# about 1e-6, 1, 1e-2 or 1e-4. Unit-variance rows alone cannot see eps: at a
# mean square of 1, eps moves the result by a few millionths, far below one
# bf16 spacing, and a kernel that used 1e-6 or 0 for 1e-5 lands within one ulp
# of the reference everywhere. At a mean square of 1e-6 eps decides the
# answer; a wrong eps there is 169-247 ulp off. Row 0 is the small one, so a
# one-row cell sees eps too.
_ROW_SCALES = (1e-3, 1.0, 1e-1, 1e-2)


def _rows(rows, cols, dtype, g):
    """`rows` x `cols` of random data, row r scaled by `_ROW_SCALES[r % 4]`."""
    scale = torch.tensor(_ROW_SCALES, device="cuda").repeat((rows + 3) // 4)[:rows, None]
    return (torch.randn(rows, cols, generator=g, device="cuda") * scale).to(dtype)


def _build(spec, seed, rows=None):
    """One cell's operands at `rows` rows (the cell's own by default).

    With `row_stride`, `x` is the leading `width` columns of a wider row, as
    kv_a_layernorm reads the latent out of the kv_a output. The whole buffer
    is returned beside it, so a caller can check the columns outside `x` and
    refill the input under a CUDA graph.
    """
    rows = spec["shape"][0] if rows is None else rows
    width, dtype = spec["shape"][-1], spec["dtype"]
    g = torch.Generator(device="cuda").manual_seed(seed)
    buf = _rows(rows, spec.get("row_stride", width), dtype, g)
    weight = torch.randn(width, generator=g, device="cuda").to(dtype)
    return buf, buf[:, :width], weight


def _ordered(t):
    """bf16 bit patterns as integers that are consecutive for consecutive values."""
    bits = t.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(bits < 0, -(bits & 0x7FFF), bits)


def _compare(out, ref):
    """At most one bf16 ulp from the reference anywhere, and bitwise almost everywhere.

    Kernel and reference evaluate the same expression, `(x * rstd) * w` in
    fp32 rounded once to bf16, and differ only in `rstd`: the order the fp32
    sum of squares is taken in, and the kernel's approximate rsqrt. That moves
    the fp32 value by about a millionth, far below one bf16 spacing (2^-8), so
    the two round to the same bf16 or to neighbours. Each output is a product,
    with no cancellation, so the bound holds element by element at every
    magnitude; the gate counts it on the bit patterns, which `rtol` cannot
    express. Neighbours are rare: a rounding boundary has to fall inside that
    millionth, so at most 2^-7 of the elements may differ at all.

    Measured over 9216 rows at 6144, 512 and 512 at row stride 576: at most
    1 ulp, at 5.7e-6, 3.0e-6 and 1.3e-6 of the elements. A reference at eps
    1e-6 or 0 instead of 1e-5 lands 169-247 ulp away, and the weight shifted
    by one column 33,000. A wrong rounding stays within one ulp, and is what
    the count is for: over the same 9216 rows, truncating instead of rounding
    to nearest puts 50% of the elements a ulp off, rounding `x * rstd` or
    `rstd` to bf16 26-28%, and a mean over width - 1 1.1% at 6144 and 13.5%
    at 512 -- which a single 6144-wide row does not show and three rows do.

    Over the 32768 rows of the NVFP4 target's step, on sm_103: at most 1 ulp,
    at 6.44e-6 of the elements at 6144, 1.85e-6 at 512 at row stride 576 and
    1.55e-6 at contiguous 512. Past the FP8 target's 9216 rows, a read wrapped
    at row 9216, one shifted by a row, or the latent read at row stride 512
    instead of 576 lands 33,279-33,448 ulp away and fails every row it
    moved; rows left unwritten from 16384 fail at 16,704-16,801 ulp.
    """
    assert out.dtype is ref.dtype is torch.bfloat16, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    dist = (_ordered(out) - _ordered(ref)).abs()
    worst = int(dist.max())
    assert worst <= 1, f"{worst} bf16 ulp from the reference; the gate is 1"
    off = int((dist > 0).sum())
    assert off * 2**7 <= dist.numel(), (
        f"{off} of {dist.numel()} elements are a ulp off; at most 2^-7 of them may be"
    )


def _check(buf, x, weight, eps):
    """One call, gated; the input buffer, columns outside `x` included, is left as it was."""
    before = buf.clone()
    out = op(x, weight, eps)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert out.is_contiguous(), f"a new contiguous tensor is returned; got strides {out.stride()}"
    assert out.data_ptr() != x.data_ptr(), "the op must return a new tensor"
    _compare(out, op.reference(x, weight, eps))
    assert torch.equal(buf, before), "the op wrote into its input"
    return out


def _words(t):
    """`t`'s bits as int64 words over whole contiguous rows, for the sweep's compare."""
    assert t.is_contiguous() and (t.shape[-1] * t.element_size()) % 8 == 0
    return t.view(torch.int64)


def _sweep(spec, seed):
    """Every row count of `sweep_rows`, each its own call on the leading rows of one input.

    The call at the top of the range is gated against the reference; each
    smaller one has to equal the leading rows of that result bit for bit. The
    kernel computes each row on its own, so a row count that changed any row's
    answer would be a grid or bounds defect, not rounding.

    Each call is compared as whole contiguous rows of int64 words (the same
    bits, four bf16 to a word) and reduced with `any()`, which is still exact:
    over 1..32768 at 6144 on sm_103 the sweep took 4.9 s this way, against
    17.4 s with bf16 `!=` and `.sum()` and 2.0 s for the op calls alone. Each
    verdict stays on the device, so the sweep synchronizes once; the first
    failing row count is re-run to count its elements.
    """
    lo, hi = spec["sweep_rows"]
    assert spec["shape"][0] == hi, "a sweep's shape is its largest call"
    buf, x, weight = _build(spec, seed)
    eps = spec["eps"]
    full = _check(buf, x, weight, eps)
    full_words = _words(full)
    differs = [
        torch.ne(_words(op(x[:rows], weight, eps)), full_words[:rows]).any()
        for rows in range(lo, hi + 1)
    ]
    failing = torch.nonzero(torch.stack(differs)).flatten()
    if failing.numel():
        rows = lo + int(failing[0])
        count = int((op(x[:rows], weight, eps) != full[:rows]).sum())
        raise AssertionError(
            f"rows={rows}: {count} elements differ from the leading rows of the {hi}-row call"
        )


def _replays(spec, seed):
    """Each captured batch in its own CUDA graph, replayed on new data.

    The replay has to equal an eager call on the same data bit for bit -- a
    graph that froze anything of its capture would differ -- and is gated
    against the reference like any other output. Each capture follows one call
    on the capture stream, as an engine's warm-up does, so the kernel's
    first-call compile happens outside it.
    """
    eps = spec["eps"]
    stream = torch.cuda.Stream()
    g = torch.Generator(device="cuda").manual_seed(seed)
    for rows in spec["graph_rows"]:
        buf, x, weight = _build(spec, seed + rows, rows=rows)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            op(x, weight, eps)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = op(x, weight, eps)
        buf.copy_(_rows(rows, buf.shape[1], buf.dtype, g))
        graph.replay()
        eager = op(x, weight, eps)
        assert torch.equal(out, eager), f"rows={rows}: graph replay differs from eager"
        _compare(out, op.reference(x, weight, eps))


def _out_call(spec, buf, x, weight, eps):
    """One `out` cell's call: `out=x` in place, or into the leading 512 columns of a fresh
    576-wide row poisoned with NaN. Returns the row the result lands in and the result."""
    if spec["out"] == "inplace":
        return buf, op(x, weight, eps, out=x)
    row = torch.full((x.shape[0], 576), float("nan"), dtype=x.dtype, device="cuda")
    return row, op(x, weight, eps, out=row[:, :512])


def _check_out_row(spec, row, tail_before, out, fresh, x):
    """`out` is bitwise the call without it; the row's last 64 columns are as they were."""
    width = spec["shape"][-1]
    assert out.data_ptr() == row.data_ptr(), "the result lands at the start of the row"
    if spec["out"] == "inplace":
        assert out.data_ptr() == x.data_ptr()
    assert torch.equal(out.view(torch.int16), fresh.view(torch.int16)), (
        f"{int((out != fresh).sum())} elements differ from the call without out"
    )
    assert torch.equal(row[:, width:].view(torch.int16), tail_before.view(torch.int16)), (
        "a column outside out was written"
    )


def _out_cell(spec, seed):
    """An `out` cell: eager at its shape, or each of its `graph_rows` captured and replayed
    on new data. The call without `out` on the same data, gated against the reference, is
    what the `out` call is held to bit for bit."""
    eps = spec["eps"]
    if "graph_rows" not in spec:
        buf, x, weight = _build(spec, seed)
        fresh = op(x, weight, eps)
        _compare(fresh, op.reference(x, weight, eps))
        tail = buf[:, spec["shape"][-1] :].clone() if spec["out"] == "inplace" else None
        row, out = _out_call(spec, buf, x, weight, eps)
        if tail is None:
            tail = torch.full_like(row[:, spec["shape"][-1] :], float("nan"))
        _check_out_row(spec, row, tail, out, fresh, x)
        return
    stream = torch.cuda.Stream()
    g = torch.Generator(device="cuda").manual_seed(seed)
    for rows in spec["graph_rows"]:
        buf, x, weight = _build(spec, seed + rows, rows=rows)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _out_call(spec, buf.clone(), buf.clone()[:, : spec["shape"][-1]], weight, eps)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            row, out = _out_call(spec, buf, x, weight, eps)
        buf.copy_(_rows(rows, buf.shape[1], buf.dtype, g))
        if spec["out"] == "row576":
            row.fill_(float("nan"))
        x_in = x.clone()
        tail = row[:, spec["shape"][-1] :].clone()
        graph.replay()
        fresh = op(x_in, weight, eps)
        _compare(fresh, op.reference(x_in, weight, eps))
        _check_out_row(spec, row, tail, out, fresh, x)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    seed = zlib.crc32(cell.why.encode()) % 2**31
    if "out" in cell.spec:
        _out_cell(cell.spec, seed)
    elif "sweep_rows" in cell.spec:
        _sweep(cell.spec, seed)
    elif "graph_rows" in cell.spec:
        _replays(cell.spec, seed)
    else:
        _check(*_build(cell.spec, seed), cell.spec["eps"])


def test_the_gate_sees_eps() -> None:
    """The scaled rows are what make eps a value the cells certify, not a passenger."""
    spec = dict(shape=(16, 6144), dtype=torch.bfloat16, eps=1e-5)
    _, x, weight = _build(spec, seed=11)
    out = op(x, weight, 1e-5)
    _compare(out, op.reference(x, weight, 1e-5))
    for wrong in (1e-6, 0.0, 1e-4):
        with pytest.raises(AssertionError, match="bf16 ulp from the reference"):
            _compare(out, op.reference(x, weight, wrong))


def test_guard_refuses_fp8_before_the_kernel_answers() -> None:
    """The hazard `is_valid` exists for: accepted, and silently plausible.

    Driving the op directly shows why the guard is not cosmetic -- it returns a
    tensor rather than raising, so a caller who forgot to dequantize gets a
    number back.
    """
    x = torch.randn(4, 2880, device="cuda").to(torch.float8_e4m3fn)
    weight = torch.randn(2880, device="cuda").to(torch.float8_e4m3fn)
    unguarded = torch.ops.trtllm.flashinfer_rmsnorm(x, weight, 1e-6)
    assert unguarded.dtype is torch.float8_e4m3fn, "the hazard this guard covers is gone"

    with pytest.raises(AssertionError, match="dequantize"):
        with validating(op):
            op(x, weight, 1e-6)


@pytest.mark.parametrize(
    "layout, match",
    [
        ("last dim strided", r"Mismatched mX\.strides\[1\] on argument #0 .* expected to be 1"),
        (
            "row stride 516",
            r"Invalid mX\.strides\[0\] on argument #0 .* expected to be divisible by 8",
        ),
        ("8-byte aligned", r"Misaligned Tensor data on argument #0 .* expected data alignment=16"),
    ],
)
def test_the_op_refuses_a_layout_it_cannot_read(layout, match) -> None:
    """The views the op raises on before the launch, which is why `is_valid` does not restate them.

    A latent view (contiguous last dim, row stride a multiple of 8 elements,
    16-byte-aligned data) is accepted and is a cell; these are not. If the op
    starts accepting one, it needs a cell or a guard.
    """
    weight = torch.randn(512, dtype=torch.bfloat16, device="cuda")
    if layout == "last dim strided":
        x = torch.randn(8, 512, 2, dtype=torch.bfloat16, device="cuda")[..., 0]
    elif layout == "row stride 516":
        x = torch.randn(8, 516, dtype=torch.bfloat16, device="cuda")[:, :512]
    else:
        x = torch.randn(8, 576, dtype=torch.bfloat16, device="cuda")[:, 4:516]
    with pytest.raises(ValueError, match=match):
        op(x, weight, 1e-5)


def test_guard_refuses_an_out_that_overlaps_x() -> None:
    """An `out` shifted by one row inside `x`'s buffer reads rows the call already wrote."""
    buf = torch.randn(9, 576, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(512, dtype=torch.bfloat16, device="cuda")
    with validating(op):
        op(buf[:8, :512], weight, 1e-5, out=buf[:8, :512])
        with pytest.raises(AssertionError, match="out must be x itself"):
            op(buf[:8, :512], weight, 1e-5, out=buf[1:9, :512])


@pytest.mark.parametrize("rows", [1, 2, 3, 16, 64, 256, 4096, 32768])
def test_latent_rows_bit_equal_to_the_concatenation(rows) -> None:
    """Old versus new on identical inputs, `torch.equal` on the bits: a RoPE owner's latent
    normalized in place inside its 576-wide kv_a row equals `cat([rmsnorm(kv[:, :512]),
    kv[:, 512:]])`, and a NoPE owner's 512-wide kv_a output normalized into a zeroed 576-wide
    row equals `cat([rmsnorm(kv), zeros(64)])`."""
    g = torch.Generator(device="cuda").manual_seed(rows)
    weight = torch.randn(512, generator=g, device="cuda").to(torch.bfloat16)
    kv = _rows(rows, 576, torch.bfloat16, g)
    old = torch.cat([op(kv[:, :512], weight, 1e-5), kv[:, 512:]], dim=-1)
    new = kv.clone()
    op(new[:, :512], weight, 1e-5, out=new[:, :512])
    assert torch.equal(new.view(torch.int16), old.view(torch.int16)), "RoPE owner"
    kv512 = _rows(rows, 512, torch.bfloat16, g)
    zeros = torch.zeros(rows, 64, dtype=torch.bfloat16, device="cuda")
    old = torch.cat([op(kv512, weight, 1e-5), zeros], dim=-1)
    new = torch.zeros(rows, 576, dtype=torch.bfloat16, device="cuda")
    op(kv512, weight, 1e-5, out=new[:, :512])
    assert torch.equal(new.view(torch.int16), old.view(torch.int16)), "NoPE owner"
