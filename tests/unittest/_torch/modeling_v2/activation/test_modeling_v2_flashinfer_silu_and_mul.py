# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_silu_and_mul catalog entry.

The entry owns what is certified. This file turns a cell's spec into tensors
and drives the inputs the guard exists to refuse. A cell whose spec says
`large` is the every-count sweep at dep4's width 49152, past the per-cell
budget: pre-merge (`-m "not large"`) deselects it and drives its `sampled`
cell instead.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.flashinfer_silu_and_mul import (  # noqa: E501  # noqa: E501
    flashinfer_silu_and_mul as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — needs the path declared above

assert torch.cuda.is_available(), "flashinfer_silu_and_mul requires a CUDA device"


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(spec["shape"], generator=g, device="cuda").to(spec["dtype"])


def _check(x: torch.Tensor, out: torch.Tensor) -> None:
    """Shape, a fresh output, and torch's dtype-aware default band.

    The band is torch's bf16 default (rtol 1.6e-2, atol 1e-5); the test adds
    none of its own. Kernel and reference both form silu(gate) * up in fp32
    and round once to the input dtype; they differ only in how the fp32 exp
    and quotient are evaluated (the kernel's `silu` uses `__expf`), which
    surfaces as one bf16 rounding flip, or as +-0 where a gate near -88 puts
    the reference below 2e-35. Measured at widths 12288 and 3072 at 9216 rows,
    on the cells' own draws: bitwise equal for randn inputs; at 16x that
    scale, one step on 1730 of 56623104 outputs and three +-0 at width 12288,
    one step on 405 of 14155776 at 3072. Wrong variants at 9216 rows land far
    outside: gelu in place of silu at 62x the band, the halves swapped at
    8105-9383x, sigmoid in place of silu at 1.7e5x. Rounding silu(gate) to bf16
    before the multiply stays inside (0.49 of the band, 1 ulp): that is below
    what a single bf16 result can show. At 32768 rows, on the cells' own
    draws: one output per draw one step off, 0.11 of the band at worst; at 16x,
    5886 of 201326592 one step off and one +-0 where the reference is 8.2e-36,
    0.072 of the band. Outputs right on rows below 9216 and wrong after fail
    the band in every row they change: rows read wrapped at 9216 by up to
    1.0e6x, rows read one row early by up to 9.1e5x, rows from 16384 left at
    zero by 62.5x.

    Width 49152, where each thread makes three vector passes per half: at
    32768 rows the two eager cells' draws put 2 and 4 outputs one step off,
    0.071 of the band; at 16x, 0.072. Wrong variants at 4096 x 49152: the third
    pass skipped (columns 16384.. left zero) 62.5x the band, one thread's last
    vector dropped 62.5x, the third pass re-reading the second's vectors
    8.1e5x, up read one vector early 5.1e5x, the halves swapped 1.2e4x, rows
    shifted by one 8.5e5x; silu(gate) rounded to bf16 first stays inside, 0.49.

    Widths 6144 and 1536 (tep8 and tp8), one vector pass per half at 384 and 96
    threads: at 32768 rows the cells' draws put 1 and 0 outputs one step off,
    0.011 of the band; at 16x, 3033 and 712 one step off plus 3 and 0 of +-0,
    0.072. Wrong variants on those draws at 32768 rows (6144, then 1536): gelu
    in place of silu 62x the band, the last vector of each row dropped and the
    rows from 16384 left zero 62.5x, the halves swapped 1.5e4x and 7.8e3x, up
    read one vector early 6.8e5x and 4.4e5x, rows shifted by one 8.6e5x and
    8.5e5x, sigmoid in place of silu 1.9e5x and 1.5e5x; silu(gate) rounded to
    bf16 first stays inside, 0.49.
    """
    assert out.shape == (*x.shape[:-1], x.shape[-1] // 2)
    assert out.data_ptr() != x.data_ptr(), "the op must return a new tensor"
    torch.testing.assert_close(out, op.reference(x))


def _words(t: torch.Tensor) -> torch.Tensor:
    """`t`'s bits as int64 words over whole contiguous rows, for the sweep's compare."""
    assert t.is_contiguous() and (t.shape[-1] * t.element_size()) % 8 == 0
    return t.view(torch.int64)


def _sampled_row_counts(rows, dense_to, period, tail) -> list[int]:
    """Every count in 1..dense_to, either side of each multiple of `period`, the last `tail`.

    All below `rows`, whose own call is the one the others are compared with.
    """
    counts = set(range(1, dense_to + 1))
    counts.update(m for k in range(period, rows + 1, period) for m in (k - 1, k, k + 1))
    counts.update(range(rows - tail + 1, rows + 1))
    return sorted(m for m in counts if m < rows)


def _eager_row_counts(spec, seed, counts) -> None:
    """Each row count in `counts` as its own call, on the leading rows of one input.

    The call at `shape` is gated against the reference; each shorter call has
    to equal its leading rows bit for bit. A CTA reads only its own row, so a
    shorter call that differed would be a launch-shape defect, not rounding.

    Each call is compared as whole rows of int64 words (the same bits, four
    bf16 to a word) and reduced with `any()` on the device; the verdicts are
    stacked and read once, and the smallest failing row count is re-run to
    count its elements. Measured on sm_103 over 1..32768 at width 12288: 6.0 s
    this way, 3.2 s for the op calls alone, and 18.6 s with bf16 `!=` and
    `.sum()` per call, a form whose peak reserved device memory reached 255 GiB.
    At 49152, every count took 20.9 s (11.0 s the op calls, 10.7 GiB
    reserved) and the 939 sampled counts 0.27 s; with one bit flipped in the
    compared rows at row 20000, the sample flagged exactly its 141 counts past
    that row.
    """
    x = _build(spec, seed)
    full = op(x)
    _check(x, full)
    rows = x.shape[0]
    full_words = _words(full)
    differs = torch.stack([torch.ne(_words(op(x[:m])), full_words[:m]).any() for m in counts])
    bad = [counts[i] for i in torch.nonzero(differs).flatten().tolist()]
    if bad:
        count = int((op(x[: bad[0]]) != full[: bad[0]]).sum())
        raise AssertionError(
            f"rows={bad[0]}: {count} elements differ from the leading rows of the {rows}-row "
            f"call; {len(bad)} row counts differ, the smallest {bad[:16]}"
        )


def _every_captured_row_count(spec, seed) -> None:
    """Each row count 1..rows captured in its own CUDA graph, replayed on new data.

    The replay has to equal an eager call on the same data bit for bit -- a
    graph that froze anything of the capture would differ -- and is gated
    against the reference like any other output. One eager call on the capture
    stream precedes each capture, as an engine's warm-up does.
    """
    rows, width = spec["shape"]
    stream = torch.cuda.Stream()
    for m in range(1, rows + 1):
        shape = dict(spec, shape=(m, width))
        x = _build(shape, seed + m)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            op(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = op(x)
        fresh = _build(shape, seed + rows + m)
        x.copy_(fresh)
        graph.replay()
        eager = op(fresh)
        assert torch.equal(out, eager), f"rows={m}: graph replay differs from eager"
        _check(fresh, out)


@pytest.mark.parametrize(
    "cell",
    [pytest.param(c, marks=pytest.mark.large) if c.spec.get("large") else c for c in op.CELLS],
    ids=[c.why[:44] for c in op.CELLS],
)
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    seed = zlib.crc32(cell.why.encode()) % 2**31
    sweep = cell.spec.get("sweep")
    rows = cell.spec["shape"][0]
    if sweep == "eager":
        _eager_row_counts(cell.spec, seed, list(range(1, rows)))
    elif sweep == "sampled":
        counts = _sampled_row_counts(
            rows, cell.spec["dense_to"], cell.spec["period"], cell.spec["tail"]
        )
        _eager_row_counts(cell.spec, seed, counts)
    elif sweep == "graph":
        _every_captured_row_count(cell.spec, seed)
    else:
        assert sweep is None, f"unknown sweep {sweep!r}"
        x = _build(cell.spec, seed)
        _check(x, op(x))


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


@pytest.mark.parametrize("width", [12288, 3072, 6144, 1536])
@pytest.mark.parametrize("layout", ["column slice", "every other row"])
def test_guard_refuses_a_strided_input(layout, width) -> None:
    """The kernel reads row r at `r * width` from the base and ignores strides.

    Shown unguarded first: the op accepts a view whose rows are not packed and
    computes every row but the first from the wrong elements. Both views keep
    the rows the kernel reads inside the buffer this test allocated, and the
    output is the op's own allocation.
    """
    if layout == "column slice":
        x = torch.randn(16, width + 256, device="cuda").to(torch.bfloat16)[:, :width]
    else:
        x = torch.randn(32, width, device="cuda").to(torch.bfloat16)[::2]
    assert not x.is_contiguous()
    out, ref = op(x), op.reference(x)
    assert torch.equal(out[0], ref[0]), "row 0 sits at the base, where both read it"
    wrong = (out[1:] != ref[1:]).float().mean().item()
    assert wrong > 0.5, f"only {wrong:.3f} of rows 1.. differ; the strided read went unseen"
    with pytest.raises(AssertionError, match="must be contiguous"):
        with validating(op):
            op(x)


def test_guard_refuses_a_misaligned_base() -> None:
    """A packed view that starts off a 16-byte boundary faults the vector loads.

    Not driven unguarded: the fault is `CUDA error: misaligned address`, which
    is sticky and would take every later test in the process with it.
    """
    buf = torch.randn(16 * 3072 + 8, device="cuda").to(torch.bfloat16)
    x = buf[1 : 1 + 16 * 3072].view(16, 3072)
    assert x.is_contiguous() and x.data_ptr() % 16 == 2
    with pytest.raises(AssertionError, match="16-byte aligned"):
        with validating(op):
            op(x)


@pytest.mark.parametrize(
    "shape,dtype,match",
    [
        ((16, 3072), torch.float32, "failed to dispatch data type"),
        ((0, 3072), torch.bfloat16, "Failed to launch kernel: invalid argument"),
    ],
    ids=["fp32", "zero rows"],
)
def test_the_op_raises_itself(shape, dtype, match) -> None:
    """What the `note` says the op refuses loudly, kept true by running it."""
    x = torch.randn(shape, device="cuda").to(dtype)
    with pytest.raises(RuntimeError, match=match):
        op(x)
