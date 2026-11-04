# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_silu_and_mul catalog entry.

The entry owns what is certified. This file turns a cell's spec into tensors
and drives the inputs the guard exists to refuse. A cell whose spec says
`large` is the every-count sweep at dep4's width 49152, past the per-cell
budget: `test_large_certified_cells` drives it, which pre-merge
(`-m "not large"`) deselects, and pre-merge drives its `sampled` cell instead.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.flashinfer_silu_and_mul import (
    flashinfer_silu_and_mul as op,
)

__extra_import_path__ = [".."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import certifying, validating  # noqa: E402,F401 — needs the path declared above

assert torch.cuda.is_available(), "flashinfer_silu_and_mul requires a CUDA device"


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(spec["shape"], generator=g, device="cuda").to(spec["dtype"])


# The shapes each shipped target calls this entry with, by target directory
# under `models/`. The entry's own `CELLS` are the op's whoever calls it.
#
# Mistral Large 4 NVFP4 at tep4 and tp4 (tp4 differs from tep4 only in its
# routed experts, which do not call this entry) calls it on two gate_up outputs,
# each TP4-sharded to [gate | up] per rank: the dense layer 0 (intermediate
# 24576, so 2 x 6144) and the one shared expert of layers 1..71 (6144, so
# 2 x 1536). Each call site passes one row per token: decode is captured at the
# batches the engine pads to (1, 2, 4, 8, 16), prefill is eager at any row count
# up to the token budget -- 32768 (`max_num_tokens` 32768 with chunked prefill
# off, so one context step carries every row of up to 16 packed prompts); the
# 9216-row cells are the smaller budget the same widths ran at first. `sweep`
# drives every row count from 1 to `shape[0]`, each as its own call: "eager"
# against the call at `shape` bit for bit, "graph" each captured in its own CUDA
# graph and replayed on new data, bit for bit against eager. The two 32768-row
# sweeps are the file's costliest cells: 6.2 s and 2.1 s on sm_103 at widths
# 12288 and 3072, peaking 4.1 and 1.0 GiB of device memory.
_ML4_WIDTH_3072 = (
    Cell(
        why="eager rows 1..9216 at width 3072: the shared expert's gate and up, "
        "2 x 1536 per rank at tep4, every prefill row count up to the token budget",
        spec=dict(shape=(9216, 3072), dtype=torch.bfloat16, sweep="eager"),
    ),
    Cell(
        why="eager rows 1..32768 at width 3072: the shared expert's gate and up, "
        "2 x 1536 per rank at tep4, every prefill row count up to max_num_tokens 32768",
        spec=dict(shape=(32768, 3072), dtype=torch.bfloat16, sweep="eager"),
    ),
    Cell(
        why="CUDA graphs at rows 1..16, width 3072: the decode batches the engine "
        "captures (1, 2, 4, 8, 16, padded) and the rest an unpadded capture adds",
        spec=dict(shape=(16, 3072), dtype=torch.bfloat16, sweep="graph"),
    ),
)
# Width 12288 is also dep4's replicated shared expert, at the same rows.
_ML4_WIDTH_12288 = (
    Cell(
        why="eager rows 1..32768 at width 12288: dense layer 0's gate and up, "
        "2 x 6144 per rank at tep4, every prefill row count up to max_num_tokens 32768",
        spec=dict(shape=(32768, 12288), dtype=torch.bfloat16, sweep="eager"),
    ),
    Cell(
        why="CUDA graphs at rows 1..16, width 12288: the decode batches the engine "
        "captures (1, 2, 4, 8, 16, padded) and the rest an unpadded capture adds",
        spec=dict(shape=(16, 12288), dtype=torch.bfloat16, sweep="graph"),
    ),
)
_ML4_TP_WORLD4 = (
    Cell(
        why="eager rows 1..9216 at width 12288: dense layer 0's gate and up, "
        "2 x 6144 per rank at tep4, every prefill row count up to the token budget",
        spec=dict(shape=(9216, 12288), dtype=torch.bfloat16, sweep="eager"),
    ),
    *_ML4_WIDTH_12288,
    *_ML4_WIDTH_3072,
)

# The same two targets over eight ranks (tep8 and tp8, attention DP off) slice
# both call sites eight ways: dense layer 0's gate_up is 2 x 3072 = 6144 per
# rank and the shared expert's 2 x 768 = 1536, at the four-rank rows (1..32768
# eager, the padded decode batches 1/2/4/8/16 captured). Each width launches a
# block size no other cell runs (384 and 96 threads, see the entry's `note`), so
# each has its own every-count sweep and graph cell. On sm_103 the two
# 32768-row sweeps took 3.5 s and 1.9 s, peaking 2.1 and 0.5 GiB of device
# memory.
_ML4_TP_WORLD8 = (
    Cell(
        why="eager rows 1..32768 at width 6144: dense layer 0's gate and up, "
        "2 x 3072 per rank at tep8 and tp8, every prefill row count up to max_num_tokens "
        "32768",
        spec=dict(shape=(32768, 6144), dtype=torch.bfloat16, sweep="eager"),
    ),
    Cell(
        why="eager rows 1..32768 at width 1536: the shared expert's gate and up, "
        "2 x 768 per rank at tep8 and tp8, every prefill row count up to max_num_tokens "
        "32768",
        spec=dict(shape=(32768, 1536), dtype=torch.bfloat16, sweep="eager"),
    ),
    Cell(
        why="CUDA graphs at rows 1..16, width 6144: the decode batches the engine "
        "captures at tep8 and tp8 (1, 2, 4, 8, 16, padded) and the rest an unpadded "
        "capture adds",
        spec=dict(shape=(16, 6144), dtype=torch.bfloat16, sweep="graph"),
    ),
    Cell(
        why="CUDA graphs at rows 1..16, width 1536: the decode batches the engine "
        "captures at tep8 and tp8 (1, 2, 4, 8, 16, padded) and the rest an unpadded "
        "capture adds",
        spec=dict(shape=(16, 1536), dtype=torch.bfloat16, sweep="graph"),
    ),
)

# Mistral Large 4 NVFP4 at dep4 and dep8 (attention data parallel) runs dense
# layer 0 replicated on each rank's own tokens, so its gate_up output is the
# whole [gate | up], 2 x 24576 = 49152, at the same per-rank rows: 1..32768
# eager, the padded decode batches 1/2/4/8/16 captured. Its replicated shared
# expert keeps width 12288, `_ML4_WIDTH_12288`. Every row count at
# 49152 moves four times the bytes of 12288, so that sweep is `large`, past the
# 10 s budget, and pre-merge drives "sampled" instead: the 32768-row call gated
# against `reference`, and each count in 1..`dense_to`, either side of every
# multiple of `period` and the last `tail` bitwise its leading rows. The period
# is one wave of resident CTAs on sm_103 (two 1024-thread CTAs on each SM). On
# sm_103 the `large` cell took 21.0 s and the sampled one 0.3 s, each reserving
# 18.0 GiB (the reference gate on the 32768 x 49152 call); the 49152 graphs
# 0.1 s.
_ML4_DP = (
    Cell(
        why="eager rows 1..32768 at width 49152: dense layer 0's gate and up, 2 x 24576 "
        "replicated on each rank's own tokens at dep4, every prefill row count up to "
        "max_num_tokens 32768",
        spec=dict(shape=(32768, 49152), dtype=torch.bfloat16, sweep="eager", large=True),
    ),
    Cell(
        why="eager rows sampled to 32768 at width 49152, pre-merge's share of the large "
        "every-count cell: every count through two waves of resident CTAs, both sides "
        "of each later wave, the last 16",
        spec=dict(
            shape=(32768, 49152),
            dtype=torch.bfloat16,
            sweep="sampled",
            dense_to=608,
            period=304,
            tail=16,
        ),
    ),
    Cell(
        why="CUDA graphs at rows 1..16, width 49152: the decode batches the engine "
        "captures at dep4 (1, 2, 4, 8, 16, padded) and the rest an unpadded capture adds",
        spec=dict(shape=(16, 49152), dtype=torch.bfloat16, sweep="graph"),
    ),
)


def _ml4_dspark(*widths: int) -> tuple[Cell, ...]:
    """DSpark's dense draft and target MLPs at verification batch sizes, graph rows 1..64.

    Their index bounds are below the eager envelope of the cells above.
    """
    return tuple(
        Cell(
            why=f"DSpark SwiGLU width {width}, graph rows 1..64",
            spec=dict(shape=(64, width), dtype=torch.bfloat16, sweep="graph"),
        )
        for width in widths
    )


TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "deepseek_v3/r1_0528_nvfp4__sm_10x__default": (
        Cell(
            why="deepseek default at attention TP8: the shared expert's gate+up, 2 x 2048/8",
            spec=dict(shape=(16, 512), dtype=torch.bfloat16),
        ),
    ),
    "mistral_large4/mistral_large_4_nvfp4__sm_103__tep4": (
        *_ML4_TP_WORLD4,
        *_ML4_TP_WORLD8,
        *_ml4_dspark(12288, 3072, 6144, 1536),
    ),
    "mistral_large4/mistral_large_4_nvfp4__sm_103__tp4": (
        *_ML4_TP_WORLD4,
        *_ML4_TP_WORLD8,
        *_ml4_dspark(12288, 3072, 6144, 1536),
    ),
    "mistral_large4/mistral_large_4_nvfp4__sm_103__dep4": (
        *_ML4_DP,
        *_ML4_WIDTH_12288,
        *_ml4_dspark(49152, 12288),
    ),
}

CELLS = certified_cells(op, TARGET_CELLS)


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


def _drive(cell) -> None:
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


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    if cell.spec.get("large"):
        pytest.skip("past the per-cell budget; test_large_certified_cells drives it")
    certifying(op)
    _drive(cell)


_LARGE_CELLS = tuple(c for c in CELLS if c.spec.get("large"))


@pytest.mark.large
@pytest.mark.parametrize("cell", _LARGE_CELLS, ids=[c.why[:44] for c in _LARGE_CELLS])
def test_large_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    _drive(cell)


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
