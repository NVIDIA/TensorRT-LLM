# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_projection entry."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("cute_dsl_projection runs sm_103 CuTe DSL kernels", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm.cute_dsl_projection import (  # noqa: E402, E501
    MTP_ROWS,
    PREFILL_MAX_ROWS,
    ROWS,
    SHARED_DOWN_PREFILL_MIN_ROWS,
    PrefillProjectionRunner,
    ProjectionRunner,
    SharedDownPrefillRunner,
    at_rows,
    build,
    build_prefill,
    build_shared_down_prefill,
    shape_of,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm.cute_dsl_projection import (  # noqa: E402, E501
    cute_dsl_projection as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

#: One unit of a TF32 operand's 10-bit mantissa.
TF32_ULP = 2.0**-10
#: One unit of fp32's mantissa.
FP32_ULP = 2.0**-23


def _compare(kernel: str, out: torch.Tensor, ref: torch.Tensor) -> None:
    """The band, by what each kernel rounds.

    bf16 output (eight kernels, the prefill GEMM's bf16 entries, and the
    shared expert's prefill down kernel): both sides accumulate in fp32 and
    round once to bf16, so they differ by at most one rounding of nearly
    equal fp32 sums: one ulp of an element, at most two of the row-scale
    units the gate counts in. 4 ulp element-wise and 2 ulp relative RMS, as
    `bmm_out` uses; a wrong product misses by orders.

    fp32 output from bf16 operands (router): products are exact, so the only
    difference is the order of K = 6144 fp32 additions against an fp64
    reference. A random-walk estimate is a few fp32 ulps of the row scale.
    The band is 512 ulp (2**-14) element-wise and 128 ulp (2**-16) RMS, wide
    enough for an accumulator that truncates instead of rounding; a wrong
    logit is O(1).

    ix_fused: fp32 operands rounded to TF32 inside the kernel, against the
    exact fp32 product. `x` holds bf16 values (as the target's `x.float()`
    does), so only the weight's rounding counts: at most half a TF32 ulp per
    product, which sums to about 0.4 TF32 ulp of relative RMS over random
    signs and 1-2 ulp at the worst element of a row. 4 ulp element-wise and
    2 ulp RMS, in TF32 units; that also holds if the hardware truncates
    instead of rounding.

    e4m3 output (the fp8 attention's entries): both sides accumulate in fp32 and
    round once to e4m3, so an element differs only where the two fp32 sums straddle an
    e4m3 rounding boundary: by one e4m3 step (3 mantissa bits), at a small fraction of
    the elements. At most one step everywhere and at most 1% of the elements off; a
    wrong product or a lost scale misses most elements by many steps.
    """
    if out.dtype == torch.float8_e4m3fn:
        got, want = out.float(), ref.float()
        assert bool(torch.isfinite(got).all()), "an element was not written"
        # One e4m3 step at the reference's magnitude (subnormal step 2^-9 below 2^-6).
        step = torch.exp2(torch.floor(torch.log2(want.abs().clamp_min(2.0**-6))) - 3)
        diff = (got - want).abs()
        assert bool((diff <= step).all()), f"max error {float((diff / step).max()):.1f} steps"
        frac = float((diff > 0).float().mean())
        assert frac <= 0.01, f"{frac:.2%} of the elements differ from the reference"
        return
    if kernel == "ix_fused":
        assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0, ulp=TF32_ULP)
    elif out.dtype == torch.float32:
        assert_within_ulp(out, ref, element_ulp=512.0, rms_ulp=128.0, ulp=FP32_ULP)
    else:
        assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


def _weight(kernel: str, g: torch.Generator) -> torch.Tensor:
    """A row-major `[N, K]` weight scaled so each output is O(1). ix_fused's
    is full-mantissa fp32, so the kernel's TF32 rounding is exercised."""
    s = shape_of(kernel)
    return (torch.randn(s.n, s.k, generator=g, device="cuda") * s.k**-0.5).to(s.in_dtype)


#: The shared expert's prefill down runner, built once per session.
_SHARED_DOWN_PREFILL: dict = {}


def _shared_down_prefill_runner() -> SharedDownPrefillRunner:
    if "runner" not in _SHARED_DOWN_PREFILL:
        g = torch.Generator(device="cuda").manual_seed(0)
        _SHARED_DOWN_PREFILL["runner"] = build_shared_down_prefill(_weight("shared_down", g))
    return _SHARED_DOWN_PREFILL["runner"]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    kernel, rows = cell.spec["kernel"], cell.spec["rows"]
    s = shape_of(kernel)
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    if cell.spec.get("prefill"):
        runner = build_prefill(kernel, _weight(kernel, g))
        assert (runner.n, runner.k, runner.out_dtype) == (s.n, s.k, s.out_dtype)
    elif cell.spec.get("shared_prefill"):
        runner = _shared_down_prefill_runner()
        assert (runner.n, runner.k, runner.out_dtype) == (s.n, s.k, s.out_dtype)
    else:
        runner = build(kernel, rows, _weight(kernel, g))
    # Another layer's weight: the target builds once against layer 0 and runs
    # every layer through that runner, so the runner must not have kept it.
    weight = _weight(kernel, g)
    # bf16 values even for ix_fused, whose caller passes `x.float()`.
    x = torch.randn(rows, s.k, generator=g, device="cuda").to(torch.bfloat16).to(s.in_dtype)
    if cell.spec.get("x_row_stride"):
        # The tail of a wider row, as the kv_b expansion reads the latent in place.
        wide = torch.zeros(rows, cell.spec["x_row_stride"], dtype=s.in_dtype, device="cuda")
        wide[:, -s.k :] = x
        x = wide[:, -s.k :]
    # NaN, so an element the kernel never writes fails the comparison.
    # (e4m3 through a bf16 fill: a float8 fill kernel is not available on every build.)
    out = torch.full((rows, s.n), float("nan"), dtype=torch.bfloat16, device="cuda").to(s.out_dtype)
    with validating(op):
        assert op(x, weight, out, runner) is out, "the op writes `out` and returns it"
    _compare(kernel, out, op.reference(x, weight, out, runner))


def test_build_refuses_a_row_count_without_a_schedule() -> None:
    """The target sends such a row count to cuBLAS; `build` must not pretend."""
    with pytest.raises(ValueError, match="no schedule for 3 rows"):
        build("qkva", 3, torch.zeros(2624, 6144, dtype=torch.bfloat16, device="cuda"))


def _unreachable(*args) -> None:
    raise AssertionError("the guard should have refused before the kernel ran")


def _runner(kernel: str, rows: int) -> ProjectionRunner:
    """A runner whose kernel is never reached: `is_valid` runs first."""
    s = shape_of(kernel)
    return ProjectionRunner(kernel, rows, s.n, s.k, s.in_dtype, s.out_dtype, _unreachable)


def _ok(kernel: str, rows: int):
    s = shape_of(kernel)
    x = torch.zeros(rows, s.k, dtype=s.in_dtype, device="cuda")
    w = torch.zeros(s.n, s.k, dtype=s.in_dtype, device="cuda")
    out = torch.empty(rows, s.n, dtype=s.out_dtype, device="cuda")
    return x, w, out, _runner(kernel, rows)


def test_guard_refuses_the_column_major_view_of_a_square_weight() -> None:
    """dense_gate_up is 6144 x 6144: the `[K, N]` view `cublas_mm` takes has
    the right shape, and the kernel checks layout on a runner's first call only."""
    x, w, out, runner = _ok("dense_gate_up", 24)
    with pytest.raises(AssertionError, match="must be contiguous"):
        with validating(op):
            op(x, w.t(), out, runner)


def test_guard_refuses_a_bf16_out_for_the_router() -> None:
    """The router kernel writes fp32 into whatever buffer it is given."""
    x, w, out, runner = _ok("router", 8)
    with pytest.raises(AssertionError, match="writes torch.float32"):
        with validating(op):
            op(x, w, out.to(torch.bfloat16), runner)


def test_guard_refuses_a_strided_input() -> None:
    """A column slice of a wider buffer. The bf16 kernels check layout on a
    runner's first call only, so nothing else stops it."""
    s = shape_of("qkva")
    wide = torch.zeros(8, s.k + 64, dtype=torch.bfloat16, device="cuda")
    _, w, out, runner = _ok("qkva", 8)
    with pytest.raises(AssertionError, match="x must be contiguous"):
        with validating(op):
            op(wide[:, : s.k], w, out, runner)


def _prefill_runner(kernel: str) -> PrefillProjectionRunner:
    """A prefill runner whose GEMM is never reached: `is_valid` runs first."""
    s = shape_of(kernel)
    return PrefillProjectionRunner(
        kernel, s.n, s.k, s.in_dtype, s.out_dtype, PREFILL_MAX_ROWS, _unreachable
    )


def test_guard_refuses_a_prefill_row_count_past_the_ceiling() -> None:
    """The target chunks above max_num_tokens; the GEMM's schedule table ends there."""
    s = shape_of("q_b")
    rows = PREFILL_MAX_ROWS + 1
    x = torch.zeros(rows, s.k, dtype=s.in_dtype, device="cuda")
    w = torch.zeros(s.n, s.k, dtype=s.in_dtype, device="cuda")
    out = torch.empty(rows, s.n, dtype=s.out_dtype, device="cuda")
    with pytest.raises(AssertionError, match="serves 1..8192 rows"):
        with validating(op):
            op(x, w, out, _prefill_runner("q_b"))


def test_guard_refuses_the_prefill_gemm_a_view_of_a_weight_it_has_met() -> None:
    """The prefill GEMM caches each weight's operand view by address and checks
    layout only on first sight: the column-major view of the square
    dense_gate_up weight shares that address and the shape."""
    s = shape_of("dense_gate_up")
    x = torch.zeros(80, s.k, dtype=s.in_dtype, device="cuda")
    w = torch.zeros(s.n, s.k, dtype=s.in_dtype, device="cuda")
    out = torch.empty(80, s.n, dtype=s.out_dtype, device="cuda")
    with pytest.raises(AssertionError, match="must be contiguous"):
        with validating(op):
            op(x, w.t(), out, _prefill_runner("dense_gate_up"))


def test_build_prefill_refuses_an_entry_the_kernel_lacks_or_mistypes() -> None:
    """Every refusal leaves the projection on cuBLAS at the call site, never on a
    launcher that writes the wrong dtype."""
    for kernel in ("qkva", "router", "mtp_eh"):
        s = (6144, 12288) if kernel == "mtp_eh" else (shape_of(kernel).n, shape_of(kernel).k)
        weight = torch.zeros(*s, dtype=torch.bfloat16, device="cuda")
        try:
            runner = build_prefill(kernel, weight)
        except ValueError as exc:
            assert "has no" in str(exc) or "writes" in str(exc), exc
            continue
        expected = torch.float32 if kernel == "router" else torch.bfloat16
        assert runner.out_dtype == expected, (kernel, runner.out_dtype)


def test_guard_refuses_a_misaligned_input() -> None:
    """The bf16 kernels' TMA needs 128-byte aligned operands."""
    s = shape_of("o_proj")
    flat = torch.zeros(8 * s.k + 8, dtype=torch.bfloat16, device="cuda")
    _, w, out, runner = _ok("o_proj", 8)
    with pytest.raises(AssertionError, match="x must be 128-byte aligned"):
        with validating(op):
            op(flat[8:].view(8, s.k), w, out, runner)


def test_the_shared_prefill_down_refuses_bad_operands_loudly() -> None:
    """The shared expert's prefill kernel checks rows, shapes, dtypes, alignment and
    overlap on every call, so `is_valid` adds nothing for it; this pins that the
    refusals are loud: 64 rows (the decode kernel's), and an `out` aligned for the
    decode kernel (32 bytes) but not for this one (128)."""
    runner = _shared_down_prefill_runner()
    s = shape_of("shared_down")
    x = torch.zeros(64, s.k, dtype=s.in_dtype, device="cuda")
    w = torch.zeros(s.n, s.k, dtype=s.in_dtype, device="cuda")
    with pytest.raises(ValueError, match="65..8192"):
        op(x, w, torch.empty(64, s.n, dtype=s.out_dtype, device="cuda"), runner)
    rows = SHARED_DOWN_PREFILL_MIN_ROWS
    x = torch.zeros(rows, s.k, dtype=s.in_dtype, device="cuda")
    # 16 bf16 elements past a 512-byte aligned allocation: 32-byte aligned, not 128.
    flat = torch.empty(rows * s.n + 16, dtype=s.out_dtype, device="cuda")
    out = flat[16:].view(rows, s.n)
    assert out.data_ptr() % 32 == 0 and out.data_ptr() % 128 != 0
    with pytest.raises(ValueError, match="alignment"):
        op(x, w, out, runner)


def test_the_mtp_rows_are_decode_rows_left_out_by_default() -> None:
    """An engine drafting one token builds exactly `ROWS`, as before the MTP3 / MTP5
    cells existed: they sit above the decode row counts and outside the default set."""
    for kernel, rows in MTP_ROWS.items():
        assert all(m > 64 for m in rows), (kernel, rows)
        assert not set(rows) & set(ROWS[kernel]), (kernel, rows)


_MTP_CELLS = [c for c in op.CELLS if c.spec["rows"] in MTP_ROWS.get(c.spec["kernel"], ())]


@pytest.mark.parametrize("cell", _MTP_CELLS, ids=[c.why[:44] for c in _MTP_CELLS])
def test_mtp_cells_replay_bitwise_in_a_cuda_graph(cell) -> None:
    """The MTP3 / MTP5 cells run inside the decode CUDA graph. A replay writes the eager
    launch's bytes, and a replay after new operands are copied into the captured buffers
    writes the new product, bitwise the eager launch's on them: nothing but the buffers'
    addresses is baked into the graph."""
    kernel, rows = cell.spec["kernel"], cell.spec["rows"]
    s = shape_of(kernel)
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    weight = _weight(kernel, g)
    runner = build(kernel, rows, weight)
    x = torch.randn(rows, s.k, generator=g, device="cuda").to(s.in_dtype)
    out = torch.empty(rows, s.n, dtype=s.out_dtype, device="cuda")
    op(x, weight, out, runner)
    eager = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(x, weight, out, runner)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager), "the replay differs from the eager launch"
    x.copy_(torch.randn(rows, s.k, generator=g, device="cuda").to(s.in_dtype))
    weight.copy_(_weight(kernel, g))
    again = torch.empty_like(out)
    op(x, weight, again, runner)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, again), "the replay on new operands differs from the eager launch"
    assert not torch.equal(again, eager), "the new operands did not change the product"
    _compare(kernel, out, op.reference(x, weight, out, runner))


#: MTP3 / MTP5 decode row counts without a cell: each bucket's edges (65, 127 / 129, 191),
#: the 24-request batches (96, 144) and partial batches at widths 4 / 6.
WIDE_ROWS = (65, 66, 68, 84, 96, 100, 126, 127, 129, 144, 150, 190, 191)
#: What `out`'s storage past its last row holds before the call; it must hold it after.
CANARY = -7.0


@pytest.mark.parametrize("kernel", [k for k, rows in MTP_ROWS.items() if rows])
def test_mtp_cells_at_fewer_rows_are_the_cells_bitwise(kernel) -> None:
    """`at_rows` re-pins an MTP cell to fewer rows: the target's MTP3 / MTP5 decode row
    counts without a cell run on the next cell up at their exact rows. Every row is bitwise
    what the cell writes for it at its own row count with the input padded by other
    (random) rows, nothing is written past the last row, the product is within the band of
    the fp32 reference, and a captured launch replays bitwise the eager one on new operands.
    """
    s = shape_of(kernel)
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(kernel.encode()) % 2**31)
    weight = _weight(kernel, g)
    # lm_head's 65..127 logits rows take its 128-row (context) instance, the next one up.
    extra = (128,) if kernel == "lm_head" else ()
    cells = {m: build(kernel, m, weight) for m in MTP_ROWS[kernel] + extra}
    graphed = set()
    for rows in WIDE_ROWS:
        cell = min((m for m in cells if m > rows), default=None)
        if cell is None:
            continue
        runner = at_rows(cells[cell], rows)
        assert runner is not None and (runner.rows, runner.n) == (rows, s.n)
        x = torch.randn(rows, s.k, generator=g, device="cuda").to(s.in_dtype)
        parent = torch.full((rows + 8, s.n), CANARY, dtype=s.out_dtype, device="cuda")
        out = parent[:rows]
        with validating(op):
            assert op(x, weight, out, runner) is out
        padded = torch.randn(cell, s.k, generator=g, device="cuda").to(s.in_dtype)
        padded[:rows] = x
        full = torch.empty(cell, s.n, dtype=s.out_dtype, device="cuda")
        op(padded, weight, full, cells[cell])
        torch.cuda.synchronize()
        assert torch.equal(out, full[:rows]), f"{rows} rows on the {cell}-row cell: not the cell's"
        assert bool((parent[rows:] == CANARY).all()), f"{rows} rows: wrote past the last row"
        _compare(kernel, out, op.reference(x, weight, out, runner))
        if cell in graphed:
            continue
        graphed.add(cell)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op(x, weight, out, runner)
        x.copy_(torch.randn(rows, s.k, generator=g, device="cuda").to(s.in_dtype))
        eager = torch.empty_like(out)
        op(x, weight, eager, runner)
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager), f"{rows} rows: the replay differs from the eager launch"
        assert bool((parent[rows:] == CANARY).all()), f"{rows} rows: the replay wrote past"


#: The projections the target sends through `_project` at decode rows.
DECODE_KERNELS = (
    "qkva",
    "q_b",
    "o_proj",
    "ix_qb",
    "dense_gate_up",
    "dense_down",
    "shared_gate_up",
    "shared_down",
)
#: MTP5 at 1-3 requests (6 / 12 / 18 rows; MTP3 at 3 is 12): captured decode row counts without
#: a cell. 16 (a cell of every kernel here) is the control: its own cell, no re-pinning.
DECODE_ROWS = (6, 12, 18, 16)


def _next_cell(cells: dict, mtp_cells: dict, rows: int):
    """The target's `_draft_cell` rule: the smallest decode cell above `rows` up to 64 that
    launches at a runtime row count, else the smallest such MTP cell. `(bucket, runner)`."""
    ups = [(m, cells[m]) for m in sorted(cells) if rows < m <= 64]
    ups += [(m, mtp_cells[m]) for m in sorted(mtp_cells) if m > rows]
    for m, up in ups:
        runner = at_rows(up, rows)
        if runner is not None:
            return m, runner
    return None, None


@pytest.mark.parametrize("rows", DECODE_ROWS)
@pytest.mark.parametrize("kernel", DECODE_KERNELS)
def test_decode_rows_on_the_next_cell_up(kernel, rows) -> None:
    """The captured MTP3 / MTP5 decode row counts without a cell run on the next cell up
    (`_cell_up` -> `_draft_cell`) instead of cuBLAS: each row bitwise what that cell writes
    for it with the input padded by other (random) rows, nothing past the last row, within
    the band of the fp32 reference, and a captured launch replays bitwise on new operands.
    At 16 rows the kernel's own cell runs as before (control)."""
    s = shape_of(kernel)
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(f"{kernel}{rows}".encode()) % 2**31)
    weight = _weight(kernel, g)
    cells = {m: build(kernel, m, weight) for m in ROWS[kernel] if m <= 64}
    mtp_cells = {m: build(kernel, m, weight) for m in MTP_ROWS[kernel]}
    if rows in cells:
        bucket, runner = rows, cells[rows]
    else:
        bucket, runner = _next_cell(cells, mtp_cells, rows)
        if runner is None:
            pytest.skip(f"{kernel}: no cell launches at {rows} rows (the target keeps cuBLAS)")
        assert (runner.rows, runner.n) == (rows, s.n)
    x = torch.randn(rows, s.k, generator=g, device="cuda").to(s.in_dtype)
    parent = torch.full((rows + 8, s.n), CANARY, dtype=s.out_dtype, device="cuda")
    out = parent[:rows]
    with validating(op):
        assert op(x, weight, out, runner) is out
    if bucket != rows:
        full_cell = cells.get(bucket) or mtp_cells[bucket]
        padded = torch.randn(bucket, s.k, generator=g, device="cuda").to(s.in_dtype)
        padded[:rows] = x
        full = torch.empty(bucket, s.n, dtype=s.out_dtype, device="cuda")
        op(padded, weight, full, full_cell)
        torch.cuda.synchronize()
        assert torch.equal(out, full[:rows]), f"{rows} rows on the {bucket}-row cell: not its"
    assert bool((parent[rows:] == CANARY).all()), f"{rows} rows: wrote past the last row"
    _compare(kernel, out, op.reference(x, weight, out, runner))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(x, weight, out, runner)
    x.copy_(torch.randn(rows, s.k, generator=g, device="cuda").to(s.in_dtype))
    eager = torch.empty_like(out)
    op(x, weight, eager, runner)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager), f"{rows} rows: the replay differs from the eager launch"
    assert bool((parent[rows:] == CANARY).all()), f"{rows} rows: the replay wrote past"


def test_at_rows_refuses_more_rows_and_a_cell_compiled_for_one_m() -> None:
    """More rows than the cell's would read and write past the operands; the tcgen05
    decode cells (q_b at 64 rows) are compiled for their M and cannot take another."""
    weight = _weight("q_b", torch.Generator(device="cuda").manual_seed(3))
    assert at_rows(build("q_b", 128, weight), 129) is None
    assert at_rows(build("q_b", 128, weight), 0) is None
    assert at_rows(build("q_b", 64, weight), 48) is None
