# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_absorbed_bmm entry."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("cute_dsl_absorbed_bmm runs sm_103 CuTe DSL kernels", allow_module_level=True)

from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.kernels.common import (  # noqa: E501
    self_attn_prefill_act_quant_mxf8 as quant,
)  # noqa: E402, E501
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm.cute_dsl_absorbed_bmm import (  # noqa: E402, E501
    DECODE_RUNTIME_ROWS,
    LAYOUTS,
    MTP_GROUPS,
    MTP_RUNTIME_ROWS,
    PREFILL_MAX_ROWS,
    AbsorbedBmmRunner,
    build,
    build_decode_fp8_q_base,
    build_prefill,
    build_prefill_fp8_q_base,
    build_v_b_mxf8_side,
)
from tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.gemm.cute_dsl_absorbed_bmm import (  # noqa: E402, E501
    cute_dsl_absorbed_bmm as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

#: What the storage around `out` holds before the call; it must hold it after.
SENTINEL = -12345.0
#: Rows of storage past the last row the op is given.
CANARY_ROWS = 8


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Both sides take bf16 operands, sum K = 192 or 512 exact products in
    fp32, and round once to bf16: they differ by at most one rounding of
    nearly equal sums, one ulp of an element. 4 ulp element-wise against the
    row scale and 2 ulp relative RMS, the band `bmm_out` uses for the same two
    GEMMs; a wrong product misses it by orders. The decode and prefill kernels
    differ only in tiling, so one band serves both."""
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


def _group(rows: int) -> str:
    return "m_le_8" if rows <= 8 else "m_ge_16" if rows <= 64 else "m_ge_128"


def _build(kernel: str, rows: int, seed: int):
    """`(a, weight, out, out_parent)` as the target passes them.

    `a` and `out` are `transpose(0, 1)` views of token-major `[M, 16, row]`
    storage: `a` the first K columns of each row, `out` the first N. The
    parent of `out` starts as SENTINEL so a write past N, or past the rows,
    shows.
    """
    lay = LAYOUTS[kernel]
    g = torch.Generator(device="cuda").manual_seed(seed)
    a_parent = torch.randn(rows, lay.heads, lay.a_row, generator=g, device="cuda")
    a = a_parent.to(torch.bfloat16)[..., : lay.k].transpose(0, 1)
    weight = torch.randn(lay.heads, lay.n, lay.k, generator=g, device="cuda") * lay.k**-0.5
    weight = weight.to(torch.bfloat16)
    # CANARY_ROWS rows past the view: same strides, and room to see an
    # unpredicated store write past M.
    parent = torch.full(
        (rows + CANARY_ROWS, lay.heads, lay.out_row), SENTINEL, dtype=torch.bfloat16, device="cuda"
    )
    out = parent[:rows, :, : lay.n].transpose(0, 1)
    return a, weight, out, parent


_PREFILL: dict[str, AbsorbedBmmRunner] = {}


def _prefill(kernel: str) -> AbsorbedBmmRunner:
    """One prefill runner per kernel for the module, as the target builds it."""
    if kernel not in _PREFILL:
        if kernel == "k_b_fp8":
            _PREFILL[kernel] = build_prefill_fp8_q_base(torch.cuda.current_device())
        else:
            _PREFILL[kernel] = build_prefill(kernel, torch.cuda.current_device())
    return _PREFILL[kernel]


#: The fp8 query buffer's bytes before the call, outside what K-B writes.
SENTINEL_BYTE = 0xA5


def _check_fp8_cell(rows: int, seed: int, decode: bool = False) -> None:
    """The e4m3 bytes K-B writes are exactly the bf16 prefill K-B's output converted the
    way the rope/append kernel converts it (nearest, saturating): same kernel, same
    accumulation, only the epilogue differs. The rope tail of each head (`[..., 512:]`)
    and the rows past M stay as they were."""
    lay = LAYOUTS["k_b"]
    a, weight, out_bf16, _ = _build("k_b", rows, seed)
    dev = torch.cuda.current_device()
    op(a, weight, out_bf16, build("k_b", _group(rows), dev) if decode else _prefill("k_b"))
    quant = torch.full(
        (rows + CANARY_ROWS, lay.heads, lay.out_row), SENTINEL_BYTE, dtype=torch.uint8,
        device="cuda",
    )  # fmt: skip
    out = quant[:rows, :, : lay.n].transpose(0, 1)
    runner = build_decode_fp8_q_base(_group(rows), dev) if decode else _prefill("k_b_fp8")
    assert rows in runner.rows
    with validating(op):
        assert op(a, weight, out, runner) is out, "the op writes `out` and returns it"
    torch.cuda.synchronize()
    want = out_bf16.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
    bad = int((out != want).sum())
    assert bad == 0, f"{bad} e4m3 bytes differ from e4m3(bf16 K-B)"
    # And within an e4m3 step of the fp32-accumulated reference.
    ref = op.reference(a, weight, out, runner).view(torch.float8_e4m3fn).float()
    got = out.view(torch.float8_e4m3fn).float()
    assert bool(((got - ref).abs() <= ref.abs() * 2.0**-3 + 2.0**-9).all())
    assert bool((quant[:rows, :, lay.n :] == SENTINEL_BYTE).all()), "the kernel wrote past N"
    assert bool((quant[rows:] == SENTINEL_BYTE).all()), "the kernel wrote past the last row"


#: o_proj's decode operands the V-B side output writes, one per held row count for the module
#: (64: the decode groups; 192: the MTP group, the wide decode o_proj's): the kernels bake
#: their addresses in, so they must outlive them.
_X8: dict = {}


def _x8_operand(held: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(held, 4096, dtype=torch.uint8, device="cuda"),
        torch.full((quant.sf_bytes(held, 4096),), 127, dtype=torch.uint8, device="cuda"),
    )


def _check_x8_cell(rows: int, seed: int) -> None:
    """The V-B's bf16 output is the plain decode cell's, bit for bit, and its side output is
    the activation quant's bytes of that output read token-major as `[M, 4096]` (rows and
    scales of rows < M; the wide operand's rows were NaN bytes before the call)."""
    dev = torch.cuda.current_device()
    group = _group(rows)
    if group not in _X8:
        held = 192 if group in MTP_GROUPS else 64
        operand = _X8.setdefault(held, _x8_operand(held))
        _X8[group] = build_v_b_mxf8_side(group, dev, operand)
    xq, sf = _X8[192 if group in MTP_GROUPS else 64]
    xq.fill_(0x7F)
    a, weight, out, parent = _build("v_b", rows, seed)
    with validating(op):
        assert op(a, weight, out, _X8[group]) is out
    plain = torch.empty_like(parent)
    op(a, weight, plain[:rows, :, :256].transpose(0, 1), build("v_b", group, dev))
    torch.cuda.synchronize()
    assert torch.equal(parent[:rows], plain[:rows]), "the bf16 output differs from the plain cell"
    assert bool((parent[rows:] == SENTINEL).all()), "the kernel wrote past the last row"
    act = parent[:rows].reshape(rows, 4096).contiguous()
    want_q, want_sf = quant.reference(act)
    assert torch.equal(xq[:rows], want_q), f"{int((xq[:rows] != want_q).sum())} bytes differ"
    got = quant.dequantize(xq[:rows].contiguous(), sf, rows, 4096)
    assert torch.equal(got, quant.dequantize(want_q, want_sf, rows, 4096)), "scales differ"


@pytest.mark.parametrize("rows", (65, 96, 127, 144, 191))
def test_mtp_group_side_output_at_other_wide_row_counts(rows) -> None:
    """The MTP group's side-output V-B launches at every MTP3 / MTP5 decode row count above 64
    (a batch of fewer than 32 sequences): the bf16 output is the plain MTP cell's and the
    side operand the activation quant's bytes of it, as at 128 / 192."""
    _check_x8_cell(rows, seed=rows + 31)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    kernel, rows = cell.spec["kernel"], cell.spec["rows"]
    if cell.spec.get("x8"):
        _check_x8_cell(rows, zlib.crc32(cell.why.encode()) % 2**31)
        return
    if cell.spec.get("fp8"):
        seed = zlib.crc32(cell.why.encode()) % 2**31
        _check_fp8_cell(rows, seed, bool(cell.spec.get("decode")))
        return
    if cell.spec.get("prefill"):
        runner = _prefill(kernel)
    else:
        runner = build(kernel, _group(rows), torch.cuda.current_device())
    assert rows in runner.rows
    a, weight, out, parent = _build(kernel, rows, zlib.crc32(cell.why.encode()) % 2**31)
    with validating(op):
        assert op(a, weight, out, runner) is out, "the op writes `out` and returns it"
    _compare(out, op.reference(a, weight, out, runner))
    n = LAYOUTS[kernel].n
    # K-B writes the latent half of the fused query; its rope tail is not the
    # kernel's to touch.
    assert bool((parent[:rows, :, n:] == SENTINEL).all()), "the kernel wrote past N"
    assert bool((parent[rows:] == SENTINEL).all()), "the kernel wrote past the last row"


@pytest.mark.parametrize("kernel", ["k_b", "v_b"])
def test_a_runner_refuses_rows_outside_its_group(kernel) -> None:
    """M=1 is in neither group and the target sends it to `bmm_out`; a row
    count of the other group must raise rather than run the wrong schedule."""
    small = build(kernel, "m_le_8", torch.cuda.current_device())
    for rows in (1, 16):
        a, weight, out, _ = _build(kernel, rows, seed=rows)
        with pytest.raises(ValueError, match="rows"):
            op(a, weight, out, small)


@pytest.mark.parametrize("kernel", ["k_b", "v_b"])
def test_the_prefill_runner_refuses_rows_past_its_ceiling(kernel) -> None:
    """The target sends a row count above 8192 to `bmm_out`."""
    a, weight, out, _ = _build(kernel, PREFILL_MAX_ROWS + 1, seed=1)
    with pytest.raises(ValueError, match="T <="):
        op(a, weight, out, _prefill(kernel))


@pytest.mark.parametrize("kernel", ["k_b", "v_b"])
@pytest.mark.parametrize("group", MTP_GROUPS)
def test_mtp_group_replays_bitwise_in_a_cuda_graph(kernel, group) -> None:
    """The MTP3 / MTP5 cells run inside the decode CUDA graph: a replay writes the eager
    launch's bytes at each row count of the group, and a replay after new operands are
    copied into the captured buffers writes the eager launch's product on them."""
    n = LAYOUTS[kernel].n
    runner = build(kernel, group, torch.cuda.current_device())
    for rows in runner.rows:
        a, weight, out, parent = _build(kernel, rows, seed=rows)
        op(a, weight, out, runner)
        eager = parent.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op(a, weight, out, runner)
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(parent, eager), f"{rows} rows: the replay differs from eager"
        a2, weight2, _, _ = _build(kernel, rows, seed=rows + 1)
        a.copy_(a2)
        weight.copy_(weight2)
        again = torch.empty_like(parent)
        op(a, weight, again[:rows, :, :n].transpose(0, 1), runner)
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(parent[:rows, :, :n], again[:rows, :, :n]), (
            f"{rows} rows: the replay on new operands differs from eager"
        )
        assert not torch.equal(parent, eager), "the new operands did not change the product"
        _compare(out, op.reference(a, weight, out, runner))


#: MTP3 / MTP5 decode row counts without a cell: each bucket's edges, the 24-request batches
#: (96, 144) and partial batches at widths 4 / 6.
WIDE_ROWS = (65, 66, 68, 84, 96, 100, 126, 127, 129, 144, 150, 190, 191)


@pytest.mark.parametrize("kernel", ["k_b", "v_b"])
def test_mtp_group_at_other_wide_row_counts_is_its_cells_bitwise(kernel) -> None:
    """The MTP group's object launches at every MTP3 / MTP5 decode row count above 64
    (`MTP_RUNTIME_ROWS`; the target sends a batch of fewer than 32 sequences there): each
    row is bitwise what it writes at 192 rows with the input padded by other (random) rows,
    nothing is written past N or past the last row, within the band of the fp32 reference;
    a captured launch at 96 rows replays bitwise on new operands. 64 and 193 still raise."""
    n = LAYOUTS[kernel].n
    runner = build(kernel, MTP_GROUPS[0], torch.cuda.current_device())
    assert tuple(runner.rows) == (128, 192), "the cells stay 128 / 192"
    assert set(WIDE_ROWS) <= set(MTP_RUNTIME_ROWS) and min(MTP_RUNTIME_ROWS) == 65
    for rows in WIDE_ROWS:
        a, weight, out, parent = _build(kernel, rows, seed=rows)
        with validating(op):
            assert op(a, weight, out, runner) is out
        a_full, _, out_full, _ = _build(kernel, 192, seed=rows + 1000)
        a_full[:, :rows].copy_(a)
        op(a_full, weight, out_full, runner)
        torch.cuda.synchronize()
        assert torch.equal(out, out_full[:, :rows]), f"{rows} rows: not the 192-row launch's"
        assert bool((parent[:rows, :, n:] == SENTINEL).all()), "the kernel wrote past N"
        assert bool((parent[rows:] == SENTINEL).all()), "the kernel wrote past the last row"
        _compare(out, op.reference(a, weight, out, runner))
    rows = 96
    a, weight, out, parent = _build(kernel, rows, seed=7)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(a, weight, out, runner)
    a2, _, _, _ = _build(kernel, rows, seed=8)
    a.copy_(a2)
    again = torch.full_like(parent, SENTINEL)
    op(a, weight, again[:rows, :, :n].transpose(0, 1), runner)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(parent, again), "the replay on new operands differs from eager"
    for rows in (64, 193):
        a, weight, out, _ = _build(kernel, rows, seed=rows)
        with pytest.raises(ValueError, match="rows"):
            op(a, weight, out, runner)


def test_mtp_group_fp8_q_base_at_other_wide_row_counts() -> None:
    """The MTP group's fp8 q-base K-B launches at every MTP3 / MTP5 decode row count above
    64, as its plain cells do: at each its e4m3 bytes are the plain MTP cell's bf16 output
    converted (nearest, saturating), nothing is written past N or past the last row, and a
    captured launch at 96 rows replays the eager launch's bytes on new operands. 64 and 193
    still raise."""
    lay = LAYOUTS["k_b"]
    dev = torch.cuda.current_device()
    runner = build_decode_fp8_q_base(MTP_GROUPS[0], dev)
    plain = build("k_b", MTP_GROUPS[0], dev)
    assert tuple(runner.rows) == (128, 192), "the cells stay 128 / 192"

    def quant_view(rows):
        quant = torch.full(
            (rows + CANARY_ROWS, lay.heads, lay.out_row), SENTINEL_BYTE, dtype=torch.uint8,
            device="cuda",
        )  # fmt: skip
        return quant, quant[:rows, :, : lay.n].transpose(0, 1)

    for rows in WIDE_ROWS:
        a, weight, out_bf16, _ = _build("k_b", rows, seed=rows)
        op(a, weight, out_bf16, plain)
        quant, out = quant_view(rows)
        with validating(op):
            assert op(a, weight, out, runner) is out
        torch.cuda.synchronize()
        want = out_bf16.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
        bad = int((out != want).sum())
        assert bad == 0, f"{rows} rows: {bad} e4m3 bytes differ from e4m3(bf16 MTP cell)"
        assert bool((quant[:rows, :, lay.n :] == SENTINEL_BYTE).all()), "the kernel wrote past N"
        assert bool((quant[rows:] == SENTINEL_BYTE).all()), "the kernel wrote past the last row"
    rows = 96
    a, weight, _, _ = _build("k_b", rows, seed=7)
    quant, out = quant_view(rows)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(a, weight, out, runner)
    a2, _, _, _ = _build("k_b", rows, seed=8)
    a.copy_(a2)
    again, again_out = quant_view(rows)
    op(a, weight, again_out, runner)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(quant, again), "the replay on new operands differs from eager"
    assert int(quant[:rows, :, : lay.n].count_nonzero()) > rows * lay.heads * lay.n // 2
    for rows in (64, 193):
        a, weight, _, _ = _build("k_b", rows, seed=rows)
        _, out = quant_view(rows)
        with pytest.raises(ValueError, match="rows"):
            op(a, weight, out, runner)


#: MTP5 at 2 / 3 requests (12 / 18 rows; MTP3 at 3 is 12), the group's lower edge (9) and two
#: more MTP5 widths (30, 42): decode row counts in 9..64 without a cell, which the target runs
#: on the M >= 16 group's object where `_cell_up` holds. 16 / 48 are the group's cells.
SMALL_ROWS = (9, 12, 18, 30, 42)


@pytest.mark.parametrize("kernel", ["k_b", "v_b"])
def test_decode_group_at_mtp_decode_row_counts_is_its_cells_bitwise(kernel) -> None:
    """The M >= 16 group's object launches at every row count in 9..64
    (`DECODE_RUNTIME_ROWS`): each row is bitwise what it writes at 64 rows with the input
    padded by other (random) rows, nothing is written past N or past the last row, within
    the band of the fp32 reference; a captured launch at 12 rows replays bitwise on new
    operands. 8 and 65 still raise; the cells stay 16 / 32 / 48 / 64."""
    n = LAYOUTS[kernel].n
    runner = build(kernel, "m_ge_16", torch.cuda.current_device())
    assert tuple(runner.rows) == (16, 32, 48, 64), "the cells stay 16 / 32 / 48 / 64"
    assert set(SMALL_ROWS) <= set(DECODE_RUNTIME_ROWS)
    assert (min(DECODE_RUNTIME_ROWS), max(DECODE_RUNTIME_ROWS)) == (9, 64)
    for rows in (*SMALL_ROWS, 16):
        a, weight, out, parent = _build(kernel, rows, seed=rows)
        with validating(op):
            assert op(a, weight, out, runner) is out
        a_full, _, out_full, _ = _build(kernel, 64, seed=rows + 1000)
        a_full[:, :rows].copy_(a)
        op(a_full, weight, out_full, runner)
        torch.cuda.synchronize()
        assert torch.equal(out, out_full[:, :rows]), f"{rows} rows: not the 64-row launch's"
        assert bool((parent[:rows, :, n:] == SENTINEL).all()), "the kernel wrote past N"
        assert bool((parent[rows:] == SENTINEL).all()), "the kernel wrote past the last row"
        _compare(out, op.reference(a, weight, out, runner))
    rows = 12
    a, weight, out, parent = _build(kernel, rows, seed=7)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(a, weight, out, runner)
    a2, _, _, _ = _build(kernel, rows, seed=8)
    a.copy_(a2)
    again = torch.full_like(parent, SENTINEL)
    op(a, weight, again[:rows, :, :n].transpose(0, 1), runner)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(parent, again), "the replay on new operands differs from eager"
    for rows in (8, 65):
        a, weight, out, _ = _build(kernel, rows, seed=rows)
        with pytest.raises(ValueError, match="rows"):
            op(a, weight, out, runner)


@pytest.mark.parametrize("rows", (*SMALL_ROWS, 16))
def test_decode_group_side_output_at_mtp_decode_row_counts(rows) -> None:
    """The M >= 16 group's side-output V-B at those counts: the bf16 output is the plain
    group object's and o_proj's MXF8 operand the activation quant's bytes of it (rows < M),
    so the target drops o_proj's own act-quant there as at the cells."""
    _check_x8_cell(rows, seed=rows + 77)


def test_decode_group_side_output_replays_in_a_cuda_graph() -> None:
    """Captured at 12 rows, a replay on new operands writes the eager launch's bf16 output
    and operand rows."""
    dev = torch.cuda.current_device()
    if 64 not in _X8:
        _X8[64] = _x8_operand(64)
    if "m_ge_16" not in _X8:
        _X8["m_ge_16"] = build_v_b_mxf8_side("m_ge_16", dev, _X8[64])
    runner = _X8["m_ge_16"]
    xq, _ = _X8[64]
    rows = 12
    a, weight, out, parent = _build("v_b", rows, seed=5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(a, weight, out, runner)
    a2, _, _, _ = _build("v_b", rows, seed=6)
    a.copy_(a2)
    op(a, weight, out, runner)
    torch.cuda.synchronize()
    eager, eager_q = parent.clone(), xq[:rows].clone()
    out.fill_(float("nan"))
    xq.fill_(0x7F)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(parent, eager) and torch.equal(xq[:rows], eager_q)


def test_decode_group_fp8_q_base_at_mtp_decode_row_counts() -> None:
    """The M >= 16 group's fp8 q-base K-B at those counts: its e4m3 bytes are the plain group
    object's bf16 output converted (nearest, saturating), nothing past N or the last row."""
    lay = LAYOUTS["k_b"]
    dev = torch.cuda.current_device()
    runner = build_decode_fp8_q_base("m_ge_16", dev)
    plain = build("k_b", "m_ge_16", dev)
    for rows in SMALL_ROWS:
        a, weight, out_bf16, _ = _build("k_b", rows, seed=rows)
        op(a, weight, out_bf16, plain)
        quant = torch.full(
            (rows + CANARY_ROWS, lay.heads, lay.out_row), SENTINEL_BYTE, dtype=torch.uint8,
            device="cuda",
        )  # fmt: skip
        out = quant[:rows, :, : lay.n].transpose(0, 1)
        with validating(op):
            assert op(a, weight, out, runner) is out
        torch.cuda.synchronize()
        want = out_bf16.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
        bad = int((out != want).sum())
        assert bad == 0, f"{rows} rows: {bad} e4m3 bytes differ from e4m3(bf16 K-B)"
        assert bool((quant[:rows, :, lay.n :] == SENTINEL_BYTE).all()), "the kernel wrote past N"
        assert bool((quant[rows:] == SENTINEL_BYTE).all()), "the kernel wrote past the last row"


@pytest.mark.parametrize("group", MTP_GROUPS)
def test_the_mtp_groups_side_output_needs_the_wide_operand(group) -> None:
    """The V-B MXF8 side output of the MTP group takes the wide o_proj's 192-row operand
    only, and a decode group takes the 64-row one only: either other operand refuses."""
    dev = torch.cuda.current_device()
    with pytest.raises(ValueError, match=r"\[192, 4096\]"):
        build_v_b_mxf8_side(group, dev, _x8_operand(64))
    with pytest.raises(ValueError, match=r"\[64, 4096\]"):
        build_v_b_mxf8_side("m_ge_16", dev, _x8_operand(192))


def test_guard_refuses_the_bmm_out_view_of_a_weight() -> None:
    """The prefill kernel caches its weight operand by address and checks it on
    first sight only; the `[16, K, N]` view `bmm_out` takes shares the address."""
    a, weight, out, _ = _build("v_b", 80, seed=2)
    unreachable = AbsorbedBmmRunner("v_b", "prefill", range(1, PREFILL_MAX_ROWS + 1), None)
    with pytest.raises(AssertionError, match="contiguous bf16"):
        with validating(op):
            op(a, weight.transpose(1, 2), out, unreachable)
