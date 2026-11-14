# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SWIGLU_MXFP8_ROWS launch geometry (ops/swiglu_rows_plan.py), CPU only, no torch.

The rows form of the family's minimax_m31_swiglu_oai_mxfp8 op keeps k_rows rows in flight per CTA on an
occupancy-true grid.
Its exactness rests on the row-to-CTA map: every real row is computed once by one CTA (all its lanes on the same row,
so the 4-lane scale shuffle sees one row), every pad row's scales are zeroed once, and a CTA visits the one-row
kernel's rows in the one-row kernel's order. A lane-level simulation of both launches (the per-row function and the
swizzled 128x4 scale layout) checks that they write the same bytes, each once. The plan's constants and formulas are
checked against the C++ launcher's source. The module is loaded from its file so the test also runs without torch.
"""

import importlib.util
import pathlib
import re

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[7]
_FAMILY = _REPO / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3"
_CU = _FAMILY / "csrc/m31/swigluOaiMxfp8Sm103.cu"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


plan = _load("swiglu_rows_plan", _FAMILY / "ops/swiglu_rows_plan.py")
cbnq = _load(
    "ctx_boundary_plan", _FAMILY / "minimax_m3_1_flash_nvfp4__sm_107__dep2ctx/ctx_boundary_plan.py"
)

pytestmark = pytest.mark.cpu_only

D = 3072  # the shared expert's intermediate size per rank
SMS = 212  # sm_107

ROWS = [1, 127, 128, 129, 1000, 1024, 1160, 8192, 16384]


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("k_rows", [1, 2, 4, 8])
def test_rows_batches_cover_every_row_once(rows, k_rows):
    padded = plan.padded_rows(rows)
    for g in sorted({1, 7, 424, plan.grid(rows, 2, SMS)}):
        seen = [r for c in range(g) for (_, _, r) in plan.cta_rows(c, g, rows, k_rows)]
        assert sorted(seen) == list(range(rows))
        pads = [r for c in range(g) for r in plan.cta_pad_rows(c, g, rows)]
        assert sorted(pads) == list(range(rows, padded))


@pytest.mark.parametrize("rows", [1, 129, 1160, 8192])
@pytest.mark.parametrize("k_rows", plan.K_ROWS_CHOICES)
def test_each_cta_keeps_the_one_row_kernels_rows_and_order(rows, k_rows):
    """Batching only groups a CTA's rows: at the same grid the CTA computes the one-row kernel's rows in its order,
    and each batch is loaded whole (at most k_rows rows, consecutive rows ``grid`` apart) before its first row runs."""
    g = plan.grid(rows, 2, SMS)
    for c in range(g):
        computed = [r for (_, _, r) in plan.cta_rows(c, g, rows, k_rows)]
        assert computed == plan.one_row_cta_rows(c, g, rows)
        batches = plan.cta_batches(c, g, rows, k_rows)
        assert [r for b in batches for r in b] == computed
        for b in batches:
            assert 1 <= len(b) <= k_rows
            assert all(b2 - b1 == g for b1, b2 in zip(b, b[1:]))


def test_grid_uses_resident_ctas():
    block = plan.threads(D)
    assert block == 384
    assert plan.ctas_per_sm(384, 1024) == 2  # sm_107: 1024 threads per SM
    assert plan.ctas_per_sm(block) == 2 and plan.SM107_THREADS_PER_SM == 1024
    assert plan.ONE_ROW_THREADS_PER_SM // block == 5  # what the one-row grid assumes
    assert plan.one_row_grid(8192, D, SMS) == 5 * SMS  # 1060 CTAs, of which 424 can be resident
    for rows in ROWS:
        assert plan.grid(rows, 2, SMS) <= 2 * SMS
    assert plan.grid(8192, 2, SMS) == 424
    assert plan.grid(1024, 2, SMS) == 424  # every resident slot, not ceil(1024 / 4) = 256
    assert plan.grid(1024, 1, SMS) == SMS  # k8 at 1 CTA/SM: every SM, not 128 CTAs
    assert plan.grid(1, 2, SMS) == 128  # pad128(1) (the rows form engages from 1024 rows only)
    # Registers: 80 per thread fit two 384-thread CTAs; 85 (88 allocated) fit one.
    assert plan.ctas_per_sm(384, 1024, regs_per_thread=80) == 2
    assert plan.ctas_per_sm(384, 1024, regs_per_thread=85) == 1
    assert plan.ctas_per_sm(1024, 1024, regs_per_thread=64) == 1
    assert (
        plan.ctas_per_sm(1024, 1024, regs_per_thread=72) == 0
    )  # no resident CTA: the one-row launch is used
    # Loads in flight per SM: 24 KiB at one row per CTA, 96 KiB at k_rows 4.
    assert plan.loads_in_flight_bytes(2, block, 1) == 24576
    assert plan.loads_in_flight_bytes(2, block, 4) == 98304


@pytest.mark.parametrize("rows", [1024, 1160, 2048, 4096, 8192, 16384])
@pytest.mark.parametrize("ctas", [1, 2])
@pytest.mark.parametrize("k_rows", plan.K_ROWS_CHOICES)
def test_grid_fills_every_sm_from_the_engage_bound(rows, ctas, k_rows):
    """From :data:`ROWS_MIN` rows the rows form uses every resident slot of every SM (no idle SM at 1024 - 4095
    rows, the captured-bucket range), and the CTAs' row counts differ by at most one."""
    g = plan.grid(rows, ctas, SMS)
    assert g == ctas * SMS
    counts = [len(plan.cta_rows(c, g, rows, k_rows)) for c in range(g)]
    assert min(counts) >= 1 and max(counts) - min(counts) <= 1
    assert sum(counts) == rows


def test_engage_rule():
    assert plan.uses_rows_form(1024, False, 2)
    assert not plan.uses_rows_form(1023, False, 2)
    assert not plan.uses_rows_form(8192, True, 2)
    assert not plan.uses_rows_form(8192, False, 0)


def test_bytes_and_price():
    x = 8192 * 2 * D * 2
    q = 8192 * D
    sf = 8192 * D // 32
    h = 8192 * D * 2
    assert plan.bytes_moved(8192, D, True) == x + q + sf + h == 176_947_200
    assert plan.bytes_moved(8192, D, False) == x + q + sf == 126_615_552
    assert (
        plan.bytes_moved(1, D, False) == 2 * D * 2 + D + 128 * D // 32
    )  # the pad scales are written too
    assert plan.ms_per_1k(31, 57, 8192) == pytest.approx(0.2157, abs=1e-4)
    assert plan.ms_per_1k(21.7, 57, 8192) == pytest.approx(0.151, abs=5e-4)
    assert plan.ms_per_1k(1.0, 57, 8192) == cbnq.ms_per_1k(1.0, 57, 8192)


def test_switch_parsing():
    for v in ("1", "true", "TRUE", "On", "yes", " 1", "true\n", "\tYES "):
        assert plan.switch_set(v) and not plan.switch_warns(v)
    for v in (None, "", "0", "false", "off", "no", " 0 ", "OFF"):
        assert not plan.switch_set(v) and not plan.switch_warns(v)
    for v in ("2", "enable", "disable", "1 1"):  # unknown: the rows form stays on, with a warning
        assert not plan.switch_set(v) and plan.switch_warns(v)
    assert plan.k_rows_from_env(None) == plan.K_ROWS_DEFAULT == 4
    assert [plan.k_rows_from_env(v) for v in ("2", "4", "8", " 8", "2\n")] == [2, 4, 8, 8, 2]
    for v in ("", "1", "3", "16", "08", "4x"):
        assert plan.k_rows_from_env(v) == plan.K_ROWS_DEFAULT


# --- lane-level simulation of both launches -------------------------------------------------------------------------


def _row_value(row, col):
    """Stands in for the per-row arithmetic (swigluOaiBf16x2 + cvt_warp_fp16_to_mxfp8): any pure function of the row's
    inputs, i.e. of (row, column)."""
    return (row * 1_000_003 + col * 7919) & 0xFF


def _row_scale(row, k):
    return 1 + (row * 31 + k) % 170  # never 0 (a pad scale) or 0xAB (the fill)


def _simulate(rows, d, launch):
    """Runs a launch's (cta, row) visits lane by lane and returns (q, h, sf) as byte arrays plus per-byte write counts.
    Every lane of the CTA handles the same row; lane colIdx % 4 == 0 stores the K32 scale its 4-lane group shares."""
    padded = plan.padded_rows(rows)
    lanes = plan.threads(d)
    q = [None] * (rows * d)
    h = [None] * (rows * d)
    sf = [0xAB] * (padded * d // 32)  # pad scales must be rewritten to 0
    writes = [0] * (len(q) + len(h) + len(sf))
    for row, is_pad in launch:
        for lane in range(lanes):
            if lane % 4 == 0:
                off = cbnq.served_sf_offset(row, lane // 4, rows, d)
                sf[off] = 0 if is_pad else _row_scale(row, lane // 4)
                writes[2 * rows * d + off] += 1
            if is_pad:
                continue
            for e in range(8):
                i = row * d + lane * 8 + e
                q[i] = _row_value(row, lane * 8 + e)
                h[i] = q[i] ^ 0x5A
                writes[i] += 1
                writes[rows * d + i] += 1
    return q, h, sf, writes


def _one_row_launch(rows, d, g):
    for c in range(g):
        for r in plan.one_row_cta_rows(c, g, rows):
            yield r, False
        for r in plan.cta_pad_rows(c, g, rows):
            yield r, True


def _rows_launch(rows, d, g, k_rows):
    for c in range(g):
        for _, _, r in plan.cta_rows(c, g, rows, k_rows):
            yield r, False
        for r in plan.cta_pad_rows(c, g, rows):
            yield r, True


@pytest.mark.parametrize("rows", [1, 127, 1031, 1160, 2049])
@pytest.mark.parametrize("k_rows", plan.K_ROWS_CHOICES)
def test_simulated_launches_write_the_same_bytes_once(rows, k_rows):
    d = 256  # 32 lanes, 8 scales per row, 2 K tiles: the swizzle's k-tile term is exercised
    sms = 5
    ref = _simulate(rows, d, _one_row_launch(rows, d, plan.one_row_grid(rows, d, sms)))
    got = _simulate(rows, d, _rows_launch(rows, d, plan.grid(rows, 2, sms), k_rows))
    assert got[:3] == ref[:3]
    assert set(got[3]) == {1} and set(ref[3]) == {
        1
    }  # every q / h / scale byte written exactly once
    assert None not in got[0] and 0xAB not in got[2]


# --- the C++ launcher matches the plan ------------------------------------------------------------------------------


def _cu():
    return _CU.read_text()


def _body(src, start, end):
    i = src.index(start)
    return src[i : src.index(end, i)]


def test_launcher_constants_match_the_plan():
    """The launcher engages from the plan's ROWS_MIN; kRows is an op argument whose default is the plan's default
    kRows; the native code reads no environment, and its kRows dispatch covers the plan's choices with the default as
    the fallback."""
    src = _cu()
    assert f"constexpr int64_t kSwigluRowsMinRows = {plan.ROWS_MIN};" in src
    op = (_CU.parent / "swigluOaiMxfp8Op.cpp").read_text()
    assert f"int k_rows={plan.K_ROWS_DEFAULT}) -> ()" in op
    assert "getenv" not in src
    dispatch = _body(src, "cudaError_t dispatchSwigluRows(int kRows", "} // namespace")
    assert set(re.findall(r"case (\d):", dispatch)) == {"2", "8"} and "default:" in dispatch
    assert "launchSwigluRows<kWriteH, 4>" in dispatch.split("default:")[1]
    assert {2, 8, plan.K_ROWS_DEFAULT} == set(plan.K_ROWS_CHOICES)


def test_launcher_grid_and_engage_rule_match_the_plan():
    src = _cu()
    rows_launch = _body(
        src, "cudaError_t launchSwigluRows(cudaLaunchConfig_t cfg", "*taken = true;"
    )
    # No resident CTA, or a failed occupancy query (error cleared), leaves the launch to the one-row form, logged.
    no_cta = rows_launch[rows_launch.index("if (ctas < 1)") : rows_launch.index("cfg.gridDim =")]
    assert "return cudaSuccess;" in no_cta  # before the grid is set or anything launches
    assert "rows form NOT ENGAGED at d %d (kRows=%d, h %s): no resident CTA" in rows_launch
    failed = rows_launch[
        rows_launch.index("if (e != cudaSuccess)") : rows_launch.index("if (ctas < 1)")
    ]
    assert "static_cast<void>(cudaGetLastError());" in failed and "return cudaSuccess;" in failed
    assert "rows form NOT ENGAGED at d %d (kRows=%d): occupancy query" in failed
    assert (
        "std::max<int64_t>(1, std::min<int64_t>(paddedRows, int64_t{multiProcessorCount} * ctas))"
        in rows_launch
    )
    # The engagement line names kRows and d, once per (device, kernel, block); the fallback line does not match it.
    assert '"invokeSwigluOaiMxfp8: SWIGLU_MXFP8 rows form kRows=%d at d %d (h %s)' in rows_launch
    assert rows_launch.count("if (fresh)") == 2
    assert src.count("rows form kRows=") == 1
    assert "cudaOccupancyMaxActiveBlocksPerMultiprocessor(ctas, kernel, block, 0)" in src
    invoke = _body(src, "cudaError_t invokeSwigluOaiMxfp8(", "} // namespace minimaxM31")
    assert "if (rowsForm && rows >= kSwigluRowsMinRows)" in invoke
    assert "dispatch(kRows, cfg," in invoke
    # Short inputs keep the one-row kernel on the 2048-threads-per-SM grid.
    tail = invoke[invoke.index("// The one-row launch") :]
    assert "int const blocksPerSm = std::max(1, 2048 / threads);" in tail
    assert "std::min<int64_t>(paddedRows, int64_t{multiProcessorCount} * blocksPerSm)" in tail
    assert "swigluOaiMxfp8Kernel<true>" in tail and "swigluOaiMxfp8Kernel<false>" in tail
    assert "RowsKernel" not in tail


def test_rows_kernel_runs_the_one_row_kernels_instructions():
    src = _cu()
    one = _body(src, "__global__ void swigluOaiMxfp8Kernel(", "\n}\n")
    rows = _body(src, "__global__ void swigluOaiMxfp8RowsKernel(", "\n}\n")
    for lane in "xyzw":
        assert f"h.{lane} = swigluOaiBf16x2(gate.{lane}, up.{lane});" in one
        assert f"h.{lane} = swigluOaiBf16x2(gate[i].{lane}, up[i].{lane});" in rows
    for line in (
        "int64_t const outIdx = static_cast<int64_t>(row) * colThreads + colIdx;",
        "*reinterpret_cast<uint4*>(&quantIn) = h;",
        "batchIdx, row, colIdx, numRows, d / 32, sfOut, QuantizationSFLayout::SWIZZLED);",
        "qOut[outIdx] = tensorrt_llm::kernels::cvt_warp_fp16_to_mxfp8<__nv_bfloat16, 32>(quantIn, sf);",
        "for (int padRow = rows + blockIdx.x; padRow < paddedRows; padRow += gridDim.x)",
        "tensorrt_llm::kernels::cvt_store_replicated_sf<1>(sf, 0x00);",
        "cudaGridDependencySynchronize();",
        "cudaTriggerProgrammaticLaunchCompletion();",
    ):
        assert line in one and line in rows, line
    # Loads of a batch come before its math; the row bound is CTA-uniform (base, i, gridDim only).
    assert rows.index("gate[i] = x[xRow + colIdx];") < rows.index("swigluOaiBf16x2(gate[i]")
    assert (
        "int const stride = gridDim.x;" in rows
    )  # the one-row kernel's stride, so CTA c keeps its rows
    assert "for (int base = blockIdx.x; base < rows; base += stride * kRows)" in rows
    assert rows.count("int const row = base + i * stride;") == 2
    load = rows[rows.index("uint4 up[kRows] = {};") : rows.index("gate[i] = x[xRow + colIdx];")]
    assert "if (row < rows)" in load  # no load past the last row
    assert "if (row >= rows)" in rows and "break;" in rows
