# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the moe_gelu entry (exact erf GELU, single-tile metadata).

The entry owns what is certified: `CELLS` is the list, `reference` and
`compare` are the gate. This file turns a cell's spec into tensors and keeps
the checks a cell cannot carry: bitwise agreement with `F.gelu` (what the
DeepSeek-V4.1 vision aligner computes), the tanh-approximation control, the
contiguity guard, the op's own rejections, and the aligner's call form --
every row count 1..2048 as a prefix of the 2048-row result, one kernel at
every row count, the stale-limit control, and CUDA-graph replay at the
aligner's graph sizes.
"""

import math
import zlib

import pytest
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.activation.moe_gelu import (  # noqa: E501
    moe_gelu as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "moe_gelu requires a CUDA device"

# The DeepSeek-V4.1 vision aligner applies GELU to w1's output, [n_llm_tokens, 5120].
WIDTH = 5120
# torch.testing.assert_close defaults for bf16.
BF16_RTOL, BF16_ATOL = 1.6e-2, 1e-5


def _single_tile(rows: int):
    """One tile holding every row: the dense-activation metadata."""
    limit = torch.tensor([rows], dtype=torch.int32, device="cuda")
    n_tiles = torch.tensor([1], dtype=torch.int32, device="cuda")
    return limit, n_tiles, rows


def _ref_erf(x: torch.Tensor) -> torch.Tensor:
    """The entry's reference over every row of `x` (fp64 erf GELU, rounded to x.dtype)."""
    return op.reference(x, *_single_tile(x.shape[0]))


def _ref_tanh_fp32(x: torch.Tensor) -> torch.Tensor:
    """The WRONG function: tanh-approximated GELU, in fp32, rounded to x.dtype."""
    xf = x.float()
    inner = math.sqrt(2.0 / math.pi) * (xf + 0.044715 * xf * xf * xf)
    return (0.5 * xf * (1.0 + torch.tanh(inner))).to(x.dtype)


def _outside_bf16_gate(a: torch.Tensor, b: torch.Tensor) -> int:
    """Elements where |a - b| exceeds assert_close's default bf16 gate (finite b only)."""
    af, bf = a.float(), b.float()
    fin = torch.isfinite(bf)
    bad = (af - bf).abs() > BF16_ATOL + BF16_RTOL * bf.abs()
    return int((bad & fin).sum().item())


def _all_bf16_bit_patterns() -> torch.Tensor:
    """Every one of the 65536 bf16 bit patterns (finite, +-inf, NaNs), zero-padded to [13, 5120]."""
    bits = torch.arange(-32768, 32768, dtype=torch.int32, device="cuda").to(torch.int16)
    vals = bits.view(torch.bfloat16)
    rows = math.ceil(vals.numel() / WIDTH)
    x = torch.zeros(rows * WIDTH, dtype=torch.bfloat16, device="cuda")
    x[: vals.numel()] = vals
    return x.view(rows, WIDTH)


def _randn(rows: int, sigma: float, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(rows, WIDTH, device="cuda", generator=g) * sigma).to(torch.bfloat16)


def _build(spec: dict, seed: int):
    """The cell's input and its single-tile metadata, `limit` rows computed of `rows`."""
    if spec["kind"] == "bits":
        x = _all_bf16_bit_patterns()
    else:
        x = _randn(spec["rows"], spec["sigma"], seed=seed)
    rows = x.shape[0]
    limit = torch.tensor([spec["limit"]], dtype=torch.int32, device="cuda")
    n_tiles = torch.tensor([1], dtype=torch.int32, device="cuda")
    return x, limit, n_tiles, rows


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    x, limit, n_tiles, rows = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    x_before = x.clone()
    tail = cell.spec["limit"] < rows
    if tail:
        # Park a NaN block of the output's exact size in the caching allocator so
        # the op's freshly allocated output reuses it; unwritten rows then stay NaN.
        sentinel = torch.full((rows, WIDTH), float("nan"), dtype=torch.bfloat16, device="cuda")
        ptr = sentinel.data_ptr()
        del sentinel
    out = op(x, limit, n_tiles, rows)
    if tail:
        assert out.data_ptr() == ptr, (
            "the output did not reuse the sentinel block; the probe is blind"
        )
    assert out.shape == x.shape and out.dtype == torch.bfloat16 and out.is_contiguous()
    assert out.data_ptr() != x.data_ptr()
    # Bitwise: the bit-pattern cell's input holds NaNs.
    assert torch.equal(x.view(torch.int16), x_before.view(torch.int16)), "input was mutated"
    op.compare(out, op.reference(x, limit, n_tiles, rows))
    # The vendor aligner calls F.gelu (approximate='none') on a bf16 tensor.
    computed = slice(0, cell.spec["limit"])
    vendor = F.gelu(x[computed], approximate="none")
    same = (out[computed].view(torch.int16) == vendor.view(torch.int16)) | (
        torch.isnan(out[computed]) & torch.isnan(vendor)
    )
    assert bool(same.all()), f"{int((~same).sum())} of {same.numel()} differ bitwise from F.gelu"


def test_tanh_approximation_control() -> None:
    """The tanh-approximated GELU is a different function, and this harness can see it."""
    x = _all_bf16_bit_patterns()
    out = op(x, *_single_tile(x.shape[0]))
    wrong = _ref_tanh_fp32(x)
    # The control is not blind: the wrong reference itself fails against fp64 erf GELU.
    n_ref = _outside_bf16_gate(wrong, _ref_erf(x))
    n_exh = _outside_bf16_gate(out, wrong)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(out, wrong, equal_nan=True)

    xr = _randn(1024, 1.0, seed=2024)
    outr = op(xr, *_single_tile(1024))
    wrong_r = _ref_tanh_fp32(xr)
    n_rand = _outside_bf16_gate(outr, wrong_r)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(outr, wrong_r)
    print(
        f"\ntanh control: outside bf16 gate -- tanh ref vs fp64 erf over all patterns {n_ref}/{x.numel()}; "
        f"op vs tanh over all patterns {n_exh}/{x.numel()}; "
        f"op vs tanh on N(0,1) [1024, {WIDTH}] {n_rand}/{xr.numel()}; "
        f"op vs fp64 erf on N(0,1) {_outside_bf16_gate(outr, _ref_erf(xr))}"
    )
    assert n_ref > 0 and n_exh > 0 and n_rand > 0


def test_non_contiguous_input_rejected_by_guard() -> None:
    big = _randn(64, 3.0, seed=5)
    big = torch.cat([big, big[:, :256]], dim=1)
    view = big[:, :WIDTH]
    assert not view.is_contiguous()
    with pytest.raises(AssertionError, match="must be contiguous"):
        with validating(op):
            op(view, *_single_tile(64))


def test_loud_rejections() -> None:
    x = _randn(64, 1.0, seed=3)
    one = torch.tensor([1], dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="must be equal to tile_tokens_dim"):
        op(x, torch.tensor([64], dtype=torch.int32, device="cuda"), one, 32)
    with pytest.raises(RuntimeError, match="tile_idx_to_mn_limit must be int32"):
        op(x, torch.tensor([64], dtype=torch.int64, device="cuda"), one, 64)
    with pytest.raises(RuntimeError, match="interm_size must be divisible by 8"):
        op(torch.zeros(4, WIDTH + 4, dtype=torch.bfloat16, device="cuda"), *_single_tile(4))
    with pytest.raises(RuntimeError, match="Unsupported input dtype"):
        op(torch.zeros(4, WIDTH, dtype=torch.float32, device="cuda"), *_single_tile(4))
    with pytest.raises(RuntimeError, match="input must be 2D"):
        op(torch.zeros(1, 4, WIDTH, dtype=torch.bfloat16, device="cuda"), *_single_tile(1))
    torch.cuda.synchronize()


# ─── The DeepSeek-V4.1 vision aligner's call form: rows 1..2048 ──────────────
#
# The multimodal prototype's aligner calls this op once per request on
# h = cells @ w1^T + b1, [C, 5120] bf16, with the dense single-tile metadata it
# builds host-side in `prepare`: tile_idx_to_mn_limit = torch.tensor([C],
# int32, cuda), num_non_exiting_tiles = torch.tensor([1], int32, cuda),
# tile_tokens_dim = C. C is at most 1902 over the 900 MMMU val documents (182,
# 546, 601 and 986 in the golden cases); N_max is 2048. The kernel walks rows
# in a grid-stride loop with grid = min(SMs x occupancy, rows), so the row
# count sizes the grid and bounds the loop.
_ALIGNER_N_MAX = 2048
_ALIGNER_GRAPH_ROWS = (182, 546, 601, 986, 1902)
# A fill kernel between calls splits the profiler's kernel stream per call.
_MARKER = torch.zeros(1, device="cuda")


def _prototype_metadata(rows: int):
    """The metadata exactly as VisionTower.prepare builds it."""
    limit = torch.tensor([rows], dtype=torch.int32, device="cuda")
    tiles = torch.tensor([1], dtype=torch.int32, device="cuda")
    return limit, tiles, rows


def test_aligner_every_row_count_is_the_same_per_row_function() -> None:
    """Every r in 1..2048 is bitwise the first r rows of the 2048-row call, which is held to fp64 erf GELU.

    Each output is filled with NaN before it is freed, so a row the kernel
    skipped cannot pick up correct bits from the previous call's block.
    """
    x = _randn(_ALIGNER_N_MAX, 3.0, seed=90)
    before = x.clone()
    full = op(x, *_prototype_metadata(_ALIGNER_N_MAX))
    op.compare(full, _ref_erf(x))
    assert torch.equal(full, F.gelu(x, approximate="none"))
    exact = float((full == _ref_erf(x)).float().mean())
    full_bits = full.view(torch.int16)
    mismatch = torch.zeros(_ALIGNER_N_MAX + 1, dtype=torch.bool, device="cuda")
    for r in range(1, _ALIGNER_N_MAX + 1):
        out = op(x[:r], *_prototype_metadata(r))
        mismatch[r] = (out.view(torch.int16) != full_bits[:r]).any()
        out.fill_(float("nan"))
        del out
    torch.cuda.synchronize()
    bad = mismatch.nonzero().flatten().tolist()
    assert not bad, (
        f"{len(bad)} row counts differ from the 2048-row result's rows, first {bad[:10]}"
    )
    assert torch.equal(x, before), "input was mutated"
    print(
        f"\n[aligner] 2048 rows: bit-exact {exact:.6f} vs fp64 erf, bitwise F.gelu; "
        f"r = 1..2048: every prefix bitwise equal"
    )


def test_aligner_one_kernel_at_every_row_count() -> None:
    """Probe: one launch of the same kernel at every row count 1..2048 (compiled ahead of time)."""
    x = _randn(_ALIGNER_N_MAX, 1.0, seed=91)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for r in range(1, _ALIGNER_N_MAX + 1):
            _MARKER.fill_(1.0)
            op(x[:r], *_prototype_metadata(r))
        torch.cuda.synchronize()
    evs = sorted(
        (e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA),
        key=lambda e: e.time_range.start,
    )
    per_call, cur = [], None
    for e in evs:
        if "FillFunctor" in e.name:
            if cur is not None:
                per_call.append(cur)
            cur = []
        elif cur is not None and not e.name.startswith(("Memcpy", "Memset")):
            cur.append(e.name)  # the metadata's host-to-device copies are not kernels
    per_call.append(cur)
    assert len(per_call) == _ALIGNER_N_MAX and all(len(c) == 1 for c in per_call)
    names = {c[0] for c in per_call}
    print(f"\n[aligner] rows 1..2048, one launch per call, kernels: {sorted(names)}")
    assert len(names) == 1, names


def test_aligner_controls() -> None:
    """The prefix check rejects a read one row off, and a stale limit (one row short) leaves the last row unwritten."""
    x = _randn(1024, 3.0, seed=92)
    full = op(x, *_prototype_metadata(1024))
    for r in (1, 2, 182, 1023):
        shifted = op(x[1 : r + 1].contiguous(), *_prototype_metadata(r))
        n_bad = int((shifted.view(torch.int16) != full[:r].view(torch.int16)).sum())
        print(f"\n[aligner control row shift] r={r}: {n_bad} / {r * WIDTH} elements differ")
        assert n_bad > r * WIDTH // 2
    rows = 601
    sentinel = torch.full((rows, WIDTH), float("nan"), dtype=torch.bfloat16, device="cuda")
    ptr = sentinel.data_ptr()
    del sentinel
    stale = torch.tensor([rows - 1], dtype=torch.int32, device="cuda")
    out = op(
        x[:rows].contiguous(), stale, torch.tensor([1], dtype=torch.int32, device="cuda"), rows
    )
    assert out.data_ptr() == ptr, (
        "the output did not reuse the sentinel block; the control is blind"
    )
    assert torch.equal(out[: rows - 1], full[: rows - 1]) and bool(torch.isnan(out[rows - 1]).all())
    print(
        f"[aligner control stale limit] limit {rows - 1} for {rows} rows: row {rows - 1} left unwritten (NaN)"
    )


def test_aligner_cuda_graph_at_the_prototype_sizes() -> None:
    """Captured at the prototype's graph sizes with its metadata tensors; replays on new data are bitwise eager."""
    pool = torch.cuda.graph_pool_handle()
    graphs = {}
    for rows in sorted(_ALIGNER_GRAPH_ROWS, reverse=True):
        x = _randn(rows, 3.0, seed=93)
        meta = _prototype_metadata(rows)
        op(x, *meta)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            out = op(x, *meta)
        graphs[rows] = (graph, x, meta, out)
    for rnd in range(2):
        for rows in _ALIGNER_GRAPH_ROWS:
            graph, x, meta, out = graphs[rows]
            x.copy_(_randn(rows, 3.0, seed=94 + 10 * rnd + rows))
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, op(x, *meta)), rows
            assert torch.equal(out, F.gelu(x, approximate="none")), rows
    print(
        f"\n[aligner] captured at C = {_ALIGNER_GRAPH_ROWS}: replays bitwise equal to eager and to F.gelu"
    )
