# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the mhc_gemm_sqrsum_fma entry.

The entry owns what is certified: `CELLS` is the list, `reference` (a float64
product built on the spot) and `compare` (a rigorous fp32 rounding-chain bound
plus a fidelity gate on `y`, torch's fp32 defaults on `r`) are the gate. This
file turns a cell's spec into tensors, runs every call into NaN-filled buffers
with guard rows, and keeps the checks a cell cannot carry: every tactic
bit-identical, CUDA-graph replay, M = 0, subnormal flushing, the wrong-variant
controls, the op's own rejections, the layout guard and the silent misreads it
exists for, the every-row-count prefix sweep with its one-row-shift control,
one compiled kernel per tile band, and capture at the engine's capture union.
"""

import zlib

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.gemm.mhc_gemm_sqrsum_fma import (  # noqa: E501
    mhc_gemm_sqrsum_fma as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "mhc_gemm_sqrsum_fma requires a CUDA device"

HC_MULT, HIDDEN = 4, 5120
K_HC = HC_MULT * HIDDEN  # 20480: the flattened [hc_mult, hidden] stream
N_HC = (2 + HC_MULT) * HC_MULT  # 24 mixes: 4 pre, 4 post, 16 comb
TILE_NS = (1, 2, 3, 4, 6, 8, 12, 24)
# The entry's fidelity gate, in whose units the controls report.
_FIDELITY = 2.0**-17


def _ref(x: torch.Tensor, w: torch.Tensor):
    """The entry's reference for x [M, K] and w [N, K]: (y64, r64, bound, l2)."""
    m, k = x.shape
    return op.reference(x, w, None, None, m, w.shape[0], k)


def _y_errors(y: torch.Tensor, x: torch.Tensor, w: torch.Tensor):
    """(max of err / bound, median of err / l2) for y against the entry's float64 reference."""
    y64, _, bound, l2 = _ref(x, w)
    err = (y.double() - y64).abs()
    return (err / bound).max().item(), (err / l2).median().item()


def _call(x: torch.Tensor, w: torch.Tensor, tile_n: int = 0, tile_m: int = 0):
    """Run the op into NaN-filled buffers with one guard row on each side.

    Checks both sides of the in-place contract: every element of y and r is
    written, the guard rows around them are left bit-for-bit untouched, and the
    op returns the buffers it filled.
    """
    m, k = x.shape
    n = w.shape[0]
    y_buf = torch.full((m + 2, n), float("nan"), device="cuda")
    r_buf = torch.full((m + 2,), float("nan"), device="cuda")
    y, r = y_buf[1 : m + 1], r_buf[1 : m + 1]
    result = op(x, w, y, r, m, n, k, tile_n, tile_m)
    assert result[0] is y and result[1] is r, "the op writes in place and returns (y, r)"
    torch.cuda.synchronize()
    assert y_buf[0].isnan().all() and y_buf[-1].isnan().all(), "y guard rows were written"
    assert r_buf[0].isnan() and r_buf[-1].isnan(), "r guard slots were written"
    assert not y.isnan().any(), "an element of y was not written"
    assert not r.isnan().any(), "an element of r was not written"
    return y, r


def _check(y: torch.Tensor, r: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> None:
    op.compare((y, r), _ref(x, w))


def _make(m: int, k: int, seed: int, flavor: str):
    """x bf16 [m, k] and w fp32 [N_HC, k].

    ``gauss``: iid normal operands.
    ``hc_like``: shaped like the real streams measured on this checkpoint --
    four correlated hc copies per token, one massive attention-sink row
    (~150x the others' norm), and heavy-tailed weights (a few entries ~100x
    the weight rms).
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    if flavor == "gauss":
        x = torch.randn(m, k, device="cuda", generator=g)
        w = torch.randn(N_HC, k, device="cuda", generator=g) * 0.05
    else:
        copies = k // HIDDEN if k % HIDDEN == 0 else 1
        base = torch.randn(m, 1, k // copies, device="cuda", generator=g)
        x = (base + 0.3 * torch.randn(m, copies, k // copies, device="cuda", generator=g)).reshape(
            m, k
        ) * 0.08
        x[0] *= 150.0
        w = torch.randn(N_HC, k, device="cuda", generator=g) * 0.05
        spikes = torch.rand(N_HC, k, device="cuda", generator=g) < 1e-3
        w = torch.where(spikes, w * 100.0, w)
    return x.to(torch.bfloat16), w.contiguous()


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    m, k = spec["M"], spec["K"]
    x, w = _make(m, k, seed=zlib.crc32(cell.why.encode()) % 2**31, flavor=spec["flavor"])
    y, r = _call(x, w, spec["tile_n"], spec["tile_m"])
    op.compare((y, r), op.reference(x, w, y, r, m, N_HC, k, spec["tile_n"], spec["tile_m"]))


@pytest.mark.parametrize("m", (1, 8, 32, 48, 128, 129, 192, 2048, 2049, 4096))
def test_every_tactic_is_bit_identical(m: int) -> None:
    """tile_n only re-tiles the grid and tile_m is not read: results never change."""
    x, w = _make(m, K_HC, seed=100 + m, flavor="hc_like")
    y0, r0 = _call(x, w)
    _check(y0, r0, x, w)
    for tile_n in TILE_NS:
        for tile_m in (0, 1, 2):
            y, r = _call(x, w, tile_n, tile_m)
            assert torch.equal(y, y0) and torch.equal(r, r0), (tile_n, tile_m)


def test_cuda_graph_replay() -> None:
    """Captured once on static buffers, replayed on new stream contents: bit-identical to eager."""
    m = 48
    x_static, w = _make(m, K_HC, seed=7, flavor="hc_like")
    y = torch.empty(m, N_HC, device="cuda")
    r = torch.empty(m, device="cuda")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(x_static, w, y, r, m, N_HC, K_HC, 1, 0)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(x_static, w, y, r, m, N_HC, K_HC, 1, 0)
    x_new, _ = _make(m, K_HC, seed=8, flavor="hc_like")
    x_static.copy_(x_new)
    graph.replay()
    torch.cuda.synchronize()
    y_eager, r_eager = _call(x_new, w, 1, 0)
    assert torch.equal(y, y_eager) and torch.equal(r, r_eager)


def test_zero_tokens_is_a_no_op() -> None:
    x, w = _make(1, K_HC, seed=9, flavor="gauss")
    y = torch.empty(0, N_HC, device="cuda")
    r = torch.empty(0, device="cuda")
    op(x[:0], w, y, r, 0, N_HC, K_HC)
    torch.cuda.synchronize()


def test_subnormal_inputs_are_flushed() -> None:
    """The kernel is built with fast math: a subnormal x element contributes 0."""
    x = torch.zeros(1, 2048, device="cuda", dtype=torch.bfloat16)
    x[0, 0] = 1e-39  # subnormal in bf16 and fp32
    w = torch.zeros(N_HC, 2048, device="cuda")
    w[:, 0] = 1e10  # the exact product, ~1e-29, is a normal fp32 value
    y, r = _call(x, w)
    assert (y == 0).all() and (r == 0).all()


# ─── deliberately-wrong controls: each must land outside the gate ───────────


def _tf32_round(w: torch.Tensor) -> torch.Tensor:
    """Round fp32 to the 10-bit TF32 mantissa (round half away from zero)."""
    bits = w.view(torch.int32)
    return ((bits + 0x1000) & ~0x1FFF).view(torch.float32)


@pytest.mark.parametrize("m", (1, 32, 4096))
def test_wrong_controls_are_rejected(m: int) -> None:
    x, w = _make(m, K_HC, seed=200 + m, flavor="hc_like")
    y, r = _call(x, w)
    good_bound, good = _y_errors(y, x, w)

    # TF32-rounded weights -- the fidelity that disqualified the tensor-core path.
    y_tf32, _ = _call(x, _tf32_round(w.clone()))
    _, tf32 = _y_errors(y_tf32, x, w)
    # bf16-rounded weights.
    y_bf16, _ = _call(x, w.bfloat16().float())
    _, bf16 = _y_errors(y_bf16, x, w)
    # The stream flattened hidden-major instead of copy-major -- a caller layout bug.
    x_perm = x.view(m, HC_MULT, HIDDEN).transpose(1, 2).contiguous().view(m, K_HC)
    y_perm, _ = _call(x_perm, w)
    perm_bound, perm = _y_errors(y_perm, x, w)
    print(
        f"\nM={m}: median err/||x*w||_2 in units of the gate: correct {good / _FIDELITY:.3g} "
        f"(rigorous-bound ratio {good_bound:.3g}), "
        f"TF32 w {tf32 / _FIDELITY:.3g}, bf16 w {bf16 / _FIDELITY:.3g}, "
        f"permuted stream {perm / _FIDELITY:.3g} (rigorous-bound ratio {perm_bound:.3g})"
    )
    assert good <= _FIDELITY
    assert tf32 > 4 * _FIDELITY, f"TF32 control only {tf32 / _FIDELITY:.3g}x the gate"
    assert bf16 > 4 * _FIDELITY
    assert perm > 4 * _FIDELITY and perm_bound > 1.0

    # sum(x^2) accumulated from bf16-rounded squares.
    r_bf16 = (x.float() * x.float()).bfloat16().float().sum(-1)
    ref = _ref(x, w)
    with pytest.raises(AssertionError):
        op.compare((y, r_bf16), ref)  # y is the correct one: only r can fail
    op.compare((y, r), ref)


# ─── rejections, and the silent failures the guard's asserts exist for ──────


def _buffers(m: int, k: int):
    x, w = _make(m, k, seed=11, flavor="gauss")
    return x, w, torch.empty(m, N_HC, device="cuda"), torch.empty(m, device="cuda")


def test_tile_n_must_divide_n() -> None:
    x, w, y, r = _buffers(4, 2048)
    for tile_n in (5, 16):
        with pytest.raises(RuntimeError, match=f"not divisible by tile_n={tile_n}"):
            op(x, w, y, r, 4, N_HC, 2048, tile_n)


def test_dtypes_are_checked_by_the_op() -> None:
    x, w, y, r = _buffers(4, 2048)
    with pytest.raises(RuntimeError, match="expected scalar type BFloat16 but found Float"):
        op(x.float(), w, y, r, 4, N_HC, 2048)
    with pytest.raises(RuntimeError, match="expected scalar type Float but found BFloat16"):
        op(x, w.bfloat16(), y, r, 4, N_HC, 2048)
    with pytest.raises(RuntimeError, match="expected scalar type Float but found BFloat16"):
        op(x, w, y.bfloat16(), r, 4, N_HC, 2048)


def test_raw_op_silently_misreads_strides_and_sizes() -> None:
    """Why `is_valid` asserts layout and sizes: the op itself accepts these and returns wrong numbers."""
    m = 8
    big, w = _make(m, K_HC + 1024, seed=12, flavor="gauss")
    w = w[:, :K_HC].contiguous()
    x_strided = big[:, :K_HC]
    y = torch.empty(m, N_HC, device="cuda")
    r = torch.empty(m, device="cuda")
    op.raw_call(x_strided, w, y, r, m, N_HC, K_HC, 0, 0)
    torch.cuda.synchronize()
    bound, _ = _y_errors(y, x_strided, w)
    assert bound > 1.0, (
        "a row-strided x was read correctly; the contiguity assert may be unnecessary"
    )

    x = x_strided.contiguous()
    op.raw_call(x, w, y, r, m, N_HC, K_HC - 1024, 0, 0)
    torch.cuda.synchronize()
    bound, _ = _y_errors(y, x, w)
    assert bound > 1.0, "a K smaller than x's width was read correctly"


def test_guard_rejects_layouts_the_op_misreads() -> None:
    m, k = 8, 2048
    x, w, y, r = _buffers(m, k)
    wide = torch.empty(m, k + 64, device="cuda", dtype=torch.bfloat16)
    cases = (
        ("x must be", (wide[:, :k], w, y, r, m, N_HC, k)),
        ("w must be", (x, w.t().contiguous().t(), y, r, m, N_HC, k)),
        ("y must be", (x, w, torch.empty(m, 2 * N_HC, device="cuda")[:, :N_HC], r, m, N_HC, k)),
        ("r must be", (x, w, y, torch.empty(2 * m, device="cuda")[::2], m, N_HC, k)),
        ("x must be", (x, w, y, r, m, N_HC, k - 1024)),
        ("x must be", (x, w, y, r, m + 4, N_HC, k)),
    )
    for match, args in cases:
        with pytest.raises(AssertionError, match=match):
            with validating(op):
                op(*args)


# ─── every row count 1..32768 in the target's tile bands, and the capture union ──
#
# DeepSeek-V4.1's core target calls this op twice per layer with M = the rows
# of the pass: the tokens the rank holds this step (any integer up to the
# engine's max_num_tokens, 16384 at most in the envelope) or, in layers 21-39
# of a step that runs the decoder bounded-replay pass, that pass's rows (the
# finishing requests' 128-row tails plus the generation rows, up to 32768).
# The DSpark drafter calls it at N = 5G rows (G draft requests, up to 256). It
# picks tile_n by M: 1 for M <= 128, 6 for 129..2048 and 24 above (`_tile_n`
# in the target), so the tile-24 band reaches 32768. M only sizes grid
# axis 0, where block (m, j) reads row m of x and writes row m of y and r.
# tile_n selects one of three compiled kernels. The tests below check both
# facts on the GPU, at every M from 1 to 32768, each M run with the tile its
# band gets; the entry's cells hold every band's edges and every dumped M
# against the float64 reference.

_N_MAX = 32768
_BANDS = ((1, 128, 1), (129, 2048, 6), (2049, _N_MAX, 24))  # (first M, last M, tile_n)
# The band points and runtime-dump row counts the entry certifies at the target's tile.
_CELL_BAND_M = tuple(sorted({c.spec["M"] for c in op.CELLS if c.spec["tile_n"] > 0}))
# The capture union per rank. G ranges over every CUDA-graph batch size an
# envelope engine captures: every G 1..32, plus TRT-LLM's padded default list
# for cuda_graph_config {enable_padding: true, max_batch_size: 256}
# (40..128 step 8, 192, 256); the padding-off list at 256 (1..32, 64, 128, 256)
# is a subset. A captured step is generation-only, so the trunk runs at G (a
# decode step) and 6G (the DSpark target-verify step, 1 + 5 draft tokens per
# request), and the drafter at 5G. The largest is 1536, so a captured call
# only ever takes tile 1 or 6.
_GSET = tuple(range(1, 33)) + tuple(range(40, 129, 8)) + (192, 256)
_CAPTURE_M = tuple(sorted({k * g for g in _GSET for k in (1, 5, 6)}))
assert len(_GSET) == 46 and len(_CAPTURE_M) == 114 and max(_CAPTURE_M) == 1536


def _target_tile(m: int) -> int:
    for lo, hi, tile in _BANDS:
        if lo <= m <= hi:
            return tile
    raise ValueError(m)


def test_every_row_count_is_the_same_per_row_function() -> None:
    """Every prefix x[:m], m = 1..32768, at its band's tile_n, returns exactly the
    first m rows of the 32768-row result, which is held to the float64 reference.

    The output buffers are NaN-filled before every call, and the row after the
    last one is a guard row that must stay NaN. So a row the kernel skipped, or
    a write past row m - 1, shows up as a mismatch.
    """
    x, w = _make(_N_MAX, K_HC, seed=4242, flavor="hc_like")
    full_y, full_r = _call(x, w, 24)
    _check(full_y, full_r, x, w)
    for tile in (1, 6):  # the 32768-row result is the same bits at every band's tile
        y_t, r_t = _call(x, w, tile)
        assert torch.equal(y_t, full_y) and torch.equal(r_t, full_r), tile
    fy, fr = full_y.view(torch.int32), full_r.view(torch.int32)
    y_buf = torch.empty(_N_MAX + 1, N_HC, device="cuda")
    r_buf = torch.empty(_N_MAX + 1, device="cuda")
    mismatch = torch.zeros(_N_MAX + 1, dtype=torch.int64, device="cuda")
    guard = torch.zeros(_N_MAX + 1, dtype=torch.int64, device="cuda")
    for m in range(1, _N_MAX + 1):
        y_buf[: m + 1].fill_(float("nan"))
        r_buf[: m + 1].fill_(float("nan"))
        op(x[:m], w, y_buf[:m], r_buf[:m], m, N_HC, K_HC, _target_tile(m))
        mismatch[m] = (y_buf[:m].view(torch.int32) != fy[:m]).sum() + (
            r_buf[:m].view(torch.int32) != fr[:m]
        ).sum()
        guard[m] = (~y_buf[m].isnan()).sum() + (~r_buf[m].isnan()).long()
    bad = mismatch.nonzero().flatten().tolist()
    over = guard.nonzero().flatten().tolist()
    assert not bad, (
        f"{len(bad)} row counts differ from the {_N_MAX}-row result's rows, first {bad[:10]}"
    )
    assert not over, f"{len(over)} row counts wrote the guard row, first {over[:10]}"


def test_row_sweep_harness_sees_a_one_row_shift() -> None:
    """Control for the sweep: its bitwise per-row check rejects a read one row off, at every band."""
    x, w = _make(_N_MAX + 1, K_HC, seed=4243, flavor="gauss")
    full_y, full_r = _call(x[:_N_MAX], w, 24)
    for m in (1, 2, 127, 128, 129, 2048, 2049, 8192, 16384, _N_MAX):
        y, r = _call(x[1 : m + 1], w, _target_tile(m))
        n_bad = int((y.view(torch.int32) != full_y.view(torch.int32)[:m]).sum().item())
        n_bad_r = int((r.view(torch.int32) != full_r.view(torch.int32)[:m]).sum().item())
        print(
            f"[control row shift] M={m}: {n_bad} / {m * N_HC} y and {n_bad_r} / {m} r elements differ"
        )
        assert n_bad > m * N_HC // 2 and n_bad_r > m // 2


def _kernels_launched(fn) -> list:
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]


_PROBE_M = tuple(
    sorted(
        set(_CELL_BAND_M)
        | set(_CAPTURE_M)
        | {2, 3, 7, 130, 2050, 4096, 8191, 8193, 12288, 16383, 16385, 24576, _N_MAX - 1}
    )
)


def test_one_compiled_kernel_per_tile_band() -> None:
    """Probe: within a band, every M launches the same single kernel; the three bands launch three.

    The op is compiled ahead of time (no JIT). This shows that M reaches the
    launch only as the grid size, and the target's tile_n is the only kernel
    choice.
    """
    x, w = _make(_N_MAX, K_HC, seed=4244, flavor="gauss")
    y = torch.empty(_N_MAX, N_HC, device="cuda")
    r = torch.empty(_N_MAX, device="cuda")
    per_band = {}
    for m in _PROBE_M:
        names = _kernels_launched(
            lambda m=m: op(x[:m], w, y[:m], r[:m], m, N_HC, K_HC, _target_tile(m))
        )
        assert len(names) == 1, (m, names)
        per_band.setdefault(_target_tile(m), set()).add(names[0])
    print(f"{len(_PROBE_M)} row counts probed")
    for tile, names in sorted(per_band.items()):
        print(f"tile_n={tile}: {sorted(names)}")
        assert len(names) == 1, (tile, names)
        assert f"mhcGemmSqrsumFmaKernel<{tile}>" in next(iter(names)), names
    assert len(per_band) == 3


@pytest.mark.parametrize("m", _CAPTURE_M)
def test_capture_union_replay_is_bitwise_eager(m: int) -> None:
    """Captured at M (the target's tile) after an eager call at M, replayed on new
    data: bitwise equal to an eager call on that data.

    The first replay, on the capture-time data, must differ from the second, on
    new data, so a replay that reused stale inputs would fail.
    """
    tile = _target_tile(m)
    x_static, w = _make(m, K_HC, seed=6000 + m, flavor="hc_like")
    y = torch.full((m, N_HC), float("nan"), device="cuda")
    r = torch.full((m,), float("nan"), device="cuda")
    op(x_static, w, y, r, m, N_HC, K_HC, tile, 0)  # the eager call at this shape
    torch.cuda.synchronize()
    y_eager0, r_eager0 = y.clone(), r.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(x_static, w, y, r, m, N_HC, K_HC, tile, 0)
    y.fill_(float("nan"))
    r.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(y, y_eager0) and torch.equal(r, r_eager0), m
    y_old = y.clone()
    x_new, _ = _make(m, K_HC, seed=7000 + m, flavor="hc_like")
    x_static.copy_(x_new)
    y.fill_(float("nan"))
    r.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    y_eager, r_eager = _call(x_new, w, tile)
    assert torch.equal(y, y_eager) and torch.equal(r, r_eager), m
    assert not torch.equal(y, y_old), "the replay did not read the new stream"
    _check(y, r, x_new, w)
