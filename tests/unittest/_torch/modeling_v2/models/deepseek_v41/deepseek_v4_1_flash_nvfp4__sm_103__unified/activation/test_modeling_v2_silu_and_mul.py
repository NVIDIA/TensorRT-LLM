# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the silu_and_mul entry (clamped SwiGLU, `swiglu_limit`).

The entry owns what is certified: `CELLS` is the list, `reference` and
`compare` are the gate. This file turns a cell's spec into tensors and keeps
the checks a cell cannot carry: the hand-read clamp branches, the every-row-count
prefix sweep with its one-compiled-variant probe and its one-row-shift control,
CUDA-graph capture at the decode row counts and at the engine's capture set,
the wrong-variant controls, the inner-stride guard, and fresh inputs at every
row count the envelope engines dumped.
"""

import zlib

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.activation.silu_and_mul import (  # noqa: E501
    silu_and_mul as op,
)
from tensorrt_llm._torch.modules.swiglu import silu_and_mul_kernel

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "silu_and_mul requires a CUDA device"

LIMIT = 10.0
# Spread of the synthetic gate/up values. At sigma 8 the normal tail beyond
# +-10 (1.25 sigma) is expected to hold about 21% of each half, so every clamp
# branch (gate above, gate below, up both sides) should be populated in every
# shape; `_assert_clamp_regions_populated` checks that rather than trusting it.
SIGMA = 8.0


def _ref(x: torch.Tensor, limit, out_dtype: torch.dtype) -> torch.Tensor:
    """The entry's reference at `swiglu_limit=limit`, output dtype `out_dtype`."""
    return op.reference(x, dtype=out_dtype, swiglu_limit=limit)


def _inputs(rows: int, d: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, 2 * d, device="cuda", dtype=torch.float32, generator=g) * SIGMA
    return x.to(dtype)


def _assert_clamp_regions_populated(x: torch.Tensor) -> None:
    d = x.shape[-1] // 2
    gate, up = x[:, :d].float(), x[:, d:].float()
    for name, n in (
        ("gate > L", (gate > LIMIT).sum().item()),
        ("gate < -L", (gate < -LIMIT).sum().item()),
        ("up > L", (up > LIMIT).sum().item()),
        ("up < -L", (up < -LIMIT).sum().item()),
    ):
        assert n > 0, f"no element with {name}: the clamp is not exercised"


def _build(spec: dict, seed: int) -> torch.Tensor:
    """The cell's `[rows, 2d]` input, as a view of a wider buffer when it names a row pad."""
    rows, d, pad = spec["rows"], spec["d"], spec["row_pad"]
    if pad == 0:
        return _inputs(rows, d, spec["x_dtype"], seed)
    big = _inputs(rows, d + pad // 2, spec["x_dtype"], seed)
    x = big[:, : 2 * d]
    assert x.stride() == (2 * d + pad, 1)
    return x


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    x = _build(spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    limit = spec["swiglu_limit"]
    if limit is not None and limit > 0 and x.numel():
        _assert_clamp_regions_populated(x)
    x_before = x.clone()
    kwargs = dict(
        dtype=spec["dtype"],
        swiglu_limit=limit,
        swiglu_alpha=spec["swiglu_alpha"],
        swiglu_beta=spec["swiglu_beta"],
    )
    out = op(x, **kwargs)
    torch.cuda.synchronize()
    out_dtype = spec["dtype"] or spec["x_dtype"]
    assert out.shape == (spec["rows"], spec["d"]) and out.dtype == out_dtype and out.is_contiguous()
    assert torch.equal(x, x_before), "input was mutated"
    op.compare(out, op.reference(x, **kwargs))
    if spec["swiglu_alpha"] is not None or spec["swiglu_beta"] is not None:
        # Passing alpha=1.0, beta=0.0 explicitly equals leaving them None.
        assert torch.equal(out, op(x, dtype=spec["dtype"], swiglu_limit=limit))


def test_clamp_at_hand_picked_values() -> None:
    """Each clamp branch on a value whose answer can be read off directly.

    gate 25 -> 10 (clamped from above); gate -25 stays -25 (no lower clamp);
    up 30 -> 10 and up -30 -> -10 (symmetric clamp).
    """
    gate = torch.tensor([25.0, -25.0, 1.0, 1.0, 25.0], device="cuda")
    up = torch.tensor([1.0, 1.0, 30.0, -30.0, 30.0], device="cuda")
    x = torch.cat([gate, up]).view(1, -1)
    out = op(x, swiglu_limit=LIMIT)
    ten = torch.tensor(10.0, device="cuda")
    expected = torch.stack(
        [
            F.silu(ten) * 1.0,
            F.silu(torch.tensor(-25.0, device="cuda")) * 1.0,
            F.silu(torch.tensor(1.0, device="cuda")) * 10.0,
            F.silu(torch.tensor(1.0, device="cuda")) * -10.0,
            F.silu(ten) * 10.0,
        ]
    ).view(1, -1)
    torch.testing.assert_close(out, expected)


def test_cuda_graph_replay() -> None:
    """Captured once, replayed on new data in the same static buffer."""
    x_static = _inputs(8, 2304, torch.bfloat16, seed=21)
    op(x_static, swiglu_limit=LIMIT)  # compile outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_static = op(x_static, swiglu_limit=LIMIT)
    for seed in (22, 23):
        x_static.copy_(_inputs(8, 2304, torch.bfloat16, seed=seed))
        graph.replay()
        torch.cuda.synchronize()
        op.compare(out_static, _ref(x_static, LIMIT, torch.bfloat16))


# ─── every row count at the shared-expert width (d = 2304) ─────────────────
#
# DeepSeek-V4.1's shared expert hands this op one row per token the rank holds
# this step, and d is always 2304. The row count is any integer from 1 to
# max_num_tokens (at most 16384 per rank). In layers 21-39 the decoder bounded
# replay (tep4/tp4) runs a decoder pass whose rows add the finishing requests'
# 128-row tails to the step's generation rows, up to 128 x 256 = 32768, and the
# shared expert runs there on all of them in one call. The DSpark drafter's
# shared expert (same d, same limit) sees its draft block, 5 rows per
# generation request: at most 1280. Rows enter the kernel only as the size of
# grid axis 0: program (i, j) reads row i and writes row i, and nothing it
# computes depends on how many rows there are. The tests below check that on
# the GPU, at every row count from 1 to 32768.

_D = 2304
_MAX_ROWS = 32768
# The core target's decode CUDA-graph batch sizes, plus two odd ones.
_GRAPH_ROWS = (1, 2, 3, 4, 7, 8, 16, 24, 32)


def _compiled_variants() -> dict:
    """Triton's in-process cache of compiled `silu_and_mul_kernel` variants on this device (key -> kernel)."""
    caches = silu_and_mul_kernel.device_caches[torch.cuda.current_device()]
    return caches[0]


def test_every_row_count_1_to_32768_is_the_same_per_row_function() -> None:
    """Every prefix x[:r], r = 1..32768, returns exactly the first r rows of the 32768-row result.

    The 32768-row result is held to the fp32 reference, so every row count is
    covered. Each output is poisoned with NaN before it is freed. The next,
    slightly longer output may reuse that allocator block, and any row the
    kernel skipped would otherwise still hold the previous call's correct bits.

    One-compiled-variant probe: the row count is not part
    of the compiled kernel's key. The sweep must add no entry to Triton's cache
    of compiled `silu_and_mul_kernel` variants, and the variant the 32768-row
    call used must be the only one with this call's constants.
    """
    x = _inputs(_MAX_ROWS, _D, torch.bfloat16, seed=41)
    _assert_clamp_regions_populated(x)
    full = op(x, swiglu_limit=LIMIT)
    op.compare(full, _ref(x, LIMIT, torch.bfloat16))
    full_bits = full.view(torch.int16)
    before = set(_compiled_variants())
    # This call's variant: bf16 output, no scale pointer, bf16 input, limit 10, alpha 1, beta 0.
    mine = [
        k
        for k in before
        if k.startswith("[('*bf16', 'D'), ('i32', 'D'), ('constexpr', None), ('*bf16', 'D')")
        and "('constexpr', 10.0), ('constexpr', 1.0), ('constexpr', 0.0)" in k
    ]
    bad = torch.zeros((), dtype=torch.int64, device="cuda")
    for r in range(1, _MAX_ROWS + 1):
        out = op(x[:r], swiglu_limit=LIMIT)
        assert out.shape == (r, _D) and out.dtype == torch.bfloat16
        bad += (out.view(torch.int16) != full_bits[:r]).sum()
        out.fill_(float("nan"))
    after = set(_compiled_variants())
    print(
        f"[rows] r = 1..{_MAX_ROWS}: {bad.item()} elements differ from the {_MAX_ROWS}-row result; "
        f"compiled variants before the sweep {len(before)}, after {len(after)}; "
        f"bf16 limit-10 variants: {len(mine)}"
    )
    assert bad.item() == 0, f"{bad.item()} elements differ from the {_MAX_ROWS}-row result's rows"
    assert after == before, (
        f"the row sweep compiled {len(after - before)} new variant(s): {sorted(after - before)}"
    )
    assert len(mine) == 1, sorted(before)


def test_row_sweep_harness_sees_a_one_row_shift() -> None:
    """Control for the sweep above: its bitwise per-row check catches a read one row off.

    A kernel that read row i + 1 for output row i would compute on real data
    and return finite, plausible numbers. The same check that the sweep
    applies must reject it at every row count, including r = 1.
    """
    x = _inputs(65, _D, torch.bfloat16, seed=42)
    full = op(x[:64], swiglu_limit=LIMIT)
    for r in (1, 2, 7, 63, 64):
        shifted = op(x[1 : r + 1], swiglu_limit=LIMIT)
        n_bad = int((shifted.view(torch.int16) != full.view(torch.int16)[:r]).sum().item())
        print(f"[control row shift] r={r}: {n_bad} / {r * _D} elements differ")
        assert n_bad > r * _D // 2


def test_cuda_graph_capture_at_decode_row_counts() -> None:
    """Captured at each decode batch size without an eager call at that size; replayed on new data.

    One eager call at 8 rows compiles the kernel. Every graph is then captured
    at a row count that has not run eagerly in this test. That works because
    the row count is not part of the compiled kernel. Each replay is checked
    against the reference and against an eager call on the same data,
    bitwise.
    """
    op(_inputs(8, _D, torch.bfloat16, seed=50), swiglu_limit=LIMIT)
    torch.cuda.synchronize()
    for rows in _GRAPH_ROWS:
        x_static = _inputs(rows, _D, torch.bfloat16, seed=51 + rows)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out_static = op(x_static, swiglu_limit=LIMIT)
        for seed in (1, 2):
            x_new = _inputs(rows, _D, torch.bfloat16, seed=1000 * seed + rows)
            x_static.copy_(x_new)
            out_static.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            op.compare(out_static, _ref(x_new, LIMIT, torch.bfloat16))
            assert torch.equal(out_static, op(x_new, swiglu_limit=LIMIT)), rows


# ─── deliberately wrong controls ────────────────────────────────────────────


def _wrong_symmetric_gate_clamp(x):
    d = x.shape[-1] // 2
    gate = torch.clamp(x[:, :d].float(), -LIMIT, LIMIT)
    up = torch.clamp(x[:, d:].float(), -LIMIT, LIMIT)
    return (F.silu(gate) * up).to(x.dtype)


def _wrong_up_gate_order(x):
    # MoE FC1's [up; gate] order applied to the dense [gate; up] layout.
    d = x.shape[-1] // 2
    gate = torch.clamp(x[:, d:].float(), max=LIMIT)
    up = torch.clamp(x[:, :d].float(), -LIMIT, LIMIT)
    return (F.silu(gate) * up).to(x.dtype)


def _wrong_unclamped(x):
    return _ref(x, None, x.dtype)


def _wrong_swigluoai(x):
    # GPT-OSS body: g * sigmoid(1.702 g) * (u + 1), same clamps.
    d = x.shape[-1] // 2
    gate = torch.clamp(x[:, :d].float(), max=LIMIT)
    up = torch.clamp(x[:, d:].float(), -LIMIT, LIMIT)
    return (gate * torch.sigmoid(1.702 * gate) * (up + 1.0)).to(x.dtype)


_WRONG = {
    "symmetric_gate_clamp": _wrong_symmetric_gate_clamp,
    "up_gate_order": _wrong_up_gate_order,
    "unclamped": _wrong_unclamped,
    "swigluoai": _wrong_swigluoai,
}


def _mismatches(out: torch.Tensor, ref: torch.Tensor) -> int:
    # Elementwise version of assert_close's default bf16 gate (rtol 1.6e-2, atol 1e-5).
    return int((~torch.isclose(out.float(), ref.float(), rtol=1.6e-2, atol=1e-5)).sum().item())


@pytest.mark.parametrize("name", list(_WRONG))
def test_wrong_variant_fails_on_inputs_beyond_the_limit(name: str) -> None:
    """Each wrong SwiGLU is rejected by the same gate the certified cases pass.

    Inputs beyond +-10 are what separate the clamp variants: the symmetric
    gate clamp and the unclamped body only differ from the right one there,
    so the same controls are also run on inputs confined to |x| < 10, where
    those two must be indistinguishable from the certified op -- which is
    why the certified cases draw at sigma 8.
    """
    x = _inputs(4096, 2304, torch.bfloat16, seed=31)
    _assert_clamp_regions_populated(x)
    out = op(x, swiglu_limit=LIMIT)
    op.compare(out, _ref(x, LIMIT, torch.bfloat16))  # the right one passes

    wrong = _WRONG[name](x)
    n_bad = _mismatches(out, wrong)
    max_abs = (out.float() - wrong.float()).abs().max().item()
    print(
        f"[control {name}] wide inputs: {n_bad} / {out.numel()} outside the bf16 gate, max |diff| {max_abs:.6g}"
    )
    with pytest.raises(AssertionError):
        torch.testing.assert_close(out, wrong)

    x_in = torch.clamp(x, -9.5, 9.5)
    out_in = op(x_in, swiglu_limit=LIMIT)
    n_bad_in = _mismatches(out_in, _WRONG[name](x_in))
    print(
        f"[control {name}] inputs within +-9.5: {n_bad_in} / {out_in.numel()} outside the bf16 gate"
    )
    if name in ("symmetric_gate_clamp", "unclamped"):
        assert n_bad_in == 0, "the clamp controls must be invisible without inputs beyond the limit"
    else:
        assert n_bad_in > 0


def test_non_unit_inner_stride_is_rejected_by_the_guard() -> None:
    """A transposed view is read as if contiguous; the guard refuses it."""
    base = torch.randn(2 * 2304, 64, device="cuda", dtype=torch.bfloat16)
    x = base.t()
    assert x.stride(-1) != 1
    with pytest.raises(AssertionError, match="unit stride"):
        with validating(op):
            op(x, swiglu_limit=LIMIT)


# ─── Capture set: decode, DSpark target verify, DSpark drafter ─────────────
#
# The captured batch sizes G, per rank: every G 1..32, plus TRT-LLM's padded
# default list for cuda_graph_config {enable_padding: true, max_batch_size:
# 256} (40..128 step 8, 192, 256); the padding-off list at 256 (1..32, 64, 128,
# 256) is a subset. A captured step is generation-only. The core target's
# shared expert hands this op G rows (next_n 1) or, with DSpark on, 6G (1 + 5
# draft tokens per request), and the drafter's shared expert (same d = 2304,
# same limit) hands it its 5-token draft block, 5G rows. Those are per-rank
# token counts in every member (the shared expert runs on the rank's own
# tokens, never on the dep4-gathered set). The engine captures one graph per
# batch size into one shared memory pool, largest first, and replays whichever
# one the step's batch selects. G, 5G and 6G over the 46 sizes: 114 row
# counts, 1..1536.

_GSET = tuple(range(1, 33)) + (40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 192, 256)
_VERIFY_GRAPH_ROWS = tuple(sorted({f * g for g in _GSET for f in (1, 5, 6)}))


@pytest.mark.parametrize("warm", [True, False], ids=["eager_first", "no_eager_at_these_rows"])
def test_cuda_graph_capture_at_gset_row_counts(warm: bool) -> None:
    """One graph per captured row count, replayed interleaved on new data.

    The row counts are G, 5G and 6G over the 46 G, all captured into one shared pool.

    `warm=True` follows the engine: an eager call at each row count before
    its capture. `warm=False` runs one eager call at 8 rows only (the kernel
    compile) and captures each row count without an eager call at it in
    this test (rows are not part of the compiled kernel). Graphs are captured largest
    first into a shared pool and then replayed in an interleaved order over
    three rounds. Before each replay the static input gets new data and the
    output is poisoned with NaN. The output must match an eager call on the
    same data bitwise, and the fp32 reference at the default bf16 gate. It
    must also differ from the previous round's output, which shows the replay
    read the new input rather than a copy baked in at capture.
    """
    op(_inputs(8, _D, torch.bfloat16, seed=60), swiglu_limit=LIMIT)
    torch.cuda.synchronize()
    pool = torch.cuda.graph_pool_handle()
    graphs = {}
    for rows in sorted(_VERIFY_GRAPH_ROWS, reverse=True):
        x_static = _inputs(rows, _D, torch.bfloat16, seed=61 + rows)
        if warm:
            op(x_static, swiglu_limit=LIMIT)
            torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            out_static = op(x_static, swiglu_limit=LIMIT)
        graphs[rows] = (graph, x_static, out_static)
    torch.cuda.synchronize()
    order = _VERIFY_GRAPH_ROWS[::2] + _VERIFY_GRAPH_ROWS[1::2][::-1]
    previous = {}
    for rnd in range(3):
        for rows in order:
            graph, x_static, out_static = graphs[rows]
            x_new = _inputs(rows, _D, torch.bfloat16, seed=7000 + 10000 * rnd + rows)
            if rnd == 0:
                _assert_clamp_regions_populated(x_new)
            x_static.copy_(x_new)
            out_static.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert out_static.shape == (rows, _D)
            op.compare(out_static, _ref(x_new, LIMIT, torch.bfloat16))
            assert torch.equal(out_static, op(x_new, swiglu_limit=LIMIT)), (rows, rnd)
            if rows in previous:
                assert not torch.equal(out_static, previous[rows]), (rows, rnd)
            previous[rows] = out_static.clone()
    print(
        f"[capture warm={warm}] {len(graphs)} graphs in one pool, rows {min(graphs)}..{max(graphs)}: "
        f"3 rounds, every replay bitwise = eager and inside the gate, each differing from the previous round"
    )


# ─── fresh inputs at the row counts the envelope engines dumped ─────────────
#
# Row counts of this call in the core target's runtime-value dumps of the
# envelope engines (max_num_tokens 16384, max_batch_size 256; tep4,
# tp4 and dep4, padded graphs and eager, DSpark on; plus a 1M-token and a
# 128K-token long-context engine), 76 values up to 16384; the DSpark drafter's
# block rows N (5 per generation request) from the same dumps' drafter
# attention records, 40 values 5..1280, which its shared expert sees too; and
# 32768, the top of the decoder bounded-replay pass.
_CG_DUMPED_ROWS = (
    1, 2, 4, 5, 6, 7, 8, 12, 16, 18, 24, 29, 32, 34, 48, 60, 96, 108, 128, 134, 144, 168, 192, 228, 240,
    288, 330, 336, 384, 432, 462, 480, 528, 546, 576, 597, 610, 624, 672, 720, 768, 780, 852, 1152, 1536,
    2778, 2784, 3870, 4208, 4245, 4388, 4394, 4591, 4868, 8467, 9193, 9240, 9364, 9370, 9524, 10441,
    10444, 10447, 10451, 16298, 16308, 16316, 16320, 16322, 16327, 16328, 16329, 16333, 16340, 16378,
    16384,
)  # fmt: skip
_CG_DRAFTER_ROWS = (
    5, 10, 15, 20, 40, 50, 80, 90, 110, 115, 120, 125, 140, 160, 190, 200, 240, 275, 280, 320, 360, 370,
    385, 400, 440, 455, 480, 520, 560, 580, 585, 600, 640, 650, 665, 710, 715, 720, 960, 1280,
)  # fmt: skip


def test_dumped_row_counts_fresh_inputs() -> None:
    """Fresh inputs at every dumped row count (and 32768), compared directly with the reference."""
    rows_all = sorted(set(_CG_DUMPED_ROWS) | set(_CG_DRAFTER_ROWS) | {_MAX_ROWS})
    for rows in rows_all:
        x = _inputs(rows, _D, torch.bfloat16, seed=90000 + rows)
        _assert_clamp_regions_populated(x)
        out = op(x, swiglu_limit=LIMIT)
        assert out.shape == (rows, _D) and out.dtype == torch.bfloat16 and out.is_contiguous()
        op.compare(out, _ref(x, LIMIT, torch.bfloat16))
    print(
        f"[dumped rows] {len(rows_all)} row counts, fresh inputs, all inside the default bf16 gate"
    )
