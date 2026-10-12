# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_rmsnorm catalog entry.

The entry owns what is certified: `CELLS` is the list, `reference` and
`compare` are the gate, `is_valid` is the guard. This file owns only what a
cell cannot carry -- turning a spec into real tensors -- plus the inputs the
guard exists to refuse.
"""

import pytest
import torch
from flashinfer.norm import rmsnorm as _flashinfer_norm_rmsnorm
from flashinfer.norm.kernels import rmsnorm as _cute_rmsnorm_kernels
from torch.profiler import ProfilerActivity, profile

from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm as op,
)
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "flashinfer_rmsnorm requires a CUDA device"

#: The DeepSeek-V4.1 vision tower's residual stream carries massive activations
#: in these channels (|x| ~ 2000).
_VIT_MASSIVE = (53, 147, 308)


def _v41_row_scales(rows: int) -> torch.Tensor:
    """Row i's magnitude class, i % 4: unit; 1e-10, where mean(x^2) is about eps 1e-20 and eps
    decides the result; 1e-11, where mean(x^2) is far below eps; all-zero."""
    classes = torch.tensor([1.0, 1e-10, 1e-11, 0.0], device="cuda")
    return classes[torch.arange(rows, device="cuda") % 4]


def _vit_rows(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """The vision tower's row classes, row i taking i % 4, in place on fp32 `x`: unit; unit with
    the massive channels at +-2000; 1e-3 scale, where mean(x^2) is about eps 1e-6; all-zero."""
    cls = torch.arange(x.shape[0], device="cuda") % 4
    x[cls == 2] *= 1e-3
    x[cls == 3] = 0.0
    sign = torch.where(
        torch.rand(x.shape[0], len(_VIT_MASSIVE), device="cuda", generator=g) < 0.5, -1.0, 1.0
    )
    massive = x[:, list(_VIT_MASSIVE)]
    x[:, list(_VIT_MASSIVE)] = torch.where((cls == 1)[:, None], 2000.0 * sign, massive)
    return x


def _build(spec, seed):
    """Tensors for one cell; the optional keys default to a contiguous randn input.

    `row_stride` / `col_offset`: `x` is the column view `parent[:, col_offset : col_offset + width]`
    of a `[rows, row_stride]` parent, as the DeepSeek-V4.1 target slices its qkv and ring buffers.
    `data`: the rows' magnitude pattern -- "eps_classes" (`_v41_row_scales`) or "vit" (`_vit_rows`,
    with the weight drawn near 1 as the vision tower's is).
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape, dtype = spec["shape"], spec["dtype"]
    stride, offset, data = spec.get("row_stride"), spec.get("col_offset", 0), spec.get("data")
    x = torch.randn(shape if stride is None else (shape[0], stride), generator=g, device="cuda")
    if data == "eps_classes":
        x = x * _v41_row_scales(shape[0])[:, None]
    elif data == "vit":
        x = _vit_rows(x, g)
    x = x.to(dtype)
    if stride is not None:
        x = x[:, offset : offset + shape[-1]]
    weight = torch.randn(shape[-1], generator=g, device="cuda")
    if data == "vit":
        weight = 1.0 + 0.5 * weight
    return x, weight.to(dtype), spec["eps"]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    x, weight, eps = _build(cell.spec, seed=abs(hash(cell.why)) % 2**31)
    out = op(x, weight, eps)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert out.data_ptr() != x.data_ptr(), "the op must return a new tensor"
    if cell.spec.get("data") is not None:
        # The magnitude-class cells: zero rows come back exactly 0, and nothing overflows.
        assert torch.isfinite(out).all() and (out[3::4] == 0).all()
    op.compare(out, op.reference(x, weight, eps))


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


def test_guard_refuses_a_strided_normalized_dim() -> None:
    """`x.stride(-1) != 1` reads the wrong elements rather than failing."""
    buf = torch.randn(8, 2880, 2, dtype=torch.bfloat16, device="cuda")
    x = buf[..., 0]
    assert x.stride(-1) != 1
    weight = torch.randn(2880, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="contiguous"):
        with validating(op):
            op(x, weight, 1e-6)


# ---------------------------------------------------------------------------
# DeepSeek-V4.1. The core target's call sites, all bf16 at eps = 1e-20 (the
# checkpoint's rms_norm_eps):
#   width 5120: attn_norm / ffn_norm / final norm on contiguous [T, 5120]
#   width 1280: q_norm on the column view qkv[:, :1280] of a [T, 1792] buffer
#   width  512: kv_norm on the column view qkv[:, 1280:] (row stride 1792,
#               element offset 1280), comp_norm on contiguous [n, 512]
#   width  128: index-K norm on contiguous [n, 128]
# and the DSpark drafter's ring write: kv3 = wkv3(main_x) is a [M, 1536]
# buffer and each of the three stages normalizes its own width-512 column view
# of it, split(kv3, [512] * 3, -1): row stride 1536, element offsets 0, 512
# and 1024. Rows: the engine's per-rank token counts, 1..16384 (the largest
# per-rank max_num_tokens an envelope engine runs). The cells hold the two
# ends; the row sweep below covers every count between, and the capture tests
# the engines' whole capture set.
_V41_EPS = 1e-20
_V41_ROWS = (1, 2, 3, 7, 24, 32, 323, 1005, 4617, 8192, 16384)
_V41_GRAPH_ROWS = (1, 2, 4, 8, 16, 24, 32)
_RING_PARENT = 1536
_RING_OFFSET = {"ring0": 0, "ring1": 512, "ring2": 1024}


def _v41_parent(rows: int, layout: str) -> torch.Tensor:
    """The buffer the target's input is a view of.

    [rows, 1792] for the two column views of the qkv projection, [rows, 1536]
    for the ring views, [rows, width] otherwise. Rows cycle through the four
    magnitude classes of `_v41_row_scales`.
    """
    if layout in ("q", "kv"):
        width = 1792
    elif layout in _RING_OFFSET:
        width = _RING_PARENT
    else:
        width = {"hidden": 5120, "comp": 512, "idx": 128}[layout]
    x = torch.randn(rows, width, device="cuda") * _v41_row_scales(rows)[:, None]
    return x.to(torch.bfloat16)


def _v41_view(parent: torch.Tensor, layout: str, rows: int) -> torch.Tensor:
    """The first `rows` rows of `parent`, in the layout the target hands the op."""
    if layout == "q":
        return parent[:rows, :1280]
    if layout == "kv":
        return parent[:rows, 1280:]
    if layout in _RING_OFFSET:
        off = _RING_OFFSET[layout]
        return parent[:rows, off : off + 512]
    return parent[:rows]


def _v41_input(rows: int, layout: str) -> torch.Tensor:
    """A [rows, width] input in the target's layout (see `_v41_parent`)."""
    return _v41_view(_v41_parent(rows, layout), layout, rows)


def test_v41_eps_1e20_widths_and_layouts() -> None:
    """Fresh inputs at sampled row counts; eps 1e-20 is honoured, not clamped."""
    torch.manual_seed(30)
    for layout in ("hidden", "q", "kv", "comp", "idx"):
        for rows in _V41_ROWS:
            x = _v41_input(rows, layout)
            if layout in ("q", "kv"):
                # A single row reports contiguous (torch ignores a size-1
                # dim's stride) and takes the kernel's compact path; every
                # other row count takes the strided path.
                assert x.stride() == (1792, 1) and x.is_contiguous() == (rows == 1)
            w = torch.randn(x.shape[-1], dtype=torch.bfloat16, device="cuda")
            out = op(x, w, _V41_EPS)
            assert out.shape == x.shape and out.dtype == x.dtype
            assert torch.isfinite(out).all()
            op.compare(out, op.reference(x, w, _V41_EPS))
            if rows >= 4:
                # Zero rows come back zero; eps-dominated rows are O(1), not
                # O(x): the same gate rejects an eps of 1e-6 on those rows.
                assert (out[3::4] == 0).all()
                tiny = slice(1, None, 4)
                wrong = op.reference(x, w, 1e-6)
                assert not torch.allclose(
                    out[tiny].float(), wrong[tiny].float(), rtol=1.6e-2, atol=1e-5
                )


def test_v41_eps_1e20_cuda_graph() -> None:
    """Each core layout captured alone after a side-stream warm-up, replayed on two fresh inputs."""
    torch.manual_seed(31)
    for layout in ("hidden", "q", "kv", "comp", "idx"):
        width = {"hidden": 5120, "q": 1280, "kv": 512, "comp": 512, "idx": 128}[layout]
        w = torch.randn(width, dtype=torch.bfloat16, device="cuda")
        for rows in _V41_GRAPH_ROWS:
            x = _v41_input(rows, layout)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                op(x, w, _V41_EPS)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = op(x, w, _V41_EPS)
            for _ in range(2):
                x.copy_(_v41_input(rows, layout))
                out.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                op.compare(out, op.reference(x, w, _V41_EPS))


# ─── Every row count 1..16384 at every layout, and the capture union ─────────
#
# The row axis of a per-row op is certified as a range: a run at N_max = 16384
# against the reference, every row count r = 1..N_max bitwise equal to the
# first r rows of that run, and a probe that the row count compiles nothing
# new. On the certified path (FlashInfer's CuTe DSL RMSNormKernel) one CTA
# normalizes several rows -- 2 at width 5120, 4 at 1280 and 512, 8 at 128,
# read from the kernel object on sm_103 -- and a predicate on the row count
# masks the rows at or past r in the last CTA. So r reaches the kernel twice:
# as the grid size and as the bound of that mask. Compiled kernels are looked
# up by dtype, width, weight bias, PDL, SM version and whether the rows are
# compact; the row count is a symbolic 64-bit integer.
#
# The column views are rows of one parent, sliced exactly as the target
# slices it.

_N_MAX = 16384
_CORE_LAYOUTS = ("hidden", "q", "kv", "comp", "idx")
_LAYOUTS = _CORE_LAYOUTS + tuple(_RING_OFFSET)
_COLUMN_VIEWS = ("q", "kv") + tuple(_RING_OFFSET)
_WIDTH = {
    "hidden": 5120,
    "q": 1280,
    "kv": 512,
    "comp": 512,
    "idx": 128,
    "ring0": 512,
    "ring1": 512,
    "ring2": 512,
}
# The capture union per rank. G ranges over every CUDA-graph batch size an
# envelope engine captures: every G 1..32, plus TRT-LLM's padded default list
# for cuda_graph_config {enable_padding: true, max_batch_size: 256}
# (40..128 step 8, 192, 256); the padding-off list at 256 (1..32, 64, 128,
# 256) is a subset. A captured step is generation-only, so the rows are G for
# a decode step, 6G for the DSpark target-verify step (1 + 5 draft tokens per
# request), 3G for that step's ratio-2 compressed rows (each verify request
# reserves ceil(6 / 2) = 3 of them, so the compressed-latent [n, 512] and
# index-K [n, 128] norms run at n = 3G), and 5G for the drafter's draft block.
# These norms run on the rank's own tokens in every member, never on a
# dep4-gathered set. Every core layout is captured at every count of the
# union, including the counts only some of the layouts reach.
_BATCHES = tuple(range(1, 33)) + tuple(range(40, 129, 8)) + (192, 256)
_CAPTURE_ROWS = tuple(sorted({m * b for b in _BATCHES for m in (1, 3, 5, 6)}))
assert len(_BATCHES) == 46 and len(_CAPTURE_ROWS) == 126 and max(_CAPTURE_ROWS) == 1536
# The profiler probe's row counts: 1..9, the capture union, the edges of the
# 2-, 4- and 8-row CTAs around 16, 64, 128, 4096, 8192 and 12288, two prefill
# counts from the sampled rows, and the top of the range.
_PROBE_ROWS = tuple(
    sorted(
        set(range(1, 10))
        | set(_CAPTURE_ROWS)
        | {15, 17, 63, 64, 65, 127, 128, 129, 323, 1005, 4095, 4096, 4097, 4617, 8191, 8192, 8193}
        | {12287, 12288, 12289, 16383, 16384}
    )
)


def _compiled_variants() -> int:
    """How many RMSNormKernel variants the certified path has compiled in this process."""
    return _cute_rmsnorm_kernels._get_compiled_rmsnorm_kernel.cache_info().misses


def _rmsnorm_into(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, eps: float = _V41_EPS
) -> None:
    """The call the op's body makes (`flashinfer.norm.rmsnorm`, same PDL setting), into a caller-owned `out`.

    Used only to see what the kernel writes past the last row: the op returns
    a fresh tensor, so the memory after its last row is not the caller's to
    inspect.
    """
    _flashinfer_norm_rmsnorm(x, weight, eps, out=out, enable_pdl=get_env_enable_pdl())


def _ordered_bits(t: torch.Tensor) -> torch.Tensor:
    b = t.view(torch.int16).int()
    return torch.where(b < 0, -(b & 0x7FFF), b)


def _error_stats(out: torch.Tensor, ref: torch.Tensor):
    """(max |out - ref|, max distance in bf16 ulps, fraction of elements bit-identical)."""
    diff = (out.float() - ref.float()).abs().max().item()
    ulps = int((_ordered_bits(out) - _ordered_bits(ref)).abs().max().item())
    exact = (out.view(torch.int16) == ref.view(torch.int16)).float().mean().item()
    return diff, ulps, exact


@pytest.mark.parametrize("layout", _LAYOUTS)
def test_v41_every_row_count_is_the_same_per_row_function(layout: str) -> None:
    """Every r = 1..16384: the op on the first r rows returns exactly the first r rows of the 16384-row result.

    The 16384-row result is held to the fp32 reference, so every row count is
    covered. Each output is poisoned with NaN before it is freed, so a row the
    kernel skipped cannot pick up the previous call's correct bits from a
    recycled allocator block. At every r the same kernel also runs into a
    NaN-filled caller buffer one row longer than r, and the row past the last
    must stay NaN. The sweep compiles no new kernel variant, and the parent
    buffer (all its columns, for the views) is left bitwise untouched.
    """
    torch.manual_seed(40 + _LAYOUTS.index(layout))
    width = _WIDTH[layout]
    parent = _v41_parent(_N_MAX, layout)
    snapshot = parent.clone()
    w = torch.randn(width, dtype=torch.bfloat16, device="cuda")
    x = _v41_view(parent, layout, _N_MAX)
    full = op(x, w, _V41_EPS)
    ref = op.reference(x, w, _V41_EPS)
    op.compare(full, ref)
    assert torch.isfinite(full).all() and (full[3::4] == 0).all()
    if layout in _RING_OFFSET:
        # The DSpark ring view gives bitwise what the op gives on a contiguous
        # copy of the same rows (the `comp` layout's kernel).
        assert full.is_contiguous()
        assert torch.equal(
            full.view(torch.int16),
            op(x.contiguous(), w, _V41_EPS).view(torch.int16),
        )
    diff, ulps, exact = _error_stats(full, ref)
    print(
        f"\n[{layout}] {_N_MAX} rows vs reference: max abs diff {diff:.6g}, "
        f"max {ulps} bf16 ulp, bit-exact {exact:.6f}"
    )

    # A one-row view reports contiguous, so for the column views the sweep
    # reaches two variants; compile both before counting.
    op(_v41_view(parent, layout, 1), w, _V41_EPS)
    variants = _compiled_variants()

    full_bits = full.view(torch.int16)
    buf = torch.empty(_N_MAX + 1, width, dtype=torch.bfloat16, device="cuda")
    mismatch = torch.zeros(_N_MAX + 1, dtype=torch.bool, device="cuda")
    mismatch_into = torch.zeros(_N_MAX + 1, dtype=torch.bool, device="cuda")
    guard = torch.zeros(_N_MAX + 1, dtype=torch.int64, device="cuda")
    for r in range(1, _N_MAX + 1):
        xr = _v41_view(parent, layout, r)
        out = op(xr, w, _V41_EPS)
        assert out.shape == (r, width) and out.dtype == torch.bfloat16
        mismatch[r] = (out.view(torch.int16) != full_bits[:r]).any()
        out.fill_(float("nan"))
        del out
        buf[: r + 1].fill_(float("nan"))
        _rmsnorm_into(xr, w, buf[:r])
        mismatch_into[r] = (buf[:r].view(torch.int16) != full_bits[:r]).any()
        guard[r] = (~buf[r].isnan()).sum()
    torch.cuda.synchronize()
    assert _compiled_variants() == variants, "the row sweep compiled a new kernel variant"
    bad = mismatch.nonzero().flatten().tolist()
    bad_into = mismatch_into.nonzero().flatten().tolist()
    over = guard.nonzero().flatten().tolist()
    assert not bad, (
        f"{len(bad)} row counts differ from the {_N_MAX}-row result's rows, first {bad[:10]}"
    )
    assert not bad_into, (
        f"{len(bad_into)} row counts differ into a caller buffer, first {bad_into[:10]}"
    )
    assert not over, f"{len(over)} row counts wrote the row past the last, first {over[:10]}"
    assert torch.equal(parent.view(torch.int16), snapshot.view(torch.int16)), (
        "the op wrote its input"
    )
    print(
        f"[{layout}] r = 1..{_N_MAX}: every prefix bitwise equal, row past the last untouched, "
        f"no compile during the sweep ({variants} variants compiled in this process)"
    )


def test_v41_row_sweep_harness_sees_a_row_shift_and_a_tail_write() -> None:
    """Controls for the sweep: its bitwise check rejects a read one row off, and its guard sees a write at row r."""
    torch.manual_seed(47)
    for layout in _LAYOUTS:
        width = _WIDTH[layout]
        parent = _v41_parent(66, layout)
        w = torch.randn(width, dtype=torch.bfloat16, device="cuda")
        full = op(_v41_view(parent, layout, 64), w, _V41_EPS)
        for r in (1, 2, 7, 63, 64):
            shifted = op(_v41_view(parent[1:], layout, r), w, _V41_EPS)
            n_bad = int((shifted.view(torch.int16) != full.view(torch.int16)[:r]).sum().item())
            print(f"[control row shift, {layout}] r={r}: {n_bad} / {r * width} elements differ")
            assert n_bad > r * width // 2
        buf = torch.full((66, width), float("nan"), dtype=torch.bfloat16, device="cuda")
        _rmsnorm_into(_v41_view(parent, layout, 64), w, buf[:64])
        assert buf[64].isnan().all()
        _rmsnorm_into(_v41_view(parent, layout, 65), w, buf[:65])  # one row too many
        written = int((~buf[64].isnan()).sum().item())
        print(
            f"[control tail write, {layout}] a 65-row call leaves {written} / {width} non-NaN in row 64"
        )
        assert written == width


@pytest.mark.parametrize("layout", _LAYOUTS)
def test_v41_one_compiled_variant_per_layout(layout: str) -> None:
    """Probe: each call launches one kernel, and every row count >= 2 launches the same one.

    At one row a column view reports contiguous (torch ignores a size-1 dim's
    stride) and launches the compact variant, the one a contiguous [1, width]
    input launches; the contiguous layouts launch the same kernel at one row
    as at every other count. No call in the probe compiles a new variant.
    """
    torch.manual_seed(48)
    width = _WIDTH[layout]
    parent = _v41_parent(_N_MAX, layout)
    w = torch.randn(width, dtype=torch.bfloat16, device="cuda")

    def launched(x: torch.Tensor) -> str:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            op(x, w, _V41_EPS)
            torch.cuda.synchronize()
        names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        assert len(names) == 1, (tuple(x.shape), names)
        return names[0]

    for r in (1, 2):
        op(_v41_view(parent, layout, r), w, _V41_EPS)
    variants = _compiled_variants()
    by_rows = {r: launched(_v41_view(parent, layout, r)) for r in _PROBE_ROWS}
    compact_one_row = launched(torch.randn(1, width, device="cuda").to(torch.bfloat16))
    assert _compiled_variants() == variants, "the probe compiled a new kernel variant"
    names = set(by_rows.values())
    assert all("RMSNormKernel" in n for n in names), names
    many = {by_rows[r] for r in _PROBE_ROWS if r >= 2}
    assert len(many) == 1, many
    assert by_rows[1] == compact_one_row
    if layout in _COLUMN_VIEWS:
        assert by_rows[1] not in many
    else:
        assert by_rows[1] in many
    print(f"\n[{layout}] {len(_PROBE_ROWS)} row counts, kernels launched:")
    for n in sorted(names):
        print(f"  {n}  (rows {[r for r in _PROBE_ROWS if by_rows[r] == n][:3]}...)")


def test_v41_capture_union_replay_is_bitwise_eager() -> None:
    """The five calls captured at every row count of the capture union, replayed on new data.

    As the engine does it: an eager call at each shape before its capture,
    one graph per row count holding the five calls, all captured into one
    shared memory pool, largest first, then replayed in an interleaved order
    over three rounds. Before each replay new data is written into the
    captured buffers (the whole [rows, 1792] qkv buffer for the two column
    views) and every output is poisoned with NaN. Each replayed output must
    equal an eager call on the same data bitwise, pass the reference gate, and
    differ from the previous round's output. A control round then writes the
    new data where the graph does not read it: the replay matches the eager
    result on the stale buffer and not the one on the new data, so the bitwise
    check would catch a replay that read stale input.
    """
    torch.manual_seed(49)
    weights = {
        layout: torch.randn(_WIDTH[layout], dtype=torch.bfloat16, device="cuda")
        for layout in _CORE_LAYOUTS
    }
    source = {"hidden": "hidden", "q": "qkv", "kv": "qkv", "comp": "comp", "idx": "idx"}

    def fresh(rows: int) -> dict:
        return {
            "hidden": _v41_parent(rows, "hidden"),
            "qkv": _v41_parent(rows, "q"),
            "comp": _v41_parent(rows, "comp"),
            "idx": _v41_parent(rows, "idx"),
        }

    pool = torch.cuda.graph_pool_handle()
    graphs = {}
    for rows in sorted(_CAPTURE_ROWS, reverse=True):
        bufs = fresh(rows)
        views = {layout: _v41_view(bufs[source[layout]], layout, rows) for layout in _CORE_LAYOUTS}
        for layout in _CORE_LAYOUTS:  # the engine's warm-up: one eager call at this shape
            op(views[layout], weights[layout], _V41_EPS)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            outs = {
                layout: op(views[layout], weights[layout], _V41_EPS) for layout in _CORE_LAYOUTS
            }
        graphs[rows] = (graph, bufs, views, outs)
    torch.cuda.synchronize()

    order = _CAPTURE_ROWS[::2] + _CAPTURE_ROWS[1::2][::-1]
    previous = {}
    worst = {layout: (0.0, 0, 1.0) for layout in _CORE_LAYOUTS}
    for rnd in range(3):
        for rows in order:
            graph, bufs, views, outs = graphs[rows]
            for key, new in fresh(rows).items():
                bufs[key].copy_(new)
            for out in outs.values():
                out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            for layout in _CORE_LAYOUTS:
                out, x, w = outs[layout], views[layout], weights[layout]
                assert out.shape == (rows, _WIDTH[layout])
                eager = op(x, w, _V41_EPS)
                assert torch.equal(out.view(torch.int16), eager.view(torch.int16)), (
                    layout,
                    rows,
                    rnd,
                )
                ref = op.reference(x, w, _V41_EPS)
                op.compare(out, ref)
                if (layout, rows) in previous:
                    assert not torch.equal(
                        out.view(torch.int16), previous[layout, rows].view(torch.int16)
                    )
                previous[layout, rows] = out.clone()
                d, u, e = _error_stats(out, ref)
                wd, wu, we = worst[layout]
                worst[layout] = (max(wd, d), max(wu, u), min(we, e))

    for rows in order:
        graph, bufs, views, outs = graphs[rows]
        elsewhere = fresh(rows)
        for out in outs.values():
            out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        for layout in _CORE_LAYOUTS:
            w = weights[layout]
            stale = op(views[layout], w, _V41_EPS)
            new = op(_v41_view(elsewhere[source[layout]], layout, rows), w, _V41_EPS)
            assert torch.equal(outs[layout].view(torch.int16), stale.view(torch.int16)), (
                layout,
                rows,
            )
            assert not torch.equal(outs[layout].view(torch.int16), new.view(torch.int16)), (
                layout,
                rows,
            )

    print(f"\n{len(_CAPTURE_ROWS)} graphs, rows {_CAPTURE_ROWS}, 3 rounds; replay vs reference:")
    for layout, (d, u, e) in worst.items():
        print(f"  [{layout}] max abs diff {d:.6g}, max {u} bf16 ulp, min bit-exact {e:.6f}")


# ─── The DSpark drafter's ring write: three width-512 views of [M, 1536] ─────
#
# `_ring_write` computes kv3 = wkv3(main_x), a fresh [M, 1536] bf16 buffer, and
# normalizes each stage's column view split(kv3, [512] * 3, -1)[s] with that
# stage's kv_norm at eps 1e-20, out of place. M is n <= 128 at prefill seeding
# (eager), G (the bonus row) and 5G (the multi-accept back-fill), G <=
# max_batch_size 256; a draft step is captured at the engine's graph batch
# sizes and runs eager at any other G. The row sweep above covers every M
# 1..16384 at the three views (layouts ring0/ring1/ring2); this section
# captures at every G 1..256 (a superset of the captured sizes), 5G, and the
# capture union up to 1280, and adds the stride/offset generalization.
_DSPARK_G = tuple(range(1, 257))
_DSPARK_RING_ROWS = tuple(
    sorted(
        set(_DSPARK_G)
        | {5 * g for g in _DSPARK_G}
        | {r for r in _CAPTURE_ROWS if r <= 5 * max(_DSPARK_G)}
    )
)
assert len(_DSPARK_RING_ROWS) == 473 and max(_DSPARK_RING_ROWS) == 1280


def test_v41_dspark_ring_views_captured_at_every_draft_batch() -> None:
    """The drafter's three kv_norm calls captured at M = G and 5G for every G 1..256, replayed on new data.

    One graph per M holds the three calls on the three column views of one
    [M, 1536] buffer, as `_ring_write` issues them. As the engine does it: an
    eager call at each shape before its capture, all graphs in one shared
    memory pool, largest first, replays interleaved over three rounds with new
    data in the whole [M, 1536] buffer and NaN-poisoned outputs. Every replay
    equals an eager call on the same views and an eager call on a contiguous
    copy of them bitwise, passes the reference gate, and differs from the
    previous round's. A control round writes the new data where the graph does
    not read it and the replay then matches only the stale result.
    """
    torch.manual_seed(60)
    rings = tuple(_RING_OFFSET)
    weights = {layout: torch.randn(512, dtype=torch.bfloat16, device="cuda") for layout in rings}
    pool = torch.cuda.graph_pool_handle()
    graphs = {}
    for rows in sorted(_DSPARK_RING_ROWS, reverse=True):
        kv3 = _v41_parent(rows, "ring0")
        views = {layout: _v41_view(kv3, layout, rows) for layout in rings}
        for layout in rings:
            op(views[layout], weights[layout], _V41_EPS)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            outs = {layout: op(views[layout], weights[layout], _V41_EPS) for layout in rings}
        for out in outs.values():
            assert out.shape == (rows, 512)
            # Contiguous at M >= 2; at M = 1 it keeps the view's row stride.
            assert out.stride() == ((512, 1) if rows >= 2 else (_RING_PARENT, 1))
        graphs[rows] = (graph, kv3, views, outs)
    torch.cuda.synchronize()

    order = _DSPARK_RING_ROWS[::2] + _DSPARK_RING_ROWS[1::2][::-1]
    previous = {}
    worst = (0.0, 0, 1.0)
    for rnd in range(3):
        for rows in order:
            graph, kv3, views, outs = graphs[rows]
            kv3.copy_(_v41_parent(rows, "ring0"))
            for out in outs.values():
                out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            for layout in rings:
                out, x, w = outs[layout], views[layout], weights[layout]
                bits = out.view(torch.int16)
                assert torch.equal(bits, op(x, w, _V41_EPS).view(torch.int16)), (
                    layout,
                    rows,
                    rnd,
                )
                assert torch.equal(bits, op(x.contiguous(), w, _V41_EPS).view(torch.int16))
                ref = op.reference(x, w, _V41_EPS)
                op.compare(out, ref)
                if (layout, rows) in previous:
                    assert not torch.equal(bits, previous[layout, rows]), (layout, rows, rnd)
                previous[layout, rows] = bits.clone()
                d, u, e = _error_stats(out, ref)
                worst = (max(worst[0], d), max(worst[1], u), min(worst[2], e))

    for rows in order:
        graph, kv3, views, outs = graphs[rows]
        elsewhere = _v41_parent(rows, "ring0")
        for out in outs.values():
            out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        for layout in rings:
            w = weights[layout]
            stale = op(views[layout], w, _V41_EPS)
            new = op(_v41_view(elsewhere, layout, rows), w, _V41_EPS)
            assert torch.equal(outs[layout].view(torch.int16), stale.view(torch.int16)), (
                layout,
                rows,
            )
            assert not torch.equal(outs[layout].view(torch.int16), new.view(torch.int16)), (
                layout,
                rows,
            )
    print(
        f"\n[dspark ring] {len(_DSPARK_RING_ROWS)} graphs x 3 views, M {_DSPARK_RING_ROWS}, 3 rounds; "
        f"replay vs reference: max abs diff {worst[0]:.6g}, max {worst[1]} bf16 ulp, min bit-exact {worst[2]:.6f}"
    )


# The stride/offset generalization: every row stride S that is a multiple of 8
# elements (16 bytes) from 512 to 2048, every column offset o that is a multiple
# of 8 with o + 512 <= S, at the CTA-edge row counts below (4 rows per CTA at
# width 512) up to the drafter's 160. A one-row view reports contiguous and
# takes the compact kernel, which needs a 128-byte-aligned start: at one row
# only the offsets that are multiples of 64 elements run (the parents here are
# fresh allocations, so 128-byte aligned); the others raise ValueError.
_GEN_STRIDES = tuple(range(512, 2049, 8))
_GEN_ROWS = (2, 3, 4, 5, 8, 9, 64, 65, 127, 128, 129, 159, 160)
_GEN_N = 160


def test_v41_dspark_any_16_byte_aligned_stride_and_offset() -> None:
    """Width 512, every 16-byte-aligned (row stride, column offset) pair in 512..2048: bitwise the contiguous copy.

    Per pair, the view's first r rows (r in `_GEN_ROWS`, and r = 1 where the
    offset is 128-byte aligned) give bitwise the first r rows of the op on a
    contiguous copy of the 160-row view; that contiguous result passes the
    reference gate, zero rows come back 0, the parent buffer is not written,
    and no kernel variant compiles (the row stride is a symbolic argument of
    the one strided kernel). A one-row view at an offset that is not a
    multiple of 64 elements raises ValueError. Mismatches accumulate on the
    device; one sync at the end.
    """
    torch.manual_seed(61)
    w = torch.randn(512, dtype=torch.bfloat16, device="cuda")
    op(_v41_view(_v41_parent(2, "ring1"), "ring1", 2), w, _V41_EPS)
    op(_v41_input(2, "comp"), w, _V41_EPS)
    op(_v41_input(1, "comp"), w, _V41_EPS)
    variants = _compiled_variants()
    scales = _v41_row_scales(_GEN_N)[:, None]
    bad_bits = torch.zeros((), dtype=torch.int64, device="cuda")
    bad_gate = torch.zeros((), dtype=torch.int64, device="cuda")
    bad_zero = torch.zeros((), dtype=torch.int64, device="cuda")
    bad_input = torch.zeros((), dtype=torch.int64, device="cuda")
    pairs = one_row_ran = one_row_raised = 0
    for stride in _GEN_STRIDES:
        parent = (torch.randn(_GEN_N, stride, device="cuda") * scales).to(torch.bfloat16)
        snapshot = parent.clone()
        for off in range(0, stride - 512 + 1, 8):
            view = parent[:, off : off + 512]
            contig = view.contiguous()
            full = op(contig, w, _V41_EPS)
            ref = op.reference(contig, w, _V41_EPS)
            bad_gate += (
                (full.float() - ref.float()).abs() > 1e-5 + 1.6e-2 * ref.float().abs()
            ).sum()
            bad_zero += (full[3::4] != 0).sum()
            full_bits = full.view(torch.int16)
            for r in _GEN_ROWS:
                out = op(view[:r], w, _V41_EPS)
                bad_bits += (out.view(torch.int16) != full_bits[:r]).sum()
            if off % 64 == 0:
                out = op(view[:1], w, _V41_EPS)
                bad_bits += (out.view(torch.int16) != full_bits[:1]).sum()
                one_row_ran += 1
            else:
                with pytest.raises(ValueError, match="Misaligned Tensor data"):
                    op(view[:1], w, _V41_EPS)
                one_row_raised += 1
            pairs += 1
        bad_input += (parent.view(torch.int16) != snapshot.view(torch.int16)).sum()
    torch.cuda.synchronize()
    print(
        f"\n[dspark generalization] {len(_GEN_STRIDES)} strides 512..2048 step 8, {pairs} (stride, offset) pairs, "
        f"rows {_GEN_ROWS}: {int(bad_bits)} elements differ from the contiguous copy, {int(bad_gate)} outside "
        f"the gate, {int(bad_zero)} nonzero zero-row elements, {int(bad_input)} input elements written, "
        f"{_compiled_variants() - variants} variants compiled; one row: {one_row_ran} pairs ran (offset % 64 == 0), "
        f"{one_row_raised} raised ValueError"
    )
    assert pairs == sum((s - 512) // 8 + 1 for s in _GEN_STRIDES) == one_row_ran + one_row_raised
    assert int(bad_bits) == 0 and int(bad_gate) == 0 and int(bad_zero) == 0 and int(bad_input) == 0
    assert _compiled_variants() == variants, "a stride or offset compiled a new kernel variant"


@pytest.mark.parametrize(
    "stride, off, rows",
    [
        (1536, 4, 8),
        (1536, 1, 8),
        (1536, 516, 8),
        (1540, 0, 8),
        (1538, 512, 8),
        (520, 2, 8),
        (1536, 8, 1),
        (1536, 520, 1),
        (1792, 1272, 1),
    ],
)
def test_v41_dspark_misaligned_stride_or_offset_raises(stride: int, off: int, rows: int) -> None:
    """Misaligned views raise ValueError; nothing runs.

    At two or more rows, a row stride or column offset that is not a multiple of
    8 elements (16 bytes). At one row, where the compact kernel runs, a start
    that is not 128-byte aligned.
    """
    parent = torch.randn(rows, stride, device="cuda").to(torch.bfloat16)
    w = torch.randn(512, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="Misaligned Tensor data|Invalid mX.strides"):
        op(parent[:, off : off + 512], w, _V41_EPS)
    torch.cuda.synchronize()


# ─── The DeepSeek-V4.1 vision tower: width 1024, eps 1e-6, rows 1..51111 ─────
#
# The ViT normalizes its
# residual stream with this op at width 1024 and eps 1e-6 -- norm1 and norm2
# of each of the 32 blocks and the final norm -- on a contiguous [P, 1024]
# bf16 tensor, P = every patch of one request (at most 17037 over the 900
# MMMU val documents). The row range is certified up to 51111 = 3 x 17037
# rows. The residual stream carries massive activations (channels 53, 147
# and 308 reach |x| ~ 2000 in the golden), so rows cycle through: unit;
# unit with those three channels at +-2000; 1e-3 scale, where mean(x^2) ~
# eps and eps sets the result; all-zero.
_VIT_WIDTH = 1024
_VIT_EPS = 1e-6
_VIT_N_MAX = 51111
# The prototype's whole-tower CUDA graph ran at the golden cases' patch
# counts; 17037 is the MMMU maximum.
_VIT_GRAPH_ROWS = (1517, 4680, 5325, 8772, 17037)
_VIT_PROBE_ROWS = (
    1,
    2,
    3,
    4,
    5,
    7,
    8,
    9,
    15,
    16,
    17,
    63,
    64,
    65,
    127,
    128,
    129,
    1517,
    4680,
    5325,
    8191,
    8192,
    8193,
    8772,
    16383,
    16384,
    16385,
    17037,
    32767,
    32768,
    32769,
    51110,
    51111,
)


def _vit_input(rows: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, _VIT_WIDTH, device="cuda", generator=g)
    return _vit_rows(x, g).to(torch.bfloat16)


def _vit_weight(seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (1.0 + 0.5 * torch.randn(_VIT_WIDTH, device="cuda", generator=g)).to(torch.bfloat16)


def test_v41_vit_every_row_count_is_the_same_per_row_function() -> None:
    """Width 1024, eps 1e-6: every r = 1..51111 is bitwise the first r rows of the 51111-row call.

    The 51111-row result is held to the fp32 reference. Each output is
    poisoned with NaN before it is freed; at every r the same kernel also runs
    into a NaN-filled caller buffer one row longer than r, whose last row must
    stay NaN; the sweep compiles no new kernel variant; the input is left
    bitwise untouched.
    """
    x = _vit_input(_VIT_N_MAX, 80)
    snapshot = x.clone()
    w = _vit_weight(81)
    full = op(x, w, _VIT_EPS)
    ref = op.reference(x, w, _VIT_EPS)
    op.compare(full, ref)
    assert torch.isfinite(full).all() and (full[3::4] == 0).all()
    diff, ulps, exact = _error_stats(full, ref)
    rpb = _cute_rmsnorm_kernels.RMSNormKernel(
        _cute_rmsnorm_kernels.get_cutlass_dtype("bfloat16"), _VIT_WIDTH, 0.0, sm_version=103
    ).rows_per_block
    print(
        f"\n[vit] {_VIT_N_MAX} rows vs reference: max abs diff {diff:.6g}, max {ulps} bf16 ulp, "
        f"bit-exact {exact:.6f}; rows per CTA {rpb}"
    )
    variants = _compiled_variants()
    full_bits = full.view(torch.int16)
    buf = torch.empty(_VIT_N_MAX + 1, _VIT_WIDTH, dtype=torch.bfloat16, device="cuda")
    mismatch = torch.zeros(_VIT_N_MAX + 1, dtype=torch.bool, device="cuda")
    mismatch_into = torch.zeros(_VIT_N_MAX + 1, dtype=torch.bool, device="cuda")
    guard = torch.zeros(_VIT_N_MAX + 1, dtype=torch.int64, device="cuda")
    for r in range(1, _VIT_N_MAX + 1):
        out = op(x[:r], w, _VIT_EPS)
        mismatch[r] = (out.view(torch.int16) != full_bits[:r]).any()
        out.fill_(float("nan"))
        del out
        buf[: r + 1].fill_(float("nan"))
        _rmsnorm_into(x[:r], w, buf[:r], _VIT_EPS)
        mismatch_into[r] = (buf[:r].view(torch.int16) != full_bits[:r]).any()
        guard[r] = (~buf[r].isnan()).sum()
    torch.cuda.synchronize()
    assert _compiled_variants() == variants, "the row sweep compiled a new kernel variant"
    bad = mismatch.nonzero().flatten().tolist()
    bad_into = mismatch_into.nonzero().flatten().tolist()
    over = guard.nonzero().flatten().tolist()
    assert not bad, (
        f"{len(bad)} row counts differ from the {_VIT_N_MAX}-row result's rows, first {bad[:10]}"
    )
    assert not bad_into, (
        f"{len(bad_into)} row counts differ into a caller buffer, first {bad_into[:10]}"
    )
    assert not over, f"{len(over)} row counts wrote the row past the last, first {over[:10]}"
    assert torch.equal(x.view(torch.int16), snapshot.view(torch.int16)), "the op wrote its input"
    print(
        f"[vit] r = 1..{_VIT_N_MAX}: every prefix bitwise equal, row past the last untouched, no compile"
    )


def test_v41_vit_eps_and_row_controls() -> None:
    """eps 1e-6 is honoured, and a one-row shift is rejected.

    The core's 1e-20 and a 1e-5 are rejected on the eps-scale rows.
    """
    x = _vit_input(4096, 82)
    w = _vit_weight(83)
    out = op(x, w, _VIT_EPS)
    op.compare(out, op.reference(x, w, _VIT_EPS))
    tiny = slice(2, None, 4)
    for wrong_eps in (1e-20, 1e-5):
        wrong = op.reference(x, w, wrong_eps)
        bad = _bf16_outside(out[tiny], wrong[tiny])
        print(
            f"\n[vit control eps {wrong_eps:g}] {bad:.4f} of the eps-scale rows' elements outside the bf16 gate"
        )
        assert bad > 0.5
    for r in (1, 2, 7, 64, 4095):
        shifted = op(x[1 : r + 1], w, _VIT_EPS)
        n_bad = int((shifted.view(torch.int16) != out[:r].view(torch.int16)).sum())
        print(f"[vit control row shift] r={r}: {n_bad} / {r * _VIT_WIDTH} elements differ")
        assert n_bad > r * _VIT_WIDTH // 2


def _bf16_outside(a: torch.Tensor, b: torch.Tensor) -> float:
    """Fraction of elements outside torch's default bf16 gate (rtol 1.6e-2, atol 1e-5)."""
    af, bf = a.float(), b.float()
    return float(((af - bf).abs() > 1e-5 + 1.6e-2 * bf.abs()).float().mean())


def test_v41_vit_one_compiled_variant() -> None:
    """Every probed row count 1..51111 launches the same single RMSNormKernel, and nothing compiles."""
    x = _vit_input(_VIT_N_MAX, 84)
    w = _vit_weight(85)
    op(x[:2], w, _VIT_EPS)
    variants = _compiled_variants()
    names = set()
    for r in _VIT_PROBE_ROWS:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            op(x[:r], w, _VIT_EPS)
            torch.cuda.synchronize()
        launched = [
            e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
        ]
        assert len(launched) == 1, (r, launched)
        names.add(launched[0])
    assert _compiled_variants() == variants, "the probe compiled a new kernel variant"
    assert len(names) == 1 and "RMSNormKernel" in next(iter(names)), names
    print(f"\n[vit] {len(_VIT_PROBE_ROWS)} row counts, one kernel: {next(iter(names))}")


def test_v41_vit_cuda_graph_at_the_prototype_sizes() -> None:
    """Captured at the prototype's graph sizes into one pool, after an eager call.

    Replays on new data are bitwise eager.
    """
    w = _vit_weight(86)
    pool = torch.cuda.graph_pool_handle()
    graphs = {}
    for rows in sorted(_VIT_GRAPH_ROWS, reverse=True):
        x = _vit_input(rows, 87)
        op(x, w, _VIT_EPS)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            out = op(x, w, _VIT_EPS)
        graphs[rows] = (graph, x, out)
    previous = {}
    for rnd in range(2):
        for rows in _VIT_GRAPH_ROWS:
            graph, x, out = graphs[rows]
            x.copy_(_vit_input(rows, 88 + rnd))
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out.view(torch.int16), op(x, w, _VIT_EPS).view(torch.int16)), rows
            op.compare(out, op.reference(x, w, _VIT_EPS))
            if rows in previous:
                assert not torch.equal(out.view(torch.int16), previous[rows].view(torch.int16)), (
                    rows
                )
            previous[rows] = out.clone()
    print(
        f"\n[vit] captured at {_VIT_GRAPH_ROWS}: every replay bitwise equal to eager and in the gate"
    )
