# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the bmm_out catalog entry.

Beyond the cells, the DeepSeek-V4.1 section runs that target's grouped output
projection at every row count 1..32768 at each group count, at the row counts
its runtime dumps show on fresh operands, under CUDA graphs at every row count
it captures, and under a profiler that holds the call to one kernel.
"""

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.bmm_out import bmm_out as op

__extra_import_path__ = [".."]
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "bmm_out requires a CUDA device"


def _build(spec, seed):
    """Materialize one cell.

    `transposed_out` is the part a spec cannot express as a plain `empty`: the
    targets hand the op a transposed view of a buffer the forward already
    holds, so the product lands in place. The parent buffer comes back too,
    because "did it land there" is the thing worth asserting.
    `transposed_a` and `transposed_b`, when a cell sets them, make the inputs
    the same kind of view: `a` of a [M, B, K] activation, `b` of a [B, N, K]
    weight -- DeepSeek-V4.1's grouped output projection passes all three.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    b, m, k, n, dt = spec["batch"], spec["m"], spec["k"], spec["n"], spec["dtype"]
    if spec.get("transposed_a", False):
        a = torch.randn(m, b, k, generator=g, device="cuda").to(dt).transpose(0, 1)
    else:
        a = torch.randn(b, m, k, generator=g, device="cuda").to(dt)
    if spec.get("transposed_b", False):
        w = torch.randn(b, n, k, generator=g, device="cuda").to(dt).transpose(1, 2)
    else:
        w = torch.randn(b, k, n, generator=g, device="cuda").to(dt)
    if spec["transposed_out"]:
        parent = torch.zeros(m, b, n, dtype=dt, device="cuda")
        out = torch.transpose(parent, 0, 1)
        # Only a decode-shaped or single-batch cell escapes this: transposing a
        # dim of extent 1 leaves a tensor torch still calls contiguous, so the
        # assertion would be about the shape rather than about the view.
        assert m == 1 or b == 1 or not out.is_contiguous()
    else:
        parent = torch.zeros(b, m, n, dtype=dt, device="cuda")
        out = parent
    return a, w, out, parent


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    a, w, out, parent = _build(cell.spec, seed=abs(hash(cell.why)) % 2**31)
    parent_ptr = parent.data_ptr()
    with validating(op):
        # Identity, not equality: the op must have written this very buffer and
        # returned it, not produced a copy that happens to match.
        assert op(a, w, out) is out, "the op writes in place and returns what it wrote"
    # The product has to be in the caller's buffer, not in a replacement the op
    # allocated: that is the whole reason this op exists over torch.bmm.
    assert parent.data_ptr() == parent_ptr
    op.compare(out, op.reference(a, w, out))


def _ok(dtype=torch.bfloat16):
    a = torch.randn(4, 8, 16, dtype=dtype, device="cuda")
    w = torch.randn(4, 16, 32, dtype=dtype, device="cuda")
    out = torch.zeros(4, 8, 32, dtype=dtype, device="cuda")
    return a, w, out


def test_guard_refuses_a_wrong_shaped_out() -> None:
    """The op silently reallocates it, detaching the caller's buffer."""
    a, w, _ = _ok()
    with pytest.raises(AssertionError, match=r"out must be \[B, M, N\]"):
        with validating(op):
            op(a, w, torch.zeros(4, 8, 31, dtype=torch.bfloat16, device="cuda"))


def test_guard_refuses_a_2d_operand() -> None:
    a, w, out = _ok()
    with pytest.raises(AssertionError, match="must be 3D"):
        with validating(op):
            op(a[0], w, out)


def test_guard_refuses_mixed_dtypes() -> None:
    """Promoted rather than rejected, which changes the required out dtype."""
    a, w, out = _ok()
    with pytest.raises(AssertionError, match="share one dtype"):
        with validating(op):
            op(a, w.to(torch.float16), out)


# ---------------------------------------------------------------------------
# DeepSeek-V4.1's grouped low-rank output projection (wo_a). The cells above
# hold it at the ends of its row range; what follows keeps the measured range
# certified. The call is made exactly in the target's layout: bf16 o [T, G*4096],
# the parameter w [G*1024, 4096] (HF [out, in]) and the caller's oa [T, G*1024]:
#   a   = o.view(T, G, 4096).transpose(0, 1)      [G, T, 4096]
#   b   = w.view(G, 1024, 4096).transpose(1, 2)   [G, 4096, 1024]
#   out = oa.view(T, G, 1024).transpose(0, 1)     [G, T, 1024]
# G is the rank's share of the 8 output groups: 8 / tp under attention TP (4, 2,
# 1 at tp 2, 4, 8), 8 under attention DP and in the DSpark drafter's attention
# (replicated, all 8 groups per rank). T is the rank's token count this step, at
# most max_num_tokens (16384 per rank at the largest envelope engine); in the
# decoder bounded-replay pass (attention TP, G = 1, 2, 4) the rows add the
# finishing requests' 128-row tails to the step's generation rows, up to
# 128 x 256 = 32768.
_GROUPS = (1, 2, 4, 8)
_NMAX = 32768
# rtol: a one-step bf16 rounding flip is <= 2^-8 relative. atol: at K = 4096
# with unit-variance operands (outputs up to ~360) the op's fp32 summation order
# differs from the reference's. Measured on sm_103 over every T in 1..32768, the
# largest excess of |op - ref| over rtol * |ref| was 6.3e-4 (G = 1), 5.1e-4
# (G = 2), 6.2e-4 (G = 4) and 5.7e-4 (G = 8): at least 3.1x under this atol, where
# 1e-3 would leave 1.5x. The group-swap control lands ~390 away.
_V41_TOL = {"rtol": 1.6e-2, "atol": 2e-3}
# At G = 1 cuBLAS serves T in 1..32 and 65..128 with a split-K kernel followed by
# cublasLt::splitKreduce_kernel: two launches, and a summation order over K that
# no other T uses. There the result is inside the gate and repeatable, but not
# bitwise the first T rows of a larger call. Every other (G, T) in 1..32768
# launches one kernel.
_SPLIT_K_T = {1: frozenset(range(1, 33)) | frozenset(range(65, 129))}
# T of this call in the core target's boot-gate runtime-value dumps (tep4, tp4,
# dep4 and tep8): the decode batches 1..8 and the eager prefill chunks, the
# decoder pass's rows among them.
_DUMPED = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    12,
    14,
    22,
    24,
    48,
    128,
    140,
    888,
    890,
    1018,
    1030,
    1935,
    1942,
    2048,
)
# The captured batch sizes G per rank: every G 1..32 plus TRT-LLM's padded
# default list for cuda_graph_config {enable_padding: true, max_batch_size: 256}
# (40..128 step 8, 192, 256); the padding-off list at 256 is a subset. A captured
# step is generation-only: the core call sees T = G (next_n 1) or 6G (DSpark
# target verify), and the DSpark drafter's attention makes the same call at 8
# groups on its 5-token draft block, 5G rows. Every group count is captured at
# the whole union G, 5G, 6G: 114 row counts, 1..1536.
_GSET = tuple(range(1, 33)) + tuple(range(40, 129, 8)) + (192, 256)
_CAPTURE = tuple(sorted({f * g for g in _GSET for f in (1, 5, 6)}))
assert len(_CAPTURE) == 114 and _CAPTURE[-1] == 1536
# Eager prefill chunks run between graph replays, as chunked prefill runs
# between decode steps.
_PREFILL = (14, 48, 888, 890, 1018, 1935, 2048, 4868, 10451, 16384)


def _v41_views(o: torch.Tensor, w: torch.Tensor, oa: torch.Tensor, groups: int, tokens: int):
    """The target's three views over the first `tokens` rows of o and oa."""
    a = o[:tokens].view(tokens, groups, 4096).transpose(0, 1)
    b = w.view(groups, 1024, 4096).transpose(1, 2)
    out = oa[:tokens].view(tokens, groups, 1024).transpose(0, 1)
    return a, b, out


def _v41_operands(groups: int, tokens: int):
    o = torch.randn(tokens, groups * 4096, device="cuda").to(torch.bfloat16)
    w = torch.randn(groups * 1024, 4096, device="cuda").to(torch.bfloat16)
    oa = torch.empty(tokens, groups * 1024, device="cuda", dtype=torch.bfloat16)
    return (o, w, oa) + _v41_views(o, w, oa, groups, tokens)


def _v41_ref(o: torch.Tensor, w: torch.Tensor, groups: int) -> torch.Tensor:
    """The entry's reference over all of o's rows, in oa's [T, G*1024] layout."""
    tokens = o.shape[0]
    oa = torch.empty(tokens, groups * 1024, device="cuda", dtype=torch.bfloat16)
    ref = op.reference(*_v41_views(o, w, oa, groups, tokens))
    return ref.transpose(0, 1).reshape(tokens, groups * 1024)


def test_dsv41_group_swap_fails_the_gate() -> None:
    """A group-permuted weight fails both gates: a group mix-up in this strided
    layout cannot pass unseen."""
    torch.manual_seed(20)
    o, w, oa, a, b, out = _v41_operands(2, 32)
    op(a, b, out)
    swapped = w.view(2, 1024, 4096).flip(0).reshape(2 * 1024, 4096)
    assert not torch.allclose(oa.float(), _v41_ref(o, swapped, 2).float(), **_V41_TOL)
    with pytest.raises(AssertionError):
        op.compare(out, op.reference(a, swapped.view(2, 1024, 4096).transpose(1, 2), out))
    op.compare(out, op.reference(a, b, out))


@pytest.mark.parametrize("groups", (2, 8))
def test_dsv41_cuda_graph_side_stream_warmup(groups: int) -> None:
    """T in {1, 2, 4, 8, 16, 24, 32}, one graph at a time after a side-stream
    warm-up, into a caller-owned oa; two replays on fresh o, each in the gate."""
    torch.manual_seed(21 + groups)
    for tokens in (1, 2, 4, 8, 16, 24, 32):
        o, w, oa, a, b, out = _v41_operands(groups, tokens)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            op(a, b, out)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op(a, b, out)
        for _ in range(2):
            o.copy_(torch.randn(tokens, groups * 4096, device="cuda").to(torch.bfloat16))
            oa.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(oa, _v41_ref(o, w, groups), **_V41_TOL)


@pytest.mark.parametrize("groups", _GROUPS)
def test_dsv41_every_row_count_1_to_32768(groups: int) -> None:
    """Every T in 1..32768 in the target's layout, twice: ascending, then in a seeded shuffle.

    One [32768, G*4096] activation and the weight; the call at T reads the first
    T rows and writes the first T rows of a [32768 + 1, G*1024] buffer that is
    NaN-filled through row T before the call. Each output row depends only on
    its own input row, so the reference is computed once over all 32768 rows
    and sliced. At every T: every element is within _V41_TOL of the reference
    (NaN fails), the first T rows are bitwise equal to the 32768-row call's
    first T rows except at the split-K T of _SPLIT_K_T, and row T (the guard
    row) is still NaN. Counters stay on the device; one sync at the end.
    """
    torch.manual_seed(4100 + groups)
    n = _NMAX
    o = torch.randn(n, groups * 4096, device="cuda").to(torch.bfloat16)
    w = torch.randn(groups * 1024, 4096, device="cuda").to(torch.bfloat16)
    ref = _v41_ref(o, w, groups)
    oa = torch.full((n + 1, groups * 1024), float("nan"), device="cuda", dtype=torch.bfloat16)
    op(*_v41_views(o, w, oa, groups, n))
    torch.testing.assert_close(oa[:n], ref, **_V41_TOL)
    assert oa[n].isnan().all(), f"the {n}-row call wrote past its last row"
    full_bits = oa[:n].clone().view(torch.int16)
    # Control for the per-row bitwise check: no row of the result equals the
    # next one, so a call that read or wrote one row off could not pass it.
    assert (full_bits[1:] != full_bits[:-1]).any(-1).all(), "neighbouring rows coincide"

    ref_f = ref.float()
    slack = _V41_TOL["rtol"] * ref_f.abs()
    gate_bad = torch.zeros(n + 1, dtype=torch.int64, device="cuda")
    prefix_bad = torch.zeros(n + 1, dtype=torch.int64, device="cuda")
    guard_bad = torch.zeros(n + 1, dtype=torch.int64, device="cuda")
    excess = torch.full((n + 1,), -float("inf"), device="cuda")
    shuffled = (
        torch.randperm(n, generator=torch.Generator().manual_seed(4200 + groups)) + 1
    ).tolist()
    for step, tokens in enumerate(list(range(1, n + 1)) + shuffled):
        oa[: tokens + 1].fill_(float("nan"))
        op(*_v41_views(o, w, oa, groups, tokens))
        over = oa[:tokens].float().sub_(ref_f[:tokens]).abs_().sub_(slack[:tokens])
        gate_bad[tokens] += (~(over <= _V41_TOL["atol"])).sum()
        excess[tokens] = torch.maximum(excess[tokens], over.max())
        prefix_bad[tokens] += (oa[:tokens].view(torch.int16) != full_bits[:tokens]).sum()
        guard_bad[tokens] += (~oa[tokens].isnan()).sum()
        del over
        if step % 2048 == 2047:
            # The ascending pass asks for a slightly larger temporary every call, which no
            # cached block fits; release them now and then rather than filling the device.
            torch.cuda.empty_cache()
    torch.cuda.synchronize()
    failing = gate_bad.nonzero().flatten().tolist()
    differing = prefix_bad.nonzero().flatten().tolist()
    wrote_guard = guard_bad.nonzero().flatten().tolist()
    worst = int(excess.argmax())
    split_k = _SPLIT_K_T.get(groups, frozenset())
    # Both passes count, so an element that differs in both is counted twice.
    frac = max((prefix_bad[t].item() / (2 * t * groups * 1024) for t in differing), default=0.0)
    print(
        f"[every T G={groups}] T=1..{n} ascending + shuffled: {len(failing)} T outside the gate, "
        f"{len(differing)} T not bitwise equal to the {n}-row call's prefix (all split-K: "
        f"{set(differing) <= split_k}; at most {frac:.3e} of a call's elements), "
        f"{len(wrote_guard)} T wrote the guard row; worst excess of |op - ref| over "
        f"rtol*|ref|: {excess[worst].item():.3e} at T={worst}"
    )
    assert not failing, (
        f"outside the gate at T = {failing[:20]} (counts {gate_bad[failing[:20]].tolist()})"
    )
    outside = [t for t in differing if t not in split_k]
    assert not outside, (
        f"not bitwise equal to the {n}-row call's first T rows at T = {outside[:20]} "
        f"(counts {prefix_bad[outside[:20]].tolist()})"
    )
    assert not wrote_guard, f"wrote row T at T = {wrote_guard[:20]}"


@pytest.mark.parametrize("tokens", _DUMPED)
@pytest.mark.parametrize("groups", _GROUPS)
def test_dsv41_dumped_row_counts_fresh_operands(groups: int, tokens: int) -> None:
    """Every boot-gate dumped T on operands allocated at exactly that size, output NaN-filled first."""
    torch.manual_seed(4400 + 16 * tokens + groups)
    o, w, oa, a, b, out = _v41_operands(groups, tokens)
    oa.fill_(float("nan"))
    ptr = oa.data_ptr()
    assert op(a, b, out) is out
    assert oa.data_ptr() == ptr and out.data_ptr() == ptr
    torch.testing.assert_close(oa, _v41_ref(o, w, groups), **_V41_TOL)


# T of this call in the runtime-value dumps of the envelope engines
# (max_num_tokens 16384, max_batch_size 256; tep4 and tp4 at G = 2, dep4 at
# G = 8; padded graphs and eager, DSpark on; plus a 1M-token and a 128K-token
# long-context engine at G = 2). At G = 8 the DSpark drafter's block rows N
# (5 per generation request) from the same dumps are added. 32768, the top of
# the decoder-pass range, closes both lists.
_ENVELOPE_DUMPED = {
    2: (
        1, 2, 4, 5, 6, 7, 8, 12, 16, 18, 24, 29, 32, 34, 48, 60, 96, 108, 128, 134, 144, 168, 192, 228,
        240, 288, 330, 336, 384, 432, 462, 480, 528, 546, 576, 610, 624, 672, 720, 768, 780, 852, 1152,
        1536, 2778, 2784, 3870, 4208, 4245, 4388, 4394, 4591, 4868, 9364, 9370, 10444, 10447, 10451,
        16308, 16316, 16320, 16322, 16329, 16333, 16340, 16384, 32768,
    ),
    8: (
        1, 2, 5, 6, 7, 12, 16, 24, 48, 96, 144, 192, 240, 288, 336, 384, 432, 480, 528, 576, 597, 624,
        672, 720, 768, 1152, 1536, 8467, 9193, 9240, 9524, 10441, 10451, 16298, 16327, 16328, 16378,
        16384, 32768,
        # drafter N
        10, 15, 20, 40, 50, 80, 90, 110, 115, 120, 125, 140, 160, 190, 200, 275, 280, 320, 360, 370,
        385, 400, 440, 455, 520, 560, 580, 585, 600, 640, 650, 665, 710, 715, 960, 1280,
    ),
}  # fmt: skip


@pytest.mark.parametrize("groups", (2, 8))
def test_dsv41_envelope_dumped_row_counts_fresh_operands(groups: int) -> None:
    """Every envelope-dumped T on operands allocated at exactly that size, output NaN-filled first."""
    rows = sorted(set(_ENVELOPE_DUMPED[groups]))
    w = torch.randn(groups * 1024, 4096, device="cuda").to(torch.bfloat16)
    for tokens in rows:
        torch.manual_seed(4600 + tokens)
        o = torch.randn(tokens, groups * 4096, device="cuda").to(torch.bfloat16)
        oa = torch.full((tokens, groups * 1024), float("nan"), device="cuda", dtype=torch.bfloat16)
        a, b, out = _v41_views(o, w, oa, groups, tokens)
        ptr = oa.data_ptr()
        op(a, b, out)
        assert oa.data_ptr() == ptr and out.data_ptr() == ptr
        torch.testing.assert_close(
            oa, _v41_ref(o, w, groups), **_V41_TOL, msg=lambda m, t=tokens: f"T={t}: {m}"
        )
        del o, oa, a, out
    print(f"[dumped G={groups}] {len(rows)} row counts {rows[0]}..{rows[-1]}, all inside the gate")


@pytest.mark.parametrize("groups", _GROUPS)
def test_dsv41_capture_union_shared_pool(groups: int) -> None:
    """One graph per T of the capture union (G, 5G, 6G over the 46 batch sizes),
    captured and replayed as the engine does.

    The weight is one tensor shared by every graph. For each T, largest first:
    an eager call at that shape, then capture into one shared pool of the call
    reading a static o [T, G*4096] (the captured input) and writing an oa
    allocated inside the capture, as a forward allocates it. Then three rounds,
    each replaying all 114 graphs in a seeded shuffled order. Before each replay
    an eager prefill-sized call runs on other buffers, new activations are
    written into the captured o, and oa is filled with NaN. Each replay's oa
    must be bitwise equal to an eager call on the same o, within _V41_TOL of the
    reference, and different on every row from the previous round's result
    (the stale-replay control: a replay that re-read the previous activations,
    or left the previous output in place, would reproduce it).
    """
    torch.manual_seed(4300 + groups)
    w = torch.randn(groups * 1024, 4096, device="cuda").to(torch.bfloat16)
    pf_n = max(_PREFILL)
    pf_o = torch.randn(pf_n, groups * 4096, device="cuda").to(torch.bfloat16)
    pf_oa = torch.empty(pf_n, groups * 1024, device="cuda", dtype=torch.bfloat16)
    pf_ref = _v41_ref(pf_o, w, groups)

    pool = torch.cuda.graph_pool_handle()
    graphs = {}
    for tokens in sorted(_CAPTURE, reverse=True):
        o = torch.randn(tokens, groups * 4096, device="cuda").to(torch.bfloat16)
        warm = torch.empty(tokens, groups * 1024, device="cuda", dtype=torch.bfloat16)
        op(*_v41_views(o, w, warm, groups, tokens))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            oa = torch.empty(tokens, groups * 1024, device="cuda", dtype=torch.bfloat16)
            op(*_v41_views(o, w, oa, groups, tokens))
        graphs[tokens] = (graph, o, oa)

    gen = torch.Generator().manual_seed(4500 + groups)
    prev = {}
    step = 0
    for rnd in range(3):
        order = [_CAPTURE[i] for i in torch.randperm(len(_CAPTURE), generator=gen).tolist()]
        for tokens in order:
            graph, o, oa = graphs[tokens]
            pf = _PREFILL[step % len(_PREFILL)]
            step += 1
            pf_oa.fill_(float("nan"))
            op(*_v41_views(pf_o, w, pf_oa, groups, pf))
            o.copy_(torch.randn(tokens, groups * 4096, device="cuda").to(torch.bfloat16))
            oa.fill_(float("nan"))
            graph.replay()
            eager = torch.full_like(oa, float("nan"))
            op(*_v41_views(o, w, eager, groups, tokens))
            torch.cuda.synchronize()
            assert torch.equal(oa, eager), f"replay != eager at T={tokens}, round {rnd}"
            torch.testing.assert_close(oa, _v41_ref(o, w, groups), **_V41_TOL)
            torch.testing.assert_close(pf_oa[:pf], pf_ref[:pf], **_V41_TOL)
            if tokens in prev:
                assert (oa != prev[tokens]).any(-1).all(), (
                    f"stale replay not rejected at T={tokens}"
                )
            prev[tokens] = oa.clone()
    print(f"[graphs G={groups}] {len(graphs)} graphs, one pool: 3 rounds bitwise == eager")


@pytest.mark.parametrize("groups", _GROUPS)
def test_dsv41_one_kernel_per_call_every_row_count(groups: int) -> None:
    """At every T in 1..32768 the call launches exactly one kernel, on a CUDA profiler trace,
    except at the split-K T of _SPLIT_K_T, where it launches exactly a split-K kernel and then
    cublasLt::splitKreduce_kernel.

    The calls run in chunks of 1024 row counts, one profiler session per chunk,
    on one stream, so the chunk's kernels in start-time order are its calls'
    kernels in order. No other call may launch a second kernel (a split-K
    reduction, or a copy or transpose for the strided views). The kernel names
    (a split-K call's first kernel) give the kernel bands: a band is a run of
    consecutive T served by one kernel. The bands are cuBLAS's choice and are
    reported, not asserted.
    """
    n = _NMAX
    split_k = _SPLIT_K_T.get(groups, frozenset())
    torch.manual_seed(4700 + groups)
    o = torch.randn(n, groups * 4096, device="cuda").to(torch.bfloat16)
    w = torch.randn(groups * 1024, 4096, device="cuda").to(torch.bfloat16)
    oa = torch.empty(n, groups * 1024, device="cuda", dtype=torch.bfloat16)
    op(*_v41_views(o, w, oa, groups, n))
    torch.cuda.synchronize()
    names = []
    chunk = 1024
    for start in range(1, n + 1, chunk):
        ts = range(start, min(start + chunk, n + 1))
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for tokens in ts:
                op(*_v41_views(o, w, oa, groups, tokens))
            torch.cuda.synchronize()
        kernels = sorted(
            (e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA),
            key=lambda e: e.time_range.start,
        )
        want = len(ts) + sum(t in split_k for t in ts)
        assert len(kernels) == want, (
            f"T {ts.start}..{ts.stop - 1}: {len(kernels)} kernels for {len(ts)} calls, "
            f"expected {want}; {sorted({e.name for e in kernels})}"
        )
        it = iter(e.name for e in kernels)
        for tokens in ts:
            first = next(it)
            if tokens in split_k:
                second = next(it)
                assert "splitK" in first and "splitKreduce_kernel" in second, (
                    tokens,
                    first,
                    second,
                )
            else:
                assert "splitK" not in first and "reduce" not in first.lower(), (tokens, first)
            names.append(first)
    bands = [1] + [t for t in range(2, n + 1) if names[t - 1] != names[t - 2]]
    distinct = sorted(set(names))
    low = {names[t - 1] for t in range(1, 8193)}
    print(
        f"[one kernel G={groups}] T=1..{n}: one kernel per call outside {len(split_k)} split-K T "
        f"(two there); {len(distinct)} distinct kernels in {len(bands)} bands "
        f"({sum(b <= 8192 for b in bands)} bands and {len(low)} kernels start at T <= 8192); "
        f"split-K kernels: {sorted({names[t - 1] for t in split_k})}; "
        f"kernels first seen above 8192: {sorted(set(names[8192:]) - low)}"
    )
