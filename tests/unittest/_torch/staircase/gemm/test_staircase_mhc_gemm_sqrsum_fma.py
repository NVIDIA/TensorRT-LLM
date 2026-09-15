# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the mhc_gemm_sqrsum_fma catalog entry.

The expected value is built here from native torch only, in **float64**. That
choice is deliberate and is what lets the tolerance stay at
`torch.testing.assert_close`'s fp32 default: the kernel accumulates K terms in
fp32 via `fmaf`, and a fp32 torch reference would be a second approximation in a
different summation order rather than a truth to compare against -- measured on
sm_103, the fp32 reference is FURTHER from fp64 at M=4096 (2.7e-05) than the
kernel is (2.4e-06), so it would have been the less accurate side of the
comparison.

Two properties carry the entry and each has its own case:

  * the TACTIC does not move bits. `selectFmaTileN` returns 1 when `M <= 32` and
    8 otherwise, so the default `tile_n=0` picks a different kernel
    instantiation depending on how many rows were submitted. If that changed a
    row's result there would be no covering rule over `M` at all, and this entry
    could only ever certify a sampled list of row counts.
  * `r` is the RAW square sum. No division by `K`, no `rsqrt`; those belong to
    `activation/mhc_split_sinkhorn`, and a port that folded them in here would
    still produce plausible numbers.

Guard evidence is assertive, never prose: each guarded branch raw-drives the op
with an argument that differs from a correct one in METADATA ONLY -- same
logical values, with any storage the kernel reads past them poisoned here -- and
asserts it is accepted and wrong before asserting the wrapper rejects it.

Nothing on the expected side comes from the op, from `refmods.py`, or from any
other catalog entry.
"""

import os
import subprocess
import sys

import pytest
import torch

from tensorrt_llm._torch.staircase.catalog.gemm.mhc_gemm_sqrsum_fma import (
    K_STEP,
    mhc_gemm_sqrsum_fma,
)

assert torch.cuda.is_available(), "mhc_gemm_sqrsum_fma requires a CUDA device"

#: DeepSeek-V4.1-Flash, from config.json's text_config. The projection maps the
#: flattened `hc_mult * hidden` stream onto the `(2 + hc) * hc` mix block.
V41_N = 24
V41_K = 20480

#: Numerical spot-check buckets. The certified INTERVAL of `M` is proved far more
#: widely by `test_row_results_are_independent_of_m`; these are samples, chosen to
#: straddle `selectFmaTileN`'s `M <= 32` threshold and the block boundaries.
ROWS = [1, 2, 3, 8, 31, 32, 33, 64, 127, 128, 129, 256, 1024, 4096]

#: The covering rule's upper end and its sweep.
M_COVER = 8192
COVER_MS = list(range(0, 130)) + [255, 256, 257, 511, 512, 1023, 1024, 4095, 4096, 8191, 8192]

#: Every `tile_n` the launcher instantiates, plus the two that mean "heuristic".
#: 24 is `N` itself; all eight are divisors of 24, which is why none of them
#: reaches the launcher's `default:` arm at this checkpoint's geometry.
TILE_N_ALL = [0, -1, 1, 2, 3, 4, 6, 8, 12, 24]

_NAMES = ("y", "r")

#: Written into storage the kernel reads only when the metadata is wrong.
POISON_X = 0.5
POISON_W = -3.0


def _operands(m: int, seed: int, n: int = V41_N, k: int = V41_K):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, generator=gen, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, generator=gen, device="cuda", dtype=torch.float32) * 0.02
    return x.contiguous(), w.contiguous()


def _ref(x: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Native-torch reference in float64, independent of the op and of any entry."""
    xd = x.double()
    return (xd @ w.double().T).float(), xd.square().sum(-1).float()


def _raw(x, w, m, n, k, tile_n=0, tile_m=0):
    """Drive the op directly, bypassing every wrapper guard."""
    y = torch.empty(m, n, device="cuda", dtype=torch.float32)
    r = torch.empty(m, device="cuda", dtype=torch.float32)
    torch.ops.trtllm.mhc_gemm_sqrsum_fma(x, w, y, r, m, n, k, tile_n, tile_m)
    torch.cuda.synchronize()
    return y, r


@pytest.mark.parametrize("m", ROWS)
def test_v41_column_by_row_bucket(m: int) -> None:
    """Numerical spot-check at this checkpoint's exact geometry, against a float64 reference.

    The tolerance is `torch.testing.assert_close`'s fp32 default, UNCHANGED. It
    holds because the reference is fp64: the kernel measures ~2.4e-06 max-abs on
    `y` (rel ~2e-07) and ~3.9e-03 on `r` against values of order `K`, both well
    inside the default `rtol=1.3e-6, atol=1e-5`. Nothing here is widened, so no
    discrimination study is owed.
    """
    x, w = _operands(m, 41 * m + 7)
    y, r = mhc_gemm_sqrsum_fma(x, w)
    want_y, want_r = _ref(x, w)
    assert y.shape == (m, V41_N) and r.shape == (m,)
    torch.testing.assert_close(y, want_y)
    torch.testing.assert_close(r, want_r)


def test_r_is_the_raw_square_sum_not_a_norm() -> None:
    """`r = sum(x**2)`, with no `/K` and no `rsqrt` -- the consumer owns those.

    A port that folded the normalization in here would produce numbers in a
    plausible range and would match nothing downstream. Pinned by scale: the raw
    sum is of order `K`, so dividing by `K` or taking a reciprocal square root
    both leave it orders of magnitude away from what this op returns.
    """
    x, w = _operands(64, 1234)
    _, r = mhc_gemm_sqrsum_fma(x, w)
    want = x.double().square().sum(-1).float()
    torch.testing.assert_close(r, want)
    assert r.min() > 0.05 * V41_K, (
        f"r max {r.max().item():.3e} is far below the raw square sum's scale (~K={V41_K}); "
        f"this op appears to be returning a mean or a norm"
    )


def test_zero_rows_are_accepted_and_return_empty() -> None:
    """`M = 0` is a served shape: a dep4 rank can hold no logical rows.

    The launcher early-returns on `M <= 0` rather than forming a zero-sized
    grid, so both outputs come back empty. Synchronizing is the assertion that
    no CUDA error was queued behind it.
    """
    x, w = _operands(0, 24601)
    y, r = mhc_gemm_sqrsum_fma(x, w)
    torch.cuda.synchronize()
    assert y.shape == (0, V41_N), y.shape
    assert r.shape == (0,), r.shape


@pytest.mark.parametrize("tile_n", TILE_N_ALL)
def test_tactic_does_not_change_bits(tile_n: int) -> None:
    """Every instantiated `tile_n`, and both spellings of "heuristic", are BIT-equal.

    This is the case the covering rule over `M` rests on. `selectFmaTileN`
    returns 1 for `M <= 32` and 8 otherwise, so `tile_n=0` silently selects a
    different kernel instantiation according to the batch size. Because each
    block computes the full `K` for its own columns with the same thread mapping
    and the same reduction tree, changing how many columns a block owns cannot
    change an element's summation order -- and that is asserted here rather than
    argued, bit for bit.

    `tile_n = -1` is included because the launcher's test is `tile_n > 0`, so
    every non-positive value means "use the heuristic" rather than being an
    error.
    """
    m = 129
    x, w = _operands(m, 4242)
    base = _raw(x, w, m, V41_N, V41_K, tile_n=1)
    got = _raw(x, w, m, V41_N, V41_K, tile_n=tile_n)
    for name, g, b in zip(_NAMES, got, base):
        assert torch.equal(g, b), (
            f"tile_n={tile_n} changed {name} relative to tile_n=1, so a row's result depends on "
            f"the tactic -- and since the tactic depends on M, no interval of M can be certified"
        )


def test_tile_m_is_discarded_by_the_launcher() -> None:
    """`tile_m` is `(void) tile_m` in the launcher, so the wrapper does not expose it.

    Driven rather than read from the comment: a caller who believed `tile_m` did
    something would get no error and no change. The wrapper's signature omits it
    entirely, and this case is what keeps that omission honest.
    """
    m = 129
    x, w = _operands(m, 555)
    base = _raw(x, w, m, V41_N, V41_K, tile_m=0)
    for tile_m in (1, 7, -3, 4096):
        got = _raw(x, w, m, V41_N, V41_K, tile_m=tile_m)
        for name, g, b in zip(_NAMES, got, base):
            assert torch.equal(g, b), f"tile_m={tile_m} changed {name}; it is not discarded"


def test_row_results_are_independent_of_m() -> None:
    """The covering rule for `M`, proved rather than sampled.

    One reference run at `M = 8192` is checked against float64, and then every
    `M` in a dense sweep from 0 to 129 plus the power-of-two neighbourhoods up
    to 8192 must reproduce that run's first `M` rows BIT-EXACTLY. The sweep
    deliberately contains 31, 32 and 33: `selectFmaTileN` switches tactic at
    `M <= 32`, so if the tactic leaked into the values this is where it would
    show. Any `M` in `[0, 8192]` is therefore certified, including the zero-row
    case and every small decode batch.
    """
    x, w = _operands(M_COVER, 2718)
    big = mhc_gemm_sqrsum_fma(x, w)
    want = _ref(x, w)
    for name, g, wref in zip(_NAMES, big, want):
        torch.testing.assert_close(g, wref, msg=f"M={M_COVER}: {name} is not the reference")

    for m in COVER_MS:
        got = mhc_gemm_sqrsum_fma(x[:m].contiguous(), w)
        for name, g, b in zip(_NAMES, got, big):
            assert torch.equal(g, b[:m]), (
                f"M={m}: {name} differs from the first {m} rows of the M={M_COVER} run, so a "
                f"row's result depends on how many rows were submitted with it"
            )


def test_short_k_below_the_vector_step_is_accepted_and_not_guarded() -> None:
    """The control that keeps the alignment guard from being over-broad.

    `K = 1023` is not a multiple of 4, and it is nonetheless accepted and
    correct, because the vectorized main loop runs only while
    `k_base + K_STEP <= K` -- at `K < 1024` every element goes through the
    scalar 2-byte tail, which is always aligned. A wrapper that required
    `K % 4 == 0` unconditionally would reject this valid call, so the guard is
    conditional and this case is what keeps it that way.
    """
    m, k = 8, K_STEP - 1
    assert k % 4 != 0, "this control is vacuous unless K is genuinely unaligned"
    x, w = _operands(m, 99, k=k)
    y, r = mhc_gemm_sqrsum_fma(x, w)
    want_y, want_r = _ref(x, w)
    torch.testing.assert_close(y, want_y)
    torch.testing.assert_close(r, want_r)


#: The misaligned-`K` evidence runs in a CHILD PROCESS, and that is not
#: fastidiousness. The fault is asynchronous and it poisons the CUDA context:
#: once it fires, every later launch in the process reports `misaligned address`
#: whatever its own shapes are. An earlier revision of the probe measured
#: `K = 2048` as misaligned for exactly that reason -- 2048 is a multiple of 4
#: and perfectly legal, it was only inheriting the previous case's fault. Run
#: in-process here, this single case would fail every test after it.
_MISALIGNED_CHILD = """
import torch, tensorrt_llm._torch.custom_ops  # noqa: F401
k = 1025
g = torch.Generator(device="cuda").manual_seed(7)
x = torch.randn(8, k, generator=g, device="cuda", dtype=torch.bfloat16).contiguous()
w = torch.randn(24, k, generator=g, device="cuda", dtype=torch.float32).contiguous()
y = torch.empty(8, 24, device="cuda", dtype=torch.float32)
r = torch.empty(8, device="cuda", dtype=torch.float32)
try:
    torch.ops.trtllm.mhc_gemm_sqrsum_fma(x, w, y, r, 8, 24, k, 0, 0)
    print("CALL_RAISED=False")
except Exception as exc:
    print("CALL_RAISED=True", type(exc).__name__)
try:
    torch.cuda.synchronize()
    print("SYNC_RAISED=False")
except Exception as exc:
    print("SYNC_RAISED=True", type(exc).__name__, str(exc).splitlines()[0][:60])
"""


def test_misaligned_k_is_silent_at_the_call_then_guarded_by_the_wrapper() -> None:
    """`K % 4 != 0` at `K >= K_STEP` is accepted by the call and kills the context later.

    Both halves of the guard rule hold, which is why this one IS asserted in the
    wrapper while `tile_n` is not. It is pure tensor metadata (`x.shape[1]`), and
    it is silent AT THE CALL SITE: the op returns without raising, and an
    unrelated later `synchronize()` takes the blame with
    `CUDA error: misaligned address`. A caller has nothing to catch at the point
    of the mistake, and everything downstream in the process is then broken.
    """
    out = subprocess.run(
        [sys.executable, "-c", _MISALIGNED_CHILD],
        capture_output=True,
        text=True,
        timeout=600,
        env=os.environ.copy(),
    )
    assert "CALL_RAISED=False" in out.stdout, (
        f"the op call raised on a misaligned K, so this is a reported error and not a silent "
        f"one, and the wrapper guard is not earned.\nstdout:\n{out.stdout}\nstderr tail:\n"
        f"{out.stderr[-800:]}"
    )
    assert "SYNC_RAISED=True" in out.stdout, (
        f"the later synchronize was clean, so a misaligned K is harmless here and the wrapper "
        f"guard rejects a valid call.\nstdout:\n{out.stdout}"
    )
    assert "misaligned address" in out.stdout, f"unexpected fault text:\n{out.stdout}"

    # And the wrapper turns that into a catchable error at the point of the call.
    x, w = _operands(8, 11, k=K_STEP + 1)
    with pytest.raises(AssertionError, match="K must be a multiple of 4"):
        mhc_gemm_sqrsum_fma(x, w)


def test_non_divisor_tile_n_is_rejected_loudly_by_the_op() -> None:
    """The launcher checks `N % tileN` itself, so the wrapper does not repeat it.

    A guard here would only change which error the caller sees, which is the
    case the catalog's rule excludes. The message is quoted because a contract
    that claims a rejection owes the text it was observed to produce.
    """
    m = 64
    x, w = _operands(m, 2024)
    for tile_n in (5, 7, 16):
        with pytest.raises(RuntimeError, match="not divisible by tile_n"):
            _raw(x, w, m, V41_N, V41_K, tile_n=tile_n)


@pytest.mark.parametrize("dtype_case", ["x_fp32", "w_bf16"])
def test_op_rejects_wrong_dtypes_itself(dtype_case: str) -> None:
    """Both dtype domains are loud, so neither is repeated in the wrapper."""
    m = 32
    x, w = _operands(m, 606)
    if dtype_case == "x_fp32":
        args, pattern = (x.float(), w), "expected scalar type BFloat16"
    else:
        args, pattern = (x, w.bfloat16()), "expected scalar type Float"
    with pytest.raises(RuntimeError, match=pattern):
        _raw(args[0], args[1], m, V41_N, V41_K)


# ─── the guard table ─────────────────────────────────────────────────────────
#
# Three cases, all pure tensor metadata, all carrying the correct logical values
# with any storage the kernel reads past them poisoned here. The alignment guard
# is the fourth and lives in its own subprocess case above, because its evidence
# destroys the CUDA context.

_M_GUARD = 64


def _b_noncontiguous_x(x, w):
    arena = torch.full((_M_GUARD, 2 * V41_K), POISON_X, device="cuda", dtype=torch.bfloat16)
    arena[:, :V41_K] = x
    return {"x": arena[:, :V41_K]}


def _b_noncontiguous_w(x, w):
    arena = torch.full((V41_N, 2 * V41_K), POISON_W, device="cuda", dtype=torch.float32)
    arena[:, :V41_K] = w
    return {"w": arena[:, :V41_K]}


def _b_k_disagreement(x, w):
    # `w` keeps only half the columns, so the kernel is handed a K that does not
    # match x's row stride and reads x's rows at the wrong offset.
    narrow = torch.full((V41_N, V41_K // 2), POISON_W, device="cuda", dtype=torch.float32)
    narrow[:, : V41_K // 2] = w[:, : V41_K // 2]
    return {"w": narrow, "k": V41_K // 2}


GUARDS = {
    "noncontiguous_x": (_b_noncontiguous_x, "x must be contiguous"),
    "noncontiguous_w": (_b_noncontiguous_w, "w must be contiguous"),
    "k_disagreement": (_b_k_disagreement, "w must be .N, K. with the SAME K"),
}


@pytest.mark.parametrize("case", sorted(GUARDS))
def test_guard_is_silent_at_the_op_then_rejected_by_the_wrapper(case: str) -> None:
    """The two halves of a guard, in the order that makes it earned rather than assumed.

    The RAW op is driven with metadata-only damage and must not raise, must stay
    finite, and must differ from the correct result -- which is what makes the
    failure silent. Only then is the wrapper asserted to reject the same call.
    """
    x, w = _operands(_M_GUARD, 31337)
    build, message = GUARDS[case]
    args = {"x": x, "w": w, "k": V41_K}
    args.update(build(x, w))

    want = _raw(x, w, _M_GUARD, V41_N, V41_K)
    got = _raw(args["x"], args["w"], _M_GUARD, V41_N, args["k"])
    for name, t in zip(_NAMES, got):
        assert torch.isfinite(t).all(), f"{case}: {name} is non-finite, a different claim"
    assert not all(torch.equal(g, b) for g, b in zip(got, want)), (
        f"{case}: the op returned the correct result, so this metadata is tolerated and the "
        f"wrapper guard is rejecting a valid call"
    )

    with pytest.raises(AssertionError, match=message):
        mhc_gemm_sqrsum_fma(args["x"], args["w"])


def test_rank3_storage_equivalent_inputs_are_correct_and_not_guarded() -> None:
    """The measurement that removed a fifth wrapper assert, kept so it cannot come back.

    An earlier revision asserted `x.dim() == 2 and w.dim() == 2`. Driven raw, a
    storage-equivalent rank-3 `x` or `w` -- same bytes, same order, one extra
    length-1 axis -- is **accepted and correct**, because the op reads
    `data_ptr()` and is told `M`, `N` and `K` explicitly. So that assert was
    rejecting harmless calls, which is a defect in the wrapper rather than a
    safety feature, and it is gone.

    The wrapper's interface is still documented as 2-D, and a wrong-rank caller
    still fails -- but through the tuple unpack that derives the shapes, as a
    `ValueError` about the interface, NOT as an `AssertionError` claiming the op
    mishandles the input. The two are different statements and this case pins
    which one the entry makes.
    """
    m = 64
    x, w = _operands(m, 31337)
    want_y, want_r = _ref(x, w)

    for label, x_arg, w_arg in (
        ("rank-3 x", x.reshape(m, 1, V41_K), w),
        ("rank-3 w", x, w.reshape(V41_N, 1, V41_K)),
    ):
        y, r = _raw(x_arg, w_arg, m, V41_N, V41_K)
        torch.testing.assert_close(y, want_y, msg=f"{label}: y is wrong at the raw op")
        torch.testing.assert_close(r, want_r, msg=f"{label}: r is wrong at the raw op")

    # The documented 2-D interface still fails on wrong rank -- naturally.
    with pytest.raises(ValueError, match="too many values to unpack"):
        mhc_gemm_sqrsum_fma(x.reshape(m, 1, V41_K), w)
    with pytest.raises(ValueError, match="too many values to unpack"):
        mhc_gemm_sqrsum_fma(x, w.reshape(V41_N, 1, V41_K))


def test_extra_x_rows_are_harmless_and_not_guarded() -> None:
    """What the wrapper must NOT reject, and what deriving `M` removes outright.

    Driven raw with an `M` smaller than `x`'s row count, the trailing rows are
    simply never read and the result is bit-equal. The wrapper derives `M` from
    `x.shape[0]`, so this cannot even be expressed through it -- deriving is
    what removes the hazard, rather than a guard.
    """
    m = 64
    x, w = _operands(m + 8, 4242)
    want = _raw(x[:m].contiguous(), w, m, V41_N, V41_K)
    got = _raw(x, w, m, V41_N, V41_K)
    for name, g, b in zip(_NAMES, got, want):
        assert torch.equal(g, b), f"{name}: trailing x rows changed the result, so they are read"


def test_cuda_graph_replay_through_the_wrapper_matches_eager() -> None:
    """Captured at the CATALOG surface the target will actually call.

    The wrapper allocates its two outputs per call, which reads like a barrier to
    capture and is not one: allocations made inside `torch.cuda.graph` come from
    the graph's private pool, so the tensors the capture call returned are the
    same memory every replay rewrites.

    Two replays with DIFFERENT inputs, because a graph that silently replayed its
    captured input would pass a single-replay check.
    """
    m = 129
    x, w = _operands(m, 5150)
    x2, _ = _operands(m, 6161)
    x_in = x.clone()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            mhc_gemm_sqrsum_fma(x_in, w)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y, r = mhc_gemm_sqrsum_fma(x_in, w)

    for src in (x, x2):
        x_in.copy_(src)
        graph.replay()
        torch.cuda.synchronize()
        want_y, want_r = _ref(src, w)
        torch.testing.assert_close(y, want_y)
        torch.testing.assert_close(r, want_r)
