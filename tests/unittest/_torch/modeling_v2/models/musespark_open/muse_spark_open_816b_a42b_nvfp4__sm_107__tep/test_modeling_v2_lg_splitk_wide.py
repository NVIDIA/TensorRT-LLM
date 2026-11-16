# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``lg_splitk`` (stage S3 / S3b port) at 16 and 32 rows on one SM107 GPU: bits, protocol, edge bench.

``ops/gemm/lg_splitk`` (the split-K kernel whose polling warps read the activation from a
sentinel arena before the PDL wait) on the served 9..32-row TEP8 tactics of the three S3 sites
(``dispatcher.wide_steps(8)``): ``o_proj`` ``[R, 1024] x [8192, 1024]^T`` split 1 (arena = the
producer's output, no generation check: the combine releases o_proj after its own wait),
``qkv`` ``[R, 8192] x [2304, 8192]^T`` split 4 (arena = the producer's output, generation check)
and ``pool`` ``[R, 8192] x [512, 8192]^T`` split 4 on two token tiles (arena = a mirror of the
producer's output, generation check). Nothing in the served path imports the kernel yet.

* L1 bits: each site x 16 / 32 rows x 1 / 2 / 4 polling warps equals, bit for bit, the served
  ``splitk_gemm.run_splitk_dense`` at the same tactic (int16 views), inputs ``normal`` and
  ``special`` (+-0, +-inf, NaN payloads, subnormals, +-max); the arena is back at rest (all
  sentinel, counters 0, generations the rest tag) and no warp fell back. ``collide`` inputs (a
  0xFFFE half in the data) fall back and still give the same bits.
* L2 late: a producer that stores 60 us after the GEMM started polling (past the 20 us deadline)
  gives the same bits through the fallback (count > 0); one storing torn words (low halves, then
  high halves 3.5 us later) gives the same bits without one.
* L3 bench (the S3 go/no-go, ``lg_splitk_bench.py`` ported): per site and row count a 62-layer
  chain [producer -> GEMM -> consumer], all PDL, one CUDA graph per arm (stock and lg at each
  polling-warp count), producer compute 0.8 / 1.5 / 3 us before its store, ABBA rounds of CUDA
  event timed replays. Prints ``LGB site rows P pre stock lg delta [ci] per-layer`` and the
  verdict per site and row count (go when the best arm saves >= 0.3 us per edge, the plan's
  no-go bar), checks bits of the last replay, a clean arena and zero fallbacks. JSON to
  ``$LG_BENCH_OUT`` when set. The chain has no cross-rank wait: an upper bound of the served
  edge.
"""

import importlib
import json
import math
import os
import statistics

import pytest
import torch

_TARGET = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open."
    "muse_spark_open_816b_a42b_nvfp4__sm_107__tep"
)


def _supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)


pytestmark = pytest.mark.skipif(not _supported(), reason="the sm107 TEP8 kernels run on SM107")

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


def _module(name: str):
    return importlib.import_module(f"{_TARGET}.{name}")


SENTINEL_I32 = 0xFFFEFFFE - (1 << 32)
#: site -> (K, N), mirror arena, generation check, producer triggers at entry.
SITES = {
    "o_proj": ((1024, 8192), False, False, False),
    "qkv": ((8192, 2304), False, True, True),
    "pool": ((8192, 512), True, True, True),
}
ROWS = (16, 32)
POLL_WARPS = (1, 2, 4)
#: Bench arms ``(polling warps, backoff ns between polling rounds)`` (``LG_BENCH_ARMS`` "P:B,...").
ARMS = tuple(
    tuple(int(v) for v in arm.split(":"))
    for arm in os.environ.get("LG_BENCH_ARMS", "1:0,4:0,4:500,4:2000").split(",")
)
GO_US_PER_EDGE = 0.3
SPECIAL_BF16 = (
    0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x7F81, 0xFFFF, 0x7FFF,
    0x7F7F, 0xFF7F, 0x0080, 0x8080, 0x0001, 0x8001, 0x3F80, 0xBF80, 0x3F81, 0x4040,
)  # fmt: skip


def _tactic(site: str, rows: int):
    lg = _module("ops.gemm.lg_splitk")
    for last, kernel in lg.site_tactics(8)[site]:
        if rows <= last:
            return lg.tactic_of(kernel)
    raise AssertionError(f"no served wide kernel for {site} at {rows} rows")


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _differing(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((_bits(a) != _bits(b)).sum().item())


def _x(rows: int, k: int, kind: str, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((rows, k), generator=g, device="cuda").to(torch.bfloat16)
    if kind in ("special", "collide"):
        flat = x.view(torch.int16).view(-1)
        idx = torch.randperm(flat.numel(), generator=g, device="cuda")[: 4 * len(SPECIAL_BF16)]
        pats = torch.tensor(
            [p - (1 << 16) if p >= 1 << 15 else p for p in SPECIAL_BF16] * 4,
            dtype=torch.int16,
            device="cuda",
        )
        flat[idx] = pats
        if kind == "collide":
            flat[idx[:3]] = torch.tensor(-2, dtype=torch.int16, device="cuda")  # 0xFFFE
    return x


def _w(k: int, n: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(10_000 + seed)
    return (torch.randn((n, k), generator=g, device="cuda") * k**-0.5).to(torch.bfloat16)


@triton.jit
def _now():
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1
    )


@triton.jit
def _producer(
    src,
    dst,
    halves,
    mirror,
    PRE_NS: tl.constexpr,
    TORN: tl.constexpr,
    BLOCK: tl.constexpr,
    EARLY: tl.constexpr,
    MIRROR: tl.constexpr,
):
    """The GEMM's producer: (``EARLY``: trigger,) wait, (else trigger,) compute ``PRE_NS``, store
    ``src`` into ``dst``. ``MIRROR``: ``dst`` is the producer's own output, stored at once after
    the wait, and the arena ``mirror`` gets the same values ``PRE_NS`` later. ``TORN``: low halves
    first, high halves ``PRE_NS / 2 + 1 us`` later."""
    if EARLY:
        tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    if not EARLY:
        tl.extra.cuda.gdc_launch_dependents()
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < halves
    if MIRROR:
        tl.store(dst + offsets, tl.load(src + offsets, mask=mask), mask=mask)
        dst = mirror
    t0 = _now()
    t = t0
    while t - t0 < PRE_NS:
        t = _now()
    value = tl.load(src + offsets, mask=mask)
    if TORN:
        tl.store(dst + offsets, value, mask=mask & (offsets % 2 == 0))
        t0 = _now()
        t = t0
        while t - t0 < PRE_NS // 2 + 1000:
            t = _now()
        tl.store(dst + offsets, value, mask=mask & (offsets % 2 == 1))
    else:
        tl.store(dst + offsets, value, mask=mask)


def _produce(src, dst, pre_ns: int, *, early: bool, mirror=None, torn: bool = False) -> None:
    halves = src.numel()
    _producer[(triton.cdiv(halves, 1024),)](
        src.view(torch.int16),
        dst.view(torch.int16),
        halves,
        (mirror if mirror is not None else dst).view(torch.int16),
        PRE_NS=int(pre_ns),
        TORN=bool(torn),
        BLOCK=1024,
        EARLY=bool(early),
        MIRROR=mirror is not None,
        launch_pdl=True,
    )


@triton.jit
def _xor(a, b):
    return a ^ b


@triton.jit
def _consumer(src, sink, words, BLOCK: tl.constexpr):
    """The GEMM's dependent: wait (the GEMM complete), trigger, read the product."""
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(src + offsets, mask=offsets < words, other=0)
    tl.atomic_xor(sink, tl.reduce(v, 0, _xor), sem="relaxed")


def _consume(src, sink) -> None:
    words = src.numel() * src.element_size() // 4
    _consumer[(triton.cdiv(words, 4096),)](
        src.view(torch.int32), sink, words, BLOCK=4096, launch_pdl=True
    )


def _arena(site: str, rows: int, tactic, rest_tag: int = 1):
    lg = _module("ops.gemm.lg_splitk")
    (k, _n), _mirror, _gen, _early = SITES[site]
    return lg.LgArena(
        rows, k, torch.device("cuda"), slices=tactic.split_k, tiles=rows // tactic.mma_n,
        rest_tag=rest_tag,
    )  # fmt: skip


def _one_call(site, rows, poll_warps, x, w, pre_ns=0, torn=False, graph=False, backoff_ns=0):
    """Producer -> lg GEMM on a fresh arena (one-call cycle); returns (out, arena, fallbacks).
    ``graph``: an eager call (compiles), then the pair captured in a CUDA graph and replayed once
    (no host launch gap between the producer and the GEMM); the fallbacks are the replay's."""
    lg = _module("ops.gemm.lg_splitk")
    (k, n), mirror, gen_check, early = SITES[site]
    tactic = _tactic(site, rows)
    arena = _arena(site, rows, tactic)
    out = torch.empty((rows, n), dtype=torch.bfloat16, device="cuda")
    plain = torch.empty_like(x)

    def pair(pre):
        if mirror:
            _produce(x, plain, pre, early=early, mirror=arena.tensor, torn=torn)
            lg.run_lg_splitk(
                plain, w.t(), out, tactic, arena, mirror=True, gen_check=gen_check,
                poll_warps=poll_warps, backoff_ns=backoff_ns,
            )  # fmt: skip
        else:
            _produce(x, arena.tensor, pre, early=early, torn=torn)
            lg.run_lg_splitk(
                arena.tensor, w.t(), out, tactic, arena, gen_check=gen_check,
                poll_warps=poll_warps, backoff_ns=backoff_ns,
            )  # fmt: skip

    if graph:
        pair(0)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            pair(pre_ns)
        torch.cuda.synchronize()
        out.zero_()
        before = arena.fallbacks()
        g.replay()
    else:
        before = arena.fallbacks()
        pair(pre_ns)
    torch.cuda.synchronize()
    return out, arena, arena.fallbacks() - before


def _reference(site, rows, x, w):
    lg = _module("ops.gemm.lg_splitk")
    (_k, n), _m, _g, _e = SITES[site]
    out = torch.empty((rows, n), dtype=torch.bfloat16, device="cuda")
    lg.stock(x, w.t(), out, _tactic(site, rows))
    torch.cuda.synchronize()
    return out


# --------------------------------------------------------------------------------------------------
# L1: bits and the arena protocol, one call
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("poll_warps, backoff", [(p, 0) for p in POLL_WARPS] + [(4, 500)])
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("site", sorted(SITES))
def test_l1_bits(site, rows, poll_warps, backoff):
    (k, n), _m, _g, _e = SITES[site]
    tactic = _tactic(site, rows)
    print(f"L1 {site} rows {rows} P {poll_warps} backoff {backoff}: tactic {tactic}", flush=True)
    for seed, kind in enumerate(("normal", "special", "normal", "special")):
        x = _x(rows, k, kind, seed)
        w = _w(k, n, seed)
        ref = _reference(site, rows, x, w)
        out, arena, fallbacks = _one_call(site, rows, poll_warps, x, w, backoff_ns=backoff)
        assert _differing(out, ref) == 0, f"{site} {rows} P{poll_warps} {kind}: bits differ"
        assert fallbacks == 0, f"{site} {rows} P{poll_warps} {kind}: {fallbacks} fallbacks"
        assert arena.is_clean(), f"{site} {rows} P{poll_warps} {kind}: arena not at rest"
    x = _x(rows, k, "collide", 99)
    w = _w(k, n, 99)
    ref = _reference(site, rows, x, w)
    out, arena, fallbacks = _one_call(site, rows, poll_warps, x, w, backoff_ns=backoff)
    assert fallbacks > 0, f"{site} {rows} P{poll_warps} collide: no fallback"
    assert _differing(out, ref) == 0, f"{site} {rows} P{poll_warps} collide: bits differ"
    assert arena.is_clean(), f"{site} {rows} P{poll_warps} collide: arena not at rest"


# --------------------------------------------------------------------------------------------------
# L2: late and torn producers
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("site", sorted(SITES))
def test_l2_late_and_torn(site, rows):
    (k, n), _m, _g, _e = SITES[site]
    for poll_warps in (1, 4):
        x = _x(rows, k, "special", 7)
        w = _w(k, n, 7)
        ref = _reference(site, rows, x, w)
        out, arena, fallbacks = _one_call(site, rows, poll_warps, x, w, pre_ns=60_000, graph=True)
        # A fallback needs the GEMM resident while the producer computes (not guaranteed when the
        # producer's CTAs fill the device): printed, not asserted (L1's collide case asserts it).
        print(
            f"L2 {site} rows {rows} P{poll_warps}: 60 us producer, {fallbacks} fallbacks",
            flush=True,
        )
        assert _differing(out, ref) == 0 and arena.is_clean()
        out, arena, fallbacks = _one_call(
            site, rows, poll_warps, x, w, pre_ns=5_000, torn=True, graph=True
        )
        assert fallbacks == 0, f"{site} {rows} P{poll_warps}: torn words fell back"
        assert _differing(out, ref) == 0 and arena.is_clean()


# --------------------------------------------------------------------------------------------------
# L3: the edge bench (go/no-go)
# --------------------------------------------------------------------------------------------------
LAYERS = int(os.environ.get("LG_BENCH_LAYERS", "62"))
PRE_NS = tuple(int(v) for v in os.environ.get("LG_BENCH_PRE_NS", "800,1500,3000").split(","))
ROUNDS = int(os.environ.get("LG_BENCH_ROUNDS", "10"))
REPLAYS = int(os.environ.get("LG_BENCH_REPLAYS", "50"))


def _build(site: str, rows: int, pre_ns: int, weights, srcs):
    lg = _module("ops.gemm.lg_splitk")
    (k, n), mirror, gen_check, early = SITES[site]
    tactic = _tactic(site, rows)
    tags = {i: (i + 1, LAYERS if i == 0 else i) for i in range(LAYERS)}
    arms = ["stock"] + [f"lg{p}b{b}" for p, b in ARMS]
    arenas = {a: _arena(site, rows, tactic, rest_tag=LAYERS) for a in arms if a != "stock"}
    plain = torch.empty((rows, k), dtype=torch.bfloat16, device="cuda")
    sink = torch.zeros(1, dtype=torch.int32, device="cuda")
    outs = {
        a: [torch.empty((rows, n), dtype=torch.bfloat16, device="cuda") for _ in range(LAYERS)]
        for a in arms
    }

    def chain(arm: str) -> None:
        for i in range(LAYERS):
            if arm == "stock":
                _produce(srcs[i], plain, pre_ns, early=early)
                lg.stock(plain, weights[i].t(), outs[arm][i], tactic)
            else:
                arena = arenas[arm]
                tag, expect = tags[i]
                p, b = (int(v) for v in arm[2:].split("b"))
                if mirror:
                    _produce(srcs[i], plain, pre_ns, early=early, mirror=arena.tensor)
                    lg.run_lg_splitk(
                        plain, weights[i].t(), outs[arm][i], tactic, arena, tag=tag,
                        expect=expect, mirror=True, gen_check=gen_check, poll_warps=p,
                        backoff_ns=b,
                    )  # fmt: skip
                else:
                    _produce(srcs[i], arena.tensor, pre_ns, early=early)
                    lg.run_lg_splitk(
                        arena.tensor, weights[i].t(), outs[arm][i], tactic, arena, tag=tag,
                        expect=expect, gen_check=gen_check, poll_warps=p, backoff_ns=b,
                    )  # fmt: skip
            _consume(outs[arm][i], sink)

    graphs = {}
    for arm in arms:
        chain(arm)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            chain(arm)
        graphs[arm] = g
    return graphs, outs, arenas, tactic


def _time(graph, replays: int) -> list:
    times = []
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(replays):
        start.record()
        graph.replay()
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop) * 1000.0)
    return times


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("site", sorted(SITES))
def test_l3_edge_bench(site, rows):
    (k, n), _m, _g, _e = SITES[site]
    weights = [_w(k, n, i) for i in range(LAYERS)]
    srcs = [_x(rows, k, "normal", 1000 + i) for i in range(LAYERS)]
    results = []
    failures = []
    for pre in PRE_NS:
        graphs, outs, arenas, tactic = _build(site, rows, pre, weights, srcs)
        for g in graphs.values():
            for _ in range(10):
                g.replay()
        torch.cuda.synchronize()
        before = {a: arena.fallbacks() for a, arena in arenas.items()}
        arms = tuple(graphs)
        per_round = {a: [] for a in arms}
        for r in range(ROUNDS):
            order = arms + arms[::-1] if r % 2 == 0 else arms[::-1] + arms
            acc = {a: [] for a in arms}
            for a in order:
                acc[a] += _time(graphs[a], REPLAYS // 2)
            for a in acc:
                per_round[a].append(statistics.median(acc[a]))
        for g in graphs.values():
            g.replay()
        torch.cuda.synchronize()
        stock_us = statistics.median(per_round["stock"])
        for a in arms[1:]:
            diffs = [y - x for x, y in zip(per_round["stock"], per_round[a])]
            mean = statistics.mean(diffs)
            half = 1.96 * statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
            same = all(_differing(o, s) == 0 for o, s in zip(outs[a], outs["stock"]))
            fallbacks = arenas[a].fallbacks() - before[a]
            clean = arenas[a].is_clean()
            row = dict(
                site=site, rows=rows, tactic=str(tactic), arm=a, pre_ns=pre,
                stock_us=stock_us, lg_us=statistics.median(per_round[a]), delta_us=mean,
                ci95=[mean - half, mean + half], per_edge_us=mean / LAYERS, fallbacks=fallbacks,
                bits_equal=same, arena_at_rest=clean,
            )  # fmt: skip
            results.append(row)
            print(
                f"LGB {site} rows {rows} {a} pre {pre}: stock {stock_us:.1f} us "
                f"lg {row['lg_us']:.1f} us delta {mean:+.1f} [{mean - half:+.1f}, {mean + half:+.1f}] "
                f"= {mean / LAYERS:+.3f} us/edge; fallbacks {fallbacks}; bits "
                f"{'equal' if same else 'DIFFER'}; arena at rest {clean}",
                flush=True,
            )
            if not same or not clean or fallbacks:
                failures.append(f"{a} pre {pre}: bits {same} clean {clean} fallbacks {fallbacks}")
        del graphs
    best = {}
    for row in results:
        best.setdefault(row["arm"], []).append(-row["per_edge_us"])
    saves = {a: statistics.mean(v) for a, v in best.items()}
    a_best = max(saves, key=saves.get)
    verdict = "GO" if saves[a_best] >= GO_US_PER_EDGE else "NO-GO"
    print(
        f"LGB-VERDICT {site} rows {rows}: best {a_best} saves {saves[a_best]:.3f} us/edge "
        f"(mean over PRE_NS {PRE_NS}); bar {GO_US_PER_EDGE} -> {verdict}; per arm "
        + ", ".join(f"{a} {v:+.3f}" for a, v in sorted(saves.items())),
        flush=True,
    )
    out = os.environ.get("LG_BENCH_OUT")
    if out:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, f"lgb_{site}_{rows}.json"), "w") as handle:
            json.dump({"verdict": verdict, "saves": saves, "rows": results}, handle, indent=1)
    assert not failures, "; ".join(failures)
