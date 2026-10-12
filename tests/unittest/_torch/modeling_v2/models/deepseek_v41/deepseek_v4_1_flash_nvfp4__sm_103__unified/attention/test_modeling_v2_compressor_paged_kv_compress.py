# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the compressor_paged_kv_compress catalog entry.

The entry owns what is certified: `CELLS` is the list, `reference` and `compare` are the gate, and
`test_certified_cells` drives every cell through them, each step of each cell gated against
`op.reference` taken from the pools as they were before the call. The cells are the DeepSeek-V4.1
ratio-2 compressor of the KV-source layers 2/8/14 at generation time -- compress_ratio 2, head_dim
512, fp32 kv_score rows ``[kv || score]``, fp32 paged state, zero APE (V4.1 has none), fp32 or bf16
output -- for plain decode (next_n 1) and DSpark verify (next_n 6), next_n 1..8 in all.

This file keeps the checks a cell cannot carry: DSpark verify with per-request acceptance rewinding
the next step's start, the prefill op's parked token consumed in one shared buffer, CUDA-graph
replay, the real cache manager's pools and metadata slices (generation-only and mixed steps), the
op's own rejections, the wrong-variant controls, the silent-misuse guards, and the engine-size
coverage (every batch size at every table width, every context lead, the capture set, offset
hazards, pool-size invariance, the multi-request envelope, every dumped step, the DSpark tail).

Those engine-size checks gate on ``_pool_windows``: the checkpoint's own decode rule (its
inference/model.py, Compressor.forward, start_pos > 0 branch -- token ``p`` fills slot ``p % 2`` and,
when ``(p + 1) % 2 == 0``, the pair is pooled per channel as ``sum(kv * softmax_over_the_pair(score))``)
over a per-position array holding what each position last received. It is independent of the pool
layout, which is what lets it run on the engine's pools (up to 65 GB) that `op.reference`, which
returns whole expected pools, cannot copy.

State geometry. DeepseekV4CacheManager in V4.1 mode puts COMPRESSOR_KV and
COMPRESSOR_SCORE of every owner layer in ONE fp32 pool ``[pages, 128, 512]``
(same base pointer; the kv and score tables name different pages of it), so
the synthetic tests share one tensor between the two arguments by default.
Synthetic pools have POOL_PAGES pages -- chosen plainly larger than any batch
here -- with a random page permutation, distinct pages for kv and score, and a
trap page behind every unused table column. test_v41_cache_manager_pools runs
the op on the real manager's pools, tables and per-step metadata instead.
"""

import math
import zlib
from types import SimpleNamespace
from typing import List, Sequence, Tuple

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.attention.compressor_paged_kv_compress import (  # noqa: E501
    compressor_paged_kv_compress as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "compressor_paged_kv_compress requires a CUDA device"

DEV = torch.device("cuda")
HEAD_DIM = 512
RATIO = 2
POOL_PAGES = 1024
TRAP_PAGE = 0
SENTINEL = 7777.0


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    return g


def _rand_rows(n: int, gen: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp32 kv and score rows. Channel 0 is an exact tie in every pair and channel
    1 a gap of 240 between even and odd positions (exp underflows to exactly 0)."""
    kv = torch.randn(n, HEAD_DIM, generator=gen, device=DEV) * 2.0
    score = torch.randn(n, HEAD_DIM, generator=gen, device=DEV) * 4.0
    score[:, 0] = 1.5
    score[:, 1] = 120.0 * (1 - 2 * (torch.randint(0, 2, (n,), generator=gen, device=DEV).float()))
    return kv, score


def _pool_windows(
    kv: torch.Tensor, score: torch.Tensor, ape: torch.Tensor, w0: int, w1: int
) -> torch.Tensor:
    """model.py's ratio-2 pooling of windows [w0, w1) of a per-position array, fp32."""
    p = torch.arange(w0 * RATIO, w1 * RATIO, device=DEV)
    k = kv[p].unflatten(0, (w1 - w0, RATIO))
    s = (score[p] + ape[p % RATIO]).unflatten(0, (w1 - w0, RATIO))
    return (k * s.softmax(dim=1)).sum(dim=1)


def _zero_ape() -> torch.Tensor:
    return torch.zeros(RATIO, HEAD_DIM, device=DEV)


def _i32(values) -> torch.Tensor:
    return torch.tensor(list(values), device=DEV, dtype=torch.int32)


class _Rows:
    """Per-position rows of one request over absolute positions [lo, hi), stored from lo, so a
    request whose history reaches position 1,048,594 costs only its window. Indexed with
    absolute positions (int, tensor or slice), like a [hi, HEAD_DIM] tensor would be."""

    def __init__(self, lo: int, hi: int):
        self.lo, self.hi = lo, hi
        self.t = torch.zeros(hi - lo, HEAD_DIM, device=DEV)

    def _k(self, k):
        if isinstance(k, slice):
            start = self.lo if k.start is None else k.start
            stop = self.hi if k.stop is None else k.stop
            assert self.lo <= start <= stop <= self.hi, (start, stop, self.lo, self.hi)
            return slice(start - self.lo, stop - self.lo)
        if isinstance(k, torch.Tensor):
            return k - self.lo
        assert self.lo <= k < self.hi, (k, self.lo, self.hi)
        return k - self.lo

    def __getitem__(self, k):
        return self.t[self._k(k)]

    def __setitem__(self, k, v):
        self.t[self._k(k)] = v

    def clone(self) -> "_Rows":
        c = _Rows.__new__(_Rows)
        c.lo, c.hi, c.t = self.lo, self.hi, self.t.clone()
        return c


class _Pool:
    """Paged fp32 compressor state and per-sequence kv / score block tables.

    ``shared=True`` is the V4.1 manager's arrangement: kv and score live in one
    tensor, on different pages."""

    def __init__(self, max_lens: Sequence[int], page_size: int, seed: int, shared: bool = True):
        self.page_size = page_size
        self.kv = torch.full((POOL_PAGES, page_size, HEAD_DIM), SENTINEL, device=DEV)
        self.score = self.kv if shared else torch.full_like(self.kv, SENTINEL)
        n_log = [max(math.ceil(L / page_size), 1) for L in max_lens]
        width = max(n_log) + 1
        cpu = torch.Generator().manual_seed(seed)
        pages = (torch.randperm(POOL_PAGES - 1, generator=cpu) + 1).tolist()
        assert 2 * sum(n_log) <= len(pages), "raise POOL_PAGES"
        self.bt_kv = torch.full((len(max_lens), width), TRAP_PAGE, dtype=torch.int32)
        self.bt_score = torch.full_like(self.bt_kv, TRAP_PAGE)
        for b, n in enumerate(n_log):
            self.bt_kv[b, :n] = torch.tensor(pages[:n], dtype=torch.int32)
            del pages[:n]
            self.bt_score[b, :n] = torch.tensor(pages[:n], dtype=torch.int32)
            del pages[:n]
        self.bt_kv = self.bt_kv.to(DEV)
        self.bt_score = self.bt_score.to(DEV)

    def slots(self, table: torch.Tensor, b: int, pos: torch.Tensor):
        return table[b].long()[pos // self.page_size], pos % self.page_size

    def seed(
        self,
        b: int,
        kv: torch.Tensor,
        score: torch.Tensor,
        ape: torch.Tensor,
        upto: int,
        poison: bool,
    ):
        """The state a prefill call leaves for positions [0, upto): kv and score + ape.
        With ``poison`` every position but ``upto - 1`` holds NaN instead."""
        if upto == 0:
            return
        pos = torch.arange(upto, device=DEV)
        kv_v = kv[:upto].clone()
        sc_v = score[:upto] + ape[pos % RATIO]
        if poison:
            kv_v[:-1] = float("nan")
            sc_v[:-1] = float("nan")
        self.kv[self.slots(self.bt_kv, b, pos)] = kv_v
        self.score[self.slots(self.bt_score, b, pos)] = sc_v

    def check(self, kvs, scores, lens, ape: torch.Tensor, first: Sequence[int]):
        """Positions [first[b], lens[b]) of sequence b hold its kv and score + ape bit
        for bit, and no slot outside the sequences' positions [0, lens[b]) changed."""
        touched = torch.zeros(POOL_PAGES, self.page_size, dtype=torch.bool, device=DEV)
        touched_sc = torch.zeros_like(touched)
        for b, (kv, sc, n, f) in enumerate(zip(kvs, scores, lens, first)):
            pos = torch.arange(n, device=DEV)
            pk, ps = self.slots(self.bt_kv, b, pos), self.slots(self.bt_score, b, pos)
            touched[pk] = True
            touched_sc[ps] = True
            live = torch.arange(f, n, device=DEV)
            assert torch.equal(self.kv[self.slots(self.bt_kv, b, live)], kv[f:n]), (
                f"seq {b}: paged kv state"
            )
            assert torch.equal(
                self.score[self.slots(self.bt_score, b, live)], sc[f:n] + ape[live % RATIO]
            ), f"seq {b}: paged score state"
        if self.score is self.kv:
            touched |= touched_sc
            assert (self.kv[~touched] == SENTINEL).all(), (
                "pool written outside the sequences' slots"
            )
        else:
            assert (self.kv[~touched] == SENTINEL).all(), (
                "kv pool written outside the sequences' slots"
            )
            assert (self.score[~touched_sc] == SENTINEL).all(), (
                "score pool written outside the sequences' slots"
            )


def _step(
    pool: _Pool,
    kvs,
    scores,
    starts: Sequence[int],
    next_n: int,
    ape,
    out_dtype=torch.float32,
    lead_rows: int = 0,
    lead_out: int = 0,
    certify: bool = False,
):
    """One generation step: request b feeds positions [starts[b], starts[b] + next_n).

    ``lead_rows`` / ``lead_out`` put NaN rows ahead of the generation rows in
    kv_score and ahead of the outputs, where a mixed batch's context rows sit:
    the op must neither read the former nor write the latter. The output buffer
    has the runtime's reserved size (ceil(next_n / 2) rows per request) plus one
    guard row, and everything past the packed outputs must stay NaN.

    ``certify`` gates the call on the entry's own ``reference`` (taken from the pools before the
    call) and ``compare``, and checks the op returns the three tensors it wrote."""
    B = len(starts)
    rows = [
        torch.cat([kv[s : s + next_n], sc[s : s + next_n]], dim=1)
        for kv, sc, s in zip(kvs, scores, starts)
    ]
    lead = torch.full((lead_rows, 2 * HEAD_DIM), float("nan"), device=DEV)
    kv_score = torch.cat([lead] + rows, dim=0).contiguous()
    n_out = [(s + next_n) // RATIO - s // RATIO for s in starts]
    cu_out = [lead_out]
    for o in n_out:
        cu_out.append(cu_out[-1] + o)
    reserved = B * math.ceil(next_n / RATIO)
    buf = torch.full((lead_out + reserved + 1, HEAD_DIM), float("nan"), device=DEV, dtype=out_dtype)
    output = buf[: lead_out + reserved]
    args = (
        kv_score,
        ape,
        pool.kv,
        pool.score,
        pool.bt_kv,
        pool.bt_score,
        output,
        _i32(s + next_n for s in starts),
        _i32(lead_rows + b * next_n for b in range(B + 1)),
        _i32(cu_out),
        B,
        pool.page_size,
        HEAD_DIM,
        RATIO,
        next_n,
    )
    expected = op.reference(*args) if certify else None
    result = op(*args)
    torch.cuda.synchronize()
    if certify:
        assert result[0] is output and result[1] is pool.kv and result[2] is pool.score, (
            "the op writes in place and returns (output, paged_kv, paged_score)"
        )
        op.compare(result, expected)
    assert torch.isnan(buf[:lead_out]).all(), "op wrote into the rows ahead of cu_kv_comp[0]"
    assert torch.isnan(buf[cu_out[-1] :]).all(), "op wrote past the last packed output row"
    return [output[cu_out[b] : cu_out[b + 1]] for b in range(B)]


def _decode_run(
    cached: List[int],
    steps: int,
    next_n: int,
    page_size: int,
    ape,
    out_dtype,
    seed: int,
    shared: bool = True,
    poison_history: bool = False,
    lead: Tuple[int, int] = (0, 0),
    certify: bool = False,
):
    """``steps`` all-accepted generation steps after a seeded history; returns per-step
    outputs and the per-position arrays the reference pools from."""
    gen = _gen(seed)
    lens = [c + steps * next_n for c in cached]
    seqs = [_rand_rows(L, gen) for L in lens]
    kvs, scores = [s[0] for s in seqs], [s[1] for s in seqs]
    pool = _Pool(lens, page_size, seed, shared)
    for b, c in enumerate(cached):
        pool.seed(b, kvs[b], scores[b], ape, c, poison_history)
    outs = []
    starts = list(cached)
    for _ in range(steps):
        got = _step(pool, kvs, scores, starts, next_n, ape, out_dtype, *lead, certify=certify)
        outs.append((list(starts), got))
        starts = [s + next_n for s in starts]
    return pool, kvs, scores, lens, outs


def _check_outputs(kvs, scores, ape, outs, next_n):
    for starts, got in outs:
        for b, (s, g) in enumerate(zip(starts, got)):
            ref = _pool_windows(kvs[b], scores[b], ape, s // RATIO, (s + next_n) // RATIO)
            assert g.shape == ref.shape, (
                f"seq {b} start {s}: {tuple(g.shape)} vs {tuple(ref.shape)}"
            )
            # Default fp32 tolerances (rtol 1.3e-6, atol 1e-5): the kernel is built with
            # --use_fast_math and runs an online softmax that divides once at the end, so it
            # is not bit-identical to torch's softmax-then-sum, but stays inside the dtype's
            # own tolerance.
            torch.testing.assert_close(g, ref, msg=lambda m: f"seq {b} start {s}: {m}")


# Histories that put the first new token at both parities and on both sides of a
# 128- and a 256-token page boundary.
CACHED = [1, 2, 127, 128, 255, 256, 1001, 8191]


# ---------------------------------------------------------------------------
# certified surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """Every step of every cell gated on the entry's reference and compare: the step's rows, the
    whole pool(s) bitwise, and every output row the op must leave alone. A step after the first
    reads the token the op itself parked. At the end every position holds what it last received
    (with a poisoned history: from start - 1 on) and no other slot of the pool changed."""
    certifying(op)
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    if spec["ape"] == "zero":
        ape = _zero_ape()
    else:
        ape = torch.randn(RATIO, HEAD_DIM, generator=_gen(seed + 1), device=DEV) * 2.0
    cached = list(spec["cached"])
    pool, kvs, scores, lens, _ = _decode_run(
        cached,
        spec["steps"],
        spec["next_n"],
        spec["page_size"],
        ape,
        spec["out_dtype"],
        seed=seed,
        shared=spec["shared_pool"],
        poison_history=spec["poison_history"],
        lead=(spec["lead_rows"], spec["lead_out"]),
        certify=True,
    )
    first = [c - 1 if c else 0 for c in cached] if spec["poison_history"] else [0] * len(lens)
    pool.check(kvs, scores, lens, ape, first=first)


def test_dspark_verify_with_rejection():
    """next_n = 6 (DSpark: 5 drafts + 1). After every step each request accepts a
    random 0..5 drafts, so the next step starts at start + 1 + accepted, at either
    parity, and rewrites the rejected positions with new values. Each step must
    pool from what every position last received -- the accepted token at start - 1
    included -- which is exactly the per-token decode rule applied in order."""
    ape = _zero_ape()
    next_n, steps, page_size = 6, 10, 128
    cached = [1, 34, 127, 128, 129, 294, 1094, 4095]
    gen = _gen(401)
    cpu = torch.Generator().manual_seed(401)
    cap = [c + steps * next_n for c in cached]
    kvs = [torch.zeros(L, HEAD_DIM, device=DEV) for L in cap]
    scores = [torch.zeros(L, HEAD_DIM, device=DEV) for L in cap]
    for b, c in enumerate(cached):
        kvs[b][:c], scores[b][:c] = _rand_rows(c, gen)
    pool = _Pool(cap, page_size, 401)
    for b, c in enumerate(cached):
        pool.seed(b, kvs[b], scores[b], ape, c, poison=False)
    starts = list(cached)
    parities = set()
    for _ in range(steps):
        for b, s in enumerate(starts):
            kvs[b][s : s + next_n], scores[b][s : s + next_n] = _rand_rows(next_n, gen)
            parities.add(s % RATIO)
        got = _step(pool, kvs, scores, starts, next_n, ape)
        _check_outputs(kvs, scores, ape, [(list(starts), got)], next_n)
        accepted = torch.randint(0, next_n, (len(starts),), generator=cpu).tolist()
        starts = [s + 1 + a for s, a in zip(starts, accepted)]
    assert parities == {0, 1}
    # Every position up to the last one written holds what it last received.
    pool.check(
        kvs,
        scores,
        [s - 1 + next_n - a for s, a in zip(starts, accepted)],
        ape,
        first=[0] * len(starts),
    )


def test_prefill_state_then_decode_in_one_buffer():
    """The engine's mixed step: the prefill op pools the contexts into the first
    rows of one output buffer and parks their odd tail token in the pool; the
    decode op runs the generation requests behind them in the same kv_score and
    output. The next steps decode everyone, the former contexts included, so the
    tail token the prefill op parked is the pending token decode reads back."""
    ape = _zero_ape()
    next_n, page_size = 6, 128
    ctx_lens = [301, 130, 7]
    gen_cached = [5, 128, 999]
    steps = 3
    gen = _gen(601)
    cap = [L + steps * next_n for L in ctx_lens] + [c + (steps + 1) * next_n for c in gen_cached]
    seqs = [_rand_rows(L, gen) for L in cap]
    kvs, scores = [s[0] for s in seqs], [s[1] for s in seqs]
    pool = _Pool(cap, page_size, 601)
    nctx = len(ctx_lens)
    for j, c in enumerate(gen_cached):
        pool.seed(nctx + j, kvs[nctx + j], scores[nctx + j], ape, c, poison=False)

    # Step 1: contexts through the prefill op, generation requests through this op.
    ctx_rows = [torch.cat([kvs[b][:L], scores[b][:L]], dim=1) for b, L in enumerate(ctx_lens)]
    gen_rows = [
        torch.cat([kvs[nctx + j][c : c + next_n], scores[nctx + j][c : c + next_n]], dim=1)
        for j, c in enumerate(gen_cached)
    ]
    kv_score = torch.cat(ctx_rows + gen_rows, dim=0).contiguous()
    ctx_out = [L // RATIO for L in ctx_lens]
    gen_out = [(c + next_n) // RATIO - c // RATIO for c in gen_cached]
    cu_seq = [0]
    for n in ctx_lens + [next_n] * len(gen_cached):
        cu_seq.append(cu_seq[-1] + n)
    cu_out = [0]
    for o in ctx_out + gen_out:
        cu_out.append(cu_out[-1] + o)
    total = sum(ctx_out) + len(gen_cached) * math.ceil(next_n / RATIO)
    buf = torch.full((total + 1, HEAD_DIM), float("nan"), device=DEV)
    output = buf[:total]
    torch.ops.trtllm.compressor_prefill_reduction(
        kv_score[: cu_seq[nctx]],
        ape,
        pool.kv,
        pool.score,
        pool.bt_kv[:nctx],
        pool.bt_score[:nctx],
        output,
        _i32(ctx_lens),
        _i32([0] * nctx),
        _i32(cu_seq[: nctx + 1]),
        _i32(cu_out[: nctx + 1]),
        nctx,
        page_size,
        HEAD_DIM,
        RATIO,
        max(ctx_out),
    )
    op(
        kv_score,
        ape,
        pool.kv,
        pool.score,
        pool.bt_kv[nctx:],
        pool.bt_score[nctx:],
        output,
        _i32(c + next_n for c in gen_cached),
        _i32(cu_seq[nctx:]),
        _i32(cu_out[nctx:]),
        len(gen_cached),
        page_size,
        HEAD_DIM,
        RATIO,
        next_n,
    )
    torch.cuda.synchronize()
    assert torch.isnan(buf[cu_out[-1] :]).all(), "write past the packed outputs"
    for b, L in enumerate(ctx_lens):
        torch.testing.assert_close(
            output[cu_out[b] : cu_out[b + 1]], _pool_windows(kvs[b], scores[b], ape, 0, L // 2)
        )
    for j, c in enumerate(gen_cached):
        b = nctx + j
        torch.testing.assert_close(
            output[cu_out[b] : cu_out[b + 1]],
            _pool_windows(kvs[b], scores[b], ape, c // 2, (c + next_n) // 2),
        )

    # Steps 2..: everyone decodes.
    starts = list(ctx_lens) + [c + next_n for c in gen_cached]
    assert any(s % RATIO for s in starts[:nctx]), "a context must leave a pending token"
    outs = []
    for _ in range(steps):
        outs.append((list(starts), _step(pool, kvs, scores, starts, next_n, ape)))
        starts = [s + next_n for s in starts]
    _check_outputs(kvs, scores, ape, outs, next_n)
    pool.check(kvs, scores, starts, ape, first=[0] * len(starts))


def test_bf16_output_against_a_bf16_operand_control():
    """bf16 output = round-to-nearest-even of the fp32 pooled value, i.e. the
    reference's ``kv.to(bf16)`` before its RMSNorm. The bf16 cells gate it call by call;
    this pools next_n 1 and 6 into one population, the size the 0.999 gate was set on, and
    shows a bf16-operand control misses that gate."""
    ape = _zero_ape()
    got, ref, wrong = [], [], []
    for next_n in (1, 6):
        _, kvs, scores, _, outs = _decode_run(
            CACHED, 6, next_n, 128, ape, torch.bfloat16, seed=700 + next_n
        )
        for starts, rows in outs:
            for b, (s, g) in enumerate(zip(starts, rows)):
                w0, w1 = s // 2, (s + next_n) // 2
                got.append(g)
                ref.append(_pool_windows(kvs[b], scores[b], ape, w0, w1).to(torch.bfloat16))
                wrong.append(
                    _pool_windows(
                        kvs[b].bfloat16().float(), scores[b].bfloat16().float(), ape, w0, w1
                    ).to(torch.bfloat16)
                )
    got, ref, wrong = torch.cat(got), torch.cat(ref), torch.cat(wrong)
    torch.testing.assert_close(got, ref)
    exact = (got == ref).float().mean().item()
    print(f"bf16 output: {exact:.6f} of {got.numel()} elements bit-identical to the reference")
    # The fp32 value before the cast is within a few fp32 ulp of the reference, so a
    # bf16 element can differ only where it straddles a bf16 rounding midpoint
    # (~1e-4 expected). 0.999 is ~10x headroom; the bf16-operand control lands far below.
    assert exact >= 0.999, f"bf16 output bit-exact fraction {exact}"
    frac = (got == wrong).float().mean().item()
    print(f"control bf16_operands vs bf16 output: bit-exact fraction {frac:.6f}")
    assert frac < 0.999


def test_cuda_graph_replay():
    """Captured once per next_n, replayed over steps whose kv_score, kv_lens and
    cu_kv_comp change in place (batch, next_n and the tables stay fixed)."""
    ape = _zero_ape()
    for next_n in (1, 6):
        steps = 6
        cached = [1, 128, 255, 1001]
        B = len(cached)
        gen = _gen(901 + next_n)
        lens = [c + (steps + 1) * next_n for c in cached]
        seqs = [_rand_rows(L, gen) for L in lens]
        kvs, scores = [s[0] for s in seqs], [s[1] for s in seqs]
        pool = _Pool(lens, 128, 901)
        reserved = B * math.ceil(next_n / RATIO)
        s_kv_score = torch.zeros(B * next_n, 2 * HEAD_DIM, device=DEV)
        s_kv_lens = torch.zeros(B, dtype=torch.int32, device=DEV)
        s_cu_seq = _i32(b * next_n for b in range(B + 1))
        s_cu_out = torch.zeros(B + 1, dtype=torch.int32, device=DEV)
        s_out = torch.zeros(reserved, HEAD_DIM, device=DEV)

        def load(starts):
            s_kv_score.copy_(
                torch.cat(
                    [
                        torch.cat([kv[s : s + next_n], sc[s : s + next_n]], 1)
                        for kv, sc, s in zip(kvs, scores, starts)
                    ]
                )
            )
            s_kv_lens.copy_(_i32(s + next_n for s in starts))
            n_out = [(s + next_n) // 2 - s // 2 for s in starts]
            s_cu_out.copy_(_i32([0] + torch.tensor(n_out).cumsum(0).tolist()))
            return n_out

        def call():
            op(
                s_kv_score,
                ape,
                pool.kv,
                pool.score,
                pool.bt_kv,
                pool.bt_score,
                s_out,
                s_kv_lens,
                s_cu_seq,
                s_cu_out,
                B,
                128,
                HEAD_DIM,
                RATIO,
                next_n,
            )

        # Warm up and capture on step "-1" (positions [c - next_n, c)); the replayed
        # steps then start from a pool whose history is fully written.
        for b, c in enumerate(cached):
            pool.seed(b, kvs[b], scores[b], ape, c, poison=False)
        load(cached)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            call()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        # Replays advance from the (already correct) state after the eager warm-up
        # step; the captured run itself wrote nothing (capture does not execute).
        starts = [c + next_n for c in cached]
        for _ in range(steps):
            n_out = load(starts)
            s_out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            off = 0
            for b, s in enumerate(starts):
                ref = _pool_windows(kvs[b], scores[b], ape, s // 2, (s + next_n) // 2)
                torch.testing.assert_close(s_out[off : off + n_out[b]], ref)
                off += n_out[b]
            assert torch.isnan(s_out[off:]).all()
            starts = [s + next_n for s in starts]


def test_v41_cache_manager_pools():
    """The op on DeepseekV4CacheManager's own V4.1 compressor-state pools: kv and
    score pools of owners 2/8/14 from ``get_buffers``, tables from the metadata's
    ``sliding_block_tables[layer_offsets[owner], COMPRESSOR_KV / COMPRESSOR_SCORE]``,
    and ``kv_lens_cuda_runtime``, ``cu_seq_lens_cuda``, ``cu_new_comp_kv_cuda[2]``,
    ``num_gen_tokens_per_seq``, ``num_total_compressed_tokens[2]`` and
    ``tokens_per_block`` of each step's prepared metadata. Plain decode (no
    speculation, window 2) and DSpark verify (max_draft_len 5, window 7, next_n 6).
    Every page the tables map for these requests is NaN except the pending token
    at start - 1, and no other byte of the shared pool may change."""
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import (
        DeepseekV4CacheManager,
        DeepseekV4TrtllmAttentionMetadata,
    )
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType as T,
    )
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm.bindings import DataType, SamplingConfig
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import (
        DeepSeekV4SparseAttentionConfig,
        DSparkDecodingConfig,
        KvCacheConfig,
    )
    from tensorrt_llm.mapping import Mapping

    # The checkpoint's text_config fields the V4.1 layout reads.
    text = dict(
        model_type="deepseek_v41_text",
        num_hidden_layers=40,
        num_nextn_predict_layers=3,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        candidate_source_layer_id=20,
        engram_layer_ids=[1, 14],
        engram_max_ngram_size=4,
        sliding_window=128,
        head_dim=HEAD_DIM,
        index_head_dim=128,
        index_topk=512,
        kv_lora_rank=448,
        qk_rope_head_dim=64,
    )
    config = SimpleNamespace(
        architectures=["DeepseekV41ForCausalLM"],
        model_type="deepseek_v41",
        text_config=SimpleNamespace(**text),
        **{k: v for k, v in text.items() if k != "model_type"},
    )
    ape = _zero_ape()
    steps = 3
    for max_draft_len, cached in ((0, [1, 127, 128, 299, 2048]), (5, [34, 127, 128, 129, 1094])):
        next_n = max_draft_len + 1
        sparse = DeepSeekV4SparseAttentionConfig(
            index_head_dim=128, window_size=128, compress_ratios=[1] * 43
        )
        manager = DeepseekV4CacheManager(
            kv_cache_config=KvCacheConfig(
                enable_block_reuse=False, max_tokens=4096 * 8, event_buffer_max_size=0
            ),
            kv_cache_type=CacheType.SELFKONLY,
            num_layers=40,
            num_kv_heads=1,
            head_dim=HEAD_DIM,
            tokens_per_block=128,
            max_seq_len=4096,
            max_batch_size=8,
            max_input_len=4096,
            mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            vocab_size=129280,
            max_num_tokens=8192,
            sparse_attn_config=sparse,
            pretrained_config=config,
            spec_config=DSparkDecodingConfig(max_draft_len=max_draft_len, block_size=max_draft_len)
            if max_draft_len
            else None,
        )
        requests = []
        try:
            assert manager.v41_layout.window(T.COMPRESSOR_KV, max_draft_len) == 2 + max_draft_len
            B = len(cached)
            cap = [c + steps * next_n for c in cached]
            for i, L in enumerate(cap):
                req = LlmRequest(
                    request_id=i,
                    max_new_tokens=64,
                    input_tokens=list(range(L)),
                    sampling_config=SamplingConfig(),
                    is_streaming=False,
                )
                assert manager.prepare_context(req)
                assert manager.resize_context(req, req.context_chunk_size)
                requests.append(req)
            gen = _gen(1001 + max_draft_len)
            seqs = [_rand_rows(L, gen) for L in cap]
            kvs, scores = [s[0] for s in seqs], [s[1] for s in seqs]
            for owner in (2, 8, 14):
                pool_kv = manager.get_buffers(owner, T.COMPRESSOR_KV)
                pool_sc = manager.get_buffers(owner, T.COMPRESSOR_SCORE)
                assert pool_kv.dtype == torch.float32 and tuple(pool_kv.shape[1:]) == (
                    128,
                    HEAD_DIM,
                )
                assert pool_kv.data_ptr() == pool_sc.data_ptr(), (
                    "V4.1 keeps kv and score in one pool"
                )
                assert pool_kv.shape == pool_sc.shape
                starts = list(cached)
                for step in range(steps):
                    md = DeepseekV4TrtllmAttentionMetadata(
                        seq_lens=torch.tensor([next_n] * B, dtype=torch.int),
                        request_ids=list(range(B)),
                        max_num_requests=manager.max_batch_size,
                        num_contexts=0,
                        prompt_lens=[s + next_n for s in starts],
                        max_num_tokens=8192,
                        kv_cache_manager=manager,
                        kv_cache_params=KVCacheParams(
                            use_cache=True, num_cached_tokens_per_seq=list(starts)
                        ),
                        mapping=Mapping(world_size=1, tp_size=1, rank=0),
                        sparse_attention_config=sparse,
                    )
                    md.prepare()
                    torch.cuda.synchronize()
                    assert md.num_gen_tokens_per_seq == next_n
                    lo = manager.layer_offsets[owner]
                    bt_kv = md.sliding_block_tables[lo, T.COMPRESSOR_KV.value][:B]
                    bt_sc = md.sliding_block_tables[lo, T.COMPRESSOR_SCORE.value][:B]
                    page = manager.tokens_per_block
                    if step == 0:
                        # Poison every mapped page of these rows, then park the pending token.
                        for b, s in enumerate(starts):
                            for t in (bt_kv[b], bt_sc[b]):
                                mapped = t[t >= 0].long()
                                pool_kv[mapped] = float("nan")
                            if s % RATIO:
                                p = s - 1
                                pool_kv[bt_kv[b, p // page].long(), p % page] = kvs[b][p]
                                pool_sc[bt_sc[b, p // page].long(), p % page] = (
                                    scores[b][p] + ape[p % RATIO]
                                )
                    for b, s in enumerate(starts):
                        for p in range(s - (s % RATIO), s + next_n):
                            assert (
                                int(bt_kv[b, p // page]) >= 0 and int(bt_sc[b, p // page]) >= 0
                            ), (b, p)
                    before = pool_kv.clone()
                    kv_lens = md.kv_lens_cuda_runtime[:B]
                    assert kv_lens.tolist() == [s + next_n for s in starts]
                    cu_seq = md.cu_seq_lens_cuda[: B + 1]
                    cu_out = md.cu_new_comp_kv_cuda[RATIO][: B + 1]
                    total = md.num_total_compressed_tokens[RATIO]
                    buf = torch.full((total + 1, HEAD_DIM), float("nan"), device=DEV)
                    kv_score = torch.cat(
                        [
                            torch.cat([kvs[b][s : s + next_n], scores[b][s : s + next_n]], 1)
                            for b, s in enumerate(starts)
                        ]
                    ).contiguous()
                    op(
                        kv_score,
                        ape,
                        pool_kv,
                        pool_sc,
                        bt_kv,
                        bt_sc,
                        buf[:total],
                        kv_lens,
                        cu_seq,
                        cu_out,
                        B,
                        page,
                        HEAD_DIM,
                        RATIO,
                        md.num_gen_tokens_per_seq,
                    )
                    torch.cuda.synchronize()
                    cu = cu_out.tolist()
                    for b, s in enumerate(starts):
                        ref = _pool_windows(kvs[b], scores[b], ape, s // 2, (s + next_n) // 2)
                        torch.testing.assert_close(
                            buf[cu[b] : cu[b + 1]], ref, msg=lambda m: f"owner {owner} b {b}: {m}"
                        )
                    assert torch.isnan(buf[cu[-1] :]).all()
                    # Exactly the new positions' kv and score slots changed.
                    expect = before.clone()
                    for b, s in enumerate(starts):
                        pos = torch.arange(s, s + next_n, device=DEV)
                        expect[bt_kv[b].long()[pos // page], pos % page] = kvs[b][pos]
                        expect[bt_sc[b].long()[pos // page], pos % page] = (
                            scores[b][pos] + ape[pos % RATIO]
                        )
                    assert torch.equal(pool_kv.view(torch.int32), expect.view(torch.int32)), (
                        f"owner {owner}: pool"
                    )
                    starts = [s + next_n for s in starts]
        finally:
            for req in requests:
                manager.free_resources(req)
            manager.shutdown()


# Mixed steps on the real manager: (cached, new) of the context requests, which come
# first in the batch, then the generation requests' histories, then max_draft_len.
# ctx3_gen32 is the target's widest decode graph batch (32 generation rows) behind
# two fresh prompts (even 300 and odd 257 tokens) and a chunked context whose chunk
# starts mid-page (513) with an odd end; ctx1tok_gen1 has a context that emits no
# compressed row, so the generation rows start at output row 0 while their kv_score
# rows start at 1; the chunked case has an odd cached count (a pair straddling the
# chunk boundary) and a history ending at max_seq_len - 95; the verify case puts
# DSpark rows (next_n 6) behind contexts.
_MIXED_GEN32 = torch.randint(
    1, 400, (28,), generator=torch.Generator().manual_seed(2024)
).tolist() + [127, 128, 2047, 2048]
_MIXED_STEPS = {
    "ctx3_gen32": ([(0, 300), (0, 257), (513, 388)], _MIXED_GEN32, 0),
    "ctx1tok_gen1": ([(0, 1)], [2047], 0),
    "ctx2_gen5_chunked": ([(255, 301), (1024, 476)], [1, 2, 129, 1000, 4000], 0),
    "ctx2_verify4": ([(0, 130), (513, 300)], [34, 127, 128, 1094], 5),
}
_MIXED_MAX_BATCH = 40
_MIXED_MAX_TOKENS = 4096 * _MIXED_MAX_BATCH * 2


@pytest.mark.parametrize("scenario", list(_MIXED_STEPS))
def test_v41_cache_manager_mixed_step_behind_contexts(scenario):
    """The target's call on a real mixed step: DeepseekV4CacheManager (V4.1 mode,
    tokens_per_block 128, max_batch_size 40, max_seq_len 4096, max_tokens 327680;
    the shared compressor-state pool it allocates is fp32 [1062, 128, 512]) and a
    DeepseekV4TrtllmAttentionMetadata prepared with the context requests ahead of the
    generation rows. The op gets exactly the target's arguments: the step's whole
    kv_score [T, 1024] (context rows NaN -- they belong to the prefill op and must not
    be read), the generation slices ``[nc:B]`` of the owner's COMPRESSOR_KV / SCORE
    sliding tables, ``kv_lens_cuda_runtime``, ``cu_seq_lens_cuda`` (B + 1 entries) and
    ``cu_new_comp_kv_cuda[2]``, and an output buffer of
    ``num_total_compressed_tokens[2]`` rows whose context rows hold what the prefill
    op wrote (a sentinel here). fp32 and bf16 (the target's) output, owners 2/8/14.
    The whole shared pool is NaN except each odd-start request's pending token."""
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import DeepseekV4CacheManager
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import (
        DeepSeekV4SparseAttentionConfig,
        DSparkDecodingConfig,
        KvCacheConfig,
    )
    from tensorrt_llm.mapping import Mapping

    ctx, gen_cached, max_draft_len = _MIXED_STEPS[scenario]
    text = dict(
        model_type="deepseek_v41_text",
        num_hidden_layers=40,
        num_nextn_predict_layers=3,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        candidate_source_layer_id=20,
        engram_layer_ids=[1, 14],
        engram_max_ngram_size=4,
        sliding_window=128,
        head_dim=HEAD_DIM,
        index_head_dim=128,
        index_topk=512,
        kv_lora_rank=448,
        qk_rope_head_dim=64,
    )
    config = SimpleNamespace(
        architectures=["DeepseekV41ForCausalLM"],
        model_type="deepseek_v41",
        text_config=SimpleNamespace(**text),
        **{k: v for k, v in text.items() if k != "model_type"},
    )
    sparse = DeepSeekV4SparseAttentionConfig(
        index_head_dim=128, window_size=128, compress_ratios=[1] * 43
    )
    manager = DeepseekV4CacheManager(
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False, max_tokens=_MIXED_MAX_TOKENS, event_buffer_max_size=0
        ),
        kv_cache_type=CacheType.SELFKONLY,
        num_layers=40,
        num_kv_heads=1,
        head_dim=HEAD_DIM,
        tokens_per_block=128,
        max_seq_len=4096,
        max_batch_size=_MIXED_MAX_BATCH,
        max_input_len=4096,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.BF16,
        compressor_dtype=DataType.FLOAT,
        vocab_size=129280,
        max_num_tokens=8192,
        sparse_attn_config=sparse,
        pretrained_config=config,
        spec_config=DSparkDecodingConfig(max_draft_len=max_draft_len, block_size=max_draft_len)
        if max_draft_len
        else None,
    )
    try:
        _mixed_step_on_manager(manager, scenario, ctx, gen_cached, max_draft_len)
    finally:
        manager.shutdown()


def _mixed_step_on_manager(manager, scenario, ctx, gen_cached, max_draft_len, max_num_tokens=8192):
    """One mixed step on ``manager``: contexts ``ctx`` = [(cached, new)] ahead of
    generation requests with histories ``gen_cached`` at next_n = max_draft_len + 1,
    prepared by DeepseekV4TrtllmAttentionMetadata and handed to the op exactly as the
    target hands it (see test_v41_cache_manager_mixed_step_behind_contexts)."""
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import (
        DeepseekV4TrtllmAttentionMetadata,
    )
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType as T,
    )
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm.bindings import SamplingConfig
    from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig
    from tensorrt_llm.mapping import Mapping

    next_n = max_draft_len + 1
    sparse = DeepSeekV4SparseAttentionConfig(
        index_head_dim=128, window_size=128, compress_ratios=[1] * 43
    )
    requests = []
    try:
        cached = [c for c, _ in ctx] + list(gen_cached)
        new = [n for _, n in ctx] + [next_n] * len(gen_cached)
        nc, B = len(ctx), len(cached)
        G = B - nc
        assert {c % RATIO for c in gen_cached} == {0, 1} or G == 1
        for i, (c, n) in enumerate(zip(cached, new)):
            req = LlmRequest(
                request_id=i,
                max_new_tokens=64,
                input_tokens=list(range(c + n)),
                sampling_config=SamplingConfig(),
                is_streaming=False,
            )
            assert manager.prepare_context(req), f"manager could not hold request {i}"
            assert manager.resize_context(req, req.context_chunk_size)
            requests.append(req)
        md = DeepseekV4TrtllmAttentionMetadata(
            seq_lens=torch.tensor(new, dtype=torch.int),
            request_ids=list(range(B)),
            max_num_requests=manager.max_batch_size,
            num_contexts=nc,
            prompt_lens=[c + n for c, n in zip(cached, new)],
            max_num_tokens=max_num_tokens,
            kv_cache_manager=manager,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=list(cached)),
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            sparse_attention_config=sparse,
        )
        md.prepare()
        torch.cuda.synchronize()

        # The offsets the op consumes start past the context part of the step.
        ctx_tokens = sum(n for _, n in ctx)
        ctx_rows = sum((c + n) // RATIO - c // RATIO for c, n in ctx)
        num_tokens = sum(new)
        assert (md.num_contexts, md.num_seqs, md.num_ctx_tokens) == (nc, B, ctx_tokens)
        assert md.num_gen_tokens_per_seq == next_n
        kv_lens = md.kv_lens_cuda_runtime[:B]
        cu_seq = md.cu_seq_lens_cuda[: B + 1]
        cu_out = md.cu_new_comp_kv_cuda[RATIO][: B + 1]
        total = md.num_total_compressed_tokens[RATIO]
        assert kv_lens.tolist() == [c + n for c, n in zip(cached, new)]
        assert cu_seq.tolist() == [0] + torch.tensor(new).cumsum(0).tolist()
        cu = cu_out.tolist()
        assert cu[nc] == ctx_rows and total == ctx_rows + G * math.ceil(next_n / RATIO)
        assert cu[B] <= total

        gen = _gen(1300 + len(scenario))
        kvs, scores = _truth_rows(gen_cached, next_n, gen)
        kv_score = torch.full((num_tokens, 2 * HEAD_DIM), float("nan"), device=DEV)
        for j, c in enumerate(gen_cached):
            r0 = ctx_tokens + j * next_n
            kv_score[r0 : r0 + next_n] = torch.cat(
                [kvs[j][c : c + next_n], scores[j][c : c + next_n]], 1
            )
        ape = _zero_ape()
        page = manager.tokens_per_block
        pool_kv = manager.get_buffers(2, T.COMPRESSOR_KV)
        for owner in (2, 8, 14):
            p_kv = manager.get_buffers(owner, T.COMPRESSOR_KV)
            p_sc = manager.get_buffers(owner, T.COMPRESSOR_SCORE)
            assert (
                p_kv.data_ptr() == p_sc.data_ptr() == pool_kv.data_ptr()
                and p_kv.shape == pool_kv.shape
            )
            assert p_kv.dtype == torch.float32 and tuple(p_kv.shape[1:]) == (128, HEAD_DIM)
            lo = manager.layer_offsets[owner]
            bt_kv = md.sliding_block_tables[lo, T.COMPRESSOR_KV.value]
            bt_sc = md.sliding_block_tables[lo, T.COMPRESSOR_SCORE.value]
            p_kv.fill_(float("nan"))
            for j, c in enumerate(gen_cached):
                b = nc + j
                for p in range(c - (c % RATIO), c + next_n):
                    assert int(bt_kv[b, p // page]) >= 0 and int(bt_sc[b, p // page]) >= 0, (
                        owner,
                        b,
                        p,
                    )
                if c % RATIO:
                    p = c - 1
                    p_kv[bt_kv[b, p // page].long(), p % page] = kvs[j][p]
                    p_sc[bt_sc[b, p // page].long(), p % page] = scores[j][p] + ape[p % RATIO]
            for out_dtype in (torch.float32, torch.bfloat16):
                buf = torch.full((total + 1, HEAD_DIM), float("nan"), device=DEV, dtype=out_dtype)
                buf[:ctx_rows] = SENTINEL
                ctx_before = buf[:ctx_rows].clone()
                before = p_kv.clone()
                op(
                    kv_score,
                    ape,
                    p_kv,
                    p_sc,
                    bt_kv[nc:B],
                    bt_sc[nc:B],
                    buf[:total],
                    kv_lens[nc:B],
                    cu_seq[nc : B + 1],
                    cu_out[nc : B + 1],
                    G,
                    page,
                    HEAD_DIM,
                    RATIO,
                    md.num_gen_tokens_per_seq,
                )
                torch.cuda.synchronize()
                assert torch.equal(buf[:ctx_rows], ctx_before), (
                    f"{scenario} owner {owner}: context output rows written"
                )
                assert torch.isnan(buf[cu[B] :]).all(), (
                    f"{scenario} owner {owner}: rows past the packed outputs"
                )
                got, ref = [], []
                for j, c in enumerate(gen_cached):
                    b = nc + j
                    got.append(buf[cu[b] : cu[b + 1]].float())
                    ref.append(
                        _pool_windows(kvs[j], scores[j], ape, c // RATIO, (c + next_n) // RATIO)
                    )
                    assert got[-1].shape == ref[-1].shape, (scenario, owner, b)
                got, ref = torch.cat(got), torch.cat(ref)
                assert got.shape[0] == cu[B] - cu[nc] and got.shape[0] > 0
                if out_dtype == torch.float32:
                    # Default fp32 tolerances, as in _check_outputs.
                    torch.testing.assert_close(
                        got, ref, msg=lambda m: f"{scenario} owner {owner}: {m}"
                    )
                else:
                    torch.testing.assert_close(
                        got.bfloat16(),
                        ref.bfloat16(),
                        msg=lambda m: f"{scenario} owner {owner}: {m}",
                    )
                    # Same gate as test_bf16_output: RNE of an fp32 value a few ulp from the reference.
                    exact = (got.bfloat16() == ref.bfloat16()).float().mean().item()
                    assert exact >= 0.999, (
                        f"{scenario} owner {owner}: bf16 bit-exact fraction {exact}"
                    )
                # Exactly the generation requests' new kv and score slots changed.
                expect = before.clone()
                for j, c in enumerate(gen_cached):
                    b = nc + j
                    pos = torch.arange(c, c + next_n, device=DEV)
                    expect[bt_kv[b].long()[pos // page], pos % page] = kvs[j][pos]
                    expect[bt_sc[b].long()[pos // page], pos % page] = (
                        scores[j][pos] + ape[pos % RATIO]
                    )
                assert torch.equal(p_kv.view(torch.int32), expect.view(torch.int32)), (
                    f"{scenario} owner {owner}: pool"
                )
            print(
                f"[mixed {scenario} owner {owner}] nc={nc} G={G} next_n={next_n} ctx_tokens={ctx_tokens} "
                f"ctx_rows={ctx_rows} gen_rows={cu[B] - cu[nc]} total={total} pool_pages={p_kv.shape[0]} "
                f"table_width={bt_kv.shape[-1]} max_kv_len={max(c + n for c, n in zip(cached, new))}"
            )

        # Control: the same call with the generation slices taken from row 0, as a caller that
        # forgot the contexts ahead would pass them. The checks above must see it.
        if nc and ctx_rows:
            lo = manager.layer_offsets[2]
            bt_kv = md.sliding_block_tables[lo, T.COMPRESSOR_KV.value]
            bt_sc = md.sliding_block_tables[lo, T.COMPRESSOR_SCORE.value]
            buf = torch.full((total + 1, HEAD_DIM), float("nan"), device=DEV)
            buf[:ctx_rows] = SENTINEL
            op(
                kv_score,
                ape,
                pool_kv,
                pool_kv,
                bt_kv[:G],
                bt_sc[:G],
                buf[:total],
                kv_lens[:G],
                cu_seq[: G + 1],
                cu_out[: G + 1],
                G,
                page,
                HEAD_DIM,
                RATIO,
                next_n,
            )
            torch.cuda.synchronize()
            ctx_hit = int((buf[:ctx_rows] != SENTINEL).any(-1).sum())
            gen_rows = buf[cu[nc] : cu[B]]
            bad_gen = int((~torch.isfinite(gen_rows)).any(-1).sum())
            print(
                f"[mixed {scenario} control: slices from row 0] context rows overwritten {ctx_hit}/{ctx_rows}, "
                f"generation rows not produced {bad_gen}/{cu[B] - cu[nc]}"
            )
            assert ctx_hit > 0 or bad_gen > 0, (
                "the mixed-step checks cannot see un-offset generation slices"
            )
    finally:
        for req in requests:
            manager.free_resources(req)


def test_rejected_by_the_op():
    """Ratio 1 (V4.1's layer 20, which does no pooling), next_n 0 and a head_dim
    without an instantiation raise from the launcher."""
    pk = torch.zeros(4, 128, HEAD_DIM, device=DEV)
    bt = torch.arange(4, device=DEV, dtype=torch.int32).view(1, 4)

    def call(ratio, head_dim, next_n):
        op(
            torch.zeros(max(next_n, 1), 2 * HEAD_DIM, device=DEV),
            torch.zeros(RATIO, HEAD_DIM, device=DEV),
            pk,
            pk,
            bt,
            bt,
            torch.zeros(4, head_dim, device=DEV),
            _i32([4]),
            _i32([0, max(next_n, 1)]),
            _i32([0, 4]),
            1,
            128,
            head_dim,
            ratio,
            next_n,
        )

    with pytest.raises(
        RuntimeError, match=r"only supports compress_ratio 2 \(V4\.1\), 4 or 128, got 1"
    ):
        call(1, HEAD_DIM, 1)
    with pytest.raises(RuntimeError, match=r"next_n must be in \[1, "):
        call(RATIO, HEAD_DIM, 0)
    with pytest.raises(RuntimeError, match="no matching instantiation for HD=256"):
        call(RATIO, 256, 1)


# ---------------------------------------------------------------------------
# discrimination: wrong variants must fail the same gate
# ---------------------------------------------------------------------------


def test_wrong_variants_fail_the_gate():
    ape = _zero_ape()
    got, ref, wrong = (
        [],
        [],
        {
            k: []
            for k in (
                "wrong_parity",
                "softmax_over_channels",
                "uniform_mean",
                "pending_token_dropped",
                "bf16_operands",
            )
        },
    )
    for next_n in (1, 6):
        _, kvs, scores, _, outs = _decode_run(
            CACHED, 6, next_n, 128, ape, torch.float32, seed=1100 + next_n
        )
        for starts, rows in outs:
            for b, (s, g) in enumerate(zip(starts, rows)):
                w0, w1 = s // 2, (s + next_n) // 2
                if w1 == w0:
                    continue
                kv, sc = kvs[b], scores[b]
                got.append(g)
                ref.append(_pool_windows(kv, sc, ape, w0, w1))
                p = torch.arange(w0 * 2, w1 * 2, device=DEV)
                k, sg = kv[p].unflatten(0, (-1, 2)), sc[p].unflatten(0, (-1, 2))
                # Emitting when p % 2 == 0 instead: each row pools positions (2k - 1, 2k).
                q = (p - 1).clamp(min=0)
                wrong["wrong_parity"].append(
                    (kv[q].unflatten(0, (-1, 2)) * sc[q].unflatten(0, (-1, 2)).softmax(1)).sum(1)
                )
                wrong["softmax_over_channels"].append((k * sg.softmax(dim=-1)).sum(1))
                wrong["uniform_mean"].append(k.mean(1))
                # A kernel that ignores the pool pools a straddling pair from its new token alone.
                r = _pool_windows(kv, sc, ape, w0, w1).clone()
                if s % 2:
                    r[0] = kv[s]
                wrong["pending_token_dropped"].append(r)
                wrong["bf16_operands"].append(
                    _pool_windows(kv.bfloat16().float(), sc.bfloat16().float(), ape, w0, w1)
                )
    got, ref = torch.cat(got), torch.cat(ref)
    torch.testing.assert_close(got, ref)
    print(
        f"correct reference: max abs err {(got - ref).abs().max().item():.3e} over {got.shape[0]} rows"
    )
    for name, rows in wrong.items():
        w = torch.cat(rows)
        print(f"control {name}: max abs diff {(got - w).abs().max().item():.3e}")
        with pytest.raises(AssertionError):
            torch.testing.assert_close(got, w)


# ---------------------------------------------------------------------------
# misuse: what the op accepts silently, and what the wrapper therefore rejects
# ---------------------------------------------------------------------------


def test_silent_misuse_is_rejected_by_the_wrapper():
    ape = _zero_ape()
    next_n = 6
    gen = _gen(1201)
    starts = [127, 128, 1001]
    lens = [s + next_n for s in starts]
    seqs = [_rand_rows(L, gen) for L in lens]
    kvs, scores = [s[0] for s in seqs], [s[1] for s in seqs]
    ref = torch.cat(
        [
            _pool_windows(kv, sc, ape, s // 2, (s + next_n) // 2)
            for kv, sc, s in zip(kvs, scores, starts)
        ]
    )
    n_rows = ref.shape[0]
    kv_score = torch.cat(
        [
            torch.cat([kv[s : s + next_n], sc[s : s + next_n]], 1)
            for kv, sc, s in zip(kvs, scores, starts)
        ]
    ).contiguous()

    def args(ks, out):
        pool = _Pool(lens, 128, 1201)
        for b, s in enumerate(starts):
            pool.seed(b, kvs[b], scores[b], ape, s, poison=False)
        n_out = [(s + next_n) // 2 - s // 2 for s in starts]
        return (
            ks,
            ape,
            pool.kv,
            pool.score,
            pool.bt_kv,
            pool.bt_score,
            out,
            _i32(lens),
            _i32(b * next_n for b in range(len(starts) + 1)),
            _i32([0] + torch.tensor(n_out).cumsum(0).tolist()),
            len(starts),
            128,
            HEAD_DIM,
            RATIO,
            next_n,
        )

    wide_in = torch.zeros(kv_score.shape[0], 3 * HEAD_DIM, device=DEV)
    wide_in[:, : 2 * HEAD_DIM] = kv_score
    wide_out = torch.zeros(n_rows, 2 * HEAD_DIM, device=DEV)
    cases = {
        "fp16_kv_score": (kv_score.half(), lambda: torch.zeros(n_rows, HEAD_DIM, device=DEV)),
        "fp16_output": (
            kv_score,
            lambda: torch.zeros(n_rows, HEAD_DIM, device=DEV, dtype=torch.float16),
        ),
        "strided_kv_score": (
            wide_in[:, : 2 * HEAD_DIM],
            lambda: torch.zeros(n_rows, HEAD_DIM, device=DEV),
        ),
        "strided_output": (kv_score, lambda: wide_out[:, :HEAD_DIM]),
    }
    for name, (ks, make_out) in cases.items():
        out = make_out()
        op.raw_call(*args(ks, out))  # outside validating: accepted, no error
        torch.cuda.synchronize()
        diff = (out.float() - ref).abs().max().item()
        print(f"raw op, {name}: returned normally, max abs diff vs reference {diff:.3e}")
        assert diff > 1.0, f"{name}: expected a silently wrong result from the bare op"
        match = "float32 or bfloat16" if name.startswith("fp16") else "must be contiguous"
        with pytest.raises(AssertionError, match=match):
            with validating(op):
                op(*args(ks, make_out()))


# ---------------------------------------------------------------------------
# The engine's own pool, every batch size, every context lead, the gates' capture
# sets
# ---------------------------------------------------------------------------
#
# The engine is the DeepSeek-V4.1 core target's accuracy gate: tep4,
# max_seq_len 16384, max_batch_size 32, max_num_tokens 8192, tokens_per_block
# 128, block reuse on, a per-rank KV quota of 72.68959879875183 GiB (the
# gate's log line "KV cache manager v2 device quota set to ..."). The manager
# built below at that quota reproduces the pool shapes the gate's runtime-value
# dump names (asserted in _engine_manager), so its compressor-state pool is the
# engine's pool, and every test in this section runs on it.

_ENGINE_QUOTA_BYTES = int(round(72.68959879875183 * (1 << 30)))
_ENGINE_MAX_SEQ = 16384
# DSpark (max_draft_len 5, overlap on) grows the manager's max_seq_len by 5 + 4 + 5.
_ENGINE_MAX_SEQ_DSPARK = 16398
# Pool shapes of the gate's runtime-value dump (rank 0): the SWA pool and
# the ratio-2 / ratio-1 COMPRESS views the fake-quant ops were handed.
_ENGINE_SWA_PAGES = 137440
_ENGINE_COMPRESS_R2_PAGES = {453118, 453119, 453120}
_ENGINE_COMPRESS_R1_PAGES = 151040
_ENGINE_MANAGERS = {}

# The largest compressor-state pool any engine at the gate settings can be
# handed. kv_cache_fraction 0.5 takes half of the FREE device memory after the weights
# load, so half of the TOTAL device memory (GB300: 297,020,948,480 B) bounds every quota
# an engine can pick. At that quota the manager's shared fp32 state pool has these
# pages (DSpark off / on x the accuracy gate's max_batch_size 32 and the boot gate's 8, at
# max_num_tokens 8192), with bf16 KV containers and with the native uint8 KV storage
# (TRTLLM_DSV41_KV_STORAGE=native), whose smaller SWA / main-KV entries
# leave the state pool more of the quota. The largest over the gate configurations (native,
# max_num_tokens 8192, batch 8, DSpark off) is 133248 pages (34.9 GB, past element offset 2^33);
# 94464 is the native gates' largest. The engine envelope's B_max (172416) is below.
_ENVELOPE_QUOTA_BYTES = 148_510_474_240
_ENVELOPE_STATE_PAGES = {
    ("bf16", 0, 32): 43518,
    ("bf16", 5, 32): 40446,
    ("bf16", 0, 8): 78078,
    ("bf16", 5, 8): 75516,
    ("native", 0, 32): 94464,
    ("native", 5, 32): 85884,
    ("native", 0, 8): 133248,
    ("native", 5, 8): 125952,
}
_ENVELOPE_B_MAX_GATES = max(_ENVELOPE_STATE_PAGES.values())

# The engine envelope (per rank): max_num_tokens up to 16384, max_batch_size up to 256,
# max_seq_len L in {16384, 32768, 65536, 131072, 262144, 1048576}, DSpark on (max_draft_len 5:
# the manager's length is Lm = L + 14 and the CUDA-graph metadata adds 5 more) and off. The
# compressor-state tables are the manager's max_blocks_per_seq columns wide: 132 / 260 / 516 /
# 1028 / 2052 / 8196 for those L, with and without DSpark (measured on managers of the envelope).
# Shared compressor-state pools at the same 0.5 x total-memory quota, measured maxima over that
# envelope (max_batch_size sampled, not swept):
# - one unsplit manager (960 tuples per storage mode):
#   native [172416, 128, 512] (max_num_tokens 16384, max_batch_size 2, DSpark off, L 16384), bf16
#   120192 (16384, 1, DSpark);
# - a PP2 stage manager built alone: stage 0
#   holds layers 0..19, so the compressor state of owners 2 / 8 / 14 gets a larger share of the quota
#   -- native [247806, 128, 512] (16384, 2, DSpark off, L 16384) and bf16 [188796, 128, 512] (16384,
#   1, DSpark, L 16384). Stage 1 (layers 20..39) holds no compressor state. 247806 is B_max.
_ENVELOPE_TOKENS = 16384
_ENVELOPE_REQS = 256
_ENVELOPE_L = (16384, 32768, 65536, 131072, 262144, 1048576)
_ENVELOPE_WIDTHS = (132, 260, 516, 1028, 2052, 8196)
_ENVELOPE_WIDTH_OF_L = dict(zip(_ENVELOPE_L, _ENVELOPE_WIDTHS))
_GSET = tuple(range(1, 33)) + (40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 192, 256)
assert len(_GSET) == 46
_ENVELOPE_STATE_PAGES[("native", 0, 2, 16384)] = 172416
_ENVELOPE_STATE_PAGES[("bf16", 5, 1, 16384)] = 120192
_ENVELOPE_STATE_PAGES[("native", 0, 2, 16384, 0)] = 247806  # PP2 stage 0
_ENVELOPE_STATE_PAGES[("bf16", 5, 1, 16384, 0)] = 188796  # PP2 stage 0
_UNSPLIT_B_MAX = 172416
_ENVELOPE_B_MAX = 247806
_ENVELOPE_BMAX_MANAGER = dict(
    max_draft_len=0,
    max_batch_size=2,
    quota=_ENVELOPE_QUOTA_BYTES,
    storage="native",
    max_num_tokens=16384,
    pp_stage=0,
)


def _envelope_kv_max(L: int, next_n: int) -> int:
    """The largest kv length after a generation step at max_seq_len L: L without DSpark (next_n 1);
    with DSpark (next_n 6) the manager's Lm = L + 14 plus the CUDA-graph metadata's 5 draft tokens."""
    return L + 14 + 5 if next_n > 1 else L


def _v41_pretrained_config():
    text = dict(
        model_type="deepseek_v41_text",
        num_hidden_layers=40,
        num_nextn_predict_layers=3,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        candidate_source_layer_id=20,
        engram_layer_ids=[1, 14],
        engram_max_ngram_size=4,
        sliding_window=128,
        head_dim=HEAD_DIM,
        index_head_dim=128,
        index_topk=512,
        kv_lora_rank=448,
        qk_rope_head_dim=64,
    )
    return SimpleNamespace(
        architectures=["DeepseekV41ForCausalLM"],
        model_type="deepseek_v41",
        text_config=SimpleNamespace(**text),
        **{k: v for k, v in text.items() if k != "model_type"},
    )


def _engine_manager(
    max_draft_len: int,
    max_batch_size: int = 32,
    quota: int = _ENGINE_QUOTA_BYTES,
    storage: str = "bf16",
    max_num_tokens: int = 8192,
    max_seq_len: int = None,
    pp_stage: int = None,
):
    """The engine-size V4.1 DeepseekV4CacheManager (one alive at a time; released at module end).

    ``max_batch_size`` 32 is the accuracy gate; 8 is the boot gate's batch size at the
    same quota, where the compressor-state pool is larger. ``quota`` defaults to the
    accuracy gate's logged quota; ``_ENVELOPE_QUOTA_BYTES`` builds the largest pools an
    engine at the gate settings can get. ``storage`` "native" builds it with
    TRTLLM_DSV41_KV_STORAGE=native (set for the construction only). ``max_seq_len`` is the
    manager's length (default 16384, or 16398 with DSpark). ``pp_stage`` 0 / 1 builds the
    stage manager of PP2 (``Mapping(world_size=4, rank=2 * stage, tp_size=2, pp_size=2)``;
    the pools do not depend on tp), built alone in this process, so
    the manager's cross-rank allreduce(MIN) of its token capacity runs over this rank only --
    the stage-alone pools, an upper bound on what a PP2 engine's stage allocates. Host tier off
    (``host_cache_size=0``): it only adds a host copy for suspend/resume and does not
    change the device pools."""
    import os

    if max_seq_len is None:
        max_seq_len = _ENGINE_MAX_SEQ_DSPARK if max_draft_len else _ENGINE_MAX_SEQ
    key = (max_draft_len, max_batch_size, quota, storage, max_num_tokens, max_seq_len, pp_stage)
    if key in _ENGINE_MANAGERS:
        return _ENGINE_MANAGERS[key]
    _release_engine_managers()
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import DeepseekV4CacheManager
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType as T,
    )
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import (
        DeepSeekV4SparseAttentionConfig,
        DSparkDecodingConfig,
        KvCacheConfig,
    )
    from tensorrt_llm.mapping import Mapping

    env_before = os.environ.get("TRTLLM_DSV41_KV_STORAGE")
    os.environ["TRTLLM_DSV41_KV_STORAGE"] = storage
    try:
        mgr = DeepseekV4CacheManager(
            kv_cache_config=KvCacheConfig(
                enable_block_reuse=True,
                max_gpu_total_bytes=quota,
                host_cache_size=0,
                event_buffer_max_size=0,
            ),
            kv_cache_type=CacheType.SELFKONLY,
            num_layers=40,
            num_kv_heads=1,
            head_dim=HEAD_DIM,
            tokens_per_block=128,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
            if pp_stage is None
            else Mapping(world_size=4, rank=2 * pp_stage, tp_size=2, pp_size=2, gpus_per_node=4),
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            vocab_size=129280,
            max_num_tokens=max_num_tokens,
            sparse_attn_config=DeepSeekV4SparseAttentionConfig(
                index_head_dim=128, window_size=128, compress_ratios=[1] * 43
            ),
            pretrained_config=_v41_pretrained_config(),
            spec_config=DSparkDecodingConfig(max_draft_len=max_draft_len, block_size=max_draft_len)
            if max_draft_len
            else None,
            max_cuda_graph_batch_size=max_batch_size,
            disable_overlap_scheduler=False,
        )
    finally:
        if env_before is None:
            os.environ.pop("TRTLLM_DSV41_KV_STORAGE")
        else:
            os.environ["TRTLLM_DSV41_KV_STORAGE"] = env_before
    if pp_stage is not None:
        assert pp_stage == 0, "only PP2 stage 0 holds the compressor state (owners 2 / 8 / 14)"
        assert list(mgr.pp_layers) == list(range(20)), list(mgr.pp_layers)
    pool = mgr.get_buffers(2, T.COMPRESSOR_KV)
    shapes = {
        "swa": tuple(mgr.get_buffers(0, T.SWA).shape),
        "compress_r2": sorted({mgr.get_buffers(o, T.COMPRESS).shape[0] for o in (2, 8, 14)}),
        "compress_r1": tuple(mgr.get_buffers(20, T.COMPRESS).shape) if pp_stage is None else None,
        "compressor_state": tuple(pool.shape),
        "swa_dtype": str(mgr.get_buffers(0, T.SWA).dtype),
    }
    print(
        f"[engine manager max_draft_len={max_draft_len} max_batch_size={max_batch_size} quota={quota} "
        f"storage={storage} max_num_tokens={max_num_tokens} max_seq_len={max_seq_len} pp_stage={pp_stage} "
        f"max_blocks_per_seq={mgr.max_blocks_per_seq}] {shapes}"
    )
    assert (shapes["swa_dtype"] == "torch.uint8") == (storage == "native"), shapes
    if key == (0, 32, _ENGINE_QUOTA_BYTES, "bf16", 8192, _ENGINE_MAX_SEQ, None):
        # The gate's dump: this manager is the engine's.
        assert shapes["swa"][0] == _ENGINE_SWA_PAGES, shapes
        assert set(shapes["compress_r2"]) == _ENGINE_COMPRESS_R2_PAGES, shapes
        assert shapes["compress_r1"][0] == _ENGINE_COMPRESS_R1_PAGES, shapes
    if quota == _ENVELOPE_QUOTA_BYTES and max_seq_len in (_ENGINE_MAX_SEQ, _ENGINE_MAX_SEQ_DSPARK):
        assert quota <= torch.cuda.mem_get_info()[1], (
            "the envelope quota exceeds this device's memory"
        )
        if pp_stage is not None:
            want = _ENVELOPE_STATE_PAGES.get(
                (storage, max_draft_len, max_batch_size, max_num_tokens, pp_stage)
            )
        elif max_num_tokens != 8192:
            want = _ENVELOPE_STATE_PAGES.get(
                (storage, max_draft_len, max_batch_size, max_num_tokens)
            )
        else:
            want = _ENVELOPE_STATE_PAGES[(storage, max_draft_len, max_batch_size)]
        if want is not None:
            assert shapes["compressor_state"][0] == want, shapes
    if max_seq_len - (14 if max_draft_len else 0) in _ENVELOPE_WIDTH_OF_L:
        assert (
            mgr.max_blocks_per_seq
            == _ENVELOPE_WIDTH_OF_L[max_seq_len - (14 if max_draft_len else 0)]
        ), mgr.max_blocks_per_seq
    for owner in (2, 8, 14):
        for role in (T.COMPRESSOR_KV, T.COMPRESSOR_SCORE):
            b = mgr.get_buffers(owner, role)
            assert (
                b.data_ptr() == pool.data_ptr()
                and b.shape == pool.shape
                and b.dtype == torch.float32
            )
    assert pool.is_contiguous() and tuple(pool.shape[1:]) == (128, HEAD_DIM)
    _ENGINE_MANAGERS[key] = mgr
    return mgr


def _release_engine_managers():
    for mgr in _ENGINE_MANAGERS.values():
        mgr.shutdown()
    _ENGINE_MANAGERS.clear()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module", autouse=True)
def _engine_managers_released_at_module_end():
    yield
    _release_engine_managers()


def _engine_state_pool():
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType as T,
    )

    return _engine_manager(0).get_buffers(2, T.COMPRESSOR_KV)


class _FarEndPool:
    """Synthetic kv / score tables over the ENGINE's shared compressor-state pool.
    Only the pages covering ``spans[b] = (lo, hi)`` are mapped, to distinct kv and
    score pages taken (shuffled) from ``pages`` -- by default the pool's LAST pages;
    every other column points at a trap page the op must never touch (by default the
    highest page not in ``pages``). The mapped pages and the trap page are filled with
    SENTINEL; the rest of the pool is left alone and covered by a checksum compare in
    ``check``."""

    def __init__(
        self,
        pool: torch.Tensor,
        spans: Sequence[Tuple[int, int]],
        seed: int,
        pages=None,
        trap=None,
        width=None,
    ):
        self.kv = self.score = pool
        self.page_size = pool.shape[1]
        ps = self.page_size
        num_pages = pool.shape[0]
        per = [list(range(lo // ps, (hi - 1) // ps + 1)) for lo, hi in spans]
        need = 2 * sum(len(p) for p in per)
        if pages is None:
            pages = list(range(num_pages - need, num_pages))
        assert len(set(pages)) >= need and max(pages) < num_pages, "not enough pages"
        pages = list(pages)[:need]
        if trap is None:
            trap = next(t for t in range(num_pages - 1, -1, -1) if t not in set(pages))
        assert trap not in set(pages) and 0 <= trap < num_pages
        self.trap = trap
        cpu = torch.Generator().manual_seed(seed)
        far = [pages[i] for i in torch.randperm(need, generator=cpu).tolist()]
        width = max(p[-1] for p in per) + 2 if width is None else width
        assert width > max(p[-1] for p in per)
        bt_kv = torch.full((len(spans), width), self.trap, dtype=torch.int32)
        bt_sc = torch.full_like(bt_kv, self.trap)
        for b, logical in enumerate(per):
            for q in logical:
                bt_kv[b, q] = far.pop()
                bt_sc[b, q] = far.pop()
        self.bt_kv, self.bt_score = bt_kv.to(DEV), bt_sc.to(DEV)
        self.bt_kv_all, self.bt_score_all = self.bt_kv, self.bt_score
        self.mapped = torch.tensor(sorted(set(pages) | {self.trap}), device=DEV)
        self.lo_page = int(self.mapped[0])
        pool[self.mapped] = SENTINEL
        self.outside = self._outside_sum()

    def _outside_sum(self) -> int:
        # Summed 8192 pages at a time: one int64 reduction over a B_max pool would need a
        # temporary as large as the pool itself.
        whole = sum(
            int(self.kv[i : i + 8192].view(torch.int32).sum(dtype=torch.int64))
            for i in range(0, self.kv.shape[0], 8192)
        )
        return whole - int(self.kv[self.mapped].view(torch.int32).sum(dtype=torch.int64))

    def slots(self, table: torch.Tensor, b: int, pos: torch.Tensor):
        return table[b].long()[pos // self.page_size], pos % self.page_size

    def put(self, b: int, pos: int, kv_row: torch.Tensor, score_row: torch.Tensor) -> None:
        """Park one token's state, as a previous call leaves it (V4.1: zero ape)."""
        ps = self.page_size
        self.kv[int(self.bt_kv[b, pos // ps]), pos % ps] = kv_row
        self.score[int(self.bt_score[b, pos // ps]), pos % ps] = score_row

    def written(self) -> torch.Tensor:
        """Number of mapped-page rows (slots) that no longer hold SENTINEL, on device."""
        return (self.kv[self.mapped] != SENTINEL).any(-1).sum()

    def check(self, truth: Sequence[Tuple[int, torch.Tensor, torch.Tensor, Sequence[int]]]) -> None:
        """truth: (b, kv, score, positions) -- every listed position holds kv / score bit for
        bit; no other slot of the mapped pages, the trap page included, and no byte of
        the rest of the pool changed."""
        n = 0
        for b, kv, sc, positions in truth:
            pos = torch.as_tensor(list(positions), device=DEV)
            assert torch.equal(self.kv[self.slots(self.bt_kv, b, pos)], kv[pos]), (
                f"seq {b}: kv state"
            )
            assert torch.equal(self.score[self.slots(self.bt_score, b, pos)], sc[pos]), (
                f"seq {b}: score state"
            )
            n += 2 * pos.numel()
        assert int(self.written()) == n, "the op wrote a slot outside the named positions"
        assert (self.kv[self.trap] == SENTINEL).all(), "the trap page was written"
        assert self._outside_sum() == self.outside, "pool written outside the mapped pages"


def _hazard_pages(num_pages: int) -> List[int]:
    """Pages on both sides of every 32-bit offset hazard of an fp32 [pages, 128, 512]
    pool (65536 elements, 256 KiB per page): byte offsets 2^31 (page 8192) and 2^32
    (16384), element offset 2^31 = byte offset 2^33 (32768), element offset 2^32 = byte
    offset 2^34 (65536), element offset 2^33 = byte offset 2^35 (131072); those the pool
    reaches."""
    return sorted(
        c
        for c in (8191, 8192, 16383, 16384, 32767, 32768, 65535, 65536, 131071, 131072)
        if c < num_pages
    )


# Histories of the 32 generation requests: both parities, both sides of a
# 128-token page edge, and up to the engine's max_seq_len.
_GEN_HISTORIES = [
    1,
    2,
    3,
    126,
    127,
    128,
    129,
    255,
    256,
    257,
    511,
    1000,
    1001,
    2047,
    2048,
    4095,
    4096,
    5003,
    6000,
    7777,
    8191,
    8192,
    9001,
    10240,
    11111,
    12287,
    13000,
    14335,
    15000,
    16001,
    16376,
    16377,
]


def _truth_rows(cached: Sequence[int], extra: int, gen: torch.Generator):
    """Per-request per-position kv / score arrays over [start - 1, start + extra) (``_Rows``,
    absolute positions): only positions from start - 1 on are ever read."""
    kvs, scores = [], []
    for c in cached:
        lo = max(c - 1, 0)
        kv, sc = _Rows(lo, c + extra), _Rows(lo, c + extra)
        kv.t, sc.t = _rand_rows(c + extra - lo, gen)
        kvs.append(kv)
        scores.append(sc)
    return kvs, scores


def _envelope_histories(n: int, W: int, next_n: int, seed: int) -> List[int]:
    """n generation histories for a table W columns wide (max_seq_len L = the envelope's L of W):
    request 0 ends on the table's last slot (W * 128 - 1, past any position the engine writes),
    request 1 at the envelope's largest kv length (_envelope_kv_max), request 2 at L, then page and
    parity edges, and the rest uniform up to the envelope's largest start -- both parities."""
    L = _ENVELOPE_L[_ENVELOPE_WIDTHS.index(W)]
    top = _envelope_kv_max(L, next_n) - next_n
    fixed = [
        W * 128 - next_n,
        top,
        L - next_n,
        1,
        2,
        3,
        126,
        127,
        128,
        129,
        255,
        256,
        257,
        16376,
        16377,
        top - 1,
        top - 2,
        L - next_n - 1,
    ]
    cpu = torch.Generator().manual_seed(seed)
    rand = torch.randint(1, top + 1, (max(n - len(fixed), 0),), generator=cpu).tolist()
    return (fixed + rand)[:n]


@pytest.mark.parametrize("next_n", [1, 6])
@pytest.mark.parametrize("W", _ENVELOPE_WIDTHS)
def test_every_batch_size_1_to_256_at_every_table_width(W: int, next_n: int):
    """On the batch axis: G generation requests size the launch grid, and the table
    width W is the row stride of both tables. At every width the manager builds over the
    envelope (132 / 260 / 516 / 1028 / 2052 / 8196 columns: L 16384 .. 1048576), calls at every
    G in 1..256 over the first G of 256 requests -- histories up to the envelope's largest kv
    length (_envelope_kv_max) and request 0 on the table's last column -- on the engine's pool with
    every request's pages at its far end, return for each request exactly the rows of the
    G = 256 call bit for bit (re-running a step stores the same state, so the calls are
    idempotent), and the G = 256 rows match the reference."""
    pool_t = _engine_state_pool()
    ape = _zero_ape()
    cached = _envelope_histories(_ENVELOPE_REQS, W, next_n, seed=1400 + W + next_n)
    assert len(cached) == _ENVELOPE_REQS and max(cached) + next_n == W * 128
    gen = _gen(1400 + W + next_n)
    kvs, scores = _truth_rows(cached, next_n, gen)
    pool = _FarEndPool(
        pool_t, [(max(c - 1, 0), c + next_n) for c in cached], seed=1400 + next_n, width=W
    )
    assert pool.bt_kv.shape == (_ENVELOPE_REQS, W)
    pending = [(b, c - 1) for b, c in enumerate(cached) if c % RATIO]
    for b, p in pending:
        pool.put(b, p, kvs[b][p], scores[b][p])
    full = _step(pool, kvs, scores, cached, next_n, ape)
    _check_outputs(kvs, scores, ape, [(list(cached), full)], next_n)
    bad = torch.zeros((), dtype=torch.int64, device=DEV)
    for G in range(1, _ENVELOPE_REQS + 1):
        pool.bt_kv, pool.bt_score = pool.bt_kv_all[:G], pool.bt_score_all[:G]
        rows = _step(pool, kvs[:G], scores[:G], cached[:G], next_n, ape)
        for b in range(G):
            bad += (rows[b].view(torch.int32) != full[b].view(torch.int32)).sum()
        # Idempotent state: the same new slots as the G = 256 call, plus the parked tokens.
        bad += (pool.written() - 2 * (len(pending) + _ENVELOPE_REQS * next_n)).abs()
    pool.bt_kv, pool.bt_score = pool.bt_kv_all, pool.bt_score_all
    assert int(bad) == 0, f"{int(bad)} elements / slots differ from the G = {_ENVELOPE_REQS} call"
    pool.check(
        [
            (b, kvs[b], scores[b], range(max(c - 1, 0) if c % RATIO else c, c + next_n))
            for b, c in enumerate(cached)
        ]
    )
    print(
        f"[G sweep W={W} next_n={next_n}] engine pool {tuple(pool_t.shape)}, far-end pages "
        f"{pool.lo_page + 1}..{pool_t.shape[0] - 1}, positions up to {max(cached) + next_n - 1}: "
        f"G 1..{_ENVELOPE_REQS} bitwise equal to the G = {_ENVELOPE_REQS} rows"
    )


@pytest.mark.parametrize("next_n", [1, 6])
def test_every_context_lead_on_the_engine_pool(next_n: int):
    """On the mixed-step offsets: the target hands the op the step's whole
    kv_score and output buffer, so the generation rows start at the context token count
    L (cu_seq_lens[0]) and the outputs at the contexts' compressed-row count O
    (cu_kv_comp[0]). For every L in 0..16384 - 8 * next_n (O = L // 2; max_num_tokens 16384),
    ten generation requests at both parities, two of them at the 1,048,576-token envelope's
    far end on 8196-column tables, return exactly the rows of the L = 0 call, bit for bit,
    and write nothing else into the output buffer; the rows ahead of them are NaN and
    never read. The dumps' leads (1016 / 508, 1934 / 967, and the batch-256 engines' up to
    16290 / 8145) are inside the sweep."""
    pool_t = _engine_state_pool()
    ape = _zero_ape()
    top = _envelope_kv_max(_ENVELOPE_L[-1], next_n) - next_n
    cached = [1, 2, 127, 128, 1001, 2048, 8191, 16376, top - 1, top]
    G = len(cached)
    gen = _gen(1500 + next_n)
    kvs, scores = _truth_rows(cached, next_n, gen)
    pool = _FarEndPool(
        pool_t,
        [(max(c - 1, 0), c + next_n) for c in cached],
        seed=1500 + next_n,
        width=_ENVELOPE_WIDTHS[-1],
    )
    for b, c in enumerate(cached):
        if c % RATIO:
            pool.put(b, c - 1, kvs[b][c - 1], scores[b][c - 1])
    base = torch.cat(_step(pool, kvs, scores, cached, next_n, ape))
    n = base.shape[0]
    rows = torch.cat(
        [
            torch.cat([kv[c : c + next_n], sc[c : c + next_n]], 1)
            for kv, sc, c in zip(kvs, scores, cached)
        ]
    )
    kv_lens = _i32(c + next_n for c in cached)
    step_cu_out = _i32(
        [0] + torch.tensor([(c + next_n) // 2 - c // 2 for c in cached]).cumsum(0).tolist()
    )
    step_cu_seq = torch.arange(G + 1, device=DEV, dtype=torch.int32) * next_n
    max_lead = _ENVELOPE_TOKENS - G * next_n
    kv_score = torch.full((_ENVELOPE_TOKENS, 2 * HEAD_DIM), float("nan"), device=DEV)
    out = torch.full((max_lead // 2 + n + 1, HEAD_DIM), float("nan"), device=DEV)
    bad = torch.zeros((), dtype=torch.int64, device=DEV)
    for L in range(max_lead + 1):
        out_lead = L // 2
        kv_score[L : L + G * next_n] = rows
        op(
            kv_score,
            ape,
            pool.kv,
            pool.score,
            pool.bt_kv,
            pool.bt_score,
            out,
            kv_lens,
            step_cu_seq + L,
            step_cu_out + out_lead,
            G,
            pool.page_size,
            HEAD_DIM,
            RATIO,
            next_n,
        )
        bad += (out[out_lead : out_lead + n].view(torch.int32) != base.view(torch.int32)).sum()
        bad += ((~torch.isnan(out)).sum() - n * HEAD_DIM).abs()
        out[out_lead : out_lead + n] = float("nan")
        kv_score[L : L + G * next_n] = float("nan")
    torch.cuda.synchronize()
    assert int(bad) == 0, (
        f"{int(bad)} output elements differ from the lead-0 call or were written elsewhere"
    )
    pool.check(
        [
            (b, kvs[b], scores[b], range(max(c - 1, 0) if c % RATIO else c, c + next_n))
            for b, c in enumerate(cached)
        ]
    )
    print(
        f"[lead sweep next_n={next_n}] leads L 0..{max_lead} (O = L // 2): {n} rows bitwise equal to lead 0"
    )


@pytest.mark.parametrize("next_n", [1, 6])
@pytest.mark.parametrize("W", _ENVELOPE_WIDTHS)
def test_cuda_graph_capture_set_on_the_engine_pool(W: int, next_n: int):
    """Captured at every graph batch G of Gset (every integer 1..32 plus the padded
    default list for cuda_graph_config {enable_padding: true, max_batch_size: 256}: 40..128 step
    8, 192, 256 -- 46 values; the gates' earlier sets 1..8 and {1, 2, 4, 8, 16, 24, 32} are
    subsets), at next_n 1 (decode: n = G output rows) and 6 (DSpark verify: n = 3G, with a
    random 0..5 accepted drafts per request between steps, so replays rewind to either parity
    and overwrite rejected positions), on tables of every width W the graph metadata uses over
    the envelope (132 .. 8196 columns) with histories up to the envelope's largest kv length.
    Warm-up and capture on step 0, then 3 replays on new kv_score / kv_lens / cu_kv_comp
    written in place. Every replay matches the reference built from what each position
    last received, an eager call on the same buffers returns the replay's rows bit for
    bit, and the pool holds exactly the new state (far-end pages, rest untouched)."""
    pool_t = _engine_state_pool()
    ape = _zero_ape()
    steps = 3
    L = _ENVELOPE_L[_ENVELOPE_WIDTHS.index(W)]
    top = (
        _envelope_kv_max(L, next_n) - (steps + 1) * next_n
    )  # every replay's kv length stays <= _envelope_kv_max
    cpu = torch.Generator().manual_seed(1600 + W + next_n)
    parities = set()
    for G in _GSET:
        cached = sorted(torch.randint(1, top, (G,), generator=cpu).tolist())
        cached[0] |= 1  # at least one odd start (a pending token read back)
        if G > 1:
            cached[-1] = top - top % 2  # an even start at the envelope's far end
        extra = (steps + 1) * next_n
        gen = _gen(1600 + 10 * G + next_n)
        kvs, scores = _truth_rows(cached, extra, gen)
        pool = _FarEndPool(
            pool_t, [(max(c - 1, 0), c + extra) for c in cached], seed=1600 + G, width=W
        )
        for b, c in enumerate(cached):
            if c % RATIO:
                pool.put(b, c - 1, kvs[b][c - 1], scores[b][c - 1])
        reserved = G * math.ceil(next_n / RATIO)
        s_kv_score = torch.zeros(G * next_n, 2 * HEAD_DIM, device=DEV)
        s_kv_lens = torch.zeros(G, dtype=torch.int32, device=DEV)
        s_cu_seq = torch.arange(G + 1, device=DEV, dtype=torch.int32) * next_n
        s_cu_out = torch.zeros(G + 1, dtype=torch.int32, device=DEV)
        s_out = torch.zeros(reserved, HEAD_DIM, device=DEV)
        e_out = torch.zeros_like(s_out)

        def load(starts):
            s_kv_score.copy_(
                torch.cat(
                    [
                        torch.cat([kv[s : s + next_n], sc[s : s + next_n]], 1)
                        for kv, sc, s in zip(kvs, scores, starts)
                    ]
                )
            )
            s_kv_lens.copy_(_i32(s + next_n for s in starts))
            n_out = [(s + next_n) // 2 - s // 2 for s in starts]
            s_cu_out.copy_(_i32([0] + torch.tensor(n_out).cumsum(0).tolist()))
            return n_out

        def call(out):
            op(
                s_kv_score,
                ape,
                pool.kv,
                pool.score,
                pool.bt_kv,
                pool.bt_score,
                out,
                s_kv_lens,
                s_cu_seq,
                s_cu_out,
                G,
                pool.page_size,
                HEAD_DIM,
                RATIO,
                next_n,
            )

        def check(starts, n_out, got):
            off = 0
            for b, s in enumerate(starts):
                ref = _pool_windows(kvs[b], scores[b], ape, s // 2, (s + next_n) // 2)
                torch.testing.assert_close(
                    got[off : off + n_out[b]], ref, msg=lambda m: f"G {G} b {b}: {m}"
                )
                off += n_out[b]
            assert torch.isnan(got[off:]).all(), f"G {G}: rows past the packed outputs written"

        starts = list(cached)
        n_out = load(starts)
        s_out.fill_(float("nan"))
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            call(s_out)  # the engine's warm-up at this batch
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        check(starts, n_out, s_out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call(s_out)
        for _ in range(steps):
            accepted = (
                torch.randint(0, next_n, (G,), generator=cpu).tolist() if next_n > 1 else [0] * G
            )
            starts = [s + 1 + a for s, a in zip(starts, accepted)]
            for b, s in enumerate(starts):
                kvs[b][s : s + next_n], scores[b][s : s + next_n] = _rand_rows(next_n, gen)
                parities.add(s % RATIO)
            n_out = load(starts)
            s_out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            check(starts, n_out, s_out)
            e_out.fill_(float("nan"))
            call(e_out)
            torch.cuda.synchronize()
            assert torch.equal(s_out.view(torch.int32), e_out.view(torch.int32)), (
                f"G {G}: replay != eager"
            )
        # The pool holds what each position last received: the parked token and every
        # position a step wrote (rejected drafts included, as later steps overwrote them).
        last = [s + next_n for s in starts]
        pool.check(
            [
                (b, kvs[b], scores[b], range(max(c - 1, 0) if c % RATIO else c, last[b]))
                for b, c in enumerate(cached)
            ]
        )
        del graph
    assert parities == {0, 1}, f"replays reached only start parities {parities}"
    print(
        f"[capture W={W} next_n={next_n}] {len(_GSET)} batches G {_GSET[0]}..{_GSET[-1]} (Gset), output rows "
        f"{math.ceil(next_n / RATIO)}G, histories up to {top}: every replay == reference and == eager bitwise"
    )


def test_graph_replay_equal_to_eager_cannot_see_a_stale_pending_token():
    """Control for the capture test: a replay and an eager call over the same pool read
    the same state, so replay == eager holds even when that state is wrong. A parked
    token overwritten by a rejected draft's value (a missed rewind) leaves replay and
    eager bitwise equal, and only the reference check rejects it -- which is why the
    capture test gates every replay on the reference, not on eager alone."""
    pool_t = _engine_state_pool()
    ape = _zero_ape()
    next_n, G = 6, 4
    cached = [33, 128, 1001, 4094]
    gen = _gen(1700)
    kvs, scores = _truth_rows(cached, next_n, gen)
    pool = _FarEndPool(pool_t, [(max(c - 1, 0), c + next_n) for c in cached], seed=1700)
    stale_kv, stale_sc = _rand_rows(1, gen)
    for b, c in enumerate(cached):
        if c % RATIO:
            pool.put(b, c - 1, stale_kv[0], stale_sc[0])  # not what position c - 1 last received
    s_kv_score = torch.cat(
        [
            torch.cat([kv[c : c + next_n], sc[c : c + next_n]], 1)
            for kv, sc, c in zip(kvs, scores, cached)
        ]
    )
    n_out = [(c + next_n) // 2 - c // 2 for c in cached]
    cu_out = _i32([0] + torch.tensor(n_out).cumsum(0).tolist())
    args = (s_kv_score, ape, pool.kv, pool.score, pool.bt_kv, pool.bt_score)
    tail = (
        _i32(c + next_n for c in cached),
        _i32(b * next_n for b in range(G + 1)),
        cu_out,
        G,
        pool.page_size,
        HEAD_DIM,
        RATIO,
        next_n,
    )
    s_out = torch.zeros(sum(n_out), HEAD_DIM, device=DEV)
    op(*args, s_out, *tail)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(*args, s_out, *tail)
    s_out.fill_(float("nan"))
    graph.replay()
    e_out = torch.zeros_like(s_out)
    op(*args, e_out, *tail)
    torch.cuda.synchronize()
    assert torch.equal(s_out.view(torch.int32), e_out.view(torch.int32))
    ref = torch.cat(
        [
            _pool_windows(kvs[b], scores[b], ape, c // 2, (c + next_n) // 2)
            for b, c in enumerate(cached)
        ]
    )
    diff = (s_out - ref).abs().max().item()
    print(
        f"[control stale pending token] replay == eager bitwise; max abs diff vs reference {diff:.3e}"
    )
    with pytest.raises(AssertionError):
        torch.testing.assert_close(s_out, ref)


# Mixed steps on the engine-size manager at the runtime-value dump's leads: a
# 1016-token context ahead of 2 decode rows ((G 2, next_n 1, n 510) in the dump;
# the prefill op's (T 1018, nc 1, max 508)), contexts of 1920 + 14 tokens ahead of
# one decode row ((G 1, n 968); prefill (T 1935, nc 2, max 960)), G = 7 behind a
# chunked context with an odd cached count, a full 8192-token step of 32 requests
# (8161 context tokens ahead of 31 decode rows), and the first two again with
# DSpark verify rows.
_ENGINE_MIXED = {
    "dump_ctx1016_gen2": ([(0, 1016)], [2047, 16000], 0),
    "dump_ctx1934_gen1": ([(0, 1920), (0, 14)], [5003], 0),
    "ctx_chunked_gen7": ([(1023, 1017)], [1, 2, 127, 128, 999, 8192, 16383], 0),
    "lead8161_gen31": ([(0, 8161)], _GEN_HISTORIES[:29] + [16382, 16383], 0),
    "dump_ctx1016_verify2": ([(0, 1016)], [2047, 16000], 5),
    "dump_ctx1934_verify1": ([(0, 1920), (0, 14)], [5003], 5),
    # Many contexts ahead (nc 12 / 31 / 24, B = 32), odd and even cached prefixes.
    "ctx12_gen20": (
        [
            (0, 700),
            (129, 611),
            (0, 1),
            (2048, 1000),
            (4095, 333),
            (0, 64),
            (777, 777),
            (16000, 300),
            (1, 2),
            (0, 1500),
            (8191, 900),
            (300, 1),
        ],
        _GEN_HISTORIES[:20],
        0,
    ),
    "ctx31_gen1": ([((37 * i) % 500, 200 + 3 * i) for i in range(31)], [5003], 0),
    "ctx24_verify8": (
        [((101 * i) % 3000, 150 + (7 * i) % 100) for i in range(24)],
        [34, 127, 128, 1094, 2047, 4096, 9001, 16001],
        5,
    ),
}


@pytest.mark.parametrize("scenario", list(_ENGINE_MIXED))
def test_engine_size_manager_mixed_step(scenario):
    """The target's exact call on a mixed step prepared by the engine-size manager
    (see _engine_manager): the same checks as test_v41_cache_manager_mixed_step_behind_contexts."""
    ctx, gen_cached, max_draft_len = _ENGINE_MIXED[scenario]
    _mixed_step_on_manager(_engine_manager(max_draft_len), scenario, ctx, gen_cached, max_draft_len)


# Mixed steps on managers of the engine envelope: bf16 storage at 0.5 x total memory,
# max_num_tokens 16384, max_batch_size 256, the manager's length Lm (L + 14 with DSpark), so the
# metadata's tables are 260 / 8196 columns wide. (contexts [(cached, new)], generation histories,
# max_draft_len, L). The requests are allocated at their full length in one resize_context, so
# they stay short: a 1,048,576-token request allocated that way does not fit
# (resize_context returned False), and at L 1,048,576 with max_batch_size 256 the manager
# (compressor-state pool 1662 pages, SWA pool 11000) did not hold 256 short requests
# (prepare_context returned False for the 248th behind a 2000-token context, and for the 255th
# with DSpark behind two short contexts). Positions near 1M on 8196-column tables are
# the synthetic tests' (test_every_batch_size_1_to_256_at_every_table_width, the envelope steps). 39
# contexts ahead of 217 verify rows at L 32768 (256 requests, the batch-256 engines' nc 39); at
# L 1,048,576 a 2000-token context ahead of 239 decode rows, and two contexts ahead of 238
# verify rows (240 requests).
_ENVELOPE_RNG = torch.Generator().manual_seed(2026)
_ENVELOPE_CTX39 = [
    (int(c), 150 + (97 * i) % 300)
    for i, c in enumerate(torch.randint(0, 400, (39,), generator=_ENVELOPE_RNG).tolist())
]
_ENVELOPE_SHORT = torch.randint(1, 400, (256,), generator=_ENVELOPE_RNG).tolist()
_ENVELOPE_MIXED_STEPS = {
    "gc_L32k_ctx39_verify217": (_ENVELOPE_CTX39, _ENVELOPE_SHORT[:217], 5, 32768),
    "gc_L1m_nods_ctx1_gen239": ([(0, 2000)], [1 + j % 120 for j in range(239)], 0, 1_048_576),
    "gc_L1m_ctx2_verify238": (
        [(0, 700), (129, 301)],
        [1 + j % 110 for j in range(238)],
        5,
        1_048_576,
    ),
}


@pytest.mark.parametrize("scenario", list(_ENVELOPE_MIXED_STEPS))
def test_envelope_manager_mixed_step(scenario):
    """The target's exact call on a mixed step prepared by a manager of the engine envelope
    (max_batch_size 256, max_num_tokens 16384, max_seq_len 32768 + 14 and 1,048,576 (+ 14); the
    metadata's tables 260 / 8196 columns wide, 256 requests in the step at L 32768 and 240 at L
    1,048,576): the same checks as test_v41_cache_manager_mixed_step_behind_contexts."""
    ctx, gen_cached, max_draft_len, L = _ENVELOPE_MIXED_STEPS[scenario]
    lm = L + (14 if max_draft_len else 0)
    assert sum(n for _, n in ctx) + len(gen_cached) * (max_draft_len + 1) <= _ENVELOPE_TOKENS
    assert len(ctx) + len(gen_cached) <= _ENVELOPE_REQS
    assert all(c + n <= lm for c, n in ctx) and all(c + max_draft_len + 1 <= lm for c in gen_cached)
    mgr = _engine_manager(
        max_draft_len,
        _ENVELOPE_REQS,
        _ENVELOPE_QUOTA_BYTES,
        "bf16",
        max_num_tokens=_ENVELOPE_TOKENS,
        max_seq_len=lm,
    )
    assert mgr.max_blocks_per_seq == _ENVELOPE_WIDTH_OF_L[L]
    _mixed_step_on_manager(
        mgr, scenario, ctx, gen_cached, max_draft_len, max_num_tokens=_ENVELOPE_TOKENS
    )


# The four engine-size pools: DSpark off / on (max_draft_len 5) at the accuracy gate's
# max_batch_size 32 and at the boot gate's 8, one quota. Their compressor-state pools
# are [22908 | 41088 | 21246 | 39678, 128, 512] fp32; the two at batch 8 pass element
# offset 2^31. Then three pools at the envelope quota: [43518 | 78078 | 75516, 128,
# 512] (DSpark off at 32 and 8, on at 8); the two at batch 8 pass element offset 2^32.
# Then two with the native KV storage at that quota: [94464, 128, 512] (batch 32, the
# size of the native gates' largest pool) and [133248, 128, 512] (batch 8), which
# passes element offset 2^33. Then the engine envelope's largest unsplit pools: the bf16
# storage's [120192, 128, 512] (max_num_tokens 16384, batch 1, DSpark) and the native
# [172416, 128, 512] (max_num_tokens 16384, batch 2, DSpark off; 45.2 GB). Then the PP2 stage-0
# managers at the same tuples: bf16 [188796, 128, 512] and B_max, native [247806, 128, 512]
# (65.0 GB).
_ENGINE_POOLS = [
    (0, 32, _ENGINE_QUOTA_BYTES, "bf16", 8192, None),
    (0, 8, _ENGINE_QUOTA_BYTES, "bf16", 8192, None),
    (5, 32, _ENGINE_QUOTA_BYTES, "bf16", 8192, None),
    (5, 8, _ENGINE_QUOTA_BYTES, "bf16", 8192, None),
    (0, 32, _ENVELOPE_QUOTA_BYTES, "bf16", 8192, None),
    (0, 8, _ENVELOPE_QUOTA_BYTES, "bf16", 8192, None),
    (5, 8, _ENVELOPE_QUOTA_BYTES, "bf16", 8192, None),
    (0, 32, _ENVELOPE_QUOTA_BYTES, "native", 8192, None),
    (0, 8, _ENVELOPE_QUOTA_BYTES, "native", 8192, None),
    (5, 1, _ENVELOPE_QUOTA_BYTES, "bf16", 16384, None),
    (0, 2, _ENVELOPE_QUOTA_BYTES, "native", 16384, None),
    (5, 1, _ENVELOPE_QUOTA_BYTES, "bf16", 16384, 0),
    (0, 2, _ENVELOPE_QUOTA_BYTES, "native", 16384, 0),
]


@pytest.mark.parametrize(
    "max_draft_len,max_batch_size,quota,storage,max_num_tokens,pp_stage",
    _ENGINE_POOLS,
    ids=[
        f"D{d}-bs{b}-{'envelope' if q == _ENVELOPE_QUOTA_BYTES else 'gate'}"
        + ("-native" if st == "native" else "")
        + (f"-mnt{t}" if t != 8192 else "")
        + (f"-pp2stage{pp}" if pp is not None else "")
        for d, b, q, st, t, pp in _ENGINE_POOLS
    ],
)
def test_offset_hazard_pages_of_the_engine_pools(
    max_draft_len: int, max_batch_size: int, quota: int, storage: str, max_num_tokens: int, pp_stage
):
    """At the far end, run: 32 generation requests at next_n = max_draft_len + 1
    whose kv / score pages are the pages on both sides of every 32-bit offset hazard the
    pool reaches (_hazard_pages) and otherwise its last pages. Outputs match the
    reference, the pool holds exactly the new state, and no other byte of the pool
    changed. Control: the same step with every page id at or past the pool's highest
    hazard wrapped below it (as a 32-bit offset would wrap) fails that check."""
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType as T,
    )

    pool_t = _engine_manager(
        max_draft_len,
        max_batch_size,
        quota,
        storage,
        max_num_tokens=max_num_tokens,
        pp_stage=pp_stage,
    ).get_buffers(2, T.COMPRESSOR_KV)
    num_pages = pool_t.shape[0]
    next_n = max_draft_len + 1
    ape = _zero_ape()
    cached = _GEN_HISTORIES
    spans = [(max(c - 1, 0), c + next_n) for c in cached]
    need = 2 * sum((hi - 1) // 128 - lo // 128 + 1 for lo, hi in spans)
    hz = _hazard_pages(num_pages)
    pages = hz + [p for p in range(num_pages - need, num_pages) if p not in hz][: need - len(hz)]
    gen = _gen(1800 + num_pages)
    kvs, scores = _truth_rows(cached, next_n, gen)

    def run(wrap: int):
        pool = _FarEndPool(pool_t, spans, seed=1800 + num_pages, pages=pages)
        for b, c in enumerate(cached):
            if c % RATIO:
                pool.put(b, c - 1, kvs[b][c - 1], scores[b][c - 1])
        truth_bt = (pool.bt_kv, pool.bt_score)
        if wrap:
            pool.bt_kv = torch.where(pool.bt_kv >= wrap, pool.bt_kv % wrap, pool.bt_kv)
            pool.bt_score = torch.where(pool.bt_score >= wrap, pool.bt_score % wrap, pool.bt_score)
        rows = _step(pool, kvs, scores, cached, next_n, ape)
        pool.bt_kv, pool.bt_score = truth_bt
        return pool, rows

    pool, rows = run(0)
    _check_outputs(kvs, scores, ape, [(list(cached), rows)], next_n)
    truth = [
        (b, kvs[b], scores[b], range(max(c - 1, 0) if c % RATIO else c, c + next_n))
        for b, c in enumerate(cached)
    ]
    pool.check(truth)
    wrap = max(hz)
    used = sorted(
        {int(p) for p in pool.bt_kv.flatten().tolist() + pool.bt_score.flatten().tolist()}
        - {pool.trap}
    )
    print(
        f"[hazard pages D={max_draft_len} bs={max_batch_size} {storage} pp_stage={pp_stage}] pool "
        f"{tuple(pool_t.shape)} "
        f"({pool_t.numel() / 2**31:.2f} x 2^31 elements): pages {used[: len(hz)]} ... {used[-1]}; "
        "outputs and state exact"
    )
    pool, _ = run(wrap)
    with pytest.raises(AssertionError):
        pool.check(truth)
    print(f"[hazard pages control] page ids >= {wrap} wrapped mod {wrap}: the check fails")


# ---------------------------------------------------------------------------
# Pool-size invariance up to the largest pool an engine can get
# ---------------------------------------------------------------------------


def _invariance_schedule(next_n: int, steps: int, seed: int):
    """32 generation requests (histories of both parities, up to max_seq_len 16384, and two
    at the 1,048,576-token envelope's far end) and ``steps`` generation steps after them; at next_n 6 each later
    step starts at s + 1 + a random 0..5 accepted drafts, so it rewinds over rejected
    positions and reaches both start parities. Returns the histories, the per-request
    kv / score arrays holding the step-0 rows, and for every later step its starts
    and the new rows it feeds."""
    cpu = torch.Generator().manual_seed(seed)
    extra = (steps + 1) * next_n
    far = (
        _envelope_kv_max(_ENVELOPE_L[-1], next_n) - extra
    )  # the 1,048,576-token envelope's far end
    cached = [1, 2, 127, 128, 129, 16384 - extra, 16383 - extra, far, far - 1]
    cached += torch.randint(1, 16384 - extra, (32 - len(cached),), generator=cpu).tolist()
    gen = _gen(seed)
    kvs, scores = _truth_rows(cached, extra, gen)
    later, starts = [], list(cached)
    for _ in range(steps - 1):
        accepted = (
            torch.randint(0, next_n, (32,), generator=cpu).tolist() if next_n > 1 else [0] * 32
        )
        starts = (
            [s + 1 + a for s, a in zip(starts, accepted)] if next_n > 1 else [s + 1 for s in starts]
        )
        later.append((list(starts), [_rand_rows(next_n, gen) for _ in starts]))
    return cached, kvs, scores, later


def _run_invariance_schedule(
    pool_t, cached, kvs0, scores0, later, next_n, seed, pages=None, trap=None
):
    """The schedule on ``pool_t`` (tables into ``pages``, or the pool's last pages): every
    step's rows match the reference built from what each position last received, and at
    the end the pool holds exactly each position's last state and nothing else changed.
    Returns the steps' packed outputs and the final contents of the mapped pages."""
    ape = _zero_ape()
    extra = (len(later) + 2) * next_n  # _invariance_schedule's (steps + 1) * next_n
    kvs = [k.clone() for k in kvs0]
    scores = [sc.clone() for sc in scores0]
    pool = _FarEndPool(
        pool_t, [(max(c - 1, 0), c + extra) for c in cached], seed=seed, pages=pages, trap=trap
    )
    for b, c in enumerate(cached):
        if c % RATIO:
            pool.put(b, c - 1, kvs[b][c - 1], scores[b][c - 1])
    outs = []
    plan = [(list(cached), None)] + later
    for starts, rows in plan:
        if rows is not None:
            for b, (s, (kv, sc)) in enumerate(zip(starts, rows)):
                kvs[b][s : s + next_n], scores[b][s : s + next_n] = kv, sc
        got = _step(pool, kvs, scores, starts, next_n, ape)
        _check_outputs(kvs, scores, ape, [(starts, got)], next_n)
        outs.append(torch.cat(got))
    last = [s + next_n for s in plan[-1][0]]
    pool.check(
        [
            (b, kvs[b], scores[b], range(max(c - 1, 0) if c % RATIO else c, last[b]))
            for b, c in enumerate(cached)
        ]
    )
    return outs, pool_t[pool.mapped].clone(), pool


@pytest.mark.parametrize("next_n", [1, 6])
def test_pool_size_invariance_up_to_the_envelope_pool(next_n: int):
    """One schedule of 32 generation requests (a decode run at next_n 1; a
    DSpark verify run with random rewinds at next_n 6; odd starts read a parked token; two
    requests at the 1,048,576-token envelope's far end) on the same logical rows at the same
    page indices of pools of B pages: views [:B] of the largest compressor-state pool the
    engine envelope gives (B_max = 247806: the live PP2 stage-0 manager with native KV
    storage, max_num_tokens 16384, batch 2, DSpark off, at 0.5 x total memory) for B = the
    smallest pool that holds those pages, the accuracy gate's 22908, the bf16 gate envelope's
    43518 (batch 32) and 78078 (batch 8), a mid size, the native gate envelope's 94464 (batch
    32) and 133248 (batch 8, the gates' old bound), the bf16 storage's largest unsplit pool
    120192, the native unsplit largest 172416, the bf16 stage-0 largest 188796, and B_max, plus
    a separate allocation of exactly the smallest size. Every run matches the reference, and
    all runs are bitwise identical -- outputs of every step and the final contents of the
    mapped pages. Then the far end: the same schedule with every request's pages among B_max's
    last pages (the last page, 247805, owned) returns the same outputs bit for bit, and no
    other byte of the 65.0 GB pool changed."""
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType as T,
    )

    pool_max = _engine_manager(**_ENVELOPE_BMAX_MANAGER).get_buffers(2, T.COMPRESSOR_KV)
    b_max = pool_max.shape[0]
    assert b_max == _ENVELOPE_B_MAX, pool_max.shape
    seed = 1900 + next_n
    cached, kvs0, scores0, later = _invariance_schedule(next_n, steps=4, seed=seed)
    extra = (len(later) + 2) * next_n
    need = 2 * sum((c + extra - 1) // 128 - max(c - 1, 0) // 128 + 1 for c in cached)
    cpu = torch.Generator().manual_seed(seed)
    low = (torch.randperm(2 * need, generator=cpu) + 1)[:need].tolist()  # pages 1..2*need, shuffled
    trap = 0
    b_small = max(low) + 1
    sizes = [
        b_small,
        22908,
        _ENVELOPE_STATE_PAGES[("bf16", 0, 32)],
        (b_small + b_max) // 2,
        _ENVELOPE_STATE_PAGES[("bf16", 0, 8)],
        _ENVELOPE_STATE_PAGES[("native", 0, 32)],
        _ENVELOPE_STATE_PAGES[("bf16", 5, 1, 16384)],
        _ENVELOPE_B_MAX_GATES,
        _UNSPLIT_B_MAX,
        _ENVELOPE_STATE_PAGES[("bf16", 5, 1, 16384, 0)],
        b_max,
    ]
    runs = {}
    for B in sizes:
        runs[B] = _run_invariance_schedule(
            pool_max[:B], cached, kvs0, scores0, later, next_n, seed, low, trap
        )[:2]
    small = torch.full((b_small, 128, HEAD_DIM), SENTINEL, device=DEV)
    runs["own"] = _run_invariance_schedule(
        small, cached, kvs0, scores0, later, next_n, seed, low, trap
    )[:2]
    del small
    ref_outs, ref_pages = runs[b_small]
    for key, (outs, pages_now) in runs.items():
        assert len(outs) == len(ref_outs)
        for i, (o, r) in enumerate(zip(outs, ref_outs)):
            assert torch.equal(o.view(torch.int32), r.view(torch.int32)), (
                f"B {key} step {i}: outputs differ"
            )
        assert torch.equal(pages_now.view(torch.int32), ref_pages.view(torch.int32)), (
            f"B {key}: state differs"
        )
    far_outs, _, far = _run_invariance_schedule(
        pool_max, cached, kvs0, scores0, later, next_n, seed
    )
    used = set(far.bt_kv.flatten().tolist()) | set(far.bt_score.flatten().tolist())
    assert b_max - 1 in used, "the last page is not owned"
    for i, (o, r) in enumerate(zip(far_outs, ref_outs)):
        assert torch.equal(o.view(torch.int32), r.view(torch.int32)), (
            f"far end step {i}: outputs differ"
        )
    n_rows = sum(o.shape[0] for o in ref_outs)
    print(
        f"[pool invariance next_n={next_n}] {len(ref_outs)} steps, {n_rows} rows, pages {min(low)}..{max(low)}: "
        f"B in {sizes} and an own [{b_small}, 128, 512] allocation bitwise identical (outputs + state) and "
        f"== reference; far end pages {far.lo_page}..{b_max - 1} of [{b_max}, 128, 512] bitwise the same rows"
    )


# ---------------------------------------------------------------------------
# The multi-request envelope of the target's mixed steps
# ---------------------------------------------------------------------------
# A step of the core target over the engine envelope carries T <= 16384 tokens
# (max_num_tokens) of <= 256 requests (max_batch_size), each at most max_seq_len L long (L up
# to 1,048,576; the manager's Lm = L + 14 with DSpark, and generation kv lengths up to Lm + 5
# under the CUDA-graph metadata, _envelope_kv_max): nc context requests first, then G generation
# requests at next_n 1 (decode, DSpark off) or 6 (DSpark verify). This op sees only the
# generation part: G, next_n, each request's kv_len, the context lead L = cu_seq_lens[0] (the
# contexts' token count), O = cu_kv_comp[0] (their compressed-row count) and the tables' width
# (the manager's max_blocks_per_seq for that L: 132 .. 8196). The steps below draw all of those
# at random over the envelope; the parked token at an odd start comes either from a real
# earlier op call (a decode step, or a verify step whose rejected drafts this step then
# overwrites -- a rewind) or is planted, with rejected-draft garbage planted behind it.

_ENV_TOKENS = _ENVELOPE_TOKENS
_ENV_REQS = _ENVELOPE_REQS
_ENV_PAGES = 2048  # plainly more than a step's pages; pool size is the invariance test's axis
_ENV_CORNERS = [
    (1, 0),
    (1, 0),
    (256, 0),
    (256, 0),
    (1, -1),
    (1, -1),
    (128, -1),
    (128, -1),
    (255, 1),
    (255, 1),
    (2, -1),
    (2, -1),
    (1, -2),
    (1, -2),
]
_ENV_PARTITIONS = (
    len(_ENV_CORNERS) + 2 * _ENVELOPE_REQS
)  # the corners, then every G 1..256 at both next_n
_ENV_GROUPS = 8

# Every (G, next_n, num_total_compressed_tokens[2]) the core target's runtime-value dump
# recorded for this op (the boot gates' logs and the TEP4 / DEP4 / TP4 accuracy-gate previews'
# logs, a line kept only if it is exactly one 4-tuple; 12 garbled lines dropped), as
# {G: (n, ...)}; next_n is 1 in all 622. The contexts ahead own n - G of the output rows.
_DUMP_DECODE_CALLS = {
    1: (
        1,
        968,
        4057,
        4069,
    ),
    2: (
        2,
        510,
    ),
    3: (3,),
    4: (4,),
    5: (5,),
    6: (6,),
    7: (
        7,
        4066,
        4073,
    ),
    8: (8,),
    9: (4054,),
    10: (4059,),
    16: (
        16,
        4050,
    ),
    17: (4093,),
    18: (4097,),
    19: (4093,),
    24: (24,),
    27: (
        1583,
        1607,
        1666,
    ),
    28: (
        1054,
        1631,
        1651,
        1671,
        1827,
    ),
    29: (
        1042,
        1071,
        1106,
        1135,
        1144,
        1185,
        1195,
        1212,
        1213,
        1229,
        1233,
        1238,
        1245,
        1257,
        1261,
        1262,
        1268,
        1273,
        1278,
        1282,
        1283,
        1290,
        1299,
        1306,
        1331,
        1341,
        1342,
        1349,
        1350,
        1355,
        1372,
        1375,
        1377,
        1385,
        1386,
        1389,
        1393,
        1403,
        1413,
        1423,
        1435,
        1437,
        1454,
        1460,
        1461,
        1483,
        1512,
        1696,
    ),
    30: (
        480,
        650,
        667,
        672,
        686,
        696,
        702,
        707,
        717,
        723,
        725,
        727,
        728,
        731,
        734,
        739,
        741,
        742,
        746,
        749,
        754,
        761,
        762,
        763,
        766,
        768,
        770,
        771,
        776,
        778,
        782,
        783,
        785,
        786,
        787,
        789,
        790,
        791,
        792,
        793,
        796,
        797,
        798,
        799,
        800,
        801,
        802,
        803,
        804,
        805,
        806,
        807,
        810,
        811,
        813,
        814,
        815,
        816,
        817,
        819,
        820,
        822,
        825,
        826,
        827,
        828,
        830,
        832,
        833,
        834,
        837,
        838,
        840,
        842,
        843,
        844,
        845,
        847,
        848,
        849,
        850,
        851,
        852,
        853,
        854,
        857,
        860,
        861,
        862,
        863,
        864,
        865,
        866,
        867,
        868,
        869,
        870,
        871,
        872,
        873,
        874,
        875,
        876,
        877,
        878,
        879,
        880,
        881,
        882,
        883,
        884,
        885,
        886,
        887,
        888,
        889,
        890,
        891,
        892,
        893,
        895,
        896,
        897,
        898,
        899,
        900,
        901,
        902,
        903,
        904,
        905,
        906,
        907,
        908,
        909,
        910,
        912,
        914,
        915,
        916,
        917,
        919,
        920,
        921,
        922,
        925,
        927,
        928,
        929,
        930,
        931,
        932,
        933,
        934,
        936,
        938,
        939,
        940,
        941,
        942,
        943,
        944,
        945,
        948,
        951,
        954,
        956,
        957,
        958,
        959,
        960,
        961,
        962,
        964,
        965,
        966,
        967,
        968,
        971,
        972,
        973,
        975,
        976,
        977,
        980,
        984,
        985,
        986,
        989,
        990,
        992,
        995,
        997,
        998,
        999,
        1003,
        1006,
        1007,
        1012,
        1013,
        1014,
        1016,
        1017,
        1018,
        1019,
        1020,
        1021,
        1022,
        1023,
        1026,
        1028,
        1031,
        1033,
        1034,
        1036,
        1038,
        1039,
        1043,
        1047,
        1052,
        1055,
        1056,
        1066,
        1068,
        1071,
        1076,
        1082,
        1083,
        1088,
        1090,
        1097,
        1100,
        1103,
        1109,
        1111,
        1112,
        1119,
        1126,
        1143,
        1159,
        1204,
        1236,
    ),
    31: (
        277,
        278,
        283,
        296,
        305,
        306,
        308,
        309,
        312,
        316,
        317,
        320,
        321,
        324,
        326,
        327,
        328,
        329,
        330,
        332,
        334,
        335,
        337,
        338,
        339,
        340,
        342,
        343,
        344,
        345,
        346,
        347,
        348,
        349,
        351,
        352,
        353,
        354,
        355,
        356,
        357,
        358,
        359,
        361,
        362,
        363,
        364,
        365,
        366,
        367,
        368,
        369,
        370,
        371,
        372,
        373,
        374,
        375,
        376,
        377,
        378,
        379,
        380,
        381,
        382,
        383,
        384,
        385,
        386,
        387,
        388,
        390,
        391,
        392,
        393,
        394,
        395,
        396,
        397,
        398,
        399,
        400,
        401,
        402,
        403,
        404,
        405,
        406,
        407,
        408,
        409,
        410,
        411,
        412,
        413,
        414,
        415,
        416,
        417,
        418,
        419,
        420,
        421,
        422,
        423,
        424,
        425,
        426,
        427,
        428,
        429,
        430,
        431,
        432,
        433,
        434,
        435,
        436,
        437,
        438,
        439,
        440,
        441,
        442,
        443,
        444,
        445,
        446,
        447,
        448,
        449,
        450,
        451,
        452,
        453,
        454,
        455,
        456,
        457,
        458,
        459,
        460,
        461,
        462,
        463,
        464,
        465,
        466,
        467,
        468,
        469,
        470,
        471,
        472,
        473,
        474,
        475,
        476,
        477,
        478,
        479,
        480,
        481,
        482,
        483,
        484,
        485,
        486,
        487,
        488,
        489,
        490,
        491,
        492,
        493,
        494,
        495,
        496,
        497,
        498,
        499,
        500,
        501,
        502,
        503,
        504,
        505,
        506,
        507,
        508,
        509,
        510,
        511,
        512,
        513,
        514,
        515,
        516,
        517,
        518,
        519,
        520,
        521,
        522,
        523,
        524,
        525,
        526,
        527,
        528,
        529,
        530,
        531,
        532,
        533,
        534,
        535,
        536,
        537,
        538,
        539,
        540,
        541,
        542,
        543,
        544,
        545,
        546,
        547,
        548,
        549,
        550,
        551,
        552,
        553,
        554,
        555,
        556,
        558,
        559,
        560,
        562,
        563,
        564,
        565,
        566,
        567,
        569,
        570,
        571,
        572,
        573,
        574,
        575,
        576,
        577,
        578,
        580,
        581,
        583,
        584,
        585,
        586,
        588,
        590,
        591,
        592,
        593,
        594,
        595,
        596,
        598,
        599,
        600,
        601,
        602,
        603,
        604,
        605,
        606,
        608,
        609,
        611,
        616,
        617,
        619,
        620,
        622,
        623,
        630,
        631,
        633,
        636,
        639,
        644,
        649,
        661,
        667,
        680,
        681,
        781,
    ),
    32: (32,),
}


# Every (G, next_n, num_total_compressed_tokens[2], max_seq_len) the runtime-value dumps of the
# envelope engines recorded for this op: four engines at max_batch_size 256, max_num_tokens
# 16384, L 32768 with DSpark (CUDA graph, eager, TP4, DEP4); one at L 131072 without DSpark; one
# at L 1,048,576 with DSpark. The DEP4 engine's ranks see different steps, so a value recorded by
# one rank only is kept. The contexts ahead own n - G * ceil(next_n / 2) of the output rows.
_ENVELOPE_DUMP_DECODE_CALLS = [
    (1, 1, 1, 131072),
    (1, 1, 15, 131072),
    (1, 6, 3, 32768),
    (1, 6, 3, 1048576),
    (1, 6, 17, 32768),
    (1, 6, 17, 1048576),
    (1, 6, 2279, 32768),
    (1, 6, 5217, 32768),
    (1, 6, 7891, 32768),
    (1, 6, 7893, 32768),
    (1, 6, 8148, 32768),
    (2, 1, 2, 131072),
    (2, 6, 6, 32768),
    (2, 6, 6, 1048576),
    (3, 6, 9, 32768),
    (4, 1, 4, 131072),
    (4, 6, 12, 32768),
    (4, 6, 12, 1048576),
    (8, 1, 8, 131072),
    (8, 6, 24, 32768),
    (8, 6, 24, 1048576),
    (10, 6, 30, 32768),
    (16, 1, 16, 131072),
    (16, 6, 48, 32768),
    (18, 6, 54, 32768),
    (22, 6, 4617, 32768),
    (22, 6, 4759, 32768),
    (23, 6, 4594, 32768),
    (24, 1, 24, 131072),
    (24, 6, 72, 32768),
    (25, 6, 4233, 32768),
    (28, 6, 84, 32768),
    (32, 1, 32, 131072),
    (32, 6, 96, 32768),
    (38, 6, 114, 32768),
    (40, 6, 120, 32768),
    (40, 6, 8144, 32768),
    (48, 6, 144, 32768),
    (55, 6, 165, 32768),
    (56, 6, 168, 32768),
    (64, 6, 192, 32768),
    (72, 6, 216, 32768),
    (74, 6, 8159, 32768),
    (77, 6, 231, 32768),
    (80, 6, 240, 32768),
    (88, 6, 264, 32768),
    (91, 6, 273, 32768),
    (96, 6, 288, 32768),
    (104, 6, 312, 32768),
    (104, 6, 8153, 32768),
    (112, 6, 336, 32768),
    (116, 6, 8152, 32768),
    (117, 6, 8155, 32768),
    (120, 6, 360, 32768),
    (128, 6, 384, 32768),
    (130, 6, 390, 32768),
    (133, 6, 8158, 32768),
    (142, 6, 426, 32768),
    (143, 6, 4677, 32768),
    (144, 6, 4680, 32768),
    (192, 6, 576, 32768),
    (256, 6, 768, 32768),
]


class _GenStep:
    """One mixed step as this op sees it. ``ctx``: [(cached, new)] of the contexts ahead
    (they only set L and O); ``starts``: each generation request's history sp; ``prior``:
    per request the start of a real earlier op call feeding the same request (None: the
    parked token and rejected-draft garbage are planted instead); ``seq``: the engine's
    max_seq_len L (16384 .. 1048576), which sets the tables' width ``W`` and the largest
    generation kv length (_envelope_kv_max)."""

    def __init__(self, ctx, starts, next_n, prior, seed, seq=16384):
        self.ctx, self.starts, self.next_n, self.prior, self.seed = ctx, starts, next_n, prior, seed
        self.seq = seq
        self.W = _ENVELOPE_WIDTH_OF_L[seq]
        self.G = len(starts)
        self.L = sum(n for _, n in ctx)
        self.O = sum((c + n) // RATIO - c // RATIO for c, n in ctx)
        lm = seq + (14 if next_n > 1 else 0)
        assert self.L + self.G * next_n <= _ENV_TOKENS and len(ctx) + self.G <= _ENV_REQS
        assert all(c + n <= lm for c, n in ctx)
        assert all(1 <= s and s + next_n <= _envelope_kv_max(seq, next_n) for s in starts)


def _rand_contexts(k: int, L: int, cpu, seq_max: int = 16384) -> List[Tuple[int, int]]:
    """k contexts of L tokens in all (uniform cuts), random cached prefixes (0, odd, even)
    with each context ending by ``seq_max``."""
    cuts = sorted(torch.randperm(L - 1, generator=cpu)[: k - 1].add(1).tolist()) if k > 1 else []
    lens = [b - a for a, b in zip([0] + cuts, cuts + [L])]
    ctx = []
    for n in lens:
        u = float(torch.rand(1, generator=cpu))
        c = 0 if u < 0.25 else int(torch.randint(0, seq_max - n + 1, (1,), generator=cpu))
        ctx.append((c, n))
    return ctx


def _rand_histories(G: int, next_n: int, cpu, seq: int = 16384) -> List[int]:
    """G histories at max_seq_len ``seq``: page-edge and envelope-edge values (the largest
    start _envelope_kv_max - next_n and its tail) mixed with uniform ones, both parities."""
    top = _envelope_kv_max(seq, next_n) - next_n
    edges = [1, 2, 126, 127, 128, 129, 255, 256, seq - next_n - 1, seq - next_n, top - 1, top]
    out = []
    for _ in range(G):
        u = float(torch.rand(1, generator=cpu))
        if u < 0.25:
            out.append(edges[int(torch.randint(0, len(edges), (1,), generator=cpu))])
        elif u < 0.4:
            out.append(int(torch.randint(max(top - 256, 1), top + 1, (1,), generator=cpu)))
        else:
            out.append(int(torch.randint(1, top + 1, (1,), generator=cpu)))
    return out


def _random_gen_step(i: int) -> _GenStep:
    """Seeded random step i: next_n 1 / 6 alternating, max_seq_len L random among the six
    envelope values; the first steps pin the envelope's corners (G 1 and 256, lead 0, the
    full 16384-token step, a full batch of 256 behind one context, and 255 contexts ahead of
    one request), then every G 1..256 at both next_n, a random number of contexts ahead (none
    in one step of five) of random total length up to what the step has left, and the parked
    tokens from a real earlier call in every other step pair."""
    cpu = torch.Generator().manual_seed(40_000 + i)
    next_n = 6 if i % 2 else 1
    seq = _ENVELOPE_L[int(torch.randint(0, len(_ENVELOPE_L), (1,), generator=cpu))]
    if i < len(_ENV_CORNERS):
        G, lead = _ENV_CORNERS[i]
    else:
        G, lead = 1 + ((i - len(_ENV_CORNERS)) // 2) % _ENV_REQS, None  # every G at both next_n
    room = _ENV_TOKENS - G * next_n
    k_max = _ENV_REQS - G
    if lead == 0 or k_max == 0 or (lead is None and float(torch.rand(1, generator=cpu)) < 0.2):
        ctx = []
    else:
        k = (
            1
            if lead == 1
            else k_max
            if lead == -2
            else int(torch.randint(1, k_max + 1, (1,), generator=cpu))
        )
        L = (
            room
            if lead in (-1, -2) or float(torch.rand(1, generator=cpu)) < 0.15
            else int(torch.randint(k, room + 1, (1,), generator=cpu))
        )
        ctx = _rand_contexts(min(k, L), L, cpu, seq)
    starts = _rand_histories(G, next_n, cpu, seq)
    prior = None
    if (i // 2) % 2:
        # the same requests' previous step: decode at sp - 1, or verify at sp - 1 - accepted
        acc = torch.randint(0, next_n, (G,), generator=cpu).tolist()
        prior = [max(s - 1 - a, 1) if s > 1 else None for s, a in zip(starts, acc)]
    return _GenStep(ctx, starts, next_n, prior, 40_000 + i, seq)


def _dump_gen_step(j: int, G: int, n: int, next_n: int = 1, seq: int = 16384) -> _GenStep:
    """The dump's (G, next_n, n) at max_seq_len ``seq`` as a step: contexts ahead owning
    exactly n - G * ceil(next_n / 2) output rows within the 16384-token budget (random count,
    cached parities and order), random histories, parked tokens from a real earlier call in
    every other step."""
    cpu = torch.Generator().manual_seed(60_000 + j)
    ctx_rows = n - G * math.ceil(next_n / RATIO)
    ctx = []
    if ctx_rows:
        k = int(torch.randint(1, min(_ENV_REQS - G, ctx_rows) + 1, (1,), generator=cpu))
        cuts = (
            sorted(torch.randperm(ctx_rows - 1, generator=cpu)[: k - 1].add(1).tolist())
            if k > 1
            else []
        )
        rows = [b - a for a, b in zip([0] + cuts, cuts + [ctx_rows])]
        odd = [bool(torch.randint(0, 2, (1,), generator=cpu)) for _ in rows]
        if sum(2 * r - int(o) for r, o in zip(rows, odd)) + G * next_n > _ENV_TOKENS:
            odd = [True] * len(rows)  # the shortest contexts owning those rows: 2r - 1 tokens
        for r, o in zip(rows, odd):
            new = 2 * r - 1 if o else 2 * r
            c = 2 * int(torch.randint(0, (seq - new) // 2, (1,), generator=cpu)) + int(o)
            ctx.append((c, new))
    starts = _rand_histories(G, next_n, cpu, seq)
    prior = [s - 1 if s > 1 else None for s in starts] if j % 2 else None
    step = _GenStep(ctx, starts, next_n, prior, 60_000 + j, seq)
    assert step.O == ctx_rows and step.O + G * math.ceil(next_n / RATIO) == n, (G, n, step.O)
    return step


class _EnvGenPool:
    """Random distinct kv / score pages of a small pool for every page a request's
    positions [lo, hi) touch, in tables ``width`` columns wide; every other table column
    points at a trap page."""

    def __init__(
        self, pool: torch.Tensor, spans: Sequence[Tuple[int, int]], seed: int, width: int = 132
    ):
        ps = pool.shape[1]
        cpu = torch.Generator().manual_seed(seed)
        free = (torch.randperm(pool.shape[0] - 1, generator=cpu) + 1).tolist()
        self.trap = 0
        self.bt_kv = torch.full((len(spans), width), self.trap, dtype=torch.int32)
        self.bt_sc = torch.full_like(self.bt_kv, self.trap)
        used = {self.trap}
        for b, (lo, hi) in enumerate(spans):
            for q in range(lo // ps, (hi - 1) // ps + 1):
                self.bt_kv[b, q] = free.pop()
                self.bt_sc[b, q] = free.pop()
                used |= {int(self.bt_kv[b, q]), int(self.bt_sc[b, q])}
        self.bt_kv, self.bt_sc = self.bt_kv.to(DEV), self.bt_sc.to(DEV)
        self.mapped = torch.tensor(sorted(used), device=DEV)


def _gen_call(
    pool_t, bt_kv, bt_sc, kv_score, starts, next_n, lead_rows, lead_out, out, G=None, ape=None
):
    """The target's call: generation rows at absolute rows lead_rows + b * next_n of the
    step's kv_score, outputs from absolute row lead_out of the step's output buffer."""
    G = len(starts) if G is None else G
    n_out = [(s + next_n) // RATIO - s // RATIO for s in starts]
    cu_out = [lead_out]
    for o in n_out:
        cu_out.append(cu_out[-1] + o)
    ape = _zero_ape() if ape is None else ape
    op(
        kv_score,
        ape,
        pool_t,
        pool_t,
        bt_kv[:G],
        bt_sc[:G],
        out,
        _i32(s + next_n for s in starts),
        _i32(lead_rows + b * next_n for b in range(G + 1)),
        _i32(cu_out),
        G,
        pool_t.shape[1],
        HEAD_DIM,
        RATIO,
        next_n,
    )
    return cu_out


def _run_gen_step(
    step: _GenStep,
    pool_t: torch.Tensor,
    starts_override=None,
    op_lead_shift: int = 0,
    pool_cls=None,
):
    """Run one envelope step on ``pool_t`` and check it: fp32 rows = reference at the
    default tolerances, the target's bf16 output = RNE of the fp32 rows bit for bit, the
    contexts' output rows and every row past the packed outputs untouched, the context
    rows of kv_score never read (NaN), and the state exactly each position's last value
    with nothing else written. Returns (fp32 rows, bf16 rows, reference rows).
    ``starts_override`` / ``op_lead_shift`` (controls only) hand the op other starts or
    another kv_score lead than the truth's. ``pool_cls`` replaces the table builder
    (default ``_EnvGenPool``; the DSpark-tail steps use ``_DSparkTailPool``)."""
    G, next_n, ps = step.G, step.next_n, pool_t.shape[1]
    gen = _gen(step.seed)
    ape = _zero_ape()
    prior = step.prior or [None] * G
    garbage = 6 if step.prior is None and next_n > 1 else 0  # rejected drafts left behind a rewind
    spans = []
    for s, p in zip(step.starts, prior):
        lo = min(s - 1, p - 1 if p is not None else s - 1)
        hi = max(s + next_n + garbage, p + next_n if p is not None else 0)
        spans.append((max(lo, 0), hi))
    env = (pool_cls or _EnvGenPool)(pool_t, spans, step.seed, step.W)
    pool_t[env.mapped] = SENTINEL
    # Per-request truth arrays over positions [lo, hi) (_Rows): what each position last received.
    kvs, scores = [], []
    for lo, hi in spans:
        kv, sc = _Rows(lo, hi), _Rows(lo, hi)
        kv.t, sc.t = _rand_rows(hi - lo, gen)
        kvs.append(kv)
        scores.append(sc)
    written = [set() for _ in range(G)]
    if step.prior is not None:
        # The earlier call: each request with a prior start p feeds [p, p + next_n) (at
        # next_n 6 some of it past this step's start: drafts this step's rewind overwrites).
        idx = [b for b in range(G) if prior[b] is not None]
        for b in idx:
            p = prior[b]
            if p % RATIO:  # its own parked token, planted
                pos = p - 1
                pool_t[int(env.bt_kv[b, pos // ps]), pos % ps] = kvs[b][pos]
                pool_t[int(env.bt_sc[b, pos // ps]), pos % ps] = scores[b][pos]
                written[b].add(pos)
        if idx:
            rows = torch.cat(
                [
                    torch.cat(
                        [
                            kvs[b][prior[b] : prior[b] + next_n],
                            scores[b][prior[b] : prior[b] + next_n],
                        ],
                        1,
                    )
                    for b in idx
                ]
            ).contiguous()
            out = torch.full(
                (len(idx) * math.ceil(next_n / RATIO) + 1, HEAD_DIM), float("nan"), device=DEV
            )
            cu = _gen_call(
                pool_t,
                env.bt_kv[idx].contiguous(),
                env.bt_sc[idx].contiguous(),
                rows,
                [prior[b] for b in idx],
                next_n,
                0,
                0,
                out,
            )
            torch.cuda.synchronize()
            for j, b in enumerate(idx):
                p = prior[b]
                ref = _pool_windows(kvs[b], scores[b], ape, p // RATIO, (p + next_n) // RATIO)
                torch.testing.assert_close(
                    out[cu[j] : cu[j + 1]], ref, msg=lambda m: f"earlier call b={b}: {m}"
                )
                written[b] |= set(range(p, p + next_n))
        # This step's positions get new tokens (at next_n 6 over the earlier call's rejected
        # drafts: the rewind); positions past them keep the earlier call's leftovers.
        for b, s in enumerate(step.starts):
            kvs[b][s : s + next_n], scores[b][s : s + next_n] = _rand_rows(next_n, gen)
    for b, s in enumerate(step.starts):
        if s % RATIO and (s - 1) not in written[b]:  # planted parked token
            pos = s - 1
            pool_t[int(env.bt_kv[b, pos // ps]), pos % ps] = kvs[b][pos]
            pool_t[int(env.bt_sc[b, pos // ps]), pos % ps] = scores[b][pos]
            written[b].add(pos)
        for pos in range(s + next_n, s + next_n + garbage):  # a rejected draft's leftovers
            g_kv, g_sc = _rand_rows(1, gen)
            pool_t[int(env.bt_kv[b, pos // ps]), pos % ps] = g_kv[0]
            pool_t[int(env.bt_sc[b, pos // ps]), pos % ps] = g_sc[0]
            kvs[b][pos], scores[b][pos] = g_kv[0], g_sc[0]
            written[b].add(pos)
        if garbage:  # and garbage over this step's own positions, which it must overwrite
            for pos in range(s, s + next_n):
                pool_t[int(env.bt_kv[b, pos // ps]), pos % ps] = 1e30
        written[b] |= set(range(s, s + next_n))
    torch.cuda.synchronize()

    # This step's kv_score: L NaN context rows, then the generation rows.
    rows = torch.cat(
        [
            torch.cat([kv[s : s + next_n], sc[s : s + next_n]], 1)
            for kv, sc, s in zip(kvs, scores, step.starts)
        ]
    )
    kv_score = torch.cat(
        [torch.full((step.L, 2 * HEAD_DIM), float("nan"), device=DEV), rows]
    ).contiguous()
    reserved = G * math.ceil(next_n / RATIO)
    op_starts = step.starts if starts_override is None else starts_override
    outs = {}
    for dt in (torch.float32, torch.bfloat16):
        buf = torch.full((step.O + reserved + 1, HEAD_DIM), float("nan"), device=DEV, dtype=dt)
        buf[: step.O] = SENTINEL
        cu = _gen_call(
            pool_t,
            env.bt_kv,
            env.bt_sc,
            kv_score,
            op_starts,
            next_n,
            step.L + op_lead_shift,
            step.O,
            buf[: step.O + reserved],
        )
        outs[dt] = (buf, cu)
    torch.cuda.synchronize()
    ref = torch.cat(
        [
            _pool_windows(kvs[b], scores[b], ape, s // RATIO, (s + next_n) // RATIO)
            for b, s in enumerate(step.starts)
        ]
    )
    N = ref.shape[0]
    for dt, (buf, cu) in outs.items():
        assert (buf[: step.O] == SENTINEL).all(), "a context output row was written"
        assert torch.isnan(buf[cu[-1] :]).all(), "a row past the packed outputs was written"
        assert cu[-1] - step.O == N, (cu[-1], step.O, N)
    f32 = outs[torch.float32][0][step.O : step.O + N]
    b16 = outs[torch.bfloat16][0][step.O : step.O + N]
    torch.testing.assert_close(f32, ref)
    assert torch.equal(b16, f32.bfloat16()), "bf16 output != RNE(fp32 output)"
    # State: every position a call wrote holds its last value; nothing else changed.
    n_slots = 0
    for b in range(G):
        pos = torch.tensor(sorted(written[b]), device=DEV)
        pk = env.bt_kv[b].long()[pos // ps]
        pc = env.bt_sc[b].long()[pos // ps]
        assert torch.equal(pool_t[pk, pos % ps], kvs[b][pos]), f"request {b}: kv state"
        assert torch.equal(pool_t[pc, pos % ps], scores[b][pos]), f"request {b}: score state"
        n_slots += 2 * pos.numel()
    assert int((pool_t[env.mapped] != SENTINEL).any(-1).sum()) == n_slots, (
        "a slot outside the named positions changed"
    )
    assert (pool_t[env.trap] == SENTINEL).all(), "the trap page was written"
    return f32, b16, ref


@pytest.mark.parametrize("group", range(_ENV_GROUPS))
def test_random_partitions_of_the_envelope(group: int):
    """Over the engine envelope: seeded random mixed steps -- G in 1..256
    generation requests (every value at both next_n) at next_n 1 and 6 behind 0..256-G
    contexts of random total length up to what the 16384-token step leaves (so leads L up to
    16383 and output offsets O up to ~8200), at a random max_seq_len L of the six envelope
    values (tables 132 .. 8196 columns wide), random histories up to the envelope's largest
    kv length (both parities, page edges and the tail included), parked tokens from a real
    earlier call (decode, or a verify whose rejected drafts this step rewinds over) or
    planted with rejected-draft garbage behind and over them. Each step: fp32 = reference,
    bf16 = RNE(fp32), context and trailing output rows untouched, state exact, nothing else
    written (``_run_gen_step``)."""
    steps = [_random_gen_step(i) for i in range(group, _ENV_PARTITIONS, _ENV_GROUPS)]
    pool_t = torch.empty(_ENV_PAGES, 128, HEAD_DIM, device=DEV)
    exact = total = 0
    for step in steps:
        _, b16, ref = _run_gen_step(step, pool_t)
        exact += int((b16 == ref.bfloat16()).sum())
        total += b16.numel()
    frac = exact / max(total, 1)
    Gs = sorted({s.G for s in steps})
    print(
        f"[envelope random group {group}] {len(steps)} steps, G {Gs[0]}..{Gs[-1]} ({len(Gs)} values), leads "
        f"{min(s.L for s in steps)}..{max(s.L for s in steps)}, O "
        f"{min(s.O for s in steps)}..{max(s.O for s in steps)}, "
        f"contexts up to {max(len(s.ctx) for s in steps)}, widths {sorted({s.W for s in steps})}, histories up to "
        f"{max(max(s.starts) for s in steps)}; bf16 vs reference bit-exact {frac:.6f}"
    )
    assert frac >= 0.999, frac  # the contract's bf16 gate (see test_bf16_output)


def test_random_partitions_cover_the_envelope():
    """The seeded steps reach the envelope's corners: every G in 1..256 at both next_n, lead
    0 and the full 16384-token step, up to 255 contexts ahead and full batches of 256, every
    max_seq_len (table width) at both next_n, odd and even starts from both a real earlier
    call and a planted token, verify rewinds, histories at 1 and at the largest start
    (_envelope_kv_max - next_n) of the 1,048,576-token envelope."""
    steps = [_random_gen_step(i) for i in range(_ENV_PARTITIONS)]
    for nn in (1, 6):
        sub = [s for s in steps if s.next_n == nn]
        want = set(range(1, _ENV_REQS + 1))
        assert {s.G for s in sub} == want, (nn, sorted(want - {s.G for s in sub}))
        assert any(s.L == 0 for s in sub) and any(s.L + s.G * nn == _ENV_TOKENS for s in sub)
        assert {s.W for s in sub} == set(_ENVELOPE_WIDTHS)
        assert any(1 in s.starts for s in sub)
        assert any(
            _envelope_kv_max(_ENVELOPE_L[-1], nn) - nn in s.starts
            for s in sub
            if s.seq == _ENVELOPE_L[-1]
        )
    assert max(len(s.ctx) for s in steps) == _ENV_REQS - 1 and any(
        len(s.ctx) + s.G == _ENV_REQS for s in steps
    )
    assert len(steps) >= 200
    odd_real = sum(
        1
        for s in steps
        if s.prior
        for b, x in enumerate(s.starts)
        if x % 2 and s.prior[b] is not None
    )
    odd_planted = sum(1 for s in steps if not s.prior for x in s.starts if x % 2)
    rewinds = sum(
        1
        for s in steps
        if s.prior and s.next_n == 6
        for b, x in enumerate(s.starts)
        if s.prior[b] is not None and s.prior[b] + 6 > x
    )
    assert odd_real and odd_planted and rewinds
    print(
        f"[coverage] {len(steps)} steps: G 1..{_ENV_REQS} at next_n 1 and 6, leads 0..{max(s.L for s in steps)}, "
        f"contexts up to {max(len(s.ctx) for s in steps)}, widths {sorted({s.W for s in steps})}, histories up to "
        f"{max(max(s.starts) for s in steps)}; odd starts parked by a real call {odd_real}, planted "
        f"{odd_planted}; verify rewinds over rejected drafts {rewinds}"
    )


def test_every_dumped_decode_step():
    """The dumps: every (G, next_n, n) the runtime-value dumps recorded for this
    op -- the gates' 622 (next_n 1, max_seq_len 16384) and the envelope engines' 62
    (batch 256 at L 32768 with DSpark, L 131072 without, L 1,048,576 with; G up to 256, n up
    to 8159) -- built as a step with exactly those values (contexts ahead owning
    n - G * ceil(next_n / 2) output rows, histories at that max_seq_len) and run as the
    random steps are."""
    pool_t = torch.empty(_ENV_PAGES, 128, HEAD_DIM, device=DEV)
    cells = [(G, 1, n, 16384) for G, ns in sorted(_DUMP_DECODE_CALLS.items()) for n in ns]
    assert len(cells) == 622
    cells += _ENVELOPE_DUMP_DECODE_CALLS
    assert len(_ENVELOPE_DUMP_DECODE_CALLS) == 62
    exact = total = 0
    for j, (G, next_n, n, seq) in enumerate(cells):
        _, b16, ref = _run_gen_step(_dump_gen_step(j, G, n, next_n, seq), pool_t)
        exact += int((b16 == ref.bfloat16()).sum())
        total += b16.numel()
    frac = exact / max(total, 1)
    print(
        f"[envelope dump] {len(cells)} dumped steps (G up to {max(c[0] for c in cells)}, n up to "
        f"{max(c[2] for c in cells)}, max_seq_len up to {max(c[3] for c in cells)}) == reference; "
        f"bf16 bit-exact {frac:.6f}"
    )
    assert frac >= 0.999, frac


def test_kernel_variant_map_over_the_envelope():
    """The profiler's launches over steps spread across the envelope (every
    fifth random step plus its corners: G 1..256, next_n 1 and 6, leads 0..16383, tables
    132 .. 8196 columns wide, histories up to 1,048,589, bf16 and fp32 output). Each call
    launches exactly one kernel, always the same instantiation
    (pagedKvCompressKernel<512, 4, 4, 2, 1, false>: head_dim 512, fp32 kv_score, fp32 state,
    ratio 2, one reduction warp, uniform next_n -- the ragged variant runs only when the op is
    given `new_tokens_per_seq`, which the entry never passes), with grid (G, 4, 1) and 32 threads -- the host picks the
    variant from head_dim, dtypes and ratio only, never from G, next_n, the lead or the
    lengths; the output dtype is a runtime argument. Every step profiled here is also
    checked against the reference (``_run_gen_step``, outside the profile)."""
    import json
    import os
    import tempfile

    steps = [_random_gen_step(i) for i in range(0, _ENV_PARTITIONS, 5)]
    steps += [_random_gen_step(i) for i in range(len(_ENV_CORNERS))]
    pool_t = torch.empty(_ENV_PAGES, 128, HEAD_DIM, device=DEV)
    launches, expected = [], []
    for step in steps:
        _run_gen_step(step, pool_t)
        env = _EnvGenPool(
            pool_t, [(max(s - 1, 0), s + step.next_n) for s in step.starts], step.seed, step.W
        )
        kv_score = torch.randn(
            step.L + step.G * step.next_n, 2 * HEAD_DIM, generator=_gen(step.seed), device=DEV
        )
        reserved = step.G * math.ceil(step.next_n / RATIO)
        for dt in (torch.bfloat16, torch.float32):
            out = torch.empty(step.O + reserved, HEAD_DIM, device=DEV, dtype=dt)
            launches.append((env, kv_score, step, out))
            expected.append([step.G, 4, 1])
    zero_ape = _zero_ape()  # built before the profile starts, so every kernel in it is the op's
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for env, kv_score, step, out in launches:
            _gen_call(
                pool_t,
                env.bt_kv,
                env.bt_sc,
                kv_score,
                step.starts,
                step.next_n,
                step.L,
                step.O,
                out,
                ape=zero_ape,
            )
        torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "trace.json")
        prof.export_chrome_trace(path)
        trace = json.load(open(path))
    kernels = [e for e in trace["traceEvents"] if e.get("cat") == "kernel"]
    ops = sorted([e for e in kernels if "pagedKvCompress" in e["name"]], key=lambda e: e["ts"])
    names = {e["name"] for e in ops}
    grids = [e["args"]["grid"] for e in ops]
    blocks = {tuple(e["args"]["block"]) for e in ops}
    others = sorted({e["name"] for e in kernels if "pagedKvCompress" not in e["name"]})
    print(
        f"[envelope variant map] {len(ops)} op launches over {len(steps)} steps x 2 output dtypes: kernels "
        f"{sorted(names)}, blocks {sorted(blocks)}, {len({g[0] for g in grids})} distinct G in grids "
        f"{grids[:2]}..{grids[-2:]} (G up to {max(g[0] for g in grids)}); leads {min(s.L for s in steps)}.."
        f"{max(s.L for s in steps)}; widths {sorted({s.W for s in steps})}; other kernels {others}"
    )
    assert others == [] and len(kernels) == len(launches), (
        "the op launched something besides one kernel per call"
    )
    assert len(names) == 1, names
    (name,) = names
    assert "pagedKvCompressKernel<512,4,4,2,1,false>" in name.replace(" ", ""), name
    assert blocks == {(32, 1, 1)}, blocks
    assert grids == expected, "grid differs from (G, 4, 1)"


def test_envelope_controls():
    """The envelope checks see the call-site defects that matter here: generation rows read
    one row early (a lead off by one), starts one position late (kv_lens off by one), and
    a verify rewind the op is not told about (the previous step's starts) each fail
    ``_run_gen_step`` on a mixed verify step where the correct call passes."""
    pool_t = torch.empty(_ENV_PAGES, 128, HEAD_DIM, device=DEV)
    step = _GenStep(
        [(0, 1000), (129, 301), (4095, 3)],
        [33, 128, 1001, 4094, 16001],
        6,
        [30, 125, None, 4090, 15999],
        77_001,
    )
    _run_gen_step(step, pool_t)
    wrong = {
        "lead one row early": dict(op_lead_shift=-1),
        "starts one late": dict(starts_override=[s + 1 for s in step.starts]),
        "untold rewind": dict(
            starts_override=[p if p is not None else s for s, p in zip(step.starts, step.prior)]
        ),
    }
    for name, kw in wrong.items():
        with pytest.raises(AssertionError):
            _run_gen_step(step, pool_t, **kw)
    print(f"[envelope controls] {sorted(wrong)}: all rejected")


# ---------------------------------------------------------------------------
# The DSpark envelope at max_seq_len 16384
# ---------------------------------------------------------------------------
# With DSpark (max_draft_len 5) and the overlap scheduler on (every gate), the executor widens
# the engine's max_seq_len to 16384 + 5 + 4 + 5 = 16398 (_ENGINE_MAX_SEQ_DSPARK). A request's
# committed tokens stay <= 16384 (the executor keeps the net max_seq_len), so contexts stay below
# position 16384, but a verify step at the end of a full-length request writes generation
# positions up to 16397: logical page 128 of the compressor-state tables, whose width stays the
# manager's max_blocks_per_seq = 132. The steps below put generation requests there.
_DSPARK_MAX_POS = _ENGINE_MAX_SEQ_DSPARK - 1
_DSPARK_COLS = 132
_DSPARK_STEPS = 48


class _DSparkTailStep(_GenStep):
    """A _GenStep whose generation positions may reach 16397 (contexts still end by 16384)."""

    def __init__(self, ctx, starts, next_n, prior, seed):
        self.ctx, self.starts, self.next_n, self.prior, self.seed = ctx, starts, next_n, prior, seed
        self.seq, self.W = _ENGINE_MAX_SEQ, _DSPARK_COLS
        self.G = len(starts)
        self.L = sum(n for _, n in ctx)
        self.O = sum((c + n) // RATIO - c // RATIO for c, n in ctx)
        assert self.L + self.G * next_n <= _ENV_TOKENS and len(ctx) + self.G <= _ENV_REQS
        assert all(c + n <= _ENGINE_MAX_SEQ for c, n in ctx)
        assert all(1 <= s and s + next_n - 1 <= _DSPARK_MAX_POS for s in starts)


class _DSparkTailPool(_EnvGenPool):
    """_EnvGenPool on the manager's 132-column tables: every page a request's positions touch is
    mapped to distinct random kv / score pages, except logical page 128 (positions 16384..16511),
    which is mapped to the pool's LAST pages (kv P-1-2b, score P-2-2b); every other column
    points at the trap page 0."""

    def __init__(
        self,
        pool: torch.Tensor,
        spans: Sequence[Tuple[int, int]],
        seed: int,
        width: int = _DSPARK_COLS,
    ):
        assert width == _DSPARK_COLS
        ps = pool.shape[1]
        P = pool.shape[0]
        cpu = torch.Generator().manual_seed(seed)
        far = set(range(P - 2 * len(spans), P))
        free = [p for p in (torch.randperm(P - 1, generator=cpu) + 1).tolist() if p not in far]
        self.trap = 0
        self.bt_kv = torch.full((len(spans), _DSPARK_COLS), self.trap, dtype=torch.int32)
        self.bt_sc = torch.full_like(self.bt_kv, self.trap)
        self.col128 = 0
        for b, (lo, hi) in enumerate(spans):
            for q in range(lo // ps, (hi - 1) // ps + 1):
                if q == 128:
                    self.bt_kv[b, q], self.bt_sc[b, q] = P - 1 - 2 * b, P - 2 - 2 * b
                    self.col128 += 1
                else:
                    self.bt_kv[b, q] = free.pop()
                    self.bt_sc[b, q] = free.pop()
        self.bt_kv, self.bt_sc = self.bt_kv.to(DEV), self.bt_sc.to(DEV)
        self.mapped = torch.tensor(
            sorted(
                {self.trap}
                | set(self.bt_kv.flatten().tolist())
                | set(self.bt_sc.flatten().tolist())
            ),
            device=DEV,
        )
        _DSparkTailPool.last = self


def _dspark_tail_step(i: int) -> _DSparkTailStep:
    """Seeded mixed step i at the DSpark tail: next_n 1 / 6 alternating, G 1..256 (32 in the
    first two steps), contexts ahead as in the envelope steps, each generation request's start drawn
    from the tail [16384 - next_n - 8, 16398 - next_n] (positions up to 16397, both parities) with
    probability 3/4 and uniformly otherwise (request 0 always ends at 16397 or 16396), and the
    parked tokens from a real earlier call in every other step pair (a verify rewind at next_n 6)."""
    cpu = torch.Generator().manual_seed(70_000 + i)
    next_n = 6 if i % 2 else 1
    G = 32 if i < 2 else int(torch.randint(1, _ENV_REQS + 1, (1,), generator=cpu))
    room = _ENV_TOKENS - G * next_n
    k_max = _ENV_REQS - G
    ctx = []
    if k_max and float(torch.rand(1, generator=cpu)) < 0.8:
        k = int(torch.randint(1, k_max + 1, (1,), generator=cpu))
        L = int(torch.randint(k, room + 1, (1,), generator=cpu))
        ctx = _rand_contexts(min(k, L), L, cpu)
    top = _DSPARK_MAX_POS + 1 - next_n
    starts = []
    for _ in range(G):
        if float(torch.rand(1, generator=cpu)) < 0.75:
            starts.append(
                int(torch.randint(_ENGINE_MAX_SEQ - next_n - 8, top + 1, (1,), generator=cpu))
            )
        else:
            starts.append(int(torch.randint(1, top + 1, (1,), generator=cpu)))
    starts[0] = top - (i // 2) % 2
    prior = None
    if (i // 2) % 2:
        acc = torch.randint(0, next_n, (G,), generator=cpu).tolist()
        prior = [max(s - 1 - a, 1) if s > 1 else None for s, a in zip(starts, acc)]
    return _DSparkTailStep(ctx, starts, next_n, prior, 70_000 + i)


def test_dspark_tail_mixed_steps():
    """_DSPARK_STEPS seeded mixed steps with generation requests at the DSpark tail (positions up to
    16397 on logical page 128 of 132-column tables, that page at the pool's far end), next_n 1 and
    6, behind random contexts, odd starts parked by a real earlier call (verify rewinds included)
    or planted, run and checked as the envelope steps are (``_run_gen_step``): fp32 = reference, bf16 =
    RNE(fp32), context and trailing output rows untouched, state exact, nothing else written."""
    steps = [_dspark_tail_step(i) for i in range(_DSPARK_STEPS)]
    pool_t = torch.empty(_ENV_PAGES, 128, HEAD_DIM, device=DEV)
    exact = total = col128 = tail_rows = 0
    parities, rewinds = set(), 0
    for step in steps:
        _, b16, ref = _run_gen_step(step, pool_t, pool_cls=_DSparkTailPool)
        col128 += _DSparkTailPool.last.col128
        exact += int((b16 == ref.bfloat16()).sum())
        total += b16.numel()
        tail_rows += sum(step.next_n for s in step.starts if s + step.next_n - 1 >= _ENGINE_MAX_SEQ)
        parities |= {s % RATIO for s in step.starts if s + step.next_n - 1 >= _ENGINE_MAX_SEQ}
        if step.prior and step.next_n == 6:
            rewinds += sum(
                1 for s, p in zip(step.starts, step.prior) if p is not None and p + 6 > s
            )
    frac = exact / max(total, 1)
    top = max(s + st.next_n - 1 for st in steps for s in st.starts)
    print(
        f"[DSpark tail] {len(steps)} steps, G {min(s.G for s in steps)}..{max(s.G for s in steps)}, "
        f"leads {min(s.L for s in steps)}..{max(s.L for s in steps)}, {tail_rows} generation rows at positions "
        f">= 16384 (top {top}, start parities {sorted(parities)}), {col128} requests on logical page 128 at the "
        f"pool's far end, verify rewinds {rewinds}; bf16 bit-exact {frac:.6f}"
    )
    assert top == _DSPARK_MAX_POS and parities == {0, 1} and col128 > 0 and rewinds > 0
    assert frac >= 0.999, frac


@pytest.mark.parametrize("next_n", [1, 6])
def test_dspark_tail_captured_on_the_engine_pool(next_n: int):
    """The largest graph batch (G 32) at the DSpark tail, captured on the ENGINE's compressor-state
    pool with 132-column tables whose logical page 128 is mapped among the pool's last pages: step
    0 is the eager warm-up (checked), then 3 replays advance every request (request 0 by a full
    accepted block, ending at position 16397; the others by 1 + random 0..5 accepted drafts, so
    replays rewind over rejected drafts at either parity). Every replay equals the reference and
    an eager call bit for bit, and the pool finally holds exactly each position's last value."""
    pool_t = _engine_state_pool()
    ape = _zero_ape()
    steps, G = 3, 32
    cpu = torch.Generator().manual_seed(71_000 + next_n)
    adv = next_n  # request 0's advance per step: 1 + (next_n - 1) accepted drafts
    c0 = _DSPARK_MAX_POS - (next_n - 1) - steps * adv
    cached = [c0] + sorted(torch.randint(c0 - 40, c0 + 1, (G - 1,), generator=cpu).tolist())
    # both start parities from step 0 on, and no request past request 0 (the envelope's end)
    cached[1] = c0 if c0 % 2 else c0 - 1
    cached[2] = c0 - 3 if c0 % 2 else c0 - 2
    assert max(cached) == c0
    extra = (steps + 1) * next_n
    gen = _gen(71_100 + next_n)
    kvs, scores = _truth_rows(cached, extra, gen)
    pool = _FarEndPool(
        pool_t,
        [(max(c - 1, 0), c + extra) for c in cached],
        seed=71_200 + next_n,
        width=_DSPARK_COLS,
    )
    assert pool.bt_kv.shape[1] == _DSPARK_COLS and (pool.bt_kv[:, 128] != pool.trap).any()
    for b, c in enumerate(cached):
        if c % RATIO:
            pool.put(b, c - 1, kvs[b][c - 1], scores[b][c - 1])
    reserved = G * math.ceil(next_n / RATIO)
    s_kv_score = torch.zeros(G * next_n, 2 * HEAD_DIM, device=DEV)
    s_kv_lens = torch.zeros(G, dtype=torch.int32, device=DEV)
    s_cu_seq = torch.arange(G + 1, device=DEV, dtype=torch.int32) * next_n
    s_cu_out = torch.zeros(G + 1, dtype=torch.int32, device=DEV)
    s_out = torch.zeros(reserved, HEAD_DIM, device=DEV)
    e_out = torch.zeros_like(s_out)

    def load(starts):
        s_kv_score.copy_(
            torch.cat(
                [
                    torch.cat([kv[s : s + next_n], sc[s : s + next_n]], 1)
                    for kv, sc, s in zip(kvs, scores, starts)
                ]
            )
        )
        s_kv_lens.copy_(_i32(s + next_n for s in starts))
        n_out = [(s + next_n) // 2 - s // 2 for s in starts]
        s_cu_out.copy_(_i32([0] + torch.tensor(n_out).cumsum(0).tolist()))
        return n_out

    def call(out):
        op(
            s_kv_score,
            ape,
            pool.kv,
            pool.score,
            pool.bt_kv,
            pool.bt_score,
            out,
            s_kv_lens,
            s_cu_seq,
            s_cu_out,
            G,
            pool.page_size,
            HEAD_DIM,
            RATIO,
            next_n,
        )

    def check(starts, n_out, got):
        off = 0
        for b, s in enumerate(starts):
            ref = _pool_windows(kvs[b], scores[b], ape, s // 2, (s + next_n) // 2)
            torch.testing.assert_close(got[off : off + n_out[b]], ref, msg=lambda m: f"b {b}: {m}")
            off += n_out[b]
        assert torch.isnan(got[off:]).all(), "rows past the packed outputs written"

    starts = list(cached)
    n_out = load(starts)
    s_out.fill_(float("nan"))
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        call(s_out)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    check(starts, n_out, s_out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call(s_out)
    parities = set()
    for _ in range(steps):
        accepted = torch.randint(0, next_n, (G,), generator=cpu).tolist() if next_n > 1 else [0] * G
        accepted[0] = next_n - 1
        starts = [s + 1 + a for s, a in zip(starts, accepted)]
        for b, s in enumerate(starts):
            kvs[b][s : s + next_n], scores[b][s : s + next_n] = _rand_rows(next_n, gen)
            parities.add(s % RATIO)
        n_out = load(starts)
        s_out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        check(starts, n_out, s_out)
        e_out.fill_(float("nan"))
        call(e_out)
        torch.cuda.synchronize()
        assert torch.equal(s_out.view(torch.int32), e_out.view(torch.int32)), "replay != eager"
    assert starts[0] + next_n - 1 == _DSPARK_MAX_POS, starts[0]
    last = [s + next_n for s in starts]
    pool.check(
        [
            (b, kvs[b], scores[b], range(max(c - 1, 0) if c % RATIO else c, last[b]))
            for b, c in enumerate(cached)
        ]
    )
    del graph
    print(
        f"[DSpark tail capture next_n={next_n}] G 32 at positions up to {_DSPARK_MAX_POS} on 132-column tables "
        f"(page 128 at the far end of the [{pool_t.shape[0]}, 128, 512] engine pool): replays == reference "
        f"and == eager bitwise; start parities {sorted(parities)}"
    )
