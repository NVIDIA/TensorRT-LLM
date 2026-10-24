# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Certify `spec_decode/dspark_markov_chain`'s LL exchange variant (``enable_ll_exchange``, kernel
``dspark_markov_chain_kernel_b1_ll``): its batch-1 draft tokens equal the S4B8 batch-1 kernel's
(``dspark_markov_chain_kernel_b1``) bit for bit, on real ranks over the multicast workspace.

Every rank builds the same full-vocabulary tensors from one seed and passes its padded shard to the op; every rank
checks LL == b1 on its own tokens, rank 0 also checks both against ``reference``. Checks (batch 1, K = 3 and 7, the
M3.1 vocabulary):

- inputs: random logits, logits on a coarse grid (many exact ties), all-equal values (zero ``markov_w2``: every
  position's token is 0), NaN-free extremes (+-bf16 max, +-the smallest bf16 denormal, +-0.0, -inf);
- forced ties at fixed vocabulary columns (their ``markov_w2`` rows zeroed, so the value is the logit exactly): across
  ranks at one local column, the last real column of rank r against the first of rank r + 1, the last row of CTA 0
  against the first of CTA 1, two warps of one CTA, the last real column of every rank; the tied value a large
  finite value, the bf16 maximum, the smallest bf16 denormal (every other column -inf) or +-0.0 (every other column
  -inf). The token must be the lowest tied column at every position;
- stale-tag safety: 64 back-to-back launches with changing inputs (no host sync), K = 7 and K = 3 interleaved with
  batch-2 calls of the Lamport kernel, then 60 back-to-back K = 7 launches; every launch's tokens equal the b1
  kernel's on its inputs;
- a CUDA graph of four LL launches replayed 16 times with new inputs copied in before each replay, against eager b1;
- TAILBK_SPEC cached inputs (``enable_cached_inputs``): tokens with the cache equal tokens without it (LL batch 1 and
  the Lamport kernel at batch 2, eager and in a CUDA graph); the cache holds ones / arange / zeros, is reused, never
  written, and a shape first met under capture allocates without caching;
- TAILBK_SPEC KV-length rewind (``dspark_markov_chain_rewind``): its tokens equal ``dspark_markov_chain``'s and its
  ``kv_lens`` equal the stock ``kv_lens[nc:bs] -= rewind; kv_lens[nc:bs].clamp_(min=0)`` (rows outside untouched),
  for rewinds of both signs and clamped rows, eager and in a CUDA graph; ``supports_rewind`` rejects bad operands.

World 4 (one GB300 tray) is spawned from one process:

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _dspark_markov_chain_ll_op_matrix.py

World 8 spans two trays and runs as one srun task per rank:

    MASTER_ADDR=<first node> MASTER_PORT=<port> python _dspark_markov_chain_ll_op_matrix.py --srun

under ``srun --ntasks=8 --ntasks-per-node=4 --kill-on-bad-exit=1``.
"""

import importlib.util
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

import torch
import torch.distributed as dist

assert torch.cuda.is_available(), "dspark_markov_chain requires CUDA devices"

_WORKER_FLAG = "--rank-worker"
DEADLINE_S = 1500
RANK_DIM = 256
VOCAB = 200064
SEEDS = 4
BACK_TO_BACK = 64
GPUS_PER_NODE = 4
"""A GB300 tray. The mapping needs it to see a world-8 group span two nodes."""

BF16_MAX = torch.finfo(torch.bfloat16).max
BF16_DENORM_MIN = 2.0**-133  # the smallest positive bf16 subnormal

RANK = 0
WORLD = 1
entry: Any = None
chain: Any = None
WS: dict = {}


def _set_ll(on: bool) -> None:
    entry.enable_batch1_variant(True)
    entry.enable_ll_exchange(on)


def _weights(vocab: int, zero_rows: Optional[List[int]] = None, zero_w2: bool = False):
    g = torch.Generator(device="cuda").manual_seed(1234)
    w1 = (torch.randn(vocab, RANK_DIM, device="cuda", generator=g) * 0.05).bfloat16()
    w2 = (torch.randn(vocab, RANK_DIM, device="cuda", generator=g) * 0.05).bfloat16()
    if zero_w2:
        w2.zero_()
    if zero_rows:
        w2[torch.tensor(zero_rows, device="cuda")] = 0
    return w1, w2


class _Shard:
    """One vocabulary's shards: this rank's padded `w2` rows and logits, and every rank's for the reference."""

    def __init__(self, vocab: int, **weights):
        self.vocab = vocab
        self.shard = vocab // WORLD
        self.rows = entry.padded_rows(self.shard, entry.CERTIFIED_BATCH)
        assert self.rows >= self.shard, f"no padded shard for {self.shard} rows"
        self.w1, w2 = _weights(vocab, **weights)
        self.w2s_all = [
            entry.pad_w2_shard(w2[r * self.shard : (r + 1) * self.shard].contiguous(), self.rows)
            for r in range(WORLD)
        ]
        self.w2s = self.w2s_all[RANK]

    def base(self, full: torch.Tensor, rank: int) -> torch.Tensor:
        b = entry.pad_logits_buffer(full.shape[0], full.shape[1], self.rows, full.device)
        b[:, :, : self.shard] = full[:, :, rank * self.shard : (rank + 1) * self.shard]
        return b

    def call(self, base: torch.Tensor, prev0: torch.Tensor) -> torch.Tensor:
        return chain(base, prev0, self.w1, self.w2s, RANK * self.shard, **WS)

    def both(self, base: torch.Tensor, prev0: torch.Tensor):
        _set_ll(False)
        b1 = self.call(base, prev0)
        _set_ll(True)
        ll = self.call(base, prev0)
        return b1, ll

    def reference(self, full: torch.Tensor, prev0: torch.Tensor) -> torch.Tensor:
        bases = [self.base(full, r) for r in range(WORLD)]
        return chain.reference(bases, prev0, self.w1, self.w2s_all, self.shard)

    def rows_per_cta(self, block: int) -> int:
        grid = entry.pick_grid(self.rows, block, 1)
        assert grid > 0
        return self.rows // grid


def _inputs(seed: int, batch: int, block: int, vocab: int, kind: str):
    g = torch.Generator(device="cuda").manual_seed(seed)
    base = torch.randn(batch, block, vocab, device="cuda", generator=g) * 3
    if kind == "ties":
        base = torch.round(base * 2) / 2
    elif kind == "equal":
        base = torch.ones_like(base)
    elif kind == "extremes":
        pool = torch.tensor(
            [
                BF16_MAX,
                -BF16_MAX,
                BF16_DENORM_MIN,
                -BF16_DENORM_MIN,
                0.0,
                -0.0,
                float("-inf"),
                1.0,
                -1.0,
            ],
            device="cuda",
        )
        base = pool[torch.randint(0, pool.numel(), base.shape, device="cuda", generator=g)]
    prev0 = torch.randint(0, vocab, (batch,), device="cuda", generator=g)
    return base.bfloat16(), prev0


def _assert_equal(what: str, ll: torch.Tensor, b1: torch.Tensor) -> None:
    assert torch.equal(ll, b1), f"{what}: LL {ll.tolist()} != b1 {b1.tolist()}"


def check_ll_matches_b1() -> None:
    sh = _Shard(VOCAB)
    sh_eq = _Shard(VOCAB, zero_w2=True)
    for block in (3, 7):
        for kind in ("random", "ties", "equal", "extremes"):
            s = sh_eq if kind == "equal" else sh
            for seed in range(SEEDS):
                full, prev0 = _inputs(seed, 1, block, s.vocab, kind)
                b1, ll = s.both(s.base(full, RANK), prev0)
                _assert_equal(f"K {block} {kind} seed {seed}", ll, b1)
                if kind == "equal":
                    assert ll.eq(0).all(), f"all-equal values: tokens {ll.tolist()} are not all 0"
                if RANK == 0:
                    ref = s.reference(full, prev0)
                    _assert_equal(f"K {block} {kind} seed {seed} vs reference", ll, ref)
    dist.barrier()
    if RANK == 0:
        print(
            f"[ll] world={WORLD} LL == b1 (random, ties, all-equal, extremes; K 3 / 7)", flush=True
        )


def _tie_patterns(shard: int, rc: int) -> List[List[int]]:
    """Global vocabulary columns that tie, one list per pattern (each pattern's token: its lowest column)."""
    last = (
        shard - 1
    )  # the last real local column (in CTA last // rc, the last CTA holding real rows)
    rows_per_warp = rc // 8
    pats = [
        [r * shard + 5 for r in range(1, WORLD)],  # across ranks at one local column (not rank 0)
        [0 * shard + last, 1 * shard + 0],  # last real column of rank 0 vs first column of rank 1
        [
            (WORLD - 1) * shard + 0,
            (WORLD - 2) * shard + last,
        ],  # ... of rank W - 2 vs rank W - 1, listed high first
        [rc, rc - 1],  # rank 0: CTA 0's last row vs CTA 1's first row
        [
            2 * shard + rc,
            2 * shard + rc - 1,
            3 % WORLD * shard + rc,
        ],  # the same across CTAs, on other ranks
        [1 * shard + rows_per_warp + 3, 1 * shard + 3],  # rank 1, CTA 0: warp 1 vs warp 0
        [r * shard + last for r in range(WORLD)][::-1],  # every rank's last real column
        [(WORLD - 1) * shard + 7, 7],  # rank W - 1 vs rank 0
    ]
    return [sorted(set(p)) for p in pats]


def check_forced_ties() -> None:
    probe = _Shard.__new__(_Shard)
    probe.shard = VOCAB // WORLD
    probe.rows = entry.padded_rows(probe.shard, entry.CERTIFIED_BATCH)
    rcs = {block: probe.rows_per_cta(block) for block in (3, 7)}
    pats_by_block = {block: _tie_patterns(probe.shard, rcs[block]) for block in (3, 7)}
    zero_rows = sorted({c for pats in pats_by_block.values() for p in pats for c in p})
    sh = _Shard(VOCAB, zero_rows=zero_rows)
    values = (
        ("large", 100.0, False),
        ("bf16max", BF16_MAX, False),
        ("denormal", BF16_DENORM_MIN, True),
        ("signed-zero", 0.0, True),
    )
    checked = 0
    for block in (3, 7):
        pats = pats_by_block[block]
        for name, tie, others_neg_inf in values:
            for offset in range(len(pats)):
                full, prev0 = _inputs(1000 + offset, 1, block, sh.vocab, "random")
                full = full.float()
                expect = []
                for k in range(block):
                    cols = pats[(offset + k) % len(pats)]
                    if others_neg_inf:
                        full[0, k, :] = float("-inf")
                    idx = torch.tensor(cols, device="cuda")
                    if name == "signed-zero":
                        # -0.0 and +0.0 alternate over the tied columns: equal under the order.
                        signs = torch.tensor(
                            [(-0.0 if i % 2 == 0 else 0.0) for i in range(len(cols))], device="cuda"
                        )
                        full[0, k, idx] = signs
                    else:
                        full[0, k, idx] = tie
                    expect.append(min(cols))
                full = full.bfloat16()
                b1, ll = sh.both(sh.base(full, RANK), prev0)
                what = f"forced ties {name} K {block} offset {offset}"
                _assert_equal(what, ll, b1)
                want = torch.tensor([expect], dtype=torch.int32, device=ll.device)
                assert torch.equal(ll, want), (
                    f"{what}: tokens {ll.tolist()} != lowest tied columns {expect}"
                )
                if RANK == 0:
                    _assert_equal(f"{what} vs reference", ll, sh.reference(full, prev0))
                checked += 1
    dist.barrier()
    if RANK == 0:
        print(
            f"[ll] world={WORLD} forced ties: {checked} cases, LL == b1 == lowest tied column",
            flush=True,
        )


def check_back_to_back_stale_tags() -> None:
    sh = _Shard(VOCAB)
    # Warm every shape once (compiles run outside any sequence below).
    for block, batch, ll in (
        (7, 1, True),
        (3, 1, True),
        (3, 2, True),
        (7, 1, False),
        (3, 1, False),
    ):
        full, prev0 = _inputs(1, batch, block, sh.vocab, "random")
        _set_ll(ll)
        sh.call(sh.base(full, RANK), prev0)
    torch.cuda.synchronize()
    dist.barrier()

    def run_sequence(shapes, seed0: int, tag: str) -> None:
        launches = []
        for i, (block, batch) in enumerate(shapes):
            full, prev0 = _inputs(
                seed0 + i, batch, block, sh.vocab, "ties" if i % 3 == 0 else "random"
            )
            launches.append((sh.base(full, RANK), prev0))
        torch.cuda.synchronize()
        _set_ll(True)
        outs = [sh.call(base, prev0) for base, prev0 in launches]  # back to back, no host sync
        torch.cuda.synchronize()
        _set_ll(False)
        refs = [sh.call(base, prev0) for base, prev0 in launches]
        torch.cuda.synchronize()
        for i, (got, ref) in enumerate(zip(outs, refs)):
            _assert_equal(f"{tag} launch {i} (K {shapes[i][0]}, batch {shapes[i][1]})", got, ref)

    mixed = [((7, 1), (3, 1), (7, 1), (3, 2))[i % 4] for i in range(BACK_TO_BACK)]
    run_sequence(mixed, 5000, "mixed back-to-back")
    run_sequence([(7, 1)] * 60, 7000, "K7 back-to-back")
    dist.barrier()
    if RANK == 0:
        print(
            f"[ll] world={WORLD} back-to-back: {BACK_TO_BACK} mixed + 60 K7 launches equal b1",
            flush=True,
        )


def check_cuda_graph() -> None:
    sh = _Shard(VOCAB)
    blocks = (7, 3, 7, 7)
    static = []
    for j, block in enumerate(blocks):
        full, prev0 = _inputs(300 + j, 1, block, sh.vocab, "random")
        static.append((sh.base(full, RANK), prev0.clone()))
    _set_ll(True)
    for base, prev0 in static:  # compiled outside capture
        sh.call(base, prev0)
    torch.cuda.synchronize()
    dist.barrier()
    graph_out = []
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            for base, prev0 in static:
                graph_out.append(sh.call(base, prev0))
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    dist.barrier()
    for rep in range(16):
        for j, (block, (base, prev0)) in enumerate(zip(blocks, static)):
            full, p = _inputs(
                400 + rep * 8 + j, 1, block, sh.vocab, "ties" if rep % 2 else "random"
            )
            base.copy_(sh.base(full, RANK))
            prev0.copy_(p)
        g.replay()
        torch.cuda.synchronize()
        got = [t.clone() for t in graph_out]
        _set_ll(False)
        refs = [sh.call(base, prev0) for base, prev0 in static]
        torch.cuda.synchronize()
        _set_ll(True)
        for j, (a, b) in enumerate(zip(got, refs)):
            _assert_equal(f"graph replay {rep} launch {j} (K {blocks[j]})", a, b)
    dist.barrier()
    if RANK == 0:
        print(
            f"[ll] world={WORLD} CUDA graph: 16 replays x 4 LL launches equal eager b1", flush=True
        )


def check_cached_inputs() -> None:
    sh = _Shard(VOCAB)
    shapes = ((7, 1), (3, 1), (7, 2))
    _set_ll(True)
    entry._cached_taken[0] = False
    entry.enable_cached_inputs(True)
    for block, batch in shapes:  # outside capture: compiles and fills the cache
        full, prev0 = _inputs(11, batch, block, sh.vocab, "random")
        sh.call(sh.base(full, RANK), prev0)
    torch.cuda.synchronize()
    dev = torch.device("cuda", torch.cuda.current_device())
    cached = {}
    for block, batch in shapes:
        key = (dev, batch, block)
        assert key in entry._const_inputs, f"no cached inputs for batch {batch} K {block}"
        nacc, rows, acc = entry._const_inputs[key]
        assert nacc.dtype == rows.dtype == acc.dtype == torch.int32
        assert torch.equal(nacc, torch.ones(batch, dtype=torch.int32, device=dev))
        assert torch.equal(rows, torch.arange(batch, dtype=torch.int32, device=dev))
        assert torch.equal(acc, torch.zeros(batch, block + 1, dtype=torch.int32, device=dev))
        cached[key] = (nacc, rows, acc)

    assert not entry.cached_inputs_taken(), "a cache fill alone is not the engagement"

    def both(base, prev0):
        entry.enable_cached_inputs(False)
        off = sh.call(base, prev0)
        entry.enable_cached_inputs(True)
        on = sh.call(base, prev0)
        return off, on

    for block, batch in shapes:
        for kind in ("random", "ties", "extremes"):
            for seed in range(SEEDS):
                full, prev0 = _inputs(600 + seed, batch, block, sh.vocab, kind)
                off, on = both(sh.base(full, RANK), prev0)
                _assert_equal(f"cached inputs K {block} batch {batch} {kind} seed {seed}", on, off)
    assert entry.cached_inputs_taken(), "a call with the cache filled must report the inputs taken"
    # A shape compiled with the cache off, then first met with it on under capture: allocated, not cached.
    miss = (dev, 3, 7)
    entry._const_inputs.pop(miss, None)
    entry.enable_cached_inputs(False)
    full, prev0 = _inputs(13, 3, 7, sh.vocab, "random")
    sh.call(sh.base(full, RANK), prev0)
    torch.cuda.synchronize()
    dist.barrier()

    graph_shapes = ((7, 1), (3, 1), (7, 2), (7, 3))
    static = []
    for j, (block, batch) in enumerate(graph_shapes):
        full, prev0 = _inputs(700 + j, batch, block, sh.vocab, "random")
        static.append((sh.base(full, RANK), prev0.clone()))
    entry.enable_cached_inputs(True)
    graph_out = []
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            for base, prev0 in static:
                graph_out.append(sh.call(base, prev0))
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    dist.barrier()
    assert miss not in entry._const_inputs, "a shape first met under capture was cached"
    for key, tensors in cached.items():
        assert all(a is b for a, b in zip(entry._const_inputs[key], tensors)), (
            f"cache {key} replaced"
        )
    for rep in range(8):
        for j, ((block, batch), (base, prev0)) in enumerate(zip(graph_shapes, static)):
            full, p = _inputs(
                800 + rep * 8 + j, batch, block, sh.vocab, "ties" if rep % 2 else "random"
            )
            base.copy_(sh.base(full, RANK))
            prev0.copy_(p)
        g.replay()
        torch.cuda.synchronize()
        got = [t.clone() for t in graph_out]
        entry.enable_cached_inputs(False)
        refs = [sh.call(base, prev0) for base, prev0 in static]
        torch.cuda.synchronize()
        entry.enable_cached_inputs(True)
        for j, (a, b) in enumerate(zip(got, refs)):
            _assert_equal(f"cached-inputs graph replay {rep} launch {j} {graph_shapes[j]}", a, b)
    for key, (nacc, rows, acc) in cached.items():  # never written by the kernels
        batch, block = key[1], key[2]
        assert torch.equal(nacc, torch.ones(batch, dtype=torch.int32, device=dev))
        assert torch.equal(rows, torch.arange(batch, dtype=torch.int32, device=dev))
        assert torch.equal(acc, torch.zeros(batch, block + 1, dtype=torch.int32, device=dev))
    entry.enable_cached_inputs(False)
    dist.barrier()
    if RANK == 0:
        print(
            f"[ll] world={WORLD} cached inputs: eager + CUDA graph tokens equal uncached; cache intact",
            flush=True,
        )


def _stock_rewind(kv_lens: torch.Tensor, amount: torch.Tensor, nc: int) -> torch.Tensor:
    want = kv_lens.clone()
    bs = nc + amount.numel()
    want[nc:bs] -= amount
    want[nc:bs].clamp_(min=0)
    return want


def _rewind_case(seed: int, batch: int, nc: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    kv_lens = torch.randint(0, 40, (nc + batch + 3,), dtype=torch.int32, device="cuda", generator=g)
    amount = torch.randint(-8, 9, (batch,), dtype=torch.int32, device="cuda", generator=g)
    amount[0] = int(kv_lens[nc]) + 5  # a clamped row
    return kv_lens, amount


def check_rewind() -> None:
    sh = _Shard(VOCAB)
    entry.enable_cached_inputs(False)
    for ll in (True, False):
        _set_ll(ll)
        for block, batch, nc in ((7, 1, 0), (3, 1, 2), (7, 2, 1), (7, 3, 0)):
            if batch > 1 and not ll:
                continue
            for seed in range(SEEDS):
                full, prev0 = _inputs(
                    900 + seed, batch, block, sh.vocab, "ties" if seed % 2 else "random"
                )
                base = sh.base(full, RANK)
                kv_lens, amount = _rewind_case(950 + seed * 7 + batch, batch, nc)
                want = _stock_rewind(kv_lens, amount, nc)
                ref = sh.call(base, prev0)
                assert entry.supports_rewind(kv_lens, amount, nc)
                got = entry.dspark_markov_chain_rewind(base, prev0, sh.w1, sh.w2s, RANK * sh.shard, kv_lens,
                                                       amount, nc, **WS)  # fmt: skip
                torch.cuda.synchronize()
                tag = f"rewind ll={ll} K {block} batch {batch} nc {nc} seed {seed}"
                _assert_equal(f"{tag} tokens", got, ref)
                assert torch.equal(kv_lens, want), (
                    f"{tag}: kv_lens {kv_lens.tolist()} != {want.tolist()}"
                )
    i32 = torch.zeros(8, dtype=torch.int32, device="cuda")
    assert not entry.supports_rewind(None, i32[:2], 0)
    assert not entry.supports_rewind(i32, i32[:2].long(), 0)
    assert not entry.supports_rewind(i32.long(), i32[:2], 0)
    assert not entry.supports_rewind(i32, i32[:2], 7)  # past the end
    assert not entry.supports_rewind(i32, i32[:0], 0)
    assert not entry.supports_rewind(i32[::2], i32[:2], 0)

    # A CUDA graph of the rewind launch (batch 1, K 7, nc 1), replayed with new inputs and kv_lens copied in.
    _set_ll(True)
    full, prev0 = _inputs(990, 1, 7, sh.vocab, "random")
    base = sh.base(full, RANK)
    kv_lens, amount = _rewind_case(991, 1, 1)
    entry.dspark_markov_chain_rewind(
        base, prev0, sh.w1, sh.w2s, RANK * sh.shard, kv_lens, amount, 1, **WS
    )
    torch.cuda.synchronize()
    dist.barrier()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            out = entry.dspark_markov_chain_rewind(base, prev0, sh.w1, sh.w2s, RANK * sh.shard, kv_lens,
                                                   amount, 1, **WS)  # fmt: skip
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    dist.barrier()
    for rep in range(8):
        f, p = _inputs(1000 + rep, 1, 7, sh.vocab, "ties" if rep % 2 else "random")
        base.copy_(sh.base(f, RANK))
        prev0.copy_(p)
        k, a = _rewind_case(1100 + rep, 1, 1)
        kv_lens.copy_(k)
        amount.copy_(a)
        want = _stock_rewind(k, a, 1)
        g.replay()
        torch.cuda.synchronize()
        got = out.clone()
        ref = sh.call(base, prev0)
        torch.cuda.synchronize()
        _assert_equal(f"rewind graph replay {rep} tokens", got, ref)
        assert torch.equal(kv_lens, want), (
            f"rewind graph replay {rep}: kv_lens {kv_lens.tolist()} != {want.tolist()}"
        )
    dist.barrier()
    if RANK == 0:
        print(
            f"[ll] world={WORLD} rewind: tokens equal the chain's, kv_lens equal the stock rewind",
            flush=True,
        )


def check_switches() -> None:
    _set_ll(True)
    assert entry.ll_exchange_engaged(1) and not entry.ll_exchange_engaged(2)
    entry.enable_batch1_variant(False)
    assert not entry.ll_exchange_engaged(1), "LL needs the batch-1 variant"
    _set_ll(False)
    assert not entry.ll_exchange_engaged(1)
    dist.barrier()


CHECKS = (
    check_switches,
    check_ll_matches_b1,
    check_forced_ties,
    check_back_to_back_stale_tags,
    check_cuda_graph,
    check_cached_inputs,
    check_rewind,
)


def _run_one_rank(device: int) -> int:
    global RANK, WORLD, WS, entry, chain
    os.environ.setdefault("TLLM_DISABLE_MPI", "1")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(device)
    dist.init_process_group(
        "cuda:nccl,cpu:gloo", init_method="env://", device_id=torch.device("cuda", device)
    )
    RANK, WORLD = rank, world

    from tensorrt_llm._torch._experimental.modeling_v2.catalog.spec_decode import (
        dspark_markov_chain as entry_module,
    )
    from tensorrt_llm.mapping import Mapping

    entry = entry_module
    chain = entry.dspark_markov_chain
    mapping = Mapping(world_size=world, rank=rank, gpus_per_node=GPUS_PER_NODE, tp_size=world)
    WS = entry.workspace_args(mapping, dist.group.WORLD)

    for check in CHECKS:
        try:
            check()
        except BaseException:
            import traceback

            print(f"[rank {RANK}] FAILED {check.__name__}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(13)
        print(f"[rank {RANK}] {check.__name__} passed", flush=True)
    dist.barrier()
    torch.cuda.synchronize()
    sys.stdout.flush()
    os._exit(0)


def _spawn_ranks() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the devices this run owns"
    world_size = len([d for d in visible.split(",") if d.strip()])
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(os.environ, MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world_size),
               TLLM_DISABLE_MPI="1")  # fmt: skip
    env = {k: v for k, v in env.items() if not k.startswith(("PMIX_", "PMI_", "OMPI_", "SLURM_"))}
    children = [
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG],
            env=dict(env, RANK=str(r), LOCAL_RANK=str(r)),
            start_new_session=True,
        )  # fmt: skip
        for r in range(world_size)
    ]

    def _kill_all() -> None:
        for p in children:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for p in children:
            p.wait()

    deadline = time.monotonic() + DEADLINE_S
    live = set(range(world_size))
    failure: Optional[str] = None
    while live and failure is None:
        if time.monotonic() > deadline:
            failure = f"did not finish in {DEADLINE_S}s (wedged)"
            break
        for r in sorted(live):
            code = children[r].poll()
            if code is None:
                continue
            live.discard(r)
            if code != 0:
                failure = f"rank {r} exited {code}"
                break
        time.sleep(0.5)
    if failure is not None:
        _kill_all()
        raise AssertionError(f"the {world_size}-rank run failed: {failure}")


def _rank_job():
    """`_rank_job.py` beside this file, by path (a sibling directory ships its own module of that name)."""
    path = Path(__file__).resolve().with_name("_rank_job.py")
    spec = importlib.util.spec_from_file_location("_spec_decode_rank_job", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    if "--srun" in sys.argv:
        sys.exit(_run_one_rank(device=_rank_job().adopt_srun_env()))
    if _WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank(device=int(os.environ["RANK"])))
    _spawn_ranks()
    print("OK")
