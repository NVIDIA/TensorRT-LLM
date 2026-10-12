# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the engram_hash entry of the DeepSeek-V4.1 target.

The op computes DeepSeek-V4.1's Engram n-gram hash ids on the device and keeps
each request's token history in a paged, position-indexed int32 pool, so a
step's ids need no host-side history. What is certified is bit-exactness
against the checkpoint's ``inference/engram.py`` ``NgramHashState`` semantics
across every way a request's tokens reach the op: one-shot prefill, chunked
prefill, decode, and speculative verify runs whose rejected tail is rewound.

The entry owns what is certified: ``CELLS`` is the list, ``reference`` and
``compare`` are the per-call gate (the hash ids, given the pool as it was before
the call). This file turns a cell's spec into a schedule of calls and keeps what
a cell cannot carry: the sweeps over every token count, the engine envelope,
the engine's own pools, block-table widths, the encoder-replay history call,
CUDA-graph capture, the rejections, the wrong-variant controls and the
checkpoint / golden comparisons.

Four sources of truth:

* the entry's ``reference``, one call at a time, reading the history from the
  pool it is given;
* ``_RefState``: the checkpoint's ``NgramHashState.forward`` rewritten here in
  torch ops for one request (int64 end to end, position-indexed cache, the
  cumulative DEAD / sequence-start blocking), driven with the same run schedule
  as the op. It carries each request's history itself, independently of the
  op's pool, so it is what certifies the pool writes across calls. The hash
  constants are rebuilt here with the checkpoint's recipe (numpy PCG64
  multipliers, sympy primes) and anchored to the config;
* the checkpoint's own ``inference/engram.py`` module, imported from the
  checkpoint directory with the checkpoint's tokenizer (real 99092-class map);
* the golden dumps (``golden/<case>/engram_hash.pt`` under ``$MODELING_V2_DSV41_GOLDEN_DIR``).
  These last two need external directories and skip, naming them, when absent.

Every comparison is exact (``torch.equal``) -- the output is integer ids. The
deliberately wrong variants (``pad_id`` 2 instead of ``token_map[2]``, int32 or
float64 products, non-cumulative blocking, multipliers seeded by layer index,
history not carried) run through the same comparison and must fail it.
"""

import functools
import importlib.util
import os
import types

import numpy as np
import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.embedding.engram_hash import (  # noqa: E501
    engram_hash as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "engram_hash requires a CUDA device"

# DeepSeek-V4.1-Flash text_config (config.json)
LAYER_IDS = (1, 14)
MAX_NGRAM = 4
N_HEADS = 8
ENGRAM_VOCAB = 16_000_000
COMPRESSED_VOCAB = 99_092
NUM_EMBEDDINGS = (384_006_168, 384_016_682)
PAD_TOKEN_ID = 2
VOCAB = 129_280
COLS = (MAX_NGRAM - 1) * N_HEADS
DEAD = -1
# Written into every pool slot up front. It is a plausible-looking positive id,
# so a read of a slot nobody wrote hashes to garbage instead of hiding as DEAD.
SENTINEL = 0x5A5A5A5A
DEV = "cuda"

# Per-request length L (max_seq_len) and DSpark: with DSpark the engine hands the manager
# Lm = L + 14 (max_draft_len 5), so positions run to Lm - 1; without it Lm = L. The block table the
# target passes, metadata.engram_token_block_table, is max_blocks_per_seq columns wide, which
# KVCacheManagerV2 sets to ceil((Lm + headroom) / 128) rounded up to a multiple of 4 (headroom 1,
# or 10 with DSpark): 132 / 260 / 516 / 1028 / 2052 / 8196 at the L below, DSpark on and off alike
# (test_engine_width_and_capture_at_every_L builds the manager at each and asserts it; the cg
# engine dumps show 8196 at L 1M, 1028 at L 128k and 260 at L 32k).
_L_SET = (16384, 32768, 65536, 131072, 262144, 1048576)
_L_CONFIGS = [(L, ds) for L in _L_SET for ds in (False, True)]


def _lm(L: int, ds: bool) -> int:
    return L + 14 if ds else L


def _engine_width(L: int, ds: bool) -> int:
    blocks = -(-(_lm(L, ds) + (10 if ds else 1)) // 128)
    return -(-blocks // 4) * 4


assert [_engine_width(L, ds) for L, ds in _L_CONFIGS] == [
    132,
    132,
    260,
    260,
    516,
    516,
    1028,
    1028,
    2052,
    2052,
    8196,
    8196,
]


# Reference activations dumped from the checkpoint's own inference code; no default
# location, so the cases that compare against them skip unless this is set.
_GOLDEN_DIR = os.environ.get("MODELING_V2_DSV41_GOLDEN_DIR", "")
# The checkpoint resolves under LLM_MODELS_ROOT, as the whole-model gates' does;
# MODELING_V2_DSV41_CKPT_DIR overrides it. A case that needs it skips when it is absent.
_MODELS_ROOT = llm_models_root()
_CKPT_DIR = os.environ.get("MODELING_V2_DSV41_CKPT_DIR") or (
    str(_MODELS_ROOT / "DeepSeek-V4.1-Flash-NVFP4") if _MODELS_ROOT else ""
)
_GOLDEN_CASES = (
    "capital_fr",
    "arith_chat",
    "repeat_ngram",
    "long_retrieval",
    "long_topk",
    "boot_natalia",
    "boot_capital",
    "boot_passkey",
)


# ─── hash constants, rebuilt with the checkpoint's recipe ──────────────────────


def _multipliers(layer_ids=LAYER_IDS) -> torch.Tensor:
    """engram.py compute_hash_multipliers: odd, per-layer numpy PCG64 seeded 10007 * layer_id."""
    bound = max(1, (np.iinfo(np.int64).max // COMPRESSED_VOCAB) // 2)
    rows = []
    for layer_id in layer_ids:
        g = np.random.default_rng(10007 * layer_id)
        rows.append(g.integers(low=0, high=bound, size=(MAX_NGRAM,), dtype=np.int64) * 2 + 1)
    return torch.tensor(np.stack(rows))


@functools.lru_cache(maxsize=None)
def _primes() -> torch.Tensor:
    """engram.py EngramLayout.from_args: layer -> n-gram size -> head, next unused prime above 15999999."""
    from sympy import isprime

    seen, layers = set(), []
    for _ in LAYER_IDS:
        per_ngram = []
        for _ in range(MAX_NGRAM - 1):
            cur, sizes = ENGRAM_VOCAB - 1, []
            for _ in range(N_HEADS):
                cur += 1
                while not isprime(cur) or cur in seen:
                    cur += 1
                seen.add(cur)
                sizes.append(cur)
            per_ngram.append(sizes)
        layers.append(per_ngram)
    return torch.tensor(layers, dtype=torch.int64)


def _offsets(primes: torch.Tensor) -> torch.Tensor:
    flat = primes.flatten(1)
    return torch.cat([torch.zeros_like(flat[:, :1]), flat.cumsum(1)[:, :-1]], dim=1)


@functools.lru_cache(maxsize=None)
def _synthetic_token_map() -> torch.Tensor:
    """A seeded stand-in for the tokenizer-derived map: every id -> [0, 99092), max class present."""
    g = torch.Generator().manual_seed(1234)
    m = torch.randint(0, COMPRESSED_VOCAB, (VOCAB,), generator=g, dtype=torch.int32)
    m[7] = COMPRESSED_VOCAB - 1  # the largest class, so products come within 1.1e15 of 2^63
    m[8] = COMPRESSED_VOCAB - 1
    assert int(m[PAD_TOKEN_ID]) != PAD_TOKEN_ID
    return m


class _Consts:
    def __init__(self, token_map: torch.Tensor, layer_ids=LAYER_IDS):
        self.token_map = token_map
        self.multipliers = _multipliers(layer_ids)
        self.primes = _primes()
        self.offsets = _offsets(self.primes)
        self.pad_id = int(token_map[PAD_TOKEN_ID])
        self.dev = types.SimpleNamespace(
            token_map=token_map.to(DEV),
            multipliers=self.multipliers.to(DEV),
            primes=self.primes.to(DEV),
            offsets=self.offsets.to(DEV),
        )


# ─── reference: the checkpoint's NgramHashState.forward, one request ────────────


class _RefState:
    """engram.py NgramHashState for one request, in torch ops on the CPU.

    ``variant`` selects a deliberately wrong implementation for the controls:
    ``int32`` / ``float64`` products, ``noncumulative`` blocking.
    """

    def __init__(
        self,
        consts: _Consts,
        max_len: int,
        variant=None,
        pad_id=None,
        multipliers=None,
        base: int = 0,
    ):
        """``base``: the cache holds positions [base, max_len) only (a request whose calls never
        reach below ``base``: every run starts at or after base + N - 1, or base is 0)."""
        self.c = consts
        self.variant = variant
        self.pad_id = consts.pad_id if pad_id is None else pad_id
        self.multipliers = consts.multipliers if multipliers is None else multipliers
        self.base = base
        self.cache = torch.full((max_len - base,), SENTINEL, dtype=torch.int64)

    def forward(self, ids: torch.Tensor, start_pos: int, mask=None) -> torch.Tensor:
        ids = ids.to(torch.int64)
        n = ids.numel()
        assert n == 0 or self.base == 0 or start_pos - (MAX_NGRAM - 1) >= self.base, (
            start_pos,
            self.base,
        )
        in_vocab = (ids >= 0) & (ids < self.c.token_map.numel())
        compressed = torch.where(
            in_vocab, self.c.token_map[ids.clamp(0, self.c.token_map.numel() - 1)].long(), DEAD
        )
        if mask is not None:
            compressed = torch.where(mask, compressed, DEAD)
        self.cache[start_pos - self.base : start_pos - self.base + n] = compressed
        positions = torch.arange(start_pos, start_pos + n)
        tokens, blocked = [], torch.zeros(n, dtype=torch.bool)
        for shift in range(MAX_NGRAM):
            source = self.cache[(positions - shift).clamp_min(0) - self.base]
            hit = (positions < shift) | (source == DEAD)
            blocked = hit if self.variant == "noncumulative" else blocked | hit
            tokens.append(torch.where(blocked, self.pad_id, source))
        tokens = torch.stack(tokens, dim=-1)  # [n, MAX_NGRAM]
        if self.variant == "int32":
            products = tokens.int().unsqueeze(1) * self.multipliers.int()
        elif self.variant == "float64":
            products = (tokens.double().unsqueeze(1) * self.multipliers.double()).long()
        else:
            products = tokens.unsqueeze(1) * self.multipliers  # [n, L, MAX_NGRAM]
        rolling, hashes = products[..., 0], []
        for i in range(1, MAX_NGRAM):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.c.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.c.offsets


# ─── op harness: paged pool, per-step block tables in batch order ───────────────


class _Pool:
    """A paged int32 token pool, sized independently of any batch, and each request's pages.

    ``num_pages`` is chosen by this test to be plainly larger than the requests
    need; every request owns a random, disjoint set of physical pages, and each
    step's block table is rebuilt in that step's batch order. ``lows[r]``, when
    given, is the lowest position request r's calls touch: its logical pages below
    that are unmapped (-1), as in the engine's windowed ENGRAM_TOKEN role, so a
    request a million positions long needs only the pages of its window.
    """

    def __init__(
        self, max_lens: dict, page_size: int, num_pages: int = 4096, seed: int = 0, lows=None
    ):
        self.page_size = page_size
        blocks = {r: -(-n // page_size) for r, n in max_lens.items()}
        first = {r: (lows or {}).get(r, 0) // page_size for r in max_lens}
        need = sum(blocks[r] - first[r] for r in blocks)
        num_pages = max(num_pages, 2 * need + 1)
        perm = torch.randperm(num_pages, generator=torch.Generator().manual_seed(seed))
        self.pages, at = {}, 0
        for r, nb in blocks.items():
            own = perm[at : at + nb - first[r]].to(torch.int32)
            self.pages[r] = (
                torch.cat([torch.full((first[r],), -1, dtype=torch.int32), own])
                if first[r]
                else own
            )
            at += nb - first[r]
        self.spare = perm[at:]  # never owned by a request
        self.max_blocks = max(blocks.values())
        self.pool = torch.full((num_pages, page_size), SENTINEL, dtype=torch.int32, device=DEV)

    def table(self, req_ids, override=None) -> torch.Tensor:
        """[B, max_blocks] int32; ``override[r]`` replaces request r's row (windowing tests)."""
        rows = []
        for r in req_ids:
            row = torch.full((self.max_blocks,), -1, dtype=torch.int32)
            src = self.pages[r] if override is None or r not in override else override[r]
            row[: src.numel()] = src
            rows.append(row)
        return torch.stack(rows).to(DEV)

    def slot(self, r: int, p: int):
        return int(self.pages[r][p // self.page_size]), p % self.page_size


def _call(
    consts: _Consts, pool: _Pool, runs, *, table=None, pad_id=None, dtypes=None, token_map=None
):
    """One op call over ``runs`` = [(req, ids[int64 cpu], start_pos, mask or None)], in batch order."""
    args, kwargs = _call_args(
        consts, pool, runs, table=table, pad_id=pad_id, dtypes=dtypes, token_map=token_map
    )
    return op(*args, **kwargs).cpu()


def _call_args(
    consts: _Consts, pool: _Pool, runs, *, table=None, pad_id=None, dtypes=None, token_map=None
):
    """The op's (positional args, kwargs) for one call over ``runs``, in batch order."""
    dtypes = dtypes or {}
    lens = [ids.numel() for _, ids, _, _ in runs]
    input_ids = torch.cat([ids for _, ids, _, _ in runs]).to(dtypes.get("ids", torch.int32))
    req_idx = torch.repeat_interleave(torch.arange(len(runs)), torch.tensor(lens)).to(
        dtypes.get("meta", torch.int32)
    )
    cu = torch.tensor([0] + list(np.cumsum(lens)), dtype=dtypes.get("meta", torch.int32))
    start = torch.tensor([s for _, _, s, _ in runs], dtype=dtypes.get("meta", torch.int32))
    masks = [m for _, _, _, m in runs]
    token_mask = None
    if any(m is not None for m in masks):
        token_mask = torch.cat(
            [torch.ones(n, dtype=torch.bool) if m is None else m for n, m in zip(lens, masks)]
        ).to(DEV)
    if table is None:
        table = pool.table([r for r, _, _, _ in runs])
    args = (
        input_ids.to(DEV),
        req_idx.to(DEV),
        cu.to(DEV),
        start.to(DEV),
        pool.pool,
        table.to(DEV).to(dtypes.get("table", torch.int32)),
        consts.dev.token_map if token_map is None else token_map,
        consts.dev.multipliers,
        consts.dev.primes,
        consts.dev.offsets,
        consts.pad_id if pad_id is None else pad_id,
    )
    return args, dict(token_mask=token_mask)


def _certify_schedule(consts, pool, steps, refs, *, tables=None, **kw):
    """Every call of ``steps`` through ``op`` (its ``is_valid`` armed by ``certifying``), gated
    by ``op.compare`` against ``op.reference`` on the pool as it was before the call, and every
    row also against the per-request ``_RefState`` (which certifies the pool writes, since it
    carries the history itself). ``tables[k]``, when given, is call k's block table. Returns
    the rows that differ from ``_RefState``."""
    bad = 0
    for k, runs in enumerate(steps):
        table = None if tables is None else tables[k]
        args, kwargs = _call_args(consts, pool, runs, table=table, **kw)
        before = pool.pool.clone()
        got = op(*args, **kwargs)
        op.compare(got, op.reference(*args[:4], before, *args[5:], **kwargs))
        got = got.cpu()
        at = 0
        for r, ids, start, mask in runs:
            want = refs[r].forward(ids, start, mask)
            bad += int((got[at : at + ids.numel()] != want).flatten(1).any(1).sum())
            at += ids.numel()
    return bad


def _run_schedule(consts, pool, steps, refs, **kw):
    """Run ``steps`` (lists of runs) through the op and the references; return the mismatching row count."""
    bad = 0
    for runs in steps:
        got = _call(consts, pool, runs, **kw)
        at = 0
        for r, ids, start, mask in runs:
            want = refs[r].forward(ids, start, mask)
            bad += int((got[at : at + ids.numel()] != want).flatten(1).any(1).sum())
            at += ids.numel()
    return bad


def _ids(n: int, g: torch.Generator) -> torch.Tensor:
    return torch.randint(0, VOCAB, (n,), generator=g, dtype=torch.int64)


# ─── constants ─────────────────────────────────────────────────────────────────


def test_constants_match_checkpoint_layout():
    """The rebuilt constants are the checkpoint's: multipliers, 48 distinct primes, table sizes."""
    m = _multipliers()
    assert m.tolist() == [
        [76632096046245, 4839876093313, 35959672319349, 73987337458391],
        [67716810739261, 51510806800915, 30921347202721, 82619226485591],
    ]
    p = _primes()
    assert tuple(p.shape) == (2, MAX_NGRAM - 1, N_HEADS)
    assert len(set(p.flatten().tolist())) == 48
    assert int(p[0, 0, 0]) == 16000057 and int(p[0, -1, -1]) == 16000463
    assert int(p[1, 0, 0]) == 16000477 and int(p[1, -1, -1]) == 16000889
    assert tuple(p.flatten(1).sum(1).tolist()) == NUM_EMBEDDINGS
    # the largest product fits int64, and nothing narrower
    assert (COMPRESSED_VOCAB - 1) * int(m.max()) < 2**63
    assert (COMPRESSED_VOCAB - 1) * int(m.max()) > 2**62


# ─── one-shot prefill ──────────────────────────────────────────────────────────


def _drive_prefill(spec):
    """One-shot prefill runs from position 0 in one call."""
    lens, page_size = list(spec["lens"]), spec["page_size"]
    consts = _Consts(_synthetic_token_map())
    g = torch.Generator().manual_seed(sum(lens) + page_size)
    reqs = {r: _ids(n, g) for r, n in enumerate(lens)}
    reqs[0][: min(4, lens[0])] = 7  # the largest compressed class at the sequence start
    pool = _Pool({r: n for r, n in enumerate(lens)}, page_size, seed=len(lens))
    refs = {r: _RefState(consts, n) for r, n in enumerate(lens)}
    runs = [(r, reqs[r], 0, None) for r in reqs]
    assert _certify_schedule(consts, pool, [runs], refs) == 0
    # the pool holds each token's compressed id at its position, and nothing else was written
    host = pool.pool.cpu()
    for r, ids in reqs.items():
        for p in (0, ids.numel() // 2, ids.numel() - 1):
            page, col = pool.slot(r, p)
            assert int(host[page, col]) == int(consts.token_map[ids[p]])
    assert int((host != SENTINEL).sum()) == sum(lens)


# ─── chunked prefill + decode, mixed batches ────────────────────────────────────

# request: (prompt length, chunk sizes -- the last takes the remainder, decode steps)
_CHUNKED = {
    0: (301, [128, 1, 2, 125], 6),
    1: (130, [127, 3], 5),
    2: (1027, [255, 1, 256], 4),
    3: (5, [1, 1, 1, 1], 7),
    4: (520, [259], 3),
    5: (2, [1], 9),
}


def _chunked_seqs(g: torch.Generator):
    return {r: _ids(n + d, g) for r, (n, _, d) in _CHUNKED.items()}


def _chunked_steps(seqs, masks=None):
    """Mixed steps: every live request contributes its next chunk or one decode token."""
    plan = {}
    for r, (n, chunks, d) in _CHUNKED.items():
        runs, at = [], 0
        for c in chunks:
            runs.append((at, at + c))
            at += c
        if at < n:
            runs.append((at, n))
        runs += [(n + i, n + i + 1) for i in range(d)]
        plan[r] = runs
    steps, k = [], 0
    while any(plan.values()):
        order = [r for r in plan if plan[r]]
        order = order[k % len(order) :] + order[: k % len(order)]
        ctx = [r for r in order if plan[r][0][0] < _CHUNKED[r][0]]
        gen = [r for r in order if r not in ctx]
        step = []
        for r in ctx + gen:
            a, b = plan[r].pop(0)
            m = None if masks is None else masks[r][a:b]
            step.append((r, seqs[r][a:b], a, m))
        steps.append(step)
        k += 1
    return steps


def _history_rows(steps) -> int:
    """Rows that read a predecessor from the pool: the first min(len, N - 1) rows of every run past position 0."""
    return sum(min(ids.numel(), MAX_NGRAM - 1) for runs in steps for _, ids, a, _ in runs if a > 0)


def _chunked_schedule(g: torch.Generator, masks=None):
    seqs = _chunked_seqs(g)
    return seqs, _chunked_steps(seqs, masks)


def _drive_chunked(spec):
    """The mixed chunked-prefill + decode schedule of six requests. With int32 indices: the
    history-not-carried control. With int64 ids, run indices, block table and token map: the
    same schedule (measured on another seed)."""
    page_size = spec["page_size"]
    consts = _Consts(_synthetic_token_map())
    if spec["index_dtype"] == torch.int64:
        seqs, steps = _chunked_schedule(torch.Generator().manual_seed(13))
        max_lens = {r: s.numel() for r, s in seqs.items()}
        pool = _Pool(max_lens, page_size, seed=1)
        refs = {r: _RefState(consts, n) for r, n in max_lens.items()}
        dtypes = {"ids": torch.int64, "meta": torch.int64, "table": torch.int64}
        tm64 = consts.dev.token_map.to(torch.int64)
        assert _certify_schedule(consts, pool, steps, refs, dtypes=dtypes, token_map=tm64) == 0
        return
    seqs, steps = _chunked_schedule(torch.Generator().manual_seed(7))
    max_lens = {r: s.numel() for r, s in seqs.items()}
    pool = _Pool(max_lens, page_size, seed=page_size)
    refs = {r: _RefState(consts, n) for r, n in max_lens.items()}
    assert len(steps) >= 10
    assert _certify_schedule(consts, pool, steps, refs) == 0
    # control: the same schedule with the pool cleared (DEAD) before every call, i.e. history not carried
    pool2 = _Pool(max_lens, page_size, seed=page_size)
    refs2 = {r: _RefState(consts, n) for r, n in max_lens.items()}
    bad = 0
    for runs in steps:
        pool2.pool.fill_(DEAD)
        bad += _run_schedule(consts, pool2, [runs], refs2)
    total = sum(n for n in max_lens.values())
    print(f"[control] history not carried, page_size={page_size}: {bad} of {total} rows differ")
    assert bad == _history_rows(steps)


# ─── speculative verify with rewind ─────────────────────────────────────────────


def _drive_verify(spec):
    """Runs of 1 + drafts tokens; only a random prefix is accepted and the next run starts after it."""
    drafts = spec["drafts"]
    consts = _Consts(_synthetic_token_map())
    g = torch.Generator().manual_seed(drafts)
    prompts = {0: 3, 1: 40, 2: 129, 3: 2, 4: 700, 5: 1}
    n_steps = 12
    max_lens = {r: n + n_steps * (drafts + 1) + 1 for r, n in prompts.items()}
    pool = _Pool(max_lens, 128, seed=11)
    refs = {r: _RefState(consts, n) for r, n in max_lens.items()}
    steps = [[(r, _ids(n, g), 0, None) for r, n in prompts.items()]]
    start = dict(prompts)
    for k in range(n_steps):
        order = list(prompts)[k % 6 :] + list(prompts)[: k % 6]
        step = []
        for r in order:
            step.append((r, _ids(drafts + 1, g), start[r], None))
            start[r] += 1 + int(
                torch.randint(0, drafts + 1, (1,), generator=g)
            )  # anchor + accepted drafts
        steps.append(step)
    assert _certify_schedule(consts, pool, steps, refs) == 0


# ─── pad id ────────────────────────────────────────────────────────────────────


def test_pad_id_is_the_compressed_pad_token():
    """pad_id is token_map[2], a compressed id; passing the raw id 2 is caught wherever padding occurs."""
    consts = _Consts(_synthetic_token_map())
    assert consts.pad_id != PAD_TOKEN_ID
    seqs, steps = _chunked_schedule(torch.Generator().manual_seed(17))
    max_lens = {r: s.numel() for r, s in seqs.items()}
    pool = _Pool(max_lens, 128, seed=8)
    refs = {r: _RefState(consts, n) for r, n in max_lens.items()}
    assert _run_schedule(consts, pool, steps, refs) == 0
    pool = _Pool(max_lens, 128, seed=8)
    refs = {r: _RefState(consts, n) for r, n in max_lens.items()}
    bad = _run_schedule(consts, pool, steps, refs, pad_id=PAD_TOKEN_ID)
    # every request's first min(len, 3) positions read past the sequence start
    expect = sum(min(n, MAX_NGRAM - 1) for n in max_lens.values())
    print(
        f"[control] pad_id=2 instead of token_map[2]={consts.pad_id}: {bad} of {sum(max_lens.values())} rows differ"
    )
    assert bad == expect


# ─── DEAD tokens (image spans) and out-of-range ids ─────────────────────────────


def _drive_dead(spec):
    """token_mask False and ids outside [0, V) are DEAD; blocking is cumulative and crosses calls."""
    consts = _Consts(_synthetic_token_map())
    g = torch.Generator().manual_seed(3)
    masks = {r: torch.ones(n + d, dtype=torch.bool) for r, (n, _, d) in _CHUNKED.items()}
    masks[0][0:1] = False  # first token
    masks[0][126:130] = False  # a span containing the 128 chunk boundary
    masks[0][131] = False  # right at a chunk start
    masks[1][60:61] = False  # a single dead token mid-chunk
    masks[1][126] = False  # 1 before a chunk start
    masks[2][250:255] = False  # ends exactly at the 255 chunk boundary
    masks[2][1027:1028] = False  # the first decode token
    masks[3][2] = False
    masks[4][100:200] = False
    seqs = _chunked_seqs(g)
    seqs[4][300] = VOCAB  # out of range -> DEAD
    seqs[4][302] = -5  # negative -> DEAD
    seqs[2][10] = VOCAB + 12345
    steps = _chunked_steps(seqs, masks)
    max_lens = {r: s.numel() for r, s in seqs.items()}
    pool = _Pool(max_lens, 128, seed=5)
    refs = {r: _RefState(consts, n) for r, n in max_lens.items()}
    assert _certify_schedule(consts, pool, steps, refs) == 0
    # DEAD is stored as -1 in the pool
    page, col = pool.slot(4, 300)
    assert int(pool.pool[page, col]) == DEAD
    # control: the same schedule against non-cumulative blocking
    pool2 = _Pool(max_lens, 128, seed=5)
    wrong = {r: _RefState(consts, n, variant="noncumulative") for r, n in max_lens.items()}
    bad = _run_schedule(consts, pool2, steps, wrong)
    print(f"[control] non-cumulative blocking: {bad} of {sum(max_lens.values())} rows differ")
    assert bad > 0


# ─── windowed pool: only the pages holding [start - 3, start) must be real ────────


def _drive_windowed(spec):
    """Pages outside the read/write window may be unmapped (-1: writes skipped) or a shared scratch page."""
    consts = _Consts(_synthetic_token_map())
    seqs, steps = _chunked_schedule(torch.Generator().manual_seed(9))
    ps = 128
    max_lens = {r: s.numel() for r, s in seqs.items()}
    pool = _Pool(max_lens, ps, seed=2)
    refs = {r: _RefState(consts, n) for r, n in max_lens.items()}
    scratch = int(pool.spare[0])
    tables = []
    for k, runs in enumerate(steps):
        override = {}
        for r, ids, a, _ in runs:
            b = a + ids.numel()
            keep = {p // ps for p in range(max(0, a - 3), a)} | {
                p // ps for p in range(max(0, b - 3), b)
            }
            row = pool.pages[r].clone()
            for blk in range(row.numel()):
                if blk not in keep:
                    row[blk] = -1 if (blk + k) % 2 else scratch
            override[r] = row
        tables.append(pool.table([r for r, *_ in runs], override))
    assert _certify_schedule(consts, pool, steps, refs, tables=tables) == 0
    # control: unmap the page holding start - 1 on a continuation call -> history rows are wrong
    pool3 = _Pool(max_lens, ps, seed=2)
    refs3 = {r: _RefState(consts, n) for r, n in max_lens.items()}
    bad = 0
    for runs in steps:
        override = {}
        for r, ids, a, _ in runs:
            row = pool3.pages[r].clone()
            if a > 0:
                row[(a - 1) // ps] = -1
            override[r] = row
        bad += _run_schedule(
            consts, pool3, [runs], refs3, table=pool3.table([r for r, *_ in runs], override)
        )
    print(f"[control] page of start-1 unmapped: {bad} rows differ")
    assert bad == _history_rows(steps)


# ─── CUDA graph: one decode step captured, replayed with new ids / positions ─────


def test_cuda_graph_decode_replay():
    consts = _Consts(_synthetic_token_map())
    g = torch.Generator().manual_seed(21)
    B, n_steps = 8, 20
    prompts = {r: int(torch.randint(1, 300, (1,), generator=g)) for r in range(B)}
    max_lens = {r: n + n_steps + 1 for r, n in prompts.items()}
    pool = _Pool(max_lens, 128, seed=4)
    refs = {r: _RefState(consts, n) for r, n in max_lens.items()}
    assert (
        _run_schedule(consts, pool, [[(r, _ids(n, g), 0, None) for r, n in prompts.items()]], refs)
        == 0
    )
    ids = torch.zeros(B, dtype=torch.int32, device=DEV)
    req = torch.arange(B, dtype=torch.int32, device=DEV)
    cu = torch.arange(B + 1, dtype=torch.int32, device=DEV)
    start = torch.tensor([prompts[r] for r in range(B)], dtype=torch.int32, device=DEV)
    table = pool.table(list(range(B)))
    out = torch.empty(B, len(LAYER_IDS), COLS, dtype=torch.int64, device=DEV)
    c = consts.dev

    def step():
        op(
            ids,
            req,
            cu,
            start,
            pool.pool,
            table,
            c.token_map,
            c.multipliers,
            c.primes,
            c.offsets,
            consts.pad_id,
            out=out,
        )

    # warm up (compiles) on a copy of the pool, then restore it
    saved = pool.pool.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step()
    torch.cuda.current_stream().wait_stream(s)
    pool.pool.copy_(saved)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    pool.pool.copy_(saved)
    pos = dict(prompts)
    for _ in range(n_steps):
        new = _ids(B, g)
        ids.copy_(new.to(torch.int32))
        start.copy_(torch.tensor([pos[r] for r in range(B)], dtype=torch.int32))
        graph.replay()
        got = out.cpu()
        for r in range(B):
            assert torch.equal(got[r : r + 1], refs[r].forward(new[r : r + 1], pos[r]))
            pos[r] += 1


# ─── index dtypes and the out= path ─────────────────────────────────────────────


def _drive_empty(spec):
    """T = 0, B = 0: an empty int64 result, no launch, nothing written."""
    consts = _Consts(_synthetic_token_map())
    pool = _Pool({0: 8}, spec["page_size"])
    c = consts.dev
    z = torch.empty(0, dtype=torch.int32, device=DEV)
    table = torch.empty(0, 1, dtype=torch.int32, device=DEV)
    cu = torch.zeros(1, dtype=torch.int32, device=DEV)
    args = (z, z, cu, z, pool.pool, table, c.token_map, c.multipliers, c.primes, c.offsets)
    before = pool.pool.clone()
    out = op(*args, consts.pad_id)
    assert tuple(out.shape) == (0, len(LAYER_IDS), COLS)
    assert int((pool.pool != SENTINEL).sum()) == 0
    op.compare(out, op.reference(*args[:4], before, *args[5:], consts.pad_id))


_DRIVERS = {
    "prefill": _drive_prefill,
    "chunked": _drive_chunked,
    "verify": _drive_verify,
    "dead": _drive_dead,
    "windowed": _drive_windowed,
    "empty": _drive_empty,
}


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """Every call of the cell's schedule through ``op``, ``op.reference`` and ``op.compare``,
    every row against the cross-call ``_RefState``, and the checks the configuration carries
    (pool contents, history-not-carried / non-cumulative / unmapped-page controls)."""
    certifying(op)
    spec = cell.spec
    assert (spec["L"], spec["N"], spec["H"]) == (len(LAYER_IDS), MAX_NGRAM, N_HEADS)
    _DRIVERS[spec["kind"]](spec)


def test_rejects_malformed_inputs():
    consts = _Consts(_synthetic_token_map())
    pool = _Pool({0: 8}, 128)
    c = consts.dev
    ids = torch.tensor([1, 2, 3], dtype=torch.int32, device=DEV)
    req = torch.zeros(3, dtype=torch.int32, device=DEV)
    cu = torch.tensor([0, 3], dtype=torch.int32, device=DEV)
    st = torch.zeros(1, dtype=torch.int32, device=DEV)
    tb = pool.table([0])
    good = dict(
        input_ids=ids,
        req_idx_per_token=req,
        cu_seq_lens=cu,
        start_pos=st,
        token_pool=pool.pool,
        block_table=tb,
        token_map=c.token_map,
        multipliers=c.multipliers,
        primes=c.primes,
        offsets=c.offsets,
        pad_id=consts.pad_id,
    )
    bad = {
        "ids int16": dict(input_ids=ids.to(torch.int16)),
        "ids on cpu": dict(input_ids=ids.cpu()),
        "strided ids": dict(input_ids=torch.arange(6, dtype=torch.int32, device=DEV)[::2]),
        "req short": dict(req_idx_per_token=req[:2]),
        "cu wrong length": dict(cu_seq_lens=torch.tensor([0, 3, 3], dtype=torch.int32, device=DEV)),
        "pool int64": dict(token_pool=pool.pool.to(torch.int64)),
        "multipliers int32": dict(multipliers=c.multipliers.int()),
        "primes 7 heads": dict(primes=c.primes[..., :7].contiguous()),
        "offsets short": dict(offsets=c.offsets[:, :-1].contiguous()),
        "mask not bool": dict(token_mask=torch.ones(3, dtype=torch.int32, device=DEV)),
        "strided mask": dict(token_mask=torch.ones(6, dtype=torch.bool, device=DEV)[::2]),
    }
    for name, patch in bad.items():
        with pytest.raises(ValueError, match="^engram_hash: ") as err:
            op(**{**good, **patch})
        print(f"[rejected] {name}: {err.value}")
    assert int((pool.pool != SENTINEL).sum()) == 0


# ─── every token count 1..16384 is the same per-token function ───
#
# The core target makes one call per step over all of the rank's tokens: context
# runs first (fresh prompts and chunks continuing from history), then generation
# runs (1-token decode or 1 + 5-token DSpark verify). A token's ids depend only on
# its own run's row, start and table, its predecessors and the pool; T reaches the
# kernel only as ``num_tokens``, which sizes the grid (64 tokens per program) and
# masks the last program. So every prefix of a mixed 16384-token step -- a valid call
# over the runs it touches, the last one cut short -- must return exactly the first r
# rows of the 16384-token call and leave the pool exactly as the first r tokens'
# writes do.


def _jit_variants(fn) -> int:
    """Compiled variants Triton holds for ``fn`` on every device (Triton >= 3.2 keeps them
    in ``device_caches[device][0]``, older releases in ``cache[device]``)."""
    dc = getattr(fn, "device_caches", None)
    if dc:
        return sum(len(v[0]) for v in dc.values())
    return sum(len(v) for v in fn.cache.values())


def _plant_history(consts, pool, refs, req, start, g, dead_every=7):
    """Store compressed ids for positions [start - 3, start) of ``req`` -- what earlier
    calls would have left -- in the pool and in the reference's cache; every
    ``dead_every``-th one is DEAD (an image token)."""
    for p in range(max(0, start - (MAX_NGRAM - 1)), start):
        c = int(consts.token_map[int(torch.randint(0, VOCAB, (1,), generator=g))])
        if int(torch.randint(0, dead_every, (1,), generator=g)) == 0:
            c = DEAD
        page, col = pool.slot(req, p)
        pool.pool[page, col] = c
        refs[req].cache[p - refs[req].base] = c


def _step_tensors(consts, pool, runs):
    """Device tensors of one call over ``runs`` = [(req, ids, start)] in batch order, plus
    each token's pool slot (flat index) and the compressed id it stores."""
    lens = [ids.numel() for _, ids, _ in runs]
    input_ids = torch.cat([ids for _, ids, _ in runs]).to(torch.int32)
    req_idx = torch.repeat_interleave(torch.arange(len(runs)), torch.tensor(lens)).to(torch.int32)
    cu = torch.tensor([0] + list(np.cumsum(lens)), dtype=torch.int32)
    start = torch.tensor([s for _, _, s in runs], dtype=torch.int32)
    table = pool.table([r for r, _, _ in runs])
    slots, vals = [], []
    assert table.shape[1] == pool.max_blocks
    for r, ids, s in runs:
        for o in range(ids.numel()):
            page, col = pool.slot(r, s + o)
            slots.append(page * pool.page_size + col)
        vals.append(consts.token_map[ids].to(torch.int32))
    return (
        input_ids.to(DEV),
        req_idx,
        cu.to(DEV),
        start.to(DEV),
        table,
        torch.tensor(slots, dtype=torch.int64, device=DEV),
        torch.cat(vals).to(DEV),
    )


# contexts (start, tokens): fresh prompts, chunks continuing from history, a 1-token context, chunks at
# L 32768 .. 1048576 and one ending at Lm = 1,048,590 (L 1M with DSpark, the last position 1,048,589)
_SWEEP_LM = _lm(1048576, True)
_SWEEP_CONTEXTS = [
    (0, 3001),
    (2048, 1500),
    (131, 1203),
    (5, 900),
    (0, 1),
    (777, 1531),
    (1_000_000, 2500),
    (262_000, 1700),
    (65_530, 1111),
    (_SWEEP_LM - 2000, 2000),
    (131_071, 344),
    (32_767, 29),
]
# 180 decode starts: page edges, the gates' and every L's last positions, and a spread
_SWEEP_DECODE_STARTS = (
    [1, 2, 3, 127, 128, 129, 16383, 4095, 4096]
    + [37 * k + 11 for k in range(11)]
    + [
        16397,
        32767,
        32781,
        65535,
        65549,
        131071,
        131085,
        262143,
        262157,
        1048575,
        _SWEEP_LM - 1,
        _SWEEP_LM - 2,
        1048448,
        1048447,
        524288,
        524287,
    ]
    + [6553 * k + 1021 for k in range(144)]
)
# 64 verify-run starts (runs of 1 + 5): page edges, runs ending at every L's Lm, and a spread
_SWEEP_VERIFY_STARTS = [
    1,
    2,
    126,
    127,
    16378,
    3000,
    16392,
    32776,
    65544,
    131080,
    262152,
    _SWEEP_LM - 6,
    _SWEEP_LM - 7,
    1048570,
] + [20959 * k + 777 for k in range(50)]


def test_every_token_count_is_a_prefix_of_16384():
    """A mixed 16384-token step in the engine's order at max_seq_len 1,048,576 with DSpark (Lm
    1,048,590): 12 contexts (10 continuing from history, one ending at Lm), 180 decode runs and 64
    verify runs of 1 + 5 tokens (positions up to 1,048,589, runs ending at every L's Lm); B =
    256; a block table of the engine's 8196 columns at that L, each request's pages below its
    window unmapped; DEAD tokens in the planted histories. The 16384-token call equals the
    reference. Then every prefix r = 1..16384 (the runs it touches, the last cut at r) is
    called into a row-prefix view of one poisoned output buffer: its r rows must equal the
    full call's rows, the next 64 rows keep the poison, and the whole pool must equal the
    planted histories plus exactly the first r tokens' writes. Mismatches are summed on the
    device. The kernel's compiled variants are counted: after r = 1, 2 and 16 (``num_tokens``
    == 1, not divisible by 16, divisible by 16: Triton's integer specialization) no further
    variant may appear."""
    consts = _Consts(_synthetic_token_map())
    g = torch.Generator().manual_seed(8192)
    runs = [(r, _ids(n, g), st) for r, (st, n) in enumerate(_SWEEP_CONTEXTS)]
    nc = len(runs)
    runs += [(nc + k, _ids(1, g), st) for k, st in enumerate(_SWEEP_DECODE_STARTS)]
    nd = len(runs)
    runs += [(nd + k, _ids(6, g), st) for k, st in enumerate(_SWEEP_VERIFY_STARTS)]
    n_max = sum(ids.numel() for _, ids, _ in runs)
    assert n_max == 16384 and len(runs) == 256, (n_max, len(runs))
    max_lens = {r: st + ids.numel() for r, ids, st in runs}
    assert max(max_lens.values()) == _SWEEP_LM
    lows = {r: max(0, st - (MAX_NGRAM - 1)) for r, _, st in runs}
    pool = _Pool(max_lens, 128, seed=8192, lows=lows)
    assert pool.max_blocks <= _engine_width(1048576, True)
    pool.max_blocks = _engine_width(1048576, True)
    refs = {r: _RefState(consts, n, base=lows[r]) for r, n in max_lens.items()}
    for r, _, s in runs:
        _plant_history(consts, pool, refs, r, s, g)
    planted = pool.pool.clone()
    ids_d, req_idx, cu, start, table, slots, vals = _step_tensors(consts, pool, runs)
    c = consts.dev

    full = op(
        ids_d,
        req_idx.to(DEV),
        cu,
        start,
        pool.pool,
        table,
        c.token_map,
        c.multipliers,
        c.primes,
        c.offsets,
        consts.pad_id,
    )
    at = 0
    for r, ids, s in runs:
        assert torch.equal(full[at : at + ids.numel()].cpu(), refs[r].forward(ids, s)), (r, s)
        at += ids.numel()
    expected_full = planted.clone()
    expected_full.view(-1)[slots] = vals
    assert torch.equal(pool.pool, expected_full)

    pool.pool.copy_(planted)
    expected = planted.clone()
    poison = -7
    obuf = torch.full((n_max + 64, len(LAYER_IDS), COLS), poison, dtype=torch.int64, device=DEV)
    bad_out = torch.zeros((), dtype=torch.int64, device=DEV)
    bad_pool = torch.zeros((), dtype=torch.int64, device=DEV)
    over = torch.zeros((), dtype=torch.int64, device=DEV)
    runs_of = req_idx.tolist()
    req_d = req_idx.to(DEV)
    from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.embedding.engram_hash import (  # noqa: E501
        _engram_hash_kernel as fn,
    )

    before = _jit_variants(fn)
    settled = None
    for r in range(1, n_max + 1):
        b = runs_of[r - 1] + 1
        res = op(
            ids_d[:r],
            req_d[:r],
            cu[: b + 1],
            start[:b],
            pool.pool,
            table,
            c.token_map,
            c.multipliers,
            c.primes,
            c.offsets,
            consts.pad_id,
            out=obuf[:r],
        )
        assert res.data_ptr() == obuf.data_ptr()
        expected.view(-1).index_copy_(0, slots[r - 1 : r], vals[r - 1 : r])
        bad_out += (obuf[:r] != full[:r]).sum()
        over += (obuf[r : r + 64] != poison).sum()
        bad_pool += (pool.pool != expected).sum()
        if r == 16:
            settled = _jit_variants(fn)
    torch.cuda.synchronize()
    after = _jit_variants(fn)
    print(
        f"\n  r = 1..{n_max} over {len(runs)} runs (positions up to {max(max_lens.values()) - 1}, table "
        f"{table.shape[1]} columns): {bad_out.item()} output elements differ from the {n_max}-token call, "
        f"{over.item()} guard elements written, {bad_pool.item()} pool slot-checks differ; kernel variants "
        f"{before} before, {settled} after r = 16, {after} after r = {n_max}"
    )
    assert bad_out.item() == 0 and over.item() == 0 and bad_pool.item() == 0
    assert after == settled and settled - before <= 3, (before, settled, after)


def test_prefix_sweep_harness_sees_a_one_token_shift():
    """Control for the sweep above: its row check rejects the outputs of a call whose
    tokens are one row off (ids shifted by one within a 64-token context), at r = 1..64."""
    consts = _Consts(_synthetic_token_map())
    g = torch.Generator().manual_seed(64)
    ids = _ids(65, g)
    pool = _Pool({0: 65, 1: 65}, 128, seed=64)
    full = _call(consts, pool, [(0, ids[:64], 0, None)])
    shifted = _call(consts, pool, [(1, ids[1:65], 0, None)])
    differ = [int((shifted[:r] != full[:r]).flatten(1).any(1).sum()) for r in range(1, 65)]
    print(
        f"  [control one-token shift] rows that differ at r = 1, 2, 7, 64: {differ[0]}/1, {differ[1]}/2, "
        f"{differ[6]}/7, {differ[63]}/64"
    )
    assert all(d == r for r, d in zip(range(1, 65), differ))


# ─── the (T, B) points of the engine's runtime-value dump, against the reference ───

# (contexts as (start, tokens), decode starts): the dumped (T, B) of the boot gates
_DUMP_STEPS = {
    "T6_B6": ([], [5, 9, 12, 20, 33, 71]),
    "T7_B7": ([], [5, 9, 12, 20, 33, 71, 130]),
    "T48_B1": ([(0, 48)], []),
    "T888_B1": ([(1024, 888)], []),
    "T1018_B3": ([(0, 1016)], [17, 2047]),
    "T1935_B3": ([(0, 1000), (2048, 934)], [129]),
    "T2048_B1": ([(0, 2048)], []),
}


@pytest.mark.parametrize("name", list(_DUMP_STEPS))
def test_dumped_steps_against_reference(name):
    ctx, dec = _DUMP_STEPS[name]
    consts = _Consts(_synthetic_token_map())
    g = torch.Generator().manual_seed(len(name) * 101)
    runs = [(k, _ids(n, g), s) for k, (s, n) in enumerate(ctx)]
    runs += [(len(ctx) + k, _ids(1, g), s) for k, s in enumerate(dec)]
    T, B = sum(ids.numel() for _, ids, _ in runs), len(runs)
    assert f"T{T}_B{B}" == name
    max_lens = {r: s + ids.numel() for r, ids, s in runs}
    pool = _Pool(max_lens, 128, seed=T)
    refs = {r: _RefState(consts, n) for r, n in max_lens.items()}
    for r, _, s in runs:
        _plant_history(consts, pool, refs, r, s, g)
    got = _call(consts, pool, [(r, ids, s, None) for r, ids, s in runs])
    at = 0
    for r, ids, s in runs:
        assert torch.equal(got[at : at + ids.numel()], refs[r].forward(ids, s)), (name, r)
        at += ids.numel()


# ─── the engine's own ENGRAM_TOKEN pool, far end ───
#
# The pool the core hashes into is the V4.1 cache manager's ENGRAM_TOKEN role,
# ``get_buffers(engram_token_owner, ENGRAM_TOKEN).squeeze(-1)``. Its page count comes
# from the manager's capacity planning, not from any batch. The manager below is built
# at the accuracy gate's settings (tep4 rank 0: max_seq_len 16384, max_batch_size
# 32, max_num_tokens 8192, tokens_per_block 128, block reuse on, device quota
# 72.68959879875183 GiB), and its COMPRESS / SWA pools are checked against the gate's
# runtime-value dump, so its ENGRAM_TOKEN pool is the engine's.

_ENGINE_QUOTA = int(round(72.68959879875183 * (1 << 30)))
_ENGINE_COMPRESS_R2_PAGES = (
    453118,
    453119,
    453120,
)  # the dump's ratio-2 COMPRESS views (per owner)
_ENGINE_SWA_PAGES = 137440
# The largest quota any engine at kv_cache_fraction 0.5 can hand the manager on this GPU:
# half of the GB300's total device memory (297,020,948,480 B from torch.cuda.mem_get_info;
# the engine takes 0.5 of the FREE memory after the weights, which is always less).
_MAX_QUOTA = 148_510_474_240
# ENGRAM_TOKEN pages per (max_batch_size, quota, max_num_tokens, KV storage, DSpark
# max_draft_len). The manager splits its quota among the pools by a typical step (one
# max_num_tokens context plus max_batch_size - 1 generation requests), so the ENGRAM_TOKEN share
# moves with the batch, with max_num_tokens, with DSpark and with the other roles' bytes: native
# Native KV storage (TRTLLM_DSV41_KV_STORAGE=native: SWA 528 B, main KV 288 B per entry) leaves
# more of the quota to it. The certified configuration set is the one the gates and witnesses
# launch: boot (2048, batch 8) and accuracy (8192, batch 32), DSpark off / 5, bf16 /
# native, and the native-storage witness's two legs, max_num_tokens 2048, batch 1, DSpark 5, native and
# bf16. All were probed at the _MAX_QUOTA ceiling (block reuse and encoder replay change
# nothing).
# Batch 1 leaves the most to this role: among unsplit managers the largest is the native-storage witness's
# native leg, 36,241,408 pages (18.56 GB, 4.64e9 elements, past 2^31 elements and 2^34 bytes); its
# bf16 contrast leg gives 32,243,712, the boot gate's native settings 26,345,472 and its bf16 ones
# 15,269,888. Every role's pages grow monotonically with the quota (14 quotas from 1 GiB to the
# ceiling at every probed configuration, both witness legs included).
# A pipeline-parallel (PP2) stage's manager holds half the layers, so at the same quota its
# ENGRAM_TOKEN share is larger: stage 0 (layers 0-19; the Engram history's owner, layer 1, is on
# it) built alone gives 57,999,360 pages native and 48,365,568 bf16 at (max_num_tokens 2048, batch 1,
# no DSpark, max_seq_len 16384), the maxima over the scanned engine envelope including PP2
# stages (every max_seq_len gives the same). Stage alone is the upper bound (an engine's cross-rank allreduce(MIN)
# can only cut a stage's capacity), and tp does not change the pools. B_max is that stage-0 pool.
_ENGINE_TOKEN_PAGES = {
    (32, _ENGINE_QUOTA, 8192, "bf16", 0): 2_359_296,
    (8, _ENGINE_QUOTA, 8192, "bf16", 0): 5_242_880,
    (32, _MAX_QUOTA, 8192, "bf16", 0): 4_456_448,
    (8, _MAX_QUOTA, 8192, "bf16", 0): 10_027_008,
    (8, _MAX_QUOTA, 2048, "bf16", 0): 15_269_888,
    (8, _MAX_QUOTA, 2048, "native", 0): 26_345_472,
    (1, _MAX_QUOTA, 2048, "bf16", 5): 32_243_712,
    (1, _MAX_QUOTA, 2048, "native", 5): 36_241_408,
    (1, _MAX_QUOTA, 2048, "bf16", 0, "pp2-stage0"): 48_365_568,
    (1, _MAX_QUOTA, 2048, "native", 0, "pp2-stage0"): 57_999_360,
}
_UNSPLIT_B_MAX = 36_241_408  # largest unsplit manager (native-storage witness, native leg)
_B_MAX_CONFIG = (1, _MAX_QUOTA, 2048, "native", 0)
_B_MAX_STAGE = 0
_B_MAX = _ENGINE_TOKEN_PAGES[_B_MAX_CONFIG + ("pp2-stage0",)]
# The block-table width the target passes: metadata.engram_token_block_table is a
# [max_num_sequences, max_blocks_per_seq] view (row stride = width) of the sliding block tables,
# and KVCacheManagerV2 sets max_blocks_per_seq = ceil((max_seq_len + headroom) / 128) rounded up
# to a multiple of 4: (16384 + 1) -> 129 -> 132 without DSpark, (16398 + 10) -> 129 -> 132 with
# it. 132 at every probed configuration.
_ENGINE_WIDTH = 132


def _engine_manager(
    max_batch_size: int = 32,
    quota: int = _ENGINE_QUOTA,
    max_num_tokens: int = 8192,
    storage: str = "bf16",
    draft: int = 0,
    L: int = 16384,
    pp_stage=None,
):
    """The V4.1 cache manager at these settings; ``pp_stage`` builds that stage's manager of a PP2
    mapping instead (tp 1: ``Mapping(world_size=2, rank=stage, tp_size=1, pp_size=2)``, built
    alone)."""
    from types import SimpleNamespace

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

    text = dict(
        model_type="deepseek_v41_text",
        num_hidden_layers=40,
        num_nextn_predict_layers=3,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        candidate_source_layer_id=20,
        engram_layer_ids=list(LAYER_IDS),
        engram_max_ngram_size=MAX_NGRAM,
        sliding_window=128,
        head_dim=512,
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
    # The layout reads the storage mode once, at construction.
    saved_storage = os.environ.get("TRTLLM_DSV41_KV_STORAGE")
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
            head_dim=512,
            tokens_per_block=128,
            # With DSpark the engine hands the manager max_seq_len L + 14 (16398 at 16384: 16384 + 5 + 4 + 5).
            max_seq_len=L + 14 if draft else L,
            max_batch_size=max_batch_size,
            mapping=(
                Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
                if pp_stage is None
                else Mapping(world_size=2, rank=pp_stage, tp_size=1, pp_size=2, gpus_per_node=4)
            ),
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            vocab_size=VOCAB,
            max_num_tokens=max_num_tokens,
            sparse_attn_config=sparse,
            pretrained_config=config,
            spec_config=DSparkDecodingConfig(max_draft_len=draft, block_size=draft)
            if draft
            else None,
        )
    finally:
        if saved_storage is None:
            os.environ.pop("TRTLLM_DSV41_KV_STORAGE", None)
        else:
            os.environ["TRTLLM_DSV41_KV_STORAGE"] = saved_storage
    assert mgr.v41_layout.kv_storage == storage and mgr.max_seq_len == (L + 14 if draft else L), (
        mgr.max_seq_len
    )
    assert mgr.max_blocks_per_seq == (
        _ENGINE_WIDTH if L == 16384 else _engine_width(L, bool(draft))
    ), (mgr.max_blocks_per_seq, L, draft)
    if (max_batch_size, quota, max_num_tokens, storage, draft, L) == (
        32,
        _ENGINE_QUOTA,
        8192,
        "bf16",
        0,
        16384,
    ):
        compress = [mgr.get_buffers(o, T.COMPRESS).shape[0] for o in (2, 8, 14)]
        swa = tuple(mgr.get_buffers(0, T.SWA).shape)
        assert sorted(compress) == sorted(_ENGINE_COMPRESS_R2_PAGES) and swa == (
            _ENGINE_SWA_PAGES,
            128,
            512,
        ), (
            f"this manager does not reproduce the gate's engine: COMPRESS pages {compress}, SWA {swa}"
        )
    raw = mgr.get_buffers(mgr.v41_layout.engram_token_owner, T.ENGRAM_TOKEN)
    token_pool = raw.squeeze(-1)
    assert token_pool.data_ptr() == raw.data_ptr() and token_pool.dtype == torch.int32
    key = (max_batch_size, quota, max_num_tokens, storage, draft) + (
        () if pp_stage is None else (f"pp2-stage{pp_stage}",)
    )
    want = _ENGINE_TOKEN_PAGES.get(key) if L == 16384 else None
    if pp_stage is not None:
        assert want is not None and mgr.v41_layout.engram_token_owner == LAYER_IDS[0], (
            key,
            mgr.v41_layout,
        )
    assert token_pool.stride() == (128, 1) and token_pool.shape[1] == 128, (
        tuple(token_pool.shape),
        token_pool.stride(),
    )
    assert want is None or tuple(token_pool.shape) == (want, 128), (tuple(token_pool.shape), want)
    return mgr, token_pool


class _FarEndPool(_Pool):
    """Requests' pages taken from the LAST pages of a given pool tensor (the very last
    page included), in random order; the whole tensor is set to SENTINEL first."""

    def __init__(self, pool: torch.Tensor, max_lens: dict, seed: int):
        self.page_size = pool.shape[1]
        n = pool.shape[0]
        blocks = {r: -(-m // self.page_size) for r, m in max_lens.items()}
        need = sum(blocks.values())
        assert 2 * need < n, (need, n)
        tail = torch.arange(n - 2 * need, n)
        perm = tail[torch.randperm(tail.numel(), generator=torch.Generator().manual_seed(seed))]
        perm = torch.cat([perm[perm == n - 1], perm[perm != n - 1]])  # the last page is owned
        self.pages, at = {}, 0
        for r, nb in blocks.items():
            self.pages[r] = perm[at : at + nb].to(torch.int32)
            at += nb
        self.spare = perm[at:]
        self.max_blocks = max(blocks.values())
        self.pool = pool
        self.pool.fill_(SENTINEL)


def _expected_pool(consts, pool, seqs, lens):
    """SENTINEL everywhere except each request's positions [0, lens[r]), which hold their compressed ids."""
    exp = torch.full_like(pool.pool, SENTINEL)
    for r, n in lens.items():
        for p in range(n):
            page, col = pool.slot(r, p)
            exp[page, col] = int(consts.token_map[seqs[r][p]])
    return exp


@pytest.mark.parametrize(
    "max_batch_size, quota, max_num_tokens, storage, draft, pp_stage",
    [
        (32, _ENGINE_QUOTA, 8192, "bf16", 0, None),
        (8, _ENGINE_QUOTA, 8192, "bf16", 0, None),
        (32, _MAX_QUOTA, 8192, "bf16", 0, None),
        (8, _MAX_QUOTA, 2048, "bf16", 0, None),
        (1, _MAX_QUOTA, 2048, "bf16", 5, None),
        (1, _MAX_QUOTA, 2048, "native", 5, None),
        (1, _MAX_QUOTA, 2048, "bf16", 0, 0),
    ],
    ids=[
        "bs32-gate-quota",
        "bs8-gate-quota",
        "bs32-max-quota",
        "boot-bs8-mnt2048-max-quota",
        "kvquant-witness-bf16-bs1-mnt2048-dspark5-max-quota",
        "kvquant-witness-native-bs1-mnt2048-dspark5-max-quota",
        "pp2-stage0-bf16-bs1-mnt2048-max-quota",
    ],
)
def test_engine_token_pool_far_end_chunked_and_decode(
    max_batch_size, quota, max_num_tokens, storage, draft, pp_stage
):
    """The chunked-prefill + decode schedule of six requests (the same as
    test_chunked_prefill_and_decode_continuation) on the engine's own ENGRAM_TOKEN pool,
    every request's pages among the pool's last pages (the last page owned): every call
    equals the reference, and the whole pool afterwards holds exactly each request's
    compressed ids at its positions and SENTINEL everywhere else. At max_batch_size 8
    the pool is [5242880, 128] int32 (2.68 GB), so its last pages sit past 2^31 bytes; at the
    largest quota (0.5 x device memory) and batch 32 it is [4456448, 128] (2.28 GB); at the boot
    gate's settings (max_num_tokens 2048, batch 8) and that quota it is [15269888, 128] (7.82 GB);
    at the native-storage witness's bf16 contrast leg (max_num_tokens 2048, batch 1, DSpark 5,
    TRTLLM_DSV41_KV_STORAGE=bf16) and that quota it is [32243712, 128] (16.51 GB); at its native leg
    [36241408, 128] (the unsplit managers' largest); on a PP2 stage-0 manager at (2048, batch 1, bf16)
    [48365568, 128] (24.76 GB). The largest configuration, the PP2 stage-0 manager in native storage
    [57999360, 128], is test_history_pool_size_invariance_b_max's."""
    mgr, token_pool = _engine_manager(
        max_batch_size, quota, max_num_tokens, storage, draft, pp_stage=pp_stage
    )
    try:
        consts = _Consts(_synthetic_token_map())
        seqs, steps = _chunked_schedule(torch.Generator().manual_seed(77))
        max_lens = {r: s.numel() for r, s in seqs.items()}
        pool = _FarEndPool(token_pool, max_lens, seed=77)
        refs = {r: _RefState(consts, n) for r, n in max_lens.items()}
        assert _run_schedule(consts, pool, steps, refs) == 0
        assert torch.equal(pool.pool, _expected_pool(consts, pool, seqs, max_lens))
        lowest = min(int(p.min()) for p in pool.pages.values())
        print(
            f"\n  engine ENGRAM_TOKEN pool {tuple(token_pool.shape)} int32"
            f"{'' if pp_stage is None else f' (PP2 stage {pp_stage} manager)'}: requests on pages "
            f"{lowest}..{token_pool.shape[0] - 1}; {len(steps)} calls == reference; pool exact, rest SENTINEL"
        )
    finally:
        mgr.shutdown()


# ─── capture at every graph batch, at every L's engine width, on that engine's pool ───

# The generation-only CUDA-graph batches an engine captures per rank, Gset: every batch 1..32 plus
# TRT-LLM's padded default list for cuda_graph_config {enable_padding: true, max_batch_size: 256}
# (1, 2, 4, 8, 16..128 step 8, 192, 256; the padding-off list at 256, 1..32, 64, 128, 256, is a
# subset). A decode step (DSpark off) hashes G tokens of G runs, a DSpark target-verify step 6 G
# tokens of G runs (next_n 6): T = G and 6 G, up to 1536. Under dep4 each rank hashes its own
# requests, so no gathered count reaches this op. The captured block table is the metadata's
# engram_token_block_table, max_blocks_per_seq columns (132 .. 8196 as L grows).
_GRAPH_BATCHES = tuple(range(1, 33)) + tuple(range(40, 129, 8)) + (192, 256)
assert len(_GRAPH_BATCHES) == 46


class _WindowFarEndPool(_Pool):
    """Each request's window of logical pages [lo // P, ceil(hi / P)) mapped to pages among the LAST
    pages of the given pool tensor (the very last page included), in random order; the request's
    other columns unmapped (-1), as the engine's windowed ENGRAM_TOKEN role leaves them. The whole
    tensor is set to SENTINEL first."""

    def __init__(self, pool: torch.Tensor, windows: dict, seed: int):
        self.page_size = P = pool.shape[1]
        n = pool.shape[0]
        first = {r: lo // P for r, (lo, hi) in windows.items()}
        blocks = {r: -(-hi // P) for r, (lo, hi) in windows.items()}
        need = sum(blocks[r] - first[r] for r in windows)
        assert 2 * need < n, (need, n)
        tail = torch.arange(n - 2 * need, n)
        perm = tail[torch.randperm(tail.numel(), generator=torch.Generator().manual_seed(seed))]
        perm = torch.cat([perm[perm == n - 1], perm[perm != n - 1]])  # the last page is owned
        self.pages, at = {}, 0
        for r in windows:
            k = blocks[r] - first[r]
            self.pages[r] = torch.cat(
                [torch.full((first[r],), -1, dtype=torch.int32), perm[at : at + k].to(torch.int32)]
            )
            at += k
        self.spare = perm[at:]
        self.max_blocks = max(blocks.values())
        self.pool = pool
        self.pool.fill_(SENTINEL)


@pytest.mark.parametrize(
    "L,dspark", _L_CONFIGS, ids=[f"L{L}-{'dspark' if ds else 'nodspark'}" for L, ds in _L_CONFIGS]
)
def test_engine_width_and_capture_at_every_L(L, dspark):
    """The engine at max_seq_len L (the manager gets Lm = L + 14 with DSpark), max_batch_size 256,
    max_num_tokens 16384: the V4.1 cache manager is built, its max_blocks_per_seq -- the width of
    metadata.engram_token_block_table -- must equal _engine_width (132 .. 8196) with max_seq_len
    not clamped, and on its own ENGRAM_TOKEN pool every graph batch G of Gset is captured: G
    requests, request 0 whose last replayed run ends exactly at Lm (block-table column
    (Lm - 1) // 128), request 1 whose first generated position opens a page, request 2 at the
    sequence start, the rest at random positions below Lm; histories planted before each (DEAD
    every ~7th); each request's window mapped to pages among the pool's last pages (the last
    page owned), the rest of its row -1. One generation step (G runs of next_n tokens: 1 without
    DSpark, 6 with it) is warmed up eagerly, captured, and replayed for 3 steps with ids and
    start_pos updated in place (verify: the next start is 1 + a random accepted count, request 0
    all accepted, so rejected positions are rewritten). Every replay equals an eager call of the
    same op on the same inputs and the reference, bit for bit, and differs from the replay
    before it; the whole pool afterwards equals the planted histories plus exactly the writes."""
    lm, width, next_n, n_steps = _lm(L, dspark), _engine_width(L, dspark), 6 if dspark else 1, 3
    mgr, token_pool = _engine_manager(
        max_batch_size=256, max_num_tokens=16384, draft=5 if dspark else 0, L=L
    )
    try:
        assert mgr.max_blocks_per_seq == width and mgr.max_seq_len == lm
        consts = _Consts(_synthetic_token_map())
        c = consts.dev
        replays = 0
        for G in _GRAPH_BATCHES:
            g = torch.Generator().manual_seed(7 * L + 1000 * next_n + G)
            span = (n_steps + 1) * next_n
            pos0 = {r: _ri(g, 3, lm - span) for r in range(G)}
            pos0[0] = lm - n_steps * next_n  # all drafts accepted: the last replayed run ends at Lm
            if G > 1:
                pos0[1] = 128 * _ri(g, 1, (lm - span) // 128)  # the first position opens a page
            if G > 2:
                pos0[2] = _ri(g, 0, 2)  # predecessors before the sequence start
            windows = {
                r: (max(0, p0 - (MAX_NGRAM - 1)), min(lm, p0 + span)) for r, p0 in pos0.items()
            }
            pool = _WindowFarEndPool(token_pool, windows, seed=G)
            pool.max_blocks = width
            refs = {r: _RefState(consts, hi + 8, base=lo) for r, (lo, hi) in windows.items()}
            for r in range(G):
                _plant_history(consts, pool, refs, r, pos0[r], g)
            expected = pool.pool.clone()
            T = G * next_n
            ids = torch.zeros(T, dtype=torch.int32, device=DEV)
            req = torch.arange(G, dtype=torch.int32, device=DEV).repeat_interleave(next_n)
            cu = torch.arange(0, T + 1, next_n, dtype=torch.int32, device=DEV)
            start = torch.tensor([pos0[r] for r in range(G)], dtype=torch.int32, device=DEV)
            table = pool.table(list(range(G)))
            assert tuple(table.shape) == (G, width) and table.stride() == (width, 1)
            out = torch.empty(T, len(LAYER_IDS), COLS, dtype=torch.int64, device=DEV)

            def step():
                op(
                    ids,
                    req,
                    cu,
                    start,
                    pool.pool,
                    table,
                    c.token_map,
                    c.multipliers,
                    c.primes,
                    c.offsets,
                    consts.pad_id,
                    out=out,
                )

            saved = pool.pool.clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                step()  # engine warm-up at this shape
            torch.cuda.current_stream().wait_stream(side)
            pool.pool.copy_(saved)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                step()
            pool.pool.copy_(saved)
            del saved
            pos, prev = dict(pos0), None
            for _ in range(n_steps):
                new = {r: _ids(next_n, g) for r in range(G)}
                ids.copy_(torch.cat([new[r] for r in range(G)]).to(torch.int32))
                start.copy_(torch.tensor([pos[r] for r in range(G)], dtype=torch.int32))
                out.fill_(-7)
                graph.replay()
                got = out.clone()
                step()  # the same call eagerly: rewrites the same slots with the same ids
                assert torch.equal(got, out), f"L={L} G={G} next_n={next_n}: replay != eager"
                assert prev is None or not torch.equal(got, prev), (
                    "a replay returned the previous step's ids"
                )
                prev = got
                got = got.cpu()
                runs = [(r, new[r], pos[r], None) for r in range(G)]
                for r in range(G):
                    want = refs[r].forward(new[r], pos[r])
                    assert torch.equal(got[r * next_n : (r + 1) * next_n], want), (
                        L,
                        G,
                        next_n,
                        r,
                        pos[r],
                    )
                _apply_writes(consts, pool, expected, runs)
                for r in range(G):
                    adv = next_n if r == 0 else 1 + (_ri(g, 0, next_n - 1) if next_n > 1 else 0)
                    pos[r] += adv
                replays += 1
            assert pos[0] == lm and max(pos0[r] + n_steps * next_n for r in range(G)) <= lm
            assert torch.equal(pool.pool, expected), (L, G, next_n)
            del graph, expected
        print(
            f"\n  L={L} DSpark={'on' if dspark else 'off'}: manager max_seq_len {mgr.max_seq_len}, "
            f"max_blocks_per_seq {mgr.max_blocks_per_seq}, ENGRAM_TOKEN pool {tuple(token_pool.shape)}; captured at "
            f"all {len(_GRAPH_BATCHES)} G (T = G x {next_n}, up to {max(_GRAPH_BATCHES) * next_n}), {replays} replays "
            f"== eager == reference, request 0 ending at Lm {lm} (column {(lm - 1) // 128} of {width}); pool exact"
        )
    finally:
        mgr.shutdown()


# ─── the multi-request envelope (B <= 256 runs, T <= 16384 rows, positions < Lm <= 1,048,590) ───
#
# The engine's configuration bounds one call: max_num_tokens <= 16384 rows per rank,
# max_batch_size <= 256 runs, max_seq_len L <= 1,048,576 positions per request (Lm = L + 14
# with DSpark). Inside that envelope a call is a
# partition of T rows into B runs: context chunks (fresh or continuing from history, with
# image spans DEAD through token_mask) and generation runs of 1 + D rows (D = 0 decode,
# D = 5 DSpark target verify, whose next run rewinds over the rejected tail). The runs are
# random here in number, kind, length, start, order, page placement and mask, and every
# call is checked against the reference, with the whole pool checked after every call.

_ENV_MAX_POS = 16384  # max_seq_len of the gates; the default position bound of the helpers below
_ENV_MAX_T = 16384  # max_num_tokens: 2048, 4096, 8192 or 16384 per rank
_ENV_MAX_B = 256  # max_batch_size


def _jit_keys(fn) -> set:
    """The compiled-variant keys Triton holds for ``fn`` (see _jit_variants)."""
    dc = getattr(fn, "device_caches", None)
    if dc:
        return {(str(d), k) for d, v in dc.items() for k in v[0].keys()}
    return {(str(d), k) for d, v in fn.cache.items() for k in v.keys()}


def _jit_clear(fn) -> None:
    """Drop Triton's in-memory compiled variants of ``fn`` (the next calls re-bind, from the disk cache)."""
    dc = getattr(fn, "device_caches", None)
    if dc:
        for v in dc.values():
            v[0].clear()
    else:
        for v in fn.cache.values():
            v.clear()


def _ri(g: torch.Generator, lo: int, hi: int) -> int:
    """Uniform int in [lo, hi]."""
    return lo + int(torch.randint(0, hi - lo + 1, (1,), generator=g))


def _composition(g: torch.Generator, total: int, parts: int) -> list:
    """``total`` split into ``parts`` positive lengths, uniformly over compositions."""
    if parts == 1:
        return [total]
    cuts = sorted((torch.randperm(total - 1, generator=g)[: parts - 1] + 1).tolist())
    edges = [0] + cuts + [total]
    return [b - a for a, b in zip(edges[:-1], edges[1:])]


def _random_T(g: torch.Generator) -> int:
    u = _ri(g, 0, 3)
    if u == 0:
        return _ri(g, 1, 64)
    if u == 1:
        return _ri(g, 65, 1024)
    return _ri(g, 1025, _ENV_MAX_T)


def _feasible_G(T: int, B: int, D: int) -> list:
    """Generation-run counts G that make a valid step of T rows and B runs (B - G contexts of >= 1 row)."""
    out = []
    for G in range(0, B + 1):
        rest = T - G * (1 + D)
        if rest < 0:
            continue
        if (B - G == 0 and rest == 0) or (B - G > 0 and rest >= B - G):
            out.append(G)
    return out


def _image_mask(g: torch.Generator, spans_near: list, size: int = _ENV_MAX_POS + 8) -> torch.Tensor:
    """bool over positions [0, size): False on 1..3 image spans, each starting near one of ``spans_near``."""
    m = torch.ones(size, dtype=torch.bool)
    for _ in range(_ri(g, 1, 3)):
        lo, hi = spans_near[_ri(g, 0, len(spans_near) - 1)]
        a = _ri(g, max(0, lo), max(max(0, lo), hi))
        m[a : a + _ri(g, 1, 300)] = False
    return m


def _plan_pair(
    g: torch.Generator,
    T: int,
    B: int,
    D: int,
    G: int,
    masked: bool,
    shuffle: bool,
    mask_all=False,
    mask_fn=None,
    max_pos: int = _ENV_MAX_POS,
):
    """Two consecutive steps of the same B requests, positions below ``max_pos`` (the manager
    length Lm). Step 1: B - G contexts (lengths a random composition of T - G (1 + D), start 0,
    a random cached length, or the last start that leaves 8 positions) and G generation runs of
    1 + D rows at random positions (one in eight at the last start whose continuation can
    still reach Lm). Step 2 continues every request that can: a generation run restarts at
    start + 1 + accepted (a random accepted count for D = 5: the rejected tail is rewritten), a
    context either continues with a random chunk (one in eight to the end of the budget or of
    Lm) or turns into a generation run at its next position. ``mask_fn(g, s, n, size)``, when
    given, builds a masked context's position mask (``size`` = max_pos + 8 positions) instead of
    ``_image_mask``. Returns (steps, max position + 1 per request)."""
    nc = B - G
    ctx_lens = _composition(g, T - G * (1 + D), nc) if nc else []
    reqs = []
    for r in range(B):
        if r < nc:
            n = ctx_lens[r]
            room = max_pos - 8 - n
            u = _ri(g, 0, 9)
            s = 0 if (room < 1 or u < 3) else (room if u == 9 else _ri(g, 1, room))
            reqs.append(("ctx", s, n))
        else:
            top = max_pos - 2 * (1 + D)
            reqs.append(("gen", top if _ri(g, 0, 7) == 0 else _ri(g, 1, top - 1), 1 + D))
    masks = {}
    if masked:
        for r, (kind, s, n) in enumerate(reqs):
            if kind == "ctx" and (mask_all or _ri(g, 0, 1)):
                if mask_fn is not None:
                    masks[r] = mask_fn(g, s, n, max_pos + 8)
                else:
                    masks[r] = _image_mask(
                        g, [(s - 20, s + n - 1), (s + n - 6, s + n - 1)], max_pos + 8
                    )
    step2, ctx2 = [], []
    for r, (kind, s, n) in enumerate(reqs):
        if kind == "gen":
            step2.append((r, "gen", s + 1 + (_ri(g, 0, D) if D else 0), 1 + D))
        elif s + n + 1 + D <= max_pos and _ri(g, 0, 1):
            step2.append((r, "gen", s + n, 1 + D))
        else:
            ctx2.append(r)
    budget = _ENV_MAX_T - sum(n for _, _, _, n in step2)
    for i, r in enumerate(ctx2):
        _, s, n = reqs[r]
        e = s + n
        cap = min(max_pos - e, budget - (len(ctx2) - i - 1))
        if cap < 1:  # at the end of Lm, or no rows left in the step: this request sits the step out
            continue
        L = (
            cap
            if _ri(g, 0, 7) == 0
            else (_ri(g, 1, cap) if _ri(g, 0, 1) else _ri(g, 1, min(cap, 64)))
        )
        step2.append((r, "ctx", e, L))
        budget -= L
    step1 = [(r, kind, s, n) for r, (kind, s, n) in enumerate(reqs)]

    def order(step):
        if shuffle:
            perm = torch.randperm(len(step), generator=g).tolist()
            return [step[i] for i in perm]
        return [x for x in step if x[1] == "ctx"] + [x for x in step if x[1] == "gen"]

    steps = []
    for step in (step1, step2) if step2 else (step1,):
        runs = []
        for r, kind, s, n in order(step):
            m = masks[r][s : s + n].clone() if (kind == "ctx" and r in masks) else None
            runs.append((r, _ids(n, g), s, m))
        steps.append(runs)
    ends = {}
    for runs in steps:
        for r, ids, s, _ in runs:
            ends[r] = max(ends.get(r, 0), s + ids.numel())
    return steps, ends


def _apply_writes(consts, pool, expected, runs):
    """What one call must leave in the pool: every run's positions hold their compressed ids (-1 = DEAD)."""
    P = pool.page_size
    for r, ids, s, m in runs:
        pos = torch.arange(s, s + ids.numel())
        slots = pool.pages[r][pos // P].long() * P + pos % P
        vals = consts.token_map[ids].to(torch.int32)
        if m is not None:
            vals = torch.where(m, vals, torch.full_like(vals, DEAD))
        expected.view(-1)[slots.to(DEV)] = vals.to(DEV)


def _run_checked(consts, pool, refs, expected, steps, **kw):
    """Run ``steps`` through the op; return (rows that differ from the reference, calls whose
    pool afterwards differs from ``expected`` updated with that call's writes)."""
    bad_rows, bad_pools = 0, 0
    for runs in steps:
        got = _call(consts, pool, runs, **kw)
        at = 0
        for r, ids, s, m in runs:
            want = refs[r].forward(ids, s, m)
            bad_rows += int((got[at : at + ids.numel()] != want).flatten(1).any(1).sum())
            at += ids.numel()
        _apply_writes(consts, pool, expected, runs)
        bad_pools += int(not torch.equal(pool.pool, expected))
    return bad_rows, bad_pools


def _lows(steps) -> dict:
    """The lowest position each request's calls touch: its first run's start - (N - 1), >= 0."""
    lows = {}
    for runs in steps:
        for r, _, s, _ in runs:
            lows[r] = min(lows.get(r, s), max(0, s - (MAX_NGRAM - 1)))
    return lows


def _env_case(
    consts,
    g,
    T,
    B,
    D,
    G,
    masked,
    shuffle,
    width,
    two_steps=True,
    mask_all=False,
    mask_fn=None,
    max_pos=_ENV_MAX_POS,
    **kw,
):
    """One random partition (and its continuation) on a fresh pool, positions below ``max_pos``:
    histories planted before every run that starts past 0 (DEAD every ~7th), block tables
    ``width`` columns wide (the engine's max_blocks_per_seq for the partition's L, _engine_width;
    test_block_table_width_only_sizes_indexing covers 128..132 and wider). Each request's pages
    below its lowest touched position are unmapped (-1), as in the engine's windowed pool."""
    steps, ends = _plan_pair(g, T, B, D, G, masked, shuffle, mask_all, mask_fn, max_pos)
    if not two_steps:
        steps = steps[:1]
    lows = _lows(steps)
    pool = _Pool(ends, 128, num_pages=4096, seed=_ri(g, 0, 10**6), lows=lows)
    assert pool.max_blocks <= width, (pool.max_blocks, width)
    pool.max_blocks = width
    refs = {r: _RefState(consts, ends[r] + 8, base=lows[r]) for r in ends}
    for r, _, s, _ in steps[0]:
        _plant_history(consts, pool, refs, r, s, g)
    expected = pool.pool.clone()
    bad = _run_checked(consts, pool, refs, expected, steps, **kw)
    return bad, steps


_ENV_EDGES = [  # (T, B, D, G, L, DSpark): the envelope's corners, run before the random partitions
    (1, 1, 0, 0, 16384, False),  # one fresh 1-token context
    (1, 1, 0, 1, 1048576, False),  # one decode row
    (8192, 1, 0, 0, 16384, False),  # one 8192-token chunk
    (8192, 32, 0, 0, 16384, True),  # 32 contexts filling the gates' max_num_tokens
    (8192, 32, 5, 31, 16384, True),  # 31 verify runs + one 8006-token context
    (8192, 32, 0, 31, 16384, False),  # 31 decode rows + one 8161-token context
    (32, 32, 0, 32, 16384, False),  # a full decode batch of the gates
    (192, 32, 5, 32, 16384, True),  # a full DSpark verify batch of the gates
    (33, 32, 0, 31, 16384, False),  # 31 decode rows + a 2-token context
    (16384, 1, 0, 0, 1048576, True),  # one 16384-token chunk
    (16384, 1, 0, 0, 16384, False),  # one 16384-token chunk that is the whole request
    (16384, 256, 0, 0, 1048576, False),  # 256 contexts filling max_num_tokens
    (16384, 256, 5, 255, 1048576, True),  # 255 verify runs + one 14854-token context
    (16384, 256, 0, 255, 262144, False),  # 255 decode rows + one 16129-token context
    (256, 256, 0, 256, 1048576, False),  # a full decode batch of 256
    (1536, 256, 5, 256, 1048576, True),  # a full DSpark verify batch of 256
    (257, 256, 0, 255, 32768, False),  # 255 decode rows + a 2-token context
    (1531, 256, 5, 255, 65536, True),  # 255 verify runs + a 1-token context
]
_ENV_PARTITIONS = 264  # every B in 1..256 at least once, every (L, DSpark) at least 22 times


def _env_shapes(g: torch.Generator, count: int, need_context: bool = False) -> list:
    """``count`` random (T, B, D, G, L, DSpark): B = k % 256 + 1 for the k-th (every B once in
    256), (L, DSpark) cycling over the 12 configurations, D = 5 (DSpark verify runs) or 0
    (decode rows) with DSpark, 0 without, T from _random_T (1..16384) with T >= B, G a random
    feasible generation-run count (< B when ``need_context``)."""
    shapes = []
    while len(shapes) < count:
        k = len(shapes)
        B = k % _ENV_MAX_B + 1
        L, ds = _L_CONFIGS[k % len(_L_CONFIGS)]
        T, D = _random_T(g), (5 * _ri(g, 0, 1) if ds else 0)
        if T < B:
            continue
        feas = [G for G in _feasible_G(T, B, D) if G < B or not need_context]
        if feas:
            shapes.append((T, B, D, feas[_ri(g, 0, len(feas) - 1)], L, ds))
    return shapes


def test_envelope_random_partitions():
    """>= 200 seeded random partitions of T in [1, 16384] rows into B in [1, 256] runs (the
    edges above first; then 264 random, every B in 1..256 at least once), each at a per-request
    length L in {16384, 32768, 65536, 131072, 262144, 1048576} with DSpark on (positions below
    Lm = L + 14) or off, and each followed by its continuation step: random context / generation
    mix, D in {0, 5} with DSpark, random cached starts incl. odd ones up to Lm - 1, runs ending
    at the far end Lm, random run order, random page placement, image spans DEAD through
    token_mask on half of the partitions (spans inside chunks and across the step boundary).
    Every call (both steps) equals the reference, and after every call the whole pool equals
    the planted histories plus exactly the writes so far. Block tables are the engine's width
    for the partition's L (_engine_width: 132 .. 8196 columns); pages below a request's window
    are unmapped."""
    consts = _Consts(_synthetic_token_map())
    g = torch.Generator().manual_seed(23_2023)
    shapes = []
    for T, B, D, G, L, ds in _ENV_EDGES:
        assert G in _feasible_G(T, B, D), (T, B, D, G)
        shapes.append((T, B, D, G, L, ds))
    shapes += _env_shapes(g, _ENV_PARTITIONS)
    bad_rows = bad_pools = calls = rows = masked_calls = 0
    seen_T2, seen_B, per_cfg, far = [], set(), {}, {}
    for k, (T, B, D, G, L, ds) in enumerate(shapes):
        masked, shuffle = bool(k % 2), bool((k // 2) % 2)
        lm, width = _lm(L, ds), _engine_width(L, ds)
        (br, bp), steps = _env_case(consts, g, T, B, D, G, masked, shuffle, width, max_pos=lm)
        bad_rows, bad_pools = bad_rows + br, bad_pools + bp
        for runs in steps:
            n = sum(ids.numel() for _, ids, _, _ in runs)
            assert 1 <= n <= _ENV_MAX_T and 1 <= len(runs) <= _ENV_MAX_B, (n, len(runs))
            last = max(s + ids.numel() for _, ids, s, _ in runs)
            assert last <= lm, (last, lm)
            far[(L, ds)] = far.get((L, ds), 0) + sum(
                int(s + ids.numel() == lm) for _, ids, s, _ in runs
            )
            calls, rows = calls + 1, rows + n
            masked_calls += int(any(m is not None and not bool(m.all()) for *_, m in runs))
            seen_T2.append(n)
        seen_B.add(B)
        per_cfg[(L, ds)] = per_cfg.get((L, ds), 0) + 1
    print(
        f"\n  {len(shapes)} partitions ({calls} calls, {rows} rows; T {min(seen_T2)}..{max(seen_T2)}, "
        f"B {min(seen_B)}..{max(seen_B)} ({len(seen_B)} distinct), {masked_calls} calls with DEAD image rows): "
        f"{bad_rows} rows differ from the reference, {bad_pools} calls left the pool inexact"
    )
    print(
        "  per (L, DSpark): partitions / runs ending at Lm (the far end, block-table column "
        "(Lm - 1) // 128 of a table of _engine_width columns): "
        + "; ".join(
            f"L {L} {'on' if ds else 'off'} (Lm {_lm(L, ds)}, width {_engine_width(L, ds)}): "
            f"{per_cfg[(L, ds)]} / {far[(L, ds)]}"
            for L, ds in _L_CONFIGS
        )
    )
    assert (
        len(shapes) >= 200
        and seen_B == set(range(1, _ENV_MAX_B + 1))
        and max(seen_T2) == _ENV_MAX_T
    )
    assert all(per_cfg[c] >= 20 and far[c] > 0 for c in _L_CONFIGS), (per_cfg, far)
    assert bad_rows == 0 and bad_pools == 0


def test_envelope_harness_sees_a_rewind_skipped_and_a_wrong_mask():
    """Controls for the envelope check: the same partitions judged against a reference that
    (a) ignores token_mask or (b) never rewinds (every verify run's next start is taken as
    start + 1 + D) must fail it."""
    consts = _Consts(_synthetic_token_map())
    g = torch.Generator().manual_seed(4242)
    # (a) masks dropped from the reference only
    steps, ends = _plan_pair(g, 3000, 8, 0, 0, masked=True, shuffle=False, mask_all=True)
    assert any(m is not None for runs in steps for *_, m in runs)
    pool = _Pool(ends, 128, seed=1)
    refs = {r: _RefState(consts, _ENV_MAX_POS + 8) for r in ends}
    for r, _, s, _ in steps[0]:
        _plant_history(consts, pool, refs, r, s, g)
    unmasked = [[(r, ids, s, None) for r, ids, s, _ in runs] for runs in steps]
    bad_a = 0
    for runs, plain in zip(steps, unmasked):
        got = _call(consts, pool, runs)
        at = 0
        for r, ids, s, _ in plain:
            bad_a += int(
                (got[at : at + ids.numel()] != refs[r].forward(ids, s)).flatten(1).any(1).sum()
            )
            at += ids.numel()
    # (b) verify: the op rewinds (the next run starts after the accepted tokens); a reference
    # fed the rejected drafts as history instead differs on the rows that read them
    D, B = 5, 16
    pool = _Pool({r: 400 for r in range(B)}, 128, seed=2)
    refs = {r: _RefState(consts, 400) for r in range(B)}
    starts = {r: _ri(g, 10, 300) for r in range(B)}
    for r in range(B):
        _plant_history(consts, pool, refs, r, starts[r], g)
    step1 = [(r, _ids(1 + D, g), starts[r], None) for r in range(B)]
    _call(consts, pool, step1)
    for r, ids, s, _ in step1:
        refs[r].forward(ids, s)
    acc = {r: _ri(g, 0, D - 1) for r in range(B)}  # at least one draft rejected
    step2 = [(r, _ids(1 + D, g), starts[r] + 1 + acc[r], None) for r in range(B)]
    got = _call(consts, pool, step2)
    bad_b, at = 0, 0
    for r, ids, s, _ in step2:
        # the no-rewind reference: the rejected drafts stay in place and this run lands after them
        wrong = _RefState(consts, 400)
        wrong.cache.copy_(refs[r].cache)
        bad_b += int(
            (got[at : at + ids.numel()] != wrong.forward(ids, starts[r] + 1 + D))
            .flatten(1)
            .any(1)
            .sum()
        )
        at += ids.numel()
    print(
        f"  [control] reference without token_mask: {bad_a} rows differ; reference without rewind: "
        f"{bad_b} of {B * (1 + D)} rows differ"
    )
    assert bad_a > 0 and bad_b > 0


# ─── the multimodal target's masked call over the envelope ───
#
# The multimodal target passes ``token_mask`` = keep, a contiguous bool [T]: False on the
# image-span rows of its context chunks (DEAD), True on text rows and on every generation
# row. It passes the mask only in a step that holds at least one context chunk of an image
# request (eager; captured steps are generation-only and pass None). Such a step can be mixed
# with decode (D = 0) or DSpark verify (D = 5) runs, a chunk may lie wholly inside a span (all
# DEAD) or hold no span row (mask present, all True), and a span of one image continues
# across chunked-prefill steps. Every partition below puts a mask on every context run.

_MM_SPAN_MAX = (
    4096  # span length drawn log-uniform in [1, 4096] (MMMU spans run to ~1.1k rows per image)
)


def _log_len(g: torch.Generator, hi: int) -> int:
    """Length in [1, hi], log-uniform (short and long spans both common)."""
    if hi <= 1:
        return 1
    u = float(torch.rand((), generator=g))
    return max(1, min(hi, int(round(hi**u))))


def _mm_mask(kind: str):
    """A ``mask_fn`` for ``_plan_pair``: a position mask over [0, size) for a context
    chunk at [s, s + n). ``text``: all True (mask present, no span); ``dead``: all False (the
    whole request is a span); ``one``: one span inside the chunk; ``multi``: 3..6 spans;
    ``cross``: a span starting in the chunk's last rows that runs past the step boundary;
    ``ends``: a span from position 0 and one ending exactly at the chunk end; ``mixed``: a
    random pick of the others per request (the sweep's kind)."""

    def fn(g, s, n, size=_ENV_MAX_POS + 8):
        def span(m, a, ln):
            a = max(0, min(a, size - 1))
            m[a : a + ln] = False

        k = kind
        if k == "mixed":
            k = ("text", "dead", "one", "multi", "multi", "cross", "cross", "ends", "rand", "rand")[
                _ri(g, 0, 9)
            ]
        m = torch.ones(size, dtype=torch.bool)
        if k == "text":
            return m
        if k == "dead":
            return torch.zeros(size, dtype=torch.bool)
        if k == "one":
            ln = _log_len(g, n)
            span(m, s + _ri(g, 0, n - ln), ln)
        elif k == "multi":
            for _ in range(_ri(g, 3, 6)):
                ln = _log_len(g, max(1, n // 3))
                span(m, s + _ri(g, 0, max(0, n - ln)), ln)
        elif k == "cross":
            a = s + n - _ri(g, 1, min(n, 64))
            span(m, a, (s + n - a) + _ri(g, 1, _MM_SPAN_MAX))
        elif k == "ends":
            span(m, 0, _log_len(g, s + n))
            ln = _log_len(g, n)
            span(m, s + n - ln, ln)
        else:  # rand: 1..6 spans anywhere around the chunk, page edges and history included
            for _ in range(_ri(g, 1, 6)):
                ln = _log_len(g, _MM_SPAN_MAX)
                anchor = (
                    s - _ri(g, 0, 300),
                    s + _ri(g, 0, max(0, n - 1)),
                    s + n - _ri(g, 0, 300),
                    128 * _ri(g, 0, (s + n) // 128) - _ri(g, 0, 8),
                    0,
                )[_ri(g, 0, 4)]
                span(m, anchor, ln)
        return m

    return fn


# (name, T, B, D, G, mask kind, L, DSpark): the masked call's corners, run before the random partitions
_MM_EDGES = [
    ("one-row DEAD context", 1, 1, 0, 0, "dead", 16384, False),
    ("one context wholly inside a span", 300, 1, 0, 0, "dead", 1048576, False),
    ("32 contexts, every row DEAD", 8192, 32, 0, 0, "dead", 16384, False),
    ("DEAD 2-row context + 31 decode rows", 33, 32, 0, 31, "dead", 16384, False),
    ("mask present, all True, one row", 1, 1, 0, 0, "text", 16384, False),
    ("mask present, all True, with verify runs", 4000, 8, 5, 3, "text", 131072, True),
    ("one span", 2048, 1, 0, 0, "one", 16384, False),
    ("several spans per context", 8192, 4, 0, 0, "multi", 16384, False),
    ("spans across the step boundary", 6000, 3, 0, 0, "cross", 16384, False),
    ("spans across the step boundary, 32 contexts", 8192, 32, 0, 0, "cross", 16384, False),
    ("spans from position 0 and to the chunk end", 8192, 32, 0, 0, "ends", 16384, False),
    ("31 verify runs + a masked 8006-row context", 8192, 32, 5, 31, "mixed", 16384, True),
    ("31 decode rows + a masked 8161-row context", 8192, 32, 0, 31, "mixed", 16384, False),
    ("full DSpark verify batch + a 1-row DEAD context", 187, 32, 5, 31, "dead", 16384, True),
    ("256 contexts, every row DEAD", 16384, 256, 0, 0, "dead", 1048576, False),
    ("one 16384-row context, several spans", 16384, 1, 0, 0, "multi", 1048576, True),
    ("spans across the step boundary, 256 contexts", 16384, 256, 0, 0, "cross", 262144, False),
    (
        "spans from position 0 and to the chunk end, 256 contexts",
        16384,
        256,
        0,
        0,
        "ends",
        65536,
        False,
    ),
    ("255 verify runs + a masked 14854-row context", 16384, 256, 5, 255, "mixed", 1048576, True),
    ("255 decode rows + a masked 16129-row context", 16384, 256, 0, 255, "mixed", 32768, False),
    ("256-request verify batch + a 1-row DEAD context", 1531, 256, 5, 255, "dead", 1048576, True),
]


def _mm_shapes(g: torch.Generator, count: int) -> list:
    """``count`` random (T, B, D, G, L, DSpark) with at least one context run (G < B), every B in
    1..256 once per 256, the 12 (L, DSpark) configurations in turn."""
    return _env_shapes(g, count, need_context=True)


def _mm_stats(steps, st: dict) -> None:
    """Tally what the masked calls of one partition covered."""
    for runs in steps:
        n = sum(ids.numel() for _, ids, _, _ in runs)
        st["calls"] += 1
        st["rows"] += n
        masks = [m for *_, m in runs]
        if not any(m is not None for m in masks):
            st["unmasked_calls"] += 1
            continue
        keep = torch.cat(
            [torch.ones(ids.numel(), dtype=torch.bool) if m is None else m for _, ids, _, m in runs]
        )
        dead = int((~keep).sum())
        gen_lens = {ids.numel() for _, ids, _, m in runs if m is None}
        st["masked_calls"] += 1
        st["masked_rows"] += n
        st["dead_rows"] += dead
        st["T_max"] = max(st["T_max"], n)
        st["B_seen"].add(len(runs))
        st["all_dead_calls"] += int(dead == n)
        st["all_true_calls"] += int(dead == 0)
        st["mixed_decode_calls"] += int(1 in gen_lens and dead > 0)
        st["mixed_verify_calls"] += int(6 in gen_lens and dead > 0)
        for _, ids, s, m in runs:
            if m is None or bool(m.all()):
                continue
            st["span_runs"] += 1
            st["pos0_dead"] += int(s == 0 and not bool(m[0]))
            st["dead_at_run_end"] += int(not bool(m[-1]))
            st["dead_at_run_start_after_history"] += int(s > 0 and not bool(m[0]))


def _mm_new_stats() -> dict:
    keys = (
        "calls",
        "rows",
        "unmasked_calls",
        "masked_calls",
        "masked_rows",
        "dead_rows",
        "T_max",
        "all_dead_calls",
        "all_true_calls",
        "mixed_decode_calls",
        "mixed_verify_calls",
        "span_runs",
        "pos0_dead",
        "dead_at_run_end",
        "dead_at_run_start_after_history",
    )
    st = {k: 0 for k in keys}
    st["B_seen"] = set()
    return st


def test_multimodal_masked_calls_over_the_envelope():
    """The multimodal target's masked call (``token_mask`` = keep, bool [T]) over the
    envelope: the 21 corners above, then 264 seeded random partitions of T in [1, 16384]
    rows into B in [1, 256] runs (every B at least once), at every (L, DSpark) of _L_CONFIGS
    (positions below Lm, block tables of the engine width for L), D in {0, 5} with DSpark, at
    least one context run each, and a mask on
    EVERY context run (random kind per request: no span, all DEAD, one span, several spans,
    spans across the step boundary, from position 0 / to the chunk end, 1..6 random spans of
    1..4096 rows around the chunk, page edges and history included); generation rows True.
    Each partition's continuation step follows (verify rewinds, contexts continuing inside a
    span). Every call equals the reference, and after every call the whole pool equals the
    planted histories plus exactly the writes so far (DEAD stored as -1). The kernel's
    compiled variants are pinned: after one masked and one unmasked call per ``num_tokens``
    class at the engine's 132-column tables, the whole sweep (tables 132 .. 8196 columns wide)
    compiles nothing."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.embedding.engram_hash import (  # noqa: E501
        _engram_hash_kernel as fn,
    )

    consts = _Consts(_synthetic_token_map())
    g = torch.Generator().manual_seed(13_2026)
    # pin the variants: T == 1, T % 16 == 0, other; masked and unmasked; int32, width 132
    _jit_clear(fn)
    for T, masked in ((1, True), (16, True), (37, True), (1, False), (16, False), (37, False)):
        (br, bp), _ = _env_case(
            consts,
            g,
            T,
            1,
            0,
            0,
            masked,
            False,
            _ENGINE_WIDTH,
            two_steps=False,
            mask_all=masked,
            mask_fn=_mm_mask("one") if masked else None,
        )
        assert br == 0 and bp == 0, (T, masked, br, bp)
    pinned = _jit_keys(fn)
    assert len(pinned) == 6, f"expected one variant per (num_tokens class, mask), got {len(pinned)}"

    st = _mm_new_stats()
    bad_rows = bad_pools = 0
    for name, T, B, D, G, kind, L, ds in _MM_EDGES:
        assert G in _feasible_G(T, B, D) and G < B, (name, T, B, D, G)
        (br, bp), steps = _env_case(
            consts,
            g,
            T,
            B,
            D,
            G,
            True,
            False,
            _engine_width(L, ds),
            mask_all=True,
            mask_fn=_mm_mask(kind),
            max_pos=_lm(L, ds),
        )
        assert br == 0 and bp == 0, f"{name}: {br} rows differ, {bp} pools inexact"
        assert _call_masked(steps[0]), name
        _mm_stats(steps, st)
    edges = {k: (set(v) if isinstance(v, set) else v) for k, v in st.items()}
    shapes = _mm_shapes(g, _ENV_PARTITIONS)
    cfgs = set()
    for k, (T, B, D, G, L, ds) in enumerate(shapes):
        (br, bp), steps = _env_case(
            consts,
            g,
            T,
            B,
            D,
            G,
            True,
            bool(k % 2),
            _engine_width(L, ds),
            mask_all=True,
            mask_fn=_mm_mask("mixed"),
            max_pos=_lm(L, ds),
        )
        bad_rows, bad_pools = bad_rows + br, bad_pools + bp
        assert _call_masked(steps[0]), (k, T, B, D, G)
        _mm_stats(steps, st)
        cfgs.add((L, ds))
    compiled = _jit_keys(fn) - pinned
    print(
        f"\n  masked envelope: {len(_MM_EDGES)} corners + {len(shapes)} random partitions, every one's first call "
        f"masked; {st['calls']} calls ({st['rows']} rows), {st['masked_calls']} with a token_mask "
        f"({st['masked_rows']} rows, {st['dead_rows']} DEAD; T up to {st['T_max']}, "
        f"{len(st['B_seen'])} distinct B), {st['unmasked_calls']} continuation calls without one; "
        f"all-DEAD calls {st['all_dead_calls']}, all-True masked calls {st['all_true_calls']}, "
        f"DEAD rows beside decode runs in {st['mixed_decode_calls']} calls and beside verify runs in "
        f"{st['mixed_verify_calls']}; {st['span_runs']} runs with spans: DEAD at position 0 in "
        f"{st['pos0_dead']}, DEAD first row after history in {st['dead_at_run_start_after_history']}, "
        f"DEAD last row in {st['dead_at_run_end']}; {bad_rows} rows differ from the reference, "
        f"{bad_pools} calls left the pool inexact; {len(pinned)} pinned variants, {len(compiled)} compiled after"
    )
    assert bad_rows == 0 and bad_pools == 0
    assert not compiled, (
        f"the masked sweep compiled {len(compiled)} variant(s) beyond the pinned ones"
    )
    assert (
        len(shapes) >= 200
        and st["B_seen"] == set(range(1, _ENV_MAX_B + 1))
        and st["T_max"] == _ENV_MAX_T
    )
    assert cfgs == set(_L_CONFIGS)
    assert st["masked_calls"] >= len(shapes) + len(_MM_EDGES)
    # An all-DEAD call needs every context of the step to draw the all-DEAD kind, which the random
    # partitions (B up to 256) rarely do; the corners hold such calls at B 1, 32 and 256.
    assert edges["all_dead_calls"] >= 3
    for key in (
        "all_true_calls",
        "mixed_decode_calls",
        "mixed_verify_calls",
        "pos0_dead",
        "dead_at_run_end",
        "dead_at_run_start_after_history",
    ):
        assert edges[key] >= 1 and st[key] > edges[key], (key, edges[key], st[key])


def _call_masked(runs) -> bool:
    """Whether the op receives a token_mask for this step (some run carries a mask)."""
    return any(m is not None for *_, m in runs)


def test_multimodal_masked_harness_sees_a_wrong_mask():
    """Controls for the masked sweep: 24 of its partitions (same planner, fresh seed) judged
    against a reference that (a) ignores ``token_mask`` or (b) applies it one row late. Every
    partition where the wrong mask differs from the true one must fail."""
    consts = _Consts(_synthetic_token_map())
    g = torch.Generator().manual_seed(13_2027)
    fails = {"ignored": [], "one row late": []}
    for T, B, D, G, _, _ in _mm_shapes(g, 24):
        steps, ends = _plan_pair(g, T, B, D, G, True, True, True, _mm_mask("mixed"))
        seed = _ri(g, 0, 10**6)
        for ctl, wrong in (
            ("ignored", lambda m: None),
            ("one row late", lambda m: None if m is None else torch.cat([m[:1], m[:-1]])),
        ):
            # the wrong mask changes a row: a DEAD row it drops, or a row it shifts
            differs = any(
                m is not None
                and (not bool(m.all()) if wrong(m) is None else not torch.equal(wrong(m), m))
                for runs in steps
                for *_, m in runs
            )
            pool = _Pool(ends, 128, num_pages=4096, seed=seed)
            pool.max_blocks = _ENGINE_WIDTH
            refs = {r: _RefState(consts, _ENV_MAX_POS + 8) for r in ends}
            gh = torch.Generator().manual_seed(seed)
            for r, _, s, _ in steps[0]:
                _plant_history(consts, pool, refs, r, s, gh)
            bad = 0
            for runs in steps:
                got = _call(consts, pool, runs)
                at = 0
                for r, ids, s, m in runs:
                    bad += int(
                        (got[at : at + ids.numel()] != refs[r].forward(ids, s, wrong(m)))
                        .flatten(1)
                        .any(1)
                        .sum()
                    )
                    at += ids.numel()
            if differs:
                fails[ctl].append(bad)
    print(
        f"  [control] partitions where the wrong mask differs: ignored {len(fails['ignored'])}, one row late "
        f"{len(fails['one row late'])}; rows differing per partition, mask ignored: min {min(fails['ignored'])} "
        f"max {max(fails['ignored'])}; one row late: min {min(fails['one row late'])} "
        f"max {max(fails['one row late'])}"
    )
    assert len(fails["ignored"]) >= 20 and len(fails["one row late"]) >= 15
    assert all(b > 0 for b in fails["ignored"]) and all(b > 0 for b in fails["one row late"])


# ─── the kernel-variant map over the envelope; B is not a launch argument ───
#
# The launch is ``_engram_hash_kernel[(ceil(T / 64),)]`` with T as ``num_tokens``; B reaches
# the kernel only through the contents of req_idx_per_token / cu_seq_lens / start_pos /
# block_table, never as an argument. Triton specializes integer arguments (== 1, divisible by
# 16) and dtypes, and HAS_MASK is a constexpr, so the variants over the envelope are indexed
# by (num_tokens class, block-table width class, mask, index dtypes); the map below compiles
# and checks each, then pins that a different B (1..256) or T within a class compiles nothing.
# Widths: 128 (divisible by 16), the engine's 132 at L 16384, 129 (neither), and the engine's
# widths at L 32768 .. 1048576 (260, 516, 1028, 2052, 8196: 4 mod 16, neither) with positions up
# to that L's Lm; a class whose Triton specialization an earlier class already compiled must
# compile nothing.
_MAP_WIDTHS = (
    (128, 16384, False),
    (_ENGINE_WIDTH, 16384, False),
    (129, 16384, False),
    (260, 32768, True),
    (516, 65536, False),
    (1028, 131072, True),
    (2052, 262144, False),
    (8196, 1048576, True),
)


def test_kernel_variant_map_over_the_envelope():
    import inspect

    from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.embedding.engram_hash import (  # noqa: E501
        _engram_hash_kernel as fn,
    )

    arg_names = list(inspect.signature(fn.fn).parameters)
    assert "num_tokens" in arg_names and not any("batch" in a for a in arg_names), arg_names
    consts = _Consts(_synthetic_token_map())
    tm64 = consts.dev.token_map.to(torch.int64)
    i64 = {"ids": torch.int64, "meta": torch.int64, "table": torch.int64}
    g = torch.Generator().manual_seed(64)
    t_classes = (("T==1", (1,)), ("T%16==0", (16, 4096, 16384)), ("T%16!=0", (37, 1001, 16383)))
    variant_of, bad_rows, bad_pools, calls, n_classes, shared = {}, 0, 0, 0, 0, []
    _jit_clear(fn)  # the map starts from no variant, so every class compiles its own
    assert not _jit_keys(fn)
    for dname, kw in (("int32", {}), ("int64", {"dtypes": i64, "token_map": tm64})):
        for masked in (False, True):
            for width, L, ds in _MAP_WIDTHS:
                assert width in (128, 129) or width == _engine_width(L, ds)
                for tname, Ts in t_classes:
                    cls = f"{tname} width={width} mask={'yes' if masked else 'no'} {dname}"
                    first = True
                    for T in Ts:
                        for B in sorted({1, min(T, _ENV_MAX_B), _ri(g, 1, min(T, _ENV_MAX_B))}):
                            D = 0
                            feas = _feasible_G(T, B, D)
                            G = (
                                0 if masked else feas[_ri(g, 0, len(feas) - 1)]
                            )  # masked: every run a context
                            before = _jit_keys(fn)
                            (br, bp), steps = _env_case(
                                consts,
                                g,
                                T,
                                B,
                                D,
                                G,
                                masked,
                                False,
                                width,
                                two_steps=False,
                                mask_all=masked,
                                max_pos=_lm(L, ds),
                                **kw,
                            )
                            has_mask = any(m is not None for *_, m in steps[0])
                            assert has_mask == masked, (cls, T, B)
                            new = _jit_keys(fn) - before
                            bad_rows, bad_pools, calls = bad_rows + br, bad_pools + bp, calls + 1
                            if first:
                                assert len(new) <= 1, (
                                    f"{cls}: its first call compiled {len(new)} variants"
                                )
                                for key in new:
                                    variant_of[key] = f"{cls} (first at T={T} B={B})"
                                if not new:
                                    shared.append(cls)
                                first = False
                                n_classes += 1
                            else:
                                assert not new, (
                                    f"{cls}: T={T} B={B} compiled {len(new)} new variant(s)"
                                )
    # closure: random envelope partitions in the target's int32 form compile nothing further
    before = _jit_keys(fn)
    for k in range(40):
        T, D = _random_T(g), 5 * (k % 2)
        B = _ri(g, 1, min(_ENV_MAX_B, T))
        L, ds = (
            _L_CONFIGS[(2 * k + 1) % len(_L_CONFIGS)]
            if D
            else _L_CONFIGS[(2 * k) % len(_L_CONFIGS)]
        )
        feas = _feasible_G(T, B, D)
        if not feas:
            continue
        (br, bp), _ = _env_case(
            consts,
            g,
            T,
            B,
            D,
            feas[-1],
            bool(k % 3),
            True,
            _engine_width(L, ds),
            max_pos=_lm(L, ds),
        )
        bad_rows, bad_pools, calls = bad_rows + br, bad_pools + bp, calls + 1
    after = _jit_keys(fn)
    print(
        f"\n  variant map: {len(variant_of)} variant(s) compiled by this map over {calls} checked calls:"
    )
    for key, cls in sorted(variant_of.items(), key=lambda kv: kv[1]):
        print(f"    {cls}")
    print(
        f"  {len(shared)} class(es) reused a variant an earlier class compiled (same Triton specialization):"
    )
    for cls in shared:
        print(f"    {cls}")
    print(
        f"  40 random envelope partitions after the map compiled {len(after - before)} further variant(s); "
        f"{bad_rows} rows differ from the reference, {bad_pools} pools inexact"
    )
    assert after == before
    assert bad_rows == 0 and bad_pools == 0
    assert n_classes == 3 * 2 * 2 * len(_MAP_WIDTHS) and len(variant_of) + len(shared) == n_classes
    # 129 and the engine's 132 .. 8196 are neither 1 nor a multiple of 16: one specialization
    assert all(" width=128 " not in cls and f" width={_ENGINE_WIDTH} " not in cls for cls in shared)
    assert len(variant_of) == 2 * 2 * 2 * 3


# ─── every (T, B) the gates' runtime-value dumps show ───

# (T, B) of the engram_hash call in every boot-gate log and the three accuracy-preview logs,
# parsed one line per tuple with an arity check and the out-of-envelope garbles dropped (B 375).
_DUMP_TB = [
    (1, 1),
    (2, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (6, 6),
    (7, 7),
    (8, 8),
    (14, 1),
    (16, 1),
    (16, 16),
    (24, 24),
    (32, 32),
    (48, 1),
    (524, 32),
    (525, 32),
    (534, 32),
    (560, 32),
    (580, 32),
    (581, 32),
    (586, 32),
    (587, 32),
    (593, 32),
    (600, 32),
    (602, 32),
    (604, 32),
    (609, 32),
    (610, 32),
    (612, 32),
    (617, 32),
    (621, 32),
    (624, 32),
    (625, 32),
    (626, 32),
    (628, 32),
    (629, 32),
    (630, 32),
    (632, 32),
    (634, 32),
    (637, 32),
    (640, 32),
    (643, 32),
    (645, 32),
    (647, 32),
    (648, 32),
    (650, 32),
    (654, 32),
    (656, 32),
    (657, 32),
    (658, 32),
    (660, 32),
    (661, 32),
    (662, 32),
    (663, 32),
    (665, 32),
    (666, 32),
    (668, 32),
    (671, 32),
    (672, 32),
    (673, 32),
    (674, 32),
    (675, 32),
    (677, 32),
    (680, 32),
    (681, 32),
    (682, 32),
    (683, 32),
    (684, 32),
    (685, 32),
    (687, 32),
    (691, 32),
    (692, 32),
    (693, 32),
    (694, 32),
    (696, 32),
    (697, 32),
    (698, 32),
    (699, 32),
    (701, 32),
    (703, 32),
    (705, 32),
    (706, 32),
    (707, 32),
    (710, 32),
    (711, 32),
    (712, 32),
    (713, 32),
    (714, 32),
    (715, 32),
    (716, 32),
    (718, 32),
    (719, 32),
    (720, 32),
    (721, 32),
    (722, 32),
    (723, 32),
    (724, 32),
    (725, 32),
    (727, 32),
    (728, 32),
    (729, 32),
    (730, 32),
    (731, 32),
    (732, 32),
    (733, 32),
    (734, 32),
    (735, 32),
    (737, 32),
    (739, 32),
    (741, 32),
    (742, 32),
    (743, 32),
    (744, 32),
    (745, 32),
    (746, 32),
    (749, 32),
    (750, 32),
    (751, 32),
    (752, 32),
    (753, 32),
    (754, 32),
    (755, 32),
    (756, 32),
    (757, 32),
    (758, 32),
    (759, 32),
    (760, 32),
    (761, 32),
    (763, 32),
    (764, 32),
    (765, 32),
    (766, 32),
    (767, 32),
    (768, 32),
    (769, 32),
    (770, 32),
    (771, 32),
    (772, 32),
    (773, 32),
    (774, 32),
    (776, 32),
    (777, 32),
    (778, 32),
    (779, 32),
    (780, 32),
    (782, 32),
    (783, 32),
    (784, 32),
    (785, 32),
    (786, 32),
    (787, 32),
    (788, 32),
    (789, 32),
    (790, 32),
    (791, 32),
    (792, 32),
    (793, 32),
    (794, 32),
    (795, 32),
    (796, 32),
    (797, 32),
    (798, 32),
    (799, 32),
    (800, 32),
    (801, 32),
    (802, 32),
    (803, 32),
    (804, 32),
    (805, 32),
    (806, 32),
    (807, 32),
    (808, 32),
    (809, 32),
    (810, 32),
    (811, 32),
    (812, 32),
    (813, 32),
    (814, 32),
    (815, 32),
    (816, 32),
    (817, 32),
    (818, 32),
    (819, 32),
    (821, 32),
    (822, 32),
    (823, 32),
    (824, 32),
    (825, 32),
    (826, 32),
    (827, 32),
    (828, 32),
    (829, 32),
    (830, 32),
    (831, 32),
    (832, 32),
    (833, 32),
    (834, 32),
    (835, 32),
    (836, 32),
    (837, 32),
    (838, 32),
    (839, 32),
    (840, 32),
    (841, 32),
    (842, 32),
    (843, 32),
    (844, 32),
    (845, 32),
    (846, 32),
    (847, 32),
    (848, 32),
    (849, 32),
    (850, 32),
    (851, 32),
    (853, 32),
    (854, 32),
    (855, 32),
    (856, 32),
    (857, 32),
    (858, 32),
    (859, 32),
    (860, 32),
    (861, 32),
    (862, 32),
    (863, 32),
    (864, 32),
    (865, 32),
    (866, 32),
    (867, 32),
    (868, 32),
    (869, 32),
    (870, 32),
    (871, 32),
    (872, 32),
    (874, 32),
    (875, 32),
    (876, 32),
    (877, 32),
    (878, 32),
    (879, 32),
    (880, 32),
    (881, 32),
    (882, 32),
    (883, 32),
    (884, 32),
    (885, 32),
    (886, 32),
    (887, 32),
    (888, 1),
    (888, 32),
    (889, 32),
    (890, 1),
    (890, 32),
    (891, 32),
    (892, 32),
    (893, 32),
    (894, 32),
    (895, 32),
    (897, 32),
    (898, 32),
    (899, 32),
    (900, 32),
    (901, 32),
    (902, 32),
    (903, 32),
    (904, 32),
    (905, 32),
    (906, 32),
    (907, 32),
    (908, 32),
    (909, 32),
    (910, 32),
    (911, 32),
    (912, 32),
    (913, 32),
    (914, 32),
    (915, 32),
    (916, 32),
    (917, 32),
    (918, 32),
    (919, 32),
    (920, 32),
    (921, 32),
    (922, 32),
    (923, 32),
    (924, 32),
    (925, 32),
    (926, 32),
    (927, 32),
    (928, 32),
    (929, 32),
    (930, 32),
    (931, 31),
    (931, 32),
    (932, 32),
    (933, 32),
    (934, 32),
    (935, 32),
    (937, 32),
    (938, 32),
    (939, 32),
    (940, 32),
    (941, 32),
    (942, 32),
    (943, 32),
    (944, 32),
    (945, 32),
    (946, 32),
    (947, 32),
    (948, 32),
    (949, 32),
    (950, 32),
    (951, 32),
    (952, 32),
    (953, 32),
    (954, 32),
    (955, 32),
    (956, 32),
    (957, 32),
    (958, 32),
    (959, 32),
    (960, 32),
    (961, 32),
    (962, 32),
    (963, 32),
    (964, 32),
    (965, 32),
    (966, 32),
    (967, 32),
    (968, 32),
    (969, 32),
    (970, 32),
    (971, 32),
    (972, 32),
    (973, 32),
    (974, 32),
    (975, 32),
    (976, 32),
    (977, 32),
    (978, 32),
    (980, 32),
    (981, 32),
    (982, 32),
    (983, 32),
    (984, 32),
    (985, 32),
    (986, 32),
    (987, 32),
    (988, 32),
    (989, 32),
    (990, 32),
    (991, 32),
    (992, 32),
    (993, 32),
    (994, 32),
    (995, 32),
    (996, 32),
    (997, 32),
    (998, 32),
    (999, 32),
    (1000, 32),
    (1001, 32),
    (1003, 32),
    (1004, 32),
    (1005, 32),
    (1006, 32),
    (1007, 32),
    (1008, 32),
    (1010, 32),
    (1011, 32),
    (1013, 32),
    (1014, 32),
    (1015, 32),
    (1017, 32),
    (1018, 3),
    (1018, 32),
    (1019, 32),
    (1020, 32),
    (1021, 32),
    (1022, 32),
    (1023, 32),
    (1024, 32),
    (1025, 32),
    (1027, 32),
    (1028, 32),
    (1029, 32),
    (1030, 32),
    (1031, 32),
    (1033, 32),
    (1034, 32),
    (1035, 32),
    (1036, 32),
    (1038, 32),
    (1039, 32),
    (1042, 32),
    (1043, 32),
    (1044, 32),
    (1045, 32),
    (1046, 32),
    (1047, 32),
    (1049, 32),
    (1050, 32),
    (1051, 32),
    (1052, 32),
    (1053, 32),
    (1054, 32),
    (1056, 32),
    (1057, 32),
    (1059, 32),
    (1062, 32),
    (1063, 32),
    (1065, 32),
    (1067, 32),
    (1069, 32),
    (1070, 32),
    (1071, 32),
    (1073, 32),
    (1074, 32),
    (1076, 32),
    (1077, 32),
    (1079, 32),
    (1081, 32),
    (1086, 32),
    (1088, 32),
    (1089, 32),
    (1093, 32),
    (1095, 32),
    (1096, 32),
    (1097, 32),
    (1098, 32),
    (1099, 32),
    (1101, 32),
    (1103, 32),
    (1107, 32),
    (1110, 32),
    (1111, 32),
    (1112, 32),
    (1113, 32),
    (1115, 32),
    (1116, 32),
    (1117, 32),
    (1118, 32),
    (1120, 32),
    (1121, 32),
    (1122, 32),
    (1123, 32),
    (1124, 32),
    (1125, 32),
    (1126, 32),
    (1129, 32),
    (1130, 32),
    (1131, 32),
    (1132, 32),
    (1135, 32),
    (1137, 32),
    (1140, 32),
    (1142, 32),
    (1146, 32),
    (1149, 32),
    (1150, 32),
    (1151, 32),
    (1153, 32),
    (1156, 32),
    (1157, 32),
    (1159, 32),
    (1160, 32),
    (1161, 32),
    (1166, 32),
    (1167, 32),
    (1168, 32),
    (1170, 32),
    (1171, 32),
    (1172, 32),
    (1173, 32),
    (1174, 32),
    (1176, 32),
    (1177, 32),
    (1180, 32),
    (1181, 32),
    (1185, 32),
    (1187, 1),
    (1188, 32),
    (1191, 32),
    (1201, 32),
    (1202, 32),
    (1203, 32),
    (1208, 32),
    (1209, 32),
    (1213, 32),
    (1216, 32),
    (1229, 32),
    (1231, 32),
    (1232, 32),
    (1235, 32),
    (1242, 32),
    (1248, 32),
    (1257, 32),
    (1267, 32),
    (1268, 32),
    (1271, 32),
    (1292, 32),
    (1304, 32),
    (1305, 32),
    (1313, 32),
    (1329, 32),
    (1332, 32),
    (1343, 32),
    (1363, 32),
    (1374, 32),
    (1375, 32),
    (1385, 32),
    (1404, 32),
    (1418, 32),
    (1421, 32),
    (1425, 32),
    (1427, 32),
    (1433, 32),
    (1437, 32),
    (1438, 32),
    (1440, 32),
    (1449, 32),
    (1453, 32),
    (1454, 32),
    (1462, 32),
    (1463, 32),
    (1469, 32),
    (1478, 32),
    (1492, 32),
    (1494, 32),
    (1496, 32),
    (1497, 32),
    (1502, 32),
    (1503, 32),
    (1507, 32),
    (1511, 32),
    (1512, 32),
    (1514, 32),
    (1524, 32),
    (1526, 32),
    (1527, 32),
    (1528, 32),
    (1532, 32),
    (1535, 32),
    (1537, 32),
    (1538, 32),
    (1540, 32),
    (1541, 32),
    (1542, 32),
    (1543, 32),
    (1544, 32),
    (1548, 32),
    (1549, 32),
    (1550, 32),
    (1553, 32),
    (1554, 32),
    (1555, 32),
    (1558, 32),
    (1563, 32),
    (1565, 32),
    (1567, 32),
    (1568, 32),
    (1569, 32),
    (1572, 32),
    (1573, 32),
    (1576, 32),
    (1577, 32),
    (1580, 32),
    (1581, 32),
    (1584, 32),
    (1591, 32),
    (1592, 32),
    (1596, 32),
    (1597, 32),
    (1598, 32),
    (1599, 32),
    (1600, 32),
    (1601, 32),
    (1604, 32),
    (1605, 32),
    (1609, 32),
    (1610, 32),
    (1613, 32),
    (1614, 32),
    (1615, 32),
    (1621, 32),
    (1623, 32),
    (1624, 32),
    (1625, 32),
    (1626, 32),
    (1631, 32),
    (1635, 32),
    (1636, 32),
    (1638, 32),
    (1640, 32),
    (1644, 32),
    (1645, 32),
    (1646, 32),
    (1650, 32),
    (1651, 32),
    (1654, 32),
    (1655, 32),
    (1656, 32),
    (1657, 32),
    (1659, 32),
    (1660, 32),
    (1666, 32),
    (1667, 32),
    (1669, 32),
    (1670, 32),
    (1671, 32),
    (1672, 32),
    (1674, 32),
    (1676, 32),
    (1677, 32),
    (1679, 32),
    (1684, 32),
    (1685, 32),
    (1686, 32),
    (1691, 32),
    (1692, 32),
    (1693, 32),
    (1694, 32),
    (1695, 32),
    (1696, 32),
    (1697, 32),
    (1698, 32),
    (1699, 32),
    (1700, 32),
    (1702, 32),
    (1705, 32),
    (1706, 32),
    (1708, 32),
    (1710, 32),
    (1712, 32),
    (1713, 32),
    (1714, 32),
    (1715, 32),
    (1716, 32),
    (1717, 32),
    (1718, 32),
    (1719, 32),
    (1721, 32),
    (1723, 32),
    (1724, 32),
    (1725, 32),
    (1726, 32),
    (1729, 32),
    (1730, 32),
    (1731, 32),
    (1732, 32),
    (1733, 32),
    (1734, 32),
    (1735, 32),
    (1736, 32),
    (1737, 32),
    (1739, 32),
    (1740, 32),
    (1742, 32),
    (1743, 32),
    (1744, 32),
    (1746, 32),
    (1749, 32),
    (1750, 32),
    (1753, 32),
    (1754, 32),
    (1755, 32),
    (1756, 32),
    (1758, 32),
    (1761, 32),
    (1764, 32),
    (1765, 32),
    (1767, 32),
    (1768, 32),
    (1770, 32),
    (1771, 32),
    (1772, 32),
    (1773, 32),
    (1774, 32),
    (1777, 32),
    (1778, 32),
    (1779, 32),
    (1780, 32),
    (1782, 32),
    (1783, 32),
    (1784, 32),
    (1785, 32),
    (1786, 32),
    (1787, 32),
    (1788, 32),
    (1789, 32),
    (1790, 32),
    (1794, 32),
    (1796, 32),
    (1799, 32),
    (1800, 32),
    (1801, 32),
    (1802, 32),
    (1803, 32),
    (1805, 32),
    (1806, 32),
    (1810, 32),
    (1811, 32),
    (1812, 32),
    (1813, 32),
    (1815, 32),
    (1820, 32),
    (1821, 32),
    (1823, 32),
    (1824, 32),
    (1827, 32),
    (1829, 32),
    (1831, 32),
    (1832, 32),
    (1833, 32),
    (1834, 32),
    (1835, 32),
    (1838, 32),
    (1840, 32),
    (1842, 32),
    (1844, 32),
    (1847, 32),
    (1848, 32),
    (1849, 32),
    (1850, 32),
    (1851, 32),
    (1853, 32),
    (1854, 32),
    (1855, 32),
    (1856, 32),
    (1858, 32),
    (1862, 32),
    (1866, 32),
    (1867, 32),
    (1868, 32),
    (1873, 32),
    (1874, 32),
    (1880, 32),
    (1882, 32),
    (1885, 32),
    (1886, 32),
    (1887, 32),
    (1889, 32),
    (1891, 32),
    (1893, 32),
    (1894, 32),
    (1898, 32),
    (1899, 32),
    (1901, 32),
    (1903, 32),
    (1905, 32),
    (1906, 32),
    (1908, 32),
    (1912, 32),
    (1915, 32),
    (1917, 32),
    (1921, 32),
    (1922, 32),
    (1925, 32),
    (1931, 32),
    (1932, 32),
    (1935, 3),
    (1940, 32),
    (1941, 32),
    (1942, 32),
    (1943, 32),
    (1944, 32),
    (1950, 32),
    (1951, 32),
    (1954, 32),
    (1955, 32),
    (1956, 32),
    (1961, 32),
    (1965, 32),
    (1967, 32),
    (1969, 32),
    (1977, 32),
    (1982, 32),
    (1983, 32),
    (1984, 32),
    (1986, 32),
    (1994, 32),
    (1998, 32),
    (1999, 32),
    (2002, 32),
    (2003, 32),
    (2005, 32),
    (2006, 32),
    (2007, 32),
    (2009, 32),
    (2010, 32),
    (2012, 32),
    (2014, 32),
    (2017, 32),
    (2018, 32),
    (2023, 32),
    (2027, 32),
    (2028, 32),
    (2034, 32),
    (2037, 32),
    (2038, 32),
    (2039, 32),
    (2043, 32),
    (2046, 32),
    (2047, 32),
    (2048, 1),
    (2049, 32),
    (2050, 32),
    (2055, 32),
    (2056, 32),
    (2064, 32),
    (2075, 32),
    (2081, 32),
    (2082, 32),
    (2084, 32),
    (2102, 32),
    (2103, 32),
    (2107, 32),
    (2112, 32),
    (2114, 32),
    (2122, 32),
    (2124, 32),
    (2135, 32),
    (2137, 32),
    (2147, 32),
    (2151, 32),
    (2165, 32),
    (2170, 32),
    (2176, 32),
    (2184, 32),
    (2190, 32),
    (2193, 32),
    (2194, 32),
    (2195, 32),
    (2209, 32),
    (2223, 32),
    (2242, 32),
    (2258, 32),
    (2261, 32),
    (2290, 32),
    (2342, 32),
    (2362, 32),
    (2379, 32),
    (2396, 32),
    (2398, 32),
    (2430, 32),
    (2440, 32),
    (2443, 32),
    (2448, 32),
    (2462, 32),
    (2486, 32),
    (2494, 32),
    (2495, 32),
    (2509, 32),
    (2518, 32),
    (2527, 32),
    (2535, 32),
    (2540, 32),
    (2552, 32),
    (2570, 32),
    (2585, 32),
    (2634, 32),
    (2655, 32),
    (2658, 32),
    (2672, 32),
    (2673, 32),
    (2683, 32),
    (2716, 32),
    (2722, 32),
    (2727, 32),
    (2742, 32),
    (2744, 32),
    (2750, 32),
    (2757, 32),
    (2778, 32),
    (2798, 32),
    (2820, 32),
    (2842, 32),
    (2848, 32),
    (2880, 32),
    (2893, 32),
    (2895, 32),
    (2939, 32),
    (2997, 32),
    (3141, 32),
    (3189, 32),
    (3234, 32),
    (3278, 32),
    (3305, 32),
    (3317, 32),
    (3365, 32),
    (3629, 32),
    (8087, 28),
    (8104, 19),
    (8112, 20),
    (8118, 10),
    (8126, 17),
    (8142, 10),
    (8142, 17),
    (8143, 17),
    (8150, 8),
    (8172, 29),
    (8173, 28),
    (8177, 28),
    (8178, 8),
    (8179, 29),
    (8192, 1),
]
# (T, B) of the engram_hash call in further engine dumps (max_num_tokens 16384, max_batch_size
# 256 / 32 / 8, max_seq_len 32768 / 131072 / 1048576), the values not already listed above.
_DUMP_TB_ENVELOPE = [
    (5, 1),
    (6, 1),
    (7, 2),
    (12, 1),
    (12, 2),
    (18, 3),
    (24, 4),
    (29, 3),
    (34, 3),
    (48, 8),
    (60, 10),
    (96, 16),
    (108, 18),
    (144, 24),
    (168, 28),
    (192, 32),
    (228, 38),
    (240, 40),
    (288, 48),
    (330, 55),
    (336, 56),
    (384, 64),
    (432, 72),
    (462, 77),
    (480, 80),
    (528, 88),
    (546, 91),
    (576, 96),
    (597, 1),
    (610, 1),
    (624, 104),
    (672, 112),
    (720, 120),
    (768, 128),
    (780, 130),
    (852, 142),
    (1152, 192),
    (1536, 256),
    (8467, 50),
    (9193, 50),
    (9240, 50),
    (9364, 158),
    (9370, 159),
    (9524, 50),
    (10441, 1),
    (10444, 1),
    (10447, 2),
    (10451, 1),
    (16298, 25),
    (16308, 74),
    (16316, 145),
    (16320, 133),
    (16322, 146),
    (16327, 23),
    (16328, 23),
    (16329, 158),
    (16333, 105),
    (16340, 40),
    (16378, 24),
    (16384, 1),
]


def test_every_dumped_step_against_reference():
    """Every dumped (T, B) built as one call with exactly those values (B - G contexts,
    G decode rows; histories planted, random starts, half with image spans), against the
    reference, the pool exact afterwards. The gates' points run at L 16384 (132-column tables);
    the envelope engines' points at the (L, DSpark) configurations in turn (positions below
    Lm, tables of the engine width for L)."""
    consts = _Consts(_synthetic_token_map())
    g = torch.Generator().manual_seed(907)
    bad_rows = bad_pools = 0
    points = [(T, B, 16384, False) for T, B in _DUMP_TB]
    points += [
        (T, B) + _L_CONFIGS[k % len(_L_CONFIGS)] for k, (T, B) in enumerate(_DUMP_TB_ENVELOPE)
    ]
    assert not set(_DUMP_TB) & set(_DUMP_TB_ENVELOPE)
    for k, (T, B, L, ds) in enumerate(points):
        feas = _feasible_G(T, B, 0)
        G = feas[_ri(g, 0, len(feas) - 1)]
        (br, bp), steps = _env_case(
            consts,
            g,
            T,
            B,
            0,
            G,
            bool(k % 2),
            bool(k % 3 == 0),
            _engine_width(L, ds),
            two_steps=False,
            max_pos=_lm(L, ds),
        )
        runs = steps[0]
        assert (sum(ids.numel() for _, ids, _, _ in runs), len(runs)) == (T, B)
        bad_rows, bad_pools = bad_rows + br, bad_pools + bp
    print(
        f"\n  {len(points)} dumped (T, B) steps ({len(_DUMP_TB)} gate points, {len(_DUMP_TB_ENVELOPE)} envelope "
        f"points; B up to {max(p[1] for p in points)}, T up to {max(p[0] for p in points)}): {bad_rows} rows "
        f"differ, {bad_pools} pools inexact"
    )
    assert bad_rows == 0 and bad_pools == 0


# ─── history-pool size invariance up to the largest ENGRAM_TOKEN pool ───


def _masked_chunked_schedule(g: torch.Generator):
    masks = {r: torch.ones(n + d, dtype=torch.bool) for r, (n, _, d) in _CHUNKED.items()}
    masks[0][126:130] = False
    masks[2][250:255] = False
    masks[4][100:200] = False
    return _chunked_schedule(g, masks)


class _FixedPool(_Pool):
    """The requests' pages fixed at given page indices of whatever pool tensor is attached."""

    def __init__(self, pages: dict, page_size: int, pool: torch.Tensor):
        self.page_size = page_size
        self.pages = pages
        self.max_blocks = max(p.numel() for p in pages.values())
        self.pool = pool


def _run_outputs(consts, pool, steps, refs):
    outs = [_call(consts, pool, runs) for runs in steps]
    bad = 0
    for got, runs in zip(outs, steps):
        at = 0
        for r, ids, s, m in runs:
            bad += int(
                (got[at : at + ids.numel()] != refs[r].forward(ids, s, m)).flatten(1).any(1).sum()
            )
            at += ids.numel()
    return torch.cat(outs), bad


def _count_not(t: torch.Tensor, value: int, chunk: int = 1 << 20) -> int:
    return sum(int((t[i : i + chunk] != value).sum()) for i in range(0, t.shape[0], chunk))


def _far_end_expected(consts, pool, seqs, max_lens, steps):
    """SENTINEL everywhere except each request's positions, which hold its compressed ids
    (-1 where the schedule's mask made the token DEAD)."""
    masks = {r: torch.ones(n, dtype=torch.bool) for r, n in max_lens.items()}
    for runs in steps:
        for r, ids, s, m in runs:
            if m is not None:
                masks[r][s : s + ids.numel()] = m
    exp = torch.full_like(pool.pool, SENTINEL)
    for r, n in max_lens.items():
        pos = torch.arange(n)
        slots = pool.pages[r][pos // 128].long() * 128 + pos % 128
        vals = consts.token_map[seqs[r]].to(torch.int32)
        exp.view(-1)[slots.to(DEV)] = torch.where(masks[r], vals, torch.full_like(vals, DEAD)).to(
            DEV
        )
    return exp


def _boundary_pages(max_lens: dict, lines: list, seed: int) -> dict:
    """Each request's pages drawn, shuffled, from windows around the given page indices (so
    the schedule's slots sit on both sides of each line)."""
    blocks = {r: -(-n // 128) for r, n in max_lens.items()}
    need = sum(blocks.values())
    half = -(-need // (2 * len(lines))) + 1
    cand = torch.cat([torch.arange(p - half, p + half) for p in lines])
    cand = cand[torch.randperm(cand.numel(), generator=torch.Generator().manual_seed(seed))]
    pages, at = {}, 0
    for r, nb in blocks.items():
        pages[r] = cand[at : at + nb].to(torch.int32)
        at += nb
    return pages


def test_history_pool_size_invariance_b_max():
    """The largest ENGRAM_TOKEN pool measured over the engine envelope including PP2 stages (stage
    alone), at 0.5 x device memory: the PP2 stage-0 manager, native storage, max_num_tokens 2048,
    max_batch_size 1, no DSpark (_B_MAX_CONFIG) -> [57999360, 128] int32 (29.70 GB, 7.42e9 elements),
    the live manager's own buffer.
    (1) Invariance: the masked chunked-prefill + decode schedule of six requests, pages at fixed
    random indices below 2 x their need, run on views pool[:B] for B = the smallest pool that
    holds them, 2^23 (the first page at byte 2^32), the old bound 10,027,008, the bf16 boot
    gate's 15,269,888, 2^24 + 1 (the first page past element 2^31), the native boot gate's
    26,345,472, B_max // 2, the witness's bf16 leg's 32,243,712, 2^25 + 1 (the first page past
    byte 2^34), the unsplit managers' largest 36,241,408, the PP2 stage-0 bf16 pool 48,365,568 and
    B_max, and
    on a separate allocation of exactly the smallest size: all give bit-identical hashes
    (= the reference) and bit-identical pages, and nothing past the smallest size is written.
    (2) Boundaries: the same schedule with its pages on both sides of page 2^23 (byte offset
    2^32), page 2^24 (element offset 2^31, byte 2^33) and page 2^25 (element offset 2^32, byte
    2^34): = reference, pool exact.
    (3) Far end: every page among the pool's last pages, the last page owned: = reference, pool
    exact, SENTINEL elsewhere. Controls: the far-end pages wrapped as a 32-bit byte offset
    (page mod 2^23), a signed 32-bit element offset (page mod 2^24) or an unsigned 32-bit element
    offset (page mod 2^25) would wrap them fail it."""
    mgr, token_pool = _engine_manager(*_B_MAX_CONFIG, pp_stage=_B_MAX_STAGE)
    try:
        assert tuple(token_pool.shape) == (_B_MAX, 128) and _B_MAX * 128 > 2**32
        consts = _Consts(_synthetic_token_map())
        seqs, steps = _masked_chunked_schedule(torch.Generator().manual_seed(2323))
        max_lens = {r: s.numel() for r, s in seqs.items()}
        ref_pool = _Pool(max_lens, 128, num_pages=1, seed=2323)  # pages below 2 x need + 1
        pages = ref_pool.pages
        b_small = max(int(p.max()) for p in pages.values()) + 1
        sizes = [
            ("smallest", b_small),
            ("2^23", 1 << 23),
            ("old bound", 10_027_008),
            ("bf16 boot", 15_269_888),
            ("2^24 + 1", (1 << 24) + 1),
            ("native boot", 26_345_472),
            ("B_max // 2", _B_MAX // 2),
            ("bf16 witness leg", 32_243_712),
            ("2^25 + 1", (1 << 25) + 1),
            ("unsplit B_max", _UNSPLIT_B_MAX),
            ("bf16 PP2 stage 0", 48_365_568),
            ("B_max", _B_MAX),
        ]
        assert [b for _, b in sizes] == sorted(b for _, b in sizes)
        token_pool.fill_(SENTINEL)
        results = {}
        for name, B in sizes:
            token_pool[:b_small].fill_(SENTINEL)
            pool = _FixedPool(pages, 128, token_pool[:B])
            refs = {r: _RefState(consts, n) for r, n in max_lens.items()}
            out, bad = _run_outputs(consts, pool, steps, refs)
            assert bad == 0, (name, bad)
            results[name] = (out, token_pool[:b_small].clone())
            assert _count_not(token_pool[b_small:], SENTINEL) == 0, name
        own = torch.full((b_small, 128), SENTINEL, dtype=torch.int32, device=DEV)
        refs = {r: _RefState(consts, n) for r, n in max_lens.items()}
        out, bad = _run_outputs(consts, _FixedPool(pages, 128, own), steps, refs)
        assert bad == 0
        results["own allocation"] = (out, own)
        base_out, base_pool = results["smallest"]
        for name, (o, p) in results.items():
            assert torch.equal(o, base_out) and torch.equal(p, base_pool), name
        print(
            f"\n  invariance: pages < {b_small}; B = {', '.join(str(b) for _, b in sizes)} (views of the manager's "
            f"[{_B_MAX}, 128] buffer: PP2 stage-0 manager, native storage, max_num_tokens 2048, batch 1, no DSpark) "
            f"and an own "
            f"[{b_small}, 128] "
            f"allocation: {len(steps)} calls, hashes and pages bit-identical, = reference; nothing written past "
            f"page {b_small - 1}"
        )

        lines = [1 << 23, 1 << 24, 1 << 25]
        assert _B_MAX > lines[-1]
        bnd = _FixedPool(_boundary_pages(max_lens, lines, 2323), 128, token_pool)
        bnd.pool.fill_(SENTINEL)
        refs = {r: _RefState(consts, n) for r, n in max_lens.items()}
        _, bad = _run_outputs(consts, bnd, steps, refs)
        assert bad == 0
        exp = _far_end_expected(consts, bnd, seqs, max_lens, steps)
        assert torch.equal(bnd.pool, exp)
        del exp
        used = torch.cat(list(bnd.pages.values()))
        straddle = [(int((used < p).sum()), int((used >= p).sum())) for p in lines]
        print(
            f"  boundaries: pages on both sides of 2^23 (byte 2^32) {straddle[0]}, 2^24 (element 2^31, byte "
            f"2^33) {straddle[1]} and 2^25 (element 2^32, byte 2^34) {straddle[2]} (below, at-or-above): "
            f"= reference, pool exact"
        )
        assert all(lo > 0 and hi > 0 for lo, hi in straddle)

        far = _FarEndPool(token_pool, max_lens, seed=2323)
        assert int(max(int(p.max()) for p in far.pages.values())) == _B_MAX - 1
        refs = {r: _RefState(consts, n) for r, n in max_lens.items()}
        _, bad = _run_outputs(consts, far, steps, refs)
        assert bad == 0
        exp = _far_end_expected(consts, far, seqs, max_lens, steps)
        assert torch.equal(far.pool, exp)
        lowest = min(int(p.min()) for p in far.pages.values())
        wrap_differs = {}
        for bits in (23, 24, 25):
            far.pool.fill_(SENTINEL)
            wrapped = _FixedPool({r: p % (1 << bits) for r, p in far.pages.items()}, 128, far.pool)
            refs = {r: _RefState(consts, n) for r, n in max_lens.items()}
            _run_outputs(consts, wrapped, steps, refs)
            wrap_differs[bits] = not torch.equal(far.pool, exp)
        print(
            f"  far end: pages {lowest}..{_B_MAX - 1} (element offsets past 2^32, byte offsets past 2^34), "
            f"= reference, pool exact, rest SENTINEL; [control] pages wrapped mod 2^23 (32-bit byte offset) -> pool "
            f"check fails: {wrap_differs[23]}; mod 2^24 (signed 32-bit element offset): {wrap_differs[24]}; "
            f"mod 2^25 (unsigned 32-bit element offset): {wrap_differs[25]}"
        )
        assert lowest > (1 << 25) and wrap_differs[23] and wrap_differs[24] and wrap_differs[25]
        del exp
    finally:
        mgr.shutdown()


# ─── the block-table width only sizes indexing ───
#
# The target passes metadata.engram_token_block_table: [max_num_sequences, max_blocks_per_seq],
# row stride = width; the width is 132 at max_seq_len 16384 and grows with L to 8196 at 1,048,576
# (_engine_width). The kernel reads row b at bt_row_stride * b and column p // P, masked by
# p // P < max_blocks. Without DSpark positions stay below L, so column L / 128 - 1 is the last
# one a call addresses; with DSpark the manager's max_seq_len is Lm = L + 14 and a verify run at
# the end of a full-length request writes positions up to Lm - 1, i.e. column L / 128.

_WIDTHS = (128, 129, 130, 131, 132, 136, 260, 516, 1028, 2052, 8196)


class _WidePool(_Pool):
    """_Pool whose per-step table is ``width`` columns wide, a column-prefix view of a
    [B + 3, width + pad] allocation (row stride width + pad). The columns past each request's
    own pages, the pad columns and the 3 rows past B hold OTHER requests' page ids, so a read or
    a write through any of them would corrupt a checked row or a checked page. ``corrupt`` =
    (req, col) replaces one of the request's own columns with a foreign page (the control)."""

    def __init__(self, max_lens: dict, seed: int, width: int, pad: int, corrupt=None):
        super().__init__(max_lens, 128, num_pages=4096, seed=seed)
        self.width, self.pad, self.corrupt = width, pad, corrupt
        self.junk = torch.Generator().manual_seed(1000 * width + pad)
        self.all_pages = torch.cat(list(self.pages.values()))
        self.all_pages = self.all_pages[self.all_pages >= 0]

    def table(self, req_ids, override=None) -> torch.Tensor:
        n = len(req_ids)
        full = self.all_pages[
            torch.randint(
                0, self.all_pages.numel(), (n + 3, self.width + self.pad), generator=self.junk
            )
        ]
        for i, r in enumerate(req_ids):
            own = self.pages[r]
            full[i, : own.numel()] = own
            if self.corrupt is not None and self.corrupt[0] == r:
                others = torch.cat([p for q, p in self.pages.items() if q != r])
                full[i, self.corrupt[1]] = others[0]
        view = full.to(DEV)[:, : self.width]
        assert view.stride() == (self.width + self.pad, 1)
        return view


def _width_run(consts, steps, ends, seed, width, pad, corrupt=None):
    """``steps`` on a fresh _WidePool (pages fixed by ``seed``) with histories planted from
    ``seed``: (all outputs, final pool, rows != reference, calls whose pool != expected)."""
    pool = _WidePool(ends, seed, width, pad, corrupt)
    refs = {r: _RefState(consts, n + 2) for r, n in ends.items()}
    g = torch.Generator().manual_seed(seed)
    for r, _, s, _ in steps[0]:
        _plant_history(consts, pool, refs, r, s, g)
    expected = pool.pool.clone()
    outs, bad_rows, bad_pools = [], 0, 0
    for runs in steps:
        got = _call(consts, pool, runs)
        at = 0
        for r, ids, s, m in runs:
            bad_rows += int(
                (got[at : at + ids.numel()] != refs[r].forward(ids, s, m)).flatten(1).any(1).sum()
            )
            at += ids.numel()
        _apply_writes(consts, pool, expected, runs)
        bad_pools += int(not torch.equal(pool.pool, expected))
        outs.append(got)
    return torch.cat(outs), pool.pool.clone(), bad_rows, bad_pools


_DSPARK_MAX_SEQ_LEN = 16398  # the manager's max_seq_len with DSpark (max_draft_len 5, overlap on)


def _dspark_tail_steps(g: torch.Generator, L: int = 16384):
    """DSpark verify runs of 6 at the end of full-length requests of max_seq_len L, into logical
    page L / 128: from L - 2 (the last real position) to L + 3, from Lm - 6 to Lm - 1 (the
    manager's last position, Lm = L + 14), beside a decode row at L - 1; then the rewound
    continuation from L."""
    lm = _lm(L, True)
    step1 = [
        (0, _ids(6, g), L - 2, None),
        (1, _ids(6, g), lm - 6, None),
        (2, _ids(1, g), L - 1, None),
    ]
    step2 = [(0, _ids(6, g), L, None), (2, _ids(6, g), L, None)]
    ends = {0: L + 6, 1: lm, 2: L + 6}
    return [step1, step2], ends


def _plain_tail_steps(g: torch.Generator, L: int):
    """Without DSpark, runs ending at max_seq_len L (logical page L / 128 - 1, the last column a
    call can address): a 200-row chunk from L - 400, a decode row at L - 2 and a 128-row chunk
    ending at L; then the chunk continued to L and the decode row at L - 1."""
    step1 = [
        (0, _ids(200, g), L - 400, None),
        (1, _ids(1, g), L - 2, None),
        (2, _ids(128, g), L - 128, None),
    ]
    step2 = [(0, _ids(200, g), L - 200, None), (1, _ids(1, g), L - 1, None)]
    ends = {0: L, 1: L, 2: L}
    return [step1, step2], ends


def _last_page_steps(g: torch.Generator):
    """Three requests whose runs end at position 16383 (logical page 127, the last column a call
    can address): a 128-token chunk from 16256, a decode row at 16383 and a 6-row verify run from
    16378; then a second step on page 127 again (the verify rewound by 2)."""
    step1 = [
        (0, _ids(128, g), 16256, None),
        (1, _ids(1, g), 16383, None),
        (2, _ids(6, g), 16378, None),
    ]
    step2 = [(2, _ids(4, g), 16380, None), (1, _ids(1, g), 16383, None)]
    ends = {0: 16384, 1: 16384, 2: 16384}
    return [step1, step2], ends


def test_block_table_width_only_sizes_indexing():
    """The same steps with block tables 128..132, 136, 260, 516, 1028, 2052 and 8196 columns wide
    (the engine's widths at L 16384 .. 1048576 among them), each as a column-prefix view whose
    row stride is the width (the engine's form) or the width + 5, with every column and row the
    call must not read holding another request's pages: the hashes and the whole pool are
    bit-identical at every width and stride, and equal the reference. The steps are the 9
    gate-envelope corners (positions below 16384), 15 random envelope partitions (D 0 / 5, image
    spans, shuffled order, each with its continuation step), three requests ending on logical
    page 127, DSpark verify runs into logical page 128 (positions up to 16397, run at widths >=
    129 only), and at every L of _L_SET the far end with and without DSpark: runs ending at L
    (last column L / 128 - 1) and DSpark verify runs ending at Lm = L + 14 (column L / 128),
    each run at every listed width above its last page (at L 1048576: 8196 only). Control: the
    same last-page steps at width 132 with the decode request's own column 127 (its history and
    its write) pointing at a foreign page fail the check."""
    consts = _Consts(_synthetic_token_map())
    g = torch.Generator().manual_seed(132)
    cases = []
    for T, B, D, G, _, _ in _ENV_EDGES[:9]:
        steps, ends = _plan_pair(
            g, T, B, D, G, masked=bool(len(cases) % 2), shuffle=bool(len(cases) % 3 == 0)
        )
        cases.append((f"edge T={T} B={B} D={D} G={G}", steps, ends))
    while len(cases) < 24:
        B = _ri(g, 1, _ENV_MAX_B)
        T, D = _random_T(g), 5 * _ri(g, 0, 1)
        feas = _feasible_G(T, B, D) if T >= B else []
        if not feas:
            continue
        G = feas[_ri(g, 0, len(feas) - 1)]
        steps, ends = _plan_pair(
            g, T, B, D, G, masked=bool(len(cases) % 2), shuffle=bool(len(cases) % 3 == 0)
        )
        cases.append((f"random T={T} B={B} D={D} G={G}", steps, ends))
    steps, ends = _last_page_steps(g)
    cases.append(("last page", steps, ends))
    steps, ends = _dspark_tail_steps(g)
    cases.append(("DSpark tail", steps, ends))
    for L in _L_SET:
        steps, ends = _plain_tail_steps(g, L)
        cases.append((f"L {L} tail", steps, ends))
        steps, ends = _dspark_tail_steps(g, L)
        cases.append((f"L {L} DSpark tail", steps, ends))
    bad_rows = bad_pools = runs_done = 0
    max_page, widths_run = {}, {}
    for k, (name, steps, ends) in enumerate(cases):
        top = max((s + ids.numel() - 1) // 128 for runs in steps for _, ids, s, _ in runs)
        max_page[name] = top
        base = None
        widths_run[name] = [w for w in _WIDTHS if w > top]
        for width in widths_run[name]:
            for pad in (0, 5):
                out, pool, br, bp = _width_run(consts, steps, ends, 7000 + k, width, pad)
                bad_rows, bad_pools, runs_done = bad_rows + br, bad_pools + bp, runs_done + 1
                if base is None:
                    base = (out, pool)
                else:
                    assert torch.equal(out, base[0]) and torch.equal(pool, base[1]), (
                        name,
                        width,
                        pad,
                    )
    k_last = [name for name, _, _ in cases].index("last page")
    steps, ends = cases[k_last][1], cases[k_last][2]
    _, _, cr, cp = _width_run(
        consts, steps, ends, 7000 + k_last, _ENGINE_WIDTH, 0, corrupt=(1, 127)
    )
    print(
        f"\n  {len(cases)} step pairs x widths {_WIDTHS} (those > the case's last logical page) x row stride "
        f"(width, width + 5) = {runs_done} runs; last logical page per case: max {max(max_page.values())} "
        f"(last page {max_page['last page']}, DSpark tail {max_page['DSpark tail']}): hashes and pools bit-identical "
        f"across widths and strides, {bad_rows} rows differ from the reference, {bad_pools} calls left the pool "
        f"inexact; [control] column 127 of a request pointing at a foreign page: {cr} rows differ, {cp} calls left "
        f"the pool inexact"
    )
    print(
        "  far end per L (last logical page, widths run): "
        + "; ".join(
            f"L {L}: {max_page[f'L {L} tail']} {widths_run[f'L {L} tail']}, DSpark {max_page[f'L {L} DSpark tail']} "
            f"{widths_run[f'L {L} DSpark tail']}"
            for L in _L_SET
        )
    )
    assert max_page["last page"] == 127 and max_page["DSpark tail"] == 128
    for L in _L_SET:
        assert (
            max_page[f"L {L} tail"] == L // 128 - 1 and max_page[f"L {L} DSpark tail"] == L // 128
        )
        assert _engine_width(L, False) in widths_run[f"L {L} tail"]
        assert _engine_width(L, True) in widths_run[f"L {L} DSpark tail"]
    assert bad_rows == 0 and bad_pools == 0
    assert cr > 0 and cp > 0


# ─── encoder replay: the history call and its empty runs ───────────────────────
#
# Under encoder SWA bounded replay the target calls the op twice in a step that holds a
# prefix hit. First the HISTORY call, over the metadata's v41_replay_history_* fields:
#   input_ids          v41_replay_history_token_ids_cuda[:hist]     int32 prompt ids
#   req_idx_per_token  v41_replay_history_req_idx_cuda[:hist]       int32 context row b
#   cu_seq_lens        v41_replay_history_cu_seq_lens_cuda[:nc + 1] int32
#   start_pos          v41_replay_history_start_pos_cuda[:nc]       int32
#   block_table        metadata.engram_token_block_table (_engine_width columns; rows 0..nc - 1 read)
#   token_mask         None
# with ONE RUN PER CONTEXT REQUEST in batch order (B = num_contexts, 1..256). A request whose
# first chunk replays from s = reused - min(128, reused) > 0 has min(3, s) rows, the prompt
# ids of positions [s - len, s), at start_pos s - len. Every other context request (no hit,
# a later chunk, or a replay from s = 0 after a hit of <= 128 tokens) has an EMPTY run:
# cu_seq_lens[b] == cu_seq_lens[b + 1], start_pos 0, no row. reused is a multiple of 2 (the
# compress ratio), so a non-empty run is 3 rows at an odd start_pos >= 1 (s >= 4) or 2 rows
# at start_pos 0 (s = 2). The call is made only when some run is non-empty; its hashes are
# discarded, its pool writes are what matters: the step's main call (all context chunks,
# then the generation runs) reads [s - 3, s) of every replay request, whose chunk starts at
# s with R = min(128, reused) replay rows and possibly nothing after them.

_REPLAY_WINDOW = 128  # replay_window: a hit of reused tokens re-forwards min(128, reused)
_HISTORY_LEN = MAX_NGRAM - 1  # engram_history_len
_REPLAY_KINDS = ("r3", "r2", "r0", "plain")


def _replay_step(
    g: torch.Generator, kinds, D: int = 0, G: int = 0, forced=None, L: int = 16384, ds: bool = False
) -> dict:
    """One encoder-replay step: len(kinds) context requests in batch order, then G generation
    runs of 1 + D rows. kinds[b]: 'r3' a replay from an even s >= 4 (3 history rows), 'r2' a
    replay from s = 2 (2 history rows at 0), 'r0' a replay from s = 0 after a hit of 2..128
    tokens (empty history run), 'plain' no replay (empty history run; the chunk continues
    from a cached c of any parity, 0 included). A replay chunk holds its R replay rows plus
    0 or more suffix rows. ``forced[b] = (start, rows)`` pins a context. Returns the history
    runs (the engine's form: every context request, the empty runs included), the main
    runs, each request's end and each context's start and R. Positions stay below the manager
    length Lm of (L, ds); the step's block tables are the engine width for L."""
    forced = forced or {}
    max_pos = _lm(L, ds)
    nc = len(kinds)
    mins, reused = [], {}
    for b, kind in enumerate(kinds):
        if kind in ("r3", "r2"):
            reused[b] = None
            mins.append(_REPLAY_WINDOW)
        elif kind == "r0":
            reused[b] = 2 * _ri(g, 1, _REPLAY_WINDOW // 2)
            mins.append(reused[b])
        else:
            mins.append(1)
    budget = _ENV_MAX_T - sum(mins) - G * (1 + D)
    assert budget >= 0 and nc + G <= _ENV_MAX_B, (kinds, D, G)
    total = _ri(g, 0, budget) if _ri(g, 0, 2) else _ri(g, 0, min(budget, 64 * nc))
    cuts = sorted(_ri(g, 0, total) for _ in range(nc - 1))
    extra = [b - a for a, b in zip([0] + cuts, cuts + [total])]
    extra = [
        0 if _ri(g, 0, 2) == 0 else e for e in extra
    ]  # a third: a chunk of exactly its minimum
    hist, main, starts, replay = [], [], [], []
    for b, kind in enumerate(kinds):
        n = mins[b] + extra[b]
        if kind == "r3":
            s = 2 * _ri(g, 2, (max_pos - n) // 2)
        elif kind == "r2":
            s = 2
        elif kind == "r0":
            s = 0
        else:
            s = 0 if _ri(g, 0, 9) < 3 else _ri(g, 1, max_pos - n)
        if b in forced:
            s, n = forced[b]
            assert n >= mins[b] and s + n <= max_pos and (kind != "r3" or (s >= 4 and s % 2 == 0))
        n_hist = min(_HISTORY_LEN, s) if kind in ("r3", "r2") else 0
        hist.append((b, _ids(n_hist, g), s - n_hist if n_hist else 0, None))
        main.append((b, _ids(n, g), s, None))
        starts.append(s)
        replay.append(mins[b] if kind != "plain" else 0)
    for j in range(G):
        main.append((nc + j, _ids(1 + D, g), _ri(g, 1, max_pos - (1 + D)), None))
    assert sum(ids.numel() for _, ids, _, _ in main) <= _ENV_MAX_T and len(main) <= _ENV_MAX_B
    ends = {r: s + ids.numel() for r, ids, s, _ in main}
    return {
        "kinds": list(kinds),
        "D": D,
        "G": G,
        "hist": hist,
        "main": main,
        "ends": ends,
        "start": starts,
        "replay": replay,
        "max_pos": max_pos,
        "width": _engine_width(L, ds),
    }


def _engine_args(pool: _Pool, runs):
    """The op's index arguments for ``runs`` as the metadata builds them (int32, batch order;
    an empty run keeps its cu_seq_lens entry, start_pos and block-table row)."""
    lens = [ids.numel() for _, ids, _, _ in runs]
    ids = torch.cat([x for _, x, _, _ in runs]).to(torch.int32)
    req = torch.repeat_interleave(torch.arange(len(runs)), torch.tensor(lens)).to(torch.int32)
    cu = torch.tensor([0] + list(np.cumsum(lens)), dtype=torch.int32)
    start = torch.tensor([s for _, _, s, _ in runs], dtype=torch.int32)
    return [ids, req, cu, start, pool.table([r for r, _, _, _ in runs])]


def _call_engine_args(consts: _Consts, pool: _Pool, args) -> torch.Tensor:
    """One op call with the index arguments ``_engine_args`` built."""
    ids, req, cu, start, table = args
    return op(
        ids.to(DEV),
        req.to(DEV),
        cu.to(DEV),
        start.to(DEV),
        pool.pool,
        table.to(DEV),
        consts.dev.token_map,
        consts.dev.multipliers,
        consts.dev.primes,
        consts.dev.offsets,
        consts.pad_id,
    ).cpu()


def _replay_case(consts: _Consts, step: dict, page_size: int, seed: int) -> dict:
    """Run one replay step the way the target does -- the history call, then the main call --
    on a fresh pool, and measure (asserting nothing):

    * the history call (A, the engine's form) against the reference, and the whole pool
      against the planted state plus exactly its writes;
    * B: the same call without its empty runs, from the same pool: rows and pool bitwise;
    * C: the empty runs given a random start_pos and another run's block-table row: rows and
      pool bitwise (an empty run's start_pos and table row are never read);
    * every empty-run request's pages unchanged by the call;
    * the main call after it against the reference (replay chunks read the history the
      history call stored), the whole pool exact;
    * [control] W: req_idx_per_token numbered over the non-empty runs only;
    * [control] the main call with the history call skipped."""
    hist, main, ends = step["hist"], step["main"], step["ends"]
    g = torch.Generator().manual_seed(seed)
    lows = {}
    for r, ids, s, _ in list(hist) + list(main):
        if ids.numel() or r >= len(hist):
            lows[r] = min(lows.get(r, s), max(0, s - _HISTORY_LEN))
    for b, kind in enumerate(
        step["kinds"]
    ):  # the replay's stale history below its start is planted too
        lows[b] = min(lows[b], max(0, step["start"][b] - _HISTORY_LEN))
    pool = _Pool(ends, page_size, num_pages=4096, seed=seed, lows=lows)
    pool.max_blocks = max(pool.max_blocks, step["width"])
    refs = {r: _RefState(consts, ends[r] + 8, base=lows[r]) for r in ends}
    for b, kind in enumerate(step["kinds"]):
        s = step["start"][b]
        if kind == "r3":
            _plant_history(consts, pool, refs, b, s - _HISTORY_LEN, g)  # what the history rows read
        if s > 0:
            # plain: the chunk's real history; replay: stale ids the history call must overwrite
            _plant_history(consts, pool, refs, b, s, g)
    for r, _, s, _ in main[len(hist) :]:
        _plant_history(consts, pool, refs, r, s, g)
    pre = pool.pool.clone()
    nonempty = [run for run in hist if run[1].numel()]
    empty = [b for b, (_, ids, _, _) in enumerate(hist) if not ids.numel()]

    def run(args):
        pool.pool.copy_(pre)
        out = _call_engine_args(consts, pool, args)
        return out, pool.pool.clone()

    args_a = _engine_args(pool, hist)
    out_a, pool_a = run(args_a)
    at, bad_hist = 0, 0
    for b, ids, s, _ in hist:
        want = refs[b].forward(ids, s)
        bad_hist += int((out_a[at : at + ids.numel()] != want).flatten(1).any(1).sum())
        at += ids.numel()
    expected = pre.clone()
    _apply_writes(consts, pool, expected, hist)
    pool_ok = torch.equal(pool_a, expected)
    out_b, pool_b = run(_engine_args(pool, nonempty))
    same_b = torch.equal(out_a, out_b) and torch.equal(pool_a, pool_b)
    args_c = [x.clone() for x in args_a]
    donor = nonempty[0][0]
    for b in empty:
        args_c[3][b] = _ri(g, 0, step["max_pos"] - 1)
        args_c[4][b] = args_a[4][donor]
    out_c, pool_c = run(args_c)
    same_c = torch.equal(out_a, out_c) and torch.equal(pool_a, pool_c)
    own = {b: pool.pages[b][pool.pages[b] >= 0].long().to(DEV) for b in empty}
    untouched = all(torch.equal(pool_a[own[b]], pre[own[b]]) for b in empty)
    args_w = list(args_a)
    lens = [ids.numel() for _, ids, _, _ in nonempty]
    args_w[1] = torch.repeat_interleave(torch.arange(len(nonempty)), torch.tensor(lens)).to(
        torch.int32
    )
    out_w, pool_w = run(args_w)
    caught_w = not (torch.equal(out_w, out_a) and torch.equal(pool_w, pool_a))
    empty_before = any(
        b < max(r for r, (_, ids, _, _) in enumerate(hist) if ids.numel()) for b in empty
    )
    # the main call on the history call's pool
    pool.pool.copy_(pool_a)
    got = _call(consts, pool, main)
    wants = [refs[r].forward(ids, s, m) for r, ids, s, m in main]
    want = torch.cat(wants)
    bad_main = int((got != want).flatten(1).any(1).sum())
    _apply_writes(consts, pool, pool_a, main)
    main_pool_ok = torch.equal(pool.pool, pool_a)
    # control: no history call
    pool.pool.copy_(pre)
    bad_skip = int((_call(consts, pool, main) != want).flatten(1).any(1).sum())
    return {
        "bad_hist": bad_hist,
        "pool_ok": pool_ok,
        "same_b": same_b,
        "same_c": same_c,
        "untouched": untouched,
        "caught_w": caught_w,
        "empty_before": empty_before,
        "bad_main": bad_main,
        "main_pool_ok": main_pool_ok,
        "bad_skip": bad_skip,
        "hist_rows": sum(ids.numel() for _, ids, _, _ in hist),
        "main_rows": int(want.shape[0]),
        "empty": empty,
    }


# (kinds in batch order, D, G, forced {b: (start, rows)}[, L, DSpark]): the corners, each at page 128
# and 256; L 16384 without DSpark unless given
_REPLAY_LM_1M = _lm(1048576, True)
_REPLAY_CORNERS = [
    (
        ["r3", "plain"],
        0,
        0,
        {0: (3970, 129), 1: (1000, 64)},
    ),  # the witness's (3, 2): replay first, empty last
    (
        ["plain", "r3"],
        0,
        0,
        {1: (3970, 128)},
    ),  # empty first; the replay chunk is its 128 replay rows only
    (["r3", "plain", "r3"], 0, 0, {}),  # empty in the middle
    (["plain", "plain", "r3", "r0", "plain"], 0, 0, {}),  # several empties around one replay
    (["r2", "plain"], 0, 0, {}),  # a 2-row run at start_pos 0 beside an empty
    (["plain", "r2"], 0, 0, {}),
    (["r0", "r2", "r0"], 0, 0, {}),  # the only non-empty run is 2 rows
    (["r3"] + ["plain", "r0"] * 15 + ["plain"], 0, 0, {}),  # B 32: 31 empties after the replay
    (["plain", "r0"] * 15 + ["plain", "r3"], 0, 0, {}),  # B 32: 31 empties before it
    (["plain"] * 16 + ["r3"] + ["r0"] * 15, 0, 0, {}),  # B 32: the replay in the middle
    (["r3"] * 32, 0, 0, {}),  # B 32, no empty run: 96 history rows
    (
        ["r3"],
        0,
        0,
        {0: (_ENV_MAX_POS - 128, 128)},
    ),  # B 1 at the last replay start (start_pos 16253)
    (["r3"], 0, 0, {0: (4, 128)}),  # the first replay start with 3 history rows (start_pos 1)
    (["r3", "plain"], 5, 6, {}),  # beside six DSpark verify runs in the main call
    (["plain", "r2", "r0", "r3"], 0, 10, {}),  # beside ten decode rows
    (
        ["plain"] * 8 + ["r3"] * 2 + ["plain"] * 8,
        5,
        14,
        {},
    ),  # 16 empties, 2 replays, 14 verify runs: B 32
    (["r3"] + ["plain"] * 255, 0, 0, {}, 1048576, False),  # B 256: 255 empties after the replay
    (["plain"] * 255 + ["r3"], 0, 0, {}, 262144, True),  # B 256: 255 empties before it
    (
        ["plain"] * 128 + ["r3"] + ["r0"] * 20 + ["plain"] * 107,
        0,
        0,
        {},
        65536,
        False,
    ),  # B 256: in the middle
    (
        ["r3"] * 127,
        0,
        0,
        {},
        1048576,
        True,
    ),  # B 127, no empty run: 381 history rows, 16256 replay rows
    (
        ["r3"],
        0,
        0,
        {0: (_REPLAY_LM_1M - 128, 128)},
        1048576,
        True,
    ),  # the last replay start at Lm 1,048,590
    (
        ["r3"],
        0,
        0,
        {0: (1048576 - 128, 128)},
        1048576,
        False,
    ),  # the last replay start at L 1,048,576
    (["r3", "plain"], 5, 250, {}, 1048576, True),  # beside 250 DSpark verify runs: B 252
    (
        ["plain", "r2", "r0", "r3"] * 4,
        0,
        240,
        {},
        131072,
        False,
    ),  # 16 contexts beside 240 decode rows: B 256
]
_REPLAY_PARTITIONS = 256


def _replay_cases(g: torch.Generator) -> list:
    """(step, page size) for the corners (both page sizes) and then the seeded random
    partitions (page 128, the engine's): every B 1..32 several times, kinds drawn per
    context request (r3 0.35, r2 0.1, r0 0.15, plain 0.4; at least one r3 / r2), D in {0, 5},
    generation runs on half of them."""
    cases = []
    for corner in _REPLAY_CORNERS:
        kinds, D, G, forced = corner[:4]
        L, ds = corner[4:] if len(corner) > 4 else (16384, D > 0)
        for ps in (128, 256):
            cases.append((_replay_step(g, kinds, D, G, forced, L, ds), ps))
    k = 0
    while k < _REPLAY_PARTITIONS:
        nc = k % _ENV_MAX_B + 1
        L, ds = _L_CONFIGS[k % len(_L_CONFIGS)]
        k += 1
        kinds = []
        for _ in range(nc):
            u = _ri(g, 0, 19)
            kinds.append("r3" if u < 7 else "r2" if u < 9 else "r0" if u < 12 else "plain")
        if not any(x in ("r3", "r2") for x in kinds):
            kinds[_ri(g, 0, nc - 1)] = "r3"
        # the step's rows fit max_num_tokens: a replay chunk holds its 128 (r3 / r2) or up to 128 (r0)
        # replay rows, so past about 120 replays a random one turns into a plain chunk
        while _REPLAY_WINDOW * sum(x != "plain" for x in kinds) + nc > _ENV_MAX_T:
            idx = [i for i, x in enumerate(kinds) if x != "plain"]
            kinds[idx[_ri(g, 0, len(idx) - 1)]] = "plain"
        if not any(x in ("r3", "r2") for x in kinds):
            kinds[kinds.index("plain")] = "r3"
        D = 5 * _ri(g, 0, 1) if ds else 0
        room = _ENV_MAX_T - _REPLAY_WINDOW * sum(x != "plain" for x in kinds) - nc
        G = min(_ri(g, 0, _ENV_MAX_B - nc), room // (1 + D)) if _ri(g, 0, 1) else 0
        cases.append((_replay_step(g, kinds, D, G, None, L, ds), 128))
    return cases


def test_replay_history_call_with_empty_runs():
    """The encoder-replay target's history call, empty runs included, over the envelope:
    the 24 corners at page 128 and 256, then 256 seeded random partitions at page 128
    (B 1..256, every value; the 12 (L, DSpark) configurations in turn, positions below Lm up to
    1,048,590; empty runs anywhere, several per call, up to 255; 2- and 3-row runs; replay
    starts up to Lm - 128; the main call's chunks of exactly their replay rows or longer, plain
    chunks from any cached start, decode or DSpark verify runs after them; tables of the engine
    width for L, 132 .. 8196 columns). Every history call equals the reference with the whole pool exact,
    is bitwise equal (rows and pool) to the same call without its empty runs, and to the
    call with the empty runs' start_pos and block-table rows changed; no empty-run request's
    page changes; the main call after it equals the reference with the pool exact. Controls:
    req_idx_per_token numbered over the non-empty runs fails exactly the calls where an
    empty run precedes a non-empty one; skipping the history call fails the main call of
    every step. No Triton variant beyond one per num_tokens class is compiled."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.embedding.engram_hash import (  # noqa: E501
        _engram_hash_kernel as fn,
    )

    consts = _Consts(_synthetic_token_map())
    g = torch.Generator().manual_seed(19_066)
    warm = _Pool({0: 64}, 128, seed=3)
    warm.max_blocks = _ENGINE_WIDTH
    for T in (1, 16, 37):  # one call per num_tokens class, in the target's int32 form
        _call(consts, warm, [(0, _ids(T, g), 0, None)])
    variants = _jit_keys(fn)
    cases = _replay_cases(g)
    tot = {
        "bad_hist": 0,
        "bad_main": 0,
        "pool_bad": 0,
        "diff_b": 0,
        "diff_c": 0,
        "touched": 0,
        "w_wrong": 0,
        "skip_missed": 0,
        "hist_rows": 0,
        "main_rows": 0,
        "w_caught": 0,
    }
    seen = {
        "B": set(),
        "first": 0,
        "middle": 0,
        "last": 0,
        "multi": 0,
        "all_but_one": 0,
        "D": set(),
        "ps": set(),
        "r2": 0,
        "r0": 0,
        "zero_compute": 0,
        "max_empty": 0,
        "max_start": 0,
        "calls_with_empty": 0,
        "widths": set(),
    }
    for i, (step, ps) in enumerate(cases):
        r = _replay_case(consts, step, ps, 50_000 + i)
        tot["bad_hist"] += r["bad_hist"]
        tot["bad_main"] += r["bad_main"]
        tot["pool_bad"] += int(not r["pool_ok"]) + int(not r["main_pool_ok"])
        tot["diff_b"] += int(not r["same_b"])
        tot["diff_c"] += int(not r["same_c"])
        tot["touched"] += int(not r["untouched"])
        tot["w_wrong"] += int(r["caught_w"] != r["empty_before"])
        tot["w_caught"] += int(r["caught_w"])
        tot["skip_missed"] += int(r["bad_skip"] == 0)
        tot["hist_rows"] += r["hist_rows"]
        tot["main_rows"] += r["main_rows"]
        nc, empty = len(step["kinds"]), r["empty"]
        seen["B"].add(nc)
        seen["ps"].add(ps)
        if step["G"]:
            seen["D"].add(step["D"])
        if empty:
            seen["calls_with_empty"] += 1
            seen["first"] += int(0 in empty)
            seen["last"] += int(nc - 1 in empty)
            seen["middle"] += int(any(0 < b < nc - 1 for b in empty))
            seen["multi"] += int(len(empty) >= 2)
            seen["all_but_one"] += int(nc >= 2 and len(empty) == nc - 1)
            seen["max_empty"] = max(seen["max_empty"], len(empty))
        seen["r2"] += step["kinds"].count("r2")
        seen["r0"] += step["kinds"].count("r0")
        seen["zero_compute"] += sum(
            1
            for b, (_, ids, _, _) in enumerate(step["main"][:nc])
            if step["replay"][b] and ids.numel() == step["replay"][b]
        )
        seen["max_start"] = max(
            [seen["max_start"]] + [s for _, ids, s, _ in step["hist"] if ids.numel()]
        )
        seen["widths"].add(step["width"])
    compiled = _jit_keys(fn) - variants
    print(
        f"\n  {len(cases)} replay steps ({len(_REPLAY_CORNERS)} corners x pages 128 / 256, then "
        f"{_REPLAY_PARTITIONS} random at 128): history calls {tot['hist_rows']} rows, main calls "
        f"{tot['main_rows']} rows; B {min(seen['B'])}..{max(seen['B'])} ({len(seen['B'])} values); "
        f"{seen['calls_with_empty']} history calls with empty runs (first {seen['first']}, middle "
        f"{seen['middle']}, last {seen['last']}; >= 2 empties {seen['multi']}, all but one empty "
        f"{seen['all_but_one']}, max {seen['max_empty']}); {seen['r2']} 2-row runs, {seen['r0']} replays from 0, "
        f"{seen['zero_compute']} replay chunks of only their replay rows; history start_pos up to "
        f"{seen['max_start']}; generation D {sorted(seen['D'])}; table widths {sorted(seen['widths'])}"
    )
    print(
        f"  {tot['bad_hist']} history rows and {tot['bad_main']} main rows differ from the reference, "
        f"{tot['pool_bad']} calls left the pool inexact; without the empty runs: {tot['diff_b']} calls differ; "
        f"empty runs' start_pos / table rows changed: {tot['diff_c']} differ; {tot['touched']} calls changed an "
        f"empty-run request's page; {len(compiled)} new Triton variant(s)"
    )
    print(
        f"  [control] req_idx over the non-empty runs only: caught in {tot['w_caught']} calls, "
        f"{tot['w_wrong']} calls where caught != (an empty run precedes a non-empty one); history call skipped: "
        f"{tot['skip_missed']} steps where the main call still matched"
    )
    assert tot["bad_hist"] == 0 and tot["bad_main"] == 0 and tot["pool_bad"] == 0
    assert tot["diff_b"] == 0 and tot["diff_c"] == 0 and tot["touched"] == 0
    assert tot["w_wrong"] == 0 and tot["w_caught"] > 0 and tot["skip_missed"] == 0
    assert not compiled
    assert (
        seen["B"] == set(range(1, _ENV_MAX_B + 1))
        and seen["ps"] == {128, 256}
        and seen["D"] == {0, 5}
    )
    assert min(seen["first"], seen["middle"], seen["last"], seen["multi"], seen["all_but_one"]) > 0
    assert (
        seen["r2"] > 0
        and seen["r0"] > 0
        and seen["zero_compute"] > 0
        and seen["max_empty"] == _ENV_MAX_B - 1
    )
    assert seen["widths"] == {_engine_width(L, ds) for L, ds in _L_CONFIGS}
    assert seen["max_start"] == _REPLAY_LM_1M - 128 - _HISTORY_LEN


# ─── the checkpoint's own engram.py and the golden dumps ────────────────────────


def _need_external():
    missing = [d for d in (_GOLDEN_DIR, _CKPT_DIR) if not os.path.isdir(d)]
    if missing:
        pytest.skip(f"golden dumps / checkpoint not present: {missing}")


@functools.lru_cache(maxsize=None)
def _vendor():
    """The checkpoint's inference/engram.py, its NgramHashState built with the checkpoint tokenizer."""
    from transformers import PreTrainedTokenizerFast

    spec = importlib.util.spec_from_file_location(
        "dsv41_vendor_engram", os.path.join(_CKPT_DIR, "inference", "engram.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(_CKPT_DIR, "tokenizer.json"))
    assert len(tok) == VOCAB
    args = types.SimpleNamespace(
        engram_layer_ids=list(LAYER_IDS),
        engram_max_ngram_size=MAX_NGRAM,
        engram_n_heads=N_HEADS,
        engram_vocab_size=ENGRAM_VOCAB,
        engram_num_embeddings=list(NUM_EMBEDDINGS),
        engram_head_dim=256,
        engram_compressed_vocab_size=COMPRESSED_VOCAB,
        engram_pad_id=PAD_TOKEN_ID,
        max_batch_size=1,
        max_seq_len=4096,
    )
    layout = mod.EngramLayout.from_args(args)
    return mod, layout, args, tok


@functools.lru_cache(maxsize=None)
def _vendor_state():
    """One vendor NgramHashState. Reusable across schedules: its cache is position-indexed and every
    schedule here starts at position 0, so each position is written before it is read."""
    mod, layout, args, tok = _vendor()
    return mod.NgramHashState(args, layout, tok)


@functools.lru_cache(maxsize=None)
def _real_consts() -> _Consts:
    state = _vendor_state()
    consts = _Consts(state.token_map.to(torch.int32))
    # the vendor's own buffers equal the constants rebuilt here
    assert torch.equal(state.multipliers, consts.multipliers)
    assert torch.equal(state.primes, consts.primes)
    assert torch.equal(state.offsets, consts.offsets)
    assert state.pad_id == consts.pad_id
    return consts


def _golden(case):
    tokens = torch.load(os.path.join(_GOLDEN_DIR, case, "tokens.pt"), weights_only=True)
    gold = torch.load(os.path.join(_GOLDEN_DIR, case, "engram_hash.pt"), weights_only=True)
    return (
        tokens["input_ids"].to(torch.int64),
        int(tokens["n_prompt"]),
        gold["hash_ids"],
        gold["compressed_ids"],
    )


def _golden_runs(n_prompt: int, total: int):
    """Chunked prompt (7, 1, 2, 64, rest) then one decode token per generated position."""
    runs, at = [], 0
    for c in (7, 1, 2, 64):
        if at + c > n_prompt:
            break
        runs.append((at, at + c))
        at += c
    if at < n_prompt:
        runs.append((at, n_prompt))
    return runs + [(p, p + 1) for p in range(n_prompt, total)]


@pytest.mark.parametrize("case", _GOLDEN_CASES)
def test_golden_and_vendor_chunked_and_decode(case):
    _need_external()
    consts = _real_consts()
    ids, n_prompt, gold, gold_comp = _golden(case)
    T = ids.numel()
    assert torch.equal(consts.token_map[ids].long(), gold_comp.long())
    spans = _golden_runs(n_prompt, T)

    # one-shot over every dumped row
    pool = _Pool({0: T}, 128, seed=T)
    one = _call(consts, pool, [(0, ids, 0, None)])
    assert torch.equal(one, gold)

    # chunked prefill + decode, alone and in a batch with an unrelated request
    pool = _Pool({0: T, 1: 64 + T}, 128, seed=T + 1)
    other = _ids(64 + T, torch.Generator().manual_seed(T))
    vendor = _vendor_state()
    got = []
    for k, (a, b) in enumerate(spans):
        mine_run = (0, ids[a:b], a, None)
        runs = [mine_run]
        if k % 2:
            extra = (1, other[a:b], a, None)
            runs = [extra, mine_run] if b - a == 1 else [mine_run, extra]
        out = _call(consts, pool, runs)
        at = 0
        for r, x, _, _ in runs:
            if r == 0:
                mine = out[at : at + x.numel()]
            at += x.numel()
        got.append(mine)
        assert torch.equal(mine, vendor(ids[None, a:b], a)[0]), (case, a, b)
    got = torch.cat(got)
    assert torch.equal(got, gold)

    # controls, each against the golden ids
    def frac(x):
        return f"{int((x != gold).flatten(1).any(1).sum())}/{T} rows"

    # This tokenizer maps token 2 to class 2, so pad_id = token_map[2] happens to equal 2 here and a
    # pad_id-2 mistake is invisible on real data; test_pad_id_is_the_compressed_pad_token shows it on a
    # map where the two differ. The raw-id control below is the one a missing token map fails.
    assert consts.pad_id == PAD_TOKEN_ID
    pool = _Pool({0: T}, 128, seed=3)
    raw_map = torch.arange(VOCAB, dtype=torch.int32, device=DEV)
    raw = torch.cat(
        [_call(consts, pool, [(0, ids[a:b], a, None)], token_map=raw_map) for a, b in spans]
    )
    ctl = {"raw ids (token map skipped)": raw}
    for variant in ("int32", "float64"):
        ref = _RefState(consts, T, variant=variant)
        ctl[variant] = torch.cat([ref.forward(ids[a:b], a) for a, b in spans])
    ref = _RefState(consts, T, multipliers=_multipliers((0, 1)))
    ctl["seeds from layer index 0/1"] = torch.cat([ref.forward(ids[a:b], a) for a, b in spans])
    pool = _Pool({0: T}, 128, seed=3)
    nohist = []
    for a, b in spans:
        pool.pool.fill_(DEAD)
        nohist.append(_call(consts, pool, [(0, ids[a:b], a, None)]))
    ctl["history not carried"] = torch.cat(nohist)
    ref = _RefState(consts, T)
    assert torch.equal(torch.cat([ref.forward(ids[a:b], a) for a, b in spans]), gold)
    print(
        f"[control] {case} (T={T}, pad_id={consts.pad_id}): "
        + ", ".join(f"{k}: {frac(v)}" for k, v in ctl.items())
    )
    for name, x in ctl.items():
        assert not torch.equal(x, gold), name


def test_vendor_image_spans_chunked():
    """Real token map, DEAD spans (vendor token_mask) across chunk and decode boundaries."""
    _need_external()
    consts = _real_consts()
    ids, n_prompt, _, _ = _golden("long_retrieval")
    T = ids.numel()
    mask = torch.ones(T, dtype=torch.bool)
    for a, b in (
        (0, 1),
        (5, 9),
        (70, 76),
        (80, 81),
        (140, 200),
        (n_prompt - 2, n_prompt + 1),
        (T - 3, T - 2),
    ):
        mask[a:b] = False
    spans = _golden_runs(n_prompt, T)
    pool = _Pool({0: T}, 128, seed=6)
    vendor = _vendor_state()
    wrong = _RefState(consts, T, variant="noncumulative")
    got, ctl = [], []
    for a, b in spans:
        out = _call(consts, pool, [(0, ids[a:b], a, mask[a:b])])
        assert torch.equal(out, vendor(ids[None, a:b], a, mask[None, a:b])[0]), (a, b)
        got.append(out)
        ctl.append(wrong.forward(ids[a:b], a, mask[a:b]))
    got, ctl = torch.cat(got), torch.cat(ctl)
    print(
        "[control] non-cumulative blocking vs vendor, image spans: "
        f"{int((ctl != got).flatten(1).any(1).sum())}/{T} rows differ"
    )
    assert not torch.equal(ctl, got)
