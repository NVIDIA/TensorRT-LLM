# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 window entry's launch levers change no output byte (``rubin_dflash2_window_triton``).

``vec_io`` moves the rows of the window's two kernels as 16-byte vectors where every stride allows
(``ALIGNED``); ``pdl`` launches the tail as a programmatic dependent of the body, waiting for the
body grid only before its first read of the body's partials. They change how the kernels fetch and
store bytes and when the tail starts, never what they compute, so every gate here is equality of
bits against the entry with the levers off: ``ctx`` as BF16 bits and the helix stats as FP32 bits,
signed zeros and NaN payloads included.

GPU (SM107), at helix8's shape: blocks of 6, 96 heads, latent 512 + rope 64, 64-token FP8 pages in
the KV cache manager's five-layer page stride, window 1024, one body split, ``kv_scale`` 1.0. Every
graph size B = 1 .. 24 on every CP rank of 8, eager and CUDA graph replay (captured with the plan on
one input set, replayed on another), over histories at every window and page edge, dummy rows,
and AgentX depths (132,096 +- 2,048, every residue mod 64); value sets with every FP8 code (the NaN
codes in a set of their own), wide score spreads, signed zeros, a dominant key and huge queries.
Every pool row and block row outside a rank's plan holds FP8 NaN codes, the page table is read
through permuted ``table_rows``, and the body workspace and the stats hold a NaN pattern before
every call: a lever that read an unplanned row or left a byte unwritten would differ. Rows without
a key of the rank keep the helix encoding: ``ctx`` +0 and stats ``(-inf, +0)``. The PDL graphs
also replay many times over changing inputs with the workspace poisoned before each replay, so a
partial read ahead of the tail's grid wait would show; with the engine-wide switch off
(``TRTLLM_ENABLE_PDL=0``) the PDL lever falls back to ordinary launches, with the same bits.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.rubin_dflash2_window_plan import (  # noqa: E501
    plan_buffers,
    rubin_dflash2_window_plan,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.rubin_dflash2_window_triton import (  # noqa: E501
    rubin_dflash2_window_triton as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.rubin_dflash2_window_triton import (  # noqa: E501
    window_vec_io_aligned,
    window_workspace,
)
from tensorrt_llm._utils import get_sm_version

_ON_SM107 = torch.cuda.is_available() and get_sm_version() == 107
pytestmark = pytest.mark.skipif(not _ON_SM107, reason="the window entry runs on SM107 (Rubin)")

# helix8's drafter window: blocks of 6, 96 heads, kv_lora 512 + rope 64, 64-token FP8 pages of a
# five-layer pool, window 1024, CP 8, one body split, kv_scale 1.0.
_BLK, _HEADS, _LATENT, _ROPE = 6, 96, 512, 64
_DIM = _LATENT + _ROPE
_PAGE, _WINDOW, _CP, _LAYERS, _SPLITS = 64, 1024, 8, 5, 1
_SCALE = 0.13086080
_KV_SCALE = 1.0
#: The NaN pattern every workspace and stats buffer holds before a call.
_POISON = 0x7FC00001

#: Histories at every edge (empty, below the block, the page edges where the head and the block
#: change ranks, the window edge), the dummy rows 0 .. 5 of a padded graph, a long history, and
#: AgentX depths 132,096 +- 2,048 that take every residue mod 64.
_EDGES = (0, 1, 2, 3, 4, 5, 6, 58, 59, 63, 64, 65, 127, 128, 129, *range(1017, 1032), 190000)
#: The edge histories that end before the fourth page: CP ranks 3 .. 7 hold no key of their rows.
_SHORT = _EDGES[:15]
_AGENTX = tuple(132096 - 2048 + 65 * i for i in range(64))
_LENGTHS = _EDGES + _AGENTX
_BATCHES = tuple(range(1, 25))
#: The value sets (``build``).
_SETS = ("normal", "wide", "codes", "zeros", "dominant", "huge", "nan")
#: The levers against the entry with all of them off.
_VARIANTS = {
    "vec_io": dict(vec_io=True),
    "pdl": dict(pdl=True),
    "vec_io+pdl": dict(vec_io=True, pdl=True),
}
#: Graph sizes where the PDL tail overlaps the body's last wave most (212 SMs: B in 1..5, 12..17
#: and 24 leave the tail's last wave room on SMs the body's last wave leaves idle), replays per
#: graph, and the input sets the replays cycle through.
_RACE_BATCHES = (1, 5, 12, 16, 17, 24)
_RACE_REPLAYS = 200
_RACE_SETS = 8


def _lengths(batch: int, shift: int = 0) -> list[int]:
    """The B = 1 .. 24 graphs walk the length list (300 requests over its 95 lengths). A graph
    whose walk holds no short history takes one as its last request, so every graph has rows
    without a key of some rank (the empty-row encoding)."""
    start = batch * (batch - 1) // 2 + shift
    lengths = [_LENGTHS[(start + r) % len(_LENGTHS)] for r in range(batch)]
    if min(lengths) > max(_SHORT):
        lengths[-1] = _SHORT[batch % len(_SHORT)]
    return lengths


def _nan_fill(t: torch.Tensor) -> torch.Tensor:
    """FP8 NaN codes over a byte tensor, 0x7F and 0xFF alternating along its last dimension."""
    t.fill_(0x7F)
    t[..., 1::2] = 0xFF
    return t


@dataclasses.dataclass
class Inputs:
    lengths: list
    table: torch.Tensor  # [B + 3, global pages] int64: request b at row table_rows[b]
    table_rows: torch.Tensor  # [B] int32
    pool: torch.Tensor  # [pages, layers, P, D] bytes of FP8; page 0 and unplanned rows NaN codes
    query: torch.Tensor  # [R, H, D] BF16
    block_kv: torch.Tensor  # [R, D] bytes of FP8, every row owned by some rank

    def num_ctx(self) -> torch.Tensor:
        return torch.tensor(self.lengths, dtype=torch.int64, device=self.pool.device)

    def local_table(self, rank: int) -> torch.Tensor:
        """Rank ``rank``'s table: its local page ``c`` is global page ``c * CP + rank``."""
        return self.table[:, rank::_CP].to(torch.int32).contiguous()

    def block_for(self, rank: int) -> torch.Tensor:
        """``block_kv`` with every row the rank does not own as NaN codes."""
        own = [(length + k) // _PAGE % _CP == rank for length in self.lengths for k in range(_BLK)]
        own = torch.tensor(own, device=self.pool.device)
        nan = _nan_fill(torch.empty_like(self.block_kv))
        return torch.where(own[:, None], self.block_kv, nan)


def _fp8_bits(values: torch.Tensor) -> torch.Tensor:
    return values.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)


def _codes(g, shape, device, nan: bool) -> torch.Tensor:
    """Uniform FP8 codes: every code, or every code but the NaN pair (those become +-448)."""
    c = torch.randint(0, 256, shape, generator=g, device=device, dtype=torch.int32)
    if not nan:
        c = torch.where((c & 0x7F) == 0x7F, (c & 0x80) | 0x7E, c)
    return c.to(torch.uint8)


def build(lengths, value_set: str, seed: int, device="cuda") -> Inputs:
    """A five-layer FP8 pool holding each request's window pages; every other row NaN codes."""
    g = torch.Generator(device=device).manual_seed(seed)
    batch, rows = len(lengths), len(lengths) * _BLK
    span = _PAGE * _CP
    gcols = -(-(max(lengths) + _BLK + span) // span) * _CP
    table = torch.zeros(batch, gcols, dtype=torch.int64)
    count = 1
    for b, length in enumerate(lengths):
        first, last = max(0, length - _WINDOW - _BLK) // _PAGE, (length + _BLK - 1) // _PAGE
        for gp in range(first, last + 1):
            table[b, gp] = count
            count += 1
    perm = torch.randperm(count - 1, generator=torch.Generator().manual_seed(seed)) + 1
    table = torch.where(table > 0, perm[(table - 1).clamp_min(0)], table)

    shape = (count, _LAYERS, _PAGE, _DIM)
    if value_set in ("codes", "nan"):
        pool = _codes(g, shape, device, nan=value_set == "nan")
        block_kv = _codes(g, (rows, _DIM), device, nan=value_set == "nan")
    elif value_set == "zeros":
        # -0 keys for even requests, +0 for odd ones: every visible score is a signed zero.
        pool = torch.zeros(shape, dtype=torch.uint8, device=device)
        block_kv = torch.zeros(rows, _DIM, dtype=torch.uint8, device=device)
        for b in range(batch):
            code = 0x80 if b % 2 == 0 else 0x00
            pool[table[b][table[b] > 0].to(device)] = code
            block_kv[b * _BLK : (b + 1) * _BLK] = code
    else:
        sigma = 0.05 if value_set == "dominant" else 0.5
        pool = _fp8_bits(sigma * torch.randn(shape, generator=g, device=device))
        block_kv = _fp8_bits(sigma * torch.randn(rows, _DIM, generator=g, device=device))
    if value_set == "dominant":
        # One key per request, at its newest history position, far above the others.
        big = _fp8_bits(torch.full((_DIM,), 448.0, device=device))
        for b, length in enumerate(lengths):
            if length:
                pool[int(table[b, (length - 1) // _PAGE]), :, (length - 1) % _PAGE] = big

    query = 0.5 * torch.randn(rows, _HEADS, _DIM, generator=g, device=device)
    if value_set == "wide":
        query[:, ::7] *= 64.0  # s - m below -150 for most keys: ex2 flushes to zero
    elif value_set == "zeros":
        sign = torch.randint(0, 2, (rows, _HEADS // 2, _DIM), generator=g, device=device)
        query[:, : _HEADS // 2] = torch.where(sign == 1, -0.0, 0.0)  # +-0 query rows
    elif value_set == "huge":
        query[:, ::5] = torch.where(query[:, ::5] < 0, -3.0e38, 3.0e38)
    query = query.bfloat16()

    # Unplanned rows: the NaN page 0, and every row of a window page outside [lo_0, L).
    _nan_fill(pool[0])
    nan_page = _nan_fill(torch.empty_like(pool[0]))
    for b, length in enumerate(lengths):
        lo0 = max(0, length - _WINDOW + 1)
        for gp in torch.nonzero(table[b] > 0).flatten().tolist():
            pos = gp * _PAGE + torch.arange(_PAGE)
            dead = ((pos < lo0) | (pos >= length)).to(device)
            page = int(table[b, gp])
            pool[page] = torch.where(dead[None, :, None], nan_page, pool[page])

    # Request b reads row table_rows[b] of a taller table; rows nobody reads point at page 0.
    order = torch.randperm(batch + 3, generator=torch.Generator().manual_seed(seed + 1))[:batch]
    tall = torch.zeros(batch + 3, gcols, dtype=torch.int64)
    tall[order] = table
    return Inputs(
        lengths=list(lengths),
        table=tall.to(device),
        table_rows=order.to(torch.int32).to(device),
        pool=pool,
        query=query,
        block_kv=block_kv,
    )


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.int16) if x.element_size() == 2 else x.view(torch.int32)


def _poison(*tensors: torch.Tensor) -> None:
    for t in tensors:
        t.view(torch.int32).fill_(_POISON)


class _Static:
    """The persistent buffers of one graph; every input set copies into them."""

    def __init__(self, sets: list, rank: int):
        batch = len(sets[0].lengths)
        dev = sets[0].pool.device
        cols = max(s.local_table(rank).shape[1] for s in sets)
        self.rank, self.layer = rank, 0
        self.num_ctx = torch.zeros(batch, dtype=torch.int64, device=dev)
        self.table = torch.zeros(batch + 3, cols, dtype=torch.int32, device=dev)
        self.table_rows = torch.zeros(batch, dtype=torch.int32, device=dev)
        pages = max(s.pool.shape[0] for s in sets)
        self.pool = torch.empty(pages, _LAYERS, _PAGE, _DIM, dtype=torch.uint8, device=dev)
        self.query = torch.empty(batch * _BLK, _HEADS, _DIM, dtype=torch.bfloat16, device=dev)
        self.block_kv = torch.empty(batch * _BLK, 1, _DIM, dtype=torch.uint8, device=dev)
        self.stats = torch.empty(batch * _BLK, _HEADS, 2, device=dev)
        self.plan = plan_buffers(batch, _BLK, _WINDOW, _PAGE, _CP, _SPLITS, dev)
        self.ws = window_workspace(batch, _BLK, _HEADS, _LATENT, _SPLITS, dev)

    def load(self, s: Inputs, layer: int) -> None:
        self.num_ctx.copy_(s.num_ctx())
        local = s.local_table(self.rank)
        self.table.zero_()
        self.table[:, : local.shape[1]] = local
        self.table_rows.copy_(s.table_rows)
        _nan_fill(self.pool)
        self.pool[: s.pool.shape[0]] = s.pool
        self.query.copy_(s.query)
        self.block_kv[:, 0] = s.block_for(self.rank)
        self.layer = layer

    def views(self):
        """One layer's ``[pages, P, D]`` pool view (page stride: five layers) and the block rows."""
        pool = self.pool.view(torch.float8_e4m3fn)[:, self.layer]
        return pool, self.block_kv.view(torch.float8_e4m3fn)[:, 0]

    def step(self, **levers) -> torch.Tensor:
        """The drafter's per-step plan, then one layer's window entry (what a graph holds)."""
        plan = rubin_dflash2_window_plan(
            self.num_ctx,
            self.table,
            self.table_rows,
            _BLK,
            _WINDOW,
            _PAGE,
            _CP,
            self.rank,
            _SPLITS,
            self.plan,
        )
        pool, block_kv = self.views()
        return op(
            self.query,
            pool,
            block_kv,
            plan,
            _SCALE,
            _LATENT,
            _KV_SCALE,
            self.ws,
            self.stats,
            **levers,
        )

    def run(self, **levers):
        """Eager: a poisoned workspace and stats, then the step; the output and stats bits."""
        _poison(self.ws.partial_out, self.ws.partial_stats, self.stats)
        out = self.step(**levers)
        return _bits(out).clone(), _bits(self.stats).clone()


_NEG_INF_BITS = int(torch.tensor(float("-inf")).view(torch.int32))


def _empty_rows(out_bits: torch.Tensor, stats_bits: torch.Tensor) -> int:
    """Rows without a key of the rank: ``ctx`` +0 everywhere and stats ``(-inf, +0)``."""
    empty = stats_bits[..., 1] == 0
    assert (stats_bits[..., 0][empty] == _NEG_INF_BITS).all(), "an empty row's max is not -inf"
    assert (out_bits[empty] == 0).all(), "an empty row's context is not +0"
    return int(empty.sum())


def _assert_same(got, want, what: str) -> None:
    assert torch.equal(got[0], want[0]), f"{what}: ctx bits differ"
    assert torch.equal(got[1], want[1]), f"{what}: stats bits differ"


@pytest.mark.parametrize("batch", _BATCHES)
def test_eager_bits_equal_with_the_levers_off(batch):
    """Every rank, every value set, every lever against the entry with the levers off."""
    empty = 0
    for k, value_set in enumerate(_SETS):
        inp = build(_lengths(batch), value_set, seed=1000 * batch + k)
        for rank in range(_CP):
            st = _Static([inp], rank)
            st.load(inp, layer=(batch + k) % _LAYERS)
            off = st.run()
            empty += _empty_rows(*off)
            for name, levers in _VARIANTS.items():
                _assert_same(st.run(**levers), off, f"B {batch} rank {rank} {value_set} {name}")
    assert empty > 0, "no rank held an empty row: the empty-row encoding went unchecked"


@pytest.mark.parametrize("batch", _BATCHES)
def test_graph_replay_bits_equal_with_the_levers_off(batch):
    """Each lever, captured with the per-step plan on one input set and replayed on another (new
    lengths, values and table rows; the workspace poisoned before the replay, outside the graph),
    equals the entry with the levers off, eager, on the second set."""
    x = build(_lengths(batch), "normal", seed=7 * batch)
    y = build(_lengths(batch, shift=11), _SETS[batch % len(_SETS)], seed=7 * batch + 1)
    layer = batch % _LAYERS
    for rank in range(_CP):
        st = _Static([x, y], rank)
        st.load(y, layer)
        want = st.run()
        for name, levers in _VARIANTS.items():
            st.load(x, layer)
            st.run(**levers)  # eager warm-up: every kernel compiled before the capture
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = st.step(**levers)
            st.load(y, layer)
            _poison(st.ws.partial_out, st.ws.partial_stats, st.stats)
            graph.replay()
            torch.cuda.synchronize()
            got = (_bits(out).clone(), _bits(st.stats).clone())
            _assert_same(got, want, f"B {batch} rank {rank} {name} replay")
            del graph


@pytest.mark.parametrize("batch", _RACE_BATCHES)
def test_pdl_replays_read_no_partial_before_the_body_ends(batch, monkeypatch):
    """One graph per PDL variant, replayed ``_RACE_REPLAYS`` times, each time on another input set
    with the workspace and stats poisoned before the replay: every replay equals the levers-off
    result of its set. Then, with ``TRTLLM_ENABLE_PDL=0``, the PDL lever runs ordinary launches."""
    rank, layer = batch % _CP, batch % _LAYERS
    sets = [
        build(_lengths(batch, shift=5 * k), _SETS[k % len(_SETS)], seed=31 * batch + k)
        for k in range(_RACE_SETS)
    ]
    st = _Static(sets, rank)
    wants = []
    for s in sets:
        st.load(s, layer)
        wants.append(st.run())
    for name in ("pdl", "vec_io+pdl"):
        levers = _VARIANTS[name]
        st.load(sets[0], layer)
        st.run(**levers)  # eager warm-up: every kernel compiled before the capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = st.step(**levers)
        for k in range(_RACE_REPLAYS):
            st.load(sets[k % _RACE_SETS], layer)
            _poison(st.ws.partial_out, st.ws.partial_stats, st.stats)
            graph.replay()
            got = (_bits(out).clone(), _bits(st.stats).clone())
            _assert_same(got, wants[k % _RACE_SETS], f"B {batch} rank {rank} {name} replay {k}")
        del graph
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "0")
    for k, s in enumerate(sets[:2]):
        st.load(s, layer)
        for name in ("pdl", "vec_io+pdl"):
            _assert_same(st.run(**_VARIANTS[name]), wants[k], f"B {batch} {name} PDL off")


def test_production_strides_take_the_vector_path():
    """helix8's buffers pass every alignment check of both kernels, so the tests above run the
    vector path, not a refused one."""
    inp = build(_lengths(24), "normal", seed=5)
    st = _Static([inp], 0)
    st.load(inp, layer=3)
    pool, block_kv = st.views()
    assert pool.stride(0) == _LAYERS * _PAGE * _DIM
    assert window_vec_io_aligned(st.query, pool, block_kv, st.ws, _LATENT)
