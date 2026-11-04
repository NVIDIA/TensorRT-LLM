# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The bound on the tep4 target's context-row attention at the 1M config (max_seq_len 1048576,
chunked prefill: a held context workspace of 2**19 rows, `_ctx_workspace_rows`), on one GPU, at a
rank's heads at tep4 (four ranks: 16 of the 64 MLA heads) and at tep8 (eight ranks over two nodes
of four: 8 of them).

`PrefillTarget._ctx_workspace` picks each step's path on host ints from the step's context KV N
(every context request's cached and new rows): while N fits the held rows the workspace path (the
rows dequantized into the first N rows of the held bf16 [2**19, 576] buffer, then FlashMLA's sparse
prefill), past them the decode path (FlashMLA's sparse fp8 decode straight off the pool, each
context row a one-row sequence, in sub-batches of at most 8192 rows). `PrefillTarget._attn_ctx` runs
it. Both are driven here as the forward drives them, at the rank's live heads (16 at tep4, 8 at
tep8) first in the kernels' 64-head query (the others zero), over an fp8_ds_mla latent pool of the
63-owner layout built on the device: every byte is 0xFF (NaN as e4m3, as a tile's fp32 scale and as
bf16 rope) but the owner rows of the driven layers on the requests' pages, so a row read from
anywhere else turns its output NaN. Each step runs layer 35 (the NoPE owner), then layer 1 (a RoPE
owner, whose gather overwrites the workspace's first N rows), then layer 39, which attends over
layer 35's latent: on a workspace-path step whose N fits the held rows twice it reads layer 35's
rows where layer 35's gather kept them, beside the workspace's first N, and otherwise gathers them
again; generation rows behind the context rows run as `_attn_gen` runs them, on
cute_dsl_ds_mla_sparse_decode at the rank's heads (FlashMLA's arithmetic, unpadded) beside the
context rows' FlashMLA schedulers.

Every step is checked three ways: every output row finite; each layer's output against the other
path on the same step (a second target whose held workspace is sized so that it takes it); and
up to 512 rows, the sub-batch edges among them, against the decode entry's fp32 reference over
slots the test computes itself (block * 63 + owner row) * 32 + offset. The two paths read the
same pool bytes through the same bf16 arithmetic and are bitwise equal on every row FlashMLA's
decode runs in one split. Its scheduler splits some rows over their keys (its `num_splits`
says which: none of an 8192-row sub-batch here, some of a ragged one, every row of a 16-row
tail) and its combine kernel merges the parts in fp32, which rounds otherwise than the prefill
kernel's one pass: those rows are held to the entry's band instead (relative L2 per row and head
under 1e-2, the rows under 4e-3). The generation rows run the same code on both and are bitwise
equal.

The steps: the KV-capacity estimate's (one fresh 32768-token prompt, split over the ranks: the
workspace path, which allocates the buffer, layer 35's rows kept); the same 32768-row chunk at
context KV 2**19 (workspace, nothing kept) and 2**19 + 1 (decode, 4 x 8192); the 1M gate prompt's
last chunk, 32751 rows over 1015808 cached keys (8192 x 3 and 8175) with three generation rows
behind; a packed step of chunks over 1032192, 524288, 0 and 2048 cached keys with twelve generation
rows; a 16-row tail over 1048544 cached keys; and nine 1000-token prompts on the engine's dense
causal index (the indexer skipped, layer 35's rows kept). The buffer is never reallocated after the
estimate's step, and the context attention's peak on a decode-path step stays under the estimate
step's.

The prefill kernel skips an index past its KV rows as it skips -1, with no error and no NaN, so
a workspace shorter than the step's context KV drops selections silently: every workspace-path
run is checked to hold exactly the step's context KV rows with every selection inside them, and
controls run the target's prefill over the same step with the workspace one row short and with
it cut to the step's new tokens (the engine's `num_ctx_kv_tokens`), which the checks refuse.
The query's heads past the rank's (16..63 at tep4, 8..63 at tep8) hold no head of this rank and
stay zero: NaN there leaves every output row bitwise unchanged on both paths, so nothing the
kernels compute on them reaches the rank's heads.

Every step runs at both head counts (`-k tep4`, `-k tep8`); the steps, the pool and the paths
do not depend on them.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_103__tep4 import (  # noqa: E501
    modeling as M,
)


def _skip_reason() -> str | None:
    if not torch.cuda.is_available():
        return "needs a GPU"
    from tensorrt_llm._utils import get_sm_version

    if get_sm_version() != 103:
        return "the mistral_large4 modeling_v2 targets are certified on sm_103 only"
    return None


pytestmark = pytest.mark.skipif(_skip_reason() is not None, reason=str(_skip_reason()))

_LAYERS = 72
#: Every fourth layer from 39 on attends over layer 35's latent (the checkpoint's
#: kv_source_layers); the pool holds the 63 owners' rows.
_SOURCES = [35 if i >= 39 and i % 4 == 3 else i for i in range(_LAYERS)]
_OWNERS = [i for i in range(_LAYERS) if _SOURCES[i] == i]
#: Driven in this order: the NoPE source, a RoPE owner, a sharer of the source.
_DRIVEN = (35, 1, 39)
_TPB = 32
_TOPK = 2048
#: The rank's live heads of the 64 MLA heads: 16 at tep4 (four ranks), 8 at tep8 (eight ranks
#: over two nodes of four).
HEADS = [pytest.param(16, id="tep4"), pytest.param(8, id="tep8")]
_KERNEL_HEADS = 64
_LAT, _NOPE, _ROW_BYTES = 576, 512, 656
#: qk_nope_head_dim + qk_rope_head_dim = 192, every layer's softmax scale.
_SCALE = 192**-0.5
#: md.block_table at the 1M config: 16 requests of ceil(1048576 / 32) + 4 columns.
_TABLE = (16, 32772)
_ROWS_B = 2**19
_SUB = 8192
_FILL = 0xFF
_DRAW = 1 << 18
_GATED = 512
_MIB = 2**20


def test_constants_are_the_rows_certifications():
    """The held rows at the 1M config (16 requests of 32768 tokens, max_seq_len 1048576, chunked
    prefill) are the workspace's 2**19 ceiling; the decode path's sub-batch is 8192 rows."""
    assert (M._CTX_WORKSPACE_ROWS, M._CTX_DECODE_ROWS) == (_ROWS_B, _SUB)
    features = SimpleNamespace(chunked_prefill=True, cache_reuse=False)
    md = SimpleNamespace(
        max_num_tokens=32768, max_num_requests=16, max_seq_len=1048576, runtime_features=features
    )
    assert M._ctx_workspace_rows(md) == _ROWS_B
    assert len(_OWNERS) == 63 and _OWNERS.index(35) == 35 and _SOURCES[39] == 35


# ---------------------------------------------------------------- the step, built on the device


def _draw_rows(n: int, g: torch.Generator, nope: bool) -> torch.Tensor:
    """`n` fp8_ds_mla rows [656] as the writer leaves them: e4m3 codes, power-of-two fp32 tile
    scales 2^-13..2^0, the bf16 rope key (zero on a NoPE layer)."""
    dev = g.device
    codes = ((torch.rand(n, _NOPE, generator=g, device=dev) * 2 - 1) * 448).to(torch.float8_e4m3fn)
    scale = torch.exp2(torch.randint(-13, 1, (n, 4), generator=g, device=dev).float())
    rope = torch.zeros(n, 64, device=dev)
    if not nope:
        rope = torch.randn(n, 64, generator=g, device=dev) * 3
    rope = rope.to(torch.bfloat16)
    return torch.cat([codes.view(torch.uint8), scale.view(torch.uint8), rope.view(torch.uint8)], 1)


def _pool(kv_lens: list[int], g: torch.Generator):
    """The pool [blocks, 63, 32, 656] and the block table: each request on distinct pages
    drawn over the whole pool, the driven owners' rows written on them."""
    dev = g.device
    counts = [-(-kv // _TPB) for kv in kv_lens]
    blocks = sum(counts) + 1
    shape = (blocks, len(_OWNERS), _TPB, _ROW_BYTES)
    pool = torch.full(shape, _FILL, dtype=torch.uint8, device=dev)
    pages = torch.randperm(blocks, generator=g, device=dev)[: sum(counts)]
    table = torch.zeros(_TABLE, dtype=torch.int32, device=dev)
    first = 0
    for r, n in enumerate(counts):
        table[r, :n] = pages[first : first + n].int()
        first += n
    for layer in (1, 35):
        row = _OWNERS.index(layer)
        for lo in range(0, pages.numel(), _DRAW // _TPB):
            part = pages[lo : lo + _DRAW // _TPB]
            pool[part, row] = _draw_rows(part.numel() * _TPB, g, layer == 35).view(-1, _TPB, 656)
    return pool, table


def _selections(pos: torch.Tensor, g: torch.Generator, dense: bool) -> torch.Tensor:
    """int32 [rows, 2048] request-local positions for rows at positions `pos`, as the indexer
    hands them over: a row that sees at most 2048 keys has all of them, then -1; else 2048
    distinct keys, one in each of 2048 equal strata of [0, p], p itself the last, in a random
    order. `dense`: the engine's causal index of a step it skipped the indexer (0..p, -1 after)."""
    dev = g.device
    n_keys = pos[:, None].long() + 1
    col = torch.arange(_TOPK + 1, device=dev)[None]
    short = torch.where(col[:, :-1] < n_keys, col[:, :-1], -1)
    if dense:
        return short.int()
    edge = col * n_keys // _TOPK
    draw = (
        edge[:, :-1]
        + (
            torch.rand(pos.numel(), _TOPK, generator=g, device=dev) * (edge[:, 1:] - edge[:, :-1])
        ).long()
    )
    draw[:, -1] = pos
    del edge
    sel = torch.where(n_keys > _TOPK, draw, short)
    del draw, short
    order = (torch.rand(sel.shape, generator=g, device=dev) + (sel < 0)).argsort(1)
    return sel.gather(1, order).int()


def _cumsum(lens: list[int]) -> torch.Tensor:
    return torch.tensor([0, *lens], dtype=torch.int64).cumsum(0)


class _Step(SimpleNamespace):
    pass


def _offsets(ctx) -> dict:
    """The context step entries `_ctx_workspace` reads, as `_ctx_kv_offsets` takes them off the
    engine's metadata under chunked prefill."""
    cached, new = [c for c, _ in ctx], [n for _, n in ctx]
    md = SimpleNamespace(
        enable_context_mla_with_cached_kv=True,
        ctx_cached_token_indptr=_cumsum(cached),
        ctx_kv_indptr=_cumsum([c + n for c, n in ctx]),
        max_ctx_seq_len=max(new),
        num_ctx_cached_tokens=sum(cached),
    )
    offsets = {
        k: v.cuda() if isinstance(v, torch.Tensor) else v
        for k, v in M._ctx_kv_offsets(md, len(ctx)).items()
    }
    # num_ctx_kv_tokens is the engine's count: the step's new context keys only.
    return dict(
        num_contexts=len(ctx), num_ctx_tokens=sum(new), num_ctx_kv_tokens=sum(new), **offsets
    )


def _step(ctx, gen=(), *, heads: int, seed: int, dense: bool = False, pool_kv=None) -> _Step:
    """A step of context requests `ctx` ((cached, new) each) and generation rows `gen` (the
    KV length of each, one query at its last position) behind them, with the state the
    target reads: the context offsets, the pool, md.block_table, the selections and one query
    per layer kind, whose first `heads` heads are the rank's (the rest zero)."""
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    kv_lens = [c + n for c, n in ctx] + list(gen)
    pool, table = _pool(pool_kv or kv_lens, g)
    tc = sum(n for _, n in ctx)
    rows = tc + len(gen)
    pos = torch.cat(
        [torch.arange(c, c + n, device=dev) for c, n in ctx]
        + [torch.tensor([kv - 1 for kv in gen], device=dev, dtype=torch.long)]
    )
    req = torch.cat(
        [torch.full((n,), r, dtype=torch.int32, device=dev) for r, (_, n) in enumerate(ctx)]
        + [torch.arange(len(ctx), len(kv_lens), dtype=torch.int32, device=dev)]
    )
    sel = _selections(pos, g, dense)
    q = torch.zeros(rows, _KERNEL_HEADS, _LAT, dtype=torch.bfloat16, device=dev)
    q[:, :heads] = (torch.randn(rows, heads, _LAT, generator=g, device=dev) * 0.5).to(q.dtype)
    q_nope = q.clone()
    q_nope[..., _NOPE:] = 0
    step = dict(
        latent_pool=pool,
        latent_kv=pool.view(-1, _TPB, 1, _ROW_BYTES),
        block_table=table,
        tokens_per_block=_TPB,
        **_offsets(ctx),
    )
    return _Step(
        step=step, tc=tc, rows=rows, n_kv=sum(c + n for c, n in ctx), req=req, pos=pos, sel=sel,
        q={35: q_nope, 1: q, 39: q_nope}, g=g, heads=heads,
    )  # fmt: skip


#: cute_dsl_ds_mla_sparse_decode's handles by head count, built once.
_NATIVE = {}


def _native_decode(heads: int):
    """The generation rows' kernel handle at the rank's `heads`, as `build_kernels` builds it."""
    if heads not in _NATIVE:
        _NATIVE[heads] = M.build_ds_mla_decode(torch.cuda.current_device(), num_heads=heads)
    return _NATIVE[heads]


def _target(heads: int, rows: int = _ROWS_B) -> M.PrefillTarget:
    """A prefill target with only what its context and generation attention read off the core:
    the held workspace's rows as `_derive_first_step` sets them (the 1M config's 2**19 unless
    `rows` says otherwise: 0 sends every step down the decode path, a step's own context KV
    sends it down the workspace path), layer 35 the one shared source, as the core builds it,
    and the generation rows' native decode at the rank's `heads`."""
    target = M.PrefillTarget.__new__(M.PrefillTarget)
    target.core = SimpleNamespace(
        ctx_workspace_rows=rows,
        # The generation rows on cute_dsl_ds_mla_sparse_decode at the rank's heads, as shipped.
        gen_attn_flash_mla=False,
        _mla_decode=_native_decode(heads),
        lat_dim=_LAT,
        idx_topk=_TOPK,
        softmax_scale=_SCALE,
        _stride_factor=len(_OWNERS) * _TPB,
        _pool_row=[_OWNERS.index(_SOURCES[i]) for i in range(_LAYERS)],
        shared_source=[i == 35 for i in range(_LAYERS)],
    )
    return target


def _run(target, s: _Step, *, cut=None) -> SimpleNamespace:
    """The step's latent attention on every driven layer, as `PrefillTarget.forward` runs it:
    the context path chosen once (`_ctx_workspace`, on the rows the target's held workspace was
    allocated with on its first context step), then per layer the context rows (`_attn_ctx`)
    and the generation rows behind them (the selections converted to slots, then `_attn_gen`,
    the native decode at the rank's heads), each layer's output kept. On a workspace-path step layer 35's gather
    is kept for layer 39 exactly when the step's context KV fits the held rows twice. With `cut`
    (a control), the workspace path keeps only the first `cut` rows of the step's context KV,
    and every layer gathers into them."""
    ctx = target._ctx_workspace(s.step, s.tc, torch.device("cuda"))
    if ctx.ws_buf is not None:
        held = target._ws_buf.shape[0]
        assert (ctx.kept is not None) == (2 * s.n_kv <= held), (s.n_kv, held)
    if cut is not None:
        assert ctx.ws_buf is not None and cut < ctx.ws_buf.shape[0]
        ctx = SimpleNamespace(
            kv_req=ctx.kv_req[:cut], kv_pos=ctx.kv_pos[:cut], ws_off=ctx.ws_off,
            ws_buf=ctx.ws_buf[:cut], kept=None, ws={},
        )  # fmt: skip
    sched = M.get_mla_metadata()[0]
    outs = {}
    for layer in _DRIVEN:
        L = SimpleNamespace(index=layer, kv_source=_SOURCES[layer])
        lat_out = torch.empty(s.rows, s.heads, _NOPE, dtype=torch.bfloat16, device="cuda")
        lat_out.view(torch.int16).fill_(-1)
        target._attn_ctx(s.step, L, s.q[layer][: s.tc], lat_out[: s.tc], s.req, s.sel, ctx)
        if s.rows > s.tc:
            # The context attention leaves the generation rows alone.
            assert bool((lat_out[s.tc :].view(torch.int16) == -1).all())
            slots = M.convert_req_index_to_global(
                s.req[s.tc :], s.step["block_table"], s.sel[s.tc :], _TPB, _TOPK,
                target.core._stride_factor, target.core._pool_row[L.kv_source],
            )  # fmt: skip
            target._attn_gen(s.step, L, s.q[layer][s.tc :], lat_out[s.tc :], slots, sched)
        outs[layer] = lat_out
    if ctx.ws_buf is not None:
        # Layer 35's gather is remembered for its sharer exactly when it went to the kept rows.
        assert (35 in ctx.ws) == (ctx.kept is not None)
    torch.cuda.synchronize()
    return SimpleNamespace(ctx=ctx, outs=outs)


def _reference_rows(s: _Step) -> torch.Tensor:
    """The rows gated against the fp32 reference: each sub-batch's first and last row, every
    generation row, the rest drawn."""
    edges = {0, s.tc - 1}
    for lo in range(0, s.tc, _SUB):
        edges |= {lo, min(lo + _SUB, s.tc) - 1}
    edges |= set(range(s.tc, s.rows))
    edges = sorted(e for e in edges if 0 <= e < s.rows)
    rest = _GATED - len(edges)
    if rest > 0 and s.tc > len(edges):
        drawn = torch.randperm(s.tc, generator=s.g, device=s.g.device)[:rest].tolist()
        edges = sorted(set(edges) | set(drawn))
    return torch.tensor(edges, device="cuda")


def _split_rows(ctx, tc: int) -> torch.Tensor:
    """The context rows FlashMLA's decode scheduler split over their keys on a decode-path run
    (more than one split in its `num_splits` prefix sums), whose parts its combine kernel merges
    in fp32; none on a workspace-path run."""
    split = torch.zeros(tc, dtype=torch.bool, device="cuda")
    if ctx.ws_buf is None:
        for lo, hi in ctx.sub_batches:
            starts = ctx.sched[hi - lo].num_splits
            split[lo:hi] = (starts[1:] - starts[:-1]) > 1
    return split


def _check(s: _Step, run: SimpleNamespace, other: SimpleNamespace) -> dict:
    """Every row finite; the sharer equal to its source; each layer's context rows bitwise the
    other path's on every row neither run split over its keys, a split row within the band;
    the generation rows bitwise; the gated rows against the fp32 reference."""
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.flash_mla_sparse_decode import (  # noqa: E501
        flash_mla_sparse_decode,
    )

    for r in (run, other):
        if r.ctx.ws_buf is not None:
            # The prefill kernel skips a selection past the workspace's rows as it skips -1.
            held_rows = r.ctx.ws_buf.shape[0]
            assert held_rows == r.ctx.kv_req.shape[0] == s.n_kv, (
                f"the workspace holds {held_rows} rows of the step's {s.n_kv}"
            )
            assert r.ctx.kept is None or r.ctx.kept.shape[0] == s.n_kv
            sel = s.sel[: s.tc]
            idx = torch.where(sel >= 0, sel + r.ctx.ws_off[s.req[: s.tc]][:, None], -1)
            assert int(idx.max()) < held_rows
    rows = _reference_rows(s)
    split = _split_rows(run.ctx, s.tc) | _split_rows(other.ctx, s.tc)
    report = dict(split_rows=int(split.sum()))
    for layer in _DRIVEN:
        out, alt = run.outs[layer], other.outs[layer]
        bad = ~torch.isfinite(out.float()).flatten(1).all(1)
        assert not bool(bad.any()), f"layer {layer}: {int(bad.sum())} rows read an unheld byte"
        differs = (out.view(torch.int16) != alt.view(torch.int16)).flatten(1).any(1)
        rel = float((out.float() - alt.float()).norm() / alt.float().norm())
        unsplit = int((differs[: s.tc] & ~split).sum())
        assert unsplit == 0, f"layer {layer}: {unsplit} unsplit rows differ between paths"
        assert not bool(differs[s.tc :].any()), f"layer {layer}: generation rows differ"
        if bool(split.any()):
            _gate(out[: s.tc][split], alt[: s.tc][split])
        src = _OWNERS.index(_SOURCES[layer])
        sel = s.sel[rows]
        req = s.req[rows].long()
        key = sel.clamp_min(0).long()
        page = s.step["block_table"][req[:, None], key // _TPB].long()
        slots = torch.where(sel >= 0, (page * len(_OWNERS) + src) * _TPB + key % _TPB, -1)
        ref = flash_mla_sparse_decode.reference(
            s.q[layer][rows], s.step["latent_kv"], slots.int(), None, _SCALE, out[rows]
        )
        _gate(out[rows], ref)
        report[layer] = dict(rows_differ=int(differs.sum()), rel_l2_paths=f"{rel:.2e}")
    assert torch.equal(run.outs[39], run.outs[35]), "the sharer differs from its source"
    assert not torch.equal(run.outs[1], run.outs[35])
    return report


def _gate(out: torch.Tensor, ref: torch.Tensor) -> None:
    """The decode entry's band: relative L2 per (row, head) under 1e-2, the whole under 4e-3."""
    o, r = out.float(), ref.float()
    per = (o - r).norm(dim=-1) / r.norm(dim=-1).clamp_min(1e-30)
    whole = float((o - r).norm() / r.norm())
    assert float(per.max()) < 1e-2 and whole < 4e-3, (float(per.max()), whole)


def _path(run) -> str:
    return "workspace" if run.ctx.ws_buf is not None else "decode"


# ----------------------------------------------------------------------------- the steps


@pytest.fixture(scope="module", params=HEADS)
def held(request):
    """The production target's first context step, the KV-capacity estimate's: one fresh
    32768-token prompt (N = 32768, a query-split step with the indexer running), at the rank's
    heads of the world (16 at tep4, 8 at tep8). It takes the workspace path, allocates the held
    [2**19, 576] buffer and keeps layer 35's rows in it for layer 39 (N fits twice); the context
    attention's peak allocation over the step's three layers is recorded."""
    heads = request.param
    target = _target(heads)
    s = _step([(0, 32768)], heads=heads, seed=1)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    run = _run(target, s)
    peak = torch.cuda.max_memory_allocated() - base
    assert _path(run) == "workspace"
    buf = target._ws_buf
    assert buf.shape == (_ROWS_B, _LAT) and buf.dtype == torch.bfloat16
    assert run.ctx.ws_buf.data_ptr() == buf.data_ptr() and run.ctx.ws_buf.shape == (32768, _LAT)
    assert run.ctx.kept is not None and run.ctx.kept.shape == (32768, _LAT)
    other = _run(_target(heads, 0), s)
    assert _path(other) == "decode" and other.ctx.sub_batches == [
        (lo, lo + _SUB) for lo in range(0, 32768, _SUB)
    ]
    report = _check(s, run, other)
    assert report["split_rows"] == 0
    print(
        f"\n[estimate] {heads} heads, N 32768 workspace; peak {peak / _MIB:.0f} MiB (buffer "
        f"included); {report}"
    )
    del s, run, other
    torch.cuda.empty_cache()
    return SimpleNamespace(target=target, ptr=buf.data_ptr(), peak=peak, heads=heads)


def test_estimate_step_allocates_the_workspace(held):
    assert held.target._ws_buf.data_ptr() == held.ptr
    assert held.peak >= _ROWS_B * _LAT * 2


def test_switch_at_the_workspace_rows(held):
    """The same 32768-row chunk -- the same queries, selections and pool bytes -- once over
    491520 cached keys (N = 2**19: the workspace path, fitting the held rows once, so layer 39
    gathers layer 35's rows again) and once over 491521 (N = 2**19 + 1: the decode path in four
    8192-row sub-batches, none split): equal outputs, bitwise.

    Controls: the workspace path over the first 2**19 - 1 rows of the step's KV (the last row's
    own key dropped) and over its first 32768 (the step's new tokens, `num_ctx_kv_tokens`), with
    the held buffer filled with 0xFF first so that no read past the cut finds a row an earlier
    gather left there, runs without an error; the checks refuse both. One dropped key of 2048 can
    carry too little weight to move a bf16 output, so the workspace's rows are checked against
    the step's context KV as well as its output; the cut to the new tokens moves the output of
    every layer."""
    new = 32768
    cached = _ROWS_B - new
    at_b = _step([(cached, new)], heads=held.heads, seed=2, pool_kv=[_ROWS_B + 1])
    past_b = _Step(**vars(at_b))
    past_b.step = dict(at_b.step, **_offsets([(cached + 1, new)]))
    past_b.n_kv = _ROWS_B + 1
    assert at_b.n_kv == _ROWS_B
    ws = _run(held.target, at_b)
    dec = _run(held.target, past_b)
    assert (_path(ws), _path(dec)) == ("workspace", "decode")
    assert ws.ctx.ws_buf.shape == (_ROWS_B, _LAT) and ws.ctx.kept is None
    assert dec.ctx.sub_batches == [(lo, lo + _SUB) for lo in range(0, new, _SUB)]
    assert sorted(dec.ctx.sched) == [_SUB]
    report = _check(at_b, ws, dec)
    assert report["split_rows"] == 0
    print(f"\n[switch] N 2**19 workspace vs 2**19 + 1 decode: {report}")
    assert at_b.step["num_ctx_kv_tokens"] == new
    for cut in (_ROWS_B - 1, new):
        held.target._ws_buf.view(torch.int16).fill_(-1)
        short = _run(held.target, at_b, cut=cut)
        differ, nan = {}, {}
        for layer in _DRIVEN:
            out, full = short.outs[layer], ws.outs[layer]
            moved = (out.view(torch.int16) != full.view(torch.int16)).flatten(1).any(1)
            differ[layer] = int(moved.sum())
            nan[layer] = int((~torch.isfinite(out.float())).flatten(1).any(1).sum())
        print(f"[switch] control: workspace cut to {cut} rows; rows differing {differ}, NaN {nan}")
        if cut == new:
            assert all(differ.values())
        with pytest.raises(AssertionError, match="the workspace holds"):
            _check(at_b, short, dec)
    assert held.target._ws_buf.data_ptr() == held.ptr


def test_last_chunk_of_the_envelope_prompt(held):
    """The 1M gate's 1048559-token prompt's last chunk, 32751 rows over 1015808 cached keys,
    with three generation rows behind it: the decode path in sub-batches of 8192 x 3 and 8175 on
    two schedulers, the generation rows on the native decode. Against the workspace path over a buffer of
    all 1048559 rows. The context attention's peak stays under the estimate step's."""
    s = _step([(1015808, 32751)], gen=[300001, 2049, 17], heads=held.heads, seed=3)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    run = _run(held.target, s)
    peak = torch.cuda.max_memory_allocated() - base
    assert _path(run) == "decode" and s.n_kv == 1048559
    assert run.ctx.sub_batches == [(0, 8192), (8192, 16384), (16384, 24576), (24576, 32751)]
    assert sorted(run.ctx.sched) == [8175, 8192]
    other = _run(_target(held.heads, s.n_kv), s)
    assert _path(other) == "workspace"
    report = _check(s, run, other)
    print(
        f"\n[last chunk] N {s.n_kv} decode; peak {peak / _MIB:.0f} MiB against the estimate "
        f"step's {held.peak / _MIB:.0f}; {report}"
    )
    assert peak < held.peak
    assert held.target._ws_buf.data_ptr() == held.ptr


def test_packed_chunks_over_long_prefixes(held):
    """Four context requests packed in one step, chunks of 16384, 8192, 4096 and 4080 rows over
    1032192, 524288, 0 and 2048 cached keys (N = 1591280), twelve generation rows behind them:
    the decode path over the 32752 context rows, each request's rows through its own pages."""
    ctx = [(1032192, 16384), (524288, 8192), (0, 4096), (2048, 4080)]
    gen = [1048000, 262145, 131073, 65537, 16385, 9217, 4097, 2049, 2048, 1025, 33, 1]
    s = _step(ctx, gen=gen, heads=held.heads, seed=4)
    run = _run(held.target, s)
    assert _path(run) == "decode" and s.n_kv == 1591280 and s.tc == 32752
    assert run.ctx.sub_batches == [(0, 8192), (8192, 16384), (16384, 24576), (24576, 32752)]
    other = _run(_target(held.heads, s.n_kv), s)
    report = _check(s, run, other)
    print(f"\n[packed] N {s.n_kv} decode, {len(gen)} generation rows: {report}")
    assert held.target._ws_buf.data_ptr() == held.ptr


def test_short_tail_over_a_1m_prefix(held):
    """A 16-row tail over 1048544 cached keys: the decode path in one 16-row call. FlashMLA's
    decode splits each of so few rows over the keys and merges the parts in fp32, so it rounds
    otherwise than the prefill kernel's one pass: held to the band against the workspace path,
    and against the fp32 reference."""
    s = _step([(1048544, 16)], heads=held.heads, seed=5)
    run = _run(held.target, s)
    assert _path(run) == "decode" and run.ctx.sub_batches == [(0, 16)]
    other = _run(_target(held.heads, s.n_kv), s)
    report = _check(s, run, other)
    assert report["split_rows"] > 0
    print(f"\n[tail] N {s.n_kv} decode, 16 rows: {report}")
    assert held.target._ws_buf.data_ptr() == held.ptr


def test_dense_step_takes_the_workspace(held):
    """Nine 1000-token prompts, every one within index_topk, so the engine skips the context
    indexer and hands its causal index (0..p, -1 after): N = 9000, the workspace path, layer
    35's rows kept. Against the decode path: bitwise on its unsplit rows, the band on the rows
    its 808-row second sub-batch splits."""
    s = _step([(0, 1000)] * 9, heads=held.heads, seed=6, dense=True)
    run = _run(held.target, s)
    assert _path(run) == "workspace" and run.ctx.ws_buf.shape == (9000, _LAT)
    assert run.ctx.kept is not None
    assert run.ctx.ws_off.tolist() == [1000 * r for r in range(9)]
    other = _run(_target(held.heads, 0), s)
    assert other.ctx.sub_batches == [(0, 8192), (8192, 9000)]
    report = _check(s, run, other)
    print(f"\n[dense] N {s.n_kv} workspace: {report}")
    assert held.target._ws_buf.data_ptr() == held.ptr


def test_heads_past_the_rank_are_never_read(held):
    """Two context requests (4096 rows over 4096 cached keys, a fresh 2048-row prompt) with two
    generation rows behind them, on the workspace path and on the decode path. `lat_out` is the
    rank's heads (16 at tep4, 8 at tep8), every element of it written (the sentinel `_run` fills
    it with is gone), its gated rows within the band of the fp32 reference over the live heads
    alone (`_check`). The query's heads past them are zero, as the forward leaves them; run
    again with them NaN, every output row of every driven layer is bitwise the same: nothing the
    kernels compute on those heads reaches the rank's. With them N(0, 0.5) instead, FlashMLA
    rounds some live rows otherwise (printed; held to the band of the zero-head output): the
    kernels' arithmetic on the live heads sees the other heads' values, so the target keeps them
    zero, as every certified cell runs them."""
    heads = held.heads
    s = _step([(4096, 4096), (0, 2048)], gen=[300001, 17], heads=heads, seed=7)
    g = torch.Generator(device="cuda").manual_seed(8)
    fills = {
        "nan": torch.full((s.rows, _KERNEL_HEADS - heads, _LAT), float("nan"), device="cuda"),
        "randn": torch.randn(s.rows, _KERNEL_HEADS - heads, _LAT, generator=g, device="cuda") / 2,
    }
    for rows_b in (None, 0):
        target = held.target if rows_b is None else _target(heads, 0)
        base = _run(target, s)
        assert _path(base) == ("workspace" if rows_b is None else "decode")
        other = _run(_target(heads, s.n_kv if rows_b == 0 else 0), s)
        _check(s, base, other)
        for layer in _DRIVEN:
            assert not bool((s.q[layer][:, heads:] != 0).any())
            assert base.outs[layer].shape == (s.rows, heads, _NOPE)
        moved = {}
        for name, fill in fills.items():
            filled = _Step(**vars(s))
            filled.q = {}
            for layer, q in s.q.items():
                filled.q[layer] = q.clone()
                filled.q[layer][:, heads:] = fill.to(q.dtype)
            run = _run(target, filled)
            moved[name] = {}
            for layer in _DRIVEN:
                out, zero = run.outs[layer], base.outs[layer]
                assert bool(torch.isfinite(out.float()).all()), (name, layer)
                rows = (out.view(torch.int16) != zero.view(torch.int16)).flatten(1).any(1)
                moved[name][layer] = int(rows.sum())
                if name == "nan":
                    assert moved[name][layer] == 0, layer
                else:
                    _gate(out, zero)
        print(f"\n[heads] {_path(base)}: live rows moved by heads {heads}..63 of the query {moved}")
