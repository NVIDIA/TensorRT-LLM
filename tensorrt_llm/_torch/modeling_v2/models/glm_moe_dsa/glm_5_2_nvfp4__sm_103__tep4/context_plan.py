# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The prefill Target's per-step bookkeeping for the context rows' attention: what every
layer of a context step shares, built once per step on the host (the context step is
eager), and the route each layer's context rows take.

- `step_tables`: the pool addressing every layer reads, as python values (CTX-HOST-DIET);
  `pool_slot` one layer's.
- `rope_step`: the context rope kernel's per-step operands, or None (the stock op).
- `dense_causal_step`: the dense-causal context attention's plan (the non-absorbed MHA's,
  `mha_plan`, or the absorbed kernel's), or None (the index path).
- `layer_route`: which of the four context routes a layer takes on this step.
- `mtp_last_plan`: the MTP layer's selected rows on a context step, or None.
- `ctx_slots`: the selection's conversion to each layer's pool rows.

Nothing here launches an attention kernel; the ops in `ops/attention/ctx_*.py` do.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import NamedTuple

import torch

from tensorrt_llm._utils import prefer_pinned

from .ops.attention import cute_dsl_mha_kv_gather_mxf8 as _kv_gather8_op
from .ops.attention import cute_dsl_mla_rope_append_prefill as _ctx_rope_op
from .ops.attention import cute_dsl_sparse_mla_prefill as _fmha_pre_op
from .ops.attention.ctx_mha import CtxMhaHandle
from .ops.attention.ctx_rope import CtxRopeHandle
from .ops.attention.cute_dsl_req_index_to_base import cute_dsl_req_index_to_base
from .ops.attention.cute_dsl_req_index_to_global import cute_dsl_req_index_to_global
from .ops.attention.cute_dsl_sparse_mla_decode import (
    INDEX_KIND_BASE_WITH_LAYER_OFFSET,
    INDEX_KIND_GLOBAL,
)
from .ops.attention.sparse_mla import MTP1_WIDTH, SparseMlaHandle
from .ops.gemm.projection import DECODE_MAX_ROWS

#: The context routes (`layer_route`): the non-absorbed MHA on a dense-causal step, the
#: absorbed sparse MLA over each request's causal window, over the indexer's selection, or
#: over the MTP layer's selected rows only.
MHA = "mha"
DENSE_CAUSAL = "dense_causal"
INDEX = "index"
SELECTED = "selected"

#: A mixed step's generation rows' q_b reads at most the last 64 bf16 q_a rows (the q_b
#: tail), so with no indexer reading q_a the norm need not store the context rows below them
#: (`q8_rows`).
Q_A_TAIL_ROWS = DECODE_MAX_ROWS

#: The token count at which the context rope/append kernel's int32 element offsets wrap (16
#: heads x 576 per token): a step at or above it takes the stock op.
ROPE_APPEND_MAX_TOKENS = 2**31 // (16 * 576)

#: The row counts the MTP layer's selected rows are padded to: decode cells every op
#: downstream has (the routed MoE takes powers of two only; anything else it would pad with
#: torch ops). Above 64 the plan declines and every row is computed.
MTP_LAST_CELLS = (2, 4, 8, 16, 32, 64)

#: MTP3 / MTP5 only: 4 / 6 generation rows per request pass 64 selected rows from 17 / 11
#: generation requests on, so the plan also pads to these (prefill kernels downstream, the
#: routed MoE's pair at a power of two), while that is still fewer rows than the step's.
MTP_LAST_WIDE_CELLS = (128, 256)


class PoolSlot(NamedTuple):
    """One layer's pool addressing on a step: its pool, its layer within the pool, whether
    the pool's pointer row names the primary pool and no secondary pool (the context kernels
    read the primary pool only), the page size, and the context requests' block-table rows."""

    pool_idx: int
    layer_in_pool: int
    primary: bool
    tpb: int
    block_offsets: torch.Tensor


def step_tables(step: dict, diet: bool) -> None:
    """CTX-HOST-DIET (`diet`): the pool addressing every layer's context attention reads, as
    python values built once per step into `step` -- `diet_layers[layer] = (pool index, layer
    in pool, primary)` from one `tolist` of each host pool table, `diet_tpb`, the per-pool
    block-table rows filled on first use -- and the stream this forward runs on, read once
    (`torch.cuda.current_stream()` costs ~3 us per call). Off, `pool_slot` indexes the host
    tensors itself; the same values reach the same kernels."""
    if not diet:
        return
    mapping = step["host_kv_cache_pool_mapping"].tolist()
    ptrs = step["host_kv_cache_pool_pointers"].tolist()
    primary = step["primary_pool"].data_ptr()
    step["diet_layers"] = [
        (int(m[0]), int(m[1]), ptrs[int(m[0])][0] == primary and ptrs[int(m[0])][1] == 0)
        for m in mapping
    ]
    step["diet_tpb"] = int(step["tokens_per_block"])
    step["diet_block_offsets"] = {}
    step["diet_main"] = torch.cuda.current_stream()


def pool_slot(step: dict, layer: int) -> PoolSlot:
    """Pool layer `layer`'s addressing on this step (`PoolSlot`): from the step's tables where
    `step_tables` built them (the block-table rows sliced once per pool and shared by its
    layers, a view nothing writes), else from the host tensors."""
    diet = step.get("diet_layers")
    if diet is not None:
        pool_idx, layer_in_pool, primary = diet[layer]
        cache = step["diet_block_offsets"]
        rows = cache.get(pool_idx)
        if rows is None:
            rows = cache[pool_idx] = step["kv_cache_block_offsets"][
                pool_idx, : step["num_contexts"]
            ]
        return PoolSlot(pool_idx, layer_in_pool, primary, step["diet_tpb"], rows)
    pool_ptrs = step["host_kv_cache_pool_pointers"]
    pool_idx, layer_in_pool = (int(v) for v in step["host_kv_cache_pool_mapping"][layer][:2])
    primary = (
        int(pool_ptrs[pool_idx, 0]) == step["primary_pool"].data_ptr()
        and int(pool_ptrs[pool_idx, 1]) == 0
    )
    return PoolSlot(
        pool_idx,
        layer_in_pool,
        primary,
        int(step["tokens_per_block"]),
        step["kv_cache_block_offsets"][pool_idx, : step["num_contexts"]],
    )


def rope_step(step: dict, tc: int, rope: CtxRopeHandle) -> tuple | None:
    """The context rope kernel's per-step operands: the new-row prefix sum and the cached
    lengths, int32, from the engine's int64 prefix sums. None sends the step's context rows
    (none on a step without them) to the stock rope/append op: the kernel not built, a request
    count or page size it was not built for, or a token count its offsets would wrap at.

    Built on the host from the prefix sums' host copies and uploaded in one copy: the
    conversions and the difference on the device would be four small kernels per step.
    Context steps are not captured, so a fresh pair per step is fine."""
    ctx = step["num_contexts"]
    compiled = rope.compiled
    if (
        not tc
        or compiled is None
        or not 0 < ctx <= _ctx_rope_op.MAX_REQUESTS
        or int(step["tokens_per_block"]) != compiled.tokens_per_block
        or tc >= ROPE_APPEND_MAX_TOKENS
    ):
        return None
    host_cached = step["host_ctx_cached_token_indptr"]
    staged = torch.empty([2 * ctx + 1], dtype=torch.int32, pin_memory=prefer_pinned())
    staged[: ctx + 1].copy_(step["host_ctx_uncached_token_indptr"][: ctx + 1])
    staged[ctx + 1 :].copy_(host_cached[1 : ctx + 1] - host_cached[:ctx])
    # The caching host allocator keeps the pinned block until this copy has run.
    dev = step["ctx_uncached_token_indptr"].device
    both = staged.to(dev, non_blocking=True)
    return both[: ctx + 1], both[ctx + 1 :]


def _dc_bounds_nosync(seq_lens: torch.Tensor, cached, total_q: int):
    """The dense-causal per-row KV windows, `(ks, ke)` = `(0, cached + local + 1)`, on the
    device without a host sync: `total_q` is trusted, the caller having checked the host
    lengths sum to it. The kernel module's `dense_causal_bounds` reads the device sum back to
    validate it, a full drain of the device on every context step."""
    starts = torch.cumsum(seq_lens, 0) - seq_lens
    pos = torch.arange(
        total_q, dtype=torch.int32, device=seq_lens.device
    ) - torch.repeat_interleave(starts, seq_lens, output_size=total_q)
    if cached is not None:
        pos = pos + torch.repeat_interleave(cached, seq_lens, output_size=total_q)
    ks = torch.zeros(total_q, dtype=torch.int32, device=seq_lens.device)
    ke = (pos + 1).to(torch.int32).contiguous()
    return ks, ke


def _dc_groups_host(host_lens: torch.Tensor, g_tok: int) -> torch.Tensor:
    """The kernel module's `dense_causal_groups`, vectorized on the host: `[G, 2] = (q0,
    n_tok)`, request-major, `q0 = cu[r] + g_tok * j`, `n_tok = min(g_tok, len_r - g_tok * j)`;
    the module's version loops in Python over T / 4 groups."""
    lens = host_lens.to(torch.int64)
    if bool((lens < 0).any()):
        raise ValueError(f"seq_lens must be non-negative, got {host_lens.tolist()}")
    ng = (lens + g_tok - 1) // g_tok
    total = int(ng.sum())
    if total == 0:
        raise ValueError("seq_lens must describe at least one query row")
    req = torch.repeat_interleave(torch.arange(lens.numel()), ng, output_size=total)
    local = (torch.arange(total) - (torch.cumsum(ng, 0) - ng)[req]) * g_tok
    q0 = (torch.cumsum(lens, 0) - lens)[req] + local
    ntok = torch.clamp(lens[req] - local, max=g_tok)
    return torch.stack((q0, ntok), 1).to(torch.int32).contiguous()


class DcGroupsUploader:
    """The per-step dense-causal group table's upload without a host sync: pinned,
    double-buffered host staging and one persistent device buffer. A pinned slot is rewritten
    only after its previous upload completed (its event, two steps old in practice), and every
    reader runs on the stream that uploaded, after the upload, so reusing the device buffer
    next step is stream-ordered. Grows, with one sync, the rare time a step needs more rows. A
    pageable `.to(device)` drains the device instead. Owned by the prefill Target (state kept
    across its steps); allocates nothing until the first upload."""

    def __init__(self) -> None:
        self._cap = 0
        self._host: list = []
        self._done: list = [None, None]
        self._buf = None
        self._slot = 0

    def _grow(self, rows: int, dev) -> None:
        torch.cuda.synchronize(dev)
        self._cap = max(rows, 2 * self._cap, 2048 + 64)
        self._host = [
            torch.empty((self._cap, 2), dtype=torch.int32, pin_memory=prefer_pinned())
            for _ in range(2)
        ]
        self._done = [None, None]
        self._buf = torch.empty((self._cap, 2), dtype=torch.int32, device=dev)

    def __call__(self, groups_host: torch.Tensor, dev) -> torch.Tensor:
        g = int(groups_host.shape[0])
        if g > self._cap:
            self._grow(g, dev)
        s = self._slot
        self._slot ^= 1
        if self._done[s] is not None:
            self._done[s].synchronize()
        self._host[s][:g].copy_(groups_host)
        out = self._buf[:g]
        out.copy_(self._host[s][:g], non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        self._done[s] = ev
        return out


def dense_causal_step(
    md,
    step: dict,
    tc: int,
    skip_ctx: bool,
    req_idx,
    dev,
    rope_ok: bool,
    mha: CtxMhaHandle,
    uploader: DcGroupsUploader,
    stats,
):
    """The dense-causal context attention's per-step operands, shared by every layer, or None
    when the context rows take the index path (counted by reason in
    `stats.dense_causal_misses`; the steps taken in `stats.dense_causal_steps`).

    Taken only when the engine skipped the indexer for the context rows: every context
    sequence is within index_topk, so the dense causal index it substituted is the whole
    window and the kernel's per-token mask reproduces it. The windows are rebuilt here rather
    than read from the metadata: the engine's `cu_seqlen_ks` are batch-flat KV offsets, while
    this kernel reads each request through its own block-table row from position 0, cached
    prefix included.

    With the non-absorbed context attention built and the step on the rope kernel
    (`rope_ok`), the trunk layers take that instead: `mha` is its plan (`mha_plan`), and the
    absorbed kernel's operands are built only if a layer falls back to it (`layer_route`)."""
    if not tc:
        return None
    reason = (
        "selection"
        if not skip_ctx
        else "block"
        if int(step["tokens_per_block"]) != _fmha_pre_op.DENSE_CAUSAL_BLOCK
        else None
    )
    if reason is not None:
        stats.dense_causal_misses[reason] = stats.dense_causal_misses.get(reason, 0) + 1
        return None
    ctx = step["num_contexts"]
    host_lens = step["seq_lens"][:ctx].to(torch.int32).cpu()
    assert int(host_lens.sum()) == tc, (host_lens.tolist(), tc)
    stats.dense_causal_steps += 1

    def absorbed():
        return _dense_causal_absorbed(md, step, tc, host_lens, req_idx, dev, mha, uploader)

    if rope_ok and mha.expand:
        return SimpleNamespace(
            mha=mha_plan(md, step, tc, host_lens, req_idx, dev, mha),
            absorbed=None,
            rebuild=absorbed,
        )
    return absorbed()


def _dense_causal_absorbed(md, step, tc: int, host_lens, req_idx, dev, mha: CtxMhaHandle, uploader):
    """The absorbed dense-causal kernel's per-step operands (`dense_causal_step`)."""
    ctx = step["num_contexts"]
    cached = None
    if int(step["num_ctx_cached_tokens"]):
        cached = torch.diff(step["ctx_cached_token_indptr"][: ctx + 1]).to(torch.int32)
    # Both without a host sync (the kernel module's helpers read the device lengths back and
    # upload pageable memory): the host already checked the row total above.
    ks, ke = _dc_bounds_nosync(step["seq_lens_cuda"][:ctx].to(torch.int32), cached, tc)
    groups = uploader(_dc_groups_host(host_lens, _fmha_pre_op.DENSE_CAUSAL_GROUP), dev)
    return SimpleNamespace(
        ks=ks,
        ke=ke,
        groups=groups,
        req_idx=req_idx[:tc],
        block_table=md.block_table,
        # One fp8 query for the step: each layer writes it and its attention reads it before
        # the next layer writes (one stream), so the bound address holds.
        quant_q=torch.empty([tc, mha.heads, mha.lat_dim], dtype=torch.float8_e4m3fn, device=dev),
        # Bound on the first layer; later layers re-bind output and layer offset.
        prepared=None,
        mha=None,
    )


def mha_plan(md, step, tc: int, host_lens, req_idx, dev, mha: CtxMhaHandle) -> SimpleNamespace:
    """The non-absorbed context attention's per-step operands, shared by every trunk layer: the
    K / V scratch (each layer's expansion writes it and its MHA reads it before the next
    layer's expansion, on one stream) and, when the step's requests have a cached prefix (KV
    block reuse), the K/V source-row plan.

    Request r's K/V source rows are its `c_r` cached positions followed by its `n_r` new rows,
    packed at `cu_k[r]`. The cached rows are the pool rows the absorbed dense-causal kernel
    reads for those positions, gathered per layer (`ctx_mha.mha_kv_gather`) from the index
    vectors built here on the host from the two host prefix sums and uploaded once (the
    context step is eager)."""
    ctx = step["num_contexts"]
    cached = int(step["num_ctx_cached_tokens"])
    tk = tc + cached
    rows = mha.heads * mha.qk_dim
    v_rows = mha.heads * mha.v_dim
    kv_dt = torch.float8_e4m3fn if mha.fp8 else torch.bfloat16
    kvb8 = mha.gather8 is not None
    # The one-launch expansion writes `[K | V]` rows; K and V are their halves.
    kv = None
    if kvb8 and mha.merged is not None:
        kv = torch.empty([tk, rows + v_rows], dtype=kv_dt, device=dev)
    plan = SimpleNamespace(
        max_len=int(host_lens.max()),
        tk=tk,
        k=torch.empty([tk, rows], dtype=kv_dt, device=dev) if kv is None else kv[:, :rows],
        v=torch.empty([tk, v_rows], dtype=kv_dt, device=dev) if kv is None else kv[:, rows:],
        kv=kv,
        cached=None,
        # The MXF8 expansion's A operands, written per layer by the gather.
        a8=_kv_gather8_op.buffers(tk, dev) if kvb8 else None,
    )
    if not cached:
        return plan
    host_cu = step["host_ctx_uncached_token_indptr"][: ctx + 1].to(torch.int64)
    host_ci = step["host_ctx_cached_token_indptr"][: ctx + 1].to(torch.int64)
    new_l, cached_l = host_cu[1:] - host_cu[:-1], host_ci[1:] - host_ci[:-1]
    assert int(new_l.sum()) == tc and int(cached_l.sum()) == cached, (
        new_l.tolist(),
        cached_l.tolist(),
    )
    cu_k = torch.zeros(ctx + 1, dtype=torch.int64)
    cu_k[1:] = torch.cumsum(cached_l + new_l, 0)
    reqs = torch.arange(ctx, dtype=torch.int64)
    req_c = torch.repeat_interleave(reqs, cached_l)
    pos_c = torch.arange(cached, dtype=torch.int64) - (host_ci[req_c] - host_ci[0])
    req_n = torch.repeat_interleave(reqs, new_l)
    tok_n = torch.arange(tc, dtype=torch.int64) - (host_cu[req_n] - host_cu[0])
    staged = torch.empty([3 * cached + tc], dtype=torch.int64, pin_memory=prefer_pinned())
    staged[:cached].copy_(cu_k[req_c] + pos_c)  # dst_c
    staged[cached : 2 * cached].copy_(host_cu[req_c] - host_cu[0])  # tok_c
    staged[2 * cached : 3 * cached].copy_(pos_c)  # pos_c
    staged[3 * cached :].copy_(cu_k[req_n] + cached_l[req_n] + tok_n)  # dst_n
    staged_cu = torch.empty([ctx + 1], dtype=torch.int32, pin_memory=prefer_pinned())
    staged_cu.copy_(cu_k)
    # The caching host allocator keeps the pinned blocks until these copies have run.
    plan.cached = SimpleNamespace(
        rows=cached,
        idx=staged.to(dev, non_blocking=True),
        cu_k=staged_cu.to(dev, non_blocking=True),
        req_idx=req_idx[:tc],
        block_table=md.block_table,
        pool_rows=step["primary_pool"].view(torch.uint8).reshape(-1, mha.lat_dim),
        # The bf16 [cached | new] rows of the bf16-MMA expansion (the MXF8 one gathers into
        # `a8` instead).
        lat=None if kvb8 else torch.empty([tk, mha.lat_dim], dtype=torch.bfloat16, device=dev),
    )
    return plan


def layer_route(dc, last, slot: PoolSlot | None, ctx_rope, w, mha: CtxMhaHandle):
    """`(route, plan)`: the route a layer's context rows take on this step and its per-step
    operands. `SELECTED` with the MTP layer's plan `last`; `INDEX` without a dense-causal plan
    `dc`; `MHA` where the step holds the non-absorbed plan and the layer can take it (its
    expansion weights `w.kvb8` / `w.kvb_aug` built, the step on the rope kernel, the layer's
    pool the primary pool); else `DENSE_CAUSAL` -- a layer declining the MHA counted
    `ctx_mha` on the stock side, the absorbed operands then built once for the step."""
    if last is not None:
        return SELECTED, last
    if dc is None:
        return INDEX, None
    if dc.mha is None:
        return DENSE_CAUSAL, dc
    if (w.kvb8 if mha.fp8 else w.kvb_aug) is not None and ctx_rope is not None and slot.primary:
        return MHA, dc.mha
    mha.count("ctx_mha", False)
    if dc.absorbed is None:
        dc.absorbed = dc.rebuild()
    return DENSE_CAUSAL, dc.absorbed


def q8_rows(qa8: bool, tc: int, need_q: bool) -> tuple[int, int]:
    """`(q8_rows, q_store_from)` for `qkv_a_rmsnorm`: where the q_a norm writes the context
    rows' q_a as the MXF8 q_b's operand (`qa8`: the fp8 MHA route with the MXF8 `q_b_e4m3`),
    and, with no indexer reading q_a this step, the only bf16 q_a reader the generation rows'
    q_b tail, the norm stores only the last `Q_A_TAIL_ROWS` context rows' bf16 q_a."""
    q8 = tc if qa8 else 0
    store_from = tc - Q_A_TAIL_ROWS if q8 and not need_q and tc > Q_A_TAIL_ROWS else 0
    return q8, store_from


def mtp_last_plan(
    md,
    step: dict,
    ctx_rope,
    ctx: int,
    tc: int,
    gen: int,
    layer: int,
    stride: int,
    rows_select_built: bool,
    max_draft_len: int,
    idx_topk: int,
    count,
    declines: dict,
):
    """The MTP layer's selected rows on a context step, or None (every row computed: no rope
    kernel on the step's context rows, no row select built, or a reason counted in
    `declines`).

    The draft head reads the layer's output at each context request's last row and at the
    generation rows only, and everything after the layer's KV write is row-local, so
    attention-to-output can run on those rows: the context last rows (padded with the last
    one up to a cell), then the generation rows. The context queries take the dense-causal
    kernel (`ks = 0`, `ke = cached + len`, request `b`, one `(q, 1)` group each), which needs
    the engine's dense index (no real selection) and the rope kernel's append. Host only,
    sync-free: the metadata's host prefix sums and one pinned upload. On B300 the MTP layer's
    context attention at 8.9k rows goes 590 -> 30 us."""
    if ctx_rope is None or not rows_select_built:
        return None
    pool_ptrs = step["host_kv_cache_pool_pointers"]
    mapping = step["host_kv_cache_pool_mapping"][layer]
    cells = MTP_LAST_CELLS
    if max_draft_len > 1:
        cells = cells + MTP_LAST_WIDE_CELLS
    r = ctx + gen
    r_pad = next((m for m in cells if m >= r), None)
    # `ctx_rope` is None without the rope kernel, so the pool check covers the rest.
    why = (
        "selection"
        if not bool(md.skip_indexer_for_ctx_reqs)
        else "block"
        if int(step["tokens_per_block"]) != _fmha_pre_op.DENSE_CAUSAL_BLOCK
        else "pool"
        if int(pool_ptrs[int(mapping[0]), 0]) != step["primary_pool"].data_ptr()
        or int(pool_ptrs[int(mapping[0]), 1]) != 0
        else "rows"
        if r_pad is None or (r_pad > MTP_LAST_CELLS[-1] and r_pad >= tc + gen)
        else None
    )
    host_cu = step["host_ctx_uncached_token_indptr"][: ctx + 1].to(torch.int64)
    host_ci = step["host_ctx_cached_token_indptr"][: ctx + 1].to(torch.int64)
    lens = host_cu[1:] - host_cu[:-1]
    if why is None and (int(host_cu[0]) != 0 or int(host_cu[ctx]) != tc or bool((lens < 1).any())):
        why = "lens"
    ke = host_ci[1:] - host_ci[:-1] + lens
    if why is None and int(ke.max()) > idx_topk:
        why = "window"
    if why is not None:
        # Not a stock-op fallback: every row then runs the same kernels. Counted apart from
        # the route counters, by reason.
        declines[why] = declines.get(why, 0) + 1
        return None
    count("mtp_ctx_last", True)
    if r_pad > MTP_LAST_CELLS[-1]:
        count("mtp_ctx_last_wide", True)
    n_pad = r_pad - gen
    last = (host_cu[1:] - 1).to(torch.int32)
    # One pinned block: rows [r_pad] | ke [n_pad] | req [n_pad] | groups [n_pad, 2] | ks [n_pad]
    staged = torch.empty([r_pad + 5 * n_pad], dtype=torch.int32, pin_memory=prefer_pinned())
    staged[:ctx] = last
    staged[ctx:n_pad] = int(last[-1])
    staged[n_pad:r_pad] = torch.arange(tc, tc + gen, dtype=torch.int32)
    o = r_pad
    staged[o : o + ctx] = ke.to(torch.int32)
    staged[o + ctx : o + n_pad] = int(ke[-1])
    o += n_pad
    staged[o : o + ctx] = torch.arange(ctx, dtype=torch.int32)
    staged[o + ctx : o + n_pad] = ctx - 1
    o += n_pad
    groups = staged[o : o + 2 * n_pad].view(n_pad, 2)
    groups[:, 0] = torch.arange(n_pad, dtype=torch.int32)
    groups[:, 1] = 1
    o += 2 * n_pad
    staged[o : o + n_pad] = 0
    dev = step["primary_pool"].device
    # The caching host allocator keeps the pinned block until this copy has run.
    buf = staged.to(dev, non_blocking=True)
    o = r_pad
    return SimpleNamespace(
        n_pad=n_pad,
        r_pad=r_pad,
        rows=buf[:r_pad],
        ctx_rows=buf[:n_pad],
        ke=buf[o : o + n_pad],
        req=buf[o + n_pad : o + 2 * n_pad],
        groups=buf[o + 2 * n_pad : o + 4 * n_pad].view(n_pad, 2),
        ks=buf[o + 4 * n_pad : o + 5 * n_pad],
        block_table=md.block_table,
        stride=int(stride),
        layer_offset=int(mapping[1]) * int(step["tokens_per_block"]),
    )


class CtxSlots:
    """One context or mixed step's selection as each layer's attention reads it (`ctx_slots`):
    `layer(buf, fresh, layer_id)` maps the request-local selection `buf` to `(context rows,
    generation rows, generation index kind, its layer)` for the attention of pool layer
    `layer_id` (`fresh`: the layer selected anew); the context half is None under the
    dense-causal context attention. Holds the step's base rows between layers."""

    def __init__(self, md, req_idx, rows, tc, gen, slot_lo, mixed_base, fmha) -> None:
        self.md, self.req_idx = md, req_idx
        self.rows, self.tc, self.gen = rows, tc, gen
        self.slot_lo, self.mixed_base = slot_lo, mixed_base
        self.fmha = fmha
        self.base = None

    def layer(self, buf, fresh: bool, layer_id: int):
        md, tc, h = self.md, self.tc, self.fmha
        if self.slot_lo == self.rows:
            return None, None, INDEX_KIND_GLOBAL, None
        if self.mixed_base:
            if fresh or self.base is None:
                self.base = cute_dsl_req_index_to_base(
                    self.req_idx[tc:], md.block_table, buf[tc:], md.tokens_per_block,
                    h.stride_factor, compiled=h.req_base,
                )  # fmt: skip
                h.count("mixed_base", True)
            return None, self.base, INDEX_KIND_BASE_WITH_LAYER_OFFSET, layer_id
        lo = self.slot_lo
        slots = cute_dsl_req_index_to_global(
            self.req_idx[lo:], md.block_table, buf[lo:], md.tokens_per_block,
            h.stride_factor, layer_id, compiled=h.req_global,
        )  # fmt: skip
        ctx_s, gen_s = (None, slots) if lo else torch.split(slots, [tc, self.gen], 0)
        return ctx_s, gen_s, INDEX_KIND_GLOBAL, None


def ctx_slots(
    md, req_idx, rows: int, tc: int, gen: int, gen_p: int, dc, mixed_base: bool,
    fmha: SparseMlaHandle,
) -> CtxSlots:  # fmt: skip
    """A context or mixed step's `CtxSlots`. Under the dense-causal context attention (`dc`)
    the context rows' pool rows are formed inside the attention kernel, so the conversion runs
    on the generation rows alone, or not at all. TRTLLM_GLM52_MIXED_BASE (`mixed_base`,
    default on): with only the generation rows converting at the MTP width, they take the
    layer-independent base rows, converted once per fresh selection, and the sparse MLA decode
    adds each layer's offset, as on a pure decode step: the same pool rows, instead of one
    global conversion per layer (78 launches per mixed step)."""
    slot_lo = tc if dc is not None else 0
    base = mixed_base and slot_lo == tc and gen > 0 and gen_p == MTP1_WIDTH
    return CtxSlots(md, req_idx, rows, tc, gen, slot_lo, base, fmha)
