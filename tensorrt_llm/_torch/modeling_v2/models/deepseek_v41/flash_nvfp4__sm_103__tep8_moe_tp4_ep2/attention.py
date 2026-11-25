# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 attention of deepseek-v4.1-flash, assembled from catalog entries.

Semantics are the checkpoint reference's `Attention` / `Compressor` / `Indexer` (HF `model.py`); the state
lives in the engine's `CSA2CacheManager` pools (the allocator), in the row formats of `csa2_pack_rows`:

* SWA pool, per layer: the last 128 rotated, fp8-quantized `kv` rows;
* GLOBAL / INDEX pools, per KV-source layer: compressed `kv` rows (fp4) and index-key rows (fp4);
* COMPRESSOR pools, per ratio-2 source layer: the fp32 value / gate of the last two positions.

Every step is graph-safe: shapes are static, positions are device data, page tables are the manager's
device tables (rewritten before each replay).
"""

import collections
import contextlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from tensorrt_llm._torch.attention.backends.sparse.params import get_indexer_mqa_logits_elem_budget
from tensorrt_llm._torch.modeling_v2.catalog.comm.allgather import allgather as cat_allgather
from tensorrt_llm._torch.modeling_v2.catalog.comm.reducescatter import (
    reducescatter as cat_reducescatter,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_bf16_lowm_proj import (
    csa2_bf16_lowm_proj,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_compressor import (
    csa2_compressor_project,
    csa2_owner_slots,
    owner_slot_descriptors,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_consumer_schedule import (
    csa2_consumer_schedule,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_indexer_candidates import (
    consumer_select_fused,
    prefetch_candidates,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_indexer_decode import (
    csa2_indexer_topk_decode,
    csa2_indexer_topk_prefill,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_indexer_descriptors import (
    csa2_indexer_descriptors,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_indexer_packed import (
    csa2_indexer_plan,
    csa2_indexer_topk_packed,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_mxfp8_linear import (
    csa2_mxfp8_epilogue_rope,
    csa2_mxfp8_linear,
    mxfp8_swizzled_scales,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_o_projection import (
    csa2_o_a_mxfp8_scales,
    csa2_o_projection,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_sparse_decode import (
    csa2_direct_decode,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_staged_context import (
    csa2_main_mapping,
    csa2_staged_context,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_store_rows import (
    csa2_stock_rope_table,
    csa2_store_rows,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_swa_slots import (
    csa2_swa_slots,
)
from tensorrt_llm._torch.modules.multi_stream_utils import do_multi_stream
from tensorrt_llm._utils import maybe_pin_memory

from ..swa_tables import swa_table_static
from .ops.attention.csa2_logits_schedule import csa2_logits_schedule
from .ops.attention.csa2_q_norm import csa2_q_norm

# `(role names)` of the manager's pools this module reads; spelled as in `CSA2CacheRole`.
SWA, GLOBAL, INDEX, COMP_KV, COMP_SCORE = (
    "SWA",
    "GLOBAL",
    "INDEX",
    "COMPRESSOR_KV",
    "COMPRESSOR_SCORE",
)

_WINDOW_ROWS = 128
_DIRECT_MAX_ROWS = 16  # rows the engine's direct decode serves in one launch
_LOGGED = set()  # attention paths already reported (once per process, never on the hot path beyond one set lookup)


def _note(m, ctx, attn: str, store: str):
    role = (
        "swa"
        if not m.ratio
        else f"ratio{m.ratio}-"
        + (
            "cand-source"
            if m.is_cand_source
            else "index-owner"
            if m.is_owner
            else "cand-consumer"
            if m.uses_cands
            else "index-source"
            if m.is_index_source
            else "reuse"
        )
    )
    key = (role, m.hl, ctx.paged, attn, store)
    if key not in _LOGGED:
        _LOGGED.add(key)
        idx = (
            "-"
            if not (m.ratio and m.is_index_source)
            else (
                ("decode" if ctx.paged else "prefill-tiled" if m.uses_cands else "prefill-packed")
                + ("+candidates" if m.uses_cands or m.is_cand_source else "")
            )
        )
        from tensorrt_llm.logger import logger

        logger.info(
            f"V41 attention path: layer={role} heads={m.hl} step={'gen' if ctx.paged else 'ctx'} "
            f"store={store} attn={attn} indexer={idx}"
        )


@dataclass
class StepCtx:
    """Per-step data every layer shares (built once by the core's forward)."""

    pos: torch.Tensor  # [n] int32 absolute positions
    rid: torch.Tensor  # [n] int64 request row of each token
    cs: torch.Tensor  # [n] first position of the token's request in this step
    ce: torch.Tensor  # [n] one past its last position
    segs: List[Tuple[int, int, int, int]]  # (t0, t1, row, span) per request
    kvm: object  # the CSA2CacheManager
    rope: Dict[int, torch.Tensor]  # 0: base-theta table, 1: compressed-KV (YaRN) table
    nreq: int  # table rows (requests in the batch, padding included)
    topk: Optional[torch.Tensor] = None  # latest index source's picks
    cands: Optional[object] = (
        None  # candidate-source layer's block masks (per request list; one tensor if `paged`)
    )
    paged: bool = (
        False  # generation step: device-side lengths (CUDA-graph safe), paged indexer kernel
    )
    uniform: bool = False  # every request has the same number of consecutive rows (a decode step)
    rows: Optional[Tuple[int, ...]] = (
        None  # a generation step of mixed lengths (eager): each request's rows, host side
    )
    cache: dict = field(default_factory=dict)  # pool / table lookups made this step
    swa: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = (
        None  # generation: (reads, writes, visible) of every layer
    )
    # the KV owner's compressed rows of this step are already in the pools (bounded-replay boundary)
    published: bool = False
    floor: Optional[torch.Tensor] = (
        None  # int64 [nreq] first window position each request's rows may read (bounded replay)
    )
    # [nreq] int64: an unconsumed Encoder replay's Global end per request (0 without one), context steps only. The
    # request's rows below it are historical: they rebuild the Encoder working state but publish no compressed rows
    hist_end: Optional[torch.Tensor] = None


def _unpublished_history(
    addr: torch.Tensor,
    first: torch.Tensor,
    cu_out: torch.Tensor,
    hist_end: torch.Tensor,
    ratio: int,
) -> torch.Tensor:
    """`addr` (an owner's GLOBAL write slot per compressed output) with -1 for the outputs an Encoder replay must not
    publish: request `r`'s outputs (`cu_out[r]:cu_out[r + 1]`) whose group `[first, first + ratio)` ends at or before
    its Global end `hist_end[r]`. Those rows are in the cache already (the replay's Global claim); stock's encoder
    replay compresses each owner from `hist_end - hist_end % ratio` on (`_prepare_owner_tables`), which is the same
    set of groups."""
    out = torch.arange(addr.shape[0], device=addr.device)
    req = torch.searchsorted(cu_out[1:].long(), out, right=True).clamp_max(hist_end.shape[0] - 1)
    return torch.where(first.long() + ratio > hist_end.long()[req], addr, -1)


def _swa_plan(ctx: StepCtx, ratios) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generation steps: the window slots of every layer, resolved once per step in one launch
    (`reads [layers, n, 128]` rows each query attends, `writes [layers, n]` its own row, `visible [layers, n]`)."""
    if ctx.swa is None:
        tables = ctx.kvm.swa_tables
        desc, ratio_t, floors = swa_table_static(tables, ratios)
        ctx.swa = csa2_swa_slots(
            ctx.pos, ctx.rid, tables, desc, ratio_t, floors[: ctx.nreq], _WINDOW_ROWS, _WINDOW_ROWS
        )
    return ctx.swa


def _swa_context_slots(ctx: StepCtx, layer: int, ratios) -> Tuple[torch.Tensor, torch.Tensor]:
    """Context chunks attended as stock does (`csa2_staged_context`): the window rows every query reads `[n, 128]` and
    the pool row each query row is published at `[n]` in `layer`
    (stock's `csa2_swa_indices` / `csa2_swa_write_slots`)."""
    tables = ctx.kvm.swa_tables
    desc, ratio_t, floors = swa_table_static(tables, ratios)
    sl = slice(layer, layer + 1)
    floors = (
        floors[: ctx.nreq] if ctx.floor is None else ctx.floor
    )  # (the bounded replay's floors from its boundary on)
    reads, writes, _ = csa2_swa_slots(
        ctx.pos, ctx.rid, tables[sl], desc[sl], ratio_t[sl], floors, _WINDOW_ROWS, _WINDOW_ROWS
    )
    return reads[0], writes[0]


def _rows_per_request(ctx: StepCtx) -> int:
    """Generation steps: the rows every request has (a token and its drafts); 1 where they differ or not decode."""
    n = ctx.pos.shape[0]
    return n // ctx.nreq if ctx.paged and ctx.uniform and ctx.nreq and n % ctx.nreq == 0 else 1


# The owners' group buffers of generation steps and their descriptors live on the engine's cache manager
# (`_OWNER_PLAN` names the attribute), so they are released with it: the KV-cache estimation pass's manager, its page
# tables and the buffers built for it are not kept past its replacement. A uniform step (what the CUDA graphs capture)
# keeps its buffers per step shape, at the addresses its graph reads (bounded by the batch sizes times the rows per
# request); eager steps of mixed lengths reuse at most `_MIXED_PLANS` shapes, the least recently used first out.
_OWNER_PLAN = "_modeling_v2_v41_owner_plans"
_MIXED_PLANS = 8


def _owner_plans(ctx: StepCtx) -> Tuple[dict, "collections.OrderedDict"]:
    """(uniform-step plans, mixed-length plans) of the step's engine cache manager."""
    kvm = getattr(ctx.kvm, "kvm", ctx.kvm)
    plans = getattr(kvm, _OWNER_PLAN, None)
    if plans is None:
        plans = ({}, collections.OrderedDict())
        setattr(kvm, _OWNER_PLAN, plans)
    return plans


def _owner_plan(ctx: StepCtx, sources, ratios) -> dict:
    """Every KV owner's compressed groups of the step (GLOBAL write slots, first positions and, at ratio 2, the
    compression batch), once per step in one launch, as the engine's metadata resolves them (`_resolve_owner_slots`)
    before the layers run. Generation steps keep their buffers per step shape; context chunks (any rows per request,
    `segs`) build them for the step."""
    plan = ctx.cache.get("owners")
    if plan is not None:
        return plan
    if not ctx.paged:
        plan = ctx.cache["owners"] = _context_owner_plan(ctx, sources, ratios)
        return plan
    n, nreq, rows = ctx.pos.shape[0], ctx.nreq, ctx.rows
    assert nreq and (rows is not None or n % nreq == 0), (
        "a generation step: every request has the same rows"
    )
    next_n, dev = (max(rows) if rows is not None else n // nreq), ctx.pos.device
    tables = [_table(ctx, o, GLOBAL) for o in sources]
    key = (
        dev.index,
        n,
        nreq,
        rows is None,
        tuple((t.data_ptr(), t.stride(0), t.shape[1]) for t in tables),
    )
    uniform_plans, mixed_plans = _owner_plans(ctx)
    plans = uniform_plans if rows is None else mixed_plans
    plan = plans.get(key)
    if plan is None:  # first use is an eager step (the engine warms up before capturing graphs)
        assert not torch.cuda.is_current_stream_capturing(), (
            "the owners' group buffers are built by an eager step"
        )
        i32 = dict(dtype=torch.int32, device=dev)
        owners = {}
        for o, t in zip(sources, tables):
            r = ratios[o]
            batch = (
                (torch.empty(nreq, **i32), torch.empty(nreq, **i32), torch.empty(nreq + 1, **i32))
                if r == 2
                else (None,) * 3
            )
            owners[o] = (
                t,
                torch.empty(n, dtype=torch.int64, device=dev),
                torch.empty(n, **i32),
                r,
                _WINDOW_ROWS // r,
                *batch,
            )
        lengths, base = torch.full((nreq,), next_n, **i32), torch.zeros(nreq, **i32)
        plan = plans[key] = dict(
            lengths=lengths,
            base=base,
            ends=torch.empty(nreq, **i32),
            cu_seq=torch.arange(0, n + 1, next_n, **i32)
            if rows is None
            else torch.empty(nreq + 1, **i32),
            owners=owners,
            desc=owner_slot_descriptors(lengths, base, list(owners.values())),
            max_outputs=-(-next_n // 2),
        )
    if (
        rows is not None
    ):  # an eager step of mixed lengths: this step's rows per request (the engine's per-request lengths)
        assert not torch.cuda.is_current_stream_capturing(), (
            "a generation step of mixed lengths runs eagerly"
        )
        mixed_plans.move_to_end(key)
        while (
            len(mixed_plans) > _MIXED_PLANS
        ):  # (an evicted plan's last use is on this stream, before this step)
            mixed_plans.popitem(last=False)
        host = maybe_pin_memory(torch.tensor([0, *rows], dtype=torch.int32))
        host[1:] = host[1:].cumsum(0)
        plan["cu_seq"].copy_(host, non_blocking=True)
        plan["lengths"].copy_(host[1:] - host[:-1], non_blocking=True)
        plan["max_outputs"] = -(-next_n // 2)
        plan["ends"].copy_(ctx.ce.index_select(0, plan["cu_seq"][:-1].long()))
    else:
        plan["ends"].copy_(ctx.ce.view(nreq, next_n)[:, 0])
    csa2_owner_slots(
        plan["ends"], plan["lengths"], plan["base"], list(plan["owners"].values()), plan["desc"]
    )
    ctx.cache["owners"] = plan
    return plan


def _context_owner_plan(ctx: StepCtx, sources, ratios) -> dict:
    """`_owner_plan` of a context chunk: request `r` (`segs[r]`) has `t1 - t0` rows ending at position `span`, as the
    engine's prepare uploads its query lengths and KV ends; its compression batch has as many output rows as the step
    has rows (the engine's `capacity`). (The dep target's form.)"""
    n, dev = ctx.pos.shape[0], ctx.pos.device
    nreq = len(ctx.segs)
    i32 = dict(dtype=torch.int32, device=dev)
    rows = [t1 - t0 for t0, t1, _, _ in ctx.segs]
    assert [r for _, _, r, _ in ctx.segs] == list(range(nreq)) and sum(rows) == n, (
        "requests in table order, every row"
    )
    cu = [0]
    for r in rows:
        cu.append(cu[-1] + r)
    # the lengths, ends and cumulative rows in one pinned, asynchronous upload (no pageable copy in the forward)
    host = maybe_pin_memory(
        torch.tensor([*rows, *(span for _, _, _, span in ctx.segs), *cu], dtype=torch.int32)
    )
    packed = host.to(dev, non_blocking=True)
    lengths, ends, cu_seq = packed[:nreq], packed[nreq : 2 * nreq], packed[2 * nreq :]
    base = torch.zeros(nreq, **i32)
    owners = {}
    for o in sources:
        r = ratios[o]
        batch = (
            (torch.empty(nreq, **i32), torch.empty(nreq, **i32), torch.empty(nreq + 1, **i32))
            if r == 2
            else (None,) * 3
        )
        owners[o] = (
            _table(ctx, o, GLOBAL),
            torch.empty(n, dtype=torch.int64, device=dev),
            torch.empty(n, **i32),
            r,
            _WINDOW_ROWS // r,
            *batch,
        )
    plan = dict(
        lengths=lengths,
        base=base,
        ends=ends,
        cu_seq=cu_seq,
        owners=owners,
        desc=owner_slot_descriptors(lengths, base, list(owners.values())),
        max_outputs=max(1, -(-max(rows) // 2)),
    )
    csa2_owner_slots(ends, lengths, base, list(owners.values()), plan["desc"])
    return plan


def _per_rows(ctx: StepCtx, key, make):
    """A value made once per context step (keyed by the step's `pos`
    tensor: the bounded replay's boundary replaces it)."""
    hit = ctx.cache.get(key)
    if hit is None or hit[0] is not ctx.pos:
        hit = ctx.cache[key] = (ctx.pos, make())
    return hit[1]


def _chunk_visible(ctx: StepCtx, ratio: int) -> torch.Tensor:
    """Context chunks: compressed rows each row sees at `ratio` (int64 `[n]`), once per step and ratio (stock's
    `csa2_visible_lengths`, which its staging mapping and its context indexer read)."""
    return _per_rows(
        ctx, ("vis", ratio), lambda: torch.div(ctx.pos.long() + 1, ratio, rounding_mode="floor")
    )


_TILE_ELEMS_CAP = 1 << 28  # fp32 logits elements of one context-chunk indexer tile (1 GiB), as dep2


def _prefill_tile(width: int) -> int:
    """Rows per indexer tile of a candidate consumer's context chunk (dep2's): as many as the engine's MQA-logits budget
    (fp32 elements of one `[rows, width]` logits tile, `TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET`, capped at 1 GiB of
    logits) allows, and never fewer than 1024."""
    return max(1024, min(get_indexer_mqa_logits_elem_budget(), _TILE_ELEMS_CAP) // max(1, width))


def _table(ctx: StepCtx, layer: int, role: str) -> torch.Tensor:
    key = ("t", layer, role)
    if key not in ctx.cache:
        ctx.cache[key] = ctx.kvm.step_table(layer, role, ctx.nreq)
    return ctx.cache[key]


def _pool(ctx: StepCtx, layer: int, role: str) -> torch.Tensor:
    key = ("p", layer, role)
    if key not in ctx.cache:
        ctx.cache[key] = ctx.kvm.pool(layer, role)
    return ctx.cache[key]


class _Attention(nn.Module):
    # The engine's `enable_heuristic_topk` (`sparse_attention_config`): wide selections run the hint-free self-sampling
    # GVR top-k when set, the exact radix top-k otherwise (exact either way, up to ties). Set by the layer from the
    # deployment's config.
    heuristic_topk: bool = True
    # The engine's context selection (`CSA2Indexer` prefill top-k): the self-sampling GVR top-k, which is exact,
    # whenever `use_self_sampling_topk` holds, whatever `enable_heuristic_topk` says (that flag selects the decode
    # engine only).
    # Set by the layer from the deployment's config.
    prefill_self_sampling: bool = True

    def __init__(self, cfg, layer_idx: int, mapping):
        super().__init__()
        self.layer = layer_idx
        self.eps = cfg.rms_norm_eps
        self.dim, self.hd, self.rd = cfg.hidden_size, cfg.head_dim, cfg.qk_rope_head_dim
        self.n_heads, self.q_rank = cfg.num_attention_heads, cfg.q_lora_rank
        self.o_rank, self.o_groups = cfg.o_lora_rank, cfg.o_groups
        self.window = cfg.sliding_window
        self.ratio = cfg.compress_ratios[layer_idx]
        self.ratios = list(cfg.compress_ratios)
        srcs = sorted(cfg.kv_source_layer_ids)
        self.sources = srcs
        self.owner = max((s for s in srcs if s <= layer_idx), default=None) if self.ratio else None
        self.is_owner = layer_idx in srcs
        self.is_index_source = layer_idx in cfg.index_source_layer_ids
        cand = cfg.candidate_source_layer_id
        self.is_cand_source = layer_idx == cand
        self.uses_cands = 0 <= cand < layer_idx and self.is_index_source
        self.cand_block, self.cand_topk = cfg.candidate_block_size, cfg.candidate_topk_blocks
        self.index_topk, self.index_heads, self.index_hd = (
            cfg.index_topk,
            cfg.index_n_heads,
            cfg.index_head_dim,
        )
        # attention TP: with attention DP the block is replicated
        adp = bool(mapping.enable_attention_dp)
        self.tp, self.tp_rank = (1, 0) if adp else (mapping.tp_size, mapping.tp_rank)
        self.group = list(range(mapping.world_size))
        self.reduce_output = True  # False: return this rank's partial sum; the layer reduces it with its own all-reduce
        self.side_stream = None  # the layer's side stream for generation-step overlap (set by the layer), None: inline
        self._side_events = (torch.cuda.Event(), torch.cuda.Event())  # (fork, side work done)
        # an owner that is also an index source derives its index weights on this second side stream, next to its
        # compression chain (the engine's `weights_stream`); None: on the side stream after the compression
        self.weights_stream = None
        self._weights_done = torch.cuda.Event()
        self._q_events = (
            torch.cuda.Event(),
            torch.cuda.Event(),
        )  # (q_a | kv done, late query done)
        self._cand_event = (
            torch.cuda.Event()
        )  # candidate source: candidates and the consumers' schedule published
        self.hl = self.n_heads // self.tp
        self.gl = self.o_groups // self.tp
        self.scale = self.hd**-0.5

        def buf(*shape, dtype=torch.bfloat16):
            return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)

        self.attn_sink = buf(self.hl, dtype=torch.float32)
        # the dense projections are the checkpoint's own MXFP8 weights:
        # E4M3 [N, K] with per-row 1x32 UE8M0 scales [N, K/32]
        fp8 = torch.float8_e4m3fn
        self.q_norm, self.kv_norm = buf(self.q_rank), buf(self.hd)
        self.wq_a, self.wq_a_s = (
            buf(self.q_rank, self.dim, dtype=fp8),
            buf(self.q_rank, self.dim // 32, dtype=torch.uint8),
        )
        self.wq_b, self.wq_b_s = (
            buf(self.hl * self.hd, self.q_rank, dtype=fp8),
            buf(self.hl * self.hd, self.q_rank // 32, dtype=torch.uint8),
        )
        self.wkv, self.wkv_s = (
            buf(self.hd, self.dim, dtype=fp8),
            buf(self.hd, self.dim // 32, dtype=torch.uint8),
        )
        self.w_qkv = None  # wq_a | wkv as one weight (and scale), built on first use
        # grouped o_a: E4M3 with 128x128 power-of-two block scales (the engine's O-A kernels' layout)
        od = self.n_heads * self.hd // self.o_groups
        self.wo_a, self.wo_a_scale = (
            buf(self.gl * self.o_rank, od, dtype=fp8),
            buf(self.gl * self.o_rank // 128, od // 128, dtype=torch.float32),
        )
        self.wo_b, self.wo_b_s = (
            buf(self.dim, self.gl * self.o_rank, dtype=fp8),
            buf(self.dim, self.gl * self.o_rank // 32, dtype=torch.uint8),
        )
        if self.is_owner:
            self.c_norm = buf(self.hd)
            self.c_wkvg_bf16 = None  # value | gate weight at ratio 2 in bf16, built on first use
            if self.ratio == 2:
                self.c_wkv, self.c_wgate = (
                    buf(self.hd, self.dim, dtype=torch.float32),
                    buf(self.hd, self.dim, dtype=torch.float32),
                )
            else:
                self.c_wkv = buf(self.hd, self.dim)
            self.i_wk, self.i_knorm = buf(self.index_hd, self.hd), buf(self.index_hd)
        if self.is_index_source:
            self.i_wq_b, self.i_wq_b_s = (
                buf(self.index_heads * self.index_hd, self.q_rank, dtype=fp8),
                buf(self.index_heads * self.index_hd, self.q_rank // 32, dtype=torch.uint8),
            )
            self.i_wproj = buf(self.index_heads, self.dim)

    # ---- helpers -----------------------------------------------------------------
    def _prep_projections(self, dev):
        """First use (an eager step): fused q_a | kv weight and the swizzled scales the quantized GEMM reads."""
        self.w_qkv = torch.cat([self.wq_a, self.wkv], 0)
        self.w_qkv_s = torch.cat([self.wq_a_s, self.wkv_s], 0)
        self.wq_a.data, self.wkv.data = self.w_qkv[: self.q_rank], self.w_qkv[self.q_rank :]
        self.wq_a_s.data, self.wkv_s.data = self.w_qkv_s[: self.q_rank], self.w_qkv_s[self.q_rank :]
        self.sw = {
            k: mxfp8_swizzled_scales(v)
            for k, v in (("qkv", self.w_qkv_s), ("q_b", self.wq_b_s), ("o_b", self.wo_b_s))
        }
        if self.is_index_source:
            self.sw["i_q_b"] = mxfp8_swizzled_scales(self.i_wq_b_s)
        self.alpha = torch.ones(1, dtype=torch.float32, device=dev)
        od = self.wo_a.shape[1]
        self.o_a_w, self.o_a_sf = (
            self.wo_a.view(self.gl, self.o_rank, od),
            self.wo_a_scale.view(self.gl, self.o_rank // 128, od // 128),
        )
        self.o_a_mx = csa2_o_a_mxfp8_scales(self.o_a_sf)

    def _index_desc(self, ctx):
        """The index owner's per-row paged descriptors, built once per step for all the layers that read its keys. Under
        attention TP (as the built-in `_deepgemm_page_mask`) the same launch also writes the DeepGEMM-masked table, with
        which the paged logits score unallocated pages -inf themselves (`ctx.cache[("masked", owner)]`)."""
        o = self.owner
        key = ("desc", o)
        if key not in ctx.cache:
            _, _, visible = _swa_plan(ctx, self.ratios)
            masked = self.tp > 1 and self.index_heads <= 32
            out = csa2_indexer_descriptors(
                _table(ctx, o, GLOBAL), ctx.rid, visible[o], self.ratio, masked=masked
            )
            ctx.cache[key] = out[:4]
            ctx.cache[("masked", o)] = out[4] if masked else None
        return ctx.cache[key]

    def _index_weights(self, x, bf16=False):
        # the engine's index weights: bf16 projection (low-M kernel at decode sizes), then `(proj * scale).float()` in
        # one launch; `bf16`: into bf16, as the engine's candidate consumers (their sparse logits kernel reads bf16
        # weights; on the side stream that projects them, not on this layer's chain to the indexer)
        return csa2_bf16_lowm_proj(
            x,
            self.i_wproj,
            self.index_hd**-0.5 * self.index_heads**-0.5,
            torch.bfloat16 if bf16 else torch.float32,
        )

    def _prefill_index_desc(self, ctx):
        """Context chunks: the index owner's GLOBAL table cut to the columns the step's spans reach (the engine's table
        is `max_seq_len / 128` columns wide, and every per-row page gather of the indexer is that wide) and the rows'
        descriptors over it (`csa2_indexer_descriptors`), once per owner and step (dep2's)."""
        o = self.owner

        def make():
            cols = -(
                -max(seg[3] for seg in ctx.segs) // _WINDOW_ROWS
            )  # a GLOBAL page holds 128 positions
            table = _table(ctx, o, GLOBAL)[:, :cols]
            return table, csa2_indexer_descriptors(
                table, ctx.rid, _chunk_visible(ctx, self.ratio), self.ratio
            )

        return _per_rows(ctx, ("pdesc", o), make)

    def _prefill_index_plan(self, ctx):
        """Context chunks: the index owner's packed indexer groups (`csa2_indexer_plan`, stock's per-step indexer plan:
        pieces bounded by the MQA-logits budget, merged up to 32768 rows) and a cache of their DeepGEMM schedules, once
        per owner and step (dep2's)."""
        o = self.owner
        return _per_rows(
            ctx,
            ("plan", o),
            lambda: (
                csa2_indexer_plan(
                    ctx.segs,
                    self.ratio,
                    ctx.kvm.kvm.max_seq_len // self.ratio,
                    get_indexer_mqa_logits_elem_budget(),
                    ctx.pos.device,
                ),
                {},
            ),
        )

    def _index_topk(self, ctx, x, q, pos, w=None):
        """`q`: the index queries as the fused store quantized them, (data, scales)."""
        o = self.owner
        if w is None:
            w = self._index_weights(x)
        if ctx.paged:
            desc = self._index_desc(ctx)
            sched = None
            if (
                self.uses_cands
            ):  # every consumer of the step scores the same candidates: one schedule per step
                ready = ctx.cache.pop("cands_ready", None)
                if (
                    ready is not None
                ):  # published (with the schedule) on the source layer's side stream
                    torch.cuda.current_stream().wait_event(ready)
                # Decode rows the one-launch select serves (the built-in Reindex consumers, `reindex_select_`)
                # need no paged sparse-logits schedule, as the built-in model builds none for them.
                if not consumer_select_fused(pos.shape[0], ctx.cands, self.index_topk):
                    key = ("csched", o)
                    if key not in ctx.cache:
                        ctx.cache[key] = csa2_consumer_schedule(
                            ctx.cands[2].reshape(-1), desc[0], ctx.cands[1]
                        )
                    sched = ctx.cache[key]
            # the candidate source publishes on a side stream (only later layers read the candidates), as the engine:
            # the second one (the engine publishes on `csa2_side_stream`, apart from the stream that projects the next
            # layers' queries), so a long publication does not hold the next layers' query projections back
            pub = self.weights_stream if self.weights_stream is not None else self.side_stream
            side = pub if self.is_cand_source and pub is not None and do_multi_stream() else None
            idx, cands = csa2_indexer_topk_decode(
                q,
                w,
                pos,
                ctx.rid,
                _pool(ctx, o, INDEX),
                _table(ctx, o, GLOBAL),
                self.ratio,
                self.index_topk,
                ctx.kvm.kvm.max_seq_len // self.ratio,
                ctx.cands if self.uses_cands else None,
                (self.cand_block, self.cand_topk) if self.is_cand_source else None,
                desc,
                self.heuristic_topk
                and not (
                    self.is_cand_source and pos.shape[0] > 256
                ),  # as the engine: exact radix for wide sources
                cand_sched=sched,
                logits_sched=ctx.cache.get(("lsched", self.layer)),
                rows_per_request=_rows_per_request(ctx),
                publish_stream=side,
                publish_done=self._cand_event if side is not None else None,
                mask_unmapped=not self.uses_cands,  # as the engine: unallocated pages masked before the top-k
                cand_decode_topk=True,
                # the built-in attention-TP selection chain (#22366): the DeepGEMM-masked page table (no mask launch)
                # and the per-row cluster GVR that also finalizes the selection
                tp_chain=self.tp > 1,
                masked_blocks=None if self.uses_cands else ctx.cache.get(("masked", o)),
            )  # as the engine at decode: the multi-CTA block top-k of the candidates
            if side is not None:
                if not consumer_select_fused(pos.shape[0], cands, self.index_topk):
                    with torch.cuda.stream(
                        side
                    ):  # the consumers' schedule, after the publication on the same stream
                        ctx.cache[("csched", o)] = csa2_consumer_schedule(
                            cands[2].reshape(-1), desc[0], cands[1]
                        )
                        self._cand_event.record()
                    ctx.cache[("csched", o)].record_stream(torch.cuda.current_stream())
                # else the consumers' one-launch select reads the candidates alone: they wait on the publication
                # itself (`publish_done`, recorded by the indexer op behind it on that stream)
                ctx.cache["cands_ready"] = self._cand_event
        elif not self.uses_cands:
            # as stock's context indexer (`CSA2Indexer._run_packed_group`, dep2's): each request's prefix gathered once
            # per packed group (whole chunks up to the MQA-logits budget and 32768 rows, not 1024-row tiles), non-paged
            # MQA logits, the prefill top-k per row, and the candidate source's blocks from the same logits
            plan, scheds = self._prefill_index_plan(ctx)
            idx, cands = csa2_indexer_topk_packed(
                q,
                w,
                _chunk_visible(ctx, self.ratio),
                _pool(ctx, o, INDEX),
                _table(ctx, o, GLOBAL),
                self.ratio,
                self.index_topk,
                plan,
                ctx.kvm.kvm.max_seq_len // self.ratio,
                (self.cand_block, self.cand_topk) if self.is_cand_source else None,
                self.prefill_self_sampling,
                scheds,
            )
        else:
            # a candidate consumer (dep2's): the step's descriptors once per owner, tiles as wide as the logits budget
            span = -(-max(s[3] for s in ctx.segs) // self.ratio)
            width = -(-span // 8) * 8
            table, desc = self._prefill_index_desc(ctx)
            idx, cands = csa2_indexer_topk_prefill(
                q,
                w,
                pos,
                ctx.rid,
                _pool(ctx, o, INDEX),
                table,
                self.ratio,
                self.index_topk,
                width,
                ctx.cands,
                None,
                tile=_prefill_tile(width),
                desc=desc,
                heuristic=self.prefill_self_sampling,
            )
        if self.is_cand_source:
            ctx.cands = cands
        ctx.topk = idx

    # ---- forward -----------------------------------------------------------------
    def forward(self, x: torch.Tensor, ctx: StepCtx, x_mxfp8=None) -> torch.Tensor:
        """`x_mxfp8`: `x` already quantized by its producer (the mHC boundary), for the quantized q_a | kv GEMM."""
        return self._forward_fused(x, ctx, x_mxfp8)

    def _compress(self, ctx, x, pos):
        """Owner layers: this step's pooled latent rows (normalised, not rotated), their index-key projection, the flat
        GLOBAL pool row each is published at and the first position of its pair: the engine's compressor
        (`CSA2Compressor.project`: `csa2_compressor_project` over the bf16 projection) on the step's compressed groups
        (`_owner_plan`), for generation steps and context chunks alike (as the dep target)."""
        L = self.layer
        plan = _owner_plan(ctx, self.sources, self.ratios)
        _, slots, first, _, _, starts, ends, cu_comp = plan["owners"][L]
        if self.ratio == 2:
            if self.c_wkvg_bf16 is None:  # (first use is an eager step; no fp32 concat is kept)
                self.c_wkvg_bf16 = torch.cat([self.c_wkv, self.c_wgate], 0).to(torch.bfloat16)
            src = csa2_bf16_lowm_proj(
                x, self.c_wkvg_bf16
            )  # the engine's `bf16_projection` of `wkv_gate`
            state = (
                _pool(ctx, L, COMP_KV),
                _pool(ctx, L, COMP_SCORE),
                _table(ctx, L, COMP_KV),
                _table(ctx, L, COMP_SCORE),
            )
        else:
            src = csa2_bf16_lowm_proj(x, self.c_wkv)
            state, starts, ends, cu_comp = (None,) * 4, plan["ends"], plan["ends"], plan["cu_seq"]
        lat, k_raw = csa2_compressor_project(
            src,
            *state,
            ends,
            starts,
            plan["cu_seq"],
            cu_comp,
            plan["max_outputs"],
            self.c_norm,
            self.eps,
            self.i_wk,
            self.ratio,
        )
        if ctx.hist_end is not None:  # an Encoder replay's historical groups publish nothing
            slots = _unpublished_history(slots, first, cu_comp, ctx.hist_end, self.ratio)
        return lat, k_raw, slots, first

    def publish_global(self, x: torch.Tensor, ctx: StepCtx) -> None:
        """Owner layer at the bounded-replay boundary (stock's `prepare_ced_global_kv`): publish the compressed KV and
        index-key rows of every row of `x` (the layer's normalised attention input) without running the layer; the layer
        then runs over the replay rows only, with `ctx.published` set."""
        assert self.ratio == 1 and self.is_owner and not ctx.paged, (
            "the replay boundary is an unpooled KV owner of a context step"
        )
        lat, k_raw, addr, first = self._compress(ctx, x, ctx.pos)
        rope_c = csa2_stock_rope_table(ctx.rope[1])
        main = (
            _pool(ctx, self.layer, GLOBAL).view(-1, 288),
            addr,
            lat,
            None,
            (first, rope_c, self.rd),
        )
        index = (
            _pool(ctx, self.layer, INDEX),
            addr,
            k_raw,
            (self.i_knorm, self.eps),
            (first, rope_c, self.rd),
        )
        csa2_store_rows(None, main, index, None)

    def _forward_fused(self, x: torch.Tensor, ctx: StepCtx, x_mxfp8=None) -> torch.Tensor:
        """Same math as `_forward_torch`; cache rows are quantized and published by one fused launch, ropes run in
        place, attention and the indexer use the engine's kernels, o_a is one grouped bmm."""
        L, pos, n, dev = self.layer, ctx.pos, x.shape[0], x.device
        rope_q = csa2_stock_rope_table(ctx.rope[1 if self.ratio else 0])
        rope_c = csa2_stock_rope_table(ctx.rope[1]) if self.ratio else None
        if self.w_qkv is None:
            self._prep_projections(dev)
        if (
            ctx.paged
        ):  # the step's compressed groups of every owner, at the step's first layer on this stream
            _owner_plan(ctx, self.sources, self.ratios)
        # Generation steps under a multi-stream (CUDA-graph) forward overlap the side work as the engine's CSA2 module
        # does: an index-source layer derives its index weights (and an owner its compressed rows) on the side stream
        # next to the query projections; any other layer projects its
        # query there while this stream publishes the window row.
        side = (
            self.side_stream
            if ctx.paged and self.side_stream is not None and do_multi_stream()
            else None
        )
        owns = bool(
            self.ratio and self.is_owner and not ctx.published
        )  # (the replay boundary published its rows)
        full = bool(self.ratio and (owns or self.is_index_source))
        side_w = comp = desc = lsched = wstream = None
        if side is not None and full:
            fork, done = self._side_events
            with torch.cuda.stream(side):
                # The step's paged descriptors and this layer's logits schedule read only the step's metadata (which the
                # side stream already follows): issued ahead of the wait for this layer's input, off its critical path.
                desc = (
                    self._index_desc(ctx)
                    if self.is_index_source and ("desc", self.owner) not in ctx.cache
                    else None
                )
                if self.is_index_source and not self.uses_cands:
                    lsched = csa2_logits_schedule(self._index_desc(ctx)[1], _rows_per_request(ctx))
                # A Reindex consumer of the one-launch select: its source's published candidate blocks' index rows go
                # to L2 while this layer projects (the built-in `prefetch_candidate_pages`, #22465), once published.
                ready = ctx.cache.get("cands_ready")
                if (
                    self.uses_cands
                    and ctx.cands is not None
                    and consumer_select_fused(x.shape[0], ctx.cands, self.index_topk)
                ):
                    if ready is not None:
                        side.wait_event(ready)
                    prefetch_candidates(
                        _pool(ctx, self.owner, INDEX), self._index_desc(ctx)[0], ctx.cands
                    )
                    for t in ctx.cands:
                        t.record_stream(side)
            fork.record()
            wstream = self.weights_stream if owns and self.is_index_source else None
            if (
                wstream is not None
            ):  # as the engine's FULL layers: the index weights next to the compression chain
                with torch.cuda.stream(wstream):
                    fork.wait()
                    x.record_stream(wstream)
                    side_w = self._index_weights(x)
                    self._weights_done.record()
            with torch.cuda.stream(side):
                fork.wait()
                # the owner's compressed rows first (the store waits for them), then the index weights (the top-k's)
                comp = self._compress(ctx, x, pos) if owns else None
                if self.is_index_source and wstream is None:
                    side_w = self._index_weights(x, bf16=self.uses_cands)
                done.record()
            if lsched is not None:
                lsched.record_stream(torch.cuda.current_stream())
                ctx.cache[("lsched", self.layer)] = lsched
            if (
                desc is not None
            ):  # the owner's per-step paged descriptors were built there, off the critical path
                masked = ctx.cache.get(("masked", self.owner))
                for t in (*desc, masked):
                    if isinstance(t, torch.Tensor):
                        t.record_stream(torch.cuda.current_stream())
        # q_a | kv in one projection; the q_norm is folded into q_b's input (and the index q_b's); the query RoPE is
        # applied by q_b's epilogue (small steps) or right after it, so every attention kernel reads a rotated query
        # outside decode graphs q_a | kv (and O-B below) run stock's native autotuned projection GEMM, as its CSA2
        # module's `native_mxfp8_gemm` (the same tuner calls in the same order at warm-up: the MoE tuner's dummy routing
        # then draws from the same RNG position as stock's)
        qr_raw, kv_raw = csa2_mxfp8_linear(
            x,
            self.w_qkv,
            self.w_qkv_s,
            self.sw["qkv"],
            self.alpha,
            self.q_rank,
            x_mxfp8=x_mxfp8,
            native_eager=True,
        )
        # generation steps run the engine's direct decode (single launch up to 16 rows, as its CSA2 module does; more
        # rows in chunks); it rotates the query as it loads it, unless q_b's
        # own epilogue already did (never with the folded q_norm)
        rope_in_decode = (
            ctx.paged
            and n <= _DIRECT_MAX_ROWS
            and not csa2_mxfp8_epilogue_rope(n, self.wq_b, rms_norm=True)
        )
        # a FULL layer whose Q-B takes the small-M kernel normalizes the latent once, as stock's `_project_query_latent`
        # (both Q-Bs read the normalized bf16 rows); other layers fold the norm into their Q-B kernel
        if full and csa2_mxfp8_epilogue_rope(n, self.wq_b):
            q_in, q_norm = csa2_q_norm(qr_raw, self.q_norm, self.eps), None
        else:
            q_in, q_norm = qr_raw, (self.q_norm, self.eps)
        side_q = side is not None and not full
        # Index-source / owner layers project the attention query on the side stream after their compression (the
        # engine's late query of FULL layers: off this stream's chain to the indexer; the attention waits for it).
        late_q = side is not None and full
        if side_q:
            fork, done = self._side_events
            fork.record()
        elif late_q:
            self._q_events[0].record()
        with torch.cuda.stream(side) if (side_q or late_q) else contextlib.nullcontext():
            if side_q:
                fork.wait()
            elif late_q:
                self._q_events[0].wait()
                q_in.record_stream(side)
            q = csa2_mxfp8_linear(
                q_in,
                self.wq_b,
                self.wq_b_s,
                self.sw["q_b"],
                self.alpha,
                0,
                None if rope_in_decode else (pos, rope_q.view(-1, 2, self.rd // 2), self.hd),
                q_norm,
            ).view(n, self.hl, self.hd)
            if side_q:
                done.record()
            elif late_q:
                self._q_events[1].record()
        if side_q or late_q:
            q.record_stream(torch.cuda.current_stream())
        swa_pool = _pool(ctx, L, SWA).view(-1, 528)
        # a context chunk of a < 64-head rank attends as stock's backend does (E4M3 staging + trtllm-gen context FMHA):
        # its rows are published in the pool first and read back through the window slots, as there
        if ctx.paged:  # generation: attention reads the window straight from the pool
            reads, writes, visible = _swa_plan(ctx, self.ratios)
            swa_slots = writes[L]
        else:
            ctx_reads, swa_slots = _swa_context_slots(ctx, L, self.ratios)
        swa = (swa_pool, swa_slots, kv_raw, (self.kv_norm, self.eps), (pos, rope_q, self.rd))
        main = index = index_q = None
        if (
            self.ratio and self.is_index_source
        ):  # this stream's half, next to the side stream's (joined below)
            index_q = csa2_mxfp8_linear(
                q_in,
                self.i_wq_b,
                self.i_wq_b_s,
                self.sw["i_q_b"],
                self.alpha,
                0,
                (pos, rope_c.view(-1, 2, self.rd // 2), self.index_hd),
                q_norm,
            ).view(n, self.index_heads, self.index_hd)
        if side is not None and full:
            torch.cuda.current_stream().wait_event(self._side_events[1])
            if wstream is not None:
                torch.cuda.current_stream().wait_event(self._weights_done)
            for t in (side_w, *(comp or ())):
                if t is not None:
                    t.record_stream(torch.cuda.current_stream())
        if owns:
            lat, k_raw, addr, first = comp if comp is not None else self._compress(ctx, x, pos)
            main = (_pool(ctx, L, GLOBAL).view(-1, 288), addr, lat, None, (first, rope_c, self.rd))
            index = (
                _pool(ctx, L, INDEX),
                addr,
                k_raw,
                (self.i_knorm, self.eps),
                (first, rope_c, self.rd),
            )
        packed_q = csa2_store_rows(swa, main, index, index_q)
        if side_q:
            torch.cuda.current_stream().wait_event(self._side_events[1])

        main_pool = main_table = topk = None
        if self.ratio:
            if self.is_index_source:
                self._index_topk(ctx, x, packed_q, pos, side_w)
            o = self.owner
            main_pool, main_table, topk = _pool(ctx, o, GLOBAL), _table(ctx, o, GLOBAL), ctx.topk
        if late_q:
            torch.cuda.current_stream().wait_event(self._q_events[1])
        _note(self, ctx, "direct_decode" if ctx.paged else "staged_context", "fused-rows")
        if not ctx.paged:
            main_logical = main_mapping = None
            if self.ratio:
                # stock hands the staging kernel the selections and the owner's page mapping (no row map here)
                main_logical = topk
                main_mapping = csa2_main_mapping(
                    main_table,
                    ctx.rid,
                    _chunk_visible(ctx, self.ratio),
                    self.ratio,
                    ctx.kvm.kvm.max_seq_len // self.ratio,
                )
            lengths = [t1 - t0 for t0, t1, _, _ in ctx.segs]
            starts = [span - (t1 - t0) for t0, t1, _, span in ctx.segs]
            att = csa2_staged_context(
                q,
                self.attn_sink,
                swa_pool,
                ctx_reads,
                main_pool if self.ratio else None,
                main_logical,
                main_mapping,
                lengths,
                starts,
                self.index_topk if self.ratio else 0,
            )
        else:
            # As the engine's index-source / Reuse layers: the layer that picked the step's selections also stores the
            # pool row each resolves to; the later layers over the same
            # selections, owner and ratio read those rows unmapped.
            slots_out = main_slots = None
            if self.ratio:
                if self.is_index_source:
                    slots_out = torch.empty(
                        n, -(-topk.shape[1] // 64) * 64, dtype=torch.int64, device=dev
                    )
                    ctx.cache["slots"] = (self.owner, self.ratio, slots_out)
                else:
                    owner, ratio, rows = ctx.cache.get("slots", (None, None, None))
                    main_slots = rows if (owner, ratio) == (self.owner, self.ratio) else None
            att = csa2_direct_decode(
                q,
                self.attn_sink,
                pos,
                ctx.rid,
                _pool(ctx, L, SWA),
                _table(ctx, L, SWA),
                self.window,
                self.scale,
                main_pool,
                main_table,
                topk,
                self.ratio,
                reads[L],
                visible[L],
                q_rope=(pos, rope_q) if rope_in_decode else None,
                slots_out=slots_out,
                main_slots=main_slots,
            )
        # inverse RoPE + grouped o_a + o_b: the low-M pair over a bf16 latent at decode sizes, else over an MXFP8 latent
        out = (
            csa2_o_projection(
                att.view(n, -1),
                pos,
                rope_q,
                self.o_a_w,
                self.o_a_sf,
                self.o_a_mx,
                self.wo_b,
                self.sw["o_b"],
                self.alpha,
                wo_b_scale=self.wo_b_s,
                native_eager=True,
            )
            if n
            else att.new_empty(0, self.dim)
        )
        if self.tp == 1 or not self.reduce_output:
            return out
        ws = len(self.group)
        rows = -(-n // ws) * ws
        outp = out if rows == n else torch.nn.functional.pad(out, [0, 0, 0, rows - n])
        red = cat_allgather(
            cat_reducescatter(outp.contiguous(), None, self.group), None, self.group
        )
        return red if rows == n else torch.split(red, [n, rows - n], 0)[0]
