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

from tensorrt_llm._torch.modeling_v2.catalog.comm.allgather import allgather as cat_allgather
from tensorrt_llm._torch.modeling_v2.catalog.comm.reducescatter import (
    reducescatter as cat_reducescatter,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention import (
    csa2_gen_fmha as _gen_fmha,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_bf16_lowm_proj import (
    csa2_bf16_lowm_proj,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_chunk_window import (
    csa2_chunk_window,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_compressor import (
    csa2_compressor_project,
    csa2_owner_slots,
    owner_slot_descriptors,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_consumer_schedule import (
    csa2_consumer_schedule,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_gen_fmha import (
    csa2_gen_fmha_decode,
    csa2_gen_fmha_supported,
    csa2_gen_fmha_tile,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_indexer_decode import (
    csa2_indexer_topk_decode,
    csa2_indexer_topk_prefill,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_indexer_descriptors import (
    csa2_indexer_descriptors,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_mxfp8_linear import (
    SMALL_M_ROWS,
    csa2_mxfp8_epilogue_rope,
    csa2_mxfp8_linear,
    mxfp8_swizzled_scales,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_o_projection import (
    csa2_o_a_mxfp8_scales,
    csa2_o_projection,
    csa2_o_projection_fp8_in,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_query_gemm_rope import (
    csa2_query_gemm_fp8,
    csa2_query_gemm_rope,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_rms_norm_mxfp8 import (
    csa2_rms_norm_mxfp8,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_rows_io import (
    csa2_scatter_rows,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_shared_stage import (
    csa2_local_window,
    csa2_shared_plan,
    csa2_shared_stage,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_sparse_context import (
    csa2_sparse_context_o_a_input,
    csa2_sparse_context_supported,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_sparse_decode import (
    csa2_direct_decode,
    csa2_direct_decode_supported,
    csa2_sparse_decode,
    csa2_sparse_decode_supported,
    csa2_sparse_decode_workspace,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_store_rows import (
    csa2_stock_rope_table,
    csa2_store_rows,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_swa_slots import (
    csa2_swa_slots,
)
from tensorrt_llm._torch.modules.linear import flashinfer_mxfp8_decode_graph_capture_active
from tensorrt_llm._torch.modules.multi_stream_utils import do_multi_stream
from tensorrt_llm._utils import maybe_pin_memory

from ..swa_tables import swa_table_static

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
_LATE_QUERY_MAX_ROWS = (
    256  # q_b beside the index q_b: both GEMMs of these rows are the swap-AB / dense CuTe ones
)
# Decode-graph steps (warm-up and capture, `flashinfer_mxfp8_decode_graph_capture_active`) of at least this many rows
# project the query straight to unit-scale E4M3 (the RoPE in the GEMM's epilogue) and the fused decode reads it as is,
# as the engine's FP8 decode query (`_fp8_q_min_tokens`, `fp8_query_gemm`: only its decode-graph projection,
# `takes_mxfp8_input`, requests it; an eager step projects the query in bf16).
_FP8_QUERY_MIN_ROWS = 32
# Context chunks attend as stock's `forward_fused_context` (the engine's context path at any row count): the shared-bank
# staging (`csa2_shared_stage`, deduplicated across the layers of a KV owner and index source), the engine's FP8 sparse
# context kernel and its O-A input written by that kernel (inverse RoPE + per-128 quantization) for the grouped
# block-128 O-A, as the dep2 target (a mixed step's rows, context and generation, attend on the fused decode, as
# stock's). Context steps of at least this many rows project the query as stock's fused query path does (`fused_query`
# from 1024 context rows: q_norm + MXFP8 in one launch, Q-B with the RoPE in its epilogue; unit-scale E4M3 for the
# shared-bank kernel); smaller ones keep `csa2_mxfp8_linear` with a separate RoPE.
_QUERY_ROPE_GEMM_MIN_ROWS = 1024
_SWA_ONLY = object()  # the staging-group key of the layers without compressed attention
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
                ("decode" if ctx.paged else "prefill-tiled")
                + ("+candidates" if m.uses_cands or m.is_cand_source else "")
            )
        )
        from tensorrt_llm.logger import logger

        logger.info(
            f"V41 attention path: layer={role} heads={m.hl} step={'gen' if ctx.paged else 'ctx'} "
            f"store={store} attn={attn} indexer={idx}"
        )


_SPARSE_WS: Dict[
    int, torch.Tensor
] = {}  # target-owned scratch of the fused sparse kernel, one per device


def _sparse_workspace(device) -> torch.Tensor:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    ws = _SPARSE_WS.get(idx)
    if ws is None:  # first use is an eager step (the engine warms up before capturing graphs)
        ws = _SPARSE_WS[idx] = csa2_sparse_decode_workspace(device)
    return ws


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
    nctx: int = -1  # context rows of a step with context rows (they come first; the rest are generation rows); -1: all
    cache: dict = field(default_factory=dict)  # pool / table lookups made this step
    swa: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = (
        None  # generation: (reads, writes, visible) of every layer
    )
    # the KV owner's compressed rows of this step are already in the pools (bounded-replay boundary)
    published: bool = False
    floor: Optional[torch.Tensor] = (
        None  # [n] first window position each row may read (bounded replay), else all
    )
    # [nreq] int32: an unconsumed Encoder replay's Global end per request (0 without one), context steps only. The
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


def _swa_write_rows(ctx: StepCtx, layer: int, ratios) -> torch.Tensor:
    """Context chunks: the flat SWA pool row of every query row's own position in `layer` (-1 = unmapped), from the same
    kernel as the generation plan but for one layer (a whole-model plan of a 32k-row chunk would be 1.4 GB)."""
    tables = ctx.kvm.swa_tables
    desc, ratio_t, floors = swa_table_static(tables, ratios)
    sl = slice(layer, layer + 1)
    return csa2_swa_slots(
        ctx.pos,
        ctx.rid,
        tables[sl],
        desc[sl],
        ratio_t[sl],
        floors[: ctx.nreq],
        _WINDOW_ROWS,
        _WINDOW_ROWS,
    )[1][0]


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
    has rows (the engine's `capacity`)."""
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


_EXTRA_STREAMS: Dict[
    Tuple[int, int], torch.cuda.Stream
] = {}  # target-owned side streams beyond the layer's own


def _extra_stream(device, index: int) -> torch.cuda.Stream:
    """Side stream `index` of `device` (1: the step descriptors), as the engine's `csa2_side_stream`."""
    key = (
        torch.device(device).index
        if torch.device(device).index is not None
        else torch.cuda.current_device(),
        index,
    )
    s = _EXTRA_STREAMS.get(key)
    if s is None:  # first use is an eager step (the engine warms up before capturing graphs)
        s = _EXTRA_STREAMS[key] = torch.cuda.Stream(device=key[0])
    return s


def _per_rows(ctx: StepCtx, key, make):
    """A value made once per context step (keyed by the step's `pos` tensor)."""
    hit = ctx.cache.get(key)
    if hit is None or hit[0] is not ctx.pos:
        hit = ctx.cache[key] = (ctx.pos, make())
    return hit[1]


def _chunk_visible(ctx: StepCtx, ratio: int):
    """Context chunks: compressed rows each row sees at `ratio` (int64
    `[n]`), once per step and ratio (None at ratio 0)."""
    if not ratio:
        return None
    return _per_rows(
        ctx, ("vis", ratio), lambda: ((ctx.pos.long() + 1) // ratio).clamp(min=0).contiguous()
    )


def _per_window(ctx: StepCtx, key, make):
    """`_per_rows` for values that also depend on the rows' window floors: the bounded replay may keep every row (same
    `pos`) and only add floors from the replay boundary on."""
    hit = ctx.cache.get(key)
    if hit is None or hit[0] is not ctx.pos or hit[1] is not ctx.floor:
        hit = ctx.cache[key] = (ctx.pos, ctx.floor, make())
    return hit[2]


def _req_cs(ctx: StepCtx) -> torch.Tensor:
    """Context chunks: first position of each request in the step (int64 `[nreq]`, 0 for a request without rows)."""
    return _per_rows(
        ctx,
        ("req_cs",),
        lambda: torch.zeros(ctx.nreq, dtype=torch.int64, device=ctx.pos.device).index_put_(
            (ctx.rid,), ctx.cs.long()
        ),
    )


def _shared_plan(ctx: StepCtx, window: int, ratio: int):
    """Context chunks: the shared-bank layout of the step at `ratio` (`csa2_shared_plan`), once per step and ratio."""
    return _per_window(
        ctx,
        ("shared_plan", ratio),
        lambda: csa2_shared_plan(ctx.segs, ctx.nreq, ctx.pos, ctx.rid, ctx.floor, window, ratio),
    )


def _shared_bank_fits(ctx: StepCtx, window: int, ratio: int, topk: int, n: int) -> bool:
    """The engine's size bound of the shared-bank context path (`_shared_context_plan`): its bank holds the step's
    window and compressed rows, at most as many as the selected-row path stages (`n * (window + the selection's
    128-key tiles)`); a step past it (a chunk over a very long history) takes the selected-row path, so the bank
    workspace stays within that bound."""
    plan = _shared_plan(ctx, window, ratio)
    extra = -(-topk // 128) * 128 if ratio else 0
    return plan.swa_rows + plan.main_rows <= n * (window + extra)


def _shared_local(ctx: StepCtx, rows: int) -> torch.Tensor:
    """The step's window copy `[rows, 528]`, shared by the layers
    (each writes the rows it reads before reading them)."""
    return _per_rows(
        ctx,
        ("shared_local", rows),
        lambda: torch.empty(rows, 528, dtype=torch.uint8, device=ctx.pos.device),
    )


def _shared_group(ctx: StepCtx, key) -> Tuple[bool, int]:
    """`(compact, parity)` of a layer's shared staging: layers of the same KV owner and index source (`key`, which holds
    the selections tensor itself) select the same bank rows, so only the first of such a run compacts (stock's
    `_csa2_shared_serial` / `_csa2_shared_parity`)."""
    st = _per_rows(ctx, ("shared_group",), lambda: {"key": None, "parity": 0})
    same = (
        st["key"] is not None
        and len(st["key"]) == len(key)
        and all(a is b for a, b in zip(st["key"], key))
    )
    st["parity"] = 1 - st["parity"] if same else 0
    st["key"] = key
    return not same, st["parity"]


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

        # The engine's staging group of a ratio-2 layer (`stage_group`: its KV owner and the index source whose
        # selections it reads), and whether it is the group's first layer of a forward: on the `gen_fmha` core that
        # layer compacts the selections and records their scale bits, the group's later layers reuse both.
        def group_of(i):
            return (
                max((s for s in srcs if s <= i), default=None),
                max((s for s in cfg.index_source_layer_ids if s <= i), default=None),
            )

        ratios = cfg.compress_ratios
        self.stage_group = group_of(layer_idx) if self.ratio == _gen_fmha.RATIO else None
        self.stage_first = self.stage_group is not None and not any(
            ratios[i] == _gen_fmha.RATIO and group_of(i) == self.stage_group
            for i in range(layer_idx)
        )
        # rows of the largest CUDA-graph step (set by the layer): the shared staging tile's size
        self.gen_fmha_graph_rows = 0
        self._q_word = (
            None  # the FP8 query's amax word for the `gen_fmha` staging (built on an eager step)
        )
        # attention TP: with attention DP the block is replicated
        adp = bool(mapping.enable_attention_dp)
        self.tp, self.tp_rank = (1, 0) if adp else (mapping.tp_size, mapping.tp_rank)
        self.group = list(range(mapping.world_size))
        self.reduce_output = True  # False: return this rank's partial sum; the layer reduces it with its own all-reduce
        self.side_stream = None  # the layer's side stream for generation-step overlap (set by the layer), None: inline
        self._side_events = (torch.cuda.Event(), torch.cuda.Event())  # (fork, side work done)
        self.query_stream = (
            None  # an index-source layer's q_b stream (set by the layer), next to the index q_b
        )
        self._query_events = (torch.cuda.Event(), torch.cuda.Event())  # (fork, q_b done)
        # an index-source layer's index-weights stream (set by the layer), next to the compressor
        self.weights_stream = None
        self._weights_events = (
            torch.cuda.Event(),
            torch.cuda.Event(),
        )  # (fork, index weights done)
        self._desc_event = (
            torch.cuda.Event()
        )  # an owner's step descriptors built on their own side stream
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
        """The index owner's per-row paged descriptors, built once per step for all the layers that read its keys."""
        o = self.owner
        key = ("desc", o)
        if key not in ctx.cache:
            _, _, visible = _swa_plan(ctx, self.ratios)
            ctx.cache[key] = csa2_indexer_descriptors(
                _table(ctx, o, GLOBAL), ctx.rid, visible[o], self.ratio
            )
        return ctx.cache[key]

    def _index_weights(self, x):
        # the engine's index weights: bf16 projection (low-M kernel up to
        # 64 rows), then `(proj * scale).float()` in one launch
        return csa2_bf16_lowm_proj(x, self.i_wproj, self.index_hd**-0.5 * self.index_heads**-0.5)

    def _index_topk(self, ctx, x, qr, pos, w=None):
        o = self.owner
        q = qr  # (data, scales): the fused store already quantized the index queries
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
                key = ("csched", o)
                if key not in ctx.cache:
                    ctx.cache[key] = csa2_consumer_schedule(
                        ctx.cands[2].reshape(-1), desc[0], ctx.cands[1]
                    )
                sched = ctx.cache[key]
            # the candidate source publishes on a side stream (only later layers read the candidates), as the engine:
            # the second one (the engine publishes on `csa2_side_stream`, its index-weights stream, apart from the
            # stream that projects the next layers' queries), so a long publication
            # does not hold the next layers' query projections back
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
                # the engine's `enable_heuristic_topk` (self-sampling GVR top-k); the exact radix for wide candidate
                # sources, as the engine
                self.heuristic_topk and not (self.is_cand_source and pos.shape[0] > 256),
                # a decode step's rows of one request (token and drafts) read its
                # keys once, as the engine's indexer (next_n)
                cand_sched=sched,
                rows_per_request=_rows_per_request(ctx),
                # as the engine's decode indexer: split-row logits where it runs them (six rows, 4+ requests)
                split_rows=True,
                publish_stream=side,
                publish_done=self._cand_event if side is not None else None,
                mask_unmapped=not self.uses_cands,  # as the engine: unallocated pages masked before the top-k
                cand_decode_topk=True,
            )  # as the engine at decode: the multi-CTA block top-k of the candidates (<= 64 rows)
            if side is not None:
                desc[0].record_stream(
                    side
                )  # (the step descriptors may come from the descriptor stream)
                with torch.cuda.stream(
                    side
                ):  # the consumers' schedule, after the publication on the same stream
                    ctx.cache[("csched", o)] = csa2_consumer_schedule(
                        cands[2].reshape(-1), desc[0], cands[1]
                    )
                    self._cand_event.record()
                ctx.cache[("csched", o)].record_stream(torch.cuda.current_stream())
                ctx.cache["cands_ready"] = self._cand_event
        else:
            span = -(-max(s[3] for s in ctx.segs) // self.ratio)
            idx, cands = csa2_indexer_topk_prefill(
                q,
                w,
                pos,
                ctx.rid,
                _pool(ctx, o, INDEX),
                _table(ctx, o, GLOBAL),
                self.ratio,
                self.index_topk,
                -(-span // 8) * 8,
                ctx.cands if self.uses_cands else None,
                (self.cand_block, self.cand_topk) if self.is_cand_source else None,
                heuristic=self.prefill_self_sampling,
            )
        if self.is_cand_source:
            ctx.cands = cands
        ctx.topk = idx

    # ---- forward -----------------------------------------------------------------
    def forward(self, x: torch.Tensor, ctx: StepCtx, x_mxfp8=None) -> torch.Tensor:
        """`x_mxfp8`: `x` already quantized by its producer (the mHC boundary), for the quantized q_a | kv GEMM. The
        engine's kernels only (64 heads per rank under attention DP; a context chunk of a rank with fewer heads runs
        the 64-head kernel on a head-padded query)."""
        assert csa2_sparse_decode_supported(64, x.device) and (
            self.hl == 64
            or not ctx.paged
            or csa2_direct_decode_supported(self.hl, x.shape[0], x.device)
        ), "the CSA2 attention kernels (SM103, 64 heads)"
        return self._forward_fused(x, ctx, x_mxfp8)

    def _compress(self, ctx, x, pos):
        """Owner layers: this step's pooled latent rows (normalised, not rotated), their index-key projection, the flat
        GLOBAL pool row each is published at and the first position of its pair: the engine's compressor
        (`CSA2Compressor.project`: `csa2_compressor_project` over the bf16 projection) on the step's compressed groups
        (`_owner_plan`), for generation steps and context chunks alike."""
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
        """The layer's math (the plain-torch oracle is the attention-context test's); cache rows are quantized and
        published by one fused launch, ropes run in place, attention and the
        indexer use the engine's kernels, o_a is one grouped bmm."""
        L, pos, n, dev = self.layer, ctx.pos, x.shape[0], x.device
        rope_q = csa2_stock_rope_table(ctx.rope[1 if self.ratio else 0])
        rope_c = csa2_stock_rope_table(ctx.rope[1]) if self.ratio else None
        if self.w_qkv is None:
            self._prep_projections(dev)
        if (
            ctx.paged
        ):  # the step's compressed groups of every owner, at the step's first layer on this stream
            _owner_plan(ctx, self.sources, self.ratios)
        # context chunks of a 64-head rank whose output needs no TP reduction: stock's shared-bank FP8 context path
        # (only a step of context rows alone: stock's fused context writes the O-A input only when every row is a
        # context row, and a mixed step's generation rows attend on its fused decode)
        nctx = n if ctx.nctx < 0 else ctx.nctx
        shared = (
            not ctx.paged
            and nctx == n
            and n > 0
            and self.hl == 64
            and self.window == 128
            and (self.tp == 1 or not self.reduce_output)
            and csa2_sparse_context_supported(dev)
            and _shared_bank_fits(ctx, self.window, self.ratio, self.index_topk, n)
        )
        # stock's fused context query (`fused_query`: 1024+ rows, 1024+ of them context rows): one q_norm + MXFP8 launch
        # for both query projections, Q-B with the RoPE in its epilogue (unit-scale E4M3 for the shared-bank kernel,
        # which reads only that)
        ctx_query = (
            not ctx.paged
            and n >= _QUERY_ROPE_GEMM_MIN_ROWS
            and nctx >= _QUERY_ROPE_GEMM_MIN_ROWS
            and self.hd == 512
            and self.rd == 64
        )
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
        side_w = comp = None
        # the index weights on a stream of their own, so an owner's
        # compressor starts at once (the engine's weights stream)
        w_stream = (
            self.weights_stream if side is not None and full and self.is_index_source else None
        )
        desc_side = None
        if w_stream is not None:
            w_fork, w_done = self._weights_events
            if ("desc", self.owner) not in ctx.cache:  # the owner's first index layer of the step
                # its step descriptors on a third side stream, which the index weights wait for (the engine's descriptor
                # stream); the visible lengths and the table they read are resolved on this stream before the fork
                _swa_plan(ctx, self.ratios), _table(ctx, self.owner, GLOBAL)
                desc_side = True
            w_fork.record()
            if desc_side:
                with torch.cuda.stream(_extra_stream(dev, 1)):
                    w_fork.wait()
                    desc_side = self._index_desc(ctx)
                    self._desc_event.record()
            with torch.cuda.stream(w_stream):
                w_fork.wait()
                side_w = self._index_weights(x)
                if desc_side:
                    self._desc_event.wait()
                w_done.record()
        if side is not None and full:
            fork, done = self._side_events
            fork.record()
            with torch.cuda.stream(side):
                fork.wait()
                if w_stream is None:
                    side_w = self._index_weights(x) if self.is_index_source else None
                comp = self._compress(ctx, x, pos) if owns else None
                done.record()
        # q_a | kv in one projection; the q_norm is folded into q_b's input (and the index q_b's); the query RoPE is
        # applied by q_b's epilogue (small steps) or right after it, so every attention kernel reads a rotated query
        qr_raw, kv_raw = csa2_mxfp8_linear(
            x, self.w_qkv, self.w_qkv_s, self.sw["qkv"], self.alpha, self.q_rank, x_mxfp8=x_mxfp8
        )
        # generation steps of at most 16 rows run the single-launch direct decode (as the engine's CSA2 module does)
        sparse = not ctx.paged or (
            not (n <= _DIRECT_MAX_ROWS and csa2_direct_decode_supported(self.hl, n, dev))
            and csa2_sparse_decode_supported(self.hl, dev)
        )
        # the direct decode rotates the query as it loads it; unless q_b's
        # own epilogue already did (never with the folded q_norm)
        rope_in_decode = (
            ctx.paged
            and not sparse
            and n <= _DIRECT_MAX_ROWS
            and not csa2_mxfp8_epilogue_rope(n, self.wq_b, rms_norm=True)
        )
        side_q = side is not None and not full
        # An index-source step past the low-M rows normalizes and quantizes the latent once for q_b and the index q_b,
        # and (up to _LATE_QUERY_MAX_ROWS) projects q_b on its own stream next to the index q_b, as the engine's decode
        # does (its late query). Past those rows both GEMMs may take the FlashInfer GEMM, whose launches share one
        # workspace: they stay on this stream, one after the other.
        share = side is not None and self.ratio and self.is_index_source and n > SMALL_M_ROWS
        late = self.query_stream if share and n <= _LATE_QUERY_MAX_ROWS else None
        qn = csa2_rms_norm_mxfp8(qr_raw, self.q_norm, self.eps) if share else None
        if late is not None:
            self._query_events[0].record()
        if side_q:
            fork, done = self._side_events
            fork.record()
        q_stream = side if side_q else late
        fast = (
            ctx.paged
            and sparse
            and self.hl == 64
            and self.hd == 512
            and self.rd == 64
            and pos.dtype == torch.int32
            and pos.is_contiguous()
        )
        fp8_q = fast and n >= _FP8_QUERY_MIN_ROWS and flashinfer_mxfp8_decode_graph_capture_active()
        # a ratio-2 staging group's decode step of 1024+ rows attends on the engine's `gen_fmha` core (E4M3 staging into
        # one bank, GB300 sparse MLA) instead of the fused decode, as `CSA2TrtllmAttention.gen_fmha_serves` routes it
        # (graph or eager); the FP8 query GEMM then also raises Q's amax word, which caps the staging scale
        gen_fmha = (
            fast
            and self.stage_group is not None
            and n >= _gen_fmha.MIN_ROWS
            and self.gl == 8
            and csa2_gen_fmha_supported(self.hl, self.index_topk, dev)
        )
        if gen_fmha and fp8_q and self._q_word is None:
            self._q_word = torch.zeros(1, dtype=torch.int32, device=dev)
        with torch.cuda.stream(q_stream) if q_stream is not None else contextlib.nullcontext():
            if side_q:
                fork.wait()
            if late is not None:
                self._query_events[0].wait()
                for t in qn:
                    t.record_stream(late)
            if ctx_query:  # (eager context steps: no side stream)
                q, qn = (csa2_query_gemm_fp8 if shared else csa2_query_gemm_rope)(
                    qr_raw,
                    (self.q_norm, self.eps),
                    self.wq_b,
                    self.sw["q_b"],
                    self.alpha,
                    pos,
                    rope_q,
                )
                q = q.view(n, self.hl, self.hd)
            # the engine's FP8 decode query: unit-scale E4M3 with the RoPE in the epilogue, never stored in bf16 (the
            # graph warm-up's eager steps compile each tile before its capture)
            elif fp8_q:
                q = csa2_query_gemm_fp8(
                    qr_raw,
                    (self.q_norm, self.eps),
                    self.wq_b,
                    self.sw["q_b"],
                    self.alpha,
                    pos,
                    rope_q,
                    x_mxfp8=qn,
                    amax=self._q_word if gen_fmha else None,
                )[0].view(n, self.hl, self.hd)
            else:
                q = csa2_mxfp8_linear(
                    qr_raw,
                    self.wq_b,
                    self.wq_b_s,
                    self.sw["q_b"],
                    self.alpha,
                    0,
                    None if rope_in_decode else (pos, rope_q.view(-1, 2, self.rd // 2), self.hd),
                    None if qn is not None else (self.q_norm, self.eps),
                    qn,
                ).view(n, self.hl, self.hd)
            if side_q:
                done.record()
            if late is not None:
                self._query_events[1].record()
        if q_stream is not None:
            q.record_stream(torch.cuda.current_stream())
        swa_pool = _pool(ctx, L, SWA).view(-1, 528)
        plan = None
        if ctx.paged:  # generation: attention reads the window straight from the pool
            reads, writes, visible = _swa_plan(ctx, self.ratios)
            stage, swa_slots, swa_target = None, writes[L], swa_pool
        elif shared:  # the step's rows go to its paged window copy, which the staging reads as stock reads the pool
            plan = _shared_plan(ctx, self.window, self.ratio)
            stage, swa_slots, swa_target = (
                None,
                plan.local_slot,
                _shared_local(ctx, plan.local_rows),
            )
        else:  # context chunk: rows of the step are attended from a step-local copy (a chunk may exceed the window)
            swa_addr = _swa_write_rows(ctx, L, self.ratios)
            # every row with a SWA slot, as stock writes them (prefix reuse publishes whole pages)
            keep = torch.ones(n, dtype=torch.bool, device=dev)
            stage = torch.empty(n, 528, dtype=torch.uint8, device=dev)
            swa_slots, swa_target = torch.arange(n, device=dev), stage
        swa = (swa_target, swa_slots, kv_raw, (self.kv_norm, self.eps), (pos, rope_q, self.rd))
        main = index = index_q = None
        if side is not None and full:
            torch.cuda.current_stream().wait_event(self._side_events[1])
            for t in ((None if w_stream is not None else side_w), *(comp or ())):
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
        if self.ratio and self.is_index_source:
            index_q = csa2_mxfp8_linear(
                qr_raw,
                self.i_wq_b,
                self.i_wq_b_s,
                self.sw["i_q_b"],
                self.alpha,
                0,
                (pos, rope_c.view(-1, 2, self.rd // 2), self.index_hd),
                None if qn is not None else (self.q_norm, self.eps),
                qn,
            ).view(n, self.index_heads, self.index_hd)
        packed_q = csa2_store_rows(swa, main, index, index_q)
        if side_q:
            torch.cuda.current_stream().wait_event(self._side_events[1])
        if late is not None:
            torch.cuda.current_stream().wait_event(self._query_events[1])
        if stage is not None:  # keep the window's last rows for the following steps
            csa2_scatter_rows(_pool(ctx, L, SWA), swa_addr, stage, keep)
        if (
            plan is not None
        ):  # history rows into the window copy, the window's last rows into the pool
            csa2_local_window(
                swa_target, _pool(ctx, L, SWA), _table(ctx, L, SWA), plan.hist, plan.keep
            )

        main_pool = main_table = topk = None
        if self.ratio:
            if self.is_index_source:
                if (
                    w_stream is not None
                ):  # (the weights stream also waited for the step descriptors)
                    torch.cuda.current_stream().wait_event(self._weights_events[1])
                    for t in (side_w, *(desc_side or ())):
                        t.record_stream(torch.cuda.current_stream())
                self._index_topk(ctx, x, packed_q, pos, side_w)
            o = self.owner
            main_pool, main_table, topk = _pool(ctx, o, GLOBAL), _table(ctx, o, GLOBAL), ctx.topk
        if plan is not None:  # (no TP reduction follows: `shared` requires it)
            _note(self, ctx, "shared_context", "fused-rows")
            return self._shared_context(
                ctx, q, plan, swa_target, main_pool, main_table, topk, rope_q
            )
        _note(
            self,
            ctx,
            "gen_fmha" if gen_fmha else "sparse_decode" if sparse else "direct_decode",
            "fused-rows",
        )
        slots = (
            (reads[L], visible[L]) if ctx.paged else (None, None)
        )  # window rows / visible length of this layer
        if sparse:
            pad = (
                64 - self.hl
            )  # zero heads (and sinks) fill the kernel's 64; their rows are dropped
            q64 = q if not pad else torch.nn.functional.pad(q, (0, 0, 0, pad))
            sink = self.attn_sink if not pad else torch.nn.functional.pad(self.attn_sink, (0, pad))
            # a generation step past the low-M rows: the combine writes the grouped O-A input itself (inverse RoPE +
            # MXFP8), as the engine's fused decode does, instead of a bf16
            # output and a separate inverse_rope_mxfp8 launch
            o_mx = (
                ctx.paged
                and not pad
                and self.gl == 8
                and n > SMALL_M_ROWS
                and pos.dtype == torch.int32
                and pos.is_contiguous()
            )
            if gen_fmha:
                tile = csa2_gen_fmha_tile(dev, n, self.gen_fmha_graph_rows, self.index_topk)
                att = csa2_gen_fmha_decode(
                    tile,
                    q,
                    self._q_word if fp8_q else None,
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
                    *slots,
                    group=self.stage_group,
                    first=self.stage_first,
                    mxfp8_out=(pos, rope_q),
                )
            else:
                att = csa2_sparse_decode(
                    _sparse_workspace(dev),
                    q64,
                    sink,
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
                    stage,
                    None if ctx.paged else ctx.cs,
                    *slots,
                    floor=ctx.floor,
                    req_cs=None if ctx.paged or ctx.floor is None else _req_cs(ctx),
                    mxfp8_out=(pos, rope_q) if o_mx else None,
                )
            if o_mx:
                att, att_mx = None, att
            elif pad:
                att = att[:, : self.hl].contiguous()
        else:  # e.g. 8 heads at TP8
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
                *slots,
                q_rope=(pos, rope_q) if rope_in_decode else None,
            )
        # inverse RoPE + grouped o_a + o_b over an MXFP8 latent (the engine's decode projection)
        if att is None:  # the combine's MXFP8 O-A input
            out = csa2_o_projection(
                None,
                pos,
                rope_q,
                self.o_a_w,
                self.o_a_sf,
                self.o_a_mx,
                self.wo_b,
                self.sw["o_b"],
                self.alpha,
                att_mxfp8=att_mx,
            )
        else:
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

    def _shared_context(
        self, ctx, q, plan, local, main_pool, main_table, topk, rope_q
    ) -> torch.Tensor:
        """Stock's context attention (`forward_fused_context` with `output_fp8`) over the step's shared bank: stage this
        layer's rows (compacting only on the first layer of a KV owner and index source), attend with the FP8 context
        kernel, which writes the O-A input rotated back and quantized
        per 128 channels, then O-A and O-B (`[n, dim]`)."""
        L, pos = self.layer, ctx.pos
        key = (main_pool, topk) if self.ratio else (_SWA_ONLY,)
        compact, parity = _shared_group(ctx, key)
        valid = None
        if compact:
            # the window slots each query may read (the request's start, the replay floor, mapped history pages)
            valid = csa2_chunk_window(
                pos,
                ctx.rid,
                ctx.cs,
                _req_cs(ctx),
                _pool(ctx, L, SWA),
                _table(ctx, L, SWA),
                self.window,
                ctx.floor,
            )[1]
        visible = _chunk_visible(ctx, self.ratio) if self.ratio else None
        max_positions = ctx.kvm.kvm.max_seq_len // self.ratio if self.ratio else 0
        bank, idx, lens, q_scale, bmm1, bmm2 = csa2_shared_stage(
            plan,
            local,
            main_pool.view(-1, 288) if self.ratio else None,
            main_table,
            topk if self.ratio else None,
            visible,
            max_positions,
            valid,
            q,
            self.scale,
            compact,
            parity,
        )
        q_scale = (
            None if q.dtype == torch.float8_e4m3fn else q_scale
        )  # an E4M3 query is already at unit scale
        data, scale = csa2_sparse_context_o_a_input(
            q, bank, idx, lens, q_scale, bmm1, bmm2, self.attn_sink, pos, rope_q, self.gl
        )
        return csa2_o_projection_fp8_in(
            data, scale, self.o_a_w, self.o_a_sf, self.wo_b, self.sw["o_b"], self.alpha
        )
