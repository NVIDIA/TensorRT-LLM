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

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from tensorrt_llm._torch.attention.backends.sparse.params import get_indexer_mqa_logits_elem_budget
from tensorrt_llm._torch.modeling_v2.catalog.comm.allgather import allgather as cat_allgather
from tensorrt_llm._torch.modeling_v2.catalog.comm.reducescatter import (
    reducescatter as cat_reducescatter,
)
from tensorrt_llm._torch.modeling_v2.catalog.gemm.bmm_out import bmm_out
from tensorrt_llm._torch.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_chunk_window import (
    csa2_chunk_window,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_compress_decode import (
    csa2_compress_decode,
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
    csa2_sparse_decode,
    csa2_sparse_decode_supported,
    csa2_sparse_decode_workspace,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_store_rows import (
    csa2_rope_inplace,
    csa2_stock_rope_table,
    csa2_store_rows,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_swa_slots import (
    csa2_swa_slots,
)

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
# Eager context chunks attend as stock's `forward_fused_context`: the shared-bank staging (`csa2_shared_stage`,
# deduplicated across the layers of a KV owner and index source), the engine's FP8 sparse context kernel, and its O-A
# input written by that kernel (inverse RoPE + per-128 quantization) for the grouped block-128 O-A; a graph bucket's
# body (`live` set) attends with the decode kernel.
_TILE_ELEMS_CAP = 1 << 28  # fp32 logits elements of one context-chunk indexer tile (1 GiB)
_SWA_ONLY = object()  # the staging-group key of the layers without compressed attention
# Context steps of at least this many rows project the query as stock's fused query path does (`csa2_query_gemm_rope`:
# stock's `fused_query` from 1024 rows); smaller ones keep `csa2_mxfp8_linear` with a separate RoPE.
_QUERY_ROPE_GEMM_MIN_ROWS = 1024
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
    key = (role, m.hl, ctx.paged, ctx.live is None, attn, store)
    if key not in _LOGGED:
        _LOGGED.add(key)
        idx = (
            "-"
            if not (m.ratio and m.is_index_source)
            else (
                (
                    "decode"
                    if ctx.paged
                    else "prefill-tiled"
                    if m.uses_cands or ctx.live is not None
                    else "prefill-packed"
                )
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
    # the KV owner's compressed rows of this step are already in the pools (bounded-replay boundary)
    published: bool = False
    floor: Optional[torch.Tensor] = (
        None  # [n] first window position each row may read (bounded replay), else all
    )
    live: Optional[torch.Tensor] = (
        None  # [n] bool: False for padding rows of a graph bucket, which publish nothing
    )
    # [n] bool: False for an Encoder replay's historical rows (eager steps; `live` carries it in graph buckets): they
    # rebuild the Encoder working state but publish no compressed rows, which the replay's Global claim holds
    publish: Optional[torch.Tensor] = None
    cache: dict = field(default_factory=dict)  # pool / table lookups made this step
    swa: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = (
        None  # generation: (reads, writes, visible) of every layer
    )


def _publish_mask(ctx) -> Optional[torch.Tensor]:
    """The rows of the step that publish their compressed (GLOBAL / INDEX) rows, or None for every row: not a graph
    bucket's padding rows (`live`), nor an Encoder replay's historical rows (`publish`)."""
    if ctx.publish is None:
        return ctx.live
    return ctx.publish if ctx.live is None else ctx.live & ctx.publish


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


def _per_rows(ctx: StepCtx, key, make):
    """A value made once per set of step rows: the bounded replay's context shares the step's cache but has its own
    rows, so an entry is only reused by the context (`pos` tensor) that made it."""
    hit = ctx.cache.get(key)
    if hit is None or hit[0] is not ctx.pos:
        hit = ctx.cache[key] = (ctx.pos, make())
    return hit[1]


def _prefill_tile(width: int) -> int:
    """Rows per indexer tile of a context chunk: as many as the engine's MQA-logits budget (fp32 elements of one
    `[rows, width]` logits tile, `TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET`, capped at 1 GiB of logits) allows, and never
    fewer than 1024."""
    return max(1024, min(get_indexer_mqa_logits_elem_budget(), _TILE_ELEMS_CAP) // max(1, width))


def _chunk_consts(ctx: StepCtx, window: int, n: int):
    """Context chunks: what every layer needs of the step's rows (`keep` = the rows written to their SWA slots: every
    live row, as stock writes each row's slot, since prefix reuse publishes whole pages; not a graph bucket's padding
    rows, which copy the last row's position but not its values; `arange(n)`), made once per step."""

    def make():
        live = (
            ctx.live
            if ctx.live is not None
            else torch.ones(n, dtype=torch.bool, device=ctx.pos.device)
        )
        return live, torch.arange(n, device=ctx.pos.device)

    return _per_rows(ctx, ("chunk", n, window), make)


def _chunk_visible(ctx: StepCtx, ratio: int):
    """Context chunks: compressed rows each row sees at `ratio` (int64
    `[n]`), once per step and ratio (None at ratio 0)."""
    if not ratio:
        return None
    return _per_rows(
        ctx, ("vis", ratio), lambda: ((ctx.pos.long() + 1) // ratio).clamp(min=0).contiguous()
    )


def _req_cs(ctx: StepCtx) -> torch.Tensor:
    """Context chunks: first position of each request in the step (int64 `[nreq]`, 0 for a request without rows)."""
    return _per_rows(
        ctx,
        ("req_cs",),
        lambda: torch.zeros(ctx.nreq, dtype=torch.int64, device=ctx.pos.device).index_put_(
            (ctx.rid,), ctx.cs.long()
        ),
    )


def _per_window(ctx: StepCtx, key, make):
    """`_per_rows` for values that also depend on the rows' window floors: the bounded replay may keep every row (same
    `pos`) and only add floors from the replay boundary on."""
    hit = ctx.cache.get(key)
    if hit is None or hit[0] is not ctx.pos or hit[1] is not ctx.floor:
        hit = ctx.cache[key] = (ctx.pos, ctx.floor, make())
    return hit[2]


def _shared_plan(ctx: StepCtx, window: int, ratio: int):
    """Eager context chunks: the shared-bank layout of the step at
    `ratio` (`csa2_shared_plan`), once per step and ratio."""
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
    st = _per_window(ctx, ("shared_group",), lambda: {"key": None, "parity": 0})
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

    def _wo_a_dense(self) -> torch.Tensor:
        """The grouped o_a dequantized to bf16 `[G*R, D]` (the TP-reduced output only)."""
        return (
            self.wo_a.float() * self.wo_a_scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
        ).to(torch.bfloat16)

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

    def _prefill_index_desc(self, ctx):
        """Context chunks: the index owner's GLOBAL table cut to the columns the step's spans reach (the engine's table
        is `max_seq_len / 128` columns wide, and every per-row page gather of the indexer is that wide) and the rows'
        descriptors over it (`csa2_indexer_descriptors`), once per owner and step."""
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
        per owner and step."""
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

    def _index_topk(self, ctx, x, qr, pos):
        o = self.owner
        q = qr  # (data, scales): the fused store quantized the index queries
        proj = cublas_mm(
            x, self.i_wproj.t()
        )  # one launch for `(proj * scale).float()`, bit for bit
        w = torch.mul(
            proj,
            self.index_hd**-0.5 * self.index_heads**-0.5,
            out=torch.empty(proj.shape, dtype=torch.float32, device=proj.device),
        )
        if ctx.paged:
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
                self._index_desc(ctx),
                # the engine's heuristic (self-sampling GVR) top-k when enabled; exact radix for wide sources
                self.heuristic_topk and not (self.is_cand_source and pos.shape[0] > 256),
                mask_unmapped=not self.uses_cands,  # as the engine: unallocated pages masked before the top-k
            )
        elif not self.uses_cands and ctx.live is None:
            # as stock's context indexer (eager steps): each request's prefix gathered once per group, non-paged MQA
            # logits, the prefill top-k per row (`csa2_indexer_topk_packed`). A graph bucket's body (`live` set: the
            # static body, the two-pass graphs) keeps the tiled path below: its `segs` are one bucket-wide span, and the
            # packed plan's host descriptors would be baked into the graph
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
                ctx.cands if self.uses_cands else None,
                (self.cand_block, self.cand_topk) if self.is_cand_source else None,
                tile=_prefill_tile(width),
                desc=desc,
                heuristic=self.prefill_self_sampling,
            )
        if self.is_cand_source:
            ctx.cands = cands
        ctx.topk = idx

    # ---- forward -----------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, ctx: StepCtx, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """`out`: a contiguous bf16 `[n, dim]` the output projection writes into (returned when given)."""
        return self._forward_fused(x, ctx, out)

    def _compress(self, ctx, x, pos):
        """Owner layers: this step's pooled latent rows (normalised, not rotated), their index-key projection, the flat
        GLOBAL pool row each is published at and the first position of its pair; one launch after the projection."""
        L = self.layer
        if self.ratio == 2:
            # Value and gate are one weight, projected in bf16 as the engine's compressor does (`bf16_projection` of its
            # bf16 `wkv_gate`, generation and context alike).
            if self.c_wkvg_bf16 is None:  # (first use is an eager step)
                self.c_wkvg_bf16 = torch.cat([self.c_wkv, self.c_wgate], 0).to(torch.bfloat16)
            src = cublas_mm(x, self.c_wkvg_bf16.t())
            state = (
                _pool(ctx, L, COMP_KV),
                _pool(ctx, L, COMP_SCORE),
                _table(ctx, L, COMP_KV),
                _table(ctx, L, COMP_SCORE),
            )
        else:
            src = cublas_mm(x, self.c_wkv.t())
            state = (None, None, None, None)
        # context steps project the index keys with one bf16 GEMM, as stock's prefill compressor (`bf16_projection`)
        return csa2_compress_decode(
            src,
            pos,
            ctx.rid,
            ctx.cs,
            ctx.ce,
            *state,
            self.c_norm,
            self.eps,
            self.i_wk,
            _table(ctx, L, GLOBAL),
            self.ratio,
            not ctx.paged and x.shape[0] >= 256,
        )

    def _store_specs(self, ctx, lat, k_raw, addr, first, rope_c):
        """The `main` / `index` store specs of `csa2_store_rows` for this step's compressed rows."""
        L = self.layer
        mask = _publish_mask(ctx)  # padding rows and Encoder replay history publish nothing
        if mask is not None:
            addr = torch.where(mask, addr, -1)
        main = (_pool(ctx, L, GLOBAL).view(-1, 288), addr, lat, None, (first, rope_c, self.rd))
        index = (
            _pool(ctx, L, INDEX),
            addr,
            k_raw,
            (self.i_knorm, self.eps),
            (first, rope_c, self.rd),
        )
        return main, index

    def publish_global(self, x: torch.Tensor, ctx: StepCtx) -> None:
        """Owner layer at the bounded-replay boundary: publish the compressed KV and index-key rows of every row of `x`
        (the layer's normalised attention input) without running the layer; the layer then runs over the replay rows
        only, with `ctx.published` set."""
        assert self.ratio == 1 and self.is_owner and not ctx.paged, (
            "the replay boundary is an unpooled KV owner of a context step"
        )
        lat, k_raw, addr, first = self._compress(ctx, x, ctx.pos)
        rope_c = csa2_stock_rope_table(ctx.rope[1])
        main, index = self._store_specs(ctx, lat, k_raw, addr, first, rope_c)
        csa2_store_rows(None, main, index, None)

    def _forward_fused(
        self, x: torch.Tensor, ctx: StepCtx, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Cache rows are quantized and published by one fused launch, ropes run in place, attention and the indexer use
        the engine's kernels, o_a is one grouped bmm."""
        pos, n, dev = ctx.pos, x.shape[0], x.device
        rope_q = csa2_stock_rope_table(ctx.rope[1 if self.ratio else 0])
        rope_c = csa2_stock_rope_table(ctx.rope[1]) if self.ratio else None
        # eager context chunks of a 64-head rank whose output needs no
        # TP reduction: stock's shared-bank FP8 context path
        shared = (
            not ctx.paged
            and ctx.live is None
            and n > 0
            and self.hl == 64
            and self.window == 128
            and (self.tp == 1 or not self.reduce_output)
            and csa2_sparse_context_supported(dev)
            and _shared_bank_fits(ctx, self.window, self.ratio, self.index_topk, n)
        )
        # eager steps only (`ctx.live` unset): the CuTe query GEMM compiles per tile on first use, which a capture
        # cannot the shared-bank context kernel reads only an E4M3 query
        # at unit scale (stock's `_unit_scale_context_query`)
        q, kv_raw, index_q = self._project(
            x, pos, rope_q, rope_c, not ctx.paged and ctx.live is None, shared
        )
        att = self._attend(x, ctx, q, kv_raw, index_q, rope_q, rope_c, shared, out)
        return att if shared else self._output(att, pos, rope_q, out)

    def _project(self, x, pos, rope_q, rope_c, fused_query: bool, fp8_query: bool = False):
        """The row-wise projections: `(q [n, hl, hd] rotated, kv_raw, index_q or None)`; they read no step metadata.
        With `fused_query` and `fp8_query` the query is E4M3 at unit
        scale (only the shared-bank context kernel reads it)."""
        n = x.shape[0]
        if self.w_qkv is None:
            self._prep_projections(x.device)
        # q_a | kv in one projection; the q_norm is folded into q_b's input (and the index q_b's); the query RoPE is
        # applied by q_b's epilogue (small steps) or right after it, so every attention kernel reads a rotated query
        qr_raw, kv_raw = csa2_mxfp8_linear(
            x, self.w_qkv, self.w_qkv_s, self.sw["qkv"], self.alpha, self.q_rank
        )
        latent = None  # the normalized q_a latent in MXFP8, when the query path made it (the index q_b reads it too)
        if fused_query and n >= _QUERY_ROPE_GEMM_MIN_ROWS and self.hd == 512 and self.rd == 64:
            # stock's context query path: q_norm + MXFP8 quantization in one launch, Q-B with the RoPE in its epilogue
            query = csa2_query_gemm_fp8 if fp8_query else csa2_query_gemm_rope
            q, latent = query(
                qr_raw, (self.q_norm, self.eps), self.wq_b, self.sw["q_b"], self.alpha, pos, rope_q
            )
            q = q.view(n, self.hl, self.hd)
        else:
            q = csa2_mxfp8_linear(
                qr_raw,
                self.wq_b,
                self.wq_b_s,
                self.sw["q_b"],
                self.alpha,
                0,
                (pos, rope_q.view(-1, 2, self.rd // 2), self.hd),
                (self.q_norm, self.eps),
            ).view(n, self.hl, self.hd)
        index_q = None
        if self.ratio and self.is_index_source:
            index_q = csa2_mxfp8_linear(
                qr_raw,
                self.i_wq_b,
                self.i_wq_b_s,
                self.sw["i_q_b"],
                self.alpha,
                0,
                (pos, rope_c.view(-1, 2, self.rd // 2), self.index_hd),
                None if latent is not None else (self.q_norm, self.eps),
                latent,
            ).view(n, self.index_heads, self.index_hd)
        return q, kv_raw, index_q

    def _attend(
        self, x, ctx: StepCtx, q, kv_raw, index_q, rope_q, rope_c, shared: bool, out=None
    ) -> torch.Tensor:
        """The rows' cache writes, the indexer and the sparse attention of `ctx`'s rows: the attention `[n, hl, hd]`
        bf16 (still rotated), or with `shared` the shared-bank path's projected output `[n, dim]`."""
        L, pos, n, dev = self.layer, ctx.pos, x.shape[0], x.device
        sparse = not ctx.paged or csa2_sparse_decode_supported(self.hl, dev)
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
            keep, arange_n = _chunk_consts(ctx, self.window, n)
            stage = torch.empty(n, 528, dtype=torch.uint8, device=dev)
            swa_slots, swa_target = arange_n, stage
        swa = (swa_target, swa_slots, kv_raw, (self.kv_norm, self.eps), (pos, rope_q, self.rd))
        main = index = None
        if self.ratio and self.is_owner and not ctx.published:
            lat, k_raw, addr, first = self._compress(ctx, x, pos)
            main, index = self._store_specs(ctx, lat, k_raw, addr, first, rope_c)
        packed_q = csa2_store_rows(swa, main, index, index_q)
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
                self._index_topk(ctx, x, packed_q, pos)
            o = self.owner
            main_pool, main_table, topk = _pool(ctx, o, GLOBAL), _table(ctx, o, GLOBAL), ctx.topk
        if plan is not None:
            _note(self, ctx, "shared_context", "fused-rows")
            return self._shared_context(
                ctx, q, plan, swa_target, main_pool, main_table, topk, rope_q, out
            )
        _note(self, ctx, "sparse_decode" if sparse else "direct_decode", "fused-rows")
        slots = (
            (reads[L], visible[L]) if ctx.paged else (None, _chunk_visible(ctx, self.ratio))
        )  # window rows / visible length
        if sparse:
            pad = (
                64 - self.hl
            )  # zero heads (and sinks) fill the kernel's 64; their rows are dropped
            q64 = q if not pad else torch.nn.functional.pad(q, (0, 0, 0, pad))
            sink = self.attn_sink if not pad else torch.nn.functional.pad(self.attn_sink, (0, pad))
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
                req_cs=None if ctx.paged else _req_cs(ctx),
            )
            if pad:
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
            )
        return att

    def _output(
        self, att: torch.Tensor, pos, rope_q, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """The output projection of the attention `[n, hl, hd]` (rotated): `[n, dim]` (into `out` when given)."""
        n, dev = att.shape[0], att.device
        if self.tp == 1 or not self.reduce_output:
            # inverse RoPE + grouped o_a + o_b over an MXFP8 latent: stock quantizes O-B's input in the O-A epilogue
            # when no TP reduction follows (`_project_output_mxfp8`), which is this target's attention-DP case
            return (
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
                    out=out,
                )
                if n
                else att.new_empty(0, self.dim)
            )
        # with a TP reduction stock keeps the latent bf16
        csa2_rope_inplace(att, pos, rope_q, inverse=True)
        a = att.view(n, self.gl, -1)
        wo = self._wo_a_dense().view(self.gl, self.o_rank, -1)
        mid = torch.empty(n, self.gl, self.o_rank, dtype=a.dtype, device=dev)
        bmm_out(a.transpose(0, 1), wo.transpose(1, 2), mid.transpose(0, 1))
        out = csa2_mxfp8_linear(mid.view(n, -1), self.wo_b, self.wo_b_s, self.sw["o_b"], self.alpha)
        ws = len(self.group)
        rows = -(-n // ws) * ws
        outp = out if rows == n else torch.nn.functional.pad(out, [0, 0, 0, rows - n])
        red = cat_allgather(
            cat_reducescatter(outp.contiguous(), None, self.group), None, self.group
        )
        return red if rows == n else torch.split(red, [n, rows - n], 0)[0]

    def _shared_context(
        self, ctx, q, plan, local, main_pool, main_table, topk, rope_q, out=None
    ) -> torch.Tensor:
        """Stock's context attention (`forward_fused_context` with `output_fp8`) over the step's shared bank: stage this
        layer's rows (compacting only on the first layer of a KV owner and index source), attend with the FP8 context
        kernel, which writes the O-A input rotated back and quantized per 128 channels, then O-A and O-B."""
        L, pos = self.layer, ctx.pos
        key = (main_pool, topk) if self.ratio else (_SWA_ONLY,)
        compact, parity = _shared_group(ctx, key)
        valid = None
        if compact:  # the window slots each query may read (the start, the replay floor, mapped history pages)
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
            data, scale, self.o_a_w, self.o_a_sf, self.wo_b, self.sw["o_b"], self.alpha, out
        )
