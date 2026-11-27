# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSA indexer's ops as the Targets call them, each on the kernel its handle holds and with
its fallback inside: `indexer_topk_buf` (where a step's selection lands), `indexer_q_side`
(rope + e4m3 pack of the indexer's query and its head weights; on a wide decode step with the
K-cache update in the same launch), `indexer_write_k` (norm, rope, fp8 pack and append of the
step's keys), `indexer_decode_logits` / `indexer_decode_topk` (the generation rows' scorer and
top-k) and `indexer_prefill_select` (the context rows' per-chunk scorer and top-k).

The handle (`build`) is built at load over the kernels `compile_at_load` compiled and again at
the first forward, once `compile_for_engine` compiled what needs the engine's values (the
K-cache update and the fused q side by pool layout, the decode top-k by MTP width, GVR).
Runtime state that outlives a step lives in an `IndexerState` the core owns and passes in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping

import torch

from tensorrt_llm._torch.attention.backends.sparse.params import use_self_sampling_gvr
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.distributed.ops import allgather
from tensorrt_llm._torch.modeling_v2.catalog._op import OpWrapper
from tensorrt_llm._torch.modules.top_k import TopK, TopKImplementation
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

from .._build import Count, try_build
from ..elementwise.cute_dsl_copy import RowCopyHandle, row_copy
from ..gemm.indexer_pre_proj import IndexerPreProjHandle
from ..gemm.projection import DECODE_MAX_ROWS, ProjectionHandle, projection
from .cute_dsl_fp8_paged_mqa_logits import ATOM_SPLIT, cute_dsl_fp8_paged_mqa_logits
from .cute_dsl_indexer_k_cache_update import build as build_k_update
from .cute_dsl_indexer_k_cache_update import cute_dsl_indexer_k_cache_update
from .cute_dsl_indexer_q_side_fused import build as build_q_side
from .cute_dsl_indexer_q_side_fused import cute_dsl_indexer_q_side_fused
from .cute_dsl_indexer_q_side_k_cache import build as build_q_side_k
from .cute_dsl_indexer_q_side_k_cache import cute_dsl_indexer_q_side_k_cache
from .cute_dsl_indexer_topk_decode import build as build_topk_decode
from .cute_dsl_indexer_topk_decode import cute_dsl_indexer_topk_decode, uses_cluster
from .cute_dsl_indexer_topk_prefill import build as build_topk_prefill
from .cute_dsl_indexer_topk_prefill import cute_dsl_indexer_topk_prefill
from .fp8_mqa_logits import fp8_mqa_logits
from .indexer_k_cache_gather_op import indexer_k_cache_gather_op
from .indexer_topk_decode import indexer_topk_decode

#: The indexer projection's columns: 128 key columns, then 32 head weights.
PROJ_COLS = 160

#: The indexer K pool's minimum bytes per token (128 e4m3 key bytes and the fp32 scale).
K_POOL_PER_TOKEN_MIN = 132

#: The decode logits width (the engine's max_seq_len) at or below which the generation rows keep
#: our decode top-k although the engine's config asks for GVR
#: (`TRTLLM_GLM52_DEC_TOPK_GVR_ABOVE`). Single-B300 CUDA graph of 20 calls (us per call, ours /
#: GVR, every set exact): GVR's tuning is frozen from the width, so on a 16384-wide engine at 192
#: rows (MTP5 x 32) it costs 5.2 / 12.5 at KV 2k / 4k against our 1.5 / 5.5, where a realtext
#: step (ISL 1k + OSL 8k) spends most of its decode; at 172032 GVR takes 8-17 (42 at 192 rows)
#: against our 58-95 at KV 160k. Only the width and the row count are fixed at capture.
DEC_TOPK_GVR_ABOVE = 16384

#: fp8_mqa_logits allocates a `[q_rows, kv_rows]` fp32 score matrix per call; prefill chunks are
#: tiled to keep it under this many elements. Context steps are not captured.
PREFILL_LOGITS_BUDGET = 1 << 24


def _tile_bounds(start: int, end: int, kv_rows: int) -> list[tuple[int, int]]:
    rows = max(1, PREFILL_LOGITS_BUDGET // max(kv_rows, 1))
    return [(a, min(a + rows, end)) for a in range(start, end, rows)]


def gvr_serves(max_seq_len: int) -> bool:
    """Whether the GVR decode top-k serves an engine of this logits width (`max_seq_len`, fixed
    at capture): wider than `DEC_TOPK_GVR_ABOVE` (`TRTLLM_GLM52_DEC_TOPK_GVR_ABOVE`)."""
    above = int(os.environ.get("TRTLLM_GLM52_DEC_TOPK_GVR_ABOVE", str(DEC_TOPK_GVR_ABOVE)))
    return max_seq_len > above


def gvr_choice(
    params, index_topk: int, *, sm_version: int, dsl_available: bool, env: str | None
) -> bool | None:
    """The decode top-k engine of the generation rows: `gvr_deterministic_topk` where they take
    the stock engine's self-sampling GVR top-k, None where they keep ours.

    Ours by default, under the engine's GVR config too: on long engines our decode top-k splits
    each long row across a CTA cluster, which matches GVR at 64k-160k KV and keeps our one-CTA
    speed at short KV, where GVR's tuning frozen from the engine width is 2-3x slower.
    `TRTLLM_GLM52_DEC_TOPK_GVR=1` (`env`) restores the stock engine's self-sampling GVR top-k
    where the stock DSA indexer would run it on this checkpoint's layers: the engine's
    `enable_heuristic_topk` with the self-sampling engine (`use_self_sampling_gvr`: its
    `use_self_sampling_topk`, CuTe DSL, sm_100 / sm_103, `index_topk` 2048 here, compress ratio
    1). The temporal-hint engine (`use_self_sampling_topk=False`) needs the previous step's
    selection as caller-owned state, which this target does not keep: ours there."""
    if env != "1" or params is None or not getattr(params, "enable_heuristic_topk", False):
        return None
    if index_topk != 2048 or not use_self_sampling_gvr(
        enable_heuristic_topk=True,
        use_self_sampling_topk=bool(getattr(params, "use_self_sampling_topk", True)),
        index_topk=index_topk,
        compress_ratio=1,
        is_cute_dsl_available=dsl_available,
        sm_version=sm_version,
    ):
        return None
    return bool(getattr(params, "gvr_deterministic_topk", False))


def build_gvr(params, index_topk: int) -> TopK | None:
    """The stock engine's GVR decode top-k module where `gvr_choice` takes it for the engine's
    sparse params (layer 0's), else None (ours)."""
    det = gvr_choice(
        params,
        index_topk,
        sm_version=get_sm_version(),
        dsl_available=IS_CUTLASS_DSL_AVAILABLE,
        env=os.environ.get("TRTLLM_GLM52_DEC_TOPK_GVR"),
    )
    if det is None:
        return None
    return TopK(
        index_topk,
        decode_implementation=TopKImplementation.CUTE_DSL_GVR,
        gvr_self_sampling=True,
        gvr_deterministic=det,
    )


class TopkPrefillCache:
    """The prefill top-k build, widened in place by `indexer_prefill_select` when a K block is
    wider than any built so far (a long cached prefix, or packed requests): the width is a
    step's, not known at load, and context steps are never captured, so the widening compiles
    there. Owned by the core, shared by every handle built over it."""

    def __init__(self, compiled, top_k: int) -> None:
        self.compiled = compiled
        self.top_k = top_k


@dataclass
class IndexerState:
    """What the indexer carries across steps: `k_checked`, whether the K-cache update's first
    (eager) call validated the slot mappings. The validation reduces them to a host bool, a
    sync that invalidates a capture, so it runs once; the fused q side + K update launches
    only after it."""

    k_checked: bool = False


def compile_at_load(
    dev: int, top_k: int, is_neox: bool, unsplit: bool, max_columns: int
) -> SimpleNamespace:
    """The indexer kernels that need no engine value: the q side (and with `unsplit`, its
    pitch-160 launcher for the prefill GEMM's column views) and the prefill top-k up to
    `max_columns`. `compile_for_engine` adds the rest at the first forward."""
    return SimpleNamespace(
        q_side=build_q_side(dev, is_neox=is_neox),
        q_side_p160=(build_q_side(dev, is_neox=is_neox, wq_pitch=PROJ_COLS) if unsplit else None),
        topk_prefill=TopkPrefillCache(
            build_topk_prefill(dev, top_k=top_k, max_columns=max_columns), top_k
        ),
        k_update={},
        q_side_k={},
        topk_decode={},
        gvr=None,
        unsplit=unsplit,
    )


def compile_for_engine(
    kernels: SimpleNamespace,
    md,
    *,
    top_k: int,
    is_neox: bool,
    widths: tuple[int, ...],
    q_side_k: bool,
    gvr: TopK | None,
) -> None:
    """The indexer kernels keyed by engine values, compiled at the first forward (eager, ahead
    of any capture) into `kernels`:

    - the K-cache update for both pool layouts (the trunk's per-layer views are strided and a
      separate draft pool's may not be; a layout first met inside a capture cannot compile
      there), with the fp32-key launchers where the prefill projection is unsplit;
    - with `q_side_k` (`TRTLLM_GLM52_WIDE_IX_QK` on an engine with the wide decode cells) the
      q side + K update in one launch, both layouts (empty: two launches);
    - the decode top-k at the 1-row width and each of `widths` (the MTP widths), against the
      engine's `max_seq_len` columns. A width without one takes the trtllm op: the kernel
      needs more columns than `top_k`, and a failed build keeps that width there too;
    - `gvr` (the stock engine's GVR top-k, `build_gvr`) only where it serves this engine's
      width (`gvr_serves`), its launchers warmed for every row count a decode step carries."""
    dev = torch.cuda.current_device()
    k_pool = md.kv_cache_manager.get_indexer_k_cache_buffers(0)
    assert (
        k_pool.dtype == torch.uint8
        and k_pool.ndim == 4
        and k_pool.shape[2] == 1
        and k_pool.shape[3] >= K_POOL_PER_TOKEN_MIN
    ), f"the fused indexer k write needs a [blocks, block, 1, >=132] uint8 pool: {k_pool.shape}"
    kernels.k_update = {
        contiguous: build_k_update(
            dev, is_neox=is_neox, cache_contiguous=contiguous, f32_input=kernels.unsplit
        )
        for contiguous in (False, True)
    }
    if q_side_k:
        kernels.q_side_k = (
            try_build(
                "indexer q side + K update",
                lambda: {
                    contiguous: build_q_side_k(dev, is_neox=is_neox, cache_contiguous=contiguous)
                    for contiguous in (False, True)
                },
            )
            or {}
        )
    max_columns = int(md.max_seq_len)
    kernels.topk_decode = {}
    if max_columns > top_k:
        for next_n in (1, *widths):
            try:
                kernels.topk_decode[next_n] = build_topk_decode(
                    dev, max_columns=max_columns, top_k=top_k, next_n=next_n
                )
            except Exception as e:  # noqa: BLE001 -- that width takes the trtllm op
                logger.warning(
                    f"modeling_v2 GLM-5.2: indexer top-k decode build failed for next_n={next_n} "
                    f"({e}); using the trtllm op"
                )
    kernels.gvr = gvr if gvr is not None and gvr_serves(max_columns) else None
    if kernels.gvr is not None:
        _warm_gvr(kernels.gvr, md, top_k, widths)


def _warm_gvr(gvr: TopK, md, top_k: int, widths: tuple[int, ...]) -> None:
    """Compile the GVR decode top-k's launchers for every row count a decode step of this
    engine can carry (1..max_batch_size sequences at each generation width), as the engine's
    `warmup_selfsampling_topk` does for its own `next_n`: a launcher is keyed by the exact row
    count and the logits row stride, and one first met inside a capture cannot compile there
    (that graph then keeps ours, `indexer_decode_topk`). The scorer's rows are the DSL
    paged-MQA arena's, `max_seq_len` rounded up to 256 columns."""
    from tensorrt_llm._torch.kernels.blackwell.top_k import (
        gvr_topk_decode_self_sampling_host as gvr_host,
    )

    seqs = int(md.max_num_requests)
    for next_n in (1, *widths):
        gvr_host.warmup_varlen(
            top_k,
            int(md.max_seq_len),
            compress_ratio=1,
            next_n=next_n,
            num_rows_list=tuple(next_n * b for b in range(1, seqs + 1)),
            row_stride=(int(md.max_seq_len) + 255) // 256 * 256,
            deterministic=gvr.gvr_deterministic,
        )


@dataclass(frozen=True)
class IndexerHandle:
    """What the indexer ops launch. `pre_proj`: `indexer_pre_proj`'s handle; `ix_qb`: the q
    projection's (where `q_b_ix_wq_b` did not write it); `q_side` / `q_side_p160`: the q side,
    on contiguous / pitch-160 head weights; `q_side_k`: the q side + K update by pool
    contiguity (empty: two launches); `k_update`: the K-cache update by pool contiguity;
    `topk_decode`: our decode top-k by width; `gvr`: the GVR decode top-k (None: ours);
    `topk_prefill`: the prefill top-k (`TopkPrefillCache`); `cos_sin`: the indexer's rope
    table (`[positions, rope_dim]` fp32); `max_ctx_len`: the decode scorer's column count, an
    engine constant (a replay whose sequences outgrew a batch maximum would read unwritten
    columns)."""

    pre_proj: IndexerPreProjHandle
    ix_qb: ProjectionHandle
    q_side: Any
    q_side_p160: Any
    q_side_k: Mapping[bool, Any]
    k_update: Mapping[bool, Any]
    topk_decode: Mapping[int, Any]
    gvr: Any
    topk_prefill: TopkPrefillCache
    cos_sin: torch.Tensor | None
    weight_scale: float
    is_neox: bool
    rope_dim: int
    heads: int
    dim: int
    top_k: int
    max_ctx_len: int
    copy: RowCopyHandle
    count: Count


def build(kernels: SimpleNamespace, **fields) -> IndexerHandle:
    """The handle over `compile_at_load` / `compile_for_engine`'s kernels and the indexer's
    constants (`fields`: the handle's other fields). Compiles nothing."""
    return IndexerHandle(
        q_side=kernels.q_side,
        q_side_p160=kernels.q_side_p160,
        q_side_k=dict(kernels.q_side_k),
        k_update=dict(kernels.k_update),
        topk_decode=dict(kernels.topk_decode),
        gvr=kernels.gvr,
        topk_prefill=kernels.topk_prefill,
        **fields,
    )


class IndexerTopkBuf(OpWrapper):
    """Where a step's selection of `rows` rows (`tc` context rows first) lands. A phase the
    engine marked skippable (every sequence shorter than index_topk) takes the dense causal
    index the engine wrote into `md.topk_indices_buffer`: the whole buffer where both phases
    skip, else that phase's rows copied into a new buffer. A branch frozen into a graph, safe
    because short and long steps capture separately."""

    def __call__(
        self, md, rows: int, tc: int, skip_ctx: bool, skip_gen: bool, handle: IndexerHandle
    ) -> torch.Tensor:
        if skip_ctx and skip_gen:
            return md.topk_indices_buffer[:rows]
        # `topk_indices_buffer` exists only while the engine's indexer skip is on.
        buf = torch.empty(
            [rows, handle.top_k], dtype=torch.int32, device=torch.cuda.current_device()
        )
        if skip_ctx and tc:
            row_copy(buf[:tc], md.topk_indices_buffer[:tc], handle.copy)
        if skip_gen and rows > tc:
            row_copy(buf[tc:], md.topk_indices_buffer[tc:rows], handle.copy)
        return buf

    def reference(
        self, md, rows: int, tc: int, skip_ctx: bool, skip_gen: bool, handle: IndexerHandle
    ) -> torch.Tensor:
        """The skipped phases' rows of the engine's buffer (the selected rows are unwritten:
        zeros here)."""
        dev = md.topk_indices_buffer.device
        buf = torch.zeros([rows, handle.top_k], dtype=torch.int32, device=dev)
        if skip_ctx and tc:
            buf[:tc] = md.topk_indices_buffer[:tc]
        if skip_gen and rows > tc:
            buf[tc:] = md.topk_indices_buffer[tc:rows]
        return buf


def _q_side_k_cell(ph, kb: torch.Tensor, k_pool, rows: int, state: IndexerState, h):
    """The q side + K update launcher for this call (`TRTLLM_GLM52_WIDE_IX_QK`), or None: on a
    wide decode step's rows (above 64), once the K update's first eager call validated the
    slot mappings (`state.k_checked`), with 8-byte aligned contiguous keys (its only key
    loads)."""
    if not h.q_side_k or not state.k_checked:
        return None
    if not (ph.wide and rows > DECODE_MAX_ROWS):
        return None
    if not kb.is_contiguous() or kb.data_ptr() % 8:
        return None
    return h.q_side_k[k_pool.is_contiguous()]


class IndexerQSide(OpWrapper):
    """The indexer's query side: `(q_fp8 [rows, heads, dim] e4m3, weights [rows, heads] fp32,
    kb)`. `ixq`: the indexer's q projection where `q_b_ix_wq_b` wrote it, else (None) the
    projection of `q_a` by `w_qb` runs here. `wq`: the head weights from `indexer_pre_proj`
    (pitch 160: its unsplit prefill rows, on the pitch-160 launcher). On a wide decode step
    where `_q_side_k_cell` serves (counted `ix_qk`) the same launch also writes this step's
    keys `kb` (normed by `kn_w` / `kn_b`) into the layer's K cache `k_pool`, and `kb` comes
    back None (written); else `kb` comes back as it came, for `indexer_write_k`."""

    def __call__(
        self,
        ph,
        q_a: torch.Tensor,
        ixq,
        w_qb: torch.Tensor,
        pos: torch.Tensor,
        wq: torch.Tensor,
        kb: torch.Tensor,
        kn_w: torch.Tensor,
        kn_b: torch.Tensor,
        md,
        k_pool: torch.Tensor,
        state: IndexerState,
        handle: IndexerHandle,
    ):
        h = handle
        rows = int(q_a.shape[0])
        if ixq is None:
            ixq = projection(ph, q_a, w_qb, h.ix_qb)
        q = torch.reshape(ixq, [rows, h.heads, h.dim])
        q_fp8 = torch.empty([rows, h.heads, h.dim], dtype=torch.float8_e4m3fn, device=q_a.device)
        weights = torch.empty([rows, h.heads], dtype=torch.float32, device=q_a.device)
        qk = _q_side_k_cell(ph, kb, k_pool, rows, state, h)
        if qk is not None:
            h.count("ix_qk", True)
            cute_dsl_indexer_q_side_k_cache(
                q, pos, h.cos_sin, wq, h.weight_scale, q_fp8, weights, kb, kn_w, kn_b, k_pool,
                md.slot_mapping_fp8, md.slot_mapping_scale, is_neox=h.is_neox, compiled=qk,
            )  # fmt: skip
            return q_fp8, weights, None
        qside = h.q_side if wq.stride(0) == h.heads else h.q_side_p160
        cute_dsl_indexer_q_side_fused(
            q, pos, h.cos_sin, wq, h.weight_scale, q_fp8, weights, is_neox=h.is_neox,
            compiled=qside,
        )  # fmt: skip
        return q_fp8, weights, kb

    def reference(
        self,
        ph,
        q_a: torch.Tensor,
        ixq,
        w_qb: torch.Tensor,
        pos: torch.Tensor,
        wq: torch.Tensor,
        kb: torch.Tensor,
        kn_w: torch.Tensor,
        kn_b: torch.Tensor,
        md,
        k_pool: torch.Tensor,
        state: IndexerState,
        handle: IndexerHandle,
    ):
        """The q side's reference (`cute_dsl_indexer_q_side_fused.reference`) over the q
        projection in fp32; the keys are `indexer_write_k`'s (`kb` returned unwritten)."""
        h = handle
        rows = int(q_a.shape[0])
        if ixq is None:
            ixq = (q_a.float() @ w_qb.float()).to(q_a.dtype)
        q = torch.reshape(ixq, [rows, h.heads, h.dim])
        q_fp8 = torch.empty([rows, h.heads, h.dim], dtype=torch.float8_e4m3fn, device=q_a.device)
        weights = torch.empty([rows, h.heads], dtype=torch.float32, device=q_a.device)
        q_fp8, weights = cute_dsl_indexer_q_side_fused.reference(
            q, pos, h.cos_sin, wq, h.weight_scale, q_fp8, weights, is_neox=h.is_neox,
            compiled=h.q_side,
        )  # fmt: skip
        return q_fp8, weights, kb


class IndexerWriteK(OpWrapper):
    """Norm (`kn_w` / `kn_b`), rope, fp8-pack and append this step's keys `kb` into the layer's
    K cache `k_pool` in one launch; `kb` None: `indexer_q_side` already wrote them. The first
    call validates the slot mappings (a host sync, so only on that eager call) and records it
    in `state`."""

    def __call__(
        self,
        md,
        k_pool: torch.Tensor,
        kb,
        kn_w: torch.Tensor,
        kn_b: torch.Tensor,
        pos: torch.Tensor,
        state: IndexerState,
        handle: IndexerHandle,
    ) -> None:
        if kb is None:
            return
        h = handle
        cute_dsl_indexer_k_cache_update(
            kb, kn_w, kn_b, pos, h.cos_sin.view(-1, 2, h.rope_dim // 2), k_pool,
            md.slot_mapping_fp8, md.slot_mapping_scale, int(kb.shape[0]), is_neox=h.is_neox,
            compiled=h.k_update[k_pool.is_contiguous()], validate=not state.k_checked,
        )  # fmt: skip
        state.k_checked = True

    def reference(
        self,
        md,
        k_pool: torch.Tensor,
        kb,
        kn_w: torch.Tensor,
        kn_b: torch.Tensor,
        pos: torch.Tensor,
        state: IndexerState,
        handle: IndexerHandle,
    ):
        """`cute_dsl_indexer_k_cache_update.reference` over the same keys (None: nothing)."""
        if kb is None:
            return None
        h = handle
        return cute_dsl_indexer_k_cache_update.reference(
            kb, kn_w, kn_b, pos, h.cos_sin.view(-1, 2, h.rope_dim // 2), k_pool,
            md.slot_mapping_fp8, md.slot_mapping_scale, int(kb.shape[0]), is_neox=h.is_neox,
            compiled=h.k_update[k_pool.is_contiguous()],
        )  # fmt: skip


class IndexerDecodeLogits(OpWrapper):
    """The generation rows' scores against the paged K cache: `[seqs * gen_p, columns]` fp32
    for `q_fp8`'s `seqs * gen_p` rows, `ctx` the context sequences ahead of them in the batch.
    Where the width has an atom split (MTP5: 6 rows of a sequence as two pseudo-sequences of 3,
    since the kernel spills at 6; the same logits rows, bit for bit) its metadata is built once
    per step and kept in `split_memo` (the Target's, one per step; None: per call), counted
    `mqa_atom_split`: lengths, table and schedule depend only on the step."""

    def __call__(
        self,
        md,
        k_pool: torch.Tensor,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        ctx: int,
        gen_p: int,
        split_memo: dict | None,
        handle: IndexerHandle,
    ) -> torch.Tensor:
        h = handle
        seqs = int(q_fp8.shape[0]) // gen_p
        # Cached plus all `gen_p` of this step's tokens; the top-k takes the same vector, which
        # keeps the selector's causal bound inside the columns the scorer wrote.
        kv_lens = md.kv_lens_cuda_runtime[ctx : ctx + seqs]
        table = md.indexer_k_cache_block_offsets[ctx : ctx + seqs]
        atom = ATOM_SPLIT.get(gen_p)
        if atom is None:
            return cute_dsl_fp8_paged_mqa_logits(
                torch.reshape(q_fp8, [seqs, gen_p, h.heads, h.dim]), k_pool, weights, kv_lens,
                table, md.scheduler_metadata_buffer, h.max_ctx_len,
            )  # fmt: skip
        factor = gen_p // atom
        key = (ctx, seqs, gen_p)
        meta = None if split_memo is None else split_memo.get(key)
        if meta is None:
            meta = cute_dsl_fp8_paged_mqa_logits.split_metadata(kv_lens, table, factor)
            if split_memo is not None:
                split_memo[key] = meta
        h.count("mqa_atom_split", True)
        return cute_dsl_fp8_paged_mqa_logits(
            torch.reshape(q_fp8, [seqs * factor, atom, h.heads, h.dim]), k_pool, weights, *meta,
            h.max_ctx_len,
        )  # fmt: skip

    def reference(
        self,
        md,
        k_pool: torch.Tensor,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        ctx: int,
        gen_p: int,
        split_memo: dict | None,
        handle: IndexerHandle,
    ) -> torch.Tensor:
        """The scorer's reference without the atom split (`split_memo` ignored)."""
        h = handle
        seqs = int(q_fp8.shape[0]) // gen_p
        return cute_dsl_fp8_paged_mqa_logits.reference(
            torch.reshape(q_fp8, [seqs, gen_p, h.heads, h.dim]), k_pool, weights,
            md.kv_lens_cuda_runtime[ctx : ctx + seqs],
            md.indexer_k_cache_block_offsets[ctx : ctx + seqs], md.scheduler_metadata_buffer,
            h.max_ctx_len,
        )  # fmt: skip


def _gvr_launch(md, gvr, logits, kv_lens, topk_buf, gen_p: int, h: IndexerHandle) -> bool:
    """The generation rows' top-k on the stock engine's self-sampling GVR engine, the call the
    stock DSA indexer makes (`Indexer.sparse_attn_indexer`'s decode `c.top_k(...)`). False
    where it did not launch: a launcher missing inside a capture (none JIT-compiles there),
    which leaves that graph on ours.

    Row `b * gen_p + t` selects among columns `[0, kv_lens[b] - gen_p + t + 1)` (`kv_lens`
    counts this step's tokens, so the bound is the causal one our kernel uses); with `n` that
    many, it writes the exact top-`min(n, 2048)` set (ties at the k-th score kept in an
    unspecified order unless `gvr_deterministic_topk`), and a row with `n <= 2048` the identity
    `0..n-1`, each followed by -1: the valid prefix the sparse MLA decode reads. Tuning is
    frozen from `max_seq_len` (the scorer's width), with no host reads, so the launch is CUDA
    graph replay safe. The engine's `radix_aux_*` scratch serves only the stock module's own
    fall-through (logits outside the engine's format gate)."""
    try:
        gvr(
            logits,
            topk_buf,
            is_prefill=False,
            sequence_lengths=kv_lens,
            scan_lengths=kv_lens,
            next_n=gen_p,
            max_seq_len=h.max_ctx_len,
            radix_aux_indices=md.radix_aux_indices,
            radix_aux_logits=md.radix_aux_logits,
        )
    except RuntimeError as e:
        if not torch.cuda.is_current_stream_capturing():
            raise
        # Raised before any launch, so the capture continues on ours.
        logger.warning(
            f"modeling_v2 GLM-5.2: GVR decode top-k not compiled for {kv_lens.shape[0]} x "
            f"{gen_p} rows inside a capture ({e}); this graph keeps our decode top-k"
        )
        return False
    h.count("ix_topk_dec_gvr", True)
    return True


class IndexerDecodeTopk(OpWrapper):
    """The generation rows' `index_topk` selection from `logits` into `topk_buf` (its rows are
    `seqs * gen_p`; `ctx` the context sequences ahead of them). On the GVR engine where the
    handle has it (`gvr`, counted `ix_topk_dec_gvr`; a graph whose launcher is missing keeps
    ours), else on our decode top-k for the width (counted `ix_topk_dec`, and
    `ix_topk_dec_cluster` where it splits rows across a CTA cluster), else (a width without
    one) the trtllm op."""

    def __call__(
        self,
        md,
        logits: torch.Tensor,
        topk_buf: torch.Tensor,
        ctx: int,
        gen_p: int,
        handle: IndexerHandle,
    ) -> None:
        h = handle
        rows = int(topk_buf.shape[0])
        kv_lens = md.kv_lens_cuda_runtime[ctx : ctx + rows // gen_p]
        if h.gvr is not None and _gvr_launch(md, h.gvr, logits, kv_lens, topk_buf, gen_p, h):
            return
        compiled = h.topk_decode.get(gen_p)
        h.count("ix_topk_dec", compiled is not None)
        if compiled is None:
            indexer_topk_decode(
                logits, kv_lens, topk_buf, gen_p, h.top_k,
                radix_aux_indices=md.radix_aux_indices, radix_aux_logits=md.radix_aux_logits,
            )  # fmt: skip
            return
        # `seqs * gen_p` logits rows on both scorer paths (the atom split reshapes rows only).
        h.count("ix_topk_dec_cluster", uses_cluster(compiled, rows))
        cute_dsl_indexer_topk_decode(
            logits, kv_lens, topk_buf, top_k=h.top_k, next_n=gen_p, compiled=compiled
        )

    def reference(
        self,
        md,
        logits: torch.Tensor,
        topk_buf: torch.Tensor,
        ctx: int,
        gen_p: int,
        handle: IndexerHandle,
    ) -> torch.Tensor:
        """Our decode top-k's reference for the width (the GVR engine's set is the same up to
        ties at the k-th score)."""
        h = handle
        kv_lens = md.kv_lens_cuda_runtime[ctx : ctx + int(topk_buf.shape[0]) // gen_p]
        return cute_dsl_indexer_topk_decode.reference(
            logits, kv_lens, topk_buf, top_k=h.top_k, next_n=gen_p,
            compiled=h.topk_decode.get(gen_p),
        )  # fmt: skip


class IndexerPrefillSelect(OpWrapper):
    """The context rows' selection into `topk_buf`: the engine's indexer prefill chunks, each
    scored against its gathered K block and selected per tile (`PREFILL_LOGITS_BUDGET`). With
    MLA chunked prefill and at least `q_split_threshold` context queries, the engine plans the
    chunks for this rank's share of the queries only (`indexer_prefill_q_split`); the other
    ranks' top-k rows are gathered after the loop, as the engine's own indexer does. A K block
    wider than any built so far widens the prefill top-k build (`TopkPrefillCache`)."""

    def __call__(
        self,
        md,
        k_pool: torch.Tensor,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        topk_buf: torch.Tensor,
        handle: IndexerHandle,
    ) -> None:
        h = handle
        cache = h.topk_prefill
        q_split = getattr(md, "indexer_prefill_q_split", None)
        for chunk in md.indexer_prefill_chunks:
            n_k = chunk.k_token_end - chunk.k_token_start
            if chunk.token_end <= chunk.token_start:
                continue
            if n_k <= 0:
                # No compressed KV for these queries: no valid index, as the engine.
                topk_buf[chunk.token_start : chunk.token_end].fill_(-1)
                continue
            if n_k > cache.compiled.max_columns:
                cache.compiled = build_topk_prefill(
                    torch.cuda.current_device(), top_k=cache.top_k, max_columns=n_k
                )
            ck, cs = indexer_k_cache_gather_op(
                k_pool, md.slot_mapping_fp8_fullkv, md.slot_mapping_scale_fullkv,
                chunk.k_token_start, n_k, h.dim,
            )  # fmt: skip
            cs = torch.reshape(cs, [-1])
            # The window vectors hold one entry per query of this chunk: they are indexed by
            # position within the chunk, not by context token.
            t0 = chunk.token_start
            assert chunk.cu_seqlen_ks.shape[0] == chunk.token_end - t0
            for lo, hi in _tile_bounds(t0, chunk.token_end, n_k):
                ks = chunk.cu_seqlen_ks[lo - t0 : hi - t0]
                ke = chunk.cu_seqlen_ke[lo - t0 : hi - t0]
                logits = fp8_mqa_logits(q_fp8[lo:hi], (ck, cs), weights[lo:hi], ks, ke)
                cute_dsl_indexer_topk_prefill(
                    logits, ks, ke, topk_buf[lo:hi], top_k=h.top_k, compiled=cache.compiled
                )
        if q_split is not None:
            q_lo, q_hi, q_sizes = q_split
            topk_buf[: sum(q_sizes)] = allgather(
                topk_buf[q_lo:q_hi], md.mapping, dim=0, sizes=q_sizes
            )

    def reference(
        self,
        md,
        k_pool: torch.Tensor,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        topk_buf: torch.Tensor,
        handle: IndexerHandle,
    ) -> torch.Tensor:
        """The same chunks and tiles on the prefill top-k's reference (single rank: no
        q split)."""
        h = handle
        out = topk_buf.clone()
        for chunk in md.indexer_prefill_chunks:
            n_k = chunk.k_token_end - chunk.k_token_start
            if chunk.token_end <= chunk.token_start:
                continue
            if n_k <= 0:
                out[chunk.token_start : chunk.token_end].fill_(-1)
                continue
            ck, cs = indexer_k_cache_gather_op(
                k_pool, md.slot_mapping_fp8_fullkv, md.slot_mapping_scale_fullkv,
                chunk.k_token_start, n_k, h.dim,
            )  # fmt: skip
            cs = torch.reshape(cs, [-1])
            t0, t1 = chunk.token_start, chunk.token_end
            ks, ke = chunk.cu_seqlen_ks, chunk.cu_seqlen_ke
            logits = fp8_mqa_logits(q_fp8[t0:t1], (ck, cs), weights[t0:t1], ks, ke)
            out[t0:t1] = cute_dsl_indexer_topk_prefill.reference(
                logits, ks, ke, out[t0:t1], top_k=h.top_k, compiled=h.topk_prefill.compiled
            )
        return out


indexer_topk_buf = IndexerTopkBuf()
indexer_q_side = IndexerQSide()
indexer_write_k = IndexerWriteK()
indexer_decode_logits = IndexerDecodeLogits()
indexer_decode_topk = IndexerDecodeTopk()
indexer_prefill_select = IndexerPrefillSelect()
