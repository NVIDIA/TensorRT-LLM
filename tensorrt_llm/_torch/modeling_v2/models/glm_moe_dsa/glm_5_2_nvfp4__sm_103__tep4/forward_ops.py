# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ops the glm-5.2-nvfp4 / sm_103 / tep4 Targets call.

A Target (`prefill.py`, `decode.py`, `wide_decode.py`) is the per-step data flow only: it
calls these ops with the step's tensors and the layer's weights, and never sees a kernel
handle. Each op picks its kernel by shape -- the compiled cell at the row count, the next
cell up at the exact rows, the prefill kernel, or the stock op where no kernel serves -- and
counts the route it took (`route_report`). The ops are grouped by what they compute
(`Rows`, `Comm`, `Gemm`, `Attn`, `Indexer`, `Mlp`, `Moe`, `Mtp`); one instance of each lives
on the ModelCore as `core.op`, which builds the kernels they launch at load time and reads
the weights they take.

The row counts a step's rows map onto depend on which Target runs it, which the Target
states as a `Phase`: the wide decode cells for the MTP3 / MTP5 decode steps above 64 rows,
and the next cell up for a count without a cell of its own.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import NamedTuple

import torch

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.distributed.ops import allgather
from tensorrt_llm._torch.modeling_v2.catalog.activation.flashinfer_silu_and_mul import (  # noqa: E501
    flashinfer_silu_and_mul,
)
from tensorrt_llm._torch.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch.modeling_v2.catalog.norm.flashinfer_rmsnorm import flashinfer_rmsnorm
from tensorrt_llm._torch.modeling_v2.catalog.quantization.fp4_quantize import fp4_quantize
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger

from .ops.activation import cute_dsl_swiglu_mxf8 as _swiglu8_op
from .ops.activation import cute_dsl_swiglu_rows as _swiglu_rows_op
from .ops.activation.cute_dsl_dense_swiglu import cute_dsl_dense_swiglu
from .ops.activation.cute_dsl_swiglu_mxf8 import cute_dsl_swiglu_mxf8
from .ops.activation.cute_dsl_swiglu_rows import cute_dsl_swiglu_rows
from .ops.attention import cute_dsl_mha_kv_gather_mxf8 as _kv_gather8_op
from .ops.attention import cute_dsl_mha_prefill as _mha_pre_op
from .ops.attention import cute_dsl_mla_rope_append_prefill as _ctx_rope_op
from .ops.attention import cute_dsl_sparse_mla_decode as _fmha_dec_op
from .ops.attention import cute_dsl_sparse_mla_prefill as _fmha_pre_op
from .ops.attention.convert_req_index_to_global import convert_req_index_to_global
from .ops.attention.cute_dsl_fp8_paged_mqa_logits import ATOM_SPLIT as _MQA_ATOM_SPLIT
from .ops.attention.cute_dsl_fp8_paged_mqa_logits import cute_dsl_fp8_paged_mqa_logits
from .ops.attention.cute_dsl_indexer_k_cache_update import cute_dsl_indexer_k_cache_update
from .ops.attention.cute_dsl_indexer_q_side_fused import cute_dsl_indexer_q_side_fused
from .ops.attention.cute_dsl_indexer_q_side_k_cache import cute_dsl_indexer_q_side_k_cache
from .ops.attention.cute_dsl_indexer_topk_decode import cute_dsl_indexer_topk_decode
from .ops.attention.cute_dsl_indexer_topk_decode import uses_cluster as ix_topk_decode_uses_cluster
from .ops.attention.cute_dsl_indexer_topk_prefill import build as build_ix_topk_prefill
from .ops.attention.cute_dsl_indexer_topk_prefill import cute_dsl_indexer_topk_prefill
from .ops.attention.cute_dsl_mha_kv_gather_mxf8 import cute_dsl_mha_kv_gather_mxf8
from .ops.attention.cute_dsl_mha_prefill import cute_dsl_mha_prefill
from .ops.attention.cute_dsl_mha_prefix_gather import cute_dsl_mha_prefix_gather
from .ops.attention.cute_dsl_mla_rope_append_prefill import cute_dsl_mla_rope_append_prefill
from .ops.attention.cute_dsl_mla_rope_generation import cute_dsl_mla_rope_generation
from .ops.attention.cute_dsl_req_index_to_base import cute_dsl_req_index_to_base
from .ops.attention.cute_dsl_req_index_to_global import cute_dsl_req_index_to_global
from .ops.attention.cute_dsl_sparse_mla_decode import (
    INDEX_KIND_BASE_WITH_LAYER_OFFSET,
    INDEX_KIND_GLOBAL,
    cute_dsl_sparse_mla_decode,
)
from .ops.attention.cute_dsl_sparse_mla_decode_union import cute_dsl_sparse_mla_decode_union
from .ops.attention.cute_dsl_sparse_mla_prefill import cute_dsl_sparse_mla_prefill
from .ops.attention.fp8_mqa_logits import fp8_mqa_logits
from .ops.attention.indexer_k_cache_gather_op import indexer_k_cache_gather_op
from .ops.attention.indexer_topk_decode import indexer_topk_decode
from .ops.attention.mla_rope_append_paged_kv_assign_q import mla_rope_append_paged_kv_assign_q
from .ops.attention.mla_rope_generation import mla_rope_generation
from .ops.comm._lamport import TWOSHOT_ABOVE as _AR_TWOSHOT_ABOVE
from .ops.comm.cute_dsl_allreduce import cute_dsl_allreduce
from .ops.comm.cute_dsl_allreduce_add_rmsnorm import cute_dsl_allreduce_add_rmsnorm
from .ops.comm.cute_dsl_allreduce_add_rmsnorm_quant_nvfp4 import (
    cute_dsl_allreduce_add_rmsnorm_quant_nvfp4,
)
from .ops.elementwise import cute_dsl_copy as _copy_op
from .ops.elementwise import cute_dsl_hidden_add as _hidden_add_op
from .ops.elementwise.cute_dsl_copy import cute_dsl_copy
from .ops.elementwise.cute_dsl_hidden_add import cute_dsl_hidden_add
from .ops.elementwise.cute_dsl_rows_select import cute_dsl_rows_select
from .ops.embedding.cute_dsl_embedding import cute_dsl_embedding
from .ops.gemm import cute_dsl_absorbed_bmm as _absorbed_bmm_op
from .ops.gemm import cute_dsl_dense_up_swiglu_mxf8 as _dense_up8_op
from .ops.gemm import cute_dsl_indexer_pre_proj_split as _ix_pre_split_op
from .ops.gemm import cute_dsl_mtp_eh_proj as _mtp_eh
from .ops.gemm import cute_dsl_projection as _projection_op
from .ops.gemm import cute_dsl_projection_mxf8 as _mxf8_op
from .ops.gemm import cute_dsl_projection_mxf8_decode as _dec8_op
from .ops.gemm import cute_dsl_q_b_ix_wq_b as _qbix_op
from .ops.gemm import cute_dsl_qkva_ix_fold as _ix_fold_op
from .ops.gemm.bmm_out import bmm_out
from .ops.gemm.cute_dsl_absorbed_bmm import cute_dsl_absorbed_bmm
from .ops.gemm.cute_dsl_dense_up_swiglu_mxf8 import cute_dsl_dense_up_swiglu_mxf8
from .ops.gemm.cute_dsl_indexer_pre_proj_split import cute_dsl_indexer_pre_proj_split
from .ops.gemm.cute_dsl_mtp_eh_proj import cute_dsl_mtp_eh_proj
from .ops.gemm.cute_dsl_projection import cute_dsl_projection
from .ops.gemm.cute_dsl_projection_mxf8 import cute_dsl_projection_mxf8
from .ops.gemm.cute_dsl_projection_mxf8_decode import cute_dsl_projection_mxf8_decode
from .ops.gemm.cute_dsl_q_b_ix_wq_b import cute_dsl_q_b_ix_wq_b
from .ops.gemm.cute_dsl_qkva_ix_fold import cute_dsl_qkva_ix_fold
from .ops.gemm.cute_dsl_shared_gate_up_swiglu import cute_dsl_shared_gate_up_swiglu
from .ops.gemm.cute_dsl_shared_up_swiglu_mxf8 import cute_dsl_shared_up_swiglu_mxf8
from .ops.moe import cute_dsl_moe_prefill_topk_routing_rep as _moe_topk_pre_rep_op
from .ops.moe import cute_dsl_moe_router as _moe_router_op
from .ops.moe import cute_dsl_moe_routing as _moe_routing_op
from .ops.moe import cute_dsl_moe_sigmoid_topk as _topk_rows_op
from .ops.moe import cute_dsl_moe_topk_routing as _moe_topk_op
from .ops.moe.cute_dsl_moe_l2_prefetch import cute_dsl_moe_l2_prefetch
from .ops.moe.cute_dsl_moe_merge import cute_dsl_moe_merge
from .ops.moe.cute_dsl_moe_nvfp4_experts import cute_dsl_moe_nvfp4_experts
from .ops.moe.cute_dsl_moe_prefill_topk_routing_rep import cute_dsl_moe_prefill_topk_routing_rep
from .ops.moe.cute_dsl_moe_router import cute_dsl_moe_router
from .ops.moe.cute_dsl_moe_routing import cute_dsl_moe_routing
from .ops.moe.cute_dsl_moe_sigmoid_topk import cute_dsl_moe_sigmoid_topk
from .ops.moe.cute_dsl_moe_topk_routing import cute_dsl_moe_topk_routing
from .ops.moe.cute_dsl_moe_topk_routing_rep import cute_dsl_moe_topk_routing_rep
from .ops.moe.cute_dsl_moe_topk_routing_wide_rep import cute_dsl_moe_topk_routing_wide_rep
from .ops.moe.cute_dsl_moe_zero import cute_dsl_moe_zero
from .ops.moe.cute_dsl_mtp_moe_experts import cute_dsl_mtp_moe_experts
from .ops.moe.noaux_tc_op import noaux_tc_op
from .ops.norm import cute_dsl_hidden_add_rmsnorm as _hidden_add_norm_op
from .ops.norm import cute_dsl_hidden_rmsnorm as _hidden_norm_op
from .ops.norm import cute_dsl_mtp_eh_norm_cat_mxf8 as _eh8_op
from .ops.norm import cute_dsl_qkv_a_rmsnorm as _qkv_a_norm_op
from .ops.norm import cute_dsl_qkv_a_rmsnorm_q8 as _qkv_a_norm8_op
from .ops.norm.cute_dsl_hidden_add_rmsnorm import cute_dsl_hidden_add_rmsnorm
from .ops.norm.cute_dsl_hidden_rmsnorm import cute_dsl_hidden_rmsnorm
from .ops.norm.cute_dsl_mtp_eh_norm_cat_mxf8 import cute_dsl_mtp_eh_norm_cat_mxf8
from .ops.norm.cute_dsl_mtp_norm_shard import cute_dsl_mtp_norm_shard
from .ops.norm.cute_dsl_qkv_a_rmsnorm import cute_dsl_qkv_a_rmsnorm
from .ops.norm.cute_dsl_qkv_a_rmsnorm_q8 import cute_dsl_qkv_a_rmsnorm_q8
from .ops.quantization.cute_dsl_act_quant_mxf8_rows import cute_dsl_act_quant_mxf8_rows
from .ops.quantization.cute_dsl_moe_quantize_nvfp4 import cute_dsl_moe_quantize_nvfp4


def _build_step_args(md: TrtllmAttentionMetadata) -> dict:
    """Project the prepared metadata onto the batch state the context attention reads: the
    rope/KV append's per-sequence offsets and pool addressing, the prefill kernel's pool and
    per-row KV windows."""
    return dict(
        num_contexts=md.num_contexts,
        num_ctx_tokens=md.num_ctx_tokens,
        ctx_cached_token_indptr=md.ctx_cached_token_indptr,
        ctx_kv_indptr=md.ctx_kv_indptr,
        max_ctx_seq_len=md.max_ctx_seq_len,
        kv_cache_block_offsets=md.kv_cache_block_offsets,
        host_kv_cache_pool_pointers=md.host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping=md.host_kv_cache_pool_mapping,
        tokens_per_block=md.tokens_per_block,
        max_seq_len=md.max_seq_len,
        cu_seqlen_ks=md.cu_seqlen_ks,
        cu_seqlen_ke=md.cu_seqlen_ke,
        primary_pool=md.kv_cache_manager.get_unique_primary_pool(),
        # The prefill rope kernel's and the dense-causal FMHA's per-request lengths.
        ctx_uncached_token_indptr=md.ctx_uncached_token_indptr,
        # Their host twins, which the engine fills first and copies to the device: the
        # prefill rope kernel's int32 operands are staged from these.
        host_ctx_uncached_token_indptr=md.host_ctx_uncached_token_indptr,
        host_ctx_cached_token_indptr=md.host_ctx_cached_token_indptr,
        num_ctx_cached_tokens=md.num_ctx_cached_tokens,
        seq_lens=md.seq_lens,
        seq_lens_cuda=md.seq_lens_cuda,
    )


_KV_RESIDUAL_DIM = 0

_MOE_MAX_T = 8192

# The MTP width the specialised MLA rope kernel and the base-row index kind are built for at
# MTP1; MTP3 / MTP5 add max_draft_len + 1 (`mtp_widths`).
_MLA_ROPE_MTP_WIDTH = 2

# The decode kernels' widest compiled row count. Above it a projection with no compiled runner
# takes the prefill GEMM; at or below it, cuBLAS.
_DECODE_MAX_ROWS = 64

#: `_DecodeSlots.layer`'s generation index kind on a pure decode step whose attention takes the
#: union kernel: the slots are the step's request-local selection (`_UnionSlots`).
_UNION_POSITIONS = "union_positions"


class _UnionSlots(NamedTuple):
    """A pure decode step's generation selection as the union attention takes it: the
    request-local top-k positions `[rows, 2048]` and each row's block-table row."""

    positions: torch.Tensor
    req_idx: torch.Tensor


# The embedding kernel's widest row count. A step above it (a `max_num_tokens` past 8192)
# gathers in slices.
_EMBED_MAX_ROWS = 8192

# Projections an MTP3 / MTP5 captured decode step's rows without a cell keep on cuBLAS instead
# of the next cell up (`Gemm.cell_up`): there the cell is the slower one. Cold-weight CUDA-graph
# timing on B300 (cuBLAS vs next cell up, us per call): ix_qb 12 rows 4.95 vs 6.11, 18 rows
# 5.02 vs 5.42 (MTP 128 cell); dense_down 6 / 12 / 18 rows 8.15 / 8.25 / 8.29 vs 9.16 / 9.16
# / 9.15. An eager draft step (`DRAFT`) keeps its cells.
_CELL_UP_KEEPS_CUBLAS = frozenset({"ix_qb", "dense_down"})

# fp8_mqa_logits allocates a `[q_rows, kv_rows]` fp32 score matrix per call; prefill chunks are
# tiled to keep it under this many elements. Context steps are not captured.
_PREFILL_LOGITS_BUDGET = 1 << 24


def _tile_bounds(start: int, end: int, kv_rows: int) -> list[tuple[int, int]]:
    rows = max(1, _PREFILL_LOGITS_BUDGET // max(kv_rows, 1))
    return [(a, min(a + rows, end)) for a in range(start, end, rows)]


def _mtp_share_reuses(md) -> bool:
    """The engine's `reuse_topk` without the flag (the caller's): a draft step after the first,
    with a stash to read."""
    return bool(
        getattr(md, "in_mtp_draft_loop", False)
        and getattr(md, "indexer_skip_topk", False)
        and getattr(md, "shared_topk_indices", None) is not None
    )


def _mtp_share_load(md, dst: torch.Tensor, seqs: int) -> None:
    """A reusing draft step's selection: stash rows `[0, seqs)` into its `seqs` one-row
    requests' rows, as the engine copies `shared_topk_indices[:num_generations]`."""
    dst.copy_(md.shared_topk_indices[:seqs])


def _mtp_share_store(md, topk_buf: torch.Tensor, ctx: int, tc: int, seqs: int, gen_p: int):
    """Draft step 0's stash, as the engine writes it after the indexer: each context request's
    last row of `topk_buf[:tc]`, then each generation request's last accepted row of
    `topk_buf[tc:]` (row `gen_p - 1` without accepted counts), from stash row 0. A dense
    (skipped) phase's rows are stashed too, as the engine does."""
    rows = None
    if seqs:
        gen_topk = topk_buf[tc : tc + seqs * gen_p]
        num_accepted = getattr(md, "mtp_num_accepted", None)
        if num_accepted is None:
            rows = gen_topk[gen_p - 1 :: gen_p]
        else:
            base = torch.arange(seqs, device=gen_topk.device, dtype=torch.long) * gen_p
            offset = (num_accepted[ctx : ctx + seqs] - 1).clamp(0, gen_p - 1)
            rows = gen_topk[base + offset]
    if ctx:
        ctx_last = torch.cumsum(md.seq_lens_cuda[:ctx].to(torch.long), dim=0) - 1
        ctx_rows = topk_buf[ctx_last]
        rows = ctx_rows if rows is None else torch.cat([ctx_rows, rows])
    if rows is None:
        return
    shared = md.shared_topk_indices
    if shared is None:
        md.shared_topk_indices = rows.contiguous()
    else:
        shared[: rows.shape[0], : rows.shape[1]].copy_(rows)


def _moe_chunk_sizes(total: int) -> list[int]:
    full, rest = divmod(total, _MOE_MAX_T)
    return [_MOE_MAX_T] * full + ([rest] if rest else [])


# The routed experts' row domains: the decode kernels (router, fused top-k, quantize, clear,
# decode grouped GEMMs) serve 1..64 rows -- `max_batch_size` 32 x MTP width 2 -- and the
# prefill kernels 65..8192. A decode step presents a power of two up to 64 after padding.
_MOE_TOP_K = _moe_routing_op.TOP_K

_MOE_DECODE_WIDTHS = (1, 2, 4, 8, 16, 32, 64)

_MOE_PREFILL_MIN = _DECODE_MAX_ROWS + 1

# From here up the prefill pair runs the exact row count. The routing schedule is exact per
# `ceil(rows / 32)` up to 512 rows, so below this a power of two keeps the compiled
# schedules few; above it the four compiled buckets cover every count.
_MOE_PAIR_EXACT_FROM = 513

# NVFP4: one e4m3 scale per 16 elements; the shared expert's per-rank intermediate.
_MOE_SF_VEC = 16

# The widest exact decode step (`Moe.decode_exact`) whose pre-MoE collective also writes the
# routed experts' NVFP4 input (`Comm.prequant`): the quant entry's widest certified cell. MTP3
# / MTP5 at batch 32 step 128 / 192 rows.
_MOE_PREQUANT_WIDE_MAX = 256


def _pow2(rows: int) -> int:
    return 1 << max(0, (rows - 1).bit_length())


def _direct_ok(t: torch.Tensor, dtype: torch.dtype) -> bool:
    """Read in place rather than staged. A staging copy is a memcpy node, which a PDL edge
    cannot pass through, so it is skipped for a 128-byte aligned contiguous tensor -- the
    collective's output, whose graph-pool address is stable."""
    return t.dtype is dtype and t.is_contiguous() and t.data_ptr() % 128 == 0


@dataclass(frozen=True)
class Phase:
    """How a Target's rows map onto the compiled cells; a constant of the Target (and of its
    draft call). `wide`: rows above 64 are an MTP3 / MTP5 decode step's, on the MTP decode
    cells (`Gemm.wide_cell`) and the wide variants that keep the decode numerics. `up`: rows up
    to 64 without a cell of their own run on the next cell up at their exact rows -- on an
    MTP3 / MTP5 decode step (`"decode"`, `TRTLLM_GLM52_DEC_CELL_UP`) or an eager draft step
    (`"draft"`, which also lifts the calls that otherwise keep cuBLAS there); None keeps the
    stock op for such a count."""

    wide: bool = False
    up: str | None = None


PREFILL = Phase()
DRAFT = Phase(up="draft")


class Rows:
    def __init__(self, core) -> None:
        self.c = core

    def rmsnorm(self, x, weight):
        """RMSNorm of hidden-width rows: the CuTe DSL kernel, else FlashInfer's."""
        c = self.c
        rows = c._rows
        hit = rows.norm is not None and _hidden_norm_op.takes(x, weight)
        c._count("hidden_norm", hit)
        if hit:
            return cute_dsl_hidden_rmsnorm(x, weight, c.eps, rows.norm)
        return flashinfer_rmsnorm(x, weight, c.eps)

    def add(self, a, b):
        """A residual add of hidden-width rows: the CuTe DSL kernel, else `torch.add`."""
        c = self.c
        rows = c._rows
        hit = rows.add is not None and _hidden_add_op.takes(a, b)
        c._count("hidden_add", hit)
        if hit:
            return cute_dsl_hidden_add(a, b, rows.add)
        return torch.add(a, b)

    def add_rmsnorm(self, a, b, weight):
        """`rmsnorm(add(a, b), weight)`; the sum is not kept. One CuTe DSL launch where its
        kernel built and the two calls would both take their CuTe DSL kernels (the pair it
        is bit for bit), else the two calls."""
        c = self.c
        rows = c._rows
        hit = (
            rows.add_norm is not None
            and rows.add is not None
            and rows.norm is not None
            and _hidden_add_norm_op.takes(a, b, weight)
        )
        c._count("hidden_add_norm", hit)
        if hit:
            return cute_dsl_hidden_add_rmsnorm(a, b, weight, c.eps, rows.add_norm)
        return self.rmsnorm(self.add(a, b), weight)

    def copy(self, dst, src):
        """`dst.copy_(src)`: the CuTe DSL copy/convert kernel for the pairs it takes."""
        c = self.c
        rows = c._rows
        hit = rows.copy is not None and _copy_op.takes(dst, src)
        c._count("copy", hit)
        if hit:
            return cute_dsl_copy(dst, src, rows.copy)
        return dst.copy_(src)

    def route(self, logits, bias):
        """`(topk_w, topk_ids)` for a chunk the fused front end does not take: the CuTe DSL
        top-k when it takes the routing, else `noaux_tc_op`."""
        c = self.c
        rows = c._rows
        args = (logits, bias, c.n_group, c.topk_group, c.topk)
        hit = rows.topk is not None and _topk_rows_op.takes(*args)
        c._count("topk_rows", hit)
        if hit:
            return cute_dsl_moe_sigmoid_topk(*args, c.routed_scale, rows.topk)
        return noaux_tc_op(*args, c.routed_scale)

    def embed(self, input_ids):
        """The embedding rows for `input_ids`, in slices the kernel takes."""
        c = self.c
        if input_ids.shape[0] <= _EMBED_MAX_ROWS:
            return cute_dsl_embedding(input_ids, c.w["embed"], c._embedding)
        return torch.cat(
            [
                cute_dsl_embedding(ids, c.w["embed"], c._embedding)
                for ids in input_ids.split(_EMBED_MAX_ROWS)
            ]
        )


class Comm:
    def __init__(self, core) -> None:
        self.c = core

    def check_rows(self, partial) -> None:
        # The all-reduce kernels take the row count from their input, so a call above the
        # token budget the peer buffers were sized for would write past every peer's buffer.
        c = self.c
        assert partial.shape[0] <= c._lamport.max_rows, (
            f"{partial.shape[0]} rows exceed the collective's budget of {c._lamport.max_rows}"
        )

    def reduce(self, partial):
        c = self.c
        n = c._lamport.max_rows
        if partial.shape[0] > n:
            # A step above the token budget the peer buffers hold reduces in slices.
            return torch.cat([self.reduce(p) for p in partial.split(n)])
        self.check_rows(partial)
        return cute_dsl_allreduce(partial, torch.empty_like(partial), c._lamport)

    def reduce_norm(self, partial, residual, weight):
        """All-reduce, add into `residual` in place, RMS norm: every trunk collective."""
        c = self.c
        n = c._lamport.max_rows
        if partial.shape[0] > n:
            # Row slices of a contiguous `residual` are views, so the in-place add still lands.
            return torch.cat(
                [
                    self.reduce_norm(p, r, weight)
                    for p, r in zip(partial.split(n), residual.split(n))
                ]
            )
        self.check_rows(partial)
        out = torch.empty_like(partial)
        return cute_dsl_allreduce_add_rmsnorm(partial, residual, weight, out, c._lamport)

    def reduce_norm_x8(self, partial, residual, weight, want: bool):
        """`Comm.reduce_norm`, plus the normalized rows' MXF8 operand where the caller's consumer
        reads one (`want`) and the collective runs two-shot (its epilogue writes the operand;
        its bf16 rows are the plain two-shot's bytes): `(out, x8)`, `x8` None otherwise."""
        c = self.c
        rows = int(partial.shape[0])
        if (
            want
            and c._lamport.fused_twoshot_x8 is not None
            and _AR_TWOSHOT_ABOVE < rows <= c._lamport.max_rows
            and not torch.cuda.is_current_stream_capturing()
        ):
            self.check_rows(partial)
            out = torch.empty_like(partial)
            n = int(partial.shape[1])
            x8 = (
                torch.empty([rows, n], dtype=torch.uint8, device=partial.device),
                torch.empty(
                    (rows + 127) // 128 * 128 * (n // 32), dtype=torch.uint8, device=partial.device
                ),
            )
            cute_dsl_allreduce_add_rmsnorm(partial, residual, weight, out, c._lamport, x8=x8)
            c._count("ar_x8", True)
            return out, x8
        return self.reduce_norm(partial, residual, weight), None

    def prequant(self, rows: int, exact: bool) -> bool:
        """Whether a trunk MoE layer's pre-MoE collective also writes the routed experts'
        NVFP4 input (`Comm.reduce_norm_quant`): at a decode width (1..64 rows), and on a wide
        decode step that runs the routed experts on its exact rows (`exact`, see
        `Moe.decode_exact`: MTP3 / MTP5 only) up to `_MOE_PREQUANT_WIDE_MAX`. There the
        routed experts then skip their stage copy and quantize: the collective's bytes are
        their quantize's bit for bit. Every other row count -- context steps, and MTP1 and
        no speculation, whose decode steps stay within 64 rows -- keeps the plain collective
        and the routed experts' own quantize."""
        return rows <= _DECODE_MAX_ROWS or (exact and rows <= _MOE_PREQUANT_WIDE_MAX)

    def reduce_norm_quant(self, partial, residual, weight, g1, rows, exact=False):
        """`Comm.reduce_norm`, plus the routed experts' NVFP4 input where `Comm.prequant` says
        the collective writes it: into the layer's `xq` / `xsf`, exactly the rows the routed
        experts would quantize. Returns `(out, prequantized)`."""
        c = self.c
        if not self.prequant(rows, exact):
            return self.reduce_norm(partial, residual, weight), False
        self.check_rows(partial)
        xq, xsf = c._moe_xq[:rows], c._moe_xsf[:rows]
        out = torch.empty_like(partial)
        out = cute_dsl_allreduce_add_rmsnorm_quant_nvfp4(
            partial, residual, weight, g1, out, xq, xsf, c._lamport
        )
        return out, True

    def reduce_norm_moe(self, partial, residual, weight, g1, rows: int, want_x8: bool, exact: bool):
        """The pre-MoE collective of a trunk MoE layer: `(o, x8, prequantized)`. At a decode
        width (and on a wide decode step's exact rows up to `_MOE_PREQUANT_WIDE_MAX`,
        `prequant`) it also writes the routed experts' NVFP4 input, and they skip their stage
        copy and quantize; otherwise it writes the shared expert's MXF8 operand where that
        reads one (`want_x8`, two-shot rows) and the routed experts quantize themselves."""
        if not self.prequant(rows, exact):
            o, x8 = self.reduce_norm_x8(partial, residual, weight, want_x8)
            return o, x8, False
        o, prequant = self.reduce_norm_quant(partial, residual, weight, g1, rows, exact)
        if rows > _DECODE_MAX_ROWS:
            self.c._count("moe_dec_prequant", prequant)
        return o, None, prequant

    def close(self, mlp_out, residual, i: int):
        """The layer's closing collective, normed for layer `i + 1`: `(x, x8)`, `x8` the next
        layer's qkva MXF8 operand where the next qkva takes the MXF8 GEMM and the collective
        runs two-shot (else None). The last trunk layer's output takes the final norm."""
        c = self.c
        if i + 1 < c.num_layers:
            a_next = c._attn[i + 1]
            want = "qkva" in c._proj_mxf8 and a_next.qkva8 is not None
            return self.reduce_norm_x8(mlp_out, residual, c._next_norm[i], want)
        return self.reduce_norm(mlp_out, residual, c._next_norm[i]), None


class Gemm:
    def __init__(self, core) -> None:
        self.c = core
        # Host lookups kept per (call, rows): the cell a count without one of its own runs on
        # (`wide_cell`, `draft_cell`). Built in the forward, they compile nothing.
        self._wide_rag: dict = {}
        self._draft_rag: dict = {}

    def dec8_takes(self, name: str, x, wd8) -> bool:
        """Whether `Gemm.project` sends `x` to the decode MXF8 GEMM: its runner built, `wd8`
        given, a decode row count, and an activation its quant reads."""
        c = self.c
        return (
            wd8 is not None
            and name in c._dec8
            and 0 < int(x.shape[0]) <= _DECODE_MAX_ROWS
            and x.dtype == torch.bfloat16
            and x.stride(-1) == 1
            and x.stride(0) % 8 == 0
            and x.data_ptr() % 16 == 0
        )

    def dec8w_takes(self, ph: Phase, name: str, x, wd8) -> bool:
        """Whether `Gemm.project` sends `x` to the wide decode MXF8 GEMM: a `Phase.wide` step's
        rows above 64 (up to 192) of a call whose decode path is the decode MXF8 GEMM (`wd8`:
        the trunk o_proj), its wide runner built, and an activation its quant reads."""
        c = self.c
        return (
            ph.wide
            and wd8 is not None
            and name in c._dec8w
            and _DECODE_MAX_ROWS < int(x.shape[0]) <= _dec8_op.WIDE_ROWS[1]
            and x.dtype == torch.bfloat16
            and x.stride(-1) == 1
            and x.stride(0) % 8 == 0
            and x.data_ptr() % 16 == 0
        )

    def mxf8_takes(self, ph: Phase, name: str, x, w8, wd8=None) -> bool:
        """Whether `Gemm.project` sends `x` to the MXF8 GEMM: its runner built, `w8` given, a
        row count above the decode cells', and an activation the quant kernel reads. On a
        `Phase.wide` step only where the call's decode path is MXF8 as well (`wd8` and its
        decode runner: the trunk o_proj) and its wide decode GEMM (`Gemm.dec8w_takes`) did not
        build, so decode rows keep that path's numerics -- the same e4m3 operands, block
        scales and K order (B300: the two GEMMs' outputs are bitwise equal at 64 rows)."""
        c = self.c
        return (
            name in c._proj_mxf8
            and w8 is not None
            and int(x.shape[0]) > _DECODE_MAX_ROWS
            and (not ph.wide or (wd8 is not None and name in c._dec8))
            and x.dtype == torch.bfloat16
            and x.stride(-1) == 1
            and x.stride(0) % 8 == 0
            and x.data_ptr() % 16 == 0
        )

    def wide_cell(self, call: str, cells: dict, rows: int, at_rows=None):
        """The MTP decode cell for `rows` (above 64) of a `Phase.wide` step: the cell built
        at `rows` (128 / 192), else the smallest cell above `rows` launched on the exact rows
        -- a batch of fewer than 32 sequences (96 / 144 rows at 24 requests, any count in
        65..191 eagerly). A row's output is then bitwise that cell's for it: the cells are
        compiled on dynamic layouts, so the tiles past the last row load zeros and store
        nothing; no padding, no copy. `at_rows(cell, rows)` re-pins a runner pinned to its
        own count (None: the cell's runner takes any count up to its own). Returns None (the
        call's bf16 fallback runs) when no cell is above `rows`, or with `_wide_bucket` off.
        The re-pinned runner is kept per (`call`, rows): built in the forward, it compiles
        nothing, so a first one inside a CUDA-graph capture is safe."""
        c = self.c
        cell = cells.get(rows)
        if cell is not None or not c._wide_bucket:
            return cell
        key = (call, rows)
        if key not in self._wide_rag:
            bucket = min((m for m in cells if m > rows), default=None)
            cell = None
            if bucket is not None:
                cell = cells[bucket] if at_rows is None else at_rows(cells[bucket], rows)
            self._wide_rag[key] = cell
        cell = self._wide_rag[key]
        c._count("dec_wide_rows", cell is not None)
        return cell

    def cell_up(self, ph: Phase, rows: int) -> bool:
        """Whether a call's `rows` (at most 64) without a decode cell of their own run on the
        next cell up at their exact rows (`Gemm.draft_cell`, the runtime-M indexer split, the next
        eh_proj K-split instance, the fused shared-expert cells at MTP decode counts): an
        eager draft step's (`ph.up == "draft"`), and an MTP3 / MTP5 decode step's
        (`ph.up == "decode"`, `TRTLLM_GLM52_DEC_CELL_UP`). The latter is graph-captured:
        `rows` is fixed per capture shape and the cell is kept per (call, rows), so each
        graph freezes one choice with no host sync. Rows with a cell never ask."""
        return rows <= _DECODE_MAX_ROWS and ph.up is not None

    def draft_cell(self, ph: Phase, call: str, rows: int, cells: dict, mtp_cells: dict, at_rows):
        """The cell an eager draft step's or an MTP3 / MTP5 decode step's `rows` (at most 64,
        no cell of their own; see `Gemm.cell_up`) run on at their exact rows: the smallest decode
        cell above `rows` up to 64 that launches at a runtime row count, else the smallest
        such MTP cell (128 / 192). `at_rows(cell, rows)` re-pins as in `Gemm.wide_cell`; a None
        from it (a cell compiled for its own M only) moves to the next cell up. The
        context-sized cells (1099 rows and up) are never taken: their tiles past the last row
        would each read the weight again. Each row is bitwise the serving cell's for it. None
        (the call's fallback runs) when no cell qualifies. Kept per (`call`, rows); builds
        nothing, so a first call inside a CUDA-graph capture is safe."""
        c = self.c
        key = (call, rows)
        if key not in self._draft_rag:
            cell = None
            ups = [cells[m] for m in sorted(cells) if rows < m <= _DECODE_MAX_ROWS]
            ups += [mtp_cells[m] for m in sorted(mtp_cells) if m > rows]
            for up in ups:
                cell = at_rows(up, rows)
                if cell is not None:
                    break
            self._draft_rag[key] = cell
        cell = self._draft_rag[key]
        c._count("draft_cell" if ph.up == "draft" else "dec_cell_up", cell is not None)
        return cell

    def project(
        self,
        ph: Phase,
        name: str,
        x,
        w_colmajor,
        out_dtype=None,
        out=None,
        w8=None,
        pre=None,
        wd8=None,
        dec_pre=False,
    ):
        """`x @ w_colmajor` on the projection's CuTe DSL kernel at a compiled row count, on
        the prefill GEMM above the decode row counts, else cuBLAS. The kernels read
        `w_colmajor.t()`, the row-major weight, same memory. `out` is an optional caller-owned
        `[rows, N]` buffer; on the cuBLAS path the result is copied in. `w8` is the layer's
        MXF8 copy of the weight (qkva / o_proj): above the decode row counts the MXF8 GEMM
        takes the call instead of the prefill GEMM. `pre` is `x`'s MXF8 operand written by its
        producer: the GEMM then skips its activation quant. A caller passes it only where
        `Gemm.mxf8_takes` holds (o_proj's bf16 context rows are not written when it does). `wd8`
        is the layer's decode MXF8 weight (q_b / o_proj): decode rows take the decode MXF8
        GEMM; `dec_pre` says its producer wrote the operand (`Gemm.dec8_takes` or `Gemm.dec8w_takes`
        must hold). On a `Phase.wide` step, rows above 64 keep the numerics of the call's
        decode path: where that is the decode MXF8 GEMM, the wide decode MXF8 GEMM
        (`Gemm.dec8w_takes`) or the context MXF8 GEMM (`Gemm.mxf8_takes`), bitwise the same output;
        else the projection's MTP decode cell at its row count or the next one up
        (`Gemm.wide_cell`), else the bf16 prefill GEMM (or cuBLAS)."""
        c = self.c
        rows = int(x.shape[0])
        wide8 = self.dec8w_takes(ph, name, x, wd8)
        if (
            (wide8 or self.dec8_takes(name, x, wd8))
            and out_dtype in (None, torch.bfloat16)
            and (out is None or (out.dtype == torch.bfloat16 and out.is_contiguous()))
        ):
            runner = c._dec8w[name] if wide8 else c._dec8[name]
            if out is None:
                out = torch.empty([rows, runner.n], dtype=torch.bfloat16, device=x.device)
            tag = "dec_proj_mxf8w" if wide8 else "dec_proj_mxf8"
            c._count(f"{tag}_pre" if dec_pre else tag, True)
            return cute_dsl_projection_mxf8_decode(x, wd8, out, runner, pre=dec_pre)
        assert not dec_pre, f"{name}: a producer's decode operand reached another path"
        if self.mxf8_takes(ph, name, x, w8, wd8) and (
            out_dtype in (None, torch.bfloat16)
            and (out is None or (out.dtype == torch.bfloat16 and out.is_contiguous()))
        ):
            mx = c._proj_mxf8[name]
            if out is None:
                out = torch.empty([rows, mx.n], dtype=torch.bfloat16, device=x.device)
            c._count("ctx_proj_mxf8", True)
            if pre is not None:
                c._count("ctx_proj_mxf8_pre", True)
            return cute_dsl_projection_mxf8(x, w8, out, mx, pre=pre)
        assert pre is None, f"{name}: a producer's MXF8 operand reached a non-MXF8 path"
        runner = c._proj.get(name, {}).get(rows)
        if ph.wide and rows > _DECODE_MAX_ROWS:
            runner = self.wide_cell(
                f"proj:{name}", c._proj_mtp.get(name, {}), rows, _projection_op.at_rows
            )
            c._count("dec_wide", runner is not None)
        if (
            runner is None
            and self.cell_up(ph, rows)
            and (ph.up == "draft" or name not in _CELL_UP_KEEPS_CUBLAS)
        ):
            runner = self.draft_cell(
                ph,
                f"proj:{name}",
                rows,
                c._proj.get(name, {}),
                c._proj_mtp.get(name, {}),
                _projection_op.at_rows,
            )
        prefill = c._proj_pre.get(name)
        if (
            runner is None
            and prefill is not None
            and _DECODE_MAX_ROWS < rows <= prefill.max_rows
            and x.dtype == prefill.in_dtype
            and x.is_contiguous()
            and out_dtype in (None, prefill.out_dtype)
            and (out is None or out.dtype == prefill.out_dtype)
        ):
            runner = prefill
        if runner is None:
            res = cublas_mm(x, w_colmajor, None, out_dtype)
            if out is None:
                return res
            return c.op.rows.copy(out, res)
        if out is None:
            out = torch.empty([rows, runner.n], dtype=runner.out_dtype, device=x.device)
        return cute_dsl_projection(x, w_colmajor.t(), out, runner)

    def project_tail(self, ph: Phase, name: str, x, start: int, w_colmajor):
        """The projection of `x[start:]`, a mixed step's generation rows after its context
        rows. Their count is any 1..63, which the decode buckets rarely match, and a miss
        goes to cuBLAS; so the smallest compiled bucket `b` above the count runs on the
        trailing `b` rows of `x` (the last `b - rows` context rows ride along, their outputs
        unused) and the result is the view of its last rows: no padding copy."""
        c = self.c
        total = int(x.shape[0])
        rows = total - start
        runners = c._proj.get(name, {})
        if rows <= _DECODE_MAX_ROWS and rows not in runners:
            bucket = min((m for m in runners if m >= rows), default=None)
            if bucket is not None and bucket <= total:
                c._count(f"{name}_tail", True)
                return self.project(ph, name, x[total - bucket :], w_colmajor)[bucket - rows :]
        return self.project(ph, name, x[start:], w_colmajor)

    def absorb_cell(self, ph: Phase, cells: dict, rows: int):
        """`cells[rows]` (a decode K-B / V-B object by its cells' row counts), else, for a
        count in 9..64 without a cell where `Gemm.cell_up` holds (MTP5 at 2 / 3 requests: 12 / 18
        rows; MTP3 at 3: 12), the M >= 16 group's object, which launches at any of them
        (`_absorbed_bmm_op.DECODE_RUNTIME_ROWS`). None otherwise. A dict lookup on host ints:
        frozen per capture shape."""
        c = self.c
        cell = cells.get(rows)
        if cell is None and rows in _absorbed_bmm_op.DECODE_RUNTIME_ROWS and self.cell_up(ph, rows):
            cell = cells.get(_DECODE_MAX_ROWS)
            c._count("dec_cell_up", cell is not None)
        return cell

    def absorbed_bmm(self, ph: Phase, kind: str, x_t, w_kn, w_nk, out_t) -> None:
        """An absorbed-MLA BMM ("k_b" or "v_b") into `out_t`: the decode kernel at its row
        counts (2..64, and 65..192 on a `Phase.wide` step: the 128 / 192 cells' object,
        which launches at any of them; 9..64 where `Gemm.cell_up` holds: the M >= 16 group's
        object, `Gemm.absorb_cell`), the prefill kernel at any other up to 8192, else `bmm_out`,
        which takes the `[H, K, N]` operand instead of `[H, N, K]`."""
        c = self.c
        rows = int(x_t.shape[1])
        runner = self.absorb_cell(ph, c._absorb[kind], rows)
        if ph.wide and rows > _DECODE_MAX_ROWS:
            runner = self.wide_cell(kind, c._absorb_mtp[kind], rows)
            c._count("dec_wide", runner is not None)
        if runner is None and rows <= _absorbed_bmm_op.PREFILL_MAX_ROWS:
            runner = c._absorb_pre[kind]
        if runner is None:
            bmm_out(x_t, w_kn, out_t)
        else:
            cute_dsl_absorbed_bmm(x_t, w_nk, out_t, runner)

    def q_b_e4m3(self, q_a_ctx, a, pre=None):
        """The fp8 context attention's query: `q_a x (q_b / s_q)` in e4m3, `[T, heads *
        qk_dim]`, on the MXF8 GEMM or the prefill GEMM's e4m3 entry (`_build_context_mha_fp8`).
        `pre` is q_a's MXF8 operand written by the norm (`Attn.qkv`'s `q8`)."""
        c = self.c
        rows = int(q_a_ctx.shape[0])
        q8 = torch.empty(
            [rows, c.heads * c.qk_dim], dtype=torch.float8_e4m3fn, device=q_a_ctx.device
        )
        if isinstance(c._q8, _mxf8_op.Mxf8ProjectionRunner):
            return cute_dsl_projection_mxf8(q_a_ctx, a.qb8, q8, c._q8, pre=pre)
        assert pre is None, "the q_a norm's MXF8 operand needs the MXF8 q_b"
        return cute_dsl_projection(q_a_ctx, a.qb8, q8, c._q8)

    def qkva_ix_fold(self, ph: Phase, x, i: int, rows: int):
        """`(qkva, key, head_weights)` of indexer layer `i` from one GEMM (the qkva projection
        with the indexer's key / head-weight rows folded in, `cute_dsl_qkva_ix_fold`), or None
        where it does not serve and the two GEMMs run: off (`TRTLLM_GLM52_WIDE_IX_FOLD=0`), an
        MTP1 engine, a step that is not a wide decode step (rows above 64 on an MTP3 / MTP5
        engine), or rows without a qkva MTP cell. Where it serves, `qkva` is bitwise the qkva
        MTP cell's (`Gemm.project` on these rows) and the indexer columns accumulate as the
        split-output projection's (its chains and reduction order); `key` / `head_weights` are
        views of the cell's side buffers, consumed by this layer's indexer before the next
        layer's call."""
        c = self.c
        w = c._ix_fold_w.get(i)
        if (
            w is None
            or not c._qkva_ix_mtp
            or not ph.wide
            or rows <= _DECODE_MAX_ROWS
            or x.dtype != torch.bfloat16
            or not x.is_contiguous()
        ):
            return None
        runner = self.wide_cell("qkva_ix_fold", c._qkva_ix_mtp, rows, _ix_fold_op.at_rows)
        c._count("ix_fold", runner is not None)
        if runner is None:
            return None
        out = torch.empty([rows, _ix_fold_op.QKVA_N], dtype=x.dtype, device=x.device)
        return cute_dsl_qkva_ix_fold(x, w, out, runner)

    def qb_ixqb_runner(self, ph: Phase, i: int, rows: int):
        """The q_b | wq_b cell for indexer layer `i` at `rows` (`Gemm.qb_ixqb`), or None where it
        does not serve and the two GEMMs run: off (`TRTLLM_GLM52_WIDE_QB_IXQB=0`), an MTP1
        engine, a step that is not a wide decode step (rows above 64 on an MTP3 / MTP5
        engine), or rows without an MTP cell. Counted as `qb_ixqb`."""
        c = self.c
        if i not in c._qbix_w or not c._qbix_mtp:
            return None
        if not ph.wide or rows <= _DECODE_MAX_ROWS:
            return None
        runner = self.wide_cell("qb_ixqb", c._qbix_mtp, rows, _qbix_op.at_rows)
        c._count("qb_ixqb", runner is not None)
        return runner

    def qb_ixqb(self, q_a, i: int, runner):
        """`(q, ix_q)` of layer `i` from one GEMM: bitwise the q_b and wq_b MTP cells' outputs
        (`cute_dsl_q_b_ix_wq_b`); `q` `[rows, 4096]` and `ix_q` contiguous halves of one
        buffer."""
        c = self.c
        rows = int(q_a.shape[0])
        out2 = torch.empty([2, rows, _qbix_op.N], dtype=q_a.dtype, device=q_a.device)
        return cute_dsl_q_b_ix_wq_b(q_a, c._qbix_w[i], out2, runner)


class _DecodeSlots:
    """One pure decode step's `Attn.decode_slots`: `layer(buf, fresh, layer_id)` maps the
    request-local selection `buf` to what the sparse MLA decode of pool layer `layer_id`
    reads, `(selection, index kind, slot layer)`. Base rows are layer-independent, so they are
    recomputed only when a layer selected afresh (`fresh`)."""

    def __init__(self, core, md, req_idx, union: bool, base_rows: bool) -> None:
        self.c, self.md, self.req_idx = core, md, req_idx
        self.union, self.base_rows = union, base_rows
        self.base = None

    def layer(self, buf, fresh: bool, layer_id: int):
        c, md = self.c, self.md
        if self.union:
            return _UnionSlots(buf, self.req_idx), _UNION_POSITIONS, layer_id
        if not self.base_rows:
            rows = cute_dsl_req_index_to_global(
                self.req_idx, md.block_table, buf, md.tokens_per_block, c._stride_factor,
                layer_id, compiled=c._req_global_compiled,
            )  # fmt: skip
            return rows, INDEX_KIND_GLOBAL, None
        if fresh or self.base is None:
            self.base = cute_dsl_req_index_to_base(
                self.req_idx, md.block_table, buf, md.tokens_per_block, c._stride_factor,
                compiled=c._req_base_compiled,
            )  # fmt: skip
        return self.base, INDEX_KIND_BASE_WITH_LAYER_OFFSET, layer_id


class Attn:
    def __init__(self, core) -> None:
        self.c = core

    def qkv(
        self,
        ph: Phase,
        x,
        a,
        q_b: bool = True,
        x8=None,
        q8_rows: int = 0,
        q_store_from: int = 0,
        qkva=None,
    ):
        """`(q, q_a, latent, q8)`. The kv norm writes in place, so the latent `[ckv | k_pe]` is
        the fused projection's tail, a strided view. `q_b=False` leaves q to the caller (None).
        `x8` is `x`'s MXF8 operand from the closing all-reduce of the layer before. With
        `q8_rows` the norm also writes the first `q8_rows` q_a rows as the MXF8 operand `q8` of
        the fp8 context attention's q_b, where its q8 kernel takes the call (else `q8` is
        None); `q_store_from` then skips the bf16 q rows below it (no reader). `qkva`: the
        projection's output the caller already wrote (`Gemm.qkva_ix_fold`), else it runs here."""
        c = self.c
        if qkva is None:
            qkva = c.op.gemm.project(ph, "qkva", x, a.qkva, w8=a.qkva8, pre=x8)
        q_a_raw, ckv_raw, _ = torch.split(qkva, [c.q_lora, c.kv_lora, c.rope_dim], -1)
        if (
            q8_rows
            and c._qkv_a_norm8 is not None
            and int(q_a_raw.shape[0]) >= _qkv_a_norm_op.WARP_ROW_MIN_ROWS
            and not torch.cuda.is_current_stream_capturing()
            and _qkv_a_norm8_op.takes(q_a_raw, a.q_norm, ckv_raw, a.kv_norm)
        ):
            q8 = _qkv_a_norm8_op.operand(q8_rows, x.device)
            q_a = cute_dsl_qkv_a_rmsnorm_q8(
                q_a_raw, a.q_norm, ckv_raw, a.kv_norm, c.eps, *q8, c._qkv_a_norm8,
                q_store_from=q_store_from,
            )  # fmt: skip
            c._count("qkv_a_q8", True)
            if q_store_from:
                c._count("qkv_a_q8_skip", True)
                if c._qkv_a_norm8.ilp:
                    c._count("norm_ilp", True)
            latent = torch.split(qkva, [c.q_lora, c.lat_dim], -1)[1]
            return None, q_a, latent, q8
        # Decode rows: the norm writes q_a's MXF8 bytes into the decode q_b GEMM's own operand
        # and releases that GEMM at entry, so q_b runs without its activation quant.
        rows = int(q_a_raw.shape[0])
        if (
            q_b
            and rows <= _DECODE_MAX_ROWS
            and c._qkv_a_norm8 is not None
            and a.qb_dec8 is not None
            and _qkv_a_norm8_op.takes(q_a_raw, a.q_norm, ckv_raw, a.kv_norm)
        ):
            xq, sf = c._dec8["q_b"].operand
            q_a = cute_dsl_qkv_a_rmsnorm_q8(
                q_a_raw, a.q_norm, ckv_raw, a.kv_norm, c.eps, xq[:rows], sf, c._qkv_a_norm8
            )
            if c.op.gemm.dec8_takes("q_b", q_a, a.qb_dec8):
                latent = torch.split(qkva, [c.q_lora, c.lat_dim], -1)[1]
                q = c.op.gemm.project(ph, "q_b", q_a, a.qb, wd8=a.qb_dec8, dec_pre=True)
                return q, q_a, latent, None
            latent = torch.split(qkva, [c.q_lora, c.lat_dim], -1)[1]
            return c.op.gemm.project(ph, "q_b", q_a, a.qb, wd8=a.qb_dec8), q_a, latent, None
        # Both kernels are this op's; the counter shows which one a step's row count took.
        wr = _qkv_a_norm_op.takes_warp_row(q_a_raw, a.q_norm, ckv_raw, a.kv_norm)
        c._count("qkv_a_warp_row" if wr else "qkv_a_cta", True)
        # Eager calls on the in-place layout skip the kernel module's per-call checks.
        if _qkv_a_norm_op.takes_direct(q_a_raw, a.q_norm, ckv_raw, a.kv_norm):
            c._count("qkv_a_direct", True)
        q_a, _ = cute_dsl_qkv_a_rmsnorm(q_a_raw, a.q_norm, ckv_raw, a.kv_norm, c.eps, c._qkv_a_norm)
        latent = torch.split(qkva, [c.q_lora, c.lat_dim], -1)[1]
        q = c.op.gemm.project(ph, "q_b", q_a, a.qb, wd8=a.qb_dec8) if q_b else None
        return q, q_a, latent, None

    def mtp_rope_specialised(self, gen_p: int) -> bool:
        """Whether the MTP layer's decode rows of `gen_p` per sequence take the specialised
        MLA rope kernel (`Attn.gen_query`): at `mtp_widths`, as the trunk; and on an engine
        drafting more than one token also on its 1-token draft steps (draft steps
        1..max_draft_len-1, `gen_p` 1), where the stock op needed the strided latent packed
        first (a `[rows, 576]` copy per draft step). The kernel writes the same bytes as the
        stock op there (query, latent row, prefix arrays, scheduler counter, FMHA scales);
        the K-B then takes its decode fp8 q-base cell, as at MTP1's width 2."""
        c = self.c
        return gen_p in c.mtp_widths or (gen_p == 1 and c.max_draft_len > 1)

    def fmha_dec_handle(self, gen_p: int) -> "_fmha_dec_op.CompiledSparseMlaDecode":
        """The decode attention's handle for `gen_p` rows per sequence: the width-2 grid up to
        2 (MTP1, no speculation, MTP3 / MTP5's 1-row draft steps), the MTP3 / MTP5 grid above."""
        c = self.c
        if gen_p <= c._fmha_dec_compiled.max_sq:
            return c._fmha_dec_compiled
        return c._fmha_dec_wide

    def qkv_gen(self, ph: Phase, x, a, i: int, select: bool):
        """A decode step's `(q, q_a, latent, kw, ixq)` for layer `i`. On an MTP3 / MTP5 wide
        decode step an indexer layer's qkva projection also writes the indexer's key / head
        weights (`kw`, `Gemm.qkva_ix_fold`), and where the step selects, q_b also writes the
        indexer's q projection (`ixq`, `Gemm.qb_ixqb`), each where its cell serves the rows;
        elsewhere `kw` / `ixq` are None and the indexer projects them itself. Up to 64 rows
        the q_a norm writes the decode MXF8 q_b's operand (`qkv`)."""
        c = self.c
        g = c.op.gemm
        ix = c._idx[i] is not None
        rows = int(x.shape[0])
        fold = g.qkva_ix_fold(ph, x, i, rows) if ix else None
        qbix = g.qb_ixqb_runner(ph, i, rows) if ix and select else None
        q, q_a, latent, _ = self.qkv(
            ph, x, a, q_b=qbix is None, qkva=None if fold is None else fold[0]
        )
        ixq = None
        if qbix is not None:
            q, ixq = g.qb_ixqb(q_a, i, qbix)
        return q, q_a, latent, None if fold is None else fold[1:], ixq

    def gen_query(
        self, ph: Phase, a, layer: int, q, latent, gen: int, gen_p: int, ctx: int, md, rope: str
    ):
        """The generation rows' absorbed query for the sparse MLA decode: K-B, then RoPE and
        the latent row's KV append; returns `(quant_q, cu_q, bmm1, bmm2)`.

        `rope` names the caller: `"trunk"` (a trunk layer of a pure decode step) and `"mtp"`
        (the MTP layer's decode draft step) take the specialised MLA rope kernel at the widths
        it is built for (`mtp_widths`; the MTP layer also at MTP3 / MTP5's 1-token draft
        steps, `mtp_rope_specialised`), `"stock"` (a mixed step's generation rows) the stock
        op. On the specialised rope a decode cell's K-B writes the e4m3 q-base into the
        query itself (the bytes the rope kernel would convert) and releases the rope kernel
        at once: the rope skips the q-base and waits for the K-B only at its end, so the two
        run concurrently and the decode MLA's wait on the rope still covers the K-B."""
        c = self.c
        if rope == "trunk":
            specialised = gen_p in c.mtp_widths
        elif rope == "mtp":
            specialised = self.mtp_rope_specialised(gen_p)
        else:
            specialised = False
        heads, dev, dt = c.heads, q.device, q.dtype
        q_nope, q_pe = torch.split(
            torch.reshape(q, [gen, heads, c.qk_dim]), [c.nope, c.rope_dim], -1
        )
        fused_q = torch.empty([gen, heads, c.lat_dim], dtype=dt, device=dev)
        fq_nope = torch.split(fused_q, [c.kv_lora, c.rope_dim], -1)[0]
        quant_q = torch.empty([gen, heads, c.lat_dim], dtype=torch.float8_e4m3fn, device=dev)
        kb8 = c.op.gemm.absorb_cell(ph, c._kb8, gen) if specialised else None
        if specialised and ph.wide and gen > _DECODE_MAX_ROWS and c._kb8_mtp:
            # MTP3 / MTP5 decode rows: the MTP cells' fp8 q-base K-B, at its cell or on the
            # next one up at the exact rows (the rope handles of widths 4 / 6 carry the
            # `q_base_from_kb` variants whenever the decode K-B built).
            kb8 = c.op.gemm.wide_cell("k_b_fp8", c._kb8_mtp, gen)
        if kb8 is not None:
            q_base = torch.split(quant_q.view(torch.uint8), [c.kv_lora, c.rope_dim], -1)[0]
            cute_dsl_absorbed_bmm(
                torch.transpose(q_nope, 0, 1), a.k_b_nk, torch.transpose(q_base, 0, 1), kb8
            )
            c._count("kb_fp8_qbase", True)
        else:
            # Over an fp8 pool the rope kernel reads this half to build the quantized query,
            # so the BMM is issued first on the same stream.
            c.op.gemm.absorbed_bmm(
                ph,
                "k_b",
                torch.transpose(q_nope, 0, 1),
                a.k_b,
                a.k_b_nk,
                torch.transpose(fq_nope, 0, 1),
            )
        cu_q = torch.empty([gen + 1], dtype=torch.int32, device=dev)
        cu_kv = torch.empty([gen + 1], dtype=torch.int32, device=dev)
        counter = torch.empty([1], dtype=torch.uint32, device=dev)
        bmm1 = torch.empty([2], dtype=torch.float32, device=dev)
        bmm2 = torch.empty([1], dtype=torch.float32, device=dev)
        rope = c._rope
        if specialised:
            if gen_p == 1:
                c._count("mla_rope_draft1", True)
            elif gen_p != _MLA_ROPE_MTP_WIDTH:
                c._count("mla_rope_wide", True)
            # The specialised kernel takes the stock op's earlier positional ABI.
            cute_dsl_mla_rope_generation(
                fused_q, q_pe, latent, rope["rotary_cos_sin"], cu_q, cu_kv, counter, bmm1, bmm2,
                quant_q, md.kv_lens_cuda_runtime, md.kv_lens_runtime, md.prompt_lens_cpu_runtime,
                ctx, md.kv_cache_block_offsets, md.host_kv_cache_pool_pointers,
                md.host_kv_cache_pool_mapping, None, None, None, None, [None, None], gen_p, layer,
                heads, 1, c.lat_dim, md.tokens_per_block, md.max_seq_len, 1, c.quant_mode,
                1.0, c.q_lora, c.kv_lora, c.nope, c.rope_dim, c.kv_lora, True,
                compiled=c._mla_rope_handle(
                    int(md.kv_cache_block_offsets.shape[-1]), dev.index, gen_p
                ),
                q_base_from_kb=kb8 is not None,
            )  # fmt: skip
        else:
            mla_rope_generation(
                fused_q=fused_q,
                q_pe=q_pe,
                # The stock op takes packed rows only.
                latent_cache=latent.contiguous(),
                rotary_cos_sin=rope["rotary_cos_sin"],
                cu_q_seqlens=cu_q,
                cu_kv_seqlens=cu_kv,
                fmha_scheduler_counter=counter,
                mla_bmm1_scale=bmm1,
                mla_bmm2_scale=bmm2,
                quant_q_buffer=quant_q,
                sequence_length=md.kv_lens_cuda_runtime,
                host_past_key_value_lengths=md.kv_lens_runtime,
                host_context_lengths=md.prompt_lens_cpu_runtime,
                num_contexts=ctx,
                kv_cache_block_offsets=md.kv_cache_block_offsets,
                host_kv_cache_pool_pointers=md.host_kv_cache_pool_pointers,
                host_kv_cache_pool_mapping=md.host_kv_cache_pool_mapping,
                kv_scale_orig_quant=None,
                kv_scale_quant_orig=None,
                kv_cache_scale_orig_quant=None,
                out_scale=None,
                block_ids_per_seq=None,
                helix_tensor_params=[None, None],
                predicted_tokens_per_seq=gen_p,
                layer_idx=layer,
                num_heads=heads,
                num_kv_heads=1,
                head_size=c.lat_dim,
                residual_dim=_KV_RESIDUAL_DIM,
                tokens_per_block=md.tokens_per_block,
                attention_window_size=md.max_seq_len,
                beam_width=1,
                quant_mode=c.quant_mode,
                q_scaling=1.0,
                q_lora_rank=c.q_lora,
                kv_lora_rank=c.kv_lora,
                qk_nope_head_dim=c.nope,
                qk_rope_head_dim=c.rope_dim,
                v_head_dim=c.kv_lora,
                rope_append=True,
            )
        return quant_q, cu_q, bmm1, bmm2

    def sparse_decode(self, quant_q, md, slots, lat_out, cu_q, bmm1, bmm2, ctx: int, gen_p: int):
        """The sparse MLA decode of the generation rows into `lat_out`: on the union kernel
        where the step's selection is the union attention's (`DecodeSlots`, MTP3 / MTP5), else
        on the decode kernel's grid for `gen_p` rows per sequence. `slots` is
        `(selection, index kind, slot layer)` as `_DecodeSlots.layer` returns it."""
        c = self.c
        slots, slot_kind, slot_layer = slots
        gen = int(quant_q.shape[0])
        seqs = gen // gen_p
        if slot_kind == _UNION_POSITIONS:
            c._count("fmha_dec_union", True)
            cute_dsl_sparse_mla_decode_union(
                quant_q,
                md.kv_cache_manager.get_unique_primary_pool(),
                slots.positions,
                md.block_table,
                slots.req_idx,
                lat_out,
                cu_q[: seqs + 1],
                md.kv_lens_cuda_runtime[ctx : ctx + seqs],
                slot_layer,
                c._stride_factor,
                compiled=c._fmha_union,
            )
        else:
            fmha = self.fmha_dec_handle(gen_p)
            if fmha is c._fmha_dec_wide:
                c._count("fmha_dec_wide", True)
            cute_dsl_sparse_mla_decode(
                quant_q,
                md.kv_cache_manager.get_unique_primary_pool(),
                slots,
                lat_out,
                cu_q[: seqs + 1],
                md.kv_lens_cuda_runtime[ctx : ctx + seqs],
                bmm1,
                bmm2,
                slot_kind,
                slot_layer,
                compiled=fmha,
            )

    def v_b(self, ph: Phase, a, lat_out, out, trunk: bool) -> bool:
        """V-B of the generation rows' attention into `out`. On a trunk layer of a pure decode
        step whose o_proj takes the decode MXF8 GEMM (or above 64 rows the wide one), the
        V-B also writes `out` as that GEMM's operand where a side-output cell serves the row
        count (above 64: where `wide_cell` gives the plain MTP cell, so the bf16 `out` is that
        cell's); returns whether it did, which o_proj then takes as `dec_pre`."""
        c = self.c
        g = c.op.gemm
        gen = int(out.shape[0])
        heads = c.heads
        o8_side = trunk and (
            g.dec8_takes("o_proj", out, a.o_dec8) or g.dec8w_takes(ph, "o_proj", out, a.o_dec8)
        )
        lat_t = torch.transpose(torch.reshape(lat_out, [gen, heads, c.kv_lora]), 0, 1)
        out_t = torch.transpose(torch.reshape(out, [gen, heads, c.v_dim]), 0, 1)
        vb8 = None
        if o8_side and gen <= _DECODE_MAX_ROWS:
            # 9..64 rows without a cell on an MTP3 / MTP5 decode step too (`Gemm.absorb_cell`):
            # o_proj then skips its own act-quant there as well.
            vb8 = c.op.gemm.absorb_cell(ph, c._vb8, gen)
        elif o8_side and c._vb8w:
            vb8 = c.op.gemm.wide_cell("v_b_x8", c._vb8w, gen)
        if vb8 is not None:
            cute_dsl_absorbed_bmm(lat_t, a.v_b_nk, out_t, vb8)
            return True
        c.op.gemm.absorbed_bmm(ph, "v_b", lat_t, a.v_b_t, a.v_b_nk, out_t)
        return False

    def decode_slots(self, md, req_idx, gen_p: int) -> "_DecodeSlots":
        """A pure decode step's trunk selection as the sparse MLA decode reads it, per layer
        (`_DecodeSlots.layer`): the union attention's request-local positions where it serves
        the step (MTP3 / MTP5 widths, a batch size it planned), the layer-independent base rows
        at a width whose decode handle takes them, else each layer's global pool rows."""
        c = self.c
        base_rows = (
            gen_p in c.mtp_widths
            and INDEX_KIND_BASE_WITH_LAYER_OFFSET in self.fmha_dec_handle(gen_p).index_kinds
        )
        union = (
            c._fmha_union is not None
            and _MLA_ROPE_MTP_WIDTH < gen_p <= c._fmha_union.max_sq
            and md.num_seqs in c._fmha_union.plans
        )
        return _DecodeSlots(c, md, req_idx, union, base_rows)

    def draft_slots(self, md, req_idx, buf, stride: int, pool_layer: int):
        """The MTP layer's selection as global rows of its own pool (the separate draft pool's
        live mapping), on the stock conversion."""
        c = self.c
        rows = convert_req_index_to_global(
            req_idx, md.block_table, buf, md.tokens_per_block, c.idx_topk, stride, pool_layer
        )
        return rows, INDEX_KIND_GLOBAL, None


class Indexer:
    def __init__(self, core) -> None:
        self.c = core
        # The K-cache update validates the slot mappings on its first (eager) call only.
        self._k_checked = False

    def topk_buf(self, md, rows: int, tc: int, skip_ctx: bool, skip_gen: bool):
        """Where this step's selection lands. A phase the engine marked skippable (every
        sequence shorter than index_topk) takes the dense causal index the engine wrote; a
        branch frozen into a graph, safe because short and long steps capture separately."""
        c = self.c
        if skip_ctx and skip_gen:
            return md.topk_indices_buffer[:rows]
        # `topk_indices_buffer` exists only while the engine's indexer skip is on.
        buf = torch.empty([rows, c.idx_topk], dtype=torch.int32, device=torch.cuda.current_device())
        if skip_ctx and tc:
            c.op.rows.copy(buf[:tc], md.topk_indices_buffer[:tc])
        if skip_gen and rows > tc:
            c.op.rows.copy(buf[tc:], md.topk_indices_buffer[tc:rows])
        return buf

    def project(
        self, ph: Phase, x, q_a, pos, ix, rows: int, need_q: bool, kw=None, ixq=None, kc=None
    ):
        """The indexer's front: `(q_fp8, weights, k_pre)`; `q_fp8` and `weights` are None on a
        step where every phase takes the engine's index. The k side always runs: the keys this
        step appends are read by later steps. `kw`: the pre-norm keys and head weights the
        caller already projected (`Gemm.qkva_ix_fold`), else the split-output projection runs;
        `ixq`: the indexer's q projection likewise (`Gemm.qb_ixqb`). `kc`: `(md, k_pool)`, the
        layer's K cache: where `Indexer.qk_handle` has a launcher the q side and the K-cache
        update run as one launch here, and `k_pre` is None (the keys are written)."""
        c = self.c
        hi, di = c.idx_heads, c.idx_dim
        dev = x.device
        qside = c._ix_qside_compiled
        if kw is not None:
            kb, wq = kw
        else:
            split = c._ix_pre_split.get(rows)
            if ph.wide and rows > _DECODE_MAX_ROWS:
                # Rows without a handle of their own take the runtime-M handle (the same kernel).
                rag = c._ix_pre_split_rag
                if rag is not None and rows > rag.max_m:
                    rag = None
                split = c.op.gemm.wide_cell("ix_split", c._ix_pre_split_mtp, rows, lambda h, r: rag)
                c._count("dec_wide", split is not None)
            elif split is None and c.op.gemm.cell_up(ph, rows):
                # An eager draft step's or an MTP3 / MTP5 decode step's count without a
                # handle: the runtime-M handle (the same kernel) instead of `x.float()` and
                # the fp32 cuBLAS projection, its split-K reduce and two copies.
                rag = c._ix_pre_split_rag
                split = rag if rag is not None and rows <= rag.max_m else None
                c._count("draft_cell" if ph.up == "draft" else "dec_cell_up", split is not None)
            pre = c._ix_pre_split_pre
            if (
                split is None
                and pre is not None
                and _DECODE_MAX_ROWS < rows <= pre.max_rows
                and x.is_contiguous()
            ):
                split = pre
            if split is not None and split is pre and c._ix_nosplit:
                # fp32 column views of one [rows, 160] output (`TRTLLM_GLM52_CTX_IX_NOSPLIT`).
                kb, wq = _ix_pre_split_op.project_unsplit(x, ix.fused_bf16, pre)
                qside = c._ix_qside_compiled_p160
                c._count("ctx_ix_nosplit", True)
            elif split is not None:
                kb = torch.empty([rows, di], dtype=x.dtype, device=dev)
                wq = torch.empty([rows, hi], dtype=torch.float32, device=dev)
                cute_dsl_indexer_pre_proj_split(x, ix.fused_bf16, kb, wq, split)
            else:
                kb = torch.empty([rows, di], dtype=x.dtype, device=dev)
                # fp32: the only `weights` dtype fp8_mqa_logits takes.
                fused = c.op.gemm.project(ph, "ix_fused", x.float(), ix.fused, torch.float32)
                k_raw, wq_raw = torch.split(fused, [di, hi], -1)
                c.op.rows.copy(kb, k_raw)
                # The scorers check `weights`' stride(1) but not stride(0), then read it
                # densely: a strided slice scores each row with another row's head weights.
                wq = wq_raw.contiguous()
        k_pre = (kb, ix.kn_w, ix.kn_b, pos)
        if not need_q:
            return None, None, k_pre
        if ixq is None:
            ixq = c.op.gemm.project(ph, "ix_qb", q_a, ix.qb)
        q = torch.reshape(ixq, [rows, hi, di])
        q_fp8 = torch.empty([rows, hi, di], dtype=torch.float8_e4m3fn, device=dev)
        weights = torch.empty([rows, hi], dtype=torch.float32, device=dev)
        qk = self.qk_handle(ph, kc, kb, rows)
        if qk is not None:
            md, k_pool = kc
            c._count("ix_qk", True)
            cute_dsl_indexer_q_side_k_cache(
                q, pos, c._idx_cos_sin, wq, c.idx_weight_scale, q_fp8, weights, kb,
                ix.kn_w, ix.kn_b, k_pool, md.slot_mapping_fp8, md.slot_mapping_scale,
                is_neox=c.indexer_is_neox, compiled=qk,
            )  # fmt: skip
            return q_fp8, weights, None
        cute_dsl_indexer_q_side_fused(
            q, pos, c._idx_cos_sin, wq, c.idx_weight_scale, q_fp8, weights,
            is_neox=c.indexer_is_neox, compiled=qside,
        )  # fmt: skip
        return q_fp8, weights, k_pre

    def qk_handle(self, ph: Phase, kc, kb: torch.Tensor, rows: int):
        """The q side + K-cache update launcher for this call (`TRTLLM_GLM52_WIDE_IX_QK`), or
        None: only with the layer's cache (`kc`) on an MTP3 / MTP5 wide decode step (rows
        above 64), once the K update's first eager call has validated the slot mappings
        (`Indexer.write_k`), with 8-byte aligned contiguous keys (its only key loads)."""
        c = self.c
        if kc is None or not c._ix_qk_compiled or not self._k_checked:
            return None
        if not (ph.wide and rows > _DECODE_MAX_ROWS):
            return None
        if not kb.is_contiguous() or kb.data_ptr() % 8:
            return None
        return c._ix_qk_compiled[kc[1].is_contiguous()]

    def write_k(self, md, k_pool, k_pre, rows: int) -> None:
        """Norm, rope, fp8-pack and append this step's keys in one launch."""
        c = self.c
        kb, kn_w, kn_b, pos = k_pre
        cute_dsl_indexer_k_cache_update(
            kb,
            kn_w,
            kn_b,
            pos,
            c._idx_cos_sin.view(-1, 2, c.rope_dim // 2),
            k_pool,
            md.slot_mapping_fp8,
            md.slot_mapping_scale,
            rows,
            is_neox=c.indexer_is_neox,
            compiled=c._ix_k_compiled[k_pool.is_contiguous()],
            # Its validation reduces the slot mappings to a host bool, a sync that invalidates
            # a capture: checked once, on the first eager call.
            validate=not self._k_checked,
        )
        self._k_checked = True

    def decode(
        self,
        md,
        k_pool,
        q_fp8,
        weights,
        topk_buf,
        ctx: int,
        seqs: int,
        gen_p: int,
        split_memo: dict | None = None,
    ):
        """Score the generation rows against the paged K cache and select `index_topk`.
        `ctx` is the number of context sequences ahead of them in the batch. `split_memo`
        holds the MTP5 atom split's metadata across one step's indexer layers (per call
        without it)."""
        c = self.c
        # Cached plus all `gen_p` of this step's tokens; both consumers take the same vector,
        # which keeps the selector's causal bound inside the columns the scorer wrote.
        kv_lens = md.kv_lens_cuda_runtime[ctx : ctx + seqs]
        atom = _MQA_ATOM_SPLIT.get(gen_p)
        if atom is None:
            logits = cute_dsl_fp8_paged_mqa_logits(
                torch.reshape(q_fp8, [seqs, gen_p, c.idx_heads, c.idx_dim]),
                k_pool,
                weights,
                kv_lens,
                md.indexer_k_cache_block_offsets[ctx : ctx + seqs],
                md.scheduler_metadata_buffer,
                c._max_ctx_len,
            )
        else:
            # MTP5's 6 rows of a sequence as two pseudo-sequences of 3 (the kernel spills at
            # 6): the same logits rows, bit for bit. Lengths, table and schedule depend only
            # on the step, so one build serves every indexer layer of the step.
            factor = gen_p // atom
            key = (ctx, seqs, gen_p)
            meta = None if split_memo is None else split_memo.get(key)
            if meta is None:
                meta = cute_dsl_fp8_paged_mqa_logits.split_metadata(
                    kv_lens, md.indexer_k_cache_block_offsets[ctx : ctx + seqs], factor
                )
                if split_memo is not None:
                    split_memo[key] = meta
            c._count("mqa_atom_split", True)
            logits = cute_dsl_fp8_paged_mqa_logits(
                torch.reshape(q_fp8, [seqs * factor, atom, c.idx_heads, c.idx_dim]),
                k_pool,
                weights,
                *meta,
                c._max_ctx_len,
            )
        # The stock engine's GVR decode top-k where the engine's sparse config asks for it.
        gvr = c._dec_topk_gvr_live
        if gvr and self.gvr_launch(md, logits, kv_lens, topk_buf, gen_p):
            return
        compiled = c._ix_topk_dec_compiled.get(gen_p)
        # Ours, or (a width without a handle) the trtllm op.
        c._count("ix_topk_dec", compiled is not None)
        if compiled is None:
            indexer_topk_decode(
                logits, kv_lens, topk_buf, gen_p, c.idx_topk,
                radix_aux_indices=md.radix_aux_indices, radix_aux_logits=md.radix_aux_logits,
            )  # fmt: skip
        else:
            # `seqs * gen_p` logits rows on both scorer paths (the atom split reshapes rows only).
            c._count("ix_topk_dec_cluster", ix_topk_decode_uses_cluster(compiled, seqs * gen_p))
            cute_dsl_indexer_topk_decode(
                logits, kv_lens, topk_buf, top_k=c.idx_topk, next_n=gen_p, compiled=compiled
            )

    def gvr_launch(self, md, logits, kv_lens, topk_buf, gen_p: int) -> bool:
        """The generation rows' top-k on the stock engine's self-sampling GVR engine, the call
        the stock DSA indexer makes (`Indexer.sparse_attn_indexer`'s decode `c.top_k(...)`).
        False where it did not launch: a launcher missing inside a capture (none JIT-compiles
        there), which leaves that graph on ours.

        Row `b * gen_p + t` selects among columns `[0, kv_lens[b] - gen_p + t + 1)` (`kv_lens`
        counts this step's tokens, so the bound is the causal one our kernel uses); with `n`
        that many, it writes the exact top-`min(n, 2048)` set (ties at the k-th score kept in
        an unspecified order unless `gvr_deterministic_topk`), and a row with `n <= 2048` the
        identity `0..n-1`, each followed by -1: the valid prefix the sparse MLA decode reads.
        Tuning is frozen from `max_seq_len` (the scorer's width), with no host reads, so the
        launch is CUDA-graph replay safe. The engine's `radix_aux_*` scratch serves only the
        stock module's own fall-through (logits outside the engine's format gate)."""
        c = self.c
        try:
            c._dec_topk_gvr(
                logits,
                topk_buf,
                is_prefill=False,
                sequence_lengths=kv_lens,
                scan_lengths=kv_lens,
                next_n=gen_p,
                max_seq_len=c._max_ctx_len,
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
        c._count("ix_topk_dec_gvr", True)
        return True

    def prefill_select(self, md, k_pool, q_fp8, weights, topk_buf) -> None:
        """The context rows' selection into `topk_buf`: the engine's indexer prefill chunks,
        each scored against its gathered K block and selected per tile. With MLA chunked
        prefill and at least `q_split_threshold` context queries, the engine plans the chunks
        for this rank's share of the queries only (`indexer_prefill_q_split`); the other
        ranks' top-k rows are gathered after the loop, as the engine's own indexer does. A K
        block wider than any built so far (a long cached prefix, or packed requests) widens
        the top-k build here: context steps are never captured."""
        c = self.c
        q_split = getattr(md, "indexer_prefill_q_split", None)
        for chunk in md.indexer_prefill_chunks:
            n_k = chunk.k_token_end - chunk.k_token_start
            if chunk.token_end <= chunk.token_start:
                continue
            if n_k <= 0:
                # No compressed KV for these queries: no valid index, as the engine.
                topk_buf[chunk.token_start : chunk.token_end].fill_(-1)
                continue
            if n_k > c._ix_topk_pre_compiled.max_columns:
                c._ix_topk_pre_compiled = build_ix_topk_prefill(
                    torch.cuda.current_device(), top_k=c.idx_topk, max_columns=n_k
                )
            ck, cs = indexer_k_cache_gather_op(
                k_pool,
                md.slot_mapping_fp8_fullkv,
                md.slot_mapping_scale_fullkv,
                chunk.k_token_start,
                n_k,
                c.idx_dim,
            )
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
                    logits, ks, ke, topk_buf[lo:hi],
                    top_k=c.idx_topk, compiled=c._ix_topk_pre_compiled,
                )  # fmt: skip
        if q_split is not None:
            q_lo, q_hi, q_sizes = q_split
            topk_buf[: sum(q_sizes)] = allgather(
                topk_buf[q_lo:q_hi], md.mapping, dim=0, sizes=q_sizes
            )

    @staticmethod
    def share_reuses(md) -> bool:
        return _mtp_share_reuses(md)

    @staticmethod
    def share_load(md, dst, seqs: int) -> None:
        _mtp_share_load(md, dst, seqs)

    @staticmethod
    def share_store(md, topk_buf, ctx: int, tc: int, seqs: int, gen_p: int) -> None:
        _mtp_share_store(md, topk_buf, ctx, tc, seqs, gen_p)


class Mlp:
    def __init__(self, core) -> None:
        self.c = core

    def dense(self, ph: Phase, o, gu, dn, w8=None, x8=None):
        """A dense layer's MLP, this rank's partial. Context rows on the MXF8 trio where it
        built (`w8`: the layer's quantized gate|up and down; `x8`: `o`'s MXF8 operand from
        the collective, else gate|up quantizes `o` itself): the SwiGLU writes the down GEMM's
        operand, no bf16 intermediate. Decode rows: the bf16 GEMMs and SwiGLU."""
        c = self.c
        if w8 is not None and c.op.gemm.mxf8_takes(ph, "dense_gate_up", o, w8[0]):
            rows = int(o.shape[0])
            if len(w8) > 2:
                # The SwiGLU into the down GEMM's operand in the gate|up GEMM's epilogue.
                act8 = cute_dsl_dense_up_swiglu_mxf8(
                    o, w8[2], *_dense_up8_op.operand(rows, o.device), c._dense_sw8, pre=x8
                )
                c._count("dense_up_swiglu", True)
            else:
                gate_up = c.op.gemm.project(ph, "dense_gate_up", o, gu, w8=w8[0], pre=x8)
                act8 = cute_dsl_swiglu_mxf8(
                    gate_up,
                    *_swiglu8_op.operand(rows, gate_up.shape[1] // 2, o.device),
                    c._swiglu8,
                )
            down = c._proj_mxf8["dense_down"]
            out = torch.empty([rows, down.n], dtype=torch.bfloat16, device=o.device)
            c._count("dense_mxf8", True)
            return cute_dsl_projection_mxf8(act8[0], w8[1], out, down, pre=act8)
        act = cute_dsl_dense_swiglu(c.op.gemm.project(ph, "dense_gate_up", o, gu), c._dense_swiglu)
        return c.op.gemm.project(ph, "dense_down", act, dn)

    def shared_after_collective(self, exact: bool) -> None:
        """On an MTP3 / MTP5 wide decode step (`Moe.decode_exact`), called on the side
        stream right after its fork, ahead of the shared expert: one plain (non-PDL) kernel,
        a 1-element fill, so the shared expert's first kernel no longer hangs off the pre-MoE
        collective by a programmatic edge. In the captured decode graph both the router (main
        stream) and the shared expert (side stream) are that collective's PDL dependents, and
        only one of them launches at its early release: at 192 rows the shared up GEMM took
        it and the router started only when the collective ended; at 128 rows both launched
        before it ended and the shared GEMM slowed the router (2-3x). Behind this node the
        router always takes the early launch and the shared expert starts once the collective
        completes, in the shadow of the top-k and FC1 (it only has to finish before FC2).
        4-GPU decode-step chain (real AR, 3 MoE layers per step, 280 alternating replays):
        -30.6 us per step at 128 rows, -17.9 us at 192. Ordering only: every byte is the
        same. `exact` False (MTP1, context steps, rows the
        prefill pair does not take unpadded) launches nothing."""
        if not exact:
            return
        c = self.c
        fence = getattr(self, "_shared_order_fence", None)
        if fence is None:
            # Allocated on the first (eager, warmup) call, then reused: only the launch
            # matters, so one persistent element serves every layer and step.
            fence = self._shared_order_fence = torch.zeros(
                1, dtype=torch.int32, device=torch.cuda.current_device()
            )
        else:
            fence.zero_()
        c._count("moe_shared_after_ar", True)

    def shared(self, ph: Phase, x, gu, gu_il, dn, out=None, trunk=False, x8=None, w8=None):
        """The shared expert, this rank's slice, a partial sum. `gu_il` is this layer's
        interleaved gate/up weight; `out` is the routed experts' output buffer when FC2 will
        accumulate onto it. A trunk layer's context rows take the prefill shared kernels
        when they built; the MTP layer's keep the two-launch path. With `x8` (`x`'s MXF8
        operand from the pre-MoE collective) and `w8` (the layer's MXF8 gate|up) the prefill
        gate|up + SwiGLU runs as the MXF8 GEMM with the SwiGLU in its epilogue. A
        `Phase.wide` step's rows above 64 take the MTP decode cells, trunk and MTP layer:
        at 128 / 192, else on the next one up (`Gemm.wide_cell`; the gate|up cells take any row
        count, the down cell is re-pinned). Rows up to 64 without a fused cell where
        `Gemm.cell_up` holds: the fused cell built at that count (`_SHARED_FUSED_UP_SEQS`), else
        the two-launch path on the projection cells next up."""
        c = self.c
        rows = int(x.shape[0])
        k = c._moe_k
        wide = None
        if ph.wide and rows > _DECODE_MAX_ROWS:
            wide = c.op.gemm.wide_cell("shared_gu", c._shared_gu_swiglu_mtp, rows)
            c._count("dec_wide", wide is not None)
        if (
            trunk
            and wide is None
            and k.shared_down_pre is not None
            and _MOE_PREFILL_MIN <= rows <= _MOE_MAX_T
        ):
            xs = x if _direct_ok(x, torch.bfloat16) else x.contiguous()
            # The prefill kernels take 128-byte aligned rows; a contiguous view that is not
            # takes the two-launch path.
            if xs.dtype is torch.bfloat16 and xs.data_ptr() % 128 == 0:
                res = out
                if res is None:
                    res = torch.empty((rows, x.shape[1]), dtype=torch.bfloat16, device=x.device)
                mid = c._moe_mid[:rows]
                if x8 is not None and w8 is not None:
                    cute_dsl_shared_up_swiglu_mxf8(xs, w8, mid, c._shared8, pre=x8)
                    c._count("shared_mxf8", True)
                else:
                    cute_dsl_shared_gate_up_swiglu(xs, gu_il, mid, k.shared_up_pre)
                return cute_dsl_projection(mid, dn.t(), res, k.shared_down_pre)
        runner = c._shared_gu_swiglu.get(rows) if wide is None else wide
        if runner is None and c._shared_gu_swiglu_up and c.op.gemm.cell_up(ph, rows):
            # An MTP3 / MTP5 decode count of 1..3 sequences: its own fused cell.
            runner = c._shared_gu_swiglu_up.get(rows)
            c._count("dec_cell_up", runner is not None)
        if runner is None:
            gate_up = c.op.gemm.project(ph, "shared_gate_up", x, gu)
            swiglu = c._rows.swiglu
            hit = swiglu is not None and _swiglu_rows_op.takes(gate_up)
            c._count("swiglu_rows", hit)
            if hit:
                act = cute_dsl_swiglu_rows(gate_up, swiglu)
            else:
                act = flashinfer_silu_and_mul(gate_up)
        else:
            act = torch.empty([rows, gu.shape[1] // 2], dtype=x.dtype, device=x.device)
            cute_dsl_shared_gate_up_swiglu(x, gu_il, act, runner)
        return c.op.gemm.project(ph, "shared_down", act, dn, out=out)


class Moe:
    def __init__(self, core) -> None:
        self.c = core
        # The slot pair the router, top-k and grouped GEMMs alternate over, one per layer.
        self.slot = 0
        # The early-release top-k launched for the current call (`experts` checks it).
        self._topk_signaled = False
        # The maps the replica top-k wrote, run on the replica pair.
        self._rep_maps = None

    def pair_ok(self, rows: int) -> bool:
        """Whether `rows` takes the prefill grouped GEMM pair: a context row count above the
        decode kernels' 64."""
        c = self.c
        return c._moe_k.pair is not None and _MOE_PREFILL_MIN <= rows <= _MOE_MAX_T

    def decode_exact(self, rows: int) -> bool:
        """Whether a wide decode step's routed experts run `rows` unpadded below 513 (an engine
        with the wide decode cells, `wide_cells`: 65..192 rows at MTP3 / MTP5 batch 32), on
        the prefill front end and pair. The prefill top-k writes the maps for any row count,
        so nothing is padded, re-routed, cleared or merged: 192 rows ran as 256 (a third more
        M, the top-k's maps discarded for a routing launch, a clear and a merge). Context
        steps keep the power-of-two padding."""
        c = self.c
        return (
            c.wide_cells
            and _MOE_PREFILL_MIN <= rows < _MOE_PAIR_EXACT_FROM
            and c._moe_k.topk_pre is not None
            and self.pair_ok(rows)
        )

    def topk_wide(self, rows: int):
        """The wide decode top-k handle for `rows` (65..256), or None (not built: MTP1, no
        speculation, or a failed build; or rows outside its domain)."""
        c = self.c
        if not _moe_topk_op.WIDE_MIN_ROWS <= rows <= _moe_topk_op.WIDE_MAX_ROWS:
            return None
        return c._moe_k.topk_wide.get(_moe_topk_op.wide_range(rows))

    def padded_rows(self, rows: int, exact: bool = False) -> int:
        """The row count routing and the grouped GEMMs see for `rows`: a power of two, or the
        exact count on the prefill pair from `_MOE_PAIR_EXACT_FROM` up, or on a wide decode
        step (`exact`, see `Moe.decode_exact`). `Moe.begin_layer` and `Moe.experts` both
        ask here, so the shared expert's rows and FC2's agree."""
        if (exact or rows >= _MOE_PAIR_EXACT_FROM) and self.pair_ok(rows):
            return rows
        return _pow2(rows)

    def dec_unpadded(self, rows: int) -> bool:
        """Whether a trunk MoE layer's decode width `rows` (1..64) runs unpadded on the fused
        top-k's maps (`TRTLLM_GLM52_MOE_DEC_UNPAD`). Only the trunk asks: there the collective
        writes the NVFP4 input (`Comm.prequant`) and the shared expert the output's partial
        (`Moe.begin_layer`), so no per-width quantize or clear runs (both are compiled for
        the powers of two only). The MTP layer quantizes and clears itself and keeps the
        padding. Every decode width has its own top-k handle (`k.topk`, `k.topk_rep`), whose
        maps the grouped GEMMs take at that width: they are compiled for the capacity and
        read the live tile count from the maps."""
        c = self.c
        return c._moe_dec_unpad and 1 <= rows <= _DECODE_MAX_ROWS and rows in c._moe_k.topk

    def begin_layer(self, rows: int, exact: bool = False):
        """Open one layer: advance the slot for the router, top-k and grouped GEMMs, and
        return the routed output rows for the shared expert to write into, or None. At a
        width the GEMMs do not pad: a power of two, or any count on the prefill pair from
        513 (or on a wide decode step, `exact`); and at a padded decode width up to 64
        (`_moe_acc_padded`). Called before the shared branch forks, so the shared expert
        writes the slot FC2 accumulates into.

        A padded width is safe for the same reasons as the others: row `i` of the shared
        expert is token `i`, and FC2 adds into `out[expanded_idx // top_k]` for the
        selections the maps hold, nothing else. The padding tokens follow the real ones and
        route to a non-local expert (`pad_expert_id`), so no tile scatters into their rows,
        which are neither cleared nor written and are sliced off. The ordering is unchanged:
        the shared expert writes on the side stream, which FC2 joins (`Moe.experts`), and
        the slot alternates per layer as at every other width. What a padded width adds
        (the staging copies, the routing kernel, the top-k's late FC1 release) does not
        touch the output."""
        c = self.c
        self.slot ^= 1
        if rows > _MOE_MAX_T:
            return None
        if self.padded_rows(rows, exact) != rows and not self.dec_unpadded(rows):
            if rows > _DECODE_MAX_ROWS or not c._moe_acc_padded:
                return None
            c._count("moe_acc_padded", True)
        return c._moe_slots[self.slot].out[:rows]

    def router(self, x, m, slot, exact: bool = False):
        """Router logits in the slot's buffer, or None (the stock projection then runs it).
        Up to 64 rows on the decode router, where above 5 rows these are `[rows, 4, 256]`
        partials only the fused top-k reads; on a wide decode step (`exact`) the same
        partials from the wide decode router when it and the wide top-k built; above, on the
        prefill router (`[rows, 256]`)."""
        c = self.c
        rows = int(x.shape[0])
        s = c._moe_slots[slot]
        if rows > _DECODE_MAX_ROWS:
            k = c._moe_k
            pre = k.router_pre
            wide = exact and k.router_wide is not None and self.topk_wide(rows) is not None
            if (pre is None and not wide) or rows > _MOE_MAX_T:
                return None
            if not _direct_ok(x, torch.bfloat16):
                x = x.contiguous()
            if x.dtype is not torch.bfloat16 or x.data_ptr() % 16:
                return None
            if wide:
                out = s.router_wide_out[:rows]
                cute_dsl_moe_router(x, m.router, out, k.router_wide)
                c._count("moe_router_wide", True)
                return out
            out = s.logits[:rows]
            cute_dsl_moe_router(x, m.router, out, pre)
            return out.view(rows, -1)
        if _direct_ok(x, torch.bfloat16):
            xs = x
        else:
            xs = s.router_x[:rows]
            c.op.rows.copy(xs, x)
        out = (
            s.router_out[:rows]
            if rows <= _moe_router_op.SIMT_MAX_ROWS
            else s.router_partials[:rows]
        )
        cute_dsl_moe_router(xs, m.router, out, c._moe_k.router[rows])
        return out

    def topk_wide_rep(self, rows: int, layer: int):
        """`(handle, hot ids)` of the wide replica top-k for a wide decode step's `rows` at trunk
        MoE layer `layer`, or None: not built (MTP1, no speculation, replicas off) or a layer
        without hot ids (the MTP layer)."""
        c = self.c
        reps = getattr(c._moe_k, "topk_wide_rep", None)
        hot = c._ctx_hot_ids.get(layer) if reps else None
        if hot is None:
            return None
        handle = reps.get(_moe_topk_op.wide_range(rows))
        return None if handle is None else (handle, hot)

    def topk(self, logits, in_slot: bool, m, slot, layer: int, exact: bool = False):
        """Sigmoid top-k with bias correction, writing this layer's tile maps: `(weights,
        ids, maps)` in the slot's buffers, or None (`noaux_tc_op` then selects). Up to 64 rows
        on the decode kernel; above, on the wide decode kernel for a wide decode step
        (`exact`, 65..256 rows, same ids, weights and tile layout as the prefill kernel's;
        a trunk layer with hot ids on its replica variant, `Moe.topk_wide_rep`) and on the
        prefill one otherwise. `in_slot`: the logits are the slot's router output,
        read in place; others are staged into it first. Matches `noaux_tc_op` on this
        ungrouped routing: same sets, weights to 6e-8. A decode or wide handle built with
        `expert_hist` also writes `layer`'s row of the L2 prefetch's counts."""
        c = self.c
        rows = int(logits.shape[0])
        s = c._moe_slots[slot]
        k = c._moe_k
        sig = None
        if self.dec_unpadded(rows) and layer < c.num_layers:
            sig = getattr(k, "topk_sig", {}).get(rows)
        self._topk_signaled = sig is not None
        if rows in k.topk_rep:
            # Decode expert replicas: this layer's hot ids (none on the MTP layer).
            handle = sig if sig is not None else k.topk_rep[rows]
            hot = c._dec_hot_ids.get(layer, c._dec_hot_none)
            if hot is not c._dec_hot_none:
                c._count("moe_dec_replica", True)
            return cute_dsl_moe_topk_routing_rep(
                logits,
                m.router_bias,
                hot,
                c.expert_offset,
                c.routed_scale,
                s.topk_out[rows],
                handle,
                expert_hist=c._pf.hist[layer] if handle.expert_hist else None,
            )
        if rows <= _DECODE_MAX_ROWS:
            handle = sig if sig is not None else k.topk[rows]
            return cute_dsl_moe_topk_routing(
                logits,
                m.router_bias,
                c.expert_offset,
                c.routed_scale,
                s.topk_out[rows],
                None,
                handle,
                expert_hist=c._pf.hist[layer] if handle.expert_hist else None,
            )
        if k.topk_pre is None or rows > _MOE_MAX_T:
            return None
        if not in_slot:
            lg = s.logits[:rows].view(rows, -1)
            c.op.rows.copy(lg, logits.reshape(rows, -1))
            logits = lg
        wide = self.topk_wide(rows) if exact else None
        planes = 1 if logits.dim() == 2 else int(logits.shape[1])
        if wide is not None and wide.logit_splits == planes:
            rep = self.topk_wide_rep(rows, layer)
            if rep is not None and rep[0].logit_splits == planes:
                # Wide decode expert replicas: the same selection, maps over this rank's home
                # and replica slots; `Moe.experts` runs them on the replica pair. Every rank
                # takes this branch for the same rows and layer.
                handle, hot = rep
                c._count("moe_wide_replica", True)
                if handle.cluster == 16:
                    c._count("moe_topk_wide_cl16", True)
                outputs = {"topk_weights": s.w, "topk_indices": s.sel, **s.maps}
                selected = cute_dsl_moe_topk_routing_wide_rep(
                    logits, m.router_bias, hot, c.expert_offset, c.routed_scale, outputs,
                    handle, expert_hist=c._pf.hist[layer] if handle.expert_hist else None,
                )  # fmt: skip
                self._rep_maps = selected[2]
                return selected
            # One fused launch for the three of the prefill top-k (histogram, prefix,
            # scatter); the maps it writes are FC1's as they stand.
            c._count("moe_topk_wide", True)
            if wide.cluster == 16:
                c._count("moe_topk_wide_cl16", True)
            outputs = {"topk_weights": s.w, "topk_indices": s.sel, **s.maps}
            return cute_dsl_moe_topk_routing(
                logits, m.router_bias, c.expert_offset, c.routed_scale, outputs, None, wide,
                expert_hist=c._pf.hist[layer] if wide.expert_hist else None,
            )  # fmt: skip
        hot = c._ctx_hot_ids.get(layer) if k.topk_pre_rep is not None else None
        if (
            hot is not None
            and not exact
            and _moe_topk_pre_rep_op.serves(rows)
            and self.padded_rows(rows) == rows
            and self.pair_ok(rows)
        ):
            # Context replicas: only where the GEMMs run these maps unpadded on the pair.
            # Every rank takes the same branch for the same rows and layer.
            outputs = {"topk_weights": s.w, "topk_indices": s.sel, **s.topk_rep_scratch, **s.maps}
            selected = cute_dsl_moe_prefill_topk_routing_rep(
                logits, m.router_bias, hot, c.expert_offset, c.routed_scale, outputs,
                k.topk_pre_rep,
            )  # fmt: skip
            self._rep_maps = selected[2]
            c._count("ctx_ar_rep", True)
            return selected
        outputs = {"topk_weights": s.w, "topk_indices": s.sel, **s.topk_scratch, **s.maps}
        return cute_dsl_moe_topk_routing(
            logits, m.router_bias, c.expert_offset, c.routed_scale, outputs, None, k.topk_pre
        )

    def experts(
        self,
        x,
        topk_ids,
        topk_w,
        topk_maps,
        m,
        rows: int,
        prequantized,
        join,
        began,
        acc,
        layer,
        exact=False,
    ):
        """One chunk's routed round trip; returns the slot's output rows.

        `topk_maps`: the maps a top-k kernel wrote for these ids, with `topk_w` in its own
        buffer; trusted only at an unpadded row count (maps built for another row count are
        an illegal address in the grouped GEMMs, not a wrong answer). `prequantized`: the
        NVFP4 input already sits in `xq` / `xsf`. `began`: `Moe.begin_layer` advanced the
        slot; otherwise this call uses the current slot and advances it after. `acc`: the
        output already holds the shared expert's partial. `join`: the stream writing that
        partial, waited on between FC1 and FC2 so the shared branch overlaps router, top-k
        and FC1. `layer` is this call site's slot of the decode FC2's tile counter: a launch
        of up to 64 tokens assigns its tiles dynamically, each layer once per forward.
        `exact`: a wide decode step (`Moe.decode_exact`), unpadded below 513 rows.
        """
        c = self.c
        k = c._moe_k
        dev = x.device
        # Pad to a power of two below 513 rows: the routing schedule is keyed by
        # `ceil(rows / 32)` there, and arbitrary widths would compile about twenty. Padding
        # rows route to a non-local expert at weight 0 (`pad_expert_id`), so routing drops
        # them, and are sliced off. The prefill pair takes 513 rows and up exactly, and a
        # wide decode step any count: its prefill top-k's maps need no routing launch.
        use_pair = self.pair_ok(rows)
        padded = self.padded_rows(rows, exact)
        if (
            padded != rows
            and began
            and acc
            and prequantized
            and topk_maps is not None
            and self.dec_unpadded(rows)
        ):
            # A trunk decode width that is not a power of two: the top-k's maps at the exact
            # rows, no padding, staging or routing launch (`Moe.dec_unpadded`).
            padded = rows
            c._count("moe_dec_unpad", True)
        if self._topk_signaled:
            self._topk_signaled = False
            if padded != rows:
                # Fail fast: a routing launch after an early-release top-k reads the ids
                # mid-write (a hang, not a wrong answer).
                raise RuntimeError(
                    f"modeling_v2 GLM-5.2: early-release top-k at {rows} rows took the padded path"
                )
            c._count("moe_dec_unpad_signal", True)
        if padded != rows:
            pad = padded - rows
            if not prequantized:
                x = torch.cat([x, torch.zeros(pad, x.shape[1], dtype=x.dtype, device=dev)])
            pad_id = _moe_routing_op.pad_expert_id(c.expert_offset)
            topk_ids = torch.cat(
                [topk_ids, torch.full((pad, _MOE_TOP_K), pad_id, dtype=topk_ids.dtype, device=dev)]
            )
            topk_w = torch.cat(
                [topk_w, torch.zeros(pad, _MOE_TOP_K, dtype=topk_w.dtype, device=dev)]
            )
        slot = self.slot
        if not began:
            self.slot ^= 1
        s = c._moe_slots[slot]
        if not prequantized:
            # An activation the engine allocated inside a CUDA-graph capture is not safe to
            # hand over as a raw pointer, so a captured call stages it. An eager call (context
            # and mixed steps) whose `x` already is the contiguous, 16-byte aligned bf16
            # `[padded, hidden]` operand goes to the quantize directly, which takes it per
            # call: a `[T, 6144]` copy per MoE layer skipped (98 MB at T ~ 8k; 3.0-3.4 ms of
            # copies per context step, measured in a served run).
            if (
                x.dtype is torch.bfloat16
                and x.is_contiguous()
                and x.data_ptr() % 16 == 0
                and tuple(x.shape) == tuple(s.x[:padded].shape)
                and not torch.cuda.is_current_stream_capturing()
            ):
                c._count("moe_stage_skip", True)
            else:
                x_s = s.x[:padded]
                c.op.rows.copy(x_s, x)
                x = x_s
        # fp32, as both top-k kernels write it and FC2 reads it: a top-k kernel's own output
        # at an unpadded width goes in as it is.
        if topk_maps is None or padded != rows:
            w_s = s.w[:padded]
            c.op.rows.copy(w_s, topk_w)
            topk_w = w_s
        if topk_maps is not None and padded == rows:
            maps = topk_maps
        else:
            # Always staged: the routing kernels launch with PDL and do not wait for their
            # inputs, and the fused top-k signals its dependents before it writes the ids.
            # The copy puts a node no PDL edge can cross between the two.
            sel = s.sel[: padded * _MOE_TOP_K]
            c.op.rows.copy(sel, topk_ids.reshape(-1))
            maps = cute_dsl_moe_routing(
                sel, c.expert_offset, s.maps, s.counts, s.barrier, k.routing
            )
        xq, xsf = c._moe_xq[:padded], c._moe_xsf[:padded]
        if prequantized:
            # Written by the collective; the padding rows are stale by design: routed to a
            # non-local expert, no grouped GEMM tile reads them.
            pass
        elif padded <= _DECODE_MAX_ROWS:
            cute_dsl_moe_quantize_nvfp4(x, m.g1, xq, xsf, k.quant[padded])
        elif k.quant_pre is not None and padded <= _MOE_MAX_T:
            cute_dsl_moe_quantize_nvfp4(x, m.g1, xq, xsf, k.quant_pre)
        else:
            xq, xsf = fp4_quantize(x, m.g1, _MOE_SF_VEC, False, False)
        # bf16: FC2 writes through a BFloat16 pointer whatever buffer it is handed.
        out = s.out[:padded]
        if not acc:
            if padded <= _DECODE_MAX_ROWS:
                cute_dsl_moe_zero(out, k.zero[padded])
            elif k.zero_pre is not None and padded <= _MOE_MAX_T:
                cute_dsl_moe_zero(out, k.zero_pre)
            else:
                out.zero_()
        # Context replicas: maps the replica top-k wrote address slots up to 64 + R - 1 and
        # run on the replica pair; any other maps are the home mapping.
        rep = use_pair and maps is self._rep_maps and k.pair_rep is not None
        self._rep_maps = None
        compiled, tiles = (k.pair_rep if rep else k.pair, s.pc) if use_pair else (k.experts, s.c)
        fc1_w, fc1_s, fc2_w, fc2_s = m.fc1_w, m.fc1_s, m.fc2_w, m.fc2_s
        alpha1, alpha2 = m.alpha1, m.alpha2
        e = compiled.experts if use_pair else fc1_w.shape[0]
        if fc1_w.shape[0] != e:
            # Decode expert replicas: the prefill pair maps home experts (or the first R
            # replica slots too, `rep`) and reads that prefix of the stacks (a contiguous
            # view of the same storage).
            fc1_w, fc1_s, fc2_w, fc2_s = fc1_w[:e], fc1_s[:e], fc2_w[:e], fc2_s[:e]
            alpha1, alpha2 = alpha1[:e], alpha2[:e]
        # The decode L2 prefetch forked at the o_proj end joins before FC2, with the shared
        # expert's stream.
        pf = c._pf.pending if c._pf is not None else None
        if pf is not None:
            c._pf.pending = None
            join = (join, pf) if join is not None else pf
        cute_dsl_moe_nvfp4_experts(
            xq, xsf, maps, topk_w, fc1_w, fc1_s, fc2_w, fc2_s,
            m.norm_const, alpha1, alpha2, out, tiles, s.sfc, join, compiled,
            None if use_pair else layer,
        )  # fmt: skip
        return out[:rows]

    def pf_fork(self, layer: int, m, rows: int, exact: bool = False) -> None:
        """Fork the L2 prefetch stream at a decode MoE layer's o_proj end and prefetch the FC1
        slices of the experts this layer's previous decode step routed the most tokens to.
        The routed call joins it before FC2 (`Moe.experts`). Only on the push top-k's widths
        (1..64 rows): the row bound is checked first, as the context steps call this too.
        Not on the MTP3 / MTP5 wide decode steps (`exact`): there the prefetch costs time
        (4-GPU chain with the real all-reduce, 2 prefetched MoE layers: +4.3 us at 128 rows,
        +2.6 us at 192, 200 alternating replays each)."""
        c = self.c
        pf = c._pf
        if pf is None or exact:
            return
        if not 1 <= rows <= _DECODE_MAX_ROWS or not c._moe_k.topk[rows].expert_hist:
            return
        pf.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(pf.stream):
            # The home prefix: the prefetch ranks this rank's 64 home experts (replica slots,
            # when present, follow them in the same storage).
            e = _moe_routing_op.LOCAL_EXPERTS
            cute_dsl_moe_l2_prefetch(pf.hist[layer], m.fc1_w[:e], m.fc1_s[:e], pf.compiled)
        pf.pending = pf.stream
        c._count("moe_l2_prefetch", True)

    def quant_first_ok(self, x, rows: int) -> bool:
        """Whether `Moe.experts` would quantize `x` itself, unstaged and unpadded, on the
        prefill quantize: then `Moe.routed` may issue that quantize ahead of the router."""
        c = self.c
        return (
            c._moe_k.quant_pre is not None
            and self.pair_ok(rows)
            and self.padded_rows(rows) == rows
            and x.dtype is torch.bfloat16
            and x.is_contiguous()
            and x.data_ptr() % 16 == 0
            and not torch.cuda.is_current_stream_capturing()
        )

    def routed(
        self,
        ph: Phase,
        x,
        m,
        rows: int,
        prequantized: bool,
        join,
        began: bool,
        acc: bool,
        layer: int,
        exact: bool = False,
    ):
        """This rank's 64-expert window. Router and top-k run on replicated hidden states,
        so every rank selects the same experts and the four windows tile the routing space
        once. No stock fallback for the grouped GEMMs: the weights are packed for these
        kernels alone. `exact`: a wide decode step, unpadded (`Moe.decode_exact`).

        Each chunk's result is a view of a slot's output buffer, and the slots alternate per
        chunk, so with more than one chunk each result is copied out before a later chunk
        reuses its slot. A single chunk returns the view itself."""
        c = self.c
        sizes = _moe_chunk_sizes(rows)
        result = None
        if len(sizes) > 1:
            result = torch.empty((rows, x.shape[1]), dtype=x.dtype, device=x.device)
        start = 0
        for chunk in torch.split(x, sizes, 0):
            n = int(chunk.shape[0])
            slot = self.slot
            # The NVFP4 quantize reads only the chunk: on an exactly sized prefill-pair chunk
            # it goes ahead of the router, before the shared expert's GEMM on the side stream
            # takes the SMs, and the latency-bound top-k overlaps the shared down instead
            # (-16 to -22 us per MoE layer at 7-8k rows on B300). Same kernel and bytes.
            first = not prequantized and self.quant_first_ok(chunk, n)
            if first:
                cute_dsl_moe_quantize_nvfp4(
                    chunk, m.g1, c._moe_xq[:n], c._moe_xsf[:n], c._moe_k.quant_pre
                )
                c._count("moe_quant_first", True)
            # fp32 logits: FC2 reads `token_final_scales` as fp32 and noaux_tc_op's weight
            # dtype follows its logits. bf16 runs clean on garbage.
            logits = self.router(chunk, m, slot, exact)
            in_slot = logits is not None
            if logits is None:
                logits = c.op.gemm.project(ph, "router", chunk, m.router_t, torch.float32)
            selected = self.topk(logits, in_slot, m, slot, layer, exact)
            if selected is None:
                topk_w, topk_ids = c.op.rows.route(logits, m.router_bias)
                maps = None
            else:
                topk_w, topk_ids, maps = selected
            part = self.experts(
                chunk, topk_ids, topk_w, maps, m, n, prequantized or first, join, began, acc,
                layer, exact,
            )  # fmt: skip
            if result is None:
                return part
            c.op.rows.copy(result[start : start + n], part)
            start += n
        return result

    def wide_exact(self, rows: int) -> bool:
        """`decode_exact` for an MTP3 / MTP5 wide decode step's trunk MoE layer, counted."""
        exact = self.decode_exact(rows)
        if exact:
            self.c._count("moe_dec_exact", True)
        return exact

    def open_layer(self, rows: int, exact: bool = False):
        """`(began, acc)` for a trunk MoE layer: up to `_MOE_MAX_T` rows the layer opens its
        slot ahead of the shared branch (`begin_layer`), and `acc` is the routed output the
        shared expert writes into (None where FC2 does not accumulate onto it); above, each
        chunk advances the slot itself and `acc` is None."""
        if rows > _MOE_MAX_T:
            return False, None
        return True, self.begin_layer(rows, exact)

    def combine(self, routed, shared, acc, rows: int, main, side):
        """The MoE layer's output: the routed rows as they stand where FC2 accumulated onto the
        shared expert's partial (`acc`), else joined with the shared branch and added, on the
        merge kernel where it takes the rows."""
        c = self.c
        if acc is not None:
            return routed
        main.wait_stream(side)
        merge = c._moe_k.merge
        if (
            merge is not None
            and _MOE_PREFILL_MIN <= rows <= _MOE_MAX_T
            and routed.shape == shared.shape
            and routed.is_contiguous()
        ):
            if not shared.is_contiguous():
                packed = torch.empty_like(shared, memory_format=torch.contiguous_format)
                shared = c.op.rows.copy(packed, shared)
            return cute_dsl_moe_merge(routed, shared, merge)
        return c.op.rows.add(routed, shared)


class Mtp:
    def __init__(self, core) -> None:
        self.c = core
        self.slot = 0

    def front(self, ph: Phase, e, h, rows: int):
        """`eh_proj(concat(enorm(e), hnorm(h)))`. At the specialised row counts each rank norms
        one 3072-wide quarter of the concat -- ranks 0-1 halves of `e`, ranks 2-3 halves of
        `h` -- and the projection's K-split partials are summed by the collective. MTP5's
        192 draft-step-0 rows take it on a `Phase.wide` step only, and so do that step's
        other row counts in 65..191 (a batch of fewer than 32 sequences), on the next
        instance up (128 / 192) at their exact rows (`Gemm.wide_cell`); with `_wide_bucket` off
        they keep bf16 (the replicated GEMM below), never the MXF8 context path."""
        c = self.c
        mw = c._mtp
        k = _mtp_eh.LOCAL_INPUT_SIZE
        wide = ph.wide and rows in _mtp_eh.MTP_ROWS
        runner = mw["ehproj"]
        ksplit = rows in _mtp_eh.SUPPORTED_M_VALUES or wide
        if not ksplit and ph.wide and rows > _DECODE_MAX_ROWS:
            cells = dict.fromkeys(_mtp_eh.WIDE_ROWS, runner)
            runner = c.op.gemm.wide_cell("mtp_eh", cells, rows, _mtp_eh.at_rows)
            ksplit = runner is not None
        elif not ksplit and c.op.gemm.cell_up(ph, rows):
            # An eager draft step's or an MTP3 / MTP5 decode step's count: the next supported
            # instance up at its exact rows (`at_rows` picks the instance; one runner holds
            # them all).
            runner = c.op.gemm.draft_cell(
                ph, "mtp_eh", rows, {_DECODE_MAX_ROWS: runner}, {}, _mtp_eh.at_rows
            )
            ksplit = runner is not None
        if ksplit and e.is_contiguous() and h.is_contiguous():
            src, nw = (e, mw["enorm"]) if c.tp_rank < 2 else (h, mw["hnorm"])
            local = torch.empty([rows, k], dtype=e.dtype, device=e.device)
            cute_dsl_mtp_norm_shard(src, nw, local, (c.tp_rank % 2) * k, c.eps, mw["norm_shard"])
            return c.op.comm.reduce(cute_dsl_mtp_eh_proj(local, mw["eh_local"], runner))
        eh8 = c._mtp_eh8
        if eh8 is not None and rows > _DECODE_MAX_ROWS and not ph.wide and _eh8_op.supports(e, h):
            # The two norms write the MXF8 GEMM's operand halves: no bf16 norms, no concat
            # copies, no activation quant.
            q8 = _eh8_op.operand(rows, e.device)
            cute_dsl_mtp_eh_norm_cat_mxf8(e, mw["enorm"], h, mw["hnorm"], c.eps, *q8, eh8.norm)
            runner = c._proj_mxf8["mtp_eh"]
            out = torch.empty([rows, runner.n], dtype=torch.bfloat16, device=e.device)
            c._count("mtp_eh_mxf8", True)
            return cute_dsl_projection_mxf8(q8[0], eh8.w8, out, runner, pre=q8)
        # The concat written by two copies into one buffer: the copy kernel takes each
        # half (a column slice of the buffer), torch.cat would be one more launch.
        en, hn = c.op.rows.rmsnorm(e, mw["enorm"]), c.op.rows.rmsnorm(h, mw["hnorm"])
        cat = torch.empty([rows, 2 * en.shape[1]], dtype=en.dtype, device=en.device)
        c.op.rows.copy(cat[:, : en.shape[1]], en)
        c.op.rows.copy(cat[:, en.shape[1] :], hn)
        # The replicated projection: the prefill GEMM above the decode row counts, else cuBLAS.
        return c.op.gemm.project(ph, "mtp_eh", cat, mw["eh"])

    def experts(self, x, topk_ids, topk_w, rows: int, frontend: bool):
        """One MTP chunk's bf16 round trip; returns the slot's output rows. With `frontend`
        the router and the fused top-k (which also clears the output) run here on the
        input in place; otherwise the caller's ids and weights select, staged with the input
        into the slot's buffers."""
        c = self.c
        mw, k = c._mtp, c._mtp_k
        dev = x.device
        padded = _pow2(rows)
        if padded != rows:
            pad = padded - rows
            x = torch.cat([x, torch.zeros(pad, x.shape[1], dtype=x.dtype, device=dev)])
            if not frontend:
                # Padding rows go to a non-local expert at weight 0, see `pad_expert_id`.
                pad_id = _moe_routing_op.pad_expert_id(c.expert_offset)
                topk_ids = torch.cat(
                    [
                        topk_ids,
                        torch.full((pad, _MOE_TOP_K), pad_id, dtype=topk_ids.dtype, device=dev),
                    ]
                )
                topk_w = torch.cat(
                    [topk_w, torch.zeros(pad, _MOE_TOP_K, dtype=topk_w.dtype, device=dev)]
                )
        slot = self.slot
        self.slot ^= 1
        s = c._mtp_slots[slot]
        out = s.out[:padded]
        if frontend:
            # The router and gather GEMM read the activation in place; weights and maps come
            # from the top-k's outputs, and the top-k zeroed the output.
            if padded <= _moe_router_op.SIMT_MAX_ROWS:
                logits = s.router_out[:padded]
            else:
                logits = s.router_partials[:padded]
            cute_dsl_moe_router(x, mw["router"], logits, c._moe_k.router[padded])
            topk_w, _, maps = cute_dsl_moe_topk_routing(
                logits,
                mw["router_bias"],
                c.expert_offset,
                c.routed_scale,
                s.topk_out[padded],
                out,
                k.topk[padded],
            )
            xs = x
        else:
            # The routing kernel does not wait before reading the ids, and `noaux_tc_op`
            # releases it before its stores are visible: these copies are what order them.
            xs = s.x[:padded]
            c.op.rows.copy(xs, x)
            w_s = s.w[:padded]
            c.op.rows.copy(w_s, topk_w)
            topk_w = w_s
            sel = s.sel[: padded * _MOE_TOP_K]
            c.op.rows.copy(sel, topk_ids.reshape(-1))
            maps = cute_dsl_moe_routing(
                sel, c.expert_offset, s.maps, s.counts, s.barrier, c._moe_k.routing
            )
            out.zero_()
        cute_dsl_mtp_moe_experts(xs, maps, topk_w, mw["fc1"], mw["fc2"], out, s.c, k.experts)
        return out[:rows]

    def routed(self, ph: Phase, x, rows: int):
        """The MTP module's window, bf16. Up to 64 rows the front end runs with the experts;
        above, the stock router projection and noaux_tc_op select. With more than one
        chunk each result is copied out before a later chunk reuses its slot."""
        c = self.c
        mw = c._mtp
        sizes = _moe_chunk_sizes(rows)
        result = None
        if len(sizes) > 1:
            result = torch.empty((rows, x.shape[1]), dtype=x.dtype, device=x.device)
        start = 0
        for chunk in torch.split(x, sizes, 0):
            n = int(chunk.shape[0])
            if n <= _DECODE_MAX_ROWS:
                part = self.experts(chunk, None, None, n, frontend=True)
            else:
                logits = c.op.gemm.project(ph, "router", chunk, mw["router_t"], torch.float32)
                topk_w, topk_ids = c.op.rows.route(logits, mw["router_bias"])
                part = self.experts(chunk, topk_ids, topk_w, n, frontend=False)
            if result is None:
                return part
            c.op.rows.copy(result[start : start + n], part)
            start += n
        return result

    def wide_exact(self, rows: int) -> bool:
        """Whether an MTP3 / MTP5 wide decode step's MTP layer runs its routed experts on the
        exact rows (`Moe.decode_exact`): only on the trunk's NVFP4 path (`_mtp_moe_fp4`)."""
        c = self.c
        return c._mtp_fp4 is not None and c.op.moe.decode_exact(rows)

    def moe(self, ph: Phase, o, rows: int, exact: bool = False):
        """The MTP layer's routed experts: on the trunk's NVFP4 path (`_mtp_moe_fp4`, the
        experts quantized at load) at layer slot `num_layers`, else on its bf16 pair. `exact`:
        a wide decode step's exact rows (`wide_exact`)."""
        c = self.c
        if c._mtp_fp4 is None:
            return self.routed(ph, o, rows)
        c._count("mtp_moe_fp4", True)
        if exact:
            c._count("moe_dec_exact", True)
        return c.op.moe.routed(
            ph, o, c._mtp_fp4, rows, False, None, False, False, c.num_layers, exact
        )

    def out(self, residual, reduced):
        """The MTP layer's output, `residual + reduced`: head-normalized where the engine
        drafts more than one token (`_mtp_out_normed`: it is fed back into the next draft
        step, as stock's), the residual add and the head norm as one launch (bit for bit the
        two); un-normalized at MTP1, whose `shared_head` applies the norm."""
        c = self.c
        if c._mtp_out_normed:
            return c.op.rows.add_rmsnorm(residual, reduced, c._mtp["head_norm"])
        return c.op.rows.add(residual, reduced)


class Ctx:
    """The context rows' attention (`PrefillTarget`): the per-step plans every layer shares,
    and the context attention itself, each picking its kernel -- the rope/append kernel or the
    stock op, the fp8 q-base K-B or the bf16 one, the non-absorbed MHA (fp8 or bf16, its K / V
    expansion on the MXF8 or the bf16 GEMMs) or the absorbed sparse MLA, dense-causal or on
    the index path -- from what built at load and what the step's pool layout and row
    counts allow. The Target sees the choices only as the flags these ops return."""

    # A mixed step's generation rows' q_b reads at most the last 64 bf16 q_a rows
    # (`Gemm.project_tail`), so with no indexer reading q_a the norm need not store the context
    # rows below them (`q8_rows`).
    Q_A_TAIL_ROWS = _DECODE_MAX_ROWS
    # The token count at which the context rope/append kernel's int32 element offsets wrap
    # (16 heads x 576 per token): a step at or above it takes the stock op.
    ROPE_APPEND_MAX_TOKENS = 2**31 // (16 * 576)

    @staticmethod
    def _dc_bounds_nosync(seq_lens: torch.Tensor, cached, total_q: int):
        """The dense-causal per-row KV windows, `(ks, ke)` = `(0, cached + local + 1)`, on the
        device without a host sync: `total_q` is trusted, the caller having checked the host
        lengths sum to it. The kernel module's `dense_causal_bounds` reads the device sum back
        to validate it, a full drain of the device on every context step."""
        starts = torch.cumsum(seq_lens, 0) - seq_lens
        pos = torch.arange(
            total_q, dtype=torch.int32, device=seq_lens.device
        ) - torch.repeat_interleave(starts, seq_lens, output_size=total_q)
        if cached is not None:
            pos = pos + torch.repeat_interleave(cached, seq_lens, output_size=total_q)
        ks = torch.zeros(total_q, dtype=torch.int32, device=seq_lens.device)
        ke = (pos + 1).to(torch.int32).contiguous()
        return ks, ke

    @staticmethod
    def _dc_groups_host(host_lens: torch.Tensor, g_tok: int) -> torch.Tensor:
        """The kernel module's `dense_causal_groups`, vectorized on the host: `[G, 2] = (q0,
        n_tok)`, request-major, `q0 = cu[r] + g_tok * j`, `n_tok = min(g_tok, len_r - g_tok *
        j)`; the module's version loops in Python over T / 4 groups."""
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

    class _DcGroupsUploader:
        """The per-step dense-causal group table's upload without a host sync: pinned,
        double-buffered host staging and one persistent device buffer. A pinned slot is
        rewritten only after its previous upload completed (its event, two steps old in
        practice), and every reader runs on the stream that uploaded, after the upload, so
        reusing the device buffer next step is stream-ordered. Grows, with one sync, the rare
        time a step needs more rows. A pageable `.to(device)` drains the device instead."""

        def __init__(self, dev) -> None:
            self._dev = dev
            self._cap = 0
            self._host: list = []
            self._done: list = [None, None]
            self._buf = None
            self._slot = 0

        def _grow(self, rows: int) -> None:
            torch.cuda.synchronize(self._dev)
            self._cap = max(rows, 2 * self._cap, 2048 + 64)
            self._host = [
                torch.empty((self._cap, 2), dtype=torch.int32, pin_memory=prefer_pinned())
                for _ in range(2)
            ]
            self._done = [None, None]
            self._buf = torch.empty((self._cap, 2), dtype=torch.int32, device=self._dev)

        def __call__(self, groups_host: torch.Tensor) -> torch.Tensor:
            g = int(groups_host.shape[0])
            if g > self._cap:
                self._grow(g)
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

    def __init__(self, core) -> None:
        self.c = core

    @staticmethod
    def diet_tables(step: dict) -> None:
        """CTX-HOST-DIET: the pool addressing every layer's context attention reads, as python
        values built once per step into `step` -- `diet_layers[layer] = (pool index, layer in
        pool, the pool's pointer row names the primary pool and no secondary pool)` from one
        `tolist` of each host pool table,
        and `diet_tpb` -- instead of indexing the host tensors per layer in every helper
        (`_mha_takes`, twice per layer, `_attn_mha`, `_attn_ctx`). The helpers pass the same
        values to the same kernels."""
        mapping = step["host_kv_cache_pool_mapping"].tolist()
        ptrs = step["host_kv_cache_pool_pointers"].tolist()
        primary = step["primary_pool"].data_ptr()
        step["diet_layers"] = [
            (int(m[0]), int(m[1]), ptrs[int(m[0])][0] == primary and ptrs[int(m[0])][1] == 0)
            for m in mapping
        ]
        step["diet_tpb"] = int(step["tokens_per_block"])
        step["diet_block_offsets"] = {}

    @staticmethod
    def diet_block_offsets(step: dict, pool_idx: int):
        """CTX-HOST-DIET: the context requests' block-table rows of pool `pool_idx`, sliced once
        per step and shared by every layer of that pool (a view; nothing writes it)."""
        cache = step["diet_block_offsets"]
        rows = cache.get(pool_idx)
        if rows is None:
            rows = cache[pool_idx] = step["kv_cache_block_offsets"][
                pool_idx, : step["num_contexts"]
            ]
        return rows

    def rope_step(self, step, tc: int) -> tuple | None:
        """The context rope kernel's per-step operands: the new-row prefix sum and the cached
        lengths, int32, from the engine's int64 prefix sums. None sends the step's context rows
        (none on a step without them) to the stock rope/append op. Context steps are not
        captured, so a fresh pair per step is fine.

        Built on the host from the prefix sums' host copies and uploaded in one copy: the
        conversions and the difference on the device would be four small kernels per step."""
        c = self.c
        ctx = step["num_contexts"]
        compiled = c._ctx_rope_compiled
        if (
            not tc
            or compiled is None
            or not 0 < ctx <= _ctx_rope_op.MAX_REQUESTS
            or int(step["tokens_per_block"]) != compiled.tokens_per_block
            or tc >= self.ROPE_APPEND_MAX_TOKENS
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

    def dense_causal_step(
        self, md, step, tc: int, skip_ctx: bool, req_idx, dev, rope_ok: bool = False
    ):
        """The dense-causal context attention's per-step operands, shared by every layer, or
        None when the context rows take the index path.

        Taken only when the engine skipped the indexer for the context rows: every context
        sequence is within index_topk, so the dense causal index it substituted is the whole
        window and the kernel's per-token mask reproduces it. The windows are rebuilt here
        rather than read from the metadata: the engine's `cu_seqlen_ks` are batch-flat KV
        offsets, while this kernel reads each request through its own block-table row from
        position 0, cached prefix included.

        With the non-absorbed context attention built and the step on the rope kernel
        (`rope_ok`), the trunk layers take that instead (`_attn_mha`): `mha` is its plan, and
        the absorbed kernel's operands are built only if a layer falls back to it."""
        c = self.c
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
            c.dense_causal_misses[reason] = c.dense_causal_misses.get(reason, 0) + 1
            return None
        ctx = step["num_contexts"]
        host_lens = step["seq_lens"][:ctx].to(torch.int32).cpu()
        assert int(host_lens.sum()) == tc, (host_lens.tolist(), tc)
        c.dense_causal_steps += 1
        if rope_ok and c._kvb_expand:
            return SimpleNamespace(
                mha=self._mha_plan(md, step, tc, host_lens, req_idx, dev),
                absorbed=None,
                rebuild=lambda: self._dense_causal_absorbed(md, step, tc, host_lens, req_idx, dev),
            )
        return self._dense_causal_absorbed(md, step, tc, host_lens, req_idx, dev)

    def _dense_causal_absorbed(self, md, step, tc: int, host_lens, req_idx, dev):
        """The absorbed dense-causal kernel's per-step operands (`_dense_causal_step`)."""
        c = self.c
        ctx = step["num_contexts"]
        cached = None
        if int(step["num_ctx_cached_tokens"]):
            cached = torch.diff(step["ctx_cached_token_indptr"][: ctx + 1]).to(torch.int32)
        # Both without a host sync (the kernel module's helpers read the device lengths back
        # and upload pageable memory): the host already checked the row total above.
        ks, ke = self._dc_bounds_nosync(step["seq_lens_cuda"][:ctx].to(torch.int32), cached, tc)
        if c._dc_uploader is None:
            c._dc_uploader = self._DcGroupsUploader(dev)
        groups = c._dc_uploader(self._dc_groups_host(host_lens, _fmha_pre_op.DENSE_CAUSAL_GROUP))
        return SimpleNamespace(
            ks=ks,
            ke=ke,
            groups=groups,
            req_idx=req_idx[:tc],
            block_table=md.block_table,
            # One fp8 query for the step: each layer writes it and its attention reads it
            # before the next layer writes (one stream), so the bound address holds.
            quant_q=torch.empty([tc, c.heads, c.lat_dim], dtype=torch.float8_e4m3fn, device=dev),
            # Bound on the first layer; later layers re-bind output and layer offset.
            prepared=None,
            mha=None,
        )

    def _mha_plan(self, md, step, tc: int, host_lens, req_idx, dev) -> SimpleNamespace:
        """The non-absorbed context attention's per-step operands, shared by every trunk layer:
        the K / V scratch (each layer's expansion writes it and its MHA reads it before the
        next layer's expansion, on one stream) and, when the step's requests have a cached
        prefix (KV block reuse), the K/V source-row plan.

        Request r's K/V source rows are its `c_r` cached positions followed by its `n_r` new
        rows, packed at `cu_k[r]`. The cached rows are the pool rows the absorbed dense-causal
        kernel reads for those positions, gathered per layer by `cute_dsl_mha_prefix_gather`
        from the index vectors built here on the host from the two host prefix sums and
        uploaded once (the context step is eager)."""
        c = self.c
        ctx = step["num_contexts"]
        cached = int(step["num_ctx_cached_tokens"])
        tk = tc + cached
        rows = c.heads * c.qk_dim
        kv_dt = torch.float8_e4m3fn if c._mha_fp8 else torch.bfloat16
        kvb8 = c._mha_fp8 and c._kvb8 is not None
        # The one-launch expansion writes `[K | V]` rows; K and V are their halves.
        kv = None
        if kvb8 and c._kvb8.merged is not None:
            kv = torch.empty([tk, rows + c.heads * c.v_dim], dtype=kv_dt, device=dev)
        plan = SimpleNamespace(
            max_len=int(host_lens.max()),
            tk=tk,
            k=torch.empty([tk, rows], dtype=kv_dt, device=dev) if kv is None else kv[:, :rows],
            v=(
                torch.empty([tk, c.heads * c.v_dim], dtype=kv_dt, device=dev)
                if kv is None
                else kv[:, rows:]
            ),
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
            pool_rows=step["primary_pool"].view(torch.uint8).reshape(-1, c.lat_dim),
            # The bf16 [cached | new] rows of the bf16-MMA expansion (the MXF8 one gathers
            # into `a8` instead).
            lat=None if kvb8 else torch.empty([tk, c.lat_dim], dtype=torch.bfloat16, device=dev),
        )
        return plan

    def mha_takes(self, a, layer: int, step, ctx_rope) -> bool:
        """Whether this layer's context rows can take the non-absorbed attention: its weights
        built, the step on the context rope kernel, and the layer's pool the primary pool."""
        c = self.c
        diet = step.get("diet_layers")
        if diet is not None:
            return (
                (a.kvb8 if c._mha_fp8 else a.kvb_aug) is not None
                and ctx_rope is not None
                and c._ctx_rope_compiled is not None
                and diet[layer][2]
            )
        pool_ptrs = step["host_kv_cache_pool_pointers"]
        pool_idx = int(step["host_kv_cache_pool_mapping"][layer][0])
        return (
            (a.kvb8 if c._mha_fp8 else a.kvb_aug) is not None
            and ctx_rope is not None
            and c._ctx_rope_compiled is not None
            and int(pool_ptrs[pool_idx, 0]) == step["primary_pool"].data_ptr()
            and int(pool_ptrs[pool_idx, 1]) == 0
        )

    def mha(self, a, layer: int, q, latent, out, tc: int, step, ctx_rope, mha, o8=None) -> bool:
        """Context attention in the non-absorbed MLA form (K / V expanded per head), on a
        dense-causal step; False (nothing launched) where the layer cannot take it.

        1. The rope kernel's `q_in_place` mode: rope(q_pe) back into the q_b output, which
           is then the MHA's query; rope(k_pe) into the latent rows and the latent row as
           e4m3 into the pool -- the same code as the absorbed mode, so the decode cache is
           the same.
        2. With a cached prefix, each request's K/V source rows `[prefix | new]`: the prefix
           from the fp8 pool, widened to bf16.
        3. K / V = those latent rows x the augmented kv_b halves (`[T_k, 4096]` each).
        4. Causal varlen MHA, d 256, scale 1/16, bottom-right aligned over the prefix, into
           this layer's attention output.

        In the fp8 form (`q` the e4m3 query from `Gemm.q_b_e4m3`) rope rotates the e4m3 query in
        place, K / V are e4m3 (the V expansion on the latent's c_kv part), and the MHA reads
        the layer's per-head dequantization table. Under the MXF8 expansion steps 2 and 3 are
        one gather writing the `[prefix | new]` rows as MXF8 and two MXF8 GEMMs. With `o8`
        (fp8 form, the MXF8-O handle built) the MHA writes the context rows' attention as
        o_proj's MXF8 operand into `o8` instead of `out`."""
        c = self.c
        if not self.mha_takes(a, layer, step, ctx_rope):
            return False
        fp8 = q.dtype == torch.float8_e4m3fn
        pool = step["primary_pool"]
        diet = step.get("diet_layers")
        if diet is not None:
            pool_idx, layer_in_pool, _ = diet[layer]
            tpb = step["diet_tpb"]
            block_offsets = self.diet_block_offsets(step, pool_idx)
        else:
            pool_idx, layer_in_pool = (
                int(v) for v in step["host_kv_cache_pool_mapping"][layer][:2]
            )
            tpb = int(step["tokens_per_block"])
            block_offsets = step["kv_cache_block_offsets"][pool_idx, : step["num_contexts"]]
        compiled = c._ctx_rope_compiled
        heads = c.heads
        q3 = torch.reshape(q, [tc, heads, c.qk_dim])
        in_place = latent.stride(0) in compiled.kernels_q_in_place and latent.stride(1) == 1
        lat = latent if in_place else latent.clone(memory_format=torch.contiguous_format)
        cute_dsl_mla_rope_append_prefill(
            None,
            q3[..., c.nope :],
            lat,
            None,
            pool,
            layer_in_pool,
            block_offsets,
            *ctx_rope,
            c._rope["rotary_cos_sin"],
            tpb,
            compiled,
            q_in_place=True,
            q_e4m3=fp8,
        )
        if fp8 and compiled.rope_par:
            c._count("rope_par", True)
        src, cu_k = lat, None
        if mha.a8 is not None:
            self._expand_mxf8(a, lat, mha, layer_in_pool * tpb, tpb)
            cu_k = None if mha.cached is None else mha.cached.cu_k
            c._count("ctx_kvb_mxf8", True)
            if mha.kv is not None:
                c._count("ctx_kvb_merged", True)
        elif mha.cached is not None:
            pre = mha.cached
            src = cute_dsl_mha_prefix_gather(
                pre.pool_rows, pre.idx, pre.rows, pre.req_idx, pre.block_table, lat, pre.lat,
                layer_in_pool * tpb, c._stride_factor, tpb, compiled=c._mha_prefix,
            )  # fmt: skip
            cu_k = pre.cu_k
        # The prefill GEMM serves up to PREFILL_MAX_ROWS rows; with a cached prefix the K/V
        # rows can exceed it, and the product is row-independent.
        k_rows = heads * c.qk_dim
        if mha.a8 is not None:
            halves = ()
        elif fp8:
            halves = (
                ("k", a.kvb8[0], mha.k, src),
                ("v", a.kvb8[1], mha.v, src[:, : c.kv_lora]),
            )
        else:
            halves = (
                ("k", a.kvb_aug[:k_rows], mha.k, src),
                ("v", a.kvb_aug[k_rows:], mha.v, src),
            )
        step_rows = _projection_op.PREFILL_MAX_ROWS
        for r0 in range(0, mha.tk, step_rows):
            r1 = min(mha.tk, r0 + step_rows)
            for half, w, dst, x in halves:
                cute_dsl_projection(x[r0:r1], w, dst[r0:r1], c._kvb_expand[half])
        use_o8 = o8 is not None and fp8 and c._mha_q8 is not None
        cute_dsl_mha_prefill(
            q3,
            mha.k.view(mha.tk, heads, c.qk_dim),
            mha.v.view(mha.tk, heads, c.v_dim),
            torch.reshape(out, [tc, heads, c.v_dim]),
            ctx_rope[0],
            mha.max_len,
            compiled=c._mha_q8.mha if use_o8 else c._mha_pre,
            cu_seqlens_k=cu_k,
            head_scales=a.mha_hs if fp8 else None,
            q8_out=o8 if use_o8 else None,
        )
        assert use_o8 == (o8 is not None), "an o_proj operand the MHA did not write"
        c._count("ctx_mha", True)
        if use_o8:
            c._count("ctx_mha_o8", True)
        if fp8:
            c._count("ctx_mha_fp8", True)
        if cu_k is not None:
            c._count("ctx_mha_cached", True)
        return True

    def _expand_mxf8(self, a, lat, mha, layer_offset: int, tpb: int) -> None:
        """K / V of this layer into `mha.k` / `mha.v` (e4m3) on the MXF8 expansion: one gather
        writes the `[cached prefix | new]` latent rows as MXF8 into `mha.a8` (cached rows as
        their pool bytes at 2^0, the new rows `lat` block-quantized), then the two GEMMs run
        on it in row slices of the runners' capacity (a multiple of 128 rows, so a slice's
        scales start at a 128-row block)."""
        c = self.c
        g = c._kvb8
        if mha.cached is not None:
            pre = mha.cached
            cute_dsl_mha_kv_gather_mxf8(
                pre.pool_rows, pre.idx, pre.rows, pre.req_idx, pre.block_table, lat, mha.a8,
                layer_offset, c._stride_factor, tpb, True, compiled=g.gather,
            )  # fmt: skip
        else:
            pool, idx, req_idx, block_table = g.idle
            cute_dsl_mha_kv_gather_mxf8(
                pool, idx, 0, req_idx, block_table, lat, mha.a8, 0, 1, 1, False,
                compiled=g.gather,
            )  # fmt: skip
        a_k, a_v, sfa_k, sfa_v = mha.a8
        w_k, w_v = a.kvb8
        blk_k, blk_v = _kv_gather8_op.sf_bytes_k(128), _kv_gather8_op.sf_bytes_v(128)
        if mha.kv is not None:
            # One launch: `[K | V]` rows from `a_k` (`w_k` is the stacked weight; the V tiles
            # read its first 512 columns, the bytes `a_v` holds).
            rkv = g.merged
            step = rkv.launch.max_rows
            for r0 in range(0, mha.tk, step):
                r1 = min(mha.tk, r0 + step)
                b = r0 // 128
                cute_dsl_projection_mxf8(
                    a_k[r0:r1], w_k, mha.kv[r0:r1], rkv, sfa=sfa_k[b * blk_k :]
                )
            return
        rk, rv = c._kvb_expand["k"], c._kvb_expand["v"]
        step = rk.launch.max_rows
        for r0 in range(0, mha.tk, step):
            r1 = min(mha.tk, r0 + step)
            b = r0 // 128
            cute_dsl_projection_mxf8(a_k[r0:r1], w_k, mha.k[r0:r1], rk, sfa=sfa_k[b * blk_k :])
            cute_dsl_projection_mxf8(a_v[r0:r1], w_v, mha.v[r0:r1], rv, sfa=sfa_v[b * blk_v :])

    def attn(
        self, a, layer: int, q, latent, out, slots, tc: int, step, ctx_rope, dc, o8=None,
        ctx_last=None,
    ) -> bool:  # fmt: skip
        """Sparse MLA over the context rows, absorbed form, on the sm103 prefill kernel. The
        index rows are already global pool rows spanning the cached prefix, so no gather; under
        `dc` there are none, and the kernel reads the causal window through the block table.

        The order is fixed: the rope/KV append mutates `fused_q` and the pool, so the quantize
        has to read the rotated query, and nothing after the append may fall back.

        Returns whether the context rows' attention went to `o8` (the non-absorbed fp8 MHA's
        MXF8 O epilogue) rather than `out`.

        `ctx_last` (the MTP layer, `PrefillTarget._mtp_last_plan`): the rope/KV append still
        covers every context row, then only the plan's queries attend, on the dense-causal
        kernel over the block table, and `out` holds those `n_pad` rows."""
        c = self.c
        op = c.op
        if dc is not None and dc.mha is not None:
            if self.mha(a, layer, q, latent, out, tc, step, ctx_rope, dc.mha, o8):
                return o8 is not None
            # The fp8 query exists only once `_mha_takes` said the layer takes the route.
            assert q.dtype != torch.float8_e4m3fn, "an fp8 query on the absorbed path"
            c._count("ctx_mha", False)
            # This layer takes the absorbed dense-causal kernel: its operands, built once.
            if dc.absorbed is None:
                dc.absorbed = dc.rebuild()
            dc = dc.absorbed
        heads, dev, dt = c.heads, q.device, q.dtype
        q_nope, q_pe = torch.split(
            torch.reshape(q, [tc, heads, c.qk_dim]), [c.nope, c.rope_dim], -1
        )
        fused_q = torch.empty([tc, heads, c.lat_dim], dtype=dt, device=dev)
        fq_nope, fq_pe = torch.split(fused_q, [c.kv_lora, c.rope_dim], -1)
        quant_q = (
            dc.quant_q
            if dc is not None
            else torch.empty([tc, heads, c.lat_dim], dtype=torch.float8_e4m3fn, device=dev)
        )
        pool = step["primary_pool"]
        diet = step.get("diet_layers")
        if diet is not None:
            pool_idx, layer_in_pool, primary = diet[layer]
            on_kernel = ctx_rope is not None and primary
        else:
            pool_ptrs = step["host_kv_cache_pool_pointers"]
            mapping = step["host_kv_cache_pool_mapping"][layer]
            pool_idx, layer_in_pool = int(mapping[0]), int(mapping[1])
            on_kernel = (
                ctx_rope is not None
                and int(pool_ptrs[pool_idx, 0]) == pool.data_ptr()
                and int(pool_ptrs[pool_idx, 1]) == 0
            )
        c._count("ctx_rope", on_kernel)
        # On the rope kernel's path the attention reads only the fp8 query: K-B writes its
        # e4m3 q-base bytes straight into `quant_q[..., :kv_lora]` and the rope kernel skips
        # its q-base groups, so the bf16 `fused_q[..., :kv_lora]` (129 MB at 8k rows) is
        # neither written nor re-read. Same bytes into the attention (FP32 -> BF16 -> E4M3,
        # nearest, on both paths). Row counts the decode groups serve keep the bf16 K-B.
        kb_fp8 = (
            on_kernel
            and c._kb_fp8_pre is not None
            and tc not in c._absorb["k_b"]
            and tc <= _absorbed_bmm_op.PREFILL_MAX_ROWS
        )
        if kb_fp8:
            c._count("ctx_kb_fp8", True)
            q_base = torch.split(quant_q.view(torch.uint8), [c.kv_lora, c.rope_dim], -1)[0]
            cute_dsl_absorbed_bmm(
                torch.transpose(q_nope, 0, 1),
                a.k_b_nk,
                torch.transpose(q_base, 0, 1),
                c._kb_fp8_pre,
            )
        else:
            op.gemm.absorbed_bmm(
                PREFILL,
                "k_b",
                torch.transpose(q_nope, 0, 1),
                a.k_b,
                a.k_b_nk,
                torch.transpose(fq_nope, 0, 1),
            )
        if on_kernel:
            in_place = latent.stride(0) in c._ctx_rope_compiled.kernels and latent.stride(1) == 1
            c._count("ctx_rope_latent_in_place", in_place)
            # One launch: rope(q_pe) into the tail of fused_q straight from the strided view,
            # rope(k_pe) and the latent row appended as e4m3 to this layer's pool rows, and
            # e4m3(fused_q). Context requests are the first `num_contexts` of the batch.
            cute_dsl_mla_rope_append_prefill(
                fused_q,
                q_pe,
                # In place: the latent is the projection row's tail (stride q_lora + 576),
                # which the kernel was compiled for, and nothing reads it after the append
                # rotates its k_pe. Any other stride gets a packed copy.
                latent if in_place else latent.clone(memory_format=torch.contiguous_format),
                quant_q,
                pool,
                layer_in_pool,
                (
                    self.diet_block_offsets(step, pool_idx)
                    if diet is not None
                    else step["kv_cache_block_offsets"][pool_idx, : step["num_contexts"]]
                ),
                *ctx_rope,
                c._rope["rotary_cos_sin"],
                step["tokens_per_block"],
                c._ctx_rope_compiled,
                skip_q_base=kb_fp8,
            )
        else:
            # The append rotates the query's rope tail in place, reading its own
            # pre-rotation contents, and the absorbed BMM wrote only the first kv_lora
            # columns: seed the tail.
            op.rows.copy(fq_pe, q_pe)
            mla_rope_append_paged_kv_assign_q(
                torch.reshape(fused_q, [tc, heads * c.lat_dim]),
                latent.contiguous(),
                step["num_contexts"],
                step["ctx_cached_token_indptr"],
                step["ctx_kv_indptr"],
                step["max_ctx_seq_len"],
                c._rope["rotary_cos_sin"],
                heads,
                # The width ahead of each head's rope tail in the query this call hands
                # over: the absorbed head is `kv_lora + rope`, so this is kv_lora, not
                # qk_nope_head_dim.
                c.kv_lora,
                c.rope_dim,
                c.kv_lora,
                step["kv_cache_block_offsets"],
                step["host_kv_cache_pool_pointers"],
                step["host_kv_cache_pool_mapping"],
                None,
                0,
                layer,
                step["tokens_per_block"],
                step["max_seq_len"],
                1,
                c.quant_mode,
            )
            op.rows.copy(quant_q, fused_q)
        if ctx_last is not None:
            assert on_kernel and dc is None, "the selected-rows attention needs the rope kernel"
            self.attn_selected(a, quant_q, pool, out, ctx_last)
            return False
        lat_out = torch.empty([tc, heads * c.kv_lora], dtype=dt, device=dev)
        if dc is not None:
            # The pool rows this layer's selection would name, `to_global` of the dense
            # causal index: the block-table row times the stride, plus this layer's offset.
            tpb = step["diet_tpb"] if diet is not None else int(step["tokens_per_block"])
            layer_offset = layer_in_pool * tpb
            if dc.prepared is None:
                dc.prepared = _fmha_pre_op.prepare_dense_causal(
                    c._fmha_pre_dc_compiled, quant_q, pool, lat_out, dc.ks, dc.ke,
                    c._fmha_bmm1, c._fmha_bmm2, dc.block_table, dc.req_idx,
                    c._stride_factor, layer_offset, dc.groups,
                )  # fmt: skip
            cute_dsl_sparse_mla_prefill(
                quant_q, pool, None, lat_out, dc.ks, dc.ke, c._fmha_bmm1, c._fmha_bmm2,
                compiled=dc.prepared, block_table=dc.block_table, req_idx=dc.req_idx,
                stride_factor=c._stride_factor, layer_offset=layer_offset,
                dense_causal_groups=dc.groups,
            )  # fmt: skip
        else:
            cute_dsl_sparse_mla_prefill(
                quant_q,
                pool,
                slots,
                lat_out,
                step["cu_seqlen_ks"],
                step["cu_seqlen_ke"],
                c._fmha_bmm1,
                c._fmha_bmm2,
                compiled=c._fmha_pre_compiled,
            )
        op.gemm.absorbed_bmm(
            PREFILL,
            "v_b",
            torch.transpose(torch.reshape(lat_out, [tc, heads, c.kv_lora]), 0, 1),
            a.v_b_t,
            a.v_b_nk,
            torch.transpose(torch.reshape(out, [tc, heads, c.v_dim]), 0, 1),
        )
        return False

    def attn_selected(self, a, quant_q, pool, out, last) -> None:
        """The plan's context queries (`last.ctx_rows` of the step's e4m3 query) on the
        dense-causal kernel, each its own group over its request's block-table row (positions
        0 .. cached + len - 1: the engine's dense index, so bitwise the index path's rows),
        then V-B into `out` (`last.n_pad` rows)."""
        c = self.c
        op = c.op
        heads, dev, n = c.heads, quant_q.device, last.n_pad
        tc = int(quant_q.shape[0])
        q_sel = torch.empty([n, heads, c.lat_dim], dtype=torch.float8_e4m3fn, device=dev)
        cute_dsl_rows_select(
            quant_q.view(tc, heads * c.lat_dim),
            last.ctx_rows,
            q_sel.view(n, heads * c.lat_dim),
            c._rows_sel,
        )
        lat_out = torch.empty([n, heads * c.kv_lora], dtype=out.dtype, device=dev)
        prepared = _fmha_pre_op.prepare_dense_causal(
            c._fmha_pre_dc_compiled, q_sel, pool, lat_out, last.ks, last.ke,
            c._fmha_bmm1, c._fmha_bmm2, last.block_table, last.req, last.stride,
            last.layer_offset, last.groups,
        )  # fmt: skip
        cute_dsl_sparse_mla_prefill(
            q_sel, pool, None, lat_out, last.ks, last.ke, c._fmha_bmm1, c._fmha_bmm2,
            compiled=prepared, block_table=last.block_table, req_idx=last.req,
            stride_factor=last.stride, layer_offset=last.layer_offset,
            dense_causal_groups=last.groups,
        )  # fmt: skip
        op.gemm.absorbed_bmm(
            PREFILL,
            "v_b",
            torch.transpose(torch.reshape(lat_out, [n, heads, c.kv_lora]), 0, 1),
            a.v_b_t,
            a.v_b_nk,
            torch.transpose(torch.reshape(out, [n, heads, c.v_dim]), 0, 1),
        )

    #: The row counts the MTP layer's selected rows are padded to: decode cells every op
    #: downstream has (the routed MoE takes powers of two only; anything else it would pad
    #: with torch ops). Above 64 the plan declines and every row is computed.
    MTP_LAST_CELLS = (2, 4, 8, 16, 32, 64)

    #: MTP3 / MTP5 only: 4 / 6 generation rows per request pass 64 selected rows from 17 / 11
    #: generation requests on, so the plan also pads to these (prefill kernels downstream, the
    #: routed MoE's pair at a power of two), while that is still fewer rows than the step's.
    MTP_LAST_WIDE_CELLS = (128, 256)

    def mtp_last_plan(
        self, md, step, ctx_rope, ctx: int, tc: int, gen: int, layer: int, stride: int
    ):
        """The MTP layer's selected rows on a context step, or None (every row computed: no rope
        kernel on the step's context rows, no row select, or a reason counted below).

        The draft head reads the layer's output at each context request's last row and at
        the generation rows only, and everything after the layer's KV write is row-local, so
        attention-to-output can run on those rows: the context last rows (padded with the
        last one up to a cell), then the generation rows. The context queries take the
        dense-causal kernel (`ks = 0`, `ke = cached + len`, request `b`, one `(q, 1)` group
        each), which needs the engine's dense index (no real selection) and the rope kernel's
        append. Host only, sync-free: the metadata's host prefix sums and one pinned upload.
        On B300 the MTP layer's context attention at 8.9k rows goes 590 -> 30 us."""
        c = self.c
        # Only with the rope kernel on this step's context rows and the row select built.
        if ctx_rope is None or c._rows_sel is None:
            return None
        pool_ptrs = step["host_kv_cache_pool_pointers"]
        mapping = step["host_kv_cache_pool_mapping"][layer]
        cells = self.MTP_LAST_CELLS
        if c.max_draft_len > 1:
            cells = cells + self.MTP_LAST_WIDE_CELLS
        r = ctx + gen
        r_pad = next((m for m in cells if m >= r), None)
        why = (
            "selection"
            if not bool(md.skip_indexer_for_ctx_reqs)
            else "block"
            if int(step["tokens_per_block"]) != _fmha_pre_op.DENSE_CAUSAL_BLOCK
            else "pool"
            if int(pool_ptrs[int(mapping[0]), 0]) != step["primary_pool"].data_ptr()
            or int(pool_ptrs[int(mapping[0]), 1]) != 0
            or c._ctx_rope_compiled is None
            else "rows"
            if r_pad is None or (r_pad > self.MTP_LAST_CELLS[-1] and r_pad >= tc + gen)
            else None
        )
        host_cu = step["host_ctx_uncached_token_indptr"][: ctx + 1].to(torch.int64)
        host_ci = step["host_ctx_cached_token_indptr"][: ctx + 1].to(torch.int64)
        lens = host_cu[1:] - host_cu[:-1]
        if why is None and (
            int(host_cu[0]) != 0 or int(host_cu[ctx]) != tc or bool((lens < 1).any())
        ):
            why = "lens"
        ke = host_ci[1:] - host_ci[:-1] + lens
        if why is None and int(ke.max()) > c.idx_topk:
            why = "window"
        if why is not None:
            # Not a stock-op fallback: every row then runs the same kernels. Counted apart
            # from the route counters, by reason.
            c.mtp_last_declines[why] = c.mtp_last_declines.get(why, 0) + 1
            return None
        c._count("mtp_ctx_last", True)
        if r_pad > self.MTP_LAST_CELLS[-1]:
            c._count("mtp_ctx_last_wide", True)
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

    def step_tables(self, step: dict) -> None:
        """CTX-HOST-DIET (`_ctx_host_diet`): the per-step pool addressing tables
        (`diet_tables`) and the stream this forward runs on, read once
        (`torch.cuda.current_stream()` costs ~3 us per call and the layers ask for it three
        times each). Off, every helper indexes the host tensors itself."""
        if self.c._ctx_host_diet:
            self.diet_tables(step)
            step["diet_main"] = torch.cuda.current_stream()

    def slots(self, md, req_idx, rows: int, tc: int, gen: int, gen_p: int, dc) -> "_CtxSlots":
        """A context or mixed step's selection as each layer's attention reads it
        (`_CtxSlots.layer`). Under the dense-causal context attention (`dc`) the context rows'
        pool rows are formed inside the attention kernel, so the conversion runs on the
        generation rows alone, or not at all. TRTLLM_GLM52_MIXED_BASE (default 1): with only the
        generation rows converting at the MTP width, they take the layer-independent base rows,
        converted once per fresh selection, and the sparse MLA decode adds each layer's
        offset, as on a pure decode step: the same pool rows, instead of one global conversion
        per layer (78 launches per mixed step)."""
        c = self.c
        slot_lo = tc if dc is not None else 0
        mixed_base = c._mixed_base and slot_lo == tc and gen > 0 and gen_p == _MLA_ROPE_MTP_WIDTH
        return _CtxSlots(c, md, req_idx, rows, tc, gen, slot_lo, mixed_base)

    def mha8(self, a, layer: int, step, ctx_rope, dc, tc: int) -> bool:
        """Whether layer `layer`'s context rows take the fp8 non-absorbed MHA: a dense-causal
        step on the MHA plan, the fp8 form built, and the layer able to take it (`mha_takes`).
        The context rows' query then comes from q_b's e4m3 output (`queries`)."""
        return (
            bool(tc)
            and dc is not None
            and dc.mha is not None
            and self.c._mha_fp8
            and self.mha_takes(a, layer, step, ctx_rope)
        )

    def q8_rows(self, mha8: bool, tc: int, need_q: bool) -> tuple[int, int]:
        """`(q8_rows, q_store_from)` for `Attn.qkv`: on the fp8 MHA route with the MXF8 q_b,
        the q_a norm writes the context rows' q_a as that GEMM's operand; with no indexer
        reading q_a this step, the only bf16 q_a reader is the generation rows' q_b tail, so
        the norm stores only the last `self.Q_A_TAIL_ROWS` context rows' bf16 q_a."""
        q8 = tc if mha8 and isinstance(self.c._q8, _mxf8_op.Mxf8ProjectionRunner) else 0
        store_from = tc - self.Q_A_TAIL_ROWS if q8 and not need_q and tc > self.Q_A_TAIL_ROWS else 0
        return q8, store_from

    def queries(self, mha8: bool, q, q_a, qa8, a, tc: int, gen: int):
        """`(q_ctx, q_gen)`: on the fp8 MHA route the context rows' e4m3 query from q_b
        (`Gemm.q_b_e4m3`, reading the norm's MXF8 operand `qa8`) and the generation rows' bf16
        query from the q_b tail (`Gemm.project_tail`); otherwise the two halves of `q`."""
        if mha8:
            g = self.c.op.gemm
            q_ctx = g.q_b_e4m3(q_a[:tc], a, pre=qa8)
            q_gen = g.project_tail(PREFILL, "q_b", q_a, tc, a.qb) if gen else None
            return q_ctx, q_gen
        q_ctx, q_gen = torch.split(q, [tc, gen], 0)
        return q_ctx, q_gen

    def o_operand(self, mha8: bool, trunk: bool, attn_out, a):
        """o_proj's MXF8 operand `(q8, sf)` for the fp8 MHA's epilogue to write, where o_proj
        takes the MXF8 GEMM (trunk layers only: the MTP layer's close reads the bf16
        `attn_out`), else None."""
        c = self.c
        if not (
            mha8
            and trunk
            and c._mha_q8 is not None
            and c.op.gemm.mxf8_takes(PREFILL, "o_proj", attn_out, a.o8)
        ):
            return None
        rows, dev = int(attn_out.shape[0]), attn_out.device
        return (
            torch.empty([rows, c.heads * c.v_dim], dtype=torch.uint8, device=dev),
            torch.empty(_mha_pre_op.sf_bytes(rows), dtype=torch.uint8, device=dev),
        )

    def gen_side(self, gen: int, tc: int, mha8: bool, ctx_last, o8):
        """`(stream, side_quant)`: the stream a mixed step's generation rows run their attention
        on beside the context rows', or None to run them after. On the fp8 MHA route the
        generation rows' attention (absorbed K-B, RoPE append, sparse MLA decode, V-B) reads
        only their query, latent, slots and pool pages, and writes only their attention rows
        and those pages, so it runs on its own stream beside the context rows' RoPE, K/V
        expansion and MHA (about 30 us per layer on the critical path at 7-8k context rows on
        B300; -9 us per layer net, the expansion GEMM sharing SMs with it). Joined before the
        generation rows' quantize and o_proj read them. Same kernels and inputs: the outputs
        are bit for bit the serial ones. The absorbed context path shares K-B / V-B with these
        rows and keeps the serial order. With `_gen_side_quant` the generation rows' quantize
        into `o8` also runs there (`side_quant`); when the context rows then do not take `o8`,
        it is dropped unread."""
        c = self.c
        if not (gen and tc and mha8 and ctx_last is None and c._gen_stream is not None):
            return None, False
        side_quant = o8 is not None and c._gen_side_quant
        c._count("ctx_gen_side", True)
        c._count("ctx_gen_side_quant", side_quant)
        return c._gen_stream, side_quant

    def quant_gen_rows(self, attn_gen, o8, tc: int) -> None:
        """The generation rows' bf16 attention into o_proj's MXF8 operand `o8`, after the
        context rows' (none on a pure context step: then only the padding scale rows)."""
        cute_dsl_act_quant_mxf8_rows(attn_gen, *o8, tc, compiled=self.c._mha_q8.rows)

    def rows_select(self, src, rows, out, scatter: bool = False):
        """`out = src[rows]`, or with `scatter` `out[rows] = src`: the CuTe DSL row select."""
        return cute_dsl_rows_select(src, rows, out, self.c._rows_sel, scatter=scatter)

    def mtp_out(self, residual, reduced):
        """The MTP layer's output on a context step: `residual + reduced`, then the module's
        own head norm where the engine drafts more than one token (`_mtp_out_normed`: it is
        fed back into the next draft step, as stock's); un-normalized at MTP1, whose
        `shared_head` applies the norm."""
        c = self.c
        out = c.op.rows.add(residual, reduced)
        if c._mtp_out_normed:
            return c.op.rows.rmsnorm(out, c._mtp["head_norm"])
        return out


class _CtxSlots:
    """One context or mixed step's `Ctx.slots`: `layer(buf, fresh, layer_id)` maps the
    request-local selection `buf` to `(context rows, generation rows, generation index kind,
    its layer)` for the attention of pool layer `layer_id` (`fresh`: the layer selected
    anew); the context half is None under the dense-causal context attention."""

    def __init__(self, core, md, req_idx, rows, tc, gen, slot_lo, mixed_base) -> None:
        self.c, self.md, self.req_idx = core, md, req_idx
        self.rows, self.tc, self.gen = rows, tc, gen
        self.slot_lo, self.mixed_base = slot_lo, mixed_base
        self.base = None

    def layer(self, buf, fresh: bool, layer_id: int):
        c, md, tc = self.c, self.md, self.tc
        if self.slot_lo == self.rows:
            return None, None, INDEX_KIND_GLOBAL, None
        if self.mixed_base:
            if fresh or self.base is None:
                self.base = cute_dsl_req_index_to_base(
                    self.req_idx[tc:], md.block_table, buf[tc:], md.tokens_per_block,
                    c._stride_factor, compiled=c._req_base_compiled,
                )  # fmt: skip
                c._count("mixed_base", True)
            return None, self.base, INDEX_KIND_BASE_WITH_LAYER_OFFSET, layer_id
        lo = self.slot_lo
        slots = cute_dsl_req_index_to_global(
            self.req_idx[lo:], md.block_table, buf[lo:], md.tokens_per_block,
            c._stride_factor, layer_id, compiled=c._req_global_compiled,
        )  # fmt: skip
        ctx_s, gen_s = (None, slots) if lo else torch.split(slots, [tc, self.gen], 0)
        return ctx_s, gen_s, INDEX_KIND_GLOBAL, None
