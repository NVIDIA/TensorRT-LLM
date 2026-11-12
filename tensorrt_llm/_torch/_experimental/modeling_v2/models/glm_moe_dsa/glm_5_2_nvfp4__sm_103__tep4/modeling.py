# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: glm-5.2-nvfp4 / sm_103 / tep4 -- self-contained modeling code.

GLM-5.2, modelopt NVFP4 export: 78 layers, hidden 6144, 64 query heads (16 per rank), layers
0-2 dense and 3-77 MoE (256 routed experts at top-8 plus a shared expert). Attention is MLA
with DeepSeek Sparse Attention: an indexer on 21 of the layers selects 2048 KV positions per
query row, and the layers in between reuse the most recent selection. Only the routed experts
are NVFP4; everything else is bf16 and the latent KV pool is fp8-e4m3. Layer 78 is the bf16
MTP module (`MTPLayer`), live only under a `speculative_config`.

`tep4` shards the attention heads, the dense and shared MLPs and `lm_head` four ways and the
routed experts four ways, so every layer ends in an all-reduce (`ops/comm/`).

Most kernels are the checkpoint-specialised CuTe DSL kernels in `kernels/`, reached through
the `cute_dsl_*` entries in `ops/` (projections, norms, indexer, absorbed BMMs, sparse MLA,
the all-reduce, the routed experts' router, top-k, routing, quantize and grouped GEMMs), and
around them the logits head (the `lm_head` shard GEMM and a Lamport all-gather that widens to
fp32), the spec worker's greedy picks (`spec_worker.py`), and the row ops the fused kernels leave over (hidden-
width norms and adds, row lookups, copies). A call site whose row count or layout has no
compiled kernel falls back to the stock op. Both attention
phases run the specialised sparse-MLA kernels: the context phase on the sm103 prefill kernel,
the decode phase on the decode kernel.
The context step's projections, absorbed BMMs and rope/KV append run on the prefill
counterparts of the decode kernels; on a step where the engine skipped the indexer for the
context rows, the context attention reads the causal window through the block table.
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace
from typing import NamedTuple, cast

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import step_contract_enabled
from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.flashinfer_silu_and_mul import (  # noqa: E501
    flashinfer_silu_and_mul,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.sparse.params import use_self_sampling_gvr
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.distributed.ops import allgather
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.modules.top_k import TopK, TopKImplementation
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks
from tensorrt_llm._torch.speculative import (
    MTPEagleWorker,
    get_spec_worker,
    should_use_separate_draft_kv_cache,
)
from tensorrt_llm._utils import get_sm_version, prefer_pinned
from tensorrt_llm.logger import logger

from . import weights as _weights
from .kernels.common import cuda_stream as _cuda_stream
from .kernels.moe import moe_fc1_kblock as _fc1_kblock
from .ops.activation import cute_dsl_swiglu_mxf8 as _swiglu8_op
from .ops.activation import cute_dsl_swiglu_rows as _swiglu_rows_op
from .ops.activation.cute_dsl_dense_swiglu import compile as compile_dense_swiglu
from .ops.activation.cute_dsl_dense_swiglu import cute_dsl_dense_swiglu
from .ops.activation.cute_dsl_swiglu_mxf8 import cute_dsl_swiglu_mxf8
from .ops.activation.cute_dsl_swiglu_rows import cute_dsl_swiglu_rows
from .ops.attention import cute_dsl_mha_kv_gather_mxf8 as _kv_gather8_op
from .ops.attention import cute_dsl_mha_prefill as _mha_pre_op
from .ops.attention import cute_dsl_mha_prefix_gather as _mha_prefix_op
from .ops.attention import cute_dsl_mla_rope_append_prefill as _ctx_rope_op
from .ops.attention import cute_dsl_mla_rope_generation as _mla_rope_op
from .ops.attention import cute_dsl_sparse_mla_decode as _fmha_dec_op
from .ops.attention import cute_dsl_sparse_mla_decode_union as _fmha_union_op
from .ops.attention import cute_dsl_sparse_mla_prefill as _fmha_pre_op
from .ops.attention.convert_req_index_to_global import convert_req_index_to_global
from .ops.attention.cute_dsl_fp8_paged_mqa_logits import ATOM_SPLIT as _MQA_ATOM_SPLIT
from .ops.attention.cute_dsl_fp8_paged_mqa_logits import cute_dsl_fp8_paged_mqa_logits
from .ops.attention.cute_dsl_indexer_k_cache_update import build as build_ix_k_cache
from .ops.attention.cute_dsl_indexer_k_cache_update import cute_dsl_indexer_k_cache_update
from .ops.attention.cute_dsl_indexer_q_side_fused import build as build_ix_q_side
from .ops.attention.cute_dsl_indexer_q_side_fused import cute_dsl_indexer_q_side_fused
from .ops.attention.cute_dsl_indexer_q_side_k_cache import build as build_ix_qk
from .ops.attention.cute_dsl_indexer_q_side_k_cache import cute_dsl_indexer_q_side_k_cache
from .ops.attention.cute_dsl_indexer_topk_decode import build as build_ix_topk_decode
from .ops.attention.cute_dsl_indexer_topk_decode import cute_dsl_indexer_topk_decode
from .ops.attention.cute_dsl_indexer_topk_decode import uses_cluster as ix_topk_decode_uses_cluster
from .ops.attention.cute_dsl_indexer_topk_prefill import build as build_ix_topk_prefill
from .ops.attention.cute_dsl_indexer_topk_prefill import cute_dsl_indexer_topk_prefill
from .ops.attention.cute_dsl_mha_kv_gather_mxf8 import cute_dsl_mha_kv_gather_mxf8
from .ops.attention.cute_dsl_mha_prefill import cute_dsl_mha_prefill
from .ops.attention.cute_dsl_mha_prefix_gather import cute_dsl_mha_prefix_gather
from .ops.attention.cute_dsl_mla_rope_append_prefill import cute_dsl_mla_rope_append_prefill
from .ops.attention.cute_dsl_mla_rope_generation import cute_dsl_mla_rope_generation
from .ops.attention.cute_dsl_req_index_to_base import build as build_req_index_to_base
from .ops.attention.cute_dsl_req_index_to_base import cute_dsl_req_index_to_base
from .ops.attention.cute_dsl_req_index_to_global import build as build_req_index_to_global
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
from .ops.comm import cute_dsl_allgather_widen as _allgather_op
from .ops.comm._lamport import TWOSHOT_ABOVE as _AR_TWOSHOT_ABOVE
from .ops.comm._lamport import build_workspace as build_lamport_workspace
from .ops.comm.cute_dsl_allgather_widen import cute_dsl_allgather_widen
from .ops.comm.cute_dsl_allreduce import cute_dsl_allreduce
from .ops.comm.cute_dsl_allreduce_add_rmsnorm import cute_dsl_allreduce_add_rmsnorm
from .ops.comm.cute_dsl_allreduce_add_rmsnorm_quant_nvfp4 import (
    cute_dsl_allreduce_add_rmsnorm_quant_nvfp4,
)
from .ops.elementwise import cute_dsl_copy as _copy_op
from .ops.elementwise import cute_dsl_hidden_add as _hidden_add_op
from .ops.elementwise import cute_dsl_rows_select as _rows_sel_op
from .ops.elementwise.cute_dsl_copy import cute_dsl_copy
from .ops.elementwise.cute_dsl_hidden_add import cute_dsl_hidden_add
from .ops.elementwise.cute_dsl_rows_select import cute_dsl_rows_select
from .ops.embedding import cute_dsl_hidden_row_gather as _row_gather_op
from .ops.embedding.cute_dsl_embedding import build as build_embedding
from .ops.embedding.cute_dsl_embedding import cute_dsl_embedding
from .ops.embedding.cute_dsl_hidden_row_gather import cute_dsl_hidden_row_gather
from .ops.gemm import cute_dsl_absorbed_bmm as _absorbed_bmm_op
from .ops.gemm import cute_dsl_dense_up_swiglu_mxf8 as _dense_up8_op
from .ops.gemm import cute_dsl_indexer_pre_proj_split as _ix_pre_split_op
from .ops.gemm import cute_dsl_mtp_eh_proj as _mtp_eh
from .ops.gemm import cute_dsl_projection as _projection_op
from .ops.gemm import cute_dsl_projection_mxf8 as _mxf8_op
from .ops.gemm import cute_dsl_projection_mxf8_decode as _dec8_op
from .ops.gemm import cute_dsl_q_b_ix_wq_b as _qbix_op
from .ops.gemm import cute_dsl_qkva_ix_fold as _ix_fold_op
from .ops.gemm import cute_dsl_shared_gate_up_swiglu as _shared_gu_swiglu_op
from .ops.gemm import cute_dsl_shared_up_swiglu_mxf8 as _shared_up8_op
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
from .ops.moe import cute_dsl_moe_l2_prefetch as _l2pf_op
from .ops.moe import cute_dsl_moe_merge as _moe_merge_op
from .ops.moe import cute_dsl_moe_nvfp4_experts as _moe_experts_op
from .ops.moe import cute_dsl_moe_prefill_topk_routing_rep as _moe_topk_pre_rep_op
from .ops.moe import cute_dsl_moe_router as _moe_router_op
from .ops.moe import cute_dsl_moe_routing as _moe_routing_op
from .ops.moe import cute_dsl_moe_sigmoid_topk as _topk_rows_op
from .ops.moe import cute_dsl_moe_topk_routing as _moe_topk_op
from .ops.moe import cute_dsl_moe_topk_routing_rep as _moe_topk_rep_op
from .ops.moe import cute_dsl_moe_topk_routing_wide_rep as _moe_topk_wide_rep_op
from .ops.moe import cute_dsl_moe_zero as _moe_zero_op
from .ops.moe import cute_dsl_mtp_moe_experts as _mtp_experts_op
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
from .ops.norm.cute_dsl_mtp_norm_shard import compile as compile_mtp_norm_shard
from .ops.norm.cute_dsl_mtp_norm_shard import cute_dsl_mtp_norm_shard
from .ops.norm.cute_dsl_qkv_a_rmsnorm import compile as compile_qkv_a_rmsnorm
from .ops.norm.cute_dsl_qkv_a_rmsnorm import cute_dsl_qkv_a_rmsnorm
from .ops.norm.cute_dsl_qkv_a_rmsnorm_q8 import cute_dsl_qkv_a_rmsnorm_q8
from .ops.quantization import cute_dsl_act_quant_mxf8_rows as _aq_rows_op
from .ops.quantization import cute_dsl_moe_quantize_nvfp4 as _moe_quant_op
from .ops.quantization.cute_dsl_act_quant_mxf8_rows import cute_dsl_act_quant_mxf8_rows
from .ops.quantization.cute_dsl_moe_quantize_nvfp4 import cute_dsl_moe_quantize_nvfp4
from .ops.sampling import cute_dsl_argmax as _argmax_op
from .spec_worker import Glm52MTPEagleWorker, Glm52Tep4DSAMetadata


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


_QUANT_MODE_FP8_KV = 128
_KV_RESIDUAL_DIM = 0
_MOE_MAX_T = 8192
# How often rank 0 logs the route counters, in decode forwards (eager forwards and graph
# captures: a replay does not re-enter Python). Context steps always report.
_ROUTE_REPORT_EVERY = 16
# The MTP width the specialised MLA rope kernel and the base-row index kind are built for at
# MTP1; MTP3 / MTP5 add max_draft_len + 1 (`mtp_widths`).
_MLA_ROPE_MTP_WIDTH = 2
# The decode kernels' widest compiled row count. Above it a projection with no compiled runner
# takes the prefill GEMM; at or below it, cuBLAS.
_DECODE_MAX_ROWS = 64
#: `slots_of`'s generation index kind on a pure decode step whose attention takes the
#: union kernel: the slots are the step's request-local selection (`_UnionSlots`).
_UNION_POSITIONS = "union_positions"


class _UnionSlots(NamedTuple):
    """A pure decode step's generation selection as the union attention takes it: the
    request-local top-k positions `[rows, 2048]` and each row's block-table row."""

    positions: torch.Tensor
    req_idx: torch.Tensor


# The fp8 context attention's dequantization scales are powers of two from strict bounds on
# its operands (`_mha_fp8_scales`); the bound's margin covers bf16 rounding of the normalized
# inputs and the e4m3 cached-prefix rows (each within 1/16 of its bf16 value).
_MHA_FP8_MARGIN = 1.1
# Extra headroom of the Q bound when q_b runs in MXF8: both of its operands are e4m3, each
# element within 1/16 of its bf16 value, so the Cauchy-Schwarz bound grows by (17/16)^2.
_MHA_FP8_Q_MXF8 = (17.0 / 16.0) ** 2
# The same headroom on the K / V bounds when the expansion GEMMs run in MXF8: the margin
# covered e4m3 latent rows (1/16 per element); the e4m3 weight adds that factor once more.
_MHA_FP8_KV_MXF8 = 17.0 / 16.0
# A mixed step's generation rows' q_b reads at most the last 64 bf16 q_a rows (`_project_tail`),
# so with no indexer reading q_a the norm need not store the context rows below them.
_Q_A_TAIL_ROWS = _DECODE_MAX_ROWS
# The embedding kernel's widest row count. A step above it (a `max_num_tokens` past 8192)
# gathers in slices.
_EMBED_MAX_ROWS = 8192
# Token counts at which a kernel's int32 element offsets wrap: the context rope/append kernel
# (16 heads x 576 per token), so a step at or above it takes the stock op; and the indexer
# q-side kernel (32 heads x 128 per token), which has no stock counterpart here.
_ROPE_APPEND_MAX_TOKENS = 2**31 // (16 * 576)
_STEP_MAX_TOKENS = 2**31 // (32 * 128)
# The sm103 prefill kernel's two scale arrays. The context path has no op that writes them,
# so they are the values the decode path's `mla_rope_generation` was measured to write:
# `[1] == [0] * log2(e)`, a natural-log softmax scale of 1/16. Both are functions of
# q_scaling and the MLA dims only, so they are constants for this checkpoint. The query is
# cast to e4m3 unscaled: `[0]` is the kernel's internal dequant factor, not the quantizer's.
_FMHA_BMM1_SCALE = (0.0625, 0.09016843885183334)
_FMHA_BMM2_SCALE = (1.0,)
# The indexer's LayerNorm epsilon, which is not the model's rms_norm_eps.
_INDEXER_K_NORM_EPS = 1e-6
# `head_dim + 4`: the fused write needs the fp8 key and its fp32 scale in one slot.
_INDEXER_PER_TOKEN_MIN = 132
# The context indexer projection's fp32 output row: 128 key columns, then 32 head weights.
_IX_PROJ_COLS = 160
# fp8_mqa_logits allocates a `[q_rows, kv_rows]` fp32 score matrix per call; prefill chunks are
# tiled to keep it under this many elements. Context steps are not captured.
_PREFILL_LOGITS_BUDGET = 1 << 24
# The fused shared-expert gate|up + SwiGLU kernel is compiled for one row count per runner
# (`_shared_gu_swiglu_op`): an MTP3 / MTP5 engine also builds it at its decode row counts of
# 1..`_SHARED_FUSED_UP_SEQS` sequences without a cell (MTP5 6 / 12 / 18, MTP3 12), served
# where `_cell_up` holds. Wider counts take the two-launch path on the projection cells.
_SHARED_FUSED_UP_SEQS = 3
# Projections an MTP3 / MTP5 captured decode step's rows without a cell keep on cuBLAS instead
# of the next cell up (`_cell_up`): there the cell is the slower one. Cold-weight CUDA-graph
# timing on B300 (cuBLAS vs next cell up, us per call): ix_qb 12 rows 4.95 vs 6.11, 18 rows
# 5.02 vs 5.42 (MTP 128 cell); dense_down 6 / 12 / 18 rows 8.15 / 8.25 / 8.29 vs 9.16 / 9.16
# / 9.15. An eager draft step (`_draft_eager`) keeps its cells.
_CELL_UP_KEEPS_CUBLAS = frozenset({"ix_qb", "dense_down"})
# The decode logits width (the engine's max_seq_len) at or below which the generation rows keep our
# decode top-k although the engine's config asks for GVR (`_dec_topk_gvr`;
# `TRTLLM_GLM52_DEC_TOPK_GVR_ABOVE`). Single-B300 CUDA graph of 20 calls (us per call, ours /
# GVR, every set exact): GVR's tuning is frozen from the width, so on a 16384-wide engine at 192
# rows (MTP5 x 32) it costs 5.2 / 12.5 at KV 2k / 4k against our 1.5 / 5.5, where a realtext
# step (ISL 1k + OSL 8k) spends most of its decode; at 172032 GVR takes 8-17 (42 at 192 rows)
# against our 58-95 at KV 160k. Only the width and the row count are fixed at capture.
_DEC_TOPK_GVR_ABOVE = 16384


def _tile_bounds(start: int, end: int, kv_rows: int) -> list[tuple[int, int]]:
    rows = max(1, _PREFILL_LOGITS_BUDGET // max(kv_rows, 1))
    return [(a, min(a + rows, end)) for a in range(start, end, rows)]


# MTP draft-loop index share: the engine's DSA indexer semantics (`Indexer.sparse_attn_indexer`
# with `mtp_index_share`), for the MTP layer's indexer, the only one a draft step runs.
#
# The engine's MTP-Eagle linear draft loop marks the metadata `in_mtp_draft_loop`, hands it the
# trunk's accepted counts (`mtp_num_accepted`) and sets `indexer_skip_topk` from the second draft
# step on (`eagle3._forward_linear_draft_loop`); it does so for every DSA metadata, and each
# indexer's own `mtp_index_share` decides whether to act. With it on, draft step 0 stashes one
# selection row per request in `shared_topk_indices` (a context request's last row, a generation
# request's last accepted row), and the later steps copy those rows instead of scoring and
# selecting. After step 0 the loop rewrites the metadata in place to one row per request in
# request order (`num_contexts = 0`), so stash row `r` is request `r`'s: contexts first, then
# generations. The keys still project and append on every step: the engine's
# `pre_indexer_proj` and `_update_k_cache` run unconditionally.


def _engine_sparse_params(model_config: ModelConfig, layer_idx: int | None):
    """Layer `layer_idx`'s DSA sparse params as the engine's attention module builds them
    (`to_sparse_params(pretrained_config, layer_idx)`), or None without a sparse config."""
    sparse_cfg = getattr(model_config, "sparse_attention_config", None)
    if sparse_cfg is None or not hasattr(sparse_cfg, "to_sparse_params"):
        return None
    return sparse_cfg.to_sparse_params(
        pretrained_config=getattr(model_config, "pretrained_config", None), layer_idx=layer_idx
    )


def _engine_mtp_index_share(model_config: ModelConfig) -> bool:
    """The engine's `mtp_index_share` for the MTP layer's indexer: its sparse params, i.e. the
    LLM arg `index_share_for_mtp_iteration`, else the checkpoint's field of that name (GLM-5.2
    NVFP4's config.json sets it true)."""
    cfg = getattr(model_config, "pretrained_config", None)
    params = _engine_sparse_params(model_config, getattr(cfg, "num_hidden_layers", None))
    return bool(getattr(params, "mtp_index_share", False))


def _dec_topk_gvr_serves(max_seq_len: int) -> bool:
    """Whether the GVR decode top-k serves an engine of this logits width (`max_seq_len`, fixed
    at capture): wider than `_DEC_TOPK_GVR_ABOVE` (`TRTLLM_GLM52_DEC_TOPK_GVR_ABOVE`)."""
    above = int(os.environ.get("TRTLLM_GLM52_DEC_TOPK_GVR_ABOVE", str(_DEC_TOPK_GVR_ABOVE)))
    return max_seq_len > above


def _dec_topk_gvr_choice(
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
_SHARED_INTER = 512
# The widest exact decode step (`_moe_decode_exact`) whose pre-MoE collective also writes the
# routed experts' NVFP4 input (`_moe_prequant`): the quant entry's widest certified cell. MTP3
# / MTP5 at batch 32 step 128 / 192 rows.
_MOE_PREQUANT_WIDE_MAX = 256
# The widest wide decode step that takes the expert replicas (`_build_moe_wide_rep`): the wide
# top-k's 65..128 range. 4-GPU decode chain with the real all-reduce on real routing (two trunk
# MoE layers with replicas): -6.5 / -5.8 us per step at 128 rows (MTP3), -8.1 / -6.4 / -4.2 us at
# 68 / 96 / 126; at 192 rows (MTP5) -2.7 / +2.8 / +0.4 us over 320 / 800 / 1600 replays a side,
# no gain past the replica top-k's +0.7 us, so 129..256 keep the wide top-k.
_WIDE_REP_MAX_ROWS = 128


def _pow2(rows: int) -> int:
    return 1 << max(0, (rows - 1).bit_length())


def _direct_ok(t: torch.Tensor, dtype: torch.dtype) -> bool:
    """Read in place rather than staged. A staging copy is a memcpy node, which a PDL edge
    cannot pass through, so it is skipped for a 128-byte aligned contiguous tensor -- the
    collective's output, whose graph-pool address is stable."""
    return t.dtype is dtype and t.is_contiguous() and t.data_ptr() % 128 == 0


def _pow2_ceil(x: torch.Tensor) -> torch.Tensor:
    return torch.exp2(torch.ceil(torch.log2(x.float().clamp_min(2.0**-100))))


def _fold_pow2(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """`w [H, rows, K] / s[h]` per head, as a new contiguous tensor in `w`'s dtype. `s` holds
    powers of two, so the division is exact; checked, since an underflow would not be."""
    s = s.float().to(w.device).view(-1, 1, 1)
    out = (w.float() / s).to(w.dtype).contiguous()
    if not torch.equal(out.float() * s, w.float()):
        raise ValueError("fp8 context attention: a power-of-two weight scale was not exact")
    return out


def _try_build(what: str, build):
    """`build()`, or None with a warning: the caller's call sites then keep the stock op."""
    try:
        return build()
    except Exception as e:  # noqa: BLE001 -- that op stays on the stock path
        logger.warning(f"modeling_v2 GLM-5.2: {what} build failed ({e}); using the stock op")
        return None


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


class GlmMoeDsaTep4ModelingV2(DecoderModel):
    # Off unless `__init__` reads them on (`TRTLLM_GLM52_MOE_DEC_UNPAD*`): a core built without
    # `__init__` (the MoE unit tests) keeps the padded decode path.
    _moe_dec_unpad = False
    _moe_dec_unpad_signal = False
    _moe_topk_signaled = False

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        cfg = model_config.pretrained_config
        mapping = model_config.mapping
        self.tp_rank = mapping.tp_rank
        self.tp_size = mapping.tp_size
        self.ep_rank = mapping.moe_ep_rank
        self.ep_size = mapping.moe_ep_size
        self.quant_mode = _QUANT_MODE_FP8_KV
        # TRTLLM_GLM52_DENSE_TOPK_ON_DEVICE (default 1): when every context sequence is shorter
        # than the index top-k, the engine skips the indexer and builds a dense causal table of
        # `num_ctx_tokens x index_topk` entries. Its host build (a CPU kernel plus a copy) costs
        # ~97 ms of `_prepare_inputs` per 8k-token context step on B300 with the GPU idle; the
        # metadata's `dense_topk_on_device` builds the same integer table on the GPU.
        self._dense_topk_on_device = os.environ.get("TRTLLM_GLM52_DENSE_TOPK_ON_DEVICE", "1") == "1"
        self._mixed_base = os.environ.get("TRTLLM_GLM52_MIXED_BASE", "1") == "1"
        # TRTLLM_GLM52_CTX_HOST_DIET (default 1; 0 restores the per-call paths): host time
        # only, same kernels and arguments. A context step reads its per-layer pool addressing
        # from python tables built once per step (`PrefillTarget._diet_tables`); the launch
        # wrappers read raw stream handles (`kernels/common/cuda_stream.py`); eager calls of the
        # decode-cell GEMMs a mixed step's generation rows take (q_b, K-B, V-B) launch TVM-FFI
        # twins of their kernels; the context rope/append and the generation sparse MLA launch a
        # call signature their `prepare_runner` passed before without its contract walk.
        # Read once, at import, by `kernels/common/cuda_stream.py`: the launch wrappers and
        # this target must agree on it.
        self._ctx_host_diet = _cuda_stream.DIET
        # TRTLLM_GLM52_FMHA_UNION (default 1): MTP3 / MTP5 pure decode steps take the
        # union decode attention (`cute_dsl_sparse_mla_decode_union`); 0 keeps the per-row one.
        self._union_on = os.environ.get("TRTLLM_GLM52_FMHA_UNION", "1") == "1"

        self.num_layers = cfg.num_hidden_layers
        self.eps = cfg.rms_norm_eps
        self.heads = cfg.num_attention_heads // mapping.tp_size
        self.nope = cfg.qk_nope_head_dim
        self.rope_dim = cfg.qk_rope_head_dim
        self.v_dim = cfg.v_head_dim
        self.kv_lora = cfg.kv_lora_rank
        self.q_lora = cfg.q_lora_rank
        self.qk_dim = self.nope + self.rope_dim
        self.lat_dim = self.kv_lora + self.rope_dim
        rope_cfg = getattr(cfg, "rope_scaling", None) or getattr(cfg, "rope_parameters", None)
        theta = (rope_cfg or {}).get("rope_theta", getattr(cfg, "rope_theta", None))
        self.theta = float(theta)

        self.idx_heads = cfg.index_n_heads
        self.idx_dim = cfg.index_head_dim
        self.idx_topk = cfg.index_topk
        # The indexer's whole softmax scale: the scorers take none, so the q side folds it into
        # `weights` together with the q quantization scale.
        self.idx_weight_scale = self.idx_dim**-0.5 * self.idx_heads**-0.5
        # The checkpoint's layout, read as stock's model_config reads it (GLM-5.2: interleaved).
        # The two layouts score alike up to ~32k tokens; past that the wrong one selects the
        # wrong keys, which a short-prompt accuracy check cannot see.
        self.indexer_is_neox = not getattr(cfg, "indexer_rope_interleave", False)

        self.dense_layers = cfg.first_k_dense_replace
        self.num_experts = cfg.n_routed_experts
        self.topk = cfg.num_experts_per_tok
        self.n_group = cfg.n_group
        self.topk_group = cfg.topk_group
        self.routed_scale = float(cfg.routed_scaling_factor)
        self.moe_inter = cfg.moe_intermediate_size
        self.local_experts = self.num_experts // mapping.moe_ep_size
        self.expert_offset = self.local_experts * self.ep_rank
        # Decode expert replicas (`weights.DEC_HOT`): every rank also holds each trunk MoE
        # layer's `dec_replicas` hottest experts in slots 64.. of its routed stacks; 0 = off.
        self.dec_replicas = _weights.dec_replicas()
        self.dec_hot = (
            {i: _weights.dec_hot(i, self.dec_replicas) for i in _weights.DEC_HOT}
            if self.dec_replicas
            else {}
        )
        # Context expert replicas (`TRTLLM_GLM52_CTX_EXPERT_REPLICAS=R`, 0 = off): a context
        # step's top-k splits the rows of each trunk MoE layer's first R replica experts
        # across the EP ranks by producer CTA, levelling the ranks' grouped-GEMM loads (the
        # slowest rank sets the all-reduce wait). Runs on the decode replicas' slots, so R is
        # at most `dec_replicas`.
        # Default: the decode replica count (TRTLLM_GLM52_DEC_EXPERT_REPLICAS=0 turns both off).
        self.ctx_replicas = int(
            os.environ.get("TRTLLM_GLM52_CTX_EXPERT_REPLICAS", str(self.dec_replicas))
        )
        if not 0 <= self.ctx_replicas <= self.dec_replicas:
            raise ValueError(
                "TRTLLM_GLM52_CTX_EXPERT_REPLICAS must be in 0..TRTLLM_GLM52_DEC_EXPERT_REPLICAS "
                f"({self.dec_replicas}), got {self.ctx_replicas}"
            )
        # Wide decode expert replicas (`TRTLLM_GLM52_WIDE_DEC_EXPERT_REPLICAS`: "1", the
        # default, or "0"): an MTP3 / MTP5 decode step of 65..256 rows moves each trunk MoE
        # layer's active hot experts whole to the least loaded EP ranks, on the context
        # replicas' slots, prefill pair and hot ids (`_build_moe_wide_rep`; needs
        # `ctx_replicas` > 0).
        self.wide_dec_replicas = os.environ.get("TRTLLM_GLM52_WIDE_DEC_EXPERT_REPLICAS", "1") == "1"
        # The wide decode top-k's schedule (`TRTLLM_GLM52_WIDE_TOPK_CLUSTER16`: "1", the
        # default, or "0"): the MTP3 / MTP5 decode steps' fused top-k (and its replica
        # variant) on a 16-CTA cluster, half the token-warps per SM, with the trimmed winner
        # lookup; "0" keeps the portable 8-CTA kernel as it was. Same ids, weights, counts
        # and tile layout either way (route `moe_topk_wide_cl16` counts the 16-CTA launches).
        cl16 = os.environ.get("TRTLLM_GLM52_WIDE_TOPK_CLUSTER16", "1") == "1"
        self.wide_topk_schedule = (16, True) if cl16 else (8, False)
        self.mtp_enabled = getattr(model_config, "spec_config", None) is not None
        # The routed experts' FC1 weight layout (`TRTLLM_GLM52_MOE_FC1_KBLK`, `moe_fc1_kblock`),
        # read once here: the loader (`weights._fc1_w`), the MTP layer's quantized experts and
        # every compiled FC1 kernel follow this one value.
        self.fc1_kblk = _fc1_kblock.enabled()
        # The same for FC2 (`TRTLLM_GLM52_MOE_FC2_KBLK`): the loader (`weights._fc2_w`), the
        # MTP layer's quantized FC2 and every compiled FC2 kernel follow this one value.
        self.fc2_kblk = _fc1_kblock.fc2_enabled()
        # Draft tokens per step (MTP-Eagle linear chain: 1, 3 or 5, see routing.py); 0 without
        # speculation. A generation step carries max_draft_len + 1 query rows per sequence.
        self.max_draft_len = (
            int(getattr(model_config.spec_config, "max_draft_len", None) or 1)
            if self.mtp_enabled
            else 0
        )
        # The generation widths (query rows per sequence) the width-specialised decode builds
        # serve: MTP1's 2, plus max_draft_len + 1 at MTP3 / MTP5 (verify and draft step 0).
        # Draft steps 1..D-1 carry 1 row per sequence.
        self.mtp_widths = (_MLA_ROPE_MTP_WIDTH,) + (
            (self.max_draft_len + 1,) if self.max_draft_len > 1 else ()
        )
        # Whether the MTP layer returns its head-normalized output. The engine's linear draft
        # loop feeds the MTP layer's return value back in as the next draft step's hidden state,
        # and stock's MTP layer returns `shared_head.norm(...)`: an un-normalized feed-back
        # collapsed MTP3 / MTP5 acceptance at draft positions 2.. (3.27 vs stock 3.98 at MTP3).
        # MTP1 never feeds back, so it keeps the norm inside `shared_head` (unchanged bytes).
        self._mtp_out_normed = self.max_draft_len > 1
        # The engine's draft-loop index share for the MTP layer's indexer (`_engine_mtp_index_share`;
        # the GLM-5.2 checkpoint turns it on): draft steps after the first reuse draft step 0's
        # selection (`_mtp_share_*`). Only a loop of more than one draft step reads the stash.
        self.mtp_index_share = self.mtp_enabled and _engine_mtp_index_share(model_config)
        self._mtp_share_stash = self.mtp_index_share and self.max_draft_len > 1
        # With TRTLLM_GLM52_DEC_TOPK_GVR=1 and the engine's sparse config asking for the GVR
        # decode top-k (`enable_heuristic_topk`, `_dec_topk_gvr_choice`), the generation rows'
        # top-k runs the stock engine's self-sampling GVR engine through its `TopK` module, as
        # the stock DSA indexer does; otherwise (None) ours, the cluster split on long engines.
        # `_derive_first_step` keeps ours on engines no wider than `_DEC_TOPK_GVR_ABOVE`
        # (`_dec_topk_gvr_serves`) and warms GVR's launchers.
        gvr_det = _dec_topk_gvr_choice(
            _engine_sparse_params(model_config, 0),
            self.idx_topk,
            sm_version=get_sm_version(),
            dsl_available=IS_CUTLASS_DSL_AVAILABLE,
            env=os.environ.get("TRTLLM_GLM52_DEC_TOPK_GVR"),
        )
        # Whether it serves this engine's width (`_derive_first_step`).
        self._dec_topk_gvr_live = False
        self._dec_topk_gvr = (
            None
            if gvr_det is None
            else TopK(
                self.idx_topk,
                decode_implementation=TopKImplementation.CUTE_DSL_GVR,
                gvr_self_sampling=True,
                gvr_deterministic=gvr_det,
            )
        )
        # Whether the running step is a decode step of an engine drafting more than one token
        # (`_select_target` sets it): its rows above 64 (verify and draft step 0 carry 32 x
        # (max_draft_len + 1)) are decode rows, which keep the numerics of the 64-row decode
        # paths (bf16 on the MTP decode cells; MXF8 where the decode path is the decode MXF8
        # GEMM, the trunk o_proj) rather than take the context paths a row count above 64
        # otherwise means.
        self._wide_decode = False
        # Such a step's rows without an MTP cell of their own (a batch of fewer than 32
        # sequences: 65..191 rows, 96 / 144 at 24 requests) run on the next cell up at their
        # exact row count (`_wide_cell`); False sends them to the bf16 fallbacks instead.
        # `_wide_rag` holds the re-pinned runners, by (call, rows); built in the forward,
        # nothing in them compiles.
        self._wide_bucket = True
        self._wide_rag: dict[tuple[str, int], object] = {}
        # Whether the running call is an eager draft step past the first (`DecodeTarget.
        # mtp_forward` outside a CUDA-graph capture) of an engine drafting more than one token:
        # one row per sequence of a context or mixed step, any count 1..64. Such a count
        # without a decode cell of its own runs on the next cell up at its exact rows (the
        # cells' runtime-M launch, as `_wide_cell` above 64) instead of cuBLAS, whose kernel
        # for a new shape loads its module inside the step: the load holds the host until the
        # GPU drains (19-25 ms), and the step loses the host's lead. `_draft_rag` holds the
        # re-pinned runners, by (call, rows).
        self._draft_eager = False
        self._draft_rag: dict[tuple[str, int], object] = {}
        # TRTLLM_GLM52_DEC_CELL_UP (default 1; 0 restores cuBLAS there): an MTP3 / MTP5
        # decode step's (`_wide_decode`, graph-captured) row counts up to 64 without a decode
        # cell of their own (MTP5 at 1-3 requests: 6 / 12 / 18; MTP3 at 3: 12) take the same
        # rule as an eager draft step (`_cell_up`): the next cell up at their exact rows.
        # Measured at MTP5 b=2 (12 rows): the indexer projection there ran as `x.float()`
        # + an fp32 SIMT cuBLAS GEMM + split-K reduce + copies (27 us per indexer layer).
        self._dec_cell_up = os.environ.get("TRTLLM_GLM52_DEC_CELL_UP", "1") == "1"
        # TRTLLM_GLM52_MOE_ACC_PADDED (default 1; 0 restores the add after FC2): at a decode
        # width up to 64 that routing pads to a power of two (MTP5 at 1-3 requests: 6 / 12 /
        # 18 -> 8 / 16 / 32) the shared expert also writes the routed slot's first rows and
        # FC2 accumulates onto them (`_moe_begin_layer`), as at the power-of-two widths: no
        # clear, and no separate shared + routed add on the critical chain after FC2.
        self._moe_acc_padded = os.environ.get("TRTLLM_GLM52_MOE_ACC_PADDED", "1") == "1"
        # TRTLLM_GLM52_MOE_DEC_UNPAD (default 1; 0 restores the padding): a trunk MoE layer at a
        # decode width up to 64 that is not a power of two (MTP5 at 1-3 requests: 6 / 12 / 18)
        # runs its grouped GEMMs on the fused top-k's own maps at the exact row count
        # (`_moe_dec_unpadded`), as the power-of-two widths do. Padding it cost seven launches
        # per layer between the top-k and FC1 (two id / weight fills and cats, two staging
        # copies, the routing kernel: 8.3 us of the critical chain at 12 rows on B300)
        # and dropped the decode replicas' maps for the routing kernel's home mapping.
        self._moe_dec_unpad = os.environ.get("TRTLLM_GLM52_MOE_DEC_UNPAD", "1") == "1"
        # TRTLLM_GLM52_MOE_DEC_UNPAD_SIGNAL (default 1): on an MTP3 / MTP5 engine the unpadded
        # trunk widths that are multiples of `max_draft_len + 1` and not powers of two also get
        # a top-k handle that releases FC1 early (`k.topk_sig`), as the power-of-two widths'
        # do: FC1 follows the top-k directly there. The MTP layer keeps the plain handles.
        self._moe_dec_unpad_signal = os.environ.get("TRTLLM_GLM52_MOE_DEC_UNPAD_SIGNAL", "1") == "1"
        # TRTLLM_GLM52_MOE_L2_PREFETCH (default 1; 0 skips building the decode L2 prefetch).
        self._moe_l2_prefetch = os.environ.get("TRTLLM_GLM52_MOE_L2_PREFETCH", "1") == "1"
        # Set by `_moe_topk` when a trunk layer selected on an early-release handle: that
        # layer's grouped GEMMs must then follow the top-k directly (`_moe_experts`).
        self._moe_topk_signaled = False
        # The fused shared-expert cells at those row counts (`_SHARED_FUSED_UP_SEQS`).
        self._shared_gu_swiglu_up: dict = {}
        self._ix_pre_split_rag = None
        # The MTP3 / MTP5 decode cells' K-B writing the e4m3 q-base beside the rope kernel
        # (`_attn_gen`), by row count; empty keeps the bf16 K-B and the full rope there.
        self._kb8_mtp: dict = {}
        # TRTLLM_GLM52_WIDE_IX_FOLD (default 1): on an MTP3 / MTP5 engine an indexer layer's
        # wide decode step (rows above 64) runs the qkva projection and the indexer's key /
        # head-weight projection as one GEMM over the two weights stacked in one buffer
        # (`_qkva_ix_fold`); 0 keeps the two GEMMs. Set up in `post_load_weights`: the
        # stacked weights by layer and the fold cells by row count; empty keeps the two.
        self._ix_fold_on = os.environ.get("TRTLLM_GLM52_WIDE_IX_FOLD", "1") == "1"
        self._ix_fold_w: dict[int, torch.Tensor] = {}
        self._qkva_ix_mtp: dict = {}
        # TRTLLM_GLM52_WIDE_QB_IXQB (default 1): likewise q_b and the indexer's wq_b (both read
        # q_a) as one GEMM over their two weights stacked `[2, 4096, 2048]` (`_qb_ixqb`), where
        # the step runs the indexer; 0 keeps the two GEMMs. Stacked weights by layer, cells by
        # row count, both set up in `post_load_weights`.
        self._qbix_on = os.environ.get("TRTLLM_GLM52_WIDE_QB_IXQB", "1") == "1"
        self._qbix_w: dict[int, torch.Tensor] = {}
        self._qbix_mtp: dict = {}
        # TRTLLM_GLM52_WIDE_IX_QK (default 1): there, too, the indexer's q side (rope + e4m3
        # pack of its q, the head weights) and this step's K-cache update run as one launch
        # (`cute_dsl_indexer_q_side_k_cache`: each kernel's CTAs in one grid, the same bytes);
        # 0 keeps the two launches. Both pool layouts' launchers, built with the K update's.
        self._ix_qk_on = os.environ.get("TRTLLM_GLM52_WIDE_IX_QK", "1") == "1"
        self._ix_qk_compiled: dict = {}

        self.w = _weights.MODEL_WEIGHTS.declare(_weights.MODEL_WEIGHTS.dims(self))

        self._derive_pending = True
        self._derived_for = None
        self._rope: dict | None = None
        self._rope_positions = 0
        self._idx_cos_sin: torch.Tensor | None = None
        self._idx_k_checked = False
        self._pool_layer_id: list[int] = []
        self._stride_factor = 0
        self._max_ctx_len = 0
        self._mtp: dict | None = None
        self._ctx_rope_compiled: _ctx_rope_op.CompiledMlaRopeAppendPrefill | None = None
        # The non-absorbed context attention (`PrefillTarget._attn_mha`): its MHA, its
        # cached-prefix gather and the two kv_b expansion runners; empty or None keeps every
        # context step on the absorbed path.
        self._mha_pre: _mha_pre_op.CompiledMhaPrefill | None = None
        self._mha_prefix: _mha_prefix_op.CompiledMhaPrefixGather | None = None
        self._kvb_expand: dict = {}
        # The fp8 form of that attention (`_build_context_mha_fp8`) and its q_b runner.
        self._mha_fp8 = False
        self._q8 = None
        # qkva / o_proj in MXF8 for the context rows (`_build_proj_mxf8`).
        self._proj_mxf8: dict = {}
        # MTP3 / MTP5: the trunk o_proj's wide decode MXF8 GEMM and the V-B writing its
        # operand (`_build_dec_mxf8`); empty keeps those rows on the context MXF8 GEMM.
        self._dec8w: dict = {}
        self._vb8w: dict = {}
        self._fmha_dec_compiled: _fmha_dec_op.CompiledSparseMlaDecode | None = None
        # MTP3 / MTP5 only: the decode attention's grid for max_draft_len + 1 rows per sequence.
        self._fmha_dec_wide: _fmha_dec_op.CompiledSparseMlaDecode | None = None
        # MTP3 / MTP5 pure decode: the union decode attention's handle (None: per-row only).
        self._fmha_union: _fmha_union_op.CompiledSparseMlaDecodeUnion | None = None
        # The routed experts' operand slots: two of each buffer, alternating per layer (trunk)
        # or per call (MTP). See `_build_moe`.
        self._moe_slot = 0
        self._mtp_slot = 0
        # Context steps on the dense-causal context attention, and the others by reason.
        self.dense_causal_steps = 0
        self.dense_causal_misses: dict[str, int] = {}
        # The dense-causal group table's sync-free uploader, built at first use.
        self._dc_uploader: _DcGroupsUploader | None = None
        # The row ops outside the fused kernels and the logits head's local GEMM, built in
        # `post_load_weights` and `build_logits_head`; None keeps a call site's stock op.
        self._rows = SimpleNamespace(
            norm=None,
            add=None,
            add_norm=None,
            gather=None,
            topk=None,
            swiglu=None,
            copy=None,
            argmax=None,
        )
        self._lm_head = None
        self._lm_head_runners: dict = {}
        # Per-route counters, by the name a smoke asserts on: `route_hits[name]` counts the
        # calls that ran the CuTe DSL route, `route_misses[name]` those its call site sent to
        # the stock op. Python-side, so they count eager forwards and each CUDA graph's
        # capture, not its replays. Created here, before the weights load, and never
        # replaced: the load-time builds above only fill in what they point at.
        self.route_hits: dict[str, int] = {}
        # Context steps on which the MTP layer computed every row, by reason
        # (`PrefillTarget._mtp_last_plan`).
        self.mtp_last_declines: dict[str, int] = {}
        self.route_misses: dict[str, int] = {}
        self._forwards = 0

    def _rope_tables(self, device, positions: int) -> dict:
        """Two layouts of one table, built in fp64 and rounded once.

        `rotary_cos_sin` is the duplicated interleaved layout the MLA ops read. The indexer
        table is `[positions, rope]` with cos in the first half and sin in the second.
        `positions` is a row count: every row depends only on its own index, so a longer table
        is the same table with more rows.
        """
        half = self.rope_dim // 2
        inv = 1.0 / (self.theta ** (torch.arange(half, dtype=torch.float64) * 2.0 / self.rope_dim))
        ang = torch.arange(positions, dtype=torch.float64)[:, None] * inv[None, :]
        cos, sin = ang.cos(), ang.sin()
        table = torch.empty(positions, self.rope_dim, 2, dtype=torch.float64)
        table[:, :half, 0] = cos
        table[:, half:, 0] = cos
        table[:, :half, 1] = sin
        table[:, half:, 1] = sin
        self._idx_cos_sin = torch.cat([cos, sin], dim=1).float().to(device).contiguous()
        return {
            "rotary_cos_sin": table.reshape(1, positions * self.rope_dim * 2)
            .float()
            .to(device)
            .contiguous(),
            "rotary_inv_freq": inv.float().to(device).contiguous(),
            "rope_dim": self.rope_dim,
            "rope_base": self.theta,
        }

    def _derive_attention(self, i: int) -> SimpleNamespace:
        """One layer's attention operands: column-major GEMM views and the two absorption
        operands of the regrouped kv_b_proj, plus each in the `[H, N, K]` layout the absorbed
        BMM kernels take (K-B's is a copy, made here)."""
        w = self.w
        kvb = w[f"l{i}_kvb"]
        hn = self.heads * self.nope
        k_b = kvb[:hn].reshape(self.heads, self.nope, self.kv_lora)
        v_b_nk = kvb[hn:].reshape(self.heads, self.v_dim, self.kv_lora)
        return SimpleNamespace(
            norm1=w[f"l{i}_norm1"],
            qkva=w[f"l{i}_qkva"].t(),
            q_norm=w[f"l{i}_q_norm"],
            kv_norm=w[f"l{i}_kv_norm"],
            qb=w[f"l{i}_qb"].t(),
            k_b=k_b,
            v_b_t=torch.transpose(v_b_nk, 1, 2),
            k_b_nk=torch.transpose(k_b, 1, 2).contiguous(),
            v_b_nk=v_b_nk,
            o=w[f"l{i}_o"].t(),
            norm2=w[f"l{i}_norm2"],
            # The non-absorbed context attention's weight, set by `_build_context_mha`.
            kvb_aug=None,
            # Its fp8 form, also set there: the K / V expansion weights with the per-head
            # scales folded in (`[4096, 576]`, `[4096, 512]`), q_b with the Q scale folded in
            # (MXF8 or bf16), and the MHA's per-head factor table.
            kvb8=None,
            qb8=None,
            mha_hs=None,
            # qkva / o_proj in MXF8 for the context rows, set in `post_load_weights`.
            qkva8=None,
            o8=None,
            # Decode rows' q_b / o_proj on MXF8 weights (`_build_dec_mxf8`), and the power-of-two
            # Q scale folded into `qb8`.
            qb_dec8=None,
            o_dec8=None,
            mha_sq=None,
        )

    def _kv_b_augmented(self, i: int) -> torch.Tensor:
        """Layer `i`'s kv_b as the non-absorbed context attention's expansion weight,
        `[heads * (qk + v), kv_lora + rope]` = `[8192, 576]` bf16: the K rows of head h are
        `[W_UK_h | 0 ; 0 | I_64]` (the nope part, then the roped k_pe copied exactly by the
        identity block), the V rows `[W_UV_h | 0]`. A new tensor; the absorbed path's views
        of kv_b are untouched."""
        kvb = self.w[f"l{i}_kvb"].detach()
        h, nope, rope, v, lora = self.heads, self.nope, self.rope_dim, self.v_dim, self.kv_lora
        w = torch.zeros(h * (nope + rope + v), lora + rope, dtype=kvb.dtype, device=kvb.device)
        wk = w[: h * (nope + rope)].view(h, nope + rope, lora + rope)
        wk[:, :nope, :lora] = kvb[: h * nope].view(h, nope, lora)
        wk[:, nope:, lora:] = torch.eye(rope, dtype=kvb.dtype, device=kvb.device)
        wv = w[h * (nope + rope) :].view(h, v, lora + rope)
        wv[:, :, :lora] = kvb[h * nope :].view(h, v, lora)
        return w

    def _build_proj_mxf8(self) -> None:
        """The context rows' qkva and o_proj as MXF8 GEMMs: e4m3 operands with e8m0 1x32
        block scales, fp32 accumulation, bf16 out. Each trunk layer's two weights are
        quantized once here (3.3 GB per rank beside the bf16 copies, which the decode cells
        keep); `_project` quantizes the activation with one launch per call. On B300 at
        T ~ 7-8k quant + GEMM takes 0.61-0.76x the bf16 prefill GEMM. Decode cells (rows
        <= 64) and the MTP layer stay bf16; an entry that fails to build keeps the bf16
        prefill GEMM."""
        self._proj_mxf8: dict[str, _mxf8_op.Mxf8ProjectionRunner] = {}
        for name, attr in (("qkva", "qkva"), ("o_proj", "o")):
            wq = [
                _mxf8_op.quantize_weight(getattr(self._attn[i], attr).t())
                for i in range(self.num_layers)
            ]
            runner = _try_build(f"MXF8 {name}", lambda name=name, w=wq[0]: _mxf8_op.build(name, w))
            if runner is None:
                continue
            for i, w in enumerate(wq):
                setattr(self._attn[i], f"{attr}8", w)
            self._proj_mxf8[name] = runner
        self._build_mlp_mxf8()

    def _build_mlp_mxf8(self) -> None:
        """The context rows' MLPs on MXF8 GEMMs, their weights quantized once here.

        Dense layers: gate|up, the SwiGLU writing the down GEMM's operand
        (`cute_dsl_swiglu_mxf8`), down (+0.17 GB per rank). At 7-8k rows 0.82 against 1.61 ms
        per context step in bf16. MoE layers: the shared expert's gate|up with the SwiGLU in
        the GEMM's epilogue (`cute_dsl_shared_up_swiglu_mxf8`, +0.47 GB per rank), on the
        operand the pre-MoE all-reduce writes; its down stays bf16 (the MXF8 down moved the
        output distribution more than the accuracy budget allows). Either part that fails to
        build keeps its bf16 path."""
        dev = torch.cuda.current_device()
        self._dense8: list | None = None
        dense = range(self.dense_layers)
        wq = {
            name: [_mxf8_op.quantize_weight(self.w[f"l{i}_mlp_{key}"]) for i in dense]
            for name, key in (("dense_gate_up", "gu"), ("dense_down", "dn"))
        }
        runners = {
            name: _try_build(f"MXF8 {name}", lambda name=name: _mxf8_op.build(name, wq[name][0]))
            for name in wq
        }
        swiglu = _try_build("SwiGLU into MXF8", lambda: _swiglu8_op.compile(dev))
        if swiglu is not None and all(runners.values()):
            self._proj_mxf8.update(runners)
            self._swiglu8 = swiglu
            self._dense8 = list(zip(wq["dense_gate_up"], wq["dense_down"]))
            # TRTLLM_GLM52_CTX_DENSE_SWIGLU_EPI (default 1): gate|up with the SwiGLU into the
            # down GEMM's MXF8 operand in its epilogue (`cute_dsl_dense_up_swiglu_mxf8`, its own
            # interleaved weight, +0.11 GB per rank): one launch instead of two and no bf16
            # [T, 6144].
            dense_sw = (
                _try_build("dense gate|up + SwiGLU MXF8", lambda: _dense_up8_op.compile(dev))
                if os.environ.get("TRTLLM_GLM52_CTX_DENSE_SWIGLU_EPI", "1") == "1"
                else None
            )
            if dense_sw is not None:
                self._dense_sw8 = dense_sw
                self._dense8 = [
                    (gu8, dn8, _dense_up8_op.prepare_weight(self.w[f"l{i}_mlp_gu"]))
                    for i, (gu8, dn8) in zip(dense, self._dense8)
                ]
        self._shared8 = _try_build("shared gate|up MXF8", lambda: _shared_up8_op.compile(dev))
        self._shared_w8 = [
            _shared_up8_op.prepare_weight(self.w[f"l{i}_mlp_gu"])
            if self._shared8 is not None and i >= self.dense_layers
            else None
            for i in range(self.num_layers)
        ]
        # The MTP module's replicated eh_proj on context rows: its two input norms write the
        # MXF8 operand, then the MXF8 GEMM (draft-only numerics: they move MTP acceptance).
        self._mtp_eh8 = None
        if self.mtp_enabled:
            w8 = _mxf8_op.quantize_weight(self._mtp["eh"].t())
            runner = _try_build("MXF8 mtp_eh", lambda: _mxf8_op.build("mtp_eh", w8))
            norm = _try_build("MTP input norms into MXF8", lambda: _eh8_op.compile(dev))
            if runner is not None and norm is not None:
                self._proj_mxf8["mtp_eh"] = runner
                self._mtp_eh8 = SimpleNamespace(norm=norm, w8=w8)

    def _build_dec_mxf8(self, dev) -> None:
        """Decode rows (1..64) of the trunk layers' q_b and o_proj on MXF8 weights the context
        path already keeps: o_proj on `o8`; q_b on the fp8 attention's `qb8` (`q_b / s_q`)
        with `log2(s_q)` added back to its block scales -- exact, s_q being a power of two,
        and checked against a direct quantization (a mismatch keeps the direct copy). At
        decode sizes these GEMMs read their weight and little else, and e4m3 halves it.

        The producers then write the operands into the launchers' own buffers, so on a pure
        decode step neither GEMM has an activation quant launch in front of it: the q_a
        norm's decode kernel (`cute_dsl_qkv_a_rmsnorm_q8`) for q_b, and the V-B GEMM's side
        output (`build_v_b_mxf8_side`, the o_proj operand's addresses compiled in) for
        o_proj. The MTP layer and a mixed step's generation rows keep the bf16 cells.

        MTP3 / MTP5 (`max_draft_len > 1`): the trunk o_proj's decode rows above 64 (up to
        192) on the wide decode MXF8 GEMM (`_dec8w`, `_dec8_op.build_wide`), its operand
        written by the MTP group's V-B side output (`_vb8w`). Same e4m3 operands and K order
        as the context MXF8 GEMM those rows took before (`_mxf8_takes`): bitwise its output,
        without the act-quant launch, the weight streamed before the PDL wait.
        TRTLLM_GLM52_WIDE_OPROJ_DEC8=0 keeps those rows on the context MXF8 GEMM."""
        self._dec8: dict = {}
        self._vb8: dict = {}
        self._dec8w: dict = {}
        self._vb8w: dict = {}
        trunk = range(self.num_layers)
        if "o_proj" in self._proj_mxf8:
            runner = _try_build(
                "decode MXF8 o_proj", lambda: _dec8_op.build("o_proj", self._attn[0].o8)
            )
            if runner is not None:
                self._dec8["o_proj"] = runner
                for i in trunk:
                    self._attn[i].o_dec8 = self._attn[i].o8
        qb8 = [self._attn[i].qb8 for i in trunk]
        if all(isinstance(w, _mxf8_op.QuantizedWeight) for w in qb8):
            weights = []
            for i, w in enumerate(qb8):
                shifted = _dec8_op.shift_scales(w, round(math.log2(self._attn[i].mha_sq)))
                direct = _mxf8_op.quantize_weight(self._attn[i].qb.t())
                same = torch.equal(direct.q, shifted.q) and torch.equal(direct.sf, shifted.sf)
                weights.append(shifted if same else direct)
            runner = _try_build("decode MXF8 q_b", lambda: _dec8_op.build("q_b", weights[0]))
            if runner is not None:
                self._dec8["q_b"] = runner
                for i, w in enumerate(weights):
                    self._attn[i].qb_dec8 = w
        if "o_proj" in self._dec8:
            for group in _absorbed_bmm_op.GROUPS:
                vb8 = _try_build(
                    f"V-B with o_proj's MXF8 operand {group}",
                    lambda group=group: _absorbed_bmm_op.build_v_b_mxf8_side(
                        group, dev, self._dec8["o_proj"].operand
                    ),
                )
                if vb8 is None:
                    self._vb8 = {}
                    break
                self._vb8.update(dict.fromkeys(vb8.rows, vb8))
        if (
            "o_proj" in self._dec8
            and self.max_draft_len > 1
            and os.environ.get("TRTLLM_GLM52_WIDE_OPROJ_DEC8", "1") == "1"
        ):
            runner = _try_build(
                "wide decode MXF8 o_proj",
                lambda: _dec8_op.build_wide("o_proj", self._attn[0].o8),
            )
            if runner is not None:
                self._dec8w["o_proj"] = runner
                vb8 = _try_build(
                    "MTP V-B with the wide o_proj's MXF8 operand",
                    lambda: _absorbed_bmm_op.build_v_b_mxf8_side(
                        _absorbed_bmm_op.MTP_GROUPS[0], dev, runner.operand
                    ),
                )
                if vb8 is not None:
                    self._vb8w = dict.fromkeys(vb8.rows, vb8)

    def _build_context_mha(self, dev) -> None:
        """The non-absorbed context attention of dense-causal steps: the causal varlen MHA,
        its cached-prefix gather, and the kv_b expansion on the prefill GEMM. The fp8 form
        (`_build_context_mha_fp8`) when it builds, else the bf16 form: the expansion on the
        `kv_b_k` / `kv_b_v` entries against one augmented weight per trunk layer
        (`a.kvb_aug`, 9.4 MB each). The MTP layer keeps the absorbed path. If any part fails
        to build, every context step keeps the absorbed path."""
        self._mha_fp8 = False
        # The fp8 form's MXF8 K / V expansion (gather + GEMM runners) and its MHA handle that
        # writes O as o_proj's MXF8 operand; None where they did not build.
        self._kvb8 = None
        self._mha_q8 = None
        self._mha_prefix = _try_build(
            "context MHA prefix gather", lambda: _mha_prefix_op.compile(dev)
        )
        if self._mha_prefix is None:
            return
        if self._build_context_mha_fp8(dev):
            return
        self._mha_pre = _try_build("context MHA", lambda: _mha_pre_op.compile(dev))
        if self._mha_pre is None:
            return
        aug = [self._kv_b_augmented(i) for i in range(self.num_layers)]
        k_rows = self.heads * self.qk_dim
        expand = {
            half: _try_build(
                f"kv_b expansion {half.upper()}",
                lambda half=half: _projection_op.build_prefill(
                    f"kv_b_{half}", aug[0][:k_rows] if half == "k" else aug[0][k_rows:]
                ),
            )
            for half in ("k", "v")
        }
        if not all(expand.values()):
            return
        for i, w in enumerate(aug):
            self._attn[i].kvb_aug = w
        self._kvb_expand = expand

    def _build_context_mha_fp8(self, dev) -> bool:
        """The fp8 context attention: q from q_b's e4m3 output, K / V from e4m3 expansion
        GEMMs, and the fp8 MHA (e4m3 QK^T and PV, fp32 accumulation, bf16 out). The
        dequantization scales are powers of two from strict bounds (`_mha_fp8_scales`),
        folded into route-only weight copies (exact) and read by the MHA as a per-head
        table, so no context row can leave the e4m3 range. The V expansion reads only the
        latent row's c_kv part (`[4096, 512]` = W_UV; the augmented V rows' rope block is
        zero). q_b runs in MXF8 when the MXF8 projections built (the bf16 copy of the
        folded weight is then not kept), else on the prefill GEMM's e4m3 entry. Against
        the bf16 form at a ~7.3k-row step on B300: -13..-15 % per layer for q_b, rope,
        expansion and MHA; attention output 4.3e-2 .. 5.9e-2 relative RMS against fp32
        (absorbed fp8 chain 6.3e-2 .. 8.1e-2). False (nothing kept) if a part fails.

        With the MXF8 projections built, two more parts when they build: the K / V expansion
        in MXF8 -- one launch per layer writes the `[cached | new]` latent rows as MXF8
        (`cute_dsl_mha_kv_gather_mxf8`: pool bytes at 2^0 for a cached row, the new rows
        block-quantized) and both expansion GEMMs run e4m3 x e4m3 with 1x32 block scales
        against the folded weights quantized once (their bf16 copies not kept; the K / V
        bounds take `_MHA_FP8_KV_MXF8` more headroom) -- as ONE launch (`kv_b_kv8`, the two
        folded weights stacked, V zero-padded to 576; the MHA handles then read K / V as the
        two halves of its `[T_k, 8192]` rows, `kv_strided`) when that entry and the strided
        MHA handle build, with the two launches' output bytes; and the MHA handle whose epilogue
        writes O as o_proj's MXF8 operand (`compile(q8=True)`), which o_proj then reads
        without its own quant launch (measured in a served run, MHA + quant + o_proj
        -11.9 .. -13.5 us per layer at a ~7.3k-row step)."""
        mha = _try_build("context MHA fp8", lambda: _mha_pre_op.compile(dev, fp8=True))
        if mha is None:
            return False
        mxf8 = "qkva" in self._proj_mxf8
        q_extra = _MHA_FP8_Q_MXF8 if mxf8 else 1.0
        h, nope, rope, v, lora = self.heads, self.nope, self.rope_dim, self.v_dim, self.kv_lora
        kvb8 = self._build_kvb8(dev) if mxf8 else None
        if kvb8 is not None and kvb8.merged is not None:
            mha_s = _try_build(
                "context MHA fp8 on strided K / V",
                lambda: _mha_pre_op.compile(dev, fp8=True, kv_strided=True),
            )
            if mha_s is None:
                kvb8.merged = None
            else:
                mha = mha_s
        merged = kvb8 is not None and kvb8.merged is not None
        kv_extra = _MHA_FP8_KV_MXF8 if kvb8 is not None else 1.0
        ranges: list[tuple[float, ...]] = []
        layers = []
        for i in range(self.num_layers):
            a = self._attn[i]
            s_q, s_k, s_v = self._mha_fp8_scales(i, q_extra, kv_extra)
            kvb = self.w[f"l{i}_kvb"].detach()
            w_k = torch.zeros(h, nope + rope, lora + rope, dtype=kvb.dtype, device=kvb.device)
            w_k[:, :nope, :lora] = kvb[: h * nope].view(h, nope, lora)
            w_k[:, nope:, lora:] = torch.eye(rope, dtype=kvb.dtype, device=kvb.device)
            w_v = kvb[h * nope :].view(h, v, lora)
            w_k8 = _fold_pow2(w_k, s_k).view(h * (nope + rope), lora + rope)
            w_v8 = _fold_pow2(w_v, s_v).view(h * v, lora)
            if merged:
                # K rows over V rows zero-padded to 576: rows quantize independently, so the
                # K rows' bytes are the K weight's and the V rows' first 512 columns the V's.
                w_kv = torch.zeros(
                    h * (nope + rope + v), lora + rope, dtype=kvb.dtype, device=kvb.device
                )
                w_kv[: h * (nope + rope)] = w_k8
                w_kv[h * (nope + rope) :, :lora] = w_v8
                w_k8, w_v8 = _mxf8_op.quantize_weight(w_kv), None
                del w_kv
            elif kvb8 is not None:
                w_k8, w_v8 = _mxf8_op.quantize_weight(w_k8), _mxf8_op.quantize_weight(w_v8)
            qb8 = _fold_pow2(a.qb.t().view(h, nope + rope, -1), torch.full((h,), s_q)).view(
                h * (nope + rope), -1
            )
            layers.append((w_k8, w_v8, _mxf8_op.quantize_weight(qb8) if mxf8 else qb8))
            hs = _mha_pre_op.fp8_head_scales(s_q, s_k, s_v).to(kvb.device)
            ranges.append(
                (s_q, float(s_k.min()), float(s_k.max()), float(s_v.min()), float(s_v.max()))
            )
            self._attn[i].mha_hs = hs
            self._attn[i].mha_sq = s_q
        w_k8, w_v8, qb8 = layers[0]
        if kvb8 is not None:
            expand = kvb8.runners
        else:
            expand = {
                "k": _try_build(
                    "kv_b expansion K e4m3",
                    lambda: _projection_op.build_prefill("kv_b_k_e4m3", w_k8),
                ),
                "v": _try_build(
                    "kv_b expansion V e4m3",
                    lambda: _projection_op.build_prefill("kv_b_v512_e4m3", w_v8),
                ),
            }
        q8 = _try_build(
            "q_b e4m3",
            lambda: (
                _mxf8_op.build("q_b_e4m3", qb8)
                if mxf8
                else _projection_op.build_prefill("q_b_e4m3", qb8)
            ),
        )
        if not all(expand.values()) or q8 is None:
            for i in range(self.num_layers):
                self._attn[i].mha_hs = None
            return False
        for i, (w_k8, w_v8, qb8) in enumerate(layers):
            self._attn[i].kvb8 = (w_k8, w_v8)
            self._attn[i].qb8 = qb8
        self._mha_pre, self._kvb_expand, self._q8 = mha, expand, q8
        self._kvb8 = kvb8
        self._mha_fp8 = True
        if "o_proj" in self._proj_mxf8:
            mha_q8 = _try_build(
                "context MHA fp8 with the MXF8 O epilogue",
                lambda: _mha_pre_op.compile(dev, fp8=True, q8=True, kv_strided=merged),
            )
            rows_q8 = _try_build("act quant rows", lambda: _aq_rows_op.compile(dev))
            if mha_q8 is not None and rows_q8 is not None:
                self._mha_q8 = SimpleNamespace(mha=mha_q8, rows=rows_q8)

        def lg(x: float) -> int:
            return int(round(math.log2(x)))

        logger.info(
            "modeling_v2 GLM-5.2: context MHA fp8, q_b "
            f"{'MXF8' if mxf8 else 'bf16'}, K / V expansion "
            f"{('MXF8, one launch' if merged else 'MXF8') if kvb8 is not None else 'bf16 MMA'}, "
            "o_proj operand "
            f"{'from the MHA' if self._mha_q8 is not None else 'quantized'}; log2 scales over "
            f"layers q [{min(lg(r[0]) for r in ranges)}, {max(lg(r[0]) for r in ranges)}] "
            f"k [{min(lg(r[1]) for r in ranges)}, {max(lg(r[2]) for r in ranges)}] "
            f"v [{min(lg(r[3]) for r in ranges)}, {max(lg(r[4]) for r in ranges)}]"
        )
        return True

    def _build_kvb8(self, dev) -> SimpleNamespace | None:
        """The MXF8 K / V expansion's gather and its two GEMM runners (built against zero
        weights: the runners take each layer's weight per call), or None if a part fails."""
        gather = _try_build("MXF8 K/V gather", lambda: _kv_gather8_op.compile(dev))
        if gather is None:
            return None
        h, qk, v, lat, lora = self.heads, self.qk_dim, self.v_dim, self.lat_dim, self.kv_lora
        bf = dict(dtype=torch.bfloat16, device=torch.device("cuda", dev))
        runners = {
            "k": _try_build(
                "MXF8 K expansion",
                lambda: _mxf8_op.build(
                    "kv_b_k8", _mxf8_op.quantize_weight(torch.zeros(h * qk, lat, **bf))
                ),
            ),
            "v": _try_build(
                "MXF8 V expansion",
                lambda: _mxf8_op.build(
                    "kv_b_v8", _mxf8_op.quantize_weight(torch.zeros(h * v, lora, **bf))
                ),
            ),
        }
        if not all(runners.values()):
            return None
        # Both in one launch (`kv_b_kv8`); None keeps the two launches.
        merged = _try_build(
            "MXF8 K | V expansion in one launch",
            lambda: _mxf8_op.build(
                "kv_b_kv8", _mxf8_op.quantize_weight(torch.zeros(h * (qk + v), lat, **bf))
            ),
        )
        d = torch.device("cuda", dev)
        # The gather's pool / index / table operands on a step without a cached prefix: read
        # by nothing (no cached rows, new rows in order), but they must be valid tensors.
        idle = (
            torch.zeros(1, lat, dtype=torch.uint8, device=d),
            torch.zeros(1, dtype=torch.int64, device=d),
            torch.zeros(1, dtype=torch.int32, device=d),
            torch.zeros(1, 1, dtype=torch.int32, device=d),
        )
        return SimpleNamespace(gather=gather, runners=runners, merged=merged, idle=idle)

    def _mha_fp8_scales(
        self, i: int, q_extra: float, kv_extra: float = 1.0
    ) -> tuple[float, torch.Tensor, torch.Tensor]:
        """Layer `i`'s power-of-two dequantization scales `(s_q, s_k [H], s_v [H])` of the fp8
        context attention, from strict bounds so no context row can exceed e4m3's 448: every
        operand is a linear map of an RMS-normalized vector `n * w` with `|n|_2 <= sqrt(dim)`,
        so by Cauchy-Schwarz `|(W (n * w))_d| <= |W_d * w|_2 sqrt(dim)`. Q = q_b(q_a norm) per
        row (nope, and for rope the norm of each rotated pair); K = W_UK c_kv (nope) and the
        roped k_pe of kv_a(norm1(h)); V = W_UV c_kv. `s = 2^ceil(log2(bound * margin / 448))`,
        per tensor for Q, per head for K and V."""
        a, f32 = self._attn[i], torch.float32
        h, nope, rope, v = self.heads, self.nope, self.rope_dim, self.v_dim
        q_lora, kv_lora = self.q_lora, self.kv_lora
        wqb = a.qb.t().detach().to(f32).view(h, nope + rope, q_lora)
        qb = (wqb * a.q_norm.detach().to(f32)).norm(dim=-1) * math.sqrt(q_lora)
        q_pair = torch.sqrt(qb[:, nope::2] ** 2 + qb[:, nope + 1 :: 2] ** 2)
        q_bound = torch.maximum(qb[:, :nope].amax(), q_pair.amax())
        kv = self.w[f"l{i}_kvb"].detach().to(f32)
        wkvn = a.kv_norm.detach().to(f32)
        k_nope = (kv[: h * nope].view(h, nope, kv_lora) * wkvn).norm(dim=-1)
        v_all = (kv[h * nope :].view(h, v, kv_lora) * wkvn).norm(dim=-1)
        pe_rows = a.qkva.t().detach().to(f32)[q_lora + kv_lora : q_lora + kv_lora + rope]
        pe = (pe_rows * a.norm1.detach().to(f32)).norm(dim=-1) * math.sqrt(pe_rows.shape[1])
        pe_pair = torch.sqrt(pe[0::2] ** 2 + pe[1::2] ** 2).amax()
        k_bound = torch.maximum(k_nope.amax(-1) * math.sqrt(kv_lora), pe_pair)
        v_bound = v_all.amax(-1) * math.sqrt(kv_lora)
        s_q = float(_pow2_ceil(q_bound * _MHA_FP8_MARGIN * q_extra / 448.0))
        return (
            s_q,
            _pow2_ceil(k_bound * _MHA_FP8_MARGIN * kv_extra / 448.0),
            _pow2_ceil(v_bound * _MHA_FP8_MARGIN * kv_extra / 448.0),
        )

    def _derive_indexer(self, i: int) -> SimpleNamespace | None:
        if f"l{i}_ix_qb" not in self.w:
            return None
        w = self.w
        return SimpleNamespace(
            fused=w[f"l{i}_ix_fused"].t(),
            fused_bf16=w[f"l{i}_ix_fused_bf16"],
            qb=w[f"l{i}_ix_qb"].t(),
            kn_w=w[f"l{i}_ix_kn_w"],
            kn_b=w[f"l{i}_ix_kn_b"],
        )

    def _stack_ix_fold_weights(self, layers) -> dict[int, torch.Tensor]:
        """For every layer with an indexer: its qkva weight and the indexer's bf16 key /
        head-weight rows (`ix_fused_bf16`) moved into one `[2848, 6144]` buffer
        (`_ix_fold_op.stack_weight`: qkva, 64 zero rows, the indexer rows), the two
        parameters re-pointed at their rows of it -- the same bytes in the same row-major
        layout, so every other reader is unchanged, and the old storage is freed. Returns the
        buffers by layer (`_qkva_ix_fold`'s weight)."""
        stacked = {}
        for i in layers:
            if f"l{i}_ix_fused_bf16" not in self.w:
                continue
            qkva, ix = self.w[f"l{i}_qkva"], self.w[f"l{i}_ix_fused_bf16"]
            buf = _ix_fold_op.stack_weight(qkva.detach(), ix.detach())
            qkva.data = buf[: int(qkva.shape[0])]
            ix.data = buf[_ix_fold_op.IX_ROW0 :]
            stacked[i] = buf
        return stacked

    def _stack_qbix_weights(self, layers) -> dict[int, torch.Tensor]:
        """For every layer with an indexer: its q_b weight and the indexer's wq_b moved into
        one `[2, 4096, 2048]` buffer (`_qbix_op.stack_weight`), the two parameters re-pointed
        at its two halves (the same bytes and row-major layout: every other reader, the MXF8
        copies included, is unchanged; the old storage is freed). Returns the buffers by
        layer (`_qb_ixqb`'s weight)."""
        stacked = {}
        for i in layers:
            if f"l{i}_ix_qb" not in self.w:
                continue
            qb, ixqb = self.w[f"l{i}_qb"], self.w[f"l{i}_ix_qb"]
            buf = _qbix_op.stack_weight(qb.detach(), ixqb.detach())
            qb.data = buf[0]
            ixqb.data = buf[1]
            stacked[i] = buf
        return stacked

    def _build_qbix(self, dev: int) -> dict:
        """The q_b | wq_b cells (`_qbix_op.build_all`), kept only where they match both
        projections' MTP cells row for row: then they serve exactly the rows those cells would
        (`_wide_cell`), with their bytes. Empty (the two GEMMs) otherwise or on a failed
        build."""
        cells = _try_build("q_b | wq_b fold", lambda: _qbix_op.build_all(dev))
        want = set(self._proj_mtp.get("q_b", {}))
        if not cells or set(cells) != want or set(self._proj_mtp.get("ix_qb", {})) != want:
            if cells:
                logger.warning(
                    "modeling_v2 GLM-5.2: q_b | wq_b fold cells do not match the MTP cells; "
                    "the two GEMMs stay separate"
                )
            return {}
        return cells

    def _build_qkva_ix_fold(self, dev: int) -> dict:
        """The fold cells (`_ix_fold_op.build_all`), kept only where they match the qkva
        projection's MTP cells row for row: the fold then serves exactly the rows the qkva
        cells would (`_wide_cell`), with their bytes. Empty (the two GEMMs) otherwise or if
        the build fails."""
        cells = _try_build("qkva indexer fold", lambda: _ix_fold_op.build_all(dev))
        if not cells or set(cells) != set(self._proj_mtp.get("qkva", {})):
            if cells:
                logger.warning(
                    "modeling_v2 GLM-5.2: qkva indexer fold cells do not match the qkva MTP "
                    "cells; the two GEMMs stay separate"
                )
            return {}
        return cells

    def _derive_moe(self, i: int) -> SimpleNamespace:
        """The routed experts' operands, with the per-tensor scalars folded into the kernels'
        alphas over this rank's window.

        modelopt stores reciprocals: `input_scale = 1/g`. Every routed expert's input_scale
        is 1.0 on this checkpoint, and one quantization of the hidden states feeds every
        expert, so a per-expert value could not be honoured: asserted, not assumed.
        """
        w = self.w
        window = slice(self.expert_offset, self.expert_offset + self.local_experts)
        isc1, isc2 = w[f"l{i}_e_isc1"], w[f"l{i}_e_isc2"]
        if self.dec_replicas:
            # The alphas follow the stacks' slots: this rank's window, then the replicas.
            window = torch.cat(
                [
                    torch.arange(self.expert_offset, self.expert_offset + self.local_experts),
                    torch.tensor(self.dec_hot[i], dtype=torch.long),
                ]
            ).to(isc1.device)
        assert torch.equal(isc1, w[f"l{i}_e_isc1_up"]), f"layer {i}: gate/up input_scale differ"
        assert torch.equal(w[f"l{i}_e_ws2_1"], w[f"l{i}_e_ws2_1_up"]), (
            f"layer {i}: gate/up weight_scale_2 differ"
        )
        assert bool((isc1 == 1.0).all()) and bool((isc2 == 1.0).all()), (
            f"layer {i}: routed-expert input_scale is not uniformly 1.0"
        )
        alpha1 = (isc1[window] * w[f"l{i}_e_ws2_1"][window]).contiguous()
        return SimpleNamespace(
            router=w[f"l{i}_router"],
            router_t=w[f"l{i}_router"].t(),
            router_bias=w[f"l{i}_router_bias"],
            fc1_w=w[f"l{i}_fc1_w"],
            fc1_s=w[f"l{i}_fc1_s"].view(torch.float8_e4m3fn),
            fc2_w=w[f"l{i}_fc2_w"],
            fc2_s=w[f"l{i}_fc2_s"].view(torch.float8_e4m3fn),
            alpha1=alpha1,
            # `o1[:1] / alpha1[:1]` with `o1 = alpha1 / isc2`: weights only, so made here
            # rather than allocated inside every capture.
            norm_const=((alpha1 / isc2[window])[:1] / alpha1[:1]).contiguous(),
            alpha2=(isc2[window] * w[f"l{i}_e_ws2_2"][window]).contiguous(),
            g1=(1.0 / isc1[:1]).contiguous(),
        )

    def _derive_mtp(self) -> dict:
        """The MTP module's own operands. Its attention and indexer are a trunk layer's at
        index `num_layers`; its experts are bf16."""
        w = self.w
        j = self.num_layers
        k = _mtp_eh.LOCAL_INPUT_SIZE
        # The front end's TP4 K-split: this rank owns columns [r*K, (r+1)*K) of the 12288-wide
        # concat. Held here, outside `w`, so no parameter appears that the weight table did
        # not declare.
        eh_local = nn.Parameter(
            w[f"l{j}_eh"][:, self.tp_rank * k : (self.tp_rank + 1) * k].contiguous(),
            requires_grad=False,
        )
        return {
            "enorm": w[f"l{j}_enorm"],
            "hnorm": w[f"l{j}_hnorm"],
            "eh": w[f"l{j}_eh"].t(),
            "eh_local": eh_local,
            "ehproj": _mtp_eh.build(eh_local),
            # Ranks 0-1 norm halves of e, ranks 2-3 halves of h; the offset is compiled in.
            "norm_shard": compile_mtp_norm_shard((self.tp_rank % 2) * k),
            "head_norm": w[f"l{j}_head_norm"],
            "router": w[f"l{j}_router"],
            "router_t": w[f"l{j}_router"].t(),
            "router_bias": w[f"l{j}_router_bias"],
            "fc1": w[f"l{j}_fc1"],
            "fc2": w[f"l{j}_fc2"],
        }

    def _mtp_moe_fp4(self) -> SimpleNamespace:
        """The MTP module's routed experts (bf16 in this checkpoint: modelopt excludes
        `model.layers.78`) quantized once to NVFP4 in the trunk's grouped-GEMM layout, so the
        MTP layer runs the trunk's routed path (router, fused top-k, FC1, FC2) instead of its
        bf16 pair: at decode 307 + 184 us per call on the bf16 GEMMs against about 105 + 44 us
        for a trunk layer. FC1 is stored in the trunk's row order already (64-row blocks
        alternating up and gate), so each expert quantizes as stored, up and gate sharing
        one global scale `448 * 6 / amax` (modelopt's `1 / weight_scale_2`), as a trunk
        expert's do. Input scales are 1.0 at both GEMMs, as the trunk's are, so `g1` and
        `norm_const` are 1 and the alphas are the `weight_scale_2`s. The bf16 stacks (~4.8 GB
        per rank) are released and handed back to the driver: the engine sizes its KV pool
        from the device's free memory. Draft-only numerics: they move MTP acceptance, never
        the verified tokens.

        Idempotent: the engine may run `post_load_weights` more than once, and the bf16 stacks
        are gone after the first, so a later call returns the first call's operands."""
        built = getattr(self, "_mtp_fp4_built", None)
        if built is not None:
            return built
        mw, j = self._mtp, self.num_layers

        def quant(stack: torch.Tensor, kblk: bool = False):
            e, n, k = stack.shape
            dev = stack.device
            qw = torch.empty((e, n, k // 2), dtype=torch.uint8, device=dev)
            qs = torch.empty(
                (e, n // _weights.SF_ATOM_MN, k // _weights.SF_ATOM_K, _weights.SF_ATOM),
                dtype=torch.uint8,
                device=dev,
            )
            ws2 = torch.empty((e,), dtype=torch.float32, device=dev)
            for x in range(e):
                gs = (448.0 * 6.0 / stack[x].abs().amax().float()).reshape(1)
                q, sf = fp4_quantize(stack[x].contiguous(), gs, 16, False, False)
                q = q.view(n, k // 2)
                # In the layout the trunk's FC1 / FC2 kernels are compiled for (`fc1_kblk` /
                # `fc2_kblk`).
                qw[x] = _fc1_kblock.to_kblock(q) if kblk else q
                qs[x] = _weights._sf_atoms(sf.view(torch.uint8).view(n, k // 16))
                ws2[x] = 1.0 / gs[0]
            return qw, qs.view(torch.float8_e4m3fn), ws2

        q1, s1, a1 = quant(mw["fc1"], self.fc1_kblk)
        q2, s2, a2 = quant(mw["fc2"], self.fc2_kblk)
        one = torch.ones(1, dtype=torch.float32, device=q1.device)
        for key in (f"l{j}_fc1", f"l{j}_fc2"):
            prm = self.w[key]
            prm.data = torch.empty(0, dtype=prm.dtype, device=prm.device)
        mw["fc1"] = mw["fc2"] = None
        torch.cuda.empty_cache()
        self._mtp_fp4_built = SimpleNamespace(
            router=mw["router"],
            router_t=mw["router_t"],
            router_bias=mw["router_bias"],
            fc1_w=q1,
            fc1_s=s1,
            fc2_w=q2,
            fc2_s=s2,
            alpha1=a1,
            norm_const=one,
            alpha2=a2,
            g1=one.clone(),
        )
        return self._mtp_fp4_built

    def post_load_weights(self) -> None:
        """Derive the operands, then build and compile every specialised kernel.

        Everything compiles here because a first compile inside a CUDA-graph capture breaks
        the graph; what needs an engine value compiles in `_derive_first_step`. Named for the
        shell hook that drives it.
        """
        dev = torch.cuda.current_device()
        layers = range(self.num_layers + (1 if self.mtp_enabled else 0))
        # Before any view of the two weights is taken: an MTP3 / MTP5 engine's indexer layers
        # hold their qkva and indexer key / head-weight rows in one buffer (`_qkva_ix_fold`).
        if self.max_draft_len > 1 and self._ix_fold_on:
            self._ix_fold_w = self._stack_ix_fold_weights(layers)
        # The same for q_b and the indexer's wq_b (`_qb_ixqb`).
        if self.max_draft_len > 1 and self._qbix_on:
            self._qbix_w = self._stack_qbix_weights(layers)
        self._attn = [self._derive_attention(i) for i in layers]
        self._idx = [self._derive_indexer(i) for i in layers]
        self._mlp = [(self.w[f"l{i}_mlp_gu"].t(), self.w[f"l{i}_mlp_dn"].t()) for i in layers]
        self._moe = [
            None if i < self.dense_layers else self._derive_moe(i) for i in range(self.num_layers)
        ]
        # Each layer's closing collective norms with the next layer's input norm; the last
        # layer's with the final norm.
        self._next_norm = [self.w[f"l{i + 1}_norm1"] for i in range(self.num_layers - 1)] + [
            self.w["final_norm"]
        ]
        self._mtp_fp4 = None
        if self.mtp_enabled:
            self._mtp = self._derive_mtp()
            self._mtp_fp4 = _try_build("MTP experts as NVFP4", self._mtp_moe_fp4)

        first_idx = next(i for i in range(self.num_layers) if self._idx[i] is not None)
        # One runner per (projection, compiled row count), built against one layer's row-major
        # weight, and one prefill GEMM per projection for the context row counts above
        # _DECODE_MAX_ROWS; `_project` sends a row count with neither to cuBLAS.
        proj_weights = {
            "qkva": self._attn[0].qkva,
            "q_b": self._attn[0].qb,
            "o_proj": self._attn[0].o,
            "ix_qb": self._idx[first_idx].qb,
            "ix_fused": self._idx[first_idx].fused,
            "router": self._moe[self.dense_layers].router_t,
            "dense_gate_up": self._mlp[0][0],
            "dense_down": self._mlp[0][1],
            "shared_gate_up": self._mlp[self.dense_layers][0],
            "shared_down": self._mlp[self.dense_layers][1],
        }
        self._proj = {
            name: _projection_op.build_all(name, w_colmajor.t())
            for name, w_colmajor in proj_weights.items()
        }
        if self._ctx_host_diet:
            # A mixed step's generation rows take q_b's decode cells eagerly (`_project_tail`);
            # the persistent cells among them (up to 16 rows) launch through a TVM-FFI twin of
            # the same kernel. Captured decode graphs keep the cells' own launchers.
            qb_w = proj_weights["q_b"].t()
            self._proj["q_b"] = {
                m: _projection_op.with_eager_twin(r, qb_w) if m <= _DECODE_MAX_ROWS else r
                for m, r in self._proj["q_b"].items()
            }
        # MTP3 / MTP5: the decode cells above 64 rows, taken on `_wide_decode` steps only (a
        # context step of the same row count keeps the paths below); empty for one draft token.
        wide = self.max_draft_len > 1
        self._proj_mtp = {
            name: _projection_op.build_all(name, w_colmajor.t(), _projection_op.MTP_ROWS[name])
            for name, w_colmajor in proj_weights.items()
            if wide and _projection_op.MTP_ROWS[name]
        }
        self._qkva_ix_mtp = self._build_qkva_ix_fold(dev) if self._ix_fold_w else {}
        self._qbix_mtp = self._build_qbix(dev) if self._qbix_w else {}
        if self.mtp_enabled:
            proj_weights["mtp_eh"] = self._mtp["eh"]
        self._proj_pre = _projection_op.build_prefill_all(
            {
                name: w_colmajor.t()
                for name, w_colmajor in proj_weights.items()
                if name in _projection_op.PREFILL_ENTRIES
            }
        )
        self._build_proj_mxf8()
        self._qkv_a_norm = compile_qkv_a_rmsnorm(dev, self.eps)
        # The q_a norm that also writes q_a's MXF8 operand for the q_b GEMM reading it.
        self._qkv_a_norm8 = _try_build(
            "q_a norm with the MXF8 q copy", lambda: _qkv_a_norm8_op.compile(dev, self.eps)
        )
        self._dense_swiglu = compile_dense_swiglu()
        # The shared expert's gate/up rows interleaved for the fused SwiGLU kernel, one copy per
        # layer: one copy for every layer runs every layer against one layer's experts.
        self._shared_gu_il = [
            None
            if i < self.dense_layers
            else _shared_gu_swiglu_op.prepare_weight(self._mlp[i][0].t())
            for i in layers
        ]
        self._shared_gu_swiglu = _shared_gu_swiglu_op.build_all(
            self._shared_gu_il[self.dense_layers]
        )
        self._shared_gu_swiglu_mtp = (
            _shared_gu_swiglu_op.build_all(
                self._shared_gu_il[self.dense_layers], _shared_gu_swiglu_op.MTP_ROWS
            )
            if wide
            else {}
        )
        # An MTP3 / MTP5 engine's decode row counts of 1..3 sequences without a cell
        # (`_SHARED_FUSED_UP_SEQS`), each its own compile; taken where `_cell_up` holds.
        up_rows = sorted(
            {
                (self.max_draft_len + 1) * seqs
                for seqs in range(1, _SHARED_FUSED_UP_SEQS + 1)
                if (self.max_draft_len + 1) * seqs <= _DECODE_MAX_ROWS
            }
            - set(self._shared_gu_swiglu)
        )
        self._shared_gu_swiglu_up = (
            _shared_gu_swiglu_op.build_all(self._shared_gu_il[self.dense_layers], tuple(up_rows))
            if wide and self._dec_cell_up
            else {}
        )
        self._embedding = build_embedding(self.w["embed"])
        # One compiled object per row-count group serves (2, 4, 8) or (16, 32, 48, 64); the
        # prefill kernel every other row count up to 8192, the 1..63 generation rows of a
        # mixed step included.
        self._absorb: dict[str, dict] = {"k_b": {}, "v_b": {}}
        for kind, runners in self._absorb.items():
            for group in _absorbed_bmm_op.GROUPS:
                runner = _absorbed_bmm_op.build(kind, group, dev)
                runners.update(dict.fromkeys(runner.rows, runner))
        self._absorb_pre = {
            kind: _absorbed_bmm_op.build_prefill(kind, dev) for kind in self._absorb
        }
        # The MTP3 / MTP5 decode row counts (128, 192), plain bf16 cells, `_wide_decode` only.
        self._absorb_mtp: dict[str, dict] = {"k_b": {}, "v_b": {}}
        for kind, runners in self._absorb_mtp.items():
            for group in _absorbed_bmm_op.MTP_GROUPS if wide else ():
                runner = _absorbed_bmm_op.build(kind, group, dev)
                runners.update(dict.fromkeys(runner.rows, runner))
        # The context step's K-B with the e4m3 q-base epilogue: on the context rope kernel's
        # path it writes the FMHA's fp8 query bytes itself. None keeps the bf16 K-B there.
        self._kb_fp8_pre = _try_build(
            "fp8 q-base K-B", lambda: _absorbed_bmm_op.build_prefill_fp8_q_base(dev)
        )
        # The decode cells' K-B writing the e4m3 q-base and releasing the rope kernel at
        # once, so the two run concurrently (`_attn_gen`); empty keeps the bf16 K-B.
        self._kb8: dict = {}
        for group in _absorbed_bmm_op.GROUPS:
            runner = _try_build(
                f"decode fp8 q-base K-B {group}",
                lambda group=group: _absorbed_bmm_op.build_decode_fp8_q_base(group, dev),
            )
            if runner is None:
                self._kb8 = {}
                break
            self._kb8.update(dict.fromkeys(runner.rows, runner))
        # The same at the MTP3 / MTP5 decode rows (128 / 192, and the rows between on the next
        # one up), `_wide_decode` only: the rope kernel of widths 4 / 6 then runs beside it.
        self._kb8_mtp = {}
        for group in _absorbed_bmm_op.MTP_GROUPS if wide and self._kb8 else ():
            runner = _try_build(
                f"decode fp8 q-base K-B {group}",
                lambda group=group: _absorbed_bmm_op.build_decode_fp8_q_base(group, dev),
            )
            if runner is not None:
                self._kb8_mtp.update(dict.fromkeys(runner.rows, runner))
        self._ix_pre_split = _ix_pre_split_op.compile_all(dev)
        self._ix_pre_split_mtp = (
            _ix_pre_split_op.compile_all(dev, _ix_pre_split_op.MTP_ROWS) if wide else {}
        )
        # The same kernel on a symbolic row count, for the decode row counts in 65..191
        # without a handle above (`_indexer_project`). None keeps them on the prefill split.
        self._ix_pre_split_rag = (
            _try_build("indexer split at runtime M", lambda: _ix_pre_split_op.compile_ragged(dev))
            if wide
            else None
        )
        # Runners re-pinned from the cells built above, if a load ran before (`_wide_cell`).
        self._wide_rag = {}
        # The same split at the context row counts, on the prefill GEMM against the decode
        # handles' bf16 weight; every indexer layer shares it. None keeps the fp32 projection.
        try:
            self._ix_pre_split_pre = _ix_pre_split_op.build_prefill(self._idx[first_idx].fused_bf16)
        except Exception as e:  # noqa: BLE001 -- the context rows take the fp32 projection
            logger.warning(
                f"modeling_v2 GLM-5.2: indexer prefill split build failed ({e}); using fp32"
            )
            self._ix_pre_split_pre = None
        self._ix_qside_compiled = build_ix_q_side(dev, is_neox=self.indexer_is_neox)
        # TRTLLM_GLM52_CTX_IX_NOSPLIT (default 1; 0 restores the split): a context step's
        # indexer projection is not split into bf16 keys + packed head weights; the K write and
        # the q side read their columns of the fp32 [T, 160] output in place (pitch-160
        # launchers), the K write rounding the keys to bf16 on load as the split does. Same
        # bytes, one launch fewer per indexer layer.
        self._ix_nosplit = (
            self._ix_pre_split_pre is not None
            and os.environ.get("TRTLLM_GLM52_CTX_IX_NOSPLIT", "1") == "1"
        )
        self._ix_qside_compiled_p160 = (
            build_ix_q_side(dev, is_neox=self.indexer_is_neox, wq_pitch=_IX_PROJ_COLS)
            if self._ix_nosplit
            else None
        )
        self._ix_topk_pre_compiled = build_ix_topk_prefill(
            dev, top_k=self.idx_topk, max_columns=_MOE_MAX_T
        )
        # The MLA rope kernel is keyed by the block-table width, an engine value, and the MTP
        # width: built per pair in `_derive_first_step` (the trunk's) and at the first eager use
        # of another.
        self._mla_rope_compiled: dict[tuple[int, int], _mla_rope_op.CompiledMlaRopeGeneration] = {}
        # The context kernel in both modes and its two scale arrays, built here because a
        # first compile inside a CUDA-graph capture is a broken graph.
        self._fmha_pre_compiled = _fmha_pre_op.compile(device_index=dev)
        self._fmha_pre_dc_compiled = _fmha_pre_op.compile(device_index=dev, dense_causal=True)
        self._fmha_bmm1 = torch.tensor(_FMHA_BMM1_SCALE, dtype=torch.float32, device=dev)
        self._fmha_bmm2 = torch.tensor(_FMHA_BMM2_SCALE, dtype=torch.float32, device=dev)
        self._build_context_mha(dev)
        self._build_dec_mxf8(dev)
        # A CUDA stream created inside a graph capture is not capturable.
        self._side_stream = torch.cuda.Stream(device=dev)
        # A mixed context step's generation-row attention on its own stream beside the context
        # rows' MHA (`PrefillTarget._layer_attention`); on by default,
        # TRTLLM_GLM52_CTX_GEN_SIDE=0 restores the serial order.
        self._gen_stream = (
            torch.cuda.Stream(device=dev)
            if os.environ.get("TRTLLM_GLM52_CTX_GEN_SIDE", "1") == "1"
            else None
        )
        # TRTLLM_GLM52_CTX_GEN_SIDE_QUANT (default 1; 0 restores the main-stream quantize): on
        # that stream, the generation rows' MXF8 quantize for o_proj runs right after their
        # attention instead of on the main stream after the join, off the critical path
        # whenever the generation side finishes first. Same kernel and bytes: the context MHA's
        # epilogue writes the rows below `tc` and their scale bytes, the quantize the rows from
        # `tc` and the padding scale bytes, all byte stores, so the two never write the same
        # byte.
        self._gen_side_quant = (
            self._gen_stream is not None
            and os.environ.get("TRTLLM_GLM52_CTX_GEN_SIDE_QUANT", "1") == "1"
        )
        self._build_moe(dev)
        # The peer buffers are sized for the whole token budget, so every collective shares
        # one workspace. `wide`: MTP3 / MTP5 decode steps (above 64 rows) open their MoE
        # layers on the two-shot build of the AR + norm + NVFP4 (`QUANT_TWOSHOT_ABOVE`); an
        # MTP1 or non-speculative engine builds and runs exactly what it did.
        self._lamport = build_lamport_workspace(
            self.w["embed"].shape[1],
            self.eps,
            _MOE_MAX_T,
            self.tp_rank,
            self.tp_size,
            wide=self.max_draft_len > 1,
        )
        # The row ops the fused kernels leave over: layer 0's input norm, the MTP module's
        # norms and output adds, the logits rows' lookup, the forward's copies, routing and
        # SwiGLU at row counts the fused front end has no kernel for, and the greedy picks.
        # Each is independent: one that fails to build keeps its call sites on the stock op.
        self._rows = SimpleNamespace(
            norm=_try_build("hidden RMSNorm", _hidden_norm_op.compile),
            add=_try_build("hidden add", _hidden_add_op.compile),
            # The MTP module's residual add + head norm in one launch (`_add_rmsnorm`): only an
            # engine drafting more than one token returns the module's output normed.
            add_norm=(
                _try_build("hidden add + RMSNorm", _hidden_add_norm_op.compile)
                if self.max_draft_len > 1
                else None
            ),
            gather=_try_build("hidden row gather", _row_gather_op.compile),
            topk=_try_build("sigmoid top-k", _topk_rows_op.compile),
            swiglu=_try_build("SwiGLU rows", _swiglu_rows_op.compile),
            copy=_try_build("copy/convert", _copy_op.compile),
            argmax=_try_build("argmax", _argmax_op.compile),
        )
        # The MTP module's context steps run attention-to-output on the draft head's rows only
        # (`PrefillTarget._mtp_last_plan`): the row gather / scatter they need.
        self._rows_sel = (
            _try_build("row gather / scatter", _rows_sel_op.compile) if self.mtp_enabled else None
        )
        if self.tp_rank == 0:
            built = " ".join(f"{k}={v is not None}" for k, v in vars(self._rows).items())
            logger.info(
                f"modeling_v2 GLM-5.2 row ops built: {built} "
                f"logits_gather={self._lamport.gather is not None}"
            )
        if self.max_draft_len > 1:
            self._warm_draft_rows()

        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)

    def _build_moe(self, dev) -> None:
        """The routed experts' buffers and compiled kernels, trunk and MTP.

        Every operand handed to these kernels lives in a buffer allocated here, in two slots
        alternating per layer. The engine allocates its tensors inside the CUDA-graph
        capture; handed to these kernels directly, replays routed as though every id were
        zero. And the kernels launch with PDL, so a layer's kernel can start while the
        previous layer's is still reading: one slot measured every layer but the last at
        cosine 0.64-0.93. The NVFP4 input (`xq` / `xsf`) is one buffer, written by the
        closing collective or the quantize launch and read by FC1 in the same layer.

        A decode kernel that fails to build raises: those steps have no stock path for the
        packed weights. A prefill kernel that fails to build is None, and its call site
        takes the stock op (the decode grouped GEMMs for the prefill pair).
        """
        hidden, inter, experts = self.w["embed"].shape[1], self.moe_inter, self.local_experts
        rows = _MOE_MAX_T
        bf16 = {"dtype": torch.bfloat16, "device": dev}
        f32 = {"dtype": torch.float32, "device": dev}
        u8 = {"dtype": torch.uint8, "device": dev}

        def optional(build, what: str = "MoE prefill kernel"):
            try:
                return build()
            except Exception as e:  # noqa: BLE001 -- that launch takes its stock op at the call site
                logger.warning(f"modeling_v2 GLM-5.2: {what} build failed ({e}); using stock op")
                return None

        def slot(mtp: bool) -> SimpleNamespace:
            counts, barrier = _moe_routing_op.allocate_workspace(dev)
            widths = _MOE_DECODE_WIDTHS if mtp else range(1, _DECODE_MAX_ROWS + 1)
            ns = SimpleNamespace(
                # The routed output is returned to the caller, so it alternates too: one
                # buffer would be overwritten by the next layer before the caller used it.
                out=torch.zeros((rows, hidden), **bf16),
                x=torch.zeros((rows, hidden), **bf16),
                w=torch.zeros((rows, _MOE_TOP_K), **f32),
                sel=torch.empty(rows * _MOE_TOP_K, dtype=torch.int32, device=dev),
                # Context replicas: one more partial tile per replica slot (trunk only).
                maps=_moe_routing_op.allocate_maps(
                    rows, dev, replicas=0 if mtp else self.ctx_replicas
                ),
                counts=counts,
                barrier=barrier,
                router_out=torch.zeros((_DECODE_MAX_ROWS, self.num_experts), **f32),
                router_partials=torch.zeros(
                    (_DECODE_MAX_ROWS, _moe_router_op.SPLIT_K, self.num_experts), **f32
                ),
                # The fused top-k checks its outputs against the exact row count: one set
                # per width; with decode expert replicas the push widths' sets hold the
                # replica top-k's `64 + replicas` tiles.
                topk_out={
                    w: _moe_topk_rep_op.allocate_outputs(w, self.dec_replicas, dev)
                    if not mtp and self.dec_replicas and _moe_topk_rep_op.uses_rep_path(w)
                    else _moe_topk_op.allocate_decode_outputs(w, dev)
                    for w in widths
                },
            )
            if mtp:
                ns.c = _mtp_experts_op.allocate_c(rows, inter, dev)
                return ns
            ns.c = _moe_experts_op.allocate_decode_c(rows, inter, dev)
            ns.sfc = _moe_experts_op.allocate_sfc(rows, inter, dev, self.ctx_replicas)
            ns.router_x = torch.zeros((_DECODE_MAX_ROWS, hidden), **bf16)
            if self.max_draft_len > 1:
                # The wide decode router's `[rows, 4, 256]` partials (MTP3 / MTP5 only).
                ns.router_wide_out = torch.zeros(
                    _moe_router_op.output_shape(_moe_router_op.WIDE_MAX_ROWS, wide=True), **f32
                )
            ns.logits = torch.zeros((rows, 1, self.num_experts), **f32)
            ns.topk_scratch = _moe_topk_op.allocate_prefill_scratch(dev)
            if self.ctx_replicas:
                ns.topk_rep_scratch = _moe_topk_pre_rep_op.allocate_scratch(self.ctx_replicas, dev)
            return ns

        self._moe_slots = [slot(False) for _ in range(2)]
        # One byte per element, as the kernels' entries allocate: cute counts the shape in
        # elements and reads half as many bytes.
        self._moe_xq = torch.zeros((rows, hidden // 2), **u8)
        self._moe_xsf = torch.zeros((rows, hidden // _MOE_SF_VEC), **u8)
        # The prefill shared expert's SwiGLU output. One is enough: up and down run back to
        # back on the shared branch's stream, which forks again only after this layer's reduce.
        self._moe_mid = torch.zeros((rows, _SHARED_INTER), **bf16)
        s0 = self._moe_slots[0]
        k = SimpleNamespace()
        # One routing function per schedule bucket any step reaches, shared with the MTP
        # layer: a bucket compiled on first use would compile inside a step.
        k.routing = _moe_routing_op.build(_moe_routing_op.ROW_COUNTS, dev)
        k.experts = _moe_experts_op.build_decode(
            c=s0.c,
            sfc=s0.sfc,
            maps=s0.maps,
            xq=self._moe_xq,
            xsf=self._moe_xsf,
            out=s0.out,
            hidden=hidden,
            inter=inter,
            # The decode expert replicas' slots follow this rank's 64 (0 when off).
            experts=experts + self.dec_replicas,
            max_rows=rows,
            fc1_kblk=self.fc1_kblk,
            fc2_kblk=self.fc2_kblk,
            # One FC2 tile counter per MoE call site: the trunk layers and the MTP layer.
            dyn_slots=self.num_layers + 1,
        )
        k.pair = optional(
            lambda: _moe_experts_op.build_prefill_pair(
                hidden, inter, experts, dev, fc1_kblk=self.fc1_kblk, fc2_kblk=self.fc2_kblk
            )
        )
        if k.pair is not None:
            for sl in self._moe_slots:
                sl.pc = _moe_experts_op.allocate_pair_c(rows, inter, dev, self.ctx_replicas)
        # Context replicas: the replica top-k and the prefill pair on the `64 + R` deep
        # stacks, both or neither (the replica maps address slots past 64).
        k.topk_pre_rep = k.pair_rep = None
        self._ctx_hot_ids = {}
        if self.ctx_replicas and k.pair is not None:
            rep = optional(
                lambda: (
                    _moe_topk_pre_rep_op.build_prefill(self.ctx_replicas, dev),
                    _moe_experts_op.build_prefill_pair(
                        hidden,
                        inter,
                        experts + self.ctx_replicas,
                        dev,
                        fc1_kblk=self.fc1_kblk,
                        fc2_kblk=self.fc2_kblk,
                    ),
                )
            )
            k.topk_pre_rep, k.pair_rep = rep if rep is not None else (None, None)
            self._ctx_hot_ids = {
                i: torch.tensor(ids[: self.ctx_replicas], dtype=torch.int32, device=dev)
                for i, ids in self.dec_hot.items()
                if len(ids) >= self.ctx_replicas
            }
        # The maps the replica top-k wrote last (`_moe_topk`); `_moe_experts` runs them on
        # the replica pair.
        self._moe_rep_maps = None
        k.router_pre = optional(lambda: _moe_router_op.build_prefill(dev))
        k.topk_pre = optional(lambda: _moe_topk_op.build_prefill(dev))
        k.quant_pre = optional(lambda: _moe_quant_op.build_prefill(dev))
        k.zero_pre = optional(lambda: _moe_zero_op.build_prefill(dev))
        k.merge = optional(lambda: _moe_merge_op.build(dev))
        # The shared expert's prefill pair, both or neither: up writes the SwiGLU output down
        # reads. Built against the first MoE layer's weights; each call passes its own.
        shared = optional(
            lambda: (
                _shared_gu_swiglu_op.build_prefill(self._shared_gu_il[self.dense_layers]),
                _projection_op.build_shared_down_prefill(self._mlp[self.dense_layers][1].t()),
            )
        )
        k.shared_up_pre, k.shared_down_pre = shared if shared is not None else (None, None)
        k.quant = {w: _moe_quant_op.build_decode(w, dev) for w in _MOE_DECODE_WIDTHS}
        k.zero = {w: _moe_zero_op.build_decode(w, dev) for w in _MOE_DECODE_WIDTHS}
        # The fused top-k releases FC1 early only at a power of two: other widths are padded
        # and run the routing kernel next, which does not wait for the ids; released early it
        # reads them mid-write (a 3-request step hung that way).
        # The decode L2 prefetch of the routed FC1 weights: forked at a decode MoE layer's
        # o_proj end, ranked by the counts the same layer's previous decode step's top-k wrote
        # into its row of `hist` (one row per MoE call site, the MTP layer's included).
        self._pf = None
        pf = (
            _try_build("decode MoE L2 prefetch", lambda: _l2pf_op.compile(dev))
            if self._moe_l2_prefetch
            else None
        )
        if pf is not None:
            self._pf = SimpleNamespace(
                compiled=pf,
                hist=torch.zeros((self.num_layers + 1, 64), dtype=torch.int32, device=dev),
                stream=torch.cuda.Stream(device=dev),
                pending=None,
            )
        k.topk = {
            w: _moe_topk_op.build_decode(
                w,
                _moe_router_op.decode_splits(w),
                dev,
                signal_gg1=(w & (w - 1)) == 0,
                expert_hist=self._pf is not None and _moe_topk_op.uses_push_path(w),
            )
            for w in range(1, _DECODE_MAX_ROWS + 1)
        }
        # Decode expert replicas: the push widths select on the replica top-k, with each
        # trunk layer's hot ids (the MTP layer passes all -1: the home mapping).
        k.topk_rep = {}
        if self.dec_replicas:
            k.topk_rep = {
                w: _moe_topk_rep_op.build_decode(
                    w,
                    _moe_router_op.decode_splits(w),
                    self.dec_replicas,
                    dev,
                    signal_gg1=(w & (w - 1)) == 0,
                    expert_hist=self._pf is not None,
                )
                for w in range(1, _DECODE_MAX_ROWS + 1)
                if _moe_topk_rep_op.uses_rep_path(w)
            }
            self._dec_hot_ids = {
                i: torch.tensor(ids, dtype=torch.int32, device=dev)
                for i, ids in self.dec_hot.items()
            }
            self._dec_hot_none = torch.full((self.dec_replicas,), -1, dtype=torch.int32, device=dev)
        # Early-release top-k handles for the unpadded trunk decode widths that are not powers
        # of two (`TRTLLM_GLM52_MOE_DEC_UNPAD_SIGNAL`): the decode step widths of an MTP3 /
        # MTP5 engine, `requests * (max_draft_len + 1)` up to 64 rows. Trunk only: the MTP
        # layer pads those widths and runs the routing kernel after its top-k, which must not
        # be released early (it does not wait for the ids).
        k.topk_sig = {}
        if self._moe_dec_unpad and self._moe_dec_unpad_signal and self.max_draft_len > 1:
            step = self.max_draft_len + 1
            sig_widths = [w for w in range(step, _DECODE_MAX_ROWS + 1, step) if w & (w - 1)]

            def build_sig(w):
                if w in k.topk_rep:
                    return _moe_topk_rep_op.build_decode(
                        w,
                        _moe_router_op.decode_splits(w),
                        self.dec_replicas,
                        dev,
                        signal_gg1=True,
                        expert_hist=self._pf is not None,
                    )
                return _moe_topk_op.build_decode(
                    w,
                    _moe_router_op.decode_splits(w),
                    dev,
                    signal_gg1=True,
                    expert_hist=self._pf is not None and _moe_topk_op.uses_push_path(w),
                )

            built = _try_build(
                "MoE unpadded decode early-release top-k",
                lambda: {w: build_sig(w) for w in sig_widths},
            )
            k.topk_sig = built or {}
        # The MTP3 / MTP5 decode widths' front end (`_moe_decode_exact`, 65..256 rows): the
        # wide decode router's split-K partials, and the fused top-k summing them, one handle
        # per row range, writing the L2 prefetch's counts. MTP1 and no speculation build
        # neither. The partials need the wide top-k: without it the router is dropped too
        # (the prefill router and top-k serve those steps); without the router the top-k
        # reads the prefill router's one plane.
        k.router_wide = None
        k.topk_wide = {}
        if self.max_draft_len > 1 and k.topk_pre is not None:
            k.router_wide = optional(
                lambda: _moe_router_op.build_wide(dev), "wide decode MoE router"
            )
            splits = _moe_router_op.SPLIT_K if k.router_wide is not None else 1
            wide = optional(
                lambda: {
                    r: _moe_topk_op.build_wide(
                        r[1],
                        splits,
                        dev,
                        expert_hist=self._pf is not None,
                        cluster=self.wide_topk_schedule[0],
                        trim=self.wide_topk_schedule[1],
                    )
                    for r in _moe_topk_op.WIDE_RANGES
                },
                "wide decode MoE top-k",
            )
            if wide is None:
                k.router_wide = None
            k.topk_wide = wide or {}
        self._build_moe_wide_rep(k, dev)
        # Shared with the MTP layer's front end: the same compiled kernel per row count.
        k.router = _moe_router_op.build_all(dev)
        self._moe_k = k
        if self.mtp_enabled and self._mtp_fp4 is None:
            mw = self._mtp
            self._mtp_slots = [slot(True) for _ in range(2)]
            m0 = self._mtp_slots[0]
            self._mtp_k = SimpleNamespace(
                experts=_mtp_experts_op.build(
                    x=m0.x,
                    c=m0.c,
                    out=m0.out,
                    topk_w=m0.w,
                    maps=m0.maps,
                    fc1_w=mw["fc1"],
                    fc2_w=mw["fc2"],
                    max_rows=rows,
                ),
                # No quantize launch on the bf16 path to carry the output clear, so the top-k
                # clears it; every MTP width is a power of two, so it always releases FC1.
                topk={
                    w: _moe_topk_op.build_decode(
                        w, _moe_router_op.decode_splits(w), dev, signal_gg1=True, fuse_zero=True
                    )
                    for w in _MOE_DECODE_WIDTHS
                },
            )

    def _warm_draft_rows(self) -> None:
        """MTP3 / MTP5, at load: launch once on zeros what an eager draft step runs at a
        per-row-count module. Those compile at load but launch only on a step of that count,
        and the first such launch loads its module inside the step: the load holds the host
        until the GPU drains (19-23 ms on a context step's eager draft step).
        Decode graphs load the counts they capture; an eager draft step takes any count 1..64.
        - The MTP layer's routed experts at every count 1..64: the top-k per count; routing,
          quantize and clear per power of two. Same path, handles and buffers as the step
          (the FC2 tile counters reset at each launch's end).
        - The eh_proj K-split instances up to 64 rows (`_mtp_front`).
        - The shared expert's gate|up at the counts no runtime-M cell serves (33..63 but 48:
          the 48 / 64 cells are compiled for their own M, and the fused MTP cells take 65
          rows and up): `_project` itself on the step's operand shape, so cuBLAS loads the
          kernels it picks for those shapes here.
        The slot parity, the MTP layer's L2 prefetch counts and the route counters are
        restored after."""
        dev = self.w["embed"].device
        j = self.num_layers
        hits, misses = dict(self.route_hits), dict(self.route_misses)
        x = torch.zeros(
            (_DECODE_MAX_ROWS, self.w["embed"].shape[1]), dtype=torch.bfloat16, device=dev
        )
        if self.mtp_enabled:
            runner = self._mtp["ehproj"]
            for m in _mtp_eh.SUPPORTED_M_VALUES:
                if m <= _DECODE_MAX_ROWS:
                    eh = torch.zeros(
                        (m, _mtp_eh.LOCAL_INPUT_SIZE), dtype=torch.bfloat16, device=dev
                    )
                    runner.forward_rows(eh, m)
            self._draft_eager = True
            try:
                for rows in range(1, _DECODE_MAX_ROWS + 1):
                    if rows not in self._shared_gu_swiglu:
                        self._project("shared_gate_up", x[:rows], self._mlp[j][0])
            finally:
                self._draft_eager = False
        if self._mtp_fp4 is not None:
            slot = self._moe_slot
            hist = self._pf.hist[j].clone() if self._pf is not None else None
            for rows in range(1, _DECODE_MAX_ROWS + 1):
                self._routed(x[:rows], self._mtp_fp4, rows, False, None, False, False, j)
            self._moe_slot = slot
            if hist is not None:
                self._pf.hist[j].copy_(hist)
        torch.cuda.synchronize()
        self.route_hits.clear()
        self.route_hits.update(hits)
        self.route_misses.clear()
        self.route_misses.update(misses)

    def _select_target(self, attn_metadata) -> "PrefillTarget | DecodeTarget":
        """Which target runs this step.

        One host-side integer read and nothing else: a decode-only CUDA graph is a
        per-capture constant environment. A mixed batch goes to prefill, so prefill is the
        general case and decode the specialization that may assume it has no context rows.
        The phase is also recorded for the projections (`_wide_decode`): with more than one
        draft token a decode step's rows reach 32 x (max_draft_len + 1), past the 64 rows
        that otherwise tell a decode row count from a context one.
        """
        decode = attn_metadata.num_contexts == 0
        self._wide_decode = decode and self.max_draft_len > 1
        return self._decode if decode else self._prefill

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Derive what the first step fixes, then route."""
        # Set on every metadata instance this model sees (the engine's and each CUDA graph
        # runner's); it takes effect from that instance's next `prepare()`.
        attn_metadata.dense_topk_on_device = self._dense_topk_on_device
        self._derive_first_step(attn_metadata)
        out = self._select_target(attn_metadata).forward(
            attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            lora_params=lora_params,
            **kwargs,
        )
        # After the step, so a report includes it. Every context step reports (they run
        # eagerly and are few); decode steps every `_ROUTE_REPORT_EVERY` Python forwards.
        self._forwards += 1
        if self.tp_rank == 0 and (
            attn_metadata.num_contexts or self._forwards % _ROUTE_REPORT_EVERY == 0
        ):
            logger.info(f"modeling_v2 GLM-5.2 routes after {self._forwards} forwards: "
                        f"{self.route_report()}")  # fmt: skip
        return out

    def _derive_first_step(self, md) -> None:
        """What depends on values the engine resolves: compiled on the first forward, which
        is eager, ahead of any capture.

        The indexer K pool views are not cached: TRT-LLM replaces the KV cache manager during
        boot, and a view taken before the swap still reads the same sizes, strides and
        pointer while writing into a dead allocation. The pool's layer mapping does survive
        that swap (same trunk, same page size), but not the speculative one, so the MTP layer
        reads its own live.
        """
        # Derived again for a new KV cache manager: the final one can hold a shorter
        # `max_seq_len`, so a narrower block table, than the one the first forward saw.
        if not self._derive_pending and md.kv_cache_manager is self._derived_for:
            return
        dev = torch.cuda.current_device()
        # A table shorter than max_seq_len is read out of bounds with no check, and
        # `rope_max_positions` is inert on the MLA path.
        if md.max_seq_len > self._rope_positions:
            self._rope_positions = md.max_seq_len
            self._rope = self._rope_tables(self.w["final_norm"].device, self._rope_positions)
        pool_map = md.host_kv_cache_pool_mapping
        self._stride_factor = pool_map.shape[0] * md.tokens_per_block
        self._pool_layer_id = [int(row[1]) for row in pool_map.tolist()]
        max_blocks = int(md.block_table.shape[1])
        self._req_global_compiled = build_req_index_to_global(
            dev, topk=self.idx_topk, max_blocks=max_blocks
        )
        self._req_base_compiled = build_req_index_to_base(
            dev, topk=self.idx_topk, max_blocks=max_blocks
        )
        # The decode scorer's column count is frozen at capture, so it is an engine constant:
        # a replay whose sequences outgrew a batch maximum would read unwritten columns.
        self._max_ctx_len = md.max_seq_len
        # A width with no handle takes the trtllm op: max_seq_len <= index_topk, or a failed build.
        # The 1-row steps, MTP1's 2, and MTP3 / MTP5's max_draft_len + 1.
        self._ix_topk_dec_compiled = {}
        for next_n in (1, *self.mtp_widths):
            try:
                self._ix_topk_dec_compiled[next_n] = build_ix_topk_decode(
                    dev, max_columns=md.max_seq_len, top_k=self.idx_topk, next_n=next_n
                )
            except Exception as e:  # noqa: BLE001 -- that width takes the trtllm op
                logger.warning(
                    f"modeling_v2 GLM-5.2: indexer top-k decode build failed for next_n={next_n} "
                    f"({e}); using the trtllm op"
                )
        # GVR only where it serves this engine's width, warmed ahead of any capture.
        self._dec_topk_gvr_live = self._dec_topk_gvr is not None and _dec_topk_gvr_serves(
            int(md.max_seq_len)
        )
        if self._dec_topk_gvr_live:
            self._warm_dec_topk_gvr(md)
        if self.mtp_enabled:
            # MTP3 / MTP5 also width 1: the MTP layer's 1-token draft steps
            # (`_mtp_rope_specialised`). A separate draft pool's block-table width compiles at
            # the first eager draft step, as width max_draft_len + 1 does.
            for p in self.mtp_widths + ((1,) if self.max_draft_len > 1 else ()):
                self._mla_rope_handle(int(md.kv_cache_block_offsets.shape[-1]), dev, p)
        # The context rope/append kernel is keyed by the page size. None keeps the stock op.
        try:
            self._ctx_rope_compiled = _ctx_rope_op.compile(
                tokens_per_block=md.tokens_per_block, device_index=dev
            )
        except Exception as e:  # noqa: BLE001 -- the context rows take the stock rope/append op
            logger.warning(
                f"modeling_v2 GLM-5.2: context rope/append build failed ({e}); using stock op"
            )
            self._ctx_rope_compiled = None
        k_pool = md.kv_cache_manager.get_indexer_k_cache_buffers(0)
        assert (
            k_pool.dtype == torch.uint8
            and k_pool.ndim == 4
            and k_pool.shape[2] == 1
            and k_pool.shape[3] >= _INDEXER_PER_TOKEN_MIN
        ), f"the fused indexer k write needs a [blocks, block, 1, >=132] uint8 pool: {k_pool.shape}"
        # Both pool layouts: the trunk's per-layer views are strided and a separate draft pool's
        # may not be, and a layout first met inside a capture cannot compile there.
        self._ix_k_compiled = {
            contiguous: build_ix_k_cache(
                dev,
                is_neox=self.indexer_is_neox,
                cache_contiguous=contiguous,
                f32_input=self._ix_nosplit,
            )
            for contiguous in (False, True)
        }
        if self.max_draft_len > 1 and self._ix_qk_on:
            self._ix_qk_compiled = (
                _try_build(
                    "indexer q side + K update",
                    lambda: {
                        contiguous: build_ix_qk(
                            dev, is_neox=self.indexer_is_neox, cache_contiguous=contiguous
                        )
                        for contiguous in (False, True)
                    },
                )
                or {}
            )
        # Every split-K variant the decode attention selects for 1..max_batch_size sequences,
        # both index kinds. `max_num_requests` is the engine's max_batch_size, in every
        # metadata it hands over (a CUDA-graph copy keeps it): no step schedules more
        # sequences, and a CUDA-graph batch is padded only up to a captured size no larger.
        # Its grid holds the kernel's default 2 query rows per sequence (MTP1, and the 1-row
        # steps). MTP3 / MTP5 add a second grid for max_draft_len + 1 rows (4 / 6), used only by
        # steps that wide: the split-K pick divides the SMs by sequences x grid width, so a
        # 1-row draft step on the wide grid would take fewer split-K CTAs per sequence (at 32
        # sequences one where the width-2 grid takes two) and leave 3 or 5 of every 4 / 6 query
        # CTAs idle. The variants are keyed by split and reducer only, so most are shared.
        self._fmha_dec_compiled = _fmha_dec_op.compile(
            max_batch_size=int(md.max_num_requests), device_index=dev
        )
        if self.max_draft_len > 1:
            self._fmha_dec_wide = _fmha_dec_op.compile(
                max_batch_size=int(md.max_num_requests),
                device_index=dev,
                max_sq=self.max_draft_len + 1,
            )
            # The trunk's pure decode steps at that width read the indexer's positions
            # directly (one gather of a pool row per group of rows that selected it); the MTP
            # layer and mixed steps keep the per-row handle above.
            if (
                self._union_on
                and self.max_draft_len + 1 in _fmha_union_op.UNION_ROWS_PER_GROUP
                and int(md.tokens_per_block) == _fmha_dec_op.LAYER_ROW_STRIDE
            ):
                self._fmha_union = _fmha_union_op.compile_union(
                    max_batch_size=int(md.max_num_requests),
                    device_index=dev,
                    max_sq=self.max_draft_len + 1,
                )
            elif self._union_on and self.max_draft_len + 1 in _fmha_union_op.UNION_ROWS_PER_GROUP:
                logger.warning(
                    f"modeling_v2 GLM-5.2: union decode attention needs tokens_per_block "
                    f"{_fmha_dec_op.LAYER_ROW_STRIDE} (got {int(md.tokens_per_block)}); "
                    "pure decode steps keep the per-row decode attention"
                )
        self._derive_pending = False
        self._derived_for = md.kv_cache_manager

    def _warm_dec_topk_gvr(self, md) -> None:
        """Compile the GVR decode top-k's launchers for every row count a decode step of this
        engine can carry (1..max_batch_size sequences at each generation width), as the
        engine's `warmup_selfsampling_topk` does for its own `next_n`: a launcher is keyed by the
        exact row count and the logits row stride, and one first met inside a capture cannot
        compile there (that graph then keeps ours, `_indexer_decode`). The scorer's rows are
        the DSL paged-MQA arena's, `max_seq_len` rounded up to 256 columns."""
        from tensorrt_llm._torch.kernels.blackwell.top_k import (
            gvr_topk_decode_self_sampling_host as gvr_host,
        )

        seqs = int(md.max_num_requests)
        for next_n in (1, *self.mtp_widths):
            gvr_host.warmup_varlen(
                self.idx_topk,
                int(md.max_seq_len),
                compress_ratio=1,
                next_n=next_n,
                num_rows_list=tuple(next_n * b for b in range(1, seqs + 1)),
                row_stride=(int(md.max_seq_len) + 255) // 256 * 256,
                deterministic=self._dec_topk_gvr.gvr_deterministic,
            )

    # ---- blocks both targets run ----

    def _count(self, route: str, hit: bool) -> None:
        """One call of `route`: on its CuTe DSL kernel (`hit`) or on the stock op."""
        counts = self.route_hits if hit else self.route_misses
        counts[route] = counts.get(route, 0) + 1

    def route_report(self) -> str:
        """`name=hits` per route, `/misses` appended where a call took the stock op."""
        names = sorted(set(self.route_hits) | set(self.route_misses))
        report = " ".join(
            f"{n}={self.route_hits.get(n, 0)}"
            + (f"/{self.route_misses[n]}" if self.route_misses.get(n) else "")
            for n in names
        )
        if self.mtp_last_declines:
            declined = ",".join(f"{k}:{v}" for k, v in sorted(self.mtp_last_declines.items()))
            report += f" mtp_ctx_last_declined={declined}"
        return report

    def _rmsnorm(self, x, weight):
        """RMSNorm of hidden-width rows: the CuTe DSL kernel, else FlashInfer's."""
        rows = self._rows
        hit = rows.norm is not None and _hidden_norm_op.takes(x, weight)
        self._count("hidden_norm", hit)
        if hit:
            return cute_dsl_hidden_rmsnorm(x, weight, self.eps, rows.norm)
        return flashinfer_rmsnorm(x, weight, self.eps)

    def _add(self, a, b):
        """A residual add of hidden-width rows: the CuTe DSL kernel, else `torch.add`."""
        rows = self._rows
        hit = rows.add is not None and _hidden_add_op.takes(a, b)
        self._count("hidden_add", hit)
        if hit:
            return cute_dsl_hidden_add(a, b, rows.add)
        return torch.add(a, b)

    def _add_rmsnorm(self, a, b, weight):
        """`_rmsnorm(_add(a, b), weight)`; the sum is not kept. One CuTe DSL launch where its
        kernel built and the two calls would both take their CuTe DSL kernels (the pair it
        is bit for bit), else the two calls."""
        rows = self._rows
        hit = (
            rows.add_norm is not None
            and rows.add is not None
            and rows.norm is not None
            and _hidden_add_norm_op.takes(a, b, weight)
        )
        self._count("hidden_add_norm", hit)
        if hit:
            return cute_dsl_hidden_add_rmsnorm(a, b, weight, self.eps, rows.add_norm)
        return self._rmsnorm(self._add(a, b), weight)

    def _copy_into(self, dst, src):
        """`dst.copy_(src)`: the CuTe DSL copy/convert kernel for the pairs it takes."""
        rows = self._rows
        hit = rows.copy is not None and _copy_op.takes(dst, src)
        self._count("copy", hit)
        if hit:
            return cute_dsl_copy(dst, src, rows.copy)
        return dst.copy_(src)

    def _route(self, logits, bias):
        """`(topk_w, topk_ids)` for a chunk the fused front end does not take: the CuTe DSL
        top-k when it takes the routing, else `noaux_tc_op`."""
        rows = self._rows
        args = (logits, bias, self.n_group, self.topk_group, self.topk)
        hit = rows.topk is not None and _topk_rows_op.takes(*args)
        self._count("topk_rows", hit)
        if hit:
            return cute_dsl_moe_sigmoid_topk(*args, self.routed_scale, rows.topk)
        return noaux_tc_op(*args, self.routed_scale)

    def build_logits_head(self, lm_head) -> None:
        """Compile the `lm_head` local GEMM (this rank's vocabulary shard) at every row
        count it has a schedule for. Taken only for the head the gather was built for:
        unpadded, column-parallel, gathering, the TP4 shard shape. Called by the shell
        once the weight is loaded; a head this does not take keeps upstream's path."""
        w = lm_head.weight
        if (
            self._lamport.gather is None
            or not getattr(lm_head, "gather_output", False)
            or getattr(lm_head, "padding_size", 0) != 0
            or int(getattr(lm_head, "tp_size", 1)) != self.tp_size
            or tuple(w.shape) != (_allgather_op.SHARD_N, self.w["embed"].shape[1])
        ):
            logger.warning("modeling_v2 GLM-5.2: lm_head not on the gathered fast path")
            return
        self._lm_head = lm_head
        # MTP3 / MTP5 add their decode row counts with a schedule of their own (MTP5's 192).
        rows = _projection_op.ROWS["lm_head"] + (
            _projection_op.MTP_ROWS["lm_head"] if self.max_draft_len > 1 else ()
        )
        self._lm_head_runners = _projection_op.build_all("lm_head", w, rows)
        if self.tp_rank == 0:
            logger.info(
                f"modeling_v2 GLM-5.2 lm_head: {len(self._lm_head_runners)} of "
                f"{len(rows)} row counts on the CuTe DSL GEMM"
            )

    def logits(self, x, lm_head, logits_processor, attn_metadata):
        """Full-vocabulary fp32 logits for the hidden rows `x`, every one of them (what
        `LogitsProcessor.forward` returns with `return_context_logits=True`): the local
        GEMM on its CuTe DSL kernel, then one Lamport launch that gathers the four shards
        and widens them. Any other head, row count or layout keeps `logits_processor`.

        MTP3 / MTP5 only: a count of 65..255 rows without its own schedule (mixed steps,
        and decode steps of fewer than 32 sequences; MTP5's 192 verify rows have one) runs
        on the next compiled bucket (128, 192 or 256): that instance launched at the exact
        rows (`_projection_op.at_rows`), each row bitwise what the padded launch wrote for
        it; with `_wide_bucket` off (or a bucket compiled for one M), on zero rows past
        `rows`, returning the first `rows`. MTP1 keeps the path above."""
        rows = int(x.shape[0])
        runner = self._lm_head_runners.get(rows) if lm_head is self._lm_head else None
        padded = rows
        if runner is None and lm_head is self._lm_head and self.max_draft_len > 1 and rows < 256:
            padded = min((m for m in self._lm_head_runners if m > rows), default=rows)
            runner = self._lm_head_runners.get(padded) if padded <= 256 else None
            if runner is not None and self._wide_bucket:
                key = ("lm_head", rows)
                if key not in self._wide_rag:
                    self._wide_rag[key] = _projection_op.at_rows(runner, rows)
                if self._wide_rag[key] is not None:
                    runner, padded = self._wide_rag[key], rows
                    self._count("logits_at_rows", True)
        if (
            runner is not None
            and x.dtype is torch.bfloat16
            and x.is_contiguous()
            and x.data_ptr() % 128 == 0
            and padded * _allgather_op.SHARD_N <= self._lamport.max_rows * self._lamport.hidden
        ):
            self._count("logits", True)
            if padded != rows:
                self._count("logits_pad", True)
                xp = torch.zeros([padded, x.shape[1]], dtype=x.dtype, device=x.device)
                xp[:rows].copy_(x)
                x = xp
            local = torch.empty([padded, runner.n], dtype=torch.bfloat16, device=x.device)
            cute_dsl_projection(x, lm_head.weight, local, runner)
            out = torch.empty(
                [padded, self.tp_size * runner.n], dtype=torch.float32, device=x.device
            )
            out = cute_dsl_allgather_widen(local, out, self._lamport)
            return out if padded == rows else out[:rows]
        self._count("logits", False)
        return logits_processor.forward(x, lm_head, attn_metadata, True)

    def gather_rows(self, hidden, rows):
        """`hidden[rows]`: the CuTe DSL row gather, else `embedding(rows, hidden)`."""
        handle = self._rows.gather
        hit = handle is not None and _row_gather_op.takes(hidden, rows)
        self._count("logits_rows", hit)
        if hit:
            return cute_dsl_hidden_row_gather(hidden, rows, handle)
        return nn.functional.embedding(rows, hidden)

    def _check_rows(self, partial) -> None:
        # The all-reduce kernels take the row count from their input, so a call above the
        # token budget the peer buffers were sized for would write past every peer's buffer.
        assert partial.shape[0] <= self._lamport.max_rows, (
            f"{partial.shape[0]} rows exceed the collective's budget of {self._lamport.max_rows}"
        )

    def _embed(self, input_ids):
        """The embedding rows for `input_ids`, in slices the kernel takes."""
        if input_ids.shape[0] <= _EMBED_MAX_ROWS:
            return cute_dsl_embedding(input_ids, self.w["embed"], self._embedding)
        return torch.cat(
            [
                cute_dsl_embedding(ids, self.w["embed"], self._embedding)
                for ids in input_ids.split(_EMBED_MAX_ROWS)
            ]
        )

    def _reduce(self, partial):
        n = self._lamport.max_rows
        if partial.shape[0] > n:
            # A step above the token budget the peer buffers hold reduces in slices.
            return torch.cat([self._reduce(p) for p in partial.split(n)])
        self._check_rows(partial)
        return cute_dsl_allreduce(partial, torch.empty_like(partial), self._lamport)

    def _dec8_takes(self, name: str, x, wd8) -> bool:
        """Whether `_project` sends `x` to the decode MXF8 GEMM: its runner built, `wd8`
        given, a decode row count, and an activation its quant reads."""
        return (
            wd8 is not None
            and name in self._dec8
            and 0 < int(x.shape[0]) <= _DECODE_MAX_ROWS
            and x.dtype == torch.bfloat16
            and x.stride(-1) == 1
            and x.stride(0) % 8 == 0
            and x.data_ptr() % 16 == 0
        )

    def _dec8w_takes(self, name: str, x, wd8) -> bool:
        """Whether `_project` sends `x` to the wide decode MXF8 GEMM: a `_wide_decode` step's
        rows above 64 (up to 192) of a call whose decode path is the decode MXF8 GEMM (`wd8`:
        the trunk o_proj), its wide runner built, and an activation its quant reads."""
        return (
            self._wide_decode
            and wd8 is not None
            and name in self._dec8w
            and _DECODE_MAX_ROWS < int(x.shape[0]) <= _dec8_op.WIDE_ROWS[1]
            and x.dtype == torch.bfloat16
            and x.stride(-1) == 1
            and x.stride(0) % 8 == 0
            and x.data_ptr() % 16 == 0
        )

    def _mxf8_takes(self, name: str, x, w8, wd8=None) -> bool:
        """Whether `_project` sends `x` to the MXF8 GEMM: its runner built, `w8` given, a
        row count above the decode cells', and an activation the quant kernel reads. On a
        `_wide_decode` step only where the call's decode path is MXF8 as well (`wd8` and its
        decode runner: the trunk o_proj) and its wide decode GEMM (`_dec8w_takes`) did not
        build, so decode rows keep that path's numerics -- the same e4m3 operands, block
        scales and K order (B300: the two GEMMs' outputs are bitwise equal at 64 rows)."""
        return (
            name in self._proj_mxf8
            and w8 is not None
            and int(x.shape[0]) > _DECODE_MAX_ROWS
            and (not self._wide_decode or (wd8 is not None and name in self._dec8))
            and x.dtype == torch.bfloat16
            and x.stride(-1) == 1
            and x.stride(0) % 8 == 0
            and x.data_ptr() % 16 == 0
        )

    def _wide_cell(self, call: str, cells: dict, rows: int, at_rows=None):
        """The MTP decode cell for `rows` (above 64) of a `_wide_decode` step: the cell built
        at `rows` (128 / 192), else the smallest cell above `rows` launched on the exact rows
        -- a batch of fewer than 32 sequences (96 / 144 rows at 24 requests, any count in
        65..191 eagerly). A row's output is then bitwise that cell's for it: the cells are
        compiled on dynamic layouts, so the tiles past the last row load zeros and store
        nothing; no padding, no copy. `at_rows(cell, rows)` re-pins a runner pinned to its
        own count (None: the cell's runner takes any count up to its own). Returns None (the
        call's bf16 fallback runs) when no cell is above `rows`, or with `_wide_bucket` off.
        The re-pinned runner is kept per (`call`, rows): built in the forward, it compiles
        nothing, so a first one inside a CUDA-graph capture is safe."""
        cell = cells.get(rows)
        if cell is not None or not self._wide_bucket:
            return cell
        key = (call, rows)
        if key not in self._wide_rag:
            bucket = min((m for m in cells if m > rows), default=None)
            cell = None
            if bucket is not None:
                cell = cells[bucket] if at_rows is None else at_rows(cells[bucket], rows)
            self._wide_rag[key] = cell
        cell = self._wide_rag[key]
        self._count("dec_wide_rows", cell is not None)
        return cell

    def _cell_up(self, rows: int) -> bool:
        """Whether a call's `rows` (at most 64) without a decode cell of their own run on the
        next cell up at their exact rows (`_draft_cell`, the runtime-M indexer split, the next
        eh_proj K-split instance, the fused shared-expert cells at MTP decode counts): an
        eager draft step's (`_draft_eager`), and an MTP3 / MTP5 decode step's
        (`_wide_decode`, `TRTLLM_GLM52_DEC_CELL_UP`). The latter is graph-captured: `rows`
        is fixed per capture shape and the cell is kept per (call, rows), so each graph
        freezes one choice with no host sync. Rows with a cell never ask."""
        return rows <= _DECODE_MAX_ROWS and (
            self._draft_eager or (self._wide_decode and self._dec_cell_up)
        )

    def _draft_cell(self, call: str, rows: int, cells: dict, mtp_cells: dict, at_rows):
        """The cell an eager draft step's or an MTP3 / MTP5 decode step's `rows` (at most 64,
        no cell of their own; see `_cell_up`) run on at their exact rows: the smallest decode
        cell above `rows` up to 64 that launches at a runtime row count, else the smallest
        such MTP cell (128 / 192). `at_rows(cell, rows)` re-pins as in `_wide_cell`; a None
        from it (a cell compiled for its own M only) moves to the next cell up. The
        context-sized cells (1099 rows and up) are never taken: their tiles past the last row
        would each read the weight again. Each row is bitwise the serving cell's for it. None
        (the call's fallback runs) when no cell qualifies. Kept per (`call`, rows); builds
        nothing, so a first call inside a CUDA-graph capture is safe."""
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
        self._count("draft_cell" if self._draft_eager else "dec_cell_up", cell is not None)
        return cell

    def _project(
        self,
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
        `_mxf8_takes` holds (o_proj's bf16 context rows are not written when it does). `wd8`
        is the layer's decode MXF8 weight (q_b / o_proj): decode rows take the decode MXF8
        GEMM; `dec_pre` says its producer wrote the operand (`_dec8_takes` or `_dec8w_takes`
        must hold). On a `_wide_decode` step, rows above 64 keep the numerics of the call's
        decode path: where that is the decode MXF8 GEMM, the wide decode MXF8 GEMM
        (`_dec8w_takes`) or the context MXF8 GEMM (`_mxf8_takes`), bitwise the same output;
        else the projection's MTP decode cell at its row count or the next one up
        (`_wide_cell`), else the bf16 prefill GEMM (or cuBLAS)."""
        rows = int(x.shape[0])
        wide8 = self._dec8w_takes(name, x, wd8)
        if (
            (wide8 or self._dec8_takes(name, x, wd8))
            and out_dtype in (None, torch.bfloat16)
            and (out is None or (out.dtype == torch.bfloat16 and out.is_contiguous()))
        ):
            runner = self._dec8w[name] if wide8 else self._dec8[name]
            if out is None:
                out = torch.empty([rows, runner.n], dtype=torch.bfloat16, device=x.device)
            tag = "dec_proj_mxf8w" if wide8 else "dec_proj_mxf8"
            self._count(f"{tag}_pre" if dec_pre else tag, True)
            return cute_dsl_projection_mxf8_decode(x, wd8, out, runner, pre=dec_pre)
        assert not dec_pre, f"{name}: a producer's decode operand reached another path"
        if self._mxf8_takes(name, x, w8, wd8) and (
            out_dtype in (None, torch.bfloat16)
            and (out is None or (out.dtype == torch.bfloat16 and out.is_contiguous()))
        ):
            mx = self._proj_mxf8[name]
            if out is None:
                out = torch.empty([rows, mx.n], dtype=torch.bfloat16, device=x.device)
            self._count("ctx_proj_mxf8", True)
            if pre is not None:
                self._count("ctx_proj_mxf8_pre", True)
            return cute_dsl_projection_mxf8(x, w8, out, mx, pre=pre)
        assert pre is None, f"{name}: a producer's MXF8 operand reached a non-MXF8 path"
        runner = self._proj.get(name, {}).get(rows)
        if self._wide_decode and rows > _DECODE_MAX_ROWS:
            runner = self._wide_cell(
                f"proj:{name}", self._proj_mtp.get(name, {}), rows, _projection_op.at_rows
            )
            self._count("dec_wide", runner is not None)
        if (
            runner is None
            and self._cell_up(rows)
            and (self._draft_eager or name not in _CELL_UP_KEEPS_CUBLAS)
        ):
            runner = self._draft_cell(
                f"proj:{name}",
                rows,
                self._proj.get(name, {}),
                self._proj_mtp.get(name, {}),
                _projection_op.at_rows,
            )
        prefill = self._proj_pre.get(name)
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
            return self._copy_into(out, res)
        if out is None:
            out = torch.empty([rows, runner.n], dtype=runner.out_dtype, device=x.device)
        return cute_dsl_projection(x, w_colmajor.t(), out, runner)

    def _absorb_cell(self, cells: dict, rows: int):
        """`cells[rows]` (a decode K-B / V-B object by its cells' row counts), else, for a
        count in 9..64 without a cell where `_cell_up` holds (MTP5 at 2 / 3 requests: 12 / 18
        rows; MTP3 at 3: 12), the M >= 16 group's object, which launches at any of them
        (`_absorbed_bmm_op.DECODE_RUNTIME_ROWS`). None otherwise. A dict lookup on host ints:
        frozen per capture shape."""
        cell = cells.get(rows)
        if cell is None and rows in _absorbed_bmm_op.DECODE_RUNTIME_ROWS and self._cell_up(rows):
            cell = cells.get(_DECODE_MAX_ROWS)
            self._count("dec_cell_up", cell is not None)
        return cell

    def _absorbed_bmm(self, kind: str, x_t, w_kn, w_nk, out_t) -> None:
        """An absorbed-MLA BMM ("k_b" or "v_b") into `out_t`: the decode kernel at its row
        counts (2..64, and 65..192 on a `_wide_decode` step: the 128 / 192 cells' object,
        which launches at any of them; 9..64 where `_cell_up` holds: the M >= 16 group's
        object, `_absorb_cell`), the prefill kernel at any other up to 8192, else `bmm_out`,
        which takes the `[H, K, N]` operand instead of `[H, N, K]`."""
        rows = int(x_t.shape[1])
        runner = self._absorb_cell(self._absorb[kind], rows)
        if self._wide_decode and rows > _DECODE_MAX_ROWS:
            runner = self._wide_cell(kind, self._absorb_mtp[kind], rows)
            self._count("dec_wide", runner is not None)
        if runner is None and rows <= _absorbed_bmm_op.PREFILL_MAX_ROWS:
            runner = self._absorb_pre[kind]
        if runner is None:
            bmm_out(x_t, w_kn, out_t)
        else:
            cute_dsl_absorbed_bmm(x_t, w_nk, out_t, runner)

    def _reduce_norm(self, partial, residual, weight):
        """All-reduce, add into `residual` in place, RMS norm: every trunk collective."""
        n = self._lamport.max_rows
        if partial.shape[0] > n:
            # Row slices of a contiguous `residual` are views, so the in-place add still lands.
            return torch.cat(
                [
                    self._reduce_norm(p, r, weight)
                    for p, r in zip(partial.split(n), residual.split(n))
                ]
            )
        self._check_rows(partial)
        out = torch.empty_like(partial)
        return cute_dsl_allreduce_add_rmsnorm(partial, residual, weight, out, self._lamport)

    def _reduce_norm_x8(self, partial, residual, weight, want: bool):
        """`_reduce_norm`, plus the normalized rows' MXF8 operand where the caller's consumer
        reads one (`want`) and the collective runs two-shot (its epilogue writes the operand;
        its bf16 rows are the plain two-shot's bytes): `(out, x8)`, `x8` None otherwise."""
        rows = int(partial.shape[0])
        if (
            want
            and self._lamport.fused_twoshot_x8 is not None
            and _AR_TWOSHOT_ABOVE < rows <= self._lamport.max_rows
            and not torch.cuda.is_current_stream_capturing()
        ):
            self._check_rows(partial)
            out = torch.empty_like(partial)
            n = int(partial.shape[1])
            x8 = (
                torch.empty([rows, n], dtype=torch.uint8, device=partial.device),
                torch.empty(
                    (rows + 127) // 128 * 128 * (n // 32), dtype=torch.uint8, device=partial.device
                ),
            )
            cute_dsl_allreduce_add_rmsnorm(partial, residual, weight, out, self._lamport, x8=x8)
            self._count("ar_x8", True)
            return out, x8
        return self._reduce_norm(partial, residual, weight), None

    def _moe_prequant(self, rows: int, exact: bool) -> bool:
        """Whether a trunk MoE layer's pre-MoE collective also writes the routed experts'
        NVFP4 input (`_reduce_norm_quant`): at a decode width (1..64 rows), and on a wide
        decode step that runs the routed experts on its exact rows (`exact`, see
        `_moe_decode_exact`: MTP3 / MTP5 only) up to `_MOE_PREQUANT_WIDE_MAX`. There the
        routed experts then skip their stage copy and quantize: the collective's bytes are
        their quantize's bit for bit. Every other row count -- context steps, and MTP1 and
        no speculation, whose decode steps stay within 64 rows -- keeps the plain collective
        and the routed experts' own quantize."""
        return rows <= _DECODE_MAX_ROWS or (exact and rows <= _MOE_PREQUANT_WIDE_MAX)

    def _reduce_norm_quant(self, partial, residual, weight, g1, rows, exact=False):
        """`_reduce_norm`, plus the routed experts' NVFP4 input where `_moe_prequant` says
        the collective writes it: into the layer's `xq` / `xsf`, exactly the rows the routed
        experts would quantize. Returns `(out, prequantized)`."""
        if not self._moe_prequant(rows, exact):
            return self._reduce_norm(partial, residual, weight), False
        self._check_rows(partial)
        xq, xsf = self._moe_xq[:rows], self._moe_xsf[:rows]
        out = torch.empty_like(partial)
        out = cute_dsl_allreduce_add_rmsnorm_quant_nvfp4(
            partial, residual, weight, g1, out, xq, xsf, self._lamport
        )
        return out, True

    def _mla_rope_handle(
        self, width: int, dev: int, p: int = _MLA_ROPE_MTP_WIDTH
    ) -> "_mla_rope_op.CompiledMlaRopeGeneration":
        """The MLA rope kernels for a block table `width` columns wide and `p` new tokens per
        sequence (one of `mtp_widths`, or 1 for MTP3 / MTP5's 1-token draft steps,
        `_mtp_rope_specialised`). The operand layout is this target's: `q_pe` the rope
        half of the `[T, heads, qk_dim]` q_b output, the latent the tail of the
        `[T, q_lora + lat_dim]` kv_a output. A pair not seen yet compiles here, which is the
        first eager forward that uses it (the trunk's are built in `_derive_first_step`).
        Both q-base variants are built: a step whose rows a decode K-B cell serves (2..64)
        takes the q-base from that GEMM, any wider one (MTP3 / MTP5's 128 / 192 rows: the
        prefill K-B) has the rope kernel emit it."""
        handle = self._mla_rope_compiled.get((width, p))
        if handle is None:
            handle = _mla_rope_op.compile(
                max_blocks_per_seq=width,
                q_pe_head_stride=self.qk_dim,
                q_pe_token_stride=self.heads * self.qk_dim,
                latent_row_stride=self.q_lora + self.lat_dim,
                device_index=dev,
                predicted_tokens_per_seq=p,
                q_base_from_kb=bool(self._kb8),
            )
            self._mla_rope_compiled[(width, p)] = handle
        return handle

    def _mtp_rope_specialised(self, gen_p: int) -> bool:
        """Whether the MTP layer's decode rows of `gen_p` per sequence take the specialised
        MLA rope kernel (`_attn_gen`): at `mtp_widths`, as the trunk; and on an engine
        drafting more than one token also on its 1-token draft steps (draft steps
        1..max_draft_len-1, `gen_p` 1), where the stock op needed the strided latent packed
        first (a `[rows, 576]` copy per draft step). The kernel writes the same bytes as the
        stock op there (query, latent row, prefix arrays, scheduler counter, FMHA scales);
        the K-B then takes its decode fp8 q-base cell, as at MTP1's width 2."""
        return gen_p in self.mtp_widths or (gen_p == 1 and self.max_draft_len > 1)

    def _qkv(
        self,
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
        projection's output the caller already wrote (`_qkva_ix_fold`), else it runs here."""
        if qkva is None:
            qkva = self._project("qkva", x, a.qkva, w8=a.qkva8, pre=x8)
        q_a_raw, ckv_raw, _ = torch.split(qkva, [self.q_lora, self.kv_lora, self.rope_dim], -1)
        if (
            q8_rows
            and self._qkv_a_norm8 is not None
            and int(q_a_raw.shape[0]) >= _qkv_a_norm_op.WARP_ROW_MIN_ROWS
            and not torch.cuda.is_current_stream_capturing()
            and _qkv_a_norm8_op.takes(q_a_raw, a.q_norm, ckv_raw, a.kv_norm)
        ):
            q8 = _qkv_a_norm8_op.operand(q8_rows, x.device)
            q_a = cute_dsl_qkv_a_rmsnorm_q8(
                q_a_raw, a.q_norm, ckv_raw, a.kv_norm, self.eps, *q8, self._qkv_a_norm8,
                q_store_from=q_store_from,
            )  # fmt: skip
            self._count("qkv_a_q8", True)
            if q_store_from:
                self._count("qkv_a_q8_skip", True)
                if self._qkv_a_norm8.ilp:
                    self._count("norm_ilp", True)
            latent = torch.split(qkva, [self.q_lora, self.lat_dim], -1)[1]
            return None, q_a, latent, q8
        # Decode rows: the norm writes q_a's MXF8 bytes into the decode q_b GEMM's own operand
        # and releases that GEMM at entry, so q_b runs without its activation quant.
        rows = int(q_a_raw.shape[0])
        if (
            q_b
            and rows <= _DECODE_MAX_ROWS
            and self._qkv_a_norm8 is not None
            and a.qb_dec8 is not None
            and _qkv_a_norm8_op.takes(q_a_raw, a.q_norm, ckv_raw, a.kv_norm)
        ):
            xq, sf = self._dec8["q_b"].operand
            q_a = cute_dsl_qkv_a_rmsnorm_q8(
                q_a_raw, a.q_norm, ckv_raw, a.kv_norm, self.eps, xq[:rows], sf, self._qkv_a_norm8
            )
            if self._dec8_takes("q_b", q_a, a.qb_dec8):
                latent = torch.split(qkva, [self.q_lora, self.lat_dim], -1)[1]
                q = self._project("q_b", q_a, a.qb, wd8=a.qb_dec8, dec_pre=True)
                return q, q_a, latent, None
            latent = torch.split(qkva, [self.q_lora, self.lat_dim], -1)[1]
            return self._project("q_b", q_a, a.qb, wd8=a.qb_dec8), q_a, latent, None
        # Both kernels are this op's; the counter shows which one a step's row count took.
        wr = _qkv_a_norm_op.takes_warp_row(q_a_raw, a.q_norm, ckv_raw, a.kv_norm)
        self._count("qkv_a_warp_row" if wr else "qkv_a_cta", True)
        # Eager calls on the in-place layout skip the kernel module's per-call checks.
        if _qkv_a_norm_op.takes_direct(q_a_raw, a.q_norm, ckv_raw, a.kv_norm):
            self._count("qkv_a_direct", True)
        q_a, _ = cute_dsl_qkv_a_rmsnorm(
            q_a_raw, a.q_norm, ckv_raw, a.kv_norm, self.eps, self._qkv_a_norm
        )
        latent = torch.split(qkva, [self.q_lora, self.lat_dim], -1)[1]
        q = self._project("q_b", q_a, a.qb, wd8=a.qb_dec8) if q_b else None
        return q, q_a, latent, None

    def _project_tail(self, name: str, x, start: int, w_colmajor):
        """The projection of `x[start:]`, a mixed step's generation rows after its context
        rows. Their count is any 1..63, which the decode buckets rarely match, and a miss
        goes to cuBLAS; so the smallest compiled bucket `b` above the count runs on the
        trailing `b` rows of `x` (the last `b - rows` context rows ride along, their outputs
        unused) and the result is the view of its last rows: no padding copy."""
        total = int(x.shape[0])
        rows = total - start
        runners = self._proj.get(name, {})
        if rows <= _DECODE_MAX_ROWS and rows not in runners:
            bucket = min((m for m in runners if m >= rows), default=None)
            if bucket is not None and bucket <= total:
                self._count(f"{name}_tail", True)
                return self._project(name, x[total - bucket :], w_colmajor)[bucket - rows :]
        return self._project(name, x[start:], w_colmajor)

    def _q_b_e4m3(self, q_a_ctx, a, pre=None):
        """The fp8 context attention's query: `q_a x (q_b / s_q)` in e4m3, `[T, heads *
        qk_dim]`, on the MXF8 GEMM or the prefill GEMM's e4m3 entry (`_build_context_mha_fp8`).
        `pre` is q_a's MXF8 operand written by the norm (`_qkv`'s `q8`)."""
        rows = int(q_a_ctx.shape[0])
        q8 = torch.empty(
            [rows, self.heads * self.qk_dim], dtype=torch.float8_e4m3fn, device=q_a_ctx.device
        )
        if isinstance(self._q8, _mxf8_op.Mxf8ProjectionRunner):
            return cute_dsl_projection_mxf8(q_a_ctx, a.qb8, q8, self._q8, pre=pre)
        assert pre is None, "the q_a norm's MXF8 operand needs the MXF8 q_b"
        return cute_dsl_projection(q_a_ctx, a.qb8, q8, self._q8)

    def _topk_buf(self, md, rows: int, tc: int, skip_ctx: bool, skip_gen: bool):
        """Where this step's selection lands. A phase the engine marked skippable (every
        sequence shorter than index_topk) takes the dense causal index the engine wrote; a
        branch frozen into a graph, safe because short and long steps capture separately."""
        if skip_ctx and skip_gen:
            return md.topk_indices_buffer[:rows]
        # `topk_indices_buffer` exists only while the engine's indexer skip is on.
        buf = torch.empty(
            [rows, self.idx_topk], dtype=torch.int32, device=torch.cuda.current_device()
        )
        if skip_ctx and tc:
            self._copy_into(buf[:tc], md.topk_indices_buffer[:tc])
        if skip_gen and rows > tc:
            self._copy_into(buf[tc:], md.topk_indices_buffer[tc:rows])
        return buf

    def _qkva_ix_fold(self, x, i: int, rows: int):
        """`(qkva, key, head_weights)` of indexer layer `i` from one GEMM (the qkva projection
        with the indexer's key / head-weight rows folded in, `cute_dsl_qkva_ix_fold`), or None
        where it does not serve and the two GEMMs run: off (`TRTLLM_GLM52_WIDE_IX_FOLD=0`), an
        MTP1 engine, a step that is not a wide decode step (rows above 64 on an MTP3 / MTP5
        engine), or rows without a qkva MTP cell. Where it serves, `qkva` is bitwise the qkva
        MTP cell's (`_project` on these rows) and the indexer columns accumulate as the
        split-output projection's (its chains and reduction order); `key` / `head_weights` are
        views of the cell's side buffers, consumed by this layer's indexer before the next
        layer's call."""
        w = self._ix_fold_w.get(i)
        if (
            w is None
            or not self._qkva_ix_mtp
            or not self._wide_decode
            or rows <= _DECODE_MAX_ROWS
            or x.dtype != torch.bfloat16
            or not x.is_contiguous()
        ):
            return None
        runner = self._wide_cell("qkva_ix_fold", self._qkva_ix_mtp, rows, _ix_fold_op.at_rows)
        self._count("ix_fold", runner is not None)
        if runner is None:
            return None
        out = torch.empty([rows, _ix_fold_op.QKVA_N], dtype=x.dtype, device=x.device)
        return cute_dsl_qkva_ix_fold(x, w, out, runner)

    def _qb_ixqb_runner(self, i: int, rows: int):
        """The q_b | wq_b cell for indexer layer `i` at `rows` (`_qb_ixqb`), or None where it
        does not serve and the two GEMMs run: off (`TRTLLM_GLM52_WIDE_QB_IXQB=0`), an MTP1
        engine, a step that is not a wide decode step (rows above 64 on an MTP3 / MTP5
        engine), or rows without an MTP cell. Counted as `qb_ixqb`."""
        if i not in self._qbix_w or not self._qbix_mtp:
            return None
        if not self._wide_decode or rows <= _DECODE_MAX_ROWS:
            return None
        runner = self._wide_cell("qb_ixqb", self._qbix_mtp, rows, _qbix_op.at_rows)
        self._count("qb_ixqb", runner is not None)
        return runner

    def _qb_ixqb(self, q_a, i: int, runner):
        """`(q, ix_q)` of layer `i` from one GEMM: bitwise the q_b and wq_b MTP cells' outputs
        (`cute_dsl_q_b_ix_wq_b`); `q` `[rows, 4096]` and `ix_q` contiguous halves of one
        buffer."""
        rows = int(q_a.shape[0])
        out2 = torch.empty([2, rows, _qbix_op.N], dtype=q_a.dtype, device=q_a.device)
        return cute_dsl_q_b_ix_wq_b(q_a, self._qbix_w[i], out2, runner)

    def _indexer_project(
        self, x, q_a, pos, ix, rows: int, need_q: bool, kw=None, ixq=None, kc=None
    ):
        """The indexer's front: `(q_fp8, weights, k_pre)`; `q_fp8` and `weights` are None on a
        step where every phase takes the engine's index. The k side always runs: the keys this
        step appends are read by later steps. `kw`: the pre-norm keys and head weights the
        caller already projected (`_qkva_ix_fold`), else the split-output projection runs;
        `ixq`: the indexer's q projection likewise (`_qb_ixqb`). `kc`: `(md, k_pool)`, the
        layer's K cache: where `_ix_qk_handle` has a launcher the q side and the K-cache
        update run as one launch here, and `k_pre` is None (the keys are written)."""
        hi, di = self.idx_heads, self.idx_dim
        dev = x.device
        qside = self._ix_qside_compiled
        if kw is not None:
            kb, wq = kw
        else:
            split = self._ix_pre_split.get(rows)
            if self._wide_decode and rows > _DECODE_MAX_ROWS:
                # Rows without a handle of their own take the runtime-M handle (the same kernel).
                rag = self._ix_pre_split_rag
                if rag is not None and rows > rag.max_m:
                    rag = None
                split = self._wide_cell("ix_split", self._ix_pre_split_mtp, rows, lambda h, r: rag)
                self._count("dec_wide", split is not None)
            elif split is None and self._cell_up(rows):
                # An eager draft step's or an MTP3 / MTP5 decode step's count without a
                # handle: the runtime-M handle (the same kernel) instead of `x.float()` and
                # the fp32 cuBLAS projection, its split-K reduce and two copies.
                rag = self._ix_pre_split_rag
                split = rag if rag is not None and rows <= rag.max_m else None
                self._count("draft_cell" if self._draft_eager else "dec_cell_up", split is not None)
            pre = self._ix_pre_split_pre
            if (
                split is None
                and pre is not None
                and _DECODE_MAX_ROWS < rows <= pre.max_rows
                and x.is_contiguous()
            ):
                split = pre
            if split is not None and split is pre and self._ix_nosplit:
                # fp32 column views of one [rows, 160] output (`TRTLLM_GLM52_CTX_IX_NOSPLIT`).
                kb, wq = _ix_pre_split_op.project_unsplit(x, ix.fused_bf16, pre)
                qside = self._ix_qside_compiled_p160
                self._count("ctx_ix_nosplit", True)
            elif split is not None:
                kb = torch.empty([rows, di], dtype=x.dtype, device=dev)
                wq = torch.empty([rows, hi], dtype=torch.float32, device=dev)
                cute_dsl_indexer_pre_proj_split(x, ix.fused_bf16, kb, wq, split)
            else:
                kb = torch.empty([rows, di], dtype=x.dtype, device=dev)
                # fp32: the only `weights` dtype fp8_mqa_logits takes.
                fused = self._project("ix_fused", x.float(), ix.fused, torch.float32)
                k_raw, wq_raw = torch.split(fused, [di, hi], -1)
                self._copy_into(kb, k_raw)
                # The scorers check `weights`' stride(1) but not stride(0), then read it
                # densely: a strided slice scores each row with another row's head weights.
                wq = wq_raw.contiguous()
        k_pre = (kb, ix.kn_w, ix.kn_b, pos)
        if not need_q:
            return None, None, k_pre
        if ixq is None:
            ixq = self._project("ix_qb", q_a, ix.qb)
        q = torch.reshape(ixq, [rows, hi, di])
        q_fp8 = torch.empty([rows, hi, di], dtype=torch.float8_e4m3fn, device=dev)
        weights = torch.empty([rows, hi], dtype=torch.float32, device=dev)
        qk = self._ix_qk_handle(kc, kb, rows)
        if qk is not None:
            md, k_pool = kc
            self._count("ix_qk", True)
            cute_dsl_indexer_q_side_k_cache(
                q, pos, self._idx_cos_sin, wq, self.idx_weight_scale, q_fp8, weights, kb,
                ix.kn_w, ix.kn_b, k_pool, md.slot_mapping_fp8, md.slot_mapping_scale,
                is_neox=self.indexer_is_neox, compiled=qk,
            )  # fmt: skip
            return q_fp8, weights, None
        cute_dsl_indexer_q_side_fused(
            q, pos, self._idx_cos_sin, wq, self.idx_weight_scale, q_fp8, weights,
            is_neox=self.indexer_is_neox, compiled=qside,
        )  # fmt: skip
        return q_fp8, weights, k_pre

    def _ix_qk_handle(self, kc, kb: torch.Tensor, rows: int):
        """The q side + K-cache update launcher for this call (`TRTLLM_GLM52_WIDE_IX_QK`), or
        None: only with the layer's cache (`kc`) on an MTP3 / MTP5 wide decode step (rows
        above 64), once the K update's first eager call has validated the slot mappings
        (`_indexer_write_k`), with 8-byte aligned contiguous keys (its only key loads)."""
        if kc is None or not self._ix_qk_compiled or not self._idx_k_checked:
            return None
        if not (self._wide_decode and rows > _DECODE_MAX_ROWS):
            return None
        if not kb.is_contiguous() or kb.data_ptr() % 8:
            return None
        return self._ix_qk_compiled[kc[1].is_contiguous()]

    def _indexer_write_k(self, md, k_pool, k_pre, rows: int) -> None:
        """Norm, rope, fp8-pack and append this step's keys in one launch."""
        kb, kn_w, kn_b, pos = k_pre
        cute_dsl_indexer_k_cache_update(
            kb,
            kn_w,
            kn_b,
            pos,
            self._idx_cos_sin.view(-1, 2, self.rope_dim // 2),
            k_pool,
            md.slot_mapping_fp8,
            md.slot_mapping_scale,
            rows,
            is_neox=self.indexer_is_neox,
            compiled=self._ix_k_compiled[k_pool.is_contiguous()],
            # Its validation reduces the slot mappings to a host bool, a sync that invalidates
            # a capture: checked once, on the first eager call.
            validate=not self._idx_k_checked,
        )
        self._idx_k_checked = True

    def _indexer_decode(
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
        # Cached plus all `gen_p` of this step's tokens; both consumers take the same vector,
        # which keeps the selector's causal bound inside the columns the scorer wrote.
        kv_lens = md.kv_lens_cuda_runtime[ctx : ctx + seqs]
        atom = _MQA_ATOM_SPLIT.get(gen_p)
        if atom is None:
            logits = cute_dsl_fp8_paged_mqa_logits(
                torch.reshape(q_fp8, [seqs, gen_p, self.idx_heads, self.idx_dim]),
                k_pool,
                weights,
                kv_lens,
                md.indexer_k_cache_block_offsets[ctx : ctx + seqs],
                md.scheduler_metadata_buffer,
                self._max_ctx_len,
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
            self._count("mqa_atom_split", True)
            logits = cute_dsl_fp8_paged_mqa_logits(
                torch.reshape(q_fp8, [seqs * factor, atom, self.idx_heads, self.idx_dim]),
                k_pool,
                weights,
                *meta,
                self._max_ctx_len,
            )
        # The stock engine's GVR decode top-k where the engine's sparse config asks for it.
        gvr = self._dec_topk_gvr_live
        if gvr and self._dec_topk_gvr_launch(md, logits, kv_lens, topk_buf, gen_p):
            return
        compiled = self._ix_topk_dec_compiled.get(gen_p)
        # Ours, or (a width without a handle) the trtllm op.
        self._count("ix_topk_dec", compiled is not None)
        if compiled is None:
            indexer_topk_decode(
                logits, kv_lens, topk_buf, gen_p, self.idx_topk,
                radix_aux_indices=md.radix_aux_indices, radix_aux_logits=md.radix_aux_logits,
            )  # fmt: skip
        else:
            # `seqs * gen_p` logits rows on both scorer paths (the atom split reshapes rows only).
            self._count("ix_topk_dec_cluster", ix_topk_decode_uses_cluster(compiled, seqs * gen_p))
            cute_dsl_indexer_topk_decode(
                logits, kv_lens, topk_buf, top_k=self.idx_topk, next_n=gen_p, compiled=compiled
            )

    def _dec_topk_gvr_launch(self, md, logits, kv_lens, topk_buf, gen_p: int) -> bool:
        """The generation rows' top-k on the stock engine's self-sampling GVR engine, the call
        the stock DSA indexer makes (`Indexer.sparse_attn_indexer`'s decode `self.top_k(...)`).
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
        try:
            self._dec_topk_gvr(
                logits,
                topk_buf,
                is_prefill=False,
                sequence_lengths=kv_lens,
                scan_lengths=kv_lens,
                next_n=gen_p,
                max_seq_len=self._max_ctx_len,
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
        self._count("ix_topk_dec_gvr", True)
        return True

    def _fmha_dec_handle(self, gen_p: int) -> "_fmha_dec_op.CompiledSparseMlaDecode":
        """The decode attention's handle for `gen_p` rows per sequence: the width-2 grid up to
        2 (MTP1, no speculation, MTP3 / MTP5's 1-row draft steps), the MTP3 / MTP5 grid above."""
        if gen_p <= self._fmha_dec_compiled.max_sq:
            return self._fmha_dec_compiled
        return self._fmha_dec_wide

    def _attn_gen(
        self,
        a,
        layer: int,
        q,
        latent,
        out,
        slots,
        gen: int,
        gen_p: int,
        ctx: int,
        md,
        slot_kind,
        slot_layer,
        specialised_rope: bool,
        o8_side: bool = False,
    ) -> bool:
        """Sparse MLA over the generation rows, absorbed form, into `out`. With `o8_side`
        (a pure decode step whose o_proj takes the decode MXF8 GEMM next, or the wide one
        above 64 rows) the V-B GEMM also writes `out` as that GEMM's operand where a
        side-output cell serves the row count (above 64: where `_wide_cell` gives the plain
        MTP cell, so the bf16 `out` is that cell's); returns whether it did."""
        heads, dev, dt = self.heads, q.device, q.dtype
        q_nope, q_pe = torch.split(
            torch.reshape(q, [gen, heads, self.qk_dim]), [self.nope, self.rope_dim], -1
        )
        fused_q = torch.empty([gen, heads, self.lat_dim], dtype=dt, device=dev)
        fq_nope = torch.split(fused_q, [self.kv_lora, self.rope_dim], -1)[0]
        quant_q = torch.empty([gen, heads, self.lat_dim], dtype=torch.float8_e4m3fn, device=dev)
        # On the specialised rope a decode cell's K-B writes the e4m3 q-base into quant_q
        # itself (the bytes the rope kernel would convert) and releases the rope kernel at
        # once: the rope skips the q-base and waits for the K-B only at its end, so the two
        # run concurrently and the decode MLA's wait on the rope still covers the K-B.
        kb8 = self._absorb_cell(self._kb8, gen) if specialised_rope else None
        if specialised_rope and self._wide_decode and gen > _DECODE_MAX_ROWS and self._kb8_mtp:
            # MTP3 / MTP5 decode rows: the MTP cells' fp8 q-base K-B, at its cell or on the
            # next one up at the exact rows (the rope handles of widths 4 / 6 carry the
            # `q_base_from_kb` variants whenever the decode K-B built).
            kb8 = self._wide_cell("k_b_fp8", self._kb8_mtp, gen)
        if kb8 is not None:
            q_base = torch.split(quant_q.view(torch.uint8), [self.kv_lora, self.rope_dim], -1)[0]
            cute_dsl_absorbed_bmm(
                torch.transpose(q_nope, 0, 1), a.k_b_nk, torch.transpose(q_base, 0, 1), kb8
            )
            self._count("kb_fp8_qbase", True)
        else:
            # Over an fp8 pool the rope kernel reads this half to build the quantized query,
            # so the BMM is issued first on the same stream.
            self._absorbed_bmm(
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
        rope = self._rope
        if specialised_rope:
            if gen_p == 1:
                self._count("mla_rope_draft1", True)
            elif gen_p != _MLA_ROPE_MTP_WIDTH:
                self._count("mla_rope_wide", True)
            # The specialised kernel takes the stock op's earlier positional ABI.
            cute_dsl_mla_rope_generation(
                fused_q, q_pe, latent, rope["rotary_cos_sin"], cu_q, cu_kv, counter, bmm1, bmm2,
                quant_q, md.kv_lens_cuda_runtime, md.kv_lens_runtime, md.prompt_lens_cpu_runtime,
                ctx, md.kv_cache_block_offsets, md.host_kv_cache_pool_pointers,
                md.host_kv_cache_pool_mapping, None, None, None, None, [None, None], gen_p, layer,
                heads, 1, self.lat_dim, md.tokens_per_block, md.max_seq_len, 1, self.quant_mode,
                1.0, self.q_lora, self.kv_lora, self.nope, self.rope_dim, self.kv_lora, True,
                compiled=self._mla_rope_handle(
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
                head_size=self.lat_dim,
                residual_dim=_KV_RESIDUAL_DIM,
                tokens_per_block=md.tokens_per_block,
                attention_window_size=md.max_seq_len,
                beam_width=1,
                quant_mode=self.quant_mode,
                q_scaling=1.0,
                q_lora_rank=self.q_lora,
                kv_lora_rank=self.kv_lora,
                qk_nope_head_dim=self.nope,
                qk_rope_head_dim=self.rope_dim,
                v_head_dim=self.kv_lora,
                rope_append=True,
            )
        lat_out = torch.empty([gen, heads * self.kv_lora], dtype=dt, device=dev)
        seqs = gen // gen_p
        if slot_kind == _UNION_POSITIONS:
            self._count("fmha_dec_union", True)
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
                self._stride_factor,
                compiled=self._fmha_union,
            )
        else:
            fmha = self._fmha_dec_handle(gen_p)
            if fmha is self._fmha_dec_wide:
                self._count("fmha_dec_wide", True)
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
        lat_t = torch.transpose(torch.reshape(lat_out, [gen, heads, self.kv_lora]), 0, 1)
        out_t = torch.transpose(torch.reshape(out, [gen, heads, self.v_dim]), 0, 1)
        vb8 = None
        if o8_side and gen <= _DECODE_MAX_ROWS:
            # 9..64 rows without a cell on an MTP3 / MTP5 decode step too (`_absorb_cell`):
            # o_proj then skips its own act-quant there as well.
            vb8 = self._absorb_cell(self._vb8, gen)
        elif o8_side and self._vb8w:
            vb8 = self._wide_cell("v_b_x8", self._vb8w, gen)
        if vb8 is not None:
            cute_dsl_absorbed_bmm(lat_t, a.v_b_nk, out_t, vb8)
            return True
        self._absorbed_bmm("v_b", lat_t, a.v_b_t, a.v_b_nk, out_t)
        return False

    def _dense_mlp(self, o, gu, dn, w8=None, x8=None):
        """A dense layer's MLP, this rank's partial. Context rows on the MXF8 trio where it
        built (`w8`: the layer's quantized gate|up and down; `x8`: `o`'s MXF8 operand from
        the collective, else gate|up quantizes `o` itself): the SwiGLU writes the down GEMM's
        operand, no bf16 intermediate. Decode rows: the bf16 GEMMs and SwiGLU."""
        if w8 is not None and self._mxf8_takes("dense_gate_up", o, w8[0]):
            rows = int(o.shape[0])
            if len(w8) > 2:
                # The SwiGLU into the down GEMM's operand in the gate|up GEMM's epilogue.
                act8 = cute_dsl_dense_up_swiglu_mxf8(
                    o, w8[2], *_dense_up8_op.operand(rows, o.device), self._dense_sw8, pre=x8
                )
                self._count("dense_up_swiglu", True)
            else:
                gate_up = self._project("dense_gate_up", o, gu, w8=w8[0], pre=x8)
                act8 = cute_dsl_swiglu_mxf8(
                    gate_up,
                    *_swiglu8_op.operand(rows, gate_up.shape[1] // 2, o.device),
                    self._swiglu8,
                )
            down = self._proj_mxf8["dense_down"]
            out = torch.empty([rows, down.n], dtype=torch.bfloat16, device=o.device)
            self._count("dense_mxf8", True)
            return cute_dsl_projection_mxf8(act8[0], w8[1], out, down, pre=act8)
        act = cute_dsl_dense_swiglu(self._project("dense_gate_up", o, gu), self._dense_swiglu)
        return self._project("dense_down", act, dn)

    def _shared_after_collective(self) -> None:
        """On an MTP3 / MTP5 wide decode step (`_moe_decode_exact`), called on the side
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
        same. MTP1 and context steps never call it."""
        fence = getattr(self, "_shared_order_fence", None)
        if fence is None:
            # Allocated on the first (eager, warmup) call, then reused: only the launch
            # matters, so one persistent element serves every layer and step.
            fence = self._shared_order_fence = torch.zeros(
                1, dtype=torch.int32, device=torch.cuda.current_device()
            )
        else:
            fence.zero_()
        self._count("moe_shared_after_ar", True)

    def _shared_mlp(self, x, gu, gu_il, dn, out=None, trunk=False, x8=None, w8=None):
        """The shared expert, this rank's slice, a partial sum. `gu_il` is this layer's
        interleaved gate/up weight; `out` is the routed experts' output buffer when FC2 will
        accumulate onto it. A trunk layer's context rows take the prefill shared kernels
        when they built; the MTP layer's keep the two-launch path. With `x8` (`x`'s MXF8
        operand from the pre-MoE collective) and `w8` (the layer's MXF8 gate|up) the prefill
        gate|up + SwiGLU runs as the MXF8 GEMM with the SwiGLU in its epilogue. A
        `_wide_decode` step's rows above 64 take the MTP decode cells, trunk and MTP layer:
        at 128 / 192, else on the next one up (`_wide_cell`; the gate|up cells take any row
        count, the down cell is re-pinned). Rows up to 64 without a fused cell where
        `_cell_up` holds: the fused cell built at that count (`_SHARED_FUSED_UP_SEQS`), else
        the two-launch path on the projection cells next up."""
        rows = int(x.shape[0])
        k = self._moe_k
        wide = None
        if self._wide_decode and rows > _DECODE_MAX_ROWS:
            wide = self._wide_cell("shared_gu", self._shared_gu_swiglu_mtp, rows)
            self._count("dec_wide", wide is not None)
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
                mid = self._moe_mid[:rows]
                if x8 is not None and w8 is not None:
                    cute_dsl_shared_up_swiglu_mxf8(xs, w8, mid, self._shared8, pre=x8)
                    self._count("shared_mxf8", True)
                else:
                    cute_dsl_shared_gate_up_swiglu(xs, gu_il, mid, k.shared_up_pre)
                return cute_dsl_projection(mid, dn.t(), res, k.shared_down_pre)
        runner = self._shared_gu_swiglu.get(rows) if wide is None else wide
        if runner is None and self._shared_gu_swiglu_up and self._cell_up(rows):
            # An MTP3 / MTP5 decode count of 1..3 sequences: its own fused cell.
            runner = self._shared_gu_swiglu_up.get(rows)
            self._count("dec_cell_up", runner is not None)
        if runner is None:
            gate_up = self._project("shared_gate_up", x, gu)
            swiglu = self._rows.swiglu
            hit = swiglu is not None and _swiglu_rows_op.takes(gate_up)
            self._count("swiglu_rows", hit)
            if hit:
                act = cute_dsl_swiglu_rows(gate_up, swiglu)
            else:
                act = flashinfer_silu_and_mul(gate_up)
        else:
            act = torch.empty([rows, gu.shape[1] // 2], dtype=x.dtype, device=x.device)
            cute_dsl_shared_gate_up_swiglu(x, gu_il, act, runner)
        return self._project("shared_down", act, dn, out=out)

    # ---- the trunk's routed experts ----

    def _moe_pair_ok(self, rows: int) -> bool:
        """Whether `rows` takes the prefill grouped GEMM pair: a context row count above the
        decode kernels' 64."""
        return self._moe_k.pair is not None and _MOE_PREFILL_MIN <= rows <= _MOE_MAX_T

    def _moe_decode_exact(self, rows: int) -> bool:
        """Whether a decode step's routed experts run `rows` unpadded below 513: the MTP3 /
        MTP5 decode widths (`max_draft_len > 1`; 65..192 rows at batch 32), on the prefill
        front end and pair. The prefill top-k writes the maps for any row count, so nothing
        is padded, re-routed, cleared or merged: 192 rows ran as 256 (a third more M, the
        top-k's maps discarded for a routing launch, a clear and a merge). MTP1 and no
        speculation never ask (their decode steps stay within 64 rows); context steps keep
        the power-of-two padding."""
        return (
            self.max_draft_len > 1
            and _MOE_PREFILL_MIN <= rows < _MOE_PAIR_EXACT_FROM
            and self._moe_k.topk_pre is not None
            and self._moe_pair_ok(rows)
        )

    def _moe_topk_wide(self, rows: int):
        """The wide decode top-k handle for `rows` (65..256), or None (not built: MTP1, no
        speculation, or a failed build; or rows outside its domain)."""
        if not _moe_topk_op.WIDE_MIN_ROWS <= rows <= _moe_topk_op.WIDE_MAX_ROWS:
            return None
        return self._moe_k.topk_wide.get(_moe_topk_op.wide_range(rows))

    def _moe_padded_rows(self, rows: int, exact: bool = False) -> int:
        """The row count routing and the grouped GEMMs see for `rows`: a power of two, or the
        exact count on the prefill pair from `_MOE_PAIR_EXACT_FROM` up, or on a wide decode
        step (`exact`, see `_moe_decode_exact`). `_moe_begin_layer` and `_moe_experts` both
        ask here, so the shared expert's rows and FC2's agree."""
        if (exact or rows >= _MOE_PAIR_EXACT_FROM) and self._moe_pair_ok(rows):
            return rows
        return _pow2(rows)

    def _moe_dec_unpadded(self, rows: int) -> bool:
        """Whether a trunk MoE layer's decode width `rows` (1..64) runs unpadded on the fused
        top-k's maps (`TRTLLM_GLM52_MOE_DEC_UNPAD`). Only the trunk asks: there the collective
        writes the NVFP4 input (`_moe_prequant`) and the shared expert the output's partial
        (`_moe_begin_layer`), so no per-width quantize or clear runs (both are compiled for
        the powers of two only). The MTP layer quantizes and clears itself and keeps the
        padding. Every decode width has its own top-k handle (`k.topk`, `k.topk_rep`), whose
        maps the grouped GEMMs take at that width: they are compiled for the capacity and
        read the live tile count from the maps."""
        return self._moe_dec_unpad and 1 <= rows <= _DECODE_MAX_ROWS and rows in self._moe_k.topk

    def _moe_begin_layer(self, rows: int, exact: bool = False):
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
        the shared expert writes on the side stream, which FC2 joins (`_moe_experts`), and
        the slot alternates per layer as at every other width. What a padded width adds
        (the staging copies, the routing kernel, the top-k's late FC1 release) does not
        touch the output."""
        self._moe_slot ^= 1
        if rows > _MOE_MAX_T:
            return None
        if self._moe_padded_rows(rows, exact) != rows and not self._moe_dec_unpadded(rows):
            if rows > _DECODE_MAX_ROWS or not self._moe_acc_padded:
                return None
            self._count("moe_acc_padded", True)
        return self._moe_slots[self._moe_slot].out[:rows]

    def _moe_router(self, x, m, slot, exact: bool = False):
        """Router logits in the slot's buffer, or None (the stock projection then runs it).
        Up to 64 rows on the decode router, where above 5 rows these are `[rows, 4, 256]`
        partials only the fused top-k reads; on a wide decode step (`exact`) the same
        partials from the wide decode router when it and the wide top-k built; above, on the
        prefill router (`[rows, 256]`)."""
        rows = int(x.shape[0])
        s = self._moe_slots[slot]
        if rows > _DECODE_MAX_ROWS:
            k = self._moe_k
            pre = k.router_pre
            wide = exact and k.router_wide is not None and self._moe_topk_wide(rows) is not None
            if (pre is None and not wide) or rows > _MOE_MAX_T:
                return None
            if not _direct_ok(x, torch.bfloat16):
                x = x.contiguous()
            if x.dtype is not torch.bfloat16 or x.data_ptr() % 16:
                return None
            if wide:
                out = s.router_wide_out[:rows]
                cute_dsl_moe_router(x, m.router, out, k.router_wide)
                self._count("moe_router_wide", True)
                return out
            out = s.logits[:rows]
            cute_dsl_moe_router(x, m.router, out, pre)
            return out.view(rows, -1)
        if _direct_ok(x, torch.bfloat16):
            xs = x
        else:
            xs = s.router_x[:rows]
            self._copy_into(xs, x)
        out = (
            s.router_out[:rows]
            if rows <= _moe_router_op.SIMT_MAX_ROWS
            else s.router_partials[:rows]
        )
        cute_dsl_moe_router(xs, m.router, out, self._moe_k.router[rows])
        return out

    def _build_moe_wide_rep(self, k, dev) -> None:
        """The MTP3 / MTP5 wide decode steps' expert replicas: `k.topk_wide_rep`, one handle per
        wide row range (keyed as `k.topk_wide`).

        At 65..256 decode rows the prefill pair streams one expert's weights per active local
        expert, so a rank's routed time follows its count of distinct active experts and the
        post-MoE all-reduce waits for the busiest rank (real routing, 75 layers: 2.66 / 2.09
        experts above the mean at 128 / 192 rows). The wide replica top-k moves each active hot
        expert whole to the EP rank with the fewest active experts (the decode replicas'
        greedy), on the context replicas' slots: the `64 + R` deep prefill pair (`k.pair_rep`)
        and each trunk layer's hot ids (`_ctx_hot_ids`); the MTP layer has none and keeps the
        wide top-k. Built only with the wide top-k (MTP3 / MTP5), the context replicas and
        `wide_dec_replicas`, and only for the ranges up to `_WIDE_REP_MAX_ROWS` (65..128); a
        failed build leaves every step on the wide top-k."""
        k.topk_wide_rep = {}
        if not (
            self.wide_dec_replicas
            and k.topk_wide
            and k.pair_rep is not None
            and self.ctx_replicas
            and self._ctx_hot_ids
        ):
            return
        splits = next(iter(k.topk_wide.values())).logit_splits
        built = _try_build(
            "MoE wide decode replica top-k",
            lambda: {
                r: _moe_topk_wide_rep_op.build_wide_rep(
                    r[1],
                    splits,
                    self.ctx_replicas,
                    dev,
                    expert_hist=h.expert_hist,
                    cluster=h.cluster,
                    trim=h.trim,
                )
                for r, h in k.topk_wide.items()
                if r[1] <= _WIDE_REP_MAX_ROWS
            },
        )
        k.topk_wide_rep = built or {}

    def _moe_topk_wide_rep(self, rows: int, layer: int):
        """`(handle, hot ids)` of the wide replica top-k for a wide decode step's `rows` at trunk
        MoE layer `layer`, or None: not built (MTP1, no speculation, replicas off) or a layer
        without hot ids (the MTP layer)."""
        reps = getattr(self._moe_k, "topk_wide_rep", None)
        hot = self._ctx_hot_ids.get(layer) if reps else None
        if hot is None:
            return None
        handle = reps.get(_moe_topk_op.wide_range(rows))
        return None if handle is None else (handle, hot)

    def _moe_topk(self, logits, in_slot: bool, m, slot, layer: int, exact: bool = False):
        """Sigmoid top-k with bias correction, writing this layer's tile maps: `(weights,
        ids, maps)` in the slot's buffers, or None (`noaux_tc_op` then selects). Up to 64 rows
        on the decode kernel; above, on the wide decode kernel for a wide decode step
        (`exact`, 65..256 rows, same ids, weights and tile layout as the prefill kernel's;
        a trunk layer with hot ids on its replica variant, `_moe_topk_wide_rep`) and on the
        prefill one otherwise. `in_slot`: the logits are the slot's router output,
        read in place; others are staged into it first. Matches `noaux_tc_op` on this
        ungrouped routing: same sets, weights to 6e-8. A decode or wide handle built with
        `expert_hist` also writes `layer`'s row of the L2 prefetch's counts."""
        rows = int(logits.shape[0])
        s = self._moe_slots[slot]
        k = self._moe_k
        sig = None
        if self._moe_dec_unpadded(rows) and layer < self.num_layers:
            sig = getattr(k, "topk_sig", {}).get(rows)
        self._moe_topk_signaled = sig is not None
        if rows in k.topk_rep:
            # Decode expert replicas: this layer's hot ids (none on the MTP layer).
            handle = sig if sig is not None else k.topk_rep[rows]
            hot = self._dec_hot_ids.get(layer, self._dec_hot_none)
            if hot is not self._dec_hot_none:
                self._count("moe_dec_replica", True)
            return cute_dsl_moe_topk_routing_rep(
                logits,
                m.router_bias,
                hot,
                self.expert_offset,
                self.routed_scale,
                s.topk_out[rows],
                handle,
                expert_hist=self._pf.hist[layer] if handle.expert_hist else None,
            )
        if rows <= _DECODE_MAX_ROWS:
            handle = sig if sig is not None else k.topk[rows]
            return cute_dsl_moe_topk_routing(
                logits,
                m.router_bias,
                self.expert_offset,
                self.routed_scale,
                s.topk_out[rows],
                None,
                handle,
                expert_hist=self._pf.hist[layer] if handle.expert_hist else None,
            )
        if k.topk_pre is None or rows > _MOE_MAX_T:
            return None
        if not in_slot:
            lg = s.logits[:rows].view(rows, -1)
            self._copy_into(lg, logits.reshape(rows, -1))
            logits = lg
        wide = self._moe_topk_wide(rows) if exact else None
        planes = 1 if logits.dim() == 2 else int(logits.shape[1])
        if wide is not None and wide.logit_splits == planes:
            rep = self._moe_topk_wide_rep(rows, layer)
            if rep is not None and rep[0].logit_splits == planes:
                # Wide decode expert replicas: the same selection, maps over this rank's home
                # and replica slots; `_moe_experts` runs them on the replica pair. Every rank
                # takes this branch for the same rows and layer.
                handle, hot = rep
                self._count("moe_wide_replica", True)
                if handle.cluster == 16:
                    self._count("moe_topk_wide_cl16", True)
                outputs = {"topk_weights": s.w, "topk_indices": s.sel, **s.maps}
                selected = cute_dsl_moe_topk_routing_wide_rep(
                    logits, m.router_bias, hot, self.expert_offset, self.routed_scale, outputs,
                    handle, expert_hist=self._pf.hist[layer] if handle.expert_hist else None,
                )  # fmt: skip
                self._moe_rep_maps = selected[2]
                return selected
            # One fused launch for the three of the prefill top-k (histogram, prefix,
            # scatter); the maps it writes are FC1's as they stand.
            self._count("moe_topk_wide", True)
            if wide.cluster == 16:
                self._count("moe_topk_wide_cl16", True)
            outputs = {"topk_weights": s.w, "topk_indices": s.sel, **s.maps}
            return cute_dsl_moe_topk_routing(
                logits, m.router_bias, self.expert_offset, self.routed_scale, outputs, None, wide,
                expert_hist=self._pf.hist[layer] if wide.expert_hist else None,
            )  # fmt: skip
        hot = self._ctx_hot_ids.get(layer) if k.topk_pre_rep is not None else None
        if (
            hot is not None
            and not exact
            and _moe_topk_pre_rep_op.serves(rows)
            and self._moe_padded_rows(rows) == rows
            and self._moe_pair_ok(rows)
        ):
            # Context replicas: only where the GEMMs run these maps unpadded on the pair.
            # Every rank takes the same branch for the same rows and layer.
            outputs = {"topk_weights": s.w, "topk_indices": s.sel, **s.topk_rep_scratch, **s.maps}
            selected = cute_dsl_moe_prefill_topk_routing_rep(
                logits, m.router_bias, hot, self.expert_offset, self.routed_scale, outputs,
                k.topk_pre_rep,
            )  # fmt: skip
            self._moe_rep_maps = selected[2]
            self._count("ctx_ar_rep", True)
            return selected
        outputs = {"topk_weights": s.w, "topk_indices": s.sel, **s.topk_scratch, **s.maps}
        return cute_dsl_moe_topk_routing(
            logits, m.router_bias, self.expert_offset, self.routed_scale, outputs, None, k.topk_pre
        )

    def _moe_experts(
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
        NVFP4 input already sits in `xq` / `xsf`. `began`: `_moe_begin_layer` advanced the
        slot; otherwise this call uses the current slot and advances it after. `acc`: the
        output already holds the shared expert's partial. `join`: the stream writing that
        partial, waited on between FC1 and FC2 so the shared branch overlaps router, top-k
        and FC1. `layer` is this call site's slot of the decode FC2's tile counter: a launch
        of up to 64 tokens assigns its tiles dynamically, each layer once per forward.
        `exact`: a wide decode step (`_moe_decode_exact`), unpadded below 513 rows.
        """
        k = self._moe_k
        dev = x.device
        # Pad to a power of two below 513 rows: the routing schedule is keyed by
        # `ceil(rows / 32)` there, and arbitrary widths would compile about twenty. Padding
        # rows route to a non-local expert at weight 0 (`pad_expert_id`), so routing drops
        # them, and are sliced off. The prefill pair takes 513 rows and up exactly, and a
        # wide decode step any count: its prefill top-k's maps need no routing launch.
        use_pair = self._moe_pair_ok(rows)
        padded = self._moe_padded_rows(rows, exact)
        if (
            padded != rows
            and began
            and acc
            and prequantized
            and topk_maps is not None
            and self._moe_dec_unpadded(rows)
        ):
            # A trunk decode width that is not a power of two: the top-k's maps at the exact
            # rows, no padding, staging or routing launch (`_moe_dec_unpadded`).
            padded = rows
            self._count("moe_dec_unpad", True)
        if self._moe_topk_signaled:
            self._moe_topk_signaled = False
            if padded != rows:
                # Fail fast: a routing launch after an early-release top-k reads the ids
                # mid-write (a hang, not a wrong answer).
                raise RuntimeError(
                    f"modeling_v2 GLM-5.2: early-release top-k at {rows} rows took the padded path"
                )
            self._count("moe_dec_unpad_signal", True)
        if padded != rows:
            pad = padded - rows
            if not prequantized:
                x = torch.cat([x, torch.zeros(pad, x.shape[1], dtype=x.dtype, device=dev)])
            pad_id = _moe_routing_op.pad_expert_id(self.expert_offset)
            topk_ids = torch.cat(
                [topk_ids, torch.full((pad, _MOE_TOP_K), pad_id, dtype=topk_ids.dtype, device=dev)]
            )
            topk_w = torch.cat(
                [topk_w, torch.zeros(pad, _MOE_TOP_K, dtype=topk_w.dtype, device=dev)]
            )
        slot = self._moe_slot
        if not began:
            self._moe_slot ^= 1
        s = self._moe_slots[slot]
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
                self._count("moe_stage_skip", True)
            else:
                x_s = s.x[:padded]
                self._copy_into(x_s, x)
                x = x_s
        # fp32, as both top-k kernels write it and FC2 reads it: a top-k kernel's own output
        # at an unpadded width goes in as it is.
        if topk_maps is None or padded != rows:
            w_s = s.w[:padded]
            self._copy_into(w_s, topk_w)
            topk_w = w_s
        if topk_maps is not None and padded == rows:
            maps = topk_maps
        else:
            # Always staged: the routing kernels launch with PDL and do not wait for their
            # inputs, and the fused top-k signals its dependents before it writes the ids.
            # The copy puts a node no PDL edge can cross between the two.
            sel = s.sel[: padded * _MOE_TOP_K]
            self._copy_into(sel, topk_ids.reshape(-1))
            maps = cute_dsl_moe_routing(
                sel, self.expert_offset, s.maps, s.counts, s.barrier, k.routing
            )
        xq, xsf = self._moe_xq[:padded], self._moe_xsf[:padded]
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
        rep = use_pair and maps is self._moe_rep_maps and k.pair_rep is not None
        self._moe_rep_maps = None
        compiled, c = (k.pair_rep if rep else k.pair, s.pc) if use_pair else (k.experts, s.c)
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
        pf = self._pf.pending if self._pf is not None else None
        if pf is not None:
            self._pf.pending = None
            join = (join, pf) if join is not None else pf
        cute_dsl_moe_nvfp4_experts(
            xq, xsf, maps, topk_w, fc1_w, fc1_s, fc2_w, fc2_s,
            m.norm_const, alpha1, alpha2, out, c, s.sfc, join, compiled,
            None if use_pair else layer,
        )  # fmt: skip
        return out[:rows]

    def _pf_fork(self, layer: int, m, rows: int, exact: bool = False) -> None:
        """Fork the L2 prefetch stream at a decode MoE layer's o_proj end and prefetch the FC1
        slices of the experts this layer's previous decode step routed the most tokens to.
        The routed call joins it before FC2 (`_moe_experts`). Only on the push top-k's widths
        (1..64 rows): the row bound is checked first, as the context steps call this too.
        Not on the MTP3 / MTP5 wide decode steps (`exact`): there the prefetch costs time
        (4-GPU chain with the real all-reduce, 2 prefetched MoE layers: +4.3 us at 128 rows,
        +2.6 us at 192, 200 alternating replays each)."""
        pf = self._pf
        if pf is None or exact:
            return
        if not 1 <= rows <= _DECODE_MAX_ROWS or not self._moe_k.topk[rows].expert_hist:
            return
        pf.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(pf.stream):
            # The home prefix: the prefetch ranks this rank's 64 home experts (replica slots,
            # when present, follow them in the same storage).
            e = _moe_routing_op.LOCAL_EXPERTS
            cute_dsl_moe_l2_prefetch(pf.hist[layer], m.fc1_w[:e], m.fc1_s[:e], pf.compiled)
        pf.pending = pf.stream
        self._count("moe_l2_prefetch", True)

    def _moe_quant_first_ok(self, x, rows: int) -> bool:
        """Whether `_moe_experts` would quantize `x` itself, unstaged and unpadded, on the
        prefill quantize: then `_routed` may issue that quantize ahead of the router."""
        return (
            self._moe_k.quant_pre is not None
            and self._moe_pair_ok(rows)
            and self._moe_padded_rows(rows) == rows
            and x.dtype is torch.bfloat16
            and x.is_contiguous()
            and x.data_ptr() % 16 == 0
            and not torch.cuda.is_current_stream_capturing()
        )

    def _routed(
        self,
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
        kernels alone. `exact`: a wide decode step, unpadded (`_moe_decode_exact`).

        Each chunk's result is a view of a slot's output buffer, and the slots alternate per
        chunk, so with more than one chunk each result is copied out before a later chunk
        reuses its slot. A single chunk returns the view itself."""
        sizes = _moe_chunk_sizes(rows)
        result = None
        if len(sizes) > 1:
            result = torch.empty((rows, x.shape[1]), dtype=x.dtype, device=x.device)
        start = 0
        for chunk in torch.split(x, sizes, 0):
            n = int(chunk.shape[0])
            slot = self._moe_slot
            # The NVFP4 quantize reads only the chunk: on an exactly sized prefill-pair chunk
            # it goes ahead of the router, before the shared expert's GEMM on the side stream
            # takes the SMs, and the latency-bound top-k overlaps the shared down instead
            # (-16 to -22 us per MoE layer at 7-8k rows on B300). Same kernel and bytes.
            first = not prequantized and self._moe_quant_first_ok(chunk, n)
            if first:
                cute_dsl_moe_quantize_nvfp4(
                    chunk, m.g1, self._moe_xq[:n], self._moe_xsf[:n], self._moe_k.quant_pre
                )
                self._count("moe_quant_first", True)
            # fp32 logits: FC2 reads `token_final_scales` as fp32 and noaux_tc_op's weight
            # dtype follows its logits. bf16 runs clean on garbage.
            logits = self._moe_router(chunk, m, slot, exact)
            in_slot = logits is not None
            if logits is None:
                logits = self._project("router", chunk, m.router_t, torch.float32)
            selected = self._moe_topk(logits, in_slot, m, slot, layer, exact)
            if selected is None:
                topk_w, topk_ids = self._route(logits, m.router_bias)
                maps = None
            else:
                topk_w, topk_ids, maps = selected
            part = self._moe_experts(
                chunk, topk_ids, topk_w, maps, m, n, prequantized or first, join, began, acc,
                layer, exact,
            )  # fmt: skip
            if result is None:
                return part
            self._copy_into(result[start : start + n], part)
            start += n
        return result

    def _mlp_and_close(
        self, attn_out, residual, i: int, rows: int, o8=None, o_pre=False, main=None, decode=False
    ):
        """o_proj, the layer's MLP, and the closing collective normed for the next layer.
        `o8` is `attn_out`'s MXF8 operand when the context attention wrote it (o_proj then
        reads it instead of `attn_out`). Returns `(x, x8)`: the next layer's input and, on a
        two-shot context step, its MXF8 operand for the next layer's qkva (else None).
        `o_pre`: the V-B GEMM wrote `attn_out`'s decode MXF8 operand (`_attn_gen`). `main`: the
        stream the step runs on, when the caller already holds it (else read here).
        `decode`: the caller is the decode target (`_moe_decode_exact`)."""
        a = self._attn[i]
        partial = self._project(
            "o_proj", attn_out, a.o, w8=a.o8, pre=o8, wd8=a.o_dec8, dec_pre=o_pre
        )
        gu, dn = self._mlp[i]
        m = self._moe[i]
        if m is None:
            w8 = self._dense8[i] if self._dense8 is not None else None
            o, x8 = self._reduce_norm_x8(partial, residual, a.norm2, w8 is not None)
            mlp_out = self._dense_mlp(o, gu, dn, w8, x8)
        else:
            exact = decode and self._moe_decode_exact(rows)
            if exact:
                self._count("moe_dec_exact", True)
            # The decode L2 prefetch of this layer's routed FC1 weights (decode widths only).
            self._pf_fork(i, m, rows, exact)
            x8 = None
            w8 = self._shared_w8[i]
            if not self._moe_prequant(rows, exact):
                # The routed experts quantize context rows themselves: the collective writes
                # the shared expert's MXF8 operand instead (two-shot rows only, above 512).
                o, x8 = self._reduce_norm_x8(partial, residual, a.norm2, w8 is not None)
                prequant = False
            else:
                # A decode width, or an MTP3 / MTP5 decode step's exact rows: the collective
                # writes the routed experts' NVFP4 input, and they skip stage copy + quantize.
                o, prequant = self._reduce_norm_quant(partial, residual, a.norm2, m.g1, rows, exact)
                if rows > _DECODE_MAX_ROWS:
                    self._count("moe_dec_prequant", prequant)
            began = rows <= _MOE_MAX_T
            acc = self._moe_begin_layer(rows, exact) if began else None
            if main is None:
                main = torch.cuda.current_stream()
            side = self._side_stream
            side.wait_stream(main)
            with torch.cuda.stream(side):
                if exact:
                    self._shared_after_collective()
                shared = self._shared_mlp(
                    o, gu, self._shared_gu_il[i], dn, out=acc, trunk=True, x8=x8, w8=w8
                )
            routed = self._routed(
                o, m, rows, prequant, side if acc is not None else None, began, acc is not None, i,
                exact,
            )  # fmt: skip
            if acc is None:
                main.wait_stream(side)
                merge = self._moe_k.merge
                if (
                    merge is not None
                    and _MOE_PREFILL_MIN <= rows <= _MOE_MAX_T
                    and routed.shape == shared.shape
                    and routed.is_contiguous()
                ):
                    if not shared.is_contiguous():
                        packed = torch.empty_like(shared, memory_format=torch.contiguous_format)
                        shared = self._copy_into(packed, shared)
                    mlp_out = cute_dsl_moe_merge(routed, shared, merge)
                else:
                    mlp_out = self._add(routed, shared)
            else:
                mlp_out = routed
        if i + 1 < self.num_layers:
            a_next = self._attn[i + 1]
            want = "qkva" in self._proj_mxf8 and a_next.qkva8 is not None
            return self._reduce_norm_x8(mlp_out, residual, self._next_norm[i], want)
        return self._reduce_norm(mlp_out, residual, self._next_norm[i]), None

    def _mtp_front(self, e, h, rows: int):
        """`eh_proj(concat(enorm(e), hnorm(h)))`. At the specialised row counts each rank norms
        one 3072-wide quarter of the concat -- ranks 0-1 halves of `e`, ranks 2-3 halves of
        `h` -- and the projection's K-split partials are summed by the collective. MTP5's
        192 draft-step-0 rows take it on a `_wide_decode` step only, and so do that step's
        other row counts in 65..191 (a batch of fewer than 32 sequences), on the next
        instance up (128 / 192) at their exact rows (`_wide_cell`); with `_wide_bucket` off
        they keep bf16 (the replicated GEMM below), never the MXF8 context path."""
        mw = self._mtp
        k = _mtp_eh.LOCAL_INPUT_SIZE
        wide = self._wide_decode and rows in _mtp_eh.MTP_ROWS
        runner = mw["ehproj"]
        ksplit = rows in _mtp_eh.SUPPORTED_M_VALUES or wide
        if not ksplit and self._wide_decode and rows > _DECODE_MAX_ROWS:
            cells = dict.fromkeys(_mtp_eh.WIDE_ROWS, runner)
            runner = self._wide_cell("mtp_eh", cells, rows, _mtp_eh.at_rows)
            ksplit = runner is not None
        elif not ksplit and self._cell_up(rows):
            # An eager draft step's or an MTP3 / MTP5 decode step's count: the next supported
            # instance up at its exact rows (`at_rows` picks the instance; one runner holds
            # them all).
            runner = self._draft_cell(
                "mtp_eh", rows, {_DECODE_MAX_ROWS: runner}, {}, _mtp_eh.at_rows
            )
            ksplit = runner is not None
        if ksplit and e.is_contiguous() and h.is_contiguous():
            src, nw = (e, mw["enorm"]) if self.tp_rank < 2 else (h, mw["hnorm"])
            local = torch.empty([rows, k], dtype=e.dtype, device=e.device)
            cute_dsl_mtp_norm_shard(
                src, nw, local, (self.tp_rank % 2) * k, self.eps, mw["norm_shard"]
            )
            return self._reduce(cute_dsl_mtp_eh_proj(local, mw["eh_local"], runner))
        eh8 = self._mtp_eh8
        if (
            eh8 is not None
            and rows > _DECODE_MAX_ROWS
            and not self._wide_decode
            and _eh8_op.supports(e, h)
        ):
            # The two norms write the MXF8 GEMM's operand halves: no bf16 norms, no concat
            # copies, no activation quant.
            q8 = _eh8_op.operand(rows, e.device)
            cute_dsl_mtp_eh_norm_cat_mxf8(e, mw["enorm"], h, mw["hnorm"], self.eps, *q8, eh8.norm)
            runner = self._proj_mxf8["mtp_eh"]
            out = torch.empty([rows, runner.n], dtype=torch.bfloat16, device=e.device)
            self._count("mtp_eh_mxf8", True)
            return cute_dsl_projection_mxf8(q8[0], eh8.w8, out, runner, pre=q8)
        # The concat written by two copies into one buffer: the copy kernel takes each
        # half (a column slice of the buffer), torch.cat would be one more launch.
        en, hn = self._rmsnorm(e, mw["enorm"]), self._rmsnorm(h, mw["hnorm"])
        cat = torch.empty([rows, 2 * en.shape[1]], dtype=en.dtype, device=en.device)
        self._copy_into(cat[:, : en.shape[1]], en)
        self._copy_into(cat[:, en.shape[1] :], hn)
        # The replicated projection: the prefill GEMM above the decode row counts, else cuBLAS.
        return self._project("mtp_eh", cat, mw["eh"])

    def _mtp_experts(self, x, topk_ids, topk_w, rows: int, frontend: bool):
        """One MTP chunk's bf16 round trip; returns the slot's output rows. With `frontend`
        the router and the fused top-k (which also clears the output) run here on the
        input in place; otherwise the caller's ids and weights select, staged with the input
        into the slot's buffers."""
        mw, k = self._mtp, self._mtp_k
        dev = x.device
        padded = _pow2(rows)
        if padded != rows:
            pad = padded - rows
            x = torch.cat([x, torch.zeros(pad, x.shape[1], dtype=x.dtype, device=dev)])
            if not frontend:
                # Padding rows go to a non-local expert at weight 0, see `pad_expert_id`.
                pad_id = _moe_routing_op.pad_expert_id(self.expert_offset)
                topk_ids = torch.cat(
                    [
                        topk_ids,
                        torch.full((pad, _MOE_TOP_K), pad_id, dtype=topk_ids.dtype, device=dev),
                    ]
                )
                topk_w = torch.cat(
                    [topk_w, torch.zeros(pad, _MOE_TOP_K, dtype=topk_w.dtype, device=dev)]
                )
        slot = self._mtp_slot
        self._mtp_slot ^= 1
        s = self._mtp_slots[slot]
        out = s.out[:padded]
        if frontend:
            # The router and gather GEMM read the activation in place; weights and maps come
            # from the top-k's outputs, and the top-k zeroed the output.
            if padded <= _moe_router_op.SIMT_MAX_ROWS:
                logits = s.router_out[:padded]
            else:
                logits = s.router_partials[:padded]
            cute_dsl_moe_router(x, mw["router"], logits, self._moe_k.router[padded])
            topk_w, _, maps = cute_dsl_moe_topk_routing(
                logits,
                mw["router_bias"],
                self.expert_offset,
                self.routed_scale,
                s.topk_out[padded],
                out,
                k.topk[padded],
            )
            xs = x
        else:
            # The routing kernel does not wait before reading the ids, and `noaux_tc_op`
            # releases it before its stores are visible: these copies are what order them.
            xs = s.x[:padded]
            self._copy_into(xs, x)
            w_s = s.w[:padded]
            self._copy_into(w_s, topk_w)
            topk_w = w_s
            sel = s.sel[: padded * _MOE_TOP_K]
            self._copy_into(sel, topk_ids.reshape(-1))
            maps = cute_dsl_moe_routing(
                sel, self.expert_offset, s.maps, s.counts, s.barrier, self._moe_k.routing
            )
            out.zero_()
        cute_dsl_mtp_moe_experts(xs, maps, topk_w, mw["fc1"], mw["fc2"], out, s.c, k.experts)
        return out[:rows]

    def _mtp_routed(self, x, rows: int):
        """The MTP module's window, bf16. Up to 64 rows the front end runs with the experts;
        above, the stock router projection and noaux_tc_op select. With more than one
        chunk each result is copied out before a later chunk reuses its slot."""
        mw = self._mtp
        sizes = _moe_chunk_sizes(rows)
        result = None
        if len(sizes) > 1:
            result = torch.empty((rows, x.shape[1]), dtype=x.dtype, device=x.device)
        start = 0
        for chunk in torch.split(x, sizes, 0):
            n = int(chunk.shape[0])
            if n <= _DECODE_MAX_ROWS:
                part = self._mtp_experts(chunk, None, None, n, frontend=True)
            else:
                logits = self._project("router", chunk, mw["router_t"], torch.float32)
                topk_w, topk_ids = self._route(logits, mw["router_bias"])
                part = self._mtp_experts(chunk, topk_ids, topk_w, n, frontend=False)
            if result is None:
                return part
            self._copy_into(result[start : start + n], part)
            start += n
        return result

    def _mtp_close(self, attn_out, residual, rows: int, decode=False):
        """The MTP layer's o_proj and MoE: its routed experts on the trunk's NVFP4 path
        (`_mtp_moe_fp4`) at layer slot `num_layers`, else on its bf16 pair. Returns the
        hidden state after the module's own head norm when the engine drafts more than one
        token (`_mtp_out_normed`: it is fed back into the next draft step, as stock's), else
        un-normalized with the norm applied by `shared_head`. `decode`: the caller is the
        decode target (`_moe_decode_exact`); there the residual add and the head norm run
        as one launch (`_add_rmsnorm`, bit for bit the two)."""
        j = self.num_layers
        a = self._attn[j]
        o = self._reduce_norm(self._project("o_proj", attn_out, a.o), residual, a.norm2)
        gu, dn = self._mlp[j]
        exact = self._mtp_fp4 is not None and decode and self._moe_decode_exact(rows)
        main = torch.cuda.current_stream()
        side = self._side_stream
        side.wait_stream(main)
        with torch.cuda.stream(side):
            if exact:
                self._shared_after_collective()
            shared = self._shared_mlp(o, gu, self._shared_gu_il[j], dn)
        if self._mtp_fp4 is not None:
            self._count("mtp_moe_fp4", True)
            if exact:
                self._count("moe_dec_exact", True)
            routed = self._routed(o, self._mtp_fp4, rows, False, None, False, False, j, exact)
        else:
            routed = self._mtp_routed(o, rows)
        main.wait_stream(side)
        reduced = self._reduce(self._add(routed, shared))
        if self._mtp_out_normed and decode:
            # MTP3 / MTP5 decode: the module's output is only ever read normed.
            return self._add_rmsnorm(residual, reduced, self._mtp["head_norm"])
        out = self._add(residual, reduced)
        if self._mtp_out_normed:
            return self._rmsnorm(out, self._mtp["head_norm"])
        return out


class PrefillTarget:
    """The general case: context rows, and possibly generation rows beside them."""

    def __init__(self, core: "GlmMoeDsaTep4ModelingV2") -> None:
        """Hold the core by reference: every operand is read through `self.core` at the call."""
        self.core = core
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks for it; read once here rather than per
        # forward. Each target owns its own flag: the two `step_args` projections differ,
        # so probing one does not prove the other's fields exist.
        self._contract_pending = step_contract_enabled()

    def _check_step_contract(self, md) -> None:
        """Opt-in first-forward fail-fast, run when TRTLLM_MODELING_V2_VALIDATE asks for it:
        the metadata fields this target consumes must exist (they are private trtllm
        surface). Calling `step_args` is the check -- it reads every field this target
        consumes. Everything checked is fixed at engine construction, so once per target
        instance is sound, and it is off in a served engine."""
        if not self._contract_pending:
            return
        try:
            self.step_args(md)
        except AttributeError as exc:
            raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
        self._contract_pending = False

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        return _build_step_args(md)

    @staticmethod
    def _diet_tables(step: dict) -> None:
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
    def _diet_block_offsets(step: dict, pool_idx: int):
        """CTX-HOST-DIET: the context requests' block-table rows of pool `pool_idx`, sliced once
        per step and shared by every layer of that pool (a view; nothing writes it)."""
        cache = step["diet_block_offsets"]
        rows = cache.get(pool_idx)
        if rows is None:
            rows = cache[pool_idx] = step["kv_cache_block_offsets"][
                pool_idx, : step["num_contexts"]
            ]
        return rows

    def _ctx_rope_step(self, step, tc: int) -> tuple | None:
        """The context rope kernel's per-step operands: the new-row prefix sum and the cached
        lengths, int32, from the engine's int64 prefix sums. None sends the step's context rows
        to the stock rope/append op. Context steps are not captured, so a fresh pair per step
        is fine.

        Built on the host from the prefix sums' host copies and uploaded in one copy: the
        conversions and the difference on the device would be four small kernels per step."""
        core = self.core
        ctx = step["num_contexts"]
        compiled = core._ctx_rope_compiled
        if (
            compiled is None
            or not 0 < ctx <= _ctx_rope_op.MAX_REQUESTS
            or int(step["tokens_per_block"]) != compiled.tokens_per_block
            or tc >= _ROPE_APPEND_MAX_TOKENS
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

    def _dense_causal_step(
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
        core = self.core
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
            core.dense_causal_misses[reason] = core.dense_causal_misses.get(reason, 0) + 1
            return None
        ctx = step["num_contexts"]
        host_lens = step["seq_lens"][:ctx].to(torch.int32).cpu()
        assert int(host_lens.sum()) == tc, (host_lens.tolist(), tc)
        core.dense_causal_steps += 1
        if rope_ok and core._kvb_expand:
            return SimpleNamespace(
                mha=self._mha_plan(md, step, tc, host_lens, req_idx, dev),
                absorbed=None,
                rebuild=lambda: self._dense_causal_absorbed(md, step, tc, host_lens, req_idx, dev),
            )
        return self._dense_causal_absorbed(md, step, tc, host_lens, req_idx, dev)

    def _dense_causal_absorbed(self, md, step, tc: int, host_lens, req_idx, dev):
        """The absorbed dense-causal kernel's per-step operands (`_dense_causal_step`)."""
        core = self.core
        ctx = step["num_contexts"]
        cached = None
        if int(step["num_ctx_cached_tokens"]):
            cached = torch.diff(step["ctx_cached_token_indptr"][: ctx + 1]).to(torch.int32)
        # Both without a host sync (the kernel module's helpers read the device lengths back
        # and upload pageable memory): the host already checked the row total above.
        ks, ke = _dc_bounds_nosync(step["seq_lens_cuda"][:ctx].to(torch.int32), cached, tc)
        if core._dc_uploader is None:
            core._dc_uploader = _DcGroupsUploader(dev)
        groups = core._dc_uploader(_dc_groups_host(host_lens, _fmha_pre_op.DENSE_CAUSAL_GROUP))
        return SimpleNamespace(
            ks=ks,
            ke=ke,
            groups=groups,
            req_idx=req_idx[:tc],
            block_table=md.block_table,
            # One fp8 query for the step: each layer writes it and its attention reads it
            # before the next layer writes (one stream), so the bound address holds.
            quant_q=torch.empty(
                [tc, core.heads, core.lat_dim], dtype=torch.float8_e4m3fn, device=dev
            ),
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
        core = self.core
        ctx = step["num_contexts"]
        cached = int(step["num_ctx_cached_tokens"])
        tk = tc + cached
        rows = core.heads * core.qk_dim
        kv_dt = torch.float8_e4m3fn if core._mha_fp8 else torch.bfloat16
        kvb8 = core._mha_fp8 and core._kvb8 is not None
        # The one-launch expansion writes `[K | V]` rows; K and V are their halves.
        kv = None
        if kvb8 and core._kvb8.merged is not None:
            kv = torch.empty([tk, rows + core.heads * core.v_dim], dtype=kv_dt, device=dev)
        plan = SimpleNamespace(
            max_len=int(host_lens.max()),
            tk=tk,
            k=torch.empty([tk, rows], dtype=kv_dt, device=dev) if kv is None else kv[:, :rows],
            v=(
                torch.empty([tk, core.heads * core.v_dim], dtype=kv_dt, device=dev)
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
            pool_rows=step["primary_pool"].view(torch.uint8).reshape(-1, core.lat_dim),
            # The bf16 [cached | new] rows of the bf16-MMA expansion (the MXF8 one gathers
            # into `a8` instead).
            lat=None if kvb8 else torch.empty([tk, core.lat_dim], dtype=torch.bfloat16, device=dev),
        )
        return plan

    def _mha_takes(self, a, layer: int, step, ctx_rope) -> bool:
        """Whether this layer's context rows can take the non-absorbed attention: its weights
        built, the step on the context rope kernel, and the layer's pool the primary pool."""
        core = self.core
        diet = step.get("diet_layers")
        if diet is not None:
            return (
                (a.kvb8 if core._mha_fp8 else a.kvb_aug) is not None
                and ctx_rope is not None
                and core._ctx_rope_compiled is not None
                and diet[layer][2]
            )
        pool_ptrs = step["host_kv_cache_pool_pointers"]
        pool_idx = int(step["host_kv_cache_pool_mapping"][layer][0])
        return (
            (a.kvb8 if core._mha_fp8 else a.kvb_aug) is not None
            and ctx_rope is not None
            and core._ctx_rope_compiled is not None
            and int(pool_ptrs[pool_idx, 0]) == step["primary_pool"].data_ptr()
            and int(pool_ptrs[pool_idx, 1]) == 0
        )

    def _attn_mha(
        self, a, layer: int, q, latent, out, tc: int, step, ctx_rope, mha, o8=None
    ) -> bool:
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

        In the fp8 form (`q` the e4m3 query from `_q_b_e4m3`) rope rotates the e4m3 query in
        place, K / V are e4m3 (the V expansion on the latent's c_kv part), and the MHA reads
        the layer's per-head dequantization table. Under the MXF8 expansion steps 2 and 3 are
        one gather writing the `[prefix | new]` rows as MXF8 and two MXF8 GEMMs. With `o8`
        (fp8 form, the MXF8-O handle built) the MHA writes the context rows' attention as
        o_proj's MXF8 operand into `o8` instead of `out`."""
        core = self.core
        if not self._mha_takes(a, layer, step, ctx_rope):
            return False
        fp8 = q.dtype == torch.float8_e4m3fn
        pool = step["primary_pool"]
        diet = step.get("diet_layers")
        if diet is not None:
            pool_idx, layer_in_pool, _ = diet[layer]
            tpb = step["diet_tpb"]
            block_offsets = self._diet_block_offsets(step, pool_idx)
        else:
            pool_idx, layer_in_pool = (
                int(v) for v in step["host_kv_cache_pool_mapping"][layer][:2]
            )
            tpb = int(step["tokens_per_block"])
            block_offsets = step["kv_cache_block_offsets"][pool_idx, : step["num_contexts"]]
        compiled = core._ctx_rope_compiled
        heads = core.heads
        q3 = torch.reshape(q, [tc, heads, core.qk_dim])
        in_place = latent.stride(0) in compiled.kernels_q_in_place and latent.stride(1) == 1
        lat = latent if in_place else latent.clone(memory_format=torch.contiguous_format)
        cute_dsl_mla_rope_append_prefill(
            None,
            q3[..., core.nope :],
            lat,
            None,
            pool,
            layer_in_pool,
            block_offsets,
            *ctx_rope,
            core._rope["rotary_cos_sin"],
            tpb,
            compiled,
            q_in_place=True,
            q_e4m3=fp8,
        )
        if fp8 and compiled.rope_par:
            core._count("rope_par", True)
        src, cu_k = lat, None
        if mha.a8 is not None:
            self._expand_mxf8(a, lat, mha, layer_in_pool * tpb, tpb)
            cu_k = None if mha.cached is None else mha.cached.cu_k
            core._count("ctx_kvb_mxf8", True)
            if mha.kv is not None:
                core._count("ctx_kvb_merged", True)
        elif mha.cached is not None:
            c = mha.cached
            src = cute_dsl_mha_prefix_gather(
                c.pool_rows, c.idx, c.rows, c.req_idx, c.block_table, lat, c.lat,
                layer_in_pool * tpb, core._stride_factor, tpb, compiled=core._mha_prefix,
            )  # fmt: skip
            cu_k = c.cu_k
        # The prefill GEMM serves up to PREFILL_MAX_ROWS rows; with a cached prefix the K/V
        # rows can exceed it, and the product is row-independent.
        k_rows = heads * core.qk_dim
        if mha.a8 is not None:
            halves = ()
        elif fp8:
            halves = (
                ("k", a.kvb8[0], mha.k, src),
                ("v", a.kvb8[1], mha.v, src[:, : core.kv_lora]),
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
                cute_dsl_projection(x[r0:r1], w, dst[r0:r1], core._kvb_expand[half])
        use_o8 = o8 is not None and fp8 and core._mha_q8 is not None
        cute_dsl_mha_prefill(
            q3,
            mha.k.view(mha.tk, heads, core.qk_dim),
            mha.v.view(mha.tk, heads, core.v_dim),
            torch.reshape(out, [tc, heads, core.v_dim]),
            ctx_rope[0],
            mha.max_len,
            compiled=core._mha_q8.mha if use_o8 else core._mha_pre,
            cu_seqlens_k=cu_k,
            head_scales=a.mha_hs if fp8 else None,
            q8_out=o8 if use_o8 else None,
        )
        assert use_o8 == (o8 is not None), "an o_proj operand the MHA did not write"
        core._count("ctx_mha", True)
        if use_o8:
            core._count("ctx_mha_o8", True)
        if fp8:
            core._count("ctx_mha_fp8", True)
        if cu_k is not None:
            core._count("ctx_mha_cached", True)
        return True

    def _expand_mxf8(self, a, lat, mha, layer_offset: int, tpb: int) -> None:
        """K / V of this layer into `mha.k` / `mha.v` (e4m3) on the MXF8 expansion: one gather
        writes the `[cached prefix | new]` latent rows as MXF8 into `mha.a8` (cached rows as
        their pool bytes at 2^0, the new rows `lat` block-quantized), then the two GEMMs run
        on it in row slices of the runners' capacity (a multiple of 128 rows, so a slice's
        scales start at a 128-row block)."""
        core = self.core
        g = core._kvb8
        if mha.cached is not None:
            c = mha.cached
            cute_dsl_mha_kv_gather_mxf8(
                c.pool_rows, c.idx, c.rows, c.req_idx, c.block_table, lat, mha.a8,
                layer_offset, core._stride_factor, tpb, True, compiled=g.gather,
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
        rk, rv = core._kvb_expand["k"], core._kvb_expand["v"]
        step = rk.launch.max_rows
        for r0 in range(0, mha.tk, step):
            r1 = min(mha.tk, r0 + step)
            b = r0 // 128
            cute_dsl_projection_mxf8(a_k[r0:r1], w_k, mha.k[r0:r1], rk, sfa=sfa_k[b * blk_k :])
            cute_dsl_projection_mxf8(a_v[r0:r1], w_v, mha.v[r0:r1], rv, sfa=sfa_v[b * blk_v :])

    def _attn_ctx(
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
        core = self.core
        if dc is not None and dc.mha is not None:
            if self._attn_mha(a, layer, q, latent, out, tc, step, ctx_rope, dc.mha, o8):
                return o8 is not None
            # The fp8 query exists only once `_mha_takes` said the layer takes the route.
            assert q.dtype != torch.float8_e4m3fn, "an fp8 query on the absorbed path"
            core._count("ctx_mha", False)
            # This layer takes the absorbed dense-causal kernel: its operands, built once.
            if dc.absorbed is None:
                dc.absorbed = dc.rebuild()
            dc = dc.absorbed
        heads, dev, dt = core.heads, q.device, q.dtype
        q_nope, q_pe = torch.split(
            torch.reshape(q, [tc, heads, core.qk_dim]), [core.nope, core.rope_dim], -1
        )
        fused_q = torch.empty([tc, heads, core.lat_dim], dtype=dt, device=dev)
        fq_nope, fq_pe = torch.split(fused_q, [core.kv_lora, core.rope_dim], -1)
        quant_q = (
            dc.quant_q
            if dc is not None
            else torch.empty([tc, heads, core.lat_dim], dtype=torch.float8_e4m3fn, device=dev)
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
        core._count("ctx_rope", on_kernel)
        # On the rope kernel's path the attention reads only the fp8 query: K-B writes its
        # e4m3 q-base bytes straight into `quant_q[..., :kv_lora]` and the rope kernel skips
        # its q-base groups, so the bf16 `fused_q[..., :kv_lora]` (129 MB at 8k rows) is
        # neither written nor re-read. Same bytes into the attention (FP32 -> BF16 -> E4M3,
        # nearest, on both paths). Row counts the decode groups serve keep the bf16 K-B.
        kb_fp8 = (
            on_kernel
            and core._kb_fp8_pre is not None
            and tc not in core._absorb["k_b"]
            and tc <= _absorbed_bmm_op.PREFILL_MAX_ROWS
        )
        if kb_fp8:
            core._count("ctx_kb_fp8", True)
            q_base = torch.split(quant_q.view(torch.uint8), [core.kv_lora, core.rope_dim], -1)[0]
            cute_dsl_absorbed_bmm(
                torch.transpose(q_nope, 0, 1),
                a.k_b_nk,
                torch.transpose(q_base, 0, 1),
                core._kb_fp8_pre,
            )
        else:
            core._absorbed_bmm(
                "k_b",
                torch.transpose(q_nope, 0, 1),
                a.k_b,
                a.k_b_nk,
                torch.transpose(fq_nope, 0, 1),
            )
        if on_kernel:
            in_place = latent.stride(0) in core._ctx_rope_compiled.kernels and latent.stride(1) == 1
            core._count("ctx_rope_latent_in_place", in_place)
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
                    self._diet_block_offsets(step, pool_idx)
                    if diet is not None
                    else step["kv_cache_block_offsets"][pool_idx, : step["num_contexts"]]
                ),
                *ctx_rope,
                core._rope["rotary_cos_sin"],
                step["tokens_per_block"],
                core._ctx_rope_compiled,
                skip_q_base=kb_fp8,
            )
        else:
            # The append rotates the query's rope tail in place, reading its own
            # pre-rotation contents, and the absorbed BMM wrote only the first kv_lora
            # columns: seed the tail.
            core._copy_into(fq_pe, q_pe)
            mla_rope_append_paged_kv_assign_q(
                torch.reshape(fused_q, [tc, heads * core.lat_dim]),
                latent.contiguous(),
                step["num_contexts"],
                step["ctx_cached_token_indptr"],
                step["ctx_kv_indptr"],
                step["max_ctx_seq_len"],
                core._rope["rotary_cos_sin"],
                heads,
                # The width ahead of each head's rope tail in the query this call hands
                # over: the absorbed head is `kv_lora + rope`, so this is kv_lora, not
                # qk_nope_head_dim.
                core.kv_lora,
                core.rope_dim,
                core.kv_lora,
                step["kv_cache_block_offsets"],
                step["host_kv_cache_pool_pointers"],
                step["host_kv_cache_pool_mapping"],
                None,
                0,
                layer,
                step["tokens_per_block"],
                step["max_seq_len"],
                1,
                core.quant_mode,
            )
            core._copy_into(quant_q, fused_q)
        if ctx_last is not None:
            assert on_kernel and dc is None, "the selected-rows attention needs the rope kernel"
            self._attn_ctx_selected(a, quant_q, pool, out, ctx_last)
            return False
        lat_out = torch.empty([tc, heads * core.kv_lora], dtype=dt, device=dev)
        if dc is not None:
            # The pool rows this layer's selection would name, `to_global` of the dense
            # causal index: the block-table row times the stride, plus this layer's offset.
            tpb = step["diet_tpb"] if diet is not None else int(step["tokens_per_block"])
            layer_offset = layer_in_pool * tpb
            if dc.prepared is None:
                dc.prepared = _fmha_pre_op.prepare_dense_causal(
                    core._fmha_pre_dc_compiled, quant_q, pool, lat_out, dc.ks, dc.ke,
                    core._fmha_bmm1, core._fmha_bmm2, dc.block_table, dc.req_idx,
                    core._stride_factor, layer_offset, dc.groups,
                )  # fmt: skip
            cute_dsl_sparse_mla_prefill(
                quant_q, pool, None, lat_out, dc.ks, dc.ke, core._fmha_bmm1, core._fmha_bmm2,
                compiled=dc.prepared, block_table=dc.block_table, req_idx=dc.req_idx,
                stride_factor=core._stride_factor, layer_offset=layer_offset,
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
                core._fmha_bmm1,
                core._fmha_bmm2,
                compiled=core._fmha_pre_compiled,
            )
        core._absorbed_bmm(
            "v_b",
            torch.transpose(torch.reshape(lat_out, [tc, heads, core.kv_lora]), 0, 1),
            a.v_b_t,
            a.v_b_nk,
            torch.transpose(torch.reshape(out, [tc, heads, core.v_dim]), 0, 1),
        )
        return False

    def _attn_ctx_selected(self, a, quant_q, pool, out, last) -> None:
        """The plan's context queries (`last.ctx_rows` of the step's e4m3 query) on the
        dense-causal kernel, each its own group over its request's block-table row (positions
        0 .. cached + len - 1: the engine's dense index, so bitwise the index path's rows),
        then V-B into `out` (`last.n_pad` rows)."""
        core = self.core
        heads, dev, n = core.heads, quant_q.device, last.n_pad
        tc = int(quant_q.shape[0])
        q_sel = torch.empty([n, heads, core.lat_dim], dtype=torch.float8_e4m3fn, device=dev)
        cute_dsl_rows_select(
            quant_q.view(tc, heads * core.lat_dim),
            last.ctx_rows,
            q_sel.view(n, heads * core.lat_dim),
            core._rows_sel,
        )
        lat_out = torch.empty([n, heads * core.kv_lora], dtype=out.dtype, device=dev)
        prepared = _fmha_pre_op.prepare_dense_causal(
            core._fmha_pre_dc_compiled, q_sel, pool, lat_out, last.ks, last.ke,
            core._fmha_bmm1, core._fmha_bmm2, last.block_table, last.req, last.stride,
            last.layer_offset, last.groups,
        )  # fmt: skip
        cute_dsl_sparse_mla_prefill(
            q_sel, pool, None, lat_out, last.ks, last.ke, core._fmha_bmm1, core._fmha_bmm2,
            compiled=prepared, block_table=last.block_table, req_idx=last.req,
            stride_factor=last.stride, layer_offset=last.layer_offset,
            dense_causal_groups=last.groups,
        )  # fmt: skip
        core._absorbed_bmm(
            "v_b",
            torch.transpose(torch.reshape(lat_out, [n, heads, core.kv_lora]), 0, 1),
            a.v_b_t,
            a.v_b_nk,
            torch.transpose(torch.reshape(out, [n, heads, core.v_dim]), 0, 1),
        )

    def _layer_attention(
        self,
        md,
        step,
        x,
        pos,
        attn_out,
        i: int,
        pool_layer: int,
        slots_of,
        topk_buf,
        rows,
        tc,
        gen,
        gen_p,
        ctx,
        need_q,
        skip_ctx,
        skip_gen,
        ctx_rope=None,
        dc=None,
        x8=None,
        ctx_last=None,
        split_memo=None,
    ):
        """One layer's attention block, both phases, into `attn_out`. `slots_of(topk_buf, fresh)`
        maps the request-local selection to this layer's pool rows, `(context, generation,
        generation index kind, its layer)` (`fresh`: this layer selected anew); the context half
        is None under the dense-causal context attention (`dc`). `x8` is `x`'s
        MXF8 operand (the closing all-reduce of the layer before wrote it). `split_memo` is the
        step's store for the decode indexer's MTP5 atom split metadata.

        Returns o_proj's MXF8 operand `(q8, sf)` when the fp8 context attention wrote the
        context rows' attention into it (their bf16 rows of `attn_out` are then not written:
        o_proj must take the operand), else None."""
        core = self.core
        a = core._attn[i]
        # The fp8 context attention takes its context rows' query from q_b's e4m3 output;
        # the generation rows' bf16 query comes from the decode cells.
        mha8 = (
            bool(tc)
            and dc is not None
            and dc.mha is not None
            and core._mha_fp8
            and self._mha_takes(a, pool_layer, step, ctx_rope)
        )
        # The q_a norm writes the context rows' q_a as the MXF8 q_b's operand; with no indexer
        # reading q_a this step, the only bf16 q_a reader is the generation rows' q_b tail.
        q8_rows = tc if mha8 and isinstance(core._q8, _mxf8_op.Mxf8ProjectionRunner) else 0
        store_from = tc - _Q_A_TAIL_ROWS if q8_rows and not need_q and tc > _Q_A_TAIL_ROWS else 0
        q, q_a, latent, qa8 = core._qkv(
            x, a, q_b=not mha8, x8=x8, q8_rows=q8_rows, q_store_from=store_from
        )
        ix = core._idx[i]
        if ix is not None:
            q_fp8, weights, k_pre = core._indexer_project(x, q_a, pos, ix, rows, need_q)
            k_pool = md.kv_cache_manager.get_indexer_k_cache_buffers(i)
            core._indexer_write_k(md, k_pool, k_pre, rows)
            if tc and not skip_ctx:
                # With MLA chunked prefill and at least `q_split_threshold` context queries, the
                # engine plans the chunks for this rank's share of the queries only
                # (`indexer_prefill_q_split`); the other ranks' top-k rows are gathered after
                # the loop, as the engine's own indexer does.
                q_split = getattr(md, "indexer_prefill_q_split", None)
                for chunk in md.indexer_prefill_chunks:
                    n_k = chunk.k_token_end - chunk.k_token_start
                    if chunk.token_end <= chunk.token_start:
                        continue
                    if n_k <= 0:
                        # No compressed KV for these queries: no valid index, as the engine.
                        topk_buf[chunk.token_start : chunk.token_end].fill_(-1)
                        continue
                    if n_k > core._ix_topk_pre_compiled.max_columns:
                        # A K block wider than any built so far (a long cached prefix, or packed
                        # requests): context steps are never captured, so widen the build here.
                        core._ix_topk_pre_compiled = build_ix_topk_prefill(
                            torch.cuda.current_device(), top_k=core.idx_topk, max_columns=n_k
                        )
                    ck, cs = indexer_k_cache_gather_op(
                        k_pool,
                        md.slot_mapping_fp8_fullkv,
                        md.slot_mapping_scale_fullkv,
                        chunk.k_token_start,
                        n_k,
                        core.idx_dim,
                    )
                    cs = torch.reshape(cs, [-1])
                    # The window vectors hold one entry per query of this chunk: they are
                    # indexed by position within the chunk, not by context token.
                    t0 = chunk.token_start
                    assert chunk.cu_seqlen_ks.shape[0] == chunk.token_end - t0
                    for lo, hi in _tile_bounds(t0, chunk.token_end, n_k):
                        ks = chunk.cu_seqlen_ks[lo - t0 : hi - t0]
                        ke = chunk.cu_seqlen_ke[lo - t0 : hi - t0]
                        logits = fp8_mqa_logits(q_fp8[lo:hi], (ck, cs), weights[lo:hi], ks, ke)
                        cute_dsl_indexer_topk_prefill(
                            logits, ks, ke, topk_buf[lo:hi],
                            top_k=core.idx_topk, compiled=core._ix_topk_pre_compiled,
                        )  # fmt: skip
                if q_split is not None:
                    q_lo, q_hi, q_sizes = q_split
                    topk_buf[: sum(q_sizes)] = allgather(
                        topk_buf[q_lo:q_hi], md.mapping, dim=0, sizes=q_sizes
                    )
            if gen and not skip_gen:
                core._indexer_decode(
                    md, k_pool, q_fp8[tc:], weights[tc:], topk_buf[tc:], ctx, gen // gen_p, gen_p,
                    split_memo,
                )  # fmt: skip
            # Draft step 0 of a context or mixed step (the later steps have no context rows
            # and take `DecodeTarget`): the MTP layer stashes its selection for them.
            if (
                i == core.num_layers
                and core._mtp_share_stash
                and md.in_mtp_draft_loop
                and not _mtp_share_reuses(md)
            ):
                _mtp_share_store(md, topk_buf, ctx, tc, gen // gen_p if gen else 0, gen_p)
        slots_ctx, slots_gen, gen_kind, gen_layer = slots_of(topk_buf, ix is not None)
        if mha8:
            q_ctx = core._q_b_e4m3(q_a[:tc], a, pre=qa8)
            q_gen = core._project_tail("q_b", q_a, tc, a.qb) if gen else None
        else:
            q_ctx, q_gen = torch.split(q, [tc, gen], 0)
        lat_ctx, lat_gen = torch.split(latent, [tc, gen], 0)
        # With `ctx_last` the context part of `attn_out` holds the plan's `n_pad` rows.
        n_ctx_out = tc if ctx_last is None else ctx_last.n_pad
        attn_ctx, attn_gen = torch.split(attn_out, [n_ctx_out, gen], 0)
        # o_proj's MXF8 operand from the fp8 MHA's epilogue, where o_proj takes the MXF8 GEMM
        # (trunk layers only: the MTP layer's close reads the bf16 `attn_out`).
        o8 = None
        if (
            mha8
            and i < core.num_layers
            and core._mha_q8 is not None
            and core._mxf8_takes("o_proj", attn_out, a.o8)
        ):
            o8 = (
                torch.empty([rows, core.heads * core.v_dim], dtype=torch.uint8, device=x.device),
                torch.empty(_mha_pre_op.sf_bytes(rows), dtype=torch.uint8, device=x.device),
            )
        # On a mixed step on the fp8 MHA route the generation rows' attention (absorbed K-B,
        # RoPE append, sparse MLA decode, V-B) reads only q_gen / lat_gen / their slots and the
        # generation requests' pool pages, and writes only attn_gen and those pages, so it runs
        # on its own stream beside the context rows' RoPE, K/V expansion and MHA instead of
        # after them (about 30 us per layer on the critical path at 7-8k context rows on B300;
        # -9 us per layer net, the expansion GEMM sharing SMs with it). Joined before the
        # generation rows' quantize and o_proj read attn_gen. Same kernels and inputs: the
        # outputs are bit for bit the serial ones. The absorbed context path shares K-B / V-B
        # with these rows and keeps the serial order.
        gen_side = bool(gen and tc and mha8 and ctx_last is None and core._gen_stream is not None)
        main = step.get("diet_main") if gen_side else None
        # With `_gen_side_quant` the generation rows' quantize into `o8` also runs there; when
        # the context rows then do not take `o8`, it is dropped unread, as below.
        side_quant = gen_side and o8 is not None and core._gen_side_quant
        if gen_side:
            core._gen_stream.wait_stream(torch.cuda.current_stream() if main is None else main)
            with torch.cuda.stream(core._gen_stream):
                core._attn_gen(
                    a, pool_layer, q_gen, lat_gen, attn_gen, slots_gen, gen, gen_p, ctx, md,
                    gen_kind, gen_layer, False,
                )  # fmt: skip
                if side_quant:
                    cute_dsl_act_quant_mxf8_rows(attn_gen, *o8, tc, compiled=core._mha_q8.rows)
            core._count("ctx_gen_side", True)
            core._count("ctx_gen_side_quant", side_quant)
        o8_done = False
        if tc:
            o8_done = self._attn_ctx(
                a, pool_layer, q_ctx, lat_ctx, attn_ctx, slots_ctx, tc, step, ctx_rope, dc, o8,
                ctx_last,
            )  # fmt: skip
        if gen and not gen_side:
            core._attn_gen(
                a, pool_layer, q_gen, lat_gen, attn_gen, slots_gen, gen, gen_p, ctx, md,
                gen_kind, gen_layer, False,
            )  # fmt: skip
        if gen_side:
            (torch.cuda.current_stream() if main is None else main).wait_stream(core._gen_stream)
        if not o8_done:
            return None
        # The generation rows' bf16 attention into the rows after the context rows (none on a
        # pure context step: then only the padding scale rows).
        if not side_quant:
            cute_dsl_act_quant_mxf8_rows(attn_gen, *o8, tc, compiled=core._mha_q8.rows)
        return o8

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        core = self.core
        self._check_step_contract(attn_metadata)
        md = attn_metadata
        step = self.step_args(md)
        if core._ctx_host_diet:
            self._diet_tables(step)
            # The stream this forward runs on, read once: torch.cuda.current_stream() costs
            # ~3 us per call and the layers ask for it three times each.
            step["diet_main"] = torch.cuda.current_stream()
        h = core._embed(input_ids) if inputs_embeds is None else inputs_embeds
        rows = h.shape[0]
        # The slot mappings and window vectors are indexed by the prepared token order; a
        # padded input would scatter into stale slots.
        assert rows == md.num_tokens, (rows, md.num_tokens)
        assert rows < _STEP_MAX_TOKENS, (
            f"{rows} tokens in one step overflow the indexer kernel's int32 offsets; "
            f"set max_num_tokens below {_STEP_MAX_TOKENS}"
        )
        ctx, tc = md.num_contexts, md.num_ctx_tokens
        gen = rows - tc
        gen_seqs = md.num_seqs - ctx
        gen_p = gen // gen_seqs if gen_seqs else 1
        skip_ctx = bool(md.skip_indexer_for_ctx_reqs) if tc else True
        skip_gen = bool(md.skip_indexer_for_gen_reqs) if gen else True
        topk_buf = core._topk_buf(md, rows, tc, skip_ctx, skip_gen)
        need_q = not (skip_ctx and skip_gen)
        pos = torch.reshape(position_ids, [-1])
        req_idx = md.req_idx_per_token[:rows]
        attn_out = torch.empty([rows, core.heads * core.v_dim], dtype=h.dtype, device=h.device)
        # Both built once per step and shared by every layer; None takes the stock rope/append
        # op, or the index path of the context attention.
        ctx_rope = self._ctx_rope_step(step, tc) if tc else None
        dc = self._dense_causal_step(
            md, step, tc, skip_ctx, req_idx, h.device, rope_ok=ctx_rope is not None
        )
        # Under `dc` the context rows' pool rows are formed inside the attention kernel, so the
        # conversion runs on the generation rows alone, or not at all.
        slot_lo = tc if dc is not None else 0
        # TRTLLM_GLM52_MIXED_BASE (default 1): with only the generation rows converting (the
        # dense-causal route) at the MTP width, the generation rows take the layer-independent
        # base rows, converted once per fresh selection, and the sparse MLA decode adds each
        # layer's offset, as on a pure decode step: the same pool rows, instead of one global
        # conversion per layer (78 launches per mixed step).
        mixed_base = core._mixed_base and slot_lo == tc and gen > 0 and gen_p == _MLA_ROPE_MTP_WIDTH
        base = None
        # The generation rows' MTP5 atom split metadata, built by the first indexer layer.
        split_memo: dict = {}

        x = core._rmsnorm(h, core._attn[0].norm1)
        x8 = None
        residual = h
        for i in range(core.num_layers):
            layer_id = core._pool_layer_id[i]

            def slots_of(buf, fresh, layer_id=layer_id):
                nonlocal base
                if slot_lo == rows:
                    return None, None, INDEX_KIND_GLOBAL, None
                if mixed_base:
                    if fresh or base is None:
                        base = cute_dsl_req_index_to_base(
                            req_idx[tc:], md.block_table, buf[tc:], md.tokens_per_block,
                            core._stride_factor, compiled=core._req_base_compiled,
                        )  # fmt: skip
                        core._count("mixed_base", True)
                    return None, base, INDEX_KIND_BASE_WITH_LAYER_OFFSET, layer_id
                slots = cute_dsl_req_index_to_global(
                    req_idx[slot_lo:], md.block_table, buf[slot_lo:], md.tokens_per_block,
                    core._stride_factor, layer_id, compiled=core._req_global_compiled,
                )  # fmt: skip
                ctx_s, gen_s = (None, slots) if slot_lo else torch.split(slots, [tc, gen], 0)
                return ctx_s, gen_s, INDEX_KIND_GLOBAL, None

            o8 = self._layer_attention(
                md, step, x, pos, attn_out, i, i, slots_of, topk_buf, rows, tc, gen, gen_p, ctx,
                need_q, skip_ctx, skip_gen, ctx_rope, dc, x8, split_memo=split_memo,
            )  # fmt: skip
            x, x8 = core._mlp_and_close(attn_out, residual, i, rows, o8, main=step.get("diet_main"))
        return x

    #: The row counts the MTP layer's selected rows are padded to: decode cells every op
    #: downstream has (the routed MoE takes powers of two only; anything else it would pad
    #: with torch ops). Above 64 the plan declines and every row is computed.
    MTP_LAST_CELLS = (2, 4, 8, 16, 32, 64)
    #: MTP3 / MTP5 only: 4 / 6 generation rows per request pass 64 selected rows from 17 / 11
    #: generation requests on, so the plan also pads to these (prefill kernels downstream, the
    #: routed MoE's pair at a power of two), while that is still fewer rows than the step's.
    MTP_LAST_WIDE_CELLS = (128, 256)

    def _mtp_last_plan(self, md, step, ctx: int, tc: int, gen: int, layer: int, stride: int):
        """The MTP layer's selected rows on a context step, or None (every row computed).

        The draft head reads the layer's output at each context request's last row and at
        the generation rows only, and everything after the layer's KV write is row-local, so
        attention-to-output can run on those rows: the context last rows (padded with the
        last one up to a cell), then the generation rows. The context queries take the
        dense-causal kernel (`ks = 0`, `ke = cached + len`, request `b`, one `(q, 1)` group
        each), which needs the engine's dense index (no real selection) and the rope kernel's
        append. Host only, sync-free: the metadata's host prefix sums and one pinned upload.
        On B300 the MTP layer's context attention at 8.9k rows goes 590 -> 30 us."""
        core = self.core
        pool_ptrs = step["host_kv_cache_pool_pointers"]
        mapping = step["host_kv_cache_pool_mapping"][layer]
        cells = self.MTP_LAST_CELLS
        if core.max_draft_len > 1:
            cells = cells + self.MTP_LAST_WIDE_CELLS
        r = ctx + gen
        r_pad = next((c for c in cells if c >= r), None)
        why = (
            "selection"
            if not bool(md.skip_indexer_for_ctx_reqs)
            else "block"
            if int(step["tokens_per_block"]) != _fmha_pre_op.DENSE_CAUSAL_BLOCK
            else "pool"
            if int(pool_ptrs[int(mapping[0]), 0]) != step["primary_pool"].data_ptr()
            or int(pool_ptrs[int(mapping[0]), 1]) != 0
            or core._ctx_rope_compiled is None
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
        if why is None and int(ke.max()) > core.idx_topk:
            why = "window"
        if why is not None:
            # Not a stock-op fallback: every row then runs the same kernels. Counted apart
            # from the route counters, by reason.
            core.mtp_last_declines[why] = core.mtp_last_declines.get(why, 0) + 1
            return None
        core._count("mtp_ctx_last", True)
        if r_pad > self.MTP_LAST_CELLS[-1]:
            core._count("mtp_ctx_last_wide", True)
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

    def mtp_forward(
        self,
        embed_tokens,
        all_rank_num_tokens,
        input_ids,
        hidden_states,
        attn_metadata,
        position_ids=None,
        **kwargs,
    ) -> torch.Tensor:
        """One draft step on the MTP module. Its KV lives in the separate draft pool: the
        pool-local layer index and the pool stride are read live, not from the trunk's. On a
        context step attention-to-output runs on the draft head's rows where it can
        (`_mtp_last_plan`); the output's other rows are then not written."""
        core = self.core
        md = attn_metadata
        step = self.step_args(md)
        rows = hidden_states.shape[0]
        ctx, tc = md.num_contexts, md.num_ctx_tokens
        gen = rows - tc
        gen_seqs = md.num_seqs - ctx
        gen_p = gen // gen_seqs if gen_seqs else 1
        j = core.num_layers
        emb = core._embed(input_ids)
        x = core._mtp_front(emb, hidden_states, rows)
        residual = x
        xn = core._rmsnorm(x, core._attn[j].norm1)
        skip_ctx = bool(md.skip_indexer_for_ctx_reqs) if tc else True
        skip_gen = bool(md.skip_indexer_for_gen_reqs) if gen else True
        topk_buf = core._topk_buf(md, rows, tc, skip_ctx, skip_gen)
        live_map = md.host_kv_cache_pool_mapping
        local = md.kv_cache_manager.layer_offsets[j]
        stride = live_map.shape[0] * md.tokens_per_block
        pool_layer = int(live_map[local][1])
        req_idx = md.req_idx_per_token[:rows]
        # The MTP layer's context rows take the trunk's rope/append kernel, at its own pool's
        # layer (`_attn_ctx` falls back to the stock op on any other pool layout), and the
        # index path.
        ctx_rope = self._ctx_rope_step(step, tc) if tc else None
        last = None
        if ctx_rope is not None and core._rows_sel is not None:
            last = self._mtp_last_plan(md, step, ctx, tc, gen, local, stride)

        def slots_of(buf, fresh):
            if last is not None:
                # The context queries read the block table: only the generation rows convert.
                if not gen:
                    return None, None, INDEX_KIND_GLOBAL, None
                slots = convert_req_index_to_global(
                    req_idx[tc:], md.block_table, buf[tc:], md.tokens_per_block, core.idx_topk,
                    stride, pool_layer,
                )  # fmt: skip
                return None, slots, INDEX_KIND_GLOBAL, None
            slots = convert_req_index_to_global(
                req_idx, md.block_table, buf, md.tokens_per_block, core.idx_topk, stride, pool_layer
            )
            ctx_s, gen_s = torch.split(slots, [tc, gen], 0)
            return ctx_s, gen_s, INDEX_KIND_GLOBAL, None

        on_kernel = core.route_hits.get("ctx_rope", 0)
        out_rows = rows if last is None else last.r_pad
        attn_out = torch.empty([out_rows, core.heads * core.v_dim], dtype=x.dtype, device=x.device)
        self._layer_attention(
            md, step, xn, torch.reshape(position_ids, [-1]), attn_out, j, local, slots_of,
            topk_buf, rows, tc, gen, gen_p, ctx, not (skip_ctx and skip_gen), skip_ctx, skip_gen,
            ctx_rope, ctx_last=last,
        )  # fmt: skip
        if tc:
            core._count("mtp_ctx_rope", core.route_hits.get("ctx_rope", 0) > on_kernel)
        if last is None:
            return core._mtp_close(attn_out, residual, rows)
        # The selected rows' residual, the close on those rows, then the rows back in place.
        res_sel = torch.empty(
            [last.r_pad, residual.shape[1]], dtype=residual.dtype, device=x.device
        )
        cute_dsl_rows_select(residual, last.rows, res_sel, core._rows_sel)
        out = core._mtp_close(attn_out, res_sel, last.r_pad)
        full = torch.empty([rows, out.shape[1]], dtype=out.dtype, device=out.device)
        return cute_dsl_rows_select(out, last.rows, full, core._rows_sel, scatter=True)


class DecodeTarget:
    """The specialization: routed to only when there are no context rows.

    Two paths only this target takes, both at the MTP widths they are built for (`mtp_widths`:
    2, and 4 / 6 at MTP3 / MTP5): the specialised MLA rope kernel, and pool rows computed once
    per selection rather than once per layer (the decode kernel adds `layer * 32` itself).
    """

    def __init__(self, core: "GlmMoeDsaTep4ModelingV2") -> None:
        """Hold the core by reference: every operand is read through `self.core` at the call."""
        self.core = core
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks for it; read once here rather than per
        # forward. Each target owns its own flag: the two `step_args` projections differ,
        # so probing one does not prove the other's fields exist.
        self._contract_pending = step_contract_enabled()

    def _check_step_contract(self, md) -> None:
        """Opt-in first-forward fail-fast, run when TRTLLM_MODELING_V2_VALIDATE asks for it:
        the metadata fields this target consumes must exist (they are private trtllm
        surface). Calling `step_args` is the check -- it reads every field this target
        consumes. Everything checked is fixed at engine construction, so once per target
        instance is sound, and it is off in a served engine."""
        if not self._contract_pending:
            return
        try:
            self.step_args(md)
        except AttributeError as exc:
            raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
        self._contract_pending = False

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        """States the two context counts instead of reading them back: this target runs only
        when `num_contexts` is zero."""
        step = _build_step_args(md)
        step["num_contexts"] = 0
        step["num_ctx_tokens"] = 0
        return step

    def _layer_attention(
        self,
        md,
        x,
        pos,
        attn_out,
        i: int,
        pool_layer: int,
        slots_of,
        topk_buf,
        rows,
        gen_p,
        skip_gen,
        specialised_rope,
        split_memo: dict | None = None,
    ):
        core = self.core
        a = core._attn[i]
        ix = core._idx[i]
        # The MTP layer on a draft step after the first, with the engine's index share on: the
        # selection is draft step 0's (`_mtp_share_load`), so the indexer's q side, scorer and
        # top-k do not run; its keys still project and append. Draft step 0 stashes.
        share = ix is not None and i == core.num_layers and core.mtp_index_share
        share_reuse = share and _mtp_share_reuses(md)
        select = not skip_gen and not share_reuse
        # MTP3 / MTP5 wide rows: the qkva projection and the indexer's key / head-weight
        # projection in one GEMM (None: the two GEMMs below).
        fold = core._qkva_ix_fold(x, i, rows) if ix is not None else None
        # And q_b with the indexer's wq_b in one GEMM where the step runs the indexer.
        qbix = core._qb_ixqb_runner(i, rows) if ix is not None and select else None
        q, q_a, latent, _ = core._qkv(
            x, a, q_b=qbix is None, qkva=None if fold is None else fold[0]
        )
        ixq = None
        if qbix is not None:
            q, ixq = core._qb_ixqb(q_a, i, qbix)
        if ix is not None:
            k_pool = md.kv_cache_manager.get_indexer_k_cache_buffers(i)
            q_fp8, weights, k_pre = core._indexer_project(
                x, q_a, pos, ix, rows, select, kw=None if fold is None else fold[1:],
                ixq=ixq, kc=(md, k_pool),
            )  # fmt: skip
            if k_pre is not None:  # else written with the q side (`_ix_qk_handle`)
                core._indexer_write_k(md, k_pool, k_pre, rows)
            if select:
                core._indexer_decode(
                    md, k_pool, q_fp8, weights, topk_buf, 0, rows // gen_p, gen_p, split_memo
                )
            elif share_reuse and not skip_gen:
                core._count("mtp_ix_share", True)
                _mtp_share_load(md, topk_buf, rows)
            if share and not share_reuse and core._mtp_share_stash and md.in_mtp_draft_loop:
                _mtp_share_store(md, topk_buf, 0, 0, rows // gen_p, gen_p)
        slots, slot_kind, slot_layer = slots_of(topk_buf, ix is not None)
        # A trunk layer whose o_proj takes the decode MXF8 GEMM (or at MTP3 / MTP5 decode
        # rows above 64 the wide one): the V-B writes its operand.
        o8_side = i < core.num_layers and (
            core._dec8_takes("o_proj", attn_out, a.o_dec8)
            or core._dec8w_takes("o_proj", attn_out, a.o_dec8)
        )
        return core._attn_gen(
            a, pool_layer, q, latent, attn_out, slots, rows, gen_p, 0, md, slot_kind, slot_layer,
            specialised_rope, o8_side,
        )  # fmt: skip

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        core = self.core
        self._check_step_contract(attn_metadata)
        md = attn_metadata
        h = core._embed(input_ids) if inputs_embeds is None else inputs_embeds
        rows = h.shape[0]
        assert rows == md.num_tokens, (rows, md.num_tokens)
        gen_p = rows // md.num_seqs
        skip_gen = bool(md.skip_indexer_for_gen_reqs)
        topk_buf = core._topk_buf(md, rows, 0, True, skip_gen)
        pos = torch.reshape(position_ids, [-1])
        req_idx = md.req_idx_per_token[:rows]
        attn_out = torch.empty([rows, core.heads * core.v_dim], dtype=h.dtype, device=h.device)
        specialised_rope = gen_p in core.mtp_widths
        # Base rows at every width with its own decode attention handle that takes them; the
        # rope flag is a separate choice.
        base_rows = (
            gen_p in core.mtp_widths
            and INDEX_KIND_BASE_WITH_LAYER_OFFSET in core._fmha_dec_handle(gen_p).index_kinds
        )
        base = None
        # MTP3 / MTP5: the union decode attention takes the selection itself.
        union = (
            core._fmha_union is not None
            and _MLA_ROPE_MTP_WIDTH < gen_p <= core._fmha_union.max_sq
            and md.num_seqs in core._fmha_union.plans
        )
        # The MTP5 atom split's indexer metadata, built by the first indexer layer for all.
        split_memo: dict = {}

        x = core._rmsnorm(h, core._attn[0].norm1)
        residual = h
        for i in range(core.num_layers):
            layer_id = core._pool_layer_id[i]

            def slots_of(buf, fresh, layer_id=layer_id):
                nonlocal base
                if union:
                    return _UnionSlots(buf, req_idx), _UNION_POSITIONS, layer_id
                if not base_rows:
                    rows_ = cute_dsl_req_index_to_global(
                        req_idx, md.block_table, buf, md.tokens_per_block, core._stride_factor,
                        layer_id, compiled=core._req_global_compiled,
                    )  # fmt: skip
                    return rows_, INDEX_KIND_GLOBAL, None
                # Layer-independent rows, recomputed only when a layer selected afresh.
                if fresh or base is None:
                    base = cute_dsl_req_index_to_base(
                        req_idx, md.block_table, buf, md.tokens_per_block, core._stride_factor,
                        compiled=core._req_base_compiled,
                    )  # fmt: skip
                return base, INDEX_KIND_BASE_WITH_LAYER_OFFSET, layer_id

            o_pre = self._layer_attention(
                md, x, pos, attn_out, i, i, slots_of, topk_buf, rows, gen_p, skip_gen,
                specialised_rope, split_memo,
            )  # fmt: skip
            x, _ = core._mlp_and_close(attn_out, residual, i, rows, o_pre=o_pre, decode=True)
        return x

    def mtp_forward(
        self,
        embed_tokens,
        all_rank_num_tokens,
        input_ids,
        hidden_states,
        attn_metadata,
        position_ids=None,
        **kwargs,
    ) -> torch.Tensor:
        """One draft step, decode rows only. Pool rows go through the stock conversion against
        the draft pool's live mapping. Run eagerly (a context or mixed step's draft loop), its
        row counts without a decode cell run on the next cell up (`_draft_eager`)."""
        core = self.core
        core._draft_eager = core.max_draft_len > 1 and not torch.cuda.is_current_stream_capturing()
        try:
            return self._mtp_step(input_ids, hidden_states, attn_metadata, position_ids)
        finally:
            core._draft_eager = False

    def _mtp_step(self, input_ids, hidden_states, attn_metadata, position_ids):
        """`mtp_forward`'s step."""
        core = self.core
        md = attn_metadata
        rows = hidden_states.shape[0]
        gen_p = rows // md.num_seqs
        j = core.num_layers
        emb = core._embed(input_ids)
        x = core._mtp_front(emb, hidden_states, rows)
        residual = x
        xn = core._rmsnorm(x, core._attn[j].norm1)
        skip_gen = bool(md.skip_indexer_for_gen_reqs)
        topk_buf = core._topk_buf(md, rows, 0, True, skip_gen)
        live_map = md.host_kv_cache_pool_mapping
        local = md.kv_cache_manager.layer_offsets[j]
        stride = live_map.shape[0] * md.tokens_per_block
        pool_layer = int(live_map[local][1])
        req_idx = md.req_idx_per_token[:rows]

        def slots_of(buf, fresh):
            rows_ = convert_req_index_to_global(
                req_idx, md.block_table, buf, md.tokens_per_block, core.idx_topk, stride, pool_layer
            )
            return rows_, INDEX_KIND_GLOBAL, None

        attn_out = torch.empty([rows, core.heads * core.v_dim], dtype=x.dtype, device=x.device)
        self._layer_attention(
            md, xn, torch.reshape(position_ids, [-1]), attn_out, j, local, slots_of, topk_buf,
            rows, gen_p, skip_gen, core._mtp_rope_specialised(gen_p),
        )  # fmt: skip
        return core._mtp_close(attn_out, residual, rows, decode=True)


class MTPLayer:
    """The checkpoint's MTP module at layer `num_hidden_layers`, replayed once per draft step.

    It differs from a trunk layer in three ways: an `eh_proj` front end mixing the next token's
    embedding into `h`, bf16 experts, and its own head norm. The runtime feeds the returned
    tensor back in as the next draft step's hidden state, so above one draft token it is
    returned head-normalized, as stock's MTP layer returns it (`_mtp_out_normed`); MTP1 never
    feeds back and keeps the norm in `shared_head`.
    Not an `nn.Module`: every weight it reads lives in the core's ParameterDict.
    """

    def __init__(self, core: GlmMoeDsaTep4ModelingV2, logits_processor) -> None:
        self.core = core
        self.logits_processor = logits_processor

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)

    def shared_head(
        self, hidden_states, lm_head, attn_metadata, return_context_logits
    ) -> torch.Tensor:
        """The module's own norm (a distinct parameter from `model.norm`; already applied by
        `_mtp_close` when `_mtp_out_normed`), then the trunk's `lm_head`."""
        core = self.core
        if core._mtp_out_normed:
            x = hidden_states
        else:
            x = core._rmsnorm(hidden_states, core._mtp["head_norm"])
        if return_context_logits:
            return core.logits(x, lm_head, self.logits_processor, attn_metadata)
        return self.logits_processor.forward(x, lm_head, attn_metadata, return_context_logits)

    def forward(
        self, embed_tokens, all_rank_num_tokens, input_ids, hidden_states, attn_metadata, **kwargs
    ):
        """Route the draft step by the phase this call's metadata shows: the worker rewrites the
        metadata in place between draft steps, so one engine step's draft calls can cross from
        prefill to decode."""
        return self.core._select_target(attn_metadata).mtp_forward(
            embed_tokens, all_rank_num_tokens, input_ids, hidden_states, attn_metadata, **kwargs
        )


class DraftModel:
    """The container the spec worker reaches the drafter through.

    GLM-5.2's MTP module carries no `embed_tokens` or `shared_head.head` of its own, so both
    are the trunk's. `embed_tokens` resolves through the core on every read: this is built
    while parameters are still meta tensors."""

    def __init__(self, core: GlmMoeDsaTep4ModelingV2, lm_head, logits_processor) -> None:
        self.core = core
        self.mtp_layers = [MTPLayer(core, logits_processor)]
        self.lm_head = lm_head

    @property
    def embed_tokens(self) -> torch.Tensor:
        return self.core.w["embed"]


class ModelingV2Glm52Nvfp4Sm103Tep4(
    DecoderModelForCausalLM[GlmMoeDsaTep4ModelingV2, PretrainedConfig]
):
    # The DSA metadata with the draft loop's one-launch refresh (`spec_worker.py`).
    runtime_hooks = RuntimeHooks(metadata_extensions=(Glm52Tep4DSAMetadata,))

    def __init__(self, model_config: ModelConfig):
        cfg = model_config.pretrained_config
        super().__init__(
            GlmMoeDsaTep4ModelingV2(model_config),
            config=model_config,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )
        self.spec_config = getattr(model_config, "spec_config", None)
        self.draft_model = None
        self.spec_worker = None
        if self.spec_config is not None:
            self.draft_model = DraftModel(self.model, self.lm_head, self.logits_processor)
            # The fourth argument matters: the engine allocates a separate draft KV pool, and a
            # worker built without it runs the draft step outside the pool swap.
            self.spec_worker = get_spec_worker(
                self.spec_config,
                model_config,
                model_config.mapping,
                should_use_separate_draft_kv_cache(self.spec_config),
            )
            if type(self.spec_worker) is MTPEagleWorker:
                # The target's worker in place of the built-in one, with the same draft KV
                # cache choice: the built-in worker with the greedy picks on the argmax.
                self.spec_worker = Glm52MTPEagleWorker(
                    self.spec_config,
                    model_config,
                    self.spec_worker.use_separate_draft_kv_cache,
                    mapping=model_config.mapping,
                    core=self.model,
                )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        return_context_logits: bool = False,
        spec_metadata=None,
        lora_params: dict | None = None,
        **kwargs,
    ):
        """One engine step.

        `resource_manager` must stay inside `**kwargs`, and `position_ids` must be handed on in
        the engine's `[1, T]` shape: the worker squeezes it itself."""
        if self.spec_worker is None:
            return super().forward(
                attn_metadata,
                cast(torch.IntTensor, input_ids),
                position_ids,
                inputs_embeds,
                return_context_logits,
                spec_metadata,
                lora_params,
                **kwargs,
            )
        hidden = self.model(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            lora_params=lora_params,
            **kwargs,
        )
        logits = self.model.logits(
            self.model.gather_rows(hidden, spec_metadata.gather_ids),
            self.lm_head,
            self.logits_processor,
            attn_metadata,
        )
        return self.spec_worker(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden,
            logits=logits,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=self.draft_model,
            resource_manager=kwargs.get("resource_manager"),
        )

    def load_weights(self, weights, *args, **kwargs):
        _weights.MODEL_WEIGHTS.load(self, weights)

    def post_load_weights(self):
        super().post_load_weights()
        self.model.post_load_weights()
        # The shell's `lm_head` is the one module of the forward the core does not own: its
        # local GEMM compiles here, against the loaded weight.
        self.model.build_logits_head(self.lm_head)
