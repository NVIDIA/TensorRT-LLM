# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: MiniMax-M3 NVFP4 / sm_107 / tp2 -- self-contained modeling code.

MiniMax-M3's language model: 60 layers, hidden 6144, 64 query heads and 4 KV heads of dim 128. Layers 0-2 are dense
attention with a dense SwiGLU-OAI MLP; layers 3-59 are MSA block-sparse attention (an index branch scores 128-token
blocks and the top 16 are attended) with a MoE of 128 routed experts (top 4, sigmoid scores with a correction bias,
renormalized and scaled by 2) plus one shared expert. The checkpoint is a modelopt mixed-precision export: NVFP4 routed
experts, MXFP8 projections, fp32 router, bf16 embedding and head; the KV cache is NVFP4 for the sparse layers and FP8
for the dense ones.

``tp2`` is two Rubin GPUs: head-split attention (32 query heads, two KV heads and two index heads per rank, one
index-K head) and tensor-parallel experts (every expert on every rank, intermediate 1536 per rank), no attention DP.
Speculative decoding, when configured, is one-model Eagle3 (draft length 3): the BF16 drafter (``M3Eagle3Draft``) and
the BF16 target head, both at checkpoint precision; decode steps of up to 2 requests verify and draft through fused
kernels at that precision (``M3Eagle3Worker``).

**The core** (``MiniMaxM3Tp2ModelingV2``) holds the weights (``weights.MODEL_WEIGHTS``) and the runtime resources the
steps use -- one MSA attention backend per layer, the TP all-reduce communicator, the RoPE table, the MXFP8 GEMM
dispatch state the engine tunes -- and dispatches each step, on the host, from its context count alone
(``_select_target``):

* ``DecodeTarget`` takes every pure generation step, of 1..``WIDE_MAX_TOKENS`` rows (up to 48 requests, each one token
  or a draft verify of 1 + 3 rows). Up to ``DECODE_MAX_TOKENS`` tokens with 1-D input ids it runs the decode-step
  kernels and picks each layer boundary's collective by the step's token bucket (``DECODE_PLANS``: tiny 1..8, b16
  9..16, b32 17..32); it runs its own generic layer for the first layer when the fused embedding does not take the
  step, and for every layer of a wider step, with the wide cells (the 48-row generic boundaries and MC finalize, the
  tensor-core router, the Triton MoE finalize, the 33..48-row GEMVs, the chunked greedy verify). The fast-path cells
  go up to 128 rows; a 129..192-row step takes the stock-exact paths.
* ``PrefillTarget`` takes every step with context rows: prefill chunks and mixed steps, every layer in its generic
  form over the MSA backend.

The two targets share no body: each owns its layer, attention and MoE compositions and its ``step_args``, the one
projection of the step metadata its forward reads. What both call are the module-level op adapters, each picking the
kernel for one call from shapes and layouts, which read no step metadata themselves.

Every fast path the decode step can take, the rows each covers (``FAST_PATHS``, each paired with the entry whose
``CELLS`` certify it) and what a step that cannot take one does are below: **uncovered** (outside every covered
range: log once, run the generic path) or **missed** (inside a covered range but a precondition is missing: raise
``FastPathUnavailable`` under ``modeling_v2='require'``, else log once and fall back). A row range is empty unless a
landed cell has certified it at the TP2 geometry on Rubin. Every fast path a GEN rank takes prints a rank-tagged
engagement line once per (path, rows) (``taken``). Kernel modules report through ``ops/m3_report``, which this
module wires to the same policy at import.

**Hand-offs between kernels.** A producer that already holds an activation in registers writes its MXFP8 form for the
next MXFP8 projection: the decode boundary all-reduce for the next layer's QKV, the sparse-decode merge and the
long-context decode for the o_proj. It publishes it in ``ops/m3_prequant.SLOT`` keyed by the BF16 tensor; the next
MXFP8 projection on a decode graph takes it, and every take clears the slot. The decode boundaries' push targets (the
o_proj / dense-down / shared-expert GEMVs and the MoE engine writing straight into the boundary all-reduce's exchange)
are passed explicitly from the layer to the producer and back.

Compiled kernels (the MoE engine, the decode GEMVs with their push variants, the index scorer, the long-row select,
the full-context decode plans) are built before the executor's warmup; a forward never compiles one, and one it finds
missing is a miss. Other state a fast path needs (multicast workspaces, push descriptors) is built on the eager decode
steps; a CUDA-graph capture only reads it, so a capture that finds it missing is a miss (``capture_missed``).

**What this target asserts rather than adapts**: SM 10.7, the topology and checkpoint (``routing.py``); the
MiniMax-M3 MSA sparse attention runtime with the fused five-way QKV/index projection, an FP8 indexer and an FP8 or
NVFP4 KV cache, no torch.compile and no piecewise prefill graphs (``_validate_runtime``); text-only input (the vision
tower is not built and a step carrying image input raises).
"""

from __future__ import annotations

import atexit
import copy
import dataclasses
import os
import threading
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
)

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import (
    ModelingV2Mode,
    resolved_mode,
    step_contract_enabled,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.fused_qk_norm_rope import (
    fused_qk_norm_rope,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (
    fp4_block_scale_moe_runner,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (
    flashinfer_fused_add_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)
from tensorrt_llm._torch.attention.backends import AttentionMetadata
from tensorrt_llm._torch.attention.backends.fmha.msa_prefill import _aligned_nvfp4_dequant_scales
from tensorrt_llm._torch.attention.backends.interface import PositionalEmbeddingParams, RopeParams
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import MiniMaxM3MsaSparseAttention
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
    MSA_REQUIRED_TOPK,
    msa_paged_kv,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.paged_cache import (
    fp8_e4m3_saturate,
)
from tensorrt_llm._torch.attention.backends.utils import create_attention
from tensorrt_llm._torch.custom_ops.torch_custom_ops import (
    FlashInferMXFP8GemmRunner,
    MXFP8QuantizeRunner,
)
from tensorrt_llm._torch.distributed import AllReduce, AllReduceFusionOp, AllReduceParams
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_speculative import (
    _ensure_draft_vocab_size,
    _get_requested_draft_moe_backend,
    _set_draft_kv_cache_quant_algo,
    _slice_spec_position_ids,
)
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.modules.embedding import LMHead
from tensorrt_llm._torch.modules.linear import (
    MXFP8LinearMethod,
    TensorParallelMode,
    flashinfer_mxfp8_autotune_active,
    flashinfer_mxfp8_decode_graph_capture_active,
)
from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor
from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel
from tensorrt_llm._torch.moe.fused_moe.routing import RoutingMethodType
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import (
    eager_on_graph,
    is_in_breakable_cuda_graph,
)
from tensorrt_llm._torch.pyexecutor.h2d_cache import copy_h2d_if_changed
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManager
from tensorrt_llm._torch.pyexecutor.runtime_hooks import MsaRuntimePolicy, RuntimeHooks
from tensorrt_llm._torch.speculative import SpecMetadata
from tensorrt_llm._torch.speculative.eagle3 import Eagle3OneModelWorker
from tensorrt_llm._torch.speculative.interface import (
    _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
    _FORCE_ACCEPT_RNG_POOL_SIZE,
    _FORCE_ACCEPT_RNG_SLOT_STRIDE,
    should_use_separate_draft_kv_cache,
)
from tensorrt_llm._torch.utils import ActType_TrtllmGen, get_model_extra_attrs
from tensorrt_llm._utils import mpi_rank, prefer_pinned
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

from . import weights as _weights
from .ops import m3_draft_step as m3_draft_step_kernel
from .ops import (
    m3_drafter_attn,
    m3_fold_tail,
    m3_mc_allreduce,
    m3_prequant,
    m3_report,
    m3_spec_tail,
)
from .ops import m3_fullctx_decode as m3_fullctx_decode_kernel
from .ops import m3_index_score as _index_score_kernel
from .ops import m3_sampled_accept as m3_sampled_accept_kernel
from .ops import m3_shard_accept as m3_shard_accept_kernel
from .ops import msa_kv_update as _msa_kv_update_kernel
from .ops import triton_sparse_decode as _sparse_decode_kernel
from .ops.activation.m3_silu_and_mul import m3_silu_and_mul
from .ops.attention.m3_draft_attention import m3_draft_attention
from .ops.attention.m3_fp8_indexer_qk_norm_rope import m3_fp8_indexer_qk_norm_rope
from .ops.attention.m3_fullctx_decode import m3_fullctx_decode
from .ops.attention.m3_index_score import m3_index_score
from .ops.attention.m3_msa_attention import m3_msa_attention
from .ops.attention.m3_producer import m3_producer
from .ops.attention.m3_qk_norm_rope_fp8 import m3_qk_norm_rope_fp8
from .ops.attention.m3_qkv_insert import m3_qkv_insert
from .ops.attention.m3_select_blocks import m3_select_blocks
from .ops.attention.m3_select_long import m3_select_long
from .ops.attention.m3_select_long_dsl import m3_select_long_dsl
from .ops.attention.m3_sparse_decode import m3_sparse_decode
from .ops.attention.m3_sparse_decode_lc import m3_sparse_decode_lc
from .ops.comm.m3_allreduce import m3_allreduce
from .ops.comm.m3_mc_allreduce_norm import m3_mc_allreduce_norm
from .ops.comm.m3_mc_finalize_allreduce_norm import m3_mc_finalize_allreduce_norm
from .ops.embedding.m3_embedding import m3_embedding
from .ops.gemm import m3_ugpu_head as m3_ugpu_head_op
from .ops.gemm.m3_bf16_linear import m3_bf16_linear
from .ops.gemm.m3_dense_swiglu import ALPHA as DENSE_ALPHA
from .ops.gemm.m3_dense_swiglu import BETA as DENSE_BETA
from .ops.gemm.m3_dense_swiglu import LIMIT as DENSE_LIMIT
from .ops.gemm.m3_dense_swiglu import m3_dense_swiglu
from .ops.gemm.m3_draft_lowm import ROWS as DRAFT_LOWM_TACTIC_ROWS
from .ops.gemm.m3_draft_lowm import m3_draft_lowm, tactic_for
from .ops.gemm.m3_fp32_linear import m3_fp32_linear
from .ops.gemm.m3_mx_gemv import (
    DENSE_DOWN,
    DENSE_GATE_UP,
    O_PROJ,
    QKV,
    QKV_SPARSE,
    SHARED_DOWN,
    SHARED_GATE_UP,
    m3_mx_gemv,
    served_max_rows,
)
from .ops.gemm.m3_mx_gemv_bf16 import m3_mx_gemv_bf16
from .ops.gemm.m3_mxfp8_gemm import m3_mxfp8_gemm
from .ops.gemm.m3_mxfp8_linear import m3_mxfp8_linear
from .ops.gemm.m3_ugpu_head import m3_ugpu_head
from .ops.long_select import long_select_blocks
from .ops.m3_moe_finalize import m3_moe_finalize
from .ops.moe.m3_moe_engine import M3MoeState, ar_push_desc, m3_moe_engine
from .ops.moe.m3_router_tc import m3_router_tc
from .ops.mx_gemv import op as mx_gemv
from .ops.norm.m3_embedding_firstnorm import DecodeEmbedding, m3_embedding_firstnorm
from .ops.spec.m3_capture_norm import m3_capture_norm
from .ops.spec.m3_chunk_norm import m3_chunk_norm
from .ops.spec.m3_draft_input import m3_draft_input
from .ops.spec.m3_draft_silu import m3_draft_silu
from .ops.spec.m3_draft_step import m3_draft_step
from .ops.spec.m3_force_vec import m3_force_vec, m3_force_vec_req
from .ops.spec.m3_sampled_accept import m3_sampled_accept
from .ops.spec.m3_shard_accept import m3_shard_accept
from .ops.spec.m3_spec_offsets import m3_spec_offsets
from .ops.spec.m3_step_prologue import m3_step_prologue

_SM = (10, 7)
# --------------------------------------------------------------------------------------------------------------------
# Row ranges
# --------------------------------------------------------------------------------------------------------------------

# The 1..8-row decode kernels' step bound: 2 requests verifying 1 + 3 drafts (the disaggregated GEN worker's batch).
TINY_MAX_TOKENS = 8  # capacity: fixed: the 1..8-row decode kernels' step bound
# Batch-4 plan: steps of 9..16 rows (3-4 requests verifying 1 + 3 drafts) also take the decode layer, on the "b16"
# plan below. Every fast path keeps its own rows (each up to TINY_MAX_TOKENS unless it names its 16-row rows), so a
# path not raised for 16 rows declines to the generic kernel inside the decode layer. False: every step above
# TINY_MAX_TOKENS rows runs the generic layer.
B4_DECODE = True
B4_MAX_TOKENS = 16 if B4_DECODE else TINY_MAX_TOKENS
# Batch-8 plan: steps of 17..32 rows (5-8 requests verifying 1 + 3 drafts) also take the decode layer, on the "b32"
# plan below; every path keeps the rows it names (a path raised only to 16 rows declines a 17..32-row step). False:
# the b16 plan's 16-row cap.
B8_DECODE = B4_DECODE
DECODE_MAX_TOKENS = 32 if B8_DECODE else B4_MAX_TOKENS
# The widest decode step this target takes (aggregated serving, up to 48 requests x (1 + 3 drafts) verify rows): decode
# rows 1..WIDE_MAX_TOKENS; the fast-path cells go up to 128 rows and a 129..192-row step takes the stock-exact paths
# (the Triton sparse-decode producer + merge, the generic GEMM). Steps past DECODE_MAX_TOKENS run the generic layer
# with the wide cells below (GENERIC_MC_*, ROUTER_TC_GEN, MOE_TRITON_FINALIZE, the W-G1 GEMVs, the wide verify).
WIDE_MAX_TOKENS = 192  # capacity: fixed: 48 requests x (1 + 3 drafts)


@dataclass(frozen=True)
class DecodePlan:
    """The layer-boundary collectives of one decode token bucket (GB300 tp4 measurements in the comments)."""

    name: str
    max_tokens: int
    # The attention boundary's multicast all-reduce + residual + RMSNorm (5.05 vs 5.27 us native at 1 token).
    attn_mc_ar: bool
    # The same kernel at the MoE boundary with the shared + routed add folded in: wins up to 48 tokens.
    moe_mc_ar: bool
    # The MoE-boundary all-reduce also emits the next QKV activation in MXFP8 for the decode GEMV / GEMM.
    qkv_prequant: bool


# tiny: every step up to 8 tokens (c1..c8 without speculation, the c1 verify, every draft pass); small: the c2
# verify; verify: the c4 / c8 verifies. Each is one multicast all-reduce bucket (ops/m3_mc_allreduce.BUCKETS).
# The multicast all-reduce + residual + RMSNorm at both boundaries of every TP2 decode step (one 8-token bucket), the
# MoE boundary's also publishing the next QKV activation in MXFP8. SM107 TP2 hop: 3.5..4.4 us at 1..8 rows against
# 8.7..9.5 us for the stock AUTO (NCCL) all-reduce + fused residual RMSNorm, and 13.3..14.0 us at the MoE boundary
# (stock add + all-reduce + mxfp8_quantize); bitwise equal to stock (ops/comm/m3_mc_allreduce_norm).
# False puts both boundaries back on the built-in all-reduce.
# On by default since r4.2: the intermittent illegal memory access first seen with the GEMM/AR stack on was the
# built-in Triton sparse decode's (fixed in r4); with the stack on, GSM8K 4-in-flight ran crash-free on r4.1.
MC_AR_TP2 = True
# B3: the 9..32-row steps' attention and MoE boundaries on the same multicast kernel (the 16-token bucket at 9..16 rows,
# epilogue option 15; the 48-token bucket at 17..32: on TP2 the row kernel launches one CTA per row, so the bucket
# only sizes the exchange slots); bitwise equal to stock at every row count (the op's 9..32-row cells). False: the
# built-in all-reduce on those steps.
B4_MC_AR = MC_AR_TP2
PLAN_MC_AR_MAX_TOKENS = DECODE_MAX_TOKENS if B4_MC_AR else TINY_MAX_TOKENS
# On those steps the MoE (past the engine's 8 rows) returns its unfinalized expert outputs and the multicast finalize
# kernel does the finalize + all-reduce + residual + RMSNorm in one launch (its cells cover 9..48 rows bitwise)
# instead of the routed finalize then the boundary. False: the routed finalize, then the boundary.
B4_MC_FINALIZE = B4_MC_AR
MC_FINALIZE_MIN_TOKENS = 9  # capacity: fixed: past the MoE engine's 8 rows
# B7 (SpecTail TR-5b): on the 9..32-row steps the boundary all-reduce (and the MoE's multicast finalize) also publishes
# the next QKV's MXFP8 activation (its quant cells at 9..32 rows are bitwise mxfp8_quantize), so QKV takes the MXFP8
# GEMV on the producer-quantized x (795376: QKV at 16 rows 4.78 vs 2.59 us/call at 8; b8 census 794573: the generic
# MXFP8 GEMMs 1,196 + quantize 256 us/step at c8 vs 537 for the GEMVs at c2).
B4_QKV_PREQUANT = B4_MC_AR
B8_QKV_PREQUANT = B4_MC_AR
DECODE_PLANS = (
    (
        DecodePlan(
            "tiny",
            TINY_MAX_TOKENS,
            attn_mc_ar=MC_AR_TP2,
            moe_mc_ar=MC_AR_TP2,
            qkv_prequant=MC_AR_TP2,
        ),
    )
    # b16: the 9..16-row steps; their boundaries on the multicast all-reduce's 16-token bucket.
    + (
        (
            DecodePlan(
                "b16",
                B4_MAX_TOKENS,
                attn_mc_ar=B4_MC_AR,
                moe_mc_ar=B4_MC_AR,
                qkv_prequant=B4_QKV_PREQUANT,
            ),
        )
        if B4_MAX_TOKENS > TINY_MAX_TOKENS
        else ()
    )
    # b32: the 17..32-row steps; their boundaries on the 48-token bucket.
    + (
        (
            DecodePlan(
                "b32",
                DECODE_MAX_TOKENS,
                attn_mc_ar=B4_MC_AR,
                moe_mc_ar=B4_MC_AR,
                qkv_prequant=B8_QKV_PREQUANT,
            ),
        )
        if DECODE_MAX_TOKENS > B4_MAX_TOKENS
        else ()
    )
)

# The dense layers' decode attention (trtllm-gen, msa_decode.dense_decode_max_seq_len): the max_seq_len it is handed
# per request count. It only sizes the multi-CTA KV split (every CTA still covers its share of the actual length; the
# result changes only by the split-merge order), and the model maximum over-splits AgentX-length caches at one
# request: GB300 TP4, 16 heads, FP8 KV, 64K context: 15.1 us at 16K vs 28.6 us at 1M. Two requests are neutral (the
# grid is unchanged) and keep the step's bound.


# TP2 skeleton: no cap (the built-in split) until a TP2 cap is measured and certified.
DENSE_DECODE_MAX_SEQ_LEN_BY_BATCH: Mapping[int, int] = MappingProxyType({})

# The dense layers' decode attention on PrimTS split-KV decode (ops/m3_fullctx_decode) instead of the trtllm-gen
# call, per (query rows per request, requests): the Q tile (``config``) and the split-KV fanout (``splits``) with the
# standalone reducer. Rubin TP2, FP8 KV, 165k context, graph-us per call (stock trtllm-gen in brackets): (4, 1) 18.8
# (29.3), (1, 1) 15.7 (28.0), (4, 2) 24.5 (29.3), (1, 2) 20.2 (28.1). The stock split is capped at 16 cluster CTAs
# per (request, KV head), i.e. 32 of 212 CTAs at one request. At one request (the E2E batch) the K/V stream is
# latency-bound, so the b1 recipes keep 12 K/V stages in flight (the load pipeline only; the same bits): locked
# clock, cold L2, 165k, (4, 1) 22.6 vs 26.9 and (1, 1) 20.7 vs 23.0 µs with the default 4 stages.
FULLCTX_DECODE_RECIPES: Mapping[Tuple[int, int], Mapping[str, Any]] = MappingProxyType(
    {
        (4, 1): MappingProxyType(
            {
                "config": {"tile_size_q": 32, "groups_tokens_heads_q": True, "kv_stages": 12},
                "splits": 48,
            }
        ),
        (1, 1): MappingProxyType({"config": {"kv_stages": 12}, "splits": 72}),
        (4, 2): MappingProxyType(
            {"config": {"tile_size_q": 32, "groups_tokens_heads_q": True}, "splits": 24}
        ),
        (1, 2): MappingProxyType({"config": {}, "splits": 40}),
    }
)
# False puts the dense layers back on the trtllm-gen call at every shape.
FULLCTX_DECODE_ENABLED = True
# The dense layers' FP8 q/k/v (decode steps of the hybrid cache): the TP2 op rounds Q/K to BF16 and then to E4M3 in
# its store, the bits of the BF16 fused_qk_norm_rope output cast to FP8. False runs that BF16 kernel and a separate cast
# (the same bits, one more kernel).
DENSE_QKV_FP8_VIA_BF16 = True
# On a full-context decode step that op also stores the new K/V rows into the layer's paged pool (the bytes
# write_msa_phase_kv's indexed copies store, one launch instead of about ten). False writes them with
# write_msa_phase_kv.
DENSE_KV_INSERT_FUSED = True
# The engagement line of that fused store (dispatch.taken, one per row count and rank).
DENSE_KV_INSERT_PATH = "m3_dense_kv_insert"
# The drafter MLP's C++ SwiGLU (DRAFT_SILU_CPP, spec_worker._m3_draft_mlp): its own taken / not-taken lines.
DRAFT_SILU_PATH = "m3_draft_silu"
# The split-KV merge on the BF16 op (minimax_m3_sm107_tp2_fullctx_reduce) instead of PrimTS's FP16 reducer and a cast:
# Rubin, 165k, graph-us per call (1, 1) 14.5 vs 18.5, (4, 1) 17.3 vs 21.3, (1, 2) 18.5 vs 23.1, (4, 2) 23.3 vs 27.5.
FULLCTX_DECODE_OWN_REDUCER = True
# The K/V length bound the full-context plans are compiled for; a step whose cache length bound is longer is a miss. It
# covers the TP2 AgentX config's max_seq_len (1 << 20) plus the speculative rows the cache reserves past it.
FULLCTX_DECODE_MAX_KV_LEN = 1 << 21  # capacity: fixed: the full-context plans' compiled K/V bound
# The Eagle3 drafter layer's generation phase on the same full-context decode (ops/m3_drafter_attn: the in-tree
# RoPE + FP8 K/V append, then the pinned PrimTS recipe) instead of thop.attention's in-tree runner, which caps its
# split at 8 CTAs per (request, KV head) on SM 10.7: 45-46 us per draft pass at 165k.
DRAFTER_FULLCTX_ENABLED = True
# On those passes the drafter's RoPE, FP8 query and FP8 K/V append run as one launch (the op
# minimax_m3_sm107tp2_drafter_rope_fp8_kv, a copy of the certified TP4/TP8 kernel at the TP2 heads: the in-tree
# preprocess's GPT-NeoX rotation and unit-scale E4M3 store, and the torch E4M3 cast of its BF16 query) instead of
# thop.trtllm_gen_generation_preprocess and the cast (3.6 vs 1.6 us per pass, 792311).
DRAFTER_ROPE_FUSED = True
# The engagement line of that launch (dispatch.taken, one per row count and rank).
DRAFTER_ROPE_PATH = "m3_drafter_rope"
# How the rotation's products are summed, as the in-tree preprocess build sums them (0: two products then the add,
# 1: fma(cos, x, sin * pair), 2: fma(sin, pair, cos * x)); the op's bitwise test pins it.
DRAFTER_ROPE_MODE = 1
# thop.trtllm_gen_generation_preprocess builds its workspace token map only for several requests, yet a one-request
# multi-token generation (the drafter's first pass at one request, any stock trtllm-gen fallback) places its rows by
# that map: without a rebuild it applies the last multi-request call's map (rows on another request's length and pages,
# token-0 drafts). ops/m3_tokmap_guard writes the one-request map (0, t) into the op's map slice right before such
# calls: what the build writes (padded rows included), so bitwise the op with a fresh map. False:.
PREPROCESS_TOKMAP_PATH = "m3_preprocess_tokmap"

# The sparse layers' NVFP4 QKV + indexer producer (ops/attention/m3_producer, every step its geometry fits) triggers its
# dependent kernel on entry, so the index scorer launches and streams its pre-wait index-K prefetch while the producer
# runs (its outputs are read only after the scorer's own grid wait). Rubin TP2, 57-layer PDL chain behind a 21 MB
# pre-wait L2 prefetch (the 165K scorer's shape): 3.59 (native) -> 1.93 us per layer at 4 rows, 3.64 -> 2.00 at 8.
PRODUCER_EARLY_TRIGGER = True
# The producer also reads its per-step inputs (position ids, cache slots, RoPE rows) before its grid wait, leaving only
# the packed QKV row after it. Their writers are host-to-device copies before the replay and plain-launched kernels
# earlier in the step (the input preprocessing, msa_kv_update or its torch fallback), so they are final before the
# producer can start; a PDL-launched writer of any of them would have to turn this off. The call site's
# position_ids.reshape(-1).contiguous().to(torch.int32) and packed.contiguous() launch a plain torch copy right before
# the producer when the dtype or layout differs: also a plain writer, so the contract holds. Bit-identical outputs.
# PRE-WAIT WRITER CONTRACT (rubin-20261015/PREWAIT_AUDIT.md item b): the kernels that read step inputs (seq_lens,
# block tables, positions, cache slots, the sampled-accept pre-wait parameters) before their grid-dependency wait rely
# on every writer of them being a host-to-device copy before the replay or a PLAIN launch earlier in the step:
# msa_kv_update, the step-input prologue (Triton scatter or the C++ op), the spec fold tail (Triton or C++), the pack
# scatter and any step-head op (the spec step correction, STEP_SPEC_OFFSETS_ONE_LAUNCH). Never give one of them a PDL
# attribute with an early trigger.
PRODUCER_PREWAIT_STEP_INPUTS = True

# The fused embedding + first RMSNorm kernel: one owner per row count and index dtype. SM107: 1.55..1.65 us against
# 3.4..6.0 us for the stock embedding + RMSNorm at 1..8 rows (the gather bitwise, the norm one ulp off on ~5e-6 of the
# elements: summation order); it also hands the first layer its residual,
# so layer 0 takes the decode layer instead of the generic one. False runs layer 0 on the generic layer.
# On by default since r4.2: the intermittent illegal memory access first seen with the GEMM/AR stack on was the
# built-in Triton sparse decode's (fixed in r4); with the stack on, GSM8K 4-in-flight ran crash-free on r4.1.
EMBED_FIRSTNORM_TP2 = True
# B5 (SpecTail TR-5b): the same kernel on the 9..16-row steps (one CTA per row, so each row's arithmetic is the 1..8-row
# one; the op's 9..16-row cells), which hands layer 0 its residual so it takes the decode layer. Not past 16 rows:
# there the stock embedding + RMSNorm it replaces differs from it by more than at 1..8 rows (7..21 normed elements, one
# past 1 ulp, at 19..28 rows; gate 795453 / debug 795562), so 17..32-row steps keep the stock layer 0.
# False: 1..8 rows.
B4_EMBED_FIRSTNORM = True
EMBED_FIRSTNORM_MAX_TOKENS = (
    (B4_MAX_TOKENS if B4_EMBED_FIRSTNORM else TINY_MAX_TOKENS) if EMBED_FIRSTNORM_TP2 else 0
)
# The o_proj pushes its rows into the attention boundary's multicast all-reduce (prepushed mode 2) up to this many
# tokens; wider steps run the GEMV and a plain multicast all-reduce. SM107 TP2: -0.5..-1.2 us per layer, bitwise equal
# to the unpushed boundary. False keeps the GEMV and the boundary separate.
# On by default since r4.2: the intermittent illegal memory access first seen with the GEMM/AR stack on was the
# built-in Triton sparse decode's (fixed in r4); with the stack on, GSM8K 4-in-flight ran crash-free on r4.1.
ATTN_PUSH_TP2 = True
# B6 (SpecTail TR-5b): the o_proj also pushes on the 9..16-row steps, into the boundary's 16-token bucket; the GEMV's
# push epilogue covers MMA width 16 with the bucket's tokens in its compile key, and the pushed boundary is bitwise the
# unpushed one (the op's prepushed 9..16-row cells). Not past 16 rows: the o_proj GEMV keeps 16 rows
# (_GEMV_B4_ROWS_ONLY), so a 17..32-row o_proj runs the generic GEMM.
# False: 1..8 rows.
B4_ATTN_PUSH = B4_MC_AR
ATTN_PUSH_MAX_TOKENS = (
    (min(PLAN_MC_AR_MAX_TOKENS, B4_MAX_TOKENS) if B4_ATTN_PUSH else TINY_MAX_TOKENS)
    if (MC_AR_TP2 and ATTN_PUSH_TP2)
    else 0
)
# The dense layers' MLP down projection pushes its rows into the post-MLP boundary's multicast all-reduce (prepushed
# mode 2), as the o_proj does into the attention boundary: the all-reduce no longer waits for the whole down GEMV grid
# before loading and pushing the rows itself. SM107 TP2 (stack-ON nsys 773230): the two non-capture dense layers'
# post-MLP hops end 5.5-5.8 us after the down GEMV vs 2.4-2.5 us for a pushed hop. Bitwise equal to the unpushed
# boundary (the same bf16 words). False keeps the down GEMV and the boundary separate.
DENSE_PUSH_TP2 = True
DENSE_PUSH_MAX_TOKENS = TINY_MAX_TOKENS if (MC_AR_TP2 and DENSE_PUSH_TP2) else 0
# The generic layer's boundaries (prefill, mixed steps, Eagle3 capture layers, decode steps past DECODE_MAX_TOKENS): the
# most rows that take the multicast all-reduce + residual + RMSNorm, alone and with the shared + routed add folded in:
# the multicast all-reduce's widest bucket (48 tokens), as the agg canon runs it (v1.8.3; GSM T0 / T1 803743 /
# 803744), under the same constant (MC_AR_TP2; off = the built-in all-reduce).
GENERIC_MC_AR_MAX_TOKENS = 48 if MC_AR_TP2 else 0  # capacity: fixed: the widest bucket
GENERIC_MC_AR_FOLDED_MAX_TOKENS = 48 if MC_AR_TP2 else 0  # capacity: fixed: the widest bucket
# The generic layer's MoE boundary past DECODE_MAX_TOKENS rows, up to this many (at most the finalize kernel's 48,
# m3_mc_allreduce.FINALIZE_MAX_TOKENS): the routed experts stay unfinalized and their finalize (+ the shared expert)
# runs inside the boundary's multicast all-reduce + residual + next RMSNorm (m3_mc_finalize_allreduce_norm), as the
# decode layer's MC_FINALIZE does. 48 as the agg canon runs it (v1.8.3; the same gates as above).
GENERIC_MC_FINALIZE_MAX_TOKENS = 48 if MC_AR_TP2 else 0  # capacity: fixed: the finalize rows
# The MoE finalize (+ the shared expert add) of a finalized MoE forward with at least MOE_TRITON_FINALIZE_MIN_TOKENS
# rows (prefill, mixed and wide decode past the folds) runs as ops/m3_moe_finalize: the trtllm-gen runner returns
# its FC2 output unfinalized and one Triton kernel gathers the top-k rows together, bitwise to the runner's own
# finalize and the separate shared add (153 -> 98 us per layer at 16K rows on SM107). False: the runner finalizes.
MOE_TRITON_FINALIZE = True
MOE_TRITON_FINALIZE_MIN_TOKENS = 49  # capacity: fixed: past the generic MC finalize's 48 rows
AR_PUSH_MODE = 2
# The 1..8-token MoE engine (router, routed experts and the shared expert in one launch, ops/moe/m3_moe_engine at
# the TP2 width 1536). Rubin TP2 ablation chain (57 layers behind a 58.4 MB PDL producer, row = (P + MoE) - P, both
# orders): 31.3 / 50.7 us/layer at 4 / 8 tokens against the built-in MoE's 42.4 / 55.4 (FlashInfer MXFP8 shared
# expert); bit for bit the built-in block's shared.add_(routed). KILL_MOE_ENGINE: the built-in MoE at every row count.
KILL_MOE_ENGINE = False
MOE_ENGINE_MAX_TOKENS = 0 if KILL_MOE_ENGINE else 8
# The engine's push of its sum into the MoE boundary's multicast all-reduce (prepushed mode 2) on the rows the
# engine serves, with the multicast boundary (DECODE_PLANS' moe_mc_ar): the boundary reads the pushed words instead
# of loading and pushing the rows itself. Bit for bit the unpushed boundary (comm/_m3_moe_push_op_matrix.py).
MOE_PUSH_MAX_TOKENS = MOE_ENGINE_MAX_TOKENS if MC_AR_TP2 else 0
# When the engine pushes its sum into the boundary all-reduce, every CTA releases the PDL dependents right after its
# grid dependency wait instead of after the push, so the all-reduce (which polls the pushed tagged words, not the grid)
# becomes resident on the SMs the engine frees. Scheduling only: the same bits (the push oracle checks on vs off).
# Rubin TP2 greedy early column, same-job A,B + B,A (778191 / 778192): c1 -15 us, about -15 at 323K (within noise).
# False: releases after the push.
MOE_EARLY_AR_RELEASE = True
# A greedy Eagle3 verify step of up to this many requests verifies with the fused shard argmax + acceptance kernel
# over the target lm_head's BF16 shard logits (ops/m3_shard_accept), at the configured draft length.
SHARD_ACCEPT_DRAFT_LEN = 3  # capacity: fixed: the built Eagle3 draft length
# B4 spec tail: the fused verify (shard / sampled accept), the fused first drafter inputs (chunk / capture norm) and the
# fused draft loop (draft_step, its draft input, the drafter boundary all-reduce) also take the 9..32-row steps, up to
# DECODE_MAX_TOKENS verify rows (DECODE_MAX_TOKENS // (draft length + 1) requests; the draft step's head takes at most
# 8). Every kernel is row-local (per-row norms, per-request keys and Philox draws), so each row gets the bits it gets
# at 1 / 2 requests; the op cells at the new rows certify it.
# False: TINY_MAX_TOKENS rows.
B4_SPEC_TAIL = True
SPEC_TAIL_MAX_TOKENS = DECODE_MAX_TOKENS if B4_SPEC_TAIL else TINY_MAX_TOKENS
SPEC_TAIL_MAX_REQUESTS = SPEC_TAIL_MAX_TOKENS // (SHARD_ACCEPT_DRAFT_LEN + 1)
# W2 (SpecTail, agg wide decode): the greedy fused verify (ops/m3_shard_accept) also takes the
# 33..WIDE_MAX_TOKENS-row decode-only steps, as whole-request chunks of the kernel's certified 48-row geometry (12
# requests per launch at draft length 3; m3_shard_accept.WIDE_SPEC_TAIL), instead of the gathered target logits +
# argmax; and the in-tree draft loop takes the drafter's top-1 from its local head shard (W2b). Sampled verifies keep
# the fused path only up to SAMPLED_VERIFY_MAX_REQUESTS (past it, the in-tree sampler). Every other fused piece keeps
# SPEC_TAIL_MAX_TOKENS rows (the drafter then runs its in-tree loop). Rubin TP2 W1b 797141 (verify + 3 drafter top-1
# passes, stock -> fused). Greedy chunks and the shard top-1 are bitwise the gathered argmax (lowest index on ties).
WIDE_SPEC_TAIL = m3_shard_accept_kernel.WIDE_SPEC_TAIL
WIDE_VERIFY_MAX_TOKENS = WIDE_MAX_TOKENS if WIDE_SPEC_TAIL else SPEC_TAIL_MAX_TOKENS
WIDE_VERIFY_MAX_REQUESTS = WIDE_VERIFY_MAX_TOKENS // (SHARD_ACCEPT_DRAFT_LEN + 1)
# 0: (the gathered logits + argmax run). Rubin TP2 greedy verify (microbench): stock 120.4 us at b1; fused
# with the coded head 117.0 us; the BF16-head + shard-accept arm is not measured yet.
SHARD_ACCEPT_MAX_REQUESTS = WIDE_VERIFY_MAX_REQUESTS
# A sampled (temperature + top-p, advanced_sampling_mode no_topk) Eagle3 verify step of up to this many requests
# verifies with the fused sampled acceptance (ops/m3_sampled_accept, the TP2-shard kernel) over the target
# lm_head's BF16 shard logits, at the configured draft length (the AgentX contract). 0: (the gathered logits
# and the in-tree sampler run). Rubin TP2 b1/b2 (p5 microbench): 180.1 / 181.8 -> 115.1 / 113.9 us per verify.
SAMPLED_VERIFY_MAX_REQUESTS = SPEC_TAIL_MAX_REQUESTS
# A step verified by the in-tree path takes the fused first drafter inputs (the capture norms and the first pass's
# gather) up to this many requests.
FIRST_DRAFTER_INPUTS_MAX_REQUESTS = 0  # capacity: fixed: off at TP2 (0)
# A step verified by the fused kernels (dispatch.SHARD_ACCEPT / SAMPLED_ACCEPT) of up to this many requests takes the
# fused first drafter inputs (spec_worker._m3_first_drafter_inputs: the accepted tokens as they lie, the first pass's
# gather rows, the fc_norm chunk norm or the armed capture norm, then the in-tree fc). 0: (the in-tree first
# drafter inputs run, the captures stay in-tree and the fused draft loop steps aside).
FUSED_FIRST_INPUTS_MAX_REQUESTS = SPEC_TAIL_MAX_REQUESTS
# On a fused-verify step of up to this many verify rows the Eagle3 captures write rmsnorm(bf16(h + residual)) *
# fc_norm[i] in one kernel (ops/spec/m3_capture_norm) once the fc_norm weights are stacked; the fused first drafter
# inputs then skip their chunk norm. 0: (the in-tree capture, then the chunk norm).
CAPTURE_NORM_MAX_ROWS = SPEC_TAIL_MAX_TOKENS
# The first draft pass's per-layer fc_norm over the captured columns in one kernel (ops/spec/m3_chunk_norm) at up
# to this many verify rows; wider first passes take the in-tree first drafter inputs. 0:.
CHUNK_NORM_MAX_ROWS = SPEC_TAIL_MAX_TOKENS
# The fused draft loop's two all-reduce boundaries per pass (o_proj -> post-attention norm, down -> final norm) of up
# to this many rows run the multicast all-reduce + residual + RMSNorm (ops/m3_mc_allreduce, opts bit 16: the
# drafter's stock rounding, normalizing the unrounded FP32 sum as FlashInfer fused_add_rmsnorm does) instead of the
# built-in all-reduce + RMSNorm. 0:. Rubin TP2 (p16 microbench): 10.0..11.0 -> 4.6..5.3 us per boundary.
# (The drafter epilogue, opts bit 16, is built for the 8-token bucket only: a B4 spec-tail first pass of 16 / 32 rows
# takes the built-in all-reduce + RMSNorm; its later passes of up to 8 rows stay multicast.)
DRAFT_MC_AR_MAX_ROWS = 8  # capacity: fixed: the drafter epilogue's 8-token bucket
# The drafter's o_proj and down projection, on the copied split-K kernel (DRAFT_LOWM_OWN_SPLITK), also push their rows
# into the drafter boundary all-reduce's exchange (the canonical words its own push would store, at the call parity),
# which then runs prepushed 2 instead of pushing x itself; the copy releases its dependents only after its own grid
# wait and the pushes (the prepushed >= 2 contract). Bitwise. TP2: the 4-row first pass only (its 1-row linears run
# FlashInfer's direct kernel). False:.
DRAFT_AR_PUSH = True
# The drafter boundary all-reduce fed by its projection's push (DRAFT_AR_PUSH): its own taken / not-taken lines.
DRAFT_AR_PUSH_PATH = "m3_draft_ar_push"
# The same push from the drafter's direct ('d') low-M linears (the 1-row passes' o_proj and down at TP2), through this
# target's copy of FlashInfer's direct kernel (ops/m3_lowm_direct; its output FlashInfer's): the owning thread
# stores the CTA's column pairs into the exchange before the kernel's last-instruction release. Bitwise. Rubin TP2
# world 2 (dpush-1 782685), per 1-row boundary: o_proj 9.22 -> 8.09 us, down 13.81 -> 12.72. False:.
DRAFT_AR_PUSH_DIRECT = True
# uGPU: a verify of these row counts computes the target logits shard as two half-vocabulary in-tree GEMMs, each on
# the green context of the die whose memory holds that half of a placed copy of the head (ops/gemm/m3_ugpu_head; the
# module's weight is untouched, so every other reader keeps ordinary memory); the logits are the in-tree head's bit for
# bit. Rubin TP2 head shard: 93.3 -> 79.5 us at
# 4 rows, 93.5 -> 80.6 / 80.7 at 12 / 16 (at 1 / 2 rows the half-shard GEMM is another kernel). In-model the decode
# step is ~45 us shorter at one request and ~90 us at two, and the gain holds at long context.
KILL_UGPU_HEAD = False
UGPU_HEAD_ROWS: Tuple[int, ...] = () if KILL_UGPU_HEAD else (4, 8, 12, 16)
# On a fused-verify step of up to this many verify rows the drafter's first fc is forked onto the aux stream right
# after the last captured target layer, overlapping the remaining target layers and the head (spec_worker).
FC_EARLY_MAX_ROWS = 0  # capacity: fixed: measured fork crossover
# A sampled fused verify of up to this many verify rows on the in-tree head GEMM forks the drafter's first fc after the
# head instead of beside it (spec_worker): the fc's split-K grid otherwise holds the SMs first and the head GEMM starts
# late. 0:. Rubin TP2 c1 (locked nsys 768559): the head started 17.4 us after its input was ready.
FC_AFTER_HEAD_MAX_ROWS = SPEC_TAIL_MAX_TOKENS
# A decode-only Eagle3 step of up to this many requests drafts with the fused draft loop (spec_worker): per pass the
# draft input in one launch (ops/m3_spec_tail.draft_input), the drafter layer's in-tree attention, MLP and
# all-reduces, and one kernel for the draft head (its 12-bit coded copy, or the BF16 fragments when the coding does
# not decode bit for bit), the cross-rank argmax and the pass's bookkeeping (ops/m3_draft_step), at the
# configured draft length. The head kernel takes one row per request, at most 8.
# The fused draft loop streams the 12-bit coded copy of the draft head (when it decodes bit for bit); False keeps the
# BF16 fragments.
DRAFT_STEP_DRAFT_LEN = SHARD_ACCEPT_DRAFT_LEN  # capacity: fixed: the built draft length
# 0: (the in-tree draft loop runs). Rubin TP2 per draft pass (head + argmax + exchange): 122.6 / 126.0 ->
# 102.8 / 102.8 us at b1 / b2 (BF16 fragments; the coded head is slower on sm_107: DRAFT_HEAD_CODED False).
DRAFT_STEP_MAX_REQUESTS = min(SPEC_TAIL_MAX_REQUESTS, 8)  # capacity: fixed: the draft head's 8 rows
# The Eagle3 drafter's BF16 linears (qkv, o_proj, gate_up, down, fc) run FlashInfer's fixed-tactic low-M kernels
# (ops/gemm/m3_draft_lowm, the TP8 port) at the rows its TP2 table is certified for, up to this many rows; other calls
# keep the in-tree cuBLAS GEMM. 0:. Rubin TP2 (mclk locked, 4 rotating sets) at 1 row: qkv 13.18 -> 9.71,
# o 7.17 -> 5.33, gate_up 22.27 -> 17.70, down 12.52 -> 9.76, fc 21.50 -> 17.77 us.
DRAFT_LOWM_MAX_ROWS = 4  # capacity: fixed: the certified low-M rows
# The split-K ('s') tactics run this target's copy of FlashInfer's kernel (ops/m3_lowm_splitk): bit for bit the
# same output at every certified cell (spec/test_modeling_v2_m3_lowm_splitk.py); the seam for the drafter-boundary push
# epilogue and the down projection's SwiGLU prologue. False: (FlashInfer's kernel).
DRAFT_LOWM_OWN_SPLITK = True
DRAFT_LOWM_ROWS: Dict[Tuple[int, int], FrozenSet[int]] = {
    nk: frozenset(m for m in rows if m <= DRAFT_LOWM_MAX_ROWS)
    for nk, rows in DRAFT_LOWM_TACTIC_ROWS.items()
}
# The identity-selection sparse decode moves from the per-row producer to the multi-query producer at this many rows
# (ops/triton_sparse_decode.MQ_MIN_TOKENS); the multi-query producer needs one KV head, so at the TP2 heads every
# identity step takes the per-row producer.
SPARSE_DECODE_MQ_MIN_TOKENS = 17  # capacity: fixed: the multi-query crossover
SPARSE_DECODE_KV_HEADS = 2  # capacity: fixed: the tp2 KV heads per rank
# The long-row top-16 selector takes engines of 129..LONG_SELECT_MAX_BLOCKS KV blocks of 128 tokens; it holds 2, 4 or
# 9 keys per thread by the row's valid-block count. At most 128 blocks, or past the maximum, the built-in selector
# runs; 0 turns the path off. Rubin TP2 (2 KV heads), 165,521 tokens, behind the scorer, ablation per layer:
# -0.9 us at 1 request and -1.2 us at 2 requests against the built-in selector.
LONG_SELECT_MIN_BLOCKS = 129  # capacity: fixed: past the built-in selector's 128
LONG_SELECT_MAX_BLOCKS = 9216  # capacity: fixed: the select kernels' built blocks per row
LONG_SELECT_KEY_WIDTHS = (2, 4, 9)
# The long-row select releases the sparse decode after it right after its own grid wait (the decode stages q and the
# request length before its wait) up to this many query rows; past it the trigger stays at the select's exit. On Rubin
# TP2 the early-launched long-context sparse decode (SPARSE_DECODE_LC) crowds the select's CTAs off the SMs already at
# one request (4 rows): releasing at the exit is 336 us per decode step faster at 1 request and 350 at 2 (in-model,
# graph time, 165K-context mixed draws, locked memory clock), so the early release is off at every row count.
SELECT_EARLY_MAX_ROWS = 0  # capacity: fixed: measured early-release crossover
# Without the early release, the long-row select releases the sparse decode once every score of its row is loaded
# (1, before the exact rank of the candidates) instead of at its exit (0). The release still follows
# the select's own grid wait, which the sparse decode's pre-wait q / length / block-table reads rely on. The decode's
# 16-CTA cluster launch and prologue then overlap the rank: Rubin TP2 in-model, locked memory clock, AgentX draws,
# 2 instances vs the r4.3 stored A: replay period -162 us fixed, -94 at a 323K-token request.
SELECT_LONG_TRIGGER_PHASE = 1
# The long-row select runs as ops/m3_select_long_dsl (ops/attention/m3_select_long_dsl): one 384-thread CTA per
# (query row, KV head) instead of one 1024-thread CTA, the same selection bit for bit (ties included), with 4, 11 or
# 24 keys per thread (LONG_SELECT_DSL_KEY_WIDTHS) and the same trigger choices. Rubin TP2 decode chain (2 KV heads,
# 4 rows, 57 layers behind the producer stream and the scorer, both orders x 3 rounds): -0.127 / -0.148 / -0.037 us
# per layer at 98K / 165K / 323K tokens against the 1024-thread kernel. False: (the 1024-thread kernel).
SELECT_LONG_DSL = True
LONG_SELECT_DSL_KEY_WIDTHS = (4, 11, 24)
# The MSA index scorer (ops/attention/m3_index_score) of a sparse decode step, up to INDEX_SCORE_MAX_REQUESTS requests
# (0 turns the path off; wider steps run the in-tree CuTe DSL scorer): its CTAs per request by request count, and up
# to INDEX_SCORE_EARLY_MAX_REQUESTS requests the early issue (Q and the first K tile right after the grid wait, one
# older block per CTA prefetched into L2 before it). The split is captured with the graph, so it is one value per
# request count over every context. Rubin TP2 (2 index heads), 165,521 tokens, behind a 58 MB PDL producer, ablation
# per layer: 1 request 3.27 us at 1536 CTAs (512: 3.76, in-tree 256-CTA scorer 6.07); 2 requests 4.97 us at 768
# (in-tree 6.71). Within 0.2 us of each context's best split from 16K to 400K tokens except 2 requests at 400K (+0.6).
# The early issue reads the request's first page id from the MSA page table before the grid wait: exact because the
# page table is written only on the host side before the replay (attention/test_modeling_v2_m3_block_table_contract.py);
# the request length read there is a hint, every load and store is bounded by the one read after the wait.
# The batch-8 plan's 3- and 4-request steps take it too (Indexer k11 795543: Rubin TP2, 512 MB flush before each call,
# graph-timed, bitwise equal to the in-tree scorer; requests at 1.0 / 0.9 / 0.8 / 0.7 x the context): 3 requests at
# 424 CTAs each 6.08 / 9.04 / 14.20 us at 98K / 165K / 323K tokens (in-tree 6.65 / 9.44 / 14.49); 4 requests at 318
# 7.41 / 10.82 / 16.80 (in-tree 7.60 / 10.97 / 16.96). 5..8 requests at about 1272 CTAs in total, 1272 // requests
# each (Indexer k11c 795819): 5 at 254 8.25 / 11.44 / 19.10 us (in-tree 8.66 / 12.10 / 19.65); 6 at 212; 7 at 181;
# 8 at 159 9.91 / 13.97 / 23.33 (10.33 / 14.47 / 23.74). Each within 0.03 us of its count's best split.
# 9..48 requests (W-IDX1, one split per bucket) win standalone (Indexer k14b 797357: 46 requests 52.29 -> 49.85 us per
# layer) but lose in-model (agg census c46 / c20: 52.13 -> 61.85 / 24.22 -> 29.06 us per layer, the in-tree scorer
# already at the single-stream bandwidth floor), so steps past 8 requests keep the in-tree scorer.
INDEX_SCORE_MAX_REQUESTS = 8  # capacity: fixed: measured in-model crossover
#: (largest request count of the bucket, CTAs per request) for 9..48 requests (W-IDX1; outside INDEX_SCORE).
INDEX_SCORE_WIDE_SPLITS = ((10, 127), (13, 97), (17, 74), (23, 55), (31, 41), (39, 32), (48, 26))
INDEX_SCORE_SPLIT_K: Mapping[int, int] = MappingProxyType(
    {1: 1536, 2: 768, 3: 424, 4: 318, 5: 254, 6: 212, 7: 181, 8: 159}
    | {r: next(s for hi, s in INDEX_SCORE_WIDE_SPLITS if r <= hi) for r in range(9, 49)}
)
INDEX_SCORE_EARLY_MAX_REQUESTS = 48  # capacity: fixed: measured early-issue crossover


# Synth only (TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS, fractional): the fused verify's forced accepted counts are
# drawn per request, keyed by the request's KV length at the verify (its prompt plus what it has accepted), instead of
# from the server-wide counter, so a request's forced counts do not depend on what the server ran before (A/B draws
# become history-independent). Production sampling never forces counts. False: (the shared counter, the in-tree
# sequence).
FORCE_AL_PER_REQUEST = True

# The decode MXFP8 GEMVs (ops/mx_gemv, SM107 TP2 table), per projection (N, K): the rows each serves, on an
# activation its producer quantized (the MoE boundary for the QKV, the attention merge for the o_proj) and on a bf16
# one (quantized in the GEMV's prologue, bit-identical to mxfp8_quantize). SM107 graph us per projection at 1..8 rows:
# 2.2..8.6 against 4.7..10.5 for the stock MXFP8 quantize + FlashInfer CuTe DSL GEMM (ops/mx_gemv/op.py).
# False puts every projection back on the stock decode-graph GEMM.
# On by default since r4.2: the intermittent illegal memory access first seen with the GEMM/AR stack on was the
# built-in Triton sparse decode's (fixed in r4); with the stack on, GSM8K 4-in-flight ran crash-free on r4.1.
MX_GEMV_TP2 = True
# MX-TAC: the decode-graph MXFP8 projections the GEMVs do not serve take one fixed per-bucket decision instead of the
# AutoTuner's per-start choice (whose L2-hot profiling left c2 bimodal at +-6%): FlashInfer's CuTe DSL quantize and
# CuTe DSL GEMM. Stock's candidates per bucket are quantize {TRT-LLM, CuTe DSL} x GEMM {CUTLASS, CuTe DSL}; on SM107
# every TP2 projection at 1..16 rows runs fastest on this pair, cold and L2-hot (graph us, cold: 4.7..10.3 against
# 5.0..11.5 with the TRT-LLM quantize and 9.7..58 with CUTLASS), and all four candidates give the same bits (quantize
# bytes and scales, GEMM outputs). A constant is rank-agreed and start-independent by construction.
# False returns these projections to the AutoTuner.
MXFP8_DECODE_QUANT_TACTIC = 0  # MXFP8QuantizeRunner.CUTE_DSL
MXFP8_DECODE_GEMM_TACTIC = 0  # FlashInferMXFP8GemmRunner.CUTE_DSL
_ALL_ROWS = frozenset(range(1, TINY_MAX_TOKENS + 1))
_GEMV_PROJECTIONS = (
    QKV_SPARSE,
    QKV,
    O_PROJ,
    SHARED_GATE_UP,
    SHARED_DOWN,
    DENSE_GATE_UP,
    DENSE_DOWN,
)
# B2: the same GEMVs at 9..16 rows (MMA width 16, the kernel's own wider configuration; contiguous k-tiles above 8
# rows): standalone graph us at 16 rows 2.3..8.9 against 4.8..10.6 for the MX-TAC stock pair (b4p1 791712). Batch 8:
# at 17..32 rows (MMA width 32; rule 1b, a different K order than the generic GEMM). GEMM-AR standalone 794929 (iso
# latency vs the FlashInfer fallback on the same MXFP8 bytes), us per call at 24 / 32 rows: QKV-sparse -2.70 / -2.75,
# QKV -2.46 / -2.77, shared gate_up -3.12 / -3.12, shared down -0.94 / -1.01; dense gate_up +1.18 / +0.91, dense down
# +0.58 / +0.13 (the GEMV loses) and o_proj +0.11 / -0.18 (flat) keep 16 rows. Each projection is raised only as far
# as the kernel serves it (served_max_rows): the bf16 GEMV's in-kernel quantize declines dense gate_up / down past 8.
# MX_GEMV_B4 False: 9..32 rows decline to MX-TAC; B8_MX_GEMV_32 False: every GEMV to 16 rows (17..32-row MXFP8
# inputs on FlashInfer's GEMM, exact).
MX_GEMV_B4 = True
B8_MX_GEMV_32 = True
GEMV_MAX_TOKENS = (
    (DECODE_MAX_TOKENS if B8_MX_GEMV_32 else B4_MAX_TOKENS) if MX_GEMV_B4 else TINY_MAX_TOKENS
)
_GEMV_B4_ROWS_ONLY = frozenset({DENSE_GATE_UP, DENSE_DOWN, O_PROJ})


def _gemv_max(nk) -> int:
    return min(GEMV_MAX_TOKENS, B4_MAX_TOKENS) if nk in _GEMV_B4_ROWS_ONLY else GEMV_MAX_TOKENS


MX_GEMV_ROWS: Dict[Tuple[int, int], FrozenSet[int]] = (
    {
        nk: frozenset(range(1, served_max_rows(nk, False, TINY_MAX_TOKENS + 1, _gemv_max(nk)) + 1))
        for nk in _GEMV_PROJECTIONS
    }
    if MX_GEMV_TP2
    else {}
)
# W-G1 (agg wide decode, rule 1b): the MXFP8 GEMV also at 33..48 rows (MMA width 48 / 64) for the projections it wins
# on, after the decode-graph MXFP8 quantize the generic GEMM path runs anyway (the same activation bytes): standalone
# graph us per call at 48 rows, GEMV against the MX-TAC FlashInfer CuTe DSL GEMM on the same MXFP8 input, QKV-sparse
# 5.87 vs 10.82, QKV 5.53 vs 10.76, shared gate_up 4.52 vs 10.51, shared down 2.84 vs 4.74. o_proj and the dense MLP
# keep their rows (flat or losing at 24..48). A different K order than the GEMM (rule 1b). False: 33..48 rows back on
# the generic GEMM, exact.
WIDE_MX_GEMV_48 = True
WIDE_GEMV_MAX_TOKENS = 48  # capacity: fixed: the MXFP8 GEMV's widest build
_GEMV_W48_PROJECTIONS = (QKV_SPARSE, QKV, SHARED_GATE_UP, SHARED_DOWN)
_GEMV_W48_ROWS: Dict[Tuple[int, int], FrozenSet[int]] = (
    {
        nk: frozenset(
            range(
                GEMV_MAX_TOKENS + 1,
                served_max_rows(nk, False, GEMV_MAX_TOKENS + 1, WIDE_GEMV_MAX_TOKENS) + 1,
            )
        )
        for nk in _GEMV_W48_PROJECTIONS
    }
    if MX_GEMV_TP2 and WIDE_MX_GEMV_48 and GEMV_MAX_TOKENS == DECODE_MAX_TOKENS
    else {}
)
for _nk, _rows_w48 in _GEMV_W48_ROWS.items():
    MX_GEMV_ROWS[_nk] = MX_GEMV_ROWS.get(_nk, frozenset()) | _rows_w48
MX_GEMV_BF16_MAX_TOKENS: Dict[Tuple[int, int], int] = (
    {nk: served_max_rows(nk, True, TINY_MAX_TOKENS + 1, _gemv_max(nk)) for nk in _GEMV_PROJECTIONS}
    if MX_GEMV_TP2
    else {}
)
# W-G1: a bf16 activation (no MXFP8 hand-off) on rows the MXFP8 GEMV serves but the bf16 GEMV's in-kernel quantize does
# not (QKV / shared gate_up at 17..32 without the QKV hand-off, all four at 33..48) is quantized by the decode-graph
# MXFP8 quantize (the generic GEMM path's own) and runs the GEMV instead of that GEMM.
MX_GEMV_QUANT_ROWS: Dict[Tuple[int, int], FrozenSet[int]] = (
    {
        nk: frozenset(r for r in MX_GEMV_ROWS.get(nk, ()) if r > MX_GEMV_BF16_MAX_TOKENS.get(nk, 0))
        for nk in _GEMV_W48_PROJECTIONS
    }
    if MX_GEMV_TP2 and WIDE_MX_GEMV_48
    else {}
)

Key = Hashable


@dataclass(frozen=True)
class RowRange:
    """The shapes one fast path takes, and the catalog entry whose cells certify each of them.

    ``keys`` are the covered shapes: row counts for most paths, (query rows per request, requests) for the attention
    kernels, a variant name for thresholds. ``certifies`` maps a cell's spec of ``entry`` to the keys that cell
    drives (empty for a cell of another variant).
    """

    path: str
    keys: FrozenSet[Key]
    entry: OpWrapper
    certifies: Callable[[Mapping[str, Any]], Iterable[Key]]

    def covers(self, key: Key) -> bool:
        return key in self.keys

    def certified(self) -> FrozenSet[Key]:
        return frozenset(k for cell in self.entry.CELLS for k in self.certifies(cell.spec))


def _rows(lo: int, hi: int) -> FrozenSet[int]:
    return frozenset(range(lo, hi + 1))


def _spec_rows(field: str, **match: Any) -> Callable[[Mapping[str, Any]], Iterable[Key]]:
    """A cell's ``spec[field]`` when the cell's other fields match ``match``."""

    def certifies(spec: Mapping[str, Any]) -> Iterable[Key]:
        if any(spec.get(name) != value for name, value in match.items()):
            return ()
        return (spec[field],)

    return certifies


def _const(*keys: Key) -> Callable[[Mapping[str, Any]], Iterable[Key]]:
    return lambda spec: keys


def _sparse_decode_producer(spec: Mapping[str, Any]) -> Iterable[Key]:
    if not spec["identity"]:
        return ()
    rows = spec["decode_query_len"] * len(spec["seq_lens"])
    mq = rows >= SPARSE_DECODE_MQ_MIN_TOKENS and SPARSE_DECODE_KV_HEADS == 1
    return ("multi_query" if mq else "per_row",)


def index_score_schedule(requests: int) -> Tuple[int, bool]:
    """The index scorer's (CTAs per request, early issue) at this many requests (a covered request count)."""
    return INDEX_SCORE_SPLIT_K[requests], requests <= INDEX_SCORE_EARLY_MAX_REQUESTS


def _index_score_requests(spec: Mapping[str, Any]) -> Iterable[Key]:
    """A cell certifies its request count when it drives the TP2 heads with that count's schedule."""
    requests = len(spec["seq_lens"])
    if spec.get("heads") != 2 or requests not in INDEX_SCORE_SPLIT_K:
        return ()
    if (spec["split_k"], spec["early"]) != index_score_schedule(requests):
        return ()
    return (requests,)


def _long_select_widths(spec: Mapping[str, Any]) -> Iterable[Key]:
    """The key widths a cell's rows reach: every width up to its blocks' when rows of mixed lengths are drawn."""
    blocks = spec["num_blocks"]
    widest = 2 if blocks <= 2048 else 4 if blocks <= 4096 else 9
    if spec["valid"] == "mixed":
        return tuple(w for w in LONG_SELECT_KEY_WIDTHS if w <= widest)
    return (widest,)


def _long_select_dsl_widths(spec: Mapping[str, Any]) -> Iterable[Key]:
    """The 384-thread select's key widths a cell's rows reach (every width up to its blocks' for mixed rows)."""
    blocks = spec["num_blocks"]
    widest = 4 if blocks <= 4 * 384 else 11 if blocks <= 11 * 384 else 24
    if spec["valid"] == "mixed":
        return tuple(w for w in LONG_SELECT_DSL_KEY_WIDTHS if w <= widest)
    return (widest,)


EMBED_FIRSTNORM = RowRange(
    "embed_firstnorm",
    _rows(1, EMBED_FIRSTNORM_MAX_TOKENS),
    m3_embedding_firstnorm,
    _spec_rows("rows"),
)
# One request decoding one token on an MXFP8 layer input: QKV GEMV + attention + o_proj in one launch.
# The identity-selection Triton producer + merge (every request within 16 blocks), certified at the TP2 heads on SM107.
SPARSE_DECODE_PRODUCER = RowRange(
    _sparse_decode_kernel.PRODUCER_PATH,
    _sparse_decode_kernel.PRODUCER_KEYS,
    m3_sparse_decode,
    _sparse_decode_producer,
)
# The long-context sparse decode at the TP2 heads (32 query / 2 KV heads): attention, split-K merge and the MXFP8
# o_proj hand-off in one launch, for every decode step with a selected table. "mxfp8" is a step of at most
# MergeMxfp8.MAX_ROWS rows (the hand-off is written and published to m3_prequant.SLOT), "plain" a larger one.
# frozenset() routes every step back to the Triton producer + merge.
SPARSE_DECODE_LC = RowRange(
    _sparse_decode_kernel.LC_PATH,
    _sparse_decode_kernel.LC_KEYS,
    m3_sparse_decode_lc,
    lambda spec: ("mxfp8" if spec["mxfp8"] else "plain",),
)
# N1 at TP2: SPARSE_DECODE_LC records each step's selected block ids per sparse layer and, before its grid wait (while
# the select ranks), pulls into L2 the K/V pages + scales of the blocks the SAME layer selected one step back (65-68% of
# a layer's selected blocks recur one step later: MSA probe 778226). Same layer, not the next one as at TP4: the MoE
# between two sparse layers streams ~250 MB through the 126 MB L2, so an earlier issue would be evicted. Hints only;
# up to N1_MAX_ROWS query rows. False:.
SPARSE_DECODE_LC_PREFETCH_SELF_KV = True
N1_MAX_ROWS = 128  # capacity: fixed: the N1 prefetch's row bound
# Every token count of a sparse layer on an NVFP4 KV cache at the TP2 heads (prefill and decode): the producer takes the
# geometry, not a row range. frozenset() routes every sparse layer back to the native producer.
PRODUCER = RowRange("m3_producer", frozenset({"nvfp4_tp2"}), m3_producer, _const("nvfp4_tp2"))
INDEX_SCORE = RowRange(
    "m3_index_score",
    _rows(1, INDEX_SCORE_MAX_REQUESTS),
    m3_index_score,
    _index_score_requests,
)
LONG_SELECT = RowRange(
    "long_select",
    frozenset(LONG_SELECT_KEY_WIDTHS) if LONG_SELECT_MAX_BLOCKS else frozenset(),
    m3_select_long,
    _long_select_widths,
)
LONG_SELECT_DSL = RowRange(
    "long_select_dsl",
    frozenset(LONG_SELECT_DSL_KEY_WIDTHS)
    if LONG_SELECT_MAX_BLOCKS and SELECT_LONG_DSL
    else frozenset(),
    m3_select_long_dsl,
    _long_select_dsl_widths,
)
MC_ALLREDUCE = RowRange(
    "mc_allreduce_norm",
    _rows(1, PLAN_MC_AR_MAX_TOKENS) if MC_AR_TP2 else frozenset(),
    m3_mc_allreduce_norm,
    _spec_rows("m", prepushed=0),
)
MC_FINALIZE = RowRange(
    "mc_finalize_allreduce_norm",
    _rows(MC_FINALIZE_MIN_TOKENS, PLAN_MC_AR_MAX_TOKENS if B4_MC_FINALIZE else TINY_MAX_TOKENS),
    m3_mc_finalize_allreduce_norm,
    _spec_rows("m"),
)
ATTN_PUSH = RowRange(
    "attn_push",
    _rows(1, ATTN_PUSH_MAX_TOKENS),
    m3_mc_allreduce_norm,
    _spec_rows("m", prepushed=AR_PUSH_MODE),
)
DENSE_PUSH = RowRange(
    "dense_push",
    _rows(1, DENSE_PUSH_MAX_TOKENS),
    m3_mc_allreduce_norm,
    _spec_rows("m", prepushed=AR_PUSH_MODE),
)
# The router of an unfused MoE of 9..48 rows (above the MoE engine's 8), a generation step's or a context step's: the
# TF32 tensor-core router (minimax_m3_router_tc, 16-CTA clusters, one reduction order; the arithmetic the built-in gate
# takes on SM107 at 9..16 generation rows), with the routed experts' NVFP4 activation from the same launch, as the agg
# canon routes these rows (2.9-3.5 us/call against ~9.8 us for the cuBLAS TF32 split-K router; closer to the fp32
# product: max |dlogit| 1.7e-6 vs 6.7e-5). Devices that cannot co-schedule 16-CTA clusters keep the fp32 router.
ROUTER_TC_GEN_MIN_ROWS = 9  # capacity: fixed: past the MoE engine's 8 rows
# Up to 48 rows, as the agg canon runs it (v1.8.3, the router GEMV keys gone): the MoE of a 9..48-row step routes on
# it, with the routed experts' NVFP4 activation from the same kernel (the decode layer's MC finalize, the generic
# layer). GB300 tp4, behind a PDL producer, with the NVFP4 activation: 3.20 / 3.82 / 7.36 us at 8 / 12 / 48 tokens.
ROUTER_TC_GEN_MAX_ROWS = 48  # capacity: fixed: the router kernel's widest build
ROUTER_TC_GEN = RowRange(
    "m3_router_tc",
    _rows(ROUTER_TC_GEN_MIN_ROWS, ROUTER_TC_GEN_MAX_ROWS),
    m3_router_tc,
    _spec_rows("m"),  # every cell checks the logits, with or without the NVFP4 activation
)
MOE_PUSH = RowRange(
    "moe_push",
    _rows(1, MOE_PUSH_MAX_TOKENS),
    m3_mc_allreduce_norm,
    _spec_rows("m", prepushed=AR_PUSH_MODE),
)
MOE_ENGINE = RowRange(
    "m3_moe_engine", _rows(1, MOE_ENGINE_MAX_TOKENS), m3_moe_engine, _spec_rows("tokens")
)
# The GEMM paths are keyed (projection (N, K), rows): each projection runs its own split-K / ring / MMA
# configuration, so a row count is certified per projection.
MX_GEMV = RowRange(
    "m3_mx_gemv",
    frozenset((nk, m) for nk, rows in MX_GEMV_ROWS.items() for m in rows),
    m3_mx_gemv,
    lambda spec: ((spec["nk"], spec["m"]),),
)
MX_GEMV_BF16 = RowRange(
    "m3_mx_gemv_bf16",
    frozenset((nk, m) for nk, hi in MX_GEMV_BF16_MAX_TOKENS.items() for m in _rows(1, hi)),
    m3_mx_gemv_bf16,
    lambda spec: ((spec["nk"], spec["m"]),),
)
# (draft length, requests) of a greedy verify step.
SHARD_ACCEPT = RowRange(
    "m3_shard_accept",
    frozenset((SHARD_ACCEPT_DRAFT_LEN, n) for n in range(1, SHARD_ACCEPT_MAX_REQUESTS + 1)),
    m3_shard_accept,
    lambda spec: ((spec["draft_len"], spec["nreq"]),),
)

# Rows of a fused-draft-loop all-reduce boundary (drafter hidden 6144) on the multicast all-reduce.
DRAFT_MC_AR = RowRange(
    "m3_draft_mc_allreduce_norm",
    _rows(1, DRAFT_MC_AR_MAX_ROWS),
    m3_mc_allreduce_norm,
    lambda spec: (spec["m"],) if spec.get("drafter") else (),
)

# (draft length, requests) of a sampled verify step.
SAMPLED_ACCEPT = RowRange(
    "m3_sampled_accept",
    frozenset((SHARD_ACCEPT_DRAFT_LEN, n) for n in range(1, SAMPLED_VERIFY_MAX_REQUESTS + 1)),
    m3_sampled_accept,
    lambda spec: ((SHARD_ACCEPT_DRAFT_LEN, spec["nreq"]),),
)


# (query rows per request, requests) of a decode step whose dense-layer attention runs on PrimTS split-KV decode.
FULLCTX_DECODE = RowRange(
    "m3_fullctx_decode",
    frozenset(FULLCTX_DECODE_RECIPES) if FULLCTX_DECODE_ENABLED else frozenset(),
    m3_fullctx_decode,
    lambda spec: ((spec["qlen"], len(spec["kv_lens"])),),
)

# (query rows per request, requests) of a generation-only drafter pass that runs on the full-context decode.
DRAFTER_FULLCTX = RowRange(
    "m3_drafter_fullctx",
    frozenset(FULLCTX_DECODE_RECIPES) if DRAFTER_FULLCTX_ENABLED else frozenset(),
    m3_fullctx_decode,
    lambda spec: ((spec["qlen"], len(spec["kv_lens"])),),
)


def _verify_rows(max_rows: int) -> FrozenSet[int]:
    """The verify rows of 1..8 requests at the shard-accept draft length, up to ``max_rows``."""
    return frozenset(
        n * (SHARD_ACCEPT_DRAFT_LEN + 1)
        for n in range(1, 9)
        if n * (SHARD_ACCEPT_DRAFT_LEN + 1) <= max_rows
    )


# Requests of a fused-verify step whose first drafter inputs are the fused ones; certified by the chunk-norm cells
# at those requests' verify rows (the norm is the only kernel of its own; the gather rows and the fc are in-tree).
FUSED_FIRST_INPUTS = RowRange(
    "m3_fused_first_inputs",
    _rows(1, FUSED_FIRST_INPUTS_MAX_REQUESTS),
    m3_chunk_norm,
    lambda spec: (
        (spec["m"] // (SHARD_ACCEPT_DRAFT_LEN + 1),)
        if spec["m"] % (SHARD_ACCEPT_DRAFT_LEN + 1) == 0
        else ()
    ),
)
# Verify rows of a fused-verify step whose Eagle3 captures apply the fc_norm.
CAPTURE_NORM = RowRange(
    "m3_capture_norm", _verify_rows(CAPTURE_NORM_MAX_ROWS), m3_capture_norm, _spec_rows("m")
)
# Verify rows of a first draft pass whose fc_norm runs as the chunk norm.
CHUNK_NORM = RowRange(
    "m3_chunk_norm", _verify_rows(CHUNK_NORM_MAX_ROWS), m3_chunk_norm, _spec_rows("m")
)

# ((N, K), rows) of a drafter BF16 linear on its low-M kernel.
DRAFT_LOWM = RowRange(
    "m3_draft_lowm",
    frozenset((nk, m) for nk, rows in DRAFT_LOWM_ROWS.items() for m in rows),
    m3_draft_lowm,
    lambda spec: ((spec["nk"], spec["m"]),),
)
# Verify rows of a step whose target logits come from the die-placed head (in-tree GEMM halves per die).
UGPU_HEAD = RowRange("m3_ugpu_head", frozenset(UGPU_HEAD_ROWS), m3_ugpu_head, _spec_rows("m"))

# (draft length, requests) of a fused draft loop, and the rows of its passes' draft input: the first pass's
# requests x (draft length + 1), one per request on the later passes.
DRAFT_STEP = RowRange(
    "m3_draft_step",
    frozenset((DRAFT_STEP_DRAFT_LEN, n) for n in range(1, DRAFT_STEP_MAX_REQUESTS + 1)),
    m3_draft_step,
    lambda spec: ((spec["draft_len"], spec["nreq"]),),
)
DRAFT_INPUT = RowRange(
    "m3_draft_input",
    frozenset(
        rows
        for n in range(1, DRAFT_STEP_MAX_REQUESTS + 1)
        for rows in (n, n * (DRAFT_STEP_DRAFT_LEN + 1))
    ),
    m3_draft_input,
    _spec_rows("m"),
)

FAST_PATHS: Tuple[RowRange, ...] = (
    EMBED_FIRSTNORM,
    FULLCTX_DECODE,
    DRAFTER_FULLCTX,
    SPARSE_DECODE_PRODUCER,
    SPARSE_DECODE_LC,
    PRODUCER,
    INDEX_SCORE,
    LONG_SELECT,
    LONG_SELECT_DSL,
    MC_ALLREDUCE,
    MC_FINALIZE,
    ATTN_PUSH,
    DENSE_PUSH,
    MOE_PUSH,
    MOE_ENGINE,
    ROUTER_TC_GEN,
    MX_GEMV,
    MX_GEMV_BF16,
    SHARD_ACCEPT,
    SAMPLED_ACCEPT,
    DRAFT_MC_AR,
    UGPU_HEAD,
    DRAFT_STEP,
    DRAFT_INPUT,
    DRAFT_LOWM,
    FUSED_FIRST_INPUTS,
    CAPTURE_NORM,
    CHUNK_NORM,
)

# --------------------------------------------------------------------------------------------------------------------
# Selection: pure functions of the step shape
# --------------------------------------------------------------------------------------------------------------------


def plan_for(num_tokens: int) -> DecodePlan:
    for plan in DECODE_PLANS:
        if num_tokens <= plan.max_tokens:
            return plan
    raise AssertionError(f"no decode plan covers {num_tokens} tokens")


def is_decode_step(num_tokens: int) -> bool:
    """Whether a pure generation step of ``num_tokens`` rows goes to the decode target."""
    return num_tokens <= DECODE_MAX_TOKENS


def moe_path(num_tokens: int) -> Optional[str]:
    """The MoE kernel of a decode step: the engine, the unfinalized MoE + finalize all-reduce, or None (generic)."""
    if MOE_ENGINE.covers(num_tokens):
        return MOE_ENGINE.path
    if MC_FINALIZE.covers(num_tokens):
        return MC_FINALIZE.path
    return None


MX_GEMV_PATH = MX_GEMV.path
MX_GEMV_BF16_PATH = MX_GEMV_BF16.path
# The dense layers' swigluoai activation as this target's C++ op (ops/gemm/m3_dense_swiglu,
# minimaxM3Sm107Tp2/swigluOai.cu, LongAttn c267035d2f: bit for bit the Triton silu_and_mul at limit 7, alpha 1.702,
# beta 1), PDL-launched with an early release so the down GEMV stages its weights during it; the down GEMV reads the
# activation only after its own grid wait. Indexer kdev17 783285 (TP2 chain): -0.69 us per dense layer vs the Triton
# kernel; LongAttn gate a 783511 (oracle 23/23). False runs the Triton kernel.
DENSE_SWIGLU_CPP_PATH = "m3_dense_swiglu"
# FlashInfer's CuTe DSL MXFP8 GEMM on the producer's MXFP8 activation (rows the GEMV does not serve).
PREQUANT_GEMM_PATH = "flashinfer_mxfp8_gemm"
# The MXFP8 quantize (tuned backend) and FlashInfer's CuTe DSL GEMM: every row the decode kernels do not serve.
GENERIC_GEMM_PATH = "mxfp8_gemm"


def linear_path(n: int, k: int, num_tokens: int, prequant: bool, low_m_gemm: bool) -> str:
    """The decode-graph MXFP8 GEMM of an [N, K] projection on ``num_tokens`` rows.

    ``prequant``: the producer published the activation in MXFP8. ``low_m_gemm`` is accepted for the call sites'
    symmetry: no low-M GEMM configuration is certified at TP2, so it selects nothing.
    """
    if prequant:
        if num_tokens in MX_GEMV_ROWS.get((n, k), ()):
            return MX_GEMV_PATH
        return PREQUANT_GEMM_PATH
    if num_tokens <= MX_GEMV_BF16_MAX_TOKENS.get((n, k), 0):
        return MX_GEMV_BF16_PATH
    if num_tokens in MX_GEMV_QUANT_ROWS.get((n, k), ()):
        # W-G1: the decode-graph MXFP8 quantize, then the MXFP8 GEMV.
        return MX_GEMV_PATH
    return GENERIC_GEMM_PATH


# --------------------------------------------------------------------------------------------------------------------
# Fallback policy
# --------------------------------------------------------------------------------------------------------------------


class FastPathUnavailable(RuntimeError):
    """A covered decode shape could not take its fast path under ``modeling_v2='require'``."""


_lock = threading.Lock()
_require = [False]
_logged: set = set()


def set_require(require: bool) -> None:
    """Whether a miss raises (``modeling_v2='require'``); set by the target at construction."""
    _require[0] = bool(require)


def require_mode() -> bool:
    return _require[0]


def _log_once(level: Callable[[str], None], path: str, reason: str, message: str) -> None:
    key = (path, reason)
    with _lock:
        if key in _logged:
            return
        _logged.add(key)
    level(message)


def missed(path: str, rows: Optional[int], reason: str) -> None:
    """``path`` covers this step but cannot run: raise under require, else log once; the caller falls back."""
    at = "" if rows is None else f" at {rows} rows"
    if _require[0]:
        raise FastPathUnavailable(
            f"modeling_v2='require': the MiniMax-M3 decode path {path!r} covers this step{at} but cannot run: "
            f"{reason}"
        )
    _log_once(
        logger.warning,
        path,
        reason,
        f"[M3 dispatch] {path}{at}: {reason}; falling back (logged once per reason)",
    )


def capture_missed(path: str, rows: Optional[int], what: str) -> None:
    """``what`` is built on eager steps but a CUDA-graph capture found it missing (see ``missed``)."""
    missed(path, rows, f"{what} is not built and a CUDA graph is being captured")


def taken(path: str, rows: Optional[int], what: str = "") -> None:
    """``path`` ran on this rank at ``rows``: one rank-tagged info line per (path, rows, what), the engagement line
    every GEN rank prints for each fast path it takes (``what``: an optional detail, e.g. the projection)."""
    at = "" if rows is None else f" at {rows} rows"
    detail = f" ({what})" if what else ""
    _log_once(
        logger.info,
        path,
        f"taken: {rows}" + (f": {what}" if what else ""),
        f"[M3 dispatch] rank {mpi_rank()}: {path} taken{at}{detail}",
    )


def long_select_covers(num_blocks: int) -> bool:
    """Whether ``LONG_SELECT`` takes rows of ``num_blocks`` KV blocks (the engine's per-row block width)."""
    return LONG_SELECT_MIN_BLOCKS <= num_blocks <= LONG_SELECT_MAX_BLOCKS


def long_select_dsl_keys() -> Tuple[Tuple[bool, bool, int], ...]:
    """The (PDL launch, early trigger, trigger phase) keys of the 384-thread select a decode step can launch, compiled
    before warmup; none when ``LONG_SELECT_DSL`` covers nothing."""
    if not LONG_SELECT_DSL.keys:
        return ()
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    pdl = get_env_enable_pdl()
    keys = [(pdl, False, SELECT_LONG_TRIGGER_PHASE)]
    if SELECT_EARLY_MAX_ROWS:
        keys.append((pdl, True, 0))
    return tuple(keys)


def long_select_fallback(num_blocks: int, rows: int) -> None:
    """The in-tree selector is about to run instead of the long-row select: a miss when ``LONG_SELECT`` covers
    ``num_blocks`` (the kernel declined operands its cells certify; raises under require), uncovered otherwise (rows of
    at most 128 blocks, or wider than ``LONG_SELECT_MAX_BLOCKS``: the in-tree selector serves them in every mode)."""
    if long_select_covers(num_blocks):
        missed(
            LONG_SELECT.path,
            rows,
            f"the long-row select declined {num_blocks}-block scores it covers",
        )
    else:
        span = f"{LONG_SELECT_MIN_BLOCKS}..{LONG_SELECT_MAX_BLOCKS}"
        uncovered(
            LONG_SELECT.path, f"{num_blocks} KV blocks, outside {span}: the built-in selector"
        )


def uncovered(path: str, reason: str) -> None:
    """The step is outside ``path``'s covered shapes: log once per (path, reason); the generic path runs."""
    _log_once(logger.info, path, reason, f"[M3 dispatch] {path} not taken: {reason} (logged once)")


def reset_logged() -> None:
    """Forget which (path, reason) pairs were logged."""
    with _lock:
        _logged.clear()


# Kernel modules report from inside their calls through ops/m3_report: the same policy.
m3_report.install(
    on_taken=taken, on_missed=missed, on_uncovered=uncovered, on_capture_missed=capture_missed
)


# --------------------------------------------------------------------------------------------------------------------
# Target constants
# --------------------------------------------------------------------------------------------------------------------

# The generic boundary takes the low-latency multicast all-reduce + residual + RMSNorm up to this many tokens; with
# the shared + routed add folded in, up to its own maximum: the decode boundaries' certified rows.
MC_AR_MAX_TOKENS = GENERIC_MC_AR_MAX_TOKENS
MC_AR_FOLDED_MAX_TOKENS = GENERIC_MC_AR_FOLDED_MAX_TOKENS
# The shared expert folds into the routed engine from one token (dedicated shared-expert CTAs).
MOE_FOLD_MIN_TOKENS = 1  # capacity: fixed: a row lower bound
# SMs left to the concurrent kernels while a persistent routed engine runs.
MOE_ENGINE_RESERVED_SMS = 24  # capacity: fixed: an SM split, not a size
# Id of the target's final hidden state in Eagle3's ``layers_to_capture``.
FINAL_HIDDEN_LAYER_ID = -1  # capacity: fixed: an Eagle3 layer id, not a size
# The trtllm-gen MoE runner's routing kind for MiniMax-M3 (sigmoid + correction bias, renormalized, scaled) and its
# gated activation (SwiGLU with the per-expert alpha / beta / clamp operands).
ROUTING_MINIMAX2 = int(RoutingMethodType.MiniMax2)
ACT_SWIGLU = int(ActType_TrtllmGen.SwiGlu)
INDEX_SCORE_PATH = INDEX_SCORE.path

# --------------------------------------------------------------------------------------------------------------------
# Streaming: also emit a response on decoding iterations 2..FIRST_CONTENT_EMIT_ITERS (RuntimeHooks), so a request's
# first content leaves one step after its first token. 1: the stock schedule.
FIRST_CONTENT_EMIT_ITERS = 2

# The runtime-value dump (TRTLLM_MODELING_V2_VALUE_DUMP=1): the metadata of every fast-path and entry decision a step
# takes, recorded on eager steps and captures (replay runs no Python), printed once at exit.
# --------------------------------------------------------------------------------------------------------------------

_VALUE_DUMP = os.environ.get("TRTLLM_MODELING_V2_VALUE_DUMP") == "1"
_VALUES: set = set()


def _record(*meta: Hashable) -> None:
    if _VALUE_DUMP:
        _VALUES.add(meta)


def _print_values() -> None:
    if _VALUES:
        rank = os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("RANK", "0"))
        for meta in sorted(_VALUES, key=repr):
            print(f"[modeling_v2 value dump] rank {rank}: {meta}", flush=True)


if _VALUE_DUMP:
    atexit.register(_print_values)


def _is_vl_config(pretrained_config: PretrainedConfig) -> bool:
    model_type = getattr(pretrained_config, "model_type", None)
    if model_type == "minimax_m3_vl":
        return True
    if model_type == "minimax_m3":
        return False
    return any(
        "ForConditionalGeneration" in a
        for a in (getattr(pretrained_config, "architectures", None) or [])
    )


def _text_model_config(model_config: ModelConfig) -> ModelConfig:
    """The model config with the language model's ``text_config`` as its pretrained config.

    The checkpoint declares the vision-language wrapper; its ``text_config`` arrives as a plain dict (no typed
    sub-config is registered), so it is materialized into a ``PretrainedConfig`` with the wrapper's dtype and
    embedding tie carried over. A text config is returned unchanged.
    """
    if not _is_vl_config(model_config.pretrained_config):
        return model_config
    cfg = copy.deepcopy(model_config)
    text = getattr(cfg.pretrained_config, "text_config", None)
    if text is None:
        raise ValueError("MiniMax-M3 VL config has no `text_config`")
    if isinstance(text, dict):
        out = PretrainedConfig()
        for k, v in text.items():
            setattr(out, k, v)
        text = out
    if (
        getattr(text, "torch_dtype", None) is None
        and getattr(cfg.pretrained_config, "torch_dtype", None) is not None
    ):
        text.torch_dtype = cfg.pretrained_config.torch_dtype
    if (
        getattr(text, "tie_word_embeddings", None) is None
        and getattr(cfg.pretrained_config, "tie_word_embeddings", None) is not None
    ):
        text.tie_word_embeddings = cfg.pretrained_config.tie_word_embeddings
    return dataclasses.replace(cfg, pretrained_config=text)


def _validate_runtime(model_config: ModelConfig) -> None:
    """The runtime this target is written for. It is set by engine arguments rather than the checkpoint, so routing
    sees only its KV cache algorithm; this check at construction refuses every other runtime.

    The MiniMax-M3 MSA backend (it owns the step metadata and the index-K cache), the fused five-way QKV/index
    projection or the separate QKV and index projections, an FP8 indexer and an FP8 or NVFP4 main KV cache.
    """
    sparse = model_config.sparse_attention_config
    if (
        sparse is None
        or getattr(sparse, "algorithm", None) != "minimax_m3"
        or sparse.implementation != "msa"
    ):
        raise ValueError(
            "this MiniMax-M3 target requires sparse_attention_config.algorithm='minimax_m3' with implementation='msa'"
        )
    if getattr(sparse, "indexer_kv_dtype", None) != "fp8":
        raise ValueError(
            "this MiniMax-M3 target requires sparse_attention_config.indexer_kv_dtype='fp8'"
        )
    mode = model_config.quant_config.quant_mode
    if mode is None or not (mode.has_fp8_kv_cache() or mode.has_fp4_kv_cache()):
        raise ValueError("this MiniMax-M3 target requires an FP8 or NVFP4 KV cache")


def _sparse_layer_ids(cfg: PretrainedConfig) -> List[int]:
    freq = cfg.sparse_attention_config["sparse_attention_freq"]
    return [i for i, f in enumerate(freq) if int(f) != 0]


def _moe_layer_ids(cfg: PretrainedConfig) -> List[int]:
    freq = getattr(cfg, "moe_layer_freq", None)
    if freq is None:
        return list(range(cfg.num_hidden_layers))
    return [i for i, f in enumerate(freq) if int(f) != 0]


class _Mxfp8GemmState(nn.Module):
    """The MXFP8 GEMM backend state the engine's warmup drives.

    The engine finds MXFP8 projections by a module's ``quant_method`` and, for every one, opts it into native GEMM
    tuning and (on decode CUDA graphs) the auto-dispatched FlashInfer GEMM, then marks each tuned. Every projection of
    this target dispatches identically, so one method object carries that state for all of them; the forward reads it
    at each call (``_mx_mode``) and passes the resolved backend to the GEMM entries.
    """

    def __init__(self) -> None:
        super().__init__()
        self.quant_method = MXFP8LinearMethod()


@dataclass(frozen=True)
class MxMode:
    """The MXFP8 GEMM dispatch of one call, resolved from ``_Mxfp8GemmState`` and the engine's contexts."""

    decode_graph: bool  # inside a decode CUDA-graph warmup or capture
    autotune: bool  # inside the engine's FlashInfer MXFP8 tuning pass
    tuned_quantize_gemm: (
        bool  # per-bucket tuned quantize + GEMM (decode graphs, tune_decode_graph_backends)
    )
    flashinfer: bool  # FlashInfer's MXFP8 GEMM on the quantized activation
    native_tuning: (
        bool  # the native GEMM through the autotuner (warmup, before its tactics are registered)
    )


def _mx_mode(state: _Mxfp8GemmState) -> MxMode:
    method = state.quant_method
    decode_graph = flashinfer_mxfp8_decode_graph_capture_active()
    autotune = flashinfer_mxfp8_autotune_active()
    return MxMode(
        decode_graph=decode_graph,
        autotune=autotune,
        tuned_quantize_gemm=method.tune_decode_graph_backends and decode_graph,
        flashinfer=method.backend == "flashinfer"
        or (
            method.backend == "auto"
            and (autotune or (method._flashinfer_autotuned and decode_graph))
        ),
        native_tuning=method.needs_native_autotune,
    )


# --------------------------------------------------------------------------------------------------------------------
# Projections: the MXFP8 GEMM dispatch (``linear_path``) and its hand-offs
# --------------------------------------------------------------------------------------------------------------------


def _mxfp8_linear(
    core, x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, mode: MxMode
) -> torch.Tensor:
    """One MXFP8 projection on the engine-tuned path: MXFP8 quantize + GEMM, backend per ``mode``."""
    _record(
        "m3_mxfp8_linear",
        tuple(w.shape),
        mode.tuned_quantize_gemm,
        mode.flashinfer,
        mode.native_tuning,
    )
    return m3_mxfp8_linear(
        x.contiguous(),
        w,
        s,
        core.mx_alpha,
        torch.bfloat16,
        mode.tuned_quantize_gemm,
        mode.autotune,
        mode.flashinfer,
        mode.native_tuning,
    )


_PINNED_RUNNERS: dict = {}


def _pinned_runner(cls, dtype):
    runner = _PINNED_RUNNERS.get((cls, dtype))
    if runner is None:
        runner = _PINNED_RUNNERS[(cls, dtype)] = cls(dtype)
    return runner


def _pinned_mxfp8_gemm(act, act_scale, weight, weight_scale, out_dtype):
    """The decode-graph MXFP8 GEMM at the target's fixed tactic (``MXFP8_DECODE_GEMM_TACTIC``), no AutoTuner: the
    same choice on every rank and every start."""
    _record("pinned_mxfp8_gemm", tuple(weight.shape), act.shape[0], MXFP8_DECODE_GEMM_TACTIC)
    runner = _pinned_runner(FlashInferMXFP8GemmRunner, out_dtype)
    return runner.forward([act, act_scale, weight, weight_scale], tactic=MXFP8_DECODE_GEMM_TACTIC)


def _pinned_mxfp8_quantize(x):
    """The decode-graph MXFP8 activation quantize at the target's fixed tactic (``MXFP8_DECODE_QUANT_TACTIC``)."""
    _record("pinned_mxfp8_quantize", x.shape[0], MXFP8_DECODE_QUANT_TACTIC)
    return _pinned_runner(MXFP8QuantizeRunner, x.dtype).forward(
        [x], tactic=MXFP8_DECODE_QUANT_TACTIC
    )


_MX_TAC_ENGAGED: set = set()


def _note_mx_tac(kind: str, n: int, k: int, rows: int) -> None:
    """MX-TAC's engagement line, once per (activation kind, projection, rows)."""
    key = (kind, n, k, rows)
    if key not in _MX_TAC_ENGAGED:
        _MX_TAC_ENGAGED.add(key)
        logger.info(
            f"[M3 sm107 tp2] MX-TAC engaged: pinned MXFP8 decision (quantize tactic "
            f"{MXFP8_DECODE_QUANT_TACTIC}, GEMM tactic {MXFP8_DECODE_GEMM_TACTIC}) "
            f"for the [{n}, {k}] projection at {rows} rows, {kind} activation"
        )


def _gemv_variants(n: int, k: int, push: Optional[str], world: int):
    """Every (rows, bf16 activation, push (world, bucket tokens, mode)) call the decode dispatch routes to an [N, K]
    projection's GEMV: an MXFP8 activation on the rows ``MX_GEMV_ROWS`` gives it, a bf16 one up to
    ``MX_GEMV_BF16_MAX_TOKENS``; each pushing when the projection's boundary push covers the rows. ``push``: "attn"
    (the o_proj into the attention boundary), "dense" (the dense MLP's down projection into the post-MLP boundary)
    or "moe" (the shared expert's down projection into the MoE boundary)."""
    push_rows = {"attn": ATTN_PUSH.keys, "dense": DENSE_PUSH.keys, "moe": MOE_PUSH.keys}.get(
        push, frozenset()
    )
    variants = []
    max_rows = max([DECODE_MAX_TOKENS, *(max(r) for r in MX_GEMV_ROWS.values() if r)])
    for rows in range(1, max_rows + 1):
        for x_bf16, served in (
            (False, rows in MX_GEMV_ROWS.get((n, k), ())),
            (True, rows <= MX_GEMV_BF16_MAX_TOKENS.get((n, k), 0)),
        ):
            if not served:
                continue
            variants.append((rows, x_bf16, None))
            if rows in push_rows:
                bucket = next(b for b in m3_mc_allreduce.BUCKETS if rows <= b)
                variants.append((rows, x_bf16, (world, bucket, AR_PUSH_MODE)))
    return variants


def _uncompiled_note() -> str:
    key = mx_gemv.last_uncompiled
    return "" if key is None else f" (its kernel {key} was not compiled before warmup)"


def _linear(
    core, x: torch.Tensor, wk: str, sk: str, low_m_gemm: bool, push: Optional[tuple] = None
) -> Tuple[torch.Tensor, bool]:
    """One MXFP8 projection ``x @ w[wk]^T``; returns (output, whether its GEMV pushed into ``push``).

    Outside decode CUDA graphs: the engine-tuned MXFP8 quantize + GEMM. Inside them the projection's (N, K) and the
    step's rows pick one path (``linear_path``):

    1. an activation its producer already quantized to MXFP8 (``m3_prequant.SLOT``, taken here): the tcgen05 decode
       GEMV on the rows it serves, else the decode-graph MXFP8 GEMM on that activation;
    2. a bf16 activation on the rows the GEMV serves quantizing in its prologue;
    3. W-G1: a bf16 activation on the rows the MXFP8 GEMV serves but the bf16 GEMV does not
       (``MX_GEMV_QUANT_ROWS``): the decode-graph MXFP8 quantize, then the MXFP8 GEMV;
    4. otherwise the decode-graph MXFP8 quantize + GEMM.

    MX-TAC: on a decode graph whose MXFP8 projections the engine tunes (``tune_decode_graph_backends``), the GEMM
    (and, on a bf16 activation, the quantize) the GEMVs do not serve take one fixed decision
    (``MXFP8_DECODE_QUANT_TACTIC`` / ``MXFP8_DECODE_GEMM_TACTIC``: FlashInfer's CuTe DSL quantize and CuTe DSL GEMM)
    instead of the AutoTuner's per-start choice (whose L2-hot profiling left c2 bimodal at +-6%). On SM107 every TP2
    projection at 1..16 rows runs fastest on this pair, cold and L2-hot (graph us, cold: 4.7..10.3 against 5.0..11.5
    with the TRT-LLM quantize and 9.7..58 with CUTLASS), and all four candidates give the same bits (quantize bytes
    and scales, GEMM outputs). A constant is rank-agreed and start-independent by construction.

    A kernel declining rows its path covers is a miss; the chain then continues down the list. ``push`` (an
    ``m3_mx_gemv`` ``ar`` tuple) is the boundary all-reduce exchange the GEMV writes its rows into. ``low_m_gemm``
    marks the projections the fused low-M MXFP8 GEMM may take (``LOW_M_GEMM_MAX_TOKENS``: no TP2 configuration is
    built, so ``linear_path`` never selects it).
    """
    w, s = core.w[wk], core.w[sk]
    x = x.contiguous()
    mode = _mx_mode(core.mxfp8)
    if not mode.decode_graph:
        return _mxfp8_linear(core, x, w, s, mode), False
    use_pdl = get_env_enable_pdl()
    rows = x.shape[0]
    n, k = w.shape
    prequant = m3_prequant.SLOT.take(x)
    if prequant is None and x.dtype != torch.bfloat16:
        path = GENERIC_GEMM_PATH
    else:
        path = linear_path(n, k, rows, prequant is not None, low_m_gemm)
    _record("linear", (n, k), rows, path, push is not None)
    if prequant is not None:
        act, act_scale = prequant
        if path == MX_GEMV_PATH:
            output = m3_mx_gemv(act, act_scale, w, s, True, None, use_pdl, push)
            if output is not None:
                taken(path, rows, f"[{n}, {k}] mxfp8 in")
                return output, push is not None and mx_gemv.last_pushed
            missed(
                path,
                rows,
                f"the GEMV declined the [{n}, {k}] projection's MXFP8 activation"
                + _uncompiled_note(),
            )
        if path == PREQUANT_GEMM_PATH:
            uncovered(MX_GEMV_PATH, f"the [{n}, {k}] projection at rows the GEMV does not serve")
        if mode.tuned_quantize_gemm:
            output = _pinned_mxfp8_gemm(act, act_scale, w, s, torch.bfloat16)
            _note_mx_tac("mxfp8", n, k, rows)
            return output, False
        return m3_mxfp8_gemm(act, act_scale, w, s, torch.bfloat16, mode.autotune), False
    if path == MX_GEMV_PATH:
        # W-G1 (MX_GEMV_QUANT_ROWS): no MXFP8 hand-off, so the decode-graph quantize the generic GEMM path would run
        # (the same bytes), then the GEMV in place of that GEMM.
        act, act_scale = _pinned_mxfp8_quantize(x)
        output = m3_mx_gemv(act, act_scale, w, s, True, None, use_pdl, None)
        if output is not None:
            taken(path, rows, f"[{n}, {k}] mxfp8 quantize + GEMV")
            return output, False
        missed(
            path,
            rows,
            f"the GEMV declined the [{n}, {k}] projection's quantized activation"
            + _uncompiled_note(),
        )
        if mode.tuned_quantize_gemm:
            output = _pinned_mxfp8_gemm(act, act_scale, w, s, torch.bfloat16)
            _note_mx_tac("mxfp8", n, k, rows)
            return output, False
        return m3_mxfp8_gemm(act, act_scale, w, s, torch.bfloat16, mode.autotune), False
    if path == MX_GEMV_BF16_PATH:
        output = m3_mx_gemv_bf16(x, w, s, True, None, use_pdl, push)
        if output is not None:
            taken(path, rows, f"[{n}, {k}] bf16 in")
            return output, push is not None and mx_gemv.last_pushed
        missed(
            path, rows, f"the GEMV declined the [{n}, {k}] projection's input" + _uncompiled_note()
        )
    if path == GENERIC_GEMM_PATH:
        uncovered(
            MX_GEMV_BF16_PATH,
            f"the [{n}, {k}] projection's bf16 input beyond the decode GEMV's rows",
        )
    if mode.tuned_quantize_gemm and x.dtype == torch.bfloat16:
        act, act_scale = _pinned_mxfp8_quantize(x)
        output = _pinned_mxfp8_gemm(act, act_scale, w, s, torch.bfloat16)
        _note_mx_tac("bf16", n, k, rows)
        return output, False
    return _mxfp8_linear(core, x, w, s, mode), False


# --------------------------------------------------------------------------------------------------------------------
# Layer boundaries: all-reduce + residual + RMSNorm
# --------------------------------------------------------------------------------------------------------------------


def _can_mc(x, folded: bool = False) -> bool:
    """Whether the generic boundary takes the multicast all-reduce + residual + RMSNorm: the rows
    ``MC_AR_MAX_TOKENS`` / ``MC_AR_FOLDED_MAX_TOKENS`` (with the shared + routed add folded in) give it."""
    return (
        isinstance(x, torch.Tensor)
        and x.dim() == 2
        and x.shape[0] <= (MC_AR_FOLDED_MAX_TOKENS if folded else MC_AR_MAX_TOKENS)
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
    )


def _mc_allreduce_norm(core, x, x2, residual, norm_w, *, quant=False, prepushed=0, capture=None):
    """The multicast all-reduce + optional second addend + residual + RMSNorm (``m3_mc_allreduce_norm``), or None
    without a workspace or for operands the kernel does not take (the same on every rank). With ``quant`` the normed
    rows are also published, MXFP8-quantized, to ``m3_prequant.SLOT`` for the next MXFP8 projection; ``capture`` =
    (out, fc_norm weight, eps) also writes the layer's Eagle3 capture norm of the updated residual."""
    workspace = m3_mc_allreduce.get_workspace(core.mapping, x.shape[-1])
    if workspace is None:
        return None
    _record(
        "m3_mc_allreduce_norm", x.shape[0], x2 is not None, quant, prepushed, capture is not None
    )
    return m3_mc_allreduce.m3_mc_allreduce_norm(
        x,
        residual.contiguous(),
        norm_w,
        core.rms_eps,
        workspace,
        use_pdl=get_env_enable_pdl(),
        x2=x2,
        quant=quant,
        prepushed=prepushed,
        capture=capture,
    )


def _native_allreduce_norm(core, x, residual, norm_w):
    """The built-in fused all-reduce + residual + RMSNorm."""
    _record("m3_allreduce", x.shape[0])
    return m3_allreduce(
        x,
        core.allreduce,
        AllReduceParams(
            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
            residual=residual,
            norm_weight=norm_w,
            eps=core.rms_eps,
            trigger_completion_at_end=False,
        ),
    )


# --------------------------------------------------------------------------------------------------------------------
# The MoE: router, routed experts (trtllm-gen NVFP4), shared expert
# --------------------------------------------------------------------------------------------------------------------


@dataclass
class _MoeScalars:
    """The routed experts' kernel scalars of one layer, derived from the checkpoint's per-tensor scales once the
    weights are real (``MiniMaxM3Tp2ModelingV2._derive_moe``)."""

    fc31_input_scale: torch.Tensor
    fc2_input_scale: torch.Tensor
    fc31_alpha: torch.Tensor
    fc2_alpha: torch.Tensor
    fc31_scale_c: torch.Tensor
    act_alpha: torch.Tensor
    act_beta: torch.Tensor
    act_clamp: torch.Tensor


def _router_logits(core, i: int, h: torch.Tensor) -> torch.Tensor:
    """The fp32 router logits of MoE layer ``i`` when the MoE runs unfused outside ``ROUTER_TC_GEN``'s rows (the decode
    steps' engines route in-kernel): ``F.linear(float(h), w)``, the built-in gate's generation-step arithmetic; the
    experts then quantize their bf16 input themselves."""
    _record("m3_fp32_linear", h.shape[0] if h.dim() == 2 else 0)
    return m3_fp32_linear(h, core.w[f"l{i}_router"])


def _decode_router_logits(core, i: int, h: torch.Tensor) -> torch.Tensor:
    """The router logits of a pure generation step on the generic layer: ``ROUTER_TC_GEN``'s rows take the TF32
    tensor-core router when the device co-schedules 16-CTA clusters (decided at prebuild); every other step the fp32
    router (``_router_logits``)."""
    rows = h.shape[0] if h.dim() == 2 else 0
    if (
        ROUTER_TC_GEN.covers(rows)
        and core.router_tc_cluster16
        and h.dtype == torch.bfloat16
        and h.stride(1) == 1
        and (h.stride(0) * 2) % 16 == 0
    ):
        taken(ROUTER_TC_GEN.path, rows)
        _record("m3_router_tc", rows)
        return m3_router_tc(h, core.w[f"l{i}_router"], None, get_env_enable_pdl())[0]
    return _router_logits(core, i, h)


def _router_tc_fp4(core, i: int, h: torch.Tensor) -> Optional[tuple]:
    """(router logits, NVFP4 activation, its scales) of an unfused MoE on ``ROUTER_TC_GEN``'s rows, a generation or a
    context step: the TF32 tensor-core router with the routed experts' NVFP4 activation quantization fused in (the agg
    canon's router on these rows); None elsewhere (the caller routes on its fp32 router)."""
    rows = h.shape[0] if h.dim() == 2 else 0
    if not (
        ROUTER_TC_GEN.covers(rows)
        and core.router_tc_cluster16
        and h.dtype == torch.bfloat16
        and h.stride(1) == 1
        and (h.stride(0) * 2) % 16 == 0
    ):
        return None
    taken(ROUTER_TC_GEN.path, rows, "with the NVFP4 activation")
    _record("m3_router_tc", rows, True)
    return tuple(
        m3_router_tc(
            h, core.w[f"l{i}_router"], core.moe[i].fc31_input_scale.reshape(1), get_env_enable_pdl()
        )
    )


def _decode_routed_unfinalized(core, i: int, h: torch.Tensor):
    """The routed experts of a pure generation step, unfinalized (``[gemm2 out, expert scales, expanded->permuted
    index]``): routed on the fused tensor-core router where it covers the rows, else on the router logits with the
    experts quantizing their bf16 input."""
    fused = _router_tc_fp4(core, i, h)
    if fused is not None:
        logits, x_fp4, x_sf = fused
        return _routed_experts(core, i, x_fp4, x_sf, logits, False)
    return _routed_experts(core, i, h, None, _decode_router_logits(core, i, h), False)


def _triton_finalize(rows: int, routed, shared: Optional[torch.Tensor]) -> torch.Tensor:
    """The finalized routed output (+ ``shared``) from the runner's unfinalized parts via ops/m3_moe_finalize
    (``MOE_TRITON_FINALIZE``). Raises if the parts are not the trtllm-gen layout (the runner did not finalize them:
    no fallback)."""
    out = None
    if isinstance(routed, (list, tuple)) and len(routed) == 3:
        out = m3_moe_finalize(routed[0], routed[1], routed[2], shared)
    if out is None:
        raise RuntimeError(
            f"the routed experts' unfinalized output at {rows} rows is not the trtllm-gen layout"
        )
    taken(
        "m3_moe_finalize", rows, "Triton finalize" + (" + shared add" if shared is not None else "")
    )
    return out


def _triton_finalize_rows(core, h: torch.Tensor) -> bool:
    """Whether a finalized MoE forward of ``h`` takes ``MOE_TRITON_FINALIZE``: at least its rows, in one runner call."""
    rows = h.shape[0] if h.dim() == 2 else 0
    return (
        MOE_TRITON_FINALIZE
        and rows >= MOE_TRITON_FINALIZE_MIN_TOKENS
        and rows <= core.moe_max_num_tokens
    )


def _routed_experts(core, i: int, x, x_sf, router_logits, do_finalize: bool):
    """The trtllm-gen NVFP4 MoE runner over this rank's tensor-parallel slice of every expert, routing in-kernel.

    ``x_sf`` None means ``x`` is the bf16 activation, quantized here with the experts' global input scale. Returns
    the finalized rows, or the unfinalized ``[gemm2 out, expert scales, expanded->permuted index]``.
    """
    moe = core.moe[i]
    if x_sf is None:
        x, x_sf = fp4_quantize(x, moe.fc31_input_scale, 16, False, False)
    rows = x.shape[0]
    x_sf = x_sf.view(rows, -1).flatten()
    _record("fp4_block_scale_moe_runner", rows, do_finalize)
    outs = fp4_block_scale_moe_runner(
        router_logits,
        core.w[f"l{i}_router_bias"],
        x,
        x_sf.view(torch.float8_e4m3fn),
        core.w[f"l{i}_e_w31"],
        core.w[f"l{i}_e_s31"].view(torch.float8_e4m3fn),
        None,
        moe.act_alpha,
        moe.act_beta,
        moe.act_clamp,
        core.w[f"l{i}_e_w2"],
        core.w[f"l{i}_e_s2"].view(torch.float8_e4m3fn),
        None,
        moe.fc31_scale_c,
        moe.fc31_alpha,
        moe.fc2_alpha,
        core.num_experts,
        core.top_k,
        None,
        None,
        core.moe_inter,
        0,
        core.num_experts,
        core.routed_scaling_factor,
        ROUTING_MINIMAX2,
        do_finalize,
        act_type=ACT_SWIGLU,
        topk_weights=None,
        topk_ids=None,
        output=None,
        tune_max_num_tokens=core.max_num_tokens,
        use_dp=False,
    )
    return outs[0] if do_finalize else outs


def _split_chunk(rows: int, num_chunks: int) -> List[int]:
    """The MoE scheduler's even split: the first ``rows % num_chunks`` chunks take one more row."""
    base, extra = divmod(rows, num_chunks)
    return [base + 1] * extra + [base] * (num_chunks - extra)


# --------------------------------------------------------------------------------------------------------------------
# Attention
# --------------------------------------------------------------------------------------------------------------------


def _nvfp4_layer(manager, i: int) -> bool:
    return bool(getattr(manager, "is_nvfp4_layer", lambda _layer_idx: False)(i))


def _producer_covered(core) -> bool:
    """Whether ``PRODUCER`` covers this target's sparse layers: an NVFP4 KV cache at the TP2 heads (32 query, two KV
    and two index heads). A covered layer that cannot take ``m3_producer`` is a miss."""
    return core.main_kv_is_nvfp4 and (core.heads, core.kv_heads, core.index_heads) == (32, 2, 2)


def _producer_layout(kv_data, kv_scale) -> bool:
    """Whether a sparse layer's NVFP4 pools are the layout ``m3_producer`` is built for: [pages, 2, 2, 128, 64] with
    its scale pool [pages, 2, 2, 128, 8]."""
    return (
        kv_data is not None
        and kv_scale is not None
        and kv_data.dim() == 5
        and tuple(kv_data.shape[1:]) == (2, 2, 128, 64)
        and kv_data.stride()[2:] == (128 * 64, 64, 1)
        and kv_scale.dim() == 5
        and tuple(kv_scale.shape[1:]) == (2, 2, 128, 8)
        and kv_scale.stride()[2:] == (128 * 8, 8, 1)
        and kv_scale.shape[0] == kv_data.shape[0]
    )


def _nvfp4_fused_write_layout(data: torch.Tensor, scales: torch.Tensor, num_heads: int) -> bool:
    """Reject invalid HND pools; select the fused producer only for its packed layout."""
    if (
        data.dim() != 5
        or scales.dim() != 5
        or data.shape[0] <= 0
        or data.shape[0] != scales.shape[0]
        or tuple(data.shape[1:]) != (2, num_heads, 128, 64)
        or tuple(scales.shape[1:]) != (2, num_heads, 128, 8)
        or data.dtype not in (torch.int8, torch.uint8)
        or scales.dtype != torch.uint8
    ):
        raise ValueError("MiniMax-M3 NVFP4 data and scale caches must have matching HND geometry")
    for cache in (data, scales):
        if (
            cache.stride(4) != 1
            or cache.stride(3) != cache.shape[4]
            or cache.stride(2) < 128 * cache.stride(3)
            or cache.stride(1) < num_heads * cache.stride(2)
            or cache.stride(0) < 2 * cache.stride(1)
        ):
            raise ValueError(
                "MiniMax-M3 NVFP4 cache rows must be packed and heads/planes/pages must not overlap"
            )
    if scales.stride(2) != 128 * 8:
        raise ValueError("MiniMax-M3 NVFP4 MSA scale cache must have packed head blocks")
    return data.stride(2) == 128 * 64 and data.stride(0) % 2 == 0 and data.stride(1) % 2 == 0


#: The step phases ("decode" / "prefill") whose producer engagement has been logged, once per process.
_PRODUCER_ENGAGED: set = set()


def _producer_miss(covered: bool, rows: Optional[int], reason: str) -> None:
    if covered:
        missed(PRODUCER.path, rows, reason)
    return None


def _sparse_qkv_producer(
    core, i: int, packed, position_ids, kv_cache_manager, msa_idx_k_cache, msa_out_cache_loc
):
    """The horizontal producer of a sparse layer: Q / index-Q (FP8) out, K / V / index-K written into the paged
    caches (NVFP4 or FP8 main cache), for every step type (token-major). An NVFP4 cache at the TP2 heads takes
    ``m3_producer`` (``PRODUCER``); a covered layer that cannot is a miss. Returns (q, index_q), or None for caches
    neither kernel takes.

    ``kv_cache_manager``, ``msa_idx_k_cache`` (the per-layer index-K cache getter) and ``msa_out_cache_loc`` are the
    calling target's ``step_args`` entries."""
    covered = _producer_covered(core)
    rows = int(packed.shape[0])
    if packed.dtype != torch.bfloat16 or position_ids is None:
        return _producer_miss(
            covered,
            rows,
            "the packed QKV dtype, position ids, head dims, norm or RoPE configuration",
        )
    manager = kv_cache_manager
    if manager is None:
        return _producer_miss(covered, rows, "the step has no KV cache manager")
    buffers = manager.get_buffers(i, kv_layout="HND")
    index_k_cache = msa_idx_k_cache(i)
    out_cache_loc = msa_out_cache_loc
    layer_uses_nvfp4 = _nvfp4_layer(manager, i)
    scale_buffers = (
        manager.get_block_scale_buffers(i, kv_layout="HND") if layer_uses_nvfp4 else None
    )
    kv_quant_scale = core.inv_kv_scales[i] if layer_uses_nvfp4 else None
    if layer_uses_nvfp4:
        supported_main = (
            buffers is not None
            and scale_buffers is not None
            and _nvfp4_fused_write_layout(buffers, scale_buffers, core.kv_heads)
            and buffers.is_cuda
            and scale_buffers.is_cuda
            and scale_buffers.device == buffers.device
            and kv_quant_scale.is_cuda
            and kv_quant_scale.dtype == torch.float32
            and kv_quant_scale.numel() >= 3
            and kv_quant_scale.is_contiguous()
        )
    else:
        supported_main = (
            buffers is not None
            and buffers.is_cuda
            and buffers.dtype == torch.float8_e4m3fn
            and buffers.dim() == 5
            and tuple(buffers.shape[1:]) == (2, core.kv_heads, 128, 128)
            and buffers.stride(4) == 1
            and buffers.stride(3) == 128
            and buffers.stride(2) == 128 * 128
            and buffers.stride(1) >= core.kv_heads * buffers.stride(2)
            and buffers.stride(0) >= 2 * buffers.stride(1)
            and buffers.stride(0) % 4 == 0
            and buffers.stride(1) % 4 == 0
        )
    supported_index = (
        index_k_cache.is_cuda
        and index_k_cache.dtype == torch.float8_e4m3fn
        and index_k_cache.dim() == 4
        and tuple(index_k_cache.shape[1:]) == (1, 128, 128)
        and index_k_cache.stride(3) == 1
        and index_k_cache.stride(2) == 128
        and index_k_cache.stride(1) >= 128 * 128
        and index_k_cache.stride(0) >= index_k_cache.stride(1)
        and index_k_cache.stride(0) % 4 == 0
        and index_k_cache.stride(1) % 4 == 0
        and buffers is not None
        and index_k_cache.shape[0] == buffers.shape[0]
    )
    cos_sin = core.rotary_cos_sin
    supported_rope = (
        cos_sin.is_cuda
        and cos_sin.dtype == torch.float32
        and cos_sin.is_contiguous()
        and cos_sin.dim() == 3
        and tuple(cos_sin.shape[1:]) == (2, 32)
    )
    if (
        not supported_main
        or not supported_index
        or not supported_rope
        or out_cache_loc is None
        or not out_cache_loc.is_cuda
        or out_cache_loc.dtype != torch.int32
        or not out_cache_loc.is_contiguous()
        or out_cache_loc.numel() < rows
    ):
        return _producer_miss(
            covered and layer_uses_nvfp4,
            rows,
            "the layer's NVFP4 / index caches, RoPE table, KV quant scale or cache slots are not the layout the "
            "producer is built for",
        )
    w = core.w
    pos = position_ids.reshape(-1).contiguous().to(torch.int32)
    if (
        layer_uses_nvfp4
        and (core.heads, core.kv_heads, core.index_heads) == (32, 2, 2)
        and _producer_layout(buffers, scale_buffers)
    ):
        _record("m3_producer", rows)
        q, index_q = m3_producer(
            packed.contiguous(),
            buffers.view(torch.uint8),
            scale_buffers.view(torch.uint8),
            index_k_cache,
            out_cache_loc[:rows],
            kv_quant_scale,
            core.rms_eps,
            w[f"l{i}_q_norm"],
            w[f"l{i}_k_norm"],
            w[f"l{i}_idx_q_norm"],
            w[f"l{i}_idx_k_norm"],
            cos_sin,
            pos,
            core.heads,
            get_env_enable_pdl(),
            PRODUCER_EARLY_TRIGGER,
            PRODUCER_PREWAIT_STEP_INPUTS,
        )
        phase = "decode" if rows <= DECODE_MAX_TOKENS else "prefill"
        if phase not in _PRODUCER_ENGAGED:
            _PRODUCER_ENGAGED.add(phase)
            logger.info(
                f"[M3 sm107 tp2] fast path engaged: m3_producer ({phase}, T={rows}, "
                f"early_trigger={PRODUCER_EARLY_TRIGGER}, prewait_step_inputs={PRODUCER_PREWAIT_STEP_INPUTS})"
            )
    else:
        if layer_uses_nvfp4 and covered:
            missed(
                PRODUCER.path,
                rows,
                f"the NVFP4 pool {tuple(buffers.shape)} / scale pool {tuple(scale_buffers.shape)} is not the "
                "producer's layout",
            )
        _record("m3_qkv_insert", rows, layer_uses_nvfp4)
        q, index_q = m3_qkv_insert(
            packed.contiguous(),
            buffers.view(torch.uint8) if layer_uses_nvfp4 else buffers,
            scale_buffers,
            index_k_cache,
            out_cache_loc[:rows],
            kv_quant_scale,
            core.heads,
            core.kv_heads,
            core.index_heads,
            core.head_dim,
            core.rotary_dim,
            core.rms_eps,
            w[f"l{i}_q_norm"],
            w[f"l{i}_k_norm"],
            w[f"l{i}_idx_q_norm"],
            w[f"l{i}_idx_k_norm"],
            cos_sin,
            pos,
        )
    return q.flatten(1), index_q.flatten(1)


def _qk_norm_rope(core, i: int, qkv: torch.Tensor, position_ids, out_fp8: bool) -> torch.Tensor:
    """The per-head Gemma RMSNorm of Q and K and the partial NeoX RoPE of a dense layer's packed [Q | K | V], in one
    kernel: in place, or an FP8 copy (Q / K normed and roped, V cast) when ``out_fp8``."""
    qkv = qkv.contiguous()
    pos = position_ids.reshape(-1).contiguous().to(torch.int32)
    args = (
        qkv,
        core.heads,
        core.kv_heads,
        core.kv_heads,
        core.head_dim,
        core.rotary_dim,
        core.rms_eps,
        core.w[f"l{i}_q_norm"],
        core.w[f"l{i}_k_norm"],
        core.rope_theta,
        True,
        pos,
        1.0,  # factor: no YARN
        0.0,  # low
        0.0,  # high
        1.0,  # attention_factor
        True,  # is_qk_norm
        True,  # use_gemma
        False,  # use_mrope
        0,
        0,
    )
    _record("qk_norm_rope", qkv.shape[0], out_fp8)
    if out_fp8:
        return m3_qk_norm_rope_fp8(*args)
    return fused_qk_norm_rope(*args)


def _dense_fp8_op_applies(qkv: torch.Tensor, position_ids) -> bool:
    """Whether a dense layer's FP8 q/k/v come from this target's op (``minimax_m3_sm107_tp2_dense_qk_norm_rope_fp8``,
    else the BF16 fused_qk_norm_rope output cast to FP8)."""
    return DENSE_QKV_FP8_VIA_BF16 and position_ids is not None and qkv.dtype == torch.bfloat16


def _dense_qk_norm_rope_fp8(
    core, i: int, qkv: torch.Tensor, position_ids, kv_dest: Optional[tuple] = None
) -> torch.Tensor:
    """A dense layer's FP8 q/k/v: QK norm + RoPE with Q/K rounded to BF16 and then to E4M3 and V cast, the bits of the
    BF16 fused_qk_norm_rope output cast to FP8 (K is stored in the FP8 cache), in one launch of this target's op
    (``DENSE_QKV_FP8_VIA_BF16``); outside its supported inputs, the BF16 kernel and the cast.

    With ``kv_dest`` = (the layer's HND pool view, the step's int32 cache slots, live tokens, a [tokens, q heads, 128]
    FP8 buffer) the op also stores the live tokens' K and V rows into the pool (the bytes ``write_msa_phase_kv``
    stores, one launch instead of about ten) and every row's Q into the buffer; ``kv_dest`` is passed only where the
    op is known to run (``_dense_fp8_op_applies``)."""
    if not _dense_fp8_op_applies(qkv, position_ids):
        return fp8_e4m3_saturate(_qk_norm_rope(core, i, qkv, position_ids, False))
    _record("m3_dense_qk_norm_rope_fp8", qkv.shape[0], kv_dest is not None)
    args = (
        qkv.contiguous(),
        core.heads,
        core.kv_heads,
        core.kv_heads,
        core.rotary_dim,
        core.rms_eps,
        core.w[f"l{i}_q_norm"],
        core.w[f"l{i}_k_norm"],
        core.rope_theta,
        True,  # NeoX RoPE
        position_ids.reshape(-1).contiguous().to(torch.int32),
        True,  # Gemma norms
    )
    if kv_dest is not None:
        return torch.ops.trtllm.minimax_m3_sm107_tp2_dense_qk_norm_rope_fp8_kv(*args, *kv_dest)
    return torch.ops.trtllm.minimax_m3_sm107_tp2_dense_qk_norm_rope_fp8(*args)


def _dense_swiglu(core, gu: torch.Tensor) -> torch.Tensor:
    """The dense MLP's SwiGLU-OAI activation: this target's C++ op (``m3_dense_swiglu``, minimaxM3Sm107Tp2/
    swigluOai.cu: bit for bit the Triton silu_and_mul at limit 7, alpha 1.702, beta 1, PDL-launched with an early
    release so the down GEMV stages its weights during it) at the rows its cells certify, in any graph or eager step;
    otherwise the Triton kernel (``m3_silu_and_mul``). Indexer kdev17 (TP2 chain): -0.69 us per dense layer vs the
    Triton kernel. Every decline is logged once."""
    path = DENSE_SWIGLU_CPP_PATH
    rows = gu.shape[0]
    max_rows = max(cell.spec["rows"] for cell in m3_dense_swiglu.CELLS)
    params = (core.swiglu_alpha, core.swiglu_limit, core.swiglu_beta)
    act = None
    if rows > max_rows:
        uncovered(path, f"{rows} rows, past its certified {max_rows}")
    elif params != (DENSE_ALPHA, DENSE_LIMIT, DENSE_BETA):
        uncovered(path, f"activation parameters {params}")
    else:
        act = m3_dense_swiglu(
            gu, core.swiglu_limit, core.swiglu_alpha, core.swiglu_beta, get_env_enable_pdl()
        )
        if act is None:
            uncovered(path, "the gate_up layout")
        else:
            _record("m3_dense_swiglu", rows)
            taken(path, rows, "dense swigluoai")
    if act is None:
        act = m3_silu_and_mul(gu, core.swiglu_limit, core.swiglu_alpha, core.swiglu_beta)
    return act


def _fp8_index_qk_norm_rope(
    core, i: int, idx_qk: torch.Tensor, position_ids, msa_idx_k_cache, msa_out_cache_loc
) -> torch.Tensor:
    """The index branch of an unfused sparse layer: index-Q in FP8 out, index-K into the FP8 index-K cache, normed and
    roped in one kernel (the built-in ``_fused_fp8_index_qk_norm_rope``). ``msa_idx_k_cache`` / ``msa_out_cache_loc``
    are the calling target's ``step_args`` entries."""
    if position_ids is None or idx_qk.dtype != torch.bfloat16:
        raise NotImplementedError(
            "MiniMax-M3 fused FP8 indexer requires BF16 activations and position_ids."
        )
    index_k_cache = msa_idx_k_cache(i)
    if index_k_cache.dtype != torch.float8_e4m3fn:
        raise ValueError(
            "MiniMax-M3 fused FP8 indexer requires an E4M3 index-K cache, "
            f"got {index_k_cache.dtype}."
        )
    rows = int(idx_qk.shape[0])
    _record("m3_fp8_indexer_qk_norm_rope", rows)
    return m3_fp8_indexer_qk_norm_rope(
        idx_qk.contiguous(),
        index_k_cache,
        msa_out_cache_loc[:rows],
        core.index_heads,
        core.index_dim,
        core.rotary_dim,
        core.rms_eps,
        core.w[f"l{i}_idx_q_norm"],
        core.w[f"l{i}_idx_k_norm"],
        core.rope_theta,
        position_ids.reshape(-1).contiguous().to(torch.int32),
    ).flatten(1)


# --------------------------------------------------------------------------------------------------------------------
# Layer helpers both targets' bodies call: the Eagle3 capture hook and the norm after a layer
# --------------------------------------------------------------------------------------------------------------------


def _plain_capture(spec_metadata, layer_id: int, hidden_states, residual) -> None:
    """The in-tree Eagle3 hidden-state capture."""
    spec_metadata.maybe_capture_hidden_states(layer_id, hidden_states, residual)


def _next_norm(core, i: int) -> torch.Tensor:
    """The norm the boundary after layer ``i`` applies: the next layer's input norm, the final norm after the last."""
    return core.w[f"l{i + 1}_in_norm"] if i + 1 < core.num_layers else core.w["final_norm"]


# --------------------------------------------------------------------------------------------------------------------
# Kernel prebuilds the core runs before the executor's warmup
# --------------------------------------------------------------------------------------------------------------------


def _in_tree_scorer():
    """The in-tree CuTe DSL index scorer's runner (importing it registers ``cute_dsl_minimax_m3_index_decode_score``);
    imported on first use, as the CuTe DSL ops are."""
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
        CuteDSLMiniMaxM3IndexDecodeScoreRunner,
    )

    return CuteDSLMiniMaxM3IndexDecodeScoreRunner


def _dummy_scorer_operands(dtype, heads, qlen, dim, page, device):
    """One request of two blocks: real-shaped operands that load a compiled scorer module."""
    rows, blocks = 1, 2
    tokens = rows * qlen
    idx_q = torch.zeros(tokens, heads, dim, dtype=torch.bfloat16, device=device).to(dtype)
    idx_k = torch.zeros(rows * blocks, 1, page, dim, dtype=torch.bfloat16, device=device).to(dtype)
    block_table = torch.arange(rows * blocks, dtype=torch.int32, device=device).view(rows, blocks)
    seq_lens = torch.full((rows,), blocks * page, dtype=torch.int32, device=device)
    score = torch.empty(heads, blocks, tokens, dtype=torch.float32, device=device)
    return idx_q, idx_k.squeeze(1), block_table, score.transpose(1, 2), seq_lens


#: Mixed (context + generation) steps run their generation rows on the built-in sparse decode
#: (``MsaDecodeFmha._run_sparse``). Its Triton producer and merge specialize on the split-K factor (16 / 8 / 4 / 2 / 1
#: by the row count on SM107), on the row count's divisibility and on the 16-byte alignment of the request rows (a
#: mixed step's generation requests follow its context requests, so its block-table and length rows start mid-buffer),
#: so ``_prebuild_builtin_sparse_decode`` launches them per decode row count up to this many rows, both alignments.
BUILTIN_SPARSE_DECODE_PREBUILD_MAX_ROWS = (  # capacity: fixed: the prebuilt built-in decode rows
    64  # capacity: fixed: the mixed steps' generation rows prebuilt
)


def _prebuild_builtin_sparse_decode(query_lens, num_heads, num_kv_heads, device) -> list:
    """Compile (and load) the built-in sparse decode's Triton producer and merge for every generation row count a mixed
    step can have, so no step compiles one after the server is ready. Each launch mirrors the operands
    ``MsaDecodeFmha._run_sparse`` passes for an NVFP4 layer (FP8 q, the HND pool's K / V planes, the scale pool's
    planes, the head-major top-k view, BF16 output) on dummy values, on the current stream. Returns the row counts."""
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.triton_sparse_decode import (
        minimax_m3_sparse_attn_decode,
    )

    page, head_dim, topk = 128, 128, MSA_REQUIRED_TOPK
    pages = topk
    pool = torch.zeros(
        pages, 2, num_kv_heads, page, head_dim // 2, dtype=torch.uint8, device=device
    )
    scales = torch.zeros(
        pages, 2, num_kv_heads, page, head_dim // 16, dtype=torch.uint8, device=device
    )
    global_scales = torch.ones(2, 4, dtype=torch.float32, device=device)
    done = []
    for qlen in sorted(set(query_lens)):
        for rows in range(qlen, BUILTIN_SPARSE_DECODE_PREBUILD_MAX_ROWS + 1, qlen):
            requests = rows // qlen
            q = torch.zeros(rows, num_heads, head_dim, dtype=torch.bfloat16, device=device).to(
                torch.float8_e4m3fn
            )
            indexes = (
                torch.arange(topk, dtype=torch.int32, device=device)
                .repeat(rows, num_kv_heads, 1)
                .contiguous()
            )
            # One leading context request: its rows are skipped, so the generation rows start mid-buffer.
            block_rows = torch.arange(topk, dtype=torch.int32, device=device).repeat(
                requests + 1, 1
            )
            lengths = torch.full((requests + 1,), topk * page, dtype=torch.int32, device=device)
            output = torch.empty(rows, num_heads, head_dim, dtype=torch.bfloat16, device=device)
            for first in (0, 1):
                minimax_m3_sparse_attn_decode(
                    q,
                    pool[:, 0],
                    pool[:, 1],
                    indexes.permute(1, 0, 2),
                    block_rows[first : first + requests],
                    lengths[first : first + requests],
                    sm_scale=head_dim**-0.5,
                    output=output,
                    decode_query_len=qlen,
                    k_block_scale=scales[:, 0],
                    v_block_scale=scales[:, 1],
                    k_global_scale=global_scales[0, :1],
                    v_global_scale=global_scales[1, :1],
                )
            done.append(rows)
    return done


# --------------------------------------------------------------------------------------------------------------------
# The core
# --------------------------------------------------------------------------------------------------------------------

# Engine states (compiled kernel + workspace) per (device, token count[, shared fold]), shared by every layer: a step
# runs the layers one after another on one stream.
_ENGINE_STATES: Dict[tuple, object] = {}


class MiniMaxM3Tp2ModelingV2(DecoderModel):
    """The MiniMax-M3 text model of this target: its weights, its runtime resources, and the per-step dispatch."""

    def __init__(self, model_config: ModelConfig[PretrainedConfig]) -> None:
        _validate_runtime(model_config)
        super().__init__(model_config)
        cfg = model_config.pretrained_config
        mapping = model_config.mapping
        self.mapping = mapping
        # Whether the 9..16-row generation router takes the TF32 tensor-core kernel (set by prebuild_kernels).
        self.router_tc_cluster16 = False
        self.vocab_size = cfg.vocab_size
        d = _weights.MODEL_WEIGHTS.dims(self)
        self.num_layers = d.num_layers
        self.hidden = d.hidden
        self.heads = d.heads
        self.kv_heads = d.kv_heads
        self.head_dim = d.head_dim
        self.index_heads = d.index_heads
        self.index_dim = d.index_dim
        self.q_size = d.heads * d.head_dim
        self.kv_size = d.kv_heads * d.head_dim
        self.moe_inter = d.moe_inter
        self.num_experts = d.num_experts
        self.top_k = int(cfg.num_experts_per_tok)
        self.routed_scaling_factor = float(getattr(cfg, "routed_scaling_factor", 1.0))
        self.rms_eps = float(cfg.rms_norm_eps)
        self.swiglu_alpha = float(getattr(cfg, "swiglu_alpha", 1.702))
        self.swiglu_beta = 1.0  # the (up + 1) offset of SwiGLU-OAI
        self.swiglu_limit = float(getattr(cfg, "swiglu_limit", 7.0))
        self.max_num_tokens = model_config.max_num_tokens
        self.moe_max_num_tokens = model_config.moe_max_num_tokens
        self.sparse_layers = frozenset(_sparse_layer_ids(cfg))
        self.moe_layers = frozenset(_moe_layer_ids(cfg))
        mode = model_config.quant_config.quant_mode
        self.main_kv_is_fp8 = bool(mode is not None and mode.has_fp8_kv_cache())
        self.main_kv_is_nvfp4 = bool(mode is not None and mode.has_fp4_kv_cache())
        # The sparse layers' QKV and index projections: one five-way packed projection, or the QKV projection and a
        # separate BF16 index projection (fuse_qkv_index_projection=False) run on two streams.
        self.fused_projection = bool(
            getattr(model_config.sparse_attention_config, "fuse_qkv_index_projection", False)
        )

        self.w = _weights.MODEL_WEIGHTS.declare(d)

        # The engine reads these two off the module tree: the MXFP8 GEMM state it tunes, and the opt-in to the
        # automatic FlashInfer MXFP8 path on decode graphs.
        self.mxfp8 = _Mxfp8GemmState()
        self._use_flashinfer_mxfp8_decode_graph_default = True

        # The built-in fused all-reduce + residual + RMSNorm of the layer boundaries the multicast kernel does not take.
        self.allreduce = AllReduce(
            mapping=mapping, strategy=model_config.allreduce_strategy, dtype=cfg.torch_dtype
        )

        # Streams: the shared / routed expert overlap and, when the MoE chunks its input, the chunking overlap.
        self.aux_stream_moe = torch.cuda.Stream()
        chunked = self.moe_max_num_tokens < model_config.max_num_tokens * mapping.dp_size
        self.aux_stream_moe_chunk = torch.cuda.Stream() if chunked else None
        self.moe_events = {i: (torch.cuda.Event(), torch.cuda.Event()) for i in self.moe_layers}
        self.moe_chunk_events = {
            i: (torch.cuda.Event(), torch.cuda.Event()) for i in self.moe_layers
        }
        # The unfused sparse layers' index branch overlaps the QKV branch on its own stream.
        self.aux_stream_attn = None if self.fused_projection else torch.cuda.Stream()
        self.attn_events = (
            {}
            if self.fused_projection
            else {i: (torch.cuda.Event(), torch.cuda.Event()) for i in self.sparse_layers}
        )

        # Partial NeoX RoPE (rotary_dim of head_dim): the cos/sin table every attention kernel reads.
        rope = RopeParams.from_config(cfg)
        self.rope_theta = rope.theta
        self.rotary_dim = int(rope.dim)
        self.rotary_cos_sin = rope.create_rope_const_params(interleave=False)[1].reshape(
            rope.max_positions, 2, -1
        )

        # One MSA attention backend per layer (dense and sparse; the step metadata is the backend's).
        quant_config = model_config.get_quant_config()
        self.attn: List[MiniMaxM3MsaSparseAttention] = []
        for i in range(self.num_layers):
            backend = create_attention(
                model_config.attn_backend,
                i,
                self.heads,
                self.head_dim,
                self.kv_heads,
                pos_embd_params=None,
                quant_config=quant_config,
                skip_create_weights_in_init=model_config.skip_create_weights_in_init,
                q_scaling=1.0,
                sparse_params=model_config.sparse_attention_config.to_sparse_params(
                    pretrained_config=cfg, layer_idx=i
                ),
            )
            backend.update_quant_config(quant_config)
            if not isinstance(backend, MiniMaxM3MsaSparseAttention):
                raise ValueError(
                    f"layer {i}: expected the MiniMax-M3 MSA backend, got {type(backend).__name__}"
                )
            if i not in self.sparse_layers:
                # The KV split trtllm-gen dense decode runs at, per request count.
                backend.dense_decode_max_seq_len_by_batch = DENSE_DECODE_MAX_SEQ_LEN_BY_BATCH
            self.attn.append(backend)

        # Built once the weights are real (``post_load_weights``): the KV (Q, K, V) scales, the MoE scalars, the
        # decode engines and workspaces, the two targets.
        self.kv_scales: Dict[int, torch.Tensor] = {}
        self.inv_kv_scales: Dict[int, torch.Tensor] = {}
        self.moe: Dict[int, _MoeScalars] = {}
        self.mx_alpha: Optional[torch.Tensor] = None
        self.moe_engines: Dict[int, Dict[Tuple[int, bool, bool], Optional[object]]] = {}
        self.embed_owners: Dict[tuple, DecodeEmbedding] = {}
        # Each sparse layer's selected block ids per (KV head, query row, slot) at its last long-context decode, which
        # that decode prefetches into L2 before its grid wait (``SPARSE_DECODE_LC_PREFETCH_SELF_KV``, up to
        # ``N1_MAX_ROWS`` query rows).
        self.n1_ids: Dict[int, torch.Tensor] = (
            {
                i: torch.full(
                    (self.kv_heads, N1_MAX_ROWS, MSA_REQUIRED_TOPK),
                    -1,
                    dtype=torch.int32,
                    device=torch.device("cuda", torch.cuda.current_device()),
                )
                for i in sorted(self.sparse_layers)
            }
            if SPARSE_DECODE_LC_PREFETCH_SELF_KV
            else {}
        )
        self._built_for: Optional[tuple] = None
        self._prefill: Optional[PrefillTarget] = None
        self._decode: Optional[DecodeTarget] = None

    def sm_scale(self, i: int) -> float:
        return (self.head_dim**-0.5) / float(self.attn[i].q_scaling)

    # ---------------------------------------------------------------------------------------------- after the load

    def post_load_weights(self) -> None:
        """Derive the kernels' state from the loaded weights, then build both targets and every decode kernel.

        Idempotent per set of weight storages: the loader's post-load walk reaches this module directly and through
        the shell, and a second call on the same storages does nothing.
        """
        identity = (
            self.w["embed"].data_ptr(),
            self.w["l0_in_norm"].data_ptr(),
            self.w["final_norm"].data_ptr(),
        )
        if self._built_for == identity:
            return
        device = self.w["embed"].device
        derived = nn.Module()
        derived.register_buffer(
            "mx_alpha", torch.ones(1, dtype=torch.float32, device=device), persistent=False
        )
        for i in range(self.num_layers):
            # No KV calibration in the checkpoint: unit (Q, K, V) scales and their reciprocals.
            derived.register_buffer(
                f"l{i}_kv_scales",
                torch.ones(3, dtype=torch.float32, device=device),
                persistent=False,
            )
            derived.register_buffer(
                f"l{i}_inv_kv_scales",
                torch.ones(3, dtype=torch.float32, device=device),
                persistent=False,
            )
        for i in sorted(self.moe_layers):
            for name, value in dataclasses.asdict(self._derive_moe(i)).items():
                derived.register_buffer(f"l{i}_{name}", value, persistent=False)
        self.derived = derived
        self.mx_alpha = derived.mx_alpha
        self.kv_scales = {i: getattr(derived, f"l{i}_kv_scales") for i in range(self.num_layers)}
        self.inv_kv_scales = {
            i: getattr(derived, f"l{i}_inv_kv_scales") for i in range(self.num_layers)
        }
        self.moe = {
            i: _MoeScalars(
                **{
                    f.name: getattr(derived, f"l{i}_{f.name}")
                    for f in dataclasses.fields(_MoeScalars)
                }
            )
            for i in sorted(self.moe_layers)
        }
        self._build_embed_owners()
        self.prebuild_kernels()
        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)
        self._built_for = identity

    def cache_derived_state(self) -> None:
        """The staged loaders' entry to the same build (``post_load_weights``)."""
        self.post_load_weights()

    def _derive_moe(self, i: int) -> _MoeScalars:
        """The routed experts' kernel scalars from the checkpoint's per-tensor scales (trtllm-gen NVFP4 layout).

        Global input scales: the reciprocal of the largest per-expert input scale (gate and up must agree). Per expert:
        ``alpha = 1 / (input_scale * (1 / weight_scale_2))`` for the gate (fc31_alpha), the up half (its GEMM2-input
        rescale ``fc31_scale_c = fc2_input_scale * up_alpha``) and the down projection; SwiGLU-OAI's activation
        constants per expert, with beta and the clamp in the pre-dequant GEMM output domain (divided by fc31_alpha).
        """
        w = self.w
        isc1, isc3, isc2 = w[f"l{i}_e_isc1"].data, w[f"l{i}_e_isc3"].data, w[f"l{i}_e_isc2"].data
        if not torch.allclose(isc1, isc3):
            raise ValueError(f"layer {i}: the routed experts' w1 and w3 input scales differ")
        fc31_input_scale = isc1.max().reciprocal()
        fc2_input_scale = isc2.max().reciprocal()
        ws2_1, ws2_3, ws2_2 = (
            w[f"l{i}_e_ws2_1"].data,
            w[f"l{i}_e_ws2_3"].data,
            w[f"l{i}_e_ws2_2"].data,
        )
        fc31_alpha = 1.0 / (fc31_input_scale * (1.0 / ws2_1))
        up_alpha = 1.0 / (fc31_input_scale * (1.0 / ws2_3))
        fc2_alpha = 1.0 / (fc2_input_scale * (1.0 / ws2_2))
        fc31_scale_c = fc2_input_scale * up_alpha
        n = self.num_experts
        device = fc31_alpha.device
        act_alpha = torch.full((n,), self.swiglu_alpha, dtype=torch.float32, device=device)
        act_beta = torch.full((n,), self.swiglu_beta, dtype=torch.float32, device=device)
        act_clamp = torch.full((n,), self.swiglu_limit, dtype=torch.float32, device=device)
        act_beta.div_(fc31_alpha)
        act_clamp.div_(fc31_alpha)
        return _MoeScalars(
            fc31_input_scale=fc31_input_scale,
            fc2_input_scale=fc2_input_scale,
            fc31_alpha=fc31_alpha,
            fc2_alpha=fc2_alpha,
            fc31_scale_c=fc31_scale_c,
            act_alpha=act_alpha,
            act_beta=act_beta,
            act_clamp=act_clamp,
        )

    def _build_embed_owners(self) -> None:
        """The fused embedding + first RMSNorm's caller-owned buffers, one per (rows, id dtype) of a decode step."""
        table, weight = self.w["embed"], self.w["l0_in_norm"]
        self.embed_owners = {
            (rows, dtype): DecodeEmbedding(table, weight, self.rms_eps, rows, dtype)
            for rows in range(1, EMBED_FIRSTNORM_MAX_TOKENS + 1)
            for dtype in (torch.int32, torch.int64)
        }

    def projections(self) -> List[Tuple[str, str, Optional[str]]]:
        """Every MXFP8 projection in layer order: (weight key, scale key, the boundary its GEMV pushes into)."""
        out = []
        for i in range(self.num_layers):
            out.append((f"l{i}_qkv_w", f"l{i}_qkv_s", None))
            out.append((f"l{i}_o_w", f"l{i}_o_s", "attn"))
            if i in self.moe_layers:
                out.append((f"l{i}_sh_gu_w", f"l{i}_sh_gu_s", None))
                out.append((f"l{i}_sh_dn_w", f"l{i}_sh_dn_s", "moe"))
            else:
                out.append((f"l{i}_mlp_gu_w", f"l{i}_mlp_gu_s", None))
                out.append((f"l{i}_mlp_dn_w", f"l{i}_mlp_dn_s", "dense"))
        return out

    def decode_query_lens(self) -> List[int]:
        """The query rows per request a decode step can have: 1, and 1 + each draft length speculation runs."""
        lens = {1}
        spec = getattr(self.model_config, "spec_config", None)
        if spec is not None:
            for drafts in (
                getattr(spec, "max_draft_len", None),
                getattr(spec, "max_total_draft_tokens", None),
                *(getattr(spec, "draft_len_schedule", None) or {}).values(),
            ):
                if drafts:
                    lens.add(1 + int(drafts))
        return sorted(lens)

    def prebuild_kernels(self) -> Dict[str, object]:
        """Compile (and load) every kernel a decode forward dispatches to that would otherwise compile on its first
        call: each projection's decode GEMVs (every row count, activation and boundary push), the MoE engines of
        ``MOE_ENGINE``'s token counts, the index scorers at the decode query lengths (this target's and the in-tree
        one wider steps take), the 384-thread long-row select, the dense layers' full-context decode plans, and the
        built-in sparse decode a mixed step's generation rows run. Runs on the current (main) stream before the
        executor's warmup, in the same order on every rank, then synchronizes; after it, GEMV compiles are frozen and
        the forward only looks kernels up (a missing one is a miss). Returns the seconds each part took and the keys.
        """
        times: Dict[str, object] = {}
        # The 9..16-row generation router: decided once, from the device, never inside a CUDA-graph capture.
        # The built-in gate's conditions, mirrored exactly: 128 experts, a hidden size in 1024s, 16-CTA clusters.
        cfg = self.model_config.pretrained_config
        self.router_tc_cluster16 = (
            int(cfg.num_local_experts) == 128
            and int(cfg.hidden_size) % 1024 == 0
            and bool(torch.ops.trtllm.minimax_m3_router_tc_cluster16(int(cfg.hidden_size)))
        )
        logger.info(
            f"[M3 prebuild] rank {mpi_rank()}: TF32 tensor-core router for {ROUTER_TC_GEN_MIN_ROWS}.."
            f"{ROUTER_TC_GEN_MAX_ROWS}-row generation steps: "
            + (
                "engaged (16-CTA clusters)"
                if self.router_tc_cluster16
                else "not used (no 16-CTA clusters)"
            )
        )
        if not self.mxfp8.quant_method.use_cutlass:
            raise ValueError("the MiniMax-M3 decode dispatch needs the CUTLASS MXFP8 GEMM")
        from .ops import m3_fullctx_decode as _fullctx_decode_kernel
        from .ops import m3_select_long_dsl as _select_long_dsl_kernel
        from .ops import m3_tokmap_guard

        device = self.w["embed"].device
        # The one-request token-map guard of the stock generation preprocess (``PREPROCESS_TOKMAP_FIX``): every path
        # that can run it (the drafter's full-context decode or its fallback, any trtllm-gen fallback), so it is
        # installed whatever else is enabled.
        m3_tokmap_guard.install(
            torch.device("cuda", torch.cuda.current_device()),
            DECODE_MAX_TOKENS,
            lambda rows: taken(PREPROCESS_TOKMAP_PATH, rows, "one-request token map"),
            lambda rows, reason: missed(PREPROCESS_TOKMAP_PATH, rows, reason),
        )
        mx_gemv.freeze(False)
        use_pdl = get_env_enable_pdl()
        world = int(self.mapping.tp_size)
        start = time.perf_counter()
        for wk, sk, push in self.projections():
            weight, scale = self.w[wk], self.w[sk]
            mx_gemv.ensure_compiled(weight, scale, use_pdl)
            n, k = weight.shape
            for rows, x_bf16, ar in _gemv_variants(n, k, push, world):
                mx_gemv.precompile(weight, scale, rows, x_bf16, use_pdl=use_pdl, ar=ar)
        torch.cuda.synchronize()
        times["gemv_s"] = time.perf_counter() - start
        start = time.perf_counter()
        for i in sorted(self.moe_layers):
            self._prebuild_engines(i)
        torch.cuda.synchronize()
        times["moe_s"] = time.perf_counter() - start
        start = time.perf_counter()
        times["scorer_keys"] = self._prebuild_index_scorer(self.decode_query_lens(), device)
        torch.cuda.synchronize()
        times["scorer_s"] = time.perf_counter() - start
        start = time.perf_counter()
        times["select_keys"] = []
        if long_select_dsl_keys():
            # The 384-thread long-row select (``SELECT_LONG_DSL``): one kernel per launch key.
            times["select_keys"] = _select_long_dsl_kernel.prebuild(long_select_dsl_keys(), device)
            torch.cuda.synchronize()
        times["select_s"] = time.perf_counter() - start
        start = time.perf_counter()
        times["fullctx_plans"] = 0
        if FULLCTX_DECODE.keys:
            # The dense layers' full-context decode: one compiled PrimTS plan per covered (rows, requests).
            times["fullctx_plans"] = _fullctx_decode_kernel.prebuild(
                device,
                {key: FULLCTX_DECODE_RECIPES[key] for key in FULLCTX_DECODE.keys},
                FULLCTX_DECODE_MAX_KV_LEN,
                FULLCTX_DECODE_OWN_REDUCER and _fullctx_decode_kernel.has_own_reducer(),
                use_pdl,
            )
            torch.cuda.synchronize()
        times["fullctx_s"] = time.perf_counter() - start
        start = time.perf_counter()
        times["builtin_decode_rows"] = _prebuild_builtin_sparse_decode(
            self.decode_query_lens(), self.heads, self.kv_heads, device
        )
        torch.cuda.synchronize()
        times["builtin_decode_s"] = time.perf_counter() - start
        mx_gemv.freeze(True)
        logger.info(
            "[M3 prebuild] decode kernels built before warmup: GEMVs %.1f s, MoE engines %.1f s, index scorer "
            "%.1f s (%d variants), long-row select %.1f s (%d variants), full-context decode %.1f s (%d plans), "
            "built-in sparse decode %.1f s (%d row counts)"
            % (
                times["gemv_s"],
                times["moe_s"],
                times["scorer_s"],
                len(times["scorer_keys"]),
                times["select_s"],
                len(times["select_keys"]),
                times["fullctx_s"],
                times["fullctx_plans"],
                times["builtin_decode_s"],
                len(times["builtin_decode_rows"]),
            )
        )
        return times

    # ------------------------------------------------------------------------------------------ the MoE engines

    def _expert_operands(self, i: int):
        """The routed experts' operands the engines read (trtllm-gen NVFP4 SwiGLU-OAI layout), and the input scale."""
        w, m = self.w, self.moe[i]
        tensors = [
            w[f"l{i}_e_w31"].data,
            w[f"l{i}_e_s31"].data,
            w[f"l{i}_e_w2"].data,
            w[f"l{i}_e_s2"].data,
            m.fc31_alpha,
            m.fc31_scale_c,
            m.act_alpha,
            m.act_beta,
            m.act_clamp,
            m.fc2_alpha,
        ]
        return tensors, m.fc31_input_scale

    def _shared_weights(self, i: int):
        w = self.w
        return (
            w[f"l{i}_sh_gu_w"].data.view(torch.float8_e4m3fn),
            w[f"l{i}_sh_gu_s"].data,
            w[f"l{i}_sh_dn_w"].data.view(torch.float8_e4m3fn),
            w[f"l{i}_sh_dn_s"].data,
        )

    def engine(self, i: int, num_tokens: int, shared: bool) -> Optional[object]:
        """Layer ``i``'s prebuilt routed-expert engine for ``num_tokens`` (1..8), router folded in (and the shared
        expert with ``shared``). None after a miss: the build failed or was never made (a forward never builds one)."""
        return self._lookup(
            i, self.moe_engines[i], (num_tokens, True, shared), MOE_ENGINE.path, num_tokens
        )

    def _lookup(self, i: int, engines: dict, key, path: str, num_tokens: int) -> Optional[object]:
        if key not in engines:
            what = f"layer {i}'s engine"
            if torch.cuda.is_current_stream_capturing():
                capture_missed(path, num_tokens, what)
            else:
                missed(path, num_tokens, f"{what} was not built before warmup")
            return None
        engine = engines[key]
        if engine is None:
            missed(path, num_tokens, "its build failed (logged)")
        return engine

    def _prebuild_engines(self, i: int) -> None:
        """Build (compile, on first use of each shared state, and warm) every engine a decode step of layer ``i`` can
        take: ``MOE_ENGINE``'s token counts with the shared expert folded in. Whether an engine is kept is agreed
        across the TP group (one host collective per engine): a build that fails on one rank drops that engine on
        every rank, so the ranks never mix an engine (and its boundary push) with the generic MoE."""
        engines = self.moe_engines.setdefault(i, {})
        for num_tokens in sorted(MOE_ENGINE.keys):
            key = (num_tokens, True, True)
            if key not in engines:
                engines[key] = m3_mc_allreduce.tp_agree_built(
                    self.mapping,
                    self._try_build(
                        i, MOE_ENGINE.path, num_tokens, self._build_engine, shared=True
                    ),
                    f"[M3 dispatch] {MOE_ENGINE.path} at {num_tokens} rows: layer {i}'s engine",
                )

    def _try_build(self, i: int, path: str, num_tokens: int, build, **kwargs) -> Optional[object]:
        try:
            return build(i, num_tokens, **kwargs)
        except (ImportError, RuntimeError, ValueError) as err:
            logger.warning(
                f"[M3 dispatch] {path} at {num_tokens} rows: layer {i}'s engine failed to build: {err}"
            )
            return None

    def _build_engine(self, i: int, num_tokens: int, shared: bool) -> object:
        """The 1..8-token engine (router, routed experts and the shared expert in one launch, at the TP2 width 1536).
        Rubin TP2 ablation chain (57 layers behind a 58.4 MB PDL producer, row = (P + MoE) - P, both orders): 31.3 /
        50.7 us/layer at 4 / 8 tokens against the built-in MoE's 42.4 / 55.4 (FlashInfer MXFP8 shared expert); bit for
        bit the built-in block's shared.add_(routed). When the engine pushes its sum into the boundary all-reduce,
        every CTA releases the PDL dependents right after its grid dependency wait instead of after the push
        (``MOE_EARLY_AR_RELEASE``), so the all-reduce (which polls the pushed tagged words, not the grid) becomes
        resident on the SMs the engine frees: scheduling only, the same bits (Rubin TP2 greedy early column, same-job
        A,B + B,A: c1 -15 us)."""
        tensors, scale = self._expert_operands(i)
        device = tensors[0].device
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        kwargs = {}
        if shared:
            kwargs["shared_weights"] = self._shared_weights(i)
        state = _ENGINE_STATES.get((str(device), num_tokens, shared))
        if state is None:
            state = _ENGINE_STATES[(str(device), num_tokens, shared)] = M3MoeState(
                device,
                num_tokens=num_tokens,
                num_ctas=sms - MOE_ENGINE_RESERVED_SMS,
                shared=shared,
                early_release=MOE_EARLY_AR_RELEASE,
            )
        engine = state.layer(
            *tensors,
            self.w[f"l{i}_router_bias"].data,
            router_weight=self.w[f"l{i}_router"].data,
            fc31_input_scale=scale.reshape(1).float(),
            **kwargs,
        )
        # Compile (first use of the state) and warm.
        m3_moe_engine(
            torch.zeros(num_tokens, tensors[0].shape[-1] * 2, dtype=torch.bfloat16, device=device),
            engine,
        )
        return engine

    # ------------------------------------------------------------------------------------------ the index scorers

    def _index_geometries(self) -> list:
        """Every sparse layer's (index dtype, index heads per rank, index dim), deduplicated."""
        geoms = []
        for i in sorted(self.sparse_layers):
            backend = self.attn[i]
            config = backend.m3_config
            dtype = torch.float8_e4m3fn if backend.indexer_kv_dtype == "fp8" else torch.bfloat16
            geom = (dtype, int(config.num_index_heads), int(config.sparse_index_dim))
            if geom not in geoms:
                geoms.append(geom)
        return geoms

    def _scorer_variants(self, query_lens) -> list:
        """The scorer keys a decode step can take: every sparse layer's index geometry at each decode query length,
        with each covered request count's schedule (``index_score_schedule``), where the kernel serves it."""
        runner = _index_score_kernel.IndexScoreRunner
        keys = []
        for dtype, heads, dim in self._index_geometries():
            for qlen in sorted(set(query_lens)):
                if not runner.is_supported(
                    q_dtype=dtype,
                    num_heads=heads,
                    head_dim=dim,
                    page_size=runner.SUPPORTED_PAGE_SIZE,
                    max_decode_query_len=qlen,
                ):
                    continue
                for requests in sorted(INDEX_SCORE.keys):
                    split_k, early = index_score_schedule(requests)
                    key = (dtype, heads, int(qlen), dim, int(split_k), bool(early))
                    if key not in keys:
                        keys.append(key)
        return keys

    def _in_tree_scorer_variants(self, query_lens) -> list:
        """The in-tree CuTe DSL scorer's keys (``CuteDSLMiniMaxM3IndexDecodeScoreRunner``: index dtype, index heads,
        decode query length, index dim, its 256-CTA split) a decode step outside ``INDEX_SCORE`` can take: every query
        length at which a step of more than ``INDEX_SCORE_MAX_REQUESTS`` requests fits in ``DECODE_MAX_TOKENS`` (e.g.
        3..8 requests decoding one row each), where that kernel serves the geometry."""
        in_tree = _in_tree_scorer()
        widest = DECODE_MAX_TOKENS // (INDEX_SCORE_MAX_REQUESTS + 1)
        keys = []
        for dtype, heads, dim in self._index_geometries():
            for qlen in sorted(set(query_lens)):
                if qlen > widest or not in_tree.is_supported(
                    q_dtype=dtype,
                    num_heads=heads,
                    head_dim=dim,
                    page_size=in_tree.SUPPORTED_PAGE_SIZE,
                    max_decode_query_len=qlen,
                ):
                    continue
                key = (dtype, heads, qlen, dim, in_tree.SPLIT_K)
                if key not in keys:
                    keys.append(key)
        return keys

    def _prebuild_index_scorer(self, query_lens, device) -> list:
        """Compile every key of ``_scorer_variants`` and ``_in_tree_scorer_variants`` and launch each once on
        real-shaped dummy operands (loading its module), on the current stream, so no decode step compiles a scorer.
        The kernels' batch, token, page and block counts are symbolic: their compiles need no step metadata. Returns
        the keys."""
        runner = _index_score_kernel.IndexScoreRunner
        keys = self._scorer_variants(query_lens)
        for key in keys:
            dtype, heads, qlen, dim, split_k, early = key
            runner._compile(dtype, heads, qlen, dim, split_k, early)
            q, k, bt, score, seq = _dummy_scorer_operands(
                dtype, heads, qlen, dim, runner.SUPPORTED_PAGE_SIZE, device
            )
            runner.kernel_cache[key](q, k, bt, score, seq)
        in_tree = _in_tree_scorer()
        in_tree_keys = self._in_tree_scorer_variants(query_lens)
        for key in in_tree_keys:
            dtype, heads, qlen, dim, split_k = key
            in_tree._compile(dtype, heads, qlen, dim, split_k)
            q, k, bt, score, seq = _dummy_scorer_operands(
                dtype, heads, qlen, dim, in_tree.SUPPORTED_PAGE_SIZE, device
            )
            in_tree.kernel_cache[key](q, k, bt, score, seq)
        return keys + in_tree_keys

    # --------------------------------------------------------------------------------------------- the dispatch

    def _select_target(self, attn_metadata: AttentionMetadata):
        """Which target runs this step: one host-side integer, fixed per CUDA-graph capture. A step with context rows
        (a mixed step included) is prefill's; every pure generation step is decode's, whatever its width."""
        return self._decode if attn_metadata.num_contexts == 0 else self._prefill

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        capture: Callable[..., None] = _plain_capture,
        capture_target: Optional[Callable[..., Optional[tuple]]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """``capture``: the Eagle3 hidden-state capture of a layer (the worker's, with its capture norm), and
        ``capture_target``: where a decode boundary all-reduce writes a capture layer's capture norm itself
        (``CAPTURE_NORM_IN_BOUNDARY``; the worker's, None without one)."""
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if kwargs.get("lora_params") is not None:
            raise ValueError("this MiniMax-M3 target serves no LoRA adapters")
        return self._select_target(attn_metadata).forward(
            attn_metadata,
            input_ids,
            position_ids,
            inputs_embeds,
            spec_metadata,
            capture,
            capture_target,
        )


# --------------------------------------------------------------------------------------------------------------------
# The targets
# --------------------------------------------------------------------------------------------------------------------


#: The cores the breakable prefill graphs' eager attention boundary looks up by key: an ``eager_on_graph`` callable
#: replays with weak references to its arguments, so they may be tensors or scalars only.
_BCG_CORES: Dict[str, "MiniMaxM3Tp2ModelingV2"] = {}


def _msa_attention_live(core, i: int, q, k, v, idx_q, md, output) -> None:
    """The MSA backend over a step's live rows (``md.num_tokens``; a padded step's pad rows of ``output`` are left as
    allocated): the KV write unless a producer wrote the caches, the sparse layers' top-k block selection, the FMHA
    over the context rows and any generation rows beside them."""
    live = int(md.num_tokens)
    sparse = i in core.sparse_layers
    nvfp4 = core.main_kv_is_nvfp4
    _record("m3_msa_attention", sparse, k is not None)
    m3_msa_attention(
        core.attn[i],
        q[:live],
        k[:live] if k is not None else None,
        v[:live] if v is not None else None,
        idx_q[:live] if idx_q is not None else None,
        md,
        output[:live],
        sparse,
        core.inv_kv_scales[i] if nvfp4 else None,
        core.kv_scales[i] if nvfp4 else None,
    )


def _prefill_attention_boundary(
    packed, position_ids, q, k, v, idx_q, output, core_key: str, i: int
) -> None:
    """The attention boundary of a prefill layer under a breakable prefill CUDA graph, run eagerly at every replay (as
    the built-in model's MiniMax-M3 attention boundary is): everything that depends on the step's live row count. A
    sparse layer passes its ``packed`` five-way projection: the horizontal producer runs here over the live rows
    (K / V / index-K into the paged caches, Q / index-Q out), then the MSA backend; a dense layer passes q / k / v."""
    core = _BCG_CORES[core_key]
    md = get_model_extra_attrs()["attention_metadata"]()
    live = int(md.num_tokens)
    if packed is not None:
        horizontal = _sparse_qkv_producer(
            core,
            i,
            packed[:live],
            position_ids[..., :live] if position_ids is not None else None,
            md.kv_cache_manager,
            md.msa_idx_k_cache,
            md.msa_out_cache_loc,
        )
        if horizontal is None:
            raise RuntimeError(
                f"MiniMax-M3 sparse layer {i}: no horizontal producer takes this step's caches (an FP8 or NVFP4 HND "
                "pool with an FP8 index-K cache)"
            )
        q, idx_q = horizontal
        k = v = None
    _msa_attention_live(core, i, q, k, v, idx_q, md, output)


_prefill_attention_eager = eager_on_graph(_prefill_attention_boundary)

#: The drafters the breakable prefill graphs' eager draft attention boundary looks up by key (``_BCG_CORES``'s rule).
_BCG_DRAFTS: Dict[str, "M3Eagle3Draft"] = {}


def _draft_attention_boundary(qkv, output, draft_key: str) -> None:
    """The drafter layer's attention under a breakable prefill CUDA graph, run eagerly at every replay (as the in-tree
    drafter's attention boundary is): the backend's RoPE + KV append + causal attention over the step's live rows,
    with the step's metadata read at the replay; ``output``'s pad rows are left as allocated."""
    draft = _BCG_DRAFTS[draft_key]
    md = get_model_extra_attrs()["attention_metadata"]()
    out = m3_draft_attention(draft.attn, qkv, md)
    output[: out.shape[0]].copy_(out)


_draft_attention_eager = eager_on_graph(_draft_attention_boundary)


class PrefillTarget:
    """A step with context rows: prefill chunks, and the mixed steps that carry generation rows beside them. Every
    layer runs in its generic form; the attention core is the MSA backend's for every row.

    Under a breakable (piecewise) prefill CUDA graph the projections, norms, MoE and boundaries are captured per padded
    row bucket and the attention boundary runs eagerly at every replay (``_prefill_attention_eager``: the live-row
    horizontal producer and the MSA backend), as the built-in model's MiniMax-M3 attention does."""

    def __init__(self, core: MiniMaxM3Tp2ModelingV2) -> None:
        self.core = core
        self._bcg_key = f"m3_tp2_core_{id(core)}"
        _BCG_CORES[self._bcg_key] = core
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks for it; read once here rather than per forward.
        self._contract_pending = step_contract_enabled()

    def _check_step_contract(self, md) -> None:
        """Opt-in first-forward fail-fast (TRTLLM_MODELING_V2_VALIDATE): the metadata fields this target consumes
        must exist (they are private trtllm surface). Calling ``step_args`` is the check; once per target instance."""
        if not self._contract_pending:
            return
        try:
            self.step_args(md)
        except AttributeError as exc:
            raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
        self._contract_pending = False

    def step_args(self, md) -> dict:
        """The one projection of the MSA metadata this target's forward reads, computed once per step.

        Counts and per-step host scalars are read here; device tensors are the metadata's persistent buffers, held
        by reference; ``msa_idx_k_cache`` is the metadata's per-layer index-K cache getter. ``metadata`` is the
        prepared object itself: the MSA backend's step (``m3_msa_attention``) takes it as its argument.
        """
        return dict(
            num_contexts=md.num_contexts,
            num_ctx_tokens=md.num_ctx_tokens,
            num_tokens=md.num_tokens,
            kv_cache_manager=md.kv_cache_manager,
            msa_idx_k_cache=md.msa_idx_k_cache,
            msa_out_cache_loc=md.msa_out_cache_loc,
            metadata=md,
        )

    def forward(
        self,
        attn_metadata,
        input_ids,
        position_ids,
        inputs_embeds,
        spec_metadata,
        capture,
        capture_target,
    ):
        """``capture_target`` is the decode boundaries' (unused: no prefill boundary writes a capture norm)."""
        core = self.core
        self._check_step_contract(attn_metadata)
        step = self.step_args(attn_metadata)
        _record("prefill", int(step["num_tokens"]), int(step["num_contexts"]))
        hidden = (
            m3_embedding(input_ids, core.w["embed"]) if inputs_embeds is None else inputs_embeds
        )
        residual = None
        for i in range(core.num_layers):
            hidden, residual = self._layer(
                i, position_ids, hidden, residual, step, spec_metadata, capture
            )
        return hidden

    def _swiglu_mlp(
        self, h: torch.Tensor, prefix: str, low_m_gemm: bool, dense: bool = False
    ) -> torch.Tensor:
        """A gated MXFP8 MLP (the dense MLP, the shared expert): gate_up, SwiGLU-OAI (the dense MLP's on this target's
        C++ op, ``_dense_swiglu``), down, no reduction."""
        core = self.core
        gu, _ = _linear(core, h, f"{prefix}_gu_w", f"{prefix}_gu_s", low_m_gemm)
        if dense:
            act = _dense_swiglu(core, gu)
        else:
            act = m3_silu_and_mul(gu, core.swiglu_limit, core.swiglu_alpha, core.swiglu_beta)
        out, _ = _linear(core, act, f"{prefix}_dn_w", f"{prefix}_dn_s", low_m_gemm)
        return out

    def _routed_chunks(self, i: int, x, router_logits):
        """The routed experts from the bf16 activation over at most ``moe_max_num_tokens`` rows per runner call:
        the MoE scheduler's even split, even chunks on the MoE chunking stream."""
        core = self.core
        rows = x.shape[0]
        num_chunks = (rows + core.moe_max_num_tokens - 1) // core.moe_max_num_tokens
        if num_chunks <= 1:
            return _routed_experts(core, i, x, None, router_logits, True)
        sizes = _split_chunk(rows, num_chunks)
        xs, logits = x.split(sizes), router_logits.split(sizes)
        aux = core.aux_stream_moe_chunk
        ev_main, ev_overlap = core.moe_chunk_events[i]
        if aux is not None:
            ev_main.record()
            with torch.cuda.stream(aux):
                ev_main.wait()
        outs: List[torch.Tensor] = []
        for c in range(num_chunks):
            if aux is not None and c % 2 == 0:
                with torch.cuda.stream(aux):
                    outs.append(_routed_experts(core, i, xs[c], None, logits[c], True))
            else:
                outs.append(_routed_experts(core, i, xs[c], None, logits[c], True))
        if aux is not None:
            with torch.cuda.stream(aux):
                ev_overlap.record()
            ev_overlap.wait()
        return torch.cat(outs)

    def _routed(self, i: int, h: torch.Tensor):
        """Router + routed experts, finalized: on ``ROUTER_TC_GEN``'s rows the TF32 tensor-core router with the
        experts' NVFP4 activation fused in (as the agg canon routes a 9..48-row step with context rows; one runner
        call), else the fp32 router logits and the experts quantizing their bf16 input (chunked)."""
        core = self.core
        fused = _router_tc_fp4(core, i, h)
        if fused is not None:
            logits, x_fp4, x_sf = fused
            return _routed_experts(core, i, x_fp4, x_sf, logits, True)
        return self._routed_chunks(i, h, _router_logits(core, i, h))

    def _moe(self, i: int, h: torch.Tensor, *, return_parts: bool):
        """The MoE block of layer ``i`` without its reduction. Returns ``(out, routed)``: with ``return_parts`` the
        shared and routed parts (routed None when the engine folded the shared expert in), else their sum in
        ``out``."""
        core = self.core
        rows = h.shape[0] if h.dim() == 2 else 0
        if (
            h.dim() == 2
            and h.dtype == torch.bfloat16
            and 1 <= rows <= MOE_ENGINE_MAX_TOKENS
            and rows >= MOE_FOLD_MIN_TOKENS
        ):
            engine = core.engine(i, rows, shared=True)
            if engine is not None:
                _record("m3_moe_engine", rows, False)
                taken(MOE_ENGINE.path, rows)
                folded = m3_moe_engine(h.contiguous(), engine, None, None)
                return folded, None
        if not return_parts and _triton_finalize_rows(core, h):
            routed_parts, shared_out = maybe_execute_in_parallel(
                lambda: _routed_experts(core, i, h, None, _router_logits(core, i, h), False),
                lambda: self._swiglu_mlp(h, f"l{i}_sh", False),
                core.moe_events[i][0],
                core.moe_events[i][1],
                None if flashinfer_mxfp8_autotune_active() else core.aux_stream_moe,
                disable_on_compile=True,
            )
            return _triton_finalize(rows, routed_parts, shared_out), None
        routed_out, shared_out = maybe_execute_in_parallel(
            lambda: self._routed(i, h),
            lambda: self._swiglu_mlp(h, f"l{i}_sh", False),
            core.moe_events[i][0],
            core.moe_events[i][1],
            # The warmup tuning pass profiles the shared expert's GEMMs: keep it on the main stream, between the
            # same collectives on every rank (the capture pass and serving still overlap it on the aux stream).
            None if flashinfer_mxfp8_autotune_active() else core.aux_stream_moe,
            disable_on_compile=True,
        )
        if return_parts:
            return shared_out, routed_out
        # In-place add into the shared output to avoid allocating a temporary.
        return shared_out.add_(routed_out), None

    def _generic_boundary(self, x, x2, residual, norm_w):
        """A generic layer boundary: the multicast all-reduce + residual + RMSNorm on the rows it takes (with the
        shared + routed add folded in), else the built-in fused all-reduce."""
        core = self.core
        if _can_mc(x, folded=x2 is not None):
            fused = _mc_allreduce_norm(core, x, x2, residual, norm_w)
            if fused is not None:
                return fused
        if x2 is not None:
            x = x.add_(x2)
        return _native_allreduce_norm(core, x, residual, norm_w)

    def _layer(self, i: int, position_ids, hidden, residual, step, spec_metadata, capture):
        """Layer ``i`` in its generic form: input norm (first layer without the fused embedding), attention, the
        attention boundary, the MoE (or the dense MLP) and the boundary into the next layer's input norm, every
        collective fused with the residual add and the norm. An Eagle3 capture layer captures the boundary's reduced
        residual sum."""
        core = self.core
        if residual is None:
            residual = hidden
            hidden = flashinfer_rmsnorm(hidden, core.w[f"l{i}_in_norm"], core.rms_eps)
        hidden, _ = self._attention(i, position_ids, hidden, step)
        if _can_mc(hidden):
            fused = _mc_allreduce_norm(core, hidden, None, residual, core.w[f"l{i}_post_norm"])
            hidden, residual = (
                fused
                if fused is not None
                else _native_allreduce_norm(core, hidden, residual, core.w[f"l{i}_post_norm"])
            )
        else:
            hidden, residual = _native_allreduce_norm(
                core, hidden, residual, core.w[f"l{i}_post_norm"]
            )
        if i in core.moe_layers:
            # Row ranges at this dispatch: 9..48 rows take the decode target's generic MC finalize boundary (as the agg
            # canon's generic layer does on every step, context rows included: the MoE unfinalized, its finalize +
            # shared expert in the multicast all-reduce + residual + next norm); the CTX stack starts at 4096 rows.
            folded = core._decode._generic_mc_finalize(i, hidden, residual, _next_norm(core, i))
            if folded is not None:
                if spec_metadata is not None and spec_metadata.is_layer_capture(i):
                    capture(spec_metadata, i, folded[1], None)
                return folded
            out, routed = self._moe(i, hidden, return_parts=_can_mc(hidden, folded=True))
        else:
            out = self._swiglu_mlp(hidden, f"l{i}_mlp", True, dense=True)
            routed = None
        normed, updated = self._generic_boundary(out, routed, residual, _next_norm(core, i))
        if spec_metadata is not None and spec_metadata.is_layer_capture(i):
            capture(spec_metadata, i, updated, None)
        return normed, updated

    def _attention(self, i: int, position_ids, hidden, step):
        """Attention of layer ``i`` up to its o_proj (no reduction). Returns (o_proj output, pushed): a prefill step
        arms no boundary push, so pushed is always False."""
        core = self.core
        if i not in core.sparse_layers:
            # Dense layers 0-2: QKV, the fused norm + RoPE (FP8 q/k/v for an FP8 cache, on this target's op; else in
            # place in bf16), the MSA backend (it writes the layer's cache), o_proj.
            qkv, _ = _linear(core, hidden, f"l{i}_qkv_w", f"l{i}_qkv_s", True)
            fused = (
                _dense_qk_norm_rope_fp8(core, i, qkv, position_ids)
                if core.main_kv_is_fp8
                else _qk_norm_rope(core, i, qkv, position_ids, False)
            )
            q, k, v = fused.split([core.q_size, core.kv_size, core.kv_size], dim=-1)
            o = self._attention_core(i, q, k, v, None, step)
            return _linear(core, o, f"l{i}_o_w", f"l{i}_o_s", True)
        if not core.fused_projection:
            if is_in_breakable_cuda_graph():
                raise RuntimeError(
                    "this MiniMax-M3 target runs a breakable prefill CUDA graph only with the five-way projection "
                    "(fuse_qkv_index_projection)"
                )
            return self._unfused_sparse_attention(i, position_ids, hidden, step)
        packed, _ = _linear(core, hidden, f"l{i}_qkv_w", f"l{i}_qkv_s", True)
        if is_in_breakable_cuda_graph():
            output = packed.new_empty(
                (packed.shape[0], core.heads * core.head_dim), dtype=torch.bfloat16
            )
            _prefill_attention_eager(
                packed, position_ids, None, None, None, None, output, self._bcg_key, i
            )
            return _linear(core, output, f"l{i}_o_w", f"l{i}_o_s", True)
        horizontal = _sparse_qkv_producer(
            core,
            i,
            packed,
            position_ids,
            step["kv_cache_manager"],
            step["msa_idx_k_cache"],
            step["msa_out_cache_loc"],
        )
        if horizontal is None:
            raise RuntimeError(
                f"MiniMax-M3 sparse layer {i}: no horizontal producer takes this step's caches (an FP8 or NVFP4 HND "
                "pool with an FP8 index-K cache)"
            )
        q, idx_q = horizontal
        o = self._attention_core(i, q, None, None, idx_q, step)
        return _linear(core, o, f"l{i}_o_w", f"l{i}_o_s", True)

    def _unfused_sparse_attention(self, i: int, position_ids, hidden, step):
        """A sparse layer without the five-way projection (``fuse_qkv_index_projection=False``): the QKV projection
        and its fused norm + RoPE (FP8 q/k/v into an FP8 cache) on the current stream, the BF16 index projection and
        the FP8 indexer kernel on the attention aux stream, then the MSA backend (it writes the main KV cache) and the
        o_proj. A layer the NVFP4 producer covers is a miss first: the producer needs the five-way projection."""
        core = self.core
        if _producer_covered(core):
            missed(PRODUCER.path, None, "fuse_qkv_index_projection is disabled")

        def _main():
            qkv, _ = _linear(core, hidden, f"l{i}_qkv_w", f"l{i}_qkv_s", True)
            fused = _qk_norm_rope(core, i, qkv, position_ids, core.main_kv_is_fp8)
            return fused.split([core.q_size, core.kv_size, core.kv_size], dim=-1)

        def _index():
            idx_qk = m3_bf16_linear(hidden, core.w[f"l{i}_idx_w"])
            return _fp8_index_qk_norm_rope(
                core, i, idx_qk, position_ids, step["msa_idx_k_cache"], step["msa_out_cache_loc"]
            )

        (q, k, v), idx_q = maybe_execute_in_parallel(
            _main,
            _index,
            core.attn_events[i][0],
            core.attn_events[i][1],
            core.aux_stream_attn,
            disable_on_compile=True,
        )
        o = self._attention_core(i, q, k, v, idx_q, step)
        return _linear(core, o, f"l{i}_o_w", f"l{i}_o_s", True)

    def _attention_core(self, i: int, q, k, v, idx_q, step) -> torch.Tensor:
        """The attention core over the step's live tokens (a padded step's pad rows are left as allocated): the MSA
        backend (``m3_msa_attention``: the KV write unless a producer wrote the caches, the sparse layers' top-k block
        selection, the FMHA over the context rows and any generation rows beside them)."""
        core = self.core
        output = q.new_empty((q.shape[0], core.heads * core.head_dim), dtype=torch.bfloat16)
        if is_in_breakable_cuda_graph():
            # The live row count is a host value of the step: read at every replay inside the eager boundary.
            _prefill_attention_eager(None, None, q, k, v, idx_q, output, self._bcg_key, i)
        else:
            _msa_attention_live(core, i, q, k, v, idx_q, step["metadata"], output)
        return output


class DecodeTarget:
    """A pure generation step (no context rows), of any width.

    Up to ``DECODE_MAX_TOKENS`` tokens with 1-D input ids (up to 8 requests verifying 1 + 3 rows), per layer:
    attention (o_proj all-reduce deferred, its rows pushed into the boundary all-reduce up to 16 rows) -> attention
    boundary (all-reduce + residual + post-attention RMSNorm) -> MoE (the 1..8-row engine pushing its sum into the
    boundary all-reduce; at 9..32 rows the unfinalized MoE with the finalize in the boundary all-reduce) or dense MLP
    (its down projection pushing into the post-MLP boundary) -> MoE / dense boundary (all-reduce + shared/routed add +
    residual + the next layer's RMSNorm, emitting the next QKV's MXFP8 activation). An Eagle3 capture layer runs the
    same decode layer and captures its feed-forward boundary's reduced residual sum (the boundary writing the capture
    norm itself when the worker gives it a target). The first layer runs the generic layer when the fused embedding
    does not take the step. A wider step (or one carrying embeddings) runs every layer in its generic form, with this
    target's attention (the dense layers' full-context decode, the decode selection and sparse decode where the
    producer wrote the caches)."""

    def __init__(self, core: MiniMaxM3Tp2ModelingV2) -> None:
        self.core = core
        self._moe_push_descs: Dict[tuple, Any] = {}
        # Set by a feed-forward boundary that wrote the layer's Eagle3 capture (see _layer's ``capture``).
        self._capture_written = False
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks for it; read once here rather than per forward.
        self._contract_pending = step_contract_enabled()

    def _check_step_contract(self, md) -> None:
        """Opt-in first-forward fail-fast (TRTLLM_MODELING_V2_VALIDATE): the metadata fields this target consumes
        must exist (they are private trtllm surface). Calling ``step_args`` is the check; once per target instance."""
        if not self._contract_pending:
            return
        try:
            self.step_args(md)
        except AttributeError as exc:
            raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
        self._contract_pending = False

    def step_args(self, md) -> dict:
        """The one projection of the MSA metadata this target's forward reads, computed once per step.

        States the two context counts instead of reading them back: this target runs only when ``num_contexts`` is
        zero, so reading them would ask the metadata to confirm what the routing already decided.

        Graph classes: counts, the generation span, the identity-selection flag, the worst-case score extent and the
        cache length bound are host values fixed per capture (prepare() stages them before the forward); the block
        table, lengths, cache slots, identity table and valid-block counts are the metadata's persistent buffers, held
        by reference and refreshed in place every step; ``msa_idx_k_cache`` / ``msa_proxy_max_score_view`` are the
        metadata's getters of the per-layer index-K cache and of the score scratch view. ``metadata`` is the prepared
        object itself: the MSA backend's step (``m3_msa_attention``) and the full-context decode's K/V write
        (``write_msa_phase_kv``) take it as their argument.
        """
        return dict(
            num_contexts=0,
            num_ctx_tokens=0,
            num_tokens=md.num_tokens,
            num_seqs=md.num_seqs,
            max_seq_len=md.max_seq_len,
            kv_cache_manager=md.kv_cache_manager,
            msa_idx_k_cache=md.msa_idx_k_cache,
            msa_out_cache_loc=md.msa_out_cache_loc,
            msa_decode_span=md.msa_decode_span,
            msa_block_table=md.msa_block_table,
            msa_seq_lens_cuda=md.msa_seq_lens_cuda,
            msa_short_ok=md._msa_short_ok,
            msa_short_table=md.msa_short_table,
            msa_proxy_max_score_view=md.msa_proxy_max_score_view,
            msa_worst_case_max_k_tiles=md.msa_worst_case_max_k_tiles,
            msa_n_valid_blocks=md.msa_n_valid_blocks,
            metadata=md,
        )

    def forward(
        self,
        attn_metadata,
        input_ids,
        position_ids,
        inputs_embeds,
        spec_metadata,
        capture,
        capture_target,
    ):
        core = self.core
        self._check_step_contract(attn_metadata)
        step = self.step_args(attn_metadata)
        # The decode kernels' rows (a per-capture constant): up to DECODE_MAX_TOKENS tokens of 1-D input ids.
        kernels = (
            inputs_embeds is None and input_ids.ndim == 1 and is_decode_step(input_ids.numel())
        )
        if kernels:
            num_tokens = int(input_ids.numel())
            plan = plan_for(num_tokens)
            _record("decode", num_tokens, plan.name)
            hidden, residual = self._embed(input_ids, num_tokens)
        else:
            # Breakable (piecewise) CUDA graphs capture steps with context rows only (prefill_cuda_graph_backend):
            # every pure generation step, the 33..192-row ones on this generic branch included, runs eagerly or on
            # the ordinary decode CUDA graphs, so this guard never fires in a supported deployment.
            if is_in_breakable_cuda_graph():
                raise RuntimeError(
                    "this MiniMax-M3 target does not run under piecewise (breakable) CUDA graphs"
                )
            plan = None
            _record("decode", int(step["num_tokens"]), "generic")
            hidden = (
                m3_embedding(input_ids, core.w["embed"]) if inputs_embeds is None else inputs_embeds
            )
            residual = None
        for i in range(core.num_layers):
            captured = spec_metadata is not None and spec_metadata.is_layer_capture(i)
            if kernels and residual is not None and captured:
                # The capture layer on the decode layer (its boundaries push); the capture reads the MoE / dense
                # boundary's reduced residual sum, as the generic layer's fused boundary capture does. Given a target
                # (``CAPTURE_NORM_IN_BOUNDARY``), the boundary all-reduce writes the capture norm itself.
                target = (
                    capture_target(spec_metadata, i, num_tokens)
                    if capture_target is not None
                    else None
                )
                self._capture_written = False
                hidden, residual = self._layer(
                    i, plan, position_ids, hidden, residual, step, capture=target
                )
                taken("capture_layer_fast", num_tokens, f"layer {i}")
                if target is not None and self._capture_written:
                    taken("capture_norm_in_boundary", num_tokens, f"layer {i}")
                    capture(spec_metadata, i, residual, None, written=True)
                else:
                    capture(spec_metadata, i, residual, None)
            elif not kernels or residual is None or captured:
                hidden, residual = self._generic_layer(
                    i, position_ids, hidden, residual, step, spec_metadata, capture
                )
            else:
                hidden, residual = self._layer(i, plan, position_ids, hidden, residual, step)
        return hidden

    def _embed(self, input_ids: torch.Tensor, num_tokens: int):
        """(hidden, residual) of the fused embedding + first RMSNorm; (embedding, None) when it does not run, so the
        first layer runs its own input norm. SM107: 1.55..1.65 us against 3.4..6.0 us for the stock embedding +
        RMSNorm at 1..8 rows (the gather bitwise, the norm one ulp off on ~5e-6 of the elements: summation order)."""
        core = self.core
        if not EMBED_FIRSTNORM.covers(num_tokens):
            uncovered(EMBED_FIRSTNORM.path, f"more than {EMBED_FIRSTNORM_MAX_TOKENS} tokens")
            return m3_embedding(input_ids, core.w["embed"]), None
        owner = core.embed_owners.get((num_tokens, input_ids.dtype))
        if owner is None or not input_ids.is_contiguous():
            missed(
                EMBED_FIRSTNORM.path,
                num_tokens,
                f"no embedding owner for {input_ids.dtype} ids (built for int32 / int64 after the load)"
                if owner is None
                else "the input ids are not contiguous",
            )
            return m3_embedding(input_ids, core.w["embed"]), None
        _record("m3_embedding_firstnorm", num_tokens, input_ids.dtype)
        taken(EMBED_FIRSTNORM.path, num_tokens, str(input_ids.dtype))
        return m3_embedding_firstnorm(
            input_ids, owner.table, owner.weight, owner.eps, owner.raw, owner.output, owner.use_pdl
        )

    def _bucket_flags(self, num_tokens: int):
        """(bucket workspace, its 8-CTA call counters) of the boundary all-reduce for ``num_tokens``, or None."""
        workspace = m3_mc_allreduce.get_workspace(self.core.mapping, self.core.hidden)
        workspace = None if workspace is None else m3_mc_allreduce._bucket(workspace, num_tokens)
        flags = None if workspace is None else workspace.flags_for(8)
        return None if flags is None else (workspace, flags)

    @staticmethod
    def _no_bucket(path: str, num_tokens: int) -> None:
        if torch.cuda.is_current_stream_capturing():
            capture_missed(path, num_tokens, "the multicast all-reduce workspace")
        else:
            missed(path, num_tokens, "no multicast all-reduce bucket holds the step")

    def _attn_push_target(self, num_tokens: int):
        """The push tuple into the attention boundary's all-reduce (the o_proj's; also the dense down projection's into
        the post-MLP boundary, the same exchange), or None (after a miss).

        None comes only from the token count or from workspace state created collectively in the same eager call on
        every rank, so all ranks agree. (A lone non-pushing rank would still be safe: its all-reduce pushes the same
        bf16 rows into the same set A.)"""
        found = self._bucket_flags(num_tokens)
        if found is None:
            self._no_bucket(ATTN_PUSH.path, num_tokens)
            return None
        workspace, flags = found
        return (
            workspace.mc.data_ptr(),
            flags.data_ptr(),
            workspace.rank,
            workspace.world_size,
            workspace.max_tokens,
            AR_PUSH_MODE,
        )

    def _moe_push_targets(self, num_tokens: int):
        """(engine push descriptor, shared down-projection push tuple) for the MoE boundary's exchange sets, or None
        (after a miss).

        This decision must match across ranks: prepushed 3 reads set B of every rank, which a non-pushing rank never
        fills. None comes only from the token count or from state created in the same eager call on every rank
        (workspace, flags, the descriptor cache), and the engine it is armed for is agreed at build time
        (``_prebuild_engines``), so it does."""
        hidden = self.core.hidden
        found = self._bucket_flags(num_tokens)
        if found is None:
            self._no_bucket(MOE_PUSH.path, num_tokens)
            return None
        workspace, flags = found
        key = (workspace.mc.data_ptr(), flags.data_ptr())
        desc = self._moe_push_descs.get(key)
        if desc is None:
            if torch.cuda.is_current_stream_capturing():
                capture_missed(MOE_PUSH.path, num_tokens, "the MoE engine's push descriptor")
                return None
            desc = self._moe_push_descs[key] = ar_push_desc(
                workspace.mc.data_ptr(),
                flags.data_ptr(),
                workspace.rank,
                workspace.world_size,
                workspace.max_tokens,
                flags.device,
            )
        set_b = workspace.mc.data_ptr() + 2 * 4 * workspace.max_tokens * workspace.world_size * (
            hidden // 2
        )
        return desc, (
            set_b,
            flags.data_ptr(),
            workspace.rank,
            workspace.world_size,
            workspace.max_tokens,
            2,
        )

    def _boundary(self, use_mc: bool, x, residual, norm_w, x2=None, quant=False, capture=None):
        """All-reduce + (second addend) + residual + RMSNorm: the multicast kernel, else the built-in all-reduce.
        ``capture``: the Eagle3 capture the multicast kernel also writes (``_capture_written`` records it)."""
        core = self.core
        if use_mc:
            fused = _mc_allreduce_norm(core, x, x2, residual, norm_w, quant=quant, capture=capture)
            if fused is not None:
                if capture is not None:
                    self._capture_written = True
                taken(
                    MC_ALLREDUCE.path,
                    x.shape[0],
                    ("x + x2" if x2 is not None else "x") + (", MXFP8 copy" if quant else ""),
                )
                return fused
            workspace = m3_mc_allreduce.get_workspace(core.mapping, x.shape[-1])
            missed(
                MC_ALLREDUCE.path,
                x.shape[0],
                m3_mc_allreduce.decline_reason(x, residual, workspace, x2=x2),
            )
        if x2 is not None:
            x = x.add_(x2)
        return _native_allreduce_norm(core, x, residual, norm_w)

    def _prepushed(
        self, i: int, x, residual, norm_w, quant, prepushed, source, x2=None, capture=None
    ):
        fused = _mc_allreduce_norm(
            self.core, x, x2, residual, norm_w, quant=quant, prepushed=prepushed, capture=capture
        )
        if fused is None:
            raise RuntimeError(f"layer {i}: rows were pushed into an all-reduce that did not run")
        if capture is not None:
            self._capture_written = True
        taken(MC_ALLREDUCE.path, x.shape[0], f"prepushed {prepushed}, {source}")
        return fused

    def _layer(self, i: int, plan: DecodePlan, position_ids, hidden, residual, step, capture=None):
        """Layer ``i`` on the decode kernels. ``capture``: (out, fc_norm weight, eps) for the feed-forward boundary to
        write the layer's Eagle3 capture norm (``CAPTURE_NORM_IN_BOUNDARY``); a boundary that writes it sets
        ``_capture_written``."""
        core = self.core
        num_tokens = hidden.shape[0]
        # Padded rows: the boundary all-reduce polls and re-arms rows t < M = residual.shape[0], CUDA-graph padding
        # rows included. Every producer pushes all rows of the same tensor: the o_proj / shared down / dense down GEMV
        # t < m = x.shape[0] (its row count comes from the host shape, not a live count), the MoE engine every row of
        # the num_tokens it was built for (= num_tokens here). The all-reduce checks x.sizes() == residual.sizes(), so
        # no padded row is left with a stale flag.
        # The GEMV dispatch (and with it every push into a boundary all-reduce) is a decode-graph one.
        decode_graph = flashinfer_mxfp8_decode_graph_capture_active()
        if not decode_graph:
            uncovered(
                "decode_graph",
                "an eager decode step: the decode-graph GEMV dispatch and the boundary pushes are off",
            )
        push = plan.attn_mc_ar and decode_graph and ATTN_PUSH.covers(num_tokens)
        armed = self._attn_push_target(num_tokens) if push else None
        hidden, attn_pushed = self._attention(i, position_ids, hidden, step, armed)
        post_norm = core.w[f"l{i}_post_norm"]
        if armed is not None and attn_pushed:
            hidden, residual = self._prepushed(
                i, hidden, residual, post_norm, False, AR_PUSH_MODE, "o_proj push"
            )
        else:
            if armed is not None:
                missed(
                    ATTN_PUSH.path,
                    num_tokens,
                    "the o_proj did not push its rows into the armed boundary all-reduce",
                )
            hidden, residual = self._boundary(plan.attn_mc_ar, hidden, residual, post_norm)

        norm_w = _next_norm(core, i)
        if i not in core.moe_layers:
            # A dense layer: its MLP's down projection pushes into the post-MLP boundary's all-reduce, as the o_proj
            # does into the attention one (the same exchange; the gate_up projection never pushes). SM107 TP2: the
            # two non-capture dense layers' post-MLP hops end 5.5-5.8 us after the down GEMV unpushed vs 2.4-2.5 us
            # pushed; bitwise equal to the unpushed boundary (the same bf16 words).
            dense_push = plan.attn_mc_ar and decode_graph and DENSE_PUSH.covers(num_tokens)
            armed = self._attn_push_target(num_tokens) if dense_push else None
            out, pushed = self._swiglu_mlp(hidden, f"l{i}_mlp", True, armed, dense=True)
            if armed is not None and pushed:
                return self._prepushed(
                    i,
                    out,
                    residual,
                    norm_w,
                    False,
                    AR_PUSH_MODE,
                    "dense down push",
                    capture=capture,
                )
            if armed is not None:
                missed(
                    DENSE_PUSH.path,
                    num_tokens,
                    "the dense MLP's down projection did not push its rows into the armed boundary all-reduce",
                )
            return self._boundary(plan.attn_mc_ar, out, residual, norm_w, capture=capture)

        quant = plan.qkv_prequant and i + 1 < core.num_layers
        moe_push = None
        if moe_path(num_tokens) == MOE_ENGINE.path:
            # The engine runs inside the MoE; looked up (or found missing) here first so a push is armed only for an
            # engine that will take it.
            engine = core.engine(i, num_tokens, shared=True)
            if (
                engine is not None
                and plan.moe_mc_ar
                and decode_graph
                and MOE_PUSH.covers(num_tokens)
            ):
                moe_push = self._moe_push_targets(num_tokens)
        elif moe_path(num_tokens) == MC_FINALIZE.path and plan.moe_mc_ar:
            if self._bucket_flags(num_tokens) is None:
                self._no_bucket(MC_FINALIZE.path, num_tokens)
            else:
                return self._mc_finalize(i, hidden, residual, norm_w, quant, capture)
        shared, routed, engine_pushed, shared_pushed = self._moe(
            i, hidden, return_parts=True, push=moe_push
        )
        if moe_push is not None:
            if routed is None:
                # The engine folded the shared expert in and pushed routed + shared into set A.
                if engine_pushed:
                    return self._prepushed(
                        i, shared, residual, norm_w, quant, 2, "MoE engine push", capture=capture
                    )
            else:
                if engine_pushed and shared_pushed:
                    return self._prepushed(
                        i,
                        shared,
                        residual,
                        norm_w,
                        quant,
                        3,
                        "MoE engine + shared push",
                        x2=routed,
                        capture=capture,
                    )
                if engine_pushed or shared_pushed:
                    raise RuntimeError("only one MoE part pushed into the boundary all-reduce")
            missed(
                MOE_PUSH.path,
                num_tokens,
                "the MoE did not push its output into the armed boundary all-reduce",
            )
        return self._boundary(
            plan.moe_mc_ar, shared, residual, norm_w, x2=routed, quant=quant, capture=capture
        )

    def _mc_finalize(self, i: int, h, residual, norm_w, quant: bool, capture=None, what: str = ""):
        """The MoE of layer ``i`` unfinalized (its shared expert beside it) and the finalize + all-reduce + residual
        + next RMSNorm in one launch (``m3_mc_finalize_allreduce_norm``): returns (normed, updated residual). The
        caller checked the rows and the multicast bucket (decided from shapes alone, so the same on every TP rank)."""
        core = self.core
        rows = h.shape[0]
        routed, (shared, _) = maybe_execute_in_parallel(
            lambda: _decode_routed_unfinalized(core, i, h),
            lambda: self._swiglu_mlp(h, f"l{i}_sh", False),
            core.moe_events[i][0],
            core.moe_events[i][1],
            None if flashinfer_mxfp8_autotune_active() else core.aux_stream_moe,
            disable_on_compile=True,
        )
        fc2, scales, perm = routed
        _record("m3_mc_finalize_allreduce_norm", rows, quant, capture is not None)
        fused = m3_mc_allreduce.m3_mc_finalize_allreduce_norm(
            fc2,
            perm.view(rows, -1).to(torch.int32),
            scales.reshape(rows, -1),
            shared,
            residual,
            norm_w,
            core.rms_eps,
            m3_mc_allreduce.get_workspace(core.mapping, core.hidden),
            use_pdl=get_env_enable_pdl(),
            quant=quant,
            capture=capture,
        )
        if fused is None:
            raise RuntimeError(
                f"layer {i}: the MC finalize all-reduce rejected the unfinalized MoE outputs"
            )
        if capture is not None:
            self._capture_written = True
        taken(MC_FINALIZE.path, rows, what or ("MXFP8 copy" if quant else ""))
        return fused

    def _generic_mc_finalize(self, i: int, h, residual, norm_w):
        """Past ``DECODE_MAX_TOKENS`` rows, up to ``GENERIC_MC_FINALIZE_MAX_TOKENS``: the generic layer's MoE boundary
        as ``_mc_finalize``; None (from shapes alone, the same on every rank) keeps the generic boundary."""
        rows = h.shape[0] if h.dim() == 2 else 0
        cap = min(GENERIC_MC_FINALIZE_MAX_TOKENS, m3_mc_allreduce.FINALIZE_MAX_TOKENS)
        if (
            not (DECODE_MAX_TOKENS < rows <= cap)
            or h.dtype != torch.bfloat16
            or torch.compiler.is_compiling()
        ):
            return None
        if self._bucket_flags(rows) is None:
            return None
        return self._mc_finalize(i, h, residual, norm_w, False, what="generic")

    def _swiglu_mlp(
        self,
        h: torch.Tensor,
        prefix: str,
        low_m_gemm: bool,
        push: Optional[tuple] = None,
        dense: bool = False,
    ):
        """A gated MXFP8 MLP (the dense MLP, the shared expert): gate_up, SwiGLU-OAI (the dense MLP's on this target's
        C++ op, ``_dense_swiglu``), down, no reduction. Returns (output, whether the down projection's GEMV pushed into
        ``push``)."""
        core = self.core
        gu, _ = _linear(core, h, f"{prefix}_gu_w", f"{prefix}_gu_s", low_m_gemm)
        if dense:
            act = _dense_swiglu(core, gu)
        else:
            act = m3_silu_and_mul(gu, core.swiglu_limit, core.swiglu_alpha, core.swiglu_beta)
        return _linear(core, act, f"{prefix}_dn_w", f"{prefix}_dn_s", low_m_gemm, push)

    def _routed_chunks(self, i: int, x, router_logits):
        """The routed experts from the bf16 activation over at most ``moe_max_num_tokens`` rows per runner call: the MoE
        scheduler's even split, even chunks on the MoE chunking stream."""
        core = self.core
        rows = x.shape[0]
        num_chunks = (rows + core.moe_max_num_tokens - 1) // core.moe_max_num_tokens
        if num_chunks <= 1:
            return _routed_experts(core, i, x, None, router_logits, True)
        sizes = _split_chunk(rows, num_chunks)
        xs, logits = x.split(sizes), router_logits.split(sizes)
        aux = core.aux_stream_moe_chunk
        ev_main, ev_overlap = core.moe_chunk_events[i]
        if aux is not None:
            ev_main.record()
            with torch.cuda.stream(aux):
                ev_main.wait()
        outs: List[torch.Tensor] = []
        for c in range(num_chunks):
            if aux is not None and c % 2 == 0:
                with torch.cuda.stream(aux):
                    outs.append(_routed_experts(core, i, xs[c], None, logits[c], True))
            else:
                outs.append(_routed_experts(core, i, xs[c], None, logits[c], True))
        if aux is not None:
            with torch.cuda.stream(aux):
                ev_overlap.record()
            ev_overlap.wait()
        return torch.cat(outs)

    def _routed(self, i: int, h: torch.Tensor):
        """Router + routed experts, finalized: on ``ROUTER_TC_GEN``'s rows the TF32 tensor-core router with the
        experts' NVFP4 activation fused in (one runner call: the rows are within ``moe_max_num_tokens``), else the
        fp32 router and the experts quantizing their bf16 input (chunked)."""
        core = self.core
        fused = _router_tc_fp4(core, i, h)
        if fused is not None:
            logits, x_fp4, x_sf = fused
            return _routed_experts(core, i, x_fp4, x_sf, logits, True)
        return self._routed_chunks(i, h, _decode_router_logits(core, i, h))

    def _moe(self, i: int, h: torch.Tensor, *, return_parts: bool, push=None):
        """The MoE block of layer ``i`` without its reduction.

        ``push`` = (the engine's ``arp`` descriptor, the shared down projection's ``ar`` tuple) the decode layer armed
        for the MoE boundary's exchange sets, or None. Returns ``(out, routed, engine_pushed, shared_pushed)``: with
        ``return_parts`` the shared and routed parts (routed None when the engine folded the shared expert in), else
        their sum in ``out``.
        """
        core = self.core
        engine_arp, shared_ar = push if push is not None else (None, None)
        rows = h.shape[0] if h.dim() == 2 else 0
        if (
            h.dim() == 2
            and h.dtype == torch.bfloat16
            and 1 <= rows <= MOE_ENGINE_MAX_TOKENS
            and rows >= MOE_FOLD_MIN_TOKENS
        ):
            engine = core.engine(i, rows, shared=True)
            if engine is not None:
                _record("m3_moe_engine", rows, engine_arp is not None)
                taken(MOE_ENGINE.path, rows)
                folded = m3_moe_engine(h.contiguous(), engine, None, engine_arp)
                return folded, None, engine_arp is not None, False
        if not return_parts and push is None and _triton_finalize_rows(core, h):
            routed_parts, (shared_out, _) = maybe_execute_in_parallel(
                lambda: _decode_routed_unfinalized(core, i, h),
                lambda: self._swiglu_mlp(h, f"l{i}_sh", False),
                core.moe_events[i][0],
                core.moe_events[i][1],
                None if flashinfer_mxfp8_autotune_active() else core.aux_stream_moe,
                disable_on_compile=True,
            )
            return _triton_finalize(rows, routed_parts, shared_out), None, False, False
        routed_out, (shared_out, shared_pushed) = maybe_execute_in_parallel(
            lambda: self._routed(i, h),
            lambda: self._swiglu_mlp(h, f"l{i}_sh", False, shared_ar),
            core.moe_events[i][0],
            core.moe_events[i][1],
            # The warmup tuning pass profiles the shared expert's GEMMs: keep it on the main stream, between the same
            # collectives on every rank (the capture pass and serving still overlap it on the aux stream).
            None if flashinfer_mxfp8_autotune_active() else core.aux_stream_moe,
            disable_on_compile=True,
        )
        if return_parts:
            return shared_out, routed_out, False, shared_pushed
        # In-place add into the shared output to avoid allocating a temporary.
        return shared_out.add_(routed_out), None, False, shared_pushed

    def _generic_boundary(self, x, x2, residual, norm_w):
        """A generic layer boundary: the multicast all-reduce + residual + RMSNorm on the rows it takes (with the
        shared + routed add folded in), else the built-in fused all-reduce."""
        core = self.core
        if _can_mc(x, folded=x2 is not None):
            fused = _mc_allreduce_norm(core, x, x2, residual, norm_w)
            if fused is not None:
                if x.shape[0] > DECODE_MAX_TOKENS:
                    taken(
                        MC_ALLREDUCE.path,
                        x.shape[0],
                        "generic " + ("x + x2" if x2 is not None else "x"),
                    )
                return fused
        if x2 is not None:
            x = x.add_(x2)
        return _native_allreduce_norm(core, x, residual, norm_w)

    def _generic_layer(self, i: int, position_ids, hidden, residual, step, spec_metadata, capture):
        """Layer ``i`` in its generic form: input norm (first layer without the fused embedding), attention, the
        attention boundary, the MoE (or the dense MLP) and the boundary into the next layer's input norm, every
        collective fused with the residual add and the norm. An Eagle3 capture layer captures the boundary's reduced
        residual sum."""
        core = self.core
        if residual is None:
            residual = hidden
            hidden = flashinfer_rmsnorm(hidden, core.w[f"l{i}_in_norm"], core.rms_eps)
        hidden, _ = self._attention(i, position_ids, hidden, step)
        hidden, residual = self._generic_boundary(hidden, None, residual, core.w[f"l{i}_post_norm"])
        if i in core.moe_layers:
            folded = self._generic_mc_finalize(i, hidden, residual, _next_norm(core, i))
            if folded is not None:
                if spec_metadata is not None and spec_metadata.is_layer_capture(i):
                    capture(spec_metadata, i, folded[1], None)
                return folded
            out, routed, _, _ = self._moe(i, hidden, return_parts=_can_mc(hidden, folded=True))
        else:
            out, _ = self._swiglu_mlp(hidden, f"l{i}_mlp", True, dense=True)
            routed = None
        normed, updated = self._generic_boundary(out, routed, residual, _next_norm(core, i))
        if spec_metadata is not None and spec_metadata.is_layer_capture(i):
            capture(spec_metadata, i, updated, None)
        return normed, updated

    def _attention(self, i: int, position_ids, hidden, step, push: Optional[tuple] = None):
        """Attention of layer ``i`` up to its o_proj (no reduction). ``push`` is the attention boundary's exchange the
        o_proj may write into. Returns (o_proj output, pushed)."""
        core = self.core
        if i not in core.sparse_layers:
            return self._dense_attention(i, position_ids, hidden, step, push)
        if not core.fused_projection:
            return self._unfused_sparse_attention(i, position_ids, hidden, step, push)
        packed, _ = _linear(core, hidden, f"l{i}_qkv_w", f"l{i}_qkv_s", True)
        horizontal = _sparse_qkv_producer(
            core,
            i,
            packed,
            position_ids,
            step["kv_cache_manager"],
            step["msa_idx_k_cache"],
            step["msa_out_cache_loc"],
        )
        if horizontal is None:
            raise RuntimeError(
                f"MiniMax-M3 sparse layer {i}: no horizontal producer takes this step's caches (an FP8 or NVFP4 HND "
                "pool with an FP8 index-K cache)"
            )
        q, idx_q = horizontal
        o = self._attention_core(i, q, None, None, idx_q, step)
        return _linear(core, o, f"l{i}_o_w", f"l{i}_o_s", True, push)

    def _dense_attention(self, i: int, position_ids, hidden, step, push):
        """Dense layers 0-2: QKV, the fused norm + RoPE (FP8 q/k/v: an FP8 cache, or the FP8 pool the dense layers sit
        in within a hybrid NVFP4 cache on a decode step), the attention core, o_proj.

        On a covered pure generation step (``_fullctx_plan``) the attention is this target's PrimTS split-KV decode
        over the layer's full causal context, and the norm/RoPE op also stores the step's new K/V rows into the
        layer's paged pool and the query compact, as the decode reads it (``DENSE_KV_INSERT_FUSED``; the bytes
        ``write_msa_phase_kv`` stores). Otherwise the MSA backend."""
        core = self.core
        qkv, _ = _linear(core, hidden, f"l{i}_qkv_w", f"l{i}_qkv_s", True)
        fp8 = core.main_kv_is_fp8 or core.main_kv_is_nvfp4
        plan = self._fullctx_plan(i, qkv.device, step)
        kv_dest = None
        if (
            plan is not None
            and fp8
            and DENSE_KV_INSERT_FUSED
            and _dense_fp8_op_applies(qkv, position_ids)
        ):
            kv_dest = (
                step["kv_cache_manager"].get_buffers(i, kv_layout="HND"),
                step["msa_out_cache_loc"],
                plan[1],
                torch.empty(
                    (qkv.shape[0], core.heads, core.head_dim),
                    dtype=torch.float8_e4m3fn,
                    device=qkv.device,
                ),
            )
        if fp8:
            fused = _dense_qk_norm_rope_fp8(core, i, qkv, position_ids, kv_dest)
            if kv_dest is not None:
                taken(DENSE_KV_INSERT_PATH, kv_dest[2])
        else:
            fused = _qk_norm_rope(core, i, qkv, position_ids, False)
        q, k, v = fused.split([core.q_size, core.kv_size, core.kv_size], dim=-1)
        if kv_dest is not None:
            q = kv_dest[3].view(q.shape[0], -1)
        o = None
        if plan is not None:
            o = self._fullctx_decode(i, q, k, v, step, plan, kv_written=kv_dest is not None)
        if o is None:
            o = self._attention_core(i, q, k, v, None, step)
        return _linear(core, o, f"l{i}_o_w", f"l{i}_o_s", True, push)

    def _fullctx_plan(self, i: int, device: torch.device, step):
        """(the PrimTS split-KV decode plan, the step's query tokens) where dense layer ``i``'s decode attention runs
        on ``ops/m3_fullctx_decode`` (``FULLCTX_DECODE``): a covered (query rows per request, requests) shape over
        the layer's FP8 pool. None otherwise (the MSA backend runs).

        Rubin TP2, FP8 KV, 165k context, graph-us per call (stock trtllm-gen in brackets): (4, 1) 18.8 (29.3), (1, 1)
        15.7 (28.0), (4, 2) 24.5 (29.3), (1, 2) 20.2 (28.1); the stock split is capped at 16 cluster CTAs per (request,
        KV head), i.e. 32 of 212 CTAs at one request. With the BF16 split-KV merge (``FULLCTX_DECODE_OWN_REDUCER``):
        (1, 1) 14.5 vs 18.5, (4, 1) 17.3 vs 21.3, (1, 2) 18.5 vs 23.1, (4, 2) 23.3 vs 27.5."""
        path = FULLCTX_DECODE.path
        span = step["msa_decode_span"]
        if span is None or span.row_first != 0:
            uncovered(path, "no decode span starting at row 0")
            return None
        nreq = int(step["num_seqs"])
        qlen = int(span.query_len)
        if not FULLCTX_DECODE.covers((qlen, nreq)):
            uncovered(path, f"{qlen} query rows x {nreq} requests is not a covered shape")
            return None
        if int(step["max_seq_len"]) > FULLCTX_DECODE_MAX_KV_LEN:
            uncovered(path, "the cache length bound is beyond the plans' K/V capacity")
            return None
        num_tokens = nreq * qlen
        pool, _ = step["kv_cache_manager"].get_kv_subpage_pool(i, "HND")
        if pool.dtype != torch.float8_e4m3fn:
            missed(path, num_tokens, "the dense layer's KV pool is not FP8")
            return None
        from .ops import m3_fullctx_decode as _fullctx_decode_kernel

        own_reducer = FULLCTX_DECODE_OWN_REDUCER
        if own_reducer and not _fullctx_decode_kernel.has_own_reducer():
            missed(path, num_tokens, "the BF16 split-KV merge op is not built")
            return None
        # Compiled before warmup (``prebuild_kernels``); the forward never compiles one.
        plan = _fullctx_decode_kernel.plan_for(device, qlen, nreq, own_reducer)
        if plan is None:
            missed(path, num_tokens, "the PrimTS decode plan is not prebuilt")
            return None
        return plan, num_tokens

    def _fullctx_decode(self, i: int, q, k, v, step, plan_tokens, *, kv_written: bool):
        """Dense layer ``i``'s decode attention on the PrimTS plan of ``_fullctx_plan``: the step's new K/V go into the
        paged cache as the MSA backend writes them (``write_msa_phase_kv``; already there when ``kv_written``), then
        every query row attends its request's full causal context."""
        from tensorrt_llm._torch.attention.backends.interface import AttentionInputType
        from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
            write_msa_phase_kv,
        )

        from .ops import m3_fullctx_decode as _fullctx_decode_kernel

        core = self.core
        plan, num_tokens = plan_tokens
        nreq = int(step["num_seqs"])
        if not kv_written:
            write_msa_phase_kv(
                core.attn[i],
                k[:num_tokens],
                v[:num_tokens],
                step["metadata"],
                AttentionInputType.mixed,
                token_offset=0,
            )
        pool, subpages = step["kv_cache_manager"].get_kv_subpage_pool(i, "HND")
        output = q.new_empty((q.shape[0], core.heads * core.head_dim), dtype=torch.bfloat16)
        k_pages, v_pages = _fullctx_decode_kernel.kv_page_views(pool, subpages)
        _record("m3_fullctx_decode", num_tokens, nreq, kv_written)
        plan.run(
            fp8_e4m3_saturate(q[:num_tokens].reshape(num_tokens, core.heads, core.head_dim)),
            k_pages,
            v_pages,
            step["msa_block_table"][:nreq],
            step["msa_seq_lens_cuda"][:nreq],
            core.sm_scale(i),
            output[:num_tokens],
        )
        taken(FULLCTX_DECODE.path, num_tokens)
        return output

    def _unfused_sparse_attention(self, i: int, position_ids, hidden, step, push):
        """A sparse layer without the five-way projection (``fuse_qkv_index_projection=False``): the QKV projection and
        its fused norm + RoPE (FP8 q/k/v into an FP8 cache) on the current stream, the BF16 index projection and the FP8
        indexer kernel on the attention aux stream, then the MSA backend (it writes the main KV cache) and the o_proj. A
        layer the NVFP4 producer covers is a miss first: the producer needs the five-way projection."""
        core = self.core
        if _producer_covered(core):
            missed(PRODUCER.path, None, "fuse_qkv_index_projection is disabled")

        def _main():
            qkv, _ = _linear(core, hidden, f"l{i}_qkv_w", f"l{i}_qkv_s", True)
            fused = _qk_norm_rope(core, i, qkv, position_ids, core.main_kv_is_fp8)
            return fused.split([core.q_size, core.kv_size, core.kv_size], dim=-1)

        def _index():
            idx_qk = m3_bf16_linear(hidden, core.w[f"l{i}_idx_w"])
            return _fp8_index_qk_norm_rope(
                core, i, idx_qk, position_ids, step["msa_idx_k_cache"], step["msa_out_cache_loc"]
            )

        (q, k, v), idx_q = maybe_execute_in_parallel(
            _main,
            _index,
            core.attn_events[i][0],
            core.attn_events[i][1],
            core.aux_stream_attn,
            disable_on_compile=True,
        )
        o = self._attention_core(i, q, k, v, idx_q, step)
        return _linear(core, o, f"l{i}_o_w", f"l{i}_o_s", True, push)

    def _attention_core(self, i: int, q, k, v, idx_q, step) -> torch.Tensor:
        """The attention core over the step's live tokens (a padded step's pad rows are left as allocated): a sparse
        layer whose caches the producer wrote runs this target's selection and sparse decode when the generation span
        starts at row 0; everything else the MSA backend (``m3_msa_attention``: the KV write unless a producer wrote the
        caches, the sparse layers' top-k block selection, the FMHA)."""
        core = self.core
        output = q.new_empty((q.shape[0], core.heads * core.head_dim), dtype=torch.bfloat16)
        live = int(step["num_tokens"])
        q_live, out_live = q[:live], output[:live]
        k_live = k[:live] if k is not None else None
        v_live = v[:live] if v is not None else None
        idx_q_live = idx_q[:live] if idx_q is not None else None
        span = step["msa_decode_span"]
        sparse = i in core.sparse_layers
        if sparse and k_live is None and span is not None and span.row_first == 0:
            self._decode_sparse_core(i, q_live, idx_q_live, step, out_live)
            return output
        nvfp4 = core.main_kv_is_nvfp4
        _record("m3_msa_attention", sparse, k_live is not None)
        m3_msa_attention(
            core.attn[i],
            q_live,
            k_live,
            v_live,
            idx_q_live,
            step["metadata"],
            out_live,
            sparse,
            core.inv_kv_scales[i] if nvfp4 else None,
            core.kv_scales[i] if nvfp4 else None,
        )
        return output

    def _index_scores(self, i: int, idx_q, step, num_tokens: int, qlen: int):
        """The per-index-head block maxima of a non-identity decode step: this target's CuTe DSL scorer at the request
        counts ``INDEX_SCORE`` covers (each count's CTAs per request and early issue, ``index_score_schedule``), the
        in-tree CuTe DSL scorer beyond them. Rubin TP2 (2 index heads), 165,521 tokens, behind a 58 MB PDL producer,
        per layer: 1 request 3.27 us at 1536 CTAs (in-tree 256-CTA scorer 6.07); 2 requests 4.97 us at 768 (in-tree
        6.71)."""
        config = self.core.attn[i].m3_config
        rows = int(step["num_seqs"])
        block_table = step["msa_block_table"][:rows]
        seq_lens = step["msa_seq_lens_cuda"][:rows]
        idx_q_view = idx_q.view(num_tokens, config.num_index_heads, config.sparse_index_dim)
        idx_k_cache = step["msa_idx_k_cache"](i)
        max_score = step["msa_proxy_max_score_view"](
            config.num_index_heads, step["msa_worst_case_max_k_tiles"], num_tokens
        )
        if INDEX_SCORE.covers(rows):
            split_k, early = index_score_schedule(rows)
            key = (
                idx_q_view.dtype,
                int(config.num_index_heads),
                int(qlen),
                int(idx_q_view.shape[-1]),
                int(split_k),
                bool(early),
            )
            if key not in _index_score_kernel.IndexScoreRunner.kernel_cache:
                # Outside require mode the op below compiles it (the only scorer this step has).
                missed(
                    INDEX_SCORE_PATH,
                    num_tokens,
                    f"the index scorer {key} was not compiled before warmup",
                )
            _record("m3_index_score", key)
            m3_index_score(
                idx_q_view,
                idx_k_cache.squeeze(1),
                block_table,
                seq_lens,
                max_score.transpose(1, 2),
                qlen,
                split_k,
                early,
            )
            taken(INDEX_SCORE_PATH, rows)
        else:
            uncovered(
                INDEX_SCORE_PATH,
                f"{rows} requests, past {INDEX_SCORE_MAX_REQUESTS}: the in-tree CuTe DSL scorer",
            )
            _in_tree_scorer()  # registers the op
            _record("cute_dsl_minimax_m3_index_decode_score", rows, qlen)
            torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score(
                idx_q_view,
                idx_k_cache.squeeze(1),
                block_table,
                seq_lens,
                max_score.transpose(1, 2),
                qlen,
            )
        return max_score

    def _select(self, i: int, max_score, step, num_tokens: int):
        """The top-16 table of a non-identity decode step from the block maxima reduced to the KV head: the long-row
        select (the 384-thread kernel, ``SELECT_LONG_DSL``, when compiled) on rows of ``LONG_SELECT_MIN_BLOCKS..
        LONG_SELECT_MAX_BLOCKS`` blocks, else the built-in selector (``m3_select_blocks``)."""
        config = self.core.attn[i].m3_config
        n_valid_blocks = step["msa_n_valid_blocks"][:num_tokens]
        group = config.num_index_heads // config.num_kv_heads
        max_score_kv = (
            max_score.view(config.num_kv_heads, group, max_score.shape[1], max_score.shape[2]).amax(
                dim=1
            )
            if group > 1
            else max_score
        )
        from .ops import m3_select_long_dsl as _select_long_dsl_kernel

        num_blocks = int(max_score_kv.shape[1])
        table = None
        dsl = False
        if long_select_covers(num_blocks):
            early_trigger = num_tokens <= SELECT_EARLY_MAX_ROWS
            if LONG_SELECT_DSL.keys:
                # Compiled before warmup (prebuild_kernels); a missing kernel falls back to the 1024-thread one.
                key = _select_long_dsl_kernel.kernel_key(
                    get_env_enable_pdl(), early_trigger, SELECT_LONG_TRIGGER_PHASE
                )
                dsl = key in _select_long_dsl_kernel.kernel_cache
                if not dsl:
                    missed(
                        LONG_SELECT_DSL.path,
                        num_tokens,
                        f"the 384-thread select {key} was not compiled before warmup",
                    )
            _record("long_select", num_blocks, num_tokens, dsl)
            table = long_select_blocks(
                max_score_kv,
                n_valid_blocks=n_valid_blocks,
                init_blocks=config.init_blocks,
                local_blocks=config.local_blocks,
                use_pdl=get_env_enable_pdl(),
                early_trigger=early_trigger,
                trigger_phase=SELECT_LONG_TRIGGER_PHASE,
                dsl=dsl,
            )
        if table is not None:
            taken((LONG_SELECT_DSL if dsl else LONG_SELECT).path, num_tokens)
        else:
            long_select_fallback(num_blocks, num_tokens)
            _record("m3_select_blocks", num_blocks, num_tokens)
            table = m3_select_blocks(
                max_score_kv,
                MSA_REQUIRED_TOPK,
                n_valid_blocks,
                config.init_blocks,
                config.local_blocks,
                True,
            )
        return table

    def _decode_sparse_core(self, i: int, q, idx_q, step, output) -> None:
        """Top-16 block selection and the block-sparse decode of one pure generation step of a sparse layer whose caches
        the producer already wrote.

        The selection is the identity when the metadata staged it (every request within 16 blocks). Otherwise the
        index scorer writes the per-index-head block maxima, reduced to the KV head, and the long-row selector takes the
        top 16 (the built-in selector beyond its shapes). On an NVFP4 layer the decode (``m3_sparse_decode``: the
        long-context decode or the Triton producer + merge) prefetches into L2, before its grid wait, the K/V pages of
        the blocks the same layer selected one step back (``SPARSE_DECODE_LC_PREFETCH_SELF_KV``: 65-68% of a layer's
        selected blocks recur one step later; same layer, not the next one as at TP4: the MoE between two sparse layers
        streams ~250 MB through the 126 MB L2, so an earlier issue would be evicted; hints only).
        """
        from .ops import m3_attn_lc as _attn_lc_kernel

        core = self.core
        backend = core.attn[i]
        num_tokens = int(q.shape[0])
        rows = int(step["num_seqs"])
        qlen = int(step["msa_decode_span"].query_len)
        block_table = step["msa_block_table"][:rows]
        seq_lens = step["msa_seq_lens_cuda"][:rows]
        num_kv_heads = int(backend.indexer.config.num_kv_heads)
        # The index scorers are compiled and loaded by prebuild_kernels, so no step compiles one and the identity can
        # be taken from the first decode step.
        identity = step["msa_short_ok"] and step["msa_short_table"] is not None
        if identity:
            topk = step["msa_short_table"][: num_kv_heads * num_tokens * MSA_REQUIRED_TOPK].view(
                num_kv_heads, num_tokens, MSA_REQUIRED_TOPK
            )
        else:
            max_score = self._index_scores(i, idx_q, step, num_tokens, qlen)
            topk = self._select(i, max_score, step, num_tokens).permute(1, 0, 2)
        manager = step["kv_cache_manager"]
        k_paged, v_paged = msa_paged_kv(manager, i)
        nvfp4 = {}
        if manager.is_nvfp4_layer(i):
            k_scale, v_scale = _aligned_nvfp4_dequant_scales(backend, core.kv_scales[i])
            scales = manager.get_block_scale_buffers(i, "HND")
            nvfp4 = dict(
                k_block_scale=scales[:, 0],
                v_block_scale=scales[:, 1],
                k_global_scale=k_scale,
                v_global_scale=v_scale,
            )
        # The long-context decode (if it runs) takes the layer's N1 id buffer; any other route leaves the slot unused,
        # and it is cleared after the call either way.
        n1_ids = core.n1_ids.get(i)
        if n1_ids is not None and nvfp4 and num_tokens <= N1_MAX_ROWS:
            _attn_lc_kernel.N1_IDS.ids = n1_ids
        _record("m3_sparse_decode", num_tokens, qlen, identity)
        try:
            m3_sparse_decode(
                q.view(num_tokens, core.heads, core.head_dim),
                k_paged,
                v_paged,
                topk,
                block_table,
                seq_lens,
                sm_scale=core.sm_scale(i),
                output=output.view(num_tokens, core.heads, core.head_dim),
                decode_query_len=qlen,
                identity_blocks=identity,
                **nvfp4,
            )
        finally:
            _attn_lc_kernel.N1_IDS.ids = None


_ENGAGED: set = set()


def _log_engaged(path: str, mapping) -> None:
    """One info line per fast path and process (every GEN rank logs its own): the run's engagement evidence."""
    if path not in _ENGAGED:
        _ENGAGED.add(path)
        rank = getattr(mapping, "tp_rank", "?") if mapping is not None else "?"
        logger.info(f"[M3 sm_107 tp2] fast path engaged: {path} (tp rank {rank})")


class M3SpecStepFold:
    """The target's speculative step-input staging (the in-tree ``SpecStepFold`` protocol).

    Between two decode graphs of a one-model Eagle3 step, the spec sampler stores the step's outputs into its per-slot
    store and the engine gathers them back into its flat input buffers for the next step. This object does both in
    one launch each (``m3_spec_tail.slot_scatter`` / ``step_gather``): integer copies only, the same values the
    engine's and the sampler's own copies write.

    The forward of a decode-only step also does both at its tail (``store_in_forward``, one ``m3_fold_tail`` launch
    inside the step graph, this target's C++ op): it stores the step's outputs for the rows ``begin_step`` armed and
    stages the next step's inputs for those rows in that order. The arming is a device-side row count and slot list
    staged before the replay, so a captured graph whose step was not armed writes nothing. The sampler then skips its
    store for exactly the armed rows (``sampler_rows_stored``), and the next step skips its gathers when its rows are
    those rows (``begin_step``). The sampler copies the store to the host on its D2H side stream and records
    ``store_copies_done`` after the copies; the forward's tail waits on it before it rewrites the store.

    * ``store``       -- the spec sampler's per-slot store (``bind_sampler_store``).
    * ``step_inputs`` -- the engine's (input_ids, draft_tokens, position_id_offsets, kv_lens_offsets) device buffers
      (``bind_engine_buffers``).
    """

    # Read by the shell's forward (the tail runs only when the fold stores in the forward).
    in_forward = True

    def __init__(self) -> None:
        self.store: Optional[object] = None
        self.step_inputs: Optional[tuple] = None
        # Device arming read by the forward's tail: ctrl[0] = armed rows (0 = write nothing), slots[:rows].
        self._ctrl: Optional[torch.Tensor] = None
        self._slots: Optional[torch.Tensor] = None
        # The rows (sequence slots) the next forward stores and stages for, as begin_step armed them.
        self._armed: Optional[tuple] = None
        # The rows the previous forward staged the next step's inputs for (confirmed by sampler_rows_stored).
        self._staged_for: Optional[tuple] = None
        # Output buffers the tail ran on: the captured graphs' static outputs (by address) and the last eager step's.
        self._graph_outputs: set = set()
        self._eager_outputs: Optional[torch.Tensor] = None
        self._logged = False
        # The (external) event the sampler records on its D2H side stream after copying the store to the host, waited
        # on by the forward's tail before it rewrites the store (see SpecSamplerBase._copy_outputs_to_host).
        self.store_copies_done: Optional[torch.cuda.Event] = None

    def bind_engine_buffers(
        self, input_ids, draft_tokens, position_id_offsets, kv_lens_offsets, max_batch_size
    ):
        self.step_inputs = (input_ids, draft_tokens, position_id_offsets, kv_lens_offsets)
        # Recorded once now so a captured wait has a completed record to wait on before the first copy.
        self.store_copies_done = torch.cuda.Event(external=True)
        self.store_copies_done.record()
        logger.info(
            "[M3 host] spec store host copies on the D2H side stream, awaited by the forward's tail"
        )
        logger.info("[M3 host] spec fold tail: one C++ launch (no Triton) at the forward's tail")
        self._ctrl = torch.zeros(1, dtype=torch.int32, device=input_ids.device)
        self._slots = torch.zeros(max(max_batch_size, 1), dtype=torch.long, device=input_ids.device)

    def bind_sampler_store(self, store: object) -> None:
        self.store = store

    def begin_step(self, seq_slots: Optional[list], num_rows: int) -> bool:
        if self._ctrl is None:
            return False
        rows = tuple(seq_slots) if seq_slots is not None else None
        staged = rows is not None and rows == self._staged_for
        self._staged_for = None
        armed = rows if rows is not None and 0 < len(rows) <= self._slots.numel() else None
        # Read only by this step's forward, so both uploads may wait for the step-input pack's flush.
        if armed is not None:
            slots = torch.tensor(armed, dtype=torch.long, pin_memory=prefer_pinned())
            copy_h2d_if_changed(self._slots[: len(armed)], slots, pack=True)
        ctrl = torch.tensor(
            [len(armed) if armed is not None else 0], dtype=torch.int32, pin_memory=prefer_pinned()
        )
        copy_h2d_if_changed(self._ctrl, ctrl, pack=True)
        self._armed = armed
        return staged

    # The launches the store wait may precede. store_copies_done is an external event: under graph capture its wait
    # is a node of its own, so the kernel launched right after it loses its programmatic (PDL) edge and with it the
    # overlap of its pre-wait part. Only a launch that waits for its predecessor's completion anyway may follow it:
    # the fold tail (a plain launch, no griddepcontrol). Waiting before the last draft_step instead cost +13 / +17
    # us/step.
    STORE_WAIT_BEFORE = frozenset({"fold_tail"})

    def _wait_store_copies(self, before: str) -> None:
        """Order the store's rewrite after the sampler's host copies of it; ``before`` names the launch that follows.
        Under graph capture a launch outside STORE_WAIT_BEFORE fails the capture."""
        if self.store_copies_done is None:
            return
        if before not in self.STORE_WAIT_BEFORE and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"M3SpecStepFold: the store wait would precede {before!r}, a launch outside "
                f"{sorted(self.STORE_WAIT_BEFORE)}; an external-event wait node drops the next kernel's PDL edge"
            )
        torch.cuda.current_stream().wait_event(self.store_copies_done)

    def store_in_forward(self, outputs: dict) -> None:
        """At the forward's tail: store the armed rows' outputs and stage the next step's inputs (one launch)."""
        if self._ctrl is None or self.store is None or self.step_inputs is None:
            return
        store = self.store
        input_ids, draft_tokens, position_id_offsets, kv_lens_offsets = self.step_inputs
        new_tokens = outputs.get("new_tokens")
        next_new_tokens = outputs.get("next_new_tokens")
        new_tokens_lens = outputs.get("new_tokens_lens")
        next_draft_tokens = outputs.get("next_draft_tokens")
        if any(
            t is None or not isinstance(t, torch.Tensor)
            for t in (new_tokens, next_new_tokens, new_tokens_lens, next_draft_tokens)
        ):
            return
        rows = new_tokens.shape[0]
        # The previous step's host copies of the store run on the sampler's side stream.
        self._wait_store_copies(before="fold_tail")
        _record("m3_fold_tail", rows)
        if not m3_fold_tail.fold_tail(
            self._ctrl,
            self._slots,
            min(rows, self._slots.numel()),
            new_tokens,
            store.new_tokens.squeeze(-1).T,
            next_new_tokens,
            store.next_new_tokens.squeeze(-1).T,
            next_draft_tokens,
            store.next_draft_tokens,
            new_tokens_lens,
            store.new_tokens_lens,
            input_ids,
            draft_tokens,
            position_id_offsets,
            kv_lens_offsets,
            next_new_tokens.shape[1],
            cpp=True,
        ):
            return
        if torch.cuda.is_current_stream_capturing():
            self._graph_outputs.add(new_tokens.data_ptr())
        else:
            self._eager_outputs = new_tokens

    def gather(
        self,
        new_tokens,
        next_draft_tokens,
        new_tokens_lens,
        previous_slots,
        previous_pos_indices,
        previous_batch_len: int,
        tokens_per_gen_step: int,
        draft_token_buffer_width: int,
        num_tokens: int,
        num_draft_tokens: int,
        first_previous_row: int,
    ) -> bool:
        if self.step_inputs is None:
            return False
        input_ids, draft_tokens, position_id_offsets, kv_lens_offsets = self.step_inputs
        return m3_spec_tail.step_gather(
            new_tokens,
            next_draft_tokens,
            new_tokens_lens,
            previous_slots,
            previous_pos_indices,
            previous_batch_len,
            tokens_per_gen_step,
            draft_token_buffer_width,
            input_ids,
            num_tokens,
            draft_tokens,
            num_draft_tokens,
            position_id_offsets,
            kv_lens_offsets,
            first_previous_row,
        )

    def sampler_rows_stored(self, outputs: dict, num_skip: int, seq_slots: list) -> bool:
        armed, self._armed = self._armed, None
        new_tokens = outputs.get("new_tokens")
        ran = isinstance(new_tokens, torch.Tensor) and (
            new_tokens is self._eager_outputs or new_tokens.data_ptr() in self._graph_outputs
        )
        self._eager_outputs = None
        stored = (
            ran
            and armed is not None
            and num_skip == 0
            and tuple(seq_slots) == armed
            and new_tokens.shape[0] == len(armed)
        )
        self._staged_for = armed if stored else None
        if stored and not self._logged:
            self._logged = True
            logger.info("[M3 host] spec step stored and next step staged in the forward tail")
        return stored

    def scatter(
        self, seq_slots, new_tokens, next_new_tokens, new_tokens_lens, next_draft_tokens
    ) -> bool:
        store = self.store
        if store is None:
            return False
        return m3_spec_tail.slot_scatter(
            seq_slots,
            new_tokens,
            store.new_tokens.squeeze(-1).T,
            next_new_tokens,
            store.next_new_tokens.squeeze(-1).T,
            new_tokens_lens,
            store.new_tokens_lens,
            next_draft_tokens,
            store.next_draft_tokens,
        )


class _DraftForward:
    """``draft_model.model`` for the in-tree worker: the drafter layer's forward, called as
    ``model(input_ids=, position_ids=, hidden_states=, attn_metadata=, spec_metadata=, all_rank_num_tokens=)`` and
    returning (hidden, hidden to carry to the next pass). ``d2t`` None: the draft vocabulary is the target's."""

    d2t = None

    def __init__(self, draft: "M3Eagle3Draft") -> None:
        self._draft = draft

    def __call__(self, **kwargs):
        return self._draft.forward_layer(**kwargs)


# The drafter's projections by weight-table name, for the low-M linears' engagement and prebuild.
_DRAFT_LINEARS = ("fc", "qkv", "o", "gu", "dn")


class M3Eagle3Draft(nn.Module):
    """The one-layer BF16 Eagle3 drafter (``speculative_model``): its weights (``weights.DRAFT_WEIGHTS``), its
    vocab-parallel BF16 head and logits processor, the drafter layer's TRTLLM attention backend (layer index = the
    target's layer count, FP8 KV) and its TP all-reduce.

    What the worker reads: ``model`` (the layer forward), ``apply_eagle3_fc`` (the per-layer fc_norms + fc over the
    captured target hidden states), ``lm_head`` and ``logits_processor``; the fused draft loop reads the layer's parts
    (``draft_input_weights``, ``attention``, ``mlp``, ``ar_norm``). The drafter embeds with the target's table.

    The BF16 projections (fc, qkv, o_proj, gate_up, down; not the head) run FlashInfer's fixed-tactic low-M kernels
    (``ops/gemm/m3_draft_lowm``, the TP8 port) at the rows ``DRAFT_LOWM`` covers, compiled before the warmup
    (``prebuild_linears``); every other call is the in-tree cuBLAS GEMM. Rubin TP2 (mclk locked, 4 rotating sets) at
    1 row: qkv 13.18 -> 9.71, o 7.17 -> 5.33, gate_up 22.27 -> 17.70, down 12.52 -> 9.76, fc 21.50 -> 17.77 us. The
    split-K ('s') tactics run this target's copy of FlashInfer's kernel (``ops/m3_lowm_splitk``, bit for bit the
    same output); with ``DRAFT_AR_PUSH`` the o_proj and down projection of the fused draft loop also push their rows
    into the drafter boundary all-reduce's exchange (``arm_push`` / ``take_pushed``), as do the direct ('d') 1-row
    linears through this target's copy of the direct kernel (``ops/m3_lowm_direct``; o_proj 9.22 -> 8.09 us,
    down 13.81 -> 12.72 us per 1-row boundary).
    """

    def __init__(
        self,
        core: MiniMaxM3Tp2ModelingV2,
        draft_config: ModelConfig,
        start_layer_idx: int,
        capture_layers: int,
    ) -> None:
        super().__init__()
        cfg = draft_config.pretrained_config
        _ensure_draft_vocab_size(cfg)
        eagle = getattr(cfg, "eagle_config", None) or {}
        mapping = draft_config.mapping
        self.draft_config = draft_config
        # The engine's draft weight mapper binds to these two.
        self.model_config = draft_config
        self.config = cfg
        self.capture_layers = int(capture_layers)
        self.layer_idx = int(start_layer_idx)
        self.dtype = cfg.torch_dtype
        self.hidden = int(cfg.hidden_size)
        self.hidden_size_in = int(getattr(cfg, "target_hidden_size", cfg.hidden_size))
        self.rms_eps = float(cfg.rms_norm_eps)
        self.return_hidden_post_norm = bool(
            eagle.get("return_hidden_post_norm", False) or getattr(cfg, "norm_output", False)
        )
        if (
            int(cfg.num_hidden_layers) != 1
            or eagle.get("eh_proj_before_attn", False)
            or eagle.get("norm_before_fc", False)
            or not getattr(cfg, "fc_norm", False)
            or getattr(cfg, "kv_lora_rank", None)
            or self.capture_layers <= 1
            or self.hidden_size_in != self.hidden
            or cfg.draft_vocab_size != cfg.vocab_size
            or getattr(cfg, "attention_bias", False)
            or getattr(cfg, "mlp_bias", False)
            or getattr(cfg, "bias", False)
            or getattr(cfg, "tie_word_embeddings", False)
        ):
            raise ValueError(
                "this MiniMax-M3 target drafts with a one-layer Llama-style Eagle3 drafter with per-layer "
                "fc_norms over "
                "several captured layers, the target's hidden size and vocabulary and no biases"
            )
        tp = mapping.tp_size
        self.heads = cfg.num_attention_heads // tp
        self.kv_heads = (cfg.num_key_value_heads + tp - 1) // tp
        self.head_dim = int(
            getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        )
        self.q_size = self.heads * self.head_dim
        self.kv_size = self.kv_heads * self.head_dim
        self.w = _weights.DRAFT_WEIGHTS.declare(_weights.DRAFT_WEIGHTS.dims(self))
        self.lm_head = LMHead(
            cfg.draft_vocab_size,
            cfg.hidden_size,
            dtype=cfg.torch_dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=draft_config.lm_head_gather_output,
            reduce_output=False,
            use_custom_cublas_mm=False,
            quant_config=None,
        )
        self.logits_processor = LogitsProcessor()
        self.allreduce = AllReduce(
            mapping=mapping, strategy=draft_config.allreduce_strategy, dtype=cfg.torch_dtype
        )
        quant_config = draft_config.get_quant_config()
        self.attn = create_attention(
            draft_config.attn_backend,
            self.layer_idx,
            self.heads,
            self.head_dim,
            self.kv_heads,
            pos_embd_params=PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gpt_neox, rope=RopeParams.from_config(cfg)
            ),
            quant_config=quant_config,
            skip_create_weights_in_init=draft_config.skip_create_weights_in_init,
            q_scaling=1.0,
            attention_chunk_size=None,
            sparse_params=None,
        )
        if not self.attn.support_fused_rope() or not self.attn.support_fused_qkv():
            raise ValueError(
                f"the drafter's attention backend {type(self.attn).__name__} must fuse RoPE and QKV"
            )
        self.attn.update_quant_config(quant_config)
        self._bcg_key = f"m3_tp2_draft_{id(self)}"
        _BCG_DRAFTS[self._bcg_key] = self
        # The drafter embeds with the target's table, read off the core on every call (the engine materializes the
        # core's weights after construction).
        self._core = [core]
        self.model = _DraftForward(self)
        # The low-M linears: (N, K, M, PDL) of every call prebuild_linears compiled, and (N, K, M) of every call that
        # logged its engagement line.
        self._lowm_built: set = set()
        self._lowm_engaged: set = set()
        # DRAFT_AR_PUSH: the weight whose next low-M call also pushes its rows into a boundary all-reduce (data
        # pointer, the bucket's multicast words, its flags, rank, world, max tokens), set by arm_push; and the data
        # pointer of the output the last armed call pushed, 0 if none (read once by take_pushed). Host state set and
        # read in one forward's issue order.
        self._push_arm: Optional[tuple] = None
        self._pushed = 0

    @property
    def embed_table(self) -> torch.Tensor:
        return self._core[0].w["embed"]

    def load_weights_from_target_model(self, target_model: nn.Module) -> None:
        """Nothing to alias: the drafter reads the target's embedding table through the core and owns its head."""

    # ------------------------------------------------------------------------------------------- the projections

    def linear(self, x: torch.Tensor, key: str) -> torch.Tensor:
        """``x @ w[key]^T``: the low-M kernel where ``DRAFT_LOWM`` covers the call, else the in-tree cuBLAS GEMM."""
        w = self.w[key]
        out = self._lowm_linear(w, x)
        return out if out is not None else m3_bf16_linear(x, w)

    def _lowm_linear(self, weight: torch.Tensor, x: torch.Tensor) -> Optional[torch.Tensor]:
        """x [M, K] @ weight^T through the low-M kernel when ``DRAFT_LOWM`` covers it; None for the in-tree GEMM.
        A projection none of whose row counts is covered never reaches the low-M checks."""
        n, k = weight.shape
        if x.dim() != 2 or not any(nk == (n, k) for nk, _ in DRAFT_LOWM.keys):
            return None
        m = int(x.shape[0])
        if not DRAFT_LOWM.covers(((n, k), m)):
            uncovered(DRAFT_LOWM.path, f"the drafter's {n}x{k} linear past its tactic table's rows")
            return None
        if not x.is_contiguous():
            # The first pass's fc may read a column window of the captured hidden states; the in-tree GEMM takes
            # strides.
            uncovered(DRAFT_LOWM.path, "a strided activation (the in-tree GEMM reads it in place)")
            return None
        use_pdl = get_env_enable_pdl()
        if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            missed(DRAFT_LOWM.path, m, "the activation or weight is not BF16")
            return None
        if (n, k, m, use_pdl) not in self._lowm_built:
            missed(
                DRAFT_LOWM.path, m, f"the {n}x{k} kernel at {m} rows was not compiled before warmup"
            )
            return None
        kind, tactic = tactic_for(n, k, m)
        if (n, k, m) not in self._lowm_engaged:
            self._lowm_engaged.add((n, k, m))
            logger.info(
                f"[M3 sm_107 tp2] fast path engaged: m3_draft_lowm {n}x{k} at {m} rows (device {x.device.index})"
            )
        out = torch.empty((m, n), dtype=x.dtype, device=x.device)
        ar = None
        if (
            self._push_arm is not None
            and self._push_arm[0] == weight.data_ptr()
            and (
                (kind == "s" and DRAFT_LOWM_OWN_SPLITK)
                or (kind == "d" and DRAFT_AR_PUSH_DIRECT and tactic[1] % 2 == 0)
            )
        ):
            ar = self._push_arm[1:]
            self._pushed = out.data_ptr()
        _record("m3_draft_lowm", n, k, m, kind, ar is not None)
        return m3_draft_lowm(x, weight, out, kind, tactic, use_pdl, DRAFT_LOWM_OWN_SPLITK, ar)

    def prebuild_linears(self) -> int:
        """Compile (and run once) every covered row count of every drafter projection, outside capture, on the
        current stream, before the executor's warmup. Returns the number of calls compiled."""
        from .ops.m3_lowm_direct import compile_direct_push
        from .ops.m3_lowm_splitk import compile_push

        use_pdl = get_env_enable_pdl()
        built = 0
        shapes = {tuple(self.w[key].shape): self.w[key] for key in _DRAFT_LINEARS}
        for (n, k), weight in sorted(shapes.items()):
            if weight.dtype != torch.bfloat16:
                continue
            for _, m in sorted(key for key in DRAFT_LOWM.keys if key[0] == (n, k)):
                if (n, k, m, use_pdl) in self._lowm_built:
                    continue
                kind, tactic = tactic_for(n, k, m)
                x = torch.zeros(m, k, dtype=torch.bfloat16, device=weight.device)
                out = torch.empty(m, n, dtype=torch.bfloat16, device=weight.device)
                m3_draft_lowm(x, weight, out, kind, tactic, use_pdl, DRAFT_LOWM_OWN_SPLITK)
                if DRAFT_AR_PUSH and DRAFT_LOWM_OWN_SPLITK and kind == "s":
                    # The push variant a boundary all-reduce may arm (o_proj / down): compiled only, never run here
                    # (its pushes need the all-reduce's exchange and a matching call).
                    compile_push(x, weight.t(), out, use_pdl, tactic)
                if DRAFT_AR_PUSH and DRAFT_AR_PUSH_DIRECT and kind == "d" and tactic[1] % 2 == 0:
                    # The direct kernel's push variant (ops/m3_lowm_direct), compiled only.
                    compile_direct_push(x, weight.t(), out, use_pdl, tactic)
                self._lowm_built.add((n, k, m, use_pdl))
                built += 1
        return built

    def arm_push(self, key: str, workspace, rows: int) -> None:
        """Arm ``w[key]``'s next covered low-M call to push into ``workspace``'s bucket for ``rows``."""
        bucket = m3_mc_allreduce._bucket(workspace, rows)
        flags = None if bucket is None else bucket.flags_for(8)
        self._pushed = 0
        self._push_arm = None
        if bucket is not None and flags is not None and bucket.max_tokens == 8:
            self._push_arm = (
                self.w[key].data_ptr(),
                bucket.mc.data_ptr(),
                flags.data_ptr(),
                bucket.rank,
                bucket.world_size,
                bucket.max_tokens,
            )

    def take_pushed(self, x: torch.Tensor) -> bool:
        """Whether the armed call pushed its rows; disarms. ``x`` is what the boundary all-reduce is about to reduce:
        it must be the pushing call's own output (the prepushed all-reduce reads only the pushed words, never ``x``),
        so an operation between that call and the all-reduce raises instead of silently reducing the wrong rows."""
        pushed, self._pushed, self._push_arm = self._pushed, 0, None
        if pushed and x.data_ptr() != pushed:
            raise RuntimeError(
                "the drafter's pushed rows are not the boundary all-reduce's input (an operation ran in between)"
            )
        return bool(pushed)

    # ------------------------------------------------------------------------------------------------- the layer

    def fc(self, hs: torch.Tensor) -> torch.Tensor:
        return self.linear(hs, "fc")

    def apply_eagle3_fc(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The captured target hidden states [T, layers x H] to the drafter's input width: each layer's fc_norm,
        then the fc (the in-tree ``Eagle3ForCausalLM.apply_eagle3_fc``)."""
        hidden_states = hidden_states.to(self.dtype)
        if hidden_states.shape[-1] != self.hidden:
            chunks = hidden_states.chunk(self.capture_layers, dim=-1)
            hidden_states = torch.cat(
                [
                    flashinfer_rmsnorm(c, self.w[f"fc_norm{i}"], self.rms_eps)
                    for i, c in enumerate(chunks)
                ],
                dim=-1,
            )
            hidden_states = self.fc(hidden_states)
        return hidden_states

    def attention(self, x: torch.Tensor, md, reduce: bool) -> torch.Tensor:
        """QKV, the backend's fused RoPE + KV append + causal attention, o_proj (all-reduced unless ``reduce`` is
        False, where the caller fuses the reduction). The backend's generation phase runs this target's full-context
        decode with the fused RoPE + FP8 K/V append when the shell installed it (``ops/m3_drafter_attn``)."""
        qkv = self.linear(x, "qkv")
        _record("m3_draft_attention", int(md.num_tokens))
        if is_in_breakable_cuda_graph():
            # A context step's draft pass is captured with the target's: the attention runs in an eager break, on the
            # step's metadata (the drafter's: the runtime's draft swap installs it on the target's object).
            if get_model_extra_attrs()["attention_metadata"]() is not md:
                raise RuntimeError(
                    "MiniMax-M3 drafter under a breakable prefill CUDA graph: its metadata is not the step's"
                )
            o = qkv.new_empty((qkv.shape[0], self.q_size), dtype=torch.bfloat16)
            _draft_attention_eager(qkv, o, self._bcg_key)
        else:
            o = m3_draft_attention(self.attn, qkv, md)
        out = self.linear(o, "o")
        return m3_allreduce(out, self.allreduce, None) if reduce else out

    def mlp(self, h: torch.Tensor, reduce: bool, cpp_silu: bool = False) -> torch.Tensor:
        """gate_up, the plain SwiGLU, down (all-reduced unless ``reduce`` is False). ``cpp_silu`` (the fused draft
        loop): the SwiGLU on this target's C++ op (``ops/spec/m3_draft_silu``, minimaxM3Sm107Tp2/siluAndMul.cu), bit
        for bit the in-tree Triton silu_and_mul in one launch either way (no Triton on the GEN decode path)."""
        gu = self.linear(h, "gu")
        act = None
        if cpp_silu:
            act = m3_draft_silu(gu)
            if act is None:
                uncovered(
                    DRAFT_SILU_PATH, "the drafter SwiGLU input layout (the Triton op runs it)"
                )
            else:
                taken(DRAFT_SILU_PATH, int(gu.shape[0]), "drafter MLP SwiGLU, one C++ launch")
        if act is None:
            act = m3_silu_and_mul(gu, None, None, None)
        out = self.linear(act, "dn")
        return m3_allreduce(out, self.allreduce, None) if reduce else out

    def ar_norm(self, x: torch.Tensor, residual: torch.Tensor, norm_key: str):
        """One draft-layer boundary in the in-tree structure: the projection's plain all-reduce (BF16 out), then the
        fused residual add + RMSNorm (FP32 add, norm of the FP32 sum). Returns (normed, residual sum)."""
        return flashinfer_fused_add_rmsnorm(
            m3_allreduce(x, self.allreduce, None), residual, self.w[norm_key], self.rms_eps
        )

    def forward_layer(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata,
        spec_metadata,
        all_rank_num_tokens: Optional[List[int]] = None,
    ):
        """The drafter layer (the in-tree ``Eagle3DraftModel.forward`` for one layer): the embedded tokens and the
        fc'd hidden states normed and concatenated, attention, the post-attention norm with the residual, the MLP and
        the final norm. Returns (normed output, the hidden state the next pass carries)."""
        previous = attn_metadata.all_rank_num_tokens
        if all_rank_num_tokens is not None:
            attn_metadata.all_rank_num_tokens = all_rank_num_tokens
        try:
            embeds = m3_embedding(input_ids, self.embed_table).to(self.dtype)
            residual = hidden_states
            h = flashinfer_rmsnorm(hidden_states, self.w["hidden_norm"], self.rms_eps)
            e = flashinfer_rmsnorm(embeds, self.w["in_norm"], self.rms_eps)
            h = torch.cat([e, h], dim=-1)
            h = self.attention(h, attn_metadata, reduce=True)
            h, residual = flashinfer_fused_add_rmsnorm(
                h, residual, self.w["post_norm"], self.rms_eps
            )
            h = self.mlp(h, reduce=True)
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, h, residual)
            h, to_save = flashinfer_fused_add_rmsnorm(
                h, residual, self.w["final_norm"], self.rms_eps
            )
            return (h, h) if self.return_hidden_post_norm else (h, to_save)
        finally:
            if all_rank_num_tokens is not None:
                attn_metadata.all_rank_num_tokens = previous


def _eagle3_capture(spec_metadata, layer_id: int, hidden_states, residual, capture_norm) -> None:
    """The Eagle3 hidden-state capture of one target layer.

    Mirrors ``Eagle3OneModelSpecMetadata.maybe_capture_hidden_states``. With ``capture_norm`` = (the drafter's stacked
    fc_norm weights [layers, H], eps), the capture writes rmsnorm(bf16(h + residual)) * fc_norm[i] in one kernel (the
    in-tree capture's bf16 add, then the in-tree fc_norm arithmetic), which the fused first drafter inputs read as is;
    otherwise it is the in-tree capture.
    """
    if capture_norm is None:
        spec_metadata.maybe_capture_hidden_states(layer_id, hidden_states, residual)
        return
    for i, captured_layer_id in enumerate(spec_metadata.layers_to_capture):
        if captured_layer_id == layer_id:
            assert spec_metadata.hidden_states is not None
            num_tokens = spec_metadata.num_capture_tokens
            if num_tokens > spec_metadata.hidden_states.shape[0]:
                raise RuntimeError(
                    "EAGLE3 hidden-state capture token count exceeds buffer capacity: "
                    f"num_tokens={num_tokens}, capacity={spec_metadata.hidden_states.shape[0]}"
                )
            if num_tokens > hidden_states.shape[0]:
                raise RuntimeError(
                    "EAGLE3 hidden-state capture token count exceeds available hidden states: "
                    f"num_tokens={num_tokens}, hidden_states={hidden_states.shape[0]}"
                )
            hidden_size = spec_metadata.hidden_size
            m3_spec_tail.capture_norm(
                hidden_states[:num_tokens],
                residual[:num_tokens] if residual is not None else None,
                capture_norm[0][i],
                capture_norm[1],
                spec_metadata.hidden_states[:num_tokens, i * hidden_size : (i + 1) * hidden_size],
            )
            break


def _sharded_head(lm_head: nn.Module) -> bool:
    """A vocab-sharded TP head without padding (the fused kernels read the local shard)."""
    return (
        getattr(lm_head, "tp_mode", None) == TensorParallelMode.COLUMN
        and getattr(lm_head, "padding_size", 0) == 0
        and getattr(lm_head, "tp_size", 1) > 1
    )


class _M3ShardDraftLogits(nn.Module):
    """W2b: the drafter's logits processor on an in-tree draft step: this rank's BF16 vocab shard of the drafter head
    (no all-gather, no fp32 cast), which the worker's greedy top-1 reads (``greedy_sample_draft_with_tp_gather``).
    Rows the caller has not selected (``return_context_logits`` False) take the original processor."""

    def __init__(self, original: nn.Module):
        super().__init__()
        self._m3_original = [original]  # not a submodule: the target shares it

    def forward(self, hidden_states, lm_head, attn_metadata, return_context_logits: bool = False):
        if not return_context_logits:
            return self._m3_original[0](
                hidden_states, lm_head, attn_metadata, return_context_logits
            )
        return lm_head.apply_linear(hidden_states, None)


class M3Eagle3Worker(Eagle3OneModelWorker):
    """``Eagle3OneModelWorker`` with the MiniMax-M3 fused verify and fused draft loop, at the in-tree numerics.

    * Verify: on decode-only batches whose verify rows the fused kernels cover (``tail_eligible``), the target's BF16
      lm_head computes this rank's vocabulary shard of the logits (the in-tree head's own local GEMM, as two
      half-vocabulary GEMMs on the die that holds each half at ``UGPU_HEAD``'s rows) and one kernel replaces the logits
      all-gather, the argmax (or the temperature + top-p draw) and the strict acceptance.
    * Draft: the drafter's first pass may take the per-layer fc_norms in one launch (or folded into the hidden-state
      capture) and run its fc beside the target head GEMM (forked right after the last captured target layer at most
      ``FC_EARLY_MAX_ROWS`` verify rows; after a sampled verify's head at most ``FC_AFTER_HEAD_MAX_ROWS``). On a
      decode-only step ``DRAFT_STEP`` covers, the draft loop is fused: each pass's draft input in one launch, the
      drafter layer's attention and MLP with their all-reduces fused with the residual add and the next RMSNorm (the
      multicast all-reduce at ``DRAFT_MC_AR``'s rows, fed by the projection's push), and one kernel for the BF16 head,
      the cross-rank argmax and the pass's bookkeeping.

    Every other batch takes the in-tree paths. Methods whose docstring says "Mirrors" copy the named in-tree method
    with the fused branches inserted.
    """

    def __init__(self, spec_config, mapping, use_separate_draft_kv_cache: bool = False) -> None:
        super().__init__(
            spec_config, mapping=mapping, use_separate_draft_kv_cache=use_separate_draft_kv_cache
        )
        # Per-step state, reset at the top of every _forward_impl: the first draft pass's gather rows when this step
        # takes the fused first drafter inputs, and the fc output a fork computed beside the target head.
        self._m3_first_gather_ids: Optional[torch.Tensor] = None
        self._m3_fc_hidden: Optional[torch.Tensor] = None
        self._m3_target_lm_head: Optional[nn.Module] = None
        # (fc_norm0 weight, stacked fc_norm weights [layers, H]), built eagerly on the first fused first pass.
        self._m3_fc_norm_w: Optional[tuple] = None
        # Armed per step by arm_capture_norm: (stacked fc_norm weights, eps) or None.
        self._m3_capture_norm: Optional[tuple] = None
        self._m3_aux_stream: Optional[torch.cuda.Stream] = None
        # Armed per step by arm_capture_norm: the drafter whose first fc is forked at the last captured target layer,
        # and the fork that did it, joined by _m3_target_accept.
        self._m3_fc_early_drafter: Optional[M3Eagle3Draft] = None
        self._m3_fc_early_stream: Optional[torch.cuda.Stream] = None
        # The 12-bit coded copy of the target head shard the fused verify streams, built before the warmup.
        # The die-placed target head (uGPU), built before the warmup, or None.
        self._m3_ugpu_head = None
        self._m3_force_vec_warm = False
        # The streamed copy of the drafter's BF16 head shard the fused draft loop reads, built before the warmup, the
        # head weight version it was built from, and the replaced copies a captured graph may still read.
        self._m3_draft_head = None
        self._m3_draft_head_version: Optional[int] = None
        self._m3_draft_head_graveyard: list = []
        # Per step: the next step's input tokens [nreq, D + 1], column 0 written by the fused verify (or the seed) and
        # columns 1.. by the fused draft passes; set as _m3_next_new_tokens once the fused draft loop filled it.
        self._m3_next_new: Optional[torch.Tensor] = None
        self._m3_next_new_tokens: Optional[torch.Tensor] = None
        # Per step: the first draft inputs are the fused ones (accepted tokens as they lie, fc output of every row).
        self._m3_fused_first_inputs = False

    def capture_target(self, spec_metadata, layer_id: int, num_tokens: int) -> Optional[tuple]:
        """Where this step's capture norm writes layer ``layer_id``'s capture, for a boundary all-reduce that writes it
        itself (CAPTURE_NORM_IN_BOUNDARY): (the capture buffer's [num_tokens, H] column block, the layer's fc_norm
        weight, eps), or None (no capture norm armed, not a captured layer, or more rows than the buffer). The
        boundary writes every row it reduces; rows past num_capture_tokens are unread."""
        capture_norm = self._m3_capture_norm
        if capture_norm is None or spec_metadata.hidden_states is None:
            return None
        if num_tokens > spec_metadata.hidden_states.shape[0]:
            return None
        hidden_size = spec_metadata.hidden_size
        for i, captured_layer_id in enumerate(spec_metadata.layers_to_capture):
            if captured_layer_id == layer_id:
                out = spec_metadata.hidden_states[
                    :num_tokens, i * hidden_size : (i + 1) * hidden_size
                ]
                return out, capture_norm[0][i], capture_norm[1]
        return None

    def capture_hidden_states(
        self, spec_metadata, layer_id: int, hidden_states, residual=None, written: bool = False
    ) -> None:
        """The Eagle3 capture with this step's capture norm (``written``: the layer's boundary all-reduce already wrote
        it, see ``capture_target``); on a step armed for the early fc fork (``arm_capture_norm``), the last captured
        layer then forks the drafter's first fc onto the aux stream, where it overlaps the remaining target layers
        and the head (joined by ``_m3_target_accept``)."""
        if not written:
            _eagle3_capture(spec_metadata, layer_id, hidden_states, residual, self._m3_capture_norm)
        drafter = self._m3_fc_early_drafter
        layers = spec_metadata.layers_to_capture
        if drafter is not None and layers and layer_id == layers[-1]:
            self._m3_fc_early_drafter = None
            self._m3_fc_early_stream = self._m3_fc_fork(
                spec_metadata, spec_metadata.num_capture_tokens, drafter
            )

    def prebuild(self, lm_head: nn.Module, draft: Optional[M3Eagle3Draft]) -> None:
        """From the loaded weights, before the executor's warmup, on every rank: the die-placed copy of the target head
        (uGPU), the fused draft loop's head copy and the drafter's low-M
        linears."""
        start = time.perf_counter()
        if self.prebuild_ugpu_head(lm_head):
            torch.cuda.synchronize()
            logger.info(
                "[M3 prebuild] uGPU target head copy placed per die before warmup: %.1f s"
                % (time.perf_counter() - start)
            )
        start = time.perf_counter()
        if self.prebuild_draft_head(draft):
            torch.cuda.synchronize()
            logger.info(
                "[M3 prebuild] fused draft head built before warmup: %.1f s"
                % (time.perf_counter() - start)
            )
        start = time.perf_counter()
        built = draft.prebuild_linears() if draft is not None else 0
        torch.cuda.synchronize()
        logger.info(
            "[M3 prebuild] drafter low-M GEMMs built before warmup: %d calls, %.1f s"
            % (built, time.perf_counter() - start)
        )

    def _forward_impl(
        self,
        input_ids,
        position_ids,
        hidden_states,
        logits,
        attn_metadata,
        spec_metadata,
        draft_model,
        resource_manager=None,
    ):
        """Mirrors Eagle3OneModelWorker._forward_impl, with the fused verify branch."""
        runtime_draft_len = spec_metadata.runtime_draft_len
        # skip the draft forward if the runtime draft length is 0
        if runtime_draft_len == 0:
            return self.skip_drafting(
                input_ids,
                position_ids,
                hidden_states,
                logits,
                attn_metadata,
                spec_metadata,
                draft_model,
            )

        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts

        raw_logits = logits

        self._m3_first_gather_ids = None
        self._m3_next_new = None
        self._m3_next_new_tokens = None
        self._m3_fused_first_inputs = False
        if logits is None:
            # tail_eligible() skipped the gathered target logits: verify from this rank's logits shard.
            accepted_tokens, num_accepted_tokens, self._m3_first_gather_ids = (
                self._m3_target_accept(
                    hidden_states,
                    spec_metadata,
                    batch_size,
                    runtime_draft_len,
                    draft_model,
                    kv_lens=getattr(attn_metadata, "kv_lens_cuda", None),
                )
            )
        else:
            self._execute_guided_decoder_if_present(logits)

            # Sample and accept tokens. ``input_ids`` is required by the relaxed-acceptance path (scans for
            # thinking-phase tokens); ignored otherwise.
            accepted_tokens, num_accepted_tokens = self.sample_and_accept_draft_tokens(
                input_ids, logits, attn_metadata, spec_metadata
            )
            if self.draft_only_eligible(attn_metadata, spec_metadata, draft_model):
                self._m3_seed_first_pass(num_accepted_tokens, spec_metadata, batch_size)

        # Mamba hybrid models need state updates after token acceptance (a no-op on this target's cache manager).
        if self._is_mamba_hybrid_cache is None:
            self._is_mamba_hybrid_cache = isinstance(
                attn_metadata.kv_cache_manager, MambaHybridCacheManager
            )
        if num_gens > 0 and self._is_mamba_hybrid_cache:
            attn_metadata.kv_cache_manager.update_mamba_states(
                attn_metadata=attn_metadata,
                num_accepted_tokens=num_accepted_tokens,
                state_indices=attn_metadata.mamba_metadata.state_indices,
            )

        sa_manager = getattr(spec_metadata.spec_resource_manager, "sa_manager", None)
        if self.sa_enhancer is not None and sa_manager is not None:
            self.sa_enhancer.extend_and_prepare(
                sa_manager=sa_manager,
                request_ids=spec_metadata.request_ids,
                accepted_tokens=accepted_tokens,
                num_accepted_tokens=num_accepted_tokens,
                num_gens=num_gens,
                num_contexts=num_contexts,
                max_draft_len=runtime_draft_len,
            )

        if self._uses_external_shared_target_kv:
            next_draft_tokens = self._forward_external_shared_target_kv_draft_loop(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                spec_metadata=spec_metadata,
                draft_model=draft_model,
                accepted_tokens=accepted_tokens,
                num_accepted_tokens=num_accepted_tokens,
                num_contexts=num_contexts,
                batch_size=batch_size,
            )
        else:
            # Save the old attn_metadata and spec_metadata
            self._prepare_attn_metadata_for_spec_dec(attn_metadata)

            # Prepare inputs for the 1st draft model forward
            position_ids = position_ids.squeeze(0)
            inputs = self.prepare_1st_drafter_inputs(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                accepted_tokens=accepted_tokens,
                attn_metadata=attn_metadata,
                spec_metadata=spec_metadata,
                draft_model=draft_model,
            )

            # Predict draft tokens. ``original_all_rank_num_tokens`` is saved here so the post-loop restore (below)
            # can put attn_metadata back into a state the target model expects.
            original_all_rank_num_tokens = attn_metadata.all_rank_num_tokens

            # Get the draft KV cache manager if using separate layouts
            draft_kv_cache_manager = self.get_draft_kv_cache_manager(resource_manager)

            next_draft_tokens = self._forward_draft_loop(
                inputs,
                attn_metadata,
                spec_metadata,
                draft_model,
                draft_kv_cache_manager,
                num_contexts,
                num_gens,
                batch_size,
                num_accepted_tokens,
                original_all_rank_num_tokens,
                resource_manager,
            )
            # restore attn_metadata to support cuda graph
            self._restore_attn_metadata_from_spec_dec(attn_metadata)
            # restore all_rank_num_tokens for attention DP
            if original_all_rank_num_tokens is not None:
                attn_metadata.all_rank_num_tokens = original_all_rank_num_tokens

        # prepare next new tokens to support overlap scheduler
        if self._m3_next_new_tokens is not None:
            # Filled in place by the fused verify or the seed (column 0) and the fused draft passes (columns 1..).
            next_new_tokens = self._m3_next_new_tokens
        else:
            next_new_tokens = self._prepare_next_new_tokens(
                accepted_tokens,
                next_draft_tokens,
                spec_metadata.batch_indices_cuda,
                batch_size,
                num_accepted_tokens,
            )

        attn_metadata.use_spec_decoding = True

        # Commit model-owned target state only after draft preparation has succeeded; SpecWorkerBase aborts the
        # still-pending snapshots on any exception above this point.
        if self._auxiliary_state_handlers:
            self.commit_auxiliary_speculative_states(
                num_accepted_tokens,
                attn_metadata.mamba_metadata.state_indices[:batch_size],
                num_contexts,
            )

        return {
            "logits": raw_logits,
            "new_tokens": accepted_tokens,
            "new_tokens_lens": num_accepted_tokens,
            "next_draft_tokens": next_draft_tokens,
            "next_new_tokens": next_new_tokens,
        }

    def prepare_1st_drafter_inputs(
        self,
        input_ids,
        position_ids,
        hidden_states,
        accepted_tokens,
        attn_metadata,
        spec_metadata,
        draft_model,
    ):
        """The fused first draft inputs on a step that verified with the fused kernels (or was seeded for them),
        else the in-tree ones."""
        fused = self._m3_first_drafter_inputs(
            position_ids,
            accepted_tokens,
            input_ids.shape[0],
            attn_metadata.num_contexts,
            spec_metadata,
            draft_model,
        )
        self._m3_fused_first_inputs = fused is not None
        if fused is not None:
            fused.update(attn_metadata=attn_metadata, spec_metadata=spec_metadata)
            return fused
        return super().prepare_1st_drafter_inputs(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            accepted_tokens=accepted_tokens,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=draft_model,
        )

    def _m3_structural_ok(self, attn_metadata, spec_metadata) -> bool:
        """The batch conditions shared by the fused verify and the fused first drafter inputs: a decode-only plain
        Eagle3 step without attention DP, guided decoding, relaxed acceptance or a suffix-automaton enhancer, of at
        most WIDE_VERIFY_MAX_TOKENS verify rows. Host state only, so the TP ranks agree."""
        if self.is_mtp_eagle or self.use_dynamic_tree:
            return False
        if self._uses_external_shared_target_kv or self.sa_enhancer is not None:
            return False
        if (
            self.guided_decoder is not None
            or self.mapping is None
            or self.mapping.enable_attention_dp
        ):
            return False
        if getattr(self.spec_config, "use_relaxed_acceptance_for_thinking", False):
            return False
        if attn_metadata.num_contexts != 0 or attn_metadata.num_seqs == 0:
            return False
        draft_len = spec_metadata.runtime_draft_len
        if draft_len == 0 or attn_metadata.num_seqs * (draft_len + 1) > WIDE_VERIFY_MAX_TOKENS:
            return False
        return not spec_metadata.wants_advanced_draft_sampling

    def tail_eligible(self, attn_metadata, spec_metadata, lm_head: nn.Module) -> bool:
        """Whether this step verifies with the fused kernels instead of the gathered target logits.

        Decode-only Eagle3 batches (``_m3_structural_ok``) over an unpadded vocab-sharded bias-free target head, all
        greedy at a request count ``SHARD_ACCEPT`` covers, or sampled under the temperature + top-p contract (see
        _m3_sampled_verify_ok) at a request count ``SAMPLED_ACCEPT`` covers, with nothing else reading the target
        logits. Every input is host state shared by all TP ranks, so the ranks agree.
        """
        if not self._m3_structural_ok(attn_metadata, spec_metadata):
            return False
        if getattr(spec_metadata, "enable_penalty", False):
            return False
        if not _sharded_head(lm_head) or getattr(lm_head, "bias", None) is not None:
            return False
        draft_len = spec_metadata.runtime_draft_len
        if spec_metadata.is_all_greedy_sample:
            key = (draft_len, attn_metadata.num_seqs)
            if not SHARD_ACCEPT.covers(key):
                uncovered(
                    SHARD_ACCEPT.path,
                    f"draft length {draft_len} at {attn_metadata.num_seqs} requests",
                )
                return False
        else:
            if not self._m3_sampled_verify_ok(spec_metadata):
                return False
            key = (draft_len, attn_metadata.num_seqs)
            if not SAMPLED_ACCEPT.covers(key):
                uncovered(
                    SAMPLED_ACCEPT.path,
                    f"draft length {draft_len} at {attn_metadata.num_seqs} requests",
                )
                return False
        self._m3_target_lm_head = lm_head
        _log_engaged(
            SHARD_ACCEPT.path if spec_metadata.is_all_greedy_sample else SAMPLED_ACCEPT.path,
            self.mapping,
        )
        return True

    def draft_only_eligible(self, attn_metadata, spec_metadata, draft_model) -> bool:
        """Whether a step verified by the in-tree path can still take the fused first drafter inputs: up to
        FIRST_DRAFTER_INPUTS_MAX_REQUESTS requests, the structural conditions of tail_eligible, without the
        target-sampling ones."""
        if attn_metadata.num_seqs > FIRST_DRAFTER_INPUTS_MAX_REQUESTS:
            uncovered(
                "m3_first_drafter_inputs", f"in-tree verify at {attn_metadata.num_seqs} requests"
            )
            return False
        return self._m3_structural_ok(attn_metadata, spec_metadata) and draft_model is not None

    def _m3_seed_first_pass(self, num_accepted_tokens, spec_metadata, batch_size: int) -> None:
        """The first draft pass's gather rows (the last accepted row of each request) from the in-tree acceptance, as
        _m3_target_accept leaves them."""
        k1 = spec_metadata.runtime_draft_len + 1
        n = num_accepted_tokens[:batch_size].to(torch.int64)
        rows = spec_metadata.batch_indices_cuda[:batch_size].to(torch.int64)
        self._m3_first_gather_ids = rows * k1 + n - 1

    def _m3_first_drafter_inputs(
        self,
        position_ids,
        accepted_tokens,
        num_tokens: int,
        num_contexts: int,
        spec_metadata,
        draft_model: Optional[M3Eagle3Draft],
    ):
        """The in-tree first draft inputs of a decode-only step, with the per-layer fc_norm RMSNorms of
        ``apply_eagle3_fc`` in one kernel (unless the capture already applied them) before the fc; the input ids are
        the accepted tokens as they lie. None to use the in-tree path."""
        if (
            self._m3_first_gather_ids is None
            or self.is_mtp_eagle
            or num_contexts != 0
            or draft_model is None
        ):
            return None
        layers = len(spec_metadata.layers_to_capture)
        if (
            draft_model.hidden_size_in != spec_metadata.hidden_size
            or layers != draft_model.capture_layers
        ):
            return None
        if not self._m3_first_inputs_covered(num_tokens, spec_metadata):
            return None
        cache = self._m3_fc_norm_w
        if cache is None or cache[0] is not draft_model.w["fc_norm0"]:
            if torch.cuda.is_current_stream_capturing():
                capture_missed("spec_tail", None, "the stacked Eagle3 fc_norm weights")
                return None
            cache = (
                draft_model.w["fc_norm0"],
                torch.stack(
                    [draft_model.w[f"fc_norm{i}"].detach() for i in range(layers)]
                ).contiguous(),
            )
            self._m3_fc_norm_w = cache
        hidden = self._m3_fc_hidden
        self._m3_fc_hidden = None
        if hidden is None or hidden.shape[0] != num_tokens:
            hs = spec_metadata.hidden_states[:num_tokens, : spec_metadata.hidden_size * layers]
            if self._m3_capture_norm is None:
                _record("m3_chunk_norm", num_tokens)
                hs = m3_spec_tail.chunk_norm(hs, cache[1], draft_model.rms_eps)
                _log_engaged(CHUNK_NORM.path, self.mapping)
            hidden = draft_model.fc(hs)
        input_ids = accepted_tokens[:, : spec_metadata.runtime_draft_len + 1].reshape(-1)
        _log_engaged(FUSED_FIRST_INPUTS.path, self.mapping)
        return {"input_ids": input_ids, "position_ids": position_ids, "hidden_states": hidden}

    def _m3_first_inputs_covered(self, num_tokens: int, spec_metadata) -> bool:
        """Whether the fused first drafter inputs of ``num_tokens`` verify rows are covered: their request count
        (``FUSED_FIRST_INPUTS``), and the chunk norm's rows (``CHUNK_NORM``) unless the captures already applied the
        fc_norm."""
        k1 = spec_metadata.runtime_draft_len + 1
        if num_tokens % k1 or not FUSED_FIRST_INPUTS.covers(num_tokens // k1):
            uncovered(
                FUSED_FIRST_INPUTS.path,
                f"{num_tokens} verify rows at draft length {k1 - 1} (the in-tree first drafter inputs run)",
            )
            return False
        if self._m3_capture_norm is None and not CHUNK_NORM.covers(num_tokens):
            uncovered(
                CHUNK_NORM.path, f"{num_tokens} verify rows (the in-tree first drafter inputs run)"
            )
            return False
        return True

    def _m3_fc_fork(
        self,
        spec_metadata,
        num_tokens: int,
        draft_model: Optional[M3Eagle3Draft],
        after: Optional[torch.cuda.Event] = None,
    ) -> Optional[torch.cuda.Stream]:
        """Launch the first draft pass's fc_norm + fc on a forked stream beside the target head GEMM (after the event
        ``after`` when given, else after everything queued on the current stream); the result is picked up by
        _m3_first_drafter_inputs. Returns the stream to join, or None when the fc path does not apply (the first
        drafter inputs then compute it)."""
        self._m3_fc_hidden = None
        if draft_model is None:
            return None
        layers = len(spec_metadata.layers_to_capture)
        if (
            draft_model.hidden_size_in != spec_metadata.hidden_size
            or layers != draft_model.capture_layers
            or self._m3_fc_norm_w is None
            or not self._m3_first_inputs_covered(num_tokens, spec_metadata)
        ):
            return None
        aux = self._m3_aux_stream
        if aux is None:
            aux = self._m3_aux_stream = torch.cuda.Stream()
        if after is not None:
            aux.wait_event(after)
        else:
            aux.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(aux):
            hs = spec_metadata.hidden_states[:num_tokens, : spec_metadata.hidden_size * layers]
            if self._m3_capture_norm is None:
                _record("m3_chunk_norm", num_tokens)
                hs = m3_spec_tail.chunk_norm(
                    hs, self._m3_fc_norm_w[1], draft_model.rms_eps, use_pdl=False
                )
                _log_engaged(CHUNK_NORM.path, self.mapping)
            self._m3_fc_hidden = draft_model.fc(hs)
        return aux

    def _m3_fc_join(self, fork: torch.cuda.Stream) -> None:
        """Join the fc fork onto the current stream and keep its output's memory alive for that stream.

        _m3_fc_hidden is allocated in fork (aux) stream order and read on the current stream. Without the record,
        dropping it returns its block to the aux stream's pool, where later aux work can take it while the current
        stream's reads are still queued (eager warm-up / mixed steps). Nothing is recorded under graph capture, as in
        maybe_execute_in_parallel: a captured graph relies on the next fork's aux.wait_stream(main) to order the
        reuse."""
        main = torch.cuda.current_stream()
        main.wait_stream(fork)
        hidden = self._m3_fc_hidden
        if (
            hidden is not None
            and hidden.is_cuda
            and not (torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing())
        ):
            hidden.record_stream(main)

    def arm_capture_norm(
        self, attn_metadata, spec_metadata, lm_head: nn.Module, draft_model: Optional[M3Eagle3Draft]
    ) -> None:
        """Before every target forward: whether this step's Eagle3 captures apply the drafter's fc_norm themselves.

        On a fused-verify step of ``CAPTURE_NORM``'s rows whose first drafter inputs are the fused ones and whose
        fc_norm weights are already stacked (by an earlier fused first pass), the capture writes
        rmsnorm(h + r) * fc_norm[i] (one kernel per captured layer instead of add + copy, or the layer's boundary
        all-reduce itself, ``capture_target``) and the fused first drafter inputs read it without their own
        chunk_norm. A fused-verify step of at most ``FC_EARLY_MAX_ROWS`` verify rows is also armed to fork the
        drafter's first fc at the last captured layer (``capture_hidden_states``).
        """
        self._m3_capture_norm = None
        self._m3_fc_early_drafter = None
        self._m3_fc_early_stream = None
        if draft_model is None:
            return
        if not self.tail_eligible(attn_metadata, spec_metadata, lm_head):
            return
        rows = attn_metadata.num_seqs * (spec_metadata.runtime_draft_len + 1)
        # The fork reads every captured column at the last captured layer, so a final-state capture, which is written
        # after the model returns, keeps the fc at its stock place.
        if (
            rows <= FC_EARLY_MAX_ROWS
            and FINAL_HIDDEN_LAYER_ID not in spec_metadata.layers_to_capture
        ):
            self._m3_fc_early_drafter = draft_model
        if not CAPTURE_NORM.covers(rows):
            uncovered(CAPTURE_NORM.path, f"{rows} verify rows (the in-tree capture runs)")
            return
        # The normed captures are read only by the fused first drafter inputs; the in-tree ones norm again.
        if not FUSED_FIRST_INPUTS.covers(attn_metadata.num_seqs):
            return
        fc_norm_w = self._m3_fc_norm_w
        if (
            fc_norm_w is not None
            and fc_norm_w[0] is draft_model.w["fc_norm0"]
            and len(spec_metadata.layers_to_capture) == fc_norm_w[1].shape[0]
        ):
            self._m3_capture_norm = (fc_norm_w[1], draft_model.rms_eps)
            _log_engaged(CAPTURE_NORM.path, self.mapping)

    @staticmethod
    def _m3_sampled_verify_ok(spec_metadata) -> bool:
        """Whether sampled verify rows can use the fused sampled op: only under advanced_sampling_mode=no_topk, the
        deploy-time guarantee that no row uses top-k or min-p (that op applies neither)."""
        mode = getattr(spec_metadata, "advanced_sampling_mode", None)
        return mode is not None and mode.skips_top_k and not mode.skips_top_p

    def _m3_target_accept(
        self,
        hidden_states,
        spec_metadata,
        batch_size: int,
        draft_len: int,
        draft_model: Optional[M3Eagle3Draft] = None,
        kv_lens: Optional[torch.Tensor] = None,
    ):
        """Fused verification (see tail_eligible); same outputs as the strict acceptance path. ``kv_lens``: the
        requests' KV lengths at this verify (the per-request forced-acceptance key, FORCE_AL_PER_REQUEST)."""
        force_total = 0
        force_vec = None
        if self.force_num_accepted_tokens != 0.0:
            is_warmup = spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing()
            if not torch.cuda.is_current_stream_capturing():
                # The draw pool is built by an H2D copy and the draw kernels JIT-compile per block width, so both
                # happen here, before the first captured step (a scratch counter keeps the real draw sequence).
                self._ensure_force_accept_rng_state(hidden_states.device)
                if not self._m3_force_vec_warm:
                    scratch = torch.zeros(1, dtype=torch.int64, device=hidden_states.device)
                    dummy = torch.empty(64, dtype=torch.int32, device=hidden_states.device)
                    for block in (16, 32, 64):
                        m3_force_vec(
                            self._force_accept_rng_pool,
                            scratch,
                            dummy,
                            block,
                            1,
                            2,
                            0.5,
                            _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
                            _FORCE_ACCEPT_RNG_SLOT_STRIDE,
                            _FORCE_ACCEPT_RNG_POOL_SIZE - 1,
                            block,
                        )
                        m3_force_vec_req(
                            self._force_accept_rng_pool,
                            dummy,
                            dummy,
                            block,
                            1,
                            2,
                            0.5,
                            _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
                            _FORCE_ACCEPT_RNG_POOL_SIZE - 1,
                            block,
                        )
                    self._m3_force_vec_warm = True
            if not is_warmup:
                force_total = min(int(self.force_num_accepted_tokens) + 1, draft_len + 1)
                force_vec = self._m3_force_vec(
                    batch_size, force_total, draft_len + 1, hidden_states.device, kv_lens
                )
        draft_tokens = spec_metadata.draft_tokens.view(batch_size, draft_len)
        if draft_tokens.dtype != torch.int32:
            draft_tokens = draft_tokens.to(torch.int32)
        # Column 0 receives the last accepted token, the next step's first input (the fused draft loop adds the rest).
        self._m3_next_new = torch.empty(
            (batch_size, draft_len + 1), dtype=torch.int32, device=hidden_states.device
        )
        rows = hidden_states[: batch_size * (draft_len + 1)]
        lm_head = self._m3_target_lm_head
        batch_idx = spec_metadata.batch_indices_cuda[:batch_size]
        # The Eagle3 fc on the aux stream: the fork made at the last captured layer, else one beside the head. A
        # sampled verify's head (the in-tree GEMM up to FC_AFTER_HEAD_MAX_ROWS verify rows) holds
        # every SM while it streams: its fc is forked after it instead (overlapping the sampled acceptance), where its
        # many split-K CTAs do not delay the head's start. Rubin TP2 c1: the head started 17.4 us after its input was
        # ready with the fc beside it.
        fork = self._m3_fc_early_stream
        self._m3_fc_early_stream = None
        head_first = (
            fork is None
            and rows.shape[0] <= FC_AFTER_HEAD_MAX_ROWS
            and not spec_metadata.is_all_greedy_sample
        )
        if fork is None and not head_first:
            fork = self._m3_fc_fork(spec_metadata, rows.shape[0], draft_model)
        # This rank's vocabulary shard of the target logits: the head's own local GEMM (or its per-die split), whose
        # all-gather the fused kernels replace (they exchange per-row records across ranks, not logits).
        ugpu = self._m3_ugpu_head
        if ugpu is not None and UGPU_HEAD.covers(rows.shape[0]):
            if m3_ugpu_head_op.current(ugpu, lm_head.weight):
                taken(UGPU_HEAD.path, rows.shape[0])
                _record("m3_ugpu_head", rows.shape[0])
                logits = m3_ugpu_head_op.m3_ugpu_head(rows, ugpu)
            else:
                missed(
                    UGPU_HEAD.path,
                    rows.shape[0],
                    "the placed head copy is not of the head's current weight (prebuild)",
                )
                logits = lm_head.apply_linear(rows, None)
        else:
            logits = lm_head.apply_linear(rows, None)
        head_done = None
        if head_first:
            head_done = torch.cuda.Event()
            head_done.record()
        vocab_offset = self.mapping.tp_rank * lm_head.weight.shape[0]
        if spec_metadata.is_all_greedy_sample:
            result = m3_shard_accept_kernel.shard_accept(
                logits,
                vocab_offset,
                self.mapping,
                draft_tokens,
                batch_idx,
                force_total,
                self._m3_next_new,
                force_vec=force_vec,
            )
            path = SHARD_ACCEPT.path
        else:
            n_rows = rows.shape[0]
            seeds, offsets = self._rng_state_per_token(spec_metadata, n_rows)
            result = m3_sampled_accept_kernel.sampled_accept(
                logits,
                vocab_offset,
                self.mapping,
                draft_tokens,
                batch_idx,
                spec_metadata.temperatures[:n_rows],
                spec_metadata.top_ks[:n_rows],
                spec_metadata.top_ps[:n_rows],
                None,
                seeds,
                offsets,
                force_total,
                self._m3_next_new,
                force_vec=force_vec,
            )
            path = "spec_tail"
            if head_done is not None:
                # Enqueued after the sampled acceptance (which starts first) but waits on the head only.
                fork = self._m3_fc_fork(spec_metadata, rows.shape[0], draft_model, after=head_done)
        if fork is not None:
            self._m3_fc_join(fork)
        if result is None:
            if torch.cuda.is_current_stream_capturing():
                capture_missed(path, None, "the fused verify's exchange workspace")
            raise RuntimeError(
                "M3 fused verify unavailable after tail_eligible() accepted the step"
            )
        return result

    def prebuild_ugpu_head(self, lm_head: nn.Module) -> bool:
        """Build the die-placed copy of the target head shard for the uGPU verify head (``UGPU_HEAD``) before the
        executor's warmup, on every rank (the module's weight is not changed): a verify of those rows computes the
        logits shard as two half-vocabulary in-tree GEMMs, each on the green context of the die whose memory holds
        that half, bit for bit the in-tree head's logits. Rubin TP2 head shard: 93.3 -> 79.5 us at 4 rows, 93.5 ->
        80.6 / 80.7 at 12 / 16; in-model the decode step is ~45 us shorter at one request and ~90 us at two. False
        (nothing changed) when the head or device does not qualify (see m3_ugpu_head.prepare_head)."""
        self._m3_ugpu_head = None
        if not UGPU_HEAD.keys or not _sharded_head(lm_head):
            return False
        self._m3_ugpu_head = m3_ugpu_head_op.prepare_head(lm_head)
        return self._m3_ugpu_head is not None

    def _m3_force_vec(
        self,
        batch_size: int,
        base_total: int,
        max_total: int,
        device,
        kv_lens: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Per-request forced accepted counts for a fractional forced acceptance, else None.

        With ``kv_lens`` (FORCE_AL_PER_REQUEST): each request draws from the fixed-seed pool at its own key, its KV
        length at this verify (its prompt plus what it has accepted so far), so its draws do not depend on the
        server's history or on the other requests; every TP rank holds the same lengths, so the ranks agree. Without
        lengths: exactly as _apply_force_accepted_tokens draws for a decode-only batch (one shared counter bump per
        step, the (counter, slot) hash), the generic path's sequence.
        """
        frac = self.force_num_accepted_tokens - int(self.force_num_accepted_tokens)
        if frac <= 0.0 or base_total >= max_total:
            return None
        self._ensure_force_accept_rng_state(device)
        out = torch.empty(batch_size, dtype=torch.int32, device=device)
        block = max(16, 1 << (batch_size - 1).bit_length())
        if FORCE_AL_PER_REQUEST and kv_lens is not None:
            keys = kv_lens[:batch_size]
            if keys.dtype != torch.int32:
                keys = keys.to(torch.int32)
            return m3_force_vec_req(
                self._force_accept_rng_pool,
                keys,
                out,
                batch_size,
                base_total,
                max_total,
                float(frac),
                _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
                _FORCE_ACCEPT_RNG_POOL_SIZE - 1,
                block,
            )
        # One launch: bump the shared counter and draw, with _apply_force_accepted_tokens' exact arithmetic.
        return m3_force_vec(
            self._force_accept_rng_pool,
            self._force_accept_rng_counter,
            out,
            batch_size,
            base_total,
            max_total,
            float(frac),
            _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
            _FORCE_ACCEPT_RNG_SLOT_STRIDE,
            _FORCE_ACCEPT_RNG_POOL_SIZE - 1,
            block,
        )

    # ---------------------------------------------------------------------------------------------- fused draft loop

    def prebuild_draft_head(self, draft_model: Optional[M3Eagle3Draft]) -> bool:
        """Build the copy of the drafter's BF16 head shard that the fused draft loop streams (rebuilt only when the
        head weight tensor or its version changed; a replaced copy is kept alive), on every rank. False when the
        drafter is not one the loop serves."""
        old = self._m3_draft_head
        if draft_model is None or not self._m3_drafter_supported(draft_model):
            self._m3_draft_head = None
            return False
        weight = draft_model.lm_head.weight
        # Idempotent: the same weight tensor at the same version keeps the built copy, so a repeated
        # cache_derived_state never re-places it under a captured CUDA graph.
        if (
            old is not None
            and old.weight is weight
            and self._m3_draft_head_version == weight._version
        ):
            return True
        if old is not None:
            # A rebuilt copy (new weight values): keep the old one alive, since a graph captured earlier may still
            # read its (placed) memory until the graphs are recaptured.
            self._m3_draft_head_graveyard.append(old)
        self._m3_draft_head = m3_draft_step_kernel.prepare_head(
            draft_model.lm_head, self.mapping.tp_rank
        )
        self._m3_draft_head_version = weight._version
        return True

    def _m3_drafter_supported(self, draft_model: M3Eagle3Draft) -> bool:
        """The drafter the fused draft loop runs (``M3Eagle3Draft`` validates its shape at construction) over this
        worker's tensor-parallel group with a bias-free vocab-sharded BF16 head without padding."""
        if self.mapping is None or self.mapping.tp_size <= 1 or self.mapping.enable_attention_dp:
            return False
        if self._d2t is not None:
            return False
        lm_head = draft_model.lm_head
        return _sharded_head(lm_head) and m3_draft_step_kernel.head_supported(lm_head)

    def _m3_draft_ar_norm(
        self,
        draft_model: M3Eagle3Draft,
        x: torch.Tensor,
        residual: torch.Tensor,
        norm_key: str,
        pushed: bool = False,
    ):
        """One drafter boundary of the fused draft loop: the multicast all-reduce + residual + RMSNorm with the
        drafter's stock rounding (``DRAFT_MC_AR``: opts bit 16, normalizing the unrounded FP32 sum as FlashInfer
        fused_add_rmsnorm does; Rubin TP2 10.0..11.0 -> 4.6..5.3 us per boundary), else the in-tree structure
        (``M3Eagle3Draft.ar_norm``). ``pushed``: the boundary's projection already pushed its rows into the exchange
        (prepushed 2). The choice depends on shapes and a collectively created workspace only, so the TP ranks
        agree. Returns (normed, residual sum)."""
        rows = x.shape[0]
        path = DRAFT_MC_AR.path
        if DRAFT_MC_AR.covers(rows):
            workspace = m3_mc_allreduce.get_workspace(self.mapping, x.shape[-1])
            out = None
            if workspace is not None:
                _record("m3_draft_mc_ar", rows, pushed)
                out = m3_mc_allreduce.m3_mc_allreduce_norm(
                    x.contiguous(),
                    residual.contiguous(),
                    draft_model.w[norm_key],
                    draft_model.rms_eps,
                    workspace,
                    use_pdl=get_env_enable_pdl(),
                    drafter_norm=True,
                    prepushed=2 if pushed else 0,
                )
            if out is not None:
                _log_engaged(path, self.mapping)
                if pushed:
                    taken(DRAFT_AR_PUSH_PATH, rows, "the projection pushed its rows (prepushed 2)")
                elif DRAFT_AR_PUSH:
                    uncovered(
                        DRAFT_AR_PUSH_PATH,
                        "the boundary's projection did not push (the AR pushes x)",
                    )
                return out
            if pushed:
                raise RuntimeError(
                    "the drafter's rows were pushed into a boundary all-reduce that did not run"
                )
            missed(path, rows, "the multicast all-reduce workspace or operand layout")
        return draft_model.ar_norm(x, residual, norm_key)

    def _m3_arm_draft_push(self, draft_model: M3Eagle3Draft, key: str, rows: int) -> None:
        """DRAFT_AR_PUSH: let projection ``key``'s next low-M call push its rows into the boundary all-reduce that
        ``_m3_draft_ar_norm`` then runs (the multicast path only; shapes and a collectively created workspace decide
        it, so the TP ranks agree)."""
        if not (DRAFT_AR_PUSH and DRAFT_MC_AR.covers(rows)):
            return
        workspace = m3_mc_allreduce.get_workspace(self.mapping, draft_model.w[key].shape[0])
        if workspace is not None:
            draft_model.arm_push(key, workspace, rows)

    def _m3_fused_loop_ok(
        self, attn_metadata, spec_metadata, batch_size: int, draft_model: Optional[M3Eagle3Draft]
    ) -> bool:
        """Whether this step drafts with the fused loop: a decode-only step the fused verify or the seed prepared
        (``_m3_first_gather_ids``) with the fused first draft inputs, a supported drafter, a (draft length, requests)
        ``DRAFT_STEP`` covers, the prebuilt head and the exchange workspace. Host state only, so the TP ranks
        agree."""
        if (
            self._m3_first_gather_ids is None
            or not self._m3_fused_first_inputs
            or draft_model is None
        ):
            return False
        path = DRAFT_STEP.path
        if not self._m3_drafter_supported(draft_model) or not hasattr(
            attn_metadata, "kv_lens_cuda"
        ):
            uncovered(
                path, "the drafter or its attention metadata is not one the fused loop serves"
            )
            return False
        draft_len = spec_metadata.runtime_draft_len
        if not DRAFT_STEP.covers((draft_len, batch_size)):
            uncovered(path, f"draft length {draft_len} at {batch_size} requests")
            return False
        head = self._m3_draft_head
        if head is None or head.weight is not draft_model.lm_head.weight:
            missed(
                path,
                batch_size,
                "the drafter head's streamed copy is not built for its current weight (prebuild)",
            )
            return False
        if m3_draft_step_kernel.workspace(self.mapping, head.wf.device) is None:
            capture_missed(path, None, "the draft-step exchange workspace")
            return False
        _log_engaged(path, self.mapping)
        return True

    def _forward_draft_loop(
        self,
        inputs,
        attn_metadata,
        spec_metadata,
        draft_model,
        draft_kv_cache_manager,
        num_contexts,
        num_gens,
        batch_size,
        num_accepted_tokens,
        original_all_rank_num_tokens,
        resource_manager,
    ):
        """The fused draft loop on a step it covers (``_m3_fused_loop_ok``), else the in-tree loop (W2b: drafting from
        the head's local shard where ``_m3_wide_draft_logits_ok``)."""
        if self._m3_fused_loop_ok(attn_metadata, spec_metadata, batch_size, draft_model):
            return self._m3_fused_draft_loop(
                inputs,
                attn_metadata,
                spec_metadata,
                draft_model,
                draft_kv_cache_manager,
                batch_size,
                num_accepted_tokens,
            )
        # W2b: on a step the fused loop declines, the drafter head returns this rank's BF16 vocab shard (not the
        # gathered fp32 logits), so the greedy top-1 runs the fused shard argmax (greedy_sample_draft_with_tp_gather).
        swap = self._m3_wide_draft_logits_ok(spec_metadata, draft_model)
        original = draft_model.logits_processor if swap else None
        if swap:
            draft_model.logits_processor = _M3ShardDraftLogits(original)
        try:
            return super()._forward_draft_loop(
                inputs,
                attn_metadata,
                spec_metadata,
                draft_model,
                draft_kv_cache_manager,
                num_contexts,
                num_gens,
                batch_size,
                num_accepted_tokens,
                original_all_rank_num_tokens,
                resource_manager,
            )
        finally:
            if swap:
                draft_model.logits_processor = original

    def _m3_wide_draft_logits_ok(self, spec_metadata, draft_model: Optional[nn.Module]) -> bool:
        """Whether the in-tree draft loop may draft from the head's local shard (W2b): ``WIDE_SPEC_TAIL``, an Eagle3
        drafter with an unpadded vocab-sharded head under plain TP, argmax drafting (no rejection sampling) and no
        guided decoder (which reads full logits). Host state only, so the TP ranks agree."""
        if not WIDE_SPEC_TAIL or draft_model is None or self.is_mtp_eagle:
            return False
        if self.guided_decoder is not None or spec_metadata.wants_advanced_draft_sampling:
            return False
        if self.mapping is None or self.mapping.enable_attention_dp:
            return False
        lm_head = getattr(draft_model, "lm_head", None)
        return (
            lm_head is not None
            and getattr(draft_model, "logits_processor", None) is not None
            and _sharded_head(lm_head)
            and getattr(lm_head, "bias", None) is None
        )

    def greedy_sample_draft_with_tp_gather(
        self, logits: torch.Tensor, spec_metadata=None, mapping_lm_head_tp=None
    ) -> torch.Tensor:
        """W2b (``WIDE_SPEC_TAIL``): the in-tree draft loop's greedy top-1 over this rank's vocab shard of the drafter
        logits, from the fused shard argmax + cross-rank key exchange (ops/m3_shard_accept at draft length 0, at
        most 48 rows per launch) instead of the per-rank max, the all-gather of (index, value) pairs and the pick. The
        same tokens (keys order BF16 values exactly; ties take the lowest global index, as the pick over ranks and the
        first-index max within a shard). Plain TP only (no LM-head TP in ADP); else, or when the logits are not a
        unit-stride BF16 shard, or before the exchange workspace exists during a capture, the in-tree sampler."""
        tokens = None
        if (
            WIDE_SPEC_TAIL
            and mapping_lm_head_tp is None
            and logits.dim() == 2
            and logits.dtype == torch.bfloat16
            and logits.stride(-1) == 1
            and self._draft_logits_are_sharded(logits, spec_metadata)
        ):
            m = logits.shape[0]
            empty = torch.empty((m, 0), dtype=torch.int32, device=logits.device)
            out = m3_shard_accept_kernel.shard_accept(
                logits,
                self.mapping.tp_rank * logits.shape[-1],
                self.mapping,
                empty,
                m3_shard_accept_kernel._local_rows(m, logits.device),
                0,
            )
            if out is not None:
                tokens = out[0][:, 0]
                _log_engaged("m3_draft_shard_top1", self.mapping)
        if tokens is None:
            return super().greedy_sample_draft_with_tp_gather(
                logits, spec_metadata, mapping_lm_head_tp
            )
        return tokens

    def _m3_fused_draft_loop(
        self,
        inputs,
        attn_metadata,
        spec_metadata,
        draft_model: M3Eagle3Draft,
        draft_kv_cache_manager,
        batch_size: int,
        num_accepted_tokens,
    ):
        """Mirrors Eagle3OneModelWorker._forward_linear_draft_loop for a decode-only Eagle3 step, with each pass's
        glue fused (see ``_m3_fused_loop_ok``):

        * the draft input -- the embedding lookup, the layer's input norm and hidden norm and their concat -- in one
          launch (``m3_spec_tail.draft_input``; the second pass gathers the last accepted rows there);
        * the drafter layer's attention and MLP (its SwiGLU on this target's C++ op), each followed by the all-reduce
          with the residual add and the next RMSNorm (the post-attention norm, then the drafter's final norm): the
          multicast boundary fed by the o_proj / down projection's push where covered (``_m3_draft_ar_norm``);
        * one kernel for the BF16 head over the head rows (pass 0: the last accepted row of each request), the
          cross-rank argmax and the bookkeeping the in-tree loop does between passes: the draft token, the next pass's
          input id and position, the draft KV lengths (and pass 0's sequence lengths) (``m3_draft_step``).

        The host metadata transitions are the in-tree loop's. The drafts land in columns 1.. of the step's next-input
        tokens (column 0 is the last accepted token), which ``_forward_impl`` returns as they lie.
        """
        draft_len = spec_metadata.runtime_draft_len
        table = draft_model.embed_table
        dev = table.device
        head = self._m3_draft_head
        gather_ids = self._m3_first_gather_ids
        ids = inputs["input_ids"]
        hidden = inputs["hidden_states"]
        pos = inputs["position_ids"]
        if ids.dtype != torch.int32:
            ids = ids.to(torch.int32)
        if pos.dtype != torch.int32:
            pos = pos.to(torch.int32)
        num_acc = num_accepted_tokens[:batch_size]
        if num_acc.dtype != torch.int32:
            num_acc = num_acc.to(torch.int32)
        next_new = self._m3_next_new
        if next_new is None:
            # Seeded from the in-tree acceptance: the last accepted token is the first-pass input at the gather row.
            next_new = torch.empty((batch_size, draft_len + 1), dtype=torch.int32, device=dev)
            next_new[:, 0] = ids[gather_ids]
        draft_tokens = next_new[:, 1:]
        w = draft_model.w
        eps = draft_model.rms_eps

        with self.draft_kv_cache_context(attn_metadata, draft_kv_cache_manager) as md:
            for i in range(draft_len):
                x, residual = m3_spec_tail.draft_input(
                    ids,
                    table,
                    hidden,
                    gather_ids if i == 1 else None,
                    w["in_norm"],
                    eps,
                    w["hidden_norm"],
                    eps,
                )
                rows = x.shape[0]
                self._m3_arm_draft_push(draft_model, "o", rows)
                h = draft_model.attention(x, md, reduce=False)
                h, residual = self._m3_draft_ar_norm(
                    draft_model, h, residual, "post_norm", pushed=draft_model.take_pushed(h)
                )
                self._m3_arm_draft_push(draft_model, "dn", rows)
                h = draft_model.mlp(h, reduce=False, cpp_silu=True)
                h, to_save = self._m3_draft_ar_norm(
                    draft_model, h, residual, "final_norm", pushed=draft_model.take_pushed(h)
                )
                if draft_model.return_hidden_post_norm:
                    to_save = h
                ids_next = torch.empty(batch_size, dtype=torch.int32, device=dev)
                pos_next = torch.empty(batch_size, dtype=torch.int32, device=dev)
                if not m3_draft_step_kernel.draft_step(
                    head,
                    h,
                    batch_size,
                    gather_ids if i == 0 else None,
                    self.mapping,
                    draft_tokens,
                    i,
                    ids_next,
                    pos,
                    pos_next,
                    md.kv_lens_cuda,
                    num_acc if i == 0 else None,
                    md._seq_lens_cuda if i == 0 else None,
                    draft_len,
                ):
                    raise RuntimeError(
                        "M3 fused draft step unavailable after _m3_fused_loop_ok() accepted the step"
                    )
                ids, pos, hidden = ids_next, pos_next, to_save
                if i == 0:
                    # The draft step set the device sequence lengths to 1 and rolled the KV lengths back.
                    md._seq_lens[:batch_size].fill_(1)
                    md.on_update()
                    has_kv_cache = md.kv_cache_manager is not None
                    if has_kv_cache and hasattr(md, "host_request_types"):
                        md.host_request_types[: md.num_contexts].fill_(1)
                    if has_kv_cache:
                        md.num_contexts = 0
                        self._prepare_flash_mla_generation_layout(md, 0, batch_size)
                    md.update_for_spec_dec()
                    md.use_spec_decoding = False
                else:
                    md.update_for_spec_dec()
        self._m3_next_new_tokens = next_new
        return draft_tokens


def _build_speculation(shell, model_config: ModelConfig):
    """The drafter config, the drafter and the worker of one-model Eagle3 (the in-tree one-engine construction for
    the llama3 drafter architecture)."""
    spec = model_config.spec_config
    if (
        not spec.spec_dec_mode.use_one_engine()
        or not spec.spec_dec_mode.is_eagle3_one_model()
        or spec.eagle3_model_arch != "llama3"
        or getattr(spec, "use_dynamic_tree", False)
    ):
        raise ValueError(
            "this MiniMax-M3 target speculates with linear one-model Eagle3 (llama3 drafter) only"
        )
    draft_config = ModelConfig.from_pretrained(
        spec.speculative_model,
        trust_remote_code=True,
        attn_backend=model_config.attn_backend,
        moe_backend=_get_requested_draft_moe_backend(model_config, spec),
        mapping=model_config.mapping,
        spec_config=model_config.spec_config,
        max_num_tokens=model_config.max_num_tokens,
        moe_max_num_tokens=model_config.moe_max_num_tokens,
    )
    _set_draft_kv_cache_quant_algo(draft_config, model_config)
    draft_config.extra_attrs = model_config.extra_attrs
    use_separate = should_use_separate_draft_kv_cache(spec)
    draft = M3Eagle3Draft(
        shell.model,
        draft_config,
        model_config.pretrained_config.num_hidden_layers,
        spec.num_capture_layers,
    )
    worker = M3Eagle3Worker(spec, model_config.mapping, use_separate_draft_kv_cache=use_separate)
    worker.set_draft_model(draft)
    shell.epilogue.append(draft)
    shell.epilogue.append(worker)
    return draft_config, draft, worker, use_separate


# --------------------------------------------------------------------------------------------------------------------
# Runtime hooks and the shell
# --------------------------------------------------------------------------------------------------------------------


def _runtime_hooks(spec_fold: Optional[M3SpecStepFold]) -> RuntimeHooks:
    """The step-loop hooks this target opts into.

    * ``pack_step_inputs``: a decode step's small input copies go out as one upload and one scatter at the graph
      replay instead of a chain of copies the next graph waits on; the scatter is parallel (a program per copy), issued
      by the decode graph's own prologue (this target's one-launch C++ op, ``m3_step_prologue``) from a window copied
      on a side stream while the previous graph runs, so no operation is left between the step graphs and the replay
      does not start with a host copy (its copy-engine -> compute turnaround was 10.3 us at c1). Sampled speculative
      batches stage their Philox seed/offset uploads, and the accepted-draft counts, into the same pack; the sampler's
      slot ids stay in a device buffer re-uploaded only when they change.
    * ``skip_unchanged_h2d``: steady decode re-uploads slot indices, query lengths and page tables that rarely change.
    * ``kv_lens_update_once``: a speculative overlap step re-derives the MSA KV-length metadata once, after the
      forward's KV-length correction (the first pass is rewritten in full by the second).
    * ``spec_offsets_op``: the overlap scheduler's speculative step correction (position_ids += offsets, kv_lens +=
      offsets) as this target's one plain launch (``m3_spec_offsets``) instead of two ATen adds at the head of the
      decode graph: the same int32 adds.
    * ``loop_on_execution_stream``: the overlap loop's own stream work and the forward on one stream, so step N+1's
      inputs wait on graph N in program order rather than through two cross-stream event hops.
    * ``merge_tuned_tactics_across_ranks``: tensor-parallel peers capture the same GEMM tactics (rank-local profiling
      noise otherwise splits them, and the slowest rank sets every step).
    * ``skip_greedy_rng_upload``: all-greedy speculative batches never read the Philox seeds.
    * ``spec_step_fold``: the spec sampler's slot stores and the engine's next-step input gathers; the decode forward
      stores its outputs and stages the next step's inputs at its own tail (``M3SpecStepFold``).
    * ``sampler_d2h_side_stream``: the spec sampler copies its per-slot store to the host on a side stream, overlapping
      the next step's input staging.
    * ``iteration_timing_one_event``: the iteration log times the device step with one event per boundary on a side
      stream (the reported step times are the same).
    * MSA: identity selections for short sequences (and short/long graph families), the one-launch KV-length update,
      unchanged block-table copies skipped, pool views built once, block-id tables copied from the page-index views
      (the stock builder makes a Python list of the whole padded row: on the Vera CPU 282 us per call at one request
      and 561 us at two, against 8 and 13 us).
    """
    return RuntimeHooks(
        first_content_emit_iters=FIRST_CONTENT_EMIT_ITERS,
        mxfp8_pinned_tactics=True,
        late_compile_detect=True,
        pack_step_inputs=True,
        pack_parallel_scatter=True,
        pack_in_graph=True,
        pack_early_copy=True,
        pack_prologue_op=m3_step_prologue,
        skip_unchanged_h2d=True,
        kv_lens_update_once=True,
        spec_offsets_op=m3_spec_offsets,
        loop_on_execution_stream=True,
        merge_tuned_tactics_across_ranks=True,
        skip_greedy_rng_upload=True,
        pack_spec_rng_uploads=True,
        spec_small_h2d=True,
        spec_step_fold=spec_fold,
        sampler_d2h_side_stream=True,
        iteration_timing_one_event=True,
        msa=MsaRuntimePolicy(
            short_sequence_identity=True,
            kv_update=_msa_kv_update_kernel.msa_kv_update,
            skip_unchanged_block_offsets=True,
            reuse_pool_views=True,
            block_ids_from_views=True,
        ),
    )


class ModelingV2MiniMaxM3Nvfp4Sm107Tp2(
    DecoderModelForCausalLM[MiniMaxM3Tp2ModelingV2, PretrainedConfig]
):
    """MiniMax-M3 NVFP4 / sm_107 / tp2, text only, with one-model Eagle3 or no speculation.

    The shell owns the vocab-parallel BF16 ``lm_head`` and the logits processor (``DecoderModelForCausalLM``); with
    speculation, the drafter (``M3Eagle3Draft``, its weights from the speculative checkpoint through
    ``load_draft_weights``) and the worker (``M3Eagle3Worker``) that verifies and drafts each step.
    """

    def __init__(self, model_config: ModelConfig[PretrainedConfig]) -> None:
        assert torch.cuda.get_device_capability() == _SM, (
            f"this target is built for sm_{_SM[0]}{_SM[1]}, the device is sm_"
            f"{''.join(map(str, torch.cuda.get_device_capability()))}"
        )
        # Under modeling_v2='require' a covered decode shape that cannot take its fast path raises.
        set_require(resolved_mode(type(self)) is ModelingV2Mode.REQUIRE)
        model_config = _text_model_config(model_config)
        cfg = model_config.pretrained_config
        spec_config = getattr(model_config, "spec_config", None)
        if model_config.quant_config.kv_cache_quant_algo == QuantAlgo.NVFP4:
            # The one-model Eagle3 layer has no NVFP4 trtllm-gen decode kernel: keep its modules and the shared
            # draft cache on FP8 while the target stays NVFP4.
            model_config.extra_attrs["draft_kv_cache_quant_algo_override"] = QuantAlgo.FP8
        super().__init__(
            MiniMaxM3Tp2ModelingV2(model_config),
            config=model_config,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )
        self.spec_config = spec_config
        self.draft_model: Optional[M3Eagle3Draft] = None
        self.draft_config: Optional[ModelConfig] = None
        self.spec_worker: Optional[M3Eagle3Worker] = None
        self.use_separate_draft_kv_cache = False
        self.layer_idx = FINAL_HIDDEN_LAYER_ID
        self._spec_fold: Optional[M3SpecStepFold] = None
        if spec_config is not None:
            (
                self.draft_config,
                self.draft_model,
                self.spec_worker,
                self.use_separate_draft_kv_cache,
            ) = _build_speculation(self, model_config)
            self._spec_fold = M3SpecStepFold()
        self.runtime_hooks = _runtime_hooks(self._spec_fold)
        if mpi_rank() == 0:
            logger.info(
                f"[M3 sm_107 tp2] routing: modeling_v2 target __tp2, decode rows 1..{WIDE_MAX_TOKENS} (decode layer "
                f"1..{DECODE_MAX_TOKENS}; fast-path cells up to 128 rows; 129..{WIDE_MAX_TOKENS} take the stock-exact "
                "paths)"
            )

    def load_weights(self, weights: Dict, *args, **kwargs) -> None:
        _weights.MODEL_WEIGHTS.load(self, weights)

    def load_draft_weights(self, weights: Dict, weight_mapper=None) -> None:
        """The drafter's checkpoint (``speculative_model``), loaded by the engine after the target's."""
        _weights.DRAFT_WEIGHTS.load(self.draft_model, weights)

    def set_guided_decoder(self, guided_decoder) -> bool:
        """Mirrors ``SpecDecOneEngineForCausalLM.set_guided_decoder``: guided decoding with speculation is installed
        on the worker."""
        if hasattr(self.spec_worker, "set_guided_decoder"):
            return self.spec_worker.set_guided_decoder(guided_decoder)
        return False

    def cache_derived_state(self) -> None:
        """Build the targets and every decode kernel once the weights are real, before the executor's warmup; with
        speculation also the drafter's full-context decode, the fused verify's coded target head, the die-placed
        target head copy, the fused draft loop's head copy and the drafter linears' low-M kernels."""
        super().cache_derived_state()
        self.model.post_load_weights()
        if self.draft_model is not None and DRAFTER_FULLCTX.keys:
            path = DRAFTER_FULLCTX.path
            recipes = {k: FULLCTX_DECODE_RECIPES[k] for k in DRAFTER_FULLCTX.keys}
            installed = m3_drafter_attn.install(
                self.draft_model,
                recipes,
                FULLCTX_DECODE_MAX_KV_LEN,
                FULLCTX_DECODE_OWN_REDUCER and m3_fullctx_decode_kernel.has_own_reducer(),
                get_env_enable_pdl(),
                fused_rope_mode=(DRAFTER_ROPE_MODE if DRAFTER_ROPE_FUSED else None),
                on_rope_taken=lambda rows, what: taken(DRAFTER_ROPE_PATH, rows, what),
                on_taken=lambda rows: taken(path, rows),
                on_uncovered=lambda reason: uncovered(path, reason),
                on_missed=lambda rows, reason: missed(path, rows, reason),
            )
            logger.info(f"[M3 prebuild] {path} installed on {installed} drafter attention layer(s)")
        if self.spec_worker is not None:
            self.spec_worker.prebuild(self.lm_head, self.draft_model)

    def post_load_weights(self) -> None:
        super().post_load_weights()
        self.model.post_load_weights()

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_context_logits: bool = False,
        spec_metadata: Optional[SpecMetadata] = None,
        resource_manager: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        """One engine step: the target forward, then the logits (or, with speculation, the worker's verify and
        draft, and on a decode step the spec fold's store of its outputs). Text only."""
        multimodal_params = kwargs.pop("multimodal_params", None) or []
        if any(p is not None and getattr(p, "multimodal_data", None) for p in multimodal_params):
            raise ValueError(
                "this MiniMax-M3 target is text only; the request carries multimodal input"
            )
        worker = self.spec_worker
        capture = _plain_capture
        capture_target = None
        if worker is not None:
            capture = worker.capture_hidden_states
            capture_target = worker.capture_target
            if spec_metadata is not None:
                worker.arm_capture_norm(
                    attn_metadata, spec_metadata, self.lm_head, self.draft_model
                )
        hidden_states = self.model(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            capture=capture,
            capture_target=capture_target,
            **kwargs,
        )
        if getattr(attn_metadata, "csa2_remote_tail_mode", None) == "source":
            # A context worker stopped at the encoder/decoder boundary: a disposable token for the context-only
            # completion, and no draft worker.
            return self.logits_processor.forward(hidden_states, self.lm_head, attn_metadata, False)
        if spec_metadata is not None and spec_metadata.is_layer_capture(FINAL_HIDDEN_LAYER_ID):
            capture(spec_metadata, FINAL_HIDDEN_LAYER_ID, hidden_states, None)
        if attn_metadata.padded_num_tokens is not None:
            hidden_states = hidden_states[: attn_metadata.num_tokens]
        if worker is None:
            return self.logits_processor.forward(
                hidden_states, self.lm_head, attn_metadata, return_context_logits
            )
        # The fused verify computes this rank's logits shard itself and needs no gathered logits.
        if worker.tail_eligible(attn_metadata, spec_metadata, self.lm_head):
            logits = None
        else:
            logits = self.logits_processor.forward(
                hidden_states[spec_metadata.gather_ids], self.lm_head, attn_metadata, True
            )
        spec_input_ids = input_ids if input_ids is not None else kwargs.get("orig_input_ids")
        spec_position_ids = position_ids
        if attn_metadata.padded_num_tokens is not None:
            if spec_input_ids is not None:
                spec_input_ids = spec_input_ids[: attn_metadata.num_tokens]
            if position_ids is not None:
                spec_position_ids = _slice_spec_position_ids(position_ids, attn_metadata.num_tokens)
        outputs = worker(
            input_ids=spec_input_ids,
            position_ids=spec_position_ids,
            hidden_states=hidden_states,
            logits=logits,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=self.draft_model,
            resource_manager=resource_manager,
        )
        spec_fold = self._spec_fold
        if (
            spec_fold is not None
            and spec_fold.in_forward
            and attn_metadata.num_contexts == 0
            and isinstance(outputs, dict)
        ):
            # The decode forward stores its outputs into the spec sampler's store and stages the next step's inputs
            # at its tail (one launch inside the step graph).
            spec_fold.store_in_forward(outputs)
        return outputs
