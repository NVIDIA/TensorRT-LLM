# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: MiniMax-M3 NVFP4 / sm_103 / tp4 -- self-contained modeling code.

MiniMax-M3's language model: 60 layers, hidden 6144, 64 query heads and 4 KV heads of dim 128. Layers 0-2 are dense
attention with a dense SwiGLU-OAI MLP; layers 3-59 are MSA block-sparse attention (an index branch scores 128-token
blocks and the top 16 are attended) with a MoE of 128 routed experts (top 4, sigmoid scores with a correction bias,
renormalized and scaled by 2) plus one shared expert. The checkpoint is a modelopt mixed-precision export: NVFP4 routed
experts, MXFP8 projections, fp32 router, bf16 embedding and head; the KV cache is NVFP4 for the sparse layers and FP8
for the dense ones.

``tp4`` is four GB300s: head-split attention (16 query heads, one KV head and one index head per rank) and
tensor-parallel experts (every expert on every rank, intermediate 768 per rank), no attention DP. Speculative
decoding, when configured, is one-model Eagle3 (draft length 3): the BF16 drafter (``M3Eagle3Draft``) and the BF16
target head, both at checkpoint precision; decode steps of up to 8 requests verify and draft through fused kernels
at that precision (``M3Eagle3Worker``).

**The core** (``MiniMaxM3Tp4ModelingV2``) holds the weights (``weights.MODEL_WEIGHTS``) and the runtime resources the
steps use -- one MSA attention backend per layer, the TP all-reduce communicator, the RoPE table, the MXFP8 GEMM
dispatch state the engine tunes -- and dispatches each step, on the host, from its context count alone
(``_select_target``):

* ``DecodeTarget`` takes every pure generation step. Up to ``DECODE_MAX_TOKENS`` tokens (up to 8 requests, each one
  token or a draft verify of 1 + 3 rows) with 1-D input ids it runs the decode-step kernels and picks each layer
  boundary's collective by the step's token bucket (``DECODE_PLANS``); it runs its own generic layer for the first
  layer when the fused embedding does not take the step, for every layer whose output Eagle3 captures, and for every
  layer of a wider step.
* ``PrefillTarget`` takes every step with context rows: prefill chunks and mixed steps, every layer in its generic
  form over the MSA backend.

The two targets share no body: each owns its layer, attention and MoE compositions and its ``step_args``, the one
projection of the step metadata its forward reads. What both call are the module-level op adapters, each picking the
kernel for one call from shapes and layouts (``_linear``, ``_qk_norm_rope``, ``_sparse_qkv_producer``, the router and
MoE-runner calls, the all-reduce + norm calls), which read no step metadata themselves.

Every fast path the decode step can take, the rows each covers (``FAST_PATHS``, each paired with the entry whose
``CELLS`` certify it) and what a step that cannot take one does are below: **uncovered** (outside every covered
range: log once, run the generic path) or **missed** (inside a covered range but a precondition is missing: raise
``FastPathUnavailable`` under ``modeling_v2='require'``, else log once and fall back).

**Hand-offs between kernels.** A producer that already holds an activation in registers writes its MXFP8 form for the
next MXFP8 projection: the decode boundary all-reduce for the next layer's QKV, the sparse-decode merge and the fused
decode attention for the o_proj. It publishes it in ``ops/m3_prequant.SLOT`` keyed by the BF16 tensor; the next
MXFP8 projection on a decode graph takes it (``_linear``), and every take clears the slot. The decode boundaries'
push targets (the o_proj / shared-expert GEMVs and the MoE engine writing straight into the boundary all-reduce's
exchange) are passed explicitly from the layer to the producer and back.

**What this target asserts rather than adapts**: SM 10.3, the topology and checkpoint (``routing.py``); the
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
    NoTarget,
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
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (  # noqa: E501
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
from tensorrt_llm._torch.attention.backends.utils import create_attention
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
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import is_in_breakable_cuda_graph
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
from tensorrt_llm._torch.utils import ActType_TrtllmGen
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

from . import weights as _weights
from .ops import m3_attn_mega as _attn_mega_kernel
from .ops import m3_attn_mega_c1 as _attn_c1_kernel
from .ops import m3_draft_step as m3_draft_step_kernel
from .ops import m3_fused_attn as _fused_attn_kernel
from .ops import m3_index_score as _index_score_kernel
from .ops import m3_mc_allreduce, m3_prequant, m3_spec_tail
from .ops import m3_sampled_accept as m3_sampled_accept_kernel
from .ops import m3_shard_accept as m3_shard_accept_kernel
from .ops import m3_target_head as m3_target_head_kernel
from .ops import msa_kv_update as _msa_kv_update_kernel
from .ops.activation.m3_silu_and_mul import m3_silu_and_mul
from .ops.attention.m3_attn_c1 import m3_attn_c1
from .ops.attention.m3_attn_mega import m3_attn_mega
from .ops.attention.m3_draft_attention import m3_draft_attention
from .ops.attention.m3_fp8_indexer_qk_norm_rope import m3_fp8_indexer_qk_norm_rope
from .ops.attention.m3_fused_decode_attn import m3_fused_decode_attn
from .ops.attention.m3_index_score import SPLIT_K as INDEX_SCORE_SPLIT_K
from .ops.attention.m3_index_score import m3_index_score
from .ops.attention.m3_msa_attention import m3_msa_attention
from .ops.attention.m3_producer import m3_producer
from .ops.attention.m3_qk_norm_rope_fp8 import m3_qk_norm_rope_fp8
from .ops.attention.m3_qkv_insert import m3_qkv_insert
from .ops.attention.m3_select_blocks import m3_select_blocks
from .ops.attention.m3_select_long import MAX_BLOCKS as SELECT_LONG_MAX_BLOCKS
from .ops.attention.m3_select_long import m3_select_long
from .ops.attention.m3_sparse_decode import m3_sparse_decode
from .ops.comm.m3_allreduce import m3_allreduce
from .ops.comm.m3_mc_allreduce_norm import m3_mc_allreduce_norm
from .ops.comm.m3_mc_finalize_allreduce_norm import m3_mc_finalize_allreduce_norm
from .ops.embedding.m3_embedding import m3_embedding
from .ops.gemm.m3_bf16_linear import m3_bf16_linear
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
)
from .ops.gemm.m3_mx_gemv_bf16 import m3_mx_gemv_bf16
from .ops.gemm.m3_mxfp8_gemm import m3_mxfp8_gemm
from .ops.gemm.m3_mxfp8_linear import m3_mxfp8_linear
from .ops.gemm.m3_target_head import MAX_ROWS as TARGET_HEAD_KERNEL_MAX_ROWS
from .ops.gemm.m3_target_head import m3_target_head
from .ops.gemm.mxfp8_low_m_gemm import mxfp8_low_m_gemm
from .ops.long_select import long_select_blocks
from .ops.m3_router_gemv import m3_router_gemv as _router_gemv
from .ops.m3_router_gemv import m3_router_gemv_fp4 as _router_gemv_fp4
from .ops.m3_router_gemv import m3_router_tc as _router_tc
from .ops.moe.m3_moe16_engine import M3Moe16State, m3_moe16_engine
from .ops.moe.m3_moe_engine import M3MoeState, ar_push_desc, m3_moe_engine
from .ops.moe.m3_router_gemv import m3_router_gemv
from .ops.moe.m3_router_gemv_fp4 import m3_router_gemv_fp4
from .ops.moe.m3_router_tc import m3_router_tc
from .ops.mx_gemv import op as mx_gemv
from .ops.mxfp8_low_m_gemm import mxfp8_low_m_gemm as _low_m_gemm
from .ops.norm.m3_embedding_firstnorm import DecodeEmbedding, m3_embedding_firstnorm
from .ops.spec.m3_draft_input import m3_draft_input
from .ops.spec.m3_draft_step import MAX_TOKENS as DRAFT_STEP_KERNEL_MAX_TOKENS
from .ops.spec.m3_draft_step import m3_draft_step
from .ops.spec.m3_force_vec import m3_force_vec
from .ops.spec.m3_shard_accept import m3_shard_accept
from .ops.triton_sparse_decode import MERGE_MXFP8, MERGE_MXFP8_MIN_TOKENS

_SM = (10, 3)
# --------------------------------------------------------------------------------------------------------------------
# Row ranges
# --------------------------------------------------------------------------------------------------------------------

# The largest generation step the decode kernels take: 8 requests verifying 1 + 3 drafts and their padding rows. A
# wider pure generation step stays on the decode target and runs its generic layers.
DECODE_MAX_TOKENS = 48  # capacity: fixed: the decode kernels' step bound


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
DECODE_PLANS = (
    DecodePlan("tiny", 8, attn_mc_ar=True, moe_mc_ar=True, qkv_prequant=True),
    DecodePlan("small", 16, attn_mc_ar=True, moe_mc_ar=True, qkv_prequant=True),
    DecodePlan("verify", DECODE_MAX_TOKENS, attn_mc_ar=True, moe_mc_ar=True, qkv_prequant=True),
)

# The sparse layers' NVFP4 QKV + indexer producer (ops/attention/m3_producer, every step its geometry fits) triggers its
# dependent kernel on entry, so the index scorer launches and stages its pre-wait loads while the producer runs (its
# outputs are read only after the scorer's own grid wait).
PRODUCER_EARLY_TRIGGER = True

# The fused embedding + first RMSNorm kernel: one owner per row count and index dtype.
EMBED_FIRSTNORM_MAX_TOKENS = 8  # capacity: fixed: built rows; wider steps embed generically
# The o_proj pushes its rows into the attention boundary's multicast all-reduce (prepushed mode 2) up to this many
# tokens; wider steps run the GEMV and a plain multicast all-reduce.
ATTN_PUSH_MAX_TOKENS = 16  # capacity: fixed: measured push crossover
AR_PUSH_MODE = 2
# The 1..8-token MoE engine (router, routed experts and the shared expert in one launch) pushes its sum into the MoE
# boundary's all-reduce (prepushed mode 2).
MOE_ENGINE_MAX_TOKENS = 8  # capacity: fixed: the engine's built rows
MOE_PUSH_MAX_TOKENS = MOE_ENGINE_MAX_TOKENS  # capacity: fixed: the engine's rows
# The 9..13-row engine (router, routed and shared experts in one kernel); its bf16 sum goes to the boundary.
MOE16_MIN_TOKENS = 9  # capacity: fixed: the 9..13-row engine's lower bound
MOE16_MAX_TOKENS = 13  # capacity: fixed: the 9..13-row engine's built rows
# From this many tokens the MoE runs unfinalized through the TRTLLM-Gen runner and its finalize (top-k gather,
# routing weights, shared expert add) runs inside the boundary all-reduce. 9..13 take it only when their engine is
# missing (outside require mode).
MC_FINALIZE_MIN_TOKENS = 9  # capacity: fixed: the rows above the engines
# The router: the fp32 GEMV at one row, the tensor-core router (fused with the routed experts' NVFP4 activation
# quantization) at 2..48 rows. GB300 tp4, behind a PDL producer, with the NVFP4 activation: 2.90 / 2.93 / 3.20 /
# 3.82 / 7.36 us at 1 / 4 / 8 / 12 / 48 tokens against the fp32 GEMV's 2.85 / 3.96 / 4.88 / 6.01 / - us.
ROUTER_GEMV_MAX_TOKENS = 1  # capacity: fixed: measured router crossover
ROUTER_TC_MAX_TOKENS = 48  # capacity: fixed: the tensor-core router's built rows
# The fused norm/RoPE/KV-insert/sparse-decode/merge kernel: up to 8 query rows per request and 16 query rows per
# step (above, its 16-CTA clusters run in waves and the three-kernel path is faster: GB300 -13% fAL spec c8).
FUSED_ATTN_MAX_QUERY_LEN = 8  # capacity: fixed: compiled query rows
FUSED_ATTN_MAX_TOKENS = 16  # capacity: fixed: measured wave crossover
# One request verifying a draft of 6 query rows runs attention + o_proj as one launch (96 CTAs: 6 clusters of 16).
ATTN_C1_QUERY_LEN = 6  # capacity: fixed: the verify kernel's query rows
# A greedy Eagle3 verify step of up to this many requests verifies with the fused shard argmax + acceptance kernel
# over the target lm_head's BF16 shard logits (ops/m3_shard_accept), at the configured draft length.
SHARD_ACCEPT_DRAFT_LEN = 3  # capacity: fixed: the built Eagle3 draft length
SHARD_ACCEPT_MAX_REQUESTS = DECODE_MAX_TOKENS // (
    SHARD_ACCEPT_DRAFT_LEN + 1
)  # capacity: fixed: decode rows / (draft length + 1)
# A fused-verify step of up to this many verify rows (one or two requests at draft length 3) computes the target
# logits shard from the 12-bit coded copy of the BF16 head (ops/m3_target_head; the same BF16 x BF16 -> fp32
# arithmetic, 75% of the bytes) instead of the in-tree head GEMM; wider verifies keep the in-tree GEMM.
TARGET_HEAD_MAX_ROWS = TARGET_HEAD_KERNEL_MAX_ROWS
# On a fused-verify step of up to this many verify rows the drafter's first fc is forked onto the aux stream right
# after the last captured target layer, overlapping the remaining target layers and the head (spec_worker).
FC_EARLY_MAX_ROWS = 8  # capacity: fixed: measured fork crossover
# A decode-only Eagle3 step of up to this many requests drafts with the fused draft loop (spec_worker): per pass the
# draft input in one launch (ops/m3_spec_tail.draft_input), the drafter layer's in-tree attention, MLP and
# all-reduces, and one kernel for the draft head (its 12-bit coded copy, or the BF16 fragments when the coding does
# not decode bit for bit), the cross-rank argmax and the pass's bookkeeping (ops/m3_draft_step), at the
# configured draft length. The head kernel takes one row per request, at most 8.
# The fused draft loop streams the 12-bit coded copy of the draft head (when it decodes bit for bit); False keeps the
# BF16 fragments.
DRAFT_HEAD_CODED = True
DRAFT_STEP_DRAFT_LEN = SHARD_ACCEPT_DRAFT_LEN  # capacity: fixed: the built draft length
DRAFT_STEP_MAX_REQUESTS = DRAFT_STEP_KERNEL_MAX_TOKENS
# The identity-selection sparse decode moves from the per-row producer to the multi-query producer at this many rows
# (ops/triton_sparse_decode.MQ_MIN_TOKENS).
SPARSE_DECODE_MQ_MIN_TOKENS = 17  # capacity: fixed: the multi-query crossover
# The long-row top-16 selector takes engines of 129..LONG_SELECT_MAX_BLOCKS KV blocks of 128 tokens; it holds 2, 4 or
# 9 keys per thread by the row's valid-block count. At most 128 blocks the built-in selector runs.
LONG_SELECT_MIN_BLOCKS = 129  # capacity: fixed: past the built-in selector's 128
LONG_SELECT_MAX_BLOCKS = SELECT_LONG_MAX_BLOCKS
LONG_SELECT_KEY_WIDTHS = (2, 4, 9)
# The long-row select releases the sparse decode after it right after its own grid wait (the decode stages q and the
# request length before its wait) up to this many query rows; on wider steps the early-launched attention CTAs crowd
# the select's CTAs off the SMs (8 requests x 4 rows: +3.1 us per layer), so they keep the trigger at the select's exit.
SELECT_EARLY_MAX_ROWS = 8  # capacity: fixed: measured early-release crossover
# The MSA index scorer (ops/attention/m3_index_score) prefetches the older index-K pages into L2 before its grid wait
# at every batch; up to this many requests it also issues Q and the first K tile right after the wait (larger batches
# are bandwidth bound, where the extra early tiles only crowd the prefetch).
INDEX_SCORE_EARLY_MAX_REQUESTS = 1  # capacity: fixed: measured early-issue crossover

# The decode MXFP8 GEMVs, per projection (N, K): the rows each serves. On an activation its producer quantized
# (the MoE boundary for the QKV, the attention merge for the o_proj) the GEMV serves every row up to 48 but the
# unsplit K=768 shared down projection at 17..32 rows, where FlashInfer's GEMM is faster.
_ALL_ROWS = frozenset(range(1, DECODE_MAX_TOKENS + 1))
MX_GEMV_ROWS: Dict[Tuple[int, int], FrozenSet[int]] = {
    QKV_SPARSE: _ALL_ROWS,
    QKV: _ALL_ROWS,
    O_PROJ: _ALL_ROWS,
    SHARED_GATE_UP: _ALL_ROWS,
    SHARED_DOWN: _ALL_ROWS - frozenset(range(17, 33)),
    DENSE_GATE_UP: _ALL_ROWS,
    DENSE_DOWN: _ALL_ROWS,
}
# On a bf16 activation the GEMV quantizes it in its prologue, which pays only up to 8 32-element blocks per thread:
# the most rows each projection takes.
MX_GEMV_BF16_MAX_TOKENS: Dict[Tuple[int, int], int] = {
    QKV_SPARSE: 16,
    QKV: 16,
    O_PROJ: 32,
    SHARED_GATE_UP: 16,
    SHARED_DOWN: 16,
    DENSE_GATE_UP: 8,
    DENSE_DOWN: 16,
}
# The fused low-M MXFP8 GEMM (projections marked for it), per projection: its built configurations' row limit
# (ops/mxfp8_low_m_gemm). The bf16 GEMV serves every one of these rows first; the GEMM takes them only if it
# declines.
LOW_M_GEMM_MAX_TOKENS: Dict[Tuple[int, int], int] = {
    QKV_SPARSE: 8,
    O_PROJ: 8,
    DENSE_GATE_UP: 1,
    DENSE_DOWN: 8,
}

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


def _fused_attn_keys() -> FrozenSet[Tuple[int, int]]:
    return frozenset(
        (qlen, reqs)
        for qlen in range(1, FUSED_ATTN_MAX_QUERY_LEN + 1)
        for reqs in range(1, FUSED_ATTN_MAX_TOKENS // qlen + 1)
    )


def _sparse_decode_producer(spec: Mapping[str, Any]) -> Iterable[Key]:
    if not spec["identity"]:
        return ()
    rows = spec["decode_query_len"] * len(spec["seq_lens"])
    return ("multi_query" if rows >= SPARSE_DECODE_MQ_MIN_TOKENS else "per_row",)


def _long_select_widths(spec: Mapping[str, Any]) -> Iterable[Key]:
    """The key widths a cell's rows reach: every width up to its blocks' when rows of mixed lengths are drawn."""
    blocks = spec["num_blocks"]
    widest = 2 if blocks <= 2048 else 4 if blocks <= 4096 else 9
    if spec["valid"] == "mixed":
        return tuple(w for w in LONG_SELECT_KEY_WIDTHS if w <= widest)
    return (widest,)


EMBED_FIRSTNORM = RowRange(
    "embed_firstnorm",
    _rows(1, EMBED_FIRSTNORM_MAX_TOKENS),
    m3_embedding_firstnorm,
    _spec_rows("rows"),
)
# One request decoding one token on an MXFP8 layer input: QKV GEMV + attention + o_proj in one launch.
ATTN_MEGA = RowRange("attn_mega", frozenset({(1, 1)}), m3_attn_mega, _const((1, 1)))
ATTN_C1 = RowRange(
    "attn_c1", frozenset({(ATTN_C1_QUERY_LEN, 1)}), m3_attn_c1, _const((ATTN_C1_QUERY_LEN, 1))
)
FUSED_DECODE_ATTN = RowRange(
    "fused_decode_attn",
    _fused_attn_keys(),
    m3_fused_decode_attn,
    lambda spec: ((spec["qlen"], len(spec["seq_lens"])),),
)
SPARSE_DECODE_PRODUCER = RowRange(
    "sparse_decode_producer",
    frozenset({"per_row", "multi_query"}),
    m3_sparse_decode,
    _sparse_decode_producer,
)
# Every token count of a sparse layer on an NVFP4 KV cache at the TP4 heads (prefill and decode): the producer takes the
# geometry, not a row range.
PRODUCER = RowRange("m3_producer", frozenset({"nvfp4_tp4"}), m3_producer, _const("nvfp4_tp4"))
INDEX_SCORE = RowRange(
    "m3_index_score",
    frozenset({"early", "prefetch"}),
    m3_index_score,
    lambda spec: ("early" if spec["early"] else "prefetch",),
)
LONG_SELECT = RowRange(
    "long_select", frozenset(LONG_SELECT_KEY_WIDTHS), m3_select_long, _long_select_widths
)
MC_ALLREDUCE = RowRange(
    "mc_allreduce_norm",
    _rows(1, DECODE_MAX_TOKENS),
    m3_mc_allreduce_norm,
    _spec_rows("m", prepushed=0),
)
ATTN_PUSH = RowRange(
    "attn_push",
    _rows(1, ATTN_PUSH_MAX_TOKENS),
    m3_mc_allreduce_norm,
    _spec_rows("m", prepushed=AR_PUSH_MODE),
)
MOE_PUSH = RowRange(
    "moe_push",
    _rows(1, MOE_PUSH_MAX_TOKENS),
    m3_mc_allreduce_norm,
    _spec_rows("m", prepushed=AR_PUSH_MODE),
)
MC_FINALIZE = RowRange(
    "mc_finalize_allreduce_norm",
    _rows(MC_FINALIZE_MIN_TOKENS, DECODE_MAX_TOKENS),
    m3_mc_finalize_allreduce_norm,
    _spec_rows("m"),
)
MOE_ENGINE = RowRange(
    "m3_moe_engine", _rows(1, MOE_ENGINE_MAX_TOKENS), m3_moe_engine, _spec_rows("tokens")
)
MOE16_ENGINE = RowRange(
    "m3_moe16_engine",
    _rows(MOE16_MIN_TOKENS, MOE16_MAX_TOKENS),
    m3_moe16_engine,
    _spec_rows("tokens"),
)
ROUTER_GEMV = RowRange(
    "m3_router_gemv", _rows(1, ROUTER_GEMV_MAX_TOKENS), m3_router_gemv, _spec_rows("m")
)
ROUTER_GEMV_FP4 = RowRange(
    "m3_router_gemv_fp4", _rows(1, ROUTER_GEMV_MAX_TOKENS), m3_router_gemv_fp4, _spec_rows("m")
)
ROUTER_TC = RowRange(
    "m3_router_tc",
    _rows(ROUTER_GEMV_MAX_TOKENS + 1, ROUTER_TC_MAX_TOKENS),
    m3_router_tc,
    _spec_rows("m"),
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
LOW_M_GEMM = RowRange(
    "mxfp8_low_m_gemm",
    frozenset((nk, m) for nk, hi in LOW_M_GEMM_MAX_TOKENS.items() for m in _rows(1, hi)),
    mxfp8_low_m_gemm,
    lambda spec: (((spec["n"], spec["k"]), spec["m"]),),
)
# (draft length, requests) of a greedy verify step.
SHARD_ACCEPT = RowRange(
    "m3_shard_accept",
    frozenset((SHARD_ACCEPT_DRAFT_LEN, n) for n in range(1, SHARD_ACCEPT_MAX_REQUESTS + 1)),
    m3_shard_accept,
    lambda spec: ((spec["draft_len"], spec["nreq"]),),
)

# Verify rows of a fused-verify step whose target logits come from the coded head.
TARGET_HEAD = RowRange(
    "m3_target_head",
    _rows(1, TARGET_HEAD_MAX_ROWS),
    m3_target_head,
    _spec_rows("m"),
)

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
    ATTN_MEGA,
    ATTN_C1,
    FUSED_DECODE_ATTN,
    SPARSE_DECODE_PRODUCER,
    PRODUCER,
    INDEX_SCORE,
    LONG_SELECT,
    MC_ALLREDUCE,
    ATTN_PUSH,
    MOE_PUSH,
    MC_FINALIZE,
    MOE_ENGINE,
    MOE16_ENGINE,
    ROUTER_GEMV,
    ROUTER_GEMV_FP4,
    ROUTER_TC,
    MX_GEMV,
    MX_GEMV_BF16,
    LOW_M_GEMM,
    SHARD_ACCEPT,
    TARGET_HEAD,
    DRAFT_STEP,
    DRAFT_INPUT,
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
    """Whether a pure generation step of ``num_tokens`` rows runs the decode kernels (else generic layers)."""
    return num_tokens <= DECODE_MAX_TOKENS


def moe_path(num_tokens: int) -> Optional[str]:
    """The MoE kernel of a decode step: an engine, the unfinalized MoE + finalize all-reduce, or None (generic)."""
    if MOE_ENGINE.covers(num_tokens):
        return MOE_ENGINE.path
    if MOE16_ENGINE.covers(num_tokens):
        return MOE16_ENGINE.path
    if MC_FINALIZE.covers(num_tokens):
        return MC_FINALIZE.path
    return None


def router_path(num_tokens: int) -> Optional[str]:
    """The router kernel when the MoE runs unfused: the fp32 GEMV, the tensor-core router, or None (built-in gate)."""
    if ROUTER_GEMV.covers(num_tokens):
        return ROUTER_GEMV.path
    if ROUTER_TC.covers(num_tokens):
        return ROUTER_TC.path
    return None


MX_GEMV_PATH = MX_GEMV.path
MX_GEMV_BF16_PATH = MX_GEMV_BF16.path
LOW_M_GEMM_PATH = LOW_M_GEMM.path
# FlashInfer's CuTe DSL MXFP8 GEMM on the producer's MXFP8 activation (rows the GEMV does not serve).
PREQUANT_GEMM_PATH = "flashinfer_mxfp8_gemm"
# The MXFP8 quantize (tuned backend) and FlashInfer's CuTe DSL GEMM: every row the decode kernels do not serve.
GENERIC_GEMM_PATH = "mxfp8_gemm"


def linear_path(n: int, k: int, num_tokens: int, prequant: bool, low_m_gemm: bool) -> str:
    """The decode-graph MXFP8 GEMM of an [N, K] projection on ``num_tokens`` rows.

    ``prequant``: the producer published the activation in MXFP8. ``low_m_gemm``: the projection is marked for the
    fused low-M GEMM.
    """
    if prequant:
        if num_tokens in MX_GEMV_ROWS.get((n, k), ()):
            return MX_GEMV_PATH
        return PREQUANT_GEMM_PATH
    if num_tokens <= MX_GEMV_BF16_MAX_TOKENS.get((n, k), 0):
        return MX_GEMV_BF16_PATH
    if low_m_gemm and num_tokens <= LOW_M_GEMM_MAX_TOKENS.get((n, k), 0):
        return LOW_M_GEMM_PATH
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


def uncovered(path: str, reason: str) -> None:
    """The step is outside ``path``'s covered shapes: log once per (path, reason); the generic path runs."""
    _log_once(logger.info, path, reason, f"[M3 dispatch] {path} not taken: {reason} (logged once)")


def reset_logged() -> None:
    """Forget which (path, reason) pairs were logged."""
    with _lock:
        _logged.clear()


# --------------------------------------------------------------------------------------------------------------------
# Target constants
# --------------------------------------------------------------------------------------------------------------------

# The generic boundary takes the low-latency multicast all-reduce + residual + RMSNorm up to this many tokens; with
# the shared + routed add folded in, up to its own maximum (m3_mc_allreduce.MAX_TOKENS).
MC_AR_MAX_TOKENS = 8  # capacity: fixed: measured crossover vs the one-shot all-reduce
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

# --------------------------------------------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------------------------------------------
# Configuration: the text tower and the runtime the target assumes
# --------------------------------------------------------------------------------------------------------------------


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
        raise NoTarget(
            "this MiniMax-M3 target requires sparse_attention_config.algorithm='minimax_m3' with implementation='msa'"
        )
    if getattr(sparse, "indexer_kv_dtype", None) != "fp8":
        raise NoTarget(
            "this MiniMax-M3 target requires sparse_attention_config.indexer_kv_dtype='fp8'"
        )
    mode = model_config.quant_config.quant_mode
    if mode is None or not (mode.has_fp8_kv_cache() or mode.has_fp4_kv_cache()):
        raise NoTarget("this MiniMax-M3 target requires an FP8 or NVFP4 KV cache")


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


def _gemv_variants(n: int, k: int, push: Optional[str], world: int):
    """Every (rows, bf16 activation, push (world, bucket tokens, mode)) call the decode dispatch routes to an [N, K]
    projection's GEMV: an MXFP8 activation on the rows ``MX_GEMV_ROWS`` gives it, a bf16 one up to
    ``MX_GEMV_BF16_MAX_TOKENS``; each pushing when the projection's boundary push covers the rows."""
    push_rows = {"attn": ATTN_PUSH.keys, "moe": MOE_PUSH.keys}.get(push, frozenset())
    variants = []
    for rows in range(1, DECODE_MAX_TOKENS + 1):
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


def _low_m_gemm_supports(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor) -> bool:
    """Whether the fused MXFP8 low-M kernel handles these operands."""
    return (
        x.dim() == 2
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.stride(-1) == 1
        and weight.dtype == torch.float8_e4m3fn
        and weight.is_contiguous()
        and weight_scale.dtype == torch.uint8
        and x.shape[1] == weight.shape[1]
        and weight.shape[1] % 256 == 0
    )


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
       GEMV on the rows it serves, else the decode-graph MXFP8 GEMM on that activation (AutoTuner-selected backend
       per token bucket);
    2. a bf16 activation on the rows the GEMV serves quantizing in its prologue;
    3. ``low_m_gemm`` projections, on its rows: the fused low-M MXFP8 GEMM;
    4. otherwise the engine-tuned MXFP8 quantize + GEMM.

    A kernel declining rows its path covers is a miss; the chain then continues down the list. ``push`` (an
    ``m3_mx_gemv`` ``ar`` tuple) is the boundary all-reduce exchange the GEMV writes its rows into.
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
    output = None
    if prequant is not None:
        act, act_scale = prequant
        if path == MX_GEMV_PATH:
            output = m3_mx_gemv(act, act_scale, w, s, True, None, use_pdl, push)
            if output is not None:
                return output, push is not None and mx_gemv.last_pushed
            missed(
                path,
                rows,
                f"the GEMV declined the [{n}, {k}] projection's MXFP8 activation"
                + _uncompiled_note(),
            )
        if path == PREQUANT_GEMM_PATH:
            uncovered(MX_GEMV_PATH, f"the [{n}, {k}] projection at rows the GEMV does not serve")
        return m3_mxfp8_gemm(act, act_scale, w, s, torch.bfloat16, mode.autotune), False
    if path == MX_GEMV_BF16_PATH:
        output = m3_mx_gemv_bf16(x, w, s, True, None, use_pdl, push)
        if output is not None:
            return output, push is not None and mx_gemv.last_pushed
        missed(
            path, rows, f"the GEMV declined the [{n}, {k}] projection's input" + _uncompiled_note()
        )
    if (
        path in (MX_GEMV_BF16_PATH, LOW_M_GEMM_PATH)
        and low_m_gemm
        and rows <= LOW_M_GEMM_MAX_TOKENS.get((n, k), 0)
        and _low_m_gemm_supports(x, w, s)
    ):
        output = _low_m_gemm(x, w, s, torch.bfloat16, use_pdl=use_pdl)
        if output is not None:
            return output, False
        if path == LOW_M_GEMM_PATH:
            missed(path, rows, f"the GEMM declined the [{n}, {k}] projection's input")
    if path == GENERIC_GEMM_PATH:
        uncovered(
            MX_GEMV_BF16_PATH,
            f"the [{n}, {k}] projection's bf16 input beyond the decode GEMV's rows",
        )
    return _mxfp8_linear(core, x, w, s, mode), False


# --------------------------------------------------------------------------------------------------------------------
# Layer boundaries: all-reduce + residual + RMSNorm
# --------------------------------------------------------------------------------------------------------------------


def _can_mc(x, folded: bool = False) -> bool:
    """Whether the generic boundary takes the multicast all-reduce + residual + RMSNorm."""
    return (
        isinstance(x, torch.Tensor)
        and x.dim() == 2
        and x.shape[0] <= (m3_mc_allreduce.MAX_TOKENS if folded else MC_AR_MAX_TOKENS)
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
    )


def _mc_allreduce_norm(core, x, x2, residual, norm_w, *, quant=False, prepushed=0):
    """The multicast all-reduce + optional second addend + residual + RMSNorm (``m3_mc_allreduce_norm``), or None
    without a workspace or for operands the kernel does not take (the same on every rank). With ``quant`` the normed
    rows are also published, MXFP8-quantized, to ``m3_prequant.SLOT`` for the next MXFP8 projection."""
    workspace = m3_mc_allreduce.get_workspace(core.mapping, x.shape[-1])
    if workspace is None:
        return None
    _record("m3_mc_allreduce_norm", x.shape[0], x2 is not None, quant, prepushed)
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
    weights are real (``MiniMaxM3Tp4ModelingV2._derive_moe``)."""

    fc31_input_scale: torch.Tensor
    fc2_input_scale: torch.Tensor
    fc31_alpha: torch.Tensor
    fc2_alpha: torch.Tensor
    fc31_scale_c: torch.Tensor
    act_alpha: torch.Tensor
    act_beta: torch.Tensor
    act_clamp: torch.Tensor


def _router_logits(core, i: int, h: torch.Tensor) -> torch.Tensor:
    """The fp32 router logits of MoE layer ``i``: the fp32 GEMV at one row, the tensor-core router up to 48, else the
    fp32 linear."""
    rows = h.shape[0] if h.dim() == 2 else 0
    path = router_path(rows)
    w = core.w[f"l{i}_router"]
    if path == ROUTER_GEMV.path:
        logits = _router_gemv(h, w, use_pdl=get_env_enable_pdl())
        if logits is not None:
            return logits
        missed(path, rows, "the router GEMV declined the operands")
    elif path == ROUTER_TC.path:
        fused = _router_tc(h, w, use_pdl=get_env_enable_pdl())
        if fused is not None:
            return fused[0]
        missed(path, rows, "the tensor-core router declined the operands")
    else:
        uncovered(ROUTER_TC.path, f"more than {ROUTER_TC_MAX_TOKENS} rows")
    return m3_fp32_linear(h, w)


def _router_fp4(core, i: int, h: torch.Tensor):
    """Router logits and the routed experts' NVFP4 activation from one kernel, or None (then ``_router_logits``)."""
    if h.dim() != 2 or h.shape[0] > ROUTER_TC_MAX_TOKENS:
        return None
    rows = h.shape[0]
    path = router_path(rows)
    if path == ROUTER_GEMV.path:
        path = ROUTER_GEMV_FP4.path
    scale = core.moe[i].fc31_input_scale
    if path == ROUTER_TC.path:
        fused = _router_tc(h, core.w[f"l{i}_router"], scale, use_pdl=get_env_enable_pdl())
    else:
        fused = _router_gemv_fp4(h, core.w[f"l{i}_router"], scale, use_pdl=get_env_enable_pdl())
    if fused is None:
        missed(path, rows, "the router declined the operands")
    return fused


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
    """Whether ``PRODUCER`` covers this target's sparse layers: an NVFP4 KV cache at the TP4 heads (16 query, one KV
    and one index head). A covered layer that cannot take ``m3_producer`` is a miss."""
    return core.main_kv_is_nvfp4 and (core.heads, core.kv_heads, core.index_heads) == (16, 1, 1)


def _producer_layout(kv_data, kv_scale) -> bool:
    """Whether a sparse layer's NVFP4 pools are the layout ``m3_producer`` is built for: [pages, 2, 1, 128, 64] with
    its scale pool [pages, 2, 1, 128, 8]."""
    return (
        kv_data is not None
        and kv_scale is not None
        and kv_data.dim() == 5
        and tuple(kv_data.shape[1:]) == (2, 1, 128, 64)
        and kv_data.stride()[2:] == (128 * 64, 64, 1)
        and kv_scale.dim() == 5
        and tuple(kv_scale.shape[1:]) == (2, 1, 128, 8)
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


def _producer_miss(covered: bool, rows: Optional[int], reason: str) -> None:
    if covered:
        missed(PRODUCER.path, rows, reason)
    return None


def _sparse_qkv_producer(
    core, i: int, packed, position_ids, kv_cache_manager, msa_idx_k_cache, msa_out_cache_loc
):
    """The horizontal producer of a sparse layer: Q / index-Q (FP8) out, K / V / index-K written into the paged
    caches (NVFP4 or FP8 main cache), for every step type (token-major). An NVFP4 cache at the TP4 heads takes
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
        and (core.heads, core.kv_heads, core.index_heads) == (16, 1, 1)
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


def _attn_mega(core, i: int, hidden: torch.Tensor, args: dict, push: Optional[tuple]):
    """(o_proj output, pushed) of the one-launch QKV + attention + o_proj kernel for one request decoding one token
    whose layer input the preceding all-reduce handed over in MXFP8; None, with nothing consumed, otherwise."""
    if not ATTN_MEGA.covers((args["decode_query_len"], int(args["seq_lens"].shape[0]))):
        return None
    if not flashinfer_mxfp8_decode_graph_capture_active():
        return None
    pending = m3_prequant.SLOT.pending
    if (
        pending is None
        or hidden.dim() != 2
        or pending[0] != hidden.data_ptr()
        or pending[1] != 1
        or pending[2] != hidden.shape[1]
    ):
        # After a layer whose boundary does not quantize (a dense layer, an Eagle3 capture layer).
        uncovered(ATTN_MEGA.path, "the layer input has no MXFP8 hand-off")
        return None
    w = core.w
    if w[f"l{i}_qkv_w"].shape != (2560, 6144) or w[f"l{i}_o_w"].shape != (6144, 2048):
        uncovered(ATTN_MEGA.path, "not the tp4 QKV / o_proj shapes")
        return None
    workspace = core.mega_ws.get(i)
    if workspace is None:
        # The first call JIT-builds the module; both only outside capture (the eager warmup steps).
        if torch.cuda.is_current_stream_capturing():
            capture_missed(ATTN_MEGA.path, 1, "the attn_mega module and workspace")
            return None
        _attn_mega_kernel._load_module()
        workspace = core.mega_ws[i] = _attn_mega_kernel.MegaWorkspace(hidden.device)
    xq, xsf = m3_prequant.SLOT.take(hidden)
    output = hidden.new_empty((1, core.heads * core.head_dim), dtype=torch.bfloat16)
    y = hidden.new_empty((1, w[f"l{i}_o_w"].shape[0]), dtype=torch.bfloat16)
    _record("m3_attn_mega", push is not None)
    _attn_mega_kernel.m3_attn_mega(
        xq,
        xsf,
        w[f"l{i}_qkv_w"],
        w[f"l{i}_qkv_s"],
        workspace,
        args["kv_data"],
        args["kv_scale"],
        args["index_k_cache"],
        args["out_cache_loc"],
        args["position_ids"],
        args["block_table"],
        args["seq_lens"],
        args["kv_quant_scale"],
        args["k_global_scale"],
        args["v_global_scale"],
        w[f"l{i}_q_norm"],
        w[f"l{i}_k_norm"],
        w[f"l{i}_idx_k_norm"],
        core.rotary_cos_sin,
        w[f"l{i}_o_w"],
        w[f"l{i}_o_s"],
        eps=core.rms_eps,
        sm_scale=core.sm_scale(i),
        decode_query_len=1,
        output=output,
        y=y,
        use_pdl=get_env_enable_pdl(),
        ar=push,
    )
    return y, push is not None


def _attn_c1(core, i: int, packed: torch.Tensor, args: dict, push: Optional[tuple]):
    """(o_proj output, True) of the one-launch attention + o_proj kernel for one request verifying a draft, pushed
    into the attention boundary's all-reduce; None, with nothing consumed, when the step or the layer does not
    qualify."""
    qlen, num_tokens = args["decode_query_len"], args["num_tokens"]
    if not ATTN_C1.covers((qlen, int(args["seq_lens"].shape[0]))):
        return None
    if not flashinfer_mxfp8_decode_graph_capture_active():
        return None
    if push is None:
        missed(ATTN_C1.path, num_tokens, "no armed attention-boundary push to write into")
        return None
    w = core.w
    if w[f"l{i}_o_w"].shape != (6144, 2048):
        uncovered(ATTN_C1.path, "not the tp4 o_proj shape")
        return None
    workspace = core.c1_ws.get(i)
    if workspace is None:
        # The first call JIT-builds the module; both only outside capture (the eager warmup steps).
        if torch.cuda.is_current_stream_capturing():
            capture_missed(ATTN_C1.path, num_tokens, "the attn_c1 module and workspace")
            return None
        _attn_c1_kernel._load_module()
        workspace = core.c1_ws[i] = _attn_c1_kernel.MegaWorkspace(packed.device)
    output = packed.new_empty((num_tokens, core.heads * core.head_dim), dtype=torch.bfloat16)
    y = packed.new_empty((num_tokens, w[f"l{i}_o_w"].shape[0]), dtype=torch.bfloat16)
    _record("m3_attn_c1", num_tokens)
    _attn_c1_kernel.m3_attn_c1(
        packed.contiguous(),
        workspace,
        args["kv_data"],
        args["kv_scale"],
        args["index_k_cache"],
        args["out_cache_loc"],
        args["position_ids"],
        args["block_table"],
        args["seq_lens"],
        args["kv_quant_scale"],
        args["k_global_scale"],
        args["v_global_scale"],
        w[f"l{i}_q_norm"],
        w[f"l{i}_k_norm"],
        w[f"l{i}_idx_k_norm"],
        core.rotary_cos_sin,
        w[f"l{i}_o_w"],
        w[f"l{i}_o_s"],
        eps=core.rms_eps,
        sm_scale=core.sm_scale(i),
        decode_query_len=qlen,
        output=output,
        y=y,
        use_pdl=get_env_enable_pdl(),
        ar=push[:5],
    )
    return y, True


def _fused_attn(core, i: int, packed: torch.Tensor, args: dict) -> torch.Tensor:
    """The fused norm/RoPE/KV-insert/sparse-decode/merge kernel after the packed QKV projection; its MXFP8 copy of
    the output goes to the o_proj through ``m3_prequant.SLOT``."""
    num_tokens = args["num_tokens"]
    width = core.heads * core.head_dim
    output = packed.new_empty((num_tokens, width), dtype=torch.bfloat16)
    mxfp8 = None
    if MERGE_MXFP8_MIN_TOKENS <= num_tokens <= MERGE_MXFP8.MAX_ROWS:
        mxfp8 = MERGE_MXFP8.buffers(width, output.device)
    w = core.w
    _record("m3_fused_decode_attn", num_tokens, args["decode_query_len"])
    _fused_attn_kernel.m3_fused_decode_attn(
        packed.contiguous(),
        args["kv_data"],
        args["kv_scale"],
        args["index_k_cache"],
        args["out_cache_loc"],
        args["position_ids"],
        args["block_table"],
        args["seq_lens"],
        args["kv_quant_scale"],
        args["k_global_scale"],
        args["v_global_scale"],
        w[f"l{i}_q_norm"],
        w[f"l{i}_k_norm"],
        w[f"l{i}_idx_k_norm"],
        core.rotary_cos_sin,
        eps=core.rms_eps,
        sm_scale=core.sm_scale(i),
        decode_query_len=args["decode_query_len"],
        output=output,
        mxfp8_out=mxfp8,
        use_pdl=get_env_enable_pdl(),
    )
    if mxfp8 is not None:
        m3_prequant.SLOT.publish(output, mxfp8[0][:num_tokens], mxfp8[1])
    return output


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
# The core
# --------------------------------------------------------------------------------------------------------------------

# Engine states (compiled kernel + workspace) per (device, token count[, shared fold]), shared by every layer: a step
# runs the layers one after another on one stream.
_ENGINE_STATES: Dict[tuple, object] = {}
_MOE16_STATES: Dict[tuple, object] = {}


class MiniMaxM3Tp4ModelingV2(DecoderModel):
    """The MiniMax-M3 text model of this target: its weights, its runtime resources, and the per-step dispatch."""

    def __init__(self, model_config: ModelConfig[PretrainedConfig]) -> None:
        _validate_runtime(model_config)
        super().__init__(model_config)
        cfg = model_config.pretrained_config
        mapping = model_config.mapping
        self.mapping = mapping
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
            self.attn.append(backend)

        # Built once the weights are real (``post_load_weights``): the KV (Q, K, V) scales, the MoE scalars, the
        # decode engines and workspaces, the two targets.
        self.kv_scales: Dict[int, torch.Tensor] = {}
        self.inv_kv_scales: Dict[int, torch.Tensor] = {}
        self.moe: Dict[int, _MoeScalars] = {}
        self.mx_alpha: Optional[torch.Tensor] = None
        self.moe_engines: Dict[int, Dict[Tuple[int, bool, bool], Optional[object]]] = {}
        self.moe16_engines: Dict[int, Dict[int, Optional[object]]] = {}
        self.embed_owners: Dict[tuple, DecodeEmbedding] = {}
        self.mega_ws: Dict[int, object] = {}
        self.c1_ws: Dict[int, object] = {}
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
                out.append((f"l{i}_mlp_dn_w", f"l{i}_mlp_dn_s", None))
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
        ``MOE_ENGINE`` / ``MOE16_ENGINE``'s token counts, and the index scorer at the decode query lengths. Runs on the
        current (main) stream before the executor's warmup, in the same order on every rank, then synchronizes; after
        it, GEMV compiles are frozen and the forward only looks kernels up (a missing one is a miss). Returns the
        seconds each part took and the scorer keys."""
        times: Dict[str, object] = {}
        if not self.mxfp8.quant_method.use_cutlass:
            raise ValueError("the MiniMax-M3 decode dispatch needs the CUTLASS MXFP8 GEMM")
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
        times["scorer_keys"] = self._prebuild_index_scorer(
            self.decode_query_lens(), self.w["embed"].device
        )
        torch.cuda.synchronize()
        times["scorer_s"] = time.perf_counter() - start
        mx_gemv.freeze(True)
        logger.info(
            "[M3 prebuild] decode kernels built before warmup: GEMVs %.1f s, MoE engines %.1f s, index scorer "
            "%.1f s (%d variants)"
            % (times["gemv_s"], times["moe_s"], times["scorer_s"], len(times["scorer_keys"]))
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

    def moe16_engine(self, i: int, num_tokens: int) -> Optional[object]:
        """Layer ``i``'s prebuilt 9..13-row engine (router, routed and shared experts in one kernel), or None."""
        if not MOE16_MIN_TOKENS <= num_tokens <= MOE16_MAX_TOKENS:
            return None
        return self._lookup(i, self.moe16_engines[i], num_tokens, MOE16_ENGINE.path, num_tokens)

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
        take: ``MOE_ENGINE``'s token counts with the shared expert folded in, and ``MOE16_ENGINE``'s. Whether an
        engine is kept is agreed across the TP group (one host collective per engine): a build that fails on one rank
        drops that engine on every rank, so the ranks never mix an engine (and its boundary push) with the generic
        MoE."""
        engines = self.moe_engines.setdefault(i, {})
        engines16 = self.moe16_engines.setdefault(i, {})
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
        for num_tokens in sorted(MOE16_ENGINE.keys):
            if num_tokens not in engines16:
                engines16[num_tokens] = m3_mc_allreduce.tp_agree_built(
                    self.mapping,
                    self._try_build(i, MOE16_ENGINE.path, num_tokens, self._build_moe16_engine),
                    f"[M3 dispatch] {MOE16_ENGINE.path} at {num_tokens} rows: layer {i}'s engine",
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
        tensors, scale = self._expert_operands(i)
        device = tensors[0].device
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        kwargs = {}
        if shared:
            kwargs["shared_weights"] = self._shared_weights(i)
        state = _ENGINE_STATES.get((str(device), num_tokens, shared))
        if state is None:
            state = _ENGINE_STATES[(str(device), num_tokens, shared)] = M3MoeState(
                device, num_tokens=num_tokens, num_ctas=sms - MOE_ENGINE_RESERVED_SMS, shared=shared
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

    def _build_moe16_engine(self, i: int, num_tokens: int) -> object:
        tensors, scale = self._expert_operands(i)
        device = tensors[0].device
        state = _MOE16_STATES.get((str(device), num_tokens))
        if state is None:
            state = _MOE16_STATES[(str(device), num_tokens)] = M3Moe16State(
                device, num_tokens=num_tokens
            )
        engine = state.layer(
            *tensors,
            self.w[f"l{i}_router_bias"].data,
            router_weight=self.w[f"l{i}_router"].data,
            fc31_input_scale=scale.reshape(1).float(),
            shared_weights=self._shared_weights(i),
        )
        # Compile (first use of the state) and warm.
        m3_moe16_engine(
            torch.zeros(num_tokens, tensors[0].shape[-1] * 2, dtype=torch.bfloat16, device=device),
            engine,
        )
        return engine

    # ------------------------------------------------------------------------------------------ the index scorer

    def _scorer_variants(self, query_lens) -> list:
        """The scorer keys a decode step can take: every sparse layer's (index dtype, index heads per rank, index dim)
        at each decode query length, with the early-issue schedule and without it, where the kernel serves it."""
        runner = _index_score_kernel.IndexScoreRunner
        keys = []
        for i in sorted(self.sparse_layers):
            backend = self.attn[i]
            config = backend.m3_config
            dtype = torch.float8_e4m3fn if backend.indexer_kv_dtype == "fp8" else torch.bfloat16
            for qlen in sorted(set(query_lens)):
                if not runner.is_supported(
                    q_dtype=dtype,
                    num_heads=config.num_index_heads,
                    head_dim=config.sparse_index_dim,
                    page_size=runner.SUPPORTED_PAGE_SIZE,
                    max_decode_query_len=qlen,
                ):
                    continue
                for early in (True, False):
                    key = (
                        dtype,
                        int(config.num_index_heads),
                        int(qlen),
                        int(config.sparse_index_dim),
                        int(INDEX_SCORE_SPLIT_K),
                        bool(early),
                    )
                    if key not in keys:
                        keys.append(key)
        return keys

    def _prebuild_index_scorer(self, query_lens, device) -> list:
        """Compile every scorer key and launch each once on real-shaped dummy operands (loading its module), on the
        current stream. The kernel's batch, token, page and block counts are symbolic: its compile needs no step
        metadata. Returns the keys."""
        runner = _index_score_kernel.IndexScoreRunner
        keys = self._scorer_variants(query_lens)
        page = runner.SUPPORTED_PAGE_SIZE
        for key in keys:
            dtype, heads, qlen, dim, split_k, early = key
            runner._compile(dtype, heads, qlen, dim, split_k, early)
            rows = 1 if early else INDEX_SCORE_EARLY_MAX_REQUESTS + 1
            blocks = 2
            tokens = rows * qlen
            idx_q = torch.zeros(tokens, heads, dim, dtype=torch.bfloat16, device=device).to(dtype)
            idx_k = torch.zeros(
                rows * blocks, 1, page, dim, dtype=torch.bfloat16, device=device
            ).to(dtype)
            block_table = torch.arange(rows * blocks, dtype=torch.int32, device=device).view(
                rows, blocks
            )
            seq_lens = torch.full((rows,), blocks * page, dtype=torch.int32, device=device)
            score = torch.empty(heads, blocks, tokens, dtype=torch.float32, device=device)
            runner.kernel_cache[key](
                idx_q, idx_k.squeeze(1), block_table, score.transpose(1, 2), seq_lens
            )
        return keys

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
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if kwargs.get("lora_params") is not None:
            raise ValueError("this MiniMax-M3 target serves no LoRA adapters")
        return self._select_target(attn_metadata).forward(
            attn_metadata, input_ids, position_ids, inputs_embeds, spec_metadata, capture
        )


# --------------------------------------------------------------------------------------------------------------------
# The targets
# --------------------------------------------------------------------------------------------------------------------


class PrefillTarget:
    """A step with context rows: prefill chunks, and the mixed steps that carry generation rows beside them. Every
    layer runs in its generic form; the attention core is the MSA backend's for every row."""

    def __init__(self, core: MiniMaxM3Tp4ModelingV2) -> None:
        self.core = core
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
        self, attn_metadata, input_ids, position_ids, inputs_embeds, spec_metadata, capture
    ):
        core = self.core
        if is_in_breakable_cuda_graph():
            raise RuntimeError(
                "this MiniMax-M3 target does not run under piecewise (breakable) prefill CUDA graphs"
            )
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

    def _swiglu_mlp(self, h: torch.Tensor, prefix: str, low_m_gemm: bool) -> torch.Tensor:
        """A gated MXFP8 MLP (the dense MLP, the shared expert): gate_up, SwiGLU-OAI, down, no reduction."""
        core = self.core
        gu, _ = _linear(core, h, f"{prefix}_gu_w", f"{prefix}_gu_s", low_m_gemm)
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
        """Router + routed experts, finalized: the fused router with the NVFP4 activation up to 48 rows, else the
        router logits and the experts quantizing their bf16 input (chunked)."""
        core = self.core
        fused = _router_fp4(core, i, h)
        if fused is not None:
            router_logits, x_fp4, x_sf = fused
            return _routed_experts(core, i, x_fp4, x_sf, router_logits, True)
        router_logits = _router_logits(core, i, h)
        return self._routed_chunks(i, h, router_logits)

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
                folded = m3_moe_engine(h.contiguous(), engine, None, None)
                return folded, None
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
            out, routed = self._moe(i, hidden, return_parts=_can_mc(hidden, folded=True))
        else:
            out = self._swiglu_mlp(hidden, f"l{i}_mlp", True)
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
            # Dense layers 0-2: QKV, the fused norm + RoPE (FP8 q/k/v for an FP8 cache, else in place in bf16), the
            # MSA backend (it writes the layer's cache), o_proj.
            qkv, _ = _linear(core, hidden, f"l{i}_qkv_w", f"l{i}_qkv_s", True)
            fused = _qk_norm_rope(core, i, qkv, position_ids, core.main_kv_is_fp8)
            q, k, v = fused.split([core.q_size, core.kv_size, core.kv_size], dim=-1)
            o = self._attention_core(i, q, k, v, None, step)
            return _linear(core, o, f"l{i}_o_w", f"l{i}_o_s", True)
        if not core.fused_projection:
            return self._unfused_sparse_attention(i, position_ids, hidden, step)
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
        live = int(step["num_tokens"])
        q_live, out_live = q[:live], output[:live]
        k_live = k[:live] if k is not None else None
        v_live = v[:live] if v is not None else None
        idx_q_live = idx_q[:live] if idx_q is not None else None
        sparse = i in core.sparse_layers
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


class DecodeTarget:
    """A pure generation step (no context rows), of any width.

    Up to ``DECODE_MAX_TOKENS`` tokens with 1-D input ids, per layer: attention (o_proj all-reduce deferred, its rows
    pushed into the boundary all-reduce up to 16 tokens) -> attention boundary (all-reduce + residual + post-attention
    RMSNorm) -> MoE (the 1..8 / 9..13-row engines, the first pushing its sum into the boundary all-reduce; the
    unfinalized MoE with the finalize in the all-reduce from 14 rows) or dense MLP -> MoE boundary (all-reduce +
    shared/routed add + residual + the next layer's RMSNorm, emitting the next QKV's MXFP8 activation). A layer whose
    output Eagle3 captures runs the generic layer, as does the first layer when the fused embedding does not take the
    step. A wider step (or one carrying embeddings) runs every layer in its generic form, with this target's attention
    (the decode selection and sparse decode where the producer wrote the caches).
    """

    def __init__(self, core: MiniMaxM3Tp4ModelingV2) -> None:
        self.core = core
        self._moe_push_descs: Dict[tuple, Any] = {}
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

        Graph classes: counts, the generation span, the identity-selection flag and the worst-case score extent are
        host values fixed per capture (prepare() stages them before the forward); the block table, lengths, cache
        slots, identity table and valid-block counts are the metadata's persistent buffers, held by reference and
        refreshed in place every step; ``msa_idx_k_cache`` / ``msa_proxy_max_score_view`` are the metadata's getters
        of the per-layer index-K cache and of the score scratch view. ``metadata`` is the prepared object itself: the
        MSA backend's step (``m3_msa_attention``) takes it as its argument.
        """
        return dict(
            num_contexts=0,
            num_ctx_tokens=0,
            num_tokens=md.num_tokens,
            num_seqs=md.num_seqs,
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
        self, attn_metadata, input_ids, position_ids, inputs_embeds, spec_metadata, capture
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
            if (
                not kernels
                or residual is None
                or (spec_metadata is not None and spec_metadata.is_layer_capture(i))
            ):
                hidden, residual = self._generic_layer(
                    i, position_ids, hidden, residual, step, spec_metadata, capture
                )
            else:
                hidden, residual = self._layer(i, plan, position_ids, hidden, residual, step)
        return hidden

    def _embed(self, input_ids: torch.Tensor, num_tokens: int):
        """(hidden, residual) of the fused embedding + first RMSNorm; (embedding, None) when it does not run, so the
        first layer runs its own input norm."""
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
        """The o_proj's push tuple into the attention boundary's all-reduce, or None (after a miss).

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

    def _boundary(self, use_mc: bool, x, residual, norm_w, x2=None, quant=False):
        """All-reduce + (second addend) + residual + RMSNorm: the multicast kernel, else the built-in all-reduce."""
        core = self.core
        if use_mc:
            fused = _mc_allreduce_norm(core, x, x2, residual, norm_w, quant=quant)
            if fused is not None:
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

    def _prepushed(self, i: int, x, residual, norm_w, quant, prepushed, x2=None):
        fused = _mc_allreduce_norm(
            self.core, x, x2, residual, norm_w, quant=quant, prepushed=prepushed
        )
        if fused is None:
            raise RuntimeError(f"layer {i}: rows were pushed into an all-reduce that did not run")
        return fused

    def _layer(self, i: int, plan: DecodePlan, position_ids, hidden, residual, step):
        core = self.core
        num_tokens = hidden.shape[0]
        # Padded rows: the boundary all-reduce polls and re-arms rows t < M = residual.shape[0], CUDA-graph padding
        # rows included. Every producer pushes all rows of the same tensor: the o_proj / shared down GEMV t < m =
        # x.shape[0] (its row count comes from the host shape, not a live count), the MoE engine every row of the
        # num_tokens it was built for (= num_tokens here). The all-reduce checks x.sizes() == residual.sizes(), so
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
            hidden, residual = self._prepushed(i, hidden, residual, post_norm, False, AR_PUSH_MODE)
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
            out, _ = self._swiglu_mlp(hidden, f"l{i}_mlp", True)
            return self._boundary(plan.attn_mc_ar, out, residual, norm_w)

        quant = plan.qkv_prequant and i + 1 < core.num_layers
        path = moe_path(num_tokens)
        moe_push = None
        if path == MOE_ENGINE.path:
            # The engine runs inside the MoE; looked up (or found missing) here first so a push is armed only for an
            # engine that will take it.
            engine = core.engine(i, num_tokens, shared=True)
            if engine is not None and plan.moe_mc_ar and decode_graph:
                moe_push = self._moe_push_targets(num_tokens)
        elif path == MOE16_ENGINE.path and decode_graph:
            # Router, routed and shared experts in one kernel; its bf16 sum goes to the boundary all-reduce.
            engine = core.moe16_engine(i, num_tokens)
            if engine is not None:
                _record("m3_moe16_engine", num_tokens)
                out = m3_moe16_engine(hidden.contiguous(), engine)
                return self._boundary(plan.moe_mc_ar, out, residual, norm_w, quant=quant)
        if path in (MOE16_ENGINE.path, MC_FINALIZE.path) and plan.moe_mc_ar:
            if self._bucket_flags(num_tokens) is None:
                self._no_bucket(MC_FINALIZE.path, num_tokens)
            else:
                (shared, fc2, scales, perm), _, _, _ = self._moe(
                    i, hidden, return_parts=False, do_finalize=False
                )
                _record("m3_mc_finalize_allreduce_norm", num_tokens, quant)
                fused = m3_mc_allreduce.m3_mc_finalize_allreduce_norm(
                    fc2,
                    perm.view(num_tokens, -1).to(torch.int32),
                    scales.reshape(num_tokens, -1),
                    shared,
                    residual,
                    norm_w,
                    core.rms_eps,
                    m3_mc_allreduce.get_workspace(core.mapping, core.hidden),
                    use_pdl=get_env_enable_pdl(),
                    quant=quant,
                )
                if fused is None:
                    raise RuntimeError(
                        "the MC finalize all-reduce rejected the unfinalized MoE outputs"
                    )
                return fused
        shared, routed, engine_pushed, shared_pushed = self._moe(
            i, hidden, return_parts=True, push=moe_push
        )
        if moe_push is not None:
            if routed is None:
                # The engine folded the shared expert in and pushed routed + shared into set A.
                if engine_pushed:
                    return self._prepushed(i, shared, residual, norm_w, quant, 2)
            else:
                if engine_pushed and shared_pushed:
                    return self._prepushed(i, shared, residual, norm_w, quant, 3, x2=routed)
                if engine_pushed or shared_pushed:
                    raise RuntimeError("only one MoE part pushed into the boundary all-reduce")
            missed(
                MOE_PUSH.path,
                num_tokens,
                "the MoE did not push its output into the armed boundary all-reduce",
            )
        return self._boundary(plan.moe_mc_ar, shared, residual, norm_w, x2=routed, quant=quant)

    def _swiglu_mlp(
        self, h: torch.Tensor, prefix: str, low_m_gemm: bool, push: Optional[tuple] = None
    ):
        """A gated MXFP8 MLP (the dense MLP, the shared expert): gate_up, SwiGLU-OAI, down, no reduction. Returns
        (output, whether the down projection's GEMV pushed into ``push``)."""
        core = self.core
        gu, _ = _linear(core, h, f"{prefix}_gu_w", f"{prefix}_gu_s", low_m_gemm)
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

    def _routed(self, i: int, h: torch.Tensor, do_finalize: bool):
        """Router + routed experts: the fused router with the NVFP4 activation up to 48 rows, else the router logits and
        the experts quantizing their bf16 input."""
        core = self.core
        fused = _router_fp4(core, i, h)
        if fused is not None:
            router_logits, x_fp4, x_sf = fused
            return _routed_experts(core, i, x_fp4, x_sf, router_logits, do_finalize)
        router_logits = _router_logits(core, i, h)
        if not do_finalize:
            return _routed_experts(core, i, h, None, router_logits, False)
        return self._routed_chunks(i, h, router_logits)

    def _moe(
        self, i: int, h: torch.Tensor, *, return_parts: bool, do_finalize: bool = True, push=None
    ):
        """The MoE block of layer ``i`` without its reduction.

        ``push`` = (the engine's ``arp`` descriptor, the shared down projection's ``ar`` tuple) the decode layer armed
        for the MoE boundary's exchange sets, or None. Returns ``(out, routed, engine_pushed, shared_pushed)``: with
        ``return_parts`` the shared and routed parts (routed None when the engine folded the shared expert in), else
        their sum in ``out``; with ``do_finalize=False`` ``out`` is ``[shared, fc2 out, expert scales, permuted
        index]``.
        """
        core = self.core
        engine_arp, shared_ar = push if push is not None else (None, None)
        rows = h.shape[0] if h.dim() == 2 else 0
        if (
            do_finalize
            and h.dim() == 2
            and h.dtype == torch.bfloat16
            and 1 <= rows <= MOE_ENGINE_MAX_TOKENS
            and rows >= MOE_FOLD_MIN_TOKENS
        ):
            engine = core.engine(i, rows, shared=True)
            if engine is not None:
                _record("m3_moe_engine", rows, engine_arp is not None)
                folded = m3_moe_engine(h.contiguous(), engine, None, engine_arp)
                return folded, None, engine_arp is not None, False
        routed_out, (shared_out, shared_pushed) = maybe_execute_in_parallel(
            lambda: self._routed(i, h, do_finalize),
            lambda: self._swiglu_mlp(h, f"l{i}_sh", False, shared_ar),
            core.moe_events[i][0],
            core.moe_events[i][1],
            # The warmup tuning pass profiles the shared expert's GEMMs: keep it on the main stream, between the same
            # collectives on every rank (the capture pass and serving still overlap it on the aux stream).
            None if flashinfer_mxfp8_autotune_active() else core.aux_stream_moe,
            disable_on_compile=True,
        )
        if not do_finalize:
            return [shared_out, *routed_out], None, False, shared_pushed
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
            out, routed, _, _ = self._moe(i, hidden, return_parts=_can_mc(hidden, folded=True))
        else:
            out, _ = self._swiglu_mlp(hidden, f"l{i}_mlp", True)
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
            # Dense layers 0-2: QKV, the fused norm + RoPE (FP8 q/k/v: an FP8 cache, or the FP8 pool the dense layers
            # sit in within a hybrid NVFP4 cache on a decode step), the MSA backend, o_proj.
            qkv, _ = _linear(core, hidden, f"l{i}_qkv_w", f"l{i}_qkv_s", True)
            out_fp8 = core.main_kv_is_fp8 or core.main_kv_is_nvfp4
            fused = _qk_norm_rope(core, i, qkv, position_ids, out_fp8)
            q, k, v = fused.split([core.q_size, core.kv_size, core.kv_size], dim=-1)
            o = self._attention_core(i, q, k, v, None, step)
            return _linear(core, o, f"l{i}_o_w", f"l{i}_o_s", True, push)
        if not core.fused_projection:
            return self._unfused_sparse_attention(i, position_ids, hidden, step, push)
        # Gathered before the QKV projection so nothing runs between it and the fused attention (PDL hand-off).
        fused = self._fused_attn_args(i, position_ids, step)
        if fused is not None:
            mega = _attn_mega(core, i, hidden, fused, push)
            if mega is not None:
                return mega
        packed, _ = _linear(core, hidden, f"l{i}_qkv_w", f"l{i}_qkv_s", True)
        if fused is not None and (
            packed.shape[0] != fused["num_tokens"] or packed.shape[1] != 2560
        ):
            missed(
                FUSED_DECODE_ATTN.path,
                fused["num_tokens"],
                f"the packed QKV is {tuple(packed.shape)}, not [rows, 2560]",
            )
            fused = None
        if fused is not None:
            c1 = _attn_c1(core, i, packed, fused, push)
            if c1 is not None:
                return c1
            o = _fused_attn(core, i, packed, fused)
            return _linear(core, o, f"l{i}_o_w", f"l{i}_o_s", True, push)
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

    def _fused_attn_args(self, i: int, position_ids, step) -> Optional[dict]:
        """Arguments of the fused decode attention for this step, or None when it does not apply.

        The layer must be an NVFP4 sparse MSA layer and the step a pure generation step whose selection the metadata
        staged as the identity (every request within 16 blocks of 128 tokens), with at most 8 query tokens per request.
        """
        core = self.core
        path = FUSED_DECODE_ATTN.path
        if i not in core.sparse_layers or not core.main_kv_is_nvfp4:
            uncovered(path, "not an NVFP4 sparse MSA layer with an FP8 indexer")
            return None
        if position_ids is None:
            uncovered(path, "not a pure generation step with positions")
            return None
        if not step["msa_short_ok"]:
            uncovered(
                path, "a request beyond 16 KV blocks: the block selection is not the identity"
            )
            return None
        span = step["msa_decode_span"]
        if span is None or span.row_first != 0:
            uncovered(path, "no decode span starting at row 0")
            return None
        num_reqs = int(step["num_seqs"])
        qlen = int(span.query_len)
        if not FUSED_DECODE_ATTN.covers((qlen, num_reqs)):
            uncovered(
                path,
                f"more than {FUSED_ATTN_MAX_QUERY_LEN} query rows per request or {FUSED_ATTN_MAX_TOKENS} per step",
            )
            return None
        manager = step["kv_cache_manager"]
        if manager is not None and not manager.is_nvfp4_layer(i):
            uncovered(path, "the layer's KV cache is not NVFP4")
            return None
        if _fused_attn_kernel._module is None:
            # Building compiles every geometry (minutes): only outside capture (the eager warmup steps).
            if torch.cuda.is_current_stream_capturing():
                capture_missed(path, num_reqs * qlen, "the fused attention module")
                return None
            _fused_attn_kernel._load_module()
        if manager is None:
            missed(path, num_reqs * qlen, "no KV-cache manager")
            return None
        k_scale, v_scale = _aligned_nvfp4_dequant_scales(core.attn[i], core.kv_scales[i])
        return dict(
            num_tokens=num_reqs * qlen,
            decode_query_len=qlen,
            kv_data=manager.get_buffers(i, kv_layout="HND").view(torch.uint8),
            kv_scale=manager.get_block_scale_buffers(i, kv_layout="HND"),
            index_k_cache=step["msa_idx_k_cache"](i),
            out_cache_loc=step["msa_out_cache_loc"],
            position_ids=position_ids.reshape(-1).contiguous().to(torch.int32),
            block_table=step["msa_block_table"][:num_reqs],
            seq_lens=step["msa_seq_lens_cuda"][:num_reqs],
            kv_quant_scale=core.inv_kv_scales[i],
            k_global_scale=k_scale,
            v_global_scale=v_scale,
        )

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

    def _decode_sparse_core(self, i: int, q, idx_q, step, output) -> None:
        """Top-16 block selection and the block-sparse decode of one pure generation step of a sparse layer whose caches
        the producer already wrote.

        The selection is the identity when the metadata staged it (every request within 16 blocks). Otherwise the CuTe
        DSL scorer writes the per-index-head block maxima, reduced to the KV head, and the long-row selector takes the
        top 16 (the built-in selector beyond its shapes).
        """
        core = self.core
        backend = core.attn[i]
        config = backend.m3_config
        num_tokens = int(q.shape[0])
        rows = int(step["num_seqs"])
        qlen = int(step["msa_decode_span"].query_len)
        block_table = step["msa_block_table"][:rows]
        seq_lens = step["msa_seq_lens_cuda"][:rows]
        num_kv_heads = int(backend.indexer.config.num_kv_heads)
        identity = step["msa_short_ok"] and step["msa_short_table"] is not None
        if identity:
            topk = step["msa_short_table"][: num_kv_heads * num_tokens * MSA_REQUIRED_TOPK].view(
                num_kv_heads, num_tokens, MSA_REQUIRED_TOPK
            )
        else:
            idx_q_view = idx_q.view(num_tokens, config.num_index_heads, config.sparse_index_dim)
            idx_k_cache = step["msa_idx_k_cache"](i)
            max_score = step["msa_proxy_max_score_view"](
                config.num_index_heads, step["msa_worst_case_max_k_tiles"], num_tokens
            )
            early = rows <= INDEX_SCORE_EARLY_MAX_REQUESTS
            key = (
                idx_q_view.dtype,
                int(config.num_index_heads),
                int(qlen),
                int(idx_q_view.shape[-1]),
                int(INDEX_SCORE_SPLIT_K),
                bool(early),
            )
            if key not in _index_score_kernel.IndexScoreRunner.kernel_cache:
                # Outside require mode the op below compiles it (the only scorer this step has).
                missed(
                    INDEX_SCORE.path,
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
                INDEX_SCORE_SPLIT_K,
                early,
            )
            n_valid_blocks = step["msa_n_valid_blocks"][:num_tokens]
            group = config.num_index_heads // config.num_kv_heads
            max_score_kv = (
                max_score.view(
                    config.num_kv_heads, group, max_score.shape[1], max_score.shape[2]
                ).amax(dim=1)
                if group > 1
                else max_score
            )
            table = long_select_blocks(
                max_score_kv,
                n_valid_blocks=n_valid_blocks,
                init_blocks=config.init_blocks,
                local_blocks=config.local_blocks,
                use_pdl=get_env_enable_pdl(),
                early_trigger=num_tokens <= SELECT_EARLY_MAX_ROWS,
            )
            if table is None:
                uncovered(
                    LONG_SELECT.path,
                    f"{max_score_kv.shape[1]} KV blocks, outside {LONG_SELECT_MIN_BLOCKS}..{LONG_SELECT_MAX_BLOCKS}: "
                    "the built-in selector",
                )
                table = m3_select_blocks(
                    max_score_kv,
                    MSA_REQUIRED_TOPK,
                    n_valid_blocks,
                    config.init_blocks,
                    config.local_blocks,
                    True,
                )
            topk = table.permute(1, 0, 2)
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
        _record("m3_sparse_decode", num_tokens, qlen, identity)
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


# --------------------------------------------------------------------------------------------------------------------
# Speculative decoding: one-model Eagle3 at checkpoint precision
# --------------------------------------------------------------------------------------------------------------------


class M3SpecStepFold:
    """The target's speculative step-input staging (the in-tree ``SpecStepFold`` protocol).

    Between two decode graphs of a one-model Eagle3 step, the spec sampler stores the step's outputs into its per-slot
    store and the engine gathers them back into its flat input buffers for the next step. This object does both in
    one launch each (``m3_spec_tail.slot_scatter`` / ``step_gather``): integer copies only, the same values the
    engine's and the sampler's own copies write. The forward never stores the sampler's outputs itself, so
    ``begin_step`` and ``sampler_rows_stored`` always report that nothing was staged.
    """

    def __init__(self) -> None:
        self.store: Optional[object] = None
        self.step_inputs: Optional[tuple] = None

    def bind_engine_buffers(
        self, input_ids, draft_tokens, position_id_offsets, kv_lens_offsets, max_batch_size
    ):
        self.step_inputs = (input_ids, draft_tokens, position_id_offsets, kv_lens_offsets)

    def bind_sampler_store(self, store: object) -> None:
        self.store = store

    def begin_step(self, seq_slots: Optional[list], num_rows: int) -> bool:
        return False

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
        return False

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


class M3Eagle3Draft(nn.Module):
    """The one-layer BF16 Eagle3 drafter (``speculative_model``): its weights (``weights.DRAFT_WEIGHTS``), its
    vocab-parallel BF16 head and logits processor, the drafter layer's TRTLLM attention backend (layer index = the
    target's layer count, FP8 KV) and its TP all-reduce.

    What the worker reads: ``model`` (the layer forward), ``apply_eagle3_fc`` (the per-layer fc_norms + fc over the
    captured target hidden states), ``lm_head`` and ``logits_processor``; the fused draft loop reads the layer's parts
    (``draft_input_weights``, ``attention``, ``mlp``, ``ar_norm``). The drafter embeds with the target's table.
    """

    def __init__(
        self,
        core: MiniMaxM3Tp4ModelingV2,
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
            raise NoTarget(
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
            raise NoTarget(
                f"the drafter's attention backend {type(self.attn).__name__} must fuse RoPE and QKV"
            )
        self.attn.update_quant_config(quant_config)
        # The drafter embeds with the target's table, read off the core on every call (the engine materializes the
        # core's weights after construction).
        self._core = [core]
        self.model = _DraftForward(self)

    @property
    def embed_table(self) -> torch.Tensor:
        return self._core[0].w["embed"]

    def load_weights_from_target_model(self, target_model: nn.Module) -> None:
        """Nothing to alias: the drafter reads the target's embedding table through the core and owns its head."""

    # ------------------------------------------------------------------------------------------------- the layer

    def fc(self, hs: torch.Tensor) -> torch.Tensor:
        return m3_bf16_linear(hs, self.w["fc"])

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
        False, where the caller fuses the reduction)."""
        qkv = m3_bf16_linear(x, self.w["qkv"])
        _record("m3_draft_attention", int(md.num_tokens))
        o = m3_draft_attention(self.attn, qkv, md)
        out = m3_bf16_linear(o, self.w["o"])
        return m3_allreduce(out, self.allreduce, None) if reduce else out

    def mlp(self, h: torch.Tensor, reduce: bool) -> torch.Tensor:
        gu = m3_bf16_linear(h, self.w["gu"])
        act = m3_silu_and_mul(gu, None, None, None)
        out = m3_bf16_linear(act, self.w["dn"])
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


class M3Eagle3Worker(Eagle3OneModelWorker):
    """``Eagle3OneModelWorker`` with the MiniMax-M3 fused verify and fused draft loop, at the in-tree numerics.

    * Verify: on decode-only batches whose verify rows the fused kernels cover (``tail_eligible``), the target's BF16
      lm_head computes this rank's vocabulary shard of the logits (the in-tree head's own local GEMM; at most
      ``TARGET_HEAD_MAX_ROWS`` verify rows stream a lossless 12-bit coding of the same BF16 weight) and one kernel
      replaces the logits all-gather, the argmax (or the temperature + top-p draw) and the strict acceptance.
    * Draft: the drafter's first pass may take the per-layer fc_norms in one launch (or folded into the hidden-state
      capture) and run its fc beside the target head GEMM (forked right after the last captured target layer at most
      ``FC_EARLY_MAX_ROWS`` verify rows). On a decode-only step ``DRAFT_STEP`` covers, the draft loop is fused: each
      pass's draft input in one launch, the drafter layer's attention and MLP with their all-reduces fused with the
      residual add and the next RMSNorm, and one kernel for the BF16 head, the cross-rank argmax and the pass's
      bookkeeping.

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
        self._m3_target_head = None
        self._m3_force_vec_warm = False
        # The streamed copy of the drafter's BF16 head shard the fused draft loop reads, built before the warmup.
        self._m3_draft_head = None
        # Per step: the next step's input tokens [nreq, D + 1], column 0 written by the fused verify (or the seed) and
        # columns 1.. by the fused draft passes; set as _m3_next_new_tokens once the fused draft loop filled it.
        self._m3_next_new: Optional[torch.Tensor] = None
        self._m3_next_new_tokens: Optional[torch.Tensor] = None
        # Per step: the first draft inputs are the fused ones (accepted tokens as they lie, fc output of every row).
        self._m3_fused_first_inputs = False

    def capture_hidden_states(
        self, spec_metadata, layer_id: int, hidden_states, residual=None
    ) -> None:
        """The Eagle3 capture with this step's capture norm; on a step armed for the early fc fork
        (``arm_capture_norm``), the last captured layer then forks the drafter's first fc onto the aux stream, where
        it overlaps the remaining target layers and the head (joined by ``_m3_target_accept``)."""
        _eagle3_capture(spec_metadata, layer_id, hidden_states, residual, self._m3_capture_norm)
        drafter = self._m3_fc_early_drafter
        layers = spec_metadata.layers_to_capture
        if drafter is not None and layers and layer_id == layers[-1]:
            self._m3_fc_early_drafter = None
            self._m3_fc_early_stream = self._m3_fc_fork(
                spec_metadata, spec_metadata.num_capture_tokens, drafter
            )

    def prebuild(self, lm_head: nn.Module, draft: Optional[M3Eagle3Draft]) -> None:
        """The fused verify's coded target head and the fused draft loop's head copy, from the loaded weights,
        before the executor's warmup, on every rank."""
        start = time.perf_counter()
        if self.prebuild_target_head(lm_head):
            torch.cuda.synchronize()
            logger.info(
                "[M3 prebuild] coded target head built before warmup: %.1f s"
                % (time.perf_counter() - start)
            )
        start = time.perf_counter()
        if self.prebuild_draft_head(draft):
            torch.cuda.synchronize()
            logger.info(
                "[M3 prebuild] fused draft head built before warmup: %.1f s"
                % (time.perf_counter() - start)
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
                    hidden_states, spec_metadata, batch_size, runtime_draft_len, draft_model
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
        most DECODE_MAX_TOKENS verify rows. Host state only, so the TP ranks agree."""
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
        if draft_len == 0 or attn_metadata.num_seqs * (draft_len + 1) > DECODE_MAX_TOKENS:
            return False
        return not spec_metadata.wants_advanced_draft_sampling

    def tail_eligible(self, attn_metadata, spec_metadata, lm_head: nn.Module) -> bool:
        """Whether this step verifies with the fused kernels instead of the gathered target logits.

        Decode-only Eagle3 batches (``_m3_structural_ok``) over an unpadded vocab-sharded bias-free target head, all
        greedy at a request count ``SHARD_ACCEPT`` covers, or sampled under the temperature + top-p contract (see
        _m3_sampled_verify_ok), with nothing else reading the target logits. Every input is host state shared by all
        TP ranks, so the ranks agree.
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
        elif not self._m3_sampled_verify_ok(spec_metadata):
            return False
        self._m3_target_lm_head = lm_head
        return True

    def draft_only_eligible(self, attn_metadata, spec_metadata, draft_model) -> bool:
        """Whether a step verified by the in-tree path can still take the fused first drafter inputs: the structural
        conditions of tail_eligible, without the target-sampling ones."""
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
                hs = m3_spec_tail.chunk_norm(hs, cache[1], draft_model.rms_eps)
            hidden = draft_model.fc(hs)
        input_ids = accepted_tokens[:, : spec_metadata.runtime_draft_len + 1].reshape(-1)
        return {"input_ids": input_ids, "position_ids": position_ids, "hidden_states": hidden}

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
                hs = m3_spec_tail.chunk_norm(
                    hs, self._m3_fc_norm_w[1], draft_model.rms_eps, use_pdl=False
                )
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

        On a fused-verify step whose fc_norm weights are already stacked (by an earlier fused first pass), the
        capture writes rmsnorm(h + r) * fc_norm[i] (one kernel per captured layer instead of add + copy) and the fused
        first drafter inputs read it without their own chunk_norm. A fused-verify step of at most
        ``FC_EARLY_MAX_ROWS`` verify rows is also armed to fork the drafter's first fc at the last captured layer
        (``capture_hidden_states``).
        """
        self._m3_capture_norm = None
        self._m3_fc_early_drafter = None
        self._m3_fc_early_stream = None
        if draft_model is None:
            return
        if not self.tail_eligible(attn_metadata, spec_metadata, lm_head):
            return
        # The fork reads every captured column at the last captured layer, so a final-state capture, which is written
        # after the model returns, keeps the fc at its stock place.
        if (
            attn_metadata.num_seqs * (spec_metadata.runtime_draft_len + 1) <= FC_EARLY_MAX_ROWS
            and FINAL_HIDDEN_LAYER_ID not in spec_metadata.layers_to_capture
        ):
            self._m3_fc_early_drafter = draft_model
        fc_norm_w = self._m3_fc_norm_w
        if (
            fc_norm_w is not None
            and fc_norm_w[0] is draft_model.w["fc_norm0"]
            and len(spec_metadata.layers_to_capture) == fc_norm_w[1].shape[0]
        ):
            self._m3_capture_norm = (fc_norm_w[1], draft_model.rms_eps)

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
    ):
        """Fused verification (see tail_eligible); same outputs as the strict acceptance path."""
        force_total = 0
        force_vec = None
        if self.force_num_accepted_tokens != 0.0:
            is_warmup = spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing()
            if not torch.cuda.is_current_stream_capturing():
                # The draw pool is built by an H2D copy and the draw kernel JIT-compiles per block width, so both
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
                    self._m3_force_vec_warm = True
            if not is_warmup:
                force_total = min(int(self.force_num_accepted_tokens) + 1, draft_len + 1)
                force_vec = self._m3_force_vec(
                    batch_size, force_total, draft_len + 1, hidden_states.device
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
        head = self._m3_coded_head(lm_head, rows.shape[0])
        # The Eagle3 fc on the aux stream: the fork made at the last captured layer, else one beside the head. A
        # sampled verify's coded head holds every SM while it streams: its fc is forked after it instead (overlapping
        # the sampled acceptance), where it does not delay the head's start.
        fork = self._m3_fc_early_stream
        self._m3_fc_early_stream = None
        head_first = fork is None and head is not None and not spec_metadata.is_all_greedy_sample
        if fork is None and not head_first:
            fork = self._m3_fc_fork(spec_metadata, rows.shape[0], draft_model)
        # This rank's vocabulary shard of the target logits: the head's own local GEMM (or its coded copy), whose
        # all-gather the fused kernels replace (they exchange per-row records across ranks, not logits).
        if head is not None:
            logits = m3_target_head_kernel.logits(head, rows)
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

    def prebuild_target_head(self, lm_head: nn.Module) -> bool:
        """Build the 12-bit coded copy of the target head shard the fused verify streams at <= TARGET_HEAD_MAX_ROWS
        rows (rebuilt from the current weight values on every call), on every rank. True when the coded copy decodes
        to the BF16 weight bit for bit; otherwise (or for a head the fused verify does not read) nothing is kept and
        the head's own GEMM runs."""
        self._m3_target_head = None
        if not _sharded_head(lm_head) or not m3_target_head_kernel.head_supported(lm_head):
            return False
        head = m3_target_head_kernel.prepare_head(lm_head)
        self._m3_target_head = head
        if not head.coded:
            logger.warning(
                f"[M3 target head] the coded shard {tuple(lm_head.weight.shape)} does not decode to the BF16 weight "
                "bit for bit; the fused verify keeps the head's BF16 GEMM"
            )
        return head.coded

    def _m3_coded_head(self, lm_head: nn.Module, rows: int):
        """The coded target head for a fused verify of ``rows`` rows, or None for the head's own GEMM: above
        ``TARGET_HEAD_MAX_ROWS`` rows (uncovered), on a shard whose coding failed its round trip (uncovered: the GEMM
        is the exact path), and when the copy was not built for this weight (a miss)."""
        path = TARGET_HEAD.path
        if not TARGET_HEAD.covers(rows):
            uncovered(path, f"{rows} verify rows (the head's own GEMM runs)")
            return None
        head = self._m3_target_head
        if head is None or head.weight is not lm_head.weight:
            missed(
                path,
                rows,
                "the coded target head is not built for the head's current weight (prebuild)",
            )
            return None
        if not head.coded:
            uncovered(
                path, "the coded shard failed its bitwise round trip (the head's own GEMM runs)"
            )
            return None
        return head

    def _m3_force_vec(
        self, batch_size: int, base_total: int, max_total: int, device
    ) -> Optional[torch.Tensor]:
        """Per-request forced accepted counts for a fractional forced acceptance, else None.

        Draws exactly as _apply_force_accepted_tokens does for a decode-only batch (same fixed-seed pool, one counter
        bump per step, the same (counter, slot) hash), so every TP rank and the generic path agree on the counts.
        """
        frac = self.force_num_accepted_tokens - int(self.force_num_accepted_tokens)
        if frac <= 0.0 or base_total >= max_total:
            return None
        self._ensure_force_accept_rng_state(device)
        out = torch.empty(batch_size, dtype=torch.int32, device=device)
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
            max(16, 1 << (batch_size - 1).bit_length()),
        )

    # ---------------------------------------------------------------------------------------------- fused draft loop

    def prebuild_draft_head(self, draft_model: Optional[M3Eagle3Draft]) -> bool:
        """Build the streamed copy of the drafter's BF16 head shard that the fused draft loop reads (rebuilt from the
        current weight values on every call), on every rank. False when the drafter is not one the loop serves."""
        self._m3_draft_head = None
        if draft_model is None or not self._m3_drafter_supported(draft_model):
            return False
        self._m3_draft_head = m3_draft_step_kernel.prepare_head(
            draft_model.lm_head, self.mapping.tp_rank, DRAFT_HEAD_CODED
        )
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
        """The fused draft loop on a step it covers (``_m3_fused_loop_ok``), else the in-tree loop."""
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
        * the drafter layer's attention and MLP, each followed by the all-reduce with the residual add and the next
          RMSNorm (the post-attention norm, then the drafter's final norm);
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
                h = draft_model.attention(x, md, reduce=False)
                h, residual = draft_model.ar_norm(h, residual, "post_norm")
                h = draft_model.mlp(h, reduce=False)
                h, to_save = draft_model.ar_norm(h, residual, "final_norm")
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
        raise NoTarget(
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
    """The step-loop hooks this target opts into, all measured on GB300 tp4 AgentX decode.

    * ``pack_step_inputs``: a decode step's small input copies go out as one upload and one scatter at the graph
      replay instead of a chain of copies the next graph waits on (~30 us of idle GPU at one request).
    * ``skip_unchanged_h2d``: steady decode re-uploads slot indices, query lengths and page tables that rarely change.
    * ``loop_on_execution_stream``: the overlap loop's own stream work and the forward on one stream, so step N+1's
      inputs wait on graph N in program order rather than through two cross-stream event hops.
    * ``merge_tuned_tactics_across_ranks``: tensor-parallel peers capture the same GEMM tactics (rank-local profiling
      noise otherwise splits them, and the slowest rank sets every step).
    * ``skip_greedy_rng_upload``: all-greedy speculative batches never read the Philox seeds.
    * ``spec_step_fold``: the spec sampler's slot stores and the engine's next-step input gathers in one launch each.
    * MSA: identity selections for short sequences (and short/long graph families), the one-launch KV-length update,
      unchanged block-table copies skipped, pool views built once.
    """
    return RuntimeHooks(
        pack_step_inputs=True,
        skip_unchanged_h2d=True,
        loop_on_execution_stream=True,
        merge_tuned_tactics_across_ranks=True,
        skip_greedy_rng_upload=True,
        spec_step_fold=spec_fold,
        msa=MsaRuntimePolicy(
            short_sequence_identity=True,
            kv_update=_msa_kv_update_kernel.msa_kv_update,
            skip_unchanged_block_offsets=True,
            reuse_pool_views=True,
        ),
    )


class ModelingV2MiniMaxM3Nvfp4Sm103Tp4(
    DecoderModelForCausalLM[MiniMaxM3Tp4ModelingV2, PretrainedConfig]
):
    """MiniMax-M3 NVFP4 / sm_103 / tp4, text only, with one-model Eagle3 or no speculation.

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
            MiniMaxM3Tp4ModelingV2(model_config),
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
        spec_fold = None
        if spec_config is not None:
            (
                self.draft_config,
                self.draft_model,
                self.spec_worker,
                self.use_separate_draft_kv_cache,
            ) = _build_speculation(self, model_config)
            spec_fold = M3SpecStepFold()
        self.runtime_hooks = _runtime_hooks(spec_fold)

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
        speculation also the fused verify's coded target head and the fused draft loop's head copy."""
        super().cache_derived_state()
        self.model.post_load_weights()
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
        draft). Text only."""
        multimodal_params = kwargs.pop("multimodal_params", None) or []
        if any(p is not None and getattr(p, "multimodal_data", None) for p in multimodal_params):
            raise ValueError(
                "this MiniMax-M3 target is text only; the request carries multimodal input"
            )
        worker = self.spec_worker
        capture = _plain_capture
        if worker is not None:
            capture = worker.capture_hidden_states
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
        return worker(
            input_ids=spec_input_ids,
            position_ids=spec_position_ids,
            hidden_states=hidden_states,
            logits=logits,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=self.draft_model,
            resource_manager=resource_manager,
        )
