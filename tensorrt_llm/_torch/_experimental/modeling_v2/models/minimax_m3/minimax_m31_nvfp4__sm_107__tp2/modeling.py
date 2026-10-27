# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: MiniMax-M3.1 (MXFP8 + packed NVFP4 experts) / sm_107 / tp2 -- self-contained modeling code.

MiniMax-M3.1's language model has MiniMax-M3's shape (60 layers, hidden 6144, 64 query heads and 4 KV heads of dim
128, 128 routed experts top 4 plus one shared expert) with a 1e7 RoPE base and MSA block-sparse attention on EVERY
layer (an index branch scores 128-token blocks and the top 16 are attended); layers 0-2 keep the dense SwiGLU-OAI MLP,
layers 3-59 the MoE. The checkpoint is an MXFP8 export whose routed experts are packed NVFP4 (compressed-tensors
names, global scales stored as multipliers; ``weights`` takes their FP32 reciprocals); the KV cache is NVFP4 with an
FP8 index-K on every layer. The layer kinds are read from the config (``sparse_attention_freq``, ``moe_layer_freq``),
so this code is the MiniMax-M3 target's with no dense-attention layer in use.

``tp4`` is four Rubin GPUs (SM 10.7): head-split attention (16 query heads, one KV head and one index head per rank) and
tensor-parallel experts (every expert on every rank, intermediate 768 per rank), no attention DP. Speculative
decoding, when configured, is one-model Eagle3 (draft length 3): the BF16 drafter (``M3Eagle3Draft``) and the BF16
target head, both at checkpoint precision; decode steps of up to 8 requests verify and draft through fused kernels
at that precision (``M3Eagle3Worker``).

**The core** (``MiniMaxM3Tp2ModelingV2``) holds the weights (``weights.MODEL_WEIGHTS``) and the runtime resources the
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
next MXFP8 projection: the decode boundary all-reduce for the next layer's QKV, the sparse-decode merge and the
long-context decode for the o_proj. It publishes it in ``kernels/m3_prequant.SLOT`` keyed by the BF16 tensor; the next
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
import signal
import sys
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
    NamedTuple,
    Optional,
    Tuple,
    Union,
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
from tensorrt_llm._torch._experimental.modeling_v2.catalog.fakes import (
    m31_fake_ops,  # noqa: F401 (registers the fake impls of the M3.1 torch ops)
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
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import m31_pins
from tensorrt_llm._torch.attention.backends import AttentionMetadata
from tensorrt_llm._torch.attention.backends.fmha.msa_prefill import _aligned_nvfp4_dequant_scales
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionInputType,
    PositionalEmbeddingParams,
    RopeParams,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import MiniMaxM3MsaSparseAttention
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels import (
    nvfp4_index_k as _nvfp4_index_k,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
    MSA_REQUIRED_TOPK,
    msa_paged_kv,
    require_msa_module,
    write_msa_phase_kv,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.paged_cache import (
    fp8_e4m3_saturate,
)
from tensorrt_llm._torch.attention.backends.utils import create_attention
from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner
from tensorrt_llm._torch.distributed import (
    AllReduce,
    AllReduceFusionOp,
    AllReduceParams,
    AllReduceStrategy,
)
from tensorrt_llm._torch.distributed.ops import (
    MNNVLAllReduce,
    get_or_scale_allreduce_mnnvl_workspace,
)
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
from tensorrt_llm._torch.pyexecutor.sampler.ops.flashinfer import (
    set_softmax_override,
    set_top_p_renorm_override,
)
from tensorrt_llm._torch.speculative import SpecMetadata
from tensorrt_llm._torch.speculative.eagle3 import Eagle3OneModelWorker
from tensorrt_llm._torch.speculative.interface import (
    _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
    _FORCE_ACCEPT_RNG_POOL_SIZE,
    _FORCE_ACCEPT_RNG_SLOT_STRIDE,
    set_chain_speculative_sampling_override,
    set_chain_speculative_sampling_slot_override,
    should_use_separate_draft_kv_cache,
)
from tensorrt_llm._torch.utils import ActType_TrtllmGen, get_model_extra_attrs, is_torch_compiling
from tensorrt_llm._utils import maybe_pin_memory, mpi_rank, prefer_pinned
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

from . import dspark as _dspark
from . import weights as _weights
from .kernels import lc2_mixed as _lc2_mixed
from .kernels import m3_attn_lc, m3_fold_tail, m3_mc_allreduce, m3_prequant, m3_report, m3_spec_tail
from .kernels import m3_draft_lowm as _draft_lowm_kernel
from .kernels import m3_draft_step as m3_draft_step_kernel
from .kernels import m3_drafter_attn as _drafter_attn_kernel
from .kernels import m3_fullctx_decode as _fullctx_decode_kernel
from .kernels import m3_index_score as _index_score_kernel
from .kernels import m3_index_score_t2sc as _index_score_t2sc_kernel
from .kernels import m3_sampled_accept as m3_sampled_accept_kernel
from .kernels import m3_select_long_dsl as _select_long_dsl_kernel
from .kernels import m3_shard_accept as m3_shard_accept_kernel
from .kernels import m3_target_head as m3_target_head_kernel
from .kernels import m3_tokmap_guard as _tokmap_guard
from .kernels import msa_kv_update as _msa_kv_update_kernel
from .kernels import triton_sparse_decode as _sparse_decode_kernel
from .kernels.long_select import long_select_blocks
from .kernels.m3_moe.op import INTER as _MOE_INTER
from .kernels.m3_moe_finalize import m3_moe_finalize
from .kernels.m3_nvfp4_index_decode import Nvfp4IndexDecodeScorer
from .kernels.mx_gemv import op as mx_gemv
from .kernels.triton_sparse_decode import MERGE_MXFP8, MERGE_MXFP8_MIN_TOKENS
from .m31_switches import m31_disabled, m31_enabled
from .ops import gqamix_sparse_decode as _gqamix_kernel
from .ops.activation.m3_silu_and_mul import m3_silu_and_mul
from .ops.activation.m3_swq import m3_swq
from .ops.attention.m3_dense_decode import m3_dense_decode
from .ops.attention.m3_dense_qk_norm_rope_fp8 import m3_dense_qk_norm_rope_fp8
from .ops.attention.m3_draft_attention import m3_draft_attention
from .ops.attention.m3_fp8_indexer_qk_norm_rope import m3_fp8_indexer_qk_norm_rope
from .ops.attention.m3_fullctx_decode import m3_fullctx_decode
from .ops.attention.m3_index_score import SPLIT_K as INDEX_SCORE_SPLIT_K
from .ops.attention.m3_index_score import m3_index_score
from .ops.attention.m3_index_score_t2sc import m3_index_score_t2sc
from .ops.attention.m3_msa_attention import m3_msa_attention
from .ops.attention.m3_producer import m3_producer
from .ops.attention.m3_qk_norm_rope_fp8 import m3_qk_norm_rope_fp8
from .ops.attention.m3_qkv_insert import m3_qkv_insert
from .ops.attention.m3_select_blocks import m3_select_blocks
from .ops.attention.m3_select_long import m3_select_long
from .ops.attention.m3_select_long_dsl import m3_select_long_dsl
from .ops.attention.m3_sparse_decode import m3_sparse_decode
from .ops.attention.m3_sparse_decode_lc import m3_sparse_decode_lc, prepare_lc2_split_workspace
from .ops.comm.m3_allreduce import m3_allreduce
from .ops.comm.m3_mc_allreduce_norm import m3_mc_allreduce_norm
from .ops.comm.m3_mc_finalize_allreduce_norm import m3_mc_finalize_allreduce_norm
from .ops.embedding.m3_embedding import m3_embedding
from .ops.gemm import m3_mxfp8_cublaslt as mxfp8_cublaslt
from .ops.gemm import m31_agg_prefill_cublaslt as agg_prefill_cublaslt
from .ops.gemm.m3_bf16_linear import m3_bf16_linear
from .ops.gemm.m3_dense_swiglu import m3_dense_swiglu
from .ops.gemm.m3_draft_lowm import ROWS as DRAFT_LOWM_TACTIC_ROWS
from .ops.gemm.m3_draft_lowm import m3_draft_lowm
from .ops.gemm.m3_fp32_linear import m3_fp32_linear
from .ops.gemm.m3_mx_gemv import (
    DENSE_DOWN,
    DENSE_GATE_UP,
    GEMV48_MIN_TOKENS,
    GEMV48_SHAPES,
    O_PROJ,
    QKV,
    QKV_SPARSE,
    SHARED_DOWN,
    SHARED_GATE_UP,
    m3_mx_gemv,
)
from .ops.gemm.m3_mx_gemv_bf16 import m3_mx_gemv_bf16
from .ops.gemm.m3_mxfp8_cublaslt import m3_mxfp8_cublaslt
from .ops.gemm.m3_mxfp8_gemm import m3_mxfp8_gemm
from .ops.gemm.m3_mxfp8_linear import m3_mxfp8_linear
from .ops.gemm.m3_mxfp8_pinned_gemm import m3_mxfp8_pinned_gemm
from .ops.gemm.m3_mxfp8_pinned_quantize import m3_mxfp8_pinned_quantize
from .ops.gemm.m3_target_head import MAX_ROWS as TARGET_HEAD_KERNEL_MAX_ROWS
from .ops.gemm.m3_target_head import m3_target_head
from .ops.moe import m31_s1b_fold
from .ops.moe.m3_moe16_engine import M3Moe16State, m3_moe16_engine
from .ops.moe.m3_moe_engine import M3MoeState, ar_push_desc, m3_moe_engine, shared_ctas_for
from .ops.moe.m3_moe_tactic_pin import run_pinned as _run_pinned_moe
from .ops.moe.m3_router_tc import m3_router_tc
from .ops.norm.m3_embedding_firstnorm import DecodeEmbedding, m3_embedding_firstnorm
from .ops.sampling import m31_chain_spec_mc as _chain_spec_mc
from .ops.sampling import m31_softmax_mc as _softmax_mc
from .ops.sampling import m31_topp_det as _topp_det
from .ops.spec.m3_draft_input import m3_draft_input
from .ops.spec.m3_draft_silu import m3_draft_silu
from .ops.spec.m3_draft_step import MAX_TOKENS as DRAFT_STEP_KERNEL_MAX_TOKENS
from .ops.spec.m3_draft_step import m3_draft_step
from .ops.spec.m3_force_vec import m3_force_vec
from .ops.spec.m3_force_vec_req import m3_force_vec_req
from .ops.spec.m3_shard_accept import m3_shard_accept
from .ops.spec.m3_spec_offsets import m3_spec_offsets
from .ops.spec.m3_step_prologue import m3_step_prologue

_SM = (10, 7)  # capacity: fixed: the device capability, not a size
# PROXYPB: compile the MSA indexer proxy's fmha_sm100 variants before warmup (``_prebuild_msa_proxy``). Compile and
# load only, no numerics change. TRTLLM_M31_DISABLE_PROXYPB=1: each variant compiles on its first call instead (which
# can be after the server is ready).
MSA_PROXY_PREBUILD = os.environ.get("TRTLLM_M31_DISABLE_PROXYPB", "0").strip() in ("", "0")
# R6: the dense-MLP layer boundary also publishes the next layer's MXFP8 QKV activation (as the MoE boundary does):
# M3.1's layers 1-3 follow a dense MLP but have the sparse five-way QKV. TRTLLM_M31_DISABLE_R6=1 restores the BF16
# hand-off (the next QKV quantizes its own input).
# VR: OFF by default. On sm_107 the hand-off's QKV output is not bitwise against the BF16 path, and R6 has not passed
# the natural-acceptance gate (with R6 off, that gate passes).
# TRTLLM_M31_ENABLE_R6=1 turns it on; TRTLLM_M31_DISABLE_R6=1 still forces it off.
_R6_ENABLED = os.environ.get("TRTLLM_M31_ENABLE_R6", "0").strip() not in ("", "0")
_R6_DISABLED = os.environ.get("TRTLLM_M31_DISABLE_R6", "0").strip() not in ("", "0")
R6_DENSE_PREQUANT = _R6_ENABLED and not _R6_DISABLED
# CTXNORMQ: a prefill layer boundary of at least ``CTXNORMQ_MIN_ROWS`` rows runs the plain all-reduce, then
# one kernel for the residual add + RMSNorm that also writes the next MXFP8 projection's activation (and, at a MoE
# boundary, the routed experts' NVFP4 input), instead of the fused all-reduce + norm followed by the consumers' own
# quantize passes. The quantized outputs equal mxfp8_quantize / fp4_quantize of the norm bit for bit; the norm itself
# is not bitwise against the fused all-reduce's (reduction order). Opt-in: TRTLLM_M31_ENABLE_CTXNORMQ=1;
# TRTLLM_M31_DISABLE_CTXNORMQ=1 wins.
CTXNORMQ = m31_enabled("CTXNORMQ") and not m31_disabled("CTXNORMQ")
# TP2AGG_ROUTER_F32 (the tp2ctx NORMQ_F32 router fold): a MoE layer's CTXNORMQ boundary runs the op's _f32
# variant, which also writes the normed rows widened to fp32 (bit for bit norm.float()); the prefill router reads them
# instead of m3_fp32_linear's own x.to(float32) cast (~300 us at 16K rows on sm_107). Same GEMM operands, so the
# logits are bitwise today's. The op comes from the build or the overlay tensorrt_llm/libs/libm31_agg_ctxnormq_f32.so;
# without it the boundary keeps the base op. Kill switch: TRTLLM_M31_DISABLE_TP2AGG_ROUTER_F32=1.
ROUTER_F32 = not m31_disabled("TP2AGG_ROUTER_F32")
# F3: on TP2 decode-graph steps of F3_MIN_ROWS..F3_MAX_ROWS rows, the o_proj's MX-TAC pinned GEMM (FlashInfer's
# dense blockscaled GEMM, epilogue-only fork kernels/m31_f3_gemm) stores its rows straight into the attention
# boundary's Lamport one-shot slots (both ranks, the stock push's -0 word rule), and the boundary runs the one-shot AR
# + residual + RMSNorm without its push loop, also writing the MXFP8 of the norm (trtllm::minimax_m31_f3_ar_prepushed,
# the stock quantize's own device functions); the next pinned GENERIC MXFP8 linear (the shared expert / dense
# gate_up) takes it instead of quantizing (_F3_MX). Bitwise to the stock chain. Kill switch: TRTLLM_M31_DISABLE_F3=1.
# TRTLLM_M31_F3_CHECK=1: each F3 boundary also runs the stock GEMM + AR + quantize in the graph and counts byte
# mismatches on the device (reported at exit / on eager steps).
TP2_F3 = not m31_disabled("F3")
# Native CUDA ops for the SwiGLU-OAI activation and the MoE finalize, bit for bit the Triton kernels they replace on
# every band. NATIVE_SWIGLU: trtllm::minimax_m31_swiglu_oai_vec (16-byte vectors, the in-tree swigluOai pair sequence
# verbatim); kill switch TRTLLM_M31_DISABLE_NATIVE_SWIGLU=1. NATIVE_FINALIZE: trtllm::minimax_m31_moe_finalize (the
# Triton kernel's per-element fma chain); kill switch TRTLLM_M31_DISABLE_NATIVE_FINALIZE=1.
NATIVE_SWIGLU = not m31_disabled("NATIVE_SWIGLU")
NATIVE_FINALIZE = not m31_disabled("NATIVE_FINALIZE")
# True serves the native ops on every band, including the few where they measured slightly slower (the dense SwiGLU
# at 88..2048 rows, the finalize at 49..448 and 2K+ rows); False keeps the Triton kernels there.
NATIVE_ALL_BANDS = True
_NATIVE_STATE: dict = {}


def _native_op(name: str):
    """``torch.ops.trtllm.<name>``, or None when this build lacks it; decided once per process."""
    if name not in _NATIVE_STATE:
        op = getattr(torch.ops.trtllm, name, None)
        if op is None:
            logger.info(f"[M3.1] native op {name} not available: the Triton kernel serves")
        _NATIVE_STATE[name] = op
    return _NATIVE_STATE[name]


def _native_dispatch(item: str, site: str, why: str = "") -> None:
    """The backend line of ``item`` (NATIVE_SWIGLU / NATIVE_FINALIZE) at ``site``, once per (item, site, outcome) per
    process: "[M3 dispatch] rank r: <item> taken (<site>)" when the native op serves, "... not taken (<site>): <why>"
    when the caller's fallback does."""
    outcome = f"not taken ({site}): {why}" if why else f"taken ({site})"
    _log_once(logger.info, item, outcome, f"[M3 dispatch] rank {mpi_rank()}: {item} {outcome}")


F3_CHECK = os.environ.get("TRTLLM_M31_F3_CHECK", "0") == "1"
F3_MIN_ROWS = 49  # capacity: fixed: the o_proj GEMV serves 1..48 rows (no pinned GEMM there)
F3_MAX_ROWS = 256  # capacity: fixed: the pinned MXFP8 table's 256 bucket (tested at 49..224)
F3_N, F3_K = 6144, 4096  # capacity: fixed: the TP2 o_proj shape the fork is tested on

# TP2W_S2_TAIL (megakernel stage S2): a generic MoE layer's decode boundary of S2_TAIL_MIN_ROWS..S2_TAIL_MAX_ROWS
# rows runs ONE kernel (trtllm::minimax_m31_s2_moe_tail) for the
# routed finalize + shared add (m3_moe_finalize's math), the one-shot Lamport all-reduce + residual + RMSNorm
# (allreduce_fusion_kernel_oneshot_lamport, its launch configuration) and the next layer's QKV MXFP8 activation
# (mxfp8_quantize's math), instead of three kernels. Microbench: bitwise to that chain at 1..224 rows, -5.6..-6.5 us
# per layer. The activation is consumed only where the QKV projection would run
# mxfp8_quantize + the same GEMM (bit-identical operands). TRTLLM_M31_TP2W_S2_CHECK=1 (debug, never timed) also runs
# the stock boundary with the all-reduce the step would use and logs the bit differences. Enabled by default;
# TRTLLM_M31_DISABLE_TP2W_S2_TAIL=1 forces it off.
TP2W_S2_TAIL = not m31_disabled("TP2W_S2_TAIL")
S2_TAIL_MIN_ROWS = 49  # capacity: fixed: the Triton finalize's rows (past the MC finalize's 48)
S2_TAIL_MAX_ROWS = (
    224  # capacity: measured: microbench rows (the in-model one-shot Lamport takes up to 256)
)
S2_CHECK = os.environ.get("TRTLLM_M31_TP2W_S2_CHECK", "0") == "1"
# TP2AGG_QKV_A1 (the tp2ctx TP2CTX_QKV_FUSED A1 kernel, kernels/qkv_fusion verbatim): an eager sparse-layer step of
# more than A1_MIN_ROWS rows whose QKV activation CTXNORMQ published runs one CuTe DSL kernel for the QKV + index MXFP8
# GEMM with the horizontal producer in its epilogue, instead of the GEMM + m3_producer. The producer math is the stock
# producer's on the kernel's own bf16 GEMM output; the GEMM is deterministic but not bitwise against the stock MXFP8
# GEMM (agg microbench: <= 59 q bytes of 67M differ, one E4M3 step; -88 / -134 / -180 us per call at 8K / 12K / 16K
# rows). Not bitwise, so OFF until its det + GSM8K McNemar gates pass; TRTLLM_M31_ENABLE_TP2AGG_QKV_A1=1 opts in,
# TRTLLM_M31_DISABLE_TP2AGG_QKV_A1=1 forces it off.
QKV_A1 = m31_enabled("TP2AGG_QKV_A1") and not m31_disabled("TP2AGG_QKV_A1")
A1_MIN_ROWS = 1024  # capacity: tp2ctx _CTX_QKV_FUSED_MIN_ROWS (steps above it)
# TP2AGG_SP: the tp2ctx TP2CTX_SP recipe on agg's eager steps of
# 4096 < T <= 16384 rows. Each CTXNORMQ boundary becomes an IPC reduce-scatter of the partial to this rank's owned rows
# (rank 0 [0, S), rank 1 [S, T), S the 128-row boundary nearest T / 2), the residual add + RMSNorm + quantize on the
# owned rows written into the workspace's gather views, then a push of the owned products to the peer; the gathered
# full-T products are published to the CTXNORMQ slots, so the consumers are unchanged. The residual stream is
# owned-only inside such a step and is unsharded before an Eagle3 capture layer and the final norm. At the pre-MoE
# boundary the router logits of the owned rows (ROUTER_ZC, the fp32 input from TP2AGG_ROUTER_F32) are gathered as a
# "logits" kind (128 fp32 = 512 B per row) instead of tp2ctx's top-k, since trtllm-gen routes in-kernel. The workspace
# is catalog/comm/tp2ctx_ipc with the peer ops of libm31_tp2ctx_peer.so, built at weight load (_sp_init, both ranks);
# without them every step keeps the ordinary boundary ("not taken" logged once). Opt-in: TRTLLM_M31_ENABLE_TP2AGG_SP=1;
# TRTLLM_M31_DISABLE_TP2AGG_SP=1 or TRTLLM_M31_DISABLE_AGG_SP=1 disables both boundaries.
# TRTLLM_M31_DISABLE_AGG_SP_PREMOE=1 keeps only post-MoE SP, gathering the residual before pre-MoE.
# The per-item ids stay tp2ctx's (TP2CTX_ROUTER_ZC / _SP_ADDNORM / _SP_OVL / _SP_PMOVL).
TP2AGG_SP = (
    m31_enabled("TP2AGG_SP") and not m31_disabled("TP2AGG_SP") and not m31_disabled("AGG_SP")
)
SP_PREMOE = m31_enabled("AGG_SP_PREMOE") and not m31_disabled("AGG_SP_PREMOE")
SP_MIN_ROWS = 4097  # capacity: eager agg SP band; the workspace also covers the Lamport overlap
SP_MAX_ROWS = 16384  # capacity: max_num_tokens of the agg configs
SP_SF_TILE_ROWS = 128  # rank 1's rows start on a 128-row MXFP8 scale-factor swizzle tile
SP_LOGITS_ROW_BYTES = 128 * 4  # the "logits" gather kind: [T, 128] fp32
_SP_STATE: dict = {}
# TRTLLM_M31_TP2CTX_SP_CHECK=1 (debug, never timed): each owned reduce-scatter is bit-compared with the all-reduce's
# rows, and the keyed bf16 view (no data) is filled with NaN, so a consumer reading it instead of the products shows.
SP_CHECK = os.environ.get("TRTLLM_M31_TP2CTX_SP_CHECK", "0") == "1"
_SP_LIB = "libm31_tp2ctx_peer.so"
_SP_NORMQ_OUT_LIB = "libm31_ctx_normq_out.so"
_SP_ADDNORM_LIB = "libm31_sp_addnorm.so"
# SP_ADDNORM: the owned-row peer add fused into the NORMQ out= launch (bitwise the two kernels). SP_PMOVL: the pre-MoE
# products pushed by the peer push kernel on a side stream under the router GEMM (scheduling only). SP_OVL: before a
# sparse layer whose QKV runs TP2AGG_QKV_A1, the owned MXFP8 rows go by copy engine on a side stream while A1 runs on
# the own rows (A1 is row-split bitwise; the stock MXFP8 GEMM is not guaranteed, so it waits for the push).
# Each is opt-in (TRTLLM_M31_ENABLE_<ID>=1); TRTLLM_M31_DISABLE_<ID>=1 wins.
SP_ADDNORM = m31_enabled("TP2CTX_SP_ADDNORM") and not m31_disabled("TP2CTX_SP_ADDNORM")
SP_PMOVL = m31_enabled("TP2CTX_SP_PMOVL") and not m31_disabled("TP2CTX_SP_PMOVL")
SP_PMOVL_MIN_ROWS = 12288  # capacity: tp2ctx measurement (A8K +1.6% below it)
SP_OVL = m31_enabled("TP2CTX_SP_OVL") and not m31_disabled("TP2CTX_SP_OVL")
SP_OVL_MIN_ROWS = 12288  # capacity: tp2ctx _SP_OVL_MIN_ROWS
# GQAMIX: a mixed ctx+gen step's sparse generation rows (MsaDecodeFmha) run the Triton block-sparse decode
# through ops/gqamix_sparse_decode, a verbatim copy of the in-tree kernel module whose two decode kernels do not
# specialize on the step's row count or on the [num_ctx:] slices' alignment, registered on MsaDecodeFmha's
# sparse_decode_hook. Those specializations compiled new variants after the server was ready. Kill switch:
# TRTLLM_M31_DISABLE_GQAMIX=1.
GQAMIX = not m31_disabled("GQAMIX")
# Eager prefill tile selection uses a deterministic score-descending, block-ascending
# tie order. Opt-in: TRTLLM_M31_ENABLE_AGG_TSEL=1; TRTLLM_M31_DISABLE_AGG_TSEL=1 restores the built-in selector.
AGG_TSEL = m31_enabled("AGG_TSEL") and not m31_disabled("AGG_TSEL")
AGG_TSEL_MIN_ROWS = 224
AGG_TSEL_MAX_BLOCKS = 5856  # the op's shared-memory bound (kMinimaxM3SelectPrefillTileMaxBlocks)
AGG_TSEL_ORACLE = m31_enabled("AGG_TSEL_ORACLE")
# TP2WIDE_PREFILL_ATTN (ported from the tp2ctx target's TP2CTX_SPARSE_OUT / TP2CTX_SCHED_LPT): the NVFP4 sparse
# prefill's MSA combine writes the attention output in place (msa_prefill: msa_sparse_out_in_place; bitwise the copied
# result), and its k2q CSR schedule runs the work items heaviest first (msa_prefill: msa_k2q_schedule_order "lpt"; the
# order only changes which CTA runs which item, never what an item computes or where it writes). Each applies only when
# the loaded MSA build accepts it. With the shared msa_prefill's E4M3X forward (kill switch
# TRTLLM_M31_DISABLE_TP2CTX_E4M3X=1), the agg 16K chunk at a 96K prefix runs 909.7 -> 530.1 us per call, output bitwise
# (microbench). Kill switches: TRTLLM_M31_DISABLE_TP2WIDE_SPARSE_OUT=1, TRTLLM_M31_DISABLE_TP2WIDE_SCHED_LPT=1.
TP2WIDE_SPARSE_OUT = not m31_disabled("TP2WIDE_SPARSE_OUT")
TP2WIDE_SCHED_LPT = not m31_disabled("TP2WIDE_SCHED_LPT")
# NVFP4_STEP_CACHE (AGG_NVFP4_INDEXK): the NVFP4 decode reader's history scheduler is built once per step, not per
# sparse layer (same values). Kill switch: TRTLLM_M31_DISABLE_NVFP4_STEP_CACHE=1.
NVFP4_STEP_CACHE = not m31_disabled("NVFP4_STEP_CACHE")
# TP2WIDE_COMBINE_MXFP8 (ported from the tp2ctx target's TP2CTX_COMBINE_MXFP8, LongAttn): with the in-place
# output, the sparse prefill's MSA combine also writes o_proj's MXFP8 input (msa_prefill: msa_sparse_out_mxfp8_sink;
# bitwise mxfp8_quantize(out, True)) when the loaded build takes o_fp8 / sf_out (the E4M3X / BF16X fork's combine, or
# the old NVFP4 sparse func), and publishes it in the CTXNORMQ slot, so o_proj's prefill MXFP8 linear skips its own
# quantize (the GEMM runs on the same operands). Calls under _COMBINE_MXFP8_MIN_ROWS rows keep the stock quantize.
# Needs CTXNORMQ (the slot's consumer). Kill switch: TRTLLM_M31_DISABLE_TP2WIDE_COMBINE_MXFP8=1.
TP2WIDE_COMBINE_MXFP8 = not m31_disabled("TP2WIDE_COMBINE_MXFP8")


def _prefill_attention_opt_ins(backends) -> None:
    """Set the MSA prefill opt-ins (TP2WIDE_SPARSE_OUT, TP2WIDE_SCHED_LPT, TP2WIDE_COMBINE_MXFP8) on every layer's
    attention backend."""
    for backend in backends:
        if TP2WIDE_SPARSE_OUT:
            backend.msa_sparse_out_in_place = True
            if TP2WIDE_COMBINE_MXFP8 and CTXNORMQ:
                backend.msa_sparse_out_mxfp8_sink = _combine_mxfp8_sink
        if TP2WIDE_SCHED_LPT:
            backend.msa_k2q_schedule_order = "lpt"


# DSFOLD: under the standalone DSpark worker, the engine's previous-batch input gathers (input ids, draft tokens,
# position / KV-length offsets) run as one m3_spec_tail.step_gather launch instead of ~a dozen eager index and copy
# launches at the step boundary. Integer copies of the same values into the same buffers; the sampler keeps its
# in-tree slot stores. Kill switch: TRTLLM_M31_DISABLE_DSFOLD=1.
DSFOLD = not m31_disabled("DSFOLD")
CTXNORMQ_MIN_ROWS = (
    1024  # capacity: NORMQ microbench (GB300): the fused pass wins at 1160 and 16384 rows
)
# --------------------------------------------------------------------------------------------------------------------
# Row ranges
# --------------------------------------------------------------------------------------------------------------------

# The largest generation step the decode target takes: 8 requests verifying 1 + 3 drafts and their padding rows.
# R1VR8 (the sm_107 TP8 target's row-bound kill, kept at TP2): the decode layer takes pure generation steps up to
# DECODE_MAX_TOKENS rows. TRTLLM_M31_DISABLE_R1VR8=1 restores the TP2-lineage 8-row bound (TP2_ROWS): every wider step
# runs the generic layers (and VR_BANDS is off with it), as at TP8. Read once, at import.
TP2_ROWS = 8  # capacity: fixed: the TP2-lineage paths' certified rows
R1VR8 = not m31_disabled("R1VR8")
DECODE_MAX_TOKENS = 48 if R1VR8 else TP2_ROWS  # capacity: fixed: the decode kernels' step bound
# VR-BANDS (MIDCONC): the decode step bound is 64 rows (8 requests verifying 1 + 7 drafts, the K7 c8 step) while every
# op keeps its own certified row range (DECODE_MAX_TOKENS and the per-op tables; the shard accept keeps its 48 rows):
# an op past its range runs its generic form and the rest of the step stays on the decode layer. Steps of 49..64 rows
# take the "wide" plan (the built-in all-reduce at both boundaries). TRTLLM_M31_DISABLE_VR_BANDS=1 restores the
# 48-row step bound. Read once, at import.
VR_BANDS = R1VR8 and not m31_disabled("VR_BANDS")
TP2WIDE_DECODE = VR_BANDS and not m31_disabled("TP2WIDE_DECODE")
DECODE_STEP_MAX = (
    448 if TP2WIDE_DECODE else 64 if VR_BANDS else DECODE_MAX_TOKENS
)  # capacity: fixed: the step bound; ops keep their own rows
# SWQ (the sm_107 TP8 lever, ported): on the shared-expert down projection's rows the decode GEMV does not serve
# (SHARED_DOWN (6144, 1536) at 17..32 rows, MX_GEMV_ROWS below) the SwiGLU-OAI writes the down projection's MXFP8
# activation directly (ops/activation/m3_swq.py: bitwise to silu_and_mul + mxfp8_quantize on every block that quantize
# emits finite; a scale-code-0 block, which it emits as inf / NaN, is kept finite), taken by ``_linear``'s prequant
# branch into the pinned MXFP8 GEMM. Compiled before warmup (prebuild). Kill switch TRTLLM_M31_DISABLE_SWQ=1 (the bf16
# SwiGLU + quantize). Read at import.
SWQ = not m31_disabled("SWQ")
SWQ_SHARED_DOWN = (6144, 1536)  # the shared expert's down projection at the TP2 width 1536
SWQ_ROWS: Dict[Tuple[int, int], FrozenSet[int]] = (
    {SWQ_SHARED_DOWN: frozenset(range(17, min(32, DECODE_MAX_TOKENS) + 1))} if SWQ else {}
)  # capacity: fixed: the shared down's rows past its GEMV (17..32)
_SWQ_LOGGED: set = set()


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
# verify; verify: the c4 / c8 verifies. Each is one multicast all-reduce bucket (kernels/m3_mc_allreduce.BUCKETS).
# SM 10.7 TP4: one bucket. MC_AR_TP4 puts every decode boundary on the multicast all-reduce + residual + RMSNorm
# (trtllm::minimax_m3_sm107_tp4_mc_allreduce_norm, its row kernel: bitwise to the stock TP4 oneshot all-reduce fusion
# on the residual and the normed rows) with the MXFP8 copy for the next QKV; False = the built-in all-reduce.
MC_AR_TP4 = True
# Up to this many rows the decode all-reduces (not the MoE finalize) take the 8-CTA cluster kernel instead of the row
# kernel: the same stock oneshot order (opts bit 32: per-column sum of squares, warp butterflies, 6-warp groups, 4 group
# sums from 0.f), so bitwise the row kernel's output, with each row's exchange spread over 8 CTAs. Rubin TP4 synth, both
# orders: c1 (4 rows) -12, c2 (8 rows) -38 us/step, but c4 / c8 (16 / 32 rows) +5 / +45, where
# 8x the CTAs per row cost more than they save. 0 = the row kernel at every row count.
# TP2 port (world-2 microbench): oneshot_cluster off at every row count (bit 32 is 0.4 us slower at world 2);
# the call still runs the 8-CTA cluster kernel because the TP4 op's row kernel takes world 4 only.
MC_AR_CLUSTER_MAX_ROWS = 0  # capacity: fixed: TP2: no oneshot-cluster rows
# Every decode / drafter boundary all-reduce launches with the max-smem carveout preference (the cluster kernel's
# option 8 at every row count, the row kernel's carveout instance), so the next kernel's CTAs (the QKV GEMV's staged
# tiles) can co-reside early. Launch attribute only: bitwise unchanged. Kill: False (the GB option table).
MC_AR_CARVEOUT = True
# The fused draft loop's boundary all-reduces (M3Eagle3Worker._m3_draft_ar_norm) launch WITHOUT that carveout: the
# kernel after the drafter's last boundary is the draft step (one CTA per SM, ~145 KB of shared memory) rather than a
# staged GEMV, and TP4 alone of the sm_107 targets ran its draft pass ~74-80 us in-model vs ~57 us standalone at world 4
# in the step census. Launch attribute only: bitwise unchanged. Kill: MC_AR_CARVEOUT (the shared setting).
DRAFT_MC_AR_CARVEOUT = False
# The fused draft loop's two all-reduce boundaries per pass (o_proj -> post-attention norm, down -> final norm) of up
# to this many rows run the multicast all-reduce + residual + RMSNorm's row kernel in its drafter form (opts bit 16:
# FlashInfer fused_add_rmsnorm's reduction order on the unrounded FP32 sum, so bitwise the drafter's stock plain
# all-reduce + RMSNorm) instead of the built-in all-reduce and the separate norm. Kill: 0.
DRAFT_MC_AR_MAX_ROWS = (
    8 if MC_AR_TP4 else 0
)  # capacity: fixed: the drafter boundary kernel's certified rows
# Those drafter boundaries run the 8-CTA cluster kernel's drafter form (opts bit 128; TP8's drafter boundary kernel)
# instead of the 1-CTA row kernel. Exact-class: deterministic, the updated residual bitwise, the RMS sum of
# squares in the cluster's order instead of FlashInfer's, so a normed row can differ by one BF16 rounding step (7/64
# rows at 1 row, 14/256 at 4). Rubin world 4, both orders: 4.39 -> 3.75 us per boundary at 1 row,
# 4.79 -> 4.14 at 4 rows, ~ -2..-3.5 us/step. Kill: False (the row kernel, bitwise the stock boundary).
# OFF: in-model timing was flat and natural acceptance length was grey (lower bound -1.16% vs the -1% line).
DRAFT_MC_AR_CLUSTER = False
# The drafter's o_proj and down projection, on the copied split-K kernel (kernels/m3_lowm_splitk), also push their rows
# into the drafter boundary all-reduce's exchange (the row kernel's slots: the canonical words its own push would store,
# at the (token, 768-column slice) call parity), which then runs prepushed 2 instead of pushing x itself; the copy
# releases its dependents only after its own grid wait and the pushes (the prepushed >= 2 contract). Bitwise. The TP8
# port: ~ -4 us/step at world 8. Kill: False.
DRAFT_AR_PUSH = True
# The drafter boundary all-reduce fed by its projection's push (DRAFT_AR_PUSH): its own taken / not-taken lines.
DRAFT_AR_PUSH_PATH = "m3_draft_ar_push"
# TP2 port: the M3 sm_107 tp2 target's row-bucket skeleton (tiny 1..8 / b16 9..16 / b48 17..DECODE_MAX_TOKENS, the
# multicast all-reduce's workspace buckets) with the TP4 base's one setting on every bucket, so the plan (and every
# boundary) is the TP4 base's at every row count; the TP2 WIDE stage sets each bucket's flags / cells on its own.
TP2_BUCKETS = tuple(
    b for b in (8, 16, 48) if b <= DECODE_MAX_TOKENS
)  # capacity: fixed: the plan buckets (MC buckets)
DECODE_PLANS = tuple(
    DecodePlan(name, rows, attn_mc_ar=MC_AR_TP4, moe_mc_ar=MC_AR_TP4, qkv_prequant=MC_AR_TP4)
    for name, rows in zip(("tiny", "b16", "b48"), TP2_BUCKETS)
)
if DECODE_STEP_MAX > DECODE_MAX_TOKENS:
    # VR-BANDS: 49..64-row steps (K7 c7..c8) with the built-in all-reduce at both boundaries.
    DECODE_PLANS += (DecodePlan("wide", 64, attn_mc_ar=False, moe_mc_ar=False, qkv_prequant=False),)

if TP2WIDE_DECODE:
    # Separate bucket preserves every existing plan through 64 rows. Each op keeps its own dispatch range.
    DECODE_PLANS += (
        DecodePlan("wide448", 448, attn_mc_ar=False, moe_mc_ar=False, qkv_prequant=False),
    )

# The dense layers' decode attention (trtllm-gen, msa_decode.dense_decode_max_seq_len): the max_seq_len it is handed
# per request count. It only sizes the multi-CTA KV split (every CTA still covers its share of the actual length; the
# result changes only by the split-merge order), and the model maximum over-splits long-context caches at one
# request: GB300 TP4, 16 heads, FP8 KV, 64K context: 15.1 us at 16K vs 28.6 us at 1M. Two requests are neutral (the
# grid is unchanged) and keep the step's bound.
# The generic layer's boundaries (prefill, mixed steps, Eagle3 capture layers): the most rows that take the multicast
# all-reduce + residual + RMSNorm, alone and with the shared + routed add folded in. TP4 skeleton: none (the built-in
# all-reduce) until the multicast all-reduce is certified on Rubin.
GENERIC_MC_AR_MAX_TOKENS = 0  # capacity: fixed: off at TP4 (the built-in all-reduce)
GENERIC_MC_AR_FOLDED_MAX_TOKENS = 0  # capacity: fixed: off at TP4 (the built-in all-reduce)
# An Eagle3 capture layer whose capture norm is armed (CAPTURE_NORM) has its fused feed-forward boundary all-reduce
# (the 8-CTA cluster kernel up to MC_AR_CLUSTER_MAX_ROWS, the row kernel above) also write the capture:
# rmsnorm(updated residual) * fc_norm[i], bit for bit the capture_norm kernel, which then does not run. The MoE
# finalize runs the shared GB300 op and keeps the capture_norm kernel. Wired at the generic layer's fused boundary
# (layers.M3DecoderLayer._feed_forward_boundary) and through M3DecodeTarget._layer's boundaries (``capture``) for a
# capture layer on the fast decode layer. Kill: False (the capture_norm kernel after the boundary).
CAPTURE_NORM_IN_BOUNDARY = True

# TP4 skeleton: no cap (the built-in split) until a TP4 SM 10.7 cap is measured and certified.
DENSE_DECODE_MAX_SEQ_LEN_BY_BATCH: Mapping[int, int] = MappingProxyType({})

# The dense layers' decode attention on PrimTS split-KV decode (kernels/m3_fullctx_decode) instead of the trtllm-gen
# call, per (query rows per request, requests): the Q tile (``config``) and the split-KV fanout (``splits``) with the
# BF16 merge. At TP4 (16 query heads, 1 KV head per rank) the stock split is capped at 16 cluster CTAs per (request,
# KV head), i.e. 16 of 212 CTAs at one request: stock dense fmhaSm107a 33.4-33.8 us per call and the stock drafter
# fmhaSm100f 45.4 us per pass at 165k. Rubin TP4, FP8 KV, 165k, chained us per call
# (microbench), every recipe passing the fp32 oracle: (4, 1) 13.70 (stock 33.6), (1, 1) 11.52. At one
# request the K/V stream is latency-bound, so both deepen the K/V load pipeline (the load pipeline only; the same
# bits): 12 stages at one query row, 8 at four (the long-context sweep: (4, 1) 12.95 / 17.00 / 20.98 us
# per call at 165k / 323k / 512k with 8 stages vs 13.03 / 17.15 / 21.30 with 12; (1, 1) is fastest with 12). Two
# requests (microbench; stock at c2: dense fmhaSm107a 26.5 us, drafter fmhaSm100f 44.0-44.6 us per pass,
# census): (4, 2) 17.34, (1, 2) 14.16.
# K/V stages in flight of the (4, 1) recipe (12 was the previous depth; the same bits at any depth).
FULLCTX_DECODE_Q4_KV_STAGES = 8  # capacity: fixed: a pipeline depth, not a size
FULLCTX_DECODE_RECIPES: Mapping[Tuple[int, int], Mapping[str, Any]] = MappingProxyType(
    {
        (4, 1): MappingProxyType(
            {
                "config": {
                    "tile_size_q": 32,
                    "groups_tokens_heads_q": True,
                    "kv_stages": FULLCTX_DECODE_Q4_KV_STAGES,
                },
                "splits": 96,
            }
        ),
        (1, 1): MappingProxyType({"config": {"kv_stages": 12}, "splits": 108}),
        (4, 2): MappingProxyType(
            {"config": {"tile_size_q": 32, "groups_tokens_heads_q": True}, "splits": 48}
        ),
        (1, 2): MappingProxyType({"config": {}, "splits": 80}),
    }
)  # capacity: fixed: pinned recipes per certified (rows, requests)
# Kill constant: False puts the dense layers back on the trtllm-gen call at every shape.
FULLCTX_DECODE_ENABLED = True
# The dense layers' FP8 q/k/v (decode steps of the hybrid cache): the op minimax_m3_sm107tp4_dense_qk_norm_rope_fp8
# (head-count generic) rounds Q/K to BF16 and then to E4M3 in its store, the bits of the BF16 fused_qk_norm_rope output
# cast to FP8. False (or a build without the op) keeps the single-rounding fused_qk_norm_rope FP8 output.
DENSE_QKV_FP8_VIA_BF16 = True
# On a full-context decode step that op also stores the new K/V rows into the layer's paged pool (the bytes
# write_msa_phase_kv's indexed copies store, one launch instead of about ten). False writes them with
# write_msa_phase_kv.
DENSE_KV_INSERT_FUSED = True
# The engagement line of that fused store (taken, one per row count and rank).
DENSE_KV_INSERT_PATH = "m3_dense_kv_insert"
# The drafter MLP's C++ SwiGLU (DRAFT_SILU_CPP, M3Eagle3Draft.mlp): its own taken / not-taken lines.
DRAFT_SILU_PATH = "m3_draft_silu"
# The split-KV merge on the BF16 op (minimax_m3_sm107tp4_fullctx_reduce, head-count generic) instead of PrimTS's FP16
# reducer and a cast.
FULLCTX_DECODE_OWN_REDUCER = True
# The merge also writes the MXFP8 copy of its BF16 rows (E4M3 + 128x4-swizzled UE8M0, bit for bit
# trtllm::mxfp8_quantize) into the MERGE_MXFP8 buffers and publishes it to m3_prequant.SLOT, so the dense layer's
# MXFP8 o_proj skips its own quantization. Kill: False leaves the o_proj to quantize its BF16 input.
FULLCTX_DECODE_MXFP8 = True
FULLCTX_DECODE_MXFP8_PATH = "m3_fullctx_merge_mxfp8"
# The K/V length bound the full-context plans are compiled for; a step whose cache length bound is longer is a miss. It
# covers the long-context serving config's max_seq_len (1 << 20) plus the speculative rows the cache reserves past it.
FULLCTX_DECODE_MAX_KV_LEN = (
    1 << 21
)  # capacity: fixed: the plans' KV-length bound (1M context + spec rows)
# The Eagle3 drafter layer's generation phase on the same full-context decode (kernels/m3_drafter_attn: the in-tree
# RoPE + FP8 K/V append, then the pinned PrimTS recipe) instead of thop.attention's in-tree runner, which caps its
# split at 8 CTAs per (request, KV head) on SM 10.7: 45.4 us per draft pass at 165k. Kill constant.
DRAFTER_FULLCTX_ENABLED = True
# The standalone DSpark drafter's attention on FlashInfer's trtllm-gen library (kernels/m3_drafter_attn
# M3DrafterSlidingFiFmha): its SM100f Q8 SwapsAb decode kernel instead of thop.attention's in-tree SM107a Q64 KeepsAb
# one (4.0 vs 9.7 us per call, 5 calls per step at c1). A different kernel, so not bitwise the in-tree output.
# Kill switch: TRTLLM_M31_DISABLE_VR_DRAFT_FMHA_SM100F=1 (the in-tree runner), read at import.
# Installed only where the drafter's decode reads its attention modules' FMHA list. The standalone DSpark drafter's
# block decode (DFlash) calls its own attention ops (FlashInfer trtllm_batch_decode_with_kv_cache for TRTLLM), so the
# library installed there did no work (no VR_DRAFT_FMHA_SM100F count at TP2; a TP8 kill A/B was flat);
# the prebuild logs "not installed" with that reason instead (the lever needs the drafter's own call site: deferred).
VR_DRAFT_FMHA_SM100F = not m31_disabled("VR_DRAFT_FMHA_SM100F")
# On those passes the drafter's RoPE, FP8 query and FP8 K/V append run as one launch (the op
# minimax_m3_sm107tp4_drafter_rope_fp8_kv: the in-tree preprocess's GPT-NeoX rotation and unit-scale E4M3 store, and
# the torch E4M3 cast of its BF16 query) instead of thop.trtllm_gen_generation_preprocess and the cast. Kill constant.
DRAFTER_ROPE_FUSED = True
# The engagement line of that launch (taken, one per row count and rank).
DRAFTER_ROPE_PATH = "m3_drafter_rope"
# thop.trtllm_gen_generation_preprocess builds its workspace token map only for several requests, yet a one-request
# multi-token generation (the drafter's first pass at one request, any stock trtllm-gen fallback) places its rows by
# that map: without a rebuild it applies the last multi-request call's map (rows on another request's length and pages,
# token-0 drafts). kernels/m3_tokmap_guard writes the one-request map (0, t) into the op's map slice right before such
# calls: what the build writes (padded rows included), so bitwise the op with a fresh map. Kill: False.
PREPROCESS_TOKMAP_FIX = True
PREPROCESS_TOKMAP_PATH = "m3_preprocess_tokmap"
# How the rotation's products are summed, as the in-tree preprocess build sums them (0: two products then the add,
# 1: fma(cos, x, sin * pair), 2: fma(sin, pair, cos * x)); the op's bitwise test pins it.
DRAFTER_ROPE_MODE = 1  # capacity: fixed: a launch mode, not a size

# The sparse layers' NVFP4 QKV + indexer producer (ops/attention/m3_producer, every step its geometry fits) triggers its
# dependent kernel on entry, so the index scorer launches and stages its pre-wait loads while the producer runs (its
# outputs are read only after the scorer's own grid wait).
PRODUCER_EARLY_TRIGGER = True
# Steps of up to this many rows keep the producer's trigger at its exit instead. On Rubin TP4 the early-launched scorer
# (its grid doubles with the requests, its pre-wait index-K prefetch running) is resident through the producer's whole
# post-QKV segment and slows the producer's dependent loads; from 16 rows the overlap pays. In-model, locked memory
# clock, tactic- and FlashInfer-pinned, A and B in one job: -75 / -176 us per decode step at 1 / 2 requests,
# +67 at 4 with the trigger at the exit. Timing only: the scorer reads the producer's outputs after its own grid wait.
# Kill: 0 (the early trigger at every row count). Set to 0 on the shipped stack: with the long-row select's late release
# and post-load trigger, the scorer's pre-wait overlap with the producer is worth more than the producer's own
# segment (both orders, double-pinned: -12 / -42 us at 1 / 2 requests with 0).
PRODUCER_LATE_TRIGGER_MAX_ROWS = 0  # capacity: fixed: measured trigger crossover (0: off)
# The producer reads its per-step inputs (position ids, cache slots, RoPE rows) before its grid wait, leaving only the
# packed QKV row after it: this target's copy of the producer (trtllm::minimax_m3_sm107_tp2_producer_pw,
# minimaxM3Sm107Tp4/producerPw4.cu, the TP2 producerPw2 at 16 query heads). Their writers are host-to-device copies
# before the replay and plain-launched kernels earlier in the step (the input preprocessing, msa_kv_update or its torch
# fallback), so they are final before the producer can start; a PDL-launched writer of any of them would have to turn
# this off. The call site's packed.contiguous() and position-id copies are plain launches too. Bit-identical outputs.
# Kill: False (the shared minimax_m3_producer_pw at 16 query heads, which reads them after its wait).
PRODUCER_PREWAIT_STEP_INPUTS = True
# QKV_PRODUCER_FOLD (the sm_107 TP8 target's #117 at the TP2 heads): on a decode step of at most 8 rows whose QKV GEMV
# runs split 8 (``mx_gemv_cfg``) the sparse layer's QKV + index GEMV runs the producer in its owner epilogue
# (kernels/mx_gemv m3_mx_gemv_producer, kernels/mx_gemv/phase_producer.producer_head): one launch and one PDL edge
# fewer per sparse layer, bit for bit the GEMV + m3_producer chain (q, index-q and every cache byte). The QKV linear
# defers its GEMV (``_linear``'s ``defer``: a ``_DeferredGemv``) and the producer call site runs the fused kernel, or
# the plain GEMV when the producer does not take the step. TP2 c1 microbench (60-layer graphs, cold L2, 2 processes):
# producer removed -1.358 us/layer, fold epilogue +1.052, net -0.306 us/layer. Kill switch
# TRTLLM_M31_DISABLE_TP2_QKV_PRODUCER_FOLD=1 (the GEMV, then m3_producer), read at import.
QKV_PRODUCER_FOLD = not m31_disabled("TP2_QKV_PRODUCER_FOLD")
QKV_PRODUCER_FOLD_PATH = "m3_qkv_producer_fold"

# The fused embedding + first RMSNorm kernel: one owner per row count and index dtype.
EMBED_FIRSTNORM_MAX_TOKENS = 8  # capacity: fixed: built rows; wider steps embed generically
# The o_proj pushes its rows into the attention boundary's multicast all-reduce (prepushed mode 2) up to this many
# tokens; wider steps run the GEMV and a plain multicast all-reduce.
ATTN_PUSH_MAX_TOKENS = 8 if MC_AR_TP4 else 0  # capacity: fixed: measured push crossover
# The dense layers' MLP down projection pushes its rows into the post-MLP boundary's multicast all-reduce (prepushed
# mode 2), as the o_proj does into the attention boundary (TP2: those hops end 5.5-5.8 us after the down
# GEMV unpushed vs 2.4-2.5 us pushed). Bitwise equal to the unpushed boundary (the same bf16 words). Kill: False.
DENSE_PUSH_TP4 = True
DENSE_PUSH_MAX_TOKENS = (
    8 if (MC_AR_TP4 and DENSE_PUSH_TP4) else 0
)  # capacity: fixed: the GEMV push rows
AR_PUSH_MODE = 2  # capacity: fixed: an exchange mode, not a size
# The 1..8-token MoE engine (router, routed experts and the shared expert in one launch). It pushes its sum into the
# MoE boundary's multicast all-reduce (prepushed mode 2) on the rows MOE_PUSH covers (with MC_AR_TP4).
MOE_ENGINE_MAX_TOKENS = 8  # capacity: fixed: the engine's built rows
MOE_PUSH_MAX_TOKENS = (
    MOE_ENGINE_MAX_TOKENS if MC_AR_TP4 else 0
)  # capacity: fixed: the engine's rows
# The 9..13-row engine (router, routed and shared experts in one kernel); its bf16 sum goes to the boundary.
# MOE16_TP2: the 9..13-row engine is the TP4 kernel, built and certified at the TP4 expert
# width (768); at the TP2 width (INTER 1536) it is untested, so OFF in the baseline: 9..13-row steps run the generic
# MoE inside the decode layer (no engine is built, MOE16_ENGINE covers no rows). TRTLLM_M31_ENABLE_MOE16_TP2=1 opts in
# (after its UT at 1536 and the natural-acceptance gate); TRTLLM_M31_DISABLE_MOE16_TP2=1 always wins. Read at import.
MOE16_TP2 = m31_enabled("MOE16_TP2") and not m31_disabled("MOE16_TP2")
# The width the 9..13-row engine's kernel is compiled for (kernels/m3_moe/m3_moe16_kernel.py I_TP / SH_I, SH_SPLIT 2):
# an opted-in build at another expert width fails (the generic MoE runs) until a TP2-width kernel exists.
MOE16_KERNEL_WIDTH = 768  # capacity: fixed: the TP4-width build of m3_moe16_kernel
MOE16_MIN_TOKENS = 9  # capacity: fixed: the 9..13-row engine's lower bound
MOE16_MAX_TOKENS = 13 if MOE16_TP2 else 0  # capacity: fixed: built rows (0: off at TP2)
# From this many tokens the MoE runs unfinalized through the TRTLLM-Gen runner and its finalize (top-k gather,
# routing weights, shared expert add) runs inside the boundary all-reduce. 9..13 take it only when their engine is
# missing (outside require mode). TP4 skeleton: no rows (the built-in MoE finalize and all-reduce).
# Fold the unfinalized MoE output into the boundary all-reduce within its certified 48-row bucket.
# The parent kill switch disables the fold at every row count.
TP2W_MC_FINALIZE = not m31_disabled("TP2W_MC_FINALIZE") and DECODE_MAX_TOKENS >= 48
# Include rows past the MoE engines by default; the small-row kill restores the 33-row lower bound.
TP2_MC_FINALIZE_SMALL = not m31_disabled("TP2_MC_FINALIZE_SMALL")
# capacity: fixed: past the MoE engines' rows (or the WIDE rows' 33 with the kill) .. the finalize op's 48-token bucket
MC_FINALIZE_MIN_TOKENS = (
    (max(9, MOE16_MAX_TOKENS + 1) if TP2_MC_FINALIZE_SMALL else 33) if TP2W_MC_FINALIZE else 9
)
MC_FINALIZE_MAX_TOKENS = 48 if TP2W_MC_FINALIZE else 0
# TP2W_MOE_FINALIZE (TP2 WIDE; port of the M3 tp2 target's MOE_TRITON_FINALIZE): a finalized MoE forward of at least
# MOE_TRITON_FINALIZE_MIN_TOKENS rows in one runner call (the decode "wide" plan's 49..64 rows, the generic layers past
# them) takes the trtllm-gen runner's FC2 output unfinalized and kernels/m3_moe_finalize finalizes it with the shared
# expert added, bitwise to the runner's finalize + shared.add_ (CTX microbench at 48..192 rows, dropped slots, graph ==
# eager; 8.3 -> 3.1 us per layer at 64 rows). Kill switch TRTLLM_M31_DISABLE_TP2W_MOE_FINALIZE=1 (the runner finalizes).
TP2W_MOE_FINALIZE = not m31_disabled("TP2W_MOE_FINALIZE")
# Reuse the CTXNORMQ boundary's NVFP4 product for ordinary BF16 prefill rows instead of re-quantizing it.
# SP keyed views always consume the slot, independently of this switch, because they contain no BF16 payload.
# The debug oracle skips graph captures and SP keyed views. Kill: TRTLLM_M31_DISABLE_TP2W_FP4SLOT_FINALIZE=1.
TP2W_FP4SLOT_FINALIZE = not m31_disabled("TP2W_FP4SLOT_FINALIZE")
_FP4SLOT_CHECK = os.environ.get("TRTLLM_M31_TP2W_FP4SLOT_CHECK", "0") == "1"
_FP4SLOT_STATE = {"taken": 0, "checked": 0, "diff": 0}
MOE_TRITON_FINALIZE_MIN_TOKENS = 49  # capacity: fixed: past the MC finalize's 48 rows
# A greedy Eagle3 verify step of up to this many requests verifies with the fused shard argmax + acceptance kernel
# over the target lm_head's BF16 shard logits (kernels/m3_shard_accept), at the configured draft length.
SHARD_ACCEPT_DRAFT_LEN = 3  # capacity: fixed: the built Eagle3 draft length
SHARD_ACCEPT_MAX_REQUESTS = DECODE_MAX_TOKENS // (
    SHARD_ACCEPT_DRAFT_LEN + 1
)  # capacity: fixed: decode rows / (draft length + 1)
# A sampled (temperature + top-p, advanced_sampling_mode=no_topk) Eagle3 verify step of up to this many requests
# verifies with the fused sampled acceptance (kernels/m3_sampled_accept) over this rank's logits shard: every request
# count a decode step holds.
SAMPLED_VERIFY_MAX_REQUESTS = SHARD_ACCEPT_MAX_REQUESTS  # capacity: fixed: a decode step's requests
# The sampled verify runs the sm_107 tp4 copy of the TP2 sampled-accept kernel
# (trtllm::minimax_m3_sm107tp4_sampled_accept: a row's CTAs launch as one thread-block cluster, bounded record polls).
# Same records and tokens. Kill: False (the shared kernel, trtllm::minimax_m3_sampled_accept: every CTA of the grid
# co-resident, unbounded polls).
SAMPLED_ACCEPT_SM107TP4 = True
# The sm_107 tp4 sampled verify loads each row's sampling parameters (temperature, top-p, Philox seed and offset: step
# inputs, not the head's output) before its PDL dependency wait, overlapping the head GEMM's tail; the same values and
# tokens. Kill: False (they load after the wait; the shared kernel always loads them after). Rubin TP2 microbench:
# verify end 20.48 -> 19.97 us at top-p 0.95.
SAMPLED_ACCEPT_PREWAIT_PARAMS = True
# The sm_107 tp4 sampled verify releases its dependents (the first draft pass's input) right after its dependency wait
# instead of at the acceptance epilogue: the dependent launches early and parks at its own griddepcontrol.wait, which
# returns only once the verify has completed, so the values are the same; it saves the launch latency after the verify
# (TP2: draft_input started 2.0-2.4 us after the verify / fc join). Kill: False. Off until its sampled-column
# in-model pairs read.
SAMPLED_ACCEPT_EARLY_TRIGGER = False
# SA-1R: the sm_107 tp4 sampled verify also multicasts each CTA's largest logits with its round-0 record (82: each
# thread's top 2 pooled, the CTA's top 8 plus the bound of everything it does not list; 9: an exact top-9 list), so a
# row whose first candidate is provably in the nucleus decides in round 0 instead of exchanging a second round. The
# same tokens, accepted counts and Philox draws as the two-round kernel, bit for bit (a fp32 margin m * 2^-21 * S
# covers the summation order; any doubt runs round 1). The prototype was exact on 39,968 rows; real
# long-context serving logits decide all 4 rows in round 0 on 79.5% (82) / 80.8% (9) of steps. Off (0, the two-round
# kernel): exact, but the untimed decode step reads +47 / +65 us slower with it on at TP4 (the earlier dependent
# launch it enables costs more than the saved round).
SAMPLED_ACCEPT_1R_K = 0  # capacity: fixed: a kernel variant, not a size
# The sm_107 tp4 sampled verify's owner walk (the one thread that finds its row's drawn token by a serial
# prefix over its strided tokens) takes those tokens 8 at a time: their weights are computed independently, then
# summed in the same order with the same early exit, so the draw, tokens and Philox draws are the one-token walk's,
# bit for bit. Rubin TP4 bench (one request, top_p 0.95): -1.06 / -1.07 us per call at SA-1R K 0 / 82. Kill: False
# (the one-token walk, the stock instantiation; the shared kernel has no 8-token walk).
# Off until its in-model TP4 pair passes.
SAMPLED_ACCEPT_ILP_WALK = False
# A step verified by the in-tree path takes the fused first drafter inputs (the capture norms and the first pass's
# gather) up to this many requests: every request count a decode step holds.
FIRST_DRAFTER_INPUTS_MAX_REQUESTS = SHARD_ACCEPT_MAX_REQUESTS  # capacity: fixed: step requests
# A fused-verify step of up to this many verify rows (one or two requests at draft length 3) computes the target
# logits shard from the 12-bit coded copy of the BF16 head (kernels/m3_target_head; the same BF16 x BF16 -> fp32
# arithmetic, 75% of the bytes) instead of the in-tree head GEMM; wider verifies keep the in-tree GEMM.
# Off at TP4 (as at TP2): at the TP4 shard (50,016 x 6144) the in-tree BF16 cuBLAS head is faster than the coded kernel,
# 47.9 vs 56.8 us at 4 and 8 rows (TP2: 93 vs 109). True re-enables the coded head.
TARGET_HEAD_CODED = False
TARGET_HEAD_MAX_ROWS = (
    TARGET_HEAD_KERNEL_MAX_ROWS if TARGET_HEAD_CODED else 0
)  # capacity: fixed: the coded head's built rows (off at TP4)
# On a fused-verify step of up to this many verify rows the drafter's first fc is forked onto the aux stream right
# after the last captured target layer, overlapping the remaining target layers and the head (M3Eagle3Worker).
FC_EARLY_MAX_ROWS = 0  # capacity: fixed: measured fork crossover (0: off)
# A sampled fused verify of up to this many verify rows on the in-tree head GEMM forks the drafter's first fc after the
# head instead of beside it (M3Eagle3Worker), as the coded head's verify already does: the fc's split-K grid otherwise
# holds the SMs first and the head GEMM starts late (TP2: 17.4 us late). Kill: 0.
FC_AFTER_HEAD_MAX_ROWS = 8  # capacity: fixed: measured fork-after-head crossover
# A decode-only Eagle3 step of up to this many requests drafts with the fused draft loop (M3Eagle3Worker): per pass the
# draft input in one launch (kernels/m3_spec_tail.draft_input), the drafter layer's in-tree attention, MLP and
# all-reduces, and one kernel for the draft head (its 12-bit coded copy, or the BF16 fragments when the coding does
# not decode bit for bit), the cross-rank argmax and the pass's bookkeeping (kernels/m3_draft_step), at the
# configured draft length. The head kernel takes one row per request, at most 8.
# The fused draft loop streams the 12-bit coded copy of the draft head (when it decodes bit for bit); False keeps the
# BF16 fragments.
DRAFT_HEAD_CODED = False
DRAFT_STEP_DRAFT_LEN = SHARD_ACCEPT_DRAFT_LEN  # capacity: fixed: the built draft length
DRAFT_STEP_MAX_REQUESTS = DRAFT_STEP_KERNEL_MAX_TOKENS
# The Eagle3 capture layers (decoder layers whose output the drafter reads) run the fast decode layer, with its
# o_proj / MoE / dense-down pushes into the boundary all-reduces, and capture the MoE / dense boundary's reduced
# residual sum (the same BF16 words the generic layer's fused boundary captures). Kill: False (the generic layer,
# no boundary pushes). Rubin TP2: the capture layers' hops end 4.4-5.3 us after their producing
# GEMV unpushed vs 2.4-2.5 us pushed.
CAPTURE_LAYERS_FAST = True
# An Eagle3 capture layer (one with a next norm) ends at the fused boundary (all-reduce + residual + next RMSNorm in
# one kernel) and captures that kernel's reduced residual sum. Kill: False keeps the feed-forward module's own
# all-reduce, captures hidden + residual and runs the next norm separately, as the built-in model does; it also
# turns CAPTURE_LAYERS_FAST off, which captures the fused boundary's sum.
CAPTURE_FUSED_BOUNDARY = True
# The Eagle3 drafter's BF16 linears (qkv, o_proj, gate_up, down, fc) run FlashInfer's fixed-tactic low-M kernels
# (ops/gemm/m3_draft_lowm, the TP2 target's port with the TP4 table) at the rows its table is certified for, up to this
# many rows; other calls keep the in-tree cuBLAS GEMM (split-K + reduce). Kill: 0. Rubin TP4 sweep (locked memory
# clock) at 1 row: qkv 8.15 -> 5.90, o 5.36 -> 3.40, gate_up 12.93 -> 9.93, down 7.68 -> 6.16 us.
DRAFT_LOWM_MAX_ROWS = 4  # capacity: fixed: the low-M tactic table's certified rows
# The drafter MLP's activation (the plain SwiGLU of its in-tree GatedMLP) runs this target's C++ op
# (ops/spec/m3_draft_silu, minimaxM3Sm107Tp4/siluAndMul.cu) instead of the Triton silu_and_mul: bit for bit, one launch
# either way (no Triton on the generation decode path). The fused draft loop's MLP (M3Eagle3Draft.mlp) routes the
# call; the in-tree draft path keeps the Triton op. Kill: False.
DRAFT_SILU_CPP = True
DRAFT_LOWM_ROWS: Dict[Tuple[int, int], FrozenSet[int]] = {
    nk: frozenset(m for m in rows if m <= DRAFT_LOWM_MAX_ROWS)
    for nk, rows in DRAFT_LOWM_TACTIC_ROWS.items()
}
# The identity-selection sparse decode moves from the per-row producer to the multi-query producer at this many rows
# (kernels/triton_sparse_decode.MQ_MIN_TOKENS).
SPARSE_DECODE_MQ_MIN_TOKENS = 17  # capacity: fixed: the multi-query crossover
# The long-row top-16 selector takes engines of 129..LONG_SELECT_MAX_BLOCKS KV blocks of 128 tokens; it holds 2, 4 or
# 9 keys per thread by the row's valid-block count. At most 128 blocks the built-in selector runs.
LONG_SELECT_MIN_BLOCKS = 129  # capacity: fixed: past the built-in selector's 128
LONG_SELECT_MAX_BLOCKS = 0  # capacity: fixed: off at TP4 (the DSL select takes the long rows)
LONG_SELECT_KEY_WIDTHS = (2, 4, 9)  # capacity: fixed: compiled keys per thread
# The long-row select releases the sparse decode after it right after its own grid wait (the decode stages q and the
# request length before its wait) up to this many query rows. On Rubin TP4 the early-launched attention CTAs crowd the
# select and the o_proj GEMV already at 2 requests (8 rows), so the early release is off at every row count: in-model,
# locked memory clock, pinned tactics, long-context draws, 2 instances vs the stored baseline, the replay period with
# the trigger phase below is -209 us per step at 2 requests (mean of draws), -39 at 4 and -12 at 8.
SELECT_EARLY_MAX_ROWS = 0  # capacity: fixed: measured early-release crossover (0: off)
# N1 at TP2 (with TP2WIDE_LC2, ported from the M3 sm_107 TP2 target): the long-context sparse decode (LC2,
# SPARSE_DECODE_LC) records each step's selected block ids per sparse layer and, before its grid wait (while the select
# ranks), pulls into L2 the K/V pages + scales of the blocks the SAME layer selected one step back (65-68% of a layer's
# selected blocks recur one step later, measured). Same layer, not the next one as at TP4: the MoE between two
# sparse layers streams ~250 MB through the 126 MB L2, so an earlier issue would be evicted. Hints only (no value
# changes); up to N1_MAX_ROWS query rows. Replaces the TP4 base's LC4 knobs (merge-once op, o_proj / next-layer KV
# prefetch, identity steps), which never ran at TP2 (LC_TP2 off; LC4 declines the TP2 heads). Kill: False.
SPARSE_DECODE_LC_PREFETCH_SELF_KV = True
N1_MAX_ROWS = (
    _sparse_decode_kernel.LC_MAX_ROWS
)  # capacity: fixed: the N1 prefetch's row bound (LC2's widest step)
# Up to SELECT_LONG_TRIGGER_MAX_ROWS query rows and without the early release, the long-row select releases the sparse
# decode once every score of its row is loaded (SELECT_LONG_TRIGGER_PHASE 1, before the exact rank of the candidates)
# instead of at its exit (0, the kill value); wider steps keep the exit. The release still follows the select's own
# grid wait, which the sparse decode's pre-wait q / length / block-table reads rely on.
SELECT_LONG_TRIGGER_PHASE = 1  # capacity: fixed: a trigger phase, not a size
SELECT_LONG_TRIGGER_MAX_ROWS = 8  # capacity: fixed: measured trigger crossover
# The long-row select ranks a row's candidates on warp 0 when they number at most 64 (a 64-key bitonic network; the
# other warps exit) and reads the row's valid count before its grid wait; the picks are the same. Kill: False (the
# all-thread rank). In-model, locked memory clock, double-pinned, A and B in one job, both orders: flat within
# noise (mean of draws -5 / -60 us at 1 / 2 requests; replay period at the 323K request +14 / +54); a FLAT-but-exact
# stack item (post-wait select time 1.76 -> 1.54 us per layer at 1 request, 165K tokens). OFF by default: at TP2 the
# warp-0 tail runs ~0.95 us/layer slower in-model next to the early-launched sparse decode (+53 us/step);
# on at TP4 only after an in-model A,B shows a gain.
SELECT_LONG_SL3 = False
# The long-row select runs as kernels/m3_select_long_dsl (ops/attention/m3_select_long_dsl): one 384-thread CTA per row
# instead of one 1024-thread CTA, the same selection bit for bit, with 4, 11 or 24 keys per thread
# (LONG_SELECT_DSL_KEY_WIDTHS) and the same trigger choices (SELECT_LONG_SL3 does not apply). TP4 has the TP8 target's
# per-rank select geometry (1 KV head, 4 rows per request), where on the Rubin decode chain (57 layers behind the
# producer stream, 1 / 2 requests) the select is -0.15 / -0.12 us per layer at 165K tokens, 0.00 / -0.01 at 323K and
# -0.04 / -0.04 at 500K against the 1024-thread kernel. Kill: False (the 1024-thread kernel).
SELECT_LONG_DSL = True
LONG_SELECT_DSL_KEY_WIDTHS = (4, 11, 24)  # capacity: fixed: compiled keys per thread
# The MSA index scorer (ops/attention/m3_index_score) prefetches the older index-K pages into L2 before its grid wait
# at every batch; up to this many requests it also issues Q and the first K tile right after the wait and caps that
# prefetch at one block per CTA (larger batches are bandwidth bound, where the extra early tiles only crowd the
# prefetch). On Rubin TP4 the 2-request step keeps the early schedule: with the producer's trigger at its exit
# (PRODUCER_LATE_TRIGGER_MAX_ROWS) the scorer no longer overlaps the producer, and the capped prefetch shortens its
# exposed pre-wait stream. In-model, locked memory clock, tactic- and FlashInfer-pinned, A and B in one job, both
# orders: -260 us per decode step at 2 requests (replay period at the 323K P90 request; -82 mean of draws), flat at
# 1 / 4 / 8. Timing only: the tiles are read after the wait. Kill: 1.
INDEX_SCORE_EARLY_MAX_REQUESTS = 2  # capacity: fixed: measured early-issue crossover
# T2SC: up to 2 requests the sparse decode scores with this target's copy of the sm_107 TP2 / TP8 scorer
# (kernels/m3_index_score_t2sc; ops/attention/m3_index_score_t2sc), whose split is chosen per request count and which
# TMA-loads an all-old first block before its grid wait; the scores are bit-identical
# to the in-tree scorer. TP4 has the TP8 target's per-rank indexer geometry (1 index head over the replicated index-K
# of its KV group), where this schedule is ~100 us per decode step faster at 1 request than split 512 without the
# prestage (TP8 in-model, greedy early column, both orders). Wider steps keep the sm_103-derived scorer above.
# Kill: False (the sm_103-derived scorer at every request count).
INDEX_SCORE_T2SC = True
# At 1 request the split is one wave of CTAs (1272 = 212 SMs x 6 resident at this CTA's shared memory): at the
# 1-index-head geometry (57-layer chain behind the producer, scores and the selection bitwise) -0.03 / -0.27 / +0.06 us
# per layer at 165K / 323K / 500K tokens against 1536. Kill: 1536.
INDEX_SCORE_T2SC_SPLIT_K: Mapping[int, int] = MappingProxyType(
    {1: 1272, 2: 768}
)  # capacity: fixed: measured splits per request count
# TP2 WIDE: from INDEX_SCORE_INTREE_MIN_REQUESTS requests on (40+ rows at K7) the sparse decode scores with the
# in-tree CuTe DSL scorer (``cute_dsl_minimax_m3_index_decode_score``, its 256-CTA split) instead of the sm_103-derived
# split-512 scorer: the same scores bit for bit. VR microbench (60-call graphs, 2 index heads, qlen 8, PDL): faster at
# every request count 5..24 and context 16K / 64K / 128K / mixed, e.g. mixed -0.97 / -1.51 / -3.28 / -3.86 / -5.98 us
# per layer at 5 / 8 / 12 / 16 / 24 requests. Steps of <= 32 rows (any request count) keep their scorer. Kill:
# TRTLLM_M31_DISABLE_TP2WIDE_INTREE_SCORER=1. Read once, at import.
INDEX_SCORE_INTREE = not m31_disabled("TP2WIDE_INTREE_SCORER")
INDEX_SCORE_INTREE_MIN_REQUESTS = (
    5  # capacity: fixed: measured crossover (4 requests = 32 rows stays as is)
)
TP2_WIDE_MIN_ROWS = (
    33  # capacity: fixed: TP2 WIDE rule: steps of <= 32 rows keep the baseline dispatch bit for bit
)
_INDEX_SCORE_INTREE_LOGGED: set = set()


def _in_tree_scorer():
    """The in-tree CuTe DSL index scorer's runner (importing it registers ``cute_dsl_minimax_m3_index_decode_score``);
    imported on first use, as the CuTe DSL ops are."""
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
        CuteDSLMiniMaxM3IndexDecodeScoreRunner,
    )

    return CuteDSLMiniMaxM3IndexDecodeScoreRunner


# The decode MXFP8 GEMVs, per projection (N, K): the rows each serves. On an activation its producer quantized
# (the MoE boundary for the QKV, the attention merge for the o_proj) the GEMV serves every row up to 48 but the
# unsplit K=768 shared down projection at 17..32 rows, where FlashInfer's GEMM is faster.
# The QKV MXFP8 GEMVs' (split, ring) at decode rows up to GEMV_QKV_CFG_MAX_ROWS: a TP4-owned override of the shared
# GB300-tuned table (kernels/mx_gemv/op.py). SM107 TP4 sweep
# (4 rows): sparse-layer QKV (8, 8) 2.70 us vs (4, 12) 3.01, dense-attention QKV (8, 8) 2.60 vs 2.83. Split 8
# sums K in a different fp32 split-K order than split 4: exact-class, not bitwise to the shared config. Kill: False.
GEMV_QKV_SPLIT8_TP4 = True
GEMV_QKV_CFG_MAX_ROWS = 8  # capacity: fixed: the measured override's rows
MX_GEMV_CFG: Dict[Tuple[int, int], tuple] = {
    QKV_SPARSE: (8, 8),
    QKV: (8, 8),
}  # capacity: fixed: (split, ring) per projection
# (exact-class, deterministic) The o_proj GEMV at decode rows up to GEMV_QKV_CFG_MAX_ROWS with strided k-tiles per split
# rank (contig 0: rank r owns k-tiles r, r + split, ...) instead of contiguous ranges, the shared (2, 8): another
# fixed fp32 order. SM107 TP4 4 rows: 2.513 / 2.533 vs 2.550 / 2.563 us; deterministic (3 runs bitwise) and
# max error vs the fp64 product equal to the contiguous order's. Kill: False.
GEMV_OPROJ_STRIDED_TP4 = True
MX_GEMV_CFG_STRIDED: Dict[Tuple[int, int], tuple] = {
    O_PROJ: (2, 8, 0)
}  # capacity: fixed: (split, ring, contig) per projection


def mx_gemv_cfg(n: int, k: int, rows: int) -> Optional[tuple]:
    """The TP4 GEMV config override for an [rows, k] x [n, k]^T MXFP8 GEMV call, or None for the shared table's."""
    if rows > GEMV_QKV_CFG_MAX_ROWS:
        return None
    if GEMV_QKV_SPLIT8_TP4 and (n, k) in MX_GEMV_CFG:
        return MX_GEMV_CFG[(n, k)]
    if GEMV_OPROJ_STRIDED_TP4:
        return MX_GEMV_CFG_STRIDED.get((n, k))
    return None


# Synth only (TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS, fractional): the fused verify's forced accepted counts are
# drawn per request, keyed by the request's KV length at the verify (its prompt plus what it has accepted), instead of
# from the server-wide counter, so a request's forced counts do not depend on what the server ran before (A/B draws
# become history-independent). Production sampling never forces counts. Kill: False (the shared counter, the in-tree
# sequence).
FORCE_AL_PER_REQUEST = True

_ALL_ROWS = frozenset(range(1, DECODE_MAX_TOKENS + 1))  # capacity: decode rows
# TP2WIDE_GEMM (TP2 WIDE, microbench: SM107, mclk locked, MXFP8 activation, weights rotated past L2, 2 processes):
# at 33..48 rows the dense MLP's MXFP8 projections take the pinned FlashInfer GEMM (MX-TAC, the armed tactic table)
# instead of the GEMV: dense gate_up 15.05 -> 10.10 us/call, dense down 13.57 -> 8.51 (M3 #20623's tp2 wide method,
# which keeps the dense MLP on the GEMM). Every other projection keeps the GEMV there (QKV / shared expert 1.7..2.7 us
# faster; o_proj even). Rows <= 32 unchanged. Kill switch TRTLLM_M31_DISABLE_TP2WIDE_GEMM=1 (the GEMV at 33..48 and the
# VR_GEMV48 / LINEAR_CUBLASLT row tables below), read at import.
TP2WIDE_GEMM = not m31_disabled("TP2WIDE_GEMM")
_DENSE_GEMM_ROWS = (
    frozenset(range(33, DECODE_MAX_TOKENS + 1)) if TP2WIDE_GEMM else frozenset()
)  # capacity: fixed: the measured GEMM band of the dense MLP
MX_GEMV_ROWS: Dict[Tuple[int, int], FrozenSet[int]] = {
    QKV_SPARSE: _ALL_ROWS,
    QKV: _ALL_ROWS,
    O_PROJ: _ALL_ROWS,
    SHARED_GATE_UP: _ALL_ROWS,
    SHARED_DOWN: _ALL_ROWS - frozenset(range(17, 33)),
    DENSE_GATE_UP: _ALL_ROWS - _DENSE_GEMM_ROWS,
    DENSE_DOWN: _ALL_ROWS - _DENSE_GEMM_ROWS,
}  # capacity: decode rows per projection
# On a bf16 activation the GEMV quantizes it in its prologue, which pays only up to 8 32-element blocks per thread:
# the most rows each projection takes.
# TP2: the GEMV's own prologue bound (kernels/mx_gemv/op._declines: tiles * 4 * mma_n <= 1024 at each projection's
# config) caps o_proj (K 4096, split 2: 16 k-tiles) at 16 rows and the dense down (K 6144, split 2: 24 k-tiles) at 8;
# the TP4 table's 32 / 16 would be covered rows the GEMV declines (a miss, raised under modeling_v2='require').
MX_GEMV_BF16_MAX_TOKENS: Dict[Tuple[int, int], int] = {
    QKV_SPARSE: 16,
    QKV: 16,
    O_PROJ: 16,
    SHARED_GATE_UP: 16,
    SHARED_DOWN: 16,
    DENSE_GATE_UP: 8,
    DENSE_DOWN: 8,
}  # capacity: fixed: measured bf16-GEMV crossover per projection
# MX-TAC (the TP2 target's, ported): the decode-graph MXFP8 projections the GEMVs do not serve (SHARED_DOWN at 17..32
# rows, rows over DECODE_MAX_TOKENS or a bf16 activation over MX_GEMV_BF16_MAX_TOKENS, or a GEMV decline) take one fixed
# decision instead of the AutoTuner's per-start, per-rank choice: FlashInfer's CuTe DSL quantize and CuTe DSL GEMM. All
# of stock's candidates (quantize {TRT-LLM, CuTe DSL} x GEMM {CUTLASS, CuTe DSL}) give the same bits at the TP4
# projections (gemm/test_modeling_v2_m3_pinned_gemm.py), so the pin is stock's output on every rank and start. Kill
# constant: False returns these projections to the AutoTuner.
MX_TAC_TP4 = True
# The tactics below are MXFP8QuantizeRunner.CUTE_DSL / FlashInferMXFP8GemmRunner.CUTE_DSL (0).
MXFP8_DECODE_QUANT_TACTIC = 0  # capacity: fixed: a tactic id, not a size
MXFP8_DECODE_GEMM_TACTIC = 0  # capacity: fixed: a tactic id, not a size
# TP2 port: ported from the sm_107 TP8 target as is. Its rows start past DECODE_MAX_TOKENS (48 here), so at K7
# c1..c4 (8..32 rows) it never engages; its shape keys (m3_mx_gemv GEMV48_SHAPES, m3_mxfp8_cublaslt.SHAPES)
# are the per-rank shapes of this target's m3_mx_gemv constants.
# Generation steps of 33..64 rows (K7 c5..c8) run past the decode bound, so their MXFP8 projections took the quantize +
# FlashInfer CuTe DSL GEMM, which steps from ~8 to ~11.8 us/call above 32 rows on the K = 6144 projections (SM107).
# Two existing kernels take those rows instead, on the same quantized activation (not bitwise to that GEMM):
# * VR_GEMV48 (switch TRTLLM_M31_DISABLE_VR_GEMV48): the GEMV (it serves up to 48 rows) on GEMV48_SHAPES at 33..48;
# * LINEAR_CUBLASLT (switch TRTLLM_M31_DISABLE_LINEAR_CUBLASLT, shared with the sm_103 targets): cuBLASLt's MXFP8 GEMM
#   (ops/gemm/m3_mxfp8_cublaslt) on its SHAPES at every other row up to 64.
# A bf16 activation is quantized first by the CuTe DSL quantize (the bytes of mxfp8_quantize). Measured us/step over the
# 240 calls of a K7 step: -583..-592 at 33..48 rows, -494..-497 at 56..64. Off: back to FlashInfer's GEMM.
VR_GEMV48 = not m31_disabled("VR_GEMV48")
LINEAR_CUBLASLT = not m31_disabled("LINEAR_CUBLASLT")
GEMV48_ROWS: Dict[Tuple[int, int], FrozenSet[int]] = (
    {
        nk: frozenset(range(max(GEMV48_MIN_TOKENS, DECODE_MAX_TOKENS + 1), mx_gemv.MAX_TOKENS + 1))
        for nk in GEMV48_SHAPES
    }
    if VR_GEMV48 and not TP2WIDE_GEMM
    else {}
)
# TP2WIDE_GEMM: ``_linear`` has no branch for GEMV48_PATH / CUBLASLT_GEMM_PATH at TP2, so their rows ran the pinned
# FlashInfer GEMM (MX-TAC) under those names; with TP2WIDE_GEMM the tables are empty and ``linear_path`` names that
# GEMM (the same kernels: a microbench found cuBLASLt flat or slower than it at 49..64 on every TP2 projection).
LINEAR_CUBLASLT_ROWS: Dict[Tuple[int, int], FrozenSet[int]] = (
    {
        nk: frozenset(
            m
            for m in range(
                max(mxfp8_cublaslt.MIN_TOKENS, DECODE_MAX_TOKENS + 1), mxfp8_cublaslt.MAX_TOKENS + 1
            )
            if m not in GEMV48_ROWS.get(nk, ())
        )
        for nk in mxfp8_cublaslt.SHAPES
    }
    if LINEAR_CUBLASLT and not TP2WIDE_GEMM
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
SPARSE_DECODE_PRODUCER = RowRange(
    "sparse_decode_producer",
    frozenset({"per_row", "multi_query"}),
    m3_sparse_decode,
    _sparse_decode_producer,
)


def _sparse_decode_lc_keys(spec: Mapping[str, Any]) -> Iterable[Key]:
    """A cell certifies the LC2 key it drives when its step lies in the rows TP2WIDE_LC2 takes."""
    rows = spec["decode_query_len"] * len(spec["seq_lens"])
    lo = _sparse_decode_kernel.lc_min_rows(spec["decode_query_len"])
    if not (lo <= rows <= _sparse_decode_kernel.LC_MAX_ROWS):
        return ()
    if spec["linear_softmax"] or spec["prewait"]:
        return ()
    return ("mxfp8" if spec["mxfp8"] else "plain",)


# TP2WIDE_LC2: the long-context sparse decode at the TP2 heads (32 query / 2 KV heads, LC2): attention, split-K merge
# and the MXFP8 o_proj hand-off in one launch, for pure generation steps with a selected top-16 table at
# LC_MIN_ROWS (33)..LC_MAX_ROWS (448; 128 with TP2WIDE_LC2_448 disabled) rows. TP2_LC2_SMALL opts K3 steps
# into LC2 from 9 rows. The MXFP8 hand-off is published to m3_prequant.SLOT. Narrower steps keep the Triton
# producer + merge. frozenset()
# (TRTLLM_M31_DISABLE_TP2WIDE_LC2=1) routes every step back to the Triton producer + merge.
SPARSE_DECODE_LC = RowRange(
    _sparse_decode_kernel.LC_PATH,
    _sparse_decode_kernel.LC_KEYS,
    m3_sparse_decode_lc,
    _sparse_decode_lc_keys,
)
# Every token count of a sparse layer on an NVFP4 KV cache at the TP2 heads (prefill and decode): the producer takes the
# geometry, not a row range.
PRODUCER = RowRange("m3_producer", frozenset({"nvfp4_tp2"}), m3_producer, _const("nvfp4_tp2"))
# The per-rank (query, KV, index) heads ``m3_producer`` is built for (ops/attention/m3_producer HEADS_Q / HEADS_KV /
# HEADS_INDEX: MiniMax-M3's 64 / 4 / 4 heads over two ranks). C6: the gate held the TP4 heads (16, 1, 1) from the copied
# TP4 target, so no TP2 layer ever took the producer (the generic m3_qkv_insert ran, uncounted and unlogged).
PRODUCER_HEADS = (32, 2, 2)
INDEX_SCORE = RowRange(
    "m3_index_score",
    frozenset({"early", "prefetch"}),
    m3_index_score,
    lambda spec: ("early" if spec["early"] else "prefetch",),
)


def index_score_t2sc_schedule(requests: int) -> Tuple[int, bool]:
    """T2SC's (CTAs per request, early issue) at this many requests (a request count in INDEX_SCORE_T2SC_SPLIT_K)."""
    return INDEX_SCORE_T2SC_SPLIT_K[requests], requests <= INDEX_SCORE_EARLY_MAX_REQUESTS


# The index heads per rank this target's scorer runs (MiniMax-M3.1: 4 index heads over 2 ranks).
INDEX_HEADS_TP2 = 2  # capacity: fixed: the per-rank index heads at TP2


def _index_score_t2sc_requests(spec: Mapping[str, Any]) -> Iterable[Key]:
    """A cell certifies its request count when it drives this target's 2 index heads with that count's schedule."""
    requests = len(spec["seq_lens"])
    if spec.get("heads") != INDEX_HEADS_TP2 or requests not in INDEX_SCORE_T2SC_SPLIT_K:
        return ()
    if (spec["split_k"], spec["early"]) != index_score_t2sc_schedule(requests):
        return ()
    return (requests,)


INDEX_SCORE_T2SC_PATH = RowRange(
    "m3_index_score_t2sc",
    frozenset(INDEX_SCORE_T2SC_SPLIT_K) if INDEX_SCORE_T2SC else frozenset(),
    m3_index_score_t2sc,
    _index_score_t2sc_requests,
)
LONG_SELECT = RowRange("long_select", frozenset(), m3_select_long, _long_select_widths)


def _long_select_dsl_widths(spec: Mapping[str, Any]) -> Iterable[Key]:
    """The 384-thread select's key widths a cell's rows reach (every width up to its blocks' for mixed rows)."""
    blocks = spec["num_blocks"]
    widest = 4 if blocks <= 4 * 384 else 11 if blocks <= 11 * 384 else 24
    if spec["valid"] == "mixed":
        return tuple(w for w in LONG_SELECT_DSL_KEY_WIDTHS if w <= widest)
    return (widest,)


LONG_SELECT_DSL = RowRange(
    "long_select_dsl",
    frozenset(LONG_SELECT_DSL_KEY_WIDTHS) if SELECT_LONG_DSL else frozenset(),
    m3_select_long_dsl,
    _long_select_dsl_widths,
)
MC_ALLREDUCE = RowRange(
    "mc_allreduce_norm",
    _rows(1, DECODE_MAX_TOKENS) if MC_AR_TP4 else frozenset(),
    m3_mc_allreduce_norm,
    _spec_rows("m", prepushed=0, drafter=None),
)
# Rows of a fused-draft-loop all-reduce boundary (drafter hidden 6144) on the multicast all-reduce's drafter form.
DRAFT_MC_AR = RowRange(
    "m3_draft_mc_allreduce_norm",
    _rows(1, DRAFT_MC_AR_MAX_ROWS),
    m3_mc_allreduce_norm,
    _spec_rows("m", drafter=True),
)
DENSE_PUSH = RowRange(
    "dense_push",
    _rows(1, DENSE_PUSH_MAX_TOKENS),
    m3_mc_allreduce_norm,
    _spec_rows("m", prepushed=AR_PUSH_MODE),
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
    _rows(MC_FINALIZE_MIN_TOKENS, MC_FINALIZE_MAX_TOKENS),
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
# The router of a pure generation step of 9..16 rows when the MoE runs unfused (above the MoE engine's 8; at 9..13 an
# MoE16 engine miss, 14..16 always): the TF32 tensor-core router (minimax_m3_router_tc, 16-CTA clusters, one reduction
# order), the arithmetic the built-in gate takes on SM107 at these rows. Devices that cannot co-schedule 16-CTA
# clusters keep the fp32 router. Context steps never take it.
ROUTER_TC_GEN_MIN_ROWS = 9  # capacity: fixed: past the MoE engine's 8 rows
# ROUTER48 (ported from the sm_107 TP8 target; switch TRTLLM_M31_DISABLE_ROUTER48): the TF32 tensor-core
# router up to the op's certified 48 rows (K7 c3 / c4: 24 / 32 rows). The router weights are replicated, so
# the op's cells are TP-independent; a microbench measured -116 / -103 us per step at 24 / 32 rows (TP8).
ROUTER48 = not m31_disabled("ROUTER48")
# Reuse the native router NVFP4 side output at its certified small-row shapes.
X1 = not m31_disabled("X1")
ROUTER_TC_GEN_MAX_ROWS = 48 if ROUTER48 else 16  # capacity: fixed: the op's certified rows
# S1B_FOLD (ops/moe/m31_s1b_fold.py): generation steps above the TC router's rows route from top-k ids (bmm over
# K-slices + one reduce / MiniMax2-select kernel; bitwise the stock router + routing). Pinned per row count at
# prebuild up to the WIDE decode cap. Kill switch TRTLLM_M31_DISABLE_S1B_FOLD=1.
S1B_MAX_ROWS = 448  # capacity: fixed: the WIDE decode rows (A1_WIDE_448)
# Diagnostics (default off): TRTLLM_M31_DISABLE_S1B_FOLD_PDL=1 launches the fold without PDL;
# TRTLLM_M31_ENABLE_S1B_FOLD_CHECK=1 also runs the stock route + MoE on every folded call, keeps the STOCK output and
# counts bitwise mismatches of the fold's output on device (logged by a host thread; ``_S1bCheck``).
S1B_FOLD_PDL = not m31_disabled("S1B_FOLD_PDL")
S1B_FOLD_CHECK = m31_enabled("S1B_FOLD_CHECK")
ROUTER_TC_GEN = RowRange(
    "m3_router_tc",
    _rows(ROUTER_TC_GEN_MIN_ROWS, ROUTER_TC_GEN_MAX_ROWS),
    m3_router_tc,
    _spec_rows("m"),  # every cell checks the logits, with or without the NVFP4 activation
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
MX_GEMV48 = RowRange(
    "m3_mx_gemv48",
    frozenset((nk, m) for nk, rows in GEMV48_ROWS.items() for m in rows),
    m3_mx_gemv,
    lambda spec: ((spec["nk"], spec["m"]),),
)
LINEAR_CUBLASLT_GEMM = RowRange(
    "m3_mxfp8_cublaslt",
    frozenset((nk, m) for nk, rows in LINEAR_CUBLASLT_ROWS.items() for m in rows),
    m3_mxfp8_cublaslt,
    lambda spec: (((spec["n"], spec["k"]), spec["m"]),),
)
# (draft length, requests) of a greedy verify step.
SHARD_ACCEPT = RowRange(
    "m3_shard_accept",
    frozenset((SHARD_ACCEPT_DRAFT_LEN, n) for n in range(1, SHARD_ACCEPT_MAX_REQUESTS + 1)),
    m3_shard_accept,
    lambda spec: ((spec["draft_len"], spec["nreq"]),),
)

# Requests of a decode step whose dense-layer attention runs at a capped KV split.
DENSE_DECODE_SPLIT = RowRange(
    "m3_dense_decode",
    frozenset(DENSE_DECODE_MAX_SEQ_LEN_BY_BATCH),
    m3_dense_decode,
    lambda spec: (len(spec["kv_lens"]),) if spec["capped"] else (),
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
# ((N, K), rows) of a drafter BF16 linear on its low-M kernel.
DRAFT_LOWM = RowRange(
    "m3_draft_lowm",
    frozenset((nk, m) for nk, rows in DRAFT_LOWM_ROWS.items() for m in rows),
    m3_draft_lowm,
    lambda spec: ((spec["nk"], spec["m"]),),
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

# The step-loop hooks (RuntimeHooks), one switch each. They move a decode step's input preparation
# off the GPU timeline between step graphs.
STEP_PACK_INPUTS = True
# The pack's host copies go out in one parallel scatter (a program per copy) when their destinations do not overlap: the
# serial one-CTA scatter pays one dependent global round trip per copy (SM107: 12.5 / 15.6 / 31.7 us at 16 / 24 / 40
# copies vs 6.5 / 6.5 / 6.9 us parallel).
STEP_PACK_PARALLEL_SCATTER = True
# The pack is issued by the decode graph's own prologue (a copy of a pinned window + the parallel scatter), so no
# operation is left between the step graphs (needs STEP_PACK_PARALLEL_SCATTER).
STEP_PACK_IN_GRAPH = True
# The window reaches the prologue's device buffer on a side copy stream while the previous graph runs; the prologue
# waits on that copy and scatters device to device, so the replay no longer starts with a host copy. Needs
# STEP_PACK_IN_GRAPH.
STEP_PACK_EARLY_COPY = True
# The step epoch (step_inputs.step_epoch): one int64 device counter the decode graph's prologue increments once per
# replay (program 0 of its first scatter launch, no extra kernel) and never resets. The per-layer megakernel pieces
# tag their per-step device flags with it (contract v1: tag = epoch << 8 | layer). Needs STEP_PACK_IN_GRAPH.
STEP_PACK_EPOCH = True
# The prologue is one launch of this target's op (ops/spec/m3_step_prologue, minimaxM3Sm107Tp4/stepPrologue.cu): the
# host copies, a CTA barrier, the device copies, the header clear and the epoch, where the default prologue runs two
# Triton scatter launches and an ATen fill (3 graph kernels -> 1, no Triton at the head of the decode graph). Same bytes
# written. Needs STEP_PACK_EARLY_COPY. Kill: False.
STEP_PACK_CPP_PROLOGUE = True
# PRE-WAIT WRITER CONTRACT: the kernels that read step inputs (seq_lens,
# block tables, positions, cache slots, the sampled-accept pre-wait parameters) before their grid-dependency wait rely
# on every writer of them being a host-to-device copy before the replay or a PLAIN launch earlier in the step:
# msa_kv_update, the step-input prologue (Triton scatter or the C++ op), the spec fold tail (Triton or C++), the pack
# scatter and any step-head op (the spec step correction, STEP_SPEC_OFFSETS_ONE_LAUNCH). Never give one of them a PDL
# attribute with an early trigger.
STEP_SKIP_UNCHANGED_H2D = True
# A changed packed 2-D metadata upload stages only each row's changed column span. On a step where a request crosses
# into a new page the MSA block table changed by one entry, but the whole padded table (32 KB a request) was staged,
# overflowing the 16 KB in-graph window into an eager upload and scatter between the step graphs (57-69 us at the
# ~1 in 19 block-crossing steps; TTFT inter-graph scan). Kill: False.
STEP_H2D_CHANGED_SPAN = True
# A speculative overlap step re-derives the MSA KV-length metadata once, after the forward's KV-length correction,
# instead of also before it (RuntimeHooks.kv_lens_update_once): the first pass (a fill, the out_cache_loc tail fill and
# an msa_kv_update at the head of the decode graph, 2.13 us at TP4) is rewritten in full by the second. Kill: False.
STEP_KV_LENS_UPDATE_ONCE = True
# The overlap scheduler's speculative step correction (position_ids += offsets, kv_lens += offsets) runs as this
# target's one plain launch (RuntimeHooks.spec_offsets_op, ops/spec/m3_spec_offsets) instead of two ATen adds at the
# head of the decode graph: the same int32 adds. TP8 graph bench (the work is per token, not per rank; 64 steps
# per graph, a predecessor kernel each): -1.27 us/step at 4 tokens / 1 request, -0.72..-0.78 at 8-32 tokens.
# Kill: False.
STEP_SPEC_OFFSETS_ONE_LAUNCH = True
STEP_LOOP_ON_EXECUTION_STREAM = True
STEP_MERGE_TUNED_TACTICS = True
STEP_SKIP_GREEDY_RNG_UPLOAD = True
# Sampled speculative batches stage their four per-step Philox seed/offset uploads into the step-input pack
# (STEP_PACK_INPUTS) instead of issuing four copies between the step graphs.
STEP_PACK_SPEC_RNG_UPLOADS = True
# The spec step's small uploads: the accepted-draft counts ride the step-input pack, and the sampler's slot ids stay in
# a device buffer re-uploaded only when they change (a pageable upload every step otherwise).
STEP_SPEC_SMALL_H2D = True
# SKIP_ACCEPTED_UPLOAD (KIMI-HO-09, RuntimeHooks.skip_accepted_draft_upload): the step's per-request accepted-draft
# counts are no longer uploaded and spec_metadata.num_accepted_draft_tokens is None. Nothing on this target reads that
# tensor (modeling.py, dspark.py, dspark_s4.py, mxdraft.py, kernels/, ops/; the in-tree DFlash / DSpark workers and
# the Eagle3 path: tests/unittest/_torch/executor/test_skip_accepted_draft_upload.py scans for readers). Kill switch:
# TRTLLM_M31_DISABLE_SKIP_ACCEPTED_UPLOAD=1 (the stock per-step upload, packed under spec_small_h2d).
STEP_SKIP_ACCEPTED_UPLOAD = not m31_disabled("SKIP_ACCEPTED_UPLOAD")
# BATCH_CHANGED_ONLY (KIMI-HO-03 / ST-18(d), RuntimeHooks.spec_batch_map_changed_only): the standalone DSpark (DFlash)
# drafter's batch_indices and batch-to-slot map are uploaded only when they change (both constant in steady decode);
# the buffers hold the same values as with the stock every-step copies. Kill switch:
# TRTLLM_M31_DISABLE_BATCH_CHANGED_ONLY=1 (both copied every step).
STEP_BATCH_CHANGED_ONLY = not m31_disabled("BATCH_CHANGED_ONLY")
STEP_SPEC_FOLD = True
# The forward of a decode-only step stores its outputs into the spec sampler's store and stages the next step's inputs
# at its tail (one launch inside the step graph), so neither the sampler's store nor the next step's gathers run
# between the step graphs (M3SpecStepFold.in_forward).
STEP_SPEC_FOLD_IN_FORWARD = True
# The forward's tail (STEP_SPEC_FOLD_IN_FORWARD) runs as this target's C++ op (kernels/m3_fold_tail cpp=,
# minimaxM3Sm107Tp4/foldTail.cu) instead of the Triton kernel: the same integer copies, one launch either way (no
# Triton on the generation decode path). Kill: False.
STEP_FOLD_TAIL_CPP = True
# With STEP_SPEC_FOLD_IN_FORWARD the spec sampler's host copies of its store (new tokens, lengths, draft tokens) still
# ran on the main stream between the step graphs (the forward, not the sampler, wrote the store, and it rewrites it at
# the next replay's tail). They run on the D2H side stream instead, and the forward's tail waits on an event the
# sampler records after them before it rewrites the store (M3SpecStepFold.store_copies_done).
STEP_SPEC_STORE_COPY_SIDE = True
# The spec sampler copies its per-slot store to the host on its D2H side stream, so the copies overlap the next
# step's input staging instead of sitting between the step graphs.
STEP_SAMPLER_D2H_SIDE_STREAM = True
# The iteration log times the device step with one event per iteration boundary on a side stream, so no timing
# record sits between two step graphs on the main stream (the reported step times are the same).
STEP_ONE_TIMING_EVENT = True
MSA_SHORT_SEQUENCE_IDENTITY = True
# The sparse decode (the LC, the Triton merge) also writes the MXFP8 o_proj hand-off (bit
# for bit torch.ops.trtllm.mxfp8_quantize of its BF16 rows) and publishes it to m3_prequant.SLOT. Kill: False publishes
# nothing, so the o_proj quantizes its BF16 input itself as the built-in path does.
MERGE_MXFP8_HANDOFF = True
MSA_KV_UPDATE = True
MSA_SKIP_UNCHANGED_BLOCK_OFFSETS = True
MSA_REUSE_POOL_VIEWS = True
# Step-input preparation fills the MSA block-id table by slice-copying each request's page-index view (MSA
# RuntimeHooks.block_ids_from_views) instead of building a Python list of the whole padded row; the table is the same.
MSA_BLOCK_IDS_FROM_VIEWS = True
# The prefill indexer proxy can use tile 128 and a row/KV-size split ladder. The planner's tile is retained at
# <=128 rows; larger spans require maxscore_qo_tile support. Both routes are opt-in until the in-model oracle gate:
# TRTLLM_M31_ENABLE_AGG_PROXY_Q128 / _AGG_PROXY_SPLITS. Matching DISABLE switches override the opt-ins.
# TRTLLM_M31_ENABLE_AGG_PROXY_ORACLE compares valid score bits and the full selected table against the plain plan.
MSA_PROXY_Q128 = m31_enabled("AGG_PROXY_Q128") and not m31_disabled("AGG_PROXY_Q128")
MSA_PROXY_SPLITS = m31_enabled("AGG_PROXY_SPLITS") and not m31_disabled("AGG_PROXY_SPLITS")
MSA_PROXY_ORACLE = m31_enabled("AGG_PROXY_ORACLE")

FAST_PATHS: Tuple[RowRange, ...] = (
    EMBED_FIRSTNORM,
    DENSE_DECODE_SPLIT,
    FULLCTX_DECODE,
    DRAFTER_FULLCTX,
    SPARSE_DECODE_PRODUCER,
    SPARSE_DECODE_LC,
    PRODUCER,
    INDEX_SCORE,
    INDEX_SCORE_T2SC_PATH,
    LONG_SELECT,
    LONG_SELECT_DSL,
    MC_ALLREDUCE,
    DRAFT_MC_AR,
    ATTN_PUSH,
    DENSE_PUSH,
    MOE_PUSH,
    MC_FINALIZE,
    MOE_ENGINE,
    MOE16_ENGINE,
    ROUTER_TC_GEN,
    MX_GEMV,
    MX_GEMV_BF16,
    MX_GEMV48,
    LINEAR_CUBLASLT_GEMM,
    SHARD_ACCEPT,
    TARGET_HEAD,
    DRAFT_STEP,
    DRAFT_LOWM,
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
    return num_tokens <= DECODE_STEP_MAX


def moe_path(num_tokens: int) -> Optional[str]:
    """The MoE kernel of a decode step: an engine, the unfinalized MoE + finalize all-reduce, or None (generic)."""
    if MOE_ENGINE.covers(num_tokens):
        return MOE_ENGINE.path
    if MOE16_ENGINE.covers(num_tokens):
        return MOE16_ENGINE.path
    if MC_FINALIZE.covers(num_tokens):
        return MC_FINALIZE.path
    return None


MX_GEMV_PATH = MX_GEMV.path
MX_GEMV_BF16_PATH = MX_GEMV_BF16.path
# The dense layers' swigluoai activation (where the fold above would apply) as this target's C++ op
# (ops/gemm/m3_dense_swiglu, minimaxM3Sm107Tp8/swigluOai.cu: bit for bit the Triton silu_and_mul), PDL-launched with
# an early release so the down GEMV stages its weights during it; the down GEMV reads the activation only after its own
# grid wait. TP2 chain bench: -0.69 us per dense layer vs the Triton kernel. Kill constant: False
# runs the Triton kernel.
DENSE_SWIGLU_CPP = True
DENSE_SWIGLU_CPP_PATH = "m3_dense_swiglu"
# FlashInfer's CuTe DSL MXFP8 GEMM on the producer's MXFP8 activation (rows the GEMV does not serve).
PREQUANT_GEMM_PATH = "flashinfer_mxfp8_gemm"
# The MXFP8 quantize (tuned backend) and FlashInfer's CuTe DSL GEMM: every row the decode kernels do not serve.
GENERIC_GEMM_PATH = "mxfp8_gemm"
# Past the decode bound: the GEMV (VR_GEMV48) or cuBLASLt's MXFP8 GEMM (LINEAR_CUBLASLT), on either activation.
GEMV48_PATH = MX_GEMV48.path
CUBLASLT_GEMM_PATH = LINEAR_CUBLASLT_GEMM.path


def linear_path(n: int, k: int, num_tokens: int, prequant: bool, low_m_gemm: bool) -> str:
    """The decode-graph MXFP8 GEMM of an [N, K] projection on ``num_tokens`` rows.

    ``prequant``: the producer published the activation in MXFP8. ``low_m_gemm`` is accepted for the call sites'
    symmetry: no low-M GEMM configuration is certified at TP4, so it selects nothing.
    """
    if num_tokens in GEMV48_ROWS.get((n, k), ()):
        return GEMV48_PATH
    if num_tokens in LINEAR_CUBLASLT_ROWS.get((n, k), ()):
        return CUBLASLT_GEMM_PATH
    if prequant:
        if num_tokens in MX_GEMV_ROWS.get((n, k), ()):
            return MX_GEMV_PATH
        return PREQUANT_GEMM_PATH
    if num_tokens <= MX_GEMV_BF16_MAX_TOKENS.get((n, k), 0):
        return MX_GEMV_BF16_PATH
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


def long_select_dsl_keys() -> Tuple[Tuple[bool, bool, int], ...]:
    """The (PDL launch, early trigger, trigger phase) keys of the 384-thread select a decode step can launch, compiled
    before warmup; none when ``SELECT_LONG_DSL`` is off."""
    if not LONG_SELECT_DSL.keys:
        return ()
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    pdl = get_env_enable_pdl()
    keys = [(pdl, False, SELECT_LONG_TRIGGER_PHASE), (pdl, False, 0)]
    if SELECT_EARLY_MAX_ROWS:
        keys.append((pdl, True, 0))
    return tuple(dict.fromkeys(keys))


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
    # Gate e: every taken path is also a work-done counter under its path name (kernels/m3_report.work).
    m3_report.work(
        path, (f"{rows} rows" if rows is not None else "") + (f": {what}" if what else "")
    )


def _mc_taken(mapping, rows: int, what: str) -> None:
    """The multicast all-reduce + norm's engagement line with its world size, kernel and this rank's call count (the
    count is the ``mc_allreduce_norm`` work counter: captures + eager calls). At world 4 above
    ``MC_AR_CLUSTER_MAX_ROWS`` the TP4 op runs its row kernel; at any other world size (TP2: world 2) it runs the 8-CTA
    cluster kernel. Logged once per (rows, detail) per rank, like ``taken``."""
    world = int(mapping.tp_size)
    kernel = (
        "row kernel" if world == 4 and rows > MC_AR_CLUSTER_MAX_ROWS else "8-CTA cluster kernel"
    )
    calls = m3_report.work(MC_ALLREDUCE.path, f"{rows} rows: {what}, world {world}, {kernel}")
    _log_once(
        logger.info,
        MC_ALLREDUCE.path,
        f"taken: {rows}: {what}",
        f"[M3 dispatch] rank {mpi_rank()}: {MC_ALLREDUCE.path} taken at {rows} rows ({what}, world {world}, "
        f"{kernel}, call {calls})",
    )


def uncovered(path: str, reason: str) -> None:
    """The step is outside ``path``'s covered shapes: log once per (path, reason); the generic path runs."""
    _log_once(logger.info, path, reason, f"[M3 dispatch] {path} not taken: {reason} (logged once)")


def reset_logged() -> None:
    """Forget which (path, reason) pairs were logged."""
    with _lock:
        _logged.clear()


# Kernel modules report from inside their calls through kernels/m3_report: the same policy.
m3_report.install(
    on_taken=taken, on_missed=missed, on_uncovered=uncovered, on_capture_missed=capture_missed
)


# --------------------------------------------------------------------------------------------------------------------
# Target constants
# --------------------------------------------------------------------------------------------------------------------

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
    """The runtime this target is written for (it is not in the checkpoint): routing declines every other one
    (``routing._runtime_mismatch``), so this is the backstop for a construction that did not go through routing.

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
    if getattr(sparse, "indexer_kv_dtype", None) not in ("fp8", "nvfp4"):
        raise ValueError(
            "this MiniMax-M3 target requires sparse_attention_config.indexer_kv_dtype='fp8' (or the agg opt-in "
            "'nvfp4', AGG_NVFP4_INDEXK)"
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


def _gemv_variants(n: int, k: int, push: Optional[str], world: int):
    """Every (rows, bf16 activation, push (world, bucket tokens, mode)) call the decode dispatch routes to an [N, K]
    projection's GEMV: an MXFP8 activation on the rows ``MX_GEMV_ROWS`` gives it, a bf16 one up to
    ``MX_GEMV_BF16_MAX_TOKENS``; each pushing when the projection's boundary push covers the rows."""
    push_rows = {"attn": ATTN_PUSH.keys, "dense": DENSE_PUSH.keys, "moe": MOE_PUSH.keys}.get(
        push, frozenset()
    )
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
    # VR_GEMV48: the generation steps past the decode bound, on the (quantized) MXFP8 activation, never pushing.
    variants.extend((rows, False, None) for rows in sorted(GEMV48_ROWS.get((n, k), ())))
    return variants


def _precompile_cfg(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    m: int,
    cfg: tuple,
    use_pdl: bool,
    ar: Optional[Tuple[int, int, int]] = None,
    prod: bool = False,
) -> Optional[tuple]:
    """``mx_gemv.precompile`` for a call with this target's (split, ring) override ``cfg`` (``mx_gemv_cfg``): the
    shared precompile resolves the shared table's config only. ``prod``: the call with the producer epilogue
    (``QKV_PRODUCER_FOLD``). Compiles without launching, outside capture, also while compiles are frozen; returns the
    kernel's key, or None when such a call is declined."""
    import cuda.bindings.driver as cuda_driver

    n, k = weight.shape
    push = None if ar is None else (0, 0, 0, *ar)
    resolved = mx_gemv._resolve(m, n, k, False, True, cfg, use_pdl, push, False, prod)
    if resolved is None:
        return None
    key, runtime_ar = resolved
    if mx_gemv.is_compiled(key):
        return key
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("_precompile_cfg allocates and compiles: call it outside capture")
    dev = weight.device
    x = torch.zeros(m, k, dtype=torch.float8_e4m3fn, device=dev)
    xsf = torch.full((128 * (k // 32),), 127, dtype=torch.uint8, device=dev)
    y = torch.empty(m, n, dtype=torch.bfloat16, device=dev)
    args = mx_gemv._views(x, xsf, weight, weight_scale, y, False)
    stream = cuda_driver.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    mx_gemv._compile(key, args, m, runtime_ar, stream, mx_gemv._extra_args(dev), mx_gemv.PROD_OFF)
    return key


_MX_TAC_ENGAGED: set = set()


def _note_mx_tac(kind: str, n: int, k: int, rows: int) -> None:
    """MX-TAC's engagement line, once per (activation kind, projection, rows)."""
    key = (kind, n, k, rows)
    if key not in _MX_TAC_ENGAGED:
        _MX_TAC_ENGAGED.add(key)
        logger.info(
            f"[M3 sm_107 tp4] MX-TAC engaged: pinned MXFP8 decision (quantize tactic "
            f"{MXFP8_DECODE_QUANT_TACTIC}, GEMM tactic {MXFP8_DECODE_GEMM_TACTIC}) "
            f"for the [{n}, {k}] projection at {rows} rows, {kind} activation"
        )


def _uncompiled_note() -> str:
    key = mx_gemv.last_uncompiled
    return "" if key is None else f" (its kernel {key} was not compiled before warmup)"


def _fold_cfg(n: int, k: int, rows: int) -> Optional[tuple]:
    """The QKV GEMV config of a call ``QKV_PRODUCER_FOLD`` takes (the sparse QKV + index projection at <= 8 rows on
    its split-8 override), or None when the producer epilogue does not serve it."""
    cfg = mx_gemv_cfg(n, k, rows)
    if not QKV_PRODUCER_FOLD or n != mx_gemv.PROD_N or rows > 8 or cfg is None or cfg[0] != 8:
        return None
    return cfg


class _DeferredGemv:
    """``QKV_PRODUCER_FOLD``: a sparse layer's QKV + index MXFP8 GEMV, recorded instead of launched.

    ``_linear(..., defer=True)`` returns one on the decode GEMV path of the 39-head packed projection at the rows
    ``_fold_cfg`` serves, with an MXFP8 hand-off. The producer call site either runs the GEMV with the producer in its
    epilogue (``_qkv_producer_fold``) or calls ``materialize`` (the plain GEMV) before anything reads the packed row;
    ``placeholder`` (unwritten, the packed shape) stands for that row until then and is never read."""

    def __init__(self, act, act_scale, weight, weight_scale, cfg, use_pdl):
        self.placeholder = torch.empty(
            act.shape[0], weight.shape[0], dtype=torch.bfloat16, device=act.device
        )
        self.act, self.act_scale = act, act_scale
        self.weight, self.weight_scale = weight, weight_scale
        self.cfg = cfg
        self.use_pdl = use_pdl
        # Set once the record ran, fused (the producer GEMV) or plain (``materialize``).
        self.done = False

    def materialize(self) -> torch.Tensor:
        """Launch the plain GEMV this record stands for; returns the packed row ``_linear`` would have returned."""
        self.done = True
        rows, (n, k) = self.act.shape[0], self.weight.shape
        output = m3_mx_gemv(
            self.act,
            self.act_scale,
            self.weight,
            self.weight_scale,
            True,
            self.cfg,
            self.use_pdl,
            None,
        )
        if output is None:
            raise RuntimeError(
                f"the deferred [{n}, {k}] GEMV declined at {rows} rows" + _uncompiled_note()
            )
        taken(MX_GEMV_PATH, rows, f"[{n}, {k}] mxfp8 in, split {self.cfg[0]}")
        return output


def _linear(
    core,
    x: torch.Tensor,
    wk: str,
    sk: str,
    low_m_gemm: bool,
    push: Optional[tuple] = None,
    defer: bool = False,
) -> Tuple[Any, bool]:
    """One MXFP8 projection ``x @ w[wk]^T``; returns (output, whether its GEMV pushed into ``push``).

    Outside decode CUDA graphs: the engine-tuned MXFP8 quantize + GEMM. Inside them the projection's (N, K) and the
    step's rows pick one path (``linear_path``):

    1. an activation its producer already quantized to MXFP8 (``m3_prequant.SLOT``, taken here): the tcgen05 decode
       GEMV on the rows it serves, else the decode-graph MXFP8 GEMM on that activation (AutoTuner-selected backend
       per token bucket);
    2. a bf16 activation on the rows the GEMV serves quantizing in its prologue;
    3. otherwise the engine-tuned MXFP8 quantize + GEMM.

    A kernel declining rows its path covers is a miss; the chain then continues down the list. ``push`` (an
    ``m3_mx_gemv`` ``ar`` tuple) is the boundary all-reduce exchange the GEMV writes its rows into. ``low_m_gemm``
    is accepted for the call sites' symmetry (``linear_path``): no TP4 low-M GEMM configuration is built. ``defer``
    (``QKV_PRODUCER_FOLD``, the sparse layers' packed QKV + index projection): on the rows ``_fold_cfg`` serves the
    GEMV is not launched; a ``_DeferredGemv`` is returned for the producer call site.
    """
    w, s = core.w[wk], core.w[sk]
    x = x.contiguous()
    mode = _mx_mode(core.mxfp8)
    # F3: an F3Push target is used only by the MX-TAC pinned GEMM branches below; no other path sees it.
    f3 = push if isinstance(push, F3Push) else None
    if f3 is not None:
        push = None
    # F3: the attention boundary's MXFP8 norm, taken where the stock path is the pinned GENERIC quantize + GEMM.
    f3mx = _F3_MX.take(x) if TP2_F3 and _F3_MX.pending is not None else None
    # TP2W_S2_TAIL: the fused MoE tail's MXFP8 activation of x, used where the stock path quantizes x itself before
    # the same GEMM (bit-identical operands). Decode graphs (tuned_quantize_gemm) take it too: their pinned MX-TAC
    # branch below quantizes with m3_mxfp8_pinned_quantize, the bytes the fused tail writes.
    s2 = (
        _S2_MX.take(x)
        if TP2W_S2_TAIL and (mode.decode_graph or not mode.tuned_quantize_gemm)
        else None
    )
    if not mode.decode_graph:
        prequant = _CTXNORMQ_MX.take(x) if CTXNORMQ and not mode.tuned_quantize_gemm else None
        if prequant is not None:
            _sp_ovl_wait(x)
            if x.data_ptr() == _COMBINE_MXFP8_STATE.get("src"):
                _COMBINE_MXFP8_STATE["src"] = None
                m3_report.work(
                    "TP2WIDE_COMBINE_MXFP8",
                    f"o_proj took the combine's MXFP8 input, {x.shape[0]} rows",
                )
                if "taken" not in _COMBINE_MXFP8_STATE:
                    _COMBINE_MXFP8_STATE["taken"] = True
                    logger.info(
                        "[M3.1] TP2WIDE_COMBINE_MXFP8 taken: o_proj consumed the combine's MXFP8 input "
                        f"({x.shape[0]} rows)"
                    )
        # Rule-1b agg prefill port: opt-in until its agg quality gates pass. The same
        # band dispatch consumes either the existing prequant pair or ordinary BF16.
        if agg_prefill_cublaslt.ENABLED and not is_in_breakable_cuda_graph():
            pinned = agg_prefill_cublaslt.try_gemm(x, w, s, core.mx_alpha, prequant)
            if pinned is not None:
                m3_report.work(
                    "AGG_PREFILL_CUBLASLT", f"N={w.shape[0]} K={w.shape[1]} M={x.shape[0]}"
                )
                return pinned, False
        if prequant is not None:
            return _ctxnormq_mxfp8_gemm(core, prequant, w, s, mode), False
        if s2 is not None:
            return _ctxnormq_mxfp8_gemm(core, s2, w, s, mode), False
        return _mxfp8_linear(core, x, w, s, mode), False
    use_pdl = get_env_enable_pdl()
    rows = x.shape[0]
    n, k = w.shape
    prequant = m3_prequant.SLOT.take(x)
    if (
        f3mx is not None
        and s2 is None
        and prequant is None
        and MX_TAC_TP4
        and core.mxfp8.quant_method.tune_decode_graph_backends
        and linear_path(n, k, rows, False, low_m_gemm) == GENERIC_GEMM_PATH
    ):
        # The stock bf16 branch below is m3_mxfp8_pinned_quantize + this GEMM: the boundary wrote those bytes.
        _record("linear_f3", (n, k), rows)
        output = m3_mxfp8_pinned_gemm(
            f3mx[0], f3mx[1], w, s, torch.bfloat16, MXFP8_DECODE_GEMM_TACTIC
        )
        _note_mx_tac("bf16", n, k, rows)
        cnt = _F3_STATE["consumed"] = _F3_STATE.get("consumed", 0) + 1
        if cnt & (cnt - 1) == 0:
            logger.info(
                f"[M3.1] TP2_F3 slot taken n={cnt}: [{n}, {k}] GEMM on the boundary's MXFP8 ({rows} rows)"
            )
        return output, False
    if (
        s2 is not None
        and prequant is None
        and MX_TAC_TP4
        and core.mxfp8.quant_method.tune_decode_graph_backends
        and linear_path(n, k, rows, False, low_m_gemm) == GENERIC_GEMM_PATH
    ):
        # The stock bf16 branch below is the pinned MX-TAC quantize + GEMM: the same GEMM on the fused tail's bytes.
        # Any other decode-graph path keeps its own quantize (the engine-tuned GEMM may not be the pinned one).
        _record("linear_s2", (n, k), rows)
        output = m3_mxfp8_pinned_gemm(s2[0], s2[1], w, s, torch.bfloat16, MXFP8_DECODE_GEMM_TACTIC)
        _note_mx_tac("bf16", n, k, rows)
        cnt = _S2_STATE["consumed"] = _S2_STATE.get("consumed", 0) + 1
        if cnt & (cnt - 1) == 0:
            logger.info(
                f"[M3.1 agg] TP2W_S2_TAIL consumed n={cnt}: [{n}, {k}] GEMM on the fused tail's MXFP8 ({rows} rows)"
            )
        return output, False
    if prequant is None and x.dtype != torch.bfloat16:
        path = GENERIC_GEMM_PATH
    else:
        path = linear_path(n, k, rows, prequant is not None, low_m_gemm)
    _record("linear", (n, k), rows, path, push is not None)
    output = None
    # MX-TAC: the decode-graph MXFP8 quantize and GEMM at the target's fixed tactics, no AutoTuner.
    pinned = MX_TAC_TP4 and core.mxfp8.quant_method.tune_decode_graph_backends
    if prequant is not None:
        act, act_scale = prequant
        fold_cfg = _fold_cfg(n, k, rows) if defer and path == MX_GEMV_PATH else None
        if fold_cfg is not None:
            return _DeferredGemv(act, act_scale, w, s, fold_cfg, use_pdl), False
        if path == MX_GEMV_PATH:
            cfg = mx_gemv_cfg(n, k, rows)
            output = m3_mx_gemv(act, act_scale, w, s, True, cfg, use_pdl, push)
            if output is not None:
                split = f", split {cfg[0]}" if cfg is not None else ""
                taken(path, rows, f"[{n}, {k}] mxfp8 in{split}")
                return output, push is not None and mx_gemv.last_pushed
            missed(
                path,
                rows,
                f"the GEMV declined the [{n}, {k}] projection's MXFP8 activation"
                + _uncompiled_note(),
            )
        if path == PREQUANT_GEMM_PATH:
            uncovered(MX_GEMV_PATH, f"the [{n}, {k}] projection at rows the GEMV does not serve")
        if pinned:
            if f3 is not None and f3.accepts(n, k, rows):
                f3.run(act, act_scale, w, s)
                _note_mx_tac("mxfp8", n, k, rows)
                return f3.out, True
            output = m3_mxfp8_pinned_gemm(
                act, act_scale, w, s, torch.bfloat16, MXFP8_DECODE_GEMM_TACTIC
            )
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
    if pinned and x.dtype == torch.bfloat16:
        act, act_scale = m3_mxfp8_pinned_quantize(x, MXFP8_DECODE_QUANT_TACTIC)
        if f3 is not None and f3.accepts(n, k, rows):
            f3.run(act, act_scale, w, s)
            _note_mx_tac("bf16", n, k, rows)
            return f3.out, True
        output = m3_mxfp8_pinned_gemm(
            act, act_scale, w, s, torch.bfloat16, MXFP8_DECODE_GEMM_TACTIC
        )
        _note_mx_tac("bf16", n, k, rows)
        return output, False
    # Stock's decode-graph dispatch for this shape (the tuned quantize + GEMM).
    return _mxfp8_linear(core, x, w, s, mode), False


def _native_swiglu(core, gu: torch.Tensor, dense: bool) -> Optional[torch.Tensor]:
    """NATIVE_SWIGLU: the SwiGLU-OAI activation of ``gu`` [rows, 2d] by the native op, or None (the caller's path).
    Served for the model's constants (limit 7, alpha 1.702, beta 1), contiguous bf16, outside torch.compile; the dense
    MLP's 88..2048-row band only with NATIVE_ALL_BANDS. The vector count nv is chosen by rows."""
    site = "dense" if dense else "shared"
    if not NATIVE_SWIGLU:
        return _native_dispatch(
            "NATIVE_SWIGLU", site, "killed (TRTLLM_M31_DISABLE_NATIVE_SWIGLU=1)"
        )
    if is_torch_compiling():
        return _native_dispatch("NATIVE_SWIGLU", site, "under torch.compile")
    if (core.swiglu_limit, core.swiglu_alpha, core.swiglu_beta) != (7.0, 1.702, 1.0):
        return _native_dispatch("NATIVE_SWIGLU", site, "SwiGLU constants other than (7, 1.702, 1)")
    if gu.dim() != 2 or gu.dtype != torch.bfloat16 or not gu.is_contiguous():
        return _native_dispatch("NATIVE_SWIGLU", site, "input is not a contiguous 2-D bf16 tensor")
    rows = int(gu.shape[0])
    if dense and not NATIVE_ALL_BANDS and 49 < rows < 4096:
        return _native_dispatch(
            "NATIVE_SWIGLU", site, "the 50..4095-row band (NATIVE_ALL_BANDS off)"
        )
    op = _native_op("minimax_m31_swiglu_oai_vec")
    if op is None:
        return _native_dispatch("NATIVE_SWIGLU", site, "the op is not registered")
    nv = 1 if rows <= 2048 else (2 if rows <= 8192 else 4)
    _native_dispatch("NATIVE_SWIGLU", site)
    _record("m31_native_swiglu", rows, dense)
    taken("m31_native_swiglu", rows, f"{'dense' if dense else 'shared'} nv {nv}")
    return op(gu, get_env_enable_pdl(), nv)


def _dense_swiglu(core, gu: torch.Tensor) -> torch.Tensor:
    """A dense MLP's SwiGLU-OAI activation: this target's C++ op (``DENSE_SWIGLU_CPP``, bit for bit the Triton kernel,
    PDL-launched with an early release) up to its certified rows, in any graph or eager step; else the Triton
    kernel."""
    native = _native_swiglu(core, gu, True)
    if native is not None:
        return native
    rows = int(gu.shape[0])
    path = DENSE_SWIGLU_CPP_PATH
    max_rows = max(cell.spec["rows"] for cell in m3_dense_swiglu.CELLS)
    if DENSE_SWIGLU_CPP and not is_torch_compiling():
        if rows > max_rows:
            uncovered(path, f"the dense swigluoai op: {rows} rows, past its certified {max_rows}")
        else:
            act = m3_dense_swiglu(
                gu, core.swiglu_limit, core.swiglu_alpha, core.swiglu_beta, get_env_enable_pdl()
            )
            if act is not None:
                taken(path, rows)
                return act
            uncovered(path, "the gate_up layout")
    return m3_silu_and_mul(gu, core.swiglu_limit, core.swiglu_alpha, core.swiglu_beta)


# --------------------------------------------------------------------------------------------------------------------
# Layer boundaries: all-reduce + residual + RMSNorm
# --------------------------------------------------------------------------------------------------------------------


def _can_mc(x, folded: bool = False) -> bool:
    """Whether the generic boundary takes the multicast all-reduce + residual + RMSNorm (``GENERIC_MC_AR_MAX_TOKENS``,
    with the shared + routed add folded in ``GENERIC_MC_AR_FOLDED_MAX_TOKENS``)."""
    return (
        isinstance(x, torch.Tensor)
        and x.dim() == 2
        and x.shape[0] <= (GENERIC_MC_AR_FOLDED_MAX_TOKENS if folded else GENERIC_MC_AR_MAX_TOKENS)
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
    )


def _mc_allreduce_norm(core, x, x2, residual, norm_w, *, quant=False, prepushed=0, capture=None):
    """The multicast all-reduce + optional second addend + residual + RMSNorm (``m3_mc_allreduce_norm``), or None
    without a workspace or for operands the kernel does not take (the same on every rank). With ``quant`` the normed
    rows are also published, MXFP8-quantized, to ``m3_prequant.SLOT`` for the next MXFP8 projection; ``capture`` =
    (out, fc_norm weight, eps): the boundary also writes the Eagle3 capture norm of the updated residual. Up to
    ``MC_AR_CLUSTER_MAX_ROWS`` rows the one-shot 8-CTA cluster form runs; ``MC_AR_CARVEOUT`` sets its shared-memory
    carveout."""
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
        oneshot_cluster=x.shape[0] <= MC_AR_CLUSTER_MAX_ROWS,
        carveout=MC_AR_CARVEOUT,
        capture=capture,
    )


# The CTXNORMQ hand-offs: the boundary's MXFP8 activation for the first prefill MXFP8 projection reading the normed
# rows, and its NVFP4 form for the routed experts. Separate from ``m3_prequant.SLOT`` so only a CTXNORMQ boundary
# feeds the prefill path.
_CTXNORMQ_MX = m3_prequant.PrequantSlot()
_CTXNORMQ_FP4 = m3_prequant.PrequantSlot()
_CTXNORMQ_STATE = {}
# TP2AGG_ROUTER_F32: the boundary's (normed, fp32 copy) for the router of the same MoE layer.
_CTXNORMQ_F32 = m3_prequant.PrequantSlot()
# TP2AGG_SP: the pre-MoE boundary's gathered full-T router logits for the keyed view.
_SP_LOGITS = m3_prequant.PrequantSlot()


def _ctxnormq_op():
    """``torch.ops.trtllm.minimax_m31_ctx_residual_rmsnorm_mxfp8``, or None. A build without the op in its library
    loads it from ``tensorrt_llm/libs/libm31_ctxnormq.so`` when that file exists (an overlay); otherwise the boundary
    keeps the fused all-reduce. Decided once per process."""
    if "op" not in _CTXNORMQ_STATE:
        op = getattr(torch.ops.trtllm, "minimax_m31_ctx_residual_rmsnorm_mxfp8", None)
        if op is None:
            import tensorrt_llm

            lib = os.path.join(os.path.dirname(tensorrt_llm.__file__), "libs", "libm31_ctxnormq.so")
            if os.path.exists(lib):
                torch.ops.load_library(lib)
                op = getattr(torch.ops.trtllm, "minimax_m31_ctx_residual_rmsnorm_mxfp8", None)
        _CTXNORMQ_STATE["op"] = op
        logger.info(
            f"[M3 dispatch] CTXNORMQ prefill boundary norm + quantize: "
            f"{'engaged' if op is not None else 'not taken: the op is not built'} (rows >= {CTXNORMQ_MIN_ROWS})"
        )
    return _CTXNORMQ_STATE["op"]


def _ctxnormq_f32_op():
    """``torch.ops.trtllm.minimax_m31_ctx_residual_rmsnorm_mxfp8_f32`` (TP2AGG_ROUTER_F32), or None: switched off, or
    neither the build nor the overlay ``tensorrt_llm/libs/libm31_agg_ctxnormq_f32.so`` has it. Decided once per
    process."""
    if "f32" not in _CTXNORMQ_STATE:
        op = (
            getattr(torch.ops.trtllm, "minimax_m31_ctx_residual_rmsnorm_mxfp8_f32", None)
            if ROUTER_F32
            else None
        )
        if ROUTER_F32 and op is None:
            import tensorrt_llm

            lib = os.path.join(
                os.path.dirname(tensorrt_llm.__file__), "libs", "libm31_agg_ctxnormq_f32.so"
            )
            if os.path.exists(lib):
                torch.ops.load_library(lib)
                op = getattr(torch.ops.trtllm, "minimax_m31_ctx_residual_rmsnorm_mxfp8_f32", None)
                m31_fake_ops.register()
        _CTXNORMQ_STATE["f32"] = op
        why = "switched off" if not ROUTER_F32 else "the op is not built"
        logger.info(
            f"[M3 dispatch] TP2AGG_ROUTER_F32 router fp32 input from the CTXNORMQ boundary: "
            f"{'engaged' if op is not None else 'not taken: ' + why}"
        )
    return _CTXNORMQ_STATE["f32"]


def _sp_supports(num_tokens: int) -> bool:
    """Whether a step of ``num_tokens`` rows is in TP2AGG_SP's band (the workspace still has to exist)."""
    return TP2AGG_SP and SP_MIN_ROWS <= num_tokens <= SP_MAX_ROWS


def _sp_own_rows(num_tokens: int, rank: int) -> Tuple[int, int]:
    """(offset, count) of the rows ``rank`` (0 or 1) owns in a TP2AGG_SP step: rank 0 [0, S), rank 1 [S, T), S the
    128-row boundary nearest T / 2 (tp2ctx_ipc ``split_row``; T / 2 when T % 256 == 0)."""
    tile = SP_SF_TILE_ROWS
    split = min(max((num_tokens // 2 + tile // 2) // tile * tile, tile), num_tokens - 1)
    return (0, split) if rank == 0 else (split, num_tokens - split)


def _sp_own_sf_bytes(num_tokens: int, rank: int, hidden: int) -> Tuple[int, int]:
    """(start, end) of ``rank``'s bytes in the swizzled MXFP8 scale factors of a T-row step: the 128-row tiles of its
    rows; rank 1's last tile includes the pad rows up to the step's 128-row multiple (tp2ctx_ipc ``own_sf_slice``)."""
    per = (hidden // 32 + 3) // 4 * 4
    split = _sp_own_rows(num_tokens, 0)[1]
    rows_pad = (num_tokens + SP_SF_TILE_ROWS - 1) // SP_SF_TILE_ROWS * SP_SF_TILE_ROWS
    return (0, split * per) if rank == 0 else (split * per, rows_pad * per)


def _sp_logits_bytes(num_tokens: int, rank: int) -> Tuple[int, int]:
    """(start, end) of ``rank``'s owned rows in the "logits" gather view ([T, 128] fp32, whole rows)."""
    off, count = _sp_own_rows(num_tokens, rank)
    return off * SP_LOGITS_ROW_BYTES, (off + count) * SP_LOGITS_ROW_BYTES


class _SpOvl(NamedTuple):
    """An SP boundary's asynchronous push (SP_OVL): the event the peer's rows wait on and the two row / scale ranges."""

    event: Any
    own_rows: Tuple[int, int]
    peer_rows: Tuple[int, int]
    own_sf: slice
    peer_sf: slice


def _sp_once(key, message: str) -> None:
    """One info line per key (engagement lines a gate job greps)."""
    noted = _SP_STATE.setdefault("noted", set())
    if key not in noted:
        noted.add(key)
        logger.info(message)


def _sp_load(op_name: str, lib_name: str):
    """``torch.ops.trtllm.<op_name>``, loading ``tensorrt_llm/libs/<lib_name>`` first when the build lacks it."""
    if not hasattr(torch.ops.trtllm, op_name):
        import tensorrt_llm

        lib = os.path.join(os.path.dirname(tensorrt_llm.__file__), "libs", lib_name)
        if os.path.exists(lib):
            torch.ops.load_library(lib)
    return getattr(torch.ops.trtllm, op_name, None)


def _sp_init(core) -> None:
    """Load the peer ops and build the IPC workspace (collective: both TP ranks, at weight load, outside capture,
    before the KV cache is sized), plus the zero-copy NORMQ out= and SP_ADDNORM ops. Once per process."""
    if not TP2AGG_SP or "ws" in _SP_STATE:
        return
    ws, why = None, None
    if getattr(core.mapping, "tp_size", 1) != 2:
        why = f"tp_size {getattr(core.mapping, 'tp_size', None)} != 2"
    elif _sp_load("minimax_m31_tp2ctx_ipc_barrier", _SP_LIB) is None:
        why = f"no peer ops (tensorrt_llm/libs/{_SP_LIB})"
    else:
        try:
            from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import tp2ctx_ipc

            ws = tp2ctx_ipc.get_workspace(
                core.mapping,
                max_tokens=SP_MAX_ROWS,
                hidden=core.hidden,
                device=torch.device("cuda", torch.cuda.current_device()),
            )
        except (ImportError, AttributeError) as e:
            why = f"no IPC workspace ({type(e).__name__}: {str(e)[:120]})"
    _SP_STATE["ws"] = ws
    if ws is None:
        logger.info(f"[M3.1 agg] TP2AGG_SP not taken: {why}")
        return
    normq_out = _sp_load("minimax_m31_ctx_residual_rmsnorm_mxfp8_out", _SP_NORMQ_OUT_LIB)
    addnorm = None
    if SP_ADDNORM and normq_out is not None and not SP_CHECK:
        addnorm = _sp_load("minimax_m31_tp2ctx_peer_add_normq_out", _SP_ADDNORM_LIB)
    _SP_STATE["normq_out"], _SP_STATE["addnorm"] = normq_out, addnorm
    logger.info(
        f"[M3.1 agg] TP2AGG_SP workspace ready: NORMQ out= {'on' if normq_out is not None else 'off'}, "
        f"SP_ADDNORM {'on' if addnorm is not None else 'off'}, SP_CHECK {'on' if SP_CHECK else 'off'}"
    )


def _sp_ws(x: torch.Tensor):
    """The workspace when a boundary over ``x``'s rows runs sequence-parallel (eager, rows in the band), else None."""
    ws = _SP_STATE.get("ws")
    if (
        ws is None
        or x.dim() != 2
        or not _sp_supports(x.shape[0])
        or not ws.supports(x.shape[0], nccl_only=False)
        or torch.cuda.is_current_stream_capturing()
        or is_torch_compiling()
        or is_in_breakable_cuda_graph()
    ):
        return None
    return ws


def _sp_consumer_ws(core, x):
    """SP publishes a keyed placeholder, so every consumer must accept its prequantized products."""
    if not core.fused_projection:
        return None
    mode = _mx_mode(core.mxfp8)
    if mode.decode_graph or mode.tuned_quantize_gemm:
        return None
    return _sp_ws(x)


def _sp_count(rows: int, owned: int, kind: str) -> None:
    """Engagement: per step-size band, logged at boundary counts 1, 2, 4, ... (read the last line per band)."""
    band = "<=8K" if rows <= 8192 else "<=12K" if rows <= 12288 else ">12K"
    counts = _SP_STATE.setdefault("counts", {})
    counts[band] = n = counts.get(band, 0) + 1
    if n & (n - 1) == 0:
        logger.info(
            f"[M3.1 agg] TP2AGG_SP taken band={band} n={n} (rows {rows}, owned {owned}, {kind})"
        )
    m3_report.work("TP2AGG_SP", f"{rows} rows, {kind}")


def _sp_put(ws, kind: str, rows: int, owned: torch.Tensor) -> torch.Tensor:
    """Copy this rank's owned rows of ``kind`` into the full-T gather view; returns the view."""
    first, count = ws.own_rows(rows)
    view = ws.gather_view(kind, rows)
    if kind == "mxfp8_sf":
        sl = ws.own_sf_slice(rows)
        src = owned.view(torch.uint8).reshape(-1)
        if src.numel() != sl.stop - sl.start:
            raise RuntimeError(
                f"TP2AGG_SP: MXFP8 scale factors {src.numel()} bytes != own_sf_slice {sl}"
            )
        view[sl].copy_(src)
    elif kind == "logits":
        view[first : first + count].copy_(owned)
    else:
        view[first : first + count].copy_(owned.view(torch.uint8).view(count, -1))
    return view


def _sp_unshard(residual, x):
    """The full-T residual: an owned-only one (inside an SP step) is gathered from both ranks first (through the
    workspace's bf16 view, then cloned: the next gather reuses the view). Anything else is returned as is."""
    ws = _SP_STATE.get("ws")
    if (
        ws is None
        or not isinstance(residual, torch.Tensor)
        or not isinstance(x, torch.Tensor)
        or residual.dim() != 2
        or x.dim() != 2
        or residual.shape[0] == x.shape[0]
        or residual.shape[0] != ws.own_rows(x.shape[0])[1]
    ):
        return residual
    rows = x.shape[0]
    first, count = ws.own_rows(rows)
    view = ws.gather_view("bf16", rows)
    view[first : first + count].copy_(residual)
    ws.push_owned(("bf16",), rows)
    _sp_once(
        ("unshard", rows), f"[M3.1 agg] TP2AGG_SP residual unshard: rows {rows} (owned {count})"
    )
    return view.clone()


def _sp_ovl_take(x: torch.Tensor):
    """The SP_OVL push pending for the keyed tensor ``x`` (cleared), or None."""
    pending, _SP_STATE["ovl"] = _SP_STATE.get("ovl"), None
    if pending is None or x.dim() != 2 or pending[0] != x.data_ptr():
        if pending is not None:
            torch.cuda.current_stream().wait_event(pending[1].event)
        return None
    return pending[1]


def _sp_ovl_wait(x: torch.Tensor) -> None:
    """A consumer that reads the whole gathered activation: wait for a pending SP_OVL push first."""
    if _SP_STATE.get("ovl") is not None:
        ovl = _sp_ovl_take(x)
        if ovl is not None:
            torch.cuda.current_stream().wait_event(ovl.event)


def _sp_boundary(core, ws, op, x, residual, norm_w, fp4_scale, allreduce, router_w, qkv_a1_next):
    """The CTXNORMQ boundary on this rank's owned rows: (the keyed bf16 view, the owned updated residual). The
    gathered MXFP8 (+ NVFP4) products and, at a pre-MoE boundary, the gathered router logits are published for the
    keyed view, so the consumers take them as today."""
    rows = x.shape[0]
    first, count = ws.own_rows(rows)
    pre_moe = fp4_scale is not None
    slot = ws.input_slot(rows)
    ref = None
    if SP_CHECK:
        ref = m3_allreduce(x.clone(), core.allreduce if allreduce is None else allreduce, None)[
            first : first + count
        ]
    if x.data_ptr() != slot.data_ptr():
        slot.copy_(x)
    normq_out = _SP_STATE.get("normq_out")
    addnorm = _SP_STATE.get("addnorm") if normq_out is not None else None
    if addnorm is not None:
        owned = torch.empty((count, x.shape[1]), dtype=x.dtype, device=x.device)
    else:
        owned = ws.reduce_scatter_owned(slot)
    if ref is not None and addnorm is None:
        # The reduce-scatter's own rows (SP_ADDNORM fuses the peer add into its norm instead: its bitwise oracle is
        # mb_addnorm, so a check run kills SP_ADDNORM).
        diff = int((owned.view(torch.int16) != ref.view(torch.int16)).sum())
        checked = _SP_STATE["checked"] = _SP_STATE.get("checked", 0) + 1
        if diff or checked <= 4:
            logger.info(
                f"[M3.1 agg] TP2AGG_SP check: rows {rows} owned reduce-scatter differs from the all-reduce in "
                f"{diff} elements (boundaries checked {checked})"
            )
    res = (residual[first : first + count] if residual.shape[0] == rows else residual).contiguous()
    kinds = ["mxfp8", "mxfp8_sf"] + (["nvfp4", "nvfp4_sf"] if pre_moe else [])
    pushed = False
    x32 = None
    fp4 = fp4_sf = None
    if normq_out is not None:
        # Zero-copy: the NORMQ op writes this rank's products straight into the gather views at the owned rows.
        gq, gsf = ws.gather_view("mxfp8", rows), ws.gather_view("mxfp8_sf", rows)
        g4 = ws.gather_view("nvfp4", rows) if pre_moe else None
        g4sf = ws.gather_view("nvfp4_sf", rows) if pre_moe else None
        normed = torch.empty_like(owned) if pre_moe else None
        if pre_moe and ROUTER_F32:
            # ROUTER_ZC: the fp32 router rows land in the router's full-step buffer (no copy, no zero fill).
            x32 = ws.router_owned_input(rows, owned.shape[1])
            _sp_once(
                ("router_zc", x32 is not None, rows),
                f"[M3.1 agg] TP2AGG_ROUTER_ZC {'taken' if x32 is not None else 'not taken'}: rows {rows}",
            )
            if x32 is None:
                x32 = torch.empty(owned.shape, dtype=torch.float32, device=owned.device)
        outs_args = (
            res,
            norm_w,
            core.rms_eps,
            normed,
            gq[first : first + count],
            gsf[ws.own_sf_slice(rows)],
            None if g4 is None else g4[first : first + count],
            None if g4sf is None else g4sf[first : first + count],
            fp4_scale,
            x32,
        )
        if addnorm is not None:
            local_ptr, peer_ptr = ws.owned_slot_ptrs(rows)
            ws.barrier()
            addnorm(local_ptr, peer_ptr, owned, *outs_args, 1, False)
            _sp_once(
                ("addnorm", rows, pre_moe),
                f"[M3.1 agg] TP2AGG_SP_ADDNORM taken: {'pre-MoE' if pre_moe else 'boundary'} rows {rows}",
            )
        else:
            normq_out(owned, *outs_args, False)
        xq, xsf = gq.view(torch.float8_e4m3fn), gsf
        if pre_moe:
            fp4, fp4_sf = g4, g4sf.reshape(-1)
            if x32 is None:
                x32 = normed.float()
            if SP_PMOVL and rows >= SP_PMOVL_MIN_ROWS:
                # SP_PMOVL: the products go now, under the router GEMM; the logits and the barrier follow.
                ws.push_owned_async(kinds, rows, engine="sm", barrier=False)
                pushed = True
                _sp_once(("pmovl", rows), f"[M3.1 agg] TP2AGG_SP_PMOVL taken: pre-MoE rows {rows}")
    else:
        f32_op = _ctxnormq_f32_op() if pre_moe else None
        pdl = get_env_enable_pdl()
        if f32_op is not None:
            outs = f32_op(owned, res, norm_w, core.rms_eps, fp4_scale, pdl)
        else:
            outs = op(owned, res, norm_w, core.rms_eps, True, fp4_scale, pdl)
        xq = _sp_put(ws, "mxfp8", rows, outs[1]).view(outs[1].dtype)
        xsf = _sp_put(ws, "mxfp8_sf", rows, outs[2]).view(outs[2].dtype)
        if pre_moe:
            fp4 = _sp_put(ws, "nvfp4", rows, outs[3]).view(outs[3].dtype)
            fp4_sf = _sp_put(ws, "nvfp4_sf", rows, outs[4]).view(outs[4].dtype).reshape(-1)
            x32 = outs[5] if f32_op is not None else outs[0].float()
    logits = None
    if pre_moe:
        # The owned rows' logits from the same full-T router call as today (bitwise per row), gathered as a kind.
        logits = _sp_put(ws, "logits", rows, ws.route_owned_full_shape(x32, router_w, rows))
        kinds = ["logits"] if pushed else kinds + ["logits"]
    ovl = None
    if (
        not pre_moe
        and qkv_a1_next
        and QKV_A1
        and SP_OVL
        and rows >= SP_OVL_MIN_ROWS
        and normq_out is not None
    ):
        event = ws.push_owned_async(kinds, rows)
        peer_first = first + count if first == 0 else 0
        ovl = _SpOvl(
            event,
            (first, count),
            (peer_first, rows - count),
            ws.own_sf_slice(rows),
            ws.peer_sf_slice(rows),
        )
    else:
        ws.push_owned(kinds, rows)
    keyed = ws.gather_view("bf16", rows)
    if SP_CHECK:
        if ovl is not None:
            ws.wait_pending_push()
            ovl = None
        keyed.fill_(float("nan"))
    _CTXNORMQ_MX.publish(keyed, xq, xsf)
    _SP_STATE["ovl"] = (keyed.data_ptr(), ovl) if ovl is not None else None
    if pre_moe:
        _CTXNORMQ_FP4.publish(keyed, fp4, fp4_sf)
        _SP_LOGITS.publish(keyed, logits, None)
    else:
        _CTXNORMQ_FP4.clear()
        _SP_LOGITS.clear()
    _CTXNORMQ_F32.clear()
    _SP_STATE["keyed"] = keyed.data_ptr()
    _record("m31_tp2agg_sp", rows, pre_moe, normq_out is not None)
    _sp_count(rows, count, "pre-MoE" if pre_moe else "boundary")
    return keyed, owned


def _sp_keyed(h: torch.Tensor) -> bool:
    """Whether ``h`` is the keyed view of this step's last SP boundary (its bf16 rows hold no data)."""
    return h.dim() == 2 and _SP_STATE.get("keyed") == h.data_ptr()


def _ctxnormq_allreduce_norm(
    core,
    x,
    residual,
    norm_w,
    fp4_scale,
    allreduce=None,
    router_w=None,
    qkv_a1_next=False,
    pre_moe=False,
):
    """A prefill boundary as the plain all-reduce + the fused residual add, RMSNorm and quantize, or None (switched
    off, fewer than ``CTXNORMQ_MIN_ROWS`` rows, op not built). Returns (normed, updated) like the fused all-reduce;
    the MXFP8 activation (and with ``fp4_scale`` the NVFP4 one) wait in the CTXNORMQ slots for their consumers.
    ``allreduce``: the all-reduce to run (default the target's AUTO one). TP2AGG_SP: a step in its band runs the
    boundary sequence-parallel (``_sp_boundary``; a pre-MoE one needs ``router_w``, the layer's router weight;
    ``qkv_a1_next``: the next projection is a sparse layer's QKV, for SP_OVL) and returns (keyed view, owned
    residual)."""
    if not CTXNORMQ or x.dim() != 2 or x.shape[0] < CTXNORMQ_MIN_ROWS:
        return None
    op = _ctxnormq_op()
    if op is None:
        return None
    ws = (
        _sp_consumer_ws(core, x)
        if TP2AGG_SP and (not pre_moe or SP_PREMOE) and (fp4_scale is None or router_w is not None)
        else None
    )
    if ws is not None:
        return _sp_boundary(
            core, ws, op, x, residual, norm_w, fp4_scale, allreduce, router_w, qkv_a1_next
        )
    residual = _sp_unshard(residual, x)
    reduced = m3_allreduce(x, core.allreduce if allreduce is None else allreduce, None)
    _record("m31_ctxnormq", x.shape[0], fp4_scale is not None)
    f32_op = _ctxnormq_f32_op() if fp4_scale is not None else None
    if f32_op is not None:
        normed, xq, xsf, xfp4, xfp4_sf, normed_f32 = f32_op(
            reduced, residual.contiguous(), norm_w, core.rms_eps, fp4_scale, get_env_enable_pdl()
        )
        _CTXNORMQ_F32.publish(normed, normed_f32, None)
    else:
        normed, xq, xsf, xfp4, xfp4_sf = op(
            reduced,
            residual.contiguous(),
            norm_w,
            core.rms_eps,
            True,
            fp4_scale,
            get_env_enable_pdl(),
        )
        _CTXNORMQ_F32.clear()
    _CTXNORMQ_MX.publish(normed, xq, xsf)
    if fp4_scale is not None:
        _CTXNORMQ_FP4.publish(normed, xfp4, xfp4_sf)
    else:
        _CTXNORMQ_FP4.clear()
    m3_report.work(
        "CTXNORMQ", f"{x.shape[0]} rows, NVFP4 {'on' if fp4_scale is not None else 'off'}"
    )
    return normed, reduced


def _ctxnormq_mxfp8_gemm(core, prequant, w, s, mode):
    """The GEMM ``m3_mxfp8_linear`` runs after its own quantize, on the CTXNORMQ activation (bit-identical operands)."""
    act, act_scale = prequant
    _record("m31_ctxnormq_gemm", tuple(w.shape), act.shape[0], mode.flashinfer, mode.native_tuning)
    if mode.flashinfer:
        return torch.ops.trtllm.flashinfer_mm_mxfp8(act, act_scale, w, s, torch.bfloat16)
    gemm = (
        torch.ops.trtllm.mxfp8_mxfp8_gemm_autotuned
        if mode.native_tuning
        else torch.ops.trtllm.mxfp8_mxfp8_gemm
    )
    return gemm(act, act_scale, w, s, core.mx_alpha, torch.bfloat16)


# TP2WIDE_COMBINE_MXFP8: the combine's MXFP8 copy is written only for calls of at least this many rows (msa_prefill
# reads the sink's ``min_rows``); at 8K rows the extra epilogue costs more than o_proj's quantize saves (tp2ctx
# A/B). The agg target's prefill chunks are 16K rows.
_COMBINE_MXFP8_MIN_ROWS = 12288
_COMBINE_MXFP8_STATE: dict = {}
_COMBINE_MXFP8_CHECK = os.environ.get("TRTLLM_M31_TP2WIDE_COMBINE_MXFP8_CHECK") == "1"


def _combine_mxfp8_allocate(out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Give the context combine views into the armed full-step quantized buffers."""
    pending = _COMBINE_MXFP8_STATE.get("pending")
    if pending is None:
        return None
    output, context_rows = pending["output"], pending["context_rows"]
    if out.data_ptr() != output.data_ptr() or out.shape != (context_rows, output.shape[1]):
        raise RuntimeError("TP2WIDE_COMBINE_MXFP8 context output differs from the armed step")
    rows, width = output.shape
    pair = (
        torch.empty((rows, width), dtype=torch.uint8, device=output.device),
        torch.zeros(
            (((rows + 127) // 128 * 128) * width // 32,), dtype=torch.uint8, device=output.device
        ),
    )
    pending["pair"] = pair
    return pair[0][:context_rows], pair[1][: ((context_rows + 127) // 128 * 128) * width // 32]


def _combine_mxfp8_sink(out: torch.Tensor, xq: torch.Tensor, xsf: torch.Tensor) -> None:
    """Record the context combine; publish only after generation and partial-tile repair."""
    pending = _COMBINE_MXFP8_STATE.get("pending")
    if pending is None:
        return
    output, context_rows = pending["output"], pending["context_rows"]
    pair = pending.get("pair")
    if (
        pair is None
        or out.data_ptr() != output.data_ptr()
        or out.shape != (context_rows, output.shape[1])
        or xq.data_ptr() != pair[0].data_ptr()
        or xsf.data_ptr() != pair[1].data_ptr()
    ):
        raise RuntimeError("TP2WIDE_COMBINE_MXFP8 combine did not use the caller's buffers")
    pending["fired"] = True


def _combine_mxfp8_finish(output: torch.Tensor) -> None:
    """Repair the last context tile plus generation rows, then publish the full C+G identity."""
    pending = _COMBINE_MXFP8_STATE.pop("pending", None)
    if pending is None or not pending.get("fired", False):
        return
    if output is not pending["output"]:
        raise RuntimeError("TP2WIDE_COMBINE_MXFP8 output changed before publication")
    context_rows = pending["context_rows"]
    xq, xsf = pending["pair"]
    tail_start = context_rows // 128 * 128
    if tail_start < output.shape[0]:
        tail_q, tail_sf = torch.ops.trtllm.mxfp8_quantize(output[tail_start:], True)
        xq[tail_start:].copy_(tail_q.view(torch.uint8))
        offset = tail_start * output.shape[1] // 32
        xsf[offset : offset + tail_sf.numel()].copy_(tail_sf.view(torch.uint8).flatten())
    if _COMBINE_MXFP8_CHECK:
        ref_q, ref_sf = torch.ops.trtllm.mxfp8_quantize(output, True)
        if not torch.equal(xq, ref_q.view(torch.uint8)) or not torch.equal(
            xsf, ref_sf.view(torch.uint8).flatten()
        ):
            raise RuntimeError("TP2WIDE_COMBINE_MXFP8_CHECK full-step quantization mismatch")
    if "engaged" not in _COMBINE_MXFP8_STATE:
        _COMBINE_MXFP8_STATE["engaged"] = True
        logger.info(
            f"[M3.1] TP2WIDE_COMBINE_MXFP8 engaged: context {context_rows}, total {output.shape[0]} rows"
        )
    _COMBINE_MXFP8_STATE["src"] = output.data_ptr()
    _CTXNORMQ_MX.publish(output, xq.view(torch.float8_e4m3fn), xsf)


_combine_mxfp8_sink.min_rows = _COMBINE_MXFP8_MIN_ROWS
_combine_mxfp8_sink.allocate = _combine_mxfp8_allocate


_GQAMIX_STATE: dict = {}


def _gqamix_decode(*args, **kwargs) -> None:
    """``minimax_m3_sparse_attn_decode`` for MsaDecodeFmha's sparse generation rows, through the target-local copy."""
    if _lc2_mixed.lc2_mixed_decode(*args, **kwargs):
        return
    if not _GQAMIX_STATE.get("logged"):
        _GQAMIX_STATE["logged"] = True
        logger.info(
            "[M3 dispatch] GQAMIX: mixed-step sparse decode through ops/gqamix_sparse_decode"
        )
    _gqamix_kernel.minimax_m3_sparse_attn_decode(*args, **kwargs)
    m3_report.work("GQAMIX", "mixed-step sparse generation rows")


_TSEL_STATE: dict = {}
_TSEL_ORACLE = {
    "calls": 0, "rows": 0, "rows_diff": 0, "rows_tie_only": 0, "rows_non_tie": 0,
    "t_last": 0.0, "t_call": 0.0, "flushed_calls": 0,
}  # fmt: skip
_TSEL_ORACLE_IDLE_S = 15.0


def _tsel_op():
    """``torch.ops.trtllm.minimax_m31_select_prefill_tile`` on sm_107, or None. A build without the op loads it from
    ``tensorrt_llm/libs/libm31_tsel.so`` when that file exists (an overlay). Decided once per process."""
    if "op" not in _TSEL_STATE:
        op = None
        if torch.cuda.get_device_capability() == (10, 7):
            op = getattr(torch.ops.trtllm, "minimax_m31_select_prefill_tile", None)
            if op is None:
                import tensorrt_llm

                lib = os.path.join(os.path.dirname(tensorrt_llm.__file__), "libs", "libm31_tsel.so")
                if os.path.exists(lib):
                    torch.ops.load_library(lib)
                    op = getattr(torch.ops.trtllm, "minimax_m31_select_prefill_tile", None)
        _TSEL_STATE["op"] = op
        print(
            f"[M3 dispatch] AGG_TSEL prefill tile select rank {mpi_rank()}: "
            f"{'engaged' if op is not None else 'not taken: not sm_107 or the op is not built'} "
            f"(rows >= {AGG_TSEL_MIN_ROWS}, blocks <= {AGG_TSEL_MAX_BLOCKS}, oracle {int(AGG_TSEL_ORACLE)})",
            file=sys.stderr,
            flush=True,
        )
    return _TSEL_STATE["op"]


def _tsel_oracle_flush(final: bool = False) -> None:
    st = _TSEL_ORACLE
    st["flushed_calls"] = st["calls"]
    print(
        f"AGG_TSEL_ORACLE{'_FINAL' if final else ''} rank {mpi_rank()} calls {st['calls']} rows {st['rows']} "
        f"rows_diff {st['rows_diff']} tie_only {st['rows_tie_only']} non_tie {st['rows_non_tie']}",
        file=sys.stderr,
        flush=True,
    )


def _tsel_oracle_check(max_score_kv, nvb, out, init_blocks, local_blocks) -> None:
    """Compare the tile table with minimax_m3_select_blocks'; a differing row is tie-only when both tables select the
    same multiset of score bit patterns (only the order among equal keys differs), else non-tie."""
    st = _TSEL_ORACLE
    ref = torch.ops.trtllm.minimax_m3_select_blocks(
        max_score_kv, nvb, MSA_REQUIRED_TOPK, init_blocks, local_blocks, True
    )
    total_q = int(out.shape[0])
    st["calls"] += 1
    st["rows"] += total_q * int(out.shape[1])
    diff = (out != ref).any(dim=-1)  # [total_q, num_kv_heads]
    n_diff = int(diff.sum())
    if n_diff:
        q_idx, h_idx = diff.nonzero(as_tuple=True)
        bits = max_score_kv.view(torch.int32)  # [heads, blocks, total_q]

        def picked(table):
            rows = table[q_idx, h_idx].long()  # [n, topk]
            vals = bits[h_idx[:, None], rows.clamp(min=0), q_idx[:, None]]
            vals = torch.where(rows >= 0, vals, torch.full_like(vals, -(2**31)))
            return vals.sort(dim=-1).values

        tie = (picked(out) == picked(ref)).all(dim=-1)
        n_tie = int(tie.sum())
        st["rows_diff"] += n_diff
        st["rows_tie_only"] += n_tie
        st["rows_non_tie"] += n_diff - n_tie
        if n_diff - n_tie and st["rows_non_tie"] == n_diff - n_tie:
            print(
                f"AGG_TSEL_ORACLE first non-tie rank {mpi_rank()}: {n_diff - n_tie} rows; "
                f"blocks {max_score_kv.shape[1]}, "
                f"rows {total_q}, init {init_blocks}, local {local_blocks}",
                file=sys.stderr,
                flush=True,
            )
    now = time.monotonic()
    st["t_call"] = now
    if now - st["t_last"] >= 30.0:
        st["t_last"] = now
        _tsel_oracle_flush()


def _tsel_oracle_idle_watch() -> None:
    """Daemon: flush once the tile select has been idle for 15 s with unflushed calls (a server killed at teardown
    never reaches atexit, so its last interval lands here)."""
    st = _TSEL_ORACLE
    while True:
        time.sleep(5.0)
        if (
            st["calls"] != st["flushed_calls"]
            and time.monotonic() - st["t_call"] >= _TSEL_ORACLE_IDLE_S
        ):
            _tsel_oracle_flush()


def _tsel_oracle_install_flushes() -> None:
    """Once per process: the final flush at exit, on SIGTERM (then the previous handler), and after 15 s idle."""
    atexit.register(_tsel_oracle_flush, True)
    threading.Thread(
        target=_tsel_oracle_idle_watch, name="agg_tsel_oracle_flush", daemon=True
    ).start()
    try:
        prev = signal.getsignal(signal.SIGTERM)

        def _on_term(signum, frame):
            _tsel_oracle_flush(True)
            if callable(prev):
                prev(signum, frame)
            elif prev == signal.SIG_DFL:
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                os.kill(os.getpid(), signal.SIGTERM)

        signal.signal(signal.SIGTERM, _on_term)
    except ValueError:  # not the main thread: atexit and the idle flush still cover it
        pass


def _tsel_select(max_score_kv, n_valid_blocks, init_blocks, local_blocks):
    """The MSA indexer's selector hook: the head-major [total_q, num_kv_heads, 16] table of
    ``select_blocks_from_maxscore``, or None (outside the tile select's rows / blocks range, or no op) to keep it."""
    if (
        torch.cuda.is_current_stream_capturing()  # the host-side rows / blocks gate is not graph-stable
        or max_score_kv.dtype != torch.float32
        or max_score_kv.dim() != 3
        or max_score_kv.stride(2) != 1
        or max_score_kv.shape[2] < AGG_TSEL_MIN_ROWS
        or max_score_kv.shape[1] > AGG_TSEL_MAX_BLOCKS
    ):
        return None
    op = _tsel_op()
    if op is None:
        return None
    num_kv_heads, num_blocks, total_q = max_score_kv.shape
    _record("m31_agg_tsel", num_blocks, total_q >= 1024)
    out = torch.empty(
        (num_kv_heads, total_q, MSA_REQUIRED_TOPK), dtype=torch.int32, device=max_score_kv.device
    ).permute(1, 0, 2)
    nvb = n_valid_blocks.to(device=max_score_kv.device, dtype=torch.int32).contiguous()
    op(max_score_kv, nvb, out, init_blocks, local_blocks)
    if not _TSEL_STATE.get("taken"):
        _TSEL_STATE["taken"] = True
        print(
            f"AGG_TSEL taken rank {mpi_rank()}: blocks {num_blocks}, rows {total_q}",
            file=sys.stderr,
            flush=True,
        )
    if AGG_TSEL_ORACLE:
        _tsel_oracle_check(max_score_kv, nvb, out, init_blocks, local_blocks)
    return out


def _tsel_register(backend) -> None:
    """Register ``_tsel_select`` on the layer backend's MSA indexer (an indexer without the hook is left alone)."""
    indexer = getattr(backend, "indexer", None)
    if indexer is not None and getattr(indexer, "selector_hook", False) is None:
        indexer.selector_hook = _tsel_select
        if AGG_TSEL_ORACLE and not _TSEL_STATE.get("flushes"):
            _TSEL_STATE["flushes"] = True
            _tsel_oracle_install_flushes()


def _gqamix_register(backend) -> None:
    """Register ``_gqamix_decode`` on the layer backend's MsaDecodeFmha (a library without the hook is left alone)."""
    manager = getattr(backend, "_fmha_manager", None)
    for fmha in getattr(manager, "fmha_libs", ()):
        if getattr(fmha, "sparse_decode_hook", False) is None:
            fmha.sparse_decode_hook = _gqamix_decode


def _native_allreduce_norm(core, x, residual, norm_w, allreduce=None):
    """The built-in fused all-reduce + residual + RMSNorm (``allreduce``: default the target's AUTO one). An
    owned-only TP2AGG_SP residual (the final norm, a fallback) is unsharded first."""
    residual = _sp_unshard(residual, x)
    _record("m3_allreduce", x.shape[0])
    return m3_allreduce(
        x,
        core.allreduce if allreduce is None else allreduce,
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


_S2_MX = m3_prequant.PrequantSlot()
_S2_STATE: dict = {}


def _s2_op():
    """``torch.ops.trtllm.minimax_m31_s2_moe_tail`` (TP2W_S2_TAIL), or None: switched off, or the build does not
    carry it. Decided once per process."""
    if "op" not in _S2_STATE:
        op = getattr(torch.ops.trtllm, "minimax_m31_s2_moe_tail", None) if TP2W_S2_TAIL else None
        _S2_STATE["op"] = op
        why = "switched off" if not TP2W_S2_TAIL else "the op is not built"
        logger.info(
            f"[M3.1 agg] TP2W_S2_TAIL fused MoE tail: {'engaged' if op is not None else 'not taken: ' + why}"
        )
    return _S2_STATE["op"]


def _s2_sf_valid(rows: int, hidden: int, device) -> torch.Tensor:
    """TP2W_S2_CHECK: which bytes of a swizzled (128x4) K32 scale tensor of ``rows`` x ``hidden`` hold valid rows (the
    rest are the pad rows up to the 128-row tile), as a bool mask over its bytes (get_sf_out_offset_128x4)."""
    key = (rows, hidden, str(device))
    mask = _S2_STATE.get(key)
    if mask is None:
        cols = hidden // 32
        padded = (rows + 127) // 128 * 128
        r = torch.arange(padded, device=device).view(-1, 1).expand(padded, cols)
        c = torch.arange(cols, device=device).view(1, -1).expand(padded, cols)
        off = (
            (r // 128) * ((cols + 3) // 4 * 512)
            + (c // 4) * 512
            + (r % 32) * 16
            + ((r % 128) // 32) * 4
            + c % 4
        )
        mask = torch.zeros(padded * ((cols + 3) // 4 * 4), dtype=torch.bool, device=device)
        mask[off.reshape(-1)] = (r < rows).reshape(-1)
        _S2_STATE[key] = mask
    return mask


def _s2_boundary(core, rows: int, parts, shared, residual, norm_w, stock_ar):
    """TP2W_S2_TAIL: the MoE boundary of ``rows`` rows in one launch from the routed experts' unfinalized ``parts``
    and the shared expert's output. Returns (normed, updated) like the fused all-reduce and publishes the next QKV's
    MXFP8 activation (``_S2_MX``), or None (nothing launched) when the operands are not the op's. ``stock_ar`` is the
    all-reduce the boundary would run (the decode layer's AUTO one; a prefill-target boundary's _prefill_allreduce)."""
    op = _s2_op()
    if op is None or not (isinstance(parts, (list, tuple)) and len(parts) == 3):
        return None
    fc2, scales, perm = parts
    lamport = core.allreduce
    ws = getattr(lamport, "workspace", None)
    if (
        ws is None
        or fc2.dtype != torch.bfloat16
        or fc2.dim() != 2
        or fc2.stride(1) != 1
        or shared is None
        or shared.dtype != torch.bfloat16
        or not shared.is_contiguous()
        or residual.shape != shared.shape
        or residual.dtype != torch.bfloat16
    ):
        return None
    scales = scales.reshape(rows, -1).contiguous()
    perm = perm.view(rows, -1).to(torch.int32).contiguous()
    residual = residual.contiguous()
    if "ar" not in _S2_STATE:
        _S2_STATE["ar"] = True
        logger.info(
            f"[M3.1 agg] TP2W_S2_TAIL replaces finalize + {type(stock_ar).__name__}("
            f"{getattr(stock_ar, 'strategy', '?')}) + quantize at {rows} rows (rank {mpi_rank()})"
        )
    ref = None
    # The check reads its counts on the host: eager steps only (a captured decode graph is checked by an eager run).
    if S2_CHECK and not torch.cuda.is_current_stream_capturing():
        x = m3_moe_finalize(fc2, scales, perm, shared)
        ref_n, ref_r = _native_allreduce_norm(core, x, residual, norm_w, stock_ar)
        ref = (ref_n, ref_r) + tuple(torch.ops.trtllm.mxfp8_quantize(ref_n, True))
    _record("m31_s2_moe_tail", rows)
    normed, updated, q, sf = op(
        fc2,
        scales,
        perm,
        shared,
        residual,
        norm_w,
        ws,
        lamport.mapping.tp_rank,
        lamport.mapping.tp_size,
        core.rms_eps,
        False,
    )
    if ref is not None:
        # All four outputs: normed, residual, MXFP8 bytes, and the scales the next QKV consumes (the valid rows'
        # bytes bit for bit; the pad rows' bytes 0 in the fused output, as mxfp8_quantize writes them).
        diffs = [
            int((a.reshape(-1).view(torch.uint8) != b.reshape(-1).view(torch.uint8)).sum())
            for a, b in zip(ref[:3], (normed, updated, q))
        ]
        valid = _s2_sf_valid(rows, normed.shape[1], sf.device)
        diffs.append(int((ref[3].reshape(-1)[valid] != sf.reshape(-1)[valid]).sum()))
        diffs.append(int((sf.reshape(-1)[~valid] != 0).sum()))
        # The decode graph's pinned QKV path (MX-TAC) quantizes with m3_mxfp8_pinned_quantize: the same bytes?
        pq, psf = m3_mxfp8_pinned_quantize(ref[0], MXFP8_DECODE_QUANT_TACTIC)
        pinned = [
            int((pq.reshape(-1).view(torch.uint8) != q.reshape(-1).view(torch.uint8)).sum()),
            int((psf.reshape(-1)[valid] != sf.reshape(-1)[valid]).sum()),
        ]
        n = _S2_STATE["checked"] = _S2_STATE.get("checked", 0) + 1
        if any(diffs) or any(pinned) or n <= 4:
            logger.info(
                f"[M3.1 agg] TP2W_S2_TAIL check: rows {rows} normed / residual / mxfp8 / sf / sf_pad differ from "
                f"the stock boundary in {' / '.join(str(d) for d in diffs)} bytes; vs the pinned quantize "
                f"{pinned[0]} / {pinned[1]} (boundaries checked {n})"
            )
    _S2_MX.publish(normed, q, sf)
    n = _S2_STATE["taken"] = _S2_STATE.get("taken", 0) + 1
    if n & (n - 1) == 0:
        logger.info(f"[M3.1 agg] TP2W_S2_TAIL taken n={n} (rows {rows})")
    return normed, updated


def _router_logits(core, i: int, h: torch.Tensor) -> torch.Tensor:
    """The fp32 router logits of MoE layer ``i`` when the MoE runs unfused (the decode steps' engines route
    in-kernel): ``F.linear(float(h), w)``, the built-in gate's arithmetic outside its tensor-core router's rows. The
    experts quantize their bf16 input themselves."""
    # TP2AGG_SP: the boundary gathered this step's logits (h is its keyed view).
    sp = _SP_LOGITS.take(h)
    if sp is not None:
        _record("m31_tp2agg_sp_logits", h.shape[0])
        return sp[0]
    # TP2AGG_ROUTER_F32: the boundary already wrote float(h).
    f32 = _CTXNORMQ_F32.take(h)
    if f32 is not None:
        _record("m31_router_f32", h.shape[0])
        return torch.nn.functional.linear(f32[0], core.w[f"l{i}_router"])
    _record("m3_fp32_linear", h.shape[0] if h.dim() == 2 else 0)
    return m3_fp32_linear(h, core.w[f"l{i}_router"])


def _decode_router_logits(
    core, i: int, h: torch.Tensor, fp4: Optional[List[torch.Tensor]] = None
) -> torch.Tensor:
    """The router logits of a pure generation step: ``ROUTER_TC_GEN``'s rows take the TF32 tensor-core router when
    the device co-schedules 16-CTA clusters (decided at prebuild) and the operands are the built-in gate's (2-D bf16,
    unit column stride, 16-byte aligned rows); every other step the fp32 router (``_router_logits``)."""
    rows = h.shape[0] if h.dim() == 2 else 0
    if (
        ROUTER_TC_GEN.covers(rows)
        and core.router_tc_cluster16
        and h.dtype == torch.bfloat16
        and h.stride(1) == 1
        and (h.stride(0) * 2) % 16 == 0
    ):
        taken(ROUTER_TC_GEN.path, rows)
        if rows > 16:
            m3_report.work("ROUTER48", f"{rows} rows past the TP4 router's 16")
        _record("m3_router_tc", rows)
        scale = core.moe[i].fc31_input_scale if fp4 is not None else None
        outputs = m3_router_tc(h, core.w[f"l{i}_router"], scale, get_env_enable_pdl())
        if fp4 is not None:
            fp4.extend(outputs[1:])
        return outputs[0]
    return _router_logits(core, i, h)


class _S1bCheck:
    """S1B_FOLD_CHECK counters: per process, device int64 [calls, mismatching calls, mismatching elements], mirrored to
    pinned host memory by a copy inside the step (captured with it) and logged by a daemon thread when they change."""

    _dev = None
    _host = None

    @classmethod
    def record(cls, got: torch.Tensor, ref: torch.Tensor, ident=None):
        if cls._dev is None:
            cls._dev = torch.zeros(3, dtype=torch.int64, device=got.device)
            cls._host = maybe_pin_memory(torch.zeros(3, dtype=torch.int64))
            threading.Thread(target=cls._log, daemon=True).start()
        bad = (got.contiguous().view(torch.int16) != ref.contiguous().view(torch.int16)).sum()
        cls._dev.add_(torch.stack([torch.ones_like(bad), (bad > 0).to(torch.int64), bad]))
        cls._host.copy_(cls._dev, non_blocking=True)
        if ident is not None and not torch.cuda.is_current_stream_capturing():
            n = int(bad)
            if n:
                logger.info(f"[M3.1] S1B_FOLD_CHECK_BAD {ident} elements={n}")

    @classmethod
    def _log(cls):
        import time

        last = None
        while True:
            time.sleep(20.0)
            now = tuple(int(v) for v in cls._host.tolist())
            if now != last:
                last = now
                logger.info(
                    f"[M3.1] S1B_FOLD_CHECK calls={now[0]} mismatching_calls={now[1]} mismatching_elements={now[2]}"
                )


def _s1b_ident(core, i: int, h: torch.Tensor, do_finalize: bool) -> str:
    """S1B_FOLD_CHECK per-call identity: layer, rows, the tactic the fold's MoE resolved, the tuning mode."""
    rows = h.shape[0]
    tactic = core.__dict__.get("_s1b_moe_tactics_seen", {}).get(
        (h.device, rows, do_finalize), "pinned"
    )
    return (
        f"layer={i} rows={rows} finalize={do_finalize} fold_tactic={tactic} "
        f"tuning={AutoTuner.get().is_tuning_mode} rank={core.mapping.tp_rank}"
    )


def _decode_route(core, i: int, h: torch.Tensor, fp4: Optional[List[torch.Tensor]] = None):
    """The routing of a pure generation step's MoE: ``("topk", (ids, weights))`` from ``S1B_FOLD`` at its pinned row
    counts above the TC router's rows, else ``("logits", router_logits)`` exactly as ``_decode_router_logits``. Keyed
    sources keep their own path: SP logits route from those logits; the boundary's fp32 rows feed the fold's GEMM in
    place of ``float(h)`` (the same operand ``m3_fp32_linear`` casts)."""
    rows = h.shape[0] if h.dim() == 2 else 0
    if (
        not (
            core.s1b_pins
            and rows in core.s1b_pins
            and not ROUTER_TC_GEN.covers(rows)
            and h.dim() == 2
        )
        or AutoTuner.get().is_tuning_mode
    ):
        return "logits", _decode_router_logits(core, i, h, fp4)
    sp = _SP_LOGITS.take(h)
    if sp is not None:
        _record("m31_tp2agg_sp_logits", rows)
        return "logits", sp[0]
    f32 = _CTXNORMQ_F32.take(h)
    x = f32[0] if f32 is not None else h.to(torch.float32)
    w = core.w[f"l{i}_router"]
    topk = m31_s1b_fold.route(
        x,
        w,
        core.w[f"l{i}_router_bias"],
        float(core.routed_scaling_factor),
        get_env_enable_pdl() and S1B_FOLD_PDL,
    )
    if topk is None:
        _record("m31_router_f32" if f32 is not None else "m3_fp32_linear", rows)
        return "logits", torch.nn.functional.linear(x, w)
    _record("m31_s1b_fold", rows)
    taken("m31_s1b_fold", rows, f"S {core.s1b_pins[rows]}")
    return "topk", topk


def _triton_finalize(rows: int, routed, shared: Optional[torch.Tensor]) -> torch.Tensor:
    """The finalized routed output (+ ``shared``) from the runner's unfinalized parts via kernels/m3_moe_finalize
    (``TP2W_MOE_FINALIZE``). Raises if the parts are not the trtllm-gen layout (the runner did not finalize them:
    no fallback)."""
    out = None
    if isinstance(routed, (list, tuple)) and len(routed) == 3:
        fc2, scales, perm = routed
        scales, perm = scales.reshape(rows, -1), perm.view(rows, -1).to(torch.int32)
        native = _native_finalize(rows, fc2, scales, perm, shared)
        if native is not None:
            return native
        out = m3_moe_finalize(fc2, scales, perm, shared)
    if out is None:
        raise RuntimeError(
            f"the routed experts' unfinalized output at {rows} rows is not the trtllm-gen layout"
        )
    _record("m3_moe_finalize", rows, shared is not None)
    taken(
        "m3_moe_finalize", rows, "Triton finalize" + (" + shared add" if shared is not None else "")
    )
    return out


def _native_finalize(rows: int, fc2, scales, perm, shared) -> Optional[torch.Tensor]:
    """NATIVE_FINALIZE: the finalize by the native op, or None (the Triton kernel). The 49..448-row and 2K+-row
    (without shared) bands only with NATIVE_ALL_BANDS. cfg by rows: 3 = (128, 1) up to 224 rows, 6 = (96, 2) above,
    4 = (64, 2) at 2K+ rows with shared."""
    site = "shared add" if shared is not None else "routed"
    if not NATIVE_FINALIZE:
        return _native_dispatch(
            "NATIVE_FINALIZE", site, "killed (TRTLLM_M31_DISABLE_NATIVE_FINALIZE=1)"
        )
    if is_torch_compiling():
        return _native_dispatch("NATIVE_FINALIZE", site, "under torch.compile")
    if not NATIVE_ALL_BANDS and (rows < 2048 or shared is None):
        return _native_dispatch(
            "NATIVE_FINALIZE", site, "outside the served bands (NATIVE_ALL_BANDS off)"
        )
    if (
        fc2.dim() != 2
        or fc2.dtype != torch.bfloat16
        or fc2.stride(1) != 1
        or fc2.shape[1] % 8 != 0
        or fc2.stride(0) % 8 != 0
        or fc2.data_ptr() % 16 != 0
        or not scales.is_contiguous()
        or not perm.is_contiguous()
        or (shared is not None and (not shared.is_contiguous() or shared.data_ptr() % 16 != 0))
    ):
        return _native_dispatch("NATIVE_FINALIZE", site, "operand layout or alignment")
    op = _native_op("minimax_m31_moe_finalize")
    if op is None:
        return _native_dispatch("NATIVE_FINALIZE", site, "the op is not registered")
    cfg = 3 if rows <= 224 else (4 if rows >= 2048 and shared is not None else 6)
    _native_dispatch("NATIVE_FINALIZE", site)
    _record("m31_native_finalize", rows, shared is not None)
    taken(
        "m31_native_finalize", rows, f"cfg {cfg}" + (" + shared add" if shared is not None else "")
    )
    return op(fc2, scales, perm, shared, cfg)


def _triton_finalize_rows(core, h: torch.Tensor) -> bool:
    """Whether a finalized MoE forward of ``h`` takes ``TP2W_MOE_FINALIZE``: at least its rows, in one runner call."""
    rows = h.shape[0] if h.dim() == 2 else 0
    return (
        TP2W_MOE_FINALIZE
        and rows >= MOE_TRITON_FINALIZE_MIN_TOKENS
        and rows <= core.moe_max_num_tokens
    )


def _moe_tactic_as_logits(core, inputs, do_finalize: bool):
    """(runner, tactic) the stock runner resolves for these rows with ROUTING LOGITS: the same AutoTuner key as
    ``fp4_block_scale_moe_runner``'s (an fp32 [rows, experts] logits operand in slot 0, no top-k tensors), so a
    top-k-ids call (``S1B_FOLD``) runs the GEMM tactic the logits path runs (the pinned / tuned one) instead of the
    tuner's miss default: the same kernels, hence the same output bits. Resolved on EVERY call, as the stock op does
    (a tactic resolved before the tuning cache is loaded would otherwise stay the miss default); only
    the runner object is cached per (rows, finalize). Logged when a row count's tactic first resolves or changes."""
    rows = inputs[2].shape[0]
    key = (inputs[2].device, rows, do_finalize)
    runners = core.__dict__.setdefault("_s1b_moe_runners", {})
    if key not in runners:
        runners[key] = FP4BlockScaleMoERunner(
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
            ACT_SWIGLU,
            tune_max_num_tokens=core.max_num_tokens,
            use_dp=False,
        )
    runner = runners[key]
    logits_shape = torch.empty(rows, core.num_experts, dtype=torch.float32, device=inputs[2].device)
    runner, tactic = AutoTuner.get().choose_one(
        "trtllm::fp4_block_scale_moe_runner",
        [runner],
        runner.tuning_config,
        [logits_shape, *inputs[1:], None, None],
    )
    tactic = [-1, -1] if tactic == -1 else tactic
    seen = core.__dict__.setdefault("_s1b_moe_tactics_seen", {})
    if seen.get(key) != list(tactic):
        seen[key] = list(tactic)
        logger.info(
            f"[M3.1] S1B_FOLD MoE tactic at {rows} rows (finalize={do_finalize}): {list(tactic)}"
        )
    return runner, tactic


def _routed_experts(core, i: int, x, x_sf, router_logits, do_finalize: bool, topk=None):
    """The trtllm-gen NVFP4 MoE runner over this rank's tensor-parallel slice of every expert, routing in-kernel.

    ``x_sf`` None means ``x`` is the bf16 activation, quantized here with the experts' global input scale. ``topk``
    (int32 ids, bf16 weights) routes from those (``S1B_FOLD``; ``router_logits`` None) instead of the logits. Returns
    the finalized rows, or the unfinalized ``[gemm2 out, expert scales, expanded->permuted index]``.
    """
    moe = core.moe[i]
    if x_sf is None:
        x, x_sf = fp4_quantize(x, moe.fc31_input_scale, 16, False, False)
    rows = x.shape[0]
    x_sf = x_sf.view(rows, -1).flatten()
    _record("fp4_block_scale_moe_runner", rows, do_finalize)
    inputs = (
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
    )
    outs = None
    if _mx_mode(core.mxfp8).decode_graph:
        outs = _run_pinned_moe(core, inputs, do_finalize, ROUTING_MINIMAX2, ACT_SWIGLU, topk=topk)
    if outs is None and topk is not None:
        runner, tactic = _moe_tactic_as_logits(core, inputs, do_finalize)
        outs = runner([*inputs, topk[1], topk[0]], tactic=tactic)
    if outs is None:
        outs = fp4_block_scale_moe_runner(
            *inputs,
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
            topk_weights=None if topk is None else topk[1],
            topk_ids=None if topk is None else topk[0],
            output=None,
            tune_max_num_tokens=core.max_num_tokens,
            use_dp=False,
        )
    if do_finalize:
        return outs[0]
    if topk is not None:
        # The thop's unfinalized return hands back its own expert_weights buffer, which a top-k-ids call never writes
        # (routing and the in-kernel finalize use topk_weights): the finalize's scales are the weights passed in.
        outs = [outs[0], topk[1], outs[2]]
    return outs


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
    """Whether ``PRODUCER`` covers this target's sparse layers: an NVFP4 KV cache at the TP2 heads (``PRODUCER_HEADS``:
    32 query, two KV and two index heads per rank), with the range on. A covered layer that cannot take ``m3_producer``
    is a miss."""
    return (
        core.main_kv_is_nvfp4
        and (core.heads, core.kv_heads, core.index_heads) == PRODUCER_HEADS
        and PRODUCER.covers("nvfp4_tp2")
    )


def _producer_layout(kv_data, kv_scale) -> bool:
    """Whether a sparse layer's NVFP4 pools are the layout ``m3_producer`` is built for (TP2: two KV heads):
    [pages, 2, 2, 128, 64] with its scale pool [pages, 2, 2, 128, 8]."""
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


def _producer_miss(covered: bool, rows: Optional[int], reason: str) -> None:
    if covered:
        missed(PRODUCER.path, rows, reason)
    return None


def _sparse_qkv_producer(
    core,
    i: int,
    packed,
    position_ids,
    kv_cache_manager,
    msa_idx_k_cache,
    msa_out_cache_loc,
    deferred: Optional[_DeferredGemv] = None,
):
    """The horizontal producer of a sparse layer: Q / index-Q (FP8) out, K / V / index-K written into the paged
    caches (NVFP4 or FP8 main cache), for every step type (token-major). An NVFP4 cache at the TP2 heads takes
    ``m3_producer`` (``PRODUCER``); a covered layer that cannot is a miss. Returns (q, index_q), or None for caches
    neither kernel takes.

    ``kv_cache_manager``, ``msa_idx_k_cache`` (the per-layer index-K cache getter) and ``msa_out_cache_loc`` are the
    calling target's ``step_args`` entries. ``deferred``: the QKV GEMV ``_linear`` recorded instead of launching
    (``QKV_PRODUCER_FOLD``; ``packed`` is then its unwritten placeholder). The ``m3_producer`` branch runs it with the
    producer in its epilogue (``_qkv_producer_fold``); any other branch launches it first (``materialize``). A None
    return leaves it unlaunched."""
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
    # AGG_NVFP4_INDEXK: the producer writes E4M3 index-K into the cache manager's FP8 sink (the main pool's pages, one
    # for every layer); the layer's NVFP4 index-K pages are packed from it right after (_pack_nvfp4_index_k).
    nvfp4_index_cache = None
    if _nvfp4_index_k.is_nvfp4_index_cache(index_k_cache):
        sink = getattr(manager, "get_nvfp4_index_k_sink", lambda: None)()
        if sink is None:
            return _producer_miss(
                covered, rows, "an NVFP4 index-K cache without the manager's FP8 sink"
            )
        nvfp4_index_cache, index_k_cache = index_k_cache, sink
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
        and PRODUCER.covers("nvfp4_tp2")
        and (core.heads, core.kv_heads, core.index_heads) == PRODUCER_HEADS
        and _producer_layout(buffers, scale_buffers)
    ):
        if deferred is not None:
            folded = _qkv_producer_fold(
                core,
                i,
                deferred,
                buffers,
                scale_buffers,
                index_k_cache,
                out_cache_loc[:rows],
                kv_quant_scale,
                cos_sin,
                pos,
            )
            if folded is not None:
                _pack_nvfp4_index_k(nvfp4_index_cache, index_k_cache, out_cache_loc, rows)
                return folded
            packed = deferred.materialize()
        early_trigger = PRODUCER_EARLY_TRIGGER and rows > PRODUCER_LATE_TRIGGER_MAX_ROWS
        _record("m3_producer", rows, early_trigger)
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
            early_trigger,
            PRODUCER_PREWAIT_STEP_INPUTS,
        )
        # Every token count takes the producer: rows past a decode step log once as one line.
        decode_rows = is_decode_step(rows)
        if decode_rows:
            # The target tag line: once per rank, the first decode producer call.
            _log_once(
                logger.info,
                "m3_sm107_tp2_target",
                "engaged",
                f"[M3 sm107 tp2] fast path engaged: m3_producer (decode, rank {mpi_rank()}, {rows} rows, nvfp4_tp2: "
                "32 query / 2 KV / 2 index heads per rank)",
            )
        taken(
            PRODUCER.path,
            rows if decode_rows else None,
            f"nvfp4_tp2, {'entry' if early_trigger else 'exit'} trigger, "
            f"prewait_step_inputs={PRODUCER_PREWAIT_STEP_INPUTS}"
            if decode_rows
            else "nvfp4_tp2, past the decode rows",
        )
    else:
        if layer_uses_nvfp4 and covered:
            missed(
                PRODUCER.path,
                rows,
                f"the NVFP4 pool {tuple(buffers.shape)} / scale pool {tuple(scale_buffers.shape)} is not the "
                "producer's layout",
            )
        if deferred is not None:
            packed = deferred.materialize()
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
    _pack_nvfp4_index_k(nvfp4_index_cache, index_k_cache, out_cache_loc, rows)
    return q.flatten(1), index_q.flatten(1)


_NVFP4_INDEXK_DECODE_LOGGED: set = set()


def _pack_nvfp4_index_k(nvfp4_index_cache, sink, out_cache_loc, rows: int) -> None:
    """AGG_NVFP4_INDEXK: pack the step's index-K rows from the producer's FP8 sink into the layer's NVFP4 pages (the
    first ``rows`` slots; negative slots skipped). No-op on an FP8 index cache."""
    if nvfp4_index_cache is None:
        return
    _nvfp4_index_k.write_from_sink(sink, out_cache_loc, rows, nvfp4_index_cache)
    if not torch.cuda.is_current_stream_capturing() and not _NVFP4_INDEXK_LOGGED:
        _NVFP4_INDEXK_LOGGED.append(True)
        logger.info(
            f"[M3.1 tp2] AGG_NVFP4_INDEXK writer taken: producer sink -> NVFP4 index-K pages ({rows} rows)"
        )


_NVFP4_INDEXK_LOGGED: list = []


class _DeferredA1:
    """``TP2AGG_QKV_A1``: a sparse layer's packed QKV + index MXFP8 GEMM on its CTXNORMQ activation, recorded instead
    of launched. The producer call site runs it fused (``fold``: the A1 kernel, GEMM + producer) or calls
    ``materialize`` (the stock GEMM ``_linear`` would have run on that activation). ``placeholder`` has the packed
    shape and no storage; it is never read. ``ovl`` (TP2AGG_SP SP_OVL): the own rows run first, under the push of
    this rank's rows to the peer; the peer's rows after its push landed (A1 is row-split bitwise)."""

    def __init__(self, core, act, act_scale, weight, weight_scale, mode, ovl=None):
        rows, n = act.shape[0], weight.shape[0]
        self.placeholder = torch.empty(1, 1, dtype=torch.bfloat16, device=act.device).expand(
            rows, n
        )
        self.core, self.mode = core, mode
        self.act, self.act_scale = act, act_scale
        self.weight, self.weight_scale = weight, weight_scale
        self.ovl = ovl
        self.done = False

    def _wait_ovl(self) -> None:
        if self.ovl is not None:
            torch.cuda.current_stream().wait_event(self.ovl.event)

    def materialize(self) -> torch.Tensor:
        self.done = True
        self._wait_ovl()
        return _ctxnormq_mxfp8_gemm(
            self.core, (self.act, self.act_scale), self.weight, self.weight_scale, self.mode
        )

    def fold(
        self, i, buffers, scale_buffers, index_k_cache, out_cache_loc, kv_quant_scale, cos_sin, pos
    ):
        """The A1 kernel for layer ``i``: (q, index_q) as the producer returns them, the caches written; None (nothing
        launched) when the caches are outside the kernel's alignment contract."""
        core = self.core
        rows = int(self.act.shape[0])
        runner = _a1_runner(core.heads, core.kv_heads, core.index_heads)
        device = self.act.device
        q = torch.empty(rows, core.heads, core.head_dim, dtype=torch.float8_e4m3fn, device=device)
        index_q = torch.empty(
            rows, core.index_heads, core.head_dim, dtype=torch.float8_e4m3fn, device=device
        )
        norms = _A1_NORMS.get(i)
        if norms is None:
            w = core.w
            norms = _A1_NORMS[i] = torch.cat(
                [w[f"l{i}_q_norm"], w[f"l{i}_k_norm"], w[f"l{i}_idx_q_norm"], w[f"l{i}_idx_k_norm"]]
            ).contiguous()
        act = self.act.view(torch.float8_e4m3fn).view(rows, -1)
        ovl = self.ovl
        parts = (
            [((0, rows), slice(None))]
            if ovl is None
            else [(ovl.own_rows, ovl.own_sf), (ovl.peer_rows, ovl.peer_sf)]
        )
        try:
            part_args = []
            for (r0, n), sf_range in parts:
                rr = slice(r0, r0 + n)
                part_args.append(
                    runner.make_args(
                        act[rr],
                        self.act_scale[sf_range],
                        self.weight.view(torch.float8_e4m3fn),
                        self.weight_scale,
                        core.mx_alpha,
                        q[rr],
                        index_q[rr],
                        buffers.view(torch.uint8),
                        scale_buffers.view(torch.uint8),
                        index_k_cache,
                        out_cache_loc[rr],
                        kv_quant_scale,
                        cos_sin,
                        pos[rr],
                        norms,
                        core.rms_eps,
                    )
                )
        except AssertionError as e:
            _log_once(
                logger.warning,
                "tp2agg_qkv_a1",
                "declined",
                f"[M3.1 agg] TP2AGG_QKV_A1 declined: {e}",
            )
            return None
        _record("m31_qkv_a1", rows, ovl is not None)
        runner.run_args(part_args[0])
        if ovl is not None:
            self._wait_ovl()
            runner.run_args(part_args[1])
            _sp_once(
                ("ovl", rows),
                f"[M3.1 agg] TP2AGG_SP_OVL taken: A1 rows {rows} as own {ovl.own_rows[1]} + peer "
                f"{ovl.peer_rows[1]} (push by copy engine under the own rows)",
            )
        self.done = True
        _a1_count(rows)
        return q.flatten(1), index_q.flatten(1)


_A1_RUNNERS: dict = {}
_A1_NORMS: dict = {}
_A1_COUNTS: dict = {}


def _a1_runner(num_heads_q: int, num_heads_kv: int, num_heads_index: int):
    """The compiled A1 runner for one per-rank head geometry (compiled on the first take, eager: the warmup's first
    long step)."""
    key = (num_heads_q, num_heads_kv, num_heads_index)
    runner = _A1_RUNNERS.get(key)
    if runner is None:
        from .kernels.qkv_fusion.kernel import M3QkvFusedRunner

        runner = _A1_RUNNERS[key] = M3QkvFusedRunner(
            num_heads_q=num_heads_q, num_heads_kv=num_heads_kv, num_heads_index=num_heads_index
        )
        logger.info(f"[M3.1 agg] TP2AGG_QKV_A1 runner compiled for heads {key}")
    return runner


def _a1_count(rows: int) -> None:
    """Per-call A1 engagement by step-size band, logged at counts 1, 2, 4, ... per band."""
    band = "<=2K" if rows <= 2048 else "<=4K" if rows <= 4096 else "<=8K" if rows <= 8192 else ">8K"
    _A1_COUNTS[band] = n = _A1_COUNTS.get(band, 0) + 1
    if n & (n - 1) == 0:
        logger.info(f"[M3.1 agg] TP2AGG_QKV_A1 taken band={band} n={n} (rows {rows})")
    m3_report.work("TP2AGG_QKV_A1", f"{rows} rows")


def _a1_defer(core, i: int, hidden: torch.Tensor) -> Optional[_DeferredA1]:
    """A ``_DeferredA1`` for layer ``i``'s packed QKV projection when A1 takes the step (switched on, eager,
    not capturing, not tuning, more than A1_MIN_ROWS rows, the CTXNORMQ activation published for ``hidden``), else
    None (the caller runs ``_linear``; the CTXNORMQ slot is then still published unless this took it)."""
    if not QKV_A1 or not CTXNORMQ or hidden.dim() != 2 or hidden.shape[0] <= A1_MIN_ROWS:
        return None
    mode = _mx_mode(core.mxfp8)
    if (
        mode.decode_graph
        or mode.tuned_quantize_gemm
        or is_torch_compiling()
        or torch.cuda.is_current_stream_capturing()
        or (core.heads, core.kv_heads, core.index_heads) != PRODUCER_HEADS
    ):
        return None
    from tensorrt_llm._torch.autotuner import AutoTuner

    if AutoTuner.get().is_tuning_mode:
        # The stock QKV GEMM profiles its shape in the tuning pass (the steps A1 does not take run it).
        return None
    w, s = core.w[f"l{i}_qkv_w"], core.w[f"l{i}_qkv_s"]
    prequant = _CTXNORMQ_MX.take(hidden.contiguous())
    if prequant is None:
        return None
    return _DeferredA1(core, prequant[0], prequant[1], w, s, mode, _sp_ovl_take(hidden))


def _qkv_producer_fold(
    core,
    i: int,
    deferred: _DeferredGemv,
    buffers,
    scale_buffers,
    index_k_cache,
    out_cache_loc,
    kv_quant_scale,
    cos_sin,
    pos,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """``QKV_PRODUCER_FOLD``: the deferred QKV GEMV with layer ``i``'s producer in its epilogue
    (``mx_gemv.m3_mx_gemv_producer``), the same outputs and cache bytes as the GEMV + ``m3_producer``; None (nothing
    launched) when the kernel declines. A ``_DeferredA1`` record runs the A1 kernel instead."""
    if isinstance(deferred, _DeferredA1):
        return deferred.fold(
            i, buffers, scale_buffers, index_k_cache, out_cache_loc, kv_quant_scale, cos_sin, pos
        )
    rows = int(deferred.act.shape[0])
    device = deferred.act.device
    q = torch.empty(rows, core.heads, core.head_dim, dtype=torch.float8_e4m3fn, device=device)
    index_q = torch.empty(
        rows, core.index_heads, core.head_dim, dtype=torch.float8_e4m3fn, device=device
    )
    w = core.w
    _record("m3_qkv_producer_fold", rows)
    ok = mx_gemv.m3_mx_gemv_producer(
        deferred.act,
        deferred.act_scale,
        deferred.weight,
        deferred.weight_scale,
        q,
        index_q,
        buffers.view(torch.uint8),
        scale_buffers.view(torch.uint8),
        index_k_cache,
        out_cache_loc,
        kv_quant_scale,
        core.rms_eps,
        w[f"l{i}_q_norm"],
        w[f"l{i}_k_norm"],
        w[f"l{i}_idx_q_norm"],
        w[f"l{i}_idx_k_norm"],
        cos_sin,
        pos,
        cfg=deferred.cfg,
        use_pdl=deferred.use_pdl,
    )
    if not ok:
        missed(
            QKV_PRODUCER_FOLD_PATH,
            rows,
            "the producer GEMV declined the step" + _uncompiled_note(),
        )
        return None
    deferred.done = True
    n, k = deferred.weight.shape
    taken(MX_GEMV_PATH, rows, f"[{n}, {k}] mxfp8 in, split {deferred.cfg[0]}, producer epilogue")
    taken(QKV_PRODUCER_FOLD_PATH, rows, f"[{n}, {k}] mxfp8 in + producer")
    # The producer ran (in the GEMV's epilogue): its engagement and the target tag line, as the m3_producer branch.
    _log_once(
        logger.info,
        "m3_sm107_tp2_target",
        "engaged",
        f"[M3 sm107 tp2] fast path engaged: m3_producer (decode, rank {mpi_rank()}, {rows} rows, nvfp4_tp2: "
        "32 query / 2 KV / 2 index heads per rank; in the QKV GEMV epilogue)",
    )
    taken(PRODUCER.path, rows, "nvfp4_tp2, in the QKV GEMV epilogue (QKV_PRODUCER_FOLD)")
    return q.flatten(1), index_q.flatten(1)


def _qk_norm_rope(
    core, i: int, qkv: torch.Tensor, position_ids, out_fp8: bool, kv_dest: Optional[tuple] = None
) -> torch.Tensor:
    """The per-head Gemma RMSNorm of Q and K and the partial NeoX RoPE of a layer's packed [Q | K | V], in one kernel:
    in place, or an FP8 copy (Q / K normed and roped, V cast) when ``out_fp8``. A dense layer's FP8 copy carries the
    BF16 output's bits cast to E4M3 (``m3_dense_qk_norm_rope_fp8``, ``DENSE_QKV_FP8_VIA_BF16``); with ``kv_dest`` =
    (the layer's HND pool, the step's cache slots, live tokens, a [tokens, q heads, 128] FP8 buffer) that kernel also
    stores the live tokens' K / V rows into the pool and every row's Q into the buffer."""
    qkv = qkv.contiguous()
    pos = position_ids.reshape(-1).contiguous().to(torch.int32)
    if (
        out_fp8
        and DENSE_QKV_FP8_VIA_BF16
        and i not in core.sparse_layers
        and qkv.dtype == torch.bfloat16
        and core.head_dim == 128
    ):
        _record("m3_dense_qk_norm_rope_fp8", qkv.shape[0], kv_dest is not None)
        return m3_dense_qk_norm_rope_fp8(
            qkv,
            core.heads,
            core.kv_heads,
            core.kv_heads,
            core.rotary_dim,
            core.rms_eps,
            core.w[f"l{i}_q_norm"],
            core.w[f"l{i}_k_norm"],
            core.rope_theta,
            True,
            pos,
            True,
            *(kv_dest or ()),
        )
    assert kv_dest is None, "only the dense FP8 kernel stores K / V"
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


# AR-DET: AllReduce AUTO's per-process timing autotuner made VR runs differ across server starts
# (with it off, 0 / 888 same-node and 0 / 296 across node pairs differ, bitwise to the
# modal autotuned output). The target turns it off for the whole process at init (AUTO then takes the C++ size / world
# heuristic, a pure function of the shape), unless TRTLLM_M31_ENABLE_AR_AUTOTUNE=1. An explicit
# TLLM_DISABLE_ALLREDUCE_AUTOTUNE is left as given. Read once, at import.
AR_AUTOTUNE_ENABLED = m31_enabled("AR_AUTOTUNE")  # the m31_switches value contract (1/true/on/yes)


def _pin_allreduce_heuristic(rank: int) -> bool:
    """Default the AllReduce autotune off for this process (see AR-DET); returns whether it is off. Logs one line."""
    if not AR_AUTOTUNE_ENABLED:
        os.environ.setdefault("TLLM_DISABLE_ALLREDUCE_AUTOTUNE", "1")
    off = os.environ.get("TLLM_DISABLE_ALLREDUCE_AUTOTUNE", "0") == "1"
    if off:
        m3_report.work("AR_DET", "AllReduce autotune off for this process")
    logger.info(
        f"M3.1 AR-DET: AllReduce autotune {'off (C++ size / world heuristic)' if off else 'on'} on rank {rank}"
    )
    return off


# PREFILL-MNNVL: the built-in all-reduce of a prefill or mixed step's layer boundaries (the rows the multicast kernel
# does not take) runs MNNVL instead of AllReduce(AUTO)'s pick. On VR TP4 every strategy is cross-process deterministic
# there and MNNVL is the fastest, 1.7..29 us per call under AUTO at 128..8192 rows (plain and with the fused residual +
# RMSNorm). Up to PREFILL_AR_MNNVL_MAX_ROWS rows (MNNVL's workspace grows with the rows; larger steps keep AUTO);
# generation steps keep AUTO. TRTLLM_M31_DISABLE_PREFILL_AR_MNNVL=1 restores AUTO. Read once, at import.
PREFILL_AR_MNNVL = not m31_disabled("PREFILL_AR_MNNVL")
PREFILL_AR_MNNVL_MAX_ROWS = 8192  # capacity: measured: the largest step the strategy table covers


def _prefill_allreduce(core, rows: int):
    """The built-in all-reduce of a prefill / mixed-step boundary of ``rows`` rows: MNNVL (PREFILL-MNNVL) up to its
    rows, else the target's AUTO one."""
    if core.allreduce_mnnvl is not None and rows <= PREFILL_AR_MNNVL_MAX_ROWS:
        taken("prefill_ar_mnnvl", None)
        return core.allreduce_mnnvl
    return core.allreduce


class MiniMaxM3Tp2ModelingV2(DecoderModel):
    """The MiniMax-M3 text model of this target: its weights, its runtime resources, and the per-step dispatch."""

    def __init__(self, model_config: ModelConfig[PretrainedConfig]) -> None:
        _validate_runtime(model_config)
        super().__init__(model_config)
        cfg = model_config.pretrained_config
        mapping = model_config.mapping
        _pin_allreduce_heuristic(mapping.tp_rank)
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
        # separate MXFP8 index projection (fuse_qkv_index_projection=False) run on two streams.
        self.fused_projection = _weights.fused_projection(model_config.sparse_attention_config)

        self.w = _weights.MODEL_WEIGHTS.declare(d)

        # The engine reads these two off the module tree: the MXFP8 GEMM state it tunes, and the opt-in to the
        # automatic FlashInfer MXFP8 path on decode graphs.
        self.mxfp8 = _Mxfp8GemmState()
        self._use_flashinfer_mxfp8_decode_graph_default = True

        # The built-in fused all-reduce + residual + RMSNorm of the layer boundaries the multicast kernel does not take.
        self.allreduce = AllReduce(
            mapping=mapping, strategy=model_config.allreduce_strategy, dtype=cfg.torch_dtype
        )
        # PREFILL-MNNVL: the prefill / mixed-step boundaries' all-reduce (None: switched off).
        self.allreduce_mnnvl = (
            AllReduce(mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=cfg.torch_dtype)
            if PREFILL_AR_MNNVL
            else None
        )
        if self.allreduce_mnnvl is not None and self.allreduce_mnnvl.mnnvl_allreduce is not None:
            # Size MNNVL's shared workspace for the largest step it takes now, so the growth lands before the KV cache
            # is sized (the memory-estimation step runs more rows than the pin covers and takes AUTO).
            get_or_scale_allreduce_mnnvl_workspace(
                mapping,
                cfg.torch_dtype,
                buffer_size_bytes=MNNVLAllReduce.get_required_workspace_size(
                    PREFILL_AR_MNNVL_MAX_ROWS, cfg.hidden_size, mapping.tp_size, cfg.torch_dtype
                ),
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
        _prefill_attention_opt_ins(self.attn)

        # Built once the weights are real (``post_load_weights``): the KV (Q, K, V) scales, the MoE scalars, the
        # decode engines and workspaces, the two targets.
        self.kv_scales: Dict[int, torch.Tensor] = {}
        self.inv_kv_scales: Dict[int, torch.Tensor] = {}
        self.moe: Dict[int, _MoeScalars] = {}
        self.mx_alpha: Optional[torch.Tensor] = None
        self.moe_engines: Dict[int, Dict[Tuple[int, bool, bool], Optional[object]]] = {}
        self.moe16_engines: Dict[int, Dict[int, Optional[object]]] = {}
        self.embed_owners: Dict[tuple, DecodeEmbedding] = {}
        # Whether the 9..16-row generation router (ROUTER_TC_GEN) runs; decided at prebuild.
        self.router_tc_cluster16 = False
        # S1B_FOLD's pinned split factor per generation row count (prebuild_kernels; empty = the stock router).
        self.s1b_pins: Dict[int, int] = {}
        # Each sparse layer's selected block ids per (KV head, query row, slot) at its last long-context decode, which
        # that decode prefetches into L2 before its grid wait (``SPARSE_DECODE_LC_PREFETCH_SELF_KV``, up to
        # ``N1_MAX_ROWS`` query rows). Allocated once here (stable addresses for CUDA-graph capture); none when
        # TP2WIDE_LC2 is killed.
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
            if SPARSE_DECODE_LC_PREFETCH_SELF_KV and _sparse_decode_kernel.LC_KEYS
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
        # MetaInitMode redirects even an explicit CUDA empty to meta. Reserve only
        # after materialization, still before executor KV capacity profiling.
        if self.main_kv_is_nvfp4:
            with torch.cuda.device(device):
                prepare_lc2_split_workspace(self.kv_heads)
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
        _sp_init(self)
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
            # The verify width the config itself declares, whatever it names its draft count.
            per_step = getattr(spec, "tokens_per_gen_step", None)
            if isinstance(per_step, int) and per_step > 1:
                lens.add(per_step)
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
        it, GEMV compiles are frozen and the forward only looks kernels up (a missing one is a miss). It also builds
        the T2SC scorer (``INDEX_SCORE_T2SC_PATH``), the 384-thread long-row select (``LONG_SELECT_DSL``) and the dense
        layers' full-context decode plans (``FULLCTX_DECODE``). Returns the seconds each part took and the scorer
        keys."""
        times: Dict[str, object] = {}
        # F3: the o_proj fork's tactics and the boundary op, before warmup / capture, on the real workspace.
        _f3_t0 = time.perf_counter()
        _f3_init(self)
        times["f3"] = round(time.perf_counter() - _f3_t0, 3)
        # The 9..16-row generation router: decided once, from the device, never inside a CUDA-graph capture.
        # The built-in gate's conditions, mirrored exactly: 128 experts, a hidden size in 1024s, 16-CTA clusters.
        self.router_tc_cluster16 = (
            self.num_experts == 128
            and self.hidden % 1024 == 0
            and bool(torch.ops.trtllm.minimax_m3_router_tc_cluster16(self.hidden))
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
        # S1B_FOLD: pin the router's split factor per generation row count above the TC router's (bitwise probe of this
        # device's cuBLASLt, outside capture); row counts without a pin keep the stock router + routing.
        start_s1b = time.perf_counter()
        self.s1b_pins = m31_s1b_fold.prebuild(
            [self.w[f"l{i}_router"] for i in sorted(self.moe_layers)],
            range(ROUTER_TC_GEN_MAX_ROWS + 1, S1B_MAX_ROWS + 1),
        )
        times["s1b_fold_pins_s"] = round(time.perf_counter() - start_s1b, 3)
        if not self.mxfp8.quant_method.use_cutlass:
            raise ValueError("the MiniMax-M3 decode dispatch needs the CUTLASS MXFP8 GEMM")
        if PREPROCESS_TOKMAP_FIX:
            # Every path that can run the stock generation preprocess (the drafter's full-context decode or its
            # fallback, any trtllm-gen fallback), so it is installed whatever else is enabled.
            _tokmap_guard.install(
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
                # An MXFP8 call this target gives a (split, ring) override (``mx_gemv_cfg``, as ``_linear`` passes
                # it) compiles that config.
                cfg = None if x_bf16 else mx_gemv_cfg(n, k, rows)
                if cfg is not None:
                    _precompile_cfg(weight, scale, rows, cfg, use_pdl, ar=ar)
                else:
                    mx_gemv.precompile(weight, scale, rows, x_bf16, use_pdl=use_pdl, ar=ar)
                # QKV_PRODUCER_FOLD: the packed QKV + index projection also takes these rows with the producer epilogue.
                fold_cfg = None if x_bf16 or ar is not None else _fold_cfg(n, k, rows)
                if fold_cfg is not None:
                    _precompile_cfg(weight, scale, rows, fold_cfg, use_pdl, prod=True)
        # SWQ (Triton) compiles on first use, which must not happen inside a graph capture: one call per down
        # projection shape it serves covers every row count.
        swq_shapes = {tuple(self.w[wk].shape) for wk, _, _ in self.projections()} & set(SWQ_ROWS)
        for n, k in sorted(swq_shapes):
            if not SWQ_ROWS[(n, k)]:
                continue
            gu = torch.zeros(
                max(SWQ_ROWS[(n, k)]), 2 * k, dtype=torch.bfloat16, device=self.w["embed"].device
            )
            m3_swq(gu, self.swiglu_limit, self.swiglu_alpha, self.swiglu_beta)
            logger.info(f"[M3 prebuild] SWQ: compiled for intermediate {k}")
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
        times["scorer_t2sc_keys"] = self._prebuild_index_scorer_t2sc(
            self.decode_query_lens(), self.w["embed"].device
        )
        times["scorer_intree_keys"] = self._prebuild_index_scorer_intree(
            self.decode_query_lens(), self.w["embed"].device
        )
        if self.sparse_layers and self.attn[min(self.sparse_layers)].indexer_kv_dtype == "nvfp4":
            # AGG_NVFP4_INDEXK: the Q8KV4 decode reader at every decode query length (DecodeTarget scores through it).
            reader = Nvfp4IndexDecodeScorer(self.index_heads)
            if not reader.available():
                raise RuntimeError(
                    "indexer_kv_dtype='nvfp4' needs the vendored fmha_sm100_q8kv8 package with run_scores_hpr (v3h); "
                    "TRTLLM_M31_DISABLE_AGG_NVFP4_INDEXK=1 falls back to FP8 index-K"
                )
            times["nvfp4_reader_qlens"] = reader.prebuild(
                self.w["embed"].device, self.decode_query_lens()
            )
            logger.info(
                f"[M3 prebuild] AGG_NVFP4_INDEXK decode reader compiled: qlen {times['nvfp4_reader_qlens']}"
            )
        torch.cuda.synchronize()
        times["scorer_s"] = time.perf_counter() - start
        start = time.perf_counter()
        times["select_keys"] = []
        if long_select_dsl_keys():
            # The 384-thread long-row select (LONG_SELECT_DSL): one kernel per launch key.
            times["select_keys"] = _select_long_dsl_kernel.prebuild(
                long_select_dsl_keys(), self.w["embed"].device
            )
            torch.cuda.synchronize()
        times["select_s"] = time.perf_counter() - start
        start = time.perf_counter()
        times["fullctx_plans"] = 0
        if FULLCTX_DECODE.keys:
            # The dense layers' full-context decode: one compiled PrimTS plan per covered (rows, requests).
            times["fullctx_plans"] = _fullctx_decode_kernel.prebuild(
                self.w["embed"].device,
                {key: FULLCTX_DECODE_RECIPES[key] for key in FULLCTX_DECODE.keys},
                FULLCTX_DECODE_MAX_KV_LEN,
                FULLCTX_DECODE_OWN_REDUCER and _fullctx_decode_kernel.has_own_reducer(),
                get_env_enable_pdl(),
            )
            torch.cuda.synchronize()
        times["fullctx_s"] = time.perf_counter() - start
        start = time.perf_counter()
        times["proxy_variants"] = self._prebuild_msa_proxy() if MSA_PROXY_PREBUILD else []
        if times["proxy_variants"]:
            m3_report.work("PROXYPB", f"{len(times['proxy_variants'])} proxy variants built")
        times["proxy_s"] = time.perf_counter() - start
        mx_gemv.freeze(True)
        logger.info(
            "[M3 prebuild] decode kernels built before warmup: GEMVs %.1f s, MoE engines %.1f s, index scorer "
            "%.1f s (%d variants), long-row select %.1f s (%d variants), full-context decode %.1f s (%d plans), "
            "MSA indexer proxy %.1f s (%d variants)"
            % (
                times["gemv_s"],
                times["moe_s"],
                times["scorer_s"],
                len(times["scorer_keys"]),
                times["select_s"],
                len(times["select_keys"]),
                times["fullctx_s"],
                times["fullctx_plans"],
                times["proxy_s"],
                len(times["proxy_variants"]),
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
                device,
                num_tokens=num_tokens,
                num_ctas=sms
                - (shared_ctas_for(num_tokens, shared) if shared else MOE_ENGINE_RESERVED_SMS),
                shared=shared,
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
        if MOE16_KERNEL_WIDTH != _MOE_INTER:
            # MOE16_TP2 opted in: the engine's kernel is the TP4-width build; never run it on TP2-width weights.
            raise RuntimeError(
                f"the 9..13-row engine is built for expert width {MOE16_KERNEL_WIDTH}, this target's is {_MOE_INTER} "
                "(no TP2-width build yet); the generic MoE runs"
            )
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
            dtype = (
                torch.float8_e4m3fn
                if backend.indexer_kv_dtype in ("fp8", "nvfp4")
                else torch.bfloat16
            )
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

    def _prebuild_msa_proxy(self) -> list:
        """Compile and load every fmha_sm100 variant the MSA indexer's proxy pass can pick (OnlyScore over the paged
        index-K: one KV head, page 128, no split-KV), so no context chunk compiles one after the server is ready (a
        65..128-token chunk's variant did, stalling the steps behind it by 0.8-1.6 ms). The variant follows only the
        longest query row: fmha_sm100 packs a row of up to 32 tokens by its pack factor, then takes the 128- or 256-row
        query tile and the single-warpgroup kernel from the packed length, so query lengths 1..129 reach every variant.
        Compile and load only, nothing runs: no numerics change. fmha_sm100 keys its cache by its installed sources, so
        a source overlay builds its own variants. Returns the variant names."""
        require_msa_module()
        from fmha_sm100 import api as fmha_api
        from fmha_sm100 import jit as fmha_jit

        only_score, page = 2, 128
        names = []
        for i in sorted(self.sparse_layers):
            backend = self.attn[i]
            dtype = (
                torch.float8_e4m3fn
                if backend.indexer_kv_dtype in ("fp8", "nvfp4")
                else torch.bfloat16
            )
            heads = int(backend.m3_config.num_index_heads)
            for qlen in range(1, 130):
                pack = fmha_api._compute_pack_factor(qlen, heads, 1)
                packed = qlen * pack
                key = (
                    fmha_jit._dlpack_dtype_code(dtype),
                    128 if packed <= 128 else 256,
                    packed <= 64,
                    only_score,
                    page,
                    False,
                    pack,
                )
                name, _ = fmha_jit._variant_key_from_runtime(*key)
                if name not in names:
                    fmha_jit.get_fmha_variant(*key)
                    names.append(name)
        return names

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

    def _prebuild_index_scorer_t2sc(self, query_lens, device) -> list:
        """T2SC (``INDEX_SCORE_T2SC_PATH``): compile this target's copy of the TP2 / TP8 scorer at every covered request
        count's schedule (``index_score_t2sc_schedule``) for every sparse layer geometry and decode query length, and
        launch each once on real-shaped dummy operands (loading its module). Returns the keys."""
        runner = _index_score_t2sc_kernel.IndexScoreRunner
        keys = []
        for i in sorted(self.sparse_layers):
            backend = self.attn[i]
            config = backend.m3_config
            dtype = (
                torch.float8_e4m3fn
                if backend.indexer_kv_dtype in ("fp8", "nvfp4")
                else torch.bfloat16
            )
            for qlen in sorted(set(query_lens)):
                if not runner.is_supported(
                    q_dtype=dtype,
                    num_heads=config.num_index_heads,
                    head_dim=config.sparse_index_dim,
                    page_size=runner.SUPPORTED_PAGE_SIZE,
                    max_decode_query_len=qlen,
                ):
                    continue
                for requests in sorted(INDEX_SCORE_T2SC_PATH.keys):
                    split_k, early = index_score_t2sc_schedule(requests)
                    key = (
                        dtype,
                        int(config.num_index_heads),
                        int(qlen),
                        int(config.sparse_index_dim),
                        int(split_k),
                        bool(early),
                    )
                    if all(key != seen for seen, _ in keys):
                        keys.append((key, requests))
        page = runner.SUPPORTED_PAGE_SIZE
        for key, rows in keys:
            dtype, heads, qlen, dim, split_k, early = key
            runner._compile(dtype, heads, qlen, dim, split_k, early)
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
        return [key for key, _ in keys]

    def _prebuild_index_scorer_intree(self, query_lens, device) -> list:
        """TP2 WIDE (``INDEX_SCORE_INTREE``): compile the in-tree CuTe DSL scorer at every sparse layer geometry and
        decode query length a step of ``INDEX_SCORE_INTREE_MIN_REQUESTS`` or more requests can take, and launch each
        once on real-shaped dummy operands (loading its module). Returns the keys."""
        if not INDEX_SCORE_INTREE:
            return []
        in_tree = _in_tree_scorer()
        keys = []
        for i in sorted(self.sparse_layers):
            backend = self.attn[i]
            config = backend.m3_config
            dtype = (
                torch.float8_e4m3fn
                if backend.indexer_kv_dtype in ("fp8", "nvfp4")
                else torch.bfloat16
            )
            for qlen in sorted(set(query_lens)):
                if max(
                    qlen * INDEX_SCORE_INTREE_MIN_REQUESTS, TP2_WIDE_MIN_ROWS
                ) > DECODE_STEP_MAX or not in_tree.is_supported(
                    q_dtype=dtype,
                    num_heads=config.num_index_heads,
                    head_dim=config.sparse_index_dim,
                    page_size=in_tree.SUPPORTED_PAGE_SIZE,
                    max_decode_query_len=qlen,
                ):
                    continue
                key = (
                    dtype,
                    int(config.num_index_heads),
                    int(qlen),
                    int(config.sparse_index_dim),
                    int(in_tree.SPLIT_K),
                )
                if key not in keys:
                    keys.append(key)
        page = in_tree.SUPPORTED_PAGE_SIZE
        for key in keys:
            dtype, heads, qlen, dim, split_k = key
            in_tree._compile(dtype, heads, qlen, dim, split_k)
            rows, blocks = 1, 2
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
            in_tree.kernel_cache[key](
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
        capture_target: Optional[Callable[..., Any]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """``capture_target``: the worker's capture-norm target (``CAPTURE_NORM_IN_BOUNDARY``) for a capture layer
        on the decode fast path; None: the capture runs its own norm."""
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


# PREFILL_BRK (ported from the M3 sm_107 tp2 target): a step with context rows runs under breakable
# (piecewise) prefill CUDA graphs, as the aggregated deployment captures prefill and mixed steps. The projections,
# norms, MoE and boundaries are captured per padded row bucket; the attention boundary (the horizontal producer over
# the live rows, then the MSA backend) runs eagerly at every replay (``_prefill_attention_eager``), reading the live
# row count from the step's metadata, as the built-in model's MiniMax-M3 attention does. Without a breakable graph the
# step runs as before. The separate (unfused) index projection still declines breakable graphs; pure generation steps
# never run under one. Kill switch TRTLLM_M31_DISABLE_PREFILL_BRK=1 (the PrefillTarget raises under a breakable
# graph, as before the port). Read at import.
PREFILL_BRK = not m31_disabled("PREFILL_BRK")
PREFILL_BRK_PATH = "m3_prefill_brk_attention"
# PREFILL_OUT_CLONE (determinism): a prefill / mixed step returns a copy of its final
# hidden, taken on the current stream right after the last boundary. Without it a later write from another stream lands
# in the returned hidden's memory before the logits processor gathers the verify rows (K7, 8 x ~50K-token decodes with a
# 3840-row mixed step: up to 8 of 272 sampled requests differ run to run; with the copy 0 of 272). The copy is bitwise
# the hidden. Opt-in: TRTLLM_M31_ENABLE_PREFILL_OUT_CLONE=1; TRTLLM_M31_DISABLE_PREFILL_OUT_CLONE=1 wins. Read at
# import.
PREFILL_OUT_CLONE = m31_enabled("PREFILL_OUT_CLONE") and not m31_disabled("PREFILL_OUT_CLONE")
PREFILL_OUT_CLONE_PATH = "m31_prefill_out_clone"

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
    if GQAMIX and sparse:
        _gqamix_register(core.attn[i])
    if AGG_TSEL and sparse:
        _tsel_register(core.attn[i])
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


class PrefillTarget:
    """A step with context rows: prefill chunks, and the mixed steps that carry generation rows beside them. Every
    layer runs in its generic form; the attention core is the MSA backend's for every row.

    Under a breakable (piecewise) prefill CUDA graph (``PREFILL_BRK``) the projections, norms, MoE and boundaries are
    captured per padded row bucket and the attention boundary runs eagerly at every replay
    (``_prefill_attention_eager``: the live-row horizontal producer and the MSA backend), as the built-in model's
    MiniMax-M3 attention does."""

    def __init__(self, core: MiniMaxM3Tp2ModelingV2) -> None:
        self.core = core
        self._bcg_key = f"m31_tp2_core_{id(core)}"
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
        """``capture_target`` is unused: the generic boundary takes no multicast all-reduce at this TP
        (``GENERIC_MC_AR_MAX_TOKENS`` 0), so a capture layer always runs its own capture norm."""
        core = self.core
        if is_in_breakable_cuda_graph():
            if not PREFILL_BRK:
                raise RuntimeError(
                    "this MiniMax-M3 target does not run under piecewise (breakable) prefill CUDA graphs "
                    "(TRTLLM_M31_DISABLE_PREFILL_BRK is set)"
                )
            taken(PREFILL_BRK_PATH, None, "attention boundary eager at every replay")
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
        if PREFILL_OUT_CLONE:
            taken(PREFILL_OUT_CLONE_PATH, None, "final hidden copied after the last boundary")
            hidden = hidden.clone()
        return hidden

    def _swiglu_mlp(
        self, h: torch.Tensor, prefix: str, low_m_gemm: bool, dense: bool = False
    ) -> torch.Tensor:
        """A gated MXFP8 MLP (the dense MLP with ``dense``, the shared expert): gate_up, SwiGLU-OAI, down, no
        reduction."""
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
        # CTXNORMQ: the boundary already wrote this activation's NVFP4 form.
        fp4 = _CTXNORMQ_FP4.take(x) if CTXNORMQ else None
        if num_chunks <= 1:
            if fp4 is not None:
                return _routed_experts(core, i, fp4[0], fp4[1], router_logits, True)
            return _routed_experts(core, i, x, None, router_logits, True)
        sizes = _split_chunk(rows, num_chunks)
        xs, logits = x.split(sizes), router_logits.split(sizes)
        sfs = [None] * num_chunks
        if fp4 is not None:
            xs = fp4[0].split(sizes)
            sfs = [t.flatten() for t in fp4[1].view(rows, -1).split(sizes)]
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
                    outs.append(_routed_experts(core, i, xs[c], sfs[c], logits[c], True))
            else:
                outs.append(_routed_experts(core, i, xs[c], sfs[c], logits[c], True))
        if aux is not None:
            with torch.cuda.stream(aux):
                ev_overlap.record()
            ev_overlap.wait()
        return torch.cat(outs)

    def _routed(self, i: int, h: torch.Tensor):
        """Router + routed experts, finalized: the fp32 router logits and the experts quantizing their bf16 input
        (chunked)."""
        return self._routed_chunks(i, h, _router_logits(self.core, i, h))

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
        if not return_parts and _triton_finalize_rows(core, h):
            # TP2AGG_SP: the keyed view holds no bf16 rows; the experts take the gathered NVFP4 products.
            keyed = _sp_keyed(h)
            fp4 = _CTXNORMQ_FP4.take(h) if keyed or (CTXNORMQ and TP2W_FP4SLOT_FINALIZE) else None
            if fp4 is not None and not keyed:
                _FP4SLOT_STATE["taken"] += 1
                if _FP4SLOT_STATE["taken"] in (1, 1000):
                    logger.info(
                        f"[M3.1 tp2] TP2W_FP4SLOT_FINALIZE taken n={_FP4SLOT_STATE['taken']} (rows {rows})"
                    )
                if _FP4SLOT_CHECK and not torch.cuda.is_current_stream_capturing():
                    ref, ref_sf = fp4_quantize(h, core.moe[i].fc31_input_scale, 16, False, False)
                    d = int((ref.view(torch.uint8) != fp4[0].view(torch.uint8)).sum()) + int(
                        (
                            ref_sf.view(torch.uint8).flatten() != fp4[1].view(torch.uint8).flatten()
                        ).sum()
                    )
                    _FP4SLOT_STATE["checked"] += 1
                    _FP4SLOT_STATE["diff"] += d
                    if d or _FP4SLOT_STATE["checked"] <= 2 or _FP4SLOT_STATE["checked"] % 500 == 0:
                        logger.info(
                            f"[M3.1 tp2] TP2W_FP4SLOT_CHECK rows {rows} diff {d} (checked "
                            f"{_FP4SLOT_STATE['checked']}, total diff {_FP4SLOT_STATE['diff']})"
                        )
            x_in, x_sf = (fp4[0], fp4[1]) if fp4 is not None else (h, None)
            routed_parts, shared_out = maybe_execute_in_parallel(
                lambda: _routed_experts(core, i, x_in, x_sf, _router_logits(core, i, h), False),
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

    def _generic_boundary(self, x, x2, residual, norm_w, qkv_a1_next: bool = False):
        """A generic layer boundary: the multicast all-reduce + residual + RMSNorm on the rows it takes (with the
        shared + routed add folded in), else the built-in fused all-reduce. ``qkv_a1_next``: the next layer is
        sparse (TP2AGG_SP's SP_OVL may overlap its push with A1)."""
        core = self.core
        if _can_mc(x, folded=x2 is not None):
            fused = _mc_allreduce_norm(core, x, x2, residual, norm_w)
            if fused is not None:
                return fused
        if x2 is not None:
            x = x.add_(x2)
        if norm_w is not core.w["final_norm"]:
            fused = _ctxnormq_allreduce_norm(
                core,
                x,
                residual,
                norm_w,
                None,
                _prefill_allreduce(core, x.shape[0]),
                qkv_a1_next=qkv_a1_next,
            )
            if fused is not None:
                return fused
        return _native_allreduce_norm(
            core, x, residual, norm_w, _prefill_allreduce(core, x.shape[0])
        )

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
                else _native_allreduce_norm(
                    core,
                    hidden,
                    residual,
                    core.w[f"l{i}_post_norm"],
                    _prefill_allreduce(core, hidden.shape[0]),
                )
            )
        else:
            fused = _ctxnormq_allreduce_norm(
                core,
                hidden,
                residual,
                core.w[f"l{i}_post_norm"],
                core.moe[i].fc31_input_scale if i in core.moe_layers else None,
                _prefill_allreduce(core, hidden.shape[0]),
                router_w=core.w[f"l{i}_router"] if i in core.moe_layers else None,
                pre_moe=True,
            )
            hidden, residual = (
                fused
                if fused is not None
                else _native_allreduce_norm(
                    core,
                    hidden,
                    residual,
                    core.w[f"l{i}_post_norm"],
                    _prefill_allreduce(core, hidden.shape[0]),
                )
            )
        if i in core.moe_layers:
            out, routed = self._moe(i, hidden, return_parts=_can_mc(hidden, folded=True))
        else:
            out = self._swiglu_mlp(hidden, f"l{i}_mlp", True, dense=True)
            routed = None
        normed, updated = self._generic_boundary(
            out, routed, residual, _next_norm(core, i), qkv_a1_next=(i + 1) in core.sparse_layers
        )
        if spec_metadata is not None and spec_metadata.is_layer_capture(i):
            # TP2AGG_SP: the capture reads the full residual (gathered from an owned-only one).
            updated = _sp_unshard(updated, normed)
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
            if is_in_breakable_cuda_graph():
                raise RuntimeError(
                    "this MiniMax-M3 target runs a breakable prefill CUDA graph only with the five-way projection "
                    "(fuse_qkv_index_projection)"
                )
            return self._unfused_sparse_attention(i, position_ids, hidden, step)
        in_bcg = is_in_breakable_cuda_graph()
        deferred = None if in_bcg else _a1_defer(core, i, hidden)
        if deferred is not None:
            packed = deferred.placeholder
        else:
            packed, _ = _linear(core, hidden, f"l{i}_qkv_w", f"l{i}_qkv_s", True)
        if in_bcg:
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
            deferred,
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
        and its fused norm + RoPE (FP8 q/k/v into an FP8 cache) on the current stream; the MXFP8 index projection
        (the checkpoint's block scales applied) and the FP8 indexer kernel on the attention aux stream; then the MSA
        backend (it writes the main KV cache) and the o_proj. The NVFP4 producer does not apply: it packs the
        five-way projection's output."""
        core = self.core
        if _producer_covered(core):
            # Not a miss: the producer packs the five-way projection's output, which this configuration lacks.
            uncovered(
                PRODUCER.path,
                "fuse_qkv_index_projection is disabled: QKV + separate MXFP8 index projection",
            )

        def _main():
            qkv, _ = _linear(core, hidden, f"l{i}_qkv_w", f"l{i}_qkv_s", True)
            fused = _qk_norm_rope(core, i, qkv, position_ids, core.main_kv_is_fp8)
            return fused.split([core.q_size, core.kv_size, core.kv_size], dim=-1)

        def _index():
            idx_qk, _ = _linear(core, hidden, f"l{i}_idx_w", f"l{i}_idx_s", True)
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
            _COMBINE_MXFP8_STATE.pop("pending", None)
            _COMBINE_MXFP8_STATE.pop("src", None)
            _CTXNORMQ_MX.clear()
            if (
                TP2WIDE_COMBINE_MXFP8
                and TP2WIDE_SPARSE_OUT
                and CTXNORMQ
                and core.main_kv_is_nvfp4
                and i in core.sparse_layers
                and int(step["num_ctx_tokens"]) >= _COMBINE_MXFP8_MIN_ROWS
                and output.shape[0] == int(step["num_tokens"])
                and not torch.cuda.is_current_stream_capturing()
                and not torch.compiler.is_compiling()
            ):
                _COMBINE_MXFP8_STATE["pending"] = {
                    "output": output,
                    "context_rows": int(step["num_ctx_tokens"]),
                }
            try:
                _msa_attention_live(core, i, q, k, v, idx_q, step["metadata"], output)
                _combine_mxfp8_finish(output)
            finally:
                _COMBINE_MXFP8_STATE.pop("pending", None)
        return output


# F3: the attention boundary's MXFP8 norm for the next pinned GENERIC MXFP8 linear (shared / dense gate_up).
_F3_MX = m3_prequant.PrequantSlot()
_F3_STATE: dict = {}


class F3Push:
    """The o_proj's F3 target (decided before the attention from the row count and import / prebuild state only, so
    every rank agrees): the launcher, the pinned tactic and the boundary's allreduce_in-shaped tensor (shape only)."""

    def __init__(self, launcher, tactic, rows: int, hidden: int, device):
        self.launcher, self.tactic, self.rows = launcher, tactic, rows
        self.out = torch.empty((rows, hidden), dtype=torch.bfloat16, device=device)
        self.ref = None  # (act, act_scale, w, s): F3_CHECK's stock re-run

    def accepts(self, n: int, k: int, rows: int) -> bool:
        return (n, k) == (F3_N, F3_K) and rows == self.rows

    def run(self, act, act_scale, w, s):
        self.launcher.run(self.tactic, act, act_scale, w, s, self.out)
        if F3_CHECK:
            self.ref = (act, act_scale, w, s)


def _f3_op():
    """``torch.ops.trtllm.minimax_m31_f3_ar_prepushed`` when the build carries it and F3 is on, else None. Decided
    once per process."""
    if "op" not in _F3_STATE:
        op = getattr(torch.ops.trtllm, "minimax_m31_f3_ar_prepushed", None) if TP2_F3 else None
        _F3_STATE["op"] = op
    return _F3_STATE["op"]


def _f3_bucket(rows: int) -> int:
    return max(64, 1 << (rows - 1).bit_length())  # mxfp8_cute_pinned_tactics._bucket


def _f3_init(core) -> None:
    """Prebuild (before warmup and capture, every rank, real weights): load the op, build the launcher on
    core.allreduce's own Lamport workspace, compile every pinned o_proj tactic of F3_MIN_ROWS..F3_MAX_ROWS and its
    alpha. Leaves core._f3_launcher None (F3 never taken) with one reason line otherwise."""
    core._f3_launcher = None
    why = None
    ar = getattr(core, "allreduce", None)
    ws = getattr(ar, "workspace", None) if ar is not None else None
    if not TP2_F3:
        why = "switched off (TRTLLM_M31_DISABLE_F3)"
    elif core.mapping.tp_size != 2:
        why = f"tp_size {core.mapping.tp_size} != 2"
    elif (
        ar.strategy not in (AllReduceStrategy.AUTO, AllReduceStrategy.ONESHOT)
        or getattr(ar, "mnnvl_allreduce", None) is not None
        or getattr(ar, "symm_mem_allreduce", None) is not None
    ):
        why = f"all-reduce strategy {ar.strategy} is not the one-shot Lamport"
    elif ws is None or ws.dtype != torch.int64 or not ws.is_cuda or ws.numel() < 7:
        why = "no Lamport workspace"
    elif _f3_op() is None:
        why = "the minimax_m31_f3_ar_prepushed op is not built"
    else:
        from tensorrt_llm._torch.custom_ops import mxfp8_cute_pinned_tactics as _pin

        from .kernels import m31_f3_gemm as _g

        first = min(core.attn_layers) if hasattr(core, "attn_layers") else 0
        n, k = core.w[f"l{first}_o_w"].shape
        if not _g.fi_base_ok():
            why = "FlashInfer dense_blockscaled_gemm_sm100.py is not the fork base 99892be6"
        elif (n, k) != (F3_N, F3_K):
            why = f"o_proj [{n}, {k}] is not [{F3_N}, {F3_K}]"
        else:
            launcher = _g.F3GemmLauncher(
                ws, core.mapping.tp_rank, F3_N, F3_K, torch.ops.trtllm.minimax_m31_f3_read_i64
            )
            tactics = sorted(
                {
                    _pin.TACTICS.get((F3_N, F3_K, _f3_bucket(r)))
                    for r in range(F3_MIN_ROWS, F3_MAX_ROWS + 1)
                }
                - {None}
            )
            if not tactics or any(t[4] != 1 for t in tactics):
                why = f"pinned tactics {tactics} are not all non-split-K"
            else:
                for t in tactics:
                    launcher.compile(t)
                launcher._alpha[ws.device] = torch.ones(1, dtype=torch.float32, device=ws.device)
                core._f3_launcher = launcher
                _F3_STATE["bad"] = torch.zeros(1, dtype=torch.int64, device=ws.device)
                _F3_STATE["checked"] = torch.zeros(1, dtype=torch.int64, device=ws.device)
                if F3_CHECK:
                    import atexit
                    import signal

                    atexit.register(_f3_report, core, True)
                    # Serving harnesses end the server with a signal, which skips atexit; the final
                    # report also runs on SIGTERM / SIGINT, then the previous handler (or the default action).
                    for _sig in (signal.SIGTERM, signal.SIGINT):
                        try:
                            _prev = signal.getsignal(_sig)

                            def _f3_on_signal(signum, frame, _prev=_prev):
                                try:
                                    _f3_report(core, True)
                                except Exception:  # noqa: BLE001
                                    pass
                                if callable(_prev):
                                    _prev(signum, frame)
                                elif _prev == signal.SIG_IGN:
                                    return
                                else:
                                    signal.signal(signum, signal.SIG_DFL)
                                    os.kill(os.getpid(), signum)

                            signal.signal(_sig, _f3_on_signal)
                        except ValueError:  # not the main thread: atexit only
                            logger.info(
                                f"[M3.1] TP2_F3 check: no {_sig.name} hook (not the main thread)"
                            )
                logger.info(
                    f"[M3.1] TP2_F3 ready rank={core.mapping.tp_rank}: rows {F3_MIN_ROWS}..{F3_MAX_ROWS}, tactics "
                    f"{tactics}, check={F3_CHECK}"
                    + (
                        ", device assert armed (every boundary, replays included)"
                        if F3_CHECK
                        else ""
                    )
                )
    if why is not None:
        logger.info(f"[M3.1] TP2_F3 not taken rank={core.mapping.tp_rank}: {why}")


def _f3_check(core, f3, residual, norm_w, normed, updated, q, sf) -> None:
    """F3_CHECK: the stock GEMM + AR + quantize on the same operands (one more collective round, in the graph), byte
    mismatches added on the device."""
    if f3.ref is None:
        _F3_STATE["bad"].add_(1)
        torch._assert_async(_F3_STATE["bad"] == 0, "TP2_F3 check: missing stock reference")
        return
    ref = m3_mxfp8_pinned_gemm(*f3.ref, torch.bfloat16, MXFP8_DECODE_GEMM_TACTIC)
    rn, rr = _native_allreduce_norm(core, ref, residual, norm_w)
    rq, rsf = m3_mxfp8_pinned_quantize(rn, MXFP8_DECODE_QUANT_TACTIC)
    bad = (normed.view(torch.int16) != rn.view(torch.int16)).sum()
    bad = bad + (updated.view(torch.int16) != rr.view(torch.int16)).sum()
    bad = bad + (q.view(torch.uint8) != rq.view(torch.uint8)).sum()
    bad = bad + (sf.view(torch.uint8).reshape(-1) != rsf.view(torch.uint8).reshape(-1)).sum()
    _F3_STATE["bad"].add_(bad)
    _F3_STATE["checked"].add_(1)
    # A device-side assert on every checked boundary, replays included (capture-safe). A mismatch on
    # any replay aborts the server, so a run that completes proves 0 bad without any end-of-run read.
    torch._assert_async(bad == 0, "TP2_F3 check: F3 boundary bytes differ from the stock chain")


def _f3_report(core, final: bool = False) -> None:
    """Log the F3 counters (host side: capture-time boundaries and slot takes; device side: F3_CHECK)."""
    if "bad" not in _F3_STATE or torch.cuda.is_current_stream_capturing():
        return
    if final and F3_CHECK:
        # A signal runs on the main thread; graph replay uses the executor stream.
        # Wait for its writes before reading the diagnostic counters on this thread.
        torch.cuda.synchronize(_F3_STATE["bad"].device)
    msg = (
        f"[M3.1] TP2_F3 {'final ' if final else ''}report rank={core.mapping.tp_rank}: captured boundaries "
        f"{_F3_STATE.get('taken', 0)}, slot taken {_F3_STATE.get('consumed', 0)}"
    )
    if F3_CHECK:
        msg += f", check boundaries {int(_F3_STATE['checked'].item())} mismatched bytes {int(_F3_STATE['bad'].item())}"
    logger.info(msg)


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

    def __init__(self, core: MiniMaxM3Tp2ModelingV2) -> None:
        self.core = core
        self._moe_push_descs: Dict[tuple, Any] = {}
        # Set by a fused feed-forward boundary that wrote the layer's Eagle3 capture (see _layer's ``capture``).
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
            max_seq_len=md.max_seq_len,
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
        # The decode kernels' rows (a per-capture constant): up to DECODE_STEP_MAX tokens of 1-D input ids.
        kernels = (
            inputs_embeds is None and input_ids.ndim == 1 and is_decode_step(input_ids.numel())
        )
        if kernels:
            num_tokens = int(input_ids.numel())
            plan = plan_for(num_tokens)
            _record("decode", num_tokens, plan.name)
            if num_tokens > TP2_ROWS:
                # R1VR8: a step past the TP2-lineage 8 rows on the decode layer.
                m3_report.work("R1VR8", f"{num_tokens}-row step on the decode layer ({plan.name})")
            if plan.name == "wide":
                taken("VR-BANDS wide decode step", num_tokens)
            elif plan.name == "wide448":
                taken("TP2WIDE_DECODE", num_tokens, f"rank {core.mapping.tp_rank}")
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
            is_capture = spec_metadata is not None and spec_metadata.is_layer_capture(i)
            if (
                kernels
                and residual is not None
                and is_capture
                and CAPTURE_LAYERS_FAST
                and CAPTURE_FUSED_BOUNDARY
            ):
                # The capture layer on the fast path (its boundary pushes); the capture reads the MoE / dense
                # boundary's reduced residual sum. With CAPTURE_NORM_IN_BOUNDARY the boundary all-reduce writes the
                # capture norm itself.
                capture_out = (
                    capture_target(spec_metadata, i, num_tokens)
                    if capture_target is not None and CAPTURE_NORM_IN_BOUNDARY
                    else None
                )
                self._capture_written = False
                hidden, residual = self._layer(
                    i, plan, position_ids, hidden, residual, step, capture=capture_out
                )
                taken("capture_layer_fast", num_tokens, f"layer {i}")
                if capture_out is not None and self._capture_written:
                    taken("capture_norm_in_boundary", num_tokens, f"layer {i}")
                    capture(spec_metadata, i, residual, None, written=True)
                else:
                    capture(spec_metadata, i, residual, None)
            elif not kernels or residual is None or is_capture:
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
        if TP2WIDE_DECODE and 64 < num_tokens <= DECODE_STEP_MAX:
            # Preserve the generic first layer's embedding and FlashInfer norm arithmetic.
            # These graph-owned tensors avoid preallocating an owner for every possible row count.
            raw = m3_embedding(input_ids, core.w["embed"])
            normed = flashinfer_rmsnorm(raw, core.w["l0_in_norm"], core.rms_eps)
            taken("TP2WIDE_INPUT", num_tokens, f"rank {core.mapping.tp_rank}")
            return normed, raw
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

    def _boundary(self, use_mc: bool, x, residual, norm_w, x2=None, quant=False, capture=None):
        """All-reduce + (second addend) + residual + RMSNorm: the multicast kernel, else the built-in all-reduce.
        ``capture``: the Eagle3 capture the multicast kernel also writes (``_capture_written`` records it)."""
        core = self.core
        if use_mc:
            fused = _mc_allreduce_norm(core, x, x2, residual, norm_w, quant=quant, capture=capture)
            if fused is not None:
                if capture is not None:
                    self._capture_written = True
                rows = x.shape[0]
                _mc_taken(
                    core.mapping,
                    rows,
                    ("x + x2" if x2 is not None else "x")
                    + (", MXFP8 copy" if quant else "")
                    + (", cluster" if rows <= MC_AR_CLUSTER_MAX_ROWS else ""),
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

    def _prepushed(self, i: int, x, residual, norm_w, quant, prepushed, x2=None, capture=None):
        fused = _mc_allreduce_norm(
            self.core, x, x2, residual, norm_w, quant=quant, prepushed=prepushed, capture=capture
        )
        if fused is None:
            raise RuntimeError(f"layer {i}: rows were pushed into an all-reduce that did not run")
        if capture is not None:
            self._capture_written = True
        rows = x.shape[0]
        _mc_taken(
            self.core.mapping,
            rows,
            f"prepushed {prepushed}" + (", cluster" if rows <= MC_AR_CLUSTER_MAX_ROWS else ""),
        )
        return fused

    def _f3_push(self, num_tokens: int, residual) -> Optional[F3Push]:
        """F3's o_proj target for this decode-graph step, or None: from the row count, the residual's shape and
        prebuild state only (every rank agrees)."""
        core = self.core
        launcher = getattr(core, "_f3_launcher", None)
        if launcher is None or not F3_MIN_ROWS <= num_tokens <= F3_MAX_ROWS:
            return None
        if (
            residual is None
            or residual.dim() != 2
            or residual.shape != (num_tokens, F3_N)
            or residual.dtype != torch.bfloat16
            or not residual.is_contiguous()
        ):
            return None
        from tensorrt_llm._torch.custom_ops import mxfp8_cute_pinned_tactics as _pin

        tactic = _pin.tactic_for(num_tokens, F3_N, F3_K)
        if tactic is None or tactic[4] != 1:
            return None
        return F3Push(launcher, tactic, num_tokens, F3_N, residual.device)

    def _f3_boundary(self, f3: F3Push, residual, norm_w):
        """The attention boundary after an F3 o_proj: the one-shot AR without its push loop + residual + RMSNorm, and
        the norm's MXFP8 (published in _F3_MX for the next pinned GENERIC MXFP8 linear)."""
        core = self.core
        rows, hidden = f3.rows, F3_N
        normed = torch.empty_like(residual)
        updated = torch.empty_like(residual)
        q = torch.empty((rows, hidden), dtype=torch.float8_e4m3fn, device=residual.device)
        sf = torch.empty(
            ((rows + 127) // 128 * 128) * (hidden // 32), dtype=torch.uint8, device=residual.device
        )
        _f3_op()(
            core.mapping.tp_rank,
            core.allreduce.workspace,
            f3.out,
            residual,
            updated,
            normed,
            norm_w,
            core.rms_eps,
            False,
            q,
            sf,
        )
        if F3_CHECK:
            _f3_check(core, f3, residual, norm_w, normed, updated, q, sf)
        _F3_MX.publish(normed, q, sf)
        _record("m31_f3_boundary", rows)
        cnt = _F3_STATE["taken"] = _F3_STATE.get("taken", 0) + 1
        if cnt & (cnt - 1) == 0:
            logger.info(
                f"[M3.1] TP2_F3 taken n={cnt} rows={rows} rank={core.mapping.tp_rank} tactic={f3.tactic}"
            )
        return normed, updated

    def _layer(self, i: int, plan: DecodePlan, position_ids, hidden, residual, step, capture=None):
        """``capture``: (out, fc_norm weight, eps) for the feed-forward boundary to write the layer's Eagle3 capture
        norm (CAPTURE_NORM_IN_BOUNDARY, for a capture layer on this path); a fused boundary that writes it sets
        ``_capture_written``."""
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
        f3 = (
            self._f3_push(num_tokens, residual)
            if armed is None and not plan.attn_mc_ar and decode_graph
            else None
        )
        if not decode_graph and i == 0:
            _f3_report(core)
        hidden, attn_pushed = self._attention(
            i, position_ids, hidden, step, armed if f3 is None else f3
        )
        post_norm = core.w[f"l{i}_post_norm"]
        if f3 is not None and attn_pushed:
            hidden, residual = self._f3_boundary(f3, residual, post_norm)
        elif armed is not None and attn_pushed:
            taken(ATTN_PUSH.path, num_tokens)
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
            # A dense layer: its MLP's down projection pushes into the post-MLP boundary's all-reduce, as the o_proj
            # does into the attention one (the same slot; the gate_up projection never pushes).
            # R6: the next layer's QKV takes its MXFP8 activation from this boundary, as after a MoE layer.
            quant = R6_DENSE_PREQUANT and plan.qkv_prequant and i + 1 < core.num_layers
            if plan.qkv_prequant and i + 1 < core.num_layers:
                _log_once(
                    logger.info,
                    "r6_dense_prequant",
                    "on" if quant else "off",
                    "[M3 dispatch] r6_dense_prequant "
                    + (
                        "taken: the dense-MLP boundary hands the next QKV its MXFP8 activation"
                        if quant
                        else "off (VR default; TRTLLM_M31_ENABLE_R6=1 turns it on), BF16 hand-off"
                    )
                    + " (logged once)",
                )
            dense_push = plan.attn_mc_ar and decode_graph and DENSE_PUSH.covers(num_tokens)
            armed = self._attn_push_target(num_tokens) if dense_push else None
            out, dense_pushed = self._swiglu_mlp(hidden, f"l{i}_mlp", True, armed, dense=True)
            if armed is not None and dense_pushed:
                taken(DENSE_PUSH.path, num_tokens)
                return self._prepushed(
                    i, out, residual, norm_w, quant, AR_PUSH_MODE, capture=capture
                )
            if armed is not None:
                missed(
                    DENSE_PUSH.path,
                    num_tokens,
                    "the dense MLP's down projection did not push its rows into the armed boundary all-reduce",
                )
            return self._boundary(
                plan.attn_mc_ar, out, residual, norm_w, quant=quant, capture=capture
            )

        quant = plan.qkv_prequant and i + 1 < core.num_layers
        path = moe_path(num_tokens)
        moe_push = None
        if path == MOE_ENGINE.path:
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
        elif path == MOE16_ENGINE.path and decode_graph:
            # Router, routed and shared experts in one kernel; its bf16 sum goes to the boundary all-reduce.
            engine = core.moe16_engine(i, num_tokens)
            if engine is not None:
                _record("m3_moe16_engine", num_tokens)
                out = m3_moe16_engine(hidden.contiguous(), engine)
                taken(MOE16_ENGINE.path, num_tokens)
                return self._boundary(
                    plan.moe_mc_ar, out, residual, norm_w, quant=quant, capture=capture
                )
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
                taken(MC_FINALIZE.path, num_tokens)
                return fused
        if (
            TP2W_S2_TAIL
            and not plan.moe_mc_ar
            and moe_push is None
            and capture is None
            and not quant
            and i in core.moe_layers
        ):
            s2 = self._s2_tail(i, hidden, residual, norm_w)
            if s2 is not None:
                return s2
        # The parts only where the boundary folds the add (the multicast all-reduce) or a push is armed; the built-in
        # all-reduce boundary ("wide" plan) adds them first anyway, so there the MoE returns the sum
        # (TP2W_MOE_FINALIZE).
        shared, routed, engine_pushed, shared_pushed = self._moe(
            i, hidden, return_parts=plan.moe_mc_ar or moe_push is not None, push=moe_push
        )
        if moe_push is not None:
            if routed is None:
                # The engine folded the shared expert in and pushed routed + shared into set A.
                if engine_pushed:
                    taken(MOE_PUSH.path, num_tokens, "engine, routed + shared")
                    return self._prepushed(i, shared, residual, norm_w, quant, 2, capture=capture)
            else:
                if engine_pushed and shared_pushed:
                    taken(MOE_PUSH.path, num_tokens, "routed, shared")
                    return self._prepushed(
                        i, shared, residual, norm_w, quant, 3, x2=routed, capture=capture
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

    def _s2_tail(self, i: int, h: torch.Tensor, residual, norm_w):
        """TP2W_S2_TAIL on the decode layer's built-in all-reduce boundary ("wide" plans): MoE layer ``i`` with its
        boundary as one fused kernel, or None when the step is not the stage's (rows outside
        S2_TAIL_MIN_ROWS..S2_TAIL_MAX_ROWS, not the Triton-finalize MoE, the final norm). A declined fused boundary
        finalizes and reduces as today."""
        core = self.core
        rows = h.shape[0] if h.dim() == 2 else 0
        if (
            h.dtype != torch.bfloat16
            or not S2_TAIL_MIN_ROWS <= rows <= S2_TAIL_MAX_ROWS
            or not _triton_finalize_rows(core, h)
            or norm_w is core.w["final_norm"]
            or is_torch_compiling()
            or _s2_op() is None
        ):
            return None
        (shared, fc2, scales, perm), _, _, _ = self._moe(
            i, h, return_parts=False, do_finalize=False
        )
        fused = _s2_boundary(
            core, rows, (fc2, scales, perm), shared, residual, norm_w, core.allreduce
        )
        if fused is not None:
            return fused
        return _native_allreduce_norm(
            core, _triton_finalize(rows, (fc2, scales, perm), shared), residual, norm_w
        )

    def _swiglu_mlp(
        self,
        h: torch.Tensor,
        prefix: str,
        low_m_gemm: bool,
        push: Optional[tuple] = None,
        dense: bool = False,
    ):
        """A gated MXFP8 MLP (the dense MLP with ``dense``, the shared expert): gate_up, SwiGLU-OAI, down, no
        reduction. Returns (output, whether the down projection's GEMV pushed into ``push``)."""
        core = self.core
        gu, _ = _linear(core, h, f"{prefix}_gu_w", f"{prefix}_gu_s", low_m_gemm)
        n, k = core.w[f"{prefix}_dn_w"].shape
        rows = gu.shape[0]
        if not dense and rows in SWQ_ROWS.get((n, k), ()) and _mx_mode(core.mxfp8).decode_graph:
            # The MXFP8 activation is published keyed by ``gu`` and taken by the down projection's prequant branch.
            act, act_scale = m3_swq(gu, core.swiglu_limit, core.swiglu_alpha, core.swiglu_beta)
            m3_prequant.SLOT.publish(gu, act, act_scale)
            if rows not in _SWQ_LOGGED:
                _SWQ_LOGGED.add(rows)
                logger.info(f"[M3 dispatch] SWQ: SwiGLU + MXFP8 producer rows={rows} [{n}, {k}]")
            m3_report.work("SWQ", f"{rows} rows [{n}, {k}]")
            return _linear(core, gu, f"{prefix}_dn_w", f"{prefix}_dn_s", low_m_gemm, push)
        if dense:
            act = _dense_swiglu(core, gu)
        else:
            act = _native_swiglu(core, gu, False)
            if act is None:
                act = m3_silu_and_mul(gu, core.swiglu_limit, core.swiglu_alpha, core.swiglu_beta)
        return _linear(core, act, f"{prefix}_dn_w", f"{prefix}_dn_s", low_m_gemm, push)

    def _routed_chunks(self, i: int, x, router_logits, topk=None):
        """The routed experts from the bf16 activation over at most ``moe_max_num_tokens`` rows per runner call: the MoE
        scheduler's even split, even chunks on the MoE chunking stream. ``topk`` (S1B_FOLD) splits with the rows."""
        core = self.core
        rows = x.shape[0]
        num_chunks = (rows + core.moe_max_num_tokens - 1) // core.moe_max_num_tokens
        if num_chunks <= 1:
            return _routed_experts(core, i, x, None, router_logits, True, topk)
        sizes = _split_chunk(rows, num_chunks)
        xs = x.split(sizes)
        if topk is None:
            logits, tks = router_logits.split(sizes), [None] * num_chunks
        else:
            logits, tks = [None] * num_chunks, list(zip(topk[0].split(sizes), topk[1].split(sizes)))
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
                    outs.append(_routed_experts(core, i, xs[c], None, logits[c], True, tks[c]))
            else:
                outs.append(_routed_experts(core, i, xs[c], None, logits[c], True, tks[c]))
        if aux is not None:
            with torch.cuda.stream(aux):
                ev_overlap.record()
            ev_overlap.wait()
        return torch.cat(outs)

    def _routed(self, i: int, h: torch.Tensor, do_finalize: bool):
        """Router + routed experts: the router logits of a generation step (``_decode_router_logits``) and the experts
        quantizing their bf16 input."""
        core = self.core
        rows = h.shape[0] if h.dim() == 2 else 0
        fp4 = [] if X1 and 9 <= rows <= min(48, core.moe_max_num_tokens) else None
        kind, route = _decode_route(core, i, h, fp4)
        router_logits, topk = (route, None) if kind == "logits" else (None, route)
        if fp4:
            x, x_sf = fp4
            _record("m31_x1_router_fp4", rows)
            taken("X1 router_fp4", rows, "native router payload consumed by routed experts")
            return _routed_experts(core, i, x, x_sf, router_logits, do_finalize)
        if S1B_FOLD_CHECK and topk is not None:
            ref_logits = m3_fp32_linear(h, core.w[f"l{i}_router"])
            if not do_finalize:
                rows = h.shape[0]
                got = _routed_experts(core, i, h, None, None, False, topk)
                ref = _routed_experts(core, i, h, None, ref_logits, False)
                _S1bCheck.record(
                    _triton_finalize(rows, got, None),
                    _triton_finalize(rows, ref, None),
                    _s1b_ident(core, i, h, False),
                )
                return ref
            got = self._routed_chunks(i, h, None, topk)
            ref = self._routed_chunks(i, h, ref_logits)
            _S1bCheck.record(got, ref, _s1b_ident(core, i, h, True))
            return ref
        if not do_finalize:
            return _routed_experts(core, i, h, None, router_logits, False, topk)
        return self._routed_chunks(i, h, router_logits, topk)

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
                taken(MOE_ENGINE.path, rows)
                return folded, None, engine_arp is not None, False
        if do_finalize and not return_parts and push is None and _triton_finalize_rows(core, h):
            routed_parts, (shared_out, _) = maybe_execute_in_parallel(
                lambda: self._routed(i, h, False),
                lambda: self._swiglu_mlp(h, f"l{i}_sh", False),
                core.moe_events[i][0],
                core.moe_events[i][1],
                None if flashinfer_mxfp8_autotune_active() else core.aux_stream_moe,
                disable_on_compile=True,
            )
            return _triton_finalize(rows, routed_parts, shared_out), None, False, False
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
            # Dense layers 0-2: QKV, the fused norm + RoPE (FP8 q/k/v: an FP8 cache, or the FP8 pool the dense layers
            # sit in within a hybrid NVFP4 cache on a decode step), the attention, o_proj. A covered (query rows,
            # requests) shape attends on the full-context PrimTS decode (``FULLCTX_DECODE``), the norm/RoPE kernel then
            # storing the new K / V rows itself (``DENSE_KV_INSERT_FUSED``); every other step the MSA backend.
            qkv, _ = _linear(core, hidden, f"l{i}_qkv_w", f"l{i}_qkv_s", True)
            out_fp8 = core.main_kv_is_fp8 or core.main_kv_is_nvfp4
            plan = self._fullctx_plan(i, qkv.device, step)
            kv_dest = None
            if (
                plan is not None
                and out_fp8
                and DENSE_KV_INSERT_FUSED
                and DENSE_QKV_FP8_VIA_BF16
                and qkv.dtype == torch.bfloat16
                and core.head_dim == 128
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
            fused = _qk_norm_rope(core, i, qkv, position_ids, out_fp8, kv_dest)
            if kv_dest is not None:
                taken(DENSE_KV_INSERT_PATH, kv_dest[2])
            q, k, v = fused.split([core.q_size, core.kv_size, core.kv_size], dim=-1)
            if kv_dest is not None:
                q = kv_dest[3].view(q.shape[0], -1)
            o = None
            if plan is not None:
                o = self._fullctx_decode(i, q, k, v, step, plan, kv_written=kv_dest is not None)
            if o is None:
                o = self._attention_core(i, q, k, v, None, step)
            return _linear(core, o, f"l{i}_o_w", f"l{i}_o_s", True, push)
        if not core.fused_projection:
            return self._unfused_sparse_attention(i, position_ids, hidden, step, push)
        # QKV_PRODUCER_FOLD: the QKV GEMV may be deferred to the producer call site.
        packed, _ = _linear(
            core, hidden, f"l{i}_qkv_w", f"l{i}_qkv_s", True, defer=QKV_PRODUCER_FOLD
        )
        deferred = None
        if isinstance(packed, _DeferredGemv):
            deferred, packed = packed, packed.placeholder
        horizontal = _sparse_qkv_producer(
            core,
            i,
            packed,
            position_ids,
            step["kv_cache_manager"],
            step["msa_idx_k_cache"],
            step["msa_out_cache_loc"],
            deferred,
        )
        if horizontal is None:
            raise RuntimeError(
                f"MiniMax-M3 sparse layer {i}: no horizontal producer takes this step's caches (an FP8 or NVFP4 HND "
                "pool with an FP8 index-K cache)"
            )
        q, idx_q = horizontal
        o = self._attention_core(i, q, None, None, idx_q, step)
        return _linear(core, o, f"l{i}_o_w", f"l{i}_o_s", True, push)

    def _fullctx_plan(self, i: int, device: torch.device, step):
        """(the PrimTS split-KV decode plan, the step's query tokens) where dense layer ``i``'s decode attention runs on
        kernels/m3_fullctx_decode: a covered (query rows per request, requests) shape over the layer's FP8 pool. None
        otherwise (the MSA backend runs)."""
        path = FULLCTX_DECODE.path
        if is_torch_compiling():
            uncovered(path, "not a pure generation step on the MSA backend")
            return None
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

    def _fullctx_decode(
        self, i: int, q, k, v, step, plan_tokens, *, kv_written: bool
    ) -> torch.Tensor:
        """Dense layer ``i``'s decode attention on the PrimTS plan of ``_fullctx_plan``: the step's new K / V go into
        the paged cache as the MSA backend writes them (``write_msa_phase_kv``; already there when ``kv_written``),
        then every query row attends its request's full causal context. The merge also emits the o_proj's MXFP8
        input (``FULLCTX_DECODE_MXFP8``) when the plan supports it."""
        core = self.core
        backend = core.attn[i]
        plan, num_tokens = plan_tokens
        nreq = int(step["num_seqs"])
        if not kv_written:
            write_msa_phase_kv(
                backend,
                k[:num_tokens],
                v[:num_tokens],
                step["metadata"],
                AttentionInputType.mixed,
                token_offset=0,
            )
        pool, subpages = step["kv_cache_manager"].get_kv_subpage_pool(i, "HND")
        output = q.new_empty((q.shape[0], core.heads * core.head_dim), dtype=torch.bfloat16)
        k_pages, v_pages = _fullctx_decode_kernel.kv_page_views(pool, subpages)
        mxfp8 = None
        if (
            FULLCTX_DECODE_MXFP8
            and output.shape[0] == num_tokens
            and _fullctx_decode_kernel.emits_mxfp8()
            and MERGE_MXFP8_MIN_TOKENS <= num_tokens <= MERGE_MXFP8.TRITON_MAX_ROWS
        ):
            mxfp8 = MERGE_MXFP8.buffers(output.shape[1], output.device)
        _record("m3_fullctx_decode", num_tokens, kv_written, mxfp8 is not None)
        emitted = plan.run(
            fp8_e4m3_saturate(q[:num_tokens].reshape(num_tokens, core.heads, core.head_dim)),
            k_pages,
            v_pages,
            step["msa_block_table"][:nreq],
            step["msa_seq_lens_cuda"][:nreq],
            core.sm_scale(i),
            output[:num_tokens],
            mxfp8=mxfp8,
        )
        if emitted:
            m3_prequant.SLOT.publish(output, *MERGE_MXFP8.handoff(mxfp8, num_tokens))
            taken(FULLCTX_DECODE_MXFP8_PATH, num_tokens)
        taken(FULLCTX_DECODE.path, num_tokens)
        return output

    def _unfused_sparse_attention(self, i: int, position_ids, hidden, step, push):
        """A sparse layer without the five-way projection (``fuse_qkv_index_projection=False``): the QKV projection
        and its fused norm + RoPE (FP8 q/k/v into an FP8 cache) on the current stream; the MXFP8 index projection
        (the checkpoint's block scales applied) and the FP8 indexer kernel on the attention aux stream; then the MSA
        backend (it writes the main KV cache) and the o_proj. The NVFP4 producer does not apply: it packs the
        five-way projection's output. ``push`` is the attention
        boundary exchange the o_proj GEMV writes its rows into, as in the five-way path."""
        core = self.core
        if _producer_covered(core):
            # Not a miss: the producer packs the five-way projection's output, which this configuration lacks.
            uncovered(
                PRODUCER.path,
                "fuse_qkv_index_projection is disabled: QKV + separate MXFP8 index projection",
            )

        def _main():
            qkv, _ = _linear(core, hidden, f"l{i}_qkv_w", f"l{i}_qkv_s", True)
            fused = _qk_norm_rope(core, i, qkv, position_ids, core.main_kv_is_fp8)
            return fused.split([core.q_size, core.kv_size, core.kv_size], dim=-1)

        def _index():
            idx_qk, _ = _linear(core, hidden, f"l{i}_idx_w", f"l{i}_idx_s", True)
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

    @property
    def _nvfp4_index_scorer(self) -> Nvfp4IndexDecodeScorer:
        scorer = getattr(self, "_nvfp4_index_scorer_obj", None)
        if scorer is None:
            scorer = self._nvfp4_index_scorer_obj = Nvfp4IndexDecodeScorer(self.core.index_heads)
        return scorer

    def _decode_sparse_core(self, i: int, q, idx_q, step, output) -> None:
        """Top-16 block selection and the block-sparse decode of one pure generation step of a sparse layer whose caches
        the producer already wrote.

        The selection is the identity when the metadata staged it (every request within 16 blocks). Otherwise the CuTe
        DSL scorer writes the per-index-head block maxima, reduced to the KV head, and the long-row selector takes the
        top 16 (the built-in selector beyond its shapes). A selected table at 33..128 rows on an NVFP4 layer runs LC2
        (TP2WIDE_LC2, ``SPARSE_DECODE_LC``), which also prefetches into L2, before its grid wait, the K/V pages of the
        blocks the same layer selected one step back (``SPARSE_DECODE_LC_PREFETCH_SELF_KV``; hints only).
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
            if _nvfp4_index_k.is_nvfp4_index_cache(idx_k_cache):
                # AGG_NVFP4_INDEXK decode reader: the Q8KV4 indexer scores the NVFP4 pages in the select layout, at
                # the FP8 path's tile width (the same selector geometry).
                tiles = int(max_score.shape[1])
                # NVFP4_STEP_CACHE: the reader's history scheduler depends on the step only; build it for the first
                # sparse layer of the step (this step dict is per forward) and reuse it.
                workspace = None
                if NVFP4_STEP_CACHE:
                    key = ("nvfp4_hpr_ws", tiles)
                    workspace = step.get(key)
                    if workspace is None:
                        workspace = step[key] = Nvfp4IndexDecodeScorer.history_workspace(
                            seq_lens, tiles
                        )
                max_score = self._nvfp4_index_scorer.score(
                    idx_q_view, idx_k_cache, block_table[:, :tiles], seq_lens, qlen, workspace
                )
                if rows not in _NVFP4_INDEXK_DECODE_LOGGED:
                    _NVFP4_INDEXK_DECODE_LOGGED.add(rows)
                    logger.info(
                        f"[M3.1 tp2] AGG_NVFP4_INDEXK decode reader taken: {rows} requests x {qlen}, Q8KV4 HPR"
                    )
            elif INDEX_SCORE_T2SC_PATH.covers(rows):
                # T2SC: this target's copy of the TP2 / TP8 scorer at the request count's split and prestage.
                split_k, early = index_score_t2sc_schedule(rows)
                key = (
                    idx_q_view.dtype,
                    int(config.num_index_heads),
                    int(qlen),
                    int(idx_q_view.shape[-1]),
                    int(split_k),
                    bool(early),
                )
                if key not in _index_score_t2sc_kernel.IndexScoreRunner.kernel_cache:
                    missed(
                        INDEX_SCORE_T2SC_PATH.path,
                        num_tokens,
                        f"the T2SC index scorer {key} was not compiled before warmup",
                    )
                _record("m3_index_score_t2sc", key)
                m3_index_score_t2sc(
                    idx_q_view,
                    idx_k_cache.squeeze(1),
                    block_table,
                    seq_lens,
                    max_score.transpose(1, 2),
                    qlen,
                    split_k,
                    early,
                )
                taken(INDEX_SCORE_T2SC_PATH.path, rows)
            elif (
                INDEX_SCORE_INTREE
                and rows >= INDEX_SCORE_INTREE_MIN_REQUESTS
                and num_tokens > TP2_WIDE_MIN_ROWS - 1
            ):
                # TP2 WIDE: the in-tree scorer (same scores), compiled before warmup (_prebuild_index_scorer_intree).
                in_tree = _in_tree_scorer()
                key = (
                    idx_q_view.dtype,
                    int(config.num_index_heads),
                    int(qlen),
                    int(idx_q_view.shape[-1]),
                    int(in_tree.SPLIT_K),
                )
                if key not in in_tree.kernel_cache:
                    missed(
                        INDEX_SCORE.path,
                        num_tokens,
                        f"the in-tree index scorer {key} was not compiled before warmup",
                    )
                if rows not in _INDEX_SCORE_INTREE_LOGGED:
                    _INDEX_SCORE_INTREE_LOGGED.add(rows)
                    logger.info(
                        f"[M3.1 tp2] TP2WIDE_INTREE_SCORER taken: {rows} requests, "
                        "the in-tree CuTe DSL scorer (split 256)"
                    )
                _record("cute_dsl_minimax_m3_index_decode_score", rows, qlen)
                torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score(
                    idx_q_view,
                    idx_k_cache.squeeze(1),
                    block_table,
                    seq_lens,
                    max_score.transpose(1, 2),
                    qlen,
                )
            else:
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
                taken(INDEX_SCORE.path, rows)
            n_valid_blocks = step["msa_n_valid_blocks"][:num_tokens]
            group = config.num_index_heads // config.num_kv_heads
            max_score_kv = (
                max_score.view(
                    config.num_kv_heads, group, max_score.shape[1], max_score.shape[2]
                ).amax(dim=1)
                if group > 1
                else max_score
            )
            early_trigger = num_tokens <= SELECT_EARLY_MAX_ROWS
            trigger_phase = (
                SELECT_LONG_TRIGGER_PHASE if num_tokens <= SELECT_LONG_TRIGGER_MAX_ROWS else 0
            )
            dsl = False
            if LONG_SELECT_DSL.keys and 128 < max_score_kv.shape[1]:
                # Compiled before warmup (prebuild_kernels); a missing kernel falls back to the 1024-thread one.
                key = _select_long_dsl_kernel.kernel_key(
                    get_env_enable_pdl(), early_trigger, trigger_phase
                )
                dsl = key in _select_long_dsl_kernel.kernel_cache
                if not dsl:
                    missed(
                        LONG_SELECT_DSL.path,
                        num_tokens,
                        f"the 384-thread select {key} was not compiled before warmup",
                    )
            table = long_select_blocks(
                max_score_kv,
                n_valid_blocks=n_valid_blocks,
                init_blocks=config.init_blocks,
                local_blocks=config.local_blocks,
                use_pdl=get_env_enable_pdl(),
                early_trigger=early_trigger,
                trigger_phase=trigger_phase,
                fast_tail=SELECT_LONG_SL3,
                dsl=dsl,
            )
            if table is not None:
                taken((LONG_SELECT_DSL if dsl else LONG_SELECT).path, num_tokens)
            else:
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
        # The long-context decode (if it runs: TP2WIDE_LC2) takes the layer's N1 id buffer; any other route (the Triton
        # producer + merge, every step of at most 32 rows) leaves the slot unused, and it is cleared after the call
        # either way.
        n1_ids = core.n1_ids.get(i)
        if n1_ids is not None and nvfp4 and num_tokens <= N1_MAX_ROWS:
            m3_attn_lc.N1_IDS.ids = n1_ids
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
            m3_attn_lc.N1_IDS.ids = None


# --------------------------------------------------------------------------------------------------------------------
# Speculative decoding: one-model Eagle3 at checkpoint precision
# --------------------------------------------------------------------------------------------------------------------


class M3SpecStepFold:
    """``SpecStepFold`` of this target.

    Between two decode graphs of a one-model Eagle3 step, the spec sampler stores the step's outputs into its per-slot
    store and the engine gathers them back into its flat input buffers for the next step. This object does both in one
    launch each (``m3_spec_tail.slot_scatter`` / ``step_gather``): integer copies only, the same values the engine's and
    the sampler's own copies write.

    With ``in_forward`` (``STEP_SPEC_FOLD_IN_FORWARD``) the forward of a decode-only step also does both at its tail
    (``store_in_forward``, one ``m3_fold_tail`` launch inside the step graph): it stores the step's outputs for the rows
    ``begin_step`` armed and stages the next step's inputs for those rows in that order. The arming is a device-side row
    count and slot list staged before the replay, so a captured graph whose step was not armed writes nothing. The
    sampler then skips its store for exactly the armed rows (``sampler_rows_stored``), and the next step skips its
    gathers when its rows are those rows (``begin_step``).

    One instance per target model; the target passes it to ``RuntimeHooks.spec_step_fold``.
    """

    def __init__(
        self, in_forward: bool = False, store_copy_wait: bool = False, cpp_tail: bool = False
    ) -> None:
        self.store: Optional[object] = None
        self.step_inputs: Optional[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None
        self.in_forward = in_forward
        # The forward's tail as this target's C++ op instead of the Triton kernel (m3_fold_tail.fold_tail(cpp=)).
        self.cpp_tail = in_forward and cpp_tail
        # Device arming read by the forward's tail: ctrl[0] = armed rows (0 = write nothing), slots[:rows].
        self._ctrl: Optional[torch.Tensor] = None
        self._slots: Optional[torch.Tensor] = None
        # The rows (sequence slots) the next forward stores and stages for, as begin_step armed them.
        self._armed: Optional[tuple[int, ...]] = None
        # The rows the previous forward staged the next step's inputs for (confirmed by sampler_rows_stored).
        self._staged_for: Optional[tuple[int, ...]] = None
        # Output buffers the tail ran on: the captured graphs' static outputs (by address) and the last eager step's.
        self._graph_outputs: set[int] = set()
        self._eager_outputs: Optional[torch.Tensor] = None
        self._logged = False
        # store_copy_wait: the sampler records this (external) event on its D2H side stream after copying the store
        # to the host, and the forward's tail waits on it before it rewrites the store; the sampler can then copy
        # on the side stream on the steps this forward stored the rows (see SpecSamplerBase._copy_outputs_to_host).
        self._store_copy_wait = in_forward and store_copy_wait
        self.store_copies_done: Optional[torch.cuda.Event] = None

    def bind_engine_buffers(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        position_id_offsets: torch.Tensor,
        kv_lens_offsets: torch.Tensor,
        max_batch_size: int,
    ) -> None:
        self.step_inputs = (input_ids, draft_tokens, position_id_offsets, kv_lens_offsets)
        if self._store_copy_wait:
            # Recorded once now so a captured wait has a completed record to wait on before the first copy.
            self.store_copies_done = torch.cuda.Event(external=True)
            self.store_copies_done.record()
            logger.info(
                "[M3 host] spec store host copies on the D2H side stream, awaited by the forward's tail"
            )
        if self.cpp_tail:
            logger.info(
                "[M3 host] spec fold tail: one C++ launch (no Triton) at the forward's tail"
            )
        if self.in_forward:
            self._ctrl = torch.zeros(1, dtype=torch.int32, device=input_ids.device)
            self._slots = torch.zeros(
                max(max_batch_size, 1), dtype=torch.long, device=input_ids.device
            )

    def bind_sampler_store(self, store: object) -> None:
        self.store = store

    def begin_step(self, seq_slots: Optional[list[int]], num_rows: int) -> bool:
        if not self.in_forward or self._ctrl is None:
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
    # the fold tail (a plain launch, no griddepcontrol).
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
            cpp=self.cpp_tail,
        ):
            return
        if torch.cuda.is_current_stream_capturing():
            self._graph_outputs.add(new_tokens.data_ptr())
        else:
            self._eager_outputs = new_tokens

    def gather(
        self,
        new_tokens: torch.Tensor,
        next_draft_tokens: torch.Tensor,
        new_tokens_lens: torch.Tensor,
        previous_slots: torch.Tensor,
        previous_pos_indices: torch.Tensor,
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

    def sampler_rows_stored(self, outputs: dict, num_skip: int, seq_slots: list[int]) -> bool:
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
        self,
        seq_slots: torch.Tensor,
        new_tokens: torch.Tensor,
        next_new_tokens: torch.Tensor,
        new_tokens_lens: torch.Tensor,
        next_draft_tokens: torch.Tensor,
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


class M3DSparkStepFold(M3SpecStepFold):
    """The step fold under the standalone DSpark worker (DSFOLD): the engine's previous-batch gathers in one launch
    (``step_gather``). The sampler's slot stores stay on its in-tree path, so ``scatter`` declines every step.
    ``taken`` / ``declined`` count the steps the gather ran / fell back to the engine's own copies."""

    def __init__(self) -> None:
        super().__init__()
        self.taken = 0
        self.declined = 0

    def gather(self, *args, **kwargs) -> bool:
        if not super().gather(*args, **kwargs):
            self.declined += 1
            uncovered(
                "DSFOLD",
                "step_gather declined the inputs (dtype / layout); the engine's copies run",
            )
            return False
        if self.taken == 0:
            logger.info(
                "[M3 dispatch] DSFOLD DSpark step-input gather: engaged (one step_gather launch)"
            )
        self.taken += 1
        m3_report.work("DSFOLD", f"step_gather (declined {self.declined})")
        return True

    def scatter(
        self, seq_slots, new_tokens, next_new_tokens, new_tokens_lens, next_draft_tokens
    ) -> bool:
        return False


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
        # The drafter embeds with the target's table, read off the core on every call (the engine materializes the
        # core's weights after construction).
        self._core = [core]
        self.model = _DraftForward(self)
        # The projections (weight keys) on the fixed-tactic low-M kernels (``DRAFT_LOWM``), set by
        # ``prebuild_linears``.
        self._lowm_keys: frozenset = frozenset()

    def prebuild_linears(self) -> int:
        """Put the drafter's BF16 projections ``DRAFT_LOWM`` covers on the fixed-tactic low-M kernels and compile
        every covered call. Once the weights are real, before the executor's warmup, on every rank. Returns the
        number of low-M calls compiled."""
        keys = ("fc", "qkv", "o", "gu", "dn")
        built = _draft_lowm_kernel.prebuild([self.w[k] for k in keys], DRAFT_LOWM.keys)
        self._lowm_keys = frozenset(k for k in keys if _draft_lowm_kernel.installed(self.w[k]))
        return built

    def _linear(self, x: torch.Tensor, key: str) -> torch.Tensor:
        """``x @ w[key]^T``: the low-M kernel at the rows ``DRAFT_LOWM`` covers (``prebuild_linears``), else the plain
        GEMM."""
        w = self.w[key]
        if key in self._lowm_keys:
            out = _draft_lowm_kernel.draft_linear(w, x, None)
            if out is not None:
                return out
        return m3_bf16_linear(x, w)

    @property
    def embed_table(self) -> torch.Tensor:
        return self._core[0].w["embed"]

    def load_weights_from_target_model(self, target_model: nn.Module) -> None:
        """Nothing to alias: the drafter reads the target's embedding table through the core and owns its head."""

    # ------------------------------------------------------------------------------------------------- the layer

    def fc(self, hs: torch.Tensor) -> torch.Tensor:
        return self._linear(hs, "fc")

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
        qkv = self._linear(x, "qkv")
        _record("m3_draft_attention", int(md.num_tokens))
        o = m3_draft_attention(self.attn, qkv, md)
        out = self._linear(o, "o")
        return m3_allreduce(out, self.allreduce, None) if reduce else out

    def mlp(self, h: torch.Tensor, reduce: bool, silu_cpp: bool = False) -> torch.Tensor:
        """gate_up, the plain SwiGLU (with ``silu_cpp``, ``DRAFT_SILU_CPP``: this target's C++ op, bit for bit the
        Triton one, in one launch), down."""
        gu = self._linear(h, "gu")
        act = None
        if silu_cpp and DRAFT_SILU_CPP:
            act = m3_draft_silu(gu)
            if act is None:
                uncovered(
                    DRAFT_SILU_PATH, "the drafter SwiGLU input layout (the Triton op runs it)"
                )
            else:
                taken(DRAFT_SILU_PATH, int(gu.shape[0]), "drafter MLP SwiGLU, one C++ launch")
        if act is None:
            act = m3_silu_and_mul(gu, None, None, None)
        out = self._linear(act, "dn")
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
        # Per step: the captured layers written with the capture norm (set by capture_hidden_states).
        self._m3_capture_normed: set = set()
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

    def capture_target(
        self, spec_metadata, layer_id: int, num_tokens: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, float]]:
        """Where this step's capture norm writes layer ``layer_id``'s capture, for a boundary all-reduce that writes it
        itself (``CAPTURE_NORM_IN_BOUNDARY``): (the capture buffer's [num_tokens, H] column block, the layer's
        fc_norm weight, eps), or None (no capture norm armed, not a captured layer, or more rows than the buffer).
        The boundary writes every row it reduces; rows past num_capture_tokens are unread."""
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
        """The Eagle3 capture with this step's capture norm (``written``: the layer's boundary all-reduce already
        wrote it, see ``capture_target``); on a step armed for the early fc fork (``arm_capture_norm``), the last
        captured layer then forks the drafter's first fc onto the aux stream, where it overlaps the remaining target
        layers and the head (joined by ``_m3_target_accept``)."""
        if not written:
            _eagle3_capture(spec_metadata, layer_id, hidden_states, residual, self._m3_capture_norm)
        if self._m3_capture_norm is not None:
            # Written with this step's capture norm: by the boundary all-reduce (written) or just above.
            self._m3_capture_normed.add(layer_id)
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
            taken("m3_first_drafter_inputs", attn_metadata.num_seqs)
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
        most TP2_ROWS verify rows (the fused tail's certified rows; R1-VR-TP8 does not widen it). Host state only, so
        the TP ranks agree."""
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
        elif attn_metadata.num_seqs > SAMPLED_VERIFY_MAX_REQUESTS:
            uncovered("m3_sampled_accept", f"sampled verify at {attn_metadata.num_seqs} requests")
            return False
        elif not self._m3_sampled_verify_ok(spec_metadata):
            return False
        self._m3_target_lm_head = lm_head
        return True

    def draft_only_eligible(self, attn_metadata, spec_metadata, draft_model) -> bool:
        """Whether a step verified by the in-tree path can still take the fused first drafter inputs: the structural
        conditions of tail_eligible, without the target-sampling ones, up to ``FIRST_DRAFTER_INPUTS_MAX_REQUESTS``
        requests."""
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
            else:
                hs = self._m3_renorm_raw_captures(hs, spec_metadata, draft_model)
            hidden = draft_model.fc(hs)
        input_ids = accepted_tokens[:, : spec_metadata.runtime_draft_len + 1].reshape(-1)
        return {"input_ids": input_ids, "position_ids": position_ids, "hidden_states": hidden}

    def _m3_renorm_raw_captures(
        self, hs: torch.Tensor, spec_metadata, draft_model: M3Eagle3Draft
    ) -> torch.Tensor:
        """The captured rows ``hs`` [rows, layers * H] for a reader that skips the fc_norm (the capture norm is armed):
        unchanged when every captured layer was written with this step's norm (capture_hidden_states records them),
        else -- not expected on any path -- a copy with each layer captured raw normalized by its own fc_norm here,
        logged as a miss, so a raw row never reaches the drafter's fc."""
        missing = [
            i
            for i, layer_id in enumerate(spec_metadata.layers_to_capture)
            if layer_id not in self._m3_capture_normed
        ]
        if not missing:
            return hs
        missed(
            "m3_capture_norm",
            int(hs.shape[0]),
            f"captured layer(s) {[spec_metadata.layers_to_capture[i] for i in missing]} were written without the "
            "capture norm; their fc_norm is applied before the drafter's fc",
        )
        hs = hs.clone()
        hidden = spec_metadata.hidden_size
        for i in missing:
            block = hs[:, i * hidden : (i + 1) * hidden]
            block.copy_(
                flashinfer_rmsnorm(
                    block.contiguous(), draft_model.w[f"fc_norm{i}"], draft_model.rms_eps
                )
            )
        return hs

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
            else:
                hs = self._m3_renorm_raw_captures(hs, spec_metadata, draft_model)
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
        # The captured layers this step wrote with its capture norm (capture_hidden_states).
        self._m3_capture_normed = set()
        # A fork result no first draft pass consumed must not reach a later step's (same-size) first inputs.
        self._m3_fc_hidden = None
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
        kv_lens: Optional[torch.Tensor] = None,
    ):
        """Fused verification (see tail_eligible); same outputs as the strict acceptance path. ``kv_lens``: the
        requests' KV lengths at this verify (the per-request forced-acceptance key, ``FORCE_AL_PER_REQUEST``)."""
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
        head = self._m3_coded_head(lm_head, rows.shape[0])
        # The Eagle3 fc on the aux stream: the fork made at the last captured layer, else one beside the head. A
        # sampled verify's coded head holds every SM while it streams: its fc is forked after it instead (overlapping
        # the sampled acceptance), where it does not delay the head's start.
        fork = self._m3_fc_early_stream
        self._m3_fc_early_stream = None
        head_first = (
            fork is None
            and (head is not None or rows.shape[0] <= FC_AFTER_HEAD_MAX_ROWS)
            and not spec_metadata.is_all_greedy_sample
        )
        if fork is None and not head_first:
            fork = self._m3_fc_fork(spec_metadata, rows.shape[0], draft_model)
        # This rank's vocabulary shard of the target logits: the head's own local GEMM (or its coded copy), whose
        # all-gather the fused kernels replace (they exchange per-row records across ranks, not logits).
        if head is not None:
            taken(TARGET_HEAD.path, rows.shape[0])
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
                one_round_k=SAMPLED_ACCEPT_1R_K if SAMPLED_ACCEPT_SM107TP4 else 0,
                prewait_params=SAMPLED_ACCEPT_PREWAIT_PARAMS,
                early_trigger=SAMPLED_ACCEPT_EARLY_TRIGGER,
                ilp_walk=SAMPLED_ACCEPT_ILP_WALK,
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
        taken(
            SHARD_ACCEPT.path if spec_metadata.is_all_greedy_sample else "m3_sampled_accept",
            batch_size,
            f"draft length {draft_len}"
            + (
                ""
                if spec_metadata.is_all_greedy_sample
                else f", SA-1R one_round_k {SAMPLED_ACCEPT_1R_K if SAMPLED_ACCEPT_SM107TP4 else 0}"
            ),
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
        self,
        batch_size: int,
        base_total: int,
        max_total: int,
        device,
        kv_lens: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Per-request forced accepted counts for a fractional forced acceptance, else None.

        ``FORCE_AL_PER_REQUEST`` (with ``kv_lens``): each request draws from the fixed-seed pool at its own key, its
        KV length at this verify (its prompt plus what it has accepted so far), so its draws do not depend on the
        server's history or on the other requests; every TP rank holds the same lengths, so the ranks agree. Without
        lengths: exactly as _apply_force_accepted_tokens draws for a decode-only batch (one shared counter bump per
        step, the (counter, slot) hash), the generic path's sequence.
        """
        frac = self.force_num_accepted_tokens - int(self.force_num_accepted_tokens)
        if frac <= 0.0 or base_total >= max_total:
            return None
        self._ensure_force_accept_rng_state(device)
        out = torch.empty(batch_size, dtype=torch.int32, device=device)
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
                max(16, 1 << (batch_size - 1).bit_length()),
            )
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

    def _m3_draft_ar_norm(
        self,
        draft_model: M3Eagle3Draft,
        x: torch.Tensor,
        residual: torch.Tensor,
        norm_key: str,
        pushed: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One drafter boundary of the fused draft loop: the multicast all-reduce + residual + RMSNorm in its drafter
        form (``DRAFT_MC_AR``; bitwise the stock boundary), else the built-in path (``M3Eagle3Draft.ar_norm``). The
        choice depends on shapes and a collectively created workspace only, so the TP ranks agree. ``pushed``: the
        projection already pushed its rows into the all-reduce (``DRAFT_AR_PUSH``)."""
        rows = x.shape[0]
        path = DRAFT_MC_AR.path
        if DRAFT_MC_AR.covers(rows):
            workspace = m3_mc_allreduce.get_workspace(self.mapping, x.shape[-1])
            out = None
            if workspace is not None:
                out = m3_mc_allreduce.m3_mc_allreduce_norm(
                    x.contiguous(),
                    residual.contiguous(),
                    draft_model.w[norm_key],
                    draft_model.rms_eps,
                    workspace,
                    use_pdl=get_env_enable_pdl(),
                    drafter_norm=True,
                    carveout=DRAFT_MC_AR_CARVEOUT,
                    prepushed=2 if pushed else 0,
                    drafter_cluster=DRAFT_MC_AR_CLUSTER,
                )
            if out is not None:
                taken(
                    path,
                    rows,
                    "drafter boundary (cluster kernel)"
                    if DRAFT_MC_AR_CLUSTER
                    else "drafter boundary (row kernel)",
                )
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

    def _m3_arm_draft_push(self, weight: torch.Tensor, rows: int) -> None:
        """``DRAFT_AR_PUSH``: let the projection ``weight``'s next low-M split-K call push its rows into the boundary
        all-reduce that ``_m3_draft_ar_norm`` then runs (the multicast path only; shapes and a collectively created
        workspace decide it, so the TP ranks agree)."""
        if not (DRAFT_AR_PUSH and DRAFT_MC_AR.covers(rows)):
            return
        workspace = m3_mc_allreduce.get_workspace(self.mapping, weight.shape[0])
        if workspace is not None:
            _draft_lowm_kernel.arm_push(weight, workspace, rows)

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
                taken(DRAFT_INPUT.path, x.shape[0])
                rows = x.shape[0]
                self._m3_arm_draft_push(w["o"], rows)
                h = draft_model.attention(x, md, reduce=False)
                h, residual = self._m3_draft_ar_norm(
                    draft_model,
                    h,
                    residual,
                    "post_norm",
                    pushed=_draft_lowm_kernel.take_pushed(h),
                )
                self._m3_arm_draft_push(w["dn"], rows)
                h = draft_model.mlp(h, reduce=False, silu_cpp=True)
                h, to_save = self._m3_draft_ar_norm(
                    draft_model,
                    h,
                    residual,
                    "final_norm",
                    pushed=_draft_lowm_kernel.take_pushed(h),
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
                taken(DRAFT_STEP.path, batch_size, f"draft length {draft_len}")
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
    the llama3 drafter architecture), or of one-model standalone DSpark (``dspark.build_dspark``: the in-tree drafter
    and worker)."""
    spec = model_config.spec_config
    if _dspark.is_standalone_dspark(spec):
        shell.model.embed_tokens = _dspark.M3EmbedTokens(shell.model)
        return _dspark.build_dspark(shell, model_config)
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


# TRTLLM_M31_DISABLE_HOSTTAIL=1 turns off the six items of the sm_103 targets' HOSTTAIL bundle together (HOST_D2H_SIDE,
# SMALL_H2D, BATCH_CHANGED_ONLY, SKIP_ACCEPTED_UPLOAD, PHILOX_PACK, KV_LENS_ONCE below), as there.
_HOSTTAIL_OFF = m31_disabled("HOSTTAIL")
# SKIP_ACCEPTED_UPLOAD (KIMI-HO-09, RuntimeHooks.skip_accepted_draft_upload): the step's per-request accepted-draft
# counts are no longer uploaded and spec_metadata.num_accepted_draft_tokens is None. Nothing on this target reads that
# tensor (modeling.py, dspark.py, dspark_s4.py, mxdraft.py, kernels/, ops/; the in-tree DFlash / DSpark workers and
# the Eagle3 path: tests/unittest/_torch/executor/test_skip_accepted_draft_upload.py scans for readers). Kill switch:
# TRTLLM_M31_DISABLE_SKIP_ACCEPTED_UPLOAD=1 (the stock per-step upload, packed under spec_small_h2d).
STEP_SKIP_ACCEPTED_UPLOAD = not (_HOSTTAIL_OFF or m31_disabled("SKIP_ACCEPTED_UPLOAD"))
# BATCH_CHANGED_ONLY (KIMI-HO-03 / ST-18(d), RuntimeHooks.spec_batch_map_changed_only): the standalone DSpark (DFlash)
# drafter's batch_indices and batch-to-slot map are uploaded only when they change (both constant in steady decode);
# the buffers hold the same values as with the stock every-step copies. Kill switch:
# TRTLLM_M31_DISABLE_BATCH_CHANGED_ONLY=1 (both copied every step).
STEP_BATCH_CHANGED_ONLY = not (_HOSTTAIL_OFF or m31_disabled("BATCH_CHANGED_ONLY"))
# The rest of the step loop's host-side items, each default on with its own kill switch; every one is exact (the same
# values reach the device, only the copies' stream, size, grouping or timing change):
# * HOST_D2H_SIDE (RuntimeHooks.sampler_d2h_side_stream): the spec sampler's store copies to the host on a side
#   stream. TRTLLM_M31_DISABLE_HOST_D2H_SIDE=1: the stock copies on the main stream (the spec fold's store_copies_done
#   event is then never re-recorded, so its tail wait is satisfied, and stream order keeps the copies ahead of it).
# * SMALL_H2D (spec_small_h2d): the sampler's slot ids stay in a device buffer re-uploaded only on change and the
#   accepted counts ride in the step-input pack. TRTLLM_M31_DISABLE_SMALL_H2D=1: the stock per-step uploads.
# * PHILOX_PACK (pack_spec_rng_uploads): sampled batches stage their Philox seed/offset uploads into the step-input
#   pack. TRTLLM_M31_DISABLE_PHILOX_PACK=1: the stock separate uploads.
# * KV_LENS_ONCE (kv_lens_update_once): a speculative overlap step re-derives the MSA KV-length metadata once, after the
#   KV-length correction. TRTLLM_M31_DISABLE_KV_LENS_ONCE=1: both passes.
# * PACK_PSCATTER (pack_parallel_scatter): the pack's non-overlapping copies scatter with a program per copy.
#   TRTLLM_M31_DISABLE_PACK_PSCATTER=1: the serial one-CTA scatter (and, as below, no in-graph pack).
# * PACK_IN_GRAPH (pack_in_graph; needs PACK_PSCATTER): the decode graph's prologue issues the pack (and, with it, the
#   step epoch). TRTLLM_M31_DISABLE_PACK_IN_GRAPH=1: the eager upload and scatter between the graphs.
# * PACK_EARLY_COPY (pack_early_copy; needs PACK_IN_GRAPH): the prologue's window is uploaded on a side copy stream
#   while the previous graph runs, and the prologue is this target's one launch (m3_step_prologue).
#   TRTLLM_M31_DISABLE_PACK_EARLY_COPY=1: the prologue copies the pinned window itself (scatter launches + fill).
# * H2D_CHANGED_SPAN (h2d_changed_span): a changed packed 2-D upload stages only each row's changed column span.
#   TRTLLM_M31_DISABLE_H2D_CHANGED_SPAN=1: the whole view.
STEP_HOST_D2H_SIDE = not (_HOSTTAIL_OFF or m31_disabled("HOST_D2H_SIDE"))
STEP_SMALL_H2D = not (_HOSTTAIL_OFF or m31_disabled("SMALL_H2D"))
STEP_PHILOX_PACK = not (_HOSTTAIL_OFF or m31_disabled("PHILOX_PACK"))
STEP_KV_LENS_ONCE = not (_HOSTTAIL_OFF or m31_disabled("KV_LENS_ONCE"))
STEP_PACK_PSCATTER = not m31_disabled("PACK_PSCATTER")
STEP_PACK_IN_GRAPH = STEP_PACK_PSCATTER and not m31_disabled("PACK_IN_GRAPH")
STEP_PACK_EARLY_COPY = STEP_PACK_IN_GRAPH and not m31_disabled("PACK_EARLY_COPY")
STEP_H2D_CHANGED_SPAN = not m31_disabled("H2D_CHANGED_SPAN")
# BLOCKOFF_GUARD (RuntimeHooks.skip_unchanged_draft_block_offsets): the standalone DSpark drafter's separate KV cache
# manager skips its per-step block-table copy (one launch between the step graphs, full max_seq_len width) when the
# copied rows' page indices are unchanged since its last copy into the same buffer; a skipped step leaves the bytes
# the copy would write. TRTLLM_M31_DISABLE_BLOCKOFF_GUARD=1 copies every step.
STEP_BLOCKOFF_GUARD = not m31_disabled("BLOCKOFF_GUARD")
# BLOCKOFF_PATCH (RuntimeHooks.patch_unchanged_draft_block_offsets; needs BLOCKOFF_GUARD): the drafter's sliding window
# changes one page index per row on about a third of the decode steps (a tail page allocated or re-allocated, a front
# page released); such a step rewrites the copied rows with the copy kernel's int32 transform done on the host and one
# host-to-device copy, instead of the full-width copy kernel. TRTLLM_M31_DISABLE_BLOCKOFF_PATCH=1 copies those steps.
STEP_BLOCKOFF_PATCH = STEP_BLOCKOFF_GUARD and not m31_disabled("BLOCKOFF_PATCH")
# SLOT_IDS_CHANGED (spec_slot_ids_changed_only; ported from the sm_103 targets): the spec
# metadata's per-request slot-id table (batch_slot_ids; rejection sampling) is re-uploaded only when the slot ids
# change, not copied every step. With BLOCKOFF_GUARD it empties the eager step-boundary stream the next decode graph
# waits on. OFF by default on this target: its token-identity gate failed here (259 / 272 requests identical, the
# rest diverging mid-sequence), so the skip misses a slot-id change on this step loop. Opt in with
# TRTLLM_M31_ENABLE_SLOT_IDS_CHANGED=1; TRTLLM_M31_DISABLE_SLOT_IDS_CHANGED=1 still wins. Read at import.
SLOT_IDS_CHANGED = m31_enabled("SLOT_IDS_CHANGED") and not m31_disabled("SLOT_IDS_CHANGED")


def _counted_pack_prologue(*args, **kwargs):
    """``m3_step_prologue`` (the decode graph's step-input pack prologue) with its gate-e counters: each call is the
    in-graph pack (PACK_IN_GRAPH) issued from the early-copied window (PACK_EARLY_COPY) by the parallel scatter
    (PACK_PSCATTER), the setting it is installed under."""
    out = m3_step_prologue(*args, **kwargs)
    m3_report.work("PACK_EARLY_COPY", "the C++ step-input prologue")
    m3_report.work("PACK_IN_GRAPH", "pack issued by the decode graph's prologue")
    m3_report.work("PACK_PSCATTER", "the prologue's parallel scatter")
    return out


def _runtime_hooks(spec_fold: Optional[M3SpecStepFold]) -> RuntimeHooks:
    """The step-loop hooks this target opts into (each behind its constant above).

    * ``pack_step_inputs``: a decode step's small input copies go out as one upload and one scatter at the graph
      replay instead of a chain of copies the next graph waits on (~30 us of idle GPU at one request).
    * ``skip_unchanged_h2d``: steady decode re-uploads slot indices, query lengths and page tables that rarely change.
    * ``loop_on_execution_stream``: the overlap loop's own stream work and the forward on one stream, so step N+1's
      inputs wait on graph N in program order rather than through two cross-stream event hops.
    * ``merge_tuned_tactics_across_ranks``: tensor-parallel peers capture the same GEMM tactics (rank-local profiling
      noise otherwise splits them, and the slowest rank sets every step).
    * ``skip_greedy_rng_upload``: all-greedy speculative batches never read the Philox seeds.
    * ``spec_step_fold``: the spec sampler's slot stores and the engine's next-step input gathers in one launch each.
    * ``pack_parallel_scatter`` / ``pack_in_graph`` / ``pack_early_copy``: the pack's copies scatter in parallel, issued
      by the decode graph's own prologue from a window uploaded ahead of the replay on a side copy stream.
    * ``pack_spec_rng_uploads`` / ``spec_small_h2d``: the spec step's seeds, accepted counts and slot ids ride the pack
      or stay on the device.
    * ``sampler_d2h_side_stream`` / ``iteration_timing_one_event``: the sampler's store copies and the iteration timing
      leave the main stream.
    * ``skip_accepted_draft_upload`` / ``spec_batch_map_changed_only`` (kill switches ``SKIP_ACCEPTED_UPLOAD`` /
      ``BATCH_CHANGED_ONLY``, ``STEP_SKIP_ACCEPTED_UPLOAD`` / ``STEP_BATCH_CHANGED_ONLY``): no per-step upload of the
      accepted-draft counts (no reader here), and the drafter's batch_indices / batch-to-slot map uploaded only on
      change.
    * MSA: identity selections for short sequences (and short/long graph families), the one-launch KV-length update,
      unchanged block-table copies skipped, pool views built once, block-id tables copied from the page-index views.
    """
    logger.info(f"[M3.1] SLOT_IDS_CHANGED (BP-3) {'on' if SLOT_IDS_CHANGED else 'off'}")
    # Gate e: the step-loop items' work is done by the shared runtime (pyexecutor) from these flags; this target logs
    # what it armed, per rank. Their work lines are the runtime's own ("[M3 host] ..."); the pack prologue below is
    # this target's op and counts its calls.
    armed = {
        "HOST_D2H_SIDE": STEP_HOST_D2H_SIDE,
        "SMALL_H2D": STEP_SMALL_H2D,
        "PHILOX_PACK": STEP_PHILOX_PACK,
        "KV_LENS_ONCE": STEP_KV_LENS_ONCE,
        "BATCH_CHANGED_ONLY": STEP_BATCH_CHANGED_ONLY,
        "SKIP_ACCEPTED_UPLOAD": STEP_SKIP_ACCEPTED_UPLOAD,
        "PACK_PSCATTER": STEP_PACK_PSCATTER,
        "PACK_IN_GRAPH": STEP_PACK_IN_GRAPH,
        "PACK_EARLY_COPY": STEP_PACK_EARLY_COPY,
        "H2D_CHANGED_SPAN": STEP_H2D_CHANGED_SPAN,
        "BLOCKOFF_GUARD": STEP_BLOCKOFF_GUARD,
        "BLOCKOFF_PATCH": STEP_BLOCKOFF_PATCH,
        "SLOT_IDS_CHANGED": SLOT_IDS_CHANGED,
    }
    logger.info(
        f"[M3 sm107 tp2] rank {mpi_rank()}: step-loop items armed (shared runtime does the work): "
        + " ".join(f"{k}={'on' if v else 'off'}" for k, v in armed.items())
    )
    return RuntimeHooks(
        fp8_saturating_casts=True,
        msa_kv_cache_fixes=True,
        draft_pool_fixes=True,
        persist_flashinfer_mxfp8_tactics=True,
        # Bundled name-keyed tactic pins (m31_pins); kill: TRTLLM_M31_DISABLE_BUNDLED_PINS=1.
        install_tactic_pins=lambda rank: m31_pins.install("tp2", rank),
        disagg_draft_kv=True,  # kill: TRTLLM_M31_DISABLE_DISAGG_DRAFT_KV=1
        first_content_emit_iters=FIRST_CONTENT_EMIT_ITERS,
        mxfp8_pinned_tactics=True,
        late_compile_detect=True,
        pack_step_inputs=STEP_PACK_INPUTS,
        pack_parallel_scatter=STEP_PACK_PSCATTER,
        pack_in_graph=STEP_PACK_IN_GRAPH,
        pack_early_copy=STEP_PACK_EARLY_COPY,
        pack_step_epoch=STEP_PACK_EPOCH,
        pack_prologue_op=_counted_pack_prologue if STEP_PACK_CPP_PROLOGUE else None,
        skip_unchanged_h2d=STEP_SKIP_UNCHANGED_H2D,
        h2d_changed_span=STEP_H2D_CHANGED_SPAN,
        kv_lens_update_once=STEP_KV_LENS_ONCE,
        spec_offsets_op=m3_spec_offsets if STEP_SPEC_OFFSETS_ONE_LAUNCH else None,
        loop_on_execution_stream=STEP_LOOP_ON_EXECUTION_STREAM,
        merge_tuned_tactics_across_ranks=STEP_MERGE_TUNED_TACTICS,
        skip_greedy_rng_upload=STEP_SKIP_GREEDY_RNG_UPLOAD,
        pack_spec_rng_uploads=STEP_PHILOX_PACK,
        spec_small_h2d=STEP_SMALL_H2D,
        skip_accepted_draft_upload=STEP_SKIP_ACCEPTED_UPLOAD,
        spec_batch_map_changed_only=STEP_BATCH_CHANGED_ONLY,
        spec_slot_ids_changed_only=SLOT_IDS_CHANGED,
        spec_step_fold=spec_fold,
        sampler_d2h_side_stream=STEP_HOST_D2H_SIDE,
        skip_unchanged_draft_block_offsets=STEP_BLOCKOFF_GUARD,
        patch_unchanged_draft_block_offsets=STEP_BLOCKOFF_PATCH,
        iteration_timing_one_event=STEP_ONE_TIMING_EVENT,
        msa=MsaRuntimePolicy(
            forward_staging=True,
            short_sequence_identity=MSA_SHORT_SEQUENCE_IDENTITY,
            kv_update=_msa_kv_update_kernel.msa_kv_update if MSA_KV_UPDATE else None,
            skip_unchanged_block_offsets=MSA_SKIP_UNCHANGED_BLOCK_OFFSETS,
            reuse_pool_views=MSA_REUSE_POOL_VIEWS,
            block_ids_from_views=MSA_BLOCK_IDS_FROM_VIEWS,
            prefill_proxy_qo_tile=128 if MSA_PROXY_Q128 else None,
            prefill_proxy_kv_splits=4 if MSA_PROXY_Q128 else None,
            prefill_proxy_kv_splits_by_rows=(
                (
                    (128, 64, 1),
                    (2048, 32, 20),
                    (4096, 16, 20),
                    (16384, 16 if m31_disabled("AGG_PROXY_SPLITCAP5") else 8, 40),
                )
                if MSA_PROXY_SPLITS
                else None
            ),
            prefill_proxy_oracle=MSA_PROXY_ORACLE,
        ),
    )


# TOPP_DET (ported from the sm_103 targets): the spec sampler's top-p renorm (FlashInfer
# top_p_renorm_probs in the no-top-k rejection path, the draft's and the verify's call each step) runs as one
# persistent C++ launch (ops/sampling/m31_topp_det, kernels/minimaxM31/toppRenormDetSm103.cu, built for sm_107
# in-tree or overlaid as tensorrt_llm/libs/libm31_topp_det.so) through the sampler's top-p renorm override, instead of
# FlashInfer's five launches with a one-CTA-per-row apply (AirTopPRenormApply: 63 us per call, 126 us per step at TP8
# K7 c1 on VR).
# Bitwise FlashInfer's deterministic renorm on rows with top_p < 1; rows with top_p >= 1 keep every token
# (a deliberate deviation: FlashInfer drops tail buckets there in ~43% of rows). Taken up to ``_topp_det.MAX_ROWS``
# rows; larger calls, inputs outside the contract and builds without the op keep the stock call. Kill switch:
# TRTLLM_M31_DISABLE_TOPP_DET=1 (stock FlashInfer top_p_renorm_probs).
TOPP_DET = not m31_disabled("TOPP_DET")
_TOPP_DET_STATE: dict = {}


def _install_topp_det() -> None:
    """Once per process: install ``_topp_det.top_p_renorm_det`` as the sampler's top-p renorm override when TOPP_DET
    is on and the op is built, and log the engagement (or why it is not taken) once per rank."""
    if _TOPP_DET_STATE:
        return
    if not TOPP_DET:
        state = "off (TRTLLM_M31_DISABLE_TOPP_DET=1); stock FlashInfer top_p_renorm_probs"
    elif _topp_det.topp_det_op() is None:
        state = "not taken: the op is not built; stock FlashInfer top_p_renorm_probs"
    else:
        set_top_p_renorm_override(_topp_det.top_p_renorm_det)
        ctas = _topp_det.ctas_per_row(_topp_det.MAX_ROWS, _topp_det.sm_count(torch.device("cuda")))
        state = f"engaged (ctas_per_row={ctas}, rows <= {_topp_det.MAX_ROWS})"
    _TOPP_DET_STATE["installed"] = state.startswith("engaged")
    logger.info(f"[M3 dispatch] ST-01 TOPP_DET det top-p renorm: {state}")


def _prebuild_topp_det() -> None:
    """The override's persistent workspace on this rank's device, allocated before warmup and any graph capture."""
    if _TOPP_DET_STATE.get("installed"):
        _topp_det.prebuild()


# SOFTMAX_MC (ported from the sm_103 targets): the spec sampler's softmax (FlashInfer
# softmax(logits, temperature) in softmax_op and compute_probs_from_logits_op, the draft's and the verify's call each
# step) keeps FlashInfer's Map kernel and runs its Reduce kernel's write pass on ``_softmax_mc.WRITE_CTAS`` CTAs per row
# (ops/sampling/m31_softmax_mc, kernels/minimaxM31/softmaxMcSm103.cu, built for sm_107 in-tree or overlaid as
# tensorrt_llm/libs/libm31_softmax_mc.so) through the sampler's softmax override, instead of one 1024-thread CTA per
# row. Bitwise FlashInfer's softmax on its Path A shapes (rows <= 128, vocab >= 24576; GB300 microbench at
# write CTAs 4 / 8 / 16). GB300 per call at 7-8 rows: 9.2 vs 21.8 us. Taken up to ``_softmax_mc.MAX_ROWS`` rows; larger
# calls at 129..448 rows use the separate Path-B operation when built and enabled.
# Inputs outside both contracts keep the stock call. Kill switch:
# TRTLLM_M31_DISABLE_SOFTMAX_MC=1 (stock FlashInfer softmax).
SOFTMAX_MC = not m31_disabled("SOFTMAX_MC")
_SOFTMAX_MC_STATE: dict = {}


def _install_softmax_mc() -> None:
    """Once per process: install ``_softmax_mc.softmax_mc`` as the sampler's softmax override when SOFTMAX_MC is on
    and the op is built, and log the engagement (or why it is not taken) once per rank."""
    if _SOFTMAX_MC_STATE:
        return
    if not SOFTMAX_MC:
        state = "off (TRTLLM_M31_DISABLE_SOFTMAX_MC=1); stock FlashInfer softmax"
    elif _softmax_mc.softmax_mc_op() is None:
        state = "not taken: the op is not built; stock FlashInfer softmax"
    else:
        set_softmax_override(_softmax_mc.softmax_mc)
        rows = _softmax_mc.max_rows()
        state = f"engaged (write_ctas={_softmax_mc.write_ctas(1)}..{_softmax_mc.write_ctas(rows)}, rows <= {rows})"
    _SOFTMAX_MC_STATE["installed"] = state.startswith("engaged")
    logger.info(f"[M3 dispatch] ST-01-SM softmax multi-CTA write: {state}")


def _prebuild_softmax_mc() -> None:
    """The override's persistent partials workspace on this rank's device, allocated before warmup and any graph
    capture."""
    if _SOFTMAX_MC_STATE.get("installed"):
        _softmax_mc.prebuild()


# REJ_MC (ported from the sm_103 targets): the spec sampler's rejection step
# (trtllm::chain_speculative_sampling in speculative/interface.rejection_sampling_one_model, once per verify step) runs
# on ctas_per_row CTAs per request (ops/sampling/m31_chain_spec_mc, kernels/minimaxM31/chainSpecSamplingMcSm103.cu,
# built for sm_107 in-tree or overlaid as tensorrt_llm/libs/libm31_chain_spec_mc.so) through the rejection sampler's
# override, instead of one 1024-thread CTA per request. ctas_per_row is sized from this device's SM count (the CTAs
# of a request meet at global-memory barriers, so batch x ctas_per_row stays co-resident). Bitwise the stock op's
# tokens and counts (GB300 microbench: K7, batch 1 / 2 / 4 / 8 x 5 cases x 64 seeds at ctas_per_row
# 4 / 8 / 16, eager and graph). GB300 per call at batch 1: 31.2 vs 41.2 us. Taken up to ``_chain_spec_mc.MAX_BATCH``
# requests; larger calls, inputs outside the contract and builds without the op keep the stock op. Kill switch:
# TRTLLM_M31_DISABLE_REJ_MC=1 (stock trtllm::chain_speculative_sampling).
REJ_MC = not m31_disabled("REJ_MC")
# REJ_SLOTS (VR only): on top of REJ_MC, the rejection step reads each request's draft
# distributions in place from the spec metadata's slot-indexed buffer (draft_probs[slot, k, :]) instead of the
# sampler first gathering them into a contiguous copy (one ~5.6 MB index_elementwise per step at c1). The same floats,
# so bitwise the gathered call. Enabled by default when the slot-indexed native op is available.
# TRTLLM_M31_DISABLE_REJ_SLOTS=1 restores the gather + REJ_MC path.
REJ_SLOTS = not m31_disabled("REJ_SLOTS")
_REJ_MC_STATE: dict = {}


def _install_rej_mc() -> None:
    """Once per process: install ``_chain_spec_mc.chain_spec_mc`` as the rejection sampler's override when REJ_MC is
    on and the op is built, and log the engagement (or why it is not taken) once per rank."""
    if _REJ_MC_STATE:
        return
    if not REJ_MC:
        state = "off (TRTLLM_M31_DISABLE_REJ_MC=1); stock trtllm::chain_speculative_sampling"
    elif _chain_spec_mc.chain_spec_mc_op() is None:
        state = "not taken: the op is not built; stock trtllm::chain_speculative_sampling"
    else:
        set_chain_speculative_sampling_override(_chain_spec_mc.chain_spec_mc)
        batch = _chain_spec_mc.max_batch()
        ctas = _chain_spec_mc.ctas_per_row(batch, _chain_spec_mc.sm_count(torch.device("cuda")))
        state = f"engaged (ctas_per_row={ctas} at batch {batch}, batch <= {batch})"
        slots = REJ_SLOTS and _chain_spec_mc.chain_spec_mc_slots_op() is not None
        if slots:
            set_chain_speculative_sampling_slot_override(_chain_spec_mc.chain_spec_mc_slots)
        state += f"; REJ_SLOTS slot-indexed draft rows {'on' if slots else 'off'}"
    _REJ_MC_STATE["installed"] = state.startswith("engaged")
    logger.info(f"[M3 dispatch] ST-01-REJ chain speculative sampling multi-CTA: {state}")


def _prebuild_rej_mc() -> None:
    """The override's persistent workspace on this rank's device, allocated before warmup and any graph capture."""
    if _REJ_MC_STATE.get("installed"):
        _chain_spec_mc.prebuild()


class ModelingV2MiniMaxM31Nvfp4Sm107Tp2(
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
        if (
            model_config.quant_config.kv_cache_quant_algo == QuantAlgo.NVFP4
            and _dspark.draft_kv_override_applies(spec_config)
        ):
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
        # A standalone DSpark drafter takes none of the Eagle3 drafter's hooks (full-context attention, low-M
        # linears, boundary capture-norm target).
        self._standalone_dspark = _dspark.is_standalone_dspark(spec_config)
        self.draft_model: Optional[nn.Module] = None
        self.draft_config: Optional[ModelConfig] = None
        self.spec_worker: Optional[Union[M3Eagle3Worker, _dspark.M3DSparkWorker]] = None
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
            # The step fold serves the Eagle3 worker's sampler slots; the DSpark worker keeps the in-tree ones.
            if STEP_SPEC_FOLD and not _dspark.is_standalone_dspark(spec_config):
                spec_fold = M3SpecStepFold(
                    in_forward=STEP_SPEC_FOLD_IN_FORWARD,
                    store_copy_wait=STEP_SPEC_STORE_COPY_SIDE,
                    cpp_tail=STEP_FOLD_TAIL_CPP,
                )
            elif DSFOLD and _dspark.is_standalone_dspark(spec_config):
                # The DSpark worker keeps the in-tree sampler slot stores; only the engine's input gathers fold.
                spec_fold = M3DSparkStepFold()
        self._spec_fold = spec_fold
        self.runtime_hooks = _runtime_hooks(spec_fold)
        _install_topp_det()
        _install_softmax_mc()
        _install_rej_mc()

    def load_weights(self, weights: Dict, *args, **kwargs) -> None:
        _weights.MODEL_WEIGHTS.load(self, weights)

    def load_draft_weights(self, weights: Dict, weight_mapper=None) -> None:
        """The drafter's checkpoint (``speculative_model``), loaded by the engine after the target's."""
        if _dspark.is_standalone_dspark(self.spec_config):
            _dspark.load_dspark_draft_weights(self, weights, weight_mapper)
            return
        _weights.DRAFT_WEIGHTS.load(self.draft_model, weights)

    def set_guided_decoder(self, guided_decoder) -> bool:
        """Mirrors ``SpecDecOneEngineForCausalLM.set_guided_decoder``: guided decoding with speculation is installed
        on the worker."""
        if hasattr(self.spec_worker, "set_guided_decoder"):
            return self.spec_worker.set_guided_decoder(guided_decoder)
        return False

    def cache_derived_state(self) -> None:
        """Build the targets and every decode kernel once the weights are real, before the executor's warmup; with
        speculation also the drafter layer's full-context decode attention (``DRAFTER_FULLCTX``), the fused verify's
        coded target head, the fused draft loop's head copy and the drafter's low-M GEMMs (``DRAFT_LOWM``)."""
        super().cache_derived_state()
        self.model.post_load_weights()
        _prebuild_topp_det()
        _prebuild_softmax_mc()
        _prebuild_rej_mc()
        draft = self.draft_model
        eagle3_draft = draft is not None and not self._standalone_dspark
        if eagle3_draft and DRAFTER_FULLCTX.keys:
            path = DRAFTER_FULLCTX.path
            installed = _drafter_attn_kernel.install(
                draft,
                {k: FULLCTX_DECODE_RECIPES[k] for k in DRAFTER_FULLCTX.keys},
                FULLCTX_DECODE_MAX_KV_LEN,
                FULLCTX_DECODE_OWN_REDUCER and _fullctx_decode_kernel.has_own_reducer(),
                get_env_enable_pdl(),
                fused_rope_mode=(
                    DRAFTER_ROPE_MODE
                    if DRAFTER_ROPE_FUSED
                    and hasattr(torch.ops.trtllm, "minimax_m3_sm107tp4_drafter_rope_fp8_kv")
                    else None
                ),
                on_rope_taken=lambda rows, what: taken(DRAFTER_ROPE_PATH, rows, what),
                on_taken=lambda rows: taken(path, rows),
                on_uncovered=lambda reason: uncovered(path, reason),
                on_missed=lambda rows, reason: missed(path, rows, reason),
            )
            logger.info(f"[M3 prebuild] {path} installed on {installed} drafter attention layer(s)")
        if self.draft_model is not None and self._standalone_dspark and VR_DRAFT_FMHA_SM100F:
            bypass = _drafter_attn_kernel.sliding_fi_bypass_reason(self.draft_model)
            if bypass is not None:
                logger.info(f"[M3 prebuild] VR_DRAFT_FMHA_SM100F: not installed: {bypass}")
            else:
                installed = _drafter_attn_kernel.install_sliding_fi(self.draft_model)
                logger.info(
                    f"[M3 prebuild] VR_DRAFT_FMHA_SM100F: FlashInfer trtllm-gen FMHA installed on {installed} "
                    "drafter attention layer(s)"
                )
        if self.spec_worker is not None:
            self.spec_worker.prebuild(self.lm_head, draft)
            if eagle3_draft:
                # The drafter linears' low-M kernels, before the warmup.
                start = time.perf_counter()
                built = draft.prebuild_linears()
                torch.cuda.synchronize()
                logger.info(
                    "[M3 prebuild] drafter low-M GEMMs built before warmup: %d calls, %.1f s"
                    % (built, time.perf_counter() - start)
                )

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
            capture_target=(
                worker.capture_target
                if worker is not None and not self._standalone_dspark
                else None
            ),
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
            # A worker may produce the gathered target logits itself (bitwise to the processor); None = the processor.
            target_logits = getattr(worker, "target_logits", None)
            logits = (
                target_logits(hidden_states, attn_metadata, spec_metadata, self.lm_head)
                if target_logits is not None
                else None
            )
            if logits is None:
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
            spec_fold.store_in_forward(outputs)
        return outputs
