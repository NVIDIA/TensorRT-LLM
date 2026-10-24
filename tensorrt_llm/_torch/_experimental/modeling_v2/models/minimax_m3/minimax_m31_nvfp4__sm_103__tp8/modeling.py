# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3.1 TP8 implementation."""

from __future__ import annotations

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
    Set,
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
    m31_fake_ops as _m31_fake_ops,
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
    require_msa_module,
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
    should_use_separate_draft_kv_cache,
)
from tensorrt_llm._torch.utils import ActType_TrtllmGen
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.functional import AllReduceStrategy, PositionEmbeddingType
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

from . import dspark as _dspark
from . import weights as _weights
from .m31_switches import m31_disabled, m31_enabled
from .ops import gqamix_sparse_decode as _gqamix_kernel
from .ops import m3_bp_tail, m3_mc_allreduce, m3_prequant, m3_spec_tail
from .ops import m3_draft_step as m3_draft_step_kernel
from .ops import m3_fused_score_select as _fused_score_select_kernel
from .ops import m3_index_score as _index_score_kernel
from .ops import m3_sampled_accept as m3_sampled_accept_kernel
from .ops import m3_select_long_dsl as _select_long_dsl_kernel
from .ops import m3_shard_accept as m3_shard_accept_kernel
from .ops import msa_kv_update as _msa_kv_update_kernel
from .ops.activation.m3_silu_and_mul import m3_silu_and_mul
from .ops.activation.m3_swq import m3_swq
from .ops.attention.m3_dense_decode import m3_dense_decode
from .ops.attention.m3_draft_attention import m3_draft_attention
from .ops.attention.m3_fp8_indexer_qk_norm_rope import m3_fp8_indexer_qk_norm_rope
from .ops.attention.m3_fused_score_select import (
    MAX_DECODE_QUERY_LEN as FUSED_SCORE_SELECT_MAX_QUERY_LEN,
)
from .ops.attention.m3_fused_score_select import (
    MAX_REQUESTS as FUSED_SCORE_SELECT_KERNEL_MAX_REQUESTS,
)
from .ops.attention.m3_fused_score_select import m3_fused_score_select
from .ops.attention.m3_index_score import SPLIT_K as INDEX_SCORE_SPLIT_K
from .ops.attention.m3_index_score import m3_index_score
from .ops.attention.m3_msa_attention import m3_msa_attention
from .ops.attention.m3_producer import m3_producer
from .ops.attention.m3_qk_norm_rope_fp8 import m3_qk_norm_rope_fp8
from .ops.attention.m3_qkv_insert import m3_qkv_insert
from .ops.attention.m3_select_blocks import m3_select_blocks
from .ops.attention.m3_select_long import MAX_BLOCKS as SELECT_LONG_MAX_BLOCKS
from .ops.attention.m3_select_long import m3_select_long
from .ops.attention.m3_select_long_dsl import m3_select_long_dsl
from .ops.attention.m3_sparse_decode import m3_sparse_decode
from .ops.attention.m3_sparse_decode_lc8 import m3_sparse_decode_lc8
from .ops.comm.m3_allreduce import m3_allreduce
from .ops.comm.m3_mc_allreduce_norm import m3_mc_allreduce_norm
from .ops.comm.m3_mc_finalize_allreduce_norm import m3_mc_finalize_allreduce_norm
from .ops.embedding.m3_embedding import m3_embedding
from .ops.gemm.m3_bf16_linear import m3_bf16_linear
from .ops.gemm.m3_draft_lowm import TACTICS as DRAFT_LOWM_TACTICS
from .ops.gemm.m3_draft_lowm import m3_draft_lowm, tactic_for
from .ops.gemm.m3_fp32_linear import m3_fp32_linear
from .ops.gemm.m3_mx_gemv import DENSE_DOWN, DENSE_GATE_UP, O_PROJ, QKV, QKV_SPARSE, m3_mx_gemv
from .ops.gemm.m3_mx_gemv_bf16 import m3_mx_gemv_bf16
from .ops.gemm.m3_mxfp8_gemm import m3_mxfp8_gemm
from .ops.gemm.m3_mxfp8_linear import m3_mxfp8_linear
from .ops.long_select import long_select_blocks
from .ops.m3_router_gemv import m3_router_gemv as _router_gemv
from .ops.m3_router_gemv import m3_router_gemv_fp4 as _router_gemv_fp4
from .ops.m3_router_gemv import m3_router_tc as _router_tc
from .ops.moe.m3_moe_engine import M3MoeState, ar_push_desc, m3_moe_engine
from .ops.moe.m3_router_gemv import m3_router_gemv
from .ops.moe.m3_router_gemv_fp4 import m3_router_gemv_fp4
from .ops.moe.m3_router_tc import m3_router_tc
from .ops.mx_gemv import op as mx_gemv
from .ops.norm.m3_embedding_firstnorm import DecodeEmbedding, m3_embedding_firstnorm
from .ops.sampling import m31_chain_spec_mc as _chain_spec_mc
from .ops.sampling import m31_softmax_mc as _softmax_mc
from .ops.sampling import m31_topp_det as _topp_det
from .ops.spec.m3_draft_input import m3_draft_input
from .ops.spec.m3_draft_step import MAX_TOKENS as DRAFT_STEP_KERNEL_MAX_TOKENS
from .ops.spec.m3_draft_step import m3_draft_step
from .ops.spec.m3_force_vec import m3_force_vec
from .ops.spec.m3_shard_accept import m3_shard_accept

_SM = (10, 3)
MSA_PROXY_PREBUILD = not m31_disabled("PROXYPB")
CTXNORMQ = not m31_disabled("CTXNORMQ")
CTXNORMQ_MIN_ROWS = 1024
FUSED_CAPTURE = not m31_disabled("FUSED_CAPTURE")
AR64AUTO = not m31_disabled("AR64AUTO")
TSEL = not m31_disabled("TSEL")
TSEL_MIN_ROWS = 256
TSEL_MAX_BLOCKS = 5856
TSEL_CHECK = os.environ.get("TRTLLM_M31_CHECK_TSEL") == "1"
GQAMIX = not m31_disabled("GQAMIX")
DSFOLD = not m31_disabled("DSFOLD")
BOUNDARY_PACK = DSFOLD and (not m31_disabled("BOUNDARY_PACK"))
BOUNDARY_PACK_MAX_ROWS = 1
_BOUNDARY_PACK_MARK = "m31_boundary_pack"
QKV_PRODUCER_FOLD = not m31_disabled("QKVPROD")
QKV_PRODUCER_FOLD_PATH = "m3_qkv_producer_fold"
_QKV_FOLD_ENGAGED: Set[int] = set()
DECODE_MAX_TOKENS = 48 if m31_disabled("R1") else 64
# Fused-layer dispatch scope; keep the larger prebuild/workspace capacity above.
FUSED_DECODE_MAX_TOKENS = 8
SWQ_SHARED_DOWN = (6144, 384)
SWQ_ROWS: Dict[Tuple[int, int], FrozenSet[int]] = (
    {} if m31_disabled("SWQ") else {SWQ_SHARED_DOWN: frozenset(range(9, 49))}
)
_SWQ_LOGGED: set = set()
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
        state = f"engaged (write_ctas={_softmax_mc.WRITE_CTAS}, rows <= {_softmax_mc.MAX_ROWS})"
    _SOFTMAX_MC_STATE["installed"] = state.startswith("engaged")
    logger.info(f"[M3 dispatch] ST-01-SM softmax multi-CTA write: {state}")


def _prebuild_softmax_mc() -> None:
    """The override's persistent partials workspace on this rank's device, allocated before warmup and any graph
    capture."""
    if _SOFTMAX_MC_STATE.get("installed"):
        _softmax_mc.prebuild()


REJ_MC = not m31_disabled("REJ_MC")
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
        ctas = _chain_spec_mc.ctas_per_row(
            _chain_spec_mc.MAX_BATCH, _chain_spec_mc.sm_count(torch.device("cuda"))
        )
        state = f"engaged (ctas_per_row={ctas}, batch <= {_chain_spec_mc.MAX_BATCH})"
    _REJ_MC_STATE["installed"] = state.startswith("engaged")
    logger.info(f"[M3 dispatch] ST-01-REJ chain speculative sampling multi-CTA: {state}")


def _prebuild_rej_mc() -> None:
    """The override's persistent workspace on this rank's device, allocated before warmup and any graph capture."""
    if _REJ_MC_STATE.get("installed"):
        _chain_spec_mc.prebuild()


_HOSTTAIL_OFF = m31_disabled("HOSTTAIL")
HOSTTAIL_D2H_SIDE = not (_HOSTTAIL_OFF or m31_disabled("HOST_D2H_SIDE"))
HOSTTAIL_SMALL_H2D = not (_HOSTTAIL_OFF or m31_disabled("SMALL_H2D"))
HOSTTAIL_BATCH_CHANGED_ONLY = not (_HOSTTAIL_OFF or m31_disabled("BATCH_CHANGED_ONLY"))
HOSTTAIL_SKIP_ACCEPTED_UPLOAD = not (_HOSTTAIL_OFF or m31_disabled("SKIP_ACCEPTED_UPLOAD"))
HOSTTAIL_PHILOX_PACK = not (_HOSTTAIL_OFF or m31_disabled("PHILOX_PACK"))
HOSTTAIL_KV_LENS_ONCE = not (_HOSTTAIL_OFF or m31_disabled("KV_LENS_ONCE"))
SLOT_IDS_CHANGED = not m31_disabled("SLOT_IDS_CHANGED")
PACK_PSCATTER = not m31_disabled("PACK_PSCATTER")
PACK_IN_GRAPH = PACK_PSCATTER and (not m31_disabled("PACK_IN_GRAPH"))
PACK_EARLY_COPY = PACK_IN_GRAPH and (not m31_disabled("PACK_EARLY_COPY"))
H2D_CHANGED_SPAN = not m31_disabled("H2D_CHANGED_SPAN")
BLOCKOFF_GUARD = not m31_disabled("BLOCKOFF_GUARD")
TAILBK = not m31_disabled("TAILBK")
_TAILBK_LOGGED: list = []


def _log_tailbk(worker) -> None:
    """The TAILBK engagement line, once per process when the model arms it: the items its worker can take (the DSpark
    worker all five; another worker's accept only A1). Nothing when TAILBK is off or without speculation."""
    if not TAILBK or worker is None or _TAILBK_LOGGED:
        return
    _TAILBK_LOGGED.append(True)
    items = "R1 B1 Z1 S1 A1" if isinstance(worker, _dspark.M3DSparkWorker) else "A1"
    logger.info(f"[M3 host] tail bookkeeping: {items}")


@dataclass(frozen=True)
class DecodePlan:
    """The layer-boundary collectives of one decode token bucket (GB300 measurements in the comments)."""

    name: str
    max_tokens: int
    attn_mc_ar: bool
    moe_mc_ar: bool
    qkv_prequant: bool


DECODE_PLANS = (
    DecodePlan("tiny", 8, attn_mc_ar=True, moe_mc_ar=True, qkv_prequant=True),
    DecodePlan("small", 16, attn_mc_ar=True, moe_mc_ar=True, qkv_prequant=True),
    DecodePlan("verify", DECODE_MAX_TOKENS, attn_mc_ar=True, moe_mc_ar=True, qkv_prequant=True),
)
DENSE_DECODE_MAX_SEQ_LEN_BY_BATCH: Mapping[int, int] = MappingProxyType({1: 16384})
DRAFT_DECODE_MAX_SEQ_LEN_BY_BATCH: Mapping[int, int] = MappingProxyType({1: 16384})
PRODUCER_EARLY_TRIGGER = True
PRODUCER_SHORTFIX = not m31_disabled("PRODUCER_EARLY_TRIGGER_SHORTFIX")
_PRODUCER_SHORTFIX_LOGGED: list = []


def _producer_early_trigger(step) -> bool:
    """Whether this step's sparse-layer producers release their dependents early (see PRODUCER_SHORTFIX)."""
    if not PRODUCER_EARLY_TRIGGER:
        return False
    no_selector = bool(step.get("msa_short_ok")) and step.get("msa_short_table") is not None
    no_selector = no_selector or False
    if not (PRODUCER_SHORTFIX and no_selector):
        return True
    if not _PRODUCER_SHORTFIX_LOGGED:
        _PRODUCER_SHORTFIX_LOGGED.append(True)
        logger.info(
            (
                "[M3 dispatch] PRODUCER_SHORTFIX: producer early trigger off on s"
                "teps without a selector (sparse decode loads q before its grid w"
                "ait)"
            )
        )
    return False


EMBED_FIRSTNORM_MAX_TOKENS = 8
ATTN_PUSH_MAX_TOKENS = 16
AR_PUSH_MODE = 2
MOE_ENGINE_MAX_TOKENS = 8
MOE_PUSH_MAX_TOKENS = MOE_ENGINE_MAX_TOKENS
MC_FINALIZE_MIN_TOKENS = MOE_ENGINE_MAX_TOKENS + 1
ROUTER_GEMV_MAX_TOKENS = 1
ROUTER_TC_MAX_TOKENS = 48
SHARD_ACCEPT_DRAFT_LEN = 3
SHARD_ACCEPT_MAX_REQUESTS = min(DECODE_MAX_TOKENS, m3_shard_accept_kernel.SHARD_ROWS) // (
    SHARD_ACCEPT_DRAFT_LEN + 1
)
FC_EARLY_MAX_ROWS = 0
DRAFT_STEP_DRAFT_LEN = SHARD_ACCEPT_DRAFT_LEN
DRAFT_STEP_MAX_REQUESTS = DRAFT_STEP_KERNEL_MAX_TOKENS
SPARSE_DECODE_MQ_MIN_TOKENS = 17
FUSED_SCORE_SELECT_MAX_REQUESTS = FUSED_SCORE_SELECT_KERNEL_MAX_REQUESTS
LONG_SELECT_MIN_BLOCKS = 129
LONG_SELECT_MAX_BLOCKS = SELECT_LONG_MAX_BLOCKS
LONG_SELECT_KEY_WIDTHS = (2, 4, 9)
SELECT_EARLY_MAX_ROWS = 8
SELECT_LONG_DSL = not m31_disabled("SELECT_DSL")
LONG_SELECT_DSL_KEY_WIDTHS = (4, 11, 24)
_SELECT_DSL_LOGGED: set = set()
LC8_PREWAIT_MAX_ROWS = 8
INDEX_SCORE_EARLY_MAX_REQUESTS = 0 if m31_disabled("INDEX_EARLY") else 1
_INDEX_EARLY_LOGGED: set = set()


def _index_score_early(rows: int, rank: int) -> bool:
    """Whether a decode step of ``rows`` requests runs the index scorer's early-issue schedule. The first time it does,
    logs the EARLY-ISSUE engagement line once per rank (host side: the graph capture runs this Python too)."""
    early = rows <= INDEX_SCORE_EARLY_MAX_REQUESTS
    if early and rank not in _INDEX_EARLY_LOGGED:
        _INDEX_EARLY_LOGGED.add(rank)
        logger.info(
            f"[M3 dispatch] INDEX_EARLY index-scorer early-issue: engaged (rows={rows}) on rank {rank}"
        )
    return early


_ALL_ROWS = frozenset(range(1, min(DECODE_MAX_TOKENS, mx_gemv.MAX_TOKENS) + 1))
MX_GEMV_ROWS: Dict[Tuple[int, int], FrozenSet[int]] = {
    QKV_SPARSE: _ALL_ROWS,
    QKV: _ALL_ROWS,
    O_PROJ: _ALL_ROWS,
    DENSE_GATE_UP: _ALL_ROWS,
    DENSE_DOWN: _ALL_ROWS,
}
MX_GEMV_BF16_MAX_TOKENS: Dict[Tuple[int, int], int] = {
    QKV_SPARSE: 16,
    QKV: 16,
    O_PROJ: 48,
    DENSE_GATE_UP: 16,
    DENSE_DOWN: 32,
}
DRAFT_LOWM_ROWS: Dict[Tuple[int, int], FrozenSet[int]] = {
    nk: frozenset(range(1, table[-1][0] + 1)) for nk, table in DRAFT_LOWM_TACTICS.items()
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
        return frozenset((k for cell in self.entry.CELLS for k in self.certifies(cell.spec)))


def _rows(lo: int, hi: int) -> FrozenSet[int]:
    return frozenset(range(lo, hi + 1))


def _spec_rows(field: str, **match: Any) -> Callable[[Mapping[str, Any]], Iterable[Key]]:
    """A cell's ``spec[field]`` when the cell's other fields match ``match``."""

    def certifies(spec: Mapping[str, Any]) -> Iterable[Key]:
        if any((spec.get(name) != value for name, value in match.items())):
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
        return tuple((w for w in LONG_SELECT_KEY_WIDTHS if w <= widest))
    return (widest,)


def _long_select_dsl_widths(spec: Mapping[str, Any]) -> Iterable[Key]:
    """The 384-thread select's key widths a cell's rows reach (every width up to its blocks' for mixed rows)."""
    blocks = spec["num_blocks"]
    widest = 4 if blocks <= 4 * 384 else 11 if blocks <= 11 * 384 else 24
    if spec["valid"] == "mixed":
        return tuple((w for w in LONG_SELECT_DSL_KEY_WIDTHS if w <= widest))
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
SPARSE_DECODE_LC8 = RowRange(
    "m3_sparse_decode_lc8",
    frozenset({"prewait", "plain"}),
    m3_sparse_decode_lc8,
    lambda spec: ("prewait" if spec["prewait"] else "plain",),
)
PRODUCER = RowRange("m3_producer", frozenset({"nvfp4_tp8"}), m3_producer, _const("nvfp4_tp8"))
FUSED_SCORE_SELECT = RowRange(
    "m3_fused_score_select",
    _rows(1, FUSED_SCORE_SELECT_MAX_REQUESTS),
    m3_fused_score_select,
    _spec_rows("batch"),
)
INDEX_SCORE = RowRange(
    "m3_index_score",
    frozenset({"prefetch", "early"}) if INDEX_SCORE_EARLY_MAX_REQUESTS else frozenset({"prefetch"}),
    m3_index_score,
    lambda spec: ("early",) if spec["early"] else ("prefetch",),
)
LONG_SELECT = RowRange(
    "long_select", frozenset(LONG_SELECT_KEY_WIDTHS), m3_select_long, _long_select_widths
)
LONG_SELECT_DSL = RowRange(
    "long_select_dsl",
    frozenset(LONG_SELECT_DSL_KEY_WIDTHS) if SELECT_LONG_DSL else frozenset(),
    m3_select_long_dsl,
    _long_select_dsl_widths,
)
MC_ALLREDUCE = RowRange(
    "mc_allreduce_norm",
    _rows(1, min(DECODE_MAX_TOKENS, m3_mc_allreduce.max_rows())),
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
    _rows(MC_FINALIZE_MIN_TOKENS, min(DECODE_MAX_TOKENS, m3_mc_allreduce.max_rows())),
    m3_mc_finalize_allreduce_norm,
    _spec_rows("m"),
)
MOE_ENGINE = RowRange(
    "m3_moe_engine", _rows(1, MOE_ENGINE_MAX_TOKENS), m3_moe_engine, _spec_rows("tokens")
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
MX_GEMV = RowRange(
    "m3_mx_gemv",
    frozenset(((nk, m) for nk, rows in MX_GEMV_ROWS.items() for m in rows)),
    m3_mx_gemv,
    lambda spec: ((spec["nk"], spec["m"]),),
)
MX_GEMV_BF16 = RowRange(
    "m3_mx_gemv_bf16",
    frozenset(((nk, m) for nk, hi in MX_GEMV_BF16_MAX_TOKENS.items() for m in _rows(1, hi))),
    m3_mx_gemv_bf16,
    lambda spec: ((spec["nk"], spec["m"]),),
)
DRAFT_LOWM = RowRange(
    "m3_draft_lowm",
    frozenset(((nk, m) for nk, rows in DRAFT_LOWM_ROWS.items() for m in rows)),
    m3_draft_lowm,
    lambda spec: ((spec["nk"], spec["m"]),),
)
SHARD_ACCEPT = RowRange(
    "m3_shard_accept",
    frozenset(((SHARD_ACCEPT_DRAFT_LEN, n) for n in range(1, SHARD_ACCEPT_MAX_REQUESTS + 1))),
    m3_shard_accept,
    lambda spec: ((spec["draft_len"], spec["nreq"]),),
)
DENSE_DECODE_SPLIT = RowRange(
    "m3_dense_decode",
    frozenset(DENSE_DECODE_MAX_SEQ_LEN_BY_BATCH),
    m3_dense_decode,
    lambda spec: (len(spec["kv_lens"]),) if spec["capped"] else (),
)
DRAFT_STEP = RowRange(
    "m3_draft_step",
    frozenset(((DRAFT_STEP_DRAFT_LEN, n) for n in range(1, DRAFT_STEP_MAX_REQUESTS + 1))),
    m3_draft_step,
    lambda spec: ((spec["draft_len"], spec["nreq"]),),
)
DRAFT_INPUT = RowRange(
    "m3_draft_input",
    frozenset(
        (
            rows
            for n in range(1, DRAFT_STEP_MAX_REQUESTS + 1)
            for rows in (n, n * (DRAFT_STEP_DRAFT_LEN + 1))
        )
    ),
    m3_draft_input,
    _spec_rows("m"),
)
FAST_PATHS: Tuple[RowRange, ...] = (
    EMBED_FIRSTNORM,
    DENSE_DECODE_SPLIT,
    SPARSE_DECODE_PRODUCER,
    SPARSE_DECODE_LC8,
    PRODUCER,
    FUSED_SCORE_SELECT,
    INDEX_SCORE,
    LONG_SELECT,
    LONG_SELECT_DSL,
    MC_ALLREDUCE,
    ATTN_PUSH,
    MOE_PUSH,
    MC_FINALIZE,
    MOE_ENGINE,
    ROUTER_GEMV,
    ROUTER_GEMV_FP4,
    ROUTER_TC,
    MX_GEMV,
    MX_GEMV_BF16,
    DRAFT_LOWM,
    SHARD_ACCEPT,
    DRAFT_STEP,
    DRAFT_INPUT,
)


def plan_for(num_tokens: int) -> DecodePlan:
    for plan in DECODE_PLANS:
        if num_tokens <= plan.max_tokens:
            return plan
    raise AssertionError(f"no decode plan covers {num_tokens} tokens")


def is_decode_step(num_tokens: int) -> bool:
    """Whether a pure generation step of ``num_tokens`` rows runs the decode kernels (else generic layers)."""
    return num_tokens <= min(FUSED_DECODE_MAX_TOKENS, DECODE_MAX_TOKENS)


def moe_path(num_tokens: int) -> Optional[str]:
    """The MoE kernel of a decode step: the engine, the unfinalized MoE + finalize all-reduce, or None (generic)."""
    if MOE_ENGINE.covers(num_tokens):
        return MOE_ENGINE.path
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
PREQUANT_GEMM_PATH = "flashinfer_mxfp8_gemm"
GENERIC_GEMM_PATH = "mxfp8_gemm"


def linear_path(n: int, k: int, num_tokens: int, prequant: bool) -> str:
    """The decode-graph MXFP8 GEMM of an [N, K] projection on ``num_tokens`` rows.

    ``prequant``: the producer published the activation in MXFP8.
    """
    if prequant:
        if num_tokens in MX_GEMV_ROWS.get((n, k), ()):
            return MX_GEMV_PATH
        return PREQUANT_GEMM_PATH
    if num_tokens <= MX_GEMV_BF16_MAX_TOKENS.get((n, k), 0):
        return MX_GEMV_BF16_PATH
    return GENERIC_GEMM_PATH


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
            f"modeling_v2='require': the MiniMax-M3 decode path {path!r} covers this step{at} but cannot run: {reason}"
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


def long_select_dsl_keys() -> Tuple[Tuple[bool, bool, int], ...]:
    """The (PDL launch, early trigger, trigger phase) keys of the 384-thread select a decode step can launch, compiled
    before warmup (trigger phase 0: released at its exit when not early, as the 1024-thread kernel); none when
    ``LONG_SELECT_DSL`` covers nothing."""
    if not LONG_SELECT_DSL.keys:
        return ()
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    pdl = get_env_enable_pdl()
    keys = [(pdl, False, 0)]
    if SELECT_EARLY_MAX_ROWS:
        keys.append((pdl, True, 0))
    return tuple(keys)


def _note_select_dsl(rank: int) -> None:
    """The SELECT_DSL engagement line, once per process and rank (host side: also during graph capture)."""
    if rank not in _SELECT_DSL_LOGGED:
        _SELECT_DSL_LOGGED.add(rank)
        logger.info(f"[M3 dispatch] SELECT_DSL 384-thread long select: engaged on rank {rank}")


def reset_logged() -> None:
    """Forget which (path, reason) pairs were logged."""
    with _lock:
        _logged.clear()


MC_AR_MAX_TOKENS = 8
MOE_FOLD_MIN_TOKENS = 1
MOE_ENGINE_RESERVED_SMS = 24
FINAL_HIDDEN_LAYER_ID = -1
ROUTING_MINIMAX2 = int(RoutingMethodType.MiniMax2)
ACT_SWIGLU = int(ActType_TrtllmGen.SwiGlu)


def _is_vl_config(pretrained_config: PretrainedConfig) -> bool:
    model_type = getattr(pretrained_config, "model_type", None)
    if model_type == "minimax_m3_vl":
        return True
    if model_type == "minimax_m3":
        return False
    return any(
        (
            "ForConditionalGeneration" in a
            for a in getattr(pretrained_config, "architectures", None) or []
        )
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

    decode_graph: bool
    autotune: bool
    tuned_quantize_gemm: bool
    flashinfer: bool
    native_tuning: bool


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


def _mxfp8_linear(
    core, x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, mode: MxMode
) -> torch.Tensor:
    """One MXFP8 projection on the engine-tuned path: MXFP8 quantize + GEMM, backend per ``mode``."""
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
                bucket = next((b for b in m3_mc_allreduce.BUCKETS if rows <= b))
                variants.append((rows, x_bf16, (world, bucket, AR_PUSH_MODE)))
    return variants


def _uncompiled_note() -> str:
    key = mx_gemv.last_uncompiled
    return "" if key is None else f" (its kernel {key} was not compiled before warmup)"


class _DeferredGemv:
    """QKVPROD (``QKV_PRODUCER_FOLD``): a sparse layer's QKV + index MXFP8 GEMV, recorded instead of launched.

    ``_linear(..., defer=True)`` returns one on the decode GEMV path of the 12-head packed projection at <= 8 rows with
    an MXFP8 hand-off. The producer call site either runs the GEMV with the producer in its epilogue
    (``_qkv_producer_fold``) or calls ``materialize`` (the plain GEMV) before anything reads the packed row;
    ``placeholder`` (unwritten, the packed shape) stands for that row until then and is never read."""

    def __init__(self, act, act_scale, weight, weight_scale, use_pdl):
        self.placeholder = torch.empty(
            act.shape[0], weight.shape[0], dtype=torch.bfloat16, device=act.device
        )
        self.act, self.act_scale = (act, act_scale)
        self.weight, self.weight_scale = (weight, weight_scale)
        self.use_pdl = use_pdl
        self.done = False

    def materialize(self) -> torch.Tensor:
        """Launch the plain GEMV this record stands for; returns the packed row the projection would have returned."""
        self.done = True
        rows, (n, k) = (self.act.shape[0], self.weight.shape)
        output = m3_mx_gemv(
            self.act, self.act_scale, self.weight, self.weight_scale, True, None, self.use_pdl, None
        )
        if output is None:
            raise RuntimeError(
                f"the deferred [{n}, {k}] GEMV declined at {rows} rows" + _uncompiled_note()
            )
        return output


def _linear(
    core, x: torch.Tensor, wk: str, sk: str, push: Optional[tuple] = None, defer: bool = False
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
    ``m3_mx_gemv`` ``ar`` tuple) is the boundary all-reduce exchange the GEMV writes its rows into.
    """
    w, s = (core.w[wk], core.w[sk])
    x = x.contiguous()
    mode = _mx_mode(core.mxfp8)
    if not mode.decode_graph:
        prequant = _CTXNORMQ_MX.take(x) if CTXNORMQ and (not mode.tuned_quantize_gemm) else None
        if prequant is not None:
            return (_ctxnormq_mxfp8_gemm(core, prequant, w, s, mode), False)
        return (_mxfp8_linear(core, x, w, s, mode), False)
    use_pdl = get_env_enable_pdl()
    rows = x.shape[0]
    n, k = w.shape
    prequant = m3_prequant.SLOT.take(x)
    if prequant is None and x.dtype != torch.bfloat16:
        path = GENERIC_GEMM_PATH
    else:
        path = linear_path(n, k, rows, prequant is not None)
    output = None
    if prequant is not None:
        act, act_scale = prequant
        if (
            defer
            and path == MX_GEMV_PATH
            and (n == mx_gemv.PROD_N)
            and (rows <= 8)
            and (push is None)
        ):
            return (_DeferredGemv(act, act_scale, w, s, use_pdl), False)
        if path == MX_GEMV_PATH:
            output = m3_mx_gemv(act, act_scale, w, s, True, None, use_pdl, push)
            if output is not None:
                return (output, push is not None and mx_gemv.last_pushed)
            missed(
                path,
                rows,
                f"the GEMV declined the [{n}, {k}] projection's MXFP8 activation"
                + _uncompiled_note(),
            )
        if path == PREQUANT_GEMM_PATH:
            uncovered(MX_GEMV_PATH, f"the [{n}, {k}] projection at rows the GEMV does not serve")
        return (m3_mxfp8_gemm(act, act_scale, w, s, torch.bfloat16, mode.autotune), False)
    if path == MX_GEMV_BF16_PATH:
        output = m3_mx_gemv_bf16(x, w, s, True, None, use_pdl, push)
        if output is not None:
            return (output, push is not None and mx_gemv.last_pushed)
        missed(
            path, rows, f"the GEMV declined the [{n}, {k}] projection's input" + _uncompiled_note()
        )
    if path == GENERIC_GEMM_PATH:
        uncovered(
            MX_GEMV_BF16_PATH,
            f"the [{n}, {k}] projection's bf16 input beyond the decode GEMV's rows",
        )
    return (_mxfp8_linear(core, x, w, s, mode), False)


def _can_mc(x, folded: bool = False) -> bool:
    """Whether the generic boundary takes the multicast all-reduce + residual + RMSNorm."""
    return (
        isinstance(x, torch.Tensor)
        and x.dim() == 2
        and (x.shape[0] <= (m3_mc_allreduce.max_rows() if folded else MC_AR_MAX_TOKENS))
        and (x.dtype == torch.bfloat16)
        and x.is_contiguous()
    )


def _mc_allreduce_norm(core, x, x2, residual, norm_w, *, quant=False, prepushed=0):
    """The multicast all-reduce + optional second addend + residual + RMSNorm (``m3_mc_allreduce_norm``), or None
    without a workspace or for operands the kernel does not take (the same on every rank). With ``quant`` the normed
    rows are also published, MXFP8-quantized, to ``m3_prequant.SLOT`` for the next MXFP8 projection."""
    workspace = m3_mc_allreduce.get_workspace(core.mapping, x.shape[-1])
    if workspace is None:
        return None
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


_CTXNORMQ_MX = m3_prequant.PrequantSlot()
_CTXNORMQ_FP4 = m3_prequant.PrequantSlot()
_CTXNORMQ_STATE = {}


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
                _m31_fake_ops.register()
                op = getattr(torch.ops.trtllm, "minimax_m31_ctx_residual_rmsnorm_mxfp8", None)
        _CTXNORMQ_STATE["op"] = op
        logger.info(
            (
                "[M3 dispatch] CTXNORMQ prefill boundary norm + quantize: "
                f"{('engaged' if op is not None else 'not taken: the op is not built')}"
                " (rows >= "
                f"{CTXNORMQ_MIN_ROWS}"
                ")"
            )
        )
    return _CTXNORMQ_STATE["op"]


def _ctxnormq_allreduce_norm(core, x, residual, norm_w, fp4_scale):
    """A prefill boundary as the plain all-reduce + the fused residual add, RMSNorm and quantize, or None (switched
    off, fewer than ``CTXNORMQ_MIN_ROWS`` rows, op not built). Returns (normed, updated) like the fused all-reduce;
    the MXFP8 activation (and with ``fp4_scale`` the NVFP4 one) wait in the CTXNORMQ slots for their consumers."""
    if not CTXNORMQ or x.dim() != 2 or x.shape[0] < CTXNORMQ_MIN_ROWS:
        return None
    op = _ctxnormq_op()
    if op is None:
        return None
    reduced = m3_allreduce(x, core.allreduce, None)
    normed, xq, xsf, xfp4, xfp4_sf = op(
        reduced, residual.contiguous(), norm_w, core.rms_eps, True, fp4_scale, get_env_enable_pdl()
    )
    _CTXNORMQ_MX.publish(normed, xq, xsf)
    if fp4_scale is not None:
        _CTXNORMQ_FP4.publish(normed, xfp4, xfp4_sf)
    else:
        _CTXNORMQ_FP4.clear()
    return (normed, reduced)


def _ctxnormq_mxfp8_gemm(core, prequant, w, s, mode):
    """The GEMM ``m3_mxfp8_linear`` runs after its own quantize, on the CTXNORMQ activation (bit-identical operands)."""
    act, act_scale = prequant
    if mode.flashinfer:
        return torch.ops.trtllm.flashinfer_mm_mxfp8(act, act_scale, w, s, torch.bfloat16)
    gemm = (
        torch.ops.trtllm.mxfp8_mxfp8_gemm_autotuned
        if mode.native_tuning
        else torch.ops.trtllm.mxfp8_mxfp8_gemm
    )
    return gemm(act, act_scale, w, s, core.mx_alpha, torch.bfloat16)


_TSEL_STATE = {}


def _tsel_op():
    """``torch.ops.trtllm.minimax_m31_select_prefill_tile`` on sm_103, or None. A build without the op in its library
    loads it from ``tensorrt_llm/libs/libm31_tsel.so`` when that file exists (an overlay). Decided once per process."""
    if "op" not in _TSEL_STATE:
        op = None
        if torch.cuda.get_device_capability() == (10, 3):
            op = getattr(torch.ops.trtllm, "minimax_m31_select_prefill_tile", None)
            if op is None:
                import tensorrt_llm

                lib = os.path.join(os.path.dirname(tensorrt_llm.__file__), "libs", "libm31_tsel.so")
                if os.path.exists(lib):
                    torch.ops.load_library(lib)
                    _m31_fake_ops.register()
                    op = getattr(torch.ops.trtllm, "minimax_m31_select_prefill_tile", None)
        _TSEL_STATE["op"] = op
        logger.info(
            (
                "[M3 dispatch] TSEL prefill tile select: "
                f"{('engaged' if op is not None else 'not taken: not sm_103 or the op is not built')}"
                " (rows >= "
                f"{TSEL_MIN_ROWS}"
                ", blocks <= "
                f"{TSEL_MAX_BLOCKS}"
                ")"
            )
        )
    return _TSEL_STATE["op"]


def _tsel_select(max_score_kv, n_valid_blocks, init_blocks, local_blocks):
    """The MSA indexer's selector hook: the head-major [total_q, num_kv_heads, 16] table of
    ``select_blocks_from_maxscore``, or None (outside the tile select's rows / blocks range, or no op) to keep it."""
    if (
        torch.cuda.is_current_stream_capturing()
        or max_score_kv.dtype != torch.float32
        or max_score_kv.dim() != 3
        or (max_score_kv.stride(2) != 1)
        or (max_score_kv.shape[2] < TSEL_MIN_ROWS)
        or (max_score_kv.shape[1] > TSEL_MAX_BLOCKS)
    ):
        return None
    op = _tsel_op()
    if op is None:
        return None
    num_kv_heads, num_blocks, total_q = max_score_kv.shape
    out = torch.empty(
        (num_kv_heads, total_q, MSA_REQUIRED_TOPK), dtype=torch.int32, device=max_score_kv.device
    ).permute(1, 0, 2)
    nvb = n_valid_blocks.to(device=max_score_kv.device, dtype=torch.int32).contiguous()
    op(max_score_kv, nvb, out, init_blocks, local_blocks)
    if TSEL_CHECK:
        ref = torch.ops.trtllm.minimax_m3_select_blocks(
            max_score_kv, nvb, MSA_REQUIRED_TOPK, init_blocks, local_blocks, True
        )
        if not torch.equal(out, ref):
            raise AssertionError(
                (
                    "TSEL check: tile table mismatch against minimax_m3_select_blo"
                    "cks ("
                    f"{int((out != ref).any(dim=-1).sum())}"
                    " rows; "
                    f"{num_blocks}"
                    " blocks, "
                    f"{total_q}"
                    " rows)"
                )
            )
        _TSEL_STATE["checked"] = _TSEL_STATE.get("checked", 0) + 1
        if _TSEL_STATE["checked"] in (1, 100, 1000):
            logger.info(
                f"[M3 dispatch] TSEL check: {_TSEL_STATE['checked']} tile-selected steps equal"
            )
    return out


def _tsel_register(backend) -> None:
    """Register ``_tsel_select`` on the layer backend's MSA indexer (an indexer without the hook is left alone)."""
    indexer = getattr(backend, "indexer", None)
    if indexer is not None and getattr(indexer, "selector_hook", False) is None:
        indexer.selector_hook = _tsel_select


_GQAMIX_STATE: dict = {}


def _gqamix_decode(*args, **kwargs) -> None:
    """``minimax_m3_sparse_attn_decode`` for MsaDecodeFmha's sparse generation rows, through the target-local copy."""
    if not _GQAMIX_STATE.get("logged"):
        _GQAMIX_STATE["logged"] = True
        logger.info(
            "[M3 dispatch] GQAMIX: mixed-step sparse decode through ops/gqamix_sparse_decode"
        )
    _gqamix_kernel.minimax_m3_sparse_attn_decode(*args, **kwargs)


def _gqamix_register(backend) -> None:
    """Register ``_gqamix_decode`` on the layer backend's MsaDecodeFmha (a library without the hook is left alone)."""
    manager = getattr(backend, "_fmha_manager", None)
    for fmha in getattr(manager, "fmha_libs", ()):
        if getattr(fmha, "sparse_decode_hook", False) is None:
            fmha.sparse_decode_hook = _gqamix_decode


def _native_allreduce_norm(core, x, residual, norm_w):
    """The built-in fused all-reduce + residual + RMSNorm: with AR64AUTO, the strategy-MNNVL instance for the rows past
    the multicast all-reduce's, up to the decode bound; the configured instance otherwise."""
    params = AllReduceParams(
        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
        residual=residual,
        norm_weight=norm_w,
        eps=core.rms_eps,
        trigger_completion_at_end=False,
    )
    wide = getattr(core, "allreduce_wide", None)
    rows = x.shape[0]
    if wide is not None and m3_mc_allreduce.max_rows() < rows <= DECODE_MAX_TOKENS:
        out = m3_allreduce(x.contiguous(), wide, params)
        if out is not None:
            _log_once(
                logger.info,
                "ar64auto",
                "taken",
                f"[M3.1] AR64AUTO taken: MNNVL all-reduce at {rows} rows",
            )
            return out
        _log_once(
            logger.warning,
            "ar64auto",
            f"declined {rows}",
            f"[M3.1] AR64AUTO declined: MNNVL refused {rows} rows; the configured all-reduce runs",
        )
    return m3_allreduce(x, core.allreduce, params)


@dataclass
class _MoeScalars:
    """The routed experts' kernel scalars of one layer, derived from the checkpoint's per-tensor scales once the
    weights are real (``MiniMaxM3Tp8ModelingV2._derive_moe``)."""

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


def _nvfp4_layer(manager, i: int) -> bool:
    return bool(getattr(manager, "is_nvfp4_layer", lambda _layer_idx: False)(i))


def _producer_covered(core) -> bool:
    """Whether ``PRODUCER`` covers this target's sparse layers: an NVFP4 KV cache at the TP8 heads (8 query, one KV
    and one index head). A covered layer that cannot take ``m3_producer`` is a miss."""
    return core.main_kv_is_nvfp4 and (core.heads, core.kv_heads, core.index_heads) == (8, 1, 1)


def _producer_layout(kv_data, kv_scale) -> bool:
    """Whether a sparse layer's NVFP4 pools are the layout ``m3_producer`` is built for: [pages, 2, 1, 128, 64] with
    its scale pool [pages, 2, 1, 128, 8]."""
    return (
        kv_data is not None
        and kv_scale is not None
        and (kv_data.dim() == 5)
        and (tuple(kv_data.shape[1:]) == (2, 1, 128, 64))
        and (kv_data.stride()[2:] == (128 * 64, 64, 1))
        and (kv_scale.dim() == 5)
        and (tuple(kv_scale.shape[1:]) == (2, 1, 128, 8))
        and (kv_scale.stride()[2:] == (128 * 8, 8, 1))
        and (kv_scale.shape[0] == kv_data.shape[0])
    )


def _nvfp4_fused_write_layout(data: torch.Tensor, scales: torch.Tensor, num_heads: int) -> bool:
    """Reject invalid HND pools; select the fused producer only for its packed layout."""
    if (
        data.dim() != 5
        or scales.dim() != 5
        or data.shape[0] <= 0
        or (data.shape[0] != scales.shape[0])
        or (tuple(data.shape[1:]) != (2, num_heads, 128, 64))
        or (tuple(scales.shape[1:]) != (2, num_heads, 128, 8))
        or (data.dtype not in (torch.int8, torch.uint8))
        or (scales.dtype != torch.uint8)
    ):
        raise ValueError("MiniMax-M3 NVFP4 data and scale caches must have matching HND geometry")
    for cache in (data, scales):
        if (
            cache.stride(4) != 1
            or cache.stride(3) != cache.shape[4]
            or cache.stride(2) < 128 * cache.stride(3)
            or (cache.stride(1) < num_heads * cache.stride(2))
            or (cache.stride(0) < 2 * cache.stride(1))
        ):
            raise ValueError(
                "MiniMax-M3 NVFP4 cache rows must be packed and heads/planes/pages must not overlap"
            )
    if scales.stride(2) != 128 * 8:
        raise ValueError("MiniMax-M3 NVFP4 MSA scale cache must have packed head blocks")
    return data.stride(2) == 128 * 64 and data.stride(0) % 2 == 0 and (data.stride(1) % 2 == 0)


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
    early_trigger: bool = PRODUCER_EARLY_TRIGGER,
):
    """The horizontal producer of a sparse layer: Q / index-Q (FP8) out, K / V / index-K written into the paged
    caches (NVFP4 or FP8 main cache), for every step type (token-major). An NVFP4 cache at the TP8 heads takes
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
            and (scale_buffers.device == buffers.device)
            and kv_quant_scale.is_cuda
            and (kv_quant_scale.dtype == torch.float32)
            and (kv_quant_scale.numel() >= 3)
            and kv_quant_scale.is_contiguous()
        )
    else:
        supported_main = (
            buffers is not None
            and buffers.is_cuda
            and (buffers.dtype == torch.float8_e4m3fn)
            and (buffers.dim() == 5)
            and (tuple(buffers.shape[1:]) == (2, core.kv_heads, 128, 128))
            and (buffers.stride(4) == 1)
            and (buffers.stride(3) == 128)
            and (buffers.stride(2) == 128 * 128)
            and (buffers.stride(1) >= core.kv_heads * buffers.stride(2))
            and (buffers.stride(0) >= 2 * buffers.stride(1))
            and (buffers.stride(0) % 4 == 0)
            and (buffers.stride(1) % 4 == 0)
        )
    supported_index = (
        index_k_cache.is_cuda
        and index_k_cache.dtype == torch.float8_e4m3fn
        and (index_k_cache.dim() == 4)
        and (tuple(index_k_cache.shape[1:]) == (1, 128, 128))
        and (index_k_cache.stride(3) == 1)
        and (index_k_cache.stride(2) == 128)
        and (index_k_cache.stride(1) >= 128 * 128)
        and (index_k_cache.stride(0) >= index_k_cache.stride(1))
        and (index_k_cache.stride(0) % 4 == 0)
        and (index_k_cache.stride(1) % 4 == 0)
        and (buffers is not None)
        and (index_k_cache.shape[0] == buffers.shape[0])
    )
    cos_sin = core.rotary_cos_sin
    supported_rope = (
        cos_sin.is_cuda
        and cos_sin.dtype == torch.float32
        and cos_sin.is_contiguous()
        and (cos_sin.dim() == 3)
        and (tuple(cos_sin.shape[1:]) == (2, 32))
    )
    if (
        not supported_main
        or not supported_index
        or (not supported_rope)
        or (out_cache_loc is None)
        or (not out_cache_loc.is_cuda)
        or (out_cache_loc.dtype != torch.int32)
        or (not out_cache_loc.is_contiguous())
        or (out_cache_loc.numel() < rows)
    ):
        return _producer_miss(
            covered and layer_uses_nvfp4,
            rows,
            (
                "the layer's NVFP4 / index caches, RoPE table, KV quant scale or "
                "cache slots are not the layout the producer is built for"
            ),
        )
    w = core.w
    pos = position_ids.reshape(-1).contiguous().to(torch.int32)
    if (
        layer_uses_nvfp4
        and (core.heads, core.kv_heads, core.index_heads) == (8, 1, 1)
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
                early_trigger,
            )
            if folded is not None:
                return folded
            packed = deferred.materialize()
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
        )
    else:
        if layer_uses_nvfp4 and covered:
            missed(
                PRODUCER.path,
                rows,
                (
                    "the NVFP4 pool "
                    f"{tuple(buffers.shape)}"
                    " / scale pool "
                    f"{tuple(scale_buffers.shape)}"
                    " is not the producer's layout"
                ),
            )
        if deferred is not None:
            packed = deferred.materialize()
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
    return (q.flatten(1), index_q.flatten(1))


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
    early_trigger: bool = True,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """QKVPROD: the deferred QKV GEMV with layer ``i``'s producer in its epilogue (``mx_gemv.m3_mx_gemv_producer``),
    the same outputs and cache bytes as the GEMV + ``m3_producer``; None (nothing launched) when the kernel declines."""
    rows = int(deferred.act.shape[0])
    device = deferred.act.device
    q = torch.empty(rows, core.heads, core.head_dim, dtype=torch.float8_e4m3fn, device=device)
    index_q = torch.empty(rows, 1, core.head_dim, dtype=torch.float8_e4m3fn, device=device)
    w = core.w
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
        trigger_early=early_trigger,
        use_pdl=deferred.use_pdl,
    )
    if not ok:
        missed(
            QKV_PRODUCER_FOLD_PATH, rows, "the producer GEMV declined the step" + _uncompiled_note()
        )
        return None
    deferred.done = True
    if rows not in _QKV_FOLD_ENGAGED:
        _QKV_FOLD_ENGAGED.add(rows)
        logger.info(
            (
                "[M3 dispatch] QKVPROD producer in the QKV GEMV epilogue: engaged"
                " (rows="
                f"{rows}"
                ") on rank "
                f"{int(core.mapping.tp_rank)}"
            )
        )
    return (q.flatten(1), index_q.flatten(1))


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
        1.0,
        0.0,
        0.0,
        1.0,
        True,
        True,
        False,
        0,
        0,
    )
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
            f"MiniMax-M3 fused FP8 indexer requires an E4M3 index-K cache, got {index_k_cache.dtype}."
        )
    rows = int(idx_qk.shape[0])
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


def _plain_capture(spec_metadata, layer_id: int, hidden_states, residual) -> None:
    """The in-tree Eagle3 hidden-state capture."""
    spec_metadata.maybe_capture_hidden_states(layer_id, hidden_states, residual)


def _next_norm(core, i: int) -> torch.Tensor:
    """The norm the boundary after layer ``i`` applies: the next layer's input norm, the final norm after the last."""
    return core.w[f"l{i + 1}_in_norm"] if i + 1 < core.num_layers else core.w["final_norm"]


_ENGINE_STATES: Dict[tuple, object] = {}
AR_AUTOTUNE_ENABLED = m31_enabled("AR_AUTOTUNE")


def _pin_allreduce_heuristic(rank: int) -> bool:
    """Default the AllReduce autotune off for this process (see AR-DET); returns whether it is off. Logs one line."""
    if not AR_AUTOTUNE_ENABLED:
        os.environ.setdefault("TLLM_DISABLE_ALLREDUCE_AUTOTUNE", "1")
    off = os.environ.get("TLLM_DISABLE_ALLREDUCE_AUTOTUNE", "0") == "1"
    logger.info(
        f"M3.1 AR-DET: AllReduce autotune {('off (C++ size / world heuristic)' if off else 'on')} on rank {rank}"
    )
    return off


class MiniMaxM3Tp8ModelingV2(DecoderModel):
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
        self.swiglu_beta = 1.0
        self.swiglu_limit = float(getattr(cfg, "swiglu_limit", 7.0))
        self.max_num_tokens = model_config.max_num_tokens
        self.moe_max_num_tokens = model_config.moe_max_num_tokens
        self.sparse_layers = frozenset(_sparse_layer_ids(cfg))
        self.moe_layers = frozenset(_moe_layer_ids(cfg))
        mode = model_config.quant_config.quant_mode
        self.main_kv_is_fp8 = bool(mode is not None and mode.has_fp8_kv_cache())
        self.main_kv_is_nvfp4 = bool(mode is not None and mode.has_fp4_kv_cache())
        self.fused_projection = _weights.fused_projection(model_config.sparse_attention_config)
        self.w = _weights.MODEL_WEIGHTS.declare(d)
        self.mxfp8 = _Mxfp8GemmState()
        self._use_flashinfer_mxfp8_decode_graph_default = True
        self.allreduce = AllReduce(
            mapping=mapping, strategy=model_config.allreduce_strategy, dtype=cfg.torch_dtype
        )
        self.allreduce_wide = None
        self._allreduce_wide_owner = None
        if AR64AUTO and DECODE_MAX_TOKENS > m3_mc_allreduce.max_rows():
            self._allreduce_wide_owner = AllReduce(
                mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=cfg.torch_dtype
            )
            self.allreduce_wide = getattr(self._allreduce_wide_owner, "mnnvl_allreduce", None)
            if self.allreduce_wide is not None:
                logger.info(
                    (
                        "[M3.1] AR64AUTO: boundaries of "
                        f"{m3_mc_allreduce.max_rows() + 1}"
                        ".."
                        f"{DECODE_MAX_TOKENS}"
                        " rows take the strategy-MNNVL all-reduce (configured: "
                        f"{model_config.allreduce_strategy}"
                        ")"
                    )
                )
            else:
                logger.warning(
                    "[M3.1] AR64AUTO: MNNVL unavailable on this mapping; every boundary keeps the configured all-reduce"
                )
        self.aux_stream_moe = torch.cuda.Stream()
        chunked = self.moe_max_num_tokens < model_config.max_num_tokens * mapping.dp_size
        self.aux_stream_moe_chunk = torch.cuda.Stream() if chunked else None
        self.moe_events = {i: (torch.cuda.Event(), torch.cuda.Event()) for i in self.moe_layers}
        self.moe_chunk_events = {
            i: (torch.cuda.Event(), torch.cuda.Event()) for i in self.moe_layers
        }
        self.aux_stream_attn = None if self.fused_projection else torch.cuda.Stream()
        self.attn_events = (
            {}
            if self.fused_projection
            else {i: (torch.cuda.Event(), torch.cuda.Event()) for i in self.sparse_layers}
        )
        rope = RopeParams.from_config(cfg)
        self.rope_theta = rope.theta
        self.rotary_dim = int(rope.dim)
        self.rotary_cos_sin = rope.create_rope_const_params(interleave=False)[1].reshape(
            rope.max_positions, 2, -1
        )
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
                backend.dense_decode_max_seq_len_by_batch = DENSE_DECODE_MAX_SEQ_LEN_BY_BATCH
            self.attn.append(backend)
        self.kv_scales: Dict[int, torch.Tensor] = {}
        self.inv_kv_scales: Dict[int, torch.Tensor] = {}
        self.moe: Dict[int, _MoeScalars] = {}
        self.mx_alpha: Optional[torch.Tensor] = None
        self.moe_engines: Dict[int, Dict[Tuple[int, bool, bool], Optional[object]]] = {}
        self.embed_owners: Dict[tuple, DecodeEmbedding] = {}
        self._built_for: Optional[tuple] = None
        self._prefill: Optional[PrefillTarget] = None
        self._decode: Optional[DecodeTarget] = None

    def sm_scale(self, i: int) -> float:
        return self.head_dim ** (-0.5) / float(self.attn[i].q_scaling)

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
        isc1, isc3, isc2 = (w[f"l{i}_e_isc1"].data, w[f"l{i}_e_isc3"].data, w[f"l{i}_e_isc2"].data)
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
        table, weight = (self.w["embed"], self.w["l0_in_norm"])
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
        ``MOE_ENGINE``'s token counts, and the index scorer at the decode query lengths. Runs on the
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
            weight, scale = (self.w[wk], self.w[sk])
            mx_gemv.ensure_compiled(weight, scale, use_pdl)
            n, k = weight.shape
            for rows, x_bf16, ar in _gemv_variants(n, k, push, world):
                mx_gemv.precompile(weight, scale, rows, x_bf16, use_pdl=use_pdl, ar=ar)
                if (
                    QKV_PRODUCER_FOLD
                    and n == mx_gemv.PROD_N
                    and (not x_bf16)
                    and (ar is None)
                    and (rows <= 8)
                ):
                    mx_gemv.precompile(weight, scale, rows, False, use_pdl=use_pdl, prod=True)
                    if PRODUCER_EARLY_TRIGGER and PRODUCER_SHORTFIX:
                        mx_gemv.precompile(
                            weight,
                            scale,
                            rows,
                            False,
                            use_pdl=use_pdl,
                            prod=True,
                            trigger_early=False,
                        )
        swq_shapes = {tuple(self.w[wk].shape) for wk, _, _ in self.projections()} & set(SWQ_ROWS)
        for n, k in sorted(swq_shapes):
            gu = torch.zeros(
                max(SWQ_ROWS[n, k]), 2 * k, dtype=torch.bfloat16, device=self.w["embed"].device
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
        torch.cuda.synchronize()
        times["scorer_s"] = time.perf_counter() - start
        start = time.perf_counter()
        times["select_keys"] = []
        if long_select_dsl_keys():
            times["select_keys"] = _select_long_dsl_kernel.prebuild(
                long_select_dsl_keys(), self.w["embed"].device
            )
            torch.cuda.synchronize()
        times["select_s"] = time.perf_counter() - start
        start = time.perf_counter()
        times["proxy_variants"] = self._prebuild_msa_proxy() if MSA_PROXY_PREBUILD else []
        times["proxy_s"] = time.perf_counter() - start
        mx_gemv.freeze(True)
        logger.info(
            (
                "[M3 prebuild] decode kernels built before warmup: GEMVs %.1f s, "
                "MoE engines %.1f s, index scorer %.1f s (%d variants), MSA index"
                "er proxy %.1f s (%d variants)"
            )
            % (
                times["gemv_s"],
                times["moe_s"],
                times["scorer_s"],
                len(times["scorer_keys"]),
                times["proxy_s"],
                len(times["proxy_variants"]),
            )
        )
        return times

    def _expert_operands(self, i: int):
        """The routed experts' operands the engines read (trtllm-gen NVFP4 SwiGLU-OAI layout), and the input scale."""
        w, m = (self.w, self.moe[i])
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
        return (tensors, m.fc31_input_scale)

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
        take: ``MOE_ENGINE``'s token counts with the shared expert folded in. Whether an
        engine is kept is agreed across the TP group (one host collective per engine): a build that fails on one rank
        drops that engine on every rank, so the ranks never mix an engine (and its boundary push) with the generic
        MoE."""
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
        tensors, scale = self._expert_operands(i)
        device = tensors[0].device
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        kwargs = {}
        if shared:
            kwargs["shared_weights"] = self._shared_weights(i)
        state = _ENGINE_STATES.get((str(device), num_tokens, shared))
        if state is None:
            state = _ENGINE_STATES[str(device), num_tokens, shared] = M3MoeState(
                device, num_tokens=num_tokens, num_ctas=sms - MOE_ENGINE_RESERVED_SMS, shared=shared
            )
        engine = state.layer(
            *tensors,
            self.w[f"l{i}_router_bias"].data,
            router_weight=self.w[f"l{i}_router"].data,
            fc31_input_scale=scale.reshape(1).float(),
            **kwargs,
        )
        m3_moe_engine(
            torch.zeros(num_tokens, tensors[0].shape[-1] * 2, dtype=torch.bfloat16, device=device),
            engine,
        )
        return engine

    def _scorer_variants(self, query_lens) -> list:
        """The scorer keys a decode step can take: every sparse layer's (index dtype, index heads per rank, index dim)
        at each decode query length, with the early-issue schedule (up to ``INDEX_SCORE_EARLY_MAX_REQUESTS`` requests,
        none here) and without it, where the kernel serves it."""
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
                for early in (True, False) if INDEX_SCORE_EARLY_MAX_REQUESTS else (False,):
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

        only_score, page = (2, 128)
        names = []
        for i in sorted(self.sparse_layers):
            backend = self.attn[i]
            dtype = torch.float8_e4m3fn if backend.indexer_kv_dtype == "fp8" else torch.bfloat16
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


def _swiglu_act(core, gu: torch.Tensor) -> torch.Tensor:
    return m3_silu_and_mul(gu, core.swiglu_limit, core.swiglu_alpha, core.swiglu_beta)


class PrefillTarget:
    """A step with context rows: prefill chunks, and the mixed steps that carry generation rows beside them. Every
    layer runs in its generic form; the attention core is the MSA backend's for every row."""

    def __init__(self, core: MiniMaxM3Tp8ModelingV2) -> None:
        self.core = core
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
        hidden = (
            m3_embedding(input_ids, core.w["embed"]) if inputs_embeds is None else inputs_embeds
        )
        residual = None
        for i in range(core.num_layers):
            hidden, residual = self._layer(
                i, position_ids, hidden, residual, step, spec_metadata, capture
            )
        return hidden

    def _swiglu_mlp(self, h: torch.Tensor, prefix: str) -> torch.Tensor:
        """A gated MXFP8 MLP (the dense MLP, the shared expert): gate_up, SwiGLU-OAI, down, no reduction."""
        core = self.core
        gu, _ = _linear(core, h, f"{prefix}_gu_w", f"{prefix}_gu_s")
        act = _swiglu_act(core, gu)
        out, _ = _linear(core, act, f"{prefix}_dn_w", f"{prefix}_dn_s")
        return out

    def _routed_chunks(self, i: int, x, router_logits):
        """The routed experts from the bf16 activation over at most ``moe_max_num_tokens`` rows per runner call:
        the MoE scheduler's even split, even chunks on the MoE chunking stream."""
        core = self.core
        rows = x.shape[0]
        num_chunks = (rows + core.moe_max_num_tokens - 1) // core.moe_max_num_tokens
        fp4 = _CTXNORMQ_FP4.take(x) if CTXNORMQ else None
        if num_chunks <= 1:
            if fp4 is not None:
                return _routed_experts(core, i, fp4[0], fp4[1], router_logits, True)
            return _routed_experts(core, i, x, None, router_logits, True)
        sizes = _split_chunk(rows, num_chunks)
        xs, logits = (x.split(sizes), router_logits.split(sizes))
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
            and (1 <= rows <= MOE_ENGINE_MAX_TOKENS)
            and (rows >= MOE_FOLD_MIN_TOKENS)
        ):
            engine = core.engine(i, rows, shared=True)
            if engine is not None:
                folded = m3_moe_engine(h.contiguous(), engine, None, None)
                return (folded, None)
        routed_out, shared_out = maybe_execute_in_parallel(
            lambda: self._routed(i, h),
            lambda: self._swiglu_mlp(h, f"l{i}_sh"),
            core.moe_events[i][0],
            core.moe_events[i][1],
            None if flashinfer_mxfp8_autotune_active() else core.aux_stream_moe,
            disable_on_compile=True,
        )
        if return_parts:
            return (shared_out, routed_out)
        return (shared_out.add_(routed_out), None)

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
        if norm_w is not core.w["final_norm"]:
            fused = _ctxnormq_allreduce_norm(core, x, residual, norm_w, None)
            if fused is not None:
                return fused
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
            fused = _ctxnormq_allreduce_norm(
                core,
                hidden,
                residual,
                core.w[f"l{i}_post_norm"],
                core.moe[i].fc31_input_scale if i in core.moe_layers else None,
            )
            hidden, residual = (
                fused
                if fused is not None
                else _native_allreduce_norm(core, hidden, residual, core.w[f"l{i}_post_norm"])
            )
        if i in core.moe_layers:
            out, routed = self._moe(i, hidden, return_parts=_can_mc(hidden, folded=True))
        else:
            out = self._swiglu_mlp(hidden, f"l{i}_mlp")
            routed = None
        normed, updated = self._generic_boundary(out, routed, residual, _next_norm(core, i))
        if spec_metadata is not None and spec_metadata.is_layer_capture(i):
            capture(spec_metadata, i, updated, None)
        return (normed, updated)

    def _attention(self, i: int, position_ids, hidden, step):
        """Attention of layer ``i`` up to its o_proj (no reduction). Returns (o_proj output, pushed): a prefill step
        arms no boundary push, so pushed is always False."""
        core = self.core
        if i not in core.sparse_layers:
            qkv, _ = _linear(core, hidden, f"l{i}_qkv_w", f"l{i}_qkv_s")
            fused = _qk_norm_rope(core, i, qkv, position_ids, core.main_kv_is_fp8)
            q, k, v = fused.split([core.q_size, core.kv_size, core.kv_size], dim=-1)
            o = self._attention_core(i, q, k, v, None, step)
            return _linear(core, o, f"l{i}_o_w", f"l{i}_o_s")
        if not core.fused_projection:
            return self._unfused_sparse_attention(i, position_ids, hidden, step)
        packed, _ = _linear(core, hidden, f"l{i}_qkv_w", f"l{i}_qkv_s", defer=QKV_PRODUCER_FOLD)
        deferred = None
        if isinstance(packed, _DeferredGemv):
            deferred, packed = (packed, packed.placeholder)
        horizontal = _sparse_qkv_producer(
            core,
            i,
            packed,
            position_ids,
            step["kv_cache_manager"],
            step["msa_idx_k_cache"],
            step["msa_out_cache_loc"],
            deferred,
            early_trigger=_producer_early_trigger(step),
        )
        if horizontal is None:
            raise RuntimeError(
                (
                    "MiniMax-M3 sparse layer "
                    f"{i}"
                    ": no horizontal producer takes this step's caches (an FP8 or NVF"
                    "P4 HND pool with an FP8 index-K cache)"
                )
            )
        q, idx_q = horizontal
        o = self._attention_core(i, q, None, None, idx_q, step)
        return _linear(core, o, f"l{i}_o_w", f"l{i}_o_s")

    def _unfused_sparse_attention(self, i: int, position_ids, hidden, step):
        """A sparse layer without the five-way projection (``fuse_qkv_index_projection=False``): the QKV projection
        and its fused norm + RoPE (FP8 q/k/v into an FP8 cache) on the current stream; the MXFP8 index projection
        (the checkpoint's block scales applied) and the FP8 indexer kernel on the attention aux stream; then the MSA
        backend (it writes the main KV cache) and the o_proj. The NVFP4 producer does not apply: it packs the
        five-way projection's output."""
        core = self.core
        if _producer_covered(core):
            uncovered(
                PRODUCER.path,
                "fuse_qkv_index_projection is disabled: QKV + separate MXFP8 index projection",
            )

        def _main():
            qkv, _ = _linear(core, hidden, f"l{i}_qkv_w", f"l{i}_qkv_s")
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
        return _linear(core, o, f"l{i}_o_w", f"l{i}_o_s")

    def _attention_core(self, i: int, q, k, v, idx_q, step) -> torch.Tensor:
        """The attention core over the step's live tokens (a padded step's pad rows are left as allocated): the MSA
        backend (``m3_msa_attention``: the KV write unless a producer wrote the caches, the sparse layers' top-k block
        selection, the FMHA over the context rows and any generation rows beside them)."""
        core = self.core
        output = q.new_empty((q.shape[0], core.heads * core.head_dim), dtype=torch.bfloat16)
        live = int(step["num_tokens"])
        q_live, out_live = (q[:live], output[:live])
        k_live = k[:live] if k is not None else None
        v_live = v[:live] if v is not None else None
        idx_q_live = idx_q[:live] if idx_q is not None else None
        sparse = i in core.sparse_layers
        nvfp4 = core.main_kv_is_nvfp4
        if TSEL and sparse:
            _tsel_register(core.attn[i])
        if GQAMIX and sparse:
            _gqamix_register(core.attn[i])
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

    Up to 8 tokens with 1-D input ids from one request, per layer: attention (o_proj all-reduce deferred, its rows
    pushed into the boundary all-reduce) -> attention boundary (all-reduce + residual + post-attention RMSNorm) ->
    MoE (the 1..8-row engine, pushing its sum into the boundary all-reduce) or dense MLP -> MoE boundary
    (all-reduce + shared/routed add + residual +
    the next layer's RMSNorm, emitting the next QKV's MXFP8 activation). A layer whose output Eagle3 captures runs the
    generic layer, as does the first layer when the fused embedding does not take the step. A wider step (or one
    carrying embeddings) runs every layer in its generic form, with this target's attention (the decode selection and
    sparse decode where the producer wrote the caches).
    """

    def __init__(self, core: MiniMaxM3Tp8ModelingV2) -> None:
        self.core = core
        self._moe_push_descs: Dict[tuple, Any] = {}
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
        kernels = (
            inputs_embeds is None
            and input_ids.ndim == 1
            and attn_metadata.num_seqs == 1
            and is_decode_step(input_ids.numel())
        )
        if kernels:
            num_tokens = int(input_ids.numel())
            plan = plan_for(num_tokens)
            hidden, residual = self._embed(input_ids, num_tokens)
        else:
            uncovered(
                "decode_kernels",
                f"outside single-request {FUSED_DECODE_MAX_TOKENS}-row fused scope, "
                "embeddings or 2-D ids: generic layers",
            )
            if is_in_breakable_cuda_graph():
                raise RuntimeError(
                    "this MiniMax-M3 target does not run under piecewise (breakable) CUDA graphs"
                )
            plan = None
            hidden = (
                m3_embedding(input_ids, core.w["embed"]) if inputs_embeds is None else inputs_embeds
            )
            residual = None
        for i in range(core.num_layers):
            captured = spec_metadata is not None and spec_metadata.is_layer_capture(i)
            if not kernels or residual is None or (captured and (not FUSED_CAPTURE)):
                hidden, residual = self._generic_layer(
                    i, position_ids, hidden, residual, step, spec_metadata, capture
                )
            else:
                hidden, residual = self._layer(i, plan, position_ids, hidden, residual, step)
                if captured:
                    capture(spec_metadata, i, residual, None)
                    _log_once(
                        logger.info,
                        "B-4",
                        "fused capture",
                        (
                            "[M3 dispatch] B-4 fused capture taken: capture layer "
                            f"{i}"
                            " runs the decode-kernel layer (logged once)"
                        ),
                    )
        return hidden

    def _embed(self, input_ids: torch.Tensor, num_tokens: int):
        """(hidden, residual) of the fused embedding + first RMSNorm; (embedding, None) when it does not run, so the
        first layer runs its own input norm."""
        core = self.core
        if not EMBED_FIRSTNORM.covers(num_tokens):
            uncovered(EMBED_FIRSTNORM.path, f"more than {EMBED_FIRSTNORM_MAX_TOKENS} tokens")
            return (m3_embedding(input_ids, core.w["embed"]), None)
        owner = core.embed_owners.get((num_tokens, input_ids.dtype))
        if owner is None or not input_ids.is_contiguous():
            missed(
                EMBED_FIRSTNORM.path,
                num_tokens,
                f"no embedding owner for {input_ids.dtype} ids (built for int32 / int64 after the load)"
                if owner is None
                else "the input ids are not contiguous",
            )
            return (m3_embedding(input_ids, core.w["embed"]), None)
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
        return (
            desc,
            (
                set_b,
                flags.data_ptr(),
                workspace.rank,
                workspace.world_size,
                workspace.max_tokens,
                2,
            ),
        )

    def _boundary(self, use_mc: bool, x, residual, norm_w, x2=None, quant=False):
        """All-reduce + (second addend) + residual + RMSNorm: the multicast kernel, else the built-in all-reduce."""
        core = self.core
        if use_mc and (not MC_ALLREDUCE.covers(x.shape[0])):
            uncovered(
                MC_ALLREDUCE.path,
                f"more than {m3_mc_allreduce.max_rows()} rows: the built-in all-reduce runs",
            )
            use_mc = False
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
            out, _ = self._swiglu_mlp(hidden, f"l{i}_mlp")
            quant = False
            return self._boundary(plan.attn_mc_ar, out, residual, norm_w, quant=quant)
        quant = plan.qkv_prequant and i + 1 < core.num_layers
        path = moe_path(num_tokens)
        if path is None:
            uncovered(
                MC_FINALIZE.path,
                f"more than {m3_mc_allreduce.max_rows()} rows: the finalized MoE and the boundary all-reduce run",
            )
        moe_push = None
        if path == MOE_ENGINE.path:
            engine = core.engine(i, num_tokens, shared=True)
            if engine is not None and plan.moe_mc_ar and decode_graph:
                moe_push = self._moe_push_targets(num_tokens)
        if path == MC_FINALIZE.path and plan.moe_mc_ar:
            if self._bucket_flags(num_tokens) is None:
                self._no_bucket(MC_FINALIZE.path, num_tokens)
            else:
                (shared, fc2, scales, perm), _, _, _ = self._moe(
                    i, hidden, return_parts=False, do_finalize=False
                )
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

    def _swiglu_mlp(self, h: torch.Tensor, prefix: str, push: Optional[tuple] = None):
        """A gated MXFP8 MLP (the dense MLP, the shared expert): gate_up, SwiGLU-OAI, down, no reduction. Returns
        (output, whether the down projection's GEMV pushed into ``push``)."""
        core = self.core
        gu, _ = _linear(core, h, f"{prefix}_gu_w", f"{prefix}_gu_s")
        n, k = core.w[f"{prefix}_dn_w"].shape
        rows = gu.shape[0]
        if rows in SWQ_ROWS.get((n, k), ()) and _mx_mode(core.mxfp8).decode_graph:
            act, act_scale = m3_swq(gu, core.swiglu_limit, core.swiglu_alpha, core.swiglu_beta)
            m3_prequant.SLOT.publish(gu, act, act_scale)
            if rows not in _SWQ_LOGGED:
                _SWQ_LOGGED.add(rows)
                logger.info(f"[M3 dispatch] SWQ: SwiGLU + MXFP8 producer rows={rows} [{n}, {k}]")
            return _linear(core, gu, f"{prefix}_dn_w", f"{prefix}_dn_s", push)
        act = _swiglu_act(core, gu)
        return _linear(core, act, f"{prefix}_dn_w", f"{prefix}_dn_s", push)

    def _routed_chunks(self, i: int, x, router_logits):
        """The routed experts from the bf16 activation over at most ``moe_max_num_tokens`` rows per runner call: the MoE
        scheduler's even split, even chunks on the MoE chunking stream."""
        core = self.core
        rows = x.shape[0]
        num_chunks = (rows + core.moe_max_num_tokens - 1) // core.moe_max_num_tokens
        if num_chunks <= 1:
            return _routed_experts(core, i, x, None, router_logits, True)
        sizes = _split_chunk(rows, num_chunks)
        xs, logits = (x.split(sizes), router_logits.split(sizes))
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
            and (h.dtype == torch.bfloat16)
            and (1 <= rows <= MOE_ENGINE_MAX_TOKENS)
            and (rows >= MOE_FOLD_MIN_TOKENS)
        ):
            engine = core.engine(i, rows, shared=True)
            if engine is not None:
                folded = m3_moe_engine(h.contiguous(), engine, None, engine_arp)
                return (folded, None, engine_arp is not None, False)
        routed_out, (shared_out, shared_pushed) = maybe_execute_in_parallel(
            lambda: self._routed(i, h, do_finalize),
            lambda: self._swiglu_mlp(h, f"l{i}_sh", shared_ar),
            core.moe_events[i][0],
            core.moe_events[i][1],
            None if flashinfer_mxfp8_autotune_active() else core.aux_stream_moe,
            disable_on_compile=True,
        )
        if not do_finalize:
            return ([shared_out, *routed_out], None, False, shared_pushed)
        if return_parts:
            return (shared_out, routed_out, False, shared_pushed)
        return (shared_out.add_(routed_out), None, False, shared_pushed)

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
            out, _ = self._swiglu_mlp(hidden, f"l{i}_mlp")
            routed = None
        normed, updated = self._generic_boundary(out, routed, residual, _next_norm(core, i))
        if spec_metadata is not None and spec_metadata.is_layer_capture(i):
            capture(spec_metadata, i, updated, None)
        return (normed, updated)

    def _attention(self, i: int, position_ids, hidden, step, push: Optional[tuple] = None):
        """Attention of layer ``i`` up to its o_proj (no reduction). ``push`` is the attention boundary's exchange the
        o_proj may write into. Returns (o_proj output, pushed)."""
        core = self.core
        if i not in core.sparse_layers:
            qkv, _ = _linear(core, hidden, f"l{i}_qkv_w", f"l{i}_qkv_s")
            out_fp8 = core.main_kv_is_fp8 or core.main_kv_is_nvfp4
            fused = _qk_norm_rope(core, i, qkv, position_ids, out_fp8)
            q, k, v = fused.split([core.q_size, core.kv_size, core.kv_size], dim=-1)
            o = self._attention_core(i, q, k, v, None, step)
            return _linear(core, o, f"l{i}_o_w", f"l{i}_o_s", push)
        if not core.fused_projection:
            return self._unfused_sparse_attention(i, position_ids, hidden, step, push)
        packed, _ = _linear(core, hidden, f"l{i}_qkv_w", f"l{i}_qkv_s", defer=QKV_PRODUCER_FOLD)
        deferred = None
        if isinstance(packed, _DeferredGemv):
            deferred, packed = (packed, packed.placeholder)
        horizontal = _sparse_qkv_producer(
            core,
            i,
            packed,
            position_ids,
            step["kv_cache_manager"],
            step["msa_idx_k_cache"],
            step["msa_out_cache_loc"],
            deferred,
            early_trigger=_producer_early_trigger(step),
        )
        if horizontal is None:
            raise RuntimeError(
                (
                    "MiniMax-M3 sparse layer "
                    f"{i}"
                    ": no horizontal producer takes this step's caches (an FP8 or NVF"
                    "P4 HND pool with an FP8 index-K cache)"
                )
            )
        q, idx_q = horizontal
        o = self._attention_core(i, q, None, None, idx_q, step)
        return _linear(core, o, f"l{i}_o_w", f"l{i}_o_s", push)

    def _unfused_sparse_attention(self, i: int, position_ids, hidden, step, push):
        """A sparse layer without the five-way projection (``fuse_qkv_index_projection=False``): the QKV projection
        and its fused norm + RoPE (FP8 q/k/v into an FP8 cache) on the current stream; the MXFP8 index projection
        (the checkpoint's block scales applied) and the FP8 indexer kernel on the attention aux stream; then the MSA
        backend (it writes the main KV cache) and the o_proj. The NVFP4 producer does not apply: it packs the
        five-way projection's output. ``push`` is the attention
        boundary exchange the o_proj GEMV writes its rows into, as in the five-way path."""
        core = self.core
        if _producer_covered(core):
            uncovered(
                PRODUCER.path,
                "fuse_qkv_index_projection is disabled: QKV + separate MXFP8 index projection",
            )

        def _main():
            qkv, _ = _linear(core, hidden, f"l{i}_qkv_w", f"l{i}_qkv_s")
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
        return _linear(core, o, f"l{i}_o_w", f"l{i}_o_s", push)

    def _attention_core(self, i: int, q, k, v, idx_q, step) -> torch.Tensor:
        """The attention core over the step's live tokens (a padded step's pad rows are left as allocated): a sparse
        layer whose caches the producer wrote runs this target's selection and sparse decode when the generation span
        starts at row 0; everything else the MSA backend (``m3_msa_attention``: the KV write unless a producer wrote the
        caches, the sparse layers' top-k block selection, the FMHA)."""
        core = self.core
        output = q.new_empty((q.shape[0], core.heads * core.head_dim), dtype=torch.bfloat16)
        live = int(step["num_tokens"])
        q_live, out_live = (q[:live], output[:live])
        k_live = k[:live] if k is not None else None
        v_live = v[:live] if v is not None else None
        idx_q_live = idx_q[:live] if idx_q is not None else None
        span = step["msa_decode_span"]
        sparse = i in core.sparse_layers
        if sparse and k_live is None and (span is not None) and (span.row_first == 0):
            self._decode_sparse_core(i, q_live, idx_q_live, step, out_live)
            return output
        nvfp4 = core.main_kv_is_nvfp4
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

        The selection is the identity when the metadata staged it (every request within 16 blocks). Otherwise a span of
        at most ``FUSED_SCORE_SELECT_MAX_REQUESTS`` requests scores and selects in one launch; a wider span runs the
        CuTe DSL scorer (the per-index-head block maxima, reduced to the KV head) and the long-row selector (the
        built-in selector beyond its shapes).
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
        fused = (
            None
            if identity or False
            else self._fused_score_select(i, idx_q, step, rows, qlen, num_tokens)
        )
        if identity:
            topk = step["msa_short_table"][: num_kv_heads * num_tokens * MSA_REQUIRED_TOPK].view(
                num_kv_heads, num_tokens, MSA_REQUIRED_TOPK
            )
        elif fused is not None:
            topk = fused.permute(1, 0, 2)
        else:
            for _rep in range(1):
                idx_q_view = idx_q.view(num_tokens, config.num_index_heads, config.sparse_index_dim)
                idx_k_cache = step["msa_idx_k_cache"](i)
                max_score = step["msa_proxy_max_score_view"](
                    config.num_index_heads, step["msa_worst_case_max_k_tiles"], num_tokens
                )
                early = _index_score_early(rows, int(core.mapping.tp_rank))
                group = config.num_index_heads // config.num_kv_heads
                key = (
                    idx_q_view.dtype,
                    int(config.num_index_heads),
                    int(qlen),
                    int(idx_q_view.shape[-1]),
                    int(INDEX_SCORE_SPLIT_K),
                    bool(early),
                )
                if key not in _index_score_kernel.IndexScoreRunner.kernel_cache:
                    missed(
                        INDEX_SCORE.path,
                        num_tokens,
                        f"the index scorer {key} was not compiled before warmup",
                    )
                score_view = max_score.transpose(1, 2)
                m3_index_score(
                    idx_q_view,
                    idx_k_cache.squeeze(1),
                    block_table,
                    seq_lens,
                    score_view,
                    qlen,
                    INDEX_SCORE_SPLIT_K,
                    early,
                )
                n_valid_blocks = step["msa_n_valid_blocks"][:num_tokens]
                max_score_kv = (
                    max_score.view(
                        config.num_kv_heads, group, max_score.shape[1], max_score.shape[2]
                    ).amax(dim=1)
                    if group > 1
                    else max_score
                )
                early_trigger = num_tokens <= SELECT_EARLY_MAX_ROWS
                dsl = False
                if LONG_SELECT_DSL.keys:
                    dsl_key = _select_long_dsl_kernel.kernel_key(
                        get_env_enable_pdl(), early_trigger, 0
                    )
                    dsl = dsl_key in _select_long_dsl_kernel.kernel_cache
                    if dsl:
                        _note_select_dsl(int(core.mapping.tp_rank))
                    else:
                        missed(
                            LONG_SELECT_DSL.path,
                            num_tokens,
                            f"the 384-thread select {dsl_key} was not compiled before warmup",
                        )
                table = long_select_blocks(
                    max_score_kv,
                    n_valid_blocks=n_valid_blocks,
                    init_blocks=config.init_blocks,
                    local_blocks=config.local_blocks,
                    use_pdl=get_env_enable_pdl(),
                    early_trigger=early_trigger,
                    dsl=dsl,
                )
                if table is None:
                    uncovered(
                        LONG_SELECT.path,
                        (
                            f"{max_score_kv.shape[1]}"
                            " KV blocks, outside "
                            f"{LONG_SELECT_MIN_BLOCKS}"
                            ".."
                            f"{LONG_SELECT_MAX_BLOCKS}"
                            ": the built-in selector"
                        ),
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

    def _fused_score_select(self, i: int, idx_q, step, rows: int, qlen: int, num_tokens: int):
        """The top-16 table [total_q, num_kv_heads, 16] of a covered decode span from the fused score + select, or None
        (after a miss when the span is covered) for the scorer + selector."""
        core = self.core
        path = FUSED_SCORE_SELECT.path
        if not FUSED_SCORE_SELECT.covers(rows):
            uncovered(
                path, f"more than {FUSED_SCORE_SELECT_MAX_REQUESTS} requests: scorer + selector"
            )
            return None
        if qlen > FUSED_SCORE_SELECT_MAX_QUERY_LEN:
            uncovered(
                path,
                f"{qlen} query rows per request (more than {FUSED_SCORE_SELECT_MAX_QUERY_LEN}): scorer + selector",
            )
            return None
        config = core.attn[i].m3_config
        idx_q_view = idx_q.view(num_tokens, config.num_index_heads, config.sparse_index_dim)
        idx_k_pages = step["msa_idx_k_cache"](i).squeeze(1)
        max_tiles = int(step["msa_worst_case_max_k_tiles"])
        n_valid = step["msa_n_valid_blocks"]
        reason = _fused_score_select_kernel.supported(
            idx_q_view, idx_k_pages, qlen, config.num_kv_heads, max_tiles
        )
        if reason is None and (
            num_tokens != rows * qlen
            or n_valid is None
            or n_valid.dtype != torch.int32
            or (int(n_valid.shape[0]) < num_tokens)
        ):
            reason = "the step's query rows or valid-block counts are not one decode span"
        if reason is not None:
            missed(path, rows, reason)
            return None
        bufs = _fused_score_select_kernel.buffers(idx_q.device, max_tiles)
        if bufs is None:
            capture_missed(path, rows, "the fused score + select scratch")
            return None
        return _fused_score_select_kernel.fused_score_select(
            idx_q_view,
            idx_k_pages,
            step["msa_block_table"][:rows],
            step["msa_seq_lens_cuda"][:rows],
            n_valid[:num_tokens].contiguous(),
            decode_query_len=qlen,
            num_kv_heads=config.num_kv_heads,
            init_blocks=config.init_blocks,
            local_blocks=config.local_blocks,
            max_tiles=max_tiles,
            scratch=bufs[0],
            counters=bufs[1],
        )


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


class M3DSparkStepFold(M3SpecStepFold):
    """The step fold under the standalone DSpark worker (DSFOLD): the engine's previous-batch gathers in one launch
    (``step_gather``). ``taken`` / ``declined`` count the steps the gather ran / fell back to the engine's own copies.

    With ``boundary_pack`` (BOUNDARY_PACK BP-1) a batch-1 decode graph also ends with the sampler's slot stores and the
    next step's gather (``graph_tail``: one ``ops/m3_bp_tail`` launch), so the boundary between two replays runs
    neither:

    * ``begin_step`` (before every replay) writes the device control word (slot, live) when it changed: live 1 with
      the step's slot when the step is one row the fold identifies (a slot inside the store), else live 0, so a replay
      the fold did not take (a dummy-only replay, an unidentified step) writes nothing at all; it reports the inputs
      staged when the previous graph gathered them for exactly these rows;
    * the graph's tail waits for the sampler's previous side-stream store copies (``store_copies_done``), then with live
      1 stores the row as the sampler's ``_store_outputs`` (same output row, same store, zero padding / truncation to
      the store widths, at the slot) and, with a static draft length, writes the engine's next-step gathers of that row
      (P = 1, ids0 = d0 = b0 = 0, both offset buffers zeroed elsewhere), and marks its outputs;
    * ``sampler_rows_stored`` (after the replay) accepts a marked step whose rows are the live ones, and arms the next
      ``begin_step``.

    Every other step (context or mixed steps, batches over one row, eager forwards, a draft length that can change,
    first steps, slots the fold could not identify) keeps the sampler's stores and the engine's gathers.
    ``bp_stores`` / ``bp_gathers`` count the steps whose stores / gathers the graph took."""

    store_copies_done: Optional[torch.cuda.Event] = None

    def __init__(self, boundary_pack: bool = False, tokens_per_step: Optional[int] = None) -> None:
        super().__init__()
        self.taken = 0
        self.declined = 0
        self.boundary_pack = boundary_pack
        self.tokens_per_step = tokens_per_step
        self.bp_stores = 0
        self.bp_gathers = 0
        self._bp_ctl: Optional[torch.Tensor] = None
        self._bp_ctl_host: Optional[list] = None
        self._bp_block = 0
        self._bp_rows: Optional[list] = None
        self._bp_staged: Optional[list] = None
        self._bp_token = id(self)

    def bind_engine_buffers(
        self, input_ids, draft_tokens, position_id_offsets, kv_lens_offsets, max_batch_size
    ):
        super().bind_engine_buffers(
            input_ids, draft_tokens, position_id_offsets, kv_lens_offsets, max_batch_size
        )
        self._bp_reset()

    def bind_sampler_store(self, store: object) -> None:
        super().bind_sampler_store(store)
        self._bp_reset()

    def _bp_reset(self) -> None:
        """A new binding (an engine or a sampler built again): BOUNDARY_PACK's buffers are made again for it."""
        self._bp_ctl = self._bp_ctl_host = None
        self._bp_rows = self._bp_staged = None
        self.store_copies_done = None

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
        return True

    def scatter(
        self, seq_slots, new_tokens, next_new_tokens, new_tokens_lens, next_draft_tokens
    ) -> bool:
        return False

    def _bp_ready(self) -> bool:
        """BOUNDARY_PACK: the control word (live 0), the tail kernel's two variants and the store-copies event, made
        once outside capture when the sampler store and the engine buffers are bound and the tail can take them."""
        if not self.boundary_pack:
            return False
        if self._bp_ctl is not None:
            return True
        store = self.store
        names = ("new_tokens", "next_new_tokens", "new_tokens_lens", "next_draft_tokens")
        if (
            self.step_inputs is None
            or not all((isinstance(getattr(store, name, None), torch.Tensor) for name in names))
            or torch.cuda.is_current_stream_capturing()
        ):
            return False
        tensors = [getattr(store, name) for name in names] + list(self.step_inputs)
        if (
            any((t.dtype != torch.int32 or not t.is_cuda for t in tensors))
            or not all((t.is_contiguous() for t in self.step_inputs))
            or (not m3_bp_tail.aligned(*tensors))
        ):
            uncovered(
                "BOUNDARY_PACK",
                "the store or the engine buffers are not aligned int32 CUDA tensors",
            )
            return False
        ctl = torch.zeros(2, dtype=torch.long, device=store.new_tokens_lens.device)
        block = m3_bp_tail.block_for(store)
        m3_bp_tail.warm_up(ctl, store, self.step_inputs, block)
        done = torch.cuda.Event(external=True)
        done.record()
        self.store_copies_done = done
        self._bp_ctl, self._bp_ctl_host, self._bp_block = (ctl, [0, 0], block)
        return True

    def begin_step(self, seq_slots: Optional[list], num_rows: int) -> bool:
        staged, self._bp_staged = (self._bp_staged, None)
        self._bp_rows = None
        if not self._bp_ready():
            return False
        num_slots = self.store.new_tokens_lens.shape[0]
        rows = list(seq_slots) if seq_slots is not None else None
        live = (
            rows is not None
            and len(rows) == num_rows
            and (0 < num_rows <= BOUNDARY_PACK_MAX_ROWS)
            and all((isinstance(slot, int) and 0 <= slot < num_slots for slot in rows))
        )
        ctl = [rows[0], 1] if live else [self._bp_ctl_host[0], 0]
        if ctl != self._bp_ctl_host:
            self._bp_ctl.copy_(
                torch.tensor(ctl, dtype=torch.long, pin_memory=prefer_pinned()), non_blocking=True
            )
            self._bp_ctl_host = ctl
        if not live:
            return False
        self._bp_rows = rows
        if staged != rows:
            return False
        if self.bp_gathers == 0:
            logger.info(
                "[M3 dispatch] BOUNDARY-PACK BP-1: sampler stores + step gather in the step graph (taken)"
            )
        self.bp_gathers += 1
        return True

    def sampler_rows_stored(self, outputs: dict, num_skip: int, seq_slots: list) -> bool:
        self._bp_staged = None
        mark = outputs.get(_BOUNDARY_PACK_MARK) if isinstance(outputs, dict) else None
        if mark is None or mark[0] != self._bp_token or self._bp_rows is None:
            return False
        rows, gathered = (mark[1], mark[2])
        if num_skip != 0 or len(seq_slots) != rows or list(seq_slots) != self._bp_rows:
            uncovered(
                "BOUNDARY_PACK",
                (
                    "a step graph stored rows that are not the sampler's (skipped con"
                    "text rows or other slot ids); the sampler stores them"
                ),
            )
            return False
        self.bp_stores += 1
        if gathered:
            self._bp_staged = list(seq_slots)
        return True

    def graph_tail(self, outputs, attn_metadata):
        """BOUNDARY_PACK, at the end of a decode step's forward: under capture of a batch-1 generation graph, the
        sampler's slot stores and the next step's gather, gated by the live word (see the class docstring); returns
        the outputs, marked when the graph has the tail. Anything else returns them unchanged (the eager path)."""
        if not self.boundary_pack or not torch.cuda.is_current_stream_capturing():
            return outputs
        path = "BOUNDARY_PACK"
        if not isinstance(outputs, dict) or self._bp_ctl is None:
            uncovered(
                path,
                "the sampler store or the engine buffers are not bound: eager stores and gathers",
            )
            return outputs
        if attn_metadata.num_contexts != 0:
            return outputs
        names = ("new_tokens", "next_new_tokens", "new_tokens_lens", "next_draft_tokens")
        srcs = [outputs.get(name) for name in names]
        if any(
            (
                not isinstance(t, torch.Tensor) or t.dtype != torch.int32 or (not t.is_cuda)
                for t in srcs
            )
        ) or [t.dim() for t in srcs] != [2, 2, 1, 2]:
            uncovered(
                path,
                "the worker outputs are not int32 [rows, width] / [rows]: eager stores and gathers",
            )
            return outputs
        rows = srcs[0].shape[0]
        if any((t.shape[0] != rows for t in srcs)):
            uncovered(path, "the worker outputs disagree on their rows: eager stores and gathers")
            return outputs
        if not 0 < rows <= BOUNDARY_PACK_MAX_ROWS:
            uncovered(
                path,
                f"a {rows}-row decode graph (only batch-1 graphs, which are never padded): eager stores and gathers",
            )
            return outputs
        src = [t[:1] for t in srcs]
        if not m3_bp_tail.aligned(*src):
            uncovered(path, "a worker output is not 16-byte aligned: eager stores and gathers")
            return outputs
        store = self.store
        num_tokens = int(attn_metadata.num_tokens)
        t = num_tokens // rows
        gather = (
            self.tokens_per_step is not None
            and t == self.tokens_per_step
            and (num_tokens == rows * t)
            and (t <= store.next_new_tokens.shape[0])
            and (t - 1 <= store.next_draft_tokens.shape[1])
        )
        if not gather:
            uncovered(
                path,
                (
                    "the next step's tokens per request may differ from this graph's "
                    f"{t}"
                    " (a draft length that can change, or "
                    f"{num_tokens}"
                    " tokens over "
                    f"{rows}"
                    " rows): the graph stores, the engine gathers"
                ),
            )
        torch.cuda.current_stream().wait_event(self.store_copies_done)
        m3_bp_tail.bp_tail(self._bp_ctl, src, store, self.step_inputs, t, gather, self._bp_block)
        marked = dict(outputs)
        marked[_BOUNDARY_PACK_MARK] = (self._bp_token, rows, gather)
        return marked


def _static_tokens_per_step(spec_config) -> Optional[int]:
    """BOUNDARY_PACK: the tokens per request of every decode step when the draft length cannot change at runtime (no
    draft-length schedule, no acceptance-rate gate that turns speculation off), else None."""
    if getattr(spec_config, "draft_len_schedule", None) is not None:
        return None
    if (
        getattr(spec_config, "acceptance_rate_window_size", None)
        and getattr(spec_config, "acceptance_rate_threshold", None) is not None
    ):
        return None
    from tensorrt_llm._torch.speculative.utils import get_static_draft_len

    return int(spec_config.get_runtime_tokens_per_gen_step(get_static_draft_len(spec_config)))


_DRAFT_LOWM_BUILT: set = set()


def _draft_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """A drafter BF16 linear ``x @ weight^T``: FlashInfer's fixed-tactic low-M kernel on the rows ``DRAFT_LOWM``
    covers (compiled before warmup; a covered call found uncompiled is a miss), else the in-tree BF16 GEMM."""
    n, k = weight.shape
    m = int(x.shape[0])
    path = DRAFT_LOWM.path
    if x.dim() != 2 or not DRAFT_LOWM.covers(((n, k), m)):
        uncovered(path, f"the drafter's {n}x{k} linear past its tactic table's rows")
    elif not x.is_contiguous():
        uncovered(path, "a strided activation (the in-tree GEMM reads it in place)")
    elif x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        missed(path, m, "the activation or weight is not BF16")
    elif (n, k, m, get_env_enable_pdl()) not in _DRAFT_LOWM_BUILT:
        missed(path, m, f"the {n}x{k} kernel at {m} rows was not compiled before warmup")
    else:
        kind, tactic = tactic_for(n, k, m)
        out = torch.empty((m, n), dtype=x.dtype, device=x.device)
        return m3_draft_lowm(x, weight, out, kind, tactic, get_env_enable_pdl())
    return m3_bf16_linear(x, weight)


def _prebuild_draft_lowm(draft: "M3Eagle3Draft") -> int:
    """Compile (and run once) every covered row count of the drafter's linears, outside capture, on the current
    stream. Returns the number of calls compiled."""
    use_pdl = get_env_enable_pdl()
    built = 0
    shapes = {tuple(draft.w[key].shape): draft.w[key] for key in ("qkv", "o", "gu", "dn", "fc")}
    for (n, k), weight in sorted(shapes.items()):
        for _, m in sorted((key for key in DRAFT_LOWM.keys if key[0] == (n, k))):
            if (n, k, m, use_pdl) in _DRAFT_LOWM_BUILT:
                continue
            kind, tactic = tactic_for(n, k, m)
            x = torch.zeros(m, k, dtype=torch.bfloat16, device=weight.device)
            out = torch.empty(m, n, dtype=torch.bfloat16, device=weight.device)
            m3_draft_lowm(x, weight, out, kind, tactic, use_pdl)
            _DRAFT_LOWM_BUILT.add((n, k, m, use_pdl))
            built += 1
    return built


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
        core: MiniMaxM3Tp8ModelingV2,
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
            or (not getattr(cfg, "fc_norm", False))
            or getattr(cfg, "kv_lora_rank", None)
            or (self.capture_layers <= 1)
            or (self.hidden_size_in != self.hidden)
            or (cfg.draft_vocab_size != cfg.vocab_size)
            or getattr(cfg, "attention_bias", False)
            or getattr(cfg, "mlp_bias", False)
            or getattr(cfg, "bias", False)
            or getattr(cfg, "tie_word_embeddings", False)
        ):
            raise ValueError(
                (
                    "this MiniMax-M3 target drafts with a one-layer Llama-style Eagle"
                    "3 drafter with per-layer fc_norms over several captured layers, "
                    "the target's hidden size and vocabulary and no biases"
                )
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
        self._core = [core]
        self.model = _DraftForward(self)

    @property
    def embed_table(self) -> torch.Tensor:
        return self._core[0].w["embed"]

    def load_weights_from_target_model(self, target_model: nn.Module) -> None:
        """Nothing to alias: the drafter reads the target's embedding table through the core and owns its head."""

    def fc(self, hs: torch.Tensor) -> torch.Tensor:
        return _draft_linear(hs, self.w["fc"])

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
        qkv = _draft_linear(x, self.w["qkv"])
        o = m3_draft_attention(self.attn, qkv, md)
        out = _draft_linear(o, self.w["o"])
        return m3_allreduce(out, self.allreduce, None) if reduce else out

    def mlp(self, h: torch.Tensor, reduce: bool) -> torch.Tensor:
        gu = _draft_linear(h, self.w["gu"])
        act = m3_silu_and_mul(gu, None, None, None)
        out = _draft_linear(act, self.w["dn"])
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
                    (
                        "EAGLE3 hidden-state capture token count exceeds buffer capacity:"
                        " num_tokens="
                        f"{num_tokens}"
                        ", capacity="
                        f"{spec_metadata.hidden_states.shape[0]}"
                    )
                )
            if num_tokens > hidden_states.shape[0]:
                raise RuntimeError(
                    (
                        "EAGLE3 hidden-state capture token count exceeds available hidden"
                        " states: num_tokens="
                        f"{num_tokens}"
                        ", hidden_states="
                        f"{hidden_states.shape[0]}"
                    )
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
        and (getattr(lm_head, "tp_size", 1) > 1)
    )


class M3Eagle3Worker(Eagle3OneModelWorker):
    """``Eagle3OneModelWorker`` with the MiniMax-M3 fused verify and fused draft loop, at the in-tree numerics.

    * Verify: on decode-only batches whose verify rows the fused kernels cover (``tail_eligible``), the target's BF16
      lm_head computes this rank's vocabulary shard of the logits (the in-tree head's own local GEMM) and one kernel
      replaces the logits all-gather, the argmax (or the temperature + top-p draw) and the strict acceptance.
    * Draft: the drafter's first pass may take the per-layer fc_norms in one launch (or folded into the hidden-state
      capture) and run its fc beside the target head GEMM. Its BF16 linears run FlashInfer's fixed-tactic low-M
      kernels at the rows ``DRAFT_LOWM`` covers (``_draft_linear``). On a decode-only step ``DRAFT_STEP`` covers,
      the draft loop is fused: each pass's draft input in one launch, the drafter layer's attention and MLP with
      their all-reduces fused with the residual add and the next RMSNorm, and one kernel for the BF16 head, the
      cross-rank argmax and the pass's bookkeeping.

    Every other batch takes the in-tree paths. Methods whose docstring says "Mirrors" copy the named in-tree method
    with the fused branches inserted.
    """

    def __init__(self, spec_config, mapping, use_separate_draft_kv_cache: bool = False) -> None:
        super().__init__(
            spec_config, mapping=mapping, use_separate_draft_kv_cache=use_separate_draft_kv_cache
        )
        self._m3_first_gather_ids: Optional[torch.Tensor] = None
        self._m3_fc_hidden: Optional[torch.Tensor] = None
        self._m3_target_lm_head: Optional[nn.Module] = None
        self._m3_fc_norm_w: Optional[tuple] = None
        self._m3_capture_norm: Optional[tuple] = None
        self._m3_aux_stream: Optional[torch.cuda.Stream] = None
        self._m3_fc_early_drafter: Optional[M3Eagle3Draft] = None
        self._m3_fc_early_stream: Optional[torch.cuda.Stream] = None
        self._m3_force_vec_warm = False
        self._m3_draft_head = None
        self._m3_next_new: Optional[torch.Tensor] = None
        self._m3_next_new_tokens: Optional[torch.Tensor] = None
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
        if drafter is not None and layers and (layer_id == layers[-1]):
            self._m3_fc_early_drafter = None
            self._m3_fc_early_stream = self._m3_fc_fork(
                spec_metadata, spec_metadata.num_capture_tokens, drafter
            )

    def prebuild(self, lm_head: nn.Module, draft: Optional[M3Eagle3Draft]) -> None:
        """The fused draft loop's head copy, the drafter linears' low-M kernels and the drafter attention's KV-split
        cap, from the loaded weights, before the executor's warmup, on every rank."""
        start = time.perf_counter()
        if self.prebuild_draft_head(draft):
            torch.cuda.synchronize()
            logger.info(
                "[M3 prebuild] fused draft head built before warmup: %.1f s"
                % (time.perf_counter() - start)
            )
        start = time.perf_counter()
        built = self.prebuild_drafter_linears(draft)
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
            accepted_tokens, num_accepted_tokens, self._m3_first_gather_ids = (
                self._m3_target_accept(
                    hidden_states, spec_metadata, batch_size, runtime_draft_len, draft_model
                )
            )
        else:
            self._execute_guided_decoder_if_present(logits)
            accepted_tokens, num_accepted_tokens = self.sample_and_accept_draft_tokens(
                input_ids, logits, attn_metadata, spec_metadata
            )
            if self.draft_only_eligible(attn_metadata, spec_metadata, draft_model):
                self._m3_seed_first_pass(num_accepted_tokens, spec_metadata, batch_size)
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
            self._prepare_attn_metadata_for_spec_dec(attn_metadata)
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
            original_all_rank_num_tokens = attn_metadata.all_rank_num_tokens
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
            self._restore_attn_metadata_from_spec_dec(attn_metadata)
            if original_all_rank_num_tokens is not None:
                attn_metadata.all_rank_num_tokens = original_all_rank_num_tokens
        if self._m3_next_new_tokens is not None:
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
        if draft_len == 0:
            return False
        if attn_metadata.num_seqs * (draft_len + 1) > DECODE_MAX_TOKENS:
            uncovered("spec_tail", f"more than {DECODE_MAX_TOKENS} verify rows")
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
        elif attn_metadata.num_seqs * (draft_len + 1) > m3_sampled_accept_kernel.MAX_ROWS:
            uncovered(
                "spec_tail",
                f"a sampled verify of more than {m3_sampled_accept_kernel.MAX_ROWS} rows",
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
            or (draft_model is None)
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
        self, spec_metadata, num_tokens: int, draft_model: Optional[M3Eagle3Draft]
    ) -> Optional[torch.cuda.Stream]:
        """Launch the first draft pass's fc_norm + fc on a forked stream beside the target head GEMM (after everything
        queued on the current stream); the result is picked up by _m3_first_drafter_inputs. Returns the stream to
        join, or None when the fc path does not apply (the first drafter inputs then compute it)."""
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
            and (not (torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing()))
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
        if (
            attn_metadata.num_seqs * (spec_metadata.runtime_draft_len + 1) <= FC_EARLY_MAX_ROWS
            and FINAL_HIDDEN_LAYER_ID not in spec_metadata.layers_to_capture
        ):
            self._m3_fc_early_drafter = draft_model
        fc_norm_w = self._m3_fc_norm_w
        if (
            fc_norm_w is not None
            and fc_norm_w[0] is draft_model.w["fc_norm0"]
            and (len(spec_metadata.layers_to_capture) == fc_norm_w[1].shape[0])
        ):
            self._m3_capture_norm = (fc_norm_w[1], draft_model.rms_eps)

    @staticmethod
    def _m3_sampled_verify_ok(spec_metadata) -> bool:
        """Whether sampled verify rows can use the fused sampled op: only under advanced_sampling_mode=no_topk, the
        deploy-time guarantee that no row uses top-k or min-p (that op applies neither)."""
        mode = getattr(spec_metadata, "advanced_sampling_mode", None)
        return mode is not None and mode.skips_top_k and (not mode.skips_top_p)

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
            is_warmup = spec_metadata.is_cuda_graph and (
                not torch.cuda.is_current_stream_capturing()
            )
            if not torch.cuda.is_current_stream_capturing():
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
        self._m3_next_new = torch.empty(
            (batch_size, draft_len + 1), dtype=torch.int32, device=hidden_states.device
        )
        rows = hidden_states[: batch_size * (draft_len + 1)]
        lm_head = self._m3_target_lm_head
        batch_idx = spec_metadata.batch_indices_cuda[:batch_size]
        fork = self._m3_fc_early_stream
        self._m3_fc_early_stream = None
        if fork is None:
            fork = self._m3_fc_fork(spec_metadata, rows.shape[0], draft_model)
        logits = lm_head.apply_linear(rows, None)
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
        if fork is not None:
            self._m3_fc_join(fork)
        if result is None:
            if torch.cuda.is_current_stream_capturing():
                capture_missed(path, None, "the fused verify's exchange workspace")
            raise RuntimeError(
                "M3 fused verify unavailable after tail_eligible() accepted the step"
            )
        return result

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

    def prebuild_draft_head(self, draft_model: Optional[M3Eagle3Draft]) -> bool:
        """Build the streamed copy of the drafter's BF16 head shard that the fused draft loop reads (rebuilt from the
        current weight values on every call), on every rank. False when the drafter is not one the loop serves."""
        self._m3_draft_head = None
        if draft_model is None or not self._m3_drafter_supported(draft_model):
            return False
        self._m3_draft_head = m3_draft_step_kernel.prepare_head(
            draft_model.lm_head, self.mapping.tp_rank
        )
        return True

    def prebuild_drafter_linears(self, draft_model: Optional[M3Eagle3Draft]) -> int:
        """Compile every drafter linear call ``DRAFT_LOWM`` covers, and cap the drafter attention's trtllm-gen KV
        split per request count (``DRAFT_DECODE_MAX_SEQ_LEN_BY_BATCH``). Called before the executor's warmup, on every
        rank. Returns the number of low-M calls compiled."""
        if draft_model is None:
            return 0
        draft_model.attn.dense_decode_max_seq_len_by_batch = DRAFT_DECODE_MAX_SEQ_LEN_BY_BATCH
        return _prebuild_draft_lowm(draft_model)

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
                ids, pos, hidden = (ids_next, pos_next, to_save)
                if i == 0:
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
    return (draft_config, draft, worker, use_separate)


def _runtime_hooks(spec_fold: Optional[M3SpecStepFold]) -> RuntimeHooks:
    """The step-loop hooks this target opts into, measured on GB300 AgentX decode (tp4 and tp8).

    * ``pack_step_inputs``: a decode step's small input copies go out as one upload and one scatter at the graph
      replay instead of a chain of copies the next graph waits on (~30 us of idle GPU at one request).
    * ``skip_unchanged_h2d``: steady decode re-uploads slot indices, query lengths and page tables that rarely change.
    * ``loop_on_execution_stream``: the overlap loop's own stream work and the forward on one stream, so step N+1's
      inputs wait on graph N in program order rather than through two cross-stream event hops.
    * ``merge_tuned_tactics_across_ranks``: tensor-parallel peers capture the same GEMM tactics (rank-local profiling
      noise otherwise splits them, and the slowest rank sets every step).
    * ``skip_greedy_rng_upload``: all-greedy speculative batches never read the Philox seeds.
    * ``spec_step_fold``: the spec sampler's slot stores and the engine's next-step input gathers in one launch each;
      under DSpark with ``BOUNDARY_PACK``, both inside a batch-1 decode graph's tail.
    * MSA: identity selections for short sequences (and short/long graph families), the one-launch KV-length update,
      unchanged block-table copies skipped, pool views built once.
    * HOSTTAIL (kill switches above ``HOSTTAIL_*``): ``sampler_d2h_side_stream`` (the sampler's store copies on its
      D2H side stream), ``spec_small_h2d`` (static slot ids, packed accepted counts), ``spec_batch_map_changed_only``
      (the drafter's batch_indices / batch-to-slot uploads only on change), ``skip_accepted_draft_upload`` (no reader
      of the accepted counts here), ``pack_spec_rng_uploads`` (Philox uploads in the pack), ``kv_lens_update_once``
      (one MSA KV-length re-derivation per speculative overlap step).
    * ``pack_parallel_scatter`` (``PACK_PSCATTER``): the pack's non-overlapping copies scatter in parallel.
    * ``pack_in_graph`` (``PACK_IN_GRAPH``) / ``pack_early_copy`` (``PACK_EARLY_COPY``): the decode graph's prologue
      issues the pack, from a window uploaded on a side copy stream while the previous graph runs (no H2D copy or
      scatter between the graphs). ``h2d_changed_span`` (``H2D_CHANGED_SPAN``): page-crossing steps stage only the
      changed page-table entries, so their pack still fits the prologue's window.
    * ``spec_tail_bookkeep`` (``TAILBK``): the sampler tail's no-op bookkeeping launches dropped, bitwise.
    """
    return RuntimeHooks(
        pack_step_inputs=True,
        pack_parallel_scatter=PACK_PSCATTER,
        pack_in_graph=PACK_IN_GRAPH,
        pack_early_copy=PACK_EARLY_COPY,
        skip_unchanged_h2d=True,
        h2d_changed_span=H2D_CHANGED_SPAN,
        loop_on_execution_stream=True,
        merge_tuned_tactics_across_ranks=True,
        skip_greedy_rng_upload=True,
        spec_step_fold=spec_fold,
        kv_lens_update_once=HOSTTAIL_KV_LENS_ONCE,
        pack_spec_rng_uploads=HOSTTAIL_PHILOX_PACK,
        spec_small_h2d=HOSTTAIL_SMALL_H2D,
        skip_accepted_draft_upload=HOSTTAIL_SKIP_ACCEPTED_UPLOAD,
        spec_batch_map_changed_only=HOSTTAIL_BATCH_CHANGED_ONLY,
        spec_slot_ids_changed_only=SLOT_IDS_CHANGED,
        sampler_d2h_side_stream=HOSTTAIL_D2H_SIDE,
        skip_unchanged_draft_block_offsets=BLOCKOFF_GUARD,
        spec_tail_bookkeep=TAILBK,
        msa=MsaRuntimePolicy(
            short_sequence_identity=True,
            kv_update=_msa_kv_update_kernel.msa_kv_update,
            skip_unchanged_block_offsets=True,
            reuse_pool_views=True,
        ),
    )


class ModelingV2MiniMaxM31Nvfp4Sm103Tp8(
    DecoderModelForCausalLM[MiniMaxM3Tp8ModelingV2, PretrainedConfig]
):
    """MiniMax-M3 NVFP4 / sm_103 / tp8, text only, with one-model Eagle3 or no speculation.

    The shell owns the vocab-parallel BF16 ``lm_head`` and the logits processor (``DecoderModelForCausalLM``); with
    speculation, the drafter (``M3Eagle3Draft``, its weights from the speculative checkpoint through
    ``load_draft_weights``) and the worker (``M3Eagle3Worker``) that verifies and drafts each step.
    """

    def __init__(self, model_config: ModelConfig[PretrainedConfig]) -> None:
        assert torch.cuda.get_device_capability() == _SM, (
            "this target is built for sm_"
            f"{_SM[0]}"
            f"{_SM[1]}"
            ", the device is sm_"
            f"{''.join(map(str, torch.cuda.get_device_capability()))}"
        )
        set_require(resolved_mode(type(self)) is ModelingV2Mode.REQUIRE)
        model_config = _text_model_config(model_config)
        cfg = model_config.pretrained_config
        spec_config = getattr(model_config, "spec_config", None)
        if (
            model_config.quant_config.kv_cache_quant_algo == QuantAlgo.NVFP4
            and _dspark.draft_kv_override_applies(spec_config)
        ):
            model_config.extra_attrs["draft_kv_cache_quant_algo_override"] = QuantAlgo.FP8
        super().__init__(
            MiniMaxM3Tp8ModelingV2(model_config),
            config=model_config,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )
        self.spec_config = spec_config
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
            if not _dspark.is_standalone_dspark(spec_config):
                spec_fold = M3SpecStepFold()
            elif DSFOLD:
                spec_fold = M3DSparkStepFold(
                    boundary_pack=BOUNDARY_PACK,
                    tokens_per_step=_static_tokens_per_step(spec_config) if BOUNDARY_PACK else None,
                )
        self.boundary_pack_fold = (
            spec_fold
            if isinstance(spec_fold, M3DSparkStepFold) and spec_fold.boundary_pack
            else None
        )
        self.runtime_hooks = _runtime_hooks(spec_fold)
        _log_tailbk(self.spec_worker)
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
        speculation also the fused draft loop's head copy and the drafter linears' low-M kernels."""
        super().cache_derived_state()
        self.model.post_load_weights()
        _prebuild_topp_det()
        _prebuild_softmax_mc()
        _prebuild_rej_mc()
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
        if any((p is not None and getattr(p, "multimodal_data", None) for p in multimodal_params)):
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
            return self.logits_processor.forward(hidden_states, self.lm_head, attn_metadata, False)
        if spec_metadata is not None and spec_metadata.is_layer_capture(FINAL_HIDDEN_LAYER_ID):
            capture(spec_metadata, FINAL_HIDDEN_LAYER_ID, hidden_states, None)
        if attn_metadata.padded_num_tokens is not None:
            hidden_states = hidden_states[: attn_metadata.num_tokens]
        if worker is None:
            return self.logits_processor.forward(
                hidden_states, self.lm_head, attn_metadata, return_context_logits
            )
        if worker.tail_eligible(attn_metadata, spec_metadata, self.lm_head):
            logits = None
        else:
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
        if self.boundary_pack_fold is not None and spec_metadata is not None:
            outputs = self.boundary_pack_fold.graph_tail(outputs, attn_metadata)
        return outputs
