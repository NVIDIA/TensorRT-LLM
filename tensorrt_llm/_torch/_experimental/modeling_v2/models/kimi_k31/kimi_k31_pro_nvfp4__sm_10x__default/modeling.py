# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 Pro NVFP4/MXFP8 target for SM100, SM103, and SM107.

Routed experts execute native NVFP4 SiTU; KDA and shared projections execute native MXFP8.
The target owns its model and loader and uses the family's operations for MLA/DSA, KDA,
attention residuals, DFlash2 and vision. Main FP8/NVFP4 storage, FP8/FP4 index keys and
FP32/BF16 recurrent state are independent. Per-layer FP8 main-cache fallback is preserved.
Attention TP/DP and conditional Helix layouts follow the family routing contract.
"""

from __future__ import annotations

import copy
import math
import os
from collections import deque
from itertools import accumulate
from typing import TYPE_CHECKING, Any, Dict, List, Literal, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from tensorrt_llm._device_copy import DeviceCopyBuffer
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.mla_rope_generation import (
    mla_rope_generation,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.bmm_out import bmm_out
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm

# Catalog interface for the native NVFP4 routed-expert runner called by the engine MoE stack.
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (  # noqa: F401
    fp4_block_scale_moe_runner,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.noaux_tc_op import (
    noaux_tc_op,  # noqa: F401
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (
    flashinfer_fused_add_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.mxfp8_quantize import (  # noqa: F401
    mxfp8_quantize,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.config import (
    KIMI_K3_COMPOSITE_MODEL_TYPES,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.activation.sigmoid_gate_mul import (
    k31_sigmoid_gate_mul,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.dflash2_window import (
    sparse_mla_rows_eager,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.input_staging import (
    stage_scatter,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm.allgather_module import (
    allgather_module,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm.allreduce_module import (
    allreduce_module,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.fused_a_gemm import (
    k31_fused_a_gemm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.dsv3_router_gemm_op import (
    dsv3_router_gemm_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.route_topk import (
    k31_route_topk,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm.attn_res_add_rmsnorm_fwd import (
    attn_res_add_rmsnorm_fwd,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm.attn_res_fwd import (
    attn_res_fwd,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm.attn_res_rmsnorm_fwd import (
    attn_res_rmsnorm_fwd,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.decode_draft_block_table import (
    decode_draft_block_table,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_global_top32 import (
    TOP_K as K31_SELECTOR_TOP_K,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_global_top32 import (
    k31_dflash2_merge_top32,
    k31_dflash2_shard_top32,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_grouped_conv import (
    dflash2_grouped_conv as k31_dflash2_grouped_conv,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_ring import (
    dflash2_ring_chunk_start,
    dflash2_ring_positions,
    dflash2_ring_rows,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_sampling import (
    DFlash2ProposalStore,
    dflash2_sampled_step,
    dflash2_selector_walk,
    dflash2_uniforms,
    greedy_request_mask,
    invalidate_dflash2_proposal_slot,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_selector_greedy_path import (
    k31_dflash2_selector_greedy_path,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_verify import (
    dflash2_sample_and_accept,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_window_attention import (
    dflash2_window_attention as k31_dflash2_window_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_window_attention import (
    dflash2_window_attention_buffers,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_window_rows import (
    dflash2_window_rows as k31_dflash2_window_rows,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.draft_context_cache import (
    DRAFT_FP8_KV_SCALE,
    to_draft_fp8,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.flashinfer_compute_probs_from_logits import (  # noqa: E501
    flashinfer_compute_probs_from_logits,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.flashinfer_sample_from_logits import (
    flashinfer_sample_from_logits,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.fused_compute_probs_from_logits import (
    fused_compute_probs_from_logits,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.fused_sample_from_logits import (
    fused_sample_from_logits,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.mla_norm_rope import (
    dflash2_mla_block_rope as k31_dflash2_mla_block_rope,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.mla_norm_rope import (
    dflash2_mla_ctx_norm_rope as k31_dflash2_mla_ctx_norm_rope,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.strict_accept import (
    strict_accept,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.vocab_parallel_argmax import (
    vocab_parallel_argmax,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.vocab_parallel_sample import (
    vocab_parallel_sample,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.kda_decode import (
    kda_decode,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.kda_mtp_verify import (
    kda_mtp_verify,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.kda_prefill import (
    kda_prefill,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.kda_prefill_conv import (
    kda_prefill_conv,
)
from tensorrt_llm._torch.attention.backends import (
    AttentionForwardArgs,
    AttentionInputType,
    AttentionMetadata,
    TrtllmAttention,
)
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionBackend,
    PositionalEmbeddingParams,
    RopeParams,
)
from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.custom_ops import cute_dsl_kimi_k3_kda_mtp_ops  # noqa: F401
from tensorrt_llm._torch.custom_ops.torch_custom_ops import MXFP8GemmRunner
from tensorrt_llm._torch.distributed import AllReduce, AllReduceFusionOp
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_kimi_k3_vl as _engine_k3
from tensorrt_llm._torch.models import modeling_kimi_k25 as _engine_k25
from tensorrt_llm._torch.models.modeling_kimi_k25 import (
    _MEDIA_PLACEHOLDER_TOKEN_ID,
    DISAGG,
    _has_meta_tensors,
)
from tensorrt_llm._torch.models.modeling_multimodal_mixin import (
    MultimodalEncoderContractError,
    MultimodalModelMixin,
)
from tensorrt_llm._torch.models.modeling_multimodal_utils import (
    find_input_mm_embeds,
    fuse_input_embeds,
    get_multimodal_embeddings,
)
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, MetaInitException
from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel
from tensorrt_llm._torch.moe.fused_moe import (
    ConfigurableMoE,
    SiTuActivation,
    TRTLLMGenFusedMoE,
    create_moe,
)
from tensorrt_llm._torch.moe.fused_moe.routing import DeepSeekV3MoeRoutingMethod
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import (
    eager_on_graph,
    is_in_breakable_cuda_graph,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManager
from tensorrt_llm._torch.pyexecutor.sampler.ops import flashinfer as flashinfer_sampling
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import GREEDY_TEMPERATURE_THRESHOLD
from tensorrt_llm._torch.speculative.dflash import (
    DFlashSpecMetadata,
    VocabParallelDraftTokens,
    dflash_allocated_ctx_limit,
    dflash_noise_block_embedding,
    dflash_position_ceiling,
    last_accepted_hidden,
)
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase
from tensorrt_llm._torch.speculative.vocab_parallel import VocabShardedLogits
from tensorrt_llm._torch.utils import AuxStreamType, get_model_extra_attrs
from tensorrt_llm._utils import get_sm_version, prefer_pinned
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.inputs.multimodal import (
    MultimodalItemPartition,
    MultimodalParams,
    select_multimodal_fields,
)
from tensorrt_llm.inputs.multimodal_data import MEDIA_MODALITIES
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo

from ..ops.activation.situ_and_mul import situ_and_mul
from ..ops.ssm.reset_recurrent_state_rows import reset_recurrent_state_rows
from ..ops.ssm.rms_norm_gated_token_major import rms_norm_gated_token_major
from . import weights as target_weights
from .ops.attention.helix import helix_draft_attention, helix_target_combine
from .ops.ssm.metadata import KimiK3MambaMetadata
from .runtime.causal_lm import TargetCausalLM
from .runtime.draft_attention import HelixDraftShard
from .runtime.draft_backbone import (
    _resolve_dspark_mla_rope_params,
    _yarn_get_mscale,
    build_dspark_mla_yarn_freqs_cis,
)
from .runtime.draft_model import DFlashForCausalLM as _BaseDFlashForCausalLM
from .runtime.draft_model import dflash2_score_edges
from .runtime.draft_worker import DFlashWorker as _BaseDFlashWorker
from .runtime.helix_mapping import setup_helix_mappings
from .weights import _k3_expert_ckpt_spec

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig


# ====================================================================================================================
# KDA layer
# ====================================================================================================================


def kda_core_inplace(hidden_states: torch.Tensor, layer_idx: str, output: torch.Tensor) -> None:
    """Run the metadata-dependent KDA core and write it into ``output``.

    Rows past the live batch (prefill-graph padding) are zeroed: the buffer is graph-pool memory another
    capture may have left non-finite, and the rows reach the MoE router after ``o_proj``.
    """
    attrs = get_model_extra_attrs()
    metadata = attrs["attention_metadata"]()
    kda_layer = attrs["kda_layers"][layer_idx]()
    _kda_forward_core(hidden_states, kda_layer.weight_view, kda_layer.layer_idx, metadata, output)
    if output.shape[0] > metadata.num_tokens:
        output[metadata.num_tokens :].zero_()


maybe_bcg_kda_core_inplace = eager_on_graph(kda_core_inplace)


def _kda_decode_rows(tensor: torch.Tensor) -> torch.Tensor:
    """A one-token decode input as the kernel reads it: the view when its head and channel axes are packed."""
    if tensor.ndim == 4 and tensor.stride(3) == 1 and tensor.stride(2) == tensor.shape[3]:
        return tensor
    return tensor.contiguous()


class _ProjectionWeights(NamedTuple):
    weight: torch.Tensor
    scale: Optional[torch.Tensor]
    alpha: Optional[torch.Tensor]
    columns: int
    cublas: bool = False


def _project(x: torch.Tensor, w: _ProjectionWeights) -> torch.Tensor:
    """BF16 or native E4M3/UE8M0 GEMM over checkpoint tensor views."""
    shape = x.shape
    x = x.reshape(-1, shape[-1])
    if w.scale is None:
        y = cublas_mm(x, w.weight.t()) if w.cublas else F.linear(x, w.weight)
    elif x.numel() == 0:
        y = x.new_empty((x.shape[0], w.columns), dtype=torch.bfloat16)
    else:
        q, scale = torch.ops.trtllm.mxfp8_quantize(x.contiguous(), True)
        gemm = (
            torch.ops.trtllm.mxfp8_mxfp8_gemm_autotuned
            if not torch.compiler.is_compiling() and AutoTuner.get().is_tuning_mode
            else torch.ops.trtllm.mxfp8_mxfp8_gemm
        )
        y = gemm(q, scale, w.weight, w.scale, w.alpha, torch.bfloat16)[:, : w.columns]
    return y.reshape(*shape[:-1], w.columns)


class _MLPWeights(NamedTuple):
    gate_up: _ProjectionWeights
    down: _ProjectionWeights
    beta: float
    linear_beta: Optional[float]
    all_reduce: Optional[AllReduce]


def _dense_mlp(x: torch.Tensor, w: _MLPWeights) -> torch.Tensor:
    gate_up = _project(x, w.gate_up)
    activated = situ_and_mul(gate_up, w.beta, w.linear_beta)
    output = _project(activated, w.down)
    if w.all_reduce is not None:
        output = allreduce_module(output, w.all_reduce, int(AllReduceFusionOp.NONE), None, None)
    return output


class _KDAWeights(NamedTuple):
    qkvg: _ProjectionWeights
    bfa: _ProjectionWeights
    fb: torch.Tensor
    output: _ProjectionWeights
    conv: torch.Tensor
    conv_decode: tuple[torch.Tensor, ...]
    conv_verify: tuple[torch.Tensor, ...]
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    norm: torch.Tensor
    norm_f32: torch.Tensor
    zero_bias: torch.Tensor
    heads: int
    head_dim: int
    eps: float
    lower_bound: float


def _kda_forward_core(x, w: _KDAWeights, layer_idx, metadata, output=None):
    """Packed prefill, one-token decode and replay verification on manager-owned pools."""
    mm = metadata.mamba_metadata
    cache = metadata.kv_cache_manager.mamba_layer_cache(layer_idx)
    conv, state = cache.conv, cache.temporal
    has_replay_caches = getattr(cache, "has_kda_replay_caches", False)
    contexts, ctx_tokens, live = metadata.num_contexts, metadata.num_ctx_tokens, metadata.num_tokens
    batch = metadata.seq_lens.shape[0]
    slots = mm.state_indices[:batch]
    gen_slots = getattr(mm, "generation_state_indices", None)
    if gen_slots is None:
        gen_slots = slots[contexts:]
    h, d = w.heads, w.head_dim
    width = h * d
    if output is None:
        output = x.new_empty((x.shape[0], h, d))
    if contexts:
        ctx_slots = slots[:contexts]
        initial = mm.has_initial_states[:contexts]
        if ctx_tokens == 0:
            reset_recurrent_state_rows(state, ctx_slots, initial, conv)
        else:
            packed = _project(x[:ctx_tokens], w.qkvg)
            bfa = _project(x[:ctx_tokens], w.bfa)
            forget = F.linear(bfa[:, :d], w.fb).view(1, ctx_tokens, h, d)
            beta = bfa[:, d : d + h].unsqueeze(0)
            reset_recurrent_state_rows(state, ctx_slots, initial)
            q, k, v = kda_prefill_conv(
                packed,
                w.conv,
                conv,
                mm.query_start_loc[: contexts + 1],
                ctx_slots,
                initial,
                getattr(mm, "kda_max_seqlen", ctx_tokens),
            )
            core = kda_prefill(
                q,
                k,
                v,
                forget,
                beta,
                state,
                ctx_slots,
                mm.query_start_loc_long[: contexts + 1],
                w.a_log,
                w.dt_bias,
                d**-0.5,
                w.lower_bound,
                "bt16",
                getattr(mm, "kda_max_seqlen", 0),
                getattr(mm, "kda_seq_order", None),
                getattr(mm, "kda_cu_chunks", None),
            )
            output[:ctx_tokens].copy_(
                rms_norm_gated_token_major(
                    core.reshape(-1, d),
                    packed[:, 3 * width :].reshape(-1, h, d),
                    w.norm,
                    w.eps,
                    gate_activation="sigmoid",
                ).reshape(ctx_tokens, h, d)
            )
        if has_replay_caches:
            cache.commit_conv_window(ctx_slots, conv)
    generations = batch - contexts
    if generations:
        rows = live - ctx_tokens
        if rows % generations:
            raise ValueError("KDA generation rows must contain a uniform number of tokens")
        steps = rows // generations
        packed = _project(x[ctx_tokens:live], w.qkvg)
        bfa = _project(x[ctx_tokens:live], w.bfa)
        forget = F.linear(bfa[:, :d], w.fb).view(1, rows, h, d)
        beta = bfa[:, d : d + h].unsqueeze(0)
        q, k, v, gate = (
            packed[:, j * width : (j + 1) * width].view(1, rows, h, d) for j in range(4)
        )
        if steps == 1:
            kernel_state, kernel_conv, kernel_slots = state, conv, gen_slots
            if state.dtype == torch.bfloat16:
                # Decode accumulates in FP32; round only when committing the selected rows.
                kernel_state = state.index_select(0, gen_slots.long()).float().contiguous()
                kernel_conv = conv.index_select(0, gen_slots.long())
                kernel_slots = None
            kda_decode(
                _kda_decode_rows(q),
                _kda_decode_rows(k),
                _kda_decode_rows(v),
                *w.conv_decode,
                w.zero_bias,
                w.zero_bias,
                w.zero_bias,
                kernel_conv[:, :width],
                kernel_conv[:, width : 2 * width],
                kernel_conv[:, 2 * width :],
                w.a_log,
                _kda_decode_rows(forget),
                w.dt_bias,
                beta if beta.stride(2) == 1 else beta.contiguous(),
                _kda_decode_rows(gate),
                w.norm_f32,
                kernel_slots,
                kernel_state,
                True,
                True,
                True,
                True,
                w.lower_bound,
                d**-0.5,
                w.eps,
                output[ctx_tokens:live].view(generations, 1, h, d),
            )
            if state.dtype == torch.bfloat16:
                state.index_copy_(0, gen_slots.long(), kernel_state.to(state.dtype))
                conv.index_copy_(0, gen_slots.long(), kernel_conv)
            if has_replay_caches:
                cache.commit_conv_window(gen_slots, conv)
        else:
            if not has_replay_caches:
                raise ValueError("Kimi K3.1 verification requires KDA replay caches")
            args = dict(
                x_q=q,
                x_k=k,
                x_v=v,
                w_q=w.conv_verify[0],
                w_k=w.conv_verify[1],
                w_v=w.conv_verify[2],
                cs_q=cache.kda_conv_q,
                cs_k=cache.kda_conv_k,
                cs_v=cache.kda_conv_v,
                g=forget,
                beta=beta,
                A_log=w.a_log,
                dt_bias=w.dt_bias,
                recurrent_state=state,
                k_cache=cache.kda_k_cache,
                g_cache=cache.kda_g_cache,
                v_cache=cache.kda_v_cache,
                beta_cache=cache.kda_beta_cache,
                ssm_state_indices=gen_slots,
                num_spec=steps - 1,
                num_accepted_tokens=cache.prev_num_accepted_tokens,
                lower_bound=w.lower_bound,
                scale=d**-0.5,
                onorm_g=gate,
                onorm_weight=w.norm_f32,
                onorm_eps=w.eps,
            )
            if state.dtype == torch.float32:
                core = kda_mtp_verify(**args)
            else:
                core = torch.ops.trtllm.kda_mtp_decode(
                    **args, cu_seqlens=None, packed_token_layout=True, fuse_output_norm=True
                )
            output[ctx_tokens:live].copy_(core.reshape(rows, h, d))
    output[live:].zero_()
    return output


# ====================================================================================================================
# MLA attention (DSA sparse MLA through the engine attention backend)
# ====================================================================================================================


def padded_attention_heads(num_heads: int) -> int:
    """Query heads the sparse MLA kernel runs for ``num_heads`` heads per rank.

    TRTLLM-Gen's sparse MLA kernel runs a head group larger than 64 in Q64 tiles, which must divide
    the group: 96 heads per rank (attention TP 1, every attention-DP layout) have no tile. Such a group
    is padded to the next multiple of 64 (128); the extra query heads are zero, and their outputs are
    dropped before the V absorption.
    """
    if num_heads > 64 and num_heads % 64:
        return (num_heads + 63) // 64 * 64
    return num_heads


class _IndexerWeights(NamedTuple):
    query: torch.Tensor
    key_score: torch.Tensor
    norm: torch.Tensor
    bias: torch.Tensor
    eps: float


def _indexer_project(indexer, qr, hidden_states):
    w = indexer.weight_view
    key, scores = F.linear(hidden_states, w.key_score).split(
        (indexer.head_dim, indexer.n_heads), dim=-1
    )
    query = cublas_mm(qr, w.query).reshape(-1, indexer.n_heads, indexer.head_dim)
    key = F.layer_norm(key.float(), (indexer.head_dim,), w.norm, w.bias, w.eps).to(
        hidden_states.dtype
    )
    half = indexer.head_dim // 2

    def quantize(x):
        a, b = x.split((half, indexer.head_dim - half), dim=-1)
        if indexer.use_fp4:
            return torch.ops.trtllm.fused_cat_fp4(a, b)
        return torch.ops.trtllm.fused_cat_fp8(a, b, True)

    (q, qs), (k, ks) = maybe_execute_in_parallel(
        lambda: quantize(query),
        lambda: quantize(key),
        indexer.ln_events[0],
        indexer.ln_events[1],
        indexer.aux_stream,
    )
    qs = qs.reshape(-1, indexer.n_heads, 1)
    if indexer.use_fp4:
        q = q.reshape(-1, indexer.n_heads, indexer.head_dim // 2)
        scores = scores.float() * indexer.weight_scale_factor
    else:
        q = q.reshape(-1, indexer.n_heads, indexer.head_dim)
        scores = scores * qs.squeeze(-1) * indexer.weight_scale_factor
    return q, k, ks, scores, qs


class _MLAWeights(NamedTuple):
    qkva: torch.Tensor
    q_norm: torch.Tensor
    kv_norm: torch.Tensor
    qb: torch.Tensor
    gate: Optional[torch.Tensor]
    output: torch.Tensor


def _mla_project(mla, hidden_states, position_ids=None):
    w = mla.weight_view
    q, kv, k_pe = cublas_mm(hidden_states, w.qkva).split(
        (mla.q_lora_rank, mla.kv_lora_rank, mla.qk_rope_head_dim), dim=-1
    )
    q, kv = maybe_execute_in_parallel(
        lambda: flashinfer_rmsnorm(q, w.q_norm, mla.rms_norm_eps),
        lambda: flashinfer_rmsnorm(kv, w.kv_norm, mla.rms_norm_eps),
        mla.ln_events[0],
        mla.ln_events[1],
        mla.aux_stream,
    )
    latent = torch.cat((kv, k_pe), dim=-1)
    query = F.linear(q, w.qb)
    if mla.mqa.indexer is None:
        return [query, kv, k_pe, latent]
    index = _indexer_project(mla.mqa.indexer, q, hidden_states)
    return [query, kv, k_pe, latent, *index]


def _mla_attn_forward_gen(
    self,
    attn_backend: AttentionBackend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    attn_metadata: AttentionMetadata,
    **kwargs,
):
    """Run absorbed attention and combine the CP-owned pages under Helix."""
    if self.mapping.has_cp_helix():
        stats = torch.empty(
            (q.shape[0], self.num_heads_attn, 2), device=q.device, dtype=torch.float32
        )
        kwargs["softmax_stats_tensor"] = stats
        partial = attn_backend.forward(
            q, k, v, attn_metadata, forward_args=AttentionForwardArgs(**kwargs)
        )
        return helix_target_combine(
            partial,
            stats,
            self.mapping,
            self.num_heads_tp,
            self.kv_lora_rank,
            attn_metadata,
            self.aux_stream,
            self.ln_events,
        )
    return attn_backend.forward(
        q,
        k,
        v,
        attn_metadata,
        forward_args=AttentionForwardArgs(**kwargs),
    )


def _mla_new_fused_q(self, q: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """``[num_tokens, num_heads_attn, kv_lora_rank + qk_rope_head_dim]``; the padding heads are zero."""
    fused_q = torch.empty(
        [num_tokens, self.num_heads_attn, self.kv_lora_rank + self.qk_rope_head_dim],
        dtype=q.dtype,
        device=q.device,
    )
    if self.num_heads_attn != self.num_heads_tp:
        fused_q[:, self.num_heads_tp :].zero_()
    return fused_q


def _mla_padded_q_pe(self, q_pe: torch.Tensor) -> torch.Tensor:
    """``q_pe`` with zero rows for the padding heads (the backend reads num_heads_attn heads)."""
    if self.num_heads_attn == self.num_heads_tp:
        return q_pe
    padded = q_pe.new_zeros([q_pe.shape[0], self.num_heads_attn, self.qk_rope_head_dim])
    padded[:, : self.num_heads_tp].copy_(q_pe)
    return padded


def _mla_attention_heads_out(self, attn_out_latent: torch.Tensor) -> torch.Tensor:
    """``[seq, num_heads_tp, kv_lora_rank]`` view of the backend output (padding heads dropped)."""
    if self.mapping.has_cp_helix():
        return attn_out_latent.view(-1, self.num_heads_tp_cp, self.kv_lora_rank)
    assert attn_out_latent.shape[1] == self.num_heads_attn * self.kv_lora_rank
    out = attn_out_latent.view([-1, self.num_heads_attn, self.kv_lora_rank])
    return out[:, : self.num_heads_tp_cp]


def _mla_rope_generation(
    self,
    fused_q: torch.Tensor,
    q_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    metadata: AttentionMetadata,
    cu_q_seqlens: torch.Tensor,
    cu_kv_seqlens: torch.Tensor,
    fmha_scheduler_counter: torch.Tensor,
    mla_bmm1_scale: Optional[torch.Tensor],
    mla_bmm2_scale: Optional[torch.Tensor],
    quant_q_buffer: Optional[torch.Tensor],
) -> None:
    """The backend's MLA generation RoPE / latent-cache append, through the catalog entry.

    The arguments the TRTLLM backend's ``mla_rope_generation`` passes on this target's path: an NVFP4, FP8 or BF16
    latent cache (no dense FP4 MLA), no folded KV norm, no precomputed scheduler, both halves in one launch.
    """
    mqa = self.mqa
    assert mqa.is_mla_enable and mqa.mla_params is not None
    assert metadata.kv_cache_manager is not None
    assert not mqa.uses_fp4_mla_attention
    # The kernel reads the RoPE cos/sin table up to the longest sequence.
    mqa._ensure_rope_table_size(metadata.max_seq_len)
    helix_tensor_params = [metadata.helix_position_offsets, metadata.helix_is_inactive_rank]
    if metadata._helix_spec_tokens_valid:
        helix_tensor_params.append(metadata.helix_local_slots)
    kv_cache_manager = metadata.kv_cache_manager
    # The kernel derives each token's request and KV slot from one per-request token count, so the
    # batch is presented token-major as for the attention call.
    with mqa.presented_token_major_for(metadata):
        mla_rope_generation(
            fused_q,
            q_pe,
            latent_cache,
            mqa.rotary_cos_sin,
            cu_q_seqlens,
            cu_kv_seqlens,
            fmha_scheduler_counter,
            mla_bmm1_scale,
            mla_bmm2_scale,
            quant_q_buffer,
            metadata.kv_lens_cuda_runtime,
            metadata.kv_lens_runtime,
            metadata.prompt_lens_cpu_runtime,
            metadata.num_contexts,
            metadata.kv_cache_block_offsets,
            kv_cache_manager.kv_cache_pool_pointers,
            kv_cache_manager.kv_cache_pool_mapping,
            None,
            None,
            getattr(mqa, "_nvfp4_mla_kv_scale_orig_quant", None),
            None,
            metadata.block_ids_per_seq,
            helix_tensor_params,
            mqa.predicted_tokens_per_seq,
            mqa.get_local_layer_idx(metadata),
            mqa.num_heads,
            mqa.num_kv_heads,
            mqa.head_dim,
            getattr(kv_cache_manager, "mla_kv_cache_residual_dim", 0),
            kv_cache_manager.tokens_per_block,
            metadata.max_seq_len,
            metadata.beam_width,
            mqa.quant_mode,
            mqa.q_scaling,
            mqa.q_lora_rank,
            mqa.kv_lora_rank,
            mqa.qk_nope_head_dim,
            mqa.qk_rope_head_dim,
            mqa.v_head_dim,
            mqa.rope_append,
        )


def _mla_bmm_bf16_out(self, a, b_no_transpose, b_transposed, output):
    """BF16 absorption BMM (cuBLAS): ``output = a @ b_transposed`` per head."""
    bmm_out(a, b_transposed, output)


def _mla_forward_absorption_generation(
    self,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    attn_metadata: AttentionMetadata,
    output: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    latent_cache: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    num_tokens = q.shape[0]
    q_nope, q_pe = q.view([-1, self.num_heads_tp, self.qk_head_dim]).split(
        [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    )

    # fused_q contains 1) the result of the following bmm with shape [num_tokens, num_heads, kv_lora_rank]
    # 2) rope(q_pe) with shape [num_tokens, num_heads, qk_rope_head_dim]. rope is applied inside AttentionOp
    num_seqs = attn_metadata.num_seqs

    # The rope kernel writes seqQOffset[batch_idx + 1] for every
    # batch_idx < batch_size, and CUB-scans total_s_len / seq_len entries
    # into seqKVOffsets (mlaKernels.cu:579, :650). `batch_size` there is the
    # count of presented rows, which equals num_seqs only while every
    # request contributes the same number of query tokens. Under the
    # token-major presentation a row is a token, so size these by the token
    # count. Uniform verification keeps the established request-major
    # allocation rather than scaling these temporary buffers by K.
    cu_seqlens_rows = (
        max(num_seqs, num_tokens) + 1
        if getattr(attn_metadata, "is_ragged_verify", False)
        else num_seqs + 1
    )
    cu_q_seqlens = torch.empty(cu_seqlens_rows, dtype=torch.int32, device=q.device)
    cu_kv_seqlens = torch.empty(cu_seqlens_rows, dtype=torch.int32, device=q.device)
    fmha_scheduler_counter = torch.empty(1, dtype=torch.uint32, device=q.device)
    use_fp8_mla = self.mqa.has_fp8_kv_cache or self.mqa.has_fp4_kv_cache

    mla_bmm1_scale = None
    mla_bmm2_scale = None
    quant_q_buffer = None
    if use_fp8_mla:
        mla_bmm1_scale = torch.empty(2, dtype=torch.float32, device=q.device)
        mla_bmm2_scale = torch.empty(1, dtype=torch.float32, device=q.device)
        quant_q_buffer = torch.empty(
            num_tokens,
            self.num_heads_attn,
            (self.kv_lora_rank + self.qk_rope_head_dim),
            dtype=torch.uint8,
            device=q.device,
        )
        if self.num_heads_attn != self.num_heads_tp:
            quant_q_buffer[:, self.num_heads_tp :].zero_()
    q_pe_attn = _mla_padded_q_pe(self, q_pe)

    if hasattr(self, "k_b_proj_trans"):
        fused_q = _mla_new_fused_q(self, q, num_tokens)

        def _mla_gen_rope():
            if self.apply_rotary_emb:
                # Non-fused backends (Vanilla / FlashInfer) do not fuse RoPE
                # in the attention kernel. Reuse apply_rope, which rotates
                # q's q_pe slice in place, then copy rotated k_pe into the
                # latent cache that the backend appends.
                assert position_ids is not None
                assert latent_cache is not None
                gen_position_ids = position_ids[..., attn_metadata.num_ctx_tokens :]
                k_pe_rope = self.apply_rope(q, k_pe, gen_position_ids)
                fused_q[:, : self.num_heads_tp, self.kv_lora_rank :] = q_pe
                latent_cache[..., self.kv_lora_rank :] = k_pe_rope
            else:
                # Fused backend (TRTLLM): RoPE, latent-cache append and the
                # trtllm-gen scheduler buffers are all produced in-kernel.
                _mla_rope_generation(
                    self,
                    fused_q,
                    q_pe_attn,
                    latent_cache,
                    attn_metadata,
                    cu_q_seqlens,
                    cu_kv_seqlens,
                    fmha_scheduler_counter,
                    mla_bmm1_scale,
                    mla_bmm2_scale,
                    quant_q_buffer,
                )

        if use_fp8_mla:
            rope_stream = None
        else:
            rope_stream = self.aux_stream
        # [num_heads, num_tokens, self.qk_nope_head_dim]
        q_nope_t = q_nope.transpose(0, 1)
        # [num_heads, num_tokens, self.kv_lora_rank]
        q_nope_out = fused_q[:, : self.num_heads_tp, : self.kv_lora_rank].transpose(0, 1)

        # [num_heads, num_tokens, self.qk_nope_head_dim] x [num_heads, kv_lora_rank, qk_nope_head_dim]
        # -> [num_heads, num_tokens, kv_lora_rank] -> [num_tokens, num_heads, kv_lora_rank]
        # The output of bmm is written directly into fused_q
        maybe_execute_in_parallel(
            lambda: _mla_bmm_bf16_out(
                self,
                q_nope_t,
                self.k_b_proj_trans,
                self.k_b_proj_trans.transpose(1, 2),
                q_nope_out,
            ),
            _mla_gen_rope,
            self.ln_events[0],
            self.ln_events[1],
            rope_stream,
        )

        fused_q = fused_q.view(
            [
                num_tokens,
                self.num_heads_attn * (self.kv_lora_rank + self.qk_rope_head_dim),
            ]
        )
    else:
        raise RuntimeError("MLA absorption requires k_b_proj_trans")

    attn_out_latent = _mla_attn_forward_gen(
        self,
        self.mqa,
        fused_q,
        None,
        None,
        position_ids,
        attn_metadata,
        attention_input_type=AttentionInputType.generation_only,
        out_scale=self.out_scale,
        latent_cache=latent_cache,  # kvcache and k_pe
        q_pe=q_pe_attn,  # used by `invokeMLARopeGeneration`
        cu_q_seqlens=cu_q_seqlens,  # used by `mlaGeneration`
        cu_kv_seqlens=cu_kv_seqlens,  # used by `mlaGeneration`
        fmha_scheduler_counter=fmha_scheduler_counter,  # used by `mlaGeneration`
        mla_bmm1_scale=mla_bmm1_scale,  # used by `mlaGeneration`
        mla_bmm2_scale=mla_bmm2_scale,  # used by `mlaGeneration`
        quant_q_buffer=quant_q_buffer,  # used by `mlaGeneration`
        **kwargs,
    )
    fused_q = None

    assert attn_out_latent.shape[0] == q.shape[0]
    # [seq, num_heads, kv_lora_rank]
    attn_out_latent = _mla_attention_heads_out(self, attn_out_latent)

    attn_output = output.view([num_tokens, self.num_heads_tp_cp, self.v_head_dim])

    # [num_heads, seq, kv_lora_rank] x [num_heads, kv_lora_rank, v_head_dim]
    # -> [num_heads, seq, v_head_dim]
    _mla_bmm_bf16_out(
        self,
        attn_out_latent.transpose(0, 1),
        self.v_b_proj,
        self.v_b_proj.transpose(1, 2),
        attn_output.transpose(0, 1),
    )

    return output


def _mla_forward_absorption_context(
    self,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    attn_metadata: AttentionMetadata,
    output: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    latent_cache: Optional[torch.Tensor] = None,
    q_rope_applied: bool = False,
    **kwargs,
) -> torch.Tensor:
    num_tokens = q.shape[0]

    q_nope, q_pe = q.view([-1, self.num_heads_tp, self.qk_head_dim]).split(
        [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    )

    if not hasattr(self, "k_b_proj_trans"):
        raise RuntimeError("MLA absorption requires k_b_proj_trans")

    # fused_q contains 1) the result of the following bmm with shape [num_tokens, num_heads, kv_lora_rank]
    # 2) rope(q_pe) with shape [num_tokens, num_heads, qk_rope_head_dim]. rope is applied inside AttentionOp
    fused_q = _mla_new_fused_q(self, q, num_tokens)
    q_pe_attn = _mla_padded_q_pe(self, q_pe)

    # [num_heads, num_tokens, self.qk_nope_head_dim]
    q_nope_t = q_nope.transpose(0, 1)
    # [num_heads, num_tokens, self.kv_lora_rank]
    q_nope_out = fused_q[:, : self.num_heads_tp, : self.kv_lora_rank].transpose(0, 1)

    # [num_heads, num_tokens, self.qk_nope_head_dim] x [num_heads, kv_lora_rank, qk_nope_head_dim]
    # -> [num_heads, num_tokens, kv_lora_rank] -> [num_tokens, num_heads, kv_lora_rank]
    # The output of bmm is written directly into fused_q
    _mla_bmm_bf16_out(
        self,
        q_nope_t,
        self.k_b_proj_trans,
        self.k_b_proj_trans.transpose(1, 2),
        q_nope_out,
    )

    if self.apply_rotary_emb or q_rope_applied:
        # RoPE on q_pe was applied outside the attention kernel.
        fused_q[:, : self.num_heads_tp, self.kv_lora_rank :] = q_pe
    fused_q = fused_q.view(
        [
            num_tokens,
            self.num_heads_attn * (self.kv_lora_rank + self.qk_rope_head_dim),
        ]
    )

    attn_out_latent = _mla_attn_forward_gen(
        self,
        self.mqa,
        fused_q,
        None,
        None,
        position_ids,
        attn_metadata,
        attention_input_type=AttentionInputType.context_only,
        out_scale=self.out_scale,
        latent_cache=latent_cache,  # kvcache and k_pe
        q_pe=q_pe_attn,  # used by applyMLARopeAndAssignQKVKernelOptContext
        **kwargs,
    )
    fused_q = None

    assert attn_out_latent.shape[0] == q.shape[0]
    # [seq, num_heads, kv_lora_rank]
    attn_out_latent = _mla_attention_heads_out(self, attn_out_latent)

    attn_output = output.view([num_tokens, self.num_heads_tp_cp, self.v_head_dim])

    # [num_heads, seq, kv_lora_rank] x [num_heads, kv_lora_rank, v_head_dim]
    # -> [num_heads, seq, v_head_dim]
    _mla_bmm_bf16_out(
        self,
        attn_out_latent.transpose(0, 1),
        self.v_b_proj,
        self.v_b_proj.transpose(1, 2),
        attn_output.transpose(0, 1),
    )

    return output


# ====================================================================================================================
# Kimi K3 MLA (NoPE, output gate)
# ====================================================================================================================


def _make_pos_embd_params(
    *,
    qk_rope_head_dim: int,
    max_position_embeddings: int,
) -> PositionalEmbeddingParams:
    """Build a valid rope config so the backend allocates a real cache.

    We use rope_gpt_neox with default theta=10000 and ``duplicate_data
    =True`` (the same convention DeepSeek-V3-style MLA uses when
    ``qk_rope_head_dim`` is present). The resulting ``rotary_cos_sin``
    has the exact shape the C++ MLA rope kernel indexes. Immediately
    after backend construction we overwrite the tensor values with
    ``(cos=1, sin=0)`` — an identity rotation, matching K3's NoPE.
    """
    rope_params = RopeParams(
        dim=qk_rope_head_dim,
        theta=10000.0,
        max_positions=max_position_embeddings,
        original_max_positions=max_position_embeddings,
        duplicate_data=True,
    )
    return PositionalEmbeddingParams(
        type=PositionEmbeddingType.rope_gpt_neox,
        rope=rope_params,
        # Match the working DeepSeek-V3-style MLA reference test
        # (tests/unittest/_torch/attention/test_attention_mla.py) which
        # sets ``is_neox=False``. The MLA fused rope kernel is GPT-J
        # style regardless of this flag, but the C++ FMHA reads this bit
        # elsewhere and stability under identity-cos-sin depends on the
        # standard non-neox layout.
        is_neox=False,
    )


def _write_identity_rope_values(cos_sin: torch.Tensor) -> None:
    """Overwrite a rotary cos/sin table with identity values in place.

    Interleaved (cos, sin) pairs: index [::2] = cos, [1::2] = sin.
    Setting cos=1 and sin=0 per position makes the rotation the
    identity — a mathematical no-op — which preserves K3's NoPE
    semantics without patching the backend.
    """
    flat = cos_sin.reshape(-1)
    with torch.no_grad():
        flat[0::2] = 1.0
        flat[1::2] = 0.0
    # Ensure the identity write reaches CUDA memory before any kernel
    # launched from a different stream can read the table.
    if cos_sin.is_cuda:
        torch.cuda.synchronize(cos_sin.device)


def _install_identity_rope_table(backend: TrtllmAttention) -> None:
    """Install an identity rotary cos/sin table on ``backend``.

    The C++ MLA rope kernels (``mla_rope_generation`` and the context
    preprocess) read this table and apply the rotation; identity values
    make that a copy, preserving K3's NoPE.

    The tensor SHAPE produced by ``create_rope_const_params`` is kept
    intact so the C++ ``float2`` indexing stays valid. Only the values
    are overwritten in place. ``_ensure_rope_table_size`` is replaced
    with an identity-preserving resize: the table may GROW (so the
    fused rope-generation op can never index out of bounds for long
    sequences) but its values are always rewritten to identity right
    after a regeneration, so the real sinusoids never leak in.
    """
    cos_sin = backend.rotary_cos_sin
    if cos_sin is None:
        raise RuntimeError(
            "backend.rotary_cos_sin is None after construction; check "
            "pos_embd_params has a valid RopeParams with dim > 0."
        )
    _write_identity_rope_values(cos_sin)

    orig_resize = backend._ensure_rope_table_size  # bound method

    def _identity_preserving_resize(required_max_positions: int) -> None:
        if required_max_positions <= backend.rope_params.max_positions:
            return
        orig_resize(required_max_positions)
        _write_identity_rope_values(backend.rotary_cos_sin)

    backend._ensure_rope_table_size = _identity_preserving_resize


# ---------------------------------------------------------------------------
# KimiK3MLAAttention.
# ---------------------------------------------------------------------------


# ====================================================================================================================
# Text model: attention residuals, latent MoE, decoder layer, model, causal LM
# ====================================================================================================================


# Identity-RoPE table positions for the MLA backends. K3 is NoPE (the table
# holds cos=1/sin=0), but the chunked-context path indexes the table by
# absolute position, so it must cover max_position_embeddings (~512MB per
# backend for the 1M-position checkpoint); a smaller table is read out of
# bounds. KIMI_K3_MLA_MAX_POSITIONS overrides the size for short-context
# deployments.
_KIMI_K3_MLA_MAX_POSITIONS_ENV = "KIMI_K3_MLA_MAX_POSITIONS"


class _K31RoutingMethod(DeepSeekV3MoeRoutingMethod):
    """The stock DeepSeek-V3 routing; with ``route_tail`` (set by ``KimiK3MoERuntime`` under attention DP) a
    65..2048-row call runs the routing-tail kernel instead, which selects the same experts in the same order and
    returns noaux_tc's weights already rounded to BF16, bit for bit."""

    route_tail = False

    def apply(self, logits: torch.Tensor, input_ids: Optional[torch.Tensor] = None):
        m = logits.shape[0]
        if (
            self.route_tail
            and 64 < m <= 2048
            and logits.dtype == torch.float32
            and logits.dim() == 2
            and logits.is_contiguous()
            and not torch.compiler.is_compiling()
        ):
            return k31_route_topk(
                logits, self.e_score_correction_bias, float(self.routing_impl.routed_scaling_factor)
            )
        return super().apply(logits, input_ids)


def _resolve_kimi_situ_betas(cfg: Any) -> tuple[float, float]:
    """Return the finite SiTu betas required by the routed-expert kernels."""
    config_situ_beta = getattr(cfg, "activation_situ_beta", None)
    situ_beta = 1.0 if config_situ_beta is None else config_situ_beta
    situ_linear_beta = getattr(cfg, "activation_situ_linear_beta", None)
    if situ_linear_beta is None:
        raise ValueError(
            "Kimi K3 routed SiTu experts require activation_situ_linear_beta; "
            "None means an identity linear branch that the fused kernels cannot represent."
        )
    if situ_beta <= 0 or situ_linear_beta <= 0:
        raise ValueError(
            f"Kimi K3 SiTu betas must be positive; got {situ_beta} and {situ_linear_beta}."
        )
    return float(situ_beta), float(situ_linear_beta)


# ---------------------------------------------------------------------------
# Config helpers.
# ---------------------------------------------------------------------------


def _get_text_config(pretrained_config: "PretrainedConfig"):
    """Return the Kimi text config, unwrapping a composite K3-family config."""
    if getattr(pretrained_config, "model_type", None) in KIMI_K3_COMPOSITE_MODEL_TYPES or (
        not hasattr(pretrained_config, "linear_attn_config")
        and hasattr(pretrained_config, "text_config")
    ):
        return pretrained_config.text_config
    return pretrained_config


def _is_kda_layer(cfg, layer_idx: int) -> bool:
    return (layer_idx + 1) in cfg.linear_attn_config["kda_layers"]


def _is_mla_layer(cfg, layer_idx: int) -> bool:
    return (layer_idx + 1) in cfg.linear_attn_config["full_attn_layers"]


# ---------------------------------------------------------------------------
# attn_res: per-token snapshot mixing (HF `_apply_attn_res`).
# ---------------------------------------------------------------------------


KIMI_K3_AUX_ATTN_RES_STREAM_ENV = "KIMI_K3_AUX_ATTN_RES_STREAM"
"""Which residual-stream value the drafter's hidden-state tap captures.

``1`` (default) captures the pre-norm attn_res mixture -- the value the next
consumer actually reads. ``0`` captures the raw running prefix sum instead.

Both conventions exist and a drafter distilled against one accepts fewer
drafts on the other with nothing raised, so this is a property of the DRAFTER
checkpoint, not a performance knob. The DFlash2 drafter is distilled on the
mixture."""

_AUX_ATTN_RES_STREAM_ENABLED = os.environ.get(KIMI_K3_AUX_ATTN_RES_STREAM_ENV, "1") == "1"

KIMI_K3_FUSED_ATTN_RES_ENV = "KIMI_K3_FUSED_ATTN_RES"
"""Set to ``0`` to disable the in-tree fused Torch op
``trtllm::attn_res_fwd`` (Blackwell only). Default: fused with fallback."""

_FUSED_ATTN_RES_ENABLED = os.environ.get(KIMI_K3_FUSED_ATTN_RES_ENV, "1") == "1"


# Largest per-rank token count for which the fused attn_res + RMSNorm epilogue ops run (the
# single-token decode shape); larger token counts take attn_res_fwd and a separate RMSNorm.
_FUSED_ATTN_RES_MAX_TOKENS = 1


# Token limit of trtllm::attn_res_fwd (cpp/tensorrt_llm/thop/attnResOp.cpp).
_ATTN_RES_OP_MAX_TOKENS = 16384


def _apply_attn_res_fused(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: target_weights.TensorWeights,
    norm: target_weights.TensorWeights,
) -> Optional[torch.Tensor]:
    """Fused attn_res via the in-tree ``trtllm::attn_res_fwd`` op.

    Returns ``None`` when the call falls outside the fused kernel's
    contract (dtype/shape/arch) so the caller can use the exact fp32 reference
    instead. ``block_residual`` is kept in the kernel-native ``[K, M, H]``
    layout. Candidate order matches the reference: snapshots first, the
    running prefix sum last.
    """
    if (
        prefix_sum.dtype is not torch.bfloat16
        or not prefix_sum.is_cuda
        or not block_residual.is_cuda
    ):
        return None
    M, H = prefix_sum.shape
    K = int(block_residual.shape[0])
    if K + 1 > 12 or not (4096 <= H <= 8192 and H % 1024 == 0):
        return None
    proj_w = proj.weight.reshape(-1).to(torch.bfloat16).contiguous()
    norm_w = norm.weight.to(torch.bfloat16).contiguous()

    def run(prefix, block):
        t = prefix.shape[0]
        output, _rsigma, _probs, _logits = attn_res_fwd(
            prefix.reshape(t, 1, H).contiguous(),
            block.reshape(K, t, 1, H).contiguous(),
            proj_w,
            norm_w,
            float(norm.eps),
        )
        return output.reshape(t, H)

    if M <= _ATTN_RES_OP_MAX_TOKENS:
        return run(prefix_sum, block_residual)
    # The op takes at most _ATTN_RES_OP_MAX_TOKENS tokens per call and is
    # per-token, so longer prefill chunks run in slices; the fp32 reference
    # below would otherwise cost seconds per iteration at 24k+ tokens.
    out = torch.empty_like(prefix_sum)
    for lo in range(0, M, _ATTN_RES_OP_MAX_TOKENS):
        hi = min(M, lo + _ATTN_RES_OP_MAX_TOKENS)
        out[lo:hi] = run(prefix_sum[lo:hi], block_residual[:, lo:hi])
    return out


def _rms_norm_eps(norm: target_weights.TensorWeights) -> float:
    if hasattr(norm, "eps"):
        return float(norm.eps)
    return float(norm.variance_epsilon)


def _note_attn_res_fusion(site: str, fused: bool, M: int, H: int, N: int) -> None:
    """Report whether the fused path was actually reached, once per shape.

    ``_FUSED_ATTN_RES_ENABLED`` only says the feature is switched on, not that
    the shape gate let the call through, and a rejected call looks exactly like
    a disabled one in the logs. Emitted at debug level: it is once per distinct
    shape, not once per process, so it is a diagnostic rather than a summary.
    """
    logger.debug_once(
        f"Kimi K3 attn-res fusion [{site}]: "
        f"{'FUSED' if fused else 'fallback'} (M={M}, H={H}, N={N})",
        key=f"kimi_k3_attn_res_fusion_{site}_{fused}_{M}_{H}_{N}",
    )


def _apply_attn_res_rmsnorm_fused(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: target_weights.TensorWeights,
    norm: target_weights.TensorWeights,
    output_norm: target_weights.TensorWeights,
) -> Optional[torch.Tensor]:
    """Fuse attention-residual mixing with its immediately following norm."""
    if (
        prefix_sum.dtype is not torch.bfloat16
        or not prefix_sum.is_cuda
        or not block_residual.is_cuda
    ):
        return None
    M, H = prefix_sum.shape
    K = int(block_residual.shape[0])
    N = K + 1
    # The fused path is taken for M <= _FUSED_ATTN_RES_MAX_TOKENS, H == 7168 and
    # N <= 12; other shapes take the unfused add + attn_res_fwd + RMSNorm path.
    if M > _FUSED_ATTN_RES_MAX_TOKENS or H != 7168 or N > 12:
        _note_attn_res_fusion("attn_res+norm", False, M, H, N)
        return None
    layer_kernel = prefix_sum.reshape(M, 1, H).contiguous()
    block_kernel = block_residual.reshape(K, M, 1, H).contiguous()
    output = attn_res_rmsnorm_fwd(
        layer_kernel,
        block_kernel,
        proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
        norm.weight.to(torch.bfloat16).contiguous(),
        output_norm.weight.to(torch.bfloat16).contiguous(),
        float(norm.eps),
        _rms_norm_eps(output_norm),
    )
    _note_attn_res_fusion("attn_res+norm", True, M, H, N)
    return output.reshape(M, H)


def _apply_attn_res_add_rmsnorm_fused(
    prefix_sum: torch.Tensor,
    addend: torch.Tensor,
    block_residual: torch.Tensor,
    proj: target_weights.TensorWeights,
    norm: target_weights.TensorWeights,
    output_norm: target_weights.TensorWeights,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Fuse ``prefix_sum + addend``, attention-residual, and trailing norm.

    The production residual add produces a BF16 tensor that remains live across
    the following MLP. The kernel therefore returns that materialized,
    BF16-rounded prefix sum alongside the normalized attention-residual output,
    while avoiding a separate add launch and a re-read of the intermediate by
    attention-residual selection.
    """
    if (
        prefix_sum.dtype is not torch.bfloat16
        or addend.dtype is not torch.bfloat16
        or not prefix_sum.is_cuda
        or not addend.is_cuda
        or not block_residual.is_cuda
        or prefix_sum.shape != addend.shape
    ):
        return None
    M, H = prefix_sum.shape
    K = int(block_residual.shape[0])
    N = K + 1
    # Same window as _apply_attn_res_rmsnorm_fused above.
    if M > _FUSED_ATTN_RES_MAX_TOKENS or H != 7168 or N > 12:
        _note_attn_res_fusion("add+attn_res+norm", False, M, H, N)
        return None
    layer_kernel = prefix_sum.reshape(M, 1, H).contiguous()
    addend_kernel = addend.reshape(M, 1, H).contiguous()
    block_kernel = block_residual.reshape(K, M, 1, H).contiguous()
    updated_prefix_sum, output = attn_res_add_rmsnorm_fwd(
        layer_kernel,
        addend_kernel,
        block_kernel,
        proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
        norm.weight.to(torch.bfloat16).contiguous(),
        output_norm.weight.to(torch.bfloat16).contiguous(),
        float(norm.eps),
        _rms_norm_eps(output_norm),
    )
    _note_attn_res_fusion("add+attn_res+norm", True, M, H, N)
    return updated_prefix_sum.reshape(M, H), output.reshape(M, H)


def _apply_attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: target_weights.TensorWeights,
    norm: target_weights.TensorWeights,
) -> torch.Tensor:
    """Exact port of HF ``modeling_kimi._apply_attn_res`` (fp32 math).

    prefix_sum:     ``[num_tokens, hidden_size]``
    block_residual: ``[num_snapshots, num_tokens, hidden_size]``

    Unless ``KIMI_K3_FUSED_ATTN_RES=0``, inputs fitting the fused kernel's
    contract dispatch directly to the in-tree ``trtllm::attn_res_fwd`` op.
    Only the fallback boundary restores the HF ``[M, K, H]`` layout.
    """
    if _FUSED_ATTN_RES_ENABLED:
        fused = _apply_attn_res_fused(prefix_sum, block_residual, proj, norm)
        if fused is not None:
            return fused
        # The fp32 reference below is orders of magnitude slower at prefill
        # sizes; say so once per shape rather than fall back silently.
        _note_attn_res_fusion(
            "attn_res",
            False,
            prefix_sum.shape[0],
            prefix_sum.shape[1],
            int(block_residual.shape[0]) + 1,
        )
    block_residual_hf = block_residual.transpose(0, 1)
    v = torch.cat((block_residual_hf, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + norm.eps)
    score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    hidden_states = torch.matmul(probs, v_float).squeeze(1)
    return hidden_states.to(v.dtype)


def _apply_attn_res_and_rmsnorm(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: target_weights.TensorWeights,
    norm: target_weights.TensorWeights,
    output_norm: target_weights.TensorWeights,
) -> torch.Tensor:
    """Apply attention-residual selection and the next RMSNorm."""
    if _FUSED_ATTN_RES_ENABLED:
        fused = _apply_attn_res_rmsnorm_fused(prefix_sum, block_residual, proj, norm, output_norm)
        if fused is not None:
            return fused
    return flashinfer_rmsnorm(
        _apply_attn_res(prefix_sum, block_residual, proj, norm),
        output_norm.weight,
        _rms_norm_eps(output_norm),
    )


def _apply_attn_res_add_and_rmsnorm(
    prefix_sum: torch.Tensor,
    addend: torch.Tensor,
    block_residual: torch.Tensor,
    proj: target_weights.TensorWeights,
    norm: target_weights.TensorWeights,
    output_norm: target_weights.TensorWeights,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add an attention output to the running residual, then select and norm."""
    if _FUSED_ATTN_RES_ENABLED:
        fused = _apply_attn_res_add_rmsnorm_fused(
            prefix_sum, addend, block_residual, proj, norm, output_norm
        )
        if fused is not None:
            return fused
    updated_prefix_sum = prefix_sum + addend
    return updated_prefix_sum, _apply_attn_res_and_rmsnorm(
        updated_prefix_sum, block_residual, proj, norm, output_norm
    )


# ---------------------------------------------------------------------------
# Latent MoE block using the unified ConfigurableMoE stack.
# ---------------------------------------------------------------------------

# Routed-expert key spellings a per-layer (ModelOpt ``quantized_layers``) declaration may use; a lookup
# over this product finds any of them without the MiniMax-M3-style prefix normalization in ``ModelConfig``.
_K3_ROUTED_EXPERT_KEY_PREFIXES = ("language_model.model.", "model.", "")
_K3_ROUTED_EXPERT_KEY_SUFFIXES = ("block_sparse_moe.experts", "mlp.experts")

# The subset of the above that can be a real module path. ``exclude_modules``
# matches with wildcards and walks ancestor prefixes, so an empty prefix would
# widen what matches instead of just missing, as it does in the dict lookup.
_K3_ROUTED_EXPERT_MODULE_PREFIXES = ("language_model.model.", "model.")

# Native routed-expert format for layers without an explicit checkpoint declaration.
_K3_DEFAULT_ROUTED_QUANT_ALGO = QuantAlgo.NVFP4


class KimiK3MoERuntime:
    """Kimi K3 latent MoE block backed by ConfigurableMoE."""

    def __init__(
        self,
        model_config: ModelConfig,
        cfg,
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
        bank: nn.ParameterDict,
    ):
        """Build the routed experts and the shared expert for one MoE layer.

        ``cfg`` is the raw ``PretrainedConfig`` rather than anything derived:
        the SiTU soft-caps and the routed-expert geometry are Kimi K3 fields
        that ``ModelConfig`` does not carry.

        ``aux_stream_dict`` is shared across every layer of the model, so the
        streams reached through it are borrowed and must not be synchronized
        or reassigned here.
        """
        self.layer_idx = layer_idx
        self.hidden_size = cfg.hidden_size
        self.num_experts = cfg.num_experts
        self.top_k = cfg.num_experts_per_token
        self.moe_hidden_size = cfg.routed_expert_hidden_size
        # ValueError (not assert): these guard unsupported checkpoint
        # configurations and must stay active under ``python -O``.
        if self.moe_hidden_size is None:
            raise ValueError("Kimi K3 runtime expects the latent MoE (routed_expert_hidden_size)")
        if not getattr(cfg, "latent_moe_use_norm", False):
            raise ValueError("Kimi K3 runtime expects latent_moe_use_norm=True")

        situ_beta, situ_linear_beta = _resolve_kimi_situ_betas(cfg)
        dtype = torch.bfloat16

        # Routing scores stay fp32; with attention-DP off the gate GEMM runs
        # bf16xbf16 with fp32 accumulate/output (checkpoint stores the gate
        # weight in bf16; saves a per-layer input cast + fp32 splitK pair on
        # the bs1 decode path). Under attention-DP the legacy upcast-to-fp32
        # GEMM is kept: the bf16-input min-latency GEMM's different reduction
        # order flips borderline top-16 picks (GSM8K 96.7 -> 96.1/96.4,
        # 3-run bisect), and the bs1-latency win is irrelevant
        # at DEP batch sizes.
        # KIMI_K3_ROUTER_BF16=1/0 forces either path.
        _router_bf16_env = os.environ.get("KIMI_K3_ROUTER_BF16")
        _router_bf16 = (
            _router_bf16_env == "1"
            if _router_bf16_env is not None
            else model_config.mapping.tp_group_shares_requests
        )
        base = f"model.layers.{layer_idx}.block_sparse_moe."
        target_weights.declare_auxiliary(
            bank,
            self,
            "gate",
            base + "gate.weight",
            (self.num_experts, cfg.hidden_size),
            torch.bfloat16 if _router_bf16 else torch.float32,
        )
        target_weights.declare_auxiliary(
            bank,
            self,
            "router_bias",
            base + "gate.e_score_correction_bias",
            (self.num_experts,),
            torch.float32,
        )
        bias_key = self.tensor_fields["router_bias"][0]
        routing_method = _K31RoutingMethod(
            top_k=cfg.num_experts_per_token,
            n_group=getattr(cfg, "num_expert_group", 1),
            topk_group=getattr(cfg, "topk_group", 1),
            routed_scaling_factor=cfg.routed_scaling_factor,
            callable_e_score_correction_bias=lambda: bank[bias_key],
            is_fused=True,
        )

        routed_moe_model_config = self._routed_moe_model_config(model_config)
        routed_quant_config = self._resolve_routed_quant_config(model_config, layer_idx)
        # Resolved here so ``load_weights`` reads the checkpoint layout off the
        # module instead of re-deriving it at each of its three call sites.
        self.expert_ckpt_spec = _k3_expert_ckpt_spec(routed_quant_config.quant_algo)
        routed_moe_kwargs = dict(
            routing_method=routing_method,
            num_experts=self.num_experts,
            hidden_size=self.moe_hidden_size,
            intermediate_size=cfg.moe_intermediate_size,
            dtype=dtype,
            # Kimi owns the latent reduction so it can order that collective
            # after the shared expert's auxiliary-stream reduction.
            reduce_results=False,
            # Lets a FUSED_COMM backend return its globally combined routed
            # output; ``_reduce_routed_output`` then skips that reduction.
            accepts_global_routed_output=True,
            model_config=routed_moe_model_config,
            override_quant_config=routed_quant_config,
            layer_idx=layer_idx,
            aux_stream_dict=aux_stream_dict,
            # Let CommunicationFactory select the best available strategy.
            communication_method=None,
            activation=SiTuActivation(
                gate_softcap=situ_beta,
                linear_softcap=situ_linear_beta,
            ),
        )
        self._check_trtllm_situ_quant(
            routed_moe_model_config.moe_backend, routed_quant_config.quant_algo
        )
        self.routed_experts = create_moe(**routed_moe_kwargs)
        if not isinstance(self.routed_experts, ConfigurableMoE):
            raise RuntimeError(
                "Kimi K3 requires ConfigurableMoE; ENABLE_CONFIGURABLE_MOE must not be disabled."
            )
        # Under attention DP the engine routes each chunk through ``routing_method.apply`` before the token
        # dispatch; up to 64 rows the backend's fused route + quantize serves it, above that the routing-tail kernel.
        self.routed_experts.routing_method.route_tail = (
            model_config.mapping.enable_attention_dp
            and getattr(cfg, "num_expert_group", 1) == 1
            and self.routed_experts.backend.input_requirement.routing_scales_dtype == torch.bfloat16
        )
        if self.routed_experts.layer_load_balancer is not None:
            raise NotImplementedError(
                "Kimi K3 packed-checkpoint streaming does not yet support "
                "dynamic EPLB or replicated expert slots."
            )
        local_expert_ids = list(self.routed_experts.backend.initial_local_expert_ids)
        if local_expert_ids != list(
            range(local_expert_ids[0], local_expert_ids[0] + len(local_expert_ids))
        ):
            raise NotImplementedError(
                "Kimi K3 packed-checkpoint streaming currently requires a "
                "contiguous static expert partition."
            )
        self.local_expert_ids = tuple(local_expert_ids)
        self.experts_per_rank = len(local_expert_ids)
        self.expert_lo = local_expert_ids[0]
        self.expert_hi = self.expert_lo + self.experts_per_rank
        # Checkpoint expert of each local routed-expert slot (identity placement).
        self.ckpt_expert_ids = self.local_expert_ids

        shared_intermediate = cfg.moe_intermediate_size * cfg.num_shared_experts
        self._shared_intermediate = shared_intermediate
        attention_dp = model_config.mapping.enable_attention_dp
        shared_model_config = copy.copy(model_config)
        shared_recipes = [
            resolve_attention_quant_config(
                model_config,
                layer_idx,
                name,
                module="block_sparse_moe.shared_experts",
                allowed_algos=(QuantAlgo.MXFP8,),
            )
            for name in ("gate_proj", "up_proj", "down_proj")
        ]
        if len({recipe.quant_algo for recipe in shared_recipes}) != 1:
            raise ValueError("Kimi K3 shared gate/up/down projections must use the same format")
        shared_model_config.quant_config = shared_recipes[0]
        # Under attention DP each rank owns different tokens, so the shared
        # expert is replicated (TP size 1) and must not reduce across ranks.
        # Direct MoE-TP leaves both branches as partials for one concatenated
        # all-reduce.
        use_shared_tp = not attention_dp and model_config.mapping.tp_size > 1
        self._reduce_routed_output = (
            use_shared_tp and not self.routed_experts.returns_global_routed_output
        )
        if self._reduce_routed_output and self.routed_experts.all_reduce is None:
            raise RuntimeError(
                "Kimi K3 direct MoE tensor parallelism requires the "
                "ConfigurableMoE all-reduce even when reduce_results=False."
            )
        self.shared_experts = target_weights.declare_mlp(
            bank,
            layer_idx,
            cfg,
            shared_model_config,
            shared=True,
            tp_size=1 if attention_dp else model_config.mapping.tp_size,
            beta=situ_beta,
            linear_beta=situ_linear_beta,
        )
        self.shared_all_reduce = self.shared_experts.all_reduce
        # Side stream (+ fork/join events) for overlapping shared-expert
        # compute with the routed chain. Only engaged when multi-stream is
        # active (CUDA graphs on); otherwise both run in order on the default
        # stream.
        self.shared_expert_stream = aux_stream_dict[AuxStreamType.MoeShared]
        if self.routed_experts.returns_global_routed_output and self.shared_all_reduce is not None:
            # Fused routed communication and the shared all-reduce must run
            # in the same stream order on every rank to avoid collective deadlock.
            self.shared_expert_stream = None
        self.moe_main_event = torch.cuda.Event()
        self.moe_shared_event = torch.cuda.Event()
        for name, shape in (
            ("routed_expert_down_proj", (self.moe_hidden_size, cfg.hidden_size)),
            ("routed_expert_up_proj", (cfg.hidden_size, self.moe_hidden_size)),
            ("routed_expert_norm", (self.moe_hidden_size,)),
        ):
            target_weights.declare_auxiliary(
                bank, self, name, base + name + ".weight", shape, dtype, cfg.rms_norm_eps
            )

    @staticmethod
    def _select_moe_tp_ep(mapping: Mapping) -> Tuple[int, int]:
        """Resolve the routed-expert ``(moe_tp, moe_ep)`` split.

        Precedence:

        1. Explicit ``moe_tensor_parallel_size`` / ``moe_expert_parallel_size``
           from the user config. Detected via
           ``mapping.moe_tp_ep_user_specified`` so the auto-resolved mapping
           default (``moe_tp=tp_size, moe_ep=1``) is NOT mistaken for a TP
           request.
        2. Default: EP-only (``moe_tp=1, moe_ep=tp_size``), the historical
           K3 layout.
        """
        tp_size = mapping.tp_size
        if getattr(mapping, "moe_tp_ep_user_specified", False):
            return mapping.moe_tp_size, mapping.moe_ep_size
        return 1, tp_size

    @staticmethod
    def _resolve_routed_quant_config(model_config: ModelConfig, layer_idx: int) -> QuantConfig:
        """Routed-expert quantization for ``layer_idx``, taken from the checkpoint.

        Per-layer declarations preserve the checkpoint's native NVFP4 recipe.
        Layers without an explicit declaration use NVFP4 with 16-value scale groups.
        ``_k3_expert_ckpt_spec`` validates the stored expert format before loading.

        An exclusion outranks the per-layer entry and the default below:
        ``create_weights`` treats an override as authoritative over anything
        ``__post_init__`` wrote, so this return value stands in for both
        quantization passes and exclusion is the one that runs second. It is
        matched as a pattern, so it is asked only about real module names.
        """
        quant_config = model_config.quant_config
        if quant_config is not None and any(
            quant_config.is_module_excluded_from_quantization(
                f"{prefix}layers.{layer_idx}.{suffix}"
            )
            for prefix in _K3_ROUTED_EXPERT_MODULE_PREFIXES
            for suffix in _K3_ROUTED_EXPERT_KEY_SUFFIXES
        ):
            logger.debug(
                "Kimi K3 layer %d routed experts: excluded from quantization, "
                "keeping them unquantized",
                layer_idx,
            )
            return QuantConfig(kv_cache_quant_algo=quant_config.kv_cache_quant_algo)

        per_layer = getattr(model_config, "quant_config_dict", None)
        if per_layer:
            for prefix in _K3_ROUTED_EXPERT_KEY_PREFIXES:
                for suffix in _K3_ROUTED_EXPERT_KEY_SUFFIXES:
                    cfg = per_layer.get(f"{prefix}layers.{layer_idx}.{suffix}")
                    if cfg is not None and cfg.quant_algo is not None:
                        logger.debug(
                            "Kimi K3 layer %d routed experts: %s (group_size=%s) "
                            "from the checkpoint",
                            layer_idx,
                            cfg.quant_algo,
                            cfg.group_size,
                        )
                        return cfg
        logger.debug(
            "Kimi K3 layer %d routed experts: no per-layer quant config in the "
            "checkpoint, defaulting to %s",
            layer_idx,
            _K3_DEFAULT_ROUTED_QUANT_ALGO,
        )
        return QuantConfig(quant_algo=_K3_DEFAULT_ROUTED_QUANT_ALGO, group_size=16)

    @staticmethod
    def _check_trtllm_situ_quant(moe_backend: str, quant_algo: Optional[QuantAlgo]) -> None:
        """Reject a routed-expert format trtllm-gen has no fused SiTu cubin for.

        trtllm-gen has fused SiTu FC1 cubins for a few input formats and no
        standalone SiTu activation kernel, so anything else has to die here
        rather than in a cubin lookup deep inside the runner. The admitted set
        is read off the backend rather than restated here.

        A staticmethod, not an inline block, so that the invariant is
        reachable from a test without constructing the whole runtime.
        """
        situ_supported = TRTLLMGenFusedMoE.situ_supported_quant_algos()
        if moe_backend != "TRTLLM" or quant_algo in situ_supported:
            return
        supported = ", ".join(sorted(algo.name for algo in situ_supported))
        raise ValueError(
            f"Kimi K3 routed experts are quantized as {quant_algo}, which the "
            "TRTLLM (trtllm-gen) MoE backend cannot serve: fused SiTu cubins "
            f"exist only for {supported}."
        )

    @staticmethod
    def _routed_moe_model_config(model_config: ModelConfig) -> ModelConfig:
        """Build a private routed-expert mapping without mutating the shared
        config. Default split is EP-only; see ``_select_moe_tp_ep``."""
        # Each backend's quantization method owns its native NVFP4 expert layout.
        supported = ("TRTLLM", "CUTEDSL", "MEGAMOE_CUTEDSL")
        if model_config.moe_backend not in supported:
            raise ValueError(
                f"Kimi K3.1 NVFP4 routed experts require MoE backend {supported}; "
                f"got {model_config.moe_backend!r}."
            )
        if model_config.moe_load_balancer is not None:
            raise NotImplementedError(
                "Kimi K3 packed-checkpoint streaming does not yet support "
                "EPLB or replicated expert slots."
            )
        mapping = model_config.mapping
        moe_tp, moe_ep = KimiK3MoERuntime._select_moe_tp_ep(mapping)
        if moe_tp < 1 or moe_ep < 1 or moe_tp * moe_ep != mapping.tp_size:
            raise ValueError(
                f"Kimi K3 routed MoE split moe_tp={moe_tp} x moe_ep={moe_ep} "
                f"must multiply to tp_size={mapping.tp_size}."
            )
        if moe_tp > 1 and mapping.enable_attention_dp:
            raise NotImplementedError(
                "Kimi K3 MoE tensor parallelism requires "
                "enable_attention_dp=false (the attention-DP dispatch/combine "
                "path is validated for EP-only splits)."
            )
        logger.info_once(
            f"Kimi K3 routed MoE parallelism: moe_tp={moe_tp}, "
            f"moe_ep={moe_ep} (tp_size={mapping.tp_size})",
            key="kimi_k3_moe_tp_ep_split",
        )

        mapping_dict = mapping.to_dict()
        mapping_dict["moe_cluster_size"] = 1
        mapping_dict["moe_tp_size"] = moe_tp
        mapping_dict["moe_ep_size"] = moe_ep
        routed_mapping = Mapping.from_dict(mapping_dict)

        routed_model_config = copy.copy(model_config)
        routed_model_config._frozen = False
        routed_model_config.extra_attrs = copy.copy(model_config.extra_attrs)
        routed_model_config.mapping = routed_mapping
        routed_model_config.moe_backend = model_config.moe_backend
        routed_model_config._frozen = True
        return routed_model_config


def resolve_attention_quant_config(
    config: ModelConfig | None,
    layer_idx: int,
    projection: str,
    *,
    module: str = "self_attn",
    allowed_algos: tuple[QuantAlgo, ...] = (),
) -> QuantConfig:
    """Resolve a checkpoint projection, including mixed-precision exclusions."""
    if config is None:
        return QuantConfig()
    global_config = config.quant_config or QuantConfig()
    names = [
        f"{prefix}layers.{layer_idx}.{module}.{projection}"
        for prefix in ("language_model.model.", "model.", "")
    ]
    if any(global_config.is_module_excluded_from_quantization(name) for name in names):
        return QuantConfig(kv_cache_quant_algo=global_config.kv_cache_quant_algo)
    declarations = config.quant_config_dict or {}
    matches = [declarations[name] for name in names if name in declarations]
    if matches:
        selected = matches[0]
        if any(match.quant_algo != selected.quant_algo for match in matches[1:]):
            raise ValueError(f"Conflicting Kimi K3 quantization aliases for {names[0]}")
    elif global_config.quant_algo == QuantAlgo.MIXED_PRECISION:
        selected = QuantConfig()
    else:
        selected = global_config
    if selected.quant_algo is not None and selected.quant_algo not in allowed_algos:
        raise ValueError(
            f"Kimi K3.1 attention projection {names[0]} is quantized as {selected.quant_algo}; "
            "this target loads BF16 attention projections"
        )
    result = copy.copy(selected)
    result.kv_cache_quant_algo = global_config.kv_cache_quant_algo
    return result


# ---------------------------------------------------------------------------
# MLA runtime.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Decoder layer.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Model.
# ---------------------------------------------------------------------------


class KimiLinearModel(DecoderModel):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        cfg = _get_text_config(model_config.pretrained_config)
        self._text_cfg = cfg
        dtype = torch.bfloat16

        # Attention and MoE phases are sequential, so their branch-overlap
        # roles share one stream; MoE-internal overlap roles remain separate.
        aux_stream_list = [torch.cuda.Stream() for _ in range(4)]
        self.aux_stream_dict = {
            AuxStreamType.Attention: aux_stream_list[0],
            AuxStreamType.MoeShared: aux_stream_list[0],
            AuxStreamType.MoeChunkingOverlap: aux_stream_list[1],
            AuxStreamType.MoeBalancer: aux_stream_list[2],
            AuxStreamType.MoeOutputMemset: aux_stream_list[3],
        }

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, dtype=dtype)
        self.w = nn.ParameterDict()
        self.layers = [
            target_weights.declare_layer(self.w, model_config, cfg, i, self.aux_stream_dict)
            for i in range(cfg.num_hidden_layers)
        ]
        self.moe_runners = nn.ModuleDict(
            {
                str(layer.layer_idx): layer.block_sparse_moe.routed_experts
                for layer in self.layers
                if layer.is_moe
            }
        )
        self.collectives = nn.ModuleList(
            collective
            for layer in self.layers
            for collective in (
                layer.linear_attn._o_allreduce if layer.is_kda else layer.mla_all_reduce,
                layer.block_sparse_moe.shared_experts.all_reduce
                if layer.is_moe
                else layer.mlp.all_reduce,
            )
            if collective is not None
        )
        for name, shape in (
            ("norm", (cfg.hidden_size,)),
            ("output_attn_res_norm", (cfg.hidden_size,)),
            ("output_attn_res_proj", (1, cfg.hidden_size)),
        ):
            target_weights.declare_auxiliary(
                self.w, self, name, f"model.{name}.weight", shape, dtype, cfg.rms_norm_eps
            )
        self.num_attn_res_snapshots = (
            cfg.num_hidden_layers + cfg.attn_res_block_size - 1
        ) // cfg.attn_res_block_size

        # Which convention the drafter tap is on is not recoverable from the
        # served output -- a mismatch only lowers acceptance -- so state it once
        # at construction rather than leaving it to be inferred from an AL.
        logger.info_once(
            "Kimi K3 aux hidden capture: mode="
            f"{'attn_res_stream' if _AUX_ATTN_RES_STREAM_ENABLED else 'prefix_only'} "
            f"({KIMI_K3_AUX_ATTN_RES_STREAM_ENV}={int(_AUX_ATTN_RES_STREAM_ENABLED)})",
            key="kimi_k3_aux_capture_mode",
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata=None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = F.embedding(input_ids, self.embed_tokens.weight)
        hidden_states = inputs_embeds

        block_residual = hidden_states.new_empty(
            self.num_attn_res_snapshots,
            hidden_states.shape[0],
            hidden_states.shape[1],
        )
        num_snapshots = 0
        capture_set = (
            getattr(spec_metadata, "_capture_layer_set", None)
            if spec_metadata is not None
            else None
        )
        for i, layer in enumerate(self.layers):
            # Drafter hidden-state capture. The drafter is distilled on the
            # aggregated stream value -- the pre-norm softmax mixture its next
            # consumer sees -- not on the raw prefix sum a layer returns (see
            # KIMI_K3_AUX_ATTN_RES_STREAM). The tap fires inside layer i+1,
            # which computes that tensor anyway; see its forward docstring.
            capture = None
            if (
                spec_metadata is not None
                and i > 0
                and (capture_set is None or self.layers[i - 1].layer_idx in capture_set)
            ):
                capture = (spec_metadata, self.layers[i - 1].layer_idx)
            valid_block_residual = block_residual[:num_snapshots]
            prefix_sum = hidden_states

            if capture is not None:
                # The mixture tap needs the PRE-norm value, so the two steps split on captured layers.
                if num_snapshots > 0:
                    hidden_states = _apply_attn_res(
                        prefix_sum,
                        valid_block_residual,
                        layer.self_attention_res_proj,
                        layer.self_attention_res_norm,
                    )
                # A property of the DRAFTER checkpoint, not a knob: a mismatch only lowers
                # acceptance, silently. hidden_states is the pre-norm attn_res mixture;
                # prefix_only wants the running prefix, already in hand as prefix_sum.
                tapped = hidden_states if _AUX_ATTN_RES_STREAM_ENABLED else prefix_sum
                capture[0].maybe_capture_hidden_states(capture[1], tapped, None)
                hidden_states = flashinfer_rmsnorm(
                    hidden_states,
                    layer.input_layernorm.weight,
                    layer.input_layernorm.variance_epsilon,
                )
            elif num_snapshots > 0:
                hidden_states = _apply_attn_res_and_rmsnorm(
                    prefix_sum,
                    valid_block_residual,
                    layer.self_attention_res_proj,
                    layer.self_attention_res_norm,
                    layer.input_layernorm,
                )
            else:
                hidden_states = flashinfer_rmsnorm(
                    hidden_states,
                    layer.input_layernorm.weight,
                    layer.input_layernorm.variance_epsilon,
                )

            if layer.layer_idx % layer.attn_res_block_size == 0:
                block_residual[num_snapshots].copy_(prefix_sum)
                num_snapshots += 1
                valid_block_residual = block_residual[:num_snapshots]
                prefix_sum = None

            if layer.is_kda:
                kda = layer.linear_attn
                w = kda.weight_view
                if is_in_breakable_cuda_graph():
                    core = hidden_states.new_empty((hidden_states.shape[0], w.heads, w.head_dim))
                    maybe_bcg_kda_core_inplace(hidden_states, kda.layer_idx_str, core)
                else:
                    core = _kda_forward_core(hidden_states, w, i, attn_metadata)
                hidden_states = _project(core.reshape(-1, w.heads * w.head_dim), w.output)
                if kda._o_allreduce is not None:
                    hidden_states = allreduce_module(
                        hidden_states, kda._o_allreduce, int(AllReduceFusionOp.NONE), None, None
                    )
            else:
                mla = layer.mla
                w = mla.weight_view
                attention_output = hidden_states.new_empty(
                    (hidden_states.shape[0], mla.attention_output_hidden_size)
                )
                from .ops.attention.graph_ops import maybe_bcg_mla_dsa_attn_inplace

                projected = torch.ops.trtllm.k31_nvfp4_mla_dsa_proj(
                    hidden_states, None, mla.layer_idx_str
                )
                maybe_bcg_mla_dsa_attn_inplace(
                    *projected[:4], projected[4:], None, mla.layer_idx_str, attention_output
                )
                if w.gate is not None:
                    attention_output = k31_sigmoid_gate_mul(
                        attention_output, F.linear(hidden_states, w.gate)
                    )
                hidden_states = F.linear(attention_output, w.output)
                if layer.mla_all_reduce is not None:
                    hidden_states = allreduce_module(
                        hidden_states, layer.mla_all_reduce, int(AllReduceFusionOp.NONE), None, None
                    )

            if prefix_sum is None:
                prefix_sum = hidden_states
                hidden_states = _apply_attn_res_and_rmsnorm(
                    prefix_sum,
                    valid_block_residual,
                    layer.mlp_res_proj,
                    layer.mlp_res_norm,
                    layer.post_attention_layernorm,
                )
            else:
                prefix_sum, hidden_states = _apply_attn_res_add_and_rmsnorm(
                    prefix_sum,
                    hidden_states,
                    valid_block_residual,
                    layer.mlp_res_proj,
                    layer.mlp_res_norm,
                    layer.post_attention_layernorm,
                )
            if layer.is_moe:
                moe = layer.block_sparse_moe
                identity = hidden_states
                router_input = hidden_states.reshape(-1, moe.hidden_size)
                if moe.gate.weight.dtype == torch.bfloat16:
                    router_logits = dsv3_router_gemm_op(
                        router_input.contiguous(), moe.gate.weight.t(), None, torch.float32
                    )
                else:
                    router_logits = F.linear(router_input.float(), moe.gate.weight)
                moe_all_reduce = (
                    moe.routed_experts.all_reduce if moe._reduce_routed_output else None
                )

                def _routed_output():
                    routed_in = k31_fused_a_gemm(
                        hidden_states, moe.routed_expert_down_proj.weight.t()
                    )
                    y = moe.routed_experts(
                        routed_in,
                        router_logits,
                        all_rank_num_tokens=getattr(attn_metadata, "all_rank_num_tokens", None),
                    )
                    if moe._reduce_routed_output:
                        return y
                    # Communication-backed paths return a complete routed result.
                    y = flashinfer_rmsnorm(
                        y, moe.routed_expert_norm.weight, moe.routed_expert_norm.variance_epsilon
                    )
                    return k31_fused_a_gemm(y, moe.routed_expert_up_proj.weight.t())

                def _shared_output():
                    """The shared expert's (reduced) output."""
                    return _dense_mlp(identity, moe.shared_weight_view)

                # Shared experts depend only on the block input, so overlap their GEMMs
                # with the routed dispatch/expert/combine chain. Multi-stream engages
                # only under CUDA graphs; otherwise both branches run in order on the
                # default stream. The shared expert includes its output all-reduce on
                # the auxiliary stream. The final join includes the shared all-reduce, so it
                # precedes the routed all-reduce: concurrent collectives on different
                # streams can corrupt SYMM_MEM all-reduce state.
                routed_out, shared_out = maybe_execute_in_parallel(
                    _routed_output,
                    _shared_output,
                    moe.moe_main_event,
                    moe.moe_shared_event,
                    moe.shared_expert_stream,
                    disable_on_compile=True,
                )
                if moe._reduce_routed_output:
                    # RMSNorm folded into the routed all-reduce (kARRMSNorm): the
                    # MNNVL kernel normalizes straight out of the lamport buffer, so
                    # the reduced row never round-trips through HBM for a norm kernel.
                    routed_latent = allreduce_module(
                        routed_out,
                        moe_all_reduce,
                        int(AllReduceFusionOp.RMS_NORM),
                        moe.routed_expert_norm.weight,
                        float(moe.routed_expert_norm.variance_epsilon),
                    )
                    routed_out = k31_fused_a_gemm(
                        routed_latent, moe.routed_expert_up_proj.weight.t()
                    )
                hidden_states = routed_out + shared_out
            else:
                hidden_states = _dense_mlp(hidden_states, layer.mlp_weight_view)

            prefix_sum = prefix_sum + hidden_states
            hidden_states = prefix_sum

        # The last layer has no successor, so this one recompute is
        # unavoidable -- output-side score weights, matching SGLang's
        # layer_idx + 1 >= end_layer branch. Unreachable for K3's capture set
        # against 93 layers; kept so a set that does include the final layer
        # gets the right tensor rather than the raw prefix sum.
        if spec_metadata is not None and len(self.layers) > 0:
            last = self.layers[-1]
            if capture_set is None or last.layer_idx in capture_set:
                tail = (
                    _apply_attn_res(
                        hidden_states,
                        block_residual[:num_snapshots],
                        self.output_attn_res_proj,
                        self.output_attn_res_norm,
                    )
                    if num_snapshots > 0 and _AUX_ATTN_RES_STREAM_ENABLED
                    else hidden_states
                )
                spec_metadata.maybe_capture_hidden_states(last.layer_idx, tail, None)

        hidden_states = _apply_attn_res_and_rmsnorm(
            hidden_states,
            block_residual[:num_snapshots],
            self.output_attn_res_proj,
            self.output_attn_res_norm,
            self.norm,
        )
        return hidden_states


# ---------------------------------------------------------------------------
# Causal LM wrapper + weight loading.
# ---------------------------------------------------------------------------


class KimiLinearForCausalLM(TargetCausalLM[KimiLinearModel, Any]):
    """Kimi K3 text core (KDA + MLA + MoE).

    Serves text-only ``kimi_linear`` checkpoints directly, and is reused as the
    text backbone by the multimodal ``KimiK3ForConditionalGeneration`` wrapper
    (``modeling_kimi_k3_vl``). The composite ``KimiK3ForConditionalGeneration``
    architecture is registered by that wrapper, not here."""

    mamba_metadata_cls = KimiK3MambaMetadata

    def __init__(self, model_config: ModelConfig):
        cfg = _get_text_config(model_config.pretrained_config)
        assert model_config.mapping.pp_size == 1, "Kimi K3 does not support pipeline parallelism"
        spec_config = getattr(model_config, "spec_config", None)
        setup_helix_mappings(self, model_config, cfg, spec_config)
        # Speculative decoding: one-engine DFlash with the DFlash2 MLA drafter. The target only
        # exposes per-layer hidden states via maybe_capture_hidden_states (KimiLinearModel.forward)
        # and verifies the drafted block (KDA and MLA multi-token verify).
        assert spec_config is None or spec_config.spec_dec_mode.is_dflash(), (
            "Kimi K3.1 supports speculative decoding only with DFlash (DFlash2 MLA drafter)"
        )
        super().__init__(
            KimiLinearModel(model_config),
            model_config,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )
        install_k31_dflash2(self, model_config)
        if self.mapping_with_cp is not None:
            frozen, model_config._frozen = model_config._frozen, False
            model_config.mapping = self.mapping_with_cp
            model_config._frozen = frozen

    @classmethod
    def get_model_defaults(cls, llm_args) -> dict:
        # - Keep block reuse opt-in because its combination with SA
        #   speculative decoding is not validated. Both settings use V2
        #   with the default use_kv_cache_manager_v2=True. The "auto"
        #   setting follows the model preference and can resolve to V1.
        # - tokens_per_block=64: keep the validated K3 page size. Earlier
        #   releases needed it to avoid a legacy MLA fallback that could not
        #   serve attention-DP's 96 query heads. The current FlashInfer
        #   TRTLLM-Gen backend declines H=96 at either page size, allowing
        #   the updated legacy TRTLLM-Gen autotuner to select a Q16 kernel.
        # - TRTLLM is the default MoE backend; explicit CuteDSL choices use
        #   the same streaming loader with their backend-owned NVFP4 layouts.
        return {
            "kv_cache_config": {
                "enable_block_reuse": False,
                "tokens_per_block": 64,
            },
            "moe_config": {"backend": "TRTLLM"},
        }

    @classmethod
    def requires_kv_cache_manager_v2(cls, pretrained_config: Any = None) -> bool:
        """Kimi K3.1 DSA keeps its indexer K-cache in a KV cache manager V2 buffer.

        Without a ``dsa_config`` (Kimi K3) the V1 hybrid managers stay available.
        """
        if pretrained_config is None:
            return False
        return getattr(_get_text_config(pretrained_config), "dsa_config", None) is not None

    @classmethod
    def get_preferred_transceiver_runtime(
        cls,
        pretrained_config: Any = None,
    ) -> Literal["PYTHON"]:
        """Kimi K3 disaggregated serving requires the Python transceiver.

        Only the Python NIXL transceiver (KvCacheTransceiverV2) can move
        the KDA recurrent state; the C++ transceiver has no KDA support.
        Adopted when the user leaves
        ``cache_transceiver_config.transceiver_runtime`` at 'auto' and the
        effective backend is NIXL. An explicit non-Python runtime is
        rejected by ``get_kv_cache_manager_cls`` rather than silently
        routed to a path that cannot transfer the recurrent state.
        """
        return "PYTHON"

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Checkpoint mapping and loading live in this target's ``weights.py``."""
        target_weights.load_text_model(self, weights)


# ====================================================================================================================
# DFlash drafter model
# ====================================================================================================================


def dflash2_grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base_kernel: torch.Tensor,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    """Short dynamic depthwise convolution across one drafted block.

    ``Conv(x)_t = sum_i k_{t,i} * x_{t-i}``, with ``k_{t,i} = base_kernel[i] +
    delta[t, i]`` (a per-position correction shared by every ``group_size``
    channels). Taps reaching past a block's start contribute nothing.

    Args:
        hidden_states: ``[B * block_size, hidden_size]``, request-major.
        delta: ``[B * block_size, taps, hidden_size // group_size]``.
        base_kernel: ``[taps, hidden_size]``.
    Returns:
        ``[B * block_size, hidden_size]``
    """
    taps, hidden_size = base_kernel.shape
    num_groups = hidden_size // group_size
    # FP32 math, one rounding at the end (what a fused kernel computes).
    blocks = hidden_states.float().unflatten(-1, (num_groups, group_size))
    coefficients = base_kernel.float().view(
        1, taps, num_groups, group_size
    ) + delta.float().unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    if taps > 1:
        position = torch.arange(hidden_states.shape[0], device=hidden_states.device) % block_size
        for tap in range(1, taps):
            shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
            # Select, not multiply: a tap past the block start reads the previous request's row,
            # which may hold anything (NaN in a padding row), and NaN * 0 is NaN.
            shifted = torch.where((position >= tap).view(-1, 1, 1), shifted, 0.0)
            output = output + coefficients[:, tap] * shifted
    return output.flatten(-2).to(hidden_states.dtype)


class DFlashForCausalLM(_BaseDFlashForCausalLM):
    """The target-owned DFlash drafter with FP32 block convolutions and the
    CuTe DSL greedy selector path."""

    # The drafter's own helix mapping when its context KV is sharded by page; None keeps it replicated.
    _helix_mapping = None
    # The registered architecture the backbone is built from in place of the checkpoint's label;
    # None keeps the engine's resolution.
    _backbone_architecture: Optional[str] = None

    def select_candidate_path(
        self, block_logits, candidate_ids, unary_logits, block_hidden_states, anchor_token_ids
    ):
        """The engine's greedy selector path (edge scores, the realized walk, ``block_logits``
        rewritten in place) as the CuTe DSL op ``ops/spec/dflash2_selector_greedy_path``; other
        selector shapes (not 32 candidates over a rank of whole 256 blocks) take the engine's."""
        if (
            candidate_ids.shape[-1] != K31_SELECTOR_TOP_K
            or self.w["predecessor_codebook"].shape[1] % 256
            or block_logits.dtype not in (torch.bfloat16, torch.float32)
        ):
            return super().select_candidate_path(
                block_logits, candidate_ids, unary_logits, block_hidden_states, anchor_token_ids
            )
        return k31_dflash2_selector_greedy_path(
            self.w["predecessor_codebook"],
            self.w["successor_codebook"],
            self.w["hidden_projection__weight"],
            candidate_ids,
            unary_logits,
            block_hidden_states,
            anchor_token_ids,
            block_logits,
        )

    def __init__(self, draft_config, *, dflash_attention_backend: str = "AUTO"):
        cfg = draft_config.pretrained_config
        if self._backbone_architecture is None:
            super().__init__(draft_config, dflash_attention_backend=dflash_attention_backend)
            return
        # The engine resolves the backbone from ``architectures``; pin it for the build only, so the
        # checkpoint's own label is what everything after construction reads.
        original_architectures = cfg.architectures
        cfg.architectures = [self._backbone_architecture]
        try:
            super().__init__(draft_config, dflash_attention_backend=dflash_attention_backend)
        finally:
            cfg.architectures = original_architectures


# ====================================================================================================================
# DFlash2 MLA drafter
# ====================================================================================================================


def dflash2_attention_mode(pretrained_config) -> Optional[str]:
    """``dflash_config.attention_mode`` of a drafter config (None when absent)."""
    dflash_config = getattr(pretrained_config, "dflash_config", None) or {}
    return dflash_config.get("attention_mode")


def dflash2_sliding_window(pretrained_config) -> int:
    """The drafter's attention window in positions (the query's own included).

    Every layer must be a ``sliding_attention`` layer of one ``sliding_window``: the block decode
    builds one set of window index lists for all layers.
    """
    window = getattr(pretrained_config, "sliding_window", None)
    layer_types = getattr(pretrained_config, "layer_types", None) or []
    if not window or any(t != "sliding_attention" for t in layer_types):
        raise ValueError(
            "The DFlash2 MLA drafter needs every layer to be a sliding_attention layer of one "
            f"sliding_window; got sliding_window={window!r}, layer_types={layer_types!r}."
        )
    if not getattr(pretrained_config, "is_causal", True):
        raise ValueError("The DFlash2 MLA drafter's block attention is causal (is_causal: true).")
    return int(window)


def mla_rope_pairs(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Rotate ``x``'s last dim as adjacent (re, im) pairs by ``freqs_cis`` (FP32 math).

    ``x`` is ``[..., rope_dim]``; ``freqs_cis`` complex ``[..., rope_dim // 2]`` broadcastable
    to ``x``'s leading dims. The interleaved (GPT-J) rotation of ``rope_interleave: true``.
    """
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)).contiguous())
    return torch.view_as_real(xc * freqs_cis).flatten(-2).to(x.dtype)


def mla_latent_norm_rope(
    latent: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    latent_dim: int,
) -> torch.Tensor:
    """``[RMSNorm(latent[..., :latent_dim]) | RoPE(latent[..., latent_dim:])]`` (FP32 math)."""
    c = latent[..., :latent_dim].float()
    c = c * torch.rsqrt(c.square().mean(-1, keepdim=True) + eps) * norm_weight.float()
    k_pe = mla_rope_pairs(latent[..., latent_dim:], freqs_cis)
    return torch.cat([c.to(latent.dtype), k_pe], dim=-1)


class MLADFlash2ForCausalLM(DFlashForCausalLM):
    """The DFlash2 drafter of Kimi K3.1: MLA block decode with an exact causal sliding window.

    The worker is the shared ``DFlashWorker``: it stores the context latents
    (:meth:`precompute_context_kv`) in a paged pool, runs :meth:`dflash_forward` and picks the
    draft tokens through the candidate selector.
    """

    # The absorbed MLA block decode owns its attention implementation. Distributed execution
    # binds its context pages to the shared manager, including aggregated serving.
    supports_unified_draft_cache = True
    unified_draft_cache_when_aggregated = True
    _uses_worker_attention_backend = False
    _default_attention_backend = "VANILLA"
    _supported_attention_backends = ("VANILLA",)
    _auto_fallback_attention_backend = "VANILLA"
    _paged_ctx_cache = True
    # Rows per slot of the worker's context ring (ops/spec/dflash2_ring), which it sets when it sizes
    # its private arena by _ctx_ring_window; None reads the table densely by position.
    _ctx_ring_rows = None
    _tp_fusions = False
    # The checkpoint labels itself model_type qwen3 (a leftover of its training code); its layers are
    # the engine's MLA drafter backbone (``K3DsparkModel``: MLA projections, SwiGLU MLP, norms, the
    # final norm and the target-shared embedding and LM head). __init__ admits only attention_mode
    # "mla" configs.
    _backbone_architecture = "K3DsparkForCausalLM"

    def __init__(
        self,
        draft_config,
        *,
        block_size: Optional[int] = None,
        dflash_attention_backend: str = "VANILLA",
        context_kv_dtype: Optional[str] = None,
    ):
        cfg = draft_config.pretrained_config
        if dflash2_attention_mode(cfg) != "mla":
            raise ValueError(
                "MLADFlash2ForCausalLM serves DFlash2 drafters with dflash_config.attention_mode "
                f"'mla', got {dflash2_attention_mode(cfg)!r}."
            )
        window = dflash2_sliding_window(cfg)
        super().__init__(draft_config, dflash_attention_backend=dflash_attention_backend)
        self._window = window
        # Every read is inside the window, so the worker may keep a window-bounded context ring.
        self._ctx_ring_window = window
        # Window-attention workspace and multi-CTA counters, allocated on the first (eager) call
        # for _max_block_rows query rows (set by the worker: max requests x block width).
        self._window_attn_buffers = None
        self._window_attn_rows = 0
        self._max_block_rows = 0
        # The TRTLLM-Gen window attention supports BF16 history on SM 10.3. Other architecture
        # or history formats use FP32 window decode, selected once before graph capture.
        self._window_attn_trtllm_gen = get_sm_version() == 103 and context_kv_dtype != "fp8"
        if self._window_attn_trtllm_gen:
            logger.info_once(
                "DFlash2 drafter window attention: TRTLLM-Gen sparse MLA decode (SM 10.3).",
                key="k31_dflash2_window_attention",
            )
        else:
            logger.info_once(
                f"DFlash2 drafter window attention: exact FP32 decode on SM {get_sm_version()} "
                f"with {context_kv_dtype} history.",
                key="k31_dflash2_window_attention",
            )
        if not self.is_dflash2:
            raise ValueError("MLADFlash2ForCausalLM needs a DFlash2 checkpoint (convs + selector).")
        # The configured width (max_draft_len + 1) replaces the checkpoint's block_size for the
        # draft block, the convolutions and the worker.
        if block_size is not None:
            self.block_size = int(block_size)

        attn0 = self.model.layers[0].self_attn
        self.qk_nope_head_dim = attn0.qk_nope_head_dim
        self.qk_rope_head_dim = attn0.qk_rope_head_dim
        self.kv_lora_rank = attn0.kv_lora_rank
        self.v_head_dim = attn0.v_head_dim
        # The worker's cache contract: one latent per token and layer, no V.
        self._num_attn_layers = len(self.model.layers)
        self._num_heads = attn0.num_heads_tp
        self._num_kv_heads = 1
        self._head_dim = attn0.kv_cache_head_dim
        self._kv_factor = 1

        rope = _resolve_dspark_mla_rope_params(cfg, draft_config)
        self._mla_rope_params = rope
        self._mla_freqs = None
        self._mla_freqs_cap = None
        mscale = _yarn_get_mscale(rope["scaling_factor"], rope["mscale_all_dim"])
        self.softmax_scale = (mscale * mscale) / math.sqrt(
            self.qk_nope_head_dim + self.qk_rope_head_dim
        )
        # FP8 E4M3 context latents at a unit scale (to_draft_fp8); the worker sizes its pool from it.
        self.context_kv_fp8 = context_kv_dtype == "fp8"
        logger.info(
            f"DFlash2 MLA drafter: {self._num_attn_layers} layers, {self._num_heads} heads per "
            f"rank, block {self.block_size}, window {self._window}, softmax scale "
            f"{self.softmax_scale:.8f}, rope theta {rope['theta']}"
        )

    # -- shape / buffers ---------------------------------------------------

    def _shard_context_kv_by_helix(self, draft_config, attention_backend: str):
        mapping = draft_config.mapping
        if mapping.has_cp_helix():
            mapping = mapping.repurpose_helix_cp_to_tp()
        sharded = mapping.helix_mapping_for_kv_heads(1)
        if sharded is mapping:
            if mapping.helix_repurposed_cp_size > 1:
                raise ValueError("MLA DFlash Helix requires target TP1 without attention DP.")
            return None
        if draft_config.pretrained_config.num_attention_heads % sharded.cp_size:
            raise ValueError("MLA DFlash attention heads must divide evenly over Helix ranks.")
        frozen, draft_config._frozen = draft_config._frozen, False
        draft_config.mapping = sharded
        draft_config._frozen = frozen
        logger.info(
            f"MLA DFlash2 context KV is page-sharded over Helix CP{sharded.cp_size}; "
            f"rank {sharded.cp_rank} owns global pages g % {sharded.cp_size} == {sharded.cp_rank}"
        )
        return sharded

    def _validate_gqa_shape(self):
        """The MLA precondition in place of the base's GQA one: one MLA shape on every layer."""
        layers = getattr(self.model, "layers", None)
        if not layers:
            return
        keys = (
            "num_heads_tp",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "kv_lora_rank",
            "v_head_dim",
        )
        base = tuple(getattr(layers[0].self_attn, k) for k in keys)
        for idx, layer in enumerate(layers[1:], start=1):
            if tuple(getattr(layer.self_attn, k, None) for k in keys) != base:
                raise ValueError(
                    f"DFlash2 MLA drafter layer {idx} differs from layer 0 ({dict(zip(keys, base))})."
                )

    def _init_rope(self):
        """No shared RoPE cache: the MLA path rotates only the rope slice (_mla_freqs_cis)."""
        self._rope_initialized = True

    def _mla_freqs_cis(self, device) -> torch.Tensor:
        """``[positions, rope_dim // 2]`` complex64 YaRN table, sized from the worker's ceiling."""
        rope = dict(self._mla_rope_params)
        if self._runtime_position_ceiling is not None:
            rope["max_positions"] = int(self._runtime_position_ceiling)
        if self._mla_freqs is None or self._mla_freqs_cap != rope["max_positions"]:
            self._mla_freqs = build_dspark_mla_yarn_freqs_cis(
                dim=self.qk_rope_head_dim,
                base=rope["theta"],
                scaling_factor=rope["scaling_factor"],
                original_max_position_embeddings=rope["original_max_positions"],
                max_position_embeddings=rope["max_positions"],
                beta_fast=rope["beta_fast"],
                beta_slow=rope["beta_slow"],
                mscale=rope["mscale"],
                mscale_all_dim=rope["mscale_all_dim"],
                device=device,
            )
            self._mla_freqs_cap = rope["max_positions"]
        return self._mla_freqs

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self._fused_kv_weight = None
        self._kv_a_norm_stacked = None
        self._mla_freqs = None
        self._window_attn_buffers = None
        self._window_attn_rows = 0
        return self

    def _build_fused_kv_buffers(self) -> None:
        """Stack every layer's latent projection and norm; split ``kv_b`` into W_UK / W_UV."""
        if self._fused_kv_weight is not None:
            return
        attns = [layer.self_attn for layer in self.model.layers]
        self._fused_kv_weight = torch.cat([a.kv_a_proj_with_mqa.weight for a in attns], 0)
        self._fused_kv_weight = self._fused_kv_weight.contiguous()
        self._fused_kv_bias = None
        self._kv_a_norm_stacked = torch.stack([a.kv_a_layernorm.weight.data for a in attns])
        self._kv_a_norm_eps = attns[0].kv_a_layernorm.variance_epsilon
        self._input_ln_eps = None
        for a in attns:
            w = a.kv_b_proj.weight.view(a.num_heads_tp, self.qk_nope_head_dim + self.v_head_dim, -1)
            a._k_b_proj = w[:, : self.qk_nope_head_dim].contiguous()  # [H, nope, latent]
            a._v_b_proj = w[:, self.qk_nope_head_dim :].contiguous()  # [H, v, latent]

    # -- context cache -----------------------------------------------------

    def precompute_context_kv(self, projected_hidden, positions):
        """Context latents of every layer: ``([N, L, 1, kv_lora_rank + rope], None)``.

        ``projected_hidden`` is ``hidden_norm(fc(captures))``; no input norm, no convolution.
        """
        if self._fused_kv_weight is None:
            self._build_fused_kv_buffers()
        n = projected_hidden.shape[0]
        weight = self._fused_kv_weight
        latent = F.linear(projected_hidden.to(weight.dtype), weight)
        latent = latent.view(n, self._num_attn_layers, self._head_dim)
        # [RMSNorm | RoPE] of every layer's latent in one launch (bitwise mla_latent_norm_rope).
        out = k31_dflash2_mla_ctx_norm_rope(
            latent,
            self._kv_a_norm_stacked,
            self._kv_a_norm_eps,
            self._mla_freqs_cis(latent.device),
            positions.view(-1),
        )
        return out.unsqueeze(2), None

    # -- block decode ------------------------------------------------------

    def _block_attention(self, query, pool, rows, lengths):
        """``[N, H, kv_lora_rank]``: the window decode of ``query`` over ``pool``'s ``rows``
        (the TRTLLM-Gen sparse MLA kernel, ``ops/spec/dflash2_window_attention``, on SM 10.3; its
        exact FP32 reference ``sparse_mla_rows_eager`` elsewhere)."""
        fp8 = pool.dtype == torch.float8_e4m3fn
        if not self._window_attn_trtllm_gen:
            return sparse_mla_rows_eager(
                query,
                pool.reshape(-1, pool.shape[-1]),
                rows,
                self.softmax_scale,
                self.kv_lora_rank,
                kv_scale=DRAFT_FP8_KV_SCALE if fp8 else 1.0,
            )
        if self._window_attn_buffers is None or self._window_attn_rows < query.shape[0]:
            # Sized for the worker's largest step (max requests x block width) on the first,
            # eager, call: never allocate during CUDA-graph capture.
            rows_cap = max(query.shape[0], self._max_block_rows)
            self._window_attn_buffers = dflash2_window_attention_buffers(
                rows_cap, query.shape[1], query.dtype, query.device
            )
            self._window_attn_rows = rows_cap
        workspace, counter = self._window_attn_buffers
        return k31_dflash2_window_attention(
            query,
            pool,
            rows,
            lengths,
            self.softmax_scale,
            self.kv_lora_rank,
            workspace,
            counter,
        )

    def dflash_forward(
        self,
        noise_embedding,
        query_positions,
        num_ctx_per_req,
        ctx_k_cache,
        ctx_v_cache,
        ctx_cache_batch_idx,
        ctx_kv_cache=None,
        ctx_page_table=None,
        block_placement=None,
    ):
        """Block decode of ``[B, block, hidden]`` noise embeddings; returns ``[B * block, hidden]``.

        ``ctx_kv_cache`` is the paged latent pool (``[pages, 1, 1, page_size, 576]`` per layer),
        ``ctx_page_table`` its page table, whose rows ``ctx_cache_batch_idx`` names. The block's
        latents are stored at ``ctx_len + j`` (the slots the next step's accepted rows overwrite)
        before each layer's decode. With a context ring (``_ctx_ring_rows``) position ``p`` is
        table row ``p mod _ctx_ring_rows``.
        """
        del ctx_k_cache, ctx_v_cache, block_placement
        if ctx_kv_cache is None or ctx_page_table is None:
            raise NotImplementedError("The DFlash2 MLA drafter reads a paged context pool only.")
        if self._fused_kv_weight is None:
            self._build_fused_kv_buffers()
        batch, block_size, hidden = noise_embedding.shape
        num_tokens = batch * block_size
        device = noise_embedding.device
        num_heads = self._num_heads
        latent_dim = self.kv_lora_rank

        ctx_len = num_ctx_per_req[:batch].long()
        slots = (
            torch.arange(batch, device=device)
            if ctx_cache_batch_idx is None
            else ctx_cache_batch_idx.long()
        )
        page_tables = ctx_page_table.index_select(0, slots).long()
        page_size = ctx_kv_cache[0].shape[-2]
        block_pos = ctx_len.view(-1, 1) + torch.arange(block_size, device=device)
        helix = self._helix_mapping
        if helix is None:
            block_pos = dflash2_ring_positions(block_pos, self._ctx_ring_rows)
            store_pages = page_tables.gather(1, block_pos // page_size).view(-1)
            store_offsets = (block_pos % page_size).view(-1)
            rows, lengths = k31_dflash2_window_rows(
                ctx_len,
                page_tables,
                block_size,
                page_size,
                self._window,
                self._window,
                self._ctx_ring_rows,
            )
        else:
            # Only the proposed block is replicated; committed history remains in CP-owned pages.
            store_pages = torch.arange(num_tokens, device=device)
            store_offsets = torch.zeros_like(store_pages)
        freqs_table = self._mla_freqs_cis(device)  # [P, rope/2] complex64
        positions_flat = query_positions.reshape(-1)

        hidden_states = noise_embedding.reshape(num_tokens, hidden)
        residual = None
        for layer_idx, layer in enumerate(self.model.layers):
            attn = layer.self_attn
            if residual is None:
                # A copy: the fused add + norm below updates the residual in place.
                residual = hidden_states.clone()
                norm = layer.input_layernorm
                h = flashinfer_rmsnorm(hidden_states, norm.weight, norm.variance_epsilon)
            else:
                norm = layer.input_layernorm
                h, residual = flashinfer_fused_add_rmsnorm(
                    hidden_states, residual, norm.weight, norm.variance_epsilon
                )
            conv_key, taps, group_size = self.attention_convs[layer_idx]
            attn_coefficients = F.linear(h, self.w[conv_key + "projection"]).view(
                h.shape[0], 2, taps, h.shape[-1] // group_size
            )
            h = k31_dflash2_grouped_conv(
                h, attn_coefficients[:, 0], self.w[conv_key + "base"][0], block_size, group_size
            )

            q_a_norm = attn.q_a_layernorm
            q = F.linear(
                flashinfer_rmsnorm(
                    F.linear(h, attn.q_a_proj.weight), q_a_norm.weight, q_a_norm.variance_epsilon
                ),
                attn.q_b_proj.weight,
            )
            q = q.view(num_tokens, num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim)
            query = torch.empty(
                num_tokens,
                num_heads,
                latent_dim + self.qk_rope_head_dim,
                dtype=q.dtype,
                device=device,
            )
            # Absorb W_UK into the query: [H, N, nope] @ [H, nope, latent].
            query[..., :latent_dim] = torch.bmm(
                q[..., : self.qk_nope_head_dim].transpose(0, 1), attn._k_b_proj.to(q.dtype)
            ).transpose(0, 1)
            # One launch: query[..., latent_dim:] = RoPE(q's rope half), and the block latents
            # [RMSNorm | RoPE] stored into the pool (FP8 pools take to_draft_fp8's bytes).
            pool = ctx_kv_cache[layer_idx][:, 0, 0]  # [pages, page_size, 576]
            block_pool = pool if helix is None else pool.new_empty((num_tokens, 1, pool.shape[-1]))
            k31_dflash2_mla_block_rope(
                q,
                query,
                F.linear(h, attn.kv_a_proj_with_mqa.weight),
                attn.kv_a_layernorm.weight,
                attn.kv_a_layernorm.variance_epsilon,
                freqs_table,
                positions_flat,
                block_pool,
                store_pages,
                store_offsets,
            )

            if helix is None:
                ctx_out = self._block_attention(query, pool, rows, lengths)
            else:
                ctx_out = helix_draft_attention(
                    query,
                    pool,
                    page_tables,
                    ctx_len,
                    block_pool[:, 0],
                    block_size,
                    self._window,
                    self.softmax_scale,
                    latent_dim,
                    helix,
                )
            attn_out = torch.einsum("nhc,hvc->nhv", ctx_out, attn._v_b_proj.to(ctx_out.dtype))
            if helix is not None:
                heads = attn.num_heads_tp_cp
                attn_out = attn_out[:, helix.cp_rank * heads : (helix.cp_rank + 1) * heads]
            a = F.linear(attn_out.reshape(num_tokens, -1), attn.o_proj.weight)
            if attn.output_all_reduce is not None:
                a = allreduce_module(
                    a, attn.output_all_reduce, int(AllReduceFusionOp.NONE), None, None
                )
            a = k31_dflash2_grouped_conv(
                a, attn_coefficients[:, 1], self.w[conv_key + "base"][1], block_size, group_size
            )

            norm = layer.post_attention_layernorm
            h2, residual = flashinfer_fused_add_rmsnorm(
                a, residual, norm.weight, norm.variance_epsilon
            )
            conv_key, taps, group_size = self.mlp_convs[layer_idx]
            mlp_coefficients = F.linear(h2, self.w[conv_key + "projection"]).view(
                h2.shape[0], 2, taps, h2.shape[-1] // group_size
            )
            h2 = k31_dflash2_grouped_conv(
                h2, mlp_coefficients[:, 0], self.w[conv_key + "base"][0], block_size, group_size
            )
            gate_up = F.linear(h2, layer.gate_up)
            mlp = F.linear(torch.ops.trtllm.silu_and_mul(gate_up), layer.down)
            if layer.mlp_all_reduce is not None:
                mlp = allreduce_module(
                    mlp, layer.mlp_all_reduce, int(AllReduceFusionOp.NONE), None, None
                )
            hidden_states = k31_dflash2_grouped_conv(
                mlp, mlp_coefficients[:, 1], self.w[conv_key + "base"][1], block_size, group_size
            )

        norm = self.model.norm
        hidden_states, _ = flashinfer_fused_add_rmsnorm(
            hidden_states, residual, norm.weight, norm.variance_epsilon
        )
        return hidden_states


# ====================================================================================================================
# DFlash2 speculative worker
# ====================================================================================================================


def dflash_context_dtype(draft_model: nn.Module) -> torch.dtype:
    """Use projection activations for the context cache, not packed weight storage (FP8 E4M3 for
    a single-latent drafter that asks for FP8 context latents)."""
    if getattr(draft_model, "context_kv_fp8", False) and getattr(draft_model, "_kv_factor", 2) == 1:
        return torch.float8_e4m3fn
    projection = getattr(draft_model, "fc", None)
    if projection is None:
        return torch.bfloat16
    return getattr(projection, "dtype", projection.weight.dtype)


def _dflash2_proposal_store(worker, spec_metadata) -> DFlash2ProposalStore:
    """The worker's :class:`DFlash2ProposalStore`, created on first use.

    Rows follow the engine's slot-indexed rejection buffers (the sequence-slot pool plus the
    padding row ``batch_slot_ids`` sends CUDA-graph padding requests to). First use is an
    eager forward (graph warmups run before every capture).
    """
    store = getattr(worker, "_dflash2_proposals", None)
    if store is not None:
        return store
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("DFlash2 proposal state must be allocated before CUDA graph capture")
    draft_probs = getattr(spec_metadata, "draft_probs", None)
    if draft_probs is not None:
        rows = int(draft_probs.shape[0])
    else:
        rows = int(spec_metadata.num_seq_slots or spec_metadata.max_num_requests) + 1
    draft_model = worker._draft_model_ref()
    store = DFlash2ProposalStore(
        rows,
        worker.max_draft_len,
        draft_model._dflash2_selector_top_k,
        spec_metadata.batch_slot_ids.device,
    )
    worker._dflash2_proposals = store
    from tensorrt_llm.logger import logger

    logger.info(
        f"DFlash2 sampled decoding: selector walk over top-{store.top_k} candidates, "
        f"{os.environ.get('TLLM_K31_DFLASH2_VERIFY', 'block')} verification, "
        f"{rows} proposal slots"
    )
    return store


def _dflash2_sample_draft(
    worker,
    draft_model,
    gen_logits: torch.Tensor,
    block_hidden_states: torch.Tensor,
    anchor_tokens: torch.Tensor,
    spec_metadata,
    num_contexts: int,
    batch_size: int,
) -> torch.Tensor:
    """Draft tokens of a sampling batch: the selector's lattice over each slot's global top-k
    candidates, one sampled walk (:func:`dflash2_selector_walk`, greedy rows the argmax), and
    the proposals saved by sequence slot for the next step's verification.

    Args:
        gen_logits: ``[num_gens, K, vocab or shard]`` drafter logits of the block slots.
        block_hidden_states: ``[num_gens, K, hidden]`` drafter hidden states of those slots.
        anchor_tokens: ``[num_gens]`` the last accepted token of each request.
    Returns:
        ``[num_gens, K]`` int32 draft tokens.
    """
    bank = draft_model.w
    top_k = draft_model._dflash2_selector_top_k
    candidate_ids, unary_logits, _ = worker._dflash2_global_top_k(
        gen_logits,
        spec_metadata,
        top_k=top_k,
        full_vocab=bank["predecessor_codebook"].shape[0],
    )
    seed, offset = worker._rng_state_per_request(
        spec_metadata, num_contexts, batch_size, slot=spec_metadata.rng_draft_slot(0)
    )
    temperatures = spec_metadata.request_temperatures[num_contexts:batch_size]
    store = _dflash2_proposal_store(worker, spec_metadata)
    slot_ids = spec_metadata.batch_slot_ids[num_contexts:batch_size]
    if top_k == K31_SELECTOR_TOP_K and bank["predecessor_codebook"].shape[1] % 256 == 0:
        # The selector op takes this module's torch walk as its reference: imported on use.
        from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_selector_draft import (
            k31_dflash2_selector_draft,
        )
        from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import GREEDY_TEMPERATURE_THRESHOLD

        # Edge scores, the sampled walk (greedy rows the argmax) and the proposal store in one
        # CuTe DSL op, bitwise the torch selector, dflash2_selector_walk and the store below.
        projection = bank["hidden_projection__weight"]
        return k31_dflash2_selector_draft(
            bank["predecessor_codebook"],
            bank["successor_codebook"],
            candidate_ids,
            unary_logits,
            F.linear(block_hidden_states.to(projection.dtype), projection),
            anchor_tokens,
            seed,
            offset,
            temperatures,
            GREEDY_TEMPERATURE_THRESHOLD,
            store.candidate_ids,
            store.q,
            store.tokens,
            slot_ids,
        )
    # Other selector shapes (the kernels hold 32 candidates and a rank of whole 256 blocks).
    projection = bank["hidden_projection__weight"]
    edge_scores = dflash2_score_edges(
        bank["predecessor_codebook"],
        bank["successor_codebook"],
        candidate_ids,
        unary_logits,
        F.linear(block_hidden_states.to(projection.dtype), projection),
        anchor_tokens.long(),
    )
    tokens, q, _ = dflash2_selector_walk(
        candidate_ids,
        edge_scores,
        dflash2_uniforms(seed, offset, candidate_ids.shape[1]),
        temperatures,
        greedy_request_mask(temperatures),
    )
    store.store(slot_ids, candidate_ids, q, tokens)
    return tokens.to(torch.int32)


class DFlashWorker(_BaseDFlashWorker):
    """The engine's DFlash worker with the Kimi K3.1 DFlash2 differences.

    Every method not defined here is the engine ``DFlashWorker``'s, unchanged. The overrides carry what
    the K3.1 target runs differently: the DFlash2 sampled selector walk and block verification
    (``sample_and_accept_draft_tokens``), the MLA drafter's single-latent context store (FP8
    context latents, the window-bounded context ring, the rank-local (helix) store), the
    device-decoded block tables of a draft KV pool, the input staging of the engine scope, and the
    first drafter inputs built from them.
    """

    def __init__(
        self,
        spec_config: "DFlashDecodingConfig",
        mapping: Mapping,
        use_separate_draft_kv_cache: bool = False,
    ):
        super().__init__(spec_config, mapping, use_separate_draft_kv_cache)
        # Rows per slot when the private paged arena is a context ring bounded by the drafter's
        # window (ops/spec/dflash2_ring): position p lives at slot row p mod _ctx_ring_rows. None when
        # the arena is dense in position or the pages are the manager's.
        self._ctx_ring_rows = None

    def set_draft_model(self, draft_model) -> None:
        super().set_draft_model(draft_model)
        self._helix_mapping = getattr(draft_model, "_helix_mapping", None)

    def _helix_shard(self) -> Optional[HelixDraftShard]:
        """This rank's share of the context pages when the drafter is helix-sharded."""
        helix = getattr(self, "_helix_mapping", None)
        if helix is None:
            return None
        return HelixDraftShard(helix.cp_size, helix.cp_rank, self._ctx_page_size)

    def _local_draft_tokens(self, draft_model) -> Optional[VocabParallelDraftTokens]:
        """How greedy draft tokens come from the drafter's own full-vocab LM head
        output on this rank when there is no vocab-parallel selection (e.g. under
        attention DP), or None for the logits path. Plain DFlash has none."""
        del draft_model
        return None

    def _init_ctx_block_offsets(self, draft_kv_cache_manager, pool, L, nkv, hd, page_size) -> bool:
        if getattr(draft_kv_cache_manager, "has_standalone_draft_cache", False):
            return super()._init_ctx_block_offsets(
                draft_kv_cache_manager, pool, L, nkv, hd, page_size
            )
        """Validate the draft KV pool layout and resolve its offset decode constant.

        ``kv_cache_block_offsets`` entries are ``pool_block_index *
        num_pool_layers * kv_factor`` plus the K/V field index. What that has to
        be divided by to index a ``get_buffers`` view depends on the manager --
        see the stride check below; attention_backend/utils.py hardcodes the V1
        divisor and is not a general contract.
        """
        mgr = draft_kv_cache_manager
        pool_mapping = mgr.kv_cache_pool_mapping
        layer_offset = mgr.layer_offsets[0]
        self._ctx_pool_idx = int(pool_mapping[layer_offset, 0])
        num_pool_layers = int((pool_mapping[:, 0] == self._ctx_pool_idx).sum())
        if num_pool_layers != L:
            logger.warning(
                f"DFlash: draft KV pool {self._ctx_pool_idx} spans {num_pool_layers} layers but "
                f"the drafter has {L}; using the private ctx arena."
            )
            return False
        # V1 slices a page-major pool, so stride(0) spans every layer; V2 wraps
        # the layer base in a dense view over interleaved pages, so stride(0) is
        # one layer slot. Guessing wrong is silent: acceptance quietly drops.
        layer_block_elems = mgr.kv_factor * nkv * page_size * hd
        stride0 = pool[0].stride(0)
        if stride0 == layer_block_elems:
            self._ctx_block_divisor = mgr.kv_factor
        elif stride0 == num_pool_layers * layer_block_elems:
            self._ctx_block_divisor = num_pool_layers * mgr.kv_factor
        else:
            logger.warning(
                f"DFlash: draft KV pool layer stride {stride0} matches neither V1's "
                f"page-major slice ({num_pool_layers * layer_block_elems}) nor V2's "
                f"interleaved view ({layer_block_elems}); using the private ctx arena "
                f"rather than guessing block indices."
            )
            return False
        logger.info(
            f"DFlash: ctx block offsets pool_idx={self._ctx_pool_idx}, "
            f"divisor={self._ctx_block_divisor}, view_pages={pool[0].size(0)}"
        )
        return True

    def _init_ctx_block_tables(self, num_slots: int, max_blocks: int) -> None:
        self._ctx_block_tables = torch.zeros(
            (num_slots, max_blocks), dtype=torch.int32, device="cuda"
        )
        self._ctx_block_counts = torch.zeros(num_slots, dtype=torch.long, device="cuda")
        logger.info(f"DFlash: ctx block tables {tuple(self._ctx_block_tables.shape)}")

    def _refresh_ctx_block_tables(self, attn_metadata, num_seqs: int, request_ids=None) -> bool:
        if self._has_unified_draft_cache():
            return super()._refresh_ctx_block_tables(attn_metadata, num_seqs, request_ids)
        """Decode this iteration's draft block table into the persistent buffer.

        In place and fixed-shape so a captured graph keeps reading the same
        tensor. Returns False when there is no table to fill, i.e. every path
        that stayed on the private arena.
        """
        if self._ctx_block_tables is None or num_seqs <= 0:
            return False
        src = getattr(attn_metadata, "draft_kv_cache_block_offsets", None)
        if src is None:
            # The table is bound but unfillable. Leaving it at zeros would send
            # every request's every layer to pool block 0 -- silently.
            raise RuntimeError(
                "DFlash: bound to the draft KV pool but "
                f"{type(attn_metadata).__name__} carries no "
                "draft_kv_cache_block_offsets this iteration."
            )
        # Field 0 of the K/V pair: both name the same block for a kv_factor=2
        # pool, and the drafter addresses blocks, not planes. The block count
        # is taken before the clamp erases the placeholders; it is informative
        # only on the python V2 backend, the cpp one already mapped
        # BAD_PAGE_INDEX to 0.
        decode_draft_block_table(
            src[self._ctx_pool_idx, :num_seqs, 0],
            self._ctx_block_divisor,
            self._ctx_block_tables,
            self._ctx_block_counts,
        )
        return True

    def _lazy_init_ctx_buffers(
        self, draft_model, spec_metadata, attn_metadata, draft_kv_cache_manager=None
    ):
        if getattr(draft_kv_cache_manager, "has_standalone_draft_cache", False):
            super()._lazy_init_ctx_buffers(
                draft_model, spec_metadata, attn_metadata, draft_kv_cache_manager
            )
            self._ctx_ring_rows = draft_model._ctx_ring_rows = None
            draft_model._max_block_rows = spec_metadata.max_num_requests * self._compute_block_size
            return
        self._validate_draft_attention_backend(draft_model)
        # Only TrtllmAttentionMetadata builds draft_kv_cache_block_offsets.
        # Without it the table stays all-zero and every request would read and
        # write pool block 0 -- silently, at acceptance 1.0.
        if draft_kv_cache_manager is not None and not hasattr(
            attn_metadata, "draft_kv_cache_block_offsets"
        ):
            logger.warning(
                f"DFlash: {type(attn_metadata).__name__} carries no draft KV block offsets; "
                "using the private ctx arena."
            )
            draft_kv_cache_manager = None
        if self._ctx_buf_inited:
            if self._ctx_kv_manager is draft_kv_cache_manager:
                return
            # KV cache estimation swaps a probe manager for the real one.
            # Holding the old pool's tensors across that writes freed memory.
            logger.info("DFlash: draft KV cache manager replaced, rebinding the ctx cache.")
            self._ctx_buf_inited = False

        # Recorded even when the buffers stay private, so that the estimation
        # pass falling back to the arena still rebinds once the real pool lands.
        self._ctx_kv_manager = draft_kv_cache_manager
        self._ctx_ring_rows = None
        if hasattr(draft_model, "_ctx_ring_rows"):
            draft_model._ctx_ring_rows = None
        self._ctx_block_tables = None
        self._ctx_block_counts = None
        max_batch = spec_metadata.max_num_requests

        # ctx_len is 1:1 with the target's positions, so max_seq_len bounds it.
        # max_position_embeddings does not: K3's drafter advertises 1048576,
        # which sizes the arena at hundreds of GiB.
        config = getattr(draft_model, "config", None)
        max_pos = getattr(config, "max_position_embeddings", None) if config else None
        # Take the served length from the KV cache manager, which every backend
        # carries on the metadata base class. A `max_seq_len` attribute is only
        # declared by some metadata classes (TrtllmAttentionMetadata has one --
        # a wrapper returning this same manager value; the base class and
        # FlashInferAttentionMetadata do not), so reading the attribute alone
        # lets max_position_embeddings win the min() below and sizes the arena
        # for the advertised positions instead of the served length. The
        # attribute stays as the fallback for no-cache attention, where it is
        # set by hand and there is no manager to ask.
        cache_manager = getattr(attn_metadata, "kv_cache_manager", None)
        if getattr(cache_manager, "is_disagg", False) and not getattr(
            cache_manager, "kda_context_only", False
        ):
            # A disaggregated generation server: the prompts ran on a context server and no
            # drafter context is transferred, so every request's drafter context starts empty.
            logger.warning_once(
                "DFlash2 drafter has no prompt context on a disaggregated generation server: "
                "acceptance is reduced until drafter-context transfer exists.",
                key="k31_dflash2_no_drafter_context",
            )
        runtime_max = getattr(cache_manager, "max_seq_len", None)
        if runtime_max is None:
            runtime_max = getattr(attn_metadata, "max_seq_len", None)
        candidates = [c for c in (runtime_max, max_pos) if c is not None]
        self._max_ctx = min(candidates) if candidates else 8192
        # py_executor_creator inflates max_seq_len by at most this much for spec
        # dec (one term is overlap-only), so compare against what is served.
        from tensorrt_llm._torch.speculative.utils import get_num_extra_kv_tokens

        spec_slack = 2 * (self.spec_config.tokens_per_gen_step - 1) + get_num_extra_kv_tokens(
            self.spec_config
        )
        if runtime_max is not None and self._max_ctx < runtime_max - spec_slack:
            # Warn, not raise: the per-request skip in _store_prefill_context
            # keeps an oversized request from taking the executor down.
            logger.warning(
                f"DFlash drafter covers {self._max_ctx} positions but the engine serves "
                f"up to {runtime_max - spec_slack}. max_position_embeddings={max_pos} is "
                "the binding constraint, so requests past it will draft nothing."
            )

        dtype = dflash_context_dtype(draft_model)

        # Reserve slot index max_batch as a scratch slot for padding/unknown
        # dummies; real requests only draw slots 0..max_batch-1, so dummy
        # writes land here and can't corrupt a real request's context.
        self._dummy_slot = max_batch
        num_slots = max_batch + 1
        # CUDA-graph padding uses ids counting down from
        # CUDA_GRAPH_DUMMY_REQUEST_ID; a floor of one draft block separates them
        # from real ids. Imported lazily to avoid an import cycle.
        from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID

        self._graph_dummy_id_floor = CUDA_GRAPH_DUMMY_REQUEST_ID - self.max_draft_len

        self._ctx_len = torch.zeros(num_slots, dtype=torch.long, device="cuda")
        self._ctx_len_host = [0] * num_slots
        self._batch_to_slot_buffer = DeviceCopyBuffer(
            torch.zeros(max_batch, dtype=torch.long, device="cuda"),
            pin_memory=prefer_pinned(),
        )

        self._free_slots = deque(range(max_batch))
        self._req_to_slot = {}
        self._req_ctx_pos = {}

        # checkpoint's trained block width
        self._resolved_block_size = getattr(draft_model, "block_size", None) or (
            self.max_draft_len + 1
        )
        # what the draft forward actually computes
        self._compute_block_size = self._draft_block_width(draft_model)
        if hasattr(draft_model, "_max_block_rows"):
            draft_model._max_block_rows = max_batch * self._compute_block_size
        if self._compute_block_size > self._resolved_block_size:
            slack = self._compute_block_size - self.max_draft_len
            raise ValueError(
                f"DFlash checkpoint was trained with block_size={self._resolved_block_size}."
                f"Lower max_draft_len to at most {self._resolved_block_size - slack}."
                f"Current max_draft_len={self.max_draft_len}."
            )

        # +block_size slack lets each backend append the per-iteration noise
        # K/V after the accumulated request context.
        assert hasattr(draft_model, "_build_fused_kv_buffers"), (
            "DFlash draft model must define _build_fused_kv_buffers."
        )
        draft_model._build_fused_kv_buffers()
        L = draft_model._num_attn_layers
        nkv = draft_model._num_kv_heads
        hd = draft_model._head_dim
        kv_factor = getattr(draft_model, "_kv_factor", 2)
        if kv_factor != 1:
            raise ValueError(
                "The Kimi K3.1 DFlash worker serves the DFlash2 MLA drafter: one latent per token "
                f"and layer (kv_factor 1), got kv_factor {kv_factor}."
            )
        capacity = self._max_ctx + self._compute_block_size
        # Published before any drafter table is built, so a drafter may read it
        # instead of its config cap. _max_ctx is attn_metadata.max_seq_len, which
        # py_executor_creator raises past model_config.max_seq_len and never
        # writes back. _compute_block_size: the block decode's j runs over it.
        draft_model._runtime_position_ceiling = dflash_position_ceiling(
            self._max_ctx, self._compute_block_size, self.max_draft_len
        )
        # The MLA drafter pages its context (to get its footprint under
        # free_gpu_memory_fraction) and reads the pages itself.
        use_paged = getattr(draft_model, "_paged_ctx_cache", False)
        self._ctx_paged = use_paged
        if self._ctx_paged:
            pool = self._managed_ctx_pool(draft_kv_cache_manager, L, nkv, hd, dtype, kv_factor)
            # The manager's page size wins when bound to it: its pool is already
            # carved, so the drafter adopts the geometry rather than imposing one.
            page_size = self._ctx_page_size if pool is None else pool[0].size(-2)
            self._ctx_page_size = page_size
            # Settle the block table before committing to the pool: it is the
            # last thing that can rule the pool out, and falling back after
            # taking the pool branch would leave no buffer allocated at all.
            if pool is not None and not self._init_ctx_block_offsets(
                draft_kv_cache_manager, pool, L, nkv, hd, page_size
            ):
                pool = None
            if pool is None:
                # --- DFlash2 context ring (ops/spec/dflash2_ring) ---
                # A drafter that reads only a sliding window keeps W + verify + block rows per
                # slot instead of max_seq_len + block.
                ring_window = getattr(draft_model, "_ctx_ring_window", None)
                if ring_window and self._helix_shard() is None:
                    ring_rows = dflash2_ring_rows(
                        int(ring_window),
                        self.max_draft_len + 1,
                        self._compute_block_size,
                        page_size,
                    )
                    if ring_rows < capacity:
                        self._ctx_ring_rows = ring_rows
                        draft_model._ctx_ring_rows = ring_rows
                        capacity = ring_rows
                        logger.info(
                            f"DFlash: drafter ctx arena is a ring of {ring_rows} rows per slot "
                            f"(window {ring_window}, verify {self.max_draft_len + 1}, block "
                            f"{self._compute_block_size}, page {page_size}) for max_ctx {self._max_ctx}"
                        )
                # --- end DFlash2 context ring ---
                # Per-slot reservation is an arena-only notion; the pool path
                # takes its pages from the manager one iteration at a time.
                self._ctx_pages_per_slot = (capacity + page_size - 1) // page_size
                total_pages = num_slots * self._ctx_pages_per_slot
                self._check_ctx_arena_fits(capacity, num_slots, L, nkv, hd, dtype, kv_factor)
                # HND layout: [L, pages, kv_factor, Hkv, page, D].
                self._ctx_kv_buf = torch.zeros(
                    (L, total_pages, kv_factor, nkv, page_size, hd),
                    dtype=dtype,
                    device="cuda",
                )
                self._ctx_page_table = torch.arange(
                    total_pages, dtype=torch.int32, device="cuda"
                ).view(num_slots, self._ctx_pages_per_slot)
            else:
                # Nothing reserved: pages come from the manager's per-request
                # block table one iteration at a time, so the footprint follows
                # the sequences served rather than max_batch x max_seq_len.
                self._ctx_kv_buf = pool
                self._init_ctx_block_tables(num_slots, draft_kv_cache_manager.max_blocks_per_seq)
                # Only a manager that publishes and matches its own blocks can
                # hand back a reused prefix's drafter K/V; the private arena and
                # an unpaired draft pool both start every request at 0.
                self._ctx_reuse_addressable = bool(
                    getattr(draft_kv_cache_manager, "enable_joint_kv_cache_reuse", False)
                )
        else:
            self._check_ctx_arena_fits(capacity, num_slots, L, nkv, hd, dtype, kv_factor)
            # One latent per token: the MLA drafter's context has no V half.
            self._ctx_k_buf = torch.zeros(
                (num_slots, L, capacity, nkv, hd), dtype=dtype, device="cuda"
            )
        self._ctx_buf_inited = True

        if not self._ctx_reuse_addressable and getattr(
            draft_kv_cache_manager, "enable_joint_kv_cache_reuse", False
        ):
            # Every fallback above (shape mismatch, stride mismatch, pool layer
            # count) logs its own reason and then costs only memory -- except
            # this one, where the scheduler goes on matching prefixes the
            # drafter can no longer read. Silent, and it looks exactly like the
            # unpaired build in every metric but acceptance length.
            logger.warning(
                "DFlash: the draft KV cache manager joins block reuse but the drafter "
                "fell back to a private ctx arena, so a reused prefix's drafter K/V "
                "stays unreachable and acceptance length drops as if unpaired."
            )

        logger.info(
            f"DFlash: allocated ctx buffers: max_batch={max_batch}, "
            f"max_ctx={self._max_ctx}, dtype={dtype}, "
            f"dflash_attention_backend={self._dflash_attention_backend}, "
            f"reuse_addressable={self._ctx_reuse_addressable}"
        )

    def _store_context_kv_paged(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        rows: torch.Tensor,
        positions: torch.Tensor,
        owned: Optional[torch.Tensor] = None,
    ) -> None:
        """Write the MLA drafter's context latents ``k`` at rank-local ``positions`` by direct
        page writes into plane 0 (one latent per token and layer: ``v`` is None).

        ``owned`` (helix): False entries rewrite their slot's current value,
        which keeps a fixed-size, CUDA-graph-safe store.
        """
        if v is not None:
            raise ValueError("The DFlash2 MLA drafter stores one latent per token (v is None).")
        if self._ctx_ring_rows and self._ctx_block_tables is None:
            # DFlash2 context ring: position p is slot row p mod _ctx_ring_rows.
            positions = dflash2_ring_positions(positions, self._ctx_ring_rows)
        fp8_cache = self._ctx_kv_buf[0].dtype == torch.float8_e4m3fn
        table = (
            self._ctx_block_tables if self._ctx_block_tables is not None else self._ctx_page_table
        )
        page_size = self._ctx_page_size
        pages = table[rows.long(), positions.long() // page_size].long()
        offsets = positions.long() % page_size
        keep = None if owned is None else owned.view(-1, *([1] * (k.dim() - 2)))
        for layer_idx in range(k.size(1)):
            pool = self._ctx_kv_buf[layer_idx]
            if fp8_cache:
                # The latent saturates (to_draft_fp8), as the block decode quantizes the query
                # it is read with; byte views because index_put support varies across builds.
                view = pool.view(torch.uint8)
                new = to_draft_fp8(k[:, layer_idx]).view(torch.uint8)
            else:
                view = pool
                new = k[:, layer_idx].to(pool.dtype)
            if keep is not None:
                new = torch.where(keep, new, view[pages, 0, :, offsets])
            view[pages, 0, :, offsets] = new

    def _prepare_kv_for_draft_forward(
        self,
        attn_metadata,
        num_accepted_tokens: torch.Tensor,
        num_contexts: int,
        batch_size: int,
    ):
        """Adjust kv_lens_cuda so the draft model sees correct RoPE positions."""
        if hasattr(attn_metadata, "kv_lens_cuda"):
            self._kv_rewind_nc = num_contexts
            self._kv_rewind_bs = batch_size
            self._kv_rewind_amount = 1 - num_accepted_tokens[num_contexts:batch_size]
            if batch_size > num_contexts:
                attn_metadata.kv_lens_cuda[num_contexts:batch_size] += 1
            self._kv_rewind_pending = True
            self._update_metadata_for_draft(attn_metadata)

    def _update_metadata_for_draft(self, attn_metadata) -> None:
        """Refresh the attention metadata after the draft forward's kv_lens prep."""
        attn_metadata.update_for_spec_dec()

    # Target-token selection and acceptance, with the engine's kernels called through the family's spec
    # operations (SpecWorkerBase's methods, minus the ragged-verify path, which the target's uniform DFlash2
    # blocks never take: it is the base method's own).
    def _sample_and_accept_draft_tokens_base(
        self, logits, draft_tokens, num_contexts, batch_size, spec_metadata
    ):
        if spec_metadata is not None and spec_metadata.is_ragged_verify:
            return super()._sample_and_accept_draft_tokens_base(
                logits, draft_tokens, num_contexts, batch_size, spec_metadata
            )
        runtime_draft_len = draft_tokens.shape[-1]
        if isinstance(logits, torch.Tensor) and logits.dim() == 1:
            logits = logits.unsqueeze(0)
        target_tokens = self._sample_tokens_for_batch(
            logits, spec_metadata, num_contexts, batch_size
        )
        accepted_tokens, num_accepted_tokens = strict_accept(
            target_tokens, draft_tokens, num_contexts, batch_size
        )
        return accepted_tokens, self._apply_force_accepted_tokens(
            num_accepted_tokens, num_contexts, runtime_draft_len, spec_metadata=spec_metadata
        )

    def _sample_tokens_for_batch(self, logits, spec_metadata, num_contexts, batch_size):
        if isinstance(logits, VocabShardedLogits):
            # VocabShardedLogits.select_tokens: the global argmax of an all-greedy batch, else a
            # temperature-only draw, from this rank's vocab shard.
            exchange = logits.head._exchange
            if spec_metadata.is_all_greedy_sample:
                return vocab_parallel_argmax(
                    logits.local, exchange.all_reduce, exchange.tp_size, exchange.tp_rank
                )
            rows = logits.num_rows
            return vocab_parallel_sample(
                logits.local,
                spec_metadata.temperatures[:rows],
                spec_metadata.seeds[:rows],
                spec_metadata.offsets[:rows],
                GREEDY_TEMPERATURE_THRESHOLD,
                exchange.all_reduce,
                exchange.tp_size,
                exchange.tp_rank,
            )
        if spec_metadata.is_all_greedy_sample:
            return torch.argmax(logits, dim=-1)
        num_tokens = logits.shape[0]
        seed, offset = self._rng_state_per_token(spec_metadata, num_tokens)
        return self._sample_from_logits(
            spec_metadata.advanced_sampling_mode,
            logits,
            spec_metadata.temperatures[:num_tokens],
            spec_metadata.top_ks[:num_tokens],
            spec_metadata.top_ps[:num_tokens],
            spec_metadata.min_ps[:num_tokens],
            seed=seed,
            offset=offset,
        )

    @staticmethod
    def _sample_from_logits(
        advanced_sampling_mode,
        logits,
        temperatures,
        top_ks,
        top_ps,
        min_ps,
        *,
        seed=None,
        offset=None,
    ):
        if advanced_sampling_mode.is_fused:
            return fused_sample_from_logits(
                logits, temperatures, top_ks, top_ps, min_ps, seed, offset
            )
        top_ks, top_ps = flashinfer_sampling.resolve_advanced_sampling_filters(
            advanced_sampling_mode, top_ks, top_ps
        )
        return flashinfer_sample_from_logits(logits, temperatures, top_ks, top_ps, seed, offset)

    @staticmethod
    def _compute_probs_from_logits(
        advanced_sampling_mode, logits, temperatures, top_ks, top_ps, min_ps
    ):
        if advanced_sampling_mode.is_fused:
            return fused_compute_probs_from_logits(logits, temperatures, top_ks, top_ps, min_ps)
        top_ks, top_ps = flashinfer_sampling.resolve_advanced_sampling_filters(
            advanced_sampling_mode, top_ks, top_ps
        )
        return flashinfer_compute_probs_from_logits(logits, temperatures, top_ks, top_ps)

    def _dflash2_global_top_k(self, gen_logits, spec_metadata, top_k: int, full_vocab: int):
        """Per-position global top-k candidates and the tensor to score into (the engine method, its
        all-gather of the vocab shards' (id, value) pairs through the family's all-gather operation).

        The selector's top-32 of BF16 / FP32 logits runs on the CuTe DSL ops
        ``ops/spec/dflash2_global_top32`` (each rank's shard, then the merge of the gathered pairs),
        bitwise ``torch.topk``; other widths and dtypes take ``torch.topk``."""
        shard = gen_logits.shape[-1]
        fast = (
            top_k == K31_SELECTOR_TOP_K
            and gen_logits.dtype in (torch.bfloat16, torch.float32)
            and gen_logits.is_contiguous()
        )
        if shard == full_vocab:
            if fast:
                unary_logits, candidate_ids = k31_dflash2_shard_top32(gen_logits, top_k, 0, False)
            else:
                unary_logits, candidate_ids = torch.topk(gen_logits, top_k, dim=-1)
            return candidate_ids, unary_logits, gen_logits
        mapping = self.mapping
        if not (
            self._draft_logits_are_sharded(gen_logits, spec_metadata)
            and mapping is not None
            and shard * mapping.tp_size == full_vocab
        ):
            raise NotImplementedError(
                f"DFlash 2 candidate selection needs the drafter's full vocab "
                f"({full_vocab}) or a plain TP column shard of it; got draft "
                f"logits of width {shard}."
            )
        if fast:
            # Interleaved (id, value) pairs of this rank's top-32, ids made global.
            combined = k31_dflash2_shard_top32(gen_logits, top_k, mapping.tp_rank * shard, True)
            gathered = allgather_module(combined, mapping, -1, None)
            unary_logits, candidate_ids = k31_dflash2_merge_top32(gathered.contiguous(), top_k)
        else:
            local_values, local_ids = torch.topk(gen_logits, top_k, dim=-1)
            local_ids = local_ids + mapping.tp_rank * shard
            # Interleaved (id, value) pairs.
            combined = torch.stack([local_ids.float(), local_values.float()], dim=-1).flatten(-2)
            gathered = allgather_module(combined, mapping, -1, None)
            unary_logits, selected = torch.topk(gathered[..., 1::2], top_k, dim=-1)
            candidate_ids = gathered[..., 0::2].gather(-1, selected).long()
        block_logits = gen_logits.new_full((*gen_logits.shape[:-1], full_vocab), float("-inf"))
        return candidate_ids, unary_logits, block_logits

    def _target_tokens_from_shard(self, spec_metadata) -> bool:
        """Whether a step handed vocab-sharded target logits selects its target tokens from this
        rank's shard (``vocab_parallel_argmax`` / ``vocab_parallel_sample``) instead of the gathered
        logits.

        Never with a guided decoder: its bitmask masks the gathered ``[rows, vocab]`` target logits
        (``CapturableGuidedDecoder.execute``) before acceptance reads them, and the shard selection
        would read unmasked logits. The engine already drops the vocab-parallel selection when a
        guided decoder is set; this keeps the worker correct whatever order the two are bound in.
        """
        return self.guided_decoder is None and (
            spec_metadata.is_all_greedy_sample or self._vocab_parallel.samples
        )

    def sample_and_accept_draft_tokens(self, logits, attn_metadata, spec_metadata):
        # A DFlash2 sampling batch verifies the selector's block by block verification
        # (dflash2_verify); everything else takes the engine's acceptance.
        if dflash2_sampled_step(self, spec_metadata):
            _dflash2_proposal_store(self, spec_metadata)
            return dflash2_sample_and_accept(self, logits, attn_metadata, spec_metadata)
        return SpecWorkerBase.sample_and_accept_draft_tokens(
            self, logits, attn_metadata, spec_metadata
        )

    def _write_ctx_len(self, updates: dict[int, int]) -> None:
        """Apply a slot-to-length mapping to _ctx_len in one async scatter."""
        if not updates:
            return
        for slot, value in updates.items():  # Update host side mirror
            self._ctx_len_host[slot] = value

        slots, values = zip(*updates.items())
        if stage_scatter(self._ctx_len, slots, values):
            return
        pinned = prefer_pinned()
        slots_pinned = torch.tensor(slots, dtype=torch.long, pin_memory=pinned)
        vals_pinned = torch.tensor(values, dtype=torch.long, pin_memory=pinned)
        self._ctx_len.index_copy_(
            0,
            slots_pinned.to("cuda", non_blocking=True),
            vals_pinned.to("cuda", non_blocking=True),
        )

    def _store_prefill_context(
        self,
        draft_model,
        spec_metadata: "DFlashSpecMetadata",
        attn_metadata,
        position_ids: torch.Tensor,
        total_target_tokens: int,
    ):
        """Capture prefill hidden states and store as initial accumulated context.

        During prefill (context requests), the target model processes all prompt
        tokens.  We project their captured hidden states through fc + hidden_norm
        and store them per-request so the draft model can use the full prompt
        as cross-attention context on subsequent gen steps.
        """
        if self._has_unified_draft_cache():
            return super()._store_prefill_context(
                draft_model, spec_metadata, attn_metadata, position_ids, total_target_tokens
            )
        if not hasattr(draft_model, "project_target_hidden"):
            return

        num_ctx_tokens = attn_metadata.num_ctx_tokens
        if num_ctx_tokens == 0:
            return

        captured_hs = spec_metadata.get_hidden_states(total_target_tokens)
        if captured_hs is None:
            return
        # Project context tokens through fc + hidden_norm
        ctx_proj = draft_model.project_target_hidden(captured_hs[:num_ctx_tokens])

        # Split by request and store/append accumulated context.
        # Context requests may arrive in chunks (chunked prefill), so we must
        # APPEND successive chunks for the same request rather than overwriting.
        # A chunk is only an append if it starts where the last one ended

        # Scalar (non-rope) token indices on the host. The device tensor holds
        # the same values, but reading it costs a D2H sync per request.
        host_pos = spec_metadata.host_position_ids
        # Without them (callers that build the metadata themselves), a context
        # chunk starts at the request's cached-token count, also on the host.
        kv_params = getattr(attn_metadata, "kv_cache_params", None)
        cached_tokens = getattr(kv_params, "num_cached_tokens_per_seq", None)
        ctx_len_updates = {}
        offset = 0
        num_contexts = attn_metadata.num_contexts
        # A context write addresses pages through row i of the block table, so
        # it must fit that row's allocation as well as the arena's length. A
        # device-decoded table needs one tolist() for the whole loop; the
        # lengths come from _ctx_len_host, so the loop body stays sync-free.
        if self._ctx_block_tables is None:
            ctx_alloc = None
        else:
            ctx_alloc = (self._ctx_block_counts[:num_contexts] * self._ctx_page_size).tolist()
        for i in range(num_contexts):
            req_id = spec_metadata.request_ids[i]
            slen = int(attn_metadata._seq_lens[i])
            chunk_proj = ctx_proj[offset : offset + slen].detach()
            chunk_pos = position_ids[offset : offset + slen].long().detach()

            if slen == 0:
                first_pos = 0
            elif host_pos is not None:
                first_pos = int(host_pos[offset])
            elif cached_tokens is not None and i < len(cached_tokens):
                first_pos = int(cached_tokens[i])
            else:
                first_pos = chunk_pos[0].item()

            # A chunk continues what we already stored only if it starts exactly
            # where the previous one ended. Everything else -- a fresh request, a
            # request id reused after completion, a prefill restarted after
            # preemption -- has to start from a clean slot.
            reset = self._req_ctx_pos.get(req_id) != first_pos
            if self._assign_slot(req_id, reset=reset, updates=ctx_len_updates) is None:
                logger.warning("DFlash: no free slots, skipping context store")
                self._req_ctx_pos.pop(req_id, None)
                offset += slen
                continue

            slot = self._req_to_slot[req_id]
            cur = ctx_len_updates.get(slot, self._ctx_len_host[slot])
            cap = self._max_ctx if ctx_alloc is None else min(self._max_ctx, ctx_alloc[i])
            if cur == 0 and first_pos > 0 and self._ctx_reuse_addressable:
                # Reuse let the target skip [0, first_pos), so only the tail
                # reaches the capture buffer. The prefix K/V are still in the
                # paired pool: precompute_context_kv is per-token in (hidden, p).
                cur = first_pos
            if cur + slen > cap:
                # Request-level, like the no-free-slots path above: truncating
                # would silently draft from a stale prefix, but killing the
                # forward would take every other in-flight request with it.
                logger.warning(
                    f"DFlash: ctx overflow on slot {slot} "
                    f"({cur} + {slen} > {cap}); skipping its context "
                    "store, so this request drafts nothing."
                )
                self._req_to_slot.pop(req_id, None)
                self._req_ctx_pos.pop(req_id, None)
                ctx_len_updates[slot] = 0
                self._free_slots.append(slot)
                offset += slen
                continue
            end = cur + slen
            actual = end - cur
            # Anchor to the true position rather than accumulating, and advance
            # even when the slot is full so the next chunk still reads as a
            # continuation.
            self._req_ctx_pos[req_id] = first_pos + slen
            if actual > 0:
                cache_dtype = (
                    self._ctx_kv_buf[0].dtype if self._ctx_paged else self._ctx_k_buf.dtype
                )
                if cache_dtype == torch.float8_e4m3fn:
                    # The K/V projection runs in the drafter's weight dtype;
                    # the store converts its outputs to the cache dtype.
                    cache_dtype = chunk_proj.dtype
                chunk_proj_cast = chunk_proj[:actual].to(cache_dtype)
                ctx_len_updates[slot] = end
                # Precompute post-norm/post-RoPE K,V for this prefill chunk
                # so decode iters can read without re-projecting.
                #
                # RoPE positions must be slot-local (cur..end), NOT the
                # target-forward positions in chunk_pos: gen-time appends and
                # draft queries number positions from _ctx_len, so the stored
                # keys must share that numbering. Without KV-block reuse the
                # two coincide, but on a prefix-cache hit chunk_pos starts at
                # the reused-token count, which would offset every stored key
                # from the numbering the queries use (RoPE attention depends
                # only on position differences, so the uniform slot-local
                # renumbering is exact).
                local_pos = torch.arange(cur, end, dtype=torch.long, device="cuda")
                chunk_k, _ = draft_model.precompute_context_kv(chunk_proj_cast, local_pos)
                # chunk_k: [actual, L, 1, latent] (the MLA drafter has no V).
                if self._ctx_paged:
                    # Manager block tables are keyed by batch position, the
                    # private arena's by slot. See _ctx_paged_index_args.
                    row = i if self._ctx_block_tables is not None else slot
                    positions = local_pos
                    store_k = chunk_k
                    if self._ctx_ring_rows and self._ctx_block_tables is None:
                        # DFlash2 context ring: a chunk longer than the ring wraps onto itself;
                        # only its last _ctx_ring_rows positions can still be read.
                        keep = dflash2_ring_chunk_start(actual, self._ctx_ring_rows)
                        positions, store_k = positions[keep:], store_k[keep:]
                    shard = self._helix_shard()
                    if shard is not None:
                        # Eager prefill: keep only this rank's pages.
                        positions, owned = shard.local_slots(positions)
                        positions, store_k = positions[owned], chunk_k[owned]
                    self._store_context_kv_paged(
                        store_k,
                        None,
                        torch.full((positions.numel(),), row, dtype=torch.long, device="cuda"),
                        positions,
                    )
                if self._ctx_k_buf is not None:
                    self._ctx_k_buf[slot, :, cur:end] = chunk_k.permute(1, 0, 2, 3)
            offset += slen
        self._write_ctx_len(ctx_len_updates)

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
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts

        raw_logits = logits
        logits_vocab_shard = False
        K = spec_metadata.runtime_draft_len
        if isinstance(logits, VocabShardedLogits):
            if K > 0 and self._target_tokens_from_shard(spec_metadata):
                # The target tokens come from the shard: this step returns
                # the shard, which the engine gathers only for a logits
                # post-processor.
                raw_logits = logits.local
                logits_vocab_shard = True
            else:
                # VocabShardedLogits.gather: the gathered [rows, vocab] float32 logits.
                logits = raw_logits = allgather_module(
                    logits.local, logits.head.lm_head.mapping, -1, None
                ).float()

        if K == 0:
            return self.skip_drafting(
                input_ids,
                position_ids,
                hidden_states,
                logits,
                attn_metadata,
                spec_metadata,
                draft_model,
            )

        draft_kv_cache_manager = self.get_draft_kv_cache_manager(resource_manager)
        if getattr(draft_kv_cache_manager, "has_standalone_draft_cache", False):
            if not self._ctx_buf_inited or self._ctx_kv_manager is not draft_kv_cache_manager:
                raise RuntimeError("Unified DFlash draft inputs must be prepared before forward")
        else:
            self._lazy_init_ctx_buffers(
                draft_model, spec_metadata, attn_metadata, draft_kv_cache_manager
            )
            spec_metadata._dflash_worker = self
            # Before any store: prefill and decode both address pages through it.
            # Returning False here means an empty batch -- the missing-offsets case
            # raises inside, with the metadata type in the message.
            self._refresh_ctx_block_tables(attn_metadata, batch_size, spec_metadata.request_ids)

        # Save context lengths so both warmup and a failed forward can roll
        # back the in-place _ctx_len updates made during drafting.
        is_warmup = spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing()
        if not torch.cuda.is_current_stream_capturing():
            # Never allocate the snapshot while capturing a CUDA graph: the
            # clone would live in the graph memory pool, and its replay-time
            # writes could alias blocks reused by later captures. Rollback is
            # only meaningful for eager/warmup forwards anyway; a failure
            # during capture aborts the graph itself, and captured ops do not
            # mutate _ctx_len until replay.
            self._saved_ctx_len = self._ctx_len.clone()
            if self._has_unified_draft_cache():
                self._saved_ctx_position_offset = self._ctx_position_offset.clone()
            self._saved_ctx_len_host = list(self._ctx_len_host)
            self._saved_req_ctx_pos = dict(self._req_ctx_pos)
            self._ctx_len_restore_pending = True

        self._execute_guided_decoder_if_present(logits)

        accepted_tokens, num_accepted_tokens = self.sample_and_accept_draft_tokens(
            logits, attn_metadata, spec_metadata
        )

        # Opt-in acceptance recording (env-gated; eager-mode measurement
        # runs only). Skipped for CUDA-graph batches (capture/replay/warmup
        # use synthetic requests and forbid the host sync).
        if (
            self._accept_stats is not None
            and num_gens > 0
            and not spec_metadata.is_cuda_graph
            and not torch.cuda.is_current_stream_capturing()
            and spec_metadata.request_ids is not None
        ):
            self._accept_stats.on_accept(
                spec_metadata.request_ids[num_contexts:batch_size],
                num_accepted_tokens[num_contexts:batch_size].tolist(),
            )

        # Update GDN/Mamba recurrent states to the accepted token's state.
        if num_gens > 0 and isinstance(attn_metadata.kv_cache_manager, MambaHybridCacheManager):
            attn_metadata.kv_cache_manager.update_mamba_states(
                attn_metadata=attn_metadata,
                num_accepted_tokens=num_accepted_tokens,
                state_indices=attn_metadata.mamba_metadata.state_indices,
            )

        self._prepare_attn_metadata_for_dflash(attn_metadata, spec_metadata)
        self._prepare_kv_for_draft_forward(
            attn_metadata, num_accepted_tokens, num_contexts, batch_size
        )

        # Collapse mrope [3, 1, N] to 1D by taking the first (temporal) dimension.
        # The draft model uses standard 1D RoPE, so only scalar positions are needed.
        if position_ids.ndim == 3:
            position_ids = position_ids[0, 0]
        else:
            position_ids = position_ids.squeeze(0)

        # Get total tokens processed by target model (for hidden state extraction)
        total_target_tokens = input_ids.shape[0]

        # Capture prefill (context) hidden states for future gen steps.
        # This gives the draft model the full prompt context, not just gen tokens.
        if num_contexts > 0:
            self._store_prefill_context(
                draft_model, spec_metadata, attn_metadata, position_ids, total_target_tokens
            )
            # Rebuild batch_to_slot after prefill assigns new slots
            if self._ctx_buf_inited and spec_metadata.request_ids:
                self._rebuild_batch_to_slot(spec_metadata.request_ids)

        drafts = not draft_model.context_kv_only
        if not drafts and num_gens > 0:
            self._check_no_real_generation(spec_metadata, num_contexts, batch_size)
        inputs = (
            self.prepare_1st_drafter_inputs(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                accepted_tokens=accepted_tokens,
                num_accepted_tokens=num_accepted_tokens,
                attn_metadata=attn_metadata,
                spec_metadata=spec_metadata,
                draft_model=draft_model,
                total_target_tokens=total_target_tokens,
            )
            if drafts
            else None
        )

        vocab_size = None
        gen_draft_tokens = None
        if num_gens > 0 and drafts:
            with self.draft_kv_cache_context(
                attn_metadata, None if self._has_unified_draft_cache() else draft_kv_cache_manager
            ):
                hidden_states_out = draft_model.dflash_forward(
                    noise_embedding=inputs["noise_embedding"],
                    query_positions=inputs["query_positions"],
                    num_ctx_per_req=inputs["num_ctx_per_req"],
                    ctx_k_cache=inputs["ctx_k_cache"],
                    ctx_v_cache=inputs["ctx_v_cache"],
                    ctx_cache_batch_idx=inputs["ctx_cache_batch_idx"],
                    ctx_kv_cache=inputs["ctx_kv_cache"],
                    ctx_page_table=inputs["ctx_page_table"],
                    block_placement=inputs["block_placement"],
                )

                # Gather K logits per gen request from the block outputs.
                # hidden_states_out is flat: [num_gens * block_size, hidden_dim].
                # Which block slots carry them is a drafter-family convention,
                # resolved through _draft_slot_ids.
                block_size = self._compute_block_size
                gen_gather_ids = self._block_gather_ids(
                    draft_model, num_gens, block_size, K, hidden_states_out.shape[0]
                )
                gen_hidden_states = hidden_states_out[gen_gather_ids]
                vocab_parallel = self._vocab_parallel
                local_tokens = (
                    self._local_draft_tokens(draft_model)
                    if vocab_parallel is None and not spec_metadata.wants_advanced_draft_sampling
                    else None
                )
                if (
                    vocab_parallel is not None
                    and vocab_parallel.draft_tokens is not None
                    and not spec_metadata.wants_advanced_draft_sampling
                ):
                    # Greedy draft tokens straight from this rank's vocab shard
                    # of the block logits, bit-identical to refining the
                    # gathered logits and taking the per-position argmax.
                    head = vocab_parallel.head
                    local_block = head.local_logits(gen_hidden_states)
                    gen_draft_tokens = vocab_parallel.draft_tokens(
                        draft_model, local_block.view(-1, K, local_block.shape[-1]), inputs
                    )
                    vocab_size = vocab_parallel.head.vocab_size
                elif local_tokens is not None:
                    # Greedy draft tokens from the full-vocab LM head output in its
                    # own dtype, bit-identical to the logits path below (the same
                    # GEMM, then the refinement and the per-position argmax).
                    local_block = draft_model.lm_head(gen_hidden_states)
                    vocab_size = local_block.shape[-1]
                    gen_draft_tokens = local_tokens(
                        draft_model, local_block.view(num_gens, K, vocab_size), inputs
                    )
                else:
                    gen_logits = draft_model.logits_processor(
                        gen_hidden_states, draft_model.lm_head, attn_metadata, True
                    )

                    vocab_size = gen_logits.shape[-1]
                    gen_logits = gen_logits.reshape(num_gens, K, vocab_size)

                    gen_logits = self._refine_block_logits(
                        draft_model, gen_logits, inputs, spec_metadata
                    )

                    # DFlash 2: replace the independent per-position picks with
                    # one coherent path through the block (absent for plain
                    # DFlash).
                    if getattr(draft_model, "lilicorr", None) is not None:
                        gen_logits = self._apply_lilicorr(
                            draft_model,
                            gen_logits,
                            gen_hidden_states.reshape(num_gens, K, -1),
                            inputs["lilicorr_anchor"],
                            spec_metadata,
                        )
                        vocab_size = gen_logits.shape[-1]
                    elif getattr(draft_model, "has_candidate_selector", False):
                        if dflash2_sampled_step(self, spec_metadata):
                            # DFlash2 sampling batch: the sampled selector walk
                            # (dflash2_sampling), verified next step by block verification.
                            gen_draft_tokens = _dflash2_sample_draft(
                                self,
                                draft_model,
                                gen_logits,
                                gen_hidden_states.reshape(num_gens, K, -1),
                                inputs["first_prev_tokens"],
                                spec_metadata,
                                num_contexts,
                                batch_size,
                            )
                        else:
                            gen_logits = self._apply_dflash2_selector(
                                draft_model,
                                gen_logits,
                                gen_hidden_states.reshape(num_gens, K, -1),
                                inputs["first_prev_tokens"],
                                spec_metadata,
                            )
                        vocab_size = gen_logits.shape[-1]

                    if gen_draft_tokens is None:
                        gen_draft_tokens = self.sample_draft_tokens(
                            gen_logits,
                            spec_metadata,
                            batch_size,
                            num_contexts=num_contexts,
                        )

                # Opt-in confidence-calibration recording (env-gated). The
                # confidence values come from an optional provider on the
                # recorder (None: no calibration rows collected; see
                # accept_stats.py). Guarded off for CUDA-graph batches and d2t
                # vocab-mapped drafters.
                if (
                    self._accept_stats is not None
                    and self._accept_stats.confidence_provider is not None
                    and not spec_metadata.is_cuda_graph
                    and not torch.cuda.is_current_stream_capturing()
                    and spec_metadata.request_ids is not None
                    and self._d2t is None
                ):
                    self._accept_stats.record_draft_confidence(
                        spec_metadata.request_ids[num_contexts:batch_size],
                        draft_model,
                        gen_hidden_states.reshape(num_gens, K, -1),
                        inputs["first_prev_tokens"],
                        gen_draft_tokens,
                    )

        else:
            # No generation rows: an empty block of draft tokens.
            gen_draft_tokens = torch.zeros((num_gens, K), dtype=torch.int32, device="cuda")

        # Context requests are not drafted by the block worker (zero placeholder
        # token); fill their draft-prob slot rows with a one-hot placeholder so
        # they are a legal distribution when they become gen requests next iter.
        self.write_context_onehot_draft_probs(spec_metadata, num_contexts, num_gens, K, vocab_size)

        if num_contexts > 0 and num_gens > 0:
            ctx_draft_tokens = torch.zeros((num_contexts, K), dtype=torch.int32, device="cuda")
            next_draft_tokens = torch.cat([ctx_draft_tokens, gen_draft_tokens], dim=0)
        elif num_contexts > 0:
            next_draft_tokens = torch.zeros((num_contexts, K), dtype=torch.int32, device="cuda")
        else:
            next_draft_tokens = gen_draft_tokens

        self._restore_attn_metadata_from_spec_dec(attn_metadata)
        self._apply_kv_rewind_after_draft(attn_metadata, spec_metadata)

        self._rollback_guided_decoder_after_verify(num_accepted_tokens)

        next_new_tokens = self._prepare_next_new_tokens(
            accepted_tokens,
            next_draft_tokens,
            spec_metadata.batch_indices_cuda,
            batch_size,
            num_accepted_tokens,
        )

        # Restore context lengths after warmup; real runs keep the updates.
        if is_warmup:
            self._ctx_len.copy_(self._saved_ctx_len)
            self._restore_ctx_len_host()
        self._ctx_len_restore_pending = False

        return {
            # The gathered float32 target logits, or this rank's vocab shard
            # when the target tokens were selected from it.
            "logits": raw_logits,
            "logits_vocab_shard": logits_vocab_shard,
            "new_tokens": accepted_tokens,
            "new_tokens_lens": num_accepted_tokens,
            "next_draft_tokens": next_draft_tokens,
            "next_new_tokens": next_new_tokens,
        }

    def _block_gather_ids(
        self, draft_model, num_gens: int, block_size: int, num_draft_tokens: int, num_rows: int
    ) -> torch.Tensor:
        """Block-output rows whose hidden states give the K draft logits."""
        ids = self._draft_slot_ids(draft_model, num_gens, block_size, num_draft_tokens)
        # Shields only the last request: at block_size == K with shift_label off, slots
        # run 1..K, so every request reads the next one's slot 0 and the last overruns.
        # Degrades, never raises.
        return ids.clamp(max=num_rows - 1)

    def _draft_first_slot(self, draft_model) -> int:
        """The block slot whose hidden state predicts the first draft token
        (``_draft_slot_ids`` takes slots ``first .. first + K - 1`` of each block)."""
        return 1

    def prepare_1st_drafter_inputs(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: DFlashSpecMetadata,
        draft_model: nn.Module,
        total_target_tokens: int = 0,
    ):
        """Prepare inputs for DFlash's draft forward.

        For gen requests, builds:
        - noise_embedding: token embeddings for [accepted + mask] tokens
        - query_positions: position IDs for the query tokens
        - num_ctx_per_req: per-request context length in the pool
        - ctx_k_cache / ctx_cache_batch_idx: slot-indexed views of the
          persistent per-layer latent pool (ctx_v_cache is None: no V).
        """
        num_contexts = attn_metadata.num_contexts
        batch_size = attn_metadata.num_seqs
        num_gens = batch_size - num_contexts

        # Resolve mask_token_id and block_size once, cache for subsequent calls
        if self._resolved_mask_token_id is None:
            if (
                hasattr(self.spec_config, "mask_token_id")
                and self.spec_config.mask_token_id is not None
            ):
                self._resolved_mask_token_id = self.spec_config.mask_token_id
            elif hasattr(draft_model, "mask_token_id"):
                self._resolved_mask_token_id = draft_model.mask_token_id
            elif hasattr(draft_model.model, "mask_token_id"):
                self._resolved_mask_token_id = draft_model.model.mask_token_id
            else:
                raise ValueError(
                    "DFlash requires mask_token_id to be set. Please set it in DFlashDecodingConfig "
                    "or ensure the draft model config has 'dflash_config.mask_token_id' or 'mask_token_id'."
                )
        mask_token_id = self._resolved_mask_token_id

        # Get the embed_tokens layer from the draft model
        embed_tokens = draft_model.draft_model_full.model.embed_tokens
        hidden_dim = (
            spec_metadata.hidden_size if spec_metadata.hidden_size > 0 else hidden_states.shape[-1]
        )
        lilicorr_anchor = None
        # The drafter's own trained mask row, if it kept one for the mask id in use.
        trained_mask_embedding = (
            getattr(draft_model, "mask_token_embedding", None)
            if mask_token_id == getattr(draft_model, "mask_token_id", None)
            else None
        )

        if (
            num_gens > 0
            and self._has_unified_draft_cache()
            and self._fused_gen_prepare_applies(spec_metadata)
        ):
            return self._prepare_paged_latent_gen_inputs(
                accepted_tokens,
                num_accepted_tokens,
                attn_metadata,
                spec_metadata,
                draft_model,
                total_target_tokens,
                mask_token_id,
                trained_mask_embedding,
            )

        if num_gens > 0:
            gen_num_accepted = num_accepted_tokens[num_contexts : num_contexts + num_gens]
            gen_accepted_tokens = accepted_tokens[num_contexts : num_contexts + num_gens, :]

            total_tokens_per_req = self._draft_tokens_per_req  # K+1
            K = spec_metadata.runtime_draft_len

            # Get captured multi-layer hidden states from spec_metadata
            captured_hs = spec_metadata.get_hidden_states(total_target_tokens)
            has_target_features = captured_hs is not None and hasattr(
                draft_model, "project_target_hidden"
            )

            # Use cached block_size (resolved once on first call)
            block_size = self._compute_block_size
            query_tokens_per_req = block_size

            # Get slots for gen requests from pre-computed mapping
            slots = self._batch_to_slot_buffer.tensor[num_contexts : num_contexts + num_gens]
            gen_rows_out = torch.arange(
                num_contexts, num_contexts + num_gens, dtype=torch.long, device="cuda"
            )

            K_plus_1 = K + 1

            bonus_idx = (gen_num_accepted - 1).clamp_min(0).long().unsqueeze(1)
            bonus = gen_accepted_tokens.gather(1, bonus_idx).squeeze(1).long()

            ctx_len_gen = self._ctx_len[slots]
            j_block = torch.arange(query_tokens_per_req, dtype=torch.long, device="cuda")
            offsets_kp1 = torch.arange(K_plus_1, dtype=torch.long, device="cuda")

            query_position = (ctx_len_gen + gen_num_accepted.long()).clamp(max=self._max_ctx)
            query_position_ids = query_position.unsqueeze(1) + j_block.unsqueeze(0)
            ctx_position_ids = ctx_len_gen.unsqueeze(1) + offsets_kp1.unsqueeze(0)

            noise_embed_2d = dflash_noise_block_embedding(
                embed_tokens, bonus, mask_token_id, query_tokens_per_req, trained_mask_embedding
            )

            # Accumulate new accepted features into context buffers
            if has_target_features:
                gen_start = attn_metadata.num_ctx_tokens
                # Target now processes exactly K+1 tokens per gen req, so the
                # captured slice is already the full set we need to project.
                gen_hs = captured_hs[gen_start : gen_start + num_gens * total_tokens_per_req]
                gen_hs_to_project = gen_hs.reshape(-1, gen_hs.shape[-1])
                projected_to_store = draft_model.project_target_hidden(gen_hs_to_project)
                if getattr(draft_model, "lilicorr", None) is not None:
                    # The last accepted output is predicted by the corresponding input
                    # row; the newly committed token itself has no target row yet.
                    projected_rows = projected_to_store.reshape(num_gens, K_plus_1, -1)
                    lilicorr_anchor = last_accepted_hidden(projected_rows, gen_num_accepted)
                gen_num_accepted_long = gen_num_accepted.long()
                col_idx = self._ctx_len[slots].unsqueeze(1) + offsets_kp1.unsqueeze(0)
                write_mask = offsets_kp1.unsqueeze(0) < gen_num_accepted_long.unsqueeze(1)
                owned = None
                shard = self._helix_shard()
                if shard is not None:
                    assert K + 1 <= shard.page_size, "helix alias slots must lie outside the write"
                    col_idx, owned = shard.local_slots(col_idx)
                    owned = owned.reshape(-1)
                if self._ctx_block_tables is not None:
                    # Bound by what the manager handed THIS request: the masked
                    # entries below are still written (fixed-size, for graph
                    # safety) and a placeholder page resolves to another's.
                    gen_capacity = self._ctx_block_counts[gen_rows_out] * self._ctx_page_size
                    col_idx = torch.minimum(col_idx, (gen_capacity - 1).unsqueeze(1)).clamp_(min=0)
                if self._ctx_block_tables is None or self._ctx_k_buf is not None:
                    col_idx = col_idx.clamp(max=self._max_ctx - 1)

                # Fixed-size writes for CUDA graph compatibility:
                # Write ALL entries but zero out invalid ones. Invalid
                # writes land at clamped column indices beyond valid
                # _ctx_len range, so they're harmless.
                slot_flat = slots.unsqueeze(1).expand(-1, K + 1).reshape(-1)
                col_flat = col_idx.reshape(-1)
                proj_flat = projected_to_store  # already [num_gens*(K+1), proj_dim]
                pos_flat = ctx_position_ids.long().reshape(-1)
                mask_1d = write_mask.reshape(-1)

                # Fast path: store the pre-projected/pre-RoPE'd K/V.
                # dflash_forward reads these directly via cache_batch_idx.
                cache_dtype = (
                    self._ctx_kv_buf[0].dtype if self._ctx_paged else self._ctx_k_buf.dtype
                )
                if cache_dtype == torch.float8_e4m3fn:
                    # Project in the drafter's weight dtype; the store converts.
                    cache_dtype = proj_flat.dtype
                k_new, _ = draft_model.precompute_context_kv(proj_flat.to(cache_dtype), pos_flat)
                k_new.mul_(mask_1d.view(-1, 1, 1, 1).to(k_new.dtype))
                slot_long = slot_flat.long()
                col_long = col_flat.long()
                if self._ctx_paged:
                    if self._ctx_block_tables is not None:
                        # Batch positions of the gen requests, matching the
                        # per-request block table's row order.
                        rows_long = gen_rows_out.unsqueeze(1).expand(-1, K + 1).reshape(-1)
                    else:
                        rows_long = slot_long
                    self._store_context_kv_paged(k_new, None, rows_long, col_long, owned)
                if self._ctx_k_buf is not None:
                    self._ctx_k_buf[slot_long, :, col_long] = k_new

                self._advance_ctx_len(slots, gen_num_accepted_long)

            num_ctx_per_req_t = self._ctx_len[slots]
            if self._ctx_block_tables is not None:
                # The write above clamps columns to the request's allocation, so
                # a context that outruns it has its tail written over the last
                # valid slot. Truncate what is advertised to match, or the read
                # attends that clobbered value -- or another request's page.
                shard = self._helix_shard()
                if shard is None:
                    allocated = dflash_allocated_ctx_limit(
                        self._ctx_block_counts, self._ctx_page_size, block_size
                    )
                else:
                    # Local page k holds global page k * cp_size + cp_rank and is
                    # the alias of every other page in that span, so k local pages
                    # cover k * cp_size global pages.
                    allocated = (
                        self._ctx_block_counts * shard.cp_size * self._ctx_page_size - block_size
                    ).clamp(min=0)
                num_ctx_per_req_t = torch.minimum(num_ctx_per_req_t, allocated[gen_rows_out])
            noise_embedding = noise_embed_2d
            query_positions = query_position_ids.long()

            # Update seq_lens for gen requests to K+1 (the number of tokens
            # the target forward actually processed).
            attn_metadata._seq_lens_cuda[num_contexts : num_contexts + num_gens] = (
                total_tokens_per_req
            )
            attn_metadata._seq_lens[num_contexts : num_contexts + num_gens] = total_tokens_per_req
        else:
            noise_embedding = hidden_states.new_empty(0, 0, hidden_dim)
            query_positions = torch.empty(0, 0, dtype=torch.long, device="cuda")
            num_ctx_per_req_t = torch.empty(0, dtype=torch.long, device="cuda")
            slots = torch.empty(0, dtype=torch.long, device="cuda")
            gen_rows_out = slots
            bonus = torch.empty(0, dtype=torch.long, device="cuda")

        return {
            "lilicorr_anchor": lilicorr_anchor,
            "noise_embedding": noise_embedding,
            "query_positions": query_positions,
            "num_ctx_per_req": num_ctx_per_req_t,
            "ctx_k_cache": self._ctx_k_buf,
            "ctx_v_cache": None,
            # Slots index the private arena; the manager's block table is keyed
            # by batch position, so the drafter reads its own rows there.
            "ctx_cache_batch_idx": (
                gen_rows_out
                if self._ctx_block_tables is not None and self._ctx_k_buf is None
                else slots
            ),
            # Anchor token per gen request (block slot 0): last accepted
            # token. The DFlash2 candidate selector conditions its first step on it.
            "first_prev_tokens": bonus,
            "ctx_kv_cache": self._ctx_kv_buf,
            "ctx_page_table": (
                self._ctx_block_tables
                if self._ctx_block_tables is not None
                else self._ctx_page_table
            ),
            "block_placement": None,
        }


# ====================================================================================================================
# DFlash2 installation on the causal LM
# ====================================================================================================================


def _replace_in_epilogue(target: nn.Module, old: Optional[nn.Module], new: nn.Module) -> None:
    """Put ``new`` where the engine's shell appended ``old`` (by identity), else append it."""
    for i, module in enumerate(target.epilogue):
        if module is old:
            target.epilogue[i] = new
            return
    target.epilogue.append(new)


def uses_k31_dflash2(spec_config) -> bool:
    """Whether ``spec_config`` is one-engine DFlash (the DFlash2 MLA drafter's mode)."""
    return (
        spec_config is not None
        and spec_config.spec_dec_mode.use_one_engine()
        and spec_config.spec_dec_mode.is_dflash()
    )


def install_k31_dflash2(target: nn.Module, model_config: ModelConfig) -> bool:
    """Attach the flat MLA drafter and its worker to the causal-LM epilogue.

    The drafted block is ``max_draft_len + 1`` wide independently of the
    checkpoint's training block size.
    """
    spec_config = getattr(model_config, "spec_config", None)
    if not uses_k31_dflash2(spec_config):
        return False
    draft_config = target.draft_config

    if draft_config is None or dflash2_attention_mode(draft_config.pretrained_config) != "mla":
        raise ValueError(
            "Kimi K3.1 runs DFlash with the DFlash2 MLA drafter only "
            "(dflash_config.attention_mode 'mla')."
        )

    draft_model = MLADFlash2ForCausalLM(
        draft_config,
        block_size=spec_config.max_draft_len + 1,
        dflash_attention_backend=spec_config.attention_backend,
        context_kv_dtype=spec_config.context_kv_dtype,
    )
    draft_model.logits_processor = target.logits_processor
    spec_worker = DFlashWorker(
        spec_config,
        model_config.mapping.mapping_for_full_rows(),
        target.use_separate_draft_kv_cache,
    )
    _replace_in_epilogue(target, target.draft_model, draft_model)
    _replace_in_epilogue(target, target.spec_worker, spec_worker)
    target.draft_model = draft_model
    target.spec_worker = spec_worker
    spec_worker.set_draft_model(draft_model)
    spec_worker.set_target_model(target)
    logger.info(
        f"K3.1 DFlash2: package drafter {type(draft_model).__name__} (block "
        f"{draft_model.block_size}) and worker {type(spec_worker).__name__}"
    )
    return True


# ====================================================================================================================
# Vision-language wrapper (engine vision tower)
# ====================================================================================================================


def _moonvit_payload_keys(modality: str, mod_data: Dict[str, Any]) -> Tuple[str, str]:
    """Return the ``(pixel_values, grid_thw)`` keys MoonViT reads from one modality payload."""
    if modality == "video":
        return (
            "pixel_values_videos" if "pixel_values_videos" in mod_data else "pixel_values",
            "video_grid_thw" if "video_grid_thw" in mod_data else "image_grid_thw",
        )
    return "pixel_values", "image_grid_thw"


# ---------------------------------------------------------------------------
# Processed media layout and processor geometry
# ---------------------------------------------------------------------------

# Key in ``multimodal_data["image"]`` that lists the chunk count of every video, in order.
_VIDEO_CHUNK_COUNTS_KEY = "video_chunk_counts"


def _moonvit_item_rows(image_data: Dict[str, Any]) -> Dict[str, List[range]]:
    """Map every processed image and video to its rows of ``image_grid_thw``.

    The input processor packs all media of a request into ``multimodal_data["image"]``: one
    grid row per image, followed by the chunk rows of each video in order.

    Returns:
        ``{"image": [...]}``, plus ``"video"`` when the payload holds videos. Entry ``i`` of a
        modality is the row range of that modality's item ``i``.
    """
    num_rows = len(image_data["image_grid_thw"])
    chunk_counts = [int(count) for count in image_data.get(_VIDEO_CHUNK_COUNTS_KEY, ())]
    num_images = num_rows - sum(chunk_counts)
    if num_images < 0 or any(count <= 0 for count in chunk_counts):
        raise ValueError(
            f"A Kimi media payload with {num_rows} grid rows cannot hold video chunk counts "
            f"{chunk_counts}"
        )
    item_rows = {"image": [range(row, row + 1) for row in range(num_images)]}
    if chunk_counts:
        starts = accumulate(chunk_counts, initial=num_images)
        item_rows["video"] = [
            range(start, start + count) for start, count in zip(starts, chunk_counts)
        ]
    return item_rows


# ---------------------------------------------------------------------------
# Vision tower
# ---------------------------------------------------------------------------


class KimiK3MoonViT3dEncoder(_engine_k3.K3MoonViT3dEncoder):
    """The engine's K3 MoonViT3d encoder with an engine-fixed attention ``max_seq_len``.

    Adds no state beyond the class default of ``_attn_max_seq_len``, so ``KimiK3VisionModel`` turns
    the engine-built encoder into this class in place.
    """

    # Per-sequence attention capacity fixed by ``set_attn_max_seq_len``; 0 until the engine
    # sets it, in which case each batch uses its own longest sequence.
    _attn_max_seq_len: int = 0

    def set_attn_max_seq_len(self, max_seq_len: int) -> None:
        """Fix the attention ``max_seq_len`` to the longest sequence the engine can schedule.

        The native attention op is cached per layer by ``max_seq_len`` (which is also its
        attention window). A per-batch value builds, and keeps, a new op with its own device
        buffers in every layer for every new image size.
        """
        if max_seq_len <= 0:
            raise ValueError(f"MoonViT attention max_seq_len must be positive, got {max_seq_len}")
        self._attn_max_seq_len = max_seq_len

    def prepare_attn_metadata(
        self,
        seq_lens: List[int],
        attn_metadata: AttentionMetadata,
    ) -> AttentionMetadata:
        batch_size = len(seq_lens)
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int, pin_memory=prefer_pinned())
        attn_metadata.num_contexts = batch_size
        attn_metadata.request_ids = list(range(1, batch_size + 1))
        attn_metadata.prompt_lens = seq_lens
        attn_metadata.seq_lens = seq_lens_tensor
        attn_metadata.max_seq_len = max(self._attn_max_seq_len, max(seq_lens))
        attn_metadata.prepare()
        return attn_metadata


class KimiK3VisionModel(_engine_k3.KimiK3VisionModel):
    """The engine's Kimi K3 MoonViT3d tower and ``PatchMergerMLPV2`` projector, built with dense attention.

    Weight loading, feature extraction and the forward are the engine's; this class adds the
    request-time media validation of ``validate_multimodal_data``.
    """

    def __init__(self, model_config: ModelConfig) -> None:
        # The tower runs dense full-mask attention. The text model's sparse
        # attention (the Kimi K3.1 DSA indexer) would otherwise swap in a
        # sparse backend for every vision attention layer.
        dense_config = copy.copy(model_config)
        dense_config._frozen = False
        dense_config.sparse_attention_config = None
        dense_config._frozen = model_config._frozen
        super().__init__(dense_config)
        self.encoder.__class__ = KimiK3MoonViT3dEncoder

    def validate_multimodal_data(self, multimodal_data: Dict[str, Any]) -> None:
        """Reject processed media that this tower cannot encode.

        Every grid row ``(t, h, w)`` must fit the temporal position table, the 2D RoPE table, and
        the patch merger, and the pixel payload must hold exactly the patches of its grids.

        Raises:
            ValueError: If a grid or a pixel payload is outside these limits.
        """
        kernel_h, kernel_w = self.merge_kernel_size
        max_frames = self.patch_embed.pos_emb.num_frames
        rope = self.encoder.rope_2d
        proj = self.patch_embed.proj
        patch_shape = (proj.in_channels, *proj.kernel_size)
        for modality in ("image", "video"):
            mod_data = multimodal_data.get(modality)
            if not isinstance(mod_data, dict):
                continue
            pv_key, thw_key = _moonvit_payload_keys(modality, mod_data)
            if mod_data.get(thw_key) is None:
                continue
            grid_thws = torch.as_tensor(mod_data[thw_key])
            if grid_thws.ndim != 2 or grid_thws.shape[-1] != 3:
                raise ValueError(
                    f"{thw_key} must have shape [num_grids, 3], got {tuple(grid_thws.shape)}"
                )
            num_patches = 0
            for t, h, w in grid_thws.tolist():
                if (
                    not (
                        1 <= t <= max_frames
                        and 1 <= h <= rope.max_height
                        and 1 <= w <= rope.max_width
                    )
                    or h % kernel_h
                    or w % kernel_w
                ):
                    raise ValueError(
                        f"MoonViT cannot encode the {modality} grid (t={t}, h={h}, w={w}); it "
                        f"supports t <= {max_frames}, h <= {rope.max_height}, "
                        f"w <= {rope.max_width}, and h and w divisible by the merge kernel "
                        f"{self.merge_kernel_size}"
                    )
                num_patches += t * h * w
            pixel_values = mod_data.get(pv_key)
            if isinstance(pixel_values, torch.Tensor) and tuple(pixel_values.shape) != (
                num_patches,
                *patch_shape,
            ):
                raise ValueError(
                    f"{pv_key} must have shape {(num_patches, *patch_shape)} for its grids, "
                    f"got {tuple(pixel_values.shape)}"
                )


# ====================================================================================================================
# Target
# ====================================================================================================================

#: Stock code this target runs outside the catalog, declared so it is not consumed silently. The whole target
#: directory is covered, by area; an entry leaves the list when a catalog entry replaces it.
UNCERTIFIED_GENERIC_CALLS = (
    # The causal LM shell, the VLM wrapper base and the multimodal plumbing around the text model.
    "tensorrt_llm._torch.models.modeling_speculative.SpecDecOneEngineForCausalLM",
    "tensorrt_llm._torch.models.modeling_utils.DecoderModel",
    "tensorrt_llm._torch.models.modeling_multimodal_utils.fuse_input_embeds",
    "tensorrt_llm._torch.models.modeling_multimodal_utils.get_multimodal_embeddings",
    # Shared stream scheduling and distributed collectives.
    "tensorrt_llm._torch.modules.multi_stream_utils.maybe_execute_in_parallel",
    "tensorrt_llm._torch.distributed.AllReduce",
    # The latent MoE: the ConfigurableMoE stack on the TRTLLM-Gen NVFP4 runner and DeepSeek-V3 routing (the router GEMM,
    # the routing tail and the latent projections are family ops).
    "tensorrt_llm._torch.moe.fused_moe.create_moe",
    "tensorrt_llm._torch.moe.fused_moe.ConfigurableMoE",
    "tensorrt_llm._torch.moe.fused_moe.TRTLLMGenFusedMoE",
    "tensorrt_llm._torch.moe.fused_moe.routing.DeepSeekV3MoeRoutingMethod",
    # KDA kernel registration and the shared KV-cache allocation/transfer interface.
    "torch.ops.trtllm.kda_mtp_decode",
    "tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2.MambaHybridCacheManagerV2",
    # The target-owned attention implementations use these shared kernel/metadata interfaces.
    "tensorrt_llm._torch.attention.backends.trtllm.TrtllmAttention",
    "tensorrt_llm._torch.attention.backends.sparse.dsa.backend.DSATrtllmAttention",
    "tensorrt_llm._torch.attention.backends.sparse.dsa.metadata.DSAtrtllmAttentionMetadata",
    "torch.ops.trtllm.attention",
    # Speculative execution interfaces, metadata and vocabulary-parallel logits.
    "tensorrt_llm._torch.speculative.dflash.DFlashSpecMetadata",
    "tensorrt_llm._torch.speculative.interface.SpecWorkerBase",
    "tensorrt_llm._torch.speculative.vocab_parallel.VocabParallelHead",
    # The vision tower and the VLM wrapper base: the engine's (subclassed).
    "tensorrt_llm._torch.models.modeling_kimi_k3_vl.KimiK3VisionModel",
    "tensorrt_llm._torch.models.modeling_kimi_k3_vl.K3MoonViT3dEncoder",
    "tensorrt_llm._torch.models.modeling_kimi_k25.KimiK25ForConditionalGeneration",
)


# ---------------------------------------------------------------------------
# Full VLM Model
# ---------------------------------------------------------------------------


class KimiK3ForConditionalGeneration(
    MultimodalModelMixin, _engine_k25.KimiK25ForConditionalGeneration
):
    """Kimi K3 vision-language model: MoonViT3d + KimiLinear text backbone.

    Derives from the engine's K2.5 wrapper for its spec-dec / weight-loading property forwarding
    (``draft_config``, ``draft_model``, ``model``, ``lm_head``, ``load_draft_weights``,
    ``multimodal_data_device_paths``, ``mm_token_ids``) and KV-manager preference, and defines what
    it runs differently: ``__init__`` (vision encoder + text backbone classes, no local tower under
    ``disable_mm_encoder`` or encoder disaggregation), the multimodal item hooks of
    ``MultimodalModelMixin`` (one encoder item per image or video), the forward (which keeps the
    prompt ids for the speculative drafter) and the weight loader of ``weights.py``. The wrapper's
    own ``vocab_size_padded`` / ``set_guided_decoder`` / ``infer_max_seq_len`` are restated here
    because the mixin, listed first, defines them too.
    """

    _VISION_MODEL_CLS = KimiK3VisionModel
    mamba_metadata_cls = KimiLinearForCausalLM.mamba_metadata_cls
    # The family input processor uses full-request image encoding.
    supports_mm_encoder_item_scheduling = False

    @classmethod
    def requires_kv_cache_manager_v2(cls, pretrained_config=None) -> bool:
        """Follow the text backbone: Kimi K3.1 DSA requires KV cache manager V2."""
        return KimiLinearForCausalLM.requires_kv_cache_manager_v2(pretrained_config)

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        *args,
        **kwargs,
    ) -> None:
        config = model_config.pretrained_config
        # PreTrainedModel.__init__ resolves config._attn_implementation and
        # rejects the default "sdpa" unless the class declares support. This
        # wrapper never runs HF attention itself (TRT-LLM attention is wired
        # inside the sub-modules), so declare support to keep HF init happy.
        self._supports_sdpa = True
        # Skip the K2.5 parent __init__ (it wires DeepSeek-V3 + K2.5 vision);
        # initialize the HF PreTrainedModel machinery directly, then build the
        # K3 components below.
        PreTrainedModel.__init__(self, config)

        # Re-entry guard (mirrors the K2.5 wrapper): the tail of this __init__
        # repoints model_config.pretrained_config at the text_config and remaps
        # the quant exclude list, so running the body twice on the same
        # instance would fail the text_config assert below. If the text
        # backbone already exists this is a re-init on a built instance: no-op.
        if hasattr(self, "llm"):
            return

        self.model_config = model_config
        self._vlm_pretrained_config = config

        # --- Vision encoder (deferred under MetaInitMode, recreated in
        # load_weights) ---
        self.mm_encoder = None
        # Text-only serving (disable_mm_encoder) and E/P disaggregation both
        # run without a local vision tower.
        self._skip_mm_encoder = DISAGG or model_config.disable_mm_encoder
        if not self._skip_mm_encoder:
            try:
                mm_encoder = self._VISION_MODEL_CLS(model_config)
                if _has_meta_tensors(mm_encoder):
                    logger.info("Vision encoder deferred to load_weights() (MetaInitMode active)")
                else:
                    self.mm_encoder = mm_encoder
            except MetaInitException:
                logger.info("Vision encoder deferred to load_weights() (MetaInitMode active)")

        text_model_config = copy.copy(model_config)
        assert hasattr(config, "text_config"), "Kimi K3 config must have text_config"
        text_model_config._frozen = False
        text_model_config.pretrained_config = config.text_config

        # Remap quant exclude_modules: language_model.X -> model.X
        if text_model_config.quant_config.exclude_modules:
            text_model_config.quant_config = copy.copy(text_model_config.quant_config)
            p = self._LANG_PREFIX
            mapped = []
            for m in text_model_config.quant_config.exclude_modules:
                if m.startswith(p):
                    rest = m[len(p) :]
                    if rest.startswith("layers."):
                        rest = "model." + rest
                    mapped.append(rest)
                else:
                    mapped.append(m)
            text_model_config.quant_config.exclude_modules = mapped

        if not text_model_config.skip_create_weights_in_init:
            text_model_config.skip_create_weights_in_init = True
        text_model_config._frozen = True

        self.llm = KimiLinearForCausalLM(text_model_config)

        self._media_placeholder_token_id = getattr(
            config, "media_placeholder_token_id", _MEDIA_PLACEHOLDER_TOKEN_ID
        )
        self._mm_token_ids = torch.tensor([self._media_placeholder_token_id], dtype=torch.int32)

        # Point model_config at the text_config so the executor reads generation
        # params (eos_token_id, ...) from the text backbone, matching K2.5.
        self.config = self.llm.config
        model_config._frozen = False
        model_config.pretrained_config = self.llm.config
        model_config._frozen = True

    @property
    def vocab_size_padded(self) -> int:
        return self.llm.vocab_size_padded

    def set_guided_decoder(self, *args, **kwargs):
        return self.llm.set_guided_decoder(*args, **kwargs)

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    @property
    def multimodal_token_ids(self) -> torch.Tensor:
        return self._mm_token_ids

    @property
    def language_model(self) -> nn.Module:
        return self.llm

    @property
    def text_embedding_layer(self) -> nn.Module:
        return self.llm.model.embed_tokens

    @property
    def embedding_dim(self) -> int:
        return self.llm.config.hidden_size

    @property
    def embedding_dtype(self) -> torch.dtype:
        return self.text_embedding_layer.weight.dtype

    def encode_multimodal_inputs(
        self,
        multimodal_params: Sequence[MultimodalParams],
    ) -> torch.Tensor:
        """Run MoonViT; rows follow param order, then grid-row order within each param."""
        if self.mm_encoder is None:
            raise ValueError("Raw multimodal inputs require a local multimodal encoder.")
        embeddings = self.mm_encoder.forward(list(multimodal_params))
        if not embeddings:
            raise MultimodalEncoderContractError("The multimodal params carry no MoonViT media")
        return embeddings[0]

    def build_multimodal_encoder_input(
        self,
        param: MultimodalParams,
        local_item_indices: Sequence[int],
        modality: Optional[str] = None,
    ) -> MultimodalParams:
        """Select images or videos from the packed MoonViT payload.

        Images and video chunks share ``multimodal_data["image"]`` (see
        ``_moonvit_item_rows``), so the items of either modality are grid-row ranges of that
        payload. The view keeps the selected rows in the requested item order.
        """
        image_data = param.multimodal_data.get("image")
        if not isinstance(image_data, dict) or image_data.get("image_grid_thw") is None:
            raise MultimodalEncoderContractError(
                "Kimi media items must be packed under multimodal_data['image']"
            )
        item_rows = _moonvit_item_rows(image_data)
        if modality is None:
            present = [name for name, rows in item_rows.items() if rows]
            if len(present) != 1:
                raise NotImplementedError(
                    "Selecting Kimi media items from a mixed image/video payload needs the modality"
                )
            modality = present[0]
        if modality not in item_rows:
            raise MultimodalEncoderContractError(
                f"The Kimi media payload holds no {modality} items"
            )
        rows = [row for index in local_item_indices for row in item_rows[modality][index]]
        row_patches = tuple(
            t * h * w for t, h, w in torch.as_tensor(image_data["image_grid_thw"]).tolist()
        )
        selected = select_multimodal_fields(
            image_data,
            {
                "pixel_values": MultimodalItemPartition(item_lengths=row_patches),
                "image_grid_thw": MultimodalItemPartition(),
            },
            rows,
        )
        selected.pop(_VIDEO_CHUNK_COUNTS_KEY, None)
        if modality == "video":
            selected[_VIDEO_CHUNK_COUNTS_KEY] = [
                len(item_rows["video"][index]) for index in local_item_indices
            ]
        residual_data = {
            key: value
            for key, value in param.multimodal_data.items()
            if key not in MEDIA_MODALITIES
        }
        residual_data["image"] = selected
        return MultimodalParams(
            multimodal_data=residual_data,
            multimodal_input=(
                copy.copy(param.multimodal_input) if param.multimodal_input is not None else None
            ),
        )

    def validate_multimodal_request_data(self, mm_data: dict) -> None:
        """Reject media that MoonViT cannot encode before the request is scheduled."""
        if self.mm_encoder is not None:
            self.mm_encoder.validate_multimodal_data(mm_data)

    def load_weights(self, weights) -> None:
        """The explicit key split of ``weights.py``: text model, vision tower, unknown keys fail."""
        target_weights.load(self, weights)

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: Optional[bool] = False,
        multimodal_params: Optional[List[MultimodalParams]] = None,
        **kwargs,
    ) -> torch.Tensor:
        multimodal_params = multimodal_params or []
        mm_embeds: List[torch.Tensor] = []
        mm_token_ids = None

        if len(multimodal_params) > 0:
            if self._skip_mm_encoder:
                raise NotImplementedError(
                    "Multimodal input is not supported: the vision encoder is disabled on this server "
                    "(disable_mm_encoder or TLLM_MULTIMODAL_DISAGGREGATED)."
                )
            mm_ctx_params = multimodal_params[: attn_metadata.num_contexts]
            mm_embeds = get_multimodal_embeddings(
                encoder_forward_fn=self.mm_encoder.forward,
                multimodal_params=mm_ctx_params,
            )
            mm_embeds = find_input_mm_embeds(mm_embeds, mm_ctx_params)

            # ``runners.prepare_multimodal_indices`` now sees Kimi's
            # in-vocab placeholder via ``self.mm_token_ids`` and emits indices
            # that match ``find_input_mm_embeds``'s active-chunk slice. The
            # previous ``(input_ids == placeholder).sum().item()`` guard was a
            # host sync used to detect chunked-prefill / KV-reuse mismatches;
            # that mismatch surfaces as a row-count error inside
            # ``fuse_input_embeds`` itself, so the explicit check is no longer
            # needed.
            if len(mm_embeds) > 0:
                mm_token_ids = self.mm_token_ids.to(input_ids.device, dtype=input_ids.dtype)
        raw_input_ids = input_ids
        input_ids, inputs_embeds = fuse_input_embeds(
            self.llm.model.embed_tokens,
            input_ids,
            mm_embeds,
            mm_token_ids=mm_token_ids,
            mm_token_indices=kwargs.get("mm_token_indices"),
            text_token_indices=kwargs.get("text_token_indices"),
        )

        # With fused image embeddings input_ids is None; the speculative
        # drafter reads the prompt token IDs from orig_input_ids instead.
        return self.llm.forward(
            attn_metadata,
            input_ids,
            position_ids,
            inputs_embeds,
            return_context_logits,
            orig_input_ids=raw_input_ids,
            **kwargs,
        )


class KimiK31ForConditionalGeneration(KimiK3ForConditionalGeneration):
    """Kimi K3.1 vision-language model: the K3 wrapper over the package's text backbone, which
    reads the K3.1 ``dsa_config`` (a DSA indexer on every MLA layer) from its own config.

    Besides the model it states what the engine asks a model class for before and while it is
    built: model defaults, KV cache manager / transceiver preferences, the hybrid KDA + MLA + DSA
    cache manager (``select_kv_cache_manager_cls``), the checkpoint read
    (``open_checkpoint_weights``) and the DSA attention backend (``attention_backend_for``).
    """

    @classmethod
    def get_model_defaults(cls, llm_args) -> dict:
        return KimiLinearForCausalLM.get_model_defaults(llm_args)

    @classmethod
    def get_preferred_transceiver_runtime(cls, pretrained_config=None):
        return KimiLinearForCausalLM.get_preferred_transceiver_runtime(pretrained_config)

    @classmethod
    def select_kv_cache_manager_cls(
        cls, model_config, kv_cache_config, is_disagg=False, cache_transceiver_config=None
    ):
        """The KDA + MLA hybrid manager with one DSA indexer K-cache buffer per MLA layer."""
        from .ops.attention.cache_manager import select_k31_kv_cache_manager_cls

        return select_k31_kv_cache_manager_cls(
            model_config, kv_cache_config, is_disagg, cache_transceiver_config
        )

    @staticmethod
    def open_checkpoint_weights(checkpoint_dir: str, mapping=None):
        """The checkpoint read for the load: lazy safetensors slices (``weights.open_checkpoint_weights``)."""
        return target_weights.open_checkpoint_weights(checkpoint_dir)

    def attention_backend_for(self, backend_name: str, sparse_params):
        """The package's DSA backend (and its metadata) for the engine's DSA parameters."""
        from .ops.attention.backend import attention_backend_for

        return attention_backend_for(backend_name, sparse_params)


class ModelingV2KimiK31ProNvfp4Sm10xDefault(KimiK31ForConditionalGeneration):
    """Kimi K3.1 Pro NVFP4/MXFP8, SM100/SM103/SM107, layout selected by Mapping."""

    handles_nvfp4_kv_cache = True

    def __init__(self, model_config, *args, **kwargs) -> None:
        super().__init__(model_config, *args, **kwargs)
        if not hasattr(torch.ops.trtllm, "mxfp8_mxfp8_gemm"):
            raise RuntimeError("The K3.1 NVFP4 target requires native MXFP8 GEMM")

    def load_weights(self, weights) -> None:
        """The explicit key split of ``weights.py``: text model, vision tower, unknown keys fail."""
        target_weights.load(self, weights)

    def prepare_disagg_generation_request(self, request: Any) -> None:
        """Discard cached DFlash2 proposals for the newly received request's slot."""
        invalidate_dflash2_proposal_slot(
            getattr(self.llm, "spec_worker", None), request.py_seq_slot
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: Optional[bool] = False,
        multimodal_params: Optional[List[MultimodalParams]] = None,
        **kwargs,
    ) -> torch.Tensor:
        if attn_metadata.num_contexts > 0:
            output = KimiK31ForConditionalGeneration.forward(
                self,
                attn_metadata,
                input_ids,
                position_ids,
                inputs_embeds,
                return_context_logits,
                multimodal_params,
                **kwargs,
            )
        else:
            output = self.llm.forward(
                attn_metadata,
                input_ids,
                position_ids,
                inputs_embeds,
                return_context_logits,
                orig_input_ids=input_ids,
                **kwargs,
            )
        if not torch.compiler.is_compiling():
            tuner = AutoTuner.get()
            if tuner.is_tuning_mode:
                MXFP8GemmRunner.sync_all_tactic_caches(tuner)
        return output
