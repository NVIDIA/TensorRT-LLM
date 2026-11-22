# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: Kimi K3.1 Pro NVFP4 (ModelOpt) / sm_107 / dep16 -- attention DP 16, MoE EP16.

One recipe and one layout, so the forward carries no recipe or layout branch:

* checkpoint: the ModelOpt NVFP4 export as stored. The routed experts are NVFP4 (per-tensor
  input scale 1.0); the shared experts and the KDA projections are MXFP8 (one UE8M0 scale per 32
  values of a row); MLA, the latent MoE projections, the router, norms, embedding and LM head are
  BF16. Nothing is quantized at load time.
* caches: FP8 (E4M3, scale 1) MLA latent KV on all 24 DSA layers, FP8 index keys, FP32 KDA
  recurrent state.
* speculative decoding: DFlash2, with an FP8 drafter context history.
* layout: attention DP over 16 ranks (each rank runs whole KDA and MLA layers on its own
  requests); the routed experts EP16, 56 per rank. Context and generation servers.
* images: the checkpoint's MoonViT tower and projector (BF16, whole on every rank) encode a
  context step's images and their rows replace the media placeholders' embeddings (the vision
  section). A server with ``disable_mm_encoder`` (a disaggregated generation server, which never
  encodes) builds no tower and runs text only.

Construction binds layout, role and recipe (module constants, closures, helper arguments); the
forward dispatches only on per-step facts (row counts, whether a step has context rows). Every
model kernel call goes through an OpWrapper, from modeling_v2's ``catalog/`` or the family's
``ops/``; small torch glue (argmax, where, cat, gather, masked_fill, sum, repeat_interleave) runs
inline.
"""

from __future__ import annotations

import copy
import math
import time
import weakref
from types import MappingProxyType
from typing import Any, Callable, NamedTuple, Optional

import numpy as np
import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch.attention.backends.fmha.phased import get_spec_decoding_position_offsets
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modeling_v2.catalog.activation.flashinfer_silu_and_mul import (
    flashinfer_silu_and_mul,
)
from tensorrt_llm._torch.modeling_v2.catalog.attention.mla_rope_generation import (
    mla_rope_generation,
)
from tensorrt_llm._torch.modeling_v2.catalog.attention.thop_attention import thop_attention
from tensorrt_llm._torch.modeling_v2.catalog.gemm.bmm_out import bmm_out
from tensorrt_llm._torch.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch.modeling_v2.catalog.moe.noaux_tc_op import noaux_tc_op
from tensorrt_llm._torch.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (
    flashinfer_fused_add_rmsnorm,
)
from tensorrt_llm._torch.modeling_v2.catalog.norm.flashinfer_rmsnorm import flashinfer_rmsnorm
from tensorrt_llm._torch.modeling_v2.catalog.quantization.fp4_quantize import fp4_quantize
from tensorrt_llm._torch.models.modeling_multimodal_utils import (
    find_input_mm_embeds,
    get_multimodal_embeddings,
)
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.pyexecutor.sampler.ops.flashinfer import resolve_advanced_sampling_filters
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import GREEDY_TEMPERATURE_THRESHOLD
from tensorrt_llm._torch.speculative import should_use_separate_draft_kv_cache
from tensorrt_llm._torch.speculative.dflash import DFlashWorker as _EngineDFlashWorker
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase
from tensorrt_llm._torch.speculative.vocab_parallel import VocabShardedLogits
from tensorrt_llm.functional import RopeEmbeddingUtils
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

from ..input_processor import MEDIA_PLACEHOLDER_TOKEN_ID, moonvit_max_image_grid
from ..ops.activation.sigmoid_gate_mul import k31_sigmoid_gate_mul
from ..ops.activation.situ_and_mul import situ_and_mul
from ..ops.attention.convert_req_index_to_global import convert_req_index_to_global
from ..ops.attention.dsa_index_key_norm import index_key_norm
from ..ops.attention.dsa_prefill_share import disagg_role
from ..ops.attention.fp8_mqa_logits import fp8_mqa_logits
from ..ops.attention.fused_cat_fp8 import fused_cat_fp8
from ..ops.attention.gather_k_cache_paged import k31_gather_k_cache_paged
from ..ops.attention.indexer_k_cache_scatter_op import indexer_k_cache_scatter_op
from ..ops.attention.indexer_topk_decode import indexer_topk_decode
from ..ops.attention.indexer_topk_prefill import indexer_topk_prefill
from ..ops.attention.rubin_dsa_backend import rubin_dsa_backend_cls
from ..ops.attention.rubin_dsa_topk_remap import rubin_dsa_topk_remap
from ..ops.attention.rubin_fp8_paged_mqa_logits import rubin_fp8_paged_mqa_logits
from ..ops.attention.rubin_gvr_topk_decode import rubin_gvr_topk_decode
from ..ops.attention.rubin_gvr_topk_prefill import rubin_gvr_topk_prefill
from ..ops.attention.rubin_idx_weights import rubin_idx_weights
from ..ops.attention.rubin_pad_heads import rubin_pad_heads
from ..ops.comm.allgather_module import allgather_module
from ..ops.gemm.bf16_linear import bf16_linear
from ..ops.gemm.rubin_linear import rubin_linear
from ..ops.gemm.rubin_mxfp8_gemm import rubin_mxfp8_gemm
from ..ops.gemm.rubin_mxfp8_linear import rubin_mxfp8_linear
from ..ops.gemm.rubin_upcast_fp32 import rubin_upcast_fp32
from ..ops.moe.dsv3_router_gemm_op import dsv3_router_gemm_op
from ..ops.moe.rubin_cutedsl_fc1 import rubin_cutedsl_fc1
from ..ops.moe.rubin_cutedsl_fc2_finalize import rubin_cutedsl_fc2_finalize
from ..ops.moe.rubin_moe_residual_add import rubin_moe_residual_add
from ..ops.moe.rubin_moe_sort import rubin_moe_sort
from ..ops.norm.attn_res_add_rmsnorm_fwd import attn_res_add_rmsnorm_fwd
from ..ops.norm.attn_res_fwd import attn_res_fwd
from ..ops.norm.attn_res_rmsnorm_fwd import attn_res_rmsnorm_fwd
from ..ops.norm.rubin_attn_res_add_rmsnorm_persistent import rubin_attn_res_add_rmsnorm_persistent
from ..ops.norm.rubin_attn_res_add_rmsnorm_persistent_x import (
    persistent_x_available,
    rubin_attn_res_add_rmsnorm_persistent_x,
)
from ..ops.norm.rubin_attn_res_moe_add_rmsnorm import (
    rubin_attn_res_moe_add_rmsnorm,
    rubin_attn_res_moe_add_rmsnorm_mx,
)
from ..ops.norm.rubin_attn_res_rmsnorm import rubin_attn_res_add_rmsnorm, rubin_attn_res_rmsnorm
from ..ops.spec.dflash2_block_verify import dflash2_block_verify_seeded
from ..ops.spec.dflash2_global_top32 import k31_dflash2_merge_top32, k31_dflash2_shard_top32
from ..ops.spec.dflash2_greedy_verify import dflash2_greedy_verify
from ..ops.spec.dflash2_grouped_conv import dflash2_grouped_conv
from ..ops.spec.dflash2_proposal_load import k31_dflash2_proposal_load
from ..ops.spec.dflash2_sampling import DFlash2ProposalStore, dflash2_sampled_step
from ..ops.spec.dflash2_selector_draft import k31_dflash2_selector_draft
from ..ops.spec.dflash2_selector_greedy_path import k31_dflash2_selector_greedy_path
from ..ops.spec.dflash2_window_rows import dflash2_window_rows
from ..ops.spec.draft_context_cache import DRAFT_FP8_KV_SCALE
from ..ops.spec.flashinfer_compute_probs_from_logits import flashinfer_compute_probs_from_logits
from ..ops.spec.flashinfer_sample_from_logits import flashinfer_sample_from_logits
from ..ops.spec.fused_compute_probs_from_logits import fused_compute_probs_from_logits
from ..ops.spec.fused_sample_from_logits import fused_sample_from_logits
from ..ops.spec.mla_norm_rope import dflash2_mla_block_rope, dflash2_mla_ctx_norm_rope
from ..ops.spec.rubin_dflash2_worker import (
    RubinDFlash2WorkerMixin,
    agree_draft_tokens,
    agree_verified_tokens,
    rubin_draft_config_kwargs,
)
from ..ops.spec.strict_accept import strict_accept
from ..ops.spec.unified_draft_cache_worker import UnifiedDraftCacheWorkerMixin
from ..ops.spec.vocab_parallel_argmax import vocab_parallel_argmax
from ..ops.spec.vocab_parallel_sample import vocab_parallel_sample
from ..ops.ssm.kda_decode import kda_decode as kda_decode_op
from ..ops.ssm.reset_recurrent_state_rows import reset_recurrent_state_rows
from ..ops.ssm.rubin_kda_dsa_cache_manager import bind_kv_cache_args, rubin_runtime_hooks
from ..ops.ssm.rubin_kda_metadata import RubinKdaMetadata
from ..ops.ssm.rubin_kda_mtp_verify import rubin_kda_mtp_verify
from ..ops.ssm.rubin_kda_prefill import rubin_kda_prefill
from ..ops.ssm.rubin_kda_prefill_conv import rubin_kda_prefill_conv
from ..ops.ssm.rubin_rms_norm_gated_token_major import rubin_rms_norm_gated_token_major
from ..ops.vision.moonvit_attention import moonvit_attention
from ..ops.vision.moonvit_gelu import moonvit_gelu
from ..ops.vision.moonvit_patch_embed import moonvit_patch_embed
from ..ops.vision.moonvit_rms_norm import moonvit_rms_norm
from ..ops.vision.moonvit_rope_2d import (
    TABLE_SIDE,
    THETA,
    moonvit_rope_2d,
    moonvit_rope_angles,
    moonvit_rope_table,
)
from . import weights as _weights
from .ops.activation.rubin_situ_and_mul_mxfp8 import rubin_situ_and_mul_mxfp8
from .ops.attention.rubin_mla_ctx_q_fp8 import rubin_mla_ctx_q_fp8
from .ops.moe.rubin_megamoe import (
    megamoe_bucket,
    megamoe_ep_group,
    megamoe_exchange_ctas,
    rubin_megamoe,
    rubin_megamoe_workspace,
)
from .ops.moe.rubin_megamoe_in_bucket import rubin_megamoe_in_bucket
from .ops.moe.rubin_megamoe_quantize import rubin_megamoe_quantize
from .ops.moe.rubin_moe_a2a import (
    rubin_moe_a2a_combine,
    rubin_moe_a2a_dispatch,
    rubin_moe_a2a_workspace,
)
from .ops.moe.rubin_moe_zero_local_rows import rubin_moe_zero_local_rows
from .ops.moe.rubin_route_topk_out import MAX_M as _MOE_ROUTE_MAX_M
from .ops.moe.rubin_route_topk_out import rubin_route_topk_out
from .ops.spec.rubin_dflash2_window_fp32 import rubin_dflash2_window_fp32

# Identity. The server role is the one the runtime publishes (``ops/attention/dsa_prefill_share``),
# fixed at construction.
_TARGET = "kimi_k31_pro_nvfp4__sm_107__dep16"
_COMPUTE_CAPABILITY = (10, 7)
_LAYOUT = "dep16"
_ROLES = ("context", "generation")

# The recipe; routing admits no other.
_QUANT_ALGO = "MIXED_PRECISION"  # hf_quant_config.json of the ModelOpt export
_INDEXER_K_DTYPE = "fp8"
_STATE_DTYPE = torch.float32  # KDA recurrent state (mamba_ssm_cache_dtype)
_DRAFT_HISTORY_DTYPE = torch.float8_e4m3fn

# The K3.1 text model the code is written for; the core asserts the config matches.
_K31 = MappingProxyType(
    {
        "hidden_size": 7168,
        "num_hidden_layers": 93,
        "vocab_size": 163840,
        "num_attention_heads": 96,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "rms_norm_eps": 1e-5,
        "intermediate_size": 33792,
        "first_k_dense_replace": 1,
        "num_experts": 896,
        "num_experts_per_token": 16,
        "num_expert_group": 1,
        "routed_scaling_factor": 1.0,
        "moe_intermediate_size": 3072,
        "num_shared_experts": 2,
        "routed_expert_hidden_size": 3584,
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "attn_res_block_size": 12,
    }
)
_K31_KDA = MappingProxyType(
    {"num_heads": 96, "head_dim": 128, "short_conv_kernel_size": 4, "gate_lower_bound": -5.0}
)
_K31_DSA = MappingProxyType(
    {"index_num_heads": 32, "index_head_dim": 128, "index_topk": 2048, "index_rope_dim": 0}
)
# The MoonViT3d tower and PatchMergerMLPV2 projector of the composite config's vision config (the
# variant the family's tower builds); a server that builds the tower asserts the config matches.
_K31_VISION = MappingProxyType(
    {
        "patch_size": 14,
        "init_pos_emb_height": 64,
        "init_pos_emb_width": 64,
        "init_pos_emb_time": 4,
        "pos_emb_type": "divided_fixed",
        "vt_num_attention_heads": 12,
        "vt_num_hidden_layers": 27,
        "vt_hidden_size": 1024,
        "vt_intermediate_size": 4096,
        "qkv_hidden_size": 1536,
        "mm_hidden_size": 1024,
        "text_hidden_size": 7168,
        "merge_kernel_size": (2, 2),
        "merge_type": "sd2_tpool",
        "projector_ln_eps": 1e-5,
        "norm_type": "rmsnorm",
        "mlp_type": "mlp2",
        "activation_func": "gelu_pytorch_tanh",
        "attn_bias": False,
        "patch_embed_proj_bias": False,
        "linear_bias": False,
    }
)

# The layout; the core asserts the Mapping matches.
_WORLD_SIZE = 16
_ATTENTION_DP = True
_HELIX_CP = 1
_MLA_QUERY_HEADS = 96
_KDA_TP = 1
_MOE_TP = 1
_MOE_EP = 16

# Launch constants per server role, logged in one line at startup; a key in _DETAILS is
# logged inside its parent's value.
# A verify step of three requests or more takes o_proj's MXFP8 operand from the verify kernel
# (kda_o_proj_mx_handoff).
# A context step adds its MoE output to the residual in one launch at any row count
# (moe_residual_one_pass). fuse_moe_residual stays a generation key: it also sends rows up to
# attn_res_pdl_max_m to the PDL mix kernels, and that bound is 0 on a context server.
# A generation server's persistent mixes run their epilogues (attn_res_mx_moe, attn_res_router_f32,
# attn_res_moe_fold, attn_res_grid_cap) and its decode indexer logits launch with PDL
# (dsa_idx_logits_pdl); see ``_X_LEVERS``. Its graph decode steps issue W_UK and the RoPE kernel
# before the indexer join (dsa_rope_before_idx_join), and its drafter projects q_a and kv_a in one
# GEMM over their stacked rows (draft_q_a_kv_a_fused); see ``_Levers``. MegaMoE's generation
# kernel runs its token exchange with 12 pusher CTAs instead of its 27 (megamoe_pusher_ctas): fewer
# exchange CTAs are resident while the kernel's 46 four-CTA clusters launch, so fewer of them wait
# for the exchange's refine to end; every byte moved and every output is the same. On graph steps the shared
# experts' gate_up GEMM keeps the exchange's CTAs and 8 spare SMs out of its grid
# (shared_gate_up_leave_sms), so the exchange starts on them.
_CONSTANTS = MappingProxyType(
    {
        "context": MappingProxyType(
            {
                "attn_res_topology": "per_token",
                "draft_ctx_latent": "precompute_context_kv",
                "draft_ctx_weights": "history",
                "draft_history_dtype": "fp8",
                "index_k_gather_triton": "layout_gated",
                "indexer_logits_elem_budget": 536870912,
                "kda_fused_prefill_conv": True,
                "kda_gated_rmsnorm_multirow": "shape_gated",
                "kda_qkvg_fused": True,
                "mla_ctx_fused_q_fp8": True,
                "mla_output_gate": "sigmoid_gate_mul",
                "moe_backend": "cutedsl",
                "moe_residual_one_pass": True,
                "mxfp8_gemm": "rubin_dense_blockscaled_gemm",
                "pad128_prefill": True,
                "router_bf16": True,
                "shared_down_mx_handoff": True,
                "shared_gate_up_fused": True,
                "topk_remap_triton": "shape_gated",
                "zero_gpu_pools": True,
            }
        ),
        "generation": MappingProxyType(
            {
                "attn_res_grid_cap": True,
                "attn_res_moe_fold": True,
                "attn_res_mx_moe": True,
                "attn_res_pdl": 2,
                "attn_res_pdl_max_m": 64,
                "attn_res_router_f32": True,
                "attn_res_topology": 1,
                "draft_block_rope": "dflash2_mla_block_rope",
                "draft_ctx_latent": "precompute_context_kv",
                "draft_history_dtype": "fp8",
                "draft_mla_norm_rope": "mla_norm_rope",
                "draft_q_a_kv_a_fused": True,
                "dsa_gate_early": True,
                "dsa_idx_fork": True,
                "dsa_idx_logits_pdl": True,
                "dsa_rope_before_idx_join": True,
                "dsa_single_atom": True,
                "fuse_moe_residual": True,
                "index_k_dtype": "fp8",
                "index_k_evict_first": True,
                "indexer_logits_elem_budget": 536870912,
                "kda_o_proj_mx_handoff": True,
                "kda_qkvg_fused": True,
                "kda_replay_seed_stacked": True,
                "latent_bf16_gemm": "cublas_linear",
                "megamoe_pusher_ctas": 12,
                "mla_output_gate": "sigmoid_gate_mul",
                "moe_backend": "megamoe_cutedsl",
                "moe_shared_stream_overlap": True,
                "mxfp8_gemm": "rubin_dense_blockscaled_gemm",
                "router_bf16": False,
                "shared_gate_up_fused": True,
                "shared_gate_up_leave_sms": True,
                "zero_gpu_pools": True,
            }
        ),
    }
)

# Detail key -> the key whose logged value it belongs to.
_DETAILS = MappingProxyType(
    {
        "attn_res_pdl_max_m": "attn_res_pdl",
    }
)


class _Levers(NamedTuple):
    """The decode levers of a server role, bound at construction (``rt.levers``, ``_levers``): a
    generation server's constants turn them on, a context server runs none. Off, a step issues,
    and a graph captures, what it did without the lever; on, the same products on the same
    operands:

    * ``draft_q_a_kv_a_fused``: each drafter layer's q_a and kv_a as one GEMM over their stacked
      rows, ``d_l{j}_q_a_kv_a`` (``draft_step``); q_a is a view of it, kv_a's rows a copy of
      ``d_kv_a``'s (which stacks the five layers' for the history projection);
    * ``dsa_rope_before_idx_join``: W_UK and the RoPE kernel, which read q and the latent rows only,
      run before the current stream joins the indexer's stream; the logits weights, the indexer
      and the attention after the join (``mla_generate``).
    """

    draft_q_a_kv_a_fused: bool = False
    dsa_rope_before_idx_join: bool = False


def _levers(role: str) -> _Levers:
    """The levers of a server role: its constants' values (``_CONSTANTS[role]``), off where unset."""
    c = _CONSTANTS[role]
    return _Levers(**{key: bool(c.get(key, False)) for key in _Levers._fields})


_CACHE_ARGS = MappingProxyType(
    {
        "context": MappingProxyType(
            {
                "indexer_k_dtype": "fp8",
                "kv_dtype": "fp8",
                "state_dtype": torch.float32,
                "zero_gpu_pools": True,
            }
        ),
        "generation": MappingProxyType(
            {
                "indexer_k_dtype": "fp8",
                "kv_dtype": "fp8",
                "state_dtype": torch.float32,
                "zero_gpu_pools": True,
            }
        ),
    }
)

# Stock engine names this file imports directly, each a computation outside the family's ops; helpers reached
# through the family's ops/ are not listed.
UNCERTIFIED_GENERIC_CALLS = (
    "tensorrt_llm._torch.models.modeling_multimodal_utils.find_input_mm_embeds",
    "tensorrt_llm._torch.models.modeling_multimodal_utils.get_multimodal_embeddings",
    "tensorrt_llm._torch.models.modeling_utils.DecoderModel",
    "tensorrt_llm._torch.models.modeling_utils.DecoderModelForCausalLM",
    "tensorrt_llm._torch.pyexecutor.sampler.ops.flashinfer.resolve_advanced_sampling_filters",
    "tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla.GREEDY_TEMPERATURE_THRESHOLD",
    "tensorrt_llm._torch.speculative.dflash.DFlashWorker",
    "tensorrt_llm._torch.speculative.interface.SpecWorkerBase",
    "tensorrt_llm._torch.speculative.should_use_separate_draft_kv_cache",
    "tensorrt_llm._torch.speculative.vocab_parallel.VocabShardedLogits",
    "tensorrt_llm.functional.RopeEmbeddingUtils",
)


# --------------------------------------------------------------------------------------------------
# Step and runtime interface: one step's batch state, the runtime resources, the weight views.
#
# A block function (the KDA, MLA + DSA, MoE and drafter blocks) reads what it needs of another
# block through these and nothing else:
#
# * ``L``: one layer's weight view at this rank's share (``_KdaW``, ``_MlaW``, ``_MoeW``,
#   ``_DenseW``), read once off the realized parameters in ``post_load_weights``;
# * ``st``: ``_Step``, this step's batch state, projected once per forward from the engine's
#   metadata (``_build_step_args``); ``st.dsa`` is the step's DSA state (``_DsaStep``);
# * ``rt``: ``_Rt``, the runtime resources (streams, communication closures, workspaces, the
#   identity RoPE table), built once in ``post_load_weights`` and never during a CUDA graph
#   capture.
#
# Block functions (positional parameters fixed):
#
# * KDA: ``kda_prefill(L, x, st, rt)``, ``kda_decode(L, x, st, rt)``, ``kda_verify(L, x, st, rt)``;
# * MLA + DSA: ``mla_prefill(L, i, x, st, rt)``, ``mla_generate(L, i, x, st, rt)``;
# * MoE: ``latent_moe(L, i, x, st, rt)``, ``dense_mlp(L0, x, st, rt)``;
# * DFlash2: ``draft_step(DW, hidden_streams, st, rt)``, the container ``DraftModel`` and the
#   worker ``DFlashWorker`` (``draft_config_for`` builds the drafter's config).
#
# The mixers and ``dense_mlp`` return ``[rows of x, hidden]`` BF16, whole: under attention DP16
# each rank runs every head and the whole dense MLP over its own requests, so no slot reduces
# anything. ``latent_moe`` returns ``_MoeOut(routed, shared)``: the routed experts' output (back
# from the all-to-all or MegaMoE) and the shared experts', whole for the rank's own rows. The next
# mix adds ``bf16(routed + shared)`` to the residual (``moe_norm``, ``moe_add``).
#
# ``i`` is the global layer index. ``x`` holds the rows the function serves. A step with context
# rows (they come first) runs ``PrefillTarget``: ``mla_prefill`` and ``kda_prefill`` get every row
# of the step and serve the generation rows after the context rows themselves (one call per layer:
# the projections, the index-key insert and the output projection run once over all rows). The MoE
# serves such a step with its context path (``st.num_contexts > 0``). A decode step runs
# ``DecodeTarget``: ``mla_generate``, ``kda_decode`` or ``kda_verify`` get every row. Rows past
# ``st.num_tokens`` are CUDA graph padding.
#
# The attention residual. Each layer mixes the running prefix sum with the bank of block snapshots
# (``attn_res``) ahead of each sublayer; a layer at a block boundary first stores its input stream
# in the bank and starts the prefix sum over. The slots are ``_Rt``'s (the core calls them, the
# blocks do not); each picks its kernel by the step's row count (``_mix_slots``).
# --------------------------------------------------------------------------------------------------

# Rows per call of the attn_res_fwd kernel; longer prefill chunks run in slices.
_ATTN_RES_OP_MAX_TOKENS = 16384
# The per-token fused mix: up to 1 row with the per-token topology, 32 with the split one.
_FUSED_MAX_ROWS = {"per_token": 1, 1: 32}
# Candidates (snapshots + the prefix sum) the persistent kernel takes: 2 to 9.
_PERSISTENT_MAX_CANDIDATES = 9
# The persistent kernel's epilogues and the decode indexer logits' PDL launch, each by a launch constant of the
# role (``_x_levers``): on for a generation server, off (today's calls) for a context server:
#   mx_moe      (attn_res_mx_moe) the KDA pre-attention mix folds the MoE residual update and hands the
#               projection the MXFP8 form (in place of rubin_moe_residual_add before it and the projection's
#               quantization after it);
#   router_f32  (attn_res_router_f32) a MoE layer's post-attention mix stores the FP32 router input (in place
#               of rubin_upcast_fp32);
#   moe_fold    (attn_res_moe_fold) the MLA pre-attention mix and the output mix fold the MoE residual update;
#   grid_cap    (attn_res_grid_cap) every persistent mix launches min(SMs, rows) CTAs;
#   logits_pdl  (dsa_idx_logits_pdl) the decode indexer logits kernel launches with programmatic dependent
#               launch.
# The first four take rubin_attn_res_add_rmsnorm_persistent_x (a CuTe DSL kernel); on a device it does not run on
# they stay off and one WARNING line says so. Every output is bit for bit what the calls they replace make; with
# all five off a step runs those calls.
_X_LEVERS = MappingProxyType(
    {
        "mx_moe": "attn_res_mx_moe",
        "router_f32": "attn_res_router_f32",
        "moe_fold": "attn_res_moe_fold",
        "grid_cap": "attn_res_grid_cap",
        "logits_pdl": "dsa_idx_logits_pdl",
    }
)
# The levers that take the epilogue op (rubin_attn_res_add_rmsnorm_persistent_x).
_X_OP_LEVERS = ("mx_moe", "router_f32", "moe_fold", "grid_cap")
# Rows the epilogue op serves: a generation step's; longer (prefill) steps keep today's persistent call.
_X_MAX_ROWS = 4096
# Rows of one R128c4 scale tile: up to here the mix stores the KDA input's MXFP8 form.
_MX_TILE_ROWS = 128
# rt.events: the (fork, join) of the attention phase's side work, and of the indexer's work on
# the layer input alone (dsa_idx_fork).
_EV_ATTN = 0
_EV_IDX = 1
# The sparse MLA attends to this many Top-K entries per query row (index_topk).
_DSA_TOPK = _K31_DSA["index_topk"]
# The indexer's output rows, one buffer per step (the engine's cached-buffer name).
_DSA_TOPK_OUT = "indexer_topk_out_buffer"


class _DsaStep(NamedTuple):
    """One step's DSA state for the MLA layers (dep16), read once a step off the family DSA metadata.

    Tensors are metadata buffers or views of them, so a captured graph reads what the replayed
    step's ``prepare`` wrote. ``nc`` / ``ng`` / ``nct`` are the step's context requests,
    generation requests and context rows (0, ``num_generations``, 0 on a decode step).
    """

    next_n: int  # rows per generation request (1 + the verified draft length); 1 without them
    skip_gen: bool  # md.skip_indexer_for_gen_reqs: the generation rows take topk_dense
    skip_ctx: bool  # md.skip_indexer_for_ctx_reqs: the context rows take topk_dense (short)
    # Index-key insert (every live row of the step, once, before the phases split).
    slot_fp8: torch.Tensor  # md.slot_mapping_fp8
    slot_scale: torch.Tensor  # md.slot_mapping_scale
    # Decode logits: FP8 DSL paged MQA logits, strided, one atom per request.
    ctx_lens: torch.Tensor  # md.gen_indexer_kv_lens_cuda_runtime
    block_table: torch.Tensor  # md.indexer_k_cache_block_offsets[nc:nc + ng]
    sched: torch.Tensor  # md.scheduler_metadata_buffer
    max_len: int  # md.get_indexer_max_seq_len()
    # Top-K.
    gen_kv_lens: torch.Tensor  # md.kv_lens_cuda_runtime[nc:nc + ng]
    topk_out: torch.Tensor  # the indexer's Top-K rows of the step [num_tokens, index_topk] int32
    topk_dense: torch.Tensor  # md.topk_indices_buffer: the dense Top-K (skip_ctx / skip_gen)
    radix_aux_indices: torch.Tensor  # md.radix_aux_indices
    radix_aux_logits: torch.Tensor  # md.radix_aux_logits
    # Local Top-K -> global latent-pool rows.
    req_idx_ctx: torch.Tensor  # md.req_idx_per_token[:nct]
    req_idx_gen: torch.Tensor  # md.req_idx_per_token[nct:num_tokens] - nc
    block_table_ctx: torch.Tensor  # md.block_table[:nc]
    block_table_gen: torch.Tensor  # md.block_table[nc:num_seqs]
    tokens_per_block: int
    # Prefill (context rows): chunks with their query tiles, and the full-KV gather slots.
    chunks: Any  # md.indexer_prefill_chunks: token / key bounds, cu_seqlen_ks / _ke, tile plan
    cu_ks: torch.Tensor | None  # md.cu_seqlen_ks[:nct]: each context row's first key
    cu_ke: torch.Tensor | None  # md.cu_seqlen_ke[:nct]: each context row's key end
    num_ctx_kv_tokens: int  # md.num_ctx_kv_tokens
    k_slot_fp8: torch.Tensor | None  # md.slot_mapping_fp8_fullkv; None without context rows
    k_slot_scale: torch.Tensor | None  # md.slot_mapping_scale_fullkv; None without context rows
    # The MLA ops' step arguments (request-major; dep16 has no token-major presentation). Under
    # attention DP the indexer never splits a prefill's queries.
    rope_gen: dict  # mla_rope_generation's metadata arguments
    sparse_gen: dict  # thop_attention's metadata arguments, generation call
    sparse_ctx: dict  # the same object: the context call reads the same batch state


class _Step(NamedTuple):
    """One step's batch state: what the blocks read of the engine's metadata, read once a step.

    ``DecodeTarget`` states ``num_contexts`` and ``num_ctx_tokens`` as 0 instead of reading them.
    """

    md: Any  # the step's attention metadata (the family DSA metadata)
    kda_md: Any  # md.mamba_metadata: KDA state slots, query offsets, snapshot plan
    cache: Any  # md.kv_cache_manager: KDA state and conv pools, DSA latent and index-key caches
    position_ids: torch.Tensor  # [num_tokens] int32
    num_tokens: int  # md.num_tokens: the live rows; the inputs may carry padding rows after them
    num_contexts: int  # context requests
    num_ctx_tokens: int  # context rows, the step's first rows
    num_generations: int  # generation requests
    rows_per_gen: int  # rows per generation request: 1 + the draft length the step verifies
    all_rank_num_tokens: tuple[int, ...]  # every DP rank's row count
    spec_md: Any  # the DFlash2 spec metadata (hidden-state taps), None without a drafter
    cuda_graph: bool  # md.is_cuda_graph: a CUDA graph's warmup or capture (the stream plans)
    dsa: _DsaStep  # the step's DSA state (indexer, Top-K, sparse MLA)


class _MoeOut(NamedTuple):
    """A latent-MoE layer's output, both parts whole; the next mix adds ``bf16(routed + shared)``."""

    routed: torch.Tensor  # [rows, hidden] BF16: latent all-reduce + norm + latent_up
    shared: torch.Tensor  # [rows, hidden] BF16: the shared experts, all-reduced


class _MixW(NamedTuple):
    """The weights of one attention-residual mix: score projection and norm, then the output norm."""

    proj: torch.Tensor  # the *_res_proj weight flattened [H]
    res_norm: torch.Tensor  # the *_res_norm weight [H] (scores the candidates)
    out_norm: torch.Tensor  # the RMSNorm the mix feeds [H]


class _Streams(NamedTuple):
    """The core's side streams, made once at construction (never during a capture)."""

    aux: torch.cuda.Stream  # rt.aux_stream: norm and indexer quant overlap, shared experts
    fork: torch.cuda.Stream  # rt.fork_stream: q_b beside the indexer's query projection
    idx: torch.cuda.Stream  # rt.idx_stream: the indexer's keys and weights (dsa_idx_fork)
    latent: torch.cuda.Stream  # latent_down beside the router
    memset: torch.cuda.Stream  # the CuteDSL experts' output memset beside FC1


class _DsaPools:
    """The DSA layers' cache views and the facts their step state needs (``rt.dsa_pools``).

    ``bind`` reads the views off the cache manager on the first step and again when the manager
    object changes (the KV estimation builds one, the final run another); it allocates nothing,
    so a capture may call it. All MLA layers share one FP8 latent pool (the manager raises
    otherwise), which their page parameters index.
    """

    def __init__(
        self, mla_layers: tuple[int, ...], rope_positions: int, gen_requests: int | None
    ) -> None:
        # Generation requests the decode-logits arena holds: one on a context server (an
        # attention-DP dummy), else the engine's request limit.
        self.gen_requests = gen_requests
        self.mla_layers = mla_layers
        self.rope_positions = rope_positions  # positions rt.mla_rope covers
        self.cache: Any = None
        self.idx_k_cache: dict[int, torch.Tensor] = {}  # layer -> its INDEX_KEY page view
        self.pool_params: dict[int, tuple[int, int]] = {}  # layer -> (page scale, layer offset)
        self.local: dict[int, int] = {}  # layer -> the manager's local layer index
        self.tokens_per_block = 0
        # The decode-logits arena of the MLA + DSA block: allocated by the first (eager) forward,
        # before any capture.
        self.idx_logits: torch.Tensor | None = None

    def bind(self, cache: Any) -> None:
        """Take the step's cache manager's views (a new manager replaces the old views)."""
        if cache is self.cache:
            return
        assert int(cache.max_seq_len) <= self.rope_positions, (
            f"the identity RoPE table covers {self.rope_positions} positions, the KV cache "
            f"{cache.max_seq_len}"
        )
        self.cache = cache
        self.idx_logits = None
        self.local = {i: int(cache.layer_offsets[i]) for i in self.mla_layers}
        self.idx_k_cache = {i: cache.get_indexer_k_cache_buffers(i) for i in self.mla_layers}
        # The prefill's paged index-key gather (k31_gather_k_cache_paged) reads each view as
        # [pages, tokens per page, 1, bytes per token] uint8 with contiguous tokens: checked once.
        for view in self.idx_k_cache.values():
            _, _, heads, per_token = view.shape
            assert (
                heads == 1
                and view.dtype == torch.uint8
                and view.stride(3) == 1
                and view.stride(1) == per_token
            ), f"the paged index-key gather cannot read a {tuple(view.shape)} {view.dtype} view"
        self.pool_params = {
            i: tuple(cache.get_primary_pool_page_index_params(self.local[i]))
            for i in self.mla_layers
        }
        cache.get_unique_primary_pool()  # raises unless the MLA layers share one pool
        self.tokens_per_block = int(cache.tokens_per_block)


class _MoeComm(NamedTuple):
    """The MoE sublayer's runtime resources (``rt.moe_comm``), built once in post_load_weights.

    One set of events serves every layer: each use records before it waits, in issue order.
    """

    latent_stream: torch.cuda.Stream  # latent_down
    memset_stream: torch.cuda.Stream  # the CuteDSL output memset (eager steps too)
    shared_fork: torch.cuda.Event
    shared_join: torch.cuda.Event
    latent_fork: torch.cuda.Event
    latent_ready: torch.cuda.Event
    memset_ready: torch.cuda.Event


class _Rt(NamedTuple):
    """The runtime resources one forward reads, built once in ``post_load_weights``.

    No field is a per-layer helper object: the blocks call the family entries with these
    resources as arguments.
    """

    dsa_pools: _DsaPools  # the DSA layers' index-key caches and page parameters
    mla_rope: torch.Tensor  # the MLA kernels' RoPE table: identity rotations (K3.1 MLA is NoPE)
    aux_stream: torch.cuda.Stream  # the side stream (norm / indexer quant overlap, shared)
    fork_stream: torch.cuda.Stream  # q_b beside the indexer's query projection (graph steps)
    idx_stream: torch.cuda.Stream  # the indexer's keys and weights (dsa_idx_fork)
    events: tuple  # (fork, join) CUDA event pairs, indexed _EV_ATTN, _EV_IDX
    attn_res_norm: Callable  # (prefix, bank, mw) -> normed
    add_norm: Callable  # (prefix, addend, bank, mw) -> (prefix + addend, normed)
    moe_norm: Callable  # (prefix, moe_out, bank, mw) -> (prefix + bf16(routed + shared), normed)
    moe_norm_mx: Callable  # moe_norm whose normed rows may come as (normed, codes, scales)
    # moe_norm with the update in the persistent kernel (lever moe_fold), or None
    moe_fold_norm: Optional[Callable]
    # (prefix, addend, bank, mw) -> (prefix, normed, FP32) (lever router_f32), or None
    mid_norm_f32: Optional[Callable]
    idx_logits_pdl: bool  # the decode indexer logits kernel launches with PDL (lever logits_pdl)
    reduce_norm: Callable  # (prefix, partial, bank, mw) -> (prefix, normed)
    attn_res_mix: Callable  # (prefix, bank, mw) -> the pre-norm mixture (a drafter tap)
    rms_norm: Callable  # (x, weight) -> normed
    moe_add: Callable  # (prefix, moe_out) -> prefix + bf16(routed + shared)
    reduce: Callable  # (out) -> out: block outputs are whole under attention DP
    moe_comm: _MoeComm  # the MoE sublayer's streams and events
    megamoe: Any  # generation: MegaMoE's per-bucket symmetric buffers and staging (the MoE block's)
    a2a: Any  # context: the all-to-all workspace and metainfo (the MoE block's)
    draft: Any  # the DFlash2 drafter's bound window entry, plan buffers and scratch
    levers: _Levers = _Levers()  # the role's decode levers (_levers); the default: all off
    # generation: SMs the shared experts' gate_up GEMM keeps out of its grid beside MegaMoE's token
    # exchange on graph steps (``_shared_gate_up_reserve``; 0: its full grid)
    shared_gate_up_reserve: int = 0


class _KdaW(NamedTuple):
    """A KDA layer's mixer at this rank's share (parameters ``l{i}_kda_<field>``).

    Checkpoint keys under ``language_model.model.layers.{i}.self_attn.``; P is this rank's KDA heads
    x 128. An MXFP8 weight is its E4M3 values (``*_w``) and UE8M0 1x32 scales relaid for the dense
    block-scaled GEMM (``*_s``).
    """

    layer: int  # derived: the global layer index, which selects the layer's state slots
    # q, k, v, g (4P), f_a (128), b (heads) rows, then zero rows to a multiple of 16 (``w1_rows``
    # stacks the six projections at load)
    qkvgfab_w: torch.Tensor
    qkvgfab_s: torch.Tensor  # their scales, stacked the same way (whole 128-row tiles each)
    fb: torch.Tensor  # f_b_proj: this rank's rows [P, 128], BF16
    o_w: torch.Tensor  # o_proj: this rank's input columns [H, P]
    o_s: torch.Tensor
    conv: torch.Tensor  # q_conv1d, k_conv1d, v_conv1d: this rank's channels, stacked [3P, 4]
    a_log: torch.Tensor  # A_log, this rank's heads, FP32
    dt_bias: torch.Tensor  # dt_bias, this rank's share, FP32
    o_norm: torch.Tensor  # o_norm.weight, the gated output RMSNorm [128]
    conv_t: torch.Tensor  # derived: conv as [3, 4, P], the decode kernel's layout
    conv_f32: torch.Tensor  # derived: conv in FP32, the verify kernel's
    o_norm_f32: torch.Tensor  # derived: o_norm in FP32, the decode and verify epilogues'
    zero_bias: torch.Tensor  # derived: the decode kernel's conv bias [P] (the layer has none)


class _MlaW(NamedTuple):
    """An MLA (DSA) layer's attention at this rank's share (parameters ``l{i}_mla_<field>``).

    Checkpoint keys under ``language_model.model.layers.{i}.self_attn.``; the ``idx_*`` fields are
    its DSA indexer's (``indexer.*``), whole on every rank. ``*_t`` fields are transposed views,
    taken once.
    """

    q_a_norm: torch.Tensor  # q_a_layernorm.weight [q_lora]
    kv_a_norm: torch.Tensor  # kv_a_layernorm.weight [kv_lora]
    q_b: (
        torch.Tensor
    )  # q_b_proj: the query heads this rank computes [heads x (nope + rope), q_lora]
    g: torch.Tensor  # g_proj, the sigmoid output gate: this rank's share [o_heads x v, H]
    o: torch.Tensor  # o_proj: this rank's input columns [H, o_heads x v]
    idx_wk_wp: (
        torch.Tensor
    )  # indexer.wk then indexer.weights_proj rows [index_dim + heads, H], FP32
    idx_k_norm_w: torch.Tensor  # indexer.k_norm.weight [index_dim], FP32
    idx_k_norm_b: torch.Tensor  # indexer.k_norm.bias [index_dim], FP32
    # Derived from l{i}_mla_kv_b (kv_b_proj regrouped: [every head's k_nope ; every head's v],
    # kv_lora wide), l{i}_mla_qkv_a (q_a_proj and kv_a_proj_with_mqa rows, stacked
    # [q_lora + kv_lora + rope, H]) and l{i}_mla_idx_wq_b (indexer.wq_b [index_heads x index_dim,
    # q_lora], BF16).
    k_b: torch.Tensor  # derived: kv_b's k_nope rows [heads, nope, kv_lora], the query absorption
    v_b_t: torch.Tensor  # derived: kv_b's v rows [heads, kv_lora, v], the output expansion
    qkv_a_t: torch.Tensor  # derived: qkv_a.t()
    idx_wq_b_t: torch.Tensor  # derived: idx_wq_b.t()


class _MoeW(NamedTuple):
    """A latent-MoE layer's dense weights at this rank's share (parameters ``l{i}_moe_<field>``).

    Checkpoint keys under ``language_model.model.layers.{i}.block_sparse_moe.``. The routed experts
    are the ``l{i}_moe_fc*`` rows of ``w2_rows``, read here like the rest.
    """

    router: torch.Tensor  # gate.weight [num_experts, H]: BF16, FP32 where router_bf16 is off
    router_bias: torch.Tensor  # gate.e_score_correction_bias [num_experts], FP32
    shared_gu_w: torch.Tensor  # shared_experts gate_proj then up_proj: this rank's rows [2S, H]
    shared_gu_s: torch.Tensor
    shared_dn_w: torch.Tensor  # shared_experts down_proj: this rank's columns [H, S]
    shared_dn_s: torch.Tensor
    latent_down: torch.Tensor  # routed_expert_down_proj [latent, H], BF16
    latent_norm: torch.Tensor  # routed_expert_norm.weight [latent]
    latent_up: torch.Tensor  # routed_expert_up_proj [H, latent], BF16
    # The routed experts of this rank's window, as ``w2_rows`` declares them (E local experts, I
    # this rank's slice of an expert's intermediate):
    fc1_w: torch.Tensor  # w1 / w3 rows, 16 gate then 16 up: [E, 2I, latent / 2] E2M1 pairs (U8)
    fc1_s: torch.Tensor  # their block scales [E, 2I, latent / 16] (E4M3 bytes), swizzled
    fc2_w: torch.Tensor  # w2 [E, latent, I / 2]
    fc2_s: torch.Tensor  # its block scales [E, latent, I / 16], swizzled
    fc31_in_scale: torch.Tensor  # [1] FP32: the experts' input quantization scale
    fc2_in_scale: torch.Tensor  # [1] FP32: FC1's output quantization scale
    fc31_alpha: torch.Tensor  # [E] FP32: FC1's dequantization scale per expert
    fc2_alpha: torch.Tensor  # [E] FP32: FC2's
    fc1_norm_const: torch.Tensor  # [E] FP32: MegaMoE's FC2 input scale per expert
    expert_offset: int  # derived: the first expert of this rank's window


# The _MoeW fields that read a parameter (the rest are derived).
_MOE_PARAMS = tuple(f for f in _MoeW._fields if f != "expert_offset")


class _DenseW(NamedTuple):
    """Layer 0's dense MLP at this rank's share (parameters ``l0_dense_<field>``), BF16.

    Checkpoint keys under ``language_model.model.layers.0.mlp.``.
    """

    gu: torch.Tensor  # gate_proj then up_proj: this rank's rows [2D, H]
    dn: torch.Tensor  # down_proj: this rank's columns [H, D]


class _LayerW(NamedTuple):
    """One decoder layer: its mixer and MLP views and its two attention-residual mixes.

    The mixes read ``l{i}_attn_*`` / ``l{i}_mlp_*`` (checkpoint keys under
    ``language_model.model.layers.{i}.``). Exactly one of ``kda`` / ``mla`` and one of ``moe`` /
    ``dense`` is set; the forward tells layers apart by these, not by type.
    """

    kda: _KdaW | None
    mla: _MlaW | None
    moe: _MoeW | None
    dense: _DenseW | None
    pre: _MixW  # the mix ahead of the mixer (self_attention_res_*, input_layernorm)
    mid: _MixW  # the mix ahead of the MLP (mlp_res_*, post_attention_layernorm)
    boundary: bool  # the layer opens an attn_res block (stores its input stream)
    n_bank: int  # snapshots in the bank when the layer starts


class _HeadW(NamedTuple):
    """The model's weights outside its layers, BF16.

    Checkpoint keys under ``language_model.model.``; the LM head is the shell's ``lm_head``.
    """

    embed: torch.Tensor  # embed_tokens.weight [vocab, H]
    out: _MixW  # the output mix (output_attn_res_*, norm)
    n_bank: int  # snapshots in the bank after the last layer


def _sparse_args(md: Any, nc: int, nct: int) -> dict:
    """thop_attention's metadata arguments of one step: what the sparse MLA call reads.

    The context counts are the step's as the target states them (0 on a decode step).
    """
    return dict(
        workspace=md.effective_workspace,
        sequence_length=md.kv_lens_cuda_runtime,
        host_past_key_value_lengths=md.kv_lens_runtime,
        host_total_kv_lens=md.host_total_kv_lens,
        context_lengths=md.prompt_lens_cuda_runtime,
        host_context_lengths=md.prompt_lens_cpu_runtime,
        host_request_types=md.host_request_types_runtime,
        max_context_q_len_override=md.max_context_q_len_override,
        kv_cache_block_offsets=md.kv_cache_block_offsets,
        host_kv_cache_pool_pointers=md.host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping=md.host_kv_cache_pool_mapping,
        cache_indirection=md.cache_indirection,
        block_ids_per_seq=md.block_ids_per_seq,
        tokens_per_block=md.tokens_per_block,
        max_num_requests=md.max_num_requests,
        max_num_sequences=md.max_num_sequences,
        beam_width=md.effective_beam_width,
        use_paged_context_fmha=md.use_paged_context_fmha,
        helix_position_offsets=md.helix_position_offsets,
        helix_is_inactive_rank=md.helix_is_inactive_rank,
        is_spec_decoding_enabled=md.is_spec_decoding_enabled,
        use_spec_decoding=md.use_spec_decoding,
        is_spec_dec_tree=md.is_spec_dec_tree,
        spec_decoding_generation_lengths=md.spec_decoding_generation_lengths,
        spec_decoding_position_offsets=get_spec_decoding_position_offsets(md),
        spec_decoding_packed_mask=md.spec_decoding_packed_mask,
        spec_decoding_bl_tree_mask_offset=md.spec_decoding_bl_tree_mask_offset,
        spec_decoding_bl_tree_mask=md.spec_decoding_bl_tree_mask,
        spec_decoding_target_max_draft_tokens=md.max_total_draft_tokens,
        force_prepare_spec_dec_tree_mask=md.force_prepare_spec_dec_tree_mask,
        spec_bl_tree_first_sparse_mask_offset_kv=md.spec_bl_tree_first_sparse_mask_offset_kv,
        num_sparse_topk=md.num_sparse_topk,
        flash_mla_tile_scheduler_metadata=md.flash_mla_tile_scheduler_metadata,
        flash_mla_num_splits=md.flash_mla_num_splits,
        num_contexts=nc,
        num_ctx_tokens=nct,
        max_context_length=md.max_context_length,
        max_seq_len=md.max_seq_len,
        trtllm_gen_jit_warmup=md.trtllm_gen_jit_warmup,
        is_cross=md.is_cross,
    )


def _rope_args(md: Any, cache: Any, nc: int) -> dict:
    """mla_rope_generation's metadata arguments of one step: what the rope call reads."""
    return dict(
        sequence_length=md.kv_lens_cuda_runtime,
        host_past_key_value_lengths=md.kv_lens_runtime,
        host_context_lengths=md.prompt_lens_cpu_runtime,
        num_contexts=nc,
        kv_cache_block_offsets=md.kv_cache_block_offsets,
        host_kv_cache_pool_pointers=cache.kv_cache_pool_pointers,
        host_kv_cache_pool_mapping=cache.kv_cache_pool_mapping,
        block_ids_per_seq=md.block_ids_per_seq,
        helix_tensor_params=[md.helix_position_offsets, md.helix_is_inactive_rank],
        residual_dim=getattr(cache, "mla_kv_cache_residual_dim", 0),
        tokens_per_block=cache.tokens_per_block,
        attention_window_size=int(md.max_seq_len),
        beam_width=md.beam_width,
    )


def _dsa_step_args(md: Any, pools: _DsaPools, num_tokens: int, nc: int, nct: int, ng: int):
    """Project the DSA metadata onto this step's ``_DsaStep`` (dep16), once per forward.

    ``nc`` / ``nct`` are 0 on a decode step (the decode target states them). The routing refuses
    ragged verification, so every generation request carries ``next_n`` rows and the MLA ops take
    the request-major presentation.
    """
    assert not md.is_ragged_verify, "dep16 runs uniform verify blocks only"
    num_gen_rows = num_tokens - nct
    next_n = num_gen_rows // ng if ng else 1
    req_idx_gen = md.req_idx_per_token[nct:num_tokens]
    sparse = _sparse_args(md, nc, nct)
    return _DsaStep(
        next_n=next_n,
        skip_gen=bool(md.skip_indexer_for_gen_reqs),
        skip_ctx=bool(md.skip_indexer_for_ctx_reqs),
        slot_fp8=md.slot_mapping_fp8,
        slot_scale=md.slot_mapping_scale,
        ctx_lens=md.gen_indexer_kv_lens_cuda_runtime,
        block_table=md.indexer_k_cache_block_offsets[nc : nc + ng],
        sched=md.scheduler_metadata_buffer,
        max_len=int(md.get_indexer_max_seq_len()),
        gen_kv_lens=md.kv_lens_cuda_runtime[nc : nc + ng],
        topk_out=md.get_empty(
            md.cuda_graph_buffers,
            (num_tokens, _DSA_TOPK),
            cache_name=_DSA_TOPK_OUT,
            dtype=torch.int32,
            capture_graph=md.is_cuda_graph,
        ),
        topk_dense=md.topk_indices_buffer,
        radix_aux_indices=md.radix_aux_indices,
        radix_aux_logits=md.radix_aux_logits,
        req_idx_ctx=md.req_idx_per_token[:nct],
        req_idx_gen=req_idx_gen - nc if nc else req_idx_gen,
        block_table_ctx=md.block_table[:nc],
        block_table_gen=md.block_table[nc : nc + ng],
        tokens_per_block=pools.tokens_per_block,
        chunks=md.indexer_prefill_chunks if nct else None,
        cu_ks=md.cu_seqlen_ks[:nct] if nct else None,
        cu_ke=md.cu_seqlen_ke[:nct] if nct else None,
        num_ctx_kv_tokens=int(md.num_ctx_kv_tokens) if nct else 0,
        # The metadata sets these only in a step with context requests (the indexer's prefill
        # plan): a generation server's metadata never has them.
        k_slot_fp8=md.slot_mapping_fp8_fullkv if nct else None,
        k_slot_scale=md.slot_mapping_scale_fullkv if nct else None,
        rope_gen=_rope_args(md, md.kv_cache_manager, nc),
        sparse_gen=sparse,
        sparse_ctx=sparse,
    )


def _build_step_args(
    md: Any,
    position_ids: torch.Tensor,
    spec_metadata: Any,
    pools: _DsaPools,
    *,
    decode: bool = False,
) -> _Step:
    """Project the engine's metadata onto this step's ``_Step`` (the step contract).

    ``decode``: the decode target's call. It runs only when the step has no context rows, so it
    states the two context counts as 0 instead of reading them. Attention DP16: the MoE runs over
    every rank's row count, and this rank pads its rows to the largest. ``pools`` must be bound to
    the step's cache manager.
    """
    num_tokens = int(md.num_tokens)
    counts = tuple(int(n) for n in md.all_rank_num_tokens)
    assert len(counts) == _WORLD_SIZE and num_tokens <= max(counts), (num_tokens, counts)
    num_contexts = 0 if decode else md.num_contexts
    num_ctx_tokens = 0 if decode else md.num_ctx_tokens
    num_generations = md.num_generations
    gen_rows = num_tokens - num_ctx_tokens
    return _Step(
        md=md,
        kda_md=md.mamba_metadata,
        cache=md.kv_cache_manager,
        position_ids=position_ids.view(-1),
        num_tokens=num_tokens,
        num_contexts=num_contexts,
        num_ctx_tokens=num_ctx_tokens,
        num_generations=num_generations,
        rows_per_gen=gen_rows // num_generations if num_generations else 1,
        all_rank_num_tokens=counts,
        spec_md=spec_metadata,
        cuda_graph=bool(md.is_cuda_graph),
        dsa=_dsa_step_args(md, pools, num_tokens, num_contexts, num_ctx_tokens, num_generations),
    )


def _layer_views(core: Any) -> tuple[_LayerW, ...]:
    """Read every layer's weight views off the realized parameters, once.

    Views only: ``w1_rows`` declares every parameter in the layout and dtype its kernel reads
    (the fused KDA projection, the FP32 router of a generation server under attention DP), so
    nothing here copies or re-points a parameter. The KDA decode / verify operands are derived
    tensors.
    """
    d = core.dims
    w = core.w
    kda_layers, mla_layers = set(d.kda_layers), set(d.mla_layers)
    moe_layers, dense_layers = set(d.moe_layers), set(d.dense_layers)
    p_kda = d.kda_heads * d.kda_head_dim
    views = []
    for i in range(d.num_layers):

        def p(name: str, i: int = i) -> torch.Tensor:
            return w[f"l{i}_{name}"]

        mla = None
        if i in mla_layers:
            kv_b = p("mla_kv_b")
            k_rows = d.mla_heads * d.nope
            mla = _MlaW(
                q_a_norm=p("mla_q_a_norm"),
                kv_a_norm=p("mla_kv_a_norm"),
                q_b=p("mla_q_b"),
                g=p("mla_g"),
                o=p("mla_o"),
                idx_wk_wp=p("mla_idx_wk_wp"),
                idx_k_norm_w=p("mla_idx_k_norm_w"),
                idx_k_norm_b=p("mla_idx_k_norm_b"),
                k_b=kv_b[:k_rows].view(d.mla_heads, d.nope, d.kv_lora),
                v_b_t=kv_b[k_rows:].view(d.mla_heads, d.v_dim, d.kv_lora).transpose(1, 2),
                qkv_a_t=p("mla_qkv_a").t(),
                idx_wq_b_t=p("mla_idx_wq_b").t(),
            )
        kda = None
        if i in kda_layers:
            conv, o_norm = p("kda_conv"), p("kda_o_norm")
            kda = _KdaW(
                layer=i,
                qkvgfab_w=p("kda_qkvgfab_w"),
                qkvgfab_s=p("kda_qkvgfab_s"),
                fb=p("kda_fb"),
                o_w=p("kda_o_w"),
                o_s=p("kda_o_s"),
                conv=conv,
                a_log=p("kda_a_log"),
                dt_bias=p("kda_dt_bias"),
                o_norm=o_norm,
                conv_t=conv.view(3, p_kda, d.kda_conv).transpose(1, 2).contiguous(),
                conv_f32=conv.float(),
                o_norm_f32=o_norm.float(),
                zero_bias=torch.zeros(p_kda, dtype=torch.bfloat16, device=conv.device),
            )
        moe = None
        if i in moe_layers:
            moe = _MoeW(
                **{f: p(f"moe_{f}") for f in _MOE_PARAMS},
                expert_offset=d.expert_offset,
            )
        dense = (
            _DenseW(**{f: p(f"dense_{f}") for f in _DenseW._fields}) if i in dense_layers else None
        )
        views.append(
            _LayerW(
                kda=kda,
                mla=mla,
                moe=moe,
                dense=dense,
                pre=_MixW(p("attn_res_proj").view(-1), p("attn_res_norm"), p("attn_norm")),
                mid=_MixW(p("mlp_res_proj").view(-1), p("mlp_res_norm"), p("mlp_norm")),
                boundary=i % d.attn_res_block == 0,
                n_bank=-(-i // d.attn_res_block),
            )
        )
    return tuple(views)


def _head_view(core: Any) -> _HeadW:
    """Read the model's own weights off the realized parameters, once."""
    d = core.dims
    w = core.w
    return _HeadW(
        embed=w["embed"],
        out=_MixW(w["out_res_proj"].view(-1), w["out_res_norm"], w["final_norm"]),
        n_bank=-(-d.num_layers // d.attn_res_block),
    )


def _identity_rope_table(positions: int, rope_dim: int, device: Any) -> torch.Tensor:
    """The MLA kernels' RoPE table with cos 1, sin 0 everywhere: the rotation is a copy.

    The layout of the engine's ``create_rope_const_params`` for ``rope_gpt_neox`` with
    ``duplicate_data``: FP32 ``[1, positions * rope_dim * 2]``, (cos, sin) pairs.
    """
    table = torch.zeros(1, positions * rope_dim * 2, dtype=torch.float32, device=device)
    table.view(-1, 2)[:, 0] = 1.0
    return table


class _XLevers(NamedTuple):
    """The levers of ``_X_LEVERS`` a server runs (``_x_levers``): on for a generation server, by its
    constants; a field left out is off."""

    mx_moe: bool = False
    router_f32: bool = False
    moe_fold: bool = False
    grid_cap: bool = False
    logits_pdl: bool = False


def _x_levers(role: str) -> _XLevers:
    """The levers ``role``'s launch constants turn on (``_X_LEVERS``: every one on a generation server, none
    on a context server), logged in one INFO line. The epilogue levers need the CuTe DSL kernel of
    ``rubin_attn_res_add_rmsnorm_persistent_x``: on a device it does not run on they stay off and one WARNING line
    names them."""
    c = _CONSTANTS[role]
    on = {k: bool(c.get(key, False)) for k, key in _X_LEVERS.items()}
    asked = [k for k in _X_OP_LEVERS if on[k]]
    if asked and not persistent_x_available():
        logger.warning(
            f"Kimi K3.1 Rubin attn_res levers: {_TARGET}/{role}: rubin_attn_res_add_rmsnorm_persistent_x's "
            f"CuTe DSL kernel does not run on this device; {', '.join(_X_LEVERS[k] for k in asked)} left off "
            "(today's calls)"
        )
        on.update(dict.fromkeys(asked, False))
    logger.info(
        f"Kimi K3.1 Rubin attn_res levers: {_TARGET}/{role}: "
        + ", ".join(f"{k}={'on' if on[k] else 'off'}" for k in _X_LEVERS)
    )
    return _XLevers(**on)


def _mix_slots(c: Any, eps: float, xl: Optional[_XLevers] = None) -> dict[str, Callable]:
    """The attention-residual slots of one server role (``c`` = ``_CONSTANTS[role]``).

    The row count picks the kernel, in this order: the PDL kernel up to
    ``attn_res_pdl_max_m`` rows, then (split topology) the persistent kernel above 32 rows, then
    the per-token kernel, then the unfused mix and norm. With ``fuse_moe_residual`` a MoE output
    joins the residual in one kernel: inside the PDL mix up to ``attn_res_pdl_max_m`` rows, else
    ``rubin_moe_residual_add`` before the mix (the deferred MoE residual update). With
    ``moe_residual_one_pass`` (a context server) that update is ``rubin_moe_residual_add`` at
    any row count, which rounds as the two torch adds do.

    ``xl``'s epilogue levers move rows the persistent kernel serves (past ``attn_res_pdl_max_m``, up to
    ``_X_MAX_ROWS``) to ``rubin_attn_res_add_rmsnorm_persistent_x``: ``grid_cap`` every such mix, ``mx_moe``
    ``moe_norm_mx``, ``moe_fold`` the extra slot ``moe_fold_norm``, ``router_f32`` the extra slot
    ``mid_norm_f32`` (None when their lever is off). Each returns today's bytes.
    """
    pdl_max_m = c["attn_res_pdl_max_m"] if c.get("attn_res_pdl") == 2 else 0
    persistent = c["attn_res_topology"] == 1
    fused_max = _FUSED_MAX_ROWS[c["attn_res_topology"]]
    fuse_moe_residual = bool(c.get("fuse_moe_residual", False))
    moe_residual_one_pass = bool(c.get("moe_residual_one_pass", False))

    def use_persistent(rows: int, k: int) -> bool:
        return persistent and rows > _FUSED_MAX_ROWS[1] and k + 1 <= _PERSISTENT_MAX_CANDIDATES

    mx_moe, router_f32, moe_fold, grid_cap = (
        bool(getattr(xl, k, False)) for k in ("mx_moe", "router_f32", "moe_fold", "grid_cap")
    )
    x_ops = mx_moe or router_f32 or moe_fold or grid_cap

    def use_x(rows: int, k: int) -> bool:
        """The persistent kernel's rows past the PDL kernel's, which the epilogue op serves."""
        return x_ops and rows > pdl_max_m and use_persistent(rows, k) and rows <= _X_MAX_ROWS

    def x_mix(prefix, addend, addend2, bank, mw: _MixW, **epilogues):
        return rubin_attn_res_add_rmsnorm_persistent_x(
            prefix,
            addend,
            addend2,
            bank,
            mw.proj,
            mw.res_norm,
            mw.out_norm,
            eps,
            eps,
            grid_cap=grid_cap,
            **epilogues,
        )

    def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return flashinfer_rmsnorm(x, weight, eps)

    def attn_res_mix(prefix: torch.Tensor, bank: torch.Tensor, mw: _MixW) -> torch.Tensor:
        rows, hidden = prefix.shape
        k = bank.shape[0]
        out = torch.empty_like(prefix) if rows > _ATTN_RES_OP_MAX_TOKENS else None
        for lo in range(0, rows, _ATTN_RES_OP_MAX_TOKENS):
            hi = min(rows, lo + _ATTN_RES_OP_MAX_TOKENS)
            mix = attn_res_fwd(
                prefix[lo:hi].reshape(hi - lo, 1, hidden),
                # A row slice of the [k, rows, hidden] bank is strided; the op reads dense inputs.
                bank[:, lo:hi].contiguous().view(k, hi - lo, 1, hidden),
                mw.proj,
                mw.res_norm,
                eps,
            )[0].view(hi - lo, hidden)
            if out is None:
                return mix
            out[lo:hi] = mix
        return out

    def attn_res_norm(prefix: torch.Tensor, bank: torch.Tensor, mw: _MixW) -> torch.Tensor:
        rows, hidden = prefix.shape
        k = bank.shape[0]
        if k == 0:
            return rms_norm(prefix, mw.out_norm)
        if rows <= pdl_max_m:
            return rubin_attn_res_rmsnorm(
                prefix, bank, mw.proj, mw.res_norm, mw.out_norm, eps, eps, early_trigger=False
            )
        if use_persistent(rows, k):
            if grid_cap and use_x(rows, k):
                return x_mix(prefix, None, None, bank, mw).output
            return rubin_attn_res_add_rmsnorm_persistent(
                prefix.view(rows, 1, hidden),
                None,
                bank.view(k, rows, 1, hidden),
                mw.proj,
                mw.res_norm,
                mw.out_norm,
                eps,
                eps,
            )[1].view(rows, hidden)
        if rows <= fused_max:
            return attn_res_rmsnorm_fwd(
                prefix.view(rows, 1, hidden),
                bank.view(k, rows, 1, hidden),
                mw.proj,
                mw.res_norm,
                mw.out_norm,
                eps,
                eps,
            ).view(rows, hidden)
        return rms_norm(attn_res_mix(prefix, bank, mw), mw.out_norm)

    def add_norm(prefix: torch.Tensor, addend: torch.Tensor, bank: torch.Tensor, mw: _MixW):
        """``(prefix + addend, normed mix)``: the sum rounds to BF16 as a torch add does."""
        rows, hidden = prefix.shape
        k = bank.shape[0]
        if k and rows <= pdl_max_m:
            return rubin_attn_res_add_rmsnorm(
                prefix,
                addend,
                bank,
                mw.proj,
                mw.res_norm,
                mw.out_norm,
                eps,
                eps,
                early_trigger=False,
            )
        if k and use_persistent(rows, k):
            if grid_cap and use_x(rows, k):
                r = x_mix(prefix, addend, None, bank, mw)
                return r.updated, r.output
            updated, normed = rubin_attn_res_add_rmsnorm_persistent(
                prefix.view(rows, 1, hidden),
                addend.view(rows, 1, hidden),
                bank.view(k, rows, 1, hidden),
                mw.proj,
                mw.res_norm,
                mw.out_norm,
                eps,
                eps,
            )
            return updated.view(rows, hidden), normed.view(rows, hidden)
        if k and rows <= fused_max:
            updated, normed = attn_res_add_rmsnorm_fwd(
                prefix.view(rows, 1, hidden),
                addend.view(rows, 1, hidden),
                bank.view(k, rows, 1, hidden),
                mw.proj,
                mw.res_norm,
                mw.out_norm,
                eps,
                eps,
            )
            return updated.view(rows, hidden), normed.view(rows, hidden)
        updated = prefix + addend
        return updated, attn_res_norm(updated, bank, mw)

    def moe_add(prefix: torch.Tensor, m: _MoeOut) -> torch.Tensor:
        """``bf16(prefix + bf16(routed + shared))``: two round-to-nearest adds either way."""
        if fuse_moe_residual:
            return rubin_moe_residual_add(m.routed, m.shared, prefix)
        if moe_residual_one_pass:
            return rubin_moe_residual_add(m.routed, m.shared, prefix, any_rows=True)
        return prefix + (m.routed + m.shared)

    def moe_norm(prefix: torch.Tensor, m: _MoeOut, bank: torch.Tensor, mw: _MixW):
        """``(prefix + bf16(routed + shared), normed mix)``."""
        if fuse_moe_residual and bank.shape[0] and prefix.shape[0] <= pdl_max_m:
            return rubin_attn_res_moe_add_rmsnorm(
                prefix,
                m.routed,
                m.shared,
                bank,
                mw.proj,
                mw.res_norm,
                mw.out_norm,
                eps,
                eps,
                early_trigger=False,
            )
        updated = moe_add(prefix, m)
        return updated, attn_res_norm(updated, bank, mw)

    def moe_norm_mx(prefix: torch.Tensor, m: _MoeOut, bank: torch.Tensor, mw: _MixW):
        """``moe_norm`` with the normed rows' MXFP8 form: ``(prefix, (normed, codes, scales))``.

        In the PDL kernel, up to one scale tile: the codes and R128c4 scales are the catalog
        quantizer's (what the projection's ``rubin_mxfp8_linear`` makes of the rows), so the KDA
        projection multiplies them without its quantization kernel. Elsewhere ``moe_norm``.
        """
        rows = prefix.shape[0]
        if fuse_moe_residual and bank.shape[0] and rows <= min(pdl_max_m, _MX_TILE_ROWS):
            updated, normed, codes, scales = rubin_attn_res_moe_add_rmsnorm_mx(
                prefix,
                m.routed,
                m.shared,
                bank,
                mw.proj,
                mw.res_norm,
                mw.out_norm,
                eps,
                eps,
                early_trigger=False,
            )
            return updated, (normed, codes, scales)
        if mx_moe and bank.shape[0] and use_x(rows, bank.shape[0]):
            r = x_mix(prefix, m.routed, m.shared, bank, mw, mx=True)
            return r.updated, (r.output, r.codes, r.scales)
        return moe_norm(prefix, m, bank, mw)

    def moe_fold_norm(prefix: torch.Tensor, m: _MoeOut, bank: torch.Tensor, mw: _MixW):
        """``moe_norm`` with the MoE residual update in the persistent kernel where it serves the rows."""
        if bank.shape[0] and use_x(prefix.shape[0], bank.shape[0]):
            r = x_mix(prefix, m.routed, m.shared, bank, mw)
            return r.updated, r.output
        return moe_norm(prefix, m, bank, mw)

    def mid_norm_f32(prefix: torch.Tensor, addend, bank: torch.Tensor, mw: _MixW):
        """A MoE layer's post-attention mix with the FP32 copy of its normed rows, the router's input:
        ``(prefix, normed, normed FP32 or None)``; ``addend`` is None at a block boundary."""
        rows, k = prefix.shape[0], bank.shape[0]
        if k and use_x(rows, k):
            r = x_mix(prefix, addend, None, bank, mw, f32=True)
            return (prefix if addend is None else r.updated), r.output, r.output_f32
        if addend is None:
            return prefix, attn_res_norm(prefix, bank, mw), None
        updated, normed = add_norm(prefix, addend, bank, mw)
        return updated, normed, None

    return dict(
        rms_norm=rms_norm,
        attn_res_mix=attn_res_mix,
        attn_res_norm=attn_res_norm,
        add_norm=add_norm,
        moe_add=moe_add,
        moe_norm=moe_norm,
        moe_norm_mx=moe_norm_mx,
        moe_fold_norm=moe_fold_norm if moe_fold else None,
        mid_norm_f32=mid_norm_f32 if router_f32 else None,
    )


def _whole(out: torch.Tensor) -> torch.Tensor:
    """A block's output under attention DP: already whole."""
    return out


def _build_runtime(core: Any) -> _Rt:
    """Build the runtime resources: streams, events, the norm slots, the RoPE table.

    Attention DP: no slot reduces. The role's MoE workspace is built here, a collective over the
    EP group (MegaMoE on a generation server, the all-to-all on a context server);
    ``DraftModel.bind_runtime`` sets ``rt.draft``.
    """
    c = _CONSTANTS[core.role]
    d = core.dims
    xl = _x_levers(core.role)
    mix = _mix_slots(c, core.eps, xl)
    # weights.py's ``dims`` keeps layer 0's MLP whole on every rank.
    assert d.dense_inter == _K31["intermediate_size"], d.dense_inter
    s = core.streams
    cfg = core.model_config
    megamoe = a2a = None
    shared_gate_up_reserve = 0
    if c["moe_backend"] == "megamoe_cutedsl":
        # MegaMoE's buffers over the EP group (collective): rows per rank of a step up to
        # max_num_tokens; a step splits into chunks of moe_max_num_tokens, at least every rank's
        # rows at once. megamoe_pusher_ctas binds the generation kernel's pusher CTAs before its
        # first build.
        megamoe = rubin_megamoe_workspace(
            megamoe_ep_group(cfg.mapping),
            d.moe_ep_rank,
            d.moe_ep,
            d.local_experts,
            d.latent,
            d.moe_inter,
            int(cfg.max_num_tokens),
            max(int(cfg.moe_max_num_tokens or 0), int(cfg.max_num_tokens) * d.dp_size),
            core.w["embed"].device,
            pusher_ctas=c["megamoe_pusher_ctas"],
        )
        shared_gate_up_reserve = _shared_gate_up_reserve(core.role, megamoe_exchange_ctas(megamoe))
    if c["moe_backend"] == "cutedsl":
        # The EP workspace holds up to max_num_tokens rows per rank. Dispatch sends NVFP4.
        # Combine sends BF16 by default and FP8 when use_low_precision_moe_combine is enabled.
        a2a = rubin_moe_a2a_workspace(
            cfg.mapping,
            num_experts=_K31["num_experts"],
            top_k=_K31["num_experts_per_token"],
            max_num_tokens_per_rank=int(cfg.max_num_tokens),
            hidden_size=d.latent,
            dtype=torch.bfloat16,
            use_low_precision_combine=cfg.use_low_precision_moe_combine,
        )
    moe_comm = _MoeComm(
        latent_stream=s.latent,
        memset_stream=s.memset,
        **{f: torch.cuda.Event() for f in _MoeComm._fields[2:]},
    )
    return _Rt(
        dsa_pools=_DsaPools(
            d.mla_layers, core.rope_positions, 1 if core.role == "context" else None
        ),
        mla_rope=_identity_rope_table(core.rope_positions, d.rope, core.w["embed"].device),
        aux_stream=s.aux,
        fork_stream=s.fork,
        idx_stream=s.idx,
        events=(
            (torch.cuda.Event(), torch.cuda.Event()),  # _EV_ATTN
            (torch.cuda.Event(), torch.cuda.Event()),  # _EV_IDX
        ),
        attn_res_norm=mix["attn_res_norm"],
        add_norm=mix["add_norm"],
        moe_norm=mix["moe_norm"],
        moe_norm_mx=mix["moe_norm_mx"],
        moe_fold_norm=mix["moe_fold_norm"],
        mid_norm_f32=mix["mid_norm_f32"],
        idx_logits_pdl=xl.logits_pdl,
        reduce_norm=mix["add_norm"],
        attn_res_mix=mix["attn_res_mix"],
        rms_norm=mix["rms_norm"],
        moe_add=mix["moe_add"],
        reduce=_whole,
        moe_comm=moe_comm,
        megamoe=megamoe,
        a2a=a2a,
        draft=None,
        levers=core.levers,
        shared_gate_up_reserve=shared_gate_up_reserve,
    )


# --------------------------------------------------------------------------------------------------
# KDA: a KDA layer's context prefill, one-token decode and DFlash2 verify block (dep16).
#
# Attention DP16: this rank runs all 96 KDA heads (K = V = 128) on its own requests and returns the
# whole ``o_proj`` output; nothing is reduced. Each block:
#
# * projects once: one MXFP8 GEMM for the fused ``[q | k | v | g | f_a | b]`` rows (the checkpoint's
#   ``in_proj_qkvgfab``), then the BF16 ``f_b`` GEMM on its ``f_a`` columns, which gives the
#   forget-gate logits;
# * runs the gated delta rule on the cache manager's FP32 state pool: the fused short convolution
#   and the BT16 chunked prefill with its mid-chunk snapshots (prefill), the one-token decode
#   kernel (decode), or the fused verify kernel, which commits the state after the verified token
#   and keeps the drafts for the next step's replay (verify);
# * applies the gated output RMSNorm (prefill: the token-major kernel; decode and verify: the kernel
#   epilogue), then the MXFP8 ``o_proj`` GEMM. A verify step of ``_KDA_O_MX_MIN_REQUESTS`` requests
#   or more has the verify kernel store the MXFP8 form of its core instead (the catalog quantizer's
#   1x32 E4M3 codes and R128c4 scales, bit for bit the projection's own quantization of the BF16
#   core) and o_proj multiplies those codes (``kda_o_proj_mx_handoff``): no BF16 core is stored, no
#   quantization runs.
#
# A decode or verify block takes ``x`` as the BF16 rows or as ``(rows, codes, scales)``: the rows with
# the MXFP8 form the mix ahead of the layer stored (E4M3 codes and the R128c4 scales of one 128-row
# tile, bit for bit the projection's own quantization of the rows); the fused projection then
# multiplies those codes and skips its quantization.
#
# dep16's generation lever of the KDA block is that o_proj hand-off; the L2 prefetch serves helix8
# and tep8.
#
# Reads of the layer view (``st.cache.mamba_layer_cache(L.layer)``): ``conv`` and ``temporal``; in
# verify, ``kda_conv_q`` / ``_k`` / ``_v``, ``kda_k_cache``, ``kda_g_cache``, ``kda_v_cache``,
# ``kda_beta_cache`` and ``prev_num_accepted_tokens``; after prefill and decode,
# ``commit_conv_window`` where the view has replay caches (``_kda_commit_window``). The engine builds
# a ``KDAReplayLayerCache`` on a generation or aggregated server, and the plain ``MambaLayerCache``
# (``conv`` and ``temporal`` only) on a disaggregated context server, which runs prefill only.
# --------------------------------------------------------------------------------------------------

# This rank's KDA heads (K = V = head_dim).
_KDA_H = _K31_KDA["num_heads"] // _KDA_TP
_KDA_D = _K31_KDA["head_dim"]
_KDA_P = _KDA_H * _KDA_D
# Columns of the fused projection: q, k, v, g (P each), f_a (head_dim), b (heads), zero padding.
_KDA_G0 = 3 * _KDA_P
_KDA_FA0 = 4 * _KDA_P
_KDA_B0 = _KDA_FA0 + _KDA_D
_KDA_B1 = _KDA_B0 + _KDA_H
_KDA_LOWER_BOUND = _K31_KDA["gate_lower_bound"]
_KDA_EPS = _K31["rms_norm_eps"]
_KDA_SCALE = _KDA_D**-0.5
# A verify step of at least _KDA_O_MX_MIN_REQUESTS requests hands o_proj the verify kernel's MXFP8
# form of the core (logged constant). The catalog-form MXFP8 output runs the verify kernel the BF16
# output runs, the two-CTA kernel above one wave of (head, request) tiles: 3 x 96 = 288 tiles > 212
# SMs, while one or two requests run the v2 kernel; from three requests on only the epilogue of the
# two-CTA kernel changes.
_KDA_O_MX = _CONSTANTS["generation"]["kda_o_proj_mx_handoff"]
_KDA_O_MX_MIN_REQUESTS = 3


def _kda_project(
    L: _KdaW, x: torch.Tensor, mx: tuple[torch.Tensor, torch.Tensor] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the fused projection ``y`` [rows, N] and the forget-gate logits [rows, P].

    ``mx``: the MXFP8 form of ``x`` (codes, R128c4 scales) when the mix stored it.
    """
    if mx is None:
        y = rubin_mxfp8_linear(x, L.qkvgfab_w, L.qkvgfab_s)
    else:
        y = rubin_mxfp8_gemm(mx[0], L.qkvgfab_w, mx[1], L.qkvgfab_s)
    # F.linear reads the f_a columns through their row stride, so nothing is copied.
    g = rubin_linear(y[:, _KDA_FA0:_KDA_B0], L.fb, False)
    return y, g


def _kda_split(x: Any) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """``(rows, mx)`` of a decode / verify block's input: BF16 rows, or rows with codes, scales."""
    if isinstance(x, tuple):
        return x[0], (x[1], x[2])
    return x, None


def _kda_heads(y: torch.Tensor, col: int) -> torch.Tensor:
    """Columns ``[col, col + P)`` of ``y`` as a ``[1, rows, H, D]`` view."""
    return y[:, col : col + _KDA_P].unflatten(1, (_KDA_H, _KDA_D)).unsqueeze(0)


def _kda_beta(y: torch.Tensor) -> torch.Tensor:
    """The beta logits as a ``[1, rows, H]`` view."""
    return y[:, _KDA_B0:_KDA_B1].unsqueeze(0)


def _kda_gen_slots(st: _Step) -> torch.Tensor:
    """The generation requests' state slots (int32)."""
    md = st.kda_md
    # The metadata keeps an aligned copy when the slice would not be 16-byte aligned.
    slots = md.generation_state_indices
    if slots is None:
        slots = md.state_indices[st.num_contexts : st.num_contexts + st.num_generations]
    return slots


def _kda_snapshot_windows(y: torch.Tensor, conv_pool: torch.Tensor, snap: Any) -> None:
    """Store the raw conv window that ends before each snapshot point into its pool slot."""
    width = conv_pool.shape[-1]
    offsets = torch.arange(-width, 0, device=y.device)
    tokens = (snap.token_indices[:, None] + offsets).reshape(-1)
    rows = y[:, : conv_pool.shape[1]].index_select(0, tokens)
    windows = rows.view(snap.num_snapshots, width, -1).transpose(1, 2)
    conv_pool.index_copy_(0, snap.slots, windows.to(conv_pool.dtype))


def _kda_commit_window(lc: Any, slots: torch.Tensor, conv: torch.Tensor) -> None:
    """Copy the live conv windows of ``slots`` into the view's replay caches, where it has them.

    Read as the engine's KDA mixer reads it: only a ``KDAReplayLayerCache`` has
    ``has_kda_replay_caches``. A disaggregated context server keeps no replay state; its
    generation server seeds the replay caches from the transferred conv state.
    """
    if getattr(lc, "has_kda_replay_caches", False):
        lc.commit_conv_window(slots, conv)


def kda_prefill(L: _KdaW, x: torch.Tensor, st: _Step, rt: _Rt) -> torch.Tensor:
    """Run the KDA mixer over every row ``x`` of a step with context rows; return ``[rows, hidden]``.

    The context rows (the first ``st.num_ctx_tokens``) run the chunked prefill, the generation
    rows after them the decode or verify kernel, padding rows past ``st.num_tokens`` are zero;
    one o_proj over all rows.
    """
    tc, n = st.num_ctx_tokens, st.num_tokens
    cores = [_kda_prefill_core(L, x[:tc], st, rt)]
    if n > tc:
        gen = _kda_verify_core if st.rows_per_gen > 1 else _kda_decode_core
        cores.append(gen(L, x[tc:n], st, rt))
    if x.shape[0] > n:
        cores.append(x.new_zeros((x.shape[0] - n, _KDA_P)))
    core = cores[0] if len(cores) == 1 else torch.cat(cores)
    return rubin_mxfp8_linear(core, L.o_w, L.o_s)


def _kda_prefill_core(L: _KdaW, x: torch.Tensor, st: _Step, rt: _Rt) -> torch.Tensor:
    """The context rows' gated, normed core ``[rows, P]`` (before o_proj)."""
    md = st.kda_md
    n = st.num_contexts
    rows = x.shape[0]
    lc = st.cache.mamba_layer_cache(L.layer)
    slots = md.state_indices[:n]
    has_init = md.has_initial_states[:n]
    if rows == 0:
        reset_recurrent_state_rows(lc.temporal, slots, has_init, lc.conv)
        _kda_commit_window(lc, slots, lc.conv)
        return x.new_empty((0, _KDA_P))
    y, g = _kda_project(L, x)
    snap = md.kda_snapshot
    if snap is not None:
        # Before the convolution, which moves the live windows on.
        _kda_snapshot_windows(y, lc.conv, snap)
    # Fresh requests start from a zero state; the convolution clears their windows itself.
    reset_recurrent_state_rows(lc.temporal, slots, has_init)
    q, k, v = rubin_kda_prefill_conv(
        src=y,
        weight=L.conv,
        conv_state=lc.conv,
        starts=md.query_start_loc[: n + 1],
        slots=slots,
        has_init=has_init,
        max_seqlen=md.kda_max_seqlen,
    )
    o = rubin_kda_prefill(
        q=q,
        k=k,
        v=v,
        g=g.view(1, rows, _KDA_H, _KDA_D),
        beta=_kda_beta(y),
        state_pool=lc.temporal,
        state_indices=slots,
        cu_seqlens=md.query_start_loc_long[: n + 1],
        A_log=L.a_log,
        dt_bias=L.dt_bias,
        scale=_KDA_SCALE,
        lower_bound=_KDA_LOWER_BOUND,
        seq_order=md.kda_seq_order,
        cu_chunks=md.kda_cu_chunks,
        snapshot_rows=None if snap is None else snap.pool_rows,
        snapshot_chunks=None if snap is None else snap.chunks,
    )
    # The next verify convolves from the replay caches' committed window.
    _kda_commit_window(lc, slots, lc.conv)
    core = rubin_rms_norm_gated_token_major(
        x=o.view(-1, _KDA_D),
        z=y[:, _KDA_G0:_KDA_FA0].view(rows, _KDA_H, _KDA_D),
        weight=L.o_norm,
        eps=_KDA_EPS,
        gate_activation="sigmoid",
    )
    return core.view(rows, _KDA_P)


def kda_decode(L: _KdaW, x: Any, st: _Step, rt: _Rt) -> torch.Tensor:
    """Run the KDA mixer over one generation row per request; return ``[rows, hidden]``."""
    x, mx = _kda_split(x)
    return rubin_mxfp8_linear(_kda_decode_core(L, x, st, rt, mx), L.o_w, L.o_s)


def _kda_decode_core(
    L: _KdaW,
    x: torch.Tensor,
    st: _Step,
    rt: _Rt,
    mx: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """One generation row per request: the core ``[rows, P]`` (before o_proj)."""
    rows = x.shape[0]
    lc = st.cache.mamba_layer_cache(L.layer)
    slots = _kda_gen_slots(st)
    y, g = _kda_project(L, x, mx)
    out = x.new_empty((rows, 1, _KDA_H, _KDA_D))
    conv = lc.conv
    kda_decode_op(
        x_q=_kda_heads(y, 0),
        x_k=_kda_heads(y, _KDA_P),
        x_v=_kda_heads(y, 2 * _KDA_P),
        w_q_t=L.conv_t[0],
        w_k_t=L.conv_t[1],
        w_v_t=L.conv_t[2],
        bias_q=L.zero_bias,
        bias_k=L.zero_bias,
        bias_v=L.zero_bias,
        conv_state_q=conv[:, :_KDA_P],
        conv_state_k=conv[:, _KDA_P : 2 * _KDA_P],
        conv_state_v=conv[:, 2 * _KDA_P :],
        a_log=L.a_log,
        g=g.view(1, rows, _KDA_H, _KDA_D),
        dt_bias=L.dt_bias,
        beta=_kda_beta(y),
        onorm_g=_kda_heads(y, _KDA_G0),
        onorm_weight=L.o_norm_f32,
        ssm_state_indices=slots,
        state=lc.temporal,
        lower_bound=_KDA_LOWER_BOUND,
        scale=_KDA_SCALE,
        onorm_eps=_KDA_EPS,
        output=out,
    )
    _kda_commit_window(lc, slots, conv)
    return out.view(rows, _KDA_P)


def kda_verify(L: _KdaW, x: Any, st: _Step, rt: _Rt) -> torch.Tensor:
    """Run the KDA mixer over the DFlash2 verify blocks ``x``; return ``[rows, hidden]``.

    Commits the state after each request's accepted prefix and keeps its drafts for the next
    step's replay. With ``_KDA_O_MX`` and at least ``_KDA_O_MX_MIN_REQUESTS`` requests, o_proj
    multiplies the MXFP8 form of the core that the verify kernel stores (``quantize_output``: the
    operands ``rubin_mxfp8_linear`` would make of the BF16 core) and no quantization kernel runs.
    """
    x, mx = _kda_split(x)
    handoff = _KDA_O_MX and _kda_gen_slots(st).shape[0] >= _KDA_O_MX_MIN_REQUESTS
    if handoff:
        codes, scales = _kda_verify_core(L, x, st, rt, mx, mx_out=True)
        return rubin_mxfp8_gemm(codes, L.o_w, scales, L.o_s)
    return rubin_mxfp8_linear(_kda_verify_core(L, x, st, rt, mx), L.o_w, L.o_s)


def _kda_verify_core(
    L: _KdaW,
    x: torch.Tensor,
    st: _Step,
    rt: _Rt,
    mx: tuple[torch.Tensor, torch.Tensor] | None = None,
    mx_out: bool = False,
) -> Any:
    """The verify blocks' core ``[rows, P]`` (before o_proj).

    ``mx_out``: its MXFP8 form instead, ``(codes [rows, P], R128c4 scales)`` in the catalog
    quantizer's 1x32 groups, which the verify kernel stores in place of the BF16 core.
    """
    rows = x.shape[0]
    lc = st.cache.mamba_layer_cache(L.layer)
    y, g = _kda_project(L, x, mx)
    # The entry fixes the packed token layout and the fused output norm. g is the finished gate
    # input (f_b ran above); o_proj takes the BF16 output, or with mx_out the catalog-form MXFP8
    # output (the verify kernel's epilogue).
    core = rubin_kda_mtp_verify(
        x_q=_kda_heads(y, 0),
        x_k=_kda_heads(y, _KDA_P),
        x_v=_kda_heads(y, 2 * _KDA_P),
        w_q=L.conv_f32[:_KDA_P],
        w_k=L.conv_f32[_KDA_P : 2 * _KDA_P],
        w_v=L.conv_f32[2 * _KDA_P :],
        cs_q=lc.kda_conv_q,
        cs_k=lc.kda_conv_k,
        cs_v=lc.kda_conv_v,
        g=g.view(1, rows, _KDA_H, _KDA_D),
        beta=_kda_beta(y),
        A_log=L.a_log,
        dt_bias=L.dt_bias,
        recurrent_state=lc.temporal,
        k_cache=lc.kda_k_cache,
        g_cache=lc.kda_g_cache,
        v_cache=lc.kda_v_cache,
        beta_cache=lc.kda_beta_cache,
        ssm_state_indices=_kda_gen_slots(st),
        num_spec=st.rows_per_gen - 1,
        num_accepted_tokens=lc.prev_num_accepted_tokens,
        lower_bound=_KDA_LOWER_BOUND,
        scale=_KDA_SCALE,
        onorm_g=_kda_heads(y, _KDA_G0),
        onorm_weight=L.o_norm_f32,
        onorm_eps=_KDA_EPS,
        quantize_output=mx_out,
    )
    if mx_out:
        return core
    return core.view(rows, _KDA_P)


# --------------------------------------------------------------------------------------------------
# MLA + DSA (dep16): the 24 DSA layers' projections, indexer, sparse MLA and output gate, flat.
#
# Every K3.1 MLA layer is a DSA layer, and both its attention and its indexer are NoPE: the MLA
# kernels read the identity RoPE table ``rt.mla_rope`` and the indexer rotates nothing
# (``index_rope_dim`` 0). One layer, in order:
#
# * projections: ``[q_a | kv_a]`` (``cublas_mm``), the two RMSNorms (overlapped on graph steps),
#   ``q_b`` (``rubin_linear``); the indexer's ``wq_b`` (``cublas_mm``) and FP32 ``[wk | weights_proj]``
#   with TF32 (``rubin_linear``; on prefill steps its input widened by ``rubin_upcast_fp32``), the key
#   LayerNorm, the two FP8 cat-quantizations (overlapped on graph steps) and the weight scaling;
# * the index keys of every live row of the step into the cache, once (``indexer_k_cache_scatter_op``);
#   on decode graph steps the indexer's work on the layer input alone (the widening,
#   ``[wk | weights_proj]``, the key norm and quantization, this insert) runs on ``rt.idx_stream``
#   from the top of the layer, and q_b on ``rt.fork_stream`` beside the indexer's query projection
#   (dsa_idx_fork);
# * context rows: per prefill chunk, the paged K gather, the FP8 MQA logits per query tile with
#   the DeepGEMM schedule the step's metadata built for the tile, the Top-K (GVR, or radix where a
#   window is no wider than the Top-K); or the dense Top-K of a short prompt;
# * generation rows: the FP8 DSL paged logits (one atom per request, L2 evict-first) into the
#   decode-logits arena, the GVR Top-K; or the dense Top-K;
# * the Top-K remap to latent-pool rows (Triton from 256 rows, C++ below);
# * the absorbed sparse MLA: W_UK (``bmm_out``), the RoPE kernel on generation rows, the sparse
#   attention op, W_UV (``bmm_out``). Both calls run at 128 heads: the query and its RoPE columns
#   are padded with zero heads (``rubin_pad_heads``), since TRTLLM-Gen's sparse MLA tiles a head
#   group above 64 heads in Q64 tiles, which must divide it, and the padding heads' outputs are
#   dropped. The context call gets its query in FP8 (``mla_ctx_fused_q_fp8``:
#   ``rubin_mla_ctx_q_fp8``, the RoPE kernel's FP8 columns) and the op skips its own quantization;
# * the output gate ``attn * sigmoid(g_proj(x))`` and ``o_proj`` (``rubin_linear``). On decode
#   graph steps ``g_proj(x)`` follows q_b on ``rt.fork_stream``, and the gate waits for g's own
#   event (dsa_gate_early).
# --------------------------------------------------------------------------------------------------

# MLA shape (_K31): per head 128 NoPE + 64 RoPE query / key columns, 128 values; the latent
# row is kv_lora + rope wide.
_MLA_Q_LORA = _K31["q_lora_rank"]
_MLA_KV_LORA = _K31["kv_lora_rank"]
_MLA_ROPE = _K31["qk_rope_head_dim"]
_MLA_NOPE = _K31["qk_nope_head_dim"]
_MLA_V = _K31["v_head_dim"]
_MLA_QK = _MLA_NOPE + _MLA_ROPE
_MLA_LATENT = _MLA_KV_LORA + _MLA_ROPE
_MLA_EPS = _K31["rms_norm_eps"]  # q_a_layernorm, kv_a_layernorm
# dep16 (attention DP): all 96 query heads, also o_proj's input heads.
_MLA_HEADS = _MLA_QUERY_HEADS
_MLA_OUT_HEADS = _MLA_QUERY_HEADS
# The sparse MLA calls' head count (the 96 heads padded to 128); the padding buffers are cleared in
# 64-element rows.
_MLA_PAD_HEADS = 128
_MLA_PAD_ROW = 64
# AttentionInputType of the attention op's two calls.
_MLA_CONTEXT = 1
_MLA_GENERATION = 2

# The DSA indexer (_K31_DSA): 32 heads x 128, position-free, Top-2048.
_IDX_HEADS = _K31_DSA["index_num_heads"]
_IDX_DIM = _K31_DSA["index_head_dim"]
_IDX_HALF = _IDX_DIM // 2  # the cat-quantization takes two halves of each head
_IDX_TOPK = _K31_DSA["index_topk"]
_IDX_K_NORM_EPS = _K31["rms_norm_eps"]  # index_k_norm_eps (_check_build ties them)
# softmax scale x heads^-0.5, folded with the per-head query scale into the logits weights.
_IDX_WEIGHT_SCALE = _IDX_DIM**-0.5 * _IDX_HEADS**-0.5
#: Logged once per process by the first CUDA graph decode step: the indexer's keys and weights ran
#: on the idx stream, beside ``[q_a | kv_a]`` (dsa_idx_fork).
_IDX_FORK_LINE = (
    "Kimi K3.1 Rubin DSA: indexer key / weights prologue on its own stream beside [q_a | kv_a]"
)
#: Logged once per process by the first CUDA graph decode step: the logits weights took the
#: query scales and the constant in one kernel (``rubin_idx_weights``).
_IDX_SHORT_LINE = "Kimi K3.1 Rubin DSA indexer decode: (w * qs) * scale in one kernel"
#: Logged once per process by the first CUDA graph decode step: ``g_proj(x)`` was issued on the
#: fork stream after q_b, off the path from the attention to ``o_proj`` (dsa_gate_early).
_MLA_GATE_FORK_LINE = "Kimi K3.1 Rubin MLA: output gate projection on the fork stream after q_b"
#: Logged once per process by the first graph decode step under dsa_rope_before_idx_join.
_DSA_ROPE_FIRST_LINE = (
    "Kimi K3.1 Rubin MLA: W_UK and the RoPE kernel before the indexer join "
    "(dsa_rope_before_idx_join)"
)
# The decode logits arena's row stride is a multiple of the DSL kernel's 256-key split.
_IDX_LOGITS_ALIGN = 256
# The decode logits run one atom per request: next_n x index heads within one 256-wide MMA tile.
_IDX_DSL_MAX_ATOM_N = 256
# The Top-K remap runs the Triton kernel from this many rows on (the C++ op below it).
_IDX_REMAP_TRITON_ROWS = 256
# The radix Top-K's split workspace: blocks per row.
_IDX_RADIX_BLOCKS = 10

# The attention op's layer constants for a K3.1 DSA layer: absorbed MLA over an FP8 latent cache
# (quant_mode 128), one KV head of 576, the identity RoPE table (rope_gpt_neox, 64 dims;
# ``_identity_rope_table``), values kv_lora wide. Unset KV scales read as 1.0 in the MLA kernels
# (the unit scales of the FP8 cache); the MLA path reads only the cos / sin table, no inverse
# frequencies. Skip-softmax, spcompress and sage attention are off. The query's FP8 scale
# (quant_scale_qkv) is each call's own: the context call's FP8 query form passes one.
_MLA_ATTN = dict(
    k=None,
    v=None,
    output_sf=None,
    out_scale=None,
    kv_scale_orig_quant=None,
    kv_scale_quant_orig=None,
    rotary_inv_freq=None,
    attention_sinks=None,
    is_fused_qkv=False,
    update_kv_cache=True,
    predicted_tokens_per_seq=1,
    num_heads=_MLA_HEADS,
    num_kv_heads=1,
    head_size=_MLA_LATENT,
    mask_type=1,
    quant_mode=128,
    q_scaling=1.0,
    position_embedding_type=2,
    rope_dim=_MLA_ROPE,
    rope_base=10000.0,
    rope_scale_type=0,
    rope_scale=1.0,
    rope_short_m_scale=1.0,
    rope_long_m_scale=1.0,
    is_mla_enable=True,
    chunked_prefill_buffer_batch_size=1,
    q_lora_rank=_MLA_Q_LORA,
    kv_lora_rank=_MLA_KV_LORA,
    qk_nope_head_dim=_MLA_NOPE,
    qk_rope_head_dim=_MLA_ROPE,
    v_head_dim=_MLA_KV_LORA,
    rope_append=True,
    mrope_rotary_cos_sin=None,
    mrope_position_deltas=None,
    attention_chunk_size=None,
    skip_softmax_stat=None,
    skip_correction_threshold=0.0,
    sparse_kv_indices=None,
    sparse_kv_offsets=None,
    sparse_attn_offsets=None,
    sparse_attn_indices_block_size=1,
    sparse_attn_kv_lens=None,
    aux_kv_cache_pool_ptr=None,
    skip_softmax_threshold_scale_factor_prefill=0.0,
    skip_softmax_threshold_scale_factor_decode=0.0,
    sage_attn_num_elts_per_blk_q=0,
    sage_attn_num_elts_per_blk_k=0,
    sage_attn_num_elts_per_blk_v=0,
    sage_attn_qk_int8=False,
    dsv4_inv_rope_cos_sin_cache=None,
    enable_dsv4_epilogue_fusion=False,
    kv_norm_weight=None,
    kv_norm_eps=1e-6,
    cross_kv=None,
    relative_attention_bias=None,
    relative_attention_max_distance=0,
    # MLA presents its query separately: the context FMHA is not paged, whatever the step's
    # metadata says.
    use_paged_context_fmha=False,
)
# The attention op's calls at 128 heads.
_MLA_ATTN_PAD128 = dict(_MLA_ATTN, num_heads=_MLA_PAD_HEADS)
# The RoPE kernel's layer constants (the same layer, the same table).
_MLA_ROPE_GEN = dict(
    kv_scale_orig_quant=None,
    kv_scale_quant_orig=None,
    kv_cache_scale_orig_quant=None,
    out_scale=None,
    predicted_tokens_per_seq=1,
    num_heads=_MLA_HEADS,
    num_kv_heads=1,
    head_size=_MLA_LATENT,
    quant_mode=128,
    q_scaling=1.0,
    q_lora_rank=_MLA_Q_LORA,
    kv_lora_rank=_MLA_KV_LORA,
    qk_nope_head_dim=_MLA_NOPE,
    qk_rope_head_dim=_MLA_ROPE,
    v_head_dim=_MLA_KV_LORA,
    rope_append=True,
)
# The RoPE kernel at the attention op's 128 heads (the generation call).
_MLA_ROPE_GEN_PAD128 = dict(_MLA_ROPE_GEN, num_heads=_MLA_PAD_HEADS)


def _mla_project(L: _MlaW, x: torch.Tensor, st: _Step, rt: _Rt) -> tuple:
    """``[q_a | kv_a]`` and the two RMSNorms on every row of ``x``: (normed q_a, latent row).

    On graph steps kv_a's norm runs on the side stream beside q_a's.
    """
    q_a, ckv, k_pe = cublas_mm(x, L.qkv_a_t).split([_MLA_Q_LORA, _MLA_KV_LORA, _MLA_ROPE], -1)
    if st.cuda_graph:
        fork, join = rt.events[_EV_ATTN]
        main = torch.cuda.current_stream()
        fork.record()
        qr = flashinfer_rmsnorm(q_a, L.q_a_norm, _MLA_EPS)
        with torch.cuda.stream(rt.aux_stream):
            fork.wait()
            ckv = flashinfer_rmsnorm(ckv, L.kv_a_norm, _MLA_EPS)
            join.record()
        join.wait()
        if not torch.cuda.is_current_stream_capturing():
            ckv.record_stream(main)
    else:
        qr = flashinfer_rmsnorm(q_a, L.q_a_norm, _MLA_EPS)
        ckv = flashinfer_rmsnorm(ckv, L.kv_a_norm, _MLA_EPS)
    return qr, torch.cat([ckv, k_pe], -1)


def _widen_fp32(x: torch.Tensor) -> torch.Tensor:
    """``x.float()``, the same bits: ``rubin_upcast_fp32`` on a contiguous ``x``, else the cast.

    ``rubin_upcast_fp32`` moves 16 bytes per access where torch's casting copy moves one element.
    Its users: the indexer's TF32 projection on a prefill step (a ``[rows, 7168]`` input, the
    largest widening of the layer) and on a decode graph step (``_idx_keys``), and a generation
    server's FP32 router (``_moe_router_input``).
    """
    return rubin_upcast_fp32(x) if x.is_contiguous() else x.float()


def _idx_project(L: _MlaW, x: torch.Tensor, qr: torch.Tensor, st: _Step, rt: _Rt, xf=None) -> tuple:
    """The indexer's FP8 queries, FP8 keys with their FP32 scales, and FP32 logits weights.

    Shapes ``[rows, 32, 128]``, ``[rows, 128]`` and ``[rows, 32]``, on every row of ``x``.
    Position-free: each head is quantized as its two halves; nothing rotates. On graph steps the
    key quantization runs on the side stream beside the query's. ``xf``: ``x`` in FP32 (the TF32
    projection's input) when the caller widened it, else ``x.float()`` here.
    """
    xf = x.float() if xf is None else xf
    ik, w = rubin_linear(xf, L.idx_wk_wp, True).split([_IDX_DIM, _IDX_HEADS], -1)
    ik = ik.to(x.dtype)
    iq = cublas_mm(qr, L.idx_wq_b_t).view(-1, _IDX_HEADS, _IDX_DIM)
    ik = index_key_norm(ik, L.idx_k_norm_w, L.idx_k_norm_b, _IDX_K_NORM_EPS)
    iq_a, iq_b = iq.split([_IDX_HALF, _IDX_DIM - _IDX_HALF], -1)
    ik_a, ik_b = ik.split([_IDX_HALF, _IDX_DIM - _IDX_HALF], -1)
    if st.cuda_graph:
        fork, join = rt.events[_EV_ATTN]
        main = torch.cuda.current_stream()
        fork.record()
        q8, qs = fused_cat_fp8(iq_a, iq_b)
        with torch.cuda.stream(rt.aux_stream):
            fork.wait()
            k8, ks = fused_cat_fp8(ik_a, ik_b)
            join.record()
        join.wait()
        if not torch.cuda.is_current_stream_capturing():
            k8.record_stream(main)
            ks.record_stream(main)
    else:
        q8, qs = fused_cat_fp8(iq_a, iq_b)
        k8, ks = fused_cat_fp8(ik_a, ik_b)
    qs = qs.view(-1, _IDX_HEADS, 1)
    w = w * qs.squeeze(-1) * _IDX_WEIGHT_SCALE
    return q8.view(-1, _IDX_HEADS, _IDX_DIM), k8, ks, w


def _idx_keys(L: _MlaW, x: torch.Tensor) -> tuple:
    """The indexer's work on ``x`` alone: FP8 keys ``[rows, 128]`` with their FP32 scales, and the
    FP32 logits weights before the query scales ``[rows, 32]``, on every row of ``x``.

    ``_idx_project``'s key half, kernel for kernel, with ``x`` widened by ``_widen_fp32`` (the bits
    of ``x.float()``). On graph steps ``mla_generate`` runs it on ``rt.idx_stream``
    (dsa_idx_fork).
    """
    ik, w = rubin_linear(_widen_fp32(x), L.idx_wk_wp, True).split([_IDX_DIM, _IDX_HEADS], -1)
    ik = ik.to(x.dtype)
    ik = index_key_norm(ik, L.idx_k_norm_w, L.idx_k_norm_b, _IDX_K_NORM_EPS)
    ik_a, ik_b = ik.split([_IDX_HALF, _IDX_DIM - _IDX_HALF], -1)
    k8, ks = fused_cat_fp8(ik_a, ik_b)
    return k8, ks, w


def _idx_queries(L: _MlaW, qr: torch.Tensor) -> tuple:
    """The indexer's FP8 queries ``[rows, 32, 128]`` and their per-head FP32 scales ``[rows, 32]``.

    ``_idx_project``'s query half, kernel for kernel. Position-free: each head is quantized as its
    two halves; nothing rotates.
    """
    iq = cublas_mm(qr, L.idx_wq_b_t).view(-1, _IDX_HEADS, _IDX_DIM)
    iq_a, iq_b = iq.split([_IDX_HALF, _IDX_DIM - _IDX_HALF], -1)
    q8, qs = fused_cat_fp8(iq_a, iq_b)
    return q8.view(-1, _IDX_HEADS, _IDX_DIM), qs.view(-1, _IDX_HEADS)


def _idx_insert(i: int, k8: torch.Tensor, ks: torch.Tensor, n: int, st: _Step, rt: _Rt) -> None:
    """Insert the index keys of the step's ``n`` live rows into layer ``i``'s cache.

    Once a step, before the phases split (context chunk gathers read the keys just written).
    """
    indexer_k_cache_scatter_op(
        k8, ks, rt.dsa_pools.idx_k_cache[i], st.dsa.slot_fp8, st.dsa.slot_scale, n
    )


def _idx_prefill_topk(q, w, out, ks, ke, k8, ks8, meta) -> None:
    """One query tile's prefill logits (FP8 MQA over the chunk's keys) and its Top-K into ``out``.

    ``meta`` is DeepGEMM's schedule of the tile, which the step's metadata built (None: the call
    runs without one, with the same logits). A window no wider than the Top-K takes the radix
    kernel, as do logits outside the GVR engine's format (the engine ``TopK`` module makes the
    same choice per call).
    """
    if q.shape[0] == 0 or k8.shape[0] == 0:
        out.fill_(-1)
        return
    logits = fp8_mqa_logits(q, k8, ks8.reshape(-1), w, ks, ke, schedule_meta=meta)
    if logits.shape[1] <= _IDX_TOPK or not (
        logits.stride(1) == 1 and logits.stride(0) % 4 == 0 and logits.data_ptr() % 16 == 0
    ):
        indexer_topk_prefill(logits, ks, ke, out, _IDX_TOPK)
    else:
        rubin_gvr_topk_prefill(logits, ks, ke, out)


def _idx_prefill(i: int, q8, w, k8, ks, st: _Step, rt: _Rt) -> torch.Tensor:
    """The context rows' request-local Top-K ``[context rows, 2048]``.

    Per prefill chunk: gather the chunk's index keys (cached and new) from the cache, then score
    the chunk's queries in the query tiles the metadata planned under the per-call logits budget,
    each with the DeepGEMM schedule the metadata built for it once a step (every DSA layer reads
    the same). A short prompt takes the dense Top-K; a step the metadata did not chunk scores its
    own keys in one call, unscheduled.
    """
    d = st.dsa
    nct = st.num_ctx_tokens
    if d.skip_ctx:
        return d.topk_dense[:nct]
    topk = d.topk_out[:nct]
    if d.chunks is None:
        if d.num_ctx_kv_tokens == 0:
            topk.fill_(-1)
        else:
            kv = d.num_ctx_kv_tokens
            _idx_prefill_topk(q8[:nct], w[:nct], topk, d.cu_ks, d.cu_ke, k8[:kv], ks[:kv], None)
        return topk
    k_cache = rt.dsa_pools.idx_k_cache[i]
    for chunk in d.chunks:
        t0, t1 = chunk.token_start, chunk.token_end
        if chunk.k_token_start >= chunk.k_token_end:
            topk[t0:t1].fill_(-1)
            continue
        keys = chunk.k_token_end - chunk.k_token_start
        ck8, cks = k31_gather_k_cache_paged(
            k_cache, d.k_slot_fp8, d.k_slot_scale, chunk.k_token_start, keys, _IDX_DIM
        )
        metas = chunk.mqa_meta or (None,) * len(chunk.tiles)
        for (c0, c1), meta in zip(chunk.tiles, metas, strict=True):
            _idx_prefill_topk(
                q8[t0 + c0 : t0 + c1],
                w[t0 + c0 : t0 + c1],
                topk[t0 + c0 : t0 + c1],
                chunk.cu_seqlen_ks[c0:c1],
                chunk.cu_seqlen_ke[c0:c1],
                ck8,
                cks,
                meta,
            )
    return topk


def _idx_logits_arena(st: _Step, rt: _Rt) -> torch.Tensor:
    """The decode logits arena ``[max rows, align256(max_len)]`` FP32, kept in ``rt.dsa_pools``.

    Allocated by the first forward with generation rows, never under capture: a captured graph
    keeps its address. A new cache manager drops it (``rt.dsa_pools.bind``).
    """
    arena = rt.dsa_pools.idx_logits
    if arena is None:
        assert not torch.cuda.is_current_stream_capturing(), (
            "the decode logits arena must exist before the first CUDA graph capture"
        )
        md = st.md
        requests = rt.dsa_pools.gen_requests or md.max_num_requests
        rows = requests * (int(md.max_total_draft_tokens or 0) + 1)
        width = -(-st.dsa.max_len // _IDX_LOGITS_ALIGN) * _IDX_LOGITS_ALIGN
        arena = torch.empty(rows, width, dtype=torch.float32, device=st.position_ids.device)
        rt.dsa_pools.idx_logits = arena
    return arena


def _idx_decode(i: int, q8, w, st: _Step, rt: _Rt) -> torch.Tensor:
    """The generation rows' request-local Top-K ``[generation rows, 2048]``.

    The FP8 DSL paged logits score each request's verify window as one atom (a window of more than
    one row with the L2 evict-first hint), into the arena; then the GVR Top-K (radix where a
    single row's width is not whole float4s). Dense Top-K rows where the step skips the indexer.
    """
    d = st.dsa
    nct, n = st.num_ctx_tokens, st.num_tokens
    if d.skip_gen:
        return d.topk_dense[nct:n]
    rows, next_n, ng = n - nct, d.next_n, st.num_generations
    assert next_n * _IDX_HEADS <= _IDX_DSL_MAX_ATOM_N, "a verify window wider than one logits atom"
    arena = _idx_logits_arena(st, rt)
    assert rows <= arena.shape[0], (
        f"{rows} generation rows, the logits arena holds {arena.shape[0]}"
    )
    out = arena[:rows, : d.max_len]
    logits = rubin_fp8_paged_mqa_logits(
        q8[nct:n].view(ng, next_n, _IDX_HEADS, _IDX_DIM),
        rt.dsa_pools.idx_k_cache[i],
        w[nct:n],
        d.ctx_lens,
        d.block_table,
        d.sched,
        d.max_len,
        next_n > 1,
        out,
        pdl=rt.idx_logits_pdl,
    )
    topk = d.topk_out[nct:n]
    if rows > 1 or d.max_len % 4 == 0:
        rubin_gvr_topk_decode(logits, d.gen_kv_lens, topk, next_n, d.max_len)
    else:
        indexer_topk_decode(
            logits,
            d.gen_kv_lens,
            topk,
            next_n,
            _IDX_TOPK,
            1,
            d.radix_aux_indices[:rows, :_IDX_RADIX_BLOCKS, :_IDX_TOPK],
            d.radix_aux_logits[:rows, :_IDX_RADIX_BLOCKS, :_IDX_TOPK],
            None,
        )
    return topk


def _idx_remap(
    i: int, topk: torch.Tensor, req_idx, block_table, st: _Step, rt: _Rt
) -> torch.Tensor:
    """Request-local Top-K columns to latent-pool rows of layer ``i`` (the sparse MLA's indices)."""
    scale, layer_offset = rt.dsa_pools.pool_params[i]
    tpb = st.dsa.tokens_per_block
    if topk.shape[0] >= _IDX_REMAP_TRITON_ROWS:
        return rubin_dsa_topk_remap(
            req_idx, block_table, topk, tpb, _IDX_TOPK, scale * tpb, layer_offset
        )
    return convert_req_index_to_global(
        req_idx, block_table, topk, tpb, _IDX_TOPK, scale * tpb, layer_offset
    )


def _mla_expand(v_b_t: torch.Tensor, latent: torch.Tensor, out: torch.Tensor) -> None:
    """``out[t, h] = latent[t, h] @ W_UV[h]``: the latent attention output to the value heads."""
    bmm_out(latent.transpose(0, 1), v_b_t, out.view(-1, latent.shape[1], _MLA_V).transpose(0, 1))


def _mla_window(i: int, st: _Step, rt: _Rt) -> int:
    """The attention window of layer ``i``, read as the engine's attention backend reads it.

    The KV cache manager V2 holds a full-context layer's window as None (only a sliding window is
    a number); the backend then takes the metadata's ``max_seq_len`` (``TrtllmAttention.forward``).
    The DSA layers have no sliding window, so every server gets ``max_seq_len`` here.
    """
    window = st.cache.max_attention_window_vec[rt.dsa_pools.local[i]]
    return int(st.md.max_seq_len) if window is None else int(window)


def _mla_absorb(L: _MlaW, q: torch.Tensor) -> tuple:
    """The generation call's query at 128 heads: (the BF16 query with W_UK in its latent columns,
    its RoPE columns), the padding heads zero in both. The RoPE kernel rotates the RoPE columns of
    all 128 heads into the FP8 copy the attention reads, beside the latent columns it copies there;
    over an FP8 or NVFP4 cache it leaves the BF16 query's RoPE columns unwritten.
    """
    rows = q.shape[0]
    q_nope, q_pe = q.view(rows, _MLA_HEADS, _MLA_QK).split([_MLA_NOPE, _MLA_ROPE], -1)
    q_pe_pad = q.new_empty(rows, _MLA_PAD_HEADS, _MLA_ROPE)
    rubin_pad_heads(q_pe_pad, q_pe, _MLA_HEADS)
    fused_q = q.new_empty(rows, _MLA_PAD_HEADS, _MLA_LATENT)
    rubin_pad_heads(
        fused_q.view(rows, _MLA_PAD_HEADS * _MLA_LATENT // _MLA_PAD_ROW, _MLA_PAD_ROW),
        None,
        _MLA_HEADS * _MLA_LATENT // _MLA_PAD_ROW,
    )
    bmm_out(q_nope.transpose(0, 1), L.k_b, fused_q[:, :_MLA_HEADS, :_MLA_KV_LORA].transpose(0, 1))
    return fused_q, q_pe_pad


def _mla_ctx_query(L: _MlaW, q_nope: torch.Tensor, rows: int) -> tuple:
    """The context call's query at 128 heads: (the BF16 query the op takes, the FP8 query).

    W_UK writes the real heads' latent columns of a BF16 buffer, converted to FP8 before the op
    (``_mla_ctx_query_fp8``).
    """
    fused_q = q_nope.new_empty(rows, _MLA_PAD_HEADS, _MLA_LATENT)
    return _mla_ctx_query_fp8(L, q_nope, fused_q)


def _mla_ctx_query_fp8(L: _MlaW, q_nope: torch.Tensor, fused_q: torch.Tensor) -> tuple:
    """The FP8 query, made before the op instead of by it: (a one-row stand-in, the FP8 query).

    W_UK writes its BF16 product into ``fused_q``, and ``rubin_mla_ctx_q_fp8`` converts those
    latent columns the way the op converts its BF16 query (zeros on the padding heads). Given the
    FP8 buffer and its unit scale, the op's RoPE kernel writes the RoPE columns in FP8 and the op
    skips its quantization and reads no BF16 query: one row stands in for the query's shape, and
    the BF16 buffer is free before the Top-K.
    """
    rows = fused_q.shape[0]
    bmm_out(q_nope.transpose(0, 1), L.k_b, fused_q[:, :_MLA_HEADS, :_MLA_KV_LORA].transpose(0, 1))
    quant_q = torch.empty(
        rows, _MLA_PAD_HEADS, _MLA_LATENT, dtype=torch.float8_e4m3fn, device=fused_q.device
    )
    rubin_mla_ctx_q_fp8(quant_q[..., :_MLA_KV_LORA], fused_q[..., :_MLA_KV_LORA], _MLA_HEADS)
    return fused_q.new_empty(1, _MLA_PAD_HEADS * _MLA_LATENT).expand(rows, -1), quant_q


def _mla_context(L, i, q, latent, q8, w, k8, ks, out, st: _Step, rt: _Rt) -> None:
    """Sparse MLA of the context rows at 128 heads, then W_UV of the 96 real heads into ``out``.

    The RoPE columns get zero padding heads and the query its latent columns (``_mla_ctx_query``),
    then the indexer's prefill Top-K, its remap and the attention op (absorbed FP8 query; the op
    applies RoPE and appends the latent KV).
    """
    rows = q.shape[0]
    d = st.dsa
    q_nope, q_pe = q.view(rows, _MLA_HEADS, _MLA_QK).split([_MLA_NOPE, _MLA_ROPE], -1)
    q_pe_pad = q.new_empty(rows, _MLA_PAD_HEADS, _MLA_ROPE)
    rubin_pad_heads(q_pe_pad, q_pe, _MLA_HEADS)
    fused_q, quant_q = _mla_ctx_query(L, q_nope, rows)
    topk = _idx_prefill(i, q8, w, k8, ks, st, rt)
    indices = _idx_remap(i, topk, d.req_idx_ctx, d.block_table_ctx, st, rt)
    # The FP8 query's scale, applied to the RoPE kernel's FP8 columns: 1, as the op's own
    # quantization of a BF16 query applies it.
    scale = torch.ones(1, dtype=torch.float32, device=q.device)
    lat = thop_attention(
        q=fused_q.view(rows, -1),
        output=q.new_empty(rows, _MLA_PAD_HEADS * _MLA_KV_LORA),
        latent_cache=latent,
        q_pe=q_pe_pad,
        rotary_cos_sin=rt.mla_rope,
        local_layer_idx=rt.dsa_pools.local[i],
        attention_input_type=_MLA_CONTEXT,
        attention_window_size=_mla_window(i, st, rt),
        rope_max_positions=rt.dsa_pools.rope_positions,
        rope_original_max_positions=rt.dsa_pools.rope_positions,
        softmax_stats_tensor=None,
        cu_q_seqlens=st.md.cu_q_seqlens,
        cu_kv_seqlens=st.md.cu_kv_seqlens,
        fmha_scheduler_counter=None,
        mla_bmm1_scale=None,
        mla_bmm2_scale=None,
        quant_q_buffer=quant_q,
        quant_scale_qkv=scale,
        sparse_attn_indices=indices,
        **{**d.sparse_ctx, **_MLA_ATTN_PAD128},
    )
    _mla_expand(L.v_b_t, lat.view(rows, _MLA_PAD_HEADS, _MLA_KV_LORA)[:, :_MLA_HEADS], out)


class _MlaQuery(NamedTuple):
    """The generation rows' query between the RoPE kernel and the attention op
    (``_mla_gen_query``)."""

    fused_q: torch.Tensor  # [rows, heads, latent]: W_UK's columns, then the RoPE kernel's
    q_pe: torch.Tensor
    cu_q: torch.Tensor
    cu_kv: torch.Tensor
    counter: torch.Tensor
    bmm1: torch.Tensor
    bmm2: torch.Tensor
    quant_q: torch.Tensor
    local: int


def _mla_gen_query(L, i, q, latent, st: _Step, rt: _Rt) -> _MlaQuery:
    """W_UK into the query buffer at 128 heads (``_mla_absorb``), then the RoPE kernel (FP8 query,
    latent KV append, the decode's scheduler buffers), in order: over an FP8 cache it reads the
    BMM's half. They read ``q`` and the latent rows only, nothing the indexer writes."""
    rows, dev = q.shape[0], q.device
    d = st.dsa
    seqs = st.num_contexts + st.num_generations + 1
    cu_q = torch.empty(seqs, dtype=torch.int32, device=dev)
    cu_kv = torch.empty(seqs, dtype=torch.int32, device=dev)
    counter = torch.empty(1, dtype=torch.uint32, device=dev)
    bmm1 = torch.empty(2, dtype=torch.float32, device=dev)
    bmm2 = torch.empty(1, dtype=torch.float32, device=dev)
    quant_q = torch.empty(rows, _MLA_PAD_HEADS, _MLA_LATENT, dtype=torch.uint8, device=dev)
    fused_q, q_pe = _mla_absorb(L, q)
    local = rt.dsa_pools.local[i]
    mla_rope_generation(
        fused_q=fused_q,
        q_pe=q_pe,
        latent_cache=latent,
        rotary_cos_sin=rt.mla_rope,
        cu_q_seqlens=cu_q,
        cu_kv_seqlens=cu_kv,
        fmha_scheduler_counter=counter,
        mla_bmm1_scale=bmm1,
        mla_bmm2_scale=bmm2,
        quant_q_buffer=quant_q,
        layer_idx=local,
        **d.rope_gen,
        **_MLA_ROPE_GEN_PAD128,
    )
    return _MlaQuery(fused_q, q_pe, cu_q, cu_kv, counter, bmm1, bmm2, quant_q, local)


def _mla_generation(L, i, q, latent, q8, w, out, st: _Step, rt: _Rt, mq=None) -> None:
    """Absorbed sparse MLA decode of the generation rows, then W_UV into ``out``.

    W_UK into the query buffer and the RoPE kernel (``_mla_gen_query``; ``mq``: their result, when
    the caller ran them before, dsa_rope_before_idx_join), then the indexer's decode Top-K, its
    remap and the attention op; W_UV reads the 96 real heads.
    """
    rows = q.shape[0]
    d = st.dsa
    if mq is None:
        mq = _mla_gen_query(L, i, q, latent, st, rt)
    topk = _idx_decode(i, q8, w, st, rt)
    indices = _idx_remap(i, topk, d.req_idx_gen, d.block_table_gen, st, rt)
    lat = thop_attention(
        q=mq.fused_q.view(rows, -1),
        output=q.new_empty(rows, _MLA_PAD_HEADS * _MLA_KV_LORA),
        latent_cache=latent,
        q_pe=mq.q_pe,
        rotary_cos_sin=rt.mla_rope,
        local_layer_idx=mq.local,
        attention_input_type=_MLA_GENERATION,
        attention_window_size=_mla_window(i, st, rt),
        rope_max_positions=rt.dsa_pools.rope_positions,
        rope_original_max_positions=rt.dsa_pools.rope_positions,
        softmax_stats_tensor=None,
        cu_q_seqlens=mq.cu_q,
        cu_kv_seqlens=mq.cu_kv,
        fmha_scheduler_counter=mq.counter,
        mla_bmm1_scale=mq.bmm1,
        mla_bmm2_scale=mq.bmm2,
        quant_q_buffer=mq.quant_q,
        quant_scale_qkv=None,
        sparse_attn_indices=indices,
        **{**d.sparse_gen, **_MLA_ATTN_PAD128},
    )
    _mla_expand(L.v_b_t, lat.view(rows, _MLA_PAD_HEADS, _MLA_KV_LORA)[:, :_MLA_HEADS], out)


def _mla_output(
    L: _MlaW, x: torch.Tensor, out: torch.Tensor, n: int, gate: Optional[tuple] = None
) -> torch.Tensor:
    """Zero the padding rows, gate with ``sigmoid(g_proj(x))`` and project: the o_proj partial.

    ``gate``: ``(g_proj(x), the event recorded after it)`` from the fork stream (decode graph
    steps, dsa_gate_early); the gate waits for that event right before it reads g. None:
    ``g_proj(x)`` runs here.
    """
    if out.shape[0] > n:
        out[n:].zero_()
    if gate is None:
        g = rubin_linear(x, L.g, False)
    else:
        g, ready = gate
        ready.wait()
    gated = k31_sigmoid_gate_mul(out, g)
    return rubin_linear(gated, L.o, False)


def _mla_prefill_witness(x: torch.Tensor) -> None:
    """Log once per rank, at its first prefill DSA layer, the query form and widening that ran.

    A run's evidence of the two prefill levers on every rank: the context call's query is FP8
    (the attention op takes the caller's FP8 query and skips its own quantization); the indexer's
    input is widened by ``rubin_upcast_fp32`` when it is contiguous (``_widen_fp32``), else by
    ``x.float()``. The logger keeps the key, so the later calls of the rank write nothing. Log only.
    """
    widen = "rubin_upcast_fp32" if x.is_contiguous() else "x.float()"
    logger.info_once(
        f"Kimi K3.1 Rubin prefill levers: {_TARGET}: context query FP8 (mla_ctx_fused_q_fp8), "
        f"indexer input widened by {widen}",
        key=f"{_TARGET} prefill levers",
    )


def mla_prefill(L: _MlaW, i: int, x: torch.Tensor, st: _Step, rt: _Rt) -> torch.Tensor:
    """Layer ``i``'s MLA + DSA over every row of a step with context rows; ``[rows, hidden]``.

    One call per layer: the projections and the index-key insert cover the whole batch, the
    context rows' attention reads the keys just written, and a mixed step's generation rows run
    the decode path after them. A rank's first call logs the forms its context rows took.
    """
    if x.shape[0] < st.num_tokens:
        raise ValueError(
            "mla_prefill takes every row of the step, context rows first: the index-K insert and "
            "the context Top-K cover the whole batch"
        )
    n, c = st.num_tokens, st.num_ctx_tokens
    qr, latent = _mla_project(L, x, st, rt)
    q = rubin_linear(qr, L.q_b, False)
    q8, k8, ks, w = _idx_project(L, x, qr, st, rt, _widen_fp32(x))
    _idx_insert(i, k8[:n], ks[:n], n, st, rt)
    out = x.new_empty(x.shape[0], _MLA_OUT_HEADS * _MLA_V)
    _mla_context(L, i, q[:c], latent[:c], q8, w, k8, ks, out[:c], st, rt)
    _mla_prefill_witness(x)
    if st.num_generations:
        _mla_generation(L, i, q[c:n], latent[c:n], q8, w, out[c:n], st, rt)
    return _mla_output(L, x, out, n)


def mla_generate(L: _MlaW, i: int, x: torch.Tensor, st: _Step, rt: _Rt) -> torch.Tensor:
    """Layer ``i``'s MLA + DSA over the rows of a decode step; ``[rows, hidden]``.

    On graph steps the indexer's work on ``x`` alone (``_idx_keys`` and the index-key insert) runs
    on ``rt.idx_stream`` from the top, beside ``[q_a | kv_a]`` and its norms (dsa_idx_fork), and
    q_b runs on ``rt.fork_stream`` once the latent row is built, beside the indexer's query
    projection (``_idx_queries``). The current stream joins the idx stream right before the logits
    weights take the query scales, the first read of anything the idx stream wrote, and the fork
    stream right before W_UK reads q. ``g_proj(x)`` follows q_b on the fork stream; the gate waits
    for g's own event (dsa_gate_early). Same kernels on the same inputs. Eager steps run every
    kernel on the current stream, in today's order.

    A generation server's graph steps (``rt.levers.dsa_rope_before_idx_join``) issue W_UK and the
    RoPE kernel, which read q and the latent rows only, before the join on the idx stream; the
    logits weights, the indexer and the attention after it.
    """
    n = st.num_tokens
    if not st.cuda_graph:
        qr, latent = _mla_project(L, x, st, rt)
        q = rubin_linear(qr, L.q_b, False)
        q8, k8, ks, w = _idx_project(L, x, qr, st, rt)
        _idx_insert(i, k8[:n], ks[:n], n, st, rt)
        out = x.new_empty(x.shape[0], _MLA_OUT_HEADS * _MLA_V)
        _mla_generation(L, i, q[:n], latent[:n], q8, w, out[:n], st, rt)
        return _mla_output(L, x, out, n)
    main = torch.cuda.current_stream()
    # Everything the side streams write they allocate themselves, and this stream reads it only
    # after their events; the next layer's forks follow those reads on this stream.
    fork, join = rt.events[_EV_IDX]
    fork.record()
    with torch.cuda.stream(rt.idx_stream):
        fork.wait()
        k8, ks, w = _idx_keys(L, x)
        _idx_insert(i, k8[:n], ks[:n], n, st, rt)
        join.record()
    qr, latent = _mla_project(L, x, st, rt)
    # _mla_project's side-stream join is behind: its event pair is free again. g_proj(x) follows
    # q_b on the fork stream (dsa_gate_early); _mla_output waits for g_ready before the gate.
    q_ready, g_ready = rt.events[_EV_ATTN]
    rt.fork_stream.wait_stream(main)
    with torch.cuda.stream(rt.fork_stream):
        q = rubin_linear(qr, L.q_b, False)
        q_ready.record()
        g = rubin_linear(x, L.g, False)
        g_ready.record()
    q8, qs = _idx_queries(L, qr)
    if rt.levers.dsa_rope_before_idx_join:
        # W_UK and the RoPE kernel read q and the latent rows only: they run before the join,
        # beside the indexer's stream; the logits weights, the indexer and the attention after it.
        q_ready.wait()  # right before W_UK reads q
        if not torch.cuda.is_current_stream_capturing():
            q.record_stream(main)
            g.record_stream(main)
        out = x.new_empty(x.shape[0], _MLA_OUT_HEADS * _MLA_V)
        mq = _mla_gen_query(L, i, q[:n], latent[:n], st, rt)
        join.wait()  # right before the first reader of w and of the index keys
        if not torch.cuda.is_current_stream_capturing():
            w.record_stream(main)
        w = rubin_idx_weights(w, qs, _IDX_WEIGHT_SCALE)
        logger.info_once(_IDX_FORK_LINE, key=_IDX_FORK_LINE)
        logger.info_once(_IDX_SHORT_LINE, key=_IDX_SHORT_LINE)
        logger.info_once(_MLA_GATE_FORK_LINE, key=_MLA_GATE_FORK_LINE)
        logger.info_once(_DSA_ROPE_FIRST_LINE, key=_DSA_ROPE_FIRST_LINE)
        _mla_generation(L, i, q[:n], latent[:n], q8, w, out[:n], st, rt, mq)
        return _mla_output(L, x, out, n, (g, g_ready))
    join.wait()  # right before the first reader of w and of the index keys
    if not torch.cuda.is_current_stream_capturing():
        w.record_stream(main)
    w = rubin_idx_weights(w, qs, _IDX_WEIGHT_SCALE)
    q_ready.wait()  # right before W_UK reads q
    if not torch.cuda.is_current_stream_capturing():
        q.record_stream(main)
        g.record_stream(main)
    logger.info_once(_IDX_FORK_LINE, key=_IDX_FORK_LINE)
    logger.info_once(_IDX_SHORT_LINE, key=_IDX_SHORT_LINE)
    logger.info_once(_MLA_GATE_FORK_LINE, key=_MLA_GATE_FORK_LINE)
    out = x.new_empty(x.shape[0], _MLA_OUT_HEADS * _MLA_V)
    _mla_generation(L, i, q[:n], latent[:n], q8, w, out[:n], st, rt)
    return _mla_output(L, x, out, n, (g, g_ready))


# --------------------------------------------------------------------------------------------------
# MoE: the latent MoE of layers 1..92 and layer 0's dense MLP (dep16: attention DP 16, MoE EP16).
#
# Every rank holds its own requests' rows and the 56 experts of its EP window. A latent-MoE layer
# runs:
#
# * the router: FP32 logits (a context server: the BF16 gate through ``dsv3_router_gemm_op``; a
#   generation server: the FP32 gate through ``rubin_linear``, whose reduction order keeps the
#   borderline top-16 picks);
# * ``latent_down`` (BF16), ``noaux_tc_op``'s top-16 and the NVFP4 input quantization;
# * the experts of every rank, which return this rank's rows complete: on a context server the
#   all-to-all dispatch, the CuteDSL grouped GEMMs on this rank's experts (sort, FC1, FC2 with the
#   finalize scatter into the combine's payload, received rows zeroed beside FC1) and the all-to-all
#   combine; on a generation server MegaMoE, the exchange fused in, one launch per chunk of at most
#   ``rt.megamoe.chunk_tokens`` rows per rank (``_moe_megamoe_chunk``);
# * ``routed_expert_norm`` and ``latent_up`` (BF16), and the two shared experts, whole (one MXFP8
#   gate_up GEMM, ``situ_and_mul``, the MXFP8 down GEMM; no all-reduce). On a context server
#   (``shared_down_mx_handoff``) SiTU stores the MXFP8 form of its rows instead
#   (``rubin_situ_and_mul_mxfp8``: E4M3 codes and R128c4 scales, bit for bit the down projection's
#   own quantization of the BF16 rows) and the down GEMM multiplies those codes; no quantization runs.
#
# It returns ``_MoeOut(routed, shared)``, both whole; the next mix adds them to the residual.
#
# The expert path is the server role's, fixed at construction, and every step of a role takes it:
# a context server's decode steps (attention DP dummy rows) and a generation server's steps with
# context rows (aggregated serving) included.
#
# Steps run in order on the current stream, except these forks: the zero of the CuteDSL output's
# received rows on ``moe_comm.memset_stream`` beside FC1 (every step), and on a CUDA graph's steps
# (warmup and capture) of a generation server ``latent_down`` on ``moe_comm.latent_stream`` from
# the FP32 router's cast on, beside the router GEMM and the routing (on a one-chunk step of the
# bucket path the NVFP4 quantization into the bucket follows it there), and the shared experts on
# ``rt.aux_stream`` beside the routed chain. Same kernels, same bits.
# --------------------------------------------------------------------------------------------------

# K3.1's MoE scalars.
_MOE_NUM_EXPERTS = _K31["num_experts"]
_MOE_TOP_K = _K31["num_experts_per_token"]
_MOE_GROUPS = _K31["num_expert_group"]
_MOE_SCALE = _K31["routed_scaling_factor"]
_MOE_EPS = _K31["rms_norm_eps"]  # routed_expert_norm
_MOE_SITU = (_K31["activation_situ_beta"], _K31["activation_situ_linear_beta"])
# NVFP4 values per input block scale, and the CuteDSL experts' row tile.
_MOE_SF_VEC = 16
_MOE_TILE = 128
# The expert kernels' activation (ActivationType.SiTu).
_MOE_ACT_SITU = 9
# The all-to-all around the CuteDSL experts: the id of a slot that routes nowhere.
_MOE_A2A_INVALID_ID = -1
# A rank whose rows of a generation chunk number one of these takes the staging path
# (noaux_tc_op, fp4_quantize, then rubin_megamoe) although the bucket routing serves them. The
# guard reads this rank's rows only, whatever bucket the chunk launches; it was measured with
# 32 rows on every rank (a full 32-row bucket), where the bucket path's routing and
# quantization kernels take longer than staging.
_MOE_ROUTE_SKIP_M = frozenset({32})
# A generation server's CUDA graph steps: latent_down (and, on a one-chunk step of the bucket path,
# the quantization into the bucket) on the latent stream beside the FP32 router and the routing.
# Logged once per process, at the first such step.
_MOE_LATENT_BESIDE_LINE = (
    "Kimi K3.1 Rubin dep16 MoE front: latent_down and the bucket quantization on the latent "
    "stream beside the FP32 router and the routing"
)
# A generation server's FP32 router takes its input from rubin_upcast_fp32 (x.float()'s bits).
# Logged once per process, at the first such widening.
_MOE_ROUTER_WIDEN_LINE = (
    "Kimi K3.1 Rubin dep16 MoE front: router input widened by rubin_upcast_fp32"
)
# A context server's shared experts hand down SiTU's MXFP8 form (logged constant).
_MOE_CTX_SHARED_MX = _CONSTANTS["context"]["shared_down_mx_handoff"]


def _moe_require(role: str, key: str, *values) -> None:
    """Fail at import if the launch constant asks for a value this block does not run."""
    if _CONSTANTS[role][key] not in values:
        raise NotImplementedError(
            f"MoE block: {role} {key}={_CONSTANTS[role][key]!r}, runs {values}"
        )


_moe_require("context", "router_bf16", True)
_moe_require("context", "moe_backend", "cutedsl")
_moe_require("generation", "router_bf16", False)
_moe_require("generation", "moe_backend", "megamoe_cutedsl")
_moe_require("generation", "latent_bf16_gemm", "cublas_linear")
# The routing kernel that writes into MegaMoE's bucket computes noaux_tc's top-k of one group.
if _MOE_GROUPS != 1:
    raise NotImplementedError(
        f"MoE block: num_expert_group={_MOE_GROUPS}; the bucket routing runs one"
    )


def _moe_cublas_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return cublas_mm(x, w.t())


#: The BF16 latent_down / latent_up GEMM (``latent_bf16_gemm``).
_moe_latent_gemm = _moe_cublas_linear


def _moe_router(L: _MoeW, x: torch.Tensor) -> torch.Tensor:
    """FP32 logits of the BF16 router GEMM (FP32 accumulation): a context server."""
    return dsv3_router_gemm_op(x, L.router.t())


def _moe_router_input(x: torch.Tensor) -> torch.Tensor:
    """The FP32 router's input, ``x.float()``'s bits (``_widen_fp32``).

    A contiguous ``x`` is widened by ``rubin_upcast_fp32``, and the first such widening of a
    process logs one INFO line; any other ``x`` by ``x.float()``.
    """
    xf = _widen_fp32(x)
    if x.is_contiguous():
        logger.info_once(_MOE_ROUTER_WIDEN_LINE, key=_MOE_ROUTER_WIDEN_LINE)
    return xf


def _moe_router_fp32(L: _MoeW, x: torch.Tensor, xf: Optional[torch.Tensor] = None) -> torch.Tensor:
    """FP32 logits of the FP32 router (``w1_rows`` declares it FP32 on a generation server). ``xf``: ``x``'s
    FP32 copy when the mix stored it (lever router_f32), else ``_moe_router_input(x)``."""
    # FP32, TF32 off
    return rubin_linear(_moe_router_input(x) if xf is None else xf, L.router, False)


def _moe_route(L: _MoeW, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(topk_ids int32, topk_weights FP32)`` of the rows' logits: the top-16 experts."""
    weights, ids = noaux_tc_op(
        logits, L.router_bias, _MOE_GROUPS, _MOE_GROUPS, _MOE_TOP_K, _MOE_SCALE
    )
    return ids, weights


def _moe_quantize(L: _MoeW, latent_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """NVFP4 values and their linear (unswizzled) block scales ``[rows, latent / 16]``."""
    x4, sf = fp4_quantize(latent_in, L.fc31_in_scale, _MOE_SF_VEC, False, False)
    return x4, sf.view(latent_in.shape[0], latent_in.shape[1] // _MOE_SF_VEC)


def _moe_shared(L: _MoeW, x: torch.Tensor, gate_up_reserve: int = 0) -> torch.Tensor:
    """The shared experts, whole: gate and up in one GEMM, then SiTU.

    The gate_up GEMM keeps ``gate_up_reserve`` SMs out of its grid (a generation server's graph
    steps, where it runs beside MegaMoE's token exchange); its tactic and output are the same.
    """
    gate_up = rubin_mxfp8_linear(x, L.shared_gu_w, L.shared_gu_s, reserve_sms=gate_up_reserve)
    h = situ_and_mul(gate_up, *_MOE_SITU)
    return rubin_mxfp8_linear(h, L.shared_dn_w, L.shared_dn_s)


def _moe_shared_ctx(L: _MoeW, x: torch.Tensor) -> torch.Tensor:
    """A context server's shared experts: SiTU hands the down GEMM its MXFP8 operand.

    Under ``_MOE_CTX_SHARED_MX`` the activation's E4M3 codes and R128c4 scales (the down
    projection's own quantization of the BF16 rows, bit for bit) go straight into the GEMM. A rank
    without rows keeps ``_moe_shared``.
    """
    if not (_MOE_CTX_SHARED_MX and x.shape[0]):
        return _moe_shared(L, x)
    gu = rubin_mxfp8_linear(x, L.shared_gu_w, L.shared_gu_s)
    codes, scales = rubin_situ_and_mul_mxfp8(gu, *_MOE_SITU)
    return rubin_mxfp8_gemm(codes, L.shared_dn_w, scales, L.shared_dn_s)


def _moe_sort(L: _MoeW, ids: torch.Tensor, weights: torch.Tensor) -> tuple:
    """moe_sort's six tables of this rank's window at the expert kernels' row tile."""
    return tuple(
        rubin_moe_sort(
            ids, weights, _MOE_NUM_EXPERTS, _MOE_TOP_K, L.expert_offset, L.fc1_w.shape[0], _MOE_TILE
        )
    )


def _moe_cutedsl(L: _MoeW, x4, sf, ids, weights, out, st: _Step, rt: _Rt) -> torch.Tensor:
    """This rank's experts through the CuteDSL grouped GEMMs into ``out`` ``[rows, latent]`` BF16.

    The finalize scatter-adds into the rows of ``out`` whose routing has a pair of this rank's
    window, the received rows: the dispatch packs each source's tokens at the front of its slot
    block and routes the slots past them nowhere (ids -1), and the combine reads only the
    received rows. Those rows alone are zeroed, on the memset stream beside FC1 on every step
    (the other rows keep their contents); on an eager step ``out`` and ``ids`` are marked used
    there.
    """
    c = rt.moe_comm
    num_local = L.fc1_w.shape[0]
    t2e, t2l, e2p, p2e, _, n = _moe_sort(L, ids, weights)
    c.latent_fork.record()  # the fork point of the memset stream
    if not st.cuda_graph:
        out.record_stream(c.memset_stream)
        ids.record_stream(c.memset_stream)
    with torch.cuda.stream(c.memset_stream):
        c.latent_fork.wait()
        rubin_moe_zero_local_rows(out, ids, L.expert_offset, num_local)
        c.memset_ready.record()
    h, h_sf = rubin_cutedsl_fc1(
        x4,
        L.fc1_w,
        sf,
        L.fc1_s,
        L.fc31_alpha,
        t2e,
        t2l,
        p2e,
        n,
        L.fc2_in_scale,
        _MOE_NUM_EXPERTS,
        _MOE_TOP_K,
        num_local,
        L.expert_offset,
        _MOE_TILE,
        _MOE_ACT_SITU,
        *_MOE_SITU,
    )
    c.memset_ready.wait()
    rubin_cutedsl_fc2_finalize(
        h,
        L.fc2_w,
        h_sf,
        L.fc2_s,
        L.fc2_alpha,
        out,
        t2e,
        t2l,
        p2e,
        n,
        weights,
        _MOE_NUM_EXPERTS,
        _MOE_TOP_K,
        num_local,
        L.expert_offset,
        _MOE_TILE,
    )
    return out


def _moe_dep_ctx(L: _MoeW, x: torch.Tensor, st: _Step, rt: _Rt) -> _MoeOut:
    """A context server's step: the all-to-all around this rank's CuteDSL experts.

    When a rank has no rows, every rank pads to the step's largest count before the collective (as
    the engine does, so that no rank runs the quantization on zero rows ahead of it): a rank with
    rows pads its router logits and latent with zero rows after the two GEMMs, a rank without
    rows runs them on zero rows. The padded rows' outputs are dropped.
    """
    rows, runtime_max = x.shape[0], max(st.all_rank_num_tokens)
    pad = runtime_max - rows if 0 in st.all_rank_num_tokens else 0
    xg = x.new_zeros(pad, x.shape[1]) if pad and not rows else x
    logits = _moe_router(L, xg)
    latent_in = _moe_latent_gemm(xg, L.latent_down)
    if pad and rows:
        logits = torch.cat([logits, logits.new_zeros(pad, logits.shape[1])])
        latent_in = torch.cat([latent_in, latent_in.new_zeros(pad, latent_in.shape[1])])
    ids, weights = _moe_route(L, logits)
    x4, sf = _moe_quantize(L, latent_in)
    x4, sf, ids, weights, at, out = rubin_moe_a2a_dispatch(
        x4,
        sf,
        ids,
        weights,
        rt.a2a,
        runtime_max,
        _MOE_A2A_INVALID_ID,
        torch.bfloat16,
        latent_in.shape[1],
    )
    out = _moe_cutedsl(L, x4, sf, ids, weights, out, st, rt)
    routed = rubin_moe_a2a_combine(out, latent_in.shape[0], rt.a2a, runtime_max, at)
    latent = flashinfer_rmsnorm(routed[:rows], L.latent_norm, _MOE_EPS)
    return _MoeOut(_moe_latent_gemm(latent, L.latent_up), _moe_shared_ctx(L, x))


def _moe_bucket_rows(rows: int) -> bool:
    """Whether a chunk's ``rows`` go into its bucket (routing and quantization) rather than staged."""
    return 0 < rows <= _MOE_ROUTE_MAX_M and rows not in _MOE_ROUTE_SKIP_M


def _moe_bucket_route(L: _MoeW, logits: torch.Tensor, b, rows: int) -> None:
    """The top-16 of the chunk's ``rows`` rows of logits into rows ``[0, rows)`` of bucket ``b``."""
    rubin_route_topk_out(
        logits,
        L.router_bias,
        _MOE_SCALE,
        b.topk_idx[:rows],
        b.topk_weights[:rows, :_MOE_TOP_K],
    )


def _moe_bucket_quantize(L: _MoeW, latent_in: torch.Tensor, b) -> None:
    """The chunk's rows quantized into bucket ``b``; its routing past them emptied."""
    rubin_megamoe_quantize(
        latent_in, L.fc31_in_scale, b.activation, b.activation_sf, b.topk_idx, b.topk_weights
    )


def _moe_bucket_launch(L: _MoeW, b, rows: int, rt: _Rt) -> torch.Tensor:
    """MegaMoE on bucket ``b`` as its two producers left it: ``[rows, latent]`` BF16."""
    return rubin_megamoe_in_bucket(
        b,
        L.fc1_w,
        L.fc1_s,
        L.fc2_w,
        L.fc2_s,
        L.fc31_alpha,
        L.fc2_alpha,
        L.fc1_norm_const,
        rt.megamoe,
        rows,
    )


def _moe_megamoe_chunk(L: _MoeW, logits, latent_in, chunk_max: int, rt: _Rt) -> torch.Tensor:
    """One chunk of this rank's rows through MegaMoE; with no rows it still launches.

    With rows (at most the routing kernel's ``_MOE_ROUTE_MAX_M``, and not a count of
    ``_MOE_ROUTE_SKIP_M``), the top-16 and the NVFP4 quantization write them into the chunk's
    bucket and MegaMoE launches on it as it is. A rank without rows, with more, or with a count of
    ``_MOE_ROUTE_SKIP_M`` stages its rows into the same bucket (``rubin_megamoe``).
    """
    rows = latent_in.shape[0]
    if _moe_bucket_rows(rows):
        b = megamoe_bucket(rt.megamoe, chunk_max, rows)
        _moe_bucket_route(L, logits, b, rows)
        _moe_bucket_quantize(L, latent_in, b)
        return _moe_bucket_launch(L, b, rows, rt)
    if rows:
        ids, weights = _moe_route(L, logits)
        x4, sf = _moe_quantize(L, latent_in)
    else:
        ids = torch.empty(0, _MOE_TOP_K, dtype=torch.int32, device=latent_in.device)
        weights = torch.empty(0, _MOE_TOP_K, dtype=torch.float32, device=latent_in.device)
        x4 = torch.empty(0, latent_in.shape[1] // 2, dtype=torch.uint8, device=latent_in.device)
        sf = torch.empty(0, latent_in.shape[1] // _MOE_SF_VEC, dtype=torch.uint8, device=x4.device)
    return rubin_megamoe(
        x4,
        sf,
        ids,
        weights,
        L.fc1_w,
        L.fc1_s,
        L.fc2_w,
        L.fc2_s,
        L.fc31_alpha,
        L.fc2_alpha,
        L.fc1_norm_const,
        rt.megamoe,
        chunk_max,
        rows,
    )


def _moe_split(rows: int, chunks: int) -> list[int]:
    """``rows`` in ``chunks`` near-equal parts, the larger first (the engine's split)."""
    q, r = divmod(rows, chunks)
    return [q + 1] * r + [q] * (chunks - r)


def _moe_megamoe(L: _MoeW, logits, latent_in, st: _Step, rt: _Rt) -> torch.Tensor:
    """This rank's rows through MegaMoE: ``[rows, latent]`` BF16, complete.

    The rows are this rank's entry of ``all_rank_num_tokens`` (any rows past them are padding).
    A step goes in ``ceil(max rows / chunk_tokens)`` chunks; every rank launches every chunk,
    and a chunk's bucket follows its largest row count over the ranks.
    """
    ws = rt.megamoe
    counts = st.all_rank_num_tokens
    rows = counts[ws.ep_rank]
    chunks = max(1, -(-max(counts) // ws.chunk_tokens))
    if chunks == 1:
        return _moe_megamoe_chunk(L, logits[:rows], latent_in[:rows], max(counts), rt)
    splits = [_moe_split(n, chunks) for n in counts]
    routed = latent_in.new_empty(rows, latent_in.shape[1])
    start = 0
    for c in range(chunks):
        n = splits[ws.ep_rank][c]
        chunk_max = max(s[c] for s in splits)
        part = slice(start, start + n)
        # Copied out: the next chunk's launch overwrites the bucket's output.
        routed[part].copy_(_moe_megamoe_chunk(L, logits[part], latent_in[part], chunk_max, rt))
        start += n
    return routed


def _moe_dep_gen_routed(L: _MoeW, x: torch.Tensor, logits, st: _Step, rt: _Rt) -> torch.Tensor:
    """A generation server's routed experts, complete: down, MegaMoE, norm, up."""
    latent_in = _moe_latent_gemm(x, L.latent_down)
    routed = _moe_megamoe(L, logits, latent_in, st, rt)
    return _moe_latent_gemm(flashinfer_rmsnorm(routed, L.latent_norm, _MOE_EPS), L.latent_up)


def _moe_dep_gen(L: _MoeW, x: torch.Tensor, st: _Step, rt: _Rt, xf=None) -> _MoeOut:
    """A generation server's step on the current stream (``xf``: see ``_moe_router_fp32``)."""
    up = _moe_dep_gen_routed(L, x, _moe_router_fp32(L, x, xf), st, rt)
    return _MoeOut(up, _moe_shared(L, x))


def _moe_dep_gen_streams(L: _MoeW, x: torch.Tensor, st: _Step, rt: _Rt, xf=None) -> _MoeOut:
    """A generation server's CUDA graph step; the same bits as in order. ``xf``: the router's FP32 input when
    the mix stored it (lever router_f32): then no widening runs and the fork follows the mix directly.

    ``latent_down`` forks onto the latent stream after the FP32 router's cast, so the router GEMM
    is placed on the SMs first and latent_down runs beside it and the routing. On a one-chunk
    step of the bucket path the quantization into the bucket follows latent_down on the latent
    stream, beside the routing into the same bucket: the routing writes its rows ``[0, rows)``,
    the quantization the activation and the routing past them, so their bytes are disjoint.
    MegaMoE launches after this stream's wait on ``latent_ready``. The shared experts run beside
    the routed chain from the router's end, as before; their gate_up GEMM keeps
    ``rt.shared_gate_up_reserve`` SMs out of its grid, so MegaMoE's token exchange starts on them.
    """
    c = rt.moe_comm
    ws = rt.megamoe
    counts = st.all_rank_num_tokens
    rows = counts[ws.ep_rank]
    b = None
    if max(counts) <= ws.chunk_tokens and _moe_bucket_rows(rows):  # one chunk, into its bucket
        b = megamoe_bucket(ws, max(counts), rows)
    if xf is None:
        xf = _moe_router_input(x)  # the FP32 router's input (_moe_router_fp32)
    c.latent_fork.record()
    # No record_stream: latent_in is read on this stream only after the wait on latent_ready,
    # and later work on the latent stream forks from this stream.
    with torch.cuda.stream(c.latent_stream):
        c.latent_fork.wait()
        latent_in = _moe_latent_gemm(x, L.latent_down)
        if b is not None:
            _moe_bucket_quantize(L, latent_in[:rows], b)
        c.latent_ready.record()
    logits = rubin_linear(xf, L.router, False)  # FP32, TF32 off (_moe_router_fp32)
    c.shared_fork.record()
    if b is not None:
        _moe_bucket_route(L, logits[:rows], b, rows)
        c.latent_ready.wait()  # right before MegaMoE, the bucket's reader
        routed = _moe_bucket_launch(L, b, rows, rt)
    else:
        c.latent_ready.wait()
        routed = _moe_megamoe(L, logits, latent_in, st, rt)
    logger.info_once(_MOE_LATENT_BESIDE_LINE, key=_MOE_LATENT_BESIDE_LINE)
    up = _moe_latent_gemm(flashinfer_rmsnorm(routed, L.latent_norm, _MOE_EPS), L.latent_up)
    # No record_stream: the output is read on this stream only after the join, and later work
    # on the side stream forks from this stream.
    with torch.cuda.stream(rt.aux_stream):
        c.shared_fork.wait()
        shared = _moe_shared(L, x, rt.shared_gate_up_reserve)
        c.shared_join.record()
    c.shared_join.wait()
    return _MoeOut(up, shared)


def latent_moe(L: _MoeW, i: int, x: torch.Tensor, st: _Step, rt: _Rt, xf=None) -> _MoeOut:
    """Run layer ``i``'s latent MoE (routed and shared experts) over ``x``; return both parts. ``xf``: ``x`` in
    FP32 when the post-attention mix stored it (lever router_f32), the generation router's input."""
    if rt.megamoe is None:  # a context server: the all-to-all and the CuteDSL experts
        return _moe_dep_ctx(L, x, st, rt)
    if st.cuda_graph:  # the graph runner's warmup and capture
        return _moe_dep_gen_streams(L, x, st, rt, xf)
    return _moe_dep_gen(L, x, st, rt, xf)


def dense_mlp(L0: _DenseW, x: torch.Tensor, st: _Step, rt: _Rt) -> torch.Tensor:
    """Run layer 0's dense SiTU MLP over ``x``; return ``[rows, hidden]``."""
    h = situ_and_mul(cublas_mm(x, L0.gu.t()), *_MOE_SITU)
    return cublas_mm(h, L0.dn.t())  # whole: attention DP replicates the MLP


# --------------------------------------------------------------------------------------------------
# DFlash2: the drafter container, its block step and the speculative worker.
#
# The drafter's weights are ``core.w["d_*"]`` (``w3_rows``). ``DraftModel`` holds no parameter of
# its own: it reads them through the core and answers what the engine's worker, the KV cache
# creator and the weight loader read off a drafter. ``draft_step`` is the block decode of the five
# drafter layers, straight-line, every kernel a family or catalog entry.
#
# dep16: attention DP, so the drafter runs whole on every rank, as the target's attention does: all 96
# query heads, ``o_proj`` and the MLP whole, no collective. The history (one FP8 latent of 576 values
# per token and drafter layer) lives in the unified draft cache, each rank holding its own requests'.
# The window attention is the exact FP32 decode over the window's pool rows, its operands staged by
# two Triton launches; the block's latents go into the pool before each layer's decode, at the slots
# the next step's accepted rows overwrite.
# --------------------------------------------------------------------------------------------------

# The engine's dflash_forward keywords, as the worker passes them.
_DRAFT_FORWARD_ARGS = (
    "noise_embedding",
    "query_positions",
    "num_ctx_per_req",
    "ctx_k_cache",
    "ctx_v_cache",
    "ctx_cache_batch_idx",
    "ctx_kv_cache",
    "ctx_page_table",
    "block_placement",
)
#: Forced-acceptance calls between two forced-count lines.
_FORCED_PERIOD = 200
# The drafter's attention window in positions, the query's own included (sliding_window).
_DRAFT_WINDOW = _weights.DRAFT_SHAPE["sliding_window"]


class _DraftLayerW(NamedTuple):
    """One drafter layer at this rank's share: views of ``core.w["d_l{j}_*"]`` (``w3_rows``)."""

    in_norm: torch.Tensor  # [H]
    post_norm: torch.Tensor  # [H]
    q_a: torch.Tensor  # [q_lora, H]
    q_a_norm: torch.Tensor  # [q_lora]
    q_b: torch.Tensor  # [heads * (nope + rope), q_lora]
    kv_a: torch.Tensor  # [kv_lora + rope, H]: rows of d_kv_a
    kv_a_norm: torch.Tensor  # [kv_lora]: a row of d_kv_a_norm
    k_b: torch.Tensor  # [heads, nope, kv_lora]: W_UK
    v_b_t: torch.Tensor  # [heads, kv_lora, v]: W_UV, transposed view
    o: torch.Tensor  # [H, o heads * v]
    gate_up: torch.Tensor  # [2 * I / tp, H]: gate rows, then up rows
    down: torch.Tensor  # [H, I / tp]
    attn_conv_base: torch.Tensor  # [2, taps, H]: side 0 the input's, side 1 the output's
    attn_conv_proj: torch.Tensor  # [2 * taps * groups, H]
    mlp_conv_base: torch.Tensor
    mlp_conv_proj: torch.Tensor
    # draft_q_a_kv_a_fused: the stacked [q_a ; kv_a] rows (d_l{j}_q_a_kv_a), which q_a views;
    # None otherwise
    q_a_kv_a: torch.Tensor | None = None


class _DraftW(NamedTuple):
    """The drafter's weights (``DW``); a history-only drafter has no layers and no head."""

    fc: torch.Tensor  # [H, captures * H]
    layers: tuple  # _DraftLayerW per layer; () on a history-only drafter
    final_norm: Optional[torch.Tensor]


class _DraftStep(NamedTuple):
    """One block step's drafter state (``st``), from the worker's drafter inputs."""

    block: int  # queries per request: max_draft_len + 1
    rows: int  # requests * block
    num_ctx: torch.Tensor  # [B] int64: each request's history length (the block's RoPE base)
    page_tables: torch.Tensor  # [B, pages] int64: each request's history pages
    page_size: int
    positions: torch.Tensor  # [rows]: the queries' positions
    pools: Any  # per drafter layer, the history pool [pages, 1, 1, page, 576]


class _DraftRt(NamedTuple):
    """The drafter's runtime (``rt.draft``), built at the worker's first prepare, before capture."""

    freqs: torch.Tensor  # [positions, rope / 2] complex64: the drafter's YaRN table
    softmax_scale: float
    eps: float
    kv_scale: float  # the FP8 history's dequant scale
    taps: int  # the block convolutions' taps
    groups: int  # channel groups of a convolution's per-position kernel
    group: int  # channels per group


class _Weight(NamedTuple):
    """An object with ``.weight``: what the engine reads off a drafter's ``fc`` or ``hidden_norm``."""

    weight: torch.Tensor


class _CandidateSelector(NamedTuple):
    """The DFlash2 candidate selector's tables, as the engine reads them."""

    predecessor_codebook: torch.Tensor  # [vocab, rank]
    successor_codebook: torch.Tensor  # [vocab, rank]
    hidden_projection_weight: torch.Tensor  # [rank, H]
    top_k: int


class _TargetEmbedding:
    """The target's embedding table (``core.w["embed"]``, whole) as the drafter's ``embed_tokens``."""

    def __init__(self, core: nn.Module) -> None:
        self._core = weakref.ref(core)

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        return nn.functional.embedding(input_ids, self._core().w["embed"])


class _DraftBackbone(NamedTuple):
    """``draft_model.draft_model_full.model``: the embedding the worker embeds the block with."""

    embed_tokens: Any


class _DraftFull(NamedTuple):
    """``draft_model.draft_model_full``: the backbone and the LM head the drafter shares."""

    model: _DraftBackbone
    lm_head: Any


def _yarn_mscale(scale: float, mscale: float) -> float:
    """YaRN's attention scale factor."""
    return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0


def _draft_rope(cfg: Any) -> dict:
    """The drafter's YaRN parameters (``rope_parameters``, else ``rope_scaling``)."""
    scaling = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
    rope_type = scaling.get("rope_type") or scaling.get("type")
    if rope_type != "yarn":
        raise ValueError(f"the DFlash2 drafter's RoPE is YaRN, got {rope_type!r}")
    return dict(
        base=float(scaling.get("rope_theta") or cfg.rope_theta),
        scaling_factor=float(scaling["factor"]),
        original_max_position_embeddings=int(scaling["original_max_position_embeddings"]),
        beta_fast=float(scaling.get("beta_fast", 32.0)),
        beta_slow=float(scaling.get("beta_slow", 1.0)),
        mscale=float(scaling.get("mscale", 1.0)),
        mscale_all_dim=float(scaling.get("mscale_all_dim", 0.0)),
    )


def _draft_freqs(rope: dict, positions: int, dim: int, device: Any) -> torch.Tensor:
    """``[positions, dim / 2]`` complex64: the YaRN rotation as adjacent (re, im) pairs.

    The engine's sinusoid table with its frequency half duplicated, read as complex pairs; mscale
    rides in the modulus.
    """
    _, packed = RopeEmbeddingUtils.create_sinusoidal_positions_yarn(
        num_pos=positions, dim=dim, duplicate_data=True, **rope
    )
    table = torch.from_numpy(packed).reshape(positions, dim, 2)
    cos = table[..., 0].contiguous().to(device)
    sin = table[..., 1].contiguous().to(device)
    return torch.complex(cos[..., : dim // 2], sin[..., : dim // 2]).contiguous()


def _draft_views(w: nn.ParameterDict, layers: int, nope: int, kv_lora: int) -> _DraftW:
    """The drafter's views over ``core.w`` (``w3_rows`` lays out every row as read here)."""
    if "d_final_norm" not in w:
        return _DraftW(w["d_fc"], (), None)
    kv_a, kv_a_norm = w["d_kv_a"], w["d_kv_a_norm"]
    latent = kv_a.shape[0] // layers
    views = []
    for j in range(layers):
        kv_b = w[f"d_l{j}_kv_b"]  # [every head's k_nope rows ; every head's v rows]
        heads = kv_b.shape[0] // (2 * nope)
        # draft_q_a_kv_a_fused: the table loads q_a stacked with kv_a; q_a views the stack.
        q_a_kv_a = w[f"d_l{j}_q_a_kv_a"] if f"d_l{j}_q_a_kv_a" in w else None
        q_a = w[f"d_l{j}_q_a"] if q_a_kv_a is None else q_a_kv_a[: q_a_kv_a.shape[0] - latent]
        views.append(
            _DraftLayerW(
                in_norm=w[f"d_l{j}_in_norm"],
                post_norm=w[f"d_l{j}_post_norm"],
                q_a=q_a,
                q_a_norm=w[f"d_l{j}_q_a_norm"],
                q_b=w[f"d_l{j}_q_b"],
                kv_a=kv_a[j * latent : (j + 1) * latent],
                kv_a_norm=kv_a_norm[j],
                k_b=kv_b[: heads * nope].view(heads, nope, kv_lora),
                v_b_t=kv_b[heads * nope :].view(heads, -1, kv_lora).transpose(1, 2),
                o=w[f"d_l{j}_o"],
                gate_up=w[f"d_l{j}_gate_up"],
                down=w[f"d_l{j}_down"],
                attn_conv_base=w[f"d_l{j}_attn_conv_base"],
                attn_conv_proj=w[f"d_l{j}_attn_conv_proj"],
                mlp_conv_base=w[f"d_l{j}_mlp_conv_base"],
                mlp_conv_proj=w[f"d_l{j}_mlp_conv_proj"],
                q_a_kv_a=q_a_kv_a,
            )
        )
    return _DraftW(w["d_fc"], tuple(views), w["d_final_norm"])


def draft_config_for(model_config: Any) -> Any:
    """Build the DFlash2 drafter checkpoint's ``ModelConfig``; None without a speculative config.

    Built as the engine builds an external drafter's. The shell keeps it as ``draft_config`` (the
    KV cache creator and the draft weight mapper read it) and passes it to ``DraftModel``. The
    drafter's architecture is weights.py's (``DRAFT_SHAPE``, ``DRAFT_DFLASH``); another one raises
    here.
    """
    spec_config = getattr(model_config, "spec_config", None)
    if spec_config is None:
        return None
    mode = spec_config.spec_dec_mode
    if not (mode.use_one_engine() and mode.is_dflash()):
        raise ValueError(
            "This target drafts with one-engine DFlash (the DFlash2 MLA drafter) only."
        )
    draft_config = ModelConfig.from_pretrained(
        spec_config.speculative_model, **rubin_draft_config_kwargs(model_config, spec_config)
    )
    # The target's KV dtype; the history's own dtype comes from the container (context_kv_fp8).
    draft_config.quant_config.kv_cache_quant_algo = model_config.quant_config.kv_cache_quant_algo
    draft_config.extra_attrs = model_config.extra_attrs
    cfg = draft_config.pretrained_config
    for key, want in _weights.DRAFT_SHAPE.items():
        got = getattr(cfg, key, None)
        if got != want:
            raise ValueError(f"the DFlash2 drafter's {key} is {got}, this target runs {want}")
    dflash = getattr(cfg, "dflash_config", None) or {}
    for key, want in _weights.DRAFT_DFLASH.items():
        if dflash.get(key) != want:
            raise ValueError(f"dflash_config.{key} {dflash.get(key)} is not {want}")
    layer_types = getattr(cfg, "layer_types", None) or []
    if not layer_types or any(t != "sliding_attention" for t in layer_types):
        raise ValueError(
            f"the DFlash2 drafter's layers are all sliding_attention layers, got {layer_types}"
        )
    if not getattr(cfg, "is_causal", True):
        raise ValueError("the DFlash2 drafter's block attention is causal")
    return draft_config


class DraftModel(nn.Module):
    """The DFlash2 drafter the engine's worker drives (dep16: whole, 96 heads per rank).

    No parameter or submodule of its own: every weight is ``core.w["d_*"]``, read through the core
    (held weakly), so meta initialization never leaves a stale tensor here. A module only because
    the engine counts a context server's drafter tensors through ``parameters()``,
    ``buffers()`` and ``modules()``.

    ``model_config`` is the drafter's ``ModelConfig`` (``draft_config_for``); the speculative
    config is the core's. On a disaggregated context server the drafter is history-only
    (``context_kv_only``): it only builds the history (``fc``, ``hidden_norm``, every layer's latent
    projection and norm) and gives the engine ``fc``'s column blocks to project the target's
    captures as they are taken.
    """

    # The engine's unified draft cache holds the history, in aggregated and disaggregated runs alike.
    unified_draft_cache_when_aggregated = True
    # One latent per token and layer (no V half), in pages the block decode reads itself.
    _kv_factor = 1
    _num_kv_heads = 1
    _paged_ctx_cache = True
    _uses_worker_attention_backend = False
    _uses_lilicorr = False
    lilicorr = None
    is_dflash2 = True
    dflash_attention_backend = "VANILLA"
    mask_token_embedding = None
    # The drafter's helix mapping: None, the history is whole on every rank.
    _helix_mapping = None

    def __init__(
        self,
        core: nn.Module,
        model_config: ModelConfig,
        lm_head: nn.Module,
        logits_processor: nn.Module,
    ) -> None:
        super().__init__()
        spec_config = core.model_config.spec_config
        self.model_config = model_config
        self.config = model_config.pretrained_config
        self._core = weakref.ref(core)
        # The shell's head, held outside the module tree (it is the shell's child).
        self._head = (lm_head, logits_processor)
        cfg = self.config
        dflash = cfg.dflash_config
        # A disaggregated context server's drafter keeps only the history (the core's weight table).
        self.context_kv_only = core.dims.draft_context_kv_only
        # The engine sizes the history pool from this flag.
        self.context_kv_fp8 = _DRAFT_HISTORY_DTYPE == torch.float8_e4m3fn
        self.block_size = spec_config.max_draft_len + 1
        self.mask_token_id = dflash["mask_token_id"]
        self.context_cache_window = int(cfg.sliding_window)
        self._num_attn_layers = cfg.num_hidden_layers
        self._num_heads = cfg.num_attention_heads // core.dims.kda_tp
        self._head_dim = cfg.kv_lora_rank + cfg.qk_rope_head_dim
        self._rope = _draft_rope(cfg)
        mscale = _yarn_mscale(self._rope["scaling_factor"], self._rope["mscale_all_dim"])
        self.softmax_scale = (
            mscale * mscale / math.sqrt(cfg.qk_nope_head_dim + cfg.qk_rope_head_dim)
        )
        self._selector_top_k = int(dflash["selector_top_k"])
        # Written by the worker before any table is sized: the positions it serves.
        self._runtime_position_ceiling = None
        self._dw: Optional[_DraftW] = None
        self._rt: Any = None
        logger.info(
            f"DFlash2 drafter: {self._num_attn_layers} layers, {self._num_heads} heads per rank, "
            f"block {self.block_size}, window {self.context_cache_window}, FP8 history, "
            f"{'context KV only' if self.context_kv_only else 'full'}"
        )

    # -- what the engine reads -------------------------------------------------

    def _w(self, key: str) -> torch.Tensor:
        return self._core().w[key]

    @property
    def lm_head(self) -> nn.Module:
        return self._head[0]

    @property
    def logits_processor(self) -> nn.Module:
        return self._head[1]

    @property
    def fc(self) -> _Weight:
        return _Weight(self._w("d_fc"))

    @property
    def hidden_norm(self) -> _Weight:
        return _Weight(self._w("d_hidden_norm"))

    @property
    def has_candidate_selector(self) -> bool:
        return not self.context_kv_only

    @property
    def candidate_selector(self) -> Optional[_CandidateSelector]:
        if self.context_kv_only:
            return None
        return _CandidateSelector(
            self._w("d_sel_pred"),
            self._w("d_sel_succ"),
            self._w("d_sel_hidden_proj"),
            self._selector_top_k,
        )

    @property
    def draft_model_full(self) -> _DraftFull:
        return _DraftFull(_DraftBackbone(_TargetEmbedding(self._core())), self.lm_head)

    @property
    def model(self) -> _DraftBackbone:
        return self.draft_model_full.model

    def _build_fused_kv_buffers(self) -> None:
        """Nothing to build: ``w3_rows`` stacks every layer's latent projection (``d_kv_a``)."""

    def load_weights_from_target_model(self, target_model: nn.Module) -> None:
        """Nothing to take: the embedding and the LM head are the target's, read through the core."""
        del target_model

    def bind_runtime(self, max_rows: int, page_size: int) -> None:
        """Read the weight views and build the drafter's runtime, before any capture.

        The worker calls it once its history buffers are bound (and again when the KV cache
        manager is replaced); ``_runtime_position_ceiling`` sizes the RoPE table.
        """
        del max_rows, page_size
        core = self._core()
        cfg = self.config
        self._dw = _draft_views(
            core.w, cfg.num_hidden_layers, cfg.qk_nope_head_dim, cfg.kv_lora_rank
        )
        positions = self._runtime_position_ceiling or cfg.max_position_embeddings
        dflash = cfg.dflash_config
        draft = _DraftRt(
            freqs=_draft_freqs(
                self._rope, int(positions), cfg.qk_rope_head_dim, self._dw.fc.device
            ),
            softmax_scale=self.softmax_scale,
            eps=float(cfg.rms_norm_eps),
            kv_scale=DRAFT_FP8_KV_SCALE,
            taps=int(dflash["conv_kernel_size"]),
            groups=cfg.hidden_size // int(dflash["conv_group_size"]),
            group=int(dflash["conv_group_size"]),
        )
        # Also an attribute: the engine counts a context server's drafter tables off the module.
        self._rt = core._rt._replace(draft=draft)

    # -- the history -----------------------------------------------------------

    def project_target_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``hidden_norm(fc(captures))``: the target's captured features in the drafter's space."""
        fc = self._w("d_fc")
        return flashinfer_rmsnorm(
            rubin_linear(hidden_states.to(fc.dtype), fc, False),
            self._w("d_hidden_norm"),
            float(self.config.rms_norm_eps),
        )

    def precompute_context_kv(self, projected_hidden: torch.Tensor, positions: torch.Tensor):
        """Every layer's history latent: ``([N, layers, 1, 576], None)``.

        One projection over the stacked ``kv_a`` rows, then [RMSNorm | RoPE] of all layers in one
        launch.
        """
        kv_a = self._w("d_kv_a")
        latent = rubin_linear(projected_hidden.to(kv_a.dtype), kv_a, False)
        latent = latent.view(projected_hidden.shape[0], self._num_attn_layers, self._head_dim)
        out = dflash2_mla_ctx_norm_rope(
            latent,
            self._w("d_kv_a_norm"),
            float(self.config.rms_norm_eps),
            self._rt.draft.freqs,
            positions.view(-1),
        )
        return out.unsqueeze(2), None

    # -- the block -------------------------------------------------------------

    def select_candidate_path(
        self, block_logits, candidate_ids, unary_logits, block_hidden_states, anchor_token_ids
    ):
        """The selector's greedy walk: ``block_logits`` rewritten in place to the walked path."""
        return k31_dflash2_selector_greedy_path(
            self._w("d_sel_pred"),
            self._w("d_sel_succ"),
            self._w("d_sel_hidden_proj"),
            candidate_ids,
            unary_logits,
            block_hidden_states,
            anchor_token_ids,
            block_logits,
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
        """The block decode of ``[B, block, hidden]`` noise embeddings; ``[B * block, hidden]``."""
        del ctx_k_cache, ctx_v_cache, block_placement
        batch, block, hidden = noise_embedding.shape
        st = _DraftStep(
            block=block,
            rows=batch * block,
            num_ctx=num_ctx_per_req[:batch].long(),
            page_tables=ctx_page_table.index_select(0, ctx_cache_batch_idx.long()).long(),
            page_size=ctx_kv_cache[0].shape[-2],
            positions=query_positions.reshape(-1),
            pools=ctx_kv_cache,
        )
        return draft_step(self._dw, noise_embedding.reshape(batch * block, hidden), st, self._rt)


def draft_step(DW: _DraftW, hidden_streams: torch.Tensor, st: _DraftStep, rt: _Rt) -> torch.Tensor:
    """One DFlash2 block over the five drafter layers; returns ``[B * block, hidden]``.

    ``hidden_streams`` is the noise block's embedding. Per layer: input norm, the attention conv's
    input side, the absorbed MLA query, the block latents into the pool, the exact FP32 window
    decode, W_UV, ``o_proj``, the conv's output side; then the MLP between the MLP conv's two
    sides. Then the final norm. Attention DP: every product is whole on the rank, no collective.
    """
    d = rt.draft
    eps, taps, groups, group = d.eps, d.taps, d.groups, d.group
    rows, block = st.rows, st.block
    heads, nope, kv_lora = DW.layers[0].k_b.shape
    latent = DW.layers[0].kv_a.shape[0]
    # Once per step: each query's window as pool rows, and the block latents' pool slots.
    window_rows, _ = dflash2_window_rows(
        st.num_ctx, st.page_tables, block, st.page_size, _DRAFT_WINDOW, _DRAFT_WINDOW, None
    )
    block_pos = st.num_ctx.view(-1, 1) + torch.arange(block, device=st.num_ctx.device)
    store_pages = st.page_tables.gather(1, block_pos // st.page_size).view(-1)
    store_offsets = (block_pos % st.page_size).view(-1)

    h = hidden_streams
    residual = h.clone()  # the fused add + norm below updates it in place
    x = flashinfer_rmsnorm(h, DW.layers[0].in_norm, eps)
    for j, L in enumerate(DW.layers):
        if j:
            x, residual = flashinfer_fused_add_rmsnorm(h, residual, L.in_norm, eps)
        coef = rubin_linear(x, L.attn_conv_proj, False).view(rows, 2, taps, groups)
        x = dflash2_grouped_conv(x, coef[:, 0], L.attn_conv_base[0], block, group)
        if L.q_a_kv_a is None:
            qa = flashinfer_rmsnorm(rubin_linear(x, L.q_a, False), L.q_a_norm, eps)
        else:
            # draft_q_a_kv_a_fused: q_a and kv_a read the same rows, one GEMM over their stack.
            qkv = rubin_linear(x, L.q_a_kv_a, False)
            qa = flashinfer_rmsnorm(qkv[:, : L.q_a.shape[0]], L.q_a_norm, eps)
        q = rubin_linear(qa, L.q_b, False)
        q = q.view(rows, heads, -1)
        query = q.new_empty(rows, heads, latent)
        bmm_out(q[..., :nope].transpose(0, 1), L.k_b, query[..., :kv_lora].transpose(0, 1))
        pool = st.pools[j][:, 0, 0]  # [pages, page, 576]
        dflash2_mla_block_rope(
            q,
            query,
            rubin_linear(x, L.kv_a, False) if L.q_a_kv_a is None else qkv[:, L.q_a.shape[0] :],
            L.kv_a_norm,
            eps,
            d.freqs,
            st.positions,
            pool,
            store_pages,
            store_offsets,
        )
        ctx = rubin_dflash2_window_fp32(
            query, pool.reshape(-1, latent), window_rows, d.softmax_scale, kv_lora, d.kv_scale
        )
        attn = ctx.new_empty(rows, heads * L.v_b_t.shape[-1])
        bmm_out(ctx.transpose(0, 1), L.v_b_t, attn.view(rows, heads, -1).transpose(0, 1))
        a = rubin_linear(attn, L.o, False)
        a = dflash2_grouped_conv(a, coef[:, 1], L.attn_conv_base[1], block, group)
        x, residual = flashinfer_fused_add_rmsnorm(a, residual, L.post_norm, eps)
        coef = rubin_linear(x, L.mlp_conv_proj, False).view(rows, 2, taps, groups)
        x = dflash2_grouped_conv(x, coef[:, 0], L.mlp_conv_base[0], block, group)
        m = rubin_linear(flashinfer_silu_and_mul(rubin_linear(x, L.gate_up, False)), L.down, False)
        h = dflash2_grouped_conv(m, coef[:, 1], L.mlp_conv_base[1], block, group)
    h, _ = flashinfer_fused_add_rmsnorm(h, residual, DW.final_norm, eps)
    return h


class DFlashWorker(RubinDFlash2WorkerMixin, UnifiedDraftCacheWorkerMixin, _EngineDFlashWorker):
    """The engine's DFlash worker with this target's DFlash2 differences.

    The family's sm_107 worker mixin (``RubinDFlash2WorkerMixin``) adds the TP ranks' agreement on the
    drafts, the FP8 history store and the draft step's host-only metadata refresh, and the family's
    unified draft cache mixin (``UnifiedDraftCacheWorkerMixin``) the managed draft cache and its
    history. Every other method not defined here is the engine's: the prefill and paged history stores,
    the fused generation prepare, the KV-length rewind. Defined here: the unified-cache check and the
    drafter's runtime (``_lazy_init_ctx_buffers``), the target-metadata hook, acceptance and sampling
    through the family's spec entries, the forced-acceptance count, the DFlash2 sampled path (selector
    walk, block verification) and ``_forward_impl``.
    """

    def __init__(self, spec_config: Any, mapping: Any, use_separate_draft_kv_cache: bool = False):
        super().__init__(spec_config, mapping, use_separate_draft_kv_cache)
        if self.force_num_accepted_tokens != 0.0:
            # CUDA graph warmup skips forced acceptance, so the draws' first use can fall inside a
            # capture: build them now (the engine's fixed-seed pool, the same on every rank).
            self._ensure_force_accept_rng_state(
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        # The sampled path's proposals by sequence slot, made on its first (eager) step.
        self._dflash2_proposals: Optional[DFlash2ProposalStore] = None
        # Forced acceptance: device totals (generation rows, accepted tokens) and Python calls.
        self._forced: Optional[torch.Tensor] = None
        self._forced_calls = 0
        # A vocab-parallel head's shards: how many, and which one this rank holds (one rank
        # without a mapping).
        self._vocab_shards = 1 if mapping is None else mapping.tp_size
        self._vocab_shard = 0 if mapping is None else mapping.tp_rank

    def _draft_reads_target_metadata(self) -> bool:
        """The drafter reads no target attention metadata: the draft step skips its refresh."""
        return False

    def _lazy_init_ctx_buffers(
        self, draft_model, spec_metadata, attn_metadata, draft_kv_cache_manager=None
    ):
        """Bind the history to the unified draft cache, then the drafter's runtime.

        Only the unified draft cache of KV cache manager V2 serves this target: anything else
        raises (routing refuses those deployments first).
        """
        if not getattr(draft_kv_cache_manager, "has_standalone_draft_cache", False):
            raise RuntimeError(
                "this target's DFlash2 drafter keeps its history in the unified draft cache of KV "
                "cache manager V2; the step's KV cache manager has none"
            )
        fresh = not (self._ctx_buf_inited and self._ctx_kv_manager is draft_kv_cache_manager)
        super()._lazy_init_ctx_buffers(
            draft_model, spec_metadata, attn_metadata, draft_kv_cache_manager
        )
        if fresh:
            draft_model.bind_runtime(
                (spec_metadata.num_seq_slots or spec_metadata.max_num_requests)
                * self._compute_block_size,
                self._ctx_page_size,
            )

    # -- target tokens and acceptance -----------------------------------------

    def _sample_and_accept_draft_tokens_base(
        self, logits, draft_tokens, num_contexts, batch_size, spec_metadata
    ):
        assert spec_metadata is None or not spec_metadata.is_ragged_verify, (
            "routing refuses ragged verification"
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

    def _apply_force_accepted_tokens(
        self, num_accepted_tokens, num_contexts, runtime_draft_len: int, spec_metadata=None
    ):
        """The engine's forced acceptance on the rows that verify drafts the drafter made; a row that
        verifies placeholder drafts keeps its own count. Counts what it forced (both verify paths call
        it)."""
        natural = self._natural_generation_counts(num_accepted_tokens, num_contexts)
        out = super()._apply_force_accepted_tokens(
            num_accepted_tokens, num_contexts, runtime_draft_len, spec_metadata=spec_metadata
        )
        if self.force_num_accepted_tokens != 0.0:
            self._keep_placeholder_counts(out, natural, num_contexts)
            self._count_forced(out, num_contexts, spec_metadata)
        return out

    def _count_forced(self, out: torch.Tensor, num_contexts: int, spec_metadata) -> None:
        """Add this call's forced rows (not the placeholder-draft rows) to the device totals; log them
        every 200 calls.

        The totals are device adds, so graph replays count; the line is written from Python calls
        outside capture (eager steps).
        """
        if self._forced is None:
            self._forced = torch.zeros(2, dtype=torch.int64, device=out.device)
        capturing = out.is_cuda and torch.cuda.is_current_stream_capturing()
        if spec_metadata is not None and spec_metadata.is_cuda_graph and not capturing:
            return  # graph warmup: the engine forced nothing
        gen = out[num_contexts:]
        forced = self._forced_rows(gen.shape[0])
        if forced is None:
            self._forced[0].add_(gen.shape[0])
            self._forced[1].add_(gen.sum())
        else:
            self._forced[0].add_(forced.sum())
            self._forced[1].add_(torch.where(forced, gen, 0).sum())
        self._forced_calls += 1
        if self._forced_calls % _FORCED_PERIOD == 0 and not capturing:
            logger.info(self.forced_line())

    def forced_line(self) -> str:
        """The forced-acceptance line: setting, calls, generation rows, accepted tokens, mean."""
        rows, accepted = self._forced.tolist() if self._forced is not None else (0, 0)
        return (
            f"Kimi K3.1 Rubin forced acceptance: {_TARGET} "
            f"value={self.force_num_accepted_tokens} calls={self._forced_calls} rows={rows} "
            f"accepted={accepted} mean={accepted / max(rows, 1):.4f}"
        )

    def _sample_tokens_for_batch(self, logits, spec_metadata, num_contexts, batch_size):
        if isinstance(logits, VocabShardedLogits):
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
        top_ks, top_ps = resolve_advanced_sampling_filters(advanced_sampling_mode, top_ks, top_ps)
        return flashinfer_sample_from_logits(logits, temperatures, top_ks, top_ps, seed, offset)

    @staticmethod
    def _compute_probs_from_logits(
        advanced_sampling_mode, logits, temperatures, top_ks, top_ps, min_ps
    ):
        if advanced_sampling_mode.is_fused:
            return fused_compute_probs_from_logits(logits, temperatures, top_ks, top_ps, min_ps)
        top_ks, top_ps = resolve_advanced_sampling_filters(advanced_sampling_mode, top_ks, top_ps)
        return flashinfer_compute_probs_from_logits(logits, temperatures, top_ks, top_ps)

    def _target_tokens_from_shard(self, spec_metadata) -> bool:
        """Whether a step's target tokens come from this rank's vocab shard.

        Never with a guided decoder, which masks the gathered logits.
        """
        return self.guided_decoder is None and (
            spec_metadata.is_all_greedy_sample or self._vocab_parallel.samples
        )

    def sample_and_accept_draft_tokens(self, logits, attn_metadata, spec_metadata):
        # A DFlash2 sampling batch verifies the selector's block by block verification; every
        # other batch takes the engine's acceptance.
        if dflash2_sampled_step(self, spec_metadata):
            return self._verify_sampled_block(logits, attn_metadata, spec_metadata)
        return SpecWorkerBase.sample_and_accept_draft_tokens(
            self, logits, attn_metadata, spec_metadata
        )

    # -- the DFlash2 sampled path ----------------------------------------------

    def _proposal_store(self, spec_metadata) -> DFlash2ProposalStore:
        """The selector's proposals by sequence slot; made on the first sampled (eager) step."""
        store = self._dflash2_proposals
        if store is None:
            assert not torch.cuda.is_current_stream_capturing(), (
                "the DFlash2 proposal store must exist before any CUDA graph capture"
            )
            draft_probs = getattr(spec_metadata, "draft_probs", None)
            rows = (
                int(draft_probs.shape[0])
                if draft_probs is not None
                else int(spec_metadata.num_seq_slots or spec_metadata.max_num_requests) + 1
            )
            store = DFlash2ProposalStore(
                rows,
                self.max_draft_len,
                self._draft_model_ref().candidate_selector.top_k,
                spec_metadata.batch_slot_ids.device,
            )
            self._dflash2_proposals = store
        return store

    def _sample_block_path(
        self, draft_model, gen_logits, block_hidden, anchor, spec_metadata, num_contexts, batch_size
    ):
        """A sampling batch's draft tokens.

        The selector's lattice over each slot's global top-k, one sampled walk (greedy rows the
        argmax), the proposals stored by sequence slot.
        """
        selector = draft_model.candidate_selector
        candidate_ids, unary_logits, _ = self._dflash2_global_top_k(
            gen_logits,
            spec_metadata,
            top_k=selector.top_k,
            full_vocab=selector.predecessor_codebook.shape[0],
        )
        seed, offset = self._rng_state_per_request(
            spec_metadata, num_contexts, batch_size, slot=spec_metadata.rng_draft_slot(0)
        )
        store = self._proposal_store(spec_metadata)
        return k31_dflash2_selector_draft(
            selector.predecessor_codebook,
            selector.successor_codebook,
            candidate_ids,
            unary_logits,
            rubin_linear(block_hidden, selector.hidden_projection_weight, False),
            anchor,
            seed,
            offset,
            spec_metadata.request_temperatures[num_contexts:batch_size],
            GREEDY_TEMPERATURE_THRESHOLD,
            store.candidate_ids,
            store.q,
            store.tokens,
            spec_metadata.batch_slot_ids[num_contexts:batch_size],
        )

    def _verify_sampled_block(self, logits, attn_metadata, spec_metadata):
        """A sampling batch's acceptance.

        Context rows sample their first token; generation rows verify the selector's block (block
        verification; greedy rows the strict rule). Returns ``(accepted_tokens [batch, K + 1],
        num_accepted_tokens [batch])``, int32.
        """
        assert not spec_metadata.is_ragged_verify, "routing refuses ragged verification"
        assert isinstance(logits, torch.Tensor), "sampled verification reads gathered logits"
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        device = logits.device
        draft_tokens = self._reshape_draft_tokens_for_accept(spec_metadata, num_gens, device)
        logits = self._reshape_logits_for_accept(logits, num_contexts, num_gens, spec_metadata)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        logits = self._apply_occurrence_penalties(
            logits, draft_tokens, num_contexts, batch_size, spec_metadata
        )
        K = draft_tokens.shape[1]
        store = self._proposal_store(spec_metadata)
        slot_ids = spec_metadata.batch_slot_ids
        accepted_tokens = torch.zeros((batch_size, K + 1), dtype=torch.int32, device=device)
        num_accepted_tokens = torch.ones(batch_size, dtype=torch.int32, device=device)
        if num_contexts > 0:
            accepted_tokens[:num_contexts, 0] = self._sample_tokens_for_batch(
                logits[:num_contexts], spec_metadata, num_contexts, num_contexts
            ).to(torch.int32)
            store.invalidate(slot_ids[:num_contexts])
        if num_gens > 0:
            start, end = num_contexts, num_contexts + num_gens * (K + 1)
            gen_logits = self._zero_padding_rows(
                logits[start:end],
                spec_metadata,
                num_contexts,
                batch_size,
                K + 1,
                packed_verify=True,
            )
            vocab = gen_logits.shape[-1]
            target_probs = self._compute_probs_from_logits(
                spec_metadata.advanced_sampling_mode,
                gen_logits,
                spec_metadata.temperatures[start:end],
                spec_metadata.top_ks[start:end],
                spec_metadata.top_ps[start:end],
                spec_metadata.min_ps[start:end],
            ).view(num_gens, K + 1, vocab)
            candidate_ids, q = k31_dflash2_proposal_load(
                store.candidate_ids,
                store.q,
                store.tokens,
                slot_ids[num_contexts:batch_size],
                draft_tokens,
            )
            seed, offset = self._rng_state_per_request(
                spec_metadata, num_contexts, batch_size, slot=spec_metadata.rng_accept_slot
            )
            tau, bonus = dflash2_block_verify_seeded(
                target_probs, draft_tokens, candidate_ids, q, seed, offset
            )
            greedy = (
                spec_metadata.request_temperatures[num_contexts:batch_size]
                <= GREEDY_TEMPERATURE_THRESHOLD
            )
            greedy_tau, greedy_bonus = dflash2_greedy_verify(
                gen_logits.view(num_gens, K + 1, vocab), draft_tokens
            )
            tau = torch.where(greedy, greedy_tau, tau)
            bonus = torch.where(greedy, greedy_bonus, bonus)
            # Drafts before tau, the bonus at tau; later positions keep drafts (emitted only under
            # a forced acceptance count).
            tokens = torch.cat((draft_tokens.long(), draft_tokens[:, -1:].long()), dim=1)
            tokens = tokens.scatter(1, tau.unsqueeze(1), bonus.unsqueeze(1))
            accepted_tokens[num_contexts:] = tokens.to(torch.int32)
            num_accepted_tokens[num_contexts:] = (tau + 1).to(torch.int32)
        num_accepted_tokens = self._apply_force_accepted_tokens(
            num_accepted_tokens, num_contexts, K, spec_metadata=spec_metadata
        )
        return self._commit_occurrence_counts(
            agree_verified_tokens(self, accepted_tokens, num_accepted_tokens),
            batch_size,
            spec_metadata,
        )

    def _dflash2_global_top_k(self, gen_logits, spec_metadata, top_k: int, full_vocab: int):
        """Per-position global top-k candidates and the tensor to score into.

        This rank's top-32; when the drafter's logits are a vocab shard, the shard's ids made
        global and the gathered pairs merged.
        """
        shard = gen_logits.shape[-1]
        if shard == full_vocab:
            unary_logits, candidate_ids = k31_dflash2_shard_top32(gen_logits, top_k, 0, False)
            return candidate_ids, unary_logits, gen_logits
        assert self._draft_logits_are_sharded(gen_logits, spec_metadata) and (
            shard * self._vocab_shards == full_vocab
        ), f"draft logits of width {shard} are neither the vocab {full_vocab} nor a TP shard of it"
        combined = k31_dflash2_shard_top32(gen_logits, top_k, self._vocab_shard * shard, True)
        gathered = allgather_module(combined, self.mapping, -1, None)
        unary_logits, candidate_ids = k31_dflash2_merge_top32(gathered.contiguous(), top_k)
        block_logits = gen_logits.new_full((*gen_logits.shape[:-1], full_vocab), float("-inf"))
        return candidate_ids, unary_logits, block_logits

    def _block_gather_ids(self, draft_model, num_gens, block_size, num_draft_tokens, num_rows):
        """Block-output rows whose hidden states give the K draft logits (slots 1 .. K)."""
        ids = self._draft_slot_ids(draft_model, num_gens, block_size, num_draft_tokens)
        return ids.clamp(max=num_rows - 1)

    # -- the step --------------------------------------------------------------

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
                # The target tokens come from the shard; the engine gathers the shard only for a
                # logits post-processor.
                raw_logits = logits.local
                logits_vocab_shard = True
            else:
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
        if not self._ctx_buf_inited or self._ctx_kv_manager is not draft_kv_cache_manager:
            raise RuntimeError("the unified draft cache must be prepared before the forward")

        # Snapshots for the rollback after a warmup or a failed forward; never inside a capture.
        is_warmup = spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing()
        if not torch.cuda.is_current_stream_capturing():
            self._saved_ctx_len = self._ctx_len.clone()
            self._saved_ctx_position_offset = self._ctx_position_offset.clone()
            self._saved_ctx_len_host = list(self._ctx_len_host)
            self._saved_req_ctx_pos = dict(self._req_ctx_pos)
            self._ctx_len_restore_pending = True

        self._execute_guided_decoder_if_present(logits)

        self._set_placeholder_rows(attn_metadata, num_gens)
        accepted_tokens, num_accepted_tokens = self.sample_and_accept_draft_tokens(
            logits, attn_metadata, spec_metadata
        )

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

        # The KDA states follow the accepted tokens.
        if num_gens > 0:
            attn_metadata.kv_cache_manager.update_mamba_states(
                attn_metadata=attn_metadata,
                num_accepted_tokens=num_accepted_tokens,
                state_indices=attn_metadata.mamba_metadata.state_indices,
            )

        self._prepare_attn_metadata_for_dflash(attn_metadata, spec_metadata)
        self._prepare_kv_for_draft_forward(
            attn_metadata, num_accepted_tokens, num_contexts, batch_size
        )

        # The drafter's RoPE is 1-D: the first (temporal) row of an mrope position tensor.
        position_ids = position_ids[0, 0] if position_ids.ndim == 3 else position_ids.squeeze(0)
        total_target_tokens = input_ids.shape[0]

        if num_contexts > 0:
            self._store_prefill_context(
                draft_model, spec_metadata, attn_metadata, position_ids, total_target_tokens
            )
            if self._ctx_buf_inited and spec_metadata.request_ids:
                self._rebuild_batch_to_slot(spec_metadata.request_ids)

        drafts = not draft_model.context_kv_only
        if not drafts and num_gens > 0:
            self._check_no_real_generation(spec_metadata, num_contexts, batch_size)

        vocab_size = None
        gen_draft_tokens = None
        if num_gens > 0 and drafts:
            inputs = self.prepare_1st_drafter_inputs(
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
            with self.draft_kv_cache_context(attn_metadata, None):
                hidden_states_out = draft_model.dflash_forward(
                    **{name: inputs[name] for name in _DRAFT_FORWARD_ARGS}
                )
                gen_hidden_states = hidden_states_out[
                    self._block_gather_ids(
                        draft_model,
                        num_gens,
                        self._compute_block_size,
                        K,
                        hidden_states_out.shape[0],
                    )
                ]
                block_hidden = gen_hidden_states.reshape(num_gens, K, -1)
                gen_logits = draft_model.logits_processor(
                    gen_hidden_states, draft_model.lm_head, attn_metadata, True
                )
                gen_logits = gen_logits.reshape(num_gens, K, gen_logits.shape[-1])
                if dflash2_sampled_step(self, spec_metadata):
                    gen_draft_tokens = self._sample_block_path(
                        draft_model,
                        gen_logits,
                        block_hidden,
                        inputs["first_prev_tokens"],
                        spec_metadata,
                        num_contexts,
                        batch_size,
                    )
                else:
                    # One coherent path through the block instead of per-position picks.
                    gen_logits = self._apply_dflash2_selector(
                        draft_model,
                        gen_logits,
                        block_hidden,
                        inputs["first_prev_tokens"],
                        spec_metadata,
                    )
                vocab_size = gen_logits.shape[-1]
                if gen_draft_tokens is None:
                    gen_draft_tokens = self.sample_draft_tokens(
                        gen_logits, spec_metadata, batch_size, num_contexts=num_contexts
                    )
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
                        block_hidden,
                        inputs["first_prev_tokens"],
                        gen_draft_tokens,
                    )
        else:
            # A context-KV-only drafter, or no generation rows: placeholder drafts.
            gen_draft_tokens = torch.zeros((num_gens, K), dtype=torch.int32, device="cuda")

        # Context requests draft nothing yet; a one-hot draft-prob row keeps them a legal
        # distribution when they turn into generation requests.
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

        # A warmup leaves the history lengths as it found them; real steps keep the updates.
        if is_warmup:
            self._ctx_len.copy_(self._saved_ctx_len)
            self._restore_ctx_len_host()
        self._ctx_len_restore_pending = False

        outputs = {
            # The gathered FP32 target logits, or this rank's vocab shard when the target tokens
            # came from it.
            "logits": raw_logits,
            "logits_vocab_shard": logits_vocab_shard,
            "new_tokens": accepted_tokens,
            "new_tokens_lens": num_accepted_tokens,
            "next_draft_tokens": next_draft_tokens,
            "next_new_tokens": next_new_tokens,
        }
        # The request-sharing TP ranks propose the group rank 0's drafts.
        agree_draft_tokens(outputs, self.mapping)
        return outputs


# --------------------------------------------------------------------------------------------------
# Vision: the MoonViT tower and its projector, and the splice of their rows into a context step's
# prompt embeddings. The generic target's image path (``K31VisionTower`` and the hooks around it,
# in kimi_k31_pro_mxfp4__sm_10x__default), written flat:
#
# * the weights are the table's ``w4_rows`` (``v_*``, BF16, whole on every rank), built on a server
#   that can run a context step and is not text-only (``Dims.vision``); ``_vision_view`` reads them
#   once after the load, with what the tower needs beside them: the 2-D rotary table, the temporal
#   rows, the attention workspace, and the attention capacity of the checkpoint processor's largest
#   image grid;
# * ``vision_encode`` is ``K31VisionTower.encode`` call for call: the patch embedding, 27 pre-norm
#   blocks (RMSNorm, the fused qkv projection, the 2-D rotary embedding, bidirectional attention
#   within each image, the output projection, RMSNorm, the GELU(tanh) MLP, each with its residual),
#   the final RMSNorm, the 2x2 merge (the temporal mean of each window) and the projector
#   (projection, exact GELU, projection, RMSNorm), on the same entries in the same order and dtypes,
#   so the rows are its bits. ``vision_forward`` encodes a request's images together, as the generic
#   tower does, while their patches fit the encoder budget (one largest image: the activations the
#   KV-capacity estimate reserves); a request past it is encoded in consecutive groups within it;
# * ``image_inputs_embeds`` runs at the shell, ahead of the core. On a context step whose requests
#   carry images it encodes each request once (the engine keeps the rows on the request for its
#   later chunks, and slices those a chunk covers past a reused prefix) and writes the rows over the
#   media placeholders the executor found (``mm_token_indices``, from ``multimodal_token_ids``);
#   every other row is the token embedding the prefill target would take. A step without images
#   gets None, and the prefill target embeds its ids as before. The prompt ids stay the DFlash2
#   worker's input;
# * ``vision_profiling_rows``: a text-only context step of the KV-capacity estimate holds one largest
#   image's rows, so the KV pool is sized with the tower's activations counted;
# * ``vision_admit`` is the executor's admission check (``validate_request``): a grid the tower
#   cannot encode, a pixel payload that is not its grids' patches, or media on a server without a
#   tower (unless the request is generation-only: its context server encoded them) fail that
#   request alone.
#
# Every rank runs the whole tower over the images of its own requests, in context steps only,
# eagerly; no generation step and no CUDA graph sees an image.
# --------------------------------------------------------------------------------------------------

_VISION_HEADS = _K31_VISION["vt_num_attention_heads"]
_VISION_HEAD_DIM = _K31_VISION["qkv_hidden_size"] // _VISION_HEADS
_VISION_MERGE = _K31_VISION["merge_kernel_size"]
_VISION_PATCH = _K31_VISION["patch_size"]
_VISION_FRAMES = _K31_VISION["init_pos_emb_time"]
#: The refusal of media on a server without a tower.
_VISION_NO_TOWER = (
    "Image input is not supported: this server runs no vision tower (disable_mm_encoder: a "
    "text-only server)"
)
#: The processed media the executor moves to the GPU: the images under ``image``, or their rows.
_VISION_DEVICE_PATHS = ("image.pixel_values", "image.image_grid_thw", "multimodal_embedding")


class _VisionBlockW(NamedTuple):
    """One MoonViT block's weights (``w4_rows`` ``v{j}_*``)."""

    norm0: torch.Tensor
    norm1: torch.Tensor
    wqkv: torch.Tensor
    wo: torch.Tensor
    fc0: torch.Tensor
    fc1: torch.Tensor


class _VisionW(NamedTuple):
    """The tower's weights and the runtime it needs, read once after the load (``_vision_view``)."""

    patch_proj: torch.Tensor
    pos_emb: torch.Tensor
    blocks: tuple[_VisionBlockW, ...]
    final_norm: torch.Tensor
    merge_fc0: torch.Tensor
    merge_fc1: torch.Tensor
    post_norm: torch.Tensor
    # [frames, 1, hidden] BF16: the temporal rows of a multi-frame grid (an image grid has one).
    time_weight: torch.Tensor
    # [TABLE_SIDE, TABLE_SIDE, head_dim / 2] complex64: the 2-D rotary angle table.
    rope_table: torch.Tensor
    # The attention op's workspace, grown in place by the op and never shrunk.
    workspace: torch.Tensor
    # (t, h, w) of the checkpoint processor's largest image.
    max_grid: tuple[int, int, int]
    # (max_seq_len, max_num_requests) of every attention call: the largest image's patches, and as
    # many sequences as the smallest images (one merge window each) that fill them.
    capacity: tuple[int, int]


def _check_vision(vision_config: Any) -> None:
    """Check the composite config's vision config is the tower this target builds (``_K31_VISION``);
    raises ``ValueError`` on another."""
    for key, want in _K31_VISION.items():
        if isinstance(vision_config, dict):
            got = vision_config.get(key)
        else:
            got = getattr(vision_config, key, None)
        if isinstance(got, list):
            got = tuple(got)
        if got != want:
            raise ValueError(f"vision_config.{key} {got!r} is not {want!r}")


def _vision_time_rows(dim: int, frames: int) -> np.ndarray:
    """``[frames, dim]`` 1-D sin-cos rows of the frame index (the sin half, then the cos half), as the
    generic tower computes its temporal rows."""
    omega = np.arange(dim // 2, dtype=np.float32)
    omega /= dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.outer(np.arange(frames, dtype=np.float32), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _vision_view(core: Any) -> _VisionW | None:
    """The tower's weights (``w4_rows``) and runtime, read off the realized parameters once; None on
    a server that builds no tower.

    The largest image grid is the one the checkpoint's HF processor produces at its limits
    (``core.media_proc_cfg``, which the shell's ``load_weights`` reads from the checkpoint's
    ``preprocessor_config.json``); without that file, the processor's default limits.
    """
    if not core.dims.vision:
        return None
    w = core.w
    device = w["v_patch_proj"].device
    media = core.media_proc_cfg
    if media is None:
        logger.warning(
            f"Kimi K3.1 Rubin vision: {_TARGET}: no preprocessor_config.json beside the "
            "checkpoint; the vision tower assumes the processor's default image limits"
        )
    grid = moonvit_max_image_grid(media or {}, _VISION_MERGE, _VISION_PATCH)
    if grid is None:
        raise ValueError(
            f"the image processor limits {media} admit no image grid of the vision tower's merge "
            f"kernel {_VISION_MERGE}"
        )
    patches = grid[0] * grid[1] * grid[2]
    time_rows = torch.from_numpy(_vision_time_rows(_K31_VISION["vt_hidden_size"], _VISION_FRAMES))
    return _VisionW(
        patch_proj=w["v_patch_proj"],
        pos_emb=w["v_pos_emb"],
        blocks=tuple(
            _VisionBlockW(*(w[f"v{j}_{name}"] for name in _VisionBlockW._fields))
            for j in range(_K31_VISION["vt_num_hidden_layers"])
        ),
        final_norm=w["v_final_norm"],
        merge_fc0=w["v_merge_fc0"],
        merge_fc1=w["v_merge_fc1"],
        post_norm=w["v_post_norm"],
        time_weight=time_rows.float().unsqueeze(1).to(dtype=torch.bfloat16, device=device),
        rope_table=moonvit_rope_table(_VISION_HEAD_DIM, TABLE_SIDE, TABLE_SIDE, THETA, device),
        workspace=torch.empty(0, dtype=torch.int8, device=device),
        max_grid=grid,
        capacity=(patches, patches // (_VISION_MERGE[0] * _VISION_MERGE[1])),
    )


def _vision_attention(
    V: _VisionW, B: _VisionBlockW, x: torch.Tensor, freqs_cis: torch.Tensor, seg_lens: list[int]
) -> torch.Tensor:
    """One block's attention: the fused qkv projection, q and k rotated, attention within each
    grid, ``wo``."""
    rows = x.shape[0]
    width = _K31_VISION["qkv_hidden_size"]
    qkv = bf16_linear(x, B.wqkv, False)
    q, k, v = qkv.split([width, width, width], dim=-1)
    q, k = moonvit_rope_2d(
        q.view(rows, _VISION_HEADS, _VISION_HEAD_DIM),
        k.view(rows, _VISION_HEADS, _VISION_HEAD_DIM),
        freqs_cis,
    )
    qkv = torch.concat([q.reshape(rows, -1), k.reshape(rows, -1), v], dim=-1)
    out = torch.empty([rows, width], dtype=x.dtype, device=x.device)
    moonvit_attention(qkv, out, seg_lens, *V.capacity, V.workspace, _VISION_HEADS, _VISION_HEAD_DIM)
    return bf16_linear(out, B.wo, False)


def vision_encode(
    V: _VisionW, pixel_values: torch.Tensor, grids: list[tuple[int, int, int]]
) -> torch.Tensor:
    """``[sum of merged cells, text hidden]`` rows of the patches of ``grids``, grid by grid: the
    generic tower's ``encode``, call for call."""
    x = moonvit_patch_embed(
        pixel_values.to(dtype=torch.bfloat16), V.patch_proj, V.pos_emb, V.time_weight, grids
    )
    freqs_cis = moonvit_rope_angles(V.rope_table, grids)
    seg_lens = [t * h * w for t, h, w in grids]
    for B in V.blocks:
        x = x + _vision_attention(V, B, moonvit_rms_norm(x, B.norm0), freqs_cis, seg_lens)
        y = moonvit_rms_norm(x, B.norm1)
        y = moonvit_gelu(bf16_linear(y, B.fc0, False), "tanh")
        x = x + bf16_linear(y, B.fc1, False)
    x = moonvit_rms_norm(x, V.final_norm)

    # sd2_tpool: each kh x kw merge window of a grid as one row (window row-major, then the window's
    # patches row-major), averaged over the grid's frames.
    kernel_h, kernel_w = _VISION_MERGE
    hidden = x.shape[-1]
    merged = []
    start = 0
    for t, h, w in grids:
        seq = x[start : start + t * h * w]
        windows = seq.view(t, h // kernel_h, kernel_h, w // kernel_w, kernel_w, hidden)
        windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)
        merged.append(windows.view((h // kernel_h) * (w // kernel_w), kernel_h * kernel_w, -1))
        start += t * h * w
    merged = torch.cat([m.reshape(m.shape[0], -1) for m in merged], dim=0)
    out = moonvit_gelu(bf16_linear(merged, V.merge_fc0, False), "none")
    out = bf16_linear(out, V.merge_fc1, False)
    return moonvit_rms_norm(out, V.post_norm, _K31_VISION["projector_ln_eps"])


def _vision_groups(
    grids: list[tuple[int, int, int]], budget: int
) -> list[list[tuple[int, int, int]]]:
    """``grids`` in consecutive groups of at most ``budget`` patches (one group when they fit;
    admission keeps each grid within the budget)."""
    groups: list[list[tuple[int, int, int]]] = []
    size = 0
    for grid in grids:
        patches = grid[0] * grid[1] * grid[2]
        if not groups or size + patches > budget:
            groups.append([])
            size = 0
        groups[-1].append(grid)
        size += patches
    return groups


def vision_forward(V: _VisionW, multimodal_params: Any) -> list[torch.Tensor]:
    """The rows of every image of ``multimodal_params``, request by request, grid by grid, as one
    tensor in a list (empty when no request carries pixels): the generic tower's ``forward``.

    A request's images are one call while their patches fit the encoder budget (the generic tower's
    whole-request call); a request past it is encoded in consecutive groups within the budget.
    """
    rows = []
    for param in multimodal_params:
        image = (param.multimodal_data or {}).get("image")
        if not isinstance(image, dict) or "pixel_values" not in image:
            continue
        grids = [
            (int(t), int(h), int(w))
            for t, h, w in torch.as_tensor(image["image_grid_thw"]).tolist()
        ]
        pixels = image["pixel_values"]
        start = 0
        for group in _vision_groups(grids, V.capacity[0]):
            stop = start + sum(t * h * w for t, h, w in group)
            rows.append(vision_encode(V, pixels[start:stop], group))
            start = stop
    if not rows:
        return []
    return [rows[0] if len(rows) == 1 else torch.cat(rows, dim=0)]


def image_inputs_embeds(
    V: _VisionW | None,
    embed: torch.Tensor,
    input_ids: torch.Tensor,
    multimodal_params: Any,
    mm_token_indices: torch.Tensor | None,
    num_contexts: int,
) -> torch.Tensor | None:
    """This step's prompt embeddings with the image rows its context chunks cover written over their
    placeholders, or None when it covers none (a text step: the prefill target embeds ``input_ids``).

    ``multimodal_params`` are the step's context requests that carry media (the executor's
    ``multimodal_params``); the engine encodes a request once (``vision_forward``), keeps its rows on
    the request and slices those of this chunk (``get_multimodal_embeddings``,
    ``find_input_mm_embeds``). ``mm_token_indices`` are the placeholder positions in ``input_ids``.

    Raises:
        NotImplementedError: If images reach a server without a tower (admission refuses first).
        ValueError: If the placeholder positions do not match the image rows.
    """
    if not multimodal_params:
        return None
    if V is None:
        raise NotImplementedError(_VISION_NO_TOWER)
    context_params = list(multimodal_params[:num_contexts])
    rows = find_input_mm_embeds(
        get_multimodal_embeddings(
            encoder_forward_fn=lambda params: vision_forward(V, params),
            multimodal_params=context_params,
        ),
        context_params,
    )
    if not rows:
        return None
    image_rows = rows[0] if len(rows) == 1 else torch.cat(list(rows))
    if mm_token_indices is None:
        raise ValueError("an image step arrived without the executor's mm_token_indices")
    if mm_token_indices.shape[0] != image_rows.shape[0]:
        raise ValueError(
            f"Multimodal token count mismatch: found {mm_token_indices.shape[0]} image tokens in "
            f"input_ids but received {image_rows.shape[0]} image embeddings."
        )
    embeds = nn.functional.embedding(input_ids, embed)
    embeds.index_copy_(
        0,
        mm_token_indices.to(device=embeds.device, dtype=torch.long, non_blocking=True),
        image_rows.to(dtype=embeds.dtype, device=embeds.device),
    )
    return embeds


def vision_profiling_rows(
    V: _VisionW | None, attn_metadata: AttentionMetadata, has_images: bool
) -> torch.Tensor | None:
    """One largest image's rows in a context step of the KV-capacity estimate without images of its
    own, for the caller to hold through the step; None in every other step.

    The engine sizes the KV pool after the peak memory of its profiling forward, a text-only context
    step: with these rows held the peak counts the tower's activations and an image step's rows. Every
    such step encodes, not only the first (the estimate's warmup steps run under the same cache
    manager, and nothing tells the measured step apart).
    """
    if V is None or has_images or attn_metadata.num_contexts == 0:
        return None
    if not getattr(attn_metadata.kv_cache_manager, "is_estimating_kv_cache", False):
        return None
    t, h, w = V.max_grid
    pixels = torch.zeros(
        t * h * w,
        3,
        _VISION_PATCH,
        _VISION_PATCH,
        dtype=torch.bfloat16,
        device=V.patch_proj.device,
    )
    rows = vision_encode(V, pixels, [V.max_grid])
    logger.info_once(
        f"Kimi K3.1 Rubin vision: {_TARGET}: KV-capacity profiling counts the vision tower's "
        f"activations of one {V.max_grid} image ({rows.shape[0]} rows)",
        key=f"{_TARGET} vision profiling",
    )
    return rows


def vision_admit(V: _VisionW | None, request: Any) -> None:
    """The admission check of one request's processed media (``request.py_multimodal_data``),
    before it is scheduled: the generic target's ``validate_image_request``.

    With a tower, every grid ``(t, h, w)`` must fit the temporal rows, the 2-D rotary table, the
    patch merge and the attention capacity (the processor's largest image), and the pixel payload
    must be exactly its grids' patches. Without one, media is refused unless the request is
    generation-only (its context server encoded the images and sends their KV). A refusal fails this
    request alone; the same condition in the forward would fail the batch.

    Raises:
        ValueError: If the request carries media this server would have to encode and cannot.
    """
    multimodal_data = getattr(request, "py_multimodal_data", None)
    if not multimodal_data:
        return
    if V is None:
        if getattr(request, "is_generation_only_request", False):
            return
        raise ValueError(_VISION_NO_TOWER)
    image = multimodal_data.get("image")
    if not isinstance(image, dict) or image.get("image_grid_thw") is None:
        return
    kernel_h, kernel_w = _VISION_MERGE
    max_patches = V.capacity[0]
    grid_thws = torch.as_tensor(image["image_grid_thw"])
    if grid_thws.ndim != 2 or grid_thws.shape[-1] != 3:
        raise ValueError(
            f"image_grid_thw must have shape [num_grids, 3], got {tuple(grid_thws.shape)}"
        )
    num_patches = 0
    for t, h, w in grid_thws.tolist():
        if (
            not (1 <= t <= _VISION_FRAMES and 1 <= h <= TABLE_SIDE and 1 <= w <= TABLE_SIDE)
            or h % kernel_h
            or w % kernel_w
            or t * h * w > max_patches
        ):
            raise ValueError(
                f"MoonViT cannot encode the image grid (t={t}, h={h}, w={w}); it supports "
                f"t <= {_VISION_FRAMES}, h <= {TABLE_SIDE}, w <= {TABLE_SIDE}, h and w divisible by "
                f"the merge kernel {_VISION_MERGE}, and at most {max_patches} patches (the "
                "processor's largest image)"
            )
        num_patches += t * h * w
    pixel_values = image.get("pixel_values")
    patch_shape = (num_patches, 3, _VISION_PATCH, _VISION_PATCH)
    if isinstance(pixel_values, torch.Tensor) and tuple(pixel_values.shape) != patch_shape:
        raise ValueError(
            f"pixel_values must have shape {patch_shape} for its grids, got "
            f"{tuple(pixel_values.shape)}"
        )


# --------------------------------------------------------------------------------------------------
# Core, the prefill and decode targets, and the shell with the engine's hooks.
#
# The core holds ``self.w`` (the weight table's parameters) and nothing else: no layer, helper or
# backend object. The two targets run the 93 layers as straight-line code over the views: the mix,
# the mixer, the mix, the MLP. The shell builds the DFlash2 drafter and its worker at construction,
# as the DS target builds its MTP worker.
# --------------------------------------------------------------------------------------------------

#: SMs the shared experts' gate_up GEMM leaves beyond MegaMoE's exchange CTAs
#: (shared_gate_up_leave_sms): one per GPC, so a GPC with an odd free SM does not hold back one of
#: the exchange's two-CTA clusters.
_SHARED_GATE_UP_SPARE_SMS = 8


def _shared_gate_up_reserve(role: str, exchange_ctas: int) -> int:
    """SMs the shared experts' gate_up GEMM keeps out of its grid on a server's graph steps: under
    its ``shared_gate_up_leave_sms`` (a generation server's) MegaMoE's exchange CTAs
    (``megamoe_exchange_ctas``) and the spares, else none. On a CUDA graph step the GEMM runs on the
    aux stream beside MegaMoE and holds most SMs when MegaMoE's token exchange launches, so the
    exchange's pushers would wait for it; its tactic and output are the same."""
    if not _CONSTANTS[role].get("shared_gate_up_leave_sms", False):
        return 0
    return exchange_ctas + _SHARED_GATE_UP_SPARE_SMS


# Positions past the configured max_seq_len the identity RoPE table also covers (the KV cache
# adds the draft tokens of a verify block).
_ROPE_HEADROOM = 4096
# The prefill Top-K launches a context server meets first on long chunks. The deterministic GVR
# engine compiles a second variant of a tier's kernel once the logits row stride passes its
# bitmap limit, 8 x 32 x BLK keys (tier 0: up to 148 rows, BLK 1024; tier 1: up to 296 rows,
# BLK 512; tier 2: BLK 256), and the engine's prefill warmup launches row strides of at most
# 33,024. One (rows, row stride) per tier, just past its limit.
_GVR_PREFILL_PRECOMPILE = ((75, 262_400), (149, 131_328), (297, 65_792))


def _one_per_target(name: str, values: Any) -> Any:
    """The value all roles of this target share (one backend class serves them all)."""
    found = set(values)
    assert len(found) == 1, f"the roles of this target disagree on {name}: {found}"
    return found.pop()


# The DSA backend, and so the DSA metadata class, of this target's constants.
_DSA_BACKEND = rubin_dsa_backend_cls(
    kv_lens_refresh_once=_one_per_target(
        "kv_lens_refresh_once",
        (bool(_CONSTANTS[r].get("kv_lens_refresh_once", False)) for r in _ROLES),
    ),
    helix_fused_meta=_one_per_target(
        "helix_fused_meta",
        (bool(_CONSTANTS[r].get("dsa_helix_fused_meta", False)) for r in _ROLES),
    ),
)


def _gvr_prefill_precompile(role: str) -> None:
    """Compile the long-chunk variants of the context server's prefill Top-K before serving.

    One launch of the target's prefill Top-K (``rubin_gvr_topk_prefill``, in the deterministic
    order the indexer runs it) per ``_GVR_PREFILL_PRECOMPILE`` shape, on zero logits with full-row
    windows; the indices are dropped. The kernels, their variants and the engine's choice between
    them are unchanged: a later step launches the variant it would otherwise have compiled itself,
    at its first long chunk. A generation server runs no prefill and skips this. Out of memory
    skips a shape, whose variant then compiles at first use as before. One line per process.
    """
    if role != "context":
        return
    top_k = _K31_DSA["index_topk"]
    t0 = time.perf_counter()
    ran = []
    for rows, stride in _GVR_PREFILL_PRECOMPILE:
        try:
            logits = torch.zeros((rows, stride), dtype=torch.float32, device="cuda")
            starts = torch.zeros((rows,), dtype=torch.int32, device="cuda")
            ends = torch.full((rows,), stride, dtype=torch.int32, device="cuda")
            out = torch.empty((rows, top_k), dtype=torch.int32, device="cuda")
            rubin_gvr_topk_prefill(logits, starts, ends, out)
        except torch.cuda.OutOfMemoryError:
            logger.warning(
                f"Kimi K3.1 Rubin prefill Top-K pre-compile: {_TARGET}: {rows}x{stride} out of "
                "memory; that variant compiles at its first use"
            )
            continue
        ran.append(f"{rows}x{stride}")
    torch.cuda.synchronize()
    logger.info_once(
        f"Kimi K3.1 Rubin prefill Top-K pre-compiled: {_TARGET}: {len(ran)} of "
        f"{len(_GVR_PREFILL_PRECOMPILE)} GVR launches past the bitmap limit ({', '.join(ran)}) "
        f"in {time.perf_counter() - t0:.1f} s",
        key=f"{_TARGET} prefill Top-K pre-compile",
    )


def _server_role() -> str:
    """The server role, ``context`` or ``generation``, as the runtime published it (an aggregated
    server takes the generation role)."""
    role = "context" if disagg_role() == "context" else "generation"
    assert role in _ROLES, f"this target serves {_ROLES}; the runtime published the role {role!r}"
    return role


def _check_build(model_config: ModelConfig) -> None:
    """Check what routing admitted: the GPU, the K3.1 text model, this layout and the recipe;
    raises ``ValueError`` on anything else."""
    cap = torch.cuda.get_device_capability()
    if cap != _COMPUTE_CAPABILITY:
        raise ValueError(f"this target runs on sm_107, not sm_{cap[0]}{cap[1]}")
    cfg = model_config.pretrained_config
    cfg = getattr(cfg, "text_config", None) or cfg
    for key, want in _K31.items():
        if getattr(cfg, key, None) != want:
            raise ValueError(f"{key} {getattr(cfg, key, None)} is not {want}")
    for group, table in (("linear_attn_config", _K31_KDA), ("dsa_config", _K31_DSA)):
        holder = getattr(cfg, group, None)
        for key, want in table.items():
            got = holder.get(key) if isinstance(holder, dict) else getattr(holder, key, None)
            if got != want:
                raise ValueError(f"{group}.{key} {got} is not {want}")
    # The indexer's key norm runs at rms_norm_eps; a checkpoint naming another epsilon is not this
    # recipe's.
    dsa = getattr(cfg, "dsa_config", None) or {}
    eps = (
        dsa.get("index_k_norm_eps")
        if isinstance(dsa, dict)
        else getattr(dsa, "index_k_norm_eps", None)
    )
    if eps is not None and eps != _K31["rms_norm_eps"]:
        raise ValueError(f"dsa_config.index_k_norm_eps {eps} is not {_K31['rms_norm_eps']}")
    m = model_config.mapping
    layout = (m.world_size, m.tp_size, m.pp_size, m.cp_size, bool(m.enable_attention_dp))
    if layout != (_WORLD_SIZE, _WORLD_SIZE, 1, _HELIX_CP, _ATTENTION_DP):
        raise ValueError(f"not {_LAYOUT}: {layout}")
    if (m.moe_tp_size, m.moe_ep_size) != (_MOE_TP, _MOE_EP):
        raise ValueError(
            f"MoE TP x EP {m.moe_tp_size} x {m.moe_ep_size}, not {_MOE_TP} x {_MOE_EP}"
        )
    quant = model_config.quant_config
    if quant.quant_algo != _QUANT_ALGO:
        raise ValueError(f"checkpoint {quant.quant_algo}, not {_QUANT_ALGO}")
    if quant.kv_cache_quant_algo != QuantAlgo.FP8:
        raise ValueError(f"KV {quant.kv_cache_quant_algo}, not FP8")
    state = str(getattr(quant, "mamba_ssm_cache_dtype", None)).removeprefix("torch.")
    if state != str(_STATE_DTYPE).removeprefix("torch."):
        raise ValueError(f"KDA state {state}, not FP32")
    sparse = model_config.sparse_attention_config
    if getattr(sparse, "algorithm", None) != "dsa":
        raise ValueError(f"sparse attention {getattr(sparse, 'algorithm', None)!r}, not DSA")
    if getattr(sparse, "indexer_k_dtype", None) != _INDEXER_K_DTYPE:
        raise ValueError(
            f"indexer_k_dtype {getattr(sparse, 'indexer_k_dtype', None)!r}, not {_INDEXER_K_DTYPE!r}"
        )
    spec = model_config.spec_config
    if spec is None or not spec.spec_dec_mode.is_dflash():
        raise ValueError("DFlash2 is part of the recipe")
    if not spec.spec_dec_mode.use_one_engine():
        raise ValueError("DFlash2 runs one-engine")


def _text_config(model_config: ModelConfig) -> ModelConfig:
    """The engine's config over the checkpoint's text config; quant exclusions named as here."""
    text = copy.copy(model_config)
    text._frozen = False
    cfg = model_config.pretrained_config
    text.pretrained_config = getattr(cfg, "text_config", None) or cfg
    quant = text.quant_config
    if quant is not None and quant.exclude_modules:
        quant = copy.copy(quant)
        names = []
        for name in quant.exclude_modules:
            if name.startswith("language_model."):
                name = name[len("language_model.") :]
                name = "model." + name if name.startswith("layers.") else name
            names.append(name)
        quant.exclude_modules = names
        text.quant_config = quant
    text.skip_create_weights_in_init = True
    text._frozen = True
    return text


def _tap(spec_md: Any, layer: int) -> tuple | None:
    """``(spec_md, layer)`` when the drafter reads layer ``layer``'s output stream, else None."""
    if spec_md is None or layer < 0 or not spec_md.is_layer_capture(layer):
        return None
    return spec_md, layer


def _mix_in(
    rt: _Rt,
    prefix: torch.Tensor,
    pending: Any,
    bank: torch.Tensor,
    mw: _MixW,
    tap,
    mx: bool = False,
):
    """The mix ahead of a mixer; returns ``(prefix, normed)``.

    ``pending``: the previous layer's MLP output, which joins the prefix sum here, in the mix
    kernel where it serves the row count: a ``_MoeOut`` through ``moe_norm``, layer 0's whole
    dense output through ``add_norm``. A tapped layer's stream, the pre-norm mixture, goes to the
    drafter first, so its mix and norm run apart. ``mx`` (a decode step's KDA layer): a
    ``_MoeOut`` goes through ``moe_norm_mx``, so ``normed`` may come as ``(rows, codes, scales)``.
    """
    moe = isinstance(pending, _MoeOut)
    if tap is None:
        if pending is None:
            return prefix, rt.attn_res_norm(prefix, bank, mw)
        if moe:
            return (rt.moe_norm_mx if mx else (rt.moe_fold_norm or rt.moe_norm))(
                prefix, pending, bank, mw
            )
        return rt.add_norm(prefix, pending, bank, mw)
    if pending is not None:
        prefix = rt.moe_add(prefix, pending) if moe else prefix + pending
    mix = rt.attn_res_mix(prefix, bank, mw)
    tap[0].maybe_capture_hidden_states(tap[1], mix, None)
    return prefix, rt.rms_norm(mix, mw.out_norm)


def _mix_out(rt: _Rt, prefix: torch.Tensor, pending: Any, bank: torch.Tensor, head: _HeadW, tap):
    """The output mix after the last layer; returns the final hidden states.

    The last MLP output joins the prefix sum first (``moe_add``), the drafter reads its tap of the
    last layer, then the mix and the final norm run. Untapped with lever moe_fold, ``moe_fold_norm`` runs the
    update inside the mix.
    """
    if tap is None and isinstance(pending, _MoeOut) and rt.moe_fold_norm is not None:
        return rt.moe_fold_norm(prefix, pending, bank, head.out)[1]
    if pending is not None:
        prefix = rt.moe_add(prefix, pending) if isinstance(pending, _MoeOut) else prefix + pending
    if tap is not None:
        tap[0].maybe_capture_hidden_states(tap[1], rt.attn_res_mix(prefix, bank, head.out), None)
    return rt.attn_res_norm(prefix, bank, head.out)


class KimiK31ProNvfp4Sm107Dep16Core(DecoderModel):
    """The K3.1 text model at this rank's share: the weight table, the views and the two targets,
    and the vision tower's views on a server that builds it."""

    def __init__(self, model_config: ModelConfig, vision_config: Any = None) -> None:
        super().__init__(model_config)
        self.role = _server_role()
        # Before the weight table: it lays out the drafter's stacked q_a / kv_a rows.
        self.levers = _levers(self.role)
        self.eps = float(model_config.pretrained_config.rms_norm_eps)
        # The composite config's vision config (None for a text-only config); with it, and without
        # disable_mm_encoder, the table holds the vision tower (Dims.vision).
        self.vision_config = vision_config
        self.dims = _weights.dims(self)
        if self.dims.vision:
            _check_vision(vision_config)
        # The checkpoint processor's image limits, which the shell's load_weights reads.
        self.media_proc_cfg: dict | None = None
        self.w = _weights.declare(self.dims)
        # Made once, never during a capture.
        self.streams = _Streams(*(torch.cuda.Stream() for _ in _Streams._fields))
        max_seq_len = model_config.max_seq_len
        self.rope_positions = (
            int(model_config.pretrained_config.max_position_embeddings)
            if max_seq_len is None
            else int(max_seq_len)
        ) + _ROPE_HEADROOM
        self._layers: tuple | None = None
        self._head: _HeadW | None = None
        self._rt: _Rt | None = None
        self._vision: _VisionW | None = None
        self._prefill: PrefillTarget | None = None
        self._decode: DecodeTarget | None = None

    def post_load_weights(self) -> None:
        """Build the views, the runtime resources and the two targets, once the weights are real.

        Runs once: the shell calls it and so does the loader's walk over the modules.
        """
        if self._prefill is not None:
            return
        self._layers = _layer_views(self)
        self._head = _head_view(self)
        self._rt = _build_runtime(self)
        self._vision = _vision_view(self)
        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)
        log_constants(
            self.role,
            draft_context_kv_only=self.dims.draft_context_kv_only,
        )
        if self._vision is not None:
            logger.info(
                f"Kimi K3.1 Rubin vision: {_TARGET}/{self.role}: vision tower and projector "
                f"loaded, largest image grid {self._vision.max_grid}"
            )
        _gvr_prefill_precompile(self.role)

    def _select_target(self, attn_metadata: AttentionMetadata) -> PrefillTarget | DecodeTarget:
        """Pick this step's target from ``num_contexts`` alone (constant per decode graph)."""
        return self._decode if attn_metadata.num_contexts == 0 else self._prefill

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_metadata: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run this step's target."""
        return self._select_target(attn_metadata).forward(
            attn_metadata, input_ids, position_ids, inputs_embeds, spec_metadata
        )


class PrefillTarget:
    """Steps with context rows, and possibly generation rows after them (a mixed batch)."""

    def __init__(self, core: KimiK31ProNvfp4Sm107Dep16Core) -> None:
        self._layers = core._layers
        self._head = core._head
        self._rt = core._rt

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        spec_metadata: Any,
    ) -> torch.Tensor:
        """Run the layers over the step's rows: one mixer call per layer over every row."""
        rt, head = self._rt, self._head
        rt.dsa_pools.bind(attn_metadata.kv_cache_manager)
        st = _build_step_args(attn_metadata, position_ids, spec_metadata, rt.dsa_pools)
        h = (
            nn.functional.embedding(input_ids, head.embed)
            if inputs_embeds is None
            else inputs_embeds
        )
        bank = h.new_empty(head.n_bank, h.shape[0], h.shape[1])
        prefix, pending = h, None
        for i, L in enumerate(self._layers):
            prefix, x = _mix_in(
                rt, prefix, pending, bank[: L.n_bank], L.pre, _tap(st.spec_md, i - 1)
            )
            if L.boundary:
                bank[L.n_bank].copy_(prefix)
            if L.kda is not None:
                o = kda_prefill(L.kda, x, st, rt)
            else:
                o = mla_prefill(L.mla, i, x, st, rt)
            if L.boundary:
                prefix = rt.reduce(o)
                x = rt.attn_res_norm(prefix, bank[: L.n_bank + 1], L.mid)
            else:
                prefix, x = rt.add_norm(prefix, rt.reduce(o), bank[: L.n_bank], L.mid)
            if L.moe is not None:
                pending = latent_moe(L.moe, i, x, st, rt)
            else:
                pending = dense_mlp(L.dense, x, st, rt)
        return _mix_out(rt, prefix, pending, bank, head, _tap(st.spec_md, len(self._layers) - 1))


class DecodeTarget:
    """Generation-only steps; it states the context counts as 0 instead of reading them."""

    def __init__(self, core: KimiK31ProNvfp4Sm107Dep16Core) -> None:
        self._layers = core._layers
        self._head = core._head
        self._rt = core._rt

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        spec_metadata: Any,
    ) -> torch.Tensor:
        """Run the layers over the generation rows (one verify block per request with DFlash2)."""
        rt, head = self._rt, self._head
        rt.dsa_pools.bind(attn_metadata.kv_cache_manager)
        st = _build_step_args(attn_metadata, position_ids, spec_metadata, rt.dsa_pools, decode=True)
        h = (
            nn.functional.embedding(input_ids, head.embed)
            if inputs_embeds is None
            else inputs_embeds
        )
        kda = kda_verify if st.rows_per_gen > 1 else kda_decode
        bank = h.new_empty(head.n_bank, h.shape[0], h.shape[1])
        prefix, pending = h, None
        for i, L in enumerate(self._layers):
            # A KDA layer's projection takes the MXFP8 form of its input from the mix.
            prefix, x = _mix_in(
                rt,
                prefix,
                pending,
                bank[: L.n_bank],
                L.pre,
                _tap(st.spec_md, i - 1),
                L.kda is not None,
            )
            if L.boundary:
                bank[L.n_bank].copy_(prefix)
            if L.kda is not None:
                o = kda(L.kda, x, st, rt)
            else:
                o = mla_generate(L.mla, i, x, st, rt)
            # A MoE layer's mix may store the router's FP32 input too (lever router_f32).
            f32 = rt.mid_norm_f32 if L.moe is not None else None
            xf = None
            if L.boundary:
                prefix = rt.reduce(o)
                if f32 is None:
                    x = rt.attn_res_norm(prefix, bank[: L.n_bank + 1], L.mid)
                else:
                    prefix, x, xf = f32(prefix, None, bank[: L.n_bank + 1], L.mid)
            elif f32 is None:
                prefix, x = rt.reduce_norm(prefix, o, bank[: L.n_bank], L.mid)
            else:
                prefix, x, xf = f32(prefix, rt.reduce(o), bank[: L.n_bank], L.mid)
            if L.moe is not None:
                pending = latent_moe(L.moe, i, x, st, rt, xf)
            else:
                pending = dense_mlp(L.dense, x, st, rt)
        return _mix_out(rt, prefix, pending, bank, head, _tap(st.spec_md, len(self._layers) - 1))


class ModelingV2KimiK31ProNvfp4Sm107Dep16(
    DecoderModelForCausalLM[KimiK31ProNvfp4Sm107Dep16Core, PretrainedConfig]
):
    """Kimi K3.1 Pro NVFP4, sm_107, dep16: the shell and the engine hooks.

    The spec decoding wiring is the DS target's: the shell holds the drafter container
    (``draft_model``) and the engine's worker (``spec_worker``, a submodule the loader walks),
    and its forward runs the trunk, the target logits and the worker. The multimodal surface is
    the generic target's: ``multimodal_token_ids``, ``multimodal_data_device_paths`` and the
    request check ``validate_request``; the forward splices a context step's image rows ahead of
    the trunk (the vision section).
    """

    # The engine hooks every Kimi K3.1 target declares (``runtime.py``), with the Rubin KDA + DSA
    # cache manager (``rubin_runtime_hooks``) and this target's DSA backend.
    runtime_hooks = rubin_runtime_hooks(_DSA_BACKEND, DraftModel)

    # The family KDA metadata class the executor builds.
    mamba_metadata_cls = RubinKdaMetadata

    def __init__(self, model_config: ModelConfig) -> None:
        _check_build(model_config)
        composite = model_config.pretrained_config
        text = _text_config(model_config)
        cfg = text.pretrained_config
        core = KimiK31ProNvfp4Sm107Dep16Core(text, getattr(composite, "vision_config", None))
        super().__init__(core, config=text, hidden_size=cfg.hidden_size, vocab_size=cfg.vocab_size)
        # The in-vocabulary media placeholder (<|media_pad|>) the input processor repeats per row.
        self._mm_token_id = int(
            getattr(composite, "media_placeholder_token_id", MEDIA_PLACEHOLDER_TOKEN_ID)
        )
        # The executor reads the generation parameters (eos_token_id, ...) off the text config.
        frozen, model_config._frozen = model_config._frozen, False
        model_config.pretrained_config = cfg
        model_config._frozen = frozen
        # The cache manager the executor builds later reads these (extra_init_kwargs).
        bind_kv_cache_args(text, **_CACHE_ARGS[core.role])
        self.spec_config = text.spec_config
        self.draft_config = draft_config_for(text)
        self.draft_model = DraftModel(core, self.draft_config, self.lm_head, self.logits_processor)
        self.spec_worker = DFlashWorker(
            self.spec_config,
            text.mapping.mapping_for_full_rows(),
            should_use_separate_draft_kv_cache(self.spec_config),
        )
        self.spec_worker.set_draft_model(self.draft_model)
        self.spec_worker.set_target_model(self)

    @classmethod
    def get_model_defaults(cls, llm_args: Any) -> dict:
        """Return the engine defaults this target needs: 64-token KV pages, block reuse opt-in, and DSA
        sparse attention with the checkpoint's indexer (``_K31_DSA``): the family's
        ``config.dsa_model_defaults``, written out."""
        return {
            "kv_cache_config": {"enable_block_reuse": False, "tokens_per_block": 64},
            "sparse_attention_config": {
                "algorithm": "dsa",
                "index_n_heads": 32,
                "index_head_dim": 128,
                "index_topk": 2048,
            },
        }

    @classmethod
    def requires_kv_cache_manager_v2(cls, pretrained_config: Any = None) -> bool:
        """Say whether the target needs KV cache manager V2: the DSA index keys live in it."""
        if pretrained_config is None:
            return False
        text = getattr(pretrained_config, "text_config", None) or pretrained_config
        return getattr(text, "dsa_config", None) is not None

    @classmethod
    def get_preferred_transceiver_runtime(cls, pretrained_config: Any = None) -> Any:
        """Return the KV transceiver runtime the target needs: only Python NIXL moves KDA state."""
        return "PYTHON"

    @property
    def multimodal_token_ids(self) -> torch.Tensor:
        """The in-vocabulary media placeholder id, from which the executor finds a context step's
        image positions (``mm_token_indices``)."""
        return torch.tensor([self._mm_token_id], dtype=torch.int32)

    @property
    def multimodal_data_device_paths(self) -> list[str]:
        """The processed media the executor moves to the GPU (the images under ``image``)."""
        return list(_VISION_DEVICE_PATHS)

    def validate_request(self, request: Any) -> None:
        """Refuse, before it is scheduled, media this server would have to encode and cannot
        (``vision_admit``): that request fails alone, where the forward would fail the batch."""
        vision_admit(self.model._vision, request)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        return_context_logits: bool = False,
        spec_metadata: Any = None,
        resource_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Run one engine step: the core, the target logits and the DFlash2 worker.

        ``position_ids`` reaches the worker in the engine's ``[1, T]`` shape. A context step whose
        requests carry images (the executor's ``multimodal_params``) hands the core the prompt
        embeddings with their rows spliced in (``image_inputs_embeds``); the worker keeps the ids.
        A text step hands the core ``inputs_embeds`` as it came (None).
        """
        vision = self.model._vision
        if inputs_embeds is None:
            inputs_embeds = image_inputs_embeds(
                vision,
                self.model._head.embed,
                input_ids,
                kwargs.get("multimodal_params"),
                kwargs.get("mm_token_indices"),
                attn_metadata.num_contexts,
            )
        # Held through the step: a KV-capacity estimate's context step encodes one largest image.
        _profiling_rows = vision_profiling_rows(vision, attn_metadata, inputs_embeds is not None)
        hidden = self.model(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
        )
        rows = attn_metadata.num_tokens if attn_metadata.padded_num_tokens is not None else None
        if rows is not None:
            hidden = hidden[:rows]
            input_ids = input_ids[:rows] if input_ids is not None else None
            position_ids = position_ids[..., :rows] if position_ids is not None else None
        logits = self.spec_worker.target_logits(
            hidden[spec_metadata.gather_ids],
            self.lm_head,
            self.logits_processor,
            attn_metadata,
            spec_metadata,
            self.draft_model,
        )
        return self.spec_worker(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden,
            logits=logits,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=self.draft_model,
            resource_manager=resource_manager,
        )

    def load_weights(self, weights: Any, *args: Any, **kwargs: Any) -> None:
        """Fill the parameters from the target checkpoint (weights.py), and with the vision tower,
        read the checkpoint processor's image limits (its largest image grid)."""
        _weights.load(self, weights)
        if self.model.dims.vision:
            self.model.media_proc_cfg = _weights.media_proc_cfg(weights)

    def load_draft_weights(self, weights: Any, weight_mapper: Any = None) -> None:
        """Fill the drafter from its checkpoint (weights.py)."""
        _weights.load_draft(self, weights, weight_mapper)

    def set_guided_decoder(self, guided_decoder: Any) -> bool:
        """Hand the engine's capturable guided decoder to the DFlash worker, which masks the gathered
        target logits before its acceptance; False without a worker."""
        worker = getattr(self, "spec_worker", None)
        return worker is not None and worker.set_guided_decoder(guided_decoder)

    def post_load_weights(self) -> None:
        """Run the shell's and then the core's post-load steps."""
        super().post_load_weights()
        self.model.post_load_weights()


# --------------------------------------------------------------------------------------------------
# The launch-constants line: ``log_constants(role)`` logs one INFO line at startup with the
# role's ``_CONSTANTS`` (a ``_DETAILS`` key inside its parent's value), so a run records the
# constants it used.
# --------------------------------------------------------------------------------------------------


def constants_line(role: str, *, draft_context_kv_only: bool | None = None) -> str:
    """The startup constants line of ``role``; the drafter's load fact overrides the expectation."""
    table = dict(_CONSTANTS[role])
    if draft_context_kv_only is not None and "draft_ctx_weights" in table:
        table["draft_ctx_weights"] = "history" if draft_context_kv_only else "full"
    details: dict[str, list[str]] = {}
    for key in sorted(table):
        parent = _DETAILS.get(key)
        if parent is not None and parent in table:
            details.setdefault(parent, []).append(f"{key}={table[key]}")
    items = []
    for key in sorted(table):
        if _DETAILS.get(key) in table:
            continue
        extra = details.get(key)
        items.append(f"{key}={table[key]}" + (f"({','.join(extra)})" if extra else ""))
    return f"Kimi K3.1 Rubin constants: {_TARGET}/{role} ({len(items)}): " + ", ".join(items)


def log_constants(role: str, *, draft_context_kv_only: bool | None = None) -> None:
    """Log this target's launch constants for ``role`` in one line."""
    logger.info(constants_line(role, draft_context_kv_only=draft_context_kv_only))
