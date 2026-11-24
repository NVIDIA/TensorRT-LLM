# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: Kimi K3.1 Pro NVFP4 (ModelOpt) / sm_107 / helix8 -- helix MLA over 8 ranks, MoE TP2 x EP4.

One recipe and one layout, so the forward carries no recipe or layout branch:

* checkpoint: the ModelOpt NVFP4 export as stored. The routed experts are NVFP4 (per-tensor
  input scale 1.0); the shared experts and the KDA projections are MXFP8 (one UE8M0 scale per 32
  values of a row); MLA, the latent MoE projections, the router, norms, embedding and LM head are
  BF16. Nothing is quantized at load time.
* caches: FP8 (E4M3, scale 1) MLA latent KV on all 24 DSA layers, FP8 index keys, FP32 KDA
  recurrent state.
* speculative decoding: DFlash2, with an FP8 drafter context history.
* layout: MLA with helix context parallelism over 8 ranks (each rank attends with all 96
  query heads over its eighth of every request's latent KV), the KDA heads split 8 ways
  (12 per rank), the routed experts TP2 x EP4 (224 half-width experts per rank).
  Generation servers only.
* images: none to encode. A generation server of a disaggregated deployment receives an image
  prompt's KV from its context server (which encoded the images: the dep16 target) together with
  the prompt ids, whose media placeholders are ordinary in-vocabulary ids at ordinary positions,
  and its requests carry no media (the generation-only frontend does not resolve them). The
  checkpoint's vision tower and projector are expected leftovers of the load; admission refuses
  media on a request that would need a context step (``validate_request``).

Construction binds layout, role and recipe (module constants, closures, helper arguments); the
forward dispatches only on per-step facts (row counts, whether a step has context rows). Every
model kernel call goes through an OpWrapper, from modeling_v2's ``catalog/`` or the family's
``ops/``; small torch glue (argmax, where, cat, gather, masked_fill, sum, repeat_interleave) runs
inline.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import math
import weakref
from types import MappingProxyType
from typing import Any, Callable, NamedTuple, Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.flashinfer_silu_and_mul import (
    flashinfer_silu_and_mul,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.mla_rope_generation import (
    mla_rope_generation,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
    thop_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.bmm_out import bmm_out
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.noaux_tc_op import noaux_tc_op
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (
    flashinfer_fused_add_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.distributed.ops import get_or_scale_allreduce_mnnvl_workspace
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_speculative import external_drafter_config_kwargs
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.pyexecutor.sampler.ops import flashinfer as flashinfer_sampling
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import GREEDY_TEMPERATURE_THRESHOLD
from tensorrt_llm._torch.speculative import should_use_separate_draft_kv_cache
from tensorrt_llm._torch.speculative.dflash import DFlashWorker as _EngineDFlashWorker
from tensorrt_llm._torch.speculative.interface import (
    _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
    _FORCE_ACCEPT_RNG_SLOT_STRIDE,
    SpecWorkerBase,
)
from tensorrt_llm._torch.speculative.spec_tail_kernels import (
    DraftBlockPlacement,
    dflash_paged_latent_prepare,
    store_context_latents,
)
from tensorrt_llm._torch.speculative.vocab_parallel import (
    _LOCAL_BLOCK_V,
    VocabShardedLogits,
    _local_argmax_kernel,
)
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.functional import RopeEmbeddingUtils
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.tools.layer_wise_benchmarks.layer_truncation import original_num_hidden_layers

from ..ops.activation.rubin_situ_and_mul_mxfp8 import rubin_situ_and_mul_mxfp8
from ..ops.activation.sigmoid_gate_mul import k31_sigmoid_gate_mul
from ..ops.activation.situ_and_mul import situ_and_mul
from ..ops.attention.convert_req_index_to_global import convert_req_index_to_global
from ..ops.attention.fp8_paged_mqa_logits import fp8_paged_mqa_logits
from ..ops.attention.fused_cat_fp8 import fused_cat_fp8
from ..ops.attention.indexer_k_cache_scatter_op import indexer_k_cache_scatter_op
from ..ops.attention.indexer_topk_decode import indexer_topk_decode
from ..ops.attention.rubin_dsa_backend import attention_backend_for as rubin_attention_backend_for
from ..ops.attention.rubin_dsa_backend import rubin_dsa_backend_cls
from ..ops.attention.rubin_dsa_key_shard import (
    rubin_dsa_key_shard_merge,
    rubin_dsa_key_shard_pack,
    rubin_dsa_key_shard_ws,
)
from ..ops.attention.rubin_dsa_key_shard_select_pack import PAGE_TOKENS as _KS_SELECT_PAGE
from ..ops.attention.rubin_dsa_key_shard_select_pack import (
    key_shard_kv_rows,
    key_shard_select_check,
    rubin_dsa_key_shard_select_pack,
    rubin_dsa_key_shard_select_ws,
)
from ..ops.attention.rubin_dsa_topk_remap import rubin_dsa_topk_remap
from ..ops.attention.rubin_fp8_paged_mqa_logits import rubin_fp8_paged_mqa_logits
from ..ops.attention.rubin_gvr_topk_decode import rubin_gvr_topk_decode
from ..ops.attention.rubin_helix_zero_or_empty_rows import rubin_helix_zero_or_empty_rows
from ..ops.attention.rubin_idx_weights import rubin_idx_weights
from ..ops.attention.rubin_merge_helix_topk import rubin_merge_helix_topk
from ..ops.attention.rubin_topk_decode_cute import rubin_topk_decode_cute
from ..ops.comm.allgather_module import allgather_module
from ..ops.comm.allreduce_module import allreduce_module
from ..ops.comm.rubin_comm import rubin_all_reduce, rubin_latent_ar_workspace
from ..ops.comm.rubin_lamport_push import rubin_lamport_finish, rubin_lamport_push
from ..ops.comm.rubin_latent_ar_norm import rubin_latent_ar_norm_row
from ..ops.comm.rubin_push_allreduce import (
    rubin_lamport_push_workspace,
    rubin_lamport_row_push_desc,
)
from ..ops.gemm.low_m_linear import k31_low_m_linear, plan_choice
from ..ops.gemm.low_m_plans import K31_LOW_M_DRAFT_PLAN_HELIX8, K31_LOW_M_DRAFT_PLAN_HELIX8_D
from ..ops.gemm.rubin_l2_prefetch import rubin_l2_prefetch
from ..ops.gemm.rubin_latent_up_resident import rubin_latent_up_resident
from ..ops.gemm.rubin_linear import rubin_linear
from ..ops.gemm.rubin_lt_gemm import LtPin, LtUnavailable, rubin_lt_gemm
from ..ops.gemm.rubin_mxfp8_gemm import mxfp8_scales_from_r128c4
from ..ops.gemm.rubin_mxfp8_linear import (
    mxfp8_act_layout,
    mxfp8_quantizes_in_gemm,
    rubin_mxfp8_linear,
    rubin_mxfp8_linear_prequantized,
)
from ..ops.gemm.rubin_mxfp8_linear_push import rubin_mxfp8_linear_prequantized_push
from ..ops.gemm.rubin_ull_bf16_gemm_push import bf16_gemm_push, bf16_gemm_push_values
from ..ops.gemm.rubin_upcast_fp32 import rubin_upcast_fp32
from ..ops.moe.dsv3_router_gemm_op import dsv3_router_gemm_op
from ..ops.moe.rubin_fc12 import check_fc12_lever_env, rubin_fc12, rubin_fc12_place_per_die
from ..ops.moe.rubin_moe_residual_add import rubin_moe_residual_add
from ..ops.moe.rubin_moe_sort import rubin_moe_sort
from ..ops.moe.rubin_route_tail import rubin_route_tail
from ..ops.moe.rubin_router_logits import rubin_router_logits
from ..ops.moe.rubin_ull_cp_tail import (
    rubin_ull_cp_tail,
    rubin_ull_cp_tail_share,
    rubin_ull_tail_workspace,
)
from ..ops.norm.attn_res_add_rmsnorm_fwd import attn_res_add_rmsnorm_fwd
from ..ops.norm.attn_res_fwd import attn_res_fwd
from ..ops.norm.attn_res_rmsnorm_fwd import attn_res_rmsnorm_fwd
from ..ops.norm.rubin_attn_res_add_rmsnorm_persistent import rubin_attn_res_add_rmsnorm_persistent
from ..ops.norm.rubin_attn_res_add_rmsnorm_persistent_x import (
    persistent_x_available,
    rubin_attn_res_add_rmsnorm_persistent_x,
)
from ..ops.norm.rubin_attn_res_lamport_add_rmsnorm import (
    rubin_attn_res_lamport_add_rmsnorm,
    rubin_attn_res_lamport_add_rmsnorm_mx,
)
from ..ops.norm.rubin_attn_res_moe_add_rmsnorm import (
    rubin_attn_res_moe_add_rmsnorm,
    rubin_attn_res_moe_add_rmsnorm_mx,
)
from ..ops.norm.rubin_attn_res_rmsnorm import rubin_attn_res_add_rmsnorm, rubin_attn_res_rmsnorm
from ..ops.norm.rubin_layer_norm import rubin_layer_norm
from ..ops.spec.dflash2_block_verify import dflash2_block_verify_seeded
from ..ops.spec.dflash2_conv_add_rmsnorm import dflash2_conv_add_rmsnorm, dflash2_rmsnorm
from ..ops.spec.dflash2_global_top32 import (
    k31_dflash2_merge_top32,
    k31_dflash2_shard_index_pairs,
    k31_dflash2_shard_top32,
)
from ..ops.spec.dflash2_greedy_verify import dflash2_greedy_verify
from ..ops.spec.dflash2_grouped_conv import dflash2_grouped_conv
from ..ops.spec.dflash2_proposal_load import k31_dflash2_proposal_load
from ..ops.spec.dflash2_sampling import (
    DFlash2ProposalStore,
    dflash2_sampled_step,
    invalidate_dflash2_proposal_slot,
)
from ..ops.spec.dflash2_selector_draft import k31_dflash2_selector_draft
from ..ops.spec.dflash2_selector_greedy_path import k31_dflash2_selector_greedy_path
from ..ops.spec.dflash2_selector_greedy_tokens import k31_dflash2_selector_greedy_tokens
from ..ops.spec.dflash2_step_glue import k31_dflash2_noise_block
from ..ops.spec.draft_context_cache import DRAFT_FP8_KV_SCALE
from ..ops.spec.flashinfer_compute_probs_from_logits import flashinfer_compute_probs_from_logits
from ..ops.spec.flashinfer_sample_from_logits import flashinfer_sample_from_logits
from ..ops.spec.fused_compute_probs_from_logits import fused_compute_probs_from_logits
from ..ops.spec.fused_sample_from_logits import fused_sample_from_logits
from ..ops.spec.mla_norm_rope import dflash2_mla_block_rope, dflash2_mla_ctx_norm_rope
from ..ops.spec.rubin_dflash2_helix_combine import rubin_dflash2_helix_combine
from ..ops.spec.rubin_dflash2_window_plan import plan_buffers, rubin_dflash2_window_plan
from ..ops.spec.rubin_dflash2_window_triton import (
    rubin_dflash2_window_triton,
    window_vec_io_aligned,
    window_workspace,
)
from ..ops.spec.rubin_ull_dflash2_helix_window import (
    helix_window_workspace,
    rubin_ull_dflash2_helix_window,
)
from ..ops.spec.rubin_ull_next_new_tokens_rewind import draft_next_new_tokens_rewind
from ..ops.spec.rubin_ull_select_accept import select_accept
from ..ops.spec.strict_accept import strict_accept
from ..ops.spec.vocab_parallel_argmax import vocab_parallel_argmax
from ..ops.spec.vocab_parallel_sample import vocab_parallel_sample
from ..ops.ssm.kda_decode import kda_decode as kda_decode_op
from ..ops.ssm.rubin_kda_dsa_cache_manager import (
    KDAReplayState,
    KimiK3HybridCacheManager,
    bind_kv_cache_args,
    select_rubin_kv_cache_manager_cls,
)
from ..ops.ssm.rubin_kda_metadata import RubinKdaMetadata
from ..ops.ssm.rubin_kda_mtp_verify import rubin_kda_mtp_verify
from . import weights as _weights

# Identity. The server role is the one routing records (``routing.disagg_role``), fixed at
# construction; the target itself reads no environment variable.
_TARGET = "kimi_k31_pro_nvfp4__sm_107__helix8"
_COMPUTE_CAPABILITY = (10, 7)
_LAYOUT = "helix8"
_ROLES = ("generation",)

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

# The layout; the core asserts the Mapping matches.
_WORLD_SIZE = 8
_ATTENTION_DP = False
_HELIX_CP = 8
_MLA_QUERY_HEADS = 96
_KDA_TP = 8
_MOE_TP = 2
_MOE_EP = 4

# Launch constants per server role, logged in one line at startup; a key in _DETAILS is
# logged inside its parent's value.
# f_a and b run in the QKVG GEMM, so no [f_a | b] -> f_b chain is left for the aux stream
# (kda_bfa_aux_stream_max_rows 0). The shared experts get no SM cap (shared_sm 0). The PDL mix
# after a pushed o_proj partial finishes its all-reduce before the grid wait
# (oproj_ar_early_poll). The attention-residual epilogue levers act on the persistent kernel's
# rows only (above attn_res_pdl_max_m) and are on: attn_res_mx_moe (a KDA layer's mix folds the
# MoE residual update and stores the MXFP8 form the QKVG GEMM multiplies), attn_res_moe_fold
# (every other MoE-fed mix and the output mix fold the update), attn_res_grid_cap (the persistent
# mixes launch min(SMs, rows) CTAs); dsa_idx_logits_pdl launches the decode indexer logits with PDL.
# attn_res_mx_pdl acts on the PDL kernel's rows (up to one 128-row scale tile) where the QKVG
# GEMM would quantize them in a kernel of its own (_kda_mx_pdl_rows: past the low-M GEMM's
# in-kernel rows): a KDA layer's fused MoE mix also stores the MXFP8 form of its normed rows, the
# codes and R128c4 scales the QKVG GEMM's own quantization makes of them (_kda_mx_exact), so the
# GEMM multiplies them and its quantization kernel does not run.
# A key-shard merge row whose certificate fails is re-selected from every rank's local logits, or
# from the failing ranks' logits and the other ranks' candidates (dsa_key_shard_failing_fallback:
# the same rows, fewer remote reads). dsa_ks_select_pack (on): after the logits, one launch of
# helix16's select / pack / merge (LL exchange, the ties_exact certificate and rescue) writes the
# kept KV pool rows, counts and mask in place of the local Top-K, the pack, the merge and the remap,
# its local select on min(dsa_ks_select_cluster_max, SMs // rows) CTAs per row, sized per step (1:
# the one-kernel exchange, which measured fastest at 16.5k local keys; more: the cluster select and
# a separate merge grid); at a row count's first use the process compares it word for word with
# today's chain on the step's logits and on tie-heavy and forced-fallback forms of them (every rank
# must agree), and keeps today's chain where they differ (``_ks_select``).
# The drafter window levers are on: draft_window_vec_io moves
# the window kernels' rows as 16-byte vectors where every stride allows (the same bytes, the same
# arithmetic); draft_window_pdl launches the window's tail as a PDL dependent of its body, waiting
# for the body only before its first read of the body's partials. A gemm_pins_* group (on) runs
# its sites' BF16 GEMMs at a pinned cuBLASLt algorithm at the row counts measured for them, once the
# process checked each on the device; see ``_GEMM_PIN_TABLE``. gemm_pins_a_class (off) adds the
# A-class pins (another summation order) of ``_GEMM_PIN_TABLE_A`` to the groups that are on.
# kda_verify_pdl launches the KDA verify as a PDL dependent of the f_b GEMM where its v2 kernel
# serves the rows (one wave of (head, request) tiles): the state, replay entries and conv windows
# load before the grid wait (the same kernel and epilogue, so the same bytes).
# kda_verify_fused_fb (with kda_verify_pdl, on the same rows): the v2 verify computes the
# forget-gate logits from the f_a columns of the fused projection in its kernel (fa / w_fb), so the
# f_b GEMM does not run and the verify launches as the QKVG GEMM's PDL dependent. A row count runs
# it once the process checked it on the device: the first eager verify at that row count runs both
# forms on the same pools, and only output and pool words equal to today's let the fused form run
# from then on; otherwise, and on a capture before the check, f_b and the verify run as today.
# moe_latent_up_resident: on a CUDA graph's decode step whose rows fall in
# moe_latent_up_resident_bands, latent_up behind the latent all-reduce row kernel (own workspace)
# is a cluster split-K GEMM whose CTAs load their share of the weight into shared memory before
# their grid wait, while the all-reduce waits for the other ranks; rank 0 of each cluster adds the
# three ranks' FP32 partials in rank order (another summation order than cuBLAS's, one BF16
# rounding either way).
# moe_cp_tail: on a decode step whose rows fall in moe_cp_tail_bands (``lo-hi`` row bands, at most
# 72 rows), latent_up runs column-parallel over the MoE TP8 group: each rank multiplies the whole
# normalized latent by its 896 output columns and pushes them into the all-gather rows of a Lamport
# call that also carries the shared experts' partials (pushed in place of their all-reduce); the
# next mix finishes the call (bf16(latent_up + bf16(shared)), the round-to-nearest residual add),
# polling it before its grid wait with moe_tail_early_poll, and a KDA layer's mix stores the MXFP8
# form under attn_res_mx_pdl as the replicated tail's mix does.
# Each latent_up row count has at most one owner: moe_cp_tail's bands, moe_latent_up_resident's
# bands and gemm_pins_moe's latent_up rows are disjoint for the levers that are on, or the import
# fails (``_MOE_LATENT_UP_OWNERS``); rows no lever owns run cuBLAS's latent_up.
# The KDA o_proj hand-off levers: the v2 verify releases o_proj right after its grid wait on a PDL
# launch (kda_decode_early_trigger with kda_verify_pdl), so that o_proj (the shared low-M / mid-M
# MXFP8 GEMM, whose weight warp streams its whole weight slice before its grid wait) stages its
# weights during the recurrence. kda_oproj_push: on reduce_norm's push-path rows (up to
# attn_res_pdl_max_m on a non-boundary layer with snapshot rows) o_proj's epilogue stores its
# partial into the Lamport slots of the all-reduce, no push copy kernel; the mix finishes the call
# as before. Scheduling and the store's destination only: the same bytes.
# dsa_oproj_push (on): on the same push-path rows, within dsa_oproj_push_bands (``lo-hi`` row
# bands, at most 72 rows), a DSA layer's o_proj runs as the BF16 split-K GEMM whose epilogue
# stores its partial into the all-reduce's Lamport slots (no push copy kernel), in one wave of
# CTAs, at the row counts where that GEMM's output equals today's o_proj word for word
# (checked once per process and row count on the device, as the GEMM pins are; otherwise today's
# GEMM and push run); the mix finishes the call as before. The same bytes.
# The mix-node levers (mix_bank_in_mix and boundary_ar_push on, draft_tap_stream and
# dense_ar_push off; each the same bytes): mix_bank_in_mix stores a block boundary's bank
# snapshot from the producer of its row instead of a copy (the boundary layer's mix stores its
# updated prefix into the bank row where its kernel serves the rows; layer 0's embedding gathers
# into bank row 0). draft_tap_stream runs the drafter's capture copies on a stream of their own
# (the tap stream): it forks from the mix that makes the tapped stream (whose memory it keeps
# alive on the tap stream) and joins where the drafter first reads the captures (the early
# projection on the aux stream, else the current stream). boundary_ar_push: on a KDA verify step's
# block boundary (1 to attn_res_pdl_max_m rows), o_proj's epilogue pushes its partial and the mix
# after the attention finishes the all-reduce as the new prefix sum (``replace``) instead of the C++
# all-reduce and rubin_attn_res_rmsnorm; each row count and bank size is checked on the device at
# its first eager call (both paths on the same partial, the verdict agreed over the group) and
# keeps today's calls if a word differs or a capture comes first. It finishes the partial the KDA
# verify's o_proj epilogue pushes, so it needs kda_oproj_push and kda_oproj_quant_k32, or the
# import fails. dense_ar_push: layer 0's dense MLP output is pushed by the next mix's push path
# (reduce_norm) instead of its C++ all-reduce and the add-and-mix slot.
# moe_shared_late (on): on a CUDA graph's decode step whose rows fall in moe_shared_late_bands
# (``lo-hi`` row bands within moe_cp_tail's), the shared experts fork after the routing tail
# instead of before it, on a stream below every other stream of the step: the step's layers and
# their side streams run on streams of the highest priority (``_LateStreams``), the shared experts
# on the core's default-priority aux stream, so FC12's CTAs take the SMs first and the shared
# experts fill the SMs FC12 leaves. Scheduling only: the same kernels on the same inputs, so the
# same bytes. The next mix still waits for the shared experts' push (``shared_join``); the bands
# keep to the rows where their chain ends well before the column-parallel tail's push GEMM.
_CONSTANTS = MappingProxyType(
    {
        "generation": MappingProxyType(
            {
                "ar_finish_fused": True,
                "ar_push_amid": True,
                "attn_res_early_trigger": True,
                "attn_res_grid_cap": True,
                "attn_res_moe_fold": True,
                "attn_res_mx_moe": True,
                "attn_res_mx_pdl": True,
                "attn_res_pdl": 2,
                "attn_res_pdl_max_m": 128,
                "attn_res_topology": 1,
                "boundary_ar_push": True,
                "dense_ar_push": False,
                "draft_agree_check": False,
                "draft_agree_zero_padded": True,
                "draft_block_rope": "dflash2_mla_block_rope",
                "draft_block_rows": False,
                "draft_conv_norm": True,
                "draft_conv_pdl": False,
                "draft_ctx_latent": "precompute_context_kv",
                "draft_greedy_tokens": True,
                "draft_history_aux": False,
                "draft_history_dtype": "fp8",
                "draft_index_pairs": True,
                "draft_lowm_gemm": True,
                "draft_lowm_gemm_d_class": True,
                "draft_mla_norm_rope": "mla_norm_rope",
                "draft_noise_block": True,
                "draft_norm_ctas": 7,
                "draft_pairs_zero_padded": True,
                "draft_q_a_kv_a_bands": "1-78",
                "draft_q_a_kv_a_fused": True,
                "draft_select_accept": True,
                "draft_tail_rewind": True,
                "draft_tap_stream": False,
                "draft_window_mla": True,
                "draft_window_mla_bands": "1-30,70-120",
                "draft_window_pdl": True,
                "draft_window_vec_io": True,
                "dsa_fork_q": True,
                "dsa_helix_fused_meta": True,
                "dsa_helix_key_shard": True,
                "dsa_helix_mask_fold": True,
                "dsa_idx_fork": True,
                "dsa_idx_logits_pdl": True,
                "dsa_key_shard_failing_fallback": True,
                "dsa_ks_select_cluster_max": 1,
                "dsa_ks_select_pack": True,
                "dsa_oproj_push": True,
                "dsa_oproj_push_bands": "1-72",
                "dsa_single_atom": True,
                "fc12_mma_n": 256,
                "fc12_prezero": True,
                "fuse_moe_residual": True,
                "gemm_pins_a_class": False,
                "gemm_pins_dense": True,
                "gemm_pins_draft": True,
                "gemm_pins_mla": True,
                "gemm_pins_moe": True,
                "index_k_dtype": "fp8",
                "index_k_evict_first": True,
                "indexer_logits_elem_budget": 536870912,
                "kda_bfa_aux_stream_max_rows": 0,
                "kda_decode_early_trigger": True,
                "kda_decode_fused_fab": False,
                "kda_l2_prefetch": True,
                "kda_oproj_push": True,
                "kda_oproj_quant_k32": True,
                "kda_qkvg_fused": True,
                "kda_replay_seed_stacked": True,
                "kda_verify_fused_fb": True,
                "kda_verify_kernel": None,
                "kda_verify_pdl": True,
                "kv_lens_refresh_once": True,
                "lamport_push_allreduce": True,
                "latent_ar_cute": 1,
                "latent_ar_oneshot_max_bytes": 6291456,
                "latent_ar_own_ws": True,
                "latent_ar_own_ws_max_m": 128,
                "latent_ar_rmsnorm_fused": True,
                "latent_ar_row": True,
                "latent_ar_row_max_m": 128,
                "latent_bf16_gemm": "cublas_linear",
                "mla_output_gate": "sigmoid_gate_mul",
                "mla_packed_proj": False,
                "mix_bank_in_mix": True,
                "moe_backend": "cutedsl_fc12",
                "moe_cp_tail": True,
                "moe_cp_tail_bands": "1-60",
                "moe_latent_down_stream": True,
                "moe_latent_up_resident": False,
                "moe_latent_up_resident_bands": "31-48",
                "moe_shared_late": True,
                "moe_shared_late_bands": "36-60",
                "moe_shared_stream_overlap": True,
                "moe_tail_early_poll": True,
                "moe_tp_ep": "tp2_ep4",
                "mxfp8_gemm": "rubin_dense_blockscaled_gemm",
                "oproj_ar_early_poll": True,
                "route_approx": True,
                "route_tail": 1,
                "route_tail_bands": "helix8",
                "router_bf16": True,
                "router_exact": "kda",
                "router_first": True,
                "router_first_wide": True,
                "shared_gate_up_fused": True,
                "shared_sm": 0,
                "vocab_parallel_select": "vocab_parallel_argmax/sample",
                "zero_gpu_pools": True,
            }
        ),
    }
)

# Detail key -> the key whose logged value it belongs to.
_DETAILS = MappingProxyType(
    {
        "attn_res_pdl_max_m": "attn_res_pdl",
        "draft_lowm_gemm_d_class": "draft_lowm_gemm",
        "draft_norm_ctas": "draft_conv_norm",
        "draft_q_a_kv_a_bands": "draft_q_a_kv_a_fused",
        "draft_window_mla_bands": "draft_window_mla",
        "dsa_ks_select_cluster_max": "dsa_ks_select_pack",
        "dsa_oproj_push_bands": "dsa_oproj_push",
        "latent_ar_oneshot_max_bytes": "latent_ar_own_ws",
        "latent_ar_own_ws_max_m": "latent_ar_own_ws",
        "latent_ar_row_max_m": "latent_ar_row",
        "moe_cp_tail_bands": "moe_cp_tail",
        "moe_latent_up_resident_bands": "moe_latent_up_resident",
        "moe_shared_late_bands": "moe_shared_late",
        "moe_tail_early_poll": "moe_cp_tail",
        "route_approx": "route_tail",
        "router_first_wide": "router_first",
    }
)


# fp8_kv_layers None: every DSA layer's latent KV is FP8.
_CACHE_ARGS = MappingProxyType(
    {
        "generation": MappingProxyType(
            {
                "fp8_kv_layers": None,
                "indexer_k_dtype": "fp8",
                "kv_dtype": "fp8",
                "state_dtype": torch.float32,
                "zero_gpu_pools": True,
            }
        ),
    }
)

# The MoE padding-row mask serves a prefill CUDA graph, which these targets do not capture.
_MOE_ARGS = MappingProxyType(
    {
        "generation": MappingProxyType(
            {
                "mask_graph_padding_rows": False,
            }
        ),
    }
)

# Stock engine names this file imports directly, each a computation outside the family's ops
# (test_rubin_flat_shape.py checks the list); helpers reached through the family's ops/ are not
# listed.
UNCERTIFIED_GENERIC_CALLS = (
    "tensorrt_llm._torch.distributed.ops.get_or_scale_allreduce_mnnvl_workspace",
    "tensorrt_llm._torch.models.modeling_speculative.external_drafter_config_kwargs",
    "tensorrt_llm._torch.models.modeling_utils.DecoderModel",
    "tensorrt_llm._torch.models.modeling_utils.DecoderModelForCausalLM",
    "tensorrt_llm._torch.pyexecutor.sampler.ops.flashinfer",
    "tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla.GREEDY_TEMPERATURE_THRESHOLD",
    "tensorrt_llm._torch.speculative.dflash.DFlashWorker",
    "tensorrt_llm._torch.speculative.interface.SpecWorkerBase",
    "tensorrt_llm._torch.speculative.interface._FORCE_ACCEPT_RNG_COUNTER_STRIDE",
    "tensorrt_llm._torch.speculative.interface._FORCE_ACCEPT_RNG_SLOT_STRIDE",
    "tensorrt_llm._torch.speculative.should_use_separate_draft_kv_cache",
    "tensorrt_llm._torch.speculative.spec_tail_kernels.DraftBlockPlacement",
    "tensorrt_llm._torch.speculative.spec_tail_kernels.dflash_paged_latent_prepare",
    "tensorrt_llm._torch.speculative.spec_tail_kernels.store_context_latents",
    "tensorrt_llm._torch.speculative.vocab_parallel.VocabShardedLogits",
    "tensorrt_llm._torch.speculative.vocab_parallel._LOCAL_BLOCK_V",
    "tensorrt_llm._torch.speculative.vocab_parallel._local_argmax_kernel",
    "tensorrt_llm._utils.prefer_pinned",
    "tensorrt_llm.functional.RopeEmbeddingUtils",
)

# The FC12 kernel's launch levers on this layout's decode steps (the MoE block's launches; the
# core places the expert weights per die for the die schedule): MMA N (None: the tuner picks 128
# or 256), the die schedule above its token floor and how its claim runs ("uniform": the whole
# scheduler warp claims, which keeps the TMA-B and MMA warps on the uniform datapath; same tiles
# as lane 0 alone), the scheduler lookahead, the SMs left to the shared experts, and how many
# K-blocks of the B / SFB weight stream the uniform claim kernel prefetches into L2 ahead of its
# loads (0: none; the same loads, so the same result). MMA N is read from the logged constant
# table, so that the log states what runs.
_FC12_MMA_N = _CONSTANTS["generation"]["fc12_mma_n"]
_FC12_DIE_SCHED = True
_FC12_DIE_CLAIM = "uniform"
_FC12_DIE_MIN_TOKENS = 24
_FC12_LOOKAHEAD = 1
_FC12_SM_RESERVE = 16
_FC12_L2_PREFETCH = 2


# --------------------------------------------------------------------------------------------------
# GEMM pins: BF16 generation GEMMs at a pinned cuBLASLt algorithm.
#
# At a row count measured for it, a site's BF16 GEMM runs through ``rubin_lt_gemm`` at a pinned
# cuBLASLt algorithm instead of the one cuBLASLt's heuristic ranks first for today's call
# (``cublas_mm``, ``rubin_linear``, ``bmm_out``, the router's ``dsv3_router_gemm_op``). Each pin is
# a configuration from the heuristic's own list for the site's problem whose output equalled today's
# call word for word (three activation families on two layers' checkpoint weights, and a repeat)
# and which was faster in every timing round of CUDA graph replays. A site group's launch constant
# (``gemm_pins_*``, on; off, every site runs today's GEMM) switches its pins. The binding
# needs cuBLAS 13 (``cublasLtGetVersion``); another version fails the check below. The first call
# of a pinned site and row count in a process, outside capture, runs both GEMMs on its operands and
# compares the words: equal, the pinned algorithm runs from then on (INFO); not, or not offered on
# this device, today's GEMM keeps running (WARNING). A capture before that check keeps today's GEMM,
# unrecorded. The same kernels as before otherwise, on the same streams.
# --------------------------------------------------------------------------------------------------

#: site -> {rows: (algorithm, tile, stages, cluster shape, split-K, reduction scheme, swizzle,
#: custom option, inner shape)}: the pinned cuBLASLt configurations whose output equals today's call
#: word for word (B-class), measured at helix8 generation rows 6 x b (k5): b2..b4 (12-24 rows),
#: b9..b12 (54-72 rows) and b21 (126 rows) by a CUPTI census of in-graph replays with cold weights
#: (kept where the paired gain had a 95 % interval below 0, beyond the A/A offset, of at least 0.25 us
#: a call and 2 us a step), and at b24 (144 rows) by a separate census at that row count. A row
#: count without an entry keeps today's GEMM. Comments: us a call, today -> pinned.
_GEMM_PIN_TABLE = MappingProxyType(
    {
        # gemm_pins_moe: latent_up after the latent all-reduce (144 rows; up to 60 rows moe_cp_tail
        # owns latent_up), latent_down on the latent stream (60 rows: split-K 4 as today's, 11.71 ->
        # 11.33), the router's FP32 logits (144 rows: split-K 8 and cuBLAS's reduce kernel, as today's).
        "moe_latent_up": MappingProxyType({144: (66, 326, 35, 11, 1, 0, 0, 3, 0)}),
        "moe_latent_down": MappingProxyType(
            {60: (66, 18, 35, 6, 4, 4, 0, 0, 0), 144: (66, 315, 35, 4, 1, 0, 0, 1, 0)}
        ),
        "moe_router": MappingProxyType({144: (66, 321, 35, 3, 8, 2, 0, 3, 0)}),
        # gemm_pins_mla (x24 DSA layers): [q_a | kv_a] at the head of the indexer chain (18 / 24
        # rows: split-K 8 as today's, 8.32 / 8.40 -> 7.39 / 7.46), o_proj after the gate (66 / 72 rows
        # 6.40 / 6.43 -> 4.83 / 4.96), the indexer's wq_b at 144 rows; W_UV has no pin left.
        "mla_qkv_a": MappingProxyType(
            {
                18: (66, 26, 35, 4, 8, 1, 0, 1, 0),
                24: (66, 26, 35, 4, 8, 2, 0, 1, 0),
                144: (66, 315, 35, 3, 1, 0, 0, 1, 0),
            }
        ),
        "mla_wuv": MappingProxyType({}),
        "mla_o": MappingProxyType(
            {
                66: (66, 318, 35, 11, 1, 0, 0, 1, 0),
                72: (66, 318, 35, 11, 1, 0, 0, 1, 0),
                144: (66, 26, 35, 3, 1, 0, 0, 3, 0),
            }
        ),
        "mla_idx_wq_b": MappingProxyType({144: (66, 15, 35, 3, 1, 0, 0, 1, 0)}),
        # gemm_pins_dense: layer 0's dense MLP, down at 54 / 60 rows (11.90 / 11.84 -> 8.99 / 9.09).
        "dense_gu": MappingProxyType({144: (66, 326, 35, 11, 1, 0, 0, 3, 0)}),
        "dense_dn": MappingProxyType(
            {
                54: (66, 15, 35, 11, 1, 0, 0, 1, 0),
                60: (66, 15, 35, 11, 1, 0, 0, 1, 0),
                144: (66, 326, 35, 11, 1, 0, 0, 3, 0),
            }
        ),
        # gemm_pins_draft: the DFlash2 drafter's five layers, serial after the target: q_b (18 / 24 /
        # 60 / 126 rows 8.61 / 9.12 / 8.35 / 11.52 -> 7.90 / 7.84 / 7.84 / 8.48), o_proj and down
        # (66 / 72 rows 6.24 / 6.34 -> 4.83 / 4.93 and 6.94 / 7.01 -> 5.18 / 5.18).
        "draft_q_b": MappingProxyType(
            {
                18: (66, 13, 35, 11, 1, 0, 0, 1, 0),
                24: (66, 13, 35, 11, 1, 0, 0, 1, 0),
                60: (66, 18, 35, 11, 1, 0, 0, 1, 0),
                126: (66, 21, 35, 7, 1, 0, 0, 2, 0),
                144: (66, 165, 35, 5, 1, 0, 0, 2, 0),
            }
        ),
        "draft_o": MappingProxyType(
            {
                66: (66, 318, 35, 11, 1, 0, 0, 1, 0),
                72: (66, 318, 35, 11, 1, 0, 0, 1, 0),
                144: (66, 26, 35, 3, 1, 0, 0, 3, 0),
            }
        ),
        "draft_gate_up": MappingProxyType({144: (66, 315, 35, 3, 1, 0, 0, 3, 0)}),
        "draft_down": MappingProxyType(
            {
                66: (66, 318, 35, 11, 1, 0, 0, 1, 0),
                72: (66, 318, 35, 11, 1, 0, 0, 1, 0),
                144: (66, 318, 35, 3, 1, 0, 0, 1, 0),
            }
        ),
        "draft_attn_conv_proj": MappingProxyType({144: (66, 315, 35, 3, 1, 0, 0, 3, 0)}),
        "draft_mlp_conv_proj": MappingProxyType({144: (66, 315, 35, 3, 1, 0, 0, 1, 0)}),
    }
)
#: site -> {rows: configuration}: A-class pins, the census's fastest algorithm where it beat the
#: fastest word-equal one by more than 0.05 us. Their output is another FP32 summation order of the
#: same BF16 products (split-K 1 where today's splits K, or another split), rounded once to the
#: output dtype as today's: a compute-order change, no dtype change. Under ``gemm_pins_a_class`` (and
#: the site group's constant) an entry replaces the table's at its row count, and the device check
#: is a band (``_GEMM_PIN_A_TOL`` of today's largest magnitude) instead of the words.
_GEMM_PIN_TABLE_A = MappingProxyType(
    {
        # latent_up at 12 rows: split-K 1 against today's 7 (9.34 -> 7.68 us, x92).
        "moe_latent_up": MappingProxyType({12: (66, 312, 35, 3, 1, 0, 0, 1, 0)}),
        # latent_down: 12 rows split-K 8 against today's 11 (9.54 -> 9.06); 54..72 rows split-K 1
        # against today's 4 (11.20 / 11.71 / 11.78 / 11.74 -> 10.05 / 10.18 / 10.46 / 10.43).
        "moe_latent_down": MappingProxyType(
            {
                12: (66, 449, 35, 3, 8, 2, 0, 1, 0),
                54: (66, 12, 35, 4, 1, 0, 0, 0, 0),
                60: (66, 12, 35, 4, 1, 0, 0, 0, 0),
                66: (66, 15, 35, 6, 1, 0, 0, 3, 0),
                72: (66, 15, 35, 6, 1, 0, 0, 0, 0),
            }
        ),
        # 12 rows, x24: [q_a | kv_a] split-K 8 against today's 6 (7.23 -> 6.66), wq_b split-K 1
        # against 2 (5.28 -> 4.22).
        "mla_qkv_a": MappingProxyType({12: (66, 394, 35, 4, 8, 2, 0, 1, 0)}),
        "mla_idx_wq_b": MappingProxyType({12: (66, 314, 35, 4, 1, 0, 0, 1, 0)}),
        # dense gate_up at 18 / 24 rows: split-K 1 against today's 4 (17.34 / 16.45 -> 14.11 /
        # 14.21).
        "dense_gu": MappingProxyType(
            {18: (66, 315, 35, 3, 1, 0, 0, 3, 0), 24: (66, 315, 35, 3, 1, 0, 0, 3, 0)}
        ),
        # drafter gate_up, x5: 12 rows split-K 8 against 11 (9.57 -> 8.90); 54..72 rows split-K 1
        # against 4 (11.20 / 10.98 / 11.62 / 11.55 -> 10.05 / 10.11 / 10.24 / 10.24).
        "draft_gate_up": MappingProxyType(
            {
                12: (66, 449, 35, 3, 8, 2, 0, 1, 0),
                54: (66, 12, 35, 4, 1, 0, 0, 0, 0),
                60: (66, 12, 35, 4, 1, 0, 0, 0, 0),
                66: (66, 15, 35, 6, 1, 0, 0, 0, 0),
                72: (66, 15, 35, 6, 1, 0, 0, 0, 0),
            }
        ),
    }
)
#: Each site group's launch constant and its sites.
_GEMM_PIN_GROUPS = MappingProxyType(
    {
        "gemm_pins_moe": ("moe_latent_up", "moe_latent_down", "moe_router"),
        "gemm_pins_mla": ("mla_qkv_a", "mla_wuv", "mla_o", "mla_idx_wq_b"),
        "gemm_pins_dense": ("dense_gu", "dense_dn"),
        "gemm_pins_draft": (
            "draft_q_b",
            "draft_o",
            "draft_gate_up",
            "draft_down",
            "draft_attn_conv_proj",
            "draft_mlp_conv_proj",
        ),
    }
)
#: The (site, rows) whose pin is A-class: the groups whose constant is on, under gemm_pins_a_class.
_GEMM_PINS_A_CLASS = frozenset(
    (site, rows)
    for key, sites in _GEMM_PIN_GROUPS.items()
    if _CONSTANTS["generation"][key] and _CONSTANTS["generation"]["gemm_pins_a_class"]
    for site in sites
    for rows in _GEMM_PIN_TABLE_A.get(site, {})
)
#: (site, rows) -> configuration, for the groups whose constant is on (A-class entries over the
#: table's under gemm_pins_a_class).
_GEMM_PINS = MappingProxyType(
    {
        (site, rows): config
        for key, sites in _GEMM_PIN_GROUPS.items()
        if _CONSTANTS["generation"][key]
        for site in sites
        for rows, config in _GEMM_PIN_TABLE[site].items()
    }
    | {(site, rows): _GEMM_PIN_TABLE_A[site][rows] for site, rows in _GEMM_PINS_A_CLASS}
)
#: An A-class pin's device check: every output element within this fraction of today's largest
#: magnitude (the census saw at most 7e-3: one or two BF16 ulps of another summation order), and
#: finite. It catches a wrong algorithm, not the rounding of another order.
_GEMM_PIN_A_TOL = 2.0**-5
#: Per process: (site, rows) -> whether the pin's check passed (absent: not checked yet).
_GEMM_PIN_CHECKED = {}
#: Logged once per process, site and row count by the pin's check: it passed (INFO; an A-class pin
#: with its own line), or it failed and today's GEMM runs (WARNING).
_GEMM_PIN_LINE = (
    "Kimi K3.1 Rubin GEMM pins: {site} at {rows} rows runs cuBLASLt's {pin}, today's GEMM's bits"
)
_GEMM_PIN_LINE_A = (
    "Kimi K3.1 Rubin GEMM pins: {site} at {rows} rows runs cuBLASLt's {pin} (A-class: another "
    "summation order, within {tol} of today's largest magnitude)"
)
_GEMM_PIN_OFF = "Kimi K3.1 Rubin GEMM pins: {site} at {rows} rows: {why}; today's GEMM runs"


def _gemm_pin_check(site: str, rows: int, pin: LtPin, a, b, today: Callable) -> torch.Tensor:
    """``today()``'s output for this call, once the pin of ``site`` at ``rows`` rows was checked
    on its operands: the pinned algorithm into a new tensor of today's output layout, word for
    word (an A-class pin: finite and within ``_GEMM_PIN_A_TOL`` of today's largest magnitude).
    Records and logs the outcome."""
    want = today()
    a_class = (site, rows) in _GEMM_PINS_A_CLASS
    why = ""
    try:
        got = rubin_lt_gemm(a, b, torch.empty_like(want), pin)
    except (LtUnavailable, ValueError) as exc:
        why = str(exc)
    else:
        if a_class:
            ref = want.float()
            bound = _GEMM_PIN_A_TOL * ref.abs().max()
            err = (got.float() - ref).abs().max()
            if not bool(torch.isfinite(got).all()) or bool(err > bound):
                why = f"its output is not within {_GEMM_PIN_A_TOL} of today's largest magnitude"
        else:
            words = torch.int16 if want.element_size() == 2 else torch.int32
            if not torch.equal(got.view(words), want.view(words)):
                why = "its output is not today's word for word"
    if why:
        logger.warning(_GEMM_PIN_OFF.format(site=site, rows=rows, why=why))
    elif a_class:
        line = _GEMM_PIN_LINE_A.format(site=site, rows=rows, pin=pin.name(), tol=_GEMM_PIN_A_TOL)
        logger.info(line)
    else:
        logger.info(_GEMM_PIN_LINE.format(site=site, rows=rows, pin=pin.name()))
    _GEMM_PIN_CHECKED[(site, rows)] = not why
    return want


def _gemm_pinned(
    site: str,
    a: torch.Tensor,
    b: torch.Tensor,
    today: Callable,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """``a @ b`` for ``site``: ``today()``, today's GEMM, or at a row count pinned for the site,
    once this process checked the pin, the pinned cuBLASLt algorithm into ``out`` (None: a new
    ``[rows, N]`` tensor of ``out_dtype``, the layout of today's output)."""
    rows = a.shape[-2]
    config = _GEMM_PINS.get((site, rows))
    if config is None:
        return today()
    pin = LtPin(*config)
    checked = _GEMM_PIN_CHECKED.get((site, rows))
    if checked is None:
        if torch.cuda.is_current_stream_capturing():
            return today()
        return _gemm_pin_check(site, rows, pin, a, b, today)
    if not checked:
        return today()
    if out is None:
        out = torch.empty(rows, b.shape[-1], dtype=out_dtype, device=a.device)
    try:
        return rubin_lt_gemm(a, b, out, pin)
    except (LtUnavailable, ValueError) as exc:
        # Another operand layout than the checked call's (the plan is keyed by the strides).
        why = f"{exc} (a call after the check)"
        key = _GEMM_PIN_OFF.format(site=site, rows=rows, why=why)
        logger.warning_once(key, key=key)
        return today()


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
# * KDA: ``kda_decode(L, x, st, rt)``, ``kda_verify(L, x, st, rt)``;
# * MLA + DSA: ``mla_generate(L, i, x, st, rt)``;
# * MoE: ``latent_moe(L, i, x, st, rt)``, ``dense_mlp(L0, x, st, rt)``;
# * DFlash2: ``draft_step(DW, hidden_streams, st, rt)``, the container ``DraftModel`` and the
#   worker ``DFlashWorker`` (``draft_config_for`` builds the drafter's config).
#
# The mixers and ``dense_mlp`` return ``[rows of x, hidden]`` BF16. The mixers return this rank's
# partial sum over the helix group: the MLA block attends with all 96 heads over the rank's eighth
# of the latent KV, combines the eight partial attentions and projects its 12 heads' share through
# o_proj; KDA runs 12 heads. The norm slot after the mixer finishes the sum over the eight ranks
# (``core.tp_mapping``: the helix ranks as TP ranks). ``dense_mlp`` returns a whole sum (it
# reduces over ``rt.moe_comm.dense_all_reduce``). ``latent_moe`` returns ``_MoeOut(routed,
# shared)``, both whole (the routed latent of the TP2 x EP4 experts is all-reduced before its norm
# and up projection): the next mix adds ``bf16(routed + shared)`` to the residual in one kernel
# (``moe_norm``, ``moe_add``). With ``moe_cp_tail`` a decode step in its row bands returns
# ``_MoeTail`` instead (see the MoE section), which the same slots finish.
#
# ``i`` is the global layer index. ``x`` holds the rows the function serves. helix8 serves
# generation steps only (helix context parallelism): ``DecodeTarget`` runs every step, with
# ``mla_generate``, ``kda_decode`` or ``kda_verify`` on every row; ``PrefillTarget`` refuses a step
# with context rows. Rows past ``st.num_tokens`` are CUDA graph padding.
#
# The attention residual. Each layer mixes the running prefix sum with the bank of block snapshots
# (``attn_res``) ahead of each sublayer; a layer at a block boundary first stores its input stream
# in the bank and starts the prefix sum over. The slots are ``_Rt``'s (the core calls them, the
# blocks do not); each picks its kernel by the step's row count (``_mix_slots``,
# ``_tp_reduce_slots``).
# --------------------------------------------------------------------------------------------------

# Rows per call of the attn_res_fwd kernel; longer prefill chunks run in slices.
_ATTN_RES_OP_MAX_TOKENS = 16384
# The per-token fused mix: up to 1 row with the per-token topology, 32 with the split one.
_FUSED_MAX_ROWS = {"per_token": 1, 1: 32}
# Candidates (snapshots + the prefix sum) the persistent kernel takes: 2 to 9.
_PERSISTENT_MAX_CANDIDATES = 9
# The attention-residual epilogue levers (launch constants) and the most rows they move to
# rubin_attn_res_add_rmsnorm_persistent_x.
_X_LEVERS = ("attn_res_mx_moe", "attn_res_moe_fold", "attn_res_grid_cap")
_X_MAX_ROWS = 4096
# The most rows whose MXFP8 scales the PDL mix stores (attn_res_mx_pdl): one R128c4 scale tile.
_MX_TILE_ROWS = 128
#: Logged once per process by the first KDA mix that hands its MXFP8 form to the QKVG GEMM from
#: the PDL kernel (``attn_res_mx_pdl``).
_KDA_MX_PDL_LINE = (
    "Kimi K3.1 Rubin KDA: the PDL mix stores the QKVG GEMM's MXFP8 input (no quantization kernel)"
)
# allreduce_module's fusion op as an int: AllReduceFusionOp.NONE.
_AR_NONE = 0
# rt.events: the (fork, join) of the attention phase's side work, and of the indexer's work on
# the layer input alone (dsa_idx_fork).
_EV_ATTN = 0
_EV_IDX = 1
# The sparse MLA attends to this many Top-K entries per query row (index_topk).
_DSA_TOPK = _K31_DSA["index_topk"]
# The DSL logits kernel's MMA N bound: next_n x index heads per atom (FP8 keys).
_DSA_MAX_ATOM_N = 256
_DSA_INDEX_HEADS = _K31_DSA["index_num_heads"]
# The indexer's output rows, one buffer per step (the engine's cached-buffer name).
_DSA_TOPK_OUT = "indexer_topk_out_buffer"


class _DsaStep(NamedTuple):
    """One step's DSA state for the MLA layers (helix8), read once a step off the family DSA metadata.

    helix8 serves generation steps only, so every MLA row is a generation row of a decode or
    verify block. Tensors are metadata buffers or views of them, so a captured graph reads what
    the replayed step's ``prepare`` wrote. ``ng`` is the step's generation requests.
    """

    next_n: int  # rows per generation request (1 + the verified draft length)
    skip_gen: bool  # md.skip_indexer_for_gen_reqs: the rows take topk_dense
    # Index-key insert (every live row of the step, once).
    slot_fp8: torch.Tensor  # md.slot_mapping_fp8
    slot_scale: torch.Tensor  # md.slot_mapping_scale
    # Key-sharded Top-K: this rank scores its own pages, the merge keeps the exact global Top-K.
    ks_next_n: int  # md.key_shard_next_n when the step's rows are key-shard eligible, else 0
    ks_block_table: torch.Tensor | None  # helix_key_shard_inputs(rows)[0]
    ks_ctx_lens: torch.Tensor | None  # helix_key_shard_inputs(rows)[1][:ng]
    ks_sched: torch.Tensor | None  # helix_key_shard_inputs(rows)[2]
    ks_row_lens: torch.Tensor | None  # helix_key_shard_inputs(rows)[3]: each row's local bound
    ks_local_len: int  # md.key_shard_max_local_len: the local logits' width
    row_bounds: torch.Tensor | None  # md.helix_spec_row_kv_lens(rows); None without spec tokens
    counts_out: torch.Tensor | None  # md.helix_sparse_row_lens_out(rows): the kept counts
    # The fallback (ks_next_n == 0; eager only): local logits, the local Top-K, the CP merge.
    fb_dsl: bool  # the DSL logits serve it: no per-token bounds
    fb_ctx_lens: torch.Tensor  # the decode-logits inputs of the step's path
    fb_block_table: torch.Tensor
    fb_sched: torch.Tensor
    fb_topk_next_n: int  # 1 with per-token bounds (row_bounds is the Top-K's row_kv_lens)
    max_len: int  # md.get_indexer_max_seq_len()
    gen_kv_lens: torch.Tensor  # md.kv_lens_cuda_runtime[:ng]
    topk_out: torch.Tensor  # the indexer's Top-K rows of the step [num_tokens, index_topk] int32
    topk_dense: torch.Tensor  # md.topk_indices_buffer: the dense Top-K (skip_gen)
    radix_aux_indices: torch.Tensor  # md.radix_aux_indices
    radix_aux_logits: torch.Tensor  # md.radix_aux_logits
    # Local Top-K -> global latent-pool rows.
    req_idx_gen: torch.Tensor  # md.req_idx_per_token[:num_tokens]
    block_table_gen: torch.Tensor  # md.block_table[:ng]
    tokens_per_block: int
    # The MLA ops' step arguments, presented as the helix ops take them.
    rope_gen: dict  # request-major + helix_tensor_params (offsets, inactive rank, local slots)
    sparse_gen: dict  # one row per generation token (token_major_gen_view(token_sparse=True))


class _Step(NamedTuple):
    """One step's batch state: what the blocks read of the engine's metadata, read once a step.

    ``DecodeTarget`` states ``num_contexts`` and ``num_ctx_tokens`` as 0 instead of reading them.
    """

    md: Any  # the step's attention metadata (the family DSA metadata)
    kda_md: Any  # md.mamba_metadata: KDA state slots, query offsets, snapshot plan
    cache: Any  # md.kv_cache_manager: KDA state and conv pools, DSA latent and index-key caches
    num_tokens: int  # md.num_tokens: the live rows; the inputs may carry padding rows after them
    num_contexts: int  # context requests
    num_ctx_tokens: int  # context rows, the step's first rows
    num_generations: int  # generation requests
    rows_per_gen: int  # rows per generation request: 1 + the draft length the step verifies
    spec_md: Any  # the DFlash2 spec metadata (hidden-state taps), None without a drafter
    cuda_graph: bool  # md.is_cuda_graph: a CUDA graph's warmup or capture (the stream plans)
    dsa: _DsaStep | None  # the step's DSA state (indexer, Top-K, sparse MLA); decode steps


class _MoeOut(NamedTuple):
    """A latent-MoE layer's output, both parts whole; the next mix adds ``bf16(routed + shared)``."""

    routed: torch.Tensor  # [rows, hidden] BF16: latent all-reduce + norm + latent_up
    shared: torch.Tensor  # [rows, hidden] BF16: the shared experts, all-reduced


class _MoeTail(NamedTuple):
    """A latent-MoE layer's output still in its Lamport call (the column-parallel tail).

    Every rank pushed its shared experts' partial into the call's slots and its columns of
    latent_up into the call's all-gather rows; the next mix finishes the call, which gives
    ``bf16(latent_up + bf16(sum of the shared partials))``, and adds it to the residual.
    """

    ws: Any  # the tail's Lamport workspace (rt.ull_tail.ws)
    cpp_one_shot: bool  # the C++ all-reduce whose sums the finish reproduces
    chunk: int  # ranks per partial sum of that all-reduce


class _Pushed(NamedTuple):
    """A mixer's partial that is already in the Lamport slots of the TP all-reduce
    (``kda_oproj_push``, ``dsa_oproj_push``): the mix after it finishes the call
    (``rt.oproj_push.finish_norm``)."""

    rows: int


class _DensePartial(NamedTuple):
    """Layer 0's dense MLP output before its all-reduce (``dense_ar_push``): the next mix sums it
    over the TP group through ``rt.reduce_norm`` (its push path where it serves the rows)."""

    partial: torch.Tensor  # [rows, hidden] BF16: this rank's share of the down projection


class _OprojPush(NamedTuple):
    """``kda_oproj_push`` / ``dsa_oproj_push``: the o_proj push of a KDA verify step's or a DSA
    layer's partial (``rt.oproj_push``)."""

    ws: Any  # the Lamport push workspace (LamportPushWs) of the TP all-reduce
    desc: torch.Tensor  # its row-push descriptor, the GEMM epilogue's (built once, eagerly)
    max_rows: int  # the most rows it serves: attn_res_pdl_max_m
    finish_norm: Callable  # (prefix, bank, mw) -> (prefix, normed), finishing the pushed call
    # dsa_oproj_push: the row bands a DSA layer's o_proj pushes in (none when off).
    dsa_bands: tuple = ()


class _MixW(NamedTuple):
    """The weights of one attention-residual mix: score projection and norm, then the output norm."""

    proj: torch.Tensor  # the *_res_proj weight flattened [H]
    res_norm: torch.Tensor  # the *_res_norm weight [H] (scores the candidates)
    out_norm: torch.Tensor  # the RMSNorm the mix feeds [H]


class _Streams(NamedTuple):
    """The core's side streams, made once at construction (never during a capture)."""

    aux: torch.cuda.Stream  # rt.aux_stream: norm and indexer quant overlap, shared experts, KDA L2
    fork: torch.cuda.Stream  # rt.fork_stream: a query chain beside the indexer (fork_q)
    idx: torch.cuda.Stream  # rt.idx_stream: the indexer's keys and weights (dsa_idx_fork)
    latent: torch.cuda.Stream  # latent_down beside the router
    prezero: torch.cuda.Stream  # FC12's output zeroed beside the routing (fc12_prezero)
    tap: torch.cuda.Stream  # the drafter's capture copies (draft_tap_stream)


class _LateStreams(NamedTuple):
    """moe_shared_late's streams, made once at construction at the highest stream priority torch
    offers (``torch.cuda.Stream.priority_range()[1]``; the default priority is CUDA's lowest).

    A band step's layers run on ``fwd`` and their side work on the others (``_moe_shared_late_rt``),
    so the shared experts, on the core's default-priority aux stream, are the one chain below the
    rest. Stream capture copies a stream's priority into each kernel node it captures, and torch
    instantiates its CUDA graphs to launch each kernel node at its own priority.
    """

    fwd: torch.cuda.Stream  # the layers (forked from the current stream once per step)
    aux: torch.cuda.Stream  # rt.aux_stream: norm and indexer quant overlap, KDA L2
    fork: torch.cuda.Stream  # rt.fork_stream
    idx: torch.cuda.Stream  # rt.idx_stream
    latent: torch.cuda.Stream  # moe_comm.latent_stream
    prezero: torch.cuda.Stream  # moe_comm.prezero_stream


def _late_streams() -> _LateStreams:
    """``_LateStreams`` at the highest priority, in field order."""
    high = torch.cuda.Stream.priority_range()[1]
    return _LateStreams(*(torch.cuda.Stream(priority=high) for _ in _LateStreams._fields))


def _on_stream(stream: torch.cuda.Stream, run: Callable, *args: Any) -> torch.Tensor:
    """``run(*args)`` on ``stream``, forked from the current stream once and joined back once.

    ``run`` allocates on ``stream`` and the current stream reads its result only after the join.
    Outside a capture the result is also marked used by the current stream, so its memory goes
    back to ``stream``'s pool only after the current stream's readers.
    """
    main = torch.cuda.current_stream()
    stream.wait_stream(main)
    with torch.cuda.stream(stream):
        out = run(*args)
    main.wait_stream(stream)
    if not torch.cuda.is_current_stream_capturing():
        out.record_stream(main)
    return out


class _DsaPools:
    """The DSA layers' cache views and the facts their step state needs (``rt.dsa_pools``).

    ``bind`` reads the views off the cache manager on the first step and again when the manager
    object changes (the KV estimation builds one, the final run another); it allocates nothing,
    so a capture may call it. All MLA layers share one FP8 latent pool (the manager raises
    otherwise), which their page parameters index.
    """

    def __init__(self, mla_layers: tuple[int, ...], rope_positions: int) -> None:
        self.mla_layers = mla_layers
        self.rope_positions = rope_positions  # positions rt.mla_rope covers
        self.cache: Any = None
        self.idx_k_cache: dict[int, torch.Tensor] = {}  # layer -> its INDEX_KEY page view
        self.pool_params: dict[int, tuple[int, int]] = {}  # layer -> (page scale, layer offset)
        self.local: dict[int, int] = {}  # layer -> the manager's local layer index
        self.tokens_per_block = 0

    def bind(self, cache: Any) -> None:
        """Take the step's cache manager's views (a new manager replaces the old views)."""
        if cache is self.cache:
            return
        assert int(cache.max_seq_len) <= self.rope_positions, (
            f"the identity RoPE table covers {self.rope_positions} positions, the KV cache "
            f"{cache.max_seq_len}"
        )
        self.cache = cache
        self.local = {i: int(cache.layer_offsets[i]) for i in self.mla_layers}
        self.idx_k_cache = {i: cache.get_indexer_k_cache_buffers(i) for i in self.mla_layers}
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

    all_reduce: Any  # the TP group's AllReduce: the shared experts' and the routed latent's sums
    dense_all_reduce: Any  # layer 0's dense MLP group (the TP group); None where it is whole
    latent_ws: dict | None  # the latent all-reduce's own Lamport workspace (latent_ar_own_ws)
    latent_stream: torch.cuda.Stream  # latent_down
    prezero_stream: torch.cuda.Stream  # FC12's output and counters zeroed (fc12_prezero)
    shared_fork: torch.cuda.Event
    shared_join: torch.cuda.Event
    latent_fork: torch.cuda.Event
    latent_ready: torch.cuda.Event
    prezero_ready: torch.cuda.Event


class _UllTail(NamedTuple):
    """The column-parallel MoE tail's resources (``rt.ull_tail``), built once in post_load_weights."""

    ws: Any  # its own Lamport workspace: slots for the shared partials, all-gather rows
    handle: Any  # the workspace's multicast buffer, kept alive
    c0: int  # this rank's first row of latent_up (its first hidden column)
    n: int  # this rank's rows of latent_up: whole 128-blocks, 1 / world of the hidden size
    tactics: tuple  # the push GEMM's tactic by row count (index 0 unused)
    cpp_one_shot: bool  # the C++ all-reduce whose sums the finish reproduces (_MoeTail)
    chunk: int  # ranks per partial sum of that all-reduce


class _Rt(NamedTuple):
    """The runtime resources one forward reads, built once in ``post_load_weights``.

    No field is a per-layer helper object: the blocks call the family entries with these
    resources as arguments.
    """

    dsa_pools: _DsaPools  # the DSA layers' index-key caches and page parameters
    mla_rope: torch.Tensor  # the MLA kernels' RoPE table: identity rotations (K3.1 MLA is NoPE)
    aux_stream: torch.cuda.Stream  # the side stream (norm / indexer quant overlap, shared, L2)
    fork_stream: torch.cuda.Stream  # a query chain of its own beside the indexer (fork_q)
    idx_stream: torch.cuda.Stream  # the indexer's keys and weights (dsa_idx_fork)
    events: tuple  # (fork, join) CUDA event pairs, indexed _EV_ATTN, _EV_IDX
    attn_res_norm: Callable  # (prefix, bank, mw) -> normed
    add_norm: Callable  # (prefix, addend, bank, mw) -> (prefix + addend, normed)
    moe_norm: Callable  # (prefix, moe_out or tail, bank, mw) -> (prefix + the MoE output, normed)
    reduce_norm: Callable  # (prefix, partial, bank, mw) -> (prefix, normed)
    attn_res_mix: Callable  # (prefix, bank, mw) -> the pre-norm mixture (a drafter tap)
    rms_norm: Callable  # (x, weight) -> normed
    moe_add: Callable  # (prefix, moe_out or tail) -> prefix + the MoE output
    reduce: Callable  # (partial) -> the whole sum over the TP group
    all_reduce: Any  # the TP group's AllReduce module (allreduce_module's)
    moe_comm: _MoeComm  # the MoE sublayer's all-reduces, workspace, streams and events
    cp_mapping: Any  # the helix mapping: the combine's all-to-all
    # The helix ranks as TP ranks: the CP merge's all-gather group, its tp_rank this rank's CP rank.
    cp_group: Any
    ks_ws: Any  # the key-shard merge's workspace: the first decode step builds it (_key_shard_ws)
    draft: Any  # the DFlash2 drafter's bound window entry, plan buffers and scratch
    # attn_res_moe_fold: moe_norm with the update inside the persistent mix (None when off).
    moe_fold_norm: Optional[Callable] = None
    # attn_res_mx_moe: moe_norm whose normed rows may come as (normed, codes, scales) (None when off).
    moe_norm_mx: Optional[Callable] = None
    # The column-parallel MoE tail (generation, moe_cp_tail; None when off).
    ull_tail: Optional[_UllTail] = None
    # kda_oproj_push / dsa_oproj_push: the push workspace and the mix that finishes a pushed
    # partial (None when both are off).
    oproj_push: Optional[_OprojPush] = None
    # draft_tap_stream: the capture copies' stream and join (None when off).
    taps: Optional["_TapStream"] = None
    # boundary_ar_push: a boundary's pushed o_proj all-reduce, finished as the prefix (None when off).
    boundary_push: Optional["_BoundaryPush"] = None
    # dense_ar_push: layer 0's dense MLP output goes to the next mix unsummed (_DensePartial).
    dense_push: bool = False
    # dsa_ks_select_pack: helix16's select / pack runner over ks_ws (None when off or unsupported).
    ks_sel: Optional[Any] = None
    # moe_shared_late: the shared experts' stream on a band step, below every other stream of the
    # step; the column-parallel tail then forks them after the routing tail (None: rt.aux_stream,
    # forked before it).
    shared_stream: Optional[torch.cuda.Stream] = None


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
    a_log: torch.Tensor  # A_log, this rank's heads, FP32
    dt_bias: torch.Tensor  # dt_bias, this rank's share, FP32
    # Derived from l{i}_kda_conv (q_conv1d, k_conv1d, v_conv1d: this rank's channels, stacked
    # [3P, 4]) and l{i}_kda_o_norm (o_norm.weight, the gated output RMSNorm [128]):
    conv_t: torch.Tensor  # derived: conv as [3, 4, P], the decode kernel's layout
    conv_f32: torch.Tensor  # derived: conv in FP32, the verify kernel's
    o_norm_f32: torch.Tensor  # derived: o_norm in FP32, the decode and verify epilogues'
    zero_bias: torch.Tensor  # derived: the decode kernel's conv bias [P] (the layer has none)
    l2_next: tuple  # derived: the next KDA layer's weights to prefetch (kda_l2_prefetch), or ()


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

    router: torch.Tensor  # gate.weight [num_experts, H], BF16
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
    expert_offset: int  # derived: the first expert of this rank's window
    after_kda: bool  # derived: the layer's mixer is KDA (the exact router serves it)


# The _MoeW fields that read a parameter (the rest are derived).
_MOE_PARAMS = tuple(f for f in _MoeW._fields if f not in ("expert_offset", "after_kda"))


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
        workspace_=md.effective_workspace,
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
        spec_decoding_position_offsets_for_cpp=md.spec_decoding_position_offsets_for_cpp,
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
    """mla_rope_generation's metadata arguments of one step: what the rope call reads.

    Under helix the KV append also takes the token positions' owners: the per-token write slots of
    a verify block (-1: another rank owns the position) when the step has them.
    """
    helix = [md.helix_position_offsets, md.helix_is_inactive_rank]
    if md._helix_spec_tokens_valid:
        helix.append(md.helix_local_slots)
    return dict(
        sequence_length=md.kv_lens_cuda_runtime,
        host_past_key_value_lengths=md.kv_lens_runtime,
        host_context_lengths=md.prompt_lens_cpu_runtime,
        num_contexts=nc,
        kv_cache_block_offsets=md.kv_cache_block_offsets,
        host_kv_cache_pool_pointers=cache.kv_cache_pool_pointers,
        host_kv_cache_pool_mapping=cache.kv_cache_pool_mapping,
        block_ids_per_seq=md.block_ids_per_seq,
        helix_tensor_params=helix,
        residual_dim=getattr(cache, "mla_kv_cache_residual_dim", 0),
        tokens_per_block=cache.tokens_per_block,
        attention_window_size=int(md.max_seq_len),
        beam_width=md.beam_width,
    )


def _token_major_sparse_args(md: Any) -> dict:
    """thop_attention's metadata arguments under helix: one row per generation token.

    A rank keeps only its share of each row's global Top-K, so the sparse MLA reads each token as
    its own row, as long as its kept entries (``presented_token_major``). The op sizes its
    multi-CTA counter and workspace from ``max_num_requests``, the row ceiling here.
    """
    args = _sparse_args(md, 0, 0)
    view = md.token_major_gen_view(token_sparse=True)
    if view is not None:
        args.update(
            sequence_length=view.sequence_length,
            host_past_key_value_lengths=view.host_past_key_value_lengths,
            host_context_lengths=view.host_context_lengths,
            kv_cache_block_offsets=view.kv_cache_block_offsets,
            max_num_requests=view.max_num_rows,
            context_lengths=view.prompt_lens_cuda,
            host_request_types=view.host_request_types,
        )
    return args


def _dsa_step_args(md: Any, pools: _DsaPools, num_tokens: int, ng: int) -> _DsaStep:
    """Project the DSA metadata onto this step's ``_DsaStep`` (helix8), once per forward.

    The routing refuses ragged verification, so every request carries ``next_n`` rows and the rope
    call takes the request-major presentation. The key-sharded Top-K serves a step when the
    metadata's key-shard inputs cover it (``key_shard_world > 1``, ``key_shard_next_n`` rows per
    request, within one atom); otherwise the fallback takes the indexer's decode inputs for the
    step's shape.
    """
    assert not md.is_ragged_verify, "helix8 runs uniform verify blocks only"
    rows = num_tokens
    next_n = rows // ng if ng else 1
    skip_gen = bool(md.skip_indexer_for_gen_reqs)
    n = int(md.key_shard_next_n)
    ks_next_n = (
        n
        if ng
        and not skip_gen
        and md.key_shard_world > 1
        and n > 0
        and rows == ng * n
        and n * _DSA_INDEX_HEADS <= _DSA_MAX_ATOM_N
        else 0
    )
    ks = md.helix_key_shard_inputs(rows) if ks_next_n else (None, None, None, None)
    row_bounds = md.helix_spec_row_kv_lens(rows)
    fb_dsl = row_bounds is None
    if fb_dsl:
        fb = (
            md.gen_indexer_kv_lens_cuda_runtime,
            md.indexer_k_cache_block_offsets[:ng],
            md.scheduler_metadata_buffer,
        )
    elif not md.use_expanded_buffers_for_mtp or next_n == 1:
        fb = (
            md.kv_lens_cuda_2d[:ng, :next_n].contiguous(),
            md.indexer_k_cache_block_offsets[:ng],
            md.scheduler_metadata_buffer
            if next_n == 1
            else md.scheduler_metadata_buffer_full_next_n,
        )
    else:
        fb = (
            md.kv_lens_expanded_cuda[:rows].view(-1, 1),
            md.block_table_expanded[:rows],
            md.scheduler_metadata_buffer_expanded,
        )
    return _DsaStep(
        next_n=next_n,
        skip_gen=skip_gen,
        slot_fp8=md.slot_mapping_fp8,
        slot_scale=md.slot_mapping_scale,
        ks_next_n=ks_next_n,
        ks_block_table=ks[0],
        ks_ctx_lens=None if ks[1] is None else ks[1][:ng],
        ks_sched=ks[2],
        ks_row_lens=ks[3],
        ks_local_len=int(md.key_shard_max_local_len) if ks_next_n else 0,
        row_bounds=row_bounds,
        counts_out=md.helix_sparse_row_lens_out(rows),
        fb_dsl=fb_dsl,
        fb_ctx_lens=fb[0],
        fb_block_table=fb[1],
        fb_sched=fb[2],
        fb_topk_next_n=1 if row_bounds is not None else next_n,
        max_len=int(md.get_indexer_max_seq_len()),
        gen_kv_lens=md.kv_lens_cuda_runtime[:ng],
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
        req_idx_gen=md.req_idx_per_token[:num_tokens],
        block_table_gen=md.block_table[:ng],
        tokens_per_block=pools.tokens_per_block,
        rope_gen=_rope_args(md, md.kv_cache_manager, 0),
        sparse_gen=_token_major_sparse_args(md),
    )


def _build_step_args(
    md: Any,
    spec_metadata: Any,
    pools: _DsaPools,
    *,
    decode: bool = False,
) -> _Step:
    """Project the engine's metadata onto this step's ``_Step`` (the step contract).

    ``decode``: the decode target's call. It runs only when the step has no context rows, so it
    states the two context counts as 0 instead of reading them. ``pools`` must be bound to the
    step's cache manager. helix8 builds no DSA step state for a step with context rows (its prefill
    target refuses them).
    """
    num_tokens = int(md.num_tokens)
    num_contexts = 0 if decode else md.num_contexts
    num_ctx_tokens = 0 if decode else md.num_ctx_tokens
    num_generations = md.num_generations
    gen_rows = num_tokens - num_ctx_tokens
    return _Step(
        md=md,
        kda_md=md.mamba_metadata,
        cache=md.kv_cache_manager,
        num_tokens=num_tokens,
        num_contexts=num_contexts,
        num_ctx_tokens=num_ctx_tokens,
        num_generations=num_generations,
        rows_per_gen=gen_rows // num_generations if num_generations else 1,
        spec_md=spec_metadata,
        cuda_graph=bool(md.is_cuda_graph),
        dsa=_dsa_step_args(md, pools, num_tokens, num_generations) if decode else None,
    )


def _layer_views(core: Any) -> tuple[_LayerW, ...]:
    """Read every layer's weight views off the realized parameters, once.

    Views only: ``w1_rows`` declares every parameter in the layout and dtype its kernel reads
    (the fused KDA projection), so nothing here copies or re-points a parameter. The KDA decode /
    verify operands are derived tensors.
    """
    c = _CONSTANTS[core.role]
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
                a_log=p("kda_a_log"),
                dt_bias=p("kda_dt_bias"),
                conv_t=conv.view(3, p_kda, d.kda_conv).transpose(1, 2).contiguous(),
                conv_f32=conv.float(),
                o_norm_f32=o_norm.float(),
                zero_bias=torch.zeros(p_kda, dtype=torch.bfloat16, device=conv.device),
                l2_next=(),
            )
        moe = None
        if i in moe_layers:
            moe = _MoeW(
                **{f: p(f"moe_{f}") for f in _MOE_PARAMS},
                expert_offset=d.expert_offset,
                after_kda=i in kda_layers,
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
    if c.get("kda_l2_prefetch"):
        # Each KDA layer but the last prefetches the next one's weights.
        kda_views = [v.kda for v in views if v.kda is not None]
        for k, (cur, nxt) in enumerate(zip(kda_views, kda_views[1:])):
            ranges = (nxt.qkvgfab_w, nxt.qkvgfab_s, nxt.o_w, nxt.o_s, nxt.fb)
            kda_views[k] = cur._replace(l2_next=ranges)
        by_layer = {v.layer: v for v in kda_views}
        views = [v if v.kda is None else v._replace(kda=by_layer[v.kda.layer]) for v in views]
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


def _x_levers(c: Any) -> tuple[bool, bool, bool]:
    """``(mx_moe, moe_fold, grid_cap)`` of a role's constants (``_X_LEVERS``). They need the CuTe
    DSL kernel of ``rubin_attn_res_add_rmsnorm_persistent_x``: on a device it does not run on they
    stay off and one WARNING line names them."""
    on = tuple(bool(c.get(k, False)) for k in _X_LEVERS)
    if any(on) and not persistent_x_available():
        asked = ", ".join(k for k, v in zip(_X_LEVERS, on) if v)
        logger.warning(
            f"Kimi K3.1 Rubin attn_res levers: {_TARGET}: rubin_attn_res_add_rmsnorm_persistent_x's "
            f"CuTe DSL kernel does not run on this device; {asked} left off (today's calls)"
        )
        return False, False, False
    return on


def _mix_slots(c: Any, eps: float) -> dict[str, Callable]:
    """The attention-residual slots of one server role (``c`` = ``_CONSTANTS[role]``).

    The row count picks the kernel, in this order: the PDL kernel up to
    ``attn_res_pdl_max_m`` rows, then (split topology) the persistent kernel above 32 rows, then
    the per-token kernel, then the unfused mix and norm. With ``fuse_moe_residual`` a MoE output
    joins the residual in one kernel: inside the PDL mix up to ``attn_res_pdl_max_m`` rows, else
    ``rubin_moe_residual_add`` before the mix (the deferred MoE residual update).

    The epilogue levers (``_x_levers``) move rows the persistent kernel serves (past
    ``attn_res_pdl_max_m``, up to ``_X_MAX_ROWS``) to ``rubin_attn_res_add_rmsnorm_persistent_x``,
    whose outputs are today's bytes: ``attn_res_grid_cap`` every such mix, ``attn_res_moe_fold``
    the extra slot ``moe_fold_norm``, ``attn_res_mx_moe`` the extra slot ``moe_norm_mx`` (None
    when their lever is off). ``attn_res_mx_pdl`` also sets ``moe_norm_mx``: on the PDL kernel's
    rows, up to one scale tile, where the QKVG GEMM would quantize the rows in a kernel of its own
    (``_kda_mx_pdl_rows``), the fused MoE mix stores the MXFP8 form too, quantized as that GEMM
    would quantize them (``_kda_mx_exact``). Every other PDL row keeps today's calls.

    A ``_MoeTail`` (``moe_cp_tail``) is finished inside the PDL mix up to ``attn_res_pdl_max_m``
    rows (with ``moe_tail_early_poll`` the mix polls the call before its grid wait; with
    ``attn_res_mx_pdl`` a KDA layer's mix also stores the MXFP8 form on ``_kda_mx_pdl_rows``, as
    for a ``_MoeOut``), else by the finish kernel before the add. ``moe_fold_norm`` takes it to
    ``moe_norm``.
    """
    pdl_max_m = c["attn_res_pdl_max_m"] if c.get("attn_res_pdl") == 2 else 0
    persistent = c["attn_res_topology"] == 1
    fused_max = _FUSED_MAX_ROWS[c["attn_res_topology"]]
    early = bool(c.get("attn_res_early_trigger", False))
    fuse_moe_residual = bool(c.get("fuse_moe_residual", False))
    mx_moe, moe_fold, grid_cap = _x_levers(c)
    x_ops = mx_moe or moe_fold or grid_cap
    mx_pdl_max_m = min(pdl_max_m, _MX_TILE_ROWS) if c.get("attn_res_mx_pdl", False) else 0
    mx_pdl = fuse_moe_residual and mx_pdl_max_m > 0
    tail_poll = bool(c.get("moe_tail_early_poll", False))

    def use_persistent(rows: int, k: int) -> bool:
        return persistent and rows > _FUSED_MAX_ROWS[1] and k + 1 <= _PERSISTENT_MAX_CANDIDATES

    def use_x(rows: int, k: int) -> bool:
        served = k > 0 and pdl_max_m < rows <= _X_MAX_ROWS and use_persistent(rows, k)
        return x_ops and served

    def x_mix(prefix, addend, addend2, bank: torch.Tensor, mw: _MixW, mx: bool = False):
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
            mx=mx,
            grid_cap=grid_cap,
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
                bank[:, lo:hi].reshape(k, hi - lo, 1, hidden),
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
                prefix, bank, mw.proj, mw.res_norm, mw.out_norm, eps, eps, early_trigger=early
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
                early_trigger=early,
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

    def moe_add(prefix: torch.Tensor, m: _MoeOut | _MoeTail) -> torch.Tensor:
        """``bf16(prefix + bf16(routed + shared))``: two round-to-nearest adds either way."""
        if isinstance(m, _MoeTail):
            out = rubin_lamport_finish(
                m.ws, torch.empty_like(prefix), True, True, False, m.cpp_one_shot, True
            )
            return prefix + out
        if fuse_moe_residual:
            return rubin_moe_residual_add(m.routed, m.shared, prefix)
        return prefix + (m.routed + m.shared)

    def tail_mix(
        prefix: torch.Tensor,
        m: _MoeTail,
        bank: torch.Tensor,
        mw: _MixW,
        mx: bool,
        updated: Optional[torch.Tensor] = None,
    ):
        """The PDL mix that finishes the tail's call (one-shot, its all-gather rows, the
        round-to-nearest add of ``moe_add``); ``mx``: it also stores the MXFP8 form of one
        128-row tile, quantized as the QKVG GEMM would quantize the rows (``_kda_mx_exact``).
        ``updated``: the buffer the updated prefix goes to (a boundary's bank row)."""
        ws, rows = m.ws, prefix.shape[0]
        args = (
            prefix,
            ws.uc_local,
            ws.flags.data_ptr(),
            bank,
            mw.proj,
            mw.res_norm,
            mw.out_norm,
            eps,
            eps,
            ws.world,
            m.chunk,
            False,  # finish_two_shot: the tail's call is one-shot
            ws.mc_base,
            ws.rank,
            True,  # add_rn: the round-to-nearest add of moe_add
            tail_poll,
            False,  # entry_trigger
            True,  # finish_ag: the call's all-gather rows hold latent_up's output
        )
        if not mx:
            return rubin_attn_res_lamport_add_rmsnorm(*args, early_trigger=early, updated=updated)
        updated, normed, codes, scales = rubin_attn_res_lamport_add_rmsnorm_mx(
            *args,
            True,  # mx_atom: one 128-row tile's atoms are its R128c4 scales
            1,  # mx_atom_blocks
            _kda_mx_exact(rows),
            early_trigger=early,
            updated=updated,
        )
        logger.info_once(_KDA_MX_PDL_LINE, key=_KDA_MX_PDL_LINE)
        return updated, (normed, codes, scales)

    def moe_norm(
        prefix: torch.Tensor,
        m: _MoeOut | _MoeTail,
        bank: torch.Tensor,
        mw: _MixW,
        updated: Optional[torch.Tensor] = None,
    ):
        """``(prefix + bf16(routed + shared), normed mix)``. ``updated`` (mix_bank_in_mix): the
        buffer a PDL mix stores the updated prefix in, which it then returns; the other kernels
        return a tensor of their own (the caller copies it)."""
        if isinstance(m, _MoeTail):
            if bank.shape[0] and prefix.shape[0] <= pdl_max_m:
                return tail_mix(prefix, m, bank, mw, False, updated)
            new = moe_add(prefix, m)
            return new, attn_res_norm(new, bank, mw)
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
                early_trigger=early,
                updated=updated,
            )
        new = moe_add(prefix, m)
        return new, attn_res_norm(new, bank, mw)

    def moe_fold_norm(
        prefix: torch.Tensor,
        m: _MoeOut | _MoeTail,
        bank: torch.Tensor,
        mw: _MixW,
        updated: Optional[torch.Tensor] = None,
    ):
        """``moe_norm`` with the MoE residual update inside the persistent mix where it serves the
        rows (one kernel instead of the update's and the mix's); a tail goes to ``moe_norm``."""
        if not isinstance(m, _MoeTail) and use_x(prefix.shape[0], bank.shape[0]):
            r = x_mix(prefix, m.routed, m.shared, bank, mw)
            return r.updated, r.output
        return moe_norm(prefix, m, bank, mw, updated)

    def moe_norm_mx(
        prefix: torch.Tensor,
        m: _MoeOut | _MoeTail,
        bank: torch.Tensor,
        mw: _MixW,
        updated: Optional[torch.Tensor] = None,
    ):
        """``moe_fold_norm`` with the normed rows' MXFP8 form where a mix that stores it serves the
        rows: ``(prefix, (normed, codes, scales))``, E4M3 codes and R128c4 scales; else
        ``moe_norm``'s ``(prefix, normed)``.

        The PDL kernel stores it up to one scale tile where the QKVG GEMM would quantize the rows
        apart (``attn_res_mx_pdl``, ``_kda_mx_pdl_rows``) in that GEMM's own quantization of
        ``normed`` (``_kda_mx_exact``), the persistent mix past ``attn_res_pdl_max_m`` rows
        (``attn_res_mx_moe``) as ``mxfp8_quantize(normed, True, 32)``. A ``_MoeTail`` takes the PDL
        mix that finishes its call, with the MXFP8 form on the same rows, else ``moe_norm``.
        ``updated``: as for ``moe_norm``.
        """
        rows = prefix.shape[0]
        if isinstance(m, _MoeTail):
            if mx_pdl and bank.shape[0] and rows <= mx_pdl_max_m and _kda_mx_pdl_rows(rows):
                return tail_mix(prefix, m, bank, mw, True, updated)
            return moe_norm(prefix, m, bank, mw, updated)
        if mx_pdl and bank.shape[0] and rows <= mx_pdl_max_m and _kda_mx_pdl_rows(rows):
            new, normed, codes, scales = rubin_attn_res_moe_add_rmsnorm_mx(
                prefix,
                m.routed,
                m.shared,
                bank,
                mw.proj,
                mw.res_norm,
                mw.out_norm,
                eps,
                eps,
                False,  # entry_trigger
                True,  # mx_atom: one 128-row tile's atoms are its R128c4 scales
                1,  # mx_atom_blocks
                early_trigger=early,
                mx_exact=_kda_mx_exact(rows),
                updated=updated,
            )
            logger.info_once(_KDA_MX_PDL_LINE, key=_KDA_MX_PDL_LINE)
            return new, (normed, codes, scales)
        if mx_moe and use_x(rows, bank.shape[0]):
            r = x_mix(prefix, m.routed, m.shared, bank, mw, mx=True)
            return r.updated, (r.output, r.codes, r.scales)
        return (moe_fold_norm if moe_fold else moe_norm)(prefix, m, bank, mw, updated)

    return dict(
        rms_norm=rms_norm,
        attn_res_mix=attn_res_mix,
        attn_res_norm=attn_res_norm,
        add_norm=add_norm,
        moe_add=moe_add,
        moe_norm=moe_norm,
        moe_fold_norm=moe_fold_norm if moe_fold else None,
        moe_norm_mx=moe_norm_mx if mx_moe or mx_pdl else None,
    )


#: Logged once per process by the first mix that finishes the pushed o_proj partials'
#: all-reduce before its grid wait (``oproj_ar_early_poll``).
_OPROJ_EARLY_POLL_LINE = (
    "Kimi K3.1 Rubin o_proj all-reduce: the mix finishes the pushed partials before its grid wait"
)


def _tp_reduce_slots(
    c: Any, eps: float, all_reduce: Any, push: Any, mix: dict
) -> dict[str, Callable]:
    """The slots that finish a partial sum over the attention TP group.

    With ``push`` (the Lamport push workspace ``_build_runtime`` built where ``ar_push_amid`` and
    ``ar_finish_fused`` are on), a partial of at most ``attn_res_pdl_max_m`` rows is pushed into
    its Lamport buffers (a copy kernel; ``kda_oproj_push``'s KDA o_proj stores its partial there
    from its epilogue instead and finishes through ``_oproj_push_slot``) and the mix after it
    finishes the all-reduce in its own kernel (``oproj_ar_early_poll``: before its wait, logging
    ``_OPROJ_EARLY_POLL_LINE`` once per process). Anything else is the plain all-reduce, then the
    add-and-mix slot.
    """
    pdl_max_m = c["attn_res_pdl_max_m"] if c.get("attn_res_pdl") == 2 else 0
    early_poll = bool(c.get("oproj_ar_early_poll", False))
    early = bool(c.get("attn_res_early_trigger", False))
    add_norm = mix["add_norm"]

    def reduce(partial: torch.Tensor) -> torch.Tensor:
        return allreduce_module(partial, all_reduce, _AR_NONE, None, None)

    def reduce_norm(prefix: torch.Tensor, partial: torch.Tensor, bank: torch.Tensor, mw: _MixW):
        rows, hidden = prefix.shape
        if push is not None and bank.shape[0] and 0 < rows <= pdl_max_m:
            ws = push.view
            one_shot, chunk = push.transport(rows)
            rubin_lamport_push(ws, partial.contiguous(), one_shot, push.pdl)
            if early_poll:
                logger.info_once(_OPROJ_EARLY_POLL_LINE, key=_OPROJ_EARLY_POLL_LINE)
            return rubin_attn_res_lamport_add_rmsnorm(
                prefix,
                ws.uc_local,
                ws.flags.data_ptr(),
                bank,
                mw.proj,
                mw.res_norm,
                mw.out_norm,
                eps,
                eps,
                ws.world,
                chunk,
                not one_shot,  # finish_two_shot
                ws.mc_base,
                ws.rank,
                False,  # add_rn: the fast-math add, as the other mix kernels
                early_poll,
                False,  # entry_trigger
                False,  # finish_ag: the call carries no all-gather rows
                early_trigger=early,
            )
        return add_norm(prefix, reduce(partial), bank, mw)

    return dict(reduce=reduce, reduce_norm=reduce_norm)


#: Logged once per process by the first KDA verify whose o_proj epilogue pushes its partial
#: (``kda_oproj_push``).
_KDA_OPROJ_PUSH_LINE = (
    "Kimi K3.1 Rubin KDA: o_proj's epilogue pushes its partial into the all-reduce (no push kernel)"
)
#: Logged once per process if a pushed o_proj call runs the plain GEMM and the push kernel instead.
_KDA_OPROJ_PUSH_FALLBACK_LINE = (
    "Kimi K3.1 Rubin KDA: an o_proj push ran the GEMM and the push kernel (no epilogue push)"
)


def _oproj_push_slot(c: Any, eps: float, push: Any) -> Optional[_OprojPush]:
    """``kda_oproj_push`` / ``dsa_oproj_push``: the push workspace and the slot that finishes a
    pushed o_proj partial.

    A KDA verify step's o_proj of 1 to ``attn_res_pdl_max_m`` rows stores its partial into the
    Lamport slots itself (``rubin_mxfp8_linear_prequantized_push``) where ``_tp_reduce_slots``'s
    ``reduce_norm`` would push it with the copy kernel; ``finish_norm`` is the mix call of that
    ``reduce_norm`` after the push, with the same arguments. A DSA layer's o_proj does the same in
    ``dsa_oproj_push_bands`` (``_dsa_oproj_push``). None without the push workspace or with both
    keys off. Builds the row-push descriptor (eagerly, with the runtime).
    """
    pdl_max_m = c["attn_res_pdl_max_m"] if c.get("attn_res_pdl") == 2 else 0
    dsa = bool(c.get("dsa_oproj_push", False))
    pushers = c.get("kda_oproj_push", False) or dsa
    if push is None or not pushers or pdl_max_m <= 0:
        return None
    dsa_bands = _moe_row_bands(c["dsa_oproj_push_bands"], _DSA_OPROJ_PUSH_MAX_M) if dsa else ()
    early_poll = bool(c.get("oproj_ar_early_poll", False))
    early = bool(c.get("attn_res_early_trigger", False))

    def finish_norm(prefix: torch.Tensor, bank: torch.Tensor, mw: _MixW):
        ws = push.view
        one_shot, chunk = push.transport(prefix.shape[0])
        if early_poll:
            logger.info_once(_OPROJ_EARLY_POLL_LINE, key=_OPROJ_EARLY_POLL_LINE)
        return rubin_attn_res_lamport_add_rmsnorm(
            prefix,
            ws.uc_local,
            ws.flags.data_ptr(),
            bank,
            mw.proj,
            mw.res_norm,
            mw.out_norm,
            eps,
            eps,
            ws.world,
            chunk,
            not one_shot,  # finish_two_shot
            ws.mc_base,
            ws.rank,
            False,  # add_rn: the fast-math add, as the other mix kernels
            early_poll,
            False,  # entry_trigger
            False,  # finish_ag: the call carries no all-gather rows
            early_trigger=early,
        )

    return _OprojPush(
        ws=push,
        desc=rubin_lamport_row_push_desc(push),
        max_rows=pdl_max_m,
        finish_norm=finish_norm,
        dsa_bands=dsa_bands,
    )


#: Logged once per process by the first block boundary whose bank snapshot its producer stored
#: (``mix_bank_in_mix``: the mix's updated prefix, or layer 0's embedding gather), no copy.
_MIX_BANK_LINE = (
    "Kimi K3.1 Rubin attn_res: a block boundary's bank snapshot is stored by its producer (no copy)"
)
#: Logged once per process by the first capture copy issued on the tap stream (``draft_tap_stream``).
_TAP_STREAM_LINE = (
    "Kimi K3.1 Rubin DFlash2: the drafter's capture copies run on the tap stream (draft_tap_stream)"
)
#: Logged once per process by the first block boundary whose o_proj all-reduce the mix finishes as
#: the prefix sum (``boundary_ar_push``).
_BOUNDARY_PUSH_LINE = (
    "Kimi K3.1 Rubin attn_res: a block boundary's o_proj all-reduce is pushed and finished by the "
    "mix as the prefix (boundary_ar_push)"
)
#: Logged once per process by layer 0's dense MLP when it leaves its all-reduce to the next mix
#: (``dense_ar_push``).
_DENSE_PUSH_LINE = (
    "Kimi K3.1 Rubin dense MLP: the output goes to the next mix unsummed, which pushes it "
    "(dense_ar_push)"
)


class _TapStream:
    """``draft_tap_stream``: the drafter's capture copies on a stream of their own (``rt.taps``).

    ``copy`` forks the tap stream from the current stream after the kernel that made the tapped
    rows, issues the capture there and keeps the rows' memory alive on the tap stream
    (``record_stream``: the caching allocator, in a capture too, hands that block to no later
    allocation before the copy is done). ``join`` makes a stream wait for every copy issued so far:
    the drafter's early projection joins on the aux stream (``DraftModel.issue_projection``), any
    other reader of the captures on the current stream (the worker), and the shell's forward at
    its end, so that a step leaves no stream unjoined. The same copies into the same buffer.
    """

    def __init__(self, stream: torch.cuda.Stream) -> None:
        self.stream = stream
        self.done = torch.cuda.Event()  # recorded on the tap stream after each copy
        self.pending = False  # copies issued that no stream has joined yet

    def copy(self, tap: tuple, rows: torch.Tensor) -> None:
        """Capture ``rows`` (the tapped layer's stream) for the drafter on the tap stream."""
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            tap[0].maybe_capture_hidden_states(tap[1], rows, None)
            self.done.record()
        rows.record_stream(self.stream)
        self.pending = True
        logger.info_once(_TAP_STREAM_LINE, key=_TAP_STREAM_LINE)

    def join(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        """Have ``stream`` (the current one by default) wait for the copies issued so far."""
        if self.pending:
            (stream or torch.cuda.current_stream()).wait_event(self.done)
            self.pending = False


class _BoundaryPush:
    """``boundary_ar_push``: a block boundary's o_proj all-reduce, pushed by o_proj's epilogue
    (``kda_verify`` with ``push``), finished by the mix after the attention as the new prefix sum
    (``rubin_attn_res_lamport_add_rmsnorm`` with ``replace``), on 1 to ``max_rows`` rows: the bits
    of ``reduce`` (the C++ all-reduce) followed by ``attn_res_norm`` (rubin_attn_res_rmsnorm).

    Device check (``mode``, ``check``): the first eager call at a row count and bank size runs both
    on the same partial (the copy push, whose slots hold the words o_proj's epilogue stores) and
    compares the prefix and the normed rows word for word; one all-reduce of the mismatch flags
    gives every rank the same verdict, so the group takes one path. A pass pushes at that shape
    from then on (INFO), a mismatch keeps today's calls there (WARNING); a capture before the check
    keeps today's calls, unrecorded.
    """

    def __init__(
        self, push: Any, max_rows: int, reduce: Callable, norm: Callable, c: Any, eps: float
    ) -> None:
        self._push = push  # the LamportPushWs of the TP all-reduce
        self.max_rows = max_rows
        self._reduce = reduce  # rt.reduce: the C++ all-reduce
        self._norm = norm  # rt.attn_res_norm
        self._eps = eps
        self._early_poll = bool(c.get("oproj_ar_early_poll", False))
        self._early = bool(c.get("attn_res_early_trigger", False))
        self.verdict: dict[tuple[int, int], bool] = {}  # (rows, candidates) -> the check passed

    def mode(self, rows: int, candidates: int) -> Optional[str]:
        """``"push"`` (checked), ``"check"`` (unchecked, eager) or None (today's calls)."""
        if not 0 < rows <= self.max_rows:
            return None
        ok = self.verdict.get((rows, candidates))
        if ok is not None:
            return "push" if ok else None
        return None if torch.cuda.is_current_stream_capturing() else "check"

    def finish(self, prefix: torch.Tensor, bank: torch.Tensor, mw: _MixW):
        """``(AR(o_proj), normed)``: the mix that finishes the pushed call as the prefix sum
        (``prefix``, the sum it replaces, gives the rows and is not read)."""
        ws = self._push.view
        one_shot, chunk = self._push.transport(prefix.shape[0])
        logger.info_once(_BOUNDARY_PUSH_LINE, key=_BOUNDARY_PUSH_LINE)
        return rubin_attn_res_lamport_add_rmsnorm(
            prefix,
            ws.uc_local,
            ws.flags.data_ptr(),
            bank,
            mw.proj,
            mw.res_norm,
            mw.out_norm,
            self._eps,
            self._eps,
            ws.world,
            chunk,
            not one_shot,  # finish_two_shot
            ws.mc_base,
            ws.rank,
            False,  # add_rn: no add (replace)
            self._early_poll,
            False,  # entry_trigger
            False,  # finish_ag: the call carries no all-gather rows
            early_trigger=self._early,
            replace=True,
        )

    def check(self, partial: torch.Tensor, prefix: torch.Tensor, bank: torch.Tensor, mw: _MixW):
        """Today's ``(reduce(partial), attn_res_norm(...))``, after the device check at this shape
        (eager, every rank at the same call)."""
        rows, candidates = prefix.shape[0], bank.shape[0] + 1
        want_prefix = self._reduce(partial)
        want = self._norm(want_prefix, bank, mw)
        one_shot, _ = self._push.transport(rows)
        rubin_lamport_push(self._push.view, partial.contiguous(), one_shot, self._push.pdl)
        got_prefix, got = self.finish(prefix, bank, mw)
        same = torch.equal(got_prefix.view(torch.int16), want_prefix.view(torch.int16))
        same = same and torch.equal(got.view(torch.int16), want.view(torch.int16))
        flags = partial.new_zeros(1, partial.shape[1])
        flags[0, 0] = 0.0 if same else 1.0
        ok = float(self._reduce(flags)[0, 0]) == 0.0  # 0 / 1 per rank: exact in BF16
        self.verdict[(rows, candidates)] = ok
        line = (
            f"Kimi K3.1 Rubin attn_res: boundary_ar_push at {rows} rows, {candidates} candidates: "
        )
        if ok:
            logger.info(
                line + "the pushed finish equals the C++ all-reduce and the mix, word for word"
            )
        else:
            logger.warning(
                line + "a word differs on some rank; the C++ all-reduce and the mix stay"
            )
        return want_prefix, want


def _key_shard_ws(mapping: Any, md: Any) -> Any:
    """The key-shard merge's workspace over ``mapping`` (the CP group as TP).

    Sized as the step metadata checks it: ``md.key_shard_rows_max`` rows (max batch x (max draft
    + 1)) of ``md.key_shard_max_local_len`` local keys (the cache manager's pages x 64).
    Collective: every rank's first decode step builds it at the same point, eagerly, before any
    CUDA graph capture. With dsa_ks_select_pack the buffer also holds the select / pack's LL
    region after the local logits.
    """
    assert not torch.cuda.is_current_stream_capturing(), "the key-shard workspace is built eagerly"
    return rubin_dsa_key_shard_ws(
        mapping,
        md.key_shard_rows_max,
        _K31_DSA["index_topk"],
        md.key_shard_max_local_len,
        # Only when on: with it off the call is today's.
        **({"select_pack": True} if _KS_SELECT_PACK else {}),
    )


def _key_shard_select(ws: Any) -> Any:
    """dsa_ks_select_pack: helix16's select / pack runner over the key-shard workspace ``ws``, or
    None where its kernels do not take the workspace's shape (warned once; today's chain runs).
    Every rank decides alike (the same shapes). Eager, with the workspace."""
    sel = rubin_dsa_key_shard_select_ws(ws, _KS_SELECT_CLUSTER_MAX)
    if sel is None:
        logger.warning_once(_KS_SELECT_UNSUPPORTED, key=_KS_SELECT_UNSUPPORTED)
    return sel


def _build_runtime(core: Any) -> _Rt:
    """Build the runtime resources: streams, events, communication closures, the RoPE table.

    Collective over the helix group as TP ranks (the all-reduce workspaces, the column-parallel
    MoE tail's): every rank runs it at the same point. The key-shard merge's workspace takes the
    step metadata's sizes, so the first decode step builds it (``_key_shard_ws``);
    ``DraftModel.bind_runtime`` sets ``rt.draft``.
    """
    c = _CONSTANTS[core.role]
    d = core.dims
    all_reduce = rubin_all_reduce(
        core.tp_mapping, core.model_config.allreduce_strategy, _WORLD_SIZE
    )
    push = None
    if c.get("lamport_push_allreduce") and c.get("ar_push_amid") and c.get("ar_finish_fused"):
        # The Lamport push workspace of the o_proj partials, for the PDL mix's rows.
        pdl_max_m = c["attn_res_pdl_max_m"] if c.get("attn_res_pdl") == 2 else 0
        push = rubin_lamport_push_workspace(all_reduce, pdl_max_m, _K31["hidden_size"])
    mix = _mix_slots(c, core.eps)
    tp = _tp_reduce_slots(c, core.eps, all_reduce, push, mix)
    # weights.py's ``dims`` splits layer 0's MLP over the whole TP group, or not at all.
    dense_split = _K31["intermediate_size"] // d.dense_inter
    assert dense_split in (1, core.tp_mapping.tp_size), f"no all-reduce group for {dense_split}"
    s = core.streams
    oproj_push = _oproj_push_slot(c, core.eps, push)
    boundary_push = None
    if c.get("boundary_ar_push") and oproj_push is not None:
        boundary_push = _BoundaryPush(
            push, oproj_push.max_rows, tp["reduce"], mix["attn_res_norm"], c, core.eps
        )
    latent_ws = None
    if c.get("latent_ar_own_ws"):
        latent_ws = rubin_latent_ar_workspace(
            core.tp_mapping,
            _WORLD_SIZE,
            d.latent,
            c["latent_ar_own_ws_max_m"],
            c["latent_ar_oneshot_max_bytes"],
        )
    moe_comm = _MoeComm(
        all_reduce=all_reduce,
        dense_all_reduce=all_reduce if dense_split > 1 else None,
        latent_ws=latent_ws,
        latent_stream=s.latent,
        prezero_stream=s.prezero,
        **{f: torch.cuda.Event() for f in _MoeComm._fields[5:]},
    )
    ull_tail = None
    if c.get("moe_cp_tail"):
        # On the MoE's TP group (the helix ranks as TP ranks), whose all-reduces rt.moe_comm runs.
        built = rubin_ull_tail_workspace(core.tp_mapping, _WORLD_SIZE, _MOE_CP_TAIL_MAX_M)
        c0, n, tactics = rubin_ull_cp_tail_share(
            _WORLD_SIZE, core.tp_mapping.tp_rank, _MOE_CP_TAIL_MAX_M
        )
        ull_tail = _UllTail(
            ws=built["ws"],
            handle=built["handle"],
            c0=c0,
            n=n,
            tactics=tactics,
            cpp_one_shot=built["cpp_one_shot"],
            chunk=built["chunk"],
        )
    return _Rt(
        dsa_pools=_DsaPools(d.mla_layers, core.rope_positions),
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
        reduce_norm=tp["reduce_norm"],
        attn_res_mix=mix["attn_res_mix"],
        rms_norm=mix["rms_norm"],
        moe_add=mix["moe_add"],
        reduce=tp["reduce"],
        all_reduce=all_reduce,
        moe_comm=moe_comm,
        cp_mapping=core.cp_mapping,
        cp_group=core.tp_mapping,
        ks_ws=None,
        draft=None,
        moe_fold_norm=mix["moe_fold_norm"],
        moe_norm_mx=mix["moe_norm_mx"],
        ull_tail=ull_tail,
        oproj_push=oproj_push,
        taps=_TapStream(s.tap) if c.get("draft_tap_stream") else None,
        boundary_push=boundary_push,
        dense_push=bool(c.get("dense_ar_push")) and push is not None and dense_split > 1,
    )


# --------------------------------------------------------------------------------------------------
# KDA: a KDA layer's one-token decode and DFlash2 verify block (helix8).
#
# Helix8: the KDA heads split 8 ways, so this rank runs 12 of the 96 (K = V = 128) on every row of
# the step and returns its ``o_proj`` partial sum; the norm slot all-reduces it. Each block:
#
# * projects once: one MXFP8 GEMM for the fused ``[q | k | v | g | f_a | b]`` rows (the checkpoint's
#   ``in_proj_qkvgfab``), then the BF16 ``f_b`` GEMM on its ``f_a`` columns, which gives the
#   forget-gate logits;
# * runs the gated delta rule on the cache manager's FP32 state pool: the one-token decode kernel
#   (decode), or the fused verify kernel, which commits the state after the verified token and
#   keeps the drafts for the next step's replay (verify);
# * applies the gated output RMSNorm in the kernel epilogue, then the MXFP8 ``o_proj`` GEMM. With
#   ``kda_oproj_quant_k32`` the verify epilogue also quantizes o_proj's input (32-value scales,
#   bitwise ``rubin_mxfp8_linear``'s quantization of the BF16 output: the entry's
#   ``mxfp8_exact``), so the verify blocks run no separate quantization kernel.
#
# Generation levers: the verify kernel's early trigger (with ``kda_verify_pdl``: o_proj released
# right after the verify's grid wait); its PDL launch behind f_b (``kda_verify_pdl``); the
# forget-gate logits computed in the verify kernel from f_a, without the f_b GEMM, once the row
# count's device check passed (``kda_verify_fused_fb``: the verify is then the QKVG GEMM's PDL
# dependent); o_proj's epilogue push into the all-reduce on the push-path rows (``kda_oproj_push``:
# kda_verify returns ``_Pushed``); and, up to 16 verify rows, the L2 prefetch of the next KDA
# layer's weights on the aux stream while the recurrence runs.
#
# A decode or verify block takes ``x`` as the BF16 rows or, with ``attn_res_mx_pdl`` where the PDL
# mix serves the rows (up to one 128-row tile) and the projection would quantize them apart
# (``_kda_mx_pdl_rows``), and with ``attn_res_mx_moe`` where the persistent mix serves them, as
# ``(rows, codes, scales)``: the rows with the MXFP8 form the mix stored (E4M3 codes and
# R128c4 scales; the PDL mix's bit for bit the projection's own quantization of the rows,
# ``_kda_mx_exact``, the persistent mix's ``mxfp8_quantize``'s); the fused projection then
# multiplies those codes and skips its quantization.
#
# Reads of the layer view (``st.cache.mamba_layer_cache(L.layer)``): ``conv`` and ``temporal``; in
# verify, ``kda_conv_q`` / ``_k`` / ``_v``, ``kda_k_cache``, ``kda_g_cache``, ``kda_v_cache``,
# ``kda_beta_cache`` and ``prev_num_accepted_tokens``; after decode, ``commit_conv_window``
# (``_kda_commit_window``): the engine builds a ``KDAReplayLayerCache`` on a generation server.
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
# Rows of the fused projection's weight: the columns above and zero rows to a multiple of 16
# (weights.py pads them so), the N the MXFP8 GEMM sees.
_KDA_FAB_N = -(-_KDA_B1 // 16) * 16
_KDA_LOWER_BOUND = _K31_KDA["gate_lower_bound"]
_KDA_EPS = _K31["rms_norm_eps"]
_KDA_SCALE = _KDA_D**-0.5
# Read from the logged constant table, so that the log states what runs.
_KDA_EARLY_TRIGGER = _CONSTANTS["generation"]["kda_decode_early_trigger"]
_KDA_PREFETCH_ROWS = 16 if _CONSTANTS["generation"]["kda_l2_prefetch"] else 0
_KDA_OPROJ_K32 = _CONSTANTS["generation"]["kda_oproj_quant_k32"]
_KDA_OPROJ_K32_LINE = (
    "Kimi K3.1 Rubin KDA: the verify epilogue writes o_proj's MXFP8 input (32-value scales)"
)
_KDA_VERIFY_PDL = _CONSTANTS["generation"]["kda_verify_pdl"]
_KDA_VERIFY_PDL_LINE = (
    "Kimi K3.1 Rubin KDA: the verify launches as f_b's PDL dependent (state, replay entries and "
    "conv windows load before the grid wait)"
)
_KDA_OPROJ_PUSH = _CONSTANTS["generation"]["kda_oproj_push"]
# boundary_ar_push finishes the partial the KDA verify's o_proj epilogue pushes at a block
# boundary (kda_oproj_push, on the K32 path): without that push the boundary would run its
# collective device check and log its line, then never push.
if _CONSTANTS["generation"]["boundary_ar_push"] and not (_KDA_OPROJ_PUSH and _KDA_OPROJ_K32):
    raise NotImplementedError(
        "attn_res: boundary_ar_push needs kda_oproj_push and kda_oproj_quant_k32 (the KDA "
        "verify's o_proj epilogue pushes the boundary's partial)"
    )
# mix_bank_in_mix: a boundary's bank snapshot stored by its producer (read from the logged table).
_MIX_BANK_IN_MIX = _CONSTANTS["generation"]["mix_bank_in_mix"]


@functools.cache
def _device_sms(index: int) -> int:
    """The SM count of CUDA device ``index``."""
    return torch.cuda.get_device_properties(index).multi_processor_count


def _kda_verify_pdl(requests: int, sms: int) -> bool:
    """Whether a verify over ``requests`` requests launches with PDL (``kda_verify_pdl``).

    Only the v2 kernel has the PDL prologue, and it serves the ``mxfp8_exact`` output within one
    wave of this rank's (head, request) tiles on ``sms`` SMs; past it the two-CTA kernel takes the
    rows as a plain launch.
    """
    return bool(_KDA_VERIFY_PDL) and _KDA_H * requests <= sms


_KDA_VERIFY_FUSED_FB = _CONSTANTS["generation"]["kda_verify_fused_fb"]
_KDA_VERIFY_FUSED_FB_LINE = (
    "Kimi K3.1 Rubin KDA: the verify computes the forget gate from f_a (no f_b GEMM) and launches "
    "as the QKVG GEMM's PDL dependent"
)
#: Logged once per process and row count by the fused forget gate's device check: it passed (INFO),
#: or it failed and the f_b GEMM runs (WARNING).
_KDA_FUSED_FB_LINE = (
    "Kimi K3.1 Rubin KDA: the verify's forget gate from f_a at {rows} rows writes the f_b GEMM "
    "path's output and pool words"
)
_KDA_FUSED_FB_OFF = (
    "Kimi K3.1 Rubin KDA: the verify's forget gate from f_a at {rows} rows: {why}; "
    "the f_b GEMM runs"
)
#: Per process: verify rows -> whether the fused forget gate's device check passed (absent: not
#: checked yet).
_KDA_FUSED_FB_CHECKED = {}
#: The layer view's pools a verify writes in place, at its requests' slots.
_KDA_VERIFY_POOLS = (
    "temporal",
    "kda_conv_q",
    "kda_conv_k",
    "kda_conv_v",
    "kda_k_cache",
    "kda_g_cache",
    "kda_v_cache",
    "kda_beta_cache",
)


def _kda_verify_fused_fb(pdl: bool, rows: int) -> Optional[bool]:
    """Whether a verify of ``rows`` rows computes its forget gate in the kernel
    (``kda_verify_fused_fb``): True once this process's device check of ``rows`` rows passed;
    None while the check is still to run (an eager call runs it); False with the constant off,
    without ``pdl`` (the v2 kernel's one wave), after a failed check, and in a capture before the
    check."""
    if not (_KDA_VERIFY_FUSED_FB and pdl):
        return False
    checked = _KDA_FUSED_FB_CHECKED.get(rows)
    if checked is None and torch.cuda.is_current_stream_capturing():
        return False
    return checked


def _kda_words(t: torch.Tensor) -> torch.Tensor:
    """``t`` as integers of its element size, so that equality is equality of the bits."""
    return t.view({1: torch.uint8, 2: torch.int16, 4: torch.int32}[t.element_size()])


def _kda_fused_fb_check(
    rows: int, lc: Any, slots: torch.Tensor, mx_out: bool, today: Callable, fused: Callable
) -> Any:
    """``today()``'s output for this verify (the f_b GEMM and the verify), once the fused forget
    gate at ``rows`` rows was checked on its operands: ``fused()`` runs first on the same pools,
    whose rows at ``slots`` are then restored; its output (with ``mx_out`` the E4M3 codes and the
    written scale bytes) and the pool rows it wrote must equal today's word for word. Records and
    logs the outcome."""
    idx = slots.long()
    pools = [getattr(lc, name) for name in _KDA_VERIFY_POOLS]
    saved = [p[idx] for p in pools]
    why = ""
    try:
        got = fused()
    except ValueError as exc:
        why = f"the fused call is refused ({exc})"
    else:
        got_rows = [p[idx] for p in pools]
    for p, s in zip(pools, saved):
        p[idx] = s
    want = today()
    if not why:
        if mx_out:
            same = torch.equal(_kda_words(got[0]), _kda_words(want[0])) and torch.equal(
                mxfp8_scales_from_r128c4(got[1], rows, _KDA_P),
                mxfp8_scales_from_r128c4(want[1], rows, _KDA_P),
            )
        else:
            same = torch.equal(_kda_words(got), _kda_words(want))
        bad = [
            name
            for name, p, r in zip(_KDA_VERIFY_POOLS, pools, got_rows)
            if not torch.equal(_kda_words(r), _kda_words(p[idx]))
        ]
        if not same:
            why = "its output is not the f_b GEMM path's word for word"
        elif bad:
            why = f"its {', '.join(bad)} rows are not the f_b GEMM path's word for word"
    if why:
        logger.warning(_KDA_FUSED_FB_OFF.format(rows=rows, why=why))
    else:
        logger.info(_KDA_FUSED_FB_LINE.format(rows=rows))
    _KDA_FUSED_FB_CHECKED[rows] = not why
    return want


@functools.cache
def _kda_mx_exact(rows: int) -> bool:
    """The ``mx_exact`` flag of the MXFP8 form a mix hands to the fused projection for ``rows``
    rows (up to one 128-row tile): the quantization ``rubin_mxfp8_linear`` applies to the BF16 rows
    there, so that ``rubin_mxfp8_linear_prequantized`` on the form is bitwise
    ``rubin_mxfp8_linear`` on the rows.

    False: the shared low-M / mid-M GEMM runs and quantizes the rows itself (in the GEMM up to its
    low-M rows, else in its quantization kernel), each 1x32 block's amax floored at ~1e-4. True:
    the dense GEMM runs after the catalog ``mxfp8_quantize`` (no floor). Up to one tile the GEMM is
    the one whose activation scale layout ``mxfp8_act_layout`` names, R128c4 for the dense GEMM.
    """
    assert rows <= _MX_TILE_ROWS, f"{rows} rows: past one tile the layout does not name the GEMM"
    return mxfp8_act_layout(rows, _KDA_FAB_N, _K31["hidden_size"])[1] == "r128c4"


@functools.cache
def _kda_mx_pdl_rows(rows: int) -> bool:
    """Whether the PDL mix stores the MXFP8 form of ``rows`` normed rows (``attn_res_mx_pdl``):
    where ``rubin_mxfp8_linear`` on the BF16 rows would run a quantization kernel ahead of the
    fused projection's GEMM, the kernel the form saves (``mxfp8_quantizes_in_gemm`` false).

    Up to the low-M GEMM's low-M rows (32 for the ``[6288, 7168]`` stack on sm_107) the GEMM
    quantizes the rows inside the kernel: the form would save no kernel and only add the mix's
    code and scale stores, so those rows keep the mix without the form.
    """
    return not mxfp8_quantizes_in_gemm(rows, _KDA_FAB_N, _K31["hidden_size"])


def _kda_project(
    L: _KdaW,
    x: torch.Tensor,
    mx: tuple[torch.Tensor, torch.Tensor] | None = None,
    gate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return the fused projection ``y`` [rows, N] and the forget-gate logits [rows, P] (None
    without ``gate``: the verify kernel computes them from the f_a columns).

    ``mx``: the MXFP8 form of ``x`` (codes, R128c4 scales) when the mix stored it.
    """
    if mx is None:
        y = rubin_mxfp8_linear(x, L.qkvgfab_w, L.qkvgfab_s)
    else:
        y = rubin_mxfp8_linear_prequantized(mx[0], mx[1], L.qkvgfab_w, L.qkvgfab_s)
    return y, (_kda_fb(L, y) if gate else None)


def _kda_fb(L: _KdaW, y: torch.Tensor) -> torch.Tensor:
    """The forget-gate logits [rows, P]: the f_b GEMM on the f_a columns of the projection ``y``."""
    # F.linear reads the f_a columns through their row stride, so nothing is copied.
    return rubin_linear(y[:, _KDA_FA0:_KDA_B0], L.fb, False)


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


def _kda_ranges(tensors: tuple) -> list[tuple[int, int]]:
    """The 16-byte aligned ``(address, bytes)`` spans of dense tensors."""
    spans = []
    for t in tensors:
        start = t.data_ptr()
        first = (start + 15) // 16 * 16
        spans.append((first, max(0, (start + t.numel() * t.element_size() - first) // 16 * 16)))
    return spans


def _kda_commit_window(lc: Any, slots: torch.Tensor, conv: torch.Tensor) -> None:
    """Copy the live conv windows of ``slots`` into the view's replay caches, where it has them.

    Read as the engine's KDA mixer reads it: only a ``KDAReplayLayerCache`` has
    ``has_kda_replay_caches``. A disaggregated context server keeps no replay state; its
    generation server seeds the replay caches from the transferred conv state.
    """
    if getattr(lc, "has_kda_replay_caches", False):
        lc.commit_conv_window(slots, conv)


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
        apply_onorm=True,
        update_conv_cache=True,
        use_lower_bound=True,
        apply_beta_sigmoid=True,
        lower_bound=_KDA_LOWER_BOUND,
        scale=_KDA_SCALE,
        onorm_eps=_KDA_EPS,
        output=out,
    )
    _kda_commit_window(lc, slots, conv)
    return out.view(rows, _KDA_P)


def kda_verify(L: _KdaW, x: Any, st: _Step, rt: _Rt, push: bool = False) -> Any:
    """Run the KDA mixer over the DFlash2 verify blocks ``x``; return ``[rows, hidden]``.

    Commits the state after each request's accepted prefix and keeps its drafts for the next
    step's replay. ``push`` (the decoder's, where the mix after this layer finishes a pushed call):
    with ``kda_oproj_push`` on 1 to ``rt.oproj_push.max_rows`` rows, o_proj stores its partial
    into the Lamport slots of the all-reduce and the block returns ``_Pushed``.
    """
    x, mx = _kda_split(x)
    if _KDA_OPROJ_K32:
        codes, scales = _kda_verify_core(L, x, st, rt, mx_out=True, mx=mx)
        logger.info_once(_KDA_OPROJ_K32_LINE, key=_KDA_OPROJ_K32_LINE)
        rows = codes.shape[0]
        op = rt.oproj_push if _KDA_OPROJ_PUSH else None
        if push and op is not None and 0 < rows <= op.max_rows:
            one_shot, _ = op.ws.transport(rows)
            if rubin_mxfp8_linear_prequantized_push(
                codes, scales, L.o_w, L.o_s, op.ws.view, op.desc, one_shot, op.ws.pdl
            ):
                logger.info_once(_KDA_OPROJ_PUSH_LINE, key=_KDA_OPROJ_PUSH_LINE)
            else:
                logger.warning_once(
                    _KDA_OPROJ_PUSH_FALLBACK_LINE, key=_KDA_OPROJ_PUSH_FALLBACK_LINE
                )
            return _Pushed(rows)
        return rubin_mxfp8_linear_prequantized(codes, scales, L.o_w, L.o_s)
    return rubin_mxfp8_linear(_kda_verify_core(L, x, st, rt, mx=mx), L.o_w, L.o_s)


def _kda_verify_core(
    L: _KdaW,
    x: torch.Tensor,
    st: _Step,
    rt: _Rt,
    mx_out: bool = False,
    mx: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> Any:
    """The verify blocks' core ``[rows, P]`` (before o_proj), or with ``mx_out`` its MXFP8 form
    ``(E4M3 [rows, P], R128c4 UE8M0 bytes)`` in the catalog quantizer's 1x32 groups
    (``mxfp8_exact``): what ``rubin_mxfp8_linear`` makes of the core. ``mx``: the MXFP8 form of
    ``x`` when the mix stored it."""
    rows = x.shape[0]
    lc = st.cache.mamba_layer_cache(L.layer)
    slots = _kda_gen_slots(st)
    pdl = _kda_verify_pdl(slots.shape[0], _device_sms(x.device.index))
    # True: the kernel computes the forget gate from f_a; False: f_b runs; None: both, the check.
    fused = _kda_verify_fused_fb(pdl, rows)
    y, g = _kda_project(L, x, mx, gate=fused is False)
    if rows <= _KDA_PREFETCH_ROWS:
        # Fork the aux stream after the projections and join what it holds (the previous KDA
        # layer's prefetch, long done); then prefetch the next layer's weights while the
        # recurrence runs. The next KDA layer joins that prefetch.
        main = torch.cuda.current_stream()
        rt.aux_stream.wait_stream(main)
        main.wait_stream(rt.aux_stream)
        if L.l2_next:
            with torch.cuda.stream(rt.aux_stream):
                rubin_l2_prefetch(_kda_ranges(L.l2_next))

    def verify(g, fa, w_fb):
        # The entry fixes the packed token layout and the fused output norm. The gate input is g
        # (the f_b GEMM's output) or fa / w_fb (the kernel runs f_b on the f_a columns); o_proj
        # takes the BF16 output, or with mx_out its catalog-form MXFP8 output (mxfp8_exact, on the
        # kernel the BF16 output takes: with fa, the v2 kernel). With pdl the verify is the PDL
        # dependent of the GEMM before it: f_b, or with fa the QKVG GEMM, whose weight warp
        # releases its dependents at entry, so that the verify can start while the QKVG GEMM and
        # the mix before it run (the mix past its own grid wait: helix8's mixes take no entry
        # trigger).
        # Before its grid wait the verify reads only this layer's state and conv pools, replay
        # caches, accepted counts, slots and weights (with fa, also this head's f_b weight rows),
        # and writes nothing. The kernels it may overlap write activation rows only; this layer's
        # projections (with fa, the f_a columns) it reads after the wait, when the QKVG GEMM, and
        # so every grid before it, has completed.
        # The early release of o_proj (kda_decode_early_trigger with pdl: right after the grid
        # wait) acts on the v2 kernel only. Every grid before the verify has completed when it
        # fires; o_proj reads only its weights before its own wait, and the grids it releases
        # early (the push kernel, the C++ all-reduce) wait before they read or write anything,
        # while o_proj's push epilogue releases the mix only after its wait.
        return rubin_kda_mtp_verify(
            x_q=_kda_heads(y, 0),
            x_k=_kda_heads(y, _KDA_P),
            x_v=_kda_heads(y, 2 * _KDA_P),
            w_q=L.conv_f32[:_KDA_P],
            w_k=L.conv_f32[_KDA_P : 2 * _KDA_P],
            w_v=L.conv_f32[2 * _KDA_P :],
            cs_q=lc.kda_conv_q,
            cs_k=lc.kda_conv_k,
            cs_v=lc.kda_conv_v,
            g=g,
            fa=fa,
            w_fb=w_fb,
            beta=_kda_beta(y),
            A_log=L.a_log,
            dt_bias=L.dt_bias,
            recurrent_state=lc.temporal,
            k_cache=lc.kda_k_cache,
            g_cache=lc.kda_g_cache,
            v_cache=lc.kda_v_cache,
            beta_cache=lc.kda_beta_cache,
            ssm_state_indices=slots,
            num_spec=st.rows_per_gen - 1,
            num_accepted_tokens=lc.prev_num_accepted_tokens,
            lower_bound=_KDA_LOWER_BOUND,
            scale=_KDA_SCALE,
            onorm_g=_kda_heads(y, _KDA_G0),
            onorm_weight=L.o_norm_f32,
            onorm_eps=_KDA_EPS,
            quantize_output=mx_out,
            use_pdl=pdl,
            early_trigger=_KDA_EARLY_TRIGGER,
            mxfp8_pdl_early_trigger=False,
            mxfp8_exact=mx_out,
        )

    fa = y[:, _KDA_FA0:_KDA_B0].unsqueeze(0)
    g_shape = (1, rows, _KDA_H, _KDA_D)
    if fused:
        logger.info_once(_KDA_VERIFY_FUSED_FB_LINE, key=_KDA_VERIFY_FUSED_FB_LINE)
        core = verify(None, fa, L.fb)
    else:
        if pdl:
            logger.info_once(_KDA_VERIFY_PDL_LINE, key=_KDA_VERIFY_PDL_LINE)
        if fused is None:
            # Today's call runs the f_b GEMM right before its verify, so that the verify's PDL
            # launch follows the f_b GEMM as in the step, after the check restored the pools.
            core = _kda_fused_fb_check(
                rows,
                lc,
                slots,
                mx_out,
                today=lambda: verify(_kda_fb(L, y).view(g_shape), None, None),
                fused=lambda: verify(None, fa, L.fb),
            )
        else:
            core = verify(g.view(g_shape), None, None)
    if mx_out:
        return core
    return core.view(rows, _KDA_P)


# --------------------------------------------------------------------------------------------------
# MLA + DSA (helix8): the 24 DSA layers' projections, indexer, sparse MLA and output gate, flat.
#
# Every K3.1 MLA layer is a DSA layer, and both its attention and its indexer are NoPE: the MLA
# kernels read the identity RoPE table ``rt.mla_rope`` and the indexer rotates nothing
# (``index_rope_dim`` 0). helix8 serves generation steps only: every rank attends with all 96 query
# heads over its eighth of each request's latent KV, and the Helix combine leaves it 12 heads, which
# the gate and o_proj take. One layer, in order:
#
# * projections as on the other layouts (``cublas_mm`` for ``[q_a | kv_a]`` and the indexer's
#   ``wq_b``, ``rubin_linear`` for ``q_b``, ``g``, ``o`` and the FP32 ``[wk | weights_proj]`` with
#   TF32), the norms and the index-key quantization overlapped on graph steps;
# * the index keys of every live row into the cache, once; on graph steps the indexer's work on the
#   layer input alone (``[wk | weights_proj]``, the key norm and quantization, this insert) runs on
#   ``rt.idx_stream`` from the top of the layer, beside ``[q_a | kv_a]``, its norms and the
#   indexer's query projection, and joins before the logits weights take the query scales
#   (dsa_idx_fork);
# * the key-sharded Top-K where the step allows it: this rank's FP8 DSL logits over its own pages
#   (L2 evict-first) into the merge workspace, the CuTe local Top-K, the pack, and the merge that
#   keeps this rank's share of the exact global Top-K with its counts and combine mask
#   (``rubin_dsa_key_shard_*``). Otherwise the fallback: the local logits and Top-K, the candidates'
#   scores all-gathered over the CP group and merged (``rubin_merge_helix_topk``); or the dense
#   Top-K;
# * the Top-K remap to latent-pool rows (Triton from 256 rows, C++ below);
# * the absorbed sparse MLA: W_UK, the RoPE kernel with the helix arguments (position offsets,
#   inactive rank, local slots), the token-major sparse attention op with its softmax statistics;
#   on graph steps q_b, W_UK and the RoPE kernel run on ``rt.fork_stream`` beside the indexer and
#   join before the attention (fork_q);
# * W_UV on all 96 heads' partials before the exchange (128 values a head instead of 512), the
#   combine mask (the merge's, or one launch over the bounds and the kept counts), the combine;
# * the output gate ``attn * sigmoid(g_proj(x))`` and ``o_proj`` (``rubin_linear``). On graph steps
#   ``g_proj(x)`` follows the RoPE kernel on ``rt.fork_stream``, beside the indexer: the attention
#   joins the RoPE kernel's event only, the gate g's own event.
#
# ``[q_a | kv_a]``, the indexer's ``wq_b``, W_UV and ``o_proj`` run at their pinned cuBLASLt
# algorithms in gemm_pins_mla's rows (``_gemm_pinned``), the same bits.
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
# helix8: all 96 query heads attend.
_MLA_HEADS = _MLA_QUERY_HEADS
# AttentionInputType of the attention op's generation call.
_MLA_GENERATION = 2
# The helix combine runs the NCCL transport's reformat eagerly: no inductor compile inside a
# multi-rank forward (a row count first seen in serving would compile there). The MNNVL FIFO
# transport does not use it.
_MLA_COMBINE_COMPILE = False
#: Logged once per process by the first CUDA graph decode step: ``g_proj(x)`` was issued on the
#: fork stream after the RoPE kernel, off the path from the combine to ``o_proj``.
_MLA_GATE_FORK_LINE = (
    "Kimi K3.1 Rubin MLA: output gate projection on the fork stream after the RoPE kernel"
)

# The DSA indexer (_K31_DSA): 32 heads x 128, position-free, Top-2048.
_IDX_HEADS = _K31_DSA["index_num_heads"]
_IDX_DIM = _K31_DSA["index_head_dim"]
_IDX_HALF = _IDX_DIM // 2  # the cat-quantization takes two halves of each head
_IDX_TOPK = _K31_DSA["index_topk"]
_IDX_K_NORM_EPS = _K31["rms_norm_eps"]  # index_k_norm_eps (_check_build ties them)
# softmax scale x heads^-0.5, folded with the per-head query scale into the logits weights.
_IDX_WEIGHT_SCALE = _IDX_DIM**-0.5 * _IDX_HEADS**-0.5
#: Logged once per process by the first indexer projection: its prologue widens x with
#: full-width accesses and scales the logits weights in one kernel.
_IDX_SHORT_LINE = (
    "Kimi K3.1 Rubin DSA indexer: x widened with full-width accesses, "
    "(w * qs) * scale in one kernel"
)
# The decode logits arena's row stride is a multiple of the DSL kernel's 256-key split.
_IDX_LOGITS_ALIGN = 256
# The decode logits run one atom per request: next_n x index heads within one 256-wide MMA tile.
_IDX_DSL_MAX_ATOM_N = 256
# The Top-K remap runs the Triton kernel from this many rows on (the C++ op below it).
_IDX_REMAP_TRITON_ROWS = 256
# The self-sampling GVR Top-K's deterministic kernels (ties to the smaller index).
_IDX_GVR_DETERMINISTIC = True
# The radix Top-K's split workspace: blocks per row.
_IDX_RADIX_BLOCKS = 10
# The key-shard logits kernel's PDL launch (logged constant): only the launch and the waits move.
_IDX_LOGITS_PDL = _CONSTANTS["generation"]["dsa_idx_logits_pdl"]
#: Logged once per process by the first CUDA graph decode step: the indexer's keys and weights ran
#: on the idx stream, beside ``[q_a | kv_a]`` (dsa_idx_fork).
_IDX_FORK_LINE = (
    "Kimi K3.1 Rubin DSA: indexer key / weights prologue on its own stream beside [q_a | kv_a]"
)
# The key-shard merge's exact re-selection of a row whose certificate fails
# (dsa_key_shard_failing_fallback): False streams every rank's local logits; True only the failing
# ranks' and takes the other ranks' received candidates (the same rows, counts and mask).
_KS_FAILING_ONLY = bool(_CONSTANTS["generation"]["dsa_key_shard_failing_fallback"])
#: Logged once per process by the first key-sharded merge under dsa_key_shard_failing_fallback.
_KS_FAILING_LINE = (
    "Kimi K3.1 Rubin DSA key shard: a re-selected row reads only the failing ranks' local logits"
)
# dsa_ks_select_pack: the key-sharded Top-K in one launch of helix16's select / pack / merge (the
# KV pool rows, counts and mask; no local Top-K, pack, merge or remap launch) at the row counts
# whose first-use device check passed on every rank.
_KS_SELECT_PACK = bool(_CONSTANTS["generation"]["dsa_ks_select_pack"])
# CTAs per row of its local select at most (dsa_ks_select_cluster_max): the per-step cluster is
# min(this, SMs // rows).
_KS_SELECT_CLUSTER_MAX = int(_CONSTANTS["generation"]["dsa_ks_select_cluster_max"])
#: Per process: (rows, with bounds) -> whether the select / pack's check passed (absent: not checked).
_KS_SELECT_CHECKED = {}
#: Logged once per process by the first select / pack launch under dsa_ks_select_pack.
_KS_SELECT_LINE = (
    "Kimi K3.1 Rubin DSA key shard: helix16's select / pack / merge writes the KV-pool rows "
    "(cluster per step; checked word for word against the Triton chain per row count)"
)
_KS_SELECT_OFF = (
    "Kimi K3.1 Rubin DSA key shard: select / pack at {rows} rows: {why}; today's chain runs"
)
_KS_SELECT_UNSUPPORTED = (
    "Kimi K3.1 Rubin DSA key shard: helix16's select / pack does not take this workspace; "
    "today's chain runs"
)

# The attention op's layer constants for a K3.1 DSA layer: absorbed MLA over an FP8 latent cache
# (quant_mode 128), one KV head of 576, the identity RoPE table (rope_gpt_neox, 64 dims;
# ``_identity_rope_table``), values kv_lora wide. Unset KV scales read as 1.0 in the MLA kernels
# (the unit scales of the FP8 cache); the MLA path reads only the cos / sin table, no inverse
# frequencies. Skip-softmax, spcompress and sage attention are off.
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
    quant_scale_qkv=None,
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


def _mla_project(L: _MlaW, x: torch.Tensor, st: _Step, rt: _Rt) -> tuple:
    """``[q_a | kv_a]`` and the two RMSNorms on every row of ``x``: (normed q_a, latent row).

    On graph steps kv_a's norm runs on the side stream beside q_a's.
    """
    qkv_a = _gemm_pinned("mla_qkv_a", x, L.qkv_a_t, lambda: cublas_mm(x, L.qkv_a_t))
    q_a, ckv, k_pe = qkv_a.split([_MLA_Q_LORA, _MLA_KV_LORA, _MLA_ROPE], -1)
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


def _idx_keys(L: _MlaW, x: torch.Tensor) -> tuple:
    """The indexer's work on ``x`` alone: FP8 keys ``[rows, 128]`` with their FP32 scales, and the
    FP32 logits weights before the query scales ``[rows, 32]``, on every row of ``x``.

    Position-free: each key is quantized as its two halves; nothing rotates. On graph steps
    ``mla_generate`` runs it on ``rt.idx_stream`` (dsa_idx_fork).
    """
    ik, w = rubin_linear(rubin_upcast_fp32(x), L.idx_wk_wp, True).split([_IDX_DIM, _IDX_HEADS], -1)
    ik = ik.to(x.dtype)
    ik = rubin_layer_norm(ik, L.idx_k_norm_w, L.idx_k_norm_b, _IDX_K_NORM_EPS)
    ik_a, ik_b = ik.split([_IDX_HALF, _IDX_DIM - _IDX_HALF], -1)
    k8, ks = fused_cat_fp8(ik_a, ik_b, True)
    return k8, ks, w


def _idx_queries(L: _MlaW, qr: torch.Tensor) -> tuple:
    """The indexer's FP8 queries ``[rows, 32, 128]`` and their per-head FP32 scales ``[rows, 32]``.

    Position-free: each head is quantized as its two halves; nothing rotates.
    """
    iq = _gemm_pinned("mla_idx_wq_b", qr, L.idx_wq_b_t, lambda: cublas_mm(qr, L.idx_wq_b_t))
    iq = iq.view(-1, _IDX_HEADS, _IDX_DIM)
    iq_a, iq_b = iq.split([_IDX_HALF, _IDX_DIM - _IDX_HALF], -1)
    q8, qs = fused_cat_fp8(iq_a, iq_b, True)
    return q8.view(-1, _IDX_HEADS, _IDX_DIM), qs.view(-1, _IDX_HEADS)


def _idx_insert(i: int, k8: torch.Tensor, ks: torch.Tensor, n: int, st: _Step, rt: _Rt) -> None:
    """Insert the index keys of the step's ``n`` live rows into layer ``i``'s cache.

    Once a step, before the phases split (context chunk gathers read the keys just written).
    """
    indexer_k_cache_scatter_op(
        k8, ks, rt.dsa_pools.idx_k_cache[i], st.dsa.slot_fp8, st.dsa.slot_scale, n
    )


def _idx_fallback_logits(i: int, q8, w, st: _Step, rt: _Rt) -> torch.Tensor:
    """The fallback's local decode logits ``[rows, max_len]``.

    The FP8 DSL kernel without per-token bounds (one atom per request, L2 evict-first for a window
    of more than one row), the DeepGEMM paged logits with them (per-token context lengths).
    """
    d = st.dsa
    n = st.num_tokens
    if d.fb_dsl:
        assert d.next_n * _IDX_HEADS <= _IDX_DSL_MAX_ATOM_N, "a verify window wider than one atom"
        width = -(-d.max_len // _IDX_LOGITS_ALIGN) * _IDX_LOGITS_ALIGN
        out = torch.empty(n, width, dtype=torch.float32, device=q8.device)[:, : d.max_len]
        return rubin_fp8_paged_mqa_logits(
            q8[:n].view(st.num_generations, d.next_n, _IDX_HEADS, _IDX_DIM),
            rt.dsa_pools.idx_k_cache[i],
            w[:n],
            d.fb_ctx_lens,
            d.fb_block_table,
            d.fb_sched,
            d.max_len,
            1,
            None,
            d.next_n > 1,
            out,
        )
    return fp8_paged_mqa_logits(
        q8[:n].view(d.fb_block_table.shape[0], -1, _IDX_HEADS, _IDX_DIM),
        rt.dsa_pools.idx_k_cache[i],
        w[:n],
        d.fb_ctx_lens,
        d.fb_block_table,
        d.fb_sched,
        d.max_len,
    )


def _idx_decode(i: int, q8, w, st: _Step, rt: _Rt) -> tuple:
    """This rank's share of the generation rows' global Top-K, the merge's combine mask, and whether
    the rows are KV pool rows already.

    The rows are ``[rows, 2048]`` local token indices, kept first, then -1. Key-sharded where the
    step allows it; otherwise the fallback (eager steps): the local logits, the local Top-K (radix
    with the per-token bounds where the step has them, the rows a rank owns no KV for emptied),
    every rank's candidate scores all-gathered over the CP group and merged, the kept counts.
    Dense Top-K rows where the step skips the indexer. With dsa_ks_select_pack, at a checked row
    count, the key-sharded rows are the same rows as KV pool rows (``_ks_select``): no remap.
    """
    d = st.dsa
    n = st.num_tokens
    if d.skip_gen:
        return d.topk_dense[:n], None, False
    topk = d.topk_out[:n]
    if d.ks_next_n:
        ws = rt.ks_ws
        logits = rubin_fp8_paged_mqa_logits(
            q8[:n].view(st.num_generations, d.ks_next_n, _IDX_HEADS, _IDX_DIM),
            rt.dsa_pools.idx_k_cache[i],
            w[:n],
            d.ks_ctx_lens,
            d.ks_block_table,
            d.ks_sched,
            d.ks_local_len,
            1,
            None,
            True,
            ws.local_logits[:n, : d.ks_local_len],
            # Only when on: with it off the call is today's.
            **({"pdl": True} if _IDX_LOGITS_PDL else {}),
        )
        bounds = d.row_bounds if d.counts_out is not None else None
        mask = None if bounds is None else torch.empty(n, dtype=torch.bool, device=topk.device)
        if _KS_SELECT_PACK and rt.ks_sel is not None and _ks_select(i, topk, bounds, mask, st, rt):
            return topk, mask, True
        local = ws.local_topk[:n]
        rubin_topk_decode_cute(logits, d.ks_row_lens, local, ws.local_k, 1)
        rubin_dsa_key_shard_pack(local, d.ks_row_lens, ws)
        if _KS_FAILING_ONLY:
            logger.info_once(_KS_FAILING_LINE, key=_KS_FAILING_LINE)
            rubin_dsa_key_shard_merge(topk, d.counts_out, bounds, mask, ws, failing_only=True)
        else:
            rubin_dsa_key_shard_merge(topk, d.counts_out, bounds, mask, ws)
        return topk, mask, False
    logits = _idx_fallback_logits(i, q8, w, st, rt)
    radix_aux = (
        d.radix_aux_indices[:n, :_IDX_RADIX_BLOCKS, :_IDX_TOPK],
        d.radix_aux_logits[:n, :_IDX_RADIX_BLOCKS, :_IDX_TOPK],
    )
    if d.row_bounds is not None:
        indexer_topk_decode(
            logits, d.gen_kv_lens, topk, d.fb_topk_next_n, _IDX_TOPK, 1, *radix_aux, d.row_bounds
        )
        # A row this rank owns no KV for was scored over a placeholder slot: no candidates.
        topk.masked_fill_((d.row_bounds == 0).unsqueeze(1), -1)
    elif n > 1 or d.max_len % 4 == 0:
        rubin_gvr_topk_decode(
            logits, d.gen_kv_lens, topk, d.fb_topk_next_n, d.max_len, _IDX_GVR_DETERMINISTIC
        )
    else:
        indexer_topk_decode(
            logits, d.gen_kv_lens, topk, d.fb_topk_next_n, _IDX_TOPK, 1, *radix_aux, None
        )
    valid = topk >= 0
    scores = torch.gather(logits.float(), 1, topk.clamp(min=0).long())
    scores = scores.masked_fill(~valid, float("-inf"))
    # The merge reads no key-shard state: a step that takes it may have no workspace at all.
    gathered = allgather_module(scores.unsqueeze(0), rt.cp_group, 0, None)
    rubin_merge_helix_topk(gathered, topk, rt.cp_group.tp_rank)
    if d.counts_out is not None:
        torch.sum(topk >= 0, dim=1, dtype=torch.int32, out=d.counts_out)
    return topk, None, False


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


def _ks_kv_rows(i: int, st: _Step, rt: _Rt) -> Any:
    """``_idx_remap``'s arguments of layer ``i`` as the select / pack takes them."""
    d = st.dsa
    scale, layer_offset = rt.dsa_pools.pool_params[i]
    tpb = d.tokens_per_block
    return key_shard_kv_rows(d.block_table_gen, d.req_idx_gen, scale * tpb, layer_offset * tpb)


def _ks_select(i: int, topk, bounds, mask, st: _Step, rt: _Rt) -> bool:
    """dsa_ks_select_pack: whether helix16's select / pack / merge wrote this step's key-sharded rows
    into ``topk`` as KV pool rows (the counts into ``d.counts_out``, the mask into ``mask``), from
    the local logits of layer ``i`` already in the workspace; False: today's chain runs.

    It runs at a row count whose check passed. The first call at a row count (and bounds or none)
    in a process, outside capture, runs the check (``_ks_select_check``) and today's chain; a
    capture before that check keeps today's chain, unrecorded.
    """
    d = st.dsa
    n = topk.shape[0]
    if d.tokens_per_block != _KS_SELECT_PAGE:
        return False
    checked = _KS_SELECT_CHECKED.get((n, bounds is not None))
    if checked is None:
        if not torch.cuda.is_current_stream_capturing():
            _ks_select_check(i, n, bounds, st, rt)
        return False
    if not checked:
        return False
    ws = rt.ks_ws
    counts = d.counts_out if d.counts_out is not None else ws.counts[:n]
    rubin_dsa_key_shard_select_pack(
        rt.ks_sel, d.ks_row_lens, n, topk, counts, bounds, mask, _ks_kv_rows(i, st, rt)
    )
    logger.info_once(_KS_SELECT_LINE, key=_KS_SELECT_LINE)
    return True


def _ks_select_check(i: int, n: int, bounds, st: _Step, rt: _Rt) -> bool:
    """The first-use device check of the select / pack at ``n`` rows: compile its kernels for every
    cluster size the workspace's row counts take, then on the step's local logits and on tie-heavy,
    forced-fallback and full-width forms of them (``key_shard_select_check``) compare its KV rows,
    counts and mask word for word with today's chain (CuTe local Top-K, pack, merge, remap). Every
    rank must agree. Records and logs (a WARNING where it differs) the outcome; the step's logits
    are restored. Collective, eager."""
    d, ws, sel = st.dsa, rt.ks_ws, rt.ks_sel
    sel.precompile(
        ws.rows_max, bounds is not None, bounds is not None and bounds.dtype == torch.int64
    )
    kv = _ks_kv_rows(i, st, rt)
    dev = d.topk_out.device

    def outputs() -> tuple:
        return (
            torch.empty(n, _IDX_TOPK, dtype=torch.int32, device=dev),
            torch.empty(n, dtype=torch.int32, device=dev),
            None if bounds is None else torch.empty(n, dtype=torch.bool, device=dev),
        )

    def today(lens) -> tuple:
        topk, counts, mask = outputs()
        local = ws.local_topk[:n]
        rubin_topk_decode_cute(ws.local_logits[:n, : d.ks_local_len], lens, local, ws.local_k, 1)
        rubin_dsa_key_shard_pack(local, lens, ws)
        rubin_dsa_key_shard_merge(topk, counts, bounds, mask, ws, failing_only=_KS_FAILING_ONLY)
        return _idx_remap(i, topk, d.req_idx_gen, d.block_table_gen, st, rt), counts, mask

    def select(lens) -> tuple:
        topk, counts, mask = outputs()
        rubin_dsa_key_shard_select_pack(sel, lens, n, topk, counts, bounds, mask, kv)
        return topk, counts, mask

    why = key_shard_select_check(
        ws, n, d.ks_local_len, d.ks_row_lens, today, select, (ws.state, sel.merge.state)
    )
    _KS_SELECT_CHECKED[(n, bounds is not None)] = not why
    if why:
        logger.warning(_KS_SELECT_OFF.format(rows=n, why=why))
    return not why


def _mla_absorb(L: _MlaW, q: torch.Tensor) -> tuple:
    """The query buffer ``[rows, heads, latent]`` with W_UK applied to the NoPE columns, and q_pe.

    Only the latent columns are written; the attention op (context) or the RoPE kernel
    (generation) fills the RoPE columns.
    """
    q_nope, q_pe = q.view(-1, _MLA_HEADS, _MLA_QK).split([_MLA_NOPE, _MLA_ROPE], -1)
    fused_q = q.new_empty(q.shape[0], _MLA_HEADS, _MLA_LATENT)
    bmm_out(q_nope.transpose(0, 1), L.k_b, fused_q[..., :_MLA_KV_LORA].transpose(0, 1))
    return fused_q, q_pe


def _mla_expand(v_b_t: torch.Tensor, latent: torch.Tensor, out: torch.Tensor) -> None:
    """``out[t, h] = latent[t, h] @ W_UV[h]``: the latent attention output to the value heads."""
    a, heads = latent.transpose(0, 1), out.view(-1, latent.shape[1], _MLA_V).transpose(0, 1)
    _gemm_pinned("mla_wuv", a, v_b_t, lambda: bmm_out(a, v_b_t, heads), out=heads)


def _mla_window(i: int, st: _Step, rt: _Rt) -> int:
    """The attention window of layer ``i``, read as the engine's attention backend reads it.

    The KV cache manager V2 holds a full-context layer's window as None (only a sliding window is
    a number); the backend then takes the metadata's ``max_seq_len`` (``TrtllmAttention.forward``).
    The DSA layers have no sliding window, so every server gets ``max_seq_len`` here.
    """
    window = st.cache.max_attention_window_vec[rt.dsa_pools.local[i]]
    return int(st.md.max_seq_len) if window is None else int(window)


def _mla_helix_mask(rows: int, st: _Step) -> torch.Tensor:
    """The rows whose partial stays out of the combine: no KV on this rank, or no kept Top-K entry.

    One launch over the bounds (per token on a verify step, else each request's local KV length
    repeated over its rows) and the kept counts; the bounds alone where the step skips the indexer.
    """
    d = st.dsa
    bounds = d.row_bounds
    if bounds is None:
        bounds = torch.repeat_interleave(d.gen_kv_lens, d.next_n, output_size=rows)
    if d.skip_gen or d.counts_out is None:
        return bounds == 0
    return rubin_helix_zero_or_empty_rows(bounds, d.counts_out)


def _mla_query(L, i, q, latent, bufs, st: _Step, rt: _Rt) -> tuple:
    """W_UK into the query buffer, then the RoPE kernel with the helix arguments: (buffer, q_pe).

    The RoPE kernel writes the FP8 query, this rank's latent KV slots and the decode's scheduler
    buffers.
    """
    fused_q, q_pe = _mla_absorb(L, q)
    cu_q, cu_kv, counter, bmm1, bmm2, quant_q = bufs
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
        layer_idx=rt.dsa_pools.local[i],
        **st.dsa.rope_gen,
        **_MLA_ROPE_GEN,
    )
    return fused_q, q_pe


def _mla_generation(L, i, x, q, latent, q8, w, bufs, st: _Step, rt: _Rt) -> tuple:
    """Sparse MLA decode over this rank's KV shard, W_UV on all 96 heads, the combine.

    Returns this rank's 12 heads of the live rows and the output gate's input: ``(g, ready)``
    on graph steps, None on eager steps (``_mla_output`` projects ``x`` there). On graph steps the
    query chain (W_UK, the RoPE kernel; q_b was forked before the indexer) runs on
    ``rt.fork_stream`` while this stream takes the indexer's Top-K and its remap, and
    ``g = g_proj(x)`` follows the RoPE kernel there. The attention joins the event recorded after
    the RoPE kernel, so it never waits for g; ``_mla_output`` joins g's event before the gate.
    ``bufs``: the RoPE kernel's outputs, allocated before the fork (``mla_generate``). g is
    allocated on the fork stream, as q is: this stream reads it only after g's event, and the
    next layer's fork waits for this stream first.
    """
    rows, dev = q.shape[0], q.device
    d = st.dsa
    gate = None
    if st.cuda_graph:
        main = torch.cuda.current_stream()
        # The norms' and the index keys' side-stream joins are behind: the pair is free again.
        q_ready, g_ready = rt.events[_EV_ATTN]
        with torch.cuda.stream(rt.fork_stream):
            fused_q, q_pe = _mla_query(L, i, q, latent, bufs, st, rt)
            q_ready.record()
            g = rubin_linear(x, L.g, False)
            g_ready.record()
        gate = (g, g_ready)
        logger.info_once(_MLA_GATE_FORK_LINE, key=_MLA_GATE_FORK_LINE)
        topk, mask, kv = _idx_decode(i, q8, w, st, rt)
        indices = topk if kv else _idx_remap(i, topk, d.req_idx_gen, d.block_table_gen, st, rt)
        q_ready.wait()
        if not torch.cuda.is_current_stream_capturing():
            q.record_stream(main)
            fused_q.record_stream(main)
            g.record_stream(main)
    else:
        fused_q, q_pe = _mla_query(L, i, q, latent, bufs, st, rt)
        topk, mask, kv = _idx_decode(i, q8, w, st, rt)
        indices = topk if kv else _idx_remap(i, topk, d.req_idx_gen, d.block_table_gen, st, rt)
    cu_q, cu_kv, counter, bmm1, bmm2, quant_q = bufs
    stats = torch.empty(rows, _MLA_HEADS, 2, dtype=torch.float32, device=dev)
    lat = thop_attention(
        q=fused_q.view(rows, -1),
        output=q.new_empty(rows, _MLA_HEADS * _MLA_KV_LORA),
        latent_cache=latent,
        q_pe=q_pe,
        rotary_cos_sin=rt.mla_rope,
        local_layer_idx=rt.dsa_pools.local[i],
        attention_input_type=_MLA_GENERATION,
        attention_window_size=_mla_window(i, st, rt),
        rope_max_positions=rt.dsa_pools.rope_positions,
        rope_original_max_positions=rt.dsa_pools.rope_positions,
        softmax_stats_tensor=stats,
        cu_q_seqlens=cu_q,
        cu_kv_seqlens=cu_kv,
        fmha_scheduler_counter=counter,
        mla_bmm1_scale=bmm1,
        mla_bmm2_scale=bmm2,
        quant_q_buffer=quant_q,
        sparse_attn_indices=indices,
        **{**d.sparse_gen, **_MLA_ATTN},
    )
    if mask is None:
        mask = _mla_helix_mask(rows, st)
    partial_v = lat.new_empty(rows, _MLA_HEADS, _MLA_V)
    _mla_expand(L.v_b_t, lat.view(rows, _MLA_HEADS, _MLA_KV_LORA), partial_v)
    out = rubin_dflash2_helix_combine(
        partial_v, stats, rt.cp_mapping, _MLA_HEADS, _MLA_V, mask, _MLA_COMBINE_COMPILE
    )
    return out, gate


def _mla_output(
    L: _MlaW,
    x: torch.Tensor,
    out: torch.Tensor,
    n: int,
    gate: Optional[tuple] = None,
    rt: Optional[_Rt] = None,
    push: bool = False,
) -> Any:
    """Zero the padding rows, gate with ``sigmoid(g_proj(x))`` and project: the o_proj partial.

    ``gate``: ``(g_proj(x), the event recorded after it)`` from the fork stream (graph steps); the
    gate waits for that event right before it reads g. None: ``g_proj(x)`` runs here. ``push``
    (the mix after this layer finishes its partial on reduce_norm's push path): with
    ``dsa_oproj_push``, o_proj stores the partial into the all-reduce's slots where
    ``_dsa_oproj_push`` serves the rows, and this returns ``_Pushed``.
    """
    if out.shape[0] > n:
        out[n:].zero_()
    if gate is None:
        g = rubin_linear(x, L.g, False)
    else:
        g, ready = gate
        ready.wait()
    gated = k31_sigmoid_gate_mul(out, g)
    if push and _DSA_OPROJ_PUSH:
        pushed = _dsa_oproj_push(L, gated, rt)
        if pushed is not None:
            return pushed
    return _gemm_pinned("mla_o", gated, L.o.t(), lambda: rubin_linear(gated, L.o, False))


# dsa_oproj_push: the o_proj push kernel's tactic by row count, ((rows upper bound, tactic), ...):
# FlashInfer's split-K BF16 GEMM with the push epilogue (``bf16_gemm_push``) on [7168, 1536], no K
# split (cuBLAS's o_proj does not split K at these rows), every grid within one wave of a Rubin
# GPU's 212 SMs (one CTA per SM: the kernel takes the whole shared memory). Up to 32 rows: 64-row
# weight tiles (112 CTAs), 8 operand stages, the token tile the rows' power of two. 33 to 47 rows
# and 65 to 72: 128-row weight tiles (56 per N tile), 5 stages, the rows in N tiles of 32 (two:
# 112 CTAs; three: 168), each N tile's CTAs streaming the same weight tiles. 48 to 64 rows: one
# 64-token tile on 64-row weight tiles (112 CTAs), 6 stages. 64-row weight tiles in N tiles of 32
# would need 224 / 336 CTAs. Rows past the last bound, or outside dsa_oproj_push_bands, keep
# today's GEMM.
_DSA_OPROJ_PUSH = _CONSTANTS["generation"]["dsa_oproj_push"]
_DSA_OPROJ_PUSH_TACTICS = (
    (8, ("splitk", 64, 8, 1, 8)),
    (16, ("splitk", 64, 16, 1, 8)),
    (32, ("splitk", 64, 32, 1, 8)),
    (47, ("splitk", 128, 32, 1, 5)),
    (64, ("splitk", 64, 64, 1, 6)),
    (72, ("splitk", 128, 32, 1, 5)),
)
#: The most rows dsa_oproj_push_bands may name: the push kernel's (and the tactics') 72.
_DSA_OPROJ_PUSH_MAX_M = 72
#: Per process: rows -> whether the push kernel's output at that row count equalled today's o_proj
#: word for word (absent: not checked yet).
_DSA_OPROJ_PUSH_CHECKED: dict = {}
#: Logged once per process and row count by the check: the push serves the rows (INFO), or today's
#: GEMM and the push kernel keep running (WARNING).
_DSA_OPROJ_PUSH_LINE = (
    "Kimi K3.1 Rubin DSA: o_proj's epilogue pushes its partial into the all-reduce at {rows} rows "
    "(today's GEMM's bits)"
)
_DSA_OPROJ_PUSH_OFF = "Kimi K3.1 Rubin DSA: o_proj push at {rows} rows: {why}; today's GEMM runs"


def _dsa_oproj_tactic(rows: int) -> Optional[tuple]:
    """``_DSA_OPROJ_PUSH_TACTICS``' tactic for ``rows``, None past its last bound."""
    for bound, tactic in _DSA_OPROJ_PUSH_TACTICS:
        if rows <= bound:
            return tactic
    return None


def _dsa_oproj_push(L: _MlaW, gated: torch.Tensor, rt: Optional[_Rt]) -> Optional[_Pushed]:
    """``dsa_oproj_push``: push ``gated @ o^T`` into the TP all-reduce's slots; ``_Pushed``, or None
    where today's GEMM runs (no push workspace, rows past it, outside the bands or past the
    tactics, or the rows' device check not passed).

    The first call of a row count in a process, outside capture, compares the push kernel's
    store-only output on its operands with today's o_proj word for word and records the outcome;
    that call runs today's GEMM. A capture before the check runs today's GEMM, unrecorded. The
    pushed words are that output with ``-0`` sent as ``+0``, as the push copy kernel sends today's.
    """
    op = rt.oproj_push if rt is not None else None
    rows = gated.shape[0]
    tactic = _dsa_oproj_tactic(rows)
    if op is None or tactic is None or not 0 < rows <= op.max_rows:
        return None
    if not _moe_in_bands(op.dsa_bands, rows):
        return None
    checked = _DSA_OPROJ_PUSH_CHECKED.get(rows)
    if checked is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        want = _gemm_pinned("mla_o", gated, L.o.t(), lambda: rubin_linear(gated, L.o, False))
        why = ""
        try:
            got = bf16_gemm_push_values(gated, L.o, torch.empty_like(want), tactic)
        except ValueError as exc:
            why = str(exc)
        else:
            if not torch.equal(got.view(torch.int16), want.view(torch.int16)):
                why = "its output is not today's word for word"
        if why:
            logger.warning(_DSA_OPROJ_PUSH_OFF.format(rows=rows, why=why))
        else:
            logger.info(_DSA_OPROJ_PUSH_LINE.format(rows=rows))
        _DSA_OPROJ_PUSH_CHECKED[rows] = not why
        return None
    if not checked:
        return None
    one_shot, _ = op.ws.transport(rows)
    shape = torch.empty(rows, L.o.shape[0], dtype=gated.dtype, device=gated.device)
    bf16_gemm_push(gated, L.o, shape, op.ws.view, one_shot, tactic, op.ws.pdl)
    return _Pushed(rows)


def mla_generate(L: _MlaW, i: int, x: torch.Tensor, st: _Step, rt: _Rt, push: bool = False) -> Any:
    """Layer ``i``'s MLA + DSA over the rows of a decode step; ``[rows, hidden]``, or ``_Pushed``
    where o_proj pushed its partial (``push``: the mix after the layer takes reduce_norm's push
    path; see ``_mla_output``).

    On graph steps the indexer's work on ``x`` alone (``_idx_keys`` and the index-key insert) runs
    on ``rt.idx_stream`` from the top, beside ``[q_a | kv_a]`` and its norms (dsa_idx_fork), and
    q_b runs on ``rt.fork_stream`` once the latent row is built, beside the indexer's query
    projection (fork_q). The current stream joins the idx stream right before the logits weights
    take the query scales: the first read of anything the idx stream wrote.
    """
    n = st.num_tokens
    if st.cuda_graph:
        # Everything the idx stream writes it allocates itself, and the current stream reads it
        # only after the join; the next layer's fork follows that read on the current stream.
        fork, join = rt.events[_EV_IDX]
        fork.record()
        with torch.cuda.stream(rt.idx_stream):
            fork.wait()
            k8, ks, w = _idx_keys(L, x)
            _idx_insert(i, k8[:n], ks[:n], n, st, rt)
            join.record()
    qr, latent = _mla_project(L, x, st, rt)
    # The RoPE kernel's outputs: cu_q / cu_kv, the FMHA scheduler counter, the BMM1 / BMM2 scales
    # and the FP8 query, written on the fork stream on graph steps. Allocated here, so that stream
    # follows every earlier user of the memory: the indexer below frees temporaries whose readers
    # are still queued on this stream.
    seqs = st.num_contexts + st.num_generations + 1
    bufs = (
        torch.empty(seqs, dtype=torch.int32, device=x.device),
        torch.empty(seqs, dtype=torch.int32, device=x.device),
        torch.empty(1, dtype=torch.uint32, device=x.device),
        torch.empty(2, dtype=torch.float32, device=x.device),
        torch.empty(1, dtype=torch.float32, device=x.device),
        torch.empty(n, _MLA_HEADS, _MLA_LATENT, dtype=torch.uint8, device=x.device),
    )
    if st.cuda_graph:
        main = torch.cuda.current_stream()
        rt.fork_stream.wait_stream(main)
        with torch.cuda.stream(rt.fork_stream):
            q = rubin_linear(qr, L.q_b, False)
        q8, qs = _idx_queries(L, qr)
        join.wait()  # right before the first reader of w and of the index keys
        if not torch.cuda.is_current_stream_capturing():
            w.record_stream(main)
        logger.info_once(_IDX_FORK_LINE, key=_IDX_FORK_LINE)
    else:
        q = rubin_linear(qr, L.q_b, False)
        k8, ks, w = _idx_keys(L, x)
        q8, qs = _idx_queries(L, qr)
        _idx_insert(i, k8[:n], ks[:n], n, st, rt)
    w = rubin_idx_weights(w, qs, _IDX_WEIGHT_SCALE)
    logger.info_once(_IDX_SHORT_LINE, key=_IDX_SHORT_LINE)
    out, gate = _mla_generation(L, i, x, q[:n], latent[:n], q8, w, bufs, st, rt)
    if x.shape[0] > n:
        # A CUDA graph's padding rows follow the live ones; _mla_output zeroes them.
        out = torch.cat([out, out.new_empty(x.shape[0] - n, out.shape[1])])
    return _mla_output(L, x, out, n, gate, rt, push)


# --------------------------------------------------------------------------------------------------
# MoE: the latent MoE of layers 1..92 and layer 0's dense MLP (helix8: MoE TP2 x EP4).
#
# A generation server. Every rank holds every row (helix hands the attention output back as whole
# rows), the 224 experts of its EP window and its TP2 half of each expert's intermediate. A
# latent-MoE layer runs:
#
# * the router: FP32 logits of the BF16 gate (``dsv3_router_gemm_op``; on a KDA layer's step of up
#   to 16 rows ``rubin_router_logits``, the same bits);
# * ``latent_down`` (BF16), the routing (``rubin_route_tail`` with the helix8 launch bands up to
#   256 rows, which also sorts the rows for this rank's experts; ``noaux_tc_op`` above) and the
#   NVFP4 input quantization;
# * this rank's experts through the fused FC12 kernel (MMA N 256, the die-local claim on the whole
#   scheduler warp) on the routing tail's tables (above 256 rows on moe_sort's), whose partial the
#   latent all-reduce over the 8 ranks sums with ``routed_expert_norm`` fused in;
# * ``latent_up`` (BF16), and the two shared experts at this rank's TP8 share (one MXFP8 gate_up
#   GEMM, ``situ_and_mul``, the MXFP8 down GEMM) with their all-reduce.
#
# It returns ``_MoeOut(routed, shared)``, both whole; the next mix adds them to the residual.
# latent_down, latent_up and the router GEMM run at their pinned cuBLASLt algorithms in
# gemm_pins_moe's rows (``_gemm_pinned``), the same bits.
#
# The MoE runs on the 8 CP ranks as one TP group (``Mapping.repurpose_helix_cp_to_tp``): the expert
# window (``L.expert_offset``, the rows of ``w2_rows``) comes from weights.py's ``dims`` on that
# mapping, and the all-reduces of ``rt.moe_comm`` are built on it.
#
# Steps run in order on the current stream, except the steps of a CUDA graph (warmup and capture):
# ``latent_down`` and the NVFP4 quantization of its output fork onto ``moe_comm.latent_stream``
# beside the router and the routing tail (the router GEMM issued before them up to 64 rows, and
# with ``router_first_wide`` from 97 rows), FC12's output and counters are zeroed on
# ``moe_comm.prezero_stream``, the shared experts run on ``rt.aux_stream`` beside the routed chain,
# and up to 128 rows the latent all-reduce runs the row kernel on its own workspace before the
# shared stream joins. Without ``moe_latent_up_resident`` these are the in-order step's kernels
# and bits. With it, latent_up behind that row kernel is ``rubin_latent_up_resident`` in its row
# bands, whose FP32 sum is taken in another order than cuBLAS's (one BF16 rounding either way):
# such a graph step's MoE output differs in bits from the eager step's on the same rows.
#
# With ``moe_cp_tail``, a decode step whose rows fall in ``moe_cp_tail_bands`` (up to 72 rows) runs
# the column-parallel tail instead: after the latent all-reduce row, each rank multiplies the whole
# normalized latent by its 896 output columns of latent_up (``rubin_ull_cp_tail``, 1 / 8 of the
# weight) and pushes them into the all-gather rows of the tail's Lamport call; the shared experts
# push their partial into the same call's slots (on ``rt.aux_stream`` in a CUDA graph's steps) in
# place of their all-reduce. It returns ``_MoeTail``; the next mix finishes the call:
# ``bf16(latent_up + bf16(shared))``, the shared sum in the C++ all-reduce's rank order. latent_up's
# FP32 sum is split-K (another order than cuBLAS's, one BF16 rounding either way).
#
# With ``moe_shared_late``, a CUDA graph's decode step whose rows fall in ``moe_shared_late_bands``
# runs on the core's ``_LateStreams`` (``_moe_shared_late_rt``): the column-parallel tail records the
# shared fork after the routing tail and runs the shared experts on ``rt.shared_stream``, the one
# stream of the step at the default priority. The same kernels on the same inputs.
# --------------------------------------------------------------------------------------------------

# K3.1's MoE scalars.
_MOE_NUM_EXPERTS = _K31["num_experts"]
_MOE_TOP_K = _K31["num_experts_per_token"]
_MOE_GROUPS = _K31["num_expert_group"]
_MOE_SCALE = _K31["routed_scaling_factor"]
_MOE_EPS = _K31["rms_norm_eps"]  # routed_expert_norm
_MOE_SITU = (_K31["activation_situ_beta"], _K31["activation_situ_linear_beta"])
# NVFP4 values per input block scale, and the expert kernels' row tile (the routing tail sorts to it).
_MOE_SF_VEC = 16
_MOE_TILE = 128
# The expert kernels' activation (ActivationType.SiTu) and SwiGLU clamp (none: a SiTU MoE).
_MOE_ACT_SITU = 9
_MOE_SWIGLU_LIMIT = float("inf")
# No all-to-all around the expert kernels: every rank holds every row.
_MOE_ALLTOALL = False
_MOE_AR_NONE, _MOE_AR_RMS_NORM = 0, 9  # AllReduceFusionOp.NONE, .RMS_NORM
_MOE_GEN = _CONSTANTS["generation"]


def _moe_require(role: str, key: str, *values) -> None:
    """Fail at import if the launch constant asks for a value this block does not run."""
    if _CONSTANTS[role][key] not in values:
        raise NotImplementedError(
            f"MoE block: {role} {key}={_CONSTANTS[role][key]!r}, runs {values}"
        )


_moe_require("generation", "router_bf16", True)
_moe_require("generation", "moe_backend", "cutedsl_fc12")
_moe_require("generation", "shared_sm", 0)
_moe_require("generation", "route_tail_bands", "helix8")
_moe_require("generation", "latent_bf16_gemm", "cublas_linear")
if _MOE_GEN["latent_ar_own_ws"]:
    _moe_require("generation", "latent_ar_cute", 1)
    _moe_require("generation", "latent_ar_row", True)
# Every row is routed: the padding-row mask of a prefill CUDA graph step is not run.
if any(args["mask_graph_padding_rows"] for args in _MOE_ARGS.values()):
    raise NotImplementedError("mask_graph_padding_rows is set; every row is routed here")

# Decode-step levers as row bounds (0: off). route_tail: top-16 + moe_sort in one cluster launch
# (approx candidates, exact picks), up to its 256 rows; router_exact: KDA layers, up to 16 rows;
# router_first: the router GEMM goes before the forked latent_down; own workspace: the latent
# all-reduce's row kernel with an early PDL trigger. router_first_wide: the router GEMM also goes
# first from 97 rows (helix8 b24 k5: 144) up to the routing tail's bound;
# 65 to 96 rows keep latent_down first (b16 k5: 96 rows measured no faster with the router first).
_MOE_ROUTE_TAIL_MAX_M = 256 if _MOE_GEN["route_tail"] == 1 else 0
_MOE_ROUTE_APPROX = bool(_MOE_GEN["route_approx"])
_MOE_ROUTE_BANDS = _MOE_GEN["route_tail_bands"]
_MOE_ROUTER_EXACT_MAX_M = 16 if _MOE_GEN["router_exact"] == "kda" else 0
_MOE_ROUTER_FIRST_MAX_M = 64 if _MOE_GEN["router_first"] else 0
#: router_first_wide's rows ``(first, last)``; ``(1, 0)``: none.
_MOE_ROUTER_FIRST_WIDE = (97, _MOE_ROUTE_TAIL_MAX_M) if _MOE_GEN["router_first_wide"] else (1, 0)
_MOE_OWN_WS_MAX_M = _MOE_GEN["latent_ar_own_ws_max_m"] if _MOE_GEN["latent_ar_own_ws"] else 0
_MOE_OWN_WS_MAX_BYTES = _MOE_GEN["latent_ar_oneshot_max_bytes"]
_MOE_FC12_PREZERO = bool(_MOE_GEN["fc12_prezero"])
#: Logged once per process by the first step whose latent_up runs the resident GEMM.
_MOE_LATENT_UP_LINE = (
    "Kimi K3.1 Rubin MoE back: latent_up streams its weight while the latent all-reduce waits "
    "(resident, split)"
)


def _moe_row_bands(spec: str, top: int) -> tuple[tuple[int, int], ...]:
    """``((lo, hi), ...)`` of a ``"lo-hi,lo-hi"`` row-band string: ascending, disjoint, within
    ``[1, top]``. ``""`` names no band."""
    bands: list[tuple[int, int]] = []
    for item in filter(None, (part.strip() for part in spec.split(","))):
        lo, sep, hi = item.partition("-")
        if not (sep and lo.isdecimal() and hi.isdecimal()):
            raise ValueError(f"row bands: {item!r} is not <lo>-<hi>")
        band = (int(lo), int(hi))
        if not 1 <= band[0] <= band[1] <= top:
            raise ValueError(f"row bands: {item!r} outside [1, {top}]")
        if bands and band[0] <= bands[-1][1]:
            raise ValueError(f"row bands: {item!r} must follow {bands[-1]} without overlap")
        bands.append(band)
    return tuple(bands)


def _moe_in_bands(bands: tuple[tuple[int, int], ...], rows: int) -> bool:
    """Whether ``rows`` falls in one of ``bands``."""
    return any(lo <= rows <= hi for lo, hi in bands)


def _moe_band_overlaps(owners) -> list[tuple[str, str, tuple[tuple[int, int], ...]]]:
    """``(owner, owner, shared rows)`` for every two owners of ``{owner: bands}`` whose bands
    share a row count."""
    found, names = [], list(owners)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            both = tuple(
                (max(x[0], y[0]), min(x[1], y[1]))
                for x in owners[a]
                for y in owners[b]
                if x[0] <= y[1] and y[0] <= x[1]
            )
            if both:
                found.append((a, b, both))
    return found


#: moe_cp_tail's row bands (none when off) and the most rows they reach (the push GEMM's 72 at
#: most; the tail's workspace and tactics are sized to it).
_MOE_CP_TAIL_BANDS = (
    _moe_row_bands(_MOE_GEN["moe_cp_tail_bands"], 72) if _MOE_GEN["moe_cp_tail"] else ()
)
_MOE_CP_TAIL_MAX_M = max((hi for _, hi in _MOE_CP_TAIL_BANDS), default=0)
#: moe_latent_up_resident's row bands (none when off; the resident GEMM serves up to 128 rows).
_MOE_LATENT_UP_RESIDENT_BANDS = (
    _moe_row_bands(_MOE_GEN["moe_latent_up_resident_bands"], 128)
    if _MOE_GEN["moe_latent_up_resident"]
    else ()
)
#: latent_up's owner -> the row bands it serves on a decode step: the column-parallel tail
#: (``latent_moe`` sends its rows to the tail first), the resident GEMM behind the latent
#: all-reduce row kernel, and gemm_pins_moe's latent_up pins (one band per pinned row count, the
#: A-class ones under gemm_pins_a_class). Every other row count runs cuBLAS's latent_up.
_MOE_LATENT_UP_OWNERS = MappingProxyType(
    {
        "moe_cp_tail": _MOE_CP_TAIL_BANDS,
        "moe_latent_up_resident": _MOE_LATENT_UP_RESIDENT_BANDS,
        "gemm_pins_moe": tuple(
            sorted((rows, rows) for site, rows in _GEMM_PINS if site == "moe_latent_up")
        ),
    }
)
# One owner per row count: two would leave a lever's constant naming rows it never serves.
if _moe_band_overlaps(_MOE_LATENT_UP_OWNERS):
    raise NotImplementedError(
        "MoE block: latent_up rows with two owners: "
        + "; ".join(
            f"{rows} of {a} and {b}" for a, b, rows in _moe_band_overlaps(_MOE_LATENT_UP_OWNERS)
        )
        + f" (owners' bands {dict(_MOE_LATENT_UP_OWNERS)})"
    )
#: Logged once per process by the first decode step that runs the column-parallel tail.
_MOE_CP_TAIL_LINE = (
    "Kimi K3.1 Rubin MoE tail: column-parallel latent_up pushed into the shared experts' Lamport "
    "call (moe_cp_tail)"
)


def _moe_cp_tail_serves(rows: int) -> bool:
    """Whether a decode step of ``rows`` rows runs the column-parallel tail."""
    return _moe_in_bands(_MOE_CP_TAIL_BANDS, rows)


#: moe_shared_late's row bands (none when off).
_MOE_SHARED_LATE_BANDS = (
    _moe_row_bands(_MOE_GEN["moe_shared_late_bands"], 72) if _MOE_GEN["moe_shared_late"] else ()
)
# The late fork acts in the column-parallel tail, where the next mix polls the shared experts'
# partial out of the tail's Lamport call; a band row the tail does not serve would name rows the
# lever never acts on.
if any(not _moe_cp_tail_serves(r) for lo, hi in _MOE_SHARED_LATE_BANDS for r in range(lo, hi + 1)):
    raise NotImplementedError(
        f"MoE block: moe_shared_late_bands {_MOE_SHARED_LATE_BANDS} reach rows outside "
        f"moe_cp_tail_bands {_MOE_CP_TAIL_BANDS}"
    )
#: Logged once per process by the first CUDA graph decode step whose shared experts fork late.
_MOE_SHARED_LATE_LINE = (
    "Kimi K3.1 Rubin MoE shared experts: forked after the routing tail, below the layers' streams "
    "(moe_shared_late)"
)


def _moe_shared_late_serves(rows: int) -> bool:
    """Whether a CUDA graph decode step of ``rows`` rows runs on moe_shared_late's streams."""
    return _moe_in_bands(_MOE_SHARED_LATE_BANDS, rows)


def _moe_shared_late_rt(rt: _Rt, late: _LateStreams, shared: torch.cuda.Stream) -> _Rt:
    """``rt`` for a moe_shared_late band step: every side stream of the layers on ``late``'s
    raised streams, the shared experts on ``shared`` (the core's default-priority aux stream).
    ``rt.taps`` (draft_tap_stream) stays the core's: the drafter joins that tap stream, so its
    capture copies keep the default priority."""
    return rt._replace(
        aux_stream=late.aux,
        fork_stream=late.fork,
        idx_stream=late.idx,
        moe_comm=rt.moe_comm._replace(latent_stream=late.latent, prezero_stream=late.prezero),
        shared_stream=shared,
    )


#: Logged once per process by the first CUDA graph decode step: FC12's NVFP4 input was quantized
#: on the latent stream, beside the routing tail.
_MOE_QUANT_BESIDE_LINE = (
    "Kimi K3.1 Rubin MoE front: FC12 input quantized on the latent stream beside the routing tail"
)
#: Logged once per process by the first CUDA graph decode step whose router GEMM goes first under
#: router_first_wide (above 96 rows).
_MOE_ROUTER_FIRST_WIDE_LINE = (
    "Kimi K3.1 Rubin MoE front: router GEMM issued before latent_down above 96 rows "
    "(router_first_wide)"
)


def _moe_cublas_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return cublas_mm(x, w.t())


#: The BF16 latent_down / latent_up GEMM (``latent_bf16_gemm``).
_moe_latent_gemm = _moe_cublas_linear


def _moe_latent_up(latent: torch.Tensor, L: _MoeW) -> torch.Tensor:
    """latent_up as the PDL dependent of the latent all-reduce row kernel: in
    ``moe_latent_up_resident``'s row bands the resident GEMM, which loads its weight before its
    grid wait; else ``_moe_latent_gemm`` (pinned at gemm_pins_moe's rows)."""
    if _moe_in_bands(_MOE_LATENT_UP_RESIDENT_BANDS, latent.shape[0]):
        up = rubin_latent_up_resident(latent, L.latent_up)
        logger.info_once(_MOE_LATENT_UP_LINE, key=_MOE_LATENT_UP_LINE)
        return up
    return _gemm_pinned(
        "moe_latent_up", latent, L.latent_up.t(), lambda: _moe_latent_gemm(latent, L.latent_up)
    )


def _moe_router(L: _MoeW, x: torch.Tensor) -> torch.Tensor:
    """FP32 logits of the BF16 router GEMM (FP32 accumulation); pinned at gemm_pins_moe's rows."""
    return _gemm_pinned(
        "moe_router",
        x,
        L.router.t(),
        lambda: dsv3_router_gemm_op(x, L.router.t(), None, torch.float32),
        out_dtype=torch.float32,
    )


def _moe_router_gen(L: _MoeW, x: torch.Tensor) -> torch.Tensor:
    """Decode rows' router logits: the exact kernel on a KDA layer up to its row bound."""
    if L.after_kda and x.shape[0] <= _MOE_ROUTER_EXACT_MAX_M:
        return rubin_router_logits(x, L.router)
    return _moe_router(L, x)


def _moe_route(L: _MoeW, logits: torch.Tensor) -> tuple:
    """Return ``(topk_ids, topk_weights, tables)``: the step's top-16 experts and their weights.

    ``tables``: the routing tail's moe_sort tables for this rank's window, None where the expert
    kernel sorts. The same experts and weights either way.
    """
    rows = logits.shape[0]
    if 1 <= rows <= _MOE_ROUTE_TAIL_MAX_M:
        ids, weights, *tables = rubin_route_tail(
            logits,
            L.router_bias,
            _MOE_SCALE,
            L.expert_offset,
            L.fc1_w.shape[0],
            _MOE_TILE,
            route_approx=_MOE_ROUTE_APPROX,
            route_tail_bands=_MOE_ROUTE_BANDS,
        )
        return ids, weights, tuple(tables)
    weights, ids = noaux_tc_op(
        logits, L.router_bias, _MOE_GROUPS, _MOE_GROUPS, _MOE_TOP_K, _MOE_SCALE
    )
    return ids, weights, None


def _moe_quantize(L: _MoeW, latent_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """NVFP4 values and their linear (unswizzled) block scales ``[rows, latent / 16]``."""
    x4, sf = fp4_quantize(latent_in, L.fc31_in_scale, _MOE_SF_VEC, False, False)
    return x4, sf.view(latent_in.shape[0], latent_in.shape[1] // _MOE_SF_VEC)


def _moe_sort(L: _MoeW, ids: torch.Tensor, weights: torch.Tensor) -> tuple:
    """moe_sort's six tables of this rank's window at the expert kernels' row tile."""
    return tuple(
        rubin_moe_sort(
            ids, weights, _MOE_NUM_EXPERTS, _MOE_TOP_K, L.expert_offset, L.fc1_w.shape[0], _MOE_TILE
        )
    )


def _moe_fc12_zeroed(rows: int, num_local: int, hidden: int, device: torch.device) -> tuple:
    """``(buffer, output, counters)`` of an FC12 call on ``rows`` rows, to zero beside the routing.

    One int32 buffer: the BF16 output ``[rows, hidden]``, then, from a 128-byte line, the counters
    of every tile count moe_sort can produce at the 128-row tile (one line per phase, two with the
    die schedule).
    """
    pairs = rows * _MOE_TOP_K
    filled = min(num_local, pairs)
    tiles = filled + (pairs - filled) // _MOE_TILE
    lines = 2 if _FC12_DIE_SCHED else 1
    counter_words = -(-tiles // 32) * 32 + 2 * lines * 32
    out_words = -(-((rows * hidden + 1) // 2) // 32) * 32
    buffer = torch.empty(out_words + counter_words, dtype=torch.int32, device=device)
    output = buffer[:out_words].view(torch.bfloat16)[: rows * hidden].view(rows, hidden)
    return buffer, output, buffer[out_words:]


def _moe_fc12(L: _MoeW, x4, sf, ids, weights, tables, zeroed) -> torch.Tensor:
    """This rank's experts through the fused FC12 kernel: ``[rows, latent]`` BF16 partial.

    ``tables``: the routing tail's moe_sort tables (None: sorted here). ``zeroed``: the output and
    counters zeroed on the prezero stream (None: the launch zeroes the output itself).
    """
    if tables is None:
        tables = _moe_sort(L, ids, weights)
    t2e, t2l, e2p, p2e, _, n = tables
    if zeroed is None:
        shape = (ids.shape[0], L.fc2_w.shape[1])
        output, counters = torch.empty(shape, dtype=torch.bfloat16, device=ids.device), None
    else:
        _, output, counters = zeroed
    # The launcher reads the NVFP4 codes as float4_e2m1fn_x2 (the engine op's caller views them
    # the same way); fp4_quantize returns them and ``w2_rows`` stores them as uint8.
    rubin_fc12(
        x4.view(torch.float4_e2m1fn_x2),
        L.fc1_w.view(torch.float4_e2m1fn_x2),
        sf,
        L.fc1_s,
        L.fc31_alpha,
        t2e,
        t2l,
        p2e,
        n,
        L.fc2_in_scale,
        L.fc2_w.view(torch.float4_e2m1fn_x2),
        L.fc2_s,
        L.fc2_alpha,
        output,
        weights,
        e2p,
        _MOE_NUM_EXPERTS,
        _MOE_TOP_K,
        L.fc1_w.shape[0],
        L.expert_offset,
        _MOE_TILE,
        _MOE_SWIGLU_LIMIT,
        _MOE_EP,
        _MOE_ALLTOALL,
        _MOE_SF_VEC,
        _MOE_ACT_SITU,
        *_MOE_SITU,
        counters,
        mma_n=_FC12_MMA_N,
        die_sched=_FC12_DIE_SCHED,
        die_min_tokens=_FC12_DIE_MIN_TOKENS,
        lookahead=_FC12_LOOKAHEAD,
        sm_reserve=_FC12_SM_RESERVE,
        die_claim=_FC12_DIE_CLAIM,
        l2_prefetch=_FC12_L2_PREFETCH,
    )
    return output


def _moe_shared(L: _MoeW, x: torch.Tensor) -> torch.Tensor:
    """The shared experts at this rank's share: gate and up in one GEMM, then SiTU."""
    gu = rubin_mxfp8_linear(x, L.shared_gu_w, L.shared_gu_s)
    # SiTU writes the down projection's MXFP8 operand itself: no BF16 round trip, no quantize kernel.
    # The prequantized GEMM never stores the padding rows: their scale bytes need no memset.
    codes, scales = rubin_situ_and_mul_mxfp8(gu, *_MOE_SITU, zero_padding=False)
    return rubin_mxfp8_linear_prequantized(codes, scales, L.shared_dn_w, L.shared_dn_s)


def _moe_one_shot(routed: torch.Tensor, ws: dict) -> bool:
    """Whether every rank's rows fit one Lamport buffer of the own workspace, one-shot."""
    nbytes = routed.numel() * ws["world_size"] * routed.element_size()
    return routed.shape[0] <= ws["max_tokens"] and nbytes <= min(
        _MOE_OWN_WS_MAX_BYTES, ws["buffer_bytes"]
    )


def _moe_tp_gen(L: _MoeW, x: torch.Tensor, st: _Step, rt: _Rt) -> _MoeOut:
    """A decode step on the current stream: FC12 experts, then the shared experts."""
    c = rt.moe_comm
    logits = _moe_router_gen(L, x)
    latent_in = _gemm_pinned(
        "moe_latent_down", x, L.latent_down.t(), lambda: _moe_latent_gemm(x, L.latent_down)
    )
    ids, weights, tables = _moe_route(L, logits)
    x4, sf = _moe_quantize(L, latent_in)
    routed = _moe_fc12(L, x4, sf, ids, weights, tables, None)
    shared = allreduce_module(_moe_shared(L, x), c.all_reduce, _MOE_AR_NONE, None, None)
    latent = allreduce_module(routed, c.all_reduce, _MOE_AR_RMS_NORM, L.latent_norm, _MOE_EPS)
    up = _gemm_pinned(
        "moe_latent_up", latent, L.latent_up.t(), lambda: _moe_latent_gemm(latent, L.latent_up)
    )
    return _MoeOut(up, shared)


def _moe_tp_gen_streams(L: _MoeW, x: torch.Tensor, st: _Step, rt: _Rt) -> _MoeOut:
    """A decode step of a CUDA graph with the stream plan: the in-order step's bits, except
    latent_up in ``moe_latent_up_resident``'s row bands (another FP32 summation order)."""
    c = rt.moe_comm
    rows = x.shape[0]
    zeroed = None
    if _MOE_FC12_PREZERO:
        # Allocated here, so the zeroing stream follows every earlier user of the memory.
        zeroed = _moe_fc12_zeroed(rows, L.fc1_w.shape[0], L.fc2_w.shape[1], x.device)
    c.latent_fork.record()
    if zeroed is not None:
        with torch.cuda.stream(c.prezero_stream):
            c.latent_fork.wait()
            zeroed[0].zero_()
            c.prezero_ready.record()
    # The router GEMM before the latent fork's kernels up to router_first's bound and in
    # router_first_wide's rows: the same GEMM on the same operands, issued first.
    wide = _MOE_ROUTER_FIRST_WIDE[0] <= rows <= _MOE_ROUTER_FIRST_WIDE[1]
    logits = _moe_router_gen(L, x) if rows <= _MOE_ROUTER_FIRST_MAX_M or wide else None
    # FC12's NVFP4 input is quantized right after latent_down on the latent stream, beside the
    # routing tail, which it does not read. No record_stream: these outputs are read on this
    # stream only after its join, and later work on the side streams forks from this stream.
    with torch.cuda.stream(c.latent_stream):
        c.latent_fork.wait()
        latent_in = _gemm_pinned(
            "moe_latent_down", x, L.latent_down.t(), lambda: _moe_latent_gemm(x, L.latent_down)
        )
        x4, sf = _moe_quantize(L, latent_in)
        c.latent_ready.record()
    if logits is None:
        logits = _moe_router_gen(L, x)
    c.shared_fork.record()
    ids, weights, tables = _moe_route(L, logits)
    c.latent_ready.wait()  # right before the first reader of x4 and sf
    if zeroed is not None:
        c.prezero_ready.wait()
    routed = _moe_fc12(L, x4, sf, ids, weights, tables, zeroed)
    logger.info_once(_MOE_QUANT_BESIDE_LINE, key=_MOE_QUANT_BESIDE_LINE)
    if wide:
        logger.info_once(_MOE_ROUTER_FIRST_WIDE_LINE, key=_MOE_ROUTER_FIRST_WIDE_LINE)
    with torch.cuda.stream(rt.aux_stream):
        c.shared_fork.wait()
        shared = allreduce_module(_moe_shared(L, x), c.all_reduce, _MOE_AR_NONE, None, None)
        c.shared_join.record()
    ws = c.latent_ws
    own = rows <= _MOE_OWN_WS_MAX_M and _moe_one_shot(routed, ws)
    if own:
        # On its own workspace: no need to wait for the shared all-reduce.
        latent = torch.empty_like(routed)
        rubin_latent_ar_norm_row(
            routed,
            L.latent_norm,
            _MOE_EPS,
            latent,
            uc_addr=ws["buffer"].data_ptr(),
            mc_addr=ws["mc"].data_ptr(),
            flags=ws["flags"],
            rank=ws["rank"],
            world_size=ws["world_size"],
            use_pdl=True,
            early_trigger=True,
            norm=True,
            mxfp8_out=None,
            sf_blocks=0,
            prepushed=False,
            early_state=False,
        )
        up = _moe_latent_up(latent, L)
        c.shared_join.wait()
        return _MoeOut(up, shared)
    # Both all-reduces use the TP group's workspace: the shared one goes first.
    c.shared_join.wait()
    latent = allreduce_module(routed, c.all_reduce, _MOE_AR_RMS_NORM, L.latent_norm, _MOE_EPS)
    up = _gemm_pinned(
        "moe_latent_up", latent, L.latent_up.t(), lambda: _moe_latent_gemm(latent, L.latent_up)
    )
    return _MoeOut(up, shared)


def _moe_cp_tail(L: _MoeW, routed: torch.Tensor, rt: _Rt) -> _MoeTail:
    """The column-parallel tail after the experts; returns the tail's call.

    The latent all-reduce + norm (the row kernel on its own workspace where the rows fit it, as
    the replicated tail), then this rank's rows of latent_up over the whole latent, pushed into
    the all-gather rows of the tail's call.
    """
    c, u = rt.moe_comm, rt.ull_tail
    rows = routed.shape[0]
    ws = c.latent_ws
    own = rows <= _MOE_OWN_WS_MAX_M and _moe_one_shot(routed, ws)
    if own:
        latent = torch.empty_like(routed)
        rubin_latent_ar_norm_row(
            routed,
            L.latent_norm,
            _MOE_EPS,
            latent,
            uc_addr=ws["buffer"].data_ptr(),
            mc_addr=ws["mc"].data_ptr(),
            flags=ws["flags"],
            rank=ws["rank"],
            world_size=ws["world_size"],
            use_pdl=True,
            early_trigger=True,
            norm=True,
            mxfp8_out=None,
            sf_blocks=0,
            prepushed=False,
            early_state=False,
        )
    else:
        latent = allreduce_module(routed, c.all_reduce, _MOE_AR_RMS_NORM, L.latent_norm, _MOE_EPS)
    rubin_ull_cp_tail(latent, L.latent_up[u.c0 : u.c0 + u.n], u.ws, u.tactics[rows], c0=u.c0)
    logger.info_once(_MOE_CP_TAIL_LINE, key=_MOE_CP_TAIL_LINE)
    return _MoeTail(u.ws, u.cpp_one_shot, u.chunk)


def _moe_tp_gen_tail(L: _MoeW, x: torch.Tensor, st: _Step, rt: _Rt) -> _MoeTail:
    """A decode step on the current stream with the column-parallel tail.

    FC12 experts, the shared experts' partial pushed into the tail's call, then the tail.
    """
    logits = _moe_router_gen(L, x)
    latent_in = _gemm_pinned(
        "moe_latent_down", x, L.latent_down.t(), lambda: _moe_latent_gemm(x, L.latent_down)
    )
    ids, weights, tables = _moe_route(L, logits)
    x4, sf = _moe_quantize(L, latent_in)
    routed = _moe_fc12(L, x4, sf, ids, weights, tables, None)
    rubin_lamport_push(rt.ull_tail.ws, _moe_shared(L, x), True, True)
    return _moe_cp_tail(L, routed, rt)


def _moe_tp_gen_streams_tail(L: _MoeW, x: torch.Tensor, st: _Step, rt: _Rt) -> _MoeTail:
    """A decode step of a CUDA graph with the column-parallel tail.

    The stream plan of ``_moe_tp_gen_streams``; the shared experts push their partial into the
    tail's call on the aux stream. On a moe_shared_late band step (``rt.shared_stream`` set) they
    fork after the routing tail and run on ``rt.shared_stream``.
    """
    c = rt.moe_comm
    rows = x.shape[0]
    late = rt.shared_stream is not None
    zeroed = None
    if _MOE_FC12_PREZERO:
        # Allocated here, so the zeroing stream follows every earlier user of the memory.
        zeroed = _moe_fc12_zeroed(rows, L.fc1_w.shape[0], L.fc2_w.shape[1], x.device)
    c.latent_fork.record()
    if zeroed is not None:
        with torch.cuda.stream(c.prezero_stream):
            c.latent_fork.wait()
            zeroed[0].zero_()
            c.prezero_ready.record()
    wide = _MOE_ROUTER_FIRST_WIDE[0] <= rows <= _MOE_ROUTER_FIRST_WIDE[1]
    logits = _moe_router_gen(L, x) if rows <= _MOE_ROUTER_FIRST_MAX_M or wide else None
    # No record_stream: these outputs are read on this stream only after its join, and later
    # work on the side streams forks from this stream.
    with torch.cuda.stream(c.latent_stream):
        c.latent_fork.wait()
        latent_in = _gemm_pinned(
            "moe_latent_down", x, L.latent_down.t(), lambda: _moe_latent_gemm(x, L.latent_down)
        )
        x4, sf = _moe_quantize(L, latent_in)
        c.latent_ready.record()
    if logits is None:
        logits = _moe_router_gen(L, x)
    if not late:
        c.shared_fork.record()
    ids, weights, tables = _moe_route(L, logits)
    if late:
        c.shared_fork.record()
    c.latent_ready.wait()  # right before the first reader of x4 and sf
    if zeroed is not None:
        c.prezero_ready.wait()
    routed = _moe_fc12(L, x4, sf, ids, weights, tables, zeroed)
    logger.info_once(_MOE_QUANT_BESIDE_LINE, key=_MOE_QUANT_BESIDE_LINE)
    if wide:
        logger.info_once(_MOE_ROUTER_FIRST_WIDE_LINE, key=_MOE_ROUTER_FIRST_WIDE_LINE)
    with torch.cuda.stream(rt.shared_stream if late else rt.aux_stream):
        c.shared_fork.wait()
        rubin_lamport_push(rt.ull_tail.ws, _moe_shared(L, x), True, True)
        c.shared_join.record()
    if late:
        logger.info_once(_MOE_SHARED_LATE_LINE, key=_MOE_SHARED_LATE_LINE)
    tail = _moe_cp_tail(L, routed, rt)
    c.shared_join.wait()
    return tail


def latent_moe(L: _MoeW, i: int, x: torch.Tensor, st: _Step, rt: _Rt) -> _MoeOut | _MoeTail:
    """Run layer ``i``'s latent MoE (routed and shared experts) over ``x``.

    Returns both parts, or the tail's call (``_MoeTail``) on a decode step whose rows fall in
    ``moe_cp_tail_bands``.
    """
    if _moe_cp_tail_serves(x.shape[0]):
        if st.cuda_graph:
            return _moe_tp_gen_streams_tail(L, x, st, rt)
        return _moe_tp_gen_tail(L, x, st, rt)
    if st.cuda_graph:  # the graph runner's warmup and capture
        return _moe_tp_gen_streams(L, x, st, rt)
    return _moe_tp_gen(L, x, st, rt)


def dense_mlp(L0: _DenseW, x: torch.Tensor, st: _Step, rt: _Rt) -> torch.Tensor | _DensePartial:
    """Run layer 0's dense SiTU MLP over ``x``; return ``[rows, hidden]`` (with ``dense_ar_push``
    this rank's partial as ``_DensePartial``). Its two GEMMs run at their pinned cuBLASLt
    algorithms in gemm_pins_dense's rows (``_gemm_pinned``), the same bits."""
    gu = _gemm_pinned("dense_gu", x, L0.gu.t(), lambda: cublas_mm(x, L0.gu.t()))
    h = situ_and_mul(gu, *_MOE_SITU)
    out = _gemm_pinned("dense_dn", h, L0.dn.t(), lambda: cublas_mm(h, L0.dn.t()))
    ar = rt.moe_comm.dense_all_reduce
    if ar is None:
        return out
    if rt.dense_push:
        # dense_ar_push: the next mix sums it (rt.reduce_norm: its push path where it serves the
        # rows, else this all-reduce and the add-and-mix slot).
        logger.info_once(_DENSE_PUSH_LINE, key=_DENSE_PUSH_LINE)
        return _DensePartial(out)
    return allreduce_module(out, ar, _MOE_AR_NONE, None, None)


# --------------------------------------------------------------------------------------------------
# DFlash2: the drafter container, its block step and the speculative worker.
#
# The drafter's weights are ``core.w["d_*"]`` (``w3_rows``). ``DraftModel`` holds no parameter of
# its own: it reads them through the core and answers what the engine's worker, the KV cache
# creator and the weight loader read off a drafter. ``draft_step`` is the block decode of the five
# drafter layers, straight-line, every kernel a family or catalog entry.
#
# helix8: the history (one FP8 latent of 576 values per token and drafter layer) is page-sharded over
# the 8 helix ranks of the unified draft cache, as the target's KV is: a rank owns the positions of
# every 8th page. Each rank computes all 96 query heads over the keys it owns (the window entry
# writes each row's partial and its log-sum-exp), then one exchange over the target's helix mapping
# merges the ranks and leaves each rank its 12 heads. ``o_proj`` and the MLP are row-parallel over the
# 8 ranks repurposed as TP, one all-reduce each; ``fc`` is whole. The block's latents go into a
# persistent scratch the window entry reads, never into the pool. Its q_b, o_proj, gate_up, down and
# conv projections run at their pinned cuBLASLt algorithms in gemm_pins_draft's rows
# (``_gemm_pinned``), the same bits.
# --------------------------------------------------------------------------------------------------

# The drafter history is _DRAFT_HISTORY_DTYPE (FP8 E4M3 at a unit scale), a target constant.
# speculative_config.context_kv_dtype may leave it to the target ("auto"; None where the config has
# no such field) or name it ("fp8"); routing and the worker refuse any other value.
_DRAFT_HISTORY_CONFIGS = (None, "auto", "fp8")
# Block verification (not token-wise rejection sampling) for a sampled batch.
_DRAFT_BLOCK_VERIFY = True
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
# Body chunks of the window entry: one (measured fastest at helix8's step).
_DRAFT_WINDOW_SPLITS = 1
# The window entry's 16-byte vector loads and stores (logged constant draft_window_vec_io; on).
# Only the width of the memory instructions changes, not a byte of the output.
_DRAFT_WINDOW_VEC_IO = _CONSTANTS["generation"]["draft_window_vec_io"]
# The window entry's tail as a PDL dependent of its body (logged constant draft_window_pdl; on):
# only the launches and the tail's grid wait move.
_DRAFT_WINDOW_PDL = _CONSTANTS["generation"]["draft_window_pdl"]
#: Logged once per process by the first drafter step under draft_window_vec_io: on how many of the
#: drafter's layers both window kernels ran with ALIGNED (16-byte vectors); a layer whose strides
#: refuse them runs the scalar path, and the kernel's launcher warns once.
_DRAFT_WINDOW_VEC_LINE = (
    "Kimi K3.1 Rubin DFlash2 window: ALIGNED=True on {} of {} drafter layers "
    "(16-byte vector loads and stores)"
)
# The drafter's BF16 projections on the family's low-M kernels (logged constant draft_lowm_gemm;
# on): a projection whose (shape, rows) K31_LOW_M_DRAFT_PLAN_HELIX8 names runs that one CuTe DSL
# launch, its K chunks summed inside the kernel (no cuBLAS reduce launch), with programmatic
# dependent launch (its weight loads start under the previous kernel); any other runs cuBLAS
# (F.linear) as without the lever. That plan's rows give F.linear's bits (cuBLAS's K chunks, added
# in cuBLAS's order), so the drafter's outputs are unchanged. draft_lowm_gemm_d_class (on) runs
# K31_LOW_M_DRAFT_PLAN_HELIX8_D instead: it adds the rows whose kernel sums the same BF16 products
# in another FP32 order (no dtype change). That moves the drafter's rounding only: the target's
# verification keeps its tokens exact, and the acceptance length is re-checked for the class.
_DRAFT_LOWM_GEMM = _CONSTANTS["generation"]["draft_lowm_gemm"]
_DRAFT_LOWM_PLAN = (
    None
    if not _DRAFT_LOWM_GEMM
    else K31_LOW_M_DRAFT_PLAN_HELIX8_D
    if _CONSTANTS["generation"]["draft_lowm_gemm_d_class"]
    else K31_LOW_M_DRAFT_PLAN_HELIX8
)
# The drafter block (max_draft_len + 1) the plans' rows, draft_q_a_kv_a_bands, the CuTe DSL
# window's draft_window_mla_bands and the draft_norm_ctas cluster were measured and certified at
# (k5: rows are multiples of 6; the acceptance length of the settings that move the drafter's
# rounding was checked at this block only). A drafter of another block runs every projection on
# cuBLAS, q_a / kv_a in two GEMMs, the Triton window and the norm a row a CTA, as without the
# levers.
_DRAFT_LOWM_BLOCK = 6
# The drafter's norms and block convolutions with programmatic dependent launch (logged constant
# draft_conv_norm; on): RMSNorm as dflash2_rmsnorm (FlashInfer's kernel, its dependents released
# right after its grid wait), the convolutions' input side launched with PDL, and each output side
# fused into the residual add + RMSNorm after it (dflash2_conv_add_rmsnorm: one launch for the two).
# Every reader of their outputs waits for its producer before reading. The same bits.
_DRAFT_CONV_NORM = _CONSTANTS["generation"]["draft_conv_norm"]
# The rows per cluster of draft_conv_norm's conv + add + RMSNorm (logged constant draft_norm_ctas,
# a detail of draft_conv_norm; 0: one CTA a row): with 7, each 7168-wide row spreads over a 7-CTA
# cluster, its sum of squares reduced in DSMEM. The residual is the same bits; the normalized
# output sums the squares in another order (no dtype change), so the drafter's rounding moves.
# Only a drafter of _DRAFT_LOWM_BLOCK's block takes the cluster.
_DRAFT_NORM_CTAS = _CONSTANTS["generation"]["draft_norm_ctas"]
# The drafter's block convolutions launched with programmatic dependent launch where they run as
# their own kernels (logged constant draft_conv_pdl; off): each waits for its producer before its
# first load and releases the next kernel right after the wait; its readers wait before reading.
# The same kernels, so the same bits.
_DRAFT_CONV_PDL = _CONSTANTS["generation"]["draft_conv_pdl"]
# Each drafter layer's q_a and kv_a as one GEMM over their stacked rows (logged constant
# draft_q_a_kv_a_fused; on): the weight table holds d_l{j}_q_a_kv_a ([q_a ; kv_a], kv_a's rows a
# copy of d_kv_a's, which the history's projection reads), q_a views its rows, the q_a norm reads
# the product's first q_lora columns and the block RoPE the rest. The same BF16 products in one
# GEMM of N = q_lora + kv_lora + rope: its K split may differ from the two GEMMs', so the bits
# may move by the FP32 summation order (no dtype change). The one GEMM runs on the steps whose rows
# fall in draft_q_a_kv_a_bands (a detail of the constant, ``lo-hi`` row bands); any other step
# (and a drafter of another block than _DRAFT_LOWM_BLOCK's, where the bands were not measured)
# projects q_a (a view of the stack) and kv_a in two GEMMs, as without the lever. On VR200 cuBLAS's
# [q_a ; kv_a] GEMM gives the two GEMMs' words at 18-78 rows (k5, b3-b13) and other words at 6 /
# 12 rows (another K split).
_DRAFT_Q_A_KV_A = _CONSTANTS["generation"]["draft_q_a_kv_a_fused"]
_DRAFT_Q_A_KV_A_BANDS = _moe_row_bands(_CONSTANTS["generation"]["draft_q_a_kv_a_bands"], 1024)
# The drafter's window on the CuTe DSL MLA decode (logged constant draft_window_mla; on):
# rubin_ull_dflash2_helix_window packs each block row's visible keys on this rank (its history
# pages of the window, then the block's own keys up to the row) into BF16 pages and runs the
# engine's MLA decode over them, each row its own sequence, in place of the window plan and the
# Triton body + tail. It returns the rank's normalized partial, its log-sum-exp as (lse, 1) stats
# (the helix combine weighs them as it weighs (max, sum)) and the rows with no local key, which the
# combine sanitizes. The FP8 keys widen to BF16 exactly; the decode feeds P to P.V in BF16 and sums
# in its own order, so the drafter's rounding moves (no stored dtype changes): the acceptance length
# is re-checked for it. Its buffers are per row count, made at the first eager step of that count
# (the warm-up before capture); a capture without them runs the Triton window.
_DRAFT_WINDOW_MLA = _CONSTANTS["generation"]["draft_window_mla"]
# The block-row counts the CuTe DSL window takes under draft_window_mla (logged detail
# draft_window_mla_bands, "lo-hi" row bands); every other row count runs the Triton window. Both
# routes' costs step with the rows rather than the draft length: the Triton tail launches
# 2 * ceil(block * heads / 64) CTAs a request, about 3 a row, one per SM at its register use, so
# from 70 rows on (k4..k8; VR200's 212 SMs) it runs a second wave; the MLA decode runs one
# sequence a row and steps up past 30 rows. VR200, one rank's window at helix8's shapes and an FP8
# history, k1 and k3..k8: the CuTe DSL route is the faster one in these bands, the slower one
# between them; past 120 rows it is not measured. A drafter of another block than
# _DRAFT_LOWM_BLOCK's runs the Triton window at every row count.
_DRAFT_WINDOW_MLA_BANDS = _moe_row_bands(_CONSTANTS["generation"]["draft_window_mla_bands"], 1024)
#: Logged once per process by the first drafter step that runs the CuTe DSL window.
_DRAFT_WINDOW_MLA_LINE = (
    "Kimi K3.1 Rubin DFlash2 window: the CuTe DSL MLA decode over the BF16-packed window "
    "(draft_window_mla)"
)
#: Logged once per process and row count by the first drafter step under draft_lowm_gemm: which of
#: the drafter's projections ran a low-M kernel at those rows, and the plan's class.
_DRAFT_LOWM_LINE = (
    "Kimi K3.1 Rubin DFlash2: drafter projections on the low-M plan ({}) at {} rows: {}"
)
# The selector's candidates from this rank's vocab shard: the drafter head's logits stay this
# rank's [rows, 20480] shard and _dflash2_global_top_k's sharded branch takes the global top-32
# (the shard's top-32 in id order, the pairs all-gathered, merged) instead of all-gathering, casting
# and taking the top-32 of the full [rows, 163840] logits. Same ids, values and order.
_DRAFT_SHARD_TOP32 = True
#: Logged once per process by the first drafter step that takes its candidates from the shard.
_DRAFT_SHARD_LINE = (
    "Kimi K3.1 Rubin drafter: selector candidates from the vocab shard (shard top-32 in id order, "
    "pairs all-gathered and merged)"
)
# The drafter tail's switches (logged constants; on: draft_select_accept, draft_greedy_tokens,
# draft_index_pairs, draft_tail_rewind, draft_noise_block; off: draft_block_rows,
# draft_history_aux). Each runs the same values through fewer launches; the drafter's and the
# target's bits are unchanged.
#
# draft_select_accept: on an all-greedy step whose target tokens come from this rank's vocab shard,
# one select_accept launch reduces the exchanged argmax pairs (vocab_parallel_argmax's last kernel),
# accepts strictly, forces the count (the engine's forced acceptance: the same pool, counter, warmup
# skip and real rows) and adds it to the forced totals, writes the KDA replay record
# (KDAReplayState.record_acceptance, where update_mamba_states writes nothing else) and the drafter's
# kv_lens prep (_prepare_kv_for_draft_forward, the rewind returned). At most
# _SELECT_ACCEPT_MAX_REQUESTS requests of each kind; any other step runs the unfused acceptance.
_DRAFT_SELECT_ACCEPT = _CONSTANTS["generation"]["draft_select_accept"]
# draft_greedy_tokens: on an all-greedy drafting step, the draft tokens come from the selector's
# walk directly (k31_dflash2_selector_greedy_tokens): the argmax of the FP32 block the greedy path
# writes, without the [n, K, 163840] -inf block and its argmax. The shard pairs keep their id order.
_DRAFT_GREEDY_TOKENS = _CONSTANTS["generation"]["draft_greedy_tokens"]
# draft_index_pairs: the shard's top-32 pairs are written in id order by the selection kernel
# (k31_dflash2_shard_index_pairs), in place of the torch-order pairs and _pairs_in_id_order's sort.
_DRAFT_INDEX_PAIRS = _CONSTANTS["generation"]["draft_index_pairs"]
# draft_block_rows: the block-output rows the draft logits read (slots 1 .. K of each request's
# block) as a strided view of the block decode's output and one copy, in place of the slot ids
# built every step and the index gather; the engine's slot convention and whole blocks only.
_DRAFT_BLOCK_ROWS = _CONSTANTS["generation"]["draft_block_rows"]
# draft_tail_rewind: the drafter's kv_lens rewind after the draft forward rides in the
# next-new-tokens launch (draft_next_new_tokens_rewind), in place of the rewind's two in-place ops
# and the gather + concat; never on a graph warmup (it rewinds nothing).
_DRAFT_TAIL_REWIND = _CONSTANTS["generation"]["draft_tail_rewind"]
# draft_history_aux: on a graph step whose drafter fc projection is issued early on the aux stream,
# the history chain of the same rows (the stacked kv_a projection and the context norm / RoPE of every
# layer) runs there too, right after the fc: its positions (each request's history length before
# this step, plus 0 .. K) are read on the aux stream before the projection, and the current stream
# waits for that read before the preparation advances the lengths. The preparation then stores the
# latents the aux stream made; only the store stays behind the acceptance. Same kernels on the same
# rows and positions.
_DRAFT_HISTORY_AUX = _CONSTANTS["generation"]["draft_history_aux"]
# draft_noise_block: the generation preparation's drafter input block (each request's bonus-token
# embedding row, then the mask row in every other slot) as one copy kernel over the target's
# embedding table (k31_dflash2_noise_block) in place of dflash_noise_block_embedding's five launches
# (id fill, cat, F.embedding, expand clone, bonus-row copy). Only without a trained mask row.
_DRAFT_NOISE_BLOCK = _CONSTANTS["generation"]["draft_noise_block"]
# select_accept holds every verify row in one program's registers: at most this many requests of
# each kind (contexts, generations) take it.
_SELECT_ACCEPT_MAX_REQUESTS = 16
# The selector's candidates per slot that the greedy-tokens walk and the index pairs take.
_SELECTOR_TOP_K = 32
#: Logged once per path by the first acceptance outside capture under draft_select_accept.
_SELECT_ACCEPT_LINE = "Kimi K3.1 Rubin DFlash2 select_accept: fused={} {}"
#: Logged once per process by the first CUDA graph generation step that drafts: the drafter's
#: projection of the step's captured target rows (the fc GEMM and the hidden norm) was issued on
#: the aux stream before the acceptance, which it does not read.
_DRAFT_FC_EARLY_LINE = (
    "Kimi K3.1 Rubin DFlash2: drafter fc projection issued on the aux stream before the acceptance"
)
#: Logged once per process by the first preparation that stores the history the aux stream made.
_DRAFT_HISTORY_AUX_LINE = (
    "Kimi K3.1 Rubin DFlash2: drafter history (kv_a, context norm / RoPE) on the aux stream after "
    "the early fc (draft_history_aux)"
)
# The drafter's two copies across the TP8 group as zero-padded all-reduces (ZeroPaddedReduce) on
# the target head's FP32 pair exchange, which runs the MNNVL all-reduce on the main stream; both
# on, each falls back to today's NCCL all-gather where it is unbound (no MNNVL FP32
# exchange, a workspace too small for its rows) or a call's shape is not its own:
# draft_pairs_zero_padded (logged constant): the top-32 (global id, value) pairs of
# _dflash2_global_top_k's sharded branch, each rank's 64 columns at its rank's offset, still in id
# order (_pairs_in_id_order); draft_agree_zero_padded (logged constant): the engine's draft-token
# agreement, rank 0's [rows, K] tokens written by rank 0 alone. A sum of one value and zeros is that
# value, and ids below 2**24 are exact in FP32, so both carry the all-gather's values; a -0.0 pair
# value arrives as +0.0 (the Lamport sentinel). Both run after the drafter's last all-reduce on the
# same stream; nothing on the aux stream calls this workspace.
_DRAFT_PAIRS_ZERO_PADDED = _CONSTANTS["generation"]["draft_pairs_zero_padded"]
_DRAFT_AGREE_ZERO_PADDED = _CONSTANTS["generation"]["draft_agree_zero_padded"]
# The agreement's mismatch census (logged constant draft_agree_check; off): the agreement still
# runs, and each rank adds on the device, per call, how many rows' local drafts differed from rank
# 0's (CUDA-graph padding rows counted apart), read back without a host synchronization every
# _DRAFT_AGREE_PERIOD steps (_report_draft_agreement). The evidence an agreement-free step needs.
_DRAFT_AGREE_CHECK = _CONSTANTS["generation"]["draft_agree_check"]
_DRAFT_AGREE_PERIOD = 200
#: Logged once per process when a zero-padded copy is bound ({} = pairs / agreement).
_DRAFT_ZERO_PADDED_LINE = (
    "Kimi K3.1 Rubin DFlash2: drafter {} as a zero-padded MNNVL all-reduce (FP32, {} rows x {} "
    "columns)"
)
#: The census line of draft_agree_check, every _DRAFT_AGREE_PERIOD steps on every rank.
_DRAFT_AGREE_LINE = (
    "K31_H8_AGREE rank={} mismatched_real={} real_rows={} mismatched_padding={} padding_rows={} "
    "steps={}"
)

#: The draft-token agreement's all-reduce row, in FP32 columns (the drafts take the first K). The
#: MNNVL kernels clear the Lamport buffer of an earlier call with the threads of the current one: a
#: row of K drafts would be a block of two threads clearing the drafter's last all-reduce buffer;
#: a 2048-column row is a block of 512.
_DRAFT_AGREEMENT_WIDTH = 2048


class ZeroPaddedReduce:
    """A copy across a group as one all-reduce of a zero-padded buffer.

    A writing rank puts ``local [rows, columns]`` at its ``offset`` in a ``[rows, width]``
    buffer, a rank without an offset writes nothing, and all ranks all-reduce the buffer: every
    column is zero on every rank but the one that wrote it, and a sum of one value and zeros is
    that value. Every rank at its own offset makes an all-gather; one writing rank, a broadcast.
    Bitwise the copy, except that a -0.0 arrives as +0.0 (the MNNVL all-reduce's Lamport
    sentinel). The ``[max_rows, width]`` buffer persists and only this rank's columns are ever
    written, so the others stay zero without a fill: one copy kernel and one MNNVL all-reduce in
    place of NCCL's ring all-gather. ``all_reduce`` is the group's engine ``AllReduce``; ``build``
    returns None unless it runs on MNNVL in the buffer's dtype and the shared MNNVL workspace
    already holds every call up to ``max_rows`` (growing it would move buffers other owners hold
    views of).
    """

    def __init__(
        self, all_reduce: Any, offset: Optional[int], columns: int, buffer: torch.Tensor
    ) -> None:
        self.all_reduce = all_reduce
        self.offset = offset
        self.columns = columns
        self.buffer = buffer

    @classmethod
    def build(
        cls,
        all_reduce: Any,
        offset: Optional[int],
        columns: int,
        width: int,
        max_rows: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Optional["ZeroPaddedReduce"]:
        mnnvl = getattr(all_reduce, "mnnvl_allreduce", None)
        if mnnvl is None or mnnvl.dtype != dtype:
            return None
        # The kernels move 16-byte vectors along a row.
        vector = 16 // dtype.itemsize
        width = -(-width // vector) * vector
        if offset is not None and not 0 <= offset <= offset + columns <= width:
            raise ValueError(f"{columns} columns at {offset} outside a row of {width}")
        world = mnnvl.mapping.tp_size
        need = max(
            mnnvl.get_required_workspace_size(rows, width, world, dtype)
            for rows in range(1, max_rows + 1)
        )
        workspace = get_or_scale_allreduce_mnnvl_workspace(mnnvl.mapping, dtype)
        if need > workspace["buffer_size_bytes"]:
            return None
        return cls(
            all_reduce, offset, columns, torch.zeros(max_rows, width, dtype=dtype, device=device)
        )

    def fits(self, rows: int, columns: int) -> bool:
        """Whether a ``[rows, columns]`` copy goes through this buffer; the same answer on every
        rank of the group."""
        return 0 < rows <= self.buffer.shape[0] and columns == self.columns

    def stage(self, local: torch.Tensor, rows: int) -> None:
        """This rank's ``local [rows, columns]`` into its columns of the buffer (no exchange)."""
        if self.offset is not None:
            self.buffer[:rows, self.offset : self.offset + local.shape[1]].copy_(local)

    def reduce(self, rows: int) -> torch.Tensor:
        """The all-reduced ``[rows, width]`` buffer, as staged."""
        return self.all_reduce(self.buffer[:rows])

    def __call__(self, local: torch.Tensor, rows: int) -> torch.Tensor:
        """The all-reduced ``[rows, width]`` buffer after this rank's ``local [rows, columns]``."""
        self.stage(local, rows)
        return self.reduce(rows)


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
    # draft_q_a_kv_a_fused: [q_a ; kv_a] rows (d_l{j}_q_a_kv_a), which q_a views; None otherwise
    q_a_kv_a: Optional[torch.Tensor] = None


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
    page_table: torch.Tensor  # the draft page table, rows named by table_rows (int32)
    table_rows: torch.Tensor  # [B]: each request's row of page_table
    page_size: int
    positions: torch.Tensor  # [rows]: the queries' positions
    pools: Any  # per drafter layer, this rank's history pool [pages, 1, 1, page, 576]


class _DraftRt(NamedTuple):
    """The drafter's runtime (``rt.draft``), built at the worker's first prepare, before capture."""

    freqs: torch.Tensor  # [positions, rope / 2] complex64: the drafter's YaRN table
    softmax_scale: float
    eps: float
    kv_scale: float  # the FP8 history's dequant scale
    taps: int  # the block convolutions' taps
    groups: int  # channel groups of a convolution's per-position kernel
    group: int  # channels per group
    mapping: Any  # the target's helix mapping: the history's page owners, the exchange's group
    cp_size: int  # its CP ranks
    cp_rank: int  # this rank's place among them
    plan: Any  # the window entry's plan buffers, for up to max_requests requests
    workspace: Any  # the window entry's body partials
    block_kv: torch.Tensor  # [max rows, 1, 576]: the block latents, in the history's dtype
    block_slots: torch.Tensor  # [max rows] int64: block_kv's row of each query
    block_offsets: torch.Tensor  # [max rows] int64: zeros (one latent per scratch row)
    stats: torch.Tensor  # [max rows, heads, 2] FP32: each row's (max, sum) over its local keys
    fc_ready: torch.cuda.Event  # recorded on the aux stream after a projection issued early
    mla: dict  # rows -> the CuTe DSL window's buffers (draft_window_mla), made at an eager step
    history_read: torch.cuda.Event  # recorded on the aux stream after it read the history positions


def _rows_key(t: torch.Tensor) -> tuple:
    """What identifies the rows a tensor views: storage address, shape, strides, dtype, device."""
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device)


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

    @property
    def table(self) -> torch.Tensor:
        """The ``[vocab, hidden]`` table, the full vocabulary on every rank."""
        return self._core().w["embed"]

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
    assert rope_type == "yarn", f"the DFlash2 drafter's RoPE is YaRN, got {rope_type!r}"
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
        # draft_q_a_kv_a_fused: the table stacks q_a with kv_a; q_a views the stack.
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
        spec_config.speculative_model, **external_drafter_config_kwargs(model_config, spec_config)
    )
    # The target's KV dtype; the history's own dtype comes from the container (context_kv_fp8).
    draft_config.quant_config.kv_cache_quant_algo = model_config.quant_config.kv_cache_quant_algo
    draft_config.extra_attrs = model_config.extra_attrs
    cfg = draft_config.pretrained_config
    for key, want in _weights.DRAFT_SHAPE.items():
        got = getattr(cfg, key, None)
        assert got == want, f"the DFlash2 drafter's {key} is {got}, this target runs {want}"
    dflash = getattr(cfg, "dflash_config", None) or {}
    for key, want in _weights.DRAFT_DFLASH.items():
        assert dflash.get(key) == want, f"dflash_config.{key} {dflash.get(key)} is not {want}"
    layer_types = getattr(cfg, "layer_types", None) or []
    assert layer_types and all(t == "sliding_attention" for t in layer_types), layer_types
    assert getattr(cfg, "is_causal", True), "the DFlash2 drafter's block attention is causal"
    return draft_config


class DraftModel(nn.Module):
    """The DFlash2 drafter the engine's worker drives (helix8: 96 heads, page-sharded history).

    No parameter or submodule of its own: every weight is ``core.w["d_*"]``, read through the core
    (held weakly), so meta initialization never leaves a stale tensor here. A module only because
    the engine counts a context server's drafter tensors through ``parameters()``,
    ``buffers()`` and ``modules()``.

    ``model_config`` is the drafter's ``ModelConfig`` (``draft_config_for``); the speculative
    config is the core's. On a context server with ``context_kv_only`` the drafter only builds the
    history (``fc``, ``hidden_norm``, every layer's latent projection and norm) and gives the
    engine ``fc``'s column blocks to project the target's captures as they are taken.
    """

    # The engine's unified draft cache holds the history, in aggregated runs too.
    supports_unified_draft_cache = True
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
        self.context_kv_only = bool(getattr(spec_config, "context_kv_only", False))
        history = getattr(spec_config, "context_kv_dtype", None)
        assert history in _DRAFT_HISTORY_CONFIGS, (
            f"this target keeps the drafter history in {_DRAFT_HISTORY_DTYPE}: "
            f"speculative_config.context_kv_dtype {history!r} names another storage"
        )
        # The engine sizes the history pool from this flag.
        self.context_kv_fp8 = _DRAFT_HISTORY_DTYPE == torch.float8_e4m3fn
        self.block_size = spec_config.max_draft_len + 1
        self.mask_token_id = dflash["mask_token_id"]
        self.context_cache_window = int(cfg.sliding_window)
        self._num_attn_layers = cfg.num_hidden_layers
        # Every rank attends with every head over its own pages.
        self._num_heads = cfg.num_attention_heads
        # The history is page-sharded over the target's own helix mapping, so the KV cache creator
        # shards the draft pages as the target's and the exchange reuses the target's group.
        self._helix_mapping = core.cp_mapping
        assert self._helix_mapping.has_cp_helix(), "helix8's drafter shards its history by page"
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
        self._freqs: Optional[torch.Tensor] = None
        self._rt: Any = None
        # The projection issued early this step: (rows key, projection, its event, the history
        # latents made with it or None), or None.
        self._early_projection: Optional[tuple] = None
        # The early projection the preparation took and its history latents (draft_history_aux),
        # until the preparation's precompute_context_kv takes them.
        self._early_latent: Optional[tuple] = None
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
        manager is replaced); ``_runtime_position_ceiling`` sizes the RoPE table, ``max_rows``
        (requests x block) and the history's ``page_size`` the window entry's buffers.
        """
        core = self._core()
        cfg = self.config
        self._dw = _draft_views(
            core.w, cfg.num_hidden_layers, cfg.qk_nope_head_dim, cfg.kv_lora_rank
        )
        positions = self._runtime_position_ceiling or cfg.max_position_embeddings
        dflash = cfg.dflash_config
        device = self._dw.fc.device
        draft = _DraftRt(
            freqs=_draft_freqs(self._rope, int(positions), cfg.qk_rope_head_dim, device),
            softmax_scale=self.softmax_scale,
            eps=float(cfg.rms_norm_eps),
            kv_scale=DRAFT_FP8_KV_SCALE,
            taps=int(dflash["conv_kernel_size"]),
            groups=cfg.hidden_size // int(dflash["conv_group_size"]),
            group=int(dflash["conv_group_size"]),
            mapping=self._helix_mapping,
            cp_size=self._helix_mapping.cp_size,
            cp_rank=self._helix_mapping.cp_rank,
            plan=plan_buffers(
                max_rows // self.block_size,
                self.block_size,
                _DRAFT_WINDOW,
                page_size,
                self._helix_mapping.cp_size,
                _DRAFT_WINDOW_SPLITS,
                device,
            ),
            workspace=window_workspace(
                max_rows // self.block_size,
                self.block_size,
                self._num_heads,
                cfg.kv_lora_rank,
                _DRAFT_WINDOW_SPLITS,
                device,
            ),
            block_kv=torch.empty(
                max_rows, 1, self._head_dim, dtype=_DRAFT_HISTORY_DTYPE, device=device
            ),
            block_slots=torch.arange(max_rows, dtype=torch.int64, device=device),
            block_offsets=torch.zeros(max_rows, dtype=torch.int64, device=device),
            stats=torch.empty(max_rows, self._num_heads, 2, dtype=torch.float32, device=device),
            fc_ready=torch.cuda.Event(),
            mla={},
            history_read=torch.cuda.Event(),
        )
        # Also an attribute: the engine counts a context server's drafter tables off the module.
        self._freqs = draft.freqs
        self._rt = core._rt._replace(draft=draft)

    # -- the history -----------------------------------------------------------

    def project_target_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``hidden_norm(fc(captures))``: the target's captured features in the drafter's space.

        When ``issue_projection`` projected these same rows earlier in the step, the current
        stream waits for that projection's event and gets it: no second projection runs.
        """
        early, self._early_projection = self._early_projection, None
        if early is None:
            return self._project(hidden_states)
        key, projected, ready, latent = early
        if key != _rows_key(hidden_states):
            raise RuntimeError(
                "DFlash2: the drafter projection issued before the acceptance was of other rows "
                f"({key[1]}) than the ones the preparation projects ({tuple(hidden_states.shape)})"
            )
        ready.wait()
        if latent is not None:
            self._early_latent = (projected, latent)
        return projected

    def _project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The fc GEMM and the hidden norm on the current stream."""
        fc = self._w("d_fc")
        return flashinfer_rmsnorm(
            rubin_linear(hidden_states.to(fc.dtype), fc, False),
            self._w("d_hidden_norm"),
            float(self.config.rms_norm_eps),
        )

    def issue_projection(
        self,
        hidden_states: torch.Tensor,
        history_positions: Optional[Callable[[], torch.Tensor]] = None,
    ) -> None:
        """Project ``hidden_states`` now, on the aux stream, for this step's preparation.

        The aux stream first waits for the current stream (everything issued so far, the target
        forward that wrote the captures included); the fc GEMM and the hidden norm then run there
        while the current stream takes the acceptance. The next ``project_target_hidden`` of the
        same rows waits for the event recorded after them and returns the result. Same kernels on
        the same rows: the same bits. The result is read on the current stream only after that
        wait, and the next step's work on the aux stream forks from the current stream. A
        projection a failed step left behind is dropped (nothing reads it).

        With ``history_positions`` (draft_history_aux) the aux stream first evaluates it (the
        rows' history positions, read before the current stream's preparation advances the
        lengths: ``wait_history_read``), then after the projection makes the rows' history
        latents (``_context_kv``), which the preparation's ``precompute_context_kv`` returns.
        """
        main = torch.cuda.current_stream()
        aux = self._rt.aux_stream
        ready = self._rt.draft.fc_ready
        latent = None
        aux.wait_stream(main)
        self.join_taps(aux)  # draft_tap_stream: the captures' copies, on the aux side
        with torch.cuda.stream(aux):
            if history_positions is not None:
                positions = history_positions()
                self._rt.draft.history_read.record()
            projected = self._project(hidden_states)
            if history_positions is not None:
                latent = self._context_kv(projected, positions)
            ready.record()
        if not torch.cuda.is_current_stream_capturing():
            projected.record_stream(main)
            if latent is not None:
                latent.record_stream(main)
        self._early_projection = (_rows_key(hidden_states), projected, ready, latent)
        logger.info_once(_DRAFT_FC_EARLY_LINE, key=_DRAFT_FC_EARLY_LINE)

    def join_taps(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        """Have ``stream`` (the current one by default) wait for this step's capture copies on the
        tap stream (``draft_tap_stream``); nothing without the lever or once joined."""
        taps = self._rt.taps if self._rt is not None else None
        if taps is not None:
            taps.join(stream)

    def wait_history_read(self) -> None:
        """The current stream waits until the aux stream has read the history positions of the
        projection issued early (draft_history_aux): the preparation advances the lengths."""
        torch.cuda.current_stream().wait_event(self._rt.draft.history_read)

    def projection_pending(self) -> bool:
        """Whether an early projection is still waiting for its preparation."""
        return self._early_projection is not None

    def precompute_context_kv(self, projected_hidden: torch.Tensor, positions: torch.Tensor):
        """Every layer's history latent: ``([N, layers, 1, 576], None)``.

        One projection over the stacked ``kv_a`` rows (``_draft_linear``), then [RMSNorm | RoPE] of
        all layers in one launch. When the preparation's projection is the early one whose
        latents the aux stream made (draft_history_aux; ``project_target_hidden`` waited for
        them), those latents.
        """
        early, self._early_latent = self._early_latent, None
        if early is not None and early[0] is projected_hidden:
            logger.info_once(_DRAFT_HISTORY_AUX_LINE, key=_DRAFT_HISTORY_AUX_LINE)
            return early[1], None
        return self._context_kv(projected_hidden, positions), None

    def _context_kv(self, projected_hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """``precompute_context_kv``'s latents ``[N, layers, 1, 576]`` on the current stream."""
        kv_a = self._w("d_kv_a")
        latent = _draft_linear(projected_hidden.to(kv_a.dtype), kv_a, self.block_size)
        latent = latent.view(projected_hidden.shape[0], self._num_attn_layers, self._head_dim)
        out = dflash2_mla_ctx_norm_rope(
            latent,
            self._w("d_kv_a_norm"),
            float(self.config.rms_norm_eps),
            self._rt.draft.freqs,
            positions.view(-1),
        )
        return out.unsqueeze(2)

    def capture_projection_blocks(self, num_capture_layers: int) -> list[torch.Tensor]:
        """``fc``'s ``[out, hidden]`` column block per captured layer (views of ``d_fc``)."""
        fc = self._w("d_fc")
        width, rest = divmod(fc.shape[1], num_capture_layers)
        assert not rest, f"fc's {fc.shape[1]} inputs do not split over {num_capture_layers} layers"
        return [fc[:, i * width : (i + 1) * width] for i in range(num_capture_layers)]

    def finish_capture_projection(self, projected: torch.Tensor) -> torch.Tensor:
        """``hidden_norm`` of the FP32 sum of the blocks' products, rounded once to fc's dtype."""
        return flashinfer_rmsnorm(
            projected.to(self._w("d_fc").dtype),
            self._w("d_hidden_norm"),
            float(self.config.rms_norm_eps),
        )

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

    def select_candidate_tokens(
        self, candidate_ids, unary_logits, block_hidden_states, anchor_token_ids, block_dtype
    ) -> torch.Tensor:
        """The greedy walk's tokens, int32 ``[n, K]``: the argmax of the block
        ``select_candidate_path`` writes into a ``-inf`` block of ``block_dtype``, without the
        block."""
        return k31_dflash2_selector_greedy_tokens(
            self._w("d_sel_pred"),
            self._w("d_sel_succ"),
            self._w("d_sel_hidden_proj"),
            candidate_ids,
            unary_logits,
            block_hidden_states,
            anchor_token_ids,
            block_dtype,
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
            page_table=ctx_page_table,
            table_rows=ctx_cache_batch_idx,
            page_size=ctx_kv_cache[0].shape[-2],
            positions=query_positions.reshape(-1),
            pools=ctx_kv_cache,
        )
        return draft_step(self._dw, noise_embedding.reshape(batch * block, hidden), st, self._rt)


def draft_step(DW: _DraftW, hidden_streams: torch.Tensor, st: _DraftStep, rt: _Rt) -> torch.Tensor:
    """One DFlash2 block over the five drafter layers; returns ``[B * block, hidden]``.

    ``hidden_streams`` is the noise block's embedding. Once per step the window plan of this rank's
    keys. Per layer: input norm, the attention conv's input side, the absorbed MLA query, the block
    latents into the scratch, the window over this rank's keys (partial and log-sum-exp), W_UV, the
    helix exchange to this rank's 12 heads, ``o_proj`` and its all-reduce, the conv's output side;
    then the MLP between the MLP conv's two sides. Then the final norm.
    """
    if _DRAFT_LOWM_GEMM or _DRAFT_CONV_NORM:
        return _draft_step_fused(DW, hidden_streams, st, rt)
    d = rt.draft
    eps, taps, groups, group = d.eps, d.taps, d.groups, d.group
    rows, block = st.rows, st.block
    heads, nope, kv_lora = DW.layers[0].k_b.shape
    latent = DW.layers[0].kv_a.shape[0]
    # The CuTe DSL window's buffers at this row count (draft_window_mla), else None: then, once per
    # step, the window's body, head and block keys this rank owns, per request.
    mla = _draft_window_mla(d, rows, block, heads, st.page_size, hidden_streams.device)
    plan = None
    if mla is None:
        plan = rubin_dflash2_window_plan(
            st.num_ctx,
            st.page_table,
            st.table_rows,
            block,
            _DRAFT_WINDOW,
            st.page_size,
            d.cp_size,
            d.cp_rank,
            _DRAFT_WINDOW_SPLITS,
            d.plan,
        )
    block_kv = d.block_kv[:rows]
    stats = d.stats[:rows]
    vec_layers = 0  # layers whose window kernels ran ALIGNED (draft_window_vec_io)
    conv_pdl = _DRAFT_CONV_PDL and get_env_enable_pdl()  # draft_conv_pdl
    # draft_q_a_kv_a_fused: [q_a ; kv_a] in one GEMM on this step (the measured block, its rows in
    # the bands).
    one_q_kv = (
        DW.layers[0].q_a_kv_a is not None
        and block == _DRAFT_LOWM_BLOCK
        and _moe_in_bands(_DRAFT_Q_A_KV_A_BANDS, rows)
    )

    h = hidden_streams
    residual = h.clone()  # the fused add + norm below updates it in place
    x = flashinfer_rmsnorm(h, DW.layers[0].in_norm, eps)
    for j, L in enumerate(DW.layers):
        if j:
            x, residual = flashinfer_fused_add_rmsnorm(h, residual, L.in_norm, eps)
        coef = _gemm_pinned(
            "draft_attn_conv_proj",
            x,
            L.attn_conv_proj.t(),
            lambda: rubin_linear(x, L.attn_conv_proj, False),
        ).view(rows, 2, taps, groups)
        x = dflash2_grouped_conv(x, coef[:, 0], L.attn_conv_base[0], block, group, conv_pdl)
        if not one_q_kv:
            qa = flashinfer_rmsnorm(rubin_linear(x, L.q_a, False), L.q_a_norm, eps)
        else:
            # draft_q_a_kv_a_fused: one GEMM over [q_a ; kv_a]; the q_a norm reads its first
            # q_lora columns, the block RoPE the rest.
            q_kv = rubin_linear(x, L.q_a_kv_a, False)
            qa = flashinfer_rmsnorm(q_kv[:, : L.q_a.shape[0]], L.q_a_norm, eps)
        q = _gemm_pinned("draft_q_b", qa, L.q_b.t(), lambda: rubin_linear(qa, L.q_b, False))
        q = q.view(rows, heads, -1)
        query = q.new_empty(rows, heads, latent)
        bmm_out(q[..., :nope].transpose(0, 1), L.k_b, query[..., :kv_lora].transpose(0, 1))
        dflash2_mla_block_rope(
            q,
            query,
            q_kv[:, L.q_a.shape[0] :] if one_q_kv else rubin_linear(x, L.kv_a, False),
            L.kv_a_norm,
            eps,
            d.freqs,
            st.positions,
            block_kv,
            d.block_slots[:rows],
            d.block_offsets[:rows],
        )
        ctx, layer_stats, empty, aligned = _draft_window(
            query, st.pools[j][:, 0, 0], block_kv[:, 0], plan, mla, d, st, stats, kv_lora
        )
        vec_layers += aligned
        v_dim = L.v_b_t.shape[-1]
        partial = ctx.new_empty(rows, heads, v_dim)
        bmm_out(ctx.transpose(0, 1), L.v_b_t, partial.transpose(0, 1))
        attn = rubin_dflash2_helix_combine(
            partial, layer_stats, d.mapping, heads, v_dim, zero_kv_mask=empty
        )
        o = _gemm_pinned("draft_o", attn, L.o.t(), lambda: rubin_linear(attn, L.o, False))
        a = allreduce_module(o, rt.all_reduce, _AR_NONE, None, None)
        a = dflash2_grouped_conv(a, coef[:, 1], L.attn_conv_base[1], block, group, conv_pdl)
        x, residual = flashinfer_fused_add_rmsnorm(a, residual, L.post_norm, eps)
        coef = _gemm_pinned(
            "draft_mlp_conv_proj",
            x,
            L.mlp_conv_proj.t(),
            lambda: rubin_linear(x, L.mlp_conv_proj, False),
        ).view(rows, 2, taps, groups)
        x = dflash2_grouped_conv(x, coef[:, 0], L.mlp_conv_base[0], block, group, conv_pdl)
        gu = _gemm_pinned(
            "draft_gate_up", x, L.gate_up.t(), lambda: rubin_linear(x, L.gate_up, False)
        )
        act = flashinfer_silu_and_mul(gu)
        m = _gemm_pinned("draft_down", act, L.down.t(), lambda: rubin_linear(act, L.down, False))
        m = allreduce_module(m, rt.all_reduce, _AR_NONE, None, None)
        h = dflash2_grouped_conv(m, coef[:, 1], L.mlp_conv_base[1], block, group, conv_pdl)
    h, _ = flashinfer_fused_add_rmsnorm(h, residual, DW.final_norm, eps)
    if vec_layers:
        line = _DRAFT_WINDOW_VEC_LINE.format(vec_layers, len(DW.layers))
        logger.info_once(line, key=_DRAFT_WINDOW_VEC_LINE)
    return h


def _draft_lowm_plan(block: int) -> Optional[dict]:
    """The low-M plan of a drafter of ``block`` queries per request: ``_DRAFT_LOWM_PLAN`` at the
    block its rows were measured at (``_DRAFT_LOWM_BLOCK``), else None (cuBLAS)."""
    return _DRAFT_LOWM_PLAN if block == _DRAFT_LOWM_BLOCK else None


def _draft_linear(
    x: torch.Tensor, weight: torch.Tensor, block: int, unplanned: Optional[Callable] = None
) -> torch.Tensor:
    """``x @ weight.T`` of a drafter projection of a ``block``-query drafter: the low-M kernel
    ``_draft_lowm_plan(block)`` names for its shape and rows (with the engine's PDL switch), else
    ``unplanned()`` (the site's ``_gemm_pinned`` call, whose GEMM is ``rubin_linear``'s), else
    ``F.linear`` (``rubin_linear``'s GEMM)."""
    plan = _draft_lowm_plan(block)
    if unplanned is not None and plan_choice(plan, x.shape[0], *weight.shape) is None:
        return unplanned()
    return k31_low_m_linear(x, weight, plan, get_env_enable_pdl())


def _draft_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float, pdl: bool) -> torch.Tensor:
    """A drafter RMSNorm: ``dflash2_rmsnorm`` under ``draft_conv_norm`` (FlashInfer's kernel, the
    dependents released after its grid wait under ``pdl``), else FlashInfer's; the same bits."""
    if _DRAFT_CONV_NORM:
        return dflash2_rmsnorm(x, weight, eps, pdl)
    return flashinfer_rmsnorm(x, weight, eps)


def _draft_conv_add_norm(
    a: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    residual: torch.Tensor,
    norm: torch.Tensor,
    eps: float,
    block: int,
    group: int,
    pdl: bool,
    early: bool,
) -> tuple:
    """A block convolution's output side, the residual add and the next RMSNorm: ``(normed,
    residual)``, ``residual`` updated in place. One launch under ``draft_conv_norm``
    (``dflash2_conv_add_rmsnorm``; ``early`` releases its dependents after its grid wait; each row
    over a ``draft_norm_ctas`` cluster when that is set), else the convolution (with PDL under
    ``draft_conv_pdl``) and FlashInfer's fused add + RMSNorm; the same bits but for the cluster's
    sum of squares, which only a drafter of ``_DRAFT_LOWM_BLOCK``'s block takes."""
    if _DRAFT_CONV_NORM:
        ctas = _DRAFT_NORM_CTAS if block == _DRAFT_LOWM_BLOCK else 0
        return dflash2_conv_add_rmsnorm(
            a, delta, base, residual, norm, eps, block, group, pdl, early, ctas
        )
    a = dflash2_grouped_conv(a, delta, base, block, group, _DRAFT_CONV_PDL and get_env_enable_pdl())
    return flashinfer_fused_add_rmsnorm(a, residual, norm, eps)


def _draft_window_mla(
    d: _DraftRt, rows: int, block: int, heads: int, page_size: int, device: torch.device
) -> Optional[Any]:
    """The CuTe DSL window's buffers for ``rows`` block rows of a ``block``-query drafter under
    ``draft_window_mla``, else None.

    Only for a drafter of ``_DRAFT_LOWM_BLOCK``'s block and at the row counts of
    ``draft_window_mla_bands``: at any other the Triton window runs and no buffers are made. Made
    at the first step of that row count that is not being captured (the warm-up before a graph's
    capture) and kept in ``d.mla``; a capture of a row count without them runs the Triton window,
    and says so once. The pack widens the FP8 history at a unit scale, so a
    history of another dequant scale keeps the Triton window."""
    if not _DRAFT_WINDOW_MLA or block != _DRAFT_LOWM_BLOCK or d.kv_scale != 1.0:
        return None
    if not _moe_in_bands(_DRAFT_WINDOW_MLA_BANDS, rows):
        return None
    ws = d.mla.get(rows)
    if ws is None:
        if torch.cuda.is_current_stream_capturing():
            logger.warning_once(
                f"DFlash2 window: no CuTe DSL window buffers for {rows} rows under capture; the "
                "Triton window runs",
                key=f"dflash2_window_mla_missing_{rows}",
            )
            return None
        ws = d.mla[rows] = helix_window_workspace(
            rows, heads, page_size, _DRAFT_WINDOW, d.cp_size, device
        )
    logger.info_once(_DRAFT_WINDOW_MLA_LINE, key=_DRAFT_WINDOW_MLA_LINE)
    return ws


def _draft_window(
    query: torch.Tensor,
    pool: torch.Tensor,
    block_kv: torch.Tensor,
    plan: Any,
    mla: Optional[Any],
    d: _DraftRt,
    st: _DraftStep,
    stats: torch.Tensor,
    kv_lora: int,
) -> tuple:
    """One drafter layer's window over this rank's keys: ``(partial, stats, empty, aligned)``.

    With ``mla`` (``_draft_window_mla``'s buffers) the CuTe DSL window: its own (lse, 1) stats and
    the rows with no local key, which the helix combine sanitizes. Else the Triton window over
    ``plan`` into ``stats`` (an empty row arrives sanitized, ``empty`` is None), and ``aligned``
    says whether both of its kernels ran with 16-byte vectors (``draft_window_vec_io``)."""
    if mla is not None:
        ctx, layer_stats, empty = rubin_ull_dflash2_helix_window(
            query,
            pool,
            st.page_table,
            st.table_rows,
            st.num_ctx,
            block_kv,
            st.block,
            d.softmax_scale,
            d.cp_size,
            d.cp_rank,
            mla,
        )
        return ctx, layer_stats, empty, False
    ctx = rubin_dflash2_window_triton(
        query,
        pool,
        block_kv,
        plan,
        d.softmax_scale,
        kv_lora,
        d.kv_scale,
        d.workspace,
        stats,
        vec_io=_DRAFT_WINDOW_VEC_IO,
        pdl=_DRAFT_WINDOW_PDL,
    )
    aligned = _DRAFT_WINDOW_VEC_IO and window_vec_io_aligned(
        query, pool, block_kv, d.workspace, kv_lora
    )
    return ctx, stats, None, aligned


def _draft_step_fused(
    DW: _DraftW, hidden_streams: torch.Tensor, st: _DraftStep, rt: _Rt
) -> torch.Tensor:
    """``draft_step`` under ``draft_lowm_gemm`` / ``draft_conv_norm``: the same block decode, with
    every BF16 projection through ``_draft_linear`` (the history's in ``precompute_context_kv``;
    a projection gemm_pins_draft pins runs its pin at the rows the plan names no kernel for), the
    norms and convolutions as the levers launch them, and the window as ``draft_step`` runs it
    (``_draft_window``: the CuTe DSL route where ``_draft_window_mla`` has buffers). Each block
    convolution's output side meets the residual add + RMSNorm that follows it (the next layer's
    input norm, or the final norm after the last layer) in ``_draft_conv_add_norm``, as
    ``draft_step`` computes them in two launches. The helix exchange and the all-reduces are
    ``draft_step``'s."""
    d = rt.draft
    eps, taps, groups, group = d.eps, d.taps, d.groups, d.group
    rows, block = st.rows, st.block
    heads, nope, kv_lora = DW.layers[0].k_b.shape
    latent = DW.layers[0].kv_a.shape[0]
    mla = _draft_window_mla(d, rows, block, heads, st.page_size, hidden_streams.device)
    plan = None
    if mla is None:
        plan = rubin_dflash2_window_plan(
            st.num_ctx,
            st.page_table,
            st.table_rows,
            block,
            _DRAFT_WINDOW,
            st.page_size,
            d.cp_size,
            d.cp_rank,
            _DRAFT_WINDOW_SPLITS,
            d.plan,
        )
    block_kv = d.block_kv[:rows]
    stats = d.stats[:rows]
    vec_layers = 0  # layers whose window kernels ran ALIGNED (draft_window_vec_io)
    # The convolutions and norms launch with PDL under draft_conv_norm and the engine's switch; the
    # final norm keeps its release at its end (its readers are the engine's). draft_conv_pdl alone
    # launches the convolutions with PDL.
    pdl = _DRAFT_CONV_NORM and get_env_enable_pdl()
    conv_pdl = (_DRAFT_CONV_NORM or _DRAFT_CONV_PDL) and get_env_enable_pdl()
    # draft_q_a_kv_a_fused: [q_a ; kv_a] in one GEMM on this step (the measured block, its rows in
    # the bands).
    one_q_kv = (
        DW.layers[0].q_a_kv_a is not None
        and block == _DRAFT_LOWM_BLOCK
        and _moe_in_bands(_DRAFT_Q_A_KV_A_BANDS, rows)
    )

    h = hidden_streams
    residual = h.clone()  # the conv + add + norm below updates it in place
    x = _draft_rmsnorm(h, DW.layers[0].in_norm, eps, pdl)
    for j, L in enumerate(DW.layers):
        coef = _draft_linear(
            x,
            L.attn_conv_proj,
            block,
            lambda: _gemm_pinned(
                "draft_attn_conv_proj",
                x,
                L.attn_conv_proj.t(),
                lambda: rubin_linear(x, L.attn_conv_proj, False),
            ),
        ).view(rows, 2, taps, groups)
        x = dflash2_grouped_conv(x, coef[:, 0], L.attn_conv_base[0], block, group, conv_pdl)
        if not one_q_kv:
            qa = _draft_rmsnorm(_draft_linear(x, L.q_a, block), L.q_a_norm, eps, pdl)
        else:
            # draft_q_a_kv_a_fused: one GEMM over [q_a ; kv_a]; the q_a norm reads its first
            # q_lora columns, the block RoPE the rest.
            q_kv = _draft_linear(x, L.q_a_kv_a, block)
            qa = _draft_rmsnorm(q_kv[:, : L.q_a.shape[0]], L.q_a_norm, eps, pdl)
        q = _draft_linear(
            qa,
            L.q_b,
            block,
            lambda: _gemm_pinned(
                "draft_q_b", qa, L.q_b.t(), lambda: rubin_linear(qa, L.q_b, False)
            ),
        ).view(rows, heads, -1)
        query = q.new_empty(rows, heads, latent)
        bmm_out(q[..., :nope].transpose(0, 1), L.k_b, query[..., :kv_lora].transpose(0, 1))
        dflash2_mla_block_rope(
            q,
            query,
            q_kv[:, L.q_a.shape[0] :] if one_q_kv else _draft_linear(x, L.kv_a, block),
            L.kv_a_norm,
            eps,
            d.freqs,
            st.positions,
            block_kv,
            d.block_slots[:rows],
            d.block_offsets[:rows],
        )
        ctx, layer_stats, empty, aligned = _draft_window(
            query, st.pools[j][:, 0, 0], block_kv[:, 0], plan, mla, d, st, stats, kv_lora
        )
        vec_layers += aligned
        v_dim = L.v_b_t.shape[-1]
        partial = ctx.new_empty(rows, heads, v_dim)
        bmm_out(ctx.transpose(0, 1), L.v_b_t, partial.transpose(0, 1))
        attn = rubin_dflash2_helix_combine(
            partial, layer_stats, d.mapping, heads, v_dim, zero_kv_mask=empty
        )
        o = _draft_linear(
            attn,
            L.o,
            block,
            lambda: _gemm_pinned("draft_o", attn, L.o.t(), lambda: rubin_linear(attn, L.o, False)),
        )
        a = allreduce_module(o, rt.all_reduce, _AR_NONE, None, None)
        x, residual = _draft_conv_add_norm(
            a, coef[:, 1], L.attn_conv_base[1], residual, L.post_norm, eps, block, group, pdl, True
        )
        coef = _draft_linear(
            x,
            L.mlp_conv_proj,
            block,
            lambda: _gemm_pinned(
                "draft_mlp_conv_proj",
                x,
                L.mlp_conv_proj.t(),
                lambda: rubin_linear(x, L.mlp_conv_proj, False),
            ),
        ).view(rows, 2, taps, groups)
        x = dflash2_grouped_conv(x, coef[:, 0], L.mlp_conv_base[0], block, group, conv_pdl)
        gu = _draft_linear(
            x,
            L.gate_up,
            block,
            lambda: _gemm_pinned(
                "draft_gate_up", x, L.gate_up.t(), lambda: rubin_linear(x, L.gate_up, False)
            ),
        )
        act = flashinfer_silu_and_mul(gu)
        m = _draft_linear(
            act,
            L.down,
            block,
            lambda: _gemm_pinned(
                "draft_down", act, L.down.t(), lambda: rubin_linear(act, L.down, False)
            ),
        )
        m = allreduce_module(m, rt.all_reduce, _AR_NONE, None, None)
        last = j + 1 == len(DW.layers)
        x, residual = _draft_conv_add_norm(
            m,
            coef[:, 1],
            L.mlp_conv_base[1],
            residual,
            DW.final_norm if last else DW.layers[j + 1].in_norm,
            eps,
            block,
            group,
            pdl,
            not last,
        )
    if vec_layers:
        line = _DRAFT_WINDOW_VEC_LINE.format(vec_layers, len(DW.layers))
        logger.info_once(line, key=_DRAFT_WINDOW_VEC_LINE)
    if _draft_lowm_plan(block) is not None:
        L = DW.layers[0]
        sites = ("attn_conv_proj", "q_a", "q_b", "kv_a", "o", "gate_up", "down")
        if one_q_kv:
            sites = ("attn_conv_proj", "q_a_kv_a", "q_b", "o", "gate_up", "down")
        kinds = []
        for name in sites:
            n, k = getattr(L, name).shape
            choice = plan_choice(_draft_lowm_plan(block), rows, n, k)
            kinds.append(f"{name}={'cublas' if choice is None else choice[0]}")
        cls = "D" if _DRAFT_LOWM_PLAN is K31_LOW_M_DRAFT_PLAN_HELIX8_D else "B"
        line = _DRAFT_LOWM_LINE.format(cls, rows, ", ".join(kinds))
        logger.info_once(line, key=f"{_DRAFT_LOWM_LINE}{rows}")
    return x


def _pairs_in_id_order(pairs: torch.Tensor) -> torch.Tensor:
    """One rank's top-32 ``(global id, value)`` FP32 pairs ``[n, K, 64]`` in ascending id order.

    ``torch.topk`` takes the lowest index first among values equal to the 32nd, and its 32-element
    sorting network receives the selected values in index order; where equal values end up depends
    on that input order. The full-vocab top-32 sees the candidates in id order; the merge sees the
    gathered pairs in position order, rank by rank. With each rank's 32 in id order the positions
    follow the ids, so the merge selects the same 32, hands the network the same sequence and
    returns the full-vocab top-32 bit for bit. The ids are distinct integers below 2**24, exact in
    FP32, so the order is unique.
    """
    p = pairs.unflatten(-1, (-1, 2))
    order = p[..., 0].argsort(dim=-1)
    return p.gather(-2, order.unsqueeze(-1).expand_as(p)).flatten(-2)


class DFlashWorker(_EngineDFlashWorker):
    """The engine's DFlash worker with this target's DFlash2 differences.

    Every method not defined here is the engine's: the managed draft cache and its history, the
    prefill and paged history stores, the fused generation prepare, the KV-length rewind. Defined
    here: the unified-cache check and the drafter's runtime (``_lazy_init_ctx_buffers``), the
    target-metadata hook, acceptance and sampling through the family's spec entries, the
    forced-acceptance count, the DFlash2 sampled path (selector walk, block verification) and
    ``_forward_impl``.
    """

    # The selector stores global candidate probabilities for block verification.
    supports_cp_rejection_sampling = True
    # draft_agree_check: the census's device totals [mismatched real rows, real rows, mismatched
    # padding rows, padding rows] (bound with the drafter's runtime), the readout in flight (event,
    # pinned copy, steps) and the steps counted (_report_draft_agreement). Class defaults, so a
    # worker assembled without __init__ has no census; so have the zero-padded copies.
    _agree_count: Optional[torch.Tensor] = None
    _agree_pending: Optional[tuple] = None
    _agree_steps = 0
    _pairs_gather: Optional["ZeroPaddedReduce"] = None
    _draft_agreement: Optional["ZeroPaddedReduce"] = None

    def __init__(self, spec_config: Any, mapping: Any, use_separate_draft_kv_cache: bool = False):
        super().__init__(spec_config, mapping, use_separate_draft_kv_cache)
        history = getattr(spec_config, "context_kv_dtype", None)
        if history not in _DRAFT_HISTORY_CONFIGS:
            raise ValueError(
                f"this target keeps the drafter history in {_DRAFT_HISTORY_DTYPE}: "
                f"speculative_config.context_kv_dtype {history!r} names another storage; leave "
                "it 'auto' or set 'fp8'"
            )
        if self.force_num_accepted_tokens != 0.0 and self._force_accept_rng_pool is None:
            raise RuntimeError(
                "the forced-acceptance draws must exist before any CUDA graph capture"
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
        # The drafter's top-32 pairs all-gather and the draft-token agreement as zero-padded MNNVL
        # all-reduces (_bind_zero_padded_copies, under draft_pairs_zero_padded /
        # draft_agree_zero_padded; None: the all-gathers).
        self._pairs_gather: Optional[ZeroPaddedReduce] = None
        self._draft_agreement: Optional[ZeroPaddedReduce] = None

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
                spec_metadata.max_num_requests * self._compute_block_size, self._ctx_page_size
            )
            self._bind_zero_padded_copies(draft_model, spec_metadata.max_num_requests)

    def _bind_zero_padded_copies(self, draft_model, max_requests: int) -> None:
        """The drafter's two TP8 copies as zero-padded all-reduces on the target head's FP32
        MNNVL pair exchange (``ZeroPaddedReduce``), each under its constant, when the selection
        is vocab-parallel: the top-32 ``(global id, value)`` pairs (each rank its 2 x top_k
        columns, in the worker mapping's rank order, as ``allgather_module``) and the draft-token
        agreement (rank 0's tokens; ids below 2**24 are exact in FP32). With draft_agree_check,
        the census's totals. Every rank binds at the same point, outside graph capture."""
        self._pairs_gather = self._draft_agreement = None
        if _DRAFT_AGREE_CHECK and self._agree_count is None:
            self._agree_count = torch.zeros(4, dtype=torch.int64, device=self._ctx_len.device)
        head = self.target_vocab_parallel_head
        if not (_DRAFT_PAIRS_ZERO_PADDED or _DRAFT_AGREE_ZERO_PADDED) or head is None:
            return
        if draft_model.context_kv_only or not draft_model.has_candidate_selector:
            return
        exchange = head._exchange
        if (exchange.tp_size, exchange.tp_rank) != (self._vocab_shards, self._vocab_shard):
            return
        device = self._ctx_len.device
        k, rank = self.max_draft_len, exchange.tp_rank
        pair_width = 2 * draft_model.candidate_selector.top_k
        if _DRAFT_PAIRS_ZERO_PADDED:
            self._pairs_gather = ZeroPaddedReduce.build(
                exchange.all_reduce,
                rank * pair_width,
                pair_width,
                exchange.tp_size * pair_width,
                max_requests * k,
                torch.float32,
                device,
            )
            if self._pairs_gather is not None:
                line = _DRAFT_ZERO_PADDED_LINE.format(
                    "top-32 pairs gather", max_requests * k, exchange.tp_size * pair_width
                )
                logger.info_once(line, key=line)
        agreement = self._draft_agreement_mapping()
        if (
            _DRAFT_AGREE_ZERO_PADDED
            and self.agree_draft_tokens_across_tp
            and agreement is not None
            and (agreement.tp_size, agreement.tp_rank) == (exchange.tp_size, rank)
            and head.vocab_size < 2**24
        ):
            self._draft_agreement = ZeroPaddedReduce.build(
                exchange.all_reduce,
                0 if rank == 0 else None,
                k,
                _DRAFT_AGREEMENT_WIDTH,
                max_requests,
                torch.float32,
                device,
            )
            if self._draft_agreement is not None:
                line = _DRAFT_ZERO_PADDED_LINE.format(
                    "draft-token agreement", max_requests, _DRAFT_AGREEMENT_WIDTH
                )
                logger.info_once(line, key=line)

    def _agree_draft_tokens(self, outputs) -> None:
        """The engine's agreement (every rank takes rank 0's ``next_draft_tokens`` and the draft
        columns of ``next_new_tokens``): one broadcast through ``_draft_agreement`` where bound and
        the shapes are its own, else the engine's all-gather. Under draft_agree_check each row's
        local drafts are compared with the agreed ones (``_count_agreement``)."""
        drafts = outputs.get("next_draft_tokens") if isinstance(outputs, dict) else None
        new_tokens = outputs.get("next_new_tokens") if isinstance(outputs, dict) else None
        agreement = self._draft_agreement
        if (
            agreement is None
            or drafts is None
            or drafts.dim() != 2
            or not agreement.fits(*drafts.shape)
            or (
                new_tokens is not None
                and tuple(new_tokens.shape) != (drafts.shape[0], drafts.shape[1] + 1)
            )
        ):
            super()._agree_draft_tokens(outputs)
        else:
            rows, width = drafts.shape
            agreed = agreement(drafts, rows)[:, :width].to(drafts.dtype)
            outputs["next_draft_tokens"] = agreed
            if new_tokens is not None:
                outputs["next_new_tokens"] = torch.cat(
                    [new_tokens[:, :1], agreed.to(new_tokens.dtype)], dim=1
                )
        agreed = outputs.get("next_draft_tokens") if isinstance(outputs, dict) else None
        if self._agree_count is not None and drafts is not None and agreed is not drafts:
            self._count_agreement(drafts, agreed)

    def _count_agreement(self, local: torch.Tensor, agreed: torch.Tensor) -> None:
        """draft_agree_check: add this call's rows whose local drafts differ from the agreed (rank
        0's) ones to the device totals, real rows and CUDA-graph padding rows (the dummy slot)
        apart. Device adds only: graph replays count."""
        rows = local.shape[0]
        differs = (local != agreed).any(-1)
        slots = None if self._batch_to_slot_buffer is None else self._batch_to_slot_buffer.tensor
        if slots is None or slots.shape[0] < rows:
            real = torch.ones_like(differs)
        else:
            real = slots[:rows] != self._dummy_slot
        count = self._agree_count
        count[0].add_((differs & real).sum())
        count[1].add_(real.sum())
        count[2].add_((differs & ~real).sum())
        count[3].add_((~real).sum())

    def _report_draft_agreement(self) -> None:
        """draft_agree_check: every ``_DRAFT_AGREE_PERIOD`` steps one ``K31_H8_AGREE`` line of the
        census's totals, read without a host synchronization (copied to pinned memory on the
        stream and logged at a later step whose check finds the copy complete); a WARNING the
        first time a real row differed. Host-side, outside graph capture, at every step's draft
        table refresh (CUDA-graph replays included)."""
        count = self._agree_count
        if count is None or torch.cuda.is_current_stream_capturing():
            return
        pending = self._agree_pending
        if pending is not None and pending[0].query():
            _, host, steps = pending
            totals = host.tolist()
            logger.info(_DRAFT_AGREE_LINE.format(self._vocab_shard, *totals, steps))
            if totals[0]:
                logger.warning_once(
                    "DFlash2 drafts differed across the TP group on real rows "
                    f"({totals[0]} of {totals[1]} by step {steps}): the agreement is required",
                    key="k31_h8_agree_mismatch",
                )
            self._agree_pending = pending = None
        self._agree_steps += 1
        if pending is None and self._agree_steps % _DRAFT_AGREE_PERIOD == 1:
            host = self.__dict__.get("_agree_host")
            if host is None:
                host = self._agree_host = torch.empty(
                    4, dtype=torch.int64, pin_memory=prefer_pinned()
                )
            host.copy_(count, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
            self._agree_pending = (event, host, self._agree_steps)

    def _refresh_ctx_block_tables(
        self, attn_metadata, num_seqs: int, request_ids: list[int] | None = None
    ) -> bool:
        """The engine's refresh; under draft_agree_check the census is read out here first
        (``_report_draft_agreement``)."""
        self._report_draft_agreement()
        return super()._refresh_ctx_block_tables(attn_metadata, num_seqs, request_ids)

    def _issue_draft_projection(
        self, draft_model, attn_metadata, spec_metadata, total_target_tokens: int, num_gens: int
    ) -> bool:
        """Issue the drafter's projection of this step's captured rows before the acceptance.

        Only where the fused preparation (``_prepare_paged_latent_gen_inputs``) projects them on
        a CUDA graph step (warmup or capture) of generation rows only, and only those rows: the
        slice that preparation projects, ``captured[num_ctx_tokens : + num_gens * width]``. The
        captures are written by the target forward, before this call in stream order; the
        acceptance's outputs reach the preparation's positions and stores, never these rows.
        Eager steps keep the projection inside the preparation. Returns whether it issued one.
        """
        if not (
            spec_metadata.is_cuda_graph
            and num_gens > 0
            and attn_metadata.num_contexts == 0
            and not draft_model.context_kv_only
            and self._fused_gen_prepare_applies(spec_metadata)
        ):
            return False
        width = self._draft_tokens_per_req
        gen_start = attn_metadata.num_ctx_tokens
        captured_hs = spec_metadata.get_hidden_states(total_target_tokens)
        gen_hs = captured_hs[gen_start : gen_start + num_gens * width]
        positions = None
        if _DRAFT_HISTORY_AUX:
            # The preparation's history positions (dflash_paged_latent_prepare's ctx_positions):
            # each request's history length before this step's advance, plus 0 .. width - 1.
            ctx_len = self._ctx_len
            slots = self._batch_to_slot_buffer.tensor[:num_gens]

            def positions():
                offsets = torch.arange(width, dtype=ctx_len.dtype, device=ctx_len.device)
                return (ctx_len[slots.long()][:, None] + offsets[None, :]).reshape(-1)

        draft_model.issue_projection(gen_hs.reshape(-1, gen_hs.shape[-1]), positions)
        return True

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
        """The engine's forced acceptance; counts what it forced (both verify paths call it)."""
        out = super()._apply_force_accepted_tokens(
            num_accepted_tokens, num_contexts, runtime_draft_len, spec_metadata=spec_metadata
        )
        if self.force_num_accepted_tokens != 0.0:
            self._count_forced(out, num_contexts, spec_metadata)
        return out

    def _count_forced(
        self, out: torch.Tensor, num_contexts: int, spec_metadata, counted: bool = False
    ) -> None:
        """Add this call's forced rows to the device totals (unless ``select_accept`` already
        added them: ``counted``); log them every 200 calls.

        Rows that verified placeholder drafts kept their own count and are left out. The totals
        are device adds, so graph replays count; the line is written from Python calls outside
        capture (eager steps).
        """
        if self._forced is None:
            self._forced = torch.zeros(2, dtype=torch.int64, device=out.device)
        capturing = out.is_cuda and torch.cuda.is_current_stream_capturing()
        if spec_metadata is not None and spec_metadata.is_cuda_graph and not capturing:
            return  # graph warmup: the engine forced nothing
        if not counted:
            gen = out[num_contexts:]
            rows = self._real_draft_rows(spec_metadata, gen.shape[0])
            if rows is None:
                self._forced[0].add_(gen.shape[0])
                self._forced[1].add_(gen.sum())
            else:
                self._forced[0].add_(rows.sum())
                self._forced[1].add_(torch.where(rows, gen, 0).sum())
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

    def _fused_select_accept(self, logits, attn_metadata, spec_metadata):
        """``(accepted_tokens, num_accepted_tokens)`` with one ``select_accept`` launch doing
        what the unfused steps do (draft_select_accept), or None for them.

        On the exchanged argmax pairs of this rank's shard (an all-greedy step whose target tokens
        come from the shard), the launch reduces the pairs (``vocab_parallel_argmax``'s last
        kernel), accepts strictly (``strict_accept``), forces the count (the engine's forced
        acceptance: the same pool, counter and warmup skip, only the rows
        ``SpecMetadata.real_draft_rows`` marks) and adds it to the forced totals, writes the KDA
        replay record (``update_mamba_states`` where that is all it writes,
        :meth:`_kda_replay_operands`) and the drafter's kv_lens prep
        (``_prepare_kv_for_draft_forward``: generation rows + 1, the rewind ``1 - accepted``
        returned). ``_prepare_attn_metadata_for_dflash`` runs first: it only saves metadata the
        launch does not read. Integer and FP32-compare work: the values of the unfused steps.
        """
        if not _DRAFT_SELECT_ACCEPT:
            return None
        if (
            not isinstance(logits, VocabShardedLogits)
            or spec_metadata.is_ragged_verify
            or not spec_metadata.is_all_greedy_sample
            or getattr(spec_metadata, "enable_penalty", False)
            or dflash2_sampled_step(self, spec_metadata)
        ):
            return self._log_select_accept("not a greedy shard step")
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        if max(num_contexts, num_gens) > _SELECT_ACCEPT_MAX_REQUESTS:
            return self._log_select_accept(f"batch {num_contexts}/{num_gens}")
        local = logits.local
        draft_tokens = self._reshape_draft_tokens_for_accept(spec_metadata, num_gens, local.device)
        draft_len = draft_tokens.shape[-1]
        if (
            draft_tokens.dim() != 2
            or draft_tokens.shape[0] != num_gens
            or (num_gens > 0 and draft_tokens.stride(-1) != 1)
            or local.dim() != 2
            or local.stride(-1) != 1
            or local.shape[0] != num_contexts + num_gens * (draft_len + 1)
        ):
            return self._log_select_accept("layout")
        kda_replay = None
        if num_gens > 0:
            why, kda_replay = self._kda_replay_operands(attn_metadata)
            if why is not None:
                return self._log_select_accept(f"kda {why}")
        kv_lens = getattr(attn_metadata, "kv_lens_cuda", None)
        self._prepare_attn_metadata_for_dflash(attn_metadata, spec_metadata)
        count = self._forced_totals(spec_metadata, local.device)
        force = self._force_accept_kernel_args(draft_len, spec_metadata, local.device)
        accepted, num_accepted, rewind = select_accept(
            self._selection_pairs(logits),
            draft_tokens,
            num_contexts,
            batch_size,
            force,
            kda_replay=kda_replay,
            kv_lens=kv_lens,
            count=count,
            real_rows=None if force is None else self._real_draft_rows(spec_metadata, num_gens),
        )
        if self.force_num_accepted_tokens != 0.0:
            self._count_forced(num_accepted, num_contexts, spec_metadata, counted=count is not None)
        if kv_lens is not None:
            self._kv_rewind_amount = rewind
            self._kv_rewind_nc = num_contexts
            self._kv_rewind_bs = batch_size
            self._kv_rewind_pending = True
            attn_metadata.update_for_spec_dec(refresh_device_metadata=False)
        self._log_select_accept(None, kv_prep=kv_lens is not None, kda=kda_replay is not None)
        return accepted, num_accepted

    def _log_select_accept(self, reason: Optional[str], kv_prep: bool = False, kda: bool = False):
        """One INFO line per path on its first call outside graph capture (the served evidence of
        which acceptance ran); returns None (the unfused path) for a reason."""
        key = reason or f"kv_prep={int(kv_prep)} kda_replay={int(kda)}"
        logged = self.__dict__.setdefault("_select_accept_logged", set())
        if key not in logged and not torch.cuda.is_current_stream_capturing():
            logged.add(key)
            what = f"reason={reason}" if reason else key
            logger.info(_SELECT_ACCEPT_LINE.format(0 if reason else 1, what))
        return None

    @staticmethod
    def _selection_pairs(logits: VocabShardedLogits) -> tuple:
        """Every rank's ``(value, global index)`` argmax pairs of the shard logits, exchanged and
        not yet reduced: ``(pairs, row stride, slot stride, slots)``, the operand of
        ``select_accept``. The local kernel and the exchange of ``vocab_parallel_argmax`` (whose
        last kernel, the pair reduction, ``select_accept`` does in its launch)."""
        local = logits.local
        exchange = logits.head._exchange
        rows, n_cols = local.shape
        layout = exchange.layout(rows, n_cols, local.device)
        if rows > 0:
            _local_argmax_kernel[(rows, layout.splits)](
                local,
                local.stride(0),
                n_cols,
                exchange.tp_rank * n_cols,
                layout.chunk,
                layout.buffer,
                layout.buffer.stride(0),
                layout.slot0,
                layout.zero_lo,
                layout.zero_hi,
                ZERO_WIDTH=layout.zero_width,
                BLOCK_V=_LOCAL_BLOCK_V,
                num_warps=8,
            )
        pairs, slots = exchange.exchange(layout)
        return pairs, pairs.stride(0), 2, slots

    @staticmethod
    def _kda_replay_operands(attn_metadata):
        """``(why_unfused, kda_replay)`` for ``select_accept``.

        ``why_unfused`` is None when ``update_mamba_states(attn_metadata, num_accepted,
        state_indices=mamba_metadata.state_indices)`` is the engine's ``KimiK3HybridCacheManager``
        one (the class behind ``RubinKdaDsaHybridCacheManager``, through
        ``MambaHybridCacheManagerV2``) and writes nothing but ``KDAReplayState.record_acceptance``;
        ``kda_replay`` is then ``(prev_num_accepted_tokens, state_indices, dummy_mask)``, or None
        when that call records nothing (a context-only KDA manager, or no local KDA layers).
        """
        mgr = attn_metadata.kv_cache_manager
        if (
            not isinstance(mgr, KimiK3HybridCacheManager)
            or type(mgr).update_mamba_states is not KimiK3HybridCacheManager.update_mamba_states
        ):
            return "manager", None
        if mgr.kda_context_only or mgr.local_num_mamba_layers == 0:
            return None, None
        replay = mgr._speculative_state
        # A subclass (RubinKdaReplayState) qualifies when it records the acceptance with the
        # base methods.
        if (
            not isinstance(replay, KDAReplayState)
            or type(replay).update is not KDAReplayState.update
            or type(replay).record_acceptance is not KDAReplayState.record_acceptance
        ):
            return "replay_class", None
        if replay.prev_num_accepted_tokens is None:
            return "replay_unbound", None
        dummy_mask = mgr._dummy_request_mask
        if dummy_mask is None:
            return "dummy_mask", None
        state_indices = attn_metadata.mamba_metadata.state_indices
        if state_indices is None:
            state_indices = mgr.get_state_indices()
        if (
            replay.prev_num_accepted_tokens.dim() != 1
            or replay.prev_num_accepted_tokens.dtype != torch.int32
            or state_indices.stride(-1) != 1
            or dummy_mask.stride(-1) != 1
        ):
            return "layout", None
        return None, (replay.prev_num_accepted_tokens, state_indices, dummy_mask)

    def _force_accept_kernel_args(
        self, runtime_draft_len: int, spec_metadata, device: torch.device
    ) -> Optional[tuple]:
        """``select_accept``'s ``force`` operand reproducing ``_apply_force_accepted_tokens``:
        None when it changes nothing, ``(base_total,)`` for a fixed count, else the fractional
        draw's pool, counter and strides."""
        if self.force_num_accepted_tokens == 0.0:
            return None
        if spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing():
            return None
        int_part = int(self.force_num_accepted_tokens)
        frac_part = self.force_num_accepted_tokens - int_part
        max_total = runtime_draft_len + 1
        base_total = min(int_part + 1, max_total)
        if frac_part > 0.0 and base_total < max_total:
            self._ensure_force_accept_rng_state(device)
            return (
                base_total,
                frac_part,
                max_total,
                self._force_accept_rng_pool,
                self._force_accept_rng_counter,
                _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
                _FORCE_ACCEPT_RNG_SLOT_STRIDE,
            )
        return (base_total,)

    def _forced_totals(self, spec_metadata, device: torch.device) -> Optional[torch.Tensor]:
        """The forced-acceptance totals for ``select_accept`` to add to (int64 ``[generation
        rows, accepted tokens]``), or None: no forced acceptance, or a graph warmup (the engine
        forces nothing there)."""
        if self.force_num_accepted_tokens == 0.0:
            return None
        if self._forced is None:
            self._forced = torch.zeros(2, dtype=torch.int64, device=device)
        capturing = torch.cuda.is_current_stream_capturing()
        if spec_metadata is not None and spec_metadata.is_cuda_graph and not capturing:
            return None
        return self._forced

    def _fused_kv_rewind_operands(
        self, attn_metadata, spec_metadata, batch_size, accepted_tokens, next_draft_tokens
    ):
        """``(kv_lens, rewind, num_contexts)`` when the pending rewind after the draft forward
        runs inside the next-new-tokens kernel (draft_tail_rewind:
        ``draft_next_new_tokens_rewind``), else None (``_apply_kv_rewind_after_draft`` and
        ``_prepare_next_new_tokens``). A warmup restores ``kv_lens`` wholesale and rewinds
        nothing, so it takes the unfused path."""
        if not _DRAFT_TAIL_REWIND:
            return None
        if spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing():
            return None
        amount = self._kv_rewind_amount
        if (
            not self._kv_rewind_pending
            or amount is None
            or not hasattr(attn_metadata, "kv_lens_cuda")
            or self._kv_rewind_bs != batch_size
            or not amount.is_contiguous()
            or next_draft_tokens.shape[0] != batch_size
            or not next_draft_tokens.is_contiguous()
            or accepted_tokens.stride(-1) != 1
        ):
            return None
        return attn_metadata.kv_lens_cuda, amount, self._kv_rewind_nc

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
                target_probs, draft_tokens, candidate_ids, q, seed, offset, _DRAFT_BLOCK_VERIFY
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
            (accepted_tokens, num_accepted_tokens), batch_size, spec_metadata
        )

    def _dflash2_global_top_k(
        self, gen_logits, spec_metadata, top_k: int, full_vocab: int, block: bool = True
    ):
        """Per-position global top-k candidates and the tensor to score into (None without
        ``block``).

        This rank's top-32; when the drafter's logits are a vocab shard, the shard's ids made
        global, its pairs in id order (``_pairs_in_id_order`` of the shard's top-32, or the
        selection kernel's index-order pairs under draft_index_pairs), gathered and merged: the
        full-vocab top-32's ids, values and order. The gather is ``_pairs_gather``'s zero-padded
        all-reduce where bound and the rows fit (draft_pairs_zero_padded), else the all-gather;
        the same rank-order layout. The tensor to score into is FP32, as the full-vocab logits
        are.
        """
        shard = gen_logits.shape[-1]
        if shard == full_vocab:
            unary_logits, candidate_ids = k31_dflash2_shard_top32(gen_logits, top_k, 0, False)
            return candidate_ids, unary_logits, gen_logits if block else None
        assert self._draft_logits_are_sharded(gen_logits, spec_metadata) and (
            shard * self._vocab_shards == full_vocab
        ), f"draft logits of width {shard} are neither the vocab {full_vocab} nor a TP shard of it"
        if _DRAFT_INDEX_PAIRS and top_k == _SELECTOR_TOP_K:
            ordered = k31_dflash2_shard_index_pairs(gen_logits, self._vocab_shard * shard)
        else:
            combined = k31_dflash2_shard_top32(gen_logits, top_k, self._vocab_shard * shard, True)
            ordered = _pairs_in_id_order(combined)
        n, slots, width = ordered.shape
        pairs = getattr(self, "_pairs_gather", None)
        if pairs is not None and pairs.fits(n * slots, width):
            gathered = pairs(ordered.view(n * slots, width), n * slots).view(n, slots, -1)
        else:
            gathered = allgather_module(ordered, self.mapping, -1, None)
        unary_logits, candidate_ids = k31_dflash2_merge_top32(gathered.contiguous(), top_k)
        if not block:
            return candidate_ids, unary_logits, None
        # The selector's greedy path fills it with -inf before it writes the candidates; the
        # sampled path does not read it.
        block_logits = gen_logits.new_empty(
            (*gen_logits.shape[:-1], full_vocab), dtype=torch.float32
        )
        return candidate_ids, unary_logits, block_logits

    def _draft_shard_head(self, draft_model, spec_metadata):
        """The vocab-parallel head whose shard the drafter's logits stay on, or None (gathered).

        The drafter reads the target's LM head and logits processor; the engine binds a
        vocab-parallel head over that LM head only behind the plain logits processor (no guided
        decoder, no penalties), whose logits are the head's, gathered and cast to FP32. The shard
        is used when the selector drafts (its candidates are the one consumer of the logits), the
        head is this worker's shard of the drafter's vocabulary and the engine takes a row of the
        shard's width for a vocab shard, as ``_dflash2_global_top_k`` asserts. Host-side facts
        only: the same on every rank, for a step and its CUDA graph.
        """
        if not (_DRAFT_SHARD_TOP32 and draft_model.has_candidate_selector):
            return None
        head = self.target_vocab_parallel_head
        target = None if self._target_model_ref is None else self._target_model_ref()
        if (
            head is None
            or target is None
            or head.lm_head is not draft_model.lm_head
            or draft_model.logits_processor is not target.logits_processor
        ):
            return None
        width = head.shard_width
        offset = width * self._vocab_shard
        if head.vocab_size != width * self._vocab_shards or head.col_base != offset:
            return None
        shard = torch.empty((0, width), device="meta")
        return head if self._draft_logits_are_sharded(shard, spec_metadata) else None

    def _prepare_paged_latent_gen_inputs(
        self,
        accepted_tokens,
        num_accepted_tokens,
        attn_metadata,
        spec_metadata,
        draft_model,
        total_target_tokens,
        mask_token_id,
        trained_mask_embedding,
    ) -> dict:
        """The engine's generation preparation; under draft_noise_block (and without a trained
        mask row) the drafter's input block is one copy kernel over the target's embedding table
        (``k31_dflash2_noise_block``). The same inputs and side effects otherwise."""
        if not _DRAFT_NOISE_BLOCK or trained_mask_embedding is not None:
            return super()._prepare_paged_latent_gen_inputs(
                accepted_tokens,
                num_accepted_tokens,
                attn_metadata,
                spec_metadata,
                draft_model,
                total_target_tokens,
                mask_token_id,
                trained_mask_embedding,
            )
        num_contexts = attn_metadata.num_contexts
        num_gens = attn_metadata.num_seqs - num_contexts
        width = self._draft_tokens_per_req
        block_size = self._compute_block_size
        shard = self._helix_shard()
        prep = dflash_paged_latent_prepare(
            num_accepted_tokens,
            accepted_tokens,
            self._batch_to_slot_buffer.tensor[num_contexts : num_contexts + num_gens],
            self._ctx_len,
            self._ctx_block_counts,
            self._ctx_block_tables,
            attn_metadata._seq_lens_cuda,
            num_contexts,
            self._dummy_slot,
            self._max_ctx,
            self._ctx_page_size,
            block_size,
            width,
            position_offset=self._ctx_position_offset if self._has_unified_draft_cache() else None,
            cp_size=1 if shard is None else shard.cp_size,
            cp_rank=0 if shard is None else shard.cp_rank,
        )
        attn_metadata._seq_lens[num_contexts : num_contexts + num_gens] = width

        noise_embedding = k31_dflash2_noise_block(
            draft_model.draft_model_full.model.embed_tokens.table,
            prep.bonus,
            mask_token_id,
            block_size,
        )
        gen_start = attn_metadata.num_ctx_tokens
        captured_hs = spec_metadata.get_hidden_states(total_target_tokens)
        gen_hs = captured_hs[gen_start : gen_start + num_gens * width]
        projected = draft_model.project_target_hidden(gen_hs.reshape(-1, gen_hs.shape[-1]))
        latent_dtype = self._ctx_kv_buf[0].dtype
        if latent_dtype == torch.float8_e4m3fn:
            # Project in the drafter's weight dtype; the store quantizes.
            latent_dtype = projected.dtype
        latent, _ = draft_model.precompute_context_kv(
            projected.to(latent_dtype), prep.ctx_positions
        )
        store_context_latents(
            latent[:, :, 0],
            self._ctx_kv_buf,
            self._ctx_kv_ptrs,
            prep.store_pages,
            prep.store_offsets,
            prep.store_mask,
        )
        return {
            "lilicorr_anchor": None,
            "noise_embedding": noise_embedding,
            "query_positions": prep.query_positions,
            "num_ctx_per_req": prep.num_ctx,
            "ctx_k_cache": self._ctx_k_buf,
            "ctx_v_cache": self._ctx_v_buf,
            "ctx_cache_batch_idx": torch.arange(
                num_contexts, num_contexts + num_gens, dtype=torch.long, device="cuda"
            ),
            "first_prev_tokens": prep.bonus,
            "ctx_kv_cache": self._ctx_kv_buf,
            "ctx_page_table": self._ctx_block_tables,
            "block_placement": DraftBlockPlacement(
                prep.blk_pages, prep.blk_offsets, prep.blk_seq_lens, prep.blk_kv_bounds
            ),
        }

    def _block_gather_ids(self, draft_model, num_gens, block_size, num_draft_tokens, num_rows):
        """Block-output rows whose hidden states give the K draft logits (slots 1 .. K)."""
        ids = self._draft_slot_ids(draft_model, num_gens, block_size, num_draft_tokens)
        return ids.clamp(max=num_rows - 1)

    def _block_rows(self, draft_model, hidden_states_out, num_gens: int, num_draft_tokens: int):
        """``hidden_states_out[_block_gather_ids(...)]``: slots 1 .. K of each request's block,
        contiguous. Under draft_block_rows, with the engine's slot convention and whole blocks,
        those rows are a strided view and one copy (``contiguous`` copies the one-slot view of
        K = 1 too); otherwise the index gather."""
        block = self._compute_block_size
        rows, hidden = hidden_states_out.shape
        if (
            _DRAFT_BLOCK_ROWS
            and type(self)._draft_slot_ids is _EngineDFlashWorker._draft_slot_ids
            and rows == num_gens * block
            and num_draft_tokens + 1 <= block
        ):
            view = hidden_states_out.view(num_gens, block, hidden)[:, 1 : num_draft_tokens + 1]
            return view.reshape(num_gens * num_draft_tokens, hidden).contiguous()
        return hidden_states_out[
            self._block_gather_ids(draft_model, num_gens, block, num_draft_tokens, rows)
        ]

    def _greedy_tokens_apply(self, draft_model, spec_metadata) -> bool:
        """Whether the greedy draft tokens come from the selector walk directly
        (draft_greedy_tokens: ``select_candidate_tokens``) instead of ``sample_draft_tokens``
        over the block: all-greedy drafting, no d2t, the 32-candidate selector, and no
        draft-confidence record."""
        if not _DRAFT_GREEDY_TOKENS:
            return False
        selector = draft_model.candidate_selector
        return (
            not spec_metadata.wants_advanced_draft_sampling
            and self._d2t is None
            and selector is not None
            and selector.top_k == _SELECTOR_TOP_K
            and (self._accept_stats is None or self._accept_stats.confidence_provider is None)
        )

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

        # On graph steps the drafter's projection of the captured target rows runs on the aux
        # stream beside the acceptance and its bookkeeping, which it does not read.
        early_fc = self._issue_draft_projection(
            draft_model, attn_metadata, spec_metadata, input_ids.shape[0], num_gens
        )
        if not early_fc:
            # draft_tap_stream: the captures are read on the current stream from here on.
            draft_model.join_taps()

        # Selection, acceptance and the bookkeeping after it in one launch where it applies
        # (draft_select_accept).
        fused = self._fused_select_accept(logits, attn_metadata, spec_metadata)
        if fused is not None:
            accepted_tokens, num_accepted_tokens = fused
        else:
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

        if fused is None:
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
            if early_fc and _DRAFT_HISTORY_AUX:
                # The aux stream read the history lengths the preparation advances.
                draft_model.wait_history_read()
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
            if early_fc and draft_model.projection_pending():
                raise RuntimeError(
                    "DFlash2: the preparation did not take the drafter projection issued early"
                )
            with self.draft_kv_cache_context(attn_metadata, None):
                hidden_states_out = draft_model.dflash_forward(
                    **{name: inputs[name] for name in _DRAFT_FORWARD_ARGS}
                )
                gen_hidden_states = self._block_rows(draft_model, hidden_states_out, num_gens, K)
                block_hidden = gen_hidden_states.reshape(num_gens, K, -1)
                shard_head = self._draft_shard_head(draft_model, spec_metadata)
                if shard_head is None:
                    gen_logits = draft_model.logits_processor(
                        gen_hidden_states, draft_model.lm_head, attn_metadata, True
                    )
                else:
                    # This rank's [rows, shard] logits, the columns the gathered logits hold at
                    # its offset: the selector's candidates come from the shard.
                    gen_logits = shard_head.local_logits(gen_hidden_states)
                    logger.info_once(_DRAFT_SHARD_LINE, key=_DRAFT_SHARD_LINE)
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
                elif self._greedy_tokens_apply(draft_model, spec_metadata):
                    # The greedy walk's tokens straight from the top-k candidates: the argmax of
                    # the block below (FP32 when the candidates come from the shard, else the
                    # logits themselves), without writing or reading the full-vocabulary block.
                    selector = draft_model.candidate_selector
                    full_vocab = selector.predecessor_codebook.shape[0]
                    candidate_ids, unary_logits, _ = self._dflash2_global_top_k(
                        gen_logits,
                        spec_metadata,
                        top_k=selector.top_k,
                        full_vocab=full_vocab,
                        block=False,
                    )
                    gen_draft_tokens = draft_model.select_candidate_tokens(
                        candidate_ids,
                        unary_logits,
                        block_hidden,
                        inputs["first_prev_tokens"].long(),
                        gen_logits.dtype if gen_logits.shape[-1] == full_vocab else torch.float32,
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
                # The drafter's vocabulary (a sampled step keeps the shard as gen_logits).
                vocab_size = gen_logits.shape[-1] if shard_head is None else shard_head.vocab_size
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
        rewind = self._fused_kv_rewind_operands(
            attn_metadata, spec_metadata, batch_size, accepted_tokens, next_draft_tokens
        )
        if rewind is not None:
            # The rewind rides in the next-new-tokens kernel (draft_tail_rewind).
            self._kv_rewind_pending = False
            next_new_tokens = draft_next_new_tokens_rewind(
                accepted_tokens[:batch_size],
                next_draft_tokens,
                num_accepted_tokens[:batch_size],
                *rewind,
            )
        else:
            self._apply_kv_rewind_after_draft(attn_metadata, spec_metadata)
        self._rollback_guided_decoder_after_verify(num_accepted_tokens)

        if rewind is None:
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

        return {
            # The gathered FP32 target logits, or this rank's vocab shard when the target tokens
            # came from it.
            "logits": raw_logits,
            "logits_vocab_shard": logits_vocab_shard,
            "new_tokens": accepted_tokens,
            "new_tokens_lens": num_accepted_tokens,
            "next_draft_tokens": next_draft_tokens,
            "next_new_tokens": next_new_tokens,
        }


# --------------------------------------------------------------------------------------------------
# Core, the prefill and decode targets, and the shell with the engine's hooks.
#
# The core holds ``self.w`` (the weight table's parameters) and nothing else: no layer, helper or
# backend object. The two targets run the 93 layers as straight-line code over the views: the mix,
# the mixer, the mix, the MLP. The shell builds the DFlash2 drafter and its worker at construction,
# as the DS target builds its MTP worker.
# --------------------------------------------------------------------------------------------------

# Where routing records the server role in the config's extra_attrs.
_ROLE_ATTR = "kimi_k31_disagg_role"
# Where routing records the values of the lever variables the target cannot set itself.
_LEVER_ENV_ATTR = "kimi_k31_lever_env"
# Positions past the configured max_seq_len the identity RoPE table also covers (the KV cache
# adds the draft tokens of a verify block).
_ROPE_HEADROOM = 4096


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


def _server_role(model_config: ModelConfig) -> str:
    """The server role, ``context`` or ``generation``, as routing recorded it.

    Routing records the role in ``extra_attrs`` and the loader carries it across its rebuild of
    the config. A config that did not come through routing has none: the target refuses it.
    """
    role = model_config.extra_attrs.get(_ROLE_ATTR)
    assert role is not None, (
        f"no server role in extra_attrs[{_ROLE_ATTR!r}]: routing.py records it when it returns "
        "this target"
    )
    assert role in _ROLES, f"this target serves {_ROLES}; routing recorded the role {role!r}"
    return role


def _check_lever_env(role: str, lever_env: dict) -> None:
    """Refuse lever variables that contradict this target's launch constants.

    ``lever_env`` is what routing read of the lever variables. The target passes its levers to the
    kernels as constants and reads none of these variables, which the engine paths it replaces
    read: one set to another value would be ignored silently. The per-call logits
    budget is the one the engine's KV cache estimate reserves.
    """
    constants = _CONSTANTS[role]
    check_fc12_lever_env(
        lever_env,
        die_layout=_FC12_DIE_SCHED,
        die_sched=_FC12_DIE_SCHED,
        die_min_tokens=_FC12_DIE_MIN_TOKENS,
        lookahead=_FC12_LOOKAHEAD,
        sm_reserve=_FC12_SM_RESERVE,
    )
    budget = lever_env.get("TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET")
    want = constants["indexer_logits_elem_budget"]
    if budget is not None and budget.strip() != str(want):
        raise ValueError(
            f"TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET={budget}: the target's per-call logits budget "
            f"is {want}, which the engine's KV cache estimate must reserve"
        )


def _check_build(model_config: ModelConfig) -> None:
    """Assert what routing admitted: the GPU, the K3.1 text model, this layout and the recipe."""
    cap = torch.cuda.get_device_capability()
    assert cap == _COMPUTE_CAPABILITY, f"this target runs on sm_107, not sm_{cap[0]}{cap[1]}"
    cfg = model_config.pretrained_config
    cfg = getattr(cfg, "text_config", None) or cfg
    for key, want in _K31.items():
        # The layer-wise benchmark builds a prefix of the layers; the pin holds the checkpoint's.
        got = (
            original_num_hidden_layers(cfg)
            if key == "num_hidden_layers"
            else getattr(cfg, key, None)
        )
        assert got == want, f"{key} {got} is not {want}"
    for group, table in (("linear_attn_config", _K31_KDA), ("dsa_config", _K31_DSA)):
        holder = getattr(cfg, group, None)
        for key, want in table.items():
            got = holder.get(key) if isinstance(holder, dict) else getattr(holder, key, None)
            assert got == want, f"{group}.{key} {got} is not {want}"
    # The indexer's key norm runs at rms_norm_eps; a checkpoint naming another epsilon is not this
    # recipe's.
    dsa = getattr(cfg, "dsa_config", None) or {}
    eps = (
        dsa.get("index_k_norm_eps")
        if isinstance(dsa, dict)
        else getattr(dsa, "index_k_norm_eps", None)
    )
    assert eps is None or eps == _K31["rms_norm_eps"], f"dsa_config.index_k_norm_eps {eps}"
    m = model_config.mapping
    layout = (m.world_size, m.tp_size, m.pp_size, m.cp_size, bool(m.enable_attention_dp))
    assert layout == (_WORLD_SIZE, 1, 1, _HELIX_CP, _ATTENTION_DP), f"not {_LAYOUT}: {layout}"
    assert m.has_cp_helix(), "helix8 runs helix context parallelism"
    assert m.moe_tp_ep_user_specified and (m.moe_tp_size, m.moe_ep_size) == (_MOE_TP, _MOE_EP), (
        m.moe_tp_size,
        m.moe_ep_size,
    )
    quant = model_config.quant_config
    assert quant.quant_algo == _QUANT_ALGO, f"checkpoint {quant.quant_algo}, not {_QUANT_ALGO}"
    assert quant.kv_cache_quant_algo == QuantAlgo.FP8, f"KV {quant.kv_cache_quant_algo}, not FP8"
    state = str(getattr(quant, "mamba_ssm_cache_dtype", None)).removeprefix("torch.")
    assert state == str(_STATE_DTYPE).removeprefix("torch."), f"KDA state {state}, not FP32"
    sparse = model_config.sparse_attention_config
    assert getattr(sparse, "algorithm", None) == "dsa", sparse
    assert getattr(sparse, "indexer_k_dtype", None) == _INDEXER_K_DTYPE, sparse
    spec = model_config.spec_config
    assert spec is not None and spec.spec_dec_mode.is_dflash(), "DFlash2 is part of the recipe"
    assert spec.spec_dec_mode.use_one_engine(), "DFlash2 runs one-engine"


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


def _helix_tp_mapping(mapping: Any) -> Any:
    """The helix ranks as TP ranks: the group of KDA, the MLPs, the MoE and every all-reduce."""
    tp = mapping.repurpose_helix_cp_to_tp()
    # The repurposed mapping passes its MoE sizes as if the user had set them; keep the flag.
    tp.moe_tp_ep_user_specified = mapping.moe_tp_ep_user_specified
    return tp


def _tap(spec_md: Any, layer: int) -> tuple | None:
    """``(spec_md, layer)`` when the drafter reads layer ``layer``'s output stream, else None."""
    if spec_md is None or layer < 0 or not spec_md.is_layer_capture(layer):
        return None
    return spec_md, layer


def _capture(rt: _Rt, tap: tuple, rows: torch.Tensor) -> None:
    """The drafter's capture of a tapped stream: on the tap stream (``draft_tap_stream``), else on
    the current stream."""
    if rt.taps is None:
        tap[0].maybe_capture_hidden_states(tap[1], rows, None)
    else:
        rt.taps.copy(tap, rows)


def _mix_in(
    rt: _Rt,
    prefix: torch.Tensor,
    pending: Any,
    bank: torch.Tensor,
    mw: _MixW,
    tap,
    mx: bool = False,
    snap: Optional[torch.Tensor] = None,
):
    """The mix ahead of a mixer; returns ``(prefix, normed)``.

    ``pending``: the previous layer's MLP output, which joins the prefix sum here, in the mix
    kernel where it serves the row count: a ``_MoeOut`` or ``_MoeTail`` through ``moe_norm``
    (``moe_fold_norm`` with ``attn_res_moe_fold``), layer 0's whole dense output through
    ``add_norm``. A tapped
    layer's stream, the pre-norm mixture, goes to the drafter first, so its mix and norm run apart.
    ``mx`` (a KDA layer): with ``attn_res_mx_moe`` a ``_MoeOut`` goes through ``moe_norm_mx``, so
    ``normed`` may come as ``(rows, codes, scales)``. ``snap`` (``mix_bank_in_mix``, a block
    boundary): the bank row of the boundary's snapshot; a MoE mix that serves the rows stores the
    updated prefix there and returns that row as ``prefix``. A ``_DensePartial``
    (``dense_ar_push``) joins the prefix through ``reduce_norm``. The tap's copy goes through
    ``_capture``.
    """
    if isinstance(pending, _DensePartial):
        if tap is None:
            return rt.reduce_norm(prefix, pending.partial, bank, mw)
        pending = rt.reduce(pending.partial)
    moe = isinstance(pending, (_MoeOut, _MoeTail))
    if tap is None:
        if pending is None:
            return prefix, rt.attn_res_norm(prefix, bank, mw)
        if moe:
            norm = (rt.moe_norm_mx if mx else None) or rt.moe_fold_norm or rt.moe_norm
            if snap is None:
                return norm(prefix, pending, bank, mw)
            return norm(prefix, pending, bank, mw, snap)
        return rt.add_norm(prefix, pending, bank, mw)
    if pending is not None:
        prefix = rt.moe_add(prefix, pending) if moe else prefix + pending
    mix = rt.attn_res_mix(prefix, bank, mw)
    _capture(rt, tap, mix)
    return prefix, rt.rms_norm(mix, mw.out_norm)


def _mix_out(rt: _Rt, prefix: torch.Tensor, pending: Any, bank: torch.Tensor, head: _HeadW, tap):
    """The output mix after the last layer; returns the final hidden states.

    The last MLP output joins the prefix sum first (``moe_add``), the drafter reads its tap of the
    last layer, then the mix and the final norm run. Untapped with ``attn_res_moe_fold``,
    ``moe_fold_norm`` runs the update inside the mix.
    """
    moe = isinstance(pending, (_MoeOut, _MoeTail))
    if tap is None and moe and rt.moe_fold_norm is not None:
        return rt.moe_fold_norm(prefix, pending, bank, head.out)[1]
    if pending is not None:
        prefix = rt.moe_add(prefix, pending) if moe else prefix + pending
    if tap is not None:
        _capture(rt, tap, rt.attn_res_mix(prefix, bank, head.out))
    return rt.attn_res_norm(prefix, bank, head.out)


class KimiK31ProNvfp4Sm107Helix8Core(DecoderModel):
    """The K3.1 text model at this rank's share: the weight table, the views and the two targets."""

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__(model_config)
        self.role = _server_role(model_config)
        self.lever_env = model_config.extra_attrs.get(_LEVER_ENV_ATTR)
        if self.lever_env is not None:
            _check_lever_env(self.role, self.lever_env)
        self.eps = float(model_config.pretrained_config.rms_norm_eps)
        # Helix CP8 for the MLA attention (and the executor); the same ranks as TP8 for the rest,
        # including every partial sum's all-reduce and the MoE. Both exist before dims(self)
        # reads them.
        self.cp_mapping = model_config.mapping
        self.tp_mapping = _helix_tp_mapping(self.cp_mapping)
        self.dims = _weights.dims(self)._replace(draft_q_a_kv_a_fused=_DRAFT_Q_A_KV_A)
        self.w = _weights.declare(self.dims)
        # Made once, never during a capture.
        self.streams = _Streams(*(torch.cuda.Stream() for _ in _Streams._fields))
        # moe_shared_late's raised streams (None when off).
        self.late_streams = _late_streams() if _MOE_SHARED_LATE_BANDS else None
        max_seq_len = model_config.max_seq_len
        self.rope_positions = (
            int(model_config.pretrained_config.max_position_embeddings)
            if max_seq_len is None
            else int(max_seq_len) + _ROPE_HEADROOM
        )
        self._layers: tuple | None = None
        self._head: _HeadW | None = None
        self._rt: _Rt | None = None
        self._prefill: PrefillTarget | None = None
        self._decode: DecodeTarget | None = None

    def post_load_weights(self) -> None:
        """Build the views, the runtime resources and the two targets, once the weights are real.

        Runs once: the shell calls it and so does the loader's walk over the modules. Collective
        over the helix group as TP ranks: every rank runs it at the same point.
        """
        if self._prefill is not None:
            return
        if self.role == "generation":
            # FC12's die schedule claims die-local SMs only over die-alternating expert weights;
            # the views below are taken of the re-homed parameters.
            rubin_fc12_place_per_die(self.w, self.dims.moe_layers, die_sched=_FC12_DIE_SCHED)
        self._layers = _layer_views(self)
        self._head = _head_view(self)
        self._rt = _build_runtime(self)
        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)
        log_constants(
            self.role,
            draft_context_kv_only=self.dims.draft_context_kv_only,
            lever_env=self.lever_env,
        )

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
    """Steps with context rows: helix8 serves none.

    helix8 is a generation-only target under helix context parallelism: a helix generation rank
    receives its KV from a context server and the engine estimates its KV pool without a context
    forward, so a step with context rows does not reach it.
    """

    def __init__(self, core: KimiK31ProNvfp4Sm107Helix8Core) -> None:
        self.core = core

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        spec_metadata: Any,
    ) -> torch.Tensor:
        """Refuse the step: helix8 runs generation steps only."""
        raise RuntimeError(
            f"{_TARGET} serves generation steps only (helix context parallelism); this step has "
            f"{attn_metadata.num_contexts} context requests"
        )


class DecodeTarget:
    """Generation-only steps; it states the context counts as 0 instead of reading them."""

    def __init__(self, core: KimiK31ProNvfp4Sm107Helix8Core) -> None:
        self.core = core
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
        """Run the layers over the generation rows (one verify block per request with DFlash2).

        A CUDA graph step whose rows fall in ``moe_shared_late_bands`` runs them on the core's
        ``late_streams`` (``_on_stream`` over ``_moe_shared_late_rt``); the step's state is built
        first, on the current stream.
        """
        rt = self._rt
        rt.dsa_pools.bind(attn_metadata.kv_cache_manager)
        if rt.ks_ws is None and attn_metadata.key_shard_world > 1:
            ks_ws = _key_shard_ws(self.core.tp_mapping, attn_metadata)
            rt = self._rt = self.core._rt = rt._replace(
                ks_ws=ks_ws, ks_sel=_key_shard_select(ks_ws) if _KS_SELECT_PACK else None
            )
        st = _build_step_args(attn_metadata, spec_metadata, rt.dsa_pools, decode=True)
        late = self.core.late_streams
        rows = (input_ids if inputs_embeds is None else inputs_embeds).shape[0]
        if late is not None and st.cuda_graph and _moe_shared_late_serves(rows):
            rt = _moe_shared_late_rt(rt, late, self.core.streams.aux)
            return _on_stream(late.fwd, self._run_layers, st, rt, input_ids, inputs_embeds)
        return self._run_layers(st, rt, input_ids, inputs_embeds)

    def _run_layers(
        self,
        st: _Step,
        rt: _Rt,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        """The embedding, the layers and the output mix on the current stream."""
        head = self._head
        if _MIX_BANK_IN_MIX and inputs_embeds is None:
            # mix_bank_in_mix: the embedding gathers into bank row 0, layer 0's snapshot (the gather
            # nn.functional.embedding runs on 1-D ids, into another buffer).
            embed = head.embed
            bank = embed.new_empty(head.n_bank, input_ids.shape[0], embed.shape[1])
            h = torch.index_select(embed, 0, input_ids, out=bank[0])
        else:
            h = (
                nn.functional.embedding(input_ids, head.embed)
                if inputs_embeds is None
                else inputs_embeds
            )
            bank = h.new_empty(head.n_bank, h.shape[0], h.shape[1])
        kda = kda_verify if st.rows_per_gen > 1 else kda_decode
        bp = rt.boundary_push if kda is kda_verify else None
        prefix, pending = h, None
        for i, L in enumerate(self._layers):
            snap = bank[L.n_bank] if L.boundary and _MIX_BANK_IN_MIX else None
            # With attn_res_mx_moe a KDA layer's projection takes the MXFP8 form of its input.
            prefix, x = _mix_in(
                rt,
                prefix,
                pending,
                bank[: L.n_bank],
                L.pre,
                _tap(st.spec_md, i - 1),
                L.kda is not None,
                snap,
            )
            if L.boundary:
                if snap is None or prefix.data_ptr() != snap.data_ptr():
                    bank[L.n_bank].copy_(prefix)
                else:
                    logger.info_once(_MIX_BANK_LINE, key=_MIX_BANK_LINE)
            # boundary_ar_push: the boundary's o_proj pushes once its shape passed the check.
            bmode = bp.mode(prefix.shape[0], L.n_bank + 1) if L.boundary and bp else None
            if L.kda is not None:
                if kda is kda_verify:
                    # kda_oproj_push: the mix after a non-boundary layer with snapshot rows
                    # finishes the call o_proj pushes (reduce_norm's push path).
                    push = (not L.boundary and L.n_bank > 0) or bmode == "push"
                    o = kda_verify(L.kda, x, st, rt, push)
                else:
                    o = kda(L.kda, x, st, rt)
            else:
                # dsa_oproj_push: the same push-path rule as the KDA verify's.
                o = mla_generate(L.mla, i, x, st, rt, not L.boundary and L.n_bank > 0)
            if L.boundary:
                if isinstance(o, _Pushed):
                    prefix, x = bp.finish(prefix, bank[: L.n_bank + 1], L.mid)
                elif bmode == "check":
                    prefix, x = bp.check(o, prefix, bank[: L.n_bank + 1], L.mid)
                else:
                    prefix = rt.reduce(o)
                    x = rt.attn_res_norm(prefix, bank[: L.n_bank + 1], L.mid)
            elif isinstance(o, _Pushed):
                prefix, x = rt.oproj_push.finish_norm(prefix, bank[: L.n_bank], L.mid)
            else:
                prefix, x = rt.reduce_norm(prefix, o, bank[: L.n_bank], L.mid)
            if L.moe is not None:
                pending = latent_moe(L.moe, i, x, st, rt)
            else:
                pending = dense_mlp(L.dense, x, st, rt)
        return _mix_out(rt, prefix, pending, bank, head, _tap(st.spec_md, len(self._layers) - 1))


class ModelingV2KimiK31ProNvfp4Sm107Helix8(
    DecoderModelForCausalLM[KimiK31ProNvfp4Sm107Helix8Core, PretrainedConfig]
):
    """Kimi K3.1 Pro NVFP4, sm_107, helix8: the shell and the engine hooks.

    The spec decoding wiring is the DS target's: the shell holds the drafter container
    (``draft_model``) and the engine's worker (``spec_worker``, a submodule the loader walks),
    and its forward runs the trunk, the target logits and the worker.
    """

    # The family KDA metadata class the executor builds.
    mamba_metadata_cls = RubinKdaMetadata

    def __init__(self, model_config: ModelConfig) -> None:
        _check_build(model_config)
        text = _text_config(model_config)
        cfg = text.pretrained_config
        core = KimiK31ProNvfp4Sm107Helix8Core(text)
        # The LM head (vocab-parallel) and the worker run over the helix ranks as TP ranks; the
        # config keeps the helix mapping, which the executor's helix bookkeeping reads.
        text._frozen = False
        text.mapping = core.tp_mapping
        super().__init__(core, config=text, hidden_size=cfg.hidden_size, vocab_size=cfg.vocab_size)
        text.mapping = core.cp_mapping
        text._frozen = True
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
            core.tp_mapping.mapping_for_full_rows(),
            should_use_separate_draft_kv_cache(self.spec_config),
        )
        self.spec_worker.set_draft_model(self.draft_model)
        self.spec_worker.set_target_model(self)

    @classmethod
    def get_model_defaults(cls, llm_args: Any) -> dict:
        """Return the engine defaults this target needs: 64-token KV pages, block reuse opt-in."""
        return {"kv_cache_config": {"enable_block_reuse": False, "tokens_per_block": 64}}

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

    @classmethod
    def select_kv_cache_manager_cls(
        cls,
        model_config: Any,
        kv_cache_config: Any,
        is_disagg: bool = False,
        cache_transceiver_config: Any = None,
    ) -> Any:
        """Return the family KDA + DSA cache manager class, after the family's config checks."""
        selected = select_rubin_kv_cache_manager_cls(
            model_config,
            kv_cache_config,
            is_disagg=is_disagg,
            cache_transceiver_config=cache_transceiver_config,
        )
        assert selected is not None, "this target runs the family KDA + DSA cache manager only"
        return selected

    @staticmethod
    def open_checkpoint_weights(checkpoint_dir: str, mapping: Any = None) -> Any:
        """Open the checkpoint lazily (weights.py)."""
        return _weights.open_checkpoint_weights(checkpoint_dir)

    def attention_backend_for(self, backend_name: str, sparse_params: Any) -> Any:
        """Return the family DSA backend (a class fact: built shells and bare ones agree)."""
        return rubin_attention_backend_for(backend_name, sparse_params, _DSA_BACKEND)

    def validate_request(self, request: Any) -> None:
        """Refuse, before it is scheduled, media on a request that is not generation-only: helix8
        runs no vision tower and no context step. A generation-only request arrives with its
        context server's KV, its images already encoded there; that request fails alone, where the
        forward would fail the batch."""
        if getattr(request, "py_multimodal_data", None) and not getattr(
            request, "is_generation_only_request", False
        ):
            raise ValueError(
                f"Image input is not supported: {_TARGET} runs no vision tower (a disaggregated "
                "generation server: its context server encodes the images)"
            )

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

        ``position_ids`` reaches the worker in the engine's ``[1, T]`` shape.
        """
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
            self.spec_worker.target_head_rows(hidden, attn_metadata, spec_metadata),
            self.lm_head,
            self.logits_processor,
            attn_metadata,
            spec_metadata,
            self.draft_model,
        )
        out = self.spec_worker(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden,
            logits=logits,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=self.draft_model,
            resource_manager=resource_manager,
        )
        taps = self.model._rt.taps
        if taps is not None:
            # draft_tap_stream: a step whose drafter read no capture still joins the tap stream.
            taps.join()
        return out

    def load_weights(self, weights: Any, *args: Any, **kwargs: Any) -> None:
        """Fill the parameters from the target checkpoint (weights.py)."""
        _weights.load(self, weights)

    def load_draft_weights(self, weights: Any, weight_mapper: Any = None) -> None:
        """Fill the drafter from its checkpoint (weights.py)."""
        _weights.load_draft(self, weights, weight_mapper)

    def set_guided_decoder(self, guided_decoder) -> bool:
        """Attach the grammar decoder to the DFlash target-token verification."""
        return self.spec_worker.set_guided_decoder(guided_decoder)

    def prepare_disagg_generation_request(self, request: Any) -> None:
        """Discard cached DFlash2 proposals for the newly received request's slot."""
        invalidate_dflash2_proposal_slot(self.spec_worker, request.py_seq_slot)

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


def drafter_line(role: str) -> str:
    """The startup drafter-switch line of ``role``: its ``draft_*`` constants as the constants line
    states them (a ``_DETAILS`` key inside its parent's value), how many of the on / off ones
    (details included) are on, and ``fp``, the first 12 hex digits of the SHA-256 of the listed
    items, which tells two drafter configurations apart in one field."""
    table = {k: v for k, v in _CONSTANTS[role].items() if k.startswith("draft_")}
    items = []
    for key in sorted(table):
        if _DETAILS.get(key) in table:
            continue
        extra = [f"{k}={table[k]}" for k in sorted(table) if _DETAILS.get(k) == key]
        items.append(f"{key}={table[key]}" + (f"({','.join(extra)})" if extra else ""))
    flags = [v for v in table.values() if isinstance(v, bool)]
    fp = hashlib.sha256(", ".join(items).encode()).hexdigest()[:12]
    return (
        f"Kimi K3.1 Rubin DFlash2 drafter switches: {_TARGET}/{role} fp={fp} "
        f"on={sum(flags)}/{len(flags)}: " + ", ".join(items)
    )


#: The lever variables the startup line states: read in C++ (the context servers' MoE routing
#: kernels) and by the engine's KV cache estimate.
_LEVER_ENV_LOGGED = ("TLLM_ROUTING_DISABLE_COOP_LAUNCH", "TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET")


def log_constants(
    role: str, *, draft_context_kv_only: bool | None = None, lever_env: dict | None = None
) -> None:
    """Log this target's launch constants for ``role`` in one line, its drafter switches in a
    second, then the lever variables.

    From ``lever_env`` (what routing read): the values of the lever variables the target cannot set
    itself.
    """
    logger.info(constants_line(role, draft_context_kv_only=draft_context_kv_only))
    logger.info(drafter_line(role))
    if lever_env is None:
        return
    items = ", ".join(f"{k}={lever_env.get(k, 'unset')}" for k in _LEVER_ENV_LOGGED)
    logger.info(f"Kimi K3.1 Rubin lever env: {_TARGET}/{role}: {items}")
