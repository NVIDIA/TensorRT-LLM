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
import math
import weakref
from types import MappingProxyType
from typing import Any, Callable, NamedTuple, Optional

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

from ..ops.activation.sigmoid_gate_mul import k31_sigmoid_gate_mul
from ..ops.activation.situ_and_mul import situ_and_mul
from ..ops.attention.convert_req_index_to_global import convert_req_index_to_global
from ..ops.attention.dsa_index_key_norm import index_key_norm
from ..ops.attention.dsa_prefill_share import disagg_role
from ..ops.attention.fp8_paged_mqa_logits import fp8_paged_mqa_logits
from ..ops.attention.fused_cat_fp8 import fused_cat_fp8
from ..ops.attention.indexer_k_cache_scatter_op import indexer_k_cache_scatter_op
from ..ops.attention.indexer_topk_decode import indexer_topk_decode
from ..ops.attention.rubin_dsa_backend import rubin_dsa_backend_cls
from ..ops.attention.rubin_dsa_topk_remap import rubin_dsa_topk_remap
from ..ops.attention.rubin_fp8_paged_mqa_logits import rubin_fp8_paged_mqa_logits
from ..ops.attention.rubin_gvr_topk_decode import rubin_gvr_topk_decode
from ..ops.attention.rubin_idx_weights import rubin_idx_weights
from ..ops.attention.rubin_merge_helix_topk import rubin_merge_helix_topk
from ..ops.attention.rubin_pad_heads import rubin_pad_heads
from ..ops.comm.allgather_module import allgather_module
from ..ops.comm.allreduce_module import allreduce_module
from ..ops.comm.rubin_comm import rubin_all_reduce, rubin_latent_ar_workspace
from ..ops.comm.rubin_lamport_push import rubin_lamport_push
from ..ops.comm.rubin_latent_ar_norm import rubin_latent_ar_norm_row
from ..ops.comm.rubin_push_allreduce import rubin_lamport_push_workspace
from ..ops.gemm.rubin_l2_prefetch import rubin_l2_prefetch
from ..ops.gemm.rubin_linear import rubin_linear
from ..ops.gemm.rubin_mxfp8_gemm import rubin_mxfp8_gemm
from ..ops.gemm.rubin_mxfp8_linear import rubin_mxfp8_linear
from ..ops.gemm.rubin_upcast_fp32 import rubin_upcast_fp32
from ..ops.moe.dsv3_router_gemm_op import dsv3_router_gemm_op
from ..ops.moe.rubin_fc12 import rubin_fc12, rubin_fc12_place_per_die
from ..ops.moe.rubin_moe_residual_add import rubin_moe_residual_add
from ..ops.moe.rubin_moe_sort import rubin_moe_sort
from ..ops.moe.rubin_route_tail import rubin_route_tail
from ..ops.moe.rubin_router_logits import rubin_router_logits
from ..ops.norm.attn_res_add_rmsnorm_fwd import attn_res_add_rmsnorm_fwd
from ..ops.norm.attn_res_fwd import attn_res_fwd
from ..ops.norm.attn_res_rmsnorm_fwd import attn_res_rmsnorm_fwd
from ..ops.norm.rubin_attn_res_add_rmsnorm_persistent import rubin_attn_res_add_rmsnorm_persistent
from ..ops.norm.rubin_attn_res_add_rmsnorm_persistent_x import (
    persistent_x_available,
    rubin_attn_res_add_rmsnorm_persistent_x,
)
from ..ops.norm.rubin_attn_res_lamport_add_rmsnorm import rubin_attn_res_lamport_add_rmsnorm
from ..ops.norm.rubin_attn_res_moe_add_rmsnorm import rubin_attn_res_moe_add_rmsnorm
from ..ops.norm.rubin_attn_res_rmsnorm import rubin_attn_res_add_rmsnorm, rubin_attn_res_rmsnorm
from ..ops.spec.dflash2_block_verify import dflash2_block_verify_seeded
from ..ops.spec.dflash2_global_top32 import k31_dflash2_merge_top32, k31_dflash2_shard_top32
from ..ops.spec.dflash2_greedy_verify import dflash2_greedy_verify
from ..ops.spec.dflash2_grouped_conv import dflash2_grouped_conv
from ..ops.spec.dflash2_proposal_load import k31_dflash2_proposal_load
from ..ops.spec.dflash2_sampling import DFlash2ProposalStore, dflash2_sampled_step
from ..ops.spec.dflash2_selector_draft import k31_dflash2_selector_draft
from ..ops.spec.dflash2_selector_greedy_path import k31_dflash2_selector_greedy_path
from ..ops.spec.draft_context_cache import DRAFT_FP8_KV_SCALE
from ..ops.spec.flashinfer_compute_probs_from_logits import flashinfer_compute_probs_from_logits
from ..ops.spec.flashinfer_sample_from_logits import flashinfer_sample_from_logits
from ..ops.spec.fused_compute_probs_from_logits import fused_compute_probs_from_logits
from ..ops.spec.fused_sample_from_logits import fused_sample_from_logits
from ..ops.spec.mla_norm_rope import dflash2_mla_block_rope, dflash2_mla_ctx_norm_rope
from ..ops.spec.rubin_dflash2_worker import (
    RubinDFlash2WorkerMixin,
    agree_draft_tokens,
    rubin_draft_config_kwargs,
)
from ..ops.spec.strict_accept import strict_accept
from ..ops.spec.unified_draft_cache_worker import UnifiedDraftCacheWorkerMixin
from ..ops.spec.vocab_parallel_argmax import vocab_parallel_argmax
from ..ops.spec.vocab_parallel_sample import vocab_parallel_sample
from ..ops.ssm.kda_decode import kda_decode as kda_decode_op
from ..ops.ssm.rubin_kda_dsa_cache_manager import bind_kv_cache_args, rubin_runtime_hooks
from ..ops.ssm.rubin_kda_metadata import RubinKdaMetadata
from ..ops.ssm.rubin_kda_mtp_verify import rubin_kda_mtp_verify
from . import weights as _weights
from .ops.attention.rubin_dsa_key_shard import (
    rubin_dsa_key_shard_merge,
    rubin_dsa_key_shard_pack,
    rubin_dsa_key_shard_ws,
)
from .ops.attention.rubin_helix_zero_or_empty_rows import rubin_helix_zero_or_empty_rows
from .ops.attention.rubin_topk_decode_cute import rubin_topk_decode_cute
from .ops.spec.rubin_dflash2_helix_combine import rubin_dflash2_helix_combine
from .ops.spec.rubin_dflash2_window_plan import plan_buffers, rubin_dflash2_window_plan
from .ops.spec.rubin_dflash2_window_triton import (
    rubin_dflash2_window_triton,
    window_vec_io_aligned,
    window_workspace,
)

# Identity. The server role is the one the runtime publishes (``ops/attention/dsa_prefill_share``),
# fixed at construction.
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
# The PDL mix after a pushed o_proj partial finishes its all-reduce before the grid wait
# (oproj_ar_early_poll). The attention-residual epilogue levers act on the persistent kernel's
# rows only (above attn_res_pdl_max_m) and are on: attn_res_mx_moe (a KDA layer's mix folds the
# MoE residual update and stores the MXFP8 form the QKVG GEMM multiplies), attn_res_moe_fold
# (every other MoE-fed mix and the output mix fold the update), attn_res_grid_cap (the persistent
# mixes launch min(SMs, rows) CTAs); dsa_idx_logits_pdl launches the decode indexer logits with PDL.
# A key-shard merge row whose certificate fails is re-selected from every rank's local logits, or
# from the failing ranks' logits and the other ranks' candidates (dsa_key_shard_failing_fallback:
# the same rows, fewer remote reads). The drafter window levers are on: draft_window_vec_io moves
# the window kernels' rows as 16-byte vectors where every stride allows (the same bytes, the same
# arithmetic); draft_window_pdl launches the window's tail as a PDL dependent of its body, waiting
# for the body only before its first read of the body's partials.
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
                "attn_res_pdl": 2,
                "attn_res_pdl_max_m": 128,
                "attn_res_topology": 1,
                "draft_block_rope": "dflash2_mla_block_rope",
                "draft_ctx_latent": "precompute_context_kv",
                "draft_history_dtype": "fp8",
                "draft_mla_norm_rope": "mla_norm_rope",
                "draft_window_pdl": True,
                "draft_window_vec_io": True,
                "dsa_fork_q": True,
                "dsa_helix_fused_meta": True,
                "dsa_helix_key_shard": True,
                "dsa_helix_mask_fold": True,
                "dsa_idx_fork": True,
                "dsa_idx_logits_pdl": True,
                "dsa_key_shard_failing_fallback": True,
                "dsa_single_atom": True,
                "fc12_mma_n": 256,
                "fc12_prezero": True,
                "fuse_moe_residual": True,
                "index_k_dtype": "fp8",
                "index_k_evict_first": True,
                "indexer_logits_elem_budget": 536870912,
                "kda_l2_prefetch": True,
                "kda_oproj_quant_k32": True,
                "kda_qkvg_fused": True,
                "kda_replay_seed_stacked": True,
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
                "moe_backend": "cutedsl_fc12",
                "moe_latent_down_stream": True,
                "moe_shared_stream_overlap": True,
                "moe_tp_ep": "tp2_ep4",
                "mxfp8_gemm": "rubin_dense_blockscaled_gemm",
                "oproj_ar_early_poll": True,
                "route_tail": 1,
                "route_tail_bands": "helix8",
                "router_bf16": True,
                "router_exact": "kda",
                "router_first": True,
                "router_first_wide": True,
                "shared_gate_up_fused": True,
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
        "latent_ar_oneshot_max_bytes": "latent_ar_own_ws",
        "latent_ar_own_ws_max_m": "latent_ar_own_ws",
        "latent_ar_row_max_m": "latent_ar_row",
        "router_first_wide": "router_first",
    }
)


_CACHE_ARGS = MappingProxyType(
    {
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

# The FC12 kernel's launch levers on this layout's decode steps (the MoE block's launches; the
# core places the expert weights per die for the die schedule): MMA N (None: the tuner picks 128
# or 256), how the die schedule's claim runs ("uniform": the whole scheduler warp claims, which
# keeps the TMA-B and MMA warps on the uniform datapath; same tiles as lane 0 alone), and how many
# K-blocks of the B / SFB weight stream the uniform claim kernel prefetches into L2 ahead of its
# loads (0: none; the same loads, so the same result). MMA N is read from the logged constant
# table, so that the log states what runs.
_FC12_MMA_N = _CONSTANTS["generation"]["fc12_mma_n"]
_FC12_DIE_CLAIM = "uniform"
_FC12_L2_PREFETCH = 2


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
# (``moe_norm``, ``moe_add``).
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
    moe_norm: Callable  # (prefix, moe_out, bank, mw) -> (prefix + bf16(routed + shared), normed)
    reduce_norm: Callable  # (prefix, partial, bank, mw) -> (prefix, normed)
    attn_res_mix: Callable  # (prefix, bank, mw) -> the pre-norm mixture (a drafter tap)
    rms_norm: Callable  # (x, weight) -> normed
    moe_add: Callable  # (prefix, moe_out) -> prefix + bf16(routed + shared)
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
    when their lever is off). The PDL kernel's rows keep today's calls.
    """
    pdl_max_m = c["attn_res_pdl_max_m"] if c.get("attn_res_pdl") == 2 else 0
    persistent = c["attn_res_topology"] == 1
    fused_max = _FUSED_MAX_ROWS[c["attn_res_topology"]]
    early = bool(c.get("attn_res_early_trigger", False))
    fuse_moe_residual = bool(c.get("fuse_moe_residual", False))
    mx_moe, moe_fold, grid_cap = _x_levers(c)
    x_ops = mx_moe or moe_fold or grid_cap

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

    def moe_add(prefix: torch.Tensor, m: _MoeOut) -> torch.Tensor:
        """``bf16(prefix + bf16(routed + shared))``: two round-to-nearest adds either way."""
        if fuse_moe_residual:
            return rubin_moe_residual_add(m.routed, m.shared, prefix)
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
                early_trigger=early,
            )
        updated = moe_add(prefix, m)
        return updated, attn_res_norm(updated, bank, mw)

    def moe_fold_norm(prefix: torch.Tensor, m: _MoeOut, bank: torch.Tensor, mw: _MixW):
        """``moe_norm`` with the MoE residual update inside the persistent mix where it serves the
        rows (one kernel instead of the update's and the mix's)."""
        if use_x(prefix.shape[0], bank.shape[0]):
            r = x_mix(prefix, m.routed, m.shared, bank, mw)
            return r.updated, r.output
        return moe_norm(prefix, m, bank, mw)

    def moe_norm_mx(prefix: torch.Tensor, m: _MoeOut, bank: torch.Tensor, mw: _MixW):
        """``moe_fold_norm`` with the normed rows' MXFP8 form where the persistent mix serves the
        rows: ``(prefix, (normed, codes, scales))``, the codes and R128c4 scales bit for bit
        ``mxfp8_quantize(normed, True, 32)``; else ``moe_norm``'s ``(prefix, normed)``."""
        if use_x(prefix.shape[0], bank.shape[0]):
            r = x_mix(prefix, m.routed, m.shared, bank, mw, mx=True)
            return r.updated, (r.output, r.codes, r.scales)
        return moe_norm(prefix, m, bank, mw)

    return dict(
        rms_norm=rms_norm,
        attn_res_mix=attn_res_mix,
        attn_res_norm=attn_res_norm,
        add_norm=add_norm,
        moe_add=moe_add,
        moe_norm=moe_norm,
        moe_fold_norm=moe_fold_norm if moe_fold else None,
        moe_norm_mx=moe_norm_mx if mx_moe else None,
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
    its Lamport buffers (a copy kernel: no GEMM here has the push epilogue) and the mix after it
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
                False,  # finish_ag: the call carries no all-gather rows
                early_trigger=early,
            )
        return add_norm(prefix, reduce(partial), bank, mw)

    return dict(reduce=reduce, reduce_norm=reduce_norm)


def _key_shard_ws(mapping: Any, md: Any) -> Any:
    """The key-shard merge's workspace over ``mapping`` (the CP group as TP).

    Sized as the step metadata checks it: ``md.key_shard_rows_max`` rows (max batch x (max draft
    + 1)) of ``md.key_shard_max_local_len`` local keys (the cache manager's pages x 64).
    Collective: every rank's first decode step builds it at the same point, eagerly, before any
    CUDA graph capture.
    """
    assert not torch.cuda.is_current_stream_capturing(), "the key-shard workspace is built eagerly"
    return rubin_dsa_key_shard_ws(
        mapping, md.key_shard_rows_max, _K31_DSA["index_topk"], md.key_shard_max_local_len
    )


def _build_runtime(core: Any) -> _Rt:
    """Build the runtime resources: streams, events, communication closures, the RoPE table.

    Collective over the helix group as TP ranks (the all-reduce workspaces): every rank runs it at
    the same point. The key-shard merge's workspace takes the step metadata's sizes, so the first
    decode step builds it (``_key_shard_ws``); ``DraftModel.bind_runtime`` sets ``rt.draft``.
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
#   ``quantize_output``), so the verify blocks run no separate quantization kernel.
#
# Generation lever: up to 16 verify rows, the L2 prefetch of the next KDA layer's weights on the aux
# stream while the recurrence runs.
#
# A decode or verify block takes ``x`` as the BF16 rows or, with ``attn_res_mx_moe`` where the
# persistent mix serves the rows, as ``(rows, codes, scales)``: the rows with the MXFP8 form the
# mix stored (E4M3 codes and R128c4 scales, bit for bit the projection's own quantization of the
# rows); the fused projection then multiplies those codes and skips its quantization.
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
_KDA_LOWER_BOUND = _K31_KDA["gate_lower_bound"]
_KDA_EPS = _K31["rms_norm_eps"]
_KDA_SCALE = _KDA_D**-0.5
# Read from the logged constant table, so that the log states what runs.
_KDA_PREFETCH_ROWS = 16 if _CONSTANTS["generation"]["kda_l2_prefetch"] else 0
_KDA_OPROJ_K32 = _CONSTANTS["generation"]["kda_oproj_quant_k32"]
_KDA_OPROJ_K32_LINE = (
    "Kimi K3.1 Rubin KDA: the verify epilogue writes o_proj's MXFP8 input (32-value scales)"
)


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
    step's replay.
    """
    x, mx = _kda_split(x)
    if _KDA_OPROJ_K32:
        codes, scales = _kda_verify_core(L, x, st, rt, mx_out=True, mx=mx)
        logger.info_once(_KDA_OPROJ_K32_LINE, key=_KDA_OPROJ_K32_LINE)
        return rubin_mxfp8_gemm(codes, L.o_w, scales, L.o_s)
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
    (``quantize_output``): what ``rubin_mxfp8_linear`` makes of the core. ``mx``: the MXFP8 form of
    ``x`` when the mix stored it."""
    rows = x.shape[0]
    lc = st.cache.mamba_layer_cache(L.layer)
    y, g = _kda_project(L, x, mx)
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
    # The entry fixes the packed token layout and the fused output norm. g is the finished gate
    # input (f_b ran above); o_proj takes the BF16 output, or with mx_out its catalog-form MXFP8
    # output (on the kernel the BF16 output takes).
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
# The sparse MLA call's head count: TRTLLM-Gen's sparse MLA tiles a head group above 64 heads in
# Q64 tiles, which must divide it, so the 96 heads run padded to 128 with zero queries
# (``rubin_pad_heads``) and the padding heads' outputs are dropped. The padding buffers are
# cleared in 64-element rows.
_MLA_PAD_HEADS = 128
_MLA_PAD_ROW = 64
# AttentionInputType of the attention op's generation call.
_MLA_GENERATION = 2
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
# The radix Top-K's split workspace: blocks per row.
_IDX_RADIX_BLOCKS = 10
# The key-shard logits kernel's PDL launch (logged constant): only the launch and the waits move.
_IDX_LOGITS_PDL = _CONSTANTS["generation"]["dsa_idx_logits_pdl"]
#: Logged once per process by the first CUDA graph decode step: the indexer's keys and weights ran
#: on the idx stream, beside ``[q_a | kv_a]`` (dsa_idx_fork).
_IDX_FORK_LINE = (
    "Kimi K3.1 Rubin DSA: indexer key / weights prologue on its own stream beside [q_a | kv_a]"
)
#: Logged once per process by the first key-sharded merge under dsa_key_shard_failing_fallback.
_KS_FAILING_LINE = (
    "Kimi K3.1 Rubin DSA key shard: a re-selected row reads only the failing ranks' local logits"
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
# The attention op's generation call at 128 heads.
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
# The RoPE kernel at the attention op's 128 heads.
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


def _idx_keys(L: _MlaW, x: torch.Tensor) -> tuple:
    """The indexer's work on ``x`` alone: FP8 keys ``[rows, 128]`` with their FP32 scales, and the
    FP32 logits weights before the query scales ``[rows, 32]``, on every row of ``x``.

    Position-free: each key is quantized as its two halves; nothing rotates. On graph steps
    ``mla_generate`` runs it on ``rt.idx_stream`` (dsa_idx_fork).
    """
    ik, w = rubin_linear(rubin_upcast_fp32(x), L.idx_wk_wp, True).split([_IDX_DIM, _IDX_HEADS], -1)
    ik = ik.to(x.dtype)
    ik = index_key_norm(ik, L.idx_k_norm_w, L.idx_k_norm_b, _IDX_K_NORM_EPS)
    ik_a, ik_b = ik.split([_IDX_HALF, _IDX_DIM - _IDX_HALF], -1)
    k8, ks = fused_cat_fp8(ik_a, ik_b)
    return k8, ks, w


def _idx_queries(L: _MlaW, qr: torch.Tensor) -> tuple:
    """The indexer's FP8 queries ``[rows, 32, 128]`` and their per-head FP32 scales ``[rows, 32]``.

    Position-free: each head is quantized as its two halves; nothing rotates.
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
    """This rank's share of the generation rows' global Top-K, and the merge's combine mask.

    The rows are ``[rows, 2048]`` local token indices, kept first, then -1. Key-sharded where the
    step allows it; otherwise the fallback (eager steps): the local logits, the local Top-K (radix
    with the per-token bounds where the step has them, the rows a rank owns no KV for emptied),
    every rank's candidate scores all-gathered over the CP group and merged, the kept counts.
    Dense Top-K rows where the step skips the indexer.
    """
    d = st.dsa
    n = st.num_tokens
    if d.skip_gen:
        return d.topk_dense[:n], None
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
            True,
            ws.local_logits[:n, : d.ks_local_len],
            # Only when on: with it off the call is today's.
            **({"pdl": True} if _IDX_LOGITS_PDL else {}),
        )
        local = ws.local_topk[:n]
        rubin_topk_decode_cute(logits, d.ks_row_lens, local, ws.local_k, 1)
        rubin_dsa_key_shard_pack(local, d.ks_row_lens, ws)
        bounds = d.row_bounds if d.counts_out is not None else None
        mask = None if bounds is None else torch.empty(n, dtype=torch.bool, device=topk.device)
        logger.info_once(_KS_FAILING_LINE, key=_KS_FAILING_LINE)
        rubin_dsa_key_shard_merge(topk, d.counts_out, bounds, mask, ws, failing_only=True)
        return topk, mask
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
        rubin_gvr_topk_decode(logits, d.gen_kv_lens, topk, d.fb_topk_next_n, d.max_len)
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
    return topk, None


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
        **_MLA_ROPE_GEN_PAD128,
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
        topk, mask = _idx_decode(i, q8, w, st, rt)
        indices = _idx_remap(i, topk, d.req_idx_gen, d.block_table_gen, st, rt)
        q_ready.wait()
        if not torch.cuda.is_current_stream_capturing():
            q.record_stream(main)
            fused_q.record_stream(main)
            g.record_stream(main)
    else:
        fused_q, q_pe = _mla_query(L, i, q, latent, bufs, st, rt)
        topk, mask = _idx_decode(i, q8, w, st, rt)
        indices = _idx_remap(i, topk, d.req_idx_gen, d.block_table_gen, st, rt)
    cu_q, cu_kv, counter, bmm1, bmm2, quant_q = bufs
    stats = torch.empty(rows, _MLA_PAD_HEADS, 2, dtype=torch.float32, device=dev)
    lat = thop_attention(
        q=fused_q.view(rows, -1),
        output=q.new_empty(rows, _MLA_PAD_HEADS * _MLA_KV_LORA),
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
        **{**d.sparse_gen, **_MLA_ATTN_PAD128},
    )
    if mask is None:
        mask = _mla_helix_mask(rows, st)
    partial_v = lat.new_empty(rows, _MLA_HEADS, _MLA_V)
    _mla_expand(L.v_b_t, lat.view(rows, _MLA_PAD_HEADS, _MLA_KV_LORA)[:, :_MLA_HEADS], partial_v)
    out = rubin_dflash2_helix_combine(
        partial_v,
        stats[:, :_MLA_HEADS].contiguous(),
        rt.cp_mapping,
        _MLA_HEADS,
        _MLA_V,
        mask,
    )
    return out, gate


def _mla_output(
    L: _MlaW, x: torch.Tensor, out: torch.Tensor, n: int, gate: Optional[tuple] = None
) -> torch.Tensor:
    """Zero the padding rows, gate with ``sigmoid(g_proj(x))`` and project: the o_proj partial.

    ``gate``: ``(g_proj(x), the event recorded after it)`` from the fork stream (graph steps); the
    gate waits for that event right before it reads g. None: ``g_proj(x)`` runs here.
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


def mla_generate(L: _MlaW, i: int, x: torch.Tensor, st: _Step, rt: _Rt) -> torch.Tensor:
    """Layer ``i``'s MLA + DSA over the rows of a decode step; ``[rows, hidden]``.

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
        torch.empty(n, _MLA_PAD_HEADS, _MLA_LATENT, dtype=torch.uint8, device=x.device),
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
    return _mla_output(L, x, out, n, gate)


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
# shared stream joins. Same kernels, same bits.
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
# The expert kernels' activation (ActivationType.SiTu).
_MOE_ACT_SITU = 9
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
_moe_require("generation", "route_tail_bands", "helix8")
_moe_require("generation", "latent_bf16_gemm", "cublas_linear")
if _MOE_GEN["latent_ar_own_ws"]:
    _moe_require("generation", "latent_ar_cute", 1)
    _moe_require("generation", "latent_ar_row", True)

# Decode-step levers as row bounds (0: off). route_tail: top-16 + moe_sort in one cluster launch
# (approx candidates, exact picks), up to its 256 rows; router_exact: KDA layers, up to 16 rows;
# router_first: the router GEMM goes before the forked latent_down; own workspace: the latent
# all-reduce's row kernel with an early PDL trigger. router_first_wide: the router GEMM also goes
# first from 97 rows (helix8 b24 k5: 144) up to the routing tail's bound;
# 65 to 96 rows keep latent_down first (b16 k5: 96 rows measured no faster with the router first).
_MOE_ROUTE_TAIL_MAX_M = 256 if _MOE_GEN["route_tail"] == 1 else 0
_MOE_ROUTE_BANDS = _MOE_GEN["route_tail_bands"]
_MOE_ROUTER_EXACT_MAX_M = 16 if _MOE_GEN["router_exact"] == "kda" else 0
_MOE_ROUTER_FIRST_MAX_M = 64 if _MOE_GEN["router_first"] else 0
#: router_first_wide's rows ``(first, last)``; ``(1, 0)``: none.
_MOE_ROUTER_FIRST_WIDE = (97, _MOE_ROUTE_TAIL_MAX_M) if _MOE_GEN["router_first_wide"] else (1, 0)
_MOE_OWN_WS_MAX_M = _MOE_GEN["latent_ar_own_ws_max_m"] if _MOE_GEN["latent_ar_own_ws"] else 0
_MOE_OWN_WS_MAX_BYTES = _MOE_GEN["latent_ar_oneshot_max_bytes"]
_MOE_FC12_PREZERO = bool(_MOE_GEN["fc12_prezero"])
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


def _moe_router(L: _MoeW, x: torch.Tensor) -> torch.Tensor:
    """FP32 logits of the BF16 router GEMM (FP32 accumulation)."""
    return dsv3_router_gemm_op(x, L.router.t())


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
    of every tile count moe_sort can produce at the 128-row tile (two lines per phase: the die
    schedule's per-die counters).
    """
    pairs = rows * _MOE_TOP_K
    filled = min(num_local, pairs)
    tiles = filled + (pairs - filled) // _MOE_TILE
    counter_words = -(-tiles // 32) * 32 + 2 * 2 * 32
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
        _MOE_EP,
        _MOE_ALLTOALL,
        _MOE_SF_VEC,
        *_MOE_SITU,
        counters,
        mma_n=_FC12_MMA_N,
        die_claim=_FC12_DIE_CLAIM,
        l2_prefetch=_FC12_L2_PREFETCH,
    )
    return output


def _moe_shared(L: _MoeW, x: torch.Tensor) -> torch.Tensor:
    """The shared experts at this rank's share: gate and up in one GEMM, then SiTU."""
    h = situ_and_mul(rubin_mxfp8_linear(x, L.shared_gu_w, L.shared_gu_s), *_MOE_SITU)
    return rubin_mxfp8_linear(h, L.shared_dn_w, L.shared_dn_s)


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
    latent_in = _moe_latent_gemm(x, L.latent_down)
    ids, weights, tables = _moe_route(L, logits)
    x4, sf = _moe_quantize(L, latent_in)
    routed = _moe_fc12(L, x4, sf, ids, weights, tables, None)
    shared = allreduce_module(_moe_shared(L, x), c.all_reduce, _MOE_AR_NONE, None, None)
    latent = allreduce_module(routed, c.all_reduce, _MOE_AR_RMS_NORM, L.latent_norm, _MOE_EPS)
    return _MoeOut(_moe_latent_gemm(latent, L.latent_up), shared)


def _moe_tp_gen_streams(L: _MoeW, x: torch.Tensor, st: _Step, rt: _Rt) -> _MoeOut:
    """A decode step of a CUDA graph with the stream plan; the same bits as in order."""
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
        latent_in = _moe_latent_gemm(x, L.latent_down)
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
        )
        up = _moe_latent_gemm(latent, L.latent_up)
        c.shared_join.wait()
        return _MoeOut(up, shared)
    # Both all-reduces use the TP group's workspace: the shared one goes first.
    c.shared_join.wait()
    latent = allreduce_module(routed, c.all_reduce, _MOE_AR_RMS_NORM, L.latent_norm, _MOE_EPS)
    return _MoeOut(_moe_latent_gemm(latent, L.latent_up), shared)


def latent_moe(L: _MoeW, i: int, x: torch.Tensor, st: _Step, rt: _Rt) -> _MoeOut:
    """Run layer ``i``'s latent MoE (routed and shared experts) over ``x``; return both parts."""
    if st.cuda_graph:  # the graph runner's warmup and capture
        return _moe_tp_gen_streams(L, x, st, rt)
    return _moe_tp_gen(L, x, st, rt)


def dense_mlp(L0: _DenseW, x: torch.Tensor, st: _Step, rt: _Rt) -> torch.Tensor:
    """Run layer 0's dense SiTU MLP over ``x``; return ``[rows, hidden]``."""
    h = situ_and_mul(cublas_mm(x, L0.gu.t()), *_MOE_SITU)
    out = cublas_mm(h, L0.dn.t())
    ar = rt.moe_comm.dense_all_reduce
    return out if ar is None else allreduce_module(out, ar, _MOE_AR_NONE, None, None)


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
# persistent scratch the window entry reads, never into the pool.
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
#: Logged once per process by the first drafter step that takes its candidates from the shard.
_DRAFT_SHARD_LINE = (
    "Kimi K3.1 Rubin drafter: selector candidates from the vocab shard (shard top-32 in id order, "
    "pairs all-gathered and merged)"
)
#: Logged once per process by the first CUDA graph generation step that drafts: the drafter's
#: projection of the step's captured target rows (the fc GEMM and the hidden norm) was issued on
#: the aux stream before the acceptance, which it does not read.
_DRAFT_FC_EARLY_LINE = (
    "Kimi K3.1 Rubin DFlash2: drafter fc projection issued on the aux stream before the acceptance"
)


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
        views.append(
            _DraftLayerW(
                in_norm=w[f"d_l{j}_in_norm"],
                post_norm=w[f"d_l{j}_post_norm"],
                q_a=w[f"d_l{j}_q_a"],
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
    """The DFlash2 drafter the engine's worker drives (helix8: 96 heads, page-sharded history).

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
        self._rt: Any = None
        # The projection issued early this step: (rows key, projection, its event), or None.
        self._early_projection: Optional[tuple] = None
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
        )
        # Also an attribute: the engine counts a context server's drafter tables off the module.
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
        key, projected, ready = early
        if key != _rows_key(hidden_states):
            raise RuntimeError(
                "DFlash2: the drafter projection issued before the acceptance was of other rows "
                f"({key[1]}) than the ones the preparation projects ({tuple(hidden_states.shape)})"
            )
        ready.wait()
        return projected

    def _project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The fc GEMM and the hidden norm on the current stream."""
        fc = self._w("d_fc")
        return flashinfer_rmsnorm(
            rubin_linear(hidden_states.to(fc.dtype), fc, False),
            self._w("d_hidden_norm"),
            float(self.config.rms_norm_eps),
        )

    def issue_projection(self, hidden_states: torch.Tensor) -> None:
        """Project ``hidden_states`` now, on the aux stream, for this step's preparation.

        The aux stream first waits for the current stream (everything issued so far, the target
        forward that wrote the captures included); the fc GEMM and the hidden norm then run there
        while the current stream takes the acceptance. The next ``project_target_hidden`` of the
        same rows waits for the event recorded after them and returns the result. Same kernels on
        the same rows: the same bits. The result is read on the current stream only after that
        wait, and the next step's work on the aux stream forks from the current stream. A
        projection a failed step left behind is dropped (nothing reads it).
        """
        main = torch.cuda.current_stream()
        aux = self._rt.aux_stream
        ready = self._rt.draft.fc_ready
        aux.wait_stream(main)
        with torch.cuda.stream(aux):
            projected = self._project(hidden_states)
            ready.record()
        if not torch.cuda.is_current_stream_capturing():
            projected.record_stream(main)
        self._early_projection = (_rows_key(hidden_states), projected, ready)
        logger.info_once(_DRAFT_FC_EARLY_LINE, key=_DRAFT_FC_EARLY_LINE)

    def projection_pending(self) -> bool:
        """Whether an early projection is still waiting for its preparation."""
        return self._early_projection is not None

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
    d = rt.draft
    eps, taps, groups, group = d.eps, d.taps, d.groups, d.group
    rows, block = st.rows, st.block
    heads, nope, kv_lora = DW.layers[0].k_b.shape
    latent = DW.layers[0].kv_a.shape[0]
    # Once per step: the window's body, head and block keys this rank owns, per request.
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

    h = hidden_streams
    residual = h.clone()  # the fused add + norm below updates it in place
    x = flashinfer_rmsnorm(h, DW.layers[0].in_norm, eps)
    for j, L in enumerate(DW.layers):
        if j:
            x, residual = flashinfer_fused_add_rmsnorm(h, residual, L.in_norm, eps)
        coef = rubin_linear(x, L.attn_conv_proj, False).view(rows, 2, taps, groups)
        x = dflash2_grouped_conv(x, coef[:, 0], L.attn_conv_base[0], block, group)
        qa = flashinfer_rmsnorm(rubin_linear(x, L.q_a, False), L.q_a_norm, eps)
        q = rubin_linear(qa, L.q_b, False)
        q = q.view(rows, heads, -1)
        query = q.new_empty(rows, heads, latent)
        bmm_out(q[..., :nope].transpose(0, 1), L.k_b, query[..., :kv_lora].transpose(0, 1))
        dflash2_mla_block_rope(
            q,
            query,
            rubin_linear(x, L.kv_a, False),
            L.kv_a_norm,
            eps,
            d.freqs,
            st.positions,
            block_kv,
            d.block_slots[:rows],
            d.block_offsets[:rows],
        )
        pool = st.pools[j][:, 0, 0]
        ctx = rubin_dflash2_window_triton(
            query,
            pool,
            block_kv[:, 0],
            plan,
            d.softmax_scale,
            kv_lora,
            d.kv_scale,
            d.workspace,
            stats,
            vec_io=_DRAFT_WINDOW_VEC_IO,
            pdl=_DRAFT_WINDOW_PDL,
        )
        if _DRAFT_WINDOW_VEC_IO:
            vec_layers += window_vec_io_aligned(query, pool, block_kv[:, 0], d.workspace, kv_lora)
        v_dim = L.v_b_t.shape[-1]
        partial = ctx.new_empty(rows, heads, v_dim)
        bmm_out(ctx.transpose(0, 1), L.v_b_t, partial.transpose(0, 1))
        attn = rubin_dflash2_helix_combine(partial, stats, d.mapping, heads, v_dim)
        a = allreduce_module(rubin_linear(attn, L.o, False), rt.all_reduce, _AR_NONE, None, None)
        a = dflash2_grouped_conv(a, coef[:, 1], L.attn_conv_base[1], block, group)
        x, residual = flashinfer_fused_add_rmsnorm(a, residual, L.post_norm, eps)
        coef = rubin_linear(x, L.mlp_conv_proj, False).view(rows, 2, taps, groups)
        x = dflash2_grouped_conv(x, coef[:, 0], L.mlp_conv_base[0], block, group)
        m = rubin_linear(flashinfer_silu_and_mul(rubin_linear(x, L.gate_up, False)), L.down, False)
        m = allreduce_module(m, rt.all_reduce, _AR_NONE, None, None)
        h = dflash2_grouped_conv(m, coef[:, 1], L.mlp_conv_base[1], block, group)
    h, _ = flashinfer_fused_add_rmsnorm(h, residual, DW.final_norm, eps)
    if vec_layers:
        line = _DRAFT_WINDOW_VEC_LINE.format(vec_layers, len(DW.layers))
        logger.info_once(line, key=_DRAFT_WINDOW_VEC_LINE)
    return h


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
                spec_metadata.max_num_requests * self._compute_block_size, self._ctx_page_size
            )

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
        draft_model.issue_projection(gen_hs.reshape(-1, gen_hs.shape[-1]))
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
            (accepted_tokens, num_accepted_tokens), batch_size, spec_metadata
        )

    def _dflash2_global_top_k(self, gen_logits, spec_metadata, top_k: int, full_vocab: int):
        """Per-position global top-k candidates and the tensor to score into.

        This rank's top-32; when the drafter's logits are a vocab shard, the shard's ids made
        global, its pairs put in id order (``_pairs_in_id_order``), gathered and merged: the
        full-vocab top-32's ids, values and order. The tensor to score into is FP32, as the
        full-vocab logits are.
        """
        shard = gen_logits.shape[-1]
        if shard == full_vocab:
            unary_logits, candidate_ids = k31_dflash2_shard_top32(gen_logits, top_k, 0, False)
            return candidate_ids, unary_logits, gen_logits
        assert self._draft_logits_are_sharded(gen_logits, spec_metadata) and (
            shard * self._vocab_shards == full_vocab
        ), f"draft logits of width {shard} are neither the vocab {full_vocab} nor a TP shard of it"
        combined = k31_dflash2_shard_top32(gen_logits, top_k, self._vocab_shard * shard, True)
        gathered = allgather_module(_pairs_in_id_order(combined), self.mapping, -1, None)
        unary_logits, candidate_ids = k31_dflash2_merge_top32(gathered.contiguous(), top_k)
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
        if not draft_model.has_candidate_selector:
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

        # On graph steps the drafter's projection of the captured target rows runs on the aux
        # stream beside the acceptance and its bookkeeping, which it does not read.
        early_fc = self._issue_draft_projection(
            draft_model, attn_metadata, spec_metadata, input_ids.shape[0], num_gens
        )

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
            if early_fc and draft_model.projection_pending():
                raise RuntimeError(
                    "DFlash2: the preparation did not take the drafter projection issued early"
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
# Core, the prefill and decode targets, and the shell with the engine's hooks.
#
# The core holds ``self.w`` (the weight table's parameters) and nothing else: no layer, helper or
# backend object. The two targets run the 93 layers as straight-line code over the views: the mix,
# the mixer, the mix, the MLP. The shell builds the DFlash2 drafter and its worker at construction,
# as the DS target builds its MTP worker.
# --------------------------------------------------------------------------------------------------

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
    if layout != (_WORLD_SIZE, 1, 1, _HELIX_CP, _ATTENTION_DP):
        raise ValueError(f"not {_LAYOUT}: {layout}")
    if not m.has_cp_helix():
        raise ValueError("helix8 runs helix context parallelism")
    if not m.moe_tp_ep_user_specified or (m.moe_tp_size, m.moe_ep_size) != (_MOE_TP, _MOE_EP):
        raise ValueError(
            f"MoE TP x EP {m.moe_tp_size} x {m.moe_ep_size}, not {_MOE_TP} x {_MOE_EP} set by the "
            "arguments"
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
    kernel where it serves the row count: a ``_MoeOut`` through ``moe_norm`` (``moe_fold_norm``
    with ``attn_res_moe_fold``), layer 0's whole dense output through ``add_norm``. A tapped
    layer's stream, the pre-norm mixture, goes to the drafter first, so its mix and norm run apart.
    ``mx`` (a KDA layer): with ``attn_res_mx_moe`` a ``_MoeOut`` goes through ``moe_norm_mx``, so
    ``normed`` may come as ``(rows, codes, scales)``.
    """
    moe = isinstance(pending, _MoeOut)
    if tap is None:
        if pending is None:
            return prefix, rt.attn_res_norm(prefix, bank, mw)
        if moe:
            norm = (rt.moe_norm_mx if mx else None) or rt.moe_fold_norm or rt.moe_norm
            return norm(prefix, pending, bank, mw)
        return rt.add_norm(prefix, pending, bank, mw)
    if pending is not None:
        prefix = rt.moe_add(prefix, pending) if moe else prefix + pending
    mix = rt.attn_res_mix(prefix, bank, mw)
    tap[0].maybe_capture_hidden_states(tap[1], mix, None)
    return prefix, rt.rms_norm(mix, mw.out_norm)


def _mix_out(rt: _Rt, prefix: torch.Tensor, pending: Any, bank: torch.Tensor, head: _HeadW, tap):
    """The output mix after the last layer; returns the final hidden states.

    The last MLP output joins the prefix sum first (``moe_add``), the drafter reads its tap of the
    last layer, then the mix and the final norm run. Untapped with ``attn_res_moe_fold``,
    ``moe_fold_norm`` runs the update inside the mix.
    """
    if tap is None and isinstance(pending, _MoeOut) and rt.moe_fold_norm is not None:
        return rt.moe_fold_norm(prefix, pending, bank, head.out)[1]
    if pending is not None:
        prefix = rt.moe_add(prefix, pending) if isinstance(pending, _MoeOut) else prefix + pending
    if tap is not None:
        tap[0].maybe_capture_hidden_states(tap[1], rt.attn_res_mix(prefix, bank, head.out), None)
    return rt.attn_res_norm(prefix, bank, head.out)


class KimiK31ProNvfp4Sm107Helix8Core(DecoderModel):
    """The K3.1 text model at this rank's share: the weight table, the views and the two targets."""

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__(model_config)
        self.role = _server_role()
        self.eps = float(model_config.pretrained_config.rms_norm_eps)
        # Helix CP8 for the MLA attention (and the executor); the same ranks as TP8 for the rest,
        # including every partial sum's all-reduce and the MoE. Both exist before dims(self)
        # reads them.
        self.cp_mapping = model_config.mapping
        self.tp_mapping = _helix_tp_mapping(self.cp_mapping)
        self.dims = _weights.dims(self)
        self.w = _weights.declare(self.dims)
        # Made once, never during a capture.
        self.streams = _Streams(*(torch.cuda.Stream() for _ in _Streams._fields))
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
            rubin_fc12_place_per_die(self.w, self.dims.moe_layers)
        self._layers = _layer_views(self)
        self._head = _head_view(self)
        self._rt = _build_runtime(self)
        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)
        log_constants(
            self.role,
            draft_context_kv_only=self.dims.draft_context_kv_only,
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
        """Run the layers over the generation rows (one verify block per request with DFlash2)."""
        rt, head = self._rt, self._head
        rt.dsa_pools.bind(attn_metadata.kv_cache_manager)
        if rt.ks_ws is None and attn_metadata.key_shard_world > 1:
            rt = self._rt = self.core._rt = rt._replace(
                ks_ws=_key_shard_ws(self.core.tp_mapping, attn_metadata)
            )
        st = _build_step_args(attn_metadata, spec_metadata, rt.dsa_pools, decode=True)
        h = (
            nn.functional.embedding(input_ids, head.embed)
            if inputs_embeds is None
            else inputs_embeds
        )
        kda = kda_verify if st.rows_per_gen > 1 else kda_decode
        bank = h.new_empty(head.n_bank, h.shape[0], h.shape[1])
        prefix, pending = h, None
        for i, L in enumerate(self._layers):
            # With attn_res_mx_moe a KDA layer's projection takes the MXFP8 form of its input.
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
            if L.boundary:
                prefix = rt.reduce(o)
                x = rt.attn_res_norm(prefix, bank[: L.n_bank + 1], L.mid)
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

    # The engine hooks every Kimi K3.1 target declares (``runtime.py``), with the Rubin KDA + DSA
    # cache manager (``rubin_runtime_hooks``) and this target's DSA backend.
    runtime_hooks = rubin_runtime_hooks(_DSA_BACKEND, DraftModel)

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
        """Fill the parameters from the target checkpoint (weights.py)."""
        _weights.load(self, weights)

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
