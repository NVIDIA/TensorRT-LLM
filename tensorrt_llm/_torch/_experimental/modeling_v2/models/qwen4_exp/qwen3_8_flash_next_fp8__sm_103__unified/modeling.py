# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: Qwen3.8-Flash-Next-FP8 (hf-236dfdf_orig) / sm_103 / unified.

One target for every layout `routing.py` admits for this checkpoint in one node (`_FP8_LAYOUTS`):
tp1; tp2, tep2 (attention TP2) and dep2 (attention DP) at two ranks; tp4, tep4 (attention TP4)
and dep4 (attention DP) at four. The core holds this rank's share of the weights, the derived
operands and the per-manager pool views; `PrefillTarget` (any batch with context rows) and
`DecodeTarget` (generation only) run a step over it.

The checkpoint is block FP8 (`FP8_BLOCK_SCALES`): only the routed experts are quantized (e4m3,
one scale per 128x128 weight block); everything else is bf16 (`weights.py`). Per rank:

* **Attention and Gated-DeltaNet heads** are split over attention TP `t` (1 at tp1 and under
  attention DP; `weights.rank_dims`, `pool.pool_geometry`): 24 / t q heads, 16 / t GDN key and
  48 / t value heads, and 2 / t KV heads (one, repeated on two ranks, at t = 4). The QSA indexer
  is replicated.
* **The mixer output is a partial sum** under attention TP (o_proj and out_proj read this rank's
  heads); one plain sum (`catalog/comm/allreduce`) completes it before the hyper-connection
  inject, which every rank runs on the same rows. Under attention DP and at tp1 nothing is summed.
* **Routed experts** are whole-expert windows: rank `r` of `t` holds experts
  `[512r / t, 512(r + 1) / t)` at the whole 640-row intermediate (`weights.ExpertWindow`),
  whatever the engine's `Mapping` says of MoE TP and EP, because 640 / t rows do not take the
  128-row FP8 weight block. Under attention TP the plain sum completes the window partials.
  Under attention DP each rank's rows are gathered (`catalog/comm/allgather`, ragged when the
  counts differ), run through the window and reduce-scattered back
  (`catalog/comm/reducescatter`; `_expert_sum_op`, or an all-to-all with
  `MOE_COMM_METHOD_ENV=NVLINK_ONE_SIDED`).
* **Replicated:** the router, the bf16 shared expert, the hyper-connections, PLE and the
  embedding run on this rank's rows. `lm_head` is the shell's: vocab-parallel under attention
  TP, whole under attention DP and at tp1.

There is **no** `model.language_model.norm.weight`: the stack-final mixer is the final norm.

**QSA sparse attention.** The index plane is written on every step so a request can cross
`dense_seq_len_threshold` later without rebuilding its prefix; above it the sparse core (index
scores, block top-k and expansion, sparse GQA) replaces the dense op. The raw index key is
stored before anything normalizes it: the compression applies the norm and the rotation itself.

**KV storage.** The paged K/V pool holds bf16, fp8-e4m3 under `kv_cache_config.dtype: fp8`
(`QM_FP8_KV_CACHE`) or NVFP4 under `nvfp4` (`QM_NVFP4_KV_CACHE`: e2m1 with e4m3 block scales in
a second pool, second-level scales 1.0; paged-context attention only). The QSA index and
position planes keep their own dtypes under all three.

**Breakable prefill CUDA graph.** With `prefill_cuda_graph_backend: breakable` the engine
captures the core's `forward` (the body, not the shell), pads a step up to its bucket and
replays it without calling the body. The shell therefore calls `begin_step` before the body on
every step: the asserts, the cache-manager bind, the target choice and the prefill target's host
bookkeeping live there. The graph breaks (the vision splice, the PLE block, each layer's
attention core and GDN recurrence) each write rows `[:n]` of a fixed output and zero rows `[n:]`.

**Host offload.** The PLE n-gram table lives in device-addressable host memory: one copy per
rank, per server or per node (`qwen4_exp/ple_host_table.py`).

**One-model MTP-Eagle.** With a `speculative_config` the core also holds the MTP module
(`MTPLayer`), replayed `max_draft_len` times by the spec worker; the trunk runs every generation
request's `max_draft_len + 1` verify rows.

* The draft layer is one more full-attention QSA layer, cut like a trunk one. Under attention TP
  it has a separate draft KV cache; under attention DP the trunk's manager maps it too
  (`_bind_draft_cache_manager`). Its experts are the trunk's window, completed the trunk's way.
* Under attention DP `_dp_pad_rows` pads each draft step's rows by the group's row counts, so
  every rank issues the same collectives at every draft step.
* Verify GDN rows run the CuTe cached replay (`ops/ssm/gdn_cached_replay_cute__verify`, the
  stock Triton replay's contract): it commits the history in the kernel below
  `CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE` requests and leaves it to the manager from there on.
* What a verify row writes that the runtime does not roll back (the PLE n-gram history and conv
  state, every attention layer's QSA index-K plane and the shared position plane) is staged
  during the forward and rewritten to the accepted prefix by `SpeculativeRollback`, which the
  spec worker calls after its draft loop.
* IndexShare (`index_share_for_mtp_iteration`): the draft-extend step captures each request's
  selection and the later draft steps rebuild theirs from it (`qsa_mtp_reuse_topk`).

**Images and video.** The shell runs the replicated bf16 vision tower (`vision.VisionTower`)
over the step's media and `begin_step`'s splice break writes the rows over the token embedding;
`validate_request` refuses media the tower would assert on.

Integration facts this file depends on:

1. The engine rebuilds the composite config and normalizes its text tower
   (`config_utils.load_pretrained_config`). The shell adopts that text config as
   `model_config.pretrained_config` before building the core, because the recurrent-state and
   PLE pool geometry are read from it.
2. `position_ids` arrives in the mRoPE layout (`[3, 1, T]`) even for text-only batches. The
   main rope reads all three rows; the QSA index plane takes coordinate 0 only.
"""

from __future__ import annotations

import contextlib
import math
import os
from typing import TYPE_CHECKING, NamedTuple

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._ipc_utils import can_access_peer
from tensorrt_llm._torch._experimental.modeling_v2._router_index import (
    NoTarget,
    engine_kv_cache_quant_algo,
    step_contract_enabled,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.fused_qk_norm_rope import (
    fused_qk_norm_rope,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
    QM_FP8_KV_CACHE,
    QM_NVFP4_KV_CACHE,
    thop_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.allgather import allgather
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.allreduce import (
    allreduce,
    allreduce_group_args,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.reducescatter import reducescatter
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.graph.eager_on_graph import (
    eager_on_graph,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.activation.fused_sigmoid_mul_inplace import (
    fused_sigmoid_mul_inplace,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.activation.silu_and_mul_rows import (
    silu_and_mul_rows,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_canonical_selection import (
    qsa_canonical_selection,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_expand_block_indices import (
    qsa_expand_block_indices,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_mtp_reuse_topk import (
    qsa_mtp_reuse_topk,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_index_scores import (
    _takes_prefill_query_tile,
    qsa_paged_index_scores,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_index_scores import (
    _takes_visible_split as _scores_take_visible_split,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_kv_store_nvfp4 import (
    qsa_paged_kv_store_nvfp4,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_sparse_gqa_nvfp4 import (
    qsa_paged_sparse_gqa_nvfp4,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_paged_sparse_gqa_nvfp4_decode import (  # noqa: E501
    qsa_paged_sparse_gqa_nvfp4_decode,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_prefill_compress import (
    qsa_prefill_compress,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_rollback_planes import (
    plane_table as _plane_table,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_rollback_planes import (
    qsa_rollback_gather,
    qsa_rollback_restore,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_topk_expand_decode import (
    orders_rows as _topk_expand_orders_rows,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_topk_expand_decode import (
    qsa_topk_expand_decode,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_topk_expand_decode import (
    supports as _topk_expand_supports,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.qsa_topk_select import (
    qsa_topk_select,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.attention.spec_rollback_commit import (
    spec_rollback_commit,
    spec_rollback_stage,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_first_rescale_tc import (
    BK as _HC_RESCALE_BK,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_first_rescale_tc import (
    _pick_split as _hc_rescale_split,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_first_rescale_tc import (
    qwen4_exp_hc_first_rescale_tc,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_inject_norm import (
    _PDL_TRIGGER as _HC_INJECT_PDL_TRIGGER,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_inject_norm import (
    qwen4_exp_hc_inject_norm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_inject_rstd import (
    qwen4_exp_hc_inject_rstd,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_lowrank_act import (
    qwen4_exp_hc_lowrank_act,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_up_gate_mean import (
    qwen4_exp_hc_up_gate_mean,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_hc_up_gate_mean_rstd import (
    qwen4_exp_hc_up_gate_mean_rstd,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.hc.qwen4_exp_post_inject import (
    qwen4_exp_post_inject,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.moe.fused_sigmoid_gate_mul_add import (
    fused_sigmoid_gate_mul_add,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.moe.renorm_moe_routing_op import (
    renorm_moe_routing_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.norm.flashinfer_gemma_rmsnorm import (
    flashinfer_gemma_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.norm.rms_norm_gated_token_major import (
    rms_norm_gated_token_major,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.ple.ple_gate_conv_inject_norm import (
    ple_gate_conv_inject_norm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.ple.ple_ngram_embed import (
    ple_ngram_embed,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.quantization.bf16_to_fp32 import (
    bf16_to_fp32,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.quantization.fp8_quantize_1x128 import (
    fp8_quantize_1x128,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.ssm.causal_conv1d_update import (
    causal_conv1d_update,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.ssm.cu_seqlens_from_seq_lens import (
    cu_seqlens_from_seq_lens,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.ssm.gdn_decode_rows_gated_norm import (
    gdn_decode_rows,
    gdn_decode_rows_gated_norm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.ssm.gdn_prefill_conv_post import (
    gdn_prefill_conv_post,
)
from tensorrt_llm._torch.attention.backends.fmha.phased import get_spec_decoding_position_offsets
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.distributed.allreduce_helper import CustomAllReduceHelper
from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.modules.fla.cached_replay import CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import (
    eager_on_graph as _keyed_eager_on_graph,
)
from tensorrt_llm._torch.pyexecutor.model_loader import effective_mamba_ssm_cache_dtype
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks
from tensorrt_llm._torch.speculative import get_spec_worker, should_use_separate_draft_kv_cache
from tensorrt_llm._torch.utils import is_gdn_flashinfer_replay_enabled, is_gdn_replay_enabled
from tensorrt_llm._utils import mpi_disabled
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceStrategy
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

from .. import ple_host_table as _ple_host
from . import pool as _pool
from . import vision as _vision
from . import weights as _weights
from .ops.attention.draft_continue_rows__decode import draft_continue_select, draft_rows_gather
from .ops.attention.qsa_fp8_pages_to_bf16__prefill import qsa_fp8_pages_to_bf16
from .ops.attention.qsa_index_scores_cute__prefill import qsa_index_scores_prefill
from .ops.attention.qsa_index_scores_cute__prefill import supports as _index_scores_cute_supports
from .ops.attention.qsa_index_scores_groups_cute__decode import (
    index_scores_groups as index_scores_groups_cute,
)
from .ops.attention.qsa_index_scores_groups_cute__decode import serves as _index_groups_cute_serves
from .ops.attention.qsa_index_split_plan__decode import index_split_plan
from .ops.attention.qsa_paged_index_store import qsa_paged_index_store
from .ops.attention.qsa_paged_kv_store import qsa_paged_kv_store
from .ops.attention.qsa_paged_sparse_gqa import qsa_paged_sparse_gqa
from .ops.attention.qsa_sparse_gqa_grouped_cute__decode import qsa_sparse_gqa_grouped_cute
from .ops.attention.qsa_sparse_gqa_rowtile__prefill import qsa_sparse_gqa_rowtile
from .ops.attention.qsa_sparse_gqa_splitk_cute__decode import qsa_sparse_gqa_splitk_cute
from .ops.attention.qsa_spec_step_metadata__decode import (
    SpecStepMetadataPrologue,
    take_step_positions,
)
from .ops.attention.qsa_topk_expand_cluster__decode import qsa_topk_expand_cluster
from .ops.attention.qsa_topk_expand_cluster__decode import supports as _topk_expand_cluster_supports
from .ops.attention.qsa_topk_rows_cute__prefill import qsa_topk_rows
from .ops.attention.qsa_topk_rows_cute__prefill import supports as _topk_rows_supports
from .ops.attention.qsa_triton_kernels__stable_jit import splitk_launch_config as _gqa_splitk_config
from .ops.attention.qsa_triton_kernels__stable_jit import warm_decode_buckets
from .ops.attention.qwen4_exp_attn_decode_prologue__decode import qwen4_exp_attn_decode_prologue
from .ops.attention.spec_rollback_begin__verify import spec_rollback_begin
from .ops.comm.hc_allreduce_inject_norm__tp_sum import fits as _hc_ar_inject_operands_fit
from .ops.comm.hc_allreduce_inject_norm__tp_sum import hc_allreduce_inject_norm
from .ops.comm.lamport_allgather__index_split import build as _selection_lamport_build
from .ops.comm.lamport_allgather__index_split import lamport_allgather
from .ops.comm.mnnvl_allreduce__tp_sum import mnnvl_tp_sum, mnnvl_workspace_args
from .ops.comm.moe_a2a__dp_dispatch_combine import build_strategy as _build_moe_a2a
from .ops.comm.moe_a2a__dp_dispatch_combine import moe_a2a_combine, moe_a2a_dispatch
from .ops.embedding.embed_rows__decode import embed_rows
from .ops.gemm.low_m_bf16_mm__decode import low_m_bf16_mm
from .ops.hc.qwen4_exp_hc_decode_rows_tc__decode import MAX_ROWS as _HC_ROWS_TC_ROWS
from .ops.hc.qwen4_exp_hc_decode_rows_tc__decode import (
    qwen4_exp_hc_decode_rows_first,
    qwen4_exp_hc_decode_rows_mix,
)
from .ops.hc.qwen4_exp_hc_decode_tc_first__decode import MAX_ROWS as _HC_TC_FIRST_ROWS
from .ops.hc.qwen4_exp_hc_decode_tc_first__decode import precompile as hc_tc_first_precompile
from .ops.hc.qwen4_exp_hc_decode_tc_first__decode import qwen4_exp_hc_decode_tc_first
from .ops.moe.fp8_block_scale_moe_runner import (
    fp8_block_scale_moe_front,
    fp8_block_scale_moe_runner,
)
from .ops.moe.moe_front_router_quant__decode import MAX_ROWS as _MOE_FRONT_ROWS
from .ops.moe.moe_front_router_quant__decode import moe_front_router_quant
from .ops.moe.moe_prefill_grouped__prefill import (
    moe_prefill_grouped,
    prefill_grouped_ready,
    warm_prefill_grouped,
)
from .ops.moe.moe_tpb_decode__decode import MAX_ROWS as _MOE_TPB_ROWS
from .ops.moe.moe_tpb_decode__decode import BlockExperts as _BlockExperts
from .ops.moe.moe_tpb_decode__decode import SharedBlocks as _TpbShared
from .ops.moe.moe_tpb_decode__decode import moe_tpb_decode
from .ops.moe.mtp_moe_combine_inject__decode import mtp_moe_combine_inject
from .ops.ssm.causal_conv1d_token_major__verify import causal_conv1d_token_major
from .ops.ssm.flashinfer_chunk_gated_delta_rule import INPLACE_MAX_TOKENS as _GDN_INPLACE_MAX_TOKENS
from .ops.ssm.flashinfer_chunk_gated_delta_rule import _chunk_carried as _gdn_chunk_carried
from .ops.ssm.flashinfer_chunk_gated_delta_rule import _chunk_cp_inplace as _gdn_chunk_cp_inplace
from .ops.ssm.flashinfer_chunk_gated_delta_rule import _reaches_cp as _gdn_reaches_cp
from .ops.ssm.flashinfer_chunk_gated_delta_rule import (
    chunk_split_first_pass,
    cp_chunk_len_for,
    flashinfer_chunk_gated_delta_rule,
    split_takes_cp,
)
from .ops.ssm.gdn_cached_replay_cute__verify import GEOMETRIES as _GDN_REPLAY_CUTE_GEOMETRIES
from .ops.ssm.gdn_cached_replay_cute__verify import conv_fold_supported as gdn_conv_fold_supported
from .ops.ssm.gdn_cached_replay_cute__verify import gdn_cached_replay_cute
from .ops.ssm.gdn_cached_replay_cute__verify import precompile as gdn_replay_precompile
from .ops.ssm.gdn_in_proj_conv__verify import CONV_DIM as _GDN_IN_PROJ_CONV_DIM
from .ops.ssm.gdn_in_proj_conv__verify import N_OUT as _GDN_IN_PROJ_N_OUT
from .ops.ssm.gdn_in_proj_conv__verify import fused_tactic as _gdn_in_proj_conv_fused_tactic
from .ops.ssm.gdn_in_proj_conv__verify import gdn_in_proj_conv
from .ops.ssm.gdn_inforward_tail import gdn_inforward_conv_swap, gdn_inforward_tail

# The certified GPU architecture, checked again because a direct instantiation bypasses routing.
_SM = (10, 3)

# `moe_config.backend` requests served by the trunk's routed experts (`fp8_block_scale_moe_runner`).
_TRUNK_MOE_BACKENDS = ("AUTO", "TRTLLM")

# The engine's resolved quantization of the one checkpoint this target loads.
_QUANT_ALGO = QuantAlgo.FP8_BLOCK_SCALES

# Qwen3.8-Flash-Next's (layers, hidden, experts, attention heads, linear value heads, vocab), and
# its vision tower's (depth, hidden, heads, merge, out_hidden, num_pos, deepstack_indexes, act).
_TEXT_SHAPE = (48, 2560, 512, 24, 48, 248320)
_VISION_SHAPE = (27, 1152, 16, 2, 2560, 2304, (), "gelu_pytorch_tanh")

# The checkpoint's settings that change what it computes without changing a weight's shape,
# at the values the target implements: a fine-tune of the same shape that changed one would
# load and run with wrong logits and no error. Read off the text config as the target reads
# them (`_text_settings`).
_TEXT_SETTINGS = {
    # The routed experts: Renormalize routing over the top 10 and a SiLU activation.
    "norm_topk_prob": True,
    "num_experts_per_tok": 10,
    "hidden_act": "silu",
    # Attention: two KV heads of 256, no projection bias, a sigmoid output gate on every head
    # (absent `attn_output_gate` reads as gated), and an output head of its own.
    "num_key_value_heads": 2,
    "head_dim": 256,
    "attention_bias": False,
    "output_gate_type": "sigmoid",
    "attn_output_gate": True,
    "tie_word_embeddings": False,
    # Rope: a quarter of the head (64 dims), theta 1e7, interleaved mRoPE over sections
    # 11 / 11 / 10, unscaled.
    "rope_theta": 1e7,
    "partial_rotary_factor": 0.25,
    "mrope_section": (11, 11, 10),
    "mrope_interleaved": True,
    "rope_factor": None,
    # The MTP module: one full-attention layer on the trunk's embedding that reads the end of
    # the stack (`mtp_use_hidden_state_from_layer: null`). The `mtp.` fields are the `mtp`
    # block's, a mapping or a config object; a field it omits reads as the value here. (Its
    # own rope theta the target reads.)
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": False,
    "mtp.num_hidden_layers": 1,
    "mtp.layer_types": ("full_attention",),
    "mtp.mtp_use_hidden_state_from_layer": None,
}

# Rows per request of the widest verify step (`max_draft_len + 1`).
_MAX_VERIFY_WIDTH = _pool.MAX_VERIFY_WIDTH

_SSM_STATE_DTYPES = _pool.SSM_STATE_DTYPES

# QSA backend metadata buffers rebuilt every step. Their absence means the engine did not
# select the QSA metadata and the index plane would never be written.
_QSA_FIELDS = (
    "qsa_block_table",
    "qsa_req_idx_per_token",
    "qsa_logical_positions",
    "qsa_sequence_lengths",
    "qsa_visible_blocks",
    "qsa_topk_indices",
    "qsa_topk_row_starts",
)

# Fields read off `attn_metadata.mamba_metadata`. Absence is silent: a smoke run passes
# with no recurrent state at all.
_MAMBA_FIELDS = (
    "state_indices",
    "has_initial_states",
    "query_start_loc",
    "query_start_loc_long",
)

# thop_attention's constant arguments: in-kernel RoPE off (the rope runs before the op),
# causal mask, no MLA, sparse or cross surface.
_CALL_CONSTANTS = dict(
    output_sf=None,
    k=None,
    v=None,
    is_fused_qkv=True,
    update_kv_cache=True,
    attention_input_type=0,
    is_mla_enable=False,
    mask_type=1,
    q_scaling=1.0,
    attention_sinks=None,
    position_embedding_type=0,
    rotary_inv_freq=None,
    rotary_cos_sin=None,
    rope_dim=0,
    rope_base=10000.0,
    rope_scale_type=0,
    rope_scale=1.0,
    rope_short_m_scale=1.0,
    rope_long_m_scale=1.0,
    rope_max_positions=1024,
    rope_original_max_positions=1024,
    out_scale=None,
    latent_cache=None,
    q_pe=None,
    q_lora_rank=None,
    kv_lora_rank=None,
    qk_nope_head_dim=None,
    qk_rope_head_dim=None,
    v_head_dim=None,
    rope_append=None,
    chunked_prefill_buffer_batch_size=1,
    attention_chunk_size=None,
    softmax_stats_tensor=None,
    sparse_kv_indices=None,
    sparse_kv_offsets=None,
    sparse_attn_indices=None,
    sparse_attn_offsets=None,
    sparse_attn_indices_block_size=0,
    mrope_rotary_cos_sin=None,
    mrope_position_deltas=None,
    helix_position_offsets=None,
    helix_is_inactive_rank=None,
    cross_kv=None,
    relative_attention_bias=None,
    relative_attention_max_distance=0,
    kv_norm_weight=None,
    kv_norm_eps=1e-6,
    skip_correction_threshold=0.0,
)

_ROUTING_METHOD_RENORM = 1
_ACT_TYPE_SWIGLU = 0
_PAD_SLOT_ID = -1
#: Whether a decode step's `low_m_bf16_mm` projections launch with PDL (the FlashInfer kernels
#: stream the weight before waiting on the producer; the cuBLAS branch ignores it). Off with
#: TRTLLM_ENABLE_PDL=0.
_DECODE_PROJ_PDL = get_env_enable_pdl()
#: (request x value head) pairs a verify step up to which the CuTe cached replay launches with PDL
#: and the conv before it triggers at its start, so the replay streams the checkpoint and history
#: (nothing earlier in the step writes them) while the conv runs.
_GDN_REPLAY_PDL_MAX_HEAD_ROWS = 24
#: A tp2/tep2 rank's GDN heads (8 key / 24 value), and the HT generation batches (from this many
#: requests: the partitioned replay) whose replay launch comes from `_GDN_REPLAY_HT_LAUNCH`.
_GDN_REPLAY_HT_GEOMETRY = (8, 24)
_GDN_REPLAY_HT_MIN_REQUESTS = 16
#: `(max requests, pdl, CuTe replay mapping or None for its default)` for a tp2/tep2 HT verify
#: batch (PDL off throughout with TRTLLM_ENABLE_PDL=0). The mappings
#: (`gdn_cached_replay_cute.default_config`'s knobs) only move where a tile is loaded and which
#: CTA runs it, so every choice is bitwise the default launch.
_GDN_REPLAY_HT_LAUNCH = (
    (24, get_env_enable_pdl(), dict(tpc=8, warps=4, pf=0, minb=5, early_acc=True)),
    (40, get_env_enable_pdl(), dict(tpc=8, warps=4, pf=1, minb=6)),
    (56, get_env_enable_pdl(), dict(tpc=8, warps=4, pf=0, minb=5, early_acc=True)),
    (80, get_env_enable_pdl(), None),
    (1 << 30, False, dict(tpc=8, warps=4, pf=0, minb=5, early_acc=True)),
)
#: Requests up to which a tp2/tep2 HT verify batch folds the conv into the CuTe replay
#: (`conv_fold=`, bitwise `causal_conv1d_token_major`); the replay writes the per-row windows and
#: the acceptance promote advances the pool window from them.
_GDN_VERIFY_CONV_FOLD_MAX_REQUESTS = 24
#: Whether a verify step's conv (`causal_conv1d_token_major`) is a PDL dependent of the input
#: projection: it reads the slots and the conv pool before `griddepcontrol.wait`, the q|k|v after.
#: Off with TRTLLM_ENABLE_PDL=0.
_GDN_CONV_PDL_WAIT = get_env_enable_pdl()
#: Whether a decode step's QSA index scoring (`_decode_index_selection`) launches with PDL: its
#: kernels wait before their first read and the decode prologue triggers at its start.
_INDEX_SCORES_PDL = get_env_enable_pdl()
#: Request groups (this rank's share) from which verify index scoring takes this target's CuTe
#: kernel (`qsa_index_scores_groups_cute__decode`) over the family's Triton one. It sums in another
#: order (~7e-8 relative), so a row's top-k can differ only at a near-tie.
_INDEX_SCORES_CUTE_MIN_GROUPS = 8
_SOFTPLUS_THRESHOLD = 20.0
_L2_NORM_EPS = 1e-6
_PLE_CLAMP_MIN = 1e-6
#: Rows up to which a hyper-connection mixer's first half is one tcgen05 launch
#: (`qwen4_exp_hc_decode_tc_first`, two 128-row MMA tiles) instead of cuBLAS + lowrank_act.
_HC_TC_FIRST_MAX_ROWS = _HC_TC_FIRST_ROWS
#: Row counts over which a decode or draft hyper-connection site (`_hc_site`) runs whole as the
#: two `qwen4_exp_hc_decode_rows_*` launches. A dispatch threshold on a per-capture row count, not
#: a buffer bound; zero rows keep the other path.
_HC_ROWS_TC_MIN_ROWS = 1
_HC_ROWS_TC_MAX_ROWS = _HC_ROWS_TC_ROWS
#: Under attention TP, a decode block's partial output of `_HC_AR_INJECT_MIN_ROWS` ..
#: `_HC_RSTD_MIN_ROWS - 1` rows skips the TP sum and the next hyper-connection site sums it in its
#: injection (`ops/comm/hc_allreduce_inject_norm__tp_sum`, bitwise the all-reduce followed by
#: `_hc_inject_norm_launch`). Smaller steps use the MNNVL sum or the decode-rows site.
_HC_AR_INJECT_MIN_ROWS = 33
#: Prefill steps of at least this many rows store the residual and a `[T, C]` fp32 rstd
#: (`qwen4_exp_hc_inject_rstd`) instead of the `[T, C*H]` normed stream, which `_hc_mixer_rstd`
#: forms on chip. A dispatch threshold on the step's rows, not a buffer bound.
_HC_RSTD_MIN_ROWS = 3072
#: Warps a program of `_hc_inject_norm_launch` takes below `_HC_INJECT_FEW_WARPS_MIN_ROWS` rows,
#: and two from there (as `qwen4_exp_hc_inject_norm` does). Dispatch thresholds, not buffer bounds.
_HC_INJECT_WARPS = 4
_HC_INJECT_FEW_WARPS_MIN_ROWS = 512
#: Rows up to which `qwen4_exp_hc_up_gate_mean` runs its epilogue on eight warps instead of four
#: (at one tile a CTA the gate's MUFU work is the critical path). Same bits.
_HC_UGM_WIDE_MAX_ROWS = 384
#: Whether `_hc_inject_norm_launch` launches with PDL and waits for its predecessor in the kernel
#: (`griddepcontrol.wait`); off with TRTLLM_ENABLE_PDL=0.
_HC_INJECT_PDL_WAIT = _HC_INJECT_PDL_TRIGGER
_QSA_POSITION_AXES = _pool.QSA_POSITION_AXES
# The compressed index keys rotate at coordinate 0 alone, as the index query does.
_QSA_MROPE_SECTION = None
# The routed experts' activation scales are the plain fp32 `amax / 448`, not the UE8M0
# power-of-two form: the runner reads them so (`fp8_block_scale_moe_runner`).
_MOE_USE_UE8M0 = False
# Rows per FP8 runner call under attention DP: the rows an attention-DP group gathers (up to
# its size times a rank's) run in chunks of it.
_MOE_MAX_ROWS = 8192
#: A decode step's routed experts up to this many rows enter the runner from one front-end launch
#: (`moe_front_router_quant`: router logits + 1x128 quantization). Past 8 rows the front end
#: contends with the concurrent shared expert.
_MOE_FRONT_MAX_ROWS = min(8, _MOE_FRONT_ROWS)
#: Past `_MOE_FRONT_MAX_ROWS` an attention-TP decode step hands the runner the router's bf16
#: logits to route itself (bitwise `renorm_moe_routing_op`), forking the shared expert only once
#: the router and quantization are queued (`before_runner`) so it overlaps the runner's GEMMs.
#: Up to this many rows (the routing kernel's bound) the runner routes and quantizes in one launch
#: (`fp8_block_scale_moe_front`) instead of the 8-CTA cluster kernel; the layer's output is bitwise
#: the cluster path's (only a slot's offset within its expert may differ; nothing reads it).
_MOE_FAST_FRONT_MAX_ROWS = 1024
#: With the block-major expert copy (`weights.MOE_TPB_ENV`, attention TP4), decode steps up to this
#: many rows take `moe_tpb_decode` over this rank's 128-row intermediate blocks. From
#: `_MOE_TPB_SHARED_MIN_ROWS` rows it also folds in this rank's shared-expert slice (`shared=`):
#: the launch holds nearly every SM, so a side-stream shared expert would wait behind it.
_MOE_TPB_MAX_ROWS = min(8, _MOE_TPB_ROWS)
#: The fold's smallest step: at a 3-row verify the side stream's shared expert mostly hides behind
#: the routed launch and folding it in costs more.
_MOE_TPB_SHARED_MIN_ROWS = 5
#: Rows per FP8 runner call under attention TP (or at one rank): the runner is certified to 16384
#: rows (past its last tuning bucket), and one call reads every expert's weights once.
_MOE_MAX_ROWS_TP = 16384
#: Smallest rows of a runner call under attention TP that run the window as two grouped FP8
#: block-scale GEMMs (`moe_prefill_grouped`: no host synchronization, CUDA graph capturable)
#: instead of the trtllm-gen runner. 0 keeps the runner at every row count (tests set it).
_MOE_PREFILL_GROUPED_MIN_ROWS = 2048
#: The expert id a padding row's slots are routed to: outside every rank's window, so the
#: runner skips the slot (`_routed_experts`).
_NO_EXPERT = -1
#: Rows up to which `_mtp_front` runs its embedding branch on the side stream; from 16 rows the
#: hidden branch's GEMM (4 streams a row) fills the GPU and the concurrent branch only slows it.
_MTP_FRONT_FORK_MAX_ROWS = 8
#: Context rows from which an attention-TP rank scores and selects only its share of a prefill
#: step's rows and gathers the others' (`_split_context_selection`), instead of every rank
#: repeating the whole unsplit index selection. Every rank holds the same bits.
_SELECTION_SPLIT_MIN_ROWS = 1024
#: A split selection cuts rows where the ranks' summed costs are equal (`_selection_cuts`); a row
#: costs its window (`position + 1` tokens) plus this many tokens of fixed work (a measured fit).
_SELECTION_BALANCE_ROW_TOKENS = 12500
#: Rows a balanced cut is rounded to (the index scores' query tile).
_SELECTION_CUT_ALIGN = 64
#: Pieces a rank's share of a split selection is processed in: each piece's gather runs on the
#: comm stream while the next is scored, so only the last gather is exposed. Same bits.
_SELECTION_PIECES = 4
#: Rows from which a context chunk's selection over a wide score table takes the row top-k kernel
#: (`qsa_topk_rows`) instead of the family's wide kernel; same bits.
_TOPK_ROWS_MIN_ROWS = 256
#: Elements of one score table: the index scores write their output rows and the selection
#: kernels address theirs with int32 arithmetic (row times the table's width), so a call scores
#: fewer rows than this over the table's columns: at `max_seq_len` 262144 (65568 columns) a
#: context part past 32752 rows is scored in chunks.
_SCORE_TABLE_ELEMENTS = 2**31


def _context_index_scores(q, index_cache, block_table, positions, requests, tpb, cr, context_rows):
    """`qsa_paged_index_scores` (visible blocks only) of a context part of a step's rows.

    Takes the tcgen05 kernel `qsa_index_scores_prefill` where it serves the call (same bits).
    """
    if _takes_prefill_query_tile(q, True, context_rows, False) and _index_scores_cute_supports(
        q, index_cache, tpb, cr
    ):
        return qsa_index_scores_prefill(q, index_cache, block_table, positions, requests, tpb, cr)
    return qsa_paged_index_scores(
        q=q,
        index_cache=index_cache,
        block_table=block_table,
        query_positions=positions,
        request_indices=requests,
        tokens_per_block=tpb,
        compress_ratio=cr,
        only_visible_blocks=True,
        context_rows=context_rows,
        prefill_query_tile=True,
        wide_split=True,
    )


#: Rows a piece of a pipelined selection keeps at least (whole query tiles of the index scores).
_SELECTION_PIECE_MIN_ROWS = 1024
#: Breakable-graph buckets of at most this many rows replay a context-only step's GDN and sparse
#: attention breaks as CUDA graphs keyed by what fixes their launches (`_gdn_graph_key`,
#: `_attn_graph_key`); a key the warmup did not capture runs eagerly. Same bits.
_GRAPH_BREAKS_MAX_TOKENS = 4096
#: Attention-TP group sizes whose prefill steps split the context selection and run
#: sequence-parallel (`PrefillTarget._sp_split`). Two ranks only: at four the sums would run in
#: another order than the all-reduce's.
_PREFILL_SPLIT_DEFAULT_TP = frozenset({2})
#: Rows from which a prefill step runs sequence-parallel: a rank's share stays past the HC
#: kernels' small-row paths, so each runs the configuration the whole step would.
_SP_MIN_ROWS = 4096
# cuBLASLt picks a bf16 projection's kernel by its exact row count up to this many rows, and
# past them by the 8-row group the row count falls in (`_projection_warm_rows`).
_KERNEL_CHOICE_EACH_ROW = 16
_KERNEL_CHOICE_ROW_GROUP = 8
# The logits rows `lm_head`'s kernels are loaded for: a step's requests' last tokens and its
# verify rows, up to 256 requests at MTP-3 or 128 at MTP-7.
_LM_HEAD_WARM_ROWS = 1024
# A context launch of at least this many rows over a bf16 or fp8 pool takes
# `qsa_sparse_gqa_rowtile`; below it the per-row kernel is faster.
_ROWTILE_MIN_ROWS = 2048  # capacity: fixed: a dispatch threshold, not a buffer size
#: Requests from which MTP verify rows over an e4m3 pool take `qsa_sparse_gqa_grouped_cute` where
#: it serves them (one load and widening of each selected K/V token for a request's rows, over the
#: union of their selections) instead of the per-row `qsa_sparse_gqa_splitk_cute`.
_GROUPED_DECODE_MIN_REQUESTS = 32  # capacity: fixed: a dispatch threshold, not a buffer size
#: Rows from which a generation step with several rows a request (MTP verify, draft-extend) scores
#: its QSA index keys over request groups (one key gather serves a request's rows); low-latency
#: steps of up to 16 rows keep the entry's own choice.
_INDEX_GROUP_MIN_ROWS = 17  # capacity: fixed: a dispatch threshold, not a buffer size
#: Requests from which an attention-TP verify step splits QSA index scoring and selection by
#: request over the TP group (`_index_split_rows`) and gathers the int32 selections in rank order.
#: Bitwise the unsplit selection; needs `n % t == 0`.
_INDEX_SPLIT_MIN_REQUESTS = 16
#: Rows up to which the split's gather is one Lamport exchange over symmetric memory
#: (`lamport_allgather__index_split`) instead of NCCL, bitwise the same; it sizes the buffers.
_INDEX_SPLIT_LAMPORT_ROWS = 768
# Over an fp8 pool the launch reads its requests' pages widened into a bf16 scratch; a launch
# whose scratch would pass this many bytes stays on the per-row kernel.
_FP8_ROWTILE_MAX_SCRATCH = 1 << 30  # capacity: fixed: bounds a per-call scratch allocation
#: Opt-in switch (`weights.switch`): a context step over an fp8 pool whose pages would widen
#: past `_FP8_ROWTILE_MAX_SCRATCH` runs one row-tile launch per group of requests that fits
#: (`_rowtile_scratch_groups`) instead of the per-row kernel. Off by default because such steps'
#: outputs then move by rounding. Read once, when the target is built.
ROWTILE_SCRATCH_GROUPS_ENV = "TRTLLM_MODELING_V2_ROWTILE_SCRATCH_GROUPS"
#: The stock MoE communication override (`communication_factory`), compared upper-cased like the
#: stock factory. Under attention DP, `NVLINK_ONE_SIDED` runs the MoE round trip as an all-to-all
#: dispatch / combine (`moe_a2a__dp_dispatch_combine`): a reference path, not a faster one; it
#: needs MPI, attention DP and MNNVL. Unset or `ALLGATHER` keeps the all-gather / reduce-scatter
#: round trip. `check_deployment` declines any other value, and the all-to-all where it cannot
#: run. Read once, when the target is built.
MOE_COMM_METHOD_ENV = "TRTLLM_FORCE_COMM_METHOD"


def _moe_comm_method() -> str:
    """`MOE_COMM_METHOD_ENV` upper-cased, `ALLGATHER` when unset."""
    return os.environ.get(MOE_COMM_METHOD_ENV, "ALLGATHER").upper()


# Rows per FP8 runner call over an attention-DP group's gathered rows (`_routed_experts`): one
# call reads the window's weights once, bitwise the `_MOE_MAX_ROWS`-row calls; the router, its
# routing and the activation quantization still run per `_MOE_MAX_ROWS` rows.
_ADP_WINDOW_MAX_ROWS = 4 * _MOE_MAX_ROWS
# What every bind of this target prints, on every rank: the disaggregated-serving test looks for
# it in each worker's log.
_LOG_PREFIX = "modeling_v2 qwen4_exp fp8 unified target"

# In-forward Mamba snapshots: a context chunk ending at a snapshot point P just before the prompt
# end L runs through L, leaving the GDN and PLE state at P in the live slot and staging the state
# at L (`PrefillTarget._gdn_split`). The fewest chunk tokens before P a split request takes: past
# `_GDN_INPLACE_MAX_TOKENS` (the main chunk's FlashInfer path) and covering the conv (3) and PLE
# (9 columns, 2 ids) windows that end at P.
_INFORWARD_MIN_MAIN = _GDN_INPLACE_MAX_TOKENS + 1

#: Rows up to which `tp_sum` takes the multicast NVLink one-shot kernel
#: (`ops/comm/mnnvl_allreduce__tp_sum`) instead of the IPC one, where the platform supports MNNVL
#: (`MNNVLAllReduce.is_mnnvl`); the two return the same bits.
_TP_SUM_MNNVL_MAX_ROWS = 32


def _runtime_hooks() -> RuntimeHooks:
    """Step-loop hooks of the fast paths: one execution stream, one-launch verify prologue."""
    return RuntimeHooks(loop_on_execution_stream=True, decode_prologue=SpecStepMetadataPrologue())


class _Replay(NamedTuple):
    """The GDN cached replay's history buffers and bookkeeping of one manager."""

    old: dict  # GDN layer -> (old_x, old_B, old_dt, old_dA_cumsum)
    cache_buf_idx: torch.Tensor  # [slots] int32: the history half each slot writes next
    prev_num_accepted: torch.Tensor  # [slots] int32: tokens pending since the checkpoint
    history: int  # rows a history half holds
    width: int  # the one verify width a step may carry per generation request


class _Bound(NamedTuple):
    """What the core reads off one KV cache manager; valid only for that manager.

    The manager, this rank's pool views and, under MTP, the verify step's runtime-owned buffers:
    the per-step conv windows (`int_conv`, by GDN layer), the rows `update_mamba_states` promotes
    them from (`int_idx`), the cached replay and the in-forward snapshot staging rows.
    """

    cm: object
    pools: _pool.RankPools
    int_idx: torch.Tensor | None = None
    int_conv: dict | None = None
    replay: _Replay | None = None
    stage: "_InforwardStage | None" = None


class _InforwardStage(NamedTuple):
    """The manager's in-forward snapshot staging rows (`Qwen4ExpHybridCacheManager`).

    Each GDN layer's SSM `[rows, HV, V, K]` and conv `[rows, conv_dim, 3]` rows, and the PLE
    layer's conv state `[rows, stream, 9]` and n-gram history `[rows, ngram_size - 1]` rows. The
    forward stages each split request's prompt-end state in its row; the manager copies it back
    after committing the snapshot at P.
    """

    ssm: dict
    conv: dict
    ple_conv: torch.Tensor
    ple_ngram: torch.Tensor


class _Split(NamedTuple):
    """One step's in-forward snapshot split, from `Mamba2Metadata.inforward_split`.

    `k` of the step's context requests end their chunk at the prompt end L with a snapshot point
    P inside it; in the GDN recurrence each is two sequences, [.., P) and the tail [P, L).

    `cu` (int64) the split sequences' boundaries; `gather` (int32) each split sequence's live
    slot and `carry` (bool) whether it starts from it (a tail starts from zeros, its final state
    dropped); `main_rows` (int64) each request's [.., P) or whole sequence and `live` (int32) its
    slot; `tail_tok` (int64) the tail rows, `cu_tail` (int64) their boundaries, `stage_rows`
    (int64) the split requests' staging rows, `tail_slots` (int64) their live slots; `conv_win`,
    `ple_win` and `ngram_win` (int64) the rows of the GDN conv (3), PLE conv (9) and n-gram id (2)
    windows that end at P, oldest first, request by request; `whole` (`chunk_split_first_pass`)
    the rows before every P (int64, or a host count when one request is split once), their
    per-request boundaries and carry flags, or None where the first pass keeps the split
    boundaries; `raw` the engine's `InforwardSnapshotSplit`, views of one persistent buffer at
    fixed offsets that a break's CUDA graph reads.
    """

    k: int
    cu: torch.Tensor
    gather: torch.Tensor
    carry: torch.Tensor
    main_rows: torch.Tensor
    live: torch.Tensor
    tail_tok: torch.Tensor
    cu_tail: torch.Tensor
    stage_rows: torch.Tensor
    tail_slots: torch.Tensor
    conv_win: torch.Tensor
    ple_win: torch.Tensor
    ngram_win: torch.Tensor
    whole: tuple | None = None
    raw: object = None


def _inforward_stage(cm, gdn_layers, ple_layer: int) -> _InforwardStage | None:
    """The in-forward snapshot staging rows of `cm`, or None when it plans no snapshots."""
    if getattr(cm, "inforward_snapshot_rows", 0) <= 0:
        return None
    _pool.require(
        cm,
        ("inforward_snapshot_stage_views", "inforward_ple_stage_views", "conv_state_shape"),
        "stages the in-forward snapshot's prompt-end GDN and PLE state",
    )
    ple = cm.inforward_ple_stage_views(ple_layer)
    assert ple is not None, (
        f"the KV cache manager plans in-forward snapshots but stages no PLE state for decoder "
        f"layer {ple_layer}"
    )
    ssm, conv = {}, {}
    for i in gdn_layers:
        ssm[i], conv[i] = cm.inforward_snapshot_stage_views(i)
    return _InforwardStage(ssm, conv, ple[0], ple[1])


class _TpPartial(NamedTuple):
    """A decode block's partial output whose TP sum is left to the next hyper-connection site."""

    partial: torch.Tensor


class _HcArInject(NamedTuple):
    """IPC all-reduce workspace and TP rank / size for the decode sites' fused TP sum + inject."""

    workspace: torch.Tensor
    rank: int
    nranks: int
    #: rows whose `nranks * rows * hidden * 2` bytes fit one of the workspace's sub-buffers
    max_rows: int


def _hc_ar_inject_defers(core, rows: int) -> bool:
    """Whether `_tp_partial` leaves a decode block's partial of `rows` rows to the next site."""
    fused = getattr(core, "hc_ar_inject", None)
    return fused is not None and max(
        _HC_AR_INJECT_MIN_ROWS, _HC_ROWS_TC_MAX_ROWS + 1
    ) <= rows < min(_HC_RSTD_MIN_ROWS, fused.max_rows + 1)


def _tp_partial(core, x: torch.Tensor):
    """A decode block's partial output for the next hyper-connection site.

    Left unsummed (`_TpPartial`) where that site sums it itself, else `core.tp_sum(x)`. A core
    without `hc_ar_inject` (another layout, or a test's stand-in core) always sums.
    """
    if _hc_ar_inject_defers(core, x.shape[0]):
        return _TpPartial(x)
    return core.tp_sum(x)


class _DpPadding(NamedTuple):
    """The group's per-rank row counts of one step's attention-DP MoE round trip (ragged split)."""

    counts: tuple[int, ...]


class _DraftBound(NamedTuple):
    """The draft layer's planes in the manager its draft step is handed.

    Its QSA index plane, that manager's position plane, its paged K/V and, under IndexShare, its
    row of the metadata's shared selection buffers.
    """

    cm: object
    index_cache: torch.Tensor
    position_cache: torch.Tensor
    kv: _pool.KvPlanes
    share_slot: int | None


class Qwen4ExpUnifiedModelingV2(DecoderModel):
    def __init__(self, model_config: ModelConfig, composite_config: PretrainedConfig):
        """`composite_config` carries the vision config and media ids the text config lacks."""
        super().__init__(model_config)
        cfg = model_config.pretrained_config
        mapping = model_config.mapping
        assert torch.cuda.get_device_capability() == _SM, (
            f"target certified on sm_{_SM[0]}{_SM[1]}, running on "
            f"sm_{''.join(map(str, torch.cuda.get_device_capability()))}"
        )
        assert getattr(cfg, "model_type", None) == "qwen4_exp_text", (
            f"the core must be built on the normalized text config; got model_type "
            f"{getattr(cfg, 'model_type', None)!r}"
        )
        self.mapping = mapping
        # Ranks the attention and recurrent heads are split over (1 under attention DP).
        self.attn_tp_size = _pool.attention_tp_size(mapping)
        # The engine's all-reduce strategy for the TP sums; the fused IPC paths need AUTO.
        self.allreduce_strategy = int(model_config.allreduce_strategy)
        # Under attention DP each rank serves its own requests; MoE runs over the TP group's rows.
        self.dp_size = mapping.tp_size if mapping.enable_attention_dp else 1
        self.dp_rank = mapping.tp_rank
        self.dp_group = list(mapping.tp_group)
        # One-model MTP-Eagle is the only speculation routing admits; None without speculation.
        spec_config = getattr(model_config, "spec_config", None)
        self.spec_config = spec_config
        self.mtp_enabled = spec_config is not None
        self.max_num_tokens = model_config.max_num_tokens
        self.moe_max_rows = _moe_max_rows(
            self.max_num_tokens, mapping.tp_size if mapping.enable_attention_dp else 1
        )

        self.num_layers = cfg.num_hidden_layers
        self.hidden = cfg.hidden_size
        self.vocab = cfg.vocab_size
        self.eps = cfg.rms_norm_eps
        layer_types = list(cfg.layer_types)
        self.is_attn = [t == "full_attention" for t in layer_types]
        self.attn_layers = [i for i, a in enumerate(self.is_attn) if a]
        self.gdn_layers = [i for i, a in enumerate(self.is_attn) if not a]

        self.hc_count = cfg.hc_count
        self.hc_lowrank = cfg.hc_lowrank
        self.stream = self.hc_count * self.hidden
        # Columns the first projection reads: the `[down; block_inject]` rows rounded up to 32.
        self.hc_first_cols = -(-(self.hc_lowrank + self.hc_count) // 32) * 32

        # RoPE: partial, neox pairs, interleaved mRoPE.
        self.head_dim = cfg.head_dim
        rope = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None)
        self.theta = float(rope["rope_theta"])
        partial = float(rope.get("partial_rotary_factor", cfg.partial_rotary_factor))
        self.rotary_dim = int(self.head_dim * partial)
        self.mrope_declared = str(rope.get("type", rope.get("rope_type"))) == "mrope"
        section = list(rope["mrope_section"])
        self.mrope_section1 = int(section[1])
        self.mrope_section2 = int(section[2])
        self.max_positions = cfg.max_position_embeddings

        # QSA sparse index geometry, replicated on every rank.
        self.index_heads = cfg.indexer_n_heads
        self.index_kv_heads = cfg.indexer_kv_heads
        self.index_head_dim = cfg.indexer_head_dim
        self.token_topk = cfg.indexer_budget
        self.compress_ratio = cfg.indexer_compress_ratio
        self.block_topk = self.token_topk // self.compress_ratio
        # Dense-attention cutoff; must match the engine's (graph families split on it).
        self.dense_threshold = self.token_topk
        sparse_cfg = getattr(model_config, "sparse_attention_config", None)
        assert getattr(sparse_cfg, "algorithm", None) == "qsa", (
            f"this target is a QSA sparse-attention assembly and the engine resolved "
            f"sparse_attention_config={sparse_cfg!r}; QSA is this architecture's own default"
        )
        qsa_params = sparse_cfg.to_sparse_params(pretrained_config=cfg)
        assert (
            qsa_params.index_n_heads,
            qsa_params.index_kv_heads,
            qsa_params.index_head_dim,
            qsa_params.token_topk,
            qsa_params.compress_ratio,
        ) == (
            self.index_heads,
            self.index_kv_heads,
            self.index_head_dim,
            self.token_topk,
            self.compress_ratio,
        ), (
            f"the engine lowered the QSA index geometry to {qsa_params!r}; the side caches "
            "are sized from the engine's copy and the kernels are called with this one"
        )
        assert qsa_params.dense_seq_len_threshold == self.dense_threshold, (
            f"the engine's resolved QSA cutoff is {qsa_params.dense_seq_len_threshold} and "
            f"this forward branches at {self.dense_threshold}: the runner captures its short "
            "decode graph family at the engine's number"
        )
        # IndexShare for the MTP draft loop, as resolved by the engine (it sizes the buffers).
        self.mtp_index_share = bool(qsa_params.mtp_index_share)
        self.index_width = (self.index_heads + self.index_kv_heads) * self.index_head_dim
        # `[q | k | v]` scratch for the index norm + rope; the v head is required, never read.
        self.index_scratch_width = self.index_width + self.index_head_dim
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)

        # The draft layer's global index (KV managers and the weight table key it there).
        self.mtp_layer_idx = self.num_layers
        mtp_block = getattr(cfg, "mtp", None)
        self.mtp_theta = (
            float(mtp_block.get("rope_theta", self.theta))
            if isinstance(mtp_block, dict)
            else self.theta
        )

        self.head_k_dim = cfg.linear_key_head_dim
        self.head_v_dim = cfg.linear_value_head_dim
        self.conv_kernel = cfg.linear_conv_kernel_dim

        self.num_experts = cfg.num_experts
        self.topk = cfg.num_experts_per_tok
        self.moe_inter = cfg.moe_intermediate_size
        self.shared_inter = cfg.shared_expert_intermediate_size
        requested = model_config.moe_backend_requested.upper()
        assert requested in _TRUNK_MOE_BACKENDS, (
            f"moe_config.backend={model_config.moe_backend_requested!r} is not implemented by "
            f"this target; the supported requests are {sorted(_TRUNK_MOE_BACKENDS)}"
        )
        quant_algo = getattr(model_config.quant_config, "quant_algo", None)
        assert quant_algo == _QUANT_ALGO, (
            f"this target loads the block-FP8 checkpoint ({_QUANT_ALGO}); the engine resolved "
            f"quant_algo {quant_algo!r}"
        )

        # One-based in the config.
        self.ple_layer = list(cfg.ple_layer_ids)[0] - 1
        self.ngram_size = cfg.ngram_size
        self.ple_dilation = self.ngram_size
        self.ngram_vocab_divisor = cfg.make_ngram_vocab_size_divisible_by
        eos = cfg.eos_token_id
        self.eos_token_id = int(eos[0] if isinstance(eos, (list, tuple)) else eos)
        self.ple_table_dtype = _weights.PLE_TABLE_DTYPE
        # How the table is stored (`ple_host_table`); the storage is `_ple_table_host` after load.
        self.ple_host_mode = _ple_host.host_table_mode()
        self.checkpoint_dir = getattr(composite_config, "_name_or_path", None)
        self._ple_table_host = None

        vcfg = getattr(composite_config, "vision_config", None)
        assert vcfg is not None, "the composite config carries no vision_config"
        self.vision_config = vcfg
        self.image_token_id = int(composite_config.image_token_id)
        self.video_token_id = int(composite_config.video_token_id)

        self.kv_cache_quant_algo = model_config.quant_config.kv_cache_quant_algo
        assert self.kv_cache_quant_algo in (None, QuantAlgo.FP8, QuantAlgo.NVFP4), (
            f"this target stores K/V as the checkpoint's bf16, as fp8-e4m3 or as NVFP4; the "
            f"engine resolved kv_cache_quant_algo {self.kv_cache_quant_algo!r}"
        )
        self.quant_mode = {
            None: 0,
            QuantAlgo.FP8: QM_FP8_KV_CACHE,
            QuantAlgo.NVFP4: QM_NVFP4_KV_CACHE,
        }[self.kv_cache_quant_algo]

        # This rank's share of the heads and experts.
        d = _weights.MODEL_WEIGHTS.dims(self)
        self.dims = d
        self.heads_q = d.heads_q
        self.heads_kv = d.heads_kv
        self.kv_head_start = d.kv_head_start
        self.q_width = d.q_width
        self.kv_width = d.kv_width
        self.qkv_width = self.q_width + 2 * self.kv_width
        # Fused input projection row: `[q | k | v | gate | index q, k | v slot]`. The in-place
        # norm + rope only touches q and k, so every column after k counts as v to it.
        self.attn_in_width = d.attn_in_rows
        assert self.attn_in_width == self.qkv_width + self.q_width + self.index_scratch_width
        assert self.attn_in_width % self.head_dim == 0, self.attn_in_width
        self.attn_in_tail_heads = self.attn_in_width // self.head_dim - self.heads_q - self.heads_kv
        self.index_col = self.qkv_width + self.q_width
        self.num_k_heads = d.num_k_heads
        self.num_v_heads = d.num_v_heads
        self.key_dim = d.key_dim
        self.value_dim = d.value_dim
        self.conv_dim = d.conv_dim
        # Column widths of the one GDN input projection's output: [q|k|v | z | a | b].
        self.in_proj_split = d.in_proj_split
        # Its weight's rows: their sum zero-padded to `weights.IN_PROJ_ROW_ALIGN`.
        self.in_proj_rows = d.in_proj_rows
        # This rank's routed experts (trunk and draft): `[expert_offset, expert_offset +
        # local_experts)`, each at the full `moe_inter`.
        self.local_experts = d.local_experts
        self.expert_offset = d.expert_offset
        assert d.moe_inter == self.moe_inter, (d.moe_inter, self.moe_inter)

        # Every weight but the n-gram table; meta-init intercepts torch.empty.
        self.w = _weights.MODEL_WEIGHTS.declare(d)

        # Derived once after load (`post_load_weights`).
        self._gdn_gating: dict | None = None
        self._cos_sin: torch.Tensor | None = None
        self._hc_first: dict | None = None
        self._hc_norm_w1: dict | None = None
        # KV scales: fp32 [1] ones (fp8 pool), fp32 [3] ones for q, K, V (NVFP4), None (bf16).
        self.kv_scale_orig_quant: torch.Tensor | None = None
        self.kv_scale_quant_orig: torch.Tensor | None = None
        # Streams are made after load: one created inside a graph capture is not capturable.
        self._side_stream: torch.cuda.Stream | None = None
        self._decode_stream: torch.cuda.Stream | None = None
        self._capture_stream: torch.cuda.Stream | None = None
        self._ple_stream: torch.cuda.Stream | None = None
        self._side_fork: torch.cuda.Event | None = None
        self._side_join: torch.cuda.Event | None = None
        self.tp_sum = None
        self.expert_sum = None
        self._prefill: "PrefillTarget | None" = None
        self._decode: "DecodeTarget | None" = None
        # Everything derived from the KV cache manager (`_bind_cache_manager`).
        self._bound: _Bound | None = None
        self._binds = 0
        # Set by `begin_step`, cleared by `forward`.
        self._step_begun = False
        self._mtp: "_MtpWeights | None" = None
        self._mtp_cos_sin: torch.Tensor | None = None
        # The verify step's side-cache rollback, registered with the spec worker.
        self.spec_rollback: SpeculativeRollback | None = (
            SpeculativeRollback() if self.mtp_enabled else None
        )
        self._draft_bound: _DraftBound | None = None

    # ---------------------------------------------------------------- load ---

    def _fuse_hc_first_projections(self) -> None:
        """Fuse each hyper-connection's `down` and `block_inject` into one GEMM operand.

        Each is `[rows, stream]`, zero-padded to a multiple of 128 rows; the checkpoint
        parameters become row views of it."""
        w = self.w

        def fuse(*params):
            rows = sum(t.shape[0] for t in params)
            padded = -(-rows // 128) * 128
            pad = torch.zeros(
                [padded - rows, self.stream], dtype=params[0].dtype, device=params[0].device
            )
            both = torch.cat([t.data for t in params] + [pad], 0)
            start = 0
            for t in params:
                n = t.shape[0]
                t.data = both[start : start + n]
                start += n
            return both

        self._hc_first = {
            (i, tag): fuse(w[f"l{i}_{tag}_down"], w[f"l{i}_{tag}_inject"])
            for i in range(self.num_layers)
            for tag in ("ahc", "mhc")
        }
        self._hc_first["fhc"] = fuse(w["fhc_down"])
        if self.mtp_enabled:
            p = f"l{self.mtp_layer_idx}_mtp_"
            self._hc_first["mtp_ahc"] = fuse(w[f"{p}ahc_down"], w[f"{p}ahc_inject"])
            self._hc_first["mtp_mhc"] = fuse(w[f"{p}mhc_down"], w[f"{p}mhc_inject"])
            self._hc_first["mtp_fhc"] = fuse(w[f"{p}fhc_down"])
        # Each site's fp32 `(1 + norm)`, keyed by the norm's data_ptr (`_hc_mixer_rstd`).
        norms = [w[f"l{i}_{tag}_norm"] for i in range(self.num_layers) for tag in ("ahc", "mhc")]
        norms.append(w["fhc_norm"])
        if self.mtp_enabled:
            p = f"l{self.mtp_layer_idx}_mtp_"
            norms += [w[f"{p}{tag}_norm"] for tag in ("ahc", "mhc", "fhc")]
        self._hc_norm_w1 = {t.data_ptr(): (1.0 + t.float()).contiguous() for t in norms}

    def _precompile_hc_rstd(self, device) -> None:
        """Pre-compile every prefill rstd kernel variant (each K split and form) over zeros."""
        hi = self.max_num_tokens
        if hi < _HC_RSTD_MIN_ROWS:
            return
        width, hidden, rank, c = self.stream, self.hidden, self.hc_lowrank, self.hc_count
        bf = torch.bfloat16
        splits = {}
        for rows in range(_HC_RSTD_MIN_ROWS, hi + 1, 256):
            splits.setdefault(_hc_rescale_split(rows, width // _HC_RESCALE_BK, device), rows)
        splits.setdefault(_hc_rescale_split(hi, width // _HC_RESCALE_BK, device), hi)
        rows = max(splits.values())
        hc = torch.zeros([rows, width], dtype=bf, device=device)
        blk = torch.zeros([rows, hidden], dtype=bf, device=device)
        inj = torch.zeros([rows, c], dtype=bf, device=device)
        norm = torch.zeros([width], dtype=bf, device=device)
        first = torch.zeros([-(-(rank + c) // 128) * 128, width], dtype=bf, device=device).t()
        up = torch.zeros([width, rank], dtype=bf, device=device).t()
        _, rstd = qwen4_exp_hc_inject_rstd(hc, None, None, self.eps, c)
        qwen4_exp_hc_inject_rstd(hc, blk, inj, self.eps, c)
        qwen4_exp_hc_inject_rstd(hc, blk, inj, self.eps, c, blk, blk[:, :1])
        for m in sorted(splits.values()):
            low, _ = qwen4_exp_hc_first_rescale_tc(hc[:m], rstd[:m], norm, first, c, rank, True)
        w1 = (1.0 + norm.float()).contiguous()
        qwen4_exp_hc_up_gate_mean_rstd(low, up, hc[: low.shape[0]], rstd[: low.shape[0]], w1, c)
        torch.cuda.synchronize(device)

    def _precompile_draft_hc_rows(self, device) -> None:
        """Pre-compile the hyper-connection sites of eager draft passes at every row count.

        Decode graphs compile at capture, but an eager draft pass would compile while serving."""
        M = getattr(self, "_mtp", None)
        if M is None or _HC_ROWS_TC_MIN_ROWS > _HC_ROWS_TC_MAX_ROWS:
            return
        width, hidden, c = self.stream, self.hidden, self.hc_count
        bf = torch.bfloat16
        hi = min(_HC_ROWS_TC_MAX_ROWS, self.max_num_tokens)
        hc = torch.zeros([hi, width], dtype=bf, device=device)
        blk = torch.zeros([hi, hidden], dtype=bf, device=device)
        inj = torch.zeros([hi, c], dtype=bf, device=device)
        n_inject = self.hc_lowrank + c
        for rows in range(_HC_ROWS_TC_MIN_ROWS, hi + 1):
            sites = (
                (M.ahc_norm, M.ahc_first, M.ahc_up, blk[:rows], inj[:rows], n_inject),
                (M.fhc_norm, M.fhc_first, M.fhc_up, None, None, self.hc_lowrank),
            )
            for norm, first, up, block_out, inject, n_cols in sites:
                out, low, _, rstd = qwen4_exp_hc_decode_rows_first(
                    hc[:rows], block_out, inject, norm, first, c, n_cols, self.eps
                )
                qwen4_exp_hc_decode_rows_mix(low, rstd, up, out, norm, c)
        torch.cuda.synchronize(device)

    def post_load_weights(self) -> None:
        """Derive load-time tensors and build the targets; a shell hook, not an override.

        Reached twice per load (the shell's hook and the loader's module walk): the targets are
        built once, until the shell's `load_weights` clears them for a reload."""
        if self._prefill is not None:
            return
        w = self.w
        device = w["embed"].device
        # The delta-rule kernels read A_log and dt_bias as contiguous fp32; the checkpoint is bf16.
        self._gdn_gating = {
            i: (
                w[f"l{i}_A_log"].float().contiguous(),
                w[f"l{i}_dt_bias"].float().contiguous(),
            )
            for i in self.gdn_layers
        }
        # Zero the GDN in-proj alignment rows so the decode GEMM's unread pad columns are finite.
        for i in self.gdn_layers:
            w[f"l{i}_in_proj_w"][sum(self.in_proj_split) :].zero_()
        # An index past the table's end is a device-side assert that poisons the process.
        # Rows of the whole table: under `server` storage `ple_table` holds only this rank's.
        host = self._ple_table_host
        table_rows = host.rows if host is not None else w["ple_table"].shape[0]
        used_rows = int(w["ple_offs"][-1]) + int(w["ple_vocab"][-1])
        assert used_rows <= table_rows, (
            f"the n-gram table has {table_rows} rows but its per-head offsets and vocabulary "
            f"sizes address {used_rows}"
        )
        assert table_rows % self.ngram_vocab_divisor == 0, (table_rows, self.ngram_vocab_divisor)
        self._cos_sin = self._index_rope_table(self.theta, device)
        # Before the draft layer's views, which read its fused first projections.
        self._fuse_hc_first_projections()
        if self.mtp_enabled:
            # The draft layer's index rope uses the MTP module's own rope base.
            self._mtp_cos_sin = (
                self._cos_sin
                if self.mtp_theta == self.theta
                else self._index_rope_table(self.mtp_theta, device)
            )
            self._mtp = _mtp_views(self)
        if self.quant_mode in (QM_FP8_KV_CACHE, QM_NVFP4_KV_CACHE):
            # Built here, not at construction: that runs under meta init.
            scales = 3 if self.quant_mode == QM_NVFP4_KV_CACHE else 1
            self.kv_scale_quant_orig = torch.ones(scales, dtype=torch.float32, device=device)
            self.kv_scale_orig_quant = 1.0 / self.kv_scale_quant_orig
        # The CPU-only load checks build the targets over host weights; they launch nothing.
        if device.type == "cuda":
            hc_tc_first_precompile(self.stream, self.hc_lowrank, self.hc_count, device)
            self._precompile_hc_rstd(device)
            if self.mtp_enabled:
                self._precompile_draft_hc_rows(device)
            self._check_ple_residency()
            self._side_stream = torch.cuda.Stream(device=device)
            # Generation-only steps run at the highest stream priority (`_on_decode_stream`).
            self._decode_stream = torch.cuda.Stream(device=device, priority=-100)
            self._capture_stream = torch.cuda.Stream(device=device)
            self._ple_stream = torch.cuda.Stream(device=device)
            # `_moe_block`'s side-stream fork and join, re-recorded at every layer.
            self._side_fork = torch.cuda.Event()
            self._side_join = torch.cuda.Event()
            self.tp_sum = self._tp_sum_op()
            self.expert_sum = self._expert_sum_op()
            self.hc_ar_inject = self._hc_ar_inject()
        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)

    def _tp_sum_op(self):
        """Return the attention-TP sum of block partials, bit-identical on every rank."""
        if self.attn_tp_size == 1:
            return lambda x: x
        group = allreduce_group_args(self.mapping, self.allreduce_strategy)

        def tp_sum(x: torch.Tensor) -> torch.Tensor:
            return allreduce(
                x,
                None,
                None,
                scale=None,
                bias=None,
                op=int(AllReduceFusionOp.NONE),
                eps=self.eps,
                trigger_completion_at_end=False,
                **group,
            )

        if self.allreduce_strategy != int(AllReduceStrategy.AUTO) or not MNNVLAllReduce.is_mnnvl(
            self.mapping, torch.bfloat16, explicitly_requested=True
        ):
            return tp_sum
        # A collective over the TP group: every rank builds the workspace here, in order.
        mnnvl = mnnvl_workspace_args(
            self.mapping, _TP_SUM_MNNVL_MAX_ROWS, self.hidden, torch.bfloat16
        )
        ipc_sum = tp_sum

        def tp_sum(x: torch.Tensor) -> torch.Tensor:  # noqa: F811
            if x.shape[0] <= _TP_SUM_MNNVL_MAX_ROWS:
                return mnnvl_tp_sum(x, **mnnvl)
            return ipc_sum(x)

        return tp_sum

    def _hc_ar_inject(self):
        """Return the decode sites' fused TP-sum + injection state, or None where unsupported."""
        if not (
            self.attn_tp_size in (2, 4)
            and self.dp_size == 1
            and self.mapping.world_size == self.attn_tp_size
            and self.hidden == 2560
            and self.hc_count == 4
            and self.allreduce_strategy == int(AllReduceStrategy.AUTO)
            and can_access_peer(self.mapping)
        ):
            return None
        group = allreduce_group_args(self.mapping, self.allreduce_strategy)
        if group["workspace"] is None:
            return None
        members = list(self.mapping.tp_group)
        rank = members.index(self.mapping.rank)
        # `get_allreduce_workspace` sized each IPC sub-buffer `size * tp_size` with this call;
        # the fused kernel's span in one is `nranks * rows * hidden * 2` bytes.
        size = CustomAllReduceHelper.max_workspace_size_auto(
            self.mapping.tp_size, support_deterministic=False
        )
        max_rows = size * self.mapping.tp_size // (self.hidden * 2 * len(members))
        return _HcArInject(group["workspace"], rank, len(members), max_rows)

    def _expert_sum_op(self):
        """Return `expert_sum(run, x, padding)`: this rank's rows of a MoE block's output.

        `run(rows)` applies this rank's expert window; `padding` comes from `_dp_pad_rows`.
        Attention TP: every rank holds every row and `tp_sum` completes the partial. Attention
        DP: all-gather the group's rows in rank order, run the window, reduce-scatter; routing
        gathered rows keeps every rank's routing of a row identical.
        `MOE_COMM_METHOD_ENV=NVLINK_ONE_SIDED` (MPI and MNNVL) swaps in the NVLink all-to-all,
        with `run` a `_WindowRun`. Unequal per-rank
        counts (`padding` set) use the ragged collectives so the window never sees padding
        rows, which would all route to the same experts and stall the group on one window."""
        if self.dp_size == 1:
            tp_sum = self.tp_sum

            def window_sum(run, x: torch.Tensor, padding: _DpPadding | None) -> torch.Tensor:
                return tp_sum(run(x))

            return window_sum
        group = self.dp_group
        # With MPI disabled the collectives run over the TP process group.
        process_group = self.mapping.tp_group_pg.boxed() if mpi_disabled() else None

        def round_trip(run, x: torch.Tensor, padding: _DpPadding | None) -> torch.Tensor:
            sizes = None if padding is None else list(padding.counts)
            gathered = allgather(x, sizes, group, process_group)
            return reducescatter(run(gathered), sizes, group, process_group)

        if _moe_comm_method() != "NVLINK_ONE_SIDED":
            return round_trip
        strategy = None if mpi_disabled() else _build_moe_a2a(self.mapping, self.max_num_tokens)
        if strategy is None:
            # `check_deployment` declines this deployment; reached only when routing is bypassed.
            raise RuntimeError(
                f"{MOE_COMM_METHOD_ENV}=NVLINK_ONE_SIDED needs MPI and MNNVL, which this "
                "deployment lacks; unset it to run the attention-DP MoE as the all-gather / "
                "reduce-scatter round trip"
            )
        self.moe_a2a = strategy

        def alltoall_round_trip(run, x: torch.Tensor, padding: _DpPadding | None) -> torch.Tensor:
            counts = [x.shape[0]] * len(group) if padding is None else list(padding.counts)
            topk_ids, topk_w = _route_rows(run.core, x, run.router, run.mm, run.pad_mask)
            got = moe_a2a_dispatch(strategy, x, topk_ids, topk_w, counts)
            partials = _window_prerouted(
                run.core, got.rows, got.topk_ids, got.topk_weights, run.experts
            )
            return moe_a2a_combine(strategy, partials)

        return alltoall_round_trip

    def _dp_pad_rows(self, md, num_tokens: int) -> _DpPadding | None:
        """Return the group's per-rank token counts for the ragged MoE round trip, or None.

        None without attention DP or with equal counts, as in every captured step."""
        if self.dp_size == 1:
            return None
        counts = tuple(int(n) for n in md.all_rank_num_tokens)
        assert len(counts) == self.dp_size, (counts, self.dp_size)
        assert counts[self.dp_rank] == num_tokens, (
            f"rank {self.dp_rank} holds {num_tokens} rows but all_rank_num_tokens says "
            f"{counts[self.dp_rank]} ({list(counts)}); the collectives would gather the wrong "
            "split"
        )
        if min(counts) == max(counts):
            return None
        return _DpPadding(counts)

    def _index_rope_table(self, theta: float, device) -> torch.Tensor:
        """Index rope table at `theta`: `[max_positions, 2 * rotary_pairs]` fp32, cos then sin."""
        exponent = torch.arange(0, self.rotary_dim, 2, dtype=torch.float64, device=device)
        inv_freq = (1.0 / (theta ** (exponent / self.rotary_dim))).to(torch.float32)
        positions = torch.arange(self.max_positions, dtype=torch.float32, device=device)
        angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=1).contiguous()

    def _check_ple_residency(self) -> None:
        """Assert via the driver that the n-gram table is host-resident; outputs cannot show it."""
        host = self._ple_table_host
        assert host.resident_on_host(), (
            f"the n-gram table ({host.describe()}) was declared in host memory, but the driver "
            "reports device memory for the pointer the forward gathers from"
        )
        logger.info(f"ple n-gram table: {host.describe()}")

    # ---------------------------------------------------------------- bind ---

    def _bind_cache_manager(self, md, position_ids) -> None:
        """Bind this rank's pool views of the step's KV cache manager and check its geometry.

        Keyed on the manager, not a first-forward flag: capacity estimation builds a throwaway
        manager before the real one, and a view held across the swap would dangle."""
        cm = md.kv_cache_manager
        assert cm is not None, "a paged KV cache is required"
        assert not (self._binds and torch.cuda.is_current_stream_capturing()), (
            f"the KV cache manager changed to {type(cm).__name__} during CUDA graph capture"
        )
        self._binds += 1
        missing = [name for name in _QSA_FIELDS if not hasattr(md, name)]
        assert not missing, (
            f"metadata fields missing: {missing}; the engine did not select QSA metadata"
        )
        assert position_ids.dtype == torch.int32, position_ids.dtype
        assert self.mrope_declared, "this target takes the mRoPE position layout"
        assert position_ids.dim() == 3 and position_ids.shape[0] == _QSA_POSITION_AXES, (
            f"expected the [{_QSA_POSITION_AXES}, 1, T] mRoPE position buffer, got "
            f"{tuple(position_ids.shape)}"
        )
        pools = {row[0] for row in md.host_kv_cache_pool_mapping.tolist()}
        assert pools <= {0, 1}, (
            f"KV addressing over {len(pools)} pools is not certified; layer->pool ids "
            f"{sorted(pools)}"
        )
        g = _pool.pool_geometry(
            self.model_config.pretrained_config,
            self.mapping,
            tokens_per_block=md.tokens_per_block,
            kv_cache_quant_algo=self.kv_cache_quant_algo,
            dtype=self.w["embed"].dtype,
        )
        assert (g.heads_kv, g.kv_head_start, g.num_k_heads, g.num_v_heads, g.conv_dim) == (
            self.heads_kv,
            self.kv_head_start,
            self.num_k_heads,
            self.num_v_heads,
            self.conv_dim,
        ), (
            f"the KV cache manager's share {g} disagrees with the weights' heads "
            f"(kv {self.heads_kv} from {self.kv_head_start}, GDN {self.num_k_heads} / "
            f"{self.num_v_heads}, conv {self.conv_dim})"
        )
        assert md.qsa_topk_indices.shape[1] == self.block_topk, (
            f"the engine sized the QSA selection buffer {md.qsa_topk_indices.shape[1]} wide "
            f"and this target selects {self.block_topk} blocks per row"
        )
        mm = md.mamba_metadata
        assert mm is not None and mm is not False, (
            "attn_metadata.mamba_metadata is absent: the runtime allocated no recurrent-state pool"
        )
        mm_missing = [name for name in _MAMBA_FIELDS if not hasattr(mm, name)]
        assert not mm_missing, f"mamba metadata fields missing: {mm_missing}"
        rank_pools = _pool.take_pools(cm, g, self.attn_layers, self.gdn_layers, self.ple_layer)
        # No sliding window: the targets pass `max_seq_len` as the window.
        windows = cm.max_attention_window_vec
        attn_windows = {windows[off] for off in rank_pools.pool_idx.values()}
        assert attn_windows <= {None, md.max_seq_len}, (
            f"the attention layers' pool windows are {sorted(map(repr, attn_windows))}; a "
            "sliding window is not implemented here"
        )
        assert md.effective_beam_width == 1, "beam search is not implemented"
        assert md.cache_indirection is None, "beam search is not implemented"
        assert not md.is_cross, "cross attention is not implemented"
        # Printed, not logged: the executor's workers keep the info log level off.
        first_attn, first_gdn = self.attn_layers[0], self.gdn_layers[0]
        print(
            f"{_LOG_PREFIX}: rank {self.mapping.rank} binds "
            f"{rank_pools.kv[first_attn].k.shape[0]} K/V pages, "
            f"{rank_pools.index_cache[first_attn].shape[0]} QSA index pages, "
            f"{rank_pools.conv[first_gdn].shape[0]} recurrent-state slots and "
            f"{rank_pools.ple_state[0].shape[0]} PLE slots (bind {self._binds})",
            flush=True,
        )
        bound = _Bound(cm, rank_pools, stage=_inforward_stage(cm, self.gdn_layers, self.ple_layer))
        if self.spec_rollback is not None:
            self.spec_rollback.set_index_planes(
                [rank_pools.index_cache[i] for i in self.attn_layers]
            )
        if self.mtp_enabled:
            bound = bound._replace(**self._bind_step_buffers(cm, g, rank_pools, md))
        self._bound = bound

    def _bind_step_buffers(self, cm, g: _pool.PoolGeometry, pools: _pool.RankPools, md) -> dict:
        """Bind the verify step's per-step buffers and return the `_Bound` fields they add.

        Per-row conv windows feed `update_mamba_states`; the recurrent state uses cached replay."""
        _pool.require(
            cm,
            ("is_speculative", "mamba_layer_cache", "get_replay_state_update_metadata"),
            "binds the verify step's per-step state buffers and replay metadata",
        )
        assert cm.is_speculative(), (
            f"the KV cache manager {type(cm).__name__} reports is_speculative() False, so it "
            "allocated none of the per-step state buffers the verify path writes"
        )
        layer_caches = {i: cm.mamba_layer_cache(i) for i in self.gdn_layers}
        int_conv = {i: cache.intermediate_conv_window for i, cache in layer_caches.items()}
        int_idx = getattr(cm, "intermediate_state_indices", None)
        g0 = self.gdn_layers[0]
        _pool.check_step_buffers(g, int_conv[g0], pools.conv[g0], int_idx)
        meta = cm.get_replay_state_update_metadata()
        if meta is None:
            raise NotImplementedError(
                "the KV cache manager runs the legacy per-step recurrent verify path "
                "(TRTLLM_USE_GDN_REPLAY=0), which this target does not assemble; leave "
                "TRTLLM_USE_GDN_REPLAY unset for the cached replay"
            )
        assert not getattr(cm, "use_gdn_flashinfer_ring_replay", False), (
            "the manager laid its replay history out for the FlashInfer ring kernel; this "
            "target runs the double-buffered Triton cached replay"
        )
        assert layer_caches[g0].intermediate_ssm is None, (
            "the cache manager runs the cached-replay verify path and also allocated a "
            "per-step recurrent buffer; the runtime allocates one or the other"
        )
        ssm0 = pools.ssm[g0]
        history, width = meta.replay_history_size, meta.replay_step_width
        _pool.check_replay_state(
            history, width, meta.prev_num_accepted_tokens, meta.cache_buf_idx, ssm0
        )
        old = {}
        for i, cache in layer_caches.items():
            old[i] = (cache.old_x, cache.old_B, cache.old_dt, cache.old_dA_cumsum)
            _pool.check_replay_history(g, i, history, ssm0.shape[0], *old[i][:3])
        replay = _Replay(old, meta.cache_buf_idx, meta.prev_num_accepted_tokens, history, width)
        if ssm0.dtype == torch.bfloat16 and ssm0.is_cuda:
            # The CuTe replay kernels of the default mapping a verify step launches, compiled
            # here: a compile costs 5-10 s, and the padded decode graphs skip the request counts
            # an eager mixed step carries. PDL up to `_GDN_REPLAY_PDL_MAX_HEAD_ROWS`
            # (request x value head) pairs.
            heads = (width, self.num_k_heads, self.num_v_heads)
            partition_from = CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE
            requests = md.max_num_requests
            if heads[1:] == _GDN_REPLAY_HT_GEOMETRY:
                requests = min(requests, _GDN_REPLAY_HT_MIN_REQUESTS - 1)
            pdl_requests = min(requests, _GDN_REPLAY_PDL_MAX_HEAD_ROWS // self.num_v_heads)
            if get_env_enable_pdl() and pdl_requests > 0:
                gdn_replay_precompile(*heads, pdl_requests, pdl=True, partition_from=partition_from)
            gdn_replay_precompile(*heads, requests, pdl=False, partition_from=partition_from)
            if (
                heads[1:] == _GDN_REPLAY_HT_GEOMETRY
                and md.max_num_requests >= _GDN_REPLAY_HT_MIN_REQUESTS
            ):
                # A tp2/tep2 HT batch takes its own mapping and conv fold (`_gdn_replay_launch`,
                # `_gdn_verify_folds_conv`). The decode warmup compiles its kernels with the norm
                # folded (`_gdn_verify_folds_norm`); a mixed step's verify rows
                # (`PrefillTarget._gdn_core`) launch them without, compiled here.
                gdn_replay_precompile(
                    *heads,
                    md.max_num_requests,
                    pdl=False,
                    partition_from=partition_from,
                    min_requests=_GDN_REPLAY_HT_MIN_REQUESTS,
                    launch=lambda n: _gdn_replay_launch(self, n),
                    conv_fold=lambda n: _gdn_verify_folds_conv(self, n),
                )
        return dict(int_idx=int_idx, int_conv=int_conv, replay=replay)

    def _bind_draft_cache_manager(self, md) -> None:
        """Bind the draft layer's index plane, position plane and K/V in its step's manager.

        That is the separate draft KV cache under attention TP, else the trunk's manager (also
        under disaggregated serving). The runner runs every captured shape eagerly first, so
        the first bind never happens under capture."""
        cm = md.kv_cache_manager
        assert not torch.cuda.is_current_stream_capturing(), (
            f"the draft layer is not bound to KV cache manager {type(cm).__name__} (first use, "
            "or the manager changed since its last bind) and CUDA graph capture is active"
        )
        g = _pool.pool_geometry(
            self.model_config.pretrained_config,
            self.mapping,
            tokens_per_block=md.tokens_per_block,
            kv_cache_quant_algo=self.kv_cache_quant_algo,
            dtype=self.w["embed"].dtype,
        )
        layer = self.mtp_layer_idx
        draft = _pool.take_draft_pools(
            cm, g, layer, separate=_pool.draft_kv_cache_is_separate(self.mapping, self.spec_config)
        )
        index_cache, position_cache, kv = draft.index_cache, draft.position_cache, draft.kv
        share_slot = None
        if self.mtp_index_share:
            assert md.sparse_metadata_params.mtp_index_share, (
                "this target resolved IndexShare on but the engine's QSA metadata did not"
            )
            slots = cm.qsa_shared_topk_slots
            _pool.check_shared_selection(md.qsa_shared_topk_indices, slots, layer, self.block_topk)
            share_slot = slots[layer]
        self._draft_bound = _DraftBound(cm, index_cache, position_cache, kv, share_slot)

    def _draft_step_bind(self, md) -> tuple[_DraftBound, int]:
        """Return the draft layer's bind and pool offset for this draft step's manager.

        Checked every call: the worker swaps managers and page tables in for the draft loop."""
        cm = md.kv_cache_manager
        assert cm is not None, "a paged KV cache is required"
        offsets = cm.layer_offsets
        assert self.mtp_layer_idx in offsets, (
            f"the MTP module's attention is global layer index {self.mtp_layer_idx}, which the "
            f"cache manager {type(cm).__name__} does not map (it maps {len(offsets)} layers); "
            "the engine allocated no KV layer for the draft module"
        )
        layer_idx = offsets[self.mtp_layer_idx]
        window = cm.max_attention_window_vec[layer_idx]
        assert window is None or window == md.max_seq_len, (
            f"the MTP layer's pool window is {window!r} but this call passes max_seq_len "
            f"{md.max_seq_len} as the attention window; a sliding window is not implemented"
        )
        if self._draft_bound is None or cm is not self._draft_bound.cm:
            self._bind_draft_cache_manager(md)
        assert cm is self._bound.cm or md.qsa_block_table is getattr(
            md, "qsa_draft_block_table", None
        ), (
            "the draft step is bound to a separate draft KV cache manager but qsa_block_table "
            "is not that manager's table: the engine's QSA metadata did not swap its page "
            "table for the draft forward"
        )
        return self._draft_bound, layer_idx

    # -------------------------------------------------------------- forward ---

    def _select_target(self, attn_metadata) -> "PrefillTarget | DecodeTarget":
        return self._decode if attn_metadata.num_contexts == 0 else self._prefill

    def begin_step(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None,
        position_ids: torch.IntTensor | None,
        inputs_embeds: torch.FloatTensor | None = None,
        lora_params: dict | None = None,
        spec_metadata=None,
        vision: tuple | None = None,
    ) -> None:
        """Validate the step, bind the cache manager and let the chosen target prepare it.

        All host reads happen here: a breakable prefill CUDA graph replays `forward` without
        calling it. `vision` is `(rows, mm_token_indices)` when the step carries media."""
        assert self._prefill is not None and self._decode is not None, (
            "load_weights must run before forward"
        )
        assert position_ids is not None
        assert isinstance(attn_metadata, TrtllmAttentionMetadata)
        assert lora_params is None, "LoRA is not implemented by this target"
        assert input_ids is not None, "this target's PLE layer hashes token ids"
        assert inputs_embeds is None, (
            "this target computes its token embedding itself; an engine-supplied "
            "inputs_embeds is not implemented"
        )
        # One-model MTP-Eagle captures no trunk layer, so the trunk owes no capture hook.
        assert (spec_metadata is not None) == self.mtp_enabled, (
            f"spec_metadata {'arrived' if spec_metadata is not None else 'is absent'} but the "
            f"model was built with mtp_enabled={self.mtp_enabled}; the draft path and the "
            "weights it loads are decided at construction"
        )
        if self._bound is None or attn_metadata.kv_cache_manager is not self._bound.cm:
            self._bind_cache_manager(attn_metadata, position_ids)
        target = self._select_target(attn_metadata)
        if target is self._prefill:
            target.prepare(attn_metadata, input_ids, position_ids, vision, spec_metadata)
        else:
            assert vision is None, "a generation-only step carries vision rows"
        self._step_begun = True

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        lora_params: dict | None = None,
        **kwargs,
    ):
        """Run the step `begin_step` prepared; nothing here may read the step from the host.

        Returns the logits input, or under speculation what `split_spec_output` takes."""
        assert self._step_begun, (
            "the shell calls begin_step before every forward; this forward has no prepared step"
        )
        self._step_begun = False
        target = self._select_target(attn_metadata)
        step = target.forward
        if isinstance(target, DecodeTarget):
            return _on_decode_stream(
                self,
                input_ids.shape[0],
                step,
                attn_metadata,
                input_ids=input_ids,
                position_ids=position_ids,
                **kwargs,
            )
        return step(attn_metadata, input_ids=input_ids, position_ids=position_ids, **kwargs)

    def split_spec_output(self, output) -> tuple[torch.Tensor, torch.Tensor]:
        """Split speculative `forward` output into `final` `[T, hidden]` and `hc` `[T, stream]`.

        `final` feeds `lm_head`, `hc` the MTP module; the prefill target packs both in one."""
        if isinstance(output, tuple):
            return output
        final, hc = torch.split(output, [self.hidden, self.stream], -1)
        return final, hc


# --------------------------------------------------------------- weight views ---


class _LayerWeights(NamedTuple):
    """One trunk layer's op views (GEMM operands `.t()`); unused attention/GDN fields are None."""

    ahc_norm: torch.Tensor
    ahc_first: torch.Tensor
    ahc_up: torch.Tensor
    mhc_norm: torch.Tensor
    mhc_first: torch.Tensor
    mhc_up: torch.Tensor
    attn_in: torch.Tensor | None
    o: torch.Tensor | None
    q_norm: torch.Tensor | None
    k_norm: torch.Tensor | None
    idx_qn: torch.Tensor | None
    idx_kn: torch.Tensor | None
    in_proj: torch.Tensor | None
    # With the zero alignment rows (`core.in_proj_rows` columns) the decode GEMM tiles whole;
    # `in_proj` is its leading `sum(in_proj_split)` columns.
    in_proj_dec: torch.Tensor | None
    out: torch.Tensor | None
    conv_w: torch.Tensor | None
    a_log: torch.Tensor | None
    dt_bias: torch.Tensor | None
    gnorm: torch.Tensor | None
    router: torch.Tensor
    experts: _weights.ExpertWindow
    # `[gate | up | gate logit | pad]` (`weights._cat_rows`), and the down projection.
    sh_gug: torch.Tensor
    sh_dn: torch.Tensor


def _layer_views(core: Qwen4ExpUnifiedModelingV2) -> list[_LayerWeights]:
    w = core.w
    layers = []
    for i in range(core.num_layers):
        attn: dict = dict(attn_in=None, o=None, q_norm=None, k_norm=None, idx_qn=None, idx_kn=None)
        gdn: dict = dict(
            in_proj=None,
            in_proj_dec=None,
            out=None,
            conv_w=None,
            a_log=None,
            dt_bias=None,
            gnorm=None,
        )
        if core.is_attn[i]:
            attn = dict(
                attn_in=w[f"l{i}_attn_in_w"].t(),
                o=w[f"l{i}_o_w"].t(),
                q_norm=w[f"l{i}_q_norm"],
                k_norm=w[f"l{i}_k_norm"],
                idx_qn=w[f"l{i}_idx_qn"],
                idx_kn=w[f"l{i}_idx_kn"],
            )
        else:
            a_log, dt_bias = core._gdn_gating[i]
            gdn = dict(
                in_proj=w[f"l{i}_in_proj_w"][: sum(core.in_proj_split)].t(),
                in_proj_dec=w[f"l{i}_in_proj_w"].t(),
                out=w[f"l{i}_out_w"].t(),
                # [conv_dim, 1, width] -> [conv_dim, width]: a view (unit width stride).
                conv_w=w[f"l{i}_conv_w"].reshape(core.conv_dim, core.conv_kernel),
                a_log=a_log,
                dt_bias=dt_bias,
                gnorm=w[f"l{i}_gnorm"],
            )
        layers.append(
            _LayerWeights(
                ahc_norm=w[f"l{i}_ahc_norm"],
                ahc_first=core._hc_first[(i, "ahc")].t(),
                ahc_up=w[f"l{i}_ahc_up"].t(),
                mhc_norm=w[f"l{i}_mhc_norm"],
                mhc_first=core._hc_first[(i, "mhc")].t(),
                mhc_up=w[f"l{i}_mhc_up"].t(),
                **attn,
                **gdn,
                router=w[f"l{i}_router"].t(),
                experts=_weights.expert_window(w, core.dims, i),
                sh_gug=w[f"l{i}_sh_gu"].t(),
                sh_dn=w[f"l{i}_sh_dn"].t(),
            )
        )
    return layers


class _StackWeights(NamedTuple):
    """Weights outside the layer loop: the stack-final mixer (final norm) and the PLE path."""

    fhc_norm: torch.Tensor
    fhc_first: torch.Tensor
    fhc_up: torch.Tensor
    ple_key: torch.Tensor
    ple_val: torch.Tensor
    ple_nk_w: torch.Tensor
    ple_nq_w: torch.Tensor
    ple_nc_w: torch.Tensor
    ple_conv_w: torch.Tensor
    ple_mult: torch.Tensor
    ple_vocab: torch.Tensor
    ple_offs: torch.Tensor
    ple_tscale: torch.Tensor


def _stack_views(core: Qwen4ExpUnifiedModelingV2) -> _StackWeights:
    w = core.w
    return _StackWeights(
        fhc_norm=w["fhc_norm"],
        fhc_first=core._hc_first["fhc"].t(),
        fhc_up=w["fhc_up"].t(),
        ple_key=w["ple_key_w"].t(),
        ple_val=w["ple_val_w"].t(),
        # Whole norm weights: the fused tail normalizes all three streams in one program.
        ple_nk_w=w["ple_nk"],
        ple_nq_w=w["ple_nq"],
        ple_nc_w=w["ple_nc"],
        ple_conv_w=w["ple_conv_w"],
        ple_mult=w["ple_mult"],
        ple_vocab=w["ple_vocab"],
        ple_offs=w["ple_offs"],
        # The table's dequantize scale at the gathered rows' rank (3), all 1s.
        ple_tscale=w["ple_tscale"].reshape(1, 1, 1),
    )


class _MtpWeights(NamedTuple):
    """The MTP module's views at this rank's share, keyed at layer `num_hidden_layers`.

    Laid out as the trunk's (GEMM operands `.t()`); `unit_inject` adds `x_e` as a site injection."""

    pre_e_norm: torch.Tensor
    pre_h_norm: torch.Tensor
    fc_e: torch.Tensor
    fc_h: torch.Tensor
    ahc_norm: torch.Tensor
    ahc_first: torch.Tensor
    ahc_up: torch.Tensor
    mhc_norm: torch.Tensor
    mhc_first: torch.Tensor
    mhc_up: torch.Tensor
    unit_inject: torch.Tensor
    # [hidden, attn_in_width]: the trunk's fused attention input layout, at this rank's heads.
    attn_in: torch.Tensor
    o: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    idx_qn: torch.Tensor
    idx_kn: torch.Tensor
    router: torch.Tensor
    experts: _weights.ExpertWindow
    sh_gug: torch.Tensor
    sh_dn: torch.Tensor
    fhc_norm: torch.Tensor
    fhc_first: torch.Tensor
    fhc_up: torch.Tensor


def _mtp_views(core: Qwen4ExpUnifiedModelingV2) -> _MtpWeights:
    """Built after `_fuse_hc_first_projections`, whose `"mtp_*"` tensors it reads."""
    w = core.w
    p = f"l{core.mtp_layer_idx}_mtp_"
    dt = w[f"{p}fc_h"].dtype
    return _MtpWeights(
        pre_e_norm=w[f"{p}pre_e_norm"],
        pre_h_norm=w[f"{p}pre_h_norm"],
        fc_e=w[f"{p}fc_e"].t(),
        fc_h=w[f"{p}fc_h"].t(),
        ahc_norm=w[f"{p}ahc_norm"],
        ahc_first=core._hc_first["mtp_ahc"].t(),
        ahc_up=w[f"{p}ahc_up"].t(),
        mhc_norm=w[f"{p}mhc_norm"],
        mhc_first=core._hc_first["mtp_mhc"].t(),
        mhc_up=w[f"{p}mhc_up"].t(),
        unit_inject=torch.ones(
            [core.max_num_tokens, core.hc_count], dtype=dt, device=w[f"{p}fc_h"].device
        ),
        attn_in=w[f"{p}attn_in_w"].t(),
        o=w[f"{p}o_w"].t(),
        q_norm=w[f"{p}q_norm"],
        k_norm=w[f"{p}k_norm"],
        idx_qn=w[f"{p}idx_qn"],
        idx_kn=w[f"{p}idx_kn"],
        router=w[f"{p}router"].t(),
        experts=_weights.expert_window(w, core.dims, core.mtp_layer_idx),
        sh_gug=w[f"{p}sh_gu"].t(),
        sh_dn=w[f"{p}sh_dn"].t(),
        fhc_norm=w[f"{p}fhc_norm"],
        fhc_first=core._hc_first["mtp_fhc"].t(),
        fhc_up=w[f"{p}fhc_up"].t(),
    )


# ------------------------------------------------------------ shared blocks ---


def _verify_rows(core, spec_metadata, num_seqs: int, num_rows: int) -> tuple[int, bool]:
    """Return `(p, is_verify)`: rows per generation sequence and whether the step verifies.

    `p` is 1 without speculation, else `runtime_draft_len + 1`; host ints, constant per graph."""
    p = num_rows // num_seqs if num_seqs else 1
    assert p * num_seqs == num_rows, (
        f"{num_rows} generation rows do not divide over {num_seqs} generation sequences; a "
        "ragged per-sequence draft length can be expressed neither to the attention call nor "
        "to the multi-token recurrence"
    )
    assert core.mtp_enabled or p == 1, (
        f"{num_rows} generation rows over {num_seqs} generation sequences with speculation "
        "off; the base path implements one query token per generation sequence"
    )
    if spec_metadata is not None and num_seqs > 0:
        declared = spec_metadata.runtime_draft_len + 1
        assert p == declared, (
            f"the batch carries {p} query rows per generation sequence but the worker's "
            f"runtime_draft_len says {declared}; the per-step conv snapshots and the replay "
            "history are sized from the second and written from the first"
        )
        _pool.check_verify_width(p)
        _pool.check_replay_width(core._bound.replay, p)
    return p, spec_metadata is not None and num_seqs > 0 and p > 1


def _verify_state(core, mm, num_decodes: int) -> tuple:
    """Return what `_gdn_verify` reads besides the layer: promotion rows and replay work items.

    Work items and the all-layer commit follow the stock Gated-DeltaNet layer's conditions."""
    assert core._bound.int_idx is not None, "the cache-manager bind has not run"
    int_idx = core._bound.int_idx[:num_decodes]
    partitioned = num_decodes >= CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE
    all_layer = partitioned and bool(
        getattr(core._bound.cm, "use_gdn_cached_replay_all_layer_commit", False)
    )
    work_items = mm.replay_work_items[:num_decodes] if partitioned else None
    n_writes = mm.replay_n_writes if partitioned else None
    return int_idx, work_items, n_writes, all_layer


def _gdn_in_proj_conv_tactic(core, x, L, si_d, verify):
    """Return the fused GDN in-proj + conv tactic for a verify step, or None if unsupported.

    The fused kernel is bitwise `low_m_bf16_mm` + `causal_conv1d_token_major`, one launch fewer."""
    rows = x.shape[0]
    w = L.in_proj_dec
    if (
        core.conv_dim != _GDN_IN_PROJ_CONV_DIM
        or tuple(w.shape) != (x.shape[1], _GDN_IN_PROJ_N_OUT)
        or w.stride(0) != 1
        or not x.is_contiguous()
        or si_d.dtype != torch.int32
        or verify[0].dtype != torch.int32
        or not L.conv_w.is_contiguous()
        or tuple(L.conv_w.shape) != (_GDN_IN_PROJ_CONV_DIM, 4)
    ):
        return None
    return _gdn_in_proj_conv_fused_tactic(rows)


def _gdn_verify(core, mq, a, b, i, L, si_d, n, p, verify, mq_out=None, z=None) -> torch.Tensor:
    """Run the recurrent half of a verify step: `n` generation sequences of `p` rows.

    The conv also writes each row's window to the per-step buffer `update_mamba_states`
    promotes from; the cached replay rebuilds each slot's state from its checkpoint and
    history, runs the `p` rows and appends them, so rejected rows cost nothing to undo. `mq` is
    the `[n * p, conv_dim]` q|k|v view, `a`/`b` the raw gate columns (the kernel applies
    `A_log`, `dt_bias`, q/k L2 norm). Returns `[1, n * p, num_v_heads, head_v_dim]`."""
    int_idx, work_items, n_writes, all_layer = verify
    pools = core._bound.pools
    replay = core._bound.replay
    ssm = pools.ssm[i]
    assert (core.num_k_heads, core.num_v_heads) in _GDN_REPLAY_CUTE_GEOMETRIES, (
        f"GDN heads {core.num_k_heads} / {core.num_v_heads} are not a CuTe replay geometry"
    )
    pdl, mapping = _gdn_replay_launch(core, n)
    fold = None
    if mq_out is not None:
        # `gdn_in_proj_conv` already ran the conv and advanced the pool and windows.
        pass
    elif _gdn_verify_folds_conv(core, n) and gdn_conv_fold_supported(
        mq, L.conv_w, pools.conv[i], core._bound.int_conv[i]
    ):
        # The replay computes the conv rows itself from the q|k|v columns (`mq`).
        fold = dict(
            x=mq,
            weight=L.conv_w,
            pool=pools.conv[i],
            window=core._bound.int_conv[i],
            window_rows=int_idx,
        )
        mq_out = mq
    else:
        # Token-major `[n*p, C]` conv output, which the replay reads q/k/v from directly.
        mq_out = causal_conv1d_token_major(
            mq,
            n,
            pools.conv[i],
            L.conv_w,
            si_d,
            core._bound.int_conv[i],
            int_idx,
            _PAD_SLOT_ID,
            launch_dependent_kernels=pdl,
            wait_producer=_GDN_CONV_PDL_WAIT,
        )
    norm = None
    if z is not None:
        # The gated RMS norm folded in (`_gdn_verify_folds_norm`, decided by the caller).
        norm = dict(
            z=torch.reshape(z, [n * p, core.num_v_heads, core.head_v_dim]),
            weight=L.gnorm,
            eps=core.eps,
        )
    old_x, old_B, old_dt, old_dA = replay.old[i]
    qv, kv, vv = torch.split(mq_out, [core.key_dim, core.key_dim, core.value_dim], 1)
    out = gdn_cached_replay_cute(
        q=torch.reshape(qv, [n, p, core.num_k_heads, core.head_k_dim]),
        k=torch.reshape(kv, [n, p, core.num_k_heads, core.head_k_dim]),
        v=torch.reshape(vv, [n, p, core.num_v_heads, core.head_v_dim]),
        g=torch.reshape(a, [n, p, core.num_v_heads]),
        beta=torch.reshape(b, [n, p, core.num_v_heads]),
        ssm_states=ssm,
        state_indices=si_d,
        old_u=old_x,
        old_k=old_B,
        old_G=old_dt,
        old_beta=old_dA,
        cache_buf_idx=replay.cache_buf_idx,
        prev_num_accepted_tokens=replay.prev_num_accepted,
        history_size=replay.history,
        scale=None,
        use_qk_l2norm_in_kernel=True,
        A_log=L.a_log,
        dt_bias=L.dt_bias,
        launch_with_pdl=pdl,
        replay_work_items=work_items,
        n_writes=n_writes,
        block_v=None,
        num_warps=None,
        use_l2_state_cache=None,
        use_l2_streaming_inputs=None,
        use_l2_shared_inputs=None,
        main_num_stages=None,
        packed_qkv=None if fold is not None else mq_out,
        use_all_layer_commit=all_layer,
        output=None,
        **({"config": mapping} if mapping is not None else {}),
        **({"conv_fold": fold} if fold is not None else {}),
        **({"gated_norm": norm} if norm is not None else {}),
    )
    return torch.reshape(out, [1, n * p, core.num_v_heads, core.head_v_dim])


def _gdn_verify_folds_norm(core, n: int, p: int) -> bool:
    """Whether a verify step folds its gated RMS norm into the CuTe replay (bitwise equal).

    Only at tp2/tep2 HT batches (up to 4 rows a request); saves a launch and an HBM round trip."""
    return (
        (core.num_k_heads, core.num_v_heads) == _GDN_REPLAY_HT_GEOMETRY
        and n >= _GDN_REPLAY_HT_MIN_REQUESTS
        and p <= 4
    )


def _gdn_verify_folds_conv(core, n: int) -> bool:
    """Whether a verify step folds its conv into the CuTe replay (tp2/tep2 HT batches)."""
    return (
        (core.num_k_heads, core.num_v_heads) == _GDN_REPLAY_HT_GEOMETRY
        and _GDN_REPLAY_HT_MIN_REQUESTS <= n <= _GDN_VERIFY_CONV_FOLD_MAX_REQUESTS
    )


def _gdn_replay_launch(core, n: int) -> tuple:
    """Return `(pdl, mapping)` for the CuTe cached replay over `n` requests."""
    heads = (core.num_k_heads, core.num_v_heads)
    if heads == _GDN_REPLAY_HT_GEOMETRY and n >= _GDN_REPLAY_HT_MIN_REQUESTS:
        for bound, pdl, mapping in _GDN_REPLAY_HT_LAUNCH:
            if n <= bound:
                return pdl, mapping
    return get_env_enable_pdl() and n * core.num_v_heads <= _GDN_REPLAY_PDL_MAX_HEAD_ROWS, None


def _draft_positions(position_ids: torch.Tensor) -> tuple:
    """Return `(pos3, pos, coords)` from `[3, 1, N]` or `[3, N]` draft-step mRoPE positions.

    `pos3` is contiguous int32 `[3, N]` (the input itself if already so; nothing writes it
    during the step), `pos` its row 0 and `coords` the `[N, 3]` transpose the plane stores."""
    assert position_ids.shape[0] == _QSA_POSITION_AXES, (
        f"expected the mRoPE position layout with {_QSA_POSITION_AXES} coordinate rows, got "
        f"{tuple(position_ids.shape)}"
    )
    if position_ids.dtype == torch.int32 and position_ids.is_contiguous():
        pos3 = torch.reshape(position_ids, [_QSA_POSITION_AXES, -1])
    else:
        pos3 = _copy_positions(position_ids)
    return pos3, pos3[0], torch.transpose(pos3, 0, 1)


def _draft_embedding(input_ids: torch.Tensor, embed_tokens: torch.Tensor) -> torch.Tensor:
    """Embed the MTP front end's tokens; negative ids read row 0.

    Rejection sampling fills slots past a request's accepted count with -1; those rows feed
    nothing the step keeps, so clamping only keeps the gather inside the table."""
    if input_ids.dim() == 1 and input_ids.is_contiguous():
        # The id clamp and gather in one launch (bitwise the composed path).
        return embed_rows(input_ids, embed_tokens, 1, True)
    return nn.functional.embedding(input_ids.clamp_min(0), embed_tokens)


def _mtp_front(core, M: _MtpWeights, embed_tokens, input_ids, hidden_states, mm=None) -> tuple:
    """Run the draft layer's front end and attention-side hyper-connection.

    `fc_embedding` of the normalized next-token embedding is added to every stream of
    `fc_hidden` as the trunk site's unit-weight injection; returns `(x, mixed, inject)`. `mm`
    runs dense projections here and in every draft-layer and MoE helper: `cublas_mm` when
    None, `_decode_projection` from `DecodeTarget`."""
    mm = mm or cublas_mm
    rows = hidden_states.shape[0]
    assert rows <= M.unit_inject.shape[0], (
        f"a draft step of {rows} rows exceeds max_num_tokens {M.unit_inject.shape[0]}"
    )
    e = _draft_embedding(input_ids, embed_tokens)
    main = torch.cuda.current_stream()
    side = core._side_stream if rows <= _MTP_FRONT_FORK_MAX_ROWS else main
    # Fork after the embedding is queued; `x_e` is consumed on `main` after the join.
    side.wait_stream(main)
    with torch.cuda.stream(side):
        x_e = mm(flashinfer_gemma_rmsnorm(e, M.pre_e_norm, core.eps), M.fc_e)
    _, h_n = qwen4_exp_hc_inject_norm(
        hidden_states, None, None, M.pre_h_norm, core.eps, core.hc_count
    )
    x_h = torch.reshape(
        mm(torch.reshape(h_n, [rows * core.hc_count, core.hidden]), M.fc_h),
        [rows, core.stream],
    )
    main.wait_stream(side)
    return _hc_site(core, x_h, x_e, M.unit_inject[:rows], M.ahc_norm, M.ahc_first, M.ahc_up, True)


def _mtp_attention_in(core, M: _MtpWeights, mixed, pos3, pos, with_index: bool, mm=None) -> tuple:
    """Compute the draft layer's attention input as a trunk attention layer does.

    Returns `(proj, qkv, gate, k, v, index_scratch)`: `gate` is `proj`'s `[T, q_width]` gate
    block, `k`/`v` the rotated heads, `index_scratch` None without `with_index`."""
    mm = mm or cublas_mm
    rows = mixed.shape[0]
    proj = mm(mixed, M.attn_in)
    index_scratch = (
        torch.empty([rows, core.index_scratch_width], dtype=mixed.dtype, device=mixed.device)
        if with_index
        else None
    )
    _attention_rope(core, proj, index_scratch, M, pos3, pos, theta=core.mtp_theta)
    if rows == 1:
        # One row: the `[q | k | v]` prefix is contiguous and not written after the rope.
        qkv = proj[:, : core.qkv_width]
    else:
        qkv = torch.empty([rows, core.qkv_width], dtype=mixed.dtype, device=mixed.device)
        qkv.copy_(proj[:, : core.qkv_width])
    _, k, v = _attn_in_heads(core, proj)
    gate = torch.split(proj, [core.qkv_width, core.q_width, core.index_scratch_width], 1)[1]
    return proj, qkv, gate, k, v, index_scratch


def _mtp_attention_planes(
    core, M: _MtpWeights, mixed, draft: _DraftBound, qsa, rows_per_request, with_index, sparse
) -> tuple:
    """Compute a generation-only draft step's attention input and write its planes.

    Returns `(qkv, gate, q_index)`. The K/V append happens here only on the sparse path (the
    dense op appends itself). The draft-extend step runs everything after the projection as
    the trunk's one-launch `qwen4_exp_attn_decode_prologue`."""
    req_idx, logical, block_table, tokens_per_block, pos3, pos, coords = qsa
    if with_index and _decode_prologue(draft.kv, sparse):
        rows = mixed.shape[0]
        proj = _decode_projection(mixed, M.attn_in)
        qkv = torch.empty([rows, core.qkv_width], dtype=mixed.dtype, device=mixed.device)
        index_scratch = torch.empty(
            [rows, core.index_scratch_width], dtype=mixed.dtype, device=mixed.device
        )
        q_index = _index_query(core, index_scratch)
        qwen4_exp_attn_decode_prologue(
            proj,
            qkv,
            q_index,
            pos3,
            coords,
            core._mtp_cos_sin,
            M.q_norm,
            M.k_norm,
            M.idx_qn,
            M.idx_kn,
            draft.kv.k,
            draft.kv.v,
            draft.index_cache,
            draft.position_cache,
            req_idx,
            logical,
            block_table,
            tokens_per_block,
            core.compress_ratio,
            core.heads_q,
            core.heads_kv,
            core.head_dim,
            core.index_heads,
            core.index_head_dim,
            core.index_col,
            core.rotary_dim,
            core.eps,
            core.mrope_section1,
            core.mrope_section2,
            rows_per_request=rows_per_request,
        )
        gate = torch.split(proj, [core.qkv_width, core.q_width, core.index_scratch_width], 1)[1]
        return qkv, gate, q_index
    proj, qkv, gate, k_dst, v_dst, index_scratch = _mtp_attention_in(
        core, M, mixed, pos3, pos, with_index, mm=_decode_projection
    )
    q_index = None
    if with_index:
        q_index = _mtp_index_branch(
            core,
            M,
            proj,
            index_scratch,
            draft,
            (req_idx, logical, block_table, tokens_per_block, coords),
        )
    if sparse:
        _store_kv(core, draft.kv, k_dst, v_dst, req_idx, logical, block_table, tokens_per_block)
    return qkv, gate, q_index


def _mtp_index_branch(core, M: _MtpWeights, proj, index_scratch, draft: _DraftBound, qsa):
    """Run the draft layer's QSA index branch on its own planes, as `_index_plane` does.

    Stores the raw index key, compresses completed groups, returns the rotated index query."""
    req_idx, logical, block_table, tokens_per_block, coords = qsa
    rows = proj.shape[0]
    token_k = torch.split(
        torch.reshape(
            proj[:, core.index_col : core.index_col + core.index_scratch_width],
            [rows, core.index_scratch_width // core.index_head_dim, core.index_head_dim],
        ),
        [core.index_heads, core.index_kv_heads, 1],
        1,
    )[1]
    qsa_paged_index_store(
        token_k,
        coords,
        draft.index_cache,
        draft.position_cache,
        req_idx,
        logical,
        block_table,
        tokens_per_block,
    )
    qsa_prefill_compress(
        logical_positions=logical,
        request_indices=req_idx,
        block_table=block_table,
        index_cache=draft.index_cache,
        position_cache=draft.position_cache,
        k_norm_weight=M.idx_kn,
        cos_sin=core._mtp_cos_sin,
        eps=core.eps,
        tokens_per_block=tokens_per_block,
        compress_ratio=core.compress_ratio,
        mrope_section=_QSA_MROPE_SECTION,
    )
    return _index_query(core, index_scratch)


def _moe_max_rows(max_num_tokens: int, dp_size: int) -> int:
    """Return the rows per routed-experts call, never fewer than `_MOE_MAX_ROWS`.

    Attention DP: `_MOE_MAX_ROWS`. Attention TP: the step's rows, capped at `_MOE_MAX_ROWS_TP`
    (the runner's certified bound) or, with the grouped prefill experts, `_ADP_WINDOW_MAX_ROWS`
    (the most rows the runner takes elsewhere, for calls the grouped experts decline)."""
    if dp_size > 1:
        return _MOE_MAX_ROWS
    if _MOE_PREFILL_GROUPED_MIN_ROWS and max_num_tokens >= _MOE_PREFILL_GROUPED_MIN_ROWS:
        return min(max(max_num_tokens, _MOE_MAX_ROWS), _ADP_WINDOW_MAX_ROWS)
    return min(max(max_num_tokens, _MOE_MAX_ROWS), _MOE_MAX_ROWS_TP)


def _runner_tune_rows(core) -> int:
    """Return the FP8 runner's `tune_max_num_tokens`: `core.moe_max_rows`, clamped.

    Tuning at the rows large context chunks run beats reusing the 8192-row tactic; past
    `_MOE_MAX_ROWS_TP` calls run the grouped experts. Every tactic writes the same bits."""
    return min(max(getattr(core, "moe_max_rows", _MOE_MAX_ROWS), _MOE_MAX_ROWS), _MOE_MAX_ROWS_TP)


def _window_runner(
    core,
    experts: _weights.ExpertWindow,
    xq,
    xsf,
    topk_ids,
    topk_w,
    output=None,
    fin_first=None,
    after_first=None,
):
    """Run this rank's whole-expert window over pre-routed rows.

    `xq` is 1x128 e4m3 rows with `[hidden / 128, rows]` scales `xsf`; top-k ids span the
    whole routing space and an id outside the window (or -1) adds nothing. Context chunks run
    `moe_prefill_grouped`, which finalizes `fin_first` rows first and hands them to
    `after_first`; on the runner `after_first` sees the whole output."""
    if (
        output is None
        and _MOE_PREFILL_GROUPED_MIN_ROWS
        and _use_prefill_grouped(core, xq.shape[0], experts.count)
    ):
        split = {} if after_first is None else {"fin_first": fin_first, "after_first": after_first}
        return moe_prefill_grouped(
            xq,
            xsf,
            topk_ids,
            topk_w,
            experts.fc1_w,
            experts.fc1_s,
            experts.fc2_w,
            experts.fc2_s,
            experts.offset,
            **split,
        )
    if after_first is not None:
        out = _window_runner(core, experts, xq, xsf, topk_ids, topk_w, output)
        after_first(out)
        return out
    return fp8_block_scale_moe_runner(
        routing_logits=None,
        routing_bias=None,
        hidden_states=xq,
        hidden_states_scale=xsf,
        gemm1_weights=experts.fc1_w,
        gemm1_weights_scale=experts.fc1_s,
        gemm2_weights=experts.fc2_w,
        gemm2_weights_scale=experts.fc2_s,
        num_experts=core.num_experts,
        top_k=core.topk,
        num_fused_shared_experts=None,
        n_group=None,
        topk_group=None,
        intermediate_size=core.moe_inter,
        local_expert_offset=experts.offset,
        local_num_experts=experts.count,
        routed_scaling_factor=None,
        routing_method_type=_ROUTING_METHOD_RENORM,
        topk_weights=topk_w,
        topk_ids=topk_ids,
        act_type=_ACT_TYPE_SWIGLU,
        gemm1_clamp_limit=None,
        output=output,
        tune_max_num_tokens=_runner_tune_rows(core),
    )


def _use_prefill_grouped(core, rows: int, local: int) -> bool:
    """Whether a call over `rows` rows of a `local`-expert window runs `moe_prefill_grouped`.

    A capture before the first eager call keeps the runner: capture cannot load kernels."""
    if not _MOE_PREFILL_GROUPED_MIN_ROWS or rows < _MOE_PREFILL_GROUPED_MIN_ROWS:
        return False
    if getattr(core, "dp_size", 1) != 1:
        return False
    if prefill_grouped_ready(local):
        return True
    if torch.cuda.is_current_stream_capturing():
        return False
    warm_prefill_grouped(torch.cuda.current_device(), local)
    return True


def _routed_experts(
    core, x, router, experts: _weights.ExpertWindow, mm=None, pad_mask=None, before_runner=None
) -> torch.Tensor:
    """Return this rank's partial of a MoE layer's routed experts over every row of `x`.

    Router logits, renormalized top-k (`renorm_moe_routing_op`) and 1x128 e4m3 quantization
    feed this rank's window (`_window_runner`); slots outside it add nothing, so the windows
    sum to the layer. Routing runs in chunks of `core.moe_max_rows` rows; an attention-DP
    rank batches up to `_ADP_WINDOW_MAX_ROWS` rows of chunks per runner call (bitwise equal).
    `pad_mask` (`[T, 1]` bool) routes a prefill graph bucket's padding rows to expert -1, so
    they come back zero instead of all landing on the first window's ten lowest experts."""
    mm = mm or cublas_mm
    if getattr(core, "dp_size", 1) > 1:
        # Attention DP routes the same gathered rows on every rank, whose steps may pick another
        # `mm`: `_decode_projection`'s low-M kernels and cuBLAS differ in the last bits, and a
        # reordered top-k is a different MoE sum. One router kernel on every rank.
        mm = cublas_mm
    rows = x.shape[0]
    if pad_mask is None and mm is _decode_projection and 0 < rows <= _MOE_FRONT_MAX_ROWS:
        if before_runner is not None:
            before_runner()
        return _routed_experts_front(core, x, router, experts)
    step = getattr(core, "moe_max_rows", _MOE_MAX_ROWS)
    if (
        pad_mask is None
        and mm is _decode_projection
        and getattr(core, "dp_size", 1) == 1
        and 0 < rows <= step
    ):
        logits = mm(x, router)
        if rows <= _MOE_FAST_FRONT_MAX_ROWS:
            out = _window_runner_front(core, experts, x, logits, before_runner)
            if out is not None:
                return out
        xq, xsf = fp8_quantize_1x128(x, use_ue8m0=_MOE_USE_UE8M0)
        if before_runner is not None:
            before_runner()
        return _window_runner_logits(core, experts, xq, xsf, logits)
    if before_runner is not None:
        before_runner()
    group = 1
    if getattr(core, "dp_size", 1) > 1 and rows > step:
        group = max(1, _ADP_WINDOW_MAX_ROWS // step)
    parts, pending = [], []
    for lo in range(0, rows, step):
        chunk = x[lo : lo + step]
        logits = mm(chunk, router)
        topk_ids, topk_w = renorm_moe_routing_op(logits, topk=core.topk, output_dtype=chunk.dtype)
        if pad_mask is not None:
            topk_ids.masked_fill_(pad_mask[lo : lo + step], _NO_EXPERT)
        xq, xsf = fp8_quantize_1x128(chunk, use_ue8m0=_MOE_USE_UE8M0)
        pending.append((xq, xsf, topk_ids, topk_w))
        if len(pending) == group or lo + step >= rows:
            if len(pending) == 1:
                args = pending[0]
            else:
                q, sf, ids, w = zip(*pending)
                codes = torch.cat([t.view(torch.uint8) for t in q]).view(q[0].dtype)
                args = (codes, torch.cat(sf, 1), torch.cat(ids), torch.cat(w))
            parts.append(_window_runner(core, experts, *args))
            pending = []
    return parts[0] if len(parts) == 1 else torch.cat(parts, 0)


def _route_rows(core, x, router, mm=None, pad_mask=None) -> tuple:
    """Route every row of `x` as `_routed_experts` does; return `(topk_ids, topk_weights)`.

    Used by the all-to-all round trip, which routes rows on their own rank."""
    mm = mm or cublas_mm
    step = getattr(core, "moe_max_rows", _MOE_MAX_ROWS)
    ids, weights = [], []
    for lo in range(0, x.shape[0], step):
        chunk = x[lo : lo + step]
        topk_ids, topk_w = renorm_moe_routing_op(
            mm(chunk, router), topk=core.topk, output_dtype=chunk.dtype
        )
        if pad_mask is not None:
            topk_ids.masked_fill_(pad_mask[lo : lo + step], _NO_EXPERT)
        ids.append(topk_ids)
        weights.append(topk_w)
    if not ids:
        # A rank without rows this step: the routing op's empties.
        logits = x.new_zeros(0, core.num_experts)
        return renorm_moe_routing_op(logits, topk=core.topk, output_dtype=x.dtype)
    return torch.cat(ids), torch.cat(weights)


def _window_prerouted(core, x, topk_ids, topk_w, experts: _weights.ExpertWindow) -> torch.Tensor:
    """Run this rank's window over rows routed by `_route_rows`; rows with no local id are 0."""
    step = getattr(core, "moe_max_rows", _MOE_MAX_ROWS)
    rows = x.shape[0]
    group = max(1, _ADP_WINDOW_MAX_ROWS // step) if rows > step else 1
    parts, pending = [], []
    for lo in range(0, rows, step):
        xq, xsf = fp8_quantize_1x128(x[lo : lo + step], use_ue8m0=_MOE_USE_UE8M0)
        pending.append((xq, xsf, topk_ids[lo : lo + step], topk_w[lo : lo + step]))
        if len(pending) == group or lo + step >= rows:
            if len(pending) == 1:
                args = pending[0]
            else:
                q, sf, ids, w = zip(*pending)
                codes = torch.cat([t.view(torch.uint8) for t in q]).view(q[0].dtype)
                args = (codes, torch.cat(sf, 1), torch.cat(ids), torch.cat(w))
            parts.append(_window_runner(core, experts, *args))
            pending = []
    return parts[0] if len(parts) == 1 else torch.cat(parts, 0)


def _window_runner_front(core, experts: _weights.ExpertWindow, x, logits, before_runner):
    """Run `_window_runner_logits` through the one-launch decode routing with fused quantization.

    Calls `before_runner` once the router projection (and quantization) are queued. Returns
    None, queuing nothing past the router, while the AutoTuner holds no explicit tactic."""
    rows, hidden = x.shape
    fold = x.is_contiguous()
    if fold:
        xq = torch.empty((rows, hidden), dtype=torch.float8_e4m3fn, device=x.device)
        xsf = torch.empty((hidden // 128, rows), dtype=torch.float32, device=x.device)
    else:
        xq, xsf = fp8_quantize_1x128(x, use_ue8m0=_MOE_USE_UE8M0)
    call = fp8_block_scale_moe_front(
        logits,
        xq,
        xsf,
        experts.fc1_w,
        experts.fc1_s,
        experts.fc2_w,
        experts.fc2_s,
        core.num_experts,
        core.topk,
        core.moe_inter,
        experts.offset,
        experts.count,
        _ROUTING_METHOD_RENORM,
    )
    if call is None:
        if not fold:
            if before_runner is not None:
                before_runner()
            return _window_runner_logits(core, experts, xq, xsf, logits)
        return None
    if before_runner is not None:
        before_runner()
    return call(x if fold else None)


def _window_runner_logits(core, experts: _weights.ExpertWindow, xq, xsf, logits) -> torch.Tensor:
    """Run the window with the runner routing `logits` itself (bitwise `renorm_moe_routing_op`)."""
    return fp8_block_scale_moe_runner(
        routing_logits=logits,
        routing_bias=None,
        hidden_states=xq,
        hidden_states_scale=xsf,
        gemm1_weights=experts.fc1_w,
        gemm1_weights_scale=experts.fc1_s,
        gemm2_weights=experts.fc2_w,
        gemm2_weights_scale=experts.fc2_s,
        num_experts=core.num_experts,
        top_k=core.topk,
        num_fused_shared_experts=None,
        n_group=None,
        topk_group=None,
        intermediate_size=core.moe_inter,
        local_expert_offset=experts.offset,
        local_num_experts=experts.count,
        routed_scaling_factor=None,
        routing_method_type=_ROUTING_METHOD_RENORM,
        topk_weights=None,
        topk_ids=None,
        act_type=_ACT_TYPE_SWIGLU,
        gemm1_clamp_limit=None,
    )


def _routed_experts_front(
    core, x, router, experts: _weights.ExpertWindow, shared: _TpbShared | None = None
) -> torch.Tensor:
    """Run `_routed_experts` for a decode step's rows (up to `_MOE_FRONT_MAX_ROWS`).

    One launch forms the bf16 router logits and the 1x128 e4m3 activation; then
    `moe_tpb_decode` over block-major experts (optionally folding in `shared`) or the runner.
    `router` is a `.t()` view of the checkpoint's `[512, hidden]` rows."""
    logits, xq, xsf = moe_front_router_quant(x.contiguous(), router.t())
    blocks = experts.blocks
    if blocks is not None and x.shape[0] <= _MOE_TPB_MAX_ROWS:
        return moe_tpb_decode(
            logits,
            xq,
            xsf,
            _BlockExperts(blocks.w1, blocks.s1, blocks.w2, blocks.s2, blocks.rank),
            shared,
        )
    assert shared is None, "the shared-expert fold rides the block launch"
    return fp8_block_scale_moe_runner(
        routing_logits=logits,
        routing_bias=None,
        hidden_states=xq,
        hidden_states_scale=xsf,
        gemm1_weights=experts.fc1_w,
        gemm1_weights_scale=experts.fc1_s,
        gemm2_weights=experts.fc2_w,
        gemm2_weights_scale=experts.fc2_s,
        num_experts=core.num_experts,
        top_k=core.topk,
        num_fused_shared_experts=None,
        n_group=None,
        topk_group=None,
        intermediate_size=core.moe_inter,
        local_expert_offset=experts.offset,
        local_num_experts=experts.count,
        routed_scaling_factor=None,
        routing_method_type=_ROUTING_METHOD_RENORM,
        topk_weights=None,
        topk_ids=None,
        act_type=_ACT_TYPE_SWIGLU,
        gemm1_clamp_limit=None,
    )


class _WindowRun(NamedTuple):
    """A MoE layer's `run` for `core.expert_sum`: `_routed_experts` over the given rows."""

    core: object
    router: torch.Tensor
    experts: _weights.ExpertWindow
    mm: object = None
    pad_mask: torch.Tensor | None = None
    before_runner: object = None

    def __call__(self, rows: torch.Tensor) -> torch.Tensor:
        return _routed_experts(
            self.core,
            rows,
            self.router,
            self.experts,
            self.mm,
            self.pad_mask,
            before_runner=self.before_runner,
        )


def _mtp_moe_parts(core, M: _MtpWeights, x, pad_rows: _DpPadding | None = None, mm=None) -> tuple:
    """Return `_mtp_moe_block`'s uncombined `(routed, gate_logits, shared)`, side stream joined."""
    mm = mm or cublas_mm
    main = torch.cuda.current_stream()
    side = core._side_stream
    side.wait_stream(main)
    with torch.cuda.stream(side):
        gug = mm(x, M.sh_gug)
        shared = mm(silu_and_mul_rows(gug, core.shared_inter), M.sh_dn)
        gate_logits = gug[:, 2 * core.shared_inter : 2 * core.shared_inter + 1]
    routed = core.expert_sum(_WindowRun(core, M.router, M.experts, mm), x, pad_rows)
    main.wait_stream(side)
    return routed, gate_logits, shared


def _mtp_moe_block(
    core, M: _MtpWeights, x, pad_rows: _DpPadding | None = None, mm=None
) -> torch.Tensor:
    """Return the draft layer's MoE, `routed + sigmoid(gate logit) * shared`, as the trunk's.

    `core.expert_sum` completes the routed partial; the shared expert runs on the side stream."""
    return fused_sigmoid_gate_mul_add(*_mtp_moe_parts(core, M, x, pad_rows, mm))


def _mtp_tail(
    core, M: _MtpWeights, x, attn, inject, pad_rows: _DpPadding | None, mm=None
) -> torch.Tensor:
    """Run the draft layer after its attention core: output projection, MoE-side site, MoE.

    `attn` is the `[T, q_width]` gated attention output; `pad_rows` is from `_draft_pad_rows`."""
    mm = mm or cublas_mm
    x, mixed, inject = _hc_site(
        core, x, _tp_partial(core, mm(attn, M.o)), inject, M.mhc_norm, M.mhc_first, M.mhc_up, True
    )
    if _combine_inject_fits(x, inject):
        # Combine and inject in one launch, bitwise equal to the composed ops below.
        routed, gate_logits, shared = _mtp_moe_parts(core, M, mixed, pad_rows, mm)
        if (
            routed.dtype == shared.dtype == torch.bfloat16
            and routed.stride(1) == shared.stride(1) == 1
        ):
            return mtp_moe_combine_inject(x, routed, gate_logits, shared, inject)
        return qwen4_exp_post_inject(
            x, fused_sigmoid_gate_mul_add(routed, gate_logits, shared), inject
        )
    return qwen4_exp_post_inject(x, _mtp_moe_block(core, M, mixed, pad_rows, mm=mm), inject)


def _ple_fold_fits(hc, routed, inject, shared) -> bool:
    """Whether the PLE launches can take these operands: 2-D, unit column stride, `hc`'s dtype."""
    rows = hc.shape[0]
    ok = (
        inject is not None
        and routed.dim() == 2
        and inject.dim() == 2
        and routed.shape[0] == inject.shape[0] == rows
        and routed.stride(1) == 1
        and inject.stride(1) == 1
        and routed.dtype == inject.dtype == hc.dtype
    )
    if ok and shared is not None:
        ok = shared.dim() == 2 and shared.stride(1) == 1 and shared.dtype == hc.dtype
    return ok


def _combine_inject_fits(x, inject) -> bool:
    """Whether `mtp_moe_combine_inject` takes `x` and `inject`: 2-D bf16, unit column stride."""
    return (
        x.dim() == 2
        and inject.dim() == 2
        and x.dtype == inject.dtype == torch.bfloat16
        and x.stride(1) == 1
        and inject.stride(1) == 1
        and x.shape[0] == inject.shape[0]
    )


class _DraftRowCounts(NamedTuple):
    """The group's per-rank row counts of one draft step, in the field `_dp_pad_rows` reads."""

    all_rank_num_tokens: list


def _draft_pad_rows(core, all_rank_num_tokens, rows: int) -> _DpPadding | None:
    """Return `_dp_pad_rows` over the row counts the spec worker passes for this draft step.

    The counts are each rank's trunk tokens on the loop's first step, then its request count."""
    if core.dp_size == 1:
        return None
    assert all_rank_num_tokens is not None, (
        "a draft step under attention DP came without the group's row counts; the MoE round "
        "trip cannot pad this rank's rows to the group's largest count"
    )
    return core._dp_pad_rows(_DraftRowCounts(list(all_rank_num_tokens)), rows)


def _decode_projection(mat_a: torch.Tensor, mat_b: torch.Tensor) -> torch.Tensor:
    """Dense bf16 projection for generation-only steps and their draft layer (graph-captured).

    `low_m_bf16_mm` picks FlashInfer's low-M PDL kernels (`_DECODE_PROJ_PDL`) where its table
    names the shape, else cuBLAS; they compile on the engine's warmup forward before capture."""
    return low_m_bf16_mm(mat_a, mat_b, _DECODE_PROJ_PDL)


def _hc_first(core, normed, first, with_inject):
    """Run a hyper-connection mixer's first half: first projection, then SiLU and injection.

    Returns `(low, inject)`; `inject` is None without `with_inject`."""
    if normed.shape[0] <= _HC_TC_FIRST_MAX_ROWS:
        return qwen4_exp_hc_decode_tc_first(
            normed, first, core.hc_lowrank, core.hc_count, with_inject
        )
    return qwen4_exp_hc_lowrank_act(
        cublas_mm(normed, first[:, : core.hc_first_cols], None, torch.float32),
        core.hc_lowrank,
        core.hc_count,
        with_inject,
        normed.dtype,
    )


def _hc_mixer(core, normed, first, up, with_inject):
    """Run one hyper-connection mixer on a normed stream; return `(mixed, inject)`."""
    low, inject = _hc_first(core, normed, first, with_inject)
    wide = normed.shape[0] <= _HC_UGM_WIDE_MAX_ROWS
    return qwen4_exp_hc_up_gate_mean(low, up, normed, core.hc_count, wide), inject


def _hc_inject_norm_launch(
    hc, block_out, inject, norm, eps, hc_count, shared, gate_logits, num_warps
):
    """Run `qwen4_exp_hc_inject_norm` (no finalize fold) at `num_warps` warps with PDL.

    The residual is bitwise at any warp count; the normed stream varies only with the fp32
    sum-of-squares order. Programs read nothing before `griddepcontrol.wait`. Returns
    `(hc_out, normed)`; `hc_out` is `hc` itself without `block_out`."""
    return qwen4_exp_hc_inject_norm(
        hc,
        block_out,
        inject,
        norm,
        eps,
        hc_count,
        shared,
        gate_logits,
        num_warps=num_warps,
        pdl_wait=_HC_INJECT_PDL_WAIT,
    )


def _hc_mixer_rstd(core, hc, rstd, norm, first, up, with_inject):
    """Run `_hc_mixer` from the residual `hc` and its `[T, C]` rstd instead of a normed stream.

    The first projection forms `bf16(hc * rstd * (1 + norm))` on its A tiles; the up
    projection's epilogue uses the fp32 value (`core._hc_norm_w1`)."""
    low, inject = qwen4_exp_hc_first_rescale_tc(
        hc, rstd, norm, first, core.hc_count, core.hc_lowrank, with_inject
    )
    w1 = core._hc_norm_w1[norm.data_ptr()]
    return qwen4_exp_hc_up_gate_mean_rstd(low, up, hc, rstd, w1, core.hc_count), inject


def _hc_site(
    core,
    hc,
    block_out,
    inject,
    norm,
    first,
    up,
    with_inject,
    shared=None,
    gate_logits=None,
):
    """Run one hyper-connection site on the residual `hc`; return `(hc, mixed, inject)`.

    Injects the pending block (`block_out` at weights `inject`, plus
    `sigmoid(gate_logits) * shared` after a MoE block), then the grouped norm and the mixer.
    A `_TpPartial` block output is summed over the TP group inside the injection launch."""
    partial = None
    if isinstance(block_out, _TpPartial):
        block_out = partial = block_out.partial
    if _HC_ROWS_TC_MIN_ROWS <= hc.shape[0] <= _HC_ROWS_TC_MAX_ROWS:
        assert partial is None, "a deferred TP partial reached a site that does not sum it"
        n_cols = core.hc_lowrank + (core.hc_count if with_inject else 0)
        hc, low, inject, rstd = qwen4_exp_hc_decode_rows_first(
            hc,
            block_out,
            inject,
            norm,
            first,
            core.hc_count,
            n_cols,
            core.eps,
            shared,
            gate_logits,
        )
        mixed = qwen4_exp_hc_decode_rows_mix(low, rstd, up, hc, norm, core.hc_count)
        return hc, mixed, inject
    if partial is not None:
        fused = getattr(core, "hc_ar_inject", None)
        if (
            fused is not None
            and hc.shape[0] < min(_HC_RSTD_MIN_ROWS, fused.max_rows + 1)
            and _hc_ar_inject_operands_fit(hc, partial, inject, shared, gate_logits)
        ):
            warps = 2 if hc.shape[0] >= _HC_INJECT_FEW_WARPS_MIN_ROWS else _HC_INJECT_WARPS
            hc, normed = hc_allreduce_inject_norm(
                partial,
                fused.workspace,
                fused.rank,
                fused.nranks,
                hc,
                inject,
                norm,
                core.eps,
                shared,
                gate_logits,
                warps,
            )
            mixed, inject = _hc_mixer(core, normed, first, up, with_inject)
            return hc, mixed, inject
        block_out = core.tp_sum(block_out)
    if hc.shape[0] >= _HC_RSTD_MIN_ROWS:
        hc, rstd = qwen4_exp_hc_inject_rstd(
            hc, block_out, inject, core.eps, core.hc_count, shared, gate_logits
        )
        mixed, inject = _hc_mixer_rstd(core, hc, rstd, norm, first, up, with_inject)
        return hc, mixed, inject
    warps = 2 if hc.shape[0] >= _HC_INJECT_FEW_WARPS_MIN_ROWS else _HC_INJECT_WARPS
    hc, normed = _hc_inject_norm_launch(
        hc, block_out, inject, norm, core.eps, core.hc_count, shared, gate_logits, warps
    )
    mixed, inject = _hc_mixer(core, normed, first, up, with_inject)
    return hc, mixed, inject


def _moe_tpb_shared_folded(core, x, experts, mm, pad_rows, pad_mask, defer) -> bool:
    """Whether `_moe_block` folds this rank's shared-expert slice into the block launch."""
    rows = x.shape[0]
    return (
        defer
        and experts.blocks is not None
        and pad_rows is None
        and pad_mask is None
        and mm is _decode_projection
        and getattr(core, "dp_size", 1) == 1
        and _MOE_TPB_SHARED_MIN_ROWS <= rows <= min(_MOE_FRONT_MAX_ROWS, _MOE_TPB_MAX_ROWS)
    )


def _moe_block(
    core,
    x,
    L: _LayerWeights,
    pad_rows: _DpPadding | None,
    mm=None,
    pad_mask=None,
    defer=False,
) -> tuple:
    """Run one MoE layer; return `(routed, gate_logits, shared)` uncombined.

    The next `qwen4_exp_hc_inject_norm` forms `routed + sigmoid(gate_logits) * shared`.
    `routed` comes from `core.expert_sum` (`pad_rows` from `_dp_pad_rows`); the shared expert
    runs on the side stream and joins before this returns. `pad_mask` keeps prefill graph
    padding rows out of the window. With `defer` (attention-TP decode) `routed` may be a
    `_TpPartial` the next site sums. When `_moe_tpb_shared_folded` holds, `routed` is the
    whole combine and the others are None."""
    mm = mm or cublas_mm
    if _moe_tpb_shared_folded(core, x, L.experts, mm, pad_rows, pad_mask, defer):
        b = L.experts.blocks
        shared = _TpbShared(x.contiguous(), b.sh_gu, b.sh_gl, b.sh_dn)
        partial = _routed_experts_front(core, x, L.router, L.experts, shared=shared)
        return _tp_partial(core, partial), None, None
    main = torch.cuda.current_stream()
    side = core._side_stream
    forked = []

    def fork_shared():
        # The side stream waits for `main`; its outputs are used on `main` after the join.
        core._side_fork.record(main)
        side.wait_event(core._side_fork)
        with torch.cuda.stream(side):
            gug = mm(x, L.sh_gug)
            shared = mm(silu_and_mul_rows(gug, core.shared_inter), L.sh_dn)
        forked.append((gug[:, 2 * core.shared_inter : 2 * core.shared_inter + 1], shared))

    # Under attention TP a decode step forks once the router and the quantization are queued
    # (`_routed_experts`' `before_runner`); every other step forks first.
    late = pad_rows is None and getattr(core, "dp_size", 1) == 1
    if not late:
        fork_shared()
    run = _WindowRun(core, L.router, L.experts, mm, pad_mask, fork_shared if late else None)
    if defer and getattr(core, "dp_size", 1) == 1:
        routed = _tp_partial(core, run(x))
    else:
        routed = core.expert_sum(run, x, pad_rows)
    if not forked:
        fork_shared()
    core._side_join.record(side)
    main.wait_event(core._side_join)
    gate_logits, shared = forked[0]
    return routed, gate_logits, shared


def _routed_experts_sp(
    core,
    x_rows,
    router,
    experts: _weights.ExpertWindow,
    gather,
    pad_mask=None,
    fin_first=None,
    after_first=None,
):
    """This rank's window partial `[T, hidden]` over every row of a sequence-parallel step.

    Each rank routes and e4m3-quantizes only its own rows (`pad_mask` sends padding rows to
    `_NO_EXPERT`); two uniform gathers carry the codes and a `[scales | ids | weights]` row."""
    topk = core.topk
    logits = cublas_mm(x_rows, router)
    topk_ids, topk_w = renorm_moe_routing_op(logits, topk=topk, output_dtype=x_rows.dtype)
    if pad_mask is not None:
        topk_ids.masked_fill_(pad_mask, _NO_EXPERT)
    xq, xsf = fp8_quantize_1x128(x_rows, use_ue8m0=_MOE_USE_UE8M0)
    nb = xsf.shape[0]
    rows = x_rows.shape[0]
    # bf16 columns: nb fp32 scales, topk int32 ids, topk bf16 weights (the width stays even).
    c0, c1, c2 = 2 * nb, 2 * nb + 2 * topk, 2 * nb + 3 * topk
    meta = torch.empty([rows, c2 + (c2 & 1)], dtype=torch.bfloat16, device=x_rows.device)
    meta[:, :c0].view(torch.float32).copy_(xsf.t())
    meta[:, c0:c1].view(torch.int32).copy_(topk_ids)
    meta[:, c1:c2].copy_(topk_w)
    codes = gather(xq.view(torch.bfloat16)).view(xq.dtype)
    every = gather(meta)
    scales = every[:, :c0].view(torch.float32).t()
    ids = every[:, c0:c1].contiguous().view(torch.int32)
    weights = every[:, c1:c2].contiguous()
    total = codes.shape[0]
    step = getattr(core, "moe_max_rows", _MOE_MAX_ROWS)
    if after_first is not None and total <= step:
        return _window_runner(
            core,
            experts,
            codes,
            scales.contiguous(),
            ids,
            weights,
            fin_first=fin_first,
            after_first=after_first,
        )
    parts = [
        _window_runner(
            core,
            experts,
            codes[lo : lo + step],
            scales[:, lo : lo + step].contiguous(),
            ids[lo : lo + step],
            weights[lo : lo + step],
        )
        for lo in range(0, total, step)
    ]
    out = parts[0] if len(parts) == 1 else torch.cat(parts, 0)
    if after_first is not None:
        after_first(out)
    return out


def _moe_block_sp(core, x_rows, L: _LayerWeights, gather, scatter, pad_mask=None) -> tuple:
    """`_moe_block` under SP, returning `(routed, gate_logits, shared)` over this rank's rows.

    The shared expert runs on the side stream, overlapping the routed path's gathers."""
    main = torch.cuda.current_stream()
    side = core._side_stream
    core._side_fork.record(main)
    side.wait_event(core._side_fork)
    with torch.cuda.stream(side):
        gug = cublas_mm(x_rows, L.sh_gug)
        shared = cublas_mm(silu_and_mul_rows(gug, core.shared_inter), L.sh_dn)
        gate_logits = gug[:, 2 * core.shared_inter : 2 * core.shared_inter + 1]
    split = getattr(scatter, "split_finalize", None)
    if split is not None:
        # Two ranks: the peer's rows are finalized and sent first, overlapping this rank's.
        start, finish, peer_rows = split
        routed = finish(
            _routed_experts_sp(
                core, x_rows, L.router, L.experts, gather, pad_mask, peer_rows, start
            )
        )
    else:
        routed = scatter(_routed_experts_sp(core, x_rows, L.router, L.experts, gather, pad_mask))
    core._side_join.record(side)
    main.wait_event(core._side_join)
    return routed, gate_logits, shared


def _projection_warm_rows(lo: int, hi: int) -> list[int]:
    """One row count of `[lo, hi]` for each kernel cuBLASLt can pick for a bf16 projection.

    Every row up to `_KERNEL_CHOICE_EACH_ROW`, then one per 8-row group, which shares a kernel."""
    each = min(hi, _KERNEL_CHOICE_EACH_ROW)
    rows = list(range(lo, each + 1))
    first = max(lo, _KERNEL_CHOICE_EACH_ROW + 1)
    if first <= hi:
        group = _KERNEL_CHOICE_ROW_GROUP
        ends = range(-(-first // group) * group, hi + group, group)
        rows += [min(end, hi) for end in ends]
    return rows


def _eager_projections(core, layers, stack: _StackWeights, mtp) -> list[tuple]:
    """Every bf16 projection an eager step can hand `cublas_mm`, as `(mat_b, out_dtype, lo, hi)`.

    One entry per cuBLASLt kernel-choice key (K, N, strides, output dtype) with its row range."""
    rows = core.max_num_tokens
    router_rows = min(getattr(core, "moe_max_rows", _MOE_MAX_ROWS), core.dp_size * rows)
    hc_cols = core.hc_first_cols
    found = {}

    def add(mat_b, hi, lo=1, out_dtype=None):
        key = (tuple(mat_b.shape), mat_b.stride(), mat_b.dtype, out_dtype)
        if key in found:
            _, _, was_lo, was_hi = found[key]
            lo, hi = min(lo, was_lo), max(hi, was_hi)
        if lo <= hi:
            found[key] = (mat_b, out_dtype, lo, hi)

    def mixer(first):
        add(first[:, :hc_cols], rows, _HC_TC_FIRST_MAX_ROWS + 1, torch.float32)

    for L in layers:
        mixer(L.ahc_first)
        mixer(L.mhc_first)
        for mat_b in (L.attn_in, L.o, L.in_proj, L.out, L.sh_gug, L.sh_dn):
            if mat_b is not None:
                add(mat_b, rows)
        add(L.router, router_rows)
    mixer(stack.fhc_first)
    add(stack.ple_key, rows)
    add(stack.ple_val, rows)
    if mtp is not None:
        for first in (mtp.ahc_first, mtp.mhc_first, mtp.fhc_first):
            mixer(first)
        for mat_b in (mtp.fc_e, mtp.attn_in, mtp.o, mtp.sh_gug, mtp.sh_dn):
            add(mat_b, rows)
        add(mtp.fc_h, core.hc_count * rows)
        add(mtp.router, router_rows)
    return list(found.values())


def _load_projection_kernels(projections, device) -> int:
    """Run each projection at every `_projection_warm_rows` count; return the calls made.

    A kernel's first lazy-loaded launch waits for queued work and stalls the next collective."""
    k_max = max(mat_b.shape[0] * hi for mat_b, _, _, hi in projections)
    flat = torch.zeros(k_max, dtype=torch.bfloat16, device=device)
    calls = 0
    for mat_b, out_dtype, lo, hi in projections:
        k = mat_b.shape[0]
        for m in _projection_warm_rows(lo, hi):
            cublas_mm(flat[: m * k].view(m, k), mat_b, None, out_dtype)
            calls += 1
    return calls


def _attn_in_heads(core, proj):
    """The main q, k, v heads of `proj` `[T, attn_in_width]` as `[T, heads, head_dim]` views."""
    q, k, v, _ = torch.split(
        torch.reshape(proj, [proj.shape[0], core.attn_in_width // core.head_dim, core.head_dim]),
        [core.heads_q, core.heads_kv, core.heads_kv, core.attn_in_tail_heads - core.heads_kv],
        1,
    )
    return q, k, v


def _packed_q(core, qkv):
    """The query heads `[T, heads_q, head_dim]` of the compact `[T, qkv_width]` packed QKV."""
    heads = core.heads_q + 2 * core.heads_kv
    return torch.reshape(qkv, [qkv.shape[0], heads, core.head_dim])[:, : core.heads_q]


def _attention_rope(core, proj, index_scratch, L, pos3, pos, theta=None) -> None:
    """Both planes' Gemma norm + rope: main q/k in place in `proj`, index q/k in `index_scratch`.

    The index q/k are rotated in a copy so the raw index key stays in `proj` for the index store;
    the heads after k pass as v only to set the row stride. `theta` defaults to the trunk's."""
    base = core.theta if theta is None else theta
    fused_qk_norm_rope(
        proj,
        num_heads_q=core.heads_q,
        num_heads_k=core.heads_kv,
        num_heads_v=core.attn_in_tail_heads,
        head_dim=core.head_dim,
        rotary_dim=core.rotary_dim,
        eps=core.eps,
        q_weight=L.q_norm,
        k_weight=L.k_norm,
        base=base,
        is_neox=True,
        position_ids=pos3,
        is_qk_norm=True,
        use_gemma=True,
        use_mrope=True,
        mrope_section1=core.mrope_section1,
        mrope_section2=core.mrope_section2,
    )
    if index_scratch is None:
        return
    index_scratch[:, : core.index_width].copy_(
        proj[:, core.index_col : core.index_col + core.index_width]
    )
    fused_qk_norm_rope(
        index_scratch,
        num_heads_q=core.index_heads,
        num_heads_k=core.index_kv_heads,
        num_heads_v=core.index_kv_heads,
        head_dim=core.index_head_dim,
        rotary_dim=core.rotary_dim,
        eps=core.eps,
        q_weight=L.idx_qn,
        k_weight=L.idx_kn,
        base=base,
        is_neox=True,
        position_ids=pos,
        is_qk_norm=True,
        use_gemma=True,
        use_mrope=False,
        mrope_section1=0,
        mrope_section2=0,
    )


def _index_query(core, index_scratch):
    """The rotated index query `[T, index_heads, index_head_dim]`, a view of `index_scratch`."""
    rows = index_scratch.shape[0]
    return torch.split(
        torch.reshape(
            index_scratch,
            [rows, core.index_heads + 2 * core.index_kv_heads, core.index_head_dim],
        ),
        [core.index_heads, core.index_kv_heads, core.index_kv_heads],
        1,
    )[0]


def _index_plane(core, proj, i, L: _LayerWeights, qsa) -> None:
    """Store layer `i`'s raw index keys and positions; compress the groups this step completes.

    The key is stored raw because the compression applies the norm and the rotation itself."""
    req_idx, logical, block_table, tokens_per_block, coords = qsa
    rows = proj.shape[0]
    token_k = torch.split(
        torch.reshape(
            proj[:, core.index_col : core.index_col + core.index_scratch_width],
            [rows, core.index_scratch_width // core.index_head_dim, core.index_head_dim],
        ),
        [core.index_heads, core.index_kv_heads, 1],
        1,
    )[1]
    pools = core._bound.pools
    index_cache = pools.index_cache[i]
    qsa_paged_index_store(
        token_k,
        coords,
        index_cache,
        pools.position_cache,
        req_idx,
        logical,
        block_table,
        tokens_per_block,
    )
    qsa_prefill_compress(
        logical_positions=logical,
        request_indices=req_idx,
        block_table=block_table,
        index_cache=index_cache,
        position_cache=pools.position_cache,
        k_norm_weight=L.idx_kn,
        cos_sin=core._cos_sin,
        eps=core.eps,
        tokens_per_block=tokens_per_block,
        compress_ratio=core.compress_ratio,
        mrope_section=_QSA_MROPE_SECTION,
    )


def _select(core, scores, row_starts, row_ends, topk_indices) -> None:
    """Each row's canonical top-`block_topk` groups of its window into `topk_indices`.

    Canonical: groups above the budget's last score, then the highest-index ties, ascending, `-1`
    last (row starts are zero, so window and table indices agree)."""
    qsa_topk_select(scores, row_starts, row_ends, topk_indices, index_topk=core.block_topk)
    qsa_canonical_selection(scores, row_starts, row_ends, topk_indices)


def _index_split_group(core) -> tuple | None:
    """`(group, process_group, rank, lamport)` a decode step's index scoring splits over, or None.

    One small in-order gather builds the NCCL communicator now; a first call cannot be captured."""
    if core.attn_tp_size < 2 or core.dp_size != 1 or core.tp_sum is None:
        return None
    group = list(core.mapping.tp_group)
    process_group = core.mapping.tp_group_pg.boxed() if mpi_disabled() else None
    probe = torch.zeros([1, 1], dtype=torch.int32, device=core._cos_sin.device)
    allgather(probe, None, group, process_group)
    return group, process_group, core.mapping.tp_rank, _index_split_lamport(core)


def _index_split_lamport(core):
    """The split gather's Lamport IPC buffers, or None where the Lamport exchange does not apply.

    Needs 2-4 attention-TP ranks with peer access, PDL and captured batches that reach the split.
    A collective: every rank builds the buffers here, before any capture."""
    graph_batches = tuple(
        getattr(getattr(core, "model_config", None), "cuda_graph_batch_sizes", ()) or ()
    )
    if not (
        2 <= core.attn_tp_size <= 4
        and get_env_enable_pdl()
        and _INDEX_SPLIT_LAMPORT_ROWS > 0
        and (not graph_batches or max(graph_batches) >= _INDEX_SPLIT_MIN_REQUESTS)
        and can_access_peer(core.mapping)
    ):
        return None
    width = core.token_topk + core.compress_ratio - 1
    rows = min(_INDEX_SPLIT_LAMPORT_ROWS, core.max_num_tokens)
    return _selection_lamport_build(core.mapping, rows * width)


def _index_split_rows(split, rows: int, rows_per_seq: int) -> tuple | None:
    """This rank's `[lo, hi)` rows of a split index scoring, or None to score every row."""
    if split is None:
        return None
    group, rank = split[0], split[2]
    t = len(group)
    requests = rows // rows_per_seq
    if requests < _INDEX_SPLIT_MIN_REQUESTS or requests % t or rows % rows_per_seq:
        return None
    share = (requests // t) * rows_per_seq
    return rank * share, (rank + 1) * share


def _index_split_plan(split, qsa, num_tokens: int, rows_per_seq: int):
    """A work-balanced request plan for the split index selection, or None to split by position.

    Made once a step for every layer, over the Lamport gather only; splitting by position puts the
    CUDA graph padding on one rank. Bitwise equal to the position split."""
    if _index_split_rows(split, num_tokens, rows_per_seq) is None:
        return None
    lam = split[3] if len(split) > 3 else None
    req_idx, logical, seq_lens_q, visible_blocks, _, row_starts = qsa[:6]
    vectors = (visible_blocks, req_idx, logical, seq_lens_q, row_starts)
    if lam is None or not all(v.is_contiguous() for v in vectors):
        return None
    return index_split_plan(
        visible_blocks[:num_tokens],
        req_idx[:num_tokens],
        logical[:num_tokens],
        seq_lens_q[:num_tokens],
        row_starts[:num_tokens],
        num_tokens // rows_per_seq,
        rows_per_seq,
        len(split[0]),
        split[2],
    )


def _decode_index_selection(
    core, split, q, qsa, num_tokens, p, device, index_cache, plan=None, q_permuted=False, order=None
):
    """A sparse decode layer's int32 `[num_tokens, token_topk + compress_ratio - 1]` selection.

    Index scores of `q` (`p` rows a request), then `_selection`, over every row or, under `split`,
    over this rank's rows gathered in rank order. `qsa` is `(req_idx, logical, seq_lens_q,
    visible_blocks, topk_indices, topk_row_starts, block_table, tokens_per_block)`. With a `plan`,
    this rank takes its slice of the permuted order (`q_permuted`: `q` is in it already). `order`:
    the step's `_RowOrder`, shared by its layers."""
    req_idx, logical, seq_lens_q, visible_blocks, topk_indices, topk_row_starts = qsa[:6]
    block_table, tokens_per_block = qsa[6], qsa[7]
    part = _index_split_rows(split, num_tokens, p)
    if part is None:
        plan = None
    lo, hi = (0, num_tokens) if part is None else part
    rows = hi - lo
    if plan is not None:
        q_rows = q[lo:hi] if q_permuted else q.index_select(0, plan.perm_rows[lo:hi])
        rows_of = (plan.req_idx, plan.logical, plan.seq_lens, plan.visible, plan.row_starts)
    else:
        q_rows = q[lo:hi]
        rows_of = (
            req_idx[lo:hi],
            logical[lo:hi],
            seq_lens_q[lo:hi],
            visible_blocks[lo:hi],
            topk_row_starts[lo:hi],
        )
    r_req, r_logical, r_seq, r_visible, r_starts = rows_of
    rpr = _index_rows_per_request(rows, p)
    if (
        rpr > 1
        and rows >= _INDEX_SCORES_CUTE_MIN_GROUPS * rpr
        and logical.dtype == torch.int64
        and req_idx.dtype == torch.int32
        and _index_groups_cute_serves(q_rows, index_cache, rpr)
    ):
        scores = index_scores_groups_cute(
            q_rows,
            index_cache,
            block_table,
            r_logical,
            r_req,
            tokens_per_block,
            core.compress_ratio,
            _INDEX_SCORES_PDL,
            rpr,
        )
    else:
        scores = qsa_paged_index_scores(
            q=q_rows,
            index_cache=index_cache,
            block_table=block_table,
            query_positions=r_logical,
            request_indices=r_req,
            tokens_per_block=tokens_per_block,
            compress_ratio=core.compress_ratio,
            only_visible_blocks=True,
            context_rows=False,
            use_pdl=_INDEX_SCORES_PDL,
            rows_per_request=rpr,
            wide_split=True,
        )
    selected = torch.empty(
        [rows, core.token_topk + core.compress_ratio - 1], dtype=torch.int32, device=device
    )
    _selection(
        core,
        scores,
        r_visible,
        r_logical,
        r_seq,
        topk_indices[lo:hi],
        r_starts,
        selected,
        order=order,
    )
    if part is None:
        return selected
    group, process_group = split[0], split[1]
    lam = split[3] if len(split) > 3 else None
    if lam is not None and lam.fits(num_tokens * selected.shape[1]):
        if plan is None:
            return lamport_allgather(selected, lam)
        return lamport_allgather(selected, lam, plan.perm_rows)
    gathered = allgather(selected, None, group, process_group)
    if plan is None:
        return gathered
    out = torch.empty_like(gathered)
    out.index_copy_(0, plan.perm_rows.long(), gathered)
    return out


def _index_rows_per_request(rows: int, rows_per_seq: int) -> int:
    """`qsa_paged_index_scores`' `rows_per_request`: grouped from `_INDEX_GROUP_MIN_ROWS` rows."""
    return rows_per_seq if rows_per_seq > 1 and rows >= _INDEX_GROUP_MIN_ROWS else 1


def _draft_continue_ok(core, md, p: int, share: bool) -> bool:
    """Whether `_draft_continue_rows` returns rows rather than None."""
    if p <= 1 or core.dp_size != 1 or share:
        return False
    return getattr(md, "qsa_mtp_num_accepted", None) is not None


def _draft_continue_rows(core, md, gen_seqs: int, p: int, share: bool):
    """The `[gen_seqs]` int64 rows a draft-extend step's outputs are read from, or None for all.

    Each request's last accepted row; the others only leave K/V and index keys behind. Not under
    attention DP (the MoE pads by the worker's rows) or IndexShare (it keeps every selection)."""
    if p <= 1 or core.dp_size != 1 or share:
        return None
    accepted = getattr(md, "qsa_mtp_num_accepted", None)
    if accepted is None:
        return None
    last = torch.clamp(accepted[:gen_seqs].to(torch.int64) - 1, min=0, max=p - 1)
    return torch.arange(gen_seqs, dtype=torch.int64, device=last.device) * p + last


class _RowOrder:
    """A step's sparse decode rows, longest visible window first, for the ordered launch of
    `qsa_topk_expand_decode`: sorted at the first layer that launches it, then read by the step's
    other layers, which select over the same rows."""

    def __init__(self):
        self._order = None

    def __call__(self, visible_blocks):
        if self._order is None:
            self._order = torch.argsort(visible_blocks, descending=True, stable=True).to(
                torch.int32
            )
        assert self._order.shape == visible_blocks.shape, "one row order serves one row set"
        return self._order


def _selection(
    core,
    scores,
    visible_blocks,
    logical,
    seq_lens_q,
    topk_indices,
    row_starts,
    out,
    context=False,
    order=None,
):
    """The canonical block top-k and expansion of `scores`' rows into `out`, fused if possible.
    `order`: a `_RowOrder` over these rows, for the ordered wide launch."""
    if (
        context
        and scores.shape[0] >= _TOPK_ROWS_MIN_ROWS
        and _topk_rows_supports(scores, topk_indices)
    ):
        qsa_topk_rows(
            scores,
            visible_blocks,
            logical,
            seq_lens_q,
            topk_indices,
            core.compress_ratio,
            core.token_topk,
            out=out,
        )
        return
    if _topk_expand_cluster_supports(scores):
        qsa_topk_expand_cluster(
            scores,
            visible_blocks,
            logical,
            seq_lens_q,
            topk_indices,
            core.compress_ratio,
            core.token_topk,
            out=out,
        )
        return
    if _topk_expand_supports(scores):
        row_order = None
        if order is not None and _topk_expand_orders_rows(scores):
            row_order = order(visible_blocks)
        qsa_topk_expand_decode(
            scores,
            visible_blocks,
            logical,
            seq_lens_q,
            topk_indices,
            core.compress_ratio,
            core.token_topk,
            out=out,
            row_order=row_order,
        )
        return
    _select(core, scores, row_starts, visible_blocks, topk_indices)
    out.copy_(
        qsa_expand_block_indices(
            topk_indices,
            logical,
            seq_lens_q,
            compress_ratio=core.compress_ratio,
            token_topk=core.token_topk,
        )
    )


def _prefill_split_enabled(attn_tp_size: int, dp_size: int) -> bool:
    """Whether a prefill step splits its selection and row-wise work over the attention-TP group."""
    return dp_size == 1 and attn_tp_size in _PREFILL_SPLIT_DEFAULT_TP


def _selection_cuts(chunks, starts, world: int, row_tokens: int = 0) -> tuple:
    """Row boundaries `(0, c_1, ..., c_{world-1}, rows)` of a context step's split selection.

    A row at position `x` costs `x + 1 + row_tokens`; cut `r` takes `r / world` of the total,
    rounded to `_SELECTION_CUT_ALIGN`. Host arithmetic, identical on every rank."""
    rows = sum(chunks)
    base = [s + 1 + row_tokens for s in starts]
    # Summed cost of a request's first `m` rows: m * base + m (m - 1) / 2.
    totals = [c * b + c * (c - 1) // 2 for c, b in zip(chunks, base)]
    total = sum(totals)
    cuts = [0]
    req, row0, done = 0, 0, 0
    for r in range(1, world):
        want = total * r // world
        while req < len(chunks) - 1 and done + totals[req] < want:
            done += totals[req]
            row0 += chunks[req]
            req += 1
        # The fewest rows `m` of request `req` whose cost reaches `want - done`: the root of
        # m^2 / 2 + (b - 1 / 2) m = rest, refined on the integers.
        b, c, rest = base[req], chunks[req], want - done
        m = int(-(b - 0.5) + ((b - 0.5) ** 2 + 2 * rest) ** 0.5)
        m = max(0, min(c, m))
        while m < c and m * b + m * (m - 1) // 2 < rest:
            m += 1
        while m > 0 and (m - 1) * b + (m - 1) * (m - 2) // 2 >= rest:
            m -= 1
        cut = int(round((row0 + m) / _SELECTION_CUT_ALIGN)) * _SELECTION_CUT_ALIGN
        cuts.append(min(rows, max(cuts[-1], cut)))
    cuts.append(rows)
    return tuple(cuts)


def _split_context_selection(
    split, score_and_select, selected, lo: int, hi: int, pipeline=None, cuts=None
) -> None:
    """Rows `[lo, hi)` of `selected` scored and selected over the attention-TP group.

    Each rank runs `score_and_select` on its share (equal, or by `cuts`) into its own
    `[per, width]` buffer; one uniform gather lays the shares out in rank order (int32 as bf16
    pairs), bitwise the unsplit result. A collective: every rank calls it with the same `hi - lo`.
    `pipeline` `(pieces, comm, fork, join[, exchange])` gathers each piece on `comm` while the
    next is scored."""
    rank, world, group, process_group = split
    rows = hi - lo
    if cuts is None:
        per = -(-rows // world)
        starts = [min(lo + r * per, hi) for r in range(world)]
        ends = [min(s + per, hi) for s in starts]
    else:
        assert len(cuts) == world + 1 and cuts[0] == 0 and cuts[-1] == rows, (cuts, rows)
        starts = [lo + c for c in cuts[:-1]]
        ends = [lo + c for c in cuts[1:]]
        per = max(e - s for s, e in zip(starts, ends))
    a, b = starts[rank], ends[rank]
    mine = torch.empty([per, selected.shape[1]], dtype=selected.dtype, device=selected.device)
    pieces = 1
    if pipeline is not None:
        pieces = max(1, min(pipeline[0], per // _SELECTION_PIECE_MIN_ROWS))
    if pieces == 1:
        if b > a:
            score_and_select(a, b, True, mine[: b - a])
        gathered = allgather(mine.view(torch.bfloat16), None, group, process_group)
        gathered = gathered.view(selected.dtype)
        if cuts is None:
            selected[lo:hi].copy_(gathered[:rows])
            return
        for r in range(world):
            n = ends[r] - starts[r]
            if n > 0:
                selected[starts[r] : ends[r]].copy_(gathered[r * per : r * per + n])
        return
    _, comm, fork, join = pipeline[:4]
    exchange = pipeline[4] if len(pipeline) > 4 else None
    size = -(-per // pieces)
    width = selected.shape[1] * selected.element_size()
    if exchange is not None and (world != 2 or size * width > exchange.capacity):
        exchange = None
    main = torch.cuda.current_stream()
    for p0 in range(0, per, size):
        p1 = min(p0 + size, per)
        if a + p0 < b:
            score_and_select(a + p0, min(a + p1, b), True, mine[p0 : min(p1, b - a)])
        fork.record(main)
        comm.wait_event(fork)
        with torch.cuda.stream(comm):
            if exchange is not None:
                # The peer's piece is a view of the receive buffer, copied out below.
                pieces_of = {rank: mine[p0:p1], 1 - rank: exchange.exchange(mine[p0:p1])}
            else:
                gathered = allgather(mine[p0:p1].view(torch.bfloat16), None, group, process_group)
                gathered = gathered.view(selected.dtype)
                pieces_of = {r: gathered[r * (p1 - p0) : (r + 1) * (p1 - p0)] for r in range(world)}
            for r in range(world):
                r0 = starts[r] + p0
                r1 = min(starts[r] + p1, ends[r])
                if r1 > r0:
                    selected[r0:r1].copy_(pieces_of[r][: r1 - r0])
    join.record(comm)
    main.wait_event(join)


def _decode_prologue(kv, sparse: bool) -> bool:
    """Whether a decode attention layer runs the fused prologue: sparse, over a bf16 or e4m3 pool.

    The dense op appends K/V itself and an NVFP4 pool needs the quantizing store."""
    return sparse and kv.k_scale is None


def _decode_prologue_call(
    core, proj, qkv, index_scratch, i, L, qsa, rows_per_request, index_q_rows=None
) -> None:
    """`qwen4_exp_attn_decode_prologue` over layer `i`'s pools.

    `index_q_rows[t]`, if given, is row `t`'s `index_scratch` row (the split's permuted order)."""
    req_idx, logical, block_table, tokens_per_block, pos3, coords = qsa
    pools = core._bound.pools
    kv = pools.kv[i]
    qwen4_exp_attn_decode_prologue(
        proj,
        qkv,
        _index_query(core, index_scratch),
        pos3,
        coords,
        core._cos_sin,
        L.q_norm,
        L.k_norm,
        L.idx_qn,
        L.idx_kn,
        kv.k,
        kv.v,
        pools.index_cache[i],
        pools.position_cache,
        req_idx,
        logical,
        block_table,
        tokens_per_block,
        core.compress_ratio,
        core.heads_q,
        core.heads_kv,
        core.head_dim,
        core.index_heads,
        core.index_head_dim,
        core.index_col,
        core.rotary_dim,
        core.eps,
        core.mrope_section1,
        core.mrope_section2,
        rows_per_request=rows_per_request,
        index_q_rows=index_q_rows,
    )


def _store_kv(core, kv, k, v, request_indices, logical_positions, block_table, tokens_per_block):
    """Append this step's K/V rows to the layer's paged pool.

    An fp8 pool takes e4m3 unscaled, matching the dense op's unit KV scales."""
    if kv.k_scale is None:
        qsa_paged_kv_store(
            k, v, kv.k, kv.v, request_indices, logical_positions, block_table, tokens_per_block
        )
        return
    qsa_paged_kv_store_nvfp4(
        k,
        v,
        kv.k,
        kv.v,
        kv.k_scale,
        kv.v_scale,
        request_indices,
        logical_positions,
        block_table,
        tokens_per_block,
        core.kv_scale_orig_quant,
    )


def _fp8_context_pages(md, tpb: int) -> int:
    """Pool pages the step's context requests hold, `ceil(kv_len / tpb)` each, from host lengths."""
    lens = md.kv_lens_runtime[: md.num_contexts]
    return int(((lens + (tpb - 1)) // tpb).sum())


def _rowtile_takes(kv, rows: int, fp8_pages) -> bool:
    """Whether `_sparse_gqa` sends a context launch of `rows` rows to the row-tile pass.

    Mirrors `_sparse_gqa_rowtile`'s bound on the host; the pass may take q as a strided view."""
    if kv.k_scale is not None or rows < _ROWTILE_MIN_ROWS:
        return False
    if kv.k.dtype == torch.bfloat16:
        return True
    return (
        kv.k.dtype == torch.float8_e4m3fn
        and fp8_pages is not None
        and fp8_pages * kv.k[0].numel() * 4 <= _FP8_ROWTILE_MAX_SCRATCH
    )


def _rowtile_scratch_groups(md, page_bytes: int) -> tuple | None:
    """Row-tile launches `(first row, end row, pages)` of an fp8 context step, or None for one.

    The fewest groups of consecutive requests whose widened pages fit `_FP8_ROWTILE_MAX_SCRATCH`,
    pages spread evenly; a request never splits (its pages always fit)."""
    num_contexts = md.num_contexts
    tpb = md.tokens_per_block
    rows = [int(r) for r in md.seq_lens[:num_contexts].tolist()]
    pages = [-(-int(k) // tpb) for k in md.kv_lens_runtime[:num_contexts].tolist()]
    cap = _FP8_ROWTILE_MAX_SCRATCH // page_bytes
    if sum(pages) <= cap:
        return None

    def fill(limit: int) -> list:
        # Contiguous groups, each closed before a request that would take it past `limit`.
        out, first, held = [], 0, 0
        for j, p in enumerate(pages):
            if held and held + p > limit:
                out.append((first, j, held))
                first, held = j, 0
            held += p
        out.append((first, len(pages), held))
        return out

    fewest = len(fill(cap))
    lo, hi = max(pages), cap
    while lo < hi:
        mid = (lo + hi) // 2
        if len(fill(mid)) <= fewest:
            hi = mid
        else:
            lo = mid + 1
    starts = [0]
    for r in rows:
        starts.append(starts[-1] + r)
    return tuple((starts[a], starts[b], n) for a, b, n in fill(lo))


def _sparse_gqa_rowtile(
    core, kv, q, block_table, selected, request_indices, tpb, query_positions, fp8_pages
):
    """The row-tile pass over a context launch, or None where it does not serve it.

    An fp8 pool's reached pages are widened exactly into a bf16 scratch pool; `fp8_pages` is
    their host count (read from the device when None)."""
    k_cache, v_cache, table = kv.k, kv.v, block_table
    if kv.k.dtype == torch.float8_e4m3fn:
        # Each slot's highest position, -1 if absent: a compare-and-max over `[slots, rows]`
        # (a scatter max would serialize a request's rows on one address).
        slots = torch.arange(block_table.shape[0], dtype=request_indices.dtype, device=q.device)
        held = request_indices[None, :] == slots[:, None]
        last = torch.where(held, query_positions[None, :].long(), -1).amax(1)
        counts = torch.div(last + tpb, tpb, rounding_mode="floor")
        if fp8_pages is None:
            fp8_pages = int(counts.sum())
        if fp8_pages * kv.k[0].numel() * 4 > _FP8_ROWTILE_MAX_SCRATCH:
            return None
        k_cache = torch.empty([fp8_pages, *kv.k.shape[1:]], dtype=q.dtype, device=q.device)
        v_cache = torch.empty_like(k_cache)
        table = torch.full_like(block_table, -1)
        qsa_fp8_pages_to_bf16(
            kv.k,
            kv.v,
            block_table,
            slots.int(),
            counts,
            torch.cumsum(counts, 0) - counts,
            table,
            k_cache,
            v_cache,
        )
    elif kv.k.dtype != q.dtype:
        return None
    return qsa_sparse_gqa_rowtile(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=table,
        selected_tokens=selected,
        request_indices=request_indices,
        tokens_per_block=tpb,
        softmax_scale=core.softmax_scale,
        query_positions=query_positions,
        compress_ratio=core.compress_ratio,
    )


def _sparse_gqa(
    core,
    kv,
    q,
    block_table,
    selected,
    request_indices,
    tpb,
    query_positions,
    fp8_pages=None,
    output_gate=None,
    rows_per_request=1,
):
    """Sparse GQA over the layer's paged pool.

    `query_positions` bounds each context row's scan (None for decode rows). Decode rows only:
    `output_gate` `[rows, Hq * D]` or `[rows, Hq, D]` makes the result `bf16(attn) *
    sigmoid(gate)`, and `rows_per_request` is the consecutive rows of each request."""
    if kv.k_scale is None and query_positions is None:
        if (
            rows_per_request > 1
            and q.shape[0] >= _GROUPED_DECODE_MIN_REQUESTS * rows_per_request
            and qsa_sparse_gqa_grouped_cute.serves(q, kv.k, selected, rows_per_request)
        ):
            return qsa_sparse_gqa_grouped_cute(
                q,
                kv.k,
                kv.v,
                block_table,
                selected,
                request_indices,
                tpb,
                core.softmax_scale,
                rows_per_request,
                output_gate,
            )
        if qsa_sparse_gqa_splitk_cute.serves(q, kv.k, selected):
            # Compile the split-K fallback's row buckets at warm-up, not on a serving step.
            warm_decode_buckets(
                q=q,
                k_cache=kv.k,
                v_cache=kv.v,
                block_table=block_table,
                selected_tokens=selected,
                request_indices=request_indices,
                tokens_per_block=tpb,
                softmax_scale=core.softmax_scale,
                compress_ratio=core.compress_ratio,
            )
            return qsa_sparse_gqa_splitk_cute(
                q,
                kv.k,
                kv.v,
                block_table,
                selected,
                request_indices,
                tpb,
                core.softmax_scale,
                output_gate,
            )
    out = _sparse_gqa_ungated(
        core, kv, q, block_table, selected, request_indices, tpb, query_positions, fp8_pages
    )
    if output_gate is not None:
        fused_sigmoid_mul_inplace(
            torch.reshape(out, [out.shape[0], -1]), output_gate, direct_launch=True
        )
    return out


def _sparse_gqa_ungated(
    core, kv, q, block_table, selected, request_indices, tpb, query_positions, fp8_pages=None
):
    """`_sparse_gqa` without the CuTe decode kernels and without the output gate."""
    if kv.k_scale is None:
        if query_positions is not None and q.shape[0] >= _ROWTILE_MIN_ROWS:
            out = _sparse_gqa_rowtile(
                core, kv, q, block_table, selected, request_indices, tpb, query_positions, fp8_pages
            )
            if out is not None:
                return out
        return qsa_paged_sparse_gqa(
            q=q,
            k_cache=kv.k,
            v_cache=kv.v,
            block_table=block_table,
            selected_tokens=selected,
            request_indices=request_indices,
            tokens_per_block=tpb,
            softmax_scale=core.softmax_scale,
            query_positions=query_positions,
            compress_ratio=None if query_positions is None else core.compress_ratio,
            output_gate=None,
        )
    if query_positions is None:
        return qsa_paged_sparse_gqa_nvfp4_decode(
            q,
            kv.k,
            kv.v,
            kv.k_scale,
            kv.v_scale,
            core.kv_scale_quant_orig,
            block_table,
            selected,
            request_indices,
            tpb,
            core.softmax_scale,
            None,
        )
    return qsa_paged_sparse_gqa_nvfp4(
        q,
        kv.k,
        kv.v,
        kv.k_scale,
        kv.v_scale,
        core.kv_scale_quant_orig,
        block_table,
        selected,
        request_indices,
        tpb,
        core.softmax_scale,
        query_positions=query_positions,
        compress_ratio=core.compress_ratio,
    )


def _copy_positions(position_ids: torch.Tensor) -> torch.Tensor:
    """A contiguous int32 `[3, N]` copy of the mRoPE positions; the rope rejects the engine's."""
    pos3 = torch.empty(
        [_QSA_POSITION_AXES, position_ids.numel() // _QSA_POSITION_AXES],
        dtype=torch.int32,
        device=position_ids.device,
    )
    pos3.copy_(torch.reshape(position_ids, [_QSA_POSITION_AXES, -1]))
    return pos3


def _step_args(md: TrtllmAttentionMetadata, num_contexts, num_ctx_tokens, paged_context) -> dict:
    """thop_attention's batch state off the prepared metadata.

    Tensors are engine-owned buffers refreshed in place; ints are per-capture constants.
    `paged_context` forces the paged-context path the engine enables only for chunked prefill
    or KV cache block reuse: an NVFP4 pool's context kernels exist on it alone."""
    return dict(
        sequence_length=md.kv_lens_cuda_runtime,
        host_past_key_value_lengths=md.kv_lens_runtime,
        host_total_kv_lens=md.host_total_kv_lens,
        context_lengths=md.prompt_lens_cuda_runtime,
        host_context_lengths=md.prompt_lens_cpu_runtime,
        host_request_types=md.host_request_types_runtime,
        kv_cache_block_offsets=md.kv_cache_block_offsets,
        host_kv_cache_pool_pointers=md.host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping=md.host_kv_cache_pool_mapping,
        workspace=md.effective_workspace,
        tokens_per_block=md.tokens_per_block,
        max_num_requests=md.max_num_requests,
        max_context_length=md.max_context_length,
        max_seq_len=md.max_seq_len,
        attention_window_size=md.max_seq_len,
        num_contexts=num_contexts,
        num_ctx_tokens=num_ctx_tokens,
        trtllm_gen_jit_warmup=md.trtllm_gen_jit_warmup,
        use_paged_context_fmha=md.use_paged_context_fmha or paged_context,
        beam_width=md.effective_beam_width,
        cache_indirection=md.cache_indirection,
        block_ids_per_seq=md.block_ids_per_seq,
        max_context_q_len_override=md.max_context_q_len_override,
        is_cross=md.is_cross,
        is_spec_decoding_enabled=md.is_spec_decoding_enabled,
        use_spec_decoding=md.use_spec_decoding,
        is_spec_dec_tree=md.is_spec_dec_tree,
        spec_decoding_generation_lengths=md.spec_decoding_generation_lengths,
        spec_decoding_position_offsets=get_spec_decoding_position_offsets(md),
        spec_decoding_packed_mask=md.spec_decoding_packed_mask,
        spec_decoding_bl_tree_mask_offset=md.spec_decoding_bl_tree_mask_offset,
        spec_decoding_bl_tree_mask=md.spec_decoding_bl_tree_mask,
        spec_decoding_target_max_draft_tokens=md.max_total_draft_tokens,
        spec_bl_tree_first_sparse_mask_offset_kv=md.spec_bl_tree_first_sparse_mask_offset_kv,
        num_sparse_topk=md.num_sparse_topk,
        flash_mla_tile_scheduler_metadata=md.flash_mla_tile_scheduler_metadata,
        flash_mla_num_splits=md.flash_mla_num_splits,
        max_num_sequences=md.max_num_sequences,
        force_prepare_spec_dec_tree_mask=md.force_prepare_spec_dec_tree_mask,
    )


# ------------------------------------------------------------------ prefill ---


def _pad_window(num_tokens: int, rows: int) -> int:
    """First row a break graph over a `rows` bucket zeroes from; part of the graph key.

    Half the bucket if the batch fills more than half (any power-of-two bucket ladder), else 0."""
    return rows // 2 if num_tokens > rows // 2 else 0


class _PrefillStep(NamedTuple):
    """One step's host-side state, built fresh by `PrefillTarget.prepare` and read by the breaks.

    `num_tokens` is the batch's rows `n`, `physical_tokens` the `T` the forward runs over (`n` or
    its CUDA graph bucket), `dp_pad_rows` the MoE padding (attention DP without a CUDA graph)."""

    num_tokens: int
    physical_tokens: int
    dp_pad_rows: _DpPadding | None
    input_ids: torch.Tensor
    attn: dict
    gdn: tuple
    ple: tuple
    qsa: tuple
    vision: tuple | None
    # A verify step: the generation rows' side-cache writes are staged for the rollback.
    stage: bool = False
    # fp8 pool only: `_fp8_context_pages`.
    fp8_pages: int | None = None
    # fp8 pool only: `_rowtile_scratch_groups`, None for one launch.
    rowtile_groups: tuple | None = None
    split: _Split | None = None
    # Graph-replayable steps only: QSA per-row metadata over all `T` rows (padding = last row).
    qsa_rows: tuple | None = None
    # `_selection_cuts`, or None for equal row counts.
    sel_cuts: tuple | None = None


class PrefillTarget:
    """Any batch with context rows (`num_contexts > 0`), possibly with generation rows."""

    def __init__(self, core: Qwen4ExpUnifiedModelingV2) -> None:
        self.core = core
        self._contract_pending = step_contract_enabled()
        self._layers = _layer_views(core)
        self._stack = _stack_views(core)
        self._step: _PrefillStep | None = None
        # `_attn_dispatch` by (rows, score-table width).
        self._attn_dispatch_memo: dict = {}
        self._coords_buf: torch.Tensor | None = None
        # Keyed CUDA graphs on short buckets; a key without a graph runs the eager core.
        self._attention_break = _keyed_eager_on_graph(
            self._attention_core,
            graph_key=self._attn_graph_key,
            graph_body=self._attention_core_graph,
        )
        self._gdn_break = _keyed_eager_on_graph(
            self._gdn_core, graph_key=self._gdn_graph_key, graph_body=self._gdn_core_graph
        )
        # Batch row count `n` (`[1]` int32), refreshed by `prepare` before every forward or
        # replay so a CUDA graph can tell its padding rows; allocated once, at a fixed address.
        self._live_rows: torch.Tensor | None = None
        # `(rank, world, group, process_group)` of the attention-TP group, or None.
        split = None
        if _prefill_split_enabled(core.attn_tp_size, core.dp_size):
            split = (
                core.mapping.tp_rank,
                core.attn_tp_size,
                list(core.mapping.tp_group),
                core.mapping.tp_group_pg.boxed() if mpi_disabled() else None,
            )
        self._selection_split = split
        self._rowtile_groups = _weights.switch(ROWTILE_SCRATCH_GROUPS_ENV)
        self._sequence_parallel = split
        # TP sum, or reduce-scatter under SP; set by `forward`, a capture-time constant per bucket.
        self._block_sum = core.tp_sum
        # `(lo, hi, gather)` of a sequence-parallel forward, or None.
        self._sp_rows = None
        # Highest priority: in a CUDA graph a default-priority stream's gather waits for the
        # main stream's persistent GEMM, which holds every SM.
        self._comm_stream = None
        self._comm_fork = None
        self._comm_join = None
        side = getattr(core, "_side_stream", None)
        pipelined = self._selection_split is not None
        if (self._sequence_parallel is not None or pipelined) and side is not None:
            self._comm_stream = torch.cuda.Stream(device=side.device, priority=-1)
            self._comm_fork = torch.cuda.Event()
            self._comm_join = torch.cuda.Event()
        # Two-rank sequence parallelism only: sized for half a max_num_tokens step's bf16 rows.
        self._ce_exchange = None
        if (
            self._comm_stream is not None
            and self._sequence_parallel is not None
            and self._sequence_parallel[1] == 2
        ):
            from .ops.comm.ce_exchange__sp_pair import CopyEngineExchange

            rows = -(-core.max_num_tokens // 2)
            self._ce_exchange = CopyEngineExchange.create(core.mapping, rows * core.hidden * 2)

    def _sp_split(self, num_tokens: int):
        """`(lo, hi, gather, scatter)` of a sequence-parallel forward over `num_tokens`, or None.

        `[lo, hi)` is this rank's share; depends on `num_tokens` alone, the same on every rank."""
        sp = self._sequence_parallel
        if sp is None or num_tokens < _SP_MIN_ROWS:
            return None
        rank, world, group, process_group = sp
        if num_tokens % world:
            return None
        rows = num_tokens // world

        def gather(t: torch.Tensor) -> torch.Tensor:
            return allgather(t, None, group, process_group)

        ce = getattr(self, "_ce_exchange", None) if world == 2 else None

        def scatter(t: torch.Tensor) -> torch.Tensor:
            if ce is not None:
                # Send the peer its rows, add the peer's partial of this rank's rows.
                lo, hi = rank * rows, (rank + 1) * rows
                peer = t[hi:] if rank == 0 else t[:lo]
                return t[lo:hi] + ce.exchange(peer.contiguous())
            return reducescatter(t, None, group, process_group)

        comm = getattr(self, "_comm_stream", None)
        if ce is not None and comm is not None:
            # Split around the producer: `start` sends early on the comm stream, `finish` adds.
            lo, hi = rank * rows, (rank + 1) * rows
            peer_rows = (hi, num_tokens) if rank == 0 else (0, lo)
            box = []

            def start(t: torch.Tensor) -> None:
                main = torch.cuda.current_stream()
                self._comm_fork.record(main)
                comm.wait_event(self._comm_fork)
                with torch.cuda.stream(comm):
                    box.append(ce.exchange(t[peer_rows[0] : peer_rows[1]]))

            def finish(t: torch.Tensor):
                self._comm_join.record(comm)
                torch.cuda.current_stream().wait_event(self._comm_join)
                return t[lo:hi] + box.pop()

            scatter.split_finalize = (start, finish, peer_rows)

        return rank * rows, (rank + 1) * rows, gather, scatter

    def _block_rows(self, x: torch.Tensor) -> torch.Tensor:
        """A block's input over every row: `x`, or the gather of every rank's rows under SP."""
        return x if self._sp_rows is None else self._sp_rows[2](x)

    def _overlap_projection(self) -> bool:
        """Whether a block's input projection runs as `_project_overlapped`."""
        return self._sp_rows is not None and self._comm_stream is not None

    def _project_overlapped(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """A sequence-parallel block's input projection over every row; `x` is this rank's rows.

        The gather on the comm stream overlaps this rank's rows; bitwise one `cublas_mm`."""
        lo, hi, gather = self._sp_rows
        total = (hi - lo) * self._sequence_parallel[1]
        out = torch.empty([total, w.shape[1]], dtype=x.dtype, device=x.device)
        main = torch.cuda.current_stream()
        comm = self._comm_stream
        self._comm_fork.record(main)
        comm.wait_event(self._comm_fork)
        ce = getattr(self, "_ce_exchange", None)
        with torch.cuda.stream(comm):
            # Only the peer's rows: a view of the receive buffer, read before the next exchange.
            whole = ce.exchange(x.contiguous()) if ce is not None else gather(x)
        torch.mm(x, w, out=out[lo:hi])
        self._comm_join.record(comm)
        main.wait_event(self._comm_join)
        if ce is not None:
            peer = slice(hi, total) if lo == 0 else slice(0, lo)
            torch.mm(whole, w, out=out[peer])
            return out
        if lo:
            torch.mm(whole[:lo], w, out=out[:lo])
        if hi < total:
            torch.mm(whole[hi:], w, out=out[hi:])
        return out

    def _overlap_reduce(self) -> bool:
        """Whether a block's output projection reduces as `_reduce_overlapped`."""
        return (
            self._sp_rows is not None
            and self._comm_stream is not None
            and self._sequence_parallel[1] == 2
        )

    def _reduce_overlapped(self, a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """A two-rank SP block's output `sum over ranks of a @ w` on this rank's rows `[lo, hi)`.

        The peer's rows are projected first and exchanged while this rank's rows are projected."""
        lo, hi, gather = self._sp_rows
        rows = hi - lo
        rank = lo // rows
        peer = a[hi:] if rank == 0 else a[:lo]
        sent = torch.mm(peer, w)
        main = torch.cuda.current_stream()
        comm = self._comm_stream
        self._comm_fork.record(main)
        comm.wait_event(self._comm_fork)
        ce = getattr(self, "_ce_exchange", None)
        with torch.cuda.stream(comm):
            if ce is not None:
                got = ce.exchange(sent)  # the peer's product of this rank's rows
            else:
                got = gather(sent)[(1 - rank) * rows : (2 - rank) * rows]
        own = torch.mm(a[lo:hi], w)
        self._comm_join.record(comm)
        main.wait_event(self._comm_join)
        return own + got

    def _split_context_rows(self, rows: int) -> bool:
        """Whether `rows` context rows take `_split_context_selection`.

        A host decision on the batch, identical on every rank, so all take the gather or none."""
        return self._selection_split is not None and rows >= _SELECTION_SPLIT_MIN_ROWS

    def _selection_pipeline(self):
        """`_split_context_selection`'s `pipeline`, or None to gather the share once."""
        if self._comm_stream is None:
            return None
        pipeline = (_SELECTION_PIECES, self._comm_stream, self._comm_fork, self._comm_join)
        ce = getattr(self, "_ce_exchange", None)
        return pipeline if ce is None else pipeline + (ce,)

    def _check_step_contract(self, md) -> None:
        """Opt-in first-forward check: surfaces an upstream metadata rename before the forward."""
        if not self._contract_pending:
            return
        try:
            self.step_args(md)
        except AttributeError as exc:
            raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
        self._contract_pending = False

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        """Project the prepared metadata onto thop_attention's batch state."""
        return _step_args(
            md, md.num_contexts, md.num_ctx_tokens, self.core.quant_mode == QM_NVFP4_KV_CACHE
        )

    # --------------------------------------------------------------- blocks ---

    def _attention_block(self, x, i, L, proj_qkv, index_scratch, attn_out, pos3, pos):
        """One gated GQA layer plus its QSA index branch, at this rank's heads.

        Norm + rope run over all `T` rows; caches, selection and attention run in the break."""
        core = self.core
        if self._overlap_projection():
            proj = self._project_overlapped(x, L.attn_in)
        else:
            proj = cublas_mm(self._block_rows(x), L.attn_in)
        gate = torch.split(proj, [core.qkv_width, core.q_width, core.index_scratch_width], 1)[1]
        _attention_rope(core, proj, index_scratch, L, pos3, pos)
        self._attention_break(proj, proj_qkv, index_scratch, attn_out, i)
        fused_sigmoid_mul_inplace(attn_out, gate, direct_launch=True)
        if self._overlap_reduce():
            return self._reduce_overlapped(attn_out, L.o)
        return self._block_sum(cublas_mm(attn_out, L.o))

    def _attn_dispatch(self, rows: int, width: int) -> tuple:
        """The sparse context path's launch choices that depend on its row count.

        A `T`-row graph is bitwise the eager break only where these match the `n` rows'."""
        memo = self._attn_dispatch_memo
        hit = memo.get((rows, width))
        if hit is not None:
            return hit
        core = self.core
        sel = core.token_topk + core.compress_ratio - 1
        q = torch.empty((rows, core.index_heads, core.index_head_dim), device="meta")
        scores = torch.empty((rows, width), device="meta")
        heads_kv = core.heads_kv
        gqa = "pair" if rows * heads_kv > 512 else _gqa_splitk_config(rows, heads_kv, sel)
        topk = torch.empty((rows, core.block_topk), dtype=torch.int32, device="meta")
        hit = (
            _scores_take_visible_split(q, True, False),
            rows >= _TOPK_ROWS_MIN_ROWS and _topk_rows_supports(scores, topk),
            _topk_expand_cluster_supports(scores),
            _topk_expand_supports(scores),
            gqa,
        )
        memo[(rows, width)] = hit
        return hit

    def _attn_graph_key(self, proj, proj_qkv, index_scratch, attn_out, i):
        """Graph key for `_attention_core_graph`, or None to run `_attention_core` eagerly.

        Context-only sparse steps over bf16 / fp8 pools without the row-tile pass, whose launch
        choices match at `n` and `T` rows. The key includes every buffer address the graph reads."""
        step = self._step
        if step is None or step.qsa_rows is None:
            return None
        rows = proj.shape[0]
        if rows != step.physical_tokens or step.attn["num_ctx_tokens"] != step.num_tokens:
            return None
        core = self.core
        kv = core._bound.pools.kv[i]
        n = step.num_tokens
        if (
            kv.k_scale is not None
            or _rowtile_takes(kv, n, step.fp8_pages)
            or step.rowtile_groups is not None
        ):
            return None
        block_table = step.qsa[6]
        width = block_table.shape[1] * step.qsa[7] // core.compress_ratio
        dispatch = self._attn_dispatch(n, width)
        if dispatch != self._attn_dispatch(rows, width):
            return None
        pools = core._bound.pools
        return (
            step.attn["num_contexts"],
            _pad_window(n, rows),
            dispatch,
            kv.k.data_ptr(),
            kv.v.data_ptr(),
            pools.index_cache[i].data_ptr(),
            pools.position_cache.data_ptr(),
            block_table.data_ptr(),
            tuple(block_table.shape),
            tuple(t.data_ptr() for t in step.qsa_rows),
            self._live_rows.data_ptr(),
        )

    def _attention_core_graph(self, proj, proj_qkv, index_scratch, attn_out, i) -> None:
        """`_attention_core` of a context-only sparse step as a CUDA graph over all `T` rows.

        Padding rows reuse the last row's metadata and store nothing (position -2 is skipped by the
        stores and never anchors a compression); their outputs are zeroed. Rows `[:n]` are bitwise
        the eager break's."""
        core = self.core
        L = self._layers[i]
        step = self._step
        req_idx, logical, seq_lens_q, visible_blocks, topk_indices, topk_row_starts, coords = (
            step.qsa_rows
        )
        block_table, tokens_per_block = step.qsa[6], step.qsa[7]
        rows = proj.shape[0]
        pad = torch.arange(rows, dtype=torch.int32, device=proj.device) >= self._live_rows
        store_positions = torch.where(pad, -2, logical)
        q_rows, k_dst, v_dst = _attn_in_heads(core, proj)
        _index_plane(
            core, proj, i, L, (req_idx, store_positions, block_table, tokens_per_block, coords)
        )
        kv = core._bound.pools.kv[i]
        _store_kv(core, kv, k_dst, v_dst, req_idx, store_positions, block_table, tokens_per_block)
        q_index = _index_query(core, index_scratch)
        selected = torch.empty(
            [rows, core.token_topk + core.compress_ratio - 1], dtype=torch.int32, device=proj.device
        )
        scores = _context_index_scores(
            q_index,
            core._bound.pools.index_cache[i],
            block_table,
            logical,
            req_idx,
            tokens_per_block,
            core.compress_ratio,
            True,
        )
        _selection(
            core,
            scores,
            visible_blocks,
            logical,
            seq_lens_q,
            topk_indices,
            topk_row_starts,
            selected,
            context=True,
        )
        proj_qkv.copy_(proj[:, : core.qkv_width])
        out = qsa_paged_sparse_gqa(
            q=_packed_q(core, proj_qkv),
            k_cache=kv.k,
            v_cache=kv.v,
            block_table=block_table,
            selected_tokens=selected,
            request_indices=req_idx,
            tokens_per_block=tokens_per_block,
            softmax_scale=core.softmax_scale,
            query_positions=logical,
            compress_ratio=core.compress_ratio,
            output_gate=None,
        )
        attn_out.copy_(torch.reshape(out, [rows, core.q_width]))
        lo = _pad_window(step.num_tokens, rows)
        attn_out[lo:].masked_fill_(torch.unsqueeze(pad[lo:], 1), 0)

    def _attention_core(self, proj, proj_qkv, index_scratch, attn_out, i) -> None:
        """Break: index store and compression, then dense or sparse attention over `n` rows.

        Writes rows `[:n]` of `attn_out` and zeroes `[n:]`. The index plane advances on every
        step so a request crossing the threshold finds its prefix compressed; a verify step stages
        it for `SpeculativeRollback` first."""
        core = self.core
        L = self._layers[i]
        step = self._step
        (
            req_idx,
            logical,
            seq_lens_q,
            visible_blocks,
            topk_indices,
            topk_row_starts,
            block_table,
            tokens_per_block,
            coords,
            sparse,
        ) = step.qsa
        num_tokens = step.num_tokens
        proj = proj[:num_tokens]
        q_rows, k_dst, v_dst = _attn_in_heads(core, proj)
        qkv = proj_qkv[:num_tokens]
        if not sparse:
            # The dense op reads a compact `[q | k | v]`.
            qkv.copy_(proj[:, : core.qkv_width])
        if step.stage:
            core.spec_rollback.stage_index(core._bound.pools.position_cache)
        _index_plane(core, proj, i, L, (req_idx, logical, block_table, tokens_per_block, coords))

        if sparse:
            kv = core._bound.pools.kv[i]
            # Nothing else writes the pool on this path; the selection reads the current token.
            _store_kv(core, kv, k_dst, v_dst, req_idx, logical, block_table, tokens_per_block)
            q_index = _index_query(core, index_scratch[:num_tokens])
            # Context rows come first and score as query tiles; generation rows score row by row.
            num_ctx = step.attn["num_ctx_tokens"]
            parts = [(0, num_ctx, True)]
            if num_ctx < num_tokens:
                parts.append((num_ctx, num_tokens, False))
            selected = torch.empty(
                [num_tokens, core.token_topk + core.compress_ratio - 1],
                dtype=torch.int32,
                device=proj.device,
            )

            # A call scores fewer than `_SCORE_TABLE_ELEMENTS / columns` rows (whole query
            # tiles): a row's scores and selection read only its query, position and request,
            # so the chunks select what one call would; below the bound a part is one call.
            columns = block_table.shape[1] * tokens_per_block // core.compress_ratio
            chunk = max(1, (_SCORE_TABLE_ELEMENTS - 1) // columns)
            if chunk >= _SELECTION_CUT_ALIGN:
                chunk -= chunk % _SELECTION_CUT_ALIGN

            def score_and_select(lo, hi, context_rows, out):
                for a in range(lo, hi, chunk):
                    b = min(a + chunk, hi)
                    scores = _context_index_scores(
                        q_index[a:b],
                        core._bound.pools.index_cache[i],
                        block_table,
                        logical[a:b],
                        req_idx[a:b],
                        tokens_per_block,
                        core.compress_ratio,
                        context_rows,
                    )
                    _selection(
                        core,
                        scores,
                        visible_blocks[a:b],
                        logical[a:b],
                        seq_lens_q[a:b],
                        topk_indices[a:b],
                        topk_row_starts[a:b],
                        out[a - lo : b - lo],
                        context=context_rows,
                    )

            for lo, hi, context_rows in parts:
                if context_rows and self._split_context_rows(hi - lo):
                    _split_context_selection(
                        self._selection_split,
                        score_and_select,
                        selected,
                        lo,
                        hi,
                        self._selection_pipeline(),
                        step.sel_cuts if lo == 0 and hi == num_ctx else None,
                    )
                else:
                    score_and_select(lo, hi, context_rows, selected[lo:hi])
            # The row-tile pass reads q in place; the other kernels need a compact copy.
            launches = []
            for lo, hi, context_rows in parts:
                if context_rows and step.rowtile_groups is not None:
                    launches += [(a, b, True, pages) for a, b, pages in step.rowtile_groups]
                else:
                    launches.append((lo, hi, context_rows, step.fp8_pages))
            for lo, hi, context_rows, pages in launches:
                if context_rows and _rowtile_takes(kv, hi - lo, pages):
                    q = q_rows[lo:hi]
                else:
                    qkv[lo:hi].copy_(proj[lo:hi, : core.qkv_width])
                    q = _packed_q(core, qkv)[lo:hi]
                out = _sparse_gqa(
                    core,
                    kv,
                    q,
                    block_table,
                    selected[lo:hi],
                    req_idx[lo:hi],
                    tokens_per_block,
                    logical[lo:hi] if context_rows else None,
                    pages,
                )
                attn_out[lo:hi].copy_(torch.reshape(out, [hi - lo, core.q_width]))
        else:
            thop_attention(
                q=qkv,
                output=attn_out[:num_tokens],
                local_layer_idx=core._bound.pools.pool_idx[i],
                num_heads=core.heads_q,
                num_kv_heads=core.heads_kv,
                head_size=core.head_dim,
                quant_mode=core.quant_mode,
                kv_scale_orig_quant=core.kv_scale_orig_quant,
                kv_scale_quant_orig=core.kv_scale_quant_orig,
                **step.attn,
                **_CALL_CONSTANTS,
            )
        attn_out[num_tokens:].zero_()

    def _gdn_block(self, x, i, L, core_out) -> torch.Tensor:
        """One Gated-DeltaNet layer over a batch with a context prefix, at this rank's heads.

        The projection and epilogue run over all `T` rows; the recurrence runs in `_gdn_core`."""
        core = self.core
        if self._overlap_projection():
            proj = self._project_overlapped(x, L.in_proj)
        else:
            proj = cublas_mm(self._block_rows(x), L.in_proj)
        z = torch.split(proj, core.in_proj_split, 1)[1]
        self._gdn_break(proj, core_out, i)
        return self._gdn_epilogue(core_out, z, L, proj.shape[0])

    def _gdn_graph_key(self, proj, out, i):
        """Graph key for `_gdn_core_graph`, or None to run `_gdn_core` eagerly.

        Context-only steps past the in-place threshold in short buckets. The key holds the request
        count, split shape, FlashInfer's CP chunk length and every buffer address read."""
        step = self._step
        if step is None or step.stage:
            return None
        rows = proj.shape[0]
        if rows != step.physical_tokens or rows > _GRAPH_BREAKS_MAX_TOKENS:
            return None
        n = step.num_tokens
        gdn = step.gdn
        num_decodes, verify = gdn[11], gdn[13]
        if num_decodes or verify is not None or n <= _GDN_INPLACE_MAX_TOKENS:
            return None
        si_all, hi_all, qsl_all, qsl_long = gdn[3], gdn[6], gdn[8], gdn[9]
        num_seqs = si_all.shape[0]
        core = self.core
        pools = core._bound.pools
        dev = proj.device
        heads = max(core.num_k_heads, core.num_v_heads)
        key = (
            num_seqs,
            _pad_window(n, rows),
            pools.ssm[i].data_ptr(),
            pools.conv[i].data_ptr(),
            si_all.data_ptr(),
            hi_all.data_ptr(),
            qsl_all.data_ptr(),
            qsl_long.data_ptr(),
            self._live_rows.data_ptr(),
        )
        sp = step.split
        if sp is None:
            cp = _gdn_reaches_cp(num_seqs, core.num_v_heads, dev)
            return ("whole", cp_chunk_len_for(n, heads, dev) if cp else 0) + key
        raw = sp.raw
        if raw is None or sp.whole is None and split_takes_cp(num_seqs, core.num_v_heads, dev):
            return None
        stage = core._bound.stage
        key += (
            sp.k,
            raw.tail_tok.shape[0],
            raw.cu.data_ptr(),
            stage.ssm[i].data_ptr(),
            stage.conv[i].data_ptr(),
        )
        if sp.whole is None:
            return ("split",) + key
        lead = sp.whole[0]
        if not isinstance(lead, int) or num_seqs != 1 or sp.k != 1:
            return None
        return ("lead", cp_chunk_len_for(lead, heads, dev)) + key

    def _gdn_core_graph(self, proj, out, i) -> None:
        """`_gdn_core` of a context-only step as a CUDA graph over all `T` rows.

        Padding rows never reach the batch's rows or states and are zeroed after, so rows `[:n]` of
        `out` and the pools are bitwise the eager break's."""
        core = self.core
        L = self._layers[i]
        step = self._step
        gdn = step.gdn
        si_all, hi_all, qsl_all, qsl_long = gdn[3], gdn[6], gdn[8], gdn[9]
        rows = proj.shape[0]
        dev = proj.device
        pools = core._bound.pools
        ssm_states = pools.ssm[i]
        mixed_qkv, _, a, b = torch.split(proj, core.in_proj_split, 1)
        # Padding rows are finite (the splice zeroes their embedding).
        q, k, v, g, beta = self._gdn_prefill_prep(mixed_qkv, a, b, L, i, qsl_all, si_all, hi_all)
        heads = max(core.num_k_heads, core.num_v_heads)
        sp = step.split
        if sp is None:
            num_seqs = si_all.shape[0]
            if _gdn_reaches_cp(num_seqs, core.num_v_heads, dev):
                fi = {
                    "_cp_chunk_len": cp_chunk_len_for(step.num_tokens, heads, dev),
                    "max_seqlen": rows,
                }
                _gdn_chunk_cp_inplace(
                    q, k, v, g, beta, None, ssm_states, si_all, hi_all, qsl_long, out, True,
                    fi_kwargs=fi,
                )  # fmt: skip
            else:
                _gdn_chunk_carried(
                    q, k, v, g, beta, None, ssm_states, si_all, hi_all, qsl_long, out, True,
                )  # fmt: skip
        else:
            raw = sp.raw
            stage = core._bound.stage
            stage_ssm, stage_conv = stage.ssm[i], stage.conv[i]
            gather = raw.gather.to(torch.int32)
            live = raw.live.to(torch.int32)
            if sp.whole is None:
                carry = torch.zeros([gather.shape[0]], dtype=torch.bool, device=dev)
                carry.index_copy_(0, raw.main_rows, hi_all.to(torch.bool))
                _gdn_chunk_carried(
                    q, k, v, g, beta, None, ssm_states, gather, carry, raw.cu, out, True,
                    keep_rows=raw.main_rows, keep_slots=live, use_cp=False,
                )  # fmt: skip
            else:
                # One request split once: the first pass covers its rows before P (`raw.cu[:2]`).
                fi = {
                    "_cp_chunk_len": cp_chunk_len_for(sp.whole[0], heads, dev),
                    "max_seqlen": rows,
                }
                _gdn_chunk_carried(
                    q, k, v, g, beta, None, ssm_states, live, hi_all.to(torch.bool), raw.cu[:2],
                    out, True, fi_kwargs=fi,
                )  # fmt: skip
            gdn_inforward_tail(
                q,
                k,
                v,
                g,
                beta,
                raw.tail_tok,
                raw.cu_tail,
                ssm_states,
                raw.tail_slots,
                stage_ssm,
                raw.stage_rows,
                out,
            )
            gdn_inforward_conv_swap(
                pools.conv[i], stage_conv, mixed_qkv, raw.tail_slots, raw.stage_rows, raw.conv_win
            )
        lo = _pad_window(step.num_tokens, rows)
        pad = torch.arange(lo, rows, dtype=torch.int32, device=dev) >= self._live_rows
        out[:, lo:].masked_fill_(pad.view(1, rows - lo, 1, 1), 0)

    def _gdn_core(self, proj, out, i) -> None:
        """Break: the recurrence over `n` rows; writes rows `[:n]` of `out`, zeroes rows `[n:]`.

        A context sequence with no carried state starts from zeros without clearing its slots
        (`has_initial_state`); a verify step's draft chains take `_gdn_verify`."""
        core = self.core
        L = self._layers[i]
        step = self._step
        (
            num_prefill_tokens,
            num_decode_tokens,
            si_p,
            si_all,
            si_d,
            hi_p,
            hi_all,
            qsl_p,
            qsl_all,
            qsl_long,
            qsl_long_p,
            num_decodes,
            gen_p,
            verify,
        ) = step.gdn
        num_tokens = step.num_tokens
        pools = core._bound.pools
        ssm_states = pools.ssm[i]
        mixed_qkv, _, a, b = torch.split(proj[:num_tokens], core.in_proj_split, 1)

        if verify is not None:
            mq_p, mq_d = torch.split(mixed_qkv, [num_prefill_tokens, num_decode_tokens], 0)
            a_p, a_d = torch.split(a, [num_prefill_tokens, num_decode_tokens], 0)
            b_p, b_d = torch.split(b, [num_prefill_tokens, num_decode_tokens], 0)
            q, k, v, g, beta = self._gdn_prefill_prep(mq_p, a_p, b_p, L, i, qsl_p, si_p, hi_p)
            flashinfer_chunk_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                scale=None,
                initial_state=ssm_states,
                initial_state_indices=si_p,
                inplace_indexed_state_update=True,
                output_final_state=False,
                cu_seqlens=qsl_long_p,
                head_first=False,
                use_qk_l2norm_in_kernel=False,
                output=out[:, :num_prefill_tokens],
                state_workspace=None,
                has_initial_state=hi_p,
                linear_gate=True,
            )
            out[:, num_prefill_tokens:num_tokens].copy_(
                _gdn_verify(
                    core,
                    mq_d,
                    a_d,
                    b_d,
                    i,
                    L,
                    si_d,
                    num_decodes,
                    gen_p,
                    verify,
                )
            )
        elif num_decode_tokens > 0:
            mq_p, mq_d = torch.split(mixed_qkv, [num_prefill_tokens, num_decode_tokens], 0)
            a_p, b_p = a[:num_prefill_tokens], b[:num_prefill_tokens]
            q, k, v, g, beta = self._gdn_prefill_prep(mq_p, a_p, b_p, L, i, qsl_p, si_p, hi_p)
            causal_conv1d_update(
                x=mq_d,
                conv_state=pools.conv[i],
                weight=L.conv_w,
                bias=None,
                silu_activation=True,
                cache_seqlens=None,
                conv_state_indices=si_d,
                pad_slot_id=_PAD_SLOT_ID,
            )
            # The two halves use disjoint slots and write disjoint row ranges of the output.
            flashinfer_chunk_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                scale=None,
                initial_state=ssm_states,
                initial_state_indices=si_p,
                inplace_indexed_state_update=True,
                output_final_state=False,
                cu_seqlens=qsl_long_p,
                head_first=False,
                use_qk_l2norm_in_kernel=False,
                output=out[:, :num_prefill_tokens],
                state_workspace=None,
                has_initial_state=hi_p,
                linear_gate=True,
            )
            # Reads the conv-updated q/k/v columns; the gated norm runs later, in the epilogue.
            gdn_decode_rows(
                proj[num_prefill_tokens:num_tokens],
                ssm_states,
                si_d,
                L.a_log,
                L.dt_bias,
                core.num_k_heads,
                core.head_k_dim,
                out[0, num_prefill_tokens:num_tokens].view(num_decode_tokens, core.value_dim),
            )
        elif step.split is not None:
            q, k, v, g, beta = self._gdn_prefill_prep(
                mixed_qkv, a, b, L, i, qsl_all, si_all, hi_all
            )
            self._gdn_split(q, k, v, g, beta, mixed_qkv, i, out[:, :num_tokens])
        else:
            q, k, v, g, beta = self._gdn_prefill_prep(
                mixed_qkv, a, b, L, i, qsl_all, si_all, hi_all
            )
            # q and k arrive L2-normalized; the recurrence must not repeat it.
            flashinfer_chunk_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                scale=None,
                initial_state=ssm_states,
                initial_state_indices=si_all,
                inplace_indexed_state_update=True,
                output_final_state=False,
                cu_seqlens=qsl_long,
                head_first=False,
                use_qk_l2norm_in_kernel=False,
                output=out[:, :num_tokens],
                state_workspace=None,
                has_initial_state=hi_all,
                linear_gate=True,
            )
        out[:, num_tokens:].zero_()

    def _gdn_split(self, q, k, v, g, beta, mixed_qkv, i, out) -> None:
        """The recurrence of a context-only step with an in-forward snapshot split (`_Split`).

        Leaves each split request's state at P in its live slot, its prompt-end state in its staging
        row, and every row's output in `out` `[1, n, HV, V]`. `chunk_split_first_pass` writes the
        state at P; `gdn_inforward_tail` runs the tails `[P, L)` from it into the staging rows
        (fp32, FlashInfer's to rounding); `gdn_inforward_conv_swap` swaps the conv windows."""
        core = self.core
        sp = self._step.split
        pools = core._bound.pools
        stage = core._bound.stage
        ssm_states, conv_states = pools.ssm[i], pools.conv[i]
        stage_ssm, stage_conv = stage.ssm[i], stage.conv[i]
        chunk_split_first_pass(
            q,
            k,
            v,
            g,
            beta,
            ssm_states,
            sp.gather,
            sp.carry,
            sp.cu,
            sp.main_rows,
            sp.live,
            out,
            num_whole_seqs=sp.live.shape[0],
            whole=sp.whole,
        )
        gdn_inforward_tail(
            q,
            k,
            v,
            g,
            beta,
            sp.tail_tok,
            sp.cu_tail,
            ssm_states,
            sp.tail_slots,
            stage_ssm,
            sp.stage_rows,
            out,
        )
        gdn_inforward_conv_swap(
            conv_states, stage_conv, mixed_qkv, sp.tail_slots, sp.stage_rows, sp.conv_win
        )

    def _gdn_prefill_prep(self, mq, a, b, L, i, qsl, si, hi):
        """The context rows' recurrence inputs `(q, k, v, g, beta)` in one pass over `mq`.

        Publishes the conv window back to the pool; q/k come out L2-normalized and the decay as the
        factor `exp(g)` (`linear_gate=True`)."""
        core = self.core
        return gdn_prefill_conv_post(
            x=mq,
            weight=L.conv_w,
            conv_states=core._bound.pools.conv[i],
            query_start_loc=qsl,
            cache_indices=si,
            has_initial_state=hi,
            a=a,
            b=b,
            A_log=L.a_log,
            dt_bias=L.dt_bias,
            num_k_heads=core.num_k_heads,
            head_k_dim=core.head_k_dim,
            num_v_heads=core.num_v_heads,
            head_v_dim=core.head_v_dim,
            pad_slot_id=_PAD_SLOT_ID,
            l2_norm_eps=_L2_NORM_EPS,
            softplus_threshold=_SOFTPLUS_THRESHOLD,
            linear_gate=True,
        )

    def _gdn_epilogue(self, out, z, L, num_tokens) -> torch.Tensor:
        """The gated RMS norm (sigmoid gate, non-Gemma weight) and the reduced output projection."""
        core = self.core
        o = rms_norm_gated_token_major(
            torch.reshape(out, [-1, core.head_v_dim]),
            torch.reshape(z, [num_tokens, core.num_v_heads, core.head_v_dim]),
            weight=L.gnorm,
            eps=core.eps,
            fp8_scale=None,
            gate_activation="sigmoid",
            direct_launch=True,
        )
        o = torch.reshape(o, [num_tokens, core.value_dim])
        if self._overlap_reduce():
            return self._reduce_overlapped(o, L.out)
        return self._block_sum(cublas_mm(o, L.out))

    def _ple_block(self, hc, hc_out, normed_out) -> None:
        """Break: the n-gram side path over `n` rows, added into the stream and normalized.

        Writes rows `[:n]` of `hc_out` and `normed_out`, zeroes `[n:]`. A sequence with no history
        is seeded with `eos_token_id`; a verify step stages its state for `SpeculativeRollback`."""
        core = self.core
        S = self._stack
        step = self._step
        conv_states, ngram_pool, si_all, carry_all, qsl = step.ple
        num_tokens = step.num_tokens
        next_norm = self._layers[core.ple_layer].ahc_norm
        conv_input = None
        if step.stage:
            core.spec_rollback.stage_ple_verify(ngram_pool, conv_states, step.input_ids)
            conv_input = torch.empty_like(hc[:num_tokens])
        elif step.split is not None:
            # The conv input rows rebuild the conv state at P (the op allocates them anyway).
            conv_input = torch.empty_like(hc[:num_tokens])
        ngram = ple_ngram_embed(
            step.input_ids,
            ngram_pool,
            si_all,
            carry_all,
            qsl,
            core.w["ple_table"],
            S.ple_tscale,
            S.ple_mult,
            S.ple_vocab,
            S.ple_offs,
            core.eos_token_id,
            num_tokens,
            row_window=core._ple_table_host.window,
        )
        if core._ple_table_host.needs_sum:
            ngram = core.tp_sum(ngram)
        ple_gate_conv_inject_norm(
            hc[:num_tokens],
            cublas_mm(ngram, S.ple_key),
            cublas_mm(ngram, S.ple_val),
            S.ple_nk_w,
            S.ple_nq_w,
            S.ple_nc_w,
            S.ple_conv_w,
            conv_states,
            si_all,
            qsl,
            next_norm,
            core.eps,
            core.hc_count,
            _PLE_CLAMP_MIN,
            core.ple_dilation,
            _PAD_SLOT_ID,
            num_tokens,
            carry_all,
            conv_input=conv_input,
            out=(hc_out[:num_tokens], normed_out[:num_tokens]),
        )
        if step.stage:
            core.spec_rollback.stage_ple_conv_input(conv_input)
        elif step.split is not None:
            self._split_ple_states(conv_states, ngram_pool, conv_input)
        hc_out[num_tokens:].zero_()
        normed_out[num_tokens:].zero_()

    def _split_ple_states(self, conv_states, ngram_pool, conv_input) -> None:
        """The PLE side of an in-forward snapshot split.

        Moves each split request's prompt-end state to its staging row and writes the state at P
        (the last conv-input rows and ids before P, oldest first) to its slot."""
        sp = self._step.split
        stage = self.core._bound.stage
        stage.ple_conv.index_copy_(0, sp.stage_rows, conv_states.index_select(0, sp.tail_slots))
        window = conv_input.index_select(0, sp.ple_win)
        conv_states.index_copy_(
            0, sp.tail_slots, window.view(sp.k, -1, window.shape[-1]).transpose(1, 2)
        )
        stage.ple_ngram.index_copy_(0, sp.stage_rows, ngram_pool.index_select(0, sp.tail_slots))
        ids = self._step.input_ids.index_select(0, sp.ngram_win)
        ngram_pool.index_copy_(0, sp.tail_slots, ids.view(sp.k, -1).to(ngram_pool.dtype))

    def _splice(self, h) -> None:
        """Break: write vision rows over their placeholders in `h`; zero rows `[n:]`."""
        step = self._step
        if step.vision is not None:
            rows, mm_token_indices = step.vision
            h.index_copy_(0, mm_token_indices, rows)
        h[step.num_tokens :].zero_()

    # -------------------------------------------------------------- forward ---

    def prepare(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor,
        position_ids: torch.IntTensor,
        vision: tuple | None,
        spec_metadata=None,
    ) -> None:
        """Build the step's host-side state (`_PrefillStep`) once, before the forward; on a
        breakable prefill graph replay this and the breaks are all of this target that runs."""
        core = self.core
        md = attn_metadata
        self._check_step_contract(md)
        attn_args = self.step_args(md)
        mm = md.mamba_metadata

        physical_tokens = input_ids.shape[0]
        assert physical_tokens in (md.num_tokens, md.padded_num_tokens), (
            f"the engine handed {physical_tokens} activation rows for a batch its metadata "
            f"describes as {md.num_tokens} tokens (padded_num_tokens={md.padded_num_tokens})"
        )
        num_tokens = md.num_tokens
        if self._live_rows is None:
            self._live_rows = torch.zeros([1], dtype=torch.int32, device=input_ids.device)
        self._live_rows.fill_(num_tokens)
        # Attention DP pads the MoE rows to the group's largest count; under a prefill CUDA
        # graph every rank runs one bucket, so this is None (a per-capture constant).
        dp_pad_rows = core._dp_pad_rows(md, physical_tokens)
        assert physical_tokens == num_tokens or dp_pad_rows is None, (
            f"a prefill CUDA graph bucket of {physical_tokens} rows with attention-DP padding "
            f"{dp_pad_rows}: the graph would freeze the group's row counts"
        )
        batch = md.num_seqs
        num_prefills = md.num_contexts
        num_decodes = batch - num_prefills
        num_prefill_tokens = md.num_ctx_tokens
        num_decode_tokens = num_tokens - num_prefill_tokens
        gen_p, is_verify = _verify_rows(core, spec_metadata, num_decodes, num_decode_tokens)
        attn_args["predicted_tokens_per_seq"] = gen_p

        si_all = mm.state_indices[:batch]
        hi_all = mm.has_initial_states[:batch]
        si_p, si_d = torch.split(si_all, [num_prefills, num_decodes], 0)
        # The decode kernel needs a 16-byte-aligned slot vector; the generation half starts
        # `num_prefills` rows in, so copy it once per step.
        si_d = torch.clone(si_d)
        hi_p = hi_all[:num_prefills]
        # `query_start_loc_long` spans the whole batch; its prefix segments the context rows.
        qsl_all = mm.query_start_loc
        qsl_p = None if qsl_all is None else qsl_all[: num_prefills + 1]
        qsl_long = mm.query_start_loc_long
        qsl_long_p = qsl_long[: num_prefills + 1]
        gdn_state = (
            num_prefill_tokens,
            num_decode_tokens,
            si_p,
            si_all,
            si_d,
            hi_p,
            hi_all,
            qsl_p,
            qsl_all,
            qsl_long,
            qsl_long_p,
            num_decodes,
            gen_p,
            _verify_state(core, mm, num_decodes) if is_verify else None,
        )

        # `has_initial_states` covers the context prefix only; every generation row carries.
        ple_carry = nn.functional.pad(hi_p, [0, num_decodes], mode="constant", value=1.0)
        # `query_start_loc_long` mis-segments PLE when `gen_p > 1`; rebuild from `seq_lens_cuda`.
        ple_qsl = (
            qsl_long
            if gen_p == 1
            else cu_seqlens_from_seq_lens(md.seq_lens_cuda[:batch], torch.int64)
        )
        pools = core._bound.pools
        ple_state = (pools.ple_state[0], pools.ple_state[1], si_all, ple_carry, ple_qsl)

        # Dense/sparse decision from a host read: a per-capture constant.
        sparse = batch > 0 and int(md.kv_lens_runtime[:batch].max()) > core.dense_threshold
        # `[T, 3]` view of the `[3, 1, T]` positions, as the QSA position plane stores them.
        coords = torch.transpose(torch.reshape(position_ids, [_QSA_POSITION_AXES, -1]), 0, 1)
        qsa = (
            md.qsa_req_idx_per_token[:num_tokens],
            md.qsa_logical_positions[:num_tokens],
            md.qsa_sequence_lengths[:num_tokens],
            md.qsa_visible_blocks[:num_tokens],
            md.qsa_topk_indices[:num_tokens],
            md.qsa_topk_row_starts[:num_tokens],
            md.qsa_block_table,
            md.tokens_per_block,
            coords[:num_tokens],
            sparse,
        )

        # Verify rows write the PLE pools and QSA side caches for every draft row; reopened every
        # forward, `SpeculativeRollback` stages what its commit needs to undo rejected ones.
        rollback = core.spec_rollback
        if rollback is not None:
            rollback.discard()
        if is_verify:
            rollback.begin(
                num_contexts=num_prefills,
                prefix_tokens=num_prefill_tokens,
                num_gens=num_decodes,
                rows_per_gen=gen_p,
                slots=si_d,
                request_indices=md.qsa_req_idx_per_token[num_prefill_tokens:num_tokens],
                logical_positions=md.qsa_logical_positions[num_prefill_tokens:num_tokens],
                block_table=md.qsa_block_table,
                tokens_per_block=md.tokens_per_block,
            )
        fp8_pages = None
        if (
            sparse
            and core.quant_mode == QM_FP8_KV_CACHE
            and num_prefill_tokens >= _ROWTILE_MIN_ROWS
        ):
            fp8_pages = _fp8_context_pages(md, md.tokens_per_block)
        rowtile_groups = None
        if fp8_pages is not None and self._rowtile_groups and num_decodes == 0:
            kv0 = core._bound.pools.kv[core.attn_layers[0]]
            rowtile_groups = _rowtile_scratch_groups(md, kv0.k[0].numel() * 4)
        qsa_rows = None
        if (
            sparse
            and num_decodes == 0
            and not is_verify
            and 0 < num_tokens <= physical_tokens <= _GRAPH_BREAKS_MAX_TOKENS
        ):
            qsa_rows = self._qsa_padded_rows(md, coords, num_tokens, physical_tokens)
        sel_cuts = None
        if sparse and qsa_rows is None and self._split_context_rows(num_prefill_tokens):
            chunks = [int(c) for c in md.seq_lens[:num_prefills].tolist()]
            kv_lens = [int(k) for k in md.kv_lens_runtime[:num_prefills].tolist()]
            if sum(chunks) == num_prefill_tokens:
                sel_cuts = _selection_cuts(
                    chunks,
                    [k - c for k, c in zip(kv_lens, chunks)],
                    self._selection_split[1],
                    _SELECTION_BALANCE_ROW_TOKENS,
                )
        self._step = _PrefillStep(
            num_tokens=num_tokens,
            physical_tokens=physical_tokens,
            dp_pad_rows=dp_pad_rows,
            input_ids=input_ids[:num_tokens],
            attn=attn_args,
            gdn=gdn_state,
            ple=ple_state,
            qsa=qsa,
            vision=vision,
            stage=is_verify,
            fp8_pages=fp8_pages,
            rowtile_groups=rowtile_groups,
            split=self._split_of(mm, num_prefills, num_decodes, hi_p, num_tokens),
            qsa_rows=qsa_rows,
            sel_cuts=sel_cuts,
        )

    def _qsa_padded_rows(self, md, coords, num_tokens: int, physical_tokens: int) -> tuple:
        """QSA per-row metadata over the bucket's `T` rows: rows past `n` copy the last real row,
        so padded rows read only what real rows read (their stores use position -2, which every
        store kernel skips). Persistent buffers: the views keep their addresses across steps."""
        n, rows = num_tokens, physical_tokens
        planes = (
            md.qsa_req_idx_per_token,
            md.qsa_logical_positions,
            md.qsa_sequence_lengths,
            md.qsa_visible_blocks,
        )
        if rows > n:
            for plane in planes:
                plane[n:rows].copy_(plane[n - 1 : n].expand(rows - n))
        # The coordinates are a fresh reshape each step: the graph reads them from its own buffer.
        buf = self._coords_buf
        if buf is None or buf.shape[0] < rows or buf.dtype != coords.dtype:
            buf = torch.empty(
                [max(rows, self.core.max_num_tokens), coords.shape[1]],
                dtype=coords.dtype,
                device=coords.device,
            )
            self._coords_buf = buf
        buf[:rows].copy_(coords[:rows])
        return tuple(plane[:rows] for plane in planes) + (
            md.qsa_topk_indices[:rows],
            md.qsa_topk_row_starts[:rows],
            buf[:rows],
        )

    def _split_of(
        self, mm, num_prefills: int, num_decodes: int, hi_p, num_tokens: int
    ) -> _Split | None:
        """The step's `_Split` from `inforward_split` (planned for context-only batches, each split
        chunk at least `_INFORWARD_MIN_MAIN` tokens before P), or None. No host read."""
        split = getattr(mm, "inforward_split", None)
        if split is None:
            return None
        stage = self.core._bound.stage
        assert stage is not None, (
            "the engine split a step for in-forward snapshots but the KV cache manager stages "
            "no state for them"
        )
        assert num_decodes == 0, "an in-forward snapshot split in a step with generation rows"
        k = split.k
        dev = split.cu.device
        # Main parts carry as their request does; tails start from zeros and drop their state.
        carry = torch.zeros([num_prefills + k], dtype=torch.bool, device=dev)
        carry.index_copy_(0, split.main_rows, hi_p.to(torch.bool))
        # The row of P in the packed batch: one past the last row of the conv window ending at P.
        p_rows = split.conv_win.view(k, -1)[:, -1:] + 1
        ple_len = stage.ple_conv.shape[-1]
        ngram_len = stage.ple_ngram.shape[-1]
        ple_win = p_rows + torch.arange(-ple_len, 0, dtype=p_rows.dtype, device=dev)
        ngram_win = p_rows + torch.arange(-ngram_len, 0, dtype=p_rows.dtype, device=dev)
        whole = None
        v_heads = next(iter(stage.ssm.values())).shape[1] if stage.ssm else 0
        if v_heads and split_takes_cp(num_prefills, v_heads, dev):
            if num_prefills == 1 and k == 1:
                # One request split once: the rows before P are a prefix (no gather).
                rows = num_tokens - split.num_tail_tokens
            else:
                # Every non-tail row, in order (a stable sort on the tail mask; no host read).
                tail_mask = torch.zeros([num_tokens], dtype=torch.int8, device=dev)
                tail_mask.index_fill_(0, split.tail_tok, 1)
                rows = torch.argsort(tail_mask, stable=True)[: num_tokens - split.num_tail_tokens]
            before = split.cu.index_select(0, split.main_rows + 1) - split.cu.index_select(
                0, split.main_rows
            )
            cu = torch.zeros([num_prefills + 1], dtype=torch.long, device=dev)
            torch.cumsum(before, 0, out=cu[1:])
            whole = (rows, cu, hi_p.to(torch.bool))
        return _Split(
            k=k,
            cu=split.cu,
            gather=split.gather.to(torch.int32),
            carry=carry,
            main_rows=split.main_rows,
            live=split.live.to(torch.int32),
            tail_tok=split.tail_tok,
            cu_tail=split.cu_tail,
            stage_rows=split.stage_rows,
            tail_slots=split.tail_slots,
            conv_win=split.conv_win,
            ple_win=ple_win.reshape(-1),
            ngram_win=ngram_win.reshape(-1),
            whole=whole,
            raw=split,
        )

    def _pad_mask(self, physical_tokens: int, device) -> torch.Tensor | None:
        """`[T, 1]` bool, True at a graph bucket's padding rows (past `_live_rows`), which the MoE
        blocks keep out of the expert window. Computed on device, so a replay recomputes it; None
        for an eager step and under attention DP (whose round trip drops padding itself)."""
        if self.core.dp_size != 1:
            return None
        if (
            physical_tokens == self._step.num_tokens
            and not torch.cuda.is_current_stream_capturing()
        ):
            return None
        rows = torch.arange(physical_tokens, dtype=torch.int32, device=device)
        return torch.unsqueeze(rows >= self._live_rows, 1)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor,
        position_ids: torch.IntTensor,
        **kwargs,
    ):
        """Run the trunk over `T` rows: the batch's `n`, or the prefill CUDA graph bucket's.

        A breakable graph captures this body per bucket and replays it without this Python, so
        nothing here reads the step: ops are row-wise over `T`, and step-dependent work runs in
        the breaks (`_splice`, `_ple_block`, `_attention_core`, `_gdn_core`), each writing rows
        `[:n]` of a fixed output allocated here and zeroing rows `[n:]`."""
        core = self.core
        num_tokens = input_ids.shape[0]
        assert num_tokens == self._step.physical_tokens, (
            f"the forward got {num_tokens} activation rows and prepare saw "
            f"{self._step.physical_tokens}"
        )
        pos3 = _copy_positions(position_ids)
        pos = pos3[0]

        # The token embedding (replicated) with this step's vision rows spliced in.
        h = nn.functional.embedding(input_ids, core.w["embed"])
        eager_on_graph(h, inner=self._splice)
        dt = h.dtype
        dev = h.device
        # Sequence parallelism (`_sp_split`): the stream runs on this rank's rows `[lo, hi)`;
        # each block gathers all rows and leaves its sum on this rank's rows.
        sp = self._sp_split(num_tokens)
        if sp is None:
            lo, hi, gather = 0, num_tokens, None
            self._block_sum = core.tp_sum
            self._sp_rows = None
        else:
            lo, hi, gather, self._block_sum = sp
            self._sp_rows = (lo, hi, gather)

        # Scratch shared by every layer that uses it; each break rewrites its whole output.
        proj_qkv = torch.empty([num_tokens, core.qkv_width], dtype=dt, device=dev)
        index_scratch = torch.empty([num_tokens, core.index_scratch_width], dtype=dt, device=dev)
        attn_out = torch.empty([num_tokens, core.q_width], dtype=dt, device=dev)
        gdn_out = torch.empty(
            [1, num_tokens, core.num_v_heads, core.head_v_dim], dtype=dt, device=dev
        )
        ple_hc = torch.empty([num_tokens, core.stream], dtype=dt, device=dev)
        ple_normed = torch.empty([num_tokens, core.stream], dtype=dt, device=dev)

        hc_count = core.hc_count
        hc = torch.cat([h[lo:hi]] * hc_count, -1)
        layers, S = self._layers, self._stack
        pad_rows = self._step.dp_pad_rows
        pad_mask = self._pad_mask(num_tokens, dev)
        # From `_HC_RSTD_MIN_ROWS` rows (fixed per bucket) sites keep rstd and the mixer norms on
        # chip; the PLE site always uses the stored-normed mixer on the PLE kernel's output.
        by_rstd = hc.shape[0] >= _HC_RSTD_MIN_ROWS
        normed = rstd = None
        if by_rstd:
            _, rstd = qwen4_exp_hc_inject_rstd(hc, None, None, core.eps, hc_count)
        else:
            _, normed = qwen4_exp_hc_inject_norm(
                hc, None, None, layers[0].ahc_norm, core.eps, hc_count
            )
        for i, L in enumerate(layers):
            if i == core.ple_layer:
                # The n-gram path convolves along each request's rows: it runs on all rows.
                whole = hc if gather is None else gather(hc)
                eager_on_graph(whole, ple_hc, ple_normed, inner=self._ple_block)
                hc, normed = ple_hc[lo:hi], ple_normed[lo:hi]
                mixed, inject = _hc_mixer(core, normed, L.ahc_first, L.ahc_up, True)
            elif by_rstd:
                mixed, inject = _hc_mixer_rstd(
                    core, hc, rstd, L.ahc_norm, L.ahc_first, L.ahc_up, True
                )
            else:
                mixed, inject = _hc_mixer(core, normed, L.ahc_first, L.ahc_up, True)
            if core.is_attn[i]:
                block_out = self._attention_block(
                    mixed, i, L, proj_qkv, index_scratch, attn_out, pos3, pos
                )
            else:
                block_out = self._gdn_block(mixed, i, L, gdn_out)
            if by_rstd:
                hc, rstd = qwen4_exp_hc_inject_rstd(hc, block_out, inject, core.eps, hc_count)
                mixed, inject = _hc_mixer_rstd(
                    core, hc, rstd, L.mhc_norm, L.mhc_first, L.mhc_up, True
                )
            else:
                hc, normed = qwen4_exp_hc_inject_norm(
                    hc, block_out, inject, L.mhc_norm, core.eps, hc_count
                )
                mixed, inject = _hc_mixer(core, normed, L.mhc_first, L.mhc_up, True)
            next_norm = layers[i + 1].ahc_norm if i + 1 < len(layers) else S.fhc_norm
            # The injection forms routed + sigmoid(gate) * shared, rounded once.
            if gather is None:
                routed, gate_logits, shared = _moe_block(
                    core, mixed, L, pad_rows, pad_mask=pad_mask
                )
            else:
                routed, gate_logits, shared = _moe_block_sp(
                    core,
                    mixed,
                    L,
                    gather,
                    self._block_sum,
                    None if pad_mask is None else pad_mask[lo:hi],
                )
            if by_rstd:
                hc, rstd = qwen4_exp_hc_inject_rstd(
                    hc, routed, inject, core.eps, hc_count, shared, gate_logits
                )
            else:
                hc, normed = qwen4_exp_hc_inject_norm(
                    hc, routed, inject, next_norm, core.eps, hc_count, shared, gate_logits
                )
            del routed, gate_logits, shared
        # The stack-final mixer with no injection weight is the final norm.
        if by_rstd:
            final, _ = _hc_mixer_rstd(core, hc, rstd, S.fhc_norm, S.fhc_first, S.fhc_up, False)
        else:
            final, _ = _hc_mixer(core, normed, S.fhc_first, S.fhc_up, False)
        if not core.mtp_enabled:
            return final if gather is None else gather(final)
        # `lm_head` reads the `[T, hidden]` mix, the MTP module the `[T, stream]` stream; packed
        # because a breakable graph body returns one tensor (`split_spec_output`).
        packed = torch.cat([final, hc], -1)
        return packed if gather is None else gather(packed)

    # ------------------------------------------------------------- the draft ---

    def _capture_selection(self, md, topk_indices, visible_blocks) -> None:
        """IndexShare: store each request's selection at the row the draft loop continues from
        (a context request's last chunk row, a generation request's last accepted row), with its
        complete-group count, in the draft layer's share slot."""
        batch = md.num_seqs
        assert md.qsa_mtp_num_accepted is not None, (
            "IndexShare needs the worker's accepted counts to pick each request's capture row; "
            "the draft loop did not supply them"
        )
        seq_lens = md.seq_lens_cuda[:batch].to(torch.int64)
        starts = torch.cumsum(seq_lens, 0) - seq_lens
        accepted = md.qsa_mtp_num_accepted[:batch].to(torch.int64)
        offsets = torch.minimum(torch.clamp(accepted - 1, min=0), seq_lens - 1)
        offsets[: md.num_contexts] = seq_lens[: md.num_contexts] - 1
        rows = starts + offsets
        slot = self.core._draft_bound.share_slot
        md.qsa_shared_topk_indices[slot, :batch].copy_(torch.index_select(topk_indices, 0, rows))
        md.qsa_shared_topk_visible_blocks[slot, :batch].copy_(
            torch.index_select(visible_blocks, 0, rows)
        )
        # Host-side readiness, recorded in program order.
        md.qsa_shared_topk_captured_slots.add(slot)

    def _draft_extend_query_rows(self, md, draft, qkv, q_index, num_ctx: int, rows: int):
        """Draft-layer sparse attention on a draft-extend step with context rows, over only the
        rows the draft loop continues from (each context request's last row, every generation
        row); the worker discards the rest. Query rows score row by row, since a chunk's last row
        sees every key the chunk stored. Returns `(out, query)`: `[n, q_width]`, int64 rows."""
        core = self.core
        num_contexts = md.num_contexts
        ends = torch.cumsum(md.seq_lens_cuda[:num_contexts].to(torch.int64), 0) - 1
        query = torch.cat(
            [ends, torch.arange(num_ctx, rows, dtype=torch.int64, device=ends.device)]
        )
        n = num_contexts + rows - num_ctx
        logical = md.qsa_logical_positions[:rows].index_select(0, query)
        req_idx = md.qsa_req_idx_per_token[:rows].index_select(0, query)
        scores = qsa_paged_index_scores(
            q=q_index.index_select(0, query),
            index_cache=draft.index_cache,
            block_table=md.qsa_block_table,
            query_positions=logical,
            request_indices=req_idx,
            tokens_per_block=md.tokens_per_block,
            compress_ratio=core.compress_ratio,
            only_visible_blocks=True,
            context_rows=False,
            wide_split=True,
        )
        selected = torch.empty(
            [n, core.token_topk + core.compress_ratio - 1], dtype=torch.int32, device=qkv.device
        )
        _selection(
            core,
            scores,
            md.qsa_visible_blocks[:rows].index_select(0, query),
            logical,
            md.qsa_sequence_lengths[:rows].index_select(0, query),
            md.qsa_topk_indices[:rows].index_select(0, query),
            md.qsa_topk_row_starts[:rows].index_select(0, query),
            selected,
        )
        out = _sparse_gqa(
            core,
            draft.kv,
            _packed_q(core, qkv).index_select(0, query),
            md.qsa_block_table,
            selected,
            req_idx,
            md.tokens_per_block,
            None,
        )
        return torch.reshape(out, [n, core.q_width]), query

    def mtp_forward(
        self,
        embed_tokens: torch.Tensor,
        all_rank_num_tokens,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        """One eager draft step with context rows (the draft loop's first). Sparse attention runs
        only on the rows the worker gathers (`_draft_extend_query_rows`) unless IndexShare
        captures every row's selection."""
        core = self.core
        assert isinstance(attn_metadata, TrtllmAttentionMetadata)
        assert core._bound is not None, "the trunk's cache-manager bind has not run"
        md = attn_metadata
        rows = hidden_states.shape[0]
        assert input_ids.shape[0] == rows and hidden_states.shape[1] == core.stream, (
            f"the draft step got {input_ids.shape[0]} token ids and {tuple(hidden_states.shape)} "
            f"hidden states; the MTP module reads one {core.stream}-wide stream row per token"
        )
        # Per call: the worker rewrites `attn_metadata` in place between draft steps.
        step = self.step_args(md)
        num_ctx = md.num_ctx_tokens
        gen_seqs = md.num_seqs - md.num_contexts
        p, _ = _verify_rows(core, None, gen_seqs, rows - num_ctx)
        step["predicted_tokens_per_seq"] = p
        draft, layer_idx = core._draft_step_bind(md)
        assert rows == md.num_tokens, (
            f"the draft step carries {rows} rows for a batch its metadata describes as "
            f"{md.num_tokens} tokens; the QSA row mapping covers the second"
        )
        pad_rows = _draft_pad_rows(core, all_rank_num_tokens, rows)
        # Same host buffer as the trunk, so every draft step takes the trunk's branch.
        batch = md.num_seqs
        sparse = batch > 0 and int(md.kv_lens_runtime[:batch].max()) > core.dense_threshold
        # IndexShare: the loop's first step runs the indexer and captures what later steps reuse.
        share = core.mtp_index_share and md.qsa_in_mtp_draft_loop
        assert not (share and md.qsa_indexer_skip_topk), (
            "the worker asked a draft step with context rows to reuse a selection"
        )
        pos3, pos, coords = _draft_positions(position_ids)
        M = core._mtp
        req_idx = md.qsa_req_idx_per_token[:rows]
        logical = md.qsa_logical_positions[:rows]
        block_table = md.qsa_block_table
        tokens_per_block = md.tokens_per_block

        x, mixed, inject = _mtp_front(core, M, embed_tokens, input_ids, hidden_states)
        proj, qkv, gate, k_dst, v_dst, index_scratch = _mtp_attention_in(
            core, M, mixed, pos3, pos, True
        )
        # The rows the tail runs on (`_draft_extend_query_rows`), or None for every row.
        tail_rows = None
        q_index = _mtp_index_branch(
            core,
            M,
            proj,
            index_scratch,
            draft,
            (req_idx, logical, block_table, tokens_per_block, coords),
        )
        if sparse and num_ctx and not share:
            kv = draft.kv
            _store_kv(core, kv, k_dst, v_dst, req_idx, logical, block_table, tokens_per_block)
            out, query = self._draft_extend_query_rows(md, draft, qkv, q_index, num_ctx, rows)
            if core.dp_size == 1:
                # Only the query rows are read: the tail runs on them, the rest stay zero.
                attn, tail_rows = out, query
                gate, x, inject = (t.index_select(0, query) for t in (gate, x, inject))
            else:
                attn = torch.zeros([rows, core.q_width], dtype=qkv.dtype, device=qkv.device)
                attn.index_copy_(0, query, out)
        elif sparse:
            kv = draft.kv
            _store_kv(core, kv, k_dst, v_dst, req_idx, logical, block_table, tokens_per_block)
            topk_indices = md.qsa_topk_indices[:rows]
            visible_blocks = md.qsa_visible_blocks[:rows]
            seq_lens_q = md.qsa_sequence_lengths[:rows]
            row_starts = md.qsa_topk_row_starts[:rows]
            parts = [(0, num_ctx, True)] if num_ctx else []
            if num_ctx < rows:
                parts.append((num_ctx, rows, False))
            selected = torch.empty(
                [rows, core.token_topk + core.compress_ratio - 1],
                dtype=torch.int32,
                device=qkv.device,
            )
            for lo, hi, context_rows in parts:
                scores = _context_index_scores(
                    q_index[lo:hi],
                    draft.index_cache,
                    block_table,
                    logical[lo:hi],
                    req_idx[lo:hi],
                    tokens_per_block,
                    core.compress_ratio,
                    context_rows,
                )
                _selection(
                    core,
                    scores,
                    visible_blocks[lo:hi],
                    logical[lo:hi],
                    seq_lens_q[lo:hi],
                    topk_indices[lo:hi],
                    row_starts[lo:hi],
                    selected[lo:hi],
                )
            if share:
                self._capture_selection(md, topk_indices, visible_blocks)
            attn = torch.empty([rows, core.q_width], dtype=qkv.dtype, device=qkv.device)
            for lo, hi, context_rows in parts:
                out = _sparse_gqa(
                    core,
                    kv,
                    _packed_q(core, qkv)[lo:hi],
                    block_table,
                    selected[lo:hi],
                    req_idx[lo:hi],
                    tokens_per_block,
                    logical[lo:hi] if context_rows else None,
                )
                attn[lo:hi].copy_(torch.reshape(out, [hi - lo, core.q_width]))
        else:
            attn = thop_attention(
                q=qkv,
                output=torch.empty([rows, core.q_width], dtype=qkv.dtype, device=qkv.device),
                local_layer_idx=layer_idx,
                num_heads=core.heads_q,
                num_kv_heads=core.heads_kv,
                head_size=core.head_dim,
                quant_mode=core.quant_mode,
                kv_scale_orig_quant=core.kv_scale_orig_quant,
                kv_scale_quant_orig=core.kv_scale_quant_orig,
                **step,
                **_CALL_CONSTANTS,
            )
        fused_sigmoid_mul_inplace(attn, gate, direct_launch=True)
        out = _mtp_tail(core, M, x, attn, inject, pad_rows)
        if tail_rows is None:
            return out
        full = out.new_zeros([rows, out.shape[1]])
        full.index_copy_(0, tail_rows, out)
        return full


# ------------------------------------------------------------------- decode ---


#: Generation-only steps up to this many rows run on a highest-priority stream with the MoE
#: shared expert on a default-priority side stream, so the routed (critical) path gets the SMs
#: first; on larger steps the starved shared expert would gate the join.
_DECODE_HIGH_PRIORITY_MAX_ROWS = 8  # capacity: fixed: a dispatch threshold, not a buffer size


def _on_decode_stream(core, rows, step, *args, **kwargs):
    """Run a generation-only step of `rows` rows on the core's high-priority decode stream,
    ordered after the caller's stream, which then waits for it (graph edges under capture; PDL
    chains stay on one stream). Above the row limit or without the stream, run in line."""
    hi = getattr(core, "_decode_stream", None) if rows <= _DECODE_HIGH_PRIORITY_MAX_ROWS else None
    if hi is None:
        return step(*args, **kwargs)
    caller = torch.cuda.current_stream()
    hi.wait_stream(caller)
    with torch.cuda.stream(hi):
        out = step(*args, **kwargs)
    caller.wait_stream(hi)
    return out


class DecodeTarget:
    """Generation-only batches (no context rows), every op inside the captured decode graph."""

    def __init__(self, core: Qwen4ExpUnifiedModelingV2) -> None:
        self.core = core
        self._contract_pending = step_contract_enabled()
        self._layers = _layer_views(core)
        self._stack = _stack_views(core)
        self._index_split = _index_split_group(core)

    def _check_step_contract(self, md) -> None:
        if not self._contract_pending:
            return
        try:
            self.step_args(md)
        except AttributeError as exc:
            raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
        self._contract_pending = False

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        """See PrefillTarget.step_args. Context counts are stated, not read: routing
        guarantees them, and a decode-only CUDA graph would freeze whatever it read."""
        return _step_args(
            md,
            num_contexts=0,
            num_ctx_tokens=0,
            paged_context=self.core.quant_mode == QM_NVFP4_KV_CACHE,
        )

    def _verify_qsl(self, seq_lens, batch: int, p: int, device) -> torch.Tensor:
        """PLE segmentation `[0, p, ..., batch * p]` of a generation-only verify step, cached
        when built outside capture (warmup) so a captured step launches nothing for it."""
        key = (batch, p, device)
        cache = self.__dict__.setdefault("_verify_qsl_cache", {})
        hit = cache.get(key)
        if hit is not None:
            return hit
        if torch.cuda.is_current_stream_capturing():
            return cu_seqlens_from_seq_lens(seq_lens, torch.int64)
        qsl = torch.arange(batch + 1, dtype=torch.int64, device=device) * p
        cache[key] = qsl
        return qsl

    # --------------------------------------------------------------- blocks ---

    def _attention_block(self, x, i, L, qkv, index_scratch, attn_out, qsa, step, stage):
        """One gated GQA layer plus its QSA index branch for a generation-only batch. The index
        plane is kept on every step; above the dense threshold the sparse core replaces the dense
        op, and over a bf16 or e4m3 pool all work up to the index scores is one launch. Verify
        rows (`stage`) have their index slots staged for `SpeculativeRollback` before the store."""
        core = self.core
        (
            req_idx,
            logical,
            seq_lens_q,
            visible_blocks,
            topk_indices,
            topk_row_starts,
            block_table,
            tokens_per_block,
            pos3,
            pos,
            coords,
            sparse,
        ) = qsa
        num_tokens = x.shape[0]
        proj = _decode_projection(x, L.attn_in)
        qkv_view, gate, _ = torch.split(
            proj, [core.qkv_width, core.q_width, core.index_scratch_width], 1
        )
        pools = core._bound.pools
        prologue = _decode_prologue(pools.kv[i], sparse)
        if stage:
            core.spec_rollback.stage_index(pools.position_cache)
        plan = getattr(self, "_index_plan", None) if sparse else None
        if prologue:
            _decode_prologue_call(
                core,
                proj,
                qkv,
                index_scratch,
                i,
                L,
                (req_idx, logical, block_table, tokens_per_block, pos3, coords),
                step["predicted_tokens_per_seq"],
                None if plan is None else plan.iq_rows,
            )
        else:
            _, k_dst, v_dst = _attn_in_heads(core, proj)
            _attention_rope(core, proj, index_scratch, L, pos3, pos)
            _index_plane(
                core, proj, i, L, (req_idx, logical, block_table, tokens_per_block, coords)
            )
            # One compact `[q | k | v]` copy: read flat by the dense op, per head by sparse GQA.
            qkv.copy_(qkv_view)

        if sparse:
            kv = pools.kv[i]
            if not prologue:
                _store_kv(core, kv, k_dst, v_dst, req_idx, logical, block_table, tokens_per_block)
            selected = _decode_index_selection(
                core,
                getattr(self, "_index_split", None),
                _index_query(core, index_scratch),
                qsa[:6] + (block_table, tokens_per_block),
                num_tokens,
                step["predicted_tokens_per_seq"],
                proj.device,
                core._bound.pools.index_cache[i],
                plan,
                q_permuted=prologue,
                order=getattr(self, "_row_order", None),
            )
            attn = _sparse_gqa(
                core,
                kv,
                _packed_q(core, qkv),
                block_table,
                selected,
                req_idx,
                tokens_per_block,
                None,
                output_gate=gate,
                rows_per_request=step["predicted_tokens_per_seq"],
            )
            out = torch.reshape(attn, [num_tokens, core.q_width])
        else:
            out = thop_attention(
                q=qkv,
                output=attn_out,
                local_layer_idx=core._bound.pools.pool_idx[i],
                num_heads=core.heads_q,
                num_kv_heads=core.heads_kv,
                head_size=core.head_dim,
                quant_mode=core.quant_mode,
                kv_scale_orig_quant=core.kv_scale_orig_quant,
                kv_scale_quant_orig=core.kv_scale_quant_orig,
                **step,
                **_CALL_CONSTANTS,
            )
            fused_sigmoid_mul_inplace(out, gate, direct_launch=True)
        return _tp_partial(core, _decode_projection(out, L.o))

    def _gdn_block(self, x, i, L, si_all, verify, gen_p) -> torch.Tensor:
        """One Gated-DeltaNet layer over a generation-only batch: one GEMM for the four input
        projections, the conv step in place, then recurrence and gated norm in one launch per
        (row, v-head), so over a bf16 pool a row's output does not depend on its batch. A verify
        step (`verify`, `gen_p` rows a sequence) takes `_gdn_verify` instead."""
        core = self.core
        pools = core._bound.pools
        tactic = (
            _gdn_in_proj_conv_tactic(core, x, L, si_all, verify) if verify is not None else None
        )
        mq_out = None
        if tactic is not None:
            # The projection and the conv of its q|k|v columns in one launch.
            full, mq_out = gdn_in_proj_conv(
                x,
                L.in_proj_dec,
                L.conv_w,
                pools.conv[i],
                si_all,
                core._bound.int_conv[i],
                verify[0],
                _PAD_SLOT_ID,
                gen_p,
                tactic,
                _DECODE_PROJ_PDL,
            )
            proj = full[:, : sum(core.in_proj_split)]
        else:
            # Its alignment columns, past `sum(in_proj_split)`, are dropped by the view.
            proj = _decode_projection(x, L.in_proj_dec)[:, : sum(core.in_proj_split)]
        if verify is not None:
            mixed_qkv, z, a, b = torch.split(proj, core.in_proj_split, 1)
            num_tokens = x.shape[0]
            fold_norm = _gdn_verify_folds_norm(core, num_tokens // gen_p, gen_p)
            out = _gdn_verify(
                core,
                mixed_qkv,
                a,
                b,
                i,
                L,
                si_all,
                num_tokens // gen_p,
                gen_p,
                verify,
                mq_out=mq_out,
                z=z if fold_norm else None,
            )
            if fold_norm:
                # The replay returned the normed, gated rows.
                o = torch.reshape(out, [num_tokens, core.value_dim])
                return _tp_partial(core, _decode_projection(o, L.out))
            return self._gdn_epilogue(out, z, L, num_tokens)
        mixed_qkv = torch.split(proj, core.in_proj_split, 1)[0]
        causal_conv1d_update(
            x=mixed_qkv,
            conv_state=pools.conv[i],
            weight=L.conv_w,
            bias=None,
            silu_activation=True,
            cache_seqlens=None,
            conv_state_indices=si_all,
            pad_slot_id=_PAD_SLOT_ID,
        )
        # Computes both gates and the q/k L2 norm itself (softplus (1, 20), eps 1e-6).
        o = gdn_decode_rows_gated_norm(
            proj,
            pools.ssm[i],
            si_all,
            L.a_log,
            L.dt_bias,
            L.gnorm,
            core.eps,
            core.num_k_heads,
            core.head_k_dim,
        )
        return _tp_partial(core, _decode_projection(o, L.out))

    def _gdn_epilogue(self, out, z, L, num_tokens) -> torch.Tensor:
        """A verify step's gated RMS norm (sigmoid gate, plain weight) and output projection."""
        core = self.core
        o = rms_norm_gated_token_major(
            torch.reshape(out, [-1, core.head_v_dim]),
            torch.reshape(z, [num_tokens, core.num_v_heads, core.head_v_dim]),
            weight=L.gnorm,
            eps=core.eps,
            fp8_scale=None,
            gate_activation="sigmoid",
            direct_launch=True,
        )
        o = torch.reshape(o, [num_tokens, core.value_dim])
        return _tp_partial(core, _decode_projection(o, L.out))

    def _ple_front(self, input_ids, ple_state, stage) -> tuple:
        """The PLE part that reads no residual: verify staging of both pools (before the n-gram
        history is written), then the n-gram embedding's `(key, value)` projections."""
        core = self.core
        S = self._stack
        conv_states, ngram_pool, si_all, qsl, rows = ple_state
        if stage:
            core.spec_rollback.stage_ple_verify(ngram_pool, conv_states, input_ids)
        ngram = ple_ngram_embed(
            input_ids,
            ngram_pool,
            si_all,
            None,
            qsl,
            core.w["ple_table"],
            S.ple_tscale,
            S.ple_mult,
            S.ple_vocab,
            S.ple_offs,
            core.eos_token_id,
            rows,
            row_window=core._ple_table_host.window,
        )
        if core._ple_table_host.needs_sum:
            ngram = core.tp_sum(ngram)
        return cublas_mm(ngram, S.ple_key), cublas_mm(ngram, S.ple_val)

    def _ple_block(
        self, hc, input_ids, ple_state, next_norm, stage, front=None, inject_block=None
    ) -> tuple:
        """The n-gram side path added into the stream and normed for the attention-side site;
        returns `(hc, normed)`. No carry flags: every generation sequence carries. A verify step
        (`stage`) stages both pools' pre-step rows and the conv input the second launch writes."""
        core = self.core
        S = self._stack
        conv_states, ngram_pool, si_all, qsl, rows = ple_state
        key, val = front if front is not None else self._ple_front(input_ids, ple_state, stage)
        conv_input = torch.empty_like(hc) if stage else None
        from_pool = self._ple_from_pool(stage)
        out = ple_gate_conv_inject_norm(
            hc,
            key,
            val,
            S.ple_nk_w,
            S.ple_nq_w,
            S.ple_nc_w,
            S.ple_conv_w,
            conv_states,
            si_all,
            qsl,
            next_norm,
            core.eps,
            core.hc_count,
            _PLE_CLAMP_MIN,
            core.ple_dilation,
            _PAD_SLOT_ID,
            rows,
            conv_input=conv_input,
            inject_block=inject_block,
            # The three launches chain through PDL (off with TRTLLM_ENABLE_PDL=0); the norm
            # weights are read before the wait.
            pdl=get_env_enable_pdl(),
            # every sequence here is a generation request the commit rewrites
            publish_end=0 if from_pool else None,
        )
        if stage:
            core.spec_rollback.stage_ple_conv_input(conv_input)
        return out

    def _ple_from_pool(self, stage) -> bool:
        """Whether this verify step's PLE launch leaves the conv pool rows unpublished, for the
        fused rollback commit to rebuild from the pool and the staged conv input."""
        return bool(stage) and self.core.spec_rollback.fused_pending()

    # -------------------------------------------------------------- forward ---

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        **kwargs,
    ):
        core = self.core
        md = attn_metadata
        self._check_step_contract(md)
        step = self.step_args(md)
        spec_metadata = kwargs.get("spec_metadata")
        mm = md.mamba_metadata
        pos3 = take_step_positions(md, position_ids)
        if pos3 is None:
            pos3 = _copy_positions(position_ids)
        pos = pos3[0]
        coords = torch.transpose(torch.reshape(position_ids, [_QSA_POSITION_AXES, -1]), 0, 1)

        if input_ids.dim() == 1 and input_ids.is_contiguous():
            # The embedding and its replication over the streams in one launch.
            hc = embed_rows(input_ids, core.w["embed"], core.hc_count, False)
        else:
            hc = torch.cat(
                [nn.functional.embedding(input_ids, core.w["embed"])] * core.hc_count, -1
            )
        num_tokens = hc.shape[0]
        dt = hc.dtype
        dev = hc.device
        batch = md.num_seqs
        # Decode CUDA graphs pad with real dummy requests, so they are in the metadata.
        assert num_tokens == md.num_tokens, (
            f"the engine handed {num_tokens} activation rows for a generation batch its "
            f"metadata describes as {md.num_tokens} tokens"
        )
        # None under a decode CUDA graph, which pads every attention-DP rank to one batch.
        pad_rows = core._dp_pad_rows(md, num_tokens)
        gen_p, is_verify = _verify_rows(core, spec_metadata, batch, num_tokens)
        step["predicted_tokens_per_seq"] = gen_p
        si_all = mm.state_indices[:batch]
        verify = _verify_state(core, mm, batch) if is_verify else None
        pools = core._bound.pools
        # As in PrefillTarget.prepare: rebuild the PLE segmentation when `gen_p > 1`.
        ple_qsl = (
            mm.query_start_loc_long
            if gen_p == 1
            else self._verify_qsl(md.seq_lens_cuda[:batch], batch, gen_p, dev)
        )
        ple_state = (pools.ple_state[0], pools.ple_state[1], si_all, ple_qsl, gen_p)

        # Rollback staging as in PrefillTarget.prepare (`fused` under a decode CUDA graph).
        rollback = core.spec_rollback
        if rollback is not None:
            rollback.discard()
        if is_verify:
            rollback.begin(
                num_contexts=0,
                prefix_tokens=0,
                num_gens=batch,
                rows_per_gen=gen_p,
                slots=si_all,
                request_indices=md.qsa_req_idx_per_token[:num_tokens],
                logical_positions=md.qsa_logical_positions[:num_tokens],
                block_table=md.qsa_block_table,
                tokens_per_block=md.tokens_per_block,
                fused=md.is_cuda_graph,
                position_cache=pools.position_cache,
            )

        # Host-read dense/sparse decision: per capture, as the short/long decode graph families.
        sparse = batch > 0 and int(md.kv_lens_runtime[:batch].max()) > core.dense_threshold
        qsa = (
            md.qsa_req_idx_per_token[:num_tokens],
            md.qsa_logical_positions[:num_tokens],
            md.qsa_sequence_lengths[:num_tokens],
            md.qsa_visible_blocks[:num_tokens],
            md.qsa_topk_indices[:num_tokens],
            md.qsa_topk_row_starts[:num_tokens],
            md.qsa_block_table,
            md.tokens_per_block,
            pos3,
            pos,
            coords,
            sparse,
        )
        split = getattr(self, "_index_split", None)
        self._index_plan = None
        if sparse and split is not None:
            self._index_plan = _index_split_plan(split, qsa, num_tokens, gen_p)
        self._row_order = _RowOrder() if sparse else None

        # Scratch shared by all attention layers; each layer rewrites what it reads.
        qkv = torch.empty([num_tokens, core.qkv_width], dtype=dt, device=dev)
        index_scratch = torch.empty([num_tokens, core.index_scratch_width], dtype=dt, device=dev)
        attn_out = torch.empty([num_tokens, core.q_width], dtype=dt, device=dev)

        hc_count = core.hc_count
        layers, S = self._layers, self._stack
        # The PLE front reads no residual, so it runs on the PLE stream beside the first layers;
        # a row-sharded table's sum is a TP collective, which stays on the main stream.
        ple_stream = getattr(core, "_ple_stream", None)
        ple_front = None
        if ple_stream is not None and core.ple_layer > 0 and not core._ple_table_host.needs_sum:
            caller = torch.cuda.current_stream()
            ple_stream.wait_stream(caller)
            with torch.cuda.stream(ple_stream):
                ple_front = self._ple_front(input_ids, ple_state, is_verify)
        # Each site injects the previous MoE block's `(routed, inject, shared, gate_logits)`.
        moe = (None, None, None, None)
        for i, L in enumerate(layers):
            if i == core.ple_layer:
                # The PLE path adds into the injected stream and normalizes it itself.
                routed, inject, shared, gate_logits = moe
                if isinstance(routed, _TpPartial):
                    routed = core.tp_sum(routed.partial)
                # Fold the previous MoE injection into the PLE launches (bitwise equal to
                # `qwen4_exp_hc_inject_norm` followed by the PLE op).
                fold = None
                if routed is not None and _ple_fold_fits(hc, routed, inject, shared):
                    fold = (routed, inject, shared, gate_logits)
                elif routed is not None or inject is not None:
                    hc, _ = qwen4_exp_hc_inject_norm(
                        hc, routed, inject, L.ahc_norm, core.eps, hc_count, shared, gate_logits
                    )
                if ple_front is not None:
                    torch.cuda.current_stream().wait_stream(ple_stream)
                hc, normed = self._ple_block(
                    hc,
                    input_ids,
                    ple_state,
                    L.ahc_norm,
                    is_verify,
                    front=ple_front,
                    inject_block=fold,
                )
                ple_front = None
                mixed, inject = _hc_mixer(core, normed, L.ahc_first, L.ahc_up, True)
            else:
                hc, mixed, inject = _hc_site(
                    core,
                    hc,
                    moe[0],
                    moe[1],
                    L.ahc_norm,
                    L.ahc_first,
                    L.ahc_up,
                    True,
                    *moe[2:],
                )
            moe = None
            if core.is_attn[i]:
                block_out = self._attention_block(
                    mixed, i, L, qkv, index_scratch, attn_out, qsa, step, is_verify
                )
            else:
                block_out = self._gdn_block(mixed, i, L, si_all, verify, gen_p)
            hc, mixed, inject = _hc_site(
                core, hc, block_out, inject, L.mhc_norm, L.mhc_first, L.mhc_up, True
            )
            routed, gate_logits, shared = _moe_block(
                core, mixed, L, pad_rows, mm=_decode_projection, defer=True
            )
            moe = (routed, inject, shared, gate_logits)
            del routed, gate_logits, shared
        hc, final, _ = _hc_site(
            core, hc, moe[0], moe[1], S.fhc_norm, S.fhc_first, S.fhc_up, False, *moe[2:]
        )
        del moe
        if spec_metadata is None:
            return final
        # `lm_head` reads the `[T, hidden]` mix, the MTP module the `[T, stream]` stream.
        return final, hc

    # ------------------------------------------------------------- the draft ---

    def _capture_selection(self, md, topk_indices, visible_blocks, batch, p):
        """IndexShare: store each request's selection at its last accepted row (row
        `num_accepted - 1` of its `p`) in the draft layer's share slot. Runs on the capture
        stream and returns it (None if in line); the caller joins it before `topk_indices` /
        `visible_blocks` are rewritten or the shared rows are read."""
        assert md.qsa_mtp_num_accepted is not None, (
            "IndexShare needs the worker's accepted counts to pick each request's capture row; "
            "the draft loop did not supply them"
        )
        stream = getattr(self.core, "_capture_stream", None)
        if stream is not None:
            stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
            slot = self.core._draft_bound.share_slot
            accepted = md.qsa_mtp_num_accepted[:batch].to(torch.int64)
            rows = torch.arange(batch, dtype=torch.int64, device=topk_indices.device) * p
            rows = rows + torch.clamp(accepted - 1, min=0, max=p - 1)
            md.qsa_shared_topk_indices[slot, :batch].copy_(
                torch.index_select(topk_indices, 0, rows)
            )
            md.qsa_shared_topk_visible_blocks[slot, :batch].copy_(
                torch.index_select(visible_blocks, 0, rows)
            )
        # Host-side readiness, recorded in program order: under capture this runs while the
        # graph is recorded, so a graph holds the reuse only after the capture it reads.
        md.qsa_shared_topk_captured_slots.add(slot)
        return stream

    def mtp_forward(
        self,
        embed_tokens: torch.Tensor,
        all_rank_num_tokens,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        """One draft step over generation rows only (`p` rows a request on the first step of a
        batch without context, one otherwise), inside the decode graph when the trunk is.

        Under IndexShare the draft-extend step captures each request's selection and later steps
        skip the index branch, rebuilding it with `qsa_mtp_reuse_topk`; the index keys they skip
        lie past the committed length and the next draft-extend step rewrites them."""
        core = self.core
        assert isinstance(attn_metadata, TrtllmAttentionMetadata)
        assert core._bound is not None, "the trunk's cache-manager bind has not run"
        md = attn_metadata
        rows = hidden_states.shape[0]
        assert input_ids.shape[0] == rows and hidden_states.shape[1] == core.stream, (
            f"the draft step got {input_ids.shape[0]} token ids and {tuple(hidden_states.shape)} "
            f"hidden states; the MTP module reads one {core.stream}-wide stream row per token"
        )
        step = self.step_args(md)
        gen_seqs = md.num_seqs
        p, _ = _verify_rows(core, None, gen_seqs, rows)
        step["predicted_tokens_per_seq"] = p
        draft, layer_idx = core._draft_step_bind(md)
        assert rows == md.num_tokens, (
            f"the draft step carries {rows} rows for a batch its metadata describes as "
            f"{md.num_tokens} tokens; the QSA row mapping covers the second"
        )
        pad_rows = _draft_pad_rows(core, all_rank_num_tokens, rows)
        # The worker advances only device KV lengths, so this matches the trunk's host decision.
        sparse = gen_seqs > 0 and int(md.kv_lens_runtime[:gen_seqs].max()) > core.dense_threshold
        share = core.mtp_index_share and md.qsa_in_mtp_draft_loop
        reuse = share and md.qsa_indexer_skip_topk
        assert not share or reuse == (p == 1), (
            f"the worker's reuse flag ({md.qsa_indexer_skip_topk}) disagrees with the step's "
            f"shape ({p} rows per request): only the draft decode steps after the draft-extend "
            "step reuse its selection"
        )
        pos3, pos, coords = _draft_positions(position_ids)
        M = core._mtp
        req_idx = md.qsa_req_idx_per_token[:rows]
        logical = md.qsa_logical_positions[:rows]
        block_table = md.qsa_block_table
        tokens_per_block = md.tokens_per_block

        x, mixed, inject = _mtp_front(
            core, M, embed_tokens, input_ids, hidden_states, mm=_decode_projection
        )
        qkv, gate, q_index = _mtp_attention_planes(
            core,
            M,
            mixed,
            draft,
            (req_idx, logical, block_table, tokens_per_block, pos3, pos, coords),
            p,
            not reuse,
            sparse,
        )
        # Rows whose outputs are read: each request's continuing row on draft-extend, else all.
        cont, pre = None, {}
        if _draft_continue_ok(core, md, p, share):
            meta_rows = (
                md.qsa_visible_blocks,
                md.qsa_sequence_lengths,
                md.qsa_logical_positions,
                md.qsa_req_idx_per_token,
            )
            if all(t.is_contiguous() for t in (md.qsa_mtp_num_accepted, *meta_rows)):
                # The rows, the metadata and the step's operands at them in two launches.
                cont, meta = draft_continue_select(md.qsa_mtp_num_accepted, gen_seqs, p, meta_rows)
                pre.update(zip(("visible", "seq_lens", "logical", "req_idx"), meta))
                named = {"x": x, "inject": inject}
                if sparse:
                    named.update(qkv=qkv, gate=gate)
                    if q_index is not None:
                        named["q_index"] = q_index.reshape(rows, -1)
                named = {k: t for k, t in named.items() if t.dim() == 2 and t.stride(1) == 1}
                pre.update(zip(named, draft_rows_gather(cont, tuple(named.values()))))
                if "q_index" in pre:
                    pre["q_index"] = pre["q_index"].view(gen_seqs, *q_index.shape[1:])
            else:
                # The fused select needs contiguous vectors; else `sel` gathers each operand.
                cont = _draft_continue_rows(core, md, gen_seqs, p, share)
        n = rows if cont is None else gen_seqs

        def sel(t, name=None):
            if cont is None:
                return t
            got = pre.get(name)
            return got if got is not None else t.index_select(0, cont)

        capture = None  # the stream the IndexShare capture runs on, joined below
        if sparse:
            kv = draft.kv
            topk_indices = md.qsa_topk_indices[:n]
            visible_blocks = sel(md.qsa_visible_blocks[:rows], "visible")
            seq_lens_q = sel(md.qsa_sequence_lengths[:rows], "seq_lens")
            q_logical, q_req_idx = sel(logical, "logical"), sel(req_idx, "req_idx")
            selected = torch.empty(
                [n, core.token_topk + core.compress_ratio - 1],
                dtype=torch.int32,
                device=qkv.device,
            )
            if reuse:
                slot = draft.share_slot
                assert slot in md.qsa_shared_topk_captured_slots, (
                    "a draft decode step would reuse a selection that no draft-extend step of "
                    "this loop captured"
                )
                qsa_mtp_reuse_topk(
                    shared_indices=md.qsa_shared_topk_indices[slot],
                    shared_visible_blocks=md.qsa_shared_topk_visible_blocks[slot],
                    request_indices=q_req_idx,
                    visible_blocks=visible_blocks,
                    output=topk_indices,
                )
                # Use the expansion's output as the selection when it matches (no copy).
                expanded = qsa_expand_block_indices(
                    topk_indices,
                    q_logical,
                    seq_lens_q,
                    compress_ratio=core.compress_ratio,
                    token_topk=core.token_topk,
                )
                if expanded.shape == selected.shape and expanded.dtype == selected.dtype:
                    selected = expanded
                else:
                    selected.copy_(expanded)
            else:
                # IndexShare's capture keeps every row's block top-k: no split there.
                selected = _decode_index_selection(
                    core,
                    None if share else getattr(self, "_index_split", None),
                    sel(q_index, "q_index"),
                    (q_req_idx, q_logical, seq_lens_q, visible_blocks, topk_indices,
                     md.qsa_topk_row_starts[:n], block_table, tokens_per_block),
                    n,
                    n // gen_seqs,
                    qkv.device,
                    draft.index_cache,
                )  # fmt: skip
                if share:
                    capture = self._capture_selection(md, topk_indices, visible_blocks, gen_seqs, p)
            attn = torch.reshape(
                _sparse_gqa(
                    core,
                    kv,
                    _packed_q(core, sel(qkv, "qkv")),
                    block_table,
                    selected,
                    q_req_idx,
                    tokens_per_block,
                    None,
                    output_gate=sel(gate, "gate"),
                    rows_per_request=n // gen_seqs if n % gen_seqs == 0 else 1,
                ),
                [n, core.q_width],
            )
        else:
            attn = thop_attention(
                q=qkv,
                output=torch.empty([rows, core.q_width], dtype=qkv.dtype, device=qkv.device),
                local_layer_idx=layer_idx,
                num_heads=core.heads_q,
                num_kv_heads=core.heads_kv,
                head_size=core.head_dim,
                quant_mode=core.quant_mode,
                kv_scale_orig_quant=core.kv_scale_orig_quant,
                kv_scale_quant_orig=core.kv_scale_quant_orig,
                **step,
                **_CALL_CONSTANTS,
            )
            fused_sigmoid_mul_inplace(attn, gate, direct_launch=True)
            attn = sel(attn)
        out = _mtp_tail(
            core, M, sel(x, "x"), attn, sel(inject, "inject"), pad_rows, mm=_decode_projection
        )
        if capture is not None:
            torch.cuda.current_stream().wait_stream(capture)
        if cont is None:
            return out
        # The worker reads only the continuing rows; the rest are left unwritten.
        full = out.new_empty([rows, out.shape[1]])
        full.index_copy_(0, cont, out)
        return full


# ------------------------------------------------------------ the MTP module ---


class MTPLayer:
    """The checkpoint's MTP module (one decoder layer behind `fc_embedding` / `fc_hidden`), run
    once per draft step and routed to a target by each call's metadata phase. Returns the
    residual un-collapsed (`shared_head` owns the mixer). Not an `nn.Module`: its weights live
    in the core's `ParameterDict`, and a second registration would break load coverage."""

    def __init__(self, core: Qwen4ExpUnifiedModelingV2, logits_processor) -> None:
        self.core = core
        self.logits_processor = logits_processor
        # The spec worker when it samples with the TP key exchange (`_sharded_head`).
        self.worker = None
        # The engine step's `spec_metadata`, set by the shell before the worker runs.
        self.spec_metadata = None

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)

    def _sharded_head(self, lm_head, rows: int) -> bool:
        """Whether the draft logits may stay this rank's vocabulary shard (greedy tokens via the
        key exchange); not under rejection sampling, which needs the gathered fp32 logits."""
        worker, md = self.worker, self.spec_metadata
        if worker is None or md is None:
            return False
        exchange = worker.key_exchange
        return bool(
            exchange is not None
            and not md.wants_advanced_draft_sampling
            and worker.guided_decoder is None
            and 1 <= rows <= exchange.MAX_ROWS
            and lm_head.tp_mode == TensorParallelMode.COLUMN
            and lm_head.gather_output
            and lm_head.padding_size == 0
        )

    def shared_head(self, hidden_states, lm_head, attn_metadata, return_context_logits):
        """The module's own stack-final mixer, then `lm_head`; with the key exchange
        (`_sharded_head`) this rank's bf16 vocabulary shard."""
        core = self.core
        M = core._mtp
        _, final, _ = _hc_site(
            core, hidden_states, None, None, M.fhc_norm, M.fhc_first, M.fhc_up, False
        )
        if self._sharded_head(lm_head, final.shape[0]):
            return lm_head(final, gather_output=False)
        return self.logits_processor.forward(final, lm_head, attn_metadata, return_context_logits)

    def forward(
        self,
        embed_tokens: torch.Tensor,
        all_rank_num_tokens,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        target = self.core._select_target(attn_metadata)
        args = (embed_tokens, all_rank_num_tokens, input_ids, position_ids, hidden_states)
        if isinstance(target, DecodeTarget):
            return _on_decode_stream(
                self.core, input_ids.shape[0], target.mtp_forward, *args, attn_metadata, **kwargs
            )
        return target.mtp_forward(*args, attn_metadata, **kwargs)


class DraftModel:
    """The spec worker's handle on the drafter: `mtp_layers`, `embed_tokens` and `lm_head`.
    `embed_tokens` reads through the core, since materialization replaces the meta tensors."""

    def __init__(self, core: Qwen4ExpUnifiedModelingV2, lm_head, logits_processor) -> None:
        self.core = core
        self.mtp_layers = [MTPLayer(core, logits_processor)]
        self.lm_head = lm_head

    @property
    def embed_tokens(self) -> torch.Tensor:
        return self.core.w["embed"]


def _spec_rollback_begin_ok(
    request_indices, logical_positions, block_table, position_cache
) -> bool:
    """Whether `spec_rollback_begin` accepts these operands (mirrors its `is_valid`)."""
    return (
        logical_positions.dtype == torch.int64
        and request_indices.dtype in (torch.int32, torch.int64)
        and block_table.dtype == torch.int32
        and block_table.dim() == 2
        and block_table.stride(1) == 1
        and position_cache.dtype == torch.int32
        and position_cache.dim() == 3
        and position_cache.shape[2] == 3
        and position_cache.stride(2) == 1
    )


class SpeculativeRollback:
    """Undoes what a verify step's rejected draft rows wrote to side caches the runtime does not
    promote (it promotes the GDN state): the PLE n-gram history and conv state, each attention
    layer's QSA index-K plane and the shared QSA position plane.

    For a request that accepted `a` rows (golden token included, `1 <= a <= p`) the commit sets
    the n-gram row to row `a - 1`'s id window minus its oldest id, the conv row to columns
    `[a, a + state_len)` of `[pre-step state | the p rows' conv input]`, and every index-K and
    position slot written by an ordinal `>= a` back to its pre-verify value.

    The targets stage during the forward and the spec worker calls `commit_speculative_states`
    after its draft loop in the same forward, so a captured graph holds both ends. A `fused` step
    (decode graphs) stages and commits the PLE pools in one launch each."""

    def __init__(self) -> None:
        self._pending: dict | None = None
        # (base addresses, page strides, plane count) of the index-K planes, once bound.
        self._planes: tuple | None = None

    def set_index_planes(self, planes: list) -> None:
        """Bind the attention layers' index-K planes (static for the manager's lifetime)."""
        ptrs, strides = _plane_table(planes)
        self._planes = (ptrs, strides, len(planes))

    # ------------------------------------------------------------- staging ---

    def discard(self) -> None:
        self._pending = None

    def begin(
        self,
        *,
        num_contexts: int,
        prefix_tokens: int,
        num_gens: int,
        rows_per_gen: int,
        slots: torch.Tensor,
        request_indices: torch.Tensor,
        logical_positions: torch.Tensor,
        block_table: torch.Tensor,
        tokens_per_block: int,
        fused: bool = False,
        position_cache: torch.Tensor | None = None,
    ) -> None:
        """Open a verify step's staging, before the trunk writes any index plane. Generation rows
        follow the `prefix_tokens` context rows as `num_gens` runs of `rows_per_gen` (row
        `s * p + t` is ordinal `t` of request `s`); each row's QSA slot is resolved as
        `qsa_paged_index_store` does and every index plane's slots are gathered. A `fused` step
        given `position_cache` also gathers the position plane, in one launch."""
        n, p = num_gens, rows_per_gen
        assert n > 0 and p > 1, (
            f"a verify step stages generation requests carrying drafts; got {n} requests of "
            f"{p} rows"
        )
        assert logical_positions.shape[0] == n * p, (
            f"{n} generation requests of {p} rows are {n * p} rows; got "
            f"{logical_positions.shape[0]} positions"
        )
        assert self._planes is not None, "the attention layers' index planes are not bound"
        ptrs, strides, count = self._planes
        if (
            fused
            and position_cache is not None
            and _spec_rollback_begin_ok(
                request_indices, logical_positions, block_table, position_cache
            )
        ):
            pages, within, index_all, position_old = spec_rollback_begin(
                request_indices,
                logical_positions,
                block_table,
                tokens_per_block,
                ptrs,
                strides,
                position_cache,
            )
            self._pending = dict(
                num_contexts=num_contexts,
                prefix=prefix_tokens,
                n=n,
                p=p,
                slots=slots.to(torch.long),
                pages=pages,
                within=within,
                staged_layers=0,
                position=(position_cache, position_old),
                index_all=index_all,
                fused=fused,
                ngram_stream=None,
            )
            return
        positions = logical_positions.to(torch.long)
        pages = block_table[request_indices.to(torch.long), positions // tokens_per_block]
        pages = pages.to(torch.long)
        within = positions % tokens_per_block
        self._pending = dict(
            num_contexts=num_contexts,
            prefix=prefix_tokens,
            n=n,
            p=p,
            slots=slots.to(torch.long),
            pages=pages,
            within=within,
            staged_layers=0,
            position=None,
            index_all=qsa_rollback_gather(ptrs, strides, pages, within, count),
            fused=fused,
            ngram_stream=None,
        )

    def fused_pending(self) -> bool:
        return self._pending is not None and bool(self._pending["fused"])

    def stage_ple_verify(
        self, ngram_pool: torch.Tensor, conv_pool: torch.Tensor, input_ids: torch.Tensor
    ) -> None:
        """Stage the PLE pools before either is written: the slots' pre-step conv rows (not on a
        fused step, `DecodeTarget._ple_from_pool`) and the n-gram candidates, row `t`'s being the
        last `context` ids of `[history | ids of rows 0..t]`."""
        pending = self._pending
        prefix, n, p = pending["prefix"], pending["n"], pending["p"]
        slots = pending["slots"]
        pending["ngram_pool"] = ngram_pool
        pending["conv_pool"] = conv_pool
        if pending["fused"]:
            pending["ngram_stream"], pending["conv_prev"] = spec_rollback_stage(
                ngram_pool,
                conv_pool,
                slots,
                input_ids[prefix : prefix + n * p],
                p,
                conv=False,
            )
            return
        pending["conv_prev"] = conv_pool[slots]
        context = ngram_pool.shape[1]
        ids = torch.reshape(input_ids[prefix : prefix + n * p], [n, p]).to(ngram_pool.dtype)
        stream = torch.cat([ngram_pool[slots], ids], 1)
        # Window k of the unfold is the history after k rows.
        pending["ngram_candidates"] = stream.unfold(1, context, 1)[:, 1:]

    def stage_ple_conv_input(self, conv_input: torch.Tensor) -> None:
        """Stage the generation rows' conv input (`[T, C]`, token-major); a mixed batch's rows
        are copied so its context prefix is not kept alive through the draft loop."""
        pending = self._pending
        prefix, n, p = pending["prefix"], pending["n"], pending["p"]
        conv_x = torch.reshape(conv_input[prefix:], [n, p, -1])
        pending["conv_x"] = conv_x.clone() if prefix else conv_x

    def stage_index(self, position_cache: torch.Tensor) -> None:
        """Before an attention layer's index store: on the first, stage the position slots."""
        pending = self._pending
        if pending["position"] is None:
            pages, within = pending["pages"], pending["within"]
            pending["position"] = (position_cache, position_cache[pages, within])
        pending["staged_layers"] += 1

    # -------------------------------------------------------------- commit ---

    def commit_speculative_states(
        self,
        num_accepted_tokens: torch.Tensor,
        state_indices: torch.Tensor,
        num_contexts: int,
    ) -> None:
        """Rewrite the staged planes to the accepted prefix. Called by the spec worker."""
        pending, self._pending = self._pending, None
        if pending is None:
            return
        n, p = pending["n"], pending["p"]
        assert num_contexts == pending["num_contexts"], (
            f"the worker commits with {num_contexts} context requests and the forward staged "
            f"{pending['num_contexts']}"
        )
        assert state_indices.shape[0] == num_contexts + n, (
            f"the worker hands {state_indices.shape[0]} slots for a batch the forward staged "
            f"as {num_contexts} + {n} requests"
        )
        assert pending.get("conv_x") is not None and pending["staged_layers"] > 0, (
            "a verify step staged neither its PLE rows nor any attention layer; the forward "
            "did not run the staging it opened"
        )
        ptrs, strides, _ = self._planes
        position, position_old = pending["position"]
        pages, within = pending["pages"], pending["within"]
        if pending["fused"]:
            spec_rollback_commit(
                num_accepted_tokens[num_contexts : num_contexts + n],
                state_indices[num_contexts:],
                pending["ngram_pool"],
                pending["ngram_stream"],
                pending["conv_pool"],
                pending["conv_prev"],
                pending["conv_x"],
                position,
                position_old,
                pages,
                within,
                ptrs,
                strides,
                pending["index_all"],
            )
            return
        accepted = num_accepted_tokens[num_contexts : num_contexts + n].to(torch.long)
        slots = state_indices[num_contexts:].to(torch.long)
        rows = torch.arange(n, device=accepted.device)
        ngram_pool, conv_pool = pending["ngram_pool"], pending["conv_pool"]
        state_len = conv_pool.shape[-1]
        ngram_pool[slots] = pending["ngram_candidates"][rows, accepted - 1]
        # [n, C, state_len + p]; window k of the unfold is the state after k tokens.
        history = torch.cat([pending["conv_prev"], torch.transpose(pending["conv_x"], 1, 2)], 2)
        conv_pool[slots] = history.unfold(2, state_len, 1)[rows, :, accepted]
        # [n * p]: True at every ordinal the sampler rejected.
        rejected = torch.reshape(
            torch.arange(p, device=accepted.device).unsqueeze(0) >= accepted.unsqueeze(1), [-1]
        )
        qsa_rollback_restore(ptrs, strides, pages, within, rejected, pending["index_all"])
        position[pages, within] = torch.where(
            rejected[:, None], position_old, position[pages, within]
        )


# -------------------------------------------------------------------- shell ---


class _WidenedHead:
    """`lm_head` widening bf16 logits with the vectorized `bf16_to_fp32` (bit-identical to
    `.float()`), so the logits processor's own `.float()` is a no-op."""

    def __init__(self, lm_head) -> None:
        self.lm_head = lm_head

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        if logits.dtype == torch.bfloat16 and logits.is_contiguous():
            return bf16_to_fp32(logits)
        return logits


class _WideningLogitsProcessor(LogitsProcessor):
    """The shell's logits processor over `_WidenedHead`, for trunk and draft logits."""

    def forward(
        self, hidden_states, lm_head, attn_metadata, return_context_logits: bool = False
    ) -> torch.Tensor:
        return super().forward(
            hidden_states, _WidenedHead(lm_head), attn_metadata, return_context_logits
        )


def _field(block, name: str, default=None):
    """`block`'s `name` -- a mapping's entry or an object's attribute -- or `default` where it
    is absent or null."""
    value = block.get(name) if isinstance(block, dict) else getattr(block, name, None)
    return default if value is None else value


def _text_settings(text) -> dict:
    """`_TEXT_SETTINGS`' entries as `text` declares them. The rope fields are the target's
    reading: the `rope_parameters` dict, or the flattened `rope_scaling` with the theta and
    the partial factor at the top level."""
    rope = getattr(text, "rope_parameters", None) or getattr(text, "rope_scaling", None) or {}
    settings = {name: getattr(text, name, None) for name in _TEXT_SETTINGS}
    settings.update(
        rope_theta=rope.get("rope_theta", getattr(text, "rope_theta", None)),
        partial_rotary_factor=rope.get(
            "partial_rotary_factor", getattr(text, "partial_rotary_factor", None)
        ),
        mrope_section=tuple(rope.get("mrope_section") or ()),
        mrope_interleaved=rope.get("mrope_interleaved"),
        rope_factor=rope.get("factor"),
        attn_output_gate=_field(text, "attn_output_gate", True),
        mtp_num_hidden_layers=_field(text, "mtp_num_hidden_layers", 1),
        mtp_use_dedicated_embeddings=_field(text, "mtp_use_dedicated_embeddings", False),
    )
    mtp = getattr(text, "mtp", None)
    settings.update(
        {
            "mtp.num_hidden_layers": _field(mtp, "num_hidden_layers", 1),
            "mtp.layer_types": tuple(_field(mtp, "layer_types", ("full_attention",))),
            "mtp.mtp_use_hidden_state_from_layer": _field(mtp, "mtp_use_hidden_state_from_layer"),
        }
    )
    return settings


class ModelingV2Qwen38FlashNextFp8Sm103Unified(
    DecoderModelForCausalLM[Qwen4ExpUnifiedModelingV2, PretrainedConfig]
):
    """Qwen3.8-Flash-Next-FP8 at tp1, tp2, tep2, dep2, tp4, tep4 or dep4 on sm_103."""

    @classmethod
    def get_preferred_kv_cache_manager_version(cls, pretrained_config=None) -> str:
        """V2: the only KV cache manager with PLE state and QSA's index and position planes."""
        return "V2"

    @staticmethod
    def check_deployment(model_config: ModelConfig, llm_args: TorchLlmArgs, layout: str) -> None:
        """Raise `NoTarget` for a checkpoint or engine choice this target does not run.

        `route` calls it once the GPU, `layout` (routing's `_FP8_LAYOUTS`) and draft path match,
        so `auto` falls back to the built-in implementation instead of failing later."""
        cfg = model_config.pretrained_config
        text = getattr(cfg, "text_config", None)
        vision = getattr(cfg, "vision_config", None)
        if text is None or vision is None:
            raise NoTarget(
                f"{type(cfg).__name__} carries no text_config/vision_config; the target reads "
                "both halves of the normalized Qwen4-Exp composite"
            )
        shape = (
            text.num_hidden_layers,
            text.hidden_size,
            text.num_experts,
            text.num_attention_heads,
            text.linear_num_value_heads,
            text.vocab_size,
        )
        vshape = (
            vision.depth,
            vision.hidden_size,
            vision.num_heads,
            vision.spatial_merge_size,
            vision.out_hidden_size,
            vision.num_position_embeddings,
            tuple(getattr(vision, "deepstack_visual_indexes", None) or ()),
            vision.hidden_act,
        )
        if (shape, vshape) != (_TEXT_SHAPE, _VISION_SHAPE):
            raise NoTarget(
                f"text shape {shape}, vision shape {vshape}; the target is Qwen3.8-Flash-Next's "
                f"{_TEXT_SHAPE}, vision {_VISION_SHAPE}"
            )
        differs = {k: v for k, v in _text_settings(text).items() if v != _TEXT_SETTINGS[k]}
        if differs:
            raise NoTarget(
                f"the checkpoint has Qwen3.8-Flash-Next's shape but declares {differs}; the "
                f"target implements { ({k: _TEXT_SETTINGS[k] for k in differs}) }"
            )
        # The opt-in switches, read here so a misspelled value fails before the build.
        mapping = model_config.mapping
        _weights.moe_tpb_requested()
        _weights.switch(ROWTILE_SCRATCH_GROUPS_ENV)
        comm_method = _moe_comm_method()
        if comm_method not in ("ALLGATHER", "NVLINK_ONE_SIDED"):
            raise NoTarget(
                f"{MOE_COMM_METHOD_ENV}={comm_method} at {layout}: the target runs the MoE round "
                "trip as ALLGATHER (the default) or NVLINK_ONE_SIDED"
            )
        if comm_method == "NVLINK_ONE_SIDED":
            from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import (
                NVLinkOneSided,
            )

            if (
                mpi_disabled()
                or not mapping.enable_attention_dp
                or not NVLinkOneSided.is_platform_supported()
            ):
                raise NoTarget(
                    f"{MOE_COMM_METHOD_ENV}=NVLINK_ONE_SIDED at {layout}: the all-to-all MoE "
                    "round trip needs MPI, attention DP and MNNVL on this node"
                )
        spec = model_config.spec_config
        if spec is not None:
            # One linear MTP chain, verified at `max_draft_len + 1` rows on the GDN cached replay.
            if (
                getattr(spec, "num_nextn_predict_layers", None) not in (None, 1)
                or getattr(spec, "use_dynamic_tree", False)
                or getattr(spec, "eagle_choices", None) is not None
                or spec.max_total_draft_tokens != spec.max_draft_len
            ):
                raise NoTarget(
                    "an MTP draft other than one linear chain over the checkpoint's single MTP "
                    "module"
                )
            if not 1 <= spec.max_draft_len < _MAX_VERIFY_WIDTH:
                raise NoTarget(
                    f"max_draft_len={spec.max_draft_len}; the verify step is certified at "
                    f"max_draft_len 1..{_MAX_VERIFY_WIDTH - 1}"
                )
            if (
                getattr(spec, "draft_len_schedule", None) is not None
                or getattr(spec, "acceptance_rate_threshold", None) is not None
            ):
                raise NoTarget(
                    "the draft length changes at runtime (draft_len_schedule, max_concurrency or "
                    "acceptance_rate_threshold), but the verify step runs at the one width "
                    "max_draft_len + 1"
                )
            if not is_gdn_replay_enabled():
                raise NoTarget(
                    "MTP with TRTLLM_USE_GDN_REPLAY=0: the verify step runs the GDN cached replay, "
                    "not the legacy per-step recurrent states; leave TRTLLM_USE_GDN_REPLAY unset"
                )
            if is_gdn_flashinfer_replay_enabled() and spec.max_draft_len + 1 in (4, 8):
                raise NoTarget(
                    f"TRTLLM_USE_GDN_FLASHINFER_REPLAY with max_draft_len={spec.max_draft_len}: "
                    "the manager lays the replay history out for the FlashInfer ring kernel, "
                    "which the verify step does not run"
                )
        prefill_graph = str(llm_args.prefill_cuda_graph_backend.value)
        if prefill_graph == "piecewise":
            raise NoTarget(
                "prefill_cuda_graph_backend='piecewise' makes the engine torch.compile the model, "
                "which the target does not implement; 'breakable' runs the prefill step as a "
                "breakable CUDA graph"
            )
        if mapping.enable_attention_dp and prefill_graph == "breakable":
            raise NoTarget(
                f"prefill_cuda_graph_backend='breakable' with attention DP ({layout}): the engine "
                "pads every rank to one prefill bucket, so a rank holding only generation "
                "requests is handed more rows than its batch, which the decode step refuses"
            )
        windows = llm_args.kv_cache_config.max_attention_window
        full = llm_args.max_seq_len or model_config.max_seq_len
        if windows is not None and (
            any(w < full for w in windows) if full else len(set(windows)) > 1
        ):
            raise NoTarget(
                f"kv_cache_config.max_attention_window={windows}; the target implements no "
                "sliding window: every attention layer's window is max_seq_len"
            )
        if getattr(llm_args.multimodal_config, "vision_gpu_normalize", False):
            raise NoTarget(
                "multimodal_config.vision_gpu_normalize: the input processor then ships uint8 "
                "pixels, and the vision tower runs at the bf16 activation dtype"
            )
        kv_cache_quant_algo = engine_kv_cache_quant_algo(model_config, llm_args)
        if kv_cache_quant_algo not in (None, QuantAlgo.FP8, QuantAlgo.NVFP4):
            raise NoTarget(
                f"a {kv_cache_quant_algo} KV cache; the target stores K/V as the checkpoint's "
                "bf16, as FP8 or as NVFP4"
            )
        if model_config.torch_dtype != torch.bfloat16:
            raise NoTarget(
                f"the engine runs the model in {model_config.torch_dtype}; the target is bf16"
            )
        if str(model_config.moe_backend_requested).upper() not in _TRUNK_MOE_BACKENDS:
            raise NoTarget(
                f"moe_config.backend={model_config.moe_backend_requested!r}; the target runs the "
                f"routed experts on {', '.join(_TRUNK_MOE_BACKENDS)}"
            )
        if llm_args.moe_config.load_balancer is not None and mapping.moe_ep_size > 1:
            raise NoTarget(
                f"moe_config.load_balancer (EPLB) at {layout}: the target runs fixed contiguous "
                "expert windows and registers no layer with the balancer, so the placement "
                "would do nothing"
            )
        state_dtype = effective_mamba_ssm_cache_dtype(
            model_config, llm_args.kv_cache_config.mamba_ssm_cache_dtype
        )
        if state_dtype not in _SSM_STATE_DTYPES:
            served = " or ".join(str(dtype).removeprefix("torch.") for dtype in _SSM_STATE_DTYPES)
            raise NoTarget(
                f"a {state_dtype} recurrent-state cache; the target serves {served} "
                "(kv_cache_config.mamba_ssm_cache_dtype)"
            )

    @property
    def inforward_snapshot_min_main(self) -> int:
        """Fewest chunk tokens before the snapshot point for an in-forward snapshot split."""
        return _INFORWARD_MIN_MAIN

    def __init__(self, model_config: ModelConfig):
        cfg = model_config.pretrained_config
        assert cfg is not None
        # Adopt the normalized text tower as the config in place: the pools are sized from it,
        # and the engine keeps its own reference to this `ModelConfig`.
        text = getattr(cfg, "text_config", None)
        assert text is not None, (
            "this target is the language tower of the composite Qwen4-Exp export; "
            "pretrained_config carries no text_config"
        )
        assert getattr(text, "architectures", None) == ["Qwen4ExpForCausalLM"], (
            f"the text tower must be the normalized Qwen4-Exp text config; got architectures "
            f"{getattr(text, 'architectures', None)!r}"
        )
        composite = cfg
        model_config.pretrained_config = text
        super().__init__(
            Qwen4ExpUnifiedModelingV2(model_config, composite),
            config=model_config,
            hidden_size=text.hidden_size,
            vocab_size=text.vocab_size,
        )
        # `lm_head` is outside the quantization scheme: a dense bf16 `LMHead`.
        assert self.lm_head.weight.dtype == torch.bfloat16, (
            f"expected a bf16 lm_head; the shell built {self.lm_head.weight.dtype}"
        )
        self.logits_processor = _WideningLogitsProcessor()
        # No separate draft checkpoint: the MTP module's weights are the target's own.
        self.draft_config = None
        self.spec_config = getattr(model_config, "spec_config", None)
        self.draft_model = None
        self.spec_worker = None
        # Read by the engine (`model_engine._spec_decode_fast_paths`, `runtime_hooks_of`).
        self.spec_decode_fast_paths = False
        self.runtime_hooks = None
        if self.spec_config is not None:
            self.draft_model = DraftModel(self.model, self.lm_head, self.logits_processor)
            # Must match the resource manager, or the draft step gets no draft layer mapped.
            self.use_separate_draft_kv_cache = should_use_separate_draft_kv_cache(self.spec_config)
            from tensorrt_llm._torch.distributed import ll_exchange

            # The engine's one-engine fast paths, written for linear one-model MTP-Eagle: the
            # only speculation routing and check_deployment admit.
            self.spec_decode_fast_paths = True
            self.runtime_hooks = _runtime_hooks()
            # Collective over the TP group (every rank constructs the model in step);
            # None without attention TP or peer memory (the NCCL gathers stay).
            key_exchange = ll_exchange.create_tp_key_exchange(
                model_config.mapping, model_config.max_num_tokens
            )
            self.spec_worker = get_spec_worker(
                self.spec_config,
                model_config,
                model_config.mapping,
                use_separate_draft_kv_cache=self.use_separate_draft_kv_cache,
                key_exchange=key_exchange,
                spec_decode_fast_paths=True,
            )
            assert self.spec_worker is not None, (
                f"the engine resolved no spec worker for spec_dec_mode "
                f"{self.spec_config.spec_dec_mode!r}"
            )
            if getattr(self.spec_worker, "key_exchange", None) is not None:
                self.draft_model.mtp_layers[0].worker = self.spec_worker
            self.spec_worker.set_draft_model(self.draft_model)
            # Nothing the runtime owns rolls back the PLE pools and QSA side caches.
            self.spec_worker.register_auxiliary_state_handler(self.model.spec_rollback)
        # Media placeholder ids, built lazily (`__init__` runs under meta init).
        self._mm_token_ids_value = (self.model.image_token_id, self.model.video_token_id)
        self._mm_token_ids_cache: torch.Tensor | None = None
        # Replicated on every rank; built once the weights are real (`post_load_weights`).
        self.vision_tower: _vision.VisionTower | None = None

    @property
    def multimodal_token_ids(self) -> torch.Tensor:
        """Placeholder ids whose rows vision embeddings replace (in vocabulary here, so the
        executor's out-of-vocabulary fallback would select nothing)."""
        if self._mm_token_ids_cache is None:
            self._mm_token_ids_cache = torch.tensor(self._mm_token_ids_value, dtype=torch.int32)
        return self._mm_token_ids_cache

    @property
    def multimodal_data_device_paths(self) -> list[str]:
        """`multimodal_data` tensors moved to the device; the `(t, h, w)` grids stay on the host."""
        return list(_vision.DEVICE_PATHS)

    def validate_request(self, request) -> None:
        """Reject at admission a media request the vision tower would assert on."""
        _vision.validate_request(self.vision_tower, request)

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
        """One engine step: `begin_step`, the vision tower (the inherited forward drops
        `multimodal_params`), then the trunk; with a spec worker the shell gathers the verified
        logits rows (`spec_metadata.gather_ids`) and hands the worker the ungathered stream."""
        self.model.begin_step(
            attn_metadata,
            input_ids,
            position_ids,
            inputs_embeds=inputs_embeds,
            lora_params=lora_params,
            spec_metadata=spec_metadata,
            vision=_vision.multimodal_embeds(self.vision_tower, kwargs),
        )
        if self.spec_worker is None:
            assert spec_metadata is None, (
                "spec_metadata arrived without a spec worker; this engine was built without a "
                "speculative_config"
            )
            return super().forward(
                attn_metadata,
                input_ids,
                position_ids,
                inputs_embeds,
                return_context_logits,
                spec_metadata,
                lora_params,
                **kwargs,
            )
        assert spec_metadata is not None, "the spec worker is built but no spec_metadata came"
        final, hc = self.model.split_spec_output(
            self.model(
                attn_metadata=attn_metadata,
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                lora_params=lora_params,
                spec_metadata=spec_metadata,
                **kwargs,
            )
        )
        rows = nn.functional.embedding(spec_metadata.gather_ids, final)
        if self.spec_worker.key_exchange is not None:
            # This rank's vocabulary shard for a greedy/temperature-only batch, else gathered fp32.
            logits = self.spec_worker.target_logits(
                rows,
                self.lm_head,
                self.logits_processor,
                attn_metadata,
                spec_metadata,
                self.draft_model,
            )
        else:
            logits = self.logits_processor.forward(rows, self.lm_head, attn_metadata, True)
        # The draft head decides from the same batch whether its logits may stay sharded.
        self.draft_model.mtp_layers[0].spec_metadata = spec_metadata
        # A prefill CUDA graph pads the rows; the worker reshapes per request, so trim them.
        if attn_metadata.padded_num_tokens is not None:
            n = attn_metadata.num_tokens
            input_ids, hc = input_ids[:n], hc[:n]
            position_ids = position_ids[..., :n]
        return self.spec_worker(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hc,
            logits=logits,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=self.draft_model,
            resource_manager=kwargs.get("resource_manager"),
        )

    def load_weights(self, weights, *args, **kwargs):
        # A reload re-derives everything `post_load_weights` builds from the weights.
        self.model._prefill = self.model._decode = None
        _weights.MODEL_WEIGHTS.load(self, weights)

    def post_load_weights(self):
        super().post_load_weights()
        self.model.post_load_weights()
        self.vision_tower = _vision.VisionTower(self.model.w, self.model.vision_config)
        # The CPU-only load checks build the shell over host weights; they launch nothing.
        if self.lm_head.weight.is_cuda:
            self.vision_tower.warm_up()
            self._load_eager_gemm_kernels()

    def _load_eager_gemm_kernels(self) -> None:
        """Load, once per rank, every cuBLASLt kernel an eager bf16 GEMM can pick (the core's
        projections and `lm_head`, at `_projection_warm_rows`) so no serving step loads one."""
        core, head = self.model, self.lm_head
        device = head.weight.device
        projections = _eager_projections(
            core, core._prefill._layers, core._prefill._stack, core._mtp
        )
        _load_projection_kernels(projections, device)
        rows = min(core.max_num_tokens, _LM_HEAD_WARM_ROWS)
        x = torch.zeros([rows, head.in_features], dtype=head.weight.dtype, device=device)
        for m in _projection_warm_rows(1, rows):
            head.apply_linear(x[:m], head.bias)
        torch.cuda.synchronize(device)
        # Load the grouped routed-experts path before any prefill CUDA graph is captured.
        dims = getattr(core, "dims", None)
        if (
            _MOE_PREFILL_GROUPED_MIN_ROWS
            and dims is not None
            and getattr(core, "dp_size", 1) == 1
            and core.max_num_tokens >= _MOE_PREFILL_GROUPED_MIN_ROWS
        ):
            warm_prefill_grouped(device, dims.local_experts)
