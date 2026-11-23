# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: Kimi K3.1 Pro (MXFP4) / sm_103 / tep (the attention-TP perf layouts: TEP8 with MoE EP8, MoE
TP2 x EP4 or MoE TP4 x EP2; TEP16 with MoE TP4 x EP4; TEP32 with MoE TP4 x EP8).

Routing (``routing.py``, ``_SM103_TEP_LAYOUTS``) sends SM 10.3 deployments at these layouts with an FP8 latent KV cache
here; every other admitted deployment runs the general target (``kimi_k31_pro_mxfp4__sm_10x__default``). The model is
the general target's model plus the performance paths below. Where none of them serves a step, the step runs the general
target's operations (that target's module docstring lists them).

**Architecture.** Kimi K3.1's language model: 93 layers, hidden 7168, vocabulary 163840. Every fourth layer (3, 7,
..., 91) and the last layer (92) are MLA attention, 24 layers: 96 query heads, q_lora 1536, kv_lora 512, a 64-wide rope
slot that K3.1 leaves unrotated (NoPE; the target installs an identity RoPE table), an output gate (`g_proj`, sigmoid)
before `o_proj`, and a DSA indexer (32 index heads x 128, no indexer RoPE, top-2048 keys per query) that selects which
cached latents each query attends to. The other 69 layers are Kimi Delta Attention (KDA): 96 heads x 128, a short
convolution of width 4, a gated delta-rule recurrence with per-request conv and recurrent state, and FP32 gate
parameters (`A_log`, `dt_bias`). Layer 0's MLP is dense (SiTU, 33792 wide); every other layer's MLP is a latent MoE:
router over 896 experts (sigmoid scores with a correction bias, top-16, renormalized), the routed experts in a
3584-wide latent space (down projection, experts of width 3072 with SiTU, RMSNorm, up projection) and two shared
experts (6144 wide, SiTU). Attention residuals (blocks of 12 layers) mix each sublayer's input from a bank of earlier
outputs. The routed experts are MXFP4 (Moonshot's `mxfp4-pack-quantized`, served as W4A8_MXFP4_MXFP8); attention, the
shared and dense MLPs, the LM head and the vision tower are BF16. The vision tower is MoonViT (27 layers, 12 heads,
hidden 1024) with a patch merger (`PatchMergerMLPV2`) into the text hidden size (`K31VisionTower`, in the
vision tower section).

**Steps.** `PrefillTarget` serves steps with context rows: the context requests' image embeddings (vision tower +
projector, encoded in the first step that covers them) are spliced over the media placeholders of the input
embeddings, then the text model runs. `DecodeTarget` serves generation-only steps (every captured decode graph): the
token ids go straight to the text model. Attention TP puts 12 (TEP8), 6 (TEP16) or 3 (TEP32) query heads on a rank,
so the MLA never pads its heads; the KDA and shared-expert heads are split over the TP group as well.

**What this target runs beyond the general target** (target ops under `ops/<category>/`, their kernels under
`kernels/<category>/`):
* attention residuals: the family PDL mixes (`ops/norm/attn_res_rmsnorm`, `attn_res_add_rmsnorm`), the engine's
  persistent mix, the mix that takes the fused MoE tail's output from its mailbox
  (`ops/norm/attn_res_mailbox_add_rmsnorm`), and the output projection with its all-reduce as one push chain over the
  MNNVL workspace, finished in the next mix (`ops/comm/oproj_allreduce_push`,
  `ops/norm/oproj_push_attn_res_add_rmsnorm`);
* the decode and verify BF16 projections on the measured low-M GEMM plans of each layout (`ops/gemm/low_m_plans`,
  `low_m_linear`);
* MLA: the token-wise projections of the layer input on one packed weight (`ops/attention/dsa_packed_proj`); the
  token-sharded sparse MLA prefill, which moves the queries from head shards to token shards and back
  (`ops/comm/rows_alltoall`) and runs the PrimTS FP8 sparse MLA over all heads of its rows
  (`ops/attention/mla_query_e4m3`, `sparse_mla_prims_ts`; not bitwise to the head-sharded prefill); on decode steps
  of up to 25 rows the query chain on the auxiliary stream beside the indexer; the GVR decode top-k's early release;
* KDA: the gate projection of a few verify rows in one launch (`ops/ssm/kda_fab_proj`), the FP32 verify recurrence
  (`ops/ssm/kda_verify_rec`), and a segment-parallel prefill chain for one long sequence
  (`ops/ssm/kda_prefill_segments`);
* the MoE: up to 32 rows the sharded MoE front (`ops/moe/moe_front`: router, latent down projection, top-16 and
  MXFP8 input in one kernel, at some TP sizes with the shared expert's gate_up and SiTU) and the fused latent-MoE tail
  (`ops/moe/latent_moe_tail`: the routed all-reduce with the RMSNorm, the shared reduce-scatter and the sharded up
  projection with a multicast all-gather; at TEP32 a two-shot instance up to 192 rows); above the front's rows the
  one-launch MoE head (`ops/moe/moe_head_gemm`) or the head sharded over the TP group (`ops/moe/moe_head_shard`), the
  CuTe DSL W4A8 grouped GEMMs (`ops/moe/w4a8_moe_cute`), and stream priorities that keep the routed critical path
  ahead of the shared expert;
* sequence-parallel prefill: the residual stream on 1 / TP of the step's rows, with row all-gathers and
  reduce-scatters over NVLink peer mappings (`ops/comm/seq_parallel_rows_mnnvl`; NCCL `seq_parallel_rows` for the
  tensors they do not serve);
* DFlash2: the vocab-parallel logits gather and FP32 all-gather in one multicast kernel each
  (`ops/comm/lamport_vocab_gather`, `lamport_allgather`), the drafter's push projection and block-convolution epilogue
  (`ops/spec/dflash2_push_conv_add_rmsnorm`, `dflash2_conv_add_rmsnorm`), the generation requests' drafter inputs in
  one launch (`ops/spec/dflash2_gen_prepare`), the verify's target probabilities (`ops/spec/fused_compute_probs_split`,
  `topp_probs_cluster`), and the drafter context projection on a side stream;
* sharded cold pages: with block reuse, each rank keeps only its slice of an evicted MLA latent or index-key page in
  the host and disk tiers (`ops/common/shard_cold_page_codec`);
* the eager forward on one large chunk of the interpreter's data stack (`run_on_large_datastack_chunk`).

**Shared with the family.** The family's top level holds routing (`routing.py`), the config and the engine hooks
(`config.py`), the input processor (`input_processor.py`), and the ops and kernels the targets call
(`ops/<category>/`, `kernels/<category>/`): among them the DSA attention, metadata, indexer and graph ops
(`ops/attention/`), the routed experts (`ops/moe/routed_experts`), the hybrid KDA (+ DSA index-key) cache managers
(`ops/ssm/`) and the DFlash2 selection, sampling, verification, ring and draft-page ops (`ops/spec/`).

**Code layout.** `modeling.py` holds the whole model: the KDA layer, the MLA attention, the MoE, the decoder layer,
the text model and causal LM, the DFlash2 drafter and its worker (`K31DFlashWorker` over the engine's
`DFlashWorker`), the vision tower and the vision-language wrapper. `weights.py` holds the checkpoint loading (the lazy
checkpoint reader, every key mapping, conversion and sharding) with this target's MLA packing. The target's own ops
live under `ops/<category>/`, their kernels under `kernels/<category>/`.

**What this target asserts** (`routing.py` raises `NoTarget` with the reason): the general target's rules (the K3.1
shape with a `dsa_config` and the MXFP4 checkpoint, DSA with FP8 index keys, an FP8 latent KV cache, KV cache manager
V2, beam width 1, no KV pool rebalance, 32-, 64- or 128-token pages, the TRTLLM MoE backend, the Python NIXL
transceiver for disaggregated serving, DFlash2 or no speculative decoding), sm_103, and one of the layouts above
without attention DP or an EP load balancer. Construction checks the MoE backend the engine resolved.

**KV dtypes.** The MLA latent cache is FP8 (per-tensor E4M3, scale 1; `kv_cache_config.dtype: fp8`); the index keys
are FP8 with a per-128 UE8M0 scale in a V2 extra buffer; the KDA states are FP32.

**Speculative decoding: DFlash2.** One-engine DFlash with the DFlash2 MLA drafter (`DraftModel`: 5 MLA layers over the
target's hidden states at layers 7, 23, 51, 67, 83, 91, block convolutions, a low-rank candidate selector over the
target's LM head) drafts a block of `max_draft_len` tokens; the target verifies the block in one step (KDA replay
verify, DSA sparse MLA with `next_n` = block width). The drafter's window attention runs on the TRTLLM-Gen sparse MLA
decode kernel (`ops/spec/dflash2_window_attention`), its context latents live in the worker's paged ring
(`ops/spec/dflash2_ring`) and in the KV cache manager's draft pages (`ops/spec/dflash2_draft_pages`, which carry them
between disaggregated servers and keep them with a reused conversation), and the worker selects, samples and accepts
through the family spec ops. A disaggregated context server (`TRTLLM_DISAGG_ROLE=context`) runs a context-only
drafter.

**Out of scope:** attention DP (the dep target), sm_107 and sm_100, the `nvfp4_ds_mla` KV cache and FP4 index keys,
pipeline and context parallelism, an EP load balancer, MoE backends other than TRTLLM, and speculative decoding other
than DFlash2."""

from __future__ import annotations

import contextlib
import copy
import inspect
import json
import math
import os
import weakref
from collections import deque
from dataclasses import replace
from types import SimpleNamespace
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from tensorrt_llm._device_copy import DeviceCopyBuffer
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionInputType,
    AttentionMetadata,
    PositionalEmbeddingParams,
    RopeParams,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modeling_v2._router_index import step_contract_enabled
from tensorrt_llm._torch.modeling_v2.catalog.attention.mla_rope_generation import (
    mla_rope_generation,
)
from tensorrt_llm._torch.modeling_v2.catalog.gemm.bmm_out import bmm_out

# Catalog entries the family routed-expert ops launch (the TRTLLM-Gen W4A8_MXFP4_MXFP8 runner, DeepSeek-V3 routing,
# the MXFP8 input quantization); their tests certify this target's shapes. The modeling_v2 claims test requires a
# target to import every catalog entry whose TARGET_CELLS name it, so the unused ones stay imported.
from tensorrt_llm._torch.modeling_v2.catalog.moe.mxe4m3_mxe2m1_block_scale_moe_runner import (  # noqa: F401
    mxe4m3_mxe2m1_block_scale_moe_runner,
)
from tensorrt_llm._torch.modeling_v2.catalog.moe.noaux_tc_op import noaux_tc_op  # noqa: F401
from tensorrt_llm._torch.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (
    flashinfer_fused_add_rmsnorm,
)
from tensorrt_llm._torch.modeling_v2.catalog.norm.flashinfer_rmsnorm import flashinfer_rmsnorm
from tensorrt_llm._torch.modeling_v2.catalog.quantization.mxfp8_quantize import (  # noqa: F401
    mxfp8_quantize,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.config import (
    dsa_model_defaults,
    text_config_of,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.activation.sigmoid_gate_mul import (
    k31_sigmoid_gate_mul,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.activation.silu_and_mul import silu_and_mul
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.convert_req_index_to_global import (
    convert_req_index_to_global,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_attention import (
    K31MLAAttention,
    K31PerfDSABackend,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_graph_ops import (
    DeferredQB,
    forward_sparse_attn_custom_op,
    initialize_sparse_attn,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_indexer import (
    deterministic_topk_order,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_metadata import (
    K31DSAtrtllmAttentionMetadata as K31DSAtrtllmAttentionMetadata,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_packed_proj import (
    packed_output_gate,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_params import (
    K31DSAParams,
    k31_dsa_params,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.indexer_topk_decode_gvr import (
    enable_gvr_early_release,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.mla_outputs import (
    create_mla_outputs,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.comm.allreduce_module import (
    FUSION_NONE,
    FUSION_RMS_NORM,
    allreduce_module,
    tp_allreduce,
    uses_mnnvl,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.common.multi_stream import (
    do_multi_stream,
    maybe_execute_in_parallel,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.common.step_contract import (
    check_step_contract,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.gemm.bf16_linear import (
    K31Linear,
    K31SituMLP,
    bf16_linear,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.gemm.fused_a_gemm import k31_fused_a_gemm
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.gemm.low_m_linear import (
    k31_low_m_linear,
    k31_planned_projection,
    low_m_choice,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.gemm.low_m_plans import (
    k31_low_m_draft_plan,
    k31_low_m_plan,
    k31_low_m_shared_expert_plan,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.dsv3_router_gemm_op import (
    dsv3_router_gemm_op,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.moe_prologue import (
    quant as prologue_quant,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.moe_prologue import (
    route as prologue_route,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.moe_prologue import runner_routing_bias
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.moe_residual_add import (
    k31_moe_residual_add,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.routed_experts import K31RoutedExperts
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.w4a8_moe_cute import (
    k31_w4a8_moe_cute,
    max_tiles_for,
    moe_workspace_bytes,
    pick_n_tile,
    prerouted_sort,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.norm.attn_res_add_rmsnorm import (
    k31_attn_res_add_rmsnorm,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.norm.attn_res_add_rmsnorm_fwd import (
    attn_res_add_rmsnorm_fwd,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.norm.attn_res_add_rmsnorm_persistent_fwd import (
    attn_res_add_rmsnorm_persistent_fwd,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.norm.attn_res_fwd import attn_res_fwd
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.norm.attn_res_rmsnorm import (
    k31_attn_res_rmsnorm,
    set_early_trigger_max_rows,
    set_reg_cap_min_rows,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.norm.attn_res_rmsnorm_fwd import (
    attn_res_rmsnorm_fwd,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.norm.rms_norm import K31RMSNorm
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.decode_draft_block_table import (
    decode_draft_block_table,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_draft_pages import (
    DFlash2DraftPagesWorkerMixin,
    dflash2_attention_mode,
    dflash2_context_kv_only,
    dflash2_draft_pages_manager,
    dflash2_sliding_window,
    dflash2_store_draft_pages,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_global_top32 import (
    TOP_K as K31_SELECTOR_TOP_K,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_global_top32 import (
    TOP_K as SELECTOR_TOP_K,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_global_top32 import (
    k31_dflash2_merge_top32,
    k31_dflash2_shard_index_pairs,
    k31_dflash2_shard_top32,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_grouped_conv import (
    K31DFlash2BlockConv,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_ring import (
    dflash2_ring_chunk_start,
    dflash2_ring_positions,
    dflash2_ring_rows,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_sampling import (
    dflash2_sample_draft,
    dflash2_sample_draft_from_candidates,
    dflash2_sampled_step,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_selector_greedy_path import (
    K31DFlash2CandidateSelector,
    k31_dflash2_selector_greedy_path,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_selector_greedy_tokens import (
    k31_dflash2_selector_greedy_tokens,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_step_bookkeeping import (
    MAX_REQUESTS as K31_BOOKKEEPING_MAX_REQUESTS,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_step_bookkeeping import (
    k31_dflash2_accept_record,
    k31_dflash2_draft_tail,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_verify import (
    dflash2_sample_and_accept,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_window_attention import (
    dflash2_window_attention,
    dflash2_window_attention_buffers,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_window_rows import (
    dflash2_window_rows as k31_dflash2_window_rows,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.flashinfer_compute_probs_from_logits import (  # noqa: E501
    flashinfer_compute_probs_from_logits,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.flashinfer_sample_from_logits import (
    flashinfer_sample_from_logits,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.fused_sample_from_logits import (
    fused_sample_from_logits,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.mla_norm_rope import (
    build_mla_yarn_freqs_cis,
    resolve_mla_rope_params,
    yarn_get_mscale,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.mla_norm_rope import (
    dflash2_mla_block_rope as k31_dflash2_mla_block_rope,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.mla_norm_rope import (
    dflash2_mla_ctx_norm_rope as k31_dflash2_mla_ctx_norm_rope,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.mla_norm_rope import (
    dflash2_mla_ctx_norm_rope_store as k31_dflash2_mla_ctx_norm_rope_store,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.strict_accept import strict_accept
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.unified_draft_cache_worker import (
    UnifiedDraftCacheWorkerMixin,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.vocab_parallel_argmax import (
    vocab_parallel_argmax,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.vocab_parallel_sample import (
    vocab_parallel_sample,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.copy_kda_replay_conv_window import (
    copy_kda_replay_conv_window,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.kda_decode import kda_decode
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.kda_dsa_cache_manager import (
    KimiK3DSAHybridCacheManager,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.kda_graph_ops import (
    maybe_bcg_kda_core_inplace,
    register_kda_layer,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.kda_metadata import KdaMetadata
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.kda_mtp_verify import kda_mtp_verify
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.kda_prefill_conv import (
    kda_prefill_conv,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.reset_recurrent_state_rows import (
    reset_recurrent_state_rows,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.rms_norm_gated_token_major import (
    rms_norm_gated_token_major,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.vision.moonvit_attention import (
    moonvit_attention,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.vision.moonvit_gelu import moonvit_gelu
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.vision.moonvit_patch_embed import (
    moonvit_patch_embed,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.vision.moonvit_rms_norm import (
    moonvit_rms_norm,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.vision.moonvit_rope_2d import (
    TABLE_SIDE,
    THETA,
    moonvit_rope_2d,
    moonvit_rope_angles,
    moonvit_rope_table,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.runtime import K31_RUNTIME_HOOKS
from tensorrt_llm._torch.models.modeling_multimodal_utils import (
    find_input_mm_embeds,
    get_multimodal_embeddings,
)
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.modules.kimi_kda.cache_manager import (
    KDAReplayState,
    KimiK3HybridCacheManager,
)
from tensorrt_llm._torch.nccl_window_tensor_scope import nccl_window_tensor_scoped_module
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import is_in_breakable_cuda_graph
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
    MambaHybridCacheManager,
    MambaHybridCacheManagerV2,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import BaseResourceManager
from tensorrt_llm._torch.pyexecutor.sampler.ops.flashinfer import resolve_advanced_sampling_filters
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import GREEDY_TEMPERATURE_THRESHOLD
from tensorrt_llm._torch.speculative import should_use_separate_draft_kv_cache
from tensorrt_llm._torch.speculative.dflash import (
    DFlashSpecMetadata,
    VocabParallelDraftTokens,
    dflash_allocated_ctx_limit,
    dflash_noise_block_embedding,
    dflash_position_ceiling,
)
from tensorrt_llm._torch.speculative.dflash import DFlashWorker as _EngineDFlashWorker
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase
from tensorrt_llm._torch.speculative.vocab_parallel import VocabShardedLogits
from tensorrt_llm._torch.utils import AuxStreamType, get_per_request_prefill_cuda_graph_flag
from tensorrt_llm._utils import maybe_pin_memory, prefer_pinned
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.llmapi.llm_args import is_dflash2_checkpoint
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo

from ..input_processor import MEDIA_PLACEHOLDER_TOKEN_ID, moonvit_max_image_grid
from ..ops.spec.rubin_dflash2_worker import agree_draft_tokens
from . import weights as target_weights
from .ops.attention.mla_query_e4m3 import k31_mla_query_e4m3
from .ops.attention.sparse_mla_prims_ts import full_lengths, k31_sparse_mla_prims_ts
from .ops.comm.lamport_allgather import RAISED_MAX_BYTES as LAMPORT_ALLGATHER_RAISED_MAX_BYTES
from .ops.comm.lamport_allgather import lamport_allgather
from .ops.comm.lamport_vocab_gather import lamport_vocab_gather, lamport_vocab_gather_serves
from .ops.comm.oproj_allreduce_push import lamport_workspace, oproj_allreduce_push, push_serves
from .ops.comm.rows_alltoall import rows_alltoall
from .ops.comm.seq_parallel_rows_mnnvl import MAX_RANKS as SP_MNNVL_MAX_RANKS
from .ops.comm.seq_parallel_rows_mnnvl import (
    reserve_sp_workspace,
    sp_gather_rows_mnnvl,
    sp_reduce_scatter_rows_mnnvl,
)
from .ops.moe import moe_front as _k31_front
from .ops.moe.latent_moe_tail import (
    KimiK3LatentMoETail,
    UnfinalizedRoutedOutput,
    fused_tail_twoshot_only_rows,
    k31_latent_moe_tail,
    k31_latent_moe_tail_copy_out,
    k31_latent_moe_tail_push_shared,
    k31_latent_moe_tail_to_mailbox,
)
from .ops.moe.moe_head_gemm import HEAD_CONFIGS as _K31_HEAD_CONFIGS
from .ops.moe.moe_head_gemm import MIN_ROWS as _K31_MOE_HEAD_MIN_ROWS
from .ops.moe.moe_head_gemm import head_config as _k31_head_config
from .ops.moe.moe_head_gemm import k31_moe_head_gemm
from .ops.moe.moe_head_shard import MAX_ROWS as _K31_HEAD_SHARD_MAX_ROWS
from .ops.moe.moe_head_shard import MIN_ROWS as _K31_HEAD_SHARD_MIN_ROWS
from .ops.moe.moe_head_shard import compile_shard as _k31_compile_head_shard
from .ops.moe.moe_head_shard import gather_bytes as _k31_head_shard_gather_bytes
from .ops.moe.moe_head_shard import k31_moe_head_shard, k31_moe_head_shard_routed
from .ops.moe.moe_head_shard import shard_config as _k31_head_shard_config
from .ops.moe.moe_head_shard import shard_serves as _k31_head_shard_serves
from .ops.moe.moe_head_shard import shard_views as _k31_head_shard_views
from .ops.norm.attn_res_mailbox_add_rmsnorm import k31_attn_res_mailbox_add_rmsnorm
from .ops.norm.oproj_push_attn_res_add_rmsnorm import k31_oproj_push_attn_res_add_rmsnorm
from .ops.spec.dflash2_conv_add_rmsnorm import (
    dflash2_conv_add_rmsnorm as k31_dflash2_conv_add_rmsnorm,
)
from .ops.spec.dflash2_gen_prepare import k31_dflash2_gen_prepare
from .ops.spec.dflash2_push_conv_add_rmsnorm import k31_dflash2_push_conv_add_rmsnorm
from .ops.spec.fused_compute_probs_split import SplitStreams, fused_compute_probs_split
from .ops.spec.topp_probs_cluster import k31_topp_probs_cluster
from .ops.spec.topp_probs_cluster import serves as topp_probs_cluster_serves
from .ops.ssm.kda_fab_proj import can_use_k3_kda_fab, k31_kda_fab_proj
from .ops.ssm.kda_prefill_segments import k31_kda_prefill_segments
from .ops.ssm.kda_verify_rec import k31_kda_verify_rec
from .ops.ssm.kda_verify_rec import supported as kda_verify_rec_supported
from .weights import PROJECTOR_PREFIX, VISION_TOWER_PREFIX, expert_ckpt_spec

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig


def _topp_cluster_serves(logits, temperatures, top_ks, top_ps) -> bool:
    """Whether the sampled verify's target probabilities take the cluster top-p kernel
    (``ops/spec/topp_probs_cluster``: one launch, several CTAs per row; the same distribution as the flashinfer chain
    of softmax + top_p_renorm_probs, not bit-equal at the top-p cut): top-p alone (the mode resolved top-k away, as
    NO_TOPK does) on inputs the kernel takes. Logs the path taken once per path."""
    if top_ks is not None or top_ps is None:
        return False
    if not topp_probs_cluster_serves(logits, temperatures, top_ps):
        logger.info_once(
            "Kimi K3.1 NO_TOPK target probabilities: flashinfer chain (inputs the cluster top-p kernel does not take).",
            key="k31_topp_probs_chain_inputs",
        )
        return False
    logger.info_once(
        "Kimi K3.1 NO_TOPK target probabilities: cluster top-p kernel "
        "(kimi_k31_pro_mxfp4__sm_103__tep/ops/spec/topp_probs_cluster).",
        key="k31_topp_probs_cluster",
    )
    return True


# ====================================================================================================================
# KDA layer
# ====================================================================================================================

# ---------------------------------------------------------------------------
# K31KdaLayer: K3.1's KDA layer.
# ---------------------------------------------------------------------------
#
# ``K31KdaLayer`` runs K3.1's one KDA configuration: BF16 projections, head_dim 128, a width-4 short convolution,
# the full-rank sigmoid output gate, a finite negative gate lower bound, FP32 ``A_log`` / ``dt_bias`` and the FP32
# recurrent state pool of the family cache manager (``ops/ssm/kda_cache_manager``). Every kernel goes through a family
# op:
#
# * prefill: the fused [q | k | v | g] GEMM, ``reset_recurrent_state_rows``, ``kda_prefill_conv``, the [f_a | b] -> f_b
#   gate GEMMs, ``kda_prefill`` (BT16), ``copy_kda_replay_conv_window`` and ``rms_norm_gated_token_major``;
# * one-token decode: the [q | k | v | g] GEMM beside the gate GEMMs on the auxiliary stream, then ``kda_decode`` on the
#   live pools by slot (output norm in the kernel);
# * DFlash2 verify: the same projections, then ``kda_mtp_verify`` on the manager's replay caches (output norm in its
#   epilogue), which commits the state after the golden token and caches the drafts for the next round's replay.
#
# After the weight load ``finalize_decode_weights`` fuses the [q | k | v | g] and [f_a | b] weights (the projections
# keep row views of them) and builds the kernels' constant weights. ``K31KDALinearAttention`` below is the layer this
# layout builds: it overrides the projections ``_project_gate_inputs`` and ``_project_output`` and defines the decode /
# verify projections (``_decode_qkvg``, ``_project_verify_inputs``), the prefill kernel (``_prefill_kernel``) and the
# verify (``forward_verify``).


#: Decode / verify steps of up to this many projection rows run the [f_a | b] -> f_b gate GEMMs on the auxiliary
#: stream beside the [q | k | v | g] GEMM.
KDA_BFA_MULTISTREAM_MAX_ROWS = 128


def _weight_holder(*shape: int, dtype: torch.dtype) -> nn.Module:
    """A module holding one uninitialized ``weight`` parameter: a KDA short convolution's [D, 1, W] weight under the
    checkpoint's ``<q|k|v>_conv1d.weight`` name."""
    holder = nn.Module()
    holder.weight = nn.Parameter(torch.empty(*shape, dtype=dtype))
    return holder


def _kda_decode_rows(tensor: torch.Tensor) -> torch.Tensor:
    """A one-token decode input as the kernel reads it: the view when its head and channel axes are packed."""
    if tensor.ndim == 4 and tensor.stride(3) == 1 and tensor.stride(2) == tensor.shape[3]:
        return tensor
    return tensor.contiguous()


class K31KdaLayer(nn.Module):
    """Kimi K3.1's KDA layer (see the module docstring); the targets subclass it for their layout's kernels."""

    def __init__(
        self,
        cfg,
        layer_idx: int,
        mapping=None,
        allreduce_strategy=None,
        aux_stream: Optional[torch.cuda.Stream] = None,
        extra_attrs: Optional[dict] = None,
    ) -> None:
        """``extra_attrs``: the model extra attributes the layer registers itself in (for the breakable CUDA graph's
        core break); None builds an unregistered layer."""
        super().__init__()
        lin = cfg.linear_attn_config
        self.hidden_size = cfg.hidden_size
        self.layer_idx = layer_idx
        self.head_dim = self.head_k_dim = lin["head_dim"]
        self.conv_size = lin["short_conv_kernel_size"]
        self.gate_lower_bound = lin.get("gate_lower_bound", None)
        lower_bound = self.gate_lower_bound
        if (
            self.head_dim != 128
            or self.conv_size != 4
            or not lin.get("use_full_rank_gate", True)
            or lower_bound is None
            or not math.isfinite(lower_bound)
            or lower_bound >= 0
        ):
            raise NotImplementedError(
                "the Kimi K3.1 KDA layer runs head_dim 128, a width-4 convolution, the full-rank output gate and a "
                f"finite negative gate_lower_bound (got {lin})"
            )

        self._kda_tp_size = mapping.tp_size if mapping is not None else 1
        self._kda_tp_rank = mapping.tp_rank if self._kda_tp_size > 1 else 0
        # Head-sharded TP sums the row-sharded o_proj partials.
        self._o_allreduce = (
            tp_allreduce(mapping, allreduce_strategy, torch.bfloat16)
            if self._kda_tp_size > 1
            else None
        )
        # Routing admits only layouts whose KDA ranks divide the heads.
        self.num_heads = lin["num_heads"] // self._kda_tp_size
        d = self.proj_size = self.num_heads * self.head_dim
        bf16 = torch.bfloat16

        # The checkpoint's projections and convolutions, under its parameter names; A_log and dt_bias keep the
        # checkpoint's FP32.
        self.q_proj = K31Linear(self.hidden_size, d, dtype=bf16)
        self.k_proj = K31Linear(self.hidden_size, d, dtype=bf16)
        self.v_proj = K31Linear(self.hidden_size, d, dtype=bf16)
        self.q_conv1d = _weight_holder(d, 1, self.conv_size, dtype=bf16)
        self.k_conv1d = _weight_holder(d, 1, self.conv_size, dtype=bf16)
        self.v_conv1d = _weight_holder(d, 1, self.conv_size, dtype=bf16)
        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16))
        )
        self.f_a_proj = K31Linear(self.hidden_size, self.head_dim, dtype=bf16)
        self.f_b_proj = K31Linear(self.head_dim, d, dtype=bf16)
        # Initialized (the checkpoint overwrites it): uninitialized memory can hold NaN bit patterns.
        self.dt_bias = nn.Parameter(
            torch.empty(d, dtype=torch.float32).uniform_(math.log(1e-3), math.log(1e-1))
        )
        self.b_proj = K31Linear(self.hidden_size, self.num_heads, dtype=bf16)
        self.g_proj = K31Linear(self.hidden_size, d, dtype=bf16)
        self.o_norm = K31RMSNorm(self.head_dim, cfg.rms_norm_eps, dtype=bf16)
        self.o_proj = K31Linear(d, self.hidden_size, dtype=bf16)

        # Built after the weight load (finalize_decode_weights): the fused [q | k | v | g] and [f_a | b] weights
        # (the projections keep row views of them), the decode kernel's transposed BF16 convolution weights, the
        # FP32 A_log / dt_bias / o_norm weight, and the packed BF16 / per-section FP32 convolution weights.
        self._qkvg_proj_weight: Optional[torch.Tensor] = None
        self._bfa_proj_weight: Optional[torch.Tensor] = None
        self._w_q_t = self._w_k_t = self._w_v_t = None
        self._A_log_f32 = self._dt_bias_f32 = self._onorm_w_f32 = None
        self._packed_conv_weight: Optional[torch.Tensor] = None
        self._mtp_conv_weights: Optional[Tuple[torch.Tensor, ...]] = None
        # Fork / join of the [f_a | b] -> f_b gate GEMMs beside the [q | k | v | g] GEMM.
        self._projection_aux_stream = aux_stream
        self._projection_fork_event = torch.cuda.Event()
        self._projection_join_event = torch.cuda.Event()
        # The in-place decode kernel's output buffer and zero convolution bias, allocated by the first decode.
        self._o_dense: Optional[torch.Tensor] = None
        self._kda_zero_bias: Optional[torch.Tensor] = None

        self.register_to_config = False
        self.layer_idx_str = str(layer_idx)
        if extra_attrs is not None:
            self.layer_idx_str = register_kda_layer(extra_attrs, self, layer_idx)
            self.register_to_config = True

    # ---------------------------------------------------------------------------------------------------------------
    # Weights
    # ---------------------------------------------------------------------------------------------------------------

    def finalize_decode_weights(self) -> None:
        """Fuse the [q | k | v | g] and [f_a | b] projection weights (the projections keep row views of the fused
        buffers) and build the decode / verify kernels' constant weights, once the checkpoint values are in."""
        if self.q_proj.weight.device.type != "cuda":
            return
        with torch.no_grad():
            self._qkvg_proj_weight = self._merge_projection_weights(
                (self.q_proj, self.k_proj, self.v_proj, self.g_proj)
            )
            self._bfa_proj_weight = self._merge_projection_weights(
                (self.f_a_proj, self.b_proj), pad_rows_to=8
            )
            self._w_q_t, self._w_k_t, self._w_v_t = (
                conv.weight.detach().squeeze(1).transpose(0, 1).to(torch.bfloat16).contiguous()
                for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d)
            )
            self._A_log_f32 = self.A_log.detach().float().contiguous()
            self._dt_bias_f32 = self.dt_bias.detach().float().contiguous()
            self._onorm_w_f32 = self.o_norm.weight.detach().float().contiguous()
            self._build_mtp_conv_weights()

    @staticmethod
    def _merge_projection_weights(modules, pad_rows_to: int = 1) -> torch.Tensor:
        """Concatenate the modules' weights (zero rows up to a multiple of ``pad_rows_to``) and repoint each module
        to its row view of the result."""
        weights = [module.weight.data for module in modules]
        padding = (-sum(weight.shape[0] for weight in weights)) % pad_rows_to
        if padding:
            weights.append(weights[0].new_zeros((padding, weights[0].shape[1])))
        fused = torch.cat(weights, dim=0).contiguous()
        offset = 0
        for module in modules:
            rows = module.weight.shape[0]
            module.weight.data = fused[offset : offset + rows]
            offset += rows
        return fused

    def _build_mtp_conv_weights(self) -> None:
        """The packed BF16 [3 * D, W] prefill convolution weight and the FP32 [D, W] per-section verify weights,
        built ahead of use (a lazy allocation under CUDA graph capture would bake capture-pool pointers in)."""
        conv_weights = tuple(
            conv.weight.detach().squeeze(1)
            for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d)
        )
        self._packed_conv_weight = torch.cat(conv_weights, dim=0).to(torch.bfloat16).contiguous()
        self._mtp_conv_weights = tuple(weight.float().contiguous() for weight in conv_weights)

    # ---------------------------------------------------------------------------------------------------------------
    # Projection and kernel hooks (the targets override these for their layout)
    # ---------------------------------------------------------------------------------------------------------------

    def _prefill_qkvg(
        self, x: torch.Tensor, x2d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(packed [T, 4 * D] [q | k | v | g] rows, output gate [1, T, D] view)`` of a prefill."""
        d = self.proj_size
        qkvg = bf16_linear(x, self._qkvg_proj_weight, False)
        return qkvg.squeeze(0), qkvg[..., 3 * d : 4 * d]

    def _project_gate_inputs(self, x: torch.Tensor):
        """``(beta, forget_gate)`` from the fused [f_a | b] GEMM and f_b."""
        hd, H = self.head_dim, self.num_heads
        bfa = bf16_linear(x, self._bfa_proj_weight, False)
        return bfa[..., hd : hd + H], bf16_linear(bfa[..., :hd], self.f_b_proj.weight, False)

    def _split_verify_qkvg(self, qkvg: torch.Tensor, forget_gate: torch.Tensor, beta: torch.Tensor):
        """``(q, k, v, forget_gate, beta, output_gate)`` with q / k / v / output gate as views of ``qkvg``."""
        d = self.proj_size
        q_proj, k_proj, v_proj = qkvg[..., : 3 * d].split(d, dim=-1)
        return q_proj, k_proj, v_proj, forget_gate, beta, qkvg[..., 3 * d : 4 * d]

    def _project_output(self, core: torch.Tensor) -> torch.Tensor:
        """o_proj of the post-norm core, summed over the head-sharded TP group."""
        out = bf16_linear(core.reshape(-1, self.proj_size), self.o_proj.weight, False)
        if self._o_allreduce is not None:
            out = allreduce_module(out, self._o_allreduce, FUSION_NONE, None, None)
        return out

    # ---------------------------------------------------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------------------------------------------------

    def forward(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        """The core eagerly, or as one break of a breakable CUDA graph; padding rows come out as zeros."""
        if self.register_to_config and is_in_breakable_cuda_graph():
            core = hidden_states.new_empty(
                (hidden_states.shape[0], self.num_heads, self.head_dim),
                dtype=torch.bfloat16,
            )
            maybe_bcg_kda_core_inplace(hidden_states, self.layer_idx_str, core)
        else:
            core = self._forward_impl(hidden_states, attn_metadata)
            num_padding_rows = hidden_states.shape[0] - attn_metadata.num_tokens
            if num_padding_rows > 0:
                # A step padded for a prefill CUDA graph but run eagerly (graph warmup, or a bucket
                # without a capture): the core covers the live rows, the padding rows are zero.
                core = torch.cat([core, core.new_zeros((num_padding_rows,) + core.shape[1:])])
        return self._project_output(core)

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The step's prefill rows, then its decode or verify rows; into ``output`` (the breakable graph's core
        buffer) when given, else a new core."""
        mamba_metadata = attn_metadata.mamba_metadata
        num_prefills = attn_metadata.num_contexts
        num_ctx_tokens = attn_metadata.num_ctx_tokens
        num_tokens = attn_metadata.num_tokens
        batch_size = attn_metadata.seq_lens.shape[0]
        state_indices = mamba_metadata.state_indices[:batch_size]
        cu_seqlens = mamba_metadata.query_start_loc_long[: num_prefills + 1]
        num_decodes = batch_size - num_prefills

        layer_cache = attn_metadata.kv_cache_manager.mamba_layer_cache(self.layer_idx)
        conv_pool = layer_cache.conv  # [slots, 3D, W - 1] bf16
        ssm_pool = layer_cache.temporal  # [slots, H, V, K] fp32
        generation_state_indices = mamba_metadata.generation_state_indices
        if generation_state_indices is None:
            generation_state_indices = state_indices[num_prefills:]

        cores: List[torch.Tensor] = []
        if num_prefills > 0:
            prefill_core = self.forward_prefill(
                hidden_states[:num_ctx_tokens],
                cu_seqlens,
                mamba_metadata,
                num_prefills,
                conv_pool,
                ssm_pool,
                state_indices[:num_prefills],
                layer_cache,
                output=output[:num_ctx_tokens] if output is not None else None,
            )
            if output is None:
                cores.append(prefill_core)
        if num_decodes > 0:
            decode_rows = num_tokens - num_ctx_tokens
            gen_output = output[num_ctx_tokens:num_tokens] if output is not None else None
            if decode_rows == num_decodes:
                core = self.forward_decode(
                    hidden_states[num_ctx_tokens:num_tokens],
                    conv_pool,
                    ssm_pool,
                    generation_state_indices,
                    mamba_metadata,
                    layer_cache,
                    output=gen_output,
                )
            else:
                # Speculative verification: each generation request carries 1 + draft_len tokens (drafts are
                # padded to the static max, so the count is uniform).
                assert decode_rows % num_decodes == 0, (
                    f"ragged generation batch: {decode_rows} tokens for {num_decodes} requests"
                )
                core = self.forward_verify(
                    hidden_states[num_ctx_tokens:num_tokens],
                    decode_rows // num_decodes,
                    layer_cache,
                    ssm_pool,
                    generation_state_indices,
                    output=gen_output,
                )
            if output is None:
                cores.append(core)
        if output is not None:
            return output
        return cores[0] if len(cores) == 1 else torch.cat(cores, dim=0)

    def _sync_kda_replay_conv_window(self, layer_cache, slot_indices, conv_pool) -> None:
        """Seed the fused verify's replay convolution windows from the live pool after it advanced."""
        if getattr(layer_cache, "has_kda_replay_caches", False):
            copy_kda_replay_conv_window(
                conv_pool,
                layer_cache.kda_conv_q,
                layer_cache.kda_conv_k,
                layer_cache.kda_conv_v,
                slot_indices,
            )

    def _store_core(self, core: torch.Tensor, output: Optional[torch.Tensor]) -> torch.Tensor:
        """The [tokens, H, V] core, copied into ``output`` when given."""
        core = core.reshape(-1, self.num_heads, self.head_dim)
        if output is None:
            return core
        output.copy_(core)
        return output

    def forward_prefill(
        self,
        x2d,
        cu_seqlens,
        mamba_metadata,
        num_prefills,
        conv_pool,
        ssm_pool,
        slot_indices,
        layer_cache=None,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Prefill: packed convolution, BT16 chunked delta rule on the indexed state pool, gated output norm."""
        assert self._qkvg_proj_weight is not None, "KDA forward before finalize_decode_weights"
        has_init = mamba_metadata.has_initial_states[:num_prefills]
        if x2d.shape[0] == 0:
            reset_recurrent_state_rows(ssm_pool, slot_indices, has_init, conv_pool)
            self._sync_kda_replay_conv_window(layer_cache, slot_indices, conv_pool)
            return self._store_core(x2d.new_empty((0, self.num_heads, self.head_dim)), output)
        x = x2d.unsqueeze(0)  # [1, T, hidden]
        # Metadata owns stable int32 indices; never convert per layer.
        slot_indices = mamba_metadata.state_indices[:num_prefills]
        packed_conv, onorm_g = self._prefill_qkvg(x, x2d)
        # Fresh requests start from zero recurrent state; the packed convolution clears their conv rows itself.
        reset_recurrent_state_rows(ssm_pool, slot_indices, has_init)
        q, k, v = kda_prefill_conv(
            packed_conv,
            self._packed_conv_weight,
            conv_pool,
            mamba_metadata.query_start_loc[: num_prefills + 1],
            slot_indices,
            has_init,
            mamba_metadata.kda_max_seqlen,
        )
        beta, g = self._project_gate_inputs(x)
        o = self._prefill_kernel(
            q,
            k,
            v,
            g.unflatten(-1, (self.num_heads, self.head_dim)),
            beta.to(torch.bfloat16),
            ssm_pool,
            slot_indices,
            cu_seqlens,
            self._A_log_f32,
            self._dt_bias_f32,
            self.head_k_dim**-0.5,
            self.gate_lower_bound,
            "bt16",
            mamba_metadata.kda_max_seqlen,
            mamba_metadata.kda_seq_order,
            mamba_metadata.kda_cu_chunks,
        )
        # Fused-verify replay caches: seed the committed conv window so the first verify round convolves the
        # correct history (pending drafts are zero for a fresh request, so the tail columns are unused).
        self._sync_kda_replay_conv_window(layer_cache, slot_indices, conv_pool)
        core = rms_norm_gated_token_major(
            o.reshape(-1, self.head_dim),
            onorm_g.reshape(-1, self.num_heads, self.head_dim),
            self.o_norm.weight,
            self.o_norm.variance_epsilon,
            gate_activation="sigmoid",
        )
        return self._store_core(core, output)

    def forward_decode(
        self,
        x2d,
        conv_pool,
        ssm_pool,
        slot_indices,
        mamba_metadata,
        layer_cache,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One-token decode: fused qkvg / [f_a | b] projections, then ``kda_decode`` on the live pools by slot."""
        d, hd, H, B = self.proj_size, self.head_dim, self.num_heads, x2d.shape[0]
        assert ssm_pool.dtype == torch.float32, "the Kimi K3.1 KDA recurrent state pool is FP32"
        assert self._qkvg_proj_weight is not None, "KDA forward before finalize_decode_weights"
        # kda_decode writes in place: BCG supplies its graph-owned core buffer; eager decode uses a persistent
        # buffer whose pointer stays stable across CUDA graph captures (the eager warmup step allocates it).
        if output is not None:
            kda_out = output.view(B, 1, H, hd)
        else:
            if self._o_dense is None:
                assert not torch.cuda.is_current_stream_capturing(), (
                    "KDA decode captured before an eager decode allocated its output buffer"
                )
                self._o_dense = torch.empty(
                    max(conv_pool.shape[0], B), 1, H, hd, dtype=torch.bfloat16, device=x2d.device
                )
            assert self._o_dense.shape[0] >= B, (
                f"KDA decode output buffer holds {self._o_dense.shape[0]} rows but the decode batch is {B}; "
                f"reallocating would corrupt previously captured CUDA graphs"
            )
            kda_out = self._o_dense[:B]
        if self._kda_zero_bias is None:
            # The layer has no q / k / v convolution bias; the kernel reads a zero one.
            self._kda_zero_bias = torch.zeros(H * hd, dtype=torch.bfloat16, device=x2d.device)

        projection_aux_stream = (
            self._projection_aux_stream if B <= KDA_BFA_MULTISTREAM_MAX_ROWS else None
        )
        qkvg, (beta, g) = maybe_execute_in_parallel(
            lambda: self._decode_qkvg(x2d),
            lambda: self._project_gate_inputs(x2d),
            self._projection_fork_event,
            self._projection_join_event,
            projection_aux_stream,
            disable_on_compile=True,
        )
        onorm_g = qkvg[:, 3 * d : 4 * d]
        beta = beta.unsqueeze(0)
        # Section views keep the live pool's slot stride (V2 manager padding included); the kernel addresses
        # the conv and state pools with the same slot indices.
        kda_decode(
            _kda_decode_rows(qkvg[:, :d].unflatten(-1, (H, hd)).unsqueeze(0)),
            _kda_decode_rows(qkvg[:, d : 2 * d].unflatten(-1, (H, hd)).unsqueeze(0)),
            _kda_decode_rows(qkvg[:, 2 * d : 3 * d].unflatten(-1, (H, hd)).unsqueeze(0)),
            self._w_q_t,
            self._w_k_t,
            self._w_v_t,
            self._kda_zero_bias,
            self._kda_zero_bias,
            self._kda_zero_bias,
            conv_pool[:, :d],
            conv_pool[:, d : 2 * d],
            conv_pool[:, 2 * d :],
            self._A_log_f32,
            _kda_decode_rows(g.unflatten(-1, (H, hd)).unsqueeze(0)),
            self._dt_bias_f32,
            beta if beta.stride(2) == 1 else beta.contiguous(),
            _kda_decode_rows(onorm_g.unflatten(-1, (H, hd)).unsqueeze(0)),
            self._onorm_w_f32,
            slot_indices,
            ssm_pool,
            float(self.gate_lower_bound),
            float(hd**-0.5),
            float(self.o_norm.variance_epsilon),
            kda_out,
        )
        # Fused-verify replay caches (spec decoding only): keep the committed conv window in sync.
        self._sync_kda_replay_conv_window(layer_cache, slot_indices, conv_pool)
        return kda_out.view(B, H, hd)

    def _verify_rows(self, x2d: torch.Tensor, num_steps: int, layer_cache):
        """``(requests, [requests, steps, hidden] view, total rows)`` of a verify, after checking its preconditions."""
        assert getattr(layer_cache, "has_kda_replay_caches", False), (
            "Kimi K3.1 KDA verify needs the cache manager's replay caches"
        )
        assert self._qkvg_proj_weight is not None, "KDA forward before finalize_decode_weights"
        num_generations = x2d.shape[0] // num_steps
        return (
            num_generations,
            x2d.view(num_generations, num_steps, -1),
            num_generations * num_steps,
        )

    def _verify_kwargs(
        self, projections, T_total: int, num_steps: int, layer_cache, ssm_pool, slot_indices
    ) -> dict:
        """The verify kernel's arguments (``kda_mtp_verify`` / the targets' variants) for the verify projections."""
        q_proj, k_proj, v_proj, forget_gate, beta_proj, onorm_g = projections
        H, K = self.num_heads, self.head_k_dim
        w_q, w_k, w_v = self._mtp_conv_weights
        return dict(
            x_q=q_proj.view(1, T_total, H, K),
            x_k=k_proj.view(1, T_total, H, K),
            x_v=v_proj.view(1, T_total, H, self.head_dim),
            w_q=w_q,
            w_k=w_k,
            w_v=w_v,
            cs_q=layer_cache.kda_conv_q,
            cs_k=layer_cache.kda_conv_k,
            cs_v=layer_cache.kda_conv_v,
            # Raw gate / beta: the kernel applies dt_bias, A_log, the lower-bound sigmoid gate and the beta sigmoid.
            g=forget_gate.view(1, T_total, H, K),
            beta=beta_proj.view(1, T_total, H),
            # .detach(): the CuTe DSL DLPack bridge rejects grad-tracking tensors.
            A_log=self.A_log.detach(),
            dt_bias=self.dt_bias.detach(),
            recurrent_state=ssm_pool,
            k_cache=layer_cache.kda_k_cache,
            g_cache=layer_cache.kda_g_cache,
            v_cache=layer_cache.kda_v_cache,
            beta_cache=layer_cache.kda_beta_cache,
            ssm_state_indices=slot_indices,
            num_spec=num_steps - 1,
            num_accepted_tokens=layer_cache.prev_num_accepted_tokens,
            lower_bound=self.gate_lower_bound,
            scale=self.head_k_dim**-0.5,
            onorm_g=onorm_g.view(1, T_total, H, self.head_dim),
            onorm_weight=self._onorm_w_f32,
            onorm_eps=self.o_norm.variance_epsilon,
        )


# ---------------------------------------------------------------------------
# K31KDALinearAttention: the KDA layer on this layout's kernels.
# ---------------------------------------------------------------------------

# Recurrent-verify batches of up to this many requests take the fused [f_a | b] -> f_b gate kernel (GB300 layer
# chain with cuBLAS qkvg: a win at 1-2 requests, a loss at 4).
_REC_FAB_MAX_REQUESTS = 2
# Attention TP sizes whose gate kernel launches without programmatic dependent launch (same bits). Launched with it,
# the kernel's CTAs become resident before the attention-residual mix ends and take SMs the [q | k | v | g] TGV GEMM's
# clusters need, which delays part of that GEMM. TEP8, GB300 layer chain behind a 5 us producer, the gate kernel's
# cost beside the GEMM in us per layer at 5 / 8 / 10 rows: +1.28 / +1.21-1.27 / +1.22-1.36 with PDL, +0.54 / +0.63 /
# +0.54-0.59 without.
_REC_FAB_NO_PDL_TP = (8,)


def _k31_decode_plan(mapping) -> Optional[dict]:
    """The measured low-M GEMM plan of this generation layout (``ops/gemm/low_m_plans``); None without a mapping."""
    if mapping is None:
        return None
    return k31_low_m_plan(mapping.tp_size, mapping.enable_attention_dp)


def _k31_shared_expert_plan(mapping) -> Optional[dict]:
    """The shared expert's low-M GEMM plan for this generation layout (``ops/gemm/low_m_plans``); None without a
    mapping."""
    if mapping is None:
        return None
    return k31_low_m_shared_expert_plan(mapping.tp_size, mapping.enable_attention_dp)


def _k31_draft_plan(mapping) -> Optional[dict]:
    """The DFlash2 drafter's low-M GEMM plan for this generation layout (``ops/gemm/low_m_plans``); None without a
    mapping."""
    if mapping is None:
        return None
    return k31_low_m_draft_plan(mapping.tp_size, mapping.enable_attention_dp)


def _k31_planned_linear(x: torch.Tensor, weight: torch.Tensor, plan) -> Optional[torch.Tensor]:
    """``x @ weight.T`` on the layout's low-M plan where it names the shape and row count, else None."""
    if plan is None or low_m_choice(plan, x, weight) is None:
        return None
    return k31_low_m_linear(x, weight, plan, True)


# Verify projections, [q | k | v | g] GEMM (main stream) beside the gate pair [f_a | b] -> f_b (aux stream), by the
# verify's row count (GB300 layer chain, cold weights, CUDA graph, us per layer, TEP16 low-M plan). In the row ranges
# below the gate pair is issued before the GEMM: issued after it, the pair waits for SMs the GEMM's CTAs hold (33-48
# rows: -4.5 to -5.1; 65-130 rows: up to -4.2; TEP32 8 rows and TEP16 10 rows: -0.35 to -0.5); elsewhere the GEMM goes
# first (within 0.2). The two run on two streams at every row count (above 128 rows: -3.5 against running them one
# after the other).
_K31_GATES_FIRST_ROWS = ((1, 10), (33, 48), (65, None))


def _k31_gates_first(num_rows: int) -> bool:
    """Whether the verify's gate pair is issued before its [q | k | v | g] GEMM (``_K31_GATES_FIRST_ROWS``)."""
    return any(
        lo <= num_rows and (hi is None or num_rows <= hi) for lo, hi in _K31_GATES_FIRST_ROWS
    )


def _k31_verify_projection_pair(
    qkvg_fn, gates_fn, fork_event, join_event, aux_stream, num_rows: int
):
    """``(qkvg_fn(), gates_fn())``: the GEMM on the current stream and the gate pair on ``aux_stream``, issued in the
    measured order for ``num_rows`` (``maybe_execute_in_parallel`` issues the GEMM first); one stream outside
    multi-stream execution or under torch.compile."""
    if not _k31_gates_first(num_rows) or not do_multi_stream() or torch.compiler.is_compiling():
        return maybe_execute_in_parallel(
            qkvg_fn, gates_fn, fork_event, join_event, aux_stream, disable_on_compile=True
        )
    fork_event.record()
    with torch.cuda.stream(aux_stream):
        fork_event.wait()
        gates = gates_fn()
        join_event.record()
    qkvg = qkvg_fn()
    join_event.wait()
    if not torch.cuda.is_current_stream_capturing():
        # The gate outputs were allocated in aux-stream order and are read on this stream.
        current = torch.cuda.current_stream()
        for t in gates:
            if t is not None:
                t.record_stream(current)
    return qkvg, gates


class _K31PendingOproj(NamedTuple):
    """An attention layer's (KDA or MLA) o_proj input and the all-reduce its push chain runs into, handed to the
    decoder layer, whose attention-residual mix finishes the all-reduce (``k31_oproj_push_attn_res_add_rmsnorm``)."""

    input: torch.Tensor
    weight: torch.Tensor
    all_reduce: nn.Module


class K31KDALinearAttention(K31KdaLayer):
    """The KDA layer (``K31KdaLayer``) on the attention-TP layouts' kernels.

    The decode / verify [q | k | v | g] GEMMs and o_proj run on the layout's low-M plan where it plans them, o_proj and
    its all-reduce as one push chain where it serves; the [f_a | b] GEMM on the plan where it names the row count; the
    prefill through the segment-parallel chain (``k31_kda_prefill_segments``); the DFlash2 verify through the FP32
    recurrent verify ``k31_kda_verify_rec`` (PDL launch) where it serves, with the fused [f_a | b] -> f_b gate kernel
    for verify batches of up to ``_REC_FAB_MAX_REQUESTS`` requests.
    """

    # Set by the decoder layer around a call whose output goes straight into its residual-add mix: where the push
    # chain serves, the layer returns a ``_K31PendingOproj`` instead of the reduced output.
    _k31_defer_oproj = False

    def __init__(
        self,
        cfg,
        layer_idx: int,
        mapping=None,
        allreduce_strategy=None,
        aux_stream: Optional[torch.cuda.Stream] = None,
        extra_attrs: Optional[dict] = None,
    ) -> None:
        super().__init__(
            cfg,
            layer_idx,
            mapping=mapping,
            allreduce_strategy=allreduce_strategy,
            aux_stream=aux_stream,
            extra_attrs=extra_attrs,
        )
        # Decode / verify [q | k | v | g] projection on the low-M plan of the layout.
        self._k31_low_m_plan = _k31_decode_plan(mapping)
        # Set around the projections of a recurrent verify of up to _REC_FAB_MAX_REQUESTS requests: beta and the
        # forget gate then come from the fused [f_a | b] -> f_b kernel.
        self._fab_active = False
        self._fab_pdl = mapping is None or mapping.tp_size not in _REC_FAB_NO_PDL_TP

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        reduce_output: bool = True,
    ) -> torch.Tensor:
        """The family layer's forward; ``reduce_output=False`` (sequence-parallel prefill, eager) returns this rank's
        o_proj partials unreduced."""
        if not reduce_output:
            core = self._forward_impl(hidden_states, attn_metadata)
            return bf16_linear(core.reshape(-1, self.proj_size), self.o_proj.weight, False)
        return super().forward(hidden_states, attn_metadata)

    def _project_output(self, core: torch.Tensor):
        """o_proj and the all-reduce as one push chain where it serves (MNNVL all-reduce, a K the push GEMM has
        tactics for); else o_proj on the layout's low-M plan where it plans it, then the all-reduce; else the family
        layer's."""
        x = core.reshape(-1, self.proj_size)
        if self._o_allreduce is not None and push_serves(x, self.o_proj.weight, self._o_allreduce):
            if self._k31_defer_oproj:
                return _K31PendingOproj(x, self.o_proj.weight, self._o_allreduce)
            return oproj_allreduce_push(x, self.o_proj.weight, self._o_allreduce)
        if (
            self._k31_low_m_plan is not None
            and low_m_choice(self._k31_low_m_plan, x, self.o_proj.weight) is not None
        ):
            out = k31_low_m_linear(x, self.o_proj.weight, self._k31_low_m_plan, True)
            if self._o_allreduce is not None:
                out = allreduce_module(out, self._o_allreduce, FUSION_NONE, None, None)
            return out
        return super()._project_output(core)

    def _decode_qkvg(self, x2d: torch.Tensor) -> torch.Tensor:
        return k31_low_m_linear(x2d, self._qkvg_proj_weight, self._k31_low_m_plan, True)

    def _project_gate_inputs(self, x: torch.Tensor):
        """``(beta, forget_gate)``: with ``_fab_active`` the fused ``[f_a | b] -> f_b`` kernel where it serves; else
        the [f_a | b] GEMM on the layout's low-M plan where it names the row count, else the family layer's."""
        hd, H = self.head_dim, self.num_heads
        if self._fab_active:
            x2 = x.reshape(-1, x.shape[-1])
            fb_weight = self.f_b_proj.weight
            if x2.is_contiguous() and can_use_k3_kda_fab(x2, self._bfa_proj_weight, fb_weight, H):
                beta, forget_gate = k31_kda_fab_proj(
                    x2, self._bfa_proj_weight, fb_weight, H, self._fab_pdl
                )
                return (
                    beta.view(*x.shape[:-1], H),
                    forget_gate.view(*x.shape[:-1], fb_weight.shape[0]),
                )
        bfa = _k31_planned_linear(
            x.reshape(-1, x.shape[-1]), self._bfa_proj_weight, self._k31_low_m_plan
        )
        if bfa is None:
            return super()._project_gate_inputs(x)
        bfa = bfa.view(*x.shape[:-1], bfa.shape[-1])
        return bfa[..., hd : hd + H], bf16_linear(bfa[..., :hd], self.f_b_proj.weight, False)

    def _project_verify_inputs(self, x: torch.Tensor, num_rows: int):
        """The verify's [q | k | v | g] GEMM on the layout's low-M plan and the gate pair on two streams, in the order
        measured for ``num_rows`` (``_k31_verify_projection_pair``)."""
        qkvg, (beta, forget_gate) = _k31_verify_projection_pair(
            lambda: k31_planned_projection(x, self._qkvg_proj_weight, self._k31_low_m_plan, True),
            lambda: self._project_gate_inputs(x),
            self._projection_fork_event,
            self._projection_join_event,
            self._projection_aux_stream,
            num_rows,
        )
        return self._split_verify_qkvg(qkvg, forget_gate, beta)

    def _prefill_kernel(self, *args) -> torch.Tensor:
        # One long sequence on few heads (TEP8 context) runs the segment-parallel chain; else kda_prefill.
        return k31_kda_prefill_segments(*args)

    def forward_verify(
        self,
        x2d,
        num_steps,
        layer_cache,
        ssm_pool,
        slot_indices,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """DFlash2 verify through the FP32 recurrent verify ``k31_kda_verify_rec`` where it serves (into ``output``
        in place when contiguous), else ``kda_mtp_verify``; both with a PDL launch."""
        num_generations, x, T_total = self._verify_rows(x2d, num_steps, layer_cache)
        rec = not torch.compiler.is_compiling() and kda_verify_rec_supported(
            ssm_pool, num_steps - 1, self.head_k_dim, self.head_dim, self.conv_size
        )
        self._fab_active = rec and num_generations <= _REC_FAB_MAX_REQUESTS
        try:
            projections = self._project_verify_inputs(x, T_total)
        finally:
            self._fab_active = False
        kwargs = self._verify_kwargs(
            projections, T_total, num_steps, layer_cache, ssm_pool, slot_indices
        )
        if not rec:
            return self._store_core(kda_mtp_verify(**kwargs, use_pdl=True), output)
        in_place = output is not None and output.is_contiguous()
        out = k31_kda_verify_rec(
            **kwargs,
            out=(
                output.view(1, T_total, self.num_heads, self.head_dim)
                if in_place
                else x2d.new_empty(
                    (1, T_total, self.num_heads, self.head_dim), dtype=torch.bfloat16
                )
            ),
        )
        if in_place:
            return output
        return self._store_core(out, output)


# ====================================================================================================================
# MLA attention (DSA sparse MLA through the engine attention backend)
# ====================================================================================================================


#: Query heads of one ShardQ sparse MLA call: all of Kimi K3.1's MLA heads (the PrimTS kernel's head count).
_SHARDQ_HEADS = 96


#: Decode calls of up to this many rows (the low-latency and middle batches of the TEP layouts) run the MLA query
#: chain on the auxiliary stream beside the DSA indexer.
_QUERY_CHAIN_BESIDE_INDEXER_MAX_ROWS = 25


def _shardq_plan_rows(rows: int) -> int:
    """The PrimTS plan capacity of a ShardQ call: the next power of two, so the number of plans (and kernel
    variants) stays bounded however the chunk sizes vary, and at least 32 rows, since the PrimTS package carries
    only the 2CTA kernel without a KV split, which its schedule policy picks for power-of-two plans of 32 rows or
    more (a plan serves calls with fewer rows)."""
    return max(32, 1 << (rows - 1).bit_length())


class K31MLA(nn.Module):
    """Kimi K3.1 MLA attention with DSA: the fused q_a / kv_a projection and its norms, the query up projection and
    absorption, the DSA indexer's top-2048 key selection and the sparse MLA attention (``mqa``, a
    ``K31MLAAttention``) over the FP8 latent cache, the V absorption and the output projection. Every MLA layer of
    K3.1 runs DSA; :class:`K31GatedMLA` adds K3.1's checkpoint layout and the output gate."""

    # Set by the decoder layer around a call whose output goes straight into its residual-add mix: where the push
    # chain serves, the output projection returns a ``_K31PendingOproj`` instead of the reduced output.
    _k31_defer_oproj = False
    #: The q_b deferral the DSA split ops read (``DeferredQB``; None: Op 1 projects q).
    dsa_deferred_q_b: Optional[DeferredQB] = None

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        predicted_tokens_per_seq: int,
        max_position_embeddings: int,
        aux_stream_dict: Optional[Dict[AuxStreamType, torch.cuda.Stream]] = None,
        pos_embd_params: Optional[PositionalEmbeddingParams] = None,
        layer_idx: Optional[int] = None,
        dtype: torch.dtype = None,
        config: Optional[ModelConfig] = None,
        reduce_output: bool = True,
        rms_norm_eps: Optional[float] = None,
    ) -> None:
        """
        Initialize the MLA module.

        Args:
            hidden_size (int): The size of the hidden dimension.
            num_attention_heads (int): The number of attention heads.
            num_key_value_heads (int): The number of key value heads.
            qk_nope_head_dim (int): The dimension of the query and key without Rope.
            qk_rope_head_dim (int): The dimension of the Rope of query and key.
            v_head_dim (int): The dimension of the value.
            q_lora_rank (int): The dimension of the compressed query.
            kv_lora_rank (int): The dimension of the compressed key and value.
            predicted_tokens_per_seq (int): The number of predicted tokens per sequence.
            max_position_embeddings (int): The maximum position embeddings.
            aux_stream_dict (Optional[Dict[AuxStreamType, torch.cuda.Stream]]): Auxiliary
                CUDA streams by role.
            pos_embd_params (PositionalEmbeddingParams): The positional embedding parameters.
            layer_idx (int): The layer index.
            dtype (torch.dtype): The data type.
            config (ModelConfig): The model configuration.
            rms_norm_eps (Optional[float]): Override the RMSNorm epsilon from
                the pretrained config. If neither source provides a value
                (e.g. config.pretrained_config is None), falls back to 1e-6.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_idx_str = str(layer_idx)
        self.dtype = dtype
        self._weights_transformed = False

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.predicted_tokens_per_seq = predicted_tokens_per_seq
        self.max_position_embeddings = max_position_embeddings
        self.pos_embd_params = pos_embd_params

        assert pos_embd_params is not None, "pos_embd_params must be provided in MLA"

        # Zero the output rows past the live batch (prefill-graph padding) after the attention
        # core; off by default, a model whose padding rows would otherwise read stale memory
        # downstream turns it on.
        self.zero_padding_output_rows = False

        if config is not None:
            if "mla_layers" not in config.extra_attrs:
                config.extra_attrs["mla_layers"] = {}
            suffix = 0
            # ``layer_idx`` is local to an attention stack, while this registry is shared
            # by target and draft modules in one-model speculative decoding. Keep the first
            # registration under ``"<layer_idx>"`` and suffix later collisions as
            # ``"<layer_idx>_<n>"`` so custom ops resolve the originating module without
            # overwriting another stack's weak reference.
            while self.layer_idx_str in config.extra_attrs["mla_layers"]:
                self.layer_idx_str = str(layer_idx) + f"_{suffix}"
                suffix += 1
            config.extra_attrs["mla_layers"][self.layer_idx_str] = weakref.ref(self)

        config = config or ModelConfig()
        self.kv_cache_dtype = config.extra_attrs.get("kv_cache_dtype", "auto")
        sparse_attn_cfg = config.sparse_attention_config
        sparse_params = k31_dsa_params(
            sparse_attn_cfg.to_sparse_params(
                pretrained_config=config.pretrained_config,
                layer_idx=self.layer_idx,
            )
            if sparse_attn_cfg is not None
            else None,
            config.pretrained_config,
        )
        self.sparse_params = sparse_params
        if not isinstance(sparse_params, K31DSAParams):
            raise NotImplementedError("the Kimi K3.1 MLA serves DSA sparse attention only")

        # Tensor parallelism (routing refuses context parallelism): the heads split over attention TP.
        self.mapping = config.mapping
        tp_size = self.mapping.tp_size
        pp_size = self.mapping.pp_size

        mapping = Mapping(
            world_size=pp_size * tp_size,
            tp_size=tp_size,
            pp_size=pp_size,
            cp_config=self.mapping.cp_config,
            rank=self.mapping.rank,
            gpus_per_node=self.mapping.gpus_per_node,
            enable_attention_dp=self.mapping.enable_attention_dp,
        )

        assert self.num_heads % tp_size == 0
        self.num_heads_tp = self.num_heads // tp_size

        if rms_norm_eps is None:
            rms_norm_eps = getattr(config.pretrained_config, "rms_norm_eps", None)
        if rms_norm_eps is None:
            # No explicit value and no rms_norm_eps on the config (a unit test that builds the MLA directly).
            rms_norm_eps = 1e-6
        quant_config = config.get_quant_config()
        self.quant_config = quant_config
        self.num_hidden_layers = getattr(config.pretrained_config, "num_hidden_layers", None)
        self.pp_mapping = config.mapping

        # The fused [q_a | kv_a | k_pe] projection runs the cuBLAS GEMM (``_project_qkv_a``).
        self.kv_a_proj_with_mqa = K31Linear(
            hidden_size,
            self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
            dtype=dtype,
            use_custom_cublas_mm=True,
        )
        self.q_a_layernorm = K31RMSNorm(self.q_lora_rank, eps=rms_norm_eps, dtype=dtype)
        self.q_b_proj = K31Linear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            dtype=dtype,
            mapping=mapping,
            tp_mode="column",
        )

        mapping_o = Mapping(
            world_size=pp_size * tp_size,
            tp_size=tp_size,
            pp_size=pp_size,
            cp_size=1,
            rank=self.mapping.rank,
            gpus_per_node=self.mapping.gpus_per_node,
            enable_attention_dp=self.mapping.enable_attention_dp,
        )

        def yarn_get_mscale(scale=1, mscale=1):
            if scale <= 1:
                return 1.0
            return 0.1 * mscale * math.log(scale) + 1.0

        mscale_all_dim = pos_embd_params.rope.mscale_all_dim
        scaling_factor = pos_embd_params.rope.scale
        mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
        q_scaling = 1.0 / (mscale * mscale)

        self.aux_stream_dict = aux_stream_dict or {}
        self.aux_stream = self.aux_stream_dict.get(AuxStreamType.Attention)
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]
        self.kv_a_layernorm = K31RMSNorm(self.kv_lora_rank, eps=rms_norm_eps, dtype=dtype)
        self.kv_b_proj = K31Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            dtype=dtype,
            mapping=mapping,
            tp_mode="column",
        )
        self.v_b_proj = nn.Parameter(
            torch.empty(
                (self.num_heads_tp, self.v_head_dim, self.kv_lora_rank),
                dtype=dtype,
            ),
            requires_grad=False,
        )
        self.mqa = K31MLAAttention(
            config.attn_backend,
            self.layer_idx,
            self.num_heads_tp,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            hidden_size=self.hidden_size,
            predicted_tokens_per_seq=self.predicted_tokens_per_seq,
            q_scaling=q_scaling,
            pos_embd_params=self.pos_embd_params,
            quant_config=quant_config,
            sparse_params=self.sparse_params,
            dtype=dtype,
            aux_stream=self.aux_stream,
            kv_cache_dtype=self.kv_cache_dtype,
            skip_correction_threshold=config.skip_correction_threshold,
            perf=True,
        )

        initialize_sparse_attn(self)
        self.o_proj = K31Linear(
            self.num_key_value_heads * self.v_head_dim,
            self.hidden_size,
            dtype=dtype,
            mapping=mapping_o,
            tp_mode="row",
            reduce_output=reduce_output,
            allreduce_strategy=config.allreduce_strategy,
        )
        self.attention_output_hidden_size = self.o_proj.in_features

        self.softmax_scale = 1.0 / (math.sqrt(self.qk_head_dim) * q_scaling)

        if not config.skip_create_weights_in_init:
            self.create_weights()

    def create_weights(self):
        # self.mqa has no weights but has states that are related to
        # quant_config, which could be modified after __init__.
        self.mqa.update_quant_config(self.quant_config)

        # The attention projections are BF16 (routing refuses quantized ones), so the K
        # absorption weight is the BF16 nope slice of kv_b_proj, filled by the weight loader.
        self.k_b_proj_trans = nn.Parameter(
            torch.empty(
                (self.num_heads_tp, self.kv_lora_rank, self.qk_nope_head_dim),
                dtype=self.dtype,
            ),
            requires_grad=False,
        )
        self._weights_transformed = False

    def _project_qkv_a(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The fused q_a / kv_a projection of ``hidden_states``: ``kv_a_proj_with_mqa``'s cuBLAS GEMM (the loader
        packs the checkpoint's q_a / kv_a into it)."""
        return self.kv_a_proj_with_mqa(hidden_states)

    def _kv_a_layernorm(self, compressed_kv: torch.Tensor) -> torch.Tensor:
        """``kv_a_layernorm`` of the BF16 latent (``flashinfer_rmsnorm``)."""
        norm = self.kv_a_layernorm
        return flashinfer_rmsnorm(compressed_kv, norm.weight, norm.variance_epsilon)

    def create_output(self, hidden_states: torch.Tensor, num_contexts: int):
        num_tokens = hidden_states.shape[0]
        return hidden_states.new_empty(
            [num_tokens, self.attention_output_hidden_size], dtype=hidden_states.dtype
        )

    def _create_outputs(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> list[torch.Tensor]:
        """The MLA output buffer (``o_proj``'s input)."""
        return [self.create_output(hidden_states, attn_metadata.num_contexts)]

    def _q_a_layernorm(self, q: torch.Tensor):
        """``(q_b input, indexer input)``: q_a_layernorm of the BF16 q_a (``flashinfer_rmsnorm``; q_b_proj is BF16,
        so no input quantization is fused into the norm), the same tensor for both."""
        norm = self.q_a_layernorm
        normed = flashinfer_rmsnorm(q, norm.weight, norm.variance_epsilon)
        return normed, normed

    def _new_fused_q(self, q: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """``[num_tokens, num_heads_tp, kv_lora_rank + qk_rope_head_dim]``."""
        return torch.empty(
            [num_tokens, self.num_heads_tp, self.kv_lora_rank + self.qk_rope_head_dim],
            dtype=q.dtype,
            device=q.device,
        )

    def _attention_heads_out(self, attn_out_latent: torch.Tensor) -> torch.Tensor:
        """``[seq, num_heads_tp, kv_lora_rank]`` view of the backend output."""
        assert attn_out_latent.shape[1] == self.num_heads_tp * self.kv_lora_rank
        return attn_out_latent.view([-1, self.num_heads_tp, self.kv_lora_rank])

    def _rope_generation(
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

        The arguments the TRTLLM backend's ``mla_rope_generation`` passes on this target's path: an FP8
        latent cache (no FP4 MLA), no folded KV norm, no precomputed scheduler, both halves in one launch.
        """
        mqa = self.mqa
        assert metadata.kv_cache_manager is not None
        # The kernel reads the RoPE cos/sin table up to the longest sequence.
        mqa.ensure_rope_table_size(metadata.max_seq_len)
        helix_tensor_params = [metadata.helix_position_offsets, metadata.helix_is_inactive_rank]
        if metadata._helix_spec_tokens_valid:
            helix_tensor_params.append(metadata.helix_local_slots)
        kv_cache_manager = metadata.kv_cache_manager
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
            None,
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

    def _bmm_bf16_out(self, a, b_no_transpose, b_transposed, output):
        """BF16 absorption BMM (cuBLAS): ``output = a @ b_transposed`` per head."""
        bmm_out(a, b_transposed, output)

    def forward_absorption_generation(
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
        if self._query_chain_beside_indexer(attn_metadata, num_tokens):
            return self._forward_generation_beside_indexer(
                q, attn_metadata, output, position_ids, latent_cache, kwargs
            )
        assert q.shape[-1] == self.num_heads_tp * self.qk_head_dim, (
            "q_b was deferred to a path that skips it"
        )
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
        assert getattr(self.mqa, "has_fp8_kv_cache", False), "the Kimi K3.1 latent KV cache is FP8"

        mla_bmm1_scale = torch.empty(2, dtype=torch.float32, device=q.device)
        mla_bmm2_scale = torch.empty(1, dtype=torch.float32, device=q.device)
        quant_q_buffer = torch.empty(
            num_tokens,
            self.num_heads_tp,
            (self.kv_lora_rank + self.qk_rope_head_dim),
            dtype=torch.uint8,
            device=q.device,
        )

        if hasattr(self, "k_b_proj_trans"):
            fused_q = self._new_fused_q(q, num_tokens)

            def _mla_gen_rope():
                # RoPE, latent-cache append and the trtllm-gen scheduler buffers are all produced in-kernel.
                self._rope_generation(
                    fused_q,
                    q_pe,
                    latent_cache,
                    attn_metadata,
                    cu_q_seqlens,
                    cu_kv_seqlens,
                    fmha_scheduler_counter,
                    mla_bmm1_scale,
                    mla_bmm2_scale,
                    quant_q_buffer,
                )

            if self.k_b_proj_trans.dtype == torch.bfloat16:
                # [num_heads, num_tokens, self.qk_nope_head_dim]
                q_nope_t = q_nope.transpose(0, 1)
                # [num_heads, num_tokens, self.kv_lora_rank]
                q_nope_out = fused_q[:, : self.num_heads_tp, : self.kv_lora_rank].transpose(0, 1)

                # [num_heads, num_tokens, self.qk_nope_head_dim] x [num_heads, kv_lora_rank, qk_nope_head_dim]
                # -> [num_heads, num_tokens, kv_lora_rank] -> [num_tokens, num_heads, kv_lora_rank]
                # The output of bmm is written directly into fused_q
                self._bmm_bf16_out(
                    q_nope_t,
                    self.k_b_proj_trans,
                    self.k_b_proj_trans.transpose(1, 2),
                    q_nope_out,
                )
                _mla_gen_rope()
            else:
                raise NotImplementedError(
                    f"Missing bmm impl for dtype: {self.k_b_proj_trans.dtype}."
                )

            fused_q = fused_q.view(
                [
                    num_tokens,
                    self.num_heads_tp * (self.kv_lora_rank + self.qk_rope_head_dim),
                ]
            )
        else:
            raise RuntimeError("MLA absorption requires k_b_proj_trans")

        attn_out_latent = self.mqa.forward(
            fused_q,
            attn_metadata,
            attention_input_type=AttentionInputType.generation_only,
            latent_cache=latent_cache,  # kvcache and k_pe
            q_pe=q_pe,  # used by `invokeMLARopeGeneration`
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
        return self._value_projection(attn_out_latent, output)

    def _value_projection(
        self, attn_out_latent: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        """The generation rows' V up-projection of the latent attention output into ``output``."""
        num_tokens = attn_out_latent.shape[0]
        # [seq, num_heads, kv_lora_rank]
        attn_out_latent = self._attention_heads_out(attn_out_latent)

        attn_output = output.view([num_tokens, self.num_heads_tp, self.v_head_dim])

        if self.v_b_proj.dtype == torch.bfloat16:
            # [num_heads, seq, kv_lora_rank] x [num_heads, kv_lora_rank, v_head_dim]
            # -> [num_heads, seq, v_head_dim]
            self._bmm_bf16_out(
                attn_out_latent.transpose(0, 1),
                self.v_b_proj,
                self.v_b_proj.transpose(1, 2),
                attn_output.transpose(0, 1),
            )
        else:
            raise NotImplementedError(f"Missing bmm impl for dtype: {self.v_b_proj.dtype}.")

        return output

    def _k31_q_b(self, q_normed: torch.Tensor) -> torch.Tensor:
        """The ``q_b`` projection Op 1 left to Op 2 (``dsa_deferred_q_b``): the same call as Op 1 makes."""
        return k31_low_m_linear(
            q_normed, self.q_b_proj.weight, getattr(self, "_k31_low_m_plan", None), True
        )

    def _k31_gen_projects_q_b(self, q: torch.Tensor, attn_metadata: AttentionMetadata) -> bool:
        """Whether the generation path runs the deferred ``q_b`` itself, beside the indexer
        (``_query_chain_beside_indexer``)."""
        return self._query_chain_beside_indexer(attn_metadata, q.shape[0])

    def _query_chain_beside_indexer(
        self, attn_metadata: AttentionMetadata, num_tokens: int
    ) -> bool:
        """Whether a generation call of ``num_tokens`` rows runs its query chain (the deferred ``q_b``, the W_UK
        absorption, mla_rope_generation) on the auxiliary stream beside the DSA indexer: a generation-only call of
        unpadded rows, at most ``_QUERY_CHAIN_BESIDE_INDEXER_MAX_ROWS`` of them, on this target's DSA backend with an
        FP8 latent pool and a BF16 absorption."""
        md = attn_metadata
        return (
            0 < num_tokens <= _QUERY_CHAIN_BESIDE_INDEXER_MAX_ROWS
            and md.num_contexts == 0
            and md.num_tokens == num_tokens
            and getattr(self.mqa, "_k31_perf", False)
            and getattr(self.mqa, "has_fp8_kv_cache", False)
            and self.k_b_proj_trans.dtype == torch.bfloat16
        )

    def _rope_generation_buffers(self, attn_metadata, num_tokens: int, device) -> SimpleNamespace:
        """mla_rope_generation's outputs besides the query: the trtllm-gen scheduler buffers, the BMM scales and the
        FP8 query, sized as ``forward_absorption_generation`` sizes them."""
        md = attn_metadata
        cu_rows = (
            max(md.num_seqs, num_tokens) + 1
            if getattr(md, "is_ragged_verify", False)
            else md.num_seqs + 1
        )
        width = self.kv_lora_rank + self.qk_rope_head_dim
        return SimpleNamespace(
            cu_q=torch.empty(cu_rows, dtype=torch.int32, device=device),
            cu_kv=torch.empty(cu_rows, dtype=torch.int32, device=device),
            counter=torch.empty(1, dtype=torch.uint32, device=device),
            bmm1=torch.empty(2, dtype=torch.float32, device=device),
            bmm2=torch.empty(1, dtype=torch.float32, device=device),
            q8=torch.empty(num_tokens, self.num_heads_tp, width, dtype=torch.uint8, device=device),
        )

    def _absorb_query(self, q: torch.Tensor, fused_q: torch.Tensor) -> torch.Tensor:
        """The query chain up to the latent query: the deferred ``q_b`` when Op 1 left it (``q`` is then the normed
        ``q_a``) and the W_UK absorption into ``fused_q``'s latent half. Returns ``q_pe``."""
        projected = self._k31_q_b(q) if q.shape[-1] != self.num_heads_tp * self.qk_head_dim else q
        q_nope, q_pe = projected.view([-1, self.num_heads_tp, self.qk_head_dim]).split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        self._bmm_bf16_out(
            q_nope.transpose(0, 1),
            self.k_b_proj_trans,
            self.k_b_proj_trans.transpose(1, 2),
            fused_q[:, : self.num_heads_tp, : self.kv_lora_rank].transpose(0, 1),
        )
        return q_pe

    def _indexer_topk(self, fused_q, attn_metadata, kwargs) -> torch.Tensor:
        """The generation rows' Top-K pool rows: the DSA backend's sparse predict (the indexer reads its own
        projections from ``kwargs``; ``fused_q`` only carries the row count)."""
        return self.mqa.sparse_attn_predict(
            fused_q, attn_metadata, True, kwargs.get("sparse_backend_args")
        )

    def _forward_generation_beside_indexer(
        self, q, attn_metadata, output, position_ids, latent_cache, kwargs
    ) -> torch.Tensor:
        """Generation rows on the TRTLLM-Gen sparse MLA with the query chain beside the DSA indexer
        (``_query_chain_beside_indexer``): this stream predicts the Top-K while the auxiliary stream runs the
        deferred ``q_b``, the W_UK absorption and mla_rope_generation; the attention call then takes that Top-K
        (``_k31_predicted_topk``) instead of predicting it. The same kernels on the same inputs as the one-stream
        path: the same bits."""
        num_tokens = q.shape[0]
        md = attn_metadata
        fused_q = self._new_fused_q(q, num_tokens)
        rope = self._rope_generation_buffers(md, num_tokens, q.device)

        def query_chain():
            q_pe = self._absorb_query(q, fused_q)
            self._rope_generation(
                fused_q, q_pe, latent_cache, md, rope.cu_q, rope.cu_kv, rope.counter, rope.bmm1, rope.bmm2, rope.q8
            )  # fmt: skip
            return q_pe

        topk, q_pe = maybe_execute_in_parallel(
            lambda: self._indexer_topk(fused_q, md, kwargs),
            query_chain,
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )
        logger.info_once(
            "Kimi K3.1 MLA: TRTLLM-Gen sparse MLA with the query chain beside the DSA indexer",
            key="k31_mla_query_chain_beside_indexer",
        )
        width = self.kv_lora_rank + self.qk_rope_head_dim
        self.mqa._k31_predicted_topk = topk
        try:
            attn_out_latent = self.mqa.forward(
                fused_q.view([num_tokens, self.num_heads_tp * width]),
                md,
                attention_input_type=AttentionInputType.generation_only,
                latent_cache=latent_cache,
                q_pe=q_pe,
                cu_q_seqlens=rope.cu_q,
                cu_kv_seqlens=rope.cu_kv,
                fmha_scheduler_counter=rope.counter,
                mla_bmm1_scale=rope.bmm1,
                mla_bmm2_scale=rope.bmm2,
                quant_q_buffer=rope.q8,
                **kwargs,
            )
        finally:
            self.mqa._k31_predicted_topk = None
        return self._value_projection(attn_out_latent, output)

    def forward_absorption_context(
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
        sparse_backend_args = kwargs.get("sparse_backend_args")
        if self._use_shardq(attn_metadata, latent_cache, sparse_backend_args):
            return self._forward_context_shardq(
                q, latent_cache, attn_metadata, output, sparse_backend_args.indexer_intermediates
            )
        num_tokens = q.shape[0]

        q_nope, q_pe = q.view([-1, self.num_heads_tp, self.qk_head_dim]).split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        if not hasattr(self, "k_b_proj_trans"):
            raise RuntimeError("MLA absorption requires k_b_proj_trans")

        # fused_q contains 1) the result of the following bmm with shape [num_tokens, num_heads, kv_lora_rank]
        # 2) rope(q_pe) with shape [num_tokens, num_heads, qk_rope_head_dim]. rope is applied inside AttentionOp
        fused_q = self._new_fused_q(q, num_tokens)

        if self.k_b_proj_trans.dtype == torch.bfloat16:
            # [num_heads, num_tokens, self.qk_nope_head_dim]
            q_nope_t = q_nope.transpose(0, 1)
            # [num_heads, num_tokens, self.kv_lora_rank]
            q_nope_out = fused_q[:, : self.num_heads_tp, : self.kv_lora_rank].transpose(0, 1)

            # [num_heads, num_tokens, self.qk_nope_head_dim] x [num_heads, kv_lora_rank, qk_nope_head_dim]
            # -> [num_heads, num_tokens, kv_lora_rank] -> [num_tokens, num_heads, kv_lora_rank]
            # The output of bmm is written directly into fused_q
            self._bmm_bf16_out(
                q_nope_t,
                self.k_b_proj_trans,
                self.k_b_proj_trans.transpose(1, 2),
                q_nope_out,
            )
        else:
            raise NotImplementedError(f"Missing bmm impl for dtype: {self.k_b_proj_trans.dtype}.")

        fused_q = fused_q.view(
            [
                num_tokens,
                self.num_heads_tp * (self.kv_lora_rank + self.qk_rope_head_dim),
            ]
        )

        attn_out_latent = self.mqa.forward(
            fused_q,
            attn_metadata,
            attention_input_type=AttentionInputType.context_only,
            latent_cache=latent_cache,  # kvcache and k_pe
            q_pe=q_pe,  # used by applyMLARopeAndAssignQKVKernelOptContext
            **kwargs,
        )
        fused_q = None

        assert attn_out_latent.shape[0] == q.shape[0]
        # [seq, num_heads, kv_lora_rank]
        attn_out_latent = self._attention_heads_out(attn_out_latent)

        attn_output = output.view([num_tokens, self.num_heads_tp, self.v_head_dim])

        if self.v_b_proj.dtype == torch.bfloat16:
            # [num_heads, seq, kv_lora_rank] x [num_heads, kv_lora_rank, v_head_dim]
            # -> [num_heads, seq, v_head_dim]
            self._bmm_bf16_out(
                attn_out_latent.transpose(0, 1),
                self.v_b_proj,
                self.v_b_proj.transpose(1, 2),
                attn_output.transpose(0, 1),
            )
        else:
            raise NotImplementedError(f"Missing bmm impl for dtype: {self.v_b_proj.dtype}.")

        return output

    # Token-sharded sparse MLA prefill ("ShardQ"). Head-sharded TP runs the context attention with num_heads / tp
    # heads per rank on every context token (12 at TEP8), a head group no sparse prefill kernel runs efficiently.
    # ShardQ splits the context query tokens over the TP ranks exactly like the DSA indexer's query split: every rank
    # scores only its own rows in the indexer (the indexer skips its top-k all-gather) and runs the PrimTS FP8 sparse
    # MLA over the FP8 latent pool for its rows with all heads. The projected queries move from head shards to token
    # shards before the absorption and the V-projected outputs back before the output gate and o_proj; each rank
    # keeps an all-head copy of the absorption and V weights, gathered once per layer. Not bitwise to the head-sharded
    # TRTLLM-Gen prefill (E4M3 query and probabilities, other accumulation order).

    def _use_shardq(self, attn_metadata, latent_cache, sparse_backend_args) -> bool:
        """Whether this context step runs the token-sharded prefill: BF16 absorption weights, an FP8 latent pool,
        and a step whose indexer queries are split over the TP group (``q_split_threshold`` context tokens or
        more)."""
        if not (
            latent_cache is not None
            and sparse_backend_args is not None
            and sparse_backend_args.indexer_intermediates
            and getattr(self.mqa, "indexer", None) is not None
            and attn_metadata.num_contexts > 0
            and getattr(attn_metadata, "indexer_prefill_q_split", None) is not None
            and attn_metadata.shared_topk_indices is None
            and not attn_metadata.is_cuda_graph
            and not torch.compiler.is_compiling()
            and self.k_b_proj_trans.dtype == torch.bfloat16
            and self.v_b_proj.dtype == torch.bfloat16
        ):
            return False
        return self._shardq_fp8_pool(attn_metadata)

    def _shardq_fp8_pool(self, attn_metadata) -> bool:
        """The latent pool is per-tensor FP8 (E4M3, D576), which the PrimTS kernel reads in place."""
        cached = getattr(self, "_shardq_pool_ok", None)
        if cached is None:
            manager = attn_metadata.kv_cache_manager
            cached = (
                manager is not None
                and getattr(manager, "head_dim", None) == self.kv_lora_rank + self.qk_rope_head_dim
                and self.mqa.has_fp8_kv_cache
            )
            self._shardq_pool_ok = cached
        return cached

    def _shardq_all_head_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """All-head copies of the absorption (``k_b_proj_trans``) and V (``v_b_proj``) weights, gathered over TP
        once."""
        weights = getattr(self, "_shardq_weights", None)
        if weights is None:
            w_uk, w_uv = sp_gather_rows_mnnvl(
                [self.k_b_proj_trans.data.contiguous(), self.v_b_proj.data.contiguous()],
                self.mapping,
            )
            weights = (w_uk.contiguous(), w_uv.contiguous())
            self._shardq_weights = weights
        return weights

    def _shardq_scales(self) -> Tuple[float, float, torch.Tensor]:
        """(softmax scale with the Q and KV descales, KV descale, device Q quantization scale); read once."""
        cached = getattr(self, "_shardq_scale_values", None)
        if cached is None:
            kv_dequant = float(self.mqa.kv_scale_quant_orig.reshape(-1)[0].item())
            q_quant = self.mqa.kv_scale_orig_quant.reshape(-1)[:1].to(torch.float32).contiguous()
            q_dequant = 1.0 / float(q_quant[0].item())
            for value in (kv_dequant, q_dequant):
                if not (math.isfinite(value) and value > 0):
                    raise RuntimeError(
                        f"ShardQ prefill needs positive finite KV scales, got {value}."
                    )
            cached = (float(self.softmax_scale) * q_dequant * kv_dequant, kv_dequant, q_quant)
            self._shardq_scale_values = cached
        return cached

    def _forward_context_shardq(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        indexer_intermediates: List[torch.Tensor],
    ) -> torch.Tensor:
        """Token-sharded sparse MLA prefill for the context rows ``q`` ``[T, H * (qk_nope + qk_rope)]`` (this rank's
        head shard); writes ``output`` ``[T, H * v_head]`` (this rank's head shard)."""
        mapping = self.mapping
        tp = mapping.tp_size
        num_tokens = q.shape[0]
        heads = self.num_heads_tp
        qk_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        v_dim = self.v_head_dim

        # Every rank appends the whole chunk's latent rows (and applies the identity RoPE to its q_pe).
        self.mqa.mla_rope_append_paged_kv_assign_q(q, latent_cache, attn_metadata)

        begin, end, sizes = attn_metadata.indexer_prefill_q_split
        bounds = [0]
        for size in sizes:
            bounds.append(bounds[-1] + size)
        rows = max(sizes)
        n = end - begin

        # Indexer top-k for this rank's rows only.
        attn_metadata.indexer_skip_q_split_gather = True
        try:
            topk = self.mqa.indexer.forward_from_projected(
                attn_metadata, q, indexer_intermediates, is_generation=False
            )
        finally:
            attn_metadata.indexer_skip_q_split_gather = False

        q_heads = q.view(num_tokens, heads, qk_dim)
        recv = rows_alltoall([q_heads[bounds[j] : bounds[j + 1]] for j in range(tp)], rows, mapping)
        q_all = recv[:, :n].transpose(0, 1).reshape(n, tp * heads, qk_dim)
        out_rows = q.new_empty(n, tp * heads, v_dim)
        w_uk, w_uv = self._shardq_all_head_weights()
        if n > 0:
            q_nope, q_pe = q_all.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            q_latent = q.new_empty(n, tp * heads, self.kv_lora_rank)
            self._bmm_bf16_out(
                q_nope.transpose(0, 1), w_uk, w_uk.transpose(1, 2), q_latent.transpose(0, 1)
            )
            softmax_scale, kv_dequant, q_quant = self._shardq_scales()
            query = torch.empty(
                n,
                tp * heads,
                self.kv_lora_rank + self.qk_rope_head_dim,
                dtype=torch.float8_e4m3fn,
                device=q.device,
            )
            k31_mla_query_e4m3(q_latent, q_pe, q_quant, query)

            attn_metadata._ensure_pool_view_cached()
            local_layer_idx = self.mqa.get_local_layer_idx(attn_metadata)
            page_scale, layer_offset = (
                attn_metadata.kv_cache_manager.get_primary_pool_page_index_params(local_layer_idx)
            )
            tpb = attn_metadata._cached_tokens_per_block
            top_k = int(topk.shape[1])
            indices = convert_req_index_to_global(
                attn_metadata._cached_req_idx_ctx[begin:end].contiguous(),
                attn_metadata._cached_block_table_ctx,
                deterministic_topk_order(topk[begin:end]).contiguous(),
                tpb,
                top_k,
                page_scale * tpb,
                layer_offset,
            ).view(n, top_k)
            pool = attn_metadata._cached_pool_view.view(torch.float8_e4m3fn).view(
                -1, self.kv_lora_rank + self.qk_rope_head_dim
            )
            # The absorbed query is dead after the quantization: its storage takes the latent output.
            k31_sparse_mla_prims_ts(
                query,
                pool,
                indices,
                full_lengths(q.device, n, top_k),
                softmax_scale,
                kv_dequant,
                q_latent,
                "native",
                _shardq_plan_rows(n),
            )
            self._bmm_bf16_out(
                q_latent.transpose(0, 1), w_uv, w_uv.transpose(1, 2), out_rows.transpose(0, 1)
            )

        out_heads = out_rows.view(n, tp, heads, v_dim).transpose(0, 1)
        back = rows_alltoall([out_heads[j] for j in range(tp)], rows, mapping)
        out_view = output.view(num_tokens, heads, v_dim)
        if all(size == rows for size in sizes):
            out_view.copy_(back.view(tp * rows, heads, v_dim))
        else:
            for j in range(tp):
                out_view[bounds[j] : bounds[j + 1]].copy_(back[j, : sizes[j]])
        return output

    def _forward_custom_op(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attn_output: list[torch.Tensor],
    ) -> None:
        """Run the DSA registered custom-op implementation."""
        forward_sparse_attn_custom_op(
            self,
            hidden_states,
            position_ids,
            attn_output,
        )

    def _project_output(
        self,
        attn_output: list[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """Apply the MLA output projection."""
        return self._project_output_impl(attn_output[0], position_ids, attn_metadata)

    def _project_output_impl(
        self,
        attn_output: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """Apply the default MLA output projection implementation.

        With ``_k31_o_allreduce`` (``K31MLARuntime``'s TP all-reduce, or None), o_proj and that all-reduce run as
        one push chain where it serves, and ``_k31_o_reduced`` tells the runtime the output is already reduced. With
        ``_k31_defer_oproj`` (set by the decoder layer around a call whose output goes straight into its residual-add
        mix) a served call returns a ``_K31PendingOproj`` instead, which that mix finishes.
        """
        (o_allreduce,) = getattr(self, "_k31_o_allreduce", (None,))
        if o_allreduce is not None and push_serves(attn_output, self.o_proj.weight, o_allreduce):
            self._k31_o_reduced = True
            if self._k31_defer_oproj:
                # The decoder layer's residual-add mix finishes the all-reduce.
                return _K31PendingOproj(attn_output, self.o_proj.weight, o_allreduce)
            return oproj_allreduce_push(attn_output, self.o_proj.weight, o_allreduce)
        o_proj = self.o_proj
        plan = getattr(self, "_k31_low_m_plan", None)
        if (
            plan is not None
            and o_proj.all_reduce is None
            and low_m_choice(plan, attn_output, o_proj.weight) is not None
        ):
            # The layout plans this o_proj: a plain GEMM; the runtime all-reduces its partials.
            return k31_low_m_linear(attn_output, o_proj.weight, plan, True)
        return self.o_proj(attn_output)

    def forward(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        attn_output = [create_mla_outputs(hidden_states, self.layer_idx_str)]
        self._forward_custom_op(
            hidden_states,
            position_ids,
            attn_output,
        )
        attn_output[0] = self._apply_output_gate(hidden_states, attn_output[0])
        return self._project_output(attn_output, position_ids, attn_metadata)

    def transform_weights(self) -> None:
        if self._weights_transformed:
            return
        self._weights_transformed = True

    def cache_derived_state(self) -> None:
        self._weights_transformed = True

    def post_load_weights(self) -> None:
        self.transform_weights()
        self.cache_derived_state()


# ====================================================================================================================
# Kimi K3.1 MLA (NoPE, output gate)
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


# ---------------------------------------------------------------------------
# K31GatedMLA.
# ---------------------------------------------------------------------------


class K31GatedMLA(K31MLA):
    """The Kimi K3.1 MLA layer: :class:`K31MLA` with K3.1's checkpoint projection layout, the identity (NoPE)
    RoPE, the KV-B runtime layout and the gated output projection. Every K3.1 MLA layer runs DSA sparse attention."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        rms_norm_eps: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        layer_idx: int = 0,
        use_output_gate: bool = True,
        max_position_embeddings: int = 8192,
        model_config: ModelConfig,
        aux_stream_dict: dict[AuxStreamType, torch.cuda.Stream],
    ) -> None:
        # The base MLA defers its weights; this class creates them once, below.
        deferred_config = copy.copy(model_config)
        deferred_config._frozen = False
        deferred_config.skip_create_weights_in_init = True
        deferred_config._frozen = model_config._frozen
        pos_embd_params = _make_pos_embd_params(
            qk_rope_head_dim=qk_rope_head_dim,
            max_position_embeddings=max_position_embeddings,
        )
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            predicted_tokens_per_seq=1,
            max_position_embeddings=max_position_embeddings,
            pos_embd_params=pos_embd_params,
            layer_idx=layer_idx,
            dtype=dtype,
            config=deferred_config,
            aux_stream_dict=aux_stream_dict,
            reduce_output=False,
            rms_norm_eps=rms_norm_eps,
        )
        # The base MLA registers the layer so breakable CUDA graphs use the
        # shared custom op. The output gate runs on its output before the
        # row-parallel o_proj.
        # Prefill-graph padding rows leave the core zero, so the gate, o_proj and
        # the MoE router after them only ever see finite rows.
        self.zero_padding_output_rows = True
        # Low-M plan of the packed DSA projections (weights.py packs them after the load).
        self._k31_low_m_plan = _k31_decode_plan(model_config.mapping)
        # Decode calls of up to _QUERY_CHAIN_BESIDE_INDEXER_MAX_ROWS rows leave q_b to Op 2
        # (ops/attention/dsa_graph_ops), whose generation path runs the query chain beside the DSA indexer
        # (_query_chain_beside_indexer).
        self.dsa_deferred_q_b = DeferredQB(
            _QUERY_CHAIN_BESIDE_INDEXER_MAX_ROWS, self._k31_q_b, self._k31_gen_projects_q_b
        )

        self.use_output_gate = use_output_gate

        if use_output_gate:
            # The gate matches o_proj's input sharding: q_b_proj's head sharding, replicated under
            # attention DP.
            self.g_proj = K31Linear(
                hidden_size,
                num_heads * v_head_dim,
                dtype=dtype,
                mapping=self.o_proj.mapping,
                tp_mode="column",
            )

        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    def _apply_output_gate(
        self,
        hidden_states: torch.Tensor,
        attn_output: torch.Tensor,
    ) -> torch.Tensor:
        # Sigmoid gate on o_proj's input. g_proj matches o_proj's input
        # sharding. The fused kernel takes a contiguous BF16 [M, N] output and a gate of its shape
        # (bit-identical to the sigmoid and product); other layouts keep the two PyTorch kernels.
        if self.use_output_gate:
            # The packed DSA projection already produced g_proj(hidden_states) as a column block.
            gate = packed_output_gate(self, hidden_states)
            if gate is None:
                gate = self.g_proj(hidden_states)
            if (
                attn_output.dtype == torch.bfloat16
                and attn_output.dim() == 2
                and attn_output.is_contiguous()
                and gate.shape == attn_output.shape
            ):
                return k31_sigmoid_gate_mul(attn_output, gate)
            return attn_output * gate.sigmoid()
        return attn_output


# ====================================================================================================================
# Text model: attention residuals, latent MoE, decoder layer, model, causal LM
# ====================================================================================================================


class K31MoEGate(nn.Module):
    """Kimi K3.1 gate weights (router and routing correction bias) of the routed experts."""

    def __init__(
        self,
        config: Any,
        *,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_token
        self.num_experts = config.num_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.gating_dim = config.hidden_size

        # The checkpoint stores the gate weight in bf16. Storing it in bf16
        # permits the single bf16xbf16 router GEMM while retaining fp32 output.
        self.weight = nn.Parameter(
            torch.empty((self.num_experts, self.gating_dim), dtype=torch.bfloat16, device=device)
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(self.num_experts, dtype=torch.float32, device=device)
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute fp32 routing logits shaped ``[num_tokens, num_experts]``."""
        hidden_2d = hidden_states.reshape(-1, self.gating_dim)
        return dsv3_router_gemm_op(hidden_2d.contiguous(), self.weight.t())


def _resolve_kimi_situ_betas(cfg: Any) -> tuple[float, float]:
    """Return the finite SiTu betas required by the routed-expert kernels."""
    config_situ_beta = getattr(cfg, "activation_situ_beta", None)
    situ_beta = 1.0 if config_situ_beta is None else config_situ_beta
    situ_linear_beta = getattr(cfg, "activation_situ_linear_beta", None)
    if situ_linear_beta is None:
        raise ValueError(
            "Kimi K3.1 routed SiTu experts require activation_situ_linear_beta; "
            "None means an identity linear branch that the fused kernels cannot represent."
        )
    if situ_beta <= 0 or situ_linear_beta <= 0:
        raise ValueError(
            f"Kimi K3.1 SiTu betas must be positive; got {situ_beta} and {situ_linear_beta}."
        )
    return float(situ_beta), float(situ_linear_beta)


# ---------------------------------------------------------------------------
# Config helpers.
# ---------------------------------------------------------------------------


def _is_kda_layer(cfg, layer_idx: int) -> bool:
    return (layer_idx + 1) in cfg.linear_attn_config["kda_layers"]


def _is_mla_layer(cfg, layer_idx: int) -> bool:
    return (layer_idx + 1) in cfg.linear_attn_config["full_attn_layers"]


# ---------------------------------------------------------------------------
# attn_res: per-token snapshot mixing (HF `_apply_attn_res`).
# ---------------------------------------------------------------------------


# Largest per-rank token count for which the fused attn_res + RMSNorm epilogue ops run (the
# single-token decode shape); larger token counts take attn_res_fwd and a separate RMSNorm.
_FUSED_ATTN_RES_MAX_TOKENS = 1


# Token limit of trtllm::attn_res_fwd (cpp/tensorrt_llm/thop/attnResOp.cpp).
_ATTN_RES_OP_MAX_TOKENS = 16384


def _apply_attn_res_fused(
    prefix_sum: torch.Tensor, block_residual: torch.Tensor, proj: nn.Linear, norm: K31RMSNorm
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
            float(norm.variance_epsilon),
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


def _rms_norm_eps(norm: nn.Module) -> float:
    return float(norm.variance_epsilon)


def _note_attn_res_fusion(site: str, fused: bool, M: int, H: int, N: int) -> None:
    """Report whether the fused path was actually reached, once per shape.

    A call the shape gate rejects takes the FP32 fallback with nothing else in
    the logs to tell it from a fused one. Emitted at debug level: it is once
    per distinct shape, not once per process, so it is a diagnostic rather
    than a summary.
    """
    logger.debug_once(
        f"Kimi K3.1 attn-res fusion [{site}]: "
        f"{'FUSED' if fused else 'fallback'} (M={M}, H={H}, N={N})",
        key=f"kimi_k3_attn_res_fusion_{site}_{fused}_{M}_{H}_{N}",
    )


def _apply_attn_res_rmsnorm_fused(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: K31RMSNorm,
    output_norm: nn.Module,
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
        float(norm.variance_epsilon),
        _rms_norm_eps(output_norm),
    )
    _note_attn_res_fusion("attn_res+norm", True, M, H, N)
    return output.reshape(M, H)


def _apply_attn_res_add_rmsnorm_fused(
    prefix_sum: torch.Tensor,
    addend: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: K31RMSNorm,
    output_norm: nn.Module,
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
        float(norm.variance_epsilon),
        _rms_norm_eps(output_norm),
    )
    _note_attn_res_fusion("add+attn_res+norm", True, M, H, N)
    return updated_prefix_sum.reshape(M, H), output.reshape(M, H)


def _apply_attn_res(
    prefix_sum: torch.Tensor, block_residual: torch.Tensor, proj: nn.Linear, norm: K31RMSNorm
) -> torch.Tensor:
    """Exact port of HF ``modeling_kimi._apply_attn_res`` (fp32 math).

    prefix_sum:     ``[num_tokens, hidden_size]``
    block_residual: ``[num_snapshots, num_tokens, hidden_size]``

    Inputs fitting the fused kernel's contract dispatch directly to the
    in-tree ``trtllm::attn_res_fwd`` op.
    Only the fallback boundary restores the HF ``[M, K, H]`` layout.
    """
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
    k = v_float * torch.rsqrt(variance + norm.variance_epsilon)
    score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    hidden_states = torch.matmul(probs, v_float).squeeze(1)
    return hidden_states.to(v.dtype)


#: Generation rows per rank the PDL attn_res mixes take on this attention-TP target; from 192 rows (and for
#: prefill) the persistent mix is faster.
_K31_MIX_PERSISTENT_ROWS = 192


def _k31_mix_persistent(m: int, n: int) -> bool:
    return m >= _K31_MIX_PERSISTENT_ROWS


def _k31_mailbox_rows(prefix_sum: torch.Tensor) -> bool:
    """Whether the next layer's mix can take a fused-tail mailbox for this residual (the PDL mix's rows)."""
    return (
        prefix_sum.dim() == 2
        and prefix_sum.dtype is torch.bfloat16
        and prefix_sum.is_contiguous()
        and 1 <= prefix_sum.shape[0] < _K31_MIX_PERSISTENT_ROWS
    )


def _k31_defer_kda_oproj(prefix_sum: Optional[torch.Tensor], tp_size: int) -> bool:
    """Whether a KDA layer hands its o_proj push chain to the residual-add mix (``_K31PendingOproj``): on every
    residual-add layer, except at attention TP 32 on the persistent mix's rows, where the chain's own finish kernel
    completes the all-reduce and the persistent mix follows, as on those rows without the push chain (the PDL mix's
    output is one BF16 step from the persistent mix's in a fraction of the elements)."""
    if prefix_sum is None:
        return False
    return tp_size != 32 or not _k31_mix_persistent(prefix_sum.shape[0], 0)


def _k31_mix_operands(prefix_sum, block_residual, proj, norm, output_norm):
    """``(res_weight, rms_weight, out_weight, rms_eps, out_eps)`` of the family attn_res mixes, or None where
    the operands are outside them (BF16 [M, 7168] rows, 2..9 candidates)."""
    if (
        prefix_sum.dtype is not torch.bfloat16
        or not prefix_sum.is_cuda
        or prefix_sum.dim() != 2
        or prefix_sum.shape[1] != 7168
        or not prefix_sum.is_contiguous()
        or not block_residual.is_contiguous()
        or not 2 <= int(block_residual.shape[0]) + 1 <= 9
    ):
        return None
    return (
        proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
        norm.weight.to(torch.bfloat16).contiguous(),
        output_norm.weight.to(torch.bfloat16).contiguous(),
        float(norm.variance_epsilon),
        _rms_norm_eps(output_norm),
    )


def _apply_attn_res_and_rmsnorm(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: K31RMSNorm,
    output_norm: nn.Module,
) -> torch.Tensor:
    """Apply attention-residual selection and the next RMSNorm.

    Generation rows take the family's PDL mix, larger batches and prefill the persistent mix.
    """
    operands = _k31_mix_operands(prefix_sum, block_residual, proj, norm, output_norm)
    if operands is not None:
        m, n = prefix_sum.shape[0], int(block_residual.shape[0]) + 1
        if _k31_mix_persistent(m, n):
            return attn_res_add_rmsnorm_persistent_fwd(prefix_sum, None, block_residual, *operands)[
                1
            ]
        return k31_attn_res_rmsnorm(prefix_sum, block_residual, *operands)
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
    proj: nn.Linear,
    norm: K31RMSNorm,
    output_norm: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add an attention output to the running residual, then select and norm.

    Generation rows take the family's PDL mix, larger batches and prefill the persistent mix.
    """
    operands = _k31_mix_operands(prefix_sum, block_residual, proj, norm, output_norm)
    if operands is not None and addend.dtype is torch.bfloat16 and addend.is_contiguous():
        m, n = prefix_sum.shape[0], int(block_residual.shape[0]) + 1
        if _k31_mix_persistent(m, n):
            return attn_res_add_rmsnorm_persistent_fwd(
                prefix_sum, addend, block_residual, *operands
            )
        return k31_attn_res_add_rmsnorm(prefix_sum, addend, block_residual, *operands)
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
# Latent MoE block: the family routed experts and the shared expert.
# ---------------------------------------------------------------------------

# Module-name spellings of a layer's routed experts.
_K3_ROUTED_EXPERT_KEY_SUFFIXES = ("block_sparse_moe.experts", "mlp.experts")

# The text-model prefixes of a routed-expert module path, for the exclusion check. ``exclude_modules``
# matches with wildcards and walks ancestor prefixes, so an empty prefix would widen what matches; only
# real module paths are asked.
_K3_ROUTED_EXPERT_MODULE_PREFIXES = ("language_model.model.", "model.")

# Routed-expert quantization. Routing admits only the MXFP4 checkpoint (compressed-tensors
# ``mxfp4-pack-quantized``), which declares nothing per layer, so its routed experts resolve to this.
_K3_DEFAULT_ROUTED_QUANT_ALGO = QuantAlgo.W4A8_MXFP4_MXFP8


class _K31PendingMoE(NamedTuple):
    """A decoder layer's output whose MoE output is still in the fused tail's mailbox: the running residual before
    the MoE add and the mailbox, handed to the next layer, whose pre-attention mix adds the mailbox rows
    (``k31_attn_res_mailbox_add_rmsnorm``). Every pending output is consumed exactly once, by that mix or by
    ``_k31_settle_moe``, before the next tail call."""

    prefix: torch.Tensor
    mailbox: torch.Tensor
    tail: Any


def _k31_settle_moe(pending: _K31PendingMoE) -> torch.Tensor:
    """The running residual with the MoE output added (the tail's copy kernel consumes the mailbox)."""
    return pending.prefix + k31_latent_moe_tail_copy_out(
        pending.mailbox, pending.prefix.shape[0], pending.tail
    )


# The MoE front's folded routed experts (<= 32 rows): token tiles of the runner sort the front writes, the rows of a
# front call, the experts per token, and one zeroed CuTe DSL workspace per (device, expert width). Above the front's
# rows (up to the fused tail's capacity) the MoE prologue routes, the CuTe DSL experts sort themselves with
# ``pick_n_tile``'s tiles, and the tail folds the top-k sum the same way.
_K31_FRONT_FOLD_TILE = 8
_K31_FRONT_FOLD_MAX_ROWS = 32
# Attention-TP sizes whose routed experts' FC1 releases FC2 right after its grid dependency up to 64 rows (the op's
# default bound is 32): FC2 streams its first weights beside FC1's last CTAs. Same bits. GB300 TEP8, w5, 120k / 200k
# context, ms per step at 7 / 8 / 10 requests: -0.04 / -0.05 / -0.09 and -0.04 / -0.06 / -0.05 (MoE EP8), -0.03 /
# -0.02 / -0.02 and -0.05 / -0.04 / -0.04 (MoE EP2); at TEP16 the same bound is +0.3 slower.
_K31_W4A8_EARLY_RELEASE_TP = (8,)
_K31_W4A8_EARLY_RELEASE_MAX_ROWS = 64
_CUTE_MOE_TOP_K = 16
_K31_FRONT_FOLD_WORKSPACES: Dict[Tuple[Optional[int], int], torch.Tensor] = {}
# TP sizes whose MoE front also computes this rank's shared-expert gate_up columns and SiTU (<= 16 rows). At TP16
# the shared expert runs whole on its stream, gate_up on the low-M plan: the unfused path's bits.
_K31_FRONT_SHARED_TP = (32,)
# Attention TP 32 above the one-shot tail's 32 rows (up to 192): a second fused tail instance, two-shot with the
# MNNVL all-reduce RMSNorm's numerics, bitwise to the unfused MoE forward those rows took before (routed all-reduce +
# RMSNorm, cuBLAS latent up projection, shared all-reduce, add).
# TP sizes whose MoE head above the front's rows runs the router and latent down GEMMs as one launch
# (``k31_moe_head_gemm``) where it gives their bits; elsewhere the two cuBLAS GEMMs run on two streams. At TP8 the
# unfused MoE forward (the rows the front + tail do not serve) takes the merged head at its rows and keeps the router
# GEMM before the shared expert's fork and the latent down projection after it elsewhere (on two streams the shared
# expert starts later and the step is slower: GB300 TEP8 without the front, 120k context, +0.9 ms at 5 and 20 rows).
_K31_MOE_HEAD_TP = (8, 16)
_K31_HEAD_COMPILED_DEVICES: set = set()

# Above the front's rows the MoE layer's critical path (router GEMM, MoE prologue, routed experts) runs on a
# high-priority stream while the latent down projection and the shared expert keep their default-priority streams:
# their CTAs no longer take SMs ahead of the critical path's. Measured at TEP16 only. GB300, one TEP16 rank's MoE
# layer (CUDA graph), us per layer: -2.4 / -1.4 / -2.7 / -2.5 / -3.2 / -3.6 at 40 / 80 / 100 / 120 / 160 / 192 rows;
# raising the latent down stream as well, or the shared expert's instead, is slower.
_K31_MOE_PRIORITY_TP = (16,)
# At these attention-TP sizes, up to the front's rows, the shared expert (gate / up, SiTU, down, and its early push to
# the fused tail) runs on the high-priority stream instead of the shared-expert aux stream. The routed W4A8 GEMMs
# fill every SM with one persistent CTA each; on a default-priority stream the shared down's CTAs get SMs only as FC2
# drains, and the fused tail waits for them after FC2. On the high-priority stream they take the SMs FC1 frees first.
# Same kernels, inputs and order (the same bits). GB300 TEP8 (MoE TP4 x EP2) MoE layer replica, 8 ranks, CUDA graph,
# us per layer at 5 / 10 / 15 rows: 49.9 / 69.1 / 83.1 -> 48.6 / 68.6 / 82.1. TEP16 / TEP32 not measured.
_K31_SHARED_PRIORITY_TP = (8,)
# Above the front's rows and below the one-launch MoE head's rows (``k31_moe_head_gemm``) at these TP sizes, the
# shared expert is issued on its stream at the start of the MoE layer, beside the router and latent down GEMMs, and
# the routing (the MoE prologue's routing half) and the routed experts' sort run right after the router GEMM while the
# latent down projection still runs; only the prologue's quantization half waits for the latent. Otherwise the
# prologue (routing + quantization) waits for both GEMMs and runs beside the shared expert's GEMMs (11-14 us instead
# of ~4). Same kernels and bits either way. GB300 TEP16, 120k context, fixed batch (draft 4), ms per step: -0.9 at 12
# requests (60 rows); at the one-launch head's rows (16 / 20 / 26 requests) routing beside two GEMMs, or the shared
# expert first before the one-launch head, is not faster than the head.
_K31_MOE_EARLY_ROUTING_TP = (16,)
# Above the front's rows up to _K31_HEAD_SHARD_MAX_ROWS, the MoE head (router + latent down) is sharded over the
# attention-TP group (``moe_head_shard``): each rank multiplies the rows by its 1 / TP of the head weight with the
# replicated head's K split and the ranks all-gather the slices, the bits of the replicated head; then the MoE prologue
# routes and quantizes (in the all-gather's launch where it can). The replicated head reads the whole head weight
# (64 MB per layer) on every rank.
# TP sizes whose sharded MoE head runs the gather and then the MoE prologue: at TP32 a rank's latent slice
# (3584 / 32 = 112 values) is not whole MXFP8 blocks, so the prologue cannot be folded into the all-gather. Same bits
# as the replicated router and latent down GEMMs it replaces. GB300, 32 ranks (8 trays), us per MoE head (+ prologue)
# layer at 33-80 rows: 20.2-22.5 -> 13.5-16.2.
_K31_HEAD_SHARD_UNFUSED_TP = (32,)
# Per TP size, a lower cap than the op's MAX_ROWS on the rows the sharded head serves: at TP8, above 96 rows (the
# latent's whole K, one MMA chain per weight tile) it is no faster in the model than the replicated head. GB300 TEP8,
# 120k context, fixed batch (draft 4), ms per step at 20 requests (100 rows): +0.06 / -0.01.
_K31_HEAD_SHARD_MAX_ROWS_TP = {8: 96}
_K31_HEAD_SHARD_COMPILED_DEVICES: set = set()
_K31_MOE_PRIORITY_STREAMS: Dict[int, torch.cuda.Stream] = {}


def _k31_head_shard_max_rows(tp: int) -> int:
    """The largest rows the sharded MoE head serves at attention TP ``tp``."""
    return _K31_HEAD_SHARD_MAX_ROWS_TP.get(tp, _K31_HEAD_SHARD_MAX_ROWS)


def _k31_moe_priority_stream() -> Optional[torch.cuda.Stream]:
    """This device's highest-priority stream for the MoE critical path (one, shared by every layer), or None where
    the device has no stream priorities."""
    device = torch.cuda.current_device()
    stream = _K31_MOE_PRIORITY_STREAMS.get(device)
    if stream is None:
        lowest, highest = torch.cuda.Stream.priority_range()
        if highest >= lowest:
            return None
        stream = _K31_MOE_PRIORITY_STREAMS[device] = torch.cuda.Stream(priority=highest)
    return stream


def _k31_shared_on_priority_stream(front_tp: int, m: int) -> bool:
    """Whether the shared expert of an ``m``-row call through the MoE front at attention TP ``front_tp`` runs on the
    high-priority stream (``_K31_SHARED_PRIORITY_TP``, up to the front's rows)."""
    return front_tp in _K31_SHARED_PRIORITY_TP and m <= _K31_FRONT_FOLD_MAX_ROWS


class K31MoE(nn.Module):
    """Kimi K3.1 latent MoE block: the routed experts (``K31RoutedExperts``) and the shared expert."""

    # Low-M GEMM plan of the latent projections (set at construction on SM 10.3 generation layouts).
    _k31_low_m_plan = None

    def __init__(
        self,
        model_config: ModelConfig,
        cfg,
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ):
        """Build the routed experts and the shared expert for one MoE layer.

        ``cfg`` is the raw ``PretrainedConfig`` rather than anything derived:
        the SiTU soft-caps and the routed-expert geometry are Kimi K3.1 fields
        that ``ModelConfig`` does not carry.

        ``aux_stream_dict`` is shared across every layer of the model, so the
        streams reached through it are borrowed and must not be synchronized
        or reassigned here.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = cfg.hidden_size
        self.num_experts = cfg.num_experts
        self.top_k = cfg.num_experts_per_token
        self.moe_hidden_size = cfg.routed_expert_hidden_size
        # ValueError (not assert): these guard unsupported checkpoint
        # configurations and must stay active under ``python -O``.
        if self.moe_hidden_size is None:
            raise ValueError("Kimi K3.1 runtime expects the latent MoE (routed_expert_hidden_size)")
        if not getattr(cfg, "latent_moe_use_norm", False):
            raise ValueError("Kimi K3.1 runtime expects latent_moe_use_norm=True")

        situ_beta, situ_linear_beta = _resolve_kimi_situ_betas(cfg)
        dtype = torch.bfloat16

        # Routing scores stay FP32: the gate GEMM takes the BF16 checkpoint weight and a BF16 input with FP32
        # accumulation and output (no per-layer input cast or FP32 split-K pair on the decode path).
        self.gate = K31MoEGate(cfg)

        routed_quant_config = self._resolve_routed_quant_config(model_config, layer_idx)
        # Resolved here so ``load_weights`` reads the checkpoint layout off the
        # module instead of re-deriving it at each of its three call sites.
        self.expert_ckpt_spec = expert_ckpt_spec(routed_quant_config.quant_algo)
        self.routed_experts = K31RoutedExperts(
            model_config,
            self._routed_mapping(model_config),
            num_experts=self.num_experts,
            hidden_size=self.moe_hidden_size,
            intermediate_size=cfg.moe_intermediate_size,
            gate_cap=situ_beta,
            linear_cap=situ_linear_beta,
            layer_idx=layer_idx,
            e_score_correction_bias=lambda: self.gate.e_score_correction_bias,
            routed_scaling_factor=self.gate.routed_scaling_factor,
            aux_stream_dict=aux_stream_dict,
        )
        local_expert_ids = self.routed_experts.initial_local_expert_ids
        self.local_expert_ids = tuple(local_expert_ids)
        self.experts_per_rank = len(local_expert_ids)
        self.expert_lo = local_expert_ids[0]
        self.expert_hi = self.expert_lo + self.experts_per_rank
        # Checkpoint expert of each local routed-expert slot (identity placement).
        self.ckpt_expert_ids = self.local_expert_ids

        shared_intermediate = cfg.moe_intermediate_size * cfg.num_shared_experts
        attention_dp = model_config.mapping.enable_attention_dp
        # Under attention DP each rank owns different tokens, so the shared
        # expert is replicated (TP size 1) and must not reduce across ranks.
        # Direct MoE-TP leaves both branches as partials for one concatenated
        # all-reduce.
        use_shared_tp = not attention_dp and model_config.mapping.tp_size > 1
        self.shared_experts = K31SituMLP(
            cfg.hidden_size,
            shared_intermediate,
            dtype=dtype,
            mapping=model_config.mapping,
            beta=situ_beta,
            linear_beta=situ_linear_beta,
            tp_size=1 if attention_dp else None,
            reduce_output=use_shared_tp,
            allreduce_strategy=model_config.allreduce_strategy,
            # Decode gate / up projection on the shared expert's low-M plan.
            low_m_plan=_k31_shared_expert_plan(model_config.mapping),
        )
        # Side stream (+ fork/join events) for overlapping shared-expert
        # compute with the routed chain. Only engaged when multi-stream is
        # active (CUDA graphs on); otherwise both run in order on the default
        # stream.
        self.shared_expert_stream = aux_stream_dict[AuxStreamType.MoeShared]
        self.moe_main_event = torch.cuda.Event()
        self.moe_shared_event = torch.cuda.Event()
        # Above the MoE front's rows the latent down projection runs on its own stream beside the router GEMM: both
        # are weight streams of the replicated head, joined before the routing reads either.
        self.latent_down_stream = aux_stream_dict[AuxStreamType.MoeOutputMemset]
        self.moe_latent_fork_event = torch.cuda.Event()
        self.moe_latent_ready_event = torch.cuda.Event()
        self.routed_expert_down_proj = nn.Linear(
            cfg.hidden_size, self.moe_hidden_size, bias=False, dtype=dtype
        )
        self.routed_expert_up_proj = nn.Linear(
            self.moe_hidden_size, cfg.hidden_size, bias=False, dtype=dtype
        )
        # The routed latent's RMSNorm weight (the norm runs in flashinfer_rmsnorm or the fused all-reduce norm).
        self.routed_expert_norm = K31RMSNorm(
            self.moe_hidden_size, eps=cfg.rms_norm_eps, dtype=dtype
        )
        # Latent down / up projections on the low-M plan of the layout.
        self._k31_low_m_plan = _k31_decode_plan(model_config.mapping)
        self._init_front_and_tail(model_config.mapping, situ_beta, situ_linear_beta)

    def _init_front_and_tail(self, mapping: Mapping, situ_beta, situ_linear_beta) -> None:
        """Attention-TP generation (TP8 / TP16 / TP32): the sharded MoE front (router + latent_down + top-16 + MXFP8, at
        TP32 also the shared gate_up + SiTU at <= 16 rows) and the fused latent-MoE tail (routed all-reduce +
        RMSNorm + up projection + shared all-reduce + add). Their workspaces are collective on the TP group and real
        device memory, so they are built in ``post_load_weights`` (every rank, outside capture and meta
        initialization), with the front's packed weight."""
        self._front_ws = self._front_w = self._tail = self._front_fold = None
        # The two-shot-only tail above the default tail's rows (``fused_tail_twoshot_only_rows``: TP32).
        self._tail_wide = None
        self._head_weight = None
        self._head_shard = None
        self._head_shard_bias = None
        # The MoE prologue routes the calls above the front's rows that fold their top-k sum (set with the front).
        self._route_quant = False
        tp = mapping.tp_size
        # TP8: the unfused MoE forward (the rows the front + tail do not serve) takes the merged head at its rows.
        self._head_only = tp not in (16, 32) and tp in _K31_MOE_HEAD_TP
        shared_cols = 6144 // tp if tp in _K31_FRONT_SHARED_TP else 0
        self._front_mapping = mapping
        self._front_tp, self._front_rank, self._front_shared_cols = tp, mapping.tp_rank, shared_cols
        self._front_caps = (float(situ_beta), float(situ_linear_beta))
        self._route_quant = True

    def post_load_weights(self) -> None:
        """Build the MoE front's workspace and the fused tail, and pack the front's weight (latent_down + router,
        at TP32 this rank's shared gate_up rows), after the load."""
        self._front_ws = _k31_front.front_workspace(self._front_mapping)
        self._tail = KimiK3LatentMoETail.get(
            self._front_mapping,
            hidden_size=7168,
            latent_size=3584,
            rms_eps=float(self.routed_expert_norm.variance_epsilon),
        )
        if fused_tail_twoshot_only_rows(self._front_tp) is not None:
            self._tail_wide = KimiK3LatentMoETail.get(
                self._front_mapping,
                hidden_size=7168,
                latent_size=3584,
                rms_eps=float(self.routed_expert_norm.variance_epsilon),
                twoshot_only=True,
            )
        self._front_w = _k31_front.front_weight(
            self.routed_expert_down_proj.weight,
            self.gate.weight,
            self._front_rank,
            self._front_tp,
            self.shared_experts.gate_up_proj.weight if self._front_shared_cols else None,
        )
        self._front_fold = self._front_fold_workspace()
        if self._front_tp in _K31_MOE_HEAD_TP:
            self._head_weight = self._stack_head_weight()
            if self._head_only:
                self._compile_head()
            self._head_shard = self._build_head_shard()
            self._head_shard_bias = self._head_shard_fold_bias()
        elif self._front_tp in _K31_HEAD_SHARD_UNFUSED_TP:
            self._head_weight = self._stack_head_weight()
            self._head_shard = self._build_head_shard()

    def _compile_head(self) -> None:
        """Compile every row range of the merged head once per device, at load: context chunks of the head's rows
        reach it eagerly too (a TP8 context server), not only the warmed-up generation graphs."""
        weight = self._head_weight
        if weight is None or weight.device.index in _K31_HEAD_COMPILED_DEVICES:
            return
        bounds = [_K31_MOE_HEAD_MIN_ROWS] + [bound + 1 for bound, _ in _K31_HEAD_CONFIGS[:-1]]
        x = torch.zeros((bounds[-1], weight.shape[1]), dtype=weight.dtype, device=weight.device)
        for rows in bounds:
            k31_moe_head_gemm(x[:rows], weight)
        _K31_HEAD_COMPILED_DEVICES.add(weight.device.index)

    def _build_head_shard(self) -> Optional[Tuple[nn.Module, int]]:
        """``(all-reduce module, gather width)`` of this rank's sharded MoE head: the TP group's all-reduce module whose
        MNNVL Lamport workspace the gather uses and the gather's BF16 width per rank; None where the sharded head does
        not run (no MNNVL workspace). Collective on the TP group: grows the group's Lamport
        workspace for the largest gather and compiles the kernels, outside capture."""
        if self._head_weight is None or not _k31_head_shard_serves(
            _K31_HEAD_SHARD_MIN_ROWS, self._front_tp
        ):
            return None
        max_rows = _k31_head_shard_max_rows(self._front_tp)
        # AUTO, not the configured all-reduce strategy: this handle never reduces, it only carries the TP group's
        # MNNVL Lamport workspace, which AUTO builds wherever MNNVL is available.
        all_reduce = tp_allreduce(self._front_mapping)
        if not uses_mnnvl(all_reduce):
            logger.warning_once(
                "Kimi K3.1 MoE: no MNNVL all-reduce workspace on the TP group; the head stays replicated",
                key="k31_moe_head_shard_off",
            )
            return None
        world = self._front_tp
        width = 2 * _k31_head_shard_gather_bytes(1, world) // (4 * world)
        lamport_workspace(all_reduce, max_rows, width, one_shot=True)
        w_latent, w_router = _k31_head_shard_views(self._head_weight, self._front_rank, world)
        device = self._head_weight.device
        if device.index not in _K31_HEAD_SHARD_COMPILED_DEVICES:
            x = torch.zeros((max_rows, w_latent.shape[1]), dtype=torch.bfloat16, device=device)
            seen = set()
            for rows in range(_K31_HEAD_SHARD_MIN_ROWS, max_rows + 1):
                config = _k31_head_shard_config(rows)
                if config in seen:
                    continue
                seen.add(config)
                _k31_compile_head_shard(
                    world,
                    x[:rows],
                    w_latent,
                    w_router,
                    config,
                    routed=world not in _K31_HEAD_SHARD_UNFUSED_TP,
                )
            torch.cuda.synchronize()
            _K31_HEAD_SHARD_COMPILED_DEVICES.add(device.index)
        logger.info_once(
            f"Kimi K3.1 MoE: head sharded over TP {world} at "
            f"{_K31_HEAD_SHARD_MIN_ROWS}-{max_rows} rows",
            key="k31_moe_head_shard_on",
        )
        return all_reduce, width

    def _sharded_head(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(router logits, routed latent)`` of every row from this rank's slice of the head and the all-gather."""
        all_reduce, width = self._head_shard
        workspace = lamport_workspace(all_reduce, hidden_states.shape[0], width, one_shot=True)
        return k31_moe_head_shard(
            hidden_states,
            self._head_weight,
            self._front_rank,
            self._front_tp,
            workspace,
        )

    def _head_shard_fold_bias(self) -> Optional[torch.Tensor]:
        """The router's correction bias where the sharded head's calls fold the MoE prologue into the all-gather (one
        launch reads the slices straight from the Lamport buffer and runs the prologue's code on them, so its bits):
        the head is sharded, the CuTe DSL experts take the folded routing (the front path's fold workspace) and the
        prologue routes with the runner's weights at unit scale; else None (the gather, then the prologue)."""
        if self._head_shard is None or self._front_fold is None or not self._route_quant:
            return None
        bias = runner_routing_bias(
            self.gate.e_score_correction_bias,
            self.routed_experts.routed_scaling_factor,
            True,
        )
        if bias is None or bias.device != self._head_weight.device or bias.data_ptr() % 16:
            return None
        logger.info_once(
            "Kimi K3.1 MoE: MoE prologue folded into the sharded head's all-gather",
            key="k31_moe_head_shard_fold",
        )
        return bias

    def _sharded_head_routed(self, hidden_states: torch.Tensor):
        """``_folded_routed_output``'s arguments of every row: the sharded head with the MoE prologue folded into its
        all-gather (the routing and MXFP8 latent of ``_prologue_folded_routed_output``, the experts sorting
        themselves with ``pick_n_tile``'s tiles)."""
        all_reduce, width = self._head_shard
        m = hidden_states.shape[0]
        workspace = lamport_workspace(all_reduce, m, width, one_shot=True)
        ids, weights, x_fp8, x_sf = k31_moe_head_shard_routed(
            hidden_states,
            self._head_weight,
            self._head_shard_bias,
            self._front_rank,
            self._front_tp,
            workspace,
        )
        return ids, weights, x_fp8, x_sf, None, pick_n_tile(m)

    def _head_shard_serves(self, m: int, hidden_states: torch.Tensor) -> bool:
        """Whether a call of ``m`` rows runs the sharded MoE head."""
        return (
            self._head_shard is not None
            and m <= _k31_head_shard_max_rows(self._front_tp)
            and _k31_head_shard_serves(m, self._front_tp)
            and hidden_states.is_contiguous()
            and hidden_states.data_ptr() % 32 == 0
        )

    def _stack_head_weight(self) -> Optional[torch.Tensor]:
        """The router and latent down weights stacked into one ``[896 + 3584, 7168]`` BF16 tensor (router rows
        first) for ``k31_moe_head_gemm``; the two parameters become views of it, so no weight is held twice. None
        where the weights are not both BF16 ``[N, 7168]``."""
        router, latent = self.gate.weight, self.routed_expert_down_proj.weight
        if (
            router.dtype != torch.bfloat16
            or latent.dtype != torch.bfloat16
            or tuple(router.shape) != (896, 7168)
            or tuple(latent.shape) != (3584, 7168)
        ):
            return None
        head = torch.empty((896 + 3584, 7168), dtype=torch.bfloat16, device=router.device)
        with torch.no_grad():
            head[:896].copy_(router)
            head[896:].copy_(latent)
        router.data = head[:896]
        latent.data = head[896:]
        return head

    def _front_fold_workspace(self) -> Optional[torch.Tensor]:
        """The zeroed CuTe DSL routed-expert workspace of the front path's folded call (the front writes the runner
        sort, the CuTe DSL GEMMs leave the expert rows unfinalized, the fused tail folds the top-k sum), or None
        where the routed experts do not run the CuTe DSL GEMMs. One workspace per device and expert shape serves
        every layer (each call re-arms it), sized for the front's 32 rows."""
        w1 = self.routed_experts.w3_w1_weight
        local = self.routed_experts.expert_size_per_partition
        if w1 is None or w1.dim() != 3 or w1.shape[0] != local or w1.shape[1] % 256:
            return None
        inter = w1.shape[1] // 2
        key = (w1.device.index, inter)
        ws = _K31_FRONT_FOLD_WORKSPACES.get(key)
        if ws is None:
            tiles = _K31_FRONT_FOLD_MAX_ROWS * _CUTE_MOE_TOP_K
            size = moe_workspace_bytes(tiles, inter, _K31_FRONT_FOLD_TILE)
            # the self-sorted calls above the front's rows, up to the tails' capacity: the largest call of each
            # of their token tiles (pick_n_tile: 8 up to 47 rows, 16 above)
            cap = max(t.max_num_tokens for t in (self._tail, self._tail_wide) if t is not None)
            for rows in (min(47, cap), cap):
                if rows > _K31_FRONT_FOLD_MAX_ROWS:
                    n_tile = pick_n_tile(rows)
                    size = max(
                        size, moe_workspace_bytes(max_tiles_for(rows, local, n_tile), inter, n_tile)
                    )
            ws = torch.zeros(size, dtype=torch.uint8, device=w1.device)
            _K31_FRONT_FOLD_WORKSPACES[key] = ws
        return ws

    def _tail_for(self, m: int) -> Optional[KimiK3LatentMoETail]:
        """The fused tail instance that serves a call of ``m`` rows (the default tail, else the two-shot-only one
        above its rows), or None."""
        if self._tail.can_run(m):
            return self._tail
        if self._tail_wide is not None and self._tail_wide.can_run(m):
            return self._tail_wide
        return None

    def _front_forward(
        self, hidden_states: torch.Tensor, to_mailbox: bool = False
    ) -> Optional[torch.Tensor]:
        """The MoE output through the front (<= 32 rows) or the replicated front, the routed experts and the fused
        tail, or None where the tail does not serve the call or where the routed experts would take the front's
        routing for more rows than the MoE token budget (``moe_max_num_tokens``) gives one call. ``to_mailbox``
        returns the tail's mailbox instead of the ``[M, hidden]`` output (``k31_latent_moe_tail_to_mailbox``)."""
        m = hidden_states.shape[0]
        fused_tail = self._tail_for(m)
        if (
            fused_tail is None
            or hidden_states.dtype != torch.bfloat16
            or not hidden_states.is_contiguous()
        ):
            return None
        act = None
        cols = 0
        folded = front_routed = None
        router_logits = routed_in = None
        caller = torch.cuda.current_stream()
        critical = self._critical_path_stream(m)
        if critical is not None:
            critical.wait_stream(caller)
            torch.cuda.set_stream(critical)
        if m <= 32:
            cols = self._front_shared_cols if m <= 16 else 0
            capturing = torch.cuda.is_current_stream_capturing()
            local = self.routed_experts.expert_size_per_partition
            if (
                self._front_fold is not None
                and _k31_front.runner_sort_fits(m, local, _K31_FRONT_FOLD_TILE)
                and not (
                    capturing
                    and not _k31_front.is_compiled(
                        m,
                        self._front_tp,
                        cols,
                        hidden_states.device,
                        *self._front_caps,
                        runner_sort=True,
                    )
                )
            ):
                # The front also writes the runner sort of this rank's experts; the CuTe DSL GEMMs leave the expert
                # rows unfinalized and the fused tail folds the top-k weighted sum.
                ids, weights, x_fp8, x_sf, act, prerouted = _k31_front.moe_front_runner_sort(
                    hidden_states,
                    self._front_w,
                    self.gate.e_score_correction_bias,
                    float(self.gate.routed_scaling_factor),
                    self._front_ws,
                    cols,
                    *self._front_caps,
                    _K31_FRONT_FOLD_TILE,
                    self.routed_experts.slot_start,
                    local,
                )
                folded = (ids, weights, x_fp8, x_sf, prerouted)
            else:
                self_sorted = self._front_fold is not None and not _k31_front.runner_sort_fits(
                    m, local, _K31_FRONT_FOLD_TILE
                )
                if not self_sorted and m > self.routed_experts.moe_max_num_tokens:
                    # The routed experts take the front's routing as one pre-routed call, which the MoE token
                    # budget would split into chunks: the unfused forward chunks these rows instead.
                    return None
                if capturing and not _k31_front.is_compiled(
                    m, self._front_tp, cols, hidden_states.device, *self._front_caps
                ):
                    return None
                ids, weights, x_fp8, x_sf, act = _k31_front.moe_front(
                    hidden_states,
                    self._front_w,
                    self.gate.e_score_correction_bias,
                    float(self.gate.routed_scaling_factor),
                    self._front_ws,
                    cols,
                    *self._front_caps,
                )
                if self_sorted:
                    # More local experts than the front's runner sort covers (MoE TP4 x EP2 at TP8): the CuTe DSL
                    # experts sort the front's routing themselves, with the front's tiles, and the fused tail still
                    # folds the top-k weighted sum.
                    logger.info_once(
                        f"Kimi K3.1 MoE front: the CuTe DSL experts sort {local} local experts "
                        "themselves, top-k sum folded into the fused tail",
                        key="k31_moe_front_self_sorted_fold",
                    )
                    folded = (ids, weights, x_fp8, x_sf, None)
                else:
                    front_routed = (ids, weights, x_fp8, x_sf)
            # The routed experts take the front's routing and quantized input: these two are not read.
            router_logits = hidden_states.new_empty((m, 896), dtype=torch.float32)
            routed_in = hidden_states.new_empty((m, 3584))
        shard = m > _K31_FRONT_FOLD_MAX_ROWS and self._head_shard_serves(m, hidden_states)
        early_routing = not shard and self._early_routing(m)
        if m > 32 and not early_routing and not shard:
            router_logits, routed_in = self._replicated_head(hidden_states)

        def _routed_output():
            f, logits, latent = folded, router_logits, routed_in
            if early_routing:
                # issued here, after the shared expert's fork: the shared expert runs beside the router / latent GEMMs
                f, logits, latent = self._early_routed_head(hidden_states)
            elif shard:
                # issued here, after the shared expert's fork, like the early routing
                if self._head_shard_bias is not None:
                    f = self._sharded_head_routed(hidden_states)
                else:
                    logits, latent = self._sharded_head(hidden_states)
            y = None
            if f is not None:
                y = self._folded_routed_output(*f)
            elif self._front_fold is not None and m > _K31_FRONT_FOLD_MAX_ROWS:
                y = self._prologue_folded_routed_output(latent, logits)
            if y is None:
                y = self.routed_experts(latent, logits, routed=front_routed)
            return y

        # The tail's shared reduce-scatter is pushed from the shared expert's stream while the routed experts run (the
        # same stores and sums as the tail's own push). Only under CUDA-graph capture, where the two branches overlap:
        # run in order, the extra launch would sit on the critical path.
        push_shared = fused_tail.pushes_shared_early(m) and torch.cuda.is_current_stream_capturing()

        def _shared_output():
            if cols:
                out = self.shared_experts.down_proj(act, reduce=False)
            else:
                out = self.shared_experts.low_m_forward(hidden_states, reduce=False)
            if push_shared:
                k31_latent_moe_tail_push_shared(out, fused_tail)
            return out

        shared_stream = self.shared_expert_stream
        if critical is None and _k31_shared_on_priority_stream(self._front_tp, m):
            shared_stream = _k31_moe_priority_stream() or shared_stream
        routed_partial, shared_partial = maybe_execute_in_parallel(
            _routed_output,
            _shared_output,
            self.moe_main_event,
            self.moe_shared_event,
            shared_stream,
            disable_on_compile=True,
        )
        if critical is not None:
            torch.cuda.set_stream(caller)
            caller.wait_stream(critical)
        tail = k31_latent_moe_tail_to_mailbox if to_mailbox else k31_latent_moe_tail
        return tail(
            routed_partial,
            shared_partial,
            self.routed_expert_norm.weight,
            self.routed_expert_up_proj.weight,
            fused_tail,
            shared_pushed=push_shared,
        )

    def _critical_path_stream(self, m: int) -> Optional[torch.cuda.Stream]:
        """The high-priority stream the MoE layer's critical path runs on for ``m`` rows (above the front's rows, at
        ``_K31_MOE_PRIORITY_TP``, under multi-stream execution), else None (the caller's stream)."""
        if (
            m <= _K31_FRONT_FOLD_MAX_ROWS
            or self._front_tp not in _K31_MOE_PRIORITY_TP
            or not do_multi_stream()
            or torch.compiler.is_compiling()
        ):
            return None
        return _k31_moe_priority_stream()

    def _replicated_head(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """The router logits and the routed latent of every row on this rank: one ``k31_moe_head_gemm`` launch at the
        rows it serves (the same bits), else the latent down projection on its own stream beside the router GEMM
        where multi-stream runs (same kernels and inputs either way)."""
        if (
            self._head_weight is not None
            and _k31_head_config(hidden_states.shape[0]) is not None
            and hidden_states.is_contiguous()
            and hidden_states.data_ptr() % 32 == 0
        ):
            return k31_moe_head_gemm(hidden_states, self._head_weight)
        return maybe_execute_in_parallel(
            lambda: self._router_logits(hidden_states),
            lambda: self._routed_projection(hidden_states, self.routed_expert_down_proj),
            self.moe_latent_fork_event,
            self.moe_latent_ready_event,
            self.latent_down_stream,
            disable_on_compile=True,
        )

    def _prologue_folded_routed_output(
        self, routed_in: torch.Tensor, router_logits: torch.Tensor
    ) -> Optional[UnfinalizedRoutedOutput]:
        """Above the front's rows: this rank's routed experts on the MoE prologue's routing (the call the routed
        experts would make), sorted by the CuTe DSL experts themselves with ``pick_n_tile``'s tiles, before the
        top-k weighted sum the fused tail folds; None where the prologue declines the call."""
        routed = self.routed_experts.route(routed_in, router_logits)
        if routed is None:
            return None
        logger.info_once(
            "Kimi K3.1 MoE: CuTe DSL experts above the front's rows, top-k sum folded into the fused tail",
            key="k31_moe_prologue_fold",
        )
        ids, weights, x_fp8, x_sf = routed
        return self._folded_routed_output(
            ids, weights, x_fp8, x_sf, None, n_tile=pick_n_tile(ids.shape[0])
        )

    def _early_routing(self, m: int) -> bool:
        """Whether a call of ``m`` rows issues the shared expert first and routes beside the latent down projection
        (``_K31_MOE_EARLY_ROUTING_TP``): above the front's rows, below the one-launch MoE head's."""
        return (
            _K31_FRONT_FOLD_MAX_ROWS < m < _K31_MOE_HEAD_MIN_ROWS
            and self._front_tp in _K31_MOE_EARLY_ROUTING_TP
            and self._front_fold is not None
            and self._route_quant
        )

    def _early_routed_head(self, hidden_states: torch.Tensor):
        """``(folded, router_logits, routed_in)`` above the front's rows: the router GEMM, the MoE prologue's routing
        half and the routed experts' sort on this stream while the latent down projection runs on its own, then the
        prologue's quantization half of the latent (the launches and bits of ``_prologue_folded_routed_output``'s
        prologue and sort, split); ``folded`` is ``_folded_routed_output``'s arguments, None where the prologue
        declines."""

        def _route():
            logits = self._router_logits(hidden_states)
            routed = prologue_route(
                logits,
                self.gate.e_score_correction_bias,
                self.routed_experts.routed_scaling_factor,
                True,
            )
            if routed is None:
                return logits, None
            ids, weights = routed
            n_tile = pick_n_tile(ids.shape[0])
            prerouted = prerouted_sort(
                ids,
                self.routed_experts.slot_start,
                self.routed_experts.expert_size_per_partition,
                n_tile,
            )
            return logits, (ids, weights, prerouted, n_tile)

        def _latent():
            return self._routed_projection(hidden_states, self.routed_expert_down_proj)

        (router_logits, routing), routed_in = maybe_execute_in_parallel(
            _route,
            _latent,
            self.moe_latent_fork_event,
            self.moe_latent_ready_event,
            self.latent_down_stream,
            disable_on_compile=True,
        )
        folded = None
        if routing is not None:
            quantized = prologue_quant(routed_in)
            if quantized is not None:
                logger.info_once(
                    "Kimi K3.1 MoE: CuTe DSL experts above the front's rows, routing beside the latent projection",
                    key="k31_moe_route_early",
                )
                ids, weights, prerouted, n_tile = routing
                folded = (ids, weights, quantized[0], quantized[1], prerouted, n_tile)
        return folded, router_logits, routed_in

    def _folded_routed_output(
        self, ids, weights, x_fp8, x_sf, prerouted, n_tile: int = _K31_FRONT_FOLD_TILE
    ) -> UnfinalizedRoutedOutput:
        """This rank's routed experts on the front's routing and runner sort (or, without ``prerouted``, their own
        sort with tiles of ``n_tile`` rows), before the top-k weighted sum."""
        if prerouted is not None:
            logger.info_once(
                "Kimi K3.1 MoE front: runner sort with the CuTe DSL experts, top-k sum folded into the fused tail",
                key="k31_moe_front_fold",
            )
        experts = self.routed_experts
        y, expert_weights, expanded_to_permuted = k31_w4a8_moe_cute(
            x_fp8,
            x_sf,
            ids,
            weights,
            experts.w3_w1_weight,
            experts.w3_w1_weight_scale,
            experts.w2_weight,
            experts.w2_weight_scale,
            experts.slot_start,
            experts.expert_size_per_partition,
            n_tile,
            self._front_fold,
            prerouted=prerouted,
            finalize=False,
            gate_cap=self._front_caps[0],
            linear_cap=self._front_caps[1],
            early_release_max_rows=(
                _K31_W4A8_EARLY_RELEASE_MAX_ROWS
                if self._front_tp in _K31_W4A8_EARLY_RELEASE_TP
                else None
            ),
        )
        return UnfinalizedRoutedOutput(
            y, expert_weights, expanded_to_permuted.view(ids.shape[0], _CUTE_MOE_TOP_K)
        )

    def forward_pending(
        self, hidden_states: torch.Tensor, prefix_sum: torch.Tensor
    ) -> Optional[_K31PendingMoE]:
        """The layer output ``prefix_sum + MoE(hidden_states)`` left pending in the fused tail's mailbox, or None
        where the tail does not serve the call (``forward`` then gives the MoE output)."""
        if self._front_w is None or not _k31_mailbox_rows(prefix_sum):
            return None
        mailbox = self._front_forward(hidden_states, to_mailbox=True)
        if mailbox is None:
            return None
        return _K31PendingMoE(prefix_sum, mailbox, self._tail_for(hidden_states.shape[0]))

    def _router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """FP32 router logits."""
        return self.gate.compute_logits(hidden_states)

    def _routed_projection(
        self, hidden_states: torch.Tensor, projection: nn.Module
    ) -> torch.Tensor:
        if (
            isinstance(projection, nn.Linear)
            and hidden_states.dim() == 2
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.stride(-1) == 1
        ):
            if low_m_choice(self._k31_low_m_plan, hidden_states, projection.weight) is not None:
                return k31_low_m_linear(
                    hidden_states, projection.weight, self._k31_low_m_plan, True
                )
            return k31_fused_a_gemm(hidden_states, projection.weight.t())
        return projection(hidden_states)

    @staticmethod
    def _select_moe_tp_ep(mapping: Mapping) -> Tuple[int, int]:
        """Resolve the routed-expert ``(moe_tp, moe_ep)`` split.

        Precedence:

        1. Explicit ``moe_tensor_parallel_size`` / ``moe_expert_parallel_size``
           from the user config. Detected via
           ``mapping.moe_tp_ep_user_specified`` so the auto-resolved mapping
           default (``moe_tp=tp_size, moe_ep=1``) is NOT mistaken for a TP
           request.
        2. Default: EP-only (``moe_tp=1, moe_ep=tp_size``).
        """
        tp_size = mapping.tp_size
        if getattr(mapping, "moe_tp_ep_user_specified", False):
            return mapping.moe_tp_size, mapping.moe_ep_size
        return 1, tp_size

    @staticmethod
    def _resolve_routed_quant_config(model_config: ModelConfig, layer_idx: int) -> QuantConfig:
        """Routed-expert quantization for ``layer_idx``: ``W4A8_MXFP4_MXFP8``, the MXFP4 checkpoint's
        (routing admits only the MXFP4 expert format, ``mxfp4-pack-quantized``; per-layer entries are not read).

        An exclusion outranks that default: ``create_weights`` treats an override as authoritative
        over anything ``__post_init__`` wrote, so this return value stands in for both quantization
        passes and exclusion is the one that runs second. It is matched as a pattern, so it is asked
        only about real module names. ``expert_ckpt_spec`` then refuses the unquantized experts.
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
                "Kimi K3.1 layer %d routed experts: excluded from quantization, "
                "keeping them unquantized",
                layer_idx,
            )
            return QuantConfig(kv_cache_quant_algo=quant_config.kv_cache_quant_algo)

        logger.debug(
            "Kimi K3.1 layer %d routed experts: %s (the MXFP4 checkpoint's routed-expert format)",
            layer_idx,
            _K3_DEFAULT_ROUTED_QUANT_ALGO,
        )
        return QuantConfig(quant_algo=_K3_DEFAULT_ROUTED_QUANT_ALGO)

    @staticmethod
    def _routed_mapping(model_config: ModelConfig) -> Mapping:
        """The routed experts' mapping (a private copy with the MoE TP x EP split; the shared config is not
        mutated). Default split is EP-only; see ``_select_moe_tp_ep``."""
        if model_config.moe_load_balancer is not None:
            raise NotImplementedError(
                "Kimi K3.1 packed-checkpoint streaming does not yet support "
                "EPLB or replicated expert slots."
            )
        mapping = model_config.mapping
        moe_tp, moe_ep = K31MoE._select_moe_tp_ep(mapping)
        if moe_tp < 1 or moe_ep < 1 or moe_tp * moe_ep != mapping.tp_size:
            raise ValueError(
                f"Kimi K3.1 routed MoE split moe_tp={moe_tp} x moe_ep={moe_ep} "
                f"must multiply to tp_size={mapping.tp_size}."
            )
        if moe_tp > 1 and mapping.enable_attention_dp:
            raise NotImplementedError(
                "Kimi K3.1 MoE tensor parallelism requires "
                "enable_attention_dp=false (the attention-DP dispatch/combine "
                "path is validated for EP-only splits)."
            )
        logger.info_once(
            f"Kimi K3.1 routed MoE parallelism: moe_tp={moe_tp}, "
            f"moe_ep={moe_ep} (tp_size={mapping.tp_size})",
            key="kimi_k3_moe_tp_ep_split",
        )

        mapping_dict = mapping.to_dict()
        mapping_dict["moe_cluster_size"] = 1
        mapping_dict["moe_tp_size"] = moe_tp
        mapping_dict["moe_ep_size"] = moe_ep
        return Mapping.from_dict(mapping_dict)

    def forward(
        self, hidden_states: torch.Tensor, all_rank_num_tokens=None
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """``hidden_states``: ``[num_tokens, hidden_size]`` bf16; returns the fused tail's MoE output, else the routed
        and shared outputs it sums."""
        if self._front_w is not None:
            output = self._front_forward(hidden_states)
            if output is not None:
                return output
        identity = hidden_states
        head_in = None
        if (
            self._head_only
            and self._head_weight is not None
            and _k31_head_config(hidden_states.shape[0]) is not None
            and hidden_states.is_contiguous()
            and hidden_states.data_ptr() % 32 == 0
        ):
            router_logits, head_in = k31_moe_head_gemm(hidden_states, self._head_weight)
        else:
            router_logits = self._router_logits(hidden_states)

        def _routed_output():
            """This rank's routed latent partial."""
            routed_in = head_in
            if routed_in is None:
                routed_in = self._routed_projection(hidden_states, self.routed_expert_down_proj)
            return self.routed_experts(routed_in, router_logits)

        def _shared_output():
            """The shared expert's (reduced) output."""
            return self.shared_experts.low_m_forward(identity)

        # Shared experts depend only on the block input, so overlap their GEMMs
        # with the routed dispatch/expert/combine chain. Multi-stream engages
        # only under CUDA graphs; otherwise both branches run in order on the
        # default stream. The shared-expert MLP includes its output all-reduce on
        # the auxiliary stream. The final join includes the shared all-reduce, so it
        # precedes the routed all-reduce: concurrent collectives on different
        # streams can corrupt SYMM_MEM all-reduce state.
        routed_out, shared_out = maybe_execute_in_parallel(
            _routed_output,
            _shared_output,
            self.moe_main_event,
            self.moe_shared_event,
            self.shared_expert_stream,
            disable_on_compile=True,
        )
        # RMSNorm folded into the routed all-reduce (kARRMSNorm): the MNNVL kernel normalizes straight out of the
        # lamport buffer, so the reduced row never round-trips through HBM for a norm kernel.
        routed_latent = allreduce_module(
            routed_out,
            self.routed_experts.all_reduce,
            FUSION_RMS_NORM,
            self.routed_expert_norm.weight,
            float(self.routed_expert_norm.variance_epsilon),
        )
        routed_out = self._routed_projection(routed_latent, self.routed_expert_up_proj)
        return routed_out, shared_out

    @property
    def supports_seq_parallel(self) -> bool:
        """Direct MoE-TP whose routed and shared partials this module reduces itself (see ``forward_seq_parallel``):
        every layout of this target (attention TP over 8 or more ranks)."""
        return True

    def forward_seq_parallel(
        self, hidden_states: torch.Tensor, sp: "_SeqParallelRows"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sequence-parallel MoE block: ``hidden_states`` and the routed and shared outputs it returns (the layer adds
        both to its residual) are this rank's ``[sp.rows, hidden]`` rows.

        The router and the latent down projection run on the local rows; one grouped all-gather brings every rank
        the full normed input (for the TP-sharded shared expert), latent and router logits (for its routed experts),
        and one grouped reduce-scatter sums the routed latent and shared partials back onto the local rows, where
        the latent norm and the up projection run.
        """
        router_logits = self._router_logits(hidden_states)
        latent = self._routed_projection(hidden_states, self.routed_expert_down_proj)
        hidden_full, latent_full, logits_full = sp.gather([hidden_states, latent, router_logits])
        routed_partial = self.routed_experts(latent_full, logits_full)
        shared_partial = self.shared_experts(hidden_full, reduce=False)
        routed_latent, shared_out = sp.reduce_scatter([routed_partial, shared_partial])
        routed_latent = flashinfer_rmsnorm(
            routed_latent,
            self.routed_expert_norm.weight,
            self.routed_expert_norm.variance_epsilon,
        )
        routed_out = self._routed_projection(routed_latent, self.routed_expert_up_proj)
        return routed_out, shared_out


def resolve_attention_quant_config(
    config: ModelConfig | None, layer_idx: int, projection: str
) -> QuantConfig:
    """Resolve a checkpoint projection, including mixed-precision exclusions."""
    if config is None:
        return QuantConfig()
    global_config = config.quant_config or QuantConfig()
    names = [
        f"{prefix}layers.{layer_idx}.self_attn.{projection}"
        for prefix in ("language_model.model.", "model.", "")
    ]
    if any(global_config.is_module_excluded_from_quantization(name) for name in names):
        return QuantConfig(kv_cache_quant_algo=global_config.kv_cache_quant_algo)
    declarations = config.quant_config_dict or {}
    matches = [declarations[name] for name in names if name in declarations]
    if matches:
        selected = matches[0]
        if any(match.quant_algo != selected.quant_algo for match in matches[1:]):
            raise ValueError(f"Conflicting Kimi K3.1 quantization aliases for {names[0]}")
    elif global_config.quant_algo == QuantAlgo.MIXED_PRECISION:
        selected = QuantConfig()
    else:
        selected = global_config
    if selected.quant_algo is not None:
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


class K31MLARuntime(nn.Module):
    """Wraps K3 MLA and applies its external TP output reduction."""

    def __init__(
        self,
        cfg: "PretrainedConfig",
        layer_idx: int,
        model_config: ModelConfig,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ) -> None:
        super().__init__()

        # Identity-RoPE table positions for the MLA backends. K3 is NoPE (the table
        # holds cos=1/sin=0), but the chunked-context path indexes the table by
        # absolute position, so it must cover max_position_embeddings (~512MB per
        # backend for the 1M-position checkpoint); a smaller table is read out of
        # bounds.
        max_positions = int(cfg.max_position_embeddings)
        self.layer_idx = layer_idx
        # K31GatedMLA owns the MLA projection and head sharding. Keep only the
        # final output reduction in this wrapper so the output gate remains
        # between attention and the row-parallel o_proj.
        mapping = model_config.mapping.mapping_for_full_rows()
        reduce_output = not mapping.enable_attention_dp and mapping.tp_size > 1
        self._o_allreduce = (
            tp_allreduce(mapping, model_config.allreduce_strategy, torch.bfloat16)
            if reduce_output
            else None
        )
        attention_config = copy.copy(model_config)
        attention_config._frozen = False
        attention_config.quant_config_dict = {
            name: resolve_attention_quant_config(model_config, layer_idx, name)
            for name in (
                "q_a_proj",
                "kv_a_proj_with_mqa",
                "q_b_proj",
                "kv_b_proj",
                "g_proj",
                "o_proj",
            )
        }
        kv_quant_algo = (
            model_config.quant_config.kv_cache_quant_algo
            if model_config.quant_config is not None
            else None
        )
        attention_config.quant_config = QuantConfig(kv_cache_quant_algo=kv_quant_algo)
        attention_config._frozen = model_config._frozen
        self.mixer = K31GatedMLA(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            q_lora_rank=cfg.q_lora_rank,
            kv_lora_rank=cfg.kv_lora_rank,
            qk_nope_head_dim=cfg.qk_nope_head_dim,
            qk_rope_head_dim=cfg.qk_rope_head_dim,
            v_head_dim=cfg.v_head_dim,
            rms_norm_eps=cfg.rms_norm_eps,
            dtype=torch.bfloat16,
            layer_idx=layer_idx,
            use_output_gate=cfg.mla_use_output_gate,
            max_position_embeddings=max_positions,
            model_config=attention_config,
            aux_stream_dict=aux_stream_dict,
        )
        # The mixer's o_proj pushes its partials through this all-reduce where the push chain serves (a tuple, so
        # the module is not registered a second time as the mixer's submodule).
        self.mixer._k31_o_allreduce = (self._o_allreduce,)
        self.mixer._k31_o_reduced = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        reduce_output: bool = True,
    ) -> torch.Tensor:
        """``reduce_output=False`` (sequence-parallel prefill) returns this rank's o_proj partials unreduced."""
        if not reduce_output:
            # No o_proj push chain either: the caller reduce-scatters the partials.
            push = getattr(self.mixer, "_k31_o_allreduce", (None,))
            self.mixer._k31_o_allreduce = (None,)
            try:
                return self.mixer(None, hidden_states, attn_metadata)
            finally:
                self.mixer._k31_o_allreduce = push
        # K31MLA.forward takes position_ids first; K3.1 is NoPE, so pass None.
        self.mixer._k31_o_reduced = False
        out = self.mixer(None, hidden_states, attn_metadata)
        if isinstance(out, _K31PendingOproj):
            return out
        if self._o_allreduce is not None and not self.mixer._k31_o_reduced:
            # Head-sharded TP: sum the row-sharded o_proj partials across
            # the head-shard group.
            out = allreduce_module(out, self._o_allreduce, FUSION_NONE, None, None)
        return out


# ---------------------------------------------------------------------------
# Sequence-parallel prefill.
# ---------------------------------------------------------------------------

# Under attention TP every rank otherwise runs the attention-residual mixing, the norms, the residual adds, the MoE
# router and the latent down / up projections on all rows of the step. An eager forward of at least
# SEQ_PARALLEL_MIN_TOKENS tokens keeps the residual stream on ``1 / tp`` of the rows per rank instead: each layer
# all-gathers the normed input of its column-parallel projections (the MoE all-gathers the latent and the router
# logits instead of recomputing them) and reduce-scatters the row-parallel outputs it would all-reduce. Not bitwise
# to the replicated forward (other reduction order). CUDA-graph and breakable-graph forwards keep the replicated
# path. Below a few thousand tokens the collectives cost more than the row-wise work they remove.
SEQ_PARALLEL_MIN_TOKENS = 8192


class _SeqParallelRows:
    """Row split of one forward's tokens over the TP group (rank ``r`` owns rows ``[r * rows, (r + 1) * rows)``).

    The split is padded to ``rows * tp_size`` rows; the padding rows of the last ranks start at zero and stay zero
    through every layer (zero rows normalize, route and project to zero), so they never reach a real row.
    """

    def __init__(self, mapping: Mapping, num_tokens: int) -> None:
        self.mapping = mapping
        self.num_tokens = num_tokens
        self.rows = -(-num_tokens // mapping.tp_size)
        self.padded_tokens = self.rows * mapping.tp_size
        self.row_begin = min(mapping.tp_rank * self.rows, num_tokens)
        self.row_end = min(self.row_begin + self.rows, num_tokens)

    def shard(self, full: torch.Tensor) -> torch.Tensor:
        """This rank's rows of a replicated ``[num_tokens, ...]`` tensor, zero-padded to ``rows``."""
        if self.row_end - self.row_begin == self.rows:
            return full[self.row_begin : self.row_end]
        local = full.new_zeros((self.rows,) + tuple(full.shape[1:]))
        local[: self.row_end - self.row_begin].copy_(full[self.row_begin : self.row_end])
        return local

    def gather(self, locals_: List[torch.Tensor]) -> List[torch.Tensor]:
        """All-gather ``[rows, ...]`` tensors into ``[padded_tokens, ...]`` (NVLink peer-store kernels; NCCL for
        tensors they do not serve)."""
        return sp_gather_rows_mnnvl(locals_, self.mapping)

    def reduce_scatter(self, partials: List[torch.Tensor]) -> List[torch.Tensor]:
        """Sum ``[num_tokens or padded_tokens, ...]`` TP partials and keep this rank's ``[rows, ...]``."""
        padded = []
        for x in partials:
            if x.shape[0] != self.padded_tokens:
                pad = x.new_zeros((self.padded_tokens,) + tuple(x.shape[1:]))
                pad[: x.shape[0]].copy_(x)
                x = pad
            padded.append(x)
        return sp_reduce_scatter_rows_mnnvl(padded, self.mapping)


# ---------------------------------------------------------------------------
# Decoder layer.
# ---------------------------------------------------------------------------


def _k31_residual_add_fits(*tensors: torch.Tensor) -> bool:
    """Whether ``k31_moe_residual_add`` takes these operands: contiguous 2D BF16 CUDA tensors of one shape."""
    shape = tensors[0].shape
    return all(
        t.dim() == 2
        and t.dtype is torch.bfloat16
        and t.is_cuda
        and t.is_contiguous()
        and t.shape == shape
        for t in tensors
    )


@nccl_window_tensor_scoped_module
class K31DecoderLayer(nn.Module):
    """One Kimi K3.1 decoder layer: before each sublayer the attention-residual mix of the earlier outputs and an
    RMSNorm, then the token mixer (KDA or MLA attention) and the MLP (dense on layer 0, the latent MoE elsewhere)."""

    # Set by the model on every layer but the last: a fused-tail MoE layer may return its output pending in the
    # tail's mailbox (``_K31PendingMoE``) for the next layer's mix.
    _k31_pending_out = False

    def __init__(
        self,
        model_config: ModelConfig,
        cfg,
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = cfg.hidden_size
        self.tp_size = model_config.mapping.tp_size
        dtype = torch.bfloat16
        self.is_moe = (
            cfg.num_experts is not None
            and layer_idx >= cfg.first_k_dense_replace
            and layer_idx % getattr(cfg, "moe_layer_freq", 1) == 0
        )

        self.is_kda = _is_kda_layer(cfg, layer_idx)
        is_mla = _is_mla_layer(cfg, layer_idx)
        if self.is_kda == is_mla:
            raise ValueError(f"Kimi K3.1 layer {layer_idx} must be exactly one of KDA/MLA")

        if self.is_kda:
            self.linear_attn = K31KDALinearAttention(
                cfg,
                layer_idx,
                mapping=model_config.mapping.mapping_for_full_rows(),
                allreduce_strategy=model_config.allreduce_strategy,
                aux_stream=aux_stream_dict[AuxStreamType.Attention],
                extra_attrs=model_config.extra_attrs,
            )
        else:
            self.self_attn = K31MLARuntime(
                cfg,
                layer_idx,
                model_config=model_config,
                aux_stream_dict=aux_stream_dict,
            )

        if self.is_moe:
            self.block_sparse_moe = K31MoE(model_config, cfg, layer_idx, aux_stream_dict)
        else:
            situ_beta = getattr(cfg, "activation_situ_beta", None) or 1.0
            situ_linear_beta = getattr(cfg, "activation_situ_linear_beta", None)
            mlp_mapping = model_config.mapping.mapping_for_full_rows()
            if mlp_mapping.enable_attention_dp:
                self.mlp_tp_size = 1
            else:
                self.mlp_tp_size = math.gcd(cfg.intermediate_size, mlp_mapping.tp_size)
                if self.mlp_tp_size > mlp_mapping.gpus_per_node:
                    self.mlp_tp_size = math.gcd(self.mlp_tp_size, mlp_mapping.gpus_per_node)
            # K3's dense layer is BF16, so a unit block size gives the same
            # subgroup selection as DeepSeek-V3. Attention DP replicates the
            # MLP because ranks own different tokens; otherwise the subgroup
            # is block-aligned and stays within one node.
            self.mlp = K31SituMLP(
                cfg.hidden_size,
                cfg.intermediate_size,
                dtype=dtype,
                mapping=mlp_mapping,
                beta=situ_beta,
                linear_beta=situ_linear_beta,
                tp_size=self.mlp_tp_size,
                reduce_output=self.mlp_tp_size > 1,
                allreduce_strategy=model_config.allreduce_strategy,
                # Decode gate / up projection on the layout's low-M plan.
                low_m_plan=_k31_decode_plan(model_config.mapping),
            )

        # The plain (whole-tensor) norms: flashinfer_rmsnorm numerics.
        self.input_layernorm = K31RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype)
        self.post_attention_layernorm = K31RMSNorm(
            cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype
        )

        # Attention residual scheme (always on for K3). The res norms are
        # weight / epsilon holders consumed field-wise (.weight /
        # .variance_epsilon) by _apply_attn_res and the fused attn_res op,
        # never called as modules.
        self.attn_res_block_size = cfg.attn_res_block_size
        assert self.attn_res_block_size is not None, (
            "Kimi K3.1 runtime expects attn_res_block_size to be set"
        )
        self.self_attention_res_norm = K31RMSNorm(
            cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype
        )
        self.mlp_res_norm = K31RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype)
        self.self_attention_res_proj = nn.Linear(cfg.hidden_size, 1, bias=False, dtype=dtype)
        self.mlp_res_proj = nn.Linear(cfg.hidden_size, 1, bias=False, dtype=dtype)

    def _mix_pending_moe(
        self, pending: _K31PendingMoE, valid_block_residual: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """``(prefix_sum, normed)`` of the pre-attention mix with the previous layer's MoE add folded in (the mix
        reads the fused tail's mailbox), or None where the mix's operands are outside the kernel."""
        operands = _k31_mix_operands(
            pending.prefix,
            valid_block_residual,
            self.self_attention_res_proj,
            self.self_attention_res_norm,
            self.input_layernorm,
        )
        if operands is None:
            return None
        return k31_attn_res_mailbox_add_rmsnorm(
            pending.prefix, pending.mailbox, valid_block_residual, *operands
        )

    def _capture_hidden_states(self, capture: Tuple[Any, int], value: torch.Tensor) -> None:
        """The drafter's hidden-state tap of ``capture[1]``."""
        capture[0].maybe_capture_hidden_states(capture[1], value, None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        num_snapshots: int,
        attn_metadata: AttentionMetadata,
        capture: Optional[Tuple[Any, int]] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Port of HF ``KimiDecoderLayer._forward_attn_residual`` (per token).

        ``block_residual`` is a preallocated snapshot bank in kernel-native
        ``[K_max, M, H]`` layout. Returns the running prefix sum and the
        number of valid bank rows.

        ``capture`` is ``(spec_metadata, layer_id)`` and taps the drafter's aux
        stream for the layer BEFORE this one: the aggregated stream for layer j
        is by definition what its next consumer sees, so the mixture computed
        below already is it. Reading it here beats recomputing it, and is only
        possible because K3 asserts pp_size == 1 -- layer j+1 is always local.
        PP support would need a recompute at the rank boundary.
        """
        valid_block_residual = block_residual[:num_snapshots]
        mixed = None
        if isinstance(hidden_states, _K31PendingMoE):
            if capture is None:
                mixed = self._mix_pending_moe(hidden_states, valid_block_residual)
            if mixed is None:
                hidden_states = _k31_settle_moe(hidden_states)
        prefix_sum = hidden_states

        if mixed is not None:
            prefix_sum, hidden_states = mixed
        elif capture is not None:
            # The mixture tap needs the PRE-norm value, so the two steps split on captured layers.
            if num_snapshots > 0:
                hidden_states = _apply_attn_res(
                    prefix_sum,
                    valid_block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                )
            # The DFlash2 drafter is distilled on the pre-norm attn_res mixture (hidden_states);
            # a tap of the running prefix would only lower acceptance, silently.
            self._capture_hidden_states(capture, hidden_states)
            hidden_states = flashinfer_rmsnorm(
                hidden_states, self.input_layernorm.weight, self.input_layernorm.variance_epsilon
            )
        elif num_snapshots > 0:
            hidden_states = _apply_attn_res_and_rmsnorm(
                prefix_sum,
                valid_block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.input_layernorm,
            )
        else:
            hidden_states = flashinfer_rmsnorm(
                hidden_states, self.input_layernorm.weight, self.input_layernorm.variance_epsilon
            )

        if self.layer_idx % self.attn_res_block_size == 0:
            block_residual[num_snapshots].copy_(prefix_sum)
            num_snapshots += 1
            valid_block_residual = block_residual[:num_snapshots]
            prefix_sum = None

        if self.is_kda:
            # On a residual-add layer the mix below finishes the o_proj all-reduce.
            self.linear_attn._k31_defer_oproj = _k31_defer_kda_oproj(prefix_sum, self.tp_size)
            try:
                hidden_states = self.linear_attn(hidden_states, attn_metadata)
            finally:
                self.linear_attn._k31_defer_oproj = False
        else:
            # As for KDA: on a residual-add layer of the PDL mix's rows the mix finishes the o_proj all-reduce.
            mixer = self.self_attn.mixer
            mixer._k31_defer_oproj = prefix_sum is not None and _k31_mailbox_rows(prefix_sum)
            try:
                hidden_states = self.self_attn(hidden_states, attn_metadata)
            finally:
                mixer._k31_defer_oproj = False

        if isinstance(hidden_states, _K31PendingOproj):
            pending = hidden_states
            operands = _k31_mix_operands(
                prefix_sum,
                valid_block_residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
                self.post_attention_layernorm,
            )
            if operands is not None:
                prefix_sum, hidden_states = k31_oproj_push_attn_res_add_rmsnorm(
                    pending.input,
                    pending.weight,
                    pending.all_reduce,
                    prefix_sum,
                    valid_block_residual,
                    *operands,
                )
            else:
                prefix_sum, hidden_states = _apply_attn_res_add_and_rmsnorm(
                    prefix_sum,
                    oproj_allreduce_push(pending.input, pending.weight, pending.all_reduce),
                    valid_block_residual,
                    self.mlp_res_proj,
                    self.mlp_res_norm,
                    self.post_attention_layernorm,
                )
        elif prefix_sum is None:
            prefix_sum = hidden_states
            hidden_states = _apply_attn_res_and_rmsnorm(
                prefix_sum,
                valid_block_residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
                self.post_attention_layernorm,
            )
        else:
            prefix_sum, hidden_states = _apply_attn_res_add_and_rmsnorm(
                prefix_sum,
                hidden_states,
                valid_block_residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
                self.post_attention_layernorm,
            )
        if self.is_moe:
            if self._k31_pending_out:
                pending = self.block_sparse_moe.forward_pending(hidden_states, prefix_sum)
                if pending is not None:
                    return pending, num_snapshots
            moe_out = self.block_sparse_moe(
                hidden_states,
                getattr(attn_metadata, "all_rank_num_tokens", None),
            )
            if isinstance(moe_out, tuple):
                routed_out, shared_out = moe_out
                if _k31_residual_add_fits(prefix_sum, routed_out, shared_out):
                    # prefix + (routed + shared) in one kernel, bitwise to the two adds.
                    return k31_moe_residual_add(prefix_sum, routed_out, shared_out), num_snapshots
                moe_out = routed_out + shared_out
            hidden_states = moe_out
        else:
            hidden_states = self.mlp.low_m_forward(hidden_states)

        prefix_sum = prefix_sum + hidden_states
        return prefix_sum, num_snapshots

    @property
    def supports_seq_parallel(self) -> bool:
        """Every row-parallel output of this layer can be handed back unreduced for a reduce-scatter."""
        if self.is_kda:
            mixer_ok = self.linear_attn._o_allreduce is not None
        else:
            mixer_ok = self.self_attn._o_allreduce is not None
        if self.is_moe:
            return mixer_ok and self.block_sparse_moe.supports_seq_parallel
        return mixer_ok

    def forward_seq_parallel(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        num_snapshots: int,
        attn_metadata: AttentionMetadata,
        sp: _SeqParallelRows,
        capture: Optional[Tuple[Any, int]] = None,
    ) -> Tuple[torch.Tensor, int]:
        """``forward`` with the residual stream, ``block_residual`` and every row-wise op on this rank's
        ``[sp.rows, hidden]`` rows (see ``SEQ_PARALLEL_MIN_TOKENS``); the attention mixer and the MLP see all rows
        through an all-gather and return partials that are reduce-scattered.
        """
        prefix_sum = hidden_states
        valid_block_residual = block_residual[:num_snapshots]

        if capture is not None:
            if num_snapshots > 0:
                hidden_states = _apply_attn_res(
                    prefix_sum,
                    valid_block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                )
            self._capture_hidden_states(capture, sp.gather([hidden_states])[0][: sp.num_tokens])
            hidden_states = flashinfer_rmsnorm(
                hidden_states, self.input_layernorm.weight, self.input_layernorm.variance_epsilon
            )
        elif num_snapshots > 0:
            hidden_states = _apply_attn_res_and_rmsnorm(
                prefix_sum,
                valid_block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.input_layernorm,
            )
        else:
            hidden_states = flashinfer_rmsnorm(
                hidden_states, self.input_layernorm.weight, self.input_layernorm.variance_epsilon
            )

        if self.layer_idx % self.attn_res_block_size == 0:
            block_residual[num_snapshots].copy_(prefix_sum)
            num_snapshots += 1
            valid_block_residual = block_residual[:num_snapshots]
            prefix_sum = None
        full = sp.gather([hidden_states])[0][: sp.num_tokens]
        if self.is_kda:
            partial = self.linear_attn(full, attn_metadata, reduce_output=False)
        else:
            partial = self.self_attn(full, attn_metadata, reduce_output=False)
        hidden_states = sp.reduce_scatter([partial])[0]

        if prefix_sum is None:
            prefix_sum = hidden_states
            hidden_states = _apply_attn_res_and_rmsnorm(
                prefix_sum,
                valid_block_residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
                self.post_attention_layernorm,
            )
        else:
            prefix_sum, hidden_states = _apply_attn_res_add_and_rmsnorm(
                prefix_sum,
                hidden_states,
                valid_block_residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
                self.post_attention_layernorm,
            )
        if self.is_moe:
            routed_out, shared_out = self.block_sparse_moe.forward_seq_parallel(hidden_states, sp)
            if _k31_residual_add_fits(prefix_sum, routed_out, shared_out):
                # prefix + (routed + shared) in one kernel, bitwise to the two adds.
                return k31_moe_residual_add(prefix_sum, routed_out, shared_out), num_snapshots
            hidden_states = routed_out + shared_out
        elif self.mlp_tp_size == self.tp_size:
            full = sp.gather([hidden_states])[0]
            partial = self.mlp(full, reduce=False)
            hidden_states = sp.reduce_scatter([partial])[0]
        else:
            # The dense MLP is sharded over a subgroup of the TP ranks (and replicated across subgroups), so its
            # partials do not sum over the whole TP group: run it on all rows and keep this rank's.
            full = self.mlp(sp.gather([hidden_states])[0])
            begin = sp.mapping.tp_rank * sp.rows
            hidden_states = full[begin : begin + sp.rows]

        prefix_sum = prefix_sum + hidden_states
        return prefix_sum, num_snapshots

    def skip_forward(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """No-op stand-in for ``forward``, matching ``DecoderLayer.skip_forward``.

        ``modeling_utils.skip_forward()`` only drops a module's weights when it
        finds this attribute, so without it the layer-wise benchmarks would
        allocate all 93 layers instead of the profiled slice.
        """
        return hidden_states, block_residual


# ---------------------------------------------------------------------------
# Model.
# ---------------------------------------------------------------------------


#: Generation rows whose attention-residual mixes release their dependents right after ``griddepcontrol.wait`` (the
#: PDL mix's early trigger) on SM 10.3 attention-TP generation layouts: every row count the PDL mixes serve. The next
#: projection or MoE head GEMM becomes resident and streams its weights while the mix finishes (GB300 fixed batch, w5,
#: per step: TEP16 100-140 rows -0.19..-0.22 ms, TEP8 30-120 rows -0.09..-0.19 ms, with the mixes' register-capped
#: builds). The release point does not change any result.
_K31_MIX_EARLY_TRIGGER_ROWS = _K31_MIX_PERSISTENT_ROWS - 1


def _k31_set_mix_early_trigger() -> None:
    """Enable the PDL mixes' early trigger up to ``_K31_MIX_EARLY_TRIGGER_ROWS`` rows (this target's SM 10.3
    attention-TP generation layouts, TEP8 / TEP16 / TEP32); the kernel's process-wide row limit (its safety argument
    covers the decode graph of these layouts)."""
    set_early_trigger_max_rows(_K31_MIX_EARLY_TRIGGER_ROWS)


#: Generation rows from which the PDL attn_res mixes with five or more candidates take their register-capped build
#: (``attn_res.reg_cap``) on SM 10.3 attention-TP generation layouts: every cluster of up to 184 tokens stays
#: resident, so no second wave runs its prologue after the wait (GB300 TEP16, mix time after its producer at 120 rows
#: 1.10 -> 0.87 ms per step).
_K31_MIX_REG_CAP_ROWS = 65


def _k31_set_mix_reg_cap() -> None:
    """Enable the PDL mixes' register-capped builds from ``_K31_MIX_REG_CAP_ROWS`` rows (this target's SM 10.3
    attention-TP generation layouts, TEP8 / TEP16 / TEP32); the kernel's process-wide row limit."""
    set_reg_cap_min_rows(_K31_MIX_REG_CAP_ROWS)


class K31TextModel(DecoderModel):
    """The Kimi K3.1 text model: the token embedding, the decoder layers with their attention-residual bank, and the
    final RMSNorm. Under DFlash2 it hands the drafter the hidden states of the layers the drafter reads."""

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        cfg = text_config_of(model_config.pretrained_config)
        dtype = torch.bfloat16

        # Attention and MoE phases are sequential, so their branch-overlap roles share one stream; the routed
        # experts' chunk overlap and the MoE latent down projection have their own.
        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]
        self.aux_stream_dict = {
            AuxStreamType.Attention: aux_stream_list[0],
            AuxStreamType.MoeShared: aux_stream_list[0],
            AuxStreamType.MoeChunkingOverlap: aux_stream_list[1],
            AuxStreamType.MoeOutputMemset: aux_stream_list[2],
        }

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                K31DecoderLayer(model_config, cfg, layer_idx, self.aux_stream_dict)
                for layer_idx in range(cfg.num_hidden_layers)
            ]
        )
        self.norm = K31RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype)

        # Consumed field-wise (.weight / .variance_epsilon) by _apply_attn_res and the fused attn_res op.
        self.output_attn_res_norm = K31RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype)
        self.output_attn_res_proj = nn.Linear(cfg.hidden_size, 1, bias=False, dtype=dtype)
        self.num_attn_res_snapshots = (
            cfg.num_hidden_layers + cfg.attn_res_block_size - 1
        ) // cfg.attn_res_block_size
        _k31_set_mix_early_trigger()
        _k31_set_mix_reg_cap()
        enable_gvr_early_release(model_config.mapping)
        for layer in self.layers[:-1]:
            layer._k31_pending_out = True

        # Sequence-parallel prefill (see SEQ_PARALLEL_MIN_TOKENS): attention TP without attention DP.
        mapping = model_config.mapping
        self._seq_parallel_mapping = mapping
        self._seq_parallel = (
            mapping.tp_size > 1
            and not mapping.enable_attention_dp
            and mapping.pp_size == 1
            and all(layer.supports_seq_parallel for layer in self.layers)
        )
        self._seq_parallel_max_tokens = int(model_config.max_num_tokens or 0)

    def post_load_weights(self) -> None:
        """Reserve the sequence-parallel collectives' workspace for the largest step, on every rank of the TP group
        at model build (before the KV cache takes the free memory), where a step can reach SEQ_PARALLEL_MIN_TOKENS and
        the peer-store kernels serve the group (larger groups run the NCCL collectives)."""
        if (
            self._seq_parallel
            and self._seq_parallel_max_tokens >= SEQ_PARALLEL_MIN_TOKENS
            and self._seq_parallel_mapping.tp_size <= SP_MNNVL_MAX_RANKS
        ):
            reserve_sp_workspace(self._seq_parallel_mapping, self._seq_parallel_max_tokens)

    def _use_seq_parallel(self, attn_metadata: AttentionMetadata, num_tokens: int) -> bool:
        """Whether this forward runs the sequence-parallel residual stream: an eager step of at least
        SEQ_PARALLEL_MIN_TOKENS tokens. A step a CUDA graph covers (its replay, its warmups, an eager run of a
        bucket padded for one) keeps the replicated path, so warmup and capture see one path."""
        return (
            self._seq_parallel
            and num_tokens >= SEQ_PARALLEL_MIN_TOKENS
            and not attn_metadata.is_cuda_graph
            and not is_in_breakable_cuda_graph()
            and attn_metadata.padded_num_tokens is None
            and not get_per_request_prefill_cuda_graph_flag()
            and not torch.compiler.is_compiling()
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
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        if self._use_seq_parallel(attn_metadata, hidden_states.shape[0]):
            return self._forward_seq_parallel(
                attn_metadata,
                hidden_states,
                _SeqParallelRows(self._seq_parallel_mapping, hidden_states.shape[0]),
                spec_metadata,
            )
        return self._forward_layers(attn_metadata, hidden_states, spec_metadata)

    def _forward_seq_parallel(
        self,
        attn_metadata: AttentionMetadata,
        inputs_embeds: torch.Tensor,
        sp: _SeqParallelRows,
        spec_metadata,
    ) -> torch.Tensor:
        """``forward`` with the residual stream on this rank's rows (``SEQ_PARALLEL_MIN_TOKENS``); the final norm
        output is all-gathered back to ``[num_tokens, hidden]``."""
        hidden_states = sp.shard(inputs_embeds)
        block_residual = hidden_states.new_empty(
            self.num_attn_res_snapshots, sp.rows, hidden_states.shape[1]
        )
        num_snapshots = 0
        capture_set = (
            getattr(spec_metadata, "_capture_layer_set", None)
            if spec_metadata is not None
            else None
        )
        for i, layer in enumerate(self.layers):
            capture = None
            if (
                spec_metadata is not None
                and i > 0
                and (capture_set is None or self.layers[i - 1].layer_idx in capture_set)
            ):
                capture = (spec_metadata, self.layers[i - 1].layer_idx)
            hidden_states, num_snapshots = layer.forward_seq_parallel(
                hidden_states, block_residual, num_snapshots, attn_metadata, sp, capture=capture
            )

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
                    if num_snapshots > 0
                    else hidden_states
                )
                spec_metadata.maybe_capture_hidden_states(
                    last.layer_idx, sp.gather([tail])[0][: sp.num_tokens], None
                )

        out = _apply_attn_res_and_rmsnorm(
            hidden_states,
            block_residual[:num_snapshots],
            self.output_attn_res_proj,
            self.output_attn_res_norm,
            self.norm,
        )
        return sp.gather([out])[0][: sp.num_tokens]

    def _forward_layers(
        self,
        attn_metadata: AttentionMetadata,
        hidden_states: torch.Tensor,
        spec_metadata,
    ) -> torch.Tensor:
        """``forward`` past the embedding."""
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
            # consumer sees -- not on the raw prefix sum a layer returns. The
            # tap fires inside layer i+1, which computes that tensor anyway;
            # see its forward docstring.
            capture = None
            if (
                spec_metadata is not None
                and i > 0
                and (capture_set is None or self.layers[i - 1].layer_idx in capture_set)
            ):
                capture = (spec_metadata, self.layers[i - 1].layer_idx)
            hidden_states, num_snapshots = layer(
                hidden_states,
                block_residual,
                num_snapshots,
                attn_metadata,
                capture=capture,
            )
        if isinstance(hidden_states, _K31PendingMoE):
            hidden_states = _k31_settle_moe(hidden_states)

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
                    if num_snapshots > 0
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


class K31ForCausalLM(DecoderModelForCausalLM[K31TextModel, Any]):
    """The Kimi K3.1 text model (KDA + MLA + MoE) with its LM head: the text backbone (``llm``) of the
    vision-language wrapper :class:`K31VLWrapper` (below), the class routing sends the checkpoint's architecture to.

    Under one-engine DFlash it also holds the DFlash2 drafter (:class:`DraftModel`) and the DFlash worker that
    verifies the drafted block and drafts the next one after the target forward."""

    mamba_metadata_cls = KdaMetadata

    def __init__(self, model_config: ModelConfig):
        cfg = text_config_of(model_config.pretrained_config)
        assert model_config.mapping.pp_size == 1, "Kimi K3.1 does not support pipeline parallelism"
        assert not model_config.mapping.has_cp_helix(), (
            "Kimi K3.1 does not support context parallelism"
        )
        spec_config = getattr(model_config, "spec_config", None)
        # Speculative decoding: one-engine DFlash with the DFlash2 MLA drafter. The target only
        # exposes per-layer hidden states via maybe_capture_hidden_states (K31TextModel.forward)
        # and verifies the drafted block (KDA and MLA multi-token verify).
        assert spec_config is None or spec_config.spec_dec_mode.is_dflash(), (
            "Kimi K3.1 supports speculative decoding only with DFlash (DFlash2 MLA drafter)"
        )
        super().__init__(
            K31TextModel(model_config),
            config=model_config,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )
        self.spec_config = spec_config
        self.draft_config = None
        self.draft_model = None
        self.spec_worker = None
        self.use_separate_draft_kv_cache = False
        # The capture slot of the final hidden states in the spec metadata.
        self.layer_idx = -1
        if spec_config is not None:
            self._build_dflash2(model_config, spec_config)

    def _build_dflash2(self, model_config: ModelConfig, spec_config) -> None:
        """The drafter's config (its checkpoint under the target's execution layout and KV dtype), the DFlash2
        drafter and the DFlash worker, appended to the epilogue in that order. The drafted block is
        ``max_draft_len + 1`` wide whatever the checkpoint's ``block_size``."""
        draft_config = dflash2_draft_config(model_config, spec_config)
        self.draft_config = draft_config
        self.use_separate_draft_kv_cache = should_use_separate_draft_kv_cache(spec_config)
        draft_model = DraftModel(
            draft_config,
            block_size=spec_config.max_draft_len + 1,
            dflash_attention_backend=spec_config.attention_backend,
            context_kv_only=dflash2_context_kv_only(spec_config),
        )
        draft_model.logits_processor = self.logits_processor
        self.draft_model = draft_model
        self.epilogue.append(draft_model)
        spec_worker = K31DFlashWorker(
            spec_config,
            model_config.mapping.mapping_for_full_rows(),
            self.use_separate_draft_kv_cache,
        )
        spec_worker.set_draft_model(draft_model)
        spec_worker.set_target_model(self)
        self.spec_worker = spec_worker
        self.epilogue.append(spec_worker)
        logger.info(
            f"K3.1 DFlash2: drafter {type(draft_model).__name__} (block {draft_model.block_size}) and worker "
            f"{type(spec_worker).__name__}"
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        spec_metadata=None,
        resource_manager=None,
        **kwargs,
    ) -> torch.Tensor:
        """The target forward, then under DFlash the worker: the target logits in its acceptance layout, the
        verification of the drafted block and the next draft."""
        hidden_states = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            **kwargs,
        )
        if spec_metadata is not None and spec_metadata.is_layer_capture(self.layer_idx):
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states)
        if attn_metadata.padded_num_tokens is not None:
            hidden_states = hidden_states[: attn_metadata.num_tokens]
        if self.spec_worker is None:
            return self.logits_processor.forward(
                hidden_states,
                self.lm_head,
                attn_metadata,
                return_context_logits,
            )
        # The target logits, in the layout the worker's acceptance reads.
        logits = self.spec_worker.target_logits(
            hidden_states[spec_metadata.gather_ids],
            self.lm_head,
            self.logits_processor,
            attn_metadata,
            spec_metadata,
            self.draft_model,
        )
        # A VLM wrapper that fuses its inputs into inputs_embeds passes the prompt tokens as orig_input_ids.
        spec_input_ids = input_ids if input_ids is not None else kwargs.get("orig_input_ids")
        spec_position_ids = position_ids
        if attn_metadata.padded_num_tokens is not None:
            if spec_input_ids is not None:
                spec_input_ids = spec_input_ids[: attn_metadata.num_tokens]
            if position_ids is not None:
                spec_position_ids = position_ids[..., : attn_metadata.num_tokens]
        # The accepted tokens and the next draft tokens.
        return self.spec_worker(
            input_ids=spec_input_ids,
            position_ids=spec_position_ids,
            hidden_states=hidden_states,
            logits=logits,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=self.draft_model,
            resource_manager=resource_manager,
        )

    def load_draft_weights(self, weights: Dict, weight_mapper=None) -> None:
        """The drafter checkpoint into :class:`DraftModel`, then the target's embedding and LM head shared with it."""
        self.draft_model.load_weights(weights=weights, weight_mapper=weight_mapper)
        self.draft_model.load_weights_from_target_model(self)

    def set_guided_decoder(self, guided_decoder) -> bool:
        if hasattr(self.spec_worker, "set_guided_decoder"):
            return self.spec_worker.set_guided_decoder(guided_decoder)
        return False

    @classmethod
    def get_model_defaults(cls, llm_args) -> dict:
        # - Block reuse is opt-in: with it on, the cache manager also keeps the KDA recurrent-state snapshots of
        #   reusable prefixes (``kv_cache_config.mamba_state_config``) and their partial attention pages, a cost only
        #   workloads with shared prefixes pay back.
        # - 64-token pages, the page size the targets' kernels are tuned and validated at (routing also admits 32 and
        #   128).
        # - moe_config.backend TRTLLM: the routed experts run on TRTLLM-Gen
        #   (the packed MXFP4 expert loader fills the routed experts' runner-layout weights).
        # - Shortest-uncached-first context admission: under load a short prompt is not queued behind long prefills,
        #   which bounds the time to first token (the KV cache manager V2 scheduler K3.1 runs on supports it).
        # - DSA sparse attention with the checkpoint's indexer (``config.dsa_model_defaults``).
        return {
            "kv_cache_config": {
                "enable_block_reuse": False,
                "tokens_per_block": 64,
            },
            "moe_config": {"backend": "TRTLLM"},
            "scheduler_config": {"context_scheduling_policy": "shortest_uncached_first"},
            **dsa_model_defaults(),
        }

    @classmethod
    def requires_kv_cache_manager_v2(cls, pretrained_config: Any = None) -> bool:
        """Kimi K3.1 DSA keeps its indexer K-cache in a KV cache manager V2 buffer (every routed config has a
        ``dsa_config``)."""
        if pretrained_config is None:
            return False
        return getattr(text_config_of(pretrained_config), "dsa_config", None) is not None

    @classmethod
    def get_preferred_transceiver_runtime(
        cls,
        pretrained_config: Any = None,
    ) -> Literal["PYTHON"]:
        """Kimi K3.1 disaggregated serving requires the Python transceiver.

        Only the Python NIXL transceiver (KvCacheTransceiverV2) can move the KDA recurrent state; the C++
        transceiver has no KDA support. Adopted when the user leaves ``cache_transceiver_config.transceiver_runtime``
        at 'auto'; routing refuses an explicit non-Python runtime or a backend other than NIXL.
        """
        return "PYTHON"


# ====================================================================================================================
# DFlash drafter model
# ====================================================================================================================


def _drafter_mlp_down_input(mlp: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The input of the drafter MLP's ``down_proj``: ``gate_up_proj`` (on the drafter's low-M plan where it serves),
    then the SwiGLU."""
    h1 = _k31_planned_linear(x, mlp.gate_up_proj.weight, mlp._k31_low_m_plan)
    if h1 is None:
        h1 = mlp.gate_up_proj(x)
    return silu_and_mul(h1)


class _DrafterSublayer(NamedTuple):
    """A drafter sublayer's output ahead of its epilogue (block convolution, residual add, RMSNorm): the row-parallel
    projection's input and module where its all-reduce is a push chain the epilogue finishes (``linear`` set), else
    the projection's all-reduced output."""

    conv: nn.Module
    x: torch.Tensor
    coefficients: torch.Tensor
    linear: Optional[nn.Module] = None


def _drafter_sublayer(
    linear: nn.Module, x: torch.Tensor, conv, coefficients, plain
) -> _DrafterSublayer:
    """``linear(x)`` (``plain()`` where no push chain serves), left to the epilogue where it is a push chain."""
    all_reduce = linear.all_reduce
    if all_reduce is not None and push_serves(x, linear.weight, all_reduce):
        return _DrafterSublayer(conv, x, coefficients, linear)
    return _DrafterSublayer(conv, plain(), coefficients)


def _drafter_mlp_sublayer(mlp: nn.Module, x: torch.Tensor, conv, coefficients) -> _DrafterSublayer:
    """The drafter MLP ahead of its epilogue: ``down_proj`` and its all-reduce pushed where served."""
    h = _drafter_mlp_down_input(mlp, x)
    sub = _drafter_sublayer(mlp.down_proj, h, conv, coefficients, lambda: mlp.down_proj(h))
    if push_serves(h, mlp.down_proj.weight, mlp.down_proj.all_reduce):
        logger.info_once(
            f"DFlash2 drafter MLP down: push GEMM + Lamport all-reduce (K={h.shape[-1]}).",
            key=f"k31_dflash2_mlp_down_push_{h.shape[-1]}",
        )
    return sub


def _conv_epilogue_serves(conv, x, coefficients, residual, norm) -> bool:
    """Whether the one-launch epilogue (``k31_dflash2_conv_add_rmsnorm``) takes these operands (``x`` the all-reduced
    rows or, before its all-reduce, any tensor of their shape and dtype)."""
    base = conv.base_kernel[1]
    return (
        x.dim() == 2
        and x.dtype == base.dtype == coefficients.dtype == norm.weight.dtype
        and x.is_contiguous()
        and base.is_contiguous()
        and residual.is_contiguous()
        and coefficients.shape == (x.shape[0], base.shape[0], x.shape[1] // conv.group_size)
        and coefficients.stride(-1) == 1
    )


def _sublayer_epilogue(sub: _DrafterSublayer, block_size, residual, norm):
    """The sublayer's epilogue on ``residual`` / ``norm``: with a pending push chain, the push GEMM and the epilogue
    that finishes its all-reduce (``k31_dflash2_push_conv_add_rmsnorm``)."""
    if sub.linear is None:
        return _conv_add_rmsnorm(sub.conv, sub.x, sub.coefficients, block_size, residual, norm)
    if _conv_epilogue_serves(sub.conv, residual, sub.coefficients, residual, norm):
        logger.info_once(
            "DFlash2 drafter sublayer epilogue finishes the pushed all-reduce "
            "(kimi_k31_pro_mxfp4__sm_103__tep/ops/spec/dflash2_push_conv_add_rmsnorm).",
            key="k31_dflash2_ar_epilogue",
        )
        return k31_dflash2_push_conv_add_rmsnorm(
            sub.x,
            sub.linear.weight,
            sub.linear.all_reduce,
            sub.coefficients,
            sub.conv.base_kernel[1],
            block_size,
            sub.conv.group_size,
            residual,
            norm.weight,
            norm.variance_epsilon,
        )
    x = oproj_allreduce_push(sub.x, sub.linear.weight, sub.linear.all_reduce)
    return _conv_add_rmsnorm(sub.conv, x, sub.coefficients, block_size, residual, norm)


def _conv_add_rmsnorm(conv, x, coefficients, block_size, residual, norm):
    """``conv.finish(x)`` then the fused residual add + RMSNorm, in one launch where the operands allow it."""
    base = conv.base_kernel[1]
    if _conv_epilogue_serves(conv, x, coefficients, residual, norm):
        return k31_dflash2_conv_add_rmsnorm(
            x,
            coefficients,
            base,
            block_size,
            conv.group_size,
            residual,
            norm.weight,
            norm.variance_epsilon,
        )
    return flashinfer_fused_add_rmsnorm(
        conv.finish(x, coefficients, block_size), residual, norm.weight, norm.variance_epsilon
    )


class _K31SideStream:
    """A worker side stream and its fork / join events."""

    def __init__(self):
        self.stream = torch.cuda.Stream()
        self.fork = torch.cuda.Event()
        self.join = torch.cuda.Event()


def _k31_plain_embedding(embed_tokens: nn.Module) -> bool:
    """Whether ``embed_tokens(ids)`` is the row lookup ``weight[ids]`` of an unsharded table: a torch
    ``nn.Embedding`` without max_norm (the target's table, which the drafter shares)."""
    return type(embed_tokens) is nn.Embedding and embed_tokens.max_norm is None


class K31BlockStore(NamedTuple):
    """The block decode's page-table rows and block store addresses, built with the drafter inputs
    (``kimi_k31_pro_mxfp4__sm_103__tep/ops/spec/dflash2_gen_prepare``) and handed to ``dflash_forward`` as its
    ``block_placement``."""

    page_tables: torch.Tensor  # [B, pages] int64
    store_pages: torch.Tensor  # [B * block] int64
    store_offsets: torch.Tensor  # [B * block] int64


# ====================================================================================================================
# DFlash2 MLA drafter
# ====================================================================================================================

# ---------------------------------------------------------------------------
# The drafter's weight holders, DraftModelBase and the drafter config.
# ---------------------------------------------------------------------------
#
# The drafter is five MLA layers that decode a block with an exact causal sliding window, the projection of the captured
# target features (``fc`` + ``hidden_norm``), the DFlash2 block convolutions around every sublayer and the candidate
# selector; the embedding and the LM head are the target's. This part holds:
#
# - the weight holders (``DraftMLA``, ``DraftMLP``, ``DraftDecoderLayer``, ``DraftBackbone``), with the engine layers'
#   parameter names and tensor-parallel shards;
# - ``DraftModelBase``, the drafter's construction, weight loading, candidate selection, YaRN table, window attention
#   and context-latent buffers;
# - the drafter config of the causal LM (``dflash2_draft_config``) and its checks.
#
# ``DraftModel(DraftModelBase)`` below adds what this target runs: its low-M GEMM plan, the projection of the target
# features, the context-latent projection and store, and the block decode (``dflash_forward``).


#: The selector rank the CuTe DSL selector ops take in whole blocks of.
_SELECTOR_RANK_BLOCK = 256


def dflash2_draft_config(model_config: ModelConfig, spec_config) -> ModelConfig:
    """The drafter's ``ModelConfig``: its checkpoint under the target's execution layout (every row of the TP
    group), the target's KV-cache dtype and the target's shared ``extra_attrs``."""
    draft_config = ModelConfig.from_pretrained(
        spec_config.speculative_model,
        trust_remote_code=True,
        attn_backend=model_config.attn_backend,
        moe_backend=(
            model_config.moe_backend
            if getattr(spec_config, "moe_backend", None) is None
            else spec_config.moe_backend
        ),
        mapping=model_config.mapping.mapping_for_full_rows(),
        spec_config=None,
        max_num_tokens=model_config.max_num_tokens,
        moe_max_num_tokens=model_config.moe_max_num_tokens,
        # The user's value: the drafter's position table follows the positions the worker serves.
        max_seq_len=model_config.max_seq_len,
    )
    draft_config.quant_config.kv_cache_quant_algo = model_config.quant_config.kv_cache_quant_algo
    draft_config.extra_attrs = model_config.extra_attrs
    return draft_config


class DraftMLA(nn.Module):
    """Parameter holder; not a layer.

    One drafter layer's MLA projections and norms (``DraftModel.dflash_forward`` runs them).

    The ``*_a`` projections are replicated; ``q_b_proj`` / ``kv_b_proj`` are column-sharded by head and ``o_proj``
    row-sharded over the attention TP group, which attention DP collapses to one rank. ``context_kv_only`` keeps
    only the latent projection and its norm (the other names stay, as None).
    """

    def __init__(self, model_config: ModelConfig, context_kv_only: bool = False):
        super().__init__()
        config = model_config.pretrained_config
        dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        # What the drafter's KV cache stores per token: one MLA latent, no V.
        self.kv_cache_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        mapping = model_config.mapping
        tp_size, dp_size = mapping.tp_size, 1
        if mapping.enable_attention_dp:
            tp_size, dp_size = 1, mapping.tp_size
        self.num_heads_tp = self.num_heads // tp_size
        self.mapping = Mapping(
            world_size=mapping.pp_size * dp_size * tp_size,
            tp_size=tp_size,
            pp_size=mapping.pp_size * dp_size,
            rank=mapping.rank,
            gpus_per_node=mapping.gpus_per_node,
            enable_attention_dp=mapping.enable_attention_dp,
        )
        self.kv_a_proj_with_mqa = K31Linear(self.hidden_size, self.kv_cache_head_dim, dtype=dtype)
        self.kv_a_layernorm = K31RMSNorm(self.kv_lora_rank, config.rms_norm_eps, dtype=dtype)
        if context_kv_only:
            self.q_a_proj = self.q_a_layernorm = self.q_b_proj = self.kv_b_proj = self.o_proj = None
            return
        self.q_a_proj = K31Linear(self.hidden_size, self.q_lora_rank, dtype=dtype)
        self.q_a_layernorm = K31RMSNorm(self.q_lora_rank, config.rms_norm_eps, dtype=dtype)
        self.q_b_proj = K31Linear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            dtype=dtype,
            mapping=self.mapping,
            tp_mode="column",
        )
        self.kv_b_proj = K31Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            dtype=dtype,
            mapping=self.mapping,
            tp_mode="column",
        )
        # The row-parallel o_proj sums over the group only when the heads are split.
        self.o_proj = K31Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            dtype=dtype,
            mapping=self.mapping,
            tp_mode="row",
            reduce_output=not mapping.enable_attention_dp and mapping.tp_size > 1,
            allreduce_strategy=model_config.allreduce_strategy,
        )


class DraftMLP(nn.Module):
    """Parameter holder; not a layer.

    One drafter layer's SwiGLU MLP weights: ``gate_up_proj`` holds this rank's ``[gate | up]`` rows
    (column-parallel), ``down_proj`` the matching input columns (row-parallel, summed over the group). Attention DP
    replicates it (a one-rank group)."""

    # The drafter's low-M GEMM plan (set with its packed projections); None keeps gate_up_proj's GEMM.
    _k31_low_m_plan = None

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        config = model_config.pretrained_config
        mapping = model_config.mapping
        if mapping.enable_attention_dp:
            # One-rank subgroups, the outer factor on the pipeline axis.
            mapping = Mapping(
                world_size=mapping.pp_size * mapping.tp_size,
                rank=mapping.rank,
                gpus_per_node=mapping.gpus_per_node,
                tp_size=1,
                pp_size=mapping.pp_size * mapping.tp_size,
                enable_attention_dp=mapping.enable_attention_dp,
            )
        self.gate_up_proj = K31Linear(
            config.hidden_size,
            2 * config.intermediate_size,
            dtype=config.torch_dtype,
            mapping=mapping,
            tp_mode="column",
        )
        self.down_proj = K31Linear(
            config.intermediate_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            mapping=mapping,
            tp_mode="row",
            allreduce_strategy=model_config.allreduce_strategy,
        )


class DraftDecoderLayer(nn.Module):
    """Parameter holder; not a layer.

    One drafter layer's weights: MLA projections, SwiGLU MLP and the two norms (``context_kv_only``: the latent
    projection and its norm alone)."""

    def __init__(self, model_config: ModelConfig, context_kv_only: bool = False):
        super().__init__()
        config = model_config.pretrained_config
        self.self_attn = DraftMLA(model_config, context_kv_only)
        if context_kv_only:
            self.mlp = self.input_layernorm = self.post_attention_layernorm = None
            return
        self.mlp = DraftMLP(model_config)
        self.input_layernorm = K31RMSNorm(
            config.hidden_size, config.rms_norm_eps, dtype=config.torch_dtype
        )
        self.post_attention_layernorm = K31RMSNorm(
            config.hidden_size, config.rms_norm_eps, dtype=config.torch_dtype
        )


class DraftBackbone(nn.Module):
    """Parameter holder; not a layer.

    The drafter's layers and final norm. ``embed_tokens`` is the target's (set when the target's weights are shared
    with the drafter); a context-only drafter has neither it nor the final norm."""

    def __init__(self, model_config: ModelConfig, context_kv_only: bool = False):
        super().__init__()
        config = model_config.pretrained_config
        self.embed_tokens = None
        self.layers = nn.ModuleList(
            DraftDecoderLayer(model_config, context_kv_only)
            for _ in range(config.num_hidden_layers)
        )
        self.norm = (
            None
            if context_kv_only
            else K31RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=config.torch_dtype)
        )


class DraftModelBase(nn.Module):
    """The DFlash2 drafter of Kimi K3.1, less the parts a target runs its own way.

    The worker (the target's ``K31DFlashWorker``) stores the context latents (``precompute_context_kv``) in a paged
    pool, runs ``dflash_forward`` and picks the draft tokens through the candidate selector; the attributes below are
    the ones it and the engine ``DFlashWorker`` it subclasses read. A target's ``DraftModel`` adds
    ``precompute_context_kv`` and ``dflash_forward``, and may replace :meth:`project_target_hidden` and
    :meth:`_k31_pack_block_projections`.
    """

    # The block's draft tokens sit in slots 1..K, not 0..K-1 (no DSpark label shift).
    shift_label = False

    # One latent per token and layer (no V) in the worker's context cache.
    _kv_factor = 1
    _num_kv_heads = 1
    # Rows per slot of the worker's context ring (ops/spec/dflash2_ring), which it sets when it sizes
    # its private arena by _ctx_ring_window; None reads the table densely by position.
    _ctx_ring_rows = None
    # The context latents of each request's last window also live in the KV cache manager's draft
    # pages (the unified draft cache): the carrier between disaggregated servers and the store reused
    # with a conversation (ops/spec/dflash2_draft_pages). The block decode reads the worker's ring.

    # The positions the worker serves, published before the first forward; None keeps the config's table length.
    _runtime_position_ceiling = None
    # Checkpoint tensors the drafter builds outside its backbone, and requires.
    WRAPPER_OWNED_WEIGHTS = ("fc.weight", "hidden_norm.weight")

    def __init__(
        self,
        draft_config: ModelConfig,
        *,
        block_size: int,
        dflash_attention_backend: str = "AUTO",
        context_kv_only: bool = False,
    ):
        super().__init__()
        cfg = draft_config.pretrained_config
        if dflash2_attention_mode(cfg) != "mla":
            raise ValueError(
                "The Kimi K3.1 drafter serves DFlash2 drafters with dflash_config.attention_mode "
                f"'mla', got {dflash2_attention_mode(cfg)!r}."
            )
        window = dflash2_sliding_window(cfg)
        # The block decode is the drafter's own (the engine worker's attention op sets do not run it): VANILLA, the
        # only backend name routing admits, also keeps the engine from building a separate draft KV manager.
        if dflash_attention_backend == "AUTO":
            dflash_attention_backend = "VANILLA"
        self.dflash_attention_backend = dflash_attention_backend
        model_config = replace(draft_config, spec_config=None, lm_head_gather_output=False)
        # ModelConfig.extra_attrs is init=False, so replace() does not carry the shared custom-op registries.
        model_config.extra_attrs = draft_config.extra_attrs
        self.model_config = model_config
        self.config = model_config.pretrained_config
        dflash_config = getattr(cfg, "dflash_config", None) or {}
        self.mask_token_id = dflash_config.get(
            "mask_token_id", getattr(cfg, "mask_token_id", cfg.vocab_size)
        )
        # The configured width (max_draft_len + 1) replaces the checkpoint's block_size for the
        # draft block, the convolutions and the worker.
        self.block_size = int(block_size)
        # DFlash 2: the block convolutions and the candidate selector, built from the checkpoint at load.
        self._dflash2_conv_taps = int(dflash_config.get("conv_kernel_size", 0) or 0)
        self._dflash2_conv_group_size = int(dflash_config.get("conv_group_size", 0) or 0)
        self._dflash2_selector_rank = int(dflash_config.get("selector_rank", 0) or 0)
        self._dflash2_selector_top_k = int(dflash_config.get("selector_top_k", 0) or 0)
        if (
            not is_dflash2_checkpoint(getattr(cfg, "architectures", None), dflash_config)
            or min(
                self._dflash2_conv_taps,
                self._dflash2_conv_group_size,
                self._dflash2_selector_rank,
                self._dflash2_selector_top_k - 1,
            )
            < 1
        ):
            raise ValueError(
                "The Kimi K3.1 drafter needs a DFlash2 checkpoint: dflash_config conv_kernel_size, "
                "conv_group_size, selector_rank >= 1 and selector_top_k >= 2."
            )
        self.attention_convs = None
        self.mlp_convs = None
        self.candidate_selector = None
        # The target's, set after construction (logits processor) and at load (embedding and LM head).
        self.logits_processor = None
        self.lm_head = None
        # A disaggregated context server's drafter (dflash2_context_kv_only) only stores the prompt's
        # context latents.
        self.context_kv_only = context_kv_only
        self.model = DraftBackbone(model_config, context_kv_only)

        self._window = window
        # Every read is inside the window, so the worker may keep a window-bounded context ring.
        self._ctx_ring_window = window
        # Window-attention workspace and multi-CTA counters, allocated on the first (eager) call
        # for _max_block_rows query rows (set by the worker: max requests x block width).
        self._window_attn_buffers = None
        self._window_attn_rows = 0
        self._max_block_rows = 0
        # The window attention runs TRTLLM-Gen's sparse MLA decode, certified on SM 10.3 (this target's GPU).
        logger.info_once(
            "DFlash2 drafter window attention: TRTLLM-Gen sparse MLA decode (SM 10.3).",
            key="k31_dflash2_window_attention",
        )

        attn0 = self.model.layers[0].self_attn
        self.qk_nope_head_dim = attn0.qk_nope_head_dim
        self.qk_rope_head_dim = attn0.qk_rope_head_dim
        self.kv_lora_rank = attn0.kv_lora_rank
        self.v_head_dim = attn0.v_head_dim
        # The worker's cache contract: one latent per token and layer, no V.
        self._num_attn_layers = len(self.model.layers)
        self._num_heads = attn0.num_heads_tp
        self._head_dim = attn0.kv_cache_head_dim
        # Every layer's latent projection stacked, its norms and the absorbed kv_b halves: built on first use.
        self._fused_kv_weight = None

        rope = resolve_mla_rope_params(cfg, draft_config)
        self._mla_rope_params = rope
        self._mla_freqs = None
        self._mla_freqs_cap = None
        mscale = yarn_get_mscale(rope["scaling_factor"], rope["mscale_all_dim"])
        self.softmax_scale = (mscale * mscale) / math.sqrt(
            self.qk_nope_head_dim + self.qk_rope_head_dim
        )
        logger.info(
            f"DFlash2 MLA drafter: {self._num_attn_layers} layers, {self._num_heads} heads per "
            f"rank, block {self.block_size}, window {self._window}, softmax scale "
            f"{self.softmax_scale:.8f}, rope theta {rope['theta']}"
        )

    @property
    def is_dflash2(self) -> bool:
        return True

    @property
    def has_candidate_selector(self) -> bool:
        return self.candidate_selector is not None

    # -- weights -----------------------------------------------------------

    def load_weights(self, weights: Dict, weight_mapper=None, **kwargs) -> None:
        """The drafter checkpoint: the DFlash2 convolutions and selector, ``fc`` / ``hidden_norm``, and this rank's
        shards of the backbone (:meth:`_load_backbone_weights`). A context-only drafter takes only ``fc`` /
        ``hidden_norm`` and each layer's latent projection and norm. The checkpoint's names are the modules' own, so
        ``weight_mapper`` is not used."""
        del weight_mapper, kwargs
        missing = [k for k in self.WRAPPER_OWNED_WEIGHTS if k not in weights]
        if missing:
            raise ValueError(f"The DFlash2 drafter checkpoint is missing {missing}.")
        if not self.context_kv_only:
            self._load_dflash2_weights(weights)
        fc = weights["fc.weight"]
        self.fc = nn.Linear(fc.shape[1], fc.shape[0], bias=False, device="cuda", dtype=fc.dtype)
        self.fc.weight.data.copy_(fc)
        norm = weights["hidden_norm.weight"]
        self.hidden_norm = nn.RMSNorm(
            norm.shape[0],
            eps=getattr(self.config, "rms_norm_eps", 1e-6),
            device="cuda",
            elementwise_affine=True,
            dtype=norm.dtype,
        )
        self.hidden_norm.weight.data.copy_(norm)
        self._load_backbone_weights(weights, torch.device("cuda"))

    def _load_backbone_weights(self, weights: Dict, device: torch.device) -> None:
        """This rank's shards of the backbone tensors (``K31Linear.load_shard``, the engine ``Linear``'s even split;
        ``gate_up_proj`` is this rank's ``[gate | up]`` rows), copied into the holders through ``device``."""
        sources = {}
        for name, module in self.model.named_modules():
            if isinstance(module, K31RMSNorm):
                sources[name] = (module, (f"{name}.weight",))
            elif isinstance(module, K31Linear):
                if name.endswith(".mlp.gate_up_proj"):
                    mlp = name[: -len("gate_up_proj")]
                    sources[name] = (module, (f"{mlp}gate_proj.weight", f"{mlp}up_proj.weight"))
                else:
                    sources[name] = (module, (f"{name}.weight",))
        missing = sorted(k for _, keys in sources.values() for k in keys if k not in weights)
        if missing:
            raise ValueError(
                f"The DFlash2 drafter checkpoint provides no {missing[:8]}{' ...' if len(missing) > 8 else ''}."
            )
        for module, keys in sources.values():
            if isinstance(module, K31RMSNorm):
                module.weight.data.copy_(weights[keys[0]])
            else:
                module.weight.data.copy_(
                    torch.cat([module.load_shard(weights[k], device) for k in keys])
                )

    def _load_dflash2_weights(self, weights: Dict) -> None:
        """Build the block convolutions and the candidate selector from the checkpoint. The selector runs on the CuTe
        DSL selector ops, which take a fixed candidate count and a rank of whole blocks."""
        if (
            self._dflash2_selector_top_k != SELECTOR_TOP_K
            or self._dflash2_selector_rank % _SELECTOR_RANK_BLOCK
        ):
            raise ValueError(
                f"The DFlash2 selector takes selector_top_k {SELECTOR_TOP_K} and a selector_rank that is a multiple "
                f"of {_SELECTOR_RANK_BLOCK}; the checkpoint declares selector_top_k "
                f"{self._dflash2_selector_top_k} and selector_rank {self._dflash2_selector_rank}."
            )

        def take(key: str) -> torch.Tensor:
            if key not in weights:
                raise ValueError(f"The DFlash2 drafter checkpoint is missing '{key}'.")
            return weights[key].to("cuda")

        hidden_size = self.config.hidden_size
        attention_convs = nn.ModuleList()
        mlp_convs = nn.ModuleList()
        for layer_idx in range(len(self.model.layers)):
            for name, convs in (("attention_conv", attention_convs), ("mlp_conv", mlp_convs)):
                prefix = f"layers.{layer_idx}.{name}."
                convs.append(
                    K31DFlash2BlockConv(
                        base_kernel=take(prefix + "base_kernel"),
                        kernel_projection_weight=take(prefix + "kernel_projection.weight"),
                        taps=self._dflash2_conv_taps,
                        group_size=self._dflash2_conv_group_size,
                        hidden_size=hidden_size,
                    )
                )
        self.attention_convs = attention_convs
        self.mlp_convs = mlp_convs
        self.candidate_selector = K31DFlash2CandidateSelector(
            predecessor_codebook=take("candidate_selector.predecessor_codebook"),
            successor_codebook=take("candidate_selector.successor_codebook"),
            hidden_projection_weight=take("candidate_selector.hidden_projection.weight"),
            top_k=self._dflash2_selector_top_k,
            vocab_size=self.config.vocab_size,
            rank=self._dflash2_selector_rank,
        )

    def load_weights_from_target_model(self, target_model: torch.nn.Module) -> None:
        """Share the target's embedding and LM head; a context-only drafter reads neither."""
        if self.context_kv_only:
            return
        self.model.embed_tokens = target_model.model.embed_tokens
        self.lm_head = target_model.lm_head

    # -- target features ---------------------------------------------------

    def project_target_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``hidden_norm(fc(hidden_states))``: the captured target features in the drafter's hidden space."""
        return self.hidden_norm(self.fc(hidden_states.to(self.fc.weight.dtype)))

    def select_candidate_path(
        self, block_logits, candidate_ids, unary_logits, block_hidden_states, anchor_token_ids
    ):
        """Restrict each block slot's logits to the selector's greedy path: the edge scores, the realized walk and
        ``block_logits`` rewritten in place (``-inf`` except at each slot's candidates, which carry the score row
        the walk realized), as the CuTe DSL op ``ops/spec/dflash2_selector_greedy_path``."""
        if block_logits.dtype not in (torch.bfloat16, torch.float32):
            raise NotImplementedError(
                f"The DFlash2 selector path takes BF16 / FP32 block logits; got {block_logits.dtype}."
            )
        selector = self.candidate_selector
        return k31_dflash2_selector_greedy_path(
            selector.predecessor_codebook,
            selector.successor_codebook,
            selector.hidden_projection.weight,
            candidate_ids,
            unary_logits,
            block_hidden_states,
            anchor_token_ids,
            block_logits,
        )

    # -- shape / buffers ---------------------------------------------------

    def _mla_rope_kwargs(self) -> dict:
        """``build_mla_yarn_freqs_cis`` arguments of the YaRN table, sized from the worker's ceiling."""
        rope = dict(self._mla_rope_params)
        if self._runtime_position_ceiling is not None:
            rope["max_positions"] = int(self._runtime_position_ceiling)
        return dict(
            dim=self.qk_rope_head_dim,
            base=rope["theta"],
            scaling_factor=rope["scaling_factor"],
            original_max_position_embeddings=rope["original_max_positions"],
            max_position_embeddings=rope["max_positions"],
            beta_fast=rope["beta_fast"],
            beta_slow=rope["beta_slow"],
            mscale=rope["mscale"],
            mscale_all_dim=rope["mscale_all_dim"],
        )

    def _mla_freqs_cis(self, device) -> torch.Tensor:
        """``[positions, rope_dim // 2]`` complex64 YaRN table on ``device``. A context-only drafter
        never builds it: its stores rotate only their positions (``dflash2_rope_rows``)."""
        kwargs = self._mla_rope_kwargs()
        if self._mla_freqs is None or self._mla_freqs_cap != kwargs["max_position_embeddings"]:
            self._mla_freqs = build_mla_yarn_freqs_cis(device=device, **kwargs)
            self._mla_freqs_cap = kwargs["max_position_embeddings"]
        return self._mla_freqs

    def _build_fused_kv_buffers(self) -> None:
        """Stack every layer's latent projection and norm; split ``kv_b`` into W_UK / W_UV."""
        if self._fused_kv_weight is not None:
            return
        attns = [layer.self_attn for layer in self.model.layers]
        self._fused_kv_weight = torch.cat([a.kv_a_proj_with_mqa.weight for a in attns], 0)
        self._fused_kv_weight = self._fused_kv_weight.contiguous()
        self._kv_a_norm_stacked = torch.stack([a.kv_a_layernorm.weight.data for a in attns])
        self._kv_a_norm_eps = attns[0].kv_a_layernorm.variance_epsilon
        if self.context_kv_only:
            # No block decode: neither the absorbed kv_b nor the packed block projections.
            return
        for a in attns:
            w = a.kv_b_proj.weight.view(a.num_heads_tp, self.qk_nope_head_dim + self.v_head_dim, -1)
            a._k_b_proj = w[:, : self.qk_nope_head_dim].contiguous()  # [H, nope, latent]
            a._v_b_proj = w[:, self.qk_nope_head_dim :].contiguous()  # [H, v, latent]
        self._k31_pack_block_projections()

    def _k31_pack_block_projections(self) -> None:
        """A target's packing of its block projections for its low-M GEMM plan, after the latent buffers are built;
        none by default."""

    def _block_attention(self, query, pool, rows, lengths):
        """``[N, H, kv_lora_rank]``: the window decode of ``query`` over ``pool``'s ``rows``
        (the TRTLLM-Gen sparse MLA kernel, ``ops/spec/dflash2_window_attention``)."""
        if self._window_attn_buffers is None or self._window_attn_rows < query.shape[0]:
            # Sized for the worker's largest step (max requests x block width) on the first,
            # eager, call: never allocate during CUDA-graph capture.
            rows_cap = max(query.shape[0], self._max_block_rows)
            self._window_attn_buffers = dflash2_window_attention_buffers(
                rows_cap, query.shape[1], query.dtype, query.device
            )
            self._window_attn_rows = rows_cap
        workspace, counter = self._window_attn_buffers
        return dflash2_window_attention(
            query,
            pool,
            rows,
            lengths,
            self.softmax_scale,
            self.kv_lora_rank,
            workspace,
            counter,
        )


# ---------------------------------------------------------------------------
# DraftModel: the drafter as this target runs it.
# ---------------------------------------------------------------------------


class DraftModel(DraftModelBase):
    """The Kimi K3.1 DFlash2 drafter (over ``DraftModelBase``) as this target runs it: the target-feature
    projection column-parallel under plain TP, the low-M GEMM plan of TEP8 (packed ``[q_a | kv_a]``), the
    context-latent projection and paged store, and the block decode with the pushed all-reduce epilogues."""

    def __init__(self, draft_config: ModelConfig, **kwargs):
        super().__init__(draft_config, **kwargs)
        # The column-parallel fc (plain TP): the all-reduce module whose MNNVL Lamport workspace gathers
        # the ranks' output columns (lamport_allgather), and this rank's rows of fc.weight (a view, set on
        # the first projection). None keeps the replicated fc. The handle never reduces, so it is built with AUTO
        # (the MNNVL workspace wherever MNNVL is available), not the configured all-reduce strategy.
        mapping = self.model_config.mapping
        self._k31_fc_allreduce = (
            tp_allreduce(mapping)
            if mapping.tp_size > 1 and not mapping.enable_attention_dp
            else None
        )
        self._k31_fc_cols: Optional[torch.Tensor] = None
        # The drafter's low-M GEMM plan for the block decode and context projections (its own at TEP8, elsewhere the
        # target's).
        self._k31_low_m_plan = _k31_draft_plan(getattr(draft_config, "mapping", None))

    def _k31_fc_columns(self) -> Optional[torch.Tensor]:
        """This rank's ``fc.weight`` rows ``[out_features / tp, in_features]`` for the column-parallel fc, or
        None where it does not apply (attention DP, one rank, an output width that does not split into whole
        16-byte vectors per rank)."""
        if (
            self._k31_fc_cols is not None
            or self._k31_fc_allreduce is None
            or getattr(self, "fc", None) is None
        ):
            return self._k31_fc_cols
        mapping = self.model_config.mapping
        width, rem = divmod(self.fc.weight.shape[0], mapping.tp_size)
        if rem or width % 8:
            return None
        self._k31_fc_cols = self.fc.weight[mapping.tp_rank * width : (mapping.tp_rank + 1) * width]
        return self._k31_fc_cols

    def project_target_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``hidden_norm(fc(hidden_states))``, column-parallel under plain TP: this rank's output columns
        (``hidden_states @ cols.T``, BF16), all-gathered as 32-bit words in one multicast kernel up to the raised
        cap (``lamport_allgather`` at ``RAISED_MAX_BYTES``: 585 rows of 7168 at any TP width) and through the
        engine all-gather above it (still cheaper than the replicated fc), then ``hidden_norm``. Each output is
        the full-K dot product of the replicated fc, rounded once to BF16; only the GEMM's accumulation order
        differs (not bitwise)."""
        cols = self._k31_fc_columns()
        if cols is None or hidden_states.dim() != 2:
            return self.hidden_norm(self.fc(hidden_states.to(self.fc.weight.dtype)))
        x = hidden_states.to(cols.dtype)
        local = _k31_planned_linear(x, cols, self._k31_low_m_plan)
        if local is None:
            local = F.linear(x, cols)
        full = lamport_allgather(
            local.contiguous().view(torch.float32),
            self.model_config.mapping,
            self._k31_fc_allreduce,
            LAMPORT_ALLGATHER_RAISED_MAX_BYTES,
        )
        return self.hidden_norm(full.view(torch.bfloat16))

    def _k31_pack_block_projections(self) -> None:
        """With a low-M plan that names the packed ``[q_a | kv_a]`` shape: one ``[q_a | kv_a]`` weight per layer (the
        modules keep row views of it), and the plan handed to the block convolutions and the MLPs. Without it the
        drafter keeps its separate projections on cuBLAS."""
        plan = self._k31_low_m_plan
        if plan is None or (2112, 7168) not in plan:
            return
        for conv in list(self.attention_convs) + list(self.mlp_convs):
            conv._k31_low_m_plan = plan
        for layer in self.model.layers:
            layer.mlp._k31_low_m_plan = plan
            a = layer.self_attn
            q_a, kv_a = a.q_a_proj, a.kv_a_proj_with_mqa
            packed = torch.cat([q_a.weight.data, kv_a.weight.data], 0).contiguous()
            q_a.weight.data = packed[: q_a.weight.shape[0]]
            kv_a.weight.data = packed[q_a.weight.shape[0] :]
            a._k31_qa_kva_weight = packed

    def precompute_context_kv(self, projected_hidden, positions, freqs=None):
        """Context latents of every layer: ``([N, L, 1, kv_lora_rank + rope], None)``.

        ``projected_hidden`` is ``hidden_norm(fc(captures))``; no input norm, no convolution.
        ``freqs`` replaces the YaRN table that ``positions`` index (``dflash2_rope_rows``).
        """
        if self._fused_kv_weight is None:
            self._build_fused_kv_buffers()
        n = projected_hidden.shape[0]
        weight = self._fused_kv_weight
        x = projected_hidden.to(weight.dtype)
        latent = _k31_planned_linear(x, weight, self._k31_low_m_plan)
        if latent is None:
            latent = F.linear(x, weight)
        latent = latent.view(n, self._num_attn_layers, self._head_dim)
        # [RMSNorm | RoPE] of every layer's latent in one launch (bitwise the op's torch reference).
        out = k31_dflash2_mla_ctx_norm_rope(
            latent,
            self._kv_a_norm_stacked,
            self._kv_a_norm_eps,
            freqs if freqs is not None else self._mla_freqs_cis(latent.device),
            positions.view(-1),
        )
        return out.unsqueeze(2), None

    def k31_context_latents(self, projected_hidden: torch.Tensor) -> torch.Tensor:
        """``[N, L, kv_lora_rank + rope]``: every layer's context latent before its norm / RoPE, the projection
        GEMM of :meth:`k31_store_context_kv`."""
        if self._fused_kv_weight is None:
            self._build_fused_kv_buffers()
        n = projected_hidden.shape[0]
        weight = self._fused_kv_weight
        x = projected_hidden.to(weight.dtype)
        latent = _k31_planned_linear(x, weight, self._k31_low_m_plan)
        if latent is None:
            latent = F.linear(x, weight)
        return latent.view(n, self._num_attn_layers, self._head_dim)

    def k31_store_context_kv(
        self,
        projected_hidden: torch.Tensor,
        positions: torch.Tensor,
        pool: torch.Tensor,
        rows: torch.Tensor,
        keep: torch.Tensor,
    ) -> None:
        """:meth:`precompute_context_kv`'s latents of every layer times ``keep``, stored at ``rows`` of the
        worker's paged arena ``pool`` (``[L, pages, 1, 1, page_size, D]``, BF16): the projection GEMM, then the
        norm / RoPE writing straight into the pages (``ops/spec/mla_norm_rope`` store entry, bitwise the
        latents, the masked multiply and the indexed store)."""
        self.k31_store_context_latents(
            self.k31_context_latents(projected_hidden), positions, pool, rows, keep
        )

    def k31_store_context_latents(
        self,
        latents: torch.Tensor,
        positions: torch.Tensor,
        pool: torch.Tensor,
        rows: torch.Tensor,
        keep: torch.Tensor,
    ) -> None:
        """The norm / RoPE store of :meth:`k31_store_context_kv` from :meth:`k31_context_latents`' output."""
        k31_dflash2_mla_ctx_norm_rope_store(
            latents,
            self._kv_a_norm_stacked,
            self._kv_a_norm_eps,
            self._mla_freqs_cis(latents.device),
            positions.view(-1),
            pool.view(pool.shape[0], -1, pool.shape[-1]),
            rows,
            keep,
        )

    def dflash_forward(
        self,
        noise_embedding,
        query_positions,
        num_ctx_per_req,
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
        page_size = ctx_kv_cache[0].shape[-2]
        if isinstance(block_placement, K31BlockStore):
            # Built with the drafter inputs (the same rows and addresses as below).
            page_tables, store_pages, store_offsets = block_placement
        else:
            page_tables = ctx_page_table.index_select(0, ctx_cache_batch_idx.long()).long()
            block_pos = ctx_len.view(-1, 1) + torch.arange(block_size, device=device)
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
        freqs_table = self._mla_freqs_cis(device)  # [P, rope/2] complex64
        positions_flat = query_positions.reshape(-1)

        hidden_states = noise_embedding.reshape(num_tokens, hidden)
        residual = None
        # The previous layer's MLP (its all-reduce pending where pushed) and its convolution, finished with the
        # next residual add + norm.
        pending_mlp = None
        for layer_idx, layer in enumerate(self.model.layers):
            attn = layer.self_attn
            if residual is None:
                # The fused add + norm below updates the residual in place: a copy, except for the fused
                # prepare's noise block (block_placement a K31BlockStore), a fresh buffer read only here.
                residual = (
                    hidden_states
                    if isinstance(block_placement, K31BlockStore)
                    else hidden_states.clone()
                )
                norm = layer.input_layernorm
                h = flashinfer_rmsnorm(hidden_states, norm.weight, norm.variance_epsilon)
            else:
                norm = layer.input_layernorm
                h, residual = _sublayer_epilogue(pending_mlp, block_size, residual, norm)
            h, attn_coefficients = self.attention_convs[layer_idx].prepare(h, block_size)

            q_a_norm = attn.q_a_layernorm
            qa_kva = getattr(attn, "_k31_qa_kva_weight", None)
            if qa_kva is not None:
                # One planned GEMM for [q_a | kv_a]; the latent block is read in place by the rope op.
                q_a_out, kv_a_out = k31_low_m_linear(h, qa_kva, self._k31_low_m_plan, True).split(
                    [attn.q_a_proj.weight.shape[0], attn.kv_a_proj_with_mqa.weight.shape[0]], -1
                )
                q_n = flashinfer_rmsnorm(q_a_out, q_a_norm.weight, q_a_norm.variance_epsilon)
                q = _k31_planned_linear(q_n, attn.q_b_proj.weight, self._k31_low_m_plan)
                if q is None:
                    q = attn.q_b_proj(q_n)
            else:
                kv_a_out = None
                q = attn.q_b_proj(
                    flashinfer_rmsnorm(attn.q_a_proj(h), q_a_norm.weight, q_a_norm.variance_epsilon)
                )
            q = q.view(num_tokens, num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim)
            query = torch.empty(
                num_tokens,
                num_heads,
                latent_dim + self.qk_rope_head_dim,
                dtype=q.dtype,
                device=device,
            )
            # Absorb W_UK into the query: [H, N, nope] @ [H, nope, latent], written straight into the
            # query's latent columns (their transposed view; the same product as a bmm and a copy).
            bmm_out(
                q[..., : self.qk_nope_head_dim].transpose(0, 1),
                attn._k_b_proj.to(q.dtype),
                query[..., :latent_dim].transpose(0, 1),
            )
            # One launch: query[..., latent_dim:] = RoPE(q's rope half), and the block latents
            # [RMSNorm | RoPE] stored into the pool.
            pool = ctx_kv_cache[layer_idx][:, 0, 0]  # [pages, page_size, 576]
            k31_dflash2_mla_block_rope(
                q,
                query,
                kv_a_out if kv_a_out is not None else attn.kv_a_proj_with_mqa(h),
                attn.kv_a_layernorm.weight,
                attn.kv_a_layernorm.variance_epsilon,
                freqs_table,
                positions_flat,
                pool,
                store_pages,
                store_offsets,
            )

            ctx_out = self._block_attention(query, pool, rows, lengths)  # [N, H, latent]
            # einsum("nhc,hvc->nhv", ctx_out, W_UV) as one bmm into the [N, H, v] rows' transposed view
            # (the einsum's product without its permute copy).
            attn_out = torch.empty(
                num_tokens, num_heads, self.v_head_dim, dtype=ctx_out.dtype, device=device
            )
            bmm_out(
                ctx_out.transpose(0, 1),
                attn._v_b_proj.to(ctx_out.dtype).transpose(1, 2),
                attn_out.transpose(0, 1),
            )
            o_in = attn_out.view(num_tokens, num_heads * self.v_head_dim)
            attn_sub = _drafter_sublayer(
                attn.o_proj,
                o_in,
                self.attention_convs[layer_idx],
                attn_coefficients,
                lambda: attn.o_proj(o_in),
            )
            norm = layer.post_attention_layernorm
            h2, residual = _sublayer_epilogue(attn_sub, block_size, residual, norm)
            h2, mlp_coefficients = self.mlp_convs[layer_idx].prepare(h2, block_size)
            pending_mlp = _drafter_mlp_sublayer(
                layer.mlp, h2, self.mlp_convs[layer_idx], mlp_coefficients
            )

        hidden_states, _ = _sublayer_epilogue(pending_mlp, block_size, residual, self.model.norm)
        return hidden_states


# ====================================================================================================================
# DFlash2 speculative worker
# ====================================================================================================================


def dflash_context_dtype(draft_model: nn.Module) -> torch.dtype:
    """Use projection activations for the context cache, not packed weight storage."""
    projection = getattr(draft_model, "fc", None)
    if projection is None:
        return torch.bfloat16
    return getattr(projection, "dtype", projection.weight.dtype)


class K31DFlashWorker(
    DFlash2DraftPagesWorkerMixin, UnifiedDraftCacheWorkerMixin, _EngineDFlashWorker
):
    """The engine's DFlash worker with the Kimi K3.1 DFlash2 differences.

    Every method not defined here is the draft-pages carrier's (``DFlash2DraftPagesWorkerMixin``), the unified
    draft cache's (``UnifiedDraftCacheWorkerMixin``) or the engine ``DFlashWorker``'s, unchanged. The overrides
    carry what the K3.1 target runs differently: the DFlash2 sampled selector walk and block verification
    (``sample_and_accept_draft_tokens``), the draft context in the KV cache manager's draft layers
    (managed history, unified draft tables, device-decoded block tables), the input staging of the engine scope,
    and the first drafter inputs built from them.
    """

    def __init__(
        self,
        spec_config: "DFlashDecodingConfig",
        mapping: Mapping,
        use_separate_draft_kv_cache: bool = False,
    ):
        super().__init__(spec_config, mapping, use_separate_draft_kv_cache)
        # The DFlash2 candidate-pair exchange multicasts on the TP group's MNNVL Lamport workspace
        # (lamport_allgather), reached through this all-reduce module of the group; None under attention DP.
        # The handle never reduces, so it is built with AUTO (the MNNVL workspace wherever MNNVL is available), not
        # the configured all-reduce strategy.
        self._pair_allreduce = (
            tp_allreduce(mapping)
            if mapping is not None and mapping.tp_size > 1 and not mapping.enable_attention_dp
            else None
        )
        # Rows per slot when the private paged arena is a context ring bounded by the drafter's
        # window (ops/spec/dflash2_ring): position p lives at slot row p mod _ctx_ring_rows. None when
        # the arena is dense in position or the pages are the manager's.
        self._ctx_ring_rows = None
        # Decode the manager draft pool's device block offsets instead of
        # building the table on the host (see _refresh_unified_block_tables).
        self._device_draft_tables = False
        self._draft_table_checks_left = 0
        # _block_gather_ids per (requests, block, draft tokens, rows).
        self._k31_gather_ids: dict[tuple[int, int, int, int], torch.Tensor] = {}

    def set_draft_model(self, draft_model) -> None:
        super().set_draft_model(draft_model)
        # The MLA drafter decodes from its own paged context pool with its own kernels: its forward reads none
        # of the target's attention metadata.
        self._draft_reads_no_attn_metadata = isinstance(draft_model, DraftModel)

    def _local_draft_tokens(self, draft_model) -> Optional[VocabParallelDraftTokens]:
        """How greedy draft tokens come from the drafter's own full-vocab LM head
        output on this rank when there is no vocab-parallel selection (e.g. under
        attention DP), or None for the logits path. Plain DFlash has none."""
        del draft_model
        return None

    def _has_unified_draft_cache(self) -> bool:
        return bool(getattr(self._ctx_kv_manager, "draft_layer_ids", ()))

    def _managed_ctx_pool(
        self,
        draft_kv_cache_manager: BaseResourceManager | None,
        L: int,
        nkv: int,
        hd: int,
        dtype: torch.dtype,
        kv_factor: int = 2,
    ) -> list[torch.Tensor] | None:
        """Per-layer views of the draft KV cache manager's pool, or None.

        The manager's buffer accessor supplies the same per-layer HND shape
        as the private arena. Unified storage uses ``get_draft_buffers`` and
        requires the pool to match the drafter and the manager's page size;
        legacy storage uses ``get_buffers`` and may fall back to the arena.

        A list, not one stacked tensor: only the two consumers that index it by
        layer need it, and the two managers disagree on what a stack would mean.
        V1 slices one page-major allocation, so the layers are strided views of
        a shared storage; V2 wraps each layer's pool base address in its own
        tensor. Everything downstream indexes ``[layer_idx]`` first anyway.

        Note the page count is not the drafter's to interpret: a V2 view spans
        the whole interleaved pool, not this layer's slice, so only shape[1:] is
        checked here. Legacy offsets are settled in _init_ctx_block_offsets;
        unified storage supplies draft block indices directly.
        """
        if draft_kv_cache_manager is None:
            # No separate draft KV cache. Attention DP alone does NOT disable
            # it: _util.py skips that bail for an external drafter, which is
            # what the DFlash2 drafter is, so it does get a manager under DP.
            return None
        unified = bool(getattr(draft_kv_cache_manager, "draft_layer_ids", ()))
        get_buffers = (
            draft_kv_cache_manager.get_draft_buffers
            if unified
            else draft_kv_cache_manager.get_buffers
        )
        layers = [get_buffers(i, kv_layout="HND") for i in range(L)]
        base = layers[0]
        page_size = draft_kv_cache_manager.tokens_per_block if unified else base.size(-2)
        # kv_factor 1 is the MLA drafter's SELFKONLY pool: one latent, no V.
        expected = (kv_factor, nkv, page_size, hd)
        for i, layer in enumerate(layers):
            if tuple(layer.shape[1:]) != expected or layer.dtype != dtype:
                if unified:
                    raise ValueError(
                        "Unified DFlash2 draft pool does not match the drafter KV layout"
                    )
                # Fall back rather than fail: the private arena is what every
                # DFlash model shipped with, so an unrecognized pool costs the
                # old memory footprint, not the run.
                logger.warning(
                    f"DFlash: draft KV pool layer {i} is {tuple(layer.shape)} / {layer.dtype}, "
                    f"not {('pages',) + expected} / {dtype}; using the private ctx arena."
                )
                return None
        logger.info(
            f"DFlash: binding the drafter ctx cache to the draft KV cache manager pool: "
            f"{L} x {tuple(base.shape)} {base.dtype}, page_size={base.size(-2)}"
        )
        return layers

    def _init_ctx_block_offsets(self, draft_kv_cache_manager, pool, L, nkv, hd, page_size) -> bool:
        """Validate the legacy pool layout and resolve its offset decode constant.

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

    def _refresh_ctx_block_tables(
        self, attn_metadata, num_seqs: int, request_ids: list[int] | None = None
    ) -> bool:
        """Decode this iteration's draft block table into the persistent buffer.

        In place and fixed-shape so a captured graph keeps reading the same
        tensor. Returns False when there is no table to fill, i.e. every path
        that stayed on the private arena.
        """
        self._ctx_block_counts_host = None
        if self._ctx_block_tables is None or num_seqs <= 0:
            return False
        if self._has_unified_draft_cache():
            if self._device_draft_tables and self._refresh_unified_block_tables(
                attn_metadata, num_seqs, request_ids
            ):
                return True
            table = self._ctx_kv_manager.get_draft_block_table(request_ids[:num_seqs])
            # Pinned: a pageable upload would wait for the step in flight.
            table = maybe_pin_memory(table)
            self._ctx_block_tables[:num_seqs].copy_(table, non_blocking=True)
            counts = [
                self._ctx_kv_manager.get_draft_num_blocks(rid) for rid in request_ids[:num_seqs]
            ]
            self._ctx_block_counts_host = counts
            self._ctx_block_counts[:num_seqs].copy_(
                torch.tensor(
                    counts,
                    dtype=torch.long,
                    device="cpu",
                    pin_memory=prefer_pinned(),
                ),
                non_blocking=True,
            )
            return True
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

    def _init_unified_draft_tables(self, manager) -> None:
        """Choose how the unified pool's block table is built each iteration."""
        self._device_draft_tables = False
        layer = manager.draft_layer_ids[0]
        pool = manager.layer_to_pool_mapping_dict[manager.layer_offsets[layer]]
        if getattr(manager, "enable_swa_scratch_reuse", False) or (
            manager._index_scale_ints[pool] != manager.get_layer_page_index_scale(layer)
        ):
            logger.warning(
                "DFlash: the draft pool's block offsets cannot be decoded on the device; "
                "building draft block tables on the host."
            )
            return
        self._ctx_pool_idx = pool
        self._ctx_block_divisor = manager.get_layer_kv_factor(layer)
        self._device_draft_tables = True
        self._draft_table_checks_left = 8
        logger.info(
            f"DFlash: draft block tables decoded on the device from pool {pool} "
            f"(divisor {self._ctx_block_divisor})"
        )

    def _refresh_unified_block_tables(
        self, attn_metadata, num_seqs: int, request_ids: list[int]
    ) -> bool:
        """Decode the draft pool's rows of the target's device block offsets.

        The target metadata already uploads every pool's offsets for this
        batch, so the draft rows need one device divide plus the per-request
        block counts, rather than a host table rebuilt from page lists.
        Returns False when the host path must build this iteration's table.
        """
        src = getattr(attn_metadata, "kv_cache_block_offsets", None)
        if src is None or src.size(0) <= self._ctx_pool_idx:
            return False
        width = min(src.size(-1), self._ctx_block_tables.size(1))
        encoded = src[self._ctx_pool_idx, :num_seqs, 0, :width]
        decoded = torch.div(encoded.clamp(min=0), self._ctx_block_divisor, rounding_mode="floor")
        self._ctx_block_tables[:num_seqs, :width].copy_(decoded)
        cache_map = self._ctx_kv_manager.kv_cache_map
        counts = []
        for request_id in request_ids[:num_seqs]:
            cache = cache_map.get(request_id)
            counts.append(cache.num_blocks if cache is not None else 0)
        self._ctx_block_counts[:num_seqs].copy_(
            torch.tensor(counts, dtype=torch.long, pin_memory=prefer_pinned()), non_blocking=True
        )
        if self._draft_table_checks_left > 0 and not torch.cuda.is_current_stream_capturing():
            self._draft_table_checks_left -= 1
            if not self._check_device_block_tables(num_seqs, request_ids, counts):
                return False
        self._ctx_block_counts_host = counts
        return True

    def _check_device_block_tables(
        self, num_seqs: int, request_ids: list[int], counts: list[int]
    ) -> bool:
        """Compare device-decoded rows with the manager's page lists (startup only)."""
        manager = self._ctx_kv_manager
        rows = []
        for row, request_id in enumerate(request_ids[:num_seqs]):
            cache = manager.kv_cache_map.get(request_id)
            if counts[row] > 0 and cache is not None and cache.is_active:
                rows.append(row)
        if not rows:
            return True
        expected = manager.get_batch_cache_indices(
            [request_ids[row] for row in rows], manager.draft_layer_ids[0]
        )
        device_rows = self._ctx_block_tables[rows].cpu()
        for k, row in enumerate(rows):
            got = device_rows[k, : counts[row]].tolist()
            want = expected[k][: counts[row]]
            if any(w >= 0 and g != w for g, w in zip(got, want)):
                logger.error(
                    "DFlash: device-decoded draft block table disagrees with the manager "
                    f"for request {request_ids[row]}; building draft tables on the host."
                )
                self._device_draft_tables = False
                return False
        return True

    def _lazy_init_ctx_buffers(
        self, draft_model, spec_metadata, attn_metadata, draft_kv_cache_manager=None
    ):
        self._validate_draft_attention_backend(draft_model)
        unified = bool(getattr(draft_kv_cache_manager, "draft_layer_ids", ()))
        # Only TrtllmAttentionMetadata builds draft_kv_cache_block_offsets.
        # Without it the table stays all-zero and every request would read and
        # write pool block 0 -- silently, at acceptance 1.0. Unified storage
        # builds its table from the manager's draft block indices instead.
        if (
            not unified
            and draft_kv_cache_manager is not None
            and not hasattr(attn_metadata, "draft_kv_cache_block_offsets")
        ):
            logger.warning(
                f"DFlash: {type(attn_metadata).__name__} carries no draft KV block offsets; "
                "using the private ctx arena."
            )
            draft_kv_cache_manager = None
        # The arena is sized below from the served length of the KV cache manager. KV cache
        # estimation runs a probe manager, whose length its small quota can clamp, before the real
        # one: rebuild for the real one.
        cache_manager = getattr(attn_metadata, "kv_cache_manager", None)
        if self._ctx_buf_inited:
            if (
                self._ctx_kv_manager is draft_kv_cache_manager
                and getattr(self, "_ctx_sized_by", None) is cache_manager
            ):
                return
            # KV cache estimation swaps a probe manager for the real one.
            # Holding the old pool's tensors across that writes freed memory.
            logger.info("DFlash: KV cache manager replaced, rebinding the ctx cache.")
            self._ctx_buf_inited = False
        self._ctx_sized_by = cache_manager

        # Recorded even when the buffers stay private, so that the estimation
        # pass falling back to the arena still rebinds once the real pool lands.
        self._ctx_kv_manager = draft_kv_cache_manager
        self._ctx_ring_rows = None
        if hasattr(draft_model, "_ctx_ring_rows"):
            draft_model._ctx_ring_rows = None
        self._ctx_block_tables = None
        self._ctx_block_counts = None
        self._ctx_block_tables_read = None
        self._ctx_block_counts_host = None
        self._paged_latent_prepare = False
        self._ctx_block_counts_host = None
        # Slots span the executor's whole seq-slot pool (requests paused in place by GEN slot
        # preemption, more than one micro-batch); max_num_requests is the captured graph bucket.
        max_batch = int(spec_metadata.num_seq_slots or spec_metadata.max_num_requests)

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
        if (
            getattr(cache_manager, "is_disagg", False)
            and not getattr(cache_manager, "kda_context_only", False)
            and dflash2_draft_pages_manager(cache_manager) is None
        ):
            # A disaggregated generation server without draft pages: no drafter context arrives
            # with the prompt, so every request's drafter context starts empty.
            logger.warning_once(
                "DFlash2 drafter has no prompt context on this disaggregated generation server "
                "(the KV cache manager holds no draft pages): acceptance is reduced.",
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
        self._ctx_position_offset = torch.zeros_like(self._ctx_len)
        self._ctx_len_host = [0] * num_slots
        self._batch_to_slot_buffer = DeviceCopyBuffer(
            torch.zeros(max_batch, dtype=torch.long, device="cuda"),
            pin_memory=prefer_pinned(),
        )
        self._managed_snapshot_slots = (
            torch.empty(max_batch, dtype=torch.long, device="cuda") if unified else None
        )

        self._free_slots = deque(range(max_batch))
        self._req_to_slot = {}
        self._req_ctx_pos = {}
        self._managed_cache_bindings = {}
        self._managed_request_ids = ()

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
        self._ctx_paged = True
        pool = self._managed_ctx_pool(draft_kv_cache_manager, L, nkv, hd, dtype, kv_factor)
        # The manager's page size wins when bound to it: its pool is already
        # carved, so the drafter adopts the geometry rather than imposing one.
        page_size = self._ctx_page_size if pool is None else pool[0].size(-2)
        self._ctx_page_size = page_size
        # Settle the block table before committing to the pool: it is the
        # last thing that can rule the pool out, and falling back after
        # taking the pool branch would leave no buffer allocated at all.
        if (
            not unified
            and pool is not None
            and not self._init_ctx_block_offsets(
                draft_kv_cache_manager, pool, L, nkv, hd, page_size
            )
        ):
            pool = None
        if pool is None:
            # --- DFlash2 context ring (ops/spec/dflash2_ring) ---
            # A drafter that reads only a sliding window keeps W + verify + block rows per
            # slot instead of max_seq_len + block.
            ring_window = getattr(draft_model, "_ctx_ring_window", None)
            if ring_window:
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
            # HND layout consumed by both FlashInfer's paged append and the
            # TRTLLM-Gen launcher: [L, pages, K/V, Hkv, page, D].
            self._ctx_kv_buf = torch.zeros(
                (L, total_pages, kv_factor, nkv, page_size, hd),
                dtype=dtype,
                device="cuda",
            )
            self._ctx_page_table = torch.arange(total_pages, dtype=torch.int32, device="cuda").view(
                num_slots, self._ctx_pages_per_slot
            )
            self._ctx_kv_indptr = torch.arange(
                0,
                total_pages + 1,
                self._ctx_pages_per_slot,
                dtype=torch.int32,
                device="cuda",
            )
        else:
            # Nothing reserved: pages come from the manager's per-request
            # block table one iteration at a time, so the footprint follows
            # the sequences served rather than max_batch x max_seq_len.
            self._ctx_kv_buf = pool
            max_blocks = (
                draft_kv_cache_manager.draft_max_blocks_per_seq
                if unified
                else draft_kv_cache_manager.max_blocks_per_seq
            )
            self._alloc_ctx_block_tables(num_slots, max_blocks)
            if unified:
                self._init_unified_draft_tables(draft_kv_cache_manager)
            # Only a manager that publishes and matches its own blocks can
            # hand back a reused prefix's drafter K/V; the private arena and
            # an unpaired draft pool both start every request at 0.
            if not unified:
                self._ctx_reuse_addressable = bool(
                    getattr(draft_kv_cache_manager, "enable_joint_kv_cache_reuse", False)
                )
        self._ctx_kv_last_page_len = torch.full(
            (num_slots,), page_size, dtype=torch.int32, device="cuda"
        )
        self._ctx_buf_inited = True

        if (
            not unified
            and not self._ctx_reuse_addressable
            and getattr(draft_kv_cache_manager, "enable_joint_kv_cache_reuse", False)
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

    def _assign_slot(self, req_id: int, reset: bool = False, updates: Optional[dict] = None):
        """Get (or refresh) this request's context slot; None when none is free.

        ``reset`` recycles an existing slot, which is how a reused request id is
        told apart from a continuation. Pass ``updates`` to defer the length
        resets into a caller's batch instead of writing _ctx_len per request;
        either way the host mirror stays in step with the device tensor.
        """

        def clear(slot: int) -> None:
            if updates is None:
                self._write_ctx_len({slot: 0})
            else:
                updates[slot] = 0

        if reset and req_id in self._req_to_slot:
            old_slot = self._req_to_slot.pop(req_id)
            clear(old_slot)
            self._free_slots.append(old_slot)
        if req_id not in self._req_to_slot:
            if not self._free_slots:
                if self._has_unified_draft_cache():
                    raise RuntimeError(
                        "Unified DFlash2 draft cache has no free request staging slot"
                    )
                return None
            slot = self._free_slots.popleft()
            self._req_to_slot[req_id] = slot
            if self._has_unified_draft_cache():
                self._managed_cache_bindings.pop(req_id, None)
                self._ctx_position_offset[slot].zero_()
            clear(slot)
            self._queue_draft_page_seed(req_id, True, slot, updates, reset)
        return self._req_to_slot[req_id]

    def _store_context_kv_paged(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        rows: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """Write the MLA drafter's context latents ``k`` at ``positions`` by direct page writes into
        plane 0 (one latent per token and layer: ``v`` is None)."""
        if v is not None:
            raise ValueError("The DFlash2 MLA drafter stores one latent per token (v is None).")
        if self._ctx_ring_rows and self._ctx_block_tables is None:
            # DFlash2 context ring: position p is slot row p mod _ctx_ring_rows.
            positions = dflash2_ring_positions(positions, self._ctx_ring_rows)
        table = (
            self._ctx_block_tables if self._ctx_block_tables is not None else self._ctx_page_table
        )
        page_size = self._ctx_page_size
        pages = table[rows.long(), positions.long() // page_size].long()
        offsets = positions.long() % page_size
        pool_all = self._ctx_kv_buf
        if (
            isinstance(pool_all, torch.Tensor)
            and pool_all.dim() == 6
            and k.dim() == 4
            and k.dtype == pool_all.dtype
            and k.size(1) == pool_all.size(0)
        ):
            # Every layer's latent [N, L, 1, D] in one indexed store: the per-layer stores below, at once.
            pool_all[:, pages, 0, :, offsets] = k
            return
        for layer_idx in range(k.size(1)):
            pool = self._ctx_kv_buf[layer_idx]
            new = k[:, layer_idx].to(pool.dtype)
            pool[pages, 0, :, offsets] = new

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
        """Refresh the attention metadata after the draft forward's kv_lens prep: host state only for a drafter
        that reads none of it (the MLA drafter) over the family DSA metadata, whose device-side indexer and
        sparse-MLA metadata the next step's ``on_update_kv_lens`` rebuilds; all of it otherwise."""
        if getattr(self, "_draft_reads_no_attn_metadata", False) and isinstance(
            attn_metadata, K31DSAtrtllmAttentionMetadata
        ):
            attn_metadata.update_for_spec_dec(refresh_device_metadata=False)
            return
        attn_metadata.update_for_spec_dec()

    def _k31_accept_record(
        self, attn_metadata, spec_metadata, num_accepted_tokens, num_contexts: int, batch_size: int
    ) -> bool:
        """The bookkeeping between acceptance and the drafter in one launch (``k31_dflash2_accept_record``):
        ``update_mamba_states`` of the K3 hybrid manager (its KDA replay state's ``record_acceptance``), then
        ``_prepare_attn_metadata_for_dflash`` and ``_prepare_kv_for_draft_forward`` with their device halves (the
        rewind ``1 - count``, ``kv_lens + 1``) in that launch; bitwise. Serves the family manager, whose replay state
        declares ``fused_accept_record`` (``ops/ssm/kda_cache_manager``). False where the torch path runs: context
        requests in the step, another manager or replay state."""
        num_gens = batch_size - num_contexts
        manager = attn_metadata.kv_cache_manager
        if (
            num_contexts != 0
            or not 0 < num_gens <= K31_BOOKKEEPING_MAX_REQUESTS
            or not isinstance(manager, KimiK3HybridCacheManager)
            or type(manager).update_mamba_states is not KimiK3HybridCacheManager.update_mamba_states
            or type(self)._prepare_kv_for_draft_forward
            is not K31DFlashWorker._prepare_kv_for_draft_forward
            or not hasattr(attn_metadata, "kv_lens_cuda")
            or num_accepted_tokens.dtype != torch.int32
            or attn_metadata.kv_lens_cuda.dtype != torch.int32
        ):
            return False
        replay = getattr(manager, "_speculative_state", None)
        state_indices = attn_metadata.mamba_metadata.state_indices
        is_dummy = manager._dummy_request_mask
        if (
            is_dummy is None
            or manager.kda_context_only
            or manager.local_num_mamba_layers == 0
            or not isinstance(replay, KDAReplayState)
            or not getattr(replay, "fused_accept_record", False)
            or replay.prev_num_accepted_tokens is None
            or replay.prev_num_accepted_tokens.dtype != torch.int32
            or replay.prev_num_accepted_tokens.dim() != 1
            or state_indices.dtype not in (torch.int32, torch.int64)
            or not all(
                t[:num_gens].is_contiguous()
                for t in (num_accepted_tokens, state_indices, is_dummy, attn_metadata.kv_lens_cuda)
            )
        ):
            return False
        logger.info_once(
            "DFlash2 accept record: KDA replay counts and drafter KV lengths in one launch "
            "(ops/spec/dflash2_step_bookkeeping).",
            key="k31_dflash2_fused_accept_record",
        )
        self._prepare_attn_metadata_for_dflash(attn_metadata, spec_metadata)
        self._kv_rewind_nc = num_contexts
        self._kv_rewind_bs = batch_size
        self._kv_rewind_amount = k31_dflash2_accept_record(
            num_accepted_tokens[:num_gens],
            state_indices[:num_gens],
            is_dummy[:num_gens],
            replay.prev_num_accepted_tokens,
            attn_metadata.kv_lens_cuda[:num_gens],
        )
        self._kv_rewind_pending = True
        self._update_metadata_for_draft(attn_metadata)
        return True

    def _k31_draft_tail(
        self,
        attn_metadata,
        spec_metadata,
        accepted_tokens,
        next_draft_tokens,
        batch_size: int,
        num_accepted_tokens,
    ) -> Optional[torch.Tensor]:
        """``_apply_kv_rewind_after_draft``, ``_rollback_guided_decoder_after_verify`` (without a guided decoder:
        nothing) and ``_prepare_next_new_tokens`` with their device work in one launch (``k31_dflash2_draft_tail``):
        the next input tokens, bitwise; None where the torch path runs."""
        rewind = self._kv_rewind_amount
        batch_rows = spec_metadata.batch_indices_cuda
        if (
            self.guided_decoder is not None
            or type(self)._apply_kv_rewind_after_draft
            is not _EngineDFlashWorker._apply_kv_rewind_after_draft
            or type(self)._prepare_next_new_tokens is not SpecWorkerBase._prepare_next_new_tokens
            or batch_size > K31_BOOKKEEPING_MAX_REQUESTS
            or batch_rows is None
            or batch_rows.dtype not in (torch.int32, torch.int64)
            or accepted_tokens.dtype != torch.int32
            or next_draft_tokens.dtype != torch.int32
            or num_accepted_tokens.dtype != torch.int32
            or not (accepted_tokens.is_contiguous() and next_draft_tokens.is_contiguous())
            or next_draft_tokens.shape[0] != batch_size
            or num_accepted_tokens.shape[0] != batch_size
        ):
            return None
        self._kv_rewind_pending = False
        is_warmup = spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing()
        apply_rewind = (
            not is_warmup and rewind is not None and hasattr(attn_metadata, "kv_lens_cuda")
        )
        if apply_rewind:
            kv_lens = attn_metadata.kv_lens_cuda[self._kv_rewind_nc : self._kv_rewind_bs]
            if (
                kv_lens.dtype != torch.int32
                or rewind.dtype != torch.int32
                or kv_lens.shape != rewind.shape
                or not rewind.is_contiguous()
            ):
                self._kv_rewind_pending = True
                return None
        else:
            # The kernel requires unit-stride placeholders. A one-request TP count can be a
            # column of the agreement buffer, and slicing it to zero retains the column stride.
            kv_lens = rewind = num_accepted_tokens.new_empty((0,))
        return k31_dflash2_draft_tail(
            kv_lens,
            rewind,
            accepted_tokens,
            batch_rows[:batch_size],
            num_accepted_tokens,
            next_draft_tokens,
            apply_rewind,
        )

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
        top_ks, top_ps = resolve_advanced_sampling_filters(advanced_sampling_mode, top_ks, top_ps)
        return flashinfer_sample_from_logits(logits, temperatures, top_ks, top_ps, seed, offset)

    def _compute_probs_from_logits(
        self, advanced_sampling_mode, logits, temperatures, top_ks, top_ps, min_ps
    ):
        if advanced_sampling_mode.is_fused:
            # 33-96 rows: <= 32-row chunks on the sampler's multi-CTA family, on parallel streams.
            streams = getattr(self, "_k31_probs_split_streams", None)
            if streams is None:
                streams = self._k31_probs_split_streams = SplitStreams()
            return fused_compute_probs_split(logits, temperatures, top_ks, top_ps, min_ps, streams)
        top_ks, top_ps = resolve_advanced_sampling_filters(advanced_sampling_mode, top_ks, top_ps)
        if _topp_cluster_serves(logits, temperatures, top_ks, top_ps):
            # NO_TOPK (temperature + per-row top-p): the cluster top-p kernel.
            return k31_topp_probs_cluster(logits, temperatures, top_ps)
        return flashinfer_compute_probs_from_logits(logits, temperatures, top_ks, top_ps)

    def _draft_block_logits(self, draft_model, hidden_states, attn_metadata):
        """The drafter's FP32 block logits ``draft_model.logits_processor(hidden_states, lm_head, ...,
        True)``: under plain TP with the column-parallel, gathered LM head (no vocab padding, even shards),
        this rank's shard of the GEMM, then the gather and widening in one multicast kernel
        (``lamport_vocab_gather``, the engine all-gather + ``.float()`` where it does not serve)."""
        lm_head = draft_model.lm_head
        if (
            self._pair_allreduce is not None
            and getattr(lm_head, "tp_mode", None) == "column"
            and getattr(lm_head, "gather_output", False)
            and getattr(lm_head, "padding_size", 0) == 0
            and getattr(lm_head, "gather_output_sizes", None) is None
            and lm_head.tp_size > 1
        ):
            local = lm_head.apply_linear(hidden_states, lm_head.bias)
            return lamport_vocab_gather(local, lm_head.mapping, self._pair_allreduce)
        return draft_model.logits_processor(hidden_states, lm_head, attn_metadata, True)

    def _dflash2_global_top_k(self, gen_logits, spec_metadata, top_k: int, full_vocab: int):
        """Per-position global top-k candidates and the tensor to score into (the engine method, its
        all-gather of the vocab shards' (id, value) pairs through the family's multicast all-gather).

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
            gathered = lamport_allgather(combined, mapping, self._pair_allreduce)
            unary_logits, candidate_ids = k31_dflash2_merge_top32(gathered.contiguous(), top_k)
        else:
            local_values, local_ids = torch.topk(gen_logits, top_k, dim=-1)
            local_ids = local_ids + mapping.tp_rank * shard
            # Interleaved (id, value) pairs.
            combined = torch.stack([local_ids.float(), local_values.float()], dim=-1).flatten(-2)
            gathered = lamport_allgather(combined, mapping, self._pair_allreduce)
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
            return dflash2_sample_and_accept(self, logits, attn_metadata, spec_metadata)
        return SpecWorkerBase.sample_and_accept_draft_tokens(
            self, logits, attn_metadata, spec_metadata
        )

    def _advance_ctx_len(self, slots: torch.Tensor, num_accepted: torch.Tensor) -> None:
        """Add this step's accepted tokens to the context lengths of ``slots``, on the device.

        Padding, warmup and attention-DP dummy rows share the dummy slot, which no request owns
        or frees. Emptying it right after the add keeps it at 0 between steps: those rows attend
        no context, and their next context K / V starts at column 0, inside the padding
        request's own pages (past them, its page-table row maps page 0, another request's).
        The store is to a fixed slot, so a captured step replays it.
        """
        self._ctx_len[slots] += num_accepted
        if self._has_unified_draft_cache():
            # Positions keep advancing past the capped history length.
            self._ctx_position_offset[slots] += (self._ctx_len[slots] - self._max_ctx).clamp_min(0)
        self._ctx_len.clamp_(max=self._max_ctx)
        self._ctx_len[self._dummy_slot].zero_()

    def _write_ctx_len(self, updates: dict[int, int]) -> None:
        """Apply a slot-to-length mapping to _ctx_len in one async scatter."""
        if not updates:
            return
        for slot, value in updates.items():  # Update host side mirror
            self._ctx_len_host[slot] = value

        slots, values = zip(*updates.items())
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
        if not hasattr(draft_model, "fc") or not hasattr(draft_model, "hidden_norm"):
            return

        num_ctx_tokens = attn_metadata.num_ctx_tokens
        if num_ctx_tokens == 0:
            return

        # The KV cache manager's draft pages (unified draft cache): each chunk's tail is stored there
        # too, so the request's window travels with its KV and stays with its reusable blocks.
        pages_manager = dflash2_draft_pages_manager(
            getattr(attn_metadata, "kv_cache_manager", None)
        )
        if self._context_only_server(draft_model, attn_metadata):
            captured_hs = spec_metadata.get_hidden_states(total_target_tokens)
            if captured_hs is not None:
                self._store_context_window(
                    draft_model,
                    spec_metadata,
                    attn_metadata,
                    captured_hs,
                    pages_manager,
                    position_ids,
                )
            return
        planner = (
            self._draft_page_planner(pages_manager, draft_model._window)
            if pages_manager is not None and not self._has_unified_draft_cache()
            else None
        )
        page_latents = []
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
        # Uploaded once after the loop, like ctx_len_updates (no per-request sync).
        position_offset_updates = {}
        offset = 0
        num_contexts = attn_metadata.num_contexts
        # A context write addresses pages through row i of the block table, so
        # it must fit that row's allocation as well as the arena's length.
        # The unified draft cache builds the counts on the host; only the
        # device-decoded table of a non-unified pool needs one tolist() for the
        # whole loop. The lengths come from _ctx_len_host, so the loop body
        # below stays sync-free.
        counts_host = self._ctx_block_counts_host
        if self._ctx_block_tables is None:
            ctx_alloc = None
        elif counts_host is not None and len(counts_host) >= num_contexts:
            ctx_alloc = [count * self._ctx_page_size for count in counts_host[:num_contexts]]
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
            # preemption -- has to start from a clean slot. Managed (unified)
            # preparation already binds fresh histories and records their
            # snapshot slots; keep those slots stable through the forward pass.
            reset = (
                not self._has_unified_draft_cache() and self._req_ctx_pos.get(req_id) != first_pos
            )
            if self._assign_slot(req_id, reset=reset, updates=ctx_len_updates) is None:
                logger.warning("DFlash: no free slots, skipping context store")
                self._req_ctx_pos.pop(req_id, None)
                self._plan_page_tail(
                    planner, page_latents, draft_model, req_id, first_pos, chunk_proj, None
                )
                offset += slen
                continue

            slot = self._req_to_slot[req_id]
            if (
                planner is not None
                and first_pos > 0
                and ctx_len_updates.get(slot, self._ctx_len_host[slot]) == 0
                and self._seed_ring_at(draft_model, pages_manager, req_id, slot, first_pos)
            ):
                # A reused prefix brought its window in the draft pages: the ring holds it now,
                # and this chunk continues at its absolute position.
                ctx_len_updates[slot] = first_pos
            cur = ctx_len_updates.get(slot, self._ctx_len_host[slot])
            cap = self._max_ctx if ctx_alloc is None else min(self._max_ctx, ctx_alloc[i])
            if cur == 0 and first_pos > 0 and self._ctx_reuse_addressable:
                # Reuse let the target skip [0, first_pos), so only the tail
                # reaches the capture buffer. The prefix K/V are still in the
                # paired pool: precompute_context_kv is per-token in (hidden, p).
                cur = first_pos
            if cur + slen > cap:
                if self._has_unified_draft_cache():
                    # The managed history cannot hold this chunk: drop it and
                    # keep the absolute position so later chunks stay aligned.
                    ctx_len_updates[slot] = 0
                    self._req_ctx_pos.pop(req_id, None)
                    position_offset_updates[slot] = first_pos + slen
                    self._managed_cache_bindings[req_id] = self._ctx_kv_manager.kv_cache_map[req_id]
                    offset += slen
                    continue
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
                self._plan_page_tail(
                    planner, page_latents, draft_model, req_id, first_pos, chunk_proj, None
                )
                offset += slen
                continue
            end = cur + slen
            actual = end - cur
            ring_k = None
            # Anchor to the true position rather than accumulating, and advance
            # even when the slot is full so the next chunk still reads as a
            # continuation.
            self._req_ctx_pos[req_id] = first_pos + slen
            if actual > 0:
                cache_dtype = self._ctx_kv_buf[0].dtype
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
                chunk_k, chunk_v = draft_model.precompute_context_kv(chunk_proj_cast, local_pos)
                if cur == first_pos:
                    # Absolute positions: the draft pages take these latents as they are.
                    ring_k = chunk_k
                # chunk_k/v: [actual, L, nkv, hd] → [L, actual, nkv, hd]
                # Manager block tables are keyed by batch position, the
                # private arena's by slot. See _ctx_paged_index_args.
                row = i if self._ctx_block_tables is not None else slot
                positions = local_pos
                store_k, store_v = chunk_k, chunk_v
                if self._ctx_ring_rows and self._ctx_block_tables is None:
                    # DFlash2 context ring: a chunk longer than the ring wraps onto itself;
                    # only its last _ctx_ring_rows positions can still be read.
                    keep = dflash2_ring_chunk_start(actual, self._ctx_ring_rows)
                    positions, store_k = positions[keep:], store_k[keep:]
                    store_v = None if store_v is None else store_v[keep:]
                self._store_context_kv_paged(
                    store_k,
                    store_v,
                    torch.full((positions.numel(),), row, dtype=torch.long, device="cuda"),
                    positions,
                )
            if self._has_unified_draft_cache():
                position_offset_updates[slot] = self._req_ctx_pos[req_id] - end
                self._managed_cache_bindings[req_id] = self._ctx_kv_manager.kv_cache_map[req_id]
            self._plan_page_tail(
                planner, page_latents, draft_model, req_id, first_pos, chunk_proj, ring_k
            )
            offset += slen
        self._write_ctx_len(ctx_len_updates)
        self._write_slot_values(self._ctx_position_offset, position_offset_updates)
        if planner is not None:
            writes = planner.writes(ctx_proj.device)
            if writes is not None:
                dflash2_store_draft_pages(
                    self._draft_page_pools(draft_model, pages_manager),
                    torch.cat(page_latents),
                    writes.pages,
                    writes.offsets,
                )
            self._draft_page_update = planner.history_update()

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
                # VocabShardedLogits.gather: the gathered [rows, vocab] float32 logits (one
                # multicast kernel on the group's Lamport workspace where it serves).
                logits = raw_logits = lamport_vocab_gather(
                    logits.local, logits.head.lm_head.mapping, self._pair_allreduce
                )

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
        if bool(getattr(draft_kv_cache_manager, "draft_layer_ids", ())):
            if not self._ctx_buf_inited or self._ctx_kv_manager is not draft_kv_cache_manager:
                raise RuntimeError("Unified DFlash2 draft inputs must be prepared before forward")
        else:
            self._lazy_init_ctx_buffers(
                draft_model, spec_metadata, attn_metadata, draft_kv_cache_manager
            )
            spec_metadata._dflash2_worker = self
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

        # The drafter's context projection on a side stream from here where the acceptance issues no collective
        # (a sampled DFlash2 step on the gathered logits), else right after the acceptance (the vocab-parallel
        # target token selection is a collective).
        ctx_early = (
            num_contexts == 0
            and not logits_vocab_shard
            and dflash2_sampled_step(self, spec_metadata)
        )
        if ctx_early:
            self._k31_context_begin(draft_model, attn_metadata, spec_metadata, input_ids.shape[0])
        accepted_tokens, num_accepted_tokens = self.sample_and_accept_draft_tokens(
            logits, attn_metadata, spec_metadata
        )
        if not ctx_early and num_contexts == 0:
            self._k31_context_begin(draft_model, attn_metadata, spec_metadata, input_ids.shape[0])

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

        if not self._k31_accept_record(
            attn_metadata, spec_metadata, num_accepted_tokens, num_contexts, batch_size
        ):
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

        # A context-KV-only drafter (disaggregated context server) stores the
        # prompt's drafter context above and never drafts.
        drafts = not draft_model.context_kv_only
        if not drafts and num_gens > 0:
            self._check_no_real_generation(spec_metadata, attn_metadata, num_contexts, batch_size)
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
        # The drafter inputs join a side-stream context projection; never left pending past here.
        self._k31_context_join()

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
                elif (
                    shard_tokens := self._k31_greedy_selector_tokens(
                        draft_model, gen_hidden_states, inputs, spec_metadata, num_gens, K
                    )
                ) is not None:
                    gen_draft_tokens, vocab_size = shard_tokens
                elif (
                    shard_tokens := self._k31_sampled_selector_tokens(
                        draft_model,
                        gen_hidden_states,
                        inputs,
                        spec_metadata,
                        num_contexts,
                        batch_size,
                        num_gens,
                        K,
                    )
                ) is not None:
                    gen_draft_tokens, vocab_size = shard_tokens
                else:
                    gen_logits = self._draft_block_logits(
                        draft_model, gen_hidden_states, attn_metadata
                    )

                    vocab_size = gen_logits.shape[-1]
                    gen_logits = gen_logits.reshape(num_gens, K, vocab_size)

                    gen_logits = self._refine_block_logits(
                        draft_model, gen_logits, inputs, spec_metadata
                    )

                    # DFlash 2: replace the independent per-position picks with
                    # one coherent path through the block (absent for plain
                    # DFlash).
                    if getattr(draft_model, "has_candidate_selector", False):
                        if dflash2_sampled_step(self, spec_metadata):
                            # DFlash2 sampling batch: the sampled selector walk
                            # (dflash2_sampling), verified next step by block verification.
                            gen_draft_tokens = dflash2_sample_draft(
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
            # Warmup and attention-DP padding rows of a context-KV-only drafter
            # get the same zero placeholder as context requests.
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
        next_new_tokens = self._k31_draft_tail(
            attn_metadata,
            spec_metadata,
            accepted_tokens,
            next_draft_tokens,
            batch_size,
            num_accepted_tokens,
        )
        if next_new_tokens is None:
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

        outputs = {
            # The gathered float32 target logits, or this rank's vocab shard
            # when the target tokens were selected from it.
            "logits": raw_logits,
            "logits_vocab_shard": logits_vocab_shard,
            "new_tokens": accepted_tokens,
            "new_tokens_lens": num_accepted_tokens,
            "next_draft_tokens": next_draft_tokens,
            "next_new_tokens": next_new_tokens,
        }
        agree_draft_tokens(outputs, self.mapping)
        return outputs

    def _k31_greedy_selector_tokens(
        self, draft_model, gen_hidden_states, inputs, spec_metadata, num_gens: int, K: int
    ) -> Optional[tuple[torch.Tensor, int]]:
        """The DFlash2 selector's all-greedy draft tokens from this rank's vocab shard of the block logits:
        ``(tokens [num_gens, K] int32, vocab)``, or None where the gathered-logits path runs.

        The gathered path (``_draft_block_logits``, ``_apply_dflash2_selector``, ``sample_draft_tokens``) all-gathers
        the FP32 block logits, takes their top-32 candidates, walks the selector into a ``-inf`` full-vocab block and
        argmaxes it. Here each rank takes its shard's top-32 as index-ordered (id, value) pairs, the ranks all-gather
        the pairs (``lamport_allgather``), and their merge is the same candidates in the same order
        (``ops/spec/dflash2_global_top32``); the walk then writes each slot's argmax token
        (``ops/spec/dflash2_selector_greedy_tokens``). The same tokens without the vocabulary gather, the block and
        its argmax: plain TP with the column-parallel gathered LM head, all-greedy drafting, the 32-candidate
        selector, no d2t / refinement."""
        lm_head = draft_model.lm_head
        selector = getattr(draft_model, "candidate_selector", None)
        if (
            self._pair_allreduce is None
            or spec_metadata.wants_advanced_draft_sampling
            or self._d2t is not None
            or not getattr(draft_model, "has_candidate_selector", False)
            or selector is None
            or dflash2_sampled_step(self, spec_metadata)
            or type(self)._refine_block_logits is not _EngineDFlashWorker._refine_block_logits
            or getattr(lm_head, "tp_mode", None) != "column"
            or not getattr(lm_head, "gather_output", False)
            or getattr(lm_head, "padding_size", 0) != 0
            or getattr(lm_head, "gather_output_sizes", None) is not None
            or lm_head.tp_size <= 1
        ):
            return None
        codebook = selector.predecessor_codebook
        full_vocab = codebook.shape[0]
        if (
            selector.top_k != K31_SELECTOR_TOP_K
            or codebook.shape[1] % 256
            or full_vocab % lm_head.tp_size
            or (full_vocab // lm_head.tp_size) % 8
            or full_vocab >= (1 << 24)
        ):
            return None
        logger.info_once(
            "DFlash2 all-greedy draft tokens from the vocab shard (top-32 pairs exchange, selector tokens).",
            key="k31_dflash2_shard_select",
        )
        local = lm_head.apply_linear(gen_hidden_states, lm_head.bias)
        shard = local.shape[-1]
        if shard * lm_head.tp_size != full_vocab or local.dtype not in (
            torch.bfloat16,
            torch.float32,
        ):
            return None
        pairs = k31_dflash2_shard_index_pairs(
            local.contiguous().view(num_gens, K, shard), lm_head.tp_rank * shard
        )
        gathered = lamport_allgather(pairs, lm_head.mapping, self._pair_allreduce)
        unary_logits, candidate_ids = k31_dflash2_merge_top32(
            gathered.contiguous(), K31_SELECTOR_TOP_K
        )
        tokens = k31_dflash2_selector_greedy_tokens(
            codebook,
            selector.successor_codebook,
            selector.hidden_projection.weight,
            candidate_ids,
            unary_logits,
            gen_hidden_states.reshape(num_gens, K, -1),
            inputs["first_prev_tokens"].long(),
            # The gathered path's block: the FP32 logits of the multicast vocab gather.
            torch.float32,
        )
        return tokens, full_vocab

    def _k31_sampled_selector_tokens(
        self,
        draft_model,
        gen_hidden_states,
        inputs,
        spec_metadata,
        num_contexts: int,
        batch_size: int,
        num_gens: int,
        K: int,
    ) -> Optional[tuple[torch.Tensor, int]]:
        """A sampled DFlash2 step's draft tokens with the selector's candidates from this rank's vocab shard:
        ``(tokens [num_gens, K] int32, vocab)``, or None where the gathered-logits path runs.

        The gathered path gathers the FP32 block logits (``_draft_block_logits``: the Lamport vocab gather),
        takes their top-32 candidates (``_dflash2_global_top_k``) and runs the sampled walk
        (``dflash2_sample_draft``). Here each rank takes its shard's top-32 as index-ordered ``(id, value)``
        pairs, the ranks all-gather the pairs (``lamport_allgather``) and their merge is the same candidates in
        the same order with the same logits, bit for bit: the pairs' Lamport mode keys the shard as the Lamport
        vocab gather delivers it and carries a ``-0.0`` logit through the pair gather whichever transport serves
        it (``ops/spec/dflash2_global_top32``). The walk, the proposal store and the draft tokens are then
        ``dflash2_sample_draft``'s. Plain TP with the column-parallel gathered BF16 LM head over a group whose
        vocab gather is the Lamport kernel, a sampled step, the 32-candidate selector, no d2t / refinement."""
        lm_head = draft_model.lm_head
        selector = getattr(draft_model, "candidate_selector", None)
        all_reduce = self._pair_allreduce
        if (
            all_reduce is None
            or not uses_mnnvl(all_reduce)
            or not dflash2_sampled_step(self, spec_metadata)
            or self._d2t is not None
            or not getattr(draft_model, "has_candidate_selector", False)
            or selector is None
            or type(self)._refine_block_logits is not _EngineDFlashWorker._refine_block_logits
            or type(self)._draft_block_logits is not K31DFlashWorker._draft_block_logits
            or type(self)._dflash2_global_top_k is not K31DFlashWorker._dflash2_global_top_k
            or getattr(lm_head, "tp_mode", None) != "column"
            or not getattr(lm_head, "gather_output", False)
            or getattr(lm_head, "padding_size", 0) != 0
            or getattr(lm_head, "gather_output_sizes", None) is not None
            or lm_head.tp_size <= 1
            or gen_hidden_states.dtype != torch.bfloat16
        ):
            return None
        codebook = selector.predecessor_codebook
        full_vocab = codebook.shape[0]
        if (
            selector.top_k != K31_SELECTOR_TOP_K
            or codebook.shape[1] % 256
            or full_vocab % lm_head.tp_size
            or (full_vocab // lm_head.tp_size) % 8
            or full_vocab >= (1 << 24)
        ):
            return None
        local = lm_head.apply_linear(gen_hidden_states, lm_head.bias)
        shard = local.shape[-1]
        if (
            shard * lm_head.tp_size != full_vocab
            or local.dtype != torch.bfloat16
            or not lamport_vocab_gather_serves(local, lm_head.mapping, all_reduce)
        ):
            # Not reached past the checks above on an MNNVL group; the gathered path recomputes the head.
            return None
        logger.info_once(
            "DFlash2 sampled draft candidates from the vocab shard (top-32 pairs exchange).",
            key="k31_dflash2_sampled_shard_select",
        )
        pairs = k31_dflash2_shard_index_pairs(
            local.view(num_gens, K, shard), lm_head.tp_rank * shard, True
        )
        gathered = lamport_allgather(pairs, lm_head.mapping, all_reduce)
        unary_logits, candidate_ids = k31_dflash2_merge_top32(
            gathered.contiguous(), K31_SELECTOR_TOP_K, True
        )
        tokens = dflash2_sample_draft_from_candidates(
            self,
            draft_model,
            candidate_ids,
            unary_logits,
            gen_hidden_states.reshape(num_gens, K, -1),
            inputs["first_prev_tokens"],
            spec_metadata,
            num_contexts,
            batch_size,
        )
        return tokens, full_vocab

    def _block_gather_ids(
        self, draft_model, num_gens: int, block_size: int, num_draft_tokens: int, num_rows: int
    ) -> torch.Tensor:
        """Block-output rows whose hidden states give the K draft logits.

        A function of the shapes only: kept per shape once built outside a CUDA-graph capture, so a captured step
        reads the stored index instead of rebuilding it (five small launches)."""
        key = (num_gens, block_size, num_draft_tokens, num_rows)
        cached = self._k31_gather_ids.get(key)
        if cached is not None:
            return cached
        ids = self._draft_slot_ids(draft_model, num_gens, block_size, num_draft_tokens)
        # Shields only the last request: at block_size == K with shift_label off, slots
        # run 1..K, so every request reads the next one's slot 0 and the last overruns.
        # Degrades, never raises.
        ids = ids.clamp(max=num_rows - 1)
        if not torch.cuda.is_current_stream_capturing():
            self._k31_gather_ids[key] = ids
        return ids

    def _k31_fused_gen_inputs(
        self,
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: DFlashSpecMetadata,
        draft_model: nn.Module,
        total_target_tokens: int,
        embed_tokens: nn.Module,
        mask_token_id: int,
        trained_mask_embedding: Optional[torch.Tensor],
    ) -> Optional[dict]:
        """``prepare_1st_drafter_inputs``' result for a step of generation requests only, with the per-request
        bookkeeping, its side effects and the head of the block decode in one launch
        (``kimi_k31_pro_mxfp4__sm_103__tep/ops/spec/dflash2_gen_prepare``, bitwise the torch path below), or None where
        that path runs.

        Applies to the private paged arena (no manager block tables or dense buffer), captured
        target features drafting from all ``K + 1`` target rows, a BF16 single-latent arena of every layer
        (the indexed store), an unsharded BF16 embedding and int32 accepted tokens and counts."""
        num_gens = attn_metadata.num_seqs - attn_metadata.num_contexts
        width = spec_metadata.runtime_draft_len + 1
        pool_all = self._ctx_kv_buf
        table = self._ctx_page_table
        weight = getattr(embed_tokens, "weight", None)
        if (
            not self._k31_fused_prepare_serves(
                attn_metadata, spec_metadata, draft_model, embed_tokens
            )
            or accepted_tokens.dtype != torch.int32
            or num_accepted_tokens.dtype != torch.int32
        ):
            # An overlapped context projection is joined before the torch path projects on this stream.
            self._k31_context_join()
            return None
        if trained_mask_embedding is not None:
            mask_row = trained_mask_embedding
        elif 0 <= mask_token_id < weight.shape[0]:
            mask_row = weight[mask_token_id]
        else:
            self._k31_context_join()
            return None
        if mask_row.dtype != torch.bfloat16 or mask_row.shape != weight.shape[1:]:
            self._k31_context_join()
            return None
        captured_hs = spec_metadata.get_hidden_states(total_target_tokens)
        if captured_hs is None:
            self._k31_context_join()
            return None
        slots = self._batch_to_slot_buffer.tensor[:num_gens]
        logger.info_once(
            "DFlash2 drafter inputs: one-launch generation prepare "
            "(kimi_k31_pro_mxfp4__sm_103__tep/ops/spec/dflash2_gen_prepare).",
            key="k31_dflash2_fused_gen_prepare",
        )
        prep = k31_dflash2_gen_prepare(
            num_accepted_tokens[:num_gens].contiguous(),
            accepted_tokens[:num_gens].contiguous(),
            slots,
            self._ctx_len,
            table,
            weight,
            mask_row.contiguous(),
            attn_metadata._seq_lens_cuda[:num_gens],
            self._compute_block_size,
            self._max_ctx,
            self._dummy_slot,
            self._ctx_ring_rows or 0,
            self._ctx_page_size,
        )
        attn_metadata._seq_lens[:num_gens] = width
        latents = self._k31_context_join()
        if latents is None:
            gen_start = attn_metadata.num_ctx_tokens
            gen_hs = captured_hs[gen_start : gen_start + num_gens * width]
            projected = draft_model.project_target_hidden(gen_hs.reshape(-1, gen_hs.shape[-1]))
            latents = draft_model.k31_context_latents(projected.to(pool_all.dtype))
        # Every layer's context latent of the accepted rows into the arena (the rest times zero), one launch.
        draft_model.k31_store_context_latents(
            latents,
            prep.ctx_positions,
            pool_all,
            prep.store_rows,
            prep.store_mask,
        )
        return {
            "noise_embedding": prep.noise,
            "query_positions": prep.query_positions,
            "num_ctx_per_req": prep.num_ctx,
            "ctx_cache_batch_idx": slots,
            "first_prev_tokens": prep.bonus,
            "ctx_kv_cache": pool_all,
            "ctx_page_table": table,
            "block_placement": K31BlockStore(prep.page_rows, prep.block_pages, prep.block_offsets),
        }

    def _k31_fused_prepare_serves(
        self, attn_metadata, spec_metadata, draft_model: nn.Module, embed_tokens: nn.Module
    ) -> bool:
        """The step's conditions of :meth:`_k31_fused_gen_inputs` other than the accepted tokens' and counts'
        dtypes and the mask row."""
        num_contexts = attn_metadata.num_contexts
        num_gens = attn_metadata.num_seqs - num_contexts
        width = spec_metadata.runtime_draft_len + 1
        pool_all = self._ctx_kv_buf
        table = self._ctx_page_table
        weight = getattr(embed_tokens, "weight", None)
        return not (
            num_contexts != 0
            or num_gens <= 0
            or width != self._draft_tokens_per_req
            or not (hasattr(draft_model, "fc") and hasattr(draft_model, "hidden_norm"))
            or self._ctx_block_tables is not None
            or self._has_unified_draft_cache()
            or table is None
            or table.dtype != torch.int32
            or not isinstance(pool_all, torch.Tensor)
            or pool_all.dim() != 6
            or pool_all.dtype != torch.bfloat16
            or pool_all.size(0) != draft_model._num_attn_layers
            or not pool_all.is_contiguous()
            or not _k31_plain_embedding(embed_tokens)
            or weight is None
            or weight.dtype != torch.bfloat16
            or weight.shape[1] % 8
            or not weight.is_contiguous()
            or attn_metadata._seq_lens_cuda.dtype != torch.int32
        )

    def _k31_context_begin(
        self, draft_model: nn.Module, attn_metadata, spec_metadata, total_target_tokens: int
    ) -> None:
        """Issue the generation rows' drafter context projection (``project_target_hidden``: the fc GEMM, its
        all-gather and ``hidden_norm``; then every layer's latent GEMM, ``k31_context_latents``) on a side stream:
        it reads only the captured target rows, so it runs beside the acceptance and the drafter's input
        bookkeeping, and :meth:`_k31_fused_gen_inputs` joins it before the latents' norm / RoPE store. The same
        kernels on the same operands, so bitwise the serial order.

        The caller issues no collective on the current stream between this call and the join, and the join
        precedes the drafter's first all-reduce, so each rank runs the group's Lamport collectives in the same
        order as without the side stream. Nothing is issued (the projection stays after the acceptance) outside
        multi-stream execution, under torch.compile, or where the one-launch prepare does not serve."""
        self._k31_ctx_pending = None
        if (
            not do_multi_stream()
            or torch.compiler.is_compiling()
            or getattr(draft_model, "context_kv_only", False)
            or not hasattr(draft_model, "k31_context_latents")
            or not self._k31_fused_prepare_serves(
                attn_metadata,
                spec_metadata,
                draft_model,
                draft_model.model.embed_tokens,
            )
        ):
            return
        captured_hs = spec_metadata.get_hidden_states(total_target_tokens)
        if captured_hs is None:
            return
        num_gens = attn_metadata.num_seqs - attn_metadata.num_contexts
        width = spec_metadata.runtime_draft_len + 1
        gen_start = attn_metadata.num_ctx_tokens
        gen_hs = captured_hs[gen_start : gen_start + num_gens * width]
        if draft_model._fused_kv_weight is None:
            draft_model._build_fused_kv_buffers()
        side = getattr(self, "_k31_ctx_side", None)
        if side is None:
            side = self._k31_ctx_side = _K31SideStream()
        logger.info_once(
            "DFlash2 drafter context projection on a side stream beside the acceptance.",
            key="k31_dflash2_ctx_overlap",
        )
        current = torch.cuda.current_stream()
        side.fork.record(current)
        with torch.cuda.stream(side.stream):
            side.fork.wait(side.stream)
            projected = draft_model.project_target_hidden(gen_hs.reshape(-1, gen_hs.shape[-1]))
            latents = draft_model.k31_context_latents(projected.to(self._ctx_kv_buf.dtype))
            side.join.record(side.stream)
        self._k31_ctx_pending = latents

    def _k31_context_join(self) -> Optional[torch.Tensor]:
        """The latents :meth:`_k31_context_begin` issued, the current stream made to wait for them; None when
        nothing is pending."""
        latents = getattr(self, "_k31_ctx_pending", None)
        if latents is None:
            return None
        self._k31_ctx_pending = None
        current = torch.cuda.current_stream()
        self._k31_ctx_side.join.wait(current)
        if not torch.cuda.is_current_stream_capturing():
            # Allocated in side-stream order and read on this stream.
            latents.record_stream(current)
        return latents

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
        - ctx_cache_batch_idx / ctx_kv_cache / ctx_page_table: the drafter's rows of its
          paged latent pool, the pool and its page table.
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
        embed_tokens = draft_model.model.embed_tokens
        hidden_dim = (
            spec_metadata.hidden_size if spec_metadata.hidden_size > 0 else hidden_states.shape[-1]
        )
        # The drafter's own trained mask row, if it kept one for the mask id in use.
        trained_mask_embedding = (
            getattr(draft_model, "mask_token_embedding", None)
            if mask_token_id == getattr(draft_model, "mask_token_id", None)
            else None
        )

        if num_gens > 0:
            fused = self._k31_fused_gen_inputs(
                accepted_tokens,
                num_accepted_tokens,
                attn_metadata,
                spec_metadata,
                draft_model,
                total_target_tokens,
                embed_tokens,
                mask_token_id,
                trained_mask_embedding,
            )
            if fused is not None:
                return fused
        if num_gens > 0:
            gen_num_accepted = num_accepted_tokens[num_contexts : num_contexts + num_gens]
            gen_accepted_tokens = accepted_tokens[num_contexts : num_contexts + num_gens, :]

            total_tokens_per_req = self._draft_tokens_per_req  # K+1
            K = spec_metadata.runtime_draft_len

            # Get captured multi-layer hidden states from spec_metadata
            captured_hs = spec_metadata.get_hidden_states(total_target_tokens)
            has_target_features = (
                captured_hs is not None
                and hasattr(draft_model, "fc")
                and hasattr(draft_model, "hidden_norm")
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
                gen_num_accepted_long = gen_num_accepted.long()
                col_idx = self._ctx_len[slots].unsqueeze(1) + offsets_kp1.unsqueeze(0)
                write_mask = offsets_kp1.unsqueeze(0) < gen_num_accepted_long.unsqueeze(1)
                if self._ctx_block_tables is not None:
                    # Bound by what the manager handed THIS request: the masked
                    # entries below are still written (fixed-size, for graph
                    # safety) and a placeholder page resolves to another's.
                    gen_capacity = self._ctx_block_counts[gen_rows_out] * self._ctx_page_size
                    col_idx = torch.minimum(col_idx, (gen_capacity - 1).unsqueeze(1)).clamp_(min=0)
                if self._ctx_block_tables is None:
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
                cache_dtype = self._ctx_kv_buf[0].dtype
                k_new, v_new = draft_model.precompute_context_kv(
                    proj_flat.to(cache_dtype), pos_flat
                )
                mask_bc = mask_1d.view(-1, 1, 1, 1).to(k_new.dtype)
                k_new.mul_(mask_bc)
                slot_long = slot_flat.long()
                col_long = col_flat.long()
                if self._ctx_block_tables is not None:
                    # Batch positions of the gen requests, matching the
                    # per-request block table's row order.
                    rows_long = gen_rows_out.unsqueeze(1).expand(-1, K + 1).reshape(-1)
                else:
                    rows_long = slot_long
                self._store_context_kv_paged(k_new, v_new, rows_long, col_long)
                self._advance_ctx_len(slots, gen_num_accepted_long)

            num_ctx_per_req_t = self._ctx_len[slots]
            if self._ctx_block_tables is not None:
                # The write above clamps columns to the request's allocation, so
                # a context that outruns it has its tail written over the last
                # valid slot. Truncate what is advertised to match, or the read
                # attends that clobbered value -- or another request's page.
                allocated = dflash_allocated_ctx_limit(
                    self._ctx_block_counts, self._ctx_page_size, block_size
                )
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
            "noise_embedding": noise_embedding,
            "query_positions": query_positions,
            "num_ctx_per_req": num_ctx_per_req_t,
            # Slots index the private arena; the manager's block table is keyed
            # by batch position, so the drafter reads its own rows there.
            "ctx_cache_batch_idx": (gen_rows_out if self._ctx_block_tables is not None else slots),
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
# Vision tower
# ====================================================================================================================
#
# Kimi K3.1's image input path: the MoonViT vision tower and the splice of its rows into the prompt embeddings.
#
# ``K31VisionTower`` turns a step's processed images (``multimodal_data["image"]``: ``pixel_values`` ``[patches, 3,
# 14, 14]`` and ``image_grid_thw``) into one row per media placeholder token, in the text model's hidden width. Every
# rank runs the whole tower (all heads, no all-reduce) over the images of its own requests, in context steps only,
# eagerly.
#
# Per call: the patch embedding (``ops/vision/moonvit_patch_embed``), 27 pre-norm blocks (RMSNorm, the fused qkv
# projection, the 2-D rotary embedding, bidirectional attention within each image (``ops/vision/moonvit_attention``, the
# catalog ``thop_attention`` in its no-KV-cache mode), the output projection, RMSNorm, the GELU(tanh) MLP, each with its
# residual), the final RMSNorm, the 2x2 patch merge (``sd2_tpool``: the temporal mean of each merge window), and the
# projector (projection, exact GELU, projection, RMSNorm); the BF16 projections are ``ops/gemm/bf16_linear`` on
# ``F.linear``. Every kernel is the one the engine's MoonViT3d tower runs, in its order and dtypes, so the image rows
# are its bits wherever it ran the tower whole: attention DP, and attention TP of 1 or a size that does not divide the
# 12 heads (8, 16, 32). At attention TP 2 and 4 it split the heads over the ranks and all-reduced the projections, so
# the rows differ there by the summation order.
#
# ``image_embeddings`` and ``splice_image_rows`` are the target-side hooks: the step's image rows (encoded once per
# request, cached on the request's multimodal data for later chunks) and the prompt embeddings with those rows written
# over the placeholder positions the executor found.
#
# Images are encoded inline, so the engine's memory profile sees no encoder batch of its own: ``profiling_image_rows``
# encodes one image of the processor's largest grid inside the KV-capacity estimate's profiling forward, and the KV pool
# is sized with the tower's activations (and the image's rows, which an image step keeps beside the text model's)
# counted. The tower's checkpoint key prefixes are ``weights.py``'s.


def multimodal_disaggregated() -> bool:
    """Whether the deployment sets encoder / prefill disaggregation (``TLLM_MULTIMODAL_DISAGGREGATED``): the
    prefill server then runs no local vision tower."""
    return os.getenv("TLLM_MULTIMODAL_DISAGGREGATED", "0") == "1"


#: The vision config fields this tower builds, at the only value it builds them with.
_SUPPORTED_VARIANT = (
    ("norm_type", "rmsnorm"),
    ("mlp_type", "mlp2"),
    ("activation_func", "gelu_pytorch_tanh"),
    ("pos_emb_type", "divided_fixed"),
    ("attn_bias", False),
    ("patch_embed_proj_bias", False),
    ("linear_bias", False),
    ("merge_type", "sd2_tpool"),
)


def _config_dict(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, dict):
        return config
    return config.to_dict() if hasattr(config, "to_dict") else vars(config)


def _sincos_rows(dim: int, num_frames: int) -> np.ndarray:
    """``[num_frames, dim]`` 1-D sin-cos embedding of the frame index (sin half, then cos half)."""
    omega = np.arange(dim // 2, dtype=np.float32)
    omega /= dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", np.arange(num_frames, dtype=np.float32).reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class _Block(nn.Module):
    """One encoder block's weights, under the checkpoint's names."""

    def __init__(self, hidden: int, qkv_hidden: int, mlp_hidden: int, **factory) -> None:
        super().__init__()
        self.norm0 = nn.Parameter(torch.empty(hidden, **factory), requires_grad=False)
        self.norm1 = nn.Parameter(torch.empty(hidden, **factory), requires_grad=False)
        self.wqkv = nn.Parameter(
            torch.empty(3 * qkv_hidden, hidden, **factory), requires_grad=False
        )
        self.wo = nn.Parameter(torch.empty(hidden, qkv_hidden, **factory), requires_grad=False)
        self.fc0 = nn.Parameter(torch.empty(mlp_hidden, hidden, **factory), requires_grad=False)
        self.fc1 = nn.Parameter(torch.empty(hidden, mlp_hidden, **factory), requires_grad=False)


class K31VisionTower(nn.Module):
    """Kimi K3.1's MoonViT3d tower and ``PatchMergerMLPV2`` projector (module docstring).

    Built from the composite config's ``vision_config`` and the text model's hidden width; :meth:`load_weights`
    fills it from the checkpoint's ``vision_tower.*`` / ``mm_projector.*`` weights.
    """

    def __init__(
        self,
        vision_config: Any,
        text_hidden_size: int,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        cfg = _config_dict(vision_config)
        for field, supported in _SUPPORTED_VARIANT:
            value = cfg.get(field, supported)
            if value != supported:
                raise ValueError(
                    f"Kimi K3.1 vision tower supports {field}={supported!r}, got {value!r}"
                )
        self.num_heads = cfg.get("vt_num_attention_heads", cfg.get("num_attention_heads", 12))
        hidden = cfg.get("vt_hidden_size", cfg.get("hidden_size", 1024))
        num_layers = cfg.get("vt_num_hidden_layers", cfg.get("num_hidden_layers", 27))
        qkv_hidden = cfg.get("qkv_hidden_size", hidden)
        self.head_dim = qkv_hidden // self.num_heads
        self.qkv_hidden = qkv_hidden
        mlp_hidden = cfg.get("vt_intermediate_size", cfg.get("intermediate_size", 4096))
        mm_hidden = cfg.get("mm_hidden_size", hidden)
        self.patch_size = cfg.get("patch_size", 14)
        self.projector_eps = cfg.get("projector_ln_eps", 1e-5)
        pos_h = cfg.get("init_pos_emb_height", 64)
        pos_w = cfg.get("init_pos_emb_width", 64)
        self.num_frames = cfg.get("init_pos_emb_time", 4)
        merge = cfg.get("merge_kernel_size", [2, 2])
        self.merge_kernel_size: Tuple[int, int] = (
            (merge, merge) if isinstance(merge, int) else (int(merge[0]), int(merge[1]))
        )
        self.dtype = dtype
        self.text_hidden_size = text_hidden_size

        factory = dict(dtype=dtype, device=device)
        merged = mm_hidden * self.merge_kernel_size[0] * self.merge_kernel_size[1]
        self.patch_proj = nn.Parameter(
            torch.empty(hidden, 3, self.patch_size, self.patch_size, **factory), requires_grad=False
        )
        self.pos_emb = nn.Parameter(
            torch.empty(pos_h, pos_w, hidden, **factory), requires_grad=False
        )
        self.register_buffer(
            "time_weight",
            torch.from_numpy(_sincos_rows(hidden, self.num_frames))
            .float()
            .unsqueeze(1)
            .to(**factory),
            persistent=False,
        )
        self.blocks = nn.ModuleList(
            [_Block(hidden, qkv_hidden, mlp_hidden, **factory) for _ in range(num_layers)]
        )
        self.final_norm = nn.Parameter(torch.empty(hidden, **factory), requires_grad=False)
        self.merge_fc0 = nn.Parameter(torch.empty(merged, merged, **factory), requires_grad=False)
        self.merge_fc1 = nn.Parameter(
            torch.empty(text_hidden_size, merged, **factory), requires_grad=False
        )
        self.post_norm = nn.Parameter(torch.empty(text_hidden_size, **factory), requires_grad=False)
        # The 2-D rotary angle table, built on the device at the first forward.
        self._rope_table: Optional[torch.Tensor] = None
        # The (t, h, w) grid of the largest image the processor produces (``build_vision_tower``): the attention
        # capacity and the KV-capacity profiling image (``profiling_image_rows``).
        self.max_image_grid: Tuple[int, int, int] = self.largest_image_grid({})
        # The attention op sizes its workspace in place per call and never shrinks it.
        self._workspace = torch.empty(0, dtype=torch.int8, device=device)

    def _checkpoint_parameters(self) -> Dict[str, nn.Parameter]:
        """Checkpoint name -> the parameter it fills."""
        names = {
            "vision_tower.patch_embed.proj.weight": self.patch_proj,
            "vision_tower.patch_embed.pos_emb.weight": self.pos_emb,
            "vision_tower.encoder.final_layernorm.weight": self.final_norm,
            "mm_projector.proj.0.weight": self.merge_fc0,
            "mm_projector.proj.2.weight": self.merge_fc1,
            "mm_projector.post_norm.weight": self.post_norm,
        }
        for i, block in enumerate(self.blocks):
            prefix = f"vision_tower.encoder.blocks.{i}."
            names.update(
                {
                    f"{prefix}norm0.weight": block.norm0,
                    f"{prefix}norm1.weight": block.norm1,
                    f"{prefix}wqkv.weight": block.wqkv,
                    f"{prefix}wo.weight": block.wo,
                    f"{prefix}mlp.fc0.weight": block.fc0,
                    f"{prefix}mlp.fc1.weight": block.fc1,
                }
            )
        return names

    @torch.no_grad()
    def load_weights(self, weights) -> bool:
        """Fill every parameter from ``weights`` (name -> tensor or lazy safetensors slice), cast to the tower's
        dtype, and return True; return False, filling nothing, for a checkpoint with no tower weights. A tower weight
        the checkpoint lacks, or a checkpoint tower weight this tower has no place for, fails the load."""
        params = self._checkpoint_parameters()
        present = {
            k for k in weights.keys() if k.startswith((VISION_TOWER_PREFIX, PROJECTOR_PREFIX))
        }
        if not present:
            return False
        missing = sorted(set(params) - present)
        unknown = sorted(present - set(params))
        if missing or unknown:
            raise ValueError(
                f"Kimi K3.1 vision checkpoint: missing {missing[:5]} ({len(missing)}), "
                f"unknown {unknown[:5]} ({len(unknown)})"
            )
        for name, param in params.items():
            value = weights[name][:]
            if tuple(value.shape) != tuple(param.shape):
                raise ValueError(
                    f"Kimi K3.1 vision checkpoint: {name} has shape {tuple(value.shape)}, "
                    f"expected {tuple(param.shape)}"
                )
            param.copy_(value)
        logger.info("Loaded %d Kimi K3.1 vision tower / projector weights", len(params))
        return True

    def validate_multimodal_data(self, multimodal_data: Dict[str, Any]) -> None:
        """Reject processed media that this tower cannot encode.

        Every grid row ``(t, h, w)`` must fit the temporal position table, the 2-D rotary table, the patch merger and
        the attention capacity (the processor's largest image); all of a request's grids together must fit the
        encoder budget, the patches of the processor's largest image (the activations the KV-capacity estimate
        reserves, :func:`profiling_image_rows`); and the pixel payload must hold exactly the patches of its grids. The
        request is refused here, at admission, rather than failing the step that would encode it.

        Raises:
            ValueError: If a grid or a pixel payload is outside these limits.
        """
        image = multimodal_data.get("image")
        if not isinstance(image, dict) or image.get("image_grid_thw") is None:
            return
        kernel_h, kernel_w = self.merge_kernel_size
        max_patches = self.attention_capacity[0]
        grid_thws = torch.as_tensor(image["image_grid_thw"])
        if grid_thws.ndim != 2 or grid_thws.shape[-1] != 3:
            raise ValueError(
                f"image_grid_thw must have shape [num_grids, 3], got {tuple(grid_thws.shape)}"
            )
        num_patches = 0
        for t, h, w in grid_thws.tolist():
            if (
                not (1 <= t <= self.num_frames and 1 <= h <= TABLE_SIDE and 1 <= w <= TABLE_SIDE)
                or h % kernel_h
                or w % kernel_w
                or t * h * w > max_patches
            ):
                raise ValueError(
                    f"MoonViT cannot encode the image grid (t={t}, h={h}, w={w}); it supports "
                    f"t <= {self.num_frames}, h <= {TABLE_SIDE}, w <= {TABLE_SIDE}, h and w "
                    f"divisible by the merge kernel {self.merge_kernel_size}, and at most {max_patches} patches "
                    f"(the processor's largest image)"
                )
            num_patches += t * h * w
        if num_patches > max_patches:
            raise ValueError(
                f"The request's images have {num_patches} patches, more than the vision encoder budget of "
                f"{max_patches} (the processor's largest image); send fewer or smaller images"
            )
        pixel_values = image.get("pixel_values")
        patch_shape = (num_patches, 3, self.patch_size, self.patch_size)
        if isinstance(pixel_values, torch.Tensor) and tuple(pixel_values.shape) != patch_shape:
            raise ValueError(
                f"pixel_values must have shape {patch_shape} for its grids, got {tuple(pixel_values.shape)}"
            )

    def largest_image_grid(self, media_proc_cfg: Dict[str, Any]) -> Tuple[int, int, int]:
        """The largest image grid of the HF image processor ``media_proc_cfg`` (empty: its default limits).

        Raises:
            ValueError: If the processor limits admit no image grid this tower merges.
        """
        grid = moonvit_max_image_grid(media_proc_cfg, self.merge_kernel_size, self.patch_size)
        if grid is None:
            raise ValueError(
                f"The image processor limits {media_proc_cfg} admit no image grid of the vision tower's merge "
                f"kernel {self.merge_kernel_size}"
            )
        return grid

    @property
    def attention_capacity(self) -> Tuple[int, int]:
        """``(max_seq_len, max_num_requests)`` of every attention call: the patches of the largest image and as many
        sequences as the smallest images (one merge window each) that fill it."""
        t, h, w = self.max_image_grid
        kernel_h, kernel_w = self.merge_kernel_size
        return t * h * w, t * h * w // (kernel_h * kernel_w)

    def _attention(
        self, x: torch.Tensor, block: _Block, freqs_cis: torch.Tensor, seg_lens: List[int]
    ) -> torch.Tensor:
        """One block's attention: the fused qkv projection, q and k rotated, attention within each grid, ``wo``."""
        rows = x.shape[0]
        width = self.qkv_hidden
        qkv = bf16_linear(x, block.wqkv, False)
        q, k, v = qkv.split([width, width, width], dim=-1)
        q, k = moonvit_rope_2d(
            q.view(rows, self.num_heads, self.head_dim),
            k.view(rows, self.num_heads, self.head_dim),
            freqs_cis,
        )
        qkv = torch.concat([q.reshape(rows, -1), k.reshape(rows, -1), v], dim=-1)
        out = torch.empty([rows, width], dtype=x.dtype, device=x.device)
        moonvit_attention(
            qkv,
            out,
            seg_lens,
            *self.attention_capacity,
            self._workspace,
            self.num_heads,
            self.head_dim,
        )
        return bf16_linear(out, block.wo, False)

    def encode(self, pixel_values: torch.Tensor, grids: List[Tuple[int, int, int]]) -> torch.Tensor:
        """``[sum of merged cells, text hidden]`` rows of the patches of ``grids``, grid by grid."""
        device = pixel_values.device
        if self._rope_table is None:
            self._rope_table = moonvit_rope_table(
                self.head_dim, TABLE_SIDE, TABLE_SIDE, THETA, device
            )
        x = moonvit_patch_embed(
            pixel_values.to(dtype=self.dtype),
            self.patch_proj,
            self.pos_emb,
            self.time_weight,
            grids,
        )
        freqs_cis = moonvit_rope_angles(self._rope_table, grids)
        seg_lens = [t * h * w for t, h, w in grids]
        for block in self.blocks:
            x = x + self._attention(moonvit_rms_norm(x, block.norm0), block, freqs_cis, seg_lens)
            h = moonvit_rms_norm(x, block.norm1)
            h = moonvit_gelu(bf16_linear(h, block.fc0, False), "tanh")
            x = x + bf16_linear(h, block.fc1, False)
        x = moonvit_rms_norm(x, self.final_norm)

        # sd2_tpool: each kh x kw merge window of a grid as one row (window row-major, then the window's patches
        # row-major), averaged over the grid's frames.
        kernel_h, kernel_w = self.merge_kernel_size
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
        out = moonvit_gelu(bf16_linear(merged, self.merge_fc0, False), "none")
        out = bf16_linear(out, self.merge_fc1, False)
        return moonvit_rms_norm(out, self.post_norm, self.projector_eps)

    def encode_largest_image(self) -> torch.Tensor:
        """The rows of one zero image at ``max_image_grid``: the tower's largest single-image activations."""
        t, h, w = self.max_image_grid
        pixels = torch.zeros(
            t * h * w,
            3,
            self.patch_size,
            self.patch_size,
            dtype=self.dtype,
            device=self.patch_proj.device,
        )
        return self.encode(pixels, [(t, h, w)])

    @torch.inference_mode()
    def forward(self, multimodal_params: List[MultimodalParams]) -> List[torch.Tensor]:
        """The rows of every image of ``multimodal_params``, request by request, grid by grid, as one tensor in a
        list (empty when no request carries pixels). Each request is encoded on its own, so one encoder call holds
        at most the encoder budget that admission enforces (``validate_multimodal_data``)."""
        rows = []
        for param in multimodal_params:
            image = (param.multimodal_data or {}).get("image")
            if not isinstance(image, dict) or "pixel_values" not in image:
                continue
            grids = [
                (int(t), int(h), int(w))
                for t, h, w in torch.as_tensor(image["image_grid_thw"]).tolist()
            ]
            rows.append(self.encode(image["pixel_values"], grids))
        if not rows:
            return []
        return [rows[0] if len(rows) == 1 else torch.cat(rows, dim=0)]


def _processor_config(checkpoint_dir: Optional[str]) -> Optional[Dict[str, Any]]:
    """The ``media_proc_cfg`` of the checkpoint's HF image processor (its ``preprocessor_config.json``)."""
    path = os.path.join(checkpoint_dir, "preprocessor_config.json") if checkpoint_dir else None
    if path is None or not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f).get("media_proc_cfg") or {}


def build_vision_tower(
    composite_config: Any, checkpoint_dir: Optional[str] = None
) -> K31VisionTower:
    """The BF16 tower of a composite (text + vision) Kimi K3.1 config, on the current CUDA device, projecting into
    the text config's hidden width, with the largest image grid of the processor in ``checkpoint_dir`` (default: the
    config's own directory)."""
    vision_config = _config_dict(getattr(composite_config, "vision_config", None))
    text_hidden_size = _config_dict(getattr(composite_config, "text_config", None)).get(
        "hidden_size", vision_config.get("text_hidden_size", 7168)
    )
    tower = K31VisionTower(
        vision_config,
        text_hidden_size,
        dtype=torch.bfloat16,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    media_proc_cfg = _processor_config(
        checkpoint_dir or getattr(composite_config, "_name_or_path", None)
    )
    if media_proc_cfg is None:
        logger.warning(
            "Kimi K3.1: no preprocessor_config.json beside the checkpoint; the vision tower assumes the "
            "processor's default image limits."
        )
    else:
        tower.max_image_grid = tower.largest_image_grid(media_proc_cfg)
    return tower


@contextlib.contextmanager
def profiling_image_rows(
    tower: Optional[K31VisionTower], attn_metadata: Any, has_images: bool
) -> Iterator[None]:
    """Hold the rows of one largest image through a context step of the KV-capacity estimate.

    The engine sizes the KV pool after the peak memory of its profiling forward, a text-only context step under the
    estimation-phase KV cache manager. In such a step, without images of its own, this encodes one zero image of the
    processor's largest grid before the text model runs and keeps its rows until the step ends, as an image step
    keeps them; the peak then covers the tower's activations. Other steps are left alone.

    Every such step encodes, not only the first. The estimating executor first runs its warmup context steps under the
    same KV cache manager; the engine then resets the peak-memory statistic and runs the measured step, and nothing the
    model sees tells those steps apart. An encode in the first step only would fall before the reset and leave the
    measured peak without the tower. The warmup encodes cost load time only: their rows are freed with their step.
    """
    rows = None
    if (
        tower is not None
        and not has_images
        and attn_metadata.num_contexts > 0
        and getattr(attn_metadata.kv_cache_manager, "is_estimating_kv_cache", False)
    ):
        rows = tower.encode_largest_image()
        logger.info_once(
            f"Kimi K3.1: KV-capacity profiling counts the vision tower's activations of one "
            f"{tower.max_image_grid} image ({rows.shape[0]} rows)",
            key="kimi_k31_profiling_image_rows",
        )
    try:
        yield
    finally:
        del rows


_NO_LOCAL_TOWER = (
    "Image input is not supported: this server runs no vision tower (disable_mm_encoder, "
    "TLLM_MULTIMODAL_DISAGGREGATED, or a checkpoint without vision tower weights)."
)


def validate_image_request(tower: Optional[K31VisionTower], request: Any) -> None:
    """The admission check of one request's processed media (``request.py_multimodal_data``), before it is scheduled.

    A tower refuses grids it cannot encode (``K31VisionTower.validate_multimodal_data``). Without a local tower
    (``tower`` None) a request with media is refused, unless it is generation-only: the decode side of P/D
    disaggregation receives the context's KV and never encodes, so it needs no tower. A refusal here fails only this
    request; the same condition in the forward would fail the whole batch.

    Raises:
        ValueError: If the request carries media this server would have to encode and cannot.
    """
    multimodal_data = getattr(request, "py_multimodal_data", None)
    if not multimodal_data:
        return
    if tower is not None:
        tower.validate_multimodal_data(multimodal_data)
    elif not getattr(request, "is_generation_only_request", False):
        raise ValueError(_NO_LOCAL_TOWER)


def image_embeddings(
    tower: Optional[K31VisionTower],
    multimodal_params: Sequence[MultimodalParams],
    num_contexts: int,
) -> List[torch.Tensor]:
    """The image rows this step's context chunks cover, in prompt order: the tower runs once per request (the first
    step that needs it) and its rows stay on the request's multimodal data for the later chunks.

    Raises:
        NotImplementedError: If an image reaches a server that runs no local vision tower, which admission
            (``validate_image_request``) refuses first.
    """
    if not multimodal_params:
        return []
    if tower is None:
        raise NotImplementedError(_NO_LOCAL_TOWER)
    context_params = list(multimodal_params[:num_contexts])
    embeds = get_multimodal_embeddings(
        encoder_forward_fn=tower.forward, multimodal_params=context_params
    )
    return find_input_mm_embeds(embeds, context_params)


def splice_image_rows(
    embed_tokens: nn.Module,
    input_ids: torch.Tensor,
    rows: Sequence[torch.Tensor],
    mm_token_indices: Optional[torch.Tensor],
) -> torch.Tensor:
    """The step's prompt embeddings with ``rows`` written over its media placeholder positions ``mm_token_indices``
    (which the executor computes on the host from the model's ``multimodal_token_ids``); every other position keeps
    its token embedding. The placeholder id is in the vocabulary, so the whole step is embedded first."""
    image_rows = rows[0] if len(rows) == 1 else torch.cat(list(rows))
    if mm_token_indices is None:
        raise ValueError("an image step arrived without the executor's mm_token_indices")
    if mm_token_indices.shape[0] != image_rows.shape[0]:
        raise ValueError(
            f"Multimodal token count mismatch: found {mm_token_indices.shape[0]} image tokens in input_ids "
            f"but received {image_rows.shape[0]} image embeddings."
        )
    embeds = embed_tokens(input_ids)
    embeds.index_copy_(
        0,
        mm_token_indices.to(device=embeds.device, dtype=torch.long, non_blocking=True),
        image_rows.to(dtype=embeds.dtype, device=embeds.device),
    )
    return embeds


# ====================================================================================================================
# Target
# ====================================================================================================================


# ---------------------------------------------------------------------------
# Full VLM Model
# ---------------------------------------------------------------------------


class K31VLWrapper(PreTrainedModel):
    """Kimi K3.1 vision-language model: the MoonViT tower (``K31VisionTower``) over the K3.1 text model.

    The text backbone is ``self.llm``. The wrapper states what the engine reads off the outer model for it
    (``draft_config``, ``draft_model``, ``spec_worker``, ``model``, ``lm_head``, ``vocab_size_padded``,
    ``load_draft_weights``, ``set_guided_decoder``, ``infer_max_seq_len``) and the KV-manager preferences, the
    multimodal surface (``multimodal_token_ids``, ``multimodal_data_device_paths`` and the request check
    ``validate_request``), the forward (the step's image rows spliced over the media placeholders, the prompt ids
    kept for the speculative drafter) and the weight loader of ``weights.py``, which builds the tower.
    Images are encoded inline, in the first context step that covers them; there is no local tower under
    ``disable_mm_encoder`` or encoder disaggregation.
    """

    _LANG_PREFIX = "language_model."
    mamba_metadata_cls = K31ForCausalLM.mamba_metadata_cls

    @classmethod
    def get_preferred_kv_cache_manager_version(cls, pretrained_config: Any = None) -> Literal["V2"]:
        return "V2"

    @classmethod
    def requires_kv_cache_manager_v2(cls, pretrained_config=None) -> bool:
        """Follow the text backbone: Kimi K3.1 DSA requires KV cache manager V2."""
        return K31ForCausalLM.requires_kv_cache_manager_v2(pretrained_config)

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        *args,
        **kwargs,
    ) -> None:
        config = model_config.pretrained_config
        # PreTrainedModel.__init__ resolves config._attn_implementation and
        # rejects the default "sdpa" unless the class declares support. This
        # wrapper never runs HF attention itself, so declare support to keep
        # HF init happy.
        self._supports_sdpa = True
        PreTrainedModel.__init__(self, config)

        # Re-entry guard: the tail of this __init__ repoints
        # model_config.pretrained_config at the text_config and remaps the
        # quant exclude list, so running the body twice on the same instance
        # would fail the text_config assert below. If the text backbone
        # already exists this is a re-init on a built instance: no-op.
        if hasattr(self, "llm"):
            return

        self.model_config = model_config
        self._vlm_pretrained_config = config

        # The vision tower is built and filled by the weight loader. Text-only
        # serving (disable_mm_encoder) and E/P disaggregation both run without
        # a local vision tower.
        self.mm_encoder: Optional[K31VisionTower] = None
        self._skip_mm_encoder = multimodal_disaggregated() or model_config.disable_mm_encoder

        text_model_config = copy.copy(model_config)
        assert hasattr(config, "text_config"), "Kimi K3.1 config must have text_config"
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

        self.llm = K31ForCausalLM(text_model_config)

        self._media_placeholder_token_id = getattr(
            config, "media_placeholder_token_id", MEDIA_PLACEHOLDER_TOKEN_ID
        )
        self._mm_token_ids = torch.tensor([self._media_placeholder_token_id], dtype=torch.int32)

        # Point model_config at the text_config so the executor reads generation
        # params (eos_token_id, ...) from the text backbone.
        self.config = self.llm.config
        model_config._frozen = False
        model_config.pretrained_config = self.llm.config
        model_config._frozen = True

    # What the spec-dec and weight-loading paths read off the outer model, from the text backbone.
    @property
    def draft_config(self):
        return self.llm.draft_config

    @property
    def draft_model(self):
        return self.llm.draft_model

    @property
    def spec_worker(self):
        return self.llm.spec_worker

    @property
    def model(self):
        return self.llm.model

    @property
    def lm_head(self):
        return self.llm.lm_head

    @property
    def vocab_size_padded(self) -> int:
        return self.llm.vocab_size_padded

    def set_guided_decoder(self, *args, **kwargs):
        return self.llm.set_guided_decoder(*args, **kwargs)

    def load_draft_weights(self, *args, **kwargs):
        return self.llm.load_draft_weights(*args, **kwargs)

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    @property
    def multimodal_data_device_paths(self) -> List[str]:
        """The processed media the executor moves to the GPU (all images under ``image``)."""
        return [
            "image.pixel_values",
            "image.image_grid_thw",
            "multimodal_embedding",
        ]

    @property
    def multimodal_token_ids(self) -> torch.Tensor:
        """The in-vocabulary media placeholder id, from which the executor finds a step's image positions."""
        return self._mm_token_ids

    @property
    def _local_tower(self) -> Optional[K31VisionTower]:
        """The vision tower this server encodes images with; None where it runs none."""
        return None if self._skip_mm_encoder else self.mm_encoder

    def validate_request(self, request) -> None:
        """Refuse, before the request is scheduled, media this server would have to encode and cannot
        (``validate_image_request``): grids MoonViT cannot encode, and media without a local vision tower on a request
        that runs its context here."""
        validate_image_request(self._local_tower, request)

    def load_weights(self, weights, weight_mapper=None, **kwargs) -> None:
        """The explicit key split of ``weights.py``: text model, vision tower, unknown keys fail.

        ``weights`` are the engine's lazy safetensors slices (the ``kimi_k31`` checkpoint stays mapped), with
        ``checkpoint_dir`` for the expert loader; the checkpoint's names are the modules' own, so no weight
        mapper applies."""
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
        tower = self._local_tower
        image_rows = image_embeddings(tower, multimodal_params or [], attn_metadata.num_contexts)
        raw_input_ids = input_ids
        if image_rows:
            inputs_embeds = splice_image_rows(
                self.llm.model.embed_tokens, input_ids, image_rows, kwargs.get("mm_token_indices")
            )
            input_ids = None
        else:
            inputs_embeds = None

        # With image rows input_ids is None; the speculative drafter reads the
        # prompt token IDs from orig_input_ids instead. A KV-capacity profiling
        # step also holds one largest image's encoding (profiling_image_rows).
        with profiling_image_rows(tower, attn_metadata, bool(image_rows)):
            return self.llm.forward(
                attn_metadata,
                input_ids,
                position_ids,
                inputs_embeds,
                return_context_logits,
                orig_input_ids=raw_input_ids,
                **kwargs,
            )


class K31TepHybridCacheManager(KimiK3DSAHybridCacheManager):
    """The family's hybrid KDA + MLA + DSA cache manager with sharded cold pages (``ops/common/shard_cold_page_codec``).

    Without a cold-page codec provider from the engine (``kv_cache_compression_config``), the manager takes the sharded
    one, which is on with block reuse and attention pools that attention TP replicates: each rank then keeps only its
    slice of an evicted MLA latent or index-key page in the host and disk tiers.
    """

    def __init__(self, *args, cold_page_codec_provider=None, **kwargs) -> None:
        self._shard_cold_page_provider = None
        if cold_page_codec_provider is None:
            from .ops.common.shard_cold_page_codec import create_shard_cold_page_provider

            # The engine passes kv_cache_config by position (MambaHybridCacheManagerV2's signature).
            positional = list(inspect.signature(MambaHybridCacheManagerV2.__init__).parameters)[1:]
            kv_cache_config = dict(zip(positional, args)).get(
                "kv_cache_config", kwargs.get("kv_cache_config")
            )
            cold_page_codec_provider = create_shard_cold_page_provider(
                kwargs["mapping"], kv_cache_config
            )
            self._shard_cold_page_provider = cold_page_codec_provider
        super().__init__(*args, cold_page_codec_provider=cold_page_codec_provider, **kwargs)

    def shutdown(self):
        try:
            super().shutdown()
        finally:
            if self._shard_cold_page_provider is not None:
                self._shard_cold_page_provider.close()


class KimiK31ForConditionalGeneration(K31VLWrapper):
    """Kimi K3.1 vision-language model: the K3 wrapper over the package's text backbone, which
    reads the K3.1 ``dsa_config`` (a DSA indexer on every MLA layer) from its own config.

    Besides the model it states what the engine asks a model class for before and while it is
    built: model defaults, KV cache manager / transceiver preferences, the hybrid KDA + MLA + DSA
    cache manager with sharded cold pages (``runtime_hooks.kv_cache_manager_cls``,
    ``K31TepHybridCacheManager``), the DSA attention backend (``runtime_hooks.attention_backend_cls``) and the
    engine hooks (``runtime_hooks``, ``runtime.py``).
    """

    runtime_hooks = replace(
        K31_RUNTIME_HOOKS,
        kv_cache_manager_cls=K31TepHybridCacheManager,
        attention_backend_cls=K31PerfDSABackend,
    )

    @classmethod
    def get_model_defaults(cls, llm_args) -> dict:
        return K31ForCausalLM.get_model_defaults(llm_args)

    @classmethod
    def get_preferred_transceiver_runtime(cls, pretrained_config=None):
        return K31ForCausalLM.get_preferred_transceiver_runtime(pretrained_config)


class PrefillTarget:
    """Steps with context rows: multimodal fusion, then the text model."""

    def __init__(self, model: "ModelingV2KimiK31ProMxfp4Sm103Tep"):
        self.model = model
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks for it; read once here rather than per forward.
        self._contract_pending = step_contract_enabled()

    def _check_step_contract(self, attn_metadata: AttentionMetadata) -> None:
        """Opt-in first-forward fail-fast: the engine metadata names the target reads exist
        (``ops/common/step_contract.py``). Fixed at engine construction, so once per path, and off in a
        served engine."""
        if not self._contract_pending:
            return
        check_step_contract(attn_metadata)
        self._contract_pending = False

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor],
        position_ids: Optional[torch.IntTensor],
        inputs_embeds: Optional[torch.FloatTensor],
        return_context_logits: bool,
        multimodal_params: Optional[List[MultimodalParams]],
        **kwargs,
    ) -> torch.Tensor:
        self._check_step_contract(attn_metadata)
        return KimiK31ForConditionalGeneration.forward(
            self.model,
            attn_metadata,
            input_ids,
            position_ids,
            inputs_embeds,
            return_context_logits,
            multimodal_params,
            **kwargs,
        )


class DecodeTarget:
    """Generation-only steps: the token ids go straight to the text model (no image rows)."""

    def __init__(self, model: "ModelingV2KimiK31ProMxfp4Sm103Tep"):
        self.model = model
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks for it; read once here rather than per forward.
        self._contract_pending = step_contract_enabled()

    def _check_step_contract(self, attn_metadata: AttentionMetadata) -> None:
        """Opt-in first-forward fail-fast: the engine metadata names the target reads exist
        (``ops/common/step_contract.py``). Fixed at engine construction, so once per path, and off in a
        served engine."""
        if not self._contract_pending:
            return
        check_step_contract(attn_metadata)
        self._contract_pending = False

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor],
        position_ids: Optional[torch.IntTensor],
        inputs_embeds: Optional[torch.FloatTensor],
        return_context_logits: bool,
        multimodal_params: Optional[List[MultimodalParams]],
        **kwargs,
    ) -> torch.Tensor:
        self._check_step_contract(attn_metadata)
        return self.model.llm.forward(
            attn_metadata,
            input_ids,
            position_ids,
            inputs_embeds,
            return_context_logits,
            orig_input_ids=input_ids,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Host data stack: one interpreter data-stack chunk per forward.
# ---------------------------------------------------------------------------
#
# Run a forward on one large chunk of the interpreter's data stack.
#
# CPython (3.11 and later) keeps the frames of Python calls on a per-thread data stack made of chunks (16 KiB unless a
# frame needs more). A call whose frame does not fit in the current chunk maps a new chunk with ``mmap``, and that chunk
# is unmapped again as soon as the call returns. An eager forward of this target reaches the end of a chunk inside its
# CuTe DSL launch paths, so every launch maps and unmaps a chunk. In a disaggregated server UCX (loaded by the KV cache
# transceiver) hooks ``munmap`` and records each unmapped range in its registration caches; before the first transfer
# nothing drains those records, so every unmap gets slower and the CUDA-graph warmup's eager forwards spend most of
# their host time in that hook.
#
# ``run_on_large_datastack_chunk`` calls the forward from a trampoline frame with ``DATASTACK_KIB`` KiB of locals. That
# frame does not fit in the current chunk, so the interpreter maps one chunk of twice its size for it, and every frame
# of the forward below it lands in the same chunk: one map and one unmap per forward instead of one per launch.
# Host-side only; no kernel, launch or tensor changes.


#: KiB of locals of the trampoline frame (half of the chunk it maps).
DATASTACK_KIB = 512


def _make_trampoline(num_locals: int) -> Callable[..., Any]:
    """A function ``trampoline(fn, *args, **kwargs) -> fn(*args, **kwargs)`` whose frame holds ``num_locals`` locals
    (assigned only in a branch that never runs, so they cost the frame's size and nothing else)."""
    targets = " = ".join(f"_slot{i}" for i in range(num_locals))
    source = (
        "def run_on_large_datastack_chunk(fn, /, *args, **kwargs):\n"
        "    if _NEVER:\n"
        f"        {targets} = None\n"
        "    return fn(*args, **kwargs)\n"
    )
    namespace = {"_NEVER": False}
    exec(compile(source, "<kimi_k31 host_datastack>", "exec"), namespace)
    return namespace["run_on_large_datastack_chunk"]


_TRAMPOLINE = _make_trampoline(DATASTACK_KIB * 1024 // 8)


def run_on_large_datastack_chunk(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """``fn(*args, **kwargs)``, with every Python frame below it on one data-stack chunk (see the module docstring)."""
    if torch.compiler.is_compiling():
        return fn(*args, **kwargs)
    return _TRAMPOLINE(fn, *args, **kwargs)


class ModelingV2KimiK31ProMxfp4Sm103Tep(KimiK31ForConditionalGeneration):
    """Kimi K3.1 Pro MXFP4, sm_103, attention TP perf layouts (TEP8, TEP16 MoE TP4 x EP4, TEP32 MoE TP4 x EP8)."""

    def __init__(self, model_config, *args, **kwargs) -> None:
        super().__init__(model_config, *args, **kwargs)
        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)

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
        # One interpreter data-stack chunk for the whole forward (no chunk map / unmap per launch; see above).
        target = self._prefill if attn_metadata.num_contexts > 0 else self._decode
        return run_on_large_datastack_chunk(
            target.forward,
            attn_metadata,
            input_ids,
            position_ids,
            inputs_embeds,
            return_context_logits,
            multimodal_params,
            **kwargs,
        )
