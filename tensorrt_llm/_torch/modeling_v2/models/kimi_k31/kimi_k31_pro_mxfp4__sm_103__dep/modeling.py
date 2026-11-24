# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: Kimi K3.1 Pro (MXFP4) / sm_103 / dep (the attention-DP perf layout: DEP16, experts EP16).

Routing (``routing.py``, ``_SM103_DEP_LAYOUTS``) sends SM 10.3 deployments at this layout with an FP8 latent KV cache
here, and admits a static expert placement from the engine's EP load balancer configuration
(``moe_config.load_balancer`` with one slot per expert and ``layer_updates_per_iter`` 0); every other admitted
deployment runs the general target (``kimi_k31_pro_mxfp4__sm_10x__default``). The model is the general target's model
plus the performance paths below. Where none of them serves a step, the step runs the general target's operations (that
target's module docstring lists them).

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
token ids go straight to the text model. Under attention DP each rank runs whole attention layers on its own requests;
the routed experts are dispatched over the 16 ranks. The sparse MLA kernel has no tile for a group of 96 query heads,
so the MLA pads them to 128 (`padded_attention_heads`: heads 96..127 of the query are zero and their outputs are
dropped before the V absorption; the KV cache, the index keys and the weights are unchanged).

**What this target runs beyond the general target** (target ops under `ops/<category>/`, their kernels under
`kernels/<category>/`):
* attention residuals: the family PDL mixes (`ops/norm/attn_res_rmsnorm`, `attn_res_add_rmsnorm`), the engine's
  persistent mix, and the mix with the previous layer's MoE residual update folded in
  (`ops/norm/attn_res_moe_add_rmsnorm`); the drafter's capture layers keep `ops/norm/attn_res_fwd` and a separate
  RMSNorm;
* the decode and verify BF16 projections on the attention-DP low-M GEMM plan, held to the general target's bits
  (`ops/gemm/low_m_plans`, `low_m_linear`);
* MLA: the token-wise projections of the layer input on one packed weight (`ops/attention/dsa_packed_proj`) and the
  fused indexer prologue (one launch quantizes the indexer Q / K, scales the head weights and writes the index-key
  cache);
* the DSA indexer prefill of long-prefix chunks shared across the attention-DP ranks
  (`ops/attention/dsa_prefill_share`), on a server that runs context chunks: a disaggregated generation server
  (``TRTLLM_DISAGG_ROLE=generation``, as the serve launcher exports it) skips it;
* KDA: the [q | k | v | g] projection split over pairs of ranks of a tray, each rank keeping one head half of the
  weight, with the pair's rows exchanged in one persistent kernel (`ops/comm/group_exchange`, see
  `_k31_kda_pair_group`);
* the MoE: up to the fused path's rows, dispatch, experts and combine as one op with the combine in the FC2 epilogue
  (`ops/moe/w4a8_ep_moe_fused`); above them the MoE prologue's routing and MXFP8 input (`ops/moe/moe_prologue`), the
  one-sided all-to-all (`ops/moe/ep_dispatch_combine`) and the CuTe DSL W4A8 grouped GEMMs or the TRTLLM-Gen
  W4A8_MXFP4_MXFP8 runner; the layer's end `prefix + (routed + shared)` in one kernel (`ops/moe/moe_residual_add`);
  a static expert placement from the load balancer configuration (`ops/moe/expert_placement`), and the routing counts
  that configuration is generated from (`ops/moe/routing_statistic`);
* the LM head vocab-parallel over the attention-DP group (`ops/comm/adp_vocab_parallel_head`) instead of the whole
  head on every rank;
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
transceiver for disaggregated serving, DFlash2 or no speculative decoding), sm_103, attention DP over 16 ranks with
EP16 experts, and a load balancer only as the static placement above. Construction checks the MoE backend the engine
resolved and, with a static placement, that every routed call can take the fused routed MoE (pure EP over the TP
ranks, a token budget per rank it serves in one call).

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

**Out of scope:** attention TP (the tep target), sm_107 and sm_100, the `nvfp4_ds_mla` KV cache and FP4 index keys,
pipeline and context parallelism, a dynamic EP load balancer, MoE backends other than TRTLLM, and speculative decoding
other than DFlash2."""

from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import weakref
from collections import deque
from dataclasses import replace
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
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_attention import (
    K31MLAAttention,
    K31PerfDSABackend,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_graph_ops import (
    DeferredQB,
    forward_sparse_attn,
    forward_sparse_attn_custom_op,
    initialize_sparse_attn,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_packed_proj import (
    packed_output_gate,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dsa_params import (
    K31DSAParams,
    k31_dsa_params,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.mla_outputs import (
    create_mla_outputs,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.comm.allgather_module import (
    allgather_module,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.comm.allreduce_module import (
    FUSION_NONE,
    allreduce_module,
    tp_allreduce,
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
    low_m_choice,
    plan_choice,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.gemm.low_m_plans import k31_low_m_plan
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.ep_dispatch_combine import (
    K31EpAllToAll,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.moe_prologue import (
    quantize_input,
    separated_routing,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.moe_residual_add import (
    k31_moe_residual_add,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.routed_experts import (
    K31RoutedExperts,
    num_chunks,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.routing_statistic import (
    record_expert_statistic,
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
    dflash2_sampled_step,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_selector_greedy_path import (
    K31DFlash2CandidateSelector,
    k31_dflash2_selector_greedy_path,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_selector_greedy_tokens import (
    k31_dflash2_selector_greedy_tokens,
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
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.fused_compute_probs_from_logits import (
    fused_compute_probs_from_logits,
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
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.kda_graph_ops import (
    KDA_LAYERS,
    maybe_bcg_kda_core_inplace,
    register_kda_layer,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.kda_metadata import KdaMetadata
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.kda_mtp_verify import kda_mtp_verify
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.kda_prefill import kda_prefill
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
from tensorrt_llm._torch.nccl_window_tensor_scope import nccl_window_tensor_scoped_module
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import is_in_breakable_cuda_graph
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManager
from tensorrt_llm._torch.pyexecutor.resource_manager import BaseResourceManager
from tensorrt_llm._torch.pyexecutor.sampler.ops.flashinfer import resolve_advanced_sampling_filters
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import GREEDY_TEMPERATURE_THRESHOLD
from tensorrt_llm._torch.route_capture import get_active_route_capture
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
from tensorrt_llm._torch.utils import AuxStreamType
from tensorrt_llm._utils import get_sm_version, maybe_pin_memory, prefer_pinned
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.llmapi.llm_args import is_dflash2_checkpoint
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo

from ..input_processor import MEDIA_PLACEHOLDER_TOKEN_ID, moonvit_max_image_grid
from . import weights as target_weights
from .ops.comm.adp_vocab_parallel_head import DEFAULT_MAX_ROWS as _ADP_EXCHANGE_ROWS
from .ops.comm.adp_vocab_parallel_head import (
    build_top32_kernels,
    exchange_supported,
    k31_adp_vocab_logits,
    k31_adp_vocab_top32,
    reserve_exchange,
)
from .ops.comm.group_exchange import group_workspace, lamport_group_exchange
from .ops.moe.expert_placement import expert_slots, layer_slot_experts
from .ops.moe.w4a8_ep_moe_fused import NUM_EXPERTS as _FUSED_MOE_NUM_EXPERTS
from .ops.moe.w4a8_ep_moe_fused import TOP_K as _FUSED_MOE_TOP_K
from .ops.moe.w4a8_ep_moe_fused import (
    fused_moe_decline,
    fused_moe_max_tokens,
    fused_moe_workspace,
    k31_w4a8_ep_moe_fused,
    publish_real_rows,
)
from .ops.norm.attn_res_moe_add_rmsnorm import k31_attn_res_moe_add_rmsnorm
from .weights import PROJECTOR_PREFIX, VISION_TOWER_PREFIX, expert_ckpt_spec

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig


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
# layout builds: it defines the [q | k | v | g] projections (``_prefill_qkvg``, ``_decode_qkvg``,
# ``_project_verify_inputs``, by the rank pair) and overrides ``_project_output`` and the verify kernel's arguments
# (``VERIFY_KERNEL_KWARGS``); every rank keeps every KDA head (attention DP).


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

    #: Extra keyword arguments of the target's ``kda_mtp_verify`` call.
    VERIFY_KERNEL_KWARGS: dict = {}

    def __init__(
        self,
        cfg,
        layer_idx: int,
        mapping=None,
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

        self.num_heads = lin["num_heads"]
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
        """o_proj of the post-norm core."""
        return bf16_linear(core.reshape(-1, self.proj_size), self.o_proj.weight, False)

    def _prefill_kernel(self, *args) -> torch.Tensor:
        """The chunked delta-rule prefill (``kda_prefill``'s arguments)."""
        return kda_prefill(*args)

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

    def forward_verify(
        self,
        x2d,
        num_steps,
        layer_cache,
        ssm_pool,
        slot_indices,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """DFlash2 verify: ``kda_mtp_verify`` (the latency-scheduled verify kernel) on packed rows, the gated
        output norm fused in its epilogue.

        The kernel replays the previous round's accepted drafts from the manager's replay caches, advances each
        request ``num_steps`` tokens (the golden token and the padded drafts), commits the state after the golden
        token and caches the new drafts for the next round's replay.
        """
        _, x, T_total = self._verify_rows(x2d, num_steps, layer_cache)
        projections = self._project_verify_inputs(x, T_total)
        out = kda_mtp_verify(
            **self._verify_kwargs(
                projections, T_total, num_steps, layer_cache, ssm_pool, slot_indices
            ),
            **self.VERIFY_KERNEL_KWARGS,
        )
        return self._store_core(out, output)


# ---------------------------------------------------------------------------
# K31KDALinearAttention: the KDA layer on this layout's kernels.
# ---------------------------------------------------------------------------


def _k31_decode_plan(mapping) -> Optional[dict]:
    """The measured low-M GEMM plan of this generation layout (``ops/gemm/low_m_plans``); None off SM 10.3."""
    if mapping is None or get_sm_version() != 103:
        return None
    return k31_low_m_plan(mapping.tp_size, mapping.enable_attention_dp)


def _k31_planned_linear(x: torch.Tensor, weight: torch.Tensor, plan) -> Optional[torch.Tensor]:
    """``x @ weight.T`` on the layout's low-M plan where it names the shape and row count, else None."""
    if plan is None or low_m_choice(plan, x, weight) is None:
        return None
    return k31_low_m_linear(x, weight, plan, True)


# Verify projections, [q | k | v | g] GEMM (main stream) beside the gate pair [f_a | b] -> f_b (aux stream), by the
# verify's row count (GB300 layer chain, cold weights, CUDA graph, us per layer, 96 heads). At 65-120 rows the gate
# pair is issued before the GEMM: issued after it, the pair waits for SMs the GEMM's CTAs hold (-2 to -3); elsewhere
# the GEMM goes first. The two run on two streams at every row count (above 128 rows: -4 to -8 against running them
# one after the other).
_K31_GATES_FIRST_ROWS = ((65, 120),)


def _k31_gates_first(num_rows: int) -> bool:
    """Whether the verify's gate pair is issued before its [q | k | v | g] GEMM (``_K31_GATES_FIRST_ROWS``)."""
    return any(
        lo <= num_rows and (hi is None or num_rows <= hi) for lo, hi in _K31_GATES_FIRST_ROWS
    )


def _k31_verify_projection_pair(
    qkvg_fn, gates_fn, fork_event, join_event, aux_stream, num_rows: int
):
    """``(qkvg_fn(), gates_fn())``: the GEMM on the current stream and the gate pair on ``aux_stream``, issued in the
    measured order for ``num_rows`` (``maybe_execute_in_parallel`` issues the GEMM first); one stream without
    ``aux_stream``, outside multi-stream execution or under torch.compile."""
    if aux_stream is None or num_rows <= 0:
        return qkvg_fn(), gates_fn()
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


# The KDA [q | k | v | g] projection, sharded over pairs of attention-DP ranks of a tray. Each rank of a pair keeps one
# head half of each section ([q_r | k_r | v_r | g_r]: 352 of the 704 MB BF16 weight per layer, 24.3 GB per rank over
# the 69 layers). Per projection the pair all-gathers its rows, each rank multiplies both ranks' rows by its half, and
# an all-to-all hands every row's halves back to its owner (``kimi_k31_pro_mxfp4__sm_103__dep/ops/comm/group_exchange``,
# one kernel each). The GEMM reads half the weight; the halves' columns are bitwise the full GEMM's
# (``_k31_kda_pair_halves``). GB300, CUDA graph, KDA layer chain at DEP16 rows per rank (40 / 80 / 100 / 120), us per
# layer: 181 -> 143, 218 -> 181, 234 -> 201, 248 -> 219. o_proj stays per rank: sharding it too needs two more exchanges
# (~5 us of latency each), which cost more than the half GEMM saves at these rows.
_K31_KDA_PAIR = 2
# Context parts of at most this many rows per rank run their half GEMM through the full-width view of the half
# (``_k31_kda_pair_halves``): cuBLAS splits K by the output width at low row counts (GB300: 4 ways for the full width
# and 2 for the half at 5-8 rows), and only the full width gives the per-rank GEMM's bits.
_K31_KDA_PAIR_WIDE_ROWS = 64


def _k31_kda_pair_group(mapping) -> Mapping:
    """The groups (consecutive attention-DP ranks of a tray, two per group) the KDA [q | k | v | g] projection is
    sharded over, as the TP groups of a mapping."""
    assert (
        mapping.enable_attention_dp
        and mapping.tp_size % _K31_KDA_PAIR == 0
        and mapping.pp_size == 1
        and mapping.cp_size == 1
    ), "the KDA pair groups need attention DP over whole pairs of a tray, without PP or CP"
    return Mapping(
        world_size=mapping.world_size,
        rank=mapping.rank,
        gpus_per_node=mapping.gpus_per_node,
        tp_size=_K31_KDA_PAIR,
        pp_size=mapping.world_size // _K31_KDA_PAIR,
    )


class _K31KdaPair:
    """A pair's shared state for the sharded KDA projection: the pair's Lamport workspace, the layers' weight halves
    and the step's row counts (``((context rows, generation rows) of pair rank 0, ... of pair rank 1)``), exchanged by
    the first KDA layer of every eager step."""

    def __init__(self, group: Mapping):
        self.ws = group_workspace(group)
        self.rank = group.tp_rank
        self.first_layer: Optional[int] = None
        self.counts: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
        self.pool: Optional[torch.Tensor] = None
        self.slots = 0
        logger.info(
            f"Kimi K3.1 pair-sharded KDA [q | k | v | g] projection: rank {group.rank} is pair rank {self.rank}, "
            f"Lamport buffer {self.ws.bytes_per_buffer} B"
        )

    def half_weight(self, rows: int, like: torch.Tensor, num_layers: int):
        """A layer's half-weight buffer ``[rows, hidden]`` and its full-width view ``[2 * rows, hidden]`` (the half
        followed by the next ``rows`` rows of one pool of ``num_layers + 1`` halves; the last one zeros)."""
        if self.pool is None or self.slots >= self.pool.shape[0] // rows - 1:
            self.pool = torch.empty(
                ((num_layers + 1) * rows, like.shape[1]), dtype=like.dtype, device=like.device
            )
            self.pool[num_layers * rows :].zero_()
            self.slots = 0
        i = self.slots
        self.slots += 1
        return self.pool[i * rows : (i + 1) * rows], self.pool[i * rows : (i + 2) * rows]

    def step_counts(self, attn_metadata: AttentionMetadata, layer_idx: int):
        """Both pair ranks' (context rows, generation rows) of this step.

        A captured step is generation only with every rank of the pair on the same batch (attention DP runs a decode
        graph only when every rank's batch size is the same), so both are this rank's. An eager step's come from one
        exchange (and one host read) in its first KDA layer.
        """
        num_ctx = attn_metadata.num_ctx_tokens
        own = (num_ctx, attn_metadata.num_tokens - num_ctx)
        if torch.cuda.is_current_stream_capturing():
            return own, own
        if layer_idx == self.first_layer or self.counts is None:
            mine = torch.tensor([[own[0], own[1], 0, 0]], dtype=torch.int32, device="cuda")
            both = (
                lamport_group_exchange(mine, self.ws, 1, True)
                .view(_K31_KDA_PAIR, 4)[:, :2]
                .tolist()
            )
            self.counts = (tuple(both[0]), tuple(both[1]))
        assert self.counts[self.rank] == own, (self.counts, own)
        return self.counts

    def project(
        self, x: torch.Tensor, rows: Tuple[int, int], halves_of, kinds, context: bool
    ) -> torch.Tensor:
        """``x`` [n, hidden] (this rank's rows of a part) times the full [q | k | v | g] weight: [n, 4 * proj].
        ``halves_of = (half, wide)``: this rank's half weight and its full-width view."""
        n, m = x.shape[0], max(rows)
        if n < m:
            x = torch.cat([x, x.new_zeros((m - n, x.shape[1]))])
        gathered = lamport_group_exchange(x.contiguous(), self.ws, 1, True).view(
            _K31_KDA_PAIR * m, -1
        )
        halves = _k31_kda_pair_halves(gathered, halves_of[0], halves_of[1], kinds, rows, context)
        return lamport_group_exchange(halves, self.ws, 4, False)[:n]


def _k31_kda_pair_halves(
    gathered: torch.Tensor,
    half: torch.Tensor,
    wide: torch.Tensor,
    kinds,
    rows: Tuple[int, int],
    context: bool,
) -> torch.Tensor:
    """``gathered`` [pair * m, hidden] (pair rank r's row t at row ``t * pair + r``) times ``half``^T, every rank's
    rows bitwise as that rank's full GEMM computes them.

    A generation part runs rank r's rows on the one-row plan ``kinds[r]`` (``_k31_kda_pair_kinds``): the plan's
    kernels and cuBLAS above the plan's rows sum every output element in an order independent of the row count and
    width, so one GEMM over both ranks' rows serves when the kinds agree. A context part runs every rank's rows alone
    at its own row count (cuBLAS's kernel choice follows the row count), through the full-width view up to
    ``_K31_KDA_PAIR_WIDE_ROWS`` rows (where it also follows the width)."""
    if not context and kinds[0] == kinds[1]:
        return k31_low_m_linear(gathered, half, kinds[0], True)
    m = gathered.shape[0] // _K31_KDA_PAIR
    out = gathered.new_empty((gathered.shape[0], half.shape[0]))
    by_rank, out_by_rank = gathered.view(m, _K31_KDA_PAIR, -1), out.view(m, _K31_KDA_PAIR, -1)
    for r in range(_K31_KDA_PAIR):
        n = rows[r]
        if n == 0:
            continue
        x = by_rank[:n, r].contiguous()
        if context and n <= _K31_KDA_PAIR_WIDE_ROWS:
            out_by_rank[:n, r] = k31_low_m_linear(x, wide, None, True)[:, : half.shape[0]]
        else:
            out_by_rank[:n, r] = k31_low_m_linear(x, half, kinds[r], True)
    return out


def _k31_kda_pair_kinds(plan, rows: Tuple[int, int], n_full: int, k: int, planned: bool):
    """Per pair rank, the one-row plan its rows' half GEMM runs on: the kernel the full GEMM would run for that rank's
    row count (the plan's choice for a generation part, cuBLAS for a context part, as the per-rank path does)."""

    def kind(n: int):
        choice = plan_choice(plan, n, n_full, k) if planned else None
        return None if choice is None else {(n_full // _K31_KDA_PAIR, k): ((1 << 30, choice),)}

    kinds = [kind(n) if n else None for n in rows]
    for r in range(_K31_KDA_PAIR):
        if rows[r] == 0:
            kinds[r] = kinds[1 - r]
    return tuple(kinds)


class K31KDALinearAttention(K31KdaLayer):
    """The KDA layer (``K31KdaLayer``) on the attention-DP layout's kernels.

    Each rank of a pair keeps one head half of the [q | k | v | g] weight (``_k31_kda_pair_group``) and the pair
    projects every rank's rows (``_K31KdaPair``); the decode / verify projections and o_proj run on the layout's low-M
    plan where it plans them; the two-CTA verify kernel loads the recurrent state evict-last.
    """

    # The two-CTA kernel loads the state evict-last: its lines then outlive the weight streams around the verify
    # (GB300 DEP16 layer chain, 65-160 rows: -2.4 to -3.3 us per layer).
    VERIFY_KERNEL_KWARGS = {"state_evict_last": True}

    def __init__(
        self,
        cfg,
        layer_idx: int,
        mapping=None,
        aux_stream: Optional[torch.cuda.Stream] = None,
        extra_attrs: Optional[dict] = None,
    ) -> None:
        # Attention DP only: the planned o_proj below runs without the TP all-reduce the family layer builds under
        # attention TP.
        assert mapping is None or mapping.enable_attention_dp, (
            "the Kimi K3.1 DEP target's KDA layer runs attention DP"
        )
        super().__init__(
            cfg,
            layer_idx,
            mapping=mapping,
            aux_stream=aux_stream,
            extra_attrs=extra_attrs,
        )
        # Decode / verify [q | k | v | g] projection on the low-M plan of the layout.
        self._k31_low_m_plan = _k31_decode_plan(mapping)
        # Pair-sharded [q | k | v | g] projection, set up with the fused weights.
        self._k31_mapping = mapping
        self._k31_extra_attrs = extra_attrs if extra_attrs is not None else {}
        self._k31_kda_registry = self._k31_extra_attrs.get(KDA_LAYERS)
        self._k31_pair: Optional[_K31KdaPair] = None
        self._k31_pair_wide: Optional[torch.Tensor] = None
        self._k31_pair_rows = None

    def finalize_decode_weights(self) -> None:
        """The family layer's fused weights; then this rank keeps only its pair half of the [q | k | v | g] weight
        (``_k31_kda_pair_group``)."""
        super().finalize_decode_weights()
        group = _k31_kda_pair_group(self._k31_mapping)
        if self._qkvg_proj_weight is None:
            return
        w = self._qkvg_proj_weight
        d = self.proj_size
        half = d // _K31_KDA_PAIR
        r = group.tp_rank
        pair = self._k31_extra_attrs.get("kimi_k31_kda_pair")
        if pair is None:
            pair = self._k31_extra_attrs["kimi_k31_kda_pair"] = _K31KdaPair(group)
        if pair.first_layer is None or self.layer_idx < pair.first_layer:
            pair.first_layer = self.layer_idx
        # Every KDA layer's half lives in one pool, so each half has a full-width view (``_k31_kda_pair_halves``).
        num_layers = len(self._k31_kda_registry) if self._k31_kda_registry else 1
        shard, wide = pair.half_weight(4 * half, w, num_layers)
        with torch.no_grad():
            for s, module in enumerate((self.q_proj, self.k_proj, self.v_proj, self.g_proj)):
                shard[s * half : (s + 1) * half].copy_(w[s * d + r * half : s * d + (r + 1) * half])
                # The modules' weights are row views of the fused buffer; point them at the half so nothing keeps it.
                module.weight.data = shard[s * half : (s + 1) * half]
        self._qkvg_proj_weight = shard
        self._k31_pair_wide = wide
        self._k31_pair = pair

    def _k31_pair_qkvg(self, x2d: torch.Tensor, generation: bool) -> torch.Tensor:
        """This rank's [n, 4 * proj] [q | k | v | g] rows of the step's context (or generation) part, by the pair."""
        part = 1 if generation else 0
        rows = (self._k31_pair_rows[0][part], self._k31_pair_rows[1][part])
        kinds = _k31_kda_pair_kinds(
            self._k31_low_m_plan, rows, 4 * self.proj_size, x2d.shape[-1], generation
        )
        return self._k31_pair.project(
            x2d, rows, (self._qkvg_proj_weight, self._k31_pair_wide), kinds, not generation
        )

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The family layer's step; with the pair-sharded projection, every step runs the pair's context exchange,
        then its generation exchange, on both ranks of the pair (a rank without rows of a part still runs its
        exchange for the other rank)."""
        pair = self._k31_pair
        rows = pair.step_counts(attn_metadata, self.layer_idx)
        self._k31_pair_rows = rows
        own_ctx, own_gen = rows[pair.rank]
        empty = hidden_states[:0]
        try:
            if own_ctx == 0 and max(rows[0][0], rows[1][0]) > 0:
                self._k31_pair_qkvg(empty, generation=False)
            core = super()._forward_impl(hidden_states, attn_metadata, output)
            if own_gen == 0 and max(rows[0][1], rows[1][1]) > 0:
                self._k31_pair_qkvg(empty, generation=True)
        finally:
            self._k31_pair_rows = None
        return core

    def _prefill_qkvg(self, x: torch.Tensor, x2d: torch.Tensor):
        """``(packed [T, 4 * D] [q | k | v | g] rows, output gate [1, T, D] view)`` of a prefill, by the pair."""
        qkvg = self._k31_pair_qkvg(x2d, generation=False)
        d = self.proj_size
        return qkvg, qkvg.unsqueeze(0)[..., 3 * d : 4 * d]

    def _decode_qkvg(self, x2d: torch.Tensor) -> torch.Tensor:
        """The [B, 4 * D] [q | k | v | g] rows of a one-token decode, by the pair."""
        return self._k31_pair_qkvg(x2d, generation=True)

    def _project_verify_inputs(self, x: torch.Tensor, num_rows: int):
        """The verify's [q | k | v | g] GEMM (by the pair) and the gate pair on two streams, in the order measured for
        ``num_rows`` (``_k31_verify_projection_pair``)."""

        def project_qkvg():
            rows = x.reshape(-1, x.shape[-1])
            return self._k31_pair_qkvg(rows, generation=True).view(*x.shape[:-1], -1)

        qkvg, (beta, forget_gate) = _k31_verify_projection_pair(
            project_qkvg,
            lambda: self._project_gate_inputs(x),
            self._projection_fork_event,
            self._projection_join_event,
            self._projection_aux_stream,
            num_rows,
        )
        return self._split_verify_qkvg(qkvg, forget_gate, beta)

    def _project_output(self, core: torch.Tensor) -> torch.Tensor:
        """o_proj on the layout's low-M plan where it plans it, else the family layer's (attention DP: no
        all-reduce)."""
        x = core.reshape(-1, self.proj_size)
        if (
            self._k31_low_m_plan is not None
            and low_m_choice(self._k31_low_m_plan, x, self.o_proj.weight) is not None
        ):
            return k31_low_m_linear(x, self.o_proj.weight, self._k31_low_m_plan, True)
        return super()._project_output(core)


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


class K31MLA(nn.Module):
    """Kimi K3.1 MLA attention with DSA: the fused q_a / kv_a projection and its norms, the query up projection and
    absorption, the DSA indexer's top-2048 key selection and the sparse MLA attention (``mqa``, a
    ``K31MLAAttention``) over the FP8 latent cache, the V absorption and the output projection. Every MLA layer of
    K3.1 runs DSA; :class:`K31GatedMLA` adds K3.1's checkpoint layout and the output gate."""

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

        self.register_to_config = False
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
            self.register_to_config = True

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

        # Tensor parallelism (routing refuses context parallelism): the heads split over attention TP, or whole
        # heads on every rank under attention DP.
        self.mapping = config.mapping
        tp_size = self.mapping.tp_size
        pp_size = self.mapping.pp_size
        dp_size = 1
        if self.mapping.enable_attention_dp:
            dp_size = tp_size
            tp_size = 1

        mapping = Mapping(
            world_size=pp_size * dp_size * tp_size,
            tp_size=tp_size,
            pp_size=pp_size * dp_size,
            cp_config=self.mapping.cp_config,
            rank=self.mapping.rank,
            gpus_per_node=self.mapping.gpus_per_node,
            enable_attention_dp=self.mapping.enable_attention_dp,
        )

        assert self.num_heads % tp_size == 0
        self.num_heads_tp = self.num_heads // tp_size
        # Query heads of the absorbed MQA backend (padded, see padded_attention_heads).
        self.num_heads_attn = padded_attention_heads(self.num_heads_tp)

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
            world_size=pp_size * dp_size * tp_size,
            tp_size=tp_size,
            pp_size=pp_size * dp_size,
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
            self.num_heads_attn,
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
            # The PDL native-tile logits lose at the DEP16 decode rows (224 per rank).
            logits_pdl=False,
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

    def forward_impl(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        attn_output: list[torch.Tensor],
    ) -> None:
        """Run the DSA sparse implementation of the MLA module."""
        forward_sparse_attn(
            self,
            position_ids,
            hidden_states,
            attn_metadata,
            attn_output,
        )
        if (
            self.zero_padding_output_rows
            and attn_metadata is not None
            and attn_output[0].shape[0] > attn_metadata.num_tokens
        ):
            attn_output[0][attn_metadata.num_tokens :].zero_()

    def _new_fused_q(self, q: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """``[num_tokens, num_heads_attn, kv_lora_rank + qk_rope_head_dim]``; the padding heads are zero."""
        fused_q = torch.empty(
            [num_tokens, self.num_heads_attn, self.kv_lora_rank + self.qk_rope_head_dim],
            dtype=q.dtype,
            device=q.device,
        )
        if self.num_heads_attn != self.num_heads_tp:
            fused_q[:, self.num_heads_tp :].zero_()
        return fused_q

    def _padded_q_pe(self, q_pe: torch.Tensor) -> torch.Tensor:
        """``q_pe`` with zero rows for the padding heads (the backend reads num_heads_attn heads)."""
        if self.num_heads_attn == self.num_heads_tp:
            return q_pe
        padded = q_pe.new_zeros([q_pe.shape[0], self.num_heads_attn, self.qk_rope_head_dim])
        padded[:, : self.num_heads_tp].copy_(q_pe)
        return padded

    def _attention_heads_out(self, attn_out_latent: torch.Tensor) -> torch.Tensor:
        """``[seq, num_heads_tp, kv_lora_rank]`` view of the backend output (padding heads dropped)."""
        assert attn_out_latent.shape[1] == self.num_heads_attn * self.kv_lora_rank
        out = attn_out_latent.view([-1, self.num_heads_attn, self.kv_lora_rank])
        return out[:, : self.num_heads_tp]

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
            self.num_heads_attn,
            (self.kv_lora_rank + self.qk_rope_head_dim),
            dtype=torch.uint8,
            device=q.device,
        )
        if self.num_heads_attn != self.num_heads_tp:
            quant_q_buffer[:, self.num_heads_tp :].zero_()
        q_pe_attn = self._padded_q_pe(q_pe)

        if hasattr(self, "k_b_proj_trans"):
            fused_q = self._new_fused_q(q, num_tokens)

            def _mla_gen_rope():
                # RoPE, latent-cache append and the trtllm-gen scheduler buffers are all produced in-kernel.
                self._rope_generation(
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

            rope_stream = None
            if self.k_b_proj_trans.dtype == torch.bfloat16:
                # [num_heads, num_tokens, self.qk_nope_head_dim]
                q_nope_t = q_nope.transpose(0, 1)
                # [num_heads, num_tokens, self.kv_lora_rank]
                q_nope_out = fused_q[:, : self.num_heads_tp, : self.kv_lora_rank].transpose(0, 1)

                # [num_heads, num_tokens, self.qk_nope_head_dim] x [num_heads, kv_lora_rank, qk_nope_head_dim]
                # -> [num_heads, num_tokens, kv_lora_rank] -> [num_tokens, num_heads, kv_lora_rank]
                # The output of bmm is written directly into fused_q
                maybe_execute_in_parallel(
                    lambda: self._bmm_bf16_out(
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
            else:
                raise NotImplementedError(
                    f"Missing bmm impl for dtype: {self.k_b_proj_trans.dtype}."
                )

            fused_q = fused_q.view(
                [
                    num_tokens,
                    self.num_heads_attn * (self.kv_lora_rank + self.qk_rope_head_dim),
                ]
            )
        else:
            raise RuntimeError("MLA absorption requires k_b_proj_trans")

        attn_out_latent = self.mqa.forward(
            fused_q,
            attn_metadata,
            attention_input_type=AttentionInputType.generation_only,
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
        num_tokens = q.shape[0]

        q_nope, q_pe = q.view([-1, self.num_heads_tp, self.qk_head_dim]).split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        if not hasattr(self, "k_b_proj_trans"):
            raise RuntimeError("MLA absorption requires k_b_proj_trans")

        # fused_q contains 1) the result of the following bmm with shape [num_tokens, num_heads, kv_lora_rank]
        # 2) rope(q_pe) with shape [num_tokens, num_heads, qk_rope_head_dim]. rope is applied inside AttentionOp
        fused_q = self._new_fused_q(q, num_tokens)
        q_pe_attn = self._padded_q_pe(q_pe)

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
                self.num_heads_attn * (self.kv_lora_rank + self.qk_rope_head_dim),
            ]
        )

        attn_out_latent = self.mqa.forward(
            fused_q,
            attn_metadata,
            attention_input_type=AttentionInputType.context_only,
            latent_cache=latent_cache,  # kvcache and k_pe
            q_pe=q_pe_attn,  # used by applyMLARopeAndAssignQKVKernelOptContext
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

    def _apply_output_gate(
        self,
        hidden_states: torch.Tensor,
        attn_output: torch.Tensor,
    ) -> torch.Tensor:
        """Transform the attention output before the output projection.

        ``attn_output`` is the row-parallel ``o_proj``'s input and
        ``hidden_states`` is the unquantized module input, so an override that
        derives a gate from ``hidden_states`` must match ``o_proj``'s input
        sharding. Runs on both the ``register_to_config`` and eager branches.

        The base is the identity.
        """
        return attn_output

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
        """Apply the default MLA output projection implementation."""
        o_proj = self.o_proj
        plan = getattr(self, "_k31_low_m_plan", None)
        if (
            plan is not None
            and o_proj.all_reduce is None
            and low_m_choice(plan, attn_output, o_proj.weight) is not None
        ):
            # The layout plans this o_proj (attention DP: a plain GEMM, no reduction in it).
            return k31_low_m_linear(attn_output, o_proj.weight, plan, True)
        return self.o_proj(attn_output)

    def forward(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        if self.register_to_config:
            attn_output = [create_mla_outputs(hidden_states, self.layer_idx_str)]
            self._forward_custom_op(
                hidden_states,
                position_ids,
                attn_output,
            )
        else:
            attn_output = self._create_outputs(hidden_states, attn_metadata)
            self.forward_impl(
                position_ids,
                hidden_states,
                attn_metadata,
                attn_output=attn_output,
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
        # Keep the base MLA registration enabled so breakable CUDA graphs use
        # the shared custom op. The output gate is a base hook and runs on both
        # the registered and eager paths before the row-parallel o_proj.
        # Prefill-graph padding rows leave the core zero, so the gate, o_proj and
        # the MoE router after them only ever see finite rows.
        self.zero_padding_output_rows = True
        # Low-M plan of the output projection (and of the packed DSA projections where they are packed).
        self._k31_low_m_plan = _k31_decode_plan(model_config.mapping)

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

        # FP32 router weight: the router GEMM runs in FP32 (see K31MoE).
        self.weight = nn.Parameter(
            torch.empty((self.num_experts, self.gating_dim), dtype=torch.float32, device=device)
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(self.num_experts, dtype=torch.float32, device=device)
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute fp32 routing logits shaped ``[num_tokens, num_experts]``."""
        hidden_2d = hidden_states.reshape(-1, self.gating_dim)
        return torch.nn.functional.linear(
            hidden_2d.to(torch.float32),
            self.weight.type(torch.float32),
            None,
        )


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


#: Generation rows per rank the PDL attn_res mixes take on this attention-DP target; the persistent mix is
#: faster from 99 rows with 7+ candidates, from 192 rows with any, and for prefill.
_K31_MIX_PERSISTENT_ROWS = 192


def _k31_mix_persistent(m: int, n: int) -> bool:
    return m >= _K31_MIX_PERSISTENT_ROWS or (m >= 99 and n >= 7)


class _K31PendingMoEParts(NamedTuple):
    """A decoder layer's output whose MoE residual update ``prefix + bf16(routed + shared)`` is left to the next
    layer's pre-attention mix (``k31_attn_res_moe_add_rmsnorm``, bitwise to the two adds)."""

    prefix: torch.Tensor
    routed: torch.Tensor
    shared: torch.Tensor


def _k31_settle_moe_parts(pending: _K31PendingMoEParts) -> torch.Tensor:
    """The running residual with the MoE output added (``k31_moe_residual_add``, the bits of the two adds)."""
    if _k31_residual_add_fits(pending.prefix, pending.routed, pending.shared):
        return k31_moe_residual_add(pending.prefix, pending.routed, pending.shared)
    return pending.prefix + (pending.routed + pending.shared)


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


class K31TargetRoutedExperts(K31RoutedExperts):
    """This rank's routed experts of one MoE layer under attention DP (``ops/moe/routed_experts``), with the fused
    path: up to its rows per rank (:meth:`forward_fused`) one fused op dispatches, runs the experts and combines; above
    them (:meth:`forward`) the call routes this rank's rows (the MoE prologue; where it declines, the routing tail or
    ``noaux_tc`` and the MXFP8 quantization), dispatches them with the one-sided all-to-all, runs this rank's experts
    on every rank's rows (the CuTe DSL grouped GEMMs up to 4096 rows, else the TRTLLM-Gen runner at the pinned tactic
    or the autotuned one) and combines the outputs back onto this rank's rows.
    """

    def forward_fused(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        all_rank_num_tokens: List[int],
        max_tokens: int,
        place: Optional[torch.Tensor] = None,
        real_rows: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """The routed MoE of this rank's ``x`` ``[ntok, 3584]`` BF16 through the fused dispatch + experts + combine
        op, or ``None`` where the call keeps :meth:`forward` (a step row count above ``max_tokens``, a step the MoE
        token budget splits into chunks, or a CUDA-graph capture before the group's workspace exists; every rank
        decides alike). ``place``: the layer's expert placement table (see ``k31_w4a8_ep_moe_fused``)."""
        ws = fused_moe_workspace(self.mapping, all_rank_num_tokens, max_tokens)
        if ws is None or num_chunks(all_rank_num_tokens, self.moe_max_num_tokens, True) != 1:
            return None
        ntok = x.shape[0]
        dev = x.device
        if ntok == 0:
            ids = torch.empty(0, _FUSED_MOE_TOP_K, dtype=torch.int32, device=dev)
            weights = torch.empty(0, _FUSED_MOE_TOP_K, dtype=torch.bfloat16, device=dev)
            x_fp8 = torch.empty(0, self.hidden_size, dtype=torch.float8_e4m3fn, device=dev)
            x_sf = torch.empty(0, self.hidden_size // 32, dtype=torch.uint8, device=dev)
        else:
            routed = self.route(x, router_logits)
            if routed is None:
                ids, weights = separated_routing(
                    router_logits, self._bias(), self.routed_scaling_factor, True
                )
                x_fp8, x_sf = quantize_input(x)
            else:
                ids, weights, x_fp8, x_sf = routed
            ids = ids.to(torch.int32)
            route_capture = get_active_route_capture()
            if route_capture is not None:
                route_capture.capture(self.layer_idx, ids, 0)
            record_expert_statistic(self.layer_idx, self.num_experts, ids)
            weights = weights.to(torch.bfloat16)
        return k31_w4a8_ep_moe_fused(
            ws,
            x_fp8,
            x_sf,
            ids.contiguous(),
            weights.contiguous(),
            self.w3_w1_weight,
            self.w3_w1_weight_scale,
            self.w2_weight,
            self.w2_weight_scale,
            self.slot_start,
            self.expert_size_per_partition,
            max(all_rank_num_tokens),
            *self.caps,
            real_rows=real_rows,
            place=place,
        )


class K31MoE(nn.Module):
    """Kimi K3.1 latent MoE block: the routed experts (``K31TargetRoutedExperts``) and the shared expert."""

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

        # Routing scores stay FP32, and the gate GEMM runs in FP32: a BF16-input GEMM's reduction order changes
        # borderline top-16 picks, and its latency gain does not matter at DEP batch sizes.
        self.gate = K31MoEGate(cfg)

        routed_quant_config = self._resolve_routed_quant_config(model_config, layer_idx)
        # Resolved here so ``load_weights`` reads the checkpoint layout off the
        # module instead of re-deriving it at each of its three call sites.
        self.expert_ckpt_spec = expert_ckpt_spec(routed_quant_config.quant_algo)
        self.routed_experts = K31TargetRoutedExperts(
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
        # Attention-DP decode: the routed MoE as one fused op (dispatch, CTA-pair FC1, FC2 with the combine in
        # its epilogue, token-side reduce), bitwise to the separate path it replaces. The decision per call depends
        # only on group-wide values (the step's largest per-rank row count).
        self._k31_extra_attrs = model_config.extra_attrs
        self._fused_moe_max_num_tokens = model_config.max_num_tokens
        # Resolved on the first forward (the weights exist by then); 0 = the separate path.
        self._fused_moe_max_tokens = None
        local_expert_ids = self.routed_experts.initial_local_expert_ids
        self.local_expert_ids = tuple(local_expert_ids)
        self.experts_per_rank = len(local_expert_ids)
        self.expert_lo = local_expert_ids[0]
        self.expert_hi = self.expert_lo + self.experts_per_rank
        # Checkpoint expert of each local routed-expert slot (identity placement unless the layer is placed).
        self.ckpt_expert_ids = self.local_expert_ids
        # Static expert placement (moe_config.load_balancer, see
        # ``kimi_k31_pro_mxfp4__sm_103__dep/ops/moe/expert_placement``): physical slot ``p`` holds checkpoint expert
        # ``slot_experts[p]``, so slot ``s`` of this rank loads ``slot_experts[lo + s]``. The fused path dispatches each
        # routed expert to its slot (``_expert_place``: expert -> slot) and sums by the identity placement's owner
        # ranks, so the layer's output is the same bits.
        slot_experts = self._k31_expert_placement(model_config, layer_idx)
        self._expert_place = None if slot_experts is None else expert_slots(slot_experts)
        self._expert_place_t = None
        if slot_experts is not None:
            self.ckpt_expert_ids = slot_experts[self.expert_lo : self.expert_hi]
            moved = sum(
                p // self.experts_per_rank != e // self.experts_per_rank
                for e, p in enumerate(self._expert_place)
            )
            logger.info_once(
                f"Kimi K3.1 routed MoE expert placement: on (moe_config.load_balancer; layer {layer_idx}: {moved} "
                f"of {self.num_experts} experts on another rank than the identity placement)",
                key="k31_expert_placement",
            )

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
            # Decode gate / up projection on the layout's low-M plan.
            low_m_plan=_k31_decode_plan(model_config.mapping),
        )
        # Side stream (+ fork/join events) for overlapping shared-expert
        # compute with the routed chain. Only engaged when multi-stream is
        # active (CUDA graphs on); otherwise both run in order on the default
        # stream.
        self.shared_expert_stream = aux_stream_dict[AuxStreamType.MoeShared]
        self.moe_main_event = torch.cuda.Event()
        self.moe_shared_event = torch.cuda.Event()
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

    def _k31_expert_placement(self, model_config: ModelConfig, layer_idx: int) -> Optional[tuple]:
        """The layer's physical slot -> expert table from moe_config.load_balancer, or ``None`` (identity placement:
        no configuration, no assignments, or the identity). A placed layer needs every routed call on the fused path
        (the separate path routes by expert id, which no longer names the slot holding the weights): attention DP over
        a pure-EP group of the TP ranks, otherwise this raises ``ValueError``; routing admits a placement only with a
        token budget per rank the fused path serves in one call."""
        slot_experts = layer_slot_experts(
            model_config.moe_load_balancer, layer_idx, self.num_experts
        )
        if slot_experts is None:
            return None
        mapping = self.routed_experts.mapping
        world = mapping.moe_ep_size
        checks = (
            (
                mapping.enable_attention_dp
                and mapping.moe_tp_size == 1
                and world == mapping.tp_size
                and world >= 2,
                "the fused routed MoE runs attention DP over a pure-EP group of the TP ranks only",
            ),
            (
                self.num_experts == _FUSED_MOE_NUM_EXPERTS
                and self.num_experts % world == 0
                and self.expert_lo == mapping.moe_ep_rank * self.experts_per_rank,
                f"the fused routed MoE serves {_FUSED_MOE_NUM_EXPERTS} experts split evenly over the EP ranks only",
            ),
        )
        for ok, why in checks:
            if not ok:
                raise ValueError(
                    "Kimi K3.1 expert placement (moe_config.load_balancer) needs the fused routed MoE for every "
                    f"routed call: {why}"
                )
        return slot_experts

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
        # moe_config.load_balancer is a static expert placement the target applies itself (``_k31_expert_placement``:
        # the fused routed MoE dispatches to the placed slots, the loader fills each slot with its checkpoint
        # expert); the routed experts keep the contiguous identity partition.
        return Mapping.from_dict(mapping_dict)

    def forward(
        self, hidden_states: torch.Tensor, all_rank_num_tokens=None, return_parts: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """``hidden_states``: ``[num_tokens, hidden_size]`` bf16; returns the MoE output, or with ``return_parts``
        the routed and shared outputs it sums."""
        identity = hidden_states
        router_logits = self._router_logits(hidden_states)

        def _routed_output():
            routed_in = self._routed_projection(hidden_states, self.routed_expert_down_proj)
            y = None
            if self._fused_moe_max_tokens is None:
                why = fused_moe_decline(
                    self.routed_experts.mapping,
                    self.routed_experts.w3_w1_weight,
                    isinstance(self.routed_experts.comm, K31EpAllToAll),
                )
                self._fused_moe_max_tokens = (
                    0
                    if why
                    else fused_moe_max_tokens(
                        self.routed_experts.mapping, self._fused_moe_max_num_tokens
                    )
                )
                logger.info_once(
                    f"Kimi K3.1 routed MoE fused path: {'off (' + why + ')' if why else 'on'} "
                    f"(up to {self._fused_moe_max_tokens} tokens per rank)",
                    key="k31_ep_moe_fused_path",
                )
            place = None
            if self._expert_place is not None:
                if self._expert_place_t is None:
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError(
                            "Kimi K3.1 expert placement: first forward under CUDA-graph capture"
                        )
                    self._expert_place_t = torch.tensor(
                        self._expert_place, dtype=torch.int32, device=routed_in.device
                    )
                place = self._expert_place_t
            if self._fused_moe_max_tokens and all_rank_num_tokens is not None:
                y = self.routed_experts.forward_fused(
                    routed_in,
                    router_logits,
                    all_rank_num_tokens,
                    self._fused_moe_max_tokens,
                    place=place,
                    real_rows=self._k31_extra_attrs.get("kimi_k31_moe_real_rows"),
                )
            if y is None and place is not None:
                # The separate path routes by logical expert id, which no longer names the slot holding the weights.
                raise RuntimeError(
                    "Kimi K3.1 expert placement needs the fused routed MoE for every call; this call "
                    f"({max(all_rank_num_tokens or [0])} tokens per rank) left it"
                )
            if y is None:
                y = self.routed_experts(
                    routed_in,
                    router_logits,
                    all_rank_num_tokens=all_rank_num_tokens,
                )
            # The combine returns this rank's complete routed latent.
            y = flashinfer_rmsnorm(
                y, self.routed_expert_norm.weight, self.routed_expert_norm.variance_epsilon
            )
            return self._routed_projection(y, self.routed_expert_up_proj)

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
        if return_parts:
            return routed_out, shared_out
        return routed_out + shared_out


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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # K31MLA.forward takes position_ids first; K3.1 is NoPE, so pass None.
        out = self.mixer(None, hidden_states, attn_metadata)
        if self._o_allreduce is not None:
            # Head-sharded TP: sum the row-sharded o_proj partials across
            # the head-shard group.
            out = allreduce_module(out, self._o_allreduce, FUSION_NONE, None, None)
        return out


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

    # Set by the model on every layer but the last: the layer may return a ``_K31PendingMoEParts`` (the MoE
    # residual update left to the next layer's pre-attention mix) instead of the running residual.
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
        if isinstance(hidden_states, _K31PendingMoEParts):
            pending = hidden_states
            operands = (
                _k31_mix_operands(
                    pending.prefix,
                    valid_block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                    self.input_layernorm,
                )
                if capture is None and num_snapshots > 0
                else None
            )
            if operands is not None and not _k31_mix_persistent(
                pending.prefix.shape[0], num_snapshots + 1
            ):
                mixed = k31_attn_res_moe_add_rmsnorm(
                    pending.prefix, pending.routed, pending.shared, valid_block_residual, *operands
                )
            else:
                hidden_states = _k31_settle_moe_parts(pending)
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
            hidden_states = self.linear_attn(hidden_states, attn_metadata)
        else:
            hidden_states = self.self_attn(hidden_states, attn_metadata)

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
            routed_out, shared_out = self.block_sparse_moe(
                hidden_states,
                getattr(attn_metadata, "all_rank_num_tokens", None),
                return_parts=True,
            )
            if (
                self._k31_pending_out
                and _k31_residual_add_fits(prefix_sum, routed_out, shared_out)
                and not _k31_mix_persistent(prefix_sum.shape[0], 2)
            ):
                # The next layer's pre-attention mix adds the MoE output to the residual.
                return _K31PendingMoEParts(prefix_sum, routed_out, shared_out), num_snapshots
            if _k31_residual_add_fits(prefix_sum, routed_out, shared_out):
                # prefix + (routed + shared) in one kernel, bitwise to the two adds.
                return k31_moe_residual_add(prefix_sum, routed_out, shared_out), num_snapshots
            hidden_states = routed_out + shared_out
        else:
            hidden_states = self.mlp.low_m_forward(hidden_states)

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


class K31TextModel(DecoderModel):
    """The Kimi K3.1 text model: the token embedding, the decoder layers with their attention-residual bank, and the
    final RMSNorm. Under DFlash2 it hands the drafter the hidden states of the layers the drafter reads."""

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        cfg = text_config_of(model_config.pretrained_config)
        dtype = torch.bfloat16

        # Attention and MoE phases are sequential, so their branch-overlap roles share one stream; the routed
        # experts' chunk overlap has its own.
        aux_stream_list = [torch.cuda.Stream() for _ in range(2)]
        self.aux_stream_dict = {
            AuxStreamType.Attention: aux_stream_list[0],
            AuxStreamType.MoeShared: aux_stream_list[0],
            AuxStreamType.MoeChunkingOverlap: aux_stream_list[1],
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
        for layer in self.layers[:-1]:
            layer._k31_pending_out = True

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

        return self._forward_layers(attn_metadata, hidden_states, spec_metadata)

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


def _drafter_mlp(mlp: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The drafter's SwiGLU MLP: ``gate_up_proj`` (on the drafter's low-M plan where it serves), the SwiGLU and
    ``down_proj``."""
    h1 = _k31_planned_linear(x, mlp.gate_up_proj.weight, mlp._k31_low_m_plan)
    if h1 is None:
        h1 = mlp.gate_up_proj(x)
    return mlp.down_proj(silu_and_mul(h1))


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
    """The Kimi K3.1 DFlash2 drafter (over ``DraftModelBase``) as this target runs it: the layout's low-M GEMM
    plan (packed ``[q_a | kv_a]``), the context-latent projection and the block decode with the planned projections
    of attention DP."""

    def __init__(self, draft_config: ModelConfig, **kwargs):
        super().__init__(draft_config, **kwargs)
        # The layout's low-M GEMM plan for the block decode and context projections (shared with the target).
        self._k31_low_m_plan = _k31_decode_plan(getattr(draft_config, "mapping", None))

    def _k31_pack_block_projections(self) -> None:
        """With a low-M plan: one ``[q_a | kv_a]`` weight per layer (the modules keep row views of it), and the
        plan handed to the block convolutions and the MLPs."""
        plan = self._k31_low_m_plan
        if plan is None:
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
        del block_placement
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
        page_tables = ctx_page_table.index_select(0, ctx_cache_batch_idx.long()).long()
        page_size = ctx_kv_cache[0].shape[-2]
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
            # Absorb W_UK into the query: [H, N, nope] @ [H, nope, latent].
            query[..., :latent_dim] = torch.bmm(
                q[..., : self.qk_nope_head_dim].transpose(0, 1), attn._k_b_proj.to(q.dtype)
            ).transpose(0, 1)
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
            attn_out = torch.einsum("nhc,hvc->nhv", ctx_out, attn._v_b_proj.to(ctx_out.dtype))
            o_in = attn_out.reshape(num_tokens, num_heads * self.v_head_dim)
            a = None
            if getattr(attn.o_proj, "all_reduce", None) is None:
                a = _k31_planned_linear(o_in, attn.o_proj.weight, self._k31_low_m_plan)
            if a is None:
                a = attn.o_proj(o_in)
            a = self.attention_convs[layer_idx].finish(a, attn_coefficients, block_size)

            norm = layer.post_attention_layernorm
            h2, residual = flashinfer_fused_add_rmsnorm(
                a, residual, norm.weight, norm.variance_epsilon
            )
            h2, mlp_coefficients = self.mlp_convs[layer_idx].prepare(h2, block_size)
            hidden_states = self.mlp_convs[layer_idx].finish(
                _drafter_mlp(layer.mlp, h2), mlp_coefficients, block_size
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
    """Use projection activations for the context cache, not packed weight storage."""
    projection = getattr(draft_model, "fc", None)
    if projection is None:
        return torch.bfloat16
    return getattr(projection, "dtype", projection.weight.dtype)


def _is_engine_type(obj, module: str, name: str) -> bool:
    """Whether ``obj``'s class is exactly the engine class ``module.name`` (not a subclass), checked by name."""
    return type(obj).__module__ == module and type(obj).__qualname__ == name


def _k31_stock_logits_processor(logits_processor) -> bool:
    """Whether ``logits_processor`` is the engine's stock ``LogitsProcessor`` that ``DecoderModelForCausalLM``
    builds (last-token gather, ``lm_head``, float cast)."""
    return _is_engine_type(
        logits_processor, "tensorrt_llm._torch.modules.logits_processor", "LogitsProcessor"
    )


def _k31_plain_lm_head(lm_head) -> bool:
    """Whether ``lm_head(x)`` is the replicated BF16 ``F.linear(x, lm_head.weight)`` (the vocab-parallel head's
    bitwise contract): unquantized, unsharded, no bias, no output-dtype GEMM, no custom cuBLAS route (routing refuses
    the engine's low-M GEMM dispatcher, the CuTe BF16 GEMM and, with speculative decoding, the LM head TP under
    attention DP)."""
    weight = getattr(lm_head, "weight", None)
    return (
        weight is not None
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
        and _is_engine_type(
            getattr(lm_head, "quant_method", None),
            "tensorrt_llm._torch.modules.linear",
            "UnquantizedLinearMethod",
        )
        and getattr(lm_head, "tp_size", 1) == 1
        and getattr(lm_head, "bias", None) is None
        and not getattr(lm_head, "use_custom_cublas_mm", False)
        and not (hasattr(lm_head, "_writes_output_dtype") and lm_head._writes_output_dtype())
    )


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
        # Rows per slot when the private paged arena is a context ring bounded by the drafter's
        # window (ops/spec/dflash2_ring): position p lives at slot row p mod _ctx_ring_rows. None when
        # the arena is dense in position or the pages are the manager's.
        self._ctx_ring_rows = None
        # Decode the manager draft pool's device block offsets instead of
        # building the table on the host (see _refresh_unified_block_tables).
        self._device_draft_tables = False
        self._draft_table_checks_left = 0

    def set_draft_model(self, draft_model) -> None:
        super().set_draft_model(draft_model)

    def post_load_weights(self) -> None:
        """The engine's, then (on every rank at once, eagerly, once the weights are loaded): where the configuration
        runs the LM heads vocab-parallel, the group's exchange workspace on NVLink peer mappings (collective;
        ``kimi_k31_pro_mxfp4__sm_103__dep/ops/comm/adp_vocab_parallel_head.reserve_exchange``); the compile of the
        drafter's CuTe DSL selector kernels (``_k31_build_draft_kernels``) and of the vocab-parallel drafter head's
        (``_k31_build_adp_draft_kernels``)."""
        super().post_load_weights()
        if self._k31_adp_enabled() and exchange_supported(self.mapping):
            reserve_exchange(self.mapping, _ADP_EXCHANGE_ROWS)
        draft_ref = getattr(self, "_draft_model_ref", None)
        draft_model = draft_ref() if draft_ref is not None else None
        if draft_model is not None:
            self._k31_build_draft_kernels(draft_model)
            if self._k31_adp_draft_head_serves(draft_model):
                self._k31_build_adp_draft_kernels(draft_model)

    def _k31_build_draft_kernels(self, draft_model) -> None:
        """Compile the CuTe DSL kernels of the DFlash2 drafter's selector by running each once on dummy rows: the
        full-vocab top-32 of the FP32 block logits and the greedy selector path at ``max_draft_len`` slots for one and
        for several requests (two kernel variants). A kernel's first call costs seconds of host time per process
        (tracing, even with a warm compile cache), which a step that first meets it after the warmup adds to its
        requests' latency; compiled here, that cost is paid while the model loads."""
        selector = getattr(draft_model, "candidate_selector", None)
        if (
            not getattr(draft_model, "has_candidate_selector", False)
            or selector is None
            or selector.top_k != K31_SELECTOR_TOP_K
            or selector.predecessor_codebook.shape[1] % 256
        ):
            return
        weight = draft_model.lm_head.weight
        vocab = selector.predecessor_codebook.shape[0]
        device = weight.device
        K = self.spec_config.max_draft_len
        if weight.shape[0] == vocab:
            k31_dflash2_shard_top32(
                torch.zeros((1, K, vocab), dtype=torch.float32, device=device),
                K31_SELECTOR_TOP_K,
                0,
                False,
            )
        for n in (1, 2):
            draft_model.select_candidate_path(
                torch.full((n, K, vocab), float("-inf"), dtype=torch.float32, device=device),
                torch.arange(K31_SELECTOR_TOP_K, device=device).expand(n, K, -1).contiguous(),
                torch.zeros((n, K, K31_SELECTOR_TOP_K), dtype=torch.float32, device=device),
                torch.zeros((n, K, weight.shape[1]), dtype=weight.dtype, device=device),
                torch.zeros((n,), dtype=torch.int64, device=device),
            )
        logger.info("DFlash2 drafter selector kernels compiled at load.")

    def _local_draft_tokens(self, draft_model) -> Optional[VocabParallelDraftTokens]:
        """How greedy draft tokens come from the drafter's own full-vocab LM head
        output on this rank when there is no vocab-parallel selection (e.g. under
        attention DP), or None for the logits path. Plain DFlash has none."""
        del draft_model
        return None

    def _has_unified_draft_cache(self) -> bool:
        return bool(getattr(self._ctx_kv_manager, "draft_layer_ids", ()))

    def prepare_managed_draft_cache(
        self, draft_model, spec_metadata, attn_metadata, resource_manager
    ) -> None:
        """Before eager execution, capture or graph replay: publish this step's real rows, then the
        draft-page work (``DFlash2DraftPagesWorkerMixin``)."""
        # The routed MoE's fused path skips this step's CUDA-graph padding rows (the tail requests).
        from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID

        attrs = draft_model.model_config.extra_attrs
        if "kimi_k31_moe_real_rows" not in attrs:
            attrs["kimi_k31_moe_real_rows"] = torch.full(
                (4,), 2**31 - 1, dtype=torch.int32, device="cuda"
            )
        publish_real_rows(
            attn_metadata.num_tokens,
            attn_metadata.num_generations,
            attn_metadata.num_ctx_tokens,
            spec_metadata.request_ids,
            CUDA_GRAPH_DUMMY_REQUEST_ID - spec_metadata.max_draft_len,
            attrs["kimi_k31_moe_real_rows"],
        )
        super().prepare_managed_draft_cache(
            draft_model, spec_metadata, attn_metadata, resource_manager
        )

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

    def _k31_adp_enabled(self) -> bool:
        """Whether the configuration runs the LM heads vocab-parallel over the attention-DP group on the steps that
        allow it (``_k31_adp_rows``): attention DP over 16 ranks (the group size the shard GEMM's bitwise sweep
        covers, kimi_k31_pro_mxfp4__sm_103__dep/ops/comm/adp_vocab_parallel_head), no context parallelism, no guided
        decoder."""
        mapping = self.mapping
        return (
            mapping is not None
            and mapping.enable_attention_dp
            and mapping.tp_size == 16
            and getattr(mapping, "cp_size", 1) == 1
            and self.guided_decoder is None
        )

    def _k31_adp_rows(self, spec_metadata, rows_per_request: int) -> Optional[list[int]]:
        """Every attention-DP rank's row count for a step of generation requests only (``rows_per_request`` rows
        each), from the step's all-rank counts, which every rank holds alike; None where the vocab-parallel head
        does not run (``_k31_adp_enabled`` false, a context request on any rank, the counts missing)."""
        if not self._k31_adp_enabled():
            return None
        gens = getattr(spec_metadata, "all_rank_num_gens", None)
        seqs = getattr(spec_metadata, "all_rank_num_seqs", None)
        if (
            gens is None
            or seqs is None
            or len(gens) != self.mapping.tp_size
            or list(gens) != list(seqs)
        ):
            return None
        rows = [int(g) * rows_per_request for g in gens]
        # The shard GEMM is the replicated head's columns bit for bit at 16 ranks for 2..320 rows per rank
        # (kimi_k31_pro_mxfp4__sm_103__dep/ops/comm/adp_vocab_parallel_head's sweep); elsewhere the replicated head
        # runs.
        if min(rows) < 2 or max(rows) > 320:
            return None
        return rows

    def _k31_adp_draft_head_serves(self, draft_model) -> bool:
        """Whether the configuration runs the drafter's LM head vocab-parallel under attention DP (on the steps
        ``_k31_adp_rows`` allows): all-greedy DFlash2 selector drafts (no rejection sampling or d2t; the
        package's selector path over 32 candidates and whole 256 blocks of rank) and a plain head over the selector's
        full vocabulary, which the group splits evenly."""
        selector = getattr(draft_model, "candidate_selector", None)
        if (
            not self._k31_adp_enabled()
            or getattr(self.spec_config, "use_rejection_sampling", False)
            or self._d2t is not None
            or not getattr(draft_model, "has_candidate_selector", False)
            or selector is None
            or selector.top_k != K31_SELECTOR_TOP_K
            or selector.predecessor_codebook.shape[1] % 256
            or type(self)._refine_block_logits is not _EngineDFlashWorker._refine_block_logits
            or not _k31_stock_logits_processor(draft_model.logits_processor)
            or not _k31_plain_lm_head(draft_model.lm_head)
        ):
            return False
        full_vocab = selector.predecessor_codebook.shape[0]
        return (
            draft_model.lm_head.weight.shape[0] == full_vocab
            and full_vocab % self.mapping.tp_size == 0
        )

    def _k31_build_adp_draft_kernels(self, draft_model) -> None:
        """Compile the CuTe DSL kernels of the drafter's vocab-parallel head path by running each once on dummy rows,
        without collectives: the shard top-32 pairs and their merge, and the selector greedy tokens at
        ``max_draft_len`` slots for one and for several requests (two kernel variants). Their first call costs seconds
        of host time per process (tracing, even with a warm compile cache), and the first step that takes the path
        can come after the warmup: compiled here, that cost is paid while the model loads."""
        selector = draft_model.candidate_selector
        weight = draft_model.lm_head.weight
        world = self.mapping.tp_size
        device = weight.device
        build_top32_kernels(weight.shape[0] // world, weight.dtype, world, device)
        K = self.spec_config.max_draft_len
        for n in (1, 2):
            k31_dflash2_selector_greedy_tokens(
                selector.predecessor_codebook,
                selector.successor_codebook,
                selector.hidden_projection.weight,
                torch.arange(K31_SELECTOR_TOP_K, device=device).expand(n, K, -1).contiguous(),
                torch.zeros((n, K, K31_SELECTOR_TOP_K), dtype=torch.float32, device=device),
                torch.zeros((n, K, weight.shape[1]), dtype=weight.dtype, device=device),
                torch.zeros((n,), dtype=torch.int64, device=device),
                torch.float32,
            )
        logger.info(
            "DFlash2 drafter LM head under attention DP: vocab-parallel head kernels compiled at load."
        )

    def target_logits(
        self, hidden_states, lm_head, logits_processor, attn_metadata, spec_metadata, draft_model
    ):
        """The target's FP32 logits of the gathered rows; under attention DP, on a step of generation requests only,
        the replicated head's logits computed vocab-parallel over the group
        (``kimi_k31_pro_mxfp4__sm_103__dep/ops/comm/adp_vocab_parallel_head``, bitwise ``lm_head(rows).float()``), so
        the sampling reads the same values."""
        rows = (
            self._k31_adp_rows(spec_metadata, spec_metadata.runtime_draft_len + 1)
            if _k31_stock_logits_processor(logits_processor) and _k31_plain_lm_head(lm_head)
            else None
        )
        if rows is not None:
            logger.info_once(
                "Target LM head under attention DP: vocab-parallel over the group "
                "(kimi_k31_pro_mxfp4__sm_103__dep/ops/comm/adp_vocab_parallel_head).",
                key="k31_adp_vocab_target_head",
            )
            return k31_adp_vocab_logits(hidden_states, lm_head.weight, self.mapping, rows)
        return super().target_logits(
            hidden_states, lm_head, logits_processor, attn_metadata, spec_metadata, draft_model
        )

    def _k31_adp_greedy_selector_tokens(
        self, draft_model, gen_hidden_states, inputs, spec_metadata, num_gens: int, K: int
    ) -> Optional[tuple[torch.Tensor, int]]:
        """The DFlash2 selector's all-greedy draft tokens under attention DP with the drafter's LM head run
        vocab-parallel over the group (``kimi_k31_pro_mxfp4__sm_103__dep/ops/comm/adp_vocab_parallel_head`` top-32: the
        full-vocab candidates of the replicated head bit for bit) and the walk writing each slot's token
        (``ops/spec/dflash2_selector_greedy_tokens``): ``(tokens [num_gens, K] int32, vocab)``, or None where the
        replicated-head path runs. Every rank with generation requests takes the same branch: the conditions are
        configuration and the step's all-rank counts."""
        if spec_metadata.wants_advanced_draft_sampling or not self._k31_adp_draft_head_serves(
            draft_model
        ):
            return None
        rows = self._k31_adp_rows(spec_metadata, K)
        if rows is None:
            return None
        selector = draft_model.candidate_selector
        weight = draft_model.lm_head.weight
        logger.info_once(
            "DFlash2 drafter LM head under attention DP: vocab-parallel top-32 over the group, selector tokens.",
            key="k31_adp_vocab_draft_head",
        )
        unary_logits, candidate_ids = k31_adp_vocab_top32(
            gen_hidden_states, weight, self.mapping, rows
        )
        tokens = k31_dflash2_selector_greedy_tokens(
            selector.predecessor_codebook,
            selector.successor_codebook,
            selector.hidden_projection.weight,
            candidate_ids.view(num_gens, K, K31_SELECTOR_TOP_K),
            unary_logits.view(num_gens, K, K31_SELECTOR_TOP_K),
            gen_hidden_states.reshape(num_gens, K, -1),
            inputs["first_prev_tokens"].long(),
            torch.float32,
        )
        return tokens, weight.shape[0]

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
                    adp_tokens := self._k31_adp_greedy_selector_tokens(
                        draft_model, gen_hidden_states, inputs, spec_metadata, num_gens, K
                    )
                ) is not None:
                    gen_draft_tokens, vocab_size = adp_tokens
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
    "Cannot encode this image prompt because this server has no vision tower (disable_mm_encoder, "
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
        NotImplementedError: If a context step needs to encode an image without a local vision tower.
    """
    # Decode-only rows already have the image KV transferred from the context worker.
    if num_contexts == 0 or not multimodal_params:
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

        # The weight loader builds MoonViT for local image prefill. A disaggregated
        # generation worker consumes the image KV from its context worker.
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


class KimiK31ForConditionalGeneration(K31VLWrapper):
    """Kimi K3.1 vision-language model: the K3 wrapper over the package's text backbone, which
    reads the K3.1 ``dsa_config`` (a DSA indexer on every MLA layer) from its own config.

    Besides the model it states what the engine asks a model class for before and while it is
    built: model defaults, KV cache manager / transceiver preferences, the hybrid KDA + MLA + DSA
    cache manager (``runtime_hooks.kv_cache_manager_cls``), the DSA attention backend
    (``runtime_hooks.attention_backend_cls``) and the engine hooks
    (``runtime_hooks``, ``runtime.py``).
    """

    runtime_hooks = replace(K31_RUNTIME_HOOKS, attention_backend_cls=K31PerfDSABackend)

    @classmethod
    def get_model_defaults(cls, llm_args) -> dict:
        return K31ForCausalLM.get_model_defaults(llm_args)

    @classmethod
    def get_preferred_transceiver_runtime(cls, pretrained_config=None):
        return K31ForCausalLM.get_preferred_transceiver_runtime(pretrained_config)


class PrefillTarget:
    """Steps with context rows: multimodal fusion, then the text model."""

    def __init__(self, model: "ModelingV2KimiK31ProMxfp4Sm103Dep"):
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

    def __init__(self, model: "ModelingV2KimiK31ProMxfp4Sm103Dep"):
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


class ModelingV2KimiK31ProMxfp4Sm103Dep(KimiK31ForConditionalGeneration):
    """Kimi K3.1 Pro MXFP4, sm_103, attention DP perf layout (DEP16, experts EP16)."""

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
