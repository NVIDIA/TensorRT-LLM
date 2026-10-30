# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: Qwen3.8-Flash-Next-NVFP4 / sm_103 / tp1 — self-contained modeling code.

The core holds the static structure (weights, derived operands, per-manager pool
views); `PrefillTarget` (any batch with context rows) and `DecodeTarget`
(generation-only) run a step over it, passing every argument to every op at the
call site.

The checkpoint is the composite Qwen4-Exp VLM export. The MTP module is built only
when the engine has a ``speculative_config`` (else ``mtp.*`` stays in
``weights.py``'s non-load set). The vision tower is always built; a text-only
request never reaches it. Every speculation-dependent change is gated on a host
int that is 1 when speculation is off. The PLE pools and the QSA side caches are
written for every verify row, so `SpeculativeRollback` (registered with the spec
worker) rewrites them to the accepted prefix after each verify step.

Differences from other targets:

* **Hyper-connections.** The inter-block residual is ``4 * 2560 = 10240`` wide.
  There is **no** ``model.language_model.norm.weight``: the stack-final
  ``hyper_connection_mixer`` is the final norm and ``lm_head`` reads its output.
  In the trunk each site is ``ops/hc/qwen4_exp_hc_inject_norm`` (the previous block's
  accumulate fused with this site's grouped norm; after a MoE block it also forms that
  block's output from the routed and shared outputs and the gate logit) and then the
  mixer: ``catalog/gemm/cublas_mm`` over the concatenated ``[down; block_inject]``
  weight, ``ops/hc/qwen4_exp_hc_lowrank_act`` and ``ops/hc/qwen4_exp_hc_up_gate_mean``
  (the up projection as a tcgen05 GEMM whose epilogue applies the sigmoid gate and the
  stream mean). ``DecodeTarget`` mixers run the first projection as one tcgen05 launch
  (``ops/hc/qwen4_exp_hc_decode_tc_first``, up to ``_HC_TC_FIRST_MAX_ROWS`` rows) or, at
  ``_HC_DECODE_GEMV_MAX_ROWS`` rows, the whole site as two GEMV launches
  (``ops/hc/qwen4_exp_hc_decode_first_cute``, ``ops/hc/qwen4_exp_hc_decode_mix_fp4_cute``);
  at the 48 MoE sites the mixer's last launch also writes the routed experts' NVFP4
  input. The MTP layer and the prefill PLE block keep the unfused
  ``qwen4_exp_gated_residual`` / ``qwen4_exp_post_inject`` / ``_grouped_norm``.
* **QSA sparse attention.** The index plane is maintained on every step so a
  request can cross ``dense_seq_len_threshold`` later without rebuilding its
  prefix; sparse selection replaces the dense core only above it. The MTP draft
  layer runs the same branch over its own indexer and its own manager's planes.
* **PLE.** Decoder layer 1 only (``ple_layer_ids: [2]`` is one-based): an n-gram
  side path over a 320,001,536-row fp8 table (47.68 GiB of a 123.5 GiB checkpoint).
* **NVFP4 MoE on every layer** (routed experts only); everything else is bf16. The
  trunk's routed experts run on trtllm-gen's block-scale runner (``TRTLLM``, and what
  ``AUTO`` selects) or on the CuteDSL sort / grouped-GEMM pipeline (``CUTEDSL``),
  chosen by ``moe_config.backend``; any other request is refused. The MTP draft
  layer's FP8 block-scale experts run on trtllm-gen under either. Under ``TRTLLM`` the
  prefill-sized launches (``_CUTE_DSL_MOE_MIN_ROWS``) run the CuteDSL pipeline over a
  second copy of the experts in its layout, about 1.3 GiB a layer, kept only when a step
  can reach that size and ``TRTLLM_MODELING_V2_MOE_PREFILL_COPY`` does not turn it off
  (``_moe_prefill_copy``). A graph-captured decode step routes its block
  projections through ``ops/gemm/low_m_bf16_mm`` (FlashInfer low-M kernels at
  M <= 32 on the shapes it lists, cuBLAS otherwise); see ``DecodeTarget.forward``.

Integration facts this file depends on:

1. The engine rebuilds the composite config and normalizes its text tower
   (``config_utils.load_pretrained_config``). The shell adopts that text config as
   ``model_config.pretrained_config`` before building the core, because the
   recurrent-state and PLE pool geometry are read from it.
2. ``position_ids`` arrives in the mRoPE layout (``[3, 1, T]``) even for text-only
   batches, because the normalized text config carries ``rope_scaling['type'] ==
   'mrope'``. The main and MTP rope read all three rows; the QSA index plane takes
   coordinate 0 only (see ``_QSA_MROPE_SECTION``).
3. The engine owns the multimodal input processor (resolved off the checkpoint's
   ``architectures[0]``; modeling_v2 has no hook), so it is the built-in
   ``Qwen3VLInputProcessorBase``. This target asserts every ``multimodal_data`` key
   where it is read.

**KV storage.** The paged K/V pool holds the checkpoint's bf16, fp8-e4m3 when the engine
resolves ``kv_cache_quant_algo: FP8`` (``kv_cache_config.dtype: fp8``), or NVFP4 when it resolves
``NVFP4`` (``kv_cache_config.dtype: nvfp4``): a lossy substitution, since the checkpoint's KV is
bf16 and the fp8 and NVFP4 attention kernels run q, K, V and the softmax probabilities through
e4m3. An NVFP4 pool stores e2m1 pairs (uint8, half a head wide) and keeps one e4m3 scale per 16
values in a second pool; the second-level scales are 1.0. The NVFP4 attention kernels exist for
the paged-context path only, so an NVFP4 engine needs chunked prefill, block reuse or
speculative decoding. The QSA index and position planes keep their own types under all three.

**Recurrent-state storage.** The Gated-DeltaNet recurrent pool holds the checkpoint's fp32
(``mamba_ssm_dtype``) or bf16 when ``kv_cache_config.mamba_ssm_cache_dtype`` asks for it, which
is also what ``auto`` resolves to for this checkpoint (`_SSM_STATE_DTYPES`): a lossy
substitution, since the state is rounded to bf16 once per decode token, once per prefill call and
at every cached-replay checkpoint commit. The conv, PLE and replay-history buffers keep their own
types under both. A bf16 pool makes the decode step call its recurrence without ``cu_seqlens``;
see ``DecodeTarget._gdn_block``.

**Breakable prefill CUDA graph.** With ``prefill_cuda_graph_backend: breakable`` the engine
captures the core's ``forward`` (the body, not the shell) for a step with context rows at each of
``prefill_capture_num_tokens``, pads any such step up to the nearest bucket and replays it without
calling the body. The shell therefore calls the core's ``begin_step`` before the body on every
step: the asserts, the cache-manager bind, the target choice and the prefill target's host
bookkeeping (``PrefillTarget.prepare``, one immutable ``_PrefillStep``) all live there. The prefill
``forward`` is row-wise over the ``T`` rows it is handed, with four breaks
(``catalog/graph/eager_on_graph``) that run eagerly between the captured segments and read the
step: the vision splice, the PLE block, and each layer's attention core and Gated-DeltaNet
recurrence. Each writes rows ``[:n]`` of a fixed output and zeroes rows ``[n:]``. Under MTP the
prefill body returns one packed ``[T, hidden + stream]`` tensor that ``split_spec_output`` splits,
and the shell hands the spec worker the batch's own ``n`` rows. With the backend off every step is
eager and unpadded and the breaks are plain calls.

**Host offload.** By default the PLE table lives in host memory the device
addresses (`weights._host_resident_empty`), and the gather reads it unchanged;
``TRTLLM_MODELING_V2_PLE_HOST_OFFLOAD=0`` restores the resident declaration. Output
is identical, so only the device-memory delta sees it. The gather stays
device-driven because row indices are produced on device inside a captured decode
forward, which cannot express a host readback.
"""

import math
import os
from typing import NamedTuple

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import step_contract_enabled
from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.flashinfer_silu_and_mul import (  # noqa: E501
    flashinfer_silu_and_mul,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.fused_qk_norm_rope import (
    fused_qk_norm_rope,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
    QM_FP8_KV_CACHE,
    QM_NVFP4_KV_CACHE,
    thop_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.graph.eager_on_graph import (
    eager_on_graph,
)
from tensorrt_llm._torch.attention.backends.fmha.phased import get_spec_decoding_position_offsets
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.modules.fla.cached_replay import CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE
from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor
from tensorrt_llm._torch.speculative import get_spec_worker, should_use_separate_draft_kv_cache
from tensorrt_llm._torch.utils import ActivationType
from tensorrt_llm.inputs.multimodal import get_multimodal_item_metadata
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

from . import weights as _weights
from .ops.activation.fused_sigmoid_mul_inplace import fused_sigmoid_mul_inplace
from .ops.activation.gelu_erf import gelu_erf
from .ops.activation.gelu_tanh import gelu_tanh
from .ops.activation.silu_and_mul_rows import silu_and_mul_rows
from .ops.attention.qsa_canonical_selection import qsa_canonical_selection
from .ops.attention.qsa_expand_block_indices import qsa_expand_block_indices
from .ops.attention.qsa_mtp_reuse_topk import qsa_mtp_reuse_topk
from .ops.attention.qsa_paged_index_scores import qsa_paged_index_scores
from .ops.attention.qsa_paged_index_store import qsa_paged_index_store
from .ops.attention.qsa_paged_kv_store import qsa_paged_kv_store
from .ops.attention.qsa_paged_kv_store_nvfp4 import qsa_paged_kv_store_nvfp4
from .ops.attention.qsa_paged_sparse_gqa import qsa_paged_sparse_gqa
from .ops.attention.qsa_paged_sparse_gqa_decode import MAX_ROWS as _SPARSE_DECODE_MAX_ROWS
from .ops.attention.qsa_paged_sparse_gqa_decode import qsa_paged_sparse_gqa_decode
from .ops.attention.qsa_paged_sparse_gqa_decode import supports as _sparse_decode_supports
from .ops.attention.qsa_paged_sparse_gqa_nvfp4 import qsa_paged_sparse_gqa_nvfp4
from .ops.attention.qsa_prefill_compress import qsa_prefill_compress
from .ops.attention.qsa_sparse_gqa_rowtile import qsa_sparse_gqa_rowtile
from .ops.attention.qsa_topk_expand_decode import qsa_topk_expand_decode
from .ops.attention.qsa_topk_expand_decode import supports as _topk_expand_supports
from .ops.attention.qsa_topk_select import qsa_topk_select
from .ops.attention.qwen4_exp_attn_decode_prologue import qwen4_exp_attn_decode_prologue
from .ops.attention.qwen4_exp_qk_norm_rope import qwen4_exp_qk_norm_rope
from .ops.attention.vision_rope_2d import vision_rope_2d
from .ops.embedding.ngram_hash import ngram_hash
from .ops.gemm.bf16_gemv_pdl import TILING as _GEMV_TILED_K
from .ops.gemm.bf16_gemv_pdl import bf16_gemv_pdl
from .ops.gemm.low_m_bf16_mm import low_m_bf16_mm
from .ops.hc.qwen4_exp_gated_residual import qwen4_exp_gated_residual
from .ops.hc.qwen4_exp_hc_decode_first_cute import qwen4_exp_hc_decode_first_cute
from .ops.hc.qwen4_exp_hc_decode_mix_fp4_cute import (
    qwen4_exp_hc_decode_mix_cute,
    qwen4_exp_hc_decode_mix_fp4_cute,
)
from .ops.hc.qwen4_exp_hc_decode_tc_first import qwen4_exp_hc_decode_tc_first
from .ops.hc.qwen4_exp_hc_inject_norm import qwen4_exp_hc_inject_norm
from .ops.hc.qwen4_exp_hc_lowrank_act import qwen4_exp_hc_lowrank_act
from .ops.hc.qwen4_exp_hc_up_gate_mean import qwen4_exp_hc_up_gate_mean
from .ops.hc.qwen4_exp_hc_up_gate_mean_fp4 import qwen4_exp_hc_up_gate_mean_fp4
from .ops.hc.qwen4_exp_post_inject import qwen4_exp_post_inject
from .ops.moe.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion import (
    cute_dsl_nvfp4_gather_grouped_gemm_act_fusion,
)
from .ops.moe.cute_dsl_nvfp4_grouped_gemm import cute_dsl_nvfp4_grouped_gemm
from .ops.moe.fp4_block_scale_moe_renorm_permuted import fp4_block_scale_moe_renorm_permuted
from .ops.moe.fp4_block_scale_moe_renorm_routed import fp4_block_scale_moe_renorm_routed
from .ops.moe.fp4_moe_renorm_cute_dsl import fp4_moe_renorm_cute_dsl
from .ops.moe.fp8_block_scale_moe_runner import fp8_block_scale_moe_runner
from .ops.moe.fused_sigmoid_gate_mul_add import fused_sigmoid_gate_mul_add
from .ops.moe.moe_sort import moe_sort
from .ops.moe.moe_unpermute_inplace import moe_unpermute_inplace
from .ops.moe.renorm_moe_routing_op import renorm_moe_routing_op
from .ops.norm.flashinfer_gemma_rmsnorm import flashinfer_gemma_rmsnorm
from .ops.norm.layer_norm import layer_norm
from .ops.norm.rms_norm_gated_token_major import rms_norm_gated_token_major
from .ops.ple.gated_ngram_injection import gated_ngram_injection
from .ops.ple.ngram_token_history import ngram_token_history
from .ops.ple.ple_gate_conv_inject_norm import ple_gate_conv_inject_norm
from .ops.ple.ple_ngram_embed import ple_ngram_embed
from .ops.quantization.bf16_to_fp32 import bf16_to_fp32
from .ops.quantization.dequantize_e4m3_per_tensor import dequantize_e4m3_per_tensor
from .ops.quantization.fp8_quantize_1x128 import fp8_quantize_1x128
from .ops.ssm.causal_conv1d_fwd import causal_conv1d_fwd
from .ops.ssm.causal_conv1d_triton_update import causal_conv1d_triton_update
from .ops.ssm.causal_conv1d_update import causal_conv1d_update
from .ops.ssm.chunk_gated_delta_rule import chunk_gated_delta_rule
from .ops.ssm.cu_seqlens_from_seq_lens import cu_seqlens_from_seq_lens
from .ops.ssm.dilated_short_conv_paged import dilated_short_conv_paged
from .ops.ssm.extract_transpose_prefill_slice import extract_transpose_prefill_slice
from .ops.ssm.flashinfer_chunk_gated_delta_rule import flashinfer_chunk_gated_delta_rule
from .ops.ssm.fused_gdn_gating import fused_gdn_gating
from .ops.ssm.fused_gdn_post_conv import fused_gdn_post_conv
from .ops.ssm.fused_recurrent_gated_delta_rule_cached_replay_update import (
    fused_recurrent_gated_delta_rule_cached_replay_update,
)
from .ops.ssm.fused_recurrent_gated_delta_rule_update import fused_recurrent_gated_delta_rule_update
from .ops.ssm.fused_sigmoid_gating_delta_rule_update import fused_sigmoid_gating_delta_rule_update
from .ops.ssm.gdn_decode_gated_norm_pdl import MAX_ROWS as _GDN_DECODE_FUSED_MAX_ROWS
from .ops.ssm.gdn_decode_gated_norm_pdl import gdn_decode_gated_norm_pdl
from .ops.ssm.gdn_decode_rows_gated_norm import gdn_decode_rows, gdn_decode_rows_gated_norm
from .ops.ssm.gdn_in_proj_conv1d_pdl import MAX_ROWS as _GDN_CONV_FUSED_MAX_ROWS
from .ops.ssm.gdn_in_proj_conv1d_pdl import gdn_in_proj_conv1d_pdl
from .ops.ssm.gdn_prefill_conv_post import gdn_prefill_conv_post
from .ops.ssm.pack_gdn_decode_qkv import pack_gdn_decode_qkv
from .ops.ssm.reset_recurrent_state_rows import reset_recurrent_state_rows
from .ops.vision.patch_embed_conv3d import patch_embed_conv3d
from .ops.vision.pos_embed_interpolate import pos_embed_interpolate

# The GPU architecture this target is certified for; a direct instantiation could
# bypass routing.
_SM = (10, 3)

# QSA backend metadata buffers rebuilt by prepare(). Their absence means the engine
# did not select the QSA metadata and the index plane would never be written.
_QSA_FIELDS = (
    "qsa_block_table",
    "qsa_req_idx_per_token",
    "qsa_logical_positions",
    "qsa_sequence_lengths",
    "qsa_visible_blocks",
    "qsa_topk_indices",
    "qsa_topk_row_starts",
)

# Fields read off `attn_metadata.mamba_metadata`. Absence is silent: a smoke run
# passes with no recurrent state at all.
_MAMBA_FIELDS = (
    "state_indices",
    "has_initial_states",
    "query_start_loc",
    "query_start_loc_long",
)


# A context launch at or above this many rows takes `qsa_sparse_gqa_rowtile`;
# measured on GB300 it loses at 543 rows and wins from 4096 rows up.
_ROWTILE_MIN_ROWS = 2048  # capacity: fixed: a dispatch threshold, not a buffer size

#: Context-step MoE launches of at least this many rows run on the CuTe DSL MoE
#: (`fp4_moe_renorm_cute_dsl`, from the second expert copy `weights.py` keeps in
#: its layout); smaller ones, every decode one among them, keep the trtllm-gen
#: runner. From GB300 micro-benches, us a layer, CuTe DSL vs trtllm-gen: 2048
#: rows 275 vs 273, 8192 rows 451 vs 555, 16384 rows 656 vs 882.
_CUTE_DSL_MOE_MIN_ROWS = 4096  # capacity: fixed: a dispatch threshold, not a buffer size


def _decode_projection(mat_a: torch.Tensor, mat_b: torch.Tensor) -> torch.Tensor:
    """A graph-captured decode step's bf16 projection, always a PDL launch.

    One row runs `bf16_gemv_pdl` on every K it tiles (each decode projection of this
    checkpoint): its weight streams into registers before it waits on the producer, and it
    triggers its own dependents. More rows run `low_m_bf16_mm` with its FlashInfer
    branches' PDL on (its cuBLAS branch, above M=32 and off its table, ignores the flag).
    Every kernel either path launches reads the activation only after
    `griddepcontrol.wait`.
    """
    if mat_a.shape[0] == 1 and mat_b.shape[0] in _GEMV_TILED_K:
        return bf16_gemv_pdl(mat_a, mat_b, True)
    return low_m_bf16_mm(mat_a, mat_b, True)


# Per-call constants of the dense attention call shape: packed-QKV causal GQA, RoPE
# applied outside (fused_qk_norm_rope), bf16, fp8-e4m3 or NVFP4 pool. The sparse / skip-softmax
# group stays None/0: QSA calls selection and sparse attention as separate kernels and
# `num_sparse_topk` is 0. `quant_mode`, the two KV scales (all three follow the pool's
# storage) and `predicted_tokens_per_seq` are passed beside these.
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
    # In-kernel RoPE disabled; rope_* values are inert placeholders.
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
    # MLA-only surface, unused here.
    kv_norm_weight=None,
    kv_norm_eps=1e-6,
    skip_correction_threshold=0.0,
)

# ------------------------------------------------------------ vision ------

#: Tower rope base (`vision_config` has no rope field); unrelated to the text 1e7.
_VISION_ROPE_THETA = 10000.0

#: LayerNorm epsilon of `norm1`, `norm2`, `merger.norm`: a literal in the reference.
_VISION_NORM_EPS = 1e-6

#: `mask_type` 0 = padding mask: bidirectional within a batch segment. Causal (1)
#: would make a patch depend on raster order.
_VISION_MASK_PADDING = 0

# Per-call constants of the tower's attention: the encoder-only mode of
# `attention/thop_attention`. Omitting `kv_cache_block_offsets`,
# `host_kv_cache_pool_pointers` and `host_kv_cache_pool_mapping` selects that mode.
# `update_kv_cache=False` is rejected ("KV cache update cannot be disabled now"),
# so it stays True.
_VISION_CALL_CONSTANTS = dict(
    k=None,
    v=None,
    output_sf=None,
    kv_cache_block_offsets=None,
    host_kv_cache_pool_pointers=None,
    host_kv_cache_pool_mapping=None,
    tokens_per_block=None,
    use_paged_context_fmha=False,
    is_fused_qkv=True,
    update_kv_cache=True,
    predicted_tokens_per_seq=1,
    # No pool, so no layer to address.
    local_layer_idx=0,
    max_context_q_len_override=None,
    cache_indirection=None,
    kv_scale_orig_quant=None,
    kv_scale_quant_orig=None,
    out_scale=None,
    quant_mode=0,
    beam_width=1,
    mask_type=_VISION_MASK_PADDING,
    q_scaling=1.0,
    # Rotation is applied outside (`ops/attention/vision_rope_2d`); rope_* inert.
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
    attention_input_type=0,
    is_mla_enable=False,
    chunked_prefill_buffer_batch_size=1,
    attention_sinks=None,
    block_ids_per_seq=None,
    latent_cache=None,
    q_pe=None,
    q_lora_rank=None,
    kv_lora_rank=None,
    qk_nope_head_dim=None,
    qk_rope_head_dim=None,
    v_head_dim=None,
    rope_append=None,
    attention_chunk_size=None,
    softmax_stats_tensor=None,
    is_spec_decoding_enabled=False,
    use_spec_decoding=False,
    is_spec_dec_tree=False,
    spec_decoding_generation_lengths=None,
    spec_decoding_position_offsets=None,
    spec_decoding_packed_mask=None,
    spec_decoding_bl_tree_mask_offset=None,
    spec_decoding_bl_tree_mask=None,
    spec_bl_tree_first_sparse_mask_offset_kv=None,
    sparse_kv_indices=None,
    sparse_kv_offsets=None,
    sparse_attn_indices=None,
    sparse_attn_offsets=None,
    sparse_attn_indices_block_size=0,
    mrope_rotary_cos_sin=None,
    mrope_position_deltas=None,
    helix_position_offsets=None,
    helix_is_inactive_rank=None,
    is_cross=False,
    cross_kv=None,
    relative_attention_bias=None,
    relative_attention_max_distance=0,
    kv_norm_weight=None,
    kv_norm_eps=1e-6,
    skip_correction_threshold=0.0,
)

#: Head geometries the encoder-only mode is certified at: MHA, `head_size` 72 or
#: 128, context rows only. `head_size` 72 is context-phase only ("unsupported
#: head_size 72" in generation).
_VISION_CERTIFIED_HEADS = {(16, 16, 72), (16, 16, 128)}

#: Head widths `ops/attention/vision_rope_2d` is certified at.
_VISION_ROPE_HEAD_DIMS = (40, 64, 72, 128)

#: The most patch rows one tower call carries, the ceiling the vision ops' cells reach:
#: 8192 placeholder tokens x `spatial_merge_size`**2 patch rows each. `VisionTower` splits
#: a request's frames into calls under it, so the bound a request meets is per frame: one
#: frame's `h * w` patches.
_VISION_CALL_MAX_PATCHES = 32768

# NVFP4 block size: one e4m3 scale per 16 contiguous elements along K.
_SF_VEC = _weights.SF_VEC
# Routing happens outside (renorm_moe_routing_op), so the runner's routing method is
# inert, but it must be 1 (Renormalize): the tier table covers (512 experts, 16
# top-k) only for it, and a tier miss is a logged error with no exception and no
# kernel launch.
_ROUTING_METHOD_RENORM = 1
_ACT_TYPE_SWIGLU = 0

# The trunk's routed-expert backend by the requested `moe_config.backend`
# (`ModelConfig.moe_backend_requested`; `moe_backend` itself is already resolved,
# AUTO -> CUTLASS). Any other request is refused at construction. The MTP draft
# layer's FP8 block-scale experts run on trtllm-gen whatever this says.
_TRUNK_MOE_BACKENDS = {"AUTO": "TRTLLM", "TRTLLM": "TRTLLM", "CUTEDSL": "CUTEDSL"}
# The CuteDSL grouped GEMMs' MMA tile M, which is also the tile `moe_sort` pads each
# expert's rows to: the GEMMs walk the sort's tile tables. sm_103 admits 128 and 256.
_CUTEDSL_TILE_SIZE = 128
# The longest MTP draft the CuteDSL pipeline is certified at: its cells capture up to 128 rows
# (32 requests of 4 verify rows).
_CUTEDSL_MAX_DRAFT_LEN = 3

# Skip sentinel the conv ops and paged short conv compare `cache_indices` against
# (the runtime's mamba `PAD_SLOT_ID`). Never a substitution: the scheduler's slot
# vector gives every row a real slot, and `ops/ple/ngram_token_history` reads its
# carry via `index_select`, so a -1 there is a device-side assert that poisons the
# process.
_PAD_SLOT_ID = -1

# Gated-DeltaNet gating form: the only pair `fused_sigmoid_gating_delta_rule_update`
# is certified at; matches `gdn_prefill_conv_post`'s default so prefill and decode
# compute the same `g`.
_SOFTPLUS_BETA = 1.0
_SOFTPLUS_THRESHOLD = 20.0

# Gated-DeltaNet Q/K L2-norm epsilon: a literal, not `rms_norm_eps`.
_L2_NORM_EPS = 1e-6

# PLE gate magnitude floor before its signed square root (a model constant).
_PLE_CLAMP_MIN = 1e-6

# --------------------------------------------------------------- host offload
#
# Where the PLE n-gram table lives (module docstring, *Host offload*). No engine
# arg owns this, hence an env var, read once at construction (the table is
# allocated once, during `load_weights`). Default on: 47.68 GiB otherwise comes
# out of the KV pool.
_PLE_HOST_OFFLOAD_ENV = "TRTLLM_MODELING_V2_PLE_HOST_OFFLOAD"

#: Spellings of "off"; anything else, including empty, leaves offload on.
_PLE_HOST_OFFLOAD_OFF = frozenset({"0", "false", "no", "off"})


def _ple_host_offload_enabled() -> bool:
    """Whether the n-gram table is allocated in host memory."""
    return os.environ.get(_PLE_HOST_OFFLOAD_ENV, "1").strip().lower() not in _PLE_HOST_OFFLOAD_OFF


# Whether trtllm-gen keeps the second, CuteDSL-layout copy of the routed experts for the
# launches of at least `_CUTE_DSL_MOE_MIN_ROWS` rows (module docstring). No engine arg owns
# it, hence an env var read at construction; "off" (the spellings above) drops the copy and
# every launch runs on trtllm-gen.
_MOE_PREFILL_COPY_ENV = "TRTLLM_MODELING_V2_MOE_PREFILL_COPY"


def _moe_prefill_copy(trunk_moe_backend: str, max_num_tokens: int) -> bool:
    """Whether the trunk keeps the second expert copy: only under trtllm-gen, only when a
    step's rows (at most `max_num_tokens`) can reach the CuteDSL dispatch threshold, and not
    when the env var turns it off."""
    return (
        trunk_moe_backend == "TRTLLM"
        and max_num_tokens >= _CUTE_DSL_MOE_MIN_ROWS
        and os.environ.get(_MOE_PREFILL_COPY_ENV, "1").strip().lower() not in _PLE_HOST_OFFLOAD_OFF
    )


# QSA's fixed constants (backend `constants.py`): one index KV head, three position
# axes, index head exactly twice the rotary width (128 == 2 * 64).
_QSA_INDEX_KV_HEADS = 1
_QSA_POSITION_AXES = 3
_QSA_INDEX_HEAD_TO_ROTARY_RATIO = 2

# The QSA index plane takes coordinate 0 alone, a deliberate departure from the
# checkpoint. `qsa_prefill_compress` (key side, `mrope_section`) and
# `fused_qk_norm_rope` on `index_scratch` (query side, `use_mrope`) must agree:
# both read the three coordinates or neither, else scores mix two position
# encodings. Both-on is certified (`mrope_section` (11, 11, 10)); None is a
# modeling choice. No cost on text; on an image or video batch the index plane sees
# a frame at one position, changing block selection only above
# `dense_seq_len_threshold`.
# To flip, set the config's `(s0, s1, s2)` and `_QSA_INDEX_USE_MROPE = True`.
_QSA_MROPE_SECTION = None
_QSA_INDEX_USE_MROPE = False

# ---------------------------------------------------------------- MTP ------

#: The checkpoint's MTP module count (`mtp_num_hidden_layers`), replayed
#: `max_draft_len` times; NOT the draft length.
_MTP_NUM_LAYERS = 1

#: How the `[T, hidden]` embedding branch enters the `[T, hc_count * hidden]` MTP
#: stream; nothing in the checkpoint or config says. True: added into all four
#: streams; False: stream 0 only. A wrong choice yields bit-correct text with every
#: draft rejected; only the acceptance gate can tell.
_MTP_EMBED_BLOCK_ALL_STREAMS = True

#: Decode batches up to this many rows run each trunk hyper-connection site as
#: `qwen4_exp_hc_decode_first_cute` + `qwen4_exp_hc_decode_mix_cute` (or its NVFP4
#: variant at an MoE site; two GEMV launches); larger ones as `qwen4_exp_hc_inject_norm`
#: and the mixer. A dispatch threshold from the GB300 micro-bench (the GEMVs win at 1
#: row and lose from 2), at most the entries' MAX_ROWS.
_HC_DECODE_GEMV_MAX_ROWS = 1  # capacity: fixed: dispatch threshold, not a buffer bound

#: Decode row counts up to which a hyper-connection mixer (so from
#: `_HC_DECODE_GEMV_MAX_ROWS + 1` rows) runs its first half (projection and
#: activations) as the one tcgen05 launch `qwen4_exp_hc_decode_tc_first` instead of
#: cuBLAS split-K (+ its reduce) + `qwen4_exp_hc_lowrank_act`: the entry's MAX_ROWS
#: (one 128-row MMA tile); larger batches keep cuBLAS.
_HC_TC_FIRST_MAX_ROWS = (
    128  # capacity: fixed: dispatch threshold (one MMA tile), not a buffer bound
)

#: FP8 block-scale block edge, on both weight-scale axes and the activation row.
_FP8_BLOCK = _weights.FP8_BLOCK
#: Values per e4m3 block scale in an NVFP4 KV pool.
_NVFP4_BLOCK = 16

#: False: plain `amax / 448` scale (the runner's recipe). True: power-of-two
#: (UE8M0), a different quantization of the same input.
_MTP_MOE_USE_UE8M0 = False

#: The history depth of the GDN cached-replay verify path (the manager's
#: `replay_history_size`). The replay entry is certified at 16 only, and a manager that
#: reports another depth is refused at bind.
_GDN_REPLAY_HISTORY = 16

#: The most query rows a verify step carries per generation request, `max_draft_len + 1`.
#: It reaches `thop_attention` as `predicted_tokens_per_seq` and the cached replay as its step
#: width `T`; the replay kernel asserts `T <= 8`, and above it the engine would verify with the
#: legacy per-step states, which this target refuses past `_MAX_LEGACY_VERIFY_WIDTH` rows.
_MAX_VERIFY_WIDTH = 8

#: The same bound on the legacy verify path (`TRTLLM_USE_GDN_REPLAY=0`): the entries only it
#: calls (`pack_gdn_decode_qkv`, `fused_gdn_gating`, the multi-token
#: `fused_recurrent_gated_delta_rule_update`) are certified at up to 4 rows per request, and its
#: per-step recurrent buffer holds one state per verify row, request and layer (6.75 GiB at 8
#: rows and 8 requests in fp32).
_MAX_LEGACY_VERIFY_WIDTH = 4

#: The recurrent-state pool dtypes this target serves (`kv_cache_config.mamba_ssm_cache_dtype`):
#: the checkpoint's fp32, or bf16, which rounds the state to bf16 once per decode token, once
#: per prefill call and at every cached-replay checkpoint commit. The engine also accepts fp16.
_SSM_STATE_DTYPES = (torch.float32, torch.bfloat16)


# ----------------------------------------------- KV cache manager binding ---
#
# The ops are plain functions over raw tensors, and the pool views taken from the
# KV cache manager are `TensorWrapper`s over raw device addresses that do not keep
# memory alive, so a view is only valid for the manager that produced it. `_Bound`
# is everything derived from one manager; the core replaces it as a unit
# (`_bind_cache_manager`) and the hot path pays one identity comparison. The MTP
# draft layer's step can see a second manager (the separate draft KV cache), bound
# the same way into `_DraftBound` (`_bind_draft_cache_manager`).
#
# The `check_*` functions hold the shape/dtype contract. They read only
# `.shape` / `.dtype`, so unit tests run them on meta tensors with no engine.
#
# For a later target with pools that are not in `attn_metadata` (mamba/GDN state,
# side caches): the reusable parts are (1) one immutable bound object per manager,
# replaced atomically and keyed on manager identity, (2) pure contract checks,
# (3) refusing a bind during CUDA graph capture that no eager step validated first
# (a re-bind for the trunk's, a first bind for the draft layer's). Once a second
# target needs it, the next step is a declarative
# `PoolSpec(name, layers, getter, tail_shape, dtype)` list consumed by one shared
# binder under `modeling_v2/` (not `catalog/`), leaving each target only its specs
# and shape constants. Pure-attention targets that read the pool pointers from
# `attn_metadata` on every call (deepseek_v3) do not need this.


class _PoolGeometry(NamedTuple):
    """The pool shapes this target expects, taken from its config and the metadata."""

    tokens_per_block: int
    heads_kv: int
    head_dim: int
    index_kv_heads: int
    index_head_dim: int
    conv_dim: int
    conv_kernel: int
    num_k_heads: int
    num_v_heads: int
    head_v_dim: int
    head_k_dim: int
    stream: int
    ple_state_len: int
    ngram_context_len: int
    dtype: torch.dtype  # the activation dtype
    kv_dtype: torch.dtype  # the paged K/V pool's dtype: bf16, e4m3 (fp8) or uint8 (NVFP4)
    kv_width: int  # its innermost extent: head_dim, or head_dim // 2 for NVFP4's e2m1 pairs


class _KvPlanes(NamedTuple):
    """One layer's paged K/V views, and under an NVFP4 pool its e4m3 block-scale views."""

    k: torch.Tensor
    v: torch.Tensor
    k_scale: torch.Tensor | None = None
    v_scale: torch.Tensor | None = None


class _Replay(NamedTuple):
    """The GDN cached-replay verify state of one manager: per slot, the recurrent pool
    row is a checkpoint and a compact history holds the tokens accepted since it."""

    old: dict  # GDN layer -> (old_x, old_B, old_dt, old_dA_cumsum) history buffers
    cache_buf_idx: torch.Tensor  # per slot: which history half is pending
    prev_num_accepted: torch.Tensor  # per slot: pending history entries
    history: int
    width: int  # verify rows per generation request
    all_layer_commit: bool  # the manager commits partitioned steps over every layer


class _Bound(NamedTuple):
    """Everything derived from one KV cache manager."""

    cm: object
    pool_idx: dict  # attention layer -> pool offset (what thop_attention addresses)
    index_cache: dict  # attention layer -> QSA index plane
    position_cache: torch.Tensor  # shared QSA position plane
    kv: dict  # attention layer -> `_KvPlanes`
    conv: dict  # GDN layer -> conv state pool
    ssm: dict  # GDN layer -> recurrent state pool
    ple_state: tuple  # (PLE conv state, PLE n-gram id history)
    # MTP only: the verify step's per-step conv buffers, and one recurrent mode. Legacy:
    # per-step recurrent buffers (`int_ssm`). Cached replay: `replay`, no `int_ssm`.
    int_idx: torch.Tensor | None = None
    int_conv: dict | None = None
    int_ssm: dict | None = None
    replay: _Replay | None = None


class _DraftBound(NamedTuple):
    """The MTP draft layer's views of the manager its draft step sees: the separate
    draft manager when the engine built one, else the trunk's."""

    cm: object
    index_cache: torch.Tensor  # the draft layer's QSA index plane
    position_cache: torch.Tensor  # that manager's QSA position plane
    kv: _KvPlanes  # the draft layer's paged K/V views
    share_slot: int | None  # its row in the metadata's shared selection buffers (IndexShare)


def _require(cm, names, what: str) -> None:
    for name in names:
        assert hasattr(cm, name), (
            f"the KV cache manager {type(cm).__name__} does not expose {name}; this target {what}"
        )


def _check_pool(t, what, lead, tail, dtype=None) -> None:
    """`t` has `lead` leading dims of any size, then exactly `tail`, at `dtype`."""
    assert t is not None, f"{what}: the runtime did not allocate it"
    expected = ["*"] * lead + [str(n) for n in tail]
    assert t.dim() == lead + len(tail) and tuple(t.shape[lead:]) == tuple(tail), (
        f"{what} is {tuple(t.shape)}, expected [{', '.join(expected)}]"
    )
    assert dtype is None or t.dtype == dtype, f"{what} is {t.dtype}, expected {dtype}"


def check_position_plane(g: _PoolGeometry, position) -> None:
    shape = (g.tokens_per_block, _QSA_POSITION_AXES)
    _check_pool(position, "QSA position plane", 1, shape, torch.int32)


def check_attention_pools(g: _PoolGeometry, layer: int, index_cache, pool) -> None:
    """QSA index plane and paged K/V for one attention layer."""
    index_shape = (g.tokens_per_block, g.index_kv_heads, g.index_head_dim)
    _check_pool(index_cache, f"layer {layer}: QSA index plane", 1, index_shape, g.dtype)
    # The sparse path needs adjacent HND K/V role planes: [pages, 2, H, block, D].
    kv_shape = (g.heads_kv, g.tokens_per_block, g.kv_width)
    _check_pool(pool, f"layer {layer}: paged K/V", 2, kv_shape, g.kv_dtype)
    assert pool.shape[1] == 2, f"layer {layer}: paged K/V has {pool.shape[1]} role planes, not 2"


def check_kv_scale_planes(g: _PoolGeometry, layer: int, pool, scales) -> None:
    """An NVFP4 layer's e4m3 block scales: one uint8 per `_NVFP4_BLOCK` values, laid out like
    the data pages (`[slots, 2, H, block, D / 16]`), and holding as many pages per slot as the
    data pool does.

    The attention op addresses a layer's data pages and its scale pages with one block-offset
    table, so a scale pool whose pages-per-slot differs from the data pool's reads and writes
    other slots' scales without any error. Both counts are read off the views' slot strides."""
    scale_shape = (g.heads_kv, g.tokens_per_block, g.head_dim // _NVFP4_BLOCK)
    _check_pool(scales, f"layer {layer}: NVFP4 block scales", 2, scale_shape, torch.uint8)
    assert scales.shape[1] == 2 and scales.shape[0] == pool.shape[0], (
        f"layer {layer}: the block scales are {tuple(scales.shape)} against paged K/V "
        f"{tuple(pool.shape)}; they must hold the same slots, K then V"
    )
    pages = []
    for what, t in (("paged K/V", pool), ("block scales", scales)):
        page = t[0, 0].numel()
        assert t.stride(0) % page == 0, (
            f"layer {layer}: the {what} slot stride {t.stride(0)} is not a whole number of "
            f"{page}-element pages"
        )
        pages.append(t.stride(0) // page)
    assert pages[0] == pages[1], (
        f"layer {layer}: the paged K/V pool holds {pages[0]} pages per slot and the block-scale "
        f"pool {pages[1]}; the attention op would address the scales of other slots"
    )


def check_recurrent_pools(g: _PoolGeometry, conv, ssm) -> None:
    _check_pool(conv, "conv state pool", 1, (g.conv_dim, g.conv_kernel - 1), g.dtype)
    # Key dim innermost.
    _check_pool(ssm, "recurrent state pool", 1, (g.num_v_heads, g.head_v_dim, g.head_k_dim))
    assert ssm.dtype in _SSM_STATE_DTYPES, (
        f"the recurrent state pool is {ssm.dtype}; this target serves "
        f"{' or '.join(str(d) for d in _SSM_STATE_DTYPES)} (kv_cache_config.mamba_ssm_cache_dtype)"
    )


def check_ple_pools(g: _PoolGeometry, ple_conv, ple_ngram, conv) -> None:
    _check_pool(ple_conv, "PLE conv state", 1, (g.stream, g.ple_state_len), g.dtype)
    # The id-history entry writes int64 into it; an int32 pool raises.
    _check_pool(ple_ngram, "PLE n-gram context pool", 1, (g.ngram_context_len,), torch.int64)
    # The forward addresses all three with one scheduler slot vector.
    assert ple_conv.shape[0] == conv.shape[0] == ple_ngram.shape[0], (
        "the PLE and Gated-DeltaNet pools must share one slot space"
    )


def check_step_buffers(g: _PoolGeometry, int_conv, conv, int_idx) -> None:
    """MTP verify buffers both recurrent modes use: the per-step conv windows
    `[slots, steps, ...]` at the pool's dtype (the promotion copies between them without
    converting) and the rows `update_mamba_states` reads its sources from."""
    assert int_conv is not None, (
        "the runtime allocated no per-step conv buffer; a partially accepted draft "
        "cannot be rolled back"
    )
    _check_pool(int_conv, "per-step conv buffer", 2, (g.conv_dim, g.conv_kernel - 1), conv.dtype)
    assert int_idx is not None and int_idx.dtype == torch.int32, (
        "the cache manager exposes no int32 intermediate_state_indices"
    )


def check_legacy_state(g: _PoolGeometry, int_ssm, ssm) -> None:
    """Legacy verify: per-step recurrent buffer `[slots, steps, ...]` at the pool's dtype."""
    assert int_ssm is not None, (
        "the cache manager runs the legacy verify path but allocated no per-step recurrent "
        "buffer; a partially accepted draft cannot be rolled back"
    )
    ssm_shape = (g.num_v_heads, g.head_v_dim, g.head_k_dim)
    _check_pool(int_ssm, "per-step recurrent buffer", 2, ssm_shape, ssm.dtype)
    assert int_ssm.shape[1] <= _MAX_LEGACY_VERIFY_WIDTH, (
        f"the legacy verify path runs {int_ssm.shape[1]} rows per generation request; its entries "
        f"are certified at up to {_MAX_LEGACY_VERIFY_WIDTH} (max_draft_len "
        f"{_MAX_LEGACY_VERIFY_WIDTH - 1}). Leave TRTLLM_USE_GDN_REPLAY unset (cached replay) for "
        "a longer draft"
    )


def check_replay_state(history, width, prev_accepted, cache_buf_idx, ssm) -> None:
    """Cached replay, manager-wide: the geometry the replay entry is certified at, and the
    per-slot bookkeeping, which must cover every pool slot."""
    assert ssm.dtype in _SSM_STATE_DTYPES, (
        f"the recurrent pool is {ssm.dtype}; the replay entry is certified at an fp32 or a "
        "bf16 checkpoint"
    )
    assert history == _GDN_REPLAY_HISTORY, (
        f"replay_history_size {history}; the replay entry is certified at {_GDN_REPLAY_HISTORY}"
    )
    assert 2 <= width <= min(_MAX_VERIFY_WIDTH, history), (
        f"replay_step_width {width}; the replay entry is certified at 2..{_MAX_VERIFY_WIDTH} "
        f"verify rows and never above the history ({history})"
    )
    for name, t in (("prev_num_accepted_tokens", prev_accepted), ("cache_buf_idx", cache_buf_idx)):
        assert t.dim() == 1 and t.shape[0] >= ssm.shape[0] and t.dtype == torch.int32, (
            f"{name} is {tuple(t.shape)} {t.dtype}; the kernel reads one int32 per pool slot "
            f"({ssm.shape[0]})"
        )


def check_replay_history(
    g: _PoolGeometry, layer: int, history, pool_slots, old_x, old_B, old_dt
) -> None:
    """One GDN layer's replay history: `[slots, 2 halves, ...]`, addressed by `history` and
    the head geometry, never by the buffer's own shape."""
    for name, t in (("old_x", old_x), ("old_B", old_B), ("old_dt", old_dt)):
        assert t is not None, f"layer {layer}: the manager's layer cache carries no {name}"
    assert old_x.shape[0] >= pool_slots, (
        f"layer {layer}: the replay history has {old_x.shape[0]} slots for a {pool_slots}-slot pool"
    )
    bf16, fp32 = torch.bfloat16, torch.float32
    x_tail, b_tail = (history, g.num_v_heads, g.head_v_dim), (history, g.num_k_heads, g.head_k_dim)
    _check_pool(old_x, f"layer {layer}: old_x", 2, x_tail, bf16)
    _check_pool(old_B, f"layer {layer}: old_B", 2, b_tail, bf16)
    _check_pool(old_dt, f"layer {layer}: old_dt", 2, (g.num_v_heads, history), fp32)
    assert old_x.shape[1] == old_B.shape[1] == old_dt.shape[1] == 2, (
        f"layer {layer}: the replay history is not double-buffered"
    )


def check_verify_width(rows: int) -> None:
    """The rows per generation request one step carries: `max_draft_len + 1` (`runtime_draft_len
    + 1` at run time), a `thop_attention` `predicted_tokens_per_seq` the entry is certified at."""
    assert 1 <= rows <= _MAX_VERIFY_WIDTH, (
        f"predicted_tokens_per_seq {rows} is outside the 1..{_MAX_VERIFY_WIDTH} range "
        "attention/thop_attention is certified at"
    )


def check_replay_width(replay, rows: int) -> None:
    """Under the cached replay a single-row generation step cannot take the ordinary
    recurrence: the pool row is a checkpoint whose pending history only the replay kernel
    applies. The manager's step width is the one row count per sequence a step may carry."""
    assert replay is None or rows == replay.width, (
        f"the batch carries {rows} query rows per generation sequence and the GDN cached "
        f"replay runs at step width {replay.width}; a runtime_draft_len of {rows - 1} (a "
        "draft-length schedule, or skipped drafting) cannot run under it. Export "
        "TRTLLM_USE_GDN_REPLAY=0 for variable draft lengths"
    )


def check_shared_selection(shared, slots, layer: int, block_topk: int) -> None:
    """IndexShare: the metadata's shared selection buffers have a row for `layer` and are
    `block_topk` wide."""
    assert shared is not None, "IndexShare is on but the QSA metadata allocated no shared buffers"
    assert layer in slots and slots[layer] < shared.shape[0], (
        f"the KV cache manager gives layer {layer} no row of the {shared.shape[0]}-row shared "
        f"selection buffers (qsa_shared_topk_slots={slots})"
    )
    assert shared.shape[2] == block_topk, (
        f"the shared selection rows are {shared.shape[2]} wide and this target selects "
        f"{block_topk} blocks per row"
    )


def _moe_layer_scalars(w, i: int, backend: str) -> dict[str, torch.Tensor]:
    """Layer `i`'s routed-expert scalars for `backend`, built from the checkpoint's
    `input_scale` / `weight_scale_2` rows.

    The checkpoint stores modelopt reciprocals (`input_scale = 1 / g_act`,
    `weight_scale_2 = 1 / g_w`), so a quantizer's global scale is `1 / input_scale` and
    the runner's scalars are products of the two.
    """
    isc1, isc1_up = w[f"l{i}_e_isc1"], w[f"l{i}_e_isc1_up"]
    assert torch.equal(isc1, isc1_up), (
        f"layer {i}: expert gate/up input_scale differ; the fused FC1 "
        "reads one activation quantization for both halves"
    )
    assert torch.equal(w[f"l{i}_e_ws2_1"], w[f"l{i}_e_ws2_1_up"]), (
        f"layer {i}: expert gate/up weight_scale_2 differ; the fused "
        "FC1 carries one alpha per expert for both halves"
    )
    # One activation quantization feeds all 512 experts, so FC1's activation
    # global scale must be one number (the shared expert is bf16, no scale).
    assert torch.equal(isc1.min(), isc1.max()), (
        f"layer {i}: the routed experts do not share one "
        "gate_proj.input_scale, so a single activation quantization "
        "cannot feed them all and this assembly is wrong for this "
        "checkpoint"
    )
    e_isc2, ws2_2 = w[f"l{i}_e_isc2"], w[f"l{i}_e_ws2_2"]
    gate1 = (isc1 * w[f"l{i}_e_ws2_1"]).contiguous()
    # Routed FC1 activation global scale (one per layer).
    scalars = dict(global_scale=(1.0 / isc1[:1]).contiguous())
    if backend == "CUTEDSL":
        # The CuteDSL FC1 epilogue requantizes with one global scale for the whole layer,
        # where trtllm-gen folds each expert's `down_proj.input_scale` into its FC1
        # scalar. Quantizing at `1 / max(input_scale)` (the smallest per-expert global
        # scale, so no block scale saturates) and putting the same factor into FC2's alpha
        # makes `global_sf * alpha2[e] == ws2_2[e]` for every expert, whatever the spread.
        isc2_max = e_isc2.max()
        scalars.update(
            # [num_experts] each: activation scale times weight scale.
            alpha1=gate1,
            alpha2=(isc2_max * ws2_2).contiguous(),
            # The FC2 activation's quantization scale, numel 1.
            global_sf=(1.0 / isc2_max).reshape(1).contiguous(),
        )
    else:
        scalars.update(
            # [num_experts] each. Swapping the first two is finite and silent.
            output1_scale_scalar=(gate1 / e_isc2).contiguous(),
            output1_scale_gate_scalar=gate1,
            output2_scale_scalar=(e_isc2 * ws2_2).contiguous(),
        )
    return scalars


class Qwen4ExpTp1ModelingV2(DecoderModel):
    def __init__(self, model_config: ModelConfig, composite_config: PretrainedConfig):
        """`composite_config` is passed in because the text sub-config adopted as
        `model_config.pretrained_config` carries neither the vision half nor the
        media token ids."""
        super().__init__(model_config)
        cfg = model_config.pretrained_config
        assert cfg is not None

        # This target IS this geometry — assert, never adapt.
        assert torch.cuda.get_device_capability() == _SM, (
            f"target certified on sm_{_SM[0]}{_SM[1]}, running on "
            f"sm_{''.join(map(str, torch.cuda.get_device_capability()))}"
        )
        assert model_config.mapping.tp_size == 1, "tp1 target"
        assert model_config.mapping.pp_size == 1, "tp1 target"
        assert not model_config.mapping.enable_attention_dp, "tp1 target"

        # `cfg` is the engine-normalized Qwen4ExpTextConfig (pool geometry is read
        # from it).
        assert getattr(cfg, "model_type", None) == "qwen4_exp_text", (
            f"the core must be built on the normalized text config; got "
            f"model_type {getattr(cfg, 'model_type', None)!r}"
        )

        dt = model_config.torch_dtype
        assert dt == torch.bfloat16, f"bf16 target, engine resolved {dt}"
        assert cfg.torch_dtype == torch.bfloat16, cfg.torch_dtype
        assert not cfg.tie_word_embeddings, "untied lm_head"
        assert not cfg.attention_bias, "no q/k/v/o bias anywhere"
        assert cfg.hidden_act == "silu", "SwiGLU over a silu gate"

        self.num_layers = cfg.num_hidden_layers
        self.hidden = cfg.hidden_size
        self.vocab = cfg.vocab_size
        self.eps = cfg.rms_norm_eps

        # `layer_types` is authoritative; `full_attention_interval` is the runtime
        # helper's fallback, so both must agree.
        layer_types = list(cfg.layer_types)
        assert len(layer_types) == self.num_layers, (len(layer_types), self.num_layers)
        assert set(layer_types) <= {"full_attention", "linear_attention"}, set(layer_types)
        self.is_attn = [t == "full_attention" for t in layer_types]
        interval = cfg.full_attention_interval
        assert self.is_attn == [i % interval == interval - 1 for i in range(self.num_layers)], (
            "layer_types disagrees with full_attention_interval"
        )
        self.attn_layers = [i for i, a in enumerate(self.is_attn) if a]
        self.gdn_layers = [i for i, a in enumerate(self.is_attn) if not a]
        assert self.attn_layers and self.gdn_layers, "this target is the hybrid assembly"

        # The checkpoint picks the speculative mode (`MTP_EAGLE_ONE_MODEL`, one module
        # replayed `max_draft_len` times); the module is built only when the engine
        # asked for speculation.
        spec_config = getattr(model_config, "spec_config", None)
        self.mtp_enabled = spec_config is not None
        declared = getattr(cfg, "mtp_num_hidden_layers", None)
        assert declared in (None, _MTP_NUM_LAYERS), (
            f"this target implements the one-layer MTP module this checkpoint "
            f"ships; the text config declares mtp_num_hidden_layers {declared}"
        )
        # The module shares the trunk's embedding and the shell's lm_head.
        assert not getattr(cfg, "mtp_use_dedicated_embeddings", False), (
            "this checkpoint's MTP module shares the decoder's embedding "
            "(`mtp_use_dedicated_embeddings: false`) and ships no "
            "`mtp.embed_tokens` tensor; a dedicated-embedding module is a "
            "different assembly"
        )
        # The nested `mtp` block, when kept: one `full_attention` layer at the trunk's
        # rope base.
        mtp_block = getattr(cfg, "mtp", None)
        self.mtp_theta = None
        if mtp_block is not None:
            assert isinstance(mtp_block, dict), (
                f"the text config's `mtp` block is {type(mtp_block).__name__}; "
                "this target reads it as the published mapping"
            )
            assert mtp_block.get("num_hidden_layers", _MTP_NUM_LAYERS) == _MTP_NUM_LAYERS, (
                f"mtp.num_hidden_layers is {mtp_block.get('num_hidden_layers')}"
            )
            assert list(mtp_block.get("layer_types", ["full_attention"])) == ["full_attention"], (
                f"the MTP module's layer_types is {mtp_block.get('layer_types')}; this "
                "target implements the full-attention form, which is the only "
                "one whose weights the checkpoint carries"
            )
            # `null` means the trunk's own output.
            assert mtp_block.get("mtp_use_hidden_state_from_layer") is None, (
                f"the MTP module asks for the hidden state of layer "
                f"{mtp_block.get('mtp_use_hidden_state_from_layer')}; this "
                "target hands it the end of the stack, which is what the "
                "runtime passes and all this file can reach"
            )
            if "rope_theta" in mtp_block:
                self.mtp_theta = float(mtp_block["rope_theta"])

        # ---- hyper-connections ------------------------------------------
        self.hc_count = cfg.hc_count
        self.hc_lowrank = cfg.hc_lowrank
        assert self.hc_count >= 2, (
            f"the hyper-connection residual carries hc_count streams; this "
            f"config declares {self.hc_count}, which is not a multi-stream "
            f"residual and is a different assembly"
        )
        self.stream = self.hc_count * self.hidden
        # Columns of the padded first-projection weight (`_fuse_hc_first_projections`)
        # the cuBLAS first projection multiplies: the real `[down; block_inject]` rows
        # rounded up to 32 (352 here). GB300, fp32 out, K 10240: 16384 rows 81.5 us
        # against 91.5 at the padded 384, 8193 rows 41.6 against 51.2, never slower
        # below; the real columns are bitwise the 384-wide product's at every row
        # count measured (1-16384; at 320 columns they are not).
        self.hc_first_cols = -(-(self.hc_lowrank + self.hc_count) // 32) * 32

        # ---- full attention ---------------------------------------------
        self.heads_q = cfg.num_attention_heads
        self.heads_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        assert self.head_dim in (64, 128, 256), "fused_qk_norm_rope head_dim set"
        assert self.heads_q % self.heads_kv == 0, "thop_attention GQA rule"
        self.q_width = self.heads_q * self.head_dim
        self.kv_width = self.heads_kv * self.head_dim
        self.qkv_width = self.q_width + 2 * self.kv_width
        # The per-head interleaved output gate makes q_proj twice as wide as q. No
        # config flag exists, so it is derived from the weight (asserted in weights.py);
        # the per-head split below would otherwise address the wrong columns.
        assert not hasattr(cfg, "attn_output_gate"), (
            "this config gained an attn_output_gate flag; the gate is derived "
            "from q_proj's width here, and the two must be reconciled"
        )

        # ---- RoPE --------------------------------------------------------
        # Partial, neox half-split pairs, no scaling; `rope_theta` is inside
        # `rope_parameters` (mirrored as `rope_scaling`).
        rope = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None)
        assert isinstance(rope, dict), f"rope parameters must be a dict, got {rope!r}"
        self.theta = float(rope["rope_theta"])
        assert self.theta > 0.0, "rope theta"
        partial = float(rope.get("partial_rotary_factor", cfg.partial_rotary_factor))
        self.rotary_dim = int(self.head_dim * partial)
        assert self.rotary_dim % 2 == 0 and 0 < self.rotary_dim <= self.head_dim, (
            f"rotary_dim {self.rotary_dim} must be even and at most head_dim "
            f"{self.head_dim}; the op enforces only the evenness half"
        )
        # The mRoPE declaration is what makes the engine hand this forward a
        # [3, 1, T] position buffer.
        self.mrope_declared = str(rope.get("type", rope.get("rope_type"))) == "mrope"
        # `fused_qk_norm_rope` implements the interleaved mRoPE layout only (pair `j`
        # reads coordinate 1 when `j % 3 == 1 and j < 3*s1`, 2 when `j % 3 == 2 and
        # j < 3*s2`, else 0); the contiguous variant is a different operator. A text
        # gate cannot see the split.
        assert rope.get("mrope_interleaved") is True, (
            f"this target passes the interleaved mRoPE layout, which is the only "
            f"one attention/fused_qk_norm_rope implements; the config declares "
            f"mrope_interleaved={rope.get('mrope_interleaved')!r}"
        )
        section = list(rope["mrope_section"])
        assert len(section) == _QSA_POSITION_AXES, (
            f"mrope_section must carry one entry per position coordinate; got {section}"
        )
        rotary_pairs = self.rotary_dim // 2
        assert sum(section) == rotary_pairs, (
            f"mrope_section {section} sums to {sum(section)} but rotary_dim "
            f"{self.rotary_dim} makes {rotary_pairs} rotation pairs; the split "
            f"partitions pairs, not dimensions"
        )
        self.mrope_section1 = int(section[1])
        self.mrope_section2 = int(section[2])
        # Row-1 section is load-bearing here; row-2 saturates (10, 11, 12 are
        # bit-identical at 32 pairs). `use_gemma` with `use_mrope` is certified only
        # at the geometry asserted below.
        assert (
            self.heads_q,
            self.heads_kv,
            self.head_dim,
            self.rotary_dim,
            self.theta,
        ) == (24, 2, 256, 64, 1e7), (
            f"the main attention rope runs use_gemma together with use_mrope, "
            f"which attention/fused_qk_norm_rope certifies only at "
            f"(24, 2, 256, 64, 1e7); this config gives "
            f"{(self.heads_q, self.heads_kv, self.head_dim, self.rotary_dim, self.theta)}"
        )
        # No YaRN or other scaling.
        assert "factor" not in rope, (
            f"this target builds a plain (unscaled) rope; rope carries {rope!r}"
        )
        self.max_positions = cfg.max_position_embeddings

        # ---- QSA sparse index --------------------------------------------
        self.index_heads = cfg.indexer_n_heads
        self.index_kv_heads = cfg.indexer_kv_heads
        self.index_head_dim = cfg.indexer_head_dim
        self.token_topk = cfg.indexer_budget
        self.compress_ratio = cfg.indexer_compress_ratio
        assert self.index_kv_heads == _QSA_INDEX_KV_HEADS, (
            f"QSA requires exactly one index KV head; this config declares {self.index_kv_heads}"
        )
        assert self.compress_ratio >= 2 and self.token_topk % self.compress_ratio == 0, (
            f"QSA selection geometry: token_topk {self.token_topk} must divide "
            f"by compress_ratio {self.compress_ratio}, which must be at least 2"
        )
        # Fused index kernels: both powers of two, index head twice the rotary width.
        assert self.index_head_dim == self.rotary_dim * _QSA_INDEX_HEAD_TO_ROTARY_RATIO, (
            f"the fused QSA index kernels require index_head_dim "
            f"({self.index_head_dim}) == 2 * rotary_dim ({self.rotary_dim})"
        )
        assert self.index_head_dim & (self.index_head_dim - 1) == 0, "index_head_dim power of two"
        assert self.compress_ratio & (self.compress_ratio - 1) == 0, "compress_ratio power of two"
        self.block_topk = self.token_topk // self.compress_ratio
        self.expanded_topk = self.token_topk + self.compress_ratio - 1
        # Dense-attention cutoff: QSA resolves max(token_topk, seq_len_threshold);
        # checked against the engine's value below because the CUDA-graph runner
        # splits its short and long decode families on it.
        self.dense_threshold = self.token_topk

        # ---- the engine's sparse configuration, checked at construction -----
        # The index/position side caches, `qsa_*` metadata and sparse kernels all hang
        # off `sparse_attention_config`; checking here fails at cold start rather than
        # at the first forward.
        sparse_cfg = getattr(model_config, "sparse_attention_config", None)
        assert getattr(sparse_cfg, "algorithm", None) == "qsa", (
            f"this target is a QSA sparse-attention assembly and the engine "
            f"resolved sparse_attention_config={sparse_cfg!r}. QSA is this "
            f"architecture's own model default, so an engine that reaches here "
            f"without it dropped the default on the way in rather than being "
            f"asked for dense attention"
        )
        # The engine's own lowering on this core's config: a mismatch with the side
        # cache sizing would otherwise be a silent mis-index.
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
            f"the engine lowered the QSA index geometry to {qsa_params!r} and "
            f"this target derived heads {self.index_heads}, kv heads "
            f"{self.index_kv_heads}, head_dim {self.index_head_dim}, token_topk "
            f"{self.token_topk}, compress_ratio {self.compress_ratio}; the side "
            f"caches are sized from the engine's copy and the kernels are "
            f"called with this one"
        )
        assert qsa_params.dense_seq_len_threshold == self.dense_threshold, (
            f"the engine's resolved QSA cutoff is "
            f"{qsa_params.dense_seq_len_threshold} and this forward branches at "
            f"{self.dense_threshold}. The runner captures its short decode "
            f"graph family at the engine's number, so a disagreement freezes "
            f"the dense body into the graph that long batches replay — which "
            f"is finite, plausible output. A serving config that raises "
            f"seq_len_threshold above token_topk is the way to get here, and "
            f"it is a configuration this target does not implement"
        )
        # IndexShare for the MTP draft loop (`sparse_attention_config.
        # index_share_for_mtp_iteration`), as the engine resolved it: the same lowering
        # sizes the metadata's shared selection buffers.
        self.mtp_index_share = bool(qsa_params.mtp_index_share)
        self.index_width = (self.index_heads + self.index_kv_heads) * self.index_head_dim
        # Packed [q | k | v] scratch for the fused norm+rope call; the spare KV head
        # is the v slot the op needs but never reads.
        self.index_scratch_width = self.index_width + self.index_head_dim
        # The fused input projection's output row (`weights._attn_in_w`):
        # [q | k | v | gate | index q, k | v slot]. Its leading columns are the
        # packed QKV the main norm+rope rewrites in place, and its tail is the
        # index scratch's layout, so the row is `qkv_width + q_width +
        # index_scratch_width` wide. The norm+rope addresses a row as
        # `(Hq + Hk + Hv) * head_dim` and never touches the v heads, so the
        # row must be a whole number of heads: everything after k is "v" to it.
        self.attn_in_width = self.qkv_width + self.q_width + self.index_scratch_width
        assert self.attn_in_width % self.head_dim == 0, (
            f"the fused projection row ({self.attn_in_width}) is not a whole number of "
            f"{self.head_dim}-wide heads, so the main norm+rope cannot address it in place"
        )
        self.attn_in_tail_heads = self.attn_in_width // self.head_dim - self.heads_q - self.heads_kv
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)

        # ---- gated deltanet ----------------------------------------------
        self.num_k_heads = cfg.linear_num_key_heads
        self.num_v_heads = cfg.linear_num_value_heads
        self.head_k_dim = cfg.linear_key_head_dim
        self.head_v_dim = cfg.linear_value_head_dim
        self.conv_kernel = cfg.linear_conv_kernel_dim
        assert self.num_v_heads % self.num_k_heads == 0, "GDN key-head broadcast"
        assert self.head_k_dim == 128 and self.head_v_dim == 128, (
            f"the delta-rule entries certify head dims 128/128; this config "
            f"declares {self.head_k_dim}/{self.head_v_dim}"
        )
        assert self.conv_kernel in (2, 3, 4), "causal_conv1d width domain"
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        # Column widths of the one input-projection GEMM's output: [q|k|v | z | a | b].
        self.in_proj_split = (self.conv_dim, self.value_dim, self.num_v_heads, self.num_v_heads)
        # Gate activation of the gated RMS norm (`silu` or `sigmoid`).
        gate_type = getattr(cfg, "output_gate_type", None)
        assert gate_type == "sigmoid", (
            f"gate activation {gate_type!r} is not the sigmoid form this "
            f"target passes to rms_norm_gated_token_major"
        )

        # ---- MoE ----------------------------------------------------------
        self.num_experts = cfg.num_experts
        self.topk = cfg.num_experts_per_tok
        self.moe_inter = cfg.moe_intermediate_size
        self.shared_inter = cfg.shared_expert_intermediate_size
        assert cfg.norm_topk_prob, "renorm_moe_routing_op always renormalizes"
        assert self.num_experts <= 512 and self.topk <= 16, (
            f"renorm_moe_routing_op takes at most 512 experts and top-16; this "
            f"config declares {self.num_experts} / {self.topk}"
        )
        # Every layer is MoE; the dense `intermediate_size` is dead config.
        assert getattr(cfg, "decoder_sparse_step", 1) == 1, "every layer must be MoE"
        assert not getattr(cfg, "mlp_only_layers", []), "every layer must be MoE"
        # trtllm-gen NVFP4 runner operand-width rules and fp4_quantize's block rule; they
        # also cover the CuteDSL FC1 / FC2 rules (K and I in 64-element scale groups, the
        # gated N in 128-row pairs of 64 up rows and 64 gate rows).
        assert self.hidden % 256 == 0, "fp4_block_scale_moe_runner hidden alignment"
        assert (2 * self.moe_inter) % 128 == 0, "fp4_block_scale_moe_runner 2I alignment"
        assert (self.hidden // _SF_VEC) % 4 == 0 and (self.moe_inter // _SF_VEC) % 4 == 0, (
            "the 128x4 block-scale swizzle needs both scale widths to be a multiple of 4"
        )
        # flashinfer_silu_and_mul's precondition is on the half width.
        assert (self.shared_inter * 2) % 16 == 0, "flashinfer_silu_and_mul half alignment"
        # Which kernels run the trunk's routed experts; it also fixes the load-time layout
        # of the expert stacks (`weights.py`), so it is decided before anything is declared.
        requested = model_config.moe_backend_requested.upper()
        assert requested in _TRUNK_MOE_BACKENDS, (
            f"moe_config.backend={model_config.moe_backend_requested!r} is not implemented by "
            f"this target; the supported requests are {sorted(_TRUNK_MOE_BACKENDS)}"
        )
        self.trunk_moe_backend = _TRUNK_MOE_BACKENDS[requested]
        self.moe_prefill_copy = _moe_prefill_copy(
            self.trunk_moe_backend, model_config.max_num_tokens
        )

        # ---- PLE -----------------------------------------------------------
        ple_ids = list(getattr(cfg, "ple_layer_ids", None) or [])
        assert len(ple_ids) == 1, (
            f"this target implements the single PLE decoder layer this "
            f"checkpoint ships; the config declares ple_layer_ids={ple_ids}"
        )
        # One-based in the config.
        self.ple_layer = ple_ids[0] - 1
        assert 0 <= self.ple_layer < self.num_layers, self.ple_layer
        assert not self.is_attn[self.ple_layer], (
            f"the PLE lifecycle state rides the recurrent pools, so its layer "
            f"must be a linear-attention layer; layer {self.ple_layer} is not"
        )
        self.ngram_size = cfg.ngram_size
        self.heads_per_ngram = cfg.heads_per_ngram
        self.ngram_heads = self.heads_per_ngram * (self.ngram_size - 1)
        self.ple_embed_dim = cfg.ple_embed_dim
        self.ple_head_dim = self.ple_embed_dim // self.ngram_heads
        assert self.ple_head_dim * self.ngram_heads == self.ple_embed_dim, (
            f"{self.ngram_heads} n-gram heads must tile ple_embed_dim {self.ple_embed_dim}"
        )
        self.ple_conv_kernel = cfg.ple_conv_kernel_size
        # Length of the runtime's paged conv state.
        self.ple_dilation = self.ngram_size
        self.ple_state_len = (self.ple_conv_kernel - 1) * self.ple_dilation
        self.ngram_context_len = self.ngram_size - 1
        # Shard count and padding rule are config; the table height is not, so the
        # table is declared at load time (`weights.py`).
        self.ple_shards = cfg.split_ngram_parts
        self.ngram_vocab_divisor = cfg.make_ngram_vocab_size_divisible_by
        assert self.ple_shards > 0 and self.ngram_vocab_divisor > 0, (
            self.ple_shards,
            self.ngram_vocab_divisor,
        )
        # Fresh-sequence seed for the n-gram id history and the hash's segment token
        # (element 0 when a list, as in the HF reference).
        eos = cfg.eos_token_id
        self.eos_token_id = int(eos[0] if isinstance(eos, (list, tuple)) else eos)

        # ---- the vision tower ------------------------------------------------
        # A 27-block ViT (no KV cache, deepstack or quantization) feeding a 2x2 patch
        # merger. Constants the config omits: `_VISION_ROPE_THETA`,
        # `_VISION_NORM_EPS`, and the merger's erf GELU.
        vcfg = getattr(composite_config, "vision_config", None)
        assert vcfg is not None, (
            "this target implements the composite export's vision tower; the "
            "composite config carries no vision_config"
        )
        assert getattr(vcfg, "model_type", None) == "qwen4_exp_vision", (
            f"vision_config.model_type is {getattr(vcfg, 'model_type', None)!r}; "
            "this target is the Qwen4-Exp vision tower"
        )
        self.v_depth = vcfg.depth
        self.v_hidden = vcfg.hidden_size
        self.v_heads = vcfg.num_heads
        self.v_inter = vcfg.intermediate_size
        self.v_merge = vcfg.spatial_merge_size
        self.v_patch = vcfg.patch_size
        self.v_tpatch = vcfg.temporal_patch_size
        self.v_channels = vcfg.in_channels
        self.v_pos_rows = vcfg.num_position_embeddings
        assert self.v_hidden % self.v_heads == 0, (self.v_hidden, self.v_heads)
        self.v_head_dim = self.v_hidden // self.v_heads
        # vision_rope_2d needs a head divisible by 4 (two axes, one rotation pair
        # each).
        assert self.v_head_dim in _VISION_ROPE_HEAD_DIMS, (
            f"ops/attention/vision_rope_2d is certified at head widths "
            f"{_VISION_ROPE_HEAD_DIMS}; this tower's is {self.v_head_dim}"
        )
        assert (self.v_heads, self.v_heads, self.v_head_dim) in _VISION_CERTIFIED_HEADS, (
            f"the tower's attention is the encoder-only (no KV cache) mode of "
            f"attention/thop_attention, certified at {sorted(_VISION_CERTIFIED_HEADS)}; "
            f"this tower is {(self.v_heads, self.v_heads, self.v_head_dim)}"
        )
        # `hidden_act` names the block MLP's activation only; the merger runs the erf
        # `nn.GELU()`. Applying it to both is wrong by under one bf16 ulp, so there
        # are two entries (`gelu_tanh`, `gelu_erf`).
        assert vcfg.hidden_act == "gelu_pytorch_tanh", (
            f"the vision blocks' MLP activation is {vcfg.hidden_act!r}; this "
            "target calls ops/activation/gelu_tanh for it"
        )
        # No deepstack (`routing.py` refuses it; this is the second lock).
        assert not list(getattr(vcfg, "deepstack_visual_indexes", []) or []), (
            f"deepstack_visual_indexes={list(vcfg.deepstack_visual_indexes)} asks "
            "for per-layer visual feature injection, which this target does not "
            "implement"
        )
        assert vcfg.out_hidden_size == self.hidden, (
            f"the merger projects into the language model's hidden width; "
            f"out_hidden_size {vcfg.out_hidden_size} against hidden {self.hidden}"
        )
        # pos_embed_interpolate asserts a square table; fail before the first image.
        side = int(math.isqrt(self.v_pos_rows))
        assert side * side == self.v_pos_rows, (
            f"num_position_embeddings {self.v_pos_rows} is not a perfect square, "
            "so it is not a square position grid"
        )
        self.v_grid_side = side
        # `spatial_merge_size ** 2` patches become one token at the merger.
        self.v_merge_dim = self.v_hidden * self.v_merge * self.v_merge
        # One flat pixel row per patch, `(in_channels, kt, kh, kw)` channel-slowest.
        self.v_pixel_width = self.v_channels * self.v_tpatch * self.v_patch * self.v_patch
        # Placeholder token ids live on the composite config's top level.
        self.image_token_id = int(composite_config.image_token_id)
        self.video_token_id = int(composite_config.video_token_id)
        assert self.image_token_id < self.vocab and self.video_token_id < self.vocab, (
            f"the media placeholders ({self.image_token_id}, {self.video_token_id}) "
            f"are in-vocabulary against vocab_size {self.vocab}, so the engine "
            "cannot find them with its out-of-vocabulary fallback and this target "
            "must publish them as `multimodal_token_ids`"
        )

        # ---- KV pool storage ------------------------------------------------
        # `None` keeps the checkpoint's bf16 K/V (`quant_mode` 0); `FP8`, what
        # `kv_cache_config.dtype: fp8` resolves to, stores them as e4m3 under
        # QM_FP8_KV_CACHE; `NVFP4` as e2m1 pairs with e4m3 block scales under
        # QM_NVFP4_KV_CACHE. Anything else is refused: the QSA backend serves its sparse path
        # over half, bf16, fp8 and (with its own kernels here) NVFP4 pools, and silently falls
        # back to dense otherwise. The index and position planes are kernel contracts, not KV
        # storage, and keep their types under all three.
        kv_algo = model_config.quant_config.kv_cache_quant_algo
        assert kv_algo in (None, QuantAlgo.FP8, QuantAlgo.NVFP4), (
            f"this target stores K/V as the checkpoint's bf16 (kv_cache_quant_algo None), "
            f"as fp8-e4m3 (FP8) or as NVFP4; the engine resolved kv_cache_quant_algo {kv_algo!r}"
        )
        self.quant_mode = {
            None: 0,
            QuantAlgo.FP8: QM_FP8_KV_CACHE,
            QuantAlgo.NVFP4: QM_NVFP4_KV_CACHE,
        }[kv_algo]
        # What `check_attention_pools` holds every manager's K/V pool to, the draft layer's
        # included: the dtype and the innermost extent (an NVFP4 byte holds two values).
        self.kv_pool_dtype = {
            None: dt,
            QuantAlgo.FP8: torch.float8_e4m3fn,
            QuantAlgo.NVFP4: torch.uint8,
        }[kv_algo]
        self.kv_pool_width = self.head_dim // 2 if kv_algo == QuantAlgo.NVFP4 else self.head_dim
        # The weight table reads the tower's geometry off this.
        self.vision_config = vcfg

        self.mtp_layer_idx = self.num_layers
        if self.mtp_theta is None:
            self.mtp_theta = self.theta
        assert self.mtp_theta > 0.0, self.mtp_theta
        if self.mtp_enabled:
            mode = spec_config.spec_dec_mode
            assert mode.is_mtp_eagle_one_model(), (
                f"this target implements the one-model MTP-Eagle draft loop, "
                f"where one module is replayed; the engine resolved "
                f"spec_dec_mode {mode!r}"
            )
            resolved = getattr(spec_config, "num_nextn_predict_layers", None)
            assert resolved == _MTP_NUM_LAYERS, (
                f"the engine resolved num_nextn_predict_layers={resolved} from "
                f"the checkpoint; this target carries the weights of "
                f"{_MTP_NUM_LAYERS} MTP module"
            )
            # `max_draft_len + 1` becomes `predicted_tokens_per_seq` on every attention call
            # and the cached replay's step width, certified at up to `_MAX_VERIFY_WIDTH`.
            max_draft = spec_config.max_draft_len
            assert 1 <= max_draft <= _MAX_VERIFY_WIDTH - 1, (
                f"max_draft_len={max_draft} needs {max_draft + 1} verify rows per request, and "
                f"attention/thop_attention and the GDN cached replay are certified at up to "
                f"{_MAX_VERIFY_WIDTH} (max_draft_len {_MAX_VERIFY_WIDTH - 1}); 8 is also the "
                "replay kernel's ceiling"
            )
            assert self.trunk_moe_backend != "CUTEDSL" or max_draft <= _CUTEDSL_MAX_DRAFT_LEN, (
                f"max_draft_len={max_draft} with moe_config.backend=CUTEDSL: the CuteDSL expert "
                f"pipeline is certified at max_draft_len up to {_CUTEDSL_MAX_DRAFT_LEN}"
            )
            # A linear tree keeps the Blackwell spec-dec mask surface inert.
            assert not getattr(spec_config, "use_dynamic_tree", False), (
                "dynamic-tree drafting is not implemented by this target"
            )
            assert getattr(spec_config, "eagle_choices", None) is None, (
                "static-tree drafting is not implemented by this target"
            )
            assert spec_config.max_total_draft_tokens == max_draft, (
                f"a linear tree mirrors max_total_draft_tokens from "
                f"max_draft_len; the engine resolved "
                f"{spec_config.max_total_draft_tokens} against {max_draft}"
            )

            # The recurrent half of the verify step has two forms, and the cache manager
            # picks one when it is built (`TRTLLM_USE_GDN_REPLAY`, default on): the cached
            # replay or the legacy per-step recurrent buffers. Both are implemented;
            # `_bind_step_buffers` reads which one a manager allocated.
            # The FP8 block-scale runner's own shape rules.
            assert self.hidden % _FP8_BLOCK == 0 and self.moe_inter % _FP8_BLOCK == 0, (
                f"fp8_block_scale_moe_runner needs hidden ({self.hidden}) and "
                f"intermediate ({self.moe_inter}) to be multiples of {_FP8_BLOCK}"
            )
            assert self.topk < self.num_experts and self.topk <= 32, (
                f"the no-groups renormalize routing kernel takes 0 < top_k <= 32 "
                f"and top_k < num_experts; this config declares {self.topk} of "
                f"{self.num_experts}"
            )

        # The dtype the table is declared at in weights.py.
        self.ple_table_dtype = torch.float8_e4m3fn
        # Resolved at construction; read by `weights._declare_ngram_table`.
        self.ple_host_offload = _ple_host_offload_enabled()
        # CPU allocation behind the offloaded table. The `ple_table` parameter is a
        # device-address view that owns none of those pages; dropping this reference
        # frees memory a kernel still points into. None when resident.
        self._ple_table_host = None

        # Every weight but the n-gram table; meta-init intercepts torch.empty.
        self.w = _weights.MODEL_WEIGHTS.declare(_weights.MODEL_WEIGHTS.dims(self))

        # Derived once after load (`post_load_weights`).
        self._moe_scalars: list | None = None
        self._moe_scalars_cd: list | None = None
        self._gdn_gating: dict | None = None
        self._cos_sin: torch.Tensor | None = None
        # The attention op's KV scales: fp32 [1] ones over an fp8 pool (never None there: the
        # generation kernels read `kvCacheScale[0]`), fp32 [3] ones (q, K, V) over an NVFP4
        # one, None over bf16. Unit, because the fp8 QSA store and sparse GQA take no scale
        # while a request's history is written by the dense and the sparse path alike, so the
        # two agree on what a slot holds only at 1.0 (the NVFP4 entries take the [3] tensor
        # and agree at any value, but the checkpoint has no KV scales to calibrate with).
        # Lossy: fp8 K/V are stored by a plain cast, so a magnitude past the e4m3 maximum (448)
        # saturates (`_KvRangeProbe` reports it under TRTLLM_MODELING_V2_VALIDATE).
        self.kv_scale_orig_quant: torch.Tensor | None = None
        self.kv_scale_quant_orig: torch.Tensor | None = None
        # MTP module operands; None with speculation off.
        self._mtp: _MtpWeights | None = None
        self._mtp_cos_sin: torch.Tensor | None = None
        # The verify step's side-cache rollback, registered with the spec worker; None
        # with speculation off.
        self.spec_rollback: SpeculativeRollback | None = (
            SpeculativeRollback() if self.mtp_enabled else None
        )
        self._prefill: "PrefillTarget | None" = None
        self._decode: "DecodeTarget | None" = None
        # The shared expert's stream; see `_moe_block`. Made in `post_load_weights`:
        # a stream created inside a graph capture is not capturable.
        self._side_stream: torch.cuda.Stream | None = None
        self.vision_tower: VisionTower | None = None
        # Everything derived from the KV cache manager (`_bind_cache_manager`);
        # None until the first forward.
        self._bound: _Bound | None = None
        self._binds = 0
        # Set by `begin_step`, cleared by `forward`: a forward without a prepared step is a bug.
        self._step_begun = False
        # The draft layer's own bind, keyed on the manager its draft step sees.
        self._draft_bound: _DraftBound | None = None

    # ---------------------------------------------------------------- load ---

    def _norm_slices(self, key: str) -> tuple:
        """Split a grouped-norm weight into its per-group slices (taken once)."""
        return tuple(torch.split(self.w[key], self.hidden, dim=0))

    def _fuse_hc_first_projections(self) -> None:
        """Concatenate each trunk hyper-connection's `down` and `block_inject`
        weights into one `[rows, stream]` tensor, zero-padded to a multiple of
        128 rows, so a site's first projection is one GEMM
        (`_hc_first[(layer, tag)]`; `_hc_first["fhc"]` is the stack-final
        mixer's `down`, padded the same way). cuBLAS was measured 1.8x (decode)
        to 4.5x (8192 rows) faster at N = 384 than at the unpadded 324; its calls
        multiply the first `hc_first_cols` (352) columns, faster still.

        The checkpoint's parameters are rebound as row views of the padded
        tensors, so this costs only the padding rows (~1.2 MB per site), and
        anything still reading them separately reads the same values."""
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

    def post_load_weights(self) -> None:
        """Derive what the targets read, then build them, now that the weights are real.

        Not an override (DecoderModel has no such hook); named for the shell hook."""
        w = self.w
        device = w["embed"].device
        moe_scalars = [
            _moe_layer_scalars(w, i, self.trunk_moe_backend) for i in range(self.num_layers)
        ]
        # With the second expert copy (`moe_prefill_copy`), its CuteDSL scalars, which
        # prefill-sized launches run on (`_CUTE_DSL_MOE_MIN_ROWS`).
        moe_scalars_cd = (
            [_moe_layer_scalars(w, i, "CUTEDSL") for i in range(self.num_layers)]
            if self.moe_prefill_copy
            else None
        )
        self._moe_scalars = moe_scalars
        self._moe_scalars_cd = moe_scalars_cd
        # Both delta-rule entries index A_log and dt_bias as `ptr + head` and certify
        # fp32; the checkpoint is bf16.
        self._gdn_gating = {
            i: (
                w[f"l{i}_A_log"].float().contiguous(),
                w[f"l{i}_dt_bias"].float().contiguous(),
            )
            for i in self.gdn_layers
        }
        # Reconcile the table height against the checkpoint's id-space tensors: an
        # index past the end is a device-side assert that poisons the process.
        table_rows = w["ple_table"].shape[0]
        used_rows = int(w["ple_offs"][-1]) + int(w["ple_vocab"][-1])
        assert used_rows <= table_rows, (
            f"the n-gram table has {table_rows} rows but its per-head offsets "
            f"and vocabulary sizes address {used_rows}"
        )
        assert table_rows % self.ngram_vocab_divisor == 0, (
            f"the n-gram table's {table_rows} rows are not a multiple of "
            f"make_ngram_vocab_size_divisible_by={self.ngram_vocab_divisor}"
        )
        assert table_rows - used_rows < self.ngram_vocab_divisor, (
            f"the n-gram table carries {table_rows - used_rows} padding rows, "
            f"more than one divisor of {self.ngram_vocab_divisor}: the shards "
            f"do not concatenate to the padded id space this checkpoint "
            f"declares"
        )
        self._cos_sin = self._index_rope_table(self.theta, device)
        if self.mtp_enabled:
            # The draft layer's index plane rotates at the module's own rope base, like its
            # attention rope.
            self._mtp_cos_sin = (
                self._cos_sin
                if self.mtp_theta == self.theta
                else self._index_rope_table(self.mtp_theta, device)
            )
        if self.quant_mode in (QM_FP8_KV_CACHE, QM_NVFP4_KV_CACHE):
            # Built here, not at construction: that runs under meta init.
            scales = 3 if self.quant_mode == QM_NVFP4_KV_CACHE else 1
            self.kv_scale_quant_orig = torch.ones(scales, dtype=torch.float32, device=device)
            self.kv_scale_orig_quant = 1.0 / self.kv_scale_quant_orig
        self._report_ple_residency()
        if self.mtp_enabled:
            self._mtp = _mtp_views(self)

        self._fuse_hc_first_projections()

        # The CPU-only load checks build the targets over host weights; they launch nothing.
        self._side_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)
        self.vision_tower = VisionTower(self)

    def _index_rope_table(self, theta: float, device) -> torch.Tensor:
        """The index branch's rope table at `theta` in the QSA kernels' layout:
        `[max_positions, 2 * rotary_pairs]` fp32, cos block then sin block."""
        pairs = self.rotary_dim // 2
        exponent = torch.arange(0, self.rotary_dim, 2, dtype=torch.float64, device=device)
        inv_freq = (1.0 / (theta ** (exponent / self.rotary_dim))).to(torch.float32)
        positions = torch.arange(self.max_positions, dtype=torch.float32, device=device)
        angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
        table = torch.cat([torch.cos(angles), torch.sin(angles)], dim=1).contiguous()
        assert table.shape == (self.max_positions, 2 * pairs), table.shape
        return table

    def _report_ple_residency(self) -> None:
        """Check where the n-gram table ended up, and report it once.

        Offload is output-identical, so a stray `.to()` or a driver that declined
        the mapping would pass every smoke test; the pointer's memory kind read
        back from the driver is the signal that cannot be faked. The report is
        printed here, before the KV pool is sized from what is left."""
        table = self.w["ple_table"]
        kind = _weights.memory_kind(table.data_ptr())
        expected = (
            _weights.CUDA_MEMORY_TYPE_HOST
            if self.ple_host_offload
            else _weights.CUDA_MEMORY_TYPE_DEVICE
        )
        assert kind == expected, (
            f"the n-gram table was declared with {_PLE_HOST_OFFLOAD_ENV}="
            f"{'1' if self.ple_host_offload else '0'}, so its pages must be "
            f"{'host' if self.ple_host_offload else 'device'} memory, but the "
            f"driver reports cudaMemoryType {kind} for the pointer the forward "
            f"gathers from. The offload is output-identical, so nothing "
            f"downstream of this would notice"
        )
        mib = 1024.0 * 1024.0
        table_mib = table.numel() * table.element_size() / mib
        free, total = torch.cuda.mem_get_info()
        print(
            f"ple n-gram table: {tuple(table.shape)} {table.dtype}, "
            f"{table_mib:.1f} MiB, "
            f"{'HOST-resident (device-addressable)' if self.ple_host_offload else 'DEVICE-resident'}"
            f" [{_PLE_HOST_OFFLOAD_ENV}="
            f"{os.environ.get(_PLE_HOST_OFFLOAD_ENV, '1 (default)')}]; "
            f"after load: allocated {torch.cuda.memory_allocated() / mib:.1f} MiB, "
            f"reserved {torch.cuda.memory_reserved() / mib:.1f} MiB, "
            f"driver-used {(total - free) / mib:.1f} MiB of {total / mib:.1f} MiB",
            flush=True,
        )

    # ------------------------------------------------------------ fail-fast ---

    def _pool_geometry(self, tokens_per_block: int) -> _PoolGeometry:
        return _PoolGeometry(
            tokens_per_block=tokens_per_block,
            heads_kv=self.heads_kv,
            head_dim=self.head_dim,
            index_kv_heads=self.index_kv_heads,
            index_head_dim=self.index_head_dim,
            conv_dim=self.conv_dim,
            conv_kernel=self.conv_kernel,
            num_k_heads=self.num_k_heads,
            num_v_heads=self.num_v_heads,
            head_v_dim=self.head_v_dim,
            head_k_dim=self.head_k_dim,
            stream=self.stream,
            ple_state_len=self.ple_state_len,
            ngram_context_len=self.ngram_context_len,
            dtype=self.w["embed"].dtype,
            kv_dtype=self.kv_pool_dtype,
            kv_width=self.kv_pool_width,
        )

    def _kv_planes(self, cm, g: _PoolGeometry, layer: int, index_cache) -> _KvPlanes:
        """One layer's paged K/V views from `cm`, checked against the contract: the data
        planes, and under NVFP4 the block-scale planes that go with them."""
        pool = cm.get_buffers(layer, kv_layout="HND")
        check_attention_pools(g, layer, index_cache, pool)
        if self.quant_mode != QM_NVFP4_KV_CACHE:
            return _KvPlanes(pool[:, 0], pool[:, 1])
        _require(cm, ("get_block_scale_buffers",), "reads an NVFP4 pool's e4m3 block-scale planes")
        scales = cm.get_block_scale_buffers(layer, kv_layout="HND")
        check_kv_scale_planes(g, layer, pool, scales)
        return _KvPlanes(pool[:, 0], pool[:, 1], scales[:, 0], scales[:, 1])

    def _bind_cache_manager(self, md, position_ids) -> None:
        """Bind this module to one KV cache manager: fail-fast on the contract, then
        take every pool view from it.

        Called once per manager, not per model: with capacity estimation on, the
        engine builds a throwaway manager, warms up, shuts it down and builds the
        real one on the same module, and a view held across the swap dangles
        silently. The result is built in locals and installed as one `_Bound`, so a
        failed bind leaves the previous binding untouched. The contract checks
        re-run on every bind."""
        # Checked first: some fields below are properties that assert on a
        # manager-less metadata instead of raising AttributeError.
        cm = md.kv_cache_manager
        assert cm is not None, "a paged KV cache is required"
        # A re-bind during capture would bake two managers' pointers into one graph;
        # the engine releases graphs before its own swap, so this is an ordering
        # violation.
        assert not (self._binds and torch.cuda.is_current_stream_capturing()), (
            f"the KV cache manager changed to {type(cm).__name__} during CUDA graph capture"
        )
        self._binds += 1

        missing = [name for name in _QSA_FIELDS if not hasattr(md, name)]
        assert not missing, (
            f"metadata fields missing: {missing}; the engine did not select QSA "
            "metadata for this model"
        )

        # `position_ids` is the int32 [3, 1, T] buffer the engine builds on an mRoPE
        # config even for text-only batches; row 0 is the text position.
        assert position_ids.dtype == torch.int32, position_ids.dtype
        assert self.mrope_declared, "this target takes the mRoPE position layout"
        assert position_ids.dim() == 3 and position_ids.shape[0] == _QSA_POSITION_AXES, (
            f"expected the [{_QSA_POSITION_AXES}, 1, T] mRoPE position buffer, got "
            f"{tuple(position_ids.shape)}"
        )

        # thop_attention addresses the pool by the layer's pool offset, not its
        # ordinal among attention layers; counting them aliases layers with finite,
        # plausible output.
        pool_idx = {i: cm.layer_offsets[i] for i in self.attn_layers}
        pools = {row[0] for row in md.host_kv_cache_pool_mapping.tolist()}
        assert pools <= {0, 1}, (
            f"KV addressing over {len(pools)} pools is not certified "
            f"(thop_attention's cells cover one and two); layer->pool ids {sorted(pools)}"
        )

        # No sliding window: both targets pass `max_seq_len` as the window. The vector
        # is indexed by pool offset and recurrent layers carry a large negative
        # sentinel, so only attention offsets are read.
        windows = cm.max_attention_window_vec
        attn_windows = {windows[off] for off in pool_idx.values()}
        assert attn_windows <= {None, md.max_seq_len}, (
            f"the attention layers' pool windows are {sorted(map(repr, attn_windows))} "
            f"but this target passes max_seq_len {md.max_seq_len} as the window; "
            "a sliding window is not implemented here"
        )

        # The NVFP4 attention kernels exist for the paged-context path only; without it the
        # dense op would not read the pool the sparse path wrote.
        assert self.quant_mode != QM_NVFP4_KV_CACHE or md.use_paged_context_fmha, (
            "an NVFP4 KV cache is read by the paged-context attention kernels only, which the "
            "engine enables for chunked prefill, KV cache block reuse or speculative decoding; "
            "none of the three is on"
        )
        g = self._pool_geometry(md.tokens_per_block)
        _require(
            cm,
            ("get_index_k_buffer", "get_qsa_position_buffer", "get_buffers"),
            "reads the QSA index plane, the shared position plane and the paged K/V",
        )
        # `qsa_topk_select` checks only that the buffer is at least `index_topk` wide,
        # then addresses it as flat `index_topk`-wide rows ignoring the real stride, so
        # a width mismatch silently mis-writes every row but the first.
        assert md.qsa_topk_indices.shape[1] == self.block_topk, (
            f"the engine sized the QSA selection buffer {md.qsa_topk_indices.shape[1]} "
            f"wide and this target selects {self.block_topk} blocks per row"
        )
        position_cache = cm.get_qsa_position_buffer()
        check_position_plane(g, position_cache)
        index_cache, kv = {}, {}
        for i in self.attn_layers:
            index_cache[i] = cm.get_index_k_buffer(i)
            kv[i] = self._kv_planes(cm, g, i, index_cache[i])

        # A missed hybrid predicate would still build and pass a smoke run with 36
        # layers carrying no state.
        mm = md.mamba_metadata
        assert mm is not None and mm is not False, (
            "attn_metadata.mamba_metadata is absent: the runtime allocated no "
            f"recurrent-state pool, so the {len(self.gdn_layers)} linear-attention "
            "layers would run with no state"
        )
        mm_missing = [name for name in _MAMBA_FIELDS if not hasattr(mm, name)]
        assert not mm_missing, f"mamba metadata fields missing: {mm_missing}"
        # `ple_layer_cache` exists only on the V2 hybrid manager; under V1 the PLE
        # state silently does not exist.
        _require(
            cm,
            ("get_conv_states", "get_ssm_states", "ple_layer_cache"),
            "reads the recurrent, conv and PLE pools",
        )
        conv = {i: cm.get_conv_states(i) for i in self.gdn_layers}
        ssm = {i: cm.get_ssm_states(i) for i in self.gdn_layers}
        conv0, ssm0 = conv[self.gdn_layers[0]], ssm[self.gdn_layers[0]]
        check_recurrent_pools(g, conv0, ssm0)

        ple_cache = cm.ple_layer_cache(self.ple_layer)
        assert ple_cache is not None, (
            f"no PLE lifecycle state for decoder layer {self.ple_layer}: the runtime "
            "allocated neither the dilated conv state nor the n-gram id history"
        )
        ple_conv, ple_ngram = ple_cache
        check_ple_pools(g, ple_conv, ple_ngram, conv0)

        assert md.effective_beam_width == 1, "beam search is not implemented"
        assert md.cache_indirection is None, "beam search is not implemented"
        assert not md.is_cross, "cross attention is not implemented"
        # With speculation on, the Blackwell trtllm-gen spec-dec mask machinery is off
        # for a linear tree at sm_103: every `spec_decoding_*` tensor stays None and
        # `predicted_tokens_per_seq` alone produces the within-block mask.
        assert not md.is_spec_decoding_enabled, (
            "the attention metadata reports is_spec_decoding_enabled; on sm_103 a "
            "linear-tree MTP forces that machinery off"
        )
        bound = _Bound(
            cm, pool_idx, index_cache, position_cache, kv, conv, ssm, (ple_conv, ple_ngram)
        )
        if self.mtp_enabled:
            bound = bound._replace(**self._bind_step_buffers(cm, g, mm, conv0, ssm0))
        self._bound = bound

    def _bind_step_buffers(self, cm, g: _PoolGeometry, mm, conv0, ssm0) -> dict:
        """Per-manager fail-fast for the MTP increment: the verify step needs
        runtime-owned buffers the base path never touches. Every row's conv window goes
        to a per-step buffer in both recurrent modes; the recurrent state is either the
        cached replay (a checkpoint per slot plus a compact history of accepted tokens,
        no per-step recurrent buffer) or, with `TRTLLM_USE_GDN_REPLAY=0`, per-step
        recurrent buffers. Returns the `_Bound` fields it adds."""
        assert cm.is_speculative(), (
            f"the KV cache manager {type(cm).__name__} reports is_speculative() False, "
            "so it allocated none of the per-step state buffers the verify path writes"
        )
        _require(
            cm,
            ("mamba_layer_cache", "get_replay_state_update_metadata"),
            "binds its per-step state buffers and replay metadata",
        )
        # Bind the per-layer views once for subsequent forwards.
        layer_caches = {i: cm.mamba_layer_cache(i) for i in self.gdn_layers}
        int_conv = {i: cache.intermediate_conv_window for i, cache in layer_caches.items()}
        int_idx = getattr(cm, "intermediate_state_indices", None)
        g0 = self.gdn_layers[0]
        check_step_buffers(g, int_conv[g0], conv0, int_idx)
        int_ssm0 = layer_caches[g0].intermediate_ssm
        if cm.get_replay_state_update_metadata() is None:
            check_legacy_state(g, int_ssm0, ssm0)
            int_ssm = {i: cache.intermediate_ssm for i, cache in layer_caches.items()}
            return dict(int_idx=int_idx, int_conv=int_conv, int_ssm=int_ssm)
        assert int_ssm0 is None, (
            "the cache manager runs the cached-replay verify path and also allocated a "
            "per-step recurrent buffer; the runtime allocates one or the other"
        )
        return dict(int_idx=int_idx, int_conv=int_conv, replay=self._bind_replay(cm, g, mm, ssm0))

    def _bind_replay(self, cm, g: _PoolGeometry, mm, ssm0) -> _Replay:
        """The cached-replay history buffers and bookkeeping of one manager."""
        meta = cm.get_replay_state_update_metadata()
        assert meta is not None, (
            "the cached-replay binding requires replay metadata "
            "(prev_num_accepted_tokens / cache_buf_idx / widths)"
        )
        history, width = meta.replay_history_size, meta.replay_step_width
        check_replay_state(history, width, meta.prev_num_accepted_tokens, meta.cache_buf_idx, ssm0)
        old = {}
        for i in self.gdn_layers:
            layer_cache = cm.mamba_layer_cache(i)
            old[i] = (
                layer_cache.old_x,
                layer_cache.old_B,
                layer_cache.old_dt,
                layer_cache.old_dA_cumsum,
            )
            check_replay_history(g, i, history, ssm0.shape[0], *old[i][:3])
        # The partitioned replay is the manager's to commit: it commits exactly when this
        # flag holds and a step carries `CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE` or more
        # generation requests, and the forward hands the kernel the work items under that
        # same condition.
        all_layer = bool(getattr(cm, "use_gdn_cached_replay_all_layer_commit", False))
        if all_layer:
            missing = [n for n in ("replay_work_items", "replay_n_writes") if not hasattr(mm, n)]
            assert not missing, (
                f"the manager's all-layer replay commit is on but the mamba metadata lacks "
                f"{missing}; the partitioned verify call needs both"
            )
        return _Replay(
            old, meta.cache_buf_idx, meta.prev_num_accepted_tokens, history, width, all_layer
        )

    def _bind_draft_cache_manager(self, md) -> None:
        """Bind the draft layer to the manager its draft step sees: its own index plane,
        that manager's position plane and its paged K/V.

        With a separate draft KV cache that is the draft manager the worker swaps in for
        the draft loop, whose pools hold the draft layer's pages and none of the trunk's;
        otherwise the trunk's manager, which maps the draft layer as well. Keyed on the
        manager like `_bind_cache_manager`. Never first bound under capture: the runner
        runs every captured batch shape eagerly first, so a bind under capture would bake
        views no eager step validated into the graph."""
        cm = md.kv_cache_manager
        assert not torch.cuda.is_current_stream_capturing(), (
            f"the draft layer is not bound to KV cache manager {type(cm).__name__} (first use, "
            "or the manager changed since its last bind) and CUDA graph capture is active; "
            "an eager step must take and check its views first"
        )
        _require(
            cm,
            ("get_index_k_buffer", "get_qsa_position_buffer", "get_buffers"),
            "reads the draft layer's QSA index plane, position plane and paged K/V",
        )
        g = self._pool_geometry(md.tokens_per_block)
        layer = self.mtp_layer_idx
        index_cache = cm.get_index_k_buffer(layer)
        position_cache = cm.get_qsa_position_buffer()
        check_position_plane(g, position_cache)
        kv = self._kv_planes(cm, g, layer, index_cache)
        share_slot = None
        if self.mtp_index_share:
            # The metadata sizes its shared selection buffers from the same resolved knob,
            # one row per sparse layer of whichever manager is bound.
            assert md.sparse_metadata_params.mtp_index_share, (
                "this target resolved IndexShare on but the engine's QSA metadata did not"
            )
            slots = cm.qsa_shared_topk_slots
            check_shared_selection(md.qsa_shared_topk_indices, slots, layer, self.block_topk)
            share_slot = slots[layer]
        self._draft_bound = _DraftBound(cm, index_cache, position_cache, kv, share_slot)

    # -------------------------------------------------------------- forward ---

    def _select_target(self, attn_metadata) -> "PrefillTarget | DecodeTarget":
        """Which target runs this step: decode if `num_contexts == 0`, else prefill.

        A draft loop rewrites `num_contexts` in place between draft steps, so the
        draft step routes through here per call."""
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
        """Check what holds for every step, bind the cache manager, choose the target and
        let the prefill target prepare the step.

        The shell calls it before `forward`, on every step. It is separate because a
        breakable prefill CUDA graph captures `forward` and replays it without calling it:
        whatever a step reads from the host (metadata, host buffers, the manager) is read
        here, where it runs eagerly on a replayed step too. `vision` is `(rows,
        mm_token_indices)` when the step carries media: the tower's `[R, hidden]` rows and
        the `[R]` packed-batch rows they replace; the prefill target splices them inside
        its forward."""
        assert self._prefill is not None and self._decode is not None, (
            "load_weights must run before forward"
        )
        assert position_ids is not None
        assert isinstance(attn_metadata, TrtllmAttentionMetadata)
        # Unimplemented inputs must fail loudly, not be dropped.
        assert lora_params is None, "LoRA is not implemented by this target"
        # `input_ids` is required on every step: the PLE block hashes token ids whether or
        # not vision rows replace some embeddings, and text gates would not catch a mis-hash.
        assert input_ids is not None, (
            "this target's PLE layer hashes token ids, so input_ids is required "
            "on every step, including one that carries vision rows"
        )
        # The token embedding is this target's own and vision rows are spliced into it
        # inside the forward; an embedding handed in from outside would be dropped on every
        # replay of a breakable prefill graph.
        assert inputs_embeds is None, (
            "this target computes its token embedding itself and splices vision rows into "
            "it; an engine-supplied inputs_embeds is not implemented"
        )
        # `spec_metadata` is present exactly when drafting. On MTP_EAGLE_ONE_MODEL it
        # sets `layers_to_capture = ()`, so the trunk owes no capture hook.
        assert (spec_metadata is not None) == self.mtp_enabled, (
            f"spec_metadata {'arrived' if spec_metadata is not None else 'is absent'} "
            f"but the model was built with mtp_enabled={self.mtp_enabled}; the "
            "draft path and the weights it loads are decided at construction"
        )
        # Keyed on the manager, not a first-forward flag: capacity estimation swaps the
        # manager under the same module (see `_bind_cache_manager`).
        if self._bound is None or attn_metadata.kv_cache_manager is not self._bound.cm:
            self._bind_cache_manager(attn_metadata, position_ids)
        target = self._select_target(attn_metadata)
        if target is self._prefill:
            target.prepare(attn_metadata, input_ids, position_ids, vision, spec_metadata)
        else:
            assert vision is None, (
                "a generation-only step carries vision rows; the executor attaches media "
                "to context requests only"
            )
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
        """Run the step `begin_step` prepared on the target it chose.

        Under a breakable prefill CUDA graph this is the body the engine captures for a
        step with context rows; a replayed step never enters it, which is why nothing here
        may read the step from the host. Returns the logits input, or under speculation the
        decode target's `(final, hc)` pair and the prefill target's one packed tensor
        (`split_spec_output`)."""
        assert self._step_begun, (
            "the shell calls begin_step before every forward; this forward has no prepared step"
        )
        self._step_begun = False
        return self._select_target(attn_metadata).forward(
            attn_metadata, input_ids=input_ids, position_ids=position_ids, **kwargs
        )

    def split_spec_output(self, output) -> tuple[torch.Tensor, torch.Tensor]:
        """`(final, hc)` out of what `forward` returned under speculation: `final` is the
        `[T, hidden]` mix `lm_head` reads and `hc` the `[T, hc_count * hidden]` stream the
        MTP module reads. The decode target returns the pair; the prefill target returns one
        `[T, hidden + stream]` tensor, of which the two are column views."""
        if isinstance(output, tuple):
            return output
        final, hc = torch.split(output, [self.hidden, self.stream], -1)
        return final, hc


class _LayerWeights(NamedTuple):
    """One trunk layer's weights as the views the ops read (GEMM operands are `.t()`
    views). Attention fields are None on a Gated-DeltaNet layer and vice versa."""

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
    out: torch.Tensor | None
    conv_w: torch.Tensor | None
    a_log: torch.Tensor | None
    dt_bias: torch.Tensor | None
    gnorm: torch.Tensor | None
    router: torch.Tensor
    moe_gscale: torch.Tensor
    fc1_w: torch.Tensor
    fc1_s: torch.Tensor
    fc2_w: torch.Tensor
    fc2_s: torch.Tensor
    # trtllm-gen scalars; None on CuteDSL.
    out1_scale: torch.Tensor | None
    out1_gate_scale: torch.Tensor | None
    out2_scale: torch.Tensor | None
    # CuteDSL scalars; None on trtllm-gen.
    alpha1: torch.Tensor | None
    alpha2: torch.Tensor | None
    global_sf: torch.Tensor | None
    # Under trtllm-gen, the second expert copy in the CuteDSL layout and its scalars,
    # which prefill-sized launches run on (`_CUTE_DSL_MOE_MIN_ROWS`); None on CuteDSL and
    # without the copy (`moe_prefill_copy`).
    fc1_w_cd: torch.Tensor | None
    fc1_s_cd: torch.Tensor | None
    fc2_w_cd: torch.Tensor | None
    fc2_s_cd: torch.Tensor | None
    alpha1_cd: torch.Tensor | None
    alpha2_cd: torch.Tensor | None
    global_sf_cd: torch.Tensor | None
    sh_gu: torch.Tensor
    sh_dn: torch.Tensor
    sh_gate: torch.Tensor
    # `[gate | up | gate logit | pad]` (`weights._cat_rows`): `sh_gu` and `sh_gate` are its
    # column blocks.
    sh_gug: torch.Tensor


def _layer_views(core: Qwen4ExpTp1ModelingV2) -> list[_LayerWeights]:
    """Every trunk layer's `_LayerWeights`, read off the realized weights."""
    w = core.w
    # The expert stacks are uint8 storage; each backend reads the packed nibbles and the
    # block scales as its own dtypes.
    if core.trunk_moe_backend == "CUTEDSL":
        w_dtype, s_dtype = torch.float4_e2m1fn_x2, torch.uint8
    else:
        w_dtype, s_dtype = torch.uint8, torch.float8_e4m3fn
    layers = []
    for i in range(core.num_layers):
        attn: dict = dict(
            attn_in=None,
            o=None,
            q_norm=None,
            k_norm=None,
            idx_qn=None,
            idx_kn=None,
        )
        gdn: dict = dict(
            in_proj=None,
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
                in_proj=w[f"l{i}_in_proj_w"].t(),
                out=w[f"l{i}_out_w"].t(),
                # [conv_dim, 1, width] -> [conv_dim, width]: a view (unit width stride).
                conv_w=w[f"l{i}_conv_w"].reshape(core.conv_dim, core.conv_kernel),
                a_log=a_log,
                dt_bias=dt_bias,
                gnorm=w[f"l{i}_gnorm"],
            )
        scalars = core._moe_scalars[i]
        cute_dsl_copy: dict = dict(
            fc1_w_cd=None,
            fc1_s_cd=None,
            fc2_w_cd=None,
            fc2_s_cd=None,
            alpha1_cd=None,
            alpha2_cd=None,
            global_sf_cd=None,
        )
        if core._moe_scalars_cd is not None:
            scalars_cd = core._moe_scalars_cd[i]
            cute_dsl_copy = dict(
                fc1_w_cd=w[f"l{i}_fc1_w_cd"],
                fc1_s_cd=w[f"l{i}_fc1_s_cd"],
                fc2_w_cd=w[f"l{i}_fc2_w_cd"],
                fc2_s_cd=w[f"l{i}_fc2_s_cd"],
                alpha1_cd=scalars_cd["alpha1"],
                alpha2_cd=scalars_cd["alpha2"],
                global_sf_cd=scalars_cd["global_sf"],
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
                moe_gscale=scalars["global_scale"],
                fc1_w=w[f"l{i}_fc1_w"].view(w_dtype),
                fc1_s=w[f"l{i}_fc1_s"].view(s_dtype),
                fc2_w=w[f"l{i}_fc2_w"].view(w_dtype),
                fc2_s=w[f"l{i}_fc2_s"].view(s_dtype),
                out1_scale=scalars.get("output1_scale_scalar"),
                out1_gate_scale=scalars.get("output1_scale_gate_scalar"),
                out2_scale=scalars.get("output2_scale_scalar"),
                alpha1=scalars.get("alpha1"),
                alpha2=scalars.get("alpha2"),
                global_sf=scalars.get("global_sf"),
                **cute_dsl_copy,
                sh_gu=w[f"l{i}_sh_gu"][: 2 * core.shared_inter].t(),
                sh_dn=w[f"l{i}_sh_dn"].t(),
                sh_gate=w[f"l{i}_sh_gu"][2 * core.shared_inter : 2 * core.shared_inter + 1].t(),
                sh_gug=w[f"l{i}_sh_gu"].t(),
            )
        )
    return layers


class _StackWeights(NamedTuple):
    """The weights outside the layer loop: the stack-final mixer (this checkpoint's
    final norm) and the PLE side path."""

    fhc_norm: torch.Tensor
    fhc_first: torch.Tensor
    fhc_up: torch.Tensor
    ple_key: torch.Tensor
    ple_val: torch.Tensor
    ple_nk: tuple
    ple_nq: tuple
    ple_nc: tuple
    ple_nk_w: torch.Tensor
    ple_nq_w: torch.Tensor
    ple_nc_w: torch.Tensor
    ple_conv_w: torch.Tensor
    ple_mult: torch.Tensor
    ple_vocab: torch.Tensor
    ple_offs: torch.Tensor
    ple_tscale: torch.Tensor


def _stack_views(core: Qwen4ExpTp1ModelingV2) -> _StackWeights:
    w = core.w
    return _StackWeights(
        fhc_norm=w["fhc_norm"],
        fhc_first=core._hc_first["fhc"].t(),
        fhc_up=w["fhc_up"].t(),
        ple_key=w["ple_key_w"].t(),
        ple_val=w["ple_val_w"].t(),
        ple_nk=core._norm_slices("ple_nk"),
        ple_nq=core._norm_slices("ple_nq"),
        ple_nc=core._norm_slices("ple_nc"),
        # The same three weights whole, for the decode target's fused tail, which
        # normalizes each stream inside one program.
        ple_nk_w=w["ple_nk"],
        ple_nq_w=w["ple_nq"],
        ple_nc_w=w["ple_nc"],
        ple_conv_w=w["ple_conv_w"],
        ple_mult=w["ple_mult"],
        ple_vocab=w["ple_vocab"],
        ple_offs=w["ple_offs"],
        # `dequantize_e4m3_per_tensor` needs the scale at the input's rank (3), all 1s.
        ple_tscale=w["ple_tscale"].reshape(1, 1, 1),
    )


class _MtpWeights(NamedTuple):
    """The MTP module's weights (keyed at its layer index, `num_hidden_layers`)."""

    pre_e_norm: torch.Tensor
    pre_h_norm: tuple
    fc_e: torch.Tensor
    fc_h: torch.Tensor
    ahc_norm: tuple
    ahc_down: torch.Tensor
    ahc_up: torch.Tensor
    ahc_inject: torch.Tensor
    mhc_norm: tuple
    mhc_down: torch.Tensor
    mhc_up: torch.Tensor
    mhc_inject: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    o: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    idx: torch.Tensor
    idx_qn: torch.Tensor
    idx_kn: torch.Tensor
    router: torch.Tensor
    fc1_w: torch.Tensor
    fc1_s: torch.Tensor
    fc2_w: torch.Tensor
    fc2_s: torch.Tensor
    sh_gu: torch.Tensor
    sh_dn: torch.Tensor
    sh_gate: torch.Tensor
    fhc_norm: tuple
    fhc_down: torch.Tensor
    fhc_up: torch.Tensor


def _mtp_views(core: Qwen4ExpTp1ModelingV2) -> _MtpWeights:
    w = core.w
    p = f"l{core.mtp_layer_idx}_mtp_"
    return _MtpWeights(
        pre_e_norm=w[f"{p}pre_e_norm"],
        pre_h_norm=core._norm_slices(f"{p}pre_h_norm"),
        fc_e=w[f"{p}fc_e"].t(),
        fc_h=w[f"{p}fc_h"].t(),
        ahc_norm=core._norm_slices(f"{p}ahc_norm"),
        ahc_down=w[f"{p}ahc_down"],
        ahc_up=w[f"{p}ahc_up"],
        ahc_inject=w[f"{p}ahc_inject"],
        mhc_norm=core._norm_slices(f"{p}mhc_norm"),
        mhc_down=w[f"{p}mhc_down"],
        mhc_up=w[f"{p}mhc_up"],
        mhc_inject=w[f"{p}mhc_inject"],
        q=w[f"{p}q_w"].t(),
        k=w[f"{p}k_w"].t(),
        v=w[f"{p}v_w"].t(),
        o=w[f"{p}o_w"].t(),
        q_norm=w[f"{p}q_norm"],
        k_norm=w[f"{p}k_norm"],
        idx=w[f"{p}idx_w"].t(),
        idx_qn=w[f"{p}idx_qn"],
        idx_kn=w[f"{p}idx_kn"],
        router=w[f"{p}router"].t(),
        fc1_w=w[f"{p}fc1_w"],
        fc1_s=w[f"{p}fc1_s"],
        fc2_w=w[f"{p}fc2_w"],
        fc2_s=w[f"{p}fc2_s"],
        sh_gu=w[f"{p}sh_gu"][: 2 * core.shared_inter].t(),
        sh_dn=w[f"{p}sh_dn"].t(),
        sh_gate=w[f"{p}sh_gu"][2 * core.shared_inter : 2 * core.shared_inter + 1].t(),
        fhc_norm=core._norm_slices(f"{p}fhc_norm"),
        fhc_down=w[f"{p}fhc_down"],
        fhc_up=w[f"{p}fhc_up"],
    )


def _select(core, scores, row_starts, row_ends, topk_indices, live_cols=None) -> None:
    """Select each row's top-`block_topk` groups into `topk_indices` with `qsa_topk_select`,
    then rewrite the row canonically, in place.

    The radix engine compacts its picks through a shared-memory atomic, so the order is
    launch-dependent, and where the tie class at the budget's last score straddles the
    budget -- every zero-score group of a row with fewer positive scores than the budget,
    on this checkpoint's relu scores -- which tied groups are kept can be too. The
    expansion keeps the row's order and the sparse GQA reduces in it, so a raw row past the
    dense threshold is not reproducible run to run. `qsa_canonical_selection` rewrites it
    into the window's canonical selection -- the groups strictly above the budget's last
    score, then the highest-index (most recent) ties, ascending, `-1` last -- a function of
    the scores alone, and the engine's own set in ascending order wherever the budget's
    last score is untied. A canonical row is prefix-packed, which the expansion and the
    IndexShare capture and reuse require.

    `index_topk` is the block budget: the entry's default (2048) is 4x the buffer width and
    the op ignores the row stride, so it mis-writes silently.

    The rewrite reads every column it is given, and `scores` is as wide as the block table,
    not the live window. A caller that knows a host-side bound on the rows' windows passes
    it as `live_cols`: the rewrite then reads only those columns and returns the same rows."""
    qsa_topk_select(scores, row_starts, row_ends, topk_indices, index_topk=core.block_topk)
    if live_cols is not None:
        scores = scores[:, :live_cols]
    qsa_canonical_selection(scores, row_starts, row_ends, topk_indices)


def _store_kv(core, kv, k, v, request_indices, logical_positions, block_table, tokens_per_block):
    """Append this step's K/V rows to the layer's paged pool: a plain move over a bf16 or fp8
    pool (an fp8 pool takes e4m3 unscaled, as the dense op does at the unit KV scales), the
    quantizing store over an NVFP4 one, at the same second-level scales as the dense op."""
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


def _sparse_gqa(
    core,
    kv,
    q,
    block_table,
    selected_tokens,
    request_indices,
    tokens_per_block,
    query_positions,
    compress_ratio,
):
    """Sparse GQA over the layer's paged pool (dequantizing an NVFP4 one). The prefill pair
    `query_positions` / `compress_ratio` is a bound on a context row's scan, or None, None."""
    if kv.k_scale is None:
        return qsa_paged_sparse_gqa(
            q=q,
            k_cache=kv.k,
            v_cache=kv.v,
            block_table=block_table,
            selected_tokens=selected_tokens,
            request_indices=request_indices,
            tokens_per_block=tokens_per_block,
            softmax_scale=core.softmax_scale,
            query_positions=query_positions,
            compress_ratio=compress_ratio,
        )
    return qsa_paged_sparse_gqa_nvfp4(
        q=q,
        k_cache=kv.k,
        v_cache=kv.v,
        k_scale_cache=kv.k_scale,
        v_scale_cache=kv.v_scale,
        kv_scale_quant_orig=core.kv_scale_quant_orig,
        block_table=block_table,
        selected_tokens=selected_tokens,
        request_indices=request_indices,
        tokens_per_block=tokens_per_block,
        softmax_scale=core.softmax_scale,
        query_positions=query_positions,
        compress_ratio=compress_ratio,
    )


def _attn_in_heads(core, proj):
    """The main q, k and v heads of the fused attention projection's output `proj`
    `[T, attn_in_width]` (`weights._attn_in_w`), as `[T, heads, head_dim]` views."""
    q, k, v, _ = torch.split(
        torch.reshape(proj, [proj.shape[0], core.attn_in_width // core.head_dim, core.head_dim]),
        [core.heads_q, core.heads_kv, core.heads_kv, core.attn_in_tail_heads - core.heads_kv],
        1,
    )
    return q, k, v


def _cute_dsl_routed(core, L, logits, xq, xsf, routed_out) -> torch.Tensor:
    """The trunk's routed experts on the CuteDSL pipeline: Renormalize routing, the sort
    into per-expert row tiles, FC1 gathering its rows from the unpermuted NVFP4 activation
    `xq` / `xsf` (linear scales) with SwiGLU and the requantize to the layer's one FC2 input
    scale fused, FC2, and the routing-weighted unpermute in a fixed slot order, which
    writes every row of `routed_out` `[T, hidden]` and returns it."""
    # fp32 weights: the dtype the unpermute scales by.
    topk_ids, topk_w = renorm_moe_routing_op(logits, topk=core.topk, output_dtype=torch.float32)
    # The sort pads each expert's (token, expert) pairs to whole tiles and emits the tile
    # tables both GEMMs walk.
    tile_expert, tile_limit, expanded_to_permuted, permuted_to_expanded, _, live_tiles = moe_sort(
        token_selected_experts=topk_ids,
        token_final_scales=topk_w,
        num_experts=core.num_experts,
        top_k=core.topk,
        local_expert_offset=0,
        local_num_experts=core.num_experts,
        tile_tokens_dim=_CUTEDSL_TILE_SIZE,
    )
    h, h_sf = cute_dsl_nvfp4_gather_grouped_gemm_act_fusion(
        input=xq.view(torch.float4_e2m1fn_x2),
        weight=L.fc1_w,
        input_scale=xsf.view(torch.uint8).view(xq.shape[0], core.hidden // _SF_VEC),
        weight_scale=L.fc1_s,
        alpha=L.alpha1,
        tile_idx_to_group_idx=tile_expert,
        tile_idx_to_mn_limit=tile_limit,
        permuted_idx_to_expanded_idx=permuted_to_expanded,
        num_non_exiting_tiles=live_tiles,
        global_sf=L.global_sf,
        num_experts=core.num_experts,
        top_k=core.topk,
        num_local_experts=core.num_experts,
        local_expert_offset=0,
        tile_size=_CUTEDSL_TILE_SIZE,
        scaling_vector_size=_SF_VEC,
        activation_type=int(ActivationType.Swiglu),
    )
    # FC2 writes one bf16 row per permuted row.
    y = cute_dsl_nvfp4_grouped_gemm(
        input=h,
        weight=L.fc2_w,
        input_scale=h_sf,
        weight_scale=L.fc2_s,
        alpha=L.alpha2,
        tile_idx_to_group_idx=tile_expert,
        num_non_exiting_tiles=live_tiles,
        num_experts=core.num_experts,
        top_k=core.topk,
        num_local_experts=core.num_experts,
        local_expert_offset=0,
        tile_size=_CUTEDSL_TILE_SIZE,
        output_dtype=torch.bfloat16,
        scaling_vector_size=_SF_VEC,
    )
    return moe_unpermute_inplace(
        permuted_input=y,
        output=routed_out,
        expanded_idx_to_permuted_idx=expanded_to_permuted,
        topk_scales=topk_w,
    )


def _live_score_columns(core, md, batch: int) -> int:
    """A host-side bound on every row's selection window in a step with context rows: the
    longest sequence's compressed length, with the verify width as slack for the draft
    rows that sit past the host's KV lengths."""
    longest = int(md.kv_lens_runtime[:batch].max()) if batch > 0 else 0
    return (longest + _MAX_VERIFY_WIDTH) // core.compress_ratio + 1


def _replay_verify(core, mq_out, a_d, b_d, ssm_states, i, L, si_d, n, p, replay) -> torch.Tensor:
    """One GDN layer's verify recurrence under the cached replay: `n` generation sequences
    of `p` rows over the conv output `mq_out` (`[n * p, conv_dim]`, token-major) and the
    raw `a`/`b` projections, with the fp32 gating parameters fused into the kernel.

    `replay` is `(work_items, n_writes, partitioned)`; the work items are given exactly
    when the manager runs its all-layer commit after the step (without that commit every
    history wrap goes uncommitted, with it a wrap committed here is committed twice).
    `mq_out` is the conv's fresh, contiguous output, so the kernel reads q/k/v from it
    directly (`packed_qkv`) and never from the views. The conv before it does not signal
    dependents, so the kernel launches without PDL."""
    work_items, n_writes, partitioned = replay
    state = core._bound.replay
    old_x, old_B, old_dt, old_dA = state.old[i]
    qv, kv, vv = torch.split(mq_out, [core.key_dim, core.key_dim, core.value_dim], 1)
    out = fused_recurrent_gated_delta_rule_cached_replay_update(
        q=torch.reshape(qv, [n, p, core.num_k_heads, core.head_k_dim]),
        k=torch.reshape(kv, [n, p, core.num_k_heads, core.head_k_dim]),
        v=torch.reshape(vv, [n, p, core.num_v_heads, core.head_v_dim]),
        g=torch.reshape(a_d, [n, p, core.num_v_heads]),
        beta=torch.reshape(b_d, [n, p, core.num_v_heads]),
        ssm_states=ssm_states,
        state_indices=si_d,
        old_u=old_x,
        old_k=old_B,
        old_G=old_dt,
        old_beta=old_dA,
        cache_buf_idx=state.cache_buf_idx,
        prev_num_accepted_tokens=state.prev_num_accepted,
        history_size=state.history,
        scale=None,
        use_qk_l2norm_in_kernel=True,
        A_log=L.a_log,
        dt_bias=L.dt_bias,
        launch_with_pdl=False,
        replay_work_items=work_items,
        n_writes=n_writes,
        block_v=None,
        num_warps=None,
        use_l2_state_cache=None,
        use_l2_streaming_inputs=None,
        use_l2_shared_inputs=None,
        main_num_stages=None,
        packed_qkv=mq_out,
        use_all_layer_commit=partitioned,
        output=None,
    )
    return torch.reshape(out, [1, n * p, core.num_v_heads, core.head_v_dim])


def _replay_work(core, mm, num_decodes: int) -> tuple:
    """`_replay_verify`'s `replay` argument for a verify step of `num_decodes` generation
    requests: the partitioned replay exactly when `update_mamba_states` runs the manager's
    all-layer commit after the step."""
    partitioned = (
        core._bound.replay.all_layer_commit
        and num_decodes >= CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE
    )
    return (
        mm.replay_work_items[:num_decodes] if partitioned else None,
        mm.replay_n_writes if partitioned else None,
        partitioned,
    )


class _KvRangeProbe:
    """`TRTLLM_MODELING_V2_VALIDATE` over an fp8 KV pool: reports, once, the largest K/V
    magnitude the first eager forward's attention layers store.

    The pool takes K/V at the unit scale, so anything past the e4m3 maximum saturates
    silently (a bf16 inf included). One host sync per attention layer of that one forward;
    a forward under CUDA graph capture is passed over without a read."""

    def __init__(self, last_layer: int) -> None:
        self._last = last_layer
        self._worst = 0.0

    def observe(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> bool:
        """Fold one attention layer's K/V in; False once the forward's last layer has
        reported."""
        if torch.cuda.is_current_stream_capturing():
            return True
        peak = float(torch.maximum(k.abs().amax(), v.abs().amax()))
        # A NaN stays: `max` would drop it.
        self._worst = math.nan if math.isnan(peak + self._worst) else max(self._worst, peak)
        if layer != self._last:
            return True
        limit = torch.finfo(torch.float8_e4m3fn).max
        # Written so that NaN fails too.
        if not self._worst <= limit:
            logger.warning(
                f"FP8 KV cache: the first forward stores a K/V magnitude of {self._worst:g}, "
                f"past the e4m3 maximum {limit:g}; the pool is written at the unit scale, "
                "so those values saturate"
            )
        return False


class _PrefillStep(NamedTuple):
    """One step's host-side state, built by `PrefillTarget.prepare` and read by the forward's
    breaks.

    `num_tokens` is the batch's row count `n` and `physical_tokens` the `T` the forward runs
    over (`n`, or the prefill CUDA graph bucket it was padded to); `input_ids` is the batch's
    `n` ids. `attn` is thop_attention's per-step arguments; `gdn`, `ple` and `qsa` are the
    recurrent, PLE and QSA bookkeeping tuples their breaks unpack, each cut to the batch's
    rows; `stage` says the speculative rollback stages this verify step; `vision` is the
    shell's `(rows, mm_token_indices)` or None. A new one replaces it every step, so no
    break reads a previous step's."""

    num_tokens: int
    physical_tokens: int
    input_ids: torch.Tensor
    attn: dict
    gdn: tuple
    ple: tuple
    qsa: tuple
    stage: bool
    vision: tuple | None


class PrefillTarget:
    """Any batch with context rows (`num_contexts > 0`), possibly with generation rows
    beside them; the recurrent layers' decode-only branches never run here."""

    def __init__(self, core: Qwen4ExpTp1ModelingV2) -> None:
        """Take the per-layer weight views once; MTP operands are read off `core._mtp`."""
        self.core = core
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks; each target owns its flag (the
        # two `step_args` projections differ).
        self._contract_pending = step_contract_enabled()
        # Dropped once it has reported.
        self._kv_probe = (
            _KvRangeProbe(core.attn_layers[-1])
            if self._contract_pending and core.quant_mode == QM_FP8_KV_CACHE
            else None
        )
        self._layers = _layer_views(core)
        self._stack = _stack_views(core)
        # The step `prepare` built, read by the breaks.
        self._step: _PrefillStep | None = None

    def _check_step_contract(self, md) -> None:
        """Opt-in first-forward fail-fast (TRTLLM_MODELING_V2_VALIDATE): calling
        `step_args` surfaces an upstream metadata rename here, not mid-forward."""
        if not self._contract_pending:
            return
        try:
            self.step_args(md)
        except AttributeError as exc:
            raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
        self._contract_pending = False

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        """Project the prepared metadata onto thop_attention's batch state, once per
        forward. Tensors are engine-owned persistent buffers refreshed in place;
        Python ints are per-capture constants."""
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
            # No sliding window: the full admitted length on every attention layer.
            attention_window_size=md.max_seq_len,
            num_contexts=md.num_contexts,
            num_ctx_tokens=md.num_ctx_tokens,
            trtllm_gen_jit_warmup=md.trtllm_gen_jit_warmup,
            use_paged_context_fmha=md.use_paged_context_fmha,
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

    # --------------------------------------------------------------- blocks ---

    def _hc_mixer(self, normed, first, up, with_inject):
        """One hyper-connection mixer from the site's normed stream: the first
        projection over `[down; block_inject; 0]` (fp32 logits), the SiLU and
        injection activations, then the up projection with the sigmoid gate and
        the stream mean in its epilogue (the gate logits never leave the GEMM).
        Returns `(mixed, inject)`; `inject` is None without `with_inject` (the
        stack-final mixer)."""
        core = self.core
        low, inject = qwen4_exp_hc_lowrank_act(
            cublas_mm(normed, first[:, : core.hc_first_cols], None, torch.float32),
            core.hc_lowrank,
            core.hc_count,
            with_inject,
            normed.dtype,
        )
        return qwen4_exp_hc_up_gate_mean(low, up, normed, core.hc_count), inject

    def _hc_moe_mixer(self, normed, first, up, gscale):
        """`_hc_mixer` at an MoE site (always with injection) whose fused up
        projection also writes the routed experts' NVFP4 input (linear scales,
        the layer's global scale `gscale`). Returns `(mixed, inject, xq, xsf)`."""
        core = self.core
        low, inject = qwen4_exp_hc_lowrank_act(
            cublas_mm(normed, first[:, : core.hc_first_cols], None, torch.float32),
            core.hc_lowrank,
            core.hc_count,
            True,
            normed.dtype,
        )
        mixed, xq, xsf = qwen4_exp_hc_up_gate_mean_fp4(low, up, normed, core.hc_count, gscale)
        return mixed, inject, xq, xsf

    def _grouped_norm(self, x: torch.Tensor, weights: tuple) -> torch.Tensor:
        """`Qwen4ExpTextRMSNorm(group_size=hidden)`: an independent RMS per `hidden`-wide
        group, then `(1 + w)` over the full width; one gemma-RMS-norm per column
        slice (row stride certified by that entry) plus a concat."""
        core = self.core
        parts = torch.split(x, [core.hidden] * core.hc_count, 1)
        return torch.cat(
            [flashinfer_gemma_rmsnorm(part, w, core.eps) for w, part in zip(weights, parts)], 1
        )

    def _moe_block(self, x, xq, xsf, L: _LayerWeights, routed_out) -> tuple:
        """One MoE layer: NVFP4 routed experts plus a bf16 shared expert, left
        uncombined. Returns `(routed, (gate_logits, shared))`: the next
        `qwen4_exp_hc_inject_norm` forms `routed + sigmoid(gate_logits) * shared`
        inside its kernel.

        The shared expert and its per-token gate logit run on the core's side
        stream, concurrently with the routed path, and join it before this
        returns, so every consumer on the main stream is ordered after both.
        `x` is the bf16 hidden states (router and shared expert); `xq` / `xsf`
        are the same states in NVFP4 with **linear** block scales, written by the
        MoE-site mixer (`_hc_moe_mixer`). On trtllm-gen the routed path hands the
        runner the router logits (its Renormalize routing is bitwise the
        `renorm_moe_routing_op` + pre-routed composition), and a launch of at
        least `_CUTE_DSL_MOE_MIN_ROWS` rows runs the CuteDSL MoE over the second
        expert copy instead when the core keeps it (`moe_prefill_copy`); on CuteDSL
        every launch is `_cute_dsl_routed`, which writes `routed_out`."""
        core = self.core
        main = torch.cuda.current_stream()
        side = core._side_stream
        # Fork: the side stream reads `x` only after everything queued on `main`
        # so far. What it allocates is consumed on `main` after the join below,
        # and the next layer's fork orders any reuse of those blocks after that.
        side.wait_stream(main)
        with torch.cuda.stream(side):
            # One GEMM forms gate|up and the gate logit (`sh_gug`'s extra rows): a separate
            # gate GEMV at prefill sizes keeps the SMs from the routed path's persistent
            # FC1 until it drains.
            gug = cublas_mm(x, L.sh_gug)
            shared = cublas_mm(silu_and_mul_rows(gug, core.shared_inter), L.sh_dn)
            gate_logits = gug[:, 2 * core.shared_inter : 2 * core.shared_inter + 1]
        # The routing reads its logits as a dense row-major buffer from data_ptr()
        # and ignores strides; cublas_mm allocates them contiguous.
        logits = cublas_mm(x, L.router)
        if core.trunk_moe_backend == "CUTEDSL":
            routed = _cute_dsl_routed(core, L, logits, xq, xsf, routed_out)
        elif core.moe_prefill_copy and x.shape[0] >= _CUTE_DSL_MOE_MIN_ROWS:
            routed = fp4_moe_renorm_cute_dsl(
                logits,
                xq,
                xsf,
                L.fc1_w_cd,
                L.fc1_s_cd,
                L.alpha1_cd,
                L.fc2_w_cd,
                L.fc2_s_cd,
                L.alpha2_cd,
                L.global_sf_cd,
                core.num_experts,
                core.topk,
            )
        else:
            routed = fp4_block_scale_moe_renorm_routed(
                logits,
                xq,
                xsf.view(torch.float8_e4m3fn),
                L.fc1_w,
                L.fc1_s,
                L.fc2_w,
                L.fc2_s,
                L.out1_scale,
                L.out1_gate_scale,
                L.out2_scale,
                core.num_experts,
                core.topk,
                core.moe_inter,
            )
        main.wait_stream(side)
        return routed, (gate_logits, shared)

    def _attention_block(self, x, i, L, proj_qkv, index_scratch, attn_out, pos3):
        """One gated GQA layer plus its QSA index branch.

        The four input projections are one GEMM against the load-time fused
        operand (`weights._attn_in_w`), whose output row is
        `[q | k | v | gate | index q, k | v slot]`. One norm+rope launch rewrites
        the main q/k heads in place on that output and writes the normalized,
        rotated index query into the scratch; v, the gate and the raw index key
        are not touched. Both are row-wise and run over all `T` rows here; the
        cache writes, the selection and the attention core read the step's batch
        and run in the `_attention_core` break, which leaves the attention output
        in `attn_out`. The gate is read through its row stride."""
        core = self.core
        num_tokens = x.shape[0]

        proj = cublas_mm(x, L.attn_in)
        gate = torch.split(proj, [core.qkv_width, core.q_width, core.index_scratch_width], 1)[1]
        # The index query lands in the q slots of the scratch; its k and v slots
        # are layout only and nothing reads them.
        q_index = torch.split(
            torch.reshape(
                index_scratch,
                [num_tokens, core.index_heads + 2 * core.index_kv_heads, core.index_head_dim],
            ),
            [core.index_heads, core.index_kv_heads, core.index_kv_heads],
            1,
        )[0]
        # Both planes' Gemma norm + rope in one launch, over the target's fp32
        # cos|sin table: the main q/k heads in place (the real interleaved mRoPE;
        # three agreeing rows reduce exactly to the plain rope on row 0), the index
        # query from the projection's index columns into `q_index` (coordinate 0
        # alone, matching `_QSA_MROPE_SECTION` on the key side).
        index_section = (_QSA_MROPE_SECTION or (0, 0, 0)) if _QSA_INDEX_USE_MROPE else (0, 0, 0)
        qwen4_exp_qk_norm_rope(
            proj,
            q_index,
            pos3,
            core._cos_sin,
            L.q_norm,
            L.k_norm,
            L.idx_qn,
            num_heads_q=core.heads_q,
            num_heads_k=core.heads_kv,
            head_dim=core.head_dim,
            index_heads=core.index_heads,
            index_head_dim=core.index_head_dim,
            index_q_col=core.qkv_width + core.q_width,
            rotary_dim=core.rotary_dim,
            eps=core.eps,
            mrope_section1=core.mrope_section1,
            mrope_section2=core.mrope_section2,
            index_mrope_section1=index_section[1],
            index_mrope_section2=index_section[2],
        )
        if self._kv_probe is not None:
            _, k_dst, v_dst = _attn_in_heads(core, proj)
            if not self._kv_probe.observe(i, k_dst, v_dst):
                self._kv_probe = None
        eager_on_graph(proj, proj_qkv, index_scratch, attn_out, i, inner=self._attention_core)
        fused_sigmoid_mul_inplace(attn_out, gate)
        return cublas_mm(attn_out, L.o)

    def _attention_core(self, proj, proj_qkv, index_scratch, attn_out, i) -> None:
        """Break: the index plane's store and compression, the dense/sparse decision and the
        attention core over the step's `n` rows. Writes rows `[:n]` of `attn_out`
        `[T, q_width]` in place and zeroes rows `[n:]`. `proj` is the layer's fused
        projection output after the norm+rope, `proj_qkv` the dense op's packed QKV
        scratch.

        The index plane advances on every step, including below the threshold, so a
        request growing past it finds its prefix compressed; above it the main K/V append
        becomes this layer's own call. The raw index key is stored before any
        norm/rotation: the compression step applies both itself, so a pre-normalized key
        is double-normalized (invisible below the sparse threshold).

        On a verify step (`stage`) the generation rows' index slots (and, on the first
        attention layer, their position slots) are snapshotted for `SpeculativeRollback`
        before the store and compression overwrite them."""
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
            live_cols,
        ) = step.qsa
        num_tokens = step.num_tokens
        proj = proj[:num_tokens]
        qkv_view, _, index_view = torch.split(
            proj, [core.qkv_width, core.q_width, core.index_scratch_width], 1
        )
        q_dst, k_dst, v_dst = _attn_in_heads(core, proj)

        index_cache = core._bound.index_cache[i]
        token_k = torch.split(
            torch.reshape(
                index_view,
                [num_tokens, core.index_scratch_width // core.index_head_dim, core.index_head_dim],
            ),
            [core.index_heads, core.index_kv_heads, 1],
            1,
        )[1]
        if step.stage:
            core.spec_rollback.stage_index(index_cache, core._bound.position_cache)
        qsa_paged_index_store(
            token_k,
            coords,
            index_cache,
            core._bound.position_cache,
            req_idx,
            logical,
            block_table,
            tokens_per_block,
        )
        q_index = torch.split(
            torch.reshape(
                index_scratch[:num_tokens],
                [num_tokens, core.index_heads + 2 * core.index_kv_heads, core.index_head_dim],
            ),
            [core.index_heads, core.index_kv_heads, core.index_kv_heads],
            1,
        )[0]
        qsa_prefill_compress(
            logical_positions=logical,
            request_indices=req_idx,
            block_table=block_table,
            index_cache=index_cache,
            position_cache=core._bound.position_cache,
            k_norm_weight=L.idx_kn,
            cos_sin=core._cos_sin,
            eps=core.eps,
            tokens_per_block=tokens_per_block,
            compress_ratio=core.compress_ratio,
            mrope_section=_QSA_MROPE_SECTION,
        )

        if sparse:
            kv = core._bound.kv[i]
            # Append the fused-norm+rope'd K/V here: nothing else writes the pool on
            # this path, and selection reads the current token from it. Over an fp8 pool the
            # store and the sparse GQA below take e4m3 unscaled, as the dense op does at the
            # unit KV scales; over an NVFP4 pool they quantize and dequantize at the dense op's
            # second-level scales.
            _store_kv(core, kv, k_dst, v_dst, req_idx, logical, block_table, tokens_per_block)
            # The step's context rows come first, consecutive rows of a request, and score
            # as query tiles; its generation rows, one a request, score row by row (a query
            # tile of them would walk every request's keys once per row). Each part selects
            # into its own rows of the top-k.
            num_ctx = step.attn["num_ctx_tokens"]
            parts = [(0, num_ctx, True)]
            if num_ctx < num_tokens:
                parts.append((num_ctx, num_tokens, False))
            # Each part's canonical block top-k and its expansion are one launch writing its
            # rows of the step's selection (QSA's row starts are zero); a score table that
            # kernel does not serve (wider than its 4096 columns) takes the composition.
            selected = torch.empty(
                [num_tokens, core.token_topk + core.compress_ratio - 1],
                dtype=torch.int32,
                device=proj.device,
            )
            for lo, hi, context_rows in parts:
                scores = qsa_paged_index_scores(
                    q=q_index[lo:hi],
                    index_cache=index_cache,
                    block_table=block_table,
                    query_positions=logical[lo:hi],
                    request_indices=req_idx[lo:hi],
                    tokens_per_block=tokens_per_block,
                    compress_ratio=core.compress_ratio,
                    only_visible_blocks=True,
                    context_rows=context_rows,
                )
                if _topk_expand_supports(scores):
                    qsa_topk_expand_decode(
                        scores,
                        visible_blocks[lo:hi],
                        logical[lo:hi],
                        seq_lens_q[lo:hi],
                        topk_indices[lo:hi],
                        core.compress_ratio,
                        core.token_topk,
                        out=selected[lo:hi],
                    )
                else:
                    _select(
                        core,
                        scores,
                        topk_row_starts[lo:hi],
                        visible_blocks[lo:hi],
                        topk_indices[lo:hi],
                        live_cols,
                    )
                    selected[lo:hi].copy_(
                        qsa_expand_block_indices(
                            topk_indices[lo:hi],
                            logical[lo:hi],
                            seq_lens_q[lo:hi],
                            compress_ratio=core.compress_ratio,
                            token_topk=core.token_topk,
                        )
                    )
            # A context launch of thousands of rows over a bf16 pool takes the row-tile
            # pass: consecutive rows of one request share their K/V tiles; rows of groups
            # that span requests, the generation rows of a mixed step among them, are served
            # row by row. Below the threshold the per-row kernel is faster.
            if num_tokens >= _ROWTILE_MIN_ROWS and kv.k.dtype == q_dst.dtype:
                out = qsa_sparse_gqa_rowtile(
                    q=q_dst,
                    k_cache=kv.k,
                    v_cache=kv.v,
                    block_table=block_table,
                    selected_tokens=selected,
                    request_indices=req_idx,
                    tokens_per_block=tokens_per_block,
                    softmax_scale=core.softmax_scale,
                    query_positions=logical,
                    compress_ratio=core.compress_ratio,
                )
            else:
                out = _sparse_gqa(
                    core,
                    kv,
                    q=q_dst,
                    block_table=block_table,
                    selected_tokens=selected,
                    request_indices=req_idx,
                    tokens_per_block=tokens_per_block,
                    # A bound, not a mask: lets the kernel stop scanning a row early
                    # (context-batch branch only).
                    query_positions=logical,
                    compress_ratio=core.compress_ratio,
                )
            attn_out[:num_tokens].copy_(torch.reshape(out, [num_tokens, core.q_width]))
        else:
            # The dense op reads its packed QKV flat: one compact copy.
            qkv = proj_qkv[:num_tokens]
            qkv.copy_(qkv_view)
            thop_attention(
                q=qkv,
                output=attn_out[:num_tokens],
                local_layer_idx=core._bound.pool_idx[i],
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

    def _gdn_verify(
        self, mq_d, a_d, b_d, ssm_states, i, L, si_d, int_idx, n, p, replay
    ) -> torch.Tensor:
        """The recurrent half of a speculative verify step: `n` generation sequences of
        `p = runtime_draft_len + 1` tokens, written so that the accepted prefix, unknown
        until the sampler has run, can be kept and the rest dropped.

        The conv writes each row's window to the runtime's per-step buffer and leaves the
        conv pool alone; `update_mamba_states` promotes window `k - 1`.
        `causal_conv1d_triton_update` is the only conv with per-step snapshots (its result
        for a pad-sentinel slot is uninitialized, safe because every row is a real
        sequence). The recurrence has two forms, bound per manager
        (`_bind_step_buffers`):

        * cached replay (`replay` given): the pool row is a checkpoint; one call rebuilds
          the current state from it and the slot's pending history, runs the `p` rows and
          appends them to the history. Rejected rows cost nothing to undo, because the
          manager advances the pending count by the accepted tokens only. A full history
          advances the checkpoint inside the kernel below
          `CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE` requests and in the manager's
          all-layer commit from there on, hence `replay` carries the work items exactly
          under the manager's own condition.
        * legacy (`replay` None): every row's recurrent state goes to the per-step buffer
          instead of the pool, and the promotion copies step `k - 1` into it. Advancing
          the pool directly would leave rejected tokens' state in every recurrent layer,
          invisible to the acceptance rate. `pack_gdn_decode_qkv` does no Q/K L2 norm,
          hence `use_qk_l2norm_in_kernel=True` below."""
        core = self.core
        # [n*p, C] -> [n, C, p]: the multi-token conv reads channel-major; its cells
        # certify this transposed view directly.
        x3 = torch.transpose(torch.reshape(mq_d, [n, p, core.conv_dim]), 1, 2)
        conv_out = causal_conv1d_triton_update(
            x=x3,
            conv_state=core._bound.conv[i],
            weight=L.conv_w,
            bias=None,
            activation="silu",
            conv_state_indices=si_d,
            pad_slot_id=_PAD_SLOT_ID,
            # Snapshots, not the accepted-token roll.
            num_accepted_tokens=None,
            intermediate_conv_window=core._bound.int_conv[i],
            intermediate_state_indices=int_idx,
            launch_dependent_kernels=False,
        )
        # Back to token-major (a view).
        mq_out = torch.reshape(torch.transpose(conv_out, 1, 2), [n * p, core.conv_dim])
        if replay is not None:
            return _replay_verify(core, mq_out, a_d, b_d, ssm_states, i, L, si_d, n, p, replay)
        q, k, v = pack_gdn_decode_qkv(
            mq_out, core.num_k_heads, core.head_k_dim, core.num_v_heads, core.head_v_dim
        )
        g = torch.reshape(
            fused_gdn_gating(
                A_log=L.a_log,
                a=a_d,
                dt_bias=L.dt_bias,
                beta=_SOFTPLUS_BETA,
                threshold=_SOFTPLUS_THRESHOLD,
            ),
            [n, p, core.num_v_heads],
        )
        beta = torch.reshape(torch.sigmoid(b_d), [n, p, core.num_v_heads])
        # Initial states are gathered from the live pool into a batch-scoped copy. The
        # per-step buffer is indexed by batch row (`intermediate_state_indices[:n]`),
        # not slot id: slot ids would write rows nothing promotes, or past the end.
        src = torch.reshape(
            nn.functional.embedding(si_d, torch.reshape(ssm_states, [ssm_states.shape[0], -1])),
            [n, core.num_v_heads, core.head_v_dim, core.head_k_dim],
        )
        out = fused_recurrent_gated_delta_rule_update(
            torch.reshape(q, [n, p, core.num_k_heads, core.head_k_dim]),
            torch.reshape(k, [n, p, core.num_k_heads, core.head_k_dim]),
            torch.reshape(v, [n, p, core.num_v_heads, core.head_v_dim]),
            g,
            beta,
            initial_state_source=src,
            initial_state_indices=int_idx,
            scale=None,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=None,
            disable_state_update=True,
            disable_output_calculation=False,
            intermediate_states_buffer=core._bound.int_ssm[i],
            cache_steps=p,
            output=None,
        )
        return torch.reshape(out, [1, n * p, core.num_v_heads, core.head_v_dim])

    def _gdn_block(self, x, i, L, core_out) -> torch.Tensor:
        """One Gated-DeltaNet layer over a batch with a context prefix: the one input
        projection GEMM (q|k|v, z, a and b are column views of its output) and the
        gated-norm epilogue run over all `T` rows here, the recurrence in the `_gdn_core`
        break, which leaves its output in `core_out` `[1, T, num_v_heads, head_v_dim]`."""
        core = self.core
        proj = cublas_mm(x, L.in_proj)
        z = torch.split(proj, core.in_proj_split, 1)[1]
        eager_on_graph(proj, core_out, i, inner=self._gdn_core)
        return self._gdn_epilogue(core_out, z, L, x.shape[0])

    def _gdn_core(self, proj, out, i) -> None:
        """Break: the recurrence over the step's `n` rows. Writes rows `[:n]` of `out`
        `[1, T, num_v_heads, head_v_dim]` in place and zeroes rows `[n:]`. Every branch
        is on a host int of the step, never a tensor value.

        Over an fp32 recurrent pool outside a verify step the context rows take one fused
        conv + post-conv pass (`_gdn_prefill_prep`) and FlashInfer's chunked recurrence, and
        a mixed step's one-token generation rows FlashInfer's single-token decode kernel;
        otherwise `_gdn_core_generic`."""
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
            is_verify,
            num_decodes,
            gen_p,
            int_idx,
            qsl_long_p,
            replay,
        ) = step.gdn
        num_tokens = step.num_tokens
        ssm_states = core._bound.ssm[i]
        if is_verify or ssm_states.dtype != torch.float32:
            self._gdn_core_generic(proj, out, i)
            return
        mixed_qkv, _, a, b = torch.split(proj[:num_tokens], core.in_proj_split, 1)
        conv_states = core._bound.conv[i]

        # A context sequence carrying nothing starts from a zeroed slot; only the
        # context prefix is a candidate (a generation row's state must survive).
        reset_recurrent_state_rows(ssm_states, si_p, hi_p, conv_states)
        if num_decode_tokens > 0:
            mq_p, mq_d = torch.split(mixed_qkv, [num_prefill_tokens, num_decode_tokens], 0)
            a_p, b_p = a[:num_prefill_tokens], b[:num_prefill_tokens]
            q, k, v, g, beta = self._gdn_prefill_prep(mq_p, a_p, b_p, L, i, qsl_p, si_p, hi_p)
            causal_conv1d_update(
                x=mq_d,
                conv_state=conv_states,
                weight=L.conv_w,
                bias=None,
                silu_activation=True,
                cache_seqlens=None,
                conv_state_indices=si_d,
                pad_slot_id=_PAD_SLOT_ID,
            )
            # The two halves run separate recurrences over disjoint slots and write
            # disjoint row ranges of the output.
            out_p = out[:, :num_prefill_tokens]
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
                output=out_p,
                state_workspace=None,
            )
            # The generation rows: the decode step's recurrence (its kernel without the
            # gated norm, which runs with the context rows' below) over their rows of the
            # projection, whose q/k/v columns the conv step rewrote; it computes both gates
            # and the q/k norm itself and reads the rows' pool slots before its PDL wait
            # (the context rows' reset and recurrence touch only theirs).
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
            )
        out[:, num_tokens:].zero_()

    def _gdn_core_generic(self, proj, out, i) -> None:
        """`_gdn_core` on a verify step or over a bf16 recurrent pool: the conv, the
        post-conv gating and the chunked delta rule over the step's `n` rows, or under
        speculation the verify path for the generation rows. The q|k|v columns are read in
        place, as the column view of the projection every consumer here is certified on;
        the a and b columns are copied out first, as the verify path reads them contiguous.
        The chunked delta rule writes its rows of `out` directly."""
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
            is_verify,
            num_decodes,
            gen_p,
            int_idx,
            qsl_long_p,
            replay,
        ) = step.gdn
        num_tokens = step.num_tokens
        mixed_qkv, _, a, b = torch.split(proj[:num_tokens], core.in_proj_split, 1)
        a, b = a.contiguous(), b.contiguous()

        conv_states = core._bound.conv[i]
        ssm_states = core._bound.ssm[i]

        # A context sequence carrying nothing starts from a zeroed slot; only the
        # context prefix is a candidate (a generation row's state must survive).
        reset_recurrent_state_rows(ssm_states, si_p, hi_p, conv_states)
        if num_decode_tokens > 0:
            mq_p, mq_d = torch.split(mixed_qkv, [num_prefill_tokens, num_decode_tokens], 0)
            # Prefill conv wants channel-major, decode token-major.
            mq_p_t = extract_transpose_prefill_slice(mq_p, num_prefill_tokens, 0, core.conv_dim)
            causal_conv1d_fwd(
                x=mq_p_t,
                weight=L.conv_w,
                bias=None,
                conv_states=conv_states,
                query_start_loc=qsl_p,
                cache_indices=si_p,
                has_initial_state=hi_p,
                silu_activation=True,
                pad_slot_id=_PAD_SLOT_ID,
                out=None,
            )
            if is_verify:
                # A context row advances its slot as on the base path; a generation
                # row is a draft chain of unknown accepted length, so it goes to the
                # rollback buffers.
                a_p, a_d = torch.split(a, [num_prefill_tokens, num_decode_tokens], 0)
                b_p, b_d = torch.split(b, [num_prefill_tokens, num_decode_tokens], 0)
                q, k, v, g, beta = fused_gdn_post_conv(
                    prefill=mq_p_t,
                    decode=None,
                    a=a_p,
                    b=b_p,
                    A_log=L.a_log,
                    dt_bias=L.dt_bias,
                    num_k_heads=core.num_k_heads,
                    head_k_dim=core.head_k_dim,
                    num_v_heads=core.num_v_heads,
                    head_v_dim=core.head_v_dim,
                    l2_norm_eps=_L2_NORM_EPS,
                    softplus_threshold=_SOFTPLUS_THRESHOLD,
                )
                # q and k are already L2-normalized by the post-conv call.
                chunk_gated_delta_rule(
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
                    use_qk_l2norm_in_kernel=False,
                    output=out[:, :num_prefill_tokens],
                )
                out_d = self._gdn_verify(
                    mq_d, a_d, b_d, ssm_states, i, L, si_d, int_idx, num_decodes, gen_p, replay
                )
                out[:, num_prefill_tokens:num_tokens].copy_(out_d)
                out[:, num_tokens:].zero_()
                return
            causal_conv1d_update(
                x=mq_d,
                conv_state=conv_states,
                weight=L.conv_w,
                bias=None,
                silu_activation=True,
                cache_seqlens=None,
                conv_state_indices=si_d,
                pad_slot_id=_PAD_SLOT_ID,
            )
            q, k, v, g, beta = fused_gdn_post_conv(
                prefill=mq_p_t,
                decode=mq_d,
                a=a,
                b=b,
                A_log=L.a_log,
                dt_bias=L.dt_bias,
                num_k_heads=core.num_k_heads,
                head_k_dim=core.head_k_dim,
                num_v_heads=core.num_v_heads,
                head_v_dim=core.head_v_dim,
                l2_norm_eps=_L2_NORM_EPS,
                softplus_threshold=_SOFTPLUS_THRESHOLD,
            )
        else:
            mq_t = extract_transpose_prefill_slice(mixed_qkv, num_tokens, 0, core.conv_dim)
            causal_conv1d_fwd(
                x=mq_t,
                weight=L.conv_w,
                bias=None,
                conv_states=conv_states,
                query_start_loc=qsl_all,
                cache_indices=si_all,
                has_initial_state=hi_all,
                silu_activation=True,
                pad_slot_id=_PAD_SLOT_ID,
                out=None,
            )
            q, k, v, g, beta = fused_gdn_post_conv(
                prefill=mq_t,
                decode=None,
                a=a,
                b=b,
                A_log=L.a_log,
                dt_bias=L.dt_bias,
                num_k_heads=core.num_k_heads,
                head_k_dim=core.head_k_dim,
                num_v_heads=core.num_v_heads,
                head_v_dim=core.head_v_dim,
                l2_norm_eps=_L2_NORM_EPS,
                softplus_threshold=_SOFTPLUS_THRESHOLD,
            )
        # q and k are already L2-normalized; the recurrence must not repeat it.
        chunk_gated_delta_rule(
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
            use_qk_l2norm_in_kernel=False,
            output=out[:, :num_tokens],
        )
        out[:, num_tokens:].zero_()

    def _gdn_prefill_prep(self, mq, a, b, L, i, qsl, si, hi):
        """The context rows' recurrence inputs: the causal conv against each sequence's
        carried window (published back to the pool), SiLU, the q|k|v split with q/k
        L2-normalized, and the decay/beta gates -- one pass over `mq`, the token-major
        q|k|v column view of the projection."""
        core = self.core
        return gdn_prefill_conv_post(
            x=mq,
            weight=L.conv_w,
            conv_states=core._bound.conv[i],
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
        )

    def _gdn_epilogue(self, out, z, L, num_tokens) -> torch.Tensor:
        """Gated RMS norm and output projection shared by every recurrence."""
        core = self.core
        # Sigmoid gate and a plain (non-gemma) weight (`output_gate_type`).
        o = rms_norm_gated_token_major(
            torch.reshape(out, [-1, core.head_v_dim]),
            torch.reshape(z, [num_tokens, core.num_v_heads, core.head_v_dim]),
            weight=L.gnorm,
            eps=core.eps,
            fp8_scale=None,
            gate_activation="sigmoid",
        )
        return cublas_mm(torch.reshape(o, [num_tokens, core.value_dim]), L.out)

    def _ple_block(self, hc, hc_out, normed_out) -> None:
        """Break: the n-gram side path on one decoder layer, over the step's `n` rows,
        added into the stream and normalized for the layer's attention-side site. Writes
        rows `[:n]` of `hc_out` (the stream) and `normed_out` (its grouped norm), both
        `[T, stream]`, in place and zeroes rows `[n:]`.

        Per-request stateful: ids become one `ngram_size`-wide window per token
        (predecessors from the paged id history); each window hashes to one row per
        head; rows gather fp8 vectors from one table and dequantize by one per-tensor
        scale; key/value projections and the gate scale one value into each stream;
        a dilated depthwise short conv over a paged state is added back. A sequence
        carrying nothing is seeded with `eos_token_id`, not 0: id 0 is a real token
        the hash cannot distinguish. The id history and the conv state publish the
        state after the whole chunk.

        Outside a verify step the window, hash, gather and dequantize are
        `ple_ngram_embed`, and the gate, the conv, the residual add and the next site's
        grouped norm `ple_gate_conv_inject_norm`. On a verify step (`stage`) the
        generation rows' pre-step pool rows, id windows and conv input go to
        `SpeculativeRollback`, which rewrites both pools to the accepted prefix once the
        sampler has decided it, so that step takes `_ple_delta`, which exposes them."""
        core = self.core
        S = self._stack
        step = self._step
        conv_states, ngram_pool, si_all, carry_all, qsl = step.ple
        num_tokens = step.num_tokens
        next_norm = self._layers[core.ple_layer].ahc_norm
        hc = hc[:num_tokens]
        if step.stage:
            stream = torch.add(hc, self._ple_delta(hc))
            stream, normed = qwen4_exp_hc_inject_norm(
                stream, None, None, next_norm, core.eps, core.hc_count
            )
        else:
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
            )
            stream, normed = ple_gate_conv_inject_norm(
                hc,
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
            )
        hc_out[:num_tokens].copy_(stream)
        normed_out[:num_tokens].copy_(normed)
        hc_out[num_tokens:].zero_()
        normed_out[num_tokens:].zero_()

    def _ple_delta(self, hc) -> torch.Tensor:
        """A verify step's PLE output for the batch's `n` stream rows `hc`, before it is
        added into the stream, through the unfused ops whose intermediates
        `SpeculativeRollback` stages: the pre-step pool rows, the id windows and the conv
        input."""
        core = self.core
        S = self._stack
        step = self._step
        conv_states, ngram_pool, si_all, carry_all, qsl = step.ple
        core.spec_rollback.stage_ple_pools(ngram_pool, conv_states)
        window = ngram_token_history(
            step.input_ids, ngram_pool, si_all, carry_all, qsl, core.eos_token_id, core.ngram_size
        )
        rows = torch.reshape(
            ngram_hash(
                window,
                layer_multipliers=S.ple_mult,
                head_vocab_sizes=S.ple_vocab,
                head_offsets=S.ple_offs,
                eos_token_id=core.eos_token_id,
                num_query_tokens=1,
            ),
            [-1, core.ngram_heads],
        )
        gathered = nn.functional.embedding(rows, core.w["ple_table"])
        ngram = torch.reshape(
            dequantize_e4m3_per_tensor(gathered, S.ple_tscale), [-1, core.ple_embed_dim]
        )

        key_normed = self._grouped_norm(cublas_mm(ngram, S.ple_key), S.ple_nk)
        query_normed = self._grouped_norm(hc, S.ple_nq)
        value = cublas_mm(ngram, S.ple_val)
        gated = gated_ngram_injection(
            key_normed, query_normed, value, core.hc_count, _PLE_CLAMP_MIN
        )
        conv_in = self._grouped_norm(gated, S.ple_nc)
        core.spec_rollback.stage_ple_rows(window, conv_in)
        # The short conv reads channel-major; a bare transpose is the certified form.
        convolved = dilated_short_conv_paged(
            x=torch.transpose(conv_in, 0, 1),
            weight=S.ple_conv_w,
            bias=None,
            conv_states=conv_states,
            query_start_loc=qsl,
            cache_indices=si_all,
            has_initial_state=carry_all,
            dilation=core.ple_dilation,
            silu_activation=True,
            pad_slot_id=_PAD_SLOT_ID,
        )
        return torch.add(gated, torch.transpose(convolved, 0, 1))

    def _splice(self, h) -> None:
        """Break: write the step's vision rows over their placeholder rows of the token
        embedding `h` `[T, hidden]` in place, and zero rows `[n:]`.

        `mm_token_indices` names every placeholder row of the packed batch in the order the
        per-request rows concatenate in. An indexed write, not a slice copy: two images in
        one prompt are separated by framing tokens, so the placeholder rows are several
        disjoint runs. A text step writes nothing but the padding. The rows are the step's:
        a replayed graph never sees the previous step's."""
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
        spec_metadata,
    ) -> None:
        """Everything this step reads from the host, kept for the forward's breaks as one
        `_PrefillStep`: the contract check, the attention op's per-step arguments and the
        batch's bookkeeping, built once and shared by every layer.

        Called by the core's `begin_step` on every step with context rows, before the
        forward. On a step a breakable prefill graph replays, this and the breaks are all of
        this target that runs."""
        core = self.core
        md = attn_metadata
        self._check_step_contract(md)
        attn_args = self.step_args(md)
        mm = md.mamba_metadata

        # **The activation rows are the batch's rows, or a prefill CUDA graph bucket's.** A
        # breakable prefill graph pads the step's rows up to `padded_num_tokens` (zero ids,
        # zero positions), and so does a padded step that runs eagerly; the forward runs over
        # all of them and every break slices the batch's own rows back out. Anything else is
        # an activation buffer longer than the batch, which would push padding rows through
        # the recurrences into the state pools.
        physical_tokens = input_ids.shape[0]
        assert physical_tokens in (md.num_tokens, md.padded_num_tokens), (
            f"the engine handed {physical_tokens} activation rows for a batch its "
            f"metadata describes as {md.num_tokens} tokens "
            f"(padded_num_tokens={md.padded_num_tokens})"
        )

        # The batch's bookkeeping, built once and shared by every layer.
        num_tokens = md.num_tokens
        batch = md.num_seqs
        num_prefills = md.num_contexts
        num_decodes = batch - num_prefills
        num_prefill_tokens = md.num_ctx_tokens
        num_decode_tokens = num_tokens - num_prefill_tokens
        # Rows per generation sequence: 1 without speculation, else
        # `runtime_draft_len + 1`; read from the metadata because the worker rewrites
        # it between draft steps.
        gen_p = num_decode_tokens // num_decodes if num_decodes else 1
        assert gen_p * num_decodes == num_decode_tokens, (
            f"{num_decode_tokens} generation rows do not divide over "
            f"{num_decodes} generation sequences; a ragged per-sequence draft "
            "length can be expressed neither to the attention call nor to the "
            "multi-token recurrence"
        )
        assert core.mtp_enabled or gen_p == 1, (
            f"{num_decode_tokens} generation rows over {num_decodes} generation "
            "sequences with speculation off; the base path implements one query "
            "token per generation sequence, and both the attention call's "
            "within-block mask and the recurrent path's cumulative-length "
            "vector assume it"
        )
        # `gen_p > 1` is load-bearing: verify parks a generation row in the rollback
        # buffers, which `update_mamba_states` promotes only if the worker drafts.
        # With `runtime_draft_len == 0` (a `draft_len_schedule` can do this) it skips
        # drafting before the promotion, leaving every recurrent layer one token
        # stale; that row is the always-accepted golden token, so the ordinary decode
        # path is right.
        if spec_metadata is not None and num_decodes > 0:
            declared_p = spec_metadata.runtime_draft_len + 1
            assert gen_p == declared_p, (
                f"the batch carries {gen_p} query rows per generation sequence "
                f"but the worker's runtime_draft_len says {declared_p}; the "
                "per-step conv snapshots and the recurrent rollback buffer are "
                "sized from the second and written from the first"
            )
            # Per-iteration counterpart of the core's `max_draft_len` bound.
            check_verify_width(gen_p)
            check_replay_width(core._bound.replay, gen_p)
        is_verify = spec_metadata is not None and num_decodes > 0 and gen_p > 1
        attn_args["predicted_tokens_per_seq"] = gen_p

        si_all = mm.state_indices[:batch]
        hi_all = mm.has_initial_states[:batch]
        si_p, si_d = torch.split(si_all, [num_prefills, num_decodes], 0)
        # The single-token decode kernel takes its slot vector through a 16-byte-aligned
        # pointer, and the generation half of the int32 slot vector starts `num_prefills`
        # rows in, aligned only when that count is a multiple of four: one copy per step
        # makes it aligned for every layer.
        si_d = torch.clone(si_d)
        hi_p = hi_all[:num_prefills]
        # `query_start_loc` covers the context prefix; `_long` the whole batch, and its
        # prefix the context rows' chunked recurrence when generation rows run beside them.
        qsl_all = mm.query_start_loc
        qsl_p = None if qsl_all is None else qsl_all[: num_prefills + 1]
        qsl_long = mm.query_start_loc_long
        qsl_long_p = qsl_long[: num_prefills + 1]
        int_idx = None
        replay = None
        if is_verify:
            # The runtime's arange over the slot pool, from the cache manager (what
            # `update_mamba_states` reads).
            assert core._bound.int_idx is not None, "the cache-manager bind has not run"
            int_idx = core._bound.int_idx[:num_decodes]
            if core._bound.replay is not None:
                replay = _replay_work(core, mm, num_decodes)
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
            is_verify,
            num_decodes,
            gen_p,
            int_idx,
            qsl_long_p,
            replay,
        )

        # PLE carry flags must cover the whole batch, but `has_initial_states` is
        # filled for the context prefix only (generation entries are stale); every
        # generation sequence carries, so pad with True (host-int sized, graph-safe).
        ple_carry = nn.functional.pad(hi_p, [0, num_decodes], mode="constant", value=1.0)
        # Under speculation `Mamba2Metadata.prepare`'s `query_start_loc_long` has one
        # row per generation sequence, so with `gen_p > 1` it is short and both PLE
        # stateful calls are silently mis-segmented; rebuild it from `seq_lens_cuda`,
        # which still holds the true counts during the trunk forward.
        if gen_p == 1:
            ple_qsl = qsl_long
        else:
            ple_qsl = cu_seqlens_from_seq_lens(md.seq_lens_cuda[:batch], torch.int64)
        ple_state = (core._bound.ple_state[0], core._bound.ple_state[1], si_all, ple_carry, ple_qsl)

        # Dense/sparse decision once per forward, from a host-side metadata read (as
        # QSA's backend does): a per-capture constant, matching the engine's separate
        # short/long decode graph families.
        sparse = batch > 0 and int(md.kv_lens_runtime[:batch].max()) > core.dense_threshold

        # The `[T, 3]` transpose of the `[3, 1, T]` position buffer, which the QSA position
        # plane stores.
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
            _live_score_columns(core, md, batch),
        )

        # A verify step's generation rows write the PLE pools and the QSA side caches for
        # every draft row, accepted or not; `SpeculativeRollback` keeps what its commit
        # needs to undo the rejected ones. Reopened on every forward, so a step whose
        # commit never ran (a failed forward) leaves nothing for the next one. Context
        # rows are never staged.
        stage = is_verify
        rollback = core.spec_rollback
        if rollback is not None:
            rollback.discard()
        if stage:
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

        self._step = _PrefillStep(
            num_tokens=num_tokens,
            physical_tokens=physical_tokens,
            input_ids=input_ids[:num_tokens],
            attn=attn_args,
            gdn=gdn_state,
            ple=ple_state,
            qsa=qsa,
            stage=stage,
            vision=vision,
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor,
        position_ids: torch.IntTensor,
        **kwargs,
    ):
        """The step `prepare` built, over the `T` activation rows the engine handed in: the
        batch's `n` rows, or the prefill CUDA graph bucket they were padded to.

        **This is the body a breakable prefill CUDA graph captures**, once per bucket, and
        replays for every step it pads to that bucket without running this Python. So
        nothing here reads the step: every op is row-wise and sized by `T` alone, its inputs
        are the engine's persistent `input_ids` / `position_ids` buffers, and every buffer is
        allocated here. What depends on the step's batch runs in one of four breaks that
        run eagerly between the captured segments and read `self._step`: `_splice` (once),
        `_ple_block` (once), `_attention_core` (every attention layer) and `_gdn_core` (every
        Gated-DeltaNet layer). Each writes rows `[:n]` of a fixed output allocated here and
        zeroes rows `[n:]`, so the padding rows stay finite through the row-wise ops and
        reach neither a state pool nor a real row. With the backend off every step is
        eager and unpadded, and the breaks are plain calls.

        Every hyper-connection site is `qwen4_exp_hc_inject_norm` (the previous block's
        injection fused with this site's grouped norm; after a MoE block it also forms that
        block's output) and then the mixer (`_hc_mixer`, `_hc_moe_mixer`)."""
        core = self.core
        num_tokens = input_ids.shape[0]
        assert num_tokens == self._step.physical_tokens, (
            f"the forward got {num_tokens} activation rows and prepare saw "
            f"{self._step.physical_tokens}"
        )

        # `pos3`: contiguous `[3, T]` for `use_mrope=True` (the engine hands a slice of a
        # larger buffer, whose row stride the ops reject, hence the copy).
        pos3 = torch.empty(
            [_QSA_POSITION_AXES, position_ids.shape[-1]],
            dtype=torch.int32,
            device=position_ids.device,
        )
        pos3.copy_(torch.reshape(position_ids, [_QSA_POSITION_AXES, -1]))

        # The token embedding with this step's vision rows spliced over their placeholders.
        h = nn.functional.embedding(input_ids, core.w["embed"])
        eager_on_graph(h, inner=self._splice)
        dt = h.dtype
        dev = h.device

        # Scratch shared by every layer that uses it; each break rewrites its whole output.
        proj_qkv = torch.empty([num_tokens, core.qkv_width], dtype=dt, device=dev)
        index_scratch = torch.empty([num_tokens, core.index_scratch_width], dtype=dt, device=dev)
        attn_out = torch.empty([num_tokens, core.q_width], dtype=dt, device=dev)
        gdn_out = torch.empty(
            [1, num_tokens, core.num_v_heads, core.head_v_dim], dtype=dt, device=dev
        )
        ple_hc = torch.empty([num_tokens, core.stream], dtype=dt, device=dev)
        ple_normed = torch.empty([num_tokens, core.stream], dtype=dt, device=dev)
        # The CuteDSL routed output, shared by all layers: each layer's unpermute rewrites
        # every row, and the next site's injection consumes it before the next layer's MoE.
        routed_out = (
            torch.empty([num_tokens, core.hidden], dtype=dt, device=dev)
            if core.trunk_moe_backend == "CUTEDSL"
            else None
        )

        # The embedding repeated across the hc_count residual streams.
        hc_count = core.hc_count
        hc = torch.cat([h] * hc_count, -1)
        layers, S = self._layers, self._stack
        _, normed = qwen4_exp_hc_inject_norm(hc, None, None, layers[0].ahc_norm, core.eps, hc_count)
        for i, L in enumerate(layers):
            if i == core.ple_layer:
                eager_on_graph(hc, ple_hc, ple_normed, inner=self._ple_block)
                hc, normed = ple_hc, ple_normed
            mixed, inject = self._hc_mixer(normed, L.ahc_first, L.ahc_up, True)
            if core.is_attn[i]:
                block_out = self._attention_block(
                    mixed, i, L, proj_qkv, index_scratch, attn_out, pos3
                )
            else:
                block_out = self._gdn_block(mixed, i, L, gdn_out)
            hc, normed = qwen4_exp_hc_inject_norm(
                hc, block_out, inject, L.mhc_norm, core.eps, hc_count
            )
            mixed, inject, xq, xsf = self._hc_moe_mixer(normed, L.mhc_first, L.mhc_up, L.moe_gscale)
            next_norm = layers[i + 1].ahc_norm if i + 1 < len(layers) else S.fhc_norm
            # The MoE block output is formed inside the injection: routed +
            # sigmoid(gate) * shared, rounded once. Both [T, hidden] operands are
            # released here, not at the next layer's MoE block, whose own peak they
            # would otherwise add to.
            routed, (gate_logits, shared) = self._moe_block(mixed, xq, xsf, L, routed_out)
            hc, normed = qwen4_exp_hc_inject_norm(
                hc, routed, inject, next_norm, core.eps, hc_count, shared, gate_logits
            )
            del routed, gate_logits, shared
        # The stack-final mixer with no injection weight is the final norm.
        final, _ = self._hc_mixer(normed, S.fhc_first, S.fhc_up, False)
        if not core.mtp_enabled:
            return final
        # `lm_head` reads the `[T, hidden]` mix; the MTP module reads the
        # `[T, hc_count * hidden]` stream it was collapsed from. They leave packed in one
        # tensor, because a breakable prefill graph's body returns one tensor; the shell
        # takes the two column views (`split_spec_output`).
        return torch.cat([final, hc], -1)

    # ------------------------------------------------------------- the draft ---

    def _mtp_moe_block(self, x, dt) -> torch.Tensor:
        """The draft layer's MoE: FP8 block-scale routed experts plus a bf16 shared expert.

        The scale buffer comes back transposed `[hidden/128, tokens]` and is passed
        straight through; never reshape it. At `tokens == hidden/128` (20, an
        ordinary decode batch) the shapes coincide, so nothing would catch a
        re-materialization. Pre-routed; `routing_bias` must be `None` (the entry
        refuses one)."""
        core = self.core
        M = core._mtp
        logits = cublas_mm(x, M.router)
        topk_ids, topk_w = renorm_moe_routing_op(logits, topk=core.topk, output_dtype=dt)
        xq, xsf = fp8_quantize_1x128(x, use_ue8m0=_MTP_MOE_USE_UE8M0)
        routed = fp8_block_scale_moe_runner(
            routing_logits=None,
            routing_bias=None,
            hidden_states=xq,
            hidden_states_scale=xsf,
            gemm1_weights=M.fc1_w,
            gemm1_weights_scale=M.fc1_s,
            gemm2_weights=M.fc2_w,
            gemm2_weights_scale=M.fc2_s,
            num_experts=core.num_experts,
            top_k=core.topk,
            num_fused_shared_experts=None,
            n_group=None,
            topk_group=None,
            intermediate_size=core.moe_inter,
            local_expert_offset=0,
            local_num_experts=core.num_experts,
            routed_scaling_factor=None,
            routing_method_type=_ROUTING_METHOD_RENORM,
            topk_weights=topk_w,
            topk_ids=topk_ids,
            act_type=_ACT_TYPE_SWIGLU,
            gemm1_clamp_limit=None,
        )
        shared = cublas_mm(flashinfer_silu_and_mul(cublas_mm(x, M.sh_gu)), M.sh_dn)
        gate_logits = cublas_mm(x, M.sh_gate)
        return fused_sigmoid_gate_mul_add(routed, gate_logits, shared)

    def _capture_selection(self, md, topk_indices, visible_blocks) -> None:
        """IndexShare: stash, per request, the selection of the row the draft loop continues
        from, with that row's complete-group count, in the draft layer's row of the shared
        buffers.

        That row is the one the worker gathers for the next draft step: a context
        request's last row of this chunk, and a generation request's last accepted row
        (`num_accepted - 1` past its first). Capturing here rather than at the first draft
        decode step keeps the indexer at one call per draft loop for a batch with context
        rows too."""
        batch = md.num_seqs
        assert md.qsa_mtp_num_accepted is not None, (
            "IndexShare needs the worker's accepted counts to pick each request's capture "
            "row; the draft loop did not supply them"
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
        # Host-side readiness, recorded in program order: under capture this runs while the
        # graph is recorded, so a graph holds the reuse only after the capture it reads.
        md.qsa_shared_topk_captured_slots.add(slot)

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
        """One draft step whose metadata still carries context rows.

        `spec_metadata` and `all_rank_num_tokens` are accepted and unused."""
        core = self.core
        assert isinstance(attn_metadata, TrtllmAttentionMetadata)
        assert core._bound is not None, (
            "the trunk's cache-manager bind has not run; the shell calls the "
            "core before the worker, so this cannot be reached first"
        )
        md = attn_metadata
        rows = hidden_states.shape[0]
        assert input_ids.shape[0] == rows, (
            f"the draft step's {input_ids.shape[0]} token ids and {rows} "
            "hidden-state rows must describe the same tokens"
        )
        assert hidden_states.shape[1] == core.stream, (
            f"the MTP module reads the trunk's {core.stream}-wide four-stream "
            f"residual; the runtime handed it {hidden_states.shape[1]} columns"
        )
        dt = hidden_states.dtype
        dev = hidden_states.device

        # Projected on every call: the worker rewrites `attn_metadata` in place
        # between draft step 0 and 1+ (`_seq_lens` to 1, `num_contexts` and
        # `num_ctx_tokens` to 0), so a first-call value is wrong on the rest.
        step = self.step_args(md)
        M = core._mtp
        num_ctx = md.num_ctx_tokens
        gen_rows = rows - num_ctx
        gen_seqs = md.num_seqs - md.num_contexts
        p = gen_rows // gen_seqs if gen_seqs else 1
        assert p * gen_seqs == gen_rows, (
            f"{gen_rows} generation rows do not divide over {gen_seqs} "
            "generation sequences; a ragged per-sequence draft length cannot be "
            "expressed to the attention call"
        )

        # The KV cache written here is not the trunk's: with a separate draft KV
        # manager the worker swaps `attn_metadata.kv_cache_manager` and
        # `kv_cache_block_offsets` for the draft loop. The layer index must be taken
        # from the current manager every call; caching the trunk's addresses another
        # layer's pages with finite, plausible drafts.
        cm = md.kv_cache_manager
        assert cm is not None, "a paged KV cache is required"
        offsets = cm.layer_offsets
        assert core.mtp_layer_idx in offsets, (
            f"the MTP module's attention is global layer index "
            f"{core.mtp_layer_idx}, which the cache manager "
            f"{type(cm).__name__} does not map (it maps {len(offsets)} layers, "
            f"is_draft={getattr(cm, 'is_draft', None)}); the engine allocated "
            "no KV layer for the draft module"
        )
        layer_idx = offsets[core.mtp_layer_idx]
        window = cm.max_attention_window_vec[layer_idx]
        assert window is None or window == md.max_seq_len, (
            f"the MTP layer's pool window is {window!r} but this call passes "
            f"max_seq_len {md.max_seq_len} as the attention window; a sliding "
            "window is not implemented here"
        )

        # The draft layer's QSA planes and the page table that addresses them. With a
        # separate draft manager the engine's QSA metadata swaps `qsa_block_table` for the
        # table it decodes from the draft manager's offsets (`prepare_for_draft_forward`);
        # a table still decoding the trunk's offsets would read and write another pool's
        # pages.
        draft = core._draft_bound
        if draft is None or cm is not draft.cm:
            core._bind_draft_cache_manager(md)
            draft = core._draft_bound
        assert cm is core._bound.cm or md.qsa_block_table is getattr(
            md, "qsa_draft_block_table", None
        ), (
            "the draft step is bound to a separate draft KV cache manager but "
            "qsa_block_table is not that manager's table: the engine's QSA metadata did "
            "not swap its page table for the draft forward"
        )
        # The `qsa_*` row mapping is sized from the metadata's token count, which the
        # worker rewrites together with it between draft steps.
        assert rows == md.num_tokens, (
            f"the draft step carries {rows} rows for a batch its metadata describes as "
            f"{md.num_tokens} tokens; the QSA row mapping covers the second"
        )
        # The trunk's dense/sparse decision from the same host buffer: the worker advances
        # only the device KV lengths through the draft loop, so every draft step of one
        # engine forward takes the branch its trunk took, in the CUDA-graph family the
        # runner chose for that trunk.
        batch = md.num_seqs
        sparse = batch > 0 and int(md.kv_lens_runtime[:batch].max()) > core.dense_threshold
        # IndexShare: the only draft step with context rows is the loop's first, its
        # draft-extend step. It always runs the indexer and, with sharing on, captures the
        # selection the draft decode steps reuse.
        share = core.mtp_index_share and md.qsa_in_mtp_draft_loop
        assert not (share and md.qsa_indexer_skip_topk), (
            "the worker asked a draft step with context rows to reuse a selection; only "
            "the draft decode steps after it reuse"
        )

        # Copy into a fresh contiguous `[3, N]`: a sliced view keeps the source row
        # stride and the op rejects it under `use_mrope`.
        assert position_ids.shape[0] == _QSA_POSITION_AXES, (
            f"expected the mRoPE position layout with {_QSA_POSITION_AXES} "
            f"coordinate rows, got {tuple(position_ids.shape)}"
        )
        pos3 = torch.empty(
            [_QSA_POSITION_AXES, position_ids.numel() // _QSA_POSITION_AXES],
            dtype=torch.int32,
            device=position_ids.device,
        )
        pos3.copy_(torch.reshape(position_ids, [_QSA_POSITION_AXES, -1]))
        # The trunk's two other views, from the copy: `pos` (coordinate 0 alone, which the
        # index rope reads) and `coords` (the `[N, 3]` transpose the position plane stores).
        pos = torch.reshape(torch.split(pos3, [1, _QSA_POSITION_AXES - 1], 0)[0], [-1])
        coords = torch.transpose(pos3, 0, 1)

        # ---- the front end --------------------------------------------------
        e = nn.functional.embedding(input_ids, embed_tokens)
        x_e = cublas_mm(flashinfer_gemma_rmsnorm(e, M.pre_e_norm, core.eps), M.fc_e)
        # `fc_hidden` is one `[hidden, hidden]` matrix applied to each of the four
        # streams (the checkpoint's norm statistics say they are distinct streams).
        h_n = self._grouped_norm(hidden_states, M.pre_h_norm)
        x_h = torch.reshape(
            cublas_mm(torch.reshape(h_n, [rows * core.hc_count, core.hidden]), M.fc_h),
            [rows, core.stream],
        )
        if _MTP_EMBED_BLOCK_ALL_STREAMS:
            x = torch.add(x_h, torch.cat([x_e] * core.hc_count, -1))
        else:
            x = torch.add(
                x_h,
                nn.functional.pad(x_e, [0, core.stream - core.hidden], mode="constant", value=0.0),
            )

        # ---- the decoder layer ----------------------------------------------
        mixed, inject = qwen4_exp_gated_residual(
            self._grouped_norm(x, M.ahc_norm), M.ahc_down, M.ahc_up, M.ahc_inject, core.hc_count
        )
        qkv = torch.empty([rows, core.qkv_width], dtype=dt, device=dev)
        attn_out = torch.empty([rows, core.q_width], dtype=dt, device=dev)
        qg = cublas_mm(mixed, M.q)
        k = cublas_mm(mixed, M.k)
        v = cublas_mm(mixed, M.v)
        qkv_h = torch.reshape(qkv, [rows, core.heads_q + 2 * core.heads_kv, core.head_dim])
        q_dst, k_dst, v_dst = torch.split(qkv_h, [core.heads_q, core.heads_kv, core.heads_kv], 1)
        q_src, gate = torch.split(
            torch.reshape(qg, [rows, core.heads_q, 2 * core.head_dim]),
            [core.head_dim, core.head_dim],
            2,
        )
        q_dst.copy_(q_src)
        k_dst.copy_(torch.reshape(k, [rows, core.heads_kv, core.head_dim]))
        v_dst.copy_(torch.reshape(v, [rows, core.heads_kv, core.head_dim]))
        # The trunk rope's certified cell at the module's own base (`mtp.rope_theta`).
        fused_qk_norm_rope(
            qkv,
            num_heads_q=core.heads_q,
            num_heads_k=core.heads_kv,
            num_heads_v=core.heads_kv,
            head_dim=core.head_dim,
            rotary_dim=core.rotary_dim,
            eps=core.eps,
            q_weight=M.q_norm,
            k_weight=M.k_norm,
            base=core.mtp_theta,
            is_neox=True,
            position_ids=pos3,
            is_qk_norm=True,
            use_gemma=True,
            use_mrope=True,
            mrope_section1=core.mrope_section1,
            mrope_section2=core.mrope_section2,
        )
        # The QSA index branch, as in the trunk's `_attention_block`: the raw key stored
        # before anything normalizes or rotates it, the query's own norm and rotation,
        # then the compression of every group this step completes. Maintained on every
        # draft step at both ends of the threshold, so the draft plane is complete when a
        # request crosses it.
        req_idx = md.qsa_req_idx_per_token[:rows]
        logical = md.qsa_logical_positions[:rows]
        block_table = md.qsa_block_table
        tokens_per_block = md.tokens_per_block
        index_cache = draft.index_cache
        qk_index = cublas_mm(mixed, M.idx)
        token_k = torch.split(
            torch.reshape(qk_index, [rows, core.index_heads + core.index_kv_heads, -1]),
            [core.index_heads, core.index_kv_heads],
            1,
        )[1]
        qsa_paged_index_store(
            token_k,
            coords,
            index_cache,
            draft.position_cache,
            req_idx,
            logical,
            block_table,
            tokens_per_block,
        )
        index_scratch = torch.empty([rows, core.index_scratch_width], dtype=dt, device=dev)
        index_scratch[:, : core.index_width].copy_(qk_index)
        fused_qk_norm_rope(
            index_scratch,
            num_heads_q=core.index_heads,
            num_heads_k=core.index_kv_heads,
            num_heads_v=core.index_kv_heads,
            head_dim=core.index_head_dim,
            rotary_dim=core.rotary_dim,
            eps=core.eps,
            q_weight=M.idx_qn,
            k_weight=M.idx_kn,
            base=core.mtp_theta,
            is_neox=True,
            position_ids=pos3 if _QSA_INDEX_USE_MROPE else pos,
            is_qk_norm=True,
            use_gemma=True,
            use_mrope=_QSA_INDEX_USE_MROPE,
            mrope_section1=(_QSA_MROPE_SECTION or (0, 0, 0))[1],
            mrope_section2=(_QSA_MROPE_SECTION or (0, 0, 0))[2],
        )
        q_index = torch.split(
            torch.reshape(
                index_scratch,
                [rows, core.index_heads + 2 * core.index_kv_heads, core.index_head_dim],
            ),
            [core.index_heads, core.index_kv_heads, core.index_kv_heads],
            1,
        )[0]
        qsa_prefill_compress(
            logical_positions=logical,
            request_indices=req_idx,
            block_table=block_table,
            index_cache=index_cache,
            position_cache=draft.position_cache,
            k_norm_weight=M.idx_kn,
            cos_sin=core._mtp_cos_sin,
            eps=core.eps,
            tokens_per_block=tokens_per_block,
            compress_ratio=core.compress_ratio,
            mrope_section=_QSA_MROPE_SECTION,
        )

        if sparse:
            kv = draft.kv
            # The append the dense op would have owned, then selection over the draft
            # layer's own compressed keys and the sparse GQA over its pool.
            _store_kv(core, kv, k_dst, v_dst, req_idx, logical, block_table, tokens_per_block)
            scores = qsa_paged_index_scores(
                q=q_index,
                index_cache=index_cache,
                block_table=block_table,
                query_positions=logical,
                request_indices=req_idx,
                tokens_per_block=tokens_per_block,
                compress_ratio=core.compress_ratio,
                only_visible_blocks=True,
                context_rows=True,
            )
            topk_indices = md.qsa_topk_indices[:rows]
            visible_blocks = md.qsa_visible_blocks[:rows]
            _select(
                core,
                scores,
                md.qsa_topk_row_starts[:rows],
                visible_blocks,
                topk_indices,
                _live_score_columns(core, md, batch),
            )
            if share:
                self._capture_selection(md, topk_indices, visible_blocks)
            selected = qsa_expand_block_indices(
                topk_indices,
                logical,
                md.qsa_sequence_lengths[:rows],
                compress_ratio=core.compress_ratio,
                token_topk=core.token_topk,
            )
            attn_out = torch.reshape(
                _sparse_gqa(
                    core,
                    kv,
                    q=q_dst,
                    block_table=block_table,
                    selected_tokens=selected,
                    request_indices=req_idx,
                    tokens_per_block=tokens_per_block,
                    query_positions=logical,
                    compress_ratio=core.compress_ratio,
                ),
                [rows, core.q_width],
            )
        else:
            attn_out = thop_attention(
                q=qkv,
                output=attn_out,
                local_layer_idx=layer_idx,
                predicted_tokens_per_seq=p,
                num_heads=core.heads_q,
                num_kv_heads=core.heads_kv,
                head_size=core.head_dim,
                quant_mode=core.quant_mode,
                kv_scale_orig_quant=core.kv_scale_orig_quant,
                kv_scale_quant_orig=core.kv_scale_quant_orig,
                **step,
                **_CALL_CONSTANTS,
            )
        fused_sigmoid_mul_inplace(attn_out, gate)
        x = qwen4_exp_post_inject(x, cublas_mm(attn_out, M.o), inject)

        mixed, inject = qwen4_exp_gated_residual(
            self._grouped_norm(x, M.mhc_norm), M.mhc_down, M.mhc_up, M.mhc_inject, core.hc_count
        )
        return qwen4_exp_post_inject(x, self._mtp_moe_block(mixed, dt), inject)


class DecodeTarget:
    """Generation-only batches (no context rows): drops the context-row recurrent
    ops and adds the fused single-token recurrence."""

    def __init__(self, core: Qwen4ExpTp1ModelingV2) -> None:
        """Take the per-layer weight views once; MTP operands are read off `core._mtp`."""
        self.core = core
        # See PrefillTarget.__init__.
        self._contract_pending = step_contract_enabled()
        self._kv_probe = (
            _KvRangeProbe(core.attn_layers[-1])
            if self._contract_pending and core.quant_mode == QM_FP8_KV_CACHE
            else None
        )
        self._layers = _layer_views(core)
        self._stack = _stack_views(core)

    def _check_step_contract(self, md) -> None:
        """Opt-in first-forward fail-fast; see PrefillTarget._check_step_contract."""
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
            # No sliding window: the full admitted length on every attention layer.
            attention_window_size=md.max_seq_len,
            num_contexts=0,
            num_ctx_tokens=0,
            trtllm_gen_jit_warmup=md.trtllm_gen_jit_warmup,
            use_paged_context_fmha=md.use_paged_context_fmha,
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

    # --------------------------------------------------------------- blocks ---

    def _hc_first(self, normed, first, with_inject):
        """The mixer's first half from the site's normed stream: the projection
        over `[down; block_inject; 0]` and the SiLU / injection activations.
        Returns `(low, inject)`. Up to `_HC_TC_FIRST_MAX_ROWS` rows (a per-capture
        constant) one tcgen05 launch, `qwen4_exp_hc_decode_tc_first`; larger batches
        cuBLAS's fp32 logits and `qwen4_exp_hc_lowrank_act`. Only sites past
        `_HC_DECODE_GEMV_MAX_ROWS` rows reach a mixer."""
        core = self.core
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

    def _hc_mixer(self, normed, first, up, with_inject):
        """One hyper-connection mixer from the site's normed stream: the first
        projection over `[down; block_inject; 0]` (fp32 logits), the SiLU and
        injection activations, then the up projection with the sigmoid gate and the
        stream mean in its epilogue. Returns `(mixed, inject)`; `inject` is None
        without `with_inject` (the stack-final mixer).

        Neither projection takes the step's `gemm` (`low_m_bf16_mm` under CUDA
        graphs): neither FlashInfer low-M kernel writes the first projection's
        fp32 logits, and the up projection's K = 320 tiles neither of them."""
        core = self.core
        low, inject = self._hc_first(normed, first, with_inject)
        return qwen4_exp_hc_up_gate_mean(low, up, normed, core.hc_count), inject

    def _hc_site(self, hc, block_out, inject, norm, first, up, with_inject, normed=None, moe=None):
        """One trunk hyper-connection site: the previous block's injection (none at
        the stack's first site and after the PLE add), this site's grouped norm and
        its mixer. Returns `(hc, mixed, inject)`. The row count is a per-capture
        constant: at most `_HC_DECODE_GEMV_MAX_ROWS` rows run two GEMV launches,
        larger batches `qwen4_exp_hc_inject_norm` and `_hc_mixer`. `normed`, when
        given, is this site's grouped norm of `hc` already computed (the PLE tail's,
        with no injection pending); the cuBLAS path starts from it.

        After a MoE block, `block_out` is its routed output and `moe` the
        `(gate_logits, shared, topk_weights, topk_rows)` `_moe_block` left
        uncombined: the site's first kernel forms the block output itself. With
        the top-k pair `block_out` is the runner's permuted FC2 rows and that
        kernel (on either route) also forms their finalize. A site handed
        `normed` has no pending block, so it never carries `moe`."""
        core = self.core
        gate_logits, shared, topk_w, topk_rows = moe if moe is not None else (None,) * 4
        if hc.shape[0] <= _HC_DECODE_GEMV_MAX_ROWS:
            n_cols = core.hc_lowrank + (core.hc_count if with_inject else 0)
            hc, part, ss = qwen4_exp_hc_decode_first_cute(
                hc,
                block_out,
                inject,
                norm,
                first,
                core.hc_count,
                n_cols,
                shared,
                gate_logits,
                topk_w,
                topk_rows,
            )
            mixed, inject = qwen4_exp_hc_decode_mix_cute(
                part, ss, up, hc, norm, core.eps, core.hc_count, with_inject
            )
            return hc, mixed, inject
        if normed is None:
            hc, normed = qwen4_exp_hc_inject_norm(
                hc,
                block_out,
                inject,
                norm,
                core.eps,
                core.hc_count,
                shared,
                gate_logits,
                topk_w,
                topk_rows,
            )
        mixed, inject = self._hc_mixer(normed, first, up, with_inject)
        return hc, mixed, inject

    def _hc_moe_site(self, hc, block_out, inject, norm, first, up, gscale):
        """`_hc_site` at an MoE site (always with injection), returning the
        routed experts' NVFP4 input (linear scales, the layer's global scale
        `gscale`) beside the mix, written by the mixer's own last launch: on the
        GEMV route `qwen4_exp_hc_decode_mix_fp4_cute`, otherwise the fused up
        projection `qwen4_exp_hc_up_gate_mean_fp4`. Returns
        `(hc, mixed, inject, xq, xsf)`."""
        core = self.core
        if hc.shape[0] <= _HC_DECODE_GEMV_MAX_ROWS:
            hc, part, ss = qwen4_exp_hc_decode_first_cute(
                hc, block_out, inject, norm, first, core.hc_count, core.hc_lowrank + core.hc_count
            )
            mixed, inject, xq, xsf = qwen4_exp_hc_decode_mix_fp4_cute(
                part, ss, up, hc, norm, core.eps, core.hc_count, True, gscale
            )
            return hc, mixed, inject, xq, xsf
        hc, normed = qwen4_exp_hc_inject_norm(hc, block_out, inject, norm, core.eps, core.hc_count)
        low, inject = self._hc_first(normed, first, True)
        mixed, xq, xsf = qwen4_exp_hc_up_gate_mean_fp4(low, up, normed, core.hc_count, gscale)
        return hc, mixed, inject, xq, xsf

    def _grouped_norm(self, x: torch.Tensor, weights: tuple) -> torch.Tensor:
        """See `PrefillTarget._grouped_norm`."""
        core = self.core
        parts = torch.split(x, [core.hidden] * core.hc_count, 1)
        return torch.cat(
            [flashinfer_gemma_rmsnorm(part, w, core.eps) for w, part in zip(weights, parts)], 1
        )

    def _moe_block(
        self, x, xq, xsf, L: _LayerWeights, routed_out, gemm=cublas_mm, fold=False
    ) -> tuple:
        """One MoE layer: NVFP4 routed experts plus a bf16 shared expert, left
        uncombined. Returns `(routed, (gate_logits, shared, topk_weights,
        topk_rows))`: the next hyper-connection site forms `routed +
        sigmoid(gate_logits) * shared` inside its first kernel (`_hc_site`'s
        `moe`). With `fold` (trtllm-gen, and the next site takes the combine; a
        per-capture constant) the runner also skips its finalize: `routed` is then
        its permuted FC2 rows and `topk_weights` / `topk_rows` its routing weights
        and index map, which that site's first kernel
        (`qwen4_exp_hc_decode_first_cute` or `qwen4_exp_hc_inject_norm`) gathers;
        without it both are None and `routed` is the finalized `[T, H]` rows.

        The shared expert and its per-token gate logit run on the core's side
        stream, concurrently with the routed path, and join it before this
        returns. `x` is the bf16 hidden states (router and shared expert); `xq` /
        `xsf` are the same states in NVFP4 with **linear** block scales, from
        `_hc_moe_site`. On trtllm-gen the routed path hands the runner the router
        logits (its Renormalize routing is bitwise the `renorm_moe_routing_op` +
        pre-routed composition); on CuteDSL it is `_cute_dsl_routed`, which writes
        `routed_out`. `gemm` is the step's bf16 projection entry (see `forward`); it
        launches on the current stream, so the shared expert's GEMMs run on the side
        stream."""
        core = self.core
        main = torch.cuda.current_stream()
        side = core._side_stream
        # Fork: the side stream reads `x` only after everything queued on `main`
        # so far. What it allocates is consumed on `main` after the join below,
        # and the next layer's fork orders any reuse of those blocks after that.
        side.wait_stream(main)
        with torch.cuda.stream(side):
            shared = gemm(flashinfer_silu_and_mul(gemm(x, L.sh_gu)), L.sh_dn)
            gate_logits = gemm(x, L.sh_gate)
        # The routing reads its logits as a dense row-major buffer from data_ptr() and
        # ignores strides; both projection entries allocate them contiguous.
        logits = gemm(x, L.router)
        if core.trunk_moe_backend == "CUTEDSL":
            routed = _cute_dsl_routed(core, L, logits, xq, xsf, routed_out)
            main.wait_stream(side)
            return routed, (gate_logits, shared, None, None)
        moe_op = fp4_block_scale_moe_renorm_permuted if fold else fp4_block_scale_moe_renorm_routed
        out = moe_op(
            logits,
            xq,
            xsf.view(torch.float8_e4m3fn),
            L.fc1_w,
            L.fc1_s,
            L.fc2_w,
            L.fc2_s,
            L.out1_scale,
            L.out1_gate_scale,
            L.out2_scale,
            core.num_experts,
            core.topk,
            core.moe_inter,
        )
        main.wait_stream(side)
        if fold:
            routed, topk_w, topk_rows = out
            return routed, (gate_logits, shared, topk_w, topk_rows)
        return out, (gate_logits, shared, None, None)

    def _attention_prologue(
        self, proj, q_index, index_view, index_cache, index_section, L, qsa, side, stage
    ) -> None:
        """The norm + rope of both planes on this stream and the QSA index branch (the
        verify step's staging, the raw-key store, then compression) on the side stream,
        which reads only the projection's index-key columns and the index pools; the
        attention core joins it before reading either."""
        core = self.core
        num_tokens = proj.shape[0]
        req_idx, logical, _, _, _, _, block_table, tokens_per_block, pos3, coords, _ = qsa
        side.wait_stream(torch.cuda.current_stream())
        qwen4_exp_qk_norm_rope(
            proj,
            q_index,
            pos3,
            core._cos_sin,
            L.q_norm,
            L.k_norm,
            L.idx_qn,
            num_heads_q=core.heads_q,
            num_heads_k=core.heads_kv,
            head_dim=core.head_dim,
            index_heads=core.index_heads,
            index_head_dim=core.index_head_dim,
            index_q_col=core.qkv_width + core.q_width,
            rotary_dim=core.rotary_dim,
            eps=core.eps,
            mrope_section1=core.mrope_section1,
            mrope_section2=core.mrope_section2,
            index_mrope_section1=index_section[1],
            index_mrope_section2=index_section[2],
        )

        # ---- the QSA index branch ------------------------------------------
        # The raw key is stored **before** anything normalizes or rotates it, because
        # the compression step pools raw keys and applies the norm and the rotation
        # itself at the group's first member's coordinates; a key normalized on the way
        # in double-normalizes every anchor that reads it, which is finite, silent, and
        # invisible below the sparse threshold. The norm + rope above never writes the
        # index key columns.
        with torch.cuda.stream(side):
            token_k = torch.split(
                torch.reshape(
                    index_view,
                    [
                        num_tokens,
                        core.index_scratch_width // core.index_head_dim,
                        core.index_head_dim,
                    ],
                ),
                [core.index_heads, core.index_kv_heads, 1],
                1,
            )[1]
            if stage:
                core.spec_rollback.stage_index(index_cache, core._bound.position_cache)
            qsa_paged_index_store(
                token_k,
                coords,
                index_cache,
                core._bound.position_cache,
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
                position_cache=core._bound.position_cache,
                k_norm_weight=L.idx_kn,
                cos_sin=core._cos_sin,
                eps=core.eps,
                tokens_per_block=tokens_per_block,
                compress_ratio=core.compress_ratio,
                mrope_section=_QSA_MROPE_SECTION,
            )

    def _attention_block(
        self, x, i, L, qkv, index_scratch, attn_out, qsa, step, stage, gemm=cublas_mm
    ):
        """One gated GQA layer plus its QSA index branch for a generation-only batch.

        The four input projections are one GEMM against the load-time fused
        operand (`weights._attn_in_w`), whose output row is
        `[q | k | v | gate | index q, k | v slot]`. One norm+rope launch rewrites
        the main q/k heads in place on that output and writes the normalized,
        rotated index query into the scratch; the sparse path reads q/k/v as views
        of it, the dense path takes one compact copy of its leading `[q | k | v]`
        (the dense op reads its QKV flat), the gate is read through its row stride,
        and the index branch reads its raw key from it.

        **The index plane is maintained on every step, at both ends of the
        threshold.** Below it the dense core runs and the index cache is still
        advanced, so a request that grows past the threshold finds its prefix
        already compressed; above it the dense core is replaced outright and the
        main K/V append becomes this layer's own call, because the op that would
        have done it is not the one running. Above the threshold over a bf16 pool,
        on a step with one row per request, everything up to the index scores is one
        launch, `qwen4_exp_attn_decode_prologue`. A verify step stays on the composed
        prologue: at a group's last token that launch reads the group's earlier keys
        from the index plane, and in a verify group the rows before it are written by
        the same launch."""
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
            coords,
            sparse,
        ) = qsa
        num_tokens = x.shape[0]

        proj = gemm(x, L.attn_in)
        main = torch.cuda.current_stream()
        side = core._side_stream
        qkv_view, gate, index_view = torch.split(
            proj, [core.qkv_width, core.q_width, core.index_scratch_width], 1
        )
        q_dst, k_dst, v_dst = _attn_in_heads(core, proj)
        # The index query lands in the q slots of the scratch; its k and v slots
        # are layout only and nothing reads them.
        q_index = torch.split(
            torch.reshape(
                index_scratch,
                [num_tokens, core.index_heads + 2 * core.index_kv_heads, core.index_head_dim],
            ),
            [core.index_heads, core.index_kv_heads, core.index_kv_heads],
            1,
        )[0]
        index_section = (_QSA_MROPE_SECTION or (0, 0, 0)) if _QSA_INDEX_USE_MROPE else (0, 0, 0)
        index_cache = core._bound.index_cache[i]
        kv = core._bound.kv[i]
        fused_prologue = sparse and not stage and kv.k_scale is None and kv.k.dtype == proj.dtype
        if fused_prologue:
            # The norm + rope of both planes, the K/V append the dense op would have
            # owned, and the index branch's raw-key and coordinate store and group
            # compression in one launch on this stream.
            qwen4_exp_attn_decode_prologue(
                proj,
                q_index,
                pos3,
                coords,
                core._cos_sin,
                L.q_norm,
                L.k_norm,
                L.idx_qn,
                L.idx_kn,
                kv.k,
                kv.v,
                index_cache,
                core._bound.position_cache,
                req_idx,
                logical,
                block_table,
                tokens_per_block,
                core.compress_ratio,
                num_heads_q=core.heads_q,
                num_heads_k=core.heads_kv,
                head_dim=core.head_dim,
                index_heads=core.index_heads,
                index_head_dim=core.index_head_dim,
                index_q_col=core.qkv_width + core.q_width,
                rotary_dim=core.rotary_dim,
                eps=core.eps,
                mrope_section1=core.mrope_section1,
                mrope_section2=core.mrope_section2,
                index_mrope_section1=index_section[1],
                index_mrope_section2=index_section[2],
                compress_mrope_section=_QSA_MROPE_SECTION,
            )
        else:
            self._attention_prologue(
                proj, q_index, index_view, index_cache, index_section, L, qsa, side, stage
            )
        if self._kv_probe is not None and not self._kv_probe.observe(i, k_dst, v_dst):
            self._kv_probe = None

        gated = False
        if sparse:
            if not fused_prologue:
                # Selection reads the index plane the side stream wrote.
                main.wait_stream(side)
                # Append the fused-norm+rope'd K/V; the selection reads the current
                # token from the pool. Over an fp8 or NVFP4 pool as in the prefill target.
                _store_kv(core, kv, k_dst, v_dst, req_idx, logical, block_table, tokens_per_block)
            scores = qsa_paged_index_scores(
                q=q_index,
                index_cache=index_cache,
                block_table=block_table,
                query_positions=logical,
                request_indices=req_idx,
                tokens_per_block=tokens_per_block,
                compress_ratio=core.compress_ratio,
                only_visible_blocks=True,
                context_rows=False,
                # Its predecessor is the prologue launch and its consumer the top-k,
                # which waits before reading the scores. With one row per request the
                # prologue writes no compressed key but the one the entry re-reads
                # after its wait, so at decode-sized batches the pages and keys are
                # read before it.
                use_pdl=fused_prologue,
                prefetch=fused_prologue and num_tokens <= _SPARSE_DECODE_MAX_ROWS,
            )
            if _topk_expand_supports(scores):
                # The canonical block top-k and its expansion to tokens in one launch
                # (QSA's row starts are zero).
                selected = qsa_topk_expand_decode(
                    scores,
                    visible_blocks,
                    logical,
                    seq_lens_q,
                    topk_indices,
                    core.compress_ratio,
                    core.token_topk,
                )
            else:
                _select(core, scores, topk_row_starts, visible_blocks, topk_indices)
                selected = qsa_expand_block_indices(
                    topk_indices,
                    logical,
                    seq_lens_q,
                    compress_ratio=core.compress_ratio,
                    token_topk=core.token_topk,
                )
            if kv.k_scale is not None:
                attn = _sparse_gqa(
                    core,
                    kv,
                    q=q_dst,
                    block_table=block_table,
                    selected_tokens=selected,
                    request_indices=req_idx,
                    tokens_per_block=tokens_per_block,
                    # The prefill pair bounds a context row's scan; none here.
                    query_positions=None,
                    compress_ratio=None,
                )
            else:
                # The output gate rides the split-K merge whenever the launch has
                # one (more than one split); otherwise it is its own kernel below.
                gated = qsa_paged_sparse_gqa.fuses_output_gate(q_dst, kv.k, selected, gate)
                if (
                    gated
                    and num_tokens <= _SPARSE_DECODE_MAX_ROWS
                    and _sparse_decode_supports(q_dst, kv.k, selected)
                ):
                    # The split-K pass, then the merge and the gate as one launch spread
                    # over the SMs.
                    attn = qsa_paged_sparse_gqa_decode(
                        q_dst,
                        kv.k,
                        kv.v,
                        block_table,
                        selected,
                        req_idx,
                        tokens_per_block,
                        core.softmax_scale,
                        gate,
                    )
                else:
                    attn = qsa_paged_sparse_gqa(
                        q=q_dst,
                        k_cache=kv.k,
                        v_cache=kv.v,
                        block_table=block_table,
                        selected_tokens=selected,
                        request_indices=req_idx,
                        tokens_per_block=tokens_per_block,
                        softmax_scale=core.softmax_scale,
                        query_positions=None,
                        compress_ratio=None,
                        output_gate=gate if gated else None,
                    )
            out = torch.reshape(attn, [num_tokens, core.q_width])
        else:
            main.wait_stream(side)
            # The dense op reads its packed QKV flat: one compact copy.
            qkv.copy_(qkv_view)
            out = thop_attention(
                q=qkv,
                output=attn_out,
                local_layer_idx=core._bound.pool_idx[i],
                num_heads=core.heads_q,
                num_kv_heads=core.heads_kv,
                head_size=core.head_dim,
                quant_mode=core.quant_mode,
                kv_scale_orig_quant=core.kv_scale_orig_quant,
                kv_scale_quant_orig=core.kv_scale_quant_orig,
                **step,
                **_CALL_CONSTANTS,
            )
        if not gated:
            fused_sigmoid_mul_inplace(out, gate)
        return gemm(out, L.o)

    def _gdn_verify(
        self, mq_d, a_d, b_d, ssm_states, i, L, si_d, int_idx, n, p, replay
    ) -> torch.Tensor:
        """Recurrent half of a speculative verify step; see
        `PrefillTarget._gdn_verify`."""
        core = self.core
        # [n*p, C] -> [n, C, p]: the multi-token conv reads channel-major.
        x3 = torch.transpose(torch.reshape(mq_d, [n, p, core.conv_dim]), 1, 2)
        conv_out = causal_conv1d_triton_update(
            x=x3,
            conv_state=core._bound.conv[i],
            weight=L.conv_w,
            bias=None,
            activation="silu",
            conv_state_indices=si_d,
            pad_slot_id=_PAD_SLOT_ID,
            # Snapshots, not the accepted-token roll.
            num_accepted_tokens=None,
            intermediate_conv_window=core._bound.int_conv[i],
            intermediate_state_indices=int_idx,
            launch_dependent_kernels=False,
        )
        # Back to token-major (a view).
        mq_out = torch.reshape(torch.transpose(conv_out, 1, 2), [n * p, core.conv_dim])
        if replay is not None:
            return _replay_verify(core, mq_out, a_d, b_d, ssm_states, i, L, si_d, n, p, replay)
        q, k, v = pack_gdn_decode_qkv(
            mq_out, core.num_k_heads, core.head_k_dim, core.num_v_heads, core.head_v_dim
        )
        g = torch.reshape(
            fused_gdn_gating(
                A_log=L.a_log,
                a=a_d,
                dt_bias=L.dt_bias,
                beta=_SOFTPLUS_BETA,
                threshold=_SOFTPLUS_THRESHOLD,
            ),
            [n, p, core.num_v_heads],
        )
        beta = torch.reshape(torch.sigmoid(b_d), [n, p, core.num_v_heads])
        # Initial states are gathered from the live pool; the per-step buffer is
        # indexed by batch row (`intermediate_state_indices[:n]`), not slot id.
        src = torch.reshape(
            nn.functional.embedding(si_d, torch.reshape(ssm_states, [ssm_states.shape[0], -1])),
            [n, core.num_v_heads, core.head_v_dim, core.head_k_dim],
        )
        out = fused_recurrent_gated_delta_rule_update(
            torch.reshape(q, [n, p, core.num_k_heads, core.head_k_dim]),
            torch.reshape(k, [n, p, core.num_k_heads, core.head_k_dim]),
            torch.reshape(v, [n, p, core.num_v_heads, core.head_v_dim]),
            g,
            beta,
            initial_state_source=src,
            initial_state_indices=int_idx,
            scale=None,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=None,
            disable_state_update=True,
            disable_output_calculation=False,
            intermediate_states_buffer=core._bound.int_ssm[i],
            cache_steps=p,
            output=None,
        )
        return torch.reshape(out, [1, n * p, core.num_v_heads, core.head_v_dim])

    def _gdn_block(self, x, i, L, gdn_state, gemm=cublas_mm, fused_conv=False) -> torch.Tensor:
        """One Gated-DeltaNet layer over a generation-only batch: one token per
        sequence, or under speculation the whole draft chain through the verify
        path. Every branch is on a host int the engine fixes for a capture.

        One GEMM for all four input projections: q|k|v, z, a and b are column views of
        its output. A graph-captured single-token step at up to `_GDN_CONV_FUSED_MAX_ROWS`
        rows runs the conv step as that GEMM's epilogue (`gdn_in_proj_conv1d_pdl`), and
        over an fp32 recurrent pool at up to `_GDN_DECODE_FUSED_MAX_ROWS` rows the
        recurrence and the gated norm as one launch (`gdn_decode_gated_norm_pdl`); past
        them, after the conv step, as one streaming launch (`gdn_decode_rows_gated_norm`).
        Over a bf16 pool the recurrence is `fused_sigmoid_gating_delta_rule_update`'s rows
        form, which is batch-invariant."""
        core = self.core
        si_all, is_verify, num_decodes, gen_p, int_idx, replay = gdn_state
        num_tokens = x.shape[0]
        ssm_states = core._bound.ssm[i]
        fp32_state = ssm_states.dtype == torch.float32

        conv_fused = fused_conv and not is_verify and num_tokens <= _GDN_CONV_FUSED_MAX_ROWS
        if conv_fused:
            proj = gdn_in_proj_conv1d_pdl(
                x,
                L.in_proj,
                core._bound.conv[i],
                L.conv_w,
                si_all,
                core.conv_dim,
                _PAD_SLOT_ID,
                True,
            )
        else:
            proj = gemm(x, L.in_proj)
        mixed_qkv, z, a, b = torch.split(proj, core.in_proj_split, 1)

        if fp32_state and conv_fused and num_tokens <= _GDN_DECODE_FUSED_MAX_ROWS:
            # The recurrence and the gated norm as one launch that reads this layer's
            # pool rows before its PDL wait: in a captured single-token step nothing
            # ahead of it writes them, and the slot indices are filled before the
            # graph is launched.
            o = gdn_decode_gated_norm_pdl(
                proj,
                ssm_states,
                si_all,
                L.a_log,
                L.dt_bias,
                L.gnorm,
                core.eps,
                core.num_k_heads,
                core.head_k_dim,
            )
            return gemm(o, L.out)
        if is_verify:
            out = self._gdn_verify(
                mixed_qkv,
                a.contiguous(),
                b.contiguous(),
                ssm_states,
                i,
                L,
                si_all,
                int_idx,
                num_decodes,
                gen_p,
                replay,
            )
            return self._gdn_epilogue(out, z, L, num_tokens, gemm)
        if not conv_fused:
            causal_conv1d_update(
                x=mixed_qkv,
                conv_state=core._bound.conv[i],
                weight=L.conv_w,
                bias=None,
                silu_activation=True,
                cache_seqlens=None,
                conv_state_indices=si_all,
                pad_slot_id=_PAD_SLOT_ID,
            )
        if fp32_state:
            # The recurrence and the gated norm as one launch over `proj`, whose q/k/v
            # columns the conv step rewrote in place; it computes both gates and the
            # q/k L2 norm itself, at the softplus pair (1, 20) and epsilon 1e-6 this
            # target's constants name, and reads this layer's pool rows before its PDL
            # wait (nothing in a single-token step writes them ahead of it).
            o = gdn_decode_rows_gated_norm(
                proj,
                ssm_states,
                si_all,
                L.a_log,
                L.dt_bias,
                L.gnorm,
                core.eps,
                core.num_k_heads,
                core.head_k_dim,
            )
            return gemm(o, L.out)
        else:
            # The rows form, `cu_seqlens` None: it finds row n's tokens at
            # `n * q.stride(1)`, so q/k/v are split from a contiguous `[T, 10240]` copy
            # reshaped to `[T, 1, 80, 128]` (K and V heads are both 128 wide), where the
            # size-1 token axis carries the row stride; a/b are contiguous `[T, 48]`.
            qkv_rows = torch.reshape(mixed_qkv.contiguous(), [num_tokens, 1, -1, core.head_k_dim])
            q, k, v = torch.split(
                qkv_rows, [core.num_k_heads, core.num_k_heads, core.num_v_heads], 2
            )
            out = fused_sigmoid_gating_delta_rule_update(
                A_log=L.a_log,
                a=torch.reshape(a.contiguous(), [num_tokens, 1, core.num_v_heads]),
                dt_bias=L.dt_bias,
                softplus_beta=_SOFTPLUS_BETA,
                softplus_threshold=_SOFTPLUS_THRESHOLD,
                q=q,
                k=k,
                v=v,
                b=torch.reshape(b.contiguous(), [num_tokens, 1, core.num_v_heads]),
                initial_state_source=ssm_states,
                initial_state_indices=si_all,
                scale=None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=None,
                output=None,
            )
        return self._gdn_epilogue(out, z, L, num_tokens, gemm)

    def _gdn_epilogue(self, out, z, L, num_tokens, gemm=cublas_mm) -> torch.Tensor:
        """Gated RMS norm (sigmoid gate, plain non-gemma weight) and output projection
        shared by every recurrence."""
        core = self.core
        o = rms_norm_gated_token_major(
            torch.reshape(out, [-1, core.head_v_dim]),
            torch.reshape(z, [num_tokens, core.num_v_heads, core.head_v_dim]),
            weight=L.gnorm,
            eps=core.eps,
            fp8_scale=None,
            gate_activation="sigmoid",
        )
        return gemm(torch.reshape(o, [num_tokens, core.value_dim]), L.out)

    def _ple_block(self, hc, input_ids, ple_state, next_norm, stage, gemm=cublas_mm) -> tuple:
        """The n-gram side path on one decoder layer, added into the stream and
        normalized for the next site. Returns `(hc, normed)`.

        The same steps as the prefill target's `_ple_block` -- id history, hash, table
        gather and dequant, the gated injection, the dilated short conv -- in three
        launches plus the two projections, because a decode batch carries at most
        `gen_p` rows per sequence:

        1. `ple_ngram_embed`: window, hash, gather, dequantize, and the id
           history published back, one program per sequence;
        2. the key and value projections, on the step's bf16 projection entry
           `gemm` (see `forward`);
        3. `ple_gate_conv_inject_norm`: the key / query grouped norms, the gate,
           the conv-input grouped norm, the conv against the paged conv state
           (published back), the residual add into the stream and this
           layer's attention-side grouped norm, one program per
           (sequence, stream).

        Every generation sequence carries its history by construction, so no carry
        flags are passed. A verify step (`stage`) takes the unfused ops instead, whose
        intermediates `SpeculativeRollback` stages (see `PrefillTarget._ple_delta`)."""
        core = self.core
        S = self._stack
        conv_states, ngram_pool, si_all, carry_all, qsl, gen_p = ple_state
        if stage:
            core.spec_rollback.stage_ple_pools(ngram_pool, conv_states)
            window = ngram_token_history(
                input_ids, ngram_pool, si_all, carry_all, qsl, core.eos_token_id, core.ngram_size
            )
            rows = torch.reshape(
                ngram_hash(
                    window,
                    layer_multipliers=S.ple_mult,
                    head_vocab_sizes=S.ple_vocab,
                    head_offsets=S.ple_offs,
                    eos_token_id=core.eos_token_id,
                    num_query_tokens=1,
                ),
                [-1, core.ngram_heads],
            )
            gathered = nn.functional.embedding(rows, core.w["ple_table"])
            ngram = torch.reshape(
                dequantize_e4m3_per_tensor(gathered, S.ple_tscale), [-1, core.ple_embed_dim]
            )
            key_normed = self._grouped_norm(cublas_mm(ngram, S.ple_key), S.ple_nk)
            query_normed = self._grouped_norm(hc, S.ple_nq)
            value = cublas_mm(ngram, S.ple_val)
            gated = gated_ngram_injection(
                key_normed, query_normed, value, core.hc_count, _PLE_CLAMP_MIN
            )
            conv_in = self._grouped_norm(gated, S.ple_nc)
            core.spec_rollback.stage_ple_rows(window, conv_in)
            # The short conv reads channel-major; a bare transpose is the certified form.
            convolved = dilated_short_conv_paged(
                x=torch.transpose(conv_in, 0, 1),
                weight=S.ple_conv_w,
                bias=None,
                conv_states=conv_states,
                query_start_loc=qsl,
                cache_indices=si_all,
                has_initial_state=carry_all,
                dilation=core.ple_dilation,
                silu_activation=True,
                pad_slot_id=_PAD_SLOT_ID,
            )
            hc = torch.add(hc, torch.add(gated, torch.transpose(convolved, 0, 1)))
            return qwen4_exp_hc_inject_norm(hc, None, None, next_norm, core.eps, core.hc_count)
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
            gen_p,
        )
        return ple_gate_conv_inject_norm(
            hc,
            gemm(ngram, S.ple_key),
            gemm(ngram, S.ple_val),
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
            gen_p,
        )

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
        # The bf16 block projections of a graph-captured step run as PDL launches
        # (`_decode_projection`); an eager step keeps cuBLAS.
        gemm = _decode_projection if md.is_cuda_graph else cublas_mm

        # Two views of one `[3, 1, T]` buffer (see PrefillTarget.forward): contiguous
        # `pos3` and `coords`.
        coords = torch.transpose(torch.reshape(position_ids, [_QSA_POSITION_AXES, -1]), 0, 1)
        pos3 = torch.empty(
            [_QSA_POSITION_AXES, position_ids.shape[-1]],
            dtype=torch.int32,
            device=position_ids.device,
        )
        pos3.copy_(torch.reshape(position_ids, [_QSA_POSITION_AXES, -1]))

        h = nn.functional.embedding(input_ids, core.w["embed"])
        num_tokens = h.shape[0]
        dt = h.dtype
        dev = h.device

        # The batch's bookkeeping, built once; every sequence is a generation sequence.
        batch = md.num_seqs
        # Host-int check (graph-safe); see PrefillTarget.prepare.
        assert num_tokens == md.num_tokens, (
            f"the engine handed {num_tokens} activation rows for a batch its "
            f"metadata describes as {md.num_tokens} tokens "
            f"(padded_num_tokens={getattr(md, 'padded_num_tokens', None)}). A "
            "prefill/piecewise CUDA-graph backend pads the activation buffer "
            "past the batch and expects the model to slice it back; this "
            "target does not implement that configuration. Decode CUDA graphs "
            "pad with real dummy requests instead, so they are in the metadata "
            "and this assert is silent for them."
        )
        # Rows per generation sequence: 1 without speculation, else
        # `runtime_draft_len + 1`.
        gen_p = num_tokens // batch if batch else 1
        assert gen_p * batch == num_tokens, (
            f"{num_tokens} generation rows do not divide over {batch} generation "
            "sequences; a ragged per-sequence draft length can be expressed "
            "neither to the attention call nor to the multi-token recurrence"
        )
        assert core.mtp_enabled or gen_p == 1, (
            f"{num_tokens} generation rows over {batch} generation sequences with "
            "speculation off; the base path implements one query token per "
            "generation sequence"
        )
        # `gen_p > 1` is load-bearing for verify (see PrefillTarget.prepare).
        if spec_metadata is not None and batch > 0:
            declared_p = spec_metadata.runtime_draft_len + 1
            assert gen_p == declared_p, (
                f"the batch carries {gen_p} query rows per generation sequence "
                f"but the worker's runtime_draft_len says {declared_p}; the "
                "per-step conv snapshots and the recurrent rollback buffer are "
                "sized from the second and written from the first"
            )
            check_verify_width(gen_p)
            check_replay_width(core._bound.replay, gen_p)
        is_verify = spec_metadata is not None and batch > 0 and gen_p > 1
        step["predicted_tokens_per_seq"] = gen_p

        si_all = mm.state_indices[:batch]
        qsl_long = mm.query_start_loc_long
        int_idx = None
        replay = None
        if is_verify:
            # The runtime's arange over the slot pool (what `update_mamba_states` reads).
            assert core._bound.int_idx is not None, "the cache-manager bind has not run"
            int_idx = core._bound.int_idx[:batch]
            if core._bound.replay is not None:
                replay = _replay_work(core, mm, batch)
        gdn_state = (si_all, is_verify, batch, gen_p, int_idx, replay)

        # Every generation sequence carries its n-gram history; the engine's
        # `has_initial_states` covers the context prefix only (empty here).
        ple_carry = torch.ones(
            [batch], dtype=mm.has_initial_states.dtype, device=mm.has_initial_states.device
        )
        # PLE segmentation vector: rebuilt from `seq_lens_cuda` when `gen_p > 1` (see
        # PrefillTarget.prepare).
        if gen_p == 1:
            ple_qsl = qsl_long
        else:
            ple_qsl = cu_seqlens_from_seq_lens(md.seq_lens_cuda[:batch], torch.int64)
        ple_state = (
            core._bound.ple_state[0],
            core._bound.ple_state[1],
            si_all,
            ple_carry,
            ple_qsl,
            gen_p,
        )

        # Dense/sparse decision once per forward, from a host-side metadata read: a
        # per-capture constant matching the engine's short/long decode graph families.
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
            coords,
            sparse,
        )

        # Staging for the speculative rollback, as on the prefill target (see
        # PrefillTarget.prepare); every row here is a generation row.
        stage = is_verify
        rollback = core.spec_rollback
        if rollback is not None:
            rollback.discard()
        if stage:
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
            )

        # Scratch shared by all attention layers: thop_attention overwrites rows
        # [:num_tokens] of `output` in place, and the packed buffers are fully
        # rewritten every layer (the QKV one on the dense path only).
        qkv = torch.empty([num_tokens, core.qkv_width], dtype=dt, device=dev)
        index_scratch = torch.empty([num_tokens, core.index_scratch_width], dtype=dt, device=dev)
        attn_out = torch.empty([num_tokens, core.q_width], dtype=dt, device=dev)
        # The CuteDSL routed output, shared by all layers: each layer's unpermute rewrites
        # every row, and the next site's injection consumes it before the next layer's MoE.
        routed_out = (
            torch.empty([num_tokens, core.hidden], dtype=dt, device=dev)
            if core.trunk_moe_backend == "CUTEDSL"
            else None
        )

        # The embedding repeated across the hc_count residual streams.
        hc_count = core.hc_count
        hc = torch.cat([h] * hc_count, -1)
        layers, S = self._layers, self._stack
        # Every hyper-connection site (`_hc_site`): the previous block's pending
        # injection, this site's grouped norm and its mixer. The stack's first
        # site has no pending injection; at the PLE layer it is applied alone
        # (the PLE block reads the injected stream), and the PLE tail returns the
        # stream with its output added plus this site's grouped norm of it, which
        # the site's cuBLAS mixer starts from (a GEMV site normalizes inside its
        # first launch and reads the stream alone). A MoE block's output is
        # pending uncombined (`moe`): the next site's first kernel forms it,
        # except before the PLE add, where the combine runs alone.
        block_out = inject = moe = None
        for i, L in enumerate(layers):
            if i == core.ple_layer:
                if block_out is not None:
                    if moe is not None:
                        # `_moe_block` never folds the finalize before this layer.
                        block_out = fused_sigmoid_gate_mul_add(block_out, *moe[:2])
                    hc = qwen4_exp_post_inject(hc, block_out, inject)
                    block_out = inject = moe = None
                hc, normed = self._ple_block(hc, input_ids, ple_state, L.ahc_norm, stage, gemm)
            else:
                normed = None
            hc, mixed, inject = self._hc_site(
                hc,
                block_out,
                inject,
                L.ahc_norm,
                L.ahc_first,
                L.ahc_up,
                True,
                normed=normed,
                moe=moe,
            )
            if core.is_attn[i]:
                block_out = self._attention_block(
                    mixed, i, L, qkv, index_scratch, attn_out, qsa, step, stage, gemm
                )
            else:
                block_out = self._gdn_block(mixed, i, L, gdn_state, gemm, md.is_cuda_graph)
            hc, mixed, inject, xq, xsf = self._hc_moe_site(
                hc, block_out, inject, L.mhc_norm, L.mhc_first, L.mhc_up, L.moe_gscale
            )
            # The trtllm-gen runner's finalize folds into the next site's first kernel
            # when that site takes the combine (every one but the PLE layer's).
            fold = core.trunk_moe_backend == "TRTLLM" and i + 1 != core.ple_layer
            block_out, moe = self._moe_block(mixed, xq, xsf, L, routed_out, gemm, fold=fold)
        # The stack-final mixer with no injection weight: it returns the mix
        # alone, and that mix *is* this checkpoint's final norm.
        hc, final, _ = self._hc_site(
            hc, block_out, inject, S.fhc_norm, S.fhc_first, S.fhc_up, False, moe=moe
        )
        if spec_metadata is None:
            return final
        # `lm_head` reads the `[T, hidden]` mix; the MTP module reads the
        # `[T, hc_count * hidden]` stream it was collapsed from.
        return final, hc

    # ------------------------------------------------------------- the draft ---

    def _mtp_moe_block(self, x, dt) -> torch.Tensor:
        """FP8 block-scale draft MoE; see `PrefillTarget._mtp_moe_block`."""
        core = self.core
        M = core._mtp
        logits = cublas_mm(x, M.router)
        topk_ids, topk_w = renorm_moe_routing_op(logits, topk=core.topk, output_dtype=dt)
        xq, xsf = fp8_quantize_1x128(x, use_ue8m0=_MTP_MOE_USE_UE8M0)
        routed = fp8_block_scale_moe_runner(
            routing_logits=None,
            routing_bias=None,
            hidden_states=xq,
            hidden_states_scale=xsf,
            gemm1_weights=M.fc1_w,
            gemm1_weights_scale=M.fc1_s,
            gemm2_weights=M.fc2_w,
            gemm2_weights_scale=M.fc2_s,
            num_experts=core.num_experts,
            top_k=core.topk,
            num_fused_shared_experts=None,
            n_group=None,
            topk_group=None,
            intermediate_size=core.moe_inter,
            local_expert_offset=0,
            local_num_experts=core.num_experts,
            routed_scaling_factor=None,
            routing_method_type=_ROUTING_METHOD_RENORM,
            topk_weights=topk_w,
            topk_ids=topk_ids,
            act_type=_ACT_TYPE_SWIGLU,
            gemm1_clamp_limit=None,
        )
        shared = cublas_mm(flashinfer_silu_and_mul(cublas_mm(x, M.sh_gu)), M.sh_dn)
        gate_logits = cublas_mm(x, M.sh_gate)
        return fused_sigmoid_gate_mul_add(routed, gate_logits, shared)

    def _capture_selection(self, md, topk_indices, visible_blocks, batch, p) -> None:
        """IndexShare: stash, per request, the selection of its last accepted draft-extend
        row with that row's complete-group count, in the draft layer's row of the shared
        buffers. Every request carries `p` rows here, and the worker continues the draft
        loop from row `num_accepted - 1` of each (the golden token counts as accepted)."""
        assert md.qsa_mtp_num_accepted is not None, (
            "IndexShare needs the worker's accepted counts to pick each request's capture "
            "row; the draft loop did not supply them"
        )
        accepted = md.qsa_mtp_num_accepted[:batch].to(torch.int64)
        rows = torch.arange(batch, dtype=torch.int64, device=topk_indices.device) * p
        rows = rows + torch.clamp(accepted - 1, min=0, max=p - 1)
        slot = self.core._draft_bound.share_slot
        md.qsa_shared_topk_indices[slot, :batch].copy_(torch.index_select(topk_indices, 0, rows))
        md.qsa_shared_topk_visible_blocks[slot, :batch].copy_(
            torch.index_select(visible_blocks, 0, rows)
        )
        # Host-side readiness; see PrefillTarget._capture_selection.
        md.qsa_shared_topk_captured_slots.add(slot)

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
        """One draft step over generation rows only (every step past the first in an
        engine forward, and the first when the batch had no context).

        `spec_metadata` and `all_rank_num_tokens` are accepted and unused."""
        core = self.core
        assert isinstance(attn_metadata, TrtllmAttentionMetadata)
        assert core._bound is not None, (
            "the trunk's cache-manager bind has not run; the shell calls the "
            "core before the worker, so this cannot be reached first"
        )
        md = attn_metadata
        rows = hidden_states.shape[0]
        assert input_ids.shape[0] == rows, (
            f"the draft step's {input_ids.shape[0]} token ids and {rows} "
            "hidden-state rows must describe the same tokens"
        )
        assert hidden_states.shape[1] == core.stream, (
            f"the MTP module reads the trunk's {core.stream}-wide four-stream "
            f"residual; the runtime handed it {hidden_states.shape[1]} columns"
        )
        dt = hidden_states.dtype
        dev = hidden_states.device

        # Projected on every call: the worker rewrites `attn_metadata` in place
        # between draft steps (see PrefillTarget.mtp_forward).
        step = self.step_args(md)
        M = core._mtp
        gen_seqs = md.num_seqs
        p = rows // gen_seqs if gen_seqs else 1
        assert p * gen_seqs == rows, (
            f"{rows} generation rows do not divide over {gen_seqs} "
            "generation sequences; a ragged per-sequence draft length cannot be "
            "expressed to the attention call"
        )

        # The KV cache written here is the draft manager's, swapped in by the worker
        # per call (see PrefillTarget.mtp_forward); take the layer index from the
        # current manager every time.
        cm = md.kv_cache_manager
        assert cm is not None, "a paged KV cache is required"
        offsets = cm.layer_offsets
        assert core.mtp_layer_idx in offsets, (
            f"the MTP module's attention is global layer index "
            f"{core.mtp_layer_idx}, which the cache manager "
            f"{type(cm).__name__} does not map (it maps {len(offsets)} layers, "
            f"is_draft={getattr(cm, 'is_draft', None)}); the engine allocated "
            "no KV layer for the draft module"
        )
        layer_idx = offsets[core.mtp_layer_idx]
        window = cm.max_attention_window_vec[layer_idx]
        assert window is None or window == md.max_seq_len, (
            f"the MTP layer's pool window is {window!r} but this call passes "
            f"max_seq_len {md.max_seq_len} as the attention window; a sliding "
            "window is not implemented here"
        )

        # The draft layer's QSA planes and the page table that addresses them. With a
        # separate draft manager the engine's QSA metadata swaps `qsa_block_table` for the
        # table it decodes from the draft manager's offsets (`prepare_for_draft_forward`);
        # a table still decoding the trunk's offsets would read and write another pool's
        # pages.
        draft = core._draft_bound
        if draft is None or cm is not draft.cm:
            core._bind_draft_cache_manager(md)
            draft = core._draft_bound
        assert cm is core._bound.cm or md.qsa_block_table is getattr(
            md, "qsa_draft_block_table", None
        ), (
            "the draft step is bound to a separate draft KV cache manager but "
            "qsa_block_table is not that manager's table: the engine's QSA metadata did "
            "not swap its page table for the draft forward"
        )
        # The `qsa_*` row mapping is sized from the metadata's token count, which the
        # worker rewrites together with it between draft steps.
        assert rows == md.num_tokens, (
            f"the draft step carries {rows} rows for a batch its metadata describes as "
            f"{md.num_tokens} tokens; the QSA row mapping covers the second"
        )
        # The trunk's dense/sparse decision from the same host buffer: the worker advances
        # only the device KV lengths through the draft loop, so every draft step of one
        # engine forward takes the branch its trunk took, in the CUDA-graph family the
        # runner chose for that trunk.
        sparse = gen_seqs > 0 and int(md.kv_lens_runtime[:gen_seqs].max()) > core.dense_threshold
        # IndexShare: the loop's draft-extend step (`p` rows per request) runs the indexer
        # and, with sharing on, captures each request's selection; every draft decode step
        # after it (one row per request) reuses that capture instead of running the indexer.
        share = core.mtp_index_share and md.qsa_in_mtp_draft_loop
        reuse = share and md.qsa_indexer_skip_topk
        assert not share or reuse == (p == 1), (
            f"the worker's reuse flag ({md.qsa_indexer_skip_topk}) disagrees with the "
            f"step's shape ({p} rows per request): only the draft decode steps after the "
            "draft-extend step reuse its selection"
        )

        # Copy into a fresh contiguous `[3, N]`: a sliced view keeps the source row
        # stride and the op rejects it under `use_mrope`.
        assert position_ids.shape[0] == _QSA_POSITION_AXES, (
            f"expected the mRoPE position layout with {_QSA_POSITION_AXES} "
            f"coordinate rows, got {tuple(position_ids.shape)}"
        )
        pos3 = torch.empty(
            [_QSA_POSITION_AXES, position_ids.numel() // _QSA_POSITION_AXES],
            dtype=torch.int32,
            device=position_ids.device,
        )
        pos3.copy_(torch.reshape(position_ids, [_QSA_POSITION_AXES, -1]))
        # The trunk's two other views, from the copy: `pos` (coordinate 0 alone, which the
        # index rope reads) and `coords` (the `[N, 3]` transpose the position plane stores).
        pos = torch.reshape(torch.split(pos3, [1, _QSA_POSITION_AXES - 1], 0)[0], [-1])
        coords = torch.transpose(pos3, 0, 1)

        # ---- the front end --------------------------------------------------
        e = nn.functional.embedding(input_ids, embed_tokens)
        x_e = cublas_mm(flashinfer_gemma_rmsnorm(e, M.pre_e_norm, core.eps), M.fc_e)
        # `fc_hidden` is one `[hidden, hidden]` matrix applied to each of the four
        # streams (the checkpoint's norm statistics say they are distinct streams).
        h_n = self._grouped_norm(hidden_states, M.pre_h_norm)
        x_h = torch.reshape(
            cublas_mm(torch.reshape(h_n, [rows * core.hc_count, core.hidden]), M.fc_h),
            [rows, core.stream],
        )
        if _MTP_EMBED_BLOCK_ALL_STREAMS:
            x = torch.add(x_h, torch.cat([x_e] * core.hc_count, -1))
        else:
            x = torch.add(
                x_h,
                nn.functional.pad(x_e, [0, core.stream - core.hidden], mode="constant", value=0.0),
            )

        # ---- the decoder layer ----------------------------------------------
        mixed, inject = qwen4_exp_gated_residual(
            self._grouped_norm(x, M.ahc_norm), M.ahc_down, M.ahc_up, M.ahc_inject, core.hc_count
        )
        qkv = torch.empty([rows, core.qkv_width], dtype=dt, device=dev)
        attn_out = torch.empty([rows, core.q_width], dtype=dt, device=dev)
        qg = cublas_mm(mixed, M.q)
        k = cublas_mm(mixed, M.k)
        v = cublas_mm(mixed, M.v)
        qkv_h = torch.reshape(qkv, [rows, core.heads_q + 2 * core.heads_kv, core.head_dim])
        q_dst, k_dst, v_dst = torch.split(qkv_h, [core.heads_q, core.heads_kv, core.heads_kv], 1)
        q_src, gate = torch.split(
            torch.reshape(qg, [rows, core.heads_q, 2 * core.head_dim]),
            [core.head_dim, core.head_dim],
            2,
        )
        q_dst.copy_(q_src)
        k_dst.copy_(torch.reshape(k, [rows, core.heads_kv, core.head_dim]))
        v_dst.copy_(torch.reshape(v, [rows, core.heads_kv, core.head_dim]))
        # The trunk rope's certified cell at the module's own base (`mtp.rope_theta`).
        fused_qk_norm_rope(
            qkv,
            num_heads_q=core.heads_q,
            num_heads_k=core.heads_kv,
            num_heads_v=core.heads_kv,
            head_dim=core.head_dim,
            rotary_dim=core.rotary_dim,
            eps=core.eps,
            q_weight=M.q_norm,
            k_weight=M.k_norm,
            base=core.mtp_theta,
            is_neox=True,
            position_ids=pos3,
            is_qk_norm=True,
            use_gemma=True,
            use_mrope=True,
            mrope_section1=core.mrope_section1,
            mrope_section2=core.mrope_section2,
        )
        req_idx = md.qsa_req_idx_per_token[:rows]
        logical = md.qsa_logical_positions[:rows]
        block_table = md.qsa_block_table
        tokens_per_block = md.tokens_per_block
        index_cache = draft.index_cache
        # The QSA index branch, as in the trunk's `_attention_block` (see the prefill
        # target's draft step). Under IndexShare a draft decode step runs none of it: what
        # it would write (its row's raw key and coordinates, and the anchor of a group that
        # row completes) sits at or past the newly committed length, and nothing reads it
        # before the next engine step's draft-extend rewrites those positions, storing
        # before it compresses and compressing before it scores.
        if not reuse:
            qk_index = cublas_mm(mixed, M.idx)
            token_k = torch.split(
                torch.reshape(qk_index, [rows, core.index_heads + core.index_kv_heads, -1]),
                [core.index_heads, core.index_kv_heads],
                1,
            )[1]
            qsa_paged_index_store(
                token_k,
                coords,
                index_cache,
                draft.position_cache,
                req_idx,
                logical,
                block_table,
                tokens_per_block,
            )
            index_scratch = torch.empty([rows, core.index_scratch_width], dtype=dt, device=dev)
            index_scratch[:, : core.index_width].copy_(qk_index)
            fused_qk_norm_rope(
                index_scratch,
                num_heads_q=core.index_heads,
                num_heads_k=core.index_kv_heads,
                num_heads_v=core.index_kv_heads,
                head_dim=core.index_head_dim,
                rotary_dim=core.rotary_dim,
                eps=core.eps,
                q_weight=M.idx_qn,
                k_weight=M.idx_kn,
                base=core.mtp_theta,
                is_neox=True,
                position_ids=pos3 if _QSA_INDEX_USE_MROPE else pos,
                is_qk_norm=True,
                use_gemma=True,
                use_mrope=_QSA_INDEX_USE_MROPE,
                mrope_section1=(_QSA_MROPE_SECTION or (0, 0, 0))[1],
                mrope_section2=(_QSA_MROPE_SECTION or (0, 0, 0))[2],
            )
            q_index = torch.split(
                torch.reshape(
                    index_scratch,
                    [rows, core.index_heads + 2 * core.index_kv_heads, core.index_head_dim],
                ),
                [core.index_heads, core.index_kv_heads, core.index_kv_heads],
                1,
            )[0]
            qsa_prefill_compress(
                logical_positions=logical,
                request_indices=req_idx,
                block_table=block_table,
                index_cache=index_cache,
                position_cache=draft.position_cache,
                k_norm_weight=M.idx_kn,
                cos_sin=core._mtp_cos_sin,
                eps=core.eps,
                tokens_per_block=tokens_per_block,
                compress_ratio=core.compress_ratio,
                mrope_section=_QSA_MROPE_SECTION,
            )

        if sparse:
            kv = draft.kv
            _store_kv(core, kv, k_dst, v_dst, req_idx, logical, block_table, tokens_per_block)
            topk_indices = md.qsa_topk_indices[:rows]
            visible_blocks = md.qsa_visible_blocks[:rows]
            if reuse:
                # The capture this loop's draft-extend step made, with the groups completed
                # since appended; the expansion then adds the incomplete tail, so the
                # positions drafted since the capture stay visible.
                slot = draft.share_slot
                assert slot in md.qsa_shared_topk_captured_slots, (
                    "a draft decode step would reuse a selection that no draft-extend step "
                    "of this loop captured"
                )
                qsa_mtp_reuse_topk(
                    shared_indices=md.qsa_shared_topk_indices[slot],
                    shared_visible_blocks=md.qsa_shared_topk_visible_blocks[slot],
                    request_indices=req_idx,
                    visible_blocks=visible_blocks,
                    output=topk_indices,
                )
            else:
                scores = qsa_paged_index_scores(
                    q=q_index,
                    index_cache=index_cache,
                    block_table=block_table,
                    query_positions=logical,
                    request_indices=req_idx,
                    tokens_per_block=tokens_per_block,
                    compress_ratio=core.compress_ratio,
                    only_visible_blocks=True,
                    context_rows=False,
                )
                _select(core, scores, md.qsa_topk_row_starts[:rows], visible_blocks, topk_indices)
                if share:
                    self._capture_selection(md, topk_indices, visible_blocks, gen_seqs, p)
            selected = qsa_expand_block_indices(
                topk_indices,
                logical,
                md.qsa_sequence_lengths[:rows],
                compress_ratio=core.compress_ratio,
                token_topk=core.token_topk,
            )
            attn_out = torch.reshape(
                _sparse_gqa(
                    core,
                    kv,
                    q=q_dst,
                    block_table=block_table,
                    selected_tokens=selected,
                    request_indices=req_idx,
                    tokens_per_block=tokens_per_block,
                    # The prefill pair bounds a context row's scan; none here.
                    query_positions=None,
                    compress_ratio=None,
                ),
                [rows, core.q_width],
            )
        else:
            attn_out = thop_attention(
                q=qkv,
                output=attn_out,
                local_layer_idx=layer_idx,
                predicted_tokens_per_seq=p,
                num_heads=core.heads_q,
                num_kv_heads=core.heads_kv,
                head_size=core.head_dim,
                quant_mode=core.quant_mode,
                kv_scale_orig_quant=core.kv_scale_orig_quant,
                kv_scale_quant_orig=core.kv_scale_quant_orig,
                **step,
                **_CALL_CONSTANTS,
            )
        fused_sigmoid_mul_inplace(attn_out, gate)
        x = qwen4_exp_post_inject(x, cublas_mm(attn_out, M.o), inject)

        mixed, inject = qwen4_exp_gated_residual(
            self._grouped_norm(x, M.mhc_norm), M.mhc_down, M.mhc_up, M.mhc_inject, core.hc_count
        )
        return qwen4_exp_post_inject(x, self._mtp_moe_block(mixed, dt), inject)


class _VisionBlockWeights(NamedTuple):
    """One vision block's weights (`.t()` GEMM views). LayerNorm has a bias, unlike
    every norm in the language tower."""

    n1_w: torch.Tensor
    n1_b: torch.Tensor
    qkv: torch.Tensor
    qkv_b: torch.Tensor
    proj: torch.Tensor
    proj_b: torch.Tensor
    n2_w: torch.Tensor
    n2_b: torch.Tensor
    fc1: torch.Tensor
    fc1_b: torch.Tensor
    fc2: torch.Tensor
    fc2_b: torch.Tensor


class _VisionWeights(NamedTuple):
    """The tower's weights: patch embedding, position table, blocks and patch merger."""

    patch_w: torch.Tensor
    patch_b: torch.Tensor
    pos: torch.Tensor
    blocks: list
    m_n_w: torch.Tensor
    m_n_b: torch.Tensor
    m_fc1: torch.Tensor
    m_fc1_b: torch.Tensor
    m_fc2: torch.Tensor
    m_fc2_b: torch.Tensor


def _vision_views(core: Qwen4ExpTp1ModelingV2) -> _VisionWeights:
    w = core.w
    return _VisionWeights(
        # Not transposed: the entry takes the checkpoint's `[out, in, kt, kh, kw]`.
        patch_w=w["vis_patch_w"],
        patch_b=w["vis_patch_b"],
        pos=w["vis_pos"],
        blocks=[
            _VisionBlockWeights(
                n1_w=w[f"l{b}_vis_n1_w"],
                n1_b=w[f"l{b}_vis_n1_b"],
                # Row layout (q heads, k, v) is already thop_attention's packed form.
                qkv=w[f"l{b}_vis_qkv_w"].t(),
                qkv_b=w[f"l{b}_vis_qkv_b"],
                proj=w[f"l{b}_vis_proj_w"].t(),
                proj_b=w[f"l{b}_vis_proj_b"],
                n2_w=w[f"l{b}_vis_n2_w"],
                n2_b=w[f"l{b}_vis_n2_b"],
                fc1=w[f"l{b}_vis_fc1_w"].t(),
                fc1_b=w[f"l{b}_vis_fc1_b"],
                fc2=w[f"l{b}_vis_fc2_w"].t(),
                fc2_b=w[f"l{b}_vis_fc2_b"],
            )
            for b in range(core.v_depth)
        ],
        m_n_w=w["vis_m_n_w"],
        m_n_b=w["vis_m_n_b"],
        m_fc1=w["vis_m_fc1_w"].t(),
        m_fc1_b=w["vis_m_fc1_b"],
        m_fc2=w["vis_m_fc2_w"].t(),
        m_fc2_b=w["vis_m_fc2_b"],
    )


def _frame_calls(
    grid: list[tuple[int, int, int]], max_rows: int
) -> list[list[tuple[int, int, int]]]:
    """Cut `grid` (one `(frames, height, width)` row per image or video) into tower calls
    of whole frames, in order, each at most `max_rows` patch rows. A call is its grid
    rows: the frames of one item that fall in it are one `(frames, height, width)` row."""
    calls: list[list[tuple[int, int, int]]] = [[]]
    room = max_rows
    for frames, height, width in grid:
        rows = height * width
        assert rows <= max_rows, (
            f"one frame is {height}x{width} = {rows} patch rows; the vision ops are "
            f"certified up to {max_rows} rows in one call"
        )
        while frames:
            take = min(frames, room // rows)
            if take == 0:
                calls.append([])
                room = max_rows
                continue
            calls[-1].append((take, height, width))
            room -= take * rows
            frames -= take
    return calls


class VisionTower:
    """One request's images or videos through the 27-block ViT and its patch merger.

    Not a target: runs only for a context request carrying media, from the shell's
    splice. Attention is bidirectional per frame (a video's frame group is a frame),
    expressed to `thop_attention` as one context "sequence" per frame; running the
    rows as one sequence is 315x-582x outside tolerance. The position-embedding
    interpolation and 2-D rope must both emit in spatial-merge order (what the
    merger's 4-to-1 reshape consumes); a raster-order emitter gives a correctly
    shaped, silently wrong tower.

    The frames run in calls of at most `_VISION_CALL_MAX_PATCHES` rows, cut only between
    frames: attention is per frame, the position embedding and rope are per grid row
    and repeat per frame, the norms and MLPs are row-wise, and a frame's `h * w` is a
    multiple of `merge**2`, so a cut keeps every merger block whole. One frame past the
    bound cannot be served."""

    def __init__(self, core: Qwen4ExpTp1ModelingV2) -> None:
        self.core = core
        self._w = _vision_views(core)
        # Persistent scratch: `thop_attention` grows `workspace` in place; never
        # the engine's own workspace.
        self._ws: torch.Tensor | None = None

    def __call__(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """`pixel_values` is `[sum(t*h*w), in_channels*kt*kh*kw]`, one flat patch per
        row, channel-slowest. `grid_thw` is one `(frames, height, width)` row per
        image or video and stays on the host (no synchronization). Returns
        `[sum(t*h*w) / merge**2, out_hidden_size]`, one row per placeholder, in grid
        row order."""
        core = self.core
        dt = pixel_values.dtype
        assert dt == core.w["embed"].dtype, (
            f"the tower runs at the activation dtype {core.w['embed'].dtype}; the "
            f"runtime handed pixel_values as {dt}. The input processor's `dtype` "
            "property is what decides this, and no cast happens here"
        )
        assert pixel_values.dim() == 2 and pixel_values.shape[1] == core.v_pixel_width, (
            f"pixel_values must be [patches, {core.v_pixel_width}] flat patch rows; "
            f"got {tuple(pixel_values.shape)}"
        )

        grid: list[tuple[int, int, int]] = []
        for row in grid_thw.tolist():
            assert len(row) == 3, f"grid_thw rows are (t, h, w); got {row}"
            frames, height, width = (int(v) for v in row)
            assert frames >= 1 and height >= 1 and width >= 1, row
            assert height % core.v_merge == 0 and width % core.v_merge == 0, (
                f"grid sides must divide by spatial_merge_size {core.v_merge}; got "
                f"height={height}, width={width}"
            )
            grid.append((frames, height, width))
        assert grid, "grid_thw describes no image or video"
        described = sum(t * h * w for t, h, w in grid)
        assert described == pixel_values.shape[0], (
            f"grid_thw describes {described} patch rows and pixel_values carries "
            f"{pixel_values.shape[0]}"
        )

        outputs = []
        start = 0
        for call in _frame_calls(grid, _VISION_CALL_MAX_PATCHES):
            rows = sum(t * h * w for t, h, w in call)
            outputs.append(
                self._run(pixel_values[start : start + rows], call, grid_thw.new_tensor(call))
            )
            start += rows
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, 0)

    def _run(
        self, pixel_values: torch.Tensor, grid: list[tuple[int, int, int]], grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """One tower call over whole frames: `grid` and `grid_thw` (the same rows, as a
        host list and a host tensor) describe exactly the rows of `pixel_values`."""
        core = self.core
        dt, dev = pixel_values.dtype, pixel_values.device
        assert pixel_values.is_cuda, "pixel_values must reach the tower on the device"

        # ---- the frame segmentation, host-side -------------------------------
        # One attention segment per frame: a video's frames do not attend to each other.
        seg_lens: list[int] = []
        for frames, height, width in grid:
            seg_lens.extend([height * width] * frames)
        num_patches = pixel_values.shape[0]
        num_segments = len(seg_lens)
        max_seg = max(seg_lens)

        # Batch-state tensors for the encoder-only call; they describe the frames.
        host_lens = torch.tensor(seg_lens, dtype=torch.int32)
        host_types = torch.zeros([num_segments], dtype=torch.int32)
        host_total = torch.tensor([num_patches, 0], dtype=torch.int32)
        dev_lens = host_lens.to(dev)
        if self._ws is None:
            self._ws = torch.empty([0], dtype=torch.int8, device=dev)

        # ---- patch embedding + absolute position -----------------------------
        V = self._w
        h = torch.add(
            patch_embed_conv3d(pixel_values, V.patch_w, V.patch_b),
            pos_embed_interpolate(V.pos, grid, core.v_merge),
        )
        attn_out = torch.empty([num_patches, core.v_hidden], dtype=dt, device=dev)

        for B in V.blocks:
            qkv = cublas_mm(layer_norm(h, B.n1_w, B.n1_b, _VISION_NORM_EPS), B.qkv, B.qkv_b)
            qkv_h = torch.reshape(qkv, [num_patches, 3 * core.v_heads, core.v_head_dim])
            q_view, k_view, _ = torch.split(qkv_h, [core.v_heads] * 3, 1)
            q_rot, k_rot = vision_rope_2d(
                q_view, k_view, grid_thw, core.v_merge, theta=_VISION_ROPE_THETA
            )
            q_view.copy_(q_rot)
            k_view.copy_(k_rot)
            attn_out = thop_attention(
                q=qkv,
                output=attn_out,
                workspace=self._ws,
                sequence_length=dev_lens,
                host_past_key_value_lengths=host_lens,
                host_total_kv_lens=host_total,
                context_lengths=dev_lens,
                host_context_lengths=host_lens,
                host_request_types=host_types,
                max_num_requests=num_segments,
                max_context_length=max_seg,
                max_seq_len=max_seg,
                attention_window_size=max_seg,
                num_contexts=num_segments,
                num_ctx_tokens=num_patches,
                num_heads=core.v_heads,
                num_kv_heads=core.v_heads,
                head_size=core.v_head_dim,
                **_VISION_CALL_CONSTANTS,
            )
            h = torch.add(h, cublas_mm(attn_out, B.proj, B.proj_b))
            mlp = cublas_mm(layer_norm(h, B.n2_w, B.n2_b, _VISION_NORM_EPS), B.fc1, B.fc1_b)
            # Non-gated MLP with the tanh GELU (not the merger's erf).
            h = torch.add(h, cublas_mm(gelu_tanh(mlp), B.fc2, B.fc2_b))

        # ---- the patch merger ------------------------------------------------
        # The norm runs over `hidden` before the 4-to-1 shuffle; the reshape is legal
        # only because both order-bearing steps emitted in spatial-merge order.
        merged = torch.reshape(
            layer_norm(h, V.m_n_w, V.m_n_b, _VISION_NORM_EPS), [-1, core.v_merge_dim]
        )
        # The erf GELU (`nn.GELU()`); within one bf16 ulp of tanh, hence a separate entry.
        return cublas_mm(gelu_erf(cublas_mm(merged, V.m_fc1, V.m_fc1_b)), V.m_fc2, V.m_fc2_b)


class MTPLayer:
    """The checkpoint's MTP module, replayed once per draft step: one more decoder
    layer (two hyper-connections, gated GQA, 512-expert MoE) behind a two-branch
    front end (`fc_embedding` over the NEXT token's embedding, `fc_hidden` over the
    hidden state per stream). It returns `x` un-collapsed; `shared_head` owns the
    module's mixer. The step runs on the targets (`mtp_forward`) over `core._mtp`.

    The MoE is FP8 128x128 weight blocks (the trunk's is NVFP4); re-quantizing it
    would change the checkpoint's meaning, visible only as lower acceptance.

    Attention is the trunk's QSA layer over the module's own indexer. Every draft step
    maintains the layer's index plane, then runs the dense core while the trunk's host KV
    lengths stay within `dense_seq_len_threshold`, and selection plus sparse GQA above it.
    With a separate draft KV manager the layer's K/V, index and position pages live in
    that manager's pools: the worker swaps `attn_metadata.kv_cache_manager` and the block
    offsets for the draft loop, `QSAAttentionMetadata.prepare_for_draft_forward` swaps
    `qsa_block_table` to the table it decodes from the draft manager's offsets, and the
    core binds the draft manager's planes once per manager (`_DraftBound`). The planes need
    no rollback: each engine step's first draft call rewrites every position from the
    newly committed length on.

    IndexShare (`sparse_attention_config.index_share_for_mtp_iteration`) runs the draft
    indexer once per MTP iteration instead of once per draft step. The draft-extend step,
    the loop's first, runs it and captures per request the selection of the row the loop
    continues from, into the metadata's shared buffers; every draft decode step after it
    skips the index branch and rebuilds its selection from that capture with
    `qsa_mtp_reuse_topk`, which appends the groups completed since; a full row gives up
    its oldest ids to them. The expansion adds the incomplete tail, so the tokens drafted
    since the capture stay visible. Every selection is canonical first (ascending,
    padding last), so the reuse keeps the order.

    Not an `nn.Module`: the weights live in the core's `ParameterDict`, and a second
    registration would break the weight load's both-directions coverage assert."""

    def __init__(self, core: Qwen4ExpTp1ModelingV2, logits_processor) -> None:
        self.core = core
        self.logits_processor = logits_processor

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)

    def shared_head(
        self, hidden_states, lm_head, attn_metadata, return_context_logits
    ) -> torch.Tensor:
        """The module's own head: its stack-final hyper-connection mixer (a full gated
        mixer with its own parameters, not a plain RMSNorm), then the trunk's
        `lm_head` through the shell's `logits_processor`."""
        core = self.core
        M = core._mtp
        parts = torch.split(hidden_states, [core.hidden] * core.hc_count, 1)
        normed = torch.cat(
            [flashinfer_gemma_rmsnorm(part, w, core.eps) for w, part in zip(M.fhc_norm, parts)],
            1,
        )
        final, _ = qwen4_exp_gated_residual(normed, M.fhc_down, M.fhc_up, None, core.hc_count)
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
        """Route the draft step by the phase this call's metadata shows, per call: the
        worker rewrites `attn_metadata` between draft steps, so one engine step's
        draft calls can cross from prefill to decode."""
        return self.core._select_target(attn_metadata).mtp_forward(
            embed_tokens,
            all_rank_num_tokens,
            input_ids,
            position_ids,
            hidden_states,
            attn_metadata,
            **kwargs,
        )


class DraftModel:
    """The container the spec worker reaches the drafter through. The runtime
    inspects only `mtp_layers` (`[0]`: MTP-Eagle replays one layer),
    `embed_tokens`, `lm_head`, and `model.d2t` (correctly absent: shared
    vocabulary).

    `embed_tokens` resolves through the core on every read: this container is
    built while parameters are meta tensors, and materialization replaces the
    tensor objects, so a captured reference stays on meta."""

    def __init__(self, core: Qwen4ExpTp1ModelingV2, lm_head, logits_processor) -> None:
        self.core = core
        self.mtp_layers = [MTPLayer(core, logits_processor)]
        self.lm_head = lm_head

    @property
    def embed_tokens(self) -> torch.Tensor:
        return self.core.w["embed"]


class SpeculativeRollback:
    """The target's auxiliary speculative-state handler: it undoes what a verify step's
    rejected draft rows wrote into the four side-cache planes the runtime's own state
    promotion does not cover.

    A verify step runs `p = runtime_draft_len + 1` rows per generation request through the
    trunk, and four ops write per-request state for every row without knowing how many the
    sampler keeps: the PLE n-gram id history (`ops/ple/ngram_token_history`), the PLE
    dilated-conv state (`ops/ssm/dilated_short_conv_paged`), each attention layer's QSA
    index-K plane (`qsa_paged_index_store`, `qsa_prefill_compress`) and the shared QSA
    position plane. The Gated-DeltaNet state is not here: the runtime promotes it.

    For a generation request that accepted `a` rows (golden token included, so
    `1 <= a <= p`) the commit rewrites:

    * the n-gram pool row to the id window of row `a - 1` minus its oldest id;
    * the conv pool row to columns `[a, a + state_len)` of `[pre-step state | conv input
      of the request's p rows]`;
    * every index-K and position slot written by a row of ordinal `>= a` to its
      pre-verify contents.

    The targets stage during the forward (`begin`, then `stage_ple_pools` /
    `stage_ple_rows` in the PLE block and `stage_index` before each attention layer's
    store) and the spec worker calls `commit_speculative_states` after its draft loop,
    inside the same model forward. Context rows and non-verify steps stage nothing, and a
    commit with nothing staged returns. Staging and commit use metadata-sized
    allocations, gathers, `torch.where` and indexed writes only, and everything staged is
    allocated by the forward and consumed by that forward's commit, so a captured graph
    holds both ends and replay re-reads the engine's buffers in place."""

    def __init__(self) -> None:
        self._pending: dict | None = None

    # ------------------------------------------------------------- staging ---

    def discard(self) -> None:
        """Drop whatever an earlier forward staged."""
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
    ) -> None:
        """Open a verify step's staging. The generation rows follow the `prefix_tokens`
        context rows as `num_gens` runs of `rows_per_gen`, so row `s * p + t` is ordinal `t`
        of generation request `s`. `slots` are those requests' recurrent slots;
        `request_indices` and `logical_positions` are the generation rows' own. Each row's
        QSA slot is resolved here, as `qsa_paged_index_store` resolves it."""
        n, p = num_gens, rows_per_gen
        assert n > 0 and p > 1, (
            f"a verify step stages generation requests carrying drafts; got {n} "
            f"requests of {p} rows"
        )
        assert logical_positions.shape[0] == n * p, (
            f"{n} generation requests of {p} rows are {n * p} rows; got "
            f"{logical_positions.shape[0]} positions"
        )
        positions = logical_positions.to(torch.long)
        pages = block_table[request_indices.to(torch.long), positions // tokens_per_block]
        self._pending = dict(
            num_contexts=num_contexts,
            prefix=prefix_tokens,
            n=n,
            p=p,
            slots=slots.to(torch.long),
            pages=pages.to(torch.long),
            within=positions % tokens_per_block,
            index=[],
            position=None,
        )

    def stage_ple_pools(self, ngram_pool: torch.Tensor, conv_pool: torch.Tensor) -> None:
        """Before the PLE block's first write: both pools, and the generation slots'
        pre-step conv rows."""
        pending = self._pending
        slots = pending["slots"]
        pending["ngram_pool"] = ngram_pool
        pending["conv_pool"] = conv_pool
        pending["conv_prev"] = conv_pool[slots]

    def stage_ple_rows(self, window: torch.Tensor, conv_input: torch.Tensor) -> None:
        """After the id history and before the conv: the generation rows' id windows
        (`[T, ngram_size]`, newest id last) and conv input (`[T, C]`, token-major). Copied
        rather than viewed, so a long context prefix in a mixed batch is not kept alive
        through the draft loop."""
        pending = self._pending
        prefix, n, p = pending["prefix"], pending["n"], pending["p"]
        # Row t of the window without its oldest id is the history after token t.
        pending["ngram_candidates"] = torch.reshape(window[prefix:, 1:], [n, p, -1]).clone()
        pending["conv_x"] = torch.reshape(conv_input[prefix:], [n, p, -1]).clone()

    def stage_index(self, index_cache: torch.Tensor, position_cache: torch.Tensor) -> None:
        """Before an attention layer's index store: the generation rows' slots of its
        index-K plane, and, on the step's first attention layer (the first to write the
        shared plane), of the position plane."""
        pending = self._pending
        pages, within = pending["pages"], pending["within"]
        if pending["position"] is None:
            pending["position"] = (position_cache, position_cache[pages, within])
        pending["index"].append((index_cache, index_cache[pages, within]))

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
            f"the worker commits with {num_contexts} context requests and the "
            f"forward staged {pending['num_contexts']}"
        )
        assert state_indices.shape[0] == num_contexts + n, (
            f"the worker hands {state_indices.shape[0]} slots for a batch the "
            f"forward staged as {num_contexts} + {n} requests"
        )
        assert pending.get("conv_x") is not None and len(pending["index"]) > 0, (
            "a verify step staged neither its PLE rows nor any attention layer; "
            "the forward did not run the staging it opened"
        )
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
        pages, within = pending["pages"], pending["within"]
        for cache, old in pending["index"]:
            cache[pages, within] = torch.where(rejected[:, None, None], old, cache[pages, within])
        cache, old = pending["position"]
        cache[pages, within] = torch.where(rejected[:, None], old, cache[pages, within])


#: Per modality, the keys under `multimodal_data[modality]`: the flat patch rows and the
#: host-side `(t, h, w)` grid.
_MEDIA_KEYS = {
    "image": ("pixel_values", "image_grid_thw"),
    "video": ("pixel_values_videos", "video_grid_thw"),
}


class _WidenedHead:
    """`lm_head` whose bf16 logits come back as fp32 from `bf16_to_fp32` (one vectorized
    kernel; torch's `.float()` is an unvectorized copy at under half the bandwidth), so the
    logits processor's own `.float()` returns them as they are. The same values bit for
    bit."""

    def __init__(self, lm_head) -> None:
        self.lm_head = lm_head

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        if logits.dtype == torch.bfloat16 and logits.is_contiguous():
            return bf16_to_fp32(logits)
        return logits


class _WideningLogitsProcessor(LogitsProcessor):
    """The shell's logits processor (its row gather, its fp32 logits) over `_WidenedHead`."""

    def forward(
        self, hidden_states, lm_head, attn_metadata, return_context_logits: bool = False
    ) -> torch.Tensor:
        return super().forward(
            hidden_states, _WidenedHead(lm_head), attn_metadata, return_context_logits
        )


class ModelingV2Qwen38FlashNextNvfp4Sm103Tp1(
    DecoderModelForCausalLM[Qwen4ExpTp1ModelingV2, PretrainedConfig]
):
    @classmethod
    def get_preferred_kv_cache_manager_version(cls, pretrained_config=None) -> str:
        """Prefer KV cache manager V2: PLE state (`ple_layer_cache`) and QSA's index and
        position planes exist only there. Under V1 the model builds and passes a
        smoke run with the PLE layer stateless. The engine resolves this from the
        built-in class; the declaration serves a direct instantiation."""
        return "V2"

    def __init__(self, model_config: ModelConfig):
        cfg = model_config.pretrained_config
        assert cfg is not None

        # Adopt the engine-normalized text tower (`Qwen4ExpTextConfig`) as the model's
        # config: everything that sizes the pools reads
        # `model_engine.model.model_config.pretrained_config`, and the composite
        # parent mirrors text geometry through a pass that silently skips keys the
        # parent already defaults. The swap is in place because the engine keeps its
        # own reference to this `ModelConfig`.
        text = getattr(cfg, "text_config", None)
        assert text is not None, (
            "this target is the language tower of the composite Qwen4-Exp "
            "export; pretrained_config carries no text_config"
        )
        assert getattr(text, "architectures", None) == ["Qwen4ExpForCausalLM"], (
            f"the text tower must be the normalized Qwen4-Exp text config; got "
            f"architectures {getattr(text, 'architectures', None)!r}"
        )
        assert text.torch_dtype is not None, "the text tower must declare a dtype"
        # Captured before the swap: the core needs the vision half and media token ids.
        composite = cfg
        model_config.pretrained_config = text

        super().__init__(
            Qwen4ExpTp1ModelingV2(model_config, composite),
            config=model_config,
            hidden_size=text.hidden_size,
            vocab_size=text.vocab_size,
        )
        # The logits' bf16 -> fp32 cast as one vectorized kernel (the draft head shares it).
        self.logits_processor = _WideningLogitsProcessor()
        # Media placeholder ids for the executor; built lazily (`__init__` runs under
        # meta init).
        self._mm_token_ids_value = (self.model.image_token_id, self.model.video_token_id)
        self._mm_token_ids_cache: torch.Tensor | None = None
        # `lm_head` is outside the quantization scheme: a dense bf16 `LMHead`; a
        # quantized head would register a different parameter set.
        assert self.lm_head.weight.dtype == torch.bfloat16, (
            f"expected a bf16 lm_head; the shell built {self.lm_head.weight.dtype}, "
            "which means the engine resolved a quantized output head for a "
            "checkpoint that stores it unquantized"
        )

        # The speculative branch, built only when `speculative_config` asked for it.
        # Written out rather than inherited: the in-tree one-engine shell builds its
        # drafter via a module-level function dispatching on `model_type` (still the
        # upstream family name), which would construct trtllm's own MTP layer.
        self.spec_config = getattr(model_config, "spec_config", None)
        self.draft_model = None
        self.spec_worker = None
        # No separate draft checkpoint; the loader reads this, so None is stated.
        self.draft_config = None
        if self.spec_config is not None:
            assert self.model.mtp_enabled, "core built without the MTP module"
            assert hasattr(self, "logits_processor"), (
                "the inherited shell no longer exposes logits_processor; the "
                "draft head and the shell's own gather both project through it"
            )
            self.draft_model = DraftModel(self.model, self.lm_head, self.logits_processor)
            # Must not be left at `get_spec_worker`'s default False: the resource
            # manager decides separately (`_should_create_separate_draft_kv_cache`) and
            # builds the target manager from target layer masks alone, so left False
            # the MTP layer is handed a manager with no `layer_offsets` entry for it.
            self.use_separate_draft_kv_cache = should_use_separate_draft_kv_cache(self.spec_config)
            self.spec_worker = get_spec_worker(
                self.spec_config,
                model_config,
                model_config.mapping,
                use_separate_draft_kv_cache=self.use_separate_draft_kv_cache,
            )
            assert self.spec_worker is not None, (
                f"the engine resolved no spec worker for spec_dec_mode "
                f"{self.spec_config.spec_dec_mode!r}"
            )
            # Caches the draft->target vocab offset map (correctly None here).
            self.spec_worker.set_draft_model(self.draft_model)
            # The PLE pools and the QSA side caches are written per verify row and nothing
            # the runtime owns rolls them back; the worker calls this handler after its
            # draft loop, inside the same forward.
            self.spec_worker.register_auxiliary_state_handler(self.model.spec_rollback)

    # ----------------------------------------------------------- multimodal ---
    # Names the executor reads off a model class by `getattr`; each is silently
    # optional and its absence costs correctness rather than raising.

    @property
    def multimodal_data_device_paths(self) -> list[str]:
        """Tensors under ``multimodal_data`` the executor moves to CUDA. Absent, it moves
        everything, putting `image_grid_thw` and `video_grid_thw` on the device where
        `pos_embed_interpolate` and `vision_rope_2d` would each pay a D2H sync."""
        return [
            "image.pixel_values",
            "video.pixel_values_videos",
            "mrope_config.mrope_position_ids",
            "mrope_config.mrope_position_deltas",
        ]

    @property
    def multimodal_token_ids(self) -> torch.Tensor:
        """Placeholder ids whose rows the vision embeddings replace. Absent, the
        executor uses `input_ids >= vocab_size`, but this checkpoint's placeholders
        (248056, 248057) are in vocabulary, so nothing is selected and the engine
        generates fluent text while blind. Built on first read (meta init); not a
        parameter or buffer."""
        if self._mm_token_ids_cache is None:
            self._mm_token_ids_cache = torch.tensor(self._mm_token_ids_value, dtype=torch.int32)
        return self._mm_token_ids_cache

    def validate_request(self, request) -> None:
        """Admission check, run by the executor on every new request. A media request
        the forward would assert on fails here as that request's own error; an assert
        inside the step ends the executor under the overlap scheduler. Reads only the
        request's host-side payload, and a request without media returns at once.

        Refused: a frame past `_VISION_CALL_MAX_PATCHES` patches (the one bound the
        tower keeps), a malformed grid or pixel payload, media whose rows disagree with
        the prompt's placeholder count, missing or mis-sized `mrope_position_ids`, and a
        request with both images and videos whose item metadata is missing or does not
        list its items."""
        data = request.py_multimodal_data
        if not data:
            return
        grids = {}
        for modality, (pixel_key, grid_key) in _MEDIA_KEYS.items():
            media = data.get(modality)
            if media is not None:
                grids[modality] = self._media_grid(modality, media, pixel_key, grid_key)
        if not grids:
            return

        mrope_ids = (data.get("mrope_config") or {}).get("mrope_position_ids")
        if (
            mrope_ids is None
            or mrope_ids.dim() != 3
            or mrope_ids.shape[0] != _QSA_POSITION_AXES
            or mrope_ids.shape[2] != request.prompt_len
        ):
            raise ValueError(
                f"a media request needs the [{_QSA_POSITION_AXES}, 1, {request.prompt_len}] "
                "mrope_config.mrope_position_ids from the input processor, one position per "
                "prompt token; got "
                f"{None if mrope_ids is None else tuple(mrope_ids.shape)}"
            )

        if len(grids) > 1:
            metadata = get_multimodal_item_metadata(data)
            self._check_item_order(grids, metadata.item_refs if metadata is not None else None)

        cumsum = data.get("multimodal_embed_mask_cumsum")
        if cumsum is not None:
            merge_unit = self.model.v_merge * self.model.v_merge
            rows = sum(t * h * w for grid in grids.values() for t, h, w in grid) // merge_unit
            placeholders = int(cumsum[-1])
            if placeholders != rows:
                raise ValueError(
                    f"the prompt holds {placeholders} image/video placeholder tokens and its "
                    f"{' and '.join(sorted(grids))} payload produces {rows} embedding rows"
                )

    def _media_grid(self, modality, media, pixel_key, grid_key):
        """One modality's `(frames, height, width)` rows, checked on the host."""
        core = self.model
        pixel_values, grid_thw = media.get(pixel_key), media.get(grid_key)
        if pixel_values is None or grid_thw is None:
            raise ValueError(
                f"the {modality} payload carries {sorted(media)}; this target reads "
                f"{pixel_key!r} and {grid_key!r} from it"
            )
        if grid_thw.dim() != 2 or grid_thw.shape[0] == 0 or grid_thw.shape[1] != 3:
            raise ValueError(
                f"{grid_key} must hold one (t, h, w) row per {modality}; got shape "
                f"{tuple(grid_thw.shape)}"
            )
        grid = [tuple(int(v) for v in row) for row in grid_thw.tolist()]
        for item, (frames, height, width) in enumerate(grid):
            if min(frames, height, width) < 1 or height % core.v_merge or width % core.v_merge:
                raise ValueError(
                    f"{modality} {item} has grid {(frames, height, width)}; every entry must "
                    f"be positive and the sides divisible by spatial_merge_size {core.v_merge}"
                )
            if height * width > _VISION_CALL_MAX_PATCHES:
                raise ValueError(
                    f"{modality} {item} has frames of {height}x{width} = {height * width} "
                    f"patches ({height * core.v_patch}x{width * core.v_patch} pixels); the "
                    f"vision tower serves frames of at most {_VISION_CALL_MAX_PATCHES} "
                    "patches, so send it at a lower resolution"
                )
        patches = sum(t * h * w for t, h, w in grid)
        if tuple(pixel_values.shape) != (patches, core.v_pixel_width):
            raise ValueError(
                f"{pixel_key} is {tuple(pixel_values.shape)}; its grids describe "
                f"{patches} patch rows of {core.v_pixel_width}"
            )
        return grid

    @staticmethod
    def _check_item_order(grids, order) -> None:
        """A request with several modalities lists its items in prompt order."""
        if not order:
            raise ValueError(
                f"a request carrying {sorted(grids)} needs item metadata to splice them in "
                "prompt order; the payload holds each modality's rows apart"
            )
        listed = {modality: [] for modality in grids}
        for entry in order:
            try:
                modality, index = entry
                listed[modality].append(int(index))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"item_refs entry {entry!r} is not an item of this request, which "
                    f"carries {sorted(grids)}; entries are (modality, index) pairs"
                ) from error
        for modality, grid in grids.items():
            if sorted(listed[modality]) != list(range(len(grid))):
                raise ValueError(
                    f"item_refs lists {modality} items {listed[modality]}; the payload "
                    f"carries {len(grid)}"
                )

    def _multimodal_embeds(self, kwargs):
        """Run the vision tower for this batch; returns `(rows, mm_token_indices)` (the
        tower's `[R, hidden]` rows in packed-batch order and the `[R]` placeholder rows they
        replace, which the prefill target splices into its embedding), or None when the step
        carries no media.

        Lives in the shell because the inherited `DecoderModelForCausalLM.forward`
        drops `multimodal_params` from `**kwargs` on the no-speculation path (a
        booting, fluent, blind engine). Host-side bookkeeping: `multimodal_params`
        is a subset of the context requests, so never zip it positionally;
        `multimodal_runtime` (chunked prefill / KV reuse only) gives the cached and
        in-chunk row counts over the request's images and videos together, in prompt
        order; `mm_token_indices` names every placeholder row.

        The tower runs once per modality a request carries, and `_in_prompt_order`
        merges the two outputs. Positions are not derived here: every coordinate of
        a vision span, a video's temporal one included, is in the payload's
        `mrope_config.mrope_position_ids`, which the engine slices into
        `position_ids`."""
        mm_params = kwargs.get("multimodal_params") or []
        if not mm_params:
            return None
        rows = []
        for params in mm_params:
            data = getattr(params, "multimodal_data", None) or {}
            image, video = data.get("image"), data.get("video")
            if image is None and video is None:
                continue
            mrope_ids = (data.get("mrope_config") or {}).get("mrope_position_ids")
            assert (
                mrope_ids is not None
                and mrope_ids.dim() == 3
                and mrope_ids.shape[0] == _QSA_POSITION_AXES
            ), (
                f"a media request needs mrope_config.mrope_position_ids of shape "
                f"[{_QSA_POSITION_AXES}, 1, L]; got "
                f"{None if mrope_ids is None else tuple(mrope_ids.shape)}"
            )
            runtime = getattr(params, "multimodal_runtime", None)
            if runtime is not None and int(runtime.num_mm_tokens_in_chunk) == 0:
                # Every placeholder row of this request lies in another chunk.
                continue
            towers = {}
            if image is not None:
                towers["image"] = self._run_tower(image, *_MEDIA_KEYS["image"])
            if video is not None:
                towers["video"] = self._run_tower(video, *_MEDIA_KEYS["video"])
            embeds = self._in_prompt_order(params, towers)
            if runtime is not None:
                # Chunked prefill or reused prefix: splice only this chunk's slice.
                start = int(runtime.num_cached_mm_tokens)
                count = int(runtime.num_mm_tokens_in_chunk)
                total = embeds.shape[0]
                assert int(runtime.total_embeds_in_request) == total, (
                    f"the executor counted {runtime.total_embeds_in_request} "
                    f"placeholder rows for this request and the tower produced "
                    f"{total} over its {sorted(towers)}; the two are derived from the "
                    "same grids and a disagreement means the merger's row count is wrong"
                )
                assert 0 <= start <= start + count <= total, (start, count, total)
                embeds = embeds[start : start + count]
            rows.append(embeds)
        if not rows:
            return None
        vision = rows[0] if len(rows) == 1 else torch.cat(rows, 0)
        mm_token_indices = kwargs.get("mm_token_indices")
        assert mm_token_indices is not None, (
            "the executor shipped multimodal params but no mm_token_indices; "
            "that pairing is what `multimodal_token_ids` on this class buys, and "
            "without it there is no sync-free way to find the placeholder rows"
        )
        assert mm_token_indices.shape[0] == vision.shape[0], (
            f"the batch has {mm_token_indices.shape[0]} placeholder rows and the "
            f"tower produced {vision.shape[0]} embedding rows"
        )
        return vision, mm_token_indices

    def _run_tower(self, media, pixel_key, grid_key):
        """The tower over one modality's payload: `(rows, grid_thw)`, the rows one per
        placeholder, item by item in the order the processor emitted them."""
        pixel_values, grid_thw = media.get(pixel_key), media.get(grid_key)
        assert pixel_values is not None and grid_thw is not None, (
            f"the multimodal payload carries {sorted(media)}; this target "
            f"reads {pixel_key!r} and {grid_key!r} from it"
        )
        assert not grid_thw.is_cuda, (
            f"{grid_key} must stay on the host; see multimodal_data_device_paths"
        )
        return self.model.vision_tower(pixel_values, grid_thw), grid_thw

    def _in_prompt_order(self, params, towers):
        """One request's tower rows in the order its placeholders appear.

        `towers` maps each modality the request carries to `(rows, grid_thw)`; each
        grid row is one item of `t*h*w / merge**2` rows. One modality is already in
        prompt order. Two are reassembled item by item from `item_metadata.item_refs`
        (`(modality, index)` per item in prompt order, `index` counting that
        modality's items); without it there is no host-side way to tell which
        placeholder run is an image and which a video."""
        if len(towers) == 1:
            return next(iter(towers.values()))[0]
        metadata = params.item_metadata
        order = metadata.item_refs if metadata is not None else None
        assert order, (
            f"a request carrying {sorted(towers)} needs item metadata to splice them "
            "in prompt order; the payload holds each modality's rows apart"
        )
        merge_unit = self.model.v_merge * self.model.v_merge
        items = {}
        for modality, (embeds, grid_thw) in towers.items():
            counts = [t * h * w // merge_unit for t, h, w in grid_thw.tolist()]
            assert sum(counts) == embeds.shape[0], (
                f"the {modality} grid describes {sum(counts)} rows and the tower "
                f"produced {embeds.shape[0]}"
            )
            items[modality] = torch.split(embeds, counts, 0)
        listed = {modality: [] for modality in items}
        for modality, index in order:
            assert modality in items, (
                f"item_refs lists a {modality!r} item; this request carries {sorted(items)}"
            )
            listed[modality].append(int(index))
        for modality, parts in items.items():
            assert sorted(listed[modality]) == list(range(len(parts))), (
                f"item_refs lists {modality} items {listed[modality]}; the "
                f"payload carries {len(parts)}"
            )
        return torch.cat([items[modality][int(index)] for modality, index in order], 0)

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
        """One engine step.

        Without a spec worker this delegates to the base forward with every argument
        it was handed (`resource_manager` stays in `**kwargs` deliberately: naming
        it would drop it from what the base receives). With one, the shell owns the
        logits gather (one-model modes set `without_logits`) and the worker needs
        ungathered hidden states; `position_ids` is passed in the engine's layout
        (flattening gives silently wrong draft positions). The vision tower runs here
        because the base forward would not pass `multimodal_params` through; its rows reach
        the core through `begin_step`, and the prefill target splices them."""
        self.model.begin_step(
            attn_metadata,
            input_ids,
            position_ids,
            inputs_embeds=inputs_embeds,
            lora_params=lora_params,
            spec_metadata=spec_metadata,
            vision=self._multimodal_embeds(kwargs),
        )
        if self.spec_worker is None:
            assert spec_metadata is None, (
                "spec_metadata arrived without a spec worker; this engine was "
                "built without a speculative_config"
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
        assert spec_metadata is not None, (
            "the spec worker is built but the engine passed no spec_metadata"
        )
        # `final` is the `[T, hidden]` mix for `lm_head`, `hc` the stream the MTP reads.
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
        # `gather_ids`: one row per context request, `runtime_draft_len + 1` per
        # generation request.
        logits = self.logits_processor.forward(
            nn.functional.embedding(spec_metadata.gather_ids, final),
            self.lm_head,
            attn_metadata,
            True,
        )
        # A prefill CUDA graph pads the step's rows; the worker reshapes generation rows per
        # request, so it takes the batch's own.
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
        _weights.MODEL_WEIGHTS.load(self, weights)

    def post_load_weights(self):
        super().post_load_weights()
        self.model.post_load_weights()
