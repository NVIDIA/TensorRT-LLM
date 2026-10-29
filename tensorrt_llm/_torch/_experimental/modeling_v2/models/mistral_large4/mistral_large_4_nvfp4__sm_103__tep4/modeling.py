# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: mistral-large-4-nvfp4 / sm_103 / tep4, native text and image inference.

72 layers, hidden 6144, 64 query heads (16 per rank at tep4, 8 at tep8). Attention is MLA
without a q-LoRA (`wq` straight to 64 x 128), a 512-wide latent plus a 64-wide rope key, and
DeepSeek Sparse Attention: an indexer on the 44 `index_sharing == 'full'` layers selects 2048
positions per query row and each 'shared' layer reuses the selection of the most recent full
layer. Every fourth layer (`i % 4 == 3`) takes no rope, for MLA and for its indexer; the NoPE
layers 39, 43, ..., 71 own no latent and attend over layer 35's. Queries are scaled by SSMax,
`s * log(pos + 1) + b` per head, on MLA and on the indexer. Layer 0 is a dense SwiGLU MLP,
layers 1-71 MoE (128 experts, softmax top-4 renormalised, one shared expert). The final logits
are soft-capped at `softmax_tanh_gating`.

The checkpoint is ModelOpt `MIXED_PRECISION`. The routed experts are NVFP4 W4A4: e2m1 weights
with one e4m3 scale per 16 inputs and a global scale per expert, and activations quantized to
NVFP4 under one static global scale per layer and projection. `wq`, `wo`, the indexer's `wq_b`
and the dense and shared MLPs are FP8 e4m3 with 128x128 power-of-two block scales and dynamic
1x128 activation scales. The rest is bf16. The latent KV pool is fp8_ds_mla (656 B per token
and owner layer: 512 e4m3 codes with one power-of-two fp32 scale per 128, and the 64-wide rope
key in bf16), owned by `MistralLarge4CacheManager`; the indexer key pool is fp8 with one scale
per key.

`tep4` shards the attention heads, the dense and shared MLPs and `lm_head` over every rank and
gives each rank `n_routed_experts / moe_ep_size` of the 128 routed experts: four ways and 32
experts on one node of four, or eight ways and 16 experts over two nodes of four (tep8), every
width read off the mapping. The latent projection, the indexer and the router are replicated.
Every layer closes twice on the all-reduce over every rank fused with the residual add and the
next RMSNorm.

The absorbed form throughout: `fused_q = [q_nope @ W_UK | rope(q_pe)]` (576 wide) against the
pooled `[c_kv | k_pe]` latent rows, then `@ W_UV`. Context rows attend over the selection (the
engine's causal index on a step it skipped the indexer) with FlashMLA's bf16 sparse prefill
over the requests' latent rows dequantized out of the pool, so they read the FP8-rounded latent
as the reference does; generation rows run sparse fp8 decode over the pool itself, at this
rank's 16 (tep4) or 8 (tep8) heads at every max_seq_len: cute_dsl_ds_mla_sparse_decode,
FlashMLA's sparse decode arithmetic without the head padding. The prefill kernel, and the
sparse decode a context row past the held workspace runs (FlashMLA's), take 64 heads: this
rank's go first in a 64-head query whose other heads are zero; heads do not interact, so the
zero heads change nothing the live ones compute. `_GEN_ATTN_FLASH_MLA` (for A/B verification)
runs the generation rows on FlashMLA's 64-head decode too, the same padding.

No path is selected by the serving configuration: each step chooses from its own sizes, and
every buffer is sized from the engine's quantities (the attention metadata's max_num_tokens,
max_num_requests, max_seq_len and runtime_features). The context rows take one of two paths,
chosen per step by the step's context KV (every context request's cached and new rows): while
it fits the held workspace, the sparse prefill over those rows dequantized into it; past it,
each context row as a one-row sequence of the sparse fp8 decode over the pool itself, as
generation rows always run. The held workspace, allocated on the first context step, has room
for the most context KV one step can attend, twice over, up to 2**19 rows
(`_ctx_workspace_rows`): a step whose context KV fits twice keeps layer 35's rows beside its
own for the NoPE layers attending over that latent, a longer one gathers them again for each
of those layers. An indexer chunk the engine packs past `max_seq_len` keys is scored request
by request, and the indexer's logits are tiled under a budget from the engine's max_num_tokens
and max_seq_len (`_prefill_logits_budget`), each tile scored with a split-KV schedule built once
per step (every SM busy whatever the tile's row count) and selected with the self-sampling GVR
top-k (`_select_ctx`).

Under chunked prefill the engine splits a step's context queries across every rank (four, or
eight over two nodes) from `q_split_threshold` rows on: each rank selects for its share, and the
shares are all-gathered into every rank's selection buffer before a layer reads it.
"""

from __future__ import annotations

import itertools
import math
import os
import time
from dataclasses import replace
from types import SimpleNamespace
from typing import NamedTuple, Optional, cast

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import step_contract_enabled
from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.flashinfer_silu_and_mul import (
    flashinfer_silu_and_mul,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.gelu_erf import gelu_erf
from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.tanh_softcap import (
    tanh_softcap,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.convert_req_index_to_global import (
    convert_req_index_to_global,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.cute_dsl_ds_mla_sparse_decode import (
    build as build_ds_mla_decode,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.cute_dsl_ds_mla_sparse_decode import (
    cute_dsl_ds_mla_sparse_decode,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.cute_dsl_fp8_paged_mqa_logits import (
    cute_dsl_fp8_paged_mqa_logits,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.cute_dsl_gvr_topk_prefill import (
    cute_dsl_gvr_topk_prefill,
    gvr_warmup,
    gvr_workspace,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.cute_dsl_indexer_k_cache_update import (
    build as build_ix_k_cache,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.cute_dsl_indexer_k_cache_update import (
    cute_dsl_indexer_k_cache_update,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.cute_dsl_indexer_q_side_fused import (
    build as build_ix_q_side,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.cute_dsl_indexer_q_side_fused import (
    cute_dsl_indexer_q_side_fused,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.cute_dsl_radix_topk_decode import (
    build as build_ix_topk_decode,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.cute_dsl_radix_topk_decode import (
    cute_dsl_radix_topk_decode,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.ds_mla_dequant_gather import (
    ds_mla_dequant_gather,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.ds_mla_rope_append import (
    ds_mla_rope_append,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.flash_mla_sparse_decode import (
    flash_mla_sparse_decode,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.flash_mla_sparse_prefill import (
    flash_mla_sparse_prefill,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.fp8_mqa_logits import (
    fp8_mqa_logits,
    mqa_logits_schedule,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.indexer_k_cache_gather_op import (
    indexer_k_cache_gather_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.indexer_topk_decode import (
    indexer_topk_decode,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.ssmax_indexer_q_side_fused import (
    ssmax_indexer_q_side_fused,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.ssmax_scaling import (
    ssmax_scaling,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.ssmax_scaling_fused_q import (
    ssmax_scaling_fused_q,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
    thop_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.allgather import allgather
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.cute_dsl_allreduce_add_rmsnorm import (
    build_workspace as build_lamport_workspace,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.cute_dsl_allreduce_add_rmsnorm import (
    cute_dsl_allreduce_add_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.bmm_out import bmm_out
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.fp8_prequantized_swap_ab_gemm import (
    fp8_prequantized_swap_ab_gemm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (
    fp4_block_scale_moe_runner,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.renorm_moe_routing_op import (
    renorm_moe_routing_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (
    flashinfer_fused_add_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.layer_norm import layer_norm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp8_quantize_1x128_packed_ue8m0 import (
    fp8_quantize_1x128_packed_ue8m0,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.vision.pixtral_rope_2d import (
    pixtral_rope_2d,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.sparse.params import (
    get_indexer_mqa_logits_workspace_bytes,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_multimodal_utils import (
    find_input_mm_embeds,
    get_multimodal_embeddings,
)
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.pyexecutor.guided_decoder import CapturableGuidedDecoder
from tensorrt_llm._torch.speculative import get_spec_worker
from tensorrt_llm._utils import mpi_disabled
from tensorrt_llm.flash_mla import get_mla_metadata
from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig

from . import weights as _weights

#: eps 1e-5, the reference's value; params.json's `vision_encoder` states none.
_VISION_EPS = 1e-5

#: The thop attention arguments the vision tower's call holds fixed: the no-KV-cache
#: mode (an encoder: the three cache-addressing arguments None), one context sequence
#: per image, the fused `[q | k | v]` rows already rotated (no rope in the op),
#: bidirectional within each sequence (mask_type 0), softmax scale head_dim^-0.5.
_VISION_ATTN = dict(
    k=None,
    v=None,
    output_sf=None,
    max_context_q_len_override=None,
    kv_cache_block_offsets=None,
    host_kv_cache_pool_pointers=None,
    host_kv_cache_pool_mapping=None,
    cache_indirection=None,
    kv_scale_orig_quant=None,
    kv_scale_quant_orig=None,
    out_scale=None,
    rotary_inv_freq=None,
    rotary_cos_sin=None,
    latent_cache=None,
    q_pe=None,
    block_ids_per_seq=None,
    attention_sinks=None,
    is_fused_qkv=True,
    update_kv_cache=True,
    predicted_tokens_per_seq=1,
    local_layer_idx=0,
    tokens_per_block=None,
    beam_width=1,
    mask_type=0,
    quant_mode=0,
    q_scaling=1.0,
    position_embedding_type=0,
    rope_dim=0,
    rope_base=10000.0,
    rope_scale_type=0,
    rope_scale=1.0,
    rope_short_m_scale=1.0,
    rope_long_m_scale=1.0,
    rope_max_positions=1024,
    rope_original_max_positions=1024,
    use_paged_context_fmha=False,
    attention_input_type=0,
    is_mla_enable=False,
    chunked_prefill_buffer_batch_size=1,
    q_lora_rank=None,
    kv_lora_rank=None,
    qk_nope_head_dim=None,
    qk_rope_head_dim=None,
    v_head_dim=None,
    rope_append=None,
    mrope_rotary_cos_sin=None,
    mrope_position_deltas=None,
    helix_position_offsets=None,
    helix_is_inactive_rank=None,
    attention_chunk_size=None,
    softmax_stats_tensor=None,
    is_spec_decoding_enabled=False,
    use_spec_decoding=False,
    is_spec_dec_tree=False,
    spec_decoding_generation_lengths=None,
    spec_decoding_position_offsets_for_cpp=None,
    spec_decoding_packed_mask=None,
    spec_decoding_bl_tree_mask_offset=None,
    spec_decoding_bl_tree_mask=None,
    spec_bl_tree_first_sparse_mask_offset_kv=None,
    sparse_kv_indices=None,
    sparse_kv_offsets=None,
    sparse_attn_indices=None,
    sparse_attn_offsets=None,
    sparse_attn_indices_block_size=0,
)

# The latent pool layout this target addresses (`MistralLarge4CacheManager.latent_layout`).
_LATENT_LAYOUT = "fp8_ds_mla"
# The query heads FlashMLA's sm100 sparse kernels run; this rank's heads go first.
_FLASH_MLA_HEADS = 64
# Generation rows' sparse MLA. False: cute_dsl_ds_mla_sparse_decode at this rank's heads, the
# shipped path. True: flash_mla_sparse_decode over the 64-head padded query, the kernel this
# target ran before (bit-identical where the two split a row alike; see the entry's note) --
# a switch for A/B verification, not a configuration. It sets the core's
# `gen_attn_flash_mla`, which every generation-row site reads; the native entry's cells cover
# the 32K and the 1M configs (key lengths to 1048576, the 1M pools' slots past 2**31 bytes).
_GEN_ATTN_FLASH_MLA = False
# The pre-routed MoE call: the runner's routing method is inert once ids and weights are
# given; 1 (renormalize) is the value its cells drive.
_MOE_ROUTING_METHOD = 1
# The routed experts' SwiGLU, the runner's `act_type` 0.
_MOE_ACT_SWIGLU = 0
# NVFP4: one e4m3 scale per 16 activations, laid out row-major (linear), the order the MoE
# runner reads; the 128x4-swizzled order is another buffer of the same length at 128-row
# multiples, and silently wrong there.
_NVFP4_SF_VEC = 16
_NVFP4_SF_LINEAR = False
# fp8_mqa_logits allocates a `[q_rows, kv_rows]` fp32 score matrix per call; the context
# queries are tiled to keep it under the engine's budget (`_prefill_logits_budget`). Context
# steps are not captured.
# capacity: fixed: the budget's ceiling, one 1 GiB fp32 tile (256 rows at 1048576 keys).
# Mainline's default reachable tile (2**31 logits) would take 8 GiB from the KV pool.
_PREFILL_LOGITS_CEILING = 1 << 28
# The context path's crossover, chosen once per step on host ints, and the held context
# workspace's ceiling (`_ctx_workspace_rows`). A step whose context KV (every context request's
# cached and new rows) fits the held workspace has it gathered, dequantized, into it (bf16
# [rows, 576]) for FlashMLA's sparse prefill; a longer one attends its context rows straight off
# the pool with the sparse decode kernel, whose cost does not grow with the prefix as the
# gather's does.
# capacity: fixed: 2**19 rows, 576 MiB, the gather-vs-decode crossover.
_CTX_WORKSPACE_ROWS = 1 << 19
# The most context rows one sparse-decode call takes on that path.
# capacity: fixed: FlashMLA's decode scheduler, which exits the process, raising nothing, past
# 11622 one-row sequences on sm_103; not an engine limit.
_CTX_DECODE_ROWS = 8192
# The decode indexer's top-k: True runs cute_dsl_radix_topk_decode, one launch that selects the
# trtllm indexer_topk_decode op's set (up to exact ties), False that op itself. The CuTe DSL
# selector serves scorer rows from 4096 columns (where the op splits each row) to 65536 (one
# CTA scans a whole row), the widths its cells cover; the scorer's column count is the engine's
# max_seq_len, so an engine outside that range keeps the op (`_derive_first_step`).
_TOPK_DECODE_CUTE_DSL = True
_TOPK_DECODE_COLUMNS = (4096, 65536)
# The attention block's glue as separate ops instead of the fused entries, for old-versus-new
# verification (both compute the same bits; the fused entries' tests hold each pair equal):
# ssmax_scaling plus the rope tail's copy or zero fill for ssmax_scaling_fused_q;
# ssmax_scaling, the head weights' fp32 copy and cute_dsl_indexer_q_side_fused for
# ssmax_indexer_q_side_fused; a fresh RMSNorm and torch.cat for the in-place latent; the copy
# out of FlashMLA's 64-head output for the in-place read. Set
# TRTLLM_MODELING_V2_UNFUSED_GLUE=1 in every rank's environment at launch; off by default.
# It sets the core's `unfused_glue`, which every site reads.
_UNFUSED_GLUE = os.environ.get("TRTLLM_MODELING_V2_UNFUSED_GLUE", "0") == "1"


def _prefill_tiles(
    token_start: int, token_end: int, n_k: int, budget: int
) -> list[tuple[int, int]]:
    """The query-row tiles `[lo, hi)` of one indexer prefill chunk over `n_k` keys under a
    logits `budget`: `floor(budget / n_k)` rows each (at least one), the last one short."""
    tile = max(1, budget // n_k)
    return [(lo, min(lo + tile, token_end)) for lo in range(token_start, token_end, tile)]


def _prefill_logits_budget(max_num_tokens: int, max_seq_len: int) -> int:
    """The most fp32 logits one indexer prefill tile holds, from the engine's max_num_tokens and
    max_seq_len: mainline's reachable MQA-logits tile (`get_indexer_mqa_logits_workspace_bytes`,
    4 bytes a logit; TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET lowers it), within
    `_PREFILL_LOGITS_CEILING`, and within max_seq_len / 64 query rows at max_seq_len keys.

    The last bound grows the tile with the engine's longest request, which it scores in at
    least 64 tiles: 2**24 logits (512 rows of 32768 keys) at max_seq_len 32768, the ceiling
    (256 rows of 1048576 keys) from max_seq_len 131072 on. Those are the budgets the 32768 and
    1048576 serving configs were certified at, so their tiles, and the fp8_mqa_logits and
    prefill top-k cells that cover them, stay as they were."""
    reachable = get_indexer_mqa_logits_workspace_bytes(max_num_tokens, max_seq_len) // 4
    return min(reachable, _PREFILL_LOGITS_CEILING, max_seq_len * (max_seq_len >> 6))


def _most_ctx_kv(md) -> int:
    """The most context KV (cached and new rows over a step's context requests) one step can
    attend, from the attention metadata alone. `runtime_features` is built once by the executor
    for the service (chunked prefill from enable_chunked_prefill, cache reuse from the
    KvCacheConfig's enable_block_reuse). With neither, every context row is new:
    max_num_tokens. Otherwise mainline's never-stall worst case for a workspace chunked prefill
    does not bound (`get_mla_context_workspace_kv_len_cap` without an
    fp8_context_mla_kv_len_cap override; the gather reads every attended row, cached ones
    included): min(max_num_requests, max_num_tokens) * max_seq_len. The target does not read
    fp8_context_mla_kv_len_cap: the KvCacheConfig is not reachable from the model, and for these
    targets the engine installs no admission cap that would bound a step to it."""
    features = md.runtime_features
    if not (features.chunked_prefill or features.cache_reuse):
        return md.max_num_tokens
    return min(md.max_num_requests, md.max_num_tokens) * md.max_seq_len


def _ctx_workspace_rows(md) -> int:
    """The held context workspace's rows: the most context KV one step can attend
    (`_most_ctx_kv`) twice over, a step's rows and a shared owner's (layer 35's) kept beside
    them, up to `_CTX_WORKSPACE_ROWS`. At max_num_tokens 32768 with neither chunked prefill nor
    block reuse that is 65536 rows, the two `[32768, 576]` gathers a step there makes; with either,
    at 16 requests of max_seq_len 32768 or more, 2**19."""
    return min(2 * _most_ctx_kv(md), _CTX_WORKSPACE_ROWS)


def _step_common(md: TrtllmAttentionMetadata, indexer_layers: list[int]) -> dict:
    """The batch state both targets read: KV lengths and pool addressing for the MLA ops, the
    indexer's key pools and slot mappings, the top-k conversion tables and the decode scorer's
    schedule. The indexer pool views are taken per step: the engine replaces its KV cache
    manager during boot, and a view of the old one writes into a dead allocation."""
    mgr = md.kv_cache_manager
    latent_pool = mgr.get_latent_pool()
    return dict(
        num_seqs=md.num_seqs,
        num_tokens=md.num_tokens,
        kv_lens_cuda=md.kv_lens_cuda_runtime,
        kv_lens_host=md.kv_lens_runtime,
        prompt_lens_host=md.prompt_lens_cpu_runtime,
        kv_cache_block_offsets=md.kv_cache_block_offsets,
        host_kv_cache_pool_pointers=md.host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping=md.host_kv_cache_pool_mapping,
        tokens_per_block=md.tokens_per_block,
        max_seq_len=md.max_seq_len,
        # [blocks, owners, tokens_per_block, 656] bytes, and the same bytes as FlashMLA's
        # pages: page `block * owners + row`, slot `page * tokens_per_block + offset`.
        latent_pool=latent_pool,
        latent_kv=latent_pool.view(-1, latent_pool.shape[2], 1, latent_pool.shape[3]),
        ix_pools={i: mgr.get_indexer_k_cache_buffers(i) for i in indexer_layers},
        block_table=md.block_table,
        req_idx_per_token=md.req_idx_per_token,
        slot_mapping_fp8=md.slot_mapping_fp8,
        slot_mapping_scale=md.slot_mapping_scale,
        skip_gen=bool(md.skip_indexer_for_gen_reqs),
        # Allocated only while the engine's short-sequence indexer skip is on; the two skip
        # flags are False whenever it is off.
        topk_indices_buffer=md.topk_indices_buffer if md.enable_indexer_skip else None,
        indexer_k_cache_block_offsets=md.indexer_k_cache_block_offsets,
        scheduler_metadata_buffer=md.scheduler_metadata_buffer,
        radix_aux_indices=md.radix_aux_indices,
        radix_aux_logits=md.radix_aux_logits,
    )


def _ctx_kv_offsets(md: TrtllmAttentionMetadata, num_contexts: int) -> dict:
    """The context rope/append op's per-request offsets: int64 prefix sums of the cached and
    of the total (cached + new) lengths over the context requests, and the longest new run.

    The engine keeps them only under `enable_context_mla_with_cached_kv` (KV reuse or chunked
    prefill). Without either, no context request has a cached prefix, so the cached sums are
    zero and the totals are the new lengths: the int32 prefix sum of `seq_lens_cuda` the
    metadata always builds, widened to the op's int64.
    """
    if md.enable_context_mla_with_cached_kv:
        return dict(
            cu_ctx_cached=md.ctx_cached_token_indptr[: num_contexts + 1],
            cu_ctx_kv=md.ctx_kv_indptr[: num_contexts + 1],
            max_ctx_new=int(md.max_ctx_seq_len),
            ctx_cached_tokens=int(md.num_ctx_cached_tokens),
        )
    cu = md.mla_prepare_ctx_cu_seqlens()
    return dict(
        cu_ctx_cached=torch.zeros(num_contexts + 1, dtype=torch.int64, device=cu.device),
        cu_ctx_kv=cu.to(torch.int64),
        max_ctx_new=int(md.seq_lens[:num_contexts].max()),
        ctx_cached_tokens=0,
    )


def _ctx_host_offsets(md: TrtllmAttentionMetadata, num_contexts: int) -> dict:
    """The host lists over the context requests: the prefix sums of the new and of the total
    (cached + new) lengths, which place each context request's rows in the step and its keys in
    the indexer's full-KV gather mappings (`_indexer_blocks`).

    The engine's host copies under `enable_context_mla_with_cached_kv`; without it no request
    has a cached prefix, so both are the prefix sum of the host `seq_lens`. Every source is a
    host tensor the metadata fills on the host, so taking them each context step syncs nothing.
    """
    if md.enable_context_mla_with_cached_kv:
        return dict(
            host_cu_ctx_new=md.host_ctx_uncached_token_indptr[: num_contexts + 1].tolist(),
            host_cu_ctx_kv=md.host_ctx_kv_indptr[: num_contexts + 1].tolist(),
        )
    host_cu = [0, *itertools.accumulate(md.seq_lens[:num_contexts].tolist())]
    return dict(host_cu_ctx_new=host_cu, host_cu_ctx_kv=host_cu)


def _indexer_blocks(chunks, step: dict) -> list:
    """The context indexer's scoring blocks for one step, in the engine's chunk order: rows
    `[token_start, token_end)` of the context rows scored against keys
    `[k_token_start, k_token_end)` of the full-KV gather mappings, each row's window
    (`cu_seqlen_ks` / `cu_seqlen_ke`) counted from the block's first key.

    An engine prefill chunk is one block, as planned, while it holds at most `max_seq_len` keys,
    the most one request holds. The engine packs consecutive context requests into one chunk
    while (summed new tokens) x (summed KV) stays within its packed-logits budget (2^28), so
    short last chunks of long requests pack past max_seq_len: a 100-token tail of a
    1048559-token request beside a 150-token prompt is one chunk of 1048709 keys, wider than
    the key scorer's and the prefill top-k's cells. Such a chunk is cut at its request
    boundaries (`_ctx_host_offsets`), each request's rows against its own keys with their
    windows re-based to its first key: a row's window lies inside its own request's keys, so
    each request selects as it would alone and no block holds more than max_seq_len keys.
    """
    tc = step["num_ctx_tokens"]
    if chunks is None:
        # One chunk over every context row: the engine plans chunks only past its chunk
        # size, or under KV reuse or chunked prefill.
        chunks = [
            SimpleNamespace(
                token_start=0,
                token_end=tc,
                k_token_start=0,
                k_token_end=step["num_ctx_kv_tokens"],
                cu_seqlen_ks=step["cu_seqlen_ks"][:tc],
                cu_seqlen_ke=step["cu_seqlen_ke"][:tc],
            )
        ]
    max_keys = step["max_seq_len"]
    cu_new, cu_kv = step["host_cu_ctx_new"], step["host_cu_ctx_kv"]
    blocks = []
    for chunk in chunks:
        if chunk.k_token_end - chunk.k_token_start <= max_keys:
            blocks.append(chunk)
            continue
        for r in range(len(cu_new) - 1):
            lo, hi = max(chunk.token_start, cu_new[r]), min(chunk.token_end, cu_new[r + 1])
            if lo >= hi:
                continue
            k_lo, k_hi = max(chunk.k_token_start, cu_kv[r]), min(chunk.k_token_end, cu_kv[r + 1])
            assert k_hi - k_lo <= max_keys, (
                f"context request {r} holds keys [{k_lo}, {k_hi}) of an indexer chunk, past "
                f"max_seq_len {max_keys}"
            )
            shift = k_lo - chunk.k_token_start
            rows = slice(lo - chunk.token_start, hi - chunk.token_start)
            blocks.append(
                SimpleNamespace(
                    token_start=lo,
                    token_end=hi,
                    k_token_start=k_lo,
                    k_token_end=k_hi,
                    cu_seqlen_ks=torch.sub(chunk.cu_seqlen_ks[rows], shift),
                    cu_seqlen_ke=torch.sub(chunk.cu_seqlen_ke[rows], shift),
                )
            )
    return blocks


def _yarn_lerope_inv_freq(cfg, alpha: torch.Tensor) -> torch.Tensor:
    """The rotary inverse frequencies, fp32, as the reference implementation builds
    them: YaRN's blend of the extrapolated and the interpolated frequency, both scaled by
    LeRoPE's `exp(alpha)`, over a ramp taken on the frequencies without LeRoPE."""
    rope = cfg.rope_parameters
    dim = cfg.qk_rope_head_dim
    theta = float(rope["rope_theta"])
    factor = float(rope["factor"])
    original = float(rope["original_max_position_embeddings"])
    pos_freqs = theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    lerope = torch.exp(alpha.detach().to(device="cpu", dtype=torch.float32))
    extrapolated = 1.0 / pos_freqs * lerope
    interpolated = 1.0 / (factor * pos_freqs) * lerope

    def correction_dim(rotations: float) -> float:
        return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(theta))

    low = max(math.floor(correction_dim(float(rope.get("beta_fast", 32)))), 0)
    high = min(math.ceil(correction_dim(float(rope.get("beta_slow", 1)))), dim - 1)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
    extrapolation = 1 - ramp
    return interpolated * (1 - extrapolation) + extrapolated * extrapolation


def _rope_tables(cfg, alpha: torch.Tensor, positions: int, dtype, device) -> SimpleNamespace:
    """The tables the rope consumers read, built from one fp32 angle grid and rounded to the
    activation dtype first, as the reference's cos/sin cache is held.

    `indexer` is `[rows, 64]`, `[cos(32) | sin(32)]`: the layout the indexer q side and
    `ds_mla_rope_append` (the MLA query tail and the latent's rope key) read; the indexer key
    side reads the same storage as `[rows, 2, 32]`. `identity` (cos 1, sin 0) is the table of a
    NoPE indexer layer, whose kernels have no no-rope mode. Every row depends only on its own
    index, so a longer table is the same table with more rows.
    """
    inv = _yarn_lerope_inv_freq(cfg, alpha)
    t = torch.arange(positions, dtype=torch.float32)
    angle = torch.einsum("i,j -> ij", t, inv)
    cos = angle.cos().to(dtype).float()
    sin = angle.sin().to(dtype).float()
    identity = torch.cat([torch.ones_like(cos), torch.zeros_like(sin)], dim=1)
    return SimpleNamespace(
        rows=positions,
        indexer=torch.cat([cos, sin], dim=1).to(device).contiguous(),
        identity=identity.to(device).contiguous(),
    )


def _moe_scalars(isc1, isc2, ws2_13, ws2_2) -> SimpleNamespace:
    """The NVFP4 MoE's call scalars from the checkpoint's per-tensor ones, fp32.

    ModelOpt stores `input_scale` and `weight_scale_2` as the reciprocals of the global scales a
    quantizer divides by (`amax / (448 * 6)`), one `isc1` / `isc2` per layer and one
    `ws2_13[e]` / `ws2_2[e]` per local expert. So the activation quantizer's global scale is
    `1 / isc1`; FC1's accumulator dequantizes by `isc1 * ws2_13[e]` (the gate half,
    `output1_scale_gate_scalar`) and, because the FC1 epilogue requantizes the SwiGLU output to
    NVFP4 under `1 / isc2`, the linear half by `isc1 * ws2_13[e] / isc2`
    (`output1_scale_scalar`); FC2's accumulator dequantizes by `isc2 * ws2_2[e]`
    (`output2_scale_scalar`). The reference's TRT-LLM-Gen path folds the same three
    (`g1_alphas`, `g1_scale_c`, `g2_alphas`)."""
    gate1 = (isc1 * ws2_13).contiguous()
    return SimpleNamespace(
        gscale=(1.0 / isc1).contiguous(),
        out1=(gate1 / isc2).contiguous(),
        out1_gate=gate1,
        out2=(isc2 * ws2_2).contiguous(),
    )


def _layer_views(core: "MistralLarge4Tep4ModelingV2") -> list[SimpleNamespace]:
    """Every layer's operands, as the views the entries read: GEMM operands transposed (`.t()`,
    the column-major `[K, N]` cuBLAS and the FP8 GEMM read), the FP8 scale storage transposed to
    the MN-major operand, `wkv_b` split into the absorption pair, and the NVFP4 MoE's call
    scalars (`_moe_scalars`). `next_norm` is the next layer's input norm, the final norm after
    the last layer. A sharer's absorption pair is its own `wkv_b`; its latent is its source's."""
    w = core.w
    heads, v_dim, kv_lora = core.heads, core.v_dim, core.kv_lora
    layers = []
    for i in range(core.num_layers):
        p = f"l{i}_"
        nope = core.qk_dim if core.nope_layer[i] else core.nope
        kvb = w[p + "wkv_b"]
        hn = heads * nope
        owner = core.kv_source[i] == i
        ix = None
        if core.full_indexer[i]:
            ix = SimpleNamespace(
                q_norm_w=w[p + "ix_q_norm_w"],
                q_norm_b=w[p + "ix_q_norm_b"],
                wq_b=w[p + "ix_wq_b"],
                wq_b_s=w[p + "ix_wq_b_s"].t(),
                wk_wp=w[p + "ix_wk_wp"].t(),
                k_norm_w=w[p + "ix_k_norm_w"],
                k_norm_b=w[p + "ix_k_norm_b"],
                ssmax_s=w[p + "ix_ssmax_s"],
                ssmax_b=w[p + "ix_ssmax_b"],
                rope=not core.indexer_nope[i],
            )
        moe = None
        if i >= core.dense_layers:
            moe = SimpleNamespace(
                router=w[p + "router"].t(),
                fc1_w=w[p + "fc1_w"],
                fc1_s=w[p + "fc1_s"],
                fc2_w=w[p + "fc2_w"],
                fc2_s=w[p + "fc2_s"],
                **vars(_moe_scalars(w[p + "isc1"], w[p + "isc2"], w[p + "ws2_13"], w[p + "ws2_2"])),
            )
        layers.append(
            SimpleNamespace(
                index=i,
                rope=not core.nope_layer[i],
                nope=nope,
                owner=owner,
                kv_source=core.kv_source[i],
                attn_norm=w[p + "attn_norm"],
                wq=w[p + "wq"],
                wq_s=w[p + "wq_s"].t(),
                ssmax_s=w[p + "ssmax_s"],
                ssmax_b=w[p + "ssmax_b"],
                wkv_a=w[p + "wkv_a"].t() if owner else None,
                kv_norm=w[p + "kv_a_norm"] if owner else None,
                k_b=kvb[:hn].reshape(heads, nope, kv_lora),
                v_b_t=torch.transpose(kvb[hn:].reshape(heads, v_dim, kv_lora), 1, 2),
                wo=w[p + "wo"],
                wo_s=w[p + "wo_s"].t(),
                ffn_norm=w[p + "ffn_norm"],
                mlp_gu=w[p + "mlp_gu"],
                mlp_gu_s=w[p + "mlp_gu_s"].t(),
                mlp_dn=w[p + "mlp_dn"],
                mlp_dn_s=w[p + "mlp_dn_s"].t(),
                indexer=ix,
                moe=moe,
                next_norm=(
                    w[f"l{i + 1}_attn_norm"] if i + 1 < core.num_layers else w["final_norm"]
                ),
            )
        )
    return layers


class MistralLarge4Tep4ModelingV2(DecoderModel):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        cfg = model_config.pretrained_config
        mapping = model_config.mapping
        self.tp_rank = mapping.tp_rank
        self.tp_size = mapping.tp_size
        # The context selection's gather: the TP ranks in the order the engine sizes their
        # query shares, on the MPI session's communicator (`post_load_weights` swaps in the TP
        # process group when MPI is disabled).
        self.tp_group = list(mapping.tp_group)
        self.tp_pg = None
        # The attention block's glue: the fused entries (ssmax_scaling_fused_q,
        # ssmax_indexer_q_side_fused, the latent RMSNorm in place, FlashMLA's decode output
        # read in place) at every max_seq_len, whose cells cover the 32K and 1M configs'
        # positions (to 1048575), rope tables and step shapes; the separate ops they replace
        # under `_UNFUSED_GLUE`, at every site.
        self.unfused_glue = _UNFUSED_GLUE
        # The generation rows' sparse MLA: cute_dsl_ds_mla_sparse_decode at the rank's heads at
        # every max_seq_len, whose cells cover the 32K and 1M configs' pools and key lengths;
        # flash_mla_sparse_decode over the 64-head padded query under `_GEN_ATTN_FLASH_MLA`.
        self.gen_attn_flash_mla = _GEN_ATTN_FLASH_MLA

        self.num_layers = cfg.num_hidden_layers
        self.hidden = cfg.hidden_size
        self.eps = cfg.rms_norm_eps
        self.heads = cfg.num_attention_heads // mapping.tp_size
        self.nope = cfg.qk_nope_head_dim
        self.rope_dim = cfg.qk_rope_head_dim
        # A NoPE layer spends the rope dims on nope: the query head stays this wide.
        self.qk_dim = self.nope + self.rope_dim
        self.v_dim = cfg.v_head_dim
        self.kv_lora = cfg.kv_lora_rank
        self.lat_dim = self.kv_lora + self.rope_dim
        # Every layer's softmax scale, NoPE layers included: qk_head_dim**-0.5, no YaRN
        # temperature (its magnitude is 1).
        self.softmax_scale = self.qk_dim**-0.5

        self.nope_layer = [bool(f) for f in cfg.nope_layer_flags]
        self.indexer_nope = [bool(f) for f in cfg.indexer_nope_layer_flags]
        self.kv_source = [int(s) for s in cfg.kv_source_layers]
        # An owner some later layer attends over (layer 35): a step whose context KV fits the
        # held workspace twice keeps its gather there for the sharers; a longer one has each
        # sharer gather its source's rows again (`_ctx_workspace`).
        self.shared_source = [
            any(s == i and j != i for j, s in enumerate(self.kv_source))
            for i in range(self.num_layers)
        ]
        self.full_indexer = [role == "full" for role in cfg.index_sharing]
        # A 'shared' layer attends with the most recent full layer's selection: each target
        # keeps one selection buffer per step, which every full layer overwrites in order.
        self.indexer_layers = [i for i in range(self.num_layers) if self.full_indexer[i]]

        self.idx_heads = cfg.index_n_heads
        self.idx_dim = cfg.index_head_dim
        self.idx_topk = cfg.index_topk
        self.idx_eps = cfg.indexer_norm_eps
        # The indexer's softmax scale and head average: its scorers take none, so the q side
        # folds both into the head weights together with the q quantization scale.
        self.idx_weight_scale = self.idx_dim**-0.5 * self.idx_heads**-0.5

        self.dense_layers = cfg.first_k_dense_replace
        self.num_experts = cfg.n_routed_experts
        self.topk = cfg.num_experts_per_tok
        self.moe_inter = cfg.moe_intermediate_size
        self.local_experts = self.num_experts // mapping.moe_ep_size
        self.expert_offset = self.local_experts * mapping.moe_ep_rank
        # capacity: max_num_tokens -- the collective's peer buffers and the MoE autotuner's
        # largest bucket.
        self.max_num_tokens = model_config.max_num_tokens
        # The rope tables' first size; grown on the first step to the engine's max_seq_len.
        self.rope_rows = model_config.max_seq_len or cfg.max_position_embeddings

        self.w = _weights.MODEL_WEIGHTS.declare(_weights.MODEL_WEIGHTS.dims(self))

        self._rope: SimpleNamespace | None = None
        self._derived_for = None
        self._ix_k = None
        self._ix_k_checked = False
        self._pool_row: list[int] = []
        self._stride_factor = 0
        self._max_ctx_len = 0
        # The engine's sizes, fixed on the first step (`_derive_first_step`).
        self.prefill_logits_budget = 0
        self.ctx_workspace_rows = 0
        # The decode top-k's handle, built on the first step at a scorer width it serves.
        self._ix_topk = None
        self._prefill = self._decode = None

    def post_load_weights(self) -> None:
        """Derive what the weights fix, build the kernels and the collective, then the targets.

        Named for the shell hook that drives it. Everything compiles here or on the first
        (eager) step: a first compile inside a CUDA-graph capture breaks the graph.
        """
        dev = torch.cuda.current_device()
        self.build_kernels(dev)
        # Collective: every rank builds it with the same arguments. The peer buffers hold
        # the whole token budget, so one workspace serves every closing collective.
        self._lamport = build_lamport_workspace(
            self.hidden, self.eps, self.max_num_tokens, self.tp_rank, self.tp_size
        )
        # The gather's communicator flavor, as the engine's own collectives pick it.
        if mpi_disabled():
            self.tp_pg = self.model_config.mapping.tp_group_pg.boxed()
        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)

    def build_kernels(self, dev: int) -> None:
        """The rank-local derivations: the rope tables, the indexer q side for the unfused glue,
        and the generation rows' sparse decode at this rank's heads."""
        self._rope = _rope_tables(
            self.model_config.pretrained_config,
            self.w["lerope_alpha"],
            self.rope_rows,
            self.w["final_norm"].dtype,
            self.w["final_norm"].device,
        )
        if self.unfused_glue:
            # The checkpoint's rope is interleaved (GPT-J) for the indexer too.
            self._ix_q = build_ix_q_side(dev, is_neox=False)
        self._mla_decode = (
            None if self.gen_attn_flash_mla else build_ds_mla_decode(dev, num_heads=self.heads)
        )

    def _derive_first_step(self, md) -> None:
        """What depends on values the engine resolves, fixed on the first step it hands over
        (eager, ahead of any capture) and again for a replaced KV cache manager."""
        if md.kv_cache_manager is self._derived_for:
            return
        dev = torch.cuda.current_device()
        mgr = md.kv_cache_manager
        # The pool layout is the KV cache manager's, which routing cannot see; every latent
        # op here reads and writes fp8_ds_mla rows addressed by owner row.
        if getattr(mgr, "latent_layout", None) != _LATENT_LAYOUT:
            raise RuntimeError(
                f"this target attends over an {_LATENT_LAYOUT} latent pool with owner-only "
                f"pages (MistralLarge4CacheManager); the engine built {type(mgr).__name__} "
                f"with layout {getattr(mgr, 'latent_layout', None)!r}. Use KV cache manager V2 "
                "and kv_cache_config.dtype auto, fp8 or fp8_ds_mla."
            )
        if list(mgr.kv_source_layers) != self.kv_source:
            raise RuntimeError("the KV cache manager's kv_source_layers differ from the config's")
        # A table shorter than max_seq_len is read past its end with no check.
        if md.max_seq_len > self._rope.rows:
            self.rope_rows = md.max_seq_len
            self._rope = _rope_tables(
                self.model_config.pretrained_config,
                self.w["lerope_alpha"],
                self.rope_rows,
                self.w["final_norm"].dtype,
                self.w["final_norm"].device,
            )
        # Layer i reads owner row `_pool_row[i]` (its source's); a block holds every owner's
        # page, so a global row is `block * owners * tpb + row * tpb + offset`.
        self._pool_row = [mgr.latent_pool_row(i) for i in range(self.num_layers)]
        self._stride_factor = len(mgr.kv_owner_layers) * md.tokens_per_block
        # The decode scorer's column count: frozen into a capture, so an engine constant.
        self._max_ctx_len = md.max_seq_len
        # The indexer's prefill logits budget and the held context workspace's rows, from the
        # engine's max_num_tokens, max_num_requests, max_seq_len and runtime features.
        self.prefill_logits_budget = _prefill_logits_budget(md.max_num_tokens, md.max_seq_len)
        self.ctx_workspace_rows = _ctx_workspace_rows(md)
        logger.info(
            f"modeling_v2 Mistral Large 4 NVFP4: sized from the engine: prefill logits budget "
            f"{self.prefill_logits_budget}, context workspace {self.ctx_workspace_rows} rows"
        )
        # The decode selector, by the scorer's column count: cute_dsl_radix_topk_decode's
        # handle (built once) at a width it serves, else none and the trtllm op selects.
        lo, hi = _TOPK_DECODE_COLUMNS
        if not (_TOPK_DECODE_CUTE_DSL and lo <= self._max_ctx_len <= hi):
            self._ix_topk = None
        elif self._ix_topk is None:
            self._ix_topk = build_ix_topk_decode(dev, next_n=1)
        # Both key-pool layouts: a launcher compiled for a contiguous view addresses a strided
        # one as if it were dense.
        if self._ix_k is None:
            self._ix_k = {
                contiguous: build_ix_k_cache(dev, is_neox=False, cache_contiguous=contiguous)
                for contiguous in (False, True)
            }
        self._derived_for = md.kv_cache_manager

    def _select_target(self, attn_metadata) -> "PrefillTarget | DecodeTarget":
        """Which target runs this step: one host-side integer read. A mixed batch goes to
        prefill, so prefill is the general case and decode the specialization that may assume
        it has no context rows."""
        return self._decode if attn_metadata.num_contexts == 0 else self._prefill

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Derive what the first step fixes, then route."""
        self._derive_first_step(attn_metadata)
        return self._select_target(attn_metadata).forward(
            attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            lora_params=lora_params,
            **kwargs,
        )


class PrefillTarget:
    """The general case: context rows, and possibly generation rows behind them."""

    # The held context workspace, bf16 [core.ctx_workspace_rows, 576]: allocated on the first
    # context step and held from then on (`_ctx_workspace`).
    _ws_buf: torch.Tensor | None = None
    # The indexer's prefill top-k scratch (`gvr_workspace`, 20 MiB): allocated on the first
    # context step that selects and held, used on the main stream only (`_select_ctx`).
    _gvr_ws: torch.Tensor | None = None

    def __init__(self, core: MistralLarge4Tep4ModelingV2) -> None:
        """Take the per-layer views once, off the real weights. Everything an engine value
        fixes is read off `self.core` at the call."""
        self.core = core
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks for it. Each target owns its own flag:
        # the two `step_args` projections differ, so probing one does not prove the other.
        self._contract_pending = step_contract_enabled()
        self._layers = _layer_views(core)
        self._embed = core.w["embed"]
        self._norm0 = core.w["l0_attn_norm"]

    def _check_step_contract(self, md) -> None:
        """Opt-in first-forward fail-fast: calling `step_args` reads every metadata field this
        target consumes (private trtllm surface). Fixed at engine construction, so once per
        target instance, and off in a served engine."""
        if not self._contract_pending:
            return
        try:
            self.step_args(md)
        except AttributeError as exc:
            raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
        self._contract_pending = False

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        """The common batch state, plus the context rows': their counts and lengths, the
        indexer's prefill windows and scoring blocks, its query split across the TP ranks and
        its key gather mappings, and the context rope/append op's offsets and their host prefix
        sums."""
        step = _step_common(md, self.core.indexer_layers)
        ctx = md.num_contexts
        step.update(
            num_contexts=ctx,
            num_ctx_tokens=md.num_ctx_tokens,
            seq_lens=md.seq_lens,
            skip_ctx=bool(md.skip_indexer_for_ctx_reqs),
            cu_seqlen_ks=md.cu_seqlen_ks,
            cu_seqlen_ke=md.cu_seqlen_ke,
            # (q_begin, q_end, per-rank sizes) once the step's context rows reach the engine's
            # q_split_threshold under chunked prefill: the chunks then cover this rank's
            # [q_begin, q_end) only.
            indexer_prefill_q_split=md.indexer_prefill_q_split,
            num_ctx_kv_tokens=md.num_ctx_kv_tokens,
            slot_mapping_fp8_fullkv=md.slot_mapping_fp8_fullkv,
            slot_mapping_scale_fullkv=md.slot_mapping_scale_fullkv,
            **_ctx_kv_offsets(md, ctx),
            **_ctx_host_offsets(md, ctx),
        )
        step["indexer_blocks"] = _indexer_blocks(md.indexer_prefill_chunks, step)
        return step

    # ---- one layer's pieces, context and generation rows ----

    def _linear_fp8(self, x, w, w_s):
        """`x @ w.T`: the bf16 activation quantized 1x128 with power-of-two scales, then the
        DeepGEMM block-scale GEMM against the load-time UE8M0 weight scales."""
        codes, scale = fp8_quantize_1x128_packed_ue8m0(x)
        return fp8_prequantized_swap_ab_gemm(codes, scale, w, w_s, torch.bfloat16, True)

    def _latent(self, x, L, nope_lat):
        """`[RMSNorm(c_kv) | k_pe]`, 576 wide: a RoPE owner's kv_a row, normalized in place; a
        NoPE owner's 512-wide latent normalized into `nope_lat`, whose rope key stays zero."""
        core = self.core
        kv = cublas_mm(x, L.wkv_a)
        if core.unfused_glue:
            c_kv = flashinfer_rmsnorm(kv[:, : core.kv_lora], L.kv_norm, core.eps)
            if L.rope:
                k_pe = kv[:, core.kv_lora :]
            else:
                k_pe = torch.zeros([kv.shape[0], core.rope_dim], dtype=kv.dtype, device=kv.device)
            return torch.cat([c_kv, k_pe], dim=-1)
        if L.rope:
            c_kv = kv[:, : core.kv_lora]
            flashinfer_rmsnorm(c_kv, L.kv_norm, core.eps, out=c_kv)
            return kv
        flashinfer_rmsnorm(kv, L.kv_norm, core.eps, out=nope_lat[:, : core.kv_lora])
        return nope_lat

    def _indexer(self, step, x, pos, L, topk_buf, tc: int, skip_ctx: bool, skip_gen: bool):
        """Append this step's keys to the layer's key pool; select `index_topk` positions per
        row into `topk_buf` for the rows whose phase the engine did not skip. Under the
        engine's query split each rank selects its share of the context rows, and the shares
        are gathered into rows [0, tc) before any layer reads them."""
        core = self.core
        ix = L.indexer
        rows = x.shape[0]
        table = core._rope.indexer if ix.rope else core._rope.identity
        kw = cublas_mm(x, ix.wk_wp)
        k = kw[:, : core.idx_dim].contiguous()
        k_pool = step["ix_pools"][L.index]
        k_pool = cute_dsl_indexer_k_cache_update(
            k, ix.k_norm_w, ix.k_norm_b, pos, table.view(-1, 2, core.rope_dim // 2), k_pool,
            step["slot_mapping_fp8"], step["slot_mapping_scale"], rows,
            is_neox=False, compiled=core._ix_k[k_pool.is_contiguous()],
            # Its range check is a host sync: once, on the first (eager) call.
            validate=not core._ix_k_checked,
        )  # fmt: skip
        core._ix_k_checked = True
        if skip_ctx and skip_gen:
            return
        qn = layer_norm(x, ix.q_norm_w, ix.q_norm_b, core.idx_eps)
        q = torch.reshape(self._linear_fp8(qn, ix.wq_b, ix.wq_b_s), [rows, core.idx_heads, -1])
        q_fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
        weights = torch.empty([rows, core.idx_heads], dtype=torch.float32, device=x.device)
        if core.unfused_glue:
            q = ssmax_scaling(q, pos, ix.ssmax_s, ix.ssmax_b)
            q_fp8, weights = cute_dsl_indexer_q_side_fused(
                q, pos, table, kw[:, core.idx_dim :].float().contiguous(), core.idx_weight_scale,
                q_fp8, weights, is_neox=False, compiled=core._ix_q,
            )  # fmt: skip
        else:
            # SSMax, rope, e4m3 and the head weights (read in place out of `kw`) in one launch.
            q_fp8, weights = ssmax_indexer_q_side_fused(
                q, pos, ix.ssmax_s, ix.ssmax_b, table, kw[:, core.idx_dim :],
                core.idx_weight_scale, q_fp8, weights,
            )  # fmt: skip
        if tc and not skip_ctx:
            self._select_ctx(step, k_pool, q_fp8, weights, topk_buf)
            q_split = step["indexer_prefill_q_split"]
            if q_split is not None:
                # The chunks covered this rank's share only. Every rank of the group makes
                # this call once per selecting layer, in layer order: the communicator pairs
                # calls by position.
                q_begin, q_end, q_sizes = q_split
                topk_buf[:tc].copy_(
                    allgather(topk_buf[q_begin:q_end], q_sizes, core.tp_group, core.tp_pg)
                )
        gen = rows - tc
        if gen and not skip_gen:
            ctx = step["num_contexts"]
            num_gens = step["num_seqs"] - ctx
            assert gen % num_gens == 0, (gen, num_gens)
            next_n = gen // num_gens
            kv_lens = step["kv_lens_cuda"][ctx : ctx + num_gens]
            logits = cute_dsl_fp8_paged_mqa_logits(
                torch.reshape(q_fp8[tc:], [num_gens, next_n, core.idx_heads, core.idx_dim]),
                k_pool,
                weights[tc:],
                kv_lens,
                step["indexer_k_cache_block_offsets"][ctx : ctx + num_gens],
                step["scheduler_metadata_buffer"],
                core._max_ctx_len,
            )
            if next_n != 1 or core._ix_topk is None:
                indexer_topk_decode(
                    logits, kv_lens, topk_buf[tc:], next_n, core.idx_topk, 1,
                    step["radix_aux_indices"], step["radix_aux_logits"],
                )  # fmt: skip
            else:
                cute_dsl_radix_topk_decode(
                    logits, kv_lens, topk_buf[tc:], next_n=1, compiled=core._ix_topk
                )

    def _ix_plan(self, step, heads: int) -> list[SimpleNamespace]:
        """The step's indexer scoring plan, the same for each of its indexer layers: per scoring
        block (`_indexer_blocks`, this rank's share of the context rows under the engine's
        query split), its rows, its keys in the step's full-KV mappings and its query tiles of
        at most the engine's `prefill_logits_budget` logits (`_prefill_tiles`), each tile with
        its rows' windows and the split-KV schedule of its logits call (`mqa_logits_schedule`).
        A block with no key has no tile.

        Built on the step's first indexer layer and kept in `step`: the windows, the key counts
        and so the schedules depend on the step alone, and the layers differ only in the keys
        and queries they score."""
        core = self.core
        plan = []
        for block in step["indexer_blocks"]:
            n_k = block.k_token_end - block.k_token_start
            if block.token_end <= block.token_start:
                continue
            tiles = []
            if n_k > 0:
                for lo, hi in _prefill_tiles(
                    block.token_start, block.token_end, n_k, core.prefill_logits_budget
                ):
                    # The windows are the block's own, indexed from its first row.
                    ks = block.cu_seqlen_ks[lo - block.token_start : hi - block.token_start]
                    ke = block.cu_seqlen_ke[lo - block.token_start : hi - block.token_start]
                    tiles.append((lo, hi, ks, ke, mqa_logits_schedule(ks, ke, n_k, heads)))
            plan.append(
                SimpleNamespace(
                    token_start=block.token_start,
                    token_end=block.token_end,
                    k_token_start=block.k_token_start,
                    n_k=n_k,
                    tiles=tiles,
                )
            )
        return plan

    def _select_ctx(self, step, k_pool, q_fp8, weights, topk_buf) -> None:
        """Select `index_topk` positions, window-local, for each context row this rank scores,
        along the step's scoring plan (`_ix_plan`): per block, gather its keys out of the
        layer's key pool through the step's full-KV mappings, then per query tile score its
        rows against them with the tile's split-KV schedule, which spreads each query block's
        keys over the SMs (a 256-row tile at 1048576 keys keeps every SM busy, the grid-stride
        call 64), and select over each row's window into `topk_buf` with the self-sampling GVR
        top-k, one streaming pass per row. Both answer as the grid-stride scorer and the radix
        top-k: the logits bitwise, the selection the same set up to the keys tied at a row's
        2048th score. A block with no key leaves its rows' slots empty (-1), and each tile's
        logits are freed before the next tile's are allocated."""
        core = self.core
        plan = step.get("ix_plan")
        if plan is None:
            plan = step["ix_plan"] = self._ix_plan(step, q_fp8.shape[1])
        if self._gvr_ws is None:
            # On the first context step, which the KV-capacity estimate's profiling forward
            # is: the pool is sized with the top-k's scratch held, and every top-k launch this
            # engine's tiles reach compiles here rather than on a served step (each tile holds
            # at most the logits budget, one block at most max_seq_len keys).
            self._gvr_ws = gvr_workspace(q_fp8.device)
            t0 = time.monotonic()
            calls = gvr_warmup(
                core.idx_topk, core._max_ctx_len, core.prefill_logits_budget, q_fp8.device,
                self._gvr_ws,
            )  # fmt: skip
            logger.info(
                f"modeling_v2 Mistral Large 4 NVFP4: indexer prefill top-k compiled for this "
                f"engine's tiles ({calls} warm calls, {time.monotonic() - t0:.1f} s)"
            )
        for block in plan:
            if block.n_k <= 0:
                # No key to select from: every slot of the block's rows is empty.
                topk_buf[block.token_start : block.token_end].fill_(-1)
                continue
            ck, cs = indexer_k_cache_gather_op(
                k_pool, step["slot_mapping_fp8_fullkv"], step["slot_mapping_scale_fullkv"],
                block.k_token_start, block.n_k, core.idx_dim,
            )  # fmt: skip
            cs = torch.reshape(cs, [-1])
            for lo, hi, ks, ke, schedule in block.tiles:
                logits = fp8_mqa_logits(
                    q_fp8[lo:hi], (ck, cs), weights[lo:hi], ks, ke, schedule_meta=schedule
                )
                cute_dsl_gvr_topk_prefill(
                    logits, ks, ke, topk_buf[lo:hi], core.idx_topk, self._gvr_ws
                )
                # Freed before the next tile's is allocated: one tile is live at a time.
                del logits

    def _attn_ctx(self, step, L, fused_q, lat_out, req_idx, topk_buf, ctx):
        """Sparse MLA over the context rows into `lat_out`, on the step's path (`ctx`).

        Workspace path: FlashMLA's bf16 sparse prefill over the context requests' latent rows
        of the source layer, dequantized out of the pool (this step's rows included, so every
        row is read FP8-rounded) into the held workspace's first rows. A shared source (layer
        35) is gathered once into the rows the step keeps beside them, when it has them, and
        read there by every layer that shares it; otherwise each of those layers gathers it
        again into the first rows. Decode path, a step past the held workspace: the selections
        as pool slots, then FlashMLA's sparse fp8 decode over the pool with every context row a
        one-row sequence, in the step's sub-batches."""
        core = self.core
        tc = fused_q.shape[0]
        src = L.kv_source
        if ctx.ws_buf is None:
            slots = convert_req_index_to_global(
                req_idx[:tc], step["block_table"], topk_buf[:tc], step["tokens_per_block"],
                core.idx_topk, core._stride_factor, core._pool_row[src],
            )  # fmt: skip
            for lo, hi in ctx.sub_batches:
                # Past 11622 rows FlashMLA's scheduler exits the process instead of raising.
                assert hi - lo <= _CTX_DECODE_ROWS, (lo, hi)
                flash_mla_sparse_decode(
                    fused_q[lo:hi], step["latent_kv"], slots[lo:hi], ctx.sched[hi - lo],
                    core.softmax_scale, lat_out[lo:hi],
                )  # fmt: skip
            return
        ws = ctx.ws.get(src)
        if ws is None:
            keep = core.shared_source[src] and ctx.kept is not None
            ws = ds_mla_dequant_gather(
                step["latent_pool"], core._pool_row[src], ctx.kv_req, ctx.kv_pos,
                step["block_table"], ctx.kept if keep else ctx.ws_buf,
            )  # fmt: skip
            if keep:
                ctx.ws[src] = ws
        # Request-local selections to workspace rows: each request's run starts at its
        # cumulative KV offset.
        sel = topk_buf[:tc]
        idx = torch.where(sel >= 0, sel + ctx.ws_off[req_idx[:tc]][:, None], sel)
        flash_mla_sparse_prefill(fused_q, ws, idx.to(torch.int32), core.softmax_scale, lat_out)

    def _attn_gen(self, step, L, fused_q, lat_out, slots, sched):
        """Sparse MLA over the generation rows behind the context rows into `lat_out`, over the
        pool: this rank's heads read off the 64-head scratch the prefill kernel shares (or, under
        `gen_attn_flash_mla`, FlashMLA's decode over all 64, one scheduler for the step)."""
        core = self.core
        if core.gen_attn_flash_mla:
            flash_mla_sparse_decode(
                fused_q, step["latent_kv"], slots, sched, core.softmax_scale, lat_out
            )
        else:
            cute_dsl_ds_mla_sparse_decode(
                fused_q, step["latent_kv"], slots, core.softmax_scale, lat_out,
                compiled=core._mla_decode,
            )  # fmt: skip

    def _mlp(self, o, L):
        """This rank's share of the layer's MLP, a partial sum: the dense SwiGLU slice on the
        dense layer; on a MoE layer the expert window (32 experts at tep4, 16 at tep8) plus the
        shared expert's slice."""
        core = self.core
        act = flashinfer_silu_and_mul(self._linear_fp8(o, L.mlp_gu, L.mlp_gu_s))
        shared = self._linear_fp8(act, L.mlp_dn, L.mlp_dn_s)
        m = L.moe
        if m is None:
            return shared
        logits = cublas_mm(o, m.router, None, torch.float32)
        ids, weights = renorm_moe_routing_op(logits, core.topk, torch.bfloat16)
        # One NVFP4 quantization under the layer's static scale serves every expert on every
        # rank: the windows' partials sum.
        xq, xs = fp4_quantize(o, m.gscale, _NVFP4_SF_VEC, False, _NVFP4_SF_LINEAR)
        routed = fp4_block_scale_moe_runner(
            None, None, xq, xs.view(torch.float8_e4m3fn), m.fc1_w, m.fc1_s, None, None, None,
            None, m.fc2_w, m.fc2_s, None, m.out1, m.out1_gate, m.out2, core.num_experts,
            core.topk, None, None, core.moe_inter, core.expert_offset, core.local_experts, None,
            _MOE_ROUTING_METHOD, True, _MOE_ACT_SWIGLU, topk_weights=weights, topk_ids=ids,
            tune_max_num_tokens=core.max_num_tokens,
        )[0]  # fmt: skip
        return torch.add(routed, shared)

    def _topk_buf(self, step, rows: int, tc: int, skip_ctx: bool, skip_gen: bool):
        """Where this step's selections land, request-local positions. A phase the engine
        skipped (every sequence within index_topk) takes the dense causal index it wrote."""
        if skip_ctx and skip_gen:
            return step["topk_indices_buffer"][:rows]
        buf = torch.empty(
            [rows, self.core.idx_topk], dtype=torch.int32, device=step["block_table"].device
        )
        if skip_ctx and tc:
            buf[:tc].copy_(step["topk_indices_buffer"][:tc])
        if skip_gen and rows > tc:
            buf[tc:].copy_(step["topk_indices_buffer"][tc:rows])
        return buf

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        core = self.core
        self._check_step_contract(attn_metadata)
        step = self.step_args(attn_metadata)
        h = (
            nn.functional.embedding(input_ids, self._embed)
            if inputs_embeds is None
            else inputs_embeds
        )
        rows = h.shape[0]
        dev = h.device
        # The slot mappings and windows follow the prepared token order; a padded input
        # would scatter keys into stale slots.
        assert rows == step["num_tokens"], (rows, step["num_tokens"])
        tc = step["num_ctx_tokens"]
        gen = rows - tc
        skip_ctx = step["skip_ctx"]
        skip_gen = step["skip_gen"] if gen else True
        pos = torch.reshape(position_ids, [-1])
        req_idx = step["req_idx_per_token"][:rows]
        topk_buf = self._topk_buf(step, rows, tc, skip_ctx, skip_gen)
        scratch = self._scratch(rows, h.dtype, dev)
        ctx = self._ctx_workspace(step, tc, dev) if tc else None

        x = flashinfer_rmsnorm(h, self._norm0, core.eps)
        residual = h
        spec_metadata = kwargs.get("spec_metadata")
        for layer_idx, L in enumerate(self._layers):
            attn_out = self._attention(
                step, L, x, pos, req_idx, topk_buf, scratch, ctx, tc, skip_ctx, skip_gen
            )
            partial = self._linear_fp8(attn_out, L.wo, L.wo_s)
            o = cute_dsl_allreduce_add_rmsnorm(
                partial, residual, L.ffn_norm, torch.empty_like(partial), core._lamport
            )
            mlp = self._mlp(o, L)
            x = cute_dsl_allreduce_add_rmsnorm(
                mlp, residual, L.next_norm, torch.empty_like(mlp), core._lamport
            )
            if spec_metadata is not None:
                spec_metadata.maybe_capture_hidden_states(layer_idx, residual)
        return x

    def _scratch(self, rows: int, dtype, dev) -> SimpleNamespace:
        """Per-step scratch every layer rewrites in turn: the attention output, the absorbed
        query at the prefill kernel's 64 heads (this rank's first, the rest zero for the step),
        the latent attention output, a NoPE owner's 576-wide latent rows (the rope key zero for
        the step), and, under `gen_attn_flash_mla`, the generation rows' FlashMLA scheduler,
        filled by the step's first decode call and shared by the rest."""
        core = self.core
        bf16 = dict(dtype=dtype, device=dev)
        return SimpleNamespace(
            attn_out=torch.empty([rows, core.heads * core.v_dim], **bf16),
            fused_q=torch.zeros([rows, _FLASH_MLA_HEADS, core.lat_dim], **bf16),
            lat_out=torch.empty([rows, core.heads, core.kv_lora], **bf16),
            nope_lat=None if core.unfused_glue else torch.zeros([rows, core.lat_dim], **bf16),
            sched=get_mla_metadata()[0] if core.gen_attn_flash_mla else None,
        )

    def _ctx_workspace(self, step, tc: int, dev) -> SimpleNamespace:
        """The context rows' path for the step, chosen once on host ints by its context KV:
        every context request's cached prefix and new tokens (the host
        `ctx_kv_indptr[num_contexts]`; the engine's `num_ctx_kv_tokens` counts the new ones
        only, and under chunked prefill every chunk after a prompt's first has a prefix).

        The held workspace, bf16 `[core.ctx_workspace_rows, 576]` (`_ctx_workspace_rows`), is
        allocated on the first context step, whichever path that step takes, and held. The
        engine's KV-capacity estimate runs context steps as its profiling forward and sizes the
        pool after their peak memory (`KvCacheCreator.configure_kv_cache_capacity`), so the pool
        is sized with the workspace held. A generation-only engine runs no context step and
        holds none.

        The workspace path, a step whose context KV fits the held rows: request and position
        per workspace row as the dequantizing gather and the prefill kernel address them, each
        request's offset, the held workspace's first `n_kv` rows, which every layer's gather
        rewrites, and, when the context KV fits twice, the next `n_kv` rows (`kept`), where a
        shared owner's gather stays for the step (`ws`).

        The decode path, a step past the held rows, with no workspace: the context rows'
        sub-batches of at most `_CTX_DECODE_ROWS`, and one FlashMLA scheduler per sub-batch
        size, filled by its first call and shared by the step's layers."""
        # Under KV reuse a context request attends over its cached prefix and its new tokens.
        n_kv = step["ctx_cached_tokens"] + step["num_ctx_tokens"]
        if self._ws_buf is None:
            # On the first context step: the KV-capacity estimate's profiling forward is one,
            # so the peak it measures, which sizes the pool, holds this buffer.
            self._ws_buf = torch.empty(
                [self.core.ctx_workspace_rows, self.core.lat_dim], dtype=torch.bfloat16, device=dev
            )
            logger.info(
                f"modeling_v2 Mistral Large 4 NVFP4: context workspace held, {self._ws_buf.shape[0]} "
                f"rows x {self.core.lat_dim} bf16 ({self._ws_buf.nbytes / 2**20:.0f} MiB)"
            )
        held = self._ws_buf.shape[0]
        if n_kv > held:
            bounds = [(lo, min(lo + _CTX_DECODE_ROWS, tc)) for lo in range(0, tc, _CTX_DECODE_ROWS)]
            logger.debug(
                f"modeling_v2 Mistral Large 4 NVFP4: context KV {n_kv} past the held {held} rows: "
                f"decode path, {len(bounds)} sub-batches"
            )
            return SimpleNamespace(
                ws_buf=None,
                sub_batches=bounds,
                sched={rows: get_mla_metadata()[0] for rows in {hi - lo for lo, hi in bounds}},
            )
        ctx = step["num_contexts"]
        cu = step["cu_ctx_kv"]
        lens = cu[1:] - cu[:-1]
        kv_req = torch.repeat_interleave(
            torch.arange(ctx, dtype=torch.int32, device=dev), lens, output_size=n_kv
        )
        kv_pos = (torch.arange(n_kv, device=dev) - cu[:-1][kv_req]).to(torch.int32)
        keep = 2 * n_kv <= held
        logger.debug(
            f"modeling_v2 Mistral Large 4 NVFP4: context KV {n_kv} of the held {held} rows: "
            f"workspace path, the shared source {'kept' if keep else 'gathered again'}"
        )
        return SimpleNamespace(
            kv_req=kv_req,
            kv_pos=kv_pos,
            ws_off=cu[:-1],
            ws_buf=self._ws_buf[:n_kv],
            kept=self._ws_buf[n_kv : 2 * n_kv] if keep else None,
            ws={},
        )

    def _attention(self, step, L, x, pos, req_idx, topk_buf, s, ctx, tc, skip_ctx, skip_gen):
        """One layer's attention block, both phases, into `s.attn_out` (returned): the query
        with SSMax, the indexer's selection, the absorbed query, the owner's rope and
        fp8_ds_mla append, sparse MLA over the context and the generation rows, and the V
        up-projection."""
        core = self.core
        rows = x.shape[0]
        q = self._linear_fp8(x, L.wq, L.wq_s)
        q3 = torch.reshape(q, [rows, core.heads, core.qk_dim])
        fq = s.fused_q[:, : core.heads]
        if core.unfused_glue:
            q3 = ssmax_scaling(q3, pos, L.ssmax_s, L.ssmax_b)
            if L.rope:
                fq[..., core.kv_lora :].copy_(q3[..., L.nope :])
            else:
                fq[..., core.kv_lora :].zero_()
        else:
            # SSMax in place; its rope half (a NoPE layer's zero) into the absorbed query's tail.
            ssmax_scaling_fused_q(
                q3, pos, L.ssmax_s, L.ssmax_b, q3, fq[..., core.kv_lora :], L.rope
            )
        if L.indexer is not None:
            self._indexer(step, x, pos, L, topk_buf, tc, skip_ctx, skip_gen)
        bmm_out(
            torch.transpose(q3[..., : L.nope], 0, 1), L.k_b,
            torch.transpose(fq[..., : core.kv_lora], 0, 1),
        )  # fmt: skip
        if L.owner:
            # Every row's latent, context and generation, before either phase reads the pool;
            # the query tail is roped in the same call.
            ds_mla_rope_append(
                self._latent(x, L, s.nope_lat), fq, pos, core._rope.indexer, req_idx,
                step["block_table"], step["latent_pool"], core._pool_row[L.index], L.rope,
            )  # fmt: skip
        if tc:
            self._attn_ctx(step, L, s.fused_q[:tc], s.lat_out[:tc], req_idx, topk_buf, ctx)
        if rows > tc:
            slots = convert_req_index_to_global(
                req_idx[tc:], step["block_table"], topk_buf[tc:], step["tokens_per_block"],
                core.idx_topk, core._stride_factor, core._pool_row[L.kv_source],
            )  # fmt: skip
            self._attn_gen(step, L, s.fused_q[tc:], s.lat_out[tc:], slots, s.sched)
        bmm_out(
            torch.transpose(s.lat_out, 0, 1),
            L.v_b_t,
            torch.transpose(torch.reshape(s.attn_out, [rows, core.heads, core.v_dim]), 0, 1),
        )
        return s.attn_out

    def dspark_forward(
        self,
        draft,
        noise_embedding,
        query_positions,
        num_ctx_per_req,
        ctx_k_cache,
        ctx_v_cache,
        ctx_cache_batch_idx,
        ctx_kv_cache=None,
        ctx_page_table=None,
    ):
        draft._build_fused_kv_buffers()
        draft._ensure_rope()
        core = self.core
        batch, block, hidden = noise_embedding.shape
        rows = batch * block
        # Gather a bounded window, not every page in the configured context.
        # Invalid padding slots are masked even if their pages are allocated.
        lengths = num_ctx_per_req[:batch].long()
        offsets = torch.arange(draft.window, device=noise_embedding.device)
        positions = (lengths - draft.window).clamp_min(0)[:, None] + offsets[None, :]
        valid = positions < lengths[:, None]
        slots = ctx_cache_batch_idx.long()
        if ctx_page_table is not None:
            page_size = ctx_kv_cache[0].shape[-2]
            table = ctx_page_table.index_select(0, slots).long()
            safe_positions = positions.clamp(max=table.shape[1] * page_size - 1)
            pages = table.gather(1, safe_positions // page_size).clamp_min(0)
            page_offsets = safe_positions % page_size
        else:
            safe_positions = positions.clamp(max=ctx_k_cache.shape[2] - 1)
        key_valid = torch.cat(
            (valid, torch.ones(batch, block, dtype=torch.bool, device=valid.device)), dim=1
        )
        query = torch.empty(
            batch,
            block,
            core.heads,
            core.lat_dim,
            dtype=noise_embedding.dtype,
            device=noise_embedding.device,
        )
        values = torch.empty(
            rows, core.heads, core.v_dim, dtype=noise_embedding.dtype, device=noise_embedding.device
        )
        residual = noise_embedding.reshape(rows, hidden).clone()
        normalized = torch.empty_like(residual)
        x = flashinfer_rmsnorm(residual, draft._layers[0].attn_norm, core.eps)
        for i, layer in enumerate(draft._layers):
            q = cublas_mm(x, layer.q, None, None).view(batch, block, core.heads, core.qk_dim)
            bmm_out(
                q[..., : core.nope].reshape(rows, core.heads, core.nope).transpose(0, 1),
                layer.k_b,
                query[..., : core.kv_lora].view(rows, core.heads, core.kv_lora).transpose(0, 1),
            )
            query[..., core.kv_lora :].copy_(draft._rotate(q[..., core.nope :], query_positions))
            kv = cublas_mm(x, layer.kv_a, None, None)
            content = flashinfer_rmsnorm(
                kv[:, : core.kv_lora].contiguous(), layer.kv_norm, core.eps
            )
            block_kv = torch.cat(
                (
                    content.view(batch, block, core.kv_lora),
                    draft._rotate(
                        kv[:, core.kv_lora :].view(batch, block, core.rope_dim), query_positions
                    ),
                ),
                dim=-1,
            )
            if ctx_page_table is not None:
                context = ctx_kv_cache[i][:, 0, 0][pages, page_offsets]
            else:
                context = ctx_k_cache[slots[:, None], i, safe_positions, 0]
            # Zero masked values as well as scores: 0 * an uninitialized
            # NaN in the latent pool must not contaminate the weighted sum.
            context = context.masked_fill(~valid.unsqueeze(-1), 0)
            keys = torch.cat((context, block_kv), dim=1)
            scores = torch.einsum("bshd,btd->bsht", query, keys).float() * core.softmax_scale
            scores = scores.masked_fill(~key_valid[:, None, None, :], float("-inf"))
            probabilities = torch.softmax(scores, dim=-1).to(keys.dtype)
            latent = torch.einsum("bsht,btc->bshc", probabilities, keys[..., : core.kv_lora])
            bmm_out(
                latent.reshape(rows, core.heads, core.kv_lora).transpose(0, 1),
                layer.v_b.transpose(1, 2),
                values.transpose(0, 1),
            )
            partial = cublas_mm(values.reshape(rows, -1), layer.out, None, None)
            o = cute_dsl_allreduce_add_rmsnorm(
                partial, residual, layer.ffn_norm, normalized, core._lamport
            )
            gate = cublas_mm(o, layer.gate, None, None)
            up = cublas_mm(o, layer.up, None, None)
            activated = flashinfer_silu_and_mul(torch.cat((gate, up), dim=-1))
            mlp = cublas_mm(activated, layer.down, None, None)
            x = cute_dsl_allreduce_add_rmsnorm(
                mlp, residual, layer.next_norm, normalized, core._lamport
            )
        return x


class DecodeTarget:
    """Generation-only steps, including uniform DSpark verification blocks; graph captured."""

    def __init__(self, core: MistralLarge4Tep4ModelingV2) -> None:
        """Take the per-layer views once, off the real weights. Everything an engine value
        fixes is read off `self.core` at the call."""
        self.core = core
        self._contract_pending = step_contract_enabled()
        self._layers = _layer_views(core)
        self._embed = core.w["embed"]
        self._norm0 = core.w["l0_attn_norm"]

    def _check_step_contract(self, md) -> None:
        """Opt-in first-forward fail-fast: calling `step_args` reads every metadata field this
        target consumes (private trtllm surface). Fixed at engine construction, so once per
        target instance, and off in a served engine."""
        if not self._contract_pending:
            return
        try:
            self.step_args(md)
        except AttributeError as exc:
            raise AssertionError(f"attention metadata surface drifted: {exc}") from exc
        self._contract_pending = False

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        """The common batch state. States the two context counts instead of reading them
        back: this target runs only when there are no context rows."""
        step = _step_common(md, self.core.indexer_layers)
        step.update(num_contexts=0, num_ctx_tokens=0)
        return step

    def _linear_fp8(self, x, w, w_s):
        """`x @ w.T`: the bf16 activation quantized 1x128 with power-of-two scales, then the
        DeepGEMM block-scale GEMM against the load-time UE8M0 weight scales."""
        codes, scale = fp8_quantize_1x128_packed_ue8m0(x)
        return fp8_prequantized_swap_ab_gemm(codes, scale, w, w_s, torch.bfloat16, True)

    def _latent(self, x, L, nope_lat):
        """`[RMSNorm(c_kv) | k_pe]`, 576 wide: a RoPE owner's kv_a row, normalized in place; a
        NoPE owner's 512-wide latent normalized into `nope_lat`, whose rope key stays zero."""
        core = self.core
        kv = cublas_mm(x, L.wkv_a)
        if core.unfused_glue:
            c_kv = flashinfer_rmsnorm(kv[:, : core.kv_lora], L.kv_norm, core.eps)
            if L.rope:
                k_pe = kv[:, core.kv_lora :]
            else:
                k_pe = torch.zeros([kv.shape[0], core.rope_dim], dtype=kv.dtype, device=kv.device)
            return torch.cat([c_kv, k_pe], dim=-1)
        if L.rope:
            c_kv = kv[:, : core.kv_lora]
            flashinfer_rmsnorm(c_kv, L.kv_norm, core.eps, out=c_kv)
            return kv
        flashinfer_rmsnorm(kv, L.kv_norm, core.eps, out=nope_lat[:, : core.kv_lora])
        return nope_lat

    def _indexer(self, step, x, pos, L, topk_buf, skip_gen: bool):
        """Append this step's keys to the layer's key pool; unless the engine skipped the
        indexer (every sequence within index_topk), score every sequence's paged keys and
        select `index_topk` positions per row into `topk_buf`."""
        core = self.core
        ix = L.indexer
        rows = x.shape[0]
        table = core._rope.indexer if ix.rope else core._rope.identity
        kw = cublas_mm(x, ix.wk_wp)
        k = kw[:, : core.idx_dim].contiguous()
        k_pool = step["ix_pools"][L.index]
        k_pool = cute_dsl_indexer_k_cache_update(
            k, ix.k_norm_w, ix.k_norm_b, pos, table.view(-1, 2, core.rope_dim // 2), k_pool,
            step["slot_mapping_fp8"], step["slot_mapping_scale"], rows,
            is_neox=False, compiled=core._ix_k[k_pool.is_contiguous()], validate=False,
        )  # fmt: skip
        if skip_gen:
            return
        qn = layer_norm(x, ix.q_norm_w, ix.q_norm_b, core.idx_eps)
        q = torch.reshape(self._linear_fp8(qn, ix.wq_b, ix.wq_b_s), [rows, core.idx_heads, -1])
        q_fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
        weights = torch.empty([rows, core.idx_heads], dtype=torch.float32, device=x.device)
        if core.unfused_glue:
            q = ssmax_scaling(q, pos, ix.ssmax_s, ix.ssmax_b)
            q_fp8, weights = cute_dsl_indexer_q_side_fused(
                q, pos, table, kw[:, core.idx_dim :].float().contiguous(), core.idx_weight_scale,
                q_fp8, weights, is_neox=False, compiled=core._ix_q,
            )  # fmt: skip
        else:
            # SSMax, rope, e4m3 and the head weights (read in place out of `kw`) in one launch.
            q_fp8, weights = ssmax_indexer_q_side_fused(
                q, pos, ix.ssmax_s, ix.ssmax_b, table, kw[:, core.idx_dim :],
                core.idx_weight_scale, q_fp8, weights,
            )  # fmt: skip
        num_gens = step["num_seqs"]
        next_n = rows // num_gens
        kv_lens = step["kv_lens_cuda"][:num_gens]
        logits = cute_dsl_fp8_paged_mqa_logits(
            torch.reshape(q_fp8, [num_gens, next_n, core.idx_heads, core.idx_dim]),
            k_pool,
            weights,
            kv_lens,
            step["indexer_k_cache_block_offsets"][:num_gens],
            step["scheduler_metadata_buffer"],
            core._max_ctx_len,
        )
        if next_n != 1 or core._ix_topk is None:
            indexer_topk_decode(
                logits, kv_lens, topk_buf, next_n, core.idx_topk, 1, step["radix_aux_indices"],
                step["radix_aux_logits"],
            )  # fmt: skip
        else:
            cute_dsl_radix_topk_decode(logits, kv_lens, topk_buf, next_n=1, compiled=core._ix_topk)

    def _mlp(self, o, L):
        """This rank's share of the layer's MLP, a partial sum: the dense SwiGLU slice on the
        dense layer; on a MoE layer the expert window (32 experts at tep4, 16 at tep8) plus the
        shared expert's slice."""
        core = self.core
        act = flashinfer_silu_and_mul(self._linear_fp8(o, L.mlp_gu, L.mlp_gu_s))
        shared = self._linear_fp8(act, L.mlp_dn, L.mlp_dn_s)
        m = L.moe
        if m is None:
            return shared
        logits = cublas_mm(o, m.router, None, torch.float32)
        ids, weights = renorm_moe_routing_op(logits, core.topk, torch.bfloat16)
        # One NVFP4 quantization under the layer's static scale serves every expert on every
        # rank: the windows' partials sum.
        xq, xs = fp4_quantize(o, m.gscale, _NVFP4_SF_VEC, False, _NVFP4_SF_LINEAR)
        routed = fp4_block_scale_moe_runner(
            None, None, xq, xs.view(torch.float8_e4m3fn), m.fc1_w, m.fc1_s, None, None, None,
            None, m.fc2_w, m.fc2_s, None, m.out1, m.out1_gate, m.out2, core.num_experts,
            core.topk, None, None, core.moe_inter, core.expert_offset, core.local_experts, None,
            _MOE_ROUTING_METHOD, True, _MOE_ACT_SWIGLU, topk_weights=weights, topk_ids=ids,
            tune_max_num_tokens=core.max_num_tokens,
        )[0]  # fmt: skip
        return torch.add(routed, shared)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        core = self.core
        self._check_step_contract(attn_metadata)
        step = self.step_args(attn_metadata)
        h = (
            nn.functional.embedding(input_ids, self._embed)
            if inputs_embeds is None
            else inputs_embeds
        )
        rows = h.shape[0]
        dev = h.device
        assert rows == step["num_tokens"] and rows % step["num_seqs"] == 0, (
            rows,
            step["num_tokens"],
            step["num_seqs"],
        )
        # A separate graph is captured for steps whose sequences all fit in index_topk, so
        # this host branch is a per-capture constant.
        skip_gen = step["skip_gen"]
        pos = torch.reshape(position_ids, [-1])
        req_idx = step["req_idx_per_token"][:rows]
        if skip_gen:
            topk_buf = step["topk_indices_buffer"][:rows]
        else:
            topk_buf = torch.empty([rows, core.idx_topk], dtype=torch.int32, device=dev)
        scratch = self._scratch(rows, h.dtype, dev)

        x = flashinfer_rmsnorm(h, self._norm0, core.eps)
        residual = h
        spec_metadata = kwargs.get("spec_metadata")
        for layer_idx, L in enumerate(self._layers):
            attn_out = self._attention(step, L, x, pos, req_idx, topk_buf, scratch, skip_gen)
            partial = self._linear_fp8(attn_out, L.wo, L.wo_s)
            o = cute_dsl_allreduce_add_rmsnorm(
                partial, residual, L.ffn_norm, torch.empty_like(partial), core._lamport
            )
            mlp = self._mlp(o, L)
            x = cute_dsl_allreduce_add_rmsnorm(
                mlp, residual, L.next_norm, torch.empty_like(mlp), core._lamport
            )
            if spec_metadata is not None:
                spec_metadata.maybe_capture_hidden_states(layer_idx, residual)
        return x

    def _scratch(self, rows: int, dtype, dev) -> SimpleNamespace:
        """Per-step scratch every layer rewrites in turn: the attention output, the absorbed
        query at this rank's heads (every element written by each layer: the absorption BMM
        and the SSMax op's rope tail or zeros), a NoPE owner's 576-wide latent rows (the rope
        key zero for the step), and the latent attention output the native decode writes.
        Under `gen_attn_flash_mla` the query is FlashMLA's 64 heads (this rank's first, the
        rest zero for the step), the scratch carries the FlashMLA scheduler the step's first
        layer fills and the rest share (captured with the step), and the latent output is kept
        only for the unfused glue, which copies FlashMLA's live heads into it."""
        core = self.core
        bf16 = dict(dtype=dtype, device=dev)
        if core.gen_attn_flash_mla:
            fused_q = torch.zeros([rows, _FLASH_MLA_HEADS, core.lat_dim], **bf16)
        else:
            fused_q = torch.empty([rows, core.heads, core.lat_dim], **bf16)
        return SimpleNamespace(
            attn_out=torch.empty([rows, core.heads * core.v_dim], **bf16),
            fused_q=fused_q,
            nope_lat=None if core.unfused_glue else torch.zeros([rows, core.lat_dim], **bf16),
            lat_out=(
                torch.empty([rows, core.heads, core.kv_lora], **bf16)
                if core.unfused_glue or not core.gen_attn_flash_mla
                else None
            ),
            sched=get_mla_metadata()[0] if core.gen_attn_flash_mla else None,
        )

    def _attention(self, step, L, x, pos, req_idx, topk_buf, s, skip_gen: bool):
        """One layer's attention block into `s.attn_out` (returned): the query with SSMax, the
        indexer's selection, the absorbed query, the owner's rope and fp8_ds_mla append,
        sparse decode over the pool, and the V up-projection of the latent output (FlashMLA's
        live heads read in place under `gen_attn_flash_mla` with the fused glue)."""
        core = self.core
        rows = x.shape[0]
        q = self._linear_fp8(x, L.wq, L.wq_s)
        q3 = torch.reshape(q, [rows, core.heads, core.qk_dim])
        fq = s.fused_q[:, : core.heads]
        if core.unfused_glue:
            q3 = ssmax_scaling(q3, pos, L.ssmax_s, L.ssmax_b)
            if L.rope:
                fq[..., core.kv_lora :].copy_(q3[..., L.nope :])
            else:
                fq[..., core.kv_lora :].zero_()
        else:
            # SSMax in place; its rope half (a NoPE layer's zero) into the absorbed query's tail.
            ssmax_scaling_fused_q(
                q3, pos, L.ssmax_s, L.ssmax_b, q3, fq[..., core.kv_lora :], L.rope
            )
        if L.indexer is not None:
            self._indexer(step, x, pos, L, topk_buf, skip_gen)
        slots = convert_req_index_to_global(
            req_idx, step["block_table"], topk_buf, step["tokens_per_block"], core.idx_topk,
            core._stride_factor, core._pool_row[L.kv_source],
        )  # fmt: skip
        bmm_out(
            torch.transpose(q3[..., : L.nope], 0, 1), L.k_b,
            torch.transpose(fq[..., : core.kv_lora], 0, 1),
        )  # fmt: skip
        if L.owner:
            ds_mla_rope_append(
                self._latent(x, L, s.nope_lat), fq, pos, core._rope.indexer, req_idx,
                step["block_table"], step["latent_pool"], core._pool_row[L.index], L.rope,
            )  # fmt: skip
        if core.gen_attn_flash_mla and core.unfused_glue:
            lat = flash_mla_sparse_decode(
                s.fused_q, step["latent_kv"], slots, s.sched, core.softmax_scale, s.lat_out
            )
        elif core.gen_attn_flash_mla:
            lat = flash_mla_sparse_decode(
                s.fused_q, step["latent_kv"], slots, s.sched, core.softmax_scale, None
            )[:, : core.heads]
        else:
            # The rank's heads, unpadded, into the latent output.
            cute_dsl_ds_mla_sparse_decode(
                s.fused_q, step["latent_kv"], slots, core.softmax_scale, s.lat_out,
                compiled=core._mla_decode,
            )  # fmt: skip
            lat = s.lat_out
        bmm_out(
            torch.transpose(lat, 0, 1),
            L.v_b_t,
            torch.transpose(torch.reshape(s.attn_out, [rows, core.heads, core.v_dim]), 0, 1),
        )
        return s.attn_out

    def dspark_forward(
        self,
        draft,
        noise_embedding,
        query_positions,
        num_ctx_per_req,
        ctx_k_cache,
        ctx_v_cache,
        ctx_cache_batch_idx,
        ctx_kv_cache=None,
        ctx_page_table=None,
    ):
        draft._build_fused_kv_buffers()
        draft._ensure_rope()
        core = self.core
        batch, block, hidden = noise_embedding.shape
        rows = batch * block
        # Gather a bounded window, not every page in the configured context.
        # Invalid padding slots are masked even if their pages are allocated.
        lengths = num_ctx_per_req[:batch].long()
        offsets = torch.arange(draft.window, device=noise_embedding.device)
        positions = (lengths - draft.window).clamp_min(0)[:, None] + offsets[None, :]
        valid = positions < lengths[:, None]
        slots = ctx_cache_batch_idx.long()
        if ctx_page_table is not None:
            page_size = ctx_kv_cache[0].shape[-2]
            table = ctx_page_table.index_select(0, slots).long()
            safe_positions = positions.clamp(max=table.shape[1] * page_size - 1)
            pages = table.gather(1, safe_positions // page_size).clamp_min(0)
            page_offsets = safe_positions % page_size
        else:
            safe_positions = positions.clamp(max=ctx_k_cache.shape[2] - 1)
        key_valid = torch.cat(
            (valid, torch.ones(batch, block, dtype=torch.bool, device=valid.device)), dim=1
        )
        query = torch.empty(
            batch,
            block,
            core.heads,
            core.lat_dim,
            dtype=noise_embedding.dtype,
            device=noise_embedding.device,
        )
        values = torch.empty(
            rows, core.heads, core.v_dim, dtype=noise_embedding.dtype, device=noise_embedding.device
        )
        residual = noise_embedding.reshape(rows, hidden).clone()
        normalized = torch.empty_like(residual)
        x = flashinfer_rmsnorm(residual, draft._layers[0].attn_norm, core.eps)
        for i, layer in enumerate(draft._layers):
            q = cublas_mm(x, layer.q, None, None).view(batch, block, core.heads, core.qk_dim)
            bmm_out(
                q[..., : core.nope].reshape(rows, core.heads, core.nope).transpose(0, 1),
                layer.k_b,
                query[..., : core.kv_lora].view(rows, core.heads, core.kv_lora).transpose(0, 1),
            )
            query[..., core.kv_lora :].copy_(draft._rotate(q[..., core.nope :], query_positions))
            kv = cublas_mm(x, layer.kv_a, None, None)
            content = flashinfer_rmsnorm(
                kv[:, : core.kv_lora].contiguous(), layer.kv_norm, core.eps
            )
            block_kv = torch.cat(
                (
                    content.view(batch, block, core.kv_lora),
                    draft._rotate(
                        kv[:, core.kv_lora :].view(batch, block, core.rope_dim), query_positions
                    ),
                ),
                dim=-1,
            )
            if ctx_page_table is not None:
                context = ctx_kv_cache[i][:, 0, 0][pages, page_offsets]
            else:
                context = ctx_k_cache[slots[:, None], i, safe_positions, 0]
            # Zero masked values as well as scores: 0 * an uninitialized
            # NaN in the latent pool must not contaminate the weighted sum.
            context = context.masked_fill(~valid.unsqueeze(-1), 0)
            keys = torch.cat((context, block_kv), dim=1)
            scores = torch.einsum("bshd,btd->bsht", query, keys).float() * core.softmax_scale
            scores = scores.masked_fill(~key_valid[:, None, None, :], float("-inf"))
            probabilities = torch.softmax(scores, dim=-1).to(keys.dtype)
            latent = torch.einsum("bsht,btc->bshc", probabilities, keys[..., : core.kv_lora])
            bmm_out(
                latent.reshape(rows, core.heads, core.kv_lora).transpose(0, 1),
                layer.v_b.transpose(1, 2),
                values.transpose(0, 1),
            )
            partial = cublas_mm(values.reshape(rows, -1), layer.out, None, None)
            o = cute_dsl_allreduce_add_rmsnorm(
                partial, residual, layer.ffn_norm, normalized, core._lamport
            )
            gate = cublas_mm(o, layer.gate, None, None)
            up = cublas_mm(o, layer.up, None, None)
            activated = flashinfer_silu_and_mul(torch.cat((gate, up), dim=-1))
            mlp = cublas_mm(activated, layer.down, None, None)
            x = cute_dsl_allreduce_add_rmsnorm(
                mlp, residual, layer.next_norm, normalized, core._lamport
            )
        return x


class DraftModel(nn.Module):
    """This specialization's BF16 DSpark backbone and recurrent head.

    Runtime owns accepted-token bookkeeping and the paged context pool. The
    draft attends the last trained window of captured target features plus
    its whole non-causal query block. All parameters live in the core table.
    """

    _uses_worker_attention_backend = False
    _paged_ctx_cache = True
    _kv_factor = 1
    _dspark_shift_label = True
    has_markov_head = True
    has_recurrent_markov_head = True

    def __init__(self, shell, draft_config):
        super().__init__()
        # Keep references without registering the parent as a child module.
        object.__setattr__(self, "_shell", shell)
        self.model_config = draft_config
        self.config = draft_config.pretrained_config
        core = shell.model
        spec = shell.spec_config
        native = spec._draft_model_config
        self.block_size = spec.block_size
        self.mask_token_id = spec.mask_token_id
        self.target_layer_ids = tuple(spec.target_layer_ids)
        self.window = int(native["ragged_attention"].split(",")[0])
        self._num_attn_layers = int(native["n_layers"])
        self._num_heads = core.heads
        self._num_kv_heads = 1
        self._head_dim = core.lat_dim
        self.dflash_attention_backend = "VANILLA"
        self._runtime_position_ceiling = None
        self._rope = None
        self._layers = None

    @property
    def core(self):
        return self._shell.model

    @property
    def lm_head(self):
        return self._shell.lm_head

    @property
    def markov_w2(self):
        return self.core.w["draft_markov_output"]

    def logits_processor(self, hidden, lm_head, attn_metadata, gather):
        logits = self._shell.logits_processor(hidden, lm_head, attn_metadata, gather)
        # The shared-head logits are soft-capped before the RNN bias is added.
        return tanh_softcap(logits, self._shell.softcap)

    def _build_fused_kv_buffers(self):
        # The worker calls this before sizing its cache. Take views once after
        # materialization; context projection is explicit per layer here.
        if self._layers is not None:
            return
        w, core = self.core.w, self.core
        self._layers = []
        for i in range(self._num_attn_layers):
            prefix = f"l{i}_draft_"
            kv_b = w[prefix + "wkv_b"].view(core.heads, core.nope + core.v_dim, core.kv_lora)
            self._layers.append(
                SimpleNamespace(
                    attn_norm=w[prefix + "attn_norm"],
                    q=w[prefix + "wq"].t(),
                    kv_a=w[prefix + "wkv_a"].t(),
                    kv_norm=w[prefix + "kv_a_norm"],
                    k_b=kv_b[:, : core.nope].contiguous(),
                    v_b=kv_b[:, core.nope :].contiguous(),
                    out=w[prefix + "wo"].t(),
                    ffn_norm=w[prefix + "ffn_norm"],
                    gate=w[prefix + "w1"].t(),
                    up=w[prefix + "w3"].t(),
                    down=w[prefix + "w2"].t(),
                    next_norm=(
                        w[f"l{i + 1}_draft_attn_norm"]
                        if i + 1 < self._num_attn_layers
                        else w["draft_final_norm"]
                    ),
                )
            )

    def _ensure_rope(self):
        cap = self._runtime_position_ceiling or self.core.rope_rows
        if self._rope is None or self._rope.rows != cap:
            w = self.core.w["draft_final_norm"]
            # alpha=0 removes LeRoPE; all draft layers use plain YaRN.
            self._rope = _rope_tables(
                self.core.model_config.pretrained_config,
                torch.zeros(self.core.rope_dim // 2),
                cap,
                w.dtype,
                w.device,
            )

    def _rotate(self, x, positions):
        table = self._rope.indexer[positions.long()]
        cos, sin = table.chunk(2, dim=-1)
        # One head axis for Q, none for the shared latent K.
        if x.ndim == positions.ndim + 2:
            cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
        even, odd = x.float()[..., 0::2], x.float()[..., 1::2]
        return (
            torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1)
            .flatten(-2)
            .to(x.dtype)
        )

    def embed_draft_block(self, anchors, block_size):
        # The route pins the trained block width; no mask token is looked up
        # in the vocabulary (its sentinel is deliberately out of range).
        weight = self.core.w["embed"]
        valid = (anchors >= 0) & (anchors < weight.shape[0])
        embedded = nn.functional.embedding(anchors.clamp(0, weight.shape[0] - 1), weight)
        embedded = embedded.masked_fill(~valid.unsqueeze(-1), 0)
        masks = self.core.w["draft_mask"].unsqueeze(0).expand(anchors.shape[0], -1, -1)
        return torch.cat((embedded.unsqueeze(1), masks), dim=1)

    def project_target_hidden(self, hidden):
        w = self.core.w
        projected = cublas_mm(hidden, w["draft_target_projection"].t(), None, None)
        return flashinfer_rmsnorm(projected, w["draft_target_norm"], self.core.eps)

    def precompute_context_kv(self, projected_hidden, positions):
        self._build_fused_kv_buffers()
        self._ensure_rope()
        core = self.core
        latent = []
        for layer in self._layers:
            kv = cublas_mm(projected_hidden, layer.kv_a, None, None)
            content = flashinfer_rmsnorm(
                kv[:, : core.kv_lora].contiguous(), layer.kv_norm, core.eps
            )
            rope = self._rotate(kv[:, core.kv_lora :], positions.reshape(-1))
            latent.append(torch.cat((content, rope), dim=-1))
        return torch.stack(latent, dim=1).unsqueeze(2), None

    def dflash_forward_with_metadata(self, *, attn_metadata, **kwargs):
        # The phase belongs to this engine step, including mixed batches.
        return self.core._select_target(attn_metadata).dspark_forward(self, **kwargs)

    def apply_markov_chain_logits(
        self,
        base_logits,
        first_prev_tokens,
        *,
        argmax_fn,
        vocab_slice=None,
        hidden_states,
    ):
        w = self.core.w
        output = w["draft_markov_output"]
        if vocab_slice is not None:
            output = output[vocab_slice]
        previous = first_prev_tokens.long()
        state = None
        refined = []
        for i in range(base_logits.shape[1]):
            weight = w["draft_markov_embedding"]
            valid = (previous >= 0) & (previous < weight.shape[0])
            embedding = nn.functional.embedding(previous.clamp(0, weight.shape[0] - 1), weight)
            embedding = embedding.masked_fill(~valid.unsqueeze(-1), 0)
            if state is None:
                state = torch.zeros_like(embedding)
            joint = cublas_mm(
                torch.cat((state, embedding, hidden_states[:, i]), dim=-1),
                w["draft_joint"].t(),
                w["draft_joint_bias"],
                None,
            )
            gate, candidate, emitted = joint.chunk(3, dim=-1)
            gate = torch.sigmoid(gate)
            state = gate * state + (1 - gate) * torch.tanh(candidate)
            bias = cublas_mm(torch.tanh(emitted), output.t(), None, None)
            logits = base_logits[:, i] + bias
            refined.append(logits)
            previous = argmax_fn(logits)
        return torch.stack(refined, dim=1)


class _VisionBlock(NamedTuple):
    """One vision transformer block's weights, as the views the ops read: the GEMM
    operands column-major (`.t()` of the stored rows)."""

    attn_norm: torch.Tensor
    qkv: torch.Tensor
    wo: torch.Tensor
    ffn_norm: torch.Tensor
    w13: torch.Tensor
    w2: torch.Tensor


def _vision_step(seg_lens: list[int], device, workspace: torch.Tensor) -> dict:
    """The batch state of one vision attention call: one context sequence per image,
    no KV cache. Every value is host-known from the images' patch counts."""
    total = sum(seg_lens)
    longest = max(seg_lens)
    lens_dev = torch.tensor(seg_lens, dtype=torch.int32, device=device)
    lens_host = torch.tensor(seg_lens, dtype=torch.int32)
    return dict(
        workspace_=workspace,
        sequence_length=lens_dev,
        host_past_key_value_lengths=lens_host,
        host_total_kv_lens=torch.tensor([total, 0], dtype=torch.int32),
        context_lengths=lens_dev,
        host_context_lengths=lens_host,
        host_request_types=torch.zeros(len(seg_lens), dtype=torch.int32),
        max_num_requests=len(seg_lens),
        max_context_length=longest,
        max_seq_len=longest,
        attention_window_size=longest,
        num_contexts=len(seg_lens),
        num_ctx_tokens=total,
    )


def _vision_attention(
    x: torch.Tensor,
    B: _VisionBlock,
    positions: torch.Tensor,
    step: dict,
    heads: int,
    head_dim: int,
    theta: float,
) -> torch.Tensor:
    """One vision block's attention over the patch rows of every image of the call:
    `[wq ; wk ; wv]` as one GEMM, q and k rotated by each patch's (row, col) in its
    own image, attention bidirectional within each image, then `wo`."""
    rows = x.shape[0]
    width = heads * head_dim
    qkv = cublas_mm(x, B.qkv)
    q, k = pixtral_rope_2d(
        torch.reshape(qkv[:, :width], [rows, heads, head_dim]),
        torch.reshape(qkv[:, width : 2 * width], [rows, heads, head_dim]),
        positions,
        theta,
    )
    qkv = torch.cat(
        [torch.reshape(q, [rows, width]), torch.reshape(k, [rows, width]), qkv[:, 2 * width :]], -1
    )
    out = torch.empty([rows, width], dtype=x.dtype, device=x.device)
    out = thop_attention(
        q=qkv,
        output=out,
        num_heads=heads,
        num_kv_heads=heads,
        head_size=head_dim,
        **step,
        **_VISION_ATTN,
    )
    return cublas_mm(out, B.wo)


def _merge_tiles(x: torch.Tensor, grid_h: int, grid_w: int, merge: int) -> torch.Tensor:
    """One image's `[grid_h * grid_w, d]` patch rows -> `[grid_h * grid_w / merge^2,
    d * merge^2]`: each merge x merge tile of the grid as one row, tiles row-major (the
    order of the image's [IMG] tokens), each row laid out (channel, tile row, tile
    column) as the merging layer's input columns are."""
    d = x.shape[-1]
    grid = torch.permute(torch.reshape(x, [grid_h, grid_w, d]), [2, 0, 1])
    tiles = nn.functional.unfold(torch.unsqueeze(grid, 0), kernel_size=merge, stride=merge)
    return torch.t(tiles[0])


class VisionTower:
    """The image input path: a request's pixels -> one row per [IMG] token, in the
    language model's hidden width. Run by the shell, in context steps only, never
    captured; every rank runs the whole tower (all heads) over its own requests.

    Per image: the 14x14 patch embedding as an unfold and one GEMM, `ln_pre`, the
    transformer blocks (RMSNorm, attention with 2-D RoPE, RMSNorm, SwiGLU, each with
    its residual), the pre-projector RMSNorm, the 2x2 patch merger and the adapter
    (`w_in`, exact GELU, `w_out`). The images of one call are concatenated along the
    rows; only the attention needs their boundaries."""

    def __init__(self, core) -> None:
        """Take the tower's weight views and geometry, once, off the real weights."""
        d = _weights.MODEL_WEIGHTS.dims(core)
        w = core.w
        self.hidden = d.v_hidden
        self.heads = d.v_heads
        self.head_dim = d.v_head_dim
        self.patch = d.v_patch
        self.merge = d.v_merge
        self.theta = d.v_theta
        self.dtype = d.dtype
        self.out_hidden = d.hidden
        self._validate = step_contract_enabled()
        self._rank = core.model_config.mapping.rank
        self.patch_w = w["vis_patch"].t()
        self.ln_pre = w["vis_ln_pre"]
        self.blocks = [
            _VisionBlock(
                attn_norm=w[f"v{j}_attn_norm"],
                qkv=w[f"v{j}_qkv"].t(),
                wo=w[f"v{j}_wo"].t(),
                ffn_norm=w[f"v{j}_ffn_norm"],
                w13=w[f"v{j}_w13"].t(),
                w2=w[f"v{j}_w2"].t(),
            )
            for j in range(d.v_layers)
        ]
        self.proj_norm = w["vis_proj_norm"]
        self.merge_w = w["vis_merge"].t()
        self.w_in = w["vis_w_in"].t()
        self.w_out = w["vis_w_out"].t()
        # The prompt token the tower's rows replace, on the host: the executor finds a
        # step's image positions with it before anything reaches the device.
        self.image_token_ids = torch.tensor([d.image_token_id], dtype=torch.int32)
        # The attention op sizes its workspace in place per call and never shrinks it.
        self._workspace = torch.empty(0, dtype=torch.int8, device=self.ln_pre.device)

    def _images(self, multimodal_params: list[MultimodalParams]) -> list[tuple]:
        """`(pixels [C, H, W], grid rows, grid cols)` of every image, in order. The
        mistral-native processor pads the batch and preserves each image's true size."""
        images = []
        unit = self.patch * self.merge
        for param in multimodal_params:
            image = (param.multimodal_data or {}).get("image")
            if image is None:
                continue
            pixels = image["pixel_values"]
            sizes = image.get("image_sizes")
            if pixels.ndim != 4 or pixels.shape[1] != 3:
                raise ValueError("Image pixels must have shape [N, 3, H, W]")
            if sizes is None or len(sizes) != pixels.shape[0]:
                raise ValueError("Image sizes must describe every image in the padded batch")
            for b in range(pixels.shape[0]):
                if len(sizes[b]) != 2:
                    raise ValueError("Each image size must be [height, width]")
                height, width = int(sizes[b][0]), int(sizes[b][1])
                if not (0 < height <= pixels.shape[-2] and 0 < width <= pixels.shape[-1]):
                    raise ValueError("Image sizes must fit within their padded pixels")
                if height % unit or width % unit:
                    raise ValueError(
                        f"image {height}x{width} is not a whole number of {unit}-pixel "
                        "merge tiles; the processor resizes to them"
                    )
                images.append(
                    (pixels[b, :, :height, :width], height // self.patch, width // self.patch)
                )
        return images

    def forward(self, multimodal_params: list[MultimodalParams]) -> torch.Tensor:
        """The rows of every image of `multimodal_params`, request by request, image by
        image: `[sum of merged tiles, text hidden]`."""
        images = self._images(multimodal_params)
        dev = self.ln_pre.device
        if not images:
            return torch.empty([0, self.out_hidden], dtype=self.dtype, device=dev)
        seg = [gh * gw for _, gh, gw in images]
        if self._validate:
            logger.info(
                "modeling_v2 image forward target=%s rank=%d grids=%s merged_rows=%d",
                __name__.rsplit(".", 1)[0],
                self._rank,
                [(gh, gw) for _, gh, gw in images],
                sum(seg) // self.merge**2,
            )
        # Each patch as one row, (channel, row, col) within it, patches row-major.
        patches = torch.cat(
            [
                torch.t(
                    nn.functional.unfold(
                        torch.unsqueeze(px.to(device=dev, dtype=self.dtype), 0),
                        kernel_size=self.patch,
                        stride=self.patch,
                    )[0]
                )
                for px, _, _ in images
            ],
            0,
        )
        positions = torch.cat(
            [
                torch.reshape(
                    torch.stack(
                        torch.meshgrid(
                            torch.arange(gh, device=dev),
                            torch.arange(gw, device=dev),
                            indexing="ij",
                        ),
                        -1,
                    ),
                    [-1, 2],
                )
                for _, gh, gw in images
            ],
            0,
        )
        step = _vision_step(seg, dev, self._workspace)

        h = flashinfer_rmsnorm(cublas_mm(patches, self.patch_w), self.ln_pre, _VISION_EPS)
        x = flashinfer_rmsnorm(h, self.blocks[0].attn_norm, _VISION_EPS)
        residual = h
        for j, B in enumerate(self.blocks):
            attn = _vision_attention(x, B, positions, step, self.heads, self.head_dim, self.theta)
            x, residual = flashinfer_fused_add_rmsnorm(attn, residual, B.ffn_norm, _VISION_EPS)
            mlp = cublas_mm(flashinfer_silu_and_mul(cublas_mm(x, B.w13)), B.w2)
            norm_next = self.blocks[j + 1].attn_norm if j + 1 < len(self.blocks) else self.proj_norm
            x, residual = flashinfer_fused_add_rmsnorm(mlp, residual, norm_next, _VISION_EPS)

        # x is the pre-projector-normed tower output.
        tiles = torch.cat(
            [
                _merge_tiles(xi, gh, gw, self.merge)
                for xi, (_, gh, gw) in zip(torch.split(x, seg, 0), images)
            ],
            0,
        )
        merged = cublas_mm(tiles, self.merge_w)
        return cublas_mm(gelu_erf(cublas_mm(merged, self.w_in)), self.w_out)


def _image_rows(
    vision: Optional[VisionTower], multimodal_params: list[MultimodalParams]
) -> Optional[torch.Tensor]:
    """This step's image rows, one per [IMG] position its context chunks cover, in
    the order those positions appear; None when the step covers none.

    The tower runs once per request, on the first step that needs it; its rows stay
    on the request's multimodal data for the later chunks. A chunk -- under chunked
    prefill, or resuming after a reused prefix -- takes rows `[cached, cached + in
    chunk)` of its request's, counted by the request's [IMG] cumsum."""
    if vision is None:
        if any((p.multimodal_data or {}).get("image") is not None for p in multimodal_params):
            raise ValueError(
                "an image reached a Mistral Large 4 target whose checkpoint has no vision encoder"
            )
        return None
    # Filter before encoding and gathering so inactive requests neither run the
    # tower nor shift the embedding offsets of active requests in a mixed batch.
    multimodal_params = [
        param
        for param in multimodal_params
        if param.multimodal_runtime is None or param.multimodal_runtime.num_mm_tokens_in_chunk > 0
    ]
    embeds = find_input_mm_embeds(
        get_multimodal_embeddings(vision.forward, multimodal_params), multimodal_params
    )
    if not embeds:
        return None
    rows = embeds[0] if len(embeds) == 1 else torch.cat(embeds, 0)
    return rows if rows.shape[0] else None


def _splice_image_rows(
    embed: torch.Tensor,
    input_ids: torch.Tensor,
    rows: torch.Tensor,
    mm_token_indices: Optional[torch.Tensor],
) -> torch.Tensor:
    """The step's input embeddings with `rows` written over its [IMG] positions,
    `mm_token_indices` (which the executor computes on the host from the shell's
    `multimodal_token_ids`); every other position keeps its token embedding."""
    if mm_token_indices is None:
        raise ValueError("an image step arrived without the executor's mm_token_indices")
    if mm_token_indices.shape[0] != rows.shape[0]:
        raise ValueError(
            f"{mm_token_indices.shape[0]} [IMG] positions in this step but "
            f"{rows.shape[0]} image rows for them"
        )
    h = nn.functional.embedding(input_ids, embed)
    return h.index_copy_(0, mm_token_indices, rows.to(h.dtype))


class ModelingV2MistralLarge4Nvfp4Sm103Tep4(
    DecoderModelForCausalLM[MistralLarge4Tep4ModelingV2, PretrainedConfig]
):
    def __init__(self, model_config: ModelConfig):
        cfg = model_config.pretrained_config
        super().__init__(
            MistralLarge4Tep4ModelingV2(model_config),
            config=model_config,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )
        self.softcap = float(cfg.softmax_tanh_gating)
        self._vision: Optional[VisionTower] = None
        vis = getattr(cfg, "vision_config", None)
        self._image_token_ids = (
            torch.tensor([vis.image_token_id], dtype=torch.int32) if vis is not None else None
        )

        self.spec_config = model_config.spec_config
        self.draft_model = None
        self.draft_config = None
        self.spec_worker = None
        if self.spec_config is not None:
            # This is allocation geometry for the independently paged BF16
            # draft, not a second registry-resolved model or target cache.
            draft_cfg = PretrainedConfig(
                num_hidden_layers=int(self.spec_config._draft_model_config["n_layers"]),
                hidden_size=cfg.hidden_size,
                num_attention_heads=cfg.num_attention_heads,
                num_key_value_heads=1,
                kv_lora_rank=cfg.kv_lora_rank,
                qk_rope_head_dim=cfg.qk_rope_head_dim,
                qk_nope_head_dim=cfg.qk_nope_head_dim,
                v_head_dim=cfg.v_head_dim,
                intermediate_size=cfg.intermediate_size,
                vocab_size=cfg.vocab_size,
                max_position_embeddings=cfg.max_position_embeddings,
                torch_dtype=torch.bfloat16,
            )
            self.draft_config = replace(
                model_config,
                pretrained_config=draft_cfg,
                quant_config=QuantConfig(),
                quant_config_dict=None,
                sparse_attention_config=None,
                spec_config=None,
                modeling_v2_target=None,
            )
            self.draft_model = DraftModel(self, self.draft_config)
            self.spec_worker = get_spec_worker(
                self.spec_config,
                model_config,
                model_config.mapping,
                use_separate_draft_kv_cache=True,
            )
            self.spec_worker.set_draft_model(self.draft_model)
            self.epilogue.append(self.spec_worker)

    def set_guided_decoder(self, guided_decoder: CapturableGuidedDecoder) -> bool:
        if self.spec_worker is None:
            return False
        return self.spec_worker.set_guided_decoder(guided_decoder)

    @property
    def multimodal_token_ids(self) -> Optional[torch.Tensor]:
        return self._image_token_ids

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
    ) -> torch.Tensor:
        """Soft-cap target logits before verification and return the worker result."""
        worker_input_ids = input_ids
        multimodal_params = kwargs.get("multimodal_params") or []
        if multimodal_params:
            rows = _image_rows(self._vision, multimodal_params)
            if rows is not None:
                inputs_embeds = _splice_image_rows(
                    self.model.w["embed"], input_ids, rows, kwargs.get("mm_token_indices")
                )
                input_ids = None
        if self.spec_worker is None:
            logits = super().forward(
                attn_metadata,
                cast(torch.IntTensor, input_ids),
                position_ids,
                inputs_embeds,
                return_context_logits,
                spec_metadata,
                lora_params,
                **kwargs,
            )
            return tanh_softcap(logits, self.softcap)
        hidden = self.model(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            lora_params=lora_params,
            **kwargs,
        )
        logits = self.logits_processor(
            nn.functional.embedding(spec_metadata.gather_ids, hidden),
            self.lm_head,
            attn_metadata,
            True,
        )
        return self.spec_worker(
            input_ids=worker_input_ids,
            position_ids=position_ids,
            hidden_states=hidden,
            logits=tanh_softcap(logits, self.softcap),
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
        if step_contract_enabled():
            logger.info(
                "modeling_v2 loaded target=%s class=%s rank=%d world=%d vision=%s",
                __name__.rsplit(".", 1)[0],
                type(self).__name__,
                self.model_config.mapping.rank,
                self.model_config.mapping.world_size,
                getattr(self.config, "vision_config", None) is not None,
            )
        if getattr(self.config, "vision_config", None) is not None:
            self._vision = VisionTower(self.model)
        if self.draft_model is not None:
            self.draft_model._build_fused_kv_buffers()
